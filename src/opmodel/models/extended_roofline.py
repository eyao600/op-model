from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from opmodel.api import DType, EngineKind, LocalOp, MemoryAccess, OpKind, OpProfile, TensorRole
from opmodel.estimator import DispatchingOpModel
from opmodel.hardware import HardwareSpec, MemoryLevel
from opmodel.models.roofline import (
    AttentionEstimator,
    CopyEstimator,
    ElementwiseEstimator,
    EmbeddingEstimator,
    NormEstimator,
    ReductionEstimator,
    SoftmaxEstimator,
    _make_profile,
    _matmul_engine,
)
from opmodel.models.simple_energy import estimate_energy
from opmodel.ops import (
    dtype_nbytes,
    footprint_from_tensors,
    get_tensors,
    parse_batched_gemm,
    parse_gemm,
)


class ExtendedRooflineModel(DispatchingOpModel):
    """Roofline model variant with a GEMM-specific utilization timeline."""

    def __init__(self) -> None:
        super().__init__(
            {
                OpKind.GEMM: ExtendedGemmEstimator(batched=False),
                OpKind.BATCHED_GEMM: ExtendedGemmEstimator(batched=True),
                OpKind.ATTENTION_PREFILL: AttentionEstimator(prefill=True),
                OpKind.ATTENTION_DECODE: AttentionEstimator(prefill=False),
                OpKind.SOFTMAX: SoftmaxEstimator(),
                OpKind.LAYERNORM: NormEstimator(layernorm=True),
                OpKind.RMSNORM: NormEstimator(layernorm=False),
                OpKind.ELEMENTWISE: ElementwiseEstimator(),
                OpKind.REDUCTION: ReductionEstimator(),
                OpKind.EMBEDDING: EmbeddingEstimator(),
                OpKind.COPY: CopyEstimator(),
            }
        )


@dataclass(frozen=True)
class GemmProblemSpec:
    batch: int
    m: int
    n: int
    k: int
    input_dtype: DType
    output_dtype: DType
    weight_is_batched: bool
    beta_zero: bool
    epilogue_reads_c: bool
    transpose_a: bool
    transpose_b: bool


@dataclass(frozen=True)
class GemmKernelSpec:
    cta_m: int
    cta_n: int
    cta_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    pipeline_stages: int
    warps_per_cta: int
    threads_per_cta: int
    registers_per_thread: int
    shared_memory_bytes_per_cta: int | None


@dataclass(frozen=True)
class GridAccounting:
    blocks_m: int
    blocks_n: int
    k_stages: int
    cta_count: int
    useful_flops: float
    issued_flops: float
    tile_efficiency: float


@dataclass(frozen=True)
class TrafficAccounting:
    memory_access: MemoryAccess
    a_logical_bytes: int
    b_logical_bytes: int
    c_read_logical_bytes: int
    d_store_logical_bytes: int
    l2_requested_bytes: int
    dram_unique_bytes: int
    smem_read_bytes: int
    smem_write_bytes: int
    sector_size_bytes: int
    line_size_bytes: int

    @property
    def smem_total_bytes(self) -> int:
        return self.smem_read_bytes + self.smem_write_bytes


@dataclass(frozen=True)
class OccupancyResult:
    num_sms: int
    resident_ctas_per_sm: int
    total_resident_ctas: int
    ctas_per_wave: int
    wave_count: int
    tail_efficiency: float
    limiting_factors: tuple[str, ...]


@dataclass(frozen=True)
class TimelineResult:
    kernel_cycles: float
    cta_cycles: float
    prologue_cycles: float
    stage_cycles: float
    epilogue_cycles: float
    compute_stage_cycles: float
    mma_issue_cycles: float
    mma_dependency_penalty_cycles: float
    mma_ilp_efficiency: float
    smem_stage_cycles: float
    global_load_issue_cycles: float
    l2_service_cycles: float
    dram_service_cycles: float
    exposed_l2_cycles: float
    exposed_dram_cycles: float
    compute_active_cycles: float
    smem_active_cycles: float
    l2_active_cycles: float
    dram_active_cycles: float
    compute_issue_utilization: float
    compute_latency_utilization: float
    compute_active_utilization: float
    smem_utilization: float
    l2_utilization: float
    dram_utilization: float
    compute_smem_overlap: float
    compute_l2_overlap: float
    compute_dram_overlap: float


class ExtendedGemmEstimator:
    def __init__(self, *, batched: bool) -> None:
        self._batched = batched

    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        problem = _problem_spec(op, batched=self._batched)
        footprint = footprint_from_tensors(op)
        engine = _matmul_engine(problem.input_dtype, hardware)
        if engine != EngineKind.TENSOR:
            return _make_profile(
                hardware=hardware,
                dtype=problem.input_dtype,
                flops=float(2 * problem.batch * problem.m * problem.n * problem.k),
                hbm_read=footprint.input_bytes + footprint.weight_bytes,
                hbm_write=footprint.output_bytes,
                engine=engine,
                footprint=footprint,
                implementation=_implementation_name(self._batched),
                diagnostics={"tensor_fallback": engine.value},
            )

        warnings: list[str] = []
        kernel = _kernel_spec(op.attrs, problem.input_dtype, hardware, warnings)
        clock_hz = _clock_hz(hardware, warnings)
        grid = _grid_accounting(problem, kernel)
        traffic = _traffic_accounting(problem, kernel, grid, hardware, warnings)
        occupancy = _occupancy(kernel, grid, hardware, warnings)
        timeline = _timeline(problem, kernel, grid, traffic, occupancy, hardware, clock_hz, warnings)
        bottlenecks = _classify_bottlenecks(problem, grid, occupancy, timeline)

        latency_s = hardware.kernel_launch_overhead_s + timeline.kernel_cycles / clock_hz
        flops_per_cycle = grid.useful_flops / timeline.kernel_cycles if timeline.kernel_cycles else 0.0
        tflops_per_s = (grid.useful_flops / latency_s / 1.0e12) if latency_s else 0.0
        energy_breakdown = estimate_energy(
            flops=grid.useful_flops,
            memory_access=traffic.memory_access,
            engine=EngineKind.TENSOR,
            dtype=problem.input_dtype,
            hardware=hardware,
            latency_s=latency_s,
        )

        diagnostics = _diagnostics(
            problem=problem,
            kernel=kernel,
            grid=grid,
            traffic=traffic,
            occupancy=occupancy,
            timeline=timeline,
            bottlenecks=bottlenecks,
            warnings=tuple(warnings),
            clock_hz=clock_hz,
            latency_s=latency_s,
            flops_per_cycle=flops_per_cycle,
            tflops_per_s=tflops_per_s,
            hardware=hardware,
        )
        return OpProfile(
            latency_s=latency_s,
            energy_j=energy_breakdown.total_j,
            flops=grid.useful_flops,
            engine=EngineKind.TENSOR,
            footprint=footprint,
            memory_access=traffic.memory_access,
            energy_breakdown=energy_breakdown,
            implementation=_implementation_name(self._batched),
            diagnostics=diagnostics,
        )


def _problem_spec(op: LocalOp, *, batched: bool) -> GemmProblemSpec:
    if batched:
        batch, m, n, k, dtype = parse_batched_gemm(op)
        weights = get_tensors(op, TensorRole.WEIGHT)
        outputs = get_tensors(op, TensorRole.OUTPUT)
        weight_is_batched = bool(weights and len(weights[0].shape) == 3)
    else:
        m, n, k, dtype = parse_gemm(op)
        batch = 1
        outputs = get_tensors(op, TensorRole.OUTPUT)
        weight_is_batched = True
    output_dtype = outputs[0].dtype if outputs else dtype
    beta_zero = bool(op.attrs.get("beta_zero", True))
    epilogue_reads_c = bool(op.attrs.get("epilogue_reads_c", not beta_zero))
    return GemmProblemSpec(
        batch=batch,
        m=m,
        n=n,
        k=k,
        input_dtype=dtype,
        output_dtype=output_dtype,
        weight_is_batched=weight_is_batched,
        beta_zero=beta_zero,
        epilogue_reads_c=epilogue_reads_c,
        transpose_a=bool(op.attrs.get("transpose_a", False)),
        transpose_b=bool(op.attrs.get("transpose_b", False)),
    )


def _kernel_spec(
    attrs: Mapping[str, Any],
    dtype: DType,
    hardware: HardwareSpec,
    warnings: list[str],
) -> GemmKernelSpec:
    default_mma = hardware.compute.fma_dims or (16, 8, 16)
    if hardware.compute.fma_dims is None:
        warnings.append("default_mma_shape_used")
    mma_m = _positive_int_attr(attrs, "mma_m", default_mma[0])
    mma_n = _positive_int_attr(attrs, "mma_n", default_mma[1])
    mma_k = _positive_int_attr(attrs, "mma_k", default_mma[2])
    cta_m = _positive_int_attr(attrs, "cta_tile_m", 128)
    cta_n = _positive_int_attr(attrs, "cta_tile_n", 128)
    cta_k = _positive_int_attr(attrs, "cta_tile_k", max(mma_k, 32))
    warp_m = _positive_int_attr(attrs, "warp_tile_m", min(cta_m, 64))
    warp_n = _positive_int_attr(attrs, "warp_tile_n", min(cta_n, 64))
    warp_k = _positive_int_attr(attrs, "warp_tile_k", cta_k)
    pipeline_stages = _positive_int_attr(attrs, "pipeline_stages", 3)
    warps_per_cta = _positive_int_attr(attrs, "warps_per_cta", 4)
    threads_per_cta = _positive_int_attr(attrs, "threads_per_cta", warps_per_cta * 32)
    registers_per_thread = _positive_int_attr(attrs, "registers_per_thread", 64)
    shared_memory_bytes_per_cta = _optional_positive_int_attr(attrs, "shared_memory_bytes_per_cta")
    if shared_memory_bytes_per_cta is None:
        stage_operand_bytes = _ceil_scalar_bytes(
            (cta_m * cta_k + cta_k * cta_n) * dtype_nbytes(dtype)
        )
        shared_memory_bytes_per_cta = stage_operand_bytes * pipeline_stages
        warnings.append("shared_memory_bytes_per_cta_estimated")
    return GemmKernelSpec(
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cta_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=warp_k,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        pipeline_stages=pipeline_stages,
        warps_per_cta=warps_per_cta,
        threads_per_cta=threads_per_cta,
        registers_per_thread=registers_per_thread,
        shared_memory_bytes_per_cta=shared_memory_bytes_per_cta,
    )


def _grid_accounting(problem: GemmProblemSpec, kernel: GemmKernelSpec) -> GridAccounting:
    blocks_m = _ceil_div(problem.m, kernel.cta_m)
    blocks_n = _ceil_div(problem.n, kernel.cta_n)
    k_stages = _ceil_div(problem.k, kernel.cta_k)
    cta_count = problem.batch * blocks_m * blocks_n
    useful_flops = float(2 * problem.batch * problem.m * problem.n * problem.k)
    issued_flops = float(
        2
        * problem.batch
        * blocks_m
        * blocks_n
        * k_stages
        * kernel.cta_m
        * kernel.cta_n
        * kernel.cta_k
    )
    return GridAccounting(
        blocks_m=blocks_m,
        blocks_n=blocks_n,
        k_stages=k_stages,
        cta_count=cta_count,
        useful_flops=useful_flops,
        issued_flops=issued_flops,
        tile_efficiency=useful_flops / issued_flops if issued_flops else 0.0,
    )


def _traffic_accounting(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    hardware: HardwareSpec,
    warnings: list[str],
) -> TrafficAccounting:
    dtype_bytes = dtype_nbytes(problem.input_dtype)
    out_dtype_bytes = dtype_nbytes(problem.output_dtype)
    l2_level = hardware.memory_levels.get("l2")
    transaction_level = l2_level or hardware.memory_levels["hbm"]
    sector_size = transaction_level.sector_size_bytes or 32
    line_size = transaction_level.line_size_bytes or max(128, sector_size)
    if transaction_level.sector_size_bytes is None:
        warnings.append(f"{transaction_level.name}_sector_size_default_32B")
    if transaction_level.line_size_bytes is None:
        warnings.append(f"{transaction_level.name}_line_size_default_{line_size}B")

    m_tiles = _tile_lengths(problem.m, kernel.cta_m)
    n_tiles = _tile_lengths(problem.n, kernel.cta_n)
    k_tiles = _tile_lengths(problem.k, kernel.cta_k)

    a_unique_tx = sum(
        _sector_round_bytes(m_len * k_len * dtype_bytes, sector_size)
        for m_len in m_tiles
        for k_len in k_tiles
    )
    b_unique_one_batch_tx = sum(
        _sector_round_bytes(k_len * n_len * dtype_bytes, sector_size)
        for n_len in n_tiles
        for k_len in k_tiles
    )
    a_requested_tx = problem.batch * grid.blocks_n * a_unique_tx
    b_requested_tx = problem.batch * grid.blocks_m * b_unique_one_batch_tx
    a_unique_dram_tx = problem.batch * a_unique_tx
    b_unique_dram_tx = (
        problem.batch if problem.weight_is_batched else 1
    ) * b_unique_one_batch_tx

    d_store_tx = problem.batch * sum(
        _sector_round_bytes(m_len * n_len * out_dtype_bytes, sector_size)
        for m_len in m_tiles
        for n_len in n_tiles
    )
    c_read_tx = d_store_tx if problem.epilogue_reads_c else 0
    l2_read = a_requested_tx + b_requested_tx + c_read_tx if l2_level is not None else None
    l2_write = d_store_tx if l2_level is not None else None
    if l2_level is None:
        hbm_read = a_requested_tx + b_requested_tx + c_read_tx
        hbm_write = d_store_tx
        warnings.append("l2_level_absent_dram_uses_cta_requested_traffic")
    else:
        hbm_read = a_unique_dram_tx + b_unique_dram_tx + c_read_tx
        hbm_write = d_store_tx

    stage_operand_bytes = _ceil_scalar_bytes(
        (kernel.cta_m * kernel.cta_k + kernel.cta_k * kernel.cta_n) * dtype_bytes
    )
    smem_write = grid.cta_count * grid.k_stages * stage_operand_bytes
    smem_read = smem_write
    sram_read = smem_read if "sram" in hardware.memory_levels else None
    sram_write = smem_write if "sram" in hardware.memory_levels else None

    return TrafficAccounting(
        memory_access=MemoryAccess(
            hbm_read_bytes=hbm_read,
            hbm_write_bytes=hbm_write,
            l2_read_bytes=l2_read,
            l2_write_bytes=l2_write,
            sram_read_bytes=sram_read,
            sram_write_bytes=sram_write,
        ),
        a_logical_bytes=_ceil_scalar_bytes(problem.batch * problem.m * problem.k * dtype_bytes),
        b_logical_bytes=_ceil_scalar_bytes(
            (problem.batch if problem.weight_is_batched else 1)
            * problem.k
            * problem.n
            * dtype_bytes
        ),
        c_read_logical_bytes=_ceil_scalar_bytes(
            problem.batch * problem.m * problem.n * out_dtype_bytes
        )
        if problem.epilogue_reads_c
        else 0,
        d_store_logical_bytes=_ceil_scalar_bytes(
            problem.batch * problem.m * problem.n * out_dtype_bytes
        ),
        l2_requested_bytes=(l2_read or 0) + (l2_write or 0),
        dram_unique_bytes=hbm_read + hbm_write,
        smem_read_bytes=smem_read,
        smem_write_bytes=smem_write,
        sector_size_bytes=sector_size,
        line_size_bytes=line_size,
    )


def _occupancy(
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    hardware: HardwareSpec,
    warnings: list[str],
) -> OccupancyResult:
    num_sms = hardware.compute.num_sms or 1
    if hardware.compute.num_sms is None:
        warnings.append("num_sms_default_1")
    max_ctas_limit = hardware.compute.max_ctas_per_sm or 1
    if hardware.compute.max_ctas_per_sm is None:
        warnings.append("max_ctas_per_sm_default_1")
    max_warps_limit = max(1, (hardware.compute.max_warps_per_sm or kernel.warps_per_cta) // kernel.warps_per_cta)
    if hardware.compute.max_warps_per_sm is None:
        warnings.append("max_warps_per_sm_default_kernel_warps")
    if hardware.compute.registers_per_sm is None:
        register_limit = max_ctas_limit
        warnings.append("registers_per_sm_absent")
    else:
        registers_per_cta = kernel.registers_per_thread * kernel.threads_per_cta
        register_limit = max(1, hardware.compute.registers_per_sm // max(1, registers_per_cta))
    if hardware.compute.shared_memory_bytes_per_sm is None:
        smem_limit = max_ctas_limit
        warnings.append("shared_memory_bytes_per_sm_absent")
    else:
        smem_limit = max(
            1,
            hardware.compute.shared_memory_bytes_per_sm
            // max(1, kernel.shared_memory_bytes_per_cta or 1),
        )
    limits = {
        "cta": max_ctas_limit,
        "warp": max_warps_limit,
        "register": register_limit,
        "shared_memory": smem_limit,
    }
    resident = max(1, min(limits.values()))
    min_limit = min(limits.values())
    limiting_factors = tuple(name for name, value in limits.items() if value == min_limit)
    ctas_per_wave = num_sms * resident
    wave_count = max(1, _ceil_div(grid.cta_count, ctas_per_wave))
    final_wave_ctas = grid.cta_count - (wave_count - 1) * ctas_per_wave
    tail_efficiency = final_wave_ctas / ctas_per_wave if ctas_per_wave else 1.0
    return OccupancyResult(
        num_sms=num_sms,
        resident_ctas_per_sm=resident,
        total_resident_ctas=ctas_per_wave,
        ctas_per_wave=ctas_per_wave,
        wave_count=wave_count,
        tail_efficiency=tail_efficiency,
        limiting_factors=limiting_factors,
    )


def _timeline(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> TimelineResult:
    peak_compute_flops_per_cycle = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_hbm_bw_per_cycle = hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz
    l2_level = hardware.memory_levels.get("l2")
    peak_l2_bw_per_cycle = (
        l2_level.bandwidth_bytes_per_s / clock_hz if l2_level is not None else 0.0
    )
    peak_smem_bw_per_cycle = _smem_bandwidth_per_cycle(hardware, clock_hz, warnings)

    total_resident = max(1, occupancy.total_resident_ctas)
    per_cta_compute = peak_compute_flops_per_cycle / total_resident
    per_cta_hbm_bw = peak_hbm_bw_per_cycle / total_resident
    per_cta_l2_bw = peak_l2_bw_per_cycle / total_resident if peak_l2_bw_per_cycle else 0.0
    per_cta_smem_bw = peak_smem_bw_per_cycle / total_resident if peak_smem_bw_per_cycle else 0.0

    stage_flops = float(2 * kernel.cta_m * kernel.cta_n * kernel.cta_k)
    mma_issue_cycles = stage_flops / max(per_cta_compute, 1.0e-12)
    mma_count = (
        _ceil_div(kernel.cta_m, kernel.mma_m)
        * _ceil_div(kernel.cta_n, kernel.mma_n)
        * _ceil_div(kernel.cta_k, kernel.mma_k)
    )
    tensor_latency = float(hardware.compute.tensor_latency_cycles or 8) # source?
    if hardware.compute.tensor_latency_cycles is None:
        warnings.append("tensor_latency_cycles_default_8")
    issue_interval = mma_issue_cycles / max(1, mma_count)
    required_chains = tensor_latency / max(issue_interval, 1.0e-12)
    independent_chains = max(
        1.0,
        float(
            _ceil_div(kernel.warp_m, kernel.mma_m)
            * _ceil_div(kernel.warp_n, kernel.mma_n)
        ),
    )
    mma_ilp_efficiency = min(1.0, independent_chains / max(required_chains, 1.0))
    mma_dependency_penalty = mma_issue_cycles * (1.0 / mma_ilp_efficiency - 1.0)
    compute_stage_cycles = mma_issue_cycles + mma_dependency_penalty

    dtype_bytes = dtype_nbytes(problem.input_dtype)
    stage_operand_tx = _sector_round_bytes(
        kernel.cta_m * kernel.cta_k * dtype_bytes,
        traffic.sector_size_bytes,
    ) + _sector_round_bytes(
        kernel.cta_k * kernel.cta_n * dtype_bytes,
        traffic.sector_size_bytes,
    )
    stage_smem_bytes = _ceil_scalar_bytes(
        2 * (kernel.cta_m * kernel.cta_k + kernel.cta_k * kernel.cta_n) * dtype_bytes
    )
    smem_stage_cycles = stage_smem_bytes / max(per_cta_smem_bw, 1.0e-12)
    global_load_issue_cycles = stage_operand_tx / max(
        per_cta_l2_bw or per_cta_hbm_bw, 1.0e-12
    )
    stage_count = max(1, grid.cta_count * grid.k_stages)
    avg_l2_stage_bytes = traffic.l2_requested_bytes / stage_count if l2_level is not None else 0.0
    avg_dram_stage_bytes = traffic.dram_unique_bytes / stage_count
    l2_service_cycles = (
        _latency_cycles(l2_level, clock_hz) + avg_l2_stage_bytes / max(per_cta_l2_bw, 1.0e-12)
        if l2_level is not None and avg_l2_stage_bytes > 0
        else 0.0
    )
    dram_service_cycles = _latency_cycles(hardware.memory_levels["hbm"], clock_hz) + (
        avg_dram_stage_bytes / max(per_cta_hbm_bw, 1.0e-12)
    )
    hide_window = kernel.pipeline_stages * compute_stage_cycles
    exposed_l2 = max(0.0, l2_service_cycles - hide_window)
    exposed_dram = max(0.0, dram_service_cycles - hide_window)
    exposed_memory = max(exposed_l2, exposed_dram)
    stage_cycles = max(compute_stage_cycles, smem_stage_cycles, global_load_issue_cycles) + exposed_memory

    output_bytes_per_cta = traffic.memory_access.hbm_write_bytes / max(1, grid.cta_count)
    c_read_bytes_per_cta = (
        traffic.c_read_logical_bytes / max(1, grid.cta_count) if problem.epilogue_reads_c else 0
    )
    epilogue_cycles = (
        _sector_round_bytes(output_bytes_per_cta + c_read_bytes_per_cta, traffic.sector_size_bytes)
        / max(per_cta_hbm_bw, 1.0e-12)
    )
    prologue_cycles = min(kernel.pipeline_stages, grid.k_stages) * global_load_issue_cycles
    cta_cycles = prologue_cycles + grid.k_stages * stage_cycles + epilogue_cycles
    kernel_cycles = occupancy.wave_count * cta_cycles

    compute_active = grid.issued_flops / max(peak_compute_flops_per_cycle, 1.0e-12)
    smem_active = traffic.smem_total_bytes / max(peak_smem_bw_per_cycle, 1.0e-12)
    l2_active = (
        traffic.l2_requested_bytes / max(peak_l2_bw_per_cycle, 1.0e-12)
        if l2_level is not None
        else 0.0
    )
    dram_active = traffic.dram_unique_bytes / max(peak_hbm_bw_per_cycle, 1.0e-12)
    compute_issue_util = _clamp01(mma_issue_cycles / max(compute_stage_cycles, 1.0e-12))
    compute_latency_util = _clamp01(mma_ilp_efficiency)
    compute_active_util = _clamp01(compute_active / max(kernel_cycles, 1.0e-12))
    smem_util = _clamp01(smem_active / max(kernel_cycles, 1.0e-12))
    l2_util = _clamp01(l2_active / max(kernel_cycles, 1.0e-12))
    dram_util = _clamp01(dram_active / max(kernel_cycles, 1.0e-12))
    compute_smem_overlap = _overlap_ratio(compute_stage_cycles, smem_stage_cycles)
    compute_l2_overlap = (
        min(1.0, hide_window / l2_service_cycles) if l2_service_cycles > 0.0 else 0.0
    )
    compute_dram_overlap = (
        min(1.0, hide_window / dram_service_cycles) if dram_service_cycles > 0.0 else 0.0
    )
    return TimelineResult(
        kernel_cycles=kernel_cycles,
        cta_cycles=cta_cycles,
        prologue_cycles=prologue_cycles,
        stage_cycles=stage_cycles,
        epilogue_cycles=epilogue_cycles,
        compute_stage_cycles=compute_stage_cycles,
        mma_issue_cycles=mma_issue_cycles,
        mma_dependency_penalty_cycles=mma_dependency_penalty,
        mma_ilp_efficiency=mma_ilp_efficiency,
        smem_stage_cycles=smem_stage_cycles,
        global_load_issue_cycles=global_load_issue_cycles,
        l2_service_cycles=l2_service_cycles,
        dram_service_cycles=dram_service_cycles,
        exposed_l2_cycles=exposed_l2,
        exposed_dram_cycles=exposed_dram,
        compute_active_cycles=compute_active,
        smem_active_cycles=smem_active,
        l2_active_cycles=l2_active,
        dram_active_cycles=dram_active,
        compute_issue_utilization=compute_issue_util,
        compute_latency_utilization=compute_latency_util,
        compute_active_utilization=compute_active_util,
        smem_utilization=smem_util,
        l2_utilization=l2_util,
        dram_utilization=dram_util,
        compute_smem_overlap=compute_smem_overlap,
        compute_l2_overlap=compute_l2_overlap,
        compute_dram_overlap=compute_dram_overlap,
    )


def _classify_bottlenecks(
    problem: GemmProblemSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    timeline: TimelineResult,
) -> tuple[str, tuple[str, ...]]:
    stage_components = {
        "compute_issue_limited": timeline.compute_stage_cycles,
        "smem_bandwidth_limited": timeline.smem_stage_cycles,
        "l2_latency_exposed"
        if timeline.exposed_l2_cycles > timeline.l2_service_cycles * 0.25
        else "l2_bandwidth_limited": timeline.l2_service_cycles,
        "dram_latency_exposed"
        if timeline.exposed_dram_cycles > timeline.dram_service_cycles * 0.25
        else "dram_bandwidth_limited": timeline.dram_service_cycles,
        "epilogue_limited": timeline.epilogue_cycles,
    }
    primary = max(stage_components.items(), key=lambda item: item[1])[0]
    secondary: list[str] = []
    if timeline.compute_latency_utilization < 0.8:
        secondary.append("compute_latency_limited")
    if timeline.mma_ilp_efficiency < 0.8:
        secondary.append("low_mma_ilp")
    if timeline.exposed_l2_cycles > 0.1 * max(timeline.stage_cycles, 1.0e-12):
        secondary.append("l2_latency_exposed")
    if timeline.exposed_dram_cycles > 0.1 * max(timeline.stage_cycles, 1.0e-12):
        secondary.append("dram_latency_exposed")
    if 0.0 < timeline.compute_l2_overlap < 0.7:
        secondary.append("poor_compute_l2_overlap")
    if 0.0 < timeline.compute_dram_overlap < 0.7:
        secondary.append("poor_compute_dram_overlap")
    if occupancy.wave_count <= 1 or occupancy.tail_efficiency < 0.6:
        secondary.append("insufficient_cta_waves")
    if grid.tile_efficiency < 0.9:
        secondary.append("edge_tile_predication_loss")
    if problem.epilogue_reads_c or timeline.epilogue_cycles > 0.2 * max(timeline.cta_cycles, 1.0e-12):
        secondary.append("epilogue_limited")
    return primary, tuple(dict.fromkeys(item for item in secondary if item != primary))


def _diagnostics(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    timeline: TimelineResult,
    bottlenecks: tuple[str, tuple[str, ...]],
    warnings: tuple[str, ...],
    clock_hz: float,
    latency_s: float,
    flops_per_cycle: float,
    tflops_per_s: float,
    hardware: HardwareSpec,
) -> dict[str, Any]:
    primary, secondary = bottlenecks
    q_smem = traffic.smem_total_bytes
    q_l2 = traffic.l2_requested_bytes
    q_dram = traffic.dram_unique_bytes
    useful = grid.useful_flops
    total_q = q_smem + q_l2 + q_dram
    peak_compute = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_smem = _smem_bandwidth_per_cycle(hardware, clock_hz, [])
    peak_l2 = (
        hardware.memory_levels["l2"].bandwidth_bytes_per_s / clock_hz
        if "l2" in hardware.memory_levels
        else 0.0
    )
    peak_dram = hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz
    return {
        "problem": {
            "batch": problem.batch,
            "m": problem.m,
            "n": problem.n,
            "k": problem.k,
            "input_dtype": problem.input_dtype.value,
            "output_dtype": problem.output_dtype.value,
            "beta_zero": problem.beta_zero,
            "epilogue_reads_c": problem.epilogue_reads_c,
            "transpose_a": problem.transpose_a,
            "transpose_b": problem.transpose_b,
        },
        "kernel": {
            "cta_tile": {"m": kernel.cta_m, "n": kernel.cta_n, "k": kernel.cta_k},
            "warp_tile": {"m": kernel.warp_m, "n": kernel.warp_n, "k": kernel.warp_k},
            "mma_shape": {"m": kernel.mma_m, "n": kernel.mma_n, "k": kernel.mma_k},
            "pipeline_stages": kernel.pipeline_stages,
            "warps_per_cta": kernel.warps_per_cta,
            "threads_per_cta": kernel.threads_per_cta,
            "registers_per_thread": kernel.registers_per_thread,
            "shared_memory_bytes_per_cta": kernel.shared_memory_bytes_per_cta,
        },
        "predicted_elapsed_cycles": timeline.kernel_cycles,
        "predicted_flop_per_cycle": flops_per_cycle,
        "predicted_tflops_per_s": tflops_per_s,
        "predicted_latency_s": latency_s,
        "clock_hz": clock_hz,
        "cta_count": grid.cta_count,
        "cta_grid": {
            "m": grid.blocks_m,
            "n": grid.blocks_n,
            "k_stages": grid.k_stages,
        },
        "cta_waves": occupancy.wave_count,
        "resident_ctas_per_sm": occupancy.resident_ctas_per_sm,
        "ctas_per_wave": occupancy.ctas_per_wave,
        "tail_efficiency": occupancy.tail_efficiency,
        "occupancy_limiting_factors": occupancy.limiting_factors,
        "useful_flops": grid.useful_flops,
        "issued_flops": grid.issued_flops,
        "tile_efficiency": grid.tile_efficiency,
        "logical_bytes": {
            "a": traffic.a_logical_bytes,
            "b": traffic.b_logical_bytes,
            "c_read": traffic.c_read_logical_bytes,
            "d_store": traffic.d_store_logical_bytes,
        },
        "transaction_bytes": {
            "l2_requested": traffic.l2_requested_bytes,
            "dram_unique": traffic.dram_unique_bytes,
            "smem_read": traffic.smem_read_bytes,
            "smem_write": traffic.smem_write_bytes,
            "sector_size": traffic.sector_size_bytes,
            "line_size": traffic.line_size_bytes,
        },
        "active_cycles": {
            "compute": timeline.compute_active_cycles,
            "smem": timeline.smem_active_cycles,
            "l2": timeline.l2_active_cycles,
            "dram": timeline.dram_active_cycles,
        },
        "stage_cycles": {
            "compute": timeline.compute_stage_cycles,
            "mma_issue": timeline.mma_issue_cycles,
            "mma_dependency_penalty": timeline.mma_dependency_penalty_cycles,
            "smem": timeline.smem_stage_cycles,
            "global_load_issue": timeline.global_load_issue_cycles,
            "l2_service": timeline.l2_service_cycles,
            "dram_service": timeline.dram_service_cycles,
            "exposed_l2": timeline.exposed_l2_cycles,
            "exposed_dram": timeline.exposed_dram_cycles,
            "stage": timeline.stage_cycles,
            "prologue": timeline.prologue_cycles,
            "epilogue": timeline.epilogue_cycles,
            "cta": timeline.cta_cycles,
        },
        "utilization": {
            "compute_issue": timeline.compute_issue_utilization,
            "compute_latency_adjusted": timeline.compute_latency_utilization,
            "compute_active": timeline.compute_active_utilization,
            "smem": timeline.smem_utilization,
            "l2": timeline.l2_utilization,
            "dram": timeline.dram_utilization,
            "mma_ilp_efficiency": timeline.mma_ilp_efficiency,
        },
        "overlap": {
            "compute_smem": timeline.compute_smem_overlap,
            "compute_l2": timeline.compute_l2_overlap,
            "compute_dram": timeline.compute_dram_overlap,
        },
        "operational_intensity": {
            "smem_flop_per_byte": useful / q_smem if q_smem else None,
            "l2_flop_per_byte": useful / q_l2 if q_l2 else None,
            "dram_flop_per_byte": useful / q_dram if q_dram else None,
            "merged_flop_per_byte": useful / total_q if total_q else None,
        },
        "roofline_bounds_flop_per_cycle": {
            "compute": peak_compute * timeline.compute_latency_utilization,
            "smem": peak_smem * timeline.smem_utilization * (useful / q_smem)
            if q_smem
            else None,
            "l2": peak_l2 * timeline.l2_utilization * (useful / q_l2)
            if q_l2 and peak_l2
            else None,
            "dram": peak_dram * timeline.dram_utilization * (useful / q_dram)
            if q_dram
            else None,
        },
        "primary_bottleneck": primary,
        "secondary_bottlenecks": secondary,
        "warnings": warnings,
        "assumptions": (
            "deterministic_tiled_gemm_access_pattern",
            "first_touch_l2_reuse_when_l2_exists",
            "dram_unique_first_touch_plus_output_writeback",
            "coarse_cta_timeline_not_cycle_accurate",
            "no_fitted_calibration_parameters",
            "shared_memory_bank_conflict_factor_1",
        ),
        "debug_trace": _debug_trace(problem, kernel, grid, traffic, occupancy, timeline, primary, secondary),
    }


def _debug_trace(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    timeline: TimelineResult,
    primary: str,
    secondary: tuple[str, ...],
) -> str:
    return "\n".join(
        (
            f"GEMM shape: B={problem.batch}, M={problem.m}, N={problem.n}, K={problem.k}",
            f"CTA tile: {kernel.cta_m}x{kernel.cta_n}x{kernel.cta_k}",
            f"CTA count: {grid.cta_count}",
            f"K stages: {grid.k_stages}",
            f"resident CTAs per SM: {occupancy.resident_ctas_per_sm}",
            f"CTA waves: {occupancy.wave_count}",
            f"tail efficiency: {occupancy.tail_efficiency:.4g}",
            f"A logical bytes: {traffic.a_logical_bytes}",
            f"B logical bytes: {traffic.b_logical_bytes}",
            f"D store bytes: {traffic.d_store_logical_bytes}",
            f"L2 requested bytes: {traffic.l2_requested_bytes}",
            f"DRAM unique bytes: {traffic.dram_unique_bytes}",
            f"SMEM bytes: {traffic.smem_total_bytes}",
            f"compute stage cycles: {timeline.compute_stage_cycles:.4g}",
            f"SMEM stage cycles: {timeline.smem_stage_cycles:.4g}",
            f"L2 service cycles: {timeline.l2_service_cycles:.4g}",
            f"DRAM service cycles: {timeline.dram_service_cycles:.4g}",
            f"exposed L2 cycles: {timeline.exposed_l2_cycles:.4g}",
            f"exposed DRAM cycles: {timeline.exposed_dram_cycles:.4g}",
            f"compute active utilization: {timeline.compute_active_utilization:.4g}",
            f"SMEM active utilization: {timeline.smem_utilization:.4g}",
            f"L2 active utilization: {timeline.l2_utilization:.4g}",
            f"DRAM active utilization: {timeline.dram_utilization:.4g}",
            f"compute/L2 overlap: {timeline.compute_l2_overlap:.4g}",
            f"compute/DRAM overlap: {timeline.compute_dram_overlap:.4g}",
            f"primary bottleneck: {primary}",
            f"secondary bottlenecks: {', '.join(secondary) if secondary else 'none'}",
        )
    )


def _implementation_name(batched: bool) -> str:
    return "extended_roofline.batched_gemm" if batched else "extended_roofline.gemm"


def _positive_int_attr(attrs: Mapping[str, Any], name: str, default: int) -> int:
    value = attrs.get(name, default)
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_positive_int_attr(attrs: Mapping[str, Any], name: str) -> int | None:
    value = attrs.get(name)
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _clock_hz(hardware: HardwareSpec, warnings: list[str]) -> float:
    if hardware.compute.clock_hz is None:
        warnings.append("clock_hz_default_1GHz")
        return 1.0e9
    return hardware.compute.clock_hz


def _smem_bandwidth_per_cycle(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    sram = hardware.memory_levels.get("sram")
    if sram is not None:
        return sram.bandwidth_bytes_per_s / clock_hz
    warnings.append("sram_level_absent_smem_bandwidth_inferred_from_hbm")
    return hardware.memory_levels["hbm"].bandwidth_bytes_per_s * 8.0 / clock_hz


def _latency_cycles(level: MemoryLevel | None, clock_hz: float) -> float:
    return 0.0 if level is None else level.latency_s * clock_hz


def _tile_lengths(total: int, tile: int) -> tuple[int, ...]:
    full, rem = divmod(total, tile)
    values = [tile] * full
    if rem:
        values.append(rem)
    return tuple(values) or (0,)


def _sector_round_bytes(value: float, sector_size: int) -> int:
    scalar = _ceil_scalar_bytes(value)
    if scalar == 0:
        return 0
    return _ceil_div(scalar, sector_size) * sector_size


def _ceil_scalar_bytes(value: float) -> int:
    return int(math.ceil(value))


def _ceil_div(numerator: int | float, denominator: int | float) -> int:
    return int(math.ceil(numerator / denominator))


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _overlap_ratio(x_cycles: float, y_cycles: float) -> float:
    denominator = max(x_cycles, y_cycles)
    if denominator <= 0.0:
        return 0.0
    return min(x_cycles, y_cycles) / denominator
