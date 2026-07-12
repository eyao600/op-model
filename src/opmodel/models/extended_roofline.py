from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from opmodel.api import DType, EngineKind, LocalOp, MemoryAccess, OpKind, OpProfile, TensorRole
from opmodel.energy import apply_calibrated_energy_model
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
    num_warp_tile_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    pipeline_stages: int
    warps_per_cta: int
    threads_per_cta: int
    registers_per_thread: int
    max_concurrent_ctas_per_sm: int | None
    slice_k: bool


@dataclass(frozen=True)
class GemmKernelTemplate:
    name: str
    cta_m: int
    cta_n: int
    cta_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    num_warp_tile_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    pipeline_stages: int
    warps_per_cta: int
    registers_per_thread: int
    source: str
    dtypes: tuple[DType, ...]


@dataclass(frozen=True)
class GemmCandidateEvaluation:
    template: GemmKernelTemplate
    kernel: GemmKernelSpec
    profile: OpProfile
    selection_energy_j: float
    cheap_rank: int
    cheap_score: float


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
    a_l2_requested_bytes: int
    b_l2_requested_bytes: int
    a_dram_unique_bytes: int
    b_dram_unique_bytes: int
    c_read_transaction_bytes: int
    d_store_transaction_bytes: int
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
    full_wave_ctas: int
    last_wave_ctas: int
    last_wave_busy_sms: int
    last_wave_lazy_sms: int
    last_wave_busy_ctas_per_sm: int
    last_wave_lazy_ctas_per_sm: int
    tail_efficiency: float
    limiting_factors: tuple[str, ...]


@dataclass(frozen=True)
class WavePipelineResult:
    active_ctas: int
    busy_sms: int
    lazy_sms: int
    busy_ctas_per_sm: int
    lazy_ctas_per_sm: int
    start_cycles: float
    work_cycles: float
    end_cycles: float
    total_cycles: float
    sm_stage_cycles: float
    sm_last_stage_cycles: float
    smem_group_cycles: float
    math_group_cycles: float
    math_issue_group_cycles: float
    math_latency_group_cycles: float
    memory_full_stage_cycles: float
    memory_last_stage_cycles: float
    l2_full_stage_cycles: float
    l2_last_stage_cycles: float
    dram_full_stage_cycles: float
    dram_last_stage_cycles: float
    epilogue_global_cycles: float
    epilogue_smem_cycles: float
    slice_k_extra_cycles: float
    epilogue_smem_bytes: float
    epilogue_l2_bytes: float
    epilogue_dram_bytes: float
    l2_epilogue_cycles: float
    dram_epilogue_cycles: float
    exposed_l2_cycles: float
    exposed_dram_cycles: float


@dataclass(frozen=True)
class TimelineResult:
    kernel_cycles: float
    cta_cycles: float
    full_wave: WavePipelineResult
    last_wave: WavePipelineResult
    prologue_cycles: float
    work_cycles: float
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
    epilogue_smem_bytes: float
    epilogue_l2_bytes: float
    epilogue_dram_bytes: float
    groups_k: int
    last_stage_groups_k: int
    last_stage_k: int
    memory_pipeline_groups: int
    last_memory_pipeline_stages: int
    compute_issue_utilization: float
    compute_latency_utilization: float
    compute_active_utilization: float
    smem_utilization: float
    l2_utilization: float
    dram_utilization: float
    compute_smem_overlap: float
    compute_l2_overlap: float
    compute_dram_overlap: float


_GEMM_SELECTION_OBJECTIVES = frozenset(("latency", "energy"))
_GEMM_SELECTION_BACKENDS = frozenset(("extended_roofline",))
_DEFAULT_GEMM_SELECTION_SHORTLIST_SIZE = 12
_EXPLICIT_KERNEL_ATTRS = frozenset(
    (
        "cta_tile_m",
        "cta_tile_n",
        "cta_tile_k",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
        "num_warp_tile_k",
        "num_warp_tile_K",
        "mma_m",
        "mma_n",
        "mma_k",
        "pipeline_stages",
        "warps_per_cta",
        "threads_per_cta",
        "registers_per_thread",
        "resident_ctas_per_sm",
        "max_concurrent_block",
        "slice_k",
        "sliceK",
    )
)

_BF16_FP16_DTYPES = (DType.BF16, DType.FP16)
_GPU_GEMM_KERNEL_CATALOG = (
    GemmKernelTemplate(
        "sm80_256x128x32_64x64x32_8w3s",
        256,
        128,
        32,
        64,
        64,
        32,
        1,
        16,
        8,
        16,
        3,
        8,
        96,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_128x256x32_64x64x32_8w3s",
        128,
        256,
        32,
        64,
        64,
        32,
        1,
        16,
        8,
        16,
        3,
        8,
        96,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_128x128x32_64x64x32_4w3s",
        128,
        128,
        32,
        64,
        64,
        32,
        1,
        16,
        8,
        16,
        3,
        4,
        64,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_128x64x32_64x32x32_4w3s",
        128,
        64,
        32,
        64,
        32,
        32,
        1,
        16,
        8,
        16,
        3,
        4,
        64,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_64x128x32_32x64x32_4w3s",
        64,
        128,
        32,
        32,
        64,
        32,
        1,
        16,
        8,
        16,
        3,
        4,
        64,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_64x64x32_32x32x32_4w3s",
        64,
        64,
        32,
        32,
        32,
        32,
        1,
        16,
        8,
        16,
        3,
        4,
        64,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_256x64x32_64x32x32_8w3s",
        256,
        64,
        32,
        64,
        32,
        32,
        1,
        16,
        8,
        16,
        3,
        8,
        96,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_64x256x32_32x64x32_8w3s",
        64,
        256,
        32,
        32,
        64,
        32,
        1,
        16,
        8,
        16,
        3,
        8,
        96,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_128x32x32_64x32x32_2w3s",
        128,
        32,
        32,
        64,
        32,
        32,
        1,
        16,
        8,
        16,
        3,
        2,
        48,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_32x128x32_32x64x32_2w3s",
        32,
        128,
        32,
        32,
        64,
        32,
        1,
        16,
        8,
        16,
        3,
        2,
        48,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_64x32x32_32x32x32_2w3s",
        64,
        32,
        32,
        32,
        32,
        32,
        1,
        16,
        8,
        16,
        3,
        2,
        48,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_32x64x32_32x32x32_2w3s",
        32,
        64,
        32,
        32,
        32,
        32,
        1,
        16,
        8,
        16,
        3,
        2,
        48,
        "cuda_sm80_catalog",
        _BF16_FP16_DTYPES,
    ),
    GemmKernelTemplate(
        "sm80_tf32_128x128x16_64x64x16_4w3s",
        128,
        128,
        16,
        64,
        64,
        16,
        1,
        16,
        8,
        8,
        3,
        4,
        64,
        "cuda_sm80_catalog",
        (DType.TF32,),
    ),
    GemmKernelTemplate(
        "sm80_tf32_64x128x16_32x64x16_4w3s",
        64,
        128,
        16,
        32,
        64,
        16,
        1,
        16,
        8,
        8,
        3,
        4,
        64,
        "cuda_sm80_catalog",
        (DType.TF32,),
    ),
    GemmKernelTemplate(
        "sm80_tf32_128x64x16_64x32x16_4w3s",
        128,
        64,
        16,
        64,
        32,
        16,
        1,
        16,
        8,
        8,
        3,
        4,
        64,
        "cuda_sm80_catalog",
        (DType.TF32,),
    ),
)


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

        if _should_select_gemm_kernel(op.attrs, hardware):
            return _estimate_selected_gemm_kernel(
                op=op,
                problem=problem,
                footprint=footprint,
                hardware=hardware,
                batched=self._batched,
            )

        warnings: list[str] = []
        kernel = _kernel_spec(op.attrs, hardware, warnings)
        return _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=self._batched,
            warnings=warnings,
        )


def _estimate_gemm_with_kernel(
    *,
    problem: GemmProblemSpec,
    footprint,
    kernel: GemmKernelSpec,
    hardware: HardwareSpec,
    batched: bool,
    warnings: list[str],
) -> OpProfile:
    clock_hz = _clock_hz(hardware, warnings)
    grid = _grid_accounting(problem, kernel)
    traffic = _traffic_accounting(problem, kernel, grid, hardware, warnings)
    occupancy = _occupancy(problem, kernel, grid, hardware, warnings)
    timeline = _timeline(problem, kernel, grid, traffic, occupancy, hardware, clock_hz, warnings)
    fixed_overhead_cycles = float(hardware.compute.device_fixed_overhead_cycles or 0)
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    bottlenecks = _classify_bottlenecks(
        problem,
        grid,
        traffic,
        occupancy,
        timeline,
        hardware,
        clock_hz,
        fixed_overhead_cycles,
    )

    latency_s = total_device_cycles / clock_hz
    flops_per_cycle = (
        grid.useful_flops / total_device_cycles if total_device_cycles else 0.0
    )
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
        fixed_overhead_cycles=fixed_overhead_cycles,
    )
    profile = OpProfile(
        latency_s=latency_s,
        energy_j=energy_breakdown.total_j,
        flops=grid.useful_flops,
        engine=EngineKind.TENSOR,
        footprint=footprint,
        memory_access=traffic.memory_access,
        energy_breakdown=energy_breakdown,
        implementation=_implementation_name(batched),
        diagnostics=diagnostics,
    )
    return apply_calibrated_energy_model(profile, hardware)


def evaluate_gemm_template_candidates(
    op: LocalOp,
    hardware: HardwareSpec,
    *,
    shortlist_size: int | None = None,
) -> tuple[GemmCandidateEvaluation, ...]:
    if op.kind not in (OpKind.GEMM, OpKind.BATCHED_GEMM):
        raise ValueError("GEMM template candidates require a GEMM op")
    problem = _problem_spec(op, batched=op.kind == OpKind.BATCHED_GEMM)
    engine = _matmul_engine(problem.input_dtype, hardware)
    if engine != EngineKind.TENSOR or hardware.kind != "gpu":
        return ()
    backend = _selection_backend(op.attrs)
    size = (
        _positive_int_value(shortlist_size, "shortlist_size")
        if shortlist_size is not None
        else _selection_shortlist_size(op.attrs)
    )
    return _evaluate_gemm_template_candidates(
        problem=problem,
        footprint=footprint_from_tensors(op),
        hardware=hardware,
        batched=op.kind == OpKind.BATCHED_GEMM,
        objective=_selection_objective(op.attrs),
        backend=backend,
        shortlist_size=size,
    )


def select_gemm_template_candidate(
    candidates: tuple[GemmCandidateEvaluation, ...],
    *,
    objective: str = "latency",
) -> GemmCandidateEvaluation:
    objective = _validate_selection_objective(objective)
    if not candidates:
        raise ValueError("No GEMM template candidates to select from")
    return min(candidates, key=lambda candidate: _candidate_objective_key(candidate, objective))


def _estimate_selected_gemm_kernel(
    *,
    op: LocalOp,
    problem: GemmProblemSpec,
    footprint,
    hardware: HardwareSpec,
    batched: bool,
) -> OpProfile:
    objective = _selection_objective(op.attrs)
    backend = _selection_backend(op.attrs)
    shortlist_size = _selection_shortlist_size(op.attrs)
    candidates = _evaluate_gemm_template_candidates(
        problem=problem,
        footprint=footprint,
        hardware=hardware,
        batched=batched,
        objective=objective,
        backend=backend,
        shortlist_size=shortlist_size,
    )
    if not candidates:
        warnings = ["gemm_selection_no_legal_catalog_template"]
        kernel = _kernel_spec(op.attrs, hardware, warnings)
        profile = _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=batched,
            warnings=warnings,
        )
        return _with_gemm_selection_diagnostics(
            profile,
            {
                "enabled": False,
                "reason": "no_legal_catalog_template",
                "objective": objective,
                "backend": backend,
                "catalog_size": len(_kernel_template_catalog(problem.input_dtype, hardware)),
                "legal_candidates": 0,
                "shortlist_size": 0,
            },
        )

    selected = select_gemm_template_candidate(candidates, objective=objective)
    return _with_gemm_selection_diagnostics(
        selected.profile,
        _gemm_selection_metadata(
            selected=selected,
            candidates=candidates,
            objective=objective,
            backend=backend,
            catalog_size=len(_kernel_template_catalog(problem.input_dtype, hardware)),
            legal_count=len(_legal_kernel_templates(problem, hardware)),
        ),
    )


def _evaluate_gemm_template_candidates(
    *,
    problem: GemmProblemSpec,
    footprint,
    hardware: HardwareSpec,
    batched: bool,
    objective: str,
    backend: str,
    shortlist_size: int,
) -> tuple[GemmCandidateEvaluation, ...]:
    if backend != "extended_roofline":
        raise ValueError(f"Unsupported GEMM selection backend: {backend}")
    legal = _legal_kernel_templates(problem, hardware)
    ranked = sorted(
        legal,
        key=lambda template: _cheap_template_score(template, problem, hardware, objective),
    )
    evaluations: list[GemmCandidateEvaluation] = []
    for rank, template in enumerate(ranked[:shortlist_size], start=1):
        kernel = _kernel_from_template(template)
        profile = _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=batched,
            warnings=[],
        )
        evaluations.append(
            GemmCandidateEvaluation(
                template=template,
                kernel=kernel,
                profile=profile,
                selection_energy_j=_selection_energy_j(profile, problem, hardware),
                cheap_rank=rank,
                cheap_score=_cheap_template_score(template, problem, hardware, objective),
            )
        )
    return tuple(evaluations)


def _should_select_gemm_kernel(attrs: Mapping[str, Any], hardware: HardwareSpec) -> bool:
    if hardware.kind != "gpu":
        return False
    return not any(name in attrs for name in _EXPLICIT_KERNEL_ATTRS)


def _selection_objective(attrs: Mapping[str, Any]) -> str:
    return _validate_selection_objective(str(attrs.get("gemm_selection_objective", "latency")))


def _validate_selection_objective(value: str) -> str:
    if value not in _GEMM_SELECTION_OBJECTIVES:
        raise ValueError(
            "gemm_selection_objective must be one of: "
            + ", ".join(sorted(_GEMM_SELECTION_OBJECTIVES))
        )
    return value


def _selection_backend(attrs: Mapping[str, Any]) -> str:
    backend = str(attrs.get("gemm_selection_backend", "extended_roofline"))
    if backend not in _GEMM_SELECTION_BACKENDS:
        raise ValueError(
            "gemm_selection_backend must be one of: "
            + ", ".join(sorted(_GEMM_SELECTION_BACKENDS))
        )
    return backend


def _selection_shortlist_size(attrs: Mapping[str, Any]) -> int:
    return _positive_int_value(
        attrs.get("gemm_selection_shortlist_size", _DEFAULT_GEMM_SELECTION_SHORTLIST_SIZE),
        "gemm_selection_shortlist_size",
    )


def _positive_int_value(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _kernel_template_catalog(
    dtype: DType, hardware: HardwareSpec
) -> tuple[GemmKernelTemplate, ...]:
    if hardware.kind != "gpu":
        return ()
    return tuple(
        template for template in _GPU_GEMM_KERNEL_CATALOG if dtype in template.dtypes
    )


def _legal_kernel_templates(
    problem: GemmProblemSpec, hardware: HardwareSpec
) -> tuple[GemmKernelTemplate, ...]:
    return tuple(
        template
        for template in _kernel_template_catalog(problem.input_dtype, hardware)
        if _template_is_legal(template, problem, hardware)
    )


def _template_is_legal(
    template: GemmKernelTemplate,
    problem: GemmProblemSpec,
    hardware: HardwareSpec,
) -> bool:
    if problem.input_dtype not in hardware.compute.tensor_flops_per_s:
        return False
    if not _divisible(template.cta_m, template.mma_m):
        return False
    if not _divisible(template.cta_n, template.mma_n):
        return False
    if not _divisible(template.cta_k, template.mma_k):
        return False
    if not _divisible(template.warp_m, template.mma_m):
        return False
    if not _divisible(template.warp_n, template.mma_n):
        return False
    if not _divisible(template.warp_k, template.mma_k):
        return False
    if not _divisible(template.cta_m, template.warp_m):
        return False
    if not _divisible(template.cta_n, template.warp_n):
        return False
    if not _divisible(template.cta_k, template.warp_k):
        return False
    spatial_warps = (template.cta_m // template.warp_m) * (
        template.cta_n // template.warp_n
    )
    if spatial_warps != template.warps_per_cta:
        return False
    threads_per_cta = template.warps_per_cta * 32
    if hardware.compute.max_warps_per_sm is not None:
        if template.warps_per_cta > hardware.compute.max_warps_per_sm:
            return False
    if hardware.compute.registers_per_sm is not None:
        registers_per_cta = template.registers_per_thread * threads_per_cta
        if registers_per_cta > hardware.compute.registers_per_sm:
            return False
    if hardware.compute.shared_memory_bytes_per_sm is not None:
        kernel = _kernel_from_template(template)
        grid = _grid_accounting(problem, kernel)
        if _shared_memory_bytes_per_cta(problem, kernel, grid) > hardware.compute.shared_memory_bytes_per_sm:
            return False
    if hardware.compute.max_ctas_per_sm is not None and hardware.compute.max_ctas_per_sm <= 0:
        return False
    return True


def _cheap_template_score(
    template: GemmKernelTemplate,
    problem: GemmProblemSpec,
    hardware: HardwareSpec,
    objective: str,
) -> float:
    kernel = _kernel_from_template(template)
    grid = _grid_accounting(problem, kernel)
    occupancy = _occupancy(problem, kernel, grid, hardware, [])
    problem_aspect = problem.m / max(problem.n, 1)
    tile_aspect = template.cta_m / max(template.cta_n, 1)
    aspect_penalty = abs(math.log2(problem_aspect / tile_aspect))
    wave_penalty = 1.0 - occupancy.tail_efficiency
    underfill_penalty = max(0, occupancy.ctas_per_wave - grid.cta_count) / max(
        occupancy.ctas_per_wave, 1
    )
    k_penalty = 0.2 if problem.k < template.cta_k else 0.0
    tile_efficiency_penalty = 1.0 - grid.tile_efficiency
    objective_penalty = 0.0
    if objective == "energy":
        smem_bytes = _shared_memory_bytes_per_cta(problem, kernel, grid)
        objective_penalty = smem_bytes / max(1, template.pipeline_stages) / 131072.0
    return (
        4.0 * tile_efficiency_penalty
        + 1.2 * underfill_penalty
        + 0.7 * wave_penalty
        + 0.35 * aspect_penalty
        + k_penalty
        + 0.2 * objective_penalty
    )


def _kernel_from_template(template: GemmKernelTemplate) -> GemmKernelSpec:
    return GemmKernelSpec(
        cta_m=template.cta_m,
        cta_n=template.cta_n,
        cta_k=template.cta_k,
        warp_m=template.warp_m,
        warp_n=template.warp_n,
        warp_k=template.warp_k,
        num_warp_tile_k=template.num_warp_tile_k,
        mma_m=template.mma_m,
        mma_n=template.mma_n,
        mma_k=template.mma_k,
        pipeline_stages=template.pipeline_stages,
        warps_per_cta=template.warps_per_cta,
        threads_per_cta=template.warps_per_cta * 32,
        registers_per_thread=template.registers_per_thread,
        max_concurrent_ctas_per_sm=None,
        slice_k=False,
    )


def _selection_energy_j(
    profile: OpProfile, problem: GemmProblemSpec, hardware: HardwareSpec
) -> float:
    issued_flops = float(profile.diagnostics.get("issued_flops", profile.flops))
    extra_flops = max(0.0, issued_flops - profile.flops)
    return profile.energy_j + extra_flops * hardware.compute.tensor_energy_j_per_flop.get(
        problem.input_dtype, 0.0
    )


def _candidate_objective_key(
    candidate: GemmCandidateEvaluation, objective: str
) -> tuple[float, float, float, int]:
    tile_efficiency = float(candidate.profile.diagnostics.get("tile_efficiency", 0.0))
    if objective == "energy":
        return (
            candidate.selection_energy_j,
            candidate.profile.latency_s,
            -tile_efficiency,
            candidate.cheap_rank,
        )
    return (
        candidate.profile.latency_s,
        candidate.selection_energy_j,
        -tile_efficiency,
        candidate.cheap_rank,
    )


def _with_gemm_selection_diagnostics(
    profile: OpProfile, metadata: Mapping[str, Any]
) -> OpProfile:
    diagnostics = dict(profile.diagnostics)
    diagnostics["gemm_selection"] = dict(metadata)
    return replace(profile, diagnostics=diagnostics)


def _gemm_selection_metadata(
    *,
    selected: GemmCandidateEvaluation,
    candidates: tuple[GemmCandidateEvaluation, ...],
    objective: str,
    backend: str,
    catalog_size: int,
    legal_count: int,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "objective": objective,
        "backend": backend,
        "catalog_size": catalog_size,
        "legal_candidates": legal_count,
        "shortlist_size": len(candidates),
        "selected_template": _template_diagnostics(selected.template),
        "selected_rank": selected.cheap_rank,
        "selected_latency_s": selected.profile.latency_s,
        "selected_energy_j": selected.profile.energy_j,
        "selection_energy_j": selected.selection_energy_j,
        "cheap_score": selected.cheap_score,
    }


def _template_diagnostics(template: GemmKernelTemplate) -> dict[str, Any]:
    return {
        "name": template.name,
        "source": template.source,
        "cta_tile": {"m": template.cta_m, "n": template.cta_n, "k": template.cta_k},
        "warp_tile": {
            "m": template.warp_m,
            "n": template.warp_n,
            "k": template.warp_k,
        },
        "num_warp_tile_k": template.num_warp_tile_k,
        "mma_shape": {"m": template.mma_m, "n": template.mma_n, "k": template.mma_k},
        "pipeline_stages": template.pipeline_stages,
        "warps_per_cta": template.warps_per_cta,
        "threads_per_cta": template.warps_per_cta * 32,
        "registers_per_thread": template.registers_per_thread,
        "dtypes": tuple(dtype.value for dtype in template.dtypes),
    }


def _divisible(numerator: int, denominator: int) -> bool:
    return denominator > 0 and numerator % denominator == 0


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
    num_warp_tile_k = _positive_int_attr(
        attrs,
        "num_warp_tile_k",
        _positive_int_attr(attrs, "num_warp_tile_K", 1)
        if "num_warp_tile_K" in attrs
        else 1,
    )
    pipeline_stages = _positive_int_attr(attrs, "pipeline_stages", 3)
    warps_per_cta = _positive_int_attr(attrs, "warps_per_cta", 4)
    threads_per_cta = _positive_int_attr(attrs, "threads_per_cta", warps_per_cta * 32)
    registers_per_thread = _positive_int_attr(attrs, "registers_per_thread", 64)
    max_concurrent_ctas_per_sm = _optional_positive_int_attr(
        attrs, "resident_ctas_per_sm"
    )
    if max_concurrent_ctas_per_sm is None:
        max_concurrent_ctas_per_sm = _optional_positive_int_attr(
            attrs, "max_concurrent_block"
        )
    return GemmKernelSpec(
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cta_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=warp_k,
        num_warp_tile_k=num_warp_tile_k,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        pipeline_stages=pipeline_stages,
        warps_per_cta=warps_per_cta,
        threads_per_cta=threads_per_cta,
        registers_per_thread=registers_per_thread,
        max_concurrent_ctas_per_sm=max_concurrent_ctas_per_sm,
        slice_k=bool(attrs.get("slice_k", attrs.get("sliceK", False))),
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


def _smem_load_per_cta_elements(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting | None = None,
) -> int:
    """EnergAIzer Fig. 4(a) S->R load work per CTA over the full K-loop."""
    k_stages = grid.k_stages if grid is not None else _ceil_div(problem.k, kernel.cta_k)
    warp_m_tiles = _ceil_div(kernel.cta_m, kernel.warp_m)
    warp_n_tiles = _ceil_div(kernel.cta_n, kernel.warp_n)
    warp_k_tiles = _ceil_div(kernel.cta_k, kernel.warp_k)
    return (
        (kernel.warp_m + kernel.warp_n)
        * warp_m_tiles
        * warp_n_tiles
        * k_stages
        * warp_k_tiles
        * kernel.warp_k
    )


def _shared_memory_bytes_per_cta(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting | None = None,
) -> int:
    return _ceil_scalar_bytes(
        _smem_load_per_cta_elements(problem, kernel, grid)
        * dtype_nbytes(problem.input_dtype)
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
    smem_read = grid.cta_count * _shared_memory_bytes_per_cta(problem, kernel, grid)
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
        a_l2_requested_bytes=a_requested_tx,
        b_l2_requested_bytes=b_requested_tx,
        a_dram_unique_bytes=a_unique_dram_tx,
        b_dram_unique_bytes=b_unique_dram_tx,
        c_read_transaction_bytes=c_read_tx,
        d_store_transaction_bytes=d_store_tx,
        l2_requested_bytes=(l2_read or 0) + (l2_write or 0),
        dram_unique_bytes=hbm_read + hbm_write,
        smem_read_bytes=smem_read,
        smem_write_bytes=smem_write,
        sector_size_bytes=sector_size,
        line_size_bytes=line_size,
    )


def _occupancy(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    hardware: HardwareSpec,
    warnings: list[str],
) -> OccupancyResult:
    num_sms = hardware.compute.num_sms or 1
    if hardware.compute.num_sms is None:
        warnings.append("num_sms_default_1")

    if kernel.max_concurrent_ctas_per_sm is not None:
        resident = kernel.max_concurrent_ctas_per_sm
        limiting_factors = ("kernel_observed_max_concurrent_block",)
    else:
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
            shared_memory_bytes_per_cta = _shared_memory_bytes_per_cta(problem, kernel, grid)
            smem_limit = max(
                1,
                hardware.compute.shared_memory_bytes_per_sm
                // max(1, shared_memory_bytes_per_cta),
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
    lazy_ctas_per_sm, busy_remainder = divmod(final_wave_ctas, num_sms)
    busy_ctas_per_sm = lazy_ctas_per_sm + (1 if busy_remainder else 0)
    busy_sms = busy_remainder if busy_remainder else (num_sms if final_wave_ctas else 0)
    lazy_sms = num_sms - busy_sms
    return OccupancyResult(
        num_sms=num_sms,
        resident_ctas_per_sm=resident,
        total_resident_ctas=ctas_per_wave,
        ctas_per_wave=ctas_per_wave,
        wave_count=wave_count,
        full_wave_ctas=ctas_per_wave,
        last_wave_ctas=final_wave_ctas,
        last_wave_busy_sms=busy_sms,
        last_wave_lazy_sms=lazy_sms,
        last_wave_busy_ctas_per_sm=busy_ctas_per_sm,
        last_wave_lazy_ctas_per_sm=lazy_ctas_per_sm,
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

    shared_latency = _shared_latency_cycles(hardware, clock_hz, warnings)
    l2_latency = (
        _memory_latency_cycles(l2_level, clock_hz, "l2", 261.5, warnings)
        if l2_level is not None
        else 0.0
    )
    dram_latency = _memory_latency_cycles(
        hardware.memory_levels["hbm"], clock_hz, "hbm", 466.3, warnings
    )
    tensor_latency = _tensor_latency_cycles(kernel, hardware, warnings)

    groups_k = _k_groups(kernel.cta_k, kernel)
    last_stage_k = (problem.k - 1) % kernel.cta_k + 1 if problem.k else 0
    last_stage_groups_k = _k_groups(last_stage_k, kernel)
    memory_pipeline_groups = _ceil_div(grid.k_stages, kernel.pipeline_stages)
    last_memory_pipeline_stages = (grid.k_stages - 1) % kernel.pipeline_stages + 1

    load_stage_count = max(1, grid.cta_count * grid.k_stages)
    avg_l2_load_bytes_per_cta_stage = (
        (traffic.a_l2_requested_bytes + traffic.b_l2_requested_bytes) / load_stage_count
        if l2_level is not None
        else 0.0
    )
    avg_dram_load_bytes_per_cta_stage = (
        (traffic.a_dram_unique_bytes + traffic.b_dram_unique_bytes) / load_stage_count
    )

    full_wave = _wave_pipeline(
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        active_ctas=occupancy.full_wave_ctas,
        busy_sms=occupancy.num_sms,
        lazy_sms=0,
        busy_ctas_per_sm=occupancy.resident_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.resident_ctas_per_sm,
        peak_compute_flops_per_cycle=peak_compute_flops_per_cycle,
        peak_smem_bw_per_cycle=peak_smem_bw_per_cycle,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        shared_latency_cycles=shared_latency,
        tensor_latency_cycles=tensor_latency,
        l2_latency_cycles=l2_latency,
        dram_latency_cycles=dram_latency,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )
    last_wave = _wave_pipeline(
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        active_ctas=occupancy.last_wave_ctas,
        busy_sms=occupancy.last_wave_busy_sms,
        lazy_sms=occupancy.last_wave_lazy_sms,
        busy_ctas_per_sm=occupancy.last_wave_busy_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.last_wave_lazy_ctas_per_sm,
        peak_compute_flops_per_cycle=peak_compute_flops_per_cycle,
        peak_smem_bw_per_cycle=peak_smem_bw_per_cycle,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        shared_latency_cycles=shared_latency,
        tensor_latency_cycles=tensor_latency,
        l2_latency_cycles=l2_latency,
        dram_latency_cycles=dram_latency,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )

    full_wave_count = max(0, occupancy.wave_count - 1)
    kernel_cycles = full_wave.total_cycles * full_wave_count + last_wave.total_cycles
    prologue_cycles = full_wave.start_cycles * full_wave_count + last_wave.start_cycles
    work_cycles = full_wave.work_cycles * full_wave_count + last_wave.work_cycles
    epilogue_cycles = full_wave.end_cycles * full_wave_count + last_wave.end_cycles
    cta_cycles = full_wave.total_cycles / max(1, occupancy.resident_ctas_per_sm)
    stage_cycles = full_wave.work_cycles / max(1, grid.k_stages)

    compute_active = grid.issued_flops / max(peak_compute_flops_per_cycle, 1.0e-12)
    epilogue_smem_bytes = (
        full_wave.epilogue_smem_bytes * full_wave_count + last_wave.epilogue_smem_bytes
    )
    epilogue_l2_bytes = (
        full_wave.epilogue_l2_bytes * full_wave_count + last_wave.epilogue_l2_bytes
    )
    epilogue_dram_bytes = (
        full_wave.epilogue_dram_bytes * full_wave_count + last_wave.epilogue_dram_bytes
    )
    smem_active = (traffic.smem_total_bytes + epilogue_smem_bytes) / max(
        peak_smem_bw_per_cycle, 1.0e-12
    )
    l2_active = (
        traffic.l2_requested_bytes / max(peak_l2_bw_per_cycle, 1.0e-12)
        if l2_level is not None
        else 0.0
    )
    dram_active = traffic.dram_unique_bytes / max(peak_hbm_bw_per_cycle, 1.0e-12)
    compute_issue_util = _clamp01(
        full_wave.math_issue_group_cycles / max(full_wave.math_group_cycles, 1.0e-12)
    )
    mma_ilp_efficiency = _clamp01(
        full_wave.math_issue_group_cycles / max(full_wave.math_latency_group_cycles, 1.0e-12)
    )
    compute_latency_util = mma_ilp_efficiency
    compute_active_util = _clamp01(compute_active / max(kernel_cycles, 1.0e-12))
    smem_util = _clamp01(smem_active / max(kernel_cycles, 1.0e-12))
    l2_util = _clamp01(l2_active / max(kernel_cycles, 1.0e-12))
    dram_util = _clamp01(dram_active / max(kernel_cycles, 1.0e-12))
    compute_smem_overlap = _overlap_ratio(
        full_wave.math_group_cycles, full_wave.smem_group_cycles
    )
    compute_l2_overlap = (
        min(1.0, full_wave.work_cycles / full_wave.l2_full_stage_cycles)
        if full_wave.l2_full_stage_cycles > 0.0
        else 0.0
    )
    compute_dram_overlap = (
        min(1.0, full_wave.work_cycles / full_wave.dram_full_stage_cycles)
        if full_wave.dram_full_stage_cycles > 0.0
        else 0.0
    )
    return TimelineResult(
        kernel_cycles=kernel_cycles,
        cta_cycles=cta_cycles,
        full_wave=full_wave,
        last_wave=last_wave,
        prologue_cycles=prologue_cycles,
        work_cycles=work_cycles,
        stage_cycles=stage_cycles,
        epilogue_cycles=epilogue_cycles,
        compute_stage_cycles=full_wave.sm_stage_cycles,
        mma_issue_cycles=full_wave.math_issue_group_cycles,
        mma_dependency_penalty_cycles=max(
            0.0, full_wave.math_group_cycles - full_wave.math_issue_group_cycles
        ),
        mma_ilp_efficiency=mma_ilp_efficiency,
        smem_stage_cycles=full_wave.smem_group_cycles,
        global_load_issue_cycles=full_wave.memory_full_stage_cycles,
        l2_service_cycles=full_wave.l2_full_stage_cycles,
        dram_service_cycles=full_wave.dram_full_stage_cycles,
        exposed_l2_cycles=full_wave.exposed_l2_cycles,
        exposed_dram_cycles=full_wave.exposed_dram_cycles,
        compute_active_cycles=compute_active,
        smem_active_cycles=smem_active,
        l2_active_cycles=l2_active,
        dram_active_cycles=dram_active,
        epilogue_smem_bytes=epilogue_smem_bytes,
        epilogue_l2_bytes=epilogue_l2_bytes,
        epilogue_dram_bytes=epilogue_dram_bytes,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
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


def _wave_pipeline(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    active_ctas: int,
    busy_sms: int,
    lazy_sms: int,
    busy_ctas_per_sm: int,
    lazy_ctas_per_sm: int,
    peak_compute_flops_per_cycle: float,
    peak_smem_bw_per_cycle: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
    shared_latency_cycles: float,
    tensor_latency_cycles: float,
    l2_latency_cycles: float,
    dram_latency_cycles: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    groups_k: int,
    last_stage_groups_k: int,
    last_stage_k: int,
    memory_pipeline_groups: int,
    last_memory_pipeline_stages: int,
) -> WavePipelineResult:
    if active_ctas <= 0:
        return WavePipelineResult(
            active_ctas=0,
            busy_sms=0,
            lazy_sms=busy_sms + lazy_sms,
            busy_ctas_per_sm=0,
            lazy_ctas_per_sm=0,
            start_cycles=0.0,
            work_cycles=0.0,
            end_cycles=0.0,
            total_cycles=0.0,
            sm_stage_cycles=0.0,
            sm_last_stage_cycles=0.0,
            smem_group_cycles=0.0,
            math_group_cycles=0.0,
            math_issue_group_cycles=0.0,
            math_latency_group_cycles=0.0,
            memory_full_stage_cycles=0.0,
            memory_last_stage_cycles=0.0,
            l2_full_stage_cycles=0.0,
            l2_last_stage_cycles=0.0,
            dram_full_stage_cycles=0.0,
            dram_last_stage_cycles=0.0,
            epilogue_global_cycles=0.0,
            epilogue_smem_cycles=0.0,
            slice_k_extra_cycles=0.0,
            epilogue_smem_bytes=0.0,
            epilogue_l2_bytes=0.0,
            epilogue_dram_bytes=0.0,
            l2_epilogue_cycles=0.0,
            dram_epilogue_cycles=0.0,
            exposed_l2_cycles=0.0,
            exposed_dram_cycles=0.0,
        )

    total_sms = max(1, busy_sms + lazy_sms)
    per_sm_smem_bw = peak_smem_bw_per_cycle / total_sms
    per_smsp_compute = peak_compute_flops_per_cycle / total_sms / 4.0
    active_warps = max(0, busy_ctas_per_sm) * kernel.warps_per_cta

    effective_warp_m = min(kernel.warp_m, kernel.cta_m)
    effective_warp_n = min(kernel.warp_n, kernel.cta_n)
    effective_warp_k = _effective_warp_k(kernel)
    dtype_bytes = dtype_nbytes(problem.input_dtype)
    warptile_smem_to_reg_bytes = (
        (effective_warp_m + effective_warp_n) * effective_warp_k * dtype_bytes
    )
    smem_group_cycles = (
        max(active_warps * warptile_smem_to_reg_bytes / max(per_sm_smem_bw, 1.0e-12),
            shared_latency_cycles)
        if active_warps > 0
        else 0.0
    )

    smsp_warps = _ceil_div(active_warps, 4) if active_warps else 0
    concurrent_mma = (
        smsp_warps
        * _ceil_div(effective_warp_m, kernel.mma_m)
        * _ceil_div(effective_warp_n, kernel.mma_n)
    )
    mma_k_iters = max(1, _ceil_div(effective_warp_k, kernel.mma_k))
    math_issue_group_cycles = (
        mma_k_iters
        * concurrent_mma
        * (2 * kernel.mma_m * kernel.mma_n * kernel.mma_k)
        / max(per_smsp_compute, 1.0e-12)
        if concurrent_mma > 0
        else 0.0
    )
    if concurrent_mma > 0 and (kernel.mma_m, kernel.mma_n, kernel.mma_k) == (1, 1, 1):
        math_latency_group_cycles = math_issue_group_cycles
    else:
        math_latency_group_cycles = mma_k_iters * tensor_latency_cycles if concurrent_mma > 0 else 0.0
    math_group_cycles = max(math_issue_group_cycles, math_latency_group_cycles)
    per_group_cycles = max(smem_group_cycles, math_group_cycles)
    sm_stage_cycles = smem_group_cycles + math_group_cycles + (
        max(0, groups_k - 1) * per_group_cycles
    )
    sm_last_stage_cycles = smem_group_cycles + math_group_cycles + (
        max(0, last_stage_groups_k - 1) * per_group_cycles
    )

    full_memory_stages = min(kernel.pipeline_stages, grid.k_stages)
    last_stage_fraction = last_stage_k / max(1, kernel.cta_k)
    last_memory_stage_equiv = max(
        last_stage_fraction,
        float(max(0, last_memory_pipeline_stages - 1)) + last_stage_fraction,
    )
    (
        memory_full_stage_cycles,
        l2_full_stage_cycles,
        dram_full_stage_cycles,
    ) = _memory_pipeline_stage_cycles(
        active_ctas=active_ctas,
        stage_equivalent=float(full_memory_stages),
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        l2_latency_cycles=l2_latency_cycles,
        dram_latency_cycles=dram_latency_cycles,
    )
    (
        memory_full_steady_cycles,
        l2_full_steady_cycles,
        dram_full_steady_cycles,
    ) = _memory_pipeline_stage_bandwidth_cycles(
        active_ctas=active_ctas,
        stage_equivalent=float(full_memory_stages),
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
    )
    (
        memory_last_stage_cycles,
        l2_last_stage_cycles,
        dram_last_stage_cycles,
    ) = _memory_pipeline_stage_cycles(
        active_ctas=active_ctas,
        stage_equivalent=last_memory_stage_equiv,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        l2_latency_cycles=l2_latency_cycles,
        dram_latency_cycles=dram_latency_cycles,
    )

    sm_path_cycles = (
        sm_stage_cycles * max(0, grid.k_stages - 1) + sm_last_stage_cycles
        if grid.k_stages > 1
        else sm_last_stage_cycles
    )
    memory_path_cycles = sm_stage_cycles + sm_last_stage_cycles + (
        memory_full_steady_cycles * max(0, memory_pipeline_groups - 1)
        + memory_last_stage_cycles
    )
    work_cycles = max(sm_path_cycles, memory_path_cycles) if grid.k_stages > 1 else sm_path_cycles
    start_cycles = (
        memory_full_stage_cycles
        if grid.k_stages >= kernel.pipeline_stages
        else memory_last_stage_cycles
    )

    wave_fraction = active_ctas / max(1, grid.cta_count)
    l2_epilogue_bytes = (
        wave_fraction
        * (traffic.d_store_transaction_bytes + traffic.c_read_transaction_bytes)
        if peak_l2_bw_per_cycle > 0.0
        else 0.0
    )
    dram_epilogue_bytes = wave_fraction * (
        traffic.d_store_transaction_bytes + traffic.c_read_transaction_bytes
    )
    output_dtype_bytes = dtype_nbytes(problem.output_dtype)
    warptile_reg_to_smem_bytes = effective_warp_m * effective_warp_n * output_dtype_bytes
    epilogue_smem_cycles = (
        max(active_warps * warptile_reg_to_smem_bytes / max(per_sm_smem_bw, 1.0e-12),
            shared_latency_cycles)
        if active_warps > 0
        else 0.0
    )
    slice_k_ld_cycles = epilogue_smem_cycles if kernel.slice_k else 0.0
    slice_k_st_cycles = (
        max(
            active_warps
            * warptile_reg_to_smem_bytes
            / max(per_sm_smem_bw * kernel.num_warp_tile_k, 1.0e-12),
            shared_latency_cycles,
        )
        if kernel.slice_k and active_warps > 0
        else 0.0
    )
    slice_k_extra_cycles = slice_k_ld_cycles + slice_k_st_cycles
    epilogue_smem_bytes = wave_fraction * traffic.d_store_transaction_bytes
    if kernel.slice_k:
        epilogue_smem_bytes += wave_fraction * traffic.d_store_transaction_bytes * (
            1.0 + 1.0 / max(1, kernel.num_warp_tile_k)
        )
    l2_epilogue_cycles = _service_cycles(
        l2_epilogue_bytes, peak_l2_bw_per_cycle, l2_latency_cycles
    )
    dram_epilogue_cycles = _service_cycles(
        dram_epilogue_bytes, peak_hbm_bw_per_cycle, dram_latency_cycles
    )
    epilogue_global_cycles = max(l2_epilogue_cycles, dram_epilogue_cycles)
    end_cycles = epilogue_smem_cycles + slice_k_extra_cycles + epilogue_global_cycles

    l2_pipeline_cycles = (
        l2_full_steady_cycles * max(0, memory_pipeline_groups - 1) + l2_last_stage_cycles
    )
    dram_pipeline_cycles = (
        dram_full_steady_cycles * max(0, memory_pipeline_groups - 1) + dram_last_stage_cycles
    )
    exposed_l2_cycles = max(0.0, l2_pipeline_cycles - sm_path_cycles)
    exposed_dram_cycles = max(0.0, dram_pipeline_cycles - sm_path_cycles)

    return WavePipelineResult(
        active_ctas=active_ctas,
        busy_sms=busy_sms,
        lazy_sms=lazy_sms,
        busy_ctas_per_sm=busy_ctas_per_sm,
        lazy_ctas_per_sm=lazy_ctas_per_sm,
        start_cycles=start_cycles,
        work_cycles=work_cycles,
        end_cycles=end_cycles,
        total_cycles=start_cycles + work_cycles + end_cycles,
        sm_stage_cycles=sm_stage_cycles,
        sm_last_stage_cycles=sm_last_stage_cycles,
        smem_group_cycles=smem_group_cycles,
        math_group_cycles=math_group_cycles,
        math_issue_group_cycles=math_issue_group_cycles,
        math_latency_group_cycles=math_latency_group_cycles,
        memory_full_stage_cycles=memory_full_stage_cycles,
        memory_last_stage_cycles=memory_last_stage_cycles,
        l2_full_stage_cycles=l2_full_stage_cycles,
        l2_last_stage_cycles=l2_last_stage_cycles,
        dram_full_stage_cycles=dram_full_stage_cycles,
        dram_last_stage_cycles=dram_last_stage_cycles,
        epilogue_global_cycles=epilogue_global_cycles,
        epilogue_smem_cycles=epilogue_smem_cycles,
        slice_k_extra_cycles=slice_k_extra_cycles,
        epilogue_smem_bytes=epilogue_smem_bytes,
        epilogue_l2_bytes=l2_epilogue_bytes,
        epilogue_dram_bytes=dram_epilogue_bytes,
        l2_epilogue_cycles=l2_epilogue_cycles,
        dram_epilogue_cycles=dram_epilogue_cycles,
        exposed_l2_cycles=exposed_l2_cycles,
        exposed_dram_cycles=exposed_dram_cycles,
    )


def _memory_pipeline_stage_cycles(
    *,
    active_ctas: int,
    stage_equivalent: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
    l2_latency_cycles: float,
    dram_latency_cycles: float,
) -> tuple[float, float, float]:
    l2_bytes = active_ctas * stage_equivalent * avg_l2_load_bytes_per_cta_stage
    dram_bytes = active_ctas * stage_equivalent * avg_dram_load_bytes_per_cta_stage
    l2_cycles = _service_cycles(l2_bytes, peak_l2_bw_per_cycle, l2_latency_cycles)
    dram_cycles = _service_cycles(dram_bytes, peak_hbm_bw_per_cycle, dram_latency_cycles)
    return max(l2_cycles, dram_cycles), l2_cycles, dram_cycles


def _memory_pipeline_stage_bandwidth_cycles(
    *,
    active_ctas: int,
    stage_equivalent: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
) -> tuple[float, float, float]:
    l2_bytes = active_ctas * stage_equivalent * avg_l2_load_bytes_per_cta_stage
    dram_bytes = active_ctas * stage_equivalent * avg_dram_load_bytes_per_cta_stage
    l2_cycles = (
        l2_bytes / peak_l2_bw_per_cycle
        if l2_bytes > 0.0 and peak_l2_bw_per_cycle > 0.0
        else 0.0
    )
    dram_cycles = (
        dram_bytes / peak_hbm_bw_per_cycle
        if dram_bytes > 0.0 and peak_hbm_bw_per_cycle > 0.0
        else 0.0
    )
    return max(l2_cycles, dram_cycles), l2_cycles, dram_cycles


def _service_cycles(bytes_count: float, bandwidth_bytes_per_cycle: float, latency_cycles: float) -> float:
    if bytes_count <= 0.0 or bandwidth_bytes_per_cycle <= 0.0:
        return 0.0
    return max(bytes_count / bandwidth_bytes_per_cycle, latency_cycles)


def _effective_warp_k(kernel: GemmKernelSpec) -> int:
    if kernel.slice_k:
        return max(1, _ceil_div(kernel.cta_k, kernel.num_warp_tile_k))
    return min(kernel.warp_k, kernel.cta_k)


def _k_groups(k_extent: int, kernel: GemmKernelSpec) -> int:
    if k_extent <= 0:
        return 0
    k_per_warp_tile = _ceil_div(k_extent, kernel.num_warp_tile_k)
    return max(1, _ceil_div(k_per_warp_tile, kernel.warp_k))


def _classify_bottlenecks(
    problem: GemmProblemSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    timeline: TimelineResult,
    hardware: HardwareSpec,
    clock_hz: float,
    fixed_overhead_cycles: float,
) -> tuple[str, tuple[str, ...]]:
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    fixed_overhead_fraction = fixed_overhead_cycles / max(total_device_cycles, 1.0e-12)
    metrics = _roofline_metrics(
        problem,
        grid,
        traffic,
        timeline,
        hardware,
        clock_hz,
        total_device_cycles,
    )
    ceiling_util = metrics["ceiling_utilization"]
    labels = {
        "compute": "compute_roof_limited",
        "smem": "smem_bandwidth_limited",
        "l2": "l2_bandwidth_limited",
        "dram": "dram_bandwidth_limited",
    }
    candidates = {
        labels[name]: value
        for name, value in ceiling_util.items()
        if value is not None and math.isfinite(value)
    }
    epilogue_ratio = timeline.epilogue_cycles / max(timeline.kernel_cycles, 1.0e-12)
    if epilogue_ratio > 0.2:
        primary = "epilogue_limited"
    elif candidates:
        primary = max(candidates.items(), key=lambda item: item[1])[0]
    else:
        primary = "compute_roof_limited"
    roofline_primary = primary

    secondary: list[str] = []
    if fixed_overhead_fraction >= 0.5:
        primary = "fixed_overhead_limited"
        secondary.append(roofline_primary)
    elif fixed_overhead_fraction >= 0.2:
        secondary.append("fixed_overhead_limited")
    if timeline.compute_latency_utilization < 0.8:
        secondary.append("compute_latency_limited")
    if timeline.mma_ilp_efficiency < 0.8:
        secondary.append("low_mma_ilp")
    if timeline.compute_smem_overlap < 0.7:
        secondary.append("poor_compute_smem_overlap")
    if timeline.exposed_l2_cycles > 0.1 * max(timeline.full_wave.work_cycles, 1.0e-12):
        secondary.append("l2_latency_exposed")
    if timeline.exposed_dram_cycles > 0.1 * max(timeline.full_wave.work_cycles, 1.0e-12):
        secondary.append("dram_latency_exposed")
    if 0.0 < timeline.compute_l2_overlap < 0.7:
        secondary.append("poor_compute_l2_overlap")
    if 0.0 < timeline.compute_dram_overlap < 0.7:
        secondary.append("poor_compute_dram_overlap")
    if occupancy.wave_count <= 1 or occupancy.tail_efficiency < 0.6:
        secondary.append("insufficient_cta_waves")
    if grid.tile_efficiency < 0.9:
        secondary.append("edge_tile_predication_loss")
    if problem.epilogue_reads_c or epilogue_ratio > 0.2:
        secondary.append("epilogue_limited")
    for name, value in ceiling_util.items():
        if value is not None and value >= 0.8:
            secondary.append(f"{name}_ceiling_near_saturation")
    return primary, tuple(dict.fromkeys(item for item in secondary if item != primary))


def _global_factors(
    *,
    useful: float,
    issued: float,
    kernel_cycles: float,
    total_device_cycles: float,
) -> dict[str, float]:
    return {
        "useful_flop_efficiency": useful / max(issued, 1.0e-12),
        "kernel_scope_efficiency": kernel_cycles / max(total_device_cycles, 1.0e-12),
        "total_device_overhead_fraction": max(0.0, total_device_cycles - kernel_cycles)
        / max(total_device_cycles, 1.0e-12),
    }


def _phase_factors(timeline: TimelineResult) -> dict[str, float]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    return {
        "prologue_cycles": timeline.prologue_cycles,
        "work_cycles": timeline.work_cycles,
        "epilogue_cycles": timeline.epilogue_cycles,
        "prologue_fraction": timeline.prologue_cycles / kernel_cycles,
        "work_fraction": timeline.work_cycles / kernel_cycles,
        "epilogue_fraction": timeline.epilogue_cycles / kernel_cycles,
    }


def _compute_factors(
    *,
    useful: float,
    issued: float,
    achieved: float,
    peak_compute: float,
    timeline: TimelineResult,
) -> dict[str, float]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    math_group_cycles = max(
        timeline.mma_issue_cycles + timeline.mma_dependency_penalty_cycles,
        1.0e-12,
    )
    return {
        "raw_peak_flop_per_cycle": peak_compute,
        "achieved_flop_per_cycle": achieved,
        "compute_roof_utilization": achieved / max(peak_compute, 1.0e-12),
        "useful_flop_efficiency": useful / max(issued, 1.0e-12),
        "compute_active_cycles": timeline.compute_active_cycles,
        "compute_active_fraction": timeline.compute_active_cycles / kernel_cycles,
        "mma_issue_cycles": timeline.mma_issue_cycles,
        "mma_dependency_penalty_cycles": timeline.mma_dependency_penalty_cycles,
        "mma_issue_fraction_of_math_group": timeline.mma_issue_cycles
        / math_group_cycles,
        "mma_dependency_fraction_of_math_group": timeline.mma_dependency_penalty_cycles
        / math_group_cycles,
        "mma_ilp_efficiency": timeline.mma_ilp_efficiency,
        "compute_issue_utilization": timeline.compute_issue_utilization,
        "compute_latency_utilization": timeline.compute_latency_utilization,
    }


def _memory_factors(
    *,
    q_bytes: float,
    peak_bytes_per_cycle: float,
    active_cycles: float,
    utilization: float,
    overlap: float | None,
    exposed_cycles: float,
    epilogue_bytes: float,
    useful: float,
    achieved: float,
    kernel_cycles: float,
) -> dict[str, float | None]:
    kernel = max(kernel_cycles, 1.0e-12)
    if q_bytes <= 0.0 or peak_bytes_per_cycle <= 0.0:
        return {
            "arithmetic_intensity": None,
            "raw_bound": None,
            "achieved": achieved,
            "ceiling_utilization": None,
            "bytes": q_bytes,
            "active_cycles_at_peak": None,
            "active_fraction_of_kernel": None,
            "timeline_utilization": utilization,
            "exposed_cycles": exposed_cycles,
            "exposed_fraction_of_kernel": exposed_cycles / kernel,
            "hidden_fraction_of_service": None,
            "compute_overlap": overlap,
            "epilogue_bytes": epilogue_bytes,
            "epilogue_byte_fraction": None,
        }

    raw_bound = peak_bytes_per_cycle * useful / q_bytes
    return {
        "arithmetic_intensity": useful / q_bytes,
        "raw_bound": raw_bound,
        "achieved": achieved,
        "ceiling_utilization": achieved / max(raw_bound, 1.0e-12),
        "bytes": q_bytes,
        "active_cycles_at_peak": active_cycles,
        "active_fraction_of_kernel": active_cycles / kernel,
        "timeline_utilization": utilization,
        "exposed_cycles": exposed_cycles,
        "exposed_fraction_of_kernel": exposed_cycles / kernel,
        "hidden_fraction_of_service": max(
            0.0, 1.0 - exposed_cycles / max(active_cycles, 1.0e-12)
        ),
        "compute_overlap": overlap,
        "epilogue_bytes": epilogue_bytes,
        "epilogue_byte_fraction": epilogue_bytes / max(q_bytes, 1.0e-12),
    }


def _critical_path_factors(timeline: TimelineResult) -> dict[str, float]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    math_group_cycles = max(
        timeline.mma_issue_cycles + timeline.mma_dependency_penalty_cycles,
        1.0e-12,
    )
    return {
        "prologue_fraction": timeline.prologue_cycles / kernel_cycles,
        "work_fraction": timeline.work_cycles / kernel_cycles,
        "epilogue_fraction": timeline.epilogue_cycles / kernel_cycles,
        "exposed_l2_fraction": timeline.exposed_l2_cycles / kernel_cycles,
        "exposed_dram_fraction": timeline.exposed_dram_cycles / kernel_cycles,
        "max_exposed_memory_fraction": max(
            timeline.exposed_l2_cycles,
            timeline.exposed_dram_cycles,
        )
        / kernel_cycles,
        "mma_dependency_fraction_of_math_group": timeline.mma_dependency_penalty_cycles
        / math_group_cycles,
        "compute_inactive_fraction": max(0.0, 1.0 - timeline.compute_active_utilization),
    }


def _classify_attribution_bottleneck(
    timeline: TimelineResult,
) -> dict[str, float | str]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    math_group_cycles = max(
        timeline.mma_issue_cycles + timeline.mma_dependency_penalty_cycles,
        1.0e-12,
    )
    scores = {
        "prologue_dominated": timeline.prologue_cycles / kernel_cycles,
        "epilogue_dominated": timeline.epilogue_cycles / kernel_cycles,
        "l2_exposed": timeline.exposed_l2_cycles / kernel_cycles,
        "dram_exposed": timeline.exposed_dram_cycles / kernel_cycles,
        "mma_dependency_limited": timeline.mma_dependency_penalty_cycles
        / math_group_cycles,
        "compute_underoccupied": max(0.0, 1.0 - timeline.compute_active_utilization),
    }
    primary = max(scores, key=scores.__getitem__)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    secondary, secondary_score = (
        sorted_scores[1] if len(sorted_scores) > 1 else (primary, scores[primary])
    )
    return {
        "primary": primary,
        "primary_score": scores[primary],
        "secondary": secondary,
        "secondary_score": secondary_score,
        **scores,
    }


def _deprecated_effective_bounds(
    *,
    useful: float,
    q_smem: float,
    q_l2: float,
    q_dram: float,
    peak_compute: float,
    peak_smem: float,
    peak_l2: float,
    peak_dram: float,
    timeline: TimelineResult,
) -> dict[str, float | None]:
    return {
        "compute": peak_compute
        * timeline.compute_issue_utilization
        * timeline.compute_latency_utilization
        * timeline.compute_active_utilization,
        "smem": peak_smem * timeline.smem_utilization * (useful / q_smem)
        if q_smem
        else None,
        "l2": peak_l2 * timeline.l2_utilization * (useful / q_l2)
        if q_l2 and peak_l2
        else None,
        "dram": peak_dram * timeline.dram_utilization * (useful / q_dram)
        if q_dram
        else None,
    }


def _validate_roofline_attribution(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    warnings: list[str] = []
    raw = metrics["raw_bounds"]
    achieved = metrics["achieved"]["kernel_scope"]
    for name, bound in raw.items():
        if bound is None or bound <= 0.0:
            continue
        utilization = achieved / bound
        if utilization > 1.05:
            warnings.append(
                f"{name} utilization exceeds raw bound by more than 5%; "
                "check traffic accounting, useful/issued FLOPs, or peak rates"
            )

    breakdown = metrics["factor_breakdown"]
    global_factors = breakdown["global"]
    useful_efficiency = global_factors["useful_flop_efficiency"]
    if useful_efficiency > 1.000001:
        warnings.append(
            f"useful FLOP efficiency is {useful_efficiency:.6g}, expected <= 1"
        )

    phase = breakdown["phase"]
    phase_sum = (
        phase["prologue_fraction"]
        + phase["work_fraction"]
        + phase["epilogue_fraction"]
    )
    if abs(phase_sum - 1.0) > 1.0e-6:
        warnings.append(f"phase fractions sum to {phase_sum:.6g}, expected 1")

    for name in ("smem", "l2", "dram"):
        factors = breakdown[name]
        active_fraction = factors["active_fraction_of_kernel"]
        timeline_utilization = factors["timeline_utilization"]
        if active_fraction is not None:
            expected = min(1.0, active_fraction)
            if abs(expected - timeline_utilization) > 1.0e-6:
                warnings.append(
                    f"{name} active fraction {active_fraction:.6g} does not match "
                    f"timeline utilization {timeline_utilization:.6g}"
                )
        active_cycles = factors["active_cycles_at_peak"]
        exposed_cycles = factors["exposed_cycles"]
        if (
            active_cycles is not None
            and exposed_cycles is not None
            and exposed_cycles > active_cycles + 1.0e-6
        ):
            warnings.append(
                f"{name} exposed cycles exceed active service cycles; "
                "check memory overlap accounting"
            )
    return tuple(warnings)


def _roofline_metrics(
    problem: GemmProblemSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    timeline: TimelineResult,
    hardware: HardwareSpec,
    clock_hz: float,
    total_device_cycles: float,
) -> dict[str, Any]:
    q_smem = traffic.smem_total_bytes + timeline.epilogue_smem_bytes
    q_l2 = traffic.l2_requested_bytes
    q_dram = traffic.dram_unique_bytes
    useful = grid.useful_flops
    issued = grid.issued_flops
    achieved_kernel = useful / max(timeline.kernel_cycles, 1.0e-12)
    achieved_total_device = useful / max(total_device_cycles, 1.0e-12)
    peak_compute = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_smem = _smem_bandwidth_per_cycle(hardware, clock_hz, [])
    peak_l2 = (
        hardware.memory_levels["l2"].bandwidth_bytes_per_s / clock_hz
        if "l2" in hardware.memory_levels
        else 0.0
    )
    peak_dram = hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz

    raw_bounds = {
        "compute": peak_compute,
        "smem": peak_smem * (useful / q_smem) if q_smem else None,
        "l2": peak_l2 * (useful / q_l2) if q_l2 and peak_l2 else None,
        "dram": peak_dram * (useful / q_dram) if q_dram else None,
    }
    ceiling_utilization = {
        name: achieved_kernel / value if value and value > 0.0 else None
        for name, value in raw_bounds.items()
    }
    total_ceiling_utilization = {
        name: achieved_total_device / value if value and value > 0.0 else None
        for name, value in raw_bounds.items()
    }
    smem_factors = _memory_factors(
        q_bytes=q_smem,
        peak_bytes_per_cycle=peak_smem,
        active_cycles=timeline.smem_active_cycles,
        utilization=timeline.smem_utilization,
        overlap=timeline.compute_smem_overlap,
        exposed_cycles=0.0,
        epilogue_bytes=timeline.epilogue_smem_bytes,
        useful=useful,
        achieved=achieved_kernel,
        kernel_cycles=timeline.kernel_cycles,
    )
    l2_factors = _memory_factors(
        q_bytes=q_l2,
        peak_bytes_per_cycle=peak_l2,
        active_cycles=timeline.l2_active_cycles,
        utilization=timeline.l2_utilization,
        overlap=timeline.compute_l2_overlap,
        exposed_cycles=timeline.exposed_l2_cycles,
        epilogue_bytes=timeline.epilogue_l2_bytes,
        useful=useful,
        achieved=achieved_kernel,
        kernel_cycles=timeline.kernel_cycles,
    )
    dram_factors = _memory_factors(
        q_bytes=q_dram,
        peak_bytes_per_cycle=peak_dram,
        active_cycles=timeline.dram_active_cycles,
        utilization=timeline.dram_utilization,
        overlap=timeline.compute_dram_overlap,
        exposed_cycles=timeline.exposed_dram_cycles,
        epilogue_bytes=timeline.epilogue_dram_bytes,
        useful=useful,
        achieved=achieved_kernel,
        kernel_cycles=timeline.kernel_cycles,
    )
    return {
        "raw_bounds": raw_bounds,
        "achieved": {
            "kernel_scope": achieved_kernel,
            "device_scope": achieved_total_device,
        },
        "ceiling_utilization": ceiling_utilization,
        "total_ceiling_utilization": total_ceiling_utilization,
        "factor_breakdown": {
            "global": _global_factors(
                useful=useful,
                issued=issued,
                kernel_cycles=timeline.kernel_cycles,
                total_device_cycles=total_device_cycles,
            ),
            "phase": _phase_factors(timeline),
            "compute": _compute_factors(
                useful=useful,
                issued=issued,
                achieved=achieved_kernel,
                peak_compute=peak_compute,
                timeline=timeline,
            ),
            "smem": smem_factors,
            "l2": l2_factors,
            "dram": dram_factors,
            "critical_path": _critical_path_factors(timeline),
            "bottleneck_classification": _classify_attribution_bottleneck(timeline),
        },
        "effective_bounds_deprecated": _deprecated_effective_bounds(
            useful=useful,
            q_smem=q_smem,
            q_l2=q_l2,
            q_dram=q_dram,
            peak_compute=peak_compute,
            peak_smem=peak_smem,
            peak_l2=peak_l2,
            peak_dram=peak_dram,
            timeline=timeline,
        ),
    }


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
    fixed_overhead_cycles: float,
) -> dict[str, Any]:
    primary, secondary = bottlenecks
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    fixed_overhead_fraction = fixed_overhead_cycles / max(total_device_cycles, 1.0e-12)
    smem_load_per_cta_elements = _smem_load_per_cta_elements(problem, kernel, grid)
    shared_memory_bytes_per_cta = _ceil_scalar_bytes(
        smem_load_per_cta_elements * dtype_nbytes(problem.input_dtype)
    )
    q_smem = traffic.smem_total_bytes + timeline.epilogue_smem_bytes
    q_l2 = traffic.l2_requested_bytes
    q_dram = traffic.dram_unique_bytes
    useful = grid.useful_flops
    total_q = q_smem + q_l2 + q_dram
    roofline_metrics = _roofline_metrics(
        problem,
        grid,
        traffic,
        timeline,
        hardware,
        clock_hz,
        total_device_cycles,
    )
    attribution_warnings = tuple(
        f"roofline_attribution: {message}"
        for message in _validate_roofline_attribution(roofline_metrics)
    )
    all_warnings = warnings + attribution_warnings
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
            "num_warp_tile_k": kernel.num_warp_tile_k,
            "mma_shape": {"m": kernel.mma_m, "n": kernel.mma_n, "k": kernel.mma_k},
            "pipeline_stages": kernel.pipeline_stages,
            "warps_per_cta": kernel.warps_per_cta,
            "threads_per_cta": kernel.threads_per_cta,
            "registers_per_thread": kernel.registers_per_thread,
            "smem_load_per_cta_elements": smem_load_per_cta_elements,
            "shared_memory_bytes_per_cta": shared_memory_bytes_per_cta,
            "max_concurrent_ctas_per_sm": kernel.max_concurrent_ctas_per_sm,
            "slice_k": kernel.slice_k,
        },
        "predicted_elapsed_cycles": total_device_cycles,
        "modeled_device_cycles": timeline.kernel_cycles,
        "device_fixed_overhead_cycles": fixed_overhead_cycles,
        "total_device_cycles": total_device_cycles,
        "device_fixed_overhead_s": fixed_overhead_cycles / clock_hz,
        "device_fixed_overhead_fraction": fixed_overhead_fraction,
        "predicted_flop_per_cycle": flops_per_cycle,
        "modeled_flop_per_cycle": (
            grid.useful_flops / timeline.kernel_cycles if timeline.kernel_cycles else 0.0
        ),
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
        "wave_shape": {
            "full_wave_ctas": occupancy.full_wave_ctas,
            "last_wave_ctas": occupancy.last_wave_ctas,
            "last_wave_busy_sms": occupancy.last_wave_busy_sms,
            "last_wave_lazy_sms": occupancy.last_wave_lazy_sms,
            "last_wave_busy_ctas_per_sm": occupancy.last_wave_busy_ctas_per_sm,
            "last_wave_lazy_ctas_per_sm": occupancy.last_wave_lazy_ctas_per_sm,
        },
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
            "a_l2_requested": traffic.a_l2_requested_bytes,
            "b_l2_requested": traffic.b_l2_requested_bytes,
            "a_dram_unique": traffic.a_dram_unique_bytes,
            "b_dram_unique": traffic.b_dram_unique_bytes,
            "c_read": traffic.c_read_transaction_bytes,
            "d_store": traffic.d_store_transaction_bytes,
            "l2_requested": traffic.l2_requested_bytes,
            "dram_unique": traffic.dram_unique_bytes,
            "smem_read": traffic.smem_read_bytes,
            "smem_write": traffic.smem_write_bytes,
            "epilogue_smem": timeline.epilogue_smem_bytes,
            "epilogue_l2": timeline.epilogue_l2_bytes,
            "epilogue_dram": timeline.epilogue_dram_bytes,
            "sector_size": traffic.sector_size_bytes,
            "line_size": traffic.line_size_bytes,
        },
        "wave_pipeline": {
            "full": _wave_pipeline_diagnostics(timeline.full_wave),
            "last": _wave_pipeline_diagnostics(timeline.last_wave),
        },
        "pipeline_components": {
            "groups_k": timeline.groups_k,
            "last_stage_groups_k": timeline.last_stage_groups_k,
            "last_stage_k": timeline.last_stage_k,
            "memory_pipeline_groups": timeline.memory_pipeline_groups,
            "last_memory_pipeline_stages": timeline.last_memory_pipeline_stages,
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
            "work": timeline.work_cycles,
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
        "roofline_bounds_flop_per_cycle": roofline_metrics[
            "effective_bounds_deprecated"
        ],
        "roofline_effective_bounds_deprecated_flop_per_cycle": roofline_metrics[
            "effective_bounds_deprecated"
        ],
        "roofline_raw_bounds_flop_per_cycle": roofline_metrics["raw_bounds"],
        "roofline_achieved_flop_per_cycle": roofline_metrics["achieved"],
        "roofline_ceiling_utilization": roofline_metrics["ceiling_utilization"],
        "roofline_total_ceiling_utilization": roofline_metrics[
            "total_ceiling_utilization"
        ],
        "roofline_factor_breakdown": roofline_metrics["factor_breakdown"],
        "memory_level_latencies_s": {
            "hbm": hardware.memory_levels["hbm"].latency_s,
            "l2": hardware.memory_levels["l2"].latency_s
            if "l2" in hardware.memory_levels
            else None,
            "sram": hardware.memory_levels["sram"].latency_s
            if "sram" in hardware.memory_levels
            else None,
            "register": None,
        },
        "primary_bottleneck": primary,
        "secondary_bottlenecks": secondary,
        "warnings": all_warnings,
        "assumptions": (
            "deterministic_tiled_gemm_access_pattern",
            "first_touch_l2_reuse_when_l2_exists",
            "dram_unique_first_touch_plus_output_writeback",
            "artifact_style_full_last_wave_pipeline",
        )
        + (
            ("fixed_device_overhead_calibrated_from_small_gemm_residuals",)
            if fixed_overhead_cycles > 0.0
            else ("no_fixed_device_overhead_configured",)
        )
        + (
            "shared_memory_bank_conflict_factor_1",
            "four_smsps_per_sm_assumed",
        ),
        "debug_trace": _debug_trace(
            problem,
            kernel,
            grid,
            traffic,
            occupancy,
            timeline,
            primary,
            secondary,
            fixed_overhead_cycles,
            total_device_cycles,
        ),
    }


def _wave_pipeline_diagnostics(wave: WavePipelineResult) -> dict[str, Any]:
    return {
        "active_ctas": wave.active_ctas,
        "busy_sms": wave.busy_sms,
        "lazy_sms": wave.lazy_sms,
        "busy_ctas_per_sm": wave.busy_ctas_per_sm,
        "lazy_ctas_per_sm": wave.lazy_ctas_per_sm,
        "start_cycles": wave.start_cycles,
        "work_cycles": wave.work_cycles,
        "end_cycles": wave.end_cycles,
        "total_cycles": wave.total_cycles,
        "sm_stage_cycles": wave.sm_stage_cycles,
        "sm_last_stage_cycles": wave.sm_last_stage_cycles,
        "smem_group_cycles": wave.smem_group_cycles,
        "math_group_cycles": wave.math_group_cycles,
        "math_issue_group_cycles": wave.math_issue_group_cycles,
        "math_latency_group_cycles": wave.math_latency_group_cycles,
        "memory_full_stage_cycles": wave.memory_full_stage_cycles,
        "memory_last_stage_cycles": wave.memory_last_stage_cycles,
        "l2_full_stage_cycles": wave.l2_full_stage_cycles,
        "l2_last_stage_cycles": wave.l2_last_stage_cycles,
        "dram_full_stage_cycles": wave.dram_full_stage_cycles,
        "dram_last_stage_cycles": wave.dram_last_stage_cycles,
        "epilogue_global_cycles": wave.epilogue_global_cycles,
        "epilogue_smem_cycles": wave.epilogue_smem_cycles,
        "slice_k_extra_cycles": wave.slice_k_extra_cycles,
        "epilogue_smem_bytes": wave.epilogue_smem_bytes,
        "epilogue_l2_bytes": wave.epilogue_l2_bytes,
        "epilogue_dram_bytes": wave.epilogue_dram_bytes,
        "l2_epilogue_cycles": wave.l2_epilogue_cycles,
        "dram_epilogue_cycles": wave.dram_epilogue_cycles,
        "exposed_l2_cycles": wave.exposed_l2_cycles,
        "exposed_dram_cycles": wave.exposed_dram_cycles,
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
    fixed_overhead_cycles: float,
    total_device_cycles: float,
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
            f"full wave cycles: {timeline.full_wave.total_cycles:.4g}",
            f"last wave cycles: {timeline.last_wave.total_cycles:.4g}",
            f"modeled device cycles: {timeline.kernel_cycles:.4g}",
            f"fixed overhead cycles: {fixed_overhead_cycles:.4g}",
            f"total device cycles: {total_device_cycles:.4g}",
            f"A logical bytes: {traffic.a_logical_bytes}",
            f"B logical bytes: {traffic.b_logical_bytes}",
            f"D store bytes: {traffic.d_store_logical_bytes}",
            f"L2 requested bytes: {traffic.l2_requested_bytes}",
            f"DRAM unique bytes: {traffic.dram_unique_bytes}",
            f"SMEM bytes: {traffic.smem_total_bytes + timeline.epilogue_smem_bytes:.4g}",
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
    if hardware.compute.num_sms is not None:
        warnings.append("sram_level_absent_smem_bandwidth_inferred_from_shared_banks")
        return float(hardware.compute.num_sms * 32 * 4)
    warnings.append("sram_level_absent_smem_bandwidth_inferred_from_hbm")
    return hardware.memory_levels["hbm"].bandwidth_bytes_per_s * 8.0 / clock_hz


def _shared_latency_cycles(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    sram = hardware.memory_levels.get("sram")
    if sram is not None and sram.latency_s > 0.0:
        return sram.latency_s * clock_hz
    _append_warning_once(warnings, "shared_latency_cycles_default_29")
    return 29.0


def _memory_latency_cycles(
    level: MemoryLevel, clock_hz: float, name: str, default_cycles: float, warnings: list[str]
) -> float:
    if level.latency_s > 0.0:
        return level.latency_s * clock_hz
    _append_warning_once(warnings, f"{name}_latency_cycles_default_{default_cycles:g}")
    return default_cycles


def _tensor_latency_cycles(
    kernel: GemmKernelSpec, hardware: HardwareSpec, warnings: list[str]
) -> float:
    if hardware.compute.tensor_latency_cycles is not None:
        return float(hardware.compute.tensor_latency_cycles)
    default = {
        (16, 8, 8): 17.5,
        (16, 8, 16): 26.0,
    }.get((kernel.mma_m, kernel.mma_n, kernel.mma_k), 8.0)
    _append_warning_once(warnings, f"tensor_latency_cycles_default_{default:g}")
    return default


def _append_warning_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)


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
