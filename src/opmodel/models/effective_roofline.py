from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from opmodel.api import EngineKind, LocalOp, OpKind, OpProfile
from opmodel.energy import apply_calibrated_energy_model
from opmodel.estimator import DispatchingOpModel
from opmodel.hardware import HardwareSpec
import opmodel.models.extended_roofline as extended
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
from opmodel.ops import dtype_nbytes, footprint_from_tensors


_SMSPS_PER_SM = 4
_LIMITING_RESOURCE_ORDER = ("tensor", "smem", "l2", "hbm")


@dataclass(frozen=True)
class SMOccupancyClass:
    name: str
    sm_count: int
    ctas_per_sm: int


@dataclass(frozen=True)
class OccupancyClassResult:
    name: str
    sm_count: int
    ctas_per_sm: int
    warps_per_sm: int
    warps_by_smsp: tuple[int, int, int, int]
    accumulator_chains_by_smsp: tuple[int, int, int, int]
    tensor_raw_flops_per_cycle_per_sm: float
    tensor_effective_flops_per_cycle_per_sm: float
    tensor_group_flops_per_sm: float
    tensor_cycles: float
    smem_raw_bytes_per_cycle_per_sm: float
    smem_effective_bytes_per_cycle_per_sm: float
    smem_group_bytes_per_sm: float
    smem_cycles: float
    local_body_cycles: float


@dataclass(frozen=True)
class MemoryWindowResult:
    total_stage_equivalents: float
    pipeline_stages: int
    window_count: int
    full_window_count: int
    final_window_stage_equivalents: float
    bytes_per_cta_stage: float
    full_window_bytes: float
    final_window_bytes: float
    total_bytes: float
    first_window_bytes: float
    remaining_bytes: float
    raw_bytes_per_cycle: float
    full_window_effective_bytes_per_cycle: float
    final_window_effective_bytes_per_cycle: float
    effective_bytes_per_cycle: float
    full_window_cycles: float
    final_window_cycles: float
    first_window_cycles: float
    remaining_cycles: float
    total_cycles: float
    concurrency_factor: float


@dataclass(frozen=True)
class EpilogueRooflineResult:
    smem_bytes: float
    l2_bytes: float
    dram_bytes: float
    smem_cycles: float
    slice_k_extra_cycles: float
    l2_cycles: float
    dram_cycles: float
    global_cycles: float
    total_cycles: float


@dataclass(frozen=True)
class WaveRooflineResult:
    active_ctas: int
    active_sms: int
    busy_sms: int
    lazy_sms: int
    busy_ctas_per_sm: int
    lazy_ctas_per_sm: int
    useful_flops: float
    issued_flops: float
    smem_read_bytes: float
    l2_main_bytes: float
    dram_main_bytes: float
    tc_raw_flops_per_cycle: float
    tc_effective_flops_per_cycle: float
    smem_raw_bytes_per_cycle: float
    smem_effective_bytes_per_cycle: float
    l2_raw_bytes_per_cycle: float
    l2_effective_bytes_per_cycle: float
    dram_raw_bytes_per_cycle: float
    dram_effective_bytes_per_cycle: float
    tc_concurrency_factor: float
    smem_concurrency_factor: float
    l2_concurrency_factor: float
    dram_concurrency_factor: float
    tc_body_cycles: float
    smem_body_cycles: float
    l2_total_cycles: float
    dram_total_cycles: float
    l2_remaining_cycles: float
    dram_remaining_cycles: float
    prologue_cycles: float
    body_cycles: float
    epilogue_cycles: float
    total_cycles: float
    tile_efficiency: float
    limiting_resource: str
    limiting_resources: tuple[str, ...]
    body_limiting_resources: tuple[str, ...]
    boundary_phase_dominance: str | None
    pure_mainloop_roofline_cycles: float
    predicted_useful_flops_per_cycle: float
    accumulator_chains_per_warp: int
    dependent_k_steps_per_chain: int
    k_groups: int
    memory_window_count: int
    occupancy_classes: tuple[OccupancyClassResult, ...]
    l2_windows: MemoryWindowResult
    dram_windows: MemoryWindowResult
    epilogue: EpilogueRooflineResult


@dataclass(frozen=True)
class EffectiveTimelineResult:
    kernel_cycles: float
    full_wave: WaveRooflineResult
    last_wave: WaveRooflineResult
    prologue_cycles: float
    body_cycles: float
    epilogue_cycles: float
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


def _nonzero_effective_rate(
    peak_rate: float, in_flight_work: float, latency_cycles: float
) -> float:
    """Return min(raw peak, in-flight work / latency), with zero-work semantics."""
    if peak_rate < 0.0:
        raise ValueError("peak rate must be nonnegative")
    if in_flight_work < 0.0:
        raise ValueError("in-flight work must be nonnegative")
    if latency_cycles < 0.0:
        raise ValueError("latency must be nonnegative")
    if peak_rate == 0.0 or in_flight_work == 0.0:
        return 0.0
    latency_rate = math.inf if latency_cycles == 0.0 else in_flight_work / latency_cycles
    return min(peak_rate, latency_rate)


def _service_from_effective_rate(work: float, rate: float) -> float:
    if work < 0.0 or rate < 0.0:
        raise ValueError("work and rate must be nonnegative")
    if work == 0.0:
        return 0.0
    if rate == 0.0:
        raise ValueError("nonzero work requires a positive effective rate")
    return work / rate


def _warps_by_smsp(warps_on_sm: int) -> tuple[int, int, int, int]:
    if warps_on_sm < 0:
        raise ValueError("warps_on_sm must be nonnegative")
    quotient, remainder = divmod(warps_on_sm, _SMSPS_PER_SM)
    result = tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(_SMSPS_PER_SM)
    )
    if sum(result) != warps_on_sm:
        raise AssertionError("SMSP warp distribution does not conserve warps")
    return result  # type: ignore[return-value]


def _sm_occupancy_classes(
    *,
    active_ctas: int,
    busy_sms: int,
    lazy_sms: int,
    busy_ctas_per_sm: int,
    lazy_ctas_per_sm: int,
) -> tuple[SMOccupancyClass, ...]:
    values = (
        active_ctas,
        busy_sms,
        lazy_sms,
        busy_ctas_per_sm,
        lazy_ctas_per_sm,
    )
    if any(value < 0 for value in values):
        raise ValueError("wave occupancy values must be nonnegative")
    represented = (
        busy_sms * busy_ctas_per_sm + lazy_sms * lazy_ctas_per_sm
    )
    if represented != active_ctas:
        raise ValueError(
            "wave occupancy decomposition does not conserve active CTAs: "
            f"expected {active_ctas}, represented {represented}"
        )
    classes = (
        SMOccupancyClass("busy", busy_sms, busy_ctas_per_sm),
        SMOccupancyClass("lazy", lazy_sms, lazy_ctas_per_sm),
    )
    return tuple(
        item for item in classes if item.sm_count > 0 and item.ctas_per_sm > 0
    )


def _memory_window_summary(
    *,
    total_stage_equivalents: float,
    pipeline_stages: int,
    active_ctas: int,
    bytes_per_cta_stage: float,
    peak_bytes_per_cycle: float,
    latency_cycles: float,
) -> MemoryWindowResult:
    if total_stage_equivalents < 0.0:
        raise ValueError("total_stage_equivalents must be nonnegative")
    if pipeline_stages <= 0:
        raise ValueError("pipeline_stages must be positive")
    if active_ctas < 0 or bytes_per_cta_stage < 0.0:
        raise ValueError("memory traffic inputs must be nonnegative")
    if peak_bytes_per_cycle < 0.0 or latency_cycles < 0.0:
        raise ValueError("memory rate and latency must be nonnegative")

    if total_stage_equivalents == 0.0 or active_ctas == 0 or bytes_per_cta_stage == 0.0:
        window_count = 0 if total_stage_equivalents == 0.0 else math.ceil(
            total_stage_equivalents / pipeline_stages
        )
        final_stages = (
            total_stage_equivalents - max(0, window_count - 1) * pipeline_stages
            if window_count
            else 0.0
        )
        return MemoryWindowResult(
            total_stage_equivalents=total_stage_equivalents,
            pipeline_stages=pipeline_stages,
            window_count=window_count,
            full_window_count=max(0, window_count - 1),
            final_window_stage_equivalents=final_stages,
            bytes_per_cta_stage=bytes_per_cta_stage,
            full_window_bytes=0.0,
            final_window_bytes=0.0,
            total_bytes=0.0,
            first_window_bytes=0.0,
            remaining_bytes=0.0,
            raw_bytes_per_cycle=peak_bytes_per_cycle,
            full_window_effective_bytes_per_cycle=0.0,
            final_window_effective_bytes_per_cycle=0.0,
            effective_bytes_per_cycle=0.0,
            full_window_cycles=0.0,
            final_window_cycles=0.0,
            first_window_cycles=0.0,
            remaining_cycles=0.0,
            total_cycles=0.0,
            concurrency_factor=0.0,
        )

    window_count = math.ceil(total_stage_equivalents / pipeline_stages)
    full_window_count = max(0, window_count - 1)
    final_stages = total_stage_equivalents - full_window_count * pipeline_stages
    if final_stages <= 0.0 or final_stages > pipeline_stages + 1.0e-12:
        raise AssertionError("invalid final memory-window stage count")
    full_bytes = active_ctas * pipeline_stages * bytes_per_cta_stage
    final_bytes = active_ctas * final_stages * bytes_per_cta_stage
    total_bytes = active_ctas * total_stage_equivalents * bytes_per_cta_stage
    reconstructed = full_window_count * full_bytes + final_bytes
    if not math.isclose(reconstructed, total_bytes, rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise AssertionError("memory windows do not conserve traffic")

    full_rate = _nonzero_effective_rate(
        peak_bytes_per_cycle, full_bytes, latency_cycles
    )
    final_rate = _nonzero_effective_rate(
        peak_bytes_per_cycle, final_bytes, latency_cycles
    )
    full_cycles = _service_from_effective_rate(full_bytes, full_rate)
    final_cycles = _service_from_effective_rate(final_bytes, final_rate)
    total_cycles = full_window_count * full_cycles + final_cycles
    first_bytes = full_bytes if window_count > 1 else final_bytes
    first_cycles = full_cycles if window_count > 1 else final_cycles
    remaining_bytes = total_bytes - first_bytes
    remaining_cycles = max(0.0, total_cycles - first_cycles)
    effective = total_bytes / total_cycles if total_cycles else 0.0
    if effective > peak_bytes_per_cycle + 1.0e-9:
        raise AssertionError("effective memory bandwidth exceeds raw peak")
    if not math.isclose(
        first_cycles + remaining_cycles,
        total_cycles,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise AssertionError("memory prologue subtraction does not conserve service time")
    return MemoryWindowResult(
        total_stage_equivalents=total_stage_equivalents,
        pipeline_stages=pipeline_stages,
        window_count=window_count,
        full_window_count=full_window_count,
        final_window_stage_equivalents=final_stages,
        bytes_per_cta_stage=bytes_per_cta_stage,
        full_window_bytes=full_bytes,
        final_window_bytes=final_bytes,
        total_bytes=total_bytes,
        first_window_bytes=first_bytes,
        remaining_bytes=remaining_bytes,
        raw_bytes_per_cycle=peak_bytes_per_cycle,
        full_window_effective_bytes_per_cycle=full_rate,
        final_window_effective_bytes_per_cycle=final_rate,
        effective_bytes_per_cycle=effective,
        full_window_cycles=full_cycles,
        final_window_cycles=final_cycles,
        first_window_cycles=first_cycles,
        remaining_cycles=remaining_cycles,
        total_cycles=total_cycles,
        concurrency_factor=(effective / peak_bytes_per_cycle if peak_bytes_per_cycle else 0.0),
    )


class EffectiveRooflineModel(DispatchingOpModel):
    """Phase-aware GEMM roofline with concurrency-derived effective ceilings."""

    def __init__(self) -> None:
        super().__init__(
            {
                OpKind.GEMM: EffectiveGemmEstimator(batched=False),
                OpKind.BATCHED_GEMM: EffectiveGemmEstimator(batched=True),
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


class EffectiveGemmEstimator:
    def __init__(self, *, batched: bool) -> None:
        self._batched = batched

    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        _effective_selection_backend(op.attrs)
        problem = extended._problem_spec(op, batched=self._batched)
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

        if extended._should_select_gemm_kernel(op.attrs, hardware):
            return _estimate_selected_gemm_kernel(
                op=op,
                problem=problem,
                footprint=footprint,
                hardware=hardware,
                batched=self._batched,
            )

        warnings: list[str] = []
        kernel = extended._kernel_spec(op.attrs, hardware, warnings)
        return _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=self._batched,
            warnings=warnings,
        )


def _effective_selection_backend(attrs: Mapping[str, Any]) -> str:
    backend = str(attrs.get("gemm_selection_backend", "effective_roofline"))
    if backend != "effective_roofline":
        raise ValueError("gemm_selection_backend must be effective_roofline")
    return backend


def evaluate_gemm_template_candidates(
    op: LocalOp,
    hardware: HardwareSpec,
    *,
    shortlist_size: int | None = None,
) -> tuple[extended.GemmCandidateEvaluation, ...]:
    if op.kind not in (OpKind.GEMM, OpKind.BATCHED_GEMM):
        raise ValueError("GEMM template candidates require a GEMM op")
    _effective_selection_backend(op.attrs)
    problem = extended._problem_spec(op, batched=op.kind == OpKind.BATCHED_GEMM)
    if _matmul_engine(problem.input_dtype, hardware) != EngineKind.TENSOR or hardware.kind != "gpu":
        return ()
    size = (
        extended._positive_int_value(shortlist_size, "shortlist_size")
        if shortlist_size is not None
        else extended._selection_shortlist_size(op.attrs)
    )
    return _evaluate_gemm_template_candidates(
        problem=problem,
        footprint=footprint_from_tensors(op),
        hardware=hardware,
        batched=op.kind == OpKind.BATCHED_GEMM,
        objective=extended._selection_objective(op.attrs),
        shortlist_size=size,
    )


def select_gemm_template_candidate(
    candidates: tuple[extended.GemmCandidateEvaluation, ...],
    *,
    objective: str = "latency",
) -> extended.GemmCandidateEvaluation:
    return extended.select_gemm_template_candidate(candidates, objective=objective)


def _evaluate_gemm_template_candidates(
    *,
    problem: extended.GemmProblemSpec,
    footprint,
    hardware: HardwareSpec,
    batched: bool,
    objective: str,
    shortlist_size: int,
) -> tuple[extended.GemmCandidateEvaluation, ...]:
    legal = extended._legal_kernel_templates(problem, hardware)
    ranked = sorted(
        legal,
        key=lambda template: extended._cheap_template_score(
            template, problem, hardware, objective
        ),
    )
    evaluations: list[extended.GemmCandidateEvaluation] = []
    for rank, template in enumerate(ranked[:shortlist_size], start=1):
        kernel = extended._kernel_from_template(template)
        profile = _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=batched,
            warnings=[],
        )
        evaluations.append(
            extended.GemmCandidateEvaluation(
                template=template,
                kernel=kernel,
                profile=profile,
                selection_energy_j=extended._selection_energy_j(
                    profile, problem, hardware
                ),
                cheap_rank=rank,
                cheap_score=extended._cheap_template_score(
                    template, problem, hardware, objective
                ),
            )
        )
    return tuple(evaluations)


def _estimate_selected_gemm_kernel(
    *,
    op: LocalOp,
    problem: extended.GemmProblemSpec,
    footprint,
    hardware: HardwareSpec,
    batched: bool,
) -> OpProfile:
    objective = extended._selection_objective(op.attrs)
    backend = _effective_selection_backend(op.attrs)
    shortlist_size = extended._selection_shortlist_size(op.attrs)
    candidates = _evaluate_gemm_template_candidates(
        problem=problem,
        footprint=footprint,
        hardware=hardware,
        batched=batched,
        objective=objective,
        shortlist_size=shortlist_size,
    )
    if not candidates:
        warnings = ["gemm_selection_no_legal_catalog_template"]
        kernel = extended._kernel_spec(op.attrs, hardware, warnings)
        profile = _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=batched,
            warnings=warnings,
        )
        return extended._with_gemm_selection_diagnostics(
            profile,
            {
                "enabled": False,
                "reason": "no_legal_catalog_template",
                "objective": objective,
                "backend": backend,
                "catalog_size": len(
                    extended._kernel_template_catalog(problem.input_dtype, hardware)
                ),
                "legal_candidates": 0,
                "shortlist_size": 0,
            },
        )
    selected = select_gemm_template_candidate(candidates, objective=objective)
    return extended._with_gemm_selection_diagnostics(
        selected.profile,
        extended._gemm_selection_metadata(
            selected=selected,
            candidates=candidates,
            objective=objective,
            backend=backend,
            catalog_size=len(
                extended._kernel_template_catalog(problem.input_dtype, hardware)
            ),
            legal_count=len(extended._legal_kernel_templates(problem, hardware)),
        ),
    )


def _estimate_gemm_with_kernel(
    *,
    problem: extended.GemmProblemSpec,
    footprint,
    kernel: extended.GemmKernelSpec,
    hardware: HardwareSpec,
    batched: bool,
    warnings: list[str],
) -> OpProfile:
    clock_hz = extended._clock_hz(hardware, warnings)
    grid = extended._grid_accounting(problem, kernel)
    traffic = extended._traffic_accounting(problem, kernel, grid, hardware, warnings)
    occupancy = extended._occupancy(problem, kernel, grid, hardware, warnings)
    timeline = _effective_timeline(
        problem, kernel, grid, traffic, occupancy, hardware, clock_hz, warnings
    )
    fixed_overhead_cycles = float(hardware.compute.device_fixed_overhead_cycles or 0)
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    latency_s = total_device_cycles / clock_hz
    flops_per_cycle = (
        grid.useful_flops / total_device_cycles if total_device_cycles else 0.0
    )
    tflops_per_s = grid.useful_flops / latency_s / 1.0e12 if latency_s else 0.0
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


def _effective_timeline(
    problem: extended.GemmProblemSpec,
    kernel: extended.GemmKernelSpec,
    grid: extended.GridAccounting,
    traffic: extended.TrafficAccounting,
    occupancy: extended.OccupancyResult,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> EffectiveTimelineResult:
    peak_compute = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_hbm = hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz
    l2_level = hardware.memory_levels.get("l2")
    peak_l2 = l2_level.bandwidth_bytes_per_s / clock_hz if l2_level else 0.0
    peak_smem = extended._smem_bandwidth_per_cycle(hardware, clock_hz, warnings)
    shared_latency = extended._shared_latency_cycles(hardware, clock_hz, warnings)
    l2_latency = (
        extended._memory_latency_cycles(l2_level, clock_hz, "l2", 261.5, warnings)
        if l2_level
        else 0.0
    )
    dram_latency = extended._memory_latency_cycles(
        hardware.memory_levels["hbm"], clock_hz, "hbm", 466.3, warnings
    )
    tensor_latency = extended._tensor_latency_cycles(kernel, hardware, warnings)

    groups_k = extended._k_groups(kernel.cta_k, kernel)
    last_stage_k = (problem.k - 1) % kernel.cta_k + 1 if problem.k else 0
    last_stage_groups_k = extended._k_groups(last_stage_k, kernel)
    memory_pipeline_groups = extended._ceil_div(
        grid.k_stages, kernel.pipeline_stages
    )
    last_memory_pipeline_stages = (
        (grid.k_stages - 1) % kernel.pipeline_stages + 1 if grid.k_stages else 0
    )
    load_stage_count = max(1, grid.cta_count * grid.k_stages)
    avg_l2 = (
        (traffic.a_l2_requested_bytes + traffic.b_l2_requested_bytes)
        / load_stage_count
        if l2_level
        else 0.0
    )
    avg_dram = (
        traffic.a_dram_unique_bytes + traffic.b_dram_unique_bytes
    ) / load_stage_count

    common = dict(
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        peak_compute_flops_per_cycle=peak_compute,
        peak_smem_bw_per_cycle=peak_smem,
        peak_l2_bw_per_cycle=peak_l2,
        peak_hbm_bw_per_cycle=peak_hbm,
        shared_latency_cycles=shared_latency,
        tensor_latency_cycles=tensor_latency,
        l2_latency_cycles=l2_latency,
        dram_latency_cycles=dram_latency,
        avg_l2_load_bytes_per_cta_stage=avg_l2,
        avg_dram_load_bytes_per_cta_stage=avg_dram,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )
    full_wave = _wave_effective_roofline(
        **common,
        active_ctas=occupancy.full_wave_ctas,
        busy_sms=occupancy.num_sms,
        lazy_sms=0,
        busy_ctas_per_sm=occupancy.resident_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.resident_ctas_per_sm,
    )
    last_wave = _wave_effective_roofline(
        **common,
        active_ctas=occupancy.last_wave_ctas,
        busy_sms=occupancy.last_wave_busy_sms,
        lazy_sms=occupancy.last_wave_lazy_sms,
        busy_ctas_per_sm=occupancy.last_wave_busy_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.last_wave_lazy_ctas_per_sm,
    )
    full_count = max(0, occupancy.wave_count - 1)
    kernel_cycles = full_count * full_wave.total_cycles + last_wave.total_cycles
    prologue = full_count * full_wave.prologue_cycles + last_wave.prologue_cycles
    body = full_count * full_wave.body_cycles + last_wave.body_cycles
    epilogue = full_count * full_wave.epilogue_cycles + last_wave.epilogue_cycles
    issued = full_count * full_wave.issued_flops + last_wave.issued_flops
    smem_bytes = full_count * (
        full_wave.smem_read_bytes + full_wave.epilogue.smem_bytes
    ) + last_wave.smem_read_bytes + last_wave.epilogue.smem_bytes
    l2_bytes = full_count * (
        full_wave.l2_main_bytes + full_wave.epilogue.l2_bytes
    ) + last_wave.l2_main_bytes + last_wave.epilogue.l2_bytes
    dram_bytes = full_count * (
        full_wave.dram_main_bytes + full_wave.epilogue.dram_bytes
    ) + last_wave.dram_main_bytes + last_wave.epilogue.dram_bytes
    epilogue_smem_bytes = (
        full_count * full_wave.epilogue.smem_bytes + last_wave.epilogue.smem_bytes
    )
    epilogue_l2_bytes = (
        full_count * full_wave.epilogue.l2_bytes + last_wave.epilogue.l2_bytes
    )
    epilogue_dram_bytes = (
        full_count * full_wave.epilogue.dram_bytes + last_wave.epilogue.dram_bytes
    )
    return EffectiveTimelineResult(
        kernel_cycles=kernel_cycles,
        full_wave=full_wave,
        last_wave=last_wave,
        prologue_cycles=prologue,
        body_cycles=body,
        epilogue_cycles=epilogue,
        compute_active_cycles=issued / peak_compute if peak_compute else 0.0,
        smem_active_cycles=smem_bytes / peak_smem if peak_smem else 0.0,
        l2_active_cycles=l2_bytes / peak_l2 if peak_l2 else 0.0,
        dram_active_cycles=dram_bytes / peak_hbm if peak_hbm else 0.0,
        epilogue_smem_bytes=epilogue_smem_bytes,
        epilogue_l2_bytes=epilogue_l2_bytes,
        epilogue_dram_bytes=epilogue_dram_bytes,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )


def _wave_effective_roofline(
    *,
    problem: extended.GemmProblemSpec,
    kernel: extended.GemmKernelSpec,
    grid: extended.GridAccounting,
    traffic: extended.TrafficAccounting,
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
) -> WaveRooflineResult:
    if active_ctas <= 0:
        return _zero_wave_roofline(busy_sms + lazy_sms)

    classes = _sm_occupancy_classes(
        active_ctas=active_ctas,
        busy_sms=busy_sms,
        lazy_sms=lazy_sms,
        busy_ctas_per_sm=busy_ctas_per_sm,
        lazy_ctas_per_sm=lazy_ctas_per_sm,
    )
    total_sms = busy_sms + lazy_sms
    if total_sms <= 0:
        raise ValueError("a nonempty wave requires at least one represented SM")
    _validate_wave_geometry(
        kernel=kernel,
        grid=grid,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )
    if any(
        value < 0.0
        for value in (
            peak_compute_flops_per_cycle,
            peak_smem_bw_per_cycle,
            peak_l2_bw_per_cycle,
            peak_hbm_bw_per_cycle,
            shared_latency_cycles,
            tensor_latency_cycles,
            l2_latency_cycles,
            dram_latency_cycles,
            avg_l2_load_bytes_per_cta_stage,
            avg_dram_load_bytes_per_cta_stage,
        )
    ):
        raise ValueError("rates, latencies, and traffic must be nonnegative")

    effective_warp_m = min(kernel.warp_m, kernel.cta_m)
    effective_warp_n = min(kernel.warp_n, kernel.cta_n)
    effective_warp_k = extended._effective_warp_k(kernel)
    dtype_bytes = dtype_nbytes(problem.input_dtype)
    mma_flops = float(2 * kernel.mma_m * kernel.mma_n * kernel.mma_k)
    accumulator_chains = (
        extended._ceil_div(effective_warp_m, kernel.mma_m)
        * extended._ceil_div(effective_warp_n, kernel.mma_n)
    )
    dependent_k_steps = max(1, extended._ceil_div(effective_warp_k, kernel.mma_k))
    smem_bytes_per_warp_group = float(
        (effective_warp_m + effective_warp_n) * effective_warp_k * dtype_bytes
    )
    executed_k_groups = (
        max(0, grid.k_stages - 1) * groups_k + last_stage_groups_k
    )
    per_sm_compute = peak_compute_flops_per_cycle / total_sms
    per_smsp_compute = per_sm_compute / _SMSPS_PER_SM
    per_sm_smem = peak_smem_bw_per_cycle / total_sms
    scalar_fallback = (kernel.mma_m, kernel.mma_n, kernel.mma_k) == (1, 1, 1)

    class_results: list[OccupancyClassResult] = []
    issued_flops = 0.0
    smem_read_bytes = 0.0
    for occupancy_class in classes:
        warps_on_sm = occupancy_class.ctas_per_sm * kernel.warps_per_cta
        warps_by_smsp = _warps_by_smsp(warps_on_sm)
        chains_by_smsp = tuple(warps * accumulator_chains for warps in warps_by_smsp)
        effective_smsps: list[float] = []
        for warps, chains in zip(warps_by_smsp, chains_by_smsp):
            if warps == 0:
                effective_smsps.append(0.0)
            elif scalar_fallback:
                effective_smsps.append(per_smsp_compute)
            else:
                effective_smsps.append(
                    _nonzero_effective_rate(
                        per_smsp_compute,
                        chains * mma_flops,
                        tensor_latency_cycles,
                    )
                )
        tc_effective_per_sm = sum(effective_smsps)
        tc_group_flops = (
            warps_on_sm * accumulator_chains * dependent_k_steps * mma_flops
        )
        tc_cycles = _service_from_effective_rate(
            executed_k_groups * tc_group_flops, tc_effective_per_sm
        )
        smem_group_bytes = warps_on_sm * smem_bytes_per_warp_group
        smem_effective_per_sm = _nonzero_effective_rate(
            per_sm_smem, smem_group_bytes, shared_latency_cycles
        )
        smem_cycles = _service_from_effective_rate(
            executed_k_groups * smem_group_bytes, smem_effective_per_sm
        )
        issued_flops += (
            occupancy_class.sm_count * executed_k_groups * tc_group_flops
        )
        smem_read_bytes += (
            occupancy_class.sm_count * executed_k_groups * smem_group_bytes
        )
        class_results.append(
            OccupancyClassResult(
                name=occupancy_class.name,
                sm_count=occupancy_class.sm_count,
                ctas_per_sm=occupancy_class.ctas_per_sm,
                warps_per_sm=warps_on_sm,
                warps_by_smsp=warps_by_smsp,
                accumulator_chains_by_smsp=chains_by_smsp,  # type: ignore[arg-type]
                tensor_raw_flops_per_cycle_per_sm=per_sm_compute,
                tensor_effective_flops_per_cycle_per_sm=tc_effective_per_sm,
                tensor_group_flops_per_sm=tc_group_flops,
                tensor_cycles=tc_cycles,
                smem_raw_bytes_per_cycle_per_sm=per_sm_smem,
                smem_effective_bytes_per_cycle_per_sm=smem_effective_per_sm,
                smem_group_bytes_per_sm=smem_group_bytes,
                smem_cycles=smem_cycles,
                local_body_cycles=max(tc_cycles, smem_cycles),
            )
        )

    tc_body_cycles = max(item.tensor_cycles for item in class_results)
    smem_body_cycles = max(item.smem_cycles for item in class_results)
    local_body_cycles = max(tc_body_cycles, smem_body_cycles)
    active_sms = sum(item.sm_count for item in classes)
    tc_raw = per_sm_compute * active_sms
    smem_raw = per_sm_smem * active_sms
    tc_effective = issued_flops / tc_body_cycles if tc_body_cycles else 0.0
    smem_effective = smem_read_bytes / smem_body_cycles if smem_body_cycles else 0.0
    _validate_effective_rate("tensor", tc_effective, tc_raw)
    _validate_effective_rate("SMEM", smem_effective, smem_raw)

    wave_fraction = active_ctas / max(1, grid.cta_count)
    useful_flops = wave_fraction * (
        2.0 * problem.batch * problem.m * problem.n * problem.k
    )
    if useful_flops > issued_flops + max(1.0e-6, issued_flops * 1.0e-12):
        raise ValueError(
            "wave useful FLOPs exceed issued FLOPs; kernel warp geometry does not "
            "cover the CTA tile"
        )
    tile_efficiency = useful_flops / issued_flops if issued_flops else 0.0

    total_stage_equivalents = max(0, grid.k_stages - 1) + (
        last_stage_k / max(1, kernel.cta_k)
    )
    l2_windows = _memory_window_summary(
        total_stage_equivalents=total_stage_equivalents,
        pipeline_stages=kernel.pipeline_stages,
        active_ctas=active_ctas,
        bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        peak_bytes_per_cycle=peak_l2_bw_per_cycle,
        latency_cycles=l2_latency_cycles,
    )
    dram_windows = _memory_window_summary(
        total_stage_equivalents=total_stage_equivalents,
        pipeline_stages=kernel.pipeline_stages,
        active_ctas=active_ctas,
        bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_bytes_per_cycle=peak_hbm_bw_per_cycle,
        latency_cycles=dram_latency_cycles,
    )
    if l2_windows.window_count != dram_windows.window_count:
        raise AssertionError("L2 and HBM memory-window grouping diverged")
    prologue_cycles = max(
        l2_windows.first_window_cycles, dram_windows.first_window_cycles
    )
    remaining_memory_cycles = max(
        l2_windows.remaining_cycles, dram_windows.remaining_cycles
    )
    body_cycles = max(local_body_cycles, remaining_memory_cycles)
    epilogue = _roofline_epilogue(
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        classes=classes,
        active_ctas=active_ctas,
        per_sm_smem_bw=per_sm_smem,
        shared_latency_cycles=shared_latency_cycles,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        l2_latency_cycles=l2_latency_cycles,
        dram_latency_cycles=dram_latency_cycles,
        effective_warp_m=effective_warp_m,
        effective_warp_n=effective_warp_n,
    )
    total_cycles = prologue_cycles + body_cycles + epilogue.total_cycles
    body_resources = _tied_max_resources(
        {
            "tensor": tc_body_cycles,
            "smem": smem_body_cycles,
            "l2": l2_windows.remaining_cycles,
            "hbm": dram_windows.remaining_cycles,
        },
        _LIMITING_RESOURCE_ORDER,
    )
    boundary = _boundary_phase_dominance(
        prologue_cycles, body_cycles, epilogue.total_cycles
    )
    limiting_resources = _phase_limiting_resources(
        prologue_cycles=prologue_cycles,
        body_cycles=body_cycles,
        epilogue_cycles=epilogue.total_cycles,
        body_resources=body_resources,
    )
    limiting_resource = limiting_resources[0] if limiting_resources else "none"
    pure_cycles = max(
        tc_body_cycles,
        smem_body_cycles,
        l2_windows.total_cycles,
        dram_windows.total_cycles,
    )
    return WaveRooflineResult(
        active_ctas=active_ctas,
        active_sms=active_sms,
        busy_sms=busy_sms,
        lazy_sms=lazy_sms,
        busy_ctas_per_sm=busy_ctas_per_sm,
        lazy_ctas_per_sm=lazy_ctas_per_sm,
        useful_flops=useful_flops,
        issued_flops=issued_flops,
        smem_read_bytes=smem_read_bytes,
        l2_main_bytes=l2_windows.total_bytes,
        dram_main_bytes=dram_windows.total_bytes,
        tc_raw_flops_per_cycle=tc_raw,
        tc_effective_flops_per_cycle=tc_effective,
        smem_raw_bytes_per_cycle=smem_raw,
        smem_effective_bytes_per_cycle=smem_effective,
        l2_raw_bytes_per_cycle=peak_l2_bw_per_cycle,
        l2_effective_bytes_per_cycle=l2_windows.effective_bytes_per_cycle,
        dram_raw_bytes_per_cycle=peak_hbm_bw_per_cycle,
        dram_effective_bytes_per_cycle=dram_windows.effective_bytes_per_cycle,
        tc_concurrency_factor=tc_effective / tc_raw if tc_raw else 0.0,
        smem_concurrency_factor=smem_effective / smem_raw if smem_raw else 0.0,
        l2_concurrency_factor=l2_windows.concurrency_factor,
        dram_concurrency_factor=dram_windows.concurrency_factor,
        tc_body_cycles=tc_body_cycles,
        smem_body_cycles=smem_body_cycles,
        l2_total_cycles=l2_windows.total_cycles,
        dram_total_cycles=dram_windows.total_cycles,
        l2_remaining_cycles=l2_windows.remaining_cycles,
        dram_remaining_cycles=dram_windows.remaining_cycles,
        prologue_cycles=prologue_cycles,
        body_cycles=body_cycles,
        epilogue_cycles=epilogue.total_cycles,
        total_cycles=total_cycles,
        tile_efficiency=tile_efficiency,
        limiting_resource=limiting_resource,
        limiting_resources=limiting_resources,
        body_limiting_resources=body_resources,
        boundary_phase_dominance=boundary,
        pure_mainloop_roofline_cycles=pure_cycles,
        predicted_useful_flops_per_cycle=(
            useful_flops / total_cycles if total_cycles else 0.0
        ),
        accumulator_chains_per_warp=accumulator_chains,
        dependent_k_steps_per_chain=dependent_k_steps,
        k_groups=executed_k_groups,
        memory_window_count=l2_windows.window_count,
        occupancy_classes=tuple(class_results),
        l2_windows=l2_windows,
        dram_windows=dram_windows,
        epilogue=epilogue,
    )


def _validate_wave_geometry(
    *,
    kernel: extended.GemmKernelSpec,
    grid: extended.GridAccounting,
    groups_k: int,
    last_stage_groups_k: int,
    last_stage_k: int,
    memory_pipeline_groups: int,
    last_memory_pipeline_stages: int,
) -> None:
    expected_groups = extended._k_groups(kernel.cta_k, kernel)
    expected_last_groups = extended._k_groups(last_stage_k, kernel)
    expected_memory_groups = (
        extended._ceil_div(grid.k_stages, kernel.pipeline_stages)
        if grid.k_stages
        else 0
    )
    expected_last_memory_stages = (
        (grid.k_stages - 1) % kernel.pipeline_stages + 1 if grid.k_stages else 0
    )
    if groups_k != expected_groups:
        raise ValueError(
            f"groups_k={groups_k} does not match kernel geometry ({expected_groups})"
        )
    if last_stage_groups_k != expected_last_groups:
        raise ValueError(
            "last_stage_groups_k does not match the final K-stage geometry: "
            f"expected {expected_last_groups}, got {last_stage_groups_k}"
        )
    if memory_pipeline_groups != expected_memory_groups:
        raise ValueError(
            "memory_pipeline_groups does not cover the derived K-stage grid: "
            f"expected {expected_memory_groups}, got {memory_pipeline_groups}"
        )
    if last_memory_pipeline_stages != expected_last_memory_stages:
        raise ValueError(
            "last_memory_pipeline_stages does not match the derived K-stage grid: "
            f"expected {expected_last_memory_stages}, got {last_memory_pipeline_stages}"
        )
    if grid.k_stages and not 0 < last_stage_k <= kernel.cta_k:
        raise ValueError("last_stage_k must be within the CTA K tile")


def _roofline_epilogue(
    *,
    problem: extended.GemmProblemSpec,
    kernel: extended.GemmKernelSpec,
    grid: extended.GridAccounting,
    traffic: extended.TrafficAccounting,
    classes: tuple[SMOccupancyClass, ...],
    active_ctas: int,
    per_sm_smem_bw: float,
    shared_latency_cycles: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
    l2_latency_cycles: float,
    dram_latency_cycles: float,
    effective_warp_m: int,
    effective_warp_n: int,
) -> EpilogueRooflineResult:
    wave_fraction = active_ctas / max(1, grid.cta_count)
    l2_bytes = 0.0
    dram_bytes = wave_fraction * (
        traffic.d_store_transaction_bytes + traffic.c_read_transaction_bytes
    )
    output_bytes = dtype_nbytes(problem.output_dtype)
    bytes_per_warp = float(effective_warp_m * effective_warp_n * output_bytes)
    class_smem_cycles: list[float] = []
    class_slice_cycles: list[float] = []
    for occupancy_class in classes:
        warps = occupancy_class.ctas_per_sm * kernel.warps_per_cta
        class_bytes = warps * bytes_per_warp
        effective = _nonzero_effective_rate(
            per_sm_smem_bw, class_bytes, shared_latency_cycles
        )
        base_cycles = _service_from_effective_rate(class_bytes, effective)
        class_smem_cycles.append(base_cycles)
        if kernel.slice_k:
            store_bytes = class_bytes / max(1, kernel.num_warp_tile_k)
            store_effective = _nonzero_effective_rate(
                per_sm_smem_bw, store_bytes, shared_latency_cycles
            )
            store_cycles = _service_from_effective_rate(store_bytes, store_effective)
            class_slice_cycles.append(base_cycles + store_cycles)
    smem_cycles = max(class_smem_cycles, default=0.0)
    slice_cycles = max(class_slice_cycles, default=0.0)
    smem_bytes = wave_fraction * traffic.d_store_transaction_bytes
    if kernel.slice_k:
        smem_bytes += wave_fraction * traffic.d_store_transaction_bytes * (
            1.0 + 1.0 / max(1, kernel.num_warp_tile_k)
        )
    l2_rate = _nonzero_effective_rate(
        peak_l2_bw_per_cycle, l2_bytes, l2_latency_cycles
    )
    dram_rate = _nonzero_effective_rate(
        peak_hbm_bw_per_cycle, dram_bytes, dram_latency_cycles
    )
    l2_cycles = _service_from_effective_rate(l2_bytes, l2_rate)
    dram_cycles = _service_from_effective_rate(dram_bytes, dram_rate)
    global_cycles = l2_cycles + dram_cycles
    total = smem_cycles + slice_cycles + global_cycles
    return EpilogueRooflineResult(
        smem_bytes=smem_bytes,
        l2_bytes=l2_bytes,
        dram_bytes=dram_bytes,
        smem_cycles=smem_cycles,
        slice_k_extra_cycles=slice_cycles,
        l2_cycles=l2_cycles,
        dram_cycles=dram_cycles,
        global_cycles=global_cycles,
        total_cycles=total,
    )


def _validate_effective_rate(name: str, effective: float, raw: float) -> None:
    if effective < -1.0e-12 or effective > raw + max(1.0e-9, raw * 1.0e-12):
        raise AssertionError(f"{name} effective rate exceeds its raw peak")


def _tied_max_resources(
    values: Mapping[str, float], order: tuple[str, ...]
) -> tuple[str, ...]:
    maximum = max(values.values(), default=0.0)
    tolerance = max(1.0e-9, abs(maximum) * 1.0e-9)
    return tuple(name for name in order if abs(values.get(name, 0.0) - maximum) <= tolerance)


def _boundary_phase_dominance(
    prologue_cycles: float, body_cycles: float, epilogue_cycles: float
) -> str | None:
    maximum = max(prologue_cycles, body_cycles, epilogue_cycles)
    tolerance = max(1.0e-9, abs(maximum) * 1.0e-9)
    if prologue_cycles >= body_cycles - tolerance and prologue_cycles >= epilogue_cycles - tolerance:
        return "prologue"
    if epilogue_cycles >= body_cycles - tolerance:
        return "epilogue"
    return None


def _phase_limiting_resources(
    *,
    prologue_cycles: float,
    body_cycles: float,
    epilogue_cycles: float,
    body_resources: tuple[str, ...],
) -> tuple[str, ...]:
    maximum = max(prologue_cycles, body_cycles, epilogue_cycles)
    tolerance = max(1.0e-9, abs(maximum) * 1.0e-9)
    result: list[str] = []
    # Boundary phases are primary whenever they are largest; prologue is stable on a
    # prologue/epilogue tie. Resource ties retain tensor/SMEM/L2/HBM order.
    if prologue_cycles >= maximum - tolerance:
        result.append("prologue")
    if epilogue_cycles >= maximum - tolerance:
        result.append("epilogue")
    if body_cycles >= maximum - tolerance:
        result.extend(body_resources)
    return tuple(result)


def _zero_wave_roofline(represented_sms: int = 0) -> WaveRooflineResult:
    memory = _memory_window_summary(
        total_stage_equivalents=0.0,
        pipeline_stages=1,
        active_ctas=0,
        bytes_per_cta_stage=0.0,
        peak_bytes_per_cycle=0.0,
        latency_cycles=0.0,
    )
    epilogue = EpilogueRooflineResult(*(0.0 for _ in range(9)))
    return WaveRooflineResult(
        active_ctas=0,
        active_sms=0,
        busy_sms=0,
        lazy_sms=max(0, represented_sms),
        busy_ctas_per_sm=0,
        lazy_ctas_per_sm=0,
        useful_flops=0.0,
        issued_flops=0.0,
        smem_read_bytes=0.0,
        l2_main_bytes=0.0,
        dram_main_bytes=0.0,
        tc_raw_flops_per_cycle=0.0,
        tc_effective_flops_per_cycle=0.0,
        smem_raw_bytes_per_cycle=0.0,
        smem_effective_bytes_per_cycle=0.0,
        l2_raw_bytes_per_cycle=0.0,
        l2_effective_bytes_per_cycle=0.0,
        dram_raw_bytes_per_cycle=0.0,
        dram_effective_bytes_per_cycle=0.0,
        tc_concurrency_factor=0.0,
        smem_concurrency_factor=0.0,
        l2_concurrency_factor=0.0,
        dram_concurrency_factor=0.0,
        tc_body_cycles=0.0,
        smem_body_cycles=0.0,
        l2_total_cycles=0.0,
        dram_total_cycles=0.0,
        l2_remaining_cycles=0.0,
        dram_remaining_cycles=0.0,
        prologue_cycles=0.0,
        body_cycles=0.0,
        epilogue_cycles=0.0,
        total_cycles=0.0,
        tile_efficiency=0.0,
        limiting_resource="none",
        limiting_resources=(),
        body_limiting_resources=(),
        boundary_phase_dominance=None,
        pure_mainloop_roofline_cycles=0.0,
        predicted_useful_flops_per_cycle=0.0,
        accumulator_chains_per_warp=0,
        dependent_k_steps_per_chain=0,
        k_groups=0,
        memory_window_count=0,
        occupancy_classes=(),
        l2_windows=memory,
        dram_windows=memory,
        epilogue=epilogue,
    )


def _aggregate_bottlenecks(
    timeline: EffectiveTimelineResult, full_wave_count: int
) -> tuple[str, tuple[str, ...]]:
    resources = {
        "tensor": full_wave_count * timeline.full_wave.tc_body_cycles
        + timeline.last_wave.tc_body_cycles,
        "smem": full_wave_count * timeline.full_wave.smem_body_cycles
        + timeline.last_wave.smem_body_cycles,
        "l2": full_wave_count * timeline.full_wave.l2_remaining_cycles
        + timeline.last_wave.l2_remaining_cycles,
        "hbm": full_wave_count * timeline.full_wave.dram_remaining_cycles
        + timeline.last_wave.dram_remaining_cycles,
    }
    body_resources = _tied_max_resources(resources, _LIMITING_RESOURCE_ORDER)
    tied = _phase_limiting_resources(
        prologue_cycles=timeline.prologue_cycles,
        body_cycles=timeline.body_cycles,
        epilogue_cycles=timeline.epilogue_cycles,
        body_resources=body_resources,
    )
    return (tied[0], tied[1:]) if tied else ("none", ())


def _wave_diagnostics(wave: WaveRooflineResult) -> dict[str, Any]:
    data = asdict(wave)
    data["raw_rates"] = {
        "tensor_flops_per_cycle": wave.tc_raw_flops_per_cycle,
        "smem_bytes_per_cycle": wave.smem_raw_bytes_per_cycle,
        "l2_bytes_per_cycle": wave.l2_raw_bytes_per_cycle,
        "hbm_bytes_per_cycle": wave.dram_raw_bytes_per_cycle,
    }
    data["effective_rates"] = {
        "tensor_flops_per_cycle": wave.tc_effective_flops_per_cycle,
        "smem_bytes_per_cycle": wave.smem_effective_bytes_per_cycle,
        "l2_bytes_per_cycle": wave.l2_effective_bytes_per_cycle,
        "hbm_bytes_per_cycle": wave.dram_effective_bytes_per_cycle,
    }
    data["concurrency_factors"] = {
        "tensor": wave.tc_concurrency_factor,
        "smem": wave.smem_concurrency_factor,
        "l2": wave.l2_concurrency_factor,
        "hbm": wave.dram_concurrency_factor,
    }
    data["service_cycles"] = {
        "tensor": wave.tc_body_cycles,
        "smem": wave.smem_body_cycles,
        "l2": wave.l2_total_cycles,
        "hbm": wave.dram_total_cycles,
    }
    data["phase_cycles"] = {
        "prologue": wave.prologue_cycles,
        "body": wave.body_cycles,
        "epilogue": wave.epilogue_cycles,
        "total": wave.total_cycles,
        "pure_roofline": wave.pure_mainloop_roofline_cycles,
    }
    return data


def _diagnostics(
    *,
    problem: extended.GemmProblemSpec,
    kernel: extended.GemmKernelSpec,
    grid: extended.GridAccounting,
    traffic: extended.TrafficAccounting,
    occupancy: extended.OccupancyResult,
    timeline: EffectiveTimelineResult,
    warnings: tuple[str, ...],
    clock_hz: float,
    latency_s: float,
    flops_per_cycle: float,
    tflops_per_s: float,
    hardware: HardwareSpec,
    fixed_overhead_cycles: float,
) -> dict[str, Any]:
    full_count = max(0, occupancy.wave_count - 1)
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    primary, secondary = _aggregate_bottlenecks(timeline, full_count)
    full_diag = _wave_diagnostics(timeline.full_wave)
    last_diag = _wave_diagnostics(timeline.last_wave)
    q_smem = traffic.smem_total_bytes + timeline.epilogue_smem_bytes
    q_l2 = traffic.l2_requested_bytes
    q_dram = traffic.dram_unique_bytes
    useful = grid.useful_flops
    issued = (
        full_count * timeline.full_wave.issued_flops + timeline.last_wave.issued_flops
    )
    peak_compute = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_smem = extended._smem_bandwidth_per_cycle(hardware, clock_hz, [])
    peak_l2 = (
        hardware.memory_levels["l2"].bandwidth_bytes_per_s / clock_hz
        if "l2" in hardware.memory_levels
        else 0.0
    )
    peak_dram = hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz
    effective_rates = {
        "tensor_flops_per_cycle": (
            issued
            / (full_count * timeline.full_wave.tc_body_cycles + timeline.last_wave.tc_body_cycles)
            if issued
            and full_count * timeline.full_wave.tc_body_cycles
            + timeline.last_wave.tc_body_cycles
            else 0.0
        ),
        "smem_bytes_per_cycle": (
            full_count * timeline.full_wave.smem_read_bytes
            + timeline.last_wave.smem_read_bytes
        )
        / max(
            full_count * timeline.full_wave.smem_body_cycles
            + timeline.last_wave.smem_body_cycles,
            1.0e-12,
        ),
        "l2_bytes_per_cycle": (
            full_count * timeline.full_wave.l2_main_bytes
            + timeline.last_wave.l2_main_bytes
        )
        / max(
            full_count * timeline.full_wave.l2_total_cycles
            + timeline.last_wave.l2_total_cycles,
            1.0e-12,
        ),
        "hbm_bytes_per_cycle": (
            full_count * timeline.full_wave.dram_main_bytes
            + timeline.last_wave.dram_main_bytes
        )
        / max(
            full_count * timeline.full_wave.dram_total_cycles
            + timeline.last_wave.dram_total_cycles,
            1.0e-12,
        ),
    }
    raw_rates = {
        "tensor_flops_per_cycle": peak_compute,
        "smem_bytes_per_cycle": peak_smem,
        "l2_bytes_per_cycle": peak_l2,
        "hbm_bytes_per_cycle": peak_dram,
    }
    concurrency_factors = {
        name.split("_")[0]: effective_rates[name] / raw if raw else 0.0
        for name, raw in raw_rates.items()
    }
    fixed_fraction = fixed_overhead_cycles / max(total_device_cycles, 1.0e-12)
    shared_memory_bytes_per_cta = extended._shared_memory_bytes_per_cta(
        problem, kernel, grid
    )
    phase_total = max(timeline.kernel_cycles, 1.0e-12)
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
            "shared_memory_bytes_per_cta": shared_memory_bytes_per_cta,
            "max_concurrent_ctas_per_sm": kernel.max_concurrent_ctas_per_sm,
            "slice_k": kernel.slice_k,
        },
        "predicted_elapsed_cycles": total_device_cycles,
        "modeled_device_cycles": timeline.kernel_cycles,
        "device_fixed_overhead_cycles": fixed_overhead_cycles,
        "total_device_cycles": total_device_cycles,
        "device_fixed_overhead_s": fixed_overhead_cycles / clock_hz,
        "device_fixed_overhead_fraction": fixed_fraction,
        "predicted_flop_per_cycle": flops_per_cycle,
        "modeled_flop_per_cycle": useful / timeline.kernel_cycles if timeline.kernel_cycles else 0.0,
        "predicted_tflops_per_s": tflops_per_s,
        "predicted_latency_s": latency_s,
        "clock_hz": clock_hz,
        "cta_count": grid.cta_count,
        "cta_grid": {"m": grid.blocks_m, "n": grid.blocks_n, "k_stages": grid.k_stages},
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
        "useful_flops": useful,
        "issued_flops": issued,
        "grid_issued_flops": grid.issued_flops,
        "tile_efficiency": useful / issued if issued else 0.0,
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
        "wave_roofline": {"full": full_diag, "last": last_diag},
        # The calibrated energy feature extractor consumes only total_cycles here.
        "wave_pipeline": {
            "full": {"total_cycles": timeline.full_wave.total_cycles},
            "last": {"total_cycles": timeline.last_wave.total_cycles},
        },
        "pipeline_components": {
            "groups_k": timeline.groups_k,
            "last_stage_groups_k": timeline.last_stage_groups_k,
            "last_stage_k": timeline.last_stage_k,
            "memory_pipeline_groups": timeline.memory_pipeline_groups,
            "last_memory_pipeline_stages": timeline.last_memory_pipeline_stages,
        },
        "effective_rates": effective_rates,
        "raw_rates": raw_rates,
        "concurrency_factors": concurrency_factors,
        "active_cycles": {
            "compute": timeline.compute_active_cycles,
            "smem": timeline.smem_active_cycles,
            "l2": timeline.l2_active_cycles,
            "dram": timeline.dram_active_cycles,
        },
        "stage_cycles": {
            "compute": full_diag["service_cycles"]["tensor"],
            "smem": full_diag["service_cycles"]["smem"],
            "l2_service": full_diag["service_cycles"]["l2"],
            "dram_service": full_diag["service_cycles"]["hbm"],
            "prologue": timeline.prologue_cycles,
            "work": timeline.body_cycles,
            "epilogue": timeline.epilogue_cycles,
        },
        "phase_cycles": {
            "prologue": timeline.prologue_cycles,
            "body": timeline.body_cycles,
            "epilogue": timeline.epilogue_cycles,
        },
        "phase_fractions": {
            "prologue": timeline.prologue_cycles / phase_total,
            "body": timeline.body_cycles / phase_total,
            "epilogue": timeline.epilogue_cycles / phase_total,
        },
        "operational_intensity": {
            "smem_flop_per_byte": useful / q_smem if q_smem else None,
            "l2_flop_per_byte": useful / q_l2 if q_l2 else None,
            "dram_flop_per_byte": useful / q_dram if q_dram else None,
        },
        "roofline_raw_bounds_flop_per_cycle": {
            "compute": peak_compute,
            "smem": peak_smem * useful / q_smem if q_smem else None,
            "l2": peak_l2 * useful / q_l2 if q_l2 and peak_l2 else None,
            "dram": peak_dram * useful / q_dram if q_dram else None,
        },
        "roofline_achieved_flop_per_cycle": {
            "kernel_scope": useful / timeline.kernel_cycles if timeline.kernel_cycles else 0.0,
            "device_scope": useful / total_device_cycles if total_device_cycles else 0.0,
        },
        "primary_bottleneck": primary,
        "secondary_bottlenecks": secondary,
        "limiting_resources": (primary,) + secondary if primary != "none" else (),
        "memory_level_latencies_s": {
            "hbm": hardware.memory_levels["hbm"].latency_s,
            "l2": hardware.memory_levels["l2"].latency_s if "l2" in hardware.memory_levels else None,
            "sram": hardware.memory_levels["sram"].latency_s if "sram" in hardware.memory_levels else None,
            "register": None,
        },
        "warnings": warnings,
        "assumptions": (
            "concurrency_derived_effective_ceilings",
            "phase_aware_roofline",
            "four_smsps_per_sm_assumed",
            "l2_hbm_mainloop_overlap",
            "serial_epilogue_l2_plus_hbm",
        ),
    }


def _implementation_name(batched: bool) -> str:
    return "effective_roofline.batched_gemm" if batched else "effective_roofline.gemm"


__all__ = [
    "EffectiveRooflineModel",
    "EffectiveGemmEstimator",
    "WaveRooflineResult",
    "OccupancyClassResult",
    "MemoryWindowResult",
    "EpilogueRooflineResult",
    "EffectiveTimelineResult",
    "evaluate_gemm_template_candidates",
    "select_gemm_template_candidate",
]
