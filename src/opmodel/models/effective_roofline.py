from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Mapping

from opmodel.api import (
    DType,
    EngineKind,
    LocalOp,
    MemoryAccess,
    OpKind,
    OpProfile,
    TensorRole,
)
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
from opmodel.ops import (
    dtype_nbytes,
    footprint_from_tensors,
    get_tensors,
    numel,
    parse_attention,
    require_one_tensor,
    tensor_nbytes,
)


_SMSPS_PER_SM = 4
_LIMITING_RESOURCE_ORDER = ("tensor", "smem", "l2", "hbm")


@dataclass(frozen=True)
class SMOccupancyClass:
    name: str
    sm_count: int
    ctas_per_sm: int


@dataclass(frozen=True)
class _WarpTileWork:
    accumulator_chains: int
    smem_bytes_per_k_group: float


@dataclass(frozen=True)
class LocalMatmulGeometry:
    cta_m: int
    cta_n: int
    reduction_k: int
    live_m: int
    live_n: int
    live_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    warps_per_cta: int
    lhs_smem_resident: bool
    rhs_smem_resident: bool
    input_bytes: float


@dataclass(frozen=True)
class LocalMiniGemmClassResult:
    name: str
    sm_count: int
    ctas_per_sm: int
    warps_per_sm: int
    warps_by_smsp: tuple[int, int, int, int]
    accumulator_chains_by_smsp: tuple[int, int, int, int]
    tensor_raw_flops_per_cycle_per_sm: float
    tensor_effective_flops_per_cycle_per_sm: float
    tensor_cycles: float
    smem_raw_bytes_per_cycle_per_sm: float
    smem_effective_bytes_per_cycle_per_sm: float
    smem_inflight_bytes_per_sm: float
    smem_cycles: float
    total_cycles: float


@dataclass(frozen=True)
class LocalMiniGemmResult:
    useful_flops: float
    issued_flops: float
    smem_read_bytes: float
    tensor_cycles: float
    smem_cycles: float
    total_cycles: float
    accumulator_chains_per_warp: int
    dependent_k_steps_per_chain: int
    reduction_groups: int
    occupancy_classes: tuple[LocalMiniGemmClassResult, ...]


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
    smem_first_group_cycles: float
    smem_steady_remaining_cycles: float
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
    latency_bearing_window_count: int
    steady_state_bytes_per_cycle: float


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
    steady_local_body_cycles: float
    finite_local_body_cycles: float
    local_fill_drain_cycles: float
    final_local_window_cycles: float
    memory_drain_path_cycles: float
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


@dataclass(frozen=True)
class FlashAttentionProblemSpec:
    batch: int
    query_heads: int
    kv_heads: int
    seq_q: int
    seq_kv: int
    head_dim: int
    value_dim: int
    input_dtype: DType
    output_dtype: DType
    causal: bool
    causal_alignment: str
    store_lse: bool


@dataclass(frozen=True)
class AmpereFlashAttentionKernelSpec:
    block_q: int
    block_k: int
    warps_per_cta: int
    threads_per_cta: int
    qk_warp_m: int
    qk_warp_n: int
    qk_warp_k: int
    pv_warp_m: int
    pv_warp_n: int
    pv_warp_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    pipeline_stages: int
    registers_per_thread: int
    shared_memory_bytes_per_cta: int
    max_concurrent_ctas_per_sm: int | None
    q_register_resident: bool
    probability_register_resident: bool
    fp32_accumulators: bool
    epilogue_through_smem: bool
    rows_per_softmax_group: int
    warps_per_softmax_group: int
    predicate_masked_exp: bool


@dataclass(frozen=True)
class FlashAttentionKVTile:
    tile_index: int
    kv_start: int
    live_kv_rows: int
    issued_kv_rows: int
    is_causal_diagonal: bool
    useful_score_elements: int
    issued_score_elements: int


@dataclass(frozen=True)
class FlashAttentionCTADescriptor:
    linear_cta: int
    batch_index: int
    query_head: int
    kv_head: int
    query_block: int
    query_start: int
    live_query_rows: int
    kv_tiles: tuple[FlashAttentionKVTile, ...]


@dataclass(frozen=True)
class FlashAttentionWave:
    index: int
    ctas: tuple[FlashAttentionCTADescriptor, ...]
    ctas_by_sm: tuple[tuple[FlashAttentionCTADescriptor, ...], ...]


class FlashKVReusePolicy(str, Enum):
    NONE = "none"
    IDEAL_WITHIN_WAVE = "ideal_within_wave"
    IDEAL_WITHIN_SWIZZLE_GROUP = "ideal_within_swizzle_group"


class SoftmaxMode(str, Enum):
    ONLINE_ATTENTION = "online_attention"
    COMPLETE_ROW = "complete_row"


@dataclass(frozen=True)
class CudaSoftmaxWorkSpec:
    mode: SoftmaxMode
    rows: int
    reduction_elements: int
    issued_elements: int
    valid_elements: int
    masked_elements: int
    output_rescale_elements: int
    has_prior_online_state: bool
    apply_scale: bool
    apply_mask: bool
    predicate_masked_exp: bool


@dataclass(frozen=True)
class CudaSoftmaxKernelSpec:
    threads_per_row: int
    rows_per_cta: int
    warps_per_cta: int
    warps_per_reduction_group: int
    vector_width: int
    cross_warp_reduction: bool
    stage_values_in_smem: bool


@dataclass(frozen=True)
class CudaSoftmaxComputeResult:
    issued_elements: int
    valid_elements: int
    cuda_scalar_ops: float
    exp_ops: float
    shuffle_ops: float
    shared_reduction_bytes: float
    barrier_count: int
    cuda_raw_ops_per_cycle: float
    cuda_effective_ops_per_cycle: float
    sfu_raw_ops_per_cycle: float
    sfu_effective_ops_per_cycle: float
    scale_mask_cycles: float
    rowmax_cycles: float
    exp_sum_cycles: float
    state_or_normalize_cycles: float
    output_rescale_cycles: float
    total_cycles: float


@dataclass(frozen=True)
class FlashSoftmaxStageResult:
    compute: CudaSoftmaxComputeResult
    causal_diagonal: bool
    online_state_rows: int
    output_rescale_elements: int


@dataclass(frozen=True)
class FlashMemoryActionResult:
    name: str
    active_ctas: int
    total_bytes: float
    inflight_bytes: float
    l2_raw_bytes_per_cycle: float
    l2_effective_bytes_per_cycle: float
    hbm_raw_bytes_per_cycle: float
    hbm_effective_bytes_per_cycle: float
    l2_cycles: float
    hbm_cycles: float
    total_cycles: float


@dataclass(frozen=True)
class FlashAttentionIterationResult:
    index: int
    active_ctas: int
    qk: LocalMiniGemmResult
    softmax: FlashSoftmaxStageResult
    pv: LocalMiniGemmResult
    k_load: FlashMemoryActionResult
    v_load: FlashMemoryActionResult
    next_k_load: FlashMemoryActionResult | None
    qk_softmax_or_v_cycles: float
    pv_or_next_k_cycles: float
    total_cycles: float


@dataclass(frozen=True)
class FlashAttentionWaveResult:
    index: int
    active_ctas: int
    active_ctas_by_iteration: tuple[int, ...]
    prologue_cycles: float
    body_cycles: float
    epilogue_cycles: float
    total_cycles: float
    iterations: tuple[FlashAttentionIterationResult, ...]


@dataclass(frozen=True)
class FlashAttentionTimelineResult:
    kernel_cycles: float
    prologue_cycles: float
    body_cycles: float
    epilogue_cycles: float
    waves: tuple[FlashAttentionWaveResult, ...]


@dataclass(frozen=True)
class StandaloneSoftmaxProblemSpec:
    row_count: int
    reduction_elements: int
    input_dtype: DType
    output_dtype: DType


@dataclass(frozen=True)
class StandaloneSoftmaxKernelSpec:
    threads_per_cta: int
    rows_per_cta: int
    warps_per_cta: int
    vector_width: int
    stage_values_in_smem: bool
    shared_memory_bytes_per_cta: int
    registers_per_thread: int
    max_concurrent_ctas_per_sm: int | None


@dataclass(frozen=True)
class StandaloneSoftmaxWaveResult:
    active_ctas: int
    active_rows: int
    load_cycles: float
    compute_cycles: float
    store_cycles: float
    total_cycles: float
    compute: CudaSoftmaxComputeResult


@dataclass(frozen=True)
class StandaloneSoftmaxTimelineResult:
    kernel_cycles: float
    full_wave: StandaloneSoftmaxWaveResult
    last_wave: StandaloneSoftmaxWaveResult
    wave_count: int


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
    latency_rate = (
        math.inf if latency_cycles == 0.0 else in_flight_work / latency_cycles
    )
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


def _maximum_cta_warp_work(
    *,
    problem: extended.GemmProblemSpec,
    kernel: extended.GemmKernelSpec,
    effective_warp_k: int,
    dtype_bytes: float,
) -> tuple[_WarpTileWork, ...]:
    """Return work-generating fragments for the largest logical CTA tile.

    A wave completes at the pace of its slowest CTA.  Consequently partial edge
    CTAs do not reduce its local service while at least one full CTA exists, but
    a dimension smaller than the CTA tile reduces every CTA's live warp and MMA
    fragments.  The fragment geometry is derived directly from the kernel; it
    does not depend on a workload class or a dimension cutoff.
    """
    return _local_matmul_warp_work(
        LocalMatmulGeometry(
            cta_m=kernel.cta_m,
            cta_n=kernel.cta_n,
            reduction_k=max(1, effective_warp_k),
            live_m=min(problem.m, kernel.cta_m),
            live_n=min(problem.n, kernel.cta_n),
            live_k=max(1, effective_warp_k),
            warp_m=kernel.warp_m,
            warp_n=kernel.warp_n,
            warp_k=effective_warp_k,
            mma_m=kernel.mma_m,
            mma_n=kernel.mma_n,
            mma_k=kernel.mma_k,
            warps_per_cta=kernel.warps_per_cta,
            lhs_smem_resident=True,
            rhs_smem_resident=True,
            input_bytes=dtype_bytes,
        )
    )


def _local_matmul_warp_work(
    geometry: LocalMatmulGeometry,
) -> tuple[_WarpTileWork, ...]:
    values = (
        geometry.cta_m,
        geometry.cta_n,
        geometry.reduction_k,
        geometry.live_m,
        geometry.live_n,
        geometry.live_k,
        geometry.warp_m,
        geometry.warp_n,
        geometry.warp_k,
        geometry.mma_m,
        geometry.mma_n,
        geometry.mma_k,
        geometry.warps_per_cta,
    )
    if any(value <= 0 for value in values):
        raise ValueError("local matmul dimensions must be positive")
    if geometry.live_m > geometry.cta_m or geometry.live_n > geometry.cta_n:
        raise ValueError("live local matmul tile exceeds its CTA tile")
    effective_warp_m = min(geometry.warp_m, geometry.cta_m)
    effective_warp_n = min(geometry.warp_n, geometry.cta_n)
    warp_rows = extended._ceil_div(geometry.cta_m, effective_warp_m)
    warp_columns = extended._ceil_div(geometry.cta_n, effective_warp_n)
    work: list[_WarpTileWork] = []
    if warp_rows * warp_columns != geometry.warps_per_cta:
        full_chains = (
            extended._ceil_div(effective_warp_m, geometry.mma_m)
            * extended._ceil_div(effective_warp_n, geometry.mma_n)
        )
        full_smem = 0.0
        if geometry.lhs_smem_resident:
            full_smem += effective_warp_m * geometry.warp_k * geometry.input_bytes
        if geometry.rhs_smem_resident:
            full_smem += effective_warp_n * geometry.warp_k * geometry.input_bytes
        return tuple(
            _WarpTileWork(full_chains, full_smem)
            for _ in range(geometry.warps_per_cta)
        )
    for warp_row in range(warp_rows):
        live_m = max(
            0,
            min(effective_warp_m, geometry.live_m - warp_row * effective_warp_m),
        )
        covered_m = (
            extended._ceil_div(live_m, geometry.mma_m) * geometry.mma_m
            if live_m
            else 0
        )
        for warp_column in range(warp_columns):
            live_n = max(
                0,
                min(
                    effective_warp_n,
                    geometry.live_n - warp_column * effective_warp_n,
                ),
            )
            covered_n = (
                extended._ceil_div(live_n, geometry.mma_n) * geometry.mma_n
                if live_n
                else 0
            )
            chains = (
                (covered_m // geometry.mma_m) * (covered_n // geometry.mma_n)
                if covered_m and covered_n
                else 0
            )
            smem = 0.0
            if chains:
                if geometry.lhs_smem_resident:
                    smem += covered_m * geometry.warp_k * geometry.input_bytes
                if geometry.rhs_smem_resident:
                    smem += covered_n * geometry.warp_k * geometry.input_bytes
            work.append(_WarpTileWork(chains, smem))
    if len(work) != geometry.warps_per_cta:
        raise AssertionError("local matmul geometry does not cover physical warps")
    return tuple(work)


def _local_mini_gemm_service(
    *,
    geometry: LocalMatmulGeometry,
    occupancy_classes: tuple[SMOccupancyClass, ...],
    represented_sms: int,
    peak_tensor_flops_per_cycle: float,
    peak_smem_bytes_per_cycle: float,
    tensor_latency_cycles: float,
    shared_latency_cycles: float,
) -> LocalMiniGemmResult:
    """Model one CTA-local MMA sequence without global traffic or launch cost."""
    if represented_sms <= 0 or not occupancy_classes:
        raise ValueError("local matmul service requires represented SMs")
    active_sms = sum(item.sm_count for item in occupancy_classes)
    if active_sms > represented_sms:
        raise ValueError("local matmul occupancy classes exceed represented SMs")
    warp_work = _local_matmul_warp_work(geometry)
    reduction_groups = extended._ceil_div(geometry.live_k, geometry.warp_k)
    dependent_steps = extended._ceil_div(geometry.warp_k, geometry.mma_k)
    mma_flops = float(2 * geometry.mma_m * geometry.mma_n * geometry.mma_k)
    per_sm_tensor = peak_tensor_flops_per_cycle / represented_sms
    per_sm_smem = peak_smem_bytes_per_cycle / represented_sms
    class_results: list[LocalMiniGemmClassResult] = []
    issued_flops = 0.0
    smem_bytes = 0.0
    for occupancy_class in occupancy_classes:
        warps_on_sm = occupancy_class.ctas_per_sm * geometry.warps_per_cta
        warps_by_smsp = _warps_by_smsp(warps_on_sm)
        chains = [0, 0, 0, 0]
        group_smem = 0.0
        for cta_index in range(occupancy_class.ctas_per_sm):
            for warp_index, item in enumerate(warp_work):
                smsp = (cta_index * geometry.warps_per_cta + warp_index) % _SMSPS_PER_SM
                chains[smsp] += item.accumulator_chains
                group_smem += item.smem_bytes_per_k_group
        per_smsp_tensor = per_sm_tensor / _SMSPS_PER_SM
        smsp_cycles: list[float] = []
        for chain_count in chains:
            rate = _nonzero_effective_rate(
                per_smsp_tensor,
                chain_count * mma_flops,
                tensor_latency_cycles,
            )
            work = reduction_groups * chain_count * dependent_steps * mma_flops
            smsp_cycles.append(_service_from_effective_rate(work, rate))
        tensor_cycles = max(smsp_cycles, default=0.0)
        tensor_work = (
            reduction_groups * sum(chains) * dependent_steps * mma_flops
        )
        tensor_effective = tensor_work / tensor_cycles if tensor_cycles else 0.0
        smem_rate = _nonzero_effective_rate(
            per_sm_smem, group_smem, shared_latency_cycles
        )
        class_smem_bytes = reduction_groups * group_smem
        class_smem_cycles = _service_from_effective_rate(class_smem_bytes, smem_rate)
        total = max(tensor_cycles, class_smem_cycles)
        issued_flops += occupancy_class.sm_count * tensor_work
        smem_bytes += occupancy_class.sm_count * class_smem_bytes
        class_results.append(
            LocalMiniGemmClassResult(
                name=occupancy_class.name,
                sm_count=occupancy_class.sm_count,
                ctas_per_sm=occupancy_class.ctas_per_sm,
                warps_per_sm=warps_on_sm,
                warps_by_smsp=warps_by_smsp,
                accumulator_chains_by_smsp=tuple(chains),  # type: ignore[arg-type]
                tensor_raw_flops_per_cycle_per_sm=per_sm_tensor,
                tensor_effective_flops_per_cycle_per_sm=tensor_effective,
                tensor_cycles=tensor_cycles,
                smem_raw_bytes_per_cycle_per_sm=per_sm_smem,
                smem_effective_bytes_per_cycle_per_sm=smem_rate,
                smem_inflight_bytes_per_sm=group_smem,
                smem_cycles=class_smem_cycles,
                total_cycles=total,
            )
        )
    useful_per_cta = float(
        2 * geometry.live_m * geometry.live_n * geometry.live_k
    )
    active_ctas = sum(
        item.sm_count * item.ctas_per_sm for item in occupancy_classes
    )
    return LocalMiniGemmResult(
        useful_flops=useful_per_cta * active_ctas,
        issued_flops=issued_flops,
        smem_read_bytes=smem_bytes,
        tensor_cycles=max(item.tensor_cycles for item in class_results),
        smem_cycles=max(item.smem_cycles for item in class_results),
        total_cycles=max(item.total_cycles for item in class_results),
        accumulator_chains_per_warp=max(
            (item.accumulator_chains for item in warp_work), default=0
        ),
        dependent_k_steps_per_chain=dependent_steps,
        reduction_groups=reduction_groups,
        occupancy_classes=tuple(class_results),
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
    """Summarize a finite stream at its persistent concurrency-limited rate."""
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
            latency_bearing_window_count=0,
            steady_state_bytes_per_cycle=peak_bytes_per_cycle,
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
    # The concurrency ceiling is a property of the persistent pipeline, not of
    # the analytically shortened final window.  Tail bytes therefore use the
    # same C_activeCTA * S_pipeline * Q_CTA,stage in-flight budget.
    final_rate = full_rate
    full_cycles = _service_from_effective_rate(full_bytes, full_rate)
    final_cycles = _service_from_effective_rate(final_bytes, final_rate)
    first_bytes = full_bytes if window_count > 1 else final_bytes
    first_cycles = full_cycles if window_count > 1 else final_cycles
    remaining_bytes = total_bytes - first_bytes
    remaining_cycles = _service_from_effective_rate(remaining_bytes, full_rate)
    total_cycles = first_cycles + remaining_cycles
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
        latency_bearing_window_count=window_count,
        steady_state_bytes_per_cycle=full_rate,
    )


class EffectiveRooflineModel(DispatchingOpModel):
    """Phase-aware GEMM roofline with concurrency-derived effective ceilings."""

    def __init__(self) -> None:
        super().__init__(
            {
                OpKind.GEMM: EffectiveGemmEstimator(batched=False),
                OpKind.BATCHED_GEMM: EffectiveGemmEstimator(batched=True),
                OpKind.ATTENTION_PREFILL: _EffectiveAttentionDispatcher(prefill=True),
                OpKind.ATTENTION_DECODE: _EffectiveAttentionDispatcher(prefill=False),
                OpKind.SOFTMAX: _EffectiveSoftmaxDispatcher(),
                OpKind.LAYERNORM: NormEstimator(layernorm=True),
                OpKind.RMSNORM: NormEstimator(layernorm=False),
                OpKind.ELEMENTWISE: ElementwiseEstimator(),
                OpKind.REDUCTION: ReductionEstimator(),
                OpKind.EMBEDDING: EmbeddingEstimator(),
                OpKind.COPY: CopyEstimator(),
            }
        )


class _EffectiveAttentionDispatcher:
    def __init__(self, *, prefill: bool) -> None:
        self._prefill = prefill
        self._fallback = AttentionEstimator(prefill=prefill)

    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        selector = op.attrs.get("attention_kernel")
        if selector is None:
            return self._fallback.estimate(op, hardware)
        if str(selector) != "flash_attention_ampere":
            raise ValueError(
                "attention_kernel must be flash_attention_ampere when specified"
            )
        if not self._prefill:
            raise ValueError("flash_attention_ampere supports prefill forward only")
        return EffectiveFlashAttentionEstimator().estimate(op, hardware)


class _EffectiveSoftmaxDispatcher:
    def __init__(self) -> None:
        self._fallback = SoftmaxEstimator()

    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        selector = op.attrs.get("softmax_kernel")
        if selector is None:
            return self._fallback.estimate(op, hardware)
        if str(selector) != "effective_cuda_ampere":
            raise ValueError(
                "softmax_kernel must be effective_cuda_ampere when specified"
            )
        return EffectiveSoftmaxEstimator().estimate(op, hardware)


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
                # Effective-roofline GEMM profiles already charge padded MMA
                # instructions in their compute event energy.  The extended
                # helper adds that padding to useful-FLOP energy and would
                # therefore count it twice here.
                selection_energy_j=profile.energy_j,
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
    compute_event_flops = _gemm_compute_event_flops(
        kernel=kernel,
        grid=grid,
        occupancy=occupancy,
        timeline=timeline,
    )
    # Global stores are serviced by L2 on their path to HBM.  The timing
    # traffic object intentionally keeps mainloop L2 requests separate from
    # the serial epilogue, but energy must charge the store transaction at
    # both memory levels.
    energy_memory_access = replace(
        traffic.memory_access,
        l2_write_bytes=(
            traffic.d_store_transaction_bytes
            if "l2" in hardware.memory_levels
            else None
        ),
        sram_write_bytes=(
            int(timeline.epilogue_smem_bytes)
            if "sram" in hardware.memory_levels
            else None
        ),
    )
    energy_breakdown = estimate_energy(
        flops=compute_event_flops,
        memory_access=energy_memory_access,
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
        compute_event_flops=compute_event_flops,
    )
    profile = OpProfile(
        latency_s=latency_s,
        energy_j=energy_breakdown.total_j,
        flops=grid.useful_flops,
        engine=EngineKind.TENSOR,
        footprint=footprint,
        memory_access=energy_memory_access,
        energy_breakdown=energy_breakdown,
        implementation=_implementation_name(batched),
        diagnostics=diagnostics,
    )
    return apply_calibrated_energy_model(profile, hardware)


def _gemm_compute_event_flops(
    *,
    kernel: extended.GemmKernelSpec,
    grid: extended.GridAccounting,
    occupancy: extended.OccupancyResult,
    timeline: EffectiveTimelineResult,
) -> float:
    """Return executed arithmetic work to charge as a compute energy event.

    Tensor instructions consume energy for the complete issued MMA shape even
    when boundary lanes do not contribute useful output elements.  Scalar
    fallback kernels do not have that tensor-tile padding contract, so retain
    their useful arithmetic count.
    """
    if (kernel.mma_m, kernel.mma_n, kernel.mma_k) == (1, 1, 1):
        return grid.useful_flops
    full_wave_count = max(0, occupancy.wave_count - 1)
    issued_flops = (
        full_wave_count * timeline.full_wave.issued_flops
        + timeline.last_wave.issued_flops
    )
    return max(grid.useful_flops, issued_flops)


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
    warp_work = _maximum_cta_warp_work(
        problem=problem,
        kernel=kernel,
        effective_warp_k=effective_warp_k,
        dtype_bytes=dtype_bytes,
    )
    accumulator_chains = max(
        (item.accumulator_chains for item in warp_work), default=0
    )
    dependent_k_steps = max(1, extended._ceil_div(effective_warp_k, kernel.mma_k))
    executed_k_groups = (
        max(0, grid.k_stages - 1) * groups_k + last_stage_groups_k
    )
    local_service = _local_mini_gemm_service(
        geometry=LocalMatmulGeometry(
            cta_m=kernel.cta_m,
            cta_n=kernel.cta_n,
            reduction_k=executed_k_groups * effective_warp_k,
            live_m=min(problem.m, kernel.cta_m),
            live_n=min(problem.n, kernel.cta_n),
            live_k=executed_k_groups * effective_warp_k,
            warp_m=kernel.warp_m,
            warp_n=kernel.warp_n,
            warp_k=effective_warp_k,
            mma_m=kernel.mma_m,
            mma_n=kernel.mma_n,
            mma_k=kernel.mma_k,
            warps_per_cta=kernel.warps_per_cta,
            lhs_smem_resident=True,
            rhs_smem_resident=True,
            input_bytes=dtype_bytes,
        ),
        occupancy_classes=classes,
        represented_sms=total_sms,
        peak_tensor_flops_per_cycle=peak_compute_flops_per_cycle,
        peak_smem_bytes_per_cycle=peak_smem_bw_per_cycle,
        tensor_latency_cycles=tensor_latency_cycles,
        shared_latency_cycles=shared_latency_cycles,
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
        chains_by_smsp_values = [0, 0, 0, 0]
        smem_bytes_by_smsp = [0.0, 0.0, 0.0, 0.0]
        for cta_index in range(occupancy_class.ctas_per_sm):
            for warp_index, item in enumerate(warp_work):
                smsp = (
                    cta_index * kernel.warps_per_cta + warp_index
                ) % _SMSPS_PER_SM
                chains_by_smsp_values[smsp] += item.accumulator_chains
                smem_bytes_by_smsp[smsp] += item.smem_bytes_per_k_group
        chains_by_smsp = tuple(chains_by_smsp_values)
        effective_smsps: list[float] = []
        smsp_tensor_cycles: list[float] = []
        for warps, chains in zip(warps_by_smsp, chains_by_smsp):
            if chains == 0:
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
            smsp_work = (
                executed_k_groups
                * chains
                * dependent_k_steps
                * mma_flops
            )
            smsp_tensor_cycles.append(
                _service_from_effective_rate(smsp_work, effective_smsps[-1])
            )
        tc_group_flops = sum(chains_by_smsp) * dependent_k_steps * mma_flops
        tc_cycles = max(smsp_tensor_cycles, default=0.0)
        tc_effective_per_sm = (
            executed_k_groups * tc_group_flops / tc_cycles if tc_cycles else 0.0
        )
        smem_group_bytes = sum(smem_bytes_by_smsp)
        smem_first_group_rate = _nonzero_effective_rate(
            per_sm_smem, smem_group_bytes, shared_latency_cycles
        )
        smem_first_group_cycles = _service_from_effective_rate(
            smem_group_bytes, smem_first_group_rate
        )
        smem_steady_remaining_cycles = _service_from_effective_rate(
            max(0, executed_k_groups - 1) * smem_group_bytes,
            smem_first_group_rate,
        )
        smem_cycles = smem_first_group_cycles + smem_steady_remaining_cycles
        smem_effective_per_sm = (
            executed_k_groups * smem_group_bytes / smem_cycles
            if smem_cycles
            else 0.0
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
                smem_first_group_cycles=smem_first_group_cycles,
                smem_steady_remaining_cycles=smem_steady_remaining_cycles,
                smem_cycles=smem_cycles,
                local_body_cycles=max(tc_cycles, smem_cycles),
            )
        )

    if not math.isclose(issued_flops, local_service.issued_flops, rel_tol=1.0e-12):
        raise AssertionError("shared local mini-GEMM service changed issued FLOPs")
    if not math.isclose(smem_read_bytes, local_service.smem_read_bytes, rel_tol=1.0e-12):
        raise AssertionError("shared local mini-GEMM service changed SMEM traffic")
    issued_flops = local_service.issued_flops
    smem_read_bytes = local_service.smem_read_bytes
    tc_body_cycles = local_service.tensor_cycles
    smem_body_cycles = local_service.smem_cycles
    steady_local_body_cycles = max(tc_body_cycles, smem_body_cycles)
    local_body_cycles = max(
        item.local_body_cycles
        + (
            min(item.tensor_cycles, item.smem_cycles) / executed_k_groups
            if executed_k_groups > 0
            else 0.0
        )
        for item in class_results
    )
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
    final_local_window_cycles = max(
        (
            max(0, last_memory_pipeline_stages - 1) * groups_k
            + last_stage_groups_k
        )
        * max(
            item.tensor_cycles / executed_k_groups,
            item.smem_cycles / executed_k_groups,
        )
        if executed_k_groups > 0
        else 0.0
        for item in class_results
    )
    body_cycles = max(
        local_body_cycles,
        remaining_memory_cycles + final_local_window_cycles,
    )
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
    memory_drain_path_cycles = remaining_memory_cycles + final_local_window_cycles
    body_resources = _body_limiting_resources(
        local_body_cycles=local_body_cycles,
        memory_drain_path_cycles=memory_drain_path_cycles,
        tc_body_cycles=tc_body_cycles,
        smem_body_cycles=smem_body_cycles,
        l2_remaining_cycles=l2_windows.remaining_cycles,
        dram_remaining_cycles=dram_windows.remaining_cycles,
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
        steady_local_body_cycles=steady_local_body_cycles,
        finite_local_body_cycles=local_body_cycles,
        local_fill_drain_cycles=max(
            0.0, local_body_cycles - steady_local_body_cycles
        ),
        final_local_window_cycles=final_local_window_cycles,
        memory_drain_path_cycles=memory_drain_path_cycles,
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


def _body_limiting_resources(
    *,
    local_body_cycles: float,
    memory_drain_path_cycles: float,
    tc_body_cycles: float,
    smem_body_cycles: float,
    l2_remaining_cycles: float,
    dram_remaining_cycles: float,
) -> tuple[str, ...]:
    maximum = max(local_body_cycles, memory_drain_path_cycles)
    tolerance = max(1.0e-9, abs(maximum) * 1.0e-9)
    limiting: set[str] = set()
    if local_body_cycles >= maximum - tolerance:
        limiting.update(
            _tied_max_resources(
                {"tensor": tc_body_cycles, "smem": smem_body_cycles},
                ("tensor", "smem"),
            )
        )
    if memory_drain_path_cycles >= maximum - tolerance:
        limiting.update(
            _tied_max_resources(
                {"l2": l2_remaining_cycles, "hbm": dram_remaining_cycles},
                ("l2", "hbm"),
            )
        )
    return tuple(name for name in _LIMITING_RESOURCE_ORDER if name in limiting)


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
        steady_local_body_cycles=0.0,
        finite_local_body_cycles=0.0,
        local_fill_drain_cycles=0.0,
        final_local_window_cycles=0.0,
        memory_drain_path_cycles=0.0,
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
        "steady_local_body": wave.steady_local_body_cycles,
        "finite_local_body": wave.finite_local_body_cycles,
        "local_fill_drain": wave.local_fill_drain_cycles,
        "final_local_window": wave.final_local_window_cycles,
        "memory_drain_path": wave.memory_drain_path_cycles,
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
    compute_event_flops: float,
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
    shared_memory_bytes_per_cta = extended._shared_memory_bytes_per_cta(problem, kernel)
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
        "compute_event_flops": compute_event_flops,
        "padded_mma_event_flops": max(0.0, compute_event_flops - useful),
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
            "finite_local_sequence_fill_and_drain",
            "final_memory_window_requires_local_drain",
            "persistent_memory_concurrency_limited_bandwidth",
            "persistent_smem_concurrency_limited_bandwidth",
            "maximum_cta_live_mma_and_smem_fragments",
            "four_smsps_per_sm_assumed",
            "l2_hbm_mainloop_overlap",
            "serial_epilogue_l2_plus_hbm",
        ),
    }


def _cuda_fp32_ops_per_cycle(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    rate = hardware.compute.vector_flops_per_s.get(DType.FP32)
    if rate is None:
        rate = hardware.compute.vector_flops_per_s.get(DType.BF16)
        warnings.append("cuda_fp32_rate_uses_vector_fallback")
    if rate is None:
        raise ValueError("effective CUDA softmax requires vector throughput")
    return rate / clock_hz


def _sfu_exp_ops_per_cycle(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    del clock_hz
    warnings.append("sfu_exp_rate_uses_ampere_default_16_ops_per_sm_cycle")
    return float(16 * (hardware.compute.num_sms or 1))


def _cuda_latency_cycles(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    del hardware, clock_hz
    warnings.append("cuda_latency_uses_ampere_default_4_cycles")
    return 4.0


def _sfu_latency_cycles(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    del hardware, clock_hz
    warnings.append("sfu_latency_uses_ampere_default_16_cycles")
    return 16.0


def _shuffle_latency_cycles(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    del hardware, clock_hz
    warnings.append("shuffle_latency_uses_ampere_default_4_cycles")
    return 4.0


def _barrier_latency_cycles(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    del hardware, clock_hz
    warnings.append("barrier_latency_uses_ampere_default_16_cycles")
    return 16.0


def _cuda_softmax_compute_service(
    *,
    work: CudaSoftmaxWorkSpec,
    kernel: CudaSoftmaxKernelSpec,
    occupancy_classes: tuple[SMOccupancyClass, ...],
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> CudaSoftmaxComputeResult:
    if not occupancy_classes:
        raise ValueError("CUDA softmax service requires an occupancy class")
    if work.rows <= 0 or work.reduction_elements <= 0:
        raise ValueError("CUDA softmax work dimensions must be positive")
    if work.valid_elements > work.issued_elements or work.masked_elements < 0:
        raise ValueError("invalid CUDA softmax element accounting")
    if work.mode == SoftmaxMode.COMPLETE_ROW and (
        work.has_prior_online_state or work.output_rescale_elements
    ):
        raise ValueError("complete-row softmax cannot carry online state")
    raw_cuda_global = _cuda_fp32_ops_per_cycle(hardware, clock_hz, warnings)
    raw_sfu_global = _sfu_exp_ops_per_cycle(hardware, clock_hz, warnings)
    num_sms = hardware.compute.num_sms or sum(c.sm_count for c in occupancy_classes)
    active_sms = sum(c.sm_count for c in occupancy_classes)
    raw_cuda = raw_cuda_global * active_sms / num_sms
    raw_sfu = raw_sfu_global * active_sms / num_sms
    cuda_latency = _cuda_latency_cycles(hardware, clock_hz, warnings)
    sfu_latency = _sfu_latency_cycles(hardware, clock_hz, warnings)
    shuffle_latency = _shuffle_latency_cycles(hardware, clock_hz, warnings)
    barrier_latency = _barrier_latency_cycles(hardware, clock_hz, warnings)

    scale_mask_ops = float(
        work.issued_elements
        * ((1 if work.apply_scale else 0) + (2 if work.apply_mask else 0))
    )
    rowmax_ops = float(max(0, work.valid_elements - work.rows))
    exp_elements = (
        work.valid_elements if work.predicate_masked_exp else work.issued_elements
    )
    exp_ops = float(exp_elements)
    exp_prep_sum_ops = float(exp_elements + max(0, work.valid_elements - work.rows))
    if work.mode == SoftmaxMode.ONLINE_ATTENTION:
        state_ops = float(work.rows * (6 if work.has_prior_online_state else 3))
        rescale_exp = work.rows if work.has_prior_online_state else 0
        exp_ops += float(rescale_exp)
    else:
        state_ops = float(work.valid_elements)
    output_rescale_ops = float(work.output_rescale_elements)
    cuda_ops = (
        scale_mask_ops + rowmax_ops + exp_prep_sum_ops + state_ops + output_rescale_ops
    )
    lanes = min(32, max(1, kernel.threads_per_row))
    warp_depth = int(math.ceil(math.log2(lanes))) if lanes > 1 else 0
    reduction_dependency = warp_depth * (shuffle_latency + cuda_latency)
    partial_warps = max(1, kernel.warps_per_reduction_group)
    cross_depth = int(math.ceil(math.log2(partial_warps))) if partial_warps > 1 else 0
    barrier_count = 2 if kernel.cross_warp_reduction else 0
    cross_dependency = (
        cross_depth * (shuffle_latency + cuda_latency)
        + barrier_count * barrier_latency
    )
    shuffle_ops = float(work.rows * (warp_depth * lanes + cross_depth * partial_warps))
    shared_reduction_bytes = (
        float(work.rows * partial_warps * 4 * 4)
        if kernel.cross_warp_reduction
        else 0.0
    )
    per_sm_cuda = raw_cuda_global / num_sms
    per_sm_sfu = raw_sfu_global / num_sms
    class_phase_cycles: list[tuple[float, float, float, float, float]] = []
    for occupancy_class in occupancy_classes:
        ctas = occupancy_class.ctas_per_sm
        active_warps = ctas * kernel.warps_per_cta
        cuda_inflight = float(active_warps * 32 * max(1, kernel.vector_width))
        sfu_inflight = float(active_warps * 32)
        class_cuda_rate = _nonzero_effective_rate(
            per_sm_cuda, cuda_inflight, cuda_latency
        )
        class_sfu_rate = _nonzero_effective_rate(
            per_sm_sfu, sfu_inflight, sfu_latency
        )
        class_phase_cycles.append(
            (
                _service_from_effective_rate(scale_mask_ops * ctas, class_cuda_rate),
                max(
                    _service_from_effective_rate(rowmax_ops * ctas, class_cuda_rate),
                    reduction_dependency + cross_dependency,
                ),
                max(
                    _service_from_effective_rate(
                        exp_prep_sum_ops * ctas, class_cuda_rate
                    ),
                    _service_from_effective_rate(exp_ops * ctas, class_sfu_rate),
                )
                + reduction_dependency
                + cross_dependency,
                _service_from_effective_rate(state_ops * ctas, class_cuda_rate),
                _service_from_effective_rate(
                    output_rescale_ops * ctas, class_cuda_rate
                ),
            )
        )
    scale_mask_cycles = max(item[0] for item in class_phase_cycles)
    rowmax_cycles = max(item[1] for item in class_phase_cycles)
    exp_sum_cycles = max(item[2] for item in class_phase_cycles)
    state_cycles = max(item[3] for item in class_phase_cycles)
    output_cycles = max(item[4] for item in class_phase_cycles)
    total = (
        scale_mask_cycles
        + rowmax_cycles
        + exp_sum_cycles
        + state_cycles
        + output_cycles
    )
    total_cuda_activity = sum(
        occupancy_class.sm_count * occupancy_class.ctas_per_sm * cuda_ops
        for occupancy_class in occupancy_classes
    )
    total_sfu_activity = sum(
        occupancy_class.sm_count * occupancy_class.ctas_per_sm * exp_ops
        for occupancy_class in occupancy_classes
    )
    cuda_effective = min(
        raw_cuda,
        total_cuda_activity / total if total else 0.0,
    )
    sfu_effective = min(
        raw_sfu,
        total_sfu_activity / total if total else 0.0,
    )
    return CudaSoftmaxComputeResult(
        issued_elements=work.issued_elements,
        valid_elements=work.valid_elements,
        cuda_scalar_ops=cuda_ops,
        exp_ops=exp_ops,
        shuffle_ops=shuffle_ops,
        shared_reduction_bytes=shared_reduction_bytes,
        barrier_count=barrier_count,
        cuda_raw_ops_per_cycle=raw_cuda,
        cuda_effective_ops_per_cycle=cuda_effective,
        sfu_raw_ops_per_cycle=raw_sfu,
        sfu_effective_ops_per_cycle=sfu_effective,
        scale_mask_cycles=scale_mask_cycles,
        rowmax_cycles=rowmax_cycles,
        exp_sum_cycles=exp_sum_cycles,
        state_or_normalize_cycles=state_cycles,
        output_rescale_cycles=output_cycles,
        total_cycles=total,
    )


def _parse_standalone_softmax(op: LocalOp) -> StandaloneSoftmaxProblemSpec:
    input_tensor = require_one_tensor(op, TensorRole.INPUT, "input")
    output_tensor = require_one_tensor(op, TensorRole.OUTPUT, "output")
    if input_tensor.shape != output_tensor.shape:
        raise ValueError("softmax input and output shapes must match")
    reduction = int(op.attrs.get("row_size", input_tensor.shape[-1]))
    if reduction <= 0 or numel(input_tensor.shape) % reduction:
        raise ValueError("softmax row_size must divide the input element count")
    return StandaloneSoftmaxProblemSpec(
        row_count=numel(input_tensor.shape) // reduction,
        reduction_elements=reduction,
        input_dtype=input_tensor.dtype,
        output_dtype=output_tensor.dtype,
    )


def _standalone_softmax_kernel(
    problem: StandaloneSoftmaxProblemSpec, attrs: Mapping[str, Any]
) -> StandaloneSoftmaxKernelSpec:
    default_threads = min(
        1024,
        max(
            32,
            2 ** math.ceil(math.log2(min(problem.reduction_elements, 1024))),
        ),
    )
    threads = int(attrs.get("threads_per_cta", default_threads))
    if threads <= 0 or threads > 1024 or threads % 32:
        raise ValueError("standalone softmax threads_per_cta must be a warp multiple <= 1024")
    rows_per_cta = int(attrs.get("rows_per_cta", 1))
    vector_width = int(attrs.get("vector_width", 4))
    registers_per_thread = int(attrs.get("registers_per_thread", 64))
    if rows_per_cta <= 0 or vector_width <= 0 or registers_per_thread <= 0:
        raise ValueError("standalone softmax kernel fields must be positive")
    stage = bool(attrs.get("softmax_stage_values_in_smem", problem.reduction_elements > 1024))
    if problem.reduction_elements > 4096:
        raise ValueError("effective standalone softmax supports complete rows up to 4096 elements")
    dtype_bytes = dtype_nbytes(problem.input_dtype)
    smem = int(problem.reduction_elements * rows_per_cta * dtype_bytes) if stage else 0
    max_concurrent = (
        int(attrs["max_concurrent_block"])
        if attrs.get("max_concurrent_block") is not None
        else None
    )
    if max_concurrent is not None and max_concurrent <= 0:
        raise ValueError("max_concurrent_block must be positive")
    return StandaloneSoftmaxKernelSpec(
        threads_per_cta=threads,
        rows_per_cta=rows_per_cta,
        warps_per_cta=threads // 32,
        vector_width=vector_width,
        stage_values_in_smem=stage,
        shared_memory_bytes_per_cta=smem,
        registers_per_thread=registers_per_thread,
        max_concurrent_ctas_per_sm=max_concurrent,
    )


def _resident_ctas_per_sm(
    *,
    warps_per_cta: int,
    threads_per_cta: int,
    registers_per_thread: int,
    shared_memory_bytes_per_cta: int,
    explicit_limit: int | None,
    hardware: HardwareSpec,
) -> int:
    limits = [explicit_limit or hardware.compute.max_ctas_per_sm or 8]
    if hardware.compute.max_warps_per_sm:
        limits.append(hardware.compute.max_warps_per_sm // warps_per_cta)
    if hardware.compute.registers_per_sm:
        limits.append(
            hardware.compute.registers_per_sm
            // max(1, registers_per_thread * threads_per_cta)
        )
    if shared_memory_bytes_per_cta and hardware.compute.shared_memory_bytes_per_sm:
        limits.append(
            hardware.compute.shared_memory_bytes_per_sm
            // shared_memory_bytes_per_cta
        )
    resident = min(limits)
    if resident <= 0:
        raise ValueError("kernel resources do not permit one resident CTA")
    return resident


def _softmax_wave(
    *,
    active_ctas: int,
    active_rows: int,
    problem: StandaloneSoftmaxProblemSpec,
    kernel: StandaloneSoftmaxKernelSpec,
    resident_ctas: int,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> StandaloneSoftmaxWaveResult:
    if active_ctas == 0:
        zero_compute = CudaSoftmaxComputeResult(*((0,) * 7), *((0.0,) * 13))
        return StandaloneSoftmaxWaveResult(0, 0, 0.0, 0.0, 0.0, 0.0, zero_compute)
    num_sms = hardware.compute.num_sms or 1
    busy, rem = divmod(active_ctas, num_sms)
    classes: list[SMOccupancyClass] = []
    if busy:
        classes.append(SMOccupancyClass("busy", rem or num_sms, busy + (1 if rem else 0)))
        if rem:
            classes.append(SMOccupancyClass("lazy", num_sms - rem, busy))
    else:
        classes.append(SMOccupancyClass("lazy", active_ctas, 1))
    rows_per_cta = min(kernel.rows_per_cta, active_rows)
    issued = rows_per_cta * problem.reduction_elements
    softmax_kernel = CudaSoftmaxKernelSpec(
        threads_per_row=kernel.threads_per_cta,
        rows_per_cta=kernel.rows_per_cta,
        warps_per_cta=kernel.warps_per_cta,
        warps_per_reduction_group=kernel.warps_per_cta,
        vector_width=kernel.vector_width,
        cross_warp_reduction=kernel.warps_per_cta > 1,
        stage_values_in_smem=kernel.stage_values_in_smem,
    )
    compute = _cuda_softmax_compute_service(
        work=CudaSoftmaxWorkSpec(
            mode=SoftmaxMode.COMPLETE_ROW,
            rows=rows_per_cta,
            reduction_elements=problem.reduction_elements,
            issued_elements=issued,
            valid_elements=issued,
            masked_elements=0,
            output_rescale_elements=0,
            has_prior_online_state=False,
            apply_scale=False,
            apply_mask=False,
            predicate_masked_exp=True,
        ),
        kernel=softmax_kernel,
        occupancy_classes=tuple(classes),
        hardware=hardware,
        clock_hz=clock_hz,
        warnings=warnings,
    )
    wave_bytes = active_rows * problem.reduction_elements * dtype_nbytes(problem.input_dtype)
    l2 = hardware.memory_levels.get("l2")
    hbm = hardware.memory_levels["hbm"]
    l2_rate = l2.bandwidth_bytes_per_s / clock_hz if l2 else math.inf
    hbm_rate = hbm.bandwidth_bytes_per_s / clock_hz
    l2_latency = l2.latency_s * clock_hz if l2 else 0.0
    hbm_latency = hbm.latency_s * clock_hz
    inflight = active_ctas * rows_per_cta * problem.reduction_elements * dtype_nbytes(problem.input_dtype)
    load = max(
        _service_from_effective_rate(wave_bytes, _nonzero_effective_rate(l2_rate, inflight, l2_latency)) if l2 else 0.0,
        _service_from_effective_rate(wave_bytes, _nonzero_effective_rate(hbm_rate, inflight, hbm_latency)),
    )
    store = load
    smem_cycles = 0.0
    if kernel.stage_values_in_smem:
        smem_bytes = wave_bytes * 3.0
        smem_peak = extended._smem_bandwidth_per_cycle(hardware, clock_hz, warnings)
        smem_latency = extended._shared_latency_cycles(hardware, clock_hz, warnings)
        smem_inflight = inflight
        smem_cycles = _service_from_effective_rate(
            smem_bytes, _nonzero_effective_rate(smem_peak, smem_inflight, smem_latency)
        )
    compute_cycles = compute.total_cycles + smem_cycles
    return StandaloneSoftmaxWaveResult(
        active_ctas=active_ctas,
        active_rows=active_rows,
        load_cycles=load,
        compute_cycles=compute_cycles,
        store_cycles=store,
        total_cycles=load + compute_cycles + store,
        compute=compute,
    )


class EffectiveSoftmaxEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        problem = _parse_standalone_softmax(op)
        kernel = _standalone_softmax_kernel(problem, op.attrs)
        warnings: list[str] = []
        clock_hz = extended._clock_hz(hardware, warnings)
        num_sms = hardware.compute.num_sms or 1
        resident = _resident_ctas_per_sm(
            warps_per_cta=kernel.warps_per_cta,
            threads_per_cta=kernel.threads_per_cta,
            registers_per_thread=kernel.registers_per_thread,
            shared_memory_bytes_per_cta=kernel.shared_memory_bytes_per_cta,
            explicit_limit=kernel.max_concurrent_ctas_per_sm,
            hardware=hardware,
        )
        cta_count = extended._ceil_div(problem.row_count, kernel.rows_per_cta)
        ctas_per_wave = num_sms * resident
        wave_count = extended._ceil_div(cta_count, ctas_per_wave)
        full_ctas = min(cta_count, ctas_per_wave)
        last_ctas = cta_count - max(0, wave_count - 1) * ctas_per_wave
        full_rows = min(problem.row_count, full_ctas * kernel.rows_per_cta)
        last_rows = problem.row_count - max(0, wave_count - 1) * ctas_per_wave * kernel.rows_per_cta
        full = _softmax_wave(
            active_ctas=full_ctas,
            active_rows=full_rows,
            problem=problem,
            kernel=kernel,
            resident_ctas=resident,
            hardware=hardware,
            clock_hz=clock_hz,
            warnings=warnings,
        )
        last = _softmax_wave(
            active_ctas=last_ctas,
            active_rows=last_rows,
            problem=problem,
            kernel=kernel,
            resident_ctas=resident,
            hardware=hardware,
            clock_hz=clock_hz,
            warnings=warnings,
        )
        timeline = StandaloneSoftmaxTimelineResult(
            kernel_cycles=max(0, wave_count - 1) * full.total_cycles
            + last.total_cycles,
            full_wave=full,
            last_wave=last,
            wave_count=wave_count,
        )
        kernel_cycles = timeline.kernel_cycles
        fixed = float(hardware.compute.device_fixed_overhead_cycles or 0)
        latency_s = (kernel_cycles + fixed) / clock_hz
        footprint = footprint_from_tensors(op)
        input_bytes = footprint.input_bytes
        output_bytes = footprint.output_bytes
        smem_value_bytes = (
            input_bytes * 3 if kernel.stage_values_in_smem else 0
        )
        shared_reduction = int(
            (max(0, wave_count - 1) * full.compute.shared_reduction_bytes * full.active_ctas)
            + last.compute.shared_reduction_bytes * last.active_ctas
        )
        memory = MemoryAccess(
            hbm_read_bytes=input_bytes,
            hbm_write_bytes=output_bytes,
            l2_read_bytes=input_bytes,
            l2_write_bytes=output_bytes,
            sram_read_bytes=(input_bytes * 2 if kernel.stage_values_in_smem else 0)
            + shared_reduction // 2,
            sram_write_bytes=(input_bytes if kernel.stage_values_in_smem else 0)
            + shared_reduction // 2,
        )
        flops = float(problem.row_count * (4 * problem.reduction_elements - 2))
        energy = estimate_energy(
            flops=flops,
            memory_access=memory,
            engine=EngineKind.VECTOR,
            dtype=problem.input_dtype,
            hardware=hardware,
            latency_s=latency_s,
        )
        diagnostics = {
            "problem": asdict(problem),
            "kernel": asdict(kernel),
            "row_count": problem.row_count,
            "reduction_length": problem.reduction_elements,
            "residency_strategy": "smem_staged" if kernel.stage_values_in_smem else "register_resident",
            "cta_count": cta_count,
            "cta_waves": wave_count,
            "resident_ctas_per_sm": resident,
            "ctas_per_wave": ctas_per_wave,
            "wave_shape": {"last_wave_ctas": last_ctas},
            "wave_pipeline": {"full": asdict(full), "last": asdict(last)},
            "cuda_softmax_compute": asdict(full.compute),
            "transaction_bytes": {
                "input_l2": input_bytes,
                "input_hbm": input_bytes,
                "output_l2": output_bytes,
                "output_hbm": output_bytes,
                "smem_value_staging": smem_value_bytes,
                "smem_reduction": shared_reduction,
                "smem_read": memory.sram_read_bytes,
                "smem_write": memory.sram_write_bytes,
            },
            "active_cycles": {
                "compute": max(0, wave_count - 1) * full.compute_cycles + last.compute_cycles,
                "smem": 0.0 if not kernel.stage_values_in_smem else max(0, wave_count - 1) * full.compute_cycles + last.compute_cycles,
                "l2": (input_bytes + output_bytes) / (hardware.memory_levels["l2"].bandwidth_bytes_per_s / clock_hz) if "l2" in hardware.memory_levels else 0.0,
                "dram": (input_bytes + output_bytes) / (hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz),
            },
            "phase_cycles": {
                "load": max(0, wave_count - 1) * full.load_cycles + last.load_cycles,
                "compute": max(0, wave_count - 1) * full.compute_cycles + last.compute_cycles,
                "store": max(0, wave_count - 1) * full.store_cycles + last.store_cycles,
            },
            "clock_hz": clock_hz,
            "primary_bottleneck": max(
                ("l2", full.load_cycles),
                ("cuda_softmax", full.compute_cycles),
                ("hbm", full.store_cycles),
                key=lambda item: item[1],
            )[0],
            "secondary_bottlenecks": (),
            "warnings": tuple(dict.fromkeys(warnings)),
            "assumptions": (
                "complete_row_one_cta_softmax",
                "persistent_concurrency_limited_bandwidth",
                "cuda_sfu_architecture_defaults",
            ),
        }
        profile = OpProfile(
            latency_s=latency_s,
            energy_j=energy.total_j,
            flops=flops,
            engine=EngineKind.VECTOR,
            footprint=footprint,
            memory_access=memory,
            energy_breakdown=energy,
            implementation="effective_roofline.softmax_cuda_ampere",
            diagnostics=diagnostics,
        )
        return apply_calibrated_energy_model(profile, hardware)


def _parse_flash_attention_problem(op: LocalOp) -> FlashAttentionProblemSpec:
    parsed = parse_attention(op)
    q, k, v, output = parsed["q"], parsed["k"], parsed["v"], parsed["output"]
    if any(tensor is None for tensor in (q, k, v, output)):
        raise ValueError("FlashAttention requires canonical Q, K, V, and output tensors")
    assert q is not None and k is not None and v is not None and output is not None
    if any(len(tensor.shape) != 4 for tensor in (q, k, v, output)):
        raise ValueError("FlashAttention tensors must have [batch, heads, sequence, dim] shape")
    if q.dtype not in (DType.BF16, DType.FP16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Ampere FlashAttention supports matching BF16 or FP16 Q/K/V")
    if output.dtype != q.dtype:
        raise ValueError("Ampere FlashAttention output dtype must match Q/K/V")
    unsupported = {
        name: op.attrs.get(name)
        for name in (
            "dropout",
            "dropout_p",
            "bias",
            "attention_bias",
            "alibi",
            "window_size",
            "windowing",
            "sparse_mask",
            "paged_kv",
            "split_kv",
        )
        if op.attrs.get(name) not in (None, False, 0, 0.0)
    }
    if unsupported:
        raise ValueError(
            "Ampere FlashAttention does not support: " + ", ".join(sorted(unsupported))
        )
    problem = FlashAttentionProblemSpec(
        batch=int(parsed["batch"]),
        query_heads=int(parsed["heads"]),
        kv_heads=int(op.attrs.get("kv_heads", k.shape[1])),
        seq_q=int(parsed["seq_q"]),
        seq_kv=int(parsed["seq_kv"]),
        head_dim=int(parsed["head_dim"]),
        value_dim=int(op.attrs.get("value_dim", v.shape[3])),
        input_dtype=q.dtype,
        output_dtype=output.dtype,
        causal=bool(op.attrs.get("causal", False)),
        causal_alignment=str(op.attrs.get("causal_alignment", "bottom_right")),
        store_lse=bool(op.attrs.get("store_lse", False)),
    )
    dimensions = (
        problem.batch,
        problem.query_heads,
        problem.kv_heads,
        problem.seq_q,
        problem.seq_kv,
        problem.head_dim,
        problem.value_dim,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("FlashAttention dimensions must be positive")
    if problem.query_heads % problem.kv_heads:
        raise ValueError("FlashAttention query_heads must be divisible by kv_heads")
    if not problem.causal:
        raise ValueError("flash_attention_ampere currently requires causal=True")
    if problem.causal_alignment != "bottom_right":
        raise ValueError("only bottom_right causal alignment is supported")
    if output.shape != (
        problem.batch,
        problem.query_heads,
        problem.seq_q,
        problem.value_dim,
    ):
        raise ValueError("FlashAttention output shape is inconsistent with Q and V")
    return problem


def _ampere_flash_attention_kernel(
    problem: FlashAttentionProblemSpec, attrs: Mapping[str, Any]
) -> AmpereFlashAttentionKernelSpec:
    block_q = int(attrs.get("block_q", attrs.get("cta_tile_m", 128)))
    block_k = int(
        attrs.get("block_k", attrs.get("cta_tile_n", 128 if problem.head_dim <= 64 else 64))
    )
    warps = int(attrs.get("warps_per_cta", attrs.get("n_warps_per_block", 8 if problem.head_dim > 128 else 4)))
    threads = int(attrs.get("threads_per_cta", warps * 32))
    pipeline = int(attrs.get("pipeline_stages", 2))
    if pipeline != 2:
        raise ValueError("Ampere FlashAttention currently requires pipeline_stages == 2")
    if threads != warps * 32:
        raise ValueError("FlashAttention threads_per_cta must equal 32 * warps_per_cta")
    warp_m = int(attrs.get("warp_tile_m", min(64, block_q)))
    warp_n = int(attrs.get("warp_tile_n", max(8, extended._ceil_div(block_k, max(1, warps // max(1, extended._ceil_div(block_q, warp_m)))))))
    dtype_bytes = dtype_nbytes(problem.input_dtype)
    smem_default = int(
        pipeline * block_k * (problem.head_dim + problem.value_dim) * dtype_bytes
    )
    explicit_occupancy = attrs.get("max_concurrent_block", attrs.get("max_concurrent_ctas_per_sm"))
    if explicit_occupancy is None:
        raise ValueError(
            "flash_attention_ampere requires max_concurrent_ctas_per_sm or max_concurrent_block"
        )
    spec = AmpereFlashAttentionKernelSpec(
        block_q=block_q,
        block_k=block_k,
        warps_per_cta=warps,
        threads_per_cta=threads,
        qk_warp_m=int(attrs.get("qk_warp_m", warp_m)),
        qk_warp_n=int(attrs.get("qk_warp_n", warp_n)),
        qk_warp_k=int(attrs.get("qk_warp_k", 16)),
        pv_warp_m=int(attrs.get("pv_warp_m", warp_m)),
        pv_warp_n=int(attrs.get("pv_warp_n", max(8, extended._ceil_div(problem.value_dim, max(1, warps // max(1, extended._ceil_div(block_q, warp_m))))))),
        pv_warp_k=int(attrs.get("pv_warp_k", 16)),
        mma_m=int(attrs.get("mma_m", 16)),
        mma_n=int(attrs.get("mma_n", 8)),
        mma_k=int(attrs.get("mma_k", 16)),
        pipeline_stages=pipeline,
        registers_per_thread=int(attrs.get("registers_per_thread", 128)),
        shared_memory_bytes_per_cta=int(attrs.get("shared_memory_bytes_per_cta", smem_default)),
        max_concurrent_ctas_per_sm=int(explicit_occupancy),
        q_register_resident=bool(attrs.get("q_register_resident", True)),
        probability_register_resident=bool(attrs.get("probability_register_resident", True)),
        fp32_accumulators=bool(attrs.get("fp32_accumulators", True)),
        epilogue_through_smem=bool(attrs.get("epilogue_through_smem", True)),
        rows_per_softmax_group=int(attrs.get("rows_per_softmax_group", max(1, block_q // warps))),
        warps_per_softmax_group=int(attrs.get("warps_per_softmax_group", 1)),
        predicate_masked_exp=bool(attrs.get("predicate_masked_exp", True)),
    )
    positive = (
        spec.block_q,
        spec.block_k,
        spec.warps_per_cta,
        spec.threads_per_cta,
        spec.qk_warp_m,
        spec.qk_warp_n,
        spec.qk_warp_k,
        spec.pv_warp_m,
        spec.pv_warp_n,
        spec.pv_warp_k,
        spec.mma_m,
        spec.mma_n,
        spec.mma_k,
        spec.registers_per_thread,
        spec.shared_memory_bytes_per_cta,
        spec.max_concurrent_ctas_per_sm or 0,
        spec.rows_per_softmax_group,
        spec.warps_per_softmax_group,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Ampere FlashAttention kernel fields must be positive")
    return spec


def _flash_attention_ctas(
    problem: FlashAttentionProblemSpec, kernel: AmpereFlashAttentionKernelSpec
) -> tuple[FlashAttentionCTADescriptor, ...]:
    ctas: list[FlashAttentionCTADescriptor] = []
    query_blocks = extended._ceil_div(problem.seq_q, kernel.block_q)
    heads_per_kv = problem.query_heads // problem.kv_heads
    for batch_index in range(problem.batch):
        for query_head in range(problem.query_heads):
            kv_head = query_head // heads_per_kv
            for query_block in range(query_blocks):
                query_start = query_block * kernel.block_q
                query_end = min(problem.seq_q, query_start + kernel.block_q)
                live_q = query_end - query_start
                max_visible_kv = min(
                    problem.seq_kv,
                    max(0, query_end + problem.seq_kv - problem.seq_q),
                )
                tiles: list[FlashAttentionKVTile] = []
                for tile_index, kv_start in enumerate(range(0, max_visible_kv, kernel.block_k)):
                    kv_end = min(problem.seq_kv, kv_start + kernel.block_k)
                    live_kv = kv_end - kv_start
                    useful = 0
                    for query_row in range(query_start, query_end):
                        row_visible = min(
                            problem.seq_kv,
                            max(0, query_row + 1 + problem.seq_kv - problem.seq_q),
                        )
                        useful += max(0, min(kv_end, row_visible) - kv_start)
                    issued = live_q * kernel.block_k
                    tiles.append(
                        FlashAttentionKVTile(
                            tile_index=tile_index,
                            kv_start=kv_start,
                            live_kv_rows=live_kv,
                            issued_kv_rows=kernel.block_k,
                            is_causal_diagonal=useful < live_q * live_kv,
                            useful_score_elements=useful,
                            issued_score_elements=issued,
                        )
                    )
                ctas.append(
                    FlashAttentionCTADescriptor(
                        linear_cta=len(ctas),
                        batch_index=batch_index,
                        query_head=query_head,
                        kv_head=kv_head,
                        query_block=query_block,
                        query_start=query_start,
                        live_query_rows=live_q,
                        kv_tiles=tuple(tiles),
                    )
                )
    expected = problem.batch * problem.query_heads * query_blocks
    if len(ctas) != expected:
        raise AssertionError("FlashAttention CTA enumeration does not conserve the grid")
    return tuple(ctas)


def _flash_attention_waves(
    ctas: tuple[FlashAttentionCTADescriptor, ...],
    resident_ctas_per_sm: int,
    hardware: HardwareSpec,
) -> tuple[FlashAttentionWave, ...]:
    num_sms = hardware.compute.num_sms or 1
    capacity = num_sms * resident_ctas_per_sm
    waves: list[FlashAttentionWave] = []
    for wave_index, start in enumerate(range(0, len(ctas), capacity)):
        wave_ctas = ctas[start : start + capacity]
        by_sm: list[list[FlashAttentionCTADescriptor]] = [[] for _ in range(num_sms)]
        for local_index, cta in enumerate(wave_ctas):
            by_sm[local_index % num_sms].append(cta)
        waves.append(
            FlashAttentionWave(
                index=wave_index,
                ctas=wave_ctas,
                ctas_by_sm=tuple(tuple(items) for items in by_sm),
            )
        )
    return tuple(waves)


def _classes_for_flash_ctas(
    wave: FlashAttentionWave, active: tuple[FlashAttentionCTADescriptor, ...]
) -> tuple[SMOccupancyClass, ...]:
    active_ids = {cta.linear_cta for cta in active}
    counts: dict[int, int] = {}
    for sm_ctas in wave.ctas_by_sm:
        count = sum(cta.linear_cta in active_ids for cta in sm_ctas)
        if count:
            counts[count] = counts.get(count, 0) + 1
    return tuple(
        SMOccupancyClass(f"ctas_{ctas_per_sm}", sm_count, ctas_per_sm)
        for ctas_per_sm, sm_count in sorted(counts.items(), reverse=True)
    )


def _round_transaction(byte_count: float, transaction_bytes: int = 32) -> float:
    if byte_count <= 0:
        return 0.0
    return float(math.ceil(byte_count / transaction_bytes) * transaction_bytes)


def _flash_memory_action_service(
    *,
    name: str,
    l2_bytes: float,
    hbm_bytes: float,
    smem_write_bytes: float,
    active_ctas: int,
    bytes_per_cta_stage: float,
    pipeline_slots: int,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> FlashMemoryActionResult:
    if active_ctas <= 0 or max(l2_bytes, hbm_bytes, smem_write_bytes) == 0:
        return FlashMemoryActionResult(name, active_ctas, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    l2 = hardware.memory_levels.get("l2")
    hbm = hardware.memory_levels["hbm"]
    l2_raw = l2.bandwidth_bytes_per_s / clock_hz if l2 else 0.0
    hbm_raw = hbm.bandwidth_bytes_per_s / clock_hz
    inflight = active_ctas * pipeline_slots * bytes_per_cta_stage
    l2_rate = (
        _nonzero_effective_rate(l2_raw, inflight, l2.latency_s * clock_hz)
        if l2 and l2_bytes
        else 0.0
    )
    hbm_rate = _nonzero_effective_rate(
        hbm_raw, inflight, hbm.latency_s * clock_hz
    ) if hbm_bytes else 0.0
    l2_cycles = _service_from_effective_rate(l2_bytes, l2_rate)
    hbm_cycles = _service_from_effective_rate(hbm_bytes, hbm_rate)
    smem_cycles = 0.0
    if smem_write_bytes:
        smem_raw = extended._smem_bandwidth_per_cycle(hardware, clock_hz, warnings)
        smem_latency = extended._shared_latency_cycles(hardware, clock_hz, warnings)
        smem_rate = _nonzero_effective_rate(smem_raw, inflight, smem_latency)
        smem_cycles = _service_from_effective_rate(smem_write_bytes, smem_rate)
    return FlashMemoryActionResult(
        name=name,
        active_ctas=active_ctas,
        total_bytes=max(l2_bytes, hbm_bytes),
        inflight_bytes=inflight,
        l2_raw_bytes_per_cycle=l2_raw,
        l2_effective_bytes_per_cycle=l2_rate,
        hbm_raw_bytes_per_cycle=hbm_raw,
        hbm_effective_bytes_per_cycle=hbm_rate,
        l2_cycles=l2_cycles,
        hbm_cycles=hbm_cycles,
        total_cycles=max(l2_cycles, hbm_cycles, smem_cycles),
    )


def _cuda_online_attention_softmax_service(
    *,
    active: tuple[FlashAttentionCTADescriptor, ...],
    tile_index: int,
    classes: tuple[SMOccupancyClass, ...],
    kernel: AmpereFlashAttentionKernelSpec,
    problem: FlashAttentionProblemSpec,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> FlashSoftmaxStageResult:
    max_rows = max(cta.live_query_rows for cta in active)
    tiles = [cta.kv_tiles[tile_index] for cta in active]
    max_valid = max(tile.useful_score_elements for tile in tiles)
    max_issued = max(tile.issued_score_elements for tile in tiles)
    prior = tile_index > 0
    output_rescale = max_rows * problem.value_dim if prior else 0
    cuda_kernel = CudaSoftmaxKernelSpec(
        threads_per_row=32 * kernel.warps_per_softmax_group,
        rows_per_cta=max_rows,
        warps_per_cta=kernel.warps_per_cta,
        warps_per_reduction_group=kernel.warps_per_softmax_group,
        vector_width=1,
        cross_warp_reduction=kernel.warps_per_softmax_group > 1,
        stage_values_in_smem=False,
    )
    compute = _cuda_softmax_compute_service(
        work=CudaSoftmaxWorkSpec(
            mode=SoftmaxMode.ONLINE_ATTENTION,
            rows=max_rows,
            reduction_elements=kernel.block_k,
            issued_elements=max_issued,
            valid_elements=max_valid,
            masked_elements=max_issued - max_valid,
            output_rescale_elements=output_rescale,
            has_prior_online_state=prior,
            apply_scale=True,
            apply_mask=any(tile.is_causal_diagonal for tile in tiles),
            predicate_masked_exp=kernel.predicate_masked_exp,
        ),
        kernel=cuda_kernel,
        occupancy_classes=classes,
        hardware=hardware,
        clock_hz=clock_hz,
        warnings=warnings,
    )
    return FlashSoftmaxStageResult(
        compute=compute,
        causal_diagonal=any(tile.is_causal_diagonal for tile in tiles),
        online_state_rows=max_rows,
        output_rescale_elements=output_rescale,
    )


def _flash_kv_action(
    *,
    name: str,
    wave: FlashAttentionWave,
    active: tuple[FlashAttentionCTADescriptor, ...],
    tile_index: int,
    problem: FlashAttentionProblemSpec,
    kernel: AmpereFlashAttentionKernelSpec,
    policy: FlashKVReusePolicy,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> FlashMemoryActionResult:
    is_value = name.startswith("v")
    dimension = problem.value_dim if is_value else problem.head_dim
    byte_width = dtype_nbytes(problem.input_dtype)
    request_sizes = [
        _round_transaction(cta.kv_tiles[tile_index].live_kv_rows * dimension * byte_width)
        for cta in active
    ]
    l2_bytes = sum(request_sizes)
    if policy == FlashKVReusePolicy.NONE:
        hbm_bytes = l2_bytes
    else:
        seen: set[tuple[int, ...]] = set()
        hbm_bytes = 0.0
        swizzle_size = 1
        if policy == FlashKVReusePolicy.IDEAL_WITHIN_SWIZZLE_GROUP:
            swizzle_size = 4
        for cta, byte_count in zip(active, request_sizes):
            group = cta.query_block // swizzle_size
            key = (
                wave.index,
                cta.batch_index,
                cta.kv_head,
                tile_index,
                group if policy == FlashKVReusePolicy.IDEAL_WITHIN_SWIZZLE_GROUP else 0,
            )
            if key not in seen:
                seen.add(key)
                hbm_bytes += byte_count
    average = l2_bytes / len(active)
    return _flash_memory_action_service(
        name=name,
        l2_bytes=l2_bytes,
        hbm_bytes=hbm_bytes,
        smem_write_bytes=l2_bytes,
        active_ctas=len(active),
        bytes_per_cta_stage=average,
        pipeline_slots=kernel.pipeline_stages,
        hardware=hardware,
        clock_hz=clock_hz,
        warnings=warnings,
    )


def _ampere_flash_attention_wave(
    *,
    wave: FlashAttentionWave,
    problem: FlashAttentionProblemSpec,
    kernel: AmpereFlashAttentionKernelSpec,
    policy: FlashKVReusePolicy,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
) -> FlashAttentionWaveResult:
    peak_tensor = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_smem = extended._smem_bandwidth_per_cycle(hardware, clock_hz, warnings)
    tensor_latency = extended._tensor_latency_cycles(kernel, hardware, warnings)  # type: ignore[arg-type]
    shared_latency = extended._shared_latency_cycles(hardware, clock_hz, warnings)
    byte_width = dtype_nbytes(problem.input_dtype)
    q_bytes = sum(
        _round_transaction(cta.live_query_rows * problem.head_dim * byte_width)
        for cta in wave.ctas
    )
    q_action = _flash_memory_action_service(
        name="q_load",
        l2_bytes=q_bytes,
        hbm_bytes=q_bytes,
        smem_write_bytes=0.0 if kernel.q_register_resident else q_bytes,
        active_ctas=len(wave.ctas),
        bytes_per_cta_stage=q_bytes / len(wave.ctas),
        pipeline_slots=1,
        hardware=hardware,
        clock_hz=clock_hz,
        warnings=warnings,
    )
    max_iterations = max((len(cta.kv_tiles) for cta in wave.ctas), default=0)
    iterations: list[FlashAttentionIterationResult] = []
    k_actions: list[FlashMemoryActionResult] = []
    active_by_iteration: list[tuple[FlashAttentionCTADescriptor, ...]] = []
    for iteration in range(max_iterations):
        active = tuple(cta for cta in wave.ctas if iteration < len(cta.kv_tiles))
        active_by_iteration.append(active)
        k_actions.append(
            _flash_kv_action(
                name=f"k_load_{iteration}",
                wave=wave,
                active=active,
                tile_index=iteration,
                problem=problem,
                kernel=kernel,
                policy=policy,
                hardware=hardware,
                clock_hz=clock_hz,
                warnings=warnings,
            )
        )
    for iteration, active in enumerate(active_by_iteration):
        classes = _classes_for_flash_ctas(wave, active)
        represented_sms = sum(item.sm_count for item in classes)
        max_q = max(cta.live_query_rows for cta in active)
        max_kv = max(cta.kv_tiles[iteration].live_kv_rows for cta in active)
        qk = _local_mini_gemm_service(
            geometry=LocalMatmulGeometry(
                cta_m=kernel.block_q,
                cta_n=kernel.block_k,
                reduction_k=problem.head_dim,
                live_m=max_q,
                live_n=max_kv,
                live_k=problem.head_dim,
                warp_m=kernel.qk_warp_m,
                warp_n=kernel.qk_warp_n,
                warp_k=kernel.qk_warp_k,
                mma_m=kernel.mma_m,
                mma_n=kernel.mma_n,
                mma_k=kernel.mma_k,
                warps_per_cta=kernel.warps_per_cta,
                lhs_smem_resident=not kernel.q_register_resident,
                rhs_smem_resident=True,
                input_bytes=byte_width,
            ),
            occupancy_classes=classes,
            represented_sms=represented_sms,
            peak_tensor_flops_per_cycle=peak_tensor * represented_sms / (hardware.compute.num_sms or represented_sms),
            peak_smem_bytes_per_cycle=peak_smem * represented_sms / (hardware.compute.num_sms or represented_sms),
            tensor_latency_cycles=tensor_latency,
            shared_latency_cycles=shared_latency,
        )
        softmax = _cuda_online_attention_softmax_service(
            active=active,
            tile_index=iteration,
            classes=classes,
            kernel=kernel,
            problem=problem,
            hardware=hardware,
            clock_hz=clock_hz,
            warnings=warnings,
        )
        pv = _local_mini_gemm_service(
            geometry=LocalMatmulGeometry(
                cta_m=kernel.block_q,
                cta_n=problem.value_dim,
                reduction_k=kernel.block_k,
                live_m=max_q,
                live_n=problem.value_dim,
                live_k=max_kv,
                warp_m=kernel.pv_warp_m,
                warp_n=kernel.pv_warp_n,
                warp_k=kernel.pv_warp_k,
                mma_m=kernel.mma_m,
                mma_n=kernel.mma_n,
                mma_k=kernel.mma_k,
                warps_per_cta=kernel.warps_per_cta,
                lhs_smem_resident=not kernel.probability_register_resident,
                rhs_smem_resident=True,
                input_bytes=byte_width,
            ),
            occupancy_classes=classes,
            represented_sms=represented_sms,
            peak_tensor_flops_per_cycle=peak_tensor * represented_sms / (hardware.compute.num_sms or represented_sms),
            peak_smem_bytes_per_cycle=peak_smem * represented_sms / (hardware.compute.num_sms or represented_sms),
            tensor_latency_cycles=tensor_latency,
            shared_latency_cycles=shared_latency,
        )
        v_action = _flash_kv_action(
            name=f"v_load_{iteration}",
            wave=wave,
            active=active,
            tile_index=iteration,
            problem=problem,
            kernel=kernel,
            policy=policy,
            hardware=hardware,
            clock_hz=clock_hz,
            warnings=warnings,
        )
        next_k = k_actions[iteration + 1] if iteration + 1 < len(k_actions) else None
        segment_a = max(v_action.total_cycles, qk.total_cycles + softmax.compute.total_cycles)
        segment_b = max(pv.total_cycles, next_k.total_cycles) if next_k else pv.total_cycles
        iterations.append(
            FlashAttentionIterationResult(
                index=iteration,
                active_ctas=len(active),
                qk=qk,
                softmax=softmax,
                pv=pv,
                k_load=k_actions[iteration],
                v_load=v_action,
                next_k_load=next_k,
                qk_softmax_or_v_cycles=segment_a,
                pv_or_next_k_cycles=segment_b,
                total_cycles=segment_a + segment_b,
            )
        )
    first_k_cycles = k_actions[0].total_cycles if k_actions else 0.0
    prologue = q_action.total_cycles + first_k_cycles
    body = sum(item.total_cycles for item in iterations)
    output_bytes = sum(
        _round_transaction(cta.live_query_rows * problem.value_dim * dtype_nbytes(problem.output_dtype))
        for cta in wave.ctas
    )
    lse_bytes = (
        sum(_round_transaction(cta.live_query_rows * 4) for cta in wave.ctas)
        if problem.store_lse
        else 0.0
    )
    store_action = _flash_memory_action_service(
        name="output_lse_store",
        l2_bytes=output_bytes + lse_bytes,
        hbm_bytes=output_bytes + lse_bytes,
        smem_write_bytes=output_bytes if kernel.epilogue_through_smem else 0.0,
        active_ctas=len(wave.ctas),
        bytes_per_cta_stage=(output_bytes + lse_bytes) / len(wave.ctas),
        pipeline_slots=1,
        hardware=hardware,
        clock_hz=clock_hz,
        warnings=warnings,
    )
    epilogue_cuda_ops = sum(
        cta.live_query_rows * problem.value_dim for cta in wave.ctas
    ) + (sum(cta.live_query_rows for cta in wave.ctas) * 2 if problem.store_lse else 0)
    cuda_rate = _cuda_fp32_ops_per_cycle(hardware, clock_hz, warnings)
    epilogue_compute = epilogue_cuda_ops / cuda_rate if cuda_rate else 0.0
    epilogue = epilogue_compute + store_action.total_cycles
    counts = tuple(len(items) for items in active_by_iteration)
    if any(later > earlier for earlier, later in zip(counts, counts[1:])):
        raise AssertionError("causal FlashAttention active CTA counts must be nonincreasing")
    return FlashAttentionWaveResult(
        index=wave.index,
        active_ctas=len(wave.ctas),
        active_ctas_by_iteration=counts,
        prologue_cycles=prologue,
        body_cycles=body,
        epilogue_cycles=epilogue,
        total_cycles=prologue + body + epilogue,
        iterations=tuple(iterations),
    )


def _flash_traffic(
    *,
    waves: tuple[FlashAttentionWave, ...],
    problem: FlashAttentionProblemSpec,
    policy: FlashKVReusePolicy,
) -> tuple[MemoryAccess, dict[str, float]]:
    byte_width = dtype_nbytes(problem.input_dtype)
    q = k_l2 = v_l2 = k_hbm = v_hbm = o = lse = 0.0
    unique_k: dict[tuple[int, int, int], float] = {}
    unique_v: dict[tuple[int, int, int], float] = {}
    for wave in waves:
        q += sum(_round_transaction(cta.live_query_rows * problem.head_dim * byte_width) for cta in wave.ctas)
        o += sum(_round_transaction(cta.live_query_rows * problem.value_dim * dtype_nbytes(problem.output_dtype)) for cta in wave.ctas)
        if problem.store_lse:
            lse += sum(_round_transaction(cta.live_query_rows * 4) for cta in wave.ctas)
        max_iterations = max((len(cta.kv_tiles) for cta in wave.ctas), default=0)
        for iteration in range(max_iterations):
            active = tuple(cta for cta in wave.ctas if iteration < len(cta.kv_tiles))
            k_sizes = [_round_transaction(cta.kv_tiles[iteration].live_kv_rows * problem.head_dim * byte_width) for cta in active]
            v_sizes = [_round_transaction(cta.kv_tiles[iteration].live_kv_rows * problem.value_dim * byte_width) for cta in active]
            for cta, kb, vb in zip(active, k_sizes, v_sizes):
                unique_key = (cta.batch_index, cta.kv_head, iteration)
                unique_k[unique_key] = max(unique_k.get(unique_key, 0.0), kb)
                unique_v[unique_key] = max(unique_v.get(unique_key, 0.0), vb)
            k_l2 += sum(k_sizes)
            v_l2 += sum(v_sizes)
            if policy == FlashKVReusePolicy.NONE:
                k_hbm += sum(k_sizes)
                v_hbm += sum(v_sizes)
            else:
                seen: set[tuple[int, ...]] = set()
                group_size = 4 if policy == FlashKVReusePolicy.IDEAL_WITHIN_SWIZZLE_GROUP else 10**9
                for cta, kb, vb in zip(active, k_sizes, v_sizes):
                    key = (wave.index, cta.batch_index, cta.kv_head, iteration, cta.query_block // group_size)
                    if key not in seen:
                        seen.add(key)
                        k_hbm += kb
                        v_hbm += vb
    if not (sum(unique_k.values()) <= k_hbm <= k_l2):
        raise AssertionError("FlashAttention K HBM traffic is outside reuse bounds")
    if not (sum(unique_v.values()) <= v_hbm <= v_l2):
        raise AssertionError("FlashAttention V HBM traffic is outside reuse bounds")
    memory = MemoryAccess(
        hbm_read_bytes=int(q + k_hbm + v_hbm),
        hbm_write_bytes=int(o + lse),
        l2_read_bytes=int(q + k_l2 + v_l2),
        l2_write_bytes=int(o + lse),
    )
    return memory, {
        "q_l2": q,
        "q_hbm": q,
        "k_l2_requested": k_l2,
        "k_hbm": k_hbm,
        "k_hbm_unique_lower_bound": sum(unique_k.values()),
        "k_hbm_requested_upper_bound": k_l2,
        "v_l2_requested": v_l2,
        "v_hbm": v_hbm,
        "v_hbm_unique_lower_bound": sum(unique_v.values()),
        "v_hbm_requested_upper_bound": v_l2,
        "o_l2": o,
        "o_hbm": o,
        "lse_l2": lse,
        "lse_hbm": lse,
        "score_l2": 0.0,
        "score_hbm": 0.0,
        "probability_l2": 0.0,
        "probability_hbm": 0.0,
    }


def _flash_pipeline_bottlenecks(
    waves: tuple[FlashAttentionWaveResult, ...],
) -> tuple[str, tuple[str, ...]]:
    contribution = {
        name: 0.0
        for name in (
            "qk_tensor",
            "pv_tensor",
            "smem",
            "cuda_softmax",
            "sfu",
            "l2",
            "hbm",
        )
    }

    def memory_resource(action: FlashMemoryActionResult) -> str:
        return "hbm" if action.hbm_cycles >= action.l2_cycles else "l2"

    for wave in waves:
        for iteration in wave.iterations:
            qk_softmax = iteration.qk.total_cycles + iteration.softmax.compute.total_cycles
            if iteration.v_load.total_cycles >= qk_softmax:
                contribution[memory_resource(iteration.v_load)] += iteration.qk_softmax_or_v_cycles
            elif iteration.qk.total_cycles >= iteration.softmax.compute.total_cycles:
                resource = "smem" if iteration.qk.smem_cycles >= iteration.qk.tensor_cycles else "qk_tensor"
                contribution[resource] += iteration.qk_softmax_or_v_cycles
            else:
                softmax = iteration.softmax.compute
                sfu_service = (
                    softmax.exp_ops / softmax.sfu_effective_ops_per_cycle
                    if softmax.sfu_effective_ops_per_cycle
                    else 0.0
                )
                cuda_service = (
                    softmax.cuda_scalar_ops / softmax.cuda_effective_ops_per_cycle
                    if softmax.cuda_effective_ops_per_cycle
                    else 0.0
                )
                contribution["sfu" if sfu_service >= cuda_service else "cuda_softmax"] += iteration.qk_softmax_or_v_cycles
            if (
                iteration.next_k_load is not None
                and iteration.next_k_load.total_cycles >= iteration.pv.total_cycles
            ):
                contribution[memory_resource(iteration.next_k_load)] += iteration.pv_or_next_k_cycles
            else:
                resource = "smem" if iteration.pv.smem_cycles >= iteration.pv.tensor_cycles else "pv_tensor"
                contribution[resource] += iteration.pv_or_next_k_cycles
    ordered = sorted(contribution, key=lambda name: contribution[name], reverse=True)
    primary = ordered[0] if contribution[ordered[0]] > 0.0 else "none"
    secondary = tuple(name for name in ordered[1:] if contribution[name] > 0.0)
    return primary, secondary


class EffectiveFlashAttentionEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        problem = _parse_flash_attention_problem(op)
        kernel = _ampere_flash_attention_kernel(problem, op.attrs)
        if problem.input_dtype not in hardware.compute.tensor_flops_per_s:
            raise ValueError("hardware lacks Tensor Core throughput for FlashAttention dtype")
        warnings: list[str] = []
        clock_hz = extended._clock_hz(hardware, warnings)
        resident = _resident_ctas_per_sm(
            warps_per_cta=kernel.warps_per_cta,
            threads_per_cta=kernel.threads_per_cta,
            registers_per_thread=kernel.registers_per_thread,
            shared_memory_bytes_per_cta=kernel.shared_memory_bytes_per_cta,
            explicit_limit=kernel.max_concurrent_ctas_per_sm,
            hardware=hardware,
        )
        ctas = _flash_attention_ctas(problem, kernel)
        waves = _flash_attention_waves(ctas, resident, hardware)
        try:
            policy = FlashKVReusePolicy(str(op.attrs.get("kv_reuse_policy", "ideal_within_wave")))
        except ValueError as exc:
            raise ValueError("invalid FlashAttention kv_reuse_policy") from exc
        wave_results = tuple(
            _ampere_flash_attention_wave(
                wave=wave,
                problem=problem,
                kernel=kernel,
                policy=policy,
                hardware=hardware,
                clock_hz=clock_hz,
                warnings=warnings,
            )
            for wave in waves
        )
        timeline = FlashAttentionTimelineResult(
            kernel_cycles=sum(item.total_cycles for item in wave_results),
            prologue_cycles=sum(item.prologue_cycles for item in wave_results),
            body_cycles=sum(item.body_cycles for item in wave_results),
            epilogue_cycles=sum(item.epilogue_cycles for item in wave_results),
            waves=wave_results,
        )
        fixed = float(hardware.compute.device_fixed_overhead_cycles or 0)
        latency_s = (timeline.kernel_cycles + fixed) / clock_hz
        memory, traffic = _flash_traffic(waves=waves, problem=problem, policy=policy)
        useful_scores = sum(tile.useful_score_elements for cta in ctas for tile in cta.kv_tiles)
        issued_scores = sum(tile.issued_score_elements for cta in ctas for tile in cta.kv_tiles)
        qk_flops = float(2 * useful_scores * problem.head_dim)
        pv_flops = float(2 * useful_scores * problem.value_dim)
        flops = qk_flops + pv_flops
        smem_read = sum(
            iteration.qk.smem_read_bytes + iteration.pv.smem_read_bytes
            for wave_result in wave_results
            for iteration in wave_result.iterations
        )
        memory = replace(memory, sram_read_bytes=int(smem_read))
        energy = estimate_energy(
            flops=flops,
            memory_access=memory,
            engine=EngineKind.TENSOR,
            dtype=problem.input_dtype,
            hardware=hardware,
            latency_s=latency_s,
        )
        active_tensor = sum(
            iteration.qk.tensor_cycles + iteration.pv.tensor_cycles
            for result in wave_results for iteration in result.iterations
        )
        active_smem = sum(
            iteration.qk.smem_cycles + iteration.pv.smem_cycles
            for result in wave_results for iteration in result.iterations
        )
        primary, secondary = _flash_pipeline_bottlenecks(wave_results)
        l2_peak = hardware.memory_levels.get("l2")
        diagnostics = {
            "problem": asdict(problem),
            "kernel": asdict(kernel),
            "causal_grid": {
                "cta_count": len(ctas),
                "query_blocks": extended._ceil_div(problem.seq_q, kernel.block_q),
                "kv_iteration_histogram": dict(
                    (count, sum(len(cta.kv_tiles) == count for cta in ctas))
                    for count in sorted({len(cta.kv_tiles) for cta in ctas})
                ),
            },
            "occupancy": {
                "resident_ctas_per_sm": resident,
                "ctas_per_wave": (hardware.compute.num_sms or 1) * resident,
                "wave_count": len(waves),
            },
            "waves": tuple(asdict(item) for item in wave_results),
            "useful_work": {
                "score_elements": useful_scores,
                "qk_flops": qk_flops,
                "pv_flops": pv_flops,
            },
            "issued_work": {
                "score_elements": issued_scores,
                "attention_pair_efficiency": useful_scores / issued_scores if issued_scores else 0.0,
                "qk_flops": sum(i.qk.issued_flops for w in wave_results for i in w.iterations),
                "pv_flops": sum(i.pv.issued_flops for w in wave_results for i in w.iterations),
            },
            "transaction_bytes": {**traffic, "smem_read": smem_read, "smem_write": 0.0},
            "kv_reuse_policy": policy.value,
            "raw_rates": {
                "tensor_flops_per_cycle": hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz,
                "l2_bytes_per_cycle": l2_peak.bandwidth_bytes_per_s / clock_hz if l2_peak else 0.0,
                "hbm_bytes_per_cycle": hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz,
            },
            "effective_rates": {
                "qk_tensor": flops / active_tensor if active_tensor else 0.0,
                "smem": smem_read / active_smem if active_smem else 0.0,
            },
            "active_cycles": {
                "compute": active_tensor,
                "smem": active_smem,
                "l2": (memory.l2_read_bytes + memory.l2_write_bytes) / (l2_peak.bandwidth_bytes_per_s / clock_hz) if l2_peak else 0.0,
                "dram": (memory.hbm_read_bytes + memory.hbm_write_bytes) / (hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz),
            },
            "service_cycles": {
                "tensor": active_tensor,
                "cuda_softmax": sum(i.softmax.compute.total_cycles for w in wave_results for i in w.iterations),
                "smem": active_smem,
            },
            "phase_cycles": {
                "prologue": timeline.prologue_cycles,
                "body": timeline.body_cycles,
                "epilogue": timeline.epilogue_cycles,
                "total": timeline.kernel_cycles,
            },
            "pipeline_segments": tuple(
                {
                    "wave": w.index,
                    "iteration": i.index,
                    "max_v_or_qk_softmax": i.qk_softmax_or_v_cycles,
                    "max_pv_or_next_k": i.pv_or_next_k_cycles,
                }
                for w in wave_results for i in w.iterations
            ),
            "cta_waves": len(waves),
            "resident_ctas_per_sm": resident,
            "ctas_per_wave": (hardware.compute.num_sms or 1) * resident,
            "wave_shape": {"last_wave_ctas": len(waves[-1].ctas) if waves else 0},
            "wave_pipeline": {
                "full": asdict(wave_results[0]) if wave_results else {},
                "last": asdict(wave_results[-1]) if wave_results else {},
            },
            "clock_hz": clock_hz,
            "primary_bottleneck": primary,
            "secondary_bottlenecks": secondary,
            "warnings": tuple(dict.fromkeys(warnings)),
            "assumptions": (
                "ampere_causal_forward_only",
                "iteration_synchronous_waves",
                "two_stage_cp_async",
                "persistent_concurrency_limited_bandwidth",
                "conservative_synchronized_tensor_cuda_waves",
                "cuda_sfu_energy_not_separately_attributed",
                "no_global_score_or_probability_materialization",
            ),
        }
        profile = OpProfile(
            latency_s=latency_s,
            energy_j=energy.total_j,
            flops=flops,
            engine=EngineKind.TENSOR,
            footprint=footprint_from_tensors(op),
            memory_access=memory,
            energy_breakdown=energy,
            implementation="effective_roofline.flash_attention_ampere_causal",
            diagnostics=diagnostics,
        )
        return apply_calibrated_energy_model(profile, hardware)


def _implementation_name(batched: bool) -> str:
    return "effective_roofline.batched_gemm" if batched else "effective_roofline.gemm"


__all__ = [
    "EffectiveRooflineModel",
    "EffectiveGemmEstimator",
    "EffectiveFlashAttentionEstimator",
    "EffectiveSoftmaxEstimator",
    "FlashAttentionProblemSpec",
    "AmpereFlashAttentionKernelSpec",
    "FlashAttentionKVTile",
    "FlashAttentionCTADescriptor",
    "FlashAttentionWave",
    "FlashKVReusePolicy",
    "LocalMatmulGeometry",
    "LocalMiniGemmResult",
    "SoftmaxMode",
    "CudaSoftmaxWorkSpec",
    "CudaSoftmaxKernelSpec",
    "CudaSoftmaxComputeResult",
    "StandaloneSoftmaxProblemSpec",
    "StandaloneSoftmaxKernelSpec",
    "StandaloneSoftmaxWaveResult",
    "StandaloneSoftmaxTimelineResult",
    "FlashSoftmaxStageResult",
    "FlashMemoryActionResult",
    "FlashAttentionIterationResult",
    "FlashAttentionWaveResult",
    "FlashAttentionTimelineResult",
    "WaveRooflineResult",
    "OccupancyClassResult",
    "MemoryWindowResult",
    "EpilogueRooflineResult",
    "EffectiveTimelineResult",
    "evaluate_gemm_template_candidates",
    "select_gemm_template_candidate",
]
