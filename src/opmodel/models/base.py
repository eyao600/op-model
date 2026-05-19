from __future__ import annotations

import math
from dataclasses import dataclass

from opmodel.api import (
    DType,
    EngineKind,
    GlobalFootprint,
    LocalOp,
    MemoryAccess,
    OpKind,
    OpProfile,
    TensorRole,
)
from opmodel.hardware import HardwareSpec
from opmodel.models.roofline import (
    RooflineModel,
    _make_profile,
    _matmul_engine,
    _vector_engine,
)
from opmodel.ops import (
    dtype_nbytes,
    footprint_from_tensors,
    get_tensors,
    numel,
    parse_batched_gemm,
    parse_gemm,
    require_one_tensor,
)


class BaseModel(RooflineModel):
    def __init__(self) -> None:
        super().__init__(
            {
                OpKind.GEMM: BaseGemmEstimator(),
                OpKind.BATCHED_GEMM: BaseBatchedGemmEstimator(),
                OpKind.SOFTMAX: BaseSoftmaxEstimator(),
            }
        )


class BaseGemmEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        m, n, k, dtype = parse_gemm(op)
        return _estimate_tiled_gemm(
            op=op,
            hardware=hardware,
            batch=1,
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            weight_is_batched=True,
            implementation="base.gemm",
        )


class BaseBatchedGemmEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        batch, m, n, k, dtype = parse_batched_gemm(op)
        weights = get_tensors(op, TensorRole.WEIGHT)
        weight_is_batched = bool(weights and len(weights[0].shape) == 3)
        return _estimate_tiled_gemm(
            op=op,
            hardware=hardware,
            batch=batch,
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            weight_is_batched=weight_is_batched,
            implementation="base.batched_gemm",
        )


class BaseSoftmaxEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        input_tensor = require_one_tensor(op, TensorRole.INPUT, "input")
        if not input_tensor.shape:
            raise ValueError("Softmax requires at least one tensor dimension")
        row_size = int(op.attrs.get("row_size", input_tensor.shape[-1]))
        rows = numel(input_tensor.shape) // row_size
        elements = rows * row_size
        dtype_bytes = dtype_nbytes(input_tensor.dtype)
        flops = float(7 * elements)
        hbm_read = _ceil_bytes(3 * elements, input_tensor.dtype)
        hbm_write = _ceil_bytes(elements, input_tensor.dtype)
        footprint = footprint_from_tensors(op)
        return _make_profile(
            hardware=hardware,
            dtype=input_tensor.dtype,
            flops=flops,
            hbm_read=hbm_read,
            hbm_write=hbm_write,
            engine=_vector_engine(input_tensor.dtype, hardware),
            footprint=footprint,
            implementation="base.softmax",
            diagnostics={
                "row_size": row_size,
                "rows": rows,
                "flops_per_element": 7,
                "memory_accesses_per_element": 4,
                "dtype_bytes": dtype_bytes,
            },
        )


@dataclass(frozen=True)
class _TileShape:
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class _TileEvaluation:
    tile: _TileShape
    blocks_m: int
    blocks_n: int
    blocks_k: int
    threadblock_count: int
    wave_count: int
    partial_wave_utilization: float
    throughput_derate: float
    memory_access: MemoryAccess
    profile: OpProfile

    @property
    def total_memory_traffic_bytes(self) -> int:
        return _total_memory_bytes(self.memory_access)


def _estimate_tiled_gemm(
    *,
    op: LocalOp,
    hardware: HardwareSpec,
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: DType,
    weight_is_batched: bool,
    implementation: str,
) -> OpProfile:
    footprint = footprint_from_tensors(op)
    engine = _matmul_engine(dtype, hardware)
    flops = float(2 * batch * m * n * k)
    if engine != EngineKind.TENSOR:
        return _make_profile(
            hardware=hardware,
            dtype=dtype,
            flops=flops,
            hbm_read=footprint.input_bytes + footprint.weight_bytes,
            hbm_write=footprint.output_bytes,
            engine=engine,
            footprint=footprint,
            implementation=implementation,
            diagnostics={"tensor_fallback": engine.value},
        )

    dataflow = str(op.attrs.get("dataflow", hardware.compute.dataflow or "output_stationary"))
    evaluations: list[_TileEvaluation] = []
    for tile in _candidate_tiles(m, n, k, dtype, hardware):
        blocks_m = math.ceil(m / tile.m)
        blocks_n = math.ceil(n / tile.n)
        blocks_k = math.ceil(k / tile.k)
        threadblocks = batch * blocks_m * blocks_n
        wave_info = _wave_info(threadblocks, hardware)
        memory_access = _gemm_memory_access(
            footprint=footprint,
            batch=batch,
            blocks_m=blocks_m,
            blocks_n=blocks_n,
            dtype=dtype,
            m=m,
            n=n,
            k=k,
            hardware=hardware,
            weight_is_batched=weight_is_batched,
        )
        diagnostics = {
            "selected_tile_shape": {"m": tile.m, "n": tile.n, "k": tile.k},
            "dataflow": dataflow,
            "blocks_m": blocks_m,
            "blocks_n": blocks_n,
            "blocks_k": blocks_k,
            "threadblock_count": threadblocks,
            "wave_count": wave_info.wave_count,
            "partial_wave_utilization": wave_info.partial_wave_utilization,
            "throughput_derate": wave_info.throughput_derate,
            "effective_compute_utilization": hardware.utilization.tensor
            * wave_info.throughput_derate,
            "num_sms": wave_info.num_sms,
        }
        profile = _make_profile(
            hardware=hardware,
            dtype=dtype,
            flops=flops,
            memory_access=memory_access,
            engine=engine,
            footprint=footprint,
            implementation=implementation,
            compute_utilization_override=hardware.utilization.tensor
            * wave_info.throughput_derate,
            diagnostics=diagnostics,
        )
        evaluations.append(
            _TileEvaluation(
                tile=tile,
                blocks_m=blocks_m,
                blocks_n=blocks_n,
                blocks_k=blocks_k,
                threadblock_count=threadblocks,
                wave_count=wave_info.wave_count,
                partial_wave_utilization=wave_info.partial_wave_utilization,
                throughput_derate=wave_info.throughput_derate,
                memory_access=memory_access,
                profile=profile,
            )
        )

    best = min(
        evaluations,
        key=lambda item: (item.profile.latency_s, item.total_memory_traffic_bytes),
    )
    diagnostics = dict(best.profile.diagnostics)
    diagnostics.update(
        {
            "tile_candidates_evaluated": len(evaluations),
            "selected_memory_traffic_bytes": best.total_memory_traffic_bytes,
        }
    )
    return _make_profile(
        hardware=hardware,
        dtype=dtype,
        flops=flops,
        memory_access=best.memory_access,
        engine=engine,
        footprint=footprint,
        implementation=implementation,
        compute_utilization_override=hardware.utilization.tensor * best.throughput_derate,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class _WaveInfo:
    num_sms: int
    wave_count: int
    partial_wave_utilization: float
    throughput_derate: float


def _wave_info(threadblock_count: int, hardware: HardwareSpec) -> _WaveInfo:
    num_sms = hardware.compute.num_sms or 1
    wave_count = max(1, math.ceil(threadblock_count / num_sms))
    last_wave_blocks = threadblock_count - (wave_count - 1) * num_sms
    partial_wave_utilization = last_wave_blocks / num_sms
    throughput_derate = threadblock_count / (wave_count * num_sms)
    return _WaveInfo(
        num_sms=num_sms,
        wave_count=wave_count,
        partial_wave_utilization=partial_wave_utilization,
        throughput_derate=throughput_derate,
    )


def _candidate_tiles(
    m: int, n: int, k: int, dtype: DType, hardware: HardwareSpec
) -> tuple[_TileShape, ...]:
    fma_m, fma_n, fma_k = hardware.compute.fma_dims or (16, 16, 16)
    candidates: list[_TileShape] = []
    for tile_m in _dimension_candidates(m, fma_m, (16, 32, 64, 128, 256)):
        for tile_n in _dimension_candidates(n, fma_n, (16, 32, 64, 128, 256)):
            for tile_k in _dimension_candidates(k, fma_k, (16, 32, 64, 128)):
                tile = _TileShape(tile_m, tile_n, tile_k)
                if _tile_fits_capacity(tile, dtype, hardware):
                    candidates.append(tile)
    if candidates:
        return tuple(candidates)

    fallback = _TileShape(min(m, fma_m), min(n, fma_n), min(k, fma_k))
    return (fallback,)


def _dimension_candidates(dim: int, multiple: int, preferred: tuple[int, ...]) -> tuple[int, ...]:
    values = {min(dim, max(multiple, value)) for value in preferred}
    values.add(min(dim, multiple))
    return tuple(sorted(values))


def _tile_fits_capacity(tile: _TileShape, dtype: DType, hardware: HardwareSpec) -> bool:
    operand_bytes = _ceil_scalar_bytes(
        (tile.m * tile.k + tile.k * tile.n) * dtype_nbytes(dtype)
    )
    accumulator_bytes = tile.m * tile.n * 4
    tile_bytes = operand_bytes + accumulator_bytes
    l2 = hardware.memory_levels.get("l2")
    if l2 is not None and l2.size_bytes is not None and tile_bytes > l2.size_bytes:
        return False
    sram = hardware.memory_levels.get("sram")
    if sram is not None and sram.size_bytes is not None and operand_bytes > sram.size_bytes:
        return False
    register = hardware.memory_levels.get("register")
    if (
        register is not None
        and register.size_bytes is not None
        and accumulator_bytes > register.size_bytes
    ):
        return False
    return True


def _gemm_memory_access(
    *,
    footprint: GlobalFootprint,
    batch: int,
    blocks_m: int,
    blocks_n: int,
    dtype: DType,
    m: int,
    n: int,
    k: int,
    hardware: HardwareSpec,
    weight_is_batched: bool,
) -> MemoryAccess:
    has_l2 = "l2" in hardware.memory_levels
    input_l2_reads = footprint.input_bytes * blocks_n
    weight_reuse_batches = 1 if weight_is_batched else batch
    weight_l2_reads = footprint.weight_bytes * blocks_m * weight_reuse_batches
    l2_operand_reads = input_l2_reads + weight_l2_reads

    if has_l2:
        hbm_read = footprint.input_bytes + footprint.weight_bytes
        hbm_write = footprint.output_bytes
        l2_read = l2_operand_reads
        l2_write = footprint.output_bytes
    else:
        hbm_read = input_l2_reads + weight_l2_reads
        hbm_write = footprint.output_bytes
        l2_read = None
        l2_write = None

    sram_read: int | None = None
    sram_write: int | None = None
    if "sram" in hardware.memory_levels:
        sram_read = l2_operand_reads
        sram_write = l2_operand_reads

    register_read: int | None = None
    register_write: int | None = None
    if "register" in hardware.memory_levels:
        fma_count = batch * m * n * k
        dtype_bytes = dtype_nbytes(dtype)
        register_read = _ceil_scalar_bytes(fma_count * (2 * dtype_bytes + 4.0))
        register_write = _ceil_scalar_bytes(fma_count * 4.0)

    return MemoryAccess(
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
        l2_read_bytes=l2_read,
        l2_write_bytes=l2_write,
        sram_read_bytes=sram_read,
        sram_write_bytes=sram_write,
        register_read_bytes=register_read,
        register_write_bytes=register_write,
    )


def _ceil_bytes(elements: int, dtype: DType) -> int:
    return _ceil_scalar_bytes(elements * dtype_nbytes(dtype))


def _ceil_scalar_bytes(value: float) -> int:
    return int(math.ceil(value))


def _total_memory_bytes(memory_access: MemoryAccess) -> int:
    return sum(
        value or 0
        for value in (
            memory_access.hbm_read_bytes,
            memory_access.hbm_write_bytes,
            memory_access.l2_read_bytes,
            memory_access.l2_write_bytes,
            memory_access.sram_read_bytes,
            memory_access.sram_write_bytes,
            memory_access.register_read_bytes,
            memory_access.register_write_bytes,
        )
    )
