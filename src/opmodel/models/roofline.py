from __future__ import annotations

from dataclasses import replace
from typing import Any

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
from opmodel.estimator import DispatchingOpModel
from opmodel.hardware import HardwareSpec
from opmodel.models.simple_energy import estimate_energy
from opmodel.ops import (
    dtype_nbytes,
    footprint_from_tensors,
    get_tensors,
    numel,
    parse_attention,
    parse_batched_gemm,
    parse_gemm,
    require_one_tensor,
    tensor_nbytes,
)


class RooflineModel(DispatchingOpModel):
    def __init__(self) -> None:
        super().__init__(
            {
                OpKind.GEMM: GemmEstimator(),
                OpKind.BATCHED_GEMM: BatchedGemmEstimator(),
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


class GemmEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        m, n, k, dtype = parse_gemm(op)
        inputs = get_tensors(op, TensorRole.INPUT)
        weights = get_tensors(op, TensorRole.WEIGHT)
        outputs = get_tensors(op, TensorRole.OUTPUT)
        flops = float(2 * m * n * k)
        hbm_read = sum(tensor_nbytes(tensor) for tensor in inputs + weights)
        hbm_write = sum(tensor_nbytes(tensor) for tensor in outputs)
        return _make_profile(
            hardware=hardware,
            dtype=dtype,
            flops=flops,
            hbm_read=hbm_read,
            hbm_write=hbm_write,
            engine=_matmul_engine(dtype, hardware),
            footprint=footprint_from_tensors(op),
            implementation="roofline.gemm",
        )


class BatchedGemmEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        batch, m, n, k, dtype = parse_batched_gemm(op)
        flops = float(2 * batch * m * n * k)
        footprint = footprint_from_tensors(op)
        return _make_profile(
            hardware=hardware,
            dtype=dtype,
            flops=flops,
            hbm_read=footprint.input_bytes + footprint.weight_bytes,
            hbm_write=footprint.output_bytes,
            engine=_matmul_engine(dtype, hardware),
            footprint=footprint,
            implementation="roofline.batched_gemm",
        )


class ElementwiseEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        output = require_one_tensor(op, TensorRole.OUTPUT, "output")
        op_count = float(op.attrs.get("op_count_per_element", 1))
        flops = float(numel(output.shape)) * op_count
        footprint = footprint_from_tensors(op)
        return _make_profile(
            hardware=hardware,
            dtype=output.dtype,
            flops=flops,
            hbm_read=footprint.input_bytes,
            hbm_write=footprint.output_bytes,
            engine=_vector_engine(output.dtype, hardware),
            footprint=footprint,
            implementation="roofline.elementwise",
        )


class ReductionEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        input_tensor = require_one_tensor(op, TensorRole.INPUT, "input")
        op_count = float(op.attrs.get("op_count_per_reduced_element", 1))
        flops = float(numel(input_tensor.shape)) * op_count
        footprint = footprint_from_tensors(op)
        return _make_profile(
            hardware=hardware,
            dtype=input_tensor.dtype,
            flops=flops,
            hbm_read=footprint.input_bytes,
            hbm_write=footprint.output_bytes,
            engine=_vector_engine(input_tensor.dtype, hardware),
            footprint=footprint,
            implementation="roofline.reduction",
        )


class NormEstimator:
    def __init__(self, *, layernorm: bool) -> None:
        self._layernorm = layernorm

    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        input_tensor = require_one_tensor(op, TensorRole.INPUT, "input")
        elements = numel(input_tensor.shape)
        affine = bool(op.attrs.get("affine", True))
        base_flops_per_element = 7 if self._layernorm else 5
        if affine:
            base_flops_per_element += 2
        flops = float(elements * base_flops_per_element)
        footprint = footprint_from_tensors(op)
        implementation = "roofline.layernorm" if self._layernorm else "roofline.rmsnorm"
        return _make_profile(
            hardware=hardware,
            dtype=input_tensor.dtype,
            flops=flops,
            hbm_read=footprint.input_bytes + footprint.weight_bytes,
            hbm_write=footprint.output_bytes,
            engine=_vector_engine(input_tensor.dtype, hardware),
            footprint=footprint,
            implementation=implementation,
            diagnostics={"normalized_shape": op.attrs.get("normalized_shape")},
        )


class SoftmaxEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        input_tensor = require_one_tensor(op, TensorRole.INPUT, "input")
        if not input_tensor.shape:
            raise ValueError("Softmax requires at least one tensor dimension")
        row_size = int(op.attrs.get("row_size", input_tensor.shape[-1]))
        rows = numel(input_tensor.shape) // row_size
        exp_flops = float(op.attrs.get("exp_flops", 4))
        flops = float(rows * row_size) * (exp_flops + 4.0)
        footprint = footprint_from_tensors(op)
        return _make_profile(
            hardware=hardware,
            dtype=input_tensor.dtype,
            flops=flops,
            hbm_read=footprint.input_bytes,
            hbm_write=footprint.output_bytes,
            engine=_vector_engine(input_tensor.dtype, hardware),
            footprint=footprint,
            implementation="roofline.softmax",
            diagnostics={"row_size": row_size, "rows": rows},
        )


class AttentionEstimator:
    def __init__(self, *, prefill: bool) -> None:
        self._prefill = prefill

    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        parsed = parse_attention(op)
        batch = parsed["batch"]
        heads = parsed["heads"]
        seq_q = parsed["seq_q"]
        seq_kv = parsed["seq_kv"]
        head_dim = parsed["head_dim"]
        dtype = parsed["dtype"]

        work_multiplier = 1.0
        if self._prefill and bool(op.attrs.get("causal", False)) and seq_q == seq_kv:
            work_multiplier = float(op.attrs.get("causal_work_multiplier", 0.5))

        attention_pairs = batch * heads * seq_q * seq_kv
        qk_flops = 2.0 * attention_pairs * head_dim * work_multiplier
        av_flops = 2.0 * attention_pairs * head_dim * work_multiplier
        softmax_flops = 8.0 * attention_pairs * work_multiplier
        total_flops = qk_flops + softmax_flops + av_flops

        q_bytes = tensor_nbytes(parsed["q"]) if parsed["q"] is not None else 0
        k_bytes = tensor_nbytes(parsed["k"]) if parsed["k"] is not None else 0
        v_bytes = tensor_nbytes(parsed["v"]) if parsed["v"] is not None else 0
        output_bytes = tensor_nbytes(parsed["output"]) if parsed["output"] is not None else 0

        flash = bool(op.attrs.get("flash_attention", False))
        score_bytes = int(attention_pairs * work_multiplier * dtype_nbytes(dtype))
        score_hbm_traffic_factor = 0.0 if flash else float(
            op.attrs.get("score_hbm_traffic_factor", 1.0)
        )
        workspace_bytes = 0 if flash else score_bytes
        score_traffic = int(score_bytes * score_hbm_traffic_factor)

        footprint = GlobalFootprint(
            input_bytes=q_bytes + k_bytes + v_bytes,
            output_bytes=output_bytes,
            workspace_bytes=workspace_bytes,
        )
        diagnostics: dict[str, Any] = {
            "qk_flops": qk_flops,
            "softmax_flops": softmax_flops,
            "av_flops": av_flops,
            "work_multiplier": work_multiplier,
            "flash_attention": flash,
            "score_hbm_traffic_factor": score_hbm_traffic_factor,
        }
        if not self._prefill:
            diagnostics["kv_cache_read_bytes"] = k_bytes + v_bytes

        implementation = "roofline.attention_prefill" if self._prefill else "roofline.attention_decode"
        return _make_profile(
            hardware=hardware,
            dtype=dtype,
            flops=total_flops,
            hbm_read=q_bytes + k_bytes + v_bytes + score_traffic,
            hbm_write=output_bytes + score_traffic,
            engine=_matmul_engine(dtype, hardware),
            footprint=footprint,
            implementation=implementation,
            diagnostics=diagnostics,
        )


class CopyEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        footprint = footprint_from_tensors(op)
        dtype = _first_dtype(op, default=DType.BF16)
        return _make_profile(
            hardware=hardware,
            dtype=dtype,
            flops=0.0,
            hbm_read=footprint.input_bytes,
            hbm_write=footprint.output_bytes,
            engine=EngineKind.MEMORY,
            footprint=footprint,
            implementation="roofline.copy",
        )


class EmbeddingEstimator:
    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        weights = get_tensors(op, TensorRole.WEIGHT)
        outputs = get_tensors(op, TensorRole.OUTPUT)
        if not weights or not outputs:
            raise ValueError("Embedding requires weight and output tensors")
        weight = weights[0]
        output = outputs[0]
        embedding_dim = int(op.attrs.get("embedding_dim", weight.shape[-1]))
        gather_count = int(op.attrs.get("gather_count", numel(output.shape) // embedding_dim))
        gathered_weight_bytes = int(gather_count * embedding_dim * dtype_nbytes(weight.dtype))
        footprint = footprint_from_tensors(op)
        return _make_profile(
            hardware=hardware,
            dtype=weight.dtype,
            flops=0.0,
            hbm_read=gathered_weight_bytes,
            hbm_write=footprint.output_bytes,
            engine=EngineKind.MEMORY,
            footprint=footprint,
            implementation="roofline.embedding",
            diagnostics={"gathered_weight_bytes": gathered_weight_bytes},
        )


def _make_profile(
    *,
    hardware: HardwareSpec,
    dtype: DType,
    flops: float,
    hbm_read: int,
    hbm_write: int,
    engine: EngineKind,
    footprint: GlobalFootprint,
    implementation: str,
    diagnostics: dict[str, Any] | None = None,
) -> OpProfile:
    hbm_bw = _effective_hbm_bandwidth(hardware)
    memory_latency = (hbm_read + hbm_write) / hbm_bw
    effective_flops = _effective_flops(dtype, engine, hardware)
    compute_latency = 0.0 if flops == 0.0 else flops / effective_flops
    latency = max(compute_latency, memory_latency)
    memory_access = MemoryAccess(hbm_read_bytes=hbm_read, hbm_write_bytes=hbm_write)
    energy_breakdown = estimate_energy(
        flops=flops,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
        engine=engine,
        dtype=dtype,
        hardware=hardware,
    )
    traffic = hbm_read + hbm_write
    base_diagnostics = {
        "compute_latency_s": compute_latency,
        "memory_latency_s": memory_latency,
        "effective_flops_per_s": effective_flops,
        "effective_hbm_bandwidth_bytes_per_s": hbm_bw,
        "arithmetic_intensity_flops_per_byte": (flops / traffic) if traffic else None,
    }
    if diagnostics:
        base_diagnostics.update(diagnostics)
    return OpProfile(
        latency_s=latency,
        energy_j=energy_breakdown.total_j,
        flops=flops,
        engine=engine,
        footprint=footprint,
        memory_access=memory_access,
        energy_breakdown=energy_breakdown,
        implementation=implementation,
        diagnostics=base_diagnostics,
    )


def _effective_hbm_bandwidth(hardware: HardwareSpec) -> float:
    hbm = hardware.memory_levels["hbm"]
    return hbm.bandwidth_bytes_per_s * hardware.utilization.memory.get("hbm", 1.0)


def _effective_flops(dtype: DType, engine: EngineKind, hardware: HardwareSpec) -> float:
    if engine == EngineKind.TENSOR:
        return hardware.compute.tensor_flops_per_s[dtype] * hardware.utilization.tensor
    if engine == EngineKind.VECTOR:
        return hardware.compute.vector_flops_per_s[dtype] * hardware.utilization.vector
    if engine == EngineKind.MEMORY:
        return float("inf")
    raise ValueError(f"Unsupported engine for v0 roofline latency: {engine}")


def _matmul_engine(dtype: DType, hardware: HardwareSpec) -> EngineKind:
    if dtype in hardware.compute.tensor_flops_per_s:
        return EngineKind.TENSOR
    if dtype in hardware.compute.vector_flops_per_s:
        return EngineKind.VECTOR
    raise ValueError(f"No tensor or vector throughput for dtype {dtype.value}")


def _vector_engine(dtype: DType, hardware: HardwareSpec) -> EngineKind:
    if dtype not in hardware.compute.vector_flops_per_s:
        raise ValueError(f"No vector throughput for dtype {dtype.value}")
    return EngineKind.VECTOR


def _first_dtype(op: LocalOp, *, default: DType) -> DType:
    if op.tensors:
        return op.tensors[0].dtype
    return default
