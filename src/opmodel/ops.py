from __future__ import annotations

import math
from typing import Any

from opmodel.api import DType, GlobalFootprint, LocalOp, TensorRole, TensorSpec


_DTYPE_NBYTES: dict[DType, float] = {
    DType.FP64: 8.0,
    DType.FP32: 4.0,
    DType.TF32: 4.0,
    DType.FP16: 2.0,
    DType.BF16: 2.0,
    DType.FP8: 1.0,
    DType.INT8: 1.0,
    DType.INT4: 0.5,
}


def dtype_nbytes(dtype: DType) -> float:
    return _DTYPE_NBYTES[dtype]


def tensor_nbytes(tensor: TensorSpec) -> int:
    return int(math.ceil(numel(tensor.shape) * dtype_nbytes(tensor.dtype)))


def numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    total = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"Tensor dimensions must be positive, got shape {shape}")
        total *= dim
    return total


def footprint_from_tensors(op: LocalOp) -> GlobalFootprint:
    return GlobalFootprint(
        input_bytes=sum(tensor_nbytes(tensor) for tensor in get_tensors(op, TensorRole.INPUT)),
        output_bytes=sum(tensor_nbytes(tensor) for tensor in get_tensors(op, TensorRole.OUTPUT)),
        weight_bytes=sum(tensor_nbytes(tensor) for tensor in get_tensors(op, TensorRole.WEIGHT)),
        workspace_bytes=sum(
            tensor_nbytes(tensor) for tensor in get_tensors(op, TensorRole.WORKSPACE)
        ),
    )


def get_tensors(op: LocalOp, role: TensorRole) -> tuple[TensorSpec, ...]:
    return tuple(tensor for tensor in op.tensors if tensor.role == role)


def require_one_tensor(op: LocalOp, role: TensorRole, label: str | None = None) -> TensorSpec:
    tensors = get_tensors(op, role)
    if len(tensors) != 1:
        name = label or role.value
        raise ValueError(f"{op.kind.value} op {op.name!r} requires exactly one {name} tensor")
    return tensors[0]


def parse_gemm(op: LocalOp) -> tuple[int, int, int, DType]:
    """
    Return m, n, k, dtype.

    Expected tensors:
    - input activation A: shape [m, k]
    - weight B: shape [k, n]
    - output C: shape [m, n]

    Support attrs:
    - transpose_a: bool
    - transpose_b: bool
    """
    a = require_one_tensor(op, TensorRole.INPUT, "input activation")
    b = require_one_tensor(op, TensorRole.WEIGHT, "weight")
    c = require_one_tensor(op, TensorRole.OUTPUT, "output")
    if len(a.shape) != 2 or len(b.shape) != 2 or len(c.shape) != 2:
        raise ValueError("GEMM requires A [m,k], B [k,n], and C [m,n] tensors")
    if a.dtype != b.dtype:
        raise ValueError("GEMM input and weight dtypes must match")

    transpose_a = bool(op.attrs.get("transpose_a", False))
    transpose_b = bool(op.attrs.get("transpose_b", False))
    m, k_a = (a.shape[1], a.shape[0]) if transpose_a else (a.shape[0], a.shape[1])
    k_b, n = (b.shape[1], b.shape[0]) if transpose_b else (b.shape[0], b.shape[1])
    if k_a != k_b:
        raise ValueError(f"GEMM inner dimensions must match, got {k_a} and {k_b}")
    if c.shape != (m, n):
        raise ValueError(f"GEMM output shape must be {(m, n)}, got {c.shape}")
    return m, n, k_a, a.dtype


def parse_batched_gemm(op: LocalOp) -> tuple[int, int, int, int, DType]:
    a = require_one_tensor(op, TensorRole.INPUT, "input activation")
    b = require_one_tensor(op, TensorRole.WEIGHT, "weight")
    c = require_one_tensor(op, TensorRole.OUTPUT, "output")
    if len(a.shape) != 3 or len(c.shape) != 3 or len(b.shape) not in (2, 3):
        raise ValueError("Batched GEMM requires A [b,m,k], B [b,k,n] or [k,n], C [b,m,n]")
    if a.dtype != b.dtype:
        raise ValueError("Batched GEMM input and weight dtypes must match")

    transpose_a = bool(op.attrs.get("transpose_a", False))
    transpose_b = bool(op.attrs.get("transpose_b", False))
    batch = a.shape[0]
    m, k_a = (a.shape[2], a.shape[1]) if transpose_a else (a.shape[1], a.shape[2])

    if len(b.shape) == 3:
        if b.shape[0] != batch:
            raise ValueError(f"Batched GEMM weight batch must be {batch}, got {b.shape[0]}")
        k_b, n = (b.shape[2], b.shape[1]) if transpose_b else (b.shape[1], b.shape[2])
    else:
        k_b, n = (b.shape[1], b.shape[0]) if transpose_b else (b.shape[0], b.shape[1])

    if k_a != k_b:
        raise ValueError(f"Batched GEMM inner dimensions must match, got {k_a} and {k_b}")
    if c.shape != (batch, m, n):
        raise ValueError(f"Batched GEMM output shape must be {(batch, m, n)}, got {c.shape}")
    return batch, m, n, k_a, a.dtype


def parse_attention(op: LocalOp) -> dict[str, Any]:
    inputs = get_tensors(op, TensorRole.INPUT)
    outputs = get_tensors(op, TensorRole.OUTPUT)
    by_layout = {tensor.layout: tensor for tensor in inputs if tensor.layout}
    q = by_layout.get("q") or (inputs[0] if len(inputs) > 0 else None)
    k = by_layout.get("k") or (inputs[1] if len(inputs) > 1 else None)
    v = by_layout.get("v") or (inputs[2] if len(inputs) > 2 else None)
    out = outputs[0] if outputs else None

    attrs = op.attrs
    batch = _int_attr(attrs, "batch", _dim(q, 0))
    heads = _int_attr(attrs, "heads", _dim(q, 1))
    seq_q = _int_attr(attrs, "seq_q", _dim(q, 2))
    seq_kv = _int_attr(attrs, "seq_kv", _dim(k, 2))
    head_dim = _int_attr(attrs, "head_dim", _dim(q, 3))
    dtype = q.dtype if q is not None else DType(str(attrs.get("dtype", DType.BF16.value)))
    return {
        "batch": batch,
        "heads": heads,
        "seq_q": seq_q,
        "seq_kv": seq_kv,
        "head_dim": head_dim,
        "dtype": dtype,
        "q": q,
        "k": k,
        "v": v,
        "output": out,
    }


def _dim(tensor: TensorSpec | None, index: int) -> int | None:
    if tensor is None or len(tensor.shape) <= index:
        return None
    return tensor.shape[index]


def _int_attr(attrs: dict[str, Any] | Any, key: str, default: int | None) -> int:
    value = attrs.get(key, default)
    if value is None:
        raise ValueError(f"Attention op requires {key} in attrs or canonical tensor shapes")
    return int(value)
