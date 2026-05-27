from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class OpKind(str, Enum):
    GEMM = "gemm"
    BATCHED_GEMM = "batched_gemm"
    ATTENTION_PREFILL = "attention_prefill"
    ATTENTION_DECODE = "attention_decode"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    EMBEDDING = "embedding"
    COPY = "copy"


class Phase(str, Enum):
    TRAIN_FWD = "train_fwd"
    TRAIN_BWD = "train_bwd"
    PREFILL = "prefill"
    DECODE = "decode"
    INFERENCE = "inference"


class TensorRole(str, Enum):
    INPUT = "input"
    WEIGHT = "weight"
    OUTPUT = "output"
    GRAD = "grad"
    WORKSPACE = "workspace"


class DType(str, Enum):
    FP64 = "fp64"
    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


class EngineKind(str, Enum):
    TENSOR = "tensor"
    VECTOR = "vector"
    MEMORY = "memory"
    MIXED = "mixed"


@dataclass(frozen=True)
class TensorSpec:
    role: TensorRole
    shape: tuple[int, ...]
    dtype: DType
    layout: str | None = None


@dataclass(frozen=True)
class LocalOp:
    name: str
    kind: OpKind
    phase: Phase
    tensors: tuple[TensorSpec, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalFootprint:
    input_bytes: int = 0
    output_bytes: int = 0
    weight_bytes: int = 0
    workspace_bytes: int = 0
    saved_activation_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.input_bytes
            + self.output_bytes
            + self.weight_bytes
            + self.workspace_bytes
            + self.saved_activation_bytes
        )


@dataclass(frozen=True)
class MemoryAccess:
    hbm_read_bytes: int = 0
    hbm_write_bytes: int = 0
    l2_read_bytes: int | None = None
    l2_write_bytes: int | None = None
    sram_read_bytes: int | None = None
    sram_write_bytes: int | None = None
    register_read_bytes: int | None = None
    register_write_bytes: int | None = None


@dataclass(frozen=True)
class EnergyBreakdown:
    compute_j: float = 0.0
    hbm_j: float = 0.0
    l2_j: float = 0.0
    sram_j: float = 0.0
    register_j: float = 0.0
    static_j: float = 0.0

    @property
    def total_j(self) -> float:
        return (
            self.compute_j
            + self.hbm_j
            + self.l2_j
            + self.sram_j
            + self.register_j
            + self.static_j
        )


@dataclass(frozen=True)
class OpProfile:
    latency_s: float
    energy_j: float
    flops: float
    engine: EngineKind
    footprint: GlobalFootprint
    memory_access: MemoryAccess
    energy_breakdown: EnergyBreakdown
    implementation: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class OpPerformanceModel(Protocol):
    def predict(self, op: LocalOp, hardware: "HardwareSpec") -> OpProfile:
        ...
