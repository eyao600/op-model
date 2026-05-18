from dataclasses import replace
from pathlib import Path

from opmodel import DType, EngineKind, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.hardware import load_hardware
from opmodel.models.roofline import RooflineModel
from opmodel.ops import tensor_nbytes


ROOT = Path(__file__).resolve().parents[1]


def _gemm_op() -> LocalOp:
    return LocalOp(
        name="gemm",
        kind=OpKind.GEMM,
        phase=Phase.TRAIN_FWD,
        tensors=(
            TensorSpec(TensorRole.INPUT, (4, 8), DType.BF16),
            TensorSpec(TensorRole.WEIGHT, (8, 16), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, (4, 16), DType.BF16),
        ),
    )


def test_gemm_profile() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml")
    op = _gemm_op()
    profile = RooflineModel().predict(op, hardware)
    assert profile.flops == 2 * 4 * 16 * 8
    assert profile.memory_access.hbm_read_bytes == sum(tensor_nbytes(t) for t in op.tensors[:2])
    assert profile.memory_access.hbm_write_bytes == tensor_nbytes(op.tensors[2])
    assert profile.engine == EngineKind.TENSOR


def test_gemm_vector_fallback() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml")
    hardware = replace(
        hardware,
        compute=replace(
            hardware.compute,
            tensor_flops_per_s={DType.FP16: hardware.compute.tensor_flops_per_s[DType.FP16]},
        ),
    )
    profile = RooflineModel().predict(_gemm_op(), hardware)
    assert profile.engine == EngineKind.VECTOR
