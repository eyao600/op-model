from pathlib import Path

from opmodel import DType, EngineKind, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.hardware import load_hardware
from opmodel.models.roofline import RooflineModel
from opmodel.ops import tensor_nbytes


ROOT = Path(__file__).resolve().parents[1]


def test_elementwise_profile() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml")
    op = LocalOp(
        name="gelu",
        kind=OpKind.ELEMENTWISE,
        phase=Phase.TRAIN_FWD,
        attrs={"op_count_per_element": 8},
        tensors=(
            TensorSpec(TensorRole.INPUT, (2, 4), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, (2, 4), DType.BF16),
        ),
    )
    profile = RooflineModel().predict(op, hardware)
    assert profile.flops == 2 * 4 * 8
    assert profile.engine == EngineKind.VECTOR
    assert profile.memory_access.hbm_read_bytes == tensor_nbytes(op.tensors[0])
    assert profile.memory_access.hbm_write_bytes == tensor_nbytes(op.tensors[1])
