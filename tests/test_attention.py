from pathlib import Path

from opmodel import DType, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.hardware import load_hardware
from opmodel.models.roofline import RooflineModel
from opmodel.ops import tensor_nbytes


ROOT = Path(__file__).resolve().parents[1]


def test_attention_prefill_causal_flash() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml")
    op = LocalOp(
        name="attn",
        kind=OpKind.ATTENTION_PREFILL,
        phase=Phase.PREFILL,
        attrs={
            "batch": 1,
            "heads": 2,
            "seq_q": 4,
            "seq_kv": 4,
            "head_dim": 8,
            "causal": True,
            "flash_attention": True,
        },
        tensors=(
            TensorSpec(TensorRole.INPUT, (1, 2, 4, 8), DType.BF16, layout="q"),
            TensorSpec(TensorRole.INPUT, (1, 2, 4, 8), DType.BF16, layout="k"),
            TensorSpec(TensorRole.INPUT, (1, 2, 4, 8), DType.BF16, layout="v"),
            TensorSpec(TensorRole.OUTPUT, (1, 2, 4, 8), DType.BF16),
        ),
    )
    profile = RooflineModel().predict(op, hardware)
    pairs = 1 * 2 * 4 * 4
    expected = (2 * pairs * 8 + 8 * pairs + 2 * pairs * 8) * 0.5
    assert profile.flops == expected
    assert profile.footprint.workspace_bytes == 0
    assert profile.diagnostics["flash_attention"] is True


def test_attention_decode_reports_kv_cache_reads() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml")
    k = TensorSpec(TensorRole.INPUT, (1, 2, 16, 8), DType.BF16, layout="k")
    v = TensorSpec(TensorRole.INPUT, (1, 2, 16, 8), DType.BF16, layout="v")
    op = LocalOp(
        name="decode",
        kind=OpKind.ATTENTION_DECODE,
        phase=Phase.DECODE,
        attrs={"batch": 1, "heads": 2, "seq_q": 1, "seq_kv": 16, "head_dim": 8},
        tensors=(
            TensorSpec(TensorRole.INPUT, (1, 2, 1, 8), DType.BF16, layout="q"),
            k,
            v,
            TensorSpec(TensorRole.OUTPUT, (1, 2, 1, 8), DType.BF16),
        ),
    )
    profile = RooflineModel().predict(op, hardware)
    assert profile.diagnostics["kv_cache_read_bytes"] == tensor_nbytes(k) + tensor_nbytes(v)
