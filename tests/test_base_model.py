from dataclasses import replace
from pathlib import Path

import pytest

from opmodel import DType, EngineKind, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.hardware import load_hardware
from opmodel.models.base import BaseModel
from opmodel.models.roofline import RooflineModel
from opmodel.registry import create_model


ROOT = Path(__file__).resolve().parents[1]


def _gemm_op(m: int = 4, n: int = 16, k: int = 8) -> LocalOp:
    return LocalOp(
        name="gemm",
        kind=OpKind.GEMM,
        phase=Phase.TRAIN_FWD,
        tensors=(
            TensorSpec(TensorRole.INPUT, (m, k), DType.BF16),
            TensorSpec(TensorRole.WEIGHT, (k, n), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, (m, n), DType.BF16),
        ),
    )


def _softmax_op() -> LocalOp:
    return LocalOp(
        name="softmax",
        kind=OpKind.SOFTMAX,
        phase=Phase.INFERENCE,
        attrs={"row_size": 8},
        tensors=(
            TensorSpec(TensorRole.INPUT, (2, 8), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, (2, 8), DType.BF16),
        ),
    )


def test_base_is_registered_and_roofline_stays_roofline() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml")

    base_profile = create_model("base").predict(_gemm_op(), hardware)
    roofline_profile = create_model("roofline").predict(_gemm_op(), hardware)
    direct_roofline_profile = RooflineModel().predict(_gemm_op(), hardware)

    assert isinstance(create_model("base"), BaseModel)
    assert base_profile.implementation == "base.gemm"
    assert roofline_profile.implementation == "roofline.gemm"
    assert roofline_profile.latency_s == direct_roofline_profile.latency_s
    assert roofline_profile.memory_access.l2_read_bytes is None


def test_base_gemm_reports_selected_tile_and_multilevel_traffic() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/a10.yaml")

    profile = BaseModel().predict(_gemm_op(m=128, n=128, k=64), hardware)

    assert profile.engine == EngineKind.TENSOR
    assert profile.diagnostics["selected_tile_shape"]
    assert profile.diagnostics["threadblock_count"] > 0
    assert profile.diagnostics["wave_count"] > 0
    assert 0.0 < profile.diagnostics["throughput_derate"] <= 1.0
    assert profile.memory_access.hbm_read_bytes > 0
    assert profile.memory_access.hbm_write_bytes > 0
    assert profile.memory_access.l2_read_bytes is not None
    assert profile.memory_access.l2_write_bytes is not None
    assert profile.energy_breakdown.compute_j > 0.0
    assert profile.energy_breakdown.hbm_j > 0.0
    assert profile.energy_breakdown.l2_j > 0.0
    assert profile.energy_j == pytest.approx(profile.energy_breakdown.total_j)


def test_base_gemm_latency_uses_wave_derated_tensor_throughput() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/a10.yaml")
    high_bandwidth_memory = {
        name: replace(level, bandwidth_bytes_per_s=1.0e30)
        for name, level in hardware.memory_levels.items()
    }
    hardware = replace(hardware, memory_levels=high_bandwidth_memory)

    profile = BaseModel().predict(_gemm_op(), hardware)
    derate = profile.diagnostics["throughput_derate"]
    expected_flops_per_s = (
        hardware.compute.tensor_flops_per_s[DType.BF16] * hardware.utilization.tensor * derate
    )
    expected_compute_latency = profile.flops / expected_flops_per_s

    assert profile.diagnostics["effective_flops_per_s"] == pytest.approx(expected_flops_per_s)
    assert profile.diagnostics["compute_latency_s"] == pytest.approx(expected_compute_latency)
    assert profile.latency_s == pytest.approx(expected_compute_latency)


def test_base_softmax_uses_rapid_forward_constants() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/a10.yaml")
    profile = BaseModel().predict(_softmax_op(), hardware)

    elements = 2 * 8
    dtype_bytes = 2
    expected_memory_bytes = 4 * elements * dtype_bytes
    expected_flops = 7 * elements
    expected_compute_j = expected_flops * hardware.compute.vector_energy_j_per_flop[DType.BF16]
    expected_hbm_j = expected_memory_bytes * hardware.memory_levels["hbm"].energy_j_per_byte

    assert profile.implementation == "base.softmax"
    assert profile.engine == EngineKind.VECTOR
    assert profile.flops == expected_flops
    assert profile.memory_access.hbm_read_bytes + profile.memory_access.hbm_write_bytes == (
        expected_memory_bytes
    )
    assert profile.energy_breakdown.compute_j == pytest.approx(expected_compute_j)
    assert profile.energy_breakdown.hbm_j == pytest.approx(expected_hbm_j)


def test_roofline_softmax_constants_are_unchanged() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/a10.yaml")
    profile = RooflineModel().predict(_softmax_op(), hardware)

    elements = 2 * 8
    dtype_bytes = 2
    assert profile.implementation == "roofline.softmax"
    assert profile.flops == 8 * elements
    assert profile.memory_access.hbm_read_bytes + profile.memory_access.hbm_write_bytes == (
        2 * elements * dtype_bytes
    )
