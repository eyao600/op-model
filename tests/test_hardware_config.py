from pathlib import Path

import pytest

from opmodel import DType
from opmodel.hardware import load_hardware


ROOT = Path(__file__).resolve().parents[1]


def test_generic_gpu_loads() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml")
    assert hardware.name == "generic_gpu"
    assert "hbm" in hardware.memory_levels


def test_generic_tpu_loads() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/tpu_generic.yaml")
    assert hardware.kind == "tpu"
    assert "hbm" in hardware.memory_levels


def test_a100_40gb_pcie_loads() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/a100_40gb_pcie.yaml")
    assert hardware.name == "a100_40gb_pcie"
    assert hardware.compute.clock_hz == 9.0e8
    assert hardware.compute.num_sms == 108
    assert hardware.compute.fma_dims == (16, 8, 16)
    assert hardware.compute.dataflow == "output_stationary"
    assert hardware.memory_levels["hbm"].size_bytes == 40 * 1024**3
    assert hardware.compute.tensor_flops_per_s[DType.BF16] > 1.9e14
    assert hardware.compute.vector_flops_per_s[DType.BF16] > 1.2e13
    assert hardware.compute.tensor_energy_j_per_flop[DType.BF16] > 0.0


def test_a10_loads() -> None:
    hardware = load_hardware(ROOT / "src/opmodel/configs/hardware/a10.yaml")
    assert hardware.name == "a10"
    assert hardware.compute.clock_hz == 9.0e8
    assert hardware.compute.num_sms == 72
    assert hardware.memory_levels["hbm"].size_bytes == 24 * 1024**3
    assert hardware.compute.tensor_flops_per_s[DType.BF16] > 6.0e13
    assert hardware.compute.vector_flops_per_s[DType.BF16] > 1.6e13
    assert hardware.memory_levels["hbm"].energy_j_per_byte > 0.0


def test_missing_hbm_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
name: bad
kind: gpu
compute:
  vector_flops_per_s:
    bf16: 1.0
memory:
  levels:
    - name: sram
      bandwidth_bytes_per_s: 1.0
      energy_j_per_byte: 0.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hbm"):
        load_hardware(path)


def test_invalid_utilization_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
name: bad
kind: gpu
compute:
  vector_flops_per_s:
    bf16: 1.0
memory:
  levels:
    - name: hbm
      bandwidth_bytes_per_s: 1.0
      energy_j_per_byte: 0.0
utilization:
  vector: 0.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Utilization vector"):
        load_hardware(path)


def test_nonpositive_memory_size_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad_size.yaml"
    path.write_text(
        """
name: bad_size
kind: gpu
compute:
  vector_flops_per_s:
    bf16: 1.0
memory:
  levels:
    - name: hbm
      size_bytes: 0
      bandwidth_bytes_per_s: 1.0
      energy_j_per_byte: 0.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="size_bytes"):
        load_hardware(path)


def test_optional_base_hardware_fields_load(tmp_path: Path) -> None:
    path = tmp_path / "base_fields.yaml"
    path.write_text(
        """
name: base_fields
kind: gpu
kernel_launch_overhead_s: 0.000001
compute:
  clock_hz: 1.0e9
  num_sms: 4
  fma_dims:
    m: 16
    n: 8
    k: 16
  dataflow: output_stationary
  vector_flops_per_s:
    bf16: 1.0
memory:
  levels:
    - name: hbm
      bandwidth_bytes_per_s: 1.0
      energy_j_per_byte: 0.0
      latency_s: 0.0000001
""",
        encoding="utf-8",
    )
    hardware = load_hardware(path)
    assert hardware.kernel_launch_overhead_s == pytest.approx(0.000001)
    assert hardware.compute.num_sms == 4
    assert hardware.compute.fma_dims == (16, 8, 16)
    assert hardware.compute.dataflow == "output_stationary"
    assert hardware.memory_levels["hbm"].latency_s == pytest.approx(0.0000001)
