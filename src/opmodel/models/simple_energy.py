from __future__ import annotations

from opmodel.api import DType, EnergyBreakdown, EngineKind
from opmodel.hardware import HardwareSpec


def estimate_energy(
    *,
    flops: float,
    hbm_read_bytes: int,
    hbm_write_bytes: int,
    engine: EngineKind,
    dtype: DType,
    hardware: HardwareSpec,
) -> EnergyBreakdown:
    compute_energy = flops * _energy_per_flop(engine, dtype, hardware)
    hbm = hardware.memory_levels["hbm"]
    hbm_energy = (hbm_read_bytes + hbm_write_bytes) * hbm.energy_j_per_byte
    return EnergyBreakdown(compute_j=compute_energy, hbm_j=hbm_energy)


def _energy_per_flop(engine: EngineKind, dtype: DType, hardware: HardwareSpec) -> float:
    if engine == EngineKind.TENSOR:
        return hardware.compute.tensor_energy_j_per_flop.get(dtype, 0.0)
    if engine == EngineKind.VECTOR:
        return hardware.compute.vector_energy_j_per_flop.get(dtype, 0.0)
    if engine == EngineKind.MIXED:
        return hardware.compute.tensor_energy_j_per_flop.get(
            dtype, hardware.compute.vector_energy_j_per_flop.get(dtype, 0.0)
        )
    return 0.0
