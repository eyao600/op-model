from __future__ import annotations

import math

from opmodel.api import DType, EnergyBreakdown, EngineKind, MemoryAccess
from opmodel.hardware import HardwareSpec


def estimate_energy(
    *,
    flops: float,
    hbm_read_bytes: int | None = None,
    hbm_write_bytes: int | None = None,
    memory_access: MemoryAccess | None = None,
    engine: EngineKind,
    dtype: DType,
    hardware: HardwareSpec,
    latency_s: float = 0.0,
) -> EnergyBreakdown:
    compute_energy = flops * _energy_per_flop(engine, dtype, hardware)
    if memory_access is None:
        memory_access = MemoryAccess(
            hbm_read_bytes=0 if hbm_read_bytes is None else hbm_read_bytes,
            hbm_write_bytes=0 if hbm_write_bytes is None else hbm_write_bytes,
        )
    return EnergyBreakdown(
        compute_j=compute_energy,
        hbm_j=_level_energy(
            "hbm",
            memory_access.hbm_read_bytes,
            memory_access.hbm_write_bytes,
            hardware,
        ),
        l2_j=_level_energy(
            "l2", memory_access.l2_read_bytes, memory_access.l2_write_bytes, hardware
        ),
        sram_j=_level_energy(
            "sram", memory_access.sram_read_bytes, memory_access.sram_write_bytes, hardware
        ),
        register_j=_level_energy(
            "register",
            memory_access.register_read_bytes,
            memory_access.register_write_bytes,
            hardware,
        ),
        static_j=hardware.static_power_w * latency_s,
    )


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


def _level_energy(
    name: str, read_bytes: int | None, write_bytes: int | None, hardware: HardwareSpec
) -> float:
    level = hardware.memory_levels.get(name)
    if level is None:
        return 0.0
    byte_count = (read_bytes or 0) + (write_bytes or 0)
    if level.energy_j_per_sector is not None:
        sector_size = level.sector_size_bytes or 32
        return math.ceil(byte_count / sector_size) * level.energy_j_per_sector
    return byte_count * level.energy_j_per_byte
