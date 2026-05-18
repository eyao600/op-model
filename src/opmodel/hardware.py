from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from opmodel.api import DType


@dataclass(frozen=True)
class MemoryLevel:
    name: str
    size_bytes: int | None
    bandwidth_bytes_per_s: float
    energy_j_per_byte: float


@dataclass(frozen=True)
class ComputeUnit:
    clock_hz: float | None
    vector_flops_per_s: Mapping[DType, float]
    tensor_flops_per_s: Mapping[DType, float]
    vector_energy_j_per_flop: Mapping[DType, float]
    tensor_energy_j_per_flop: Mapping[DType, float]


@dataclass(frozen=True)
class Utilization:
    vector: float = 1.0
    tensor: float = 1.0
    memory: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    kind: str
    compute: ComputeUnit
    memory_levels: Mapping[str, MemoryLevel]
    utilization: Utilization = field(default_factory=Utilization)


def load_hardware(path: str | Path) -> HardwareSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Hardware config must be a mapping")
    hardware = _parse_hardware(data)
    _validate_hardware(hardware)
    return hardware


def _parse_hardware(data: Mapping[str, Any]) -> HardwareSpec:
    try:
        name = str(data["name"])
        kind = str(data["kind"])
        compute_data = _expect_mapping(data["compute"], "compute")
    except KeyError as exc:
        raise ValueError(f"Missing required hardware field: {exc.args[0]}") from exc

    compute = ComputeUnit(
        clock_hz=_optional_float(compute_data.get("clock_hz")),
        vector_flops_per_s=_dtype_float_map(
            compute_data.get("vector_flops_per_s", {}), "compute.vector_flops_per_s"
        ),
        tensor_flops_per_s=_dtype_float_map(
            compute_data.get("tensor_flops_per_s", {}), "compute.tensor_flops_per_s"
        ),
        vector_energy_j_per_flop=_dtype_float_map(
            compute_data.get("vector_energy_j_per_flop", {}),
            "compute.vector_energy_j_per_flop",
        ),
        tensor_energy_j_per_flop=_dtype_float_map(
            compute_data.get("tensor_energy_j_per_flop", {}),
            "compute.tensor_energy_j_per_flop",
        ),
    )

    memory_data = _expect_mapping(data.get("memory", {}), "memory")
    levels_data = memory_data.get("levels", [])
    if not isinstance(levels_data, list):
        raise ValueError("memory.levels must be a list")
    memory_levels: dict[str, MemoryLevel] = {}
    for index, item in enumerate(levels_data):
        level_data = _expect_mapping(item, f"memory.levels[{index}]")
        level = MemoryLevel(
            name=str(level_data.get("name", "")),
            size_bytes=_optional_int(level_data.get("size_bytes")),
            bandwidth_bytes_per_s=float(level_data.get("bandwidth_bytes_per_s", 0.0)),
            energy_j_per_byte=float(level_data.get("energy_j_per_byte", 0.0)),
        )
        if not level.name:
            raise ValueError(f"memory.levels[{index}].name is required")
        if level.name in memory_levels:
            raise ValueError(f"Duplicate memory level: {level.name}")
        memory_levels[level.name] = level

    utilization_data = _expect_mapping(data.get("utilization", {}), "utilization")
    utilization = Utilization(
        vector=float(utilization_data.get("vector", 1.0)),
        tensor=float(utilization_data.get("tensor", 1.0)),
        memory={
            str(name): float(value)
            for name, value in _expect_mapping(
                utilization_data.get("memory", {}), "utilization.memory"
            ).items()
        },
    )

    return HardwareSpec(
        name=name,
        kind=kind,
        compute=compute,
        memory_levels=memory_levels,
        utilization=utilization,
    )


def _validate_hardware(hardware: HardwareSpec) -> None:
    if "hbm" not in hardware.memory_levels:
        raise ValueError("Hardware config must include a memory level named 'hbm'")
    if not hardware.compute.vector_flops_per_s and not hardware.compute.tensor_flops_per_s:
        raise ValueError("Hardware config must define at least one compute engine")

    for engine_name, throughputs in (
        ("vector", hardware.compute.vector_flops_per_s),
        ("tensor", hardware.compute.tensor_flops_per_s),
    ):
        for dtype, value in throughputs.items():
            if value <= 0:
                raise ValueError(f"{engine_name} throughput for {dtype.value} must be positive")

    for engine_name, energy_map in (
        ("vector", hardware.compute.vector_energy_j_per_flop),
        ("tensor", hardware.compute.tensor_energy_j_per_flop),
    ):
        for dtype, value in energy_map.items():
            if value < 0:
                raise ValueError(f"{engine_name} energy for {dtype.value} must be non-negative")

    for level in hardware.memory_levels.values():
        if level.bandwidth_bytes_per_s <= 0:
            raise ValueError(f"Memory bandwidth for {level.name} must be positive")
        if level.energy_j_per_byte < 0:
            raise ValueError(f"Memory energy for {level.name} must be non-negative")

    _validate_utilization("vector", hardware.utilization.vector)
    _validate_utilization("tensor", hardware.utilization.tensor)
    for name, value in hardware.utilization.memory.items():
        if name not in hardware.memory_levels:
            raise ValueError(f"Utilization references unknown memory level: {name}")
        _validate_utilization(f"memory.{name}", value)


def _validate_utilization(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Utilization {name} must be in (0, 1]")


def _dtype_float_map(data: Any, field_name: str) -> dict[DType, float]:
    mapping = _expect_mapping(data, field_name)
    result: dict[DType, float] = {}
    for key, value in mapping.items():
        try:
            dtype = DType(str(key))
        except ValueError as exc:
            raise ValueError(f"Unknown dtype in {field_name}: {key}") from exc
        result[dtype] = float(value)
    return result


def _expect_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
