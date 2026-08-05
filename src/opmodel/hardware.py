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
    energy_j_per_sector: float | None = None
    latency_s: float = 0.0
    latency_cycles: float | None = None
    line_size_bytes: int | None = None
    sector_size_bytes: int | None = None
    bank_count: int | None = None
    bank_width_bytes: int | None = None


@dataclass(frozen=True)
class ComputeUnit:
    clock_hz: float | None
    vector_flops_per_s: Mapping[DType, float]
    tensor_flops_per_s: Mapping[DType, float]
    vector_energy_j_per_flop: Mapping[DType, float]
    tensor_energy_j_per_flop: Mapping[DType, float]
    num_sms: int | None = None
    fma_dims: tuple[int, int, int] | None = None
    dataflow: str | None = None
    max_ctas_per_sm: int | None = None
    max_warps_per_sm: int | None = None
    registers_per_sm: int | None = None
    shared_memory_bytes_per_sm: int | None = None
    max_async_copy_groups: int | None = None
    tensor_latency_cycles: int | None = None
    device_fixed_overhead_cycles: int | None = None


@dataclass(frozen=True)
class Utilization:
    vector: float = 1.0
    tensor: float = 1.0
    memory: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EnergyModelPowerCoefficients:
    base_power_w: float = 0.0
    sm_resident_power_w: float = 0.0
    tc_active_power_w: float = 0.0
    dram_active_power_w: float = 0.0
    dram_exposed_power_w: float = 0.0
    l2_active_power_w: float = 0.0
    smem_active_power_w: float = 0.0


@dataclass(frozen=True)
class EnergyModelSpec:
    model_level: str = "E3"
    power_coefficients: EnergyModelPowerCoefficients = field(
        default_factory=EnergyModelPowerCoefficients
    )
    feature_order: tuple[str, ...] = (
        "time_kernel_s",
        "time_sm_resident_s",
        "time_tc_active_s",
        "time_dram_active_s",
        "time_l2_active_s",
        "time_smem_active_s",
    )
    calibration: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    kind: str
    compute: ComputeUnit
    memory_levels: Mapping[str, MemoryLevel]
    utilization: Utilization = field(default_factory=Utilization)
    static_power_w: float = 0.0
    energy_model: EnergyModelSpec | None = None


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
        num_sms=_optional_int(compute_data.get("num_sms")),
        fma_dims=_optional_fma_dims(compute_data.get("fma_dims")),
        dataflow=_optional_str(compute_data.get("dataflow")),
        max_ctas_per_sm=_optional_int(compute_data.get("max_ctas_per_sm")),
        max_warps_per_sm=_optional_int(compute_data.get("max_warps_per_sm")),
        registers_per_sm=_optional_int(compute_data.get("registers_per_sm")),
        shared_memory_bytes_per_sm=_optional_int(
            compute_data.get("shared_memory_bytes_per_sm")
        ),
        max_async_copy_groups=_optional_int(compute_data.get("max_async_copy_groups")),
        tensor_latency_cycles=_optional_int(compute_data.get("tensor_latency_cycles")),
        device_fixed_overhead_cycles=_optional_int(
            compute_data.get("device_fixed_overhead_cycles")
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
            energy_j_per_sector=_optional_float(level_data.get("energy_j_per_sector")),
            latency_s=float(level_data.get("latency_s", 0.0)),
            latency_cycles=_optional_float(level_data.get("latency_cycles")),
            line_size_bytes=_optional_int(level_data.get("line_size_bytes")),
            sector_size_bytes=_optional_int(level_data.get("sector_size_bytes")),
            bank_count=_optional_int(level_data.get("bank_count")),
            bank_width_bytes=_optional_int(level_data.get("bank_width_bytes")),
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
        static_power_w=float(data.get("static_power_w", 0.0)),
        energy_model=_parse_energy_model(data.get("energy_model")),
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

    if hardware.compute.num_sms is not None and hardware.compute.num_sms <= 0:
        raise ValueError("compute.num_sms must be positive")
    if hardware.compute.fma_dims is not None:
        for dim in hardware.compute.fma_dims:
            if dim <= 0:
                raise ValueError("compute.fma_dims values must be positive")
    for field_name in (
        "max_ctas_per_sm",
        "max_warps_per_sm",
        "registers_per_sm",
        "shared_memory_bytes_per_sm",
        "max_async_copy_groups",
    ):
        value = getattr(hardware.compute, field_name)
        if value is not None and value <= 0:
            raise ValueError(f"compute.{field_name} must be positive")
    for field_name in ("tensor_latency_cycles", "device_fixed_overhead_cycles"):
        value = getattr(hardware.compute, field_name)
        if value is not None and value < 0:
            raise ValueError(f"compute.{field_name} must be non-negative")
    if hardware.static_power_w < 0:
        raise ValueError("static_power_w must be non-negative")
    if hardware.energy_model is not None:
        _validate_energy_model(hardware.energy_model)

    for level in hardware.memory_levels.values():
        if level.size_bytes is not None and level.size_bytes <= 0:
            raise ValueError(f"Memory size_bytes for {level.name} must be positive")
        if level.bandwidth_bytes_per_s <= 0:
            raise ValueError(f"Memory bandwidth for {level.name} must be positive")
        if level.energy_j_per_byte < 0:
            raise ValueError(f"Memory energy for {level.name} must be non-negative")
        if level.energy_j_per_sector is not None and level.energy_j_per_sector < 0:
            raise ValueError(
                f"Memory sector energy for {level.name} must be non-negative"
            )
        if level.latency_s < 0:
            raise ValueError(f"Memory latency for {level.name} must be non-negative")
        if level.latency_cycles is not None and level.latency_cycles < 0:
            raise ValueError(
                f"Memory latency cycles for {level.name} must be non-negative"
            )
        for field_name in (
            "line_size_bytes",
            "sector_size_bytes",
            "bank_count",
            "bank_width_bytes",
        ):
            value = getattr(level, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"Memory {field_name} for {level.name} must be positive")

    _validate_utilization("vector", hardware.utilization.vector)
    _validate_utilization("tensor", hardware.utilization.tensor)
    for name, value in hardware.utilization.memory.items():
        if name not in hardware.memory_levels:
            raise ValueError(f"Utilization references unknown memory level: {name}")
        _validate_utilization(f"memory.{name}", value)


def _validate_utilization(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Utilization {name} must be in (0, 1]")


def _parse_energy_model(data: Any) -> EnergyModelSpec | None:
    if data is None:
        return None
    mapping = _expect_mapping(data, "energy_model")
    coefficients_data = _expect_mapping(
        mapping.get("power_coefficients", {}),
        "energy_model.power_coefficients",
    )
    feature_order_data = mapping.get("feature_order", EnergyModelSpec.feature_order)
    if not isinstance(feature_order_data, (list, tuple)):
        raise ValueError("energy_model.feature_order must be a list")
    return EnergyModelSpec(
        model_level=str(mapping.get("model_level", "E3")),
        power_coefficients=EnergyModelPowerCoefficients(
            base_power_w=float(coefficients_data.get("base_power_w", 0.0)),
            sm_resident_power_w=float(
                coefficients_data.get("sm_resident_power_w", 0.0)
            ),
            tc_active_power_w=float(coefficients_data.get("tc_active_power_w", 0.0)),
            dram_active_power_w=float(
                coefficients_data.get("dram_active_power_w", 0.0)
            ),
            dram_exposed_power_w=float(
                coefficients_data.get("dram_exposed_power_w", 0.0)
            ),
            l2_active_power_w=float(coefficients_data.get("l2_active_power_w", 0.0)),
            smem_active_power_w=float(
                coefficients_data.get("smem_active_power_w", 0.0)
            ),
        ),
        feature_order=tuple(str(item) for item in feature_order_data),
        calibration=dict(_expect_mapping(mapping.get("calibration", {}), "energy_model.calibration")),
    )


def _validate_energy_model(energy_model: EnergyModelSpec) -> None:
    if energy_model.model_level not in {"E0", "E1", "E2", "E3"}:
        raise ValueError("energy_model.model_level must be one of E0, E1, E2, E3")
    if not energy_model.feature_order:
        raise ValueError("energy_model.feature_order must not be empty")
    coefficients = energy_model.power_coefficients
    for field_name in (
        "base_power_w",
        "sm_resident_power_w",
        "tc_active_power_w",
        "dram_active_power_w",
        "dram_exposed_power_w",
        "l2_active_power_w",
        "smem_active_power_w",
    ):
        if getattr(coefficients, field_name) < 0.0:
            raise ValueError(f"energy_model.power_coefficients.{field_name} must be non-negative")


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_fma_dims(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            return (int(value["m"]), int(value["n"]), int(value["k"]))
        except KeyError as exc:
            raise ValueError(f"compute.fma_dims missing {exc.args[0]}") from exc
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError("compute.fma_dims must be a three-item sequence or m/n/k mapping")
