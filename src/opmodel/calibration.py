from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from opmodel.api import OpKind
from opmodel.energy import (
    E3_POWER_FEATURE_ORDER,
    extract_gemm_e3_features,
    fixed_event_energy_j,
)
from opmodel.hardware import (
    EnergyModelPowerCoefficients,
    EnergyModelSpec,
    HardwareSpec,
    load_hardware,
)
from opmodel.registry import create_model
from opmodel.validation.artifact_accuracy import (
    DEFAULT_ARTIFACT_DATA_DIR,
    DEFAULT_HARDWARE_DIR,
    ArtifactSample,
    _hardware_info_for_file,
    _is_bf16_row,
    _matches_config_frequency,
    _sample_from_row,
    _supported_op_kind,
)


_POWER_COEFFICIENT_FIELDS = (
    "base_power_w",
    "sm_resident_power_w",
    "tc_active_power_w",
    "dram_active_power_w",
    "l2_active_power_w",
    "smem_active_power_w",
)


@dataclass(frozen=True)
class CalibrationMetrics:
    count: int
    mae_j: float
    mape_pct: float
    median_ape_pct: float
    p90_ape_pct: float


@dataclass(frozen=True)
class EnergyCalibrationResult:
    hardware_name: str
    input_hardware_path: Path
    energy_model: EnergyModelSpec
    fit_metrics_by_level: Mapping[str, CalibrationMetrics]
    validation_metrics_by_level: Mapping[str, CalibrationMetrics]
    fit_rows: int
    validation_rows: int
    clipped_coefficients: tuple[str, ...]
    skip_counts: Mapping[str, int]


def calibrate_energy_from_artifact_database(
    *,
    hardware_name: str,
    data_dir: str | Path = DEFAULT_ARTIFACT_DATA_DIR,
    hardware_dir: str | Path = DEFAULT_HARDWARE_DIR,
    fit_fraction: float = 0.7,
    limit: int | None = None,
) -> EnergyCalibrationResult:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in (0, 1)")

    hardware_path = _resolve_hardware_path(hardware_name, hardware_dir)
    hardware = load_hardware(hardware_path)
    samples, skip_counts = _artifact_gemm_samples(
        data_dir=data_dir,
        hardware_name=hardware.name,
        limit=limit,
    )
    if len(samples) < 2:
        raise ValueError("At least two GEMM samples are required for fit and validation")

    split_index = int(math.floor(len(samples) * fit_fraction))
    split_index = min(max(split_index, 1), len(samples) - 1)
    fit_samples = samples[:split_index]
    validation_samples = samples[split_index:]

    feature_rows, targets = _fit_matrix(fit_samples, hardware)
    theta, *_ = np.linalg.lstsq(feature_rows, targets, rcond=None)
    clipped_coefficients = tuple(
        field
        for field, value in zip(_POWER_COEFFICIENT_FIELDS, theta)
        if float(value) < 0.0
    )
    theta = np.maximum(theta, 0.0)
    coefficients = EnergyModelPowerCoefficients(
        base_power_w=float(theta[0]),
        sm_resident_power_w=float(theta[1]),
        tc_active_power_w=float(theta[2]),
        dram_active_power_w=float(theta[3]),
        l2_active_power_w=float(theta[4]),
        smem_active_power_w=float(theta[5]),
    )

    energy_model = EnergyModelSpec(
        model_level="E3",
        power_coefficients=coefficients,
        feature_order=E3_POWER_FEATURE_ORDER,
        calibration={
            "source": "artifact_validation_database",
            "calibration_date": date.today().isoformat(),
            "fit_fraction": fit_fraction,
            "fit_rows": len(fit_samples),
            "validation_rows": len(validation_samples),
            "clipped_coefficients": clipped_coefficients,
            "notes": (
                "FLOP and byte event coefficients are fixed from hardware config; "
                "least-squares fit calibrates only residual power attribution terms."
            ),
        },
    )
    calibrated_hardware = replace(hardware, energy_model=energy_model)
    fit_metrics = _metrics_by_level(fit_samples, calibrated_hardware)
    validation_metrics = _metrics_by_level(validation_samples, calibrated_hardware)

    calibration_metadata = dict(energy_model.calibration)
    calibration_metadata["fit_mape_pct"] = fit_metrics["E3"].mape_pct
    calibration_metadata["validation_mape_pct"] = validation_metrics["E3"].mape_pct
    energy_model = replace(energy_model, calibration=calibration_metadata)
    return EnergyCalibrationResult(
        hardware_name=hardware.name,
        input_hardware_path=hardware_path,
        energy_model=energy_model,
        fit_metrics_by_level=fit_metrics,
        validation_metrics_by_level=validation_metrics,
        fit_rows=len(fit_samples),
        validation_rows=len(validation_samples),
        clipped_coefficients=clipped_coefficients,
        skip_counts=skip_counts,
    )


def write_calibrated_hardware_config(
    *,
    input_hardware_path: str | Path,
    output_hardware_path: str | Path,
    energy_model: EnergyModelSpec,
) -> None:
    input_path = Path(input_hardware_path)
    output_path = Path(output_hardware_path)
    with input_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Hardware config must be a mapping")
    output_data = dict(data)
    output_data["energy_model"] = energy_model_to_config(energy_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output_data, handle, sort_keys=False)


def energy_model_to_config(energy_model: EnergyModelSpec) -> dict[str, Any]:
    coefficients = energy_model.power_coefficients
    return {
        "model_level": energy_model.model_level,
        "feature_order": list(energy_model.feature_order),
        "power_coefficients": {
            "base_power_w": coefficients.base_power_w,
            "sm_resident_power_w": coefficients.sm_resident_power_w,
            "tc_active_power_w": coefficients.tc_active_power_w,
            "dram_active_power_w": coefficients.dram_active_power_w,
            "l2_active_power_w": coefficients.l2_active_power_w,
            "smem_active_power_w": coefficients.smem_active_power_w,
        },
        "calibration": _yaml_plain(dict(energy_model.calibration)),
    }


def format_calibration_report(result: EnergyCalibrationResult) -> str:
    lines = [
        f"Calibrated hardware: {result.hardware_name}",
        f"Fit rows: {result.fit_rows}",
        f"Validation rows: {result.validation_rows}",
        "Power coefficients:",
    ]
    coefficients = result.energy_model.power_coefficients
    for field in _POWER_COEFFICIENT_FIELDS:
        lines.append(f"  {field}: {getattr(coefficients, field):.8g} W")
    if result.clipped_coefficients:
        lines.append(
            "Clipped negative coefficients: "
            + ", ".join(result.clipped_coefficients)
        )
    lines.append("Fit metrics:")
    lines.extend(_format_metrics(result.fit_metrics_by_level))
    lines.append("Validation metrics:")
    lines.extend(_format_metrics(result.validation_metrics_by_level))
    if result.skip_counts:
        lines.append("Skipped:")
        for reason, count in sorted(result.skip_counts.items()):
            lines.append(f"  {reason}: {count}")
    return "\n".join(lines)


def _artifact_gemm_samples(
    *,
    data_dir: str | Path,
    hardware_name: str,
    limit: int | None,
) -> tuple[tuple[ArtifactSample, ...], dict[str, int]]:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Artifact data directory does not exist: {data_path}")

    samples: list[ArtifactSample] = []
    skip_counts: dict[str, int] = {}
    for csv_path in sorted(data_path.glob("*.csv")):
        hardware_info = _hardware_info_for_file(csv_path.name)
        if hardware_info is None:
            _increment(skip_counts, "unsupported_hardware_file")
            continue
        if not _matches_config_frequency(csv_path.name):
            _increment(skip_counts, "unsupported_frequency_file")
            continue
        file_hardware_name, _hardware_filename = hardware_info
        if file_hardware_name != hardware_name:
            continue
        op_kind = _supported_op_kind(csv_path.name)
        if op_kind != OpKind.BATCHED_GEMM.value:
            continue

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=1):
                if limit is not None and len(samples) >= limit:
                    return tuple(samples), skip_counts
                if not _is_bf16_row(row):
                    _increment(skip_counts, "non_bf16_row")
                    continue
                try:
                    samples.append(
                        _sample_from_row(
                            csv_path,
                            row_index,
                            file_hardware_name,
                            op_kind,
                            row,
                        )
                    )
                except (KeyError, ValueError) as exc:
                    reason = str(exc).splitlines()[0]
                    _increment(skip_counts, f"invalid_row:{reason}")
    return tuple(samples), skip_counts


def _fit_matrix(
    samples: tuple[ArtifactSample, ...],
    hardware: HardwareSpec,
) -> tuple[np.ndarray, np.ndarray]:
    model = create_model("extended_roofline")
    uncalibrated_hardware = replace(hardware, energy_model=None)
    feature_rows: list[tuple[float, ...]] = []
    targets: list[float] = []
    for sample in samples:
        profile = model.predict(sample.op, uncalibrated_hardware)
        features = extract_gemm_e3_features(profile)
        feature_rows.append(features.power_vector())
        targets.append(sample.measured_energy_j - fixed_event_energy_j(profile))
    return np.asarray(feature_rows, dtype=float), np.asarray(targets, dtype=float)


def _metrics_by_level(
    samples: tuple[ArtifactSample, ...],
    hardware: HardwareSpec,
) -> dict[str, CalibrationMetrics]:
    model = create_model("extended_roofline")
    by_level: dict[str, list[tuple[float, float]]] = {
        "E0": [],
        "E1": [],
        "E2": [],
        "E3": [],
    }
    for sample in samples:
        profile = model.predict(sample.op, hardware)
        energy_model = profile.diagnostics.get("energy_model", {})
        energy_by_level = (
            energy_model.get("energy_by_level_j", {})
            if isinstance(energy_model, Mapping)
            else {}
        )
        for level in by_level:
            predicted = float(energy_by_level.get(level, profile.energy_j))
            by_level[level].append((predicted, sample.measured_energy_j))
    return {level: _metrics(rows) for level, rows in by_level.items()}


def _metrics(rows: list[tuple[float, float]]) -> CalibrationMetrics:
    if not rows:
        return CalibrationMetrics(
            count=0,
            mae_j=float("nan"),
            mape_pct=float("nan"),
            median_ape_pct=float("nan"),
            p90_ape_pct=float("nan"),
        )
    absolute_errors = [abs(predicted - measured) for predicted, measured in rows]
    pct_errors = [
        abs(predicted - measured) / measured * 100.0
        for predicted, measured in rows
        if measured > 0.0
    ]
    pct_errors_sorted = sorted(pct_errors)
    return CalibrationMetrics(
        count=len(rows),
        mae_j=float(sum(absolute_errors) / len(absolute_errors)),
        mape_pct=float(sum(pct_errors) / len(pct_errors)) if pct_errors else float("nan"),
        median_ape_pct=_percentile(pct_errors_sorted, 0.5),
        p90_ape_pct=_percentile(pct_errors_sorted, 0.9),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    index = quantile * (len(values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return values[lower]
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _format_metrics(metrics_by_level: Mapping[str, CalibrationMetrics]) -> list[str]:
    lines: list[str] = []
    for level in ("E0", "E1", "E2", "E3"):
        metrics = metrics_by_level[level]
        lines.append(
            f"  {level}: n={metrics.count}, MAE={metrics.mae_j:.6g} J, "
            f"MAPE={metrics.mape_pct:.3g}%, "
            f"median={metrics.median_ape_pct:.3g}%, p90={metrics.p90_ape_pct:.3g}%"
        )
    return lines


def _resolve_hardware_path(hardware_name: str, hardware_dir: str | Path) -> Path:
    hardware_path = Path(hardware_name)
    if hardware_path.exists():
        return hardware_path
    candidate = Path(hardware_dir) / f"{hardware_name}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Hardware config does not exist: {candidate}")


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _yaml_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _yaml_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_yaml_plain(item) for item in value]
    return value
