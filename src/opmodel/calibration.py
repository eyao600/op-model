from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from opmodel.api import OpKind
from opmodel.energy import (
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
from opmodel.validation.gemm_latency import (
    DEFAULT_TRAINING_PER_CLASS as GEMM_OVERHEAD_TRAINING_PER_CLASS,
    _estimate_fixed_overhead_cycles,
    _select_training_samples as _select_latency_overhead_training_samples,
    _with_fixed_overhead,
    classify_gemm_size,
)


_POWER_COEFFICIENT_FIELDS = (
    "base_power_w",
    "sm_resident_power_w",
    "tc_active_power_w",
    "dram_active_power_w",
    "l2_active_power_w",
    "smem_active_power_w",
)
DEFAULT_ENERGY_TRAINING_PER_CLASS = 6
ENERGY_CALIBRATION_LATENCY_APE_LIMIT_PCT = 20.0
ENERGY_CALIBRATION_QUANTILE_OFFSET = 0.1
_ENERGY_CALIBRATION_CLASSES = (
    "large",
    "regular",
    "skinny",
    "small",
    "small_k",
    "vector_like",
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
    hardware_path = _resolve_hardware_path(hardware_name, hardware_dir)
    hardware = load_hardware(hardware_path)
    samples, skip_counts = _artifact_gemm_samples(
        data_dir=data_dir,
        hardware_name=hardware.name,
        limit=limit,
    )
    if len(samples) < 2:
        raise ValueError("At least two GEMM samples are required for fit and validation")

    fixed_overhead_training_samples = _select_latency_overhead_training_samples(
        samples,
        training_per_class=GEMM_OVERHEAD_TRAINING_PER_CLASS,
    )
    energy_model_name = "effective_roofline"
    fixed_overhead_cycles = _estimate_fixed_overhead_cycles(
        samples=fixed_overhead_training_samples,
        hardware=hardware,
        model=create_model(energy_model_name),
    )
    profiled_hardware = _with_fixed_overhead(hardware, fixed_overhead_cycles)
    fixed_overhead_training_keys = {
        _sample_key(sample) for sample in fixed_overhead_training_samples
    }
    energy_samples = tuple(
        sample
        for sample in samples
        if _sample_key(sample) not in fixed_overhead_training_keys
    )

    fit_samples = _select_static_power_calibration_samples(
        energy_samples,
        hardware=profiled_hardware,
        training_per_class=DEFAULT_ENERGY_TRAINING_PER_CLASS,
        latency_ape_limit_pct=ENERGY_CALIBRATION_LATENCY_APE_LIMIT_PCT,
        quantile_offset=ENERGY_CALIBRATION_QUANTILE_OFFSET,
        stratification="measured_energy",
        model_name=energy_model_name,
    )
    fit_keys = {_sample_key(sample) for sample in fit_samples}
    validation_samples = tuple(
        sample for sample in energy_samples if _sample_key(sample) not in fit_keys
    )
    if not fit_samples or not validation_samples:
        raise ValueError("Calibration requires both fit and validation rows")

    coefficients = _fit_static_dram_power_coefficients(
        fit_samples,
        profiled_hardware,
        model_name=energy_model_name,
    )

    energy_model = EnergyModelSpec(
        model_level="E3",
        power_coefficients=coefficients,
        feature_order=("time_kernel_s", "time_dram_active_s"),
        calibration={
            "source": "artifact_validation_database",
            "calibration_date": date.today().isoformat(),
            "policy": "normalized_static_plus_dram_active_power_nnls",
            "profile_model": energy_model_name,
            "latency_ape_limit_pct": ENERGY_CALIBRATION_LATENCY_APE_LIMIT_PCT,
            "training_per_class": DEFAULT_ENERGY_TRAINING_PER_CLASS,
            "stratification": "measured_energy",
            "quantile_offset": ENERGY_CALIBRATION_QUANTILE_OFFSET,
            "regression": "nonnegative_least_squares",
            "weighting": "measured_energy_inverse",
            "feature_scaling": "weighted_l2_unit_norm",
            "base_fixed_overhead_cycles": hardware.compute.device_fixed_overhead_cycles,
            "calibrated_fixed_overhead_cycles": fixed_overhead_cycles,
            "fit_rows": len(fit_samples),
            "validation_rows": len(validation_samples),
            "calibration_rows": [
                {"source_file": sample.source_file, "row_index": sample.row_index}
                for sample in fit_samples
            ],
            "clipped_coefficients": (),
            "notes": (
                "FLOP and byte event coefficients are fixed from hardware config; "
                "normalized nonnegative regression calibrates only residual base "
                "and DRAM-active power terms. Other E3 durations are excluded from "
                "this minimal model because the calibration feature audit found "
                "strong collinearity with kernel time."
            ),
        },
    )
    calibrated_hardware = replace(profiled_hardware, energy_model=energy_model)
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
        clipped_coefficients=(),
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


def _select_static_power_calibration_samples(
    samples: tuple[ArtifactSample, ...],
    *,
    hardware: HardwareSpec,
    training_per_class: int,
    latency_ape_limit_pct: float,
    quantile_offset: float,
    stratification: str,
    model_name: str = "effective_roofline",
) -> tuple[ArtifactSample, ...]:
    model = create_model(model_name)
    uncalibrated_hardware = replace(hardware, energy_model=None)
    rows: list[tuple[ArtifactSample, float, str]] = []
    for sample in samples:
        if sample.measured_energy_j <= 0.0 or sample.measured_latency_ms <= 0.0:
            continue
        profile = model.predict(sample.op, uncalibrated_hardware)
        latency_ape_pct = abs(
            profile.latency_s * 1000.0 / sample.measured_latency_ms - 1.0
        ) * 100.0
        if latency_ape_pct >= latency_ape_limit_pct:
            continue
        scale = (
            sample.measured_energy_j
            if stratification == "measured_energy"
            else max(1.0, float(profile.flops))
        )
        rows.append((sample, scale, _sample_class(sample)))

    selected: list[ArtifactSample] = []
    for kernel_class in _ENERGY_CALIBRATION_CLASSES:
        class_rows = sorted(
            (row for row in rows if row[2] == kernel_class),
            key=lambda row: (row[1], _sample_key(row[0])),
        )
        selected.extend(
            sample
            for sample, _scale in _stratified_by_scale(
                [(sample, scale) for sample, scale, _ in class_rows],
                count=min(training_per_class, len(class_rows)),
                quantile_offset=quantile_offset,
            )
        )
    return tuple(sorted(selected, key=_sample_key))


def _stratified_by_scale(
    rows: list[tuple[ArtifactSample, float]],
    *,
    count: int,
    quantile_offset: float,
) -> tuple[tuple[ArtifactSample, float], ...]:
    if count <= 0 or not rows:
        return ()
    if count >= len(rows):
        return tuple(rows)
    selected: list[tuple[ArtifactSample, float]] = []
    used: set[tuple[str, int]] = set()
    for index in range(count):
        quantile = (
            (index + quantile_offset)
            / max(1.0, count - 1 + 2 * quantile_offset)
            if count > 1
            else 0.5
        )
        candidate_index = round(quantile * (len(rows) - 1))
        candidate = rows[candidate_index]
        if _sample_key(candidate[0]) in used:
            candidate = next(row for row in rows if _sample_key(row[0]) not in used)
        selected.append(candidate)
        used.add(_sample_key(candidate[0]))
    return tuple(selected)


def _fit_static_dram_power_coefficients(
    samples: tuple[ArtifactSample, ...],
    hardware: HardwareSpec,
    *,
    model_name: str = "effective_roofline",
) -> EnergyModelPowerCoefficients:
    rows = _power_fit_rows(samples, hardware, model_name=model_name)
    base_power_w, dram_active_power_w = _fit_nonnegative_two_feature_power(rows)
    return EnergyModelPowerCoefficients(
        base_power_w=base_power_w,
        dram_active_power_w=dram_active_power_w,
    )


def _power_fit_rows(
    samples: tuple[ArtifactSample, ...],
    hardware: HardwareSpec,
    *,
    model_name: str = "effective_roofline",
) -> list[tuple[float, float, float, float]]:
    model = create_model(model_name)
    uncalibrated_hardware = replace(hardware, energy_model=None)
    rows: list[tuple[float, float, float, float]] = []
    for sample in samples:
        profile = model.predict(sample.op, uncalibrated_hardware)
        features = extract_gemm_e3_features(profile)
        if features.time_kernel_s <= 0.0 or sample.measured_energy_j <= 0.0:
            continue
        residual_j = sample.measured_energy_j - fixed_event_energy_j(profile)
        rows.append(
            (
                features.time_kernel_s,
                features.time_dram_active_s,
                residual_j,
                sample.measured_energy_j,
            )
        )
    return rows


def _fit_nonnegative_two_feature_power(
    rows: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    scales = _weighted_feature_scales(rows)
    scaled_rows = [
        (
            time_kernel_s / scales[0],
            time_dram_active_s / scales[1],
            residual_j,
            measured_energy_j,
        )
        for time_kernel_s, time_dram_active_s, residual_j, measured_energy_j in rows
    ]
    best: tuple[float, tuple[float, float]] | None = None
    active_sets = ((0, 1), (0,), (1,), ())
    for active in active_sets:
        scaled_coefficients = [0.0, 0.0]
        if active:
            matrix, rhs = _normal_equations_for_active_features(
                scaled_rows,
                active=active,
            )
            solution = _solve_2x2_or_1x1(matrix, rhs)
            if any(value < -1e-9 for value in solution):
                continue
            for index, value in zip(active, solution):
                scaled_coefficients[index] = max(0.0, float(value))
        coefficients = (
            scaled_coefficients[0] / scales[0],
            scaled_coefficients[1] / scales[1],
        )
        objective = _weighted_relative_sse(rows, coefficients)
        if best is None or objective < best[0]:
            best = (objective, coefficients)
    return best[1] if best is not None else (0.0, 0.0)


def _weighted_feature_scales(
    rows: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    sums = [0.0, 0.0]
    for time_kernel_s, time_dram_active_s, _residual_j, measured_energy_j in rows:
        if measured_energy_j <= 0.0:
            continue
        features = (time_kernel_s, time_dram_active_s)
        for index, feature in enumerate(features):
            sums[index] += (feature / measured_energy_j) ** 2
    return (
        max(math.sqrt(sums[0]), 1.0e-30),
        max(math.sqrt(sums[1]), 1.0e-30),
    )


def _normal_equations_for_active_features(
    rows: list[tuple[float, float, float, float]],
    *,
    active: tuple[int, ...],
) -> tuple[list[list[float]], list[float]]:
    matrix = [[0.0 for _ in active] for _ in active]
    rhs = [0.0 for _ in active]
    for time_kernel_s, time_dram_active_s, residual_j, measured_energy_j in rows:
        weight = 1.0 / measured_energy_j
        features = (time_kernel_s, time_dram_active_s)
        for row_index, feature_index in enumerate(active):
            weighted_feature = features[feature_index] * weight
            rhs[row_index] += weighted_feature * residual_j * weight
            for col_index, other_feature_index in enumerate(active):
                matrix[row_index][col_index] += (
                    weighted_feature * features[other_feature_index] * weight
                )
    return matrix, rhs


def _solve_2x2_or_1x1(matrix: list[list[float]], rhs: list[float]) -> tuple[float, ...]:
    if len(rhs) == 0:
        return ()
    if len(rhs) == 1:
        denominator = matrix[0][0]
        return (rhs[0] / denominator if denominator else 0.0,)
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if abs(determinant) <= 1e-30:
        return (0.0, 0.0)
    return (
        (rhs[0] * d - b * rhs[1]) / determinant,
        (a * rhs[1] - rhs[0] * c) / determinant,
    )


def _weighted_relative_sse(
    rows: list[tuple[float, float, float, float]],
    coefficients: tuple[float, float],
) -> float:
    total = 0.0
    for time_kernel_s, time_dram_active_s, residual_j, measured_energy_j in rows:
        if measured_energy_j <= 0.0:
            continue
        weight = 1.0 / measured_energy_j
        predicted = (
            time_kernel_s * coefficients[0]
            + time_dram_active_s * coefficients[1]
        )
        total += ((predicted - residual_j) * weight) ** 2
    return total


def _sample_class(sample: ArtifactSample) -> str:
    batch = int(sample.dimensions["batch"])
    dim_m = int(sample.dimensions["M"])
    dim_n = int(sample.dimensions["N"])
    dim_k = int(sample.dimensions["K"])
    return classify_gemm_size(batch=batch, dim_m=dim_m, dim_n=dim_n, dim_k=dim_k)


def _sample_key(sample: ArtifactSample) -> tuple[str, int]:
    return sample.source_file, sample.row_index


def _metrics_by_level(
    samples: tuple[ArtifactSample, ...],
    hardware: HardwareSpec,
) -> dict[str, CalibrationMetrics]:
    model = create_model("effective_roofline")
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
