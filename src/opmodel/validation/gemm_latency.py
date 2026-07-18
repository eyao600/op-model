from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from opmodel.api import OpKind
from opmodel.hardware import HardwareSpec, load_hardware
from opmodel.registry import create_model
from opmodel.validation.artifact_accuracy import (
    DEFAULT_HARDWARE_DIR,
    ArtifactSample,
    _csv_value,
    _hardware_info_for_file,
    _is_bf16_row,
    _matches_config_frequency,
    _sample_from_row,
    _supported_op_kind,
)


DEFAULT_GEMM_LATENCY_DATA_DIR = Path("data")
DEFAULT_TRAINING_PER_CLASS = 8
DEFAULT_ROOFLINE_COMPARISON_MODELS = (
    "roofline",
    "extended_roofline",
    "effective_roofline",
)
_TRAINING_CLASSES = ("vector_like", "small")
_VALIDATION_GROUPS = ("all", "hardware", "kernel_class")


@dataclass(frozen=True)
class GemmLatencyRow:
    source_file: str
    row_index: int
    split: str
    hardware: str
    kernel_class: str
    batch: int
    dim_m: int
    dim_n: int
    dim_k: int
    trans: str
    measured_latency_ms: float
    predicted_latency_ms: float
    latency_ratio: float
    latency_abs_pct_error: float
    latency_signed_pct_error: float
    measured_energy_j: float
    predicted_energy_j: float
    energy_ratio: float
    energy_abs_pct_error: float
    energy_signed_pct_error: float
    fixed_overhead_cycles: int
    modeled_device_cycles: float | None
    total_device_cycles: float | None
    predicted_tflops_per_s: float | None
    primary_bottleneck: str | None
    secondary_bottlenecks: Any


@dataclass(frozen=True)
class GemmLatencyMetrics:
    group: str
    name: str
    split: str
    count: int
    energy_count: int
    latency_mape_pct: float
    latency_median_ape_pct: float
    latency_p90_ape_pct: float
    latency_geomean_ratio: float
    latency_mean_signed_pct_error: float
    energy_mape_pct: float
    energy_median_ape_pct: float
    energy_p90_ape_pct: float
    energy_geomean_ratio: float
    energy_mean_signed_pct_error: float


@dataclass(frozen=True)
class FixedOverheadCalibration:
    hardware: str
    base_fixed_overhead_cycles: int | None
    calibrated_fixed_overhead_cycles: int
    training_rows: tuple[tuple[str, int], ...]
    candidate_rows: int


@dataclass(frozen=True)
class GemmLatencyValidationReport:
    rows: tuple[GemmLatencyRow, ...]
    metrics: tuple[GemmLatencyMetrics, ...]
    fixed_overheads: Mapping[str, FixedOverheadCalibration]
    skip_counts: Mapping[str, int]


@dataclass(frozen=True)
class GemmLatencyModelComparison:
    model_name: str
    baseline_model_name: str
    group: str
    name: str
    count: int
    energy_count: int
    latency_mape_pct: float
    latency_mape_delta_pct: float
    latency_median_ape_pct: float
    latency_p90_ape_pct: float
    latency_geomean_ratio: float
    energy_mape_pct: float
    energy_mape_delta_pct: float
    energy_median_ape_pct: float
    energy_p90_ape_pct: float
    energy_geomean_ratio: float


@dataclass(frozen=True)
class GemmRooflineComparisonReport:
    baseline_model_name: str
    model_names: tuple[str, ...]
    reports: Mapping[str, GemmLatencyValidationReport]
    metrics: tuple[GemmLatencyModelComparison, ...]


def run_gemm_roofline_comparison(
    *,
    data_dir: str | Path = DEFAULT_GEMM_LATENCY_DATA_DIR,
    hardware_dir: str | Path = DEFAULT_HARDWARE_DIR,
    model_names: Iterable[str] = DEFAULT_ROOFLINE_COMPARISON_MODELS,
    baseline_model_name: str = "roofline",
    calibrate_fixed_overhead: bool = True,
    training_per_class: int = DEFAULT_TRAINING_PER_CLASS,
    limit: int | None = None,
) -> GemmRooflineComparisonReport:
    names = tuple(str(name) for name in model_names)
    if not names:
        raise ValueError("model_names must contain at least one model")
    if len(set(names)) != len(names):
        raise ValueError("model_names must be unique")
    if baseline_model_name not in names:
        raise ValueError("baseline_model_name must be included in model_names")

    reports = {
        name: run_gemm_latency_validation(
            data_dir=data_dir,
            hardware_dir=hardware_dir,
            model_name=name,
            calibrate_fixed_overhead=calibrate_fixed_overhead,
            training_per_class=training_per_class,
            limit=limit,
        )
        for name in names
    }
    baseline_metrics = {
        (metric.group, metric.name): metric
        for metric in reports[baseline_model_name].metrics
    }
    comparisons: list[GemmLatencyModelComparison] = []
    for name in names:
        for metric in reports[name].metrics:
            baseline = baseline_metrics.get((metric.group, metric.name))
            if baseline is None:
                raise ValueError(
                    "model reports have incompatible validation groups: "
                    f"{name} has {metric.group}/{metric.name} but the baseline does not"
                )
            comparisons.append(
                GemmLatencyModelComparison(
                    model_name=name,
                    baseline_model_name=baseline_model_name,
                    group=metric.group,
                    name=metric.name,
                    count=metric.count,
                    energy_count=metric.energy_count,
                    latency_mape_pct=metric.latency_mape_pct,
                    latency_mape_delta_pct=(
                        metric.latency_mape_pct - baseline.latency_mape_pct
                    ),
                    latency_median_ape_pct=metric.latency_median_ape_pct,
                    latency_p90_ape_pct=metric.latency_p90_ape_pct,
                    latency_geomean_ratio=metric.latency_geomean_ratio,
                    energy_mape_pct=metric.energy_mape_pct,
                    energy_mape_delta_pct=(
                        metric.energy_mape_pct - baseline.energy_mape_pct
                    ),
                    energy_median_ape_pct=metric.energy_median_ape_pct,
                    energy_p90_ape_pct=metric.energy_p90_ape_pct,
                    energy_geomean_ratio=metric.energy_geomean_ratio,
                )
            )
    return GemmRooflineComparisonReport(
        baseline_model_name=baseline_model_name,
        model_names=names,
        reports=reports,
        metrics=tuple(comparisons),
    )


def run_gemm_latency_validation(
    *,
    data_dir: str | Path = DEFAULT_GEMM_LATENCY_DATA_DIR,
    hardware_dir: str | Path = DEFAULT_HARDWARE_DIR,
    model_name: str = "extended_roofline",
    calibrate_fixed_overhead: bool = True,
    training_per_class: int = DEFAULT_TRAINING_PER_CLASS,
    limit: int | None = None,
) -> GemmLatencyValidationReport:
    if training_per_class <= 0:
        raise ValueError("training_per_class must be positive")

    samples, skip_counts = _gemm_samples(data_dir=data_dir, limit=limit)
    base_hardware = _load_hardware_for_samples(samples, hardware_dir)
    model = create_model(model_name)

    training_keys_by_hardware: dict[str, set[tuple[str, int]]] = defaultdict(set)
    calibrated_hardware: dict[str, HardwareSpec] = {}
    overheads: dict[str, FixedOverheadCalibration] = {}
    for hardware_name, hardware in base_hardware.items():
        hardware_samples = tuple(
            sample for sample in samples if sample.hardware == hardware_name
        )
        if calibrate_fixed_overhead:
            selected = _select_training_samples(
                hardware_samples,
                training_per_class=training_per_class,
            )
            overhead = _estimate_fixed_overhead_cycles(
                samples=selected,
                hardware=hardware,
                model=model,
            )
        else:
            selected = ()
            overhead = int(hardware.compute.device_fixed_overhead_cycles or 0)

        keys = {_sample_key(sample) for sample in selected}
        training_keys_by_hardware[hardware_name] = keys
        calibrated_hardware[hardware_name] = _with_fixed_overhead(hardware, overhead)
        overheads[hardware_name] = FixedOverheadCalibration(
            hardware=hardware_name,
            base_fixed_overhead_cycles=hardware.compute.device_fixed_overhead_cycles,
            calibrated_fixed_overhead_cycles=overhead,
            training_rows=tuple(sorted(keys)),
            candidate_rows=sum(
                1 for sample in hardware_samples if _is_training_candidate(sample)
            ),
        )

    rows: list[GemmLatencyRow] = []
    for sample in samples:
        split = (
            "train"
            if _sample_key(sample) in training_keys_by_hardware[sample.hardware]
            else "validation"
        )
        rows.append(
            _predict_row(
                sample=sample,
                hardware=calibrated_hardware[sample.hardware],
                model=model,
                split=split,
            )
        )

    validation_rows = tuple(row for row in rows if row.split == "validation")
    return GemmLatencyValidationReport(
        rows=tuple(rows),
        metrics=_aggregate_metrics(validation_rows),
        fixed_overheads=overheads,
        skip_counts=dict(skip_counts),
    )


def write_gemm_latency_csv(
    report: GemmLatencyValidationReport, path: str | Path
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({field: _csv_value(getattr(row, field)) for field in _CSV_FIELDS})


def write_gemm_latency_params(
    report: GemmLatencyValidationReport, path: str | Path
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        hardware: {
            "base_fixed_overhead_cycles": item.base_fixed_overhead_cycles,
            "calibrated_fixed_overhead_cycles": item.calibrated_fixed_overhead_cycles,
            "candidate_rows": item.candidate_rows,
            "training_rows": [
                {"source_file": source_file, "row_index": row_index}
                for source_file, row_index in item.training_rows
            ],
        }
        for hardware, item in sorted(report.fixed_overheads.items())
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_gemm_latency_report(report: GemmLatencyValidationReport) -> str:
    train_count = sum(1 for row in report.rows if row.split == "train")
    validation_count = sum(1 for row in report.rows if row.split == "validation")
    lines = [
        f"Processed GEMM rows: {len(report.rows)}",
        f"Training rows for fixed overhead: {train_count}",
        f"Held-out validation rows: {validation_count}",
    ]
    if report.fixed_overheads:
        lines.append("Fixed overhead cycles:")
        for hardware, item in sorted(report.fixed_overheads.items()):
            lines.append(
                "  "
                f"{hardware}: base={_fmt_optional_int(item.base_fixed_overhead_cycles)}, "
                f"used={item.calibrated_fixed_overhead_cycles}, "
                f"train={len(item.training_rows)}, "
                f"candidates={item.candidate_rows}"
            )
    if report.metrics:
        lines.append("Held-out latency accuracy:")
        for metric in report.metrics:
            lines.append(
                "  "
                f"{metric.group}/{metric.name}: "
                f"n={metric.count}, "
                f"MAPE={metric.latency_mape_pct:.2f}%, "
                f"median={metric.latency_median_ape_pct:.2f}%, "
                f"p90={metric.latency_p90_ape_pct:.2f}%, "
                f"ratio={metric.latency_geomean_ratio:.3g}, "
                f"signed={metric.latency_mean_signed_pct_error:.2f}%"
            )
        lines.append("Held-out energy accuracy:")
        for metric in report.metrics:
            if metric.energy_count == 0:
                lines.append(
                    "  "
                    f"{metric.group}/{metric.name}: "
                    "disabled"
                )
                continue
            lines.append(
                "  "
                f"{metric.group}/{metric.name}: "
                f"n={metric.energy_count}, "
                f"MAPE={metric.energy_mape_pct:.2f}%, "
                f"median={metric.energy_median_ape_pct:.2f}%, "
                f"p90={metric.energy_p90_ape_pct:.2f}%, "
                f"ratio={metric.energy_geomean_ratio:.3g}, "
                f"signed={metric.energy_mean_signed_pct_error:.2f}%"
            )
    else:
        lines.append("No held-out validation rows were processed.")
    if report.skip_counts:
        lines.append("Skipped:")
        for reason, count in sorted(report.skip_counts.items()):
            lines.append(f"  {reason}: {count}")
    return "\n".join(lines)


def format_gemm_roofline_comparison(
    report: GemmRooflineComparisonReport,
) -> str:
    baseline = report.baseline_model_name
    baseline_report = report.reports[baseline]
    validation_count = sum(
        1 for row in baseline_report.rows if row.split == "validation"
    )
    lines = [
        "GEMM roofline model comparison:",
        f"  models={', '.join(report.model_names)}",
        f"  baseline={baseline}",
        f"  held-out rows={validation_count}",
        "Held-out latency accuracy (MAPE delta versus baseline):",
    ]
    for group in _VALIDATION_GROUPS:
        group_metrics = tuple(metric for metric in report.metrics if metric.group == group)
        group_names = tuple(dict.fromkeys(metric.name for metric in group_metrics))
        for group_name in group_names:
            lines.append(f"  {group}/{group_name}:")
            by_model = {
                metric.model_name: metric
                for metric in group_metrics
                if metric.name == group_name
            }
            for model_name in report.model_names:
                metric = by_model[model_name]
                lines.append(
                    "    "
                    f"{model_name}: n={metric.count}, "
                    f"MAPE={metric.latency_mape_pct:.2f}%, "
                    f"delta={metric.latency_mape_delta_pct:+.2f}pp, "
                    f"median={metric.latency_median_ape_pct:.2f}%, "
                    f"p90={metric.latency_p90_ape_pct:.2f}%, "
                    f"ratio={metric.latency_geomean_ratio:.3g}"
                )
    lines.append("Held-out energy accuracy (MAPE delta versus baseline):")
    for group in _VALIDATION_GROUPS:
        group_metrics = tuple(metric for metric in report.metrics if metric.group == group)
        group_names = tuple(dict.fromkeys(metric.name for metric in group_metrics))
        for group_name in group_names:
            lines.append(f"  {group}/{group_name}:")
            by_model = {
                metric.model_name: metric
                for metric in group_metrics
                if metric.name == group_name
            }
            for model_name in report.model_names:
                metric = by_model[model_name]
                if metric.energy_count == 0:
                    lines.append(f"    {model_name}: disabled")
                    continue
                lines.append(
                    "    "
                    f"{model_name}: n={metric.energy_count}, "
                    f"MAPE={metric.energy_mape_pct:.2f}%, "
                    f"delta={metric.energy_mape_delta_pct:+.2f}pp, "
                    f"median={metric.energy_median_ape_pct:.2f}%, "
                    f"p90={metric.energy_p90_ape_pct:.2f}%, "
                    f"ratio={metric.energy_geomean_ratio:.3g}"
                )
    lines.append("Calibrated fixed overhead cycles by model/hardware:")
    for model_name in report.model_names:
        overheads = report.reports[model_name].fixed_overheads
        values = ", ".join(
            f"{hardware}={item.calibrated_fixed_overhead_cycles}"
            for hardware, item in sorted(overheads.items())
        )
        lines.append(f"  {model_name}: {values or 'none'}")
    return "\n".join(lines)


def classify_gemm_size(*, batch: int, dim_m: int, dim_n: int, dim_k: int) -> str:
    min_mn = min(dim_m, dim_n)
    max_mn = max(dim_m, dim_n)
    if dim_m == 1 and dim_n > 1 and dim_k > 1:
        return "vector_like"
    if dim_m * dim_n <= 4096:
        return "small"
    if min_mn <= 64 and max_mn / max(min_mn, 1) >= 8:
        return "skinny"
    if dim_k <= 32:
        return "small_k"
    if dim_m * dim_n >= 1_048_576 or 2 * batch * dim_m * dim_n * dim_k >= 1.0e12:
        return "large"
    return "regular"


def _gemm_samples(
    *, data_dir: str | Path, limit: int | None
) -> tuple[tuple[ArtifactSample, ...], Mapping[str, int]]:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"GEMM latency data directory does not exist: {data_path}")

    samples: list[ArtifactSample] = []
    skip_counts: Counter[str] = Counter()
    for csv_path in sorted(data_path.glob("*.csv")):
        hardware_info = _hardware_info_for_file(csv_path.name)
        if hardware_info is None:
            skip_counts["unsupported_hardware_file"] += 1
            continue
        if not _matches_config_frequency(csv_path.name):
            skip_counts["unsupported_frequency_file"] += 1
            continue
        op_kind = _supported_op_kind(csv_path.name)
        if op_kind != OpKind.BATCHED_GEMM.value:
            if "bf16" in csv_path.name.lower():
                skip_counts["non_gemm_bf16_file"] += 1
            else:
                skip_counts["non_gemm_file"] += 1
            continue

        hardware_name, _hardware_filename = hardware_info
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=1):
                if limit is not None and len(samples) >= limit:
                    return tuple(samples), dict(skip_counts)
                if not _is_bf16_row(row):
                    skip_counts["non_bf16_row"] += 1
                    continue
                try:
                    samples.append(
                        _sample_from_row(
                            csv_path,
                            row_index,
                            hardware_name,
                            op_kind,
                            row,
                        )
                    )
                except (KeyError, ValueError) as exc:
                    reason = str(exc).splitlines()[0]
                    skip_counts[f"invalid_row:{reason}"] += 1
    return tuple(samples), dict(skip_counts)


def _load_hardware_for_samples(
    samples: Iterable[ArtifactSample],
    hardware_dir: str | Path,
) -> dict[str, HardwareSpec]:
    hardware_path = Path(hardware_dir)
    hardware: dict[str, HardwareSpec] = {}
    for sample in samples:
        if sample.hardware in hardware:
            continue
        info = _hardware_info_for_file(sample.source_file)
        if info is None:
            continue
        hardware_name, hardware_filename = info
        hardware[hardware_name] = load_hardware(hardware_path / hardware_filename)
    return hardware


def _select_training_samples(
    samples: tuple[ArtifactSample, ...],
    *,
    training_per_class: int,
) -> tuple[ArtifactSample, ...]:
    selected: list[ArtifactSample] = []
    for kernel_class in _TRAINING_CLASSES:
        candidates = tuple(
            sample
            for sample in samples
            if _sample_class(sample) == kernel_class and _is_training_candidate(sample)
        )
        selected.extend(
            _stratified_by_log_flops(
                candidates,
                count=min(training_per_class, len(candidates)),
            )
        )
    return tuple(sorted(selected, key=_training_tiebreak_key))


def _stratified_by_log_flops(
    samples: tuple[ArtifactSample, ...], *, count: int
) -> tuple[ArtifactSample, ...]:
    if count <= 0 or not samples:
        return ()
    ranked = sorted(samples, key=_training_tiebreak_key)
    if count >= len(ranked):
        return tuple(ranked)

    by_flops = sorted(ranked, key=lambda sample: (_log2_flops(sample), _training_tiebreak_key(sample)))
    selected: list[ArtifactSample] = []
    used: set[tuple[str, int]] = set()
    for index in range(count):
        start = math.floor(index * len(by_flops) / count)
        end = math.floor((index + 1) * len(by_flops) / count)
        bin_samples = by_flops[start:end] or [by_flops[min(start, len(by_flops) - 1)]]
        target = _median(_log2_flops(sample) for sample in bin_samples)
        best = min(
            (sample for sample in bin_samples if _sample_key(sample) not in used),
            key=lambda sample: (abs(_log2_flops(sample) - target), _training_tiebreak_key(sample)),
            default=None,
        )
        if best is None:
            best = min(
                (sample for sample in by_flops if _sample_key(sample) not in used),
                key=lambda sample: (abs(_log2_flops(sample) - target), _training_tiebreak_key(sample)),
            )
        selected.append(best)
        used.add(_sample_key(best))
    return tuple(selected)


def _estimate_fixed_overhead_cycles(
    *,
    samples: tuple[ArtifactSample, ...],
    hardware: HardwareSpec,
    model: Any,
) -> int:
    if not samples:
        return int(hardware.compute.device_fixed_overhead_cycles or 0)

    zero_overhead_hardware = _with_fixed_overhead(hardware, 0)
    residuals: list[float] = []
    for sample in samples:
        profile = model.predict(sample.op, zero_overhead_hardware)
        clock_hz = _profile_clock_hz(profile, hardware)
        modeled_cycles = _profile_modeled_cycles(profile, clock_hz)
        measured_cycles = sample.measured_latency_ms * clock_hz / 1000.0
        residuals.append(measured_cycles - modeled_cycles)
    return int(round(max(0.0, _median(residuals))))


def _predict_row(
    *,
    sample: ArtifactSample,
    hardware: HardwareSpec,
    model: Any,
    split: str,
) -> GemmLatencyRow:
    profile = model.predict(sample.op, hardware)
    diagnostics = profile.diagnostics
    clock_hz = _profile_clock_hz(profile, hardware)
    modeled_cycles = _profile_modeled_cycles(profile, clock_hz)
    native_total_cycles = _optional_float(diagnostics.get("total_device_cycles"))
    fixed_overhead_cycles = int(hardware.compute.device_fixed_overhead_cycles or 0)
    if native_total_cycles is None:
        total_device_cycles = modeled_cycles + fixed_overhead_cycles
        predicted_latency_s = total_device_cycles / clock_hz
        predicted_energy_j = (
            profile.energy_j
            + hardware.static_power_w * fixed_overhead_cycles / clock_hz
        )
    else:
        total_device_cycles = native_total_cycles
        predicted_latency_s = profile.latency_s
        predicted_energy_j = profile.energy_j
    predicted_latency_ms = predicted_latency_s * 1000.0
    latency_ratio = predicted_latency_ms / sample.measured_latency_ms
    energy_ratio = predicted_energy_j / sample.measured_energy_j
    dims = _sample_dims(sample)
    return GemmLatencyRow(
        source_file=sample.source_file,
        row_index=sample.row_index,
        split=split,
        hardware=sample.hardware,
        kernel_class=classify_gemm_size(
            batch=dims["batch"],
            dim_m=dims["M"],
            dim_n=dims["N"],
            dim_k=dims["K"],
        ),
        batch=dims["batch"],
        dim_m=dims["M"],
        dim_n=dims["N"],
        dim_k=dims["K"],
        trans=str(dims.get("trans", "")),
        measured_latency_ms=sample.measured_latency_ms,
        predicted_latency_ms=predicted_latency_ms,
        latency_ratio=latency_ratio,
        latency_abs_pct_error=abs(latency_ratio - 1.0) * 100.0,
        latency_signed_pct_error=(latency_ratio - 1.0) * 100.0,
        measured_energy_j=sample.measured_energy_j,
        predicted_energy_j=predicted_energy_j,
        energy_ratio=energy_ratio,
        energy_abs_pct_error=abs(energy_ratio - 1.0) * 100.0,
        energy_signed_pct_error=(energy_ratio - 1.0) * 100.0,
        fixed_overhead_cycles=fixed_overhead_cycles,
        modeled_device_cycles=modeled_cycles,
        total_device_cycles=total_device_cycles,
        predicted_tflops_per_s=(
            _optional_float(diagnostics.get("predicted_tflops_per_s"))
            or (profile.flops / predicted_latency_s / 1.0e12 if predicted_latency_s else 0.0)
        ),
        primary_bottleneck=(
            _optional_str(diagnostics.get("primary_bottleneck"))
            or _base_roofline_bottleneck(diagnostics)
        ),
        secondary_bottlenecks=diagnostics.get("secondary_bottlenecks"),
    )


def _profile_clock_hz(profile: Any, hardware: HardwareSpec) -> float:
    return float(
        profile.diagnostics.get("clock_hz")
        or hardware.compute.clock_hz
        or 1.0e9
    )


def _profile_modeled_cycles(profile: Any, clock_hz: float) -> float:
    modeled = profile.diagnostics.get("modeled_device_cycles")
    return float(modeled) if modeled is not None else float(profile.latency_s * clock_hz)


def _base_roofline_bottleneck(diagnostics: Mapping[str, Any]) -> str | None:
    compute = diagnostics.get("compute_latency_s")
    memory = diagnostics.get("memory_latency_s")
    if compute is None or memory is None:
        return None
    return "compute" if float(compute) >= float(memory) else "hbm"


def _aggregate_metrics(rows: tuple[GemmLatencyRow, ...]) -> tuple[GemmLatencyMetrics, ...]:
    groups: dict[tuple[str, str], list[GemmLatencyRow]] = defaultdict(list)
    for row in rows:
        groups[("all", "all")].append(row)
        groups[("hardware", row.hardware)].append(row)
        groups[("kernel_class", row.kernel_class)].append(row)

    metrics: list[GemmLatencyMetrics] = []
    for group in _VALIDATION_GROUPS:
        names = sorted(name for group_name, name in groups if group_name == group)
        if group == "all" and "all" in names:
            names = ["all"]
        for name in names:
            metrics.append(_metrics(group, name, groups[(group, name)]))
    return tuple(metrics)


def _metrics(
    group: str,
    name: str,
    rows: list[GemmLatencyRow],
) -> GemmLatencyMetrics:
    latency_ape = sorted(row.latency_abs_pct_error for row in rows)
    latency_signed = [row.latency_signed_pct_error for row in rows]
    energy_rows = [row for row in rows if _energy_validation_enabled(row)]
    energy_ape = sorted(row.energy_abs_pct_error for row in energy_rows)
    energy_signed = [row.energy_signed_pct_error for row in energy_rows]
    return GemmLatencyMetrics(
        group=group,
        name=name,
        split="validation",
        count=len(rows),
        energy_count=len(energy_rows),
        latency_mape_pct=_mean(latency_ape),
        latency_median_ape_pct=_percentile(latency_ape, 0.5),
        latency_p90_ape_pct=_percentile(latency_ape, 0.9),
        latency_geomean_ratio=_geomean(row.latency_ratio for row in rows),
        latency_mean_signed_pct_error=_mean(latency_signed),
        energy_mape_pct=_mean(energy_ape),
        energy_median_ape_pct=_percentile(energy_ape, 0.5),
        energy_p90_ape_pct=_percentile(energy_ape, 0.9),
        energy_geomean_ratio=_geomean(row.energy_ratio for row in energy_rows),
        energy_mean_signed_pct_error=_mean(energy_signed),
    )


def _energy_validation_enabled(row: GemmLatencyRow) -> bool:
    return row.hardware != "a10"


def _sample_class(sample: ArtifactSample) -> str:
    dims = _sample_dims(sample)
    return classify_gemm_size(
        batch=dims["batch"],
        dim_m=dims["M"],
        dim_n=dims["N"],
        dim_k=dims["K"],
    )


def _is_training_candidate(sample: ArtifactSample) -> bool:
    dims = _sample_dims(sample)
    return _sample_class(sample) in _TRAINING_CLASSES and dims["K"] <= 128


def _sample_dims(sample: ArtifactSample) -> dict[str, Any]:
    return {
        "batch": int(sample.dimensions["batch"]),
        "M": int(sample.dimensions["M"]),
        "N": int(sample.dimensions["N"]),
        "K": int(sample.dimensions["K"]),
        "trans": sample.dimensions.get("trans", ""),
    }


def _sample_key(sample: ArtifactSample) -> tuple[str, int]:
    return sample.source_file, sample.row_index


def _training_tiebreak_key(sample: ArtifactSample) -> tuple[int, int, int, int, str, int]:
    dims = _sample_dims(sample)
    return (
        dims["batch"],
        dims["M"],
        dims["N"],
        dims["K"],
        sample.source_file,
        sample.row_index,
    )


def _log2_flops(sample: ArtifactSample) -> float:
    dims = _sample_dims(sample)
    flops = 2 * dims["batch"] * dims["M"] * dims["N"] * dims["K"]
    return math.log2(max(1, flops))


def _with_fixed_overhead(hardware: HardwareSpec, fixed_overhead_cycles: int) -> HardwareSpec:
    return replace(
        hardware,
        compute=replace(
            hardware.compute,
            device_fixed_overhead_cycles=int(fixed_overhead_cycles),
        ),
    )


def _median(values: Iterable[float]) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return float("nan")
    return _percentile(sorted_values, 0.5)


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


def _mean(values: Iterable[float]) -> float:
    values_tuple = tuple(float(value) for value in values)
    if not values_tuple:
        return float("nan")
    return sum(values_tuple) / len(values_tuple)


def _geomean(values: Iterable[float]) -> float:
    positive = [float(value) for value in values if value > 0.0]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _fmt_optional_int(value: int | None) -> str:
    return "none" if value is None else str(value)


_CSV_FIELDS = (
    "source_file",
    "row_index",
    "split",
    "hardware",
    "kernel_class",
    "batch",
    "dim_m",
    "dim_n",
    "dim_k",
    "trans",
    "measured_latency_ms",
    "predicted_latency_ms",
    "latency_ratio",
    "latency_abs_pct_error",
    "latency_signed_pct_error",
    "measured_energy_j",
    "predicted_energy_j",
    "energy_ratio",
    "energy_abs_pct_error",
    "energy_signed_pct_error",
    "fixed_overhead_cycles",
    "modeled_device_cycles",
    "total_device_cycles",
    "predicted_tflops_per_s",
    "primary_bottleneck",
    "secondary_bottlenecks",
)
