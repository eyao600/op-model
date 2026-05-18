from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping

from opmodel.api import DType, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.hardware import HardwareSpec, load_hardware
from opmodel.registry import create_model


DEFAULT_ARTIFACT_DATA_DIR = Path(
    "/app/nanocad/projects/personal/yaoe888/energaizer-ispass26-artifact/database/data"
)
DEFAULT_HARDWARE_DIR = Path(__file__).resolve().parents[1] / "configs" / "hardware"

_HARDWARE_BY_PREFIX = {
    "yz8_": ("a100_40gb_pcie", "a100_40gb_pcie.yaml"),
    "a10_": ("a10", "a10.yaml"),
}


@dataclass(frozen=True)
class ArtifactSample:
    source_file: str
    row_index: int
    hardware: str
    op_kind: str
    dimensions: Mapping[str, Any]
    measured_latency_ms: float
    measured_energy_j: float
    op: LocalOp


@dataclass(frozen=True)
class ArtifactAccuracyRow:
    source_file: str
    row_index: int
    hardware: str
    op_kind: str
    dimensions: str
    measured_latency_ms: float
    predicted_latency_ms: float
    latency_ratio: float
    latency_abs_pct_error: float
    measured_energy_j: float
    predicted_energy_j: float
    energy_ratio: float
    energy_abs_pct_error: float


@dataclass(frozen=True)
class AccuracyAggregate:
    hardware: str
    op_kind: str
    count: int
    latency_geomean_ratio: float
    energy_geomean_ratio: float
    latency_mape_pct: float
    energy_mape_pct: float


@dataclass(frozen=True)
class ArtifactValidationReport:
    rows: tuple[ArtifactAccuracyRow, ...]
    skip_counts: Mapping[str, int]

    def aggregates(self) -> tuple[AccuracyAggregate, ...]:
        groups: dict[tuple[str, str], list[ArtifactAccuracyRow]] = defaultdict(list)
        for row in self.rows:
            groups[(row.hardware, row.op_kind)].append(row)

        aggregates: list[AccuracyAggregate] = []
        for (hardware, op_kind), rows in sorted(groups.items()):
            aggregates.append(
                AccuracyAggregate(
                    hardware=hardware,
                    op_kind=op_kind,
                    count=len(rows),
                    latency_geomean_ratio=_geomean(row.latency_ratio for row in rows),
                    energy_geomean_ratio=_geomean(row.energy_ratio for row in rows),
                    latency_mape_pct=_mean(row.latency_abs_pct_error for row in rows),
                    energy_mape_pct=_mean(row.energy_abs_pct_error for row in rows),
                )
            )
        return tuple(aggregates)


def run_artifact_validation(
    *,
    data_dir: str | Path = DEFAULT_ARTIFACT_DATA_DIR,
    hardware_dir: str | Path = DEFAULT_HARDWARE_DIR,
    model_name: str = "roofline",
    limit: int | None = None,
) -> ArtifactValidationReport:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Artifact data directory does not exist: {data_path}")

    hardware_path = Path(hardware_dir)
    model = create_model(model_name)
    hardware_cache: dict[str, HardwareSpec] = {}
    rows: list[ArtifactAccuracyRow] = []
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
        if op_kind is None:
            if "bf16" in csv_path.name.lower():
                skip_counts["unsupported_bf16_file"] += 1
            else:
                skip_counts["non_bf16_file"] += 1
            continue

        hardware_name, hardware_filename = hardware_info
        hardware = hardware_cache.get(hardware_name)
        if hardware is None:
            hardware = load_hardware(hardware_path / hardware_filename)
            hardware_cache[hardware_name] = hardware

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            file_rows = 0
            for row_index, row in enumerate(reader, start=1):
                if limit is not None and file_rows >= limit:
                    break
                if not _is_bf16_row(row):
                    skip_counts["non_bf16_row"] += 1
                    continue
                try:
                    sample = _sample_from_row(csv_path, row_index, hardware_name, op_kind, row)
                    rows.append(_predict_sample(sample, hardware, model))
                    file_rows += 1
                except (KeyError, ValueError) as exc:
                    reason = str(exc).splitlines()[0]
                    skip_counts[f"invalid_row:{reason}"] += 1

    return ArtifactValidationReport(rows=tuple(rows), skip_counts=dict(skip_counts))


def write_csv_report(report: ArtifactValidationReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({field: getattr(row, field) for field in _CSV_FIELDS})


def write_normalized_bar_plot(report: ArtifactValidationReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_normalized_bar_plot_svg(report.aggregates()), encoding="utf-8")


def format_text_report(report: ArtifactValidationReport) -> str:
    lines = [f"Processed supported rows: {len(report.rows)}"]
    aggregates = report.aggregates()
    if aggregates:
        lines.append("Aggregate accuracy by hardware/op:")
        for aggregate in aggregates:
            lines.append(
                "  "
                f"{aggregate.hardware}/{aggregate.op_kind}: "
                f"n={aggregate.count}, "
                f"time MAPE={aggregate.latency_mape_pct:.2f}%, "
                f"energy MAPE={aggregate.energy_mape_pct:.2f}%, "
                f"time ratio={aggregate.latency_geomean_ratio:.3g}, "
                f"energy ratio={aggregate.energy_geomean_ratio:.3g}"
            )
    else:
        lines.append("No supported bf16 rows were processed.")

    if report.skip_counts:
        lines.append("Skipped:")
        for reason, count in sorted(report.skip_counts.items()):
            lines.append(f"  {reason}: {count}")
    return "\n".join(lines)


def _sample_from_row(
    csv_path: Path,
    row_index: int,
    hardware: str,
    op_kind: str,
    row: Mapping[str, str],
) -> ArtifactSample:
    measured_latency_ms = _float_field(row, "time")
    measured_energy_j = _float_field(row, "energy")
    if measured_latency_ms <= 0.0:
        raise ValueError("time must be positive")
    if measured_energy_j <= 0.0:
        raise ValueError("energy must be positive")

    if op_kind == OpKind.BATCHED_GEMM.value:
        dimensions, op = _gemm_sample(csv_path, row_index, row)
    elif op_kind == OpKind.LAYERNORM.value:
        dimensions, op = _layernorm_sample(csv_path, row_index, row)
    elif op_kind == OpKind.SOFTMAX.value:
        dimensions, op = _softmax_sample(csv_path, row_index, row)
    else:
        raise ValueError(f"unsupported op kind {op_kind}")

    return ArtifactSample(
        source_file=csv_path.name,
        row_index=row_index,
        hardware=hardware,
        op_kind=op_kind,
        dimensions=dimensions,
        measured_latency_ms=measured_latency_ms,
        measured_energy_j=measured_energy_j,
        op=op,
    )


def _gemm_sample(
    csv_path: Path, row_index: int, row: Mapping[str, str]
) -> tuple[Mapping[str, Any], LocalOp]:
    batch = _int_field(row, "batch")
    dim_m = _int_field(row, "dimM")
    dim_n = _int_field(row, "dimN")
    dim_k = _int_field(row, "dimK")
    trans = str(row.get("trans", "nn")).lower()
    transpose_a = len(trans) >= 1 and trans[0] == "t"
    transpose_b = len(trans) >= 2 and trans[1] == "t"

    a_shape = (batch, dim_k, dim_m) if transpose_a else (batch, dim_m, dim_k)
    b_shape = (batch, dim_n, dim_k) if transpose_b else (batch, dim_k, dim_n)
    c_shape = (batch, dim_m, dim_n)
    attrs = {"transpose_a": transpose_a, "transpose_b": transpose_b}
    dimensions = {
        "batch": batch,
        "M": dim_m,
        "N": dim_n,
        "K": dim_k,
        "trans": trans,
    }
    return dimensions, LocalOp(
        name=_op_name(csv_path, row_index),
        kind=OpKind.BATCHED_GEMM,
        phase=Phase.INFERENCE,
        attrs=attrs,
        tensors=(
            TensorSpec(TensorRole.INPUT, a_shape, DType.BF16),
            TensorSpec(TensorRole.WEIGHT, b_shape, DType.BF16),
            TensorSpec(TensorRole.OUTPUT, c_shape, DType.BF16),
        ),
    )


def _layernorm_sample(
    csv_path: Path, row_index: int, row: Mapping[str, str]
) -> tuple[Mapping[str, Any], LocalOp]:
    batch = _int_field(row, "batch")
    dim = _int_field(row, "dim")
    shape = (batch, dim)
    dimensions = {"batch": batch, "dim": dim}
    return dimensions, LocalOp(
        name=_op_name(csv_path, row_index),
        kind=OpKind.LAYERNORM,
        phase=Phase.INFERENCE,
        attrs={"normalized_shape": dim, "affine": True},
        tensors=(
            TensorSpec(TensorRole.INPUT, shape, DType.BF16),
            TensorSpec(TensorRole.WEIGHT, (dim,), DType.BF16),
            TensorSpec(TensorRole.WEIGHT, (dim,), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, shape, DType.BF16),
        ),
    )


def _softmax_sample(
    csv_path: Path, row_index: int, row: Mapping[str, str]
) -> tuple[Mapping[str, Any], LocalOp]:
    batch = _int_field(row, "batch")
    dim = _int_field(row, "dim")
    shape = (batch, dim)
    dimensions = {"batch": batch, "dim": dim}
    return dimensions, LocalOp(
        name=_op_name(csv_path, row_index),
        kind=OpKind.SOFTMAX,
        phase=Phase.INFERENCE,
        attrs={"row_size": dim},
        tensors=(
            TensorSpec(TensorRole.INPUT, shape, DType.BF16),
            TensorSpec(TensorRole.OUTPUT, shape, DType.BF16),
        ),
    )


def _predict_sample(
    sample: ArtifactSample, hardware: HardwareSpec, model: Any
) -> ArtifactAccuracyRow:
    profile = model.predict(sample.op, hardware)
    predicted_latency_ms = profile.latency_s * 1000.0
    predicted_energy_j = profile.energy_j
    latency_ratio = predicted_latency_ms / sample.measured_latency_ms
    energy_ratio = predicted_energy_j / sample.measured_energy_j
    return ArtifactAccuracyRow(
        source_file=sample.source_file,
        row_index=sample.row_index,
        hardware=sample.hardware,
        op_kind=sample.op_kind,
        dimensions=_format_dimensions(sample.dimensions),
        measured_latency_ms=sample.measured_latency_ms,
        predicted_latency_ms=predicted_latency_ms,
        latency_ratio=latency_ratio,
        latency_abs_pct_error=abs(latency_ratio - 1.0) * 100.0,
        measured_energy_j=sample.measured_energy_j,
        predicted_energy_j=predicted_energy_j,
        energy_ratio=energy_ratio,
        energy_abs_pct_error=abs(energy_ratio - 1.0) * 100.0,
    )


def _hardware_info_for_file(filename: str) -> tuple[str, str] | None:
    lower_name = filename.lower()
    for prefix, info in _HARDWARE_BY_PREFIX.items():
        if lower_name.startswith(prefix):
            return info
    return None


def _supported_op_kind(filename: str) -> str | None:
    lower_name = filename.lower()
    if "bf16" not in lower_name:
        return None
    if "gemm" in lower_name:
        return OpKind.BATCHED_GEMM.value
    if "layernorm" in lower_name:
        return OpKind.LAYERNORM.value
    if "softmax" in lower_name:
        return OpKind.SOFTMAX.value
    return None


def _matches_config_frequency(filename: str) -> bool:
    lower_name = filename.lower()
    return "freq" not in lower_name or "freq900" in lower_name


def _is_bf16_row(row: Mapping[str, str]) -> bool:
    prec = _lower_or_none(row.get("prec"))
    if prec is not None:
        return prec == DType.BF16.value
    prec_m = _lower_or_none(row.get("precM"))
    prec_a = _lower_or_none(row.get("precA"))
    if prec_m is not None or prec_a is not None:
        return prec_m == DType.BF16.value and prec_a == DType.BF16.value
    return False


def _lower_or_none(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value.strip().lower()


def _float_field(row: Mapping[str, str], field: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"missing {field}")
    return float(value.replace(",", ""))


def _int_field(row: Mapping[str, str], field: str) -> int:
    value = _float_field(row, field)
    if not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _op_name(csv_path: Path, row_index: int) -> str:
    return f"{csv_path.stem}_row{row_index}"


def _format_dimensions(dimensions: Mapping[str, Any]) -> str:
    return ",".join(f"{key}={value}" for key, value in dimensions.items())


def _geomean(values: Iterable[float]) -> float:
    positive_values = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive_values:
        return float("nan")
    return math.exp(sum(math.log(value) for value in positive_values) / len(positive_values))


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return float("nan")
    return sum(values_list) / len(values_list)


def _normalized_bar_plot_svg(aggregates: tuple[AccuracyAggregate, ...]) -> str:
    if not aggregates:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220" '
            'viewBox="0 0 640 220">'
            '<rect width="640" height="220" fill="white"/>'
            '<text x="24" y="48" font-family="Arial, sans-serif" font-size="18">'
            "No supported bf16 rows processed"
            "</text></svg>\n"
        )

    left = 72
    right = 24
    panel_height = 170
    panel_gap = 74
    top_time = 52
    top_energy = top_time + panel_height + panel_gap
    bottom = 90
    group_width = 112
    width = max(760, left + right + group_width * len(aggregates))
    height = top_energy + panel_height + bottom
    plot_width = width - left - right

    def x_for_group(index: int) -> float:
        return left + (index + 0.5) * plot_width / len(aggregates)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{left}" y="24" font-family="Arial, sans-serif" '
            'font-size="18" font-weight="700">Normalized prediction accuracy</text>'
        ),
    ]

    _append_ratio_panel(
        parts=parts,
        title="time predicted / measured",
        aggregates=aggregates,
        values=[aggregate.latency_geomean_ratio for aggregate in aggregates],
        color="#2563eb",
        left=left,
        right=right,
        width=width,
        top=top_time,
        panel_height=panel_height,
        x_for_group=x_for_group,
    )
    _append_ratio_panel(
        parts=parts,
        title="energy predicted / measured",
        aggregates=aggregates,
        values=[aggregate.energy_geomean_ratio for aggregate in aggregates],
        color="#f97316",
        left=left,
        right=right,
        width=width,
        top=top_energy,
        panel_height=panel_height,
        x_for_group=x_for_group,
    )

    for index, aggregate in enumerate(aggregates):
        center = x_for_group(index)
        label = f"{aggregate.hardware}/{aggregate.op_kind}"
        parts.append(
            f'<text x="{center:.2f}" y="{top_energy + panel_height + 20}" text-anchor="end" '
            'font-family="Arial, sans-serif" font-size="11" '
            f'transform="rotate(-35 {center:.2f} {top_energy + panel_height + 20})">'
            f"{escape(label)}</text>"
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


def _append_ratio_panel(
    *,
    parts: list[str],
    title: str,
    aggregates: tuple[AccuracyAggregate, ...],
    values: list[float],
    color: str,
    left: int,
    right: int,
    width: int,
    top: int,
    panel_height: int,
    x_for_group: Any,
) -> None:
    y_max = _nice_axis_limit(max(1.0, *values) * 1.15)

    def y_for_ratio(value: float) -> float:
        return top + panel_height - (value / y_max) * panel_height

    parts.append(
        f'<text x="{left}" y="{top - 14}" font-family="Arial, sans-serif" '
        f'font-size="13" font-weight="700">{title}</text>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + panel_height}" x2="{width - right}" '
        f'y2="{top + panel_height}" stroke="#333"/>'
    )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" stroke="#333"/>')

    tick_count = 4
    for tick in range(tick_count + 1):
        value = y_max * tick / tick_count
        y = y_for_ratio(value)
        parts.append(
            f'<line x1="{left - 4}" y1="{y:.2f}" x2="{width - right}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11">{value:.2g}</text>'
        )

    perfect_y = y_for_ratio(1.0)
    parts.append(
        f'<line x1="{left}" y1="{perfect_y:.2f}" x2="{width - right}" '
        f'y2="{perfect_y:.2f}" stroke="#111827" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<text x="{width - right - 4}" y="{perfect_y - 6:.2f}" text-anchor="end" '
        'font-family="Arial, sans-serif" font-size="11">1.0 perfect</text>'
    )

    bar_width = 32
    for index, (aggregate, value) in enumerate(zip(aggregates, values, strict=True)):
        center = x_for_group(index)
        x = center - bar_width / 2
        y = y_for_ratio(value)
        height_px = top + panel_height - y
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" '
            f'height="{height_px:.2f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{center:.2f}" y="{max(top + 12, y - 5):.2f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="10">{value:.2g}</text>'
        )


def _nice_axis_limit(value: float) -> float:
    if value <= 1.5:
        return 1.5
    if value <= 2.0:
        return 2.0
    if value <= 3.0:
        return 3.0
    return math.ceil(value)


_CSV_FIELDS = (
    "source_file",
    "row_index",
    "hardware",
    "op_kind",
    "dimensions",
    "measured_latency_ms",
    "predicted_latency_ms",
    "latency_ratio",
    "latency_abs_pct_error",
    "measured_energy_j",
    "predicted_energy_j",
    "energy_ratio",
    "energy_abs_pct_error",
)
