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
_GEMM_WORKLOAD_BATCHES = (16, 32, 64, 128)
_GEMM_WORKLOAD_MNKS = (256, 512, 1024, 2048)
_SOFTMAX_WORKLOAD_BATCHES = tuple(2**power for power in range(13, 20))
_SOFTMAX_WORKLOAD_DIMS = (512, 1024)


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
class ArtifactPerformanceDetailsRow:
    values: Mapping[str, Any]


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
    performance_details: tuple[ArtifactPerformanceDetailsRow, ...] = ()

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
    performance_details: list[ArtifactPerformanceDetailsRow] = []
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
                    accuracy_row, detail_row = _predict_sample(sample, hardware, model)
                    rows.append(accuracy_row)
                    performance_details.append(detail_row)
                    file_rows += 1
                except (KeyError, ValueError) as exc:
                    reason = str(exc).splitlines()[0]
                    skip_counts[f"invalid_row:{reason}"] += 1

    return ArtifactValidationReport(
        rows=tuple(rows),
        skip_counts=dict(skip_counts),
        performance_details=tuple(performance_details),
    )


def write_csv_report(report: ArtifactValidationReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({field: getattr(row, field) for field in _CSV_FIELDS})


def write_performance_details_csv_report(
    report: ArtifactValidationReport, path: str | Path
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PERFORMANCE_DETAIL_CSV_FIELDS)
        writer.writeheader()
        for row in report.performance_details:
            writer.writerow(
                {
                    field: _csv_value(row.values.get(field))
                    for field in _PERFORMANCE_DETAIL_CSV_FIELDS
                }
            )


def read_csv_report(path: str | Path) -> ArtifactValidationReport:
    rows: list[ArtifactAccuracyRow] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ArtifactAccuracyRow(
                    source_file=str(row["source_file"]),
                    row_index=int(row["row_index"]),
                    hardware=str(row["hardware"]),
                    op_kind=str(row["op_kind"]),
                    dimensions=str(row["dimensions"]),
                    measured_latency_ms=float(row["measured_latency_ms"]),
                    predicted_latency_ms=float(row["predicted_latency_ms"]),
                    latency_ratio=float(row["latency_ratio"]),
                    latency_abs_pct_error=float(row["latency_abs_pct_error"]),
                    measured_energy_j=float(row["measured_energy_j"]),
                    predicted_energy_j=float(row["predicted_energy_j"]),
                    energy_ratio=float(row["energy_ratio"]),
                    energy_abs_pct_error=float(row["energy_abs_pct_error"]),
                )
            )
    return ArtifactValidationReport(rows=tuple(rows), skip_counts={})


def write_normalized_scatter_plot(report: ArtifactValidationReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_normalized_scatter_plot_svg(report.rows), encoding="utf-8")


def write_gemm_workload_bar_plot(
    report: ArtifactValidationReport,
    path: str | Path,
    *,
    y_max: float | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _gemm_workload_bar_plot_svg(report.rows, y_max=y_max), encoding="utf-8"
    )


def write_softmax_workload_bar_plot(
    report: ArtifactValidationReport,
    path: str | Path,
    *,
    y_max: float | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _softmax_workload_bar_plot_svg(report.rows, y_max=y_max), encoding="utf-8"
    )


def write_validation_plots(
    report: ArtifactValidationReport,
    path: str | Path,
    *,
    workload_y_max: float | None = None,
) -> tuple[Path, ...]:
    scatter_path = Path(path)
    gemm_path = _derived_plot_path(scatter_path, "gemm_workloads")
    softmax_path = _derived_plot_path(scatter_path, "softmax_workloads")
    energy_breakdown_path = _derived_plot_path(
        scatter_path, "energy_breakdown_workloads"
    )
    write_normalized_scatter_plot(report, scatter_path)
    write_gemm_workload_bar_plot(report, gemm_path, y_max=workload_y_max)
    write_softmax_workload_bar_plot(report, softmax_path, y_max=workload_y_max)
    write_energy_breakdown_workload_plot(report, energy_breakdown_path)
    return (scatter_path, gemm_path, softmax_path, energy_breakdown_path)


def write_energy_breakdown_workload_plot(
    report: ArtifactValidationReport, path: str | Path
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_energy_breakdown_workload_plot_svg(report), encoding="utf-8")


def write_normalized_bar_plot(report: ArtifactValidationReport, path: str | Path) -> None:
    write_normalized_scatter_plot(report, path)


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
) -> tuple[ArtifactAccuracyRow, ArtifactPerformanceDetailsRow]:
    profile = model.predict(sample.op, hardware)
    predicted_latency_ms = profile.latency_s * 1000.0
    predicted_energy_j = profile.energy_j
    latency_ratio = predicted_latency_ms / sample.measured_latency_ms
    energy_ratio = predicted_energy_j / sample.measured_energy_j
    accuracy_row = ArtifactAccuracyRow(
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
    return accuracy_row, _performance_details_row(sample, hardware, profile, accuracy_row)


def _performance_details_row(
    sample: ArtifactSample,
    hardware: HardwareSpec,
    profile: Any,
    accuracy_row: ArtifactAccuracyRow,
) -> ArtifactPerformanceDetailsRow:
    diagnostics = profile.diagnostics
    memory_access = profile.memory_access
    dtype = _diagnostic_value(diagnostics, "problem", "input_dtype")
    hardware_dtype = (
        dtype
        if isinstance(dtype, DType)
        else DType(str(dtype))
        if dtype is not None
        else _first_tensor_dtype(sample.op)
    )
    peak_compute = _peak_compute_throughput(profile.engine.value, hardware_dtype, hardware)
    compute_config_util = _compute_config_utilization(profile.engine.value, hardware)
    predicted_latency_s = profile.latency_s
    hbm_total = _total_bytes(memory_access.hbm_read_bytes, memory_access.hbm_write_bytes)
    l2_total = _total_bytes(memory_access.l2_read_bytes, memory_access.l2_write_bytes)
    sram_total = _total_bytes(memory_access.sram_read_bytes, memory_access.sram_write_bytes)
    register_total = _total_bytes(
        memory_access.register_read_bytes,
        memory_access.register_write_bytes,
    )
    smem_read = _diagnostic_value(diagnostics, "transaction_bytes", "smem_read")
    smem_write = _diagnostic_value(diagnostics, "transaction_bytes", "smem_write")
    smem_total = _total_bytes(smem_read, smem_write)

    values: dict[str, Any] = {
        "source_file": accuracy_row.source_file,
        "row_index": accuracy_row.row_index,
        "hardware": accuracy_row.hardware,
        "op_kind": accuracy_row.op_kind,
        "dimensions": accuracy_row.dimensions,
        "implementation": profile.implementation,
        "engine": profile.engine.value,
        "input_dtype": hardware_dtype.value,
        "output_dtype": _diagnostic_value(diagnostics, "problem", "output_dtype"),
        "problem_batch": _diagnostic_value(diagnostics, "problem", "batch"),
        "problem_m": _diagnostic_value(diagnostics, "problem", "m"),
        "problem_n": _diagnostic_value(diagnostics, "problem", "n"),
        "problem_k": _diagnostic_value(diagnostics, "problem", "k"),
        "beta_zero": _diagnostic_value(diagnostics, "problem", "beta_zero"),
        "epilogue_reads_c": _diagnostic_value(
            diagnostics, "problem", "epilogue_reads_c"
        ),
        "transpose_a": _diagnostic_value(diagnostics, "problem", "transpose_a"),
        "transpose_b": _diagnostic_value(diagnostics, "problem", "transpose_b"),
        "measured_latency_ms": accuracy_row.measured_latency_ms,
        "predicted_latency_ms": accuracy_row.predicted_latency_ms,
        "latency_ratio": accuracy_row.latency_ratio,
        "measured_energy_j": accuracy_row.measured_energy_j,
        "predicted_energy_j": accuracy_row.predicted_energy_j,
        "energy_ratio": accuracy_row.energy_ratio,
        "compute_energy_j": profile.energy_breakdown.compute_j,
        "dram_energy_j": profile.energy_breakdown.hbm_j,
        "l2_energy_j": profile.energy_breakdown.l2_j,
        "smem_energy_j": profile.energy_breakdown.sram_j,
        "static_power_w": hardware.static_power_w,
        "static_energy_j": profile.energy_breakdown.static_j,
        "flops": profile.flops,
        "predicted_elapsed_cycles": diagnostics.get("predicted_elapsed_cycles"),
        "clock_hz": diagnostics.get("clock_hz"),
        "predicted_flop_per_cycle": diagnostics.get("predicted_flop_per_cycle"),
        "predicted_tflops_per_s": diagnostics.get("predicted_tflops_per_s"),
        "achieved_flops_per_s": (profile.flops / predicted_latency_s)
        if predicted_latency_s
        else None,
        "compute_latency_s": diagnostics.get("compute_latency_s"),
        "memory_latency_s": diagnostics.get("memory_latency_s"),
        "roofline_latency_s": diagnostics.get("roofline_latency_s"),
        "kernel_launch_overhead_s": diagnostics.get("kernel_launch_overhead_s"),
        "effective_flops_per_s": diagnostics.get("effective_flops_per_s"),
        "effective_hbm_bandwidth_bytes_per_s": diagnostics.get(
            "effective_hbm_bandwidth_bytes_per_s"
        ),
        "arithmetic_intensity_flops_per_byte": diagnostics.get(
            "arithmetic_intensity_flops_per_byte"
        ),
        "peak_compute_throughput_flops_per_s": peak_compute,
        "compute_config_utilization_factor": compute_config_util,
        "effective_configured_compute_throughput_flops_per_s": (
            peak_compute * compute_config_util if peak_compute is not None else None
        ),
        "compute_issue_utilization_factor": _diagnostic_value(
            diagnostics, "utilization", "compute_issue"
        ),
        "compute_latency_utilization_factor": _diagnostic_value(
            diagnostics, "utilization", "compute_latency_adjusted"
        ),
        "compute_active_utilization_factor": _diagnostic_value(
            diagnostics, "utilization", "compute_active"
        ),
        "mma_ilp_efficiency": _diagnostic_value(
            diagnostics, "utilization", "mma_ilp_efficiency"
        ),
        "cta_tile_m": _diagnostic_value(diagnostics, "kernel", "cta_tile", "m"),
        "cta_tile_n": _diagnostic_value(diagnostics, "kernel", "cta_tile", "n"),
        "cta_tile_k": _diagnostic_value(diagnostics, "kernel", "cta_tile", "k"),
        "warp_tile_m": _diagnostic_value(diagnostics, "kernel", "warp_tile", "m"),
        "warp_tile_n": _diagnostic_value(diagnostics, "kernel", "warp_tile", "n"),
        "warp_tile_k": _diagnostic_value(diagnostics, "kernel", "warp_tile", "k"),
        "mma_shape_m": _diagnostic_value(diagnostics, "kernel", "mma_shape", "m"),
        "mma_shape_n": _diagnostic_value(diagnostics, "kernel", "mma_shape", "n"),
        "mma_shape_k": _diagnostic_value(diagnostics, "kernel", "mma_shape", "k"),
        "pipeline_stages": _diagnostic_value(diagnostics, "kernel", "pipeline_stages"),
        "warps_per_cta": _diagnostic_value(diagnostics, "kernel", "warps_per_cta"),
        "threads_per_cta": _diagnostic_value(diagnostics, "kernel", "threads_per_cta"),
        "registers_per_thread": _diagnostic_value(
            diagnostics, "kernel", "registers_per_thread"
        ),
        "shared_memory_bytes_per_cta": _diagnostic_value(
            diagnostics, "kernel", "shared_memory_bytes_per_cta"
        ),
        "tile_strategy": _tile_strategy(diagnostics),
        "cta_grid_m": _diagnostic_value(diagnostics, "cta_grid", "m"),
        "cta_grid_n": _diagnostic_value(diagnostics, "cta_grid", "n"),
        "cta_grid_k_stages": _diagnostic_value(diagnostics, "cta_grid", "k_stages"),
        "cta_count": diagnostics.get("cta_count"),
        "cta_waves": diagnostics.get("cta_waves"),
        "resident_ctas_per_sm": diagnostics.get("resident_ctas_per_sm"),
        "ctas_per_wave": diagnostics.get("ctas_per_wave"),
        "tail_efficiency": diagnostics.get("tail_efficiency"),
        "tile_efficiency": diagnostics.get("tile_efficiency"),
        "occupancy_limiting_factors": diagnostics.get("occupancy_limiting_factors"),
        "hbm_read_bytes": memory_access.hbm_read_bytes,
        "hbm_write_bytes": memory_access.hbm_write_bytes,
        "hbm_total_bytes": hbm_total,
        "l2_read_bytes": memory_access.l2_read_bytes,
        "l2_write_bytes": memory_access.l2_write_bytes,
        "l2_total_bytes": l2_total,
        "sram_read_bytes": memory_access.sram_read_bytes,
        "sram_write_bytes": memory_access.sram_write_bytes,
        "sram_total_bytes": sram_total,
        "register_read_bytes": memory_access.register_read_bytes,
        "register_write_bytes": memory_access.register_write_bytes,
        "register_total_bytes": register_total,
        "logical_a_bytes": _diagnostic_value(diagnostics, "logical_bytes", "a"),
        "logical_b_bytes": _diagnostic_value(diagnostics, "logical_bytes", "b"),
        "logical_c_read_bytes": _diagnostic_value(
            diagnostics, "logical_bytes", "c_read"
        ),
        "logical_d_store_bytes": _diagnostic_value(
            diagnostics, "logical_bytes", "d_store"
        ),
        "l2_requested_transaction_bytes": _diagnostic_value(
            diagnostics, "transaction_bytes", "l2_requested"
        ),
        "dram_unique_transaction_bytes": _diagnostic_value(
            diagnostics, "transaction_bytes", "dram_unique"
        ),
        "smem_read_transaction_bytes": smem_read,
        "smem_write_transaction_bytes": smem_write,
        "smem_total_transaction_bytes": smem_total,
        "sector_size_bytes": _diagnostic_value(
            diagnostics, "transaction_bytes", "sector_size"
        ),
        "line_size_bytes": _diagnostic_value(
            diagnostics, "transaction_bytes", "line_size"
        ),
        "hbm_peak_bandwidth_bytes_per_s": _peak_memory_bandwidth("hbm", hardware),
        "hbm_config_utilization_factor": _memory_config_utilization("hbm", hardware),
        "hbm_effective_configured_bandwidth_bytes_per_s": _effective_memory_bandwidth(
            "hbm", hardware
        ),
        "hbm_active_utilization_factor": _diagnostic_value(
            diagnostics, "utilization", "dram"
        ),
        "hbm_average_bandwidth_bytes_per_s": _average_bandwidth(
            hbm_total, predicted_latency_s
        ),
        "l2_peak_bandwidth_bytes_per_s": _peak_memory_bandwidth("l2", hardware),
        "l2_config_utilization_factor": _memory_config_utilization("l2", hardware),
        "l2_effective_configured_bandwidth_bytes_per_s": _effective_memory_bandwidth(
            "l2", hardware
        ),
        "l2_active_utilization_factor": _diagnostic_value(
            diagnostics, "utilization", "l2"
        ),
        "l2_average_bandwidth_bytes_per_s": _average_bandwidth(l2_total, predicted_latency_s),
        "smem_peak_bandwidth_bytes_per_s": _smem_peak_bandwidth(hardware),
        "smem_config_utilization_factor": _memory_config_utilization("sram", hardware),
        "smem_effective_configured_bandwidth_bytes_per_s": _smem_effective_bandwidth(
            hardware
        ),
        "smem_active_utilization_factor": _diagnostic_value(
            diagnostics, "utilization", "smem"
        ),
        "smem_average_bandwidth_bytes_per_s": _average_bandwidth(
            smem_total, predicted_latency_s
        ),
        "compute_active_cycles": _diagnostic_value(diagnostics, "active_cycles", "compute"),
        "smem_active_cycles": _diagnostic_value(diagnostics, "active_cycles", "smem"),
        "l2_active_cycles": _diagnostic_value(diagnostics, "active_cycles", "l2"),
        "dram_active_cycles": _diagnostic_value(diagnostics, "active_cycles", "dram"),
        "compute_stage_cycles": _diagnostic_value(diagnostics, "stage_cycles", "compute"),
        "mma_issue_cycles": _diagnostic_value(diagnostics, "stage_cycles", "mma_issue"),
        "mma_dependency_penalty_cycles": _diagnostic_value(
            diagnostics, "stage_cycles", "mma_dependency_penalty"
        ),
        "smem_stage_cycles": _diagnostic_value(diagnostics, "stage_cycles", "smem"),
        "global_load_issue_cycles": _diagnostic_value(
            diagnostics, "stage_cycles", "global_load_issue"
        ),
        "l2_service_cycles": _diagnostic_value(diagnostics, "stage_cycles", "l2_service"),
        "dram_service_cycles": _diagnostic_value(
            diagnostics, "stage_cycles", "dram_service"
        ),
        "exposed_l2_cycles": _diagnostic_value(diagnostics, "stage_cycles", "exposed_l2"),
        "exposed_dram_cycles": _diagnostic_value(
            diagnostics, "stage_cycles", "exposed_dram"
        ),
        "stage_cycles": _diagnostic_value(diagnostics, "stage_cycles", "stage"),
        "prologue_cycles": _diagnostic_value(diagnostics, "stage_cycles", "prologue"),
        "epilogue_cycles": _diagnostic_value(diagnostics, "stage_cycles", "epilogue"),
        "cta_cycles": _diagnostic_value(diagnostics, "stage_cycles", "cta"),
        "compute_smem_overlap_factor": _diagnostic_value(
            diagnostics, "overlap", "compute_smem"
        ),
        "compute_l2_overlap_factor": _diagnostic_value(
            diagnostics, "overlap", "compute_l2"
        ),
        "compute_dram_overlap_factor": _diagnostic_value(
            diagnostics, "overlap", "compute_dram"
        ),
        "smem_flop_per_byte": _diagnostic_value(
            diagnostics, "operational_intensity", "smem_flop_per_byte"
        ),
        "l2_flop_per_byte": _diagnostic_value(
            diagnostics, "operational_intensity", "l2_flop_per_byte"
        ),
        "dram_flop_per_byte": _diagnostic_value(
            diagnostics, "operational_intensity", "dram_flop_per_byte"
        ),
        "merged_flop_per_byte": _diagnostic_value(
            diagnostics, "operational_intensity", "merged_flop_per_byte"
        ),
        "roofline_bound_compute_flop_per_cycle": _diagnostic_value(
            diagnostics, "roofline_bounds_flop_per_cycle", "compute"
        ),
        "roofline_bound_smem_flop_per_cycle": _diagnostic_value(
            diagnostics, "roofline_bounds_flop_per_cycle", "smem"
        ),
        "roofline_bound_l2_flop_per_cycle": _diagnostic_value(
            diagnostics, "roofline_bounds_flop_per_cycle", "l2"
        ),
        "roofline_bound_dram_flop_per_cycle": _diagnostic_value(
            diagnostics, "roofline_bounds_flop_per_cycle", "dram"
        ),
        "hbm_memory_latency_s": _diagnostic_value(
            diagnostics, "memory_level_latencies_s", "hbm"
        ),
        "l2_memory_latency_s": _diagnostic_value(
            diagnostics, "memory_level_latencies_s", "l2"
        ),
        "sram_memory_latency_s": _diagnostic_value(
            diagnostics, "memory_level_latencies_s", "sram"
        ),
        "register_memory_latency_s": _diagnostic_value(
            diagnostics, "memory_level_latencies_s", "register"
        ),
        "primary_bottleneck": diagnostics.get("primary_bottleneck"),
        "secondary_bottlenecks": diagnostics.get("secondary_bottlenecks"),
        "warnings": diagnostics.get("warnings"),
        "assumptions": diagnostics.get("assumptions"),
    }
    return ArtifactPerformanceDetailsRow(values=values)


def _diagnostic_value(data: Mapping[str, Any], *path: str) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _tile_strategy(diagnostics: Mapping[str, Any]) -> str | None:
    cta_m = _diagnostic_value(diagnostics, "kernel", "cta_tile", "m")
    cta_n = _diagnostic_value(diagnostics, "kernel", "cta_tile", "n")
    cta_k = _diagnostic_value(diagnostics, "kernel", "cta_tile", "k")
    warp_m = _diagnostic_value(diagnostics, "kernel", "warp_tile", "m")
    warp_n = _diagnostic_value(diagnostics, "kernel", "warp_tile", "n")
    warp_k = _diagnostic_value(diagnostics, "kernel", "warp_tile", "k")
    mma_m = _diagnostic_value(diagnostics, "kernel", "mma_shape", "m")
    mma_n = _diagnostic_value(diagnostics, "kernel", "mma_shape", "n")
    mma_k = _diagnostic_value(diagnostics, "kernel", "mma_shape", "k")
    if cta_m is None or warp_m is None or mma_m is None:
        return None
    stages = _diagnostic_value(diagnostics, "kernel", "pipeline_stages")
    warps = _diagnostic_value(diagnostics, "kernel", "warps_per_cta")
    return (
        f"cta={cta_m}x{cta_n}x{cta_k};"
        f"warp={warp_m}x{warp_n}x{warp_k};"
        f"mma={mma_m}x{mma_n}x{mma_k};"
        f"stages={stages};warps={warps}"
    )


def _first_tensor_dtype(op: LocalOp) -> DType:
    if not op.tensors:
        return DType.BF16
    return op.tensors[0].dtype


def _peak_compute_throughput(
    engine: str, dtype: DType, hardware: HardwareSpec
) -> float | None:
    if engine == "tensor":
        return hardware.compute.tensor_flops_per_s.get(dtype)
    if engine == "vector":
        return hardware.compute.vector_flops_per_s.get(dtype)
    return None


def _compute_config_utilization(engine: str, hardware: HardwareSpec) -> float | None:
    if engine == "tensor":
        return hardware.utilization.tensor
    if engine == "vector":
        return hardware.utilization.vector
    return None


def _peak_memory_bandwidth(name: str, hardware: HardwareSpec) -> float | None:
    level = hardware.memory_levels.get(name)
    return None if level is None else level.bandwidth_bytes_per_s


def _memory_config_utilization(name: str, hardware: HardwareSpec) -> float | None:
    if name not in hardware.memory_levels:
        return None
    return hardware.utilization.memory.get(name, 1.0)


def _effective_memory_bandwidth(name: str, hardware: HardwareSpec) -> float | None:
    peak = _peak_memory_bandwidth(name, hardware)
    utilization = _memory_config_utilization(name, hardware)
    if peak is None or utilization is None:
        return None
    return peak * utilization


def _smem_peak_bandwidth(hardware: HardwareSpec) -> float | None:
    sram = _peak_memory_bandwidth("sram", hardware)
    if sram is not None:
        return sram
    hbm = _peak_memory_bandwidth("hbm", hardware)
    return None if hbm is None else hbm * 8.0


def _smem_effective_bandwidth(hardware: HardwareSpec) -> float | None:
    sram = _effective_memory_bandwidth("sram", hardware)
    if sram is not None:
        return sram
    hbm = _effective_memory_bandwidth("hbm", hardware)
    return None if hbm is None else hbm * 8.0


def _total_bytes(read_bytes: Any, write_bytes: Any) -> int | None:
    if read_bytes is None and write_bytes is None:
        return None
    return int(read_bytes or 0) + int(write_bytes or 0)


def _average_bandwidth(total_bytes: Any, latency_s: float) -> float | None:
    if total_bytes is None or latency_s <= 0.0:
        return None
    return float(total_bytes) / latency_s


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (tuple, list)):
        return "|".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return "|".join(f"{key}={item}" for key, item in value.items())
    return value


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


@dataclass(frozen=True)
class _ScatterPoint:
    hardware: str
    op_kind: str
    working_set_bytes: int
    latency_ratio: float
    energy_ratio: float


@dataclass(frozen=True)
class _WorkloadPoint:
    workload: str
    hardware: str
    latency_ratio: float
    energy_ratio: float
    count: int


@dataclass(frozen=True)
class _WorkloadSpec:
    key: str
    label: str
    working_set_bytes: int


@dataclass(frozen=True)
class _EnergyBreakdownWorkloadPoint:
    workload: str
    hardware: str
    components: Mapping[str, float]
    total_ratio: float
    count: int


def _normalized_scatter_plot_svg(rows: tuple[ArtifactAccuracyRow, ...]) -> str:
    points = tuple(_scatter_point(row) for row in rows)
    if not points:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220" '
            'viewBox="0 0 640 220">'
            '<rect width="640" height="220" fill="white"/>'
            '<text x="24" y="48" font-family="Arial, sans-serif" font-size="18">'
            "No supported bf16 rows processed"
            "</text></svg>\n"
        )

    op_order = tuple(
        op_kind
        for op_kind in (OpKind.BATCHED_GEMM.value, OpKind.LAYERNORM.value, OpKind.SOFTMAX.value)
        if any(point.op_kind == op_kind for point in points)
    )
    op_order += tuple(
        op_kind
        for op_kind in sorted({point.op_kind for point in points})
        if op_kind not in set(op_order)
    )
    hardware_order = tuple(sorted({point.hardware for point in points}))
    colors = _hardware_colors(hardware_order)

    left = 84
    right = 28
    top = 76
    panel_width = 310
    panel_height = 210
    column_gap = 46
    row_gap = 82
    legend_height = 34
    bottom = 62
    width = left + right + len(op_order) * panel_width + (len(op_order) - 1) * column_gap
    height = top + legend_height + 2 * panel_height + row_gap + bottom
    row_tops = (top + legend_height, top + legend_height + panel_height + row_gap)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{left}" y="24" font-family="Arial, sans-serif" '
            'font-size="18" font-weight="700">Normalized prediction scatter by working set</text>'
        ),
        (
            f'<text x="{left}" y="46" font-family="Arial, sans-serif" '
            'font-size="12" fill="#4b5563">x: bf16 working-set bytes, log scale; '
            "y: predicted / measured ratio, log scale</text>"
        ),
    ]

    legend_x = left
    for hardware in hardware_order:
        parts.append(
            f'<circle cx="{legend_x + 5}" cy="{top + 8}" r="5" fill="{colors[hardware]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 16}" y="{top + 12}" font-family="Arial, sans-serif" '
            f'font-size="12">{escape(hardware)}</text>'
        )
        legend_x += 132

    metric_ranges = {
        "latency": _log_range([point.latency_ratio for point in points] + [1.0, 10.0]),
        "energy": _log_range([point.energy_ratio for point in points] + [1.0, 10.0]),
    }
    metrics = (
        ("latency", "latency predicted / measured", lambda point: point.latency_ratio),
        ("energy", "energy predicted / measured", lambda point: point.energy_ratio),
    )

    for row_index, (metric_key, metric_title, value_getter) in enumerate(metrics):
        row_top = row_tops[row_index]
        parts.append(
            f'<text x="18" y="{row_top + panel_height / 2:.2f}" '
            'font-family="Arial, sans-serif" font-size="13" font-weight="700" '
            f'transform="rotate(-90 18 {row_top + panel_height / 2:.2f})">'
            f"{escape(metric_title)}</text>"
        )
        for col_index, op_kind in enumerate(op_order):
            panel_points = tuple(point for point in points if point.op_kind == op_kind)
            x_range = _log_range([point.working_set_bytes for point in panel_points])
            x = left + col_index * (panel_width + column_gap)
            _append_scatter_panel(
                parts=parts,
                title=op_kind,
                points=panel_points,
                colors=colors,
                metric_value=value_getter,
                x=x,
                y=row_top,
                width=panel_width,
                height=panel_height,
                x_range=x_range,
                y_range=metric_ranges[metric_key],
                show_y_axis=col_index == 0,
                show_x_axis=row_index == len(metrics) - 1,
            )
    parts.append("</svg>\n")
    return "\n".join(parts)


def _gemm_workload_bar_plot_svg(
    rows: tuple[ArtifactAccuracyRow, ...], *, y_max: float | None = None
) -> str:
    workloads = tuple(
        sorted(
            (
                _WorkloadSpec(
                    key=f"B{batch}_MNK{mnk}",
                    label=f"B{batch}/MNK{mnk}",
                    working_set_bytes=_gemm_square_working_set_bytes(batch, mnk),
                )
                for batch in _GEMM_WORKLOAD_BATCHES
                for mnk in _GEMM_WORKLOAD_MNKS
            ),
            key=lambda workload: (workload.working_set_bytes, workload.label),
        )
    )

    def workload_for_row(row: ArtifactAccuracyRow) -> str | None:
        if row.op_kind != OpKind.BATCHED_GEMM.value:
            return None
        dimensions = _parse_dimensions(row.dimensions)
        batch = int(dimensions["batch"])
        m = int(dimensions["M"])
        n = int(dimensions["N"])
        k = int(dimensions["K"])
        if (
            batch in _GEMM_WORKLOAD_BATCHES
            and m == n
            and n == k
            and m in _GEMM_WORKLOAD_MNKS
        ):
            return f"B{batch}_MNK{m}"
        return None

    points = _workload_points(rows, workload_for_row)
    return _normalized_workload_bar_plot_svg(
        title="Square GEMM workload ratios",
        subtitle="Filtered to B in [16, 32, 64, 128] and M=N=K in [256, 512, 1024, 2048]",
        workloads=workloads,
        points=points,
        y_max=y_max,
    )


def _softmax_workload_bar_plot_svg(
    rows: tuple[ArtifactAccuracyRow, ...], *, y_max: float | None = None
) -> str:
    workloads = tuple(
        sorted(
            (
                _WorkloadSpec(
                    key=f"B{batch}_D{dim}",
                    label=f"B2^{int(math.log2(batch))}/D{dim}",
                    working_set_bytes=_softmax_working_set_bytes(batch, dim),
                )
                for batch in _SOFTMAX_WORKLOAD_BATCHES
                for dim in _SOFTMAX_WORKLOAD_DIMS
            ),
            key=lambda workload: (workload.working_set_bytes, workload.label),
        )
    )

    def workload_for_row(row: ArtifactAccuracyRow) -> str | None:
        if row.op_kind != OpKind.SOFTMAX.value:
            return None
        dimensions = _parse_dimensions(row.dimensions)
        batch = int(dimensions["batch"])
        dim = int(dimensions["dim"])
        if batch in _SOFTMAX_WORKLOAD_BATCHES and dim in _SOFTMAX_WORKLOAD_DIMS:
            return f"B{batch}_D{dim}"
        return None

    points = _workload_points(rows, workload_for_row)
    return _normalized_workload_bar_plot_svg(
        title="Softmax workload ratios",
        subtitle="Filtered to B=2^[13, 19] and Dim in [512, 1024]",
        workloads=workloads,
        points=points,
        y_max=y_max,
    )


_ENERGY_COMPONENTS = (
    ("static", "static", "#7c3aed"),
    ("compute", "compute", "#dc2626"),
    ("dram", "DRAM", "#2563eb"),
    ("l2", "L2", "#0891b2"),
    ("smem", "SMEM", "#16a34a"),
)


def _energy_breakdown_workload_plot_svg(report: ArtifactValidationReport) -> str:
    if not report.performance_details:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220" '
            'viewBox="0 0 720 220">'
            '<rect width="720" height="220" fill="white"/>'
            '<text x="28" y="42" font-family="Arial, sans-serif" font-size="18" '
            'font-weight="700">Workload energy breakdown ratios</text>'
            '<text x="28" y="84" font-family="Arial, sans-serif" font-size="14">'
            "No performance details available"
            "</text></svg>\n"
        )

    gemm_workloads = tuple(
        sorted(
            (
                _WorkloadSpec(
                    key=f"B{batch}_MNK{mnk}",
                    label=f"B{batch}/MNK{mnk}",
                    working_set_bytes=_gemm_square_working_set_bytes(batch, mnk),
                )
                for batch in _GEMM_WORKLOAD_BATCHES
                for mnk in _GEMM_WORKLOAD_MNKS
            ),
            key=lambda workload: (workload.working_set_bytes, workload.label),
        )
    )
    softmax_workloads = tuple(
        sorted(
            (
                _WorkloadSpec(
                    key=f"B{batch}_D{dim}",
                    label=f"B2^{int(math.log2(batch))}/D{dim}",
                    working_set_bytes=_softmax_working_set_bytes(batch, dim),
                )
                for batch in _SOFTMAX_WORKLOAD_BATCHES
                for dim in _SOFTMAX_WORKLOAD_DIMS
            ),
            key=lambda workload: (workload.working_set_bytes, workload.label),
        )
    )

    def gemm_workload_for_detail(values: Mapping[str, Any]) -> str | None:
        if values.get("op_kind") != OpKind.BATCHED_GEMM.value:
            return None
        dimensions = _parse_dimensions(str(values["dimensions"]))
        batch = int(dimensions["batch"])
        m = int(dimensions["M"])
        n = int(dimensions["N"])
        k = int(dimensions["K"])
        if (
            batch in _GEMM_WORKLOAD_BATCHES
            and m == n
            and n == k
            and m in _GEMM_WORKLOAD_MNKS
        ):
            return f"B{batch}_MNK{m}"
        return None

    def softmax_workload_for_detail(values: Mapping[str, Any]) -> str | None:
        if values.get("op_kind") != OpKind.SOFTMAX.value:
            return None
        dimensions = _parse_dimensions(str(values["dimensions"]))
        batch = int(dimensions["batch"])
        dim = int(dimensions["dim"])
        if batch in _SOFTMAX_WORKLOAD_BATCHES and dim in _SOFTMAX_WORKLOAD_DIMS:
            return f"B{batch}_D{dim}"
        return None

    gemm_points = _energy_breakdown_workload_points(
        report.performance_details, gemm_workload_for_detail
    )
    softmax_points = _energy_breakdown_workload_points(
        report.performance_details, softmax_workload_for_detail
    )
    if not gemm_points and not softmax_points:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220" '
            'viewBox="0 0 720 220">'
            '<rect width="720" height="220" fill="white"/>'
            '<text x="28" y="42" font-family="Arial, sans-serif" font-size="18" '
            'font-weight="700">Workload energy breakdown ratios</text>'
            '<text x="28" y="84" font-family="Arial, sans-serif" font-size="14">'
            "No matching workloads in report"
            "</text></svg>\n"
        )

    hardware_order = tuple(
        sorted({point.hardware for point in (*gemm_points, *softmax_points)})
    )
    left = 72
    right = 28
    top = 104
    panel_height = 205
    panel_gap = 130
    bottom = 118
    group_width = max(76, 32 + len(hardware_order) * 18)
    max_workload_count = max(len(gemm_workloads), len(softmax_workloads))
    width = left + right + group_width * max_workload_count
    top_gemm = top
    top_softmax = top_gemm + panel_height + panel_gap
    height = top_softmax + panel_height + bottom
    y_max = _nice_axis_limit(
        max(1.0, *(point.total_ratio for point in (*gemm_points, *softmax_points))) * 1.15
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{left}" y="26" font-family="Arial, sans-serif" '
            'font-size="18" font-weight="700">Workload energy breakdown ratios</text>'
        ),
        (
            f'<text x="{left}" y="48" font-family="Arial, sans-serif" '
            'font-size="12" fill="#4b5563">Stacked components are component energy / '
            "measured energy; a10 and a100 bars are side by side per workload</text>"
        ),
    ]

    legend_x = left
    for _key, label, color in _ENERGY_COMPONENTS:
        parts.append(
            f'<rect x="{legend_x}" y="68" width="11" height="11" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 17}" y="78" font-family="Arial, sans-serif" '
            f'font-size="12">{escape(label)}</text>'
        )
        legend_x += 86
    hardware_order_label = " / ".join(hardware_order)
    parts.append(
        f'<text x="{legend_x + 10}" y="78" font-family="Arial, sans-serif" '
        f'font-size="12" fill="#4b5563">bar order: {escape(hardware_order_label)}</text>'
    )

    _append_energy_breakdown_panel(
        parts=parts,
        title="Square GEMM energy predicted / measured",
        workloads=gemm_workloads,
        points=gemm_points,
        hardware_order=hardware_order,
        left=left,
        top=top_gemm,
        width=width,
        right=right,
        panel_height=panel_height,
        group_width=group_width,
        y_max=y_max,
    )
    _append_energy_breakdown_panel(
        parts=parts,
        title="Softmax energy predicted / measured",
        workloads=softmax_workloads,
        points=softmax_points,
        hardware_order=hardware_order,
        left=left,
        top=top_softmax,
        width=width,
        right=right,
        panel_height=panel_height,
        group_width=group_width,
        y_max=y_max,
    )
    parts.append("</svg>\n")
    return "\n".join(parts)


def _energy_breakdown_workload_points(
    details: tuple[ArtifactPerformanceDetailsRow, ...],
    workload_for_detail: Any,
) -> tuple[_EnergyBreakdownWorkloadPoint, ...]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for detail in details:
        workload = workload_for_detail(detail.values)
        if workload is not None:
            groups[(workload, str(detail.values["hardware"]))].append(detail.values)

    points: list[_EnergyBreakdownWorkloadPoint] = []
    for (workload, hardware), grouped_details in sorted(groups.items()):
        component_values: dict[str, float] = {}
        for key, _label, _color in _ENERGY_COMPONENTS:
            field = f"{key}_energy_j"
            component_values[key] = _mean(
                _float_or_zero(values.get(field)) / _float_or_zero(values["measured_energy_j"])
                for values in grouped_details
                if _float_or_zero(values["measured_energy_j"]) > 0.0
            )
        total_ratio = sum(component_values.values())
        points.append(
            _EnergyBreakdownWorkloadPoint(
                workload=workload,
                hardware=hardware,
                components=component_values,
                total_ratio=total_ratio,
                count=len(grouped_details),
            )
        )
    return tuple(points)


def _append_energy_breakdown_panel(
    *,
    parts: list[str],
    title: str,
    workloads: tuple[_WorkloadSpec, ...],
    points: tuple[_EnergyBreakdownWorkloadPoint, ...],
    hardware_order: tuple[str, ...],
    left: int,
    top: int,
    width: int,
    right: int,
    panel_height: int,
    group_width: int,
    y_max: float,
) -> None:
    points_by_key = {(point.workload, point.hardware): point for point in points}

    def y_for_ratio(value: float) -> float:
        return top + panel_height - (value / y_max) * panel_height

    parts.append(
        f'<text x="{left}" y="{top - 14}" font-family="Arial, sans-serif" '
        f'font-size="13" font-weight="700">{escape(title)}</text>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + panel_height}" x2="{width - right}" '
        f'y2="{top + panel_height}" stroke="#333"/>'
    )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" stroke="#333"/>')

    for tick in range(5):
        value = y_max * tick / 4
        y = y_for_ratio(value)
        parts.append(
            f'<line x1="{left - 4}" y1="{y:.2f}" x2="{width - right}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="10">{value:.2g}</text>'
        )

    perfect_y = y_for_ratio(1.0)
    parts.append(
        f'<line x1="{left}" y1="{perfect_y:.2f}" x2="{width - right}" '
        f'y2="{perfect_y:.2f}" stroke="#111827" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<text x="{width - right - 4}" y="{perfect_y - 6:.2f}" text-anchor="end" '
        'font-family="Arial, sans-serif" font-size="10">1.0 measured energy</text>'
    )

    bar_width = 12
    bar_gap = 4
    total_bar_width = len(hardware_order) * bar_width + (len(hardware_order) - 1) * bar_gap
    label_y = top + panel_height + 22
    hardware_label_y = top + panel_height + 38
    for index, workload in enumerate(workloads):
        center = left + (index + 0.5) * group_width
        start_x = center - total_bar_width / 2
        for hardware_index, hardware in enumerate(hardware_order):
            point = points_by_key.get((workload.key, hardware))
            if point is None:
                continue
            x = start_x + hardware_index * (bar_width + bar_gap)
            stack_base = top + panel_height
            for key, _label, color in _ENERGY_COMPONENTS:
                value = max(0.0, point.components.get(key, 0.0))
                height = min(value, y_max) / y_max * panel_height
                y = stack_base - height
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" '
                    f'height="{height:.2f}" fill="{color}"/>'
                )
                stack_base = y
            parts.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{hardware_label_y}" '
                'text-anchor="middle" font-family="Arial, sans-serif" font-size="8">'
                f"{escape(_short_hardware_label(hardware))}</text>"
            )
        parts.append(
            f'<text x="{center:.2f}" y="{label_y}" text-anchor="end" '
            'font-family="Arial, sans-serif" font-size="10" '
            f'transform="rotate(-45 {center:.2f} {label_y})">'
            f"{escape(workload.label)}</text>"
        )


def _normalized_workload_bar_plot_svg(
    *,
    title: str,
    subtitle: str,
    workloads: tuple[_WorkloadSpec, ...],
    points: tuple[_WorkloadPoint, ...],
    y_max: float | None = None,
) -> str:
    if not points:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220" '
            'viewBox="0 0 720 220">'
            '<rect width="720" height="220" fill="white"/>'
            f'<text x="28" y="42" font-family="Arial, sans-serif" font-size="18" '
            f'font-weight="700">{escape(title)}</text>'
            f'<text x="28" y="68" font-family="Arial, sans-serif" font-size="12" '
            f'fill="#4b5563">{escape(subtitle)}</text>'
            '<text x="28" y="112" font-family="Arial, sans-serif" font-size="14">'
            "No matching workloads in report"
            "</text></svg>\n"
        )

    hardware_order = tuple(sorted({point.hardware for point in points}))
    colors = _hardware_colors(hardware_order)
    points_by_key = {
        (point.workload, point.hardware): point
        for point in points
    }
    left = 72
    right = 28
    top = 92
    panel_height = 185
    panel_gap = 76
    bottom = 118
    group_width = max(68, 24 + len(hardware_order) * 17)
    width = left + right + group_width * len(workloads)
    top_latency = top
    top_energy = top_latency + panel_height + panel_gap
    height = top_energy + panel_height + bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{left}" y="26" font-family="Arial, sans-serif" '
            f'font-size="18" font-weight="700">{escape(title)}</text>'
        ),
        (
            f'<text x="{left}" y="48" font-family="Arial, sans-serif" '
            f'font-size="12" fill="#4b5563">{escape(subtitle)}</text>'
        ),
    ]

    legend_x = left
    for hardware in hardware_order:
        parts.append(
            f'<rect x="{legend_x}" y="65" width="11" height="11" fill="{colors[hardware]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 17}" y="75" font-family="Arial, sans-serif" '
            f'font-size="12">{escape(hardware)}</text>'
        )
        legend_x += 132

    if y_max is None:
        y_max = _nice_axis_limit(
            max(
                1.0,
                *(point.latency_ratio for point in points),
                *(point.energy_ratio for point in points),
            )
            * 1.15
        )
    elif y_max <= 0.0 or not math.isfinite(y_max):
        raise ValueError("y_max must be positive and finite")

    def x_for_group(index: int) -> float:
        return left + (index + 0.5) * group_width

    _append_workload_bar_panel(
        parts=parts,
        title="latency predicted / measured",
        workloads=workloads,
        hardware_order=hardware_order,
        points_by_key=points_by_key,
        colors=colors,
        metric_value=lambda point: point.latency_ratio,
        left=left,
        top=top_latency,
        width=width,
        right=right,
        panel_height=panel_height,
        y_max=y_max,
        x_for_group=x_for_group,
    )
    _append_workload_bar_panel(
        parts=parts,
        title="energy predicted / measured",
        workloads=workloads,
        hardware_order=hardware_order,
        points_by_key=points_by_key,
        colors=colors,
        metric_value=lambda point: point.energy_ratio,
        left=left,
        top=top_energy,
        width=width,
        right=right,
        panel_height=panel_height,
        y_max=y_max,
        x_for_group=x_for_group,
    )

    label_y = top_energy + panel_height + 22
    for index, workload in enumerate(workloads):
        center = x_for_group(index)
        parts.append(
            f'<text x="{center:.2f}" y="{label_y}" text-anchor="end" '
            'font-family="Arial, sans-serif" font-size="10" '
            f'transform="rotate(-45 {center:.2f} {label_y})">'
            f"{escape(workload.label)}</text>"
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


def _append_workload_bar_panel(
    *,
    parts: list[str],
    title: str,
    workloads: tuple[_WorkloadSpec, ...],
    hardware_order: tuple[str, ...],
    points_by_key: Mapping[tuple[str, str], _WorkloadPoint],
    colors: Mapping[str, str],
    metric_value: Any,
    left: int,
    top: int,
    width: int,
    right: int,
    panel_height: int,
    y_max: float,
    x_for_group: Any,
) -> None:
    def y_for_ratio(value: float) -> float:
        return top + panel_height - (value / y_max) * panel_height

    parts.append(
        f'<text x="{left}" y="{top - 14}" font-family="Arial, sans-serif" '
        f'font-size="13" font-weight="700">{escape(title)}</text>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + panel_height}" x2="{width - right}" '
        f'y2="{top + panel_height}" stroke="#333"/>'
    )
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" stroke="#333"/>')

    for tick in range(5):
        value = y_max * tick / 4
        y = y_for_ratio(value)
        parts.append(
            f'<line x1="{left - 4}" y1="{y:.2f}" x2="{width - right}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="10">{value:.2g}</text>'
        )

    perfect_y = y_for_ratio(1.0)
    parts.append(
        f'<line x1="{left}" y1="{perfect_y:.2f}" x2="{width - right}" '
        f'y2="{perfect_y:.2f}" stroke="#111827" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<text x="{width - right - 4}" y="{perfect_y - 6:.2f}" text-anchor="end" '
        'font-family="Arial, sans-serif" font-size="10">1.0 perfect</text>'
    )

    bar_width = 12
    bar_gap = 3
    total_bar_width = len(hardware_order) * bar_width + (len(hardware_order) - 1) * bar_gap
    for index, workload in enumerate(workloads):
        center = x_for_group(index)
        start_x = center - total_bar_width / 2
        for hardware_index, hardware in enumerate(hardware_order):
            point = points_by_key.get((workload.key, hardware))
            if point is None:
                continue
            value = metric_value(point)
            y = y_for_ratio(min(value, y_max))
            x = start_x + hardware_index * (bar_width + bar_gap)
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" '
                f'height="{top + panel_height - y:.2f}" fill="{colors[hardware]}"/>'
            )


def _append_scatter_panel(
    *,
    parts: list[str],
    title: str,
    points: tuple[_ScatterPoint, ...],
    colors: Mapping[str, str],
    metric_value: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    show_y_axis: bool,
    show_x_axis: bool,
) -> None:
    x_min, x_max = x_range
    y_min, y_max = y_range
    parts.append(
        f'<text x="{x}" y="{y - 16}" font-family="Arial, sans-serif" '
        f'font-size="13" font-weight="700">{escape(title)}</text>'
    )
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        'fill="#ffffff" stroke="#d1d5db"/>'
    )

    for tick in _log_ticks(x_min, x_max):
        tick_x = _log_x(tick, x_min, x_max, x, width)
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{y}" x2="{tick_x:.2f}" '
            f'y2="{y + height}" stroke="#f3f4f6"/>'
        )
        if show_x_axis:
            parts.append(
                f'<text x="{tick_x:.2f}" y="{y + height + 18}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="10">{_format_bytes(tick)}</text>'
            )

    for tick in _log_ticks(y_min, y_max):
        tick_y = _log_y(tick, y_min, y_max, y, height)
        parts.append(
            f'<line x1="{x}" y1="{tick_y:.2f}" x2="{x + width}" '
            f'y2="{tick_y:.2f}" stroke="#e5e7eb"/>'
        )
        if show_y_axis:
            parts.append(
                f'<text x="{x - 8}" y="{tick_y + 4:.2f}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="10">{_format_ratio(tick)}</text>'
            )

    perfect_y = _log_y(1.0, y_min, y_max, y, height)
    parts.append(
        f'<line x1="{x}" y1="{perfect_y:.2f}" x2="{x + width}" '
        f'y2="{perfect_y:.2f}" stroke="#111827" stroke-dasharray="5 4"/>'
    )
    if show_y_axis:
        parts.append(
            f'<text x="{x + 5}" y="{perfect_y - 6:.2f}" '
            'font-family="Arial, sans-serif" font-size="10">1.0 perfect</text>'
        )

    for point in points:
        value = metric_value(point)
        if point.working_set_bytes <= 0 or value <= 0 or not math.isfinite(value):
            continue
        point_x = _log_x(point.working_set_bytes, x_min, x_max, x, width)
        point_y = _log_y(value, y_min, y_max, y, height)
        parts.append(
            f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="2.2" '
            f'fill="{colors[point.hardware]}" fill-opacity="0.48"/>'
        )


def _scatter_point(row: ArtifactAccuracyRow) -> _ScatterPoint:
    return _ScatterPoint(
        hardware=row.hardware,
        op_kind=row.op_kind,
        working_set_bytes=_working_set_bytes(row),
        latency_ratio=row.latency_ratio,
        energy_ratio=row.energy_ratio,
    )


def _workload_points(
    rows: tuple[ArtifactAccuracyRow, ...], workload_for_row: Any
) -> tuple[_WorkloadPoint, ...]:
    groups: dict[tuple[str, str], list[ArtifactAccuracyRow]] = defaultdict(list)
    for row in rows:
        workload = workload_for_row(row)
        if workload is not None:
            groups[(workload, row.hardware)].append(row)

    points: list[_WorkloadPoint] = []
    for (workload, hardware), grouped_rows in sorted(groups.items()):
        points.append(
            _WorkloadPoint(
                workload=workload,
                hardware=hardware,
                latency_ratio=_geomean(row.latency_ratio for row in grouped_rows),
                energy_ratio=_geomean(row.energy_ratio for row in grouped_rows),
                count=len(grouped_rows),
            )
        )
    return tuple(points)


def _working_set_bytes(row: ArtifactAccuracyRow) -> int:
    dimensions = _parse_dimensions(row.dimensions)
    dtype_bytes = 2
    if row.op_kind == OpKind.BATCHED_GEMM.value:
        batch = int(dimensions["batch"])
        m = int(dimensions["M"])
        n = int(dimensions["N"])
        k = int(dimensions["K"])
        return dtype_bytes * batch * (m * k + k * n + m * n)
    if row.op_kind == OpKind.LAYERNORM.value:
        batch = int(dimensions["batch"])
        dim = int(dimensions["dim"])
        return dtype_bytes * (batch * dim + 2 * dim + batch * dim)
    if row.op_kind == OpKind.SOFTMAX.value:
        batch = int(dimensions["batch"])
        dim = int(dimensions["dim"])
        return _softmax_working_set_bytes(batch, dim)
    raise ValueError(f"unsupported op kind for working-set bytes: {row.op_kind}")


def _gemm_square_working_set_bytes(batch: int, mnk: int) -> int:
    dtype_bytes = 2
    return dtype_bytes * batch * (mnk * mnk + mnk * mnk + mnk * mnk)


def _softmax_working_set_bytes(batch: int, dim: int) -> int:
    dtype_bytes = 2
    return dtype_bytes * (batch * dim + batch * dim)


def _parse_dimensions(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in value.split(","):
        key, raw_value = part.split("=", 1)
        result[key] = raw_value
    return result


def _float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _short_hardware_label(hardware: str) -> str:
    if hardware == "a100_40gb_pcie":
        return "a100"
    return hardware


def _hardware_colors(hardware_order: tuple[str, ...]) -> dict[str, str]:
    palette = ("#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2")
    return {hardware: palette[index % len(palette)] for index, hardware in enumerate(hardware_order)}


def _log_range(values: Iterable[float]) -> tuple[float, float]:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return (0.0, 1.0)
    lower = math.floor(math.log10(min(positive)))
    upper = math.ceil(math.log10(max(positive)))
    if lower == upper:
        lower -= 1
        upper += 1
    return (float(lower), float(upper))


def _log_ticks(lower: float, upper: float) -> tuple[float, ...]:
    return tuple(10.0**power for power in range(int(lower), int(upper) + 1))


def _log_x(value: float, lower: float, upper: float, x: int, width: int) -> float:
    return x + ((math.log10(value) - lower) / (upper - lower)) * width


def _log_y(value: float, lower: float, upper: float, y: int, height: int) -> float:
    return y + height - ((math.log10(value) - lower) / (upper - lower)) * height


def _format_bytes(value: float) -> str:
    units = (("T", 1.0e12), ("G", 1.0e9), ("M", 1.0e6), ("K", 1.0e3))
    for suffix, scale in units:
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _format_ratio(value: float) -> str:
    if 0.01 <= value < 100.0:
        return f"{value:g}"
    return f"{value:.0e}"


def _nice_axis_limit(value: float) -> float:
    if value <= 1.5:
        return 1.5
    if value <= 2.0:
        return 2.0
    if value <= 3.0:
        return 3.0
    return math.ceil(value)


def _derived_plot_path(path: Path, suffix: str) -> Path:
    extension = path.suffix or ".svg"
    return path.with_name(f"{path.stem}_{suffix}{extension}")


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


_PERFORMANCE_DETAIL_CSV_FIELDS = (
    "source_file",
    "row_index",
    "hardware",
    "op_kind",
    "dimensions",
    "implementation",
    "engine",
    "input_dtype",
    "output_dtype",
    "problem_batch",
    "problem_m",
    "problem_n",
    "problem_k",
    "beta_zero",
    "epilogue_reads_c",
    "transpose_a",
    "transpose_b",
    "measured_latency_ms",
    "predicted_latency_ms",
    "latency_ratio",
    "measured_energy_j",
    "predicted_energy_j",
    "energy_ratio",
    "compute_energy_j",
    "dram_energy_j",
    "l2_energy_j",
    "smem_energy_j",
    "static_power_w",
    "static_energy_j",
    "flops",
    "predicted_elapsed_cycles",
    "clock_hz",
    "predicted_flop_per_cycle",
    "predicted_tflops_per_s",
    "achieved_flops_per_s",
    "compute_latency_s",
    "memory_latency_s",
    "roofline_latency_s",
    "kernel_launch_overhead_s",
    "effective_flops_per_s",
    "effective_hbm_bandwidth_bytes_per_s",
    "arithmetic_intensity_flops_per_byte",
    "peak_compute_throughput_flops_per_s",
    "compute_config_utilization_factor",
    "effective_configured_compute_throughput_flops_per_s",
    "compute_issue_utilization_factor",
    "compute_latency_utilization_factor",
    "compute_active_utilization_factor",
    "mma_ilp_efficiency",
    "tile_strategy",
    "cta_tile_m",
    "cta_tile_n",
    "cta_tile_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "mma_shape_m",
    "mma_shape_n",
    "mma_shape_k",
    "pipeline_stages",
    "warps_per_cta",
    "threads_per_cta",
    "registers_per_thread",
    "shared_memory_bytes_per_cta",
    "cta_grid_m",
    "cta_grid_n",
    "cta_grid_k_stages",
    "cta_count",
    "cta_waves",
    "resident_ctas_per_sm",
    "ctas_per_wave",
    "tail_efficiency",
    "tile_efficiency",
    "occupancy_limiting_factors",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "hbm_total_bytes",
    "l2_read_bytes",
    "l2_write_bytes",
    "l2_total_bytes",
    "sram_read_bytes",
    "sram_write_bytes",
    "sram_total_bytes",
    "register_read_bytes",
    "register_write_bytes",
    "register_total_bytes",
    "logical_a_bytes",
    "logical_b_bytes",
    "logical_c_read_bytes",
    "logical_d_store_bytes",
    "l2_requested_transaction_bytes",
    "dram_unique_transaction_bytes",
    "smem_read_transaction_bytes",
    "smem_write_transaction_bytes",
    "smem_total_transaction_bytes",
    "sector_size_bytes",
    "line_size_bytes",
    "hbm_peak_bandwidth_bytes_per_s",
    "hbm_config_utilization_factor",
    "hbm_effective_configured_bandwidth_bytes_per_s",
    "hbm_active_utilization_factor",
    "hbm_average_bandwidth_bytes_per_s",
    "l2_peak_bandwidth_bytes_per_s",
    "l2_config_utilization_factor",
    "l2_effective_configured_bandwidth_bytes_per_s",
    "l2_active_utilization_factor",
    "l2_average_bandwidth_bytes_per_s",
    "smem_peak_bandwidth_bytes_per_s",
    "smem_config_utilization_factor",
    "smem_effective_configured_bandwidth_bytes_per_s",
    "smem_active_utilization_factor",
    "smem_average_bandwidth_bytes_per_s",
    "compute_active_cycles",
    "smem_active_cycles",
    "l2_active_cycles",
    "dram_active_cycles",
    "compute_stage_cycles",
    "mma_issue_cycles",
    "mma_dependency_penalty_cycles",
    "smem_stage_cycles",
    "global_load_issue_cycles",
    "l2_service_cycles",
    "dram_service_cycles",
    "exposed_l2_cycles",
    "exposed_dram_cycles",
    "stage_cycles",
    "prologue_cycles",
    "epilogue_cycles",
    "cta_cycles",
    "compute_smem_overlap_factor",
    "compute_l2_overlap_factor",
    "compute_dram_overlap_factor",
    "smem_flop_per_byte",
    "l2_flop_per_byte",
    "dram_flop_per_byte",
    "merged_flop_per_byte",
    "roofline_bound_compute_flop_per_cycle",
    "roofline_bound_smem_flop_per_cycle",
    "roofline_bound_l2_flop_per_cycle",
    "roofline_bound_dram_flop_per_cycle",
    "hbm_memory_latency_s",
    "l2_memory_latency_s",
    "sram_memory_latency_s",
    "register_memory_latency_s",
    "primary_bottleneck",
    "secondary_bottlenecks",
    "warnings",
    "assumptions",
)
