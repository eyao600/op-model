from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from opmodel.validation.artifact_accuracy import (
    DEFAULT_ARTIFACT_DATA_DIR,
    format_text_report,
    run_artifact_validation,
    write_csv_report,
    write_validation_plots,
)


ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = ROOT / "src/opmodel/configs/hardware"


def test_artifact_validation_filters_bf16_and_reports_ratios(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        ["batch", "dimM", "dimN", "dimK", "trans", "precM", "precA", "time", "energy"],
        [
            {
                "batch": "2",
                "dimM": "4",
                "dimN": "8",
                "dimK": "16",
                "trans": "nn",
                "precM": "bf16",
                "precA": "bf16",
                "time": "2.0",
                "energy": "0.2",
            },
            {
                "batch": "2",
                "dimM": "4",
                "dimN": "8",
                "dimK": "16",
                "trans": "nn",
                "precM": "fp32",
                "precA": "fp32",
                "time": "2.0",
                "energy": "0.2",
            },
        ],
    )
    _write_csv(
        data_dir / "yz8_layernorm_bf16_freq900_lut.csv",
        ["batch", "dim", "time", "energy", "prec"],
        [{"batch": "2", "dim": "8", "time": "1.5", "energy": "0.05", "prec": "bf16"}],
    )
    _write_csv(
        data_dir / "yz8_softmax_bf16_freq900_lut.csv",
        ["batch", "dim", "time", "energy", "prec"],
        [{"batch": "2", "dim": "8", "time": "1.0", "energy": "0.025", "prec": "bf16"}],
    )
    _write_csv(
        data_dir / "a10_conv2d_bf16_freq900_lut.csv",
        ["b", "m", "c", "hw", "rs", "time", "energy", "prec"],
        [{"b": "1", "m": "1", "c": "1", "hw": "1", "rs": "1", "time": "1", "energy": "1", "prec": "bf16"}],
    )
    _write_csv(
        data_dir / "netsres_layernorm_bf16_nolock.csv",
        ["batch", "dim", "time", "energy", "prec"],
        [{"batch": "2", "dim": "8", "time": "1.0", "energy": "10.0", "prec": "bf16"}],
    )

    report = run_artifact_validation(data_dir=data_dir, hardware_dir=HARDWARE_DIR)

    assert len(report.rows) == 3
    assert {row.op_kind for row in report.rows} == {"batched_gemm", "layernorm", "softmax"}
    assert {row.hardware for row in report.rows} == {"a10", "a100_40gb_pcie"}
    gemm = next(row for row in report.rows if row.op_kind == "batched_gemm")
    assert gemm.measured_energy_j == pytest.approx(0.2)
    assert gemm.latency_ratio == pytest.approx(
        gemm.predicted_latency_ms / gemm.measured_latency_ms
    )
    assert gemm.energy_ratio == pytest.approx(gemm.predicted_energy_j / gemm.measured_energy_j)
    assert report.skip_counts["non_bf16_row"] == 1
    assert report.skip_counts["unsupported_bf16_file"] == 1
    assert report.skip_counts["unsupported_hardware_file"] == 1

    text = format_text_report(report)
    assert "Processed supported rows: 3" in text
    assert "a10/batched_gemm" in text


def test_artifact_energy_uses_csv_value_directly(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "a10_softmax_bf16_freq900_lut.csv",
        ["batch", "dim", "time", "energy", "prec"],
        [{"batch": "2", "dim": "8", "time": "10.0", "energy": "0.125", "prec": "bf16"}],
    )

    report = run_artifact_validation(data_dir=data_dir, hardware_dir=HARDWARE_DIR)

    assert report.rows[0].measured_energy_j == pytest.approx(0.125)


def test_artifact_report_writes_csv_and_svg(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        ["batch", "dimM", "dimN", "dimK", "trans", "precM", "precA", "time", "energy"],
        [
            {
                "batch": "16",
                "dimM": "256",
                "dimN": "256",
                "dimK": "256",
                "trans": "nn",
                "precM": "bf16",
                "precA": "bf16",
                "time": "2.0",
                "energy": "0.2",
            }
        ],
    )
    _write_csv(
        data_dir / "a10_softmax_bf16_freq900_lut.csv",
        ["batch", "dim", "time", "energy", "prec"],
        [
            {"batch": "2", "dim": "8", "time": "1.0", "energy": "0.025", "prec": "bf16"},
            {
                "batch": "8192",
                "dim": "512",
                "time": "1.0",
                "energy": "0.025",
                "prec": "bf16",
            },
        ],
    )
    report = run_artifact_validation(data_dir=data_dir, hardware_dir=HARDWARE_DIR)

    csv_path = tmp_path / "reports" / "accuracy.csv"
    svg_path = tmp_path / "reports" / "accuracy.svg"
    write_csv_report(report, csv_path)
    plot_paths = write_validation_plots(report, svg_path)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["hardware"] == "a10"
    assert {row["op_kind"] for row in rows} == {"batched_gemm", "softmax"}

    svg = svg_path.read_text(encoding="utf-8")
    assert "Normalized prediction scatter by working set" in svg
    assert "working-set bytes" in svg
    assert "a10" in svg
    assert "softmax" in svg
    assert "1.0 perfect" in svg
    assert "latency predicted / measured" in svg
    assert "energy predicted / measured" in svg
    assert len(plot_paths) == 4
    gemm_svg = (tmp_path / "reports" / "accuracy_gemm_workloads.svg").read_text(
        encoding="utf-8"
    )
    softmax_svg = (tmp_path / "reports" / "accuracy_softmax_workloads.svg").read_text(
        encoding="utf-8"
    )
    assert "Square GEMM workload ratios" in gemm_svg
    assert "B16/MNK256" in gemm_svg
    assert "Softmax workload ratios" in softmax_svg
    assert "B2^13/D512" in softmax_svg


def test_validation_ingests_effective_softmax_and_flashattention(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "yz8_softmax_bf16_freq900_lut.csv",
        ["batch", "dim", "time", "energy", "prec"],
        [{"batch": "128", "dim": "512", "time": "0.1", "energy": "0.01", "prec": "bf16"}],
    )
    _write_csv(
        tmp_path / "yz8_flashattention_freq900_lut.csv",
        [
            "batch",
            "n_head",
            "seq_len",
            "head_dim",
            "time",
            "energy",
            "precM",
            "precA",
            "kernel_name",
            "max_concurrent_block",
        ],
        [
            {
                "batch": "1",
                "n_head": "4",
                "seq_len": "128",
                "head_dim": "64",
                "time": "0.1",
                "energy": "0.01",
                "precM": "bf16",
                "precA": "bf16",
                "kernel_name": (
                    "Flash_fwd_kernel_traits<(int)64, (int)128, "
                    "(int)128, (int)4"
                ),
                "max_concurrent_block": "2",
            }
        ],
    )
    report = run_artifact_validation(
        data_dir=tmp_path,
        hardware_dir=HARDWARE_DIR,
        model_name="effective_roofline",
    )
    assert {row.op_kind for row in report.rows} == {
        "softmax",
        "attention_prefill",
    }
    assert all(row.predicted_latency_ms > 0.0 for row in report.rows)
    assert all(row.predicted_energy_j > 0.0 for row in report.rows)


def test_real_artifact_data_smoke() -> None:
    if not DEFAULT_ARTIFACT_DATA_DIR.is_dir():
        pytest.skip(f"artifact data directory not present: {DEFAULT_ARTIFACT_DATA_DIR}")

    report = run_artifact_validation(
        data_dir=DEFAULT_ARTIFACT_DATA_DIR,
        hardware_dir=HARDWARE_DIR,
        limit=1,
    )

    assert report.rows
    assert {"a10", "a100_40gb_pcie"}.issubset({row.hardware for row in report.rows})
    assert all(math.isfinite(row.latency_ratio) for row in report.rows)
    assert all(math.isfinite(row.energy_ratio) for row in report.rows)


def test_real_artifact_data_base_model_smoke() -> None:
    if not DEFAULT_ARTIFACT_DATA_DIR.is_dir():
        pytest.skip(f"artifact data directory not present: {DEFAULT_ARTIFACT_DATA_DIR}")

    report = run_artifact_validation(
        data_dir=DEFAULT_ARTIFACT_DATA_DIR,
        hardware_dir=HARDWARE_DIR,
        model_name="base",
        limit=1,
    )

    assert report.rows
    assert {"a10", "a100_40gb_pcie"}.issubset({row.hardware for row in report.rows})
    assert all(math.isfinite(row.latency_ratio) for row in report.rows)
    assert all(math.isfinite(row.energy_ratio) for row in report.rows)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
