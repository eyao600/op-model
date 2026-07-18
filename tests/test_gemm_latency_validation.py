from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from opmodel.hardware import load_hardware
from opmodel.validation.gemm_latency import (
    classify_gemm_size,
    format_gemm_roofline_comparison,
    run_gemm_latency_validation,
    run_gemm_roofline_comparison,
    _with_fixed_overhead,
)


ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = ROOT / "src/opmodel/configs/hardware"


def test_gemm_size_classification_precedence() -> None:
    assert classify_gemm_size(batch=1, dim_m=1, dim_n=4096, dim_k=16) == "vector_like"
    assert classify_gemm_size(batch=1, dim_m=1, dim_n=4096, dim_k=1) == "small"
    assert classify_gemm_size(batch=1, dim_m=1, dim_n=1, dim_k=128) == "small"
    assert classify_gemm_size(batch=1, dim_m=4096, dim_n=1, dim_k=16) == "small"
    assert classify_gemm_size(batch=1, dim_m=4096, dim_n=128, dim_k=1) == "small_k"
    assert classify_gemm_size(batch=1, dim_m=64, dim_n=64, dim_k=16) == "small"
    assert classify_gemm_size(batch=1, dim_m=64, dim_n=1024, dim_k=128) == "skinny"
    assert classify_gemm_size(batch=1, dim_m=128, dim_n=128, dim_k=16) == "small_k"
    assert classify_gemm_size(batch=256, dim_m=1024, dim_n=1024, dim_k=128) == "large"
    assert classify_gemm_size(batch=1_000_000, dim_m=128, dim_n=128, dim_k=128) == "large"
    assert classify_gemm_size(batch=1, dim_m=128, dim_n=128, dim_k=128) == "regular"


def test_fixed_overhead_training_rows_are_held_out(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_gemm_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        [
            _row(batch=1, m=1, n=64, k=32, time="0.20"),
            _row(batch=1, m=16, n=16, k=64, time="0.25"),
            _row(batch=1, m=64, n=1024, k=128, time="0.50"),
            _row(batch=1, m=128, n=128, k=16, time="0.45"),
            _row(batch=256, m=1024, n=1024, k=128, time="2.50"),
            _row(batch=1, m=128, n=128, k=128, time="0.80"),
        ],
    )

    report = run_gemm_latency_validation(
        data_dir=data_dir,
        hardware_dir=HARDWARE_DIR,
        training_per_class=1,
    )

    train_rows = [row for row in report.rows if row.split == "train"]
    validation_rows = [row for row in report.rows if row.split == "validation"]
    assert len(train_rows) == 2
    assert {row.kernel_class for row in train_rows} == {"vector_like", "small"}
    assert len(validation_rows) == 4
    assert {row.kernel_class for row in validation_rows} == {
        "skinny",
        "small_k",
        "large",
        "regular",
    }
    assert report.fixed_overheads["a10"].calibrated_fixed_overhead_cycles >= 0
    assert all(metric.split == "validation" for metric in report.metrics)
    all_metric = next(metric for metric in report.metrics if metric.group == "all")
    assert all_metric.count == 4
    assert all_metric.energy_mape_pct >= 0.0
    assert all_metric.energy_median_ape_pct >= 0.0
    assert all_metric.energy_p90_ape_pct >= 0.0
    assert all_metric.energy_geomean_ratio > 0.0
    assert math.isfinite(all_metric.energy_mean_signed_pct_error)
    assert all(row.measured_energy_j == 0.1 for row in report.rows)
    assert all(row.predicted_energy_j >= 0.0 for row in report.rows)


def test_no_calibration_uses_config_overhead(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_gemm_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        [_row(batch=1, m=1, n=64, k=32), _row(batch=1, m=16, n=16, k=64)],
    )

    hardware = load_hardware(HARDWARE_DIR / "a10.yaml")
    report = run_gemm_latency_validation(
        data_dir=data_dir,
        hardware_dir=HARDWARE_DIR,
        calibrate_fixed_overhead=False,
    )

    overhead = report.fixed_overheads["a10"]
    assert overhead.training_rows == ()
    assert overhead.calibrated_fixed_overhead_cycles == hardware.compute.device_fixed_overhead_cycles
    assert {row.split for row in report.rows} == {"validation"}


def test_with_fixed_overhead_preserves_other_hardware_fields() -> None:
    hardware = load_hardware(HARDWARE_DIR / "a10.yaml")
    updated = _with_fixed_overhead(hardware, 123)

    assert updated.compute.device_fixed_overhead_cycles == 123
    assert replace(updated.compute, device_fixed_overhead_cycles=hardware.compute.device_fixed_overhead_cycles) == hardware.compute
    assert updated.memory_levels == hardware.memory_levels
    assert updated.utilization == hardware.utilization


def test_cli_validate_gemm_latency_writes_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_gemm_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        [
            _row(batch=1, m=1, n=64, k=32, time="0.20"),
            _row(batch=1, m=16, n=16, k=64, time="0.25"),
            _row(batch=1, m=128, n=128, k=128, time="0.80"),
        ],
    )
    output_csv = tmp_path / "latency.csv"
    output_params = tmp_path / "params.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opmodel.cli",
            "validate-gemm-latency",
            "--data-dir",
            str(data_dir),
            "--hardware-dir",
            str(HARDWARE_DIR),
            "--training-per-class",
            "1",
            "--output-csv",
            str(output_csv),
            "--output-params",
            str(output_params),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Held-out validation rows: 1" in result.stdout
    assert "Held-out latency accuracy:" in result.stdout
    assert "Held-out energy accuracy:" in result.stdout
    assert output_csv.exists()
    assert output_params.exists()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert {
        "measured_energy_j",
        "predicted_energy_j",
        "energy_ratio",
        "energy_abs_pct_error",
        "energy_signed_pct_error",
    }.issubset(rows[0])
    params = json.loads(output_params.read_text(encoding="utf-8"))
    assert "a10" in params
    assert params["a10"]["training_rows"]


def test_base_roofline_validation_applies_calibrated_fixed_overhead(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_gemm_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        [
            _row(batch=1, m=1, n=64, k=32, time="0.20"),
            _row(batch=1, m=16, n=16, k=64, time="0.25"),
            _row(batch=1, m=128, n=128, k=128, time="0.80"),
        ],
    )
    report = run_gemm_latency_validation(
        data_dir=data_dir,
        hardware_dir=HARDWARE_DIR,
        model_name="roofline",
        training_per_class=1,
    )

    overhead = report.fixed_overheads["a10"].calibrated_fixed_overhead_cycles
    assert overhead > 0
    assert all(row.modeled_device_cycles is not None for row in report.rows)
    assert all(row.total_device_cycles is not None for row in report.rows)
    assert all(
        math.isclose(
            row.total_device_cycles - row.modeled_device_cycles,
            overhead,
            abs_tol=1.0e-6,
        )
        for row in report.rows
    )


def test_roofline_comparison_includes_all_models_and_deltas(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_gemm_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        [
            _row(batch=1, m=1, n=64, k=32, time="0.20"),
            _row(batch=1, m=16, n=16, k=64, time="0.25"),
            _row(batch=1, m=128, n=128, k=128, time="0.80"),
        ],
    )
    comparison = run_gemm_roofline_comparison(
        data_dir=data_dir,
        hardware_dir=HARDWARE_DIR,
        training_per_class=1,
    )

    assert comparison.model_names == (
        "roofline",
        "extended_roofline",
        "effective_roofline",
    )
    assert set(comparison.reports) == set(comparison.model_names)
    overall = [metric for metric in comparison.metrics if metric.group == "all"]
    assert [metric.model_name for metric in overall] == list(comparison.model_names)
    assert overall[0].latency_mape_delta_pct == 0.0
    text = format_gemm_roofline_comparison(comparison)
    assert "baseline=roofline" in text
    assert "extended_roofline" in text
    assert "effective_roofline" in text
    assert "MAPE delta versus baseline" in text


def test_cli_compares_all_roofline_models(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_gemm_csv(
        data_dir / "a10_gemm_bf16bf16_freq900_lut.csv",
        [
            _row(batch=1, m=1, n=64, k=32, time="0.20"),
            _row(batch=1, m=16, n=16, k=64, time="0.25"),
            _row(batch=1, m=128, n=128, k=128, time="0.80"),
        ],
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opmodel.cli",
            "validate-gemm-latency",
            "--data-dir",
            str(data_dir),
            "--hardware-dir",
            str(HARDWARE_DIR),
            "--training-per-class",
            "1",
            "--compare-rooflines",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "GEMM roofline model comparison:" in result.stdout
    assert "roofline:" in result.stdout
    assert "extended_roofline:" in result.stdout
    assert "effective_roofline:" in result.stdout


def _write_gemm_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "batch",
        "dimM",
        "dimN",
        "dimK",
        "trans",
        "precM",
        "precA",
        "time",
        "energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    *,
    batch: int,
    m: int,
    n: int,
    k: int,
    time: str = "1.0",
) -> dict[str, str]:
    return {
        "batch": str(batch),
        "dimM": str(m),
        "dimN": str(n),
        "dimK": str(k),
        "trans": "nn",
        "precM": "bf16",
        "precA": "bf16",
        "time": time,
        "energy": "0.1",
    }
