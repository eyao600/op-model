from __future__ import annotations

import json
import os
import subprocess
import sys
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_predict_returns_json() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opmodel.cli",
            "predict",
            "--hardware",
            str(ROOT / "src/opmodel/configs/hardware/gpu_generic.yaml"),
            "--op",
            str(ROOT / "examples/gemm.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    data = json.loads(result.stdout)
    assert data["latency_s"] > 0
    assert data["energy_j"] > 0
    assert data["flops"] > 0
    assert data["engine"] == "tensor"
    assert data["implementation"] == "roofline.gemm"


def test_cli_validate_artifact_writes_report_and_plot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (data_dir / "a10_softmax_bf16_freq900_lut.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch", "dim", "time", "energy", "prec"])
        writer.writeheader()
        writer.writerow({"batch": "2", "dim": "8", "time": "1.0", "energy": "0.025", "prec": "bf16"})

    output_csv = tmp_path / "accuracy.csv"
    output_plot = tmp_path / "accuracy.svg"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opmodel.cli",
            "validate-artifact",
            "--data-dir",
            str(data_dir),
            "--hardware-dir",
            str(ROOT / "src/opmodel/configs/hardware"),
            "--output-csv",
            str(output_csv),
            "--output-plot",
            str(output_plot),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Processed supported rows: 1" in result.stdout
    assert output_csv.exists()
    assert output_plot.exists()
    assert "Normalized prediction accuracy" in output_plot.read_text(encoding="utf-8")
