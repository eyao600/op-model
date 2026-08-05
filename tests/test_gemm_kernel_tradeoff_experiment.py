from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from experiments.gemm_kernel_tradeoff.run import (
    DEFAULT_HARDWARE,
    LLM_SHAPES,
    LOCKED_SHAPES,
    GemmShape,
    build_shape_result,
    evaluate_shapes,
    main,
    make_gemm_op,
    parse_batched_shape,
    parse_shape,
    pareto_template_names,
)
from opmodel.api import DType, OpKind
from opmodel.hardware import load_hardware


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_locked_shape_matrix_covers_controls_and_tradeoffs() -> None:
    assert len(LOCKED_SHAPES) == 13
    assert LOCKED_SHAPES[0] == GemmShape(
        "small_control", 64, 64, 64, "control"
    )
    assert any(shape.edge_case for shape in LOCKED_SHAPES)
    assert {shape.family for shape in LOCKED_SHAPES} >= {
        "control",
        "square",
        "rectangle",
        "wide",
        "tall",
        "wide_skinny",
        "tall_skinny",
        "vector_like",
        "edge",
    }
    assert max(shape.k for shape in LOCKED_SHAPES) == 8192


def test_shape_parser_and_gemm_construction() -> None:
    shape = parse_shape("custom:96:160:64")
    op = make_gemm_op(shape)

    assert shape == GemmShape("custom", 96, 160, 64, "custom")
    assert op.kind is OpKind.GEMM
    assert op.attrs["gemm_selection_backend"] == "effective_roofline"
    assert [tensor.shape for tensor in op.tensors] == [
        (96, 64),
        (64, 160),
        (96, 160),
    ]
    assert all(tensor.dtype is DType.BF16 for tensor in op.tensors)

    with pytest.raises(argparse.ArgumentTypeError, match="LABEL:M:N:K"):
        parse_shape("missing-dimensions")

    batched_shape = parse_batched_shape("attention:32:1:2048:128")
    batched_op = make_gemm_op(batched_shape)
    assert batched_shape == GemmShape(
        "attention", 1, 2048, 128, "custom", batch=32, batched=True
    )
    assert batched_op.kind is OpKind.BATCHED_GEMM
    assert [tensor.shape for tensor in batched_op.tensors] == [
        (32, 1, 128),
        (32, 128, 2048),
        (32, 1, 2048),
    ]


def test_llm_shape_suite_covers_dense_attention_fusion_and_moe() -> None:
    assert len(LLM_SHAPES) == 27
    assert sum(shape.batched for shape in LLM_SHAPES) == 4
    assert {shape.family for shape in LLM_SHAPES} >= {
        "hidden_projection",
        "gqa_projection",
        "ffn_up",
        "ffn_down",
        "vocab_projection",
        "attention",
        "fused_projection",
        "moe",
    }
    assert {shape.m for shape in LLM_SHAPES if shape.family == "hidden_projection"} == {
        1,
        16,
        128,
        512,
        2048,
    }


def test_effective_candidate_evaluation_and_regret_formulas() -> None:
    hardware = load_hardware(DEFAULT_HARDWARE)
    shape = GemmShape("tiny", 64, 64, 64, "test")

    (result,) = evaluate_shapes(
        (shape,),
        hardware,
        min_candidates=5,
        runtime_budget_s=10.0,
    )

    assert len(result.candidates) >= 5
    assert result.pareto_templates == pareto_template_names(result.candidates)
    assert result.latency_winner.profile.latency_s == min(
        candidate.profile.latency_s for candidate in result.candidates
    )
    assert result.energy_winner.profile.energy_j == min(
        candidate.profile.energy_j for candidate in result.candidates
    )
    assert result.energy_regret_fraction == pytest.approx(
        result.latency_winner.profile.energy_j
        / result.energy_winner.profile.energy_j
        - 1.0
    )
    assert result.energy_saving_fraction == pytest.approx(
        1.0
        - result.energy_winner.profile.energy_j
        / result.latency_winner.profile.energy_j
    )
    assert result.latency_regret_fraction == pytest.approx(
        result.energy_winner.profile.latency_s
        / result.latency_winner.profile.latency_s
        - 1.0
    )
    assert build_shape_result(shape, result.candidates) == result


def test_deep_shapes_expose_strong_latency_energy_tradeoff() -> None:
    hardware = load_hardware(DEFAULT_HARDWARE)
    shapes = tuple(
        shape
        for shape in LOCKED_SHAPES
        if shape.label in {"deep_square", "deep_rectangle"}
    )

    results = evaluate_shapes(
        shapes,
        hardware,
        min_candidates=12,
        runtime_budget_s=10.0,
    )
    by_label = {result.shape.label: result for result in results}

    assert by_label["deep_square"].energy_saving_fraction > 0.11
    assert 0.0 < by_label["deep_square"].latency_regret_fraction < 0.05
    assert by_label["deep_rectangle"].energy_saving_fraction > 0.10
    assert 0.0 < by_label["deep_rectangle"].latency_regret_fraction < 0.01
    assert all(result.strong_tradeoff_example for result in results)


def test_experiment_writes_csv_and_png_artifacts(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--hardware",
            str(DEFAULT_HARDWARE),
            "--output-dir",
            str(tmp_path),
            "--shape",
            "small:64:64:64",
            "--shape",
            "square:1536:1536:128",
            "--min-candidates",
            "5",
            "--runtime-budget-s",
            "10",
            "--dpi",
            "72",
        ]
    )

    assert exit_code == 0
    candidates_path = tmp_path / "candidates.csv"
    summary_path = tmp_path / "shape_summary.csv"
    assert candidates_path.exists()
    assert summary_path.exists()

    with candidates_path.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    assert {row["shape_label"] for row in candidates} == {"small", "square"}
    assert all(row["dtype"] == "bf16" for row in candidates)
    assert all(row["template"].startswith("sm80_") for row in candidates)
    assert all("hbm_read_bytes" in row for row in candidates)
    assert all("transaction_dram_total_bytes" in row for row in candidates)
    assert all("l2_capacity_miss_fraction" in row for row in candidates)
    assert all("pareto_optimal" in row for row in candidates)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 2
    assert all("energy_regret_percent" in row for row in summary)
    assert all("energy_saving_percent" in row for row in summary)
    assert all("latency_regret_percent" in row for row in summary)

    for name in (
        "latency_energy.png",
        "objective_regret.png",
        "winner_traffic.png",
        "winner_energy_breakdown.png",
        "winner_pair_dynamic_energy.png",
    ):
        path = tmp_path / name
        assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_coverage_guard_writes_diagnostic_only(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--hardware",
            str(DEFAULT_HARDWARE),
            "--output-dir",
            str(tmp_path),
            "--shape",
            "unsupported:2048:2048:4096",
            "--min-candidates",
            "13",
            "--runtime-budget-s",
            "10",
        ]
    )

    assert exit_code == 2
    with (tmp_path / "coverage.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "shape_label": "unsupported",
            "m": "2048",
            "n": "2048",
            "k": "4096",
            "candidate_count": "12",
            "minimum_required": "13",
            "passes": "False",
        }
    ]
    assert not (tmp_path / "candidates.csv").exists()
    assert not (tmp_path / "latency_energy.png").exists()
