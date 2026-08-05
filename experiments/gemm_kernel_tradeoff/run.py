from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opmodel.api import DType, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.hardware import HardwareSpec, load_hardware
from opmodel.models.effective_roofline import (
    evaluate_gemm_template_candidates,
    select_gemm_template_candidate,
)
from opmodel.models.extended_roofline import GemmCandidateEvaluation

DEFAULT_HARDWARE = ROOT / "src/opmodel/configs/hardware/a100_40gb_pcie.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "validation_artifacts/gemm_kernel_tradeoff"
ALL_CANDIDATES_SHORTLIST = 999
MIB = float(1024**2)


@dataclass(frozen=True)
class GemmShape:
    label: str
    m: int
    n: int
    k: int
    family: str
    edge_case: bool = False
    batch: int = 1
    batched: bool = False

    @property
    def dimensions(self) -> str:
        prefix = f"b{self.batch}:" if self.batched else ""
        return f"{prefix}{self.m}x{self.n}x{self.k}"


LOCKED_SHAPES = (
    GemmShape("small_control", 64, 64, 64, "control"),
    GemmShape("large_shallow_control", 4096, 2048, 128, "control"),
    GemmShape("square_shallow", 1536, 1536, 128, "square"),
    GemmShape("wide_shallow", 512, 4096, 128, "wide"),
    GemmShape("tall_shallow", 4096, 512, 128, "tall"),
    GemmShape("vector_like_deep", 1, 4096, 8192, "vector_like"),
    GemmShape("wide_skinny_deep", 129, 4096, 8192, "wide_skinny"),
    GemmShape("tall_skinny_deep", 4096, 129, 8192, "tall_skinny"),
    GemmShape("deep_square_mid", 3072, 3072, 1024, "square"),
    GemmShape("deep_rectangle", 3072, 4096, 1024, "rectangle"),
    GemmShape("wide_deep_capacity", 512, 4096, 8192, "wide"),
    GemmShape("deep_square", 4096, 4096, 8192, "square"),
    GemmShape("tile_edge_deep", 4096, 4097, 8192, "edge", edge_case=True),
)


LLM_SHAPES = (
    # Dense projections across decode, continuous batching, and prefill.
    GemmShape("hidden_decode_1", 1, 4096, 4096, "hidden_projection"),
    GemmShape("hidden_decode_16", 16, 4096, 4096, "hidden_projection"),
    GemmShape("hidden_prefill_128", 128, 4096, 4096, "hidden_projection"),
    GemmShape("hidden_prefill_512", 512, 4096, 4096, "hidden_projection"),
    GemmShape("hidden_prefill_2048", 2048, 4096, 4096, "hidden_projection"),
    GemmShape("gqa_kv_decode", 1, 1024, 4096, "gqa_projection"),
    GemmShape("gqa_kv_prefill", 512, 1024, 4096, "gqa_projection"),
    GemmShape("ffn_up_decode", 1, 11008, 4096, "ffn_up"),
    GemmShape("ffn_up_prefill", 512, 11008, 4096, "ffn_up"),
    GemmShape("ffn_down_decode", 1, 4096, 11008, "ffn_down"),
    GemmShape("ffn_down_prefill", 512, 4096, 11008, "ffn_down"),
    GemmShape("vocab_decode", 1, 32000, 4096, "vocab_projection"),
    GemmShape("vocab_prefill", 512, 32000, 4096, "vocab_projection"),
    # Per-query-head attention GEMMs keep the head batch explicit.
    GemmShape("attention_decode_qk", 1, 2048, 128, "attention", batch=32, batched=True),
    GemmShape("attention_decode_pv", 1, 128, 2048, "attention", batch=32, batched=True),
    GemmShape("attention_prefill_qk", 512, 512, 128, "attention", batch=32, batched=True),
    GemmShape("attention_prefill_pv", 512, 128, 512, "attention", batch=32, batched=True),
    # Common inference-engine weight fusions.
    GemmShape("fused_gqa_qkv_decode", 1, 6144, 4096, "fused_projection"),
    GemmShape("fused_gqa_qkv_prefill", 512, 6144, 4096, "fused_projection"),
    GemmShape("fused_gate_up_decode", 1, 22016, 4096, "fused_projection"),
    GemmShape("fused_gate_up_prefill", 512, 22016, 4096, "fused_projection"),
    # Routed-token counts for a single MoE expert.
    GemmShape("moe_up_8", 8, 11008, 4096, "moe"),
    GemmShape("moe_up_32", 32, 11008, 4096, "moe"),
    GemmShape("moe_up_128", 128, 11008, 4096, "moe"),
    GemmShape("moe_down_8", 8, 4096, 11008, "moe"),
    GemmShape("moe_down_32", 32, 4096, 11008, "moe"),
    GemmShape("moe_down_128", 128, 4096, 11008, "moe"),
)

# A compact summary for the paired-winner figure: three HBM-driven tradeoffs,
# one L2/SMEM-driven tradeoff, and one same-winner control.
LLM_DYNAMIC_PAIR_PLOT_LABELS = (
    "vocab_prefill",
    "ffn_up_prefill",
    "ffn_down_prefill",
    "hidden_prefill_512",
    "hidden_prefill_2048",
)


@dataclass(frozen=True)
class CoverageRow:
    shape: GemmShape
    candidate_count: int
    minimum_required: int

    @property
    def passes(self) -> bool:
        return self.candidate_count >= self.minimum_required


@dataclass(frozen=True)
class ShapeResult:
    shape: GemmShape
    candidates: tuple[GemmCandidateEvaluation, ...]
    latency_winner: GemmCandidateEvaluation
    energy_winner: GemmCandidateEvaluation
    pareto_templates: frozenset[str]

    @property
    def same_winner(self) -> bool:
        return self.latency_winner.template.name == self.energy_winner.template.name

    @property
    def energy_regret_fraction(self) -> float:
        optimum = self.energy_winner.profile.energy_j
        if optimum <= 0.0:
            return 0.0
        return self.latency_winner.profile.energy_j / optimum - 1.0

    @property
    def energy_saving_fraction(self) -> float:
        latency_energy = self.latency_winner.profile.energy_j
        if latency_energy <= 0.0:
            return 0.0
        return 1.0 - self.energy_winner.profile.energy_j / latency_energy

    @property
    def latency_regret_fraction(self) -> float:
        optimum = self.latency_winner.profile.latency_s
        if optimum <= 0.0:
            return 0.0
        return self.energy_winner.profile.latency_s / optimum - 1.0

    @property
    def aligned_success_example(self) -> bool:
        return (
            not self.shape.edge_case
            and not self.same_winner
            and self.energy_regret_fraction >= 0.015
            and self.latency_regret_fraction <= 0.02
        )

    @property
    def strong_tradeoff_example(self) -> bool:
        return (
            not self.shape.edge_case
            and not self.same_winner
            and self.energy_saving_fraction >= 0.10
            and 0.0 < self.latency_regret_fraction <= 0.10
        )


class CoverageError(RuntimeError):
    def __init__(self, coverage: tuple[CoverageRow, ...]) -> None:
        self.coverage = coverage
        failed = [row for row in coverage if not row.passes]
        details = ", ".join(
            f"{row.shape.label}={row.candidate_count}" for row in failed
        )
        super().__init__(
            f"catalogue coverage below required minimum for {details}"
        )


class RuntimeBudgetError(RuntimeError):
    def __init__(self, elapsed_s: float, coverage: tuple[CoverageRow, ...]) -> None:
        self.elapsed_s = elapsed_s
        self.coverage = coverage
        super().__init__(f"candidate evaluation exceeded runtime budget ({elapsed_s:.3f}s)")


def make_gemm_op(shape: GemmShape) -> LocalOp:
    if shape.batched:
        tensors = (
            TensorSpec(TensorRole.INPUT, (shape.batch, shape.m, shape.k), DType.BF16),
            TensorSpec(TensorRole.WEIGHT, (shape.batch, shape.k, shape.n), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, (shape.batch, shape.m, shape.n), DType.BF16),
        )
        kind = OpKind.BATCHED_GEMM
    else:
        tensors = (
            TensorSpec(TensorRole.INPUT, (shape.m, shape.k), DType.BF16),
            TensorSpec(TensorRole.WEIGHT, (shape.k, shape.n), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, (shape.m, shape.n), DType.BF16),
        )
        kind = OpKind.GEMM
    return LocalOp(
        name=f"gemm_kernel_tradeoff_{shape.label}",
        kind=kind,
        phase=Phase.INFERENCE,
        tensors=tensors,
        attrs={"gemm_selection_backend": "effective_roofline"},
    )


def parse_shape(value: str) -> GemmShape:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be LABEL:M:N:K")
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("shape label must not be empty")
    try:
        m, n, k = (int(part) for part in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc
    if min(m, n, k) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return GemmShape(label, m, n, k, "custom")


def parse_batched_shape(value: str) -> GemmShape:
    parts = value.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("batched shape must be LABEL:BATCH:M:N:K")
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("shape label must not be empty")
    try:
        batch, m, n, k = (int(part) for part in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc
    if min(batch, m, n, k) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return GemmShape(label, m, n, k, "custom", batch=batch, batched=True)


def evaluate_shapes(
    shapes: Sequence[GemmShape],
    hardware: HardwareSpec,
    *,
    min_candidates: int,
    runtime_budget_s: float,
) -> tuple[ShapeResult, ...]:
    started = time.perf_counter()
    evaluated: list[tuple[GemmShape, tuple[GemmCandidateEvaluation, ...]]] = []
    coverage: list[CoverageRow] = []
    for shape in shapes:
        candidates = evaluate_gemm_template_candidates(
            make_gemm_op(shape),
            hardware,
            shortlist_size=ALL_CANDIDATES_SHORTLIST,
        )
        evaluated.append((shape, candidates))
        coverage.append(CoverageRow(shape, len(candidates), min_candidates))
        elapsed_s = time.perf_counter() - started
        if elapsed_s > runtime_budget_s:
            raise RuntimeBudgetError(elapsed_s, tuple(coverage))

    coverage_tuple = tuple(coverage)
    if any(not row.passes for row in coverage_tuple):
        raise CoverageError(coverage_tuple)

    return tuple(
        build_shape_result(shape, candidates) for shape, candidates in evaluated
    )


def build_shape_result(
    shape: GemmShape,
    candidates: tuple[GemmCandidateEvaluation, ...],
) -> ShapeResult:
    latency_winner = select_gemm_template_candidate(candidates, objective="latency")
    energy_winner = select_gemm_template_candidate(candidates, objective="energy")
    return ShapeResult(
        shape=shape,
        candidates=candidates,
        latency_winner=latency_winner,
        energy_winner=energy_winner,
        pareto_templates=pareto_template_names(candidates),
    )


def pareto_template_names(
    candidates: Sequence[GemmCandidateEvaluation],
) -> frozenset[str]:
    frontier: set[str] = set()
    for candidate in candidates:
        latency = candidate.profile.latency_s
        energy = candidate.profile.energy_j
        dominated = any(
            other.profile.latency_s <= latency
            and other.profile.energy_j <= energy
            and (
                other.profile.latency_s < latency
                or other.profile.energy_j < energy
            )
            for other in candidates
            if other is not candidate
        )
        if not dominated:
            frontier.add(candidate.template.name)
    return frozenset(frontier)


def study_succeeds(results: Sequence[ShapeResult]) -> bool:
    different_winners = sum(not result.same_winner for result in results)
    return different_winners >= 3 and any(result.strong_tradeoff_example for result in results)


def _optional_int(value: int | None) -> int:
    return 0 if value is None else int(value)


def hierarchy_traffic_bytes(
    candidate: GemmCandidateEvaluation, hierarchy: str
) -> int:
    access = candidate.profile.memory_access
    if hierarchy == "hbm":
        return access.hbm_read_bytes + access.hbm_write_bytes
    if hierarchy == "l2":
        return _optional_int(access.l2_read_bytes) + _optional_int(access.l2_write_bytes)
    if hierarchy == "sram":
        return _optional_int(access.sram_read_bytes) + _optional_int(access.sram_write_bytes)
    if hierarchy == "register":
        return _optional_int(access.register_read_bytes) + _optional_int(
            access.register_write_bytes
        )
    raise ValueError(f"unknown hierarchy: {hierarchy}")


def _energy_components(candidate: GemmCandidateEvaluation) -> dict[str, float]:
    energy = candidate.profile.energy_breakdown
    return {
        "compute": energy.compute_j,
        "hbm": energy.hbm_j,
        "l2": energy.l2_j,
        "sram": energy.sram_j,
        "register": energy.register_j,
        "static": energy.static_j,
    }


def candidate_rows(results: Sequence[ShapeResult]) -> Iterable[dict[str, object]]:
    for result in results:
        for candidate in result.candidates:
            template = candidate.template
            access = candidate.profile.memory_access
            diagnostics = candidate.profile.diagnostics
            transaction = diagnostics.get("transaction_bytes", {})
            energy = _energy_components(candidate)
            yield {
                "shape_label": result.shape.label,
                "shape_family": result.shape.family,
                "op_kind": (
                    OpKind.BATCHED_GEMM.value
                    if result.shape.batched
                    else OpKind.GEMM.value
                ),
                "batch": result.shape.batch,
                "m": result.shape.m,
                "n": result.shape.n,
                "k": result.shape.k,
                "dtype": DType.BF16.value,
                "candidate_count": len(result.candidates),
                "template": template.name,
                "cta_m": template.cta_m,
                "cta_n": template.cta_n,
                "cta_k": template.cta_k,
                "warp_m": template.warp_m,
                "warp_n": template.warp_n,
                "warp_k": template.warp_k,
                "pipeline_stages": template.pipeline_stages,
                "warps_per_cta": template.warps_per_cta,
                "latency_s": candidate.profile.latency_s,
                "latency_us": candidate.profile.latency_s * 1.0e6,
                "total_energy_j": candidate.profile.energy_j,
                "total_energy_mj": candidate.profile.energy_j * 1.0e3,
                "compute_energy_j": energy["compute"],
                "hbm_energy_j": energy["hbm"],
                "l2_energy_j": energy["l2"],
                "sram_energy_j": energy["sram"],
                "register_energy_j": energy["register"],
                "static_energy_j": energy["static"],
                "hbm_read_bytes": access.hbm_read_bytes,
                "hbm_write_bytes": access.hbm_write_bytes,
                "l2_read_bytes": _optional_int(access.l2_read_bytes),
                "l2_write_bytes": _optional_int(access.l2_write_bytes),
                "sram_read_bytes": _optional_int(access.sram_read_bytes),
                "sram_write_bytes": _optional_int(access.sram_write_bytes),
                "register_read_bytes": _optional_int(access.register_read_bytes),
                "register_write_bytes": _optional_int(access.register_write_bytes),
                "transaction_l2_requested_bytes": transaction.get("l2_requested", 0),
                "transaction_dram_unique_bytes": transaction.get("dram_unique", 0),
                "transaction_dram_capacity_miss_bytes": transaction.get(
                    "dram_capacity_miss", 0
                ),
                "transaction_dram_total_bytes": transaction.get("dram_total", 0),
                "l2_capacity_bytes": transaction.get("l2_capacity", 0),
                "l2_reuse_working_set_bytes": transaction.get(
                    "l2_reuse_working_set", 0
                ),
                "l2_capacity_miss_fraction": transaction.get(
                    "l2_capacity_miss_fraction", 0
                ),
                "transaction_smem_read_bytes": transaction.get("smem_read", 0),
                "transaction_smem_write_bytes": transaction.get("smem_write", 0),
                "tile_efficiency": diagnostics.get("tile_efficiency", 0),
                "pareto_optimal": template.name in result.pareto_templates,
                "latency_optimal": template.name == result.latency_winner.template.name,
                "energy_optimal": template.name == result.energy_winner.template.name,
            }


def summary_rows(results: Sequence[ShapeResult]) -> Iterable[dict[str, object]]:
    success = study_succeeds(results)
    for result in results:
        latency = result.latency_winner
        energy = result.energy_winner
        yield {
            "shape_label": result.shape.label,
            "shape_family": result.shape.family,
            "edge_case": result.shape.edge_case,
            "op_kind": (
                OpKind.BATCHED_GEMM.value
                if result.shape.batched
                else OpKind.GEMM.value
            ),
            "batch": result.shape.batch,
            "m": result.shape.m,
            "n": result.shape.n,
            "k": result.shape.k,
            "candidate_count": len(result.candidates),
            "latency_template": latency.template.name,
            "energy_template": energy.template.name,
            "same_winner": result.same_winner,
            "latency_optimal_us": latency.profile.latency_s * 1.0e6,
            "latency_choice_energy_mj": latency.profile.energy_j * 1.0e3,
            "energy_choice_latency_us": energy.profile.latency_s * 1.0e6,
            "energy_optimal_mj": energy.profile.energy_j * 1.0e3,
            "energy_saving_percent": result.energy_saving_fraction * 100.0,
            "energy_regret_percent": result.energy_regret_fraction * 100.0,
            "latency_regret_percent": result.latency_regret_fraction * 100.0,
            "latency_hbm_bytes": hierarchy_traffic_bytes(latency, "hbm"),
            "energy_hbm_bytes": hierarchy_traffic_bytes(energy, "hbm"),
            "latency_l2_bytes": hierarchy_traffic_bytes(latency, "l2"),
            "energy_l2_bytes": hierarchy_traffic_bytes(energy, "l2"),
            "latency_sram_bytes": hierarchy_traffic_bytes(latency, "sram"),
            "energy_sram_bytes": hierarchy_traffic_bytes(energy, "sram"),
            "aligned_success_example": result.aligned_success_example,
            "strong_tradeoff_example": result.strong_tradeoff_example,
            "study_success": success,
        }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_coverage_csv(path: Path, coverage: Sequence[CoverageRow]) -> None:
    rows = (
        {
            "shape_label": row.shape.label,
            "m": row.shape.m,
            "n": row.shape.n,
            "k": row.shape.k,
            "candidate_count": row.candidate_count,
            "minimum_required": row.minimum_required,
            "passes": row.passes,
        }
        for row in coverage
    )
    write_csv(path, rows)


def _plotting_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for PNG output; install the opmodel research extra"
        ) from exc
    return plt


def write_plots(results: Sequence[ShapeResult], output_dir: Path, *, dpi: int) -> None:
    plt = _plotting_modules()
    _write_latency_energy_plot(plt, results, output_dir / "latency_energy.png", dpi)
    _write_regret_plot(plt, results, output_dir / "objective_regret.png", dpi)
    _write_traffic_plot(plt, results, output_dir / "winner_traffic.png", dpi)
    _write_energy_plot(plt, results, output_dir / "winner_energy_breakdown.png", dpi)
    _write_dynamic_energy_pair_plot(
        plt,
        results,
        output_dir / "winner_pair_dynamic_energy.png",
        dpi,
    )


def _shape_tick(result: ShapeResult) -> str:
    return f"{result.shape.label}\n{result.shape.dimensions}"


def _winner_tile(candidate: GemmCandidateEvaluation) -> str:
    template = candidate.template
    return f"{template.cta_m}x{template.cta_n}"


def _write_latency_energy_plot(plt, results, path: Path, dpi: int) -> None:
    columns = 3
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 4.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, result in zip(axes.flat, results):
        latencies = [candidate.profile.latency_s * 1.0e6 for candidate in result.candidates]
        energies = [candidate.profile.energy_j * 1.0e3 for candidate in result.candidates]
        axis.scatter(latencies, energies, color="#64748b", alpha=0.8, s=35, label="candidate")
        frontier = sorted(
            (
                candidate
                for candidate in result.candidates
                if candidate.template.name in result.pareto_templates
            ),
            key=lambda candidate: candidate.profile.latency_s,
        )
        axis.plot(
            [candidate.profile.latency_s * 1.0e6 for candidate in frontier],
            [candidate.profile.energy_j * 1.0e3 for candidate in frontier],
            color="#0f172a",
            linewidth=1.4,
            label="Pareto front",
        )
        latency = result.latency_winner
        energy = result.energy_winner
        axis.scatter(
            [latency.profile.latency_s * 1.0e6],
            [latency.profile.energy_j * 1.0e3],
            color="#dc2626",
            marker="*",
            s=150,
            zorder=4,
            label="latency-optimal",
        )
        axis.scatter(
            [energy.profile.latency_s * 1.0e6],
            [energy.profile.energy_j * 1.0e3],
            color="#059669",
            marker="X",
            s=90,
            zorder=5,
            label="energy-optimal",
        )
        axis.annotate(
            f"L: {_winner_tile(latency)}",
            (latency.profile.latency_s * 1.0e6, latency.profile.energy_j * 1.0e3),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
            color="#991b1b",
        )
        axis.annotate(
            f"E: {_winner_tile(energy)}",
            (energy.profile.latency_s * 1.0e6, energy.profile.energy_j * 1.0e3),
            xytext=(5, -13),
            textcoords="offset points",
            fontsize=8,
            color="#047857",
        )
        axis.set_title(
            f"{result.shape.label}: {result.shape.dimensions} ({len(result.candidates)} kernels)"
        )
        axis.set_xlabel("Latency (us)")
        axis.set_ylabel("Total energy (mJ)")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(results) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("A100 BF16 GEMM kernel strategies: latency versus total energy", fontsize=15)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _write_regret_plot(plt, results, path: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    positions = list(range(len(results)))
    width = 0.38
    axis.bar(
        [position - width / 2 for position in positions],
        [result.energy_saving_fraction * 100.0 for result in results],
        width,
        color="#dc2626",
        label="energy saved by accepting the energy-optimal kernel",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [result.latency_regret_fraction * 100.0 for result in results],
        width,
        color="#059669",
        label="extra latency of energy-optimal kernel",
    )
    axis.axhline(10.0, color="#991b1b", linestyle="--", linewidth=1, alpha=0.7)
    axis.set_xticks(positions, [_shape_tick(result) for result in results], rotation=30, ha="right")
    axis.set_ylabel("Change relative to latency-optimal kernel (%)")
    axis.set_title("Energy saved versus latency paid when changing kernel objective")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _write_traffic_plot(plt, results, path: Path, dpi: int) -> None:
    figure, axes = plt.subplots(
        3, 1, figsize=(15, 13), sharex=True, constrained_layout=True
    )
    positions = list(range(len(results)))
    width = 0.38
    colors = {"hbm": "#dc2626", "l2": "#2563eb", "sram": "#059669"}
    for axis, hierarchy in zip(axes, ("hbm", "l2", "sram")):
        axis.bar(
            [position - width / 2 for position in positions],
            [hierarchy_traffic_bytes(result.latency_winner, hierarchy) / MIB for result in results],
            width,
            color=colors[hierarchy],
            alpha=0.55,
            label="latency-optimal",
        )
        axis.bar(
            [position + width / 2 for position in positions],
            [hierarchy_traffic_bytes(result.energy_winner, hierarchy) / MIB for result in results],
            width,
            color=colors[hierarchy],
            alpha=0.95,
            hatch="//",
            label="energy-optimal",
        )
        axis.set_ylabel(f"{hierarchy.upper()} traffic (MiB)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
    for axis in axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xticks(
        positions,
        [_shape_tick(result) for result in results],
        rotation=30,
        ha="right",
    )
    figure.suptitle(
        "Hierarchy-specific modeled traffic (levels are intentionally not summed)",
        fontsize=15,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _write_energy_plot(plt, results, path: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    labels: list[str] = []
    candidates: list[GemmCandidateEvaluation] = []
    for result in results:
        labels.extend((f"{result.shape.label}\nlatency", f"{result.shape.label}\nenergy"))
        candidates.extend((result.latency_winner, result.energy_winner))
    positions = list(range(len(candidates)))
    bottoms = [0.0] * len(candidates)
    colors = {
        "compute": "#7c3aed",
        "hbm": "#dc2626",
        "l2": "#2563eb",
        "sram": "#059669",
        "register": "#d97706",
        "static": "#64748b",
    }
    for component in colors:
        values = [
            _energy_components(candidate)[component] * 1.0e3 for candidate in candidates
        ]
        axis.bar(
            positions,
            values,
            bottom=bottoms,
            color=colors[component],
            label=component,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Energy component (mJ)")
    axis.set_title("Latency-optimal versus energy-optimal kernel energy breakdown")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=6, fontsize=8)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _dynamic_energy_components(
    candidate: GemmCandidateEvaluation,
) -> tuple[float, float, float, float, float]:
    components = _energy_components(candidate)
    dynamic = max(0.0, candidate.profile.energy_j - components["static"])
    hbm = components["hbm"]
    smem = components["sram"]
    l2 = components["l2"]
    other = max(0.0, dynamic - hbm - smem - l2)
    return other, hbm, smem, l2, dynamic


def _write_dynamic_energy_pair_plot(plt, results, path: Path, dpi: int) -> None:
    compact_results = {
        result.shape.label: result
        for result in results
        if result.shape.label in LLM_DYNAMIC_PAIR_PLOT_LABELS
    }
    compact_layout = len(compact_results) == len(LLM_DYNAMIC_PAIR_PLOT_LABELS)
    if compact_layout:
        results = tuple(
            compact_results[label] for label in LLM_DYNAMIC_PAIR_PLOT_LABELS
        )
    columns = len(results) if compact_layout else 3
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=((7.5 if compact_layout else 12), (3.9 if compact_layout else 4.8) * rows),
        squeeze=False,
        constrained_layout=True,
    )
    energy_legend_added = False
    for plot_index, (axis, result) in enumerate(zip(axes.flat, results)):
        latency = result.latency_winner
        energy = result.energy_winner
        pair = (latency,) if result.same_winner else (latency, energy)
        x_values = [candidate.profile.latency_s * 1.0e6 for candidate in pair]
        components = [_dynamic_energy_components(candidate) for candidate in pair]
        dynamic_values = [component[4] * 1.0e3 for component in components]
        other_values = [component[0] * 1.0e3 for component in components]

        if len(x_values) == 1:
            x_span = max(1.0, x_values[0] * 0.04)
        else:
            x_span = max(abs(x_values[1] - x_values[0]), max(x_values) * 0.015, 1.0e-6)
        bar_width = 0.12 * x_span
        for candidate_index, (candidate, x_value, component) in enumerate(
            zip(pair, x_values, components)
        ):
            other, hbm, smem, l2, dynamic = (value * 1.0e3 for value in component)
            show_labels = plot_index == 0 and candidate_index == 0
            axis.bar(
                x_value,
                hbm,
                width=bar_width,
                bottom=other,
                color="#dc2626",
                alpha=0.9,
                label=("HBM" if compact_layout else "HBM dynamic energy")
                if show_labels
                else None,
                zorder=2,
            )
            axis.bar(
                x_value,
                smem,
                width=bar_width,
                bottom=other + hbm,
                color="#059669",
                alpha=0.9,
                label=("SMEM" if compact_layout else "SMEM dynamic energy")
                if show_labels
                else None,
                zorder=2,
            )
            axis.bar(
                x_value,
                l2,
                width=bar_width,
                bottom=other + hbm + smem,
                color="#2563eb",
                alpha=0.9,
                label=("L2" if compact_layout else "L2 dynamic energy")
                if show_labels
                else None,
                zorder=2,
            )
            marker = "*" if candidate is latency else "X"
            color = "#dc2626" if candidate is latency else "#059669"
            label = None
            if candidate is latency and plot_index == 0:
                label = "latency-opt." if compact_layout else "latency-optimal"
            elif candidate is energy and not energy_legend_added:
                label = "energy-opt." if compact_layout else "energy-optimal"
                energy_legend_added = True
            axis.scatter(
                [x_value],
                [dynamic],
                marker=marker,
                s=130 if candidate is latency else 85,
                color=color,
                edgecolors="#0f172a",
                linewidths=0.6,
                label=label,
                zorder=5,
            )

        latency_other, _, _, _, latency_dynamic = _dynamic_energy_components(latency)
        _, _, _, _, energy_dynamic = _dynamic_energy_components(energy)
        latency_us = latency.profile.latency_s * 1.0e6
        energy_us = energy.profile.latency_s * 1.0e6
        dynamic_saving = (
            1.0 - energy_dynamic / latency_dynamic if latency_dynamic > 0.0 else 0.0
        )
        latency_increase = (
            energy.profile.latency_s / latency.profile.latency_s - 1.0
            if latency.profile.latency_s > 0.0
            else 0.0
        )

        y_floor = min(latency_other * 1.0e3, *other_values)
        y_top = max(dynamic_values)
        y_span = max(y_top - y_floor, y_top * 0.04, 1.0e-9)
        axis.axhline(
            latency_other * 1.0e3,
            color="#64748b",
            linestyle=":",
            linewidth=0.8,
            alpha=0.75,
            label=(
                "latency baseline dynamic - HBM - L2 - SMEM"
                if plot_index == 0 and not compact_layout
                else None
            ),
            zorder=1,
        )
        if result.same_winner:
            axis.text(
                latency_us,
                y_top + 0.18 * y_span,
                (
                    "same optimum\n0% energy\n0% latency"
                    if compact_layout
                    else "same optimum\n0% energy / 0% latency"
                ),
                ha="center",
                va="bottom",
                fontsize=6 if compact_layout else 7,
                color="#334155",
            )
        else:
            axis.annotate(
                "",
                xy=(energy_us, energy_dynamic * 1.0e3),
                xytext=(latency_us, latency_dynamic * 1.0e3),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#0f172a",
                    "linewidth": 1.2,
                    "connectionstyle": "arc3,rad=-0.12",
                },
                zorder=4,
            )
            axis.text(
                (latency_us + energy_us) / 2.0,
                max(latency_dynamic, energy_dynamic) * 1.0e3 + 0.18 * y_span,
                f"-{dynamic_saving * 100.0:.1f}% energy\n+{latency_increase * 100.0:.1f}% latency",
                ha="center",
                va="bottom",
                fontsize=6 if compact_layout else 7,
                color="#0f172a",
            )

        x_center = sum(x_values) / len(x_values)
        x_half_range = max(x_span * 0.75, x_center * 0.012, 1.0e-6)
        axis.set_xlim(x_center - x_half_range, x_center + x_half_range)
        axis.set_ylim(y_floor - 0.04 * y_span, y_top + 0.36 * y_span)
        axis.set_title(
            f"{result.shape.label}\n{result.shape.dimensions}",
            fontsize=8 if compact_layout else 9,
        )
        if not compact_layout:
            axis.set_xlabel("Latency (us)", fontsize=8)
            axis.set_ylabel("Dynamic energy (mJ)", fontsize=8)
        axis.set_box_aspect(3.0 if compact_layout else 0.9)
        if compact_layout:
            axis.locator_params(axis="x", nbins=3)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.18, zorder=0)

    for axis in list(axes.flat)[len(results) :]:
        axis.set_visible(False)
    handles: list[object] = []
    labels: list[str] = []
    for axis in list(axes.flat)[: len(results)]:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985 if compact_layout else 0.925),
        ncol=5 if compact_layout else 6,
        fontsize=5.5 if compact_layout else 7.5,
        frameon=False,
    )
    if not compact_layout:
        figure.suptitle(
            "Latency- versus energy-optimal LLM GEMM kernels: "
            "static-subtracted dynamic energy",
            fontsize=14,
            y=0.992,
        )
    if compact_layout:
        figure.supxlabel("Latency (us)", fontsize=9, y=0.012)
        figure.supylabel("Dynamic energy (mJ)", fontsize=9, x=0.006)
        figure.get_layout_engine().set(
            rect=(0.012, 0.045, 0.995, 0.84),
            w_pad=0.002,
            h_pad=0.01,
            wspace=0.005,
        )
    else:
        figure.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _print_summary(results: Sequence[ShapeResult]) -> None:
    print(
        "shape                      kernels  energy_saved  latency_penalty  "
        "latency_template -> energy_template"
    )
    for result in results:
        print(
            f"{result.shape.label:<26} {len(result.candidates):>7}  "
            f"{result.energy_saving_fraction * 100.0:>11.3f}%  "
            f"{result.latency_regret_fraction * 100.0:>13.3f}%  "
            f"{result.latency_winner.template.name} -> "
            f"{result.energy_winner.template.name}"
        )
    verdict = "demonstrated" if study_succeeds(results) else "null_result"
    print(f"study_verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare effective-roofline latency and total energy across legal "
            "A100 BF16 GEMM kernel catalogue strategies."
        )
    )
    parser.add_argument("--hardware", default=str(DEFAULT_HARDWARE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--suite",
        choices=("locked", "llm"),
        default="locked",
        help="built-in shape suite used when no custom shapes are supplied",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="replace the locked matrix with repeatable LABEL:M:N:K shapes",
    )
    parser.add_argument(
        "--batched-shape",
        action="append",
        type=parse_batched_shape,
        help="replace the built-in suite with repeatable LABEL:BATCH:M:N:K shapes",
    )
    parser.add_argument("--min-candidates", type=int, default=12)
    parser.add_argument("--runtime-budget-s", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_candidates <= 0:
        raise ValueError("--min-candidates must be positive")
    if args.runtime_budget_s <= 0.0:
        raise ValueError("--runtime-budget-s must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    custom_shapes = tuple(args.shape or ()) + tuple(args.batched_shape or ())
    if custom_shapes:
        shapes = custom_shapes
    else:
        shapes = LLM_SHAPES if args.suite == "llm" else LOCKED_SHAPES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = load_hardware(args.hardware)

    try:
        results = evaluate_shapes(
            shapes,
            hardware,
            min_candidates=args.min_candidates,
            runtime_budget_s=args.runtime_budget_s,
        )
    except CoverageError as exc:
        coverage_path = output_dir / "coverage.csv"
        write_coverage_csv(coverage_path, exc.coverage)
        print(f"ERROR: {exc}")
        print(f"Wrote {coverage_path}")
        return 2
    except RuntimeBudgetError as exc:
        coverage_path = output_dir / "coverage.csv"
        write_coverage_csv(coverage_path, exc.coverage)
        print(f"ERROR: {exc}")
        print(f"Wrote partial {coverage_path}")
        return 3

    candidates_path = output_dir / "candidates.csv"
    summary_path = output_dir / "shape_summary.csv"
    write_csv(candidates_path, candidate_rows(results))
    write_csv(summary_path, summary_rows(results))
    try:
        write_plots(results, output_dir, dpi=args.dpi)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print(f"Wrote {candidates_path}")
        print(f"Wrote {summary_path}")
        return 4

    _print_summary(results)
    print(f"Wrote {candidates_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote PNG figures under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
