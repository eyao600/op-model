from __future__ import annotations

import json
import inspect
import math
from dataclasses import replace
from pathlib import Path

import pytest

from opmodel import DType, EngineKind, LocalOp, OpKind, Phase, TensorRole, TensorSpec
from opmodel.cli import profile_to_dict
from opmodel.hardware import load_hardware
from opmodel.models.effective_roofline import (
    EffectiveRooflineModel,
    _memory_window_summary,
    _nonzero_effective_rate,
    _service_from_effective_rate,
    _sm_occupancy_classes,
    _warps_by_smsp,
)
from opmodel.registry import create_model


ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "src/opmodel/configs/hardware/a10.yaml"


def _attrs(**updates: object) -> dict[str, object]:
    attrs: dict[str, object] = {
        "cta_tile_m": 128,
        "cta_tile_n": 128,
        "cta_tile_k": 32,
        "warp_tile_m": 64,
        "warp_tile_n": 64,
        "warp_tile_k": 32,
        "mma_m": 16,
        "mma_n": 8,
        "mma_k": 16,
        "pipeline_stages": 3,
        "warps_per_cta": 4,
        "threads_per_cta": 128,
        "registers_per_thread": 64,
    }
    attrs.update(updates)
    return attrs


def _gemm(
    m: int = 128,
    n: int = 128,
    k: int = 128,
    *,
    attrs: dict[str, object] | None = None,
    dtype: DType = DType.BF16,
) -> LocalOp:
    return LocalOp(
        name="gemm",
        kind=OpKind.GEMM,
        phase=Phase.INFERENCE,
        tensors=(
            TensorSpec(TensorRole.INPUT, (m, k), dtype),
            TensorSpec(TensorRole.WEIGHT, (k, n), dtype),
            TensorSpec(TensorRole.OUTPUT, (m, n), dtype),
        ),
        attrs=_attrs() if attrs is None else attrs,
    )


def _batched_gemm() -> LocalOp:
    return LocalOp(
        name="bmm",
        kind=OpKind.BATCHED_GEMM,
        phase=Phase.INFERENCE,
        tensors=(
            TensorSpec(TensorRole.INPUT, (3, 64, 96), DType.BF16),
            TensorSpec(TensorRole.WEIGHT, (3, 96, 80), DType.BF16),
            TensorSpec(TensorRole.OUTPUT, (3, 64, 80), DType.BF16),
        ),
        attrs=_attrs(cta_tile_m=64, cta_tile_n=64),
    )


def test_effective_rate_selects_latency_and_peak_ceilings() -> None:
    assert _nonzero_effective_rate(100.0, 20.0, 4.0) == 5.0
    assert _nonzero_effective_rate(100.0, 1000.0, 4.0) == 100.0
    assert _nonzero_effective_rate(100.0, 0.0, 4.0) == 0.0


def test_effective_rate_matches_latency_floor() -> None:
    for peak, work, latency in ((100.0, 20.0, 4.0), (100.0, 1000.0, 4.0)):
        rate = _nonzero_effective_rate(peak, work, latency)
        assert math.isclose(
            _service_from_effective_rate(work, rate),
            max(work / peak, latency),
        )


def test_exact_smsps_warp_distribution_conserves_nonmultiple_of_four() -> None:
    assert _warps_by_smsp(7) == (2, 2, 2, 1)
    assert sum(_warps_by_smsp(13)) == 13
    assert _warps_by_smsp(0) == (0, 0, 0, 0)


def test_occupancy_classes_conserve_ctas_and_reject_mismatch() -> None:
    classes = _sm_occupancy_classes(
        active_ctas=11,
        busy_sms=3,
        lazy_sms=1,
        busy_ctas_per_sm=3,
        lazy_ctas_per_sm=2,
    )
    assert [(item.name, item.sm_count, item.ctas_per_sm) for item in classes] == [
        ("busy", 3, 3),
        ("lazy", 1, 2),
    ]
    with pytest.raises(ValueError, match="does not conserve active CTAs"):
        _sm_occupancy_classes(
            active_ctas=12,
            busy_sms=3,
            lazy_sms=1,
            busy_ctas_per_sm=3,
            lazy_ctas_per_sm=2,
        )


def test_memory_windows_conserve_partial_tail_and_subtract_prologue() -> None:
    summary = _memory_window_summary(
        total_stage_equivalents=5.5,
        pipeline_stages=2,
        active_ctas=3,
        bytes_per_cta_stage=64.0,
        peak_bytes_per_cycle=32.0,
        latency_cycles=20.0,
    )
    assert summary.window_count == 3
    assert summary.full_window_count == 2
    assert summary.final_window_stage_equivalents == 1.5
    assert math.isclose(
        summary.full_window_count * summary.full_window_bytes
        + summary.final_window_bytes,
        summary.total_bytes,
    )
    assert math.isclose(
        summary.first_window_cycles + summary.remaining_cycles,
        summary.total_cycles,
    )
    assert summary.effective_bytes_per_cycle <= summary.raw_bytes_per_cycle


def test_zero_byte_memory_window_has_no_latency_charge() -> None:
    summary = _memory_window_summary(
        total_stage_equivalents=4.0,
        pipeline_stages=2,
        active_ctas=8,
        bytes_per_cta_stage=0.0,
        peak_bytes_per_cycle=32.0,
        latency_cycles=200.0,
    )
    assert summary.total_cycles == 0.0
    assert summary.first_window_cycles == 0.0


def test_registry_gemm_batched_and_json_safe_diagnostics() -> None:
    hardware = load_hardware(HARDWARE)
    model = create_model("effective_roofline")
    assert isinstance(model, EffectiveRooflineModel)
    gemm = model.predict(_gemm(), hardware)
    batched = model.predict(_batched_gemm(), hardware)
    assert gemm.implementation == "effective_roofline.gemm"
    assert batched.implementation == "effective_roofline.batched_gemm"
    assert gemm.engine == EngineKind.TENSOR
    assert batched.flops == 2 * 3 * 64 * 80 * 96
    json.dumps(profile_to_dict(gemm))


def test_effective_model_never_calls_detailed_wave_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import opmodel.models.extended_roofline as extended

    def fail(**_: object) -> None:
        raise AssertionError("detailed wave pipeline must not run")

    monkeypatch.setattr(extended, "_wave_pipeline", fail)
    EffectiveRooflineModel().predict(_gemm(), load_hardware(HARDWARE))


def test_effective_wave_predictor_matches_pipeline_input_contract() -> None:
    from opmodel.models.effective_roofline import _wave_effective_roofline
    from opmodel.models.extended_roofline import _wave_pipeline

    assert tuple(inspect.signature(_wave_effective_roofline).parameters) == tuple(
        inspect.signature(_wave_pipeline).parameters
    )


def test_wave_diagnostics_conserve_occupancy_and_bound_rates() -> None:
    profile = EffectiveRooflineModel().predict(
        _gemm(384, 256, 70), load_hardware(HARDWARE)
    )
    diagnostics = profile.diagnostics
    assert diagnostics["useful_flops"] <= diagnostics["issued_flops"]
    for wave in diagnostics["wave_roofline"].values():
        if wave["active_ctas"] == 0:
            continue
        represented = sum(
            item["sm_count"] * item["ctas_per_sm"]
            for item in wave["occupancy_classes"]
        )
        assert represented == wave["active_ctas"]
        for resource, raw_name, effective_name in (
            ("tensor", "tc_raw_flops_per_cycle", "tc_effective_flops_per_cycle"),
            ("smem", "smem_raw_bytes_per_cycle", "smem_effective_bytes_per_cycle"),
            ("l2", "l2_raw_bytes_per_cycle", "l2_effective_bytes_per_cycle"),
            ("hbm", "dram_raw_bytes_per_cycle", "dram_effective_bytes_per_cycle"),
        ):
            assert wave[effective_name] <= wave[raw_name] + 1.0e-9, resource


def test_underfilled_tail_is_evaluated_with_lower_concurrency() -> None:
    attrs = _attrs(
        cta_tile_m=64,
        cta_tile_n=64,
        warp_tile_m=32,
        warp_tile_n=32,
        resident_ctas_per_sm=2,
    )
    profile = EffectiveRooflineModel().predict(
        _gemm(64 * 145, 64, 32, attrs=attrs), load_hardware(HARDWARE)
    )
    waves = profile.diagnostics["wave_roofline"]
    assert waves["last"]["active_ctas"] < waves["full"]["active_ctas"]
    assert (
        waves["last"]["tc_effective_flops_per_cycle"]
        < waves["full"]["tc_effective_flops_per_cycle"]
    )
    assert (
        waves["last"]["dram_effective_bytes_per_cycle"]
        <= waves["full"]["dram_effective_bytes_per_cycle"]
    )


def test_fixed_overhead_energy_and_non_tensor_fallback() -> None:
    hardware = load_hardware(HARDWARE)
    model = EffectiveRooflineModel()
    baseline = model.predict(_gemm(), replace(hardware, compute=replace(
        hardware.compute, device_fixed_overhead_cycles=0
    )))
    overhead_hardware = replace(hardware, compute=replace(
        hardware.compute, device_fixed_overhead_cycles=1234
    ))
    with_overhead = model.predict(_gemm(), overhead_hardware)
    assert math.isclose(
        with_overhead.latency_s - baseline.latency_s,
        1234 / with_overhead.diagnostics["clock_hz"],
    )
    assert with_overhead.energy_j >= 0.0

    fallback_hardware = replace(
        hardware,
        compute=replace(
            hardware.compute,
            tensor_flops_per_s={DType.FP16: hardware.compute.tensor_flops_per_s[DType.FP16]},
        ),
    )
    fallback = model.predict(_gemm(), fallback_hardware)
    assert fallback.engine == EngineKind.VECTOR
    assert fallback.implementation == "effective_roofline.gemm"


def test_automatic_selection_uses_effective_backend() -> None:
    profile = EffectiveRooflineModel().predict(
        _gemm(attrs={"gemm_selection_shortlist_size": 2}),
        load_hardware(HARDWARE),
    )
    selection = profile.diagnostics["gemm_selection"]
    assert selection["enabled"]
    assert selection["backend"] == "effective_roofline"
    assert selection["shortlist_size"] == 2

    with pytest.raises(ValueError, match="must be effective_roofline"):
        EffectiveRooflineModel().predict(
            _gemm(attrs={"gemm_selection_backend": "extended_roofline"}),
            load_hardware(HARDWARE),
        )


def test_slice_k_epilogue_adds_only_epilogue_smem_service() -> None:
    hardware = load_hardware(HARDWARE)
    ordinary = EffectiveRooflineModel().predict(_gemm(), hardware)
    sliced = EffectiveRooflineModel().predict(
        _gemm(
            attrs=_attrs(
                slice_k=True,
                num_warp_tile_k=2,
                warps_per_cta=8,
                threads_per_cta=256,
            )
        ),
        hardware,
    )
    ordinary_wave = ordinary.diagnostics["wave_roofline"]["last"]
    sliced_wave = sliced.diagnostics["wave_roofline"]["last"]
    assert sliced_wave["epilogue"]["slice_k_extra_cycles"] > 0.0
    assert sliced_wave["epilogue"]["smem_bytes"] > ordinary_wave["epilogue"]["smem_bytes"]
