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
    LocalMatmulGeometry,
    SMOccupancyClass,
    _ampere_flash_attention_kernel,
    _flash_attention_ctas,
    _local_mini_gemm_service,
    _memory_window_summary,
    _nonzero_effective_rate,
    _parse_flash_attention_problem,
    _service_from_effective_rate,
    _sm_occupancy_classes,
    _warps_by_smsp,
    evaluate_gemm_template_candidates,
)
from opmodel.registry import create_model


ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "src/opmodel/configs/hardware/a10.yaml"
A100_HARDWARE = ROOT / "src/opmodel/configs/hardware/a100_40gb_pcie.yaml"


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


def _effective_softmax(*, selector: bool = True, reduction: int = 512) -> LocalOp:
    attrs: dict[str, object] = {"row_size": reduction}
    if selector:
        attrs["softmax_kernel"] = "effective_cuda_ampere"
    shape = (128, reduction)
    return LocalOp(
        "softmax",
        OpKind.SOFTMAX,
        Phase.INFERENCE,
        (
            TensorSpec(TensorRole.INPUT, shape, DType.BF16),
            TensorSpec(TensorRole.OUTPUT, shape, DType.BF16),
        ),
        attrs,
    )


def _flash_attention(*, reuse: str = "ideal_within_wave") -> LocalOp:
    q_shape = (1, 4, 192, 64)
    kv_shape = (1, 2, 192, 64)
    return LocalOp(
        "flash",
        OpKind.ATTENTION_PREFILL,
        Phase.PREFILL,
        (
            TensorSpec(TensorRole.INPUT, q_shape, DType.BF16, "q"),
            TensorSpec(TensorRole.INPUT, kv_shape, DType.BF16, "k"),
            TensorSpec(TensorRole.INPUT, kv_shape, DType.BF16, "v"),
            TensorSpec(TensorRole.OUTPUT, q_shape, DType.BF16),
        ),
        {
            "attention_kernel": "flash_attention_ampere",
            "causal": True,
            "kv_heads": 2,
            "block_q": 128,
            "block_k": 128,
            "warps_per_cta": 4,
            "max_concurrent_ctas_per_sm": 2,
            "kv_reuse_policy": reuse,
        },
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


def test_memory_windows_use_persistent_concurrency_limited_rate() -> None:
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
    expected_first_bytes = 3 * 2 * 64.0
    expected_first_rate = expected_first_bytes / 20.0
    assert summary.first_window_bytes == expected_first_bytes
    assert math.isclose(summary.first_window_cycles, 20.0)
    assert math.isclose(summary.remaining_cycles, summary.remaining_bytes / expected_first_rate)
    assert math.isclose(
        summary.total_cycles,
        summary.first_window_cycles + summary.remaining_cycles,
    )
    assert summary.full_window_effective_bytes_per_cycle == expected_first_rate
    assert summary.latency_bearing_window_count == summary.window_count
    assert summary.steady_state_bytes_per_cycle == expected_first_rate
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


def test_memory_windows_use_peak_bandwidth_when_first_window_saturates() -> None:
    summary = _memory_window_summary(
        total_stage_equivalents=6.5,
        pipeline_stages=4,
        active_ctas=8,
        bytes_per_cta_stage=64.0,
        peak_bytes_per_cycle=32.0,
        latency_cycles=10.0,
    )
    assert summary.effective_bytes_per_cycle == 32.0
    assert math.isclose(summary.total_cycles, summary.total_bytes / 32.0)
    assert math.isclose(
        summary.remaining_cycles,
        summary.remaining_bytes / 32.0,
    )
    assert summary.full_window_effective_bytes_per_cycle == 32.0
    assert summary.final_window_effective_bytes_per_cycle == 32.0
    assert summary.latency_bearing_window_count == summary.window_count
    assert summary.steady_state_bytes_per_cycle == 32.0


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
    assert summary.latency_bearing_window_count == 0
    assert summary.steady_state_bytes_per_cycle == 32.0


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
        for item in wave["occupancy_classes"]:
            remaining_bytes = max(0, wave["k_groups"] - 1) * item[
                "smem_group_bytes_per_sm"
            ]
            assert math.isclose(
                item["smem_steady_remaining_cycles"],
                remaining_bytes / item["smem_effective_bytes_per_cycle_per_sm"],
            )
            assert math.isclose(
                item["smem_cycles"],
                item["smem_first_group_cycles"]
                + item["smem_steady_remaining_cycles"],
            )


@pytest.mark.parametrize(
    ("shared_latency_s", "latency_bound"),
    ((1.0e-3, True), (1.0e-12, False)),
)
def test_smem_uses_persistent_concurrency_limited_rate(
    shared_latency_s: float,
    latency_bound: bool,
) -> None:
    hardware = load_hardware(HARDWARE)
    memory_levels = dict(hardware.memory_levels)
    memory_levels["sram"] = replace(
        memory_levels["sram"], latency_s=shared_latency_s
    )
    hardware = replace(hardware, memory_levels=memory_levels)
    profile = EffectiveRooflineModel().predict(_gemm(384, 256, 70), hardware)
    shared_latency_cycles = shared_latency_s * profile.diagnostics["clock_hz"]

    for wave in profile.diagnostics["wave_roofline"].values():
        if wave["active_ctas"] == 0:
            continue
        for item in wave["occupancy_classes"]:
            group_bytes = item["smem_group_bytes_per_sm"]
            expected_rate = min(
                item["smem_raw_bytes_per_cycle_per_sm"],
                group_bytes / shared_latency_cycles,
            )
            assert (expected_rate < item["smem_raw_bytes_per_cycle_per_sm"]) == (
                latency_bound
            )
            assert math.isclose(
                item["smem_first_group_cycles"],
                group_bytes / expected_rate,
            )
            expected_remaining = (
                max(0, wave["k_groups"] - 1) * group_bytes / expected_rate
            )
            assert math.isclose(
                item["smem_steady_remaining_cycles"], expected_remaining
            )
            assert math.isclose(
                item["smem_cycles"],
                item["smem_first_group_cycles"] + expected_remaining,
            )


def test_softmax_effective_selector_is_opt_in_and_conserves_traffic() -> None:
    hardware = load_hardware(A100_HARDWARE)
    model = EffectiveRooflineModel()
    legacy = model.predict(_effective_softmax(selector=False), hardware)
    effective = model.predict(_effective_softmax(), hardware)
    assert legacy.implementation == "roofline.softmax"
    assert effective.implementation == "effective_roofline.softmax_cuda_ampere"
    assert effective.memory_access.hbm_read_bytes == 128 * 512 * 2
    assert effective.memory_access.hbm_write_bytes == 128 * 512 * 2
    assert effective.diagnostics["residency_strategy"] == "register_resident"
    assert effective.diagnostics["transaction_bytes"]["smem_value_staging"] == 0
    assert effective.diagnostics["cuda_softmax_compute"]["exp_ops"] > 0


def test_softmax_rejects_rows_outside_one_cta_template() -> None:
    with pytest.raises(ValueError, match="up to 4096"):
        EffectiveRooflineModel().predict(
            _effective_softmax(reduction=8192), load_hardware(A100_HARDWARE)
        )


def test_causal_flashattention_conserves_grid_gqa_and_pairs() -> None:
    op = _flash_attention()
    problem = _parse_flash_attention_problem(op)
    kernel = _ampere_flash_attention_kernel(problem, op.attrs)
    ctas = _flash_attention_ctas(problem, kernel)
    assert len(ctas) == 8
    assert {cta.kv_head for cta in ctas if cta.query_head < 2} == {0}
    assert {cta.kv_head for cta in ctas if cta.query_head >= 2} == {1}
    useful_pairs = sum(
        tile.useful_score_elements for cta in ctas for tile in cta.kv_tiles
    )
    assert useful_pairs == 4 * 192 * 193 // 2
    assert all(
        tile.useful_score_elements <= tile.issued_score_elements
        for cta in ctas
        for tile in cta.kv_tiles
    )


def test_flashattention_pipeline_has_no_score_materialization() -> None:
    profile = EffectiveRooflineModel().predict(
        _flash_attention(), load_hardware(A100_HARDWARE)
    )
    assert profile.implementation == "effective_roofline.flash_attention_ampere_causal"
    traffic = profile.diagnostics["transaction_bytes"]
    assert traffic["score_l2"] == traffic["score_hbm"] == 0
    assert traffic["probability_l2"] == traffic["probability_hbm"] == 0
    for wave in profile.diagnostics["waves"]:
        active = wave["active_ctas_by_iteration"]
        assert all(later <= earlier for earlier, later in zip(active, active[1:]))
        assert wave["total_cycles"] == pytest.approx(
            wave["prologue_cycles"] + wave["body_cycles"] + wave["epilogue_cycles"]
        )
        for iteration in wave["iterations"]:
            assert iteration["total_cycles"] == pytest.approx(
                iteration["qk_softmax_or_v_cycles"]
                + iteration["pv_or_next_k_cycles"]
            )


def test_flashattention_reuse_reduces_hbm_not_l2_requests() -> None:
    hardware = load_hardware(A100_HARDWARE)
    model = EffectiveRooflineModel()
    none = model.predict(_flash_attention(reuse="none"), hardware)
    reused = model.predict(_flash_attention(reuse="ideal_within_wave"), hardware)
    none_bytes = none.diagnostics["transaction_bytes"]
    reused_bytes = reused.diagnostics["transaction_bytes"]
    assert none_bytes["k_l2_requested"] == reused_bytes["k_l2_requested"]
    assert none_bytes["v_l2_requested"] == reused_bytes["v_l2_requested"]
    assert reused_bytes["k_hbm"] < none_bytes["k_hbm"]
    assert reused_bytes["v_hbm"] < none_bytes["v_hbm"]
    assert (
        reused_bytes["k_hbm_unique_lower_bound"]
        <= reused_bytes["k_hbm"]
        <= reused_bytes["k_hbm_requested_upper_bound"]
    )
    assert (
        reused_bytes["v_hbm_unique_lower_bound"]
        <= reused_bytes["v_hbm"]
        <= reused_bytes["v_hbm_requested_upper_bound"]
    )


def test_local_mini_gemm_residency_controls_smem_operand_traffic() -> None:
    common = dict(
        cta_m=64,
        cta_n=64,
        reduction_k=64,
        live_m=64,
        live_n=64,
        live_k=64,
        warp_m=32,
        warp_n=32,
        warp_k=16,
        mma_m=16,
        mma_n=8,
        mma_k=16,
        warps_per_cta=4,
        rhs_smem_resident=True,
        input_bytes=2.0,
    )
    kwargs = dict(
        occupancy_classes=(SMOccupancyClass("busy", 1, 1),),
        represented_sms=1,
        peak_tensor_flops_per_cycle=1024.0,
        peak_smem_bytes_per_cycle=128.0,
        tensor_latency_cycles=26.0,
        shared_latency_cycles=29.0,
    )
    both = _local_mini_gemm_service(
        geometry=LocalMatmulGeometry(lhs_smem_resident=True, **common), **kwargs
    )
    rhs_only = _local_mini_gemm_service(
        geometry=LocalMatmulGeometry(lhs_smem_resident=False, **common), **kwargs
    )
    assert rhs_only.issued_flops == both.issued_flops
    assert rhs_only.smem_read_bytes < both.smem_read_bytes


def test_standalone_softmax_smem_schedule_conserves_value_staging() -> None:
    profile = EffectiveRooflineModel().predict(
        _effective_softmax(reduction=2048), load_hardware(A100_HARDWARE)
    )
    input_bytes = 128 * 2048 * 2
    assert profile.diagnostics["residency_strategy"] == "smem_staged"
    assert profile.diagnostics["transaction_bytes"]["smem_value_staging"] == 3 * input_bytes
    reduction = profile.diagnostics["transaction_bytes"]["smem_reduction"]
    assert (
        profile.memory_access.sram_read_bytes
        + profile.memory_access.sram_write_bytes
        - reduction
        == 3 * input_bytes
    )


def test_flashattention_rejects_noncausal_and_missing_occupancy() -> None:
    hardware = load_hardware(A100_HARDWARE)
    op = _flash_attention()
    with pytest.raises(ValueError, match="requires causal=True"):
        EffectiveRooflineModel().predict(
            replace(op, attrs={**op.attrs, "causal": False}), hardware
        )
    attrs = dict(op.attrs)
    attrs.pop("max_concurrent_ctas_per_sm")
    with pytest.raises(ValueError, match="requires max_concurrent"):
        EffectiveRooflineModel().predict(replace(op, attrs=attrs), hardware)


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


def test_tensor_gemm_energy_charges_padded_mma_events() -> None:
    hardware = load_hardware(HARDWARE)
    profile = EffectiveRooflineModel().predict(_gemm(1, 128, 128), hardware)
    diagnostics = profile.diagnostics
    event_flops = diagnostics["compute_event_flops"]

    assert event_flops == diagnostics["issued_flops"]
    assert event_flops > diagnostics["useful_flops"]
    assert diagnostics["padded_mma_event_flops"] == (
        event_flops - diagnostics["useful_flops"]
    )
    assert profile.energy_breakdown.compute_j == pytest.approx(
        event_flops * hardware.compute.tensor_energy_j_per_flop[DType.BF16]
    )


def test_catalogue_smem_legality_uses_k_invariant_resident_footprint() -> None:
    hardware = load_hardware(A100_HARDWARE)
    shallow = evaluate_gemm_template_candidates(
        _gemm(256, 256, 64, attrs={}), hardware, shortlist_size=999
    )
    deep = evaluate_gemm_template_candidates(
        _gemm(256, 256, 4096, attrs={}), hardware, shortlist_size=999
    )

    assert len(shallow) == len(deep) == 12
    template_name = "sm80_128x128x32_64x64x32_4w3s"
    shallow_candidate = next(
        candidate for candidate in shallow if candidate.template.name == template_name
    )
    deep_candidate = next(
        candidate for candidate in deep if candidate.template.name == template_name
    )

    shallow_kernel = shallow_candidate.profile.diagnostics["kernel"]
    deep_kernel = deep_candidate.profile.diagnostics["kernel"]
    assert shallow_kernel["shared_memory_bytes_per_cta"] == 49_152
    assert deep_kernel["shared_memory_bytes_per_cta"] == 49_152
    assert (
        deep_candidate.profile.memory_access.sram_read_bytes
        > shallow_candidate.profile.memory_access.sram_read_bytes
    )


def test_gemm_energy_charges_output_store_at_l2_and_hbm() -> None:
    hardware = load_hardware(HARDWARE)
    profile = create_model("effective_roofline").predict(_gemm(65, 33, 64), hardware)

    store_bytes = profile.diagnostics["transaction_bytes"]["d_store"]
    assert store_bytes > 65 * 33 * 2
    assert profile.memory_access.l2_write_bytes == store_bytes
    assert profile.memory_access.hbm_write_bytes == store_bytes


def test_gemm_energy_charges_modeled_epilogue_smem_traffic() -> None:
    hardware = load_hardware(HARDWARE)
    profile = create_model("effective_roofline").predict(_gemm(65, 33, 64), hardware)

    epilogue_bytes = profile.diagnostics["transaction_bytes"]["epilogue_smem"]
    assert epilogue_bytes > 0
    assert profile.memory_access.sram_write_bytes == epilogue_bytes


def test_scalar_gemm_energy_does_not_charge_tensor_tile_padding() -> None:
    hardware = load_hardware(HARDWARE)
    profile = EffectiveRooflineModel().predict(
        _gemm(
            1,
            128,
            128,
            attrs=_attrs(mma_m=1, mma_n=1, mma_k=1),
        ),
        hardware,
    )

    assert profile.diagnostics["compute_event_flops"] == profile.flops
    assert profile.diagnostics["padded_mma_event_flops"] == 0.0


def test_automatic_selection_uses_effective_backend() -> None:
    profile = EffectiveRooflineModel().predict(
        _gemm(attrs={"gemm_selection_shortlist_size": 2}),
        load_hardware(HARDWARE),
    )
    selection = profile.diagnostics["gemm_selection"]
    assert selection["enabled"]
    assert selection["backend"] == "effective_roofline"
    assert selection["shortlist_size"] == 2
    assert selection["selection_energy_j"] == selection["selected_energy_j"]

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
