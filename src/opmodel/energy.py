from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from opmodel.api import EnergyBreakdown, OpProfile
from opmodel.hardware import EnergyModelPowerCoefficients, EnergyModelSpec, HardwareSpec


E3_EVENT_FEATURE_ORDER = (
    "flops_total",
    "bytes_dram_total",
    "bytes_l2_total",
    "bytes_smem_total",
)
E3_POWER_FEATURE_ORDER = (
    "time_kernel_s",
    "time_sm_resident_s",
    "time_tc_active_s",
    "time_dram_active_s",
    "time_dram_exposed_s",
    "time_l2_active_s",
    "time_smem_active_s",
)
E3_FEATURE_ORDER = E3_EVENT_FEATURE_ORDER + E3_POWER_FEATURE_ORDER


@dataclass(frozen=True)
class EnergyFeatureRow:
    flops_total: float
    bytes_dram_total: float
    bytes_l2_total: float
    bytes_smem_total: float
    time_kernel_s: float
    time_sm_resident_s: float
    time_tc_active_s: float
    time_dram_active_s: float
    time_dram_exposed_s: float
    time_l2_active_s: float
    time_smem_active_s: float

    def power_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in E3_POWER_FEATURE_ORDER)


@dataclass(frozen=True)
class EnergyPrediction:
    event_j: float
    baseline_j: float
    residency_j: float
    tc_active_j: float
    dram_active_j: float
    dram_exposed_j: float
    l2_active_j: float
    smem_active_j: float

    @property
    def active_state_j(self) -> float:
        return (
            self.tc_active_j
            + self.dram_active_j
            + self.dram_exposed_j
            + self.l2_active_j
            + self.smem_active_j
        )

    @property
    def non_event_j(self) -> float:
        return self.baseline_j + self.residency_j + self.active_state_j

    @property
    def e0_j(self) -> float:
        return self.event_j

    @property
    def e1_j(self) -> float:
        return self.event_j + self.baseline_j

    @property
    def e2_j(self) -> float:
        return self.e1_j + self.residency_j

    @property
    def e3_j(self) -> float:
        return self.e2_j + self.active_state_j

    def total_for_level(self, model_level: str) -> float:
        level = model_level.upper()
        if level == "E0":
            return self.e0_j
        if level == "E1":
            return self.e1_j
        if level == "E2":
            return self.e2_j
        return self.e3_j

    def to_diagnostics(
        self,
        *,
        features: EnergyFeatureRow,
        energy_model: EnergyModelSpec,
    ) -> dict[str, Any]:
        return {
            "model_level": energy_model.model_level,
            "calibrated": True,
            "feature_order": E3_FEATURE_ORDER,
            "power_feature_order": E3_POWER_FEATURE_ORDER,
            "features": asdict(features),
            "power_coefficients": asdict(energy_model.power_coefficients),
            "fixed_event_energy_j": self.event_j,
            "predicted_residual_energy_j": self.non_event_j,
            "energy_by_level_j": {
                "E0": self.e0_j,
                "E1": self.e1_j,
                "E2": self.e2_j,
                "E3": self.e3_j,
            },
            "term_energy_j": {
                "event": self.event_j,
                "baseline": self.baseline_j,
                "residency": self.residency_j,
                "tc_active": self.tc_active_j,
                "dram_active": self.dram_active_j,
                "dram_exposed": self.dram_exposed_j,
                "l2_active": self.l2_active_j,
                "smem_active": self.smem_active_j,
                "active_state": self.active_state_j,
            },
            "calibration": dict(energy_model.calibration),
        }


def fixed_event_energy_j(profile: OpProfile) -> float:
    breakdown = profile.energy_breakdown
    return (
        breakdown.compute_j
        + breakdown.hbm_j
        + breakdown.l2_j
        + breakdown.sram_j
        + breakdown.register_j
    )


def extract_gemm_e3_features(profile: OpProfile) -> EnergyFeatureRow:
    diagnostics = profile.diagnostics
    clock_hz = float(diagnostics.get("clock_hz") or 0.0)
    if clock_hz <= 0.0:
        raise ValueError("E3 energy features require diagnostics.clock_hz")

    memory_access = profile.memory_access
    transaction_bytes = _mapping_value(diagnostics, "transaction_bytes")
    time_tc_active_s = _active_time_s(diagnostics, "compute", clock_hz)
    time_dram_active_s = _active_time_s(diagnostics, "dram", clock_hz)
    return EnergyFeatureRow(
        flops_total=float(profile.flops),
        bytes_dram_total=float(
            _total_bytes(memory_access.hbm_read_bytes, memory_access.hbm_write_bytes) or 0
        ),
        bytes_l2_total=float(
            _total_bytes(memory_access.l2_read_bytes, memory_access.l2_write_bytes) or 0
        ),
        bytes_smem_total=float(
            _total_bytes(
                transaction_bytes.get("smem_read"),
                transaction_bytes.get("smem_write"),
            )
            or 0
        )
        + float(transaction_bytes.get("epilogue_smem") or 0.0),
        time_kernel_s=float(profile.latency_s),
        time_sm_resident_s=_sm_resident_time_s(diagnostics, clock_hz),
        time_tc_active_s=time_tc_active_s,
        time_dram_active_s=time_dram_active_s,
        time_dram_exposed_s=max(0.0, time_dram_active_s - time_tc_active_s),
        time_l2_active_s=_active_time_s(diagnostics, "l2", clock_hz),
        time_smem_active_s=_active_time_s(diagnostics, "smem", clock_hz),
    )


def predict_e3_energy(
    *,
    features: EnergyFeatureRow,
    event_energy_j: float,
    power_coefficients: EnergyModelPowerCoefficients,
) -> EnergyPrediction:
    return EnergyPrediction(
        event_j=event_energy_j,
        baseline_j=features.time_kernel_s * power_coefficients.base_power_w,
        residency_j=features.time_sm_resident_s
        * power_coefficients.sm_resident_power_w,
        tc_active_j=features.time_tc_active_s * power_coefficients.tc_active_power_w,
        dram_active_j=features.time_dram_active_s
        * power_coefficients.dram_active_power_w,
        dram_exposed_j=features.time_dram_exposed_s
        * power_coefficients.dram_exposed_power_w,
        l2_active_j=features.time_l2_active_s * power_coefficients.l2_active_power_w,
        smem_active_j=features.time_smem_active_s
        * power_coefficients.smem_active_power_w,
    )


def apply_calibrated_energy_model(profile: OpProfile, hardware: HardwareSpec) -> OpProfile:
    energy_model = hardware.energy_model
    if energy_model is None:
        return profile

    features = extract_gemm_e3_features(profile)
    prediction = predict_e3_energy(
        features=features,
        event_energy_j=fixed_event_energy_j(profile),
        power_coefficients=energy_model.power_coefficients,
    )
    chosen_total = prediction.total_for_level(energy_model.model_level)
    diagnostics = dict(profile.diagnostics)
    diagnostics["energy_model"] = prediction.to_diagnostics(
        features=features,
        energy_model=energy_model,
    )
    return replace(
        profile,
        energy_j=chosen_total,
        energy_breakdown=_with_non_event_energy(
            profile.energy_breakdown,
            max(0.0, chosen_total - prediction.event_j),
        ),
        diagnostics=diagnostics,
    )


def _with_non_event_energy(
    breakdown: EnergyBreakdown, non_event_energy_j: float
) -> EnergyBreakdown:
    return replace(breakdown, static_j=non_event_energy_j)


def _active_time_s(
    diagnostics: Mapping[str, Any], resource: str, clock_hz: float
) -> float:
    active_cycles = _mapping_value(diagnostics, "active_cycles")
    return float(active_cycles.get(resource) or 0.0) / clock_hz


def _sm_resident_time_s(diagnostics: Mapping[str, Any], clock_hz: float) -> float:
    wave_count = int(diagnostics.get("cta_waves") or 0)
    if wave_count <= 0:
        return 0.0

    wave_pipeline = _mapping_value(diagnostics, "wave_pipeline")
    full_wave = _mapping_value(wave_pipeline, "full")
    last_wave = _mapping_value(wave_pipeline, "last")
    full_wave_count = max(0, wave_count - 1)
    full_wave_time_s = float(full_wave.get("total_cycles") or 0.0) / clock_hz
    last_wave_time_s = float(last_wave.get("total_cycles") or 0.0) / clock_hz

    resident_ctas_per_sm = float(diagnostics.get("resident_ctas_per_sm") or 0.0)
    ctas_per_wave = float(diagnostics.get("ctas_per_wave") or 0.0)
    wave_shape = _mapping_value(diagnostics, "wave_shape")
    last_wave_ctas = float(wave_shape.get("last_wave_ctas") or 0.0)
    if resident_ctas_per_sm <= 0.0 or ctas_per_wave <= 0.0:
        last_active_fraction = 1.0 if last_wave_ctas > 0.0 else 0.0
    else:
        num_sms = ctas_per_wave / resident_ctas_per_sm
        last_active_fraction = min(num_sms, last_wave_ctas) / max(num_sms, 1.0e-12)
    return full_wave_count * full_wave_time_s + last_active_fraction * last_wave_time_s


def _mapping_value(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, Mapping) else {}


def _total_bytes(read_bytes: Any, write_bytes: Any) -> int | None:
    if read_bytes is None and write_bytes is None:
        return None
    return int(read_bytes or 0) + int(write_bytes or 0)
