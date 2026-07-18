from __future__ import annotations

import math

import pytest

from opmodel.calibration import (
    _fit_nonnegative_two_feature_power,
    _weighted_feature_scales,
)
from opmodel.energy import EnergyFeatureRow, predict_e3_energy
from opmodel.hardware import EnergyModelPowerCoefficients


def test_normalized_power_fit_recovers_watts_across_feature_scales() -> None:
    rows = []
    for index in range(1, 9):
        kernel_s = index * 1.0e-5
        dram_s = (9 - index) * 2.0e-7
        residual_j = 48.0 * kernel_s + 14.0 * dram_s
        measured_j = 2.0e-4 + residual_j
        rows.append((kernel_s, dram_s, residual_j, measured_j))

    first = _fit_nonnegative_two_feature_power(rows)
    second = _fit_nonnegative_two_feature_power(rows)

    assert first == second
    assert first == pytest.approx((48.0, 14.0), rel=1.0e-10)
    assert all(math.isfinite(value) and value >= 0.0 for value in first)
    assert all(value > 0.0 for value in _weighted_feature_scales(rows))


def test_nonnegative_power_fit_clamps_unphysical_residual() -> None:
    rows = [
        (1.0e-5, 1.0e-7, -1.0e-3, 2.0e-3),
        (2.0e-5, 3.0e-7, -2.0e-3, 3.0e-3),
    ]

    assert _fit_nonnegative_two_feature_power(rows) == (0.0, 0.0)


def test_e3_energy_levels_are_additive() -> None:
    features = EnergyFeatureRow(
        flops_total=1.0,
        bytes_dram_total=2.0,
        bytes_l2_total=3.0,
        bytes_smem_total=4.0,
        time_kernel_s=2.0,
        time_sm_resident_s=3.0,
        time_tc_active_s=5.0,
        time_dram_active_s=7.0,
        time_l2_active_s=11.0,
        time_smem_active_s=13.0,
    )
    coefficients = EnergyModelPowerCoefficients(
        base_power_w=2.0,
        sm_resident_power_w=3.0,
        tc_active_power_w=5.0,
        dram_active_power_w=7.0,
        l2_active_power_w=11.0,
        smem_active_power_w=13.0,
    )

    prediction = predict_e3_energy(
        features=features,
        event_energy_j=17.0,
        power_coefficients=coefficients,
    )

    assert prediction.e0_j == 17.0
    assert prediction.e1_j == 21.0
    assert prediction.e2_j == 30.0
    assert prediction.e3_j == 30.0 + 25.0 + 49.0 + 121.0 + 169.0
