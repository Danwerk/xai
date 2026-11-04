import numpy as np
import pytest

from src.evaluation import (
    accept_refinement,
    scenario_loss_interval,
    seasonal_error_scale,
)


def test_mase_scale_uses_seasonal_training_differences():
    # Järjestikuse tunni vahe on 1, eelmise päeva sama tunni vahe 24.
    training = np.arange(100, dtype=float)
    assert seasonal_error_scale(training, season=24) == 24
    assert 12 / seasonal_error_scale(training, season=24) == 0.5


def test_constant_training_series_has_no_mase_scale():
    with pytest.raises(ValueError, match="positiivne"):
        seasonal_error_scale(np.ones(48))


def test_global_improvement_does_not_override_scenario_harm():
    assert not accept_refinement(-2, 1, 100)
    assert not accept_refinement(1, -2, 100)
    assert not accept_refinement(-2, -1, 39)
    assert not accept_refinement(-2, 0, 100)
    assert accept_refinement(-2, -1, 40)


def test_scenario_interval_keeps_hourly_mask_and_ignores_other_losses():
    mask = np.tile(np.r_[np.ones(12), np.zeros(12)], 20).astype(bool)
    difference = np.where(mask, -2.0, 100.0)
    result = scenario_loss_interval(difference, mask, block=24, repeats=100)
    assert result["difference"] == -2
    assert result["ci_low"] == result["ci_high"] == -2
    assert result["valid_draws"] == 100


def test_identical_policies_have_exact_zero_difference():
    result = scenario_loss_interval(np.zeros(336), np.ones(336, dtype=bool))
    assert result["difference"] == result["adjusted_low"] == result["adjusted_high"] == 0
