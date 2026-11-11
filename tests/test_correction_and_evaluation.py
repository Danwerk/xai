from math import factorial

import numpy as np
import pandas as pd
import pytest

from src.correction import adaptive_correction, past_average
from src.evaluation import (
    clock_loss_interval, correction_diagnostics, forecast_metrics,
    persistence_forecast, scenario_loss_interval,
)
from src.shap_explainer import repair_gain_attributions


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_credits_match_independent_coalition_enumeration(loss):
    c = np.array([[8.0, 7.0, -2.0], [2.0, -2.0, 0.0]])
    y = np.array([10.0, 3.0])
    power = 2 if loss == "squared" else 1

    def value(mask):
        total = sum((c[:, j] for j in range(3) if mask & (1 << j)), np.zeros(2))
        return abs(y) ** power - abs(y - total) ** power

    expected = np.zeros_like(c)
    for j in range(3):
        for mask in range(8):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(2 - size) / factorial(3)
            expected[:, j] += weight * (value(mask | (1 << j)) - value(mask))
    actual = repair_gain_attributions(y, np.zeros(2), c, loss)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), value(7))
    assert actual[1, 2] == 0


def feedback_data():
    rng = np.random.default_rng(42)
    times = pd.date_range("2020-01-01", periods=400, freq="h")
    frame = pd.DataFrame(
        {
            "target_time": times,
            "origin_time": times - pd.Timedelta(hours=1),
            "hour": times.hour,
            "base_prediction": 100.0,
            "target_next_hour": 100 + rng.normal(size=400),
        }
    )
    components = pd.DataFrame(rng.normal(size=(400, 3)), columns=list("abc"))
    return frame, components


@pytest.mark.parametrize("mode", ["credit", "standalone", "scalar_credit", "scalar_ls"])
@pytest.mark.parametrize("by_hour", [False, True])
def test_every_policy_is_prefix_causal_bounded_and_reconstructs(mode, by_hour):
    frame, components = feedback_data()
    setting = {"window_hours": 168, "by_hour": by_hour, "strength": 1.0}
    feedback = {"min_history_points": 3, "warmup_hours": 72}
    before, effective, weights = adaptive_correction(frame, components, setting, mode, feedback)
    np.testing.assert_allclose(before, frame.base_prediction + effective.sum(axis=1))
    assert weights.ge(0).all().all() and weights.le(1).all().all()
    assert weights.iloc[:72].eq(1).all().all()
    prefix, _, prefix_weights = adaptive_correction(
        frame.iloc[:201], components.iloc[:201], setting, mode, feedback
    )
    np.testing.assert_allclose(before[:201], prefix)
    np.testing.assert_allclose(weights.iloc[:201], prefix_weights)
    frame.loc[200:, "target_next_hour"] = 1e9
    after, _, _ = adaptive_correction(frame, components, setting, mode, feedback)
    np.testing.assert_allclose(before[:201], after[:201])
    if mode.startswith("scalar"):
        np.testing.assert_allclose(weights.a, weights.b)


def test_scalar_least_squares_needs_no_shap_credit(monkeypatch):
    import src.correction as correction

    frame, components = feedback_data()
    total = components.sum(axis=1).to_frame("one")

    def forbidden(*args, **kwargs):
        raise AssertionError("B4 must not request credit attribution")

    monkeypatch.setattr(correction, "repair_gain_attributions", forbidden)
    setting = {"window_hours": 168, "by_hour": False, "strength": 1.0}
    feedback = {"min_history_points": 3, "warmup_hours": 72}
    a, _, _ = adaptive_correction(frame, components, setting, "scalar_ls", feedback)
    b, _, _ = adaptive_correction(frame, total, setting, "scalar_ls", feedback)
    np.testing.assert_allclose(a, b)


def test_clock_window_expires_history_across_gaps():
    times = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-02-01"])
    frame = pd.DataFrame({"target_time": times, "hour": [0, 0, 0]})
    out = past_average(np.array([[10.0], [20.0], [30.0]]), frame, 168, True, 1)
    assert out.iloc[1, 0] == 10
    assert np.isnan(out.iloc[2, 0])


def test_identity_and_invalid_alignment():
    frame, components = feedback_data()
    p, _, weights = adaptive_correction(frame, components, None, "credit", {})
    np.testing.assert_allclose(p, frame.base_prediction + components.sum(axis=1))
    assert weights.eq(1).all().all()
    with pytest.raises(ValueError):
        adaptive_correction(frame.iloc[::-1], components.iloc[::-1], None, "credit", {})


def test_harm_magnitude_uses_all_hours_and_named_reference():
    scores = forecast_metrics([10, 10], [10, 14], [12, 12], [10, 13])
    assert scores["MAE"] == 2
    assert scores["harmful_rate_vs_base"] == 0.5
    assert scores["harmful_magnitude_vs_base"] == 1
    assert scores["harmful_conditional_mean_vs_base"] == 2
    assert scores["harmful_magnitude_vs_residual"] == 0.5
    assert scores["improvement_vs_base_pct"] == 0


def test_clock_bootstrap_masks_missing_calendar_hours():
    times = pd.date_range("2020-01-01", periods=100, freq="h")
    differences = np.arange(100.0)
    mask = np.ones(100, bool)
    mask[30:50] = False
    args = dict(block=24, repeats=100, comparisons=3, seed=42)
    padded = differences.copy()
    padded[~mask] = 0
    a = clock_loss_interval(times[mask], differences[mask], np.ones(mask.sum(), bool), **args)
    b = scenario_loss_interval(padded, mask, **args)
    assert a == b
    assert a["adjusted_low"] <= a["ci_low"] <= a["ci_high"] <= a["adjusted_high"]


def test_persistence_uses_only_available_load_on_the_supplied_rows():
    times = pd.date_range("2025-01-01", periods=4, freq="h")
    frame = pd.DataFrame({"target_time": times, "origin_time": times - pd.Timedelta(hours=1),
                          "load_now": [100, 110, 105, 120], "target_next_hour": [110, 105, 120, 125]})
    before = persistence_forecast(frame)
    np.testing.assert_array_equal(before, [100, 110, 105, 120])
    frame["target_next_hour"] = -1e9
    np.testing.assert_array_equal(persistence_forecast(frame), before)
    before[0] = 0
    assert frame.load_now.iloc[0] == 100
    with pytest.raises(ValueError):
        persistence_forecast(frame.assign(origin_time=times - pd.Timedelta(hours=2)))


def diagnostic_example():
    frame = pd.DataFrame({"base_prediction": [10., 8., 8.], "prediction_residual": [10., 10., 10.],
                          "target_next_hour": [10., 8., 12.]})
    components = pd.DataFrame({"one": [2., 2., 2.], "two": [-2., 0., 0.]})
    weights = pd.DataFrame({"one": [0., 0., 0.], "two": [0., 1., 1.]})
    predicted = frame.base_prediction + (components * weights).sum(axis=1)
    return frame, components, weights, predicted


def test_diagnostics_separate_cancellation_activity_and_realised_gain():
    frame, components, weights, predicted = diagnostic_example()
    details, summary = correction_diagnostics(frame, components, weights, predicted)
    np.testing.assert_allclose(details.net_change, [0, -2, -2])
    np.testing.assert_allclose(details.cancellation_share, [1, 0, 0])
    np.testing.assert_allclose(details.gain_vs_residual, [0, 2, -2])
    assert summary["changed_hours"] == 2
    assert summary["parts_changed_pct"] == 100
    assert summary["helped_pct_when_changed"] == summary["harmed_pct_when_changed"] == 50
    assert summary["mae_gain_overall"] == 0
    assert summary["mean_cancellation_when_parts_change"] == pytest.approx(1/3)
    original = frame.copy()
    correction_diagnostics(frame, components, weights, predicted)
    pd.testing.assert_frame_equal(frame, original)


def test_diagnostics_identity_has_no_activity_and_no_conditional_success_rate():
    frame, components, weights, _ = diagnostic_example()
    weights[:] = 1
    details, summary = correction_diagnostics(frame, components, weights, frame.prediction_residual)
    assert summary["changed_hours"] == 0 and summary["changed_pct"] == 0
    assert np.isnan(summary["helped_pct_when_changed"])
    assert details.cancellation_share.isna().all()


def test_diagnostics_reject_wrong_reconstruction_and_alignment():
    frame, components, weights, predicted = diagnostic_example()
    with pytest.raises(AssertionError):
        correction_diagnostics(frame, components, weights, predicted + 1)
    with pytest.raises(ValueError):
        correction_diagnostics(frame, components, weights[weights.columns[::-1]], predicted)
    with pytest.raises(ValueError):
        correction_diagnostics(frame, components, weights + 2, predicted)
