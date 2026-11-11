import numpy as np
import pandas as pd
import pytest

from src.simple_correction import bias_forecast, combined_forecasts


def observations():
    times = pd.date_range("2018-01-01", periods=240, freq="h")
    return pd.DataFrame(
        {
            "target_time": times,
            "origin_time": times - pd.Timedelta(hours=1),
            "hour": times.hour,
            "prediction_residual": 1000.0,
            "target_next_hour": 1000.0 + np.arange(len(times)),
        }
    )


SETTING = {"window_hours": 168, "strength": 1.0}
FEEDBACK = {"min_history_points": 3, "warmup_hours": 72}


def test_same_hour_history_warmup_and_window_endpoint():
    frame = observations()
    predicted = bias_forecast(frame, SETTING, FEEDBACK)
    np.testing.assert_array_equal(predicted[:72], np.full(72, 1000.0))
    assert predicted[72] == 1024.0  # Previous errors: 0, 24, 48.
    assert predicted[168] == 1072.0  # Includes the left endpoint at hour 0.
    assert predicted[192] == 1096.0  # Hour 0 has expired; 24..168 remain.
    half = bias_forecast(frame, {**SETTING, "strength": 0.5}, FEEDBACK)
    assert half[72] == 1012.0


def test_future_outcomes_do_not_change_current_prediction_and_input_is_unchanged():
    frame = observations()
    original = frame.copy(deep=True)
    predicted = bias_forecast(frame, SETTING, FEEDBACK)
    pd.testing.assert_frame_equal(frame, original)
    prefix = bias_forecast(frame.iloc[:121], SETTING, FEEDBACK)
    np.testing.assert_array_equal(predicted[:121], prefix)
    frame.loc[120:, "target_next_hour"] = 1e9
    changed = bias_forecast(frame, SETTING, FEEDBACK)
    np.testing.assert_array_equal(predicted[:121], changed[:121])


def test_gaps_expire_old_history_and_identity_is_a_copy():
    frame = observations().iloc[:3].copy()
    frame["target_time"] = pd.to_datetime(["2018-01-01", "2018-01-02", "2018-02-01"])
    frame["origin_time"] = frame.target_time - pd.Timedelta(hours=1)
    frame["hour"] = 0
    frame["target_next_hour"] = [1010.0, 1020.0, 1030.0]
    predicted = bias_forecast(frame, SETTING, {"min_history_points": 1, "warmup_hours": 0})
    np.testing.assert_array_equal(predicted, [1000.0, 1010.0, 1000.0])
    identity = bias_forecast(frame, None, FEEDBACK)
    identity[0] = -1
    assert frame.prediction_residual.iloc[0] == 1000.0


def test_invalid_time_order_and_settings_fail():
    frame = observations()
    with pytest.raises(ValueError):
        bias_forecast(frame.iloc[::-1], SETTING, FEEDBACK)
    with pytest.raises(ValueError):
        bias_forecast(frame, {**SETTING, "window_hours": 0}, FEEDBACK)
    with pytest.raises(ValueError):
        bias_forecast(frame, {**SETTING, "strength": 2}, FEEDBACK)


def test_explicit_first_stage_uses_its_own_errors_and_keeps_inputs_unchanged():
    frame = observations()
    original = frame.copy(deep=True)
    first_stage = np.full(len(frame), 1100.0)
    predicted = bias_forecast(frame, SETTING, FEEDBACK, reference=first_stage)
    np.testing.assert_array_equal(predicted[:72], first_stage[:72])
    # Errors relative to 1100 are -100, -76, -52; mean -76, giving 1024.
    assert predicted[72] == 1024.0
    assert bias_forecast(frame, SETTING, FEEDBACK)[72] == 1024.0
    # Half-strength distinguishes own-stream errors from reusing residual-model errors.
    half = bias_forecast(frame, {**SETTING, "strength": 0.5}, FEEDBACK, reference=first_stage)
    assert half[72] == 1062.0
    changed = frame.copy()
    changed.loc[120:, "target_next_hour"] = 1e9
    after = bias_forecast(changed, SETTING, FEEDBACK, reference=first_stage)
    np.testing.assert_array_equal(predicted[:121], after[:121])
    identity = bias_forecast(frame, None, FEEDBACK, reference=first_stage)
    identity[0] = -1
    assert first_stage[0] == 1100.0
    pd.testing.assert_frame_equal(frame, original)


def test_combined_methods_are_causal_and_scalar_control_ignores_decomposition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = observations().assign(base_prediction=990.0)
    frame["target_next_hour"] = 996 + np.sin(np.arange(len(frame)))
    original = frame.copy(deep=True)
    components = pd.DataFrame({"recent_load": 20.0, "clock": -10.0}, index=frame.index)
    setting = {"window_hours": 168, "strength": 1.0, "by_hour": True}
    before = combined_forecasts(frame, components, setting, SETTING, FEEDBACK)
    changed = frame.copy()
    changed.loc[120:, "target_next_hour"] = 1e9
    after = combined_forecasts(changed, components, setting, SETTING, FEEDBACK)
    alternative_parts = pd.DataFrame({"recent_load": 100.0, "clock": -90.0}, index=frame.index)
    altered = combined_forecasts(frame, alternative_parts, setting, SETTING, FEEDBACK)
    for name in before:
        np.testing.assert_allclose(before[name][:121], after[name][:121])
        np.testing.assert_allclose(before[name][:72],
                                   frame.base_prediction.iloc[:72] if name == "base"
                                   else frame.prediction_residual.iloc[:72])
    for name in ("matched_scalar", "scalar_bias"):
        np.testing.assert_array_equal(before[name], altered[name])
    np.testing.assert_allclose(before["xai_bias"], bias_forecast(
        frame, SETTING, FEEDBACK, reference=before["credit_feedback"]
    ))
    pd.testing.assert_frame_equal(frame, original)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("reference", [np.ones(2), np.ones((240, 1)), np.full(240, np.nan)])
def test_invalid_reference_forecasts_fail(reference):
    with pytest.raises(ValueError, match="Reference forecasts"):
        bias_forecast(observations(), SETTING, FEEDBACK, reference=reference)


def test_misaligned_reference_series_fails():
    frame = observations()
    with pytest.raises(ValueError, match="align"):
        bias_forecast(frame, SETTING, FEEDBACK, reference=pd.Series(1000.0, index=frame.index + 1))
