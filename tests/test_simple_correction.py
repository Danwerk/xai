import numpy as np
import pandas as pd
import pytest

from src.simple_correction import bias_forecast


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
