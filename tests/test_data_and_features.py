import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import load_and_audit, validate_for_modeling
from src.feature_engineering import (
    build_features,
    feature_columns,
    known_error_features,
    load_estonia_hourly,
)


@pytest.mark.parametrize("problem", ["duplicate", "gap", "nan", "inf", "negative", "flag"])
def test_invalid_sources_fail_without_repair(hourly_data, tmp_path, problem):
    data = hourly_data.copy()
    if problem == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    elif problem == "gap":
        data = data.drop(2)
    else:
        col, value = {
            "nan": ("nat_demand", np.nan),
            "inf": ("T2M_toc", np.inf),
            "negative": ("nat_demand", -1),
            "flag": ("school", 4),
        }[problem]
        data.loc[2, col] = value
    path = tmp_path / "raw.csv"
    data.to_csv(path, index=False)
    read, quality = load_and_audit(path)
    with pytest.raises(ValueError):
        validate_for_modeling(read, quality)


def test_positive_outliers_preserved_and_missing_source_fails(hourly_data, tmp_path):
    hourly_data.loc[10, "nat_demand"] = 1
    path = tmp_path / "raw.csv"
    hourly_data.iloc[::-1].to_csv(path, index=False)
    frame, quality = load_and_audit(path)
    assert validate_for_modeling(frame, quality).nat_demand.min() == 1
    assert not quality["was_sorted"]
    with pytest.raises(FileNotFoundError):
        load_and_audit(tmp_path / "missing.csv")


def test_features_have_exact_forecast_alignment_and_no_future_inputs(hourly_data, cfg):
    before = build_features(hourly_data, cfg).set_index("origin_time")
    origin = before.index[40]
    i = hourly_data.index[hourly_data.datetime.eq(origin)][0]
    row = before.loc[origin]
    assert row.target_next_hour == hourly_data.nat_demand[i + 1]
    assert row.load_day == hourly_data.nat_demand[i - 23]
    assert row.load_week == hourly_data.nat_demand[i - 167]
    assert row.load_std_24 == pytest.approx(hourly_data.nat_demand.iloc[i - 23 : i + 1].std())
    assert row.temperature == pytest.approx(hourly_data.T2M_toc[i - 24])
    changed = hourly_data.copy()
    changed.loc[changed.datetime > origin, changed.columns.difference(["datetime"])] = 10000
    after = build_features(changed, cfg).set_index("origin_time")
    _, columns = feature_columns("panama")
    pd.testing.assert_series_equal(before.loc[origin, columns], after.loc[origin, columns])
    assert before.loc[origin, "target_next_hour"] != after.loc[origin, "target_next_hour"]
    assert not {"actual_ramp", "target_next_hour", "hour"}.intersection(columns)


def test_calendar_boundary_and_required_history_only(hourly_data, cfg):
    hourly_data.loc[hourly_data.datetime >= "2017-12-10", "holiday"] = 1
    frame = build_features(hourly_data, cfg).set_index("origin_time")
    assert frame.loc[pd.Timestamp("2017-12-09 23:00"), "holiday_known"] == 0
    assert frame.loc[pd.Timestamp("2017-12-10 00:00"), "holiday_known"] == 1
    assert frame.index.min() == hourly_data.datetime.iloc[167]


def test_error_memory_uses_only_available_timestamped_targets():
    target = pd.date_range("2018-01-01", periods=200, freq="h")
    frame = pd.DataFrame(
        {
            "target_time": target,
            "origin_time": target - pd.Timedelta(hours=1),
            "base_prediction": 0.0,
            "target_next_hour": np.arange(200.0),
        }
    )
    before = known_error_features(frame)
    frame.loc[100:, "target_next_hour"] = 1e9
    after = known_error_features(frame)
    pd.testing.assert_frame_equal(before.iloc[:101], after.iloc[:101])
    assert before.loc[100, "known_error_24"] == 76
    assert after.loc[101, "known_error_1"] == 1e9
    gap = frame.iloc[[0, 2]].reset_index(drop=True)
    assert known_error_features(gap).loc[1, "known_error_1"] == 0


def test_estonia_interval_labels_missing_quarters_and_dst_calendar(tmp_path, cfg):
    times = pd.date_range("2024-03-20", periods=5000, freq="15min", tz="UTC")
    raw = pd.DataFrame({"datetime": times, "consumption": 1000 + np.arange(5000) // 4})
    raw = raw.drop(index=4001)
    path = tmp_path / "estonia.csv"
    raw.to_csv(path, index=False)
    hourly, quality = load_estonia_hourly(path, 2024)
    assert quality["imputed_values"] == 0 and quality["missing_hourly_load"] == 1
    row = hourly.set_index("datetime").loc[pd.Timestamp("2024-03-20 01:00")]
    assert row.nat_demand == 1000
    frame = build_features(hourly, cfg, "estonia").set_index("origin_time")
    assert frame.loc[pd.Timestamp("2024-03-31 00:00"), "hour"] == 4
    assert frame.loc[pd.Timestamp("2024-03-30 00:00"), "hour"] == 3
    assert "temperature" not in frame and "holiday_known" not in frame


def test_load_latency_cannot_silently_break_feedback(hourly_data, cfg):
    cfg["load_delay_hours"] = 1
    with pytest.raises(ValueError):
        build_features(hourly_data, cfg)
