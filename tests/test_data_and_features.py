import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import load_and_audit, validate_for_modeling
from src.feature_engineering import build_features, split_features, load_run_features
from src.common import ROOT, sha256, write_json


@pytest.mark.parametrize("expanded", [False, True])
def test_run_features_rebuilt_exactly_without_intermediate_files(
    hourly_data, cfg, tmp_path, expanded
):
    from src.expanded_features import build_expanded_features

    # A full four-week history is needed by the expanded feature set.
    data = pd.concat([hourly_data] * 3, ignore_index=True)
    data["datetime"] = pd.date_range("2017-01-01", periods=len(data), freq="h")
    raw_path = tmp_path / "raw.csv"
    data.to_csv(raw_path, index=False)
    raw, _ = load_and_audit(raw_path)
    saved_config = {**cfg, "raw_path": str(raw_path), "weather_delay_hours": 26}
    modules = [
        "src/data_preprocessing.py",
        "src/feature_engineering.py",
        "src/expanded_features.py",
    ]
    write_json(
        tmp_path / "manifest.json",
        {
            "raw_sha256": sha256(raw_path),
            "source_sha256": {
                str((ROOT / name).relative_to(ROOT)): sha256(ROOT / name) for name in modules
            },
        },
    )
    config_name = "original_experiment" if expanded else "experiment"
    write_json(tmp_path / f"configs/{config_name}.json", saved_config)
    if expanded:
        write_json(tmp_path / "configs/expanded_study.json", {"raw_path": str(raw_path)})
    existing = set(tmp_path.rglob("*"))
    expected = (
        build_expanded_features(raw, saved_config)[0]
        if expanded
        else build_features(raw, saved_config)
    )
    restored = load_run_features(tmp_path, expanded=expanded)
    assert len(restored) > 0
    pd.testing.assert_frame_equal(restored, expected, check_exact=True)
    assert set(tmp_path.rglob("*")) == existing

    raw_path.write_text(raw_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Algandmed"):
        load_run_features(tmp_path, expanded=expanded)


def test_audit_preserves_positive_outlier_and_sorts(hourly_data, tmp_path):
    hourly_data.loc[10, "nat_demand"] = 1
    path = tmp_path / "source.csv"
    hourly_data.iloc[::-1].to_csv(path, index=False)
    d, audit = load_and_audit(path)
    validate_for_modeling(d, audit)
    assert len(d) == 400 and d.nat_demand.min() == 1
    assert audit["was_sorted"] is False
    assert audit["non_hourly_intervals"] == 0


@pytest.mark.parametrize("problem", ["duplicate", "gap", "nan", "inf", "negative", "badflag"])
def test_invalid_data_fails_without_repair(hourly_data, tmp_path, problem):
    d = hourly_data.copy()
    if problem == "duplicate":
        d = pd.concat([d, d.iloc[[0]]])
    elif problem == "gap":
        d = d.drop(2)
    else:
        col, value = {
            "nan": ("nat_demand", np.nan),
            "inf": ("T2M_toc", np.inf),
            "negative": ("nat_demand", -1),
            "badflag": ("holiday", 2),
        }[problem]
        d.loc[2, col] = value
    path = tmp_path / "invalid.csv"
    d.to_csv(path, index=False)
    data, audit = load_and_audit(path)
    with pytest.raises(ValueError):
        validate_for_modeling(data, audit)


def test_missing_file_does_not_generate_data(tmp_path):
    path = tmp_path / "absent.csv"
    with pytest.raises(FileNotFoundError):
        load_and_audit(path)
    assert not path.exists()


def test_exact_origin_target_and_seasonal_lags(hourly_data, cfg):
    f = build_features(hourly_data, cfg)
    row = f.iloc[0]
    i = int(np.flatnonzero(hourly_data.datetime == row.origin_time)[0])
    y = hourly_data.nat_demand
    assert row.target_next_hour == y[i + 1]
    assert row.load_now == y[i]
    assert row.load_day == y[i - 23]
    assert row.load_week == y[i - 167]
    assert row.load_lag_2 == y[i - 2]
    assert row.target_time == row.origin_time + pd.Timedelta(hours=1)
    assert row.hour == row.target_time.hour
    assert row.dayofweek == row.target_time.dayofweek


def test_rolling_and_weather_delay(hourly_data, cfg):
    f = build_features(hourly_data, cfg)
    row = f.iloc[0]
    i = int(np.flatnonzero(hourly_data.datetime == row.origin_time)[0])
    assert row.load_mean_6 == hourly_data.nat_demand.iloc[i - 5 : i + 1].mean()
    assert row.load_std_24 == pytest.approx(hourly_data.nat_demand.iloc[i - 23 : i + 1].std())
    assert row.load_ramp_1h == 1
    assert row.temperature == pytest.approx(hourly_data.T2M_toc[i - 24])


def test_future_mutation_cannot_change_any_input(hourly_data, cfg):
    origin = hourly_data.datetime.iloc[250]
    before = build_features(hourly_data, cfg).set_index("origin_time")
    changed = hourly_data.copy()
    future = changed.datetime > origin
    for c in changed.columns.difference(["datetime"]):
        changed.loc[future, c] = 9999
    after = build_features(changed, cfg).set_index("origin_time")
    inputs = cfg["baseline_features"] + [
        c for cols in cfg["candidate_groups"].values() for c in cols
    ]
    pd.testing.assert_series_equal(before.loc[origin, inputs], after.loc[origin, inputs])
    assert before.loc[origin, "target_next_hour"] != after.loc[origin, "target_next_hour"]


def test_calendar_date_boundary_does_not_read_tomorrow(hourly_data, cfg):
    d = hourly_data.copy()
    d.loc[d.datetime >= "2017-12-30", ["holiday", "school"]] = [1, 0]
    f = build_features(d, cfg).set_index("origin_time")
    assert f.loc[pd.Timestamp("2017-12-29 23:00"), "holiday_known"] == 0
    assert f.loc[pd.Timestamp("2017-12-30 01:00"), "holiday_known"] == 1
    assert f.loc[pd.Timestamp("2017-12-29 23:00"), "hour"] == 0


def test_configurable_load_latency(hourly_data, cfg):
    cfg["load_delay_hours"] = 24
    row = build_features(hourly_data, cfg).iloc[0]
    i = int(np.flatnonzero(hourly_data.datetime == row.origin_time)[0])
    assert row.load_now == hourly_data.nat_demand[i - 24]
    assert row.load_day == hourly_data.nat_demand[i - 47]


def test_split_uses_target_timestamp(cfg):
    f = pd.DataFrame(
        {
            "origin_time": pd.to_datetime(
                ["2017-12-31 22:00", "2017-12-31 23:00", "2018-12-31 23:00", "2019-12-31 23:00"]
            )
        }
    )
    f["target_time"] = f.origin_time + pd.Timedelta(hours=1)
    s = split_features(f, cfg)
    assert len(s["train"]) == 1
    assert s["diagnosis"].target_time.iloc[0] == pd.Timestamp("2018-01-01")
    assert s["test"].target_time.iloc[0] == pd.Timestamp("2020-01-01")


@pytest.mark.parametrize(
    "field", ["target_next_hour", "actual_ramp", "actual_temperature", "holiday_retrospective"]
)
def test_config_cannot_promote_retrospective_labels_to_features(hourly_data, cfg, field):
    cfg["baseline_features"].append(field)
    with pytest.raises(ValueError, match="tagasivaatelisi"):
        build_features(hourly_data, cfg)
