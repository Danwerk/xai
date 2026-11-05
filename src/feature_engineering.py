"""One fixed feature set; origin t predicts the completed interval ending at t+1."""

import numpy as np
import pandas as pd

from .data_preprocessing import load_and_audit, validate_for_modeling

BASE_FEATURES = [
    "load_now",
    "load_day",
    "load_week",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "weekend",
]
WEATHER_FEATURES = ["temperature", "humidity", "wind"]
CORRECTION_FEATURES = [
    "load_ramp_1h",
    "load_ramp_3h",
    "load_mean_6",
    "load_std_6",
    "load_mean_24",
    "load_std_24",
    "load_lag_2",
    "load_lag_3",
]
ERROR_FEATURES = [
    "known_error_1",
    "known_error_24",
    "known_error_168",
    "known_error_mean24",
    "known_error_mean168",
]


def split_year(frame, year):
    """Expanding training history and one forecast year, with common row order."""
    train = frame[frame.target_time < f"{year}-01-01"]
    valid = frame[frame.target_time.dt.year.eq(year)].copy().reset_index(drop=True)
    if train.empty or valid.empty or train.target_time.max() > valid.origin_time.min():
        raise ValueError(f"Invalid chronological fold: {year}")
    return train, valid


def feature_columns(dataset):
    if dataset not in ("panama", "estonia"):
        raise ValueError(f"Unknown dataset: {dataset}")
    base = BASE_FEATURES + (WEATHER_FEATURES if dataset == "panama" else [])
    correction = base + CORRECTION_FEATURES
    if dataset == "panama":
        correction += ["holiday_known", "school_known"]
    return base, correction


def load_estonia_hourly(path, end_year):
    """UTC start-labelled measurements -> completed, right-labelled hourly means."""
    raw = pd.read_csv(path)
    raw["datetime"] = pd.to_datetime(raw.datetime, utc=True, errors="raise")
    raw = raw[raw.datetime < f"{end_year + 1}-01-01"].copy()
    if raw.empty or raw.datetime.isna().any() or raw.datetime.duplicated().any():
        raise ValueError("Estonia requires nonempty, unique valid timestamps.")
    raw["consumption"] = pd.to_numeric(raw.consumption, errors="raise")
    if np.isinf(raw.consumption).any():
        raise ValueError("Infinite source consumption.")
    nonpositive = int(raw.consumption.le(0).sum())
    raw.loc[raw.consumption.le(0), "consumption"] = np.nan
    grouped = (
        raw.set_index("datetime")
        .consumption.sort_index()
        .resample("h", label="right", closed="left")
    )
    counts = grouped.count()
    expected = counts.groupby(counts.index.normalize()).transform("max")
    load = grouped.mean().where(counts.eq(expected) & expected.isin([1, 4]))
    frame = pd.DataFrame({"datetime": load.index.tz_localize(None), "nat_demand": load.to_numpy()})
    return frame, {
        "source_rows": len(raw),
        "hourly_intervals": len(load),
        "missing_hourly_load": int(load.isna().sum()),
        "nonpositive_load_unavailable": nonpositive,
        "imputed_values": 0,
        "hourly_label": "right endpoint of completed interval",
        "split_timezone": "UTC",
        "calendar_timezone": "Europe/Tallinn",
    }


def build_features(raw, cfg, dataset="panama"):
    if cfg["load_delay_hours"] != 0 or cfg["weather_delay_hours"] < 1:
        raise ValueError(
            "This protocol assumes immediate load/error availability and delayed weather."
        )
    d = raw.set_index("datetime").sort_index()
    if (
        d.index.has_duplicates
        or not d.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Features require unique regular hourly observations.")
    target = d.index + pd.Timedelta(hours=1)
    calendar = (
        target.tz_localize("UTC").tz_convert("Europe/Tallinn") if dataset == "estonia" else target
    )
    load = d.nat_demand
    f = pd.DataFrame(index=d.index)
    f["origin_time"], f["target_time"] = d.index, target
    f["target_next_hour"] = load.shift(-1)
    f["actual_ramp"] = load.shift(-1) - load  # Evaluation only; never a model input.
    f["load_now"], f["load_day"], f["load_week"] = load, load.shift(23), load.shift(167)
    f["hour"] = calendar.hour
    for name, value, period in (
        ("hour", calendar.hour, 24),
        ("dow", calendar.dayofweek, 7),
        ("month", calendar.month - 1, 12),
    ):
        f[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    f["weekend"] = (calendar.dayofweek >= 5).astype(int)
    for lag in (1, 3):
        f[f"load_ramp_{lag}h"] = load.diff(lag)
    for lag in (2, 3):
        f[f"load_lag_{lag}"] = load.shift(lag)
    for window in (6, 24):
        f[f"load_mean_{window}"] = load.rolling(window).mean()
        f[f"load_std_{window}"] = load.rolling(window).std()
    if dataset == "panama":
        for name, source in (("temperature", "T2M"), ("humidity", "QV2M"), ("wind", "W2M")):
            f[name] = (
                d[[f"{source}_{city}" for city in ("toc", "san", "dav")]]
                .mean(axis=1)
                .shift(cfg["weather_delay_hours"])
            )
        same_day = d.index.normalize() == target.normalize()
        f["holiday_known"] = d.holiday.where(same_day, 0)
        f["school_known"] = d.school.where(same_day, 0)
    _, columns = feature_columns(dataset)
    f = f.dropna(subset=["target_next_hour", *columns]).reset_index(drop=True)
    if f.empty or not np.isfinite(f[["target_next_hour", *columns]].to_numpy()).all():
        raise ValueError("No finite complete forecast windows.")
    return f


def load_dataset(path, cfg, dataset):
    options = cfg["datasets"][dataset]
    if dataset == "estonia":
        raw, quality = load_estonia_hourly(path, options["test_year"])
    else:
        raw, quality = load_and_audit(path)
        validate_for_modeling(raw, quality)
        raw = raw[raw.datetime < f"{options['test_year'] + 1}-01-01"]
    frame = build_features(raw, cfg, dataset)
    test = frame[frame.target_time.dt.year == options["test_year"]]
    expected = len(
        pd.date_range(
            f"{options['test_year']}-01-01",
            f"{options['test_year']+1}-01-01",
            freq="h",
            inclusive="left",
        )
    )
    quality.update(
        valid_forecasts=len(frame),
        test_forecasts=len(test),
        calendar_test_hours=expected,
        unscored_test_hours=expected - len(test),
        feature_policy="fixed required inputs only; common mask for every method",
    )
    return frame, quality


def known_error_features(history):
    """Residual y-f whose target has arrived by the current forecast origin."""
    times = pd.DatetimeIndex(history.target_time)
    origins = pd.DatetimeIndex(history.origin_time)
    if (
        times.has_duplicates
        or not times.is_monotonic_increasing
        or not (times - origins == pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Error history must have ordered unique one-hour targets.")
    errors = pd.Series((history.target_next_hour - history.base_prediction).to_numpy(), index=times)
    out = pd.DataFrame(index=history.index)
    for lag in (1, 24, 168):
        out[f"known_error_{lag}"] = errors.reindex(origins - pd.Timedelta(hours=lag - 1)).to_numpy()
    for window in (24, 168):
        out[f"known_error_mean{window}"] = (
            errors.rolling(f"{window}h", closed="left").mean().to_numpy()
        )
    return out.fillna(0)
