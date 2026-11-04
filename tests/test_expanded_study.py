import numpy as np
import pandas as pd
import pytest

from src.expanded_features import build_expanded_features
from src.model_comparison import temporal_split, rank_candidates, make_model, compare_model_pairs
from src.evaluation import paired_intervals
from src.common import read_artifact_csv, write_csv
from src.train_baseline import prediction_frame


def test_all_model_pairs_use_the_full_comparison_family():
    names = ["Ridge", "GBR", "HistGBR"]
    predicted = {}
    for year, period in ((2019, "confirmation"), (2020, "retrospective")):
        frame = pd.DataFrame({"origin_time": pd.date_range(f"{year}-01-01", periods=200, freq="h")})
        frame["target_time"] = frame.origin_time + pd.Timedelta(hours=1)
        frame["target_next_hour"] = np.arange(200) + 100.0
        for i, name in enumerate(names):
            predicted[period, name] = prediction_frame(frame, frame.target_next_hour + i, name)
    result = compare_model_pairs(
        predicted, names, {"bootstrap_blocks": [24, 168], "bootstrap_repeats": 50, "seed": 42}
    )
    assert len(result) == 12 and result.family_comparisons.eq(3).all()
    for row in result.itertuples():
        assert row.difference == names.index(row.A) - names.index(row.B)
        assert row.adjusted_low <= row.ci_low <= row.ci_high <= row.adjusted_high


def test_candidate_search_saves_scores_without_hourly_predictions(tmp_path):
    from run_expanded_study import screen_features, select_finalists
    from src.train_baseline import metrics

    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, 2)), columns=["x", "z"])
    frame["target_next_hour"] = 2 * frame.x + 5 * frame.z
    train, valid = frame.iloc[:80], frame.iloc[80:]
    settings = {
        "models": {"Ridge": [{"alpha": 1.0}, {"alpha": 100.0}]},
        "feature_screen_model": "Ridge",
        "screen_years": [2017, 2018],
    }
    bundles = {"baseline": ["x"], "all_features": ["x", "z"]}

    def evaluate(name, parameter, columns, year):
        model = make_model(name, settings["models"][name][parameter], settings)
        model.fit(train[columns], train.target_next_hour)
        predicted = model.predict(valid[columns])
        return {
            **metrics(valid.target_next_hour, predicted),
            "fit_seconds": 0,
            "n_features": len(columns),
        }

    shared = screen_features(evaluate, bundles, {"extra": ["z"]}, settings, tmp_path)
    ranking, finalists, winner = select_finalists(evaluate, bundles, shared, settings, tmp_path)
    assert shared == "all_features" and winner == "Ridge"
    expected = evaluate("Ridge", 0, bundles[shared], 2017)
    assert np.isclose(finalists.iloc[0].MAE, expected["MAE"])
    assert len(read_artifact_csv(tmp_path / "selection/model_folds.csv")) == 8
    assert not (tmp_path / "predictions").exists()
    assert not (tmp_path / "models").exists()


def longer_data(hourly_data):
    d = pd.concat([hourly_data] * 3, ignore_index=True)
    d["datetime"] = pd.date_range("2016-11-01", periods=len(d), freq="h")
    d["nat_demand"] = np.arange(len(d), dtype=float) + 500
    return d


def test_extended_seasonal_target_alignment(hourly_data, cfg):
    raw = longer_data(hourly_data)
    f, groups, bundles = build_expanded_features(raw, cfg)
    row = f.iloc[0]
    i = raw.index[raw.datetime.eq(row.origin_time)][0]
    expected = [raw.nat_demand[i - (168 * k - 1)] for k in (1, 2, 3, 4)]
    assert row.same_hour_weekly_mean == pytest.approx(np.mean(expected))
    assert row.history_mean_168 == raw.nat_demand.iloc[i - 167 : i + 1].mean()
    assert len(groups) == 8 and len(bundles) == 14
    assert len(bundles["all_features"]) == len(set(bundles["all_features"]))


def test_all_expanded_inputs_invariant_to_future(hourly_data, cfg):
    raw = longer_data(hourly_data)
    f, _, bundles = build_expanded_features(raw, cfg)
    origin = f.origin_time.iloc[30]
    altered = raw.copy()
    for col in altered.columns.difference(["datetime"]):
        altered.loc[altered.datetime > origin, col] = 10000
    changed, _, _ = build_expanded_features(altered, cfg)
    pd.testing.assert_series_equal(
        f.set_index("origin_time").loc[origin, bundles["all_features"]],
        changed.set_index("origin_time").loc[origin, bundles["all_features"]],
    )


def test_no_weather_bundle_excludes_derived_weather(hourly_data, cfg):
    _, groups, bundles = build_expanded_features(longer_data(hourly_data), cfg)
    forbidden = set(
        groups["regional_weather"]
        + groups["weather_interactions"]
        + ["temperature", "humidity", "wind"]
    )
    assert not forbidden.intersection(bundles["without_weather"])


def test_temporal_split_gap_excludes_future_labels():
    times = pd.date_range("2016-12-20", periods=1000, freq="h")
    d = pd.DataFrame({"target_time": times, "origin_time": times - pd.Timedelta(hours=1)})
    train, valid = temporal_split(d, 2017, 24)
    assert train.target_time.max() < pd.Timestamp("2016-12-31")
    assert valid.target_time.min() == pd.Timestamp("2017-01-01")


def test_candidate_ranking_weights_observations():
    d = pd.DataFrame(
        [
            {
                "model": "Ridge",
                "feature_set": "baseline",
                "parameter_id": 0,
                "n": 1,
                "MAE": 100.0,
                "RMSE": 100.0,
                "fit_seconds": 1.0,
                "n_features": 2,
            },
            {
                "model": "Ridge",
                "feature_set": "baseline",
                "parameter_id": 0,
                "n": 99,
                "MAE": 0.0,
                "RMSE": 0.0,
                "fit_seconds": 1.0,
                "n_features": 2,
            },
        ]
    )
    row = rank_candidates(d).iloc[0]
    assert row.MAE == 1 and row.RMSE == 10


def test_adjusted_intervals_enclose_nominal_interval():
    y = np.zeros(500)
    a = np.arange(500) % 17
    result = paired_intervals(y, a, y, 24, 500, 42, 4)
    assert (
        result["adjusted_low"] <= result["ci_low"] <= result["ci_high"] <= result["adjusted_high"]
    )


def test_histogram_boosting_has_no_random_early_stopping():
    model = make_model(
        "HistGBR", {"early_stopping": False, "max_iter": 10}, {"seed": 42, "workers": 1}
    )
    assert model.early_stopping is False


def test_saved_features_preserve_float64_split_boundaries(tmp_path):
    values = np.r_[
        np.nextafter(1.0, 0.0), 1.0, np.nextafter(1.0, 2.0), 0.3066101993807098, 2.0134463526192805
    ]
    path = tmp_path / "boundaries.csv"
    write_csv(path, pd.DataFrame({"feature": values}))
    restored = read_artifact_csv(path).feature.to_numpy()
    np.testing.assert_array_equal(restored, values)
    np.testing.assert_array_equal(restored > 1.0, values > 1.0)
