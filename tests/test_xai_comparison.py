import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from src.xai_comparison import (
    EXPLAINERS, compare_explainers, component_forecast, explain_components, training_background,
)

OPTIONS = {"background_size": 8, "lime_samples": 128, "lime_kernel_width": 0.75,
           "lime_ridge_alpha": 1.0, "batch_rows": 32}
SETTING = {"window_hours": 168, "by_hour": True, "strength": 1.0}
SIMPLE = {"window_hours": 168, "strength": 0.25}
FEEDBACK = {"min_history_points": 3, "warmup_hours": 72}


@pytest.fixture
def fitted():
    rng = np.random.default_rng(7)
    times = pd.date_range("2018-01-01", periods=480, freq="h")
    frame = pd.DataFrame({"load_now": rng.normal(size=480),
                          "load_day": rng.normal(size=480),
                          "hour_sin": np.sin(2 * np.pi * times.hour / 24),
                          "target_time": times,
                          "origin_time": times - pd.Timedelta(hours=1),
                          "hour": times.hour, "base_prediction": 100.0})
    frame["target_next_hour"] = 100 + 5 * frame.load_now + frame.load_day * frame.hour_sin
    inputs = ["load_now", "load_day", "hour_sin"]
    training, valid = frame.iloc[:240].copy(), frame.iloc[240:].copy()
    with threadpool_limits(limits=1):
        model = HistGradientBoostingRegressor(
            max_iter=10, max_leaf_nodes=7, min_samples_leaf=10,
            early_stopping=False, random_state=42,
        ).fit(training[inputs], training.target_next_hour - training.base_prediction)
        valid["prediction_residual"] = valid.base_prediction + model.predict(valid[inputs])
    return model, training, valid, inputs


def test_same_model_identity_warmup_and_no_file_output(fitted, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, training, valid, inputs = fitted
    original = valid.copy(deep=True)
    with threadpool_limits(limits=1):
        result = compare_explainers(model, training, valid, inputs, SETTING, SIMPLE,
                                   FEEDBACK, options=OPTIONS)
        identity = compare_explainers(model, training, valid, inputs, None, None,
                                     FEEDBACK, options=OPTIONS)
    pd.testing.assert_frame_equal(valid, original)
    columns = result["components"]["tree_shap"].columns
    for method in EXPLAINERS:
        assert result["components"][method].columns.equals(columns)
        np.testing.assert_allclose(
            valid.base_prediction + result["components"][method].sum(axis=1)
            + result["remainders"][method], valid.prediction_residual, atol=1e-8,
        )
        np.testing.assert_allclose(identity["forecasts"][method], valid.prediction_residual)
        np.testing.assert_allclose(result["forecasts"][method][:72], valid.prediction_residual.iloc[:72])
        assert result["weights"][method].ge(0).all().all()
        assert result["weights"][method].le(1).all().all()
        assert "unallocated_remainder" not in result["weights"][method]
    assert result["background_last_target"] <= valid.origin_time.min()
    assert result["timing"].total_seconds.ge(0).all()
    assert result["timing"].n.eq(len(valid)).all()
    checks = result["checks"].set_index("method")
    assert checks.loc["tree_shap", "native_gap_max_abs"] < 1e-7
    assert checks.loc["lime", "native_gap_mae"] > 0
    assert not list(tmp_path.iterdir())


def test_background_and_feedback_do_not_use_future_outcomes(fitted):
    model, training, valid, inputs = fitted
    first = training_background(training, inputs, 16, 42)
    pd.testing.assert_frame_equal(first, training_background(training, inputs, 16, 42))
    assert first.index.isin(training.index).all() and first.columns.tolist() == inputs
    with threadpool_limits(limits=1):
        before = compare_explainers(model, training, valid, inputs, SETTING, SIMPLE,
                                    FEEDBACK, options=OPTIONS)
        changed = valid.copy()
        changed.loc[changed.index[120]:, "target_next_hour"] = 1e9
        after = compare_explainers(model, training, changed, inputs, SETTING, SIMPLE,
                                   FEEDBACK, options=OPTIONS)
    for method in ["hourly_bias", *EXPLAINERS]:
        np.testing.assert_allclose(before["forecasts"][method][:121],
                                   after["forecasts"][method][:121])
    for method in EXPLAINERS:
        pd.testing.assert_frame_equal(before["components"][method], after["components"][method])
    with pytest.raises(ValueError, match="precede"):
        compare_explainers(model, valid, training, inputs, SETTING, SIMPLE, FEEDBACK, options=OPTIONS)


def test_full_and_subset_explanations_use_same_random_draws(fitted):
    model, training, valid, inputs = fitted
    X = valid[inputs].iloc[:8]
    with threadpool_limits(limits=1):
        for method in EXPLAINERS:
            full, rest, _ = explain_components(model, X, method, training, options=OPTIONS)
            subset, subrest, _ = explain_components(model, X.iloc[[1, 4, 7]], method,
                                                   training, options=OPTIONS)
            np.testing.assert_allclose(subset, full.iloc[[1, 4, 7]], atol=1e-10)
            np.testing.assert_allclose(subrest, rest.iloc[[1, 4, 7]], atol=1e-10)


def test_lime_contribution_units_and_ablation_on_known_linear_model():
    rng = np.random.default_rng(8)
    training = pd.DataFrame({"load_now": rng.normal(20, 3, 120),
                             "load_day": rng.normal(50, 8, 120),
                             "weekend": rng.integers(0, 2, 120)})
    target = 7 + 2 * training.load_now - 3 * training.load_day + 4 * training.weekend
    model = LinearRegression().fit(training, target)
    X = training.iloc[:6].copy()
    options = dict(OPTIONS, lime_ridge_alpha=0.0, lime_samples=256)
    with threadpool_limits(limits=1):
        for method in ("lime", "feature_ablation"):
            parts, remainder, info = explain_components(model, X, method, training, options=options)
            np.testing.assert_allclose(parts.sum(axis=1), model.predict(X), atol=1e-7)
            np.testing.assert_allclose(remainder, 0, atol=1e-7)
            if method == "lime":
                assert info["lime_local_r2_mean"] > 1 - 1e-10
                np.testing.assert_allclose(parts.recent_load,
                                           2 * (X.load_now - training.load_now.mean()), atol=1e-7)
            else:
                background = training_background(training, list(X.columns), options["background_size"], 42)
                np.testing.assert_allclose(parts.recent_load,
                                           2 * (X.load_now - background.load_now.mean()), atol=1e-7)


def test_interaction_remainder_is_reported_and_held_fixed(monkeypatch):
    class Interaction:
        def predict(self, X):
            return (X.load_now * X.load_day).to_numpy()

    training = pd.DataFrame({"load_now": [0.0], "load_day": [0.0]})
    X = pd.DataFrame({"load_now": [2.0], "load_day": [3.0]})
    parts, remainder, info = explain_components(Interaction(), X, "feature_ablation",
                                               training, options=OPTIONS)
    np.testing.assert_allclose(parts.iloc[0], [6, 6, 0])
    np.testing.assert_allclose(remainder, [-6])
    assert info["native_gap_mae"] == 6
    assert info["reconstruction_max_abs"] == 0
    frame = pd.DataFrame({"base_prediction": [100.0]})

    def suppress_all(adjusted, components, *args):
        return adjusted.base_prediction.to_numpy(), components * 0, components * 0

    monkeypatch.setattr("src.xai_comparison.adaptive_correction", suppress_all)
    predicted, weights = component_forecast(frame, parts, remainder, SETTING, FEEDBACK)
    np.testing.assert_allclose(predicted, [94])
    assert "unallocated_remainder" not in weights


def test_unknown_method_and_misaligned_remainder_are_rejected(fitted):
    model, training, valid, inputs = fitted
    with pytest.raises(ValueError, match="Unknown"):
        explain_components(model, valid[inputs], "not_lime", training, options=OPTIONS)
    with pytest.raises(ValueError, match="align"):
        component_forecast(valid, valid[inputs], pd.Series([0.0]), SETTING, FEEDBACK)


def test_ablation_replaces_correlated_group_members_from_one_reference_row():
    class GroupInteraction:
        def predict(self, X):
            return (X.load_now * X.load_lag_2 + X.load_day).to_numpy()

    training = pd.DataFrame({"load_now": [1.0, 3.0], "load_lag_2": [2.0, 5.0],
                             "load_day": [10.0, 20.0]})
    X = pd.DataFrame({"load_now": [2.0], "load_lag_2": [4.0], "load_day": [30.0]})
    parts, remainder, _ = explain_components(GroupInteraction(), X, "feature_ablation",
                                             training, options=OPTIONS)
    np.testing.assert_allclose(parts.iloc[0], [-0.5, 15, 23.5])
    np.testing.assert_allclose(remainder, [0])
