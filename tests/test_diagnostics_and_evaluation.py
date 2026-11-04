import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from src.evaluation import paired_block_bootstrap, audit_card
from src.historical_cases import find_similar_cases, fit_similarity_scaler
from src.refinement_experiments import (
    freeze_decisions,
    specialist_forecast,
    residual_history,
    specialist_variants,
    select_specialist,
    specialist_intervals,
    develop_specialist,
    evaluate_specialist,
)
from src.common import write_json, read_json
from src.shap_explainer import explain_prediction
from src.train_baseline import metrics, prediction_frame
from src.card_generator import diagnostic_hypothesis
from src.feature_engineering import build_features
from src.scenario_detector import fit_thresholds, detect_scenarios
from tools.check_results import check_predictions


@pytest.mark.parametrize("changed", ["truth", "sign", "time"])
def test_result_check_rejects_wrong_data_even_when_metrics_match(changed):
    frame = pd.DataFrame(
        {
            "origin_time": pd.date_range("2019-01-01", periods=4, freq="h"),
            "target_next_hour": [10.0, 12.0, 14.0, 16.0],
        }
    )
    frame["target_time"] = frame.origin_time + pd.Timedelta(hours=1)
    predicted = prediction_frame(frame, [11.0, 11.0, 15.0, 15.0], "example")
    scores = pd.Series(metrics(frame.target_next_hour, predicted.prediction))
    check_predictions(predicted, frame, scores)
    if changed == "truth":
        predicted[["prediction", "target_next_hour"]] += 5
    elif changed == "sign":
        predicted["error"] *= -1
    else:
        predicted["target_time"] += pd.Timedelta(hours=1)
    measured = metrics(predicted.target_next_hour, predicted.prediction)
    assert all(np.isclose(scores[name], measured[name]) for name in ("MAE", "RMSE", "bias"))
    with pytest.raises(AssertionError):
        check_predictions(predicted, frame, scores)


def test_specialist_uses_training_threshold_and_known_inputs(monkeypatch):
    train = pd.DataFrame({"load_ramp_1h": [0.0, 1.0, 2.0, 3.0, 4.0], "load_now": range(5)})
    frame = pd.DataFrame(
        {
            "origin_time": pd.date_range("2019-01-01", periods=3, freq="h"),
            "target_next_hour": [100.0, 200.0, 300.0],
            "load_now": [100.0, 200.0, 300.0],
            "load_ramp_1h": [2.0, 3.0, 4.0],
            "actual_ramp": [1000.0, -1000.0, 0.0],
        }
    )
    frame["target_time"] = frame.origin_time + pd.Timedelta(hours=1)
    baseline = prediction_frame(frame, [10.0, 10.0, 10.0], "GBR")
    cfg = {"specialist_quantile": 0.75, "baseline_features": ["load_now"]}

    class Expert:
        def predict(self, inputs):
            assert len(inputs)
            return inputs.load_now + 1000

    def fit(training, features, config):
        assert training.index.tolist() == [3, 4]
        assert features == ["load_now"]
        return Expert()

    monkeypatch.setattr("src.refinement_experiments.fit_gbr", fit)
    result, details = specialist_forecast(train, frame, baseline, cfg)
    assert details == {"gate_threshold": 3.0, "expert_train_n": 2}
    assert result.prediction.tolist() == [10.0, 1200.0, 1300.0]
    assert result.use_specialist.tolist() == [False, True, True]
    changed = frame.copy()
    changed[["target_next_hour", "actual_ramp"]] += 100000
    repeated, _ = specialist_forecast(train, changed, baseline, cfg)
    pd.testing.assert_series_equal(result.prediction, repeated.prediction)
    pd.testing.assert_series_equal(result.use_specialist, repeated.use_specialist)
    frame["load_ramp_1h"] = 0.0
    inactive, _ = specialist_forecast(train, frame, baseline, cfg)
    assert not inactive.use_specialist.any()
    pd.testing.assert_series_equal(inactive.prediction, baseline.prediction)


def test_residual_history_uses_only_earlier_years(monkeypatch):
    frame = pd.DataFrame(
        {"target_time": pd.to_datetime([f"{year}-06-01" for year in (2015, 2016, 2017)])}
    )
    frame["origin_time"] = frame.target_time - pd.Timedelta(hours=1)
    frame["load_now"] = [10.0, 20.0, 30.0]
    frame["target_next_hour"] = [11.0, 22.0, 33.0]
    trained = []

    class Model:
        def __init__(self, last_time):
            self.last_time = last_time

        def predict(self, inputs):
            assert (inputs.origin_time >= self.last_time).all()
            return inputs.load_now + 1

    def fit(train, columns, cfg):
        trained.append(train.target_time.dt.year.tolist())
        return Model(train.target_time.max())

    monkeypatch.setattr("src.refinement_experiments.fit_gbr", fit)
    result = residual_history(frame, {"baseline_features": ["load_now", "origin_time"]})
    assert trained == [[2015], [2015, 2016]]
    assert result.target_time.dt.year.tolist() == [2016, 2017]
    assert result.correction.tolist() == [1.0, 2.0]


def test_all_specialist_predictions_ignore_future_outcomes(hourly_data, cfg):
    cfg = dict(cfg, gbr={**cfg["gbr"], "n_estimators": 3})
    cfg["specialist"] = {
        **cfg["specialist"],
        "model": {**cfg["specialist"]["model"], "max_iter": 3},
    }
    frame = build_features(hourly_data, cfg)
    train, hold = frame.iloc[:150].copy(), frame.iloc[150:].reset_index(drop=True)
    history = train.assign(
        base_prediction=train.load_now,
        base_ramp=0.0,
        correction=train.target_next_hour - train.load_now,
    )
    base = prediction_frame(hold, hold.load_now.to_numpy(), "GBR")
    before = specialist_variants(train, hold, base, history, 0.5, cfg)
    hold[
        ["target_next_hour", "actual_ramp", "actual_temperature", "holiday_retrospective"]
    ] += 10000
    after = specialist_variants(train, hold, base, history, 0.5, cfg)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.parametrize("worse_growth", [False, True])
def test_specialist_selection_rejects_hidden_growth_rmse_increase(worse_growth):
    rows = []
    for year in (2018, 2019):
        for scope in ("overall", "rapid_increase"):
            for model in ("GBR", "correction"):
                improved = model == "correction"
                rmse = 7.0 if improved else 10.0
                if improved and worse_growth and scope == "rapid_increase" and year == 2019:
                    rmse = 11.0
                rows.append(
                    dict(
                        year=year,
                        scope=scope,
                        model=model,
                        stage="development",
                        MAE=4.0 if improved else 8.0,
                        RMSE=rmse,
                    )
                )
    selected, _ = select_specialist(pd.DataFrame(rows))
    assert selected == ("GBR" if worse_growth else "correction")


def test_specialist_selection_rejects_final_scores():
    scores = pd.DataFrame({"stage": ["development", "final"]})
    with pytest.raises(ValueError, match="arendusperioodi"):
        select_specialist(scores)


def test_specialist_development_rejects_test_period(cfg):
    frame = pd.DataFrame({"target_time": pd.to_datetime([cfg["selection_end"]])})
    with pytest.raises(ValueError, match="lõpphindamise perioodi"):
        develop_specialist(None, frame, None, cfg)


def test_frozen_specialist_ignores_future_outcomes_and_later_settings(
    hourly_data, cfg, tmp_path, monkeypatch
):
    cfg = dict(
        cfg, gbr={**cfg["gbr"], "n_estimators": 3}, bootstrap_repeats=20, bootstrap_blocks=[24]
    )
    cfg["specialist"] = {
        **cfg["specialist"],
        "model": {**cfg["specialist"]["model"], "max_iter": 3},
    }
    template = build_features(hourly_data, cfg).iloc[:60].copy()

    def year_frame(year):
        frame = template.copy()
        frame["target_time"] = pd.date_range(f"{year}-01-01", periods=len(frame), freq="h")
        frame["origin_time"] = frame.target_time - pd.Timedelta(hours=1)
        return frame

    training = pd.concat([year_frame(y) for y in (2015, 2016, 2017)], ignore_index=True)
    development = pd.concat([year_frame(y) for y in (2018, 2019)], ignore_index=True)
    base = prediction_frame(development, development.load_now.to_numpy() + 0.2, "GBR")
    _, decision = develop_specialist(training, development, base, cfg)
    write_json(tmp_path / "manifest.json", {"specialist_experiment": decision})
    cfg["specialist"]["model"]["max_iter"] = 99
    frozen = read_json(tmp_path / "manifest.json")["specialist_experiment"]
    assert decision["settings"]["specialist"]["model"]["max_iter"] == 3
    assert frozen["evaluation_kind"] == "retrospective"

    def forbidden_selection(*args):
        pytest.fail("Lõpphindamine ei tohi mudelit uuesti valida.")

    monkeypatch.setattr("src.refinement_experiments.select_specialist", forbidden_selection)
    training = pd.concat([training, development], ignore_index=True)
    future = year_frame(2020)
    base = prediction_frame(future, future.load_now.to_numpy() + 0.2, "GBR")
    before_scores, before = evaluate_specialist(training, future, base, frozen)
    future[
        ["target_next_hour", "actual_ramp", "actual_temperature", "holiday_retrospective"]
    ] += 1000
    after_scores, after = evaluate_specialist(training, future, base, frozen)
    pd.testing.assert_frame_equal(
        before.drop(columns=["target_next_hour", "error", "abs_error"]),
        after.drop(columns=["target_next_hour", "error", "abs_error"]),
    )
    assert not np.allclose(before_scores.MAE, after_scores.MAE)
    assert frozen == decision


def test_specialist_intervals_cover_mae_and_rmse_with_the_same_sign():
    actual = np.arange(240, dtype=float)
    mask = np.arange(240) % 4 == 0
    result = specialist_intervals(
        actual, actual + 1, actual + 5, mask, {"seed": 42, "bootstrap_repeats": 100}, 24
    )
    for name in ("adjusted_low", "adjusted_high", "rmse_adjusted_low", "rmse_adjusted_high"):
        assert result[name] == pytest.approx(-4.0)
    assert result["valid_draws"] == 100


def test_metric_sign_and_mape():
    m = metrics([10, 20], [12, 16])
    assert m["MAE"] == 3
    assert m["bias"] == -1
    assert m["RMSE"] == pytest.approx(np.sqrt(10))
    assert m["MAPE"] == 20
    assert metrics([0, 1], [1, 1])["MAPE"] is None


def test_block_bootstrap_paired_sign_and_identical_models():
    y = np.arange(400) + 100
    same = paired_block_bootstrap(y, y + 3, y + 3, 24, 200)
    assert same["ci_low"] == same["ci_high"] == 0
    different = paired_block_bootstrap(y, y + 1, y + 4, 24, 200)
    assert different["MAE_difference_A_minus_B"] == -3
    assert different["ci_high"] == -3


def test_bootstrap_reproducible_and_sensitive_to_blocks():
    y = np.zeros(480)
    a = np.repeat([0, 10, 0, 10], 120)
    b = np.zeros(480)
    x = paired_block_bootstrap(y, a, b, 24, 400, 42)
    assert x == paired_block_bootstrap(y, a, b, 24, 400, 42)
    long = paired_block_bootstrap(y, a, b, 120, 400, 42)
    assert long["ci_high"] - long["ci_low"] > x["ci_high"] - x["ci_low"]


def test_shap_reconstructs_prediction():
    x = pd.DataFrame({"a": np.arange(100), "b": np.sin(np.arange(100))})
    model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42).fit(
        x, x.a + x.b
    )
    values, base = explain_prediction(model, x.iloc[:8])
    np.testing.assert_allclose(base + values.sum(axis=1), model.predict(x.iloc[:8]), atol=1e-6)


def test_historical_cases_exclude_future_and_unavailable_target():
    d = pd.DataFrame(
        {
            "origin_time": pd.date_range("2018-01-01", periods=5, freq="h"),
            "load_now": [1.0, 2, 3, 4, 5],
            "prediction": [1.0, 2, 3, 4, 5],
            "target_next_hour": [1.0, 2, 3, 4, 5],
            "error": 0.0,
        }
    )
    d["target_time"] = d.origin_time + pd.Timedelta(hours=1)
    scaler = fit_similarity_scaler(d.iloc[:2], ["load_now"])
    result = find_similar_cases(d.iloc[2], d, ["load_now"], scaler)
    assert len(result) == 2
    assert result.target_time.max() <= d.origin_time.iloc[2]


def test_hypothesis_abstains_when_insufficient():
    cohort = pd.DataFrame({"scenario": ["rapid_increase"], "n": [3], "MAE_minus_ordinary": [20.0]})
    result = diagnostic_hypothesis(["rapid_increase"], cohort, [{"feature": "load_now"}], 5)
    assert result["hypothesis"] is None and result["experiment"] is None


def test_equal_budget_and_deterministic_ties(cfg):
    rows = [
        {"case": case, "group": g, "manual_order": i, "residual_score": 1.0, "xai_score": 1.0}
        for case, groups in cfg["case_studies"].items()
        for i, g in enumerate(groups)
    ]
    decisions = freeze_decisions(pd.DataFrame(rows), cfg)
    for method in decisions:
        assert sum(map(len, decisions[method].values())) == 4
        for case, groups in decisions[method].items():
            assert len(groups) == 2 and set(groups) <= set(cfg["case_studies"][case])
    assert decisions["Residual"] == decisions["XAI"]


def test_card_audit_detects_incorrect_numbers_and_future_evidence():
    card = {
        "target_time": "2018-01-01 01:00",
        "origin_time": "2018-01-01",
        "prediction": 12.0,
        "actual": 10.0,
        "error": 2.0,
        "absolute_error": 2.0,
        "base_value": 11.0,
        "shap_contributions": [{"shap": 1.0, "interpretation": "Suurendab prognoosi"}],
        "historical_cases": [],
        "diagnosis": {"hypothesis": None, "experiment": None},
        "evidence_sources": ["a", "b", "c"],
        "limitations": ["SHAP ei selgita veapõhjust."],
    }
    assert audit_card(card)["all_pass"]
    card["error"] = 3
    assert not audit_card(card)["numeric_error_correct"]
    card["historical_cases"] = [{"target_time": "2018-01-02"}]
    assert not audit_card(card)["history_available"]


def test_scenario_thresholds_frozen_and_retrospective(hourly_data, cfg):
    f = build_features(hourly_data, cfg)
    train, future = f.iloc[:100].copy(), f.iloc[100:].copy()
    thresholds = fit_thresholds(train, pd.DataFrame({"abs_error": np.arange(100)}))
    snapshot = dict(thresholds)
    before = detect_scenarios(future, thresholds)
    future["target_next_hour"] = 1
    after = detect_scenarios(future, thresholds)
    assert thresholds == snapshot
    assert before.peak.all() and not after.peak.any()
    pd.testing.assert_series_equal(before.known_ramp, after.known_ramp)
    pd.testing.assert_series_equal(before.known_heat, after.known_heat)


def test_residual_and_manual_choices_ignore_xai_scores(cfg):
    evidence = pd.DataFrame(
        [
            {"case": case, "group": group, "manual_order": i, "residual_score": i, "xai_score": i}
            for case, groups in cfg["case_studies"].items()
            for i, group in enumerate(groups)
        ]
    )
    before = freeze_decisions(evidence, cfg)
    evidence["xai_score"] = -evidence.xai_score
    after = freeze_decisions(evidence, cfg)
    assert before["Manual"] == after["Manual"]
    assert before["Residual"] == after["Residual"]
    assert before["XAI"] != after["XAI"]


@pytest.mark.parametrize("loss", ["absolute", "squared"])
def test_repair_credits_are_exact_symmetric_and_have_dummy_player(loss):
    from src.shap_explainer import repair_gain_attributions

    # Two interchangeable corrections, a cancelling correction and an inactive module.
    correction = np.array([[2.0, 2.0, -1.0, 0.0], [-2.0, -2.0, 1.0, 0.0]])
    actual = np.array([5.0, -5.0])
    credit = repair_gain_attributions(actual, np.zeros(2), correction, loss)
    np.testing.assert_allclose(credit[:, 0], credit[:, 1])
    np.testing.assert_allclose(credit[:, 3], 0)
    power = 1 if loss == "absolute" else 2
    gain = abs(actual) ** power - abs(actual - correction.sum(axis=1)) ** power
    np.testing.assert_allclose(credit.sum(axis=1), gain)
    order = [2, 0, 3, 1]
    permuted = repair_gain_attributions(actual, np.zeros(2), correction[:, order], loss)
    np.testing.assert_allclose(permuted, credit[:, order])


def test_repair_credit_detects_overshoot_and_cancellation():
    from src.shap_explainer import repair_gain_attributions

    credit = repair_gain_attributions([10.0], [0.0], [[8.0, 7.0]], "squared")
    np.testing.assert_allclose(credit, [[40.0, 35.0]])
    # A helpful direction can still overshoot enough to make the complete repair harmful.
    assert repair_gain_attributions([10.0], [0.0], [[30.0]], "squared")[0, 0] < 0
    np.testing.assert_allclose(repair_gain_attributions([10.0], [0.0], [[3.0, -3.0]]).sum(), 0)


def test_error_memory_cannot_use_current_or_future_target():
    from src.refinement_experiments import known_error_features

    times = pd.date_range("2018-01-01", periods=200, freq="h")
    frame = pd.DataFrame(
        {
            "target_time": times,
            "origin_time": times - pd.Timedelta(hours=1),
            "target_next_hour": np.arange(200.0),
            "base_prediction": np.zeros(200),
        }
    )
    before = known_error_features(frame)
    frame.loc[100:, "target_next_hour"] = 1e9
    after = known_error_features(frame)
    pd.testing.assert_frame_equal(before.iloc[:101], after.iloc[:101])
    assert before.loc[100, "known_error_24"] == 76
    assert before.loc[100, "known_error_168"] == 0
    assert after.loc[101, "known_error_1"] == 1e9


def test_error_memory_uses_timestamps_across_gaps():
    from src.refinement_experiments import known_error_features

    times = pd.to_datetime(["2018-01-01 00:00", "2018-01-01 02:00"])
    frame = pd.DataFrame(
        {
            "target_time": times,
            "origin_time": times - pd.Timedelta(hours=1),
            "target_next_hour": [100.0, 200.0],
            "base_prediction": [0.0, 0.0],
        }
    )
    assert known_error_features(frame).loc[1, "known_error_1"] == 0


def test_attribution_layer_reconstructs_and_matches_affine_raw_coordinates(cfg):
    from src.refinement_experiments import fit_attribution_layer

    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        rng.normal(size=(180, len(cfg["baseline_features"]))), columns=cfg["baseline_features"]
    )
    frame["target_time"] = pd.date_range("2018-01-01", periods=len(frame), freq="h")
    frame["origin_time"] = frame.target_time - pd.Timedelta(hours=1)
    frame["base_prediction"] = 10.0
    frame["target_next_hour"] = 10 + frame.load_now * np.sin(
        2 * np.pi * frame.target_time.dt.hour / 24
    )
    for c in cfg["baseline_features"]:
        frame[f"phi_{c}"] = frame[c] - 2.0
    training, valid = frame.iloc[:120], frame.iloc[120:]
    setting = {"space": "shap", "memory": False, "harmonics": 2, "alpha": 1.0}
    predicted, components, _ = fit_attribution_layer(
        training, valid, setting, cfg["baseline_features"], cfg
    )
    np.testing.assert_allclose(predicted, 10 + components.sum(axis=1))
    # With no clock interactions, standardized affine coordinates carry identical information.
    setting["harmonics"] = 0
    a, _, _ = fit_attribution_layer(training, valid, setting, cfg["baseline_features"], cfg)
    setting["space"] = "raw"
    b, _, _ = fit_attribution_layer(training, valid, setting, cfg["baseline_features"], cfg)
    np.testing.assert_allclose(a, b, atol=1e-8)
    with pytest.raises(ValueError, match="unavailable"):
        fit_attribution_layer(valid, training, setting, cfg["baseline_features"], cfg)


def test_attribution_selection_rejects_final_results():
    from src.refinement_experiments import select_attribution_layers

    with pytest.raises(ValueError, match="final"):
        select_attribution_layers(pd.DataFrame({"stage": ["final"]}), {})


@pytest.mark.parametrize("by_hour", [True, False])
def test_repair_feedback_is_causal_and_reconstructs(by_hour):
    from src.refinement_experiments import repair_feedback

    rng = np.random.default_rng(42)
    times = pd.date_range("2018-01-01", periods=300, freq="h")
    frame = pd.DataFrame(
        {
            "target_time": times,
            "target_next_hour": rng.normal(size=300),
            "base_prediction": np.zeros(300),
        }
    )
    components = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    before, effective, weights = repair_feedback(frame, components, 168, by_hour)
    np.testing.assert_allclose(before, effective.sum(axis=1))
    assert weights.ge(0).all().all() and weights.le(1).all().all()
    frame.loc[200:, "target_next_hour"] = 1e9
    after, _, _ = repair_feedback(frame, components, 168, by_hour)
    np.testing.assert_allclose(before[:201], after[:201])


def test_transfer_intervals_missing_values_and_local_clock(tmp_path):
    from src.refinement_experiments import transfer_load_frame

    times = pd.date_range("2024-01-01", periods=4000, freq="15min", tz="UTC")
    raw = pd.DataFrame({"datetime": times, "consumption": 1000 + np.arange(4000) // 4})
    raw = raw.drop(index=3601)  # One quarter is absent; that whole hour is unavailable.
    path = tmp_path / "load.csv"
    raw.to_csv(path, index=False)
    frame, bundles, quality = transfer_load_frame(path)
    assert quality["imputed_values"] == 0 and quality["missing_hourly_load"] == 1
    assert "temperature" not in frame and "holiday_known" not in frame
    assert "target_next_hour" not in bundles["history_calendar"]
    query = frame.set_index("origin_time").loc[pd.Timestamp("2024-01-29 04:00")]
    assert query.load_now == 1000 + 675
    assert query.target_next_hour == 1000 + 676
    assert query.hour == 7  # Target 05 UTC is 07 in Tallinn in January.
    unavailable = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=901)
    assert not frame.target_time.eq(unavailable).any()


def test_clock_bootstrap_preserves_missing_hours():
    from src.evaluation import clock_loss_interval, scenario_loss_interval

    times = pd.date_range("2024-01-01", periods=100, freq="h")
    difference = np.arange(100.0)
    mask = np.ones(100, dtype=bool)
    args = dict(block=24, repeats=100, seed=42, comparisons=3)
    assert clock_loss_interval(times, difference, mask, **args) == scenario_loss_interval(
        difference, mask, **args
    )
    keep = np.ones(100, dtype=bool)
    keep[40:60] = False
    padded = difference.copy()
    padded[~keep] = 0
    assert clock_loss_interval(
        times[keep], difference[keep], mask[keep], **args
    ) == scenario_loss_interval(padded, keep, **args)
