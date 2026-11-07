from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.evaluation import METHODS
from src.feature_engineering import feature_columns, split_year
from src.models import residual_forecasts
from src.study import develop, oof_history, policy_forecasts, select_policies, validate_config, variant_specs


def test_selection_uses_complete_development_only_and_identity_on_ties(cfg):
    rows = []
    for year in (2018, 2019):
        for scope in ("overall", "rapid_increase", "rapid_decrease"):
            for name, mae in (("residual", 10), ("better", 9), ("tie", 10)):
                rows.append(dict(year=year, scope=scope, method=name, MAE=mae, stage="development"))
    scores = pd.DataFrame(rows)
    options = {"one": {"better": {"weight": 0.5}}, "two": {"tie": {"weight": 1}}}
    choices = select_policies(scores, options, cfg)
    assert choices["one"]["setting"] == {"weight": 0.5}
    assert choices["two"]["candidate"] == "identity"
    with pytest.raises(ValueError):
        select_policies(scores.assign(stage="final"), options, cfg)
    with pytest.raises(ValueError):
        select_policies(scores.drop(scores.index[-2]), options, cfg)


def test_protocol_rejects_test_selection_and_random_early_stopping(cfg):
    cfg["datasets"]["panama"]["development_years"].append(2020)
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg["datasets"]["panama"]["development_years"].pop()
    cfg["residual_model"]["early_stopping"] = True
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_unknown_base_family_is_rejected_before_training(cfg):
    cfg["base_model_family"] = "random_forest"
    with pytest.raises(ValueError, match="base_model_family"):
        validate_config(cfg)


@pytest.mark.parametrize("base_family", ["gbr", "histgbr"])
def test_in_memory_training_and_all_policies_write_no_files(cfg, tmp_path, monkeypatch, base_family):
    monkeypatch.chdir(tmp_path)
    cfg = deepcopy(cfg)
    cfg["base_model_family"] = base_family
    cfg["base_model"].update(n_estimators=3, min_samples_leaf=5)
    cfg["residual_model"].update(max_iter=4, min_samples_leaf=5)
    cfg["selection"].update(feedback_windows_hours=[168], feedback_strengths=[1.0])
    cfg["workers"] = 1
    rng = np.random.default_rng(4)
    base_columns, columns = feature_columns("panama")
    parts = []
    for year in range(2015, 2021):
        times = pd.date_range(f"{year}-01-01", periods=96, freq="h")
        frame = pd.DataFrame(rng.normal(size=(96, len(columns))), columns=columns)
        frame["load_now"] = 100 + rng.normal(size=96) * 10
        frame["target_next_hour"] = frame.load_now + rng.normal(size=96) * 5
        frame["actual_ramp"] = frame.target_next_hour - frame.load_now
        frame["target_time"], frame["origin_time"], frame["hour"] = (
            times, times - pd.Timedelta(hours=1), times.hour
        )
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)
    thresholds = {"increase": 2.0, "decrease": -2.0}
    with threadpool_limits(limits=1):
        history = oof_history(data, 2020, base_columns, cfg)
        scores, choices, decay = develop(history, columns, thresholds, cfg)
        assert set(scores.year) == {2018, 2019}
        assert scores.stage.eq("development").all()
        before = deepcopy(choices)
        train, valid = split_year(history, 2019)
        forecasts, components, _ = residual_forecasts(train, valid, columns, cfg)
        forecasts, weights = policy_forecasts(
            valid, forecasts, components, {"choices": choices, "decay": decay}, cfg
        )
    assert choices == before
    expected = set(METHODS) | set(variant_specs(choices["credit_feedback"]["setting"], cfg))
    assert set(forecasts) == expected
    assert all(len(values) == len(valid) and np.isfinite(values).all() for values in forecasts.values())
    assert weights["credit_feedback"].ge(0).all().all()
    assert weights["credit_feedback"].le(1).all().all()
    assert not list(tmp_path.iterdir())
