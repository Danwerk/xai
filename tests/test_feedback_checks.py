from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.correction import adaptive_correction, past_average
from src.evaluation import PRIMARY
from src.study import select_decay, variant_specs


@pytest.mark.parametrize("by_hour", [False, True])
@pytest.mark.parametrize("min_points", [1, 2])
def test_decay_matches_independent_clock_weighted_average(by_hour, min_points):
    offsets = [0, 1, 5, 24, 25, 72, 168, 169, 190, 400, 401]
    times = pd.Timestamp("2024-01-01") + pd.to_timedelta(offsets, unit="h")
    frame = pd.DataFrame({"target_time": times, "hour": times.hour})
    values = np.array([[i, (-1) ** i * (i + 1)] for i in range(len(times))], dtype=float)
    values[2, 1] = np.nan
    expected = np.full(values.shape, np.nan)
    for i, current in enumerate(times):
        mask = (times < current) & (times >= current - pd.Timedelta(hours=168))
        if by_hour:
            mask &= times.hour == current.hour
        for j in range(values.shape[1]):
            selected = mask & np.isfinite(values[:, j])
            if selected.sum() >= min_points:
                age = (current - times[selected]) / pd.Timedelta(hours=1)
                expected[i, j] = np.average(values[selected, j], weights=2.0 ** (-age / 24))
    actual = past_average(values, frame, 168, by_hour, min_points, half_life_hours=24)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)


def test_recent_credit_has_more_weight_and_current_credit_is_excluded():
    times = pd.date_range("2024-01-01", periods=3, freq="h")
    frame = pd.DataFrame({"target_time": times, "hour": times.hour})
    values = np.array([[10.0], [-10.0], [999999.0]])
    uniform = past_average(values, frame, 24, False, 1)
    decayed = past_average(values, frame, 24, False, 1, half_life_hours=1)
    assert uniform.iloc[2, 0] == 0
    assert decayed.iloc[2, 0] == pytest.approx(-10 / 3)
    with pytest.raises(ValueError):
        past_average(values, frame, 24, False, 1, half_life_hours=0)


@pytest.mark.parametrize("loss", ["squared", "absolute"])
@pytest.mark.parametrize("half_life", [None, 24])
@pytest.mark.parametrize("by_hour", [False, True])
def test_variants_are_prefix_causal_and_reconstruct(loss, half_life, by_hour):
    rng = np.random.default_rng(19)
    times = pd.date_range("2024-01-01", periods=400, freq="h")
    frame = pd.DataFrame(
        {
            "target_time": times,
            "origin_time": times - pd.Timedelta(hours=1),
            "hour": times.hour,
            "base_prediction": 100.0,
            "target_next_hour": 100 + rng.normal(size=len(times)),
        }
    )
    components = pd.DataFrame(rng.normal(size=(len(times), 3)), columns=list("abc"))
    setting = {"window_hours": 168, "strength": 1, "by_hour": by_hour}
    feedback = {"min_history_points": 3, "warmup_hours": 72}
    kwargs = {"credit_loss": loss, "half_life_hours": half_life}
    predicted, effective, weights = adaptive_correction(
        frame, components, setting, "credit", feedback, **kwargs
    )
    np.testing.assert_allclose(predicted, frame.base_prediction + effective.sum(axis=1))
    assert weights.ge(0).all().all() and weights.le(1).all().all()
    assert weights.iloc[:72].eq(1).all().all()
    prefix, _, _ = adaptive_correction(
        frame.iloc[:201], components.iloc[:201], setting, "credit", feedback, **kwargs
    )
    np.testing.assert_allclose(predicted[:201], prefix)
    changed = frame.copy()
    changed.loc[200:, "target_next_hour"] = 1e9
    after, _, _ = adaptive_correction(changed, components, setting, "credit", feedback, **kwargs)
    np.testing.assert_allclose(predicted[:201], after[:201])


def development_scores(cfg):
    rows = []
    maes = {"residual": 10, PRIMARY: 9.7, "decay_24": 9.6, "decay_168": 9.5, "decay_672": 9.4}
    for year in cfg["datasets"]["panama"]["development_years"]:
        for scope in ("overall", "rapid_increase", "rapid_decrease"):
            for method, mae in maes.items():
                rows.append(
                    {
                        "year": year,
                        "scope": scope,
                        "method": method,
                        "MAE": mae,
                        "stage": "development",
                    }
                )
    return pd.DataFrame(rows)


def test_decay_selection_is_development_only_guarded_and_uniform_on_ties(cfg):
    scores = development_scores(cfg)
    scores.loc[scores.method.eq("decay_672") & scores.scope.eq("rapid_increase"), "MAE"] = 12
    assert select_decay(scores, cfg)["half_life_hours"] == 168
    ties = scores.copy()
    ties.loc[~ties.method.eq("residual"), "MAE"] = 9.7
    assert select_decay(ties, cfg)["memory"] == "uniform"
    with pytest.raises(ValueError):
        select_decay(scores.assign(stage="final"), cfg)
    with pytest.raises(ValueError):
        select_decay(scores.assign(year=2020), cfg)
    with pytest.raises(ValueError):
        select_decay(scores.iloc[:-1], cfg)
    with pytest.raises(ValueError):
        select_decay(scores[~scores.scope.eq("rapid_decrease")], cfg)


def test_specs_isolate_mechanisms_and_do_not_mutate_primary(cfg):
    primary = {"window_hours": 672, "strength": 1.0, "by_hour": True}
    before = deepcopy(primary)
    specs = variant_specs(primary, cfg, half_life=168)
    assert specs["mae_credit"]["setting"] == primary
    assert specs["mae_credit"]["half_life"] is None
    assert specs["mae_credit"]["credit_loss"] == "absolute"
    assert specs["recency_credit"]["setting"] == primary
    assert specs["recency_credit"]["credit_loss"] == "squared"
    assert specs["recency_credit"]["half_life"] == 168
    assert len([name for name in specs if name.startswith("sensitivity_")]) == 9
    assert all(spec["setting"]["by_hour"] for spec in specs.values())
    specs["mae_credit"]["setting"]["strength"] = 0
    assert primary == before
    assert specs["recency_credit"]["setting"] == before
    identity = variant_specs(None, cfg)
    assert identity["recency_credit"]["setting"] is identity["mae_credit"]["setting"] is None
