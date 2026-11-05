import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

from src.feature_engineering import feature_columns
from src.compare_forecasting_models import compare_models, development_years, make_models


def development_frame():
    columns, _ = feature_columns("panama")
    parts = []
    for year, count in ((2017, 30), (2018, 24), (2019, 48), (2020, 12)):
        target = pd.date_range(f"{year}-01-01", periods=count, freq="h")
        values = np.arange(count * len(columns), dtype=float).reshape(count, -1) + 100 * (
            year - 2017
        )
        part = pd.DataFrame(values, columns=columns)
        part["target_time"], part["origin_time"] = target, target - pd.Timedelta(hours=1)
        part["target_next_hour"] = np.arange(count, dtype=float) + (year - 2017) * 20
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def test_all_models_share_train_only_preprocessing_and_no_test_selection(cfg, monkeypatch):
    import src.compare_forecasting_models as experiment

    records = []

    class Recorder:
        def fit(self, x, y):
            self.record = {"train": x.copy(), "target": np.asarray(y).copy()}
            records.append(self.record)
            return self

        def predict(self, x):
            self.record["valid"] = x.copy()
            return np.zeros(len(x))

    monkeypatch.setattr(
        experiment,
        "make_models",
        lambda *args: {name: Recorder() for name in ("Ridge", "RandomForest", "GBR", "HistGBR")},
    )
    frame = development_frame()
    summary, scores, folds = compare_models(frame, cfg)
    assert [f["train_n"] for f in folds] == [30, 54]
    assert [f["development_n"] for f in folds] == [24, 48]
    assert len(scores) == 8 and set(summary.n) == {72}
    for start in (0, 4):
        shared = records[start]
        np.testing.assert_allclose(shared["train"].mean(axis=0), 0, atol=1e-12)
        assert not np.allclose(shared["valid"].mean(axis=0), 0)
        for record in records[start : start + 4]:
            np.testing.assert_array_equal(shared["train"], record["train"])
            np.testing.assert_array_equal(shared["valid"], record["valid"])
            np.testing.assert_array_equal(shared["target"], record["target"])
    expected = frame[frame.target_time.dt.year.isin([2018, 2019])].target_next_hour.to_numpy()
    assert summary.MAE.iloc[0] == pytest.approx(abs(expected).mean())
    assert summary.RMSE.iloc[0] == pytest.approx(np.sqrt((expected**2).mean()))
    assert summary.bias.iloc[0] == pytest.approx(-expected.mean())
    future = frame.target_time.dt.year.ge(2020)
    frame.loc[future, feature_columns("panama")[0] + ["target_next_hour"]] = 1e12
    repeated, _, _ = compare_models(frame, cfg)
    pd.testing.assert_frame_equal(summary, repeated)


def test_four_fixed_families_reuse_existing_model_settings(cfg):
    models = make_models(cfg)
    assert [type(m) for m in models.values()] == [
        Ridge,
        RandomForestRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    ]
    for name, key in (("GBR", "base_model"), ("HistGBR", "residual_model")):
        assert all(models[name].get_params()[k] == v for k, v in cfg[key].items())
    assert models["HistGBR"].early_stopping is False
    assert models["RandomForest"].random_state == cfg["seed"]


def test_preliminary_protocol_rejects_test_year_and_random_early_stopping(cfg):
    cfg["datasets"]["panama"]["development_years"].append(2020)
    with pytest.raises(ValueError):
        development_years(cfg)
    cfg["datasets"]["panama"]["development_years"].pop()
    cfg["residual_model"]["early_stopping"] = True
    with pytest.raises(ValueError):
        development_years(cfg)
