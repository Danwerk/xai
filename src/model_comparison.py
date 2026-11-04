"""Controlled temporal model comparison and limited hyperparameter search."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .evaluation import paired_intervals


def compare_model_pairs(predictions, names, cfg):
    """Kõigi mudelipaaride uuriv võrdlus fikseeritud prognooside põhjal."""
    rows = []
    comparisons = len(names) * (len(names) - 1) // 2
    for period in ("confirmation", "retrospective"):
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pa, pb = predictions[period, a], predictions[period, b]
                pd.testing.assert_series_equal(pa.target_time, pb.target_time)
                for block in cfg["bootstrap_blocks"]:
                    estimate = paired_intervals(
                        pa.target_next_hour,
                        pa.prediction,
                        pb.prediction,
                        block,
                        cfg["bootstrap_repeats"],
                        cfg["seed"],
                        comparisons,
                    )
                    rows.append({"period": period, "A": a, "B": b, **estimate})
    return pd.DataFrame(rows)


def temporal_split(frame, year, gap_hours=24):
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    train = frame[frame.target_time < start - pd.Timedelta(hours=gap_hours)].reset_index(drop=True)
    valid = frame[frame.target_time.between(start, end, inclusive="left")].reset_index(drop=True)
    if train.empty or valid.empty:
        raise ValueError(f"Tühja perioodiga ajajaotus: {year}")
    if train.target_time.max() > valid.origin_time.min():
        raise ValueError("Treeningu siht ei olnud esimese valideerimisprognoosi ajal teada.")
    return train, valid


def make_model(name, params, cfg):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    classes = {
        "RandomForest": RandomForestRegressor,
        "ExtraTrees": ExtraTreesRegressor,
        "GBR": GradientBoostingRegressor,
        "HistGBR": HistGradientBoostingRegressor,
    }
    options = dict(params, random_state=cfg["seed"])
    if name in ("RandomForest", "ExtraTrees"):
        options["n_jobs"] = cfg["workers"]
    return classes[name](**options)


def fit_predict(model, train, valid, columns, workers=2):
    with threadpool_limits(limits=workers):
        model.fit(train[columns], train.target_next_hour)
        return model.predict(valid[columns])


def rank_candidates(folds):
    """Pool per-origin losses, not unweighted fold scores; no confirmation inputs."""
    rows = []
    for (model, view, parameter), group in folds.groupby(
        ["model", "feature_set", "parameter_id"], sort=False
    ):
        weights = group.n / group.n.sum()
        rows.append(
            {
                "model": model,
                "feature_set": view,
                "parameter_id": int(parameter),
                "MAE": float(np.sum(weights * group.MAE)),
                "RMSE": float(np.sqrt(np.sum(weights * group.RMSE**2))),
                "fold_MAE_std": float(group.MAE.std(ddof=0)),
                "fit_seconds": float(group.fit_seconds.sum()),
                "n_features": int(group.n_features.iloc[0]),
                "n": int(group.n.sum()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["MAE", "n_features", "model", "parameter_id", "feature_set"], kind="stable")
        .reset_index(drop=True)
    )
