"""Fixed GBR and naive references on identical forecast origins."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


def metrics(actual, predicted):
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Mõõdikud vajavad sama pikkusega lõplikke vaatlusi.")
    e = p - y
    return {
        "n": len(y),
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAPE": float(100 * np.mean(np.abs(e / y))) if (y > 0).all() else None,
        "R2": float(r2_score(y, p)) if len(y) > 1 else None,
        "bias": float(e.mean()),
    }


def fit_gbr(train, features, cfg, seed=None):
    params = dict(cfg["gbr"])
    if seed is not None:
        params["random_state"] = seed
    model = GradientBoostingRegressor(**params)
    model.fit(train[features], train.target_next_hour)
    return model


def prediction_frame(frame, predictions, model_name):
    out = frame[["origin_time", "target_time", "target_next_hour"]].copy()
    out["prediction"] = predictions
    out["error"] = out.prediction - out.target_next_hour
    out["abs_error"] = out.error.abs()
    out["model"] = model_name
    return out


def baseline_predictions(model, frame, features):
    return {
        name: prediction_frame(frame, values, name)
        for name, values in {
            "Persistence": frame.load_now,
            "SeasonalNaive": frame.load_day,
            "GBR": model.predict(frame[features]),
        }.items()
    }


def temporal_training_checks(train, cfg):
    rows = []
    features = cfg["baseline_features"]
    for year in (2016, 2017):
        fit = train[train.target_time < f"{year}-01-01"]
        hold = train[
            (train.target_time >= f"{year}-01-01") & (train.target_time < f"{year + 1}-01-01")
        ]
        if fit.empty or hold.empty:
            continue
        model = fit_gbr(fit, features, cfg)
        for name, p in baseline_predictions(model, hold, features).items():
            rows.append({"year": year, "model": name, **metrics(p.target_next_hour, p.prediction)})
    return pd.DataFrame(rows)
