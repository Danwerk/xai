"""Four fixed forecasting families, evaluated on development years in memory."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .feature_engineering import feature_columns, split_year
from .evaluation import prediction_metrics
from .study import validate_config


def make_models(cfg):
    """One fixed setting per family; existing GBR/HistGBR settings are reused."""
    options = cfg["preliminary_models"]
    return {
        "Ridge": Ridge(**options["ridge"]),
        "RandomForest": RandomForestRegressor(
            **options["random_forest"], random_state=cfg["seed"], n_jobs=cfg["workers"]
        ),
        "GBR": GradientBoostingRegressor(**cfg["base_model"], random_state=cfg["seed"]),
        "HistGBR": HistGradientBoostingRegressor(**cfg["residual_model"], random_state=cfg["seed"]),
    }


def development_years(cfg):
    validate_config(cfg)
    return cfg["datasets"]["panama"]["development_years"]


def compare_models(frame, cfg):
    years = development_years(cfg)
    # Drop final-period rows even if a caller supplied a complete feature table.
    frame = frame[frame.target_time < f"{max(years)+1}-01-01"].copy()
    if not frame.target_time.is_unique or not frame.target_time.is_monotonic_increasing:
        raise ValueError("Expected ordered unique forecast targets.")
    columns, _ = feature_columns("panama")
    losses, fold_scores, folds = {}, [], []
    for year in years:
        train, valid = split_year(frame, year)
        # The SAME train-fitted transform and numeric matrices go to all four models.
        scaler = StandardScaler().fit(train[columns])
        x_train, x_valid = scaler.transform(train[columns]), scaler.transform(valid[columns])
        folds.append(
            {
                "year": year,
                "train_n": len(train),
                "development_n": len(valid),
                "train_first_target": str(train.target_time.min()),
                "train_last_target": str(train.target_time.max()),
                "development_first_target": str(valid.target_time.min()),
                "development_last_target": str(valid.target_time.max()),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            }
        )
        with threadpool_limits(limits=cfg["workers"]):
            for name, model in make_models(cfg).items():
                print(f"Development {year}: {name}", flush=True)
                model.fit(x_train.copy(), train.target_next_hour)
                prediction = model.predict(x_valid.copy())
                error = np.asarray(prediction) - valid.target_next_hour.to_numpy()
                losses.setdefault(name, []).append(error)
                fold_scores.append(
                    {
                        "model": name,
                        "year": year,
                        **prediction_metrics(valid.target_next_hour, prediction),
                    }
                )
    rows = []
    for name, errors in losses.items():
        pooled = np.concatenate(errors)
        row = {"model": name, **prediction_metrics(np.zeros(len(pooled)), pooled)}
        row.update(
            {
                f"MAE_{score['year']}": score["MAE"]
                for score in fold_scores
                if score["model"] == name
            }
        )
        rows.append(row)
    # Pool all hourly losses, not unweighted averages of fold RMSE or bias.
    summary = (
        pd.DataFrame(rows)
        .sort_values(["MAE", "RMSE", "model"], kind="stable")
        .reset_index(drop=True)
    )
    return summary, fold_scores, folds
