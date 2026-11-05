"""Small development-only comparison; never changes the correction experiment."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT, CONFIG_PATH, config_reference, read_config, sha256, write_csv, write_json
from .data_preprocessing import load_and_audit, validate_for_modeling
from .feature_engineering import build_features, feature_columns, split_year
from .evaluation import prediction_metrics
from .reporting import markdown_table
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


def run(output):
    cfg = read_config()
    years = development_years(cfg)
    raw_path = ROOT / cfg["datasets"]["panama"]["path"]
    sources = [
        Path(__file__),
        ROOT / "src/data_preprocessing.py",
        ROOT / "src/feature_engineering.py",
        ROOT / "src/common.py",
        ROOT / "src/evaluation.py",
        ROOT / "src/study.py",
        ROOT / "requirements.txt",
        CONFIG_PATH,
        raw_path,
    ]
    hashes = {
        (
            str(path.resolve().relative_to(ROOT))
            if path.resolve().is_relative_to(ROOT)
            else str(path.resolve())
        ): sha256(path)
        for path in sources
    }
    raw, quality = load_and_audit(raw_path)
    validate_for_modeling(raw, quality)
    # No final-year value enters feature generation, model fitting or scoring.
    raw = raw[raw.datetime < f"{max(years)+1}-01-01"].copy()
    frame = build_features(raw, cfg, "panama")
    summary, scores, folds = compare_models(frame, cfg)
    if any(sha256(ROOT / path) != digest for path, digest in hashes.items()):
        raise ValueError("Preliminary experiment sources changed during execution.")
    output = Path(output)
    write_csv(output / "preliminary_models.csv", summary)
    columns, _ = feature_columns("panama")
    metadata = {
        "study": "preliminary forecasting model selection",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "pooled development MAE; ties: RMSE, model name",
        "recommended_family": summary.iloc[0].model,
        "test_evaluated": False,
        "main_pipeline_modified": False,
        "development_years": years,
        "features": columns,
        "preprocessing": "shared feature builder/common mask; same StandardScaler fitted only on each training fold for every model; no target scaling",
        "config_ref": config_reference(cfg),
        "folds": folds,
        "fold_metrics": scores,
        "source_sha256": hashes,
        "python": sys.version,
        "packages": {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "threadpoolctl")
        },
    }
    write_json(output / "preliminary_models.json", metadata)
    text = [
        "# Preliminary forecasting model comparison",
        f"Development-only comparison on Panama {', '.join(map(str, years))}. Each year is predicted using training targets strictly before that year. The final test period is excluded before feature generation. Source quality auditing and hashing cover the original CSV but its test observations do not enter modelling or selection.",
        f"All four models use the same {len(columns)} base input features, feature availability rules, common observation mask and target timestamps. One StandardScaler is fitted on each training fold and its identical outputs are supplied to every model, including tree models. The shared mask follows the main feature builder, including required correction-input availability; those extra inputs are not supplied to these forecasting models.",
        "Each family has one fixed parameter setting from configs/experiment.json, the only configuration file. There is no hyperparameter search, feature search or correction/XAI experiment here.",
        markdown_table(
            summary[["model", "n", *[f"MAE_{year}" for year in years], "MAE", "RMSE", "bias"]]
        ),
        f"Development MAE favors **{summary.iloc[0].model}** among these fixed candidates. Overall metrics pool individual hourly errors across development years. Bias is prediction minus actual load; all errors use the source load units.",
        "This preliminary result describes the tested settings and is not an unbiased test estimate or a universal ranking of model families. It does not automatically replace the main GBR or change any residual-correction result. A main-model change would require a separately fixed experiment; the existing thesis comparisons retain their original backbone.",
        "The HistGBR here predicts load from base features directly. It is distinct from the main experiment's direct baseline, which also receives base predictions, extra load features and known errors.",
        "Reproduce: `.\\.venv\\Scripts\\python.exe -B -m src.compare_forecasting_models`. Parameters are in [experiment.json](../configs/experiment.json). Fold boundaries, preprocessing statistics, configuration/source hashes and package versions are in [preliminary_models.json](preliminary_models.json); [CSV](preliminary_models.csv).",
    ]
    (output / "preliminary_models.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    run(args.output_dir)
