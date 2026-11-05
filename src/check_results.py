"""Independently reconstruct forecasts, weights, metrics and data alignment."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .common import ROOT, latest_run, read_artifact_csv, read_json, read_run_config, sha256
from .correction import adaptive_correction
from .evaluation import METHODS, PRIMARY, score_forecasts, scope_masks
from .feature_engineering import load_dataset
from .shap_explainer import repair_gain_attributions
from .study import (
    FAMILIES,
    policy_candidates,
    select_policies,
    select_decay,
    source_hashes,
    validate_frozen,
    variant_specs,
    variant_forecasts,
)


def check_dataset(run, dataset, manifest, full=False):
    record = manifest["datasets"][dataset]
    cfg = read_run_config(manifest)
    predicted = read_artifact_csv(
        run / dataset / "predictions.csv", parse_dates=["origin_time", "target_time"]
    )
    ledger = read_artifact_csv(
        run / dataset / "feedback.csv", parse_dates=["origin_time", "target_time"]
    )
    if not predicted.target_time.equals(ledger.target_time) or not predicted.origin_time.equals(
        ledger.origin_time
    ):
        raise ValueError("Feedback and forecast timestamps differ.")
    if not predicted.target_time.is_unique or not predicted.target_time.is_monotonic_increasing:
        raise ValueError("Unordered or duplicate forecast targets.")
    assert predicted.target_time.sub(predicted.origin_time).eq(pd.Timedelta(hours=1)).all()
    assert predicted.target_time.dt.year.eq(cfg["datasets"][dataset]["test_year"]).all()
    assert pd.Timestamp(record["training_target_max"]) <= predicted.origin_time.min()
    assert pd.Timestamp(record["selection_frozen_utc"]) <= pd.Timestamp(record["evaluated_utc"])
    components = ledger[[f"potential_{g}" for g in record["component_groups"]]].copy()
    components.columns = record["component_groups"]
    np.testing.assert_allclose(
        predicted.prediction_base + components.sum(axis=1), predicted.prediction_residual, atol=1e-6
    )
    frame = predicted.rename(columns={"prediction_base": "base_prediction"})
    credit = repair_gain_attributions(
        predicted.target_next_hour, frame.base_prediction, components, "squared"
    )
    np.testing.assert_allclose(
        credit, ledger[[f"potential_squared_credit_{g}" for g in components]], atol=1e-6
    )
    selection = read_json(run / "selection.json")
    choices = selection["choices"]
    fixed = (choices["fixed"]["setting"] or {"weight": 1.0})["weight"]
    np.testing.assert_allclose(
        frame.base_prediction + fixed * components.sum(axis=1),
        predicted.prediction_fixed,
        atol=1e-6,
    )
    for method in list(FAMILIES) + ["matched_scalar", "matched_standalone", "no_context"]:
        primary_setting = choices["credit_feedback"]["setting"]
        setting = (
            choices[method]["setting"]
            if method in FAMILIES
            else (dict(primary_setting) if primary_setting else None)
        )
        if method == "no_context" and setting:
            setting["by_hour"] = False
        mode = FAMILIES.get(
            method,
            {
                "matched_scalar": "scalar_credit",
                "matched_standalone": "standalone",
                "no_context": "credit",
            }.get(method),
        )
        expected, _, weights = adaptive_correction(
            frame, components, setting, mode, cfg["feedback"]
        )
        saved = ledger[[f"weight_{method}__{g}" for g in components]].to_numpy()
        assert np.isfinite(saved).all() and (saved >= 0).all() and (saved <= 1).all()
        np.testing.assert_allclose(saved, weights, atol=1e-10)
        np.testing.assert_allclose(predicted[f"prediction_{method}"], expected, atol=1e-6)
    specs = variant_specs(choices[PRIMARY]["setting"], cfg, selection["decay"]["half_life_hours"])
    expected_variants = variant_forecasts(frame, components, specs, cfg)
    for name, expected in expected_variants.items():
        np.testing.assert_allclose(predicted[f"prediction_{name}"], expected, atol=1e-8, rtol=1e-10)
    forecasts = {
        column.removeprefix("prediction_"): predicted[column].to_numpy()
        for column in predicted
        if column.startswith("prediction_")
    }
    assert set(forecasts) == set(METHODS) | set(specs)
    expected = score_forecasts(frame, forecasts, record["thresholds"], cfg, "final")
    actual = read_artifact_csv(run / dataset / "metrics.csv")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-8, rtol=1e-8)
    intervals = read_artifact_csv(run / dataset / "comparisons.csv")
    masks = scope_masks(frame, record["thresholds"])
    for row in intervals.itertuples():
        mask = masks[row.scope]
        y = predicted.loc[mask, "target_next_hour"].to_numpy()
        delta = abs(forecasts[row.A][mask] - y) - abs(forecasts[row.B][mask] - y)
        assert np.isclose(row.difference, delta.mean())
        assert row.adjusted_low <= row.ci_low <= row.ci_high <= row.adjusted_high
    if full:
        path = ROOT / cfg["datasets"][dataset]["path"]
        assert sha256(path) == record["raw_sha256"], "Raw source changed since the run."
        features, _ = load_dataset(path, cfg, dataset)
        valid = features[
            features.target_time.dt.year.eq(cfg["datasets"][dataset]["test_year"])
        ].reset_index(drop=True)
        pd.testing.assert_series_equal(valid.target_time, predicted.target_time)
        pd.testing.assert_series_equal(valid.origin_time, predicted.origin_time)
        np.testing.assert_allclose(valid.target_next_hour, predicted.target_next_hour)
        np.testing.assert_allclose(valid.actual_ramp, predicted.actual_ramp)
        np.testing.assert_array_equal(valid.hour, predicted.hour)
    print(f"{dataset}: timestamps, reconstruction, causal feedback and all metrics passed.")


def check_run(run, full=False):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    cfg = read_run_config(manifest)
    assert manifest["status"] == "complete"
    if full:
        assert (
            source_hashes() == manifest["source_sha256"]
        ), "Scientific source changed: use the recorded version or reproduce the run."
    for path, digest in manifest["artifact_sha256"].items():
        assert sha256(run / path) == digest, f"Changed artifact: {path}"
    selection = read_json(run / "selection.json")
    validate_frozen(selection, cfg, manifest["source_sha256"])
    if (run / "panama/development.csv").exists():
        scores = read_artifact_csv(run / "panama/development.csv")
        assert select_policies(scores, policy_candidates(cfg), cfg) == selection["choices"]
        assert select_decay(scores, cfg) == selection["decay"]
    for dataset, record in manifest["datasets"].items():
        if record["evaluation_status"] == "complete":
            check_dataset(run, dataset, manifest, full)


def prune_duplicate_runs():
    """Remove older, byte-identical CSV results in the single managed run folder."""
    published = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "reports").glob("*")
        if path.suffix in (".md", ".json")
    )
    pointer = read_json(ROOT / "results/latest_thesis_run.json")
    protected = {(ROOT / pointer["path"]).resolve()}
    container = (ROOT / "results/thesis_runs").resolve()
    if container.parent != ROOT.resolve() / "results":
        raise ValueError("Managed run directory escaped the workspace results folder.")
    groups = {}
    for folder in sorted(container.glob("*")):
        if not folder.is_dir() or folder.resolve() != folder.absolute():
            continue
        path = folder.resolve()
        if not (path / "manifest.json").is_file():
            continue
        manifest = read_json(path / "manifest.json")
        if manifest.get("status") != "complete":
            continue
        csvs = {p: h for p, h in manifest["artifact_sha256"].items() if p.endswith(".csv")}
        if not csvs or any(sha256(path / p) != h for p, h in csvs.items()):
            continue
        groups.setdefault(tuple(sorted(csvs.items())), []).append(path)
        if path.name in published:
            protected.add(path)
    removable = {path for folders in groups.values() for path in folders[:-1]} - protected
    for path in sorted(removable):
        if path.resolve() != path or path.resolve().parent != container:
            raise ValueError("Duplicate cleanup target escaped its managed folder.")
        shutil.rmtree(path)
        print(f"Removed duplicate run: {path.name}")
    return removable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--prune-duplicates",
        action="store_true",
        help="Remove older identical results after validation",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also rebuild input timestamps and actual values from raw data",
    )
    args = parser.parse_args()
    check_run(args.run or latest_run(), args.full)
    if args.prune_duplicates:
        prune_duplicate_runs()
