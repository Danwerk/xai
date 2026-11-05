"""One chronological experiment: development, frozen policies, then final evaluation."""

from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import (
    ROOT,
    config_reference,
    read_json,
    read_run_config,
    sha256,
    write_csv,
    write_json,
)
from .correction import adaptive_correction
from .evaluation import PRIMARY, METHODS, compare_forecasts, score_forecasts
from .feature_engineering import feature_columns, known_error_features, load_dataset, split_year
from .models import base_forecast, residual_forecasts
from .shap_explainer import repair_gain_attributions

FAMILIES = {"scalar_adaptive": "scalar_ls", "standalone_feedback": "standalone", PRIMARY: "credit"}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(cfg):
    if cfg["load_delay_hours"] != 0 or cfg["weather_delay_hours"] < 1:
        raise ValueError(
            "Only immediate measured-load availability and delayed weather are supported."
        )
    if cfg["residual_model"].get("early_stopping") is not False:
        raise ValueError("Random early stopping is forbidden in this temporal protocol.")
    for dataset, options in cfg["datasets"].items():
        feature_columns(dataset)
        years = options["development_years"]
        if years != sorted(set(years)) or any(y >= options["test_year"] for y in years):
            raise ValueError("Development must be unique ordered years strictly before test.")
        boundary = min(years) if years else options["test_year"]
        if pd.Timestamp(options["threshold_end"]) > pd.Timestamp(f"{boundary}-01-01"):
            raise ValueError("Scenario thresholds cannot use development/test targets.")
    if (
        not cfg["datasets"]["panama"]["development_years"]
        or cfg["datasets"]["estonia"]["development_years"]
    ):
        raise ValueError("Develop in Panama; transfer without Estonian parameter selection.")
    selection = cfg["selection"]
    if 1.0 not in selection["fixed_weights"] or any(
        not 0 <= v <= 1 for v in selection["fixed_weights"]
    ):
        raise ValueError("Fixed weights must be in [0,1] and include the unchanged correction.")
    if not selection["feedback_windows_hours"] or any(
        w <= 0 for w in selection["feedback_windows_hours"]
    ):
        raise ValueError("Feedback windows must be positive.")
    if not selection["feedback_strengths"] or any(
        not 0 <= s <= 1 for s in selection["feedback_strengths"]
    ):
        raise ValueError("Feedback strengths must be in [0,1].")
    if cfg["feedback"]["min_history_points"] < 1 or cfg["feedback"]["warmup_hours"] < 0:
        raise ValueError("Invalid feedback cold-start policy.")
    if cfg["evaluation"]["bootstrap_repeats"] < 2 or any(
        b < 1 for b in cfg["evaluation"]["bootstrap_blocks_hours"]
    ):
        raise ValueError("Invalid bootstrap configuration.")
    if cfg["evaluation"]["primary_block_hours"] not in cfg["evaluation"]["bootstrap_blocks_hours"]:
        raise ValueError("Primary block must belong to the declared sensitivity checks.")


def source_hashes():
    # Presentation changes must not invalidate frozen scientific choices.
    paths = [
        ROOT / "src" / name
        for name in (
            "common.py",
            "data_preprocessing.py",
            "feature_engineering.py",
            "models.py",
            "correction.py",
            "shap_explainer.py",
            "evaluation.py",
            "study.py",
        )
    ] + [
        ROOT / "run_pipeline.py",
        ROOT / "requirements.txt",
    ]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def git_state():
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def annual_forecast(frame, year, columns, cfg):
    train, hold = split_year(frame, year)
    hold["base_prediction"] = base_forecast(train, hold, columns, cfg)
    print(f"  Base OOF {year}: {len(hold)} forecasts", flush=True)
    return hold


def oof_history(frame, test_year, columns, cfg):
    first = int(frame.target_time.dt.year.min()) + 1
    history = pd.concat(
        [annual_forecast(frame, year, columns, cfg) for year in range(first, test_year)],
        ignore_index=True,
    )
    return pd.concat([history, known_error_features(history)], axis=1)


def feedback_grid(cfg):
    return {
        f"w{window}_h{int(by_hour)}_s{strength}": {
            "window_hours": window,
            "by_hour": by_hour,
            "strength": strength,
        }
        for window in cfg["selection"]["feedback_windows_hours"]
        for by_hour in (True, False)
        for strength in cfg["selection"]["feedback_strengths"]
    }


def policy_candidates(cfg):
    grid = feedback_grid(cfg)
    return {
        "fixed": {f"fixed_{w}": {"weight": w} for w in cfg["selection"]["fixed_weights"]},
        **{
            family: {f"{family}_{name}": setting for name, setting in grid.items()}
            for family in FAMILIES
        },
    }


def development_ratios(scores, methods, cfg):
    """One validation/objective for primary policy and memory-decay selection."""
    if scores.empty or not scores.stage.eq("development").all():
        raise ValueError("Selection accepts development scores only.")
    years = cfg["datasets"]["panama"]["development_years"]
    if set(scores.year) != set(years):
        raise ValueError("Selection requires all and only declared development years.")
    reference = scores[scores.method == "residual"].set_index(["year", "scope"]).MAE
    expected = pd.MultiIndex.from_product(
        [years, ["overall", "rapid_increase", "rapid_decrease"]], names=["year", "scope"]
    )
    if reference.index.has_duplicates or set(reference.index) != set(expected):
        raise ValueError("Missing or duplicate development year/scope coverage.")
    ratios = {}
    for method in methods:
        measured = scores[scores.method.eq(method)].set_index(["year", "scope"]).MAE
        if measured.index.has_duplicates or set(measured.index) != set(reference.index):
            raise ValueError("Incomplete candidate scope/year coverage.")
        ratio = measured.reindex(reference.index) / reference
        if not np.isfinite(ratio).all():
            raise ValueError("Selection needs finite development MAE ratios.")
        ratios[method] = ratio
    return ratios


def select_policies(scores, candidates, cfg):
    ratios = development_ratios(
        scores, [name for options in candidates.values() for name in options], cfg
    )
    chosen = {}
    for family, options in candidates.items():
        # The full residual correction is available to EVERY family as a safe fallback.
        ranking = [(1.0, 0, "identity", None)]
        for candidate, setting in options.items():
            ratio = ratios[candidate]
            if ratio.max() <= cfg["selection"]["max_scope_mae_ratio"]:
                ranking.append((float(ratio.mean()), 1, candidate, setting))
        score, _, candidate, setting = min(ranking, key=lambda item: item[:3])
        chosen[family] = {"candidate": candidate, "setting": setting, "development_ratio": score}
    return chosen


def residual_development_folds(history, columns, cfg):
    for year in cfg["datasets"]["panama"]["development_years"]:
        train, valid = split_year(history, year)
        forecasts, components, _ = residual_forecasts(train, valid, columns, cfg)
        yield year, valid, forecasts, components


def variant_specs(primary, cfg, half_life=None, development=False):
    """Matched credit/memory ablations and the declared parameter sensitivity."""
    options = cfg["feedback_checks"]
    specs = {
        "mae_credit": {"setting": deepcopy(primary), "credit_loss": "absolute", "half_life": None}
    }
    decays = (
        {f"decay_{h}": h for h in options["half_lives_hours"]}
        if development
        else {"recency_credit": half_life}
    )
    for name, value in decays.items():
        specs[name] = {"setting": deepcopy(primary), "credit_loss": "squared", "half_life": value}
    by_hour = primary["by_hour"] if primary else True
    for window in options["windows_hours"]:
        for strength in options["strengths"]:
            specs[f"sensitivity_w{window}_s{strength}"] = {
                "setting": {"window_hours": window, "by_hour": by_hour, "strength": strength},
                "credit_loss": "squared",
                "half_life": None,
            }
    return specs


def variant_forecasts(frame, components, specs, cfg):
    return {
        name: adaptive_correction(
            frame,
            components,
            spec["setting"],
            "credit",
            cfg["feedback"],
            credit_loss=spec["credit_loss"],
            half_life_hours=spec["half_life"],
        )[0]
        for name, spec in specs.items()
    }


def select_decay(scores, cfg):
    """Development-only MAE-ratio objective/guard; uniform is fallback/tie winner."""
    options = [
        (PRIMARY, None),
        *[(f"decay_{h}", h) for h in cfg["feedback_checks"]["half_lives_hours"]],
    ]
    ratios = development_ratios(scores, [name for name, _ in options], cfg)
    ranking = []
    for name, half_life in options:
        ratio = ratios[name]
        if half_life is None or ratio.max() <= cfg["selection"]["max_scope_mae_ratio"]:
            ranking.append((float(ratio.mean()), half_life is not None, half_life or 0, name))
    score, _, half_life, candidate = min(ranking)
    return {
        "candidate": candidate,
        "half_life_hours": half_life or None,
        "development_ratio": score,
        "memory": "exponential" if half_life else "uniform",
    }


def develop(history, columns, thresholds, cfg):
    candidates = policy_candidates(cfg)
    rows, folds = [], []
    for year, valid, forecasts, components in residual_development_folds(history, columns, cfg):
        for name, setting in candidates["fixed"].items():
            forecasts[name] = (
                valid.base_prediction.to_numpy()
                + setting["weight"] * components.sum(axis=1).to_numpy()
            )
        for family, mode in FAMILIES.items():
            for name, setting in candidates[family].items():
                forecasts[name], _, _ = adaptive_correction(
                    valid, components, setting, mode, cfg["feedback"]
                )
        rows.append(score_forecasts(valid, forecasts, thresholds, cfg, "development"))
        folds.append((valid, components, forecasts))
        print(f"  Development {year}: evaluated fixed and adaptive policies", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    choices = select_policies(scores, candidates, cfg)
    primary = choices[PRIMARY]
    selected_primary = "residual" if primary["candidate"] == "identity" else primary["candidate"]
    specs = variant_specs(primary["setting"], cfg, development=True)
    # Reuse the SAME fitted folds. No second model training or final-period rows.
    for valid, components, forecasts in folds:
        variants = {
            "base": forecasts["base"],
            "residual": forecasts["residual"],
            PRIMARY: forecasts[selected_primary],
            **variant_forecasts(valid, components, specs, cfg),
        }
        measured = score_forecasts(valid, variants, thresholds, cfg, "development")
        rows.append(measured[~measured.method.isin(["base", "residual"])])
    scores = pd.concat(rows, ignore_index=True)
    decay = select_decay(scores, cfg)
    selected = {
        "residual" if choice["candidate"] == "identity" else choice["candidate"]
        for choice in choices.values()
    }
    selected.add(decay["candidate"])
    scores["selected"] = scores.method.isin(selected)
    return scores, choices, decay


def policy_forecasts(valid, forecasts, components, selection, cfg):
    """Apply frozen choices and change one mechanism at a time in ablations."""
    forecasts = dict(forecasts)
    choices = selection["choices"]
    weight = (choices["fixed"]["setting"] or {"weight": 1.0})["weight"]
    forecasts["fixed"] = (
        valid.base_prediction.to_numpy() + weight * components.sum(axis=1).to_numpy()
    )
    traces = {}
    for name, mode in FAMILIES.items():
        forecasts[name], _, traces[name] = adaptive_correction(
            valid, components, choices[name]["setting"], mode, cfg["feedback"]
        )
    primary = choices[PRIMARY]["setting"]
    for name, mode in (
        ("matched_scalar", "scalar_credit"),
        ("matched_standalone", "standalone"),
        ("no_context", "credit"),
    ):
        setting = deepcopy(primary)
        if name == "no_context" and setting is not None:
            setting["by_hour"] = False
        forecasts[name], _, traces[name] = adaptive_correction(
            valid, components, setting, mode, cfg["feedback"]
        )
    if set(forecasts) != set(METHODS):
        raise ValueError("Experiment does not match the declared comparison ladder.")
    specs = variant_specs(primary, cfg, selection["decay"]["half_life_hours"])
    forecasts.update(variant_forecasts(valid, components, specs, cfg))
    return forecasts, traces


def save_final(output, valid, forecasts, components, traces, thresholds, cfg):
    predicted = valid[
        ["origin_time", "target_time", "hour", "target_next_hour", "actual_ramp"]
    ].copy()
    for name in forecasts:
        predicted[f"prediction_{name}"] = forecasts[name]
    write_csv(output / "predictions.csv", predicted)
    ledger = valid[["origin_time", "target_time"]].copy()
    credit = repair_gain_attributions(
        valid.target_next_hour, valid.base_prediction, components, "squared"
    )
    for i, group in enumerate(components):
        ledger[f"potential_{group}"] = components[group].to_numpy()
        ledger[f"potential_squared_credit_{group}"] = credit[:, i]
        for name, weights in traces.items():
            ledger[f"weight_{name}__{group}"] = weights[group].to_numpy()
    write_csv(output / "feedback.csv", ledger)
    scores = score_forecasts(valid, forecasts, thresholds, cfg, "final")
    comparisons = compare_forecasts(valid, forecasts, thresholds, cfg)
    write_csv(output / "metrics.csv", scores)
    write_csv(output / "comparisons.csv", comparisons)
    return scores


def validate_frozen(selection, cfg, hashes):
    if selection.get("config_ref") != config_reference(cfg) or selection["source_sha256"] != hashes:
        raise ValueError(
            "Frozen selection uses different configuration or study code. Reproduce development first."
        )
    if selection.get("status") != "frozen":
        raise ValueError("Selection is not frozen.")


def run_study(
    cfg, output, datasets=("panama", "estonia"), selection_path=None, development_only=False
):
    validate_config(cfg)
    if tuple(datasets) not in (("panama",), ("estonia",), ("panama", "estonia")):
        raise ValueError("Evaluate Panama, Estonia, or Panama then Estonia.")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Use a new or empty run directory; completed evidence is never overwritten."
        )
    if "panama" not in datasets and selection_path is None:
        raise ValueError("Transfer-only evaluation needs an explicitly frozen selection file.")
    if development_only and tuple(datasets) != ("panama",):
        raise ValueError("Development-only mode applies to Panama only.")
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    manifest = {
        "status": "running",
        "protocol": cfg["protocol"],
        "started_utc": now(),
        "study_type": "retrospective_reanalysis",
        "test_periods_previously_seen": True,
        "config_ref": config_reference(cfg),
        "source_sha256": hashes,
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            p: version(p)
            for p in (
                "numpy",
                "pandas",
                "scikit-learn",
                "shap",
                "scipy",
                "matplotlib",
                "threadpoolctl",
            )
        },
        "datasets": {},
    }
    write_json(output / "manifest.json", manifest)
    try:
        frozen = read_json(selection_path) if selection_path else None
        if frozen:
            validate_frozen(frozen, cfg, hashes)
        with threadpool_limits(limits=cfg["workers"]):
            for dataset in datasets:
                print(f"Dataset: {dataset}", flush=True)
                opts = cfg["datasets"][dataset]
                raw_path = ROOT / opts["path"]
                raw_hash = sha256(raw_path)
                frame, quality = load_dataset(raw_path, cfg, dataset)
                base_columns, columns = feature_columns(dataset)
                initial = frame[frame.target_time < opts["threshold_end"]]
                if initial.empty:
                    raise ValueError("No pre-development threshold observations.")
                thresholds = {
                    "increase": float(initial.actual_ramp.quantile(0.95)),
                    "decrease": float(initial.actual_ramp.quantile(0.05)),
                }
                folder = output / dataset
                write_csv(
                    folder / "training_summary.csv",
                    initial[columns].describe().T.reset_index(names="feature"),
                )
                history = oof_history(frame, opts["test_year"], base_columns, cfg)
                if dataset == "panama":
                    # Development receives no final-year rows, predictions or losses.
                    development, choices, decay = develop(history, columns, thresholds, cfg)
                    write_csv(folder / "development.csv", development)
                    frozen = {
                        "status": "frozen",
                        "selected_utc": now(),
                        "config_ref": config_reference(cfg),
                        "source_sha256": hashes,
                        "development_raw_sha256": raw_hash,
                        "development_years": opts["development_years"],
                        "choices": choices,
                        "decay": decay,
                        "selection_rule": "minimum mean scope/year MAE ratio; each ratio <= declared maximum; full correction fallback; identity wins exact ties",
                    }
                validate_frozen(frozen, cfg, hashes)
                if (
                    sha256(ROOT / cfg["datasets"]["panama"]["path"])
                    != frozen["development_raw_sha256"]
                ):
                    raise ValueError("Development source changed after selection.")
                write_json(output / "selection.json", frozen)
                record = {
                    "raw_sha256": raw_hash,
                    "thresholds": thresholds,
                    "quality": quality,
                    "base_features": base_columns,
                    "correction_features": columns,
                    "selection_frozen_utc": frozen["selected_utc"],
                    "evaluation_status": "not_evaluated",
                }
                manifest["datasets"][dataset] = record
                write_json(output / "manifest.json", manifest)
                if development_only:
                    continue
                # Final-year models/predictions are created only AFTER the file above exists.
                hold = annual_forecast(frame, opts["test_year"], base_columns, cfg)
                combined = pd.concat(
                    [
                        history.drop(columns=[c for c in history if c.startswith("known_error")]),
                        hold,
                    ],
                    ignore_index=True,
                )
                combined = pd.concat([combined, known_error_features(combined)], axis=1)
                train, valid = split_year(combined, opts["test_year"])
                forecasts, components, inputs = residual_forecasts(train, valid, columns, cfg)
                forecasts, traces = policy_forecasts(valid, forecasts, components, frozen, cfg)
                scores = save_final(folder, valid, forecasts, components, traces, thresholds, cfg)
                record.update(
                    evaluation_status="complete",
                    evaluated_utc=now(),
                    training_target_max=str(train.target_time.max()),
                    training_n=len(train),
                    residual_and_direct_features=inputs,
                    component_groups=list(components),
                    test_first=str(valid.target_time.min()),
                    test_last=str(valid.target_time.max()),
                )
                print(
                    scores[scores.scope.eq("overall")][
                        ["method", "MAE", "harmful_rate_vs_base"]
                    ].to_string(index=False),
                    flush=True,
                )
                if sha256(raw_path) != raw_hash:
                    raise ValueError("Source data changed during evaluation.")
        read_run_config(manifest)
        if source_hashes() != hashes:
            raise ValueError("Study code changed during the run.")
        manifest.update(
            status="development_complete" if development_only else "complete", finished_utc=now()
        )
        manifest["artifact_sha256"] = {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        if not development_only:
            write_json(
                ROOT / "results/latest_thesis_run.json",
                {
                    "path": (
                        str(output.relative_to(ROOT))
                        if output.is_relative_to(ROOT)
                        else str(output)
                    ),
                    "manifest_sha256": sha256(output / "manifest.json"),
                },
            )
        return output
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), finished_utc=now())
        write_json(output / "manifest.json", manifest)
        raise
