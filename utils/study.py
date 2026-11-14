"""In-memory training and development selection for the comparison notebook."""

from copy import deepcopy

import numpy as np
import pandas as pd

from .correction import adaptive_correction
from .simple_correction import bias_forecast
from .evaluation import PRIMARY, METHODS, score_forecasts
from .feature_engineering import known_error_features, feature_columns, split_year
from .models import base_forecast, residual_forecasts

FAMILIES = {"scalar_adaptive": "scalar_ls", "standalone_feedback": "standalone", PRIMARY: "credit"}


def validate_config(cfg):
    if cfg["base_model_family"] not in ("gbr", "histgbr"):
        raise ValueError("base_model_family must be gbr or histgbr.")
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
    if not cfg["datasets"].get("panama", {}).get("development_years"):
        raise ValueError("Panama development years must be specified.")
    if any(options["development_years"] for name, options in cfg["datasets"].items()
           if name != "panama"):
        raise ValueError("Correction settings are selected only on Panama development years.")
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
    xai = cfg["xai_comparison"]
    if any(not isinstance(xai[name], int) or xai[name] < 1
           for name in ("background_size", "lime_samples", "batch_rows", "benchmark_rows", "timing_repeats")):
        raise ValueError("XAI sample sizes and timing repeats must be positive integers.")
    if (xai["lime_samples"] < 2 or not np.isfinite(xai["lime_kernel_width"])
            or xai["lime_kernel_width"] <= 0 or not np.isfinite(xai["lime_ridge_alpha"])
            or xai["lime_ridge_alpha"] < 0):
        raise ValueError("Invalid LIME sampling, kernel width or regularization.")
    seeds = xai["stability_seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Background sensitivity needs unique nonnegative integer seeds.")
    combined_dataset = cfg["combined_correction"]["dataset"]
    if combined_dataset not in cfg["datasets"]:
        raise ValueError("Combined correction must use an active evaluation dataset.")
    diagnostics = cfg["diagnostics"]
    if (not np.isfinite(diagnostics["change_tolerance"]) or diagnostics["change_tolerance"] <= 0
            or any(not isinstance(diagnostics[name], int) or diagnostics[name] < 0
                   for name in ("case_hours_before", "case_hours_after"))):
        raise ValueError("Diagnostics require positive tolerance and nonnegative case windows.")


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
        "hourly_bias": {
            f"hourly_bias_w{w}_s{s}": {"window_hours": w, "strength": s}
            for w in cfg["selection"]["feedback_windows_hours"]
            for s in cfg["selection"]["feedback_strengths"]
        },
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


def develop(history, columns, thresholds, cfg, *, return_folds=False):
    candidates = policy_candidates(cfg)
    rows, folds = [], []
    for year, valid, forecasts, components in residual_development_folds(history, columns, cfg):
        valid["prediction_residual"] = forecasts["residual"]
        for name, setting in candidates["hourly_bias"].items():
            forecasts[name] = bias_forecast(valid, setting, cfg["feedback"])
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
    result = (scores, choices, decay)
    return (*result, folds) if return_folds else result


def policy_forecasts(valid, forecasts, components, selection, cfg):
    """Apply frozen choices and change one mechanism at a time in ablations."""
    forecasts = dict(forecasts)
    choices = selection["choices"]
    valid = valid.assign(prediction_residual=forecasts["residual"])
    forecasts["hourly_bias"] = bias_forecast(
        valid, choices["hourly_bias"]["setting"], cfg["feedback"]
    )
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
