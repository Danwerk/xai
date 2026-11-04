"""Tunnusemuudatuste, katsete arvu ja abimudeli võrdlus."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from .train_baseline import fit_gbr, metrics, prediction_frame
from .evaluation import paired_intervals, scenario_loss_interval, seasonal_error_scale
from .model_comparison import make_model
from .scenario_detector import detect_scenarios

BRIDGES = {
    "temperature_nonlinear": ["temperature"],
    "temperature_time": ["temperature", "hour_sin", "hour_cos"],
    "weather_history": ["temperature", "humidity", "wind"],
    "ramp": ["load_now"],
    "rolling": ["load_now", "load_day"],
    "calendar_history": ["load_day", "load_week", "dow_sin", "dow_cos"],
}


def safe_rho(a, b):
    if pd.Series(a).nunique() < 2 or pd.Series(b).nunique() < 2:
        return 0.0
    value = spearmanr(a, b).statistic
    return abs(float(value)) if np.isfinite(value) else 0.0


def decision_evidence(frame, predictions, shap_values, group_comparison, pfi, cards, history, cfg):
    """All evidence is from diagnosis 2018. No selection scores are accessible."""
    rows = []
    pfi_mean = pfi.groupby("group").MAE_increase.mean().clip(lower=0)
    pfi_scale = max(float(pfi_mean.max()), 1e-9)
    contrast = group_comparison.set_index("feature").difference.abs().fillna(0)
    contrast_scale = max(float(contrast.max()), 1e-9)
    for case, groups in cfg["case_studies"].items():
        for position, group in enumerate(groups):
            cols = cfg["candidate_groups"][group]
            residual = max(safe_rho(frame[c], predictions.error) for c in cols)
            bridge = BRIDGES[group]
            importance = float(pfi_mean.reindex(bridge).fillna(0).mean() / pfi_scale)
            matched_contrast = float(contrast.reindex(bridge).fillna(0).mean() / contrast_scale)
            coupling = float(
                np.mean([max(safe_rho(frame[c], shap_values[b]) for c in cols) for b in bridge])
            )
            votes = sum(card["diagnosis"]["experiment"] == group for card in cards) / max(
                len(cards), 1
            )
            historical = 0.0
            if len(history):
                query_ids = history.query_id.astype(int).unique()
                hist_error = history.groupby("query_id").error.mean().reindex(query_ids).to_numpy()
                for c in cols:
                    historical = max(historical, safe_rho(frame.loc[query_ids, c], hist_error))
            score = (
                residual * (1 + 0.25 * importance)
                + 0.25 * matched_contrast * coupling
                + 0.05 * votes
                + 0.05 * historical
            )
            rows.append(
                {
                    "case": case,
                    "group": group,
                    "manual_order": position,
                    "residual_score": residual,
                    "bridge_pfi": importance,
                    "matched_shap_contrast": matched_contrast,
                    "shap_candidate_coupling": coupling,
                    "card_vote": votes,
                    "historical_association": historical,
                    "xai_score": score,
                    "information_period": "2018; valikuaasta ja lõpphindamine suletud",
                }
            )
    return pd.DataFrame(rows)


def freeze_decisions(evidence, cfg):
    decisions = {}
    for method, score, ascending in (
        ("Manual", "manual_order", True),
        ("Residual", "residual_score", False),
        ("XAI", "xai_score", False),
    ):
        decisions[method] = {}
        for case in cfg["case_studies"]:
            ranked = evidence[evidence.case == case].sort_values(
                [score, "group"], ascending=[ascending, True], kind="stable"
            )
            decisions[method][case] = ranked.group.head(cfg["budget_per_case"]).tolist()
    return decisions


def first_choices(decisions):
    """Ühe katse eelarve kasutab algse järjestuse esimest, mitte hilisemat võitjat."""
    return pd.DataFrame(
        {"method": method, "case": case, "group": groups[0]}
        for method, cases in decisions.items()
        for case, groups in cases.items()
    )


def specialist_forecast(train, frame, baseline, cfg):
    """Vali abimudel ainult juba teada oleva koormusmuutuse järgi."""
    assert baseline.target_time.equals(frame.target_time), "Prognooside tunnid ei kattu."
    threshold = train.load_ramp_1h.quantile(cfg.get("specialist_quantile", 0.95))
    training = train[train.load_ramp_1h >= threshold]
    expert = fit_gbr(training, cfg["baseline_features"], cfg)
    active = frame.load_ramp_1h.ge(threshold).to_numpy()
    predicted = baseline.prediction.to_numpy().copy()
    if active.any():
        predicted[active] = expert.predict(frame.loc[active, cfg["baseline_features"]])
    result = prediction_frame(frame, predicted, "GBR_specialist")
    result["use_specialist"] = active
    return result, {"gate_threshold": float(threshold), "expert_train_n": len(training)}


def residual_history(frame, cfg):
    """Õpi vigadest, mis tekkisid ainult varasematel aastatel treenitud mudelitel."""
    rows = []
    for year in sorted(frame.target_time.dt.year.unique())[1:]:
        train = frame[frame.target_time < f"{year}-01-01"]
        hold = frame[frame.target_time.dt.year == year].copy()
        assert train.target_time.max() <= hold.origin_time.min()
        model = fit_gbr(train, cfg["baseline_features"], cfg)
        hold["base_prediction"] = model.predict(hold[cfg["baseline_features"]])
        rows.append(hold)
    history = pd.concat(rows, ignore_index=True)
    history["base_ramp"] = history.base_prediction - history.load_now
    history["correction"] = history.target_next_hour - history.base_prediction
    return history


def specialist_variants(train, frame, baseline, history, threshold, cfg, only=None):
    """Ühine treening kõigile osakaaludele; tuleviku tegelikke väärtusi ei kasutata."""
    assert baseline.target_time.equals(frame.target_time), "Prognooside tunnid ei kattu."
    options = cfg["specialist"]
    columns = cfg["baseline_features"] + [
        name for group in options["feature_groups"] for name in cfg["candidate_groups"][group]
    ]
    correction_columns = columns + ["base_prediction", "base_ramp"]
    base = baseline.prediction.to_numpy()
    inputs = frame.assign(base_prediction=base, base_ramp=base - frame.load_now.to_numpy())
    variants = {"GBR": base}
    original, _ = specialist_forecast(train, frame, baseline, cfg)
    for weight in options["blend_weights"]:
        name = f"hard_blend_{weight}"
        if only is None or name in only:
            variants[name] = base + weight * (original.prediction.to_numpy() - base)
    for name, family, params in (
        ("GBR_features", "GBR", cfg["gbr"]),
        ("HistGBR_features", "HistGBR", options["model"]),
    ):
        if only is None or name in only:
            model = make_model(family, params, cfg).fit(train[columns], train.target_next_hour)
            variants[name] = model.predict(inputs[columns])
    for name, family, growth_weight in (
        ("residual_ridge", "Ridge", 1),
        ("residual_hist", "HistGBR", 1),
        ("growth_weighted", "HistGBR", options["growth_weight"]),
        ("growth_only", "HistGBR", 0),
    ):
        names = {f"{name}_{weight}" for weight in options["blend_weights"]}
        if only is not None and not names.intersection(only):
            continue
        subset = history[history.actual_ramp >= threshold] if growth_weight == 0 else history
        params = {"alpha": 100.0} if family == "Ridge" else options["model"]
        model = make_model(family, params, cfg)
        fit_options = {}
        if family != "Ridge" and growth_weight > 1:
            fit_options["sample_weight"] = np.where(
                subset.actual_ramp >= threshold, growth_weight, 1.0
            )
        model.fit(subset[correction_columns], subset.correction, **fit_options)
        correction = model.predict(inputs[correction_columns])
        active = base + correction - frame.load_now.to_numpy() >= threshold
        if name != "growth_only":
            active = np.ones(len(frame), dtype=bool)
        for weight in options["blend_weights"]:
            variant = f"{name}_{weight}"
            if only is None or variant in only:
                variants[variant] = base + weight * correction * active
    return variants


def select_specialist(scores):
    """Vähenda halvimat MAE/RMSE suhet üle mõlema aasta ja hinnatava osa."""
    if not scores.stage.eq("development").all():
        raise ValueError("Mudeli valik tohib kasutada ainult arendusperioodi tulemusi.")
    base = scores[scores.model == "GBR"].set_index(["year", "scope"])
    ratios = scores.set_index(["year", "scope", "model"])[["MAE", "RMSE"]].div(
        base[["MAE", "RMSE"]]
    )
    ranking = ratios.max(axis=1).groupby("model").max().sort_values(kind="stable")
    return ranking.index[0] if ranking.iloc[0] < 1 else "GBR", ranking


def specialist_intervals(actual, a, b, mask, cfg, block):
    """MAE ja RMSE paarisvahed: samad ajaplokid ja kaheksa võrdluse parandus."""
    rng = np.random.default_rng(cfg["seed"])
    ea, eb = np.asarray(a) - actual, np.asarray(b) - actual
    count = len(actual)
    estimates = []
    for _ in range(cfg["bootstrap_repeats"]):
        starts = rng.integers(0, count, int(np.ceil(count / block)))
        indexes = ((starts[:, None] + np.arange(block)) % count).ravel()[:count]
        chosen = indexes[np.asarray(mask)[indexes]]
        if len(chosen):
            x, y = ea[chosen], eb[chosen]
            estimates.append(
                [
                    np.abs(x).mean() - np.abs(y).mean(),
                    np.sqrt((x**2).mean()) - np.sqrt((y**2).mean()),
                ]
            )
    low, high = np.quantile(estimates, [0.025 / 8, 1 - 0.025 / 8], axis=0)
    return {
        "adjusted_low": low[0],
        "adjusted_high": high[0],
        "rmse_adjusted_low": low[1],
        "rmse_adjusted_high": high[1],
        "block_hours": block,
        "comparisons": 8,
        "valid_draws": len(estimates),
    }


def specialist_metrics(frame, variants, training, threshold, stage):
    rows = []
    scale = seasonal_error_scale(training.target_next_hour)
    for year in sorted(frame.target_time.dt.year.unique()):
        for scope in ("overall", "rapid_increase"):
            mask = frame.target_time.dt.year == year
            if scope == "rapid_increase":
                mask &= frame.actual_ramp >= threshold
            base = metrics(frame.loc[mask, "target_next_hour"], variants["GBR"][mask])
            for name, predicted in variants.items():
                measured = metrics(frame.loc[mask, "target_next_hour"], predicted[mask])
                rows.append(
                    dict(
                        year=int(year),
                        model=name,
                        scope=scope,
                        stage=stage,
                        reference="GBR",
                        **measured,
                        MASE=measured["MAE"] / scale,
                        delta_MAE=measured["MAE"] - base["MAE"],
                        delta_RMSE=measured["RMSE"] - base["RMSE"],
                        growth_threshold=threshold,
                        training_n=len(training),
                    )
                )
    return pd.DataFrame(rows)


def develop_specialist(training, development, baseline, cfg):
    """Vali variant ainult varasematel andmetel; hilisemat testiosa siia ei anta."""
    if development.target_time.max() >= pd.Timestamp(cfg["selection_end"]):
        raise ValueError("Arendusandmed ulatuvad lõpphindamise perioodi.")
    assert training.target_time.max() <= development.origin_time.min()
    threshold = float(training.actual_ramp.quantile(0.95))
    with threadpool_limits(limits=2):
        variants = specialist_variants(
            training, development, baseline, residual_history(training, cfg), threshold, cfg
        )
    scores = specialist_metrics(development, variants, training, threshold, "development")
    selected, ranking = select_specialist(scores)
    scores["selected"] = scores.model.eq(selected)
    scores["selection_worst_ratio"] = scores.model.map(ranking)
    decision = {
        "selected_variant": selected,
        "selection_years": sorted(development.target_time.dt.year.unique().tolist()),
        "selection_rule": "minimax_MAE_RMSE_overall_rapid_increase",
        "growth_threshold": threshold,
        "test_start": cfg["selection_end"],
        "evaluation_kind": "retrospective",
        "test_period_previously_seen": True,
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "settings": deepcopy(
            {
                key: cfg[key]
                for key in (
                    "seed",
                    "gbr",
                    "baseline_features",
                    "candidate_groups",
                    "specialist_quantile",
                    "specialist",
                    "bootstrap_repeats",
                    "bootstrap_blocks",
                )
            }
        ),
    }
    return scores, decision


def evaluate_specialist(training, frame, baseline, decision):
    """Hinda fikseeritud valikut. Siin uusi variante ei valita."""
    if frame.target_time.min() < pd.Timestamp(decision["test_start"]):
        raise ValueError("Hindamisandmed kattuvad arendusperioodiga.")
    assert training.target_time.max() <= frame.origin_time.min()
    cfg = decision["settings"]
    selected, threshold = decision["selected_variant"], decision["growth_threshold"]
    keep = {selected, "GBR", "GBR_features", "HistGBR_features", "hard_blend_1.0"}
    with threadpool_limits(limits=2):
        variants = specialist_variants(
            training, frame, baseline, residual_history(training, cfg), threshold, cfg, only=keep
        )
    final = specialist_metrics(frame, variants, training, threshold, "final")
    final["selected"] = final.model.eq(selected)
    rows = []
    actual = frame.target_next_hour.to_numpy()
    for reference in ("GBR", "HistGBR_features"):
        for scope in ("overall", "rapid_increase"):
            mask = (
                np.ones(len(frame), dtype=bool)
                if scope == "overall"
                else frame.actual_ramp >= threshold
            )
            row = final[(final.model == selected) & (final.scope == scope)].iloc[0].to_dict()
            other = final[(final.model == reference) & (final.scope == scope)].iloc[0]
            row.update(
                reference=reference,
                stage="interval",
                selected=True,
                delta_MAE=row["MAE"] - other.MAE,
                delta_RMSE=row["RMSE"] - other.RMSE,
            )
            for block in cfg["bootstrap_blocks"]:
                rows.append(
                    {
                        **row,
                        **specialist_intervals(
                            actual, variants[selected], variants[reference], mask, cfg, block
                        ),
                    }
                )
    scores = pd.concat([final, pd.DataFrame(rows)], ignore_index=True)
    result = prediction_frame(frame, variants[selected], selected)
    for name, predicted in variants.items():
        if name not in (selected, "GBR"):
            result[f"prediction_{name}"] = predicted
    result["baseline_prediction"] = variants["GBR"]
    return scores, result


def budget_sensitivity(parts, decisions, selected, cfg, thresholds, predictions):
    """Võrdle üht ja kaht katset; lisa puuduvad prognoosid ühisesse mällu."""
    first = first_choices(decisions)
    choices = pd.concat(
        [
            first.assign(budget=1),
            selected[["method", "case", "group"]].assign(budget=2),
        ],
        ignore_index=True,
    )
    final_train = pd.concat(
        [parts[key] for key in ("train", "diagnosis", "selection")], ignore_index=True
    )
    rows = []
    for period, training in (("selection", parts["train"]), ("test", final_train)):
        frame = parts[period]
        flags = detect_scenarios(frame, thresholds)
        scale = seasonal_error_scale(training.target_next_hour)
        for group in sorted(choices.group.unique()):
            if group not in predictions[period]:
                columns = cfg["baseline_features"] + cfg["candidate_groups"][group]
                model = fit_gbr(training, columns, cfg)
                predictions[period][group] = prediction_frame(
                    frame, model.predict(frame[columns]), group
                )
        for choice in choices.itertuples(index=False):
            predicted = predictions[period][choice.group]
            mask = (
                (flags.heat | flags.cold)
                if choice.case == "temperature"
                else (flags.rapid_increase | flags.rapid_decrease)
            )
            measured = metrics(predicted.target_next_hour, predicted.prediction)
            rows.append(
                {
                    **choice._asdict(),
                    "year": int(frame.target_time.dt.year.iloc[0]),
                    **{name: measured[name] for name in ("n", "MAE", "RMSE", "bias")},
                    "MASE": measured["MAE"] / scale,
                    "MASE_scale": scale,
                    "scenario_n": int(mask.sum()),
                    "scenario_MAE": float(predicted.loc[mask, "abs_error"].mean()),
                }
            )
    intervals = []
    flags = detect_scenarios(parts["test"], thresholds)
    for case in cfg["case_studies"]:
        chosen = first[first.case == case].set_index("method")
        a = predictions["test"][chosen.loc["XAI", "group"]]
        b = predictions["test"][chosen.loc["Residual", "group"]]
        mask = (
            (flags.heat | flags.cold)
            if case == "temperature"
            else (flags.rapid_increase | flags.rapid_decrease)
        )
        for block in (168, 24):
            overall = paired_intervals(
                a.target_next_hour,
                a.prediction,
                b.prediction,
                block,
                cfg["bootstrap_repeats"],
                cfg["seed"],
                comparisons=2,
            )
            targeted = scenario_loss_interval(
                a.abs_error - b.abs_error,
                mask,
                block,
                cfg["bootstrap_repeats"],
                cfg["seed"],
                comparisons=2,
            )
            intervals.extend(
                [
                    {"case": case, "scope": "overall", **overall},
                    {"case": case, "scope": "scenario", **targeted},
                ]
            )
    return choices, pd.DataFrame(rows), pd.DataFrame(intervals)


def run_refinements(train, selection, decisions, cfg):
    rows, predictions, fitted = [], {}, {}
    base = cfg["baseline_features"]
    # Kattuv mudel treenitakse üks kord; iga meetodi katsete arv jääb samaks.
    for method, cases in decisions.items():
        for case, groups in cases.items():
            for group in groups:
                features = base + cfg["candidate_groups"][group]
                if group not in fitted:
                    fitted[group] = fit_gbr(train, features, cfg)
                    predictions[group] = prediction_frame(
                        selection, fitted[group].predict(selection[features]), group
                    )
                pred = predictions[group]
                quarter_mae = (
                    pred.assign(quarter=pred.target_time.dt.quarter)
                    .groupby("quarter")
                    .abs_error.mean()
                )
                rows.append(
                    {
                        "method": method,
                        "case": case,
                        "group": group,
                        **metrics(pred.target_next_hour, pred.prediction),
                        "quarter_MAE_std": float(quarter_mae.std(ddof=0)),
                        "n_features": len(features),
                        "tree_nodes": int(
                            sum(t[0].tree_.node_count for t in fitted[group].estimators_)
                        ),
                        "hyperparameter_trials": 1,
                    }
                )
    trials = pd.DataFrame(rows)
    selected = (
        trials.sort_values(["MAE", "group"], kind="stable")
        .groupby(["method", "case"], sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return trials, selected, predictions


# The attribution study is separate from the original RQ2 comparison. Its repair
# layer edits prediction contributions; the exact loss game audits those edits.
def attribution_group(name):
    if name.startswith("known_error"):
        return "error_memory"
    if name in ("load_day", "load_week"):
        return "seasonal_load"
    if name in ("load_now", "load_lag_2", "load_lag_3"):
        return "recent_load"
    if name.startswith("hour_") or name == "clock_offset":
        return "clock"
    if name.startswith(("temperature", "humidity", "wind")):
        return "weather"
    if name.startswith(("load_ramp", "load_mean", "load_std")):
        return "dynamics"
    return "calendar"


def known_error_features(history):
    """At origin t only errors whose targets are <=t may enter the layer."""
    ordered = history.sort_values("target_time")
    times = pd.DatetimeIndex(ordered.target_time)
    if times.has_duplicates:
        raise ValueError("Historical forecast targets must be unique.")
    errors = pd.Series((ordered.target_next_hour - ordered.base_prediction).to_numpy(), index=times)
    result = pd.DataFrame(index=ordered.index)
    for lag in (1, 24, 168):
        result[f"known_error_{lag}"] = errors.reindex(
            pd.DatetimeIndex(ordered.origin_time) - pd.Timedelta(hours=lag - 1)
        ).to_numpy()
    # Timestamp rolling excludes the current target even across data gaps.
    past = errors.rolling("24h", closed="left").mean()
    result["known_error_mean24"] = past.to_numpy()
    past = errors.rolling("168h", closed="left").mean()
    result["known_error_mean168"] = past.to_numpy()
    return result.fillna(0).reindex(history.index)


def attribution_basis(frame, columns, space, memory, harmonics, cfg):
    """Same periodic linear architecture for raw values and SHAP coordinates."""
    base = cfg["baseline_features"]
    inputs = [f"phi_{c}" for c in base] if space == "shap" else list(base)
    names = list(base)
    if memory:
        extra = [c for c in columns if c not in base]
        extra += [c for c in frame if c.startswith("known_error")]
        inputs += extra
        names += extra
    hour = frame["hour"].to_numpy() if "hour" in frame else frame.target_time.dt.hour.to_numpy()
    clock = np.column_stack(
        [np.ones(len(frame))]
        + [
            fn(2 * np.pi * k * hour / 24)
            for k in range(1, harmonics + 1)
            for fn in (np.sin, np.cos)
        ]
    )
    values = frame[inputs].to_numpy()
    design = np.column_stack(
        [(values[:, :, None] * clock[:, None, :]).reshape(len(frame), -1), clock[:, 1:]]
    )
    groups = [attribution_group(c) for c in names for _ in range(clock.shape[1])]
    groups += ["clock"] * (clock.shape[1] - 1)
    return design, groups


def fit_attribution_layer(training, valid, setting, columns, cfg):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if training.target_time.max() > valid.origin_time.min():
        raise ValueError("Repair training targets were unavailable at the first forecast origin.")
    args = (setting["space"], setting["memory"], setting["harmonics"], cfg)
    x, groups = attribution_basis(training, columns, *args)
    z, _ = attribution_basis(valid, columns, *args)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=setting["alpha"]).fit(
        scaler.transform(x), training.target_next_hour - training.base_prediction
    )
    terms = scaler.transform(z) * model.coef_
    names = list(dict.fromkeys(groups)) + ["offset"]
    components = pd.DataFrame(
        {g: terms[:, np.array(groups) == g].sum(axis=1) for g in names[:-1]},
        index=valid.index,
    )
    components["offset"] = float(model.intercept_)
    predicted = valid.base_prediction.to_numpy() + components.sum(axis=1).to_numpy()
    if not np.allclose(predicted, valid.base_prediction + model.predict(scaler.transform(z))):
        raise ValueError("Repair components do not reconstruct the edited forecast.")
    parameters = {
        "setting": setting,
        "groups": groups,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "training_n": len(training),
        "training_target_max": str(training.target_time.max()),
    }
    return predicted, components, parameters


def attribution_scores(frame, forecasts, threshold, stage, settings=None):
    rows = []
    for name, predicted in forecasts.items():
        for scope, mask in (
            ("overall", np.ones(len(frame), dtype=bool)),
            ("rapid_increase", frame.actual_ramp.to_numpy() >= threshold["increase"]),
            ("rapid_decrease", frame.actual_ramp.to_numpy() <= threshold["decrease"]),
        ):
            if not mask.any():
                continue
            measured = metrics(frame.target_next_hour.to_numpy()[mask], np.asarray(predicted)[mask])
            error = np.abs(np.asarray(predicted) - frame.target_next_hour.to_numpy())
            original = np.abs(frame.base_prediction - frame.target_next_hour).to_numpy()
            rows.append(
                {
                    "model": name,
                    "year": int(frame.target_time.dt.year.iloc[0]),
                    "scope": scope,
                    "stage": stage,
                    **measured,
                    "harm_share": float((error[mask] > original[mask] + 1e-9).mean()),
                    "delta_MAE": float((error[mask] - original[mask]).mean()),
                    **((settings or {}).get(name, {})),
                }
            )
    return pd.DataFrame(rows)


def select_attribution_layers(scores, settings):
    """The same minimax criterion and candidate count for each representation."""
    if not scores.stage.eq("development").all():
        raise ValueError("Attribution selection cannot access final-period scores.")
    base = scores[scores.model == "GBR"].set_index(["year", "scope"]).MAE
    ranking = []
    for name, setting in settings.items():
        measured = scores[scores.model == name].set_index(["year", "scope"]).MAE
        ratio = measured / base
        if ratio.isna().any() or len(ratio) != len(base):
            raise ValueError("Incomplete development scope/year coverage.")
        ranking.append((float(ratio.max()), float(ratio.mean()), name, setting))
    chosen = {}
    for _, _, name, setting in sorted(ranking, key=lambda row: row[:3]):
        family = setting["space"] + ("_memory" if setting["memory"] else "")
        chosen.setdefault(family, {**setting, "candidate": name})
    return chosen


def repair_feedback(
    frame, components, window, by_hour=True, strength=1.0, interactions=True, scalar=False
):
    """Suppress edits using ONLY earlier realised repair credits.

    The full potential repair is audited even if an edit was not deployed.
    This makes feedback observable without using the next target. The matched
    control omits cross-module interactions, using stand-alone gain instead.
    """
    from .shap_explainer import repair_gain_attributions

    if scalar:
        components = components.sum(axis=1).to_frame("total")
    c = components.to_numpy()
    residual = (frame.target_next_hour - frame.base_prediction).to_numpy()
    credit = (
        repair_gain_attributions(frame.target_next_hour, frame.base_prediction, c, "squared")
        if interactions
        else c * (2 * residual[:, None] - c)
    )
    past = pd.DataFrame(credit, index=frame.index, columns=components.columns)
    magnitude = past.abs()
    if by_hour:
        hour = frame["hour"] if "hour" in frame else frame.target_time.dt.hour
        average = past.groupby(hour).transform(
            lambda x: x.shift(1).rolling(window // 24, min_periods=3).mean()
        )
        scale = magnitude.groupby(hour).transform(
            lambda x: x.shift(1).rolling(window // 24, min_periods=3).mean()
        )
    else:
        average = past.shift(1).rolling(window, min_periods=24).mean()
        scale = magnitude.shift(1).rolling(window, min_periods=24).mean()
    weights = (1 + strength * average / scale.clip(lower=1e-12)).clip(0, 1).fillna(1)
    effective = components * weights
    return frame.base_prediction.to_numpy() + effective.sum(axis=1).to_numpy(), effective, weights


def transfer_load_frame(path, end_year=None):
    """Elering measured load only: UTC intervals, local calendar, no imputation.

    Resampling labels each completed hourly interval at its right endpoint.
    Missing targets/history remain missing and are excluded, never fabricated.
    Weather and holiday/school flags are unavailable and are not substituted.
    """
    raw = pd.read_csv(path, parse_dates=["datetime"])
    raw["datetime"] = pd.to_datetime(raw.datetime, utc=True)
    if end_year is not None:
        raw = raw[raw.datetime < f"{end_year + 1}-01-01"].copy()
    if raw.datetime.duplicated().any():
        raise ValueError("External consumption contains duplicate timestamps.")
    nonpositive = int(raw.consumption.le(0).sum())
    raw.loc[raw.consumption.le(0), "consumption"] = np.nan
    grouped = (
        raw.set_index("datetime")
        .consumption.sort_index()
        .resample("h", label="right", closed="left")
    )
    load = grouped.mean()
    # A single hourly value or all four quarter-hour values are required.
    counts = grouped.count()
    expected = counts.groupby(counts.index.normalize()).transform("max")
    load = load.where(counts.eq(expected) & expected.isin([1, 4]))
    origin = load.index.tz_localize(None)
    target = origin + pd.Timedelta(hours=1)
    local = pd.DatetimeIndex(target).tz_localize("UTC").tz_convert("Europe/Tallinn")
    f = pd.DataFrame({"origin_time": origin, "target_time": target}, index=origin)
    x = pd.Series(load.to_numpy(), index=origin)
    f["target_next_hour"], f["actual_ramp"] = x.shift(-1), x.shift(-1) - x
    f["load_now"], f["load_day"], f["load_week"] = x, x.shift(23), x.shift(167)
    f["hour"], f["dayofweek"] = local.hour, local.dayofweek
    for name, value, period in (
        ("hour", local.hour, 24),
        ("dow", local.dayofweek, 7),
        ("month", local.month - 1, 12),
    ):
        f[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    f["weekend"] = (local.dayofweek >= 5).astype(int)
    for lag in (1, 3):
        f[f"load_ramp_{lag}h"] = x.diff(lag)
    for lag in (2, 3):
        f[f"load_lag_{lag}"] = x.shift(lag)
    for window in (6, 24):
        f[f"load_mean_{window}"] = x.rolling(window).mean()
        f[f"load_std_{window}"] = x.rolling(window).std()
    for lag in (1, 2, 3, 6, 12, 47, 71, 335, 671):
        f[f"recent_load_lag_{lag}"] = x.shift(lag)
    for window in (6, 24, 168):
        for stat in ("mean", "std", "min", "max"):
            f[f"history_{stat}_{window}"] = getattr(x.rolling(window), stat)()
    for day in range(7):
        f[f"weekday_{day}"] = (local.dayofweek == day).astype(int)
    f["working_hour"] = ((local.hour >= 8) & (local.hour <= 18) & (local.dayofweek < 5)).astype(int)
    excluded = {
        "origin_time",
        "target_time",
        "target_next_hour",
        "actual_ramp",
        "hour",
        "dayofweek",
    }
    columns = [c for c in f if c not in excluded]
    frame = f.dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete external forecast windows.")
    quality = {
        "source_rows": len(raw),
        "hourly_intervals": len(load),
        "missing_hourly_load": int(load.isna().sum()),
        "valid_forecasts": len(frame),
        "timezone_calendar": "Europe/Tallinn",
        "split_timezone": "UTC",
        "hourly_label": "right endpoint of completed interval",
        "imputed_values": 0,
        "nonpositive_load_unavailable": nonpositive,
    }
    return frame, {"history_calendar": columns}, quality


def attribution_intervals(frame, forecasts, threshold, cfg, year):
    from .evaluation import clock_loss_interval

    references = [
        "GBR",
        "raw_memory",
        "HistResidual",
        "HistGBR",
        "Ridge",
        "standalone_feedback",
        "scalar_feedback",
        "shap_memory",
    ]
    actual = frame.target_next_hour.to_numpy()
    primary = np.abs(forecasts["credit_feedback"] - actual)
    rows = []
    for reference in references:
        difference = primary - np.abs(forecasts[reference] - actual)
        for scope, mask in (
            ("overall", np.ones(len(frame), dtype=bool)),
            ("rapid_increase", frame.actual_ramp.to_numpy() >= threshold["increase"]),
            ("rapid_decrease", frame.actual_ramp.to_numpy() <= threshold["decrease"]),
        ):
            for block in cfg["bootstrap_blocks"]:
                rows.append(
                    {
                        "A": "credit_feedback",
                        "B": reference,
                        "scope": scope,
                        "year": year,
                        **clock_loss_interval(
                            frame.target_time,
                            difference,
                            mask,
                            block=block,
                            repeats=cfg["bootstrap_repeats"],
                            seed=cfg["seed"],
                            comparisons=len(references) * 3,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_attribution_study(cfg, output, development_only=False, transfer=False):
    """Annual OOF contribution edits; no model or intermediate table is saved.

    Public trial/comparison tables are reused. Only one local final forecast
    table and one figure are added. Choices are saved before final prediction.
    """
    from time import perf_counter
    from .common import ROOT, read_json, read_artifact_csv, sha256, write_csv, write_json
    from .data_preprocessing import load_and_audit, validate_for_modeling
    from .expanded_features import build_expanded_features
    from .shap_explainer import explain_prediction, repair_gain_attributions

    output = Path(output)
    cfg = deepcopy(cfg)
    options = deepcopy(cfg["attribution_refinement"])
    cfg["bootstrap_repeats"] = options["bootstrap_repeats"]
    if cfg["load_delay_hours"] != 0:
        raise ValueError(
            "This study assumes load and the previous forecast error are available at origin t."
        )
    manifest = read_json(output / "manifest.json")
    experiment = "attribution_transfer" if transfer else "attribution_refinement"
    forecast_path = output / (
        "refinements/test_attribution_estonia.csv"
        if transfer
        else "refinements/test_attribution.csv"
    )
    previously_evaluated = manifest.get(experiment, {}).get("status") == "complete"
    source_hashes = {
        p: sha256(ROOT / p)
        for p in (
            "src/refinement_experiments.py",
            "src/shap_explainer.py",
            "src/expanded_features.py",
            "src/feature_engineering.py",
            "configs/experiment.json",
            "run_pipeline.py",
        )
    }
    if transfer:
        frozen = deepcopy(manifest["attribution_refinement"])
        if frozen["status"] != "complete":
            raise ValueError("Complete and freeze the Panama study before transfer evaluation.")
        cfg["raw_path"] = options["transfer_path"]
        cfg["train_end"] = "2021-01-01"
        cfg["baseline_features"] = [
            c for c in cfg["baseline_features"] if c not in ("temperature", "humidity", "wind")
        ]
        cfg["candidate_groups"]["calendar_history"] = ["load_lag_2", "load_lag_3"]
        options.update(test_year=options["transfer_test_year"], selection_years=[])
        frame, bundles, quality = transfer_load_frame(ROOT / cfg["raw_path"], options["test_year"])
    else:
        if sha256(ROOT / cfg["raw_path"]) != manifest["raw_sha256"]:
            raise ValueError("Raw data differ from the parent experiment.")
        raw, quality = load_and_audit(ROOT / cfg["raw_path"])
        validate_for_modeling(raw, quality)
        frame, _, bundles = build_expanded_features(raw, cfg)
    columns = list(
        dict.fromkeys(
            cfg["baseline_features"]
            + [
                c
                for group in cfg["specialist"]["feature_groups"]
                for c in cfg["candidate_groups"][group]
            ]
        )
    )
    initial = frame[frame.target_time < cfg["train_end"]]
    threshold = {
        "increase": float(initial.actual_ramp.quantile(0.95)),
        "decrease": float(initial.actual_ramp.quantile(0.05)),
    }
    settings = {
        f"{space}{'_memory' if memory else ''}_h{h}_a{a}": {
            "space": space,
            "memory": memory,
            "harmonics": h,
            "alpha": a,
        }
        for space in ("raw", "shap")
        for memory in (False, True)
        for h in options["harmonics"]
        for a in options["alphas"]
    }
    clock_start = perf_counter()

    def explained_year(year):
        start = pd.Timestamp(year=year, month=1, day=1)
        train = frame[frame.target_time < start]
        hold = frame[frame.target_time.dt.year == year].copy().reset_index(drop=True)
        if train.empty or hold.empty or train.target_time.max() > hold.origin_time.min():
            raise ValueError("Invalid chronological attribution fold.")
        model = fit_gbr(train, cfg["baseline_features"], cfg)
        values, base = explain_prediction(model, hold[cfg["baseline_features"]])
        hold["base_prediction"] = model.predict(hold[cfg["baseline_features"]])
        hold["base_value"] = base
        hold = pd.concat([hold, values.add_prefix("phi_")], axis=1)
        print(f"Attribution OOF {year}: {len(hold)} rows", flush=True)
        return hold

    def controls(training, valid):
        start = valid.target_time.min()
        direct = frame[frame.target_time < start]
        forecasts = {"GBR": valid.base_prediction.to_numpy()}
        for name, family, params, cols in (
            ("HistGBR", "HistGBR", cfg["specialist"]["model"], columns),
            ("Ridge", "Ridge", {"alpha": 1.0}, bundles["history_calendar"]),
        ):
            model = make_model(family, params, cfg).fit(direct[cols], direct.target_next_hour)
            forecasts[name] = model.predict(valid[cols])
        cols = columns + ["base_prediction"] + [c for c in training if c.startswith("known_error")]
        model = make_model("HistGBR", cfg["specialist"]["model"], cfg).fit(
            training[cols], training.target_next_hour - training.base_prediction
        )
        forecasts["HistResidual"] = valid.base_prediction.to_numpy() + model.predict(valid[cols])
        values, expected = explain_prediction(model, valid[cols])
        groups = [
            attribution_group(c) if c != "base_prediction" else "forecast_anchor" for c in cols
        ]
        components = pd.DataFrame(
            {
                group: values.loc[:, np.array(groups) == group].sum(axis=1).to_numpy()
                for group in dict.fromkeys(groups)
            },
            index=valid.index,
        )
        components["offset"] = expected
        np.testing.assert_allclose(components.sum(axis=1), model.predict(valid[cols]), atol=1e-6)
        return forecasts, components

    def append_summary(name, rows):
        path = ROOT / f"reports/{name}.csv"
        old = read_artifact_csv(path) if path.exists() else pd.DataFrame()
        if "experiment" in old:
            old = old[old.experiment != experiment]
        write_csv(
            path,
            pd.concat(
                [old, rows.assign(experiment=experiment, source_run=output.name)], ignore_index=True
            ),
        )

    with threadpool_limits(limits=2):
        first_year = int(frame.target_time.dt.year.min()) + 1
        history = pd.concat(
            [explained_year(y) for y in range(first_year, options["test_year"])], ignore_index=True
        )
        history = pd.concat([history, known_error_features(history)], axis=1)
        development, fold_components = [], {}
        for year in options["selection_years"]:
            training = history[history.target_time.dt.year < year]
            valid = history[history.target_time.dt.year == year]
            forecasts, residual_components = controls(training, valid)
            fold_components[year, "HistResidual"] = residual_components
            for name, setting in settings.items():
                predicted, components, _ = fit_attribution_layer(
                    training, valid, setting, columns, cfg
                )
                forecasts[name] = predicted
                fold_components[year, name] = components
            development.append(
                attribution_scores(valid, forecasts, threshold, "development", settings)
            )
            print(f"Attribution development {year}: {len(settings)} layer settings", flush=True)
        scores = pd.concat(development, ignore_index=True) if development else pd.DataFrame()
        choices = (
            deepcopy(frozen["choices"]) if transfer else select_attribution_layers(scores, settings)
        )
        feedback_settings, feedback_rows = {}, []
        for year in options["selection_years"]:
            valid = history[history.target_time.dt.year == year]
            forecasts = {}
            for basis in ("credit", "standalone", "scalar"):
                for window in options["feedback_windows"]:
                    for by_hour in (True, False):
                        for strength in options["feedback_strengths"]:
                            name = f"{basis}_w{window}_h{int(by_hour)}_s{strength}"
                            setting = {
                                "window": window,
                                "by_hour": by_hour,
                                "strength": strength,
                                "interactions": basis != "standalone",
                                "scalar": basis == "scalar",
                            }
                            feedback_settings[name] = setting
                            forecasts[name], _, _ = repair_feedback(
                                valid, fold_components[year, "HistResidual"], **setting
                            )
            feedback_rows.append(attribution_scores(valid, forecasts, threshold, "development"))
        feedback_scores = (
            pd.concat(feedback_rows, ignore_index=True) if feedback_rows else pd.DataFrame()
        )
        reference = (
            None
            if transfer
            else scores[scores.model == "HistResidual"].set_index(["year", "scope"]).MAE
        )
        feedback_choices = deepcopy(frozen["feedback_choices"]) if transfer else {}
        for basis in ([] if transfer else ["credit", "standalone", "scalar"]):
            ranked = [(1.0, "HistResidual")]
            for name, setting in feedback_settings.items():
                if not name.startswith(basis + "_"):
                    continue
                measured = (
                    feedback_scores[feedback_scores.model == name].set_index(["year", "scope"]).MAE
                )
                ratio = measured / reference
                # Predeclared harm guard: at most 5% deterioration in any scope/year.
                if ratio.max() <= 1.05:
                    ranked.append((float(ratio.mean()), name))
            winner = min(ranked)[1]
            feedback_choices[basis + "_feedback"] = {
                "candidate": winner,
                "setting": feedback_settings.get(winner),
                "development_ratio": min(ranked)[0],
            }
        scores = pd.concat([scores, feedback_scores], ignore_index=True)
        decision = {
            "status": "development_complete",
            "study_type": "frozen_transfer" if transfer else "exploratory_retrospective",
            "test_period_previously_seen": not transfer or previously_evaluated,
            "settings": options,
            "choices": choices,
            "data_quality": quality,
            "feedback_choices": feedback_choices,
            "feedback_selection_rule": "minimum mean scope/year MAE ratio; maximum ratio <=1.05; unchanged layer is a candidate",
            "thresholds": threshold,
            "selection_frozen_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "minimum worst overall/increase/decrease MAE ratio across development years",
            "development_seconds": perf_counter() - clock_start,
            "raw_sha256": sha256(ROOT / cfg["raw_path"]),
            "source_sha256": source_hashes,
        }
        chosen_ids = {v["candidate"] for v in choices.values()} | {
            v["candidate"] for v in feedback_choices.values()
        }
        if len(scores):
            scores["selected"] = scores.model.isin(chosen_ids)
        if transfer:
            decision["parent_selection_frozen_utc"] = frozen["selection_frozen_utc"]
            decision["feature_mapping"] = {
                "baseline": cfg["baseline_features"],
                "repair": columns,
                "ridge": bundles["history_calendar"],
            }
        manifest[experiment] = decision
        if any(sha256(ROOT / p) != digest for p, digest in source_hashes.items()):
            raise ValueError("Study code/configuration changed while the experiment was running.")
        write_json(output / "manifest.json", manifest)
        write_json(
            ROOT / "results/latest_run.json",
            {"path": str(output), "manifest_sha256": sha256(output / "manifest.json")},
        )
        if len(scores):
            append_summary("trials", scores)
        print("Frozen attribution settings: " + str(choices), flush=True)
        print("Frozen feedback settings: " + str(feedback_choices), flush=True)
        if development_only:
            return scores, decision

        # No final-period forecast/explanation existed in this function before freezing.
        valid = explained_year(options["test_year"])
        combined = pd.concat(
            [history.drop(columns=[c for c in history if c.startswith("known_error")]), valid],
            ignore_index=True,
        )
        combined = pd.concat([combined, known_error_features(combined)], axis=1)
        valid = combined[combined.target_time.dt.year == options["test_year"]]
        training = combined[combined.target_time.dt.year < options["test_year"]]
        forecasts, residual_components = controls(training, valid)
        components_by_family, parameters = {}, {}
        for family, setting in choices.items():
            forecast, components, parameter = fit_attribution_layer(
                training, valid, setting, columns, cfg
            )
            forecasts[family] = forecast
            components_by_family[family] = components
            parameters[family] = parameter
        for family, choice in feedback_choices.items():
            if choice["setting"] is None:
                forecasts[family] = forecasts["HistResidual"]
                components_by_family[family] = residual_components
            else:
                forecasts[family], components_by_family[family], _ = repair_feedback(
                    valid, residual_components, **choice["setting"]
                )
        final = attribution_scores(valid, forecasts, threshold, "final")
        scores = pd.concat([scores, final], ignore_index=True)
        primary = "credit_feedback"
        comparisons = attribution_intervals(valid, forecasts, threshold, cfg, options["test_year"])
        predicted = valid[["origin_time", "target_time", "target_next_hour", "actual_ramp"]].copy()
        for name, values in forecasts.items():
            predicted[f"prediction_{name}"] = values
        components = components_by_family[primary]
        credits = repair_gain_attributions(
            valid.target_next_hour, valid.base_prediction, components
        )
        squared = repair_gain_attributions(
            valid.target_next_hour, valid.base_prediction, components, "squared"
        )
        for i, group in enumerate(components):
            predicted[f"correction_{group}"] = components[group].to_numpy()
            predicted[f"gain_{group}"] = credits[:, i]
            predicted[f"squared_gain_{group}"] = squared[:, i]
        write_csv(forecast_path, predicted)
        decision.update(
            status="complete",
            parameters=parameters,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=perf_counter() - clock_start,
            primary_method=primary,
            comparison_family_size=int(comparisons.comparisons.iloc[0]),
            interval_alignment="clock_hours",
            forecast_sha256=sha256(forecast_path),
            maximum_credit_reconstruction_error=float(
                np.max(
                    np.abs(
                        credits.sum(axis=1)
                        - (
                            np.abs(valid.base_prediction - valid.target_next_hour).to_numpy()
                            - np.abs(forecasts[primary] - valid.target_next_hour.to_numpy())
                        )
                    )
                )
            ),
        )
        manifest[experiment] = decision
        if any(sha256(ROOT / p) != digest for p, digest in source_hashes.items()):
            raise ValueError("Study code/configuration changed during final evaluation.")
        write_json(output / "manifest.json", manifest)
        write_json(
            ROOT / "results/latest_run.json",
            {"path": str(output), "manifest_sha256": sha256(output / "manifest.json")},
        )
        append_summary("trials", scores)
        append_summary("comparisons", pd.DataFrame(comparisons))
        print(
            final[["model", "scope", "MAE", "RMSE", "harm_share"]].to_string(index=False),
            flush=True,
        )
        return scores, decision
