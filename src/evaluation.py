"""Time-dependent paired uncertainty and diagnostic artifact audits."""

import numpy as np
import pandas as pd


def seasonal_error_scale(training_values, season=24):
    """MASE nimetaja: sama tunni päevavahed ainult treeningperioodil."""
    values = np.asarray(training_values, dtype=float)
    if season < 1 or len(values) <= season:
        raise ValueError("MASE jaoks on vaja hooajast pikemat treeningrida.")

    scale = float(np.mean(np.abs(values[season:] - values[:-season])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MASE võrdlusskaala peab olema lõplik ja positiivne.")

    return scale


def accept_refinement(overall_change, scenario_change, scenario_count, min_count=40):
    """Võta muudatus vastu, kui valikuperioodil paranevad mõlemad vead."""
    return bool(scenario_count >= min_count and overall_change < 0 and scenario_change < 0)


def scenario_loss_interval(
    loss_difference, scenario_mask, block=168, repeats=2000, seed=42, comparisons=2
):
    """Vali terveid ajaplokke; arvuta vahe ainult neis olevatel sihttundidel.

    Vahe on uue reegli absoluutviga miinus senise reegli absoluutviga.
    Olukorrasilt liigub koos tunniga, et sündmuste koondumine ajas säiliks.
    """
    difference = np.asarray(loss_difference, dtype=float)
    mask = np.asarray(scenario_mask, dtype=bool)
    if len(difference) != len(mask) or not mask.any() or not 1 <= block <= len(mask):
        raise ValueError("Paarisvõrdlus vajab sama tunni vigu ja olukorrasilte.")

    rng = np.random.default_rng(seed)
    count = len(difference)
    block_count = int(np.ceil(count / block))
    estimates = []

    for _ in range(repeats):
        starts = rng.integers(0, count, block_count)
        indexes = ((starts[:, None] + np.arange(block)) % count).ravel()[:count]
        selected = indexes[mask[indexes]]
        if len(selected):
            estimates.append(difference[selected].mean())

    low, high = np.quantile(estimates, [0.025, 0.975])
    adjusted_low, adjusted_high = np.quantile(
        estimates, [0.025 / comparisons, 1 - 0.025 / comparisons]
    )
    return {
        "difference": float(difference[mask].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "adjusted_low": float(adjusted_low),
        "adjusted_high": float(adjusted_high),
        "valid_draws": len(estimates),
        "block_hours": block,
        "comparisons": comparisons,
    }


def clock_loss_interval(times, difference, mask, **kwargs):
    """Bootstrap actual clock hours; absent/unscored forecast windows stay masked."""
    index = pd.DatetimeIndex(times)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Clock bootstrap requires ordered unique target times.")
    full = pd.date_range(index.min(), index.max(), freq="h")
    values = pd.Series(np.asarray(difference, float), index=index).reindex(full, fill_value=0)
    eligible = pd.Series(np.asarray(mask, bool), index=index).reindex(full, fill_value=False)
    return scenario_loss_interval(values.to_numpy(), eligible.to_numpy(), **kwargs)


def paired_block_bootstrap(actual, a, b, block=168, repeats=2000, seed=42):
    """Circular block bootstrap of paired absolute-loss difference A minus B."""
    y, a, b = [np.asarray(x, float) for x in (actual, a, b)]
    if (
        len(y) != len(a)
        or len(y) != len(b)
        or len(y) < 2
        or block < 1
        or block > len(y)
        or repeats < 2
    ):
        raise ValueError("Sobimatud bootstrap-parameetrid või eri pikkusega paarid.")
    interval = paired_intervals(y, a, b, block, repeats, seed)
    return {
        "MAE_difference_A_minus_B": interval["difference"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "block_hours": block,
        "repeats": repeats,
        "interpretation": "Negatiivne erinevus soosib A-d; nulli hõlmav intervall ei toeta kindlat erinevust. Intervall on nominaalne ja eksploratiivne.",
    }


def audit_card(card):
    reconstructed = card["base_value"] + sum(c["shap"] for c in card["shap_contributions"])
    checks = {
        "numeric_error_correct": bool(
            np.isclose(card["prediction"] - card["actual"], card["error"])
        ),
        "absolute_error_correct": bool(np.isclose(abs(card["error"]), card["absolute_error"])),
        "shap_additive": bool(np.isclose(reconstructed, card["prediction"], atol=1e-5)),
        "shap_wording_correct": all(
            c["interpretation"].startswith(
                "Suurendab" if c["shap"] > 0 else ("Vähendab" if c["shap"] < 0 else "Panus")
            )
            for c in card["shap_contributions"]
        ),
        "history_available": all(
            pd.Timestamp(h["target_time"]) <= pd.Timestamp(card["origin_time"])
            for h in card["historical_cases"]
        ),
        "hypothesis_testable_or_abstained": (
            card["diagnosis"]["hypothesis"] is None and card["diagnosis"]["experiment"] is None
        )
        or (
            card["diagnosis"]["hypothesis"] is not None
            and card["diagnosis"]["experiment"] is not None
        ),
        "traceable": len(card["evidence_sources"]) >= 3,
        "prediction_not_error_caveat": "veapõhjust" in card["limitations"][0],
    }
    return {
        "target_time": card["target_time"],
        **checks,
        "all_pass": all(checks.values()),
        "scope": "Automaatne struktuuri- ja arvukontroll; inimese arusaadavust ega põhjuslikku korrektsust ei tõenda.",
    }


def mean_bias_interval(predictions, block=168, repeats=2000, seed=42):
    """Bootstrap signed residual mean for scenario-conditional bias.

    Re-sample the whole hourly mask with errors to preserve event clustering.
    """
    e = np.asarray(predictions, float)
    rng = np.random.default_rng(seed)
    n = len(e)
    estimates = []
    for _ in range(repeats):
        starts = rng.integers(0, n, int(np.ceil(n / block)))
        sample = e[((starts[:, None] + np.arange(block)) % n).ravel()[:n]]
        if np.isfinite(sample).any():
            estimates.append(float(np.nanmean(sample)))
    if not estimates:
        return {"bias_ci_low": None, "bias_ci_high": None}
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"bias_ci_low": float(low), "bias_ci_high": float(high)}


def paired_intervals(y, a, b, block=168, repeats=2000, seed=42, comparisons=1):
    """Nominal and Bonferroni intervals of paired MAE differences."""
    y, a, b = [np.asarray(x, float) for x in (y, a, b)]
    if (
        not (len(y) == len(a) == len(b))
        or not 1 <= block <= len(y)
        or comparisons < 1
        or repeats < 2
    ):
        raise ValueError("Paarisvõrdluse mõõtmed või parameetrid ei sobi.")
    diff = np.abs(a - y) - np.abs(b - y)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(repeats)
    for i in range(repeats):
        starts = rng.integers(0, n, int(np.ceil(n / block)))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        draws[i] = diff[idx].mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    adjusted_low, adjusted_high = np.quantile(draws, [0.025 / comparisons, 1 - 0.025 / comparisons])
    return {
        "difference": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "adjusted_low": float(adjusted_low),
        "adjusted_high": float(adjusted_high),
        "block_hours": block,
        "family_comparisons": comparisons,
    }
