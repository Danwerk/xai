"""Common forecast losses, correction harm and clock-aligned paired intervals."""

import numpy as np
import pandas as pd

PRIMARY = "credit_feedback"
METHODS = {
    "base": "B0 Base GBR",
    "direct": "B1 Direct HistGBR",
    "residual": "B2 Standard residual correction",
    "fixed": "B3 Fixed-weight correction",
    "scalar_adaptive": "B4 Scalar least-squares adaptation (no XAI)",
    "standalone_feedback": "B5 Standalone component feedback",
    "credit_feedback": "B6 Proposed interaction-aware credit feedback",
    "matched_scalar": "A1 One credit weight, matched settings",
    "matched_standalone": "A2 No interaction credit, matched settings",
    "no_context": "A3 No hour conditioning, matched settings",
}


def prediction_metrics(actual, predicted):
    y, p = np.asarray(actual, float), np.asarray(predicted, float)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isfinite([y, p]).all():
        raise ValueError("Metrics need aligned nonempty finite vectors.")
    error = p - y
    return {
        "n": len(y),
        "MAE": float(abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
    }


def forecast_metrics(actual, predicted, base, residual, tolerance=1e-9):
    y, p, b, r = [np.asarray(x, float) for x in (actual, predicted, base, residual)]
    if (
        y.ndim != 1
        or not len(y)
        or any(x.shape != y.shape or not np.isfinite(x).all() for x in (y, p, b, r))
    ):
        raise ValueError("Metrics need aligned nonempty finite vectors.")
    loss = abs(p - y)
    result = prediction_metrics(y, p)
    for name, reference in (("base", b), ("residual", r)):
        reference_loss = abs(reference - y)
        difference = loss - reference_loss
        harmful = difference > tolerance
        positive = np.where(harmful, difference, 0)
        denom = reference_loss.mean()
        result.update(
            {
                f"improvement_vs_{name}_pct": (
                    float(100 * (1 - loss.mean() / denom)) if denom else None
                ),
                f"harmful_rate_vs_{name}": float(harmful.mean()),
                # Unconditional mean positive loss increase, includes zero on other hours.
                f"harmful_magnitude_vs_{name}": float(positive.mean()),
                f"harmful_conditional_mean_vs_{name}": (
                    float(difference[harmful].mean()) if harmful.any() else 0.0
                ),
            }
        )
    return result


def scope_masks(frame, thresholds):
    return {
        "overall": np.ones(len(frame), dtype=bool),
        "rapid_increase": frame.actual_ramp.ge(thresholds["increase"]).to_numpy(),
        "rapid_decrease": frame.actual_ramp.le(thresholds["decrease"]).to_numpy(),
    }


def score_forecasts(frame, forecasts, thresholds, cfg, stage):
    rows = []
    y = frame.target_next_hour.to_numpy()
    for name, p in forecasts.items():
        for scope, mask in scope_masks(frame, thresholds).items():
            if mask.any():
                rows.append(
                    {
                        "method": name,
                        "scope": scope,
                        "stage": stage,
                        "year": int(frame.target_time.dt.year.iloc[0]),
                        **forecast_metrics(
                            y[mask],
                            np.asarray(p)[mask],
                            np.asarray(forecasts["base"])[mask],
                            np.asarray(forecasts["residual"])[mask],
                            cfg["evaluation"]["harm_tolerance"],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compare_forecasts(frame, forecasts, thresholds, cfg):
    rows = []
    masks = scope_masks(frame, thresholds)
    # Credit/memory and parameter sensitivity are descriptive ablations, not
    # extra hypotheses in the original baseline-comparison bootstrap family.
    references = [name for name in METHODS if name != PRIMARY]
    comparisons = len(references) * sum(mask.any() for mask in masks.values())
    y = frame.target_next_hour.to_numpy()
    for reference in references:
        difference = abs(forecasts[PRIMARY] - y) - abs(forecasts[reference] - y)
        for scope, mask in masks.items():
            if not mask.any():
                continue
            for block in cfg["evaluation"]["bootstrap_blocks_hours"]:
                if block > len(
                    pd.date_range(frame.target_time.min(), frame.target_time.max(), freq="h")
                ):
                    raise ValueError("Evaluation period is shorter than a bootstrap block.")
                rows.append(
                    {
                        "A": PRIMARY,
                        "B": reference,
                        "scope": scope,
                        **clock_loss_interval(
                            frame.target_time,
                            difference,
                            mask,
                            block=block,
                            repeats=cfg["evaluation"]["bootstrap_repeats"],
                            seed=cfg["seed"],
                            comparisons=comparisons,
                        ),
                    }
                )
    return pd.DataFrame(rows)


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
