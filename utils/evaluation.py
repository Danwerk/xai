"""Common forecast losses, correction harm and clock-aligned paired intervals."""

import numpy as np
import pandas as pd

PRIMARY = "credit_feedback"
METHODS = {
    "base": "B0 Base forecast",
    "direct": "B1 Direct HistGBR",
    "residual": "B2 Standard residual correction",
    "fixed": "B3 Fixed-weight correction",
    "scalar_adaptive": "B4 Scalar least-squares adaptation (no XAI)",
    "standalone_feedback": "B5 Standalone component feedback",
    "credit_feedback": "B6 Proposed interaction-aware credit feedback",
    "hourly_bias": "Same-hour rolling bias correction (no XAI)",
    "matched_scalar": "A1 One credit weight, matched settings",
    "matched_standalone": "A2 No interaction credit, matched settings",
    "no_context": "A3 No hour conditioning, matched settings",
}


def persistence_forecast(frame):
    """One-hour naive forecast: reuse the most recently completed load interval.

    No fitting, target values, grouping or feedback. Use the caller's common
    evaluation rows; this does not claim coverage of otherwise unscored hours.
    """
    values = frame.load_now.to_numpy(dtype=float, copy=True)
    if (frame.empty or not np.isfinite(values).all()
            or not frame.target_time.sub(frame.origin_time).eq(pd.Timedelta(hours=1)).all()):
        raise ValueError("Persistence requires finite available load and a one-hour forecast horizon.")
    return values


def correction_diagnostics(frame, components, weights, prediction, *, tolerance=1e-6):
    """Retrospective activity, cancellation and loss changes, without model fitting.

    Explain the change from the SAME unweighted residual forecast. Sum of
    (weight - 1) * component must reconstruct that change. Offset is included
    as a component but is not a physical input feature. Opposite component
    changes can cancel: 1 - abs(sum(delta)) / sum(abs(delta)). No-activity
    conditional statistics are NaN, not an observed zero success rate.
    """
    predicted = np.asarray(prediction, dtype=float)
    if (frame.empty or not frame.index.equals(components.index)
            or not components.index.equals(weights.index)
            or not components.columns.equals(weights.columns)
            or predicted.shape != (len(frame),)
            or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(components.to_numpy()).all()
            or not np.isfinite(weights.to_numpy()).all()
            or not np.isfinite(predicted).all()
            or not weights.ge(0).all().all() or not weights.le(1).all().all()):
        raise ValueError("Diagnostics require finite aligned components, bounded weights and predictions.")
    reference = frame.prediction_residual.to_numpy(dtype=float)
    actual = frame.target_next_hour.to_numpy(dtype=float)
    if not np.isfinite([reference, actual]).all():
        raise ValueError("Diagnostics require finite reference forecasts and outcomes.")
    np.testing.assert_allclose(frame.base_prediction + components.sum(axis=1), reference,
                               rtol=1e-9, atol=1e-7)
    delta = components * (weights - 1)
    net = delta.sum(axis=1).to_numpy()
    np.testing.assert_allclose(predicted - reference, net, rtol=1e-9, atol=1e-7)
    gross = delta.abs().sum(axis=1).to_numpy()
    parts_changed, changed = gross > tolerance, np.abs(net) > tolerance
    ratio = np.divide(np.abs(net), gross, out=np.zeros_like(net), where=parts_changed)
    cancellation = np.where(parts_changed, np.clip(1 - ratio, 0, 1), np.nan)
    gain = abs(actual - reference) - abs(actual - predicted)
    details = pd.DataFrame({
        "parts_changed": parts_changed, "prediction_changed": changed,
        "net_change": net, "gross_change": gross, "cancellation_share": cancellation,
        "gain_vs_residual": gain,
        "helped": gain > tolerance, "harmed": gain < -tolerance,
        "changed_components": delta.abs().gt(tolerance).sum(axis=1).to_numpy(),
    }, index=frame.index)
    summary = {
        "n": len(frame), "changed_hours": int(changed.sum()),
        "changed_pct": float(100 * changed.mean()),
        "parts_changed_pct": float(100 * parts_changed.mean()),
        "mean_abs_change": float(np.abs(net).mean()),
        "mean_abs_change_when_changed": float(np.abs(net[changed]).mean()) if changed.any() else np.nan,
        "helped_pct_when_changed": float(100 * (gain[changed] > tolerance).mean()) if changed.any() else np.nan,
        "harmed_pct_when_changed": float(100 * (gain[changed] < -tolerance).mean()) if changed.any() else np.nan,
        "mae_gain_overall": float(gain.mean()),
        "mean_cancellation_when_parts_change": float(cancellation[parts_changed].mean()) if parts_changed.any() else np.nan,
    }
    return details, summary


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


def scenario_loss_interval(
    loss_difference, scenario_mask, block=168, repeats=2000, seed=42, comparisons=2
):
    """Resample complete time blocks and evaluate only eligible target hours.

    Differences are the new rule's absolute error minus the reference error.
    Scenario labels travel with each hour to preserve event clustering.
    """
    difference = np.asarray(loss_difference, dtype=float)
    mask = np.asarray(scenario_mask, dtype=bool)
    if len(difference) != len(mask) or not mask.any() or not 1 <= block <= len(mask):
        raise ValueError("Paired comparisons require aligned hourly errors and scenario labels.")

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
