"""Proposed realised-credit feedback and its matched ablations.

Weights use only outcomes before the current target. Credit audits the potential
full correction, including components suppressed in the deployed forecast.
"""

import numpy as np
import pandas as pd

from .shap_explainer import repair_gain_attributions


def attribution_group(name):
    if name.startswith("known_error"):
        return "error_memory"
    if name == "base_prediction":
        return "forecast_anchor"
    if name in ("load_day", "load_week"):
        return "seasonal_load"
    if name in ("load_now", "load_lag_2", "load_lag_3"):
        return "recent_load"
    if name.startswith("hour_"):
        return "clock"
    if name in ("temperature", "humidity", "wind"):
        return "weather"
    if name.startswith(("load_ramp", "load_mean", "load_std")):
        return "dynamics"
    return "calendar"


def past_average(values, frame, window_hours, by_hour, min_points, half_life_hours=None):
    """Actual clock window; gaps never extend memory into older calendar days."""
    if half_life_hours is not None and (not np.isfinite(half_life_hours) or half_life_hours <= 0):
        raise ValueError("Memory half-life must be finite and positive.")
    values = pd.DataFrame(values, index=frame.index)
    result = pd.DataFrame(np.nan, index=frame.index, columns=values.columns)
    partitions = frame.groupby("hour").groups.values() if by_hour else [frame.index]
    for indexes in partitions:
        group = values.loc[indexes].copy()
        group.index = pd.DatetimeIndex(frame.loc[indexes, "target_time"])
        if half_life_hours is None:
            averaged = (
                group.rolling(f"{window_hours}h", closed="left", min_periods=min_points)
                .mean()
                .to_numpy()
            )
        else:
            averaged = _decayed_average(group, window_hours, min_points, half_life_hours)
        result.loc[indexes] = averaged
    return result


def _decayed_average(group, window_hours, min_points, half_life_hours):
    """Truncated exponential memory: weight = 2 ** (-age_hours / half_life).

    Update only AFTER reading the past average. The left window endpoint is
    included, matching rolling(closed="left"), and missing values do not count.
    """
    times = pd.DatetimeIndex(group.index).as_unit("ns").asi8
    values = group.to_numpy(dtype=float)
    valid = np.isfinite(values)
    values = np.where(valid, values, 0)
    output = np.full(values.shape, np.nan)
    numerator, mass = np.zeros(values.shape[1]), np.zeros(values.shape[1])
    counts = np.zeros(values.shape[1], dtype=int)
    left, hour = 0, pd.Timedelta(hours=1).value
    rate = np.log(0.5) / (half_life_hours * hour)
    for i, time in enumerate(times):
        if i:
            decay = np.exp(rate * (time - times[i - 1]))
            numerator *= decay
            mass *= decay
        while left < i and times[left] < time - window_hours * hour:
            weight = np.exp(rate * (time - times[left]))
            numerator -= weight * values[left]
            mass -= weight * valid[left]
            counts -= valid[left]
            left += 1
        if left == i:
            numerator.fill(0)
            mass.fill(0)
        eligible = (counts >= min_points) & (mass > 0)
        output[i, eligible] = numerator[eligible] / mass[eligible]
        numerator += values[i]
        mass += valid[i]
        counts += valid[i]
    return output


def adaptive_correction(
    frame, components, setting, mode, feedback, *, credit_loss="squared", half_life_hours=None
):
    """Modes: credit (proposed), standalone, scalar_credit, scalar_ls (B4)."""
    if mode not in ("credit", "standalone", "scalar_credit", "scalar_ls"):
        raise ValueError(f"Unknown feedback mode: {mode}")
    if credit_loss not in ("squared", "absolute") or (
        credit_loss != "squared" and mode != "credit"
    ):
        raise ValueError("Alternative credit loss is supported only for component credit.")
    times = pd.DatetimeIndex(frame.target_time)
    if (
        not frame.index.equals(components.index)
        or times.has_duplicates
        or not times.is_monotonic_increasing
        or not frame.target_time.sub(frame.origin_time).eq(pd.Timedelta(hours=1)).all()
        or not np.isfinite(components.to_numpy()).all()
    ):
        raise ValueError(
            "Feedback needs aligned finite components and chronological one-hour targets."
        )
    if setting is None:
        weights = components * 0 + 1
    else:
        if setting["window_hours"] <= 0 or not 0 <= setting["strength"] <= 1:
            raise ValueError("Invalid feedback window or strength.")
        scalar = mode.startswith("scalar")
        edits = components.sum(axis=1).to_frame("total") if scalar else components
        residual = (frame.target_next_hour - frame.base_prediction).to_numpy()
        c = edits.to_numpy()

        def average(v):
            return past_average(
                v,
                frame,
                setting["window_hours"],
                setting["by_hour"],
                feedback["min_history_points"],
                half_life_hours,
            )

        if mode == "scalar_ls":
            # Historical least-squares fit of ONE multiplier: no SHAP or credit allocation.
            optimum = (average(c * residual[:, None]) / average(c * c).clip(lower=1e-12)).clip(0, 1)
            local_weights = 1 + setting["strength"] * (optimum - 1)
        else:
            if mode == "standalone":
                credits = c * (2 * residual[:, None] - c)
            else:
                credits = repair_gain_attributions(
                    frame.target_next_hour, frame.base_prediction, c, credit_loss
                )
            local_weights = (
                1 + setting["strength"] * average(credits) / average(abs(credits)).clip(lower=1e-12)
            ).clip(0, 1)
        local_weights = local_weights.fillna(1)
        warm = times < times.min() + pd.Timedelta(hours=feedback["warmup_hours"])
        local_weights.loc[warm] = 1
        weights = pd.DataFrame(
            (
                np.repeat(local_weights.to_numpy(), len(components.columns), axis=1)
                if scalar
                else local_weights.to_numpy()
            ),
            index=components.index,
            columns=components.columns,
        )
    effective = components * weights
    predicted = frame.base_prediction.to_numpy() + effective.sum(axis=1).to_numpy()
    return predicted, effective, weights
