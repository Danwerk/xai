"""Same-hour rolling bias correction and one fixed two-stage comparison, in memory."""

import numpy as np
import pandas as pd

from .correction import adaptive_correction


def bias_forecast(frame, setting, feedback, *, reference=None):
    """Add a past same-hour mean error to a supplied forecast stream.

    frame needs origin_time, target_time, hour, target_next_hour and
    prediction_residual (unless reference is supplied). Each prediction uses
    only earlier target outcomes. Errors refer to the supplied first-stage
    stream, not to this method's output or another method's errors.

    setting contains window_hours and strength; None returns the unchanged
    forecast. feedback contains min_history_points and warmup_hours, using the
    same cold-start convention as utils.correction.adaptive_correction.
    Returns an array aligned to frame, without changing inputs or writing files.
    """
    if reference is None:
        reference = frame.prediction_residual.to_numpy(dtype=float)
    elif isinstance(reference, pd.Series) and not reference.index.equals(frame.index):
        raise ValueError("Reference forecasts must align with the frame index.")
    reference = np.asarray(reference, dtype=float)
    if reference.shape != (len(frame),) or not np.isfinite(reference).all():
        raise ValueError("Reference forecasts must be a finite aligned one-dimensional vector.")
    if setting is None:
        return reference.copy()
    times = pd.DatetimeIndex(frame.target_time)
    if (
        frame.empty
        or not frame.index.is_unique
        or times.hasnans
        or not times.is_monotonic_increasing
        or times.has_duplicates
        or not frame.target_time.sub(frame.origin_time).eq(pd.Timedelta(hours=1)).all()
        or not np.isfinite(reference).all()
        or not np.isfinite(frame.target_next_hour).all()
        or frame.hour.isna().any()
        or not np.isfinite(setting["window_hours"])
        or setting["window_hours"] <= 0
        or not 0 <= setting["strength"] <= 1
        or feedback["min_history_points"] < 1
        or feedback["warmup_hours"] < 0
    ):
        raise ValueError("Bias correction needs ordered finite one-hour observations and valid settings.")
    errors = pd.Series(frame.target_next_hour.to_numpy() - reference, index=frame.index)
    correction = pd.Series(0.0, index=frame.index)
    for indexes in frame.groupby("hour").groups.values():
        past = pd.Series(
            errors.loc[indexes].to_numpy(),
            index=pd.DatetimeIndex(frame.loc[indexes, "target_time"]),
        )
        correction.loc[indexes] = (
            past.rolling(
                f"{setting['window_hours']}h",
                closed="left",
                min_periods=feedback["min_history_points"],
            )
            .mean()
            .fillna(0)
            .to_numpy()
        )
    warmup = times < times.min() + pd.Timedelta(hours=feedback["warmup_hours"])
    correction.loc[warmup] = 0
    return reference + setting["strength"] * correction.to_numpy()


def combined_forecasts(frame, components, xai_setting, bias_setting, feedback):
    """One fixed XAI + bias experiment and its matched no-XAI control.

    XAI and scalar feedback share settings. The scalar control consumes only
    the full residual prediction, without any explanation. Both second stages
    use their OWN first-stage past errors with the same bias settings.
    """
    residual = frame.prediction_residual.to_numpy(dtype=float)
    np.testing.assert_allclose(frame.base_prediction + components.sum(axis=1), residual, atol=1e-6)
    xai, _, _ = adaptive_correction(frame, components, xai_setting, "credit", feedback)
    total = pd.DataFrame({"total": residual - frame.base_prediction.to_numpy()}, index=frame.index)
    scalar, _, _ = adaptive_correction(frame, total, xai_setting, "scalar_credit", feedback)
    return {
        "base": frame.base_prediction.to_numpy(),
        "residual": residual.copy(),
        "hourly_bias": bias_forecast(frame, bias_setting, feedback),
        "credit_feedback": xai,
        "matched_scalar": scalar,
        "xai_bias": bias_forecast(frame, bias_setting, feedback, reference=xai),
        "scalar_bias": bias_forecast(frame, bias_setting, feedback, reference=scalar),
    }
