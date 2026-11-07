"""Same-hour rolling bias correction, without SHAP or file output."""

import numpy as np
import pandas as pd


def bias_forecast(frame, setting, feedback):
    """Add a past same-hour mean error to the standard residual forecast.

    frame needs origin_time, target_time, hour, target_next_hour and
    prediction_residual. Each prediction uses only earlier target outcomes.
    Errors refer to the unchanged residual forecast, not this method's output.

    setting contains window_hours and strength; None returns the unchanged
    forecast. feedback contains min_history_points and warmup_hours, using the
    same cold-start convention as src.correction.adaptive_correction.
    Returns an array aligned to frame, without changing inputs or writing files.
    """
    reference = frame.prediction_residual.to_numpy(dtype=float)
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
