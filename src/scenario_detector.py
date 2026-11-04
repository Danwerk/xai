"""Thresholds fitted on train only; retrospective labels are never inputs."""

import pandas as pd

LABELS = {
    "peak": "Tipukoormus (tagasivaateline)",
    "rapid_increase": "Kiire kasv (tagasivaateline)",
    "rapid_decrease": "Kiire langus (tagasivaateline)",
    "heat": "Äärmuslik kuumus (tagasivaateline)",
    "cold": "Madal temperatuur (tagasivaateline)",
    "holiday": "Püha (tagasivaateline)",
    "combined": "Koosesinevad äärmused (tagasivaateline)",
    "known_heat": "Varem teadaolev kuumus",
    "known_ramp": "Varem teadaolev kiire muutus",
    "ordinary": "Tavapärased tingimused",
}


def fit_thresholds(train, train_predictions):
    return {
        "peak": float(train.target_next_hour.quantile(0.95)),
        "increase": float(train.actual_ramp.quantile(0.95)),
        "decrease": float(train.actual_ramp.quantile(0.05)),
        "heat": float(train.actual_temperature.quantile(0.95)),
        "cold": float(train.actual_temperature.quantile(0.05)),
        "known_heat": float(train.temperature.quantile(0.95)),
        "known_ramp": float(train.load_ramp_1h.abs().quantile(0.95)),
        "large_error": float(train_predictions.abs_error.quantile(0.95)),
        "error_threshold_note": "Treeningu sobitusjääkide 95. protsentiil; võib üldistamisel olla liiga madal.",
    }


def detect_scenarios(f, thresholds):
    s = pd.DataFrame(index=f.index)
    s["peak"] = f.target_next_hour >= thresholds["peak"]
    s["rapid_increase"] = f.actual_ramp >= thresholds["increase"]
    s["rapid_decrease"] = f.actual_ramp <= thresholds["decrease"]
    s["heat"] = f.actual_temperature >= thresholds["heat"]
    s["cold"] = f.actual_temperature <= thresholds["cold"]
    s["holiday"] = f.holiday_retrospective.eq(1)
    s["combined"] = s[["peak", "rapid_increase", "rapid_decrease", "heat", "cold"]].sum(axis=1) >= 2
    s["known_heat"] = f.temperature >= thresholds["known_heat"]
    s["known_ramp"] = f.load_ramp_1h.abs() >= thresholds["known_ramp"]
    s["ordinary"] = ~s[["peak", "rapid_increase", "rapid_decrease", "heat", "cold", "holiday"]].any(
        axis=1
    )
    return s
