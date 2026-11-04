"""Signed residuals use prediction minus observation throughout."""

import pandas as pd
from scipy.stats import spearmanr

from .scenario_detector import LABELS
from .train_baseline import metrics


def scenario_summary(predictions, scenarios, threshold, min_n=40):
    rows = []
    reference = predictions.loc[scenarios.ordinary, "abs_error"].mean()
    for name in scenarios:
        p = predictions.loc[scenarios[name]]
        if p.empty:
            rows.append(
                {"scenario": name, "label": LABELS[name], "n": 0, "evidence": "Vaatlused puuduvad"}
            )
            continue
        rows.append(
            {
                "scenario": name,
                "label": LABELS[name],
                **metrics(p.target_next_hour, p.prediction),
                "large_error_share": float((p.abs_error > threshold).mean()),
                "abs_error_p50": float(p.abs_error.quantile(0.5)),
                "abs_error_p90": float(p.abs_error.quantile(0.9)),
                "abs_error_p95": float(p.abs_error.quantile(0.95)),
                "MAE_minus_ordinary": (
                    float(p.abs_error.mean() - reference) if pd.notna(reference) else None
                ),
                "evidence": (
                    "Piisav kirjeldavaks analüüsiks"
                    if len(p) >= min_n
                    else "Vähe vaatlusi; tugevaid järeldusi ei tehta"
                ),
            }
        )
    return pd.DataFrame(rows)


def residual_associations(frame, predictions):
    rows = []
    for name in ("temperature", "hour", "load_ramp_1h", "load_std_24", "holiday_known"):
        for outcome in ("error", "abs_error"):
            r, p = spearmanr(frame[name], predictions[outcome])
            rows.append(
                {
                    "feature": name,
                    "outcome": outcome,
                    "spearman": float(r) if pd.notna(r) else None,
                    "iid_pvalue_not_used": float(p) if pd.notna(p) else None,
                    "interpretation": "Kirjeldav seos; aegridade sõltuvuse tõttu iid-p-väärtust otsustamiseks ei kasutata.",
                }
            )
    return pd.DataFrame(rows)
