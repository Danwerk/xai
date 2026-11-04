"""Prediction explanations, grouped permutation and context matched contrasts."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import shap


def repair_gain_attributions(actual, baseline, corrections, loss="absolute"):
    """Exact responsibility for realised gain of additive correction modules.

    Players switch fitted corrections on/off; raw inputs are never masked.
    v(S) = L(y, baseline) - L(y, baseline + sum(c_j for j in S)).
    These are retrospective repair credits, NOT SHAP values of the new forecast
    or causal responsibility for the original error. Squared loss has a closed
    form; absolute loss enumerates the small module game exactly.
    """
    y, base, c = map(lambda x: np.asarray(x, dtype=float), (actual, baseline, corrections))
    if (
        y.ndim != 1
        or base.shape != y.shape
        or c.ndim != 2
        or len(c) != len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(base).all()
        or not np.isfinite(c).all()
        or not 1 <= c.shape[1] <= 10
    ):
        raise ValueError("Repair credits require aligned finite values and 1–10 modules.")
    residual = y - base
    total = c.sum(axis=1)
    if loss == "squared":
        credits = c * (2 * residual - total)[:, None]
        gain = total * (2 * residual - total)
    elif loss == "absolute":
        from math import comb

        players = c.shape[1]
        masks = np.arange(2**players)
        membership = ((masks[:, None] >> np.arange(players)) & 1).astype(float)
        value = np.abs(residual)[:, None] - np.abs(residual[:, None] - c @ membership.T)
        credits = np.zeros_like(c)
        for j in range(players):
            absent = masks[(masks & (1 << j)) == 0]
            weights = np.array(
                [1 / (players * comb(players - 1, int(s).bit_count())) for s in absent]
            )
            credits[:, j] = (value[:, absent | (1 << j)] - value[:, absent]) @ weights
        gain = np.abs(residual) - np.abs(residual - total)
    else:
        raise ValueError("Supported repair losses: absolute, squared.")
    if not np.allclose(credits.sum(axis=1), gain, atol=1e-7, rtol=1e-9):
        raise ValueError("Repair responsibility does not reconstruct realised gain.")
    return credits


def explain_prediction(model, X):
    explainer = shap.TreeExplainer(
        model, feature_perturbation="tree_path_dependent", model_output="raw"
    )
    explanation = explainer(X, check_additivity=True)
    values = np.asarray(explanation.values)
    base = np.broadcast_to(np.asarray(explanation.base_values).reshape(-1), (len(X),))
    reconstructed = values.sum(axis=1) + base
    predicted = model.predict(X)
    if not np.allclose(reconstructed, predicted, rtol=1e-7, atol=1e-6):
        raise ValueError("SHAP-i liidetavuse kontroll ebaõnnestus.")
    return pd.DataFrame(values, index=X.index, columns=X.columns), base


def permutation_importance(model, X, y, groups, repeats=5, seed=42):
    rng = np.random.default_rng(seed)
    baseline = np.abs(model.predict(X) - np.asarray(y)).mean()
    rows = []
    for name, cols in groups.items():
        for repeat in range(repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled.loc[:, cols] = X[cols].iloc[order].to_numpy()
            delta = float(np.abs(model.predict(shuffled) - np.asarray(y)).mean() - baseline)
            rows.append(
                {
                    "group": name,
                    "repeat": repeat,
                    "MAE_increase": delta,
                    "method": "Ühine juhuslik permutatsioon; sõltuvusstruktuur võib muutuda",
                }
            )
    return pd.DataFrame(rows)


def compare_explanation_groups(values, frame, predictions, threshold):
    """Match large/small errors within target hour and weekday; exclude nearby days.

    Matching standardization uses only the diagnosis context, never test data.
    This retrospective descriptive contrast does not identify error causes.
    """
    large = predictions.abs_error > threshold
    small = ~large
    context = frame[["temperature", "load_now", "hour", "dayofweek"]]
    z = StandardScaler().fit_transform(context)
    pairs = []
    for i in np.flatnonzero(large):
        eligible = np.flatnonzero(
            small
            & frame.hour.eq(frame.loc[i, "hour"])
            & frame.dayofweek.eq(frame.loc[i, "dayofweek"])
            & (frame.origin_time - frame.loc[i, "origin_time"]).abs().gt(pd.Timedelta(days=1))
        )
        if not len(eligible):
            continue
        distance = np.linalg.norm(z[eligible] - z[i], axis=1)
        j = eligible[np.argmin(distance)]
        if distance.min() <= 1.5:
            pairs.append((i, j))
    rows = []
    for col in values:
        a = [abs(values.loc[i, col]) for i, _ in pairs]
        b = [abs(values.loc[j, col]) for _, j in pairs]
        rows.append(
            {
                "feature": col,
                "pairs": len(pairs),
                "large_mean_abs_shap": float(np.mean(a)) if a else None,
                "matched_small_mean_abs_shap": float(np.mean(b)) if b else None,
                "difference": float(np.mean(np.asarray(a) - b)) if a else None,
                "interpretation": "Sama tunni ja nädalapäeva sobitatud kontekst; korduvate kontrollidega kirjeldav võrdlus, mitte veapõhjuse tõend.",
            }
        )
    return pd.DataFrame(rows)


def analyze_feature_importance_stability(explanations):
    """Compare ranks on the SAME anchors across models / temporal subsets."""
    rows = []
    names = list(explanations)
    for i, a in enumerate(names):
        x = explanations[a].abs().mean()
        for b in names[i + 1 :]:
            y = explanations[b].abs().mean().reindex(x.index)
            rho = spearmanr(x, y).statistic
            k = min(5, len(x))
            rows.append(
                {
                    "sample_a": a,
                    "sample_b": b,
                    "rank_spearman": float(rho) if np.isfinite(rho) else None,
                    "top5_jaccard": len(set(x.nlargest(k).index) & set(y.nlargest(k).index))
                    / len(set(x.nlargest(k).index) | set(y.nlargest(k).index)),
                }
            )
    return pd.DataFrame(rows)
