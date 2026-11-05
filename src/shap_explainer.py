"""TreeSHAP correction decomposition and exact realised repair credits."""

import numpy as np
import pandas as pd
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
