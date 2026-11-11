"""TreeSHAP, official tabular LIME and grouped feature ablation, in memory."""

from time import perf_counter

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import Ridge

from .correction import adaptive_correction, attribution_group
from .models import correction_components
from .shap_explainer import explain_prediction
from .simple_correction import bias_forecast

EXPLAINERS = ("tree_shap", "lime", "feature_ablation")


def training_background(training, inputs, size, seed):
    """Sample inputs from this training fold only, without target outcomes."""
    if training.empty or not training.index.is_unique or not isinstance(size, int) or size < 1:
        raise ValueError("Background sampling needs nonempty indexed training data and positive size.")
    return training[inputs].sample(n=min(size, len(training)), random_state=seed)


def _lime_components(model, X, training, options, seed, progress):
    """Convert LIME coefficients to contributions in the model output's units.

    Continuous contribution = coefficient * standardized input. A categorical
    equality indicator is one at the explained row. Regression label 1 is the
    positive explanation. Keep all inputs; do not fit on evaluation outcomes.
    """
    categorical = [i for i, name in enumerate(X.columns)
                   if name in ("weekend", "holiday_known", "school_known")]
    explainer = LimeTabularExplainer(
        training[X.columns].to_numpy(dtype=float), mode="regression",
        feature_names=list(X.columns), categorical_features=categorical,
        discretize_continuous=False, sample_around_instance=True,
        feature_selection="none", random_state=seed,
        kernel_width=options["lime_kernel_width"] * np.sqrt(X.shape[1]),
    )
    values = np.empty(X.shape)
    offsets, scores = np.empty(len(X)), np.empty(len(X))
    # Row-addressed seeds make full-period, subset and repeated calls agree.
    keys = pd.util.hash_pandas_object(X.index, index=False).to_numpy(dtype=np.uint64)

    def predict(array):
        return model.predict(pd.DataFrame(array, columns=X.columns))

    for position, row in enumerate(X.to_numpy(dtype=float)):
        key = int(keys[position])
        row_seed = int(np.random.SeedSequence(
            [seed, key & 0xFFFFFFFF, key >> 32]
        ).generate_state(1)[0])
        explainer.random_state = np.random.RandomState(row_seed)
        explainer.base.random_state = explainer.random_state
        explanation = explainer.explain_instance(
            row, predict, num_features=X.shape[1], num_samples=options["lime_samples"],
            model_regressor=Ridge(alpha=options["lime_ridge_alpha"], fit_intercept=True),
        )
        coefficients = np.zeros(X.shape[1])
        for feature, coefficient in explanation.local_exp[1]:
            coefficients[feature] = coefficient
        encoded = (row - explainer.scaler.mean_) / explainer.scaler.scale_
        encoded[categorical] = 1.0
        values[position] = coefficients * encoded
        offsets[position] = explanation.intercept[1]
        scores[position] = explanation.score
        np.testing.assert_allclose(values[position].sum() + offsets[position],
                                   np.asarray(explanation.local_pred).item(), atol=1e-7)
        if progress and ((position + 1) % 512 == 0 or position + 1 == len(X)):
            print(f"  LIME {position + 1}/{len(X)}", flush=True)
    components = correction_components(pd.DataFrame(values, index=X.index, columns=X.columns), offsets)
    return components, scores


def _ablation_components(model, X, background, predicted, batch_rows):
    """Replace one whole group at a time; average effects over training rows.

    Group contribution = f(x) - mean_b f(x with that group taken from b).
    Interactions are not allocated as in SHAP. This need not be additive.
    """
    names = np.array([attribution_group(name) for name in X.columns])
    components = pd.DataFrame(index=X.index)
    reference = background.to_numpy(dtype=float)
    for group in dict.fromkeys(names):
        columns = np.flatnonzero(names == group)
        contributions = np.empty(len(X))
        for start in range(0, len(X), batch_rows):
            stop = min(start + batch_rows, len(X))
            hybrids = np.repeat(X.iloc[start:stop].to_numpy(dtype=float)[:, None, :],
                                len(reference), axis=1)
            hybrids[:, :, columns] = reference[None, :, columns]
            replaced = model.predict(pd.DataFrame(
                hybrids.reshape(-1, X.shape[1]), columns=X.columns
            )).reshape(stop - start, len(reference)).mean(axis=1)
            contributions[start:stop] = predicted[start:stop] - replaced
        components[group] = contributions
    components["offset"] = float(model.predict(background).mean())
    return components


def explain_components(model, X, method, training, *, options, seed=42, progress=False):
    """Return native explanation, explicit fixed remainder and measured runtime.

    Remainder = f(x) - native intercept - grouped contributions. Keep it unchanged
    in every forecast, never relabel it as an attribution or fit it on outcomes.
    Native fidelity is measured BEFORE completion. Closure is an adapter check.
    """
    if method not in EXPLAINERS:
        raise ValueError(f"Unknown explanation method: {method}")
    if (X.empty or not X.index.is_unique or not np.isfinite(X.to_numpy()).all()
            or not np.isfinite(training[X.columns].to_numpy()).all()):
        raise ValueError("Explanations require nonempty finite inputs with unique row identities.")
    start = perf_counter()
    predicted = model.predict(X)
    scores = np.full(len(X), np.nan)
    if method == "tree_shap":
        values, offset = explain_prediction(model, X)
        components = correction_components(values, offset)
    elif method == "lime":
        components, scores = _lime_components(model, X, training, options, seed, progress)
    else:
        background = training_background(training, list(X.columns), options["background_size"], seed)
        components = _ablation_components(model, X, background, predicted, options["batch_rows"])
    remainder = pd.Series(predicted - components.sum(axis=1).to_numpy(), index=X.index,
                          name="unallocated_remainder")
    if not np.isfinite(components.to_numpy()).all() or not np.isfinite(remainder).all():
        raise ValueError("Explanation produced nonfinite contributions or remainder.")
    reconstructed = components.sum(axis=1) + remainder
    np.testing.assert_allclose(reconstructed, predicted, rtol=1e-10, atol=1e-8)
    elapsed = perf_counter() - start
    return components, remainder, {
        "explanation_seconds": elapsed,
        "native_gap_mae": float(remainder.abs().mean()),
        "native_gap_max_abs": float(remainder.abs().max()),
        "native_gap_rmse": float(np.sqrt(np.mean(remainder.to_numpy() ** 2))),
        "reconstruction_max_abs": float(np.abs(reconstructed - predicted).max()),
        "lime_local_r2_mean": float(scores.mean()),
        "lime_local_r2_min": float(scores.min()),
    }


def component_forecast(frame, components, remainder, setting, feedback):
    """Unchanged credit rule; only native contributions/intercept are players."""
    if not frame.index.equals(remainder.index):
        raise ValueError("Remainder must align with the forecast rows.")
    adjusted = frame.copy()
    adjusted["base_prediction"] = frame.base_prediction + remainder
    prediction, _, weights = adaptive_correction(adjusted, components, setting, "credit", feedback)
    return prediction, weights


def compare_explainers(model, training, valid, inputs, setting, simple_setting, feedback,
                       *, options, seed=42, progress=False):
    """Three explanations on one model, with identical weighting policy.

    Timing excludes shared model fitting and initial base/residual forecasts.
    It includes explainer setup, its own model evaluations, checks and feedback.
    """
    if (training.empty or valid.empty
            or training.target_time.max() > valid.origin_time.min()):
        raise ValueError("Explanation training must precede the evaluation forecast origins.")
    X = valid[inputs]
    reference = valid.prediction_residual.to_numpy()
    np.testing.assert_allclose(reference, valid.base_prediction + model.predict(X), atol=1e-6)
    forecasts = {"base": valid.base_prediction.to_numpy(), "residual": reference.copy()}
    components, remainders, weights, checks, timing = {}, {}, {}, [], []
    start = perf_counter()
    forecasts["hourly_bias"] = bias_forecast(valid, simple_setting, feedback)
    seconds = perf_counter() - start
    timing.append({"method": "hourly_bias", "n": len(valid), "explanation_seconds": 0.0,
                   "feedback_seconds": seconds, "total_seconds": seconds,
                   "ms_per_forecast": 1000 * seconds / len(valid)})
    for method in EXPLAINERS:
        if progress:
            print(f"  {method}: {len(X)} rows", flush=True)
        parts, remainder, info = explain_components(
            model, X, method, training, options=options, seed=seed, progress=progress
        )
        components[method], remainders[method] = parts, remainder
        np.testing.assert_allclose(valid.base_prediction + parts.sum(axis=1) + remainder,
                                   reference, rtol=1e-10, atol=1e-8)
        start = perf_counter()
        forecasts[method], weights[method] = component_forecast(
            valid, parts, remainder, setting, feedback
        )
        feedback_seconds = perf_counter() - start
        total = info["explanation_seconds"] + feedback_seconds
        timing.append({"method": method, "n": len(valid),
                       "explanation_seconds": info["explanation_seconds"],
                       "feedback_seconds": feedback_seconds, "total_seconds": total,
                       "ms_per_forecast": 1000 * total / len(valid)})
        checks.append({"method": method, "n": len(valid), **info})
        if progress:
            print(f"  {method}: done in {total:.2f}s", flush=True)
    background = training_background(training, inputs, options["background_size"], seed)
    return {"forecasts": forecasts, "components": components, "remainders": remainders,
            "weights": weights, "timing": pd.DataFrame(timing), "checks": pd.DataFrame(checks),
            "background": background,
            "background_last_target": training.loc[background.index, "target_time"].max()}
