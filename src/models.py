"""GBR/HistGBR base; direct and residual HistGBR share OOF rows and inputs."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from .correction import attribution_group
from .feature_engineering import ERROR_FEATURES
from .shap_explainer import explain_prediction


def make_base_model(cfg):
    """HistGBR reuses the fixed HistGBR settings from the development comparison."""
    if cfg["base_model_family"] == "gbr":
        return GradientBoostingRegressor(**cfg["base_model"], random_state=cfg["seed"])
    if cfg["base_model_family"] == "histgbr":
        return HistGradientBoostingRegressor(**cfg["residual_model"], random_state=cfg["seed"])
    raise ValueError("base_model_family must be gbr or histgbr.")


def base_forecast(training, valid, columns, cfg):
    if training.empty or valid.empty or training.target_time.max() > valid.origin_time.min():
        raise ValueError("Base model training targets were not available at the forecast origin.")
    model = make_base_model(cfg)
    model.fit(training[columns], training.target_next_hour)
    return model.predict(valid[columns])


def residual_forecasts(training, valid, columns, cfg):
    if training.empty or valid.empty or training.target_time.max() > valid.origin_time.min():
        raise ValueError("Residual training overlaps evaluation.")
    inputs = columns + ["base_prediction"] + ERROR_FEATURES
    options = dict(cfg["residual_model"], random_state=cfg["seed"])
    direct = HistGradientBoostingRegressor(**options).fit(
        training[inputs], training.target_next_hour
    )
    residual = HistGradientBoostingRegressor(**options).fit(
        training[inputs], training.target_next_hour - training.base_prediction
    )
    correction = residual.predict(valid[inputs])
    values, expected = explain_prediction(residual, valid[inputs])
    groups = [attribution_group(c) for c in inputs]
    components = pd.DataFrame(
        {
            group: values.loc[:, np.array(groups) == group].sum(axis=1).to_numpy()
            for group in dict.fromkeys(groups)
        },
        index=valid.index,
    )
    components["offset"] = expected
    np.testing.assert_allclose(components.sum(axis=1), correction, atol=1e-6)
    forecasts = {
        "base": valid.base_prediction.to_numpy(),
        "direct": direct.predict(valid[inputs]),
        "residual": valid.base_prediction.to_numpy() + correction,
    }
    return forecasts, components, inputs
