"""Earlier out-of-sample cases, scaled using training observations only."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def find_similar_cases(query, history, features, scaler, k=5):
    known = (history.target_time <= query.origin_time) & (history.origin_time < query.origin_time)
    pool = history.loc[known].copy()
    if pool.empty:
        return pool
    z = scaler.transform(pool[features])
    q = scaler.transform(pd.DataFrame([query[features].to_dict()], columns=features))
    pool["distance"] = np.linalg.norm(z - q[0], axis=1)
    return pool.nsmallest(k, "distance")[
        [
            "origin_time",
            "target_time",
            "target_next_hour",
            "prediction",
            "error",
            "distance",
            *features,
        ]
    ]


def fit_similarity_scaler(train, features):
    return StandardScaler().fit(train[features])
