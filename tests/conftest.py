"""Synthetic arithmetic fixtures only; never scientific result data."""

import numpy as np
import pandas as pd
import pytest

from src.common import read_config


@pytest.fixture
def cfg():
    return read_config()


@pytest.fixture
def hourly_data():
    n = 400
    d = pd.DataFrame(
        {
            "datetime": pd.date_range("2017-12-20", periods=n, freq="h"),
            "nat_demand": np.arange(n, dtype=float) + 500,
            "holiday": 0,
            "school": 1,
            "Holiday_ID": 0,
        }
    )
    for city in ("toc", "san", "dav"):
        for name, value in (("T2M", 25), ("QV2M", 0.015), ("TQL", 0.01), ("W2M", 3)):
            d[f"{name}_{city}"] = value + np.arange(n) / 1000
    return d
