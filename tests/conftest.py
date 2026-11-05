import numpy as np
import pandas as pd
import pytest

from src.common import read_config


@pytest.fixture
def cfg():
    return read_config()


@pytest.fixture
def hourly_data():
    n = 1100
    t = np.arange(n)
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2017-11-25", periods=n, freq="h"),
            "nat_demand": 500 + t * 0.05 + 40 * np.sin(2 * np.pi * t / 24),
            "holiday": np.zeros(n),
            "school": np.ones(n),
            "Holiday_ID": np.zeros(n),
        }
    )
    for city in ("toc", "san", "dav"):
        for variable, value in (("T2M", 26.0), ("QV2M", 0.01), ("TQL", 0.2), ("W2M", 3.0)):
            frame[f"{variable}_{city}"] = value + 0.001 * t
    return frame
