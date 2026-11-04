"""Validate source observations without deleting unusual values."""

from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, write_json

RAW_PATH = ROOT / "data/raw/continuous dataset.csv"
WEATHER = [f"{v}_{c}" for c in ("toc", "san", "dav") for v in ("T2M", "QV2M", "TQL", "W2M")]


def load_and_audit(path):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Algandmestik puudub: {path}. Lisa tegelik CSV; sünteetilist asendust ei looda."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"datetime", "nat_demand", *WEATHER, "holiday", "school", "Holiday_ID"}
    if missing := required - set(df):
        raise ValueError(f"Puuduvad veerud: {sorted(missing)}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    if df.datetime.isna().any() or df.datetime.dt.tz is not None:
        raise ValueError("Oodatakse kehtivaid kohaliku Panama aja ajatempleid ilma nihketa.")
    original_types = df.dtypes.astype(str).to_dict()
    for col in required - {"datetime"}:
        df[col] = pd.to_numeric(df[col], errors="raise")
    sorted_before = df.datetime.is_monotonic_increasing
    duplicate_rows = int(df.duplicated().sum())
    df = df.sort_values("datetime").reset_index(drop=True)
    numeric = df.select_dtypes("number")
    delta = df.datetime.diff().dropna()
    audit = {
        "rows": len(df),
        "columns": len(df.columns),
        "first": str(df.datetime.min()),
        "last": str(df.datetime.max()),
        "original_dtypes": original_types,
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": int(df.datetime.duplicated().sum()),
        "non_hourly_intervals": int(delta.ne(pd.Timedelta(hours=1)).sum()),
        "missing_hours": int(
            len(
                pd.date_range(df.datetime.min(), df.datetime.max(), freq="h").difference(
                    df.datetime
                )
            )
        ),
        "was_sorted": bool(sorted_before),
        "nonfinite": int((~np.isfinite(numeric.to_numpy())).sum()),
        "nonpositive_load": int(df.nat_demand.le(0).sum()),
        "load_below_200": int(df.nat_demand.lt(200).sum()),
        "flag_changes_within_day": {
            c: int(df.groupby(df.datetime.dt.date)[c].nunique().gt(1).sum())
            for c in ("holiday", "school")
        },
        "holiday_disagreements": int(((df.Holiday_ID > 0) != (df.holiday == 1)).sum()),
        "timezone": "Ajatemplid märgendamata; Panama kohalik aeg UTC−05:00 on allikakirjeldusel põhinev eeldus.",
        "outlier_policy": "Ebatavalisi positiivseid koormusi ei eemaldata ega asendata.",
    }
    return df, audit


def validate_for_modeling(df, audit):
    if (
        any(
            audit[k]
            for k in (
                "duplicate_timestamps",
                "non_hourly_intervals",
                "nonfinite",
                "nonpositive_load",
            )
        )
        or df.isna().any().any()
    ):
        raise ValueError(
            "Andmekvaliteet ei võimalda vaikimisi modelleerimist; vaata quality.json. Andmeid ei parandata automaatselt."
        )
    for c in ("holiday", "school"):
        if not df[c].isin([0, 1]).all():
            raise ValueError(f"{c} ei ole binaarne.")
    return df


def prepare_data(path=RAW_PATH):
    df, audit = load_and_audit(path)
    return validate_for_modeling(df, audit)


if __name__ == "__main__":
    data, quality = load_and_audit(RAW_PATH)
    write_json(ROOT / "results/eda/quality.json", quality)
    validate_for_modeling(data, quality)
    print(f"Kontrollitud {len(data)} vaatlust; algandmeid ei muudetud.")
