from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "continuous dataset.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

def prepare_data(path: Path) -> pd.DataFrame:
    """Parse, sort, and validate data without deleting rows or outliers."""
    df = pd.read_csv(path)

    required = {"datetime", "nat_demand"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["datetime"] = pd.to_datetime(
        df["datetime"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found")
    if not df["datetime"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Missing or irregular hourly timestamps found")
    if df.isna().any().any():
        raise ValueError("Missing values found; inspect them before deciding how to handle")

    numeric = df.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Infinite or non-finite numerical values found")
    if not df["nat_demand"].gt(0).all():
        raise ValueError("Non-positive load found; inspect before training")

    return df


def main() -> None:
    df = prepare_data(RAW_PATH)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")

    min_idx = df["nat_demand"].idxmin()
    print(f"Rows: {len(df):,}; columns: {len(df.columns)}")
    print(f"Period: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Minimum load: {df.loc[min_idx, 'nat_demand']:.2f}, "
        f"at {df.loc[min_idx, 'datetime']}"
    )
    print(f"Saved to: {CLEANED_PATH}")


if __name__ == "__main__":
    main()
