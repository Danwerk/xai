"""Origin t predicts t+1. Every non-calendar input is available at or before t."""

import numpy as np
import pandas as pd

from .common import ROOT, read_json, sha256
from .data_preprocessing import load_and_audit, validate_for_modeling


def load_run_features(run, *, expanded=False):
    """Koosta tunnused algandmetest ja katse seadistustest, vahefaile loomata.

    Mudelit ei treenita. Algandmete ja tunnusearvutuse kontrollsummad hoiavad
    ära uute sisendite kogemata sidumise varasemate prognoosidega.
    """
    manifest = read_json(run / "manifest.json")
    config_name = "original_experiment" if expanded else "experiment"
    config = read_json(run / f"configs/{config_name}.json")
    source_config = read_json(run / "configs/expanded_study.json") if expanded else config
    raw_path = ROOT / source_config["raw_path"]
    if sha256(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Algandmed on pärast katset muutunud. Käivita uuring uuesti.")
    modules = ["src/data_preprocessing.py", "src/feature_engineering.py"]
    if expanded:
        modules.append("src/expanded_features.py")
    source_hashes = {
        name.replace("\\", "/"): digest for name, digest in manifest["source_sha256"].items()
    }
    if any(sha256(ROOT / name) != source_hashes[name] for name in modules):
        raise ValueError("Tunnuste arvutamise kood on muutunud. Käivita uuring uuesti.")
    raw, quality = load_and_audit(raw_path)
    validate_for_modeling(raw, quality)
    if expanded:
        from .expanded_features import build_expanded_features

        return build_expanded_features(raw, config)[0]
    return build_features(raw, config)


def build_features(df, cfg):
    d = df.set_index("datetime").copy()
    if (
        d.index.has_duplicates
        or not d.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Tunnused nõuavad regulaarset, järjestatud tunniseeriat.")
    delay = int(cfg["load_delay_hours"])
    weather_delay = int(cfg["weather_delay_hours"])
    if delay < 0 or weather_delay < 1:
        raise ValueError("Koormuse viivitus peab olema >=0 ja ilma viivitus >=1 tund.")
    load = d.nat_demand.shift(delay)
    temp = d[[f"T2M_{c}" for c in ("toc", "san", "dav")]].mean(axis=1).shift(weather_delay)
    target_time = d.index + pd.Timedelta(hours=1)
    f = pd.DataFrame(index=d.index)
    f["origin_time"] = d.index
    f["target_time"] = target_time
    f["target_next_hour"] = d.nat_demand.shift(-1)
    f["load_now"] = load
    f["load_day"] = d.nat_demand.shift(23 + delay)
    f["load_week"] = d.nat_demand.shift(167 + delay)
    f["hour"] = target_time.hour
    f["dayofweek"] = target_time.dayofweek
    for name, val, period in (
        ("hour", target_time.hour, 24),
        ("dow", target_time.dayofweek, 7),
        ("month", target_time.month - 1, 12),
    ):
        f[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
    f["weekend"] = (target_time.dayofweek >= 5).astype(int)
    f["temperature"] = temp
    f["humidity"] = (
        d[[f"QV2M_{c}" for c in ("toc", "san", "dav")]].mean(axis=1).shift(weather_delay)
    )
    f["wind"] = d[[f"W2M_{c}" for c in ("toc", "san", "dav")]].mean(axis=1).shift(weather_delay)
    f["temperature_sq"] = temp**2
    f["cooling_proxy"] = (temp - 24).clip(lower=0)
    f["temperature_hour_sin"] = temp * f.hour_sin
    f["temperature_hour_cos"] = temp * f.hour_cos
    f["temperature_mean_24"] = temp.rolling(24, min_periods=24).mean()
    f["temperature_change_24"] = temp.diff(24)
    f["load_ramp_1h"] = load.diff()
    f["load_ramp_3h"] = load.diff(3)
    f["load_lag_2"] = load.shift(2)
    f["load_lag_3"] = load.shift(3)
    for window in (6, 24):
        f[f"load_mean_{window}"] = load.rolling(window, min_periods=window).mean()
        f[f"load_std_{window}"] = load.rolling(window, min_periods=window).std(ddof=1)
    same_day = d.index.normalize() == target_time.normalize()
    f["holiday_known"] = d.holiday.where(same_day, 0)
    f["school_known"] = d.school.where(same_day, 0)
    f["holiday_retrospective"] = d.holiday.shift(-1)
    f["actual_temperature"] = d[[f"T2M_{c}" for c in ("toc", "san", "dav")]].mean(axis=1).shift(-1)
    f["actual_ramp"] = d.nat_demand.shift(-1) - d.nat_demand
    used = cfg["baseline_features"] + [c for g in cfg["candidate_groups"].values() for c in g]
    forbidden = {
        "target_next_hour",
        "target_time",
        "origin_time",
        "actual_ramp",
        "actual_temperature",
        "holiday_retrospective",
    }
    if forbidden.intersection(used):
        raise ValueError(
            "Mudeli tunnustesse ei tohi konfigureerida sihtmärki ega tagasivaatelisi silte."
        )
    return f.dropna(subset=["target_next_hour", *used]).reset_index(drop=True)


def split_features(f, cfg):
    t = f.target_time
    bounds = [pd.Timestamp(cfg[k]) for k in ("train_end", "diagnosis_end", "selection_end")]
    masks = [
        t < bounds[0],
        (t >= bounds[0]) & (t < bounds[1]),
        (t >= bounds[1]) & (t < bounds[2]),
        t >= bounds[2],
    ]
    splits = {
        name: f.loc[mask].reset_index(drop=True)
        for name, mask in zip(("train", "diagnosis", "selection", "test"), masks)
    }
    if any(d.empty for d in splits.values()):
        raise ValueError(
            "Fikseeritud ajaline jaotus jätab tühja perioodi. Muuda protokolli enne katseid."
        )
    for left, right in zip(list(splits.values())[:-1], list(splits.values())[1:]):
        assert left.target_time.max() <= right.origin_time.min()
    return splits


def feature_manifest(cfg):
    rows = []
    for group, names in {"baseline": cfg["baseline_features"], **cfg["candidate_groups"]}.items():
        for name in names:
            weather = any(s in name for s in ("temperature", "humidity", "wind", "cooling"))
            rows.append(
                {
                    "feature": name,
                    "group": group,
                    "timing": (
                        f"Ilm t−{cfg['weather_delay_hours']} või varasem"
                        if weather
                        else "Koormus t−viivitus või varasem; kalendriaeg t+1 on teada"
                    ),
                    "limitation": (
                        "Ilma avaldamisviivitus ei ole verifitseeritud"
                        if weather
                        else "Koormuse kohene kättesaadavus on eeldus; püha/kool: sama päeva viimane teadaolev märgend"
                    ),
                }
            )
    return pd.DataFrame(rows)
