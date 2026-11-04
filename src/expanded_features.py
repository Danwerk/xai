"""Domain-motivated additions, kept separate from the original RQ2 experiment."""

import numpy as np
import pandas as pd

from .feature_engineering import build_features


def build_expanded_features(raw, base_cfg):
    frame = build_features(raw, base_cfg).set_index("origin_time")
    source = raw.set_index("datetime")
    load = source.nat_demand.shift(base_cfg["load_delay_hours"])
    weather_delay = base_cfg["weather_delay_hours"]
    extra = pd.DataFrame(index=source.index)
    groups = {}

    def add(group, name, value):
        extra[name] = value
        groups.setdefault(group, []).append(name)

    for lag in (1, 2, 3, 6, 12):
        add("recent_history", f"recent_load_lag_{lag}", load.shift(lag))

    for lag in (47, 71, 335):
        add("seasonal_history", f"seasonal_load_lag_{lag}", load.shift(lag))
    weekly = pd.concat([load.shift(168 * k - 1) for k in (1, 2, 3, 4)], axis=1)
    add(
        "seasonal_history",
        "same_hour_weekly_mean",
        weekly.mean(axis=1).where(weekly.notna().all(axis=1)),
    )
    add(
        "seasonal_history",
        "same_hour_weekly_std",
        weekly.std(axis=1).where(weekly.notna().all(axis=1)),
    )

    for window in (6, 24, 168):
        rolling = load.rolling(window, min_periods=window)
        for statistic in ("mean", "std", "min", "max"):
            add(
                "multiscale_history", f"history_{statistic}_{window}", getattr(rolling, statistic)()
            )

    for lag in (1, 3, 6, 24):
        add("dynamics", f"ramp_{lag}", load.diff(lag))
    add("dynamics", "load_acceleration", load.diff().diff())
    add("dynamics", "relative_ramp", load.diff() / load.shift(1).where(load.shift(1).abs() > 1))
    mean24 = load.rolling(24, min_periods=24).mean()
    std24 = load.rolling(24, min_periods=24).std()
    add("dynamics", "deviation_from_day_mean", load - mean24)
    add("dynamics", "standardized_load_deviation", (load - mean24) / std24.clip(lower=1))

    weather = source[[f"T2M_{c}" for c in ("toc", "san", "dav")]].shift(weather_delay)
    humidity = source[[f"QV2M_{c}" for c in ("toc", "san", "dav")]].shift(weather_delay)
    for city in ("toc", "san", "dav"):
        add("regional_weather", f"regional_temperature_{city}", weather[f"T2M_{city}"])
    add("regional_weather", "regional_temperature_range", weather.max(axis=1) - weather.min(axis=1))
    add("regional_weather", "regional_humidity_range", humidity.max(axis=1) - humidity.min(axis=1))

    temp = weather.mean(axis=1)
    target_time = source.index + pd.Timedelta(hours=1)
    for threshold in (24, 26, 28):
        add("weather_interactions", f"cooling_proxy_{threshold}", (temp - threshold).clip(lower=0))
    add("weather_interactions", "temperature_humidity_product", temp * humidity.mean(axis=1))
    add("weather_interactions", "temperature_daily_change", temp.diff(24))
    add(
        "weather_interactions",
        "temperature_hour_sin_expanded",
        temp * np.sin(2 * np.pi * target_time.hour / 24),
    )
    add(
        "weather_interactions",
        "temperature_hour_cos_expanded",
        temp * np.cos(2 * np.pi * target_time.hour / 24),
    )

    for day in range(7):
        add("calendar_detail", f"weekday_{day}", (target_time.dayofweek == day).astype(int))
    add(
        "calendar_detail",
        "working_hour",
        ((target_time.hour >= 8) & (target_time.hour <= 18) & (target_time.dayofweek < 5)).astype(
            int
        ),
    )
    # Known calendar flags already have the conservative date-boundary policy.
    groups["calendar_detail"].extend(["holiday_known", "school_known"])

    add("seasonal_deviation", "deviation_same_hour_day", load - load.shift(24))
    add("seasonal_deviation", "deviation_same_hour_week", load - load.shift(168))
    add("seasonal_deviation", "day_vs_week_target", load.shift(23) - load.shift(167))
    add(
        "seasonal_deviation",
        "relative_week_load",
        load / load.shift(168).where(load.shift(168).abs() > 1),
    )

    frame = frame.join(extra)
    base = list(base_cfg["baseline_features"])

    def combine_features(group_names):
        columns = base + [column for name in group_names for column in groups[name]]
        return list(dict.fromkeys(columns))

    bundles = {"baseline": base}
    for name in groups:
        bundles["add_" + name] = combine_features([name])
    history = [
        "recent_history",
        "seasonal_history",
        "multiscale_history",
        "dynamics",
        "seasonal_deviation",
    ]
    bundles["history_dynamics"] = combine_features(history)
    bundles["weather_calendar"] = combine_features(
        ["regional_weather", "weather_interactions", "calendar_detail"]
    )
    bundles["all_features"] = combine_features(list(groups))
    weather_columns = set(
        ["temperature", "humidity", "wind"]
        + groups["regional_weather"]
        + groups["weather_interactions"]
    )
    bundles["without_weather"] = [c for c in bundles["all_features"] if c not in weather_columns]
    bundles["history_calendar"] = [
        c for c in combine_features(history + ["calendar_detail"]) if c not in weather_columns
    ]
    needed = list(dict.fromkeys(c for columns in bundles.values() for c in columns))
    frame = frame.dropna(subset=["target_next_hour", *needed]).reset_index()
    if not np.isfinite(frame[needed].to_numpy()).all():
        raise ValueError(
            "Laiendatud tunnustes on lõpmatud väärtused; automaatset asendust ei tehta."
        )
    return frame, groups, bundles


def feature_catalog(groups):
    reasons = {
        "recent_history": "Viimaste tundide koormus kirjeldab lühiajalist mälu; võrreldakse baasi viimase väärtusega.",
        "seasonal_history": "Sama sihttunni pikem ajalugu eristab korduvat hooajalisust juhuslikust muutusest.",
        "multiscale_history": "Mitme ajaskaala tase, hajuvus ja ulatus kirjeldavad koormusrežiimi.",
        "dynamics": "Kiirus, kiirendus ja suhteline kõrvalekalle võivad kirjeldada üleminekuid; abi ei eeldata.",
        "regional_weather": "Linnade erinevused võivad kaduda keskmistamisel; tugevat korrelatsiooni arvestatakse tõlgenduses.",
        "weather_interactions": "Jahutusvajaduse proksid ja aja-ilma interaktsioonid võimaldavad tingimuslikku tundlikkust.",
        "calendar_detail": "Nädalapäeva detail ja tööaja proksi võivad eristada tarbimisrežiime; püha/kool on ainult teadaolev märgend.",
        "seasonal_deviation": "Sama tunni tavalisest päevast/nädalast kõrvalekaldumine võib eristada uut tarbimismustrit.",
    }
    return pd.DataFrame(
        [
            {
                "group": g,
                "feature": c,
                "rationale": reasons[g],
                "availability": "Ainult t või varasem koormus, t−24 või varasem ilm; sihttunni kellaaeg on teada.",
            }
            for g, columns in groups.items()
            for c in columns
        ]
    )
