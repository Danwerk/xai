"""Shared display names; no plotting or model dependencies."""

MODEL_NAMES = {
    "Ridge": "Ridge-regressioon",
    "RandomForest": "Juhumets",
    "ExtraTrees": "Juhuslike lõigetega mets",
    "GBR": "Gradientvõimendus (GBR)",
    "HistGBR": "Histogrammi-gradientvõimendus",
    "Persistence": "Viimase koormuse prognoos",
    "SeasonalNaive": "Eelmise päeva prognoos",
}

REPAIR_GROUP_NAMES = {
    "recent_load": "Lühike koormusajalugu",
    "seasonal_load": "Päeva ja nädala koormus",
    "clock": "Kellaaeg",
    "calendar": "Nädal ja kalender",
    "weather": "Varasem ilm",
    "dynamics": "Koormuse dünaamika",
    "error_memory": "Varem selgunud prognoosivead",
    "forecast_anchor": "Algprognoosi väärtus",
    "offset": "Parandaja võrdlusväärtus",
}

REPAIR_METHOD_NAMES = {
    "GBR": "Algne GBR",
    "HistGBR": "Otsene HistGBR",
    "Ridge": "Otsene Ridge",
    "HistResidual": "Veaparandaja",
    "raw": "Algtunnuste perioodiline parandus",
    "raw_memory": "Algtunnused ja veamälu",
    "shap": "SHAP-panuste perioodiline parandus",
    "shap_memory": "SHAP-panused ja veamälu",
    "credit_feedback": "Paranduskasu XAI-tagasiside",
    "standalone_feedback": "Tagasiside ilma koosmõju jaotuseta",
    "scalar_feedback": "Ühe osakaaluga tagasiside",
}

EXPERIMENT_NAMES = {
    "ramp": "viimase ühe ja kolme tunni koormusmuutused",
    "rolling": "viimase 6 ja 24 tunni keskmine koormus ja standardhälve",
    "temperature_time": "temperatuuri ja kellaaja koosmõju",
}


def feature_label(name, simple=False):
    simple_labels = {
        "hour_sin": "Prognoositava tunni kellaaeg (tunnus 1)",
        "hour_cos": "Prognoositava tunni kellaaeg (tunnus 2)",
        "dow_sin": "Nädalapäev (tunnus 1)",
        "dow_cos": "Nädalapäev (tunnus 2)",
        "month_sin": "Kuu (tunnus 1)",
        "month_cos": "Kuu (tunnus 2)",
        "humidity": "Varem mõõdetud õhu niiskusesisaldus",
    }
    if simple and name in simple_labels:
        return simple_labels[name]
    labels = {
        "load_now": "Viimane teadaolev koormus",
        "load_day": "Prognoositava tunni koormus eelmisel päeval",
        "load_week": "Prognoositava tunni koormus eelmisel nädalal",
        "same_hour_weekly_mean": "Nelja nädala sama tunni keskmine",
        "same_hour_weekly_std": "Nelja nädala sama tunni standardhälve",
        "relative_week_load": "Koormuse suhe eelmise nädalaga",
        "working_hour": "Tööaja tunnus",
        "temperature": "Varem mõõdetud temperatuur",
        "humidity": "Varem mõõdetud eriniiskus",
        "wind": "Varem mõõdetud tuule kiirus",
        "ramp_1": "Koormuse teadaolev tunnimuutus",
        "holiday_known": "Teadaolev püha",
        "school_known": "Teadaolev kooliperiood",
        "hour_sin": "Prognoositava tunni kellaaeg (siinus)",
        "hour_cos": "Prognoositava tunni kellaaeg (koosinus)",
        "dow_sin": "Nädalapäev (siinus)",
        "dow_cos": "Nädalapäev (koosinus)",
        "month_sin": "Kuu (siinus)",
        "month_cos": "Kuu (koosinus)",
        "weekend": "Nädalavahetus",
        "load_acceleration": "Tunnimuutuse muutus",
        "relative_ramp": "Suhteline tunnimuutus",
        "deviation_from_day_mean": "Kõrvalekalle ööpäeva keskmisest",
        "standardized_load_deviation": "Standardiseeritud koormuskõrvalekalle",
        "deviation_same_hour_day": "Kõrvalekalle eelmise päeva samast tunnist",
        "deviation_same_hour_week": "Kõrvalekalle eelmise nädala samast tunnist",
        "day_vs_week_target": "Päeva ja nädala sihttunni erinevus",
    }
    if name in labels:
        return labels[name]
    if name.startswith(("recent_load_lag_", "seasonal_load_lag_")):
        return f'Koormus {name.rsplit("_",1)[1]} tundi varem'
    if name.startswith("history_"):
        _, stat, hours = name.split("_")
        return (
            f"{hours} tunni "
            + {"mean": "keskmine", "std": "standardhälve", "min": "miinimum", "max": "maksimum"}[
                stat
            ]
        )
    if name.startswith("weekday_"):
        return ["Esmaspäev", "Teisipäev", "Kolmapäev", "Neljapäev", "Reede", "Laupäev", "Pühapäev"][
            int(name.rsplit("_", 1)[1])
        ]
    if name.startswith("ramp_"):
        return f'Koormuse {name.rsplit("_",1)[1]} tunni muutus'
    return "Tunnuse kood: " + name
