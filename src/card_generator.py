"""Kaardijuhtumite valik, arvandmed ja soovitusreeglid."""

import numpy as np

from .labels import EXPERIMENT_NAMES
from .scenario_detector import LABELS


def make_repair_audit(row):
    """Two separate ledgers: a change to the forecast and its realised gain."""
    baseline = float(row["prediction_GBR"])
    actual = float(row["target_next_hour"])
    prediction = float(row["prediction_credit_feedback"])
    groups = [c.removeprefix("correction_") for c in row.index if c.startswith("correction_")]
    items = [
        {"group": g, "correction": float(row[f"correction_{g}"]), "gain": float(row[f"gain_{g}"])}
        for g in groups
    ]
    gain = abs(baseline - actual) - abs(prediction - actual)
    if not items or not np.isclose(
        sum(v["correction"] for v in items), prediction - baseline, atol=1e-6
    ):
        raise ValueError("Repair card edits do not reconstruct the forecast.")
    if not np.isclose(sum(v["gain"] for v in items), gain, atol=1e-6):
        raise ValueError("Repair card credits do not reconstruct the absolute-error reduction.")
    return {
        "target_time": str(row["target_time"]),
        "baseline": baseline,
        "prediction": prediction,
        "actual": actual,
        "absolute_error_before": abs(baseline - actual),
        "absolute_error_after": abs(prediction - actual),
        "gain": gain,
        "items": items,
        "limitations": "Tagasivaateline jaotus arvestab paranduste koosmõju. See ei tõenda algse vea põhjust ega sama paranduse tulevast kasu.",
    }


def select_card_cases(predictions, scenarios, count):
    """Vali eri olukordade näited ja säilita iga valiku põhjus."""
    selected = {
        0: "Valimi esimene juhtum. Kasutame seda varasemate juhtumite leidmise kontrolliks."
    }
    selected.setdefault(
        int(predictions.abs_error.idxmin()),
        "Valimi väikseima veaga juhtum võrdluseks suurte vigadega.",
    )
    for scenario in ("peak", "rapid_increase", "rapid_decrease", "heat", "cold", "holiday"):
        eligible = predictions.loc[scenarios[scenario]]
        if len(eligible):
            selected.setdefault(
                int(eligible.abs_error.idxmax()),
                f"Valimi suurima veaga näide olukorrast „{LABELS[scenario].removesuffix(' (tagasivaateline)')}”.",
            )
    for index in predictions.abs_error.sort_values(ascending=False).index:
        selected.setdefault(int(index), "Üks valimi suurima veaga juhtumeid.")
    return dict(list(selected.items())[:count])


def diagnostic_hypothesis(scenarios, cohort, top_shap, historical_n, min_n=40):
    rules = (
        ("rapid_increase", {"load_now", "load_day", "load_week"}, "ramp"),
        ("rapid_decrease", {"load_now", "load_day", "load_week"}, "rolling"),
        ("heat", {"temperature", "humidity"}, "temperature_time"),
    )
    for scenario, features, group in rules:
        if scenario not in scenarios:
            continue
        evidence = cohort[cohort.scenario == scenario]
        if evidence.empty:
            continue
        row = evidence.iloc[0]
        supporting = [item["feature"] for item in top_shap[:3] if item["feature"] in features]
        if (
            row.n >= min_n
            and np.isfinite(row.get("MAE_minus_ordinary", np.nan))
            and row.MAE_minus_ordinary > 0
            and historical_n >= 3
            and supporting
        ):
            source = "Varasema koormuse tunnus" if group != "temperature_time" else "Ilmatunnus"
            return {
                "status": "Katsetatav muudatus",
                "hypothesis": f"{source} on SHAP-i järgi selle prognoosi kolme olulisema tunnuse hulgas. Võimalik lisainfo mudelile: {EXPERIMENT_NAMES[group]}.",
                "experiment": group,
                "scenario": scenario,
                "cohort_n": int(row.n),
                "MAE_minus_ordinary": float(row.MAE_minus_ordinary),
                "historical_n": historical_n,
                "supporting_shap_features": supporting,
                "limitations": "SHAP ei tõenda vea põhjust. Muudatuse kasu tuleb eraldi katsetada.",
            }
    return {
        "status": "Muudatust ei soovitata",
        "hypothesis": None,
        "experiment": None,
        "limitations": "Andmeid ei ole piisavalt või soovitusreegli tingimused ei täitu.",
    }


def make_card(row, prediction, shap_row, base, scenarios, cohort, historical, cfg):
    contributions = [
        {
            "feature": name,
            "value": float(row[name]),
            "shap": float(shap_row[name]),
            "interpretation": (
                "Suurendab prognoosi võrdlusväärtuse suhtes"
                if shap_row[name] > 0
                else (
                    "Vähendab prognoosi võrdlusväärtuse suhtes"
                    if shap_row[name] < 0
                    else "Panus prognoosi on null"
                )
            ),
        }
        for name in shap_row.abs().sort_values(ascending=False).index
    ]
    active = [c for c in scenarios.index if bool(scenarios[c]) and c != "ordinary"]
    diagnosis = diagnostic_hypothesis(
        active, cohort, contributions, len(historical), cfg["min_scenario_n"]
    )
    return {
        "origin_time": str(row.origin_time),
        "target_time": str(row.target_time),
        "prediction": float(prediction.prediction),
        "actual": float(prediction.target_next_hour),
        "error": float(prediction.error),
        "absolute_error": float(prediction.abs_error),
        "scenarios": [LABELS[s] for s in active] or [LABELS["ordinary"]],
        "observed_patterns": {
            "known_ramp": float(row.load_ramp_1h),
            "realized_ramp": float(row.actual_ramp),
            "known_temperature": float(row.temperature),
        },
        "base_value": float(base),
        "shap_contributions": contributions,
        "historical_cases": (
            historical.astype({"origin_time": str, "target_time": str}).to_dict("records")
            if len(historical)
            else []
        ),
        "diagnosis": diagnosis,
        "evidence_sources": [
            f"{cfg.get('artifact_dir', 'results')}/{p}"
            for p in (
                "baseline/diagnosis_GBR.csv",
                "explanations/shap_values.csv",
                "scenarios/diagnosis.csv",
                "explanations/historical_cases.csv",
            )
        ],
        "limitations": [
            "SHAP kirjeldab prognoosi, mitte veapõhjust.",
            "Omavahel seotud tunnused võivad mõjutada SHAP-panuste jaotust.",
            "Sarnastel juhtumitel ei pruugi olla sama vea põhjus.",
            "Koormuse ühik pole kinnitatud. Kasutame algandmete ühikut.",
        ],
    }
