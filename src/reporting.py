"""Estonian figures and data-derived scientific interpretations."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import read_json, write_json

TRANSLATIONS = {
    "model": "Mudel",
    "method": "Meetod",
    "case": "Juhtumiuuring",
    "group": "Tunnuserühma kood",
    "feature": "Tunnuse kood",
    "n": "Vaatlusi",
    "prediction": "Prognoos",
    "target_next_hour": "Tegelik koormus",
    "bias": "Keskmine suunatud viga",
    "quarter_MAE_std": "Kvartalite MAE standardhälve",
    "n_features": "Tunnuste arv",
    "tree_nodes": "Puusõlmede arv",
    "target_time": "Prognoosi sihtaeg",
    "scenario": "Stsenaariumi kood",
    "label": "Stsenaariumi kirjeldus",
    "evidence": "Tõendite piisavus",
    "MAE_increase": "MAE suurenemine",
    "MAE_minus_ordinary": "MAE erinevus tavatingimustest",
    "large_error_share": "Suurte vigade osakaal",
    "abs_error_p50": "Absoluutvea mediaan",
    "abs_error_p90": "Absoluutvea P90",
    "abs_error_p95": "Absoluutvea P95",
    "rank_spearman": "Järjestuste Spearmani seos",
    "all_pass": "Kõik automaatkontrollid läbitud",
    "ci_low": "Usaldusvahemiku alampiir",
    "ci_high": "Usaldusvahemiku ülempiir",
    "period": "Periood",
    "bias_ci_low": "Suunatud vea UV alampiir",
    "bias_ci_high": "Suunatud vea UV ülempiir",
    "delta_MAE_vs_baseline": "MAE muutus baasist",
    "hypothesis_supported_selection": "Valiku MAE alla baasi",
    "hypothesis_supported_test": "Lõpphindamise MAE alla baasi",
    "MAE_difference_A_minus_B": "MAE erinevus A miinus B",
    "block_hours": "Ploki pikkus tundides",
    "repeats": "Korduste arv",
    "interpretation": "Tõlgendus",
    "A": "Meetod A",
    "B": "Meetod B",
    "target_scenario_n": "Juhtumiuuringu stsenaariumivaatlusi",
    "target_scenario_MAE": "Juhtumiuuringu stsenaariumide MAE",
    "target_scenario_delta_MAE": "Stsenaariumide MAE muutus baasist",
}
VALUES = {
    "Persistence": "Viimase koormuse prognoos",
    "SeasonalNaive": "Eelmise päeva prognoos",
    "GBR": "Gradientvõimendus (GBR)",
    "Manual": "Eelmääratud valik",
    "Residual": "Veapõhine valik",
    "XAI": "XAI-põhine valik",
    "temperature": "Temperatuur",
    "rapid_change": "Kiired koormusmuutused",
    "overall": "Kõik tunnid",
    "scenario": "Probleemsed tunnid",
}


def markdown_table(df):
    """No optional tabulate dependency; all displayed headers are Estonian."""
    data = df.copy()
    for column in ("model", "method", "case", "A", "B", "Meetod A", "Meetod B"):
        if column in data:
            data[column] = data[column].map(lambda x: VALUES.get(x, x))
    data = data.rename(columns=TRANSLATIONS)
    for c in data:
        data[c] = data[c].map(
            lambda v: (
                ("Jah" if v else "Ei")
                if isinstance(v, bool)
                else (f"{v:.4f}".replace(".", ",") if isinstance(v, float) else str(v))
            )
        )
    return (
        "| "
        + " | ".join(map(str, data.columns))
        + " |\n| "
        + " | ".join(["---"] * len(data.columns))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in data.astype(str).values)
    )


def save_figure(fig, path, tight=True):
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(fig)


def show_table(run, path, columns=None, rename=None, limit=None, formatter=markdown_table):
    from IPython.display import Markdown, display

    frame = pd.read_csv(run / path)
    if columns is not None:
        frame = frame[columns]
    frame = frame.rename(columns=rename or {})
    display(Markdown(formatter(frame.head(limit) if limit is not None else frame)))


def show_figure(run, path, interpretation, width=1000):
    from IPython.display import Image, Markdown, display

    display(Image(filename=str(run / path), width=width))
    display(Markdown("**Tõlgendus.** " + interpretation))


def show_run(run, expanded=False):
    from IPython.display import Markdown, display

    manifest = read_json(run / "manifest.json")
    label = "Uuriv laiendus" if expanded else "Kasutatud käivitus"
    text = f"{label}: `{run.name}`. Algfaili SHA-256: `{manifest['raw_sha256']}`."
    if expanded:
        text += " Uut pimedat sõltumatut testi ei ole; 2020 on tagasivaateline."
    display(Markdown(text))


def make_figures(out, baseline, scenarios, importance, stability):
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    notes = {}
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for model, group in baseline.groupby("model"):
        ax.plot(group.period, group.MAE, marker="o", label=VALUES.get(model, model))
    ax.set(title="Võrdlusmudelite prognoosivead", xlabel="Periood", ylabel="MAE (CSV koormusühik)")
    ax.legend()
    save_figure(fig, folder / "baseline.png", tight=False)
    diagnostic = baseline[baseline.period == "Diagnoosimine 2018"]
    best = diagnostic.loc[diagnostic.MAE.idxmin()]
    notes["baseline"] = (
        f"Joonis ja tabel võrdlevad samu sihtaegu. 2018. aasta madalaim MAE oli mudelil {best.model}: {best.MAE:.3f}. GBR on uurimisobjekt sõltumata võrdluse võitjast; see fikseeriti enne diagnoosimist. 2019. aasta tulemust kasutatakse tunnusekatsete valimiseks, mitte baasparameetrite ümbervalimiseks."
    )
    diagnosis_predictions = pd.read_csv(
        out / "baseline/diagnosis_GBR.csv", parse_dates=["target_time"]
    )
    first_week = diagnosis_predictions.iloc[:168]
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.plot(first_week.target_time, first_week.target_next_hour, label="Tegelik koormus")
    ax.plot(first_week.target_time, first_week.prediction, label="GBR prognoos", alpha=0.8)
    ax.set(
        title="GBR prognoosi esimene diagnoosimisnädal",
        xlabel="Panama kohalik aeg",
        ylabel="Elektrikoormus (CSV ühik)",
    )
    ax.legend()
    save_figure(fig, folder / "forecast.png", tight=False)
    notes["forecast"] = (
        f"Joonisel on ette määratud esimese 168 diagnoositunni tegelik koormus ja GBR prognoos. Selle akna MAE on {first_week.abs_error.mean():.3f} ühikut. Üks nädal ei esinda kogu aasta täpsust, kuid võimaldab näha ajalisi kõrvalekaldeid; mudeli hindamise põhitõend jääb sama perioodi täielikule tabelile."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hist(diagnosis_predictions.error, bins=70)
    axes[0].set(
        title="Suunatud prognoosivea jaotus",
        xlabel="Prognoos miinus tegelik (CSV ühik)",
        ylabel="Vaatluste arv",
    )
    hourly_bias = diagnosis_predictions.groupby(
        diagnosis_predictions.target_time.dt.hour
    ).error.mean()
    axes[1].plot(hourly_bias.index, hourly_bias.values, marker="o")
    axes[1].axhline(0, color="grey", ls="--")
    axes[1].set(
        title="Keskmine suunatud viga sihttunniti",
        xlabel="Sihttunni kellaaeg",
        ylabel="Keskmine viga (CSV ühik)",
    )
    save_figure(fig, folder / "residuals.png", tight=False)
    notes["residuals"] = (
        f"Joonis näitab kõiki diagnoosimisperioodi jääke ning nende keskmisi sihttunniti. Üldine bias on {diagnosis_predictions.error.mean():.3f}, suurima absoluutse tunnikeskmisega kellaaeg {int(hourly_bias.abs().idxmax())}. See võib toetada ajakonteksti hüpoteesi, kuid segavad tegurid ja ajaline sõltuvus takistavad põhjuslikku järeldust. Tunnuseid valitakse hiljem eraldi valideerimisaastal."
    )
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.barh(scenarios.label, scenarios.MAE)
    ax.set(title="GBR-i vead diagnoosimisperioodi stsenaariumides", xlabel="MAE (CSV koormusühik)")
    save_figure(fig, folder / "scenarios.png", tight=False)
    worst = scenarios.loc[scenarios.MAE.idxmax()]
    notes["scenarios"] = (
        f"Joonis ja tabel näitavad kattuvaid stsenaariume. Suurim kirjeldav MAE oli stsenaariumis „{worst.label}”: {worst.MAE:.3f}, n={int(worst.n)}. Stsenaariumid ei ole sõltumatud rühmad. Suunatud vea plokk-usaldusvahemikud aitavad hinnata süsteemset kõrvalekallet; väikese valimi puhul järeldusest loobutakse. Tagasivaateline silt ei ole tulevase sündmuse tuvastaja."
    )
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    imp = importance.sort_values().tail(13)
    ax.barh(imp.index, imp.values)
    ax.set(
        title="Keskmine absoluutne SHAP-panus diagnoosimisaastal",
        xlabel="Mudeli väljundi panus (CSV koormusühik)",
        ylabel="Tunnuse kood",
    )
    save_figure(fig, folder / "shap.png", tight=False)
    notes["xai"] = (
        f"Joonis kirjeldab prognoosi panuseid fikseeritud diagnoosimisvalimil. Suurim keskmine absoluutne SHAP-panus on tunnusel {importance.idxmax()}: {importance.max():.3f}. PFI mõõdab eraldi permuteerimisega kaasnevat täpsuse muutust. Kumbki ei tõenda prognoosivea põhjust. Stabiilsuse mediaanne järjestuskorrelatsioon on {stability.rank_spearman.median():.3f}; alamperioodide muutus võib kajastada ka konteksti muutumist."
    )
    pfi = pd.read_csv(out / "explanations/permutation_importance.csv")
    pfi_summary = pfi.groupby("group").MAE_increase.agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.barh(pfi_summary.index, pfi_summary["mean"], xerr=pfi_summary["std"])
    ax.set(
        title="Üksiktunnuste ja tunnuserühmade permutatsioonolulisus",
        xlabel="MAE suurenemine (CSV koormusühik)",
        ylabel="Tunnuse või rühma kood",
    )
    save_figure(fig, folder / "pfi.png", tight=False)
    pfi_top = pfi_summary["mean"].idxmax()
    notes["pfi"] = (
        f"Joonisel on permuteerimise keskmine MAE suurenemine ja viie korduse standardhälve. Kõrgeim väärtus on rühmal/tunnusel {pfi_top}: {pfi_summary.loc[pfi_top, 'mean']:.3f}. See kirjeldab fikseeritud mudeli tundlikkust; ebarealistlikud tunnusekombinatsioonid ja korrelatsioon võivad olulisust moonutada. Vearibad ei ole usaldusvahemikud ega põhjusliku mõju hinnangud. Rühmade võrdlus toetab üksiktunnuste ettevaatlikku tõlgendust."
    )
    notes["refinements"] = (
        "Tabelid sisaldavad kõiki katseid, ka täpsust halvendanud muudatusi. Iga meetod kasutab kaht katset kummaski juhtumiuuringus, sama GBR-i ja sama kandidaatide hulka. Võrdne loogiline eelarve säilib ka ühiste mudeliarvutuste vahemällu salvestamisel. Valitud kandidaat võib jääda baasist halvemaks; parandust ei eeldata."
    )
    write_json(out / "reports/interpretations.json", notes)
    return notes


def plot_repair_ledger(predicted, scores, path, title):
    """Pair forecast edits with exact realised error-reduction responsibility."""
    import numpy as np
    from .labels import REPAIR_GROUP_NAMES, REPAIR_METHOD_NAMES

    actual = predicted.target_next_hour.to_numpy()
    baseline = predicted.prediction_GBR.to_numpy()
    repaired = predicted.prediction_credit_feedback.to_numpy()
    index = int(np.argmax(np.abs(baseline - actual)))
    row = predicted.iloc[index]
    groups = [c.removeprefix("correction_") for c in predicted if c.startswith("correction_")]
    order = sorted(groups, key=lambda g: abs(row[f"gain_{g}"]))
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    keep = ["GBR", "Ridge", "HistGBR", "HistResidual", "standalone_feedback", "credit_feedback"]
    overall = (
        scores[(scores.stage == "final") & (scores.scope == "overall")]
        .set_index("model")
        .reindex(keep)
    )
    axes[0, 0].barh(
        [REPAIR_METHOD_NAMES[n] for n in keep], overall.MAE, color=["#7a8896"] * 5 + ["#167d8d"]
    )
    axes[0, 0].invert_yaxis()
    axes[0, 0].set(title="Kõigi hindamistundide viga", xlabel="MAE")
    start, end = max(0, index - 24), min(len(predicted), index + 25)
    window = predicted.iloc[start:end]
    for name, label, color in (
        ("target_next_hour", "Tegelik", "#222222"),
        ("prediction_GBR", "Algprognoos", "#9b7685"),
        ("prediction_credit_feedback", "XAI-tagasiside", "#167d8d"),
    ):
        axes[0, 1].plot(pd.to_datetime(window.target_time), window[name], label=label, color=color)
    axes[0, 1].axvline(pd.Timestamp(row.target_time), color="#999999", linestyle=":")
    axes[0, 1].set(title="Algmudeli suurima veaga tund ja selle ümbrus", ylabel="Koormus")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].legend()
    for axis, prefix, caption in (
        (axes[1, 0], "correction_", "Millest koosnes prognoosi muudatus?"),
        (axes[1, 1], "gain_", "Kuidas jaotus tegelik vea vähenemine?"),
    ):
        values = np.array([row[prefix + g] for g in order])
        axis.barh(
            [REPAIR_GROUP_NAMES[g] for g in order],
            values,
            color=np.where(values >= 0, "#167d8d", "#c26553"),
        )
        axis.axvline(0, color="#333333", linewidth=0.7)
        axis.set(
            title=caption,
            xlabel=(
                "Prognoosi muutus"
                if prefix == "correction_"
                else "Absoluutvea vähenemise Shapley-jaotus"
            ),
        )
    gain = abs(row.prediction_GBR - row.target_next_hour) - abs(
        row.prediction_credit_feedback - row.target_next_hour
    )
    figure.suptitle(
        f"{title}\nJuhtum {row.target_time}; absoluutvea vähenemine {gain:.2f}", fontsize=14
    )
    figure.supxlabel(
        "Tagasivaateline parandusmäng. Panuste summa taastab mõõdetud muutuse; see ei tõenda vea põhjust.",
        fontsize=10,
    )
    save_figure(figure, path, tight=False)
