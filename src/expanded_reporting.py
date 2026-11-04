"""Question-led Estonian figures with measured interpretations for each figure."""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import read_json, write_json
from .labels import MODEL_NAMES, feature_label
from .reporting import markdown_table, save_figure

GROUP_NAMES = {
    "recent_history": "Hiljutised koormused",
    "seasonal_history": "Pikem hooajaline ajalugu",
    "multiscale_history": "Eri pikkusega ajavahemike koormus",
    "dynamics": "Koormuse muutused",
    "regional_weather": "Piirkondlik ilm",
    "weather_interactions": "Ilma ja aja koosmõju",
    "calendar_detail": "Täpsem kalender",
    "seasonal_deviation": "Hooajalisest profiilist kõrvalekalded",
}
FEATURE_NAMES = {
    "baseline": "Algsed tunnused",
    "history_dynamics": "Koormuse ajalugu ja muutused",
    "weather_calendar": "Ilm ja kalender",
    "all_features": "Kõik tunnused",
    "without_weather": "Tunnused ilma ilmaandmeteta",
    "history_calendar": "Koormusajalugu ja kalender",
}
FEATURE_NAMES.update({"add_" + k: "Baas + " + v.lower() for k, v in GROUP_NAMES.items()})
FEATURE_NAMES.update({"remove_" + k: "Eemaldatud: " + v.lower() for k, v in GROUP_NAMES.items()})
HEADERS = {
    "model": "Mudel",
    "feature_set": "Tunnusevariant",
    "parameter_id": "Seadistuse tunnus",
    "year": "Aasta",
    "period": "Periood",
    "n": "Vaatlusi",
    "bias": "Keskmine suunatud viga",
    "fit_seconds": "Sobitamise ja prognoosimise aeg (s)",
    "fit_predict_seconds": "Sobitamise ja prognoosimise aeg (s)",
    "n_features": "Tunnuste arv",
    "fold_MAE_std": "Aastate MAE standardhälve",
    "tree_nodes": "Puusõlmede arv",
    "A": "Mudel A",
    "B": "Mudel B",
    "difference": "MAE erinevus A miinus B",
    "ci_low": "95% UV alampiir",
    "ci_high": "95% UV ülempiir",
    "adjusted_low": "Korrigeeritud UV alampiir",
    "adjusted_high": "Korrigeeritud UV ülempiir",
    "block_hours": "Ploki pikkus tundides",
    "label": "Stsenaarium",
}


def display_table(frame):
    frame = frame.copy()
    for col in ("model", "A", "B"):
        if col in frame:
            frame[col] = frame[col].map(lambda s: MODEL_NAMES.get(s, s))
    if "feature_set" in frame:
        frame["feature_set"] = frame.feature_set.map(lambda s: FEATURE_NAMES.get(s, s))
    if "period" in frame:
        frame["period"] = frame.period.replace(
            {"confirmation": "Ajaline kontroll 2019", "retrospective": "Retrospektiivne 2020"}
        )
    return markdown_table(frame.rename(columns=HEADERS))


def plot_temporal_design(save):
    """Draw the shared training and evaluation timeline."""
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for i, year in enumerate([2017, 2018, 2019, 2020]):
        ax.broken_barh([(2015, year - 2015)], (i - 0.3, 0.55), facecolors="#90b7c9")
        ax.broken_barh(
            [(year, 0.9 if year < 2020 else 0.49)],
            (i - 0.3, 0.55),
            facecolors="#e6a348" if year < 2019 else "#4c956c",
        )
    ax.set_yticks(range(4), ["2017 valik", "2018 valik", "2019 kontroll", "2020 tagasivaade"])
    ax.set_xticks(range(2015, 2021))
    ax.set(title="Ajaline katsekorraldus: treening enne hindamist", xlabel="Kalendriaasta")
    save(
        fig,
        "01_temporal_design.png",
        "Joonis eristab varasematel andmetel sobitamist ja hilisema aasta hindamist. Tunnuse- ja parameetrivalik kasutab 2017–2018; finalistid fikseeritakse enne 2019 mõõdikuid. 2020 on varem nähtud retrospektiivne võrdlus. Skeem ei näita mõõtkavas 24-tunnist piirilünka; täpsed ajatemplid ja vaatlusarvud on folds.csv failis. See piirab valikuandmete juhusliku tulevikulekke võimalust, kuid ei loo uut sõltumatut andmestikku.",
    )


def plot_feature_effects(feature_rank, frozen_features, save):
    """Keep addition and removal comparisons together."""
    additions = feature_rank[feature_rank.kind == "addition"].copy().set_index("feature_set")
    base = additions.loc["baseline", "MAE"]
    delta = (additions.MAE - base).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.barh(
        [FEATURE_NAMES.get(v, v) for v in delta.index],
        delta.values,
        color=np.where(delta.values < 0, "#4c956c", "#b75d69"),
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        title="Tunnusevariantide mõju samale HistGBR-ile",
        xlabel="MAE muutus baasist (CSV koormusühik)",
    )
    selected = frozen_features["selected_feature_set"]
    save(
        fig,
        "02_feature_additions.png",
        f'Joonis vastab küsimusele, milline ette määratud tunnusekomplekt aitab fikseeritud mudelit 2017–2018 valikuaastatel. Parim variant on „{FEATURE_NAMES[selected]}”: MAE {additions.loc[selected,"MAE"]:.3f}, baasil {base:.3f}, erinevus {additions.loc[selected,"MAE"]-base:.3f}. Negatiivne tulba väärtus tähendab paranemist samadel vaatlustel. See on HistGBR-iga tehtud sõel ja selle valikutulemus võib olla optimistlik; variant kontrollitakse teiste mudelite ning hilisema 2019 aasta peal.',
    )
    full = additions.loc["all_features", "MAE"]
    ablation = feature_rank[feature_rank.kind == "ablation"].copy().set_index("feature_set")
    effect = (ablation.MAE - full).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.barh(
        [FEATURE_NAMES[v] for v in effect.index],
        effect.values,
        color=np.where(effect.values > 0, "#517c9f", "#d6a451"),
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        title="Rühma eemaldamine kõigi tunnuste mudelist",
        xlabel="MAE muutus pärast eemaldamist (CSV koormusühik)",
    )
    largest = effect.idxmax()
    save(
        fig,
        "03_feature_ablation.png",
        f"Joonis hindab rühma tingimuslikku kasu ülejäänud tunnuste olemasolul. Kõigi tunnuste MAE on {full:.3f}; suurim eemaldusjärgne MAE muutus on variandil „{FEATURE_NAMES[largest]}”: {effect[largest]:+.3f}. Positiivne muutus tähendab, et rühma eemaldamine halvendas täpsust, negatiivne et lihtsam variant oli parem. Korrelatsioon ja tunnuste asendatavus võivad rolli hajutada; lisamiskatse ja eemalduskatse ei pea andma sama järeldust. Need tulemused ei tõenda põhjuslikku mõju tarbimisele.",
    )


def make_expanded_report(out, cfg, frame):
    def read(name):
        return pd.read_csv(out / name)

    feature_rank = read("selection/feature_ranking.csv")
    candidates = read("selection/model_folds.csv")
    ranking = read("selection/model_ranking.csv")
    finalists = read("selection/finalists.csv")
    performance = read("evaluation/performance.csv")
    scenarios = read("evaluation/scenarios.csv")
    intervals = read("evaluation/paired_intervals.csv")
    freeze = read_json(out / "selection/frozen_finalists.json")
    frozen_features = read_json(out / "selection/frozen_features.json")
    groups = read_json(out / "configs/feature_groups.json")
    window_info = read_json(out / "explanations/window.json")
    champion = freeze["champion"]
    compare = champion if champion != "GBR" else next(n for n in finalists.model if n != "GBR")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    captions = {}

    def save(fig, name, text):
        save_figure(fig, figures / name)
        captions[name] = text

    def predictions(period, name):
        return pd.read_csv(
            out / f"predictions/{period}_{name}.csv", parse_dates=["target_time", "origin_time"]
        )

    def phase_title(period):
        return "Ajaline kontroll 2019" if period == "confirmation" else "Retrospektiivne 2020"

    plot_temporal_design(save)
    plot_feature_effects(feature_rank, frozen_features, save)
    matrix = ranking.pivot_table(index="model", columns="feature_set", values="MAE", aggfunc="min")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(matrix.columns)), [FEATURE_NAMES[c] for c in matrix.columns])
    ax.set_yticks(range(len(matrix)), [MODEL_NAMES[n] for n in matrix.index])
    for (i, j), v in np.ndenumerate(matrix.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    ax.set_title("Mudeliperekond ja tunnusevaade: valikuaastate MAE")
    fig.colorbar(im, ax=ax, label="MAE (CSV koormusühik)")
    save(
        fig,
        "04_model_feature_heatmap.png",
        f"Joonis näitab iga mudeliperekonna parima kahe hulgast valitud seadistuse MAE-d kummalgi ühisel tunnusevaatel. Valikuperioodi madalaim skoor on mudelil {MODEL_NAMES[champion]}: {ranking.iloc[0].MAE:.3f}. Mõlemad tunnusevaated ja mõlemad seadistused olid kõigile perekondadele kättesaadavad. Ka parameetrite hulgast minimaalse valimine suurendab valikuoptimismi; kindlama tõendi annab järgnev 2019 kontroll.",
    )
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for _, row in finalists.iterrows():
        folds = candidates[
            (candidates.model == row.model)
            & (candidates.parameter_id == row.parameter_id)
            & (candidates.feature_set == row.feature_set)
        ].sort_values("year")
        later = performance[performance.model == row.model].sort_values("year")
        ax.plot(
            list(folds.year) + list(later.year),
            list(folds.MAE) + list(later.MAE),
            marker="o",
            label=MODEL_NAMES[row.model],
        )
    ax.axvline(2018.5, color="grey", ls="--")
    ax.set_xticks([2017, 2018, 2019, 2020])
    ax.set(
        title="Fikseeritud finalistide täpsus eri aastatel",
        xlabel="Hindamisaasta",
        ylabel="MAE (CSV koormusühik)",
    )
    ax.legend(fontsize=8)
    numerical = performance[performance.period == "confirmation"].sort_values("MAE").iloc[0]
    save(
        fig,
        "05_model_temporal_stability.png",
        f"Joonis näitab samade ette fikseeritud finalistide vigu eri treening- ja hindamisperioodidel. 2017/2018 punkte kasutati valikuks, 2019/2020 punkte hilisemaks kontrolliks. 2019 arvuliselt parim mudel on {MODEL_NAMES[numerical.model]}, MAE {numerical.MAE:.3f}. Aastate vahel muutuvad nii treeningmaht kui andmejaotus; ühte joont ei tõlgendata kontrollitud ajamõjuna. 2020 on osaline aasta ning selle skoor ei muutnud finalistide valikut.",
    )
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for _, row in finalists.iterrows():
        ax.scatter(row.fit_seconds, row.MAE, s=70)
        ax.annotate(
            MODEL_NAMES[row.model],
            (row.fit_seconds, row.MAE),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set(
        title="Täpsuse ja arvutuskulu kompromiss valikuaastatel",
        xlabel="Kahe voldi sobitamise ja prognoosimise aeg (s, logaritmiline)",
        ylabel="Kaalutud MAE (CSV koormusühik)",
    )
    save(
        fig,
        "06_accuracy_cost.png",
        f"Joonis seob finalistide MAE sama masina kahe hindamisvoldi sobitamise+prognoosimise ajaga. Kiireima finalisti aeg on {finalists.fit_seconds.min():.2f} s, aeglaseima {finalists.fit_seconds.max():.2f} s. Väiksem kulu võib olla praktiline eelis ka sarnase täpsuse korral. Mõõtmised ei ole kordustega riistvaraline jõudlustest; vahemälu ja samaaegne koormus võivad aegu mõjutada. Seetõttu ei kasutata aega statistilise täpsuse võitja määramiseks.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, period in zip(axes, ["confirmation", "retrospective"]):
        part = intervals[(intervals.period == period) & (intervals.block_hours == 168)].reset_index(
            drop=True
        )
        for i, row in part.iterrows():
            ax.plot([row.adjusted_low, row.adjusted_high], [i, i], lw=2, color="#799cba")
            ax.plot([row.ci_low, row.ci_high], [i, i], lw=5, color="#315b78")
            ax.scatter(row.difference, i, color="black", s=25, zorder=3)
        ax.axvline(0, color="grey", ls="--")
        ax.set_yticks(range(len(part)), [MODEL_NAMES[n] for n in part.B])
        ax.set(title=phase_title(period), xlabel="Valikuvõitja MAE miinus konkurendi MAE")
    save(
        fig,
        "07_paired_uncertainty.png",
        "Joonis võrdleb ette valitud võitja ja iga ML-konkurendi sama tunni absoluutvigu, kasutades 168 tunni plokke. Tumedam lõik on nominaalne 95% intervall, heledam nelja võrdluse Bonferroni tasemega intervall. Nullist vasakul olev erinevus soosib valikuvõitjat; nulli hõlmav korrigeeritud intervall ei kinnita kindlat eelist selle konkurendi ees. Ringplokid arvestavad kohalikku sõltuvust, kuid ei kõrvalda valikuoptimismi ega andmestiku varasema kasutamise mõju. 2019 on peamine kontroll; 2020 on retrospektiivne tundlikkuspilt.",
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), layout="constrained")
    scenario_order = [
        "ordinary",
        "peak",
        "rapid_increase",
        "rapid_decrease",
        "heat",
        "cold",
        "holiday",
        "combined",
    ]
    common_max = scenarios[scenarios.scenario.isin(scenario_order)].MAE.max()
    from .scenario_detector import LABELS

    for ax, period in zip(axes, ["confirmation", "retrospective"]):
        part = scenarios[(scenarios.period == period) & scenarios.scenario.isin(scenario_order)]
        matrix = part.pivot(index="model", columns="scenario", values="MAE").reindex(
            columns=scenario_order
        )
        im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=common_max)
        ax.set_xticks(
            range(len(scenario_order)),
            [
                textwrap.fill(LABELS[n].replace(" (tagasivaateline)", ""), 14)
                for n in scenario_order
            ],
            rotation=65,
            ha="right",
        )
        ax.set_yticks(range(len(matrix)), [MODEL_NAMES[n] for n in matrix.index], fontsize=8)
        for (i, j), v in np.ndenumerate(matrix.values):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
        ax.set_title(phase_title(period))
    fig.colorbar(im, ax=axes, label="Stsenaariumi MAE (CSV ühik)", shrink=0.65)
    save(
        fig,
        "08_scenario_comparison.png",
        "Joonis küsib, kas üldine mudeliparemus püsib probleemsetes olukordades. Sama värviskaala võimaldab vaadata 2019 ja 2020 veataset; arvud on MAE samadel stsenaariumivaatlustel. Lävendid pärinevad ainult varaseima voldi treeningandmetest. Stsenaariumid kattuvad ning väike n võib olla ebakindel; täpsed arvud on scenarios.csv tabelis. Üldise MAE võitjat ei nimetata automaatselt kõikide stsenaariumide võitjaks.",
    )
    confirm_inputs = frame[frame.target_time.dt.year == cfg["confirmation_year"]].reset_index(
        drop=True
    )
    cp = predictions("confirmation", champion)
    reference = predictions("confirmation", "GBR")
    other = predictions("confirmation", compare)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, name, pred in zip(axes, ["GBR", compare], [reference, other]):
        residual = pred.assign(hour=pred.target_time.dt.hour, day=pred.target_time.dt.dayofweek)
        mat = residual.pivot_table(index="day", columns="hour", values="error", aggfunc="mean")
        limit = max(abs(reference.error).quantile(0.9), abs(other.error).quantile(0.9))
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks(range(0, 24, 3), range(0, 24, 3))
        ax.set_yticks(range(7), ["E", "T", "K", "N", "R", "L", "P"])
        ax.set(title=MODEL_NAMES[name], xlabel="Sihttunni kellaaeg", ylabel="Nädalapäev")
    fig.colorbar(im, ax=axes, label="Keskmine prognoos miinus tegelik (CSV ühik)", shrink=0.7)
    save(
        fig,
        "09_hour_weekday_bias.png",
        f"Joonis näitab 2019 keskmist suunatud viga kellaaja ja nädalapäeva lõikes. Võrreldakse GBR-i ja mudelit {MODEL_NAMES[compare]}; positiivne väärtus tähendab üleprognoosi. GBR üldine bias on {reference.error.mean():.3f}, võrdlusmudelil {other.error.mean():.3f}. Värvimuster võib aidata püstitada kalendrikonteksti hüpoteesi, kuid rühmade keskmine võib peita suuri üksikvigu. See on diagnoos pärast finalistide fikseerimist, tunnuseid selle joonise järgi enam ei valita.",
    )
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for name in finalists.model:
        p = predictions("confirmation", name)
        errors = np.sort(p.abs_error)
        ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors), label=MODEL_NAMES[name])
    ax.set_xscale("symlog", linthresh=1)
    ax.set(
        title="Milline osa 2019 vigadest jääb etteantud piirist väiksemaks?",
        xlabel="Absoluutviga (CSV ühik, sümmeetriline logaritmiline skaala)",
        ylabel="Vaatluste kumulatiivne osakaal",
    )
    ax.legend(fontsize=8)
    save(
        fig,
        "10_absolute_error_ecdf.png",
        f"Joonis näitab kogu absoluutvigade jaotust, mitte ainult keskmist. Valikuvõitja {MODEL_NAMES[champion]} absoluutvea mediaan on {cp.abs_error.median():.3f} ja 95. protsentiil {cp.abs_error.quantile(.95):.3f}. Kõrgem kumulatiivne osakaal sama veapiiri juures tähendab rohkem sellest piirist väiksemaid vigu. Kõverate ristumine viitab kompromissile väikeste ja suurte vigade vahel; see ei anna iseenesest statistilise paremuse testi.",
    )
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    gain = (reference.abs_error - cp.abs_error).cumsum()
    ax.plot(cp.target_time, gain)
    ax.axhline(0, color="grey", ls="--")
    ax.set(
        title="Valikuvõitja kumuleeruv absoluutvea sääst võrreldes GBR-iga",
        xlabel="2019 sihtaeg (Panama kohalik aeg)",
        ylabel="Kogunenud absoluutvea vahe (CSV ühik)",
    )
    save(
        fig,
        "11_cumulative_loss_difference.png",
        f"Joonis akumuleerib tundhaaval GBR absoluutvea miinus valikuvõitja absoluutvea. Aasta lõpu summa on {gain.iloc[-1]:.2f}; ülespoole liikuv lõik soosib valikuvõitjat, allapoole liikuv GBR-i. Üksikud muutused aitavad leida ajavahemikke, mil eelis tekib või kaob. See sõltuv summa ei ole usaldusvahemik ega operatiivse rahasäästu hinnang. Kui valikuvõitja on GBR ise, on vahe definitsiooni järgi null.",
    )
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    first = cp.iloc[:168]
    ax.plot(
        first.target_time, first.target_next_hour, label="Tegelik koormus", color="black", lw=1.4
    )
    for name in dict.fromkeys(["GBR", champion, compare]):
        p = predictions("confirmation", name).iloc[:168]
        ax.plot(p.target_time, p.prediction, label=MODEL_NAMES[name], alpha=0.8)
    ax.set(
        title="Ette määratud esimese 2019 nädala koormus ja prognoosid",
        xlabel="Panama kohalik aeg",
        ylabel="Elektrikoormus (CSV ühik)",
    )
    ax.legend(fontsize=8)
    save(
        fig,
        "12_forecast_week.png",
        f"Joonis näitab ette määratud esimest 168 kontrolltundi; seda nädalat ei valitud mudeli edu järgi. Valikuvõitja nädala MAE on {first.abs_error.mean():.3f}, kogu 2019 MAE {cp.abs_error.mean():.3f}. Ajaliselt joondatud kõverad võivad paljastada nihkunud tippe ja üleminekute vigu. Üks nädal ei asenda terve aasta ja stsenaariumide hindamist.",
    )
    ids = window_info["context_ids"]
    win = confirm_inputs.loc[ids]
    center = pd.Timestamp(window_info["center_target_time"])
    explained = window_info["explained_models"]
    fig, axes = plt.subplots(
        4 + len(explained),
        1,
        figsize=(13, 3 + 2 * len(explained) + 5),
        sharex=True,
        layout="constrained",
    )
    axes[0].plot(win.target_time, win.target_next_hour, color="black", label="Tegelik")
    for name in explained:
        pred = predictions("confirmation", name).loc[ids]
        axes[0].plot(pred.target_time, pred.prediction, label=MODEL_NAMES[name])
        axes[1].plot(pred.target_time, pred.error, label=MODEL_NAMES[name])
    axes[0].set(
        title="Suurima GBR-vea juhtum: prognoos, teadaolev ajalugu ja panused",
        ylabel="Koormus (CSV ühik)",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Suunatud viga")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[2].plot(win.target_time, win.ramp_1, color="#946b2d")
    axes[2].set(ylabel="Teadaolev tunnimuutus")
    axes[3].plot(win.target_time, win.temperature, color="#9e4864")
    axes[3].set(ylabel="Viivitatud temperatuur (°C)")
    for ax, name in zip(axes[4:], explained):
        shap_frame = read(f"explanations/{name}_shap.csv").set_index("context_id")
        columns = [c for c in shap_frame if c not in ["target_time", "base_value"]]
        top = shap_frame.loc[ids, columns].abs().mean().nlargest(3).index
        for col in top:
            ax.plot(win.target_time, shap_frame.loc[ids, col], label=feature_label(col))
        ax.set(
            ylabel="SHAP-panus",
            title=MODEL_NAMES[name] + ": akna kolm suurimat keskmist absoluutpanust",
        )
        ax.legend(fontsize=7, ncol=3)
    for ax in axes:
        ax.axvline(center, color="grey", ls="--")
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Sihtaeg (Panama kohalik aeg); püstjoon märgib valitud veajuhtumit")
    center_error = reference.loc[window_info["center_context_id"], "error"]
    save(
        fig,
        "13_temporal_diagnostic_window.png",
        f"Joonis toob kokku 48 tundi enne ja kuni 24 tundi pärast suurimat 2019 GBR-viga hetkel {center}; GBR suunatud viga oli {center_error:.3f} ühikut. Aken on tagasivaateline juhtumianalüüs, järgnevad tunnid ei olnud juhtumi prognoosihetkel teada. Iga sisendtunnuse joon näitab selle rea prognoosihetkel lubatud väärtust; temperatuur on 24 tundi viivitatud. SHAP-paneelid selgitavad vastava mudeli prognoosi, mitte vea põhjust. Panuste ja vea koosesinemine toetab kontrollitavat hüpoteesi, mitte tõestatud mehhanismi. Väga suure vea järgi valitud aken ei esinda tüüpilist täpsust.",
    )
    shap_inputs = read("explanations/inputs.csv").set_index("context_id")
    global_ids = window_info["global_context_ids"]
    importance = []
    feature_group = {c: g for g, columns in groups.items() for c in columns}
    for name in explained:
        shap_frame = read(f"explanations/{name}_shap.csv").set_index("context_id")
        columns = [c for c in shap_frame if c not in ["target_time", "base_value"]]
        means = shap_frame.loc[global_ids, columns].abs().mean()
        by_group = means.groupby([feature_group.get(c, "baseline") for c in means.index]).sum()
        by_group = by_group / by_group.sum()
        importance.append(by_group.rename(name))
    importance = pd.concat(importance, axis=1).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    importance.rename(index=lambda s: GROUP_NAMES.get(s, "Baas")).rename(
        columns=MODEL_NAMES
    ).plot.barh(ax=ax)
    ax.set(
        title="Kahe puumudeli prognoosipanuste jaotus tunnuserühmade vahel",
        xlabel="Rühma osakaal keskmiste absoluutsete SHAP-panuste summast",
        ylabel="Tunnuserühm",
    )
    save(
        fig,
        "14_cross_model_shap.png",
        f'Joonis kasutab ainult juhuslikku 2019 ankurvalimit, millest illustratiivne veaaken on välja jäetud. Selgitatavad mudelid on {", ".join(MODEL_NAMES[n] for n in explained)}; iga mudeli rühmaosakaalud summeeruvad üheks. Rühmitamine aitab võrrelda erineva tunnuste arvuga mudeleid, kuid panuste absoluutväärtuste summeerimine ei ole unikaalne rühma-SHAP ega põhjuslik tähtsus. Korrelatsioon võib jagada sama infot mitme tunnuse vahel. Sarnane panusejaotus ei tõenda võrdset täpsust ja erinev jaotus ei tõenda viga.',
    )
    tree = window_info["tree_champion"]
    shap_frame = read(f"explanations/{tree}_shap.csv").set_index("context_id")
    dep_columns = [c for c in ["temperature", "ramp_1"] if c in shap_frame]
    dep_columns.extend(
        c for c in ["load_now", "load_day"] if c in shap_frame and c not in dep_columns
    )
    dep_columns = dep_columns[:2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, col in zip(axes, dep_columns):
        scatter = ax.scatter(
            shap_inputs.loc[global_ids, col],
            shap_frame.loc[global_ids, col],
            c=shap_inputs.loc[global_ids, "hour"],
            s=12,
            alpha=0.6,
            cmap="viridis",
        )
        ax.set(
            title=f"{MODEL_NAMES[tree]}: {feature_label(col)}",
            xlabel="Prognoosihetkel teadaolev tunnuse väärtus",
            ylabel="Tunnuse SHAP-panus (CSV ühik)",
        )
    fig.colorbar(scatter, ax=axes, label="Sihttunni kellaaeg", shrink=0.75)
    save(
        fig,
        "15_shap_dependence.png",
        f'Joonis kirjeldab mudeli {MODEL_NAMES[tree]} prognoosipanuse seost sisendtunnusega fikseeritud juhuslikul 2019 valimil. Näidatud tunnused on {", ".join(dep_columns)}; võimaluse korral kasutatakse temperatuuri ja rampi, nende puudumisel baas-koormustunnuseid. Värv tähistab sihttunni kellaaega. See võib nähtavaks teha mittelineaarsust ja ajakonteksti, kuid sama väärtuse erinevad panused võivad tuleneda teistest tunnustest ja korrelatsioonist. Joonis ei näita mõõdetud koormuse ega prognoosivea põhjuslikku vastust tunnuse muutmisele.',
    )
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    bins = pd.qcut(confirm_inputs.temperature, 10, duplicates="drop")
    for name in dict.fromkeys(["GBR", champion, compare]):
        p = predictions("confirmation", name)
        tab = (
            pd.DataFrame(
                {"temperature": confirm_inputs.temperature, "abs_error": p.abs_error, "bin": bins}
            )
            .groupby("bin", observed=True)
            .agg(temperature=("temperature", "mean"), MAE=("abs_error", "mean"))
        )
        ax.plot(tab.temperature, tab.MAE, marker="o", label=MODEL_NAMES[name])
    ax.set(
        title="Prognoosivead teadaoleva viivitatud temperatuuri lõikes",
        xlabel="Temperatuurirühma keskmine (°C)",
        ylabel="MAE (CSV koormusühik)",
    )
    ax.legend(fontsize=8)
    save(
        fig,
        "16_weather_error_profile.png",
        "Joonis võrdleb 2019 vigu viivitatud temperatuuri kümnes kirjeldavas kvantiilirühmas. Rühmad moodustati alles hindamise järel ning need ei lähe mudelisse ega stsenaariumilävendite määramisse. Erinevad kõverad näitavad, millises teadaolevas ilmastikukontekstis mudelite täpsus erineb; need ei eralda kellaaja, hooaja ja koormustaseme segavat mõju. Keskmine temperatuur ei kirjelda äärmuse kogu varieeruvust, mistõttu tuleb vaadata ka lävendipõhiseid stsenaariume ja vaatluste arvu.",
    )
    write_json(out / "reports/figure_interpretations.json", captions)
    confirmed = intervals[(intervals.period == "confirmation") & (intervals.block_hours == 168)]
    superiority = bool((confirmed.adjusted_high < 0).all())
    winner2019 = (
        performance[(performance.period == "confirmation") & performance.model.isin(cfg["models"])]
        .sort_values("MAE")
        .iloc[0]
    )
    winner2020 = (
        performance[(performance.period == "retrospective") & performance.model.isin(cfg["models"])]
        .sort_values("MAE")
        .iloc[0]
    )
    conclusion = f"2017–2018 valikuperioodi võitja on **{MODEL_NAMES[champion]}**. 2019 arvuliselt parim finalist on **{MODEL_NAMES[winner2019.model]}**, MAE **{winner2019.MAE:.3f}**; 2020 retrospektiivselt parim on **{MODEL_NAMES[winner2020.model]}**, MAE **{winner2020.MAE:.3f}**. "
    conclusion += (
        "2019 peamises 168 tunni ploki kontrollis jäävad kõik neli Bonferroni tasemega intervalli nullist allapoole; see toetab valikuvõitja täpsuseelist võrreldud finalistide ees selles perioodis."
        if superiority
        else "2019 peamised paarisintervallid ei kinnita valikuvõitja kindlat paremust kõigi konkurentide ees. Võitja nimetus jääb vastava valikukriteeriumi ja perioodi piiresse."
    )
    write_json(
        out / "reports/conclusion.json",
        {
            "selection_champion": champion,
            "confirmation_numerical_best": winner2019.model,
            "retrospective_numerical_best": winner2020.model,
            "adjusted_superiority_over_all_2019": superiority,
            "text": conclusion,
        },
    )
