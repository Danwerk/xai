"""Koosta lühike uurimisaruanne olemasolevatest prognoosidest; mudeleid ei treenita."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import latest_run, read_artifact_csv, read_json, sha256, write_csv, write_json
from src.evaluation import accept_refinement, scenario_loss_interval, seasonal_error_scale
from src.feature_engineering import load_run_features, split_features
from src.labels import MODEL_NAMES, feature_label
from src.card_views import number
from src.expanded_reporting import FEATURE_NAMES
from src.model_comparison import temporal_split
from src.reporting import markdown_table, save_figure
from src.scenario_detector import detect_scenarios, LABELS
from src.train_baseline import metrics


def calculate_review(original, expanded):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results/focused_reviews" / stamp
    output.mkdir(parents=True)
    sources = {}

    def read(path, dates=None):
        sources[str(path.relative_to(ROOT))] = sha256(path)
        return read_artifact_csv(path, parse_dates=dates)

    # Mõõdikud arvutatakse samadest külmutatud prognoosidest.
    frame = load_run_features(expanded, expanded=True)
    configuration = read_json(expanded / "configs/expanded_study.json")
    performance = read(expanded / "evaluation/performance.csv")
    metric_rows = []

    for period, year in (("confirmation", 2019), ("retrospective", 2020)):
        train, valid = temporal_split(frame, year, configuration["gap_hours"])
        scale = seasonal_error_scale(train.target_next_hour)
        for model in performance.loc[performance.period == period, "model"]:
            prediction = read(expanded / f"predictions/{period}_{model}.csv")
            pd.testing.assert_series_equal(
                pd.to_datetime(prediction.target_time), valid.target_time, check_names=False
            )
            measured = metrics(prediction.target_next_hour, prediction.prediction)
            metric_rows.append(
                {
                    "year": year,
                    "model": model,
                    "n": measured["n"],
                    "MAE": measured["MAE"],
                    "RMSE": measured["RMSE"],
                    "MASE": measured["MAE"] / scale,
                    "bias": measured["bias"],
                }
            )

    metric_table = pd.DataFrame(metric_rows)
    write_csv(output / "metrics.csv", metric_table)

    # Vastuvõtureegli otsus kasutab ainult 2019. aastat.
    thresholds_path = original / "scenarios/train_thresholds.json"
    thresholds = read_json(thresholds_path)
    sources[str(thresholds_path.relative_to(ROOT))] = sha256(thresholds_path)
    selected = read(original / "refinements/selected.csv")
    cases = {
        "rapid_change": ("Kiired koormusmuutused", ["rapid_increase", "rapid_decrease"]),
        "temperature": ("Kõrge või madal temperatuur", ["heat", "cold"]),
    }
    decisions = []
    parts = split_features(
        load_run_features(original), read_json(original / "configs/experiment.json")
    )
    selection = parts["selection"]
    flags = detect_scenarios(selection, thresholds)
    base = read(original / "baseline/selection_GBR.csv", ["target_time"])

    for case, (label, scenario_names) in cases.items():
        # Sama tulemusega valikumeetodid ei ole sõltumatud korduskatsed.
        for group in selected.loc[selected.case == case, "group"].unique():
            proposed = read(original / f"refinements/selection_{group}.csv", ["target_time"])
            pd.testing.assert_series_equal(base.target_time, proposed.target_time)
            pd.testing.assert_series_equal(base.target_time, selection.target_time)
            mask = flags[scenario_names].any(axis=1)
            difference = proposed.abs_error - base.abs_error
            overall_change = float(difference.mean())
            scenario_change = float(difference[mask].mean())
            decisions.append(
                {
                    "case": case,
                    "label": label,
                    "group": group,
                    "selection_n": int(mask.sum()),
                    "selection_overall_change": overall_change,
                    "selection_scenario_change": scenario_change,
                    "accepted": accept_refinement(overall_change, scenario_change, int(mask.sum())),
                }
            )

    write_json(
        output / "decisions.json",
        {
            "study_type": "Uuriv järelkatse; 2020 tulemusi oli varem nähtud.",
            "decision_year": 2019,
            "evaluation_year": 2020,
            "rule": "Üldine MAE ja sihtolukorra MAE peavad vähenema; sihtolukorras vähemalt 40 tundi.",
            "decisions": decisions,
        },
    )

    # Hilisem hindamine ei muuda vastuvõtuotsust.
    test = parts["test"]
    test_flags = detect_scenarios(test, thresholds)
    test_base = read(original / "baseline/test_GBR.csv", ["target_time"])
    refinement_rows = []

    for decision in decisions:
        proposed = read(original / f"refinements/test_{decision['group']}.csv", ["target_time"])
        pd.testing.assert_series_equal(test_base.target_time, proposed.target_time)
        pd.testing.assert_series_equal(test_base.target_time, test.target_time)
        mask = test_flags[cases[decision["case"]][1]].any(axis=1)
        chosen = proposed if decision["accepted"] else test_base
        policy_difference = chosen.abs_error - proposed.abs_error
        interval = scenario_loss_interval(policy_difference, mask, comparisons=len(decisions))
        refinement_rows.append(
            {
                **decision,
                "test_n": int(mask.sum()),
                "base_overall_MAE": float(test_base.abs_error.mean()),
                "proposed_overall_MAE": float(proposed.abs_error.mean()),
                "guarded_overall_MAE": float(chosen.abs_error.mean()),
                "base_scenario_MAE": float(test_base.loc[mask, "abs_error"].mean()),
                "proposed_scenario_MAE": float(proposed.loc[mask, "abs_error"].mean()),
                "guarded_scenario_MAE": float(chosen.loc[mask, "abs_error"].mean()),
                **interval,
            }
        )

    refinements = pd.DataFrame(refinement_rows)
    write_csv(output / "refinement_check.csv", refinements)
    plot_effect(output, refinements)

    source_files = [
        Path(__file__),
        *sorted((ROOT / "src").glob("*.py")),
    ]
    manifest = {
        "status": "complete",
        "original_run": str(original),
        "expanded_run": str(expanded),
        "new_model_fits": 0,
        "source_sha256": sources,
        "code_and_protocol_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_files
        },
        "parent_manifest_sha256": {
            str(folder.relative_to(ROOT)): sha256(folder / "manifest.json")
            for folder in (original, expanded)
        },
        "artifacts": {path.name: sha256(path) for path in output.iterdir() if path.is_file()},
    }
    # Kontrollime, et aruande koostamine ei muutnud lähteprognoose.
    assert all(sha256(ROOT / name) == digest for name, digest in sources.items())
    write_json(output / "manifest.json", manifest)
    write_json(
        ROOT / "results/latest_focused_review.json",
        {"path": str(output), "manifest_sha256": sha256(output / "manifest.json")},
    )
    print(f"Koondaruanne valmis: {output}")
    return output


def plot_effect(output, refinements):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = refinements.label.str.replace(
        "Kõrge või madal temperatuur", "Kõrge või madal\ntemperatuur"
    )

    for axis, area, title in zip(
        axes, ("overall", "scenario"), ("Kõik tunnid", "Probleemsed tunnid")
    ):
        change = refinements[f"proposed_{area}_MAE"] - refinements[f"base_{area}_MAE"]
        axis.bar(labels, change, color=np.where(change < 0, "#397c6c", "#ae5454"))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=title, ylabel="MAE muutus võrreldes baasiga")
        axis.tick_params(axis="x", labelsize=9)

    figure.suptitle("Kas tunnusemuudatus aitab soovitud olukorras? (2020)")
    save_figure(figure, output / "refinement_effect.png")


def card_case_sections(original):
    """Seo kaks 2018. aasta kaarti hilisemate rühmapõhiste katsetega."""
    cards = [
        (path.parent.name, read_json(path))
        for path in sorted((original / "cards").glob("card_*/evidence.json"))
    ]
    config = read_json(original / "configs/experiment.json")
    trials = read_artifact_csv(original / "refinements/all_trials.csv")
    selected = read_artifact_csv(original / "refinements/selected.csv")
    tests = read_artifact_csv(original / "refinements/test_metrics.csv")
    baseline = read_artifact_csv(original / "baseline/metrics.csv").query("model == 'GBR'")
    sections = [
        "Näited valitakse ainult 2018. aasta kaartide põhjal: kaartide senises järjekorras "
        "võetakse esimene vastava olukorra hüpoteesiga kaart; sellise puudumisel esimene "
        "selle olukorra kaart. Hilisem täpsusemuutus ei juhi "
        "näite valikut. 2019. ja 2020. aasta arvud kirjeldavad kogu juhtumiuuringut, "
        "mitte üksiku kaardil näidatud tunni parandamist."
    ]
    for case, title, scenarios in (
        ("rapid_change", "Kiire koormusmuutus", ["rapid_increase", "rapid_decrease"]),
        ("temperature", "Kõrge või madal temperatuur", ["heat", "cold"]),
    ):
        candidates = [
            (name, card)
            for name, card in cards
            if set(card["scenarios"]) & {LABELS[s] for s in scenarios}
        ]
        name, card = next(
            (
                item
                for item in candidates
                if item[1]["diagnosis"]["experiment"] in config["case_studies"][case]
            ),
            candidates[0],
        )
        group = selected.query("method == 'XAI' and case == @case").iloc[0].group
        choice = trials.query("method == 'XAI' and case == @case and group == @group").iloc[0]
        test = tests.query("method == 'XAI' and case == @case").iloc[0]
        contributions = "; ".join(
            f"{feature_label(item['feature'], simple=True).lower()}: {number(item['shap'])}"
            for item in card["shap_contributions"][:3]
        )
        hypothesis = (
            card["diagnosis"]["hypothesis"]
            or "Kaart jättis hüpoteesi pakkumata, sest reegli jaoks ei olnud piisavalt tõendeid."
        )
        if card["diagnosis"]["experiment"] == group:
            connection = "Kaardi soovitatud tunnuserühm langes kokku hilisema XAI-põhise valikuga. See ei tõenda üksikkaardi mõju valikule."
        else:
            connection = "Allpool hinnatud muudatus valiti eraldi koondskoori ja 2019. aasta võrdluse abil. Seda ei saa esitada selle kaardi soovituse tulemusena."
        rows = []
        for year, period, result in (
            (2019, "Valik 2019", choice),
            (2020, "Lõpphindamine 2020", test),
        ):
            rows.append(
                {
                    "Aasta": year,
                    "Algmudeli üldine MAE": baseline.loc[baseline.period == period, "MAE"].iloc[0],
                    "Muudatuse üldine MAE": result.MAE,
                    "Olukorra tunde": int(result.target_scenario_n),
                    "Algmudeli olukorra MAE": result.target_scenario_MAE
                    - result.target_scenario_delta_MAE,
                    "Muudatuse olukorra MAE": result.target_scenario_MAE,
                }
            )
        added = ", ".join(config["candidate_groups"][group])
        conclusion = (
            "Üldine viga vähenes, kuid probleemsete tundide viga suurenes. Soovitud olukorra parandamine ei leidnud toetust."
            if test.delta_MAE_vs_baseline < 0 and test.target_scenario_delta_MAE > 0
            else "Tabel näitab kirjeldavaid erinevusi. Muudatuse kasu tuleb hinnata koos erinevuse suuruse, ebakindluse ja üldise ning olukorrapõhise veaga."
        )
        sections.append(
            f"#### {title}\n\n"
            f"**Juhtum.** {card['target_time']} (Panama kohalik aeg), {name}. "
            f"Prognoos {number(card['prediction'])}, tegelik koormus {number(card['actual'])}, "
            f"viga {number(card['error'])} lähtefaili koormusühikut. "
            f"{card.get('selection_reason', 'Varasema katse valitud näidisjuhtum.')}\n\n"
            f"**Mudeli selgitus.** Kolm suurimat absoluutset SHAP-panust: {contributions}. "
            f"Varasemaid sarnaseid juhtumeid: {len(card['historical_cases'])}. "
            "Panused kirjeldavad prognoosi kujunemist.\n\n"
            f"**Kaardi järeldus.** {hypothesis}\n\n"
            f"**Kontrollkatse.** Algsele GBR-ile lisati rühm `{group}`: {added}. {connection}\n\n"
            + markdown_table(pd.DataFrame(rows))
            + f"\n\n**Mida õppisime?** {conclusion}"
        )
    return "\n\n".join(sections)


def result_sections(original, expanded, focused, cards):
    """Arvud loetakse tulemustest; uurimuse selgitus jääb Markdown-faili."""
    model_table = read_artifact_csv(focused / "metrics.csv")
    model_table["model"] = model_table.model.map(MODEL_NAMES)
    model_table = model_table.rename(
        columns={"year": "Aasta", "model": "Mudel", "n": "Tunde", "bias": "Suunatud viga"}
    )
    ranking = read_artifact_csv(expanded / "selection/feature_ranking.csv")
    features = (
        ranking.set_index("feature_set")
        .loc[
            ["baseline", "history_calendar", "all_features", "remove_seasonal_history"],
            ["n_features", "MAE", "RMSE"],
        ]
        .reset_index()
    )
    features["feature_set"] = features.feature_set.map(FEATURE_NAMES)
    features = features.rename(columns={"feature_set": "Tunnused", "n_features": "Tunnuste arv"})
    comparison = read_artifact_csv(focused / "refinement_check.csv")
    changes = comparison[
        [
            "label",
            "test_n",
            "base_overall_MAE",
            "proposed_overall_MAE",
            "base_scenario_MAE",
            "proposed_scenario_MAE",
        ]
    ].rename(
        columns={
            "label": "Olukord",
            "test_n": "Probleemseid tunde",
            "base_overall_MAE": "Algmudeli üldine MAE",
            "proposed_overall_MAE": "Muudatuse üldine MAE",
            "base_scenario_MAE": "Algmudeli olukorra MAE",
            "proposed_scenario_MAE": "Muudatuse olukorra MAE",
        }
    )
    guard = comparison[
        [
            "label",
            "selection_overall_change",
            "selection_scenario_change",
            "accepted",
            "difference",
            "adjusted_low",
            "adjusted_high",
        ]
    ].copy()
    guard["accepted"] = guard.accepted.map({True: "Võta muudatus vastu", False: "Säilita algmudel"})
    guard = guard.rename(
        columns={
            "label": "Olukord",
            "selection_overall_change": "Üldise MAE muutus 2019",
            "selection_scenario_change": "Olukorra MAE muutus 2019",
            "accepted": "Otsus",
            "difference": "Olukorra MAE vahe 2020",
            "adjusted_low": "Vahemiku alampiir",
            "adjusted_high": "Vahemiku ülempiir",
        }
    )
    intervals = read_artifact_csv(original / "evaluation/budget_intervals.csv")
    intervals = intervals.loc[
        intervals.block_hours == 168,
        ["case", "scope", "difference", "adjusted_low", "adjusted_high"],
    ].copy()
    intervals["scope"] = intervals.scope.map(
        {"overall": "Kõik tunnid", "scenario": "Probleemsed tunnid"}
    )
    intervals = intervals.rename(
        columns={
            "scope": "Hinnatav osa",
            "difference": "XAI miinus veapõhise valiku MAE",
            "adjusted_low": "Vahemiku alampiir",
            "adjusted_high": "Vahemiku ülempiir",
        }
    )
    stability = read_artifact_csv(original / "explanations/stability.csv")
    index = read_artifact_csv(cards / "card_index.csv")
    count = int(index.has_hypothesis.sum())
    top5_range = f"{stability.top5_jaccard.min():.2f}–{stability.top5_jaccard.max():.2f}".replace(
        ".", ","
    )
    rank_range = f"{stability.rank_spearman.min():.3f}–{stability.rank_spearman.max():.3f}".replace(
        ".", ","
    )
    performance = read_artifact_csv(expanded / "evaluation/performance.csv")
    names = read_json(expanded / "configs/expanded_study.json")["models"]
    pairs = read_artifact_csv(expanded / "evaluation/all_pairwise.csv")
    pair_rows = []
    for period in ("confirmation", "retrospective"):
        ordered = performance[
            (performance.period == period) & performance.model.isin(names)
        ].sort_values("MAE")
        best, runner = ordered.iloc[0].model, ordered.iloc[1].model
        pair = pairs[
            (pairs.period == period)
            & (pairs.block_hours == 168)
            & (
                ((pairs.A == best) & (pairs.B == runner))
                | ((pairs.B == best) & (pairs.A == runner))
            )
        ].iloc[0]
        difference, low, high = pair.difference, pair.adjusted_low, pair.adjusted_high
        if pair.A != best:
            difference, low, high = -difference, -high, -low
        pair_rows.append(
            {
                "Aasta": 2019 if period == "confirmation" else 2020,
                "Võrdlus": f"{MODEL_NAMES[best]} − {MODEL_NAMES[runner]}",
                "MAE vahe": difference,
                "Vahemiku alampiir": low,
                "Vahemiku ülempiir": high,
            }
        )
    provenance = (
        "- [Mudelite mõõdikud](metrics.csv).\n"
        "- [Kõigi tunnuse-, mudeli- ja muutmiskatsete koondtulemused](trials.csv).\n"
        "- [Paarisvõrdlused ja nende usaldusvahemikud](comparisons.csv).\n\n"
        "Tunnipõhised kohalikud väljundid tekivad uuringu käivitamisel `results/` alla. "
        "Neid, treenitud mudeleid ega lähtekoodi varukoopiaid GitHubi ei lisata. "
        "Kordamiseks on vaja algandmeid, seda lähtekoodi ja `configs/` seadistusi.\n\n"
        "Tunnusetabeleid ja ajalisi andmejaotusi eraldi failidesse ei salvestata: "
        "need koostatakse vajaduse korral algandmetest ning katse seadistustest uuesti mälus. "
        "Tunnuse- ja mudelivalikust säilivad mõõdikud; lõppvõrdlustest säilivad kontrolliks "
        "ja usaldusvahemike arvutamiseks vajalikud prognoosid. Eelarvekatse kasutab olemasolevaid "
        "prognoose neid kopeerimata. See muudab failide salvestamist, mitte katsemeetodit "
        "ega siin esitatud teadustulemusi."
    )
    return {
        "cases": card_case_sections(original),
        "models": markdown_table(model_table),
        "features": markdown_table(features),
        "pairs": markdown_table(pd.DataFrame(pair_rows)),
        "refinements": markdown_table(changes),
        "budget": markdown_table(intervals),
        "guard": markdown_table(guard),
        "effect_figure": "![Tunnusemuudatuse mõju üldisele ja probleemsete tundide veale](refinement_effect.png)",
        "cards": (
            f"Koostatud on **{len(index)} juhtumi kaks vaadet**. Kontrollitav hüpotees esitati {count} juhul; "
            f"{len(index) - count} juhul jäi see tõendite vähesuse tõttu esitamata. Kõik algsete kaartide arvukontrollid läbisid kontrolli. "
            f"Viie olulisima tunnuse Jaccardi sarnasus oli salvestatud stabiilsusvõrdlustes "
            f"{top5_range} ja järjestuste Spearmani seos {rank_range}. "
            "Need on koondatud tunnusejärjestuste tulemused, mitte tõend iga üksikjuhtumi selgituse stabiilsuse kohta.\n\n"
            "Kaardivaated ja piloodimaterjalid tekivad käsuga `python tools/build_card_study.py` "
            "kausta `results/card_studies/`. Osalejate tulemusi veel pole."
        ),
        "sources": provenance,
    }


def specialist_section(original):
    results = read_artifact_csv(original / "refinements/specialist_metrics.csv")
    measured = results[results.stage.isin(["development", "final"])]
    chosen = measured[measured.selected]
    base = measured[measured.model == "GBR"].set_index(["year", "scope"])
    rows = []
    for row in chosen.itertuples():
        old = base.loc[row.year, row.scope]
        rows.append(
            {
                "Aasta": row.year,
                "Hinnatav osa": "Kõik tunnid" if row.scope == "overall" else "Kiire kasv",
                "Tunde": row.n,
                "Algmudeli MAE": old.MAE,
                "Parandatud MAE": row.MAE,
                "MAE vähenemine (%)": 100 * (1 - row.MAE / old.MAE),
                "Algmudeli RMSE": old.RMSE,
                "Parandatud RMSE": row.RMSE,
            }
        )
    intervals = results[(results.stage == "interval") & (results.block_hours == 168)]
    comparisons = pd.DataFrame(
        {
            "Võrdlusmudel": "Algne GBR" if row.reference == "GBR" else "HistGBR samade tunnustega",
            "Hinnatav osa": "Kõik tunnid" if row.scope == "overall" else "Kiire kasv",
            "MAE vahe": row.delta_MAE,
            "MAE vahe 95% vahemik": f"[{number(row.adjusted_low)}; {number(row.adjusted_high)}]",
            "RMSE vahe": row.delta_RMSE,
            "RMSE vahe 95% vahemik": f"[{number(row.rmse_adjusted_low)}; {number(row.rmse_adjusted_high)}]",
        }
        for row in intervals.itertuples()
    )
    count = measured[measured.stage == "development"].model.nunique() - 1
    winner = chosen.model.iloc[0]
    descriptions = {
        "GBR": "Ükski variant ei täitnud kõiki tingimusi; säilib algmudel.",
        "GBR_features": "GBR kasutab lisatud koormusajaloo ja kalendri tunnuseid.",
        "HistGBR_features": "HistGBR prognoosib koormust samade lisatunnuste abil.",
        "hard_blend": "Algmudeli prognoos ühendatakse kiire kasvu abimudeli prognoosiga.",
        "residual_ridge": "Ridge parandab GBR-i prognoosi varasemate aastate prognoosivigade põhjal.",
        "residual_hist": "HistGBR parandab GBR-i prognoosi varasemate aastate prognoosivigade põhjal.",
        "growth_weighted": "Veaparandaja annab kiire kasvu treeningnäidetele suurema kaalu.",
        "growth_only": "Parandaja õpib ainult kiire kasvu näidetest ja rakendub oodatava suure kasvu korral.",
    }
    method = winner.rsplit("_", 1)[0] if winner[-1].isdigit() else winner
    original = measured[(measured.stage == "final") & (measured.model == "hard_blend_1.0")]
    failed = original.set_index("scope")
    final_base = base.loc[2020]
    return (
        f"Algmudeliga võrreldi {count} varianti. Valituks osutus `{winner}`: "
        + descriptions[method]
        + " "
        + "Valik tehti 2018.–2019. aasta MAE ja RMSE järgi, arvestades eraldi kõiki tunde ja kiiret kasvu. "
        "Vähendati nende kaheksa suhte suurimat väärtust; kui ükski variant poleks kõigis parem, "
        "oleks säilinud algmudel.\n\n"
        + markdown_table(pd.DataFrame(rows))
        + "\n\n2020. aasta paarisvõrdlused (negatiivne vahe soosib parandatud mudelit):\n\n"
        + markdown_table(comparisons)
        + "\n\nVahemikud kasutavad 168 tunni plokke ja kaheksa võrdluse Bonferroni parandust "
        "(kaks võrdlusmudelit × kaks hinnatavat osa × kaks mõõdikut). 24 tunni plokkidega "
        "tundlikkuskontroll on paarisvõrdluste koondtabelis. Kõik ebaõnnestunud variandid "
        "on katsete koondtabelis. "
        "Arendusperioodi valik ise ei ole sõltumatu tõend paranemisest.\n\n"
        + f"Esimese, täieliku asendamisega abimudeli üldine MAE oli 2020. aastal {number(failed.loc['overall', 'MAE'])} "
        + f"(algmudel {number(final_base.loc['overall', 'MAE'])}) ja kiire kasvu MAE {number(failed.loc['rapid_increase', 'MAE'])} "
        + f"(algmudel {number(final_base.loc['rapid_increase', 'MAE'])}). See variant ei parandanud mõlemat osa. "
        "Ka ainult kiire kasvu vigadel treenitud parandaja halvendas arendusaastate üldist tulemust.\n\n"
        "2020. aasta tulemused on uurivad, sest seda perioodi ja andmestikku oli varem vaadatud. "
        "Uue otsingu varianti ei valitud 2020. aasta tulemuse põhjal, kuid vahemikud ei kõrvalda varasema "
        "andmete uurimise mõju. Katse ei tõenda SHAP-i lisakasu ega paremust kõigi varem võrreldud mudelite ees."
    )


def export_summary(original, expanded, focused):
    """GitHubi jaoks säilivad ainult väikesed teadustulemuste koondid."""
    reports = ROOT / "reports"
    write_csv(reports / "metrics.csv", read_artifact_csv(focused / "metrics.csv"))
    tables = []
    for kind, folder, name in (
        ("feature_screen", expanded, "selection/feature_screen.csv"),
        ("model_selection", expanded, "selection/model_folds.csv"),
        ("refinement_selection", original, "refinements/all_trials.csv"),
        ("refinement_test", original, "refinements/test_metrics.csv"),
        ("budget", original, "refinements/budget_metrics.csv"),
        ("specialist", original, "refinements/specialist_metrics.csv"),
    ):
        table = read_artifact_csv(folder / name)
        if kind == "specialist":
            table = table[table.stage != "interval"]
        tables.append(table.assign(experiment=kind, source_run=folder.name))
    prior_trials = read_artifact_csv(reports / "trials.csv")
    retained = prior_trials[
        prior_trials.experiment.isin(["attribution_refinement", "attribution_transfer"])
    ]
    write_csv(reports / "trials.csv", pd.concat([*tables, retained], ignore_index=True))
    comparisons = []
    for kind, folder, name in (
        ("models", expanded, "evaluation/all_pairwise.csv"),
        ("original", original, "evaluation/paired_block_bootstrap.csv"),
        ("budget", original, "evaluation/budget_intervals.csv"),
        ("acceptance_rule", focused, "refinement_check.csv"),
    ):
        comparisons.append(
            read_artifact_csv(folder / name).assign(experiment=kind, source_run=folder.name)
        )
    specialist = read_artifact_csv(original / "refinements/specialist_metrics.csv")
    comparisons.append(
        specialist[specialist.stage == "interval"].assign(
            experiment="specialist", source_run=original.name
        )
    )
    prior_comparisons = read_artifact_csv(reports / "comparisons.csv")
    retained = prior_comparisons[
        prior_comparisons.experiment.isin(["attribution_refinement", "attribution_transfer"])
    ]
    write_csv(reports / "comparisons.csv", pd.concat([*comparisons, retained], ignore_index=True))
    shutil.copyfile(focused / "refinement_effect.png", reports / "refinement_effect.png")


def update_report(sections):
    path = ROOT / "reports/thesis_results.md"
    report = path.read_text(encoding="utf-8")
    for name, content in sections.items():
        pattern = rf"<!-- results:{name} -->.*?<!-- /results:{name} -->"
        replacement = f"<!-- results:{name} -->\n\n{content}\n\n<!-- /results:{name} -->"
        report, count = re.subn(pattern, lambda _: replacement, report, flags=re.S)
        if count != 1:
            raise ValueError(f"Põhiaruandes peab olema üks tulemuste plokk: {name}")
    path.write_text(report, encoding="utf-8")


def refresh_attribution_intervals(original):
    """Upgrade row-based intervals to clock-hour blocks without retraining."""
    from src.refinement_experiments import attribution_intervals

    manifest = read_json(original / "manifest.json")
    changed = False
    path = ROOT / "reports/comparisons.csv"
    table = read_artifact_csv(path)
    for experiment, filename in (
        ("attribution_refinement", "test_attribution.csv"),
        ("attribution_transfer", "test_attribution_estonia.csv"),
    ):
        decision = manifest.get(experiment, {})
        if (
            decision.get("status") != "complete"
            or decision.get("interval_alignment") == "clock_hours"
        ):
            continue
        predicted = read_artifact_csv(
            original / f"refinements/{filename}", parse_dates=["target_time"]
        )
        forecasts = {
            c.removeprefix("prediction_"): predicted[c].to_numpy()
            for c in predicted
            if c.startswith("prediction_")
        }
        configuration = read_json(ROOT / "configs/experiment.json")
        configuration["bootstrap_repeats"] = decision["settings"]["bootstrap_repeats"]
        intervals = attribution_intervals(
            predicted,
            forecasts,
            decision["thresholds"],
            configuration,
            decision["settings"]["test_year"],
        )
        table = pd.concat(
            [
                table[table.experiment != experiment],
                intervals.assign(experiment=experiment, source_run=original.name),
            ],
            ignore_index=True,
        )
        decision.update(
            interval_alignment="clock_hours",
            intervals_updated_utc=datetime.now(timezone.utc).isoformat(),
            interval_correction="Missing forecast windows are masked on the actual hourly calendar; forecasts and selection unchanged.",
            interval_source_sha256={
                p: sha256(ROOT / p)
                for p in (
                    "src/evaluation.py",
                    "src/refinement_experiments.py",
                    "tools/build_report.py",
                )
            },
        )
        changed = True
    if changed:
        write_csv(path, table)
        write_json(original / "manifest.json", manifest)
        write_json(
            ROOT / "results/latest_run.json",
            {"path": str(original), "manifest_sha256": sha256(original / "manifest.json")},
        )


def attribution_section(original):
    from src.card_generator import make_repair_audit
    from src.card_views import render_repair_audit
    from src.labels import REPAIR_METHOD_NAMES
    from src.reporting import plot_repair_ledger

    manifest = read_json(original / "manifest.json")
    trials = read_artifact_csv(ROOT / "reports/trials.csv")
    comparisons = read_artifact_csv(ROOT / "reports/comparisons.csv")
    sections = [
        "Uus katseseeria eristab prognoosi panust ja paranduse tegelikku kasu. "
        "Esmalt katsetati ajatingimuslikku SHAP-panuste lineaarset parandust. "
        "See jäi 2018.–2019. aasta arenduses võrreldavale algtunnuste kihile alla; "
        "ebaõnnestunud variandid säilivad katsete koondis. Järgmiseks katsetati "
        "veaparandaja osade juhtimist varem selgunud paranduskasu põhjal.",
        "**Meetod.** Ajaliselt treeningust välja jäetud GBR-i vigadel õpib HistGBR-i "
        "veaparandaja. Tema üksiktunnuste TreeSHAP-panused summeeritakse sisulisteks "
        "parandusosadeks c_j; võrdlusväärtus on eraldi osa. Nende summa taastab paranduse. "
        "See summeerimine ei ole algse mudeli kanoonilise rühma-SHAP-i uus arvutus. "
        "Rühmad on uue parandusmängu mängijad ning algseid sisendeid selles mängus ei maskeerita.",
        "Olgu r = y − f(x) ja C = Σ c_j. Parandusmäng on "
        "v(S) = L(y,f(x)) − L(y,f(x)+Σ_(j∈S)c_j). Ruutvea korral on selle täpne "
        "Shapley-jaotus **ψ_j = c_j(2r − C)** ja Σ ψ_j = r² − (r−C)². "
        "Absoluutvea jaotus arvutatakse väikese parandusmängu kõigi koalitsioonide põhjal. "
        "Need on realiseerunud paranduskasu jaotused, mitte põhjuslikud veadiagnoosid "
        "ega muudetud prognoosi uued sisendtunnuste SHAP-väärtused. Ruutvea valem on "
        "Shapley-mängu algebraline erijuht; seda ei esitata uue üldise Shapley teoreemina.",
        "**Tagasiside.** Varem selgunud potentsiaalse täisparanduse ψ_j keskmistatakse "
        "sama sihttunni varasemate juhtumite või jooksva ajaloo üle. Negatiivse kasuga "
        "osa osakaalu vähendatakse. Prognoosihetkel t ei kasutata y_(t+1) ega selle kasujaotust. "
        "Esimesed piisava ajaloota juhtumid kasutavad täisparandust. Võrdlused on "
        "ilma tagasisideta veaparandaja; sama andmestikuga üksikosa iseseisval kasul põhinev "
        "tagasiside; ning ühe üldise osakaaluga tagasiside, mis ei vaja osade SHAP-jaotust. "
        "Seega saab eraldi hinnata tagasiside ning osade koosmõju jaotamise lisakasu.",
        "**Valik ja hindamine.** Perioodilise paranduskihi mõlemad esitused said "
        "sama 3 × 3 seadistuste hulga. Tagasiside iga liik sai sama kahe akna, kahe "
        "ajatingimuse ja kahe tugevuse hulga ning muutmata jätmise võimaluse. "
        "Valik kasutab ainult 2018.–2019. aasta vigu. Tagasiside valik vähendab üldise, "
        "kasvu ja languse MAE suhete keskmist ning ei luba üheski arendusaasta olukorras "
        "üle 5% halvenemist. See piir on uuriv disainivalik, mitte tõestatud optimaalne lävend. "
        "Kõik valikud salvestati enne hilisema perioodi prognooside arvutamist. "
        "Uue katseseeria GBR-i treeningread erinevad algsest RQ2-katsest, sest kõigile "
        "mudelitele kehtib nelja nädala ajaloo ühismask. Neid GBR-i tulemusi ei samastata.",
    ]
    for experiment, country, suffix in (
        ("attribution_refinement", "Panama", "panama"),
        ("attribution_transfer", "Eesti", "estonia"),
    ):
        decision = manifest.get(experiment, {})
        if decision.get("status") != "complete":
            sections.append(f"**{country}:** selle katseseeria lõpphindamine pole lõpetatud.")
            continue
        final = trials[(trials.experiment == experiment) & (trials.stage == "final")]
        path = original / (
            "refinements/test_attribution_estonia.csv"
            if suffix == "estonia"
            else "refinements/test_attribution.csv"
        )
        predicted = read_artifact_csv(path)
        year = decision["settings"]["test_year"]
        sections.append(f"**{country}, {year}: {len(predicted)} hindamistundi.**")
        if suffix == "estonia":
            quality = decision["data_quality"]
            sections.append(
                "Eraldi ülekandekatse kasutab Eleringi tegelikku tarbimist. "
                "Panama paranduskihi ja tagasiside seadistusi Eestis ümber ei valitud. "
                "Mudelid sobitati kohalikel varasematel andmetel, algoritmi valikureegel jäi fikseerituks. "
                "Ilm ning püha/kooli märgendid puuduvad ja jäeti kõigil vastavatel mudelitel välja. "
                "Tunni koormus tähistab lõppenud intervalli keskmist; jaotused on UTC järgi, kalender "
                "Europe/Tallinn järgi. Allalaaditud 2025. aasta osa jäeti enne prognoosivigade "
                "arvutamist kõrvale selle puudulikkuse tõttu, mitte mudeli tulemuse põhjal. "
                f"Kasutatud 2018.–2024. aasta osas oli {quality['missing_hourly_load']} puuduvat "
                f"või sobimatut tunniväärtust; neist {quality['nonpositive_load_unavailable']} olid "
                "nullkoormused, mida käsitleti riikliku koormuse puuduva mõõtmisena. "
                "Väärtusi ei interpoleeritud. Puuduliku ajaloo või sihiga aknad jäeti välja. "
                "Uuring eeldab viimase lõppenud tunni koormuse ja vea kohest saadavust; "
                "tegelikke avaldamisviivitusi selle allika põhjal ei kinnitatud."
            )
        else:
            sections.append(
                "2020. aasta oli varem uuritud: tulemus on tagasivaateline arendustõend."
            )
        rows = []
        for name in (
            "GBR",
            "Ridge",
            "HistGBR",
            "HistResidual",
            "raw_memory",
            "shap_memory",
            "scalar_feedback",
            "standalone_feedback",
            "credit_feedback",
        ):
            measured = final[final.model == name].set_index("scope")
            rows.append(
                {
                    "Mudel": REPAIR_METHOD_NAMES[name],
                    "MAE": measured.loc["overall", "MAE"],
                    "RMSE": measured.loc["overall", "RMSE"],
                    "Kasvu MAE": measured.loc["rapid_increase", "MAE"],
                    "Languse MAE": measured.loc["rapid_decrease", "MAE"],
                    "Halvenenud tunde (%)": measured.loc["overall", "harm_share"] * 100,
                }
            )
        sections.append(markdown_table(pd.DataFrame(rows)))
        sections.append(
            "Halvenenud tundide osakaal võrdleb iga mudeli absoluutviga algse GBR-iga, "
            "mitte veaparandajaga. Väiksem MAE ei taga väiksemat halvenenud tundide osakaalu."
        )
        subset = comparisons[
            (comparisons.experiment == experiment) & (comparisons.block_hours == 168)
        ]
        selected = subset[
            (
                (subset.scope == "overall")
                & subset.B.isin(["HistResidual", "Ridge", "standalone_feedback", "scalar_feedback"])
            )
            | (
                (subset.B == "HistResidual")
                & subset.scope.isin(["rapid_increase", "rapid_decrease"])
            )
        ]
        interval_rows = [
            {
                "Võrdlusmudel": REPAIR_METHOD_NAMES[row.B],
                "Hinnatav osa": {
                    "overall": "Kõik tunnid",
                    "rapid_increase": "Kiire kasv",
                    "rapid_decrease": "Kiire langus",
                }[row.scope],
                "MAE vahe": row.difference,
                "Korrigeeritud 95% vahemik": f"[{number(row.adjusted_low, 4)}; {number(row.adjusted_high, 4)}]",
            }
            for row in selected.itertuples()
        ]
        sections.append(markdown_table(pd.DataFrame(interval_rows)))
        sections.append(
            f"Negatiivne vahe soosib XAI-tagasisidet. {decision['comparison_family_size']} "
            f"võrdluse Bonferroni parandus, 168 tunni plokid, {decision['settings']['bootstrap_repeats']} "
            "kordust; 24 tunni tundlikkuskontroll on koondtabelis. Valikuprotsessi varasemat "
            "andmete uurimise mõju vahemik ei kõrvalda. Eesti ja Panama ühikuid omavahel ei võrrelda."
            " Puuduvad hindamisaknad on tegelikul tunnikalendril maskitud, nii et plokk "
            "tähendab tunde, mitte puuduliku tabeli järjestikuseid ridu."
        )
        figure = ROOT / f"reports/attribution_{suffix}.png"
        plot_repair_ledger(
            predicted, final, figure, f"{country} {year}: prognoosi muudatus ja mõõdetud kasu"
        )
        sections.append(f"![Paranduse ja kasu kaks vaadet: {country}](attribution_{suffix}.png)")
        gain = abs(predicted.prediction_GBR - predicted.target_next_hour) - abs(
            predicted.prediction_credit_feedback - predicted.target_next_hour
        )
        for label, index in (
            (
                "Algmudeli suurima absoluutveaga juhtum",
                int(abs(predicted.prediction_GBR - predicted.target_next_hour).idxmax()),
            ),
            ("Paranduse suurima kahjuga juhtum", int(gain.idxmin())),
        ):
            sections.append(
                f"**{label}.** Näite valikureegel on kirjeldav ega tõenda tüüpilist kasu."
            )
            sections.append(render_repair_audit(make_repair_audit(predicted.loc[index])))
    sections.append(
        "**Uudsuse piir.** Prognoosi-SHAP, veafunktsiooni Shapley-jaotus, Shapley-regressioon "
        "ning SHAP-iga juhitud mudelikaalud on varasemas kirjanduses olemas. Käesoleva töö "
        "uuritav kombinatsioon on varem realiseerunud paranduskasu täpse osajaotuse kasutamine "
        "elektrikoormuse paranduse ajatingimuslikus tagasisides ning sama jaotuse näitamine "
        "koos prognoosi muudatusega. Täpset sama kombinatsiooni sihitud otsingus ei leitud; "
        "esmakordsus pole tõendatud. Otsingu päringud ja lähimate tööde erinevused on "
        "[uurimismärkmes](research_directions.md). Mõõdetud täpsus ei tõenda kaartide "
        "arusaadavust ega kasu päris kasutajale."
    )
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Uuenda ainult kahe kaardijuhtumi läbikäiku; eeldab üksnes põhikatse tulemusi.",
    )
    parser.add_argument(
        "--specialist-only", action="store_true", help="Uuenda ainult abimudeli katse tulemusi."
    )
    parser.add_argument(
        "--attribution-only",
        action="store_true",
        help="Uuenda paranduskasu XAI-katse aruannet ja jooniseid olemasolevatest prognoosidest.",
    )
    args = parser.parse_args()
    original = latest_run()
    if args.attribution_only:
        refresh_attribution_intervals(original)
        update_report({"attribution": attribution_section(original)})
        print("Paranduskasu XAI-katse aruanne ja joonised uuendatud.")
        return
    if args.specialist_only:
        update_report({"specialist": specialist_section(original)})
        added = read_artifact_csv(original / "refinements/specialist_metrics.csv").assign(
            experiment="specialist", source_run=original.name
        )
        for name, mask in (
            ("trials", added.stage != "interval"),
            ("comparisons", added.stage == "interval"),
        ):
            path = ROOT / f"reports/{name}.csv"
            old = read_artifact_csv(path)
            write_csv(
                path,
                pd.concat([old[old.experiment != "specialist"], added[mask]], ignore_index=True),
            )
        print("Abimudeli katse aruanne ja koondtabel uuendatud.")
        return
    if args.cards_only:
        update_report({"cases": card_case_sections(original)})
        print("Kahe kaardijuhtumi läbikäik uuendatud. Mudeleid ei treenitud.")
        return
    expanded = latest_run("latest_expanded_run.json")
    pointer = ROOT / "results/latest_focused_review.json"
    focused = latest_run("latest_focused_review.json") if pointer.exists() else None
    meta = read_json(focused / "manifest.json") if focused else {}
    if meta.get("original_run") != str(original) or meta.get("expanded_run") != str(expanded):
        focused = calculate_review(original, expanded)
    cards = latest_run("latest_card_study.json")
    if Path(read_json(cards / "manifest.json")["parent_run"]) != original:
        raise ValueError("Koosta enne aruannet sama põhikäivituse kasutajakatse materjalid.")
    export_summary(original, expanded, focused)
    update_report(result_sections(original, expanded, focused, cards))
    update_report({"specialist": specialist_section(original)})
    if (
        read_json(original / "manifest.json").get("attribution_refinement", {}).get("status")
        == "complete"
    ):
        refresh_attribution_intervals(original)
        update_report({"attribution": attribution_section(original)})
    print("Põhiaruande tabelid ja viited uuendatud. Olemasolevaid katseid ei korratud.")


if __name__ == "__main__":
    main()
