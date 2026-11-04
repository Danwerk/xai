"""Exploratory, temporally controlled feature and model comparison extension."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd

from src.common import ROOT, read_config, sha256, write_csv, write_json
from src.data_preprocessing import load_and_audit, validate_for_modeling
from src.expanded_features import build_expanded_features, feature_catalog
from src.model_comparison import (
    temporal_split,
    make_model,
    fit_predict,
    rank_candidates,
    compare_model_pairs,
)
from src.evaluation import paired_intervals
from src.train_baseline import metrics, prediction_frame
from src.scenario_detector import fit_thresholds, detect_scenarios
from src.error_analyzer import scenario_summary
from src.shap_explainer import explain_prediction


def screen_features(evaluate, bundles, groups, cfg, out):
    """Compare predefined feature bundles and freeze the shared feature view."""
    screen_rows = []
    studies = dict(bundles)
    for group, columns in groups.items():
        studies["remove_" + group] = [c for c in bundles["all_features"] if c not in columns]
    for variant, columns in studies.items():
        for year in cfg["screen_years"]:
            result = evaluate(cfg["feature_screen_model"], 0, columns, year)
            screen_rows.append(
                {
                    "feature_set": variant,
                    "year": year,
                    "kind": "ablation" if variant.startswith("remove_") else "addition",
                    **result,
                }
            )
        print("Tunnusekatse: " + variant, flush=True)
    screen = pd.DataFrame(screen_rows)
    write_csv(out / "selection/feature_screen.csv", screen)
    screen_rank = rank_candidates(screen.assign(model=cfg["feature_screen_model"], parameter_id=0))
    screen_rank["kind"] = screen_rank.feature_set.map(
        lambda s: "ablation" if s.startswith("remove_") else "addition"
    )
    write_csv(out / "selection/feature_ranking.csv", screen_rank)
    feature_winner = screen_rank[screen_rank.kind == "addition"].iloc[0].feature_set
    shared = feature_winner if feature_winner != "baseline" else "all_features"
    frozen_features = {
        "selected_feature_set": feature_winner,
        "model_comparison_views": ["baseline", shared],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_outcomes_computed": False,
        "rule": "Lowest observation-weighted 2017/2018 MAE among predefined addition bundles; tie: feature count, lexical name.",
    }
    write_json(out / "selection/frozen_features.json", frozen_features)
    return shared


def select_finalists(evaluate, bundles, shared, cfg, out):
    """Rank the fixed candidates using only the earlier selection years."""
    rows = []
    for name, settings in cfg["models"].items():
        for parameter in range(len(settings)):
            for view in ["baseline", shared]:
                for year in cfg["screen_years"]:
                    result = evaluate(name, parameter, bundles[view], year)
                    rows.append(
                        {
                            "model": name,
                            "parameter_id": parameter,
                            "feature_set": view,
                            "year": year,
                            **result,
                        }
                    )
                print(
                    f"Mudelikandidaat: {name}, seadistus {parameter}, tunnused {view}",
                    flush=True,
                )
    selection = pd.DataFrame(rows)
    ranking = rank_candidates(selection)
    finalists = ranking.groupby("model", sort=False).head(1).reset_index(drop=True)
    champion = ranking.iloc[0].model
    write_csv(out / "selection/model_folds.csv", selection)
    write_csv(out / "selection/model_ranking.csv", ranking)
    write_csv(out / "selection/finalists.csv", finalists)
    freeze = {
        "champion": champion,
        "finalists": finalists[["model", "parameter_id", "feature_set"]].to_dict("records"),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_outcomes_computed": False,
        "retrospective_outcomes_computed": False,
        "feature_decision_sha256": sha256(out / "selection/frozen_features.json"),
    }
    write_json(out / "selection/frozen_finalists.json", freeze)
    return ranking, finalists, champion


def run(cfg, out):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Kasuta uut väljundkausta; varasemad katsed säilitatakse.")
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = read_config()
    write_json(out / "configs/expanded_study.json", cfg)
    write_json(out / "configs/original_experiment.json", base_cfg)
    sources = [
        ROOT / "run_expanded_study.py",
        ROOT / "requirements.txt",
        ROOT / "requirements-lock.txt",
    ]
    sources.extend(sorted((ROOT / "src").glob("*.py")))
    sources.extend(sorted((ROOT / "tests").glob("*.py")))
    sources.extend(sorted((ROOT / "tools").glob("*.py")))
    manifest = {
        "status": "running",
        "study_type": "Exploratory extension; previously inspected data, not an independent blind test",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "pandas",
                "numpy",
                "scikit-learn",
                "scipy",
                "shap",
                "matplotlib",
                "joblib",
                "threadpoolctl",
            ]
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "completed_stages": [],
    }
    write_json(out / "manifest.json", manifest)

    def stage(name):
        manifest["completed_stages"].append(name)
        write_json(out / "manifest.json", manifest)
        print("Valmis: " + name, flush=True)

    try:
        raw, quality = load_and_audit(ROOT / cfg["raw_path"])
        validate_for_modeling(raw, quality)
        manifest["raw_sha256"] = sha256(ROOT / cfg["raw_path"])
        write_json(out / "processed/quality.json", quality)
        frame, groups, bundles = build_expanded_features(raw, base_cfg)
        write_csv(out / "processed/feature_catalog.csv", feature_catalog(groups))
        write_json(out / "configs/feature_bundles.json", bundles)
        write_json(out / "configs/feature_groups.json", groups)
        folds = {
            year: temporal_split(frame, year, cfg["gap_hours"]) for year in cfg["screen_years"]
        }
        write_csv(
            out / "selection/folds.csv",
            pd.DataFrame(
                [
                    {
                        "year": year,
                        "train_n": len(a),
                        "valid_n": len(b),
                        "train_first": a.target_time.min(),
                        "train_last": a.target_time.max(),
                        "valid_first": b.target_time.min(),
                        "valid_last": b.target_time.max(),
                        "gap_hours": cfg["gap_hours"],
                    }
                    for year, (a, b) in folds.items()
                ]
            ),
        )
        stage(
            f"1. {len(frame)} ühist vaatlust; {len(groups)} uut tunnuserühma ja {len(bundles)} varianti"
        )
        cache = {}

        def evaluate(name, parameter, columns, year):
            params = cfg["models"][name][parameter]
            key = (name, parameter, tuple(columns), year)
            if key in cache:
                return cache[key]
            train, valid = folds[year]
            model = make_model(name, params, cfg)
            start = time.perf_counter()
            predicted = fit_predict(model, train, valid, columns, cfg["workers"])
            elapsed = time.perf_counter() - start
            result = {
                **metrics(valid.target_next_hour, predicted),
                "fit_seconds": elapsed,
                "n_features": len(columns),
            }
            cache[key] = result
            return result

        shared = screen_features(evaluate, bundles, groups, cfg, out)
        stage("2. Tunnuseuuring ja eemalduskatsed; ühine tunnusevaade fikseeritud")
        ranking, finalists, champion = select_finalists(evaluate, bundles, shared, cfg, out)
        stage("3. Viie ML-perekonna võrdne kandidaatide arv; finalistid fikseeritud")
        # Lävendid määratakse varaseima treeningperioodi põhjal.
        earliest_train = folds[min(cfg["screen_years"])][0]
        threshold_model = make_model("GBR", cfg["models"]["GBR"][0], cfg)
        training_prediction = fit_predict(
            threshold_model, earliest_train, earliest_train, bundles["baseline"], cfg["workers"]
        )
        thresholds = fit_thresholds(
            earliest_train, prediction_frame(earliest_train, training_prediction, "GBR")
        )
        write_json(out / "evaluation/scenario_thresholds.json", thresholds)
        model_objects = {}
        all_predictions = {}
        performance = []
        scenarios = []
        for year, period in [
            (cfg["confirmation_year"], "confirmation"),
            (cfg["retrospective_year"], "retrospective"),
        ]:
            train, valid = temporal_split(frame, year, cfg["gap_hours"])
            masks = detect_scenarios(valid, thresholds)
            write_csv(
                out / f"evaluation/{period}_scenario_flags.csv",
                pd.concat([valid[["target_time"]], masks], axis=1),
            )
            for _, row in finalists.iterrows():
                name = row.model
                columns = bundles[row.feature_set]
                model = make_model(name, cfg["models"][name][int(row.parameter_id)], cfg)
                start = time.perf_counter()
                predicted = fit_predict(model, train, valid, columns, cfg["workers"])
                elapsed = time.perf_counter() - start
                pred = prediction_frame(valid, predicted, name)
                model_objects[(period, name)] = (model, columns)
                all_predictions[(period, name)] = pred
                write_csv(out / f"predictions/{period}_{name}.csv", pred)
                nodes = None
                if name in ("RandomForest", "ExtraTrees"):
                    nodes = sum(t.tree_.node_count for t in model.estimators_)
                elif name == "GBR":
                    nodes = sum(t[0].tree_.node_count for t in model.estimators_)
                elif name == "HistGBR":
                    nodes = sum(len(t.nodes) for trees in model._predictors for t in trees)
                performance.append(
                    {
                        "period": period,
                        "year": year,
                        "model": name,
                        "feature_set": row.feature_set,
                        "parameter_id": int(row.parameter_id),
                        **metrics(valid.target_next_hour, predicted),
                        "fit_predict_seconds": elapsed,
                        "n_features": len(columns),
                        "tree_nodes": nodes,
                    }
                )
                summary = scenario_summary(
                    pred, masks, thresholds["large_error"], base_cfg["min_scenario_n"]
                )
                scenarios.append(summary.assign(period=period, model=name))
                print(f"Hilisem kontroll: {year}, {name}", flush=True)
            for name, column in [("Persistence", "load_now"), ("SeasonalNaive", "load_day")]:
                pred = prediction_frame(valid, valid[column], name)
                all_predictions[(period, name)] = pred
                write_csv(out / f"predictions/{period}_{name}.csv", pred)
                performance.append(
                    {
                        "period": period,
                        "year": year,
                        "model": name,
                        "feature_set": "baseline",
                        "parameter_id": -1,
                        **metrics(valid.target_next_hour, pred.prediction),
                        "fit_predict_seconds": 0.0,
                        "n_features": 1,
                        "tree_nodes": None,
                    }
                )
                scenarios.append(
                    scenario_summary(
                        pred, masks, thresholds["large_error"], base_cfg["min_scenario_n"]
                    ).assign(period=period, model=name)
                )
        performance = pd.DataFrame(performance)
        write_csv(out / "evaluation/performance.csv", performance)
        write_csv(out / "evaluation/scenarios.csv", pd.concat(scenarios, ignore_index=True))
        comparisons = []
        for period in ("confirmation", "retrospective"):
            a = all_predictions[(period, champion)]
            for other in cfg["models"]:
                if other == champion:
                    continue
                b = all_predictions[(period, other)]
                for block in cfg["bootstrap_blocks"]:
                    interval = paired_intervals(
                        a.target_next_hour,
                        a.prediction,
                        b.prediction,
                        block,
                        cfg["bootstrap_repeats"],
                        cfg["seed"],
                        len(cfg["models"]) - 1,
                    )
                    comparisons.append({"period": period, "A": champion, "B": other, **interval})
        write_csv(out / "evaluation/paired_intervals.csv", pd.DataFrame(comparisons))
        write_csv(
            out / "evaluation/all_pairwise.csv",
            compare_model_pairs(all_predictions, list(cfg["models"]), cfg),
        )
        stage("4. 2019 ajaline kontroll, 2020 retrospektiivne võrdlus ja paarisintervallid")
        tree_names = [n for n in finalists.model if n != "Ridge"]
        tree_champion = next(n for n in ranking.model if n in tree_names)
        second_tree = (
            tree_champion
            if tree_champion != "GBR"
            else next(n for n in ranking.model if n in tree_names and n != "GBR")
        )
        explain_names = ["GBR", second_tree]
        _, context = temporal_split(frame, cfg["confirmation_year"], cfg["gap_hours"])
        baseline_pred = all_predictions[("confirmation", "GBR")]
        center = int(baseline_pred.abs_error.idxmax())
        window = np.arange(max(0, center - 48), min(len(context), center + 25))
        rng = np.random.default_rng(cfg["seed"])
        ids = np.unique(
            np.r_[
                window,
                rng.choice(len(context), min(cfg["shap_sample"], len(context)), replace=False),
            ]
        )
        shap_inputs = context.iloc[ids].copy()
        write_csv(out / "explanations/inputs.csv", shap_inputs.assign(context_id=ids))
        for name in explain_names:
            model, columns = model_objects[("confirmation", name)]
            values, base = explain_prediction(model, shap_inputs[columns])
            exported = values.copy()
            exported.insert(0, "base_value", base)
            exported.insert(0, "target_time", shap_inputs.target_time)
            exported.insert(0, "context_id", ids)
            write_csv(out / f"explanations/{name}_shap.csv", exported)
        write_json(
            out / "explanations/window.json",
            {
                "center_context_id": center,
                "context_ids": window.tolist(),
                "center_target_time": str(context.loc[center, "target_time"]),
                "selection_rule": "Largest 2019 absolute error of the preselected GBR finalist; descriptive case, not representative accuracy.",
                "explained_models": explain_names,
                "tree_champion": tree_champion,
                "global_context_ids": [int(i) for i in ids if i not in set(window)],
            },
        )
        stage("5. Kahe puumudeli SHAP ja ajaliselt joondatud juhtumiaken")
        from src.expanded_reporting import make_expanded_report

        make_expanded_report(out, cfg, frame)
        stage("6. Eestikeelsed võrdlusjoonised ja mõõdetud aruanne")
        manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["frozen_finalists_sha256"] = sha256(out / "selection/frozen_finalists.json")
        manifest["artifact_sha256"] = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(out / "manifest.json", manifest)
        write_json(
            ROOT / "results/latest_expanded_run.json",
            {"path": str(out), "manifest_sha256": sha256(out / "manifest.json")},
        )
        return out
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(out / "manifest.json", manifest)
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Laiendatud tunnuste ja ML-meetodite uuriv võrdlus"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/expanded_study.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    cfg = read_config(args.config)
    out = args.output_dir or ROOT / "results/expanded_runs" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    print(run(cfg, out), flush=True)
