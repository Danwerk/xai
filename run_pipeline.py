"""Run all scientific stages on original data. No synthetic fallback."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd

from src.common import ROOT, read_config, read_json, sha256, write_csv, write_json
from src.data_preprocessing import load_and_audit, validate_for_modeling
from src.eda import run_eda
from src.feature_engineering import build_features, split_features, feature_manifest
from src.train_baseline import (
    fit_gbr,
    baseline_predictions,
    temporal_training_checks,
    prediction_frame,
    metrics,
)
from src.scenario_detector import fit_thresholds, detect_scenarios
from src.error_analyzer import scenario_summary, residual_associations
from src.shap_explainer import (
    explain_prediction,
    permutation_importance,
    compare_explanation_groups,
    analyze_feature_importance_stability,
)
from src.historical_cases import find_similar_cases, fit_similarity_scaler
from src.card_generator import make_card, select_card_cases
from src.card_views import save_role_cards
from src.refinement_experiments import (
    decision_evidence,
    freeze_decisions,
    run_refinements,
    budget_sensitivity,
    develop_specialist,
    evaluate_specialist,
    run_attribution_study,
)
from src.evaluation import audit_card, paired_block_bootstrap, mean_bias_interval
from src.reporting import make_figures


def log(message):
    print(message, flush=True)


def run(cfg, out):
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Väljundkaust peab olema uus või tühi, et säilitada varasemate katsete tõendid."
        )
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg["artifact_dir"] = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    write_json(out / "configs/experiment.json", cfg)
    source_paths = [
        ROOT / name
        for name in (
            "run_pipeline.py",
            "requirements.txt",
            "requirements-lock.txt",
            "pyproject.toml",
        )
        if (ROOT / name).exists()
    ]
    for folder, pattern in (
        ("src", "*.py"),
        ("tools", "*.py"),
        ("tests", "*.py"),
        ("templates", "*"),
    ):
        source_paths.extend(sorted((ROOT / folder).glob(pattern)))
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256(out / "configs/experiment.json"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "pandas",
                "scikit-learn",
                "shap",
                "scipy",
                "matplotlib",
                "joblib",
                "pytest",
                "nbformat",
                "nbclient",
                "jinja2",
            )
        },
        "completed_stages": [],
    }
    write_json(out / "manifest.json", manifest)

    def stage(name):
        manifest["completed_stages"].append(name)
        write_json(out / "manifest.json", manifest)
        log(f"Valmis: {name}")

    try:
        raw = ROOT / cfg["raw_path"]
        data, quality = load_and_audit(raw)
        manifest["raw_sha256"] = sha256(raw)
        write_json(out / "eda/quality.json", quality)
        validate_for_modeling(data, quality)
        stage("1. Andmekontroll")
        run_eda(data, out, cfg["train_end"])
        stage("2. Treeningandmete EDA")
        features = build_features(data, cfg)
        splits = split_features(features, cfg)
        write_csv(out / "processed/feature_manifest.csv", feature_manifest(cfg))
        write_json(
            out / "processed/row_accounting.json",
            {
                "raw_rows": len(data),
                "usable_rows": len(features),
                "excluded_boundary_rows": len(data) - len(features),
                "reason": "Ainult minevikuakende algus ja viimase rea puuduva sihi lõpp; sama mask kõigil mudelitel.",
            },
        )
        train, diagnosis, selection, test = [
            splits[k] for k in ("train", "diagnosis", "selection", "test")
        ]
        base_features = cfg["baseline_features"]
        stage("3. Ajaliselt kontrollitud tunnused")
        model = fit_gbr(train, base_features, cfg)
        checks = temporal_training_checks(train, cfg)
        write_csv(out / "baseline/training_temporal_checks.csv", checks)
        train_prediction = prediction_frame(train, model.predict(train[base_features]), "GBR")
        diagnosis_predictions = baseline_predictions(model, diagnosis, base_features)
        dp = diagnosis_predictions["GBR"]
        write_csv(out / "baseline/diagnosis_GBR.csv", dp)
        stage("4. Fikseeritud GBR ja võrdlusmudelid")
        thresholds = fit_thresholds(train, train_prediction)
        write_json(out / "scenarios/train_thresholds.json", thresholds)
        ds = detect_scenarios(diagnosis, thresholds)
        scenario_table = scenario_summary(dp, ds, thresholds["large_error"], cfg["min_scenario_n"])
        for i, row in scenario_table.iterrows():
            masked = dp.error.where(ds[row.scenario]).to_numpy()
            interval = mean_bias_interval(
                masked, block=168, repeats=cfg["bootstrap_repeats"], seed=cfg["seed"]
            )
            for k, v in interval.items():
                scenario_table.loc[i, k] = v
        write_csv(out / "scenarios/diagnosis.csv", scenario_table)
        write_csv(out / "scenarios/residual_associations.csv", residual_associations(diagnosis, dp))
        stage("5. Stsenaariumid ja prognoosivigade analüüs")
        rng = np.random.default_rng(cfg["seed"])
        sample_ids = np.sort(
            rng.choice(len(diagnosis), min(cfg["shap_sample"], len(diagnosis)), replace=False)
        )
        sample = diagnosis.iloc[sample_ids].reset_index(drop=True)
        sp = dp.iloc[sample_ids].reset_index(drop=True)
        ss = ds.iloc[sample_ids].reset_index(drop=True)
        values, base_values = explain_prediction(model, sample[base_features])
        shap_export = values.copy()
        shap_export.insert(0, "base_value", base_values)
        shap_export.insert(0, "target_time", sample.target_time)
        shap_export.insert(0, "sample_id", np.arange(len(sample)))
        write_csv(out / "explanations/shap_values.csv", shap_export)
        write_csv(out / "explanations/shap_input.csv", sample)
        comparison = compare_explanation_groups(values, sample, sp, thresholds["large_error"])
        write_csv(out / "explanations/matched_shap_comparison.csv", comparison)
        pfi_ids = np.sort(
            rng.choice(len(diagnosis), min(cfg["pfi_sample"], len(diagnosis)), replace=False)
        )
        groups = {c: [c] for c in base_features}
        groups.update(
            {
                "koormusajalugu": ["load_now", "load_day", "load_week"],
                "ilm": ["temperature", "humidity", "wind"],
                "kalendriaeg": [
                    c for c in base_features if c.endswith(("_sin", "_cos")) or c == "weekend"
                ],
            }
        )
        pfi = permutation_importance(
            model,
            diagnosis.loc[pfi_ids, base_features],
            diagnosis.loc[pfi_ids, "target_next_hour"],
            groups,
            cfg["pfi_repeats"],
            cfg["seed"],
        )
        write_csv(out / "explanations/permutation_importance.csv", pfi)
        stable = {
            f"Alamperiood {q}": values.loc[sample.target_time.dt.quarter == q] for q in range(1, 5)
        }
        # Sama SHAP-valim, treeningust eemaldatakse üks 2017. aasta kvartal.
        for q in (1, 3):
            subset = train[
                ~((train.target_time.dt.year == 2017) & (train.target_time.dt.quarter == q))
            ]
            variant = fit_gbr(subset, base_features, cfg)
            stable[f"Kordussobitus ilma 2017 Q{q}"] = explain_prediction(
                variant, sample[base_features]
            )[0]
        stable["Fikseeritud mudel"] = values
        stability = analyze_feature_importance_stability(stable)
        write_csv(out / "explanations/stability.csv", stability)
        stage("6. SHAP, PFI ja selgituste stabiilsus")
        similarity_features = [
            "load_now",
            "load_day",
            "temperature",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
        scaler = fit_similarity_scaler(train, similarity_features)
        history_pool = diagnosis.copy()
        history_pool["prediction"], history_pool["error"] = dp.prediction, dp.error
        card_cases = select_card_cases(sp, ss, cfg["card_count"])
        cards, historical_frames = [], []
        for number, (i, reason) in enumerate(card_cases.items()):
            historical = find_similar_cases(
                sample.loc[i], history_pool, similarity_features, scaler
            )
            if len(historical):
                historical_frames.append(historical.assign(query_id=i))
            card = make_card(
                sample.loc[i],
                sp.loc[i],
                values.loc[i],
                base_values[i],
                ss.loc[i],
                scenario_table,
                historical,
                cfg,
            )
            card["sample_id"] = i
            card["selection_reason"] = reason
            save_role_cards(card, out / "cards" / f"card_{number:02d}")
            cards.append(card)
        history = (
            pd.concat(historical_frames, ignore_index=True) if historical_frames else pd.DataFrame()
        )
        write_csv(out / "explanations/historical_cases.csv", history)
        audits = pd.DataFrame([audit_card(c) for c in cards])
        write_csv(out / "cards/audit.csv", audits)
        if not audits.all_pass.all():
            raise ValueError("Diagnostikakaartide automaatkontroll ebaõnnestus.")
        stage("7. Kontrollitud diagnostikakaardid ja ajaloolised analoogid")
        evidence = decision_evidence(sample, sp, values, comparison, pfi, cards, history, cfg)
        decisions = freeze_decisions(evidence, cfg)
        write_csv(out / "refinements/decision_evidence.csv", evidence)
        write_json(
            out / "refinements/frozen_decisions.json",
            {
                "decisions": decisions,
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection_outcomes_computed": False,
                "rule": "src/card_generator.py",
            },
        )
        frozen_hash = sha256(out / "refinements/frozen_decisions.json")
        # 2019. aasta tulemusi kasutatakse alles pärast valikute fikseerimist.
        log("Kandidaadid fikseeritud. Algavad 2019. aasta võrdluskatsed.")
        selection_baselines = baseline_predictions(model, selection, base_features)
        baseline_rows = []
        for period, predictions in (
            ("Diagnoosimine 2018", diagnosis_predictions),
            ("Valik 2019", selection_baselines),
        ):
            for name, p in predictions.items():
                baseline_rows.append(
                    {"period": period, "model": name, **metrics(p.target_next_hour, p.prediction)}
                )
                if period == "Valik 2019" and name == "GBR":
                    write_csv(out / f"baseline/selection_{name}.csv", p)
        trials, selected, candidate_predictions = run_refinements(train, selection, decisions, cfg)
        base_mae = metrics(selection.target_next_hour, selection_baselines["GBR"].prediction)["MAE"]
        trials["delta_MAE_vs_baseline"] = trials.MAE - base_mae
        trials["hypothesis_supported_selection"] = trials.delta_MAE_vs_baseline < 0
        selection_flags = detect_scenarios(selection, thresholds)
        selection_scenarios = []
        for i, row in trials.iterrows():
            pred = candidate_predictions[row.group]
            summary = scenario_summary(
                pred, selection_flags, thresholds["large_error"], cfg["min_scenario_n"]
            )
            selection_scenarios.append(
                summary.assign(method=row.method, case=row.case, group=row.group)
            )
            mask = (
                (selection_flags.heat | selection_flags.cold)
                if row.case == "temperature"
                else (selection_flags.rapid_increase | selection_flags.rapid_decrease)
            )
            if mask.any():
                scenario_mae = float(pred.loc[mask, "abs_error"].mean())
                baseline_scenario_mae = float(
                    selection_baselines["GBR"].loc[mask, "abs_error"].mean()
                )
                trials.loc[i, "target_scenario_n"] = int(mask.sum())
                trials.loc[i, "target_scenario_MAE"] = scenario_mae
                trials.loc[i, "target_scenario_delta_MAE"] = scenario_mae - baseline_scenario_mae
        write_csv(
            out / "refinements/selection_scenarios.csv",
            pd.concat(selection_scenarios, ignore_index=True),
        )
        write_csv(out / "refinements/all_trials.csv", trials)
        write_csv(out / "refinements/selected.csv", selected)
        manifest["selected_models"] = selected[["method", "case", "group"]].to_dict("records")
        specialist_scores, specialist_decision = develop_specialist(
            train,
            pd.concat([diagnosis, selection], ignore_index=True),
            pd.concat([dp, selection_baselines["GBR"]], ignore_index=True),
            cfg,
        )
        manifest["specialist_experiment"] = specialist_decision
        stage("8. Võrdse eelarvega tunnusekatsed; lõppvalikud fikseeritud")
        final_train = pd.concat([train, diagnosis, selection], ignore_index=True)
        final_model = fit_gbr(final_train, base_features, cfg)
        test_baselines = baseline_predictions(final_model, test, base_features)
        for name, p in test_baselines.items():
            if name == "GBR":
                write_csv(out / f"baseline/test_{name}.csv", p)
            baseline_rows.append(
                {
                    "period": "Lõpphindamine 2020",
                    "model": name,
                    **metrics(p.target_next_hour, p.prediction),
                }
            )
        baseline_table = pd.DataFrame(baseline_rows)
        write_csv(out / "baseline/metrics.csv", baseline_table)
        final_predictions, final_models = {}, {}
        test_rows, stability_rows, seed_cache = [], [], {}
        ts = detect_scenarios(test, thresholds)
        base_test_mae = metrics(test.target_next_hour, test_baselines["GBR"].prediction)["MAE"]
        for _, row in selected.iterrows():
            group = row.group
            cols = base_features + cfg["candidate_groups"][group]
            if group not in final_models:
                final_models[group] = fit_gbr(final_train, cols, cfg)
                final_predictions[group] = prediction_frame(
                    test, final_models[group].predict(test[cols]), group
                )
            pred = final_predictions[group]
            result = metrics(pred.target_next_hour, pred.prediction)
            mask = (
                (ts.heat | ts.cold)
                if row.case == "temperature"
                else (ts.rapid_increase | ts.rapid_decrease)
            )
            targeted = (
                {
                    "target_scenario_n": int(mask.sum()),
                    "target_scenario_MAE": float(pred.loc[mask, "abs_error"].mean()),
                    "target_scenario_delta_MAE": float(
                        pred.loc[mask, "abs_error"].mean()
                        - test_baselines["GBR"].loc[mask, "abs_error"].mean()
                    ),
                }
                if mask.any()
                else {}
            )
            test_rows.append(
                {
                    "method": row.method,
                    "case": row.case,
                    "group": group,
                    **result,
                    **targeted,
                    "delta_MAE_vs_baseline": result["MAE"] - base_test_mae,
                    "hypothesis_supported_test": result["MAE"] < base_test_mae,
                    "n_features": len(cols),
                    "tree_nodes": int(
                        sum(t[0].tree_.node_count for t in final_models[group].estimators_)
                    ),
                }
            )
            for seed in cfg["stability_seeds"]:
                if seed == cfg["gbr"]["random_state"]:
                    alt = candidate_predictions[group]
                else:
                    if (group, seed) not in seed_cache:
                        altmodel = fit_gbr(train, cols, cfg, seed)
                        seed_cache[group, seed] = prediction_frame(
                            selection, altmodel.predict(selection[cols]), group
                        )
                    alt = seed_cache[group, seed]
                stability_rows.append(
                    {
                        "method": row.method,
                        "case": row.case,
                        "group": group,
                        "seed": seed,
                        **metrics(alt.target_next_hour, alt.prediction),
                    }
                )
        test_table = pd.DataFrame(test_rows)
        write_csv(out / "refinements/test_metrics.csv", test_table)
        write_csv(out / "refinements/seed_stability.csv", pd.DataFrame(stability_rows))
        comparisons = []
        for case in cfg["case_studies"]:
            chosen = test_table[test_table.case == case].set_index("method").group.to_dict()
            for a, b in [
                ("XAI", "Residual"),
                ("XAI", "Manual"),
                ("Residual", "Manual"),
                ("XAI", "GBR"),
                ("Residual", "GBR"),
                ("Manual", "GBR"),
            ]:
                pa = final_predictions[chosen[a]].prediction
                pb = (
                    test_baselines["GBR"].prediction
                    if b == "GBR"
                    else final_predictions[chosen[b]].prediction
                )
                for block in cfg["bootstrap_blocks"]:
                    comparisons.append(
                        {
                            "case": case,
                            "A": a,
                            "B": b,
                            **paired_block_bootstrap(
                                test.target_next_hour,
                                pa,
                                pb,
                                block,
                                cfg["bootstrap_repeats"],
                                cfg["seed"],
                            ),
                        }
                    )
        comparison_table = pd.DataFrame(comparisons)
        write_csv(out / "evaluation/paired_block_bootstrap.csv", comparison_table)
        predictions = {"selection": candidate_predictions, "test": final_predictions}
        choices, budget_metrics, budget_intervals = budget_sensitivity(
            splits,
            decisions,
            selected,
            cfg,
            thresholds,
            predictions,
        )
        write_csv(out / "refinements/budget_choices.csv", choices)
        write_csv(out / "refinements/budget_metrics.csv", budget_metrics)
        write_csv(out / "evaluation/budget_intervals.csv", budget_intervals)
        for period, variants in predictions.items():
            for group in choices.group.unique():
                write_csv(out / f"refinements/{period}_{group}.csv", variants[group])
        specialist_metrics, specialist_predictions = evaluate_specialist(
            final_train,
            test,
            test_baselines["GBR"],
            read_json(out / "manifest.json")["specialist_experiment"],
        )
        specialist_metrics = pd.concat([specialist_scores, specialist_metrics], ignore_index=True)
        ranking = specialist_scores.groupby("model").selection_worst_ratio.first()
        specialist_metrics["selection_worst_ratio"] = specialist_metrics.model.map(ranking)
        write_csv(out / "refinements/specialist_metrics.csv", specialist_metrics)
        write_csv(out / "refinements/test_specialist.csv", specialist_predictions)
        make_figures(out, baseline_table, scenario_table, values.abs().mean(), stability)
        stage("9–10. Lõpphindamine ja teaduslik aruanne")
        manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["frozen_decision_sha256"] = frozen_hash
        manifest["artifact_sha256"] = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(out / "manifest.json", manifest)
        write_json(
            ROOT / "results/latest_run.json",
            {"path": str(out), "manifest_sha256": sha256(out / "manifest.json")},
        )
        return out
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_json(out / "manifest.json", manifest)
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elektrikoormuse XAI-uuringu täielik toru")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--attribution-only",
        action="store_true",
        help="Run the attribution repair study using the current run's output directory.",
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Freeze attribution settings without calculating final-period forecasts.",
    )
    parser.add_argument(
        "--attribution-transfer",
        action="store_true",
        help="Evaluate the frozen method on the independent Elering load series.",
    )
    args = parser.parse_args()
    configuration = read_config(args.config)
    if args.development_only and not args.attribution_only:
        parser.error("--development-only requires --attribution-only")
    if args.attribution_only or args.attribution_transfer:
        from src.common import latest_run

        run_attribution_study(
            configuration,
            args.output_dir or latest_run(),
            args.development_only,
            args.attribution_transfer,
        )
        sys.exit(0)
    output = args.output_dir or ROOT / "results/runs" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    completed = run(configuration, output)
    if configuration.get("attribution_refinement"):
        run_attribution_study(configuration, completed)
    log(str(completed))
