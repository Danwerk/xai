"""Kontrolli koondtabeleid ja salvestatud katsetulemuste sisulist kooskõla."""

import argparse
import hashlib
import json
from os.path import relpath
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import latest_run, read_json, read_artifact_csv, sha256, manifest_core_sha256
from src.card_views import build_card_view, render_card_view, evidence_id
from src.evaluation import audit_card, seasonal_error_scale
from src.feature_engineering import load_run_features, split_features
from src.refinement_experiments import freeze_decisions, first_choices, select_specialist
from src.scenario_detector import detect_scenarios
from src.model_comparison import rank_candidates, temporal_split
from src.train_baseline import metrics, prediction_frame


def read_predictions(path):
    return read_artifact_csv(path, parse_dates=["origin_time", "target_time"])


def check_predictions(predicted, frame, scores, training=None):
    assert predicted.origin_time.equals(frame.origin_time), "Prognoosi tegemise ajad ei kattu."
    assert predicted.target_time.equals(frame.target_time), "Hindamistunnid ei kattu."
    np.testing.assert_allclose(predicted.target_next_hour, frame.target_next_hour)
    errors = predicted.prediction - predicted.target_next_hour
    np.testing.assert_allclose(predicted.error, errors)
    np.testing.assert_allclose(predicted.abs_error, errors.abs())
    measured = metrics(predicted.target_next_hour, predicted.prediction)
    for name in ("n", "MAE", "RMSE", "bias"):
        assert np.isclose(scores[name], measured[name]), f"Mõõdik {name} ei vasta prognoosidele."
    if pd.notna(scores.get("MASE")):
        scale = seasonal_error_scale(training.target_next_hour)
        assert np.isclose(scores["MASE"], measured["MAE"] / scale), "MASE skaala ei klapi."


def check_parent(run, parent):
    manifest = read_json(run / "manifest.json")
    assert Path(manifest["parent_run"]) == parent, "Uuringuosad pärinevad eri käivitustest."
    if manifest["parent_manifest_sha256"] != sha256(parent / "manifest.json"):
        if "parent_core_sha256" in manifest:
            assert manifest["parent_core_sha256"] == manifest_core_sha256(
                parent / "manifest.json"
            ), "Lähtekatse sisulised tulemused on muutunud."
        else:
            # Legacy materials predate the extra studies. Reconstruct the exact
            # original manifest, allowing only its native LF/CRLF serialization.
            core = read_json(parent / "manifest.json")
            for key in ("attribution_refinement", "attribution_transfer"):
                core.pop(key, None)
            encoded = json.dumps(core, ensure_ascii=False, indent=2, allow_nan=False)
            expected = manifest.get(
                "parent_manifest_before_attribution_extension_sha256",
                manifest["parent_manifest_sha256"],
            )
            assert expected in {
                hashlib.sha256(s.encode("utf-8")).hexdigest()
                for s in (encoded, encoded.replace("\n", "\r\n"))
            }, "Lähtekatse on muutunud."


def check_summary():
    for name in ("metrics.csv", "trials.csv"):
        table = read_artifact_csv(ROOT / "reports" / name)
        assert len(table) and np.isfinite(table[["MAE", "RMSE", "bias"]]).all().all()
        assert table.n.gt(0).all() and table.MAE.ge(0).all()
        assert table.RMSE.ge(table.MAE - 1e-10).all()
    comparisons = read_artifact_csv(ROOT / "reports/comparisons.csv")
    assert len(comparisons)
    intervals = comparisons.dropna(subset=["ci_low", "ci_high"])
    assert intervals.ci_low.le(intervals.ci_high).all()
    adjusted = intervals.dropna(subset=["adjusted_low", "adjusted_high"])
    assert adjusted.adjusted_low.le(adjusted.ci_low).all()
    assert adjusted.adjusted_high.ge(adjusted.ci_high).all()
    print("Koondtabelite mõõdikud ja vahemikud on kontrollitud.")


def check_original(run):
    config = read_json(run / "configs/experiment.json")
    parts = split_features(load_run_features(run), config)
    for left, right in zip(list(parts.values())[:-1], list(parts.values())[1:]):
        assert left.target_time.max() <= right.origin_time.min(), "Treening kasutab tulevast sihti."
    forecasts = {}
    periods = {
        "Diagnoosimine 2018": "diagnosis",
        "Valik 2019": "selection",
        "Lõpphindamine 2020": "test",
    }
    for _, row in read_artifact_csv(run / "baseline/metrics.csv").iterrows():
        period = periods[row.period]
        frame = parts[period]
        if row.model == "GBR":
            predicted = read_predictions(run / f"baseline/{period}_GBR.csv")
            forecasts[period] = predicted
        else:
            column = "load_now" if row.model == "Persistence" else "load_day"
            predicted = prediction_frame(frame, frame[column], row.model)
        check_predictions(predicted, frame, row)

    trials = read_artifact_csv(run / "refinements/all_trials.csv")
    selected = read_artifact_csv(run / "refinements/selected.csv")
    frozen = read_json(run / "refinements/frozen_decisions.json")["decisions"]
    evidence = read_artifact_csv(run / "refinements/decision_evidence.csv")
    assert freeze_decisions(evidence, config) == frozen, "Katsete valik ei vasta valikureeglile."
    assert read_json(run / "manifest.json")["selected_models"] == selected[
        ["method", "case", "group"]
    ].to_dict("records")
    for _, row in selected.iterrows():
        candidates = trials[(trials.method == row.method) & (trials.case == row.case)]
        assert (
            row.group == candidates.sort_values(["MAE", "group"]).iloc[0].group
        ), "Muudatus pole valitud 2019. aasta tulemuse järgi."
        check_predictions(
            read_predictions(run / f"refinements/selection_{row.group}.csv"),
            parts["selection"],
            row,
        )
    for _, row in read_artifact_csv(run / "refinements/test_metrics.csv").iterrows():
        check_predictions(
            read_predictions(run / f"refinements/test_{row.group}.csv"), parts["test"], row
        )

    sample = read_artifact_csv(run / "explanations/shap_input.csv", parse_dates=["target_time"])
    values = read_artifact_csv(run / "explanations/shap_values.csv", parse_dates=["target_time"])
    assert sample.target_time.equals(values.target_time), "SHAP-ridade ajad ei kattu."
    diagnosis = forecasts["diagnosis"].set_index("target_time")
    np.testing.assert_allclose(
        values.base_value + values[config["baseline_features"]].sum(axis=1),
        diagnosis.loc[values.target_time, "prediction"],
        atol=1e-5,
    )
    for path in sorted((run / "cards").glob("card_*/evidence.json")):
        card = read_json(path)
        assert audit_card(card)[
            "all_pass"
        ], f"Kaardi arvutus või ajapiir ei klapi: {path.parent.name}"
        source = diagnosis.loc[pd.Timestamp(card["target_time"])]
        np.testing.assert_allclose(
            [card["prediction"], card["actual"]], [source.prediction, source.target_next_hour]
        )
        shap_row = values.iloc[card["sample_id"]]
        assert pd.Timestamp(card["target_time"]) == shap_row.target_time
        np.testing.assert_allclose(
            [item["shap"] for item in card["shap_contributions"]],
            [shap_row[item["feature"]] for item in card["shap_contributions"]],
        )
        for role in ("user", "developer"):
            assert render_card_view(build_card_view(card, role)) == (
                path.parent / f"{role}.md"
            ).read_text(encoding="utf-8"), "Kaarditekst ei vasta lähteandmetele."
    return parts


def check_models(run):
    config = read_json(run / "configs/expanded_study.json")
    frame = load_run_features(run, expanded=True)
    performance = read_artifact_csv(run / "evaluation/performance.csv")
    for _, row in performance.iterrows():
        _, valid = temporal_split(frame, int(row.year), config["gap_hours"])
        check_predictions(
            read_predictions(run / f"predictions/{row.period}_{row.model}.csv"), valid, row
        )
    ranking = rank_candidates(read_artifact_csv(run / "selection/model_folds.csv"))
    assert read_json(run / "selection/frozen_finalists.json")["champion"] == ranking.iloc[0].model
    bundles = read_json(run / "configs/feature_bundles.json")
    for path in (run / "explanations").glob("*_shap.csv"):
        model = path.stem.removesuffix("_shap")
        values = read_artifact_csv(path).set_index("context_id")
        feature_set = (
            performance.query("period == 'confirmation' and model == @model").iloc[0].feature_set
        )
        predicted = read_predictions(run / f"predictions/confirmation_{model}.csv")
        np.testing.assert_allclose(
            values.base_value + values[bundles[feature_set]].sum(axis=1),
            predicted.prediction.iloc[values.index],
            atol=1e-5,
        )
    for table in (run / "evaluation/paired_intervals.csv", run / "evaluation/all_pairwise.csv"):
        for row in read_artifact_csv(table).itertuples():
            a = read_predictions(run / f"predictions/{row.period}_{row.A}.csv")
            b = read_predictions(run / f"predictions/{row.period}_{row.B}.csv")
            assert np.isclose(
                row.difference, (a.abs_error - b.abs_error).mean()
            ), "Mudelite paarisvahe ei klapi."


def check_budget(run, parts):
    frozen = read_json(run / "refinements/frozen_decisions.json")["decisions"]
    choices = read_artifact_csv(run / "refinements/budget_choices.csv")
    for budget, expected in (
        (1, first_choices(frozen)),
        (2, read_artifact_csv(run / "refinements/selected.csv")[["method", "case", "group"]]),
    ):
        assert (
            choices.loc[choices.budget == budget, expected.columns]
            .reset_index(drop=True)
            .equals(expected.reset_index(drop=True))
        )
    thresholds = read_json(run / "scenarios/train_thresholds.json")
    final_train = pd.concat(
        [parts[key] for key in ("train", "diagnosis", "selection")], ignore_index=True
    )
    for _, row in read_artifact_csv(run / "refinements/budget_metrics.csv").iterrows():
        period = "selection" if row.year == 2019 else "test"
        path = run / f"refinements/{period}_{row.group}.csv"
        predicted = read_predictions(path)
        check_predictions(
            predicted,
            parts[period],
            row,
            training=parts["train"] if period == "selection" else final_train,
        )
        flags = detect_scenarios(parts[period], thresholds)
        mask = (
            (flags.heat | flags.cold)
            if row.case == "temperature"
            else (flags.rapid_increase | flags.rapid_decrease)
        )
        assert row.scenario_n == mask.sum()
        assert np.isclose(
            row.scenario_MAE, predicted.loc[mask, "abs_error"].mean()
        ), "Olukorra viga ei klapi."


def check_study(run, parent):
    check_parent(run, parent)
    index = read_artifact_csv(run / "card_index.csv").set_index("case_id")
    for case_id, row in index.iterrows():
        folder = parent / "cards" / case_id
        assert row.evidence_id == evidence_id(read_json(folder / "evidence.json"))
        for role in ("user", "developer"):
            assert (run / row[f"{role}_card"]).resolve() == (folder / f"{role}.md").resolve()
    assignments = read_artifact_csv(run / "study/assignments.csv")
    for _, group in assignments.groupby(["role", "sequence"]):
        assert group.case_id.is_unique and group.layout.value_counts().to_dict() == {
            "structured": 2,
            "plain": 2,
        }
    for row in assignments.drop_duplicates("material").itertuples():
        evidence = parent / "cards" / row.case_id / "evidence.json"
        material = run / row.material
        link = Path(relpath(evidence, material.parent)).as_posix()
        expected = render_card_view(
            build_card_view(read_json(evidence), row.role), row.layout, evidence_link=link
        )
        assert (
            material.read_text(encoding="utf-8") == expected
        ), "Kasutajakatse materjal ei vasta lähteandmetele."


def check_attribution(run):
    from src.shap_explainer import repair_gain_attributions
    from src.card_generator import make_repair_audit

    manifest = read_json(run / "manifest.json")
    trials = read_artifact_csv(ROOT / "reports/trials.csv")
    comparisons = read_artifact_csv(ROOT / "reports/comparisons.csv")
    for experiment, filename in (
        ("attribution_refinement", "test_attribution.csv"),
        ("attribution_transfer", "test_attribution_estonia.csv"),
    ):
        decision = manifest.get(experiment, {})
        if decision.get("status") != "complete":
            continue
        path = run / f"refinements/{filename}"
        assert sha256(path) == decision["forecast_sha256"]
        data = read_predictions(path)
        assert data.target_time.is_unique and data.target_time.is_monotonic_increasing
        assert data.target_time.sub(data.origin_time).eq(pd.Timedelta(hours=1)).all()
        assert data.target_time.dt.year.eq(decision["settings"]["test_year"]).all()
        assert decision["interval_alignment"] == "clock_hours"
        if experiment == "attribution_transfer":
            assert decision["choices"] == manifest["attribution_refinement"]["choices"]
            assert (
                decision["feedback_choices"]
                == manifest["attribution_refinement"]["feedback_choices"]
            )
            assert decision["data_quality"]["imputed_values"] == 0
        for parameters in decision["parameters"].values():
            assert pd.Timestamp(parameters["training_target_max"]) <= data.origin_time.min()
        components = data[[c for c in data if c.startswith("correction_")]]
        np.testing.assert_allclose(
            data.prediction_GBR + components.sum(axis=1), data.prediction_credit_feedback
        )
        for loss, prefix in (("absolute", "gain_"), ("squared", "squared_gain_")):
            expected = repair_gain_attributions(
                data.target_next_hour, data.prediction_GBR, components, loss
            )
            saved = data[[prefix + c.removeprefix("correction_") for c in components]]
            np.testing.assert_allclose(saved, expected, atol=1e-6)
        make_repair_audit(data.iloc[0])
        scopes = {
            "overall": np.ones(len(data), dtype=bool),
            "rapid_increase": data.actual_ramp.ge(decision["thresholds"]["increase"]).to_numpy(),
            "rapid_decrease": data.actual_ramp.le(decision["thresholds"]["decrease"]).to_numpy(),
        }
        final = trials[(trials.experiment == experiment) & (trials.stage == "final")]
        for row in final.itertuples():
            mask = scopes[row.scope]
            measured = metrics(
                data.loc[mask, "target_next_hour"], data.loc[mask, f"prediction_{row.model}"]
            )
            for field in ("n", "MAE", "RMSE", "bias"):
                assert np.isclose(getattr(row, field), measured[field])
        for row in comparisons[comparisons.experiment == experiment].itertuples():
            mask = scopes[row.scope]
            y = data.loc[mask, "target_next_hour"]
            delta = abs(data.loc[mask, f"prediction_{row.A}"] - y) - abs(
                data.loc[mask, f"prediction_{row.B}"] - y
            )
            assert np.isclose(row.difference, delta.mean())
        print(f"{experiment}: prognoosid, ajapiirid, paranduspanused ja mõõdikud kontrollitud.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Kontrolli ka olemasolevaid kohalikke katsetulemusi. Mudeleid ei treenita.",
    )
    args = parser.parse_args()
    check_summary()
    if not args.full:
        return
    original = latest_run()
    check_attribution(original)
    parts = check_original(original)
    print("Põhikatse prognoosid, mudelivalik ja kaardid on kontrollitud.")
    if (original / "refinements/test_specialist.csv").is_file():
        scores = read_artifact_csv(original / "refinements/specialist_metrics.csv")
        chosen, _ = select_specialist(scores[scores.stage == "development"])
        saved = read_predictions(original / "refinements/test_specialist.csv")
        assert saved.model.eq(chosen).all(), "Valik ei vasta arendusaastate tulemustele."
        decision = read_json(original / "manifest.json").get("specialist_experiment", {})
        if "settings" in decision:
            assert decision["selected_variant"] == chosen, "Salvestatud valik on muutunud."
            assert decision["evaluation_kind"] == "retrospective"
            assert decision["test_period_previously_seen"] is True
            assert saved.target_time.min() >= pd.Timestamp(decision["test_start"])
        np.testing.assert_allclose(
            saved.baseline_prediction,
            read_predictions(original / "baseline/test_GBR.csv").prediction,
        )
        training = pd.concat(
            [parts[key] for key in ("train", "diagnosis", "selection")], ignore_index=True
        )
        check_predictions(
            saved,
            parts["test"],
            scores.query("stage == 'final' and selected and scope == 'overall'").iloc[0],
            training,
        )
        threshold = parts["train"].actual_ramp.quantile(0.95)
        columns = {chosen: "prediction", "GBR": "baseline_prediction"}
        for _, row in scores[scores.stage == "final"].iterrows():
            mask = (
                np.ones(len(saved), dtype=bool)
                if row.scope == "overall"
                else parts["test"].actual_ramp >= threshold
            )
            frame = parts["test"].loc[mask].reset_index(drop=True)
            predicted = prediction_frame(
                frame,
                saved.loc[mask, columns.get(row.model, f"prediction_{row.model}")].to_numpy(),
                row.model,
            )
            check_predictions(predicted, frame, row, training)
        print("Abimudeli katse: prognoosid ja mõõdikud on kontrollitud.")
    if (original / "refinements/budget_metrics.csv").is_file():
        check_budget(original, parts)
        print("Katsete arvu analüüs: kontrollitud.")
    else:
        print("Katsete arvu analüüs: selle käivituse tulemused puuduvad.")
    for pointer, label in (
        ("latest_expanded_run.json", "Mudelite võrdlus"),
        ("latest_card_study.json", "Kasutajakatse materjalid"),
    ):
        if not (ROOT / "results" / pointer).is_file():
            print(f"{label}: kohalikud tulemused puuduvad, seda osa ei kontrollitud.")
            continue
        run = latest_run(pointer)
        if pointer == "latest_expanded_run.json":
            check_models(run)
        else:
            check_study(run, original)
        print(f"{label}: kontrollitud.")


if __name__ == "__main__":
    main()
