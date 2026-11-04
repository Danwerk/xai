import copy
from os.path import relpath
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.card_views import build_card_view, evidence_id, render_card_view, save_role_cards
from src.evaluation import paired_block_bootstrap, paired_intervals
from src.refinement_experiments import first_choices
from src.card_generator import select_card_cases
from src.labels import feature_label
from tools.build_card_study import make_study_materials


@pytest.fixture
def card():
    return {
        "origin_time": "2018-01-01 00:00:00",
        "target_time": "2018-01-01 01:00:00",
        "prediction": 12.0,
        "actual": 10.0,
        "error": 2.0,
        "absolute_error": 2.0,
        "base_value": 10.0,
        "scenarios": ["Kiire kasv (tagasivaateline)"],
        "selection_reason": "Kiire kasvu suurima veaga näide selgituste valimis.",
        "observed_patterns": {"known_ramp": 2.0, "realized_ramp": 4.0, "known_temperature": 25.0},
        "shap_contributions": [
            {"feature": "load_now", "value": 10.0, "shap": 3.0},
            {"feature": "temperature", "value": 25.0, "shap": -2.0},
            {"feature": "load_day", "value": 8.0, "shap": 0.7},
            {"feature": "load_week", "value": 8.0, "shap": 0.3},
        ],
        "historical_cases": [],
        "diagnosis": {"hypothesis": None, "experiment": None},
    }


def test_two_roles_keep_the_same_facts_and_abstention(card):
    before = copy.deepcopy(card)
    user = build_card_view(card, "user")
    developer = build_card_view(card, "developer")
    assert card == before
    assert user["common"] == developer["common"]
    assert user["evidence_id"] == developer["evidence_id"] == evidence_id(card)
    assert not user["common"]["hypothesis_available"]
    assert "ei soovitata" in render_card_view(user)
    assert card["selection_reason"] in render_card_view(user)
    assert card["selection_reason"] in render_card_view(developer)
    assert "ei näita prognoosi usaldusvahemikku" in render_card_view(user)


def test_short_view_retains_remainder_and_developer_all_inputs(card):
    user = render_card_view(build_card_view(card, "user"))
    developer = render_card_view(build_card_view(card, "developer"))
    assert "Ülejäänud tunnused kokku" in user and "0,300" in user
    assert "Tunnus `load_week`" in developer
    assert "Tunnus `" not in user


@pytest.mark.parametrize("role", ["user", "developer"])
def test_control_presents_exactly_the_same_facts(card, role):
    view = build_card_view(card, role)
    structured = render_card_view(view)
    plain = render_card_view(view, "plain")
    facts = [fact for section in view["sections"] for fact in section["facts"]]
    assert [line for line in structured.splitlines() if line.startswith("**")] == [
        f"**{fact['label']}.** {fact['text']}" for fact in facts
    ]
    assert [line for line in plain.splitlines() if line.startswith("- ")] == [
        f"- {fact['label']}: {fact['text']}" for fact in facts
    ]


def test_hypothesis_remains_uncertain_in_both_roles(card):
    card["diagnosis"] = {
        "hypothesis": "Muutuse tunnus võib aidata.",
        "experiment": "ramp",
        "cohort_n": 60,
        "MAE_minus_ordinary": 3.0,
    }
    for role in ("user", "developer"):
        view = build_card_view(card, role)
        assert view["common"]["hypothesis_available"]
        text = render_card_view(view)
        assert "ei tõenda vea põhjust" in text
        assert "kogu 2018. aasta andmetel" in text
    assert "Kontrollkatse" in render_card_view(build_card_view(card, "developer"))


def test_study_reuses_cards_and_only_saves_assigned_controls(card, tmp_path):
    output = tmp_path / "study"
    output.mkdir()
    index = []
    for i in range(12):
        case = copy.deepcopy(card)
        if i < 2:
            case["diagnosis"] = {
                "hypothesis": "Muutuse tunnus võib aidata.",
                "experiment": "ramp",
                "cohort_n": 60,
                "MAE_minus_ordinary": 3.0,
            }
        folder = tmp_path / "run/cards" / f"card_{i:02d}"
        save_role_cards(case, folder)
        assert {p.name for p in folder.iterdir()} == {"evidence.json", "user.md", "developer.md"}
        source = Path(relpath(folder, output)).as_posix()
        index.append(
            {
                "case_id": folder.name,
                "has_hypothesis": i < 2,
                "user_card": f"{source}/user.md",
                "developer_card": f"{source}/developer.md",
            }
        )
    make_study_materials(output, pd.DataFrame(index))
    assignments = pd.read_csv(output / "study/assignments.csv")
    controls = {
        output / name for name in assignments.loc[assignments.layout == "plain", "material"]
    }
    assert len(controls) == 8
    assert controls == set((output / "cards").rglob("*.*"))
    for name in assignments.material.unique():
        material = output / name
        assert material.is_file()
        link = material.read_text(encoding="utf-8").split("](", 1)[1].split(")", 1)[0]
        assert (material.parent / link).is_file()


def test_budget_one_uses_order_not_later_scores():
    decisions = {
        "XAI": {"rapid_change": ["ramp", "rolling"]},
        "Residual": {"rapid_change": ["rolling", "ramp"]},
    }
    choices = first_choices(decisions).set_index("method")
    assert choices.loc["XAI", "group"] == "ramp"
    assert choices.loc["Residual", "group"] == "rolling"


def test_repair_card_distinguishes_forecast_direction_from_gain():
    from src.card_generator import make_repair_audit
    from src.card_views import render_repair_audit

    row = pd.Series(
        {
            "target_time": "2024-01-01",
            "target_next_hour": 10.0,
            "prediction_GBR": 20.0,
            "prediction_credit_feedback": 14.0,
            "correction_clock": 3.0,
            "correction_offset": -9.0,
            "gain_clock": -3.0,
            "gain_offset": 9.0,
        }
    )
    audit = make_repair_audit(row)
    assert audit["gain"] == 6
    assert "Vea vähenemise Shapley-jaotus" in render_repair_audit(audit)
    assert "ei tõenda algse vea põhjust" in render_repair_audit(audit)
    row["gain_clock"] = 100.0
    with pytest.raises(ValueError, match="credits"):
        make_repair_audit(row)


@pytest.mark.parametrize("core_recorded", [True, False])
def test_study_parent_allows_only_xai_metadata_extensions(tmp_path, core_recorded):
    from src.common import write_json, sha256, manifest_core_sha256
    from tools.check_results import check_parent

    parent, child = tmp_path / "run", tmp_path / "study"
    original = {
        "status": "complete",
        "raw_sha256": "unchanged",
        "artifact_sha256": {"card": "same"},
    }
    write_json(parent / "manifest.json", original)
    study = {"parent_run": str(parent), "parent_manifest_sha256": sha256(parent / "manifest.json")}
    if core_recorded:
        study["parent_core_sha256"] = manifest_core_sha256(parent / "manifest.json")
    write_json(child / "manifest.json", study)
    write_json(
        parent / "manifest.json", {**original, "attribution_refinement": {"status": "complete"}}
    )
    check_parent(child, parent)
    write_json(parent / "manifest.json", {**original, "raw_sha256": "changed"})
    with pytest.raises(AssertionError):
        check_parent(child, parent)


def test_user_clock_label_avoids_encoding_jargon():
    assert "koosinus" not in feature_label("hour_cos", simple=True)
    assert "kellaaeg" in feature_label("hour_cos", simple=True)


def test_shared_bootstrap_preserves_original_intervals():
    actual = np.zeros(480)
    predicted = np.repeat([1, 4, 2, 7], 120)
    old_api = paired_block_bootstrap(actual, predicted, actual, 24, 100)
    new_api = paired_intervals(actual, predicted, actual, 24, 100)
    assert old_api["MAE_difference_A_minus_B"] == new_api["difference"]
    assert old_api["ci_low"] == new_api["ci_low"]
    assert old_api["ci_high"] == new_api["ci_high"]


def test_card_selection_keeps_small_error_and_scenario_coverage():
    predictions = pd.DataFrame({"abs_error": [8, 1, 10, 5]})
    flags = pd.DataFrame(
        False,
        index=predictions.index,
        columns=["peak", "rapid_increase", "rapid_decrease", "heat", "cold", "holiday"],
    )
    flags.loc[3, "heat"] = True
    cases = select_card_cases(predictions, flags, 4)
    assert list(cases) == [0, 1, 3, 2]
    assert "väikseima veaga" in cases[1]
    assert "kuumus" in cases[3]
    assert "suurima veaga" in cases[2]
    # The first matching rule is retained when the same row covers two situations.
    flags.loc[3, "peak"] = True
    assert "Tipukoormus" in select_card_cases(predictions, flags, 4)[3]
