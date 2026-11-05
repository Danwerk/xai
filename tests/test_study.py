from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.common import (
    config_reference,
    read_artifact_csv,
    read_json,
    read_run_config,
    sha256,
    write_csv,
    write_json,
)
from src.evaluation import METHODS
from src.feature_engineering import feature_columns
from src.study import run_study, select_policies, validate_config, validate_frozen, variant_specs


def test_selection_uses_complete_development_only_and_identity_on_ties(cfg):
    rows = []
    for year in (2018, 2019):
        for scope in ("overall", "rapid_increase", "rapid_decrease"):
            for name, mae in (("residual", 10), ("better", 9), ("tie", 10)):
                rows.append(dict(year=year, scope=scope, method=name, MAE=mae, stage="development"))
    scores = pd.DataFrame(rows)
    options = {"one": {"better": {"weight": 0.5}}, "two": {"tie": {"weight": 1}}}
    choices = select_policies(scores, options, cfg)
    assert choices["one"]["setting"] == {"weight": 0.5}
    assert choices["two"]["candidate"] == "identity"
    with pytest.raises(ValueError):
        select_policies(scores.assign(stage="final"), options, cfg)
    with pytest.raises(ValueError):
        select_policies(scores.drop(scores.index[-2]), options, cfg)


def test_frozen_selection_rejects_parameter_and_source_drift(cfg):
    hashes = {"source": "123"}
    frozen = {"status": "frozen", "config_ref": config_reference(cfg), "source_sha256": hashes}
    validate_frozen(frozen, cfg, hashes)
    with pytest.raises(ValueError):
        validate_frozen(frozen, cfg, {"source": "changed"})
    cfg["residual_model"]["max_iter"] += 1
    with pytest.raises(ValueError):
        validate_frozen(frozen, cfg, hashes)


def test_protocol_rejects_test_selection_and_random_early_stopping(cfg):
    cfg["datasets"]["panama"]["development_years"].append(2020)
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg["datasets"]["panama"]["development_years"].pop()
    cfg["residual_model"]["early_stopping"] = True
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_single_config_reference_rejects_changed_settings(cfg, monkeypatch):
    import src.common as common

    record = {"config_ref": config_reference(cfg)}
    assert read_run_config(record) == cfg
    assert set(record["config_ref"]) == {"path", "sha256"}
    changed = deepcopy(cfg)
    changed["feedback_checks"]["strengths"] = [0.1]
    monkeypatch.setattr(common, "read_config", lambda: changed)
    with pytest.raises(ValueError, match="settings differ"):
        read_run_config(record)


def test_duplicate_cleanup_keeps_unique_published_and_incomplete_results(tmp_path, monkeypatch):
    import src.check_results as checker

    monkeypatch.setattr(checker, "ROOT", tmp_path)

    def result(name, value):
        folder = tmp_path / "results/thesis_runs" / name
        write_csv(folder / "metrics.csv", pd.DataFrame({"MAE": [value]}))
        manifest = {
            "status": "complete",
            "artifact_sha256": {"metrics.csv": sha256(folder / "metrics.csv")},
        }
        write_json(folder / "manifest.json", manifest)
        return folder

    old = result("01_old", 1)
    latest = result("02_latest", 1)
    unique = result("03_unique", 2)
    published = result("00_published", 1)
    incomplete = result("04_incomplete", 1)
    manifest = read_json(incomplete / "manifest.json")
    manifest["status"] = "running"
    write_json(incomplete / "manifest.json", manifest)
    changed = result("05_changed", 1)
    write_csv(changed / "metrics.csv", pd.DataFrame({"MAE": [99]}))
    write_json(tmp_path / "results/latest_thesis_run.json", {"path": str(latest)})
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/thesis_results.md").write_text(str(published), encoding="utf-8")
    assert checker.prune_duplicate_runs() == {old}
    assert checker.prune_duplicate_runs() == set()
    assert latest.exists() and unique.exists()
    assert published.exists() and incomplete.exists() and changed.exists()


def test_full_pipeline_freezes_before_final_and_reconstructs_artifacts(monkeypatch, tmp_path, cfg):
    import src.common as common
    import src.study as study
    import src.check_results as checker
    from src.reporting import build_report

    rng = np.random.default_rng(4)
    _, columns = feature_columns("panama")
    parts = []
    for year in range(2015, 2025):
        target = pd.date_range(f"{year}-01-01", periods=96, freq="h")
        frame = pd.DataFrame(rng.normal(size=(96, len(columns))), columns=columns)
        frame["load_now"] = 100 + 10 * rng.normal(size=96)
        frame["target_next_hour"] = frame.load_now + 5 * rng.normal(size=96)
        frame["origin_time"], frame["target_time"], frame["hour"] = (
            target - pd.Timedelta(hours=1),
            target,
            target.hour,
        )
        frame["actual_ramp"] = frame.target_next_hour - frame.load_now
        parts.append(frame)
    all_features = pd.concat(parts, ignore_index=True)
    for name in ("panama", "estonia"):
        raw_path = tmp_path / f"{name}.csv"
        raw_path.write_text("raw test input", encoding="utf-8")
        cfg["datasets"][name]["path"] = str(raw_path)
    cfg["base_model"].update(n_estimators=3, min_samples_leaf=5)
    cfg["residual_model"].update(max_iter=4, min_samples_leaf=5)
    cfg["selection"].update(feedback_windows_hours=[168], feedback_strengths=[1.0])
    cfg["evaluation"].update(
        bootstrap_repeats=20, bootstrap_blocks_hours=[24], primary_block_hours=24
    )
    cfg["workers"] = 1
    # Simulate the single settings source without writing to the user's config.
    monkeypatch.setattr(common, "read_config", lambda: cfg)
    output = tmp_path / "run"

    def load(path, config, dataset):
        f = all_features[
            all_features.target_time.dt.year <= config["datasets"][dataset]["test_year"]
        ].copy()
        n = int(f.target_time.dt.year.eq(config["datasets"][dataset]["test_year"]).sum())
        return f, {"test_forecasts": n, "calendar_test_hours": n, "unscored_test_hours": 0}

    monkeypatch.setattr(study, "load_dataset", load)
    monkeypatch.setattr(checker, "load_dataset", load)
    real_base = study.base_forecast
    real_residual = study.residual_forecasts
    residual_years = []

    def residual_spy(train, valid, cols, config):
        year = int(valid.target_time.dt.year.iloc[0])
        residual_years.append(year)
        if year in (2020, 2024):
            assert "decay" in read_json(output / "selection.json")
        return real_residual(train, valid, cols, config)

    monkeypatch.setattr(study, "residual_forecasts", residual_spy)

    def spy(train, valid, cols, config):
        if (
            valid.target_time.dt.year.iloc[0] in (2020, 2024)
            and valid.target_time.dt.year.iloc[0] == config["datasets"]["estonia"]["test_year"]
        ):
            assert (output / "selection.json").is_file()
            assert "decay" in read_json(output / "selection.json")
        if (
            valid.target_time.dt.year.iloc[0] == 2020
            and train.target_time.dt.year.min() == 2015
            and not (output / "panama/predictions.csv").exists()
        ):
            # Estonia's OOF 2020 also comes after the original freeze.
            assert (output / "selection.json").is_file()
        return real_base(train, valid, cols, config)

    monkeypatch.setattr(study, "base_forecast", spy)
    # Do not move the user's latest-run pointer during a unit test.
    real_write = study.write_json

    def write(path, value):
        if str(path).endswith("latest_thesis_run.json"):
            return
        real_write(path, value)

    monkeypatch.setattr(study, "write_json", write)
    run_study(cfg, output)
    assert residual_years == [2018, 2019, 2020, 2024]
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["test_periods_previously_seen"] is True
    assert "config" not in manifest
    assert manifest["config_ref"] == config_reference(cfg)
    assert not (output / "config.json").exists()
    for dataset in ("panama", "estonia"):
        metrics = read_artifact_csv(output / dataset / "metrics.csv")
        frozen = read_json(output / "selection.json")
        specs = variant_specs(frozen["choices"]["credit_feedback"]["setting"], cfg)
        assert set(metrics.method) == set(METHODS) | set(specs)
        assert metrics[metrics.scope.eq("overall")].n.nunique() == 1
        assert {"mae_credit", "recency_credit"}.issubset(metrics.method)
    checker.check_run(output, full=True)
    build_report(output, tmp_path / "report")
    assert (tmp_path / "report/thesis_results.md").is_file()
    report = (tmp_path / "report/thesis_results.md").read_text(encoding="utf-8")
    assert "Credit definition and memory aging" in report
    assert "Window and feedback-strength sensitivity" in report
    assert not (tmp_path / "report/feedback_checks.md").exists()
    public_metrics = read_artifact_csv(tmp_path / "report/metrics.csv")
    assert set(public_metrics.method) == set(metrics.method)
    public_run = read_json(tmp_path / "report/run.json")
    assert "config" not in public_run
    assert "config" not in read_json(output / "selection.json")
    assert public_run["selection"]["choices"] == read_json(output / "selection.json")["choices"]
    assert not (tmp_path / "report/selection.json").exists()
    with pytest.raises(ValueError):
        run_study(cfg, output)
