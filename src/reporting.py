"""A single data-derived thesis report, with no model fitting or selection."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    ROOT,
    read_artifact_csv,
    read_json,
    read_run_config,
    sha256,
    write_csv,
    write_json,
)
from .evaluation import METHODS, PRIMARY


def markdown_table(frame):
    view = frame.copy()
    for c in view:
        view[c] = view[c].map(lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
    return (
        "| "
        + " | ".join(view.columns)
        + " |\n| "
        + " | ".join(["---"] * len(view.columns))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in view.astype(str).values)
    )


def result_figure(data, dataset, path):
    overall = data[data.scope.eq("overall")].set_index("method").reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    labels = list(METHODS)
    positions = list(range(len(labels)))
    axes[0].barh(
        positions, overall.MAE, color=["#176b67" if n == PRIMARY else "#7a99b8" for n in labels]
    )
    axes[0].set_yticks(positions, labels=labels)
    axes[0].tick_params(axis="y", labelleft=True)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="MAE (source load units)", title=f"{dataset}: common evaluation hours")
    axes[1].barh(positions, overall.harmful_magnitude_vs_base, color="#a77f60")
    for position, (_, row) in enumerate(overall.iterrows()):
        axes[1].annotate(
            f"{row.harmful_rate_vs_base * 100:.1f}% harmful hours",
            (row.harmful_magnitude_vs_base, position),
            fontsize=8,
            va="center",
            xytext=(4, 0),
            textcoords="offset points",
        )
    axes[1].set(
        xlabel="Mean positive loss increase (all hours)",
        title="Correction harm vs base",
    )
    axes[1].set_xlim(0, max(overall.harmful_magnitude_vs_base.max() * 1.65, 1))
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def robustness_sections(overall):
    columns = [
        "method",
        "n",
        "MAE",
        "RMSE",
        "improvement_vs_residual_pct",
        "harmful_rate_vs_base",
        "harmful_magnitude_vs_base",
    ]
    sensitivity = overall[overall.method.str.startswith("sensitivity_")]
    measured = overall.set_index("method")
    return [
        "### Credit definition and memory aging",
        markdown_table(
            overall[
                overall.method.isin(["base", "residual", PRIMARY, "mae_credit", "recency_credit"])
            ][columns]
        ),
        "### Window and feedback-strength sensitivity",
        markdown_table(sensitivity[columns]),
        f"MAE-credit minus primary MAE: {measured.loc['mae_credit', 'MAE'] - measured.loc[PRIMARY, 'MAE']:+.4f}; "
        f"recency minus uniform MAE: {measured.loc['recency_credit', 'MAE'] - measured.loc[PRIMARY, 'MAE']:+.4f} "
        f"(negative means lower error). Sensitivity MAEs span {sensitivity.MAE.min():.4f} to {sensitivity.MAE.max():.4f}. "
        "These are descriptive checks, without test-based parameter selection or significance claims. "
        "Harm rate is a fraction; magnitude averages positive absolute-error increase vs base over all hours.",
    ]


def build_report(run, destination=None):
    run = Path(run)
    destination = Path(destination or ROOT / "reports")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run / "manifest.json")
    cfg = read_run_config(manifest)
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs can be reported.")
    selection = read_json(run / "selection.json")
    parts = [
        "# Realised-credit-based adaptive residual correction",
        "Research question: does realised-credit-based component weighting improve residual correction over fixed scaling and simpler adaptation? The mechanism question compares interaction-aware credit with standalone component benefit.",
        "This is a retrospective reanalysis of previously inspected Panama 2020 and Estonia 2024 periods. Cleanup and new comparisons do not create a new blind test. All choices are frozen using Panama 2018–2019 before final-year models are fitted; Estonia receives the same configuration and policy choices without local tuning.",
        "## Method and comparisons",
        "Base GBR → OOF-trained HistGBR correction → grouped TreeSHAP contributions → past realised credit → bounded component weights → forecast → feedback after the actual value arrives. See the [method and protocol in README](../README.md#metoodika) and the matched ablations below; detailed equations belong to the thesis document.",
        markdown_table(
            pd.DataFrame([{"Method": key, "Role": value} for key, value in METHODS.items()])
        ),
        "The direct HistGBR predicts load using the same OOF training rows, features and parameters as the residual HistGBR. Only the target (load vs residual) and additive base anchor differ. B2–B6 and all ablations share one fitted base/residual model per fold.",
        "## Frozen development choices",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "Family": name,
                        "Candidate": choice["candidate"],
                        "Setting": str(choice["setting"]),
                        "Development ratio": choice["development_ratio"],
                    }
                    for name, choice in selection["choices"].items()
                ]
            )
        ),
        "Identity means development selected the unchanged full correction. Identical predictions are reported as a selection outcome, not independent evidence. The three matched ablations use B6's settings without re-tuning.",
        "## Credit, memory and parameter robustness",
        "The same study also evaluates MAE-credit, recency-weighted memory and window/strength sensitivity, using the SAME fitted models, components and timestamps. MAE-credit changes only squared to absolute realised game loss. Memory aging weights past credits by `2 ** (-age_hours / half_life_hours)` inside the same finite window; the numerator and denominator use the same decay. Cold start and hour conditioning remain matched to B6.",
        f"Development compared half-lives {cfg['feedback_checks']['half_lives_hours']} hours and uniform fallback, using B6's MAE-ratio objective/scope guard; uniform wins ties. Frozen memory: **{selection['decay']['memory']}**, half-life **{selection['decay']['half_life_hours']}** hours. Estonia receives the same selection without local tuning.",
        f"Every combination of windows {cfg['feedback_checks']['windows_hours']} hours and strengths {cfg['feedback_checks']['strengths']} is reported with squared credit, uniform memory and B6 hour context. No sensitivity winner is selected on the final periods. A row matching B6 is a duplicate control. If B6 is identity, matched loss/memory ablations remain identities; active sensitivity rows then use predeclared same-hour context.",
        "All primary and robustness choices are frozen together before final-year models are fitted. The added robustness rows are descriptive; paired bootstrap intervals retain the original nine baseline/ablation comparisons and their multiplicity correction.",
    ]
    metrics, comparisons, trials = [], [], []
    for dataset, record in manifest["datasets"].items():
        if record["evaluation_status"] != "complete":
            continue
        table = read_artifact_csv(run / dataset / "metrics.csv")
        intervals = read_artifact_csv(run / dataset / "comparisons.csv")
        metrics.append(table.assign(dataset=dataset))
        comparisons.append(intervals.assign(dataset=dataset))
        if (run / dataset / "development.csv").exists():
            trials.append(
                read_artifact_csv(run / dataset / "development.csv").assign(dataset=dataset)
            )
        quality = record["quality"]
        parts += [
            f"## {dataset}: {record['test_first']} to {record['test_last']}",
            f"Scored {quality['test_forecasts']} of {quality['calendar_test_hours']} calendar hours; unscored: {quality['unscored_test_hours']}. Every method uses these same timestamps. Hours beyond the end of the source series are also unscored; missing values are never interpolated.",
        ]
        overall = table[table.scope.eq("overall")]
        main_overall = overall[overall.method.isin(METHODS)]
        parts.append(
            markdown_table(
                main_overall[
                    [
                        "method",
                        "n",
                        "MAE",
                        "RMSE",
                        "improvement_vs_base_pct",
                        "improvement_vs_residual_pct",
                    ]
                ]
            )
        )
        parts.append(
            markdown_table(
                main_overall[
                    [
                        "method",
                        "harmful_rate_vs_base",
                        "harmful_magnitude_vs_base",
                        "harmful_rate_vs_residual",
                        "harmful_magnitude_vs_residual",
                    ]
                ]
            )
        )
        primary = overall.set_index("method").loc[PRIMARY]
        parts.append(
            f"Proposed MAE improvement vs standard residual correction: {primary.improvement_vs_residual_pct:.3f}%. Harmful-correction rates are fractions; harm magnitude is the mean positive increase in absolute error over ALL scored hours. Lower MAE does not imply fewer or less severe harmful corrections."
        )
        parts.append(
            markdown_table(
                table[
                    table.method.isin(
                        [
                            "residual",
                            PRIMARY,
                            "standalone_feedback",
                            "matched_standalone",
                            "no_context",
                        ]
                    )
                ][["method", "scope", "n", "MAE", "RMSE"]]
            )
        )
        block = cfg["evaluation"]["primary_block_hours"]
        parts.append(
            markdown_table(
                intervals[intervals.block_hours.eq(block)][
                    ["B", "scope", "difference", "adjusted_low", "adjusted_high"]
                ]
            )
        )
        parts.append(
            "Differences are proposed minus reference MAE; negative favors proposed. Circular paired clock-hour block bootstrap uses masked unscored hours. Bonferroni covers all declared method/scope comparisons within a dataset at each block length; the second block length is a sensitivity analysis. It does not correct historical research choices or prove practical importance."
        )
        figure = f"results_{dataset}.png"
        result_figure(table, dataset, destination / figure)
        parts.append(f"![{dataset} accuracy and correction harm]({figure})")
        parts.extend(robustness_sections(overall))
    parts += [
        "## Limits and reproducibility",
        "The v2 protocol removes unused four-week feature eligibility restrictions, raw/SHAP Ridge repair layers and legacy model searches. It uses true clock-hour feedback windows and a shared 72-hour cold start. Consequently these are newly computed v2 results, not a relabelling of the former study.",
        "Input availability assumes immediate completed-hour load/error observation. Panama weather is delayed reanalysis, not archived forecasts; its actual publication delay is unverified. Panama timestamp/units assumptions remain source limitations. Estonia uses UTC splits and Europe/Tallinn calendar features; absent weather and holiday labels are omitted. Source load units are never compared between countries.",
        "Grouped TreeSHAP terms reconstruct a correction but are not independently trained experts or causal effects. Squared-loss potential credits drive feedback; changing input groups may change the policy. No novelty priority or human usability claim is made.",
        f"Run: `{run.name}`. Settings are read only from [experiment.json](../configs/experiment.json). Source/config/data hashes, package versions, quality checks, training boundaries and frozen policy choices are in [run.json](run.json), without a configuration copy. Git revision at execution: `{manifest['git']['revision']}`; dirty working tree: `{manifest['git']['dirty']}`.",
        "Reproduce with the commands in [README](../README.md). Old exploratory code and results are available in Git history at `1220640`; no archive directory is used.",
    ]
    if not metrics:
        raise ValueError("No evaluated datasets in this run.")
    write_csv(destination / "metrics.csv", pd.concat(metrics, ignore_index=True))
    write_csv(destination / "comparisons.csv", pd.concat(comparisons, ignore_index=True))
    if trials:
        write_csv(destination / "trials.csv", pd.concat(trials, ignore_index=True))
    else:
        write_csv(
            destination / "trials.csv",
            pd.DataFrame(columns=["dataset", "method", "stage", "scope", "year", "MAE", "RMSE"]),
        )
    write_json(
        destination / "run.json",
        {
            "run": run.name,
            "manifest_sha256": sha256(run / "manifest.json"),
            "protocol": manifest["protocol"],
            "config_ref": manifest["config_ref"],
            "selection": {
                key: value
                for key, value in selection.items()
                if key not in ("config_ref", "source_sha256")
            },
            "source_sha256": manifest["source_sha256"],
            "git": manifest["git"],
            "datasets": manifest["datasets"],
            "packages": manifest["packages"],
            "python": manifest.get("python"),
            "platform": manifest.get("platform"),
        },
    )
    (destination / "thesis_results.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
