r"""Training-only EDA for the thesis, without model fitting or output files.

Run from the repository root:
    .\.venv\Scripts\python.exe -B -m src.eda --dataset panama
    .\.venv\Scripts\python.exe -B -m src.eda --dataset estonia
    .\.venv\Scripts\python.exe -B -m src.eda --dataset all --no-show

The only file added for EDA is this module. Tables are printed and figures are
displayed in memory; nothing is saved. Source quality checks use the existing
loaders over the source data. Descriptive statistics and plots use observations
before threshold_end from experiment.json (Panama: 2018-01-01;
Estonia: 2021-01-01).
"""

import argparse
import sys

import pandas as pd

from .common import ROOT, read_config
from .data_preprocessing import load_and_audit, validate_for_modeling
from .feature_engineering import build_features, feature_columns, load_estonia_hourly


def prepare_eda(raw, cfg, dataset):
    """Describe training data; retain missing hourly slots in time statistics."""
    options = cfg["datasets"][dataset]
    cutoff = pd.Timestamp(options["threshold_end"])
    first_holdout = min([*options["development_years"], options["test_year"]])
    if cutoff > pd.Timestamp(f"{first_holdout}-01-01"):
        raise ValueError("EDA boundary must precede development and test targets.")
    training = raw.loc[raw.datetime < cutoff].sort_values("datetime").copy()
    if len(training) < 169:
        raise ValueError("EDA requires at least 169 hourly training observations.")
    features = build_features(training, cfg, dataset)
    assert features.target_time.max() < cutoff
    times = pd.DatetimeIndex(training.datetime)
    if dataset == "estonia":
        times = times.tz_localize("UTC").tz_convert("Europe/Tallinn")
    load = pd.Series(training.nat_demand.to_numpy(), index=times, name="load")
    # Never drop NaN hours before shifts: 24 rows must remain 24 real hours.
    acf = pd.Series(
        {lag: load.autocorr(lag) for lag in range(1, min(336, len(load) - 2) + 1)},
        name="autocorrelation",
    )
    ramps = features.actual_ramp
    thresholds = {"decrease": float(ramps.quantile(0.05)), "increase": float(ramps.quantile(0.95))}
    _, columns = feature_columns(dataset)
    variable_features = features[columns].loc[:, features[columns].nunique().gt(1)]
    correlations = (
        variable_features.corrwith(features.target_next_hour)
        .dropna()
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    calendar = pd.DataFrame(
        {
            "load": load.to_numpy(),
            "hour": times.hour,
            "weekday": times.dayofweek,
            "month": times.month,
        }
    )
    calendar["weekend"] = calendar.weekday.ge(5)
    overview = pd.Series(
        {
            "Algus (jaotuse ajavööndis)": str(training.datetime.min()),
            "Lõpp (jaotuse ajavööndis)": str(training.datetime.max()),
            "Treeningu lõpp, välja arvatud": str(cutoff),
            "Tunnikohti": len(load),
            "Puuduvaid koormusi": int(load.isna().sum()),
            "Kehtivaid prognoosiaknaid": len(features),
            "Keskmine koormus": load.mean(),
            "Koormuse standardhälve": load.std(),
            "Koormuse miinimum": load.min(),
            "Koormuse mediaan": load.median(),
            "Koormuse maksimum": load.max(),
            "Kiire languse lävend (P05)": thresholds["decrease"],
            "Kiire kasvu lävend (P95)": thresholds["increase"],
        },
        name="väärtus",
    )
    return {
        "dataset": dataset,
        "cutoff": cutoff,
        "load": load,
        "features": features,
        "overview": overview,
        "calendar": calendar,
        "acf": acf,
        "thresholds": thresholds,
        "feature_correlations": correlations,
    }


def plot_eda(analysis):
    """Five descriptive figures; returns Figure objects for IDE/interactive use."""
    import matplotlib.pyplot as plt

    load, features = analysis["load"], analysis["features"]
    calendar, dataset = analysis["calendar"], analysis["dataset"]
    timezone = "Tallinna kohalik aeg" if dataset == "estonia" else "Panama kohalik aeg (eeldus)"
    figures = []

    def figure(title, columns=2):
        fig, axes = plt.subplots(
            1, columns, figsize=(6 * columns, 4), layout="constrained", squeeze=False
        )
        fig.suptitle(f"{dataset}: {title} - ainult treeningandmed")
        figures.append(fig)
        return axes[0]

    axes = figure("Koormus ajas")
    daily = load.resample("D").mean()
    axes[0].plot(daily.index, daily.values, linewidth=0.8)
    axes[0].set(title="Päevakeskmine koormus", xlabel=timezone, ylabel="Koormus (lähteühik)")
    start = load.index.min().normalize()
    start += pd.DateOffset(days=(7 - start.dayofweek) % 7)
    if start < load.index.min():
        start += pd.DateOffset(weeks=1)
    if start + pd.DateOffset(weeks=1) > load.index.max():
        start = load.index.min()
    week = load[(load.index >= start) & (load.index < start + pd.DateOffset(weeks=1))]
    axes[1].plot(week.index, week.values, linewidth=0.8)
    axes[1].set(
        title="Esimene esmaspäevast algav nädal\n(lühikese rea puhul esimesed kuni 7 päeva)",
        xlabel=timezone,
        ylabel="Koormus (lähteühik)",
    )
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)

    axes = figure("Kalendrimustrid", columns=3)
    for weekend, label in ((False, "Tööpäev"), (True, "Nädalavahetus")):
        profile = (
            calendar[calendar.weekend.eq(weekend)].groupby("hour").load.mean().reindex(range(24))
        )
        axes[0].plot(profile.index, profile.values, label=label)
    axes[0].set(
        xlabel="Kohalik kellaaeg", ylabel="Keskmine koormus (lähteühik)", xticks=[0, 6, 12, 18, 23]
    )
    axes[0].legend()
    weekday = calendar.groupby("weekday").load.mean().reindex(range(7))
    axes[1].bar(range(7), weekday.values)
    axes[1].set(
        xticks=range(7),
        xticklabels=["E", "T", "K", "N", "R", "L", "P"],
        xlabel="Nädalapäev",
        ylabel="Keskmine koormus (lähteühik)",
    )
    monthly = calendar.groupby("month").load.mean().reindex(range(1, 13))
    axes[2].bar(monthly.index, monthly.values)
    axes[2].set(xticks=range(1, 13), xlabel="Kuu", ylabel="Keskmine koormus (lähteühik)")

    axes = figure("Koormuse ja tunnimuutuste jaotused")
    axes[0].hist(load.dropna(), bins=60)
    axes[0].set(
        xlabel="Koormus (lähteühik)", ylabel="Tunde", title="Kõik kehtivad treeningkoormused"
    )
    axes[1].hist(features.actual_ramp, bins=80)
    for name, label in (("decrease", "P05"), ("increase", "P95")):
        value = analysis["thresholds"][name]
        axes[1].axvline(value, linestyle="--", label=f"{label}: {value:.2f}")
    axes[1].legend()
    axes[1].set(
        xlabel="Järgmise tunni koormusmuutus (lähteühik)",
        ylabel="Prognoosiaknaid",
        title="Põhikatsega sama maski ja reegli lävendid",
    )

    axes = figure("Ajaline sõltuvus", columns=1)
    acf = analysis["acf"]
    axes[0].plot(acf.index, acf.values)
    for lag in (24, 168):
        if lag in acf.index:
            axes[0].axvline(lag, linestyle="--", label=f"{lag} h: {acf.loc[lag]:.3f}")
    axes[0].axhline(0, color="grey", linewidth=0.7)
    axes[0].set(
        xlabel="Viide tegelikes tundides", ylabel="Koormuse autokorrelatsioon", ylim=(-1, 1)
    )
    axes[0].legend()

    axes = figure("Prognoosihetkel teadaoleva info seosed sihtkoormusega")
    correlations = analysis["feature_correlations"].head(12).sort_values()
    axes[0].barh(correlations.index, correlations.values)
    axes[0].set(
        xlabel="Pearsoni korrelatsioon", xlim=(-1, 1), title="12 tugevamat seost; tunnuste koodid"
    )
    if "temperature" in features:
        density = axes[1].hexbin(
            features.temperature, features.target_next_hour, gridsize=35, mincnt=1
        )
        axes[1].figure.colorbar(density, ax=axes[1], label="Prognoosiaknaid")
        axes[1].set(
            xlabel="Viivitatud temperatuur (lähteühik)",
            ylabel="Järgmise tunni koormus (lähteühik)",
            title="Ilmainfo, mida mudel päriselt saab",
        )
    else:
        density = axes[1].hexbin(
            features.load_week, features.target_next_hour, gridsize=35, mincnt=1
        )
        axes[1].figure.colorbar(density, ax=axes[1], label="Prognoosiaknaid")
        axes[1].set(
            xlabel="Eelmise nädala sihttunni koormus",
            ylabel="Järgmise tunni koormus (lähteühik)",
            title="Nädalaviite seos sihtkoormusega",
        )
    for fig in figures:
        for axis in fig.axes:
            axis.grid(alpha=0.15)
    return figures


def run_eda(cfg, dataset="panama", show=True):
    """Print data checks/tables and draw EDA; never writes files or trains models."""
    path = ROOT / cfg["datasets"][dataset]["path"]
    if dataset == "panama":
        raw, quality = load_and_audit(path)
        validate_for_modeling(raw, quality)
        quality_keys = [
            "rows",
            "first",
            "last",
            "missing_hours",
            "duplicate_timestamps",
            "nonfinite",
            "nonpositive_load",
            "load_below_200",
        ]
    else:
        raw, quality = load_estonia_hourly(path, cfg["datasets"][dataset]["test_year"])
        quality_keys = [
            "source_rows",
            "hourly_intervals",
            "missing_hourly_load",
            "nonpositive_load_unavailable",
            "imputed_values",
        ]
    analysis = prepare_eda(raw, cfg, dataset)
    print(f"\n{dataset.upper()}: algandmete kvaliteedikontroll (võib hõlmata hilisemaid aastaid)")
    print(pd.Series({key: quality[key] for key in quality_keys}).to_string())
    print("\nEDA: ainult alltoodud piirist varasemad treeningvaatlused")
    print(analysis["overview"].to_string())
    print("\nKoormuse autokorrelatsioon (kirjeldav, ilma iid-usalduspiirideta)")
    print(analysis["acf"].reindex([1, 24, 168]).to_string())
    print("\nTunnuste Pearsoni seosed järgmise tunni koormusega")
    print(analysis["feature_correlations"].round(4).to_string())
    print("\nPuuduvaid tunde ei täideta; ebatavalised positiivsed koormused säilivad.")
    print("Seosed on kirjeldavad ega tõenda põhjuslikkust või tunnuse prognoosikasu.")
    if dataset == "panama":
        print(
            "Panama aja- ja ühikueeldused ning ilma tegelik avaldamisviivitus vajavad allikapõhist kinnitust."
        )
    analysis["figures"] = plot_eda(analysis)
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("panama", "estonia", "all"), default="panama")
    parser.add_argument(
        "--no-show", action="store_true", help="Arvuta joonised ilma graafikaaknaid avamata"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    cfg = read_config()
    for dataset in (("panama", "estonia") if args.dataset == "all" else (args.dataset,)):
        run_eda(cfg, dataset, show=not args.no_show)
    if args.no_show:
        import matplotlib.pyplot as plt

        plt.close("all")
