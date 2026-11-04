from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import save_figure
from .common import write_json
from .data_preprocessing import load_and_audit, validate_for_modeling

ROOT = Path(__file__).resolve().parents[1]
WEEKDAYS = ["E", "T", "K", "N", "R", "L", "P"]


def save_plot(fig, folder, name):
    save_figure(fig, folder / name)


def plot_01(train, folder):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    daily = train.set_index("datetime").nat_demand.resample("D").mean()
    axes[0].plot(daily.index, daily, lw=0.8)
    axes[0].set(
        title="Treeningperioodi keskmine ööpäevane koormus",
        xlabel="Panama kohalik kuupäev",
        ylabel="Elektrikoormus (CSV ühik)",
    )
    sample = train.set_index("datetime").loc["2017-06-25":"2017-07-03", "nat_demand"]
    axes[1].plot(sample.index, sample, lw=1.15)
    axes[1].set(
        title="Ebatavaline periood treeningandmetes: juuni–juuli 2017",
        xlabel="Panama kohalik aeg",
        ylabel="Elektrikoormus (CSV ühik)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    save_plot(fig, folder, "01_load_timeseries.png")


def plot_02(train, folder):
    data = train.assign(hour=train.datetime.dt.hour, day=train.datetime.dt.dayofweek)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for weekend, label in ((False, "Tööpäev"), (True, "Nädalavahetus")):
        hourly = data[(data.day >= 5) == weekend].groupby("hour").nat_demand.mean()
        axes[0].plot(hourly.index, hourly.values, label=label)
    axes[0].set(
        title="Keskmine tunnikoormus (treeningandmed)", xlabel="Kellaaeg", ylabel="Elektrikoormus"
    )
    axes[0].legend()
    weekdays = data.groupby("day").nat_demand.mean()
    axes[1].bar(range(7), weekdays.values)
    axes[1].set_xticks(range(7), WEEKDAYS)
    axes[1].set(title="Keskmine koormus nädalapäeviti", ylabel="Elektrikoormus")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    save_plot(fig, folder, "02_load_patterns.png")


def plot_03(train, folder):
    data = train.assign(hour=train.datetime.dt.hour)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    density = axes[0].hexbin(data.T2M_dav, data.nat_demand, gridsize=42, mincnt=1, cmap="Blues")
    fig.colorbar(density, ax=axes[0], label="Vaatluste arv")
    axes[0].set(
        title="Temperatuuri ja koormuse seos",
        xlabel="Temperatuur (T2M_dav)",
        ylabel="Elektrikoormus",
    )
    bins = np.unique(np.quantile(data.T2M_dav, np.linspace(0, 1, 15)))
    data["bin"] = pd.cut(data.T2M_dav, bins=bins, include_lowest=True)
    for mask, label in (
        (data.hour.between(9, 18), "Päev (09–18)"),
        (data.hour.between(0, 5), "Öö (00–05)"),
    ):
        group = (
            data[mask]
            .groupby("bin", observed=True)
            .agg(temp=("T2M_dav", "mean"), load=("nat_demand", "mean"))
        )
        axes[1].plot(group.temp, group.load, marker="o", markersize=3, label=label)
    axes[1].set(
        title="Koormus temperatuurivahemike ja kellaaja järgi",
        xlabel="Temperatuur (T2M_dav)",
        ylabel="Keskmine koormus",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    save_plot(fig, folder, "03_temperature_demand.png")


def plot_04(train, folder):
    changes = train.nat_demand.diff().dropna()
    middle = changes[changes.between(changes.quantile(0.005), changes.quantile(0.995))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].hist(middle, bins=75)
    for quantile in (0.01, 0.99):
        axes[0].axvline(changes.quantile(quantile), ls="--", color="grey")
    axes[0].set(
        title="Tunnimuutuste jaotus (äärmised 0,5% välja jäetud)",
        xlabel="Koormuse muutus ühe tunniga",
        ylabel="Vaatluste arv",
    )
    data = train.assign(hour=train.datetime.dt.hour, change=train.nat_demand.diff())
    axes[1].boxplot(
        [data.loc[data.hour == h, "change"].dropna() for h in range(24)],
        positions=list(range(24)),
        showfliers=False,
    )
    axes[1].set_xticks(range(0, 24, 3), range(0, 24, 3))
    axes[1].set(
        title="Koormuse tunnimuutused kellaaja järgi", xlabel="Kellaaeg", ylabel="Koormuse muutus"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    save_plot(fig, folder, "04_load_changes.png")


def plot_05(train, folder):
    data = train.assign(hour=train.datetime.dt.hour)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    data.boxplot("nat_demand", by="holiday", ax=axes[0], showfliers=False)
    axes[0].set(
        title="Pühade ja koormuse seos (korrigeerimata)",
        xlabel="Püha tunnus (0 = ei, 1 = jah)",
        ylabel="Elektrikoormus",
    )
    for flag, label in ((0, "Tavapäev"), (1, "Püha")):
        hourly = data[data.holiday == flag].groupby("hour").nat_demand.mean()
        axes[1].plot(
            hourly.index, hourly.values, label=f"{label} (n={sum(data.holiday == flag):,} tundi)"
        )
    axes[1].set(title="Keskmine tunniprofiil", xlabel="Kellaaeg", ylabel="Keskmine koormus")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.suptitle("Kirjeldav seos, mitte püha põhjuslik mõju", fontsize=11)
    save_plot(fig, folder, "05_holiday_effect.png")


def plot_07(train, folder):
    columns = [
        "nat_demand",
        "T2M_toc",
        "T2M_san",
        "T2M_dav",
        "QV2M_san",
        "TQL_san",
        "W2M_toc",
        "holiday",
        "school",
    ]
    correlations = train[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.7), layout="constrained")
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for (row, col), value in np.ndenumerate(correlations.to_numpy()):
        ax.text(
            col,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(value) > 0.72 else "black",
        )
    ax.set_title("Treeningandmete tunnuste Pearsoni korrelatsioonid")
    fig.colorbar(image, ax=ax, shrink=0.75)
    save_plot(fig, folder, "07_correlations.png")


def plot_08(train, folder):
    lags = np.arange(1, 169)
    values = [train.nat_demand.autocorr(lag=int(lag)) for lag in lags]
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(lags, values, lw=1.2)
    for lag in (1, 24, 168):
        ax.axvline(lag, color="grey", ls="--", lw=0.8)
    ax.set(
        title="Koormuse autokorrelatsioon (treeningandmed)",
        xlabel="Nihe tundides",
        ylabel="Autokorrelatsioon",
    )
    ax.grid(alpha=0.2)
    save_plot(fig, folder, "08_autocorrelation.png")


def plot_09(train, folder):
    data = train.assign(year=train.datetime.dt.year, month=train.datetime.dt.month)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    monthly = data.groupby("month").nat_demand.mean()
    axes[0].plot(monthly.index, monthly.values, marker="o")
    axes[0].set(
        title="Keskmine koormus kalendrikuude järgi", xlabel="Kuu", ylabel="Keskmine koormus"
    )
    axes[0].set_xticks(range(1, 13))
    for year, group in data.groupby("year"):
        values = group.groupby("month").nat_demand.mean()
        axes[1].plot(values.index, values.values, marker=".", label=str(year))
    axes[1].set(
        title="Kuude keskmised treeningaastate kaupa", xlabel="Kuu", ylabel="Keskmine koormus"
    )
    axes[1].legend()
    axes[1].set_xticks(range(1, 13))
    for ax in axes:
        ax.grid(alpha=0.2)
    save_plot(fig, folder, "09_monthly_profiles.png")


def run_eda(df, output_dir, train_end="2018-01-01"):
    from .common import write_csv

    train = df[df.datetime < train_end].copy()

    figures_dir = Path(output_dir) / "figures" / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_01(train, figures_dir)
    for plot in (plot_02, plot_03, plot_04, plot_05):
        plot(train, figures_dir)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    extreme = train.loc[train.nat_demand.idxmin(), "datetime"]
    window = train[
        train.datetime.between(extreme - pd.Timedelta(days=2), extreme + pd.Timedelta(days=2))
    ]
    ax.plot(window.datetime, window.nat_demand)
    ax.set(
        title="Treeningperioodi madalaima koormuse ajaline kontekst",
        xlabel="Panama kohalik aeg",
        ylabel="Elektrikoormus (CSV ühik)",
    )
    save_plot(fig, figures_dir, "06_extreme_events.png")
    for plot in (plot_07, plot_08, plot_09):
        plot(train, figures_dir)

    overview = train.select_dtypes("number").describe().T.reset_index(names="Tunnus")
    write_csv(Path(output_dir) / "eda/train_summary.csv", overview)
    hourly = train.groupby(train.datetime.dt.hour).nat_demand.mean()
    holiday = train.groupby("holiday").nat_demand.mean()
    ramp = train.nat_demand.diff()
    monthly = train.groupby(train.datetime.dt.month).nat_demand.mean()
    common = " Need on treeningandmete kirjeldavad seosed; põhjuslikku mõju ega testperioodi tulemust ei järeldata. Hüpoteese kontrollitakse eraldi valideerimisaastal."
    captions = {
        "01_load_timeseries.png": f"Joonis näitab koormuse ajalist arengut ainult treeningperioodil (n={len(train)}). Keskmine koormus on {train.nat_demand.mean():.2f} ja standardhälve {train.nat_demand.std():.2f} CSV ühikut. Ajaline muutumine põhjendab kronoloogilist hindamist ning minevikukoormuse tunnuseid."
        + common,
        "02_load_patterns.png": f"Joonis võrdleb tunni- ja nädalapäevamustreid. Madalaima keskmise koormuse tund on {hourly.idxmin()} ja kõrgeima tund {hourly.idxmax()}; keskmiste vahe on {hourly.max()-hourly.min():.2f} ühikut. See toetab kellaaja tsüklilise kodeerimise hüpoteesi; profiilid ei määra üksikjuhtumi prognoosi."
        + common,
        "03_temperature_demand.png": f"Joonis kirjeldab sama tunni Davidi temperatuuri ja koormuse seost: Pearsoni r={train.T2M_dav.corr(train.nat_demand):.3f}. Päeva-öö eristus osutab võimalikule ajast sõltuvale seosele. See motiveerib temperatuuri-aja interaktsiooni katset; mudelis kasutatakse kättesaadavuse tõttu viivitatud ilma, mitte joonise sama tunni ilma."
        + common,
        "04_load_changes.png": f"Joonis kirjeldab koormuse tunnimuutusi ja kellaajast sõltuvat jaotust. 5. ja 95. protsentiil on {ramp.quantile(.05):.2f} ja {ramp.quantile(.95):.2f}. Histogrammi sabade peitmine on ainult kujundusvõte, ühtegi vaatlust mudelist ei eemaldata. Hüpotees: ramp- ja muutlikkustunnused võivad aidata; SHAP ise veapõhjust ei tuvasta."
        + common,
        "05_holiday_effect.png": f"Joonis näitab püha märgendiga ja märgendita tundide võrdlust. Keskmiste koormuste erinevus (püha miinus tavapäev) on {holiday.get(1, 0)-holiday.get(0, 0):.2f} ühikut. Kellaaja, aasta ja ilma mõju pole eraldatud; püha põhjuslikku mõju ei saa hinnata. Kalendritunnuse lisamine jääb kontrollitavaks katseks."
        + common,
        "06_extreme_events.png": f"Joonis näitab treeningperioodi madalaima väärtuse konteksti: {train.nat_demand.min():.2f} ühikut hetkel {extreme}. Andmed üksi ei erista katkestust mõõtmisveast. Vaatlus säilitatakse, sest eemaldamiseks puudub tõend. XAI-d ja ajaloolisi analooge kasutatakse piirangutega diagnoosiks."
        + common,
        "07_correlations.png": f"Joonis esitab tunnuste paariskorrelatsioonid. Tocumeni ja Santiago temperatuuride r={train.T2M_toc.corr(train.T2M_san):.3f}. Seotud ilmaandmete tõttu kasutatakse linnade keskmisi ning SHAP-i kõrval rühmitatud PFI-d; sõltuvus võib panuseid tunnuste vahel ümber jagada."
        + common,
        "08_autocorrelation.png": f"Joonis näitab koormuse autokorrelatsiooni kuni 168 tunnini. Nihkel 1 on r={train.nat_demand.autocorr(1):.3f}, 24 tunnil {train.nat_demand.autocorr(24):.3f} ja 168 tunnil {train.nat_demand.autocorr(168):.3f}. Ajasõltuvus põhjendab hooajalisi võrdlusmudeleid ja plokk-bootstrap-i, kuid ei taga väljaspool valimit täpsust."
        + common,
        "09_monthly_profiles.png": f"Joonis võrdleb kalendrikuude ja aastate keskmisi. Kuukeskmiste ulatus on {monthly.min():.2f}–{monthly.max():.2f} ühikut. Osalised aastad ja trend võivad kuude võrdlust segada. Hüpotees: tsükliline kuutunnus aitab hooajalisust esitada; 2020. aasta ülekantavus vajab eraldi hindamist."
        + common,
    }
    write_json(Path(output_dir) / "eda/interpretations.json", captions)


def main():
    parser = argparse.ArgumentParser(description="Panama elektrikoormuse uuriv andmeanalüüs")
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/continuous dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    data, quality = load_and_audit(args.input)
    write_json(args.output_dir / "eda/quality.json", quality)
    validate_for_modeling(data, quality)
    run_eda(data, args.output_dir)


if __name__ == "__main__":
    main()
