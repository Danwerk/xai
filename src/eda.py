from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEATHER = [f'{name}_{city}' for name in ('T2M', 'QV2M', 'TQL', 'W2M')
           for city in ('toc', 'san', 'dav')]
REQUIRED = set(WEATHER + ['datetime', 'nat_demand', 'holiday', 'school', 'Holiday_ID'])
WEEKDAYS = ['E', 'T', 'K', 'N', 'R', 'L', 'P']


def load_and_validate(path):
    df = pd.read_csv(path)
    missing_cols = REQUIRED - set(df.columns)
    if missing_cols:
        raise ValueError(f'Puuduvad veerud: {missing_cols}')

    df['datetime'] = pd.to_datetime(df['datetime'])
    sorted_before = df.datetime.is_monotonic_increasing
    duplicate_times = int(df.datetime.duplicated().sum())
    duplicate_rows = int(df.duplicated().sum())
    df = df.sort_values('datetime').reset_index(drop=True)
    missing = df.isna().sum()
    numeric = df.select_dtypes('number')
    nonfinite = {col: int((~np.isfinite(numeric[col])).sum()) for col in numeric
                 if (~np.isfinite(numeric[col])).any()}

    qc = {
        'rows': len(df), 'columns': len(df.columns),
        'first_timestamp': str(df.datetime.min()),
        'last_timestamp': str(df.datetime.max()),
        'was_sorted': bool(sorted_before),
        'duplicate_timestamps': duplicate_times,
        'duplicate_rows': duplicate_rows,
        'non_hourly_intervals': int((df.datetime.diff().dropna() != pd.Timedelta(hours=1)).sum()),
        'missing_cells': int(missing.sum()),
        'missing_by_column': missing[missing > 0].astype(int).to_dict(),
        'nonfinite_numeric': nonfinite,
        'nonpositive_load': int((df.nat_demand <= 0).sum()),
        'constant_columns': [col for col in numeric if numeric[col].nunique() <= 1],
        'holiday_id_disagreement': int(((df.Holiday_ID > 0) != (df.holiday == 1)).sum()),
        'flag_changes_within_day': {
            col: int((df.groupby(df.datetime.dt.date)[col].nunique() > 1).sum())
            for col in ('holiday', 'school')
        },
        'timestamp_timezone': 'CSV-failis määramata'
    }
    if any((duplicate_times, qc['non_hourly_intervals'], qc['missing_cells'],
            len(nonfinite), qc['nonpositive_load'])):
        raise ValueError(f'Andmete kvaliteedikontroll ebaõnnestus: {qc}')
    return df, qc


def split_periods(df):
    return {
        'train': df[df.datetime < '2018-01-01'].copy(),
        'validation': df[df.datetime.between('2018-01-01', '2020-01-01', inclusive='left')].copy(),
        'test': df[df.datetime >= '2020-01-01'].copy()
    }


def save_plot(fig, folder, name):
    fig.savefig(folder / name, dpi=210, bbox_inches='tight')
    plt.close(fig)


def plot_01(df, train, folder):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout='constrained')
    daily = df.set_index('datetime').nat_demand.resample('D').mean()
    axes[0].plot(daily.index, daily, lw=.8)
    for date in ('2018-01-01', '2020-01-01'):
        axes[0].axvline(pd.Timestamp(date), ls='--', color='grey')
    axes[0].set(title='Kogu perioodi keskmine ööpäevane koormus (andmekontroll)', ylabel='Elektrikoormus')
    sample = train.set_index('datetime').loc['2017-06-25':'2017-07-03', 'nat_demand']
    axes[1].plot(sample.index, sample, lw=1.15)
    axes[1].set(title='Ebatavaline periood treeningandmetes: juuni–juuli 2017', ylabel='Elektrikoormus')
    for ax in axes:
        ax.grid(alpha=.2)
    save_plot(fig, folder, '01_load_timeseries.png')


def plot_02(train, folder):
    data = train.assign(hour=train.datetime.dt.hour, day=train.datetime.dt.dayofweek)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for weekend, label in ((False, 'Tööpäev'), (True, 'Nädalavahetus')):
        hourly = data[(data.day >= 5) == weekend].groupby('hour').nat_demand.mean()
        axes[0].plot(hourly.index, hourly.values, label=label)
    axes[0].set(title='Keskmine tunnikoormus (treeningandmed)', xlabel='Kellaaeg', ylabel='Elektrikoormus')
    axes[0].legend()
    weekdays = data.groupby('day').nat_demand.mean()
    axes[1].bar(range(7), weekdays.values)
    axes[1].set_xticks(range(7), WEEKDAYS)
    axes[1].set(title='Keskmine koormus nädalapäeviti', ylabel='Elektrikoormus')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    save_plot(fig, folder, '02_load_patterns.png')


def plot_03(train, folder):
    data = train.assign(hour=train.datetime.dt.hour)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    density = axes[0].hexbin(data.T2M_dav, data.nat_demand, gridsize=42, mincnt=1, cmap='Blues')
    fig.colorbar(density, ax=axes[0], label='Vaatluste arv')
    axes[0].set(title='Temperatuuri ja koormuse seos', xlabel='Temperatuur (T2M_dav)', ylabel='Elektrikoormus')
    bins = np.unique(np.quantile(data.T2M_dav, np.linspace(0, 1, 15)))
    data['bin'] = pd.cut(data.T2M_dav, bins=bins, include_lowest=True)
    for mask, label in ((data.hour.between(9, 18), 'Päev (09–18)'),
                        (data.hour.between(0, 5), 'Öö (00–05)')):
        group = data[mask].groupby('bin', observed=True).agg(temp=('T2M_dav', 'mean'), load=('nat_demand', 'mean'))
        axes[1].plot(group.temp, group.load, marker='o', markersize=3, label=label)
    axes[1].set(title='Koormus temperatuurivahemike ja kellaaja järgi',
                xlabel='Temperatuur (T2M_dav)', ylabel='Keskmine koormus')
    axes[1].legend()
    axes[1].grid(alpha=.2)
    save_plot(fig, folder, '03_temperature_demand.png')


def plot_04(train, folder):
    changes = train.nat_demand.diff().dropna()
    middle = changes[changes.between(changes.quantile(.005), changes.quantile(.995))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    axes[0].hist(middle, bins=75)
    for quantile in (.01, .99):
        axes[0].axvline(changes.quantile(quantile), ls='--', color='grey')
    axes[0].set(title='Tunnimuutuste jaotus (äärmised 0,5% välja jäetud)',
                xlabel='Koormuse muutus ühe tunniga', ylabel='Vaatluste arv')
    data = train.assign(hour=train.datetime.dt.hour, change=train.nat_demand.diff())
    axes[1].boxplot([data.loc[data.hour == h, 'change'].dropna() for h in range(24)],
                    positions=list(range(24)), showfliers=False)
    axes[1].set_xticks(range(0, 24, 3), range(0, 24, 3))
    axes[1].set(title='Koormuse tunnimuutused kellaaja järgi', xlabel='Kellaaeg', ylabel='Koormuse muutus')
    for ax in axes:
        ax.grid(alpha=.2)
    save_plot(fig, folder, '04_load_changes.png')


def plot_05(train, folder):
    data = train.assign(hour=train.datetime.dt.hour)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    data.boxplot('nat_demand', by='holiday', ax=axes[0], showfliers=False)
    axes[0].set(title='Pühade ja koormuse seos (korrigeerimata)',
                xlabel='Püha tunnus (0 = ei, 1 = jah)', ylabel='Elektrikoormus')
    for flag, label in ((0, 'Tavapäev'), (1, 'Püha')):
        hourly = data[data.holiday == flag].groupby('hour').nat_demand.mean()
        axes[1].plot(hourly.index, hourly.values,
                     label=f'{label} (n={sum(data.holiday == flag):,} tundi)')
    axes[1].set(title='Keskmine tunniprofiil', xlabel='Kellaaeg', ylabel='Keskmine koormus')
    axes[1].legend()
    axes[1].grid(alpha=.2)
    fig.suptitle('Kirjeldav seos, mitte püha põhjuslik mõju', fontsize=11)
    save_plot(fig, folder, '05_holiday_effect.png')


def plot_06(train, validation, folder):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout='constrained')
    cases = ((train, '2017-06-26', '2017-06-29', 'Treeningandmed: 27.06.2017 võimalik anomaalia'),
             (validation, '2019-01-19', '2019-01-22', 'Valideerimisandmed: 20.01.2019 võimalik anomaalia (andmekontroll)'))
    for ax, (data, start, end, title) in zip(axes, cases):
        series = data.set_index('datetime').loc[start:end, 'nat_demand']
        ax.plot(series.index, series.values, marker='.', markersize=3)
        ax.set(title=title, ylabel='Elektrikoormus')
        ax.grid(alpha=.2)
    save_plot(fig, folder, '06_extreme_events.png')


def plot_07(train, folder):
    columns = ['nat_demand', 'T2M_toc', 'T2M_san', 'T2M_dav', 'QV2M_san',
               'TQL_san', 'W2M_toc', 'holiday', 'school']
    correlations = train[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.7), layout='constrained')
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha='right')
    ax.set_yticks(range(len(columns)), columns)
    for (row, col), value in np.ndenumerate(correlations.to_numpy()):
        ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(value) > .72 else 'black')
    ax.set_title('Treeningandmete tunnuste Pearsoni korrelatsioonid')
    fig.colorbar(image, ax=ax, shrink=.75)
    save_plot(fig, folder, '07_correlations.png')


def plot_08(train, folder):
    lags = np.arange(1, 169)
    values = [train.nat_demand.autocorr(lag=int(lag)) for lag in lags]
    fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
    ax.plot(lags, values, lw=1.2)
    for lag in (1, 24, 168):
        ax.axvline(lag, color='grey', ls='--', lw=.8)
    ax.set(title='Koormuse autokorrelatsioon (treeningandmed)',
           xlabel='Nihe tundides', ylabel='Autokorrelatsioon')
    ax.grid(alpha=.2)
    save_plot(fig, folder, '08_autocorrelation.png')


def plot_09(train, folder):
    data = train.assign(year=train.datetime.dt.year, month=train.datetime.dt.month)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    monthly = data.groupby('month').nat_demand.mean()
    axes[0].plot(monthly.index, monthly.values, marker='o')
    axes[0].set(title='Keskmine koormus kalendrikuude järgi', xlabel='Kuu', ylabel='Keskmine koormus')
    axes[0].set_xticks(range(1, 13))
    for year, group in data.groupby('year'):
        values = group.groupby('month').nat_demand.mean()
        axes[1].plot(values.index, values.values, marker='.', label=str(year))
    axes[1].set(title='Kuude keskmised treeningaastate kaupa', xlabel='Kuu', ylabel='Keskmine koormus')
    axes[1].legend()
    axes[1].set_xticks(range(1, 13))
    for ax in axes:
        ax.grid(alpha=.2)
    save_plot(fig, folder, '09_monthly_profiles.png')


def run_eda(input_path, output_dir):
    df, _ = load_and_validate(Path(input_path))
    splits = split_periods(df)
    train = splits['train']
    validation = splits['validation']

    figures_dir = Path(output_dir) / 'figures' / 'eda'
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_01(df, train, figures_dir)
    for plot in (plot_02, plot_03, plot_04, plot_05):
        plot(train, figures_dir)
    plot_06(train, validation, figures_dir)
    for plot in (plot_07, plot_08, plot_09):
        plot(train, figures_dir)

    print(f'Analüüs valmis. Andmeridu: {len(df)}')
    print(f'Treeningandmeid: {len(train)}, valideerimisandmeid: {len(validation)}, '
          f'testandmeid: {len(splits["test"])}')
    print(f'Koormuse autokorrelatsioon (1 tund): {train.nat_demand.autocorr(1):.3f}')
    print(f'Koormuse ja temperatuuri korrelatsioon: {train.nat_demand.corr(train.T2M_dav):.3f}')
    print(f'Graafikud salvestati kausta: {figures_dir}')


def main():
    parser = argparse.ArgumentParser(description='Panama elektrikoormuse uuriv andmeanalüüs')
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/continuous dataset.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    run_eda(args.input, args.output_dir)


if __name__ == '__main__':
    main()
