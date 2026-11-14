# Electricity load forecast correction

This thesis project compares XAI-based load forecast correction with simpler methods
and studies the effect of the explanation method on accuracy and computation time.
The active experiments use **Panama only**: settings are selected on **2018–2019**
and evaluated on the available **2020** observations, ending in June.

## Setup

Windows and Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place `continuous dataset.csv` in `data/raw/`. It contains the Panama load,
weather and calendar observations. Select the project's **.venv** notebook kernel
and use **Restart Kernel → Run All**.

## Notebooks and shared code

1. [00_data_analysis.ipynb](00_data_analysis.ipynb) — source audit, availability
   assumptions and pre-development load profiles.
2. [01_compare_correction_methods.ipynb](01_compare_correction_methods.ipynb) —
   training, correction selection, explanation-method comparisons and diagnostics.

Both notebooks read their inputs directly and run independently. The combined
XAI-plus-bias experiment reuses the same Panama evaluation models and hours;
it is a retrospective extension, not an independent confirmation period.

- `data/` — source data, retained unchanged and excluded from Git.
- [utils/](utils/) — shared preprocessing, modelling, correction and evaluation helpers.
- [utils/plotting.py](utils/plotting.py) — TalTech palette, fonts and drawing helpers.
  Use `with plot_style("taltech"):` and `overrides=ANALYSIS_SIZES` for compact figures.
- [configs/experiment.json](configs/experiment.json) — shared experiment settings.

Results, figures and interpretation stay in the notebooks. Inactive source files and
earlier presentation PNGs are retained; those PNGs are not results of the current
Panama-only run. Source assumptions and methodological limits are documented in the notebooks.
