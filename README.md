# Elektrikoormuse prognoosi parandajate võrdlus

Kogu käivitatav analüüs on failis [compare_correction_methods.ipynb](compare_correction_methods.ipynb). Vali projekti `.venv` Pythoni kernel ja käivita **Restart Kernel → Run All**.

Notebook sisaldab järjest andmete ülevaadet, baasmudelite arendusvõrdlust, ajalist OOF-treeningut, parandajate seadete valikut, Panama ja Eesti hindamist, lisakontrolle ning tulemuste tabeleid ja graafikuid. Tulemused jäävad notebooki; eraldi tulemusekaustu ega CSV-, pildi- või Markdown-aruandeid ei genereerita.

## Käivitamine

Windows / Python 3.11. Sõltuvused ja notebooki kernel:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Algandmed peavad olema kaustas `data/raw/`:

- `continuous dataset.csv` — [Panama koormus-, ilma- ja kalendriandmed](https://data.mendeley.com/datasets/byx7sztj59/1).
- `estonia_load.csv` — [Eleringi andmed](https://dashboard.elering.ee/assets/api-doc.html), veergudega `datetime` (UTC) ja `consumption`.

Need kaks CSV-d on sisendandmed ja säilivad. Kõik katseseaded asuvad failis [configs/experiment.json](configs/experiment.json).

## Meetodid ja kood

- `src/data_preprocessing.py`, `src/feature_engineering.py` — andmete kontroll ja ajaliselt joondatud tunnused.
- `src/compare_forecasting_models.py`, `src/models.py` — baasmudelite võrdlus ning baas- ja parandusmudelid.
- `src/simple_correction.py` — lihtne sama kellaaja varasemate vigade keskmisel põhinev parandaja.
- `src/correction.py`, `src/shap_explainer.py` — SHAP-komponendid, krediit ja adaptiivsed kaalud.
- `src/study.py`, `src/evaluation.py` — mälus toimuv treening, arendusvalik, mõõdikud ja paarisvõrdlused.

Moodulid ei kirjuta katseväljundeid. Testid kontrollivad ajajoondust, tulevikuinfo välistamist, krediite ja meetodite kooskõla:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Panama 2020 ja Eesti 2024 on varem uuritud perioodid. Võrdlus on tagasivaateline; seaded valitakse Panama 2018–2019 arenduses. Täpne metoodika, eeldused, piirangud ja tulemuste tõlgendus on notebookis.
