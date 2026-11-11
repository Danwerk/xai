# Elektrikoormuse prognoosi parandajate võrdlus

Kogu käivitatav analüüs on failis [compare_correction_methods.ipynb](compare_correction_methods.ipynb). Vali projekti `.venv` Pythoni kernel ja käivita **Restart Kernel → Run All**.

Notebook sisaldab järjest andmete ülevaadet, baasmudelite arendusvõrdlust, ajalist OOF-treeningut, parandajate seadete valikut, Panama ja Eesti hindamist, lisakontrolle ning tulemuste tabeleid ja graafikuid. Tulemused jäävad notebooki; eraldi tulemusekaustu ega CSV-, pildi- või Markdown-aruandeid ei genereerita.

Lisaküsimus: **Kuidas mõjutab XAI-meetodi valik elektrikoormuse prognoositäpsust ja arvutuskulu, kui selgitusi kasutatakse prognoosi parandamiseks?** Notebook võrdleb TreeSHAP-i, LIME-i ja tunnuserühmade asendamist sama treenitud parandusmudeli ning kaalureegliga. Selgitamata osa jääb muutmata ja selle suurus on eraldi näha. Näidatakse täpsust, kahjulikkust, arvutusaega ja juhuseemne mõju.

Lisatud on üks fikseeritud ühendatud parandaja katse: XAI-prognoosi enda varasemate vigade parandus ning sama lisakihiga ilma XAI-ta kontroll. Plaan fikseeritakse Panama arenduskatse järel ja hinnatakse eraldi Eesti 2025. aastal. Kõik tulemused jäävad samasse notebooki.

Praktilise osa kontrollid sisaldavad ka naiivset järgmise tunni alusprognoosi, SHAP-kaalude muutuste ja tühistumise diagnoosi, reeglipõhiselt valitud abi/kahju näiteid ning uurimisküsimuste koondtabelit.

**Andmepiirang:** Panama 24 h võrra nihutatud MERRA-2 ilm ei vasta selle tegelikule avaldamisviivitusele. Koormuse kohene saabumine ja osa kohaliku CSV ühiku-/intervallimetaandmeid jäävad eeldusteks. Tulemused on ajaloolise simulatsiooni algühikutes; notebook eristab allikatest kinnitatud faktid ja lahtised eeldused.

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
- `src/simple_correction.py` — sama kellaaja varasemate vigade parandus ning XAI ja veataseme paranduse ühendatud katse koos kontrolliga.
- `src/correction.py`, `src/shap_explainer.py` — SHAP-komponendid, krediit ja adaptiivsed kaalud.
- `src/xai_comparison.py` — TreeSHAP-i, ametliku LIME-i ja tunnuserühmade asendamise võrdlus; selgitamata osa kontroll ja ajamõõtmine.
- `src/study.py`, `src/evaluation.py` — mälus toimuv treening, arendusvalik, mõõdikud, naiivne prognoos, paranduste diagnoos ja paarisvõrdlused.

Moodulid ei kirjuta katseväljundeid. Testid kontrollivad ajajoondust, tulevikuinfo välistamist, krediite ja meetodite kooskõla:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Panama 2020 ja Eesti 2024 on varem uuritud perioodid. Võrdlus on tagasivaateline; seaded valitakse Panama 2018–2019 arenduses. Täpne metoodika, eeldused, piirangud ja tulemuste tõlgendus on notebookis.
