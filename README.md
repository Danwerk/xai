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
- `src/plotting.py` — korduvkasutatavad graafikustiilid, TalTechi värvid ja font ning graafikute ja skeemide joonistusfunktsioonid.

Moodulid ei kirjuta katseväljundeid.

Panama 2020 ja Eesti 2024 on varem uuritud perioodid. Võrdlus on tagasivaateline; seaded valitakse Panama 2018–2019 arenduses. Täpne metoodika, eeldused, piirangud ja tulemuste tõlgendus on notebookis.

## Graafikute ja skeemide korduvkasutus

Üldine joonistuskood asub failis [src/plotting.py](src/plotting.py). Seda saab importida
nii notebookist kui ka teistest Pythoni failidest, kui projekti juurkaust on imporditeel.
Katsete andmevalik ja meetodite nimetused jäävad notebooki. Slaidide tekst ja
esitlusjooniste genereerimiskood on notebookist eemaldatud; varem genereeritud
esitlusjoonised säilivad projekti juurkaustas `xai_*.png` failidena.

TalTechi stiilis graafik oma andmetega:

```python
import matplotlib.pyplot as plt
from src.plotting import TALTECH_COLORS, horizontal_bars, plot_style, style_chart_axes

with plot_style(overrides={"font.size": 12}):
    fig, ax = plt.subplots(figsize=(10, 5))
    horizontal_bars(
        ax, ["Variant A", "Variant B"], [12.4, 10.8],
        [TALTECH_COLORS["ink"], TALTECH_COLORS["accent"]],
    )
    style_chart_axes(ax)
    ax.set_xlabel("Keskmine viga")
    fig.tight_layout()
    plt.show()
```

Skeem kasutab koordinaate vahemikus 0–1:

```python
import matplotlib.pyplot as plt
from src.plotting import (
    TALTECH_COLORS, diagram_arrow, diagram_box, diagram_canvas, diagram_text, plot_style,
)

with plot_style():
    fig, ax = diagram_canvas(figsize=(10, 4))
    diagram_box(ax, 0.10, 0.35, 0.30, 0.30)
    diagram_text(ax, 0.25, 0.50, "Sisend", ha="center")
    diagram_box(ax, 0.60, 0.35, 0.30, 0.30, edge=TALTECH_COLORS["accent"])
    diagram_text(ax, 0.75, 0.50, "Tulemus", ha="center")
    diagram_arrow(ax, (0.41, 0.50), (0.59, 0.50))
    plt.show()
```

`TALTECH_COLORS`, `TALTECH_FONT` ja stiilisõnastikud koondavad kujundusseaded.
`plot_style()` rakendab neid ajutiselt ja taastab varasemad seaded ploki lõpus.
Vaikimisi kasutatakse TalTechi stiili; `plot_style("analysis")` annab notebookis
kasutatava analüüsistiili ja `plot_style("presentation")` suurema kirjaga esitlusstiili.
Verdanale on lisatud DejaVu Sansi varufont.

Lisaks on olemas `analysis_dates` ajatelje jaoks, `analysis_intervals` valmis
usaldusvahemike kuvamiseks, `label_bars` väärtussiltide jaoks ning `diagram_icon`
kella ja liugurite joonistamiseks. Funktsioonid tagastavad Matplotlibi objektid;
nende kohandamine, näitamine ja soovi korral `fig.savefig(...)` abil salvestamine
jääb kutsuja otsustada.
