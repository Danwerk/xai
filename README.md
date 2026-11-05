# Elektrikoormuse prognoosi parandamine tagasiside abil

Magistritöö: **Realised-credit-based adaptive residual correction for short-term electricity load forecasting using XAI-based component attribution.**

Projekt ennustab järgmise tunni elektrikoormust. Uurime, kas prognoosi saab parandada, õppides sellest, millised varasemad parandused tegelikult aitasid ja millised suurendasid viga.

## Metoodika

1. **Baasprognoos.** GBR-mudel ennustab koormust varasema koormuse, kalendri ja võimaluse korral ilmaandmete põhjal.
2. **Veaparandus.** Teine mudel, HistGBR, ennustab baasprognoosi viga ja pakub paranduse.
3. **Paranduse osad.** SHAP-selgitusmeetod jagab paranduse osadeks, näiteks koormusajaloo või kellaaegadega seotud panusteks.
4. **Tagasiside.** Kui tegelik koormus saabub, hinnatakse osade kasu koos teiste osadega. Seda nimetame *realiseerunud krediidiks*. Varasema kasu põhjal määratakse osade kaalud järgmistes prognoosides.

**Pakutud meetodi keskne idee on paranduse osade kaalumine nende tegeliku varasema kasu järgi.** Põhivariant hindab kasu ruutvea vähenemisena. Kaalud jäävad vahemikku 0–1. Krediiti hinnatakse kogu võimaliku paranduse kohta, ka nende osade puhul, mille kaalu prognoosis vähendati. Mudelid on aasta jooksul fikseeritud; muutuvad ainult paranduse kaalud.

## Mida võrreldakse?

| Variant tulemuste failides | Selgitus |
|---|---|
| B0 `base` | Algne GBR-prognoos. |
| B1 `direct` | HistGBR ennustab koormust otse, kasutades samu sisendeid kui parandusmudel. |
| B2 `residual` | Baasprognoos koos parandusmudeli täieliku parandusega. |
| B3 `fixed` | Parandusele rakendatakse üht arenduses valitud püsivat kaalu. |
| B4 `scalar_adaptive` | Kogu parandusel on üks kohanduv kaal, ilma komponentide krediidita. |
| B5 `standalone_feedback` | Iga parandusosa kaal sõltub selle eraldi hinnatud kasust. |
| **B6 `credit_feedback`** | **Pakutud meetod: osade krediit arvestab nende koosmõju.** |

Lisakontrollid ehk *ablation'id* uurivad, kas kasu tuleb eraldi komponentide kaaludest, koosmõju arvestamisest või kellaajast. Samuti võrreldakse:

- ruutvea- ja absoluutveapõhist krediiti;
- võrdse kaaluga mälu ning mälu, mis rõhutab uuemaid kogemusi;
- üheksat mäluakna ja tagasiside tugevuse kombinatsiooni.

Kõik need võrdlused käivituvad koos ja lähevad **ühte põhiaruandesse**. Tundlikkusanalüüsist ei valita testitulemuste põhjal uut võitjat.

## Kuidas katseid hinnatakse?

| Andmestik | Parameetrite valik | Lõpphindamine |
|---|---|---|
| Panama | 2018–2019 arendusandmed | Andmestikus olemasolev 2020. aasta osa |
| Eesti | Panamas valitud seaded, ilma kohaliku otsinguta | 2024 |

Treening kasutab alati varasemat aega. Parandusmudel õpib vigadest, mis tekkisid varasemate aastate ette tehtud prognoosides. Kõik parandusvariandid jagavad samu mudeleid, sisendtunnuseid ja hindamistunde. Eesti mudelid treenitakse Eesti varasematel andmetel.

Seaded fikseeritakse enne lõpphindamist. Valik arvestab nii kogu perioodi kui ka kiireid koormuse tõuse ja langusi; nende olukordade piirid leitakse varasemast treeningust. Tegelikku koormust kasutatakse tagasisideks alles pärast selle saabumist. Puuduvaid koormusi ei interpoleerita ja hindamata tunnid raporteeritakse.

Peamised mõõdikud:

- **MAE:** keskmine absoluutne prognoosiviga. **RMSE** annab suurtele vigadele rohkem kaalu.
- **Suhteline paranemine:** kui palju väheneb MAE võrreldes baasprognoosi ja standardse täisparandusega.
- **Kahjulike paranduste osakaal ja suurus:** kui sageli ja kui palju parandus viga suurendab baasprognoosi või standardparandusega võrreldes. Kahju suurus on keskmine üle kõigi hinnatud tundide.

Lisaks näidatakse `bias`-i ehk keskmist üle- või alaprognoosi. Põhivõrdluste usaldusvahemikud arvestavad järjestikuste tundide sõltuvust; täiendavad tundlikkuse kontrollid on kirjeldavad.

## Käivitamine

Kasutatud keskkond on Windows ja Python 3.11. Sõltuvused on fikseeritud ühes [requirements.txt](requirements.txt) failis.

Lisa algandmed kausta `data/raw/`:

- `continuous dataset.csv` — Panama algne koormuse, ilma ja kalendri [andmestik](https://data.mendeley.com/datasets/byx7sztj59/1).
- `estonia_load.csv` — [Eleringi andmed](https://dashboard.elering.ee/assets/api-doc.html), veergudega `datetime` (UTC) ja `consumption`.

Käivita PowerShellis projekti juurkaustast:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B run_pipeline.py
.\.venv\Scripts\python.exe -B -m src.check_results --full
.\.venv\Scripts\python.exe -B -m src.build_report
```

**Kõik katseseaded asuvad failis [configs/experiment.json](configs/experiment.json).** Seadete muutmise järel tuleb tulemused uuesti arvutada või taastada varasemale käivitusele vastavad seaded.

## Kust tulemusi vaadata?

- [Põhiaruanne](reports/thesis_results.md) — meetodite võrdlus, lisakontrollid ja joonised.
- [Kõik mõõdikud](reports/metrics.csv) — tulemused tabelina.
- `results/thesis_runs/<käivitus>/` — prognoosid, tagasiside, arendustulemused ja enne hindamist fikseeritud valikud.

`results/latest_thesis_run.json` viitab viimasele lõpetatud käivitusele. Väljundite kontrollsummad aitavad veenduda, et tulemused vastavad kasutatud koodile ja seadetele.

## Lisakäsud

EDA ehk andmete esmane uurimine näitab treeningandmete kvaliteeti ja mustreid. See ei salvesta lisafaile. Baasmudeli eeluuring võrdleb Ridge'i, Random Foresti, GBR-i ja HistGBR-i samadel tunnustel ja arendusandmetel; see ei asenda automaatselt põhikatse mudelit. [Eeluuringu tulemused](reports/preliminary_models.md).

```powershell
# Andmete ülevaade ja joonised
.\.venv\Scripts\python.exe -B -m src.eda --dataset all

# Baasmudelite eeluuring
.\.venv\Scripts\python.exe -B -m src.compare_forecasting_models

# Automaattestid
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider

# Vanade identsete käivituste eemaldamine pärast aruande uuendamist
.\.venv\Scripts\python.exe -B -m src.check_results --full --prune-duplicates
```

Puhastamine säilitab praeguse ja aruannetes viidatud käivitused. Muud käivitusvalikud leiab käsuga `.\.venv\Scripts\python.exe -B run_pipeline.py --help`.

## Olulised piirangud

Panama 2020 ja Eesti 2024 tulemusi on varem uuritud. Seega on tegu tagasivaatelise analüüsiga; uueks sõltumatuks kinnituseks on vaja seni vaatamata perioodi.

Eeldame, et lõppenud tunni koormus ja prognoosiviga on kohe teada. Panama aja-, intervalli- ja ühikueeldused ning ilmaandmete tegelik avaldamisviivitus vajavad allikapõhist kinnitust. Ilmaandmed on reanalüüs, mitte arhiveeritud ilmaprognoosid. Eesti jaotused kasutavad UTC aega, kalendritunnused Tallinna kohalikku aega.

SHAP-i osad ei tõenda põhjuslikkust. Väiksem keskmine viga ei taga, et kahjulikke parandusi oleks vähem. Detailsed valemid ja teaduslik põhjendus vormistatakse magistritöö Word-dokumendis; meetodi teostus asub [src/correction.py](src/correction.py).
