# Elektrikoormuse prognoosivea diagnoosimine

Uurime, kas veaanalüüs, SHAP ja varasemad juhtumid aitavad valida parema mudelimuudatuse kui tavaline veaanalüüs.

**[Teadustulemused ja uurimisküsimuste vastused](reports/thesis_results.md)** on loetavad ilma arvutusi käivitamata. Seniste katsete väikesed koondtabelid on `reports/metrics.csv`, `reports/trials.csv` ja `reports/comparisons.csv`. Need sisaldavad ka ebaõnnestunud katseid.

Uus XAI-katse kasutab **paranduskasu tagasisidet**: veaparandaja osade Shapley-jaotus näitab nende osa varem realiseerunud kasust või kahjust ning juhib järgmiste prognooside parandust. Panama arenduses valitud meetodit kontrolliti muutmata seadistustega Eesti 2024. aasta tarbimisel. Aruanne sisaldab tulemusi, koosmõju eemaldamise kontrolli ja kahe arvveeruga paranduskaarte. [Uudsusotsing ja panuse piirid](reports/research_directions.md) eristavad seda lähedastest varasematest meetoditest.

## Mis GitHubi läheb?

| Asukoht | Sisu |
|---|---|
| `configs/` | Algse ja laiendatud katse seadistused. |
| `src/` | Andmetöötlus, tunnused, mudelid, selgitused ja hindamine. |
| `templates/` | Diagnostikakaartide mallid. |
| `notebooks/` | Analüüsi tekst ja kood ilma mahukate salvestatud väljunditeta. |
| `reports/` | Põhiaruanne, kolm koond-CSV-d ja üks kokkuvõttejoonis. |
| `tests/` | Oluliste arvutuste automaattestid. |
| `tools/` | Aruande, kaartide, järelanalüüside ja kontrollide tööriistad. |

Algandmed, `.venv/`, vahemälud ja kogu `results/` on Gitist välja jäetud. GitHubist võetud projektis pole treenitud mudeleid ega tunnipõhiseid katseväljundeid. Treenitud mudeleid failidesse ei kirjutata. Tunnuse- ja mudelivaliku vaheprognoosid arvutatakse mälus ning neist salvestatakse ainult mõõdikud. Lõppvõrdluse ja kaartide jaoks vajalikud kohalikud väljundid tekivad uuringu käivitamisel `results/` alla.

Töödeldud tunnusetabeleid ja treeningu ning hindamise andmejaotusi eraldi CSV-dena ei salvestata. Vajaduse korral koostavad tööriistad ja märkmikud need algandmetest ning katse seadistustest uuesti mälus; selleks pole vaja mudelit treenida. Katsete arvu analüüs kasutab põhikäivituse prognoose mälus ning treenib ainult puuduvad mudelivariandid. Kõik selle tabelid ja vajalikud prognoosid jäävad sama põhikäivituse alla. Kasutamata korrelatsiooni-, äärmusjuhtumi- ja stsenaariumide vahetabeleid ei kirjutata.

Salvestame mõõdikud, kontrolliks vajalikud prognoosid ja selgitused, loetavad kaardid, joonised ning katse seadistused ja kontrollsummad. Algse katse valimata tunnusekandidaatide ja lihtsate võrdlusmudelite tunniprognoose eraldi ei salvestata. Kaartide uuringumaterjalid tekivad ainult eraldi käsuga `tools/build_card_study.py`.

Teaduslikud järeldused koondatakse põhiaruandesse. Põhikatse ja laiendatud katse ei kirjuta sellest eraldi aruandekoopiaid. Kaartidel säilivad kasutaja- ja arendajavaade ning kasutajakatse võrdluseks vajalikud lihtsad loendid. Varasemat eraldi tehnilise kaardi koopiat enam ei looda. Märkmike tabelite ja jooniste ühine abikood asub `src/reporting.py` failis.

Kaartide täpsed lähteandmed säilivad failis `evidence.json`. Vaadete teksti eraldi JSON-failina ei salvestata; kaartide loomisel ja kontrollimisel koostatakse see lähteandmetest mälus.

Iga põhikaardi jaoks on üks `evidence.json` ning kasutaja- ja arendajatekst. Kasutajakatse viitab neile samadele kaartidele. Eraldi luuakse ainult katses kasutatava nelja juhtumi kaheksa lihtsat loendit. Kasutamata aruandekoopiaid, lõpphindamise olukorratabelit ja kordavat tulemuste joonist ei looda. Enne lõpphindamist tehtud mudelivalik säilib käivituse manifestis.

Kaartidega seotud kood on kahes failis: `src/card_generator.py` valib juhtumid, koondab arvud ja rakendab soovitusreegleid; `src/card_views.py` koostab mõlema rolli teksti. Ühised eestikeelsed nimetused on failis `src/labels.py`, paigutus mallis `templates/role_card.md.j2`.

## Paigaldamine

Python 3.11 või uuem, PowerShell projekti juurkaustas:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Lisa algandmed faili `data/raw/continuous dataset.csv`. Andmestiku allikas ja kasutuse piirangud on [põhiaruandes](reports/thesis_results.md). Andmeid automaatselt alla ei laadita ega sünteesita. `requirements-lock.txt` kirjeldab algset Windowsi katsekeskkonda; `requirements-dev.txt` sisaldab valikulist koodivormindajat.

## Käivitamine

Iga uus põhikäivitus treenib mudelid uuesti. Varem salvestatud mudelit ei laadita. Kogu uuringu järjekord on:

```powershell
.\.venv\Scripts\python.exe -B run_pipeline.py
.\.venv\Scripts\python.exe -B run_expanded_study.py
.\.venv\Scripts\python.exe -B tools\build_card_study.py
.\.venv\Scripts\python.exe -B tools\build_report.py
```

`run_pipeline.py` teeb algse XAI-katse ning võrdleb ühe ja kahe katsega muudatuse valimist. `run_expanded_study.py` võrdleb tunnuseid ja viit mudeliperekonda ning arvutab nende paarisvõrdlused. Katsete arvu analüüs ja paarisvõrdlus salvestatakse põhikäivituste juurde. `tools/` sisaldab kolme abiprogrammi: kasutajakatse materjalid, põhiaruande uuendamine ja tulemuste kontroll. Muutunud tulemuste korral tuleb autoril üle vaadata ka aruande järeldused.

Abimudeli katse kuulub põhikäivitusse. Võrreldakse prognooside ühendamist, varasemate vigade parandamist ja kiire kasvu näidete suuremat kaalu. Seadistused on `specialist` all; `specialist_quantile` määrab esimese võrdlusvariandi kasvulävendi. Variant valitakse 2018.–2019. aasta üldise ja kiire kasvu MAE ning RMSE põhjal. Kui ükski variant kõiki neid näitajaid ei paranda, säilib algmudel. Seejärel hinnatakse valikut 2020. aastal, mida oli juba varem uuritud. Valitud lahendus parandab GBR-i prognoosi HistGBR-iga, mis õpib varasemate aastate treeninguvälistest vigadest.

Kõigi variantide mõõdikud jäävad ühte faili `refinements/specialist_metrics.csv`. Valitud variandi ja nelja võrdlusmudeli lõppprognoosid on samas failis `refinements/test_specialist.csv`; eraldi mudeli- ega vaheprognoosifaile ei looda. Tulemused on märkmikus 07 ja põhiaruandes; ainult selle osa uuendamiseks saab kasutada `tools/build_report.py --specialist-only`.

Senised tulemused on tagasivaatelised. 2020. aasta uuesti arvutamine ei ole uus sõltumatu katse. Vaatamata andmetel kontrollimise kord on põhiaruande osas „Sõltumatu kontroll enne tugevat järeldust”; selleks vajalikke uusi andmeid projektis veel pole. Uuel testiosal esitatakse tulemus ka siis, kui parandust ei tule.

Abimudeli valik ja hindamine on koodis eraldi: `develop_specialist` saab ainult treeningu ja arenduse andmed. Põhikäivitus salvestab valitud variandi, seadistused, kasvulävendi ning hindamise reeglid olemasolevasse manifesti enne lõpphindamise mudelite treenimist. `evaluate_specialist` loeb sealt fikseeritud valiku ega vali hilisemate tulemuste järgi uut varianti. Käivitus on märgitud tagasivaateliseks; uusi protokolli- ega vahefaile ei looda.

Võti `-B` väldib Pythoni `__pycache__` failide loomist.

Kohalikud `results/latest_*.json` failid aitavad tööriistadel leida sama käivituse väljundid; manifestid sisaldavad seadistusi ja kontrollsummasid. Käivitused saavad eraldi kaustad. Neid võib pärast vajalike koondite koostamist kustutada; uue detailse analüüsi jaoks tuleb uuring siis uuesti arvutada.

## Märkmikud ja kontrollid

Uus paranduskasu katse käivitub koos põhitoruga. Ainult selle osa arvutamiseks olemasoleva põhikäivituse juurde kasuta:

```powershell
.\.venv\Scripts\python.exe -B run_pipeline.py --attribution-only --development-only
.\.venv\Scripts\python.exe -B run_pipeline.py --attribution-only
.\.venv\Scripts\python.exe -B run_pipeline.py --attribution-transfer
.\.venv\Scripts\python.exe -B tools\build_report.py --attribution-only
```

Esimene käsk on valikuline arenduskontroll. Teine valib seadistused 2018.–2019. aasta põhjal ja hindab varem nähtud Panama 2020. aastat. Kolmas kasutab fikseeritud valikut Eestis, ilma kohaliku parameetriotsinguta. Ülekandekatse vajab faili `data/raw/estonia_load.csv`, veergudega `datetime` (UTC) ja `consumption`. Allikas on [Eleringi avalik süsteemiandmete API](https://dashboard.elering.ee/assets/api-doc.html), päring `GET /api/system/with-plan?start=<UTC>&end=<UTC>`, tabel `data.real`. Selle uuringu kasutatav osa on 2018–2024. Koormuse nullväärtused ja puudulikud tunnid jäävad hindamisest välja; neid ei interpoleerita. 2025. aasta väljavõte oli puudulik ja seda ei kasutatud. Täpne ajajoondus ning tunnuste vastendus on aruandes ja käivituse manifestis.

Vahemudeleid ega tunnusetabeleid ei salvestata. Lisandub üks lõppprognooside ja paranduspanuste CSV kummagi andmestiku kohta ning kaks tulemuste joonist. Kõigi variantide mõõdikud ja paarisvõrdlused lähevad olemasolevatesse `reports/trials.csv` ja `reports/comparisons.csv` failidesse; fikseeritud valikud ja seaded olemasolevasse manifesti. Algsete kaartide piloot ja selle vastusemallid on sellest katsest eraldi.

Kaartidele keskenduva osa jaoks piisab põhikatsest. Seejärel loo 12 juhtumi rollivaated ja kasutajakatse materjalid ning uuenda põhiaruande kahe näite läbikäiku:

```powershell
.\.venv\Scripts\python.exe -B tools\build_card_study.py
.\.venv\Scripts\python.exe -B tools\build_report.py --cards-only
```

Kaardid tekivad `results/runs/<käivitus>/cards/` alla. Kasutajakatse kaustas `results/card_studies/<käivitus>/README.md` on kaartide sisukord ning `study/` all juhend, ülesannete järjekorrad ja tühjad vastusemallid. Märkmik 06 kuvab kaardivaateid. Kaardid on juhtumite hilisemaks uurimiseks; kasutajate vastused tuleb päriselt koguda. Uuringu maht ja kaks juhtuminäidet on [põhiaruandes](reports/thesis_results.md).

Pärast katsete arvutamist ava märkmikud IDE-s või Jupyteris:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Märkmikud 01–07 käsitlevad põhiosa ja 09–11 lisaanalüüse. Uurimisküsimuste vastused on põhiaruandes. Käivita märkmiku lahtrid ülevalt alla, kasutades projekti `.venv` keskkonda. GitHubi jaoks jäta mahukad lahtriväljundid salvestamata.

Kiire kontroll toimib ka ilma algandmete ja kohalike katseväljunditeta:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B tools\check_results.py
```

`pytest` kontrollib olulisi arvutusi ja reegleid väikeste näidetega. `check_results.py` kontrollib avalike koondtabelite mõõdikuid ja usaldusvahemikke. Käsuga `tools/check_results.py --full` võrreldakse ka olemasolevaid kohalikke tulemusi: hindamistunde ja tegelikku koormust, prognoosivigu ja mõõdikuid, mudelivalikut ning SHAP-i ja kaartide kooskõla. Kontroll ei treeni mudeleid ega loo faile. Puuduvate uuringuosade kohta öeldakse selgelt, et neid ei kontrollitud.

Tunnuste uuesti koostamiseks kasutab kood `src.feature_engineering.load_run_features()` funktsiooni. See kontrollib, et algandmed ja tunnuste arvutamise kood vastavad katsele. Kui need on muutunud, tuleb uuring uuesti käivitada. Salvestatud arvväärtuste lugemisel hoiab `src.common.read_artifact_csv()` alles vajalikud komakohad.
