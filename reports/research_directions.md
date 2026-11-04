**Uued uurimissuunad: diagnoosist kontrollitud mudeliparanduseni**

Koostatud 03.11.2026 olemasoleva projekti ja sihitud teaduskirjanduse otsingu põhjal. See on uute katsete ettepanek, mitte uute mudelite katsetulemuste aruanne. Olemasolevaid tulemusi ja katseprotokolle ei muudetud.

**Rakenduse täiendus 03.11.2026.** Allpool olev algne ülevaade kirjeldab olukorda enne uut arendusringi. Nüüd on rakendatud ja katsetatud paranduskasu XAI-tagasiside; selle tulemused on [põhiaruandes](thesis_results.md). Järgmine lõik dokumenteerib lisauuringu, tehtud valikud ja uudsuse piiri.

**Uudsusotsing enne meetodi valikut ja selle täpsustamisel.** Kasutasin veebiotsingut, seejärel lähimate tööde kirjastajalehti, arXiv-i, ülikoolide ja autorite avalikke täistekste. Otsing hõlmas muu hulgas päringuid `load forecasting Shapley repair`, `load forecasting loss attribution`, `forecast correction Shapley`, `SHAP values model correction`, `Shapley calibration layer`, `load forecasting Shapley regression`, `Shapley error reduction online forecasting`, `load forecasting Shapley feedback`, `Shapley correction components`, `Shapley gain gating forecasting`, `SHAP contribution rescaling forecasting` ja `load forecasting repair credits`. Otsingu kuupäev on 03.11.2026. See on lai sihitud uudsusotsing, mitte reprodutseeritud andmebaasipõhine süstemaatiline ülevaade; otsingu puuduv vaste ei tõesta meetodi esmakordsust.

| Lähim leitud töö | Kattuvus ja otsus |
|---|---|
| [TempShap, 23.10.2026](https://arxiv.org/html/2610.33330v1) | Ajaliselt detailne veafunktsiooni Shapley-jaotus on juba olemas. Lugesin raamistikku ja hindamisprotokolli. Selle üldidee esmakordsust meie töös ei väida. |
| [Buckmann ja Joseph, 2023](https://www.ijcb.org/sites/default/files/journal/v19n4/ijcb-v19n4-interpretable-machine-learning-workflow-application-economic-forecasting.pdf) | Shapley-regressiooniga mudeli panuste kooskõla hindamine on olemas. Pelgalt regressioon SHAP-panustel pole uus XAI-algoritm. |
| [Debiasing SHAP scores in random forests](https://www.econstor.eu/bitstream/10419/317998/1/10182_2023_Article_479.pdf) | Olemas on SHAP-skooride skaleerimine, mis säilitab algse prognoosi. Meie prognoosi parandamise katse muudab prognoosi, kuid skaleerimise üldidee pole uus. |
| [SHAP-MoE tuuleprognoosis](https://www.mdpi.com/1996-1073/19/1/124) ja [SHAP-iga juhitud EV-koormuse dünaamiline stacking](https://doi.org/10.1016/j.egyai.2026.100829) | SHAP-i kasutamine mudeli ühendamiskaaludes on juba olemas ka energiavaldkonnas. Meie eristus ei saa olla lihtsalt „SHAP juhib kaale”. Kirjastaja indekseeritud meetodikirjeldused näitavad sisendtunnuste SHAP-vektoreid ja ekspertide kaalumist. |
| [TreeSHAP-Importance-Weighted Feature-Group Fusion, 2026](https://doi.org/10.3390/math14132419) | Lähedane tabulaarne rühmade ühendamine: globaalsete absoluutsete SHAP-panuste softmax-kaalud ja valideerimisel valitud fikseeritud segamiskordaja. Kontrollisin meetodi võrrandite kirjeldust. Meie tagasisides on alus varem realiseerunud paranduskasu, märgiga koosmõjujaotus ja sama tunni ajaloo järgi uuenevad osakaalud. |
| [Beyond forecast leaderboards, 2024/2026](https://arxiv.org/html/2412.08916v3) | Prognoosiansambli liikmete hindamine nende lisatud täpsuse ja Shapley-väärtuse kaudu on olemas. Meie mängijad on ühe fikseeritud veaparandaja sisulised parandusosad, mitte eraldi täisprognoosimudelid. Mängu kasujaotus läheb järgmise tunni paranduse tagasisidesse. |
| [Chen jt, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9349278/), [Park jt, 2026](https://doi.org/10.32604/cmc.2026.079734), [Bhupatiraju ja Ahn, 2025](https://arxiv.org/abs/2507.22220) | Veafunktsiooni selgitamine ja XAI-põhine mudeli parandamine on olemas. Neid üldpõhimõtteid meie leiutisena ei esitata. |

Osa kirjastajate otsepäringuid oli piiratud (429 või JavaScripti kontroll); nendel juhtudel kasutasin kirjastaja indekseeritud abstrakti või meetodikirjeldust ning ei nimeta seda kogu täisteksti lugemiseks. Üldotsingus nähtud automaatseid populaarteaduslikke kokkuvõtteid ja kontrollimata ideelehti ei kasutatud teaduslike uudsusväidete alusena. Ühtegi nende tööde lähtekoodi ei kopeeritud projekti; teostus kirjutati olemasoleva toru jaoks.

**Valitud eristus.** Meetod ühendab: (1) ajaliselt treeningust välja jäetud vigadel õppiva parandaja; (2) selle panuste jagamise sisulisteks paranduse osadeks; (3) tegeliku veavähenemise täpse Shapley-jaotuse nende osade vahel; (4) selle jaotuse kasutamise ainult varasemate sama tunni juhtumite põhjal järgmiste paranduste juhtimiseks; (5) prognoosi muudatuse ja mõõdetud kasu kuvamise eraldi arvveergudes. Otsingus ei leidnud elektrikoormuse uuringut, mis teostaks täpselt seda tagasisideahelat koos koosmõju eemaldamise kontrolli ja fikseeritud ülekandekatsega. See on põhjendatav uudsuse kandidaat, mitte garanteeritud esmakordsus.

**Rakendatud väärtused.** Kui algprognoos on f ja potentsiaalse parandaja osad c_j, defineerime parandusmängu `v(S) = L(y,f) − L(y,f+Σ_(j∈S)c_j)`. Mängijad lülitavad paranduse osi sisse/välja; andmete sisendlahtreid ei maskeerita. Ruutvea korral on täpne jaotus `ψ_j = c_j(2(y−f)−Σc_k)`, mistõttu osade summa taastab kogu ruutvea vähenemise. See on tuntud Shapley-mängu algebraline erijuht, mitte uus üldteoreem. Võrreldav koosmõjuta väärtus on `c_j(2(y−f)−c_j)`. Just nende erinevuse praktilist kasu kontrollime. Absoluutvea kuvamisel arvutatakse väikese mängu täpsed väärtused kõigi koalitsioonidega.

Tagasiside kaal on `clip(1 + tugevus × varasemate ψ_j keskmine / varasemate |ψ_j| keskmine, 0, 1)`. Piisava ajaloota kaal on üks. Sama tunni variant kasutab kuni akna_pikkus/24 varasemat selle kohaliku tunni juhtumit, mitte tingimata järjestikuseid päevi; puuduvaid aknaid ei täideta. Kasu arvutatakse varasema potentsiaalse täisparanduse kohta ka siis, kui mõnd osa tol tunnil päris väljundis vähendati. Kõik selleks vajalikud sihtväärtused peavad olema järgmise otsuse tegemise hetkeks juba selgunud. See on arvutatud parandusmäng, mitte tõend tarbimise või andmekanali põhjuslikust mõjust.

**Arenduse käik ja negatiivsed tulemused.** Kõigepealt katsetasin sama perioodilise lineaarse paranduskihi õppimist algtunnustel ning prognoosi SHAP-panustel, mõlemat koos veamäluga ja ilma. Iga pere sai sama üheksa seadistust. SHAP-esitus jäi 2018.–2019. aasta valikus algtunnuste esitusele alla. Seda katset ei peidetud ega nimetatud uueks leiutiseks. Järgmine variant kasutas tugevama HistGBR-i parandaja kasujaotust tagasisidena. Arendus lubas kaks ajalooakent (168 ja 672), jooksva või sama tunni tingimuse ning tugevused 0,25 ja 1; võrreldavad koosmõjuta ja ühe üldkaaluga kontrollid said sama hulga seadeid. Täisparanduse säilitamine oli kandidaat. Kõik need mõõdikud on olemasolevas katsete CSV-s.

Panama 2020. aasta prognoosid arvutati pärast valiku salvestamist. Varem nähtud perioodi ei nimetata pimedaks kinnituseks. Eesti jaoks ei valitud seadistusi uuesti. Andmete saadavuse audit tehti enne Eesti prognoosivigade arvutamist: 2025. aasta väljavõte sisaldas palju puuduvaid tarbimisväärtusi, mistõttu valiti eelnev täielikum 2024. aasta. Selleks ajaks olid meetod ja kontrollid Panama põhjal fikseeritud. Eesti sisendivastendus jätab puuduvad ilma- ja pühamärgised välja; seega kontrollitakse meetodi ülekannet, mitte täpselt sama Panama tunnusevektori ülekannet.

Mõlema uuringu lõppprognoose ei muudetud pärast tulemuste nägemist. Eesti puudulike akende tõttu täpsustati pärast esimest arvutust usaldusvahemike rakendust: ajaplokk järgib täielikku tunnikalendrit ning puuduvad prognoosiaknad on maskitud. Sama parandus rakendati mõlemale andmestikule fikseeritud 5000 korduse ja samade plokipikkustega. See muutis ebakindluse arvutust, mitte prognoose ega mudelivalikut; muudatus on manifestis dokumenteeritud.

**Tulemuse tõlgendus.** Põhiaruanne sisaldab kõiki lõppvõrdlusi ja korrigeeritud vahemikke. Tugevaim XAI-spetsiifiline kontroll on sama parandaja sama tagasiside koosmõju arvestava ja koosmõjuta variandi vahel. Inimese arusaadavuse paranemist pole selles katses mõõdetud. Nii meetodi uudsus, prognoositäpsuse kasu kui ka kaardi kasu inimesele jäävad eraldi väideteks.

Minu soovitus on ehitada töö keskne panus küsimuse ümber: **kas veale suunatud selgitus aitab inimesel või fikseeritud valikureeglil valida piiratud katsete arvuga paranduse, mis vähendab viga ka diagnoositud olukorras?** Selle küsimuse jaoks on projektis juba tugev alus. Kõige paljulubavam järgmine kombinatsioon on veafunktsiooni selgitamine, sarnaste veajuhtumite kasutamine ja muudatuse kontroll enne kasutuselevõttu. Selle kombinatsiooni esmakordsus ei ole praeguse otsinguga tõendatud.

**Mida läbi vaatasin**

Lugesin põhiaruande, kõigi kümne märkmiku analüüsiteksti, katsete konfiguratsioonid, tunnuste koostamise, SHAP-i, analoogide, kaartide, muudatuste valiku ja veaparandaja teostused. Vaatasin üle avalikud koondid ning kohapeal säilinud põhikatse, kaartide ja abimudeli tulemused.

`trials.csv` sisaldab 208 tulemusrida: tunnuste sõelumine 44, mudelivalik 40, muudatuste valik 12, muudatuste lõpphindamine 6, katse-eelarve võrdlus 24 ja abimudeli katsed 82. Need pole 208 sõltumatut eksperimenti: read sisaldavad ka korduvaid aastaid, hindamisrühmi ja sama prognoosiga valikumeetodeid.

Käsk `python -B tools/check_results.py --full` läbis olemasolevate koondite, põhikatse, abimudeli ja kasutajakatse materjalide kontrollid. Laiendatud mudelivõrdluse ja eelarveanalüüsi detailsed kohalikud väljundid puuduvad; nende puhul sain hinnata säilinud koonde, aruannet ja teostust, kuid mitte kõiki tunniprognoose uuesti kontrollida.

**Seniste katsete sisuline järeldus**

| Katse | Oluline tulemus | Mida järgmine katse peaks lahendama? |
|---|---|---|
| Algsed baasid ja ajaline kontroll | Õppiv mudel ületab lihtsaid prognoose; viga sõltub ajast ja olukorrast. | Võrdlus peab sisaldama tugevat õppivat mudelit, mitte ainult lihtsat baasi. |
| 14 nimelist tunnusekomplekti, neist 13 sisuliselt erinevat, ning 8 eemaldamiskatset | 2017–2018 keskmine HistGBR-i MAE: baas 14,2640; ajalugu ja kalender 13,8780; kõik tunnused 13,9002. | Rohkem tunnuseid ei taga paremat mudelit. Uurida kontekstist sõltuvat vajalikkust. |
| Viis mudeliperekonda | 2020 MAE: Ridge 17,7844; HistGBR 17,9449; ExtraTrees 22,8697. Ridge'i eelis HistGBR-i ees pole korrigeeritud võrdluses tõendatud. | Uus lahendus peab võistlema Ridge'i ja HistGBR-iga samadel tundidel. |
| Algne XAI ja veapõhise valiku võrdlus | Kahe katse eelarvega jõudsid kõik meetodid samade lõppvalikuteni. | Vajame valikuülesandeid, kus diagnoosidel on võimalus sisuliselt erineda. |
| Koormusmuutuste tunnused | 2020 üldine MAE 24,8427 → 22,8779, kuid kiire kasvu ja languse MAE 37,2407 → 38,9260. | Parandust hinnata eraldi nii üldiselt kui ka probleemsetes olukordades. |
| Temperatuurimuudatus | Üldine MAE 24,8427 → 24,8423. | Ilma tähtsust ei saa samastada selle lisainfoga ühe tunni horisondil ja 24 tundi viivitatud sisendina. |
| Ühe katse eelarve | Kiirete muutuste puhul valis veapõhine meetod jooksva statistika, XAI koormusmuutused. 2020 vastavad olukorra MAE-d 36,4200 ja 38,9260; üldine MAE soosib teist valikut. | Praegune XAI järjestus pole tugevaim lähtepunkt. Olukorraspetsiifiline kasu tuleb skooris selgelt määratleda. |
| Vastuvõtureegel | Mõlemad tunnusemuudatused lükati arendusperioodil tagasi. | Uurida muutmise kasu kõrval ka põhjendatud muutmata jätmist. |
| 17 abimudeli ja paranduse varianti | Parim arendusvalik `residual_hist_1.0`: 2020 üldine MAE 24,8427 → 17,7564; kiire kasvu MAE 42,3448 → 25,9335. | Parandaja kasu on paljulubav; XAI osa selle kasu saavutamisel on endiselt mõõtmata. |
| Tugevam kontroll abimudelile | Sama 23 tunnusega HistGBR-i üldine MAE 18,2528; kiire kasvu MAE 31,5972. Parandaja üldise MAE eelis pole korrigeeritud intervalliga kindel; kiire kasvu MAE eelis on. | Hoida lisatunnused, põhiprognoos ja treeningread kontrollvõrdlustes võimalikult võrreldavad. |
| Kaardid ja stabiilsus | 12 kaarti, neist 2 hüpoteesiga. Globaalsete tähtsusjärjestuste stabiilsus on kõrge. Päris kasutajavastuseid pole. | Mõõta kohaliku soovituse õigsust, inimese valikut ja selgitusest loobumise asjakohasust. |

Kiire kasvu 72 tundi ja kiire kasvu ning languse 209 tundi on erinevad hindamisrühmad. Nende tulemusi ei tohi sama sündmuse mõõdikuna kokku panna. Samuti erinevad algse ja laiendatud uuringu GBR-id tunnuste ning treeningukorralduse poolest.

**Täiendav kirjeldav leid olemasolevatest prognoosidest**

Arvutasin faili `results/runs/20261102T161235294142Z/refinements/test_specialist.csv` 4273 rea põhjal suunatud vigade autokorrelatsioonid. Algmudeli viga on `baseline_prediction − target_next_hour`; parandatud viga on `prediction − target_next_hour`. Kasutasin pandas Series.autocorr-i vastava tunninihkega.

| Näitaja 2020 olemasoleval perioodil | Algne GBR | Parandatud prognoos |
|---|---:|---:|
| Vea autokorrelatsioon, 1 tund | 0,556 | 0,234 |
| Vea autokorrelatsioon, 24 tundi | 0,521 | 0,379 |
| Vea autokorrelatsioon, 168 tundi | 0,324 | 0,312 |

Parandatud mudeli keskmine viga on sihttunnil 9 ligikaudu +20,970 ja sihttunnil 19 −15,449, kummaski 178 vaatlust. Parandatud absoluutviga on algmudelist väiksem umbes 66,9% tundidest. Jaanuaris ja veebruaris ületab parandaja sama 23 tunnusega HistGBR-i MAE järgi, märtsis–mais jääb talle alla; juunis on taas parem. See soosib ajast ja kontekstist sõltuva paranduse uurimist.

Need on juba nähtud testiperioodi kirjeldused. Autokorrelatsioon ei taga uut prognoositavat lisakasu ega tõenda vea põhjust. Samad analüüsid tuleb teha arendusperioodil enne uue mudeli seadistamist. Andmete erinevust 2020. aastal ei nimetata siin tõendatud COVID-i mõjuks.

Tegin ka olemasoleva 2018. aasta `decision_evidence.csv` põhjal skoori komponentide eemaldamise kirjeldava kontrolli, uusi mudeleid treenimata. Täisskoori esimesed valikud on kiire muutuse korral `ramp` ja temperatuurijuhtumil `temperature_time`. Liikme `0,25 × matched_shap_contrast × shap_candidate_coupling` eemaldamisel muutuvad need vastavalt `rolling` ja `temperature_nonlinear`. Ainult PFI-liikme eemaldamine muudab kiire muutuse valiku `rolling`-uks. Kaartide või analoogide liikme ühekaupa eemaldamine ei muuda kummagi juhtumi esimest valikut. See ei eemalda SHAP-i kogu mõjust ega kogu kaardiloome ahelast; kontrollisin just salvestatud skooriliikmeid. Kahe kandidaadi eelarvega jääb kõigis neis kontrollides mõlema juhtumi esimese kahe kandidaadi hulk samaks. Tulemus näitab komponentide mõju praegusele järjestusele, mitte nende üldist kasu või kasutust. Praeguste kaalude ning kolme kandidaadi põhjal ei ole põhjendatud järeldada, et kaardid või analoogid on üldiselt kasutud.

**1. Veafunktsiooni selgitamine koos sihitud parandusvalikuga — soovitan keskseks meetodikatsesuunaks**

Praegune `explain_prediction()` selgitab mudeli väljundit. Suur `load_now` panus võib tähendada täiesti õiget prognoosiloogikat ka siis, kui prognoos eksib mingil muul põhjusel. Kõige suurema prognoosipanusega tunnuse järgi paranduse valimine võib seetõttu viia vale muudatuseni.

Katsetaksin pärast tegeliku koormuse selgumist konkreetse juhtumi absoluut- või ruutvea selgitamist: `h_y(x) = |f(x) − y|` või `(f(x) − y)^2`. See on tagasivaateline diagnostika. Eraldi variant on ajaliselt treeninguvälistel vigadel õppiv riskimudel `g(x) ≈ E[|Y − f(X)| | X=x]`, mida saab selgitada ka enne sihtväärtuse selgumist. Need variandid vastavad erinevatele küsimustele ja nende tulemusi tuleb eraldi nimetada.

Tehniline detail: fikseeritud y korral lihtsalt suunatud vea `f(x) − y` selgitamine annab sama SHAP-panuste jaotuse kui f(x), sest lahutatakse konstant. See ei oleks sisuline uus diagnostika. Regressiooni ruutvea selgitamist ei saa ka automaatselt samastada TreeExplainer-i klassifitseerimise `log_loss` valikuga. Vajalik on vastava veafunktsiooni toetatud selgitaja või mudelist sõltumatu teostus.

Võrdleksin sama kandidaatide hulga ja eelarvega: ainult veaanalüüs; veaanalüüs koos praeguse prognoosi-SHAP-iga; veaanalüüs koos veafunktsiooni selgitusega. Seejärel lisaksin ühes eraldi võrdluses analoogid. Selgitused arvutada ajaliselt eraldatud diagnoosiandmetel; valikureeglid fikseerida enne hindamist. Korrelatsioonis ajalootunnuseid käsitleda rühmadena ning kontrollida häirimisel, et ei tekiks võimatuid tunnusekombinatsioone.

Mõõdikud: paranduse tulevane MAE, probleemse olukorra MAE, halvima ette määratud olukorra viga, kahjulike muudatuste osakaal ja vajalik katsete arv. Uudsuse kandidaat on **veafunktsiooni selgituse mõõdetud lisakasu parandusvalikul**, mitte veafunktsiooni SHAP ise: selle jaoks on olemas [Chen, Lundberg ja Lee töö](https://pmc.ncbi.nlm.nih.gov/articles/PMC9349278/).

**2. Ainult varem selgunud vigadest õppiv ajaparandus — kõige põhjendatum kiire täpsuskatse**

Praegune parandaja kasutab koormusajalugu, kalendrit, põhiprognoosi ja selle eeldatavat muutust. Selle sisendites pole eraldi põhimudeli viimase tunni või eelmise päeva sama tunni teadaolevat viga. Säilinud 24 ja 168 tunni veasõltuvus annab nende proovimiseks konkreetse põhjuse.

Katsetaksin prognoosile lisatava paranduse sisenditena `e_t`, `e_(t−1)`, `e_(t−23)` ja `e_(t−167)` ning varasemate vigade jooksvaid keskmisi. Prognoositava tunni t+1 suhtes on eelmise päeva sama tunni teadaolev viga e_(t−23). Kõigi vigade tegelik avaldamisaeg peab jääma prognoosihetke sisse. Treeningtunnused tuleb koostada ajalooliselt väljastatud prognoosidest või ajaliselt treeningust välja jäetud prognoosidest; jooksva mudeli tagantjärele arvutatud treeningsobitusvead ei sobi.

Esimesed võrdlused võiksid olla tavaline Ridge-parandaja, HistGBR-parandaja ning lihtne sujuvalt uuenev nihkeparandus. Siis kontrollida, kas piiratud parandus või ainult teatud olukorras parandamine kaitseb tunde, kus algprognoos oli parem. Kasutada sama sisendihulka ka otsese prognoosi kontrollmudelis, et eristada arhitektuuri kasu lisainfo kasust.

XAI roll oleks näidata ja kontrollida, kas soovitus tugineb korduvale ajamustrile, värskele nihkele või koormusdünaamikale. Koormusprognoosi veaparandajad on juba olemas, näiteks [2026. aasta koostööl põhinev prognoosimudel](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2026.1966347/full). Seega ootan siit eeskätt täpsuse paranemise võimalust, mitte eraldiseisvat esmakordsust.

**3. Sarnaste selgituste põhjal leitud veamustrid ja paranduste mälu — suurema uudsuspotentsiaaliga kombinatsioon**

Praegused olukorrad on käsitsi määratud: kiire kasv, kuumus, püha jne. Võib leiduda varjatud olukord, näiteks tööpäeva hommik, ebatavaline nädalaprofiil ja hiljutine tasememuutus, mille puhul mudel kasutab valesti sama ajalooallikat.

Rühmitaksin diagnoosiperioodi juhtumeid võrdlevalt kolmes ruumis: sisendtunnused; prognoosi SHAP-panused; veafunktsiooni selgitused. Kõigis variantides peab klastrite arv ja keerukuse eelarve olema võrreldav. Ainult ilus või väikese hajuvusega klaster ei ole edu: rühm peab hilisemal perioodil säilitama veamustri ja võimaldama kasuliku paranduse valimist.

Iga varem uuritud rühma juurde talletaksime katsetatud parandused ja nende tegeliku kasu või kahju hilisemal arendusplokil. Uue diagnoosi puhul saab küsida: milline muudatus aitas sarnase selgitusmustriga juhtumeid? See muudab olemasolevad analoogid passiivsest kontekstist kontrollitud parandusajalooks. Inimene võib kinnitada või vaidlustada nii analoogi sobivuse kui ka pakutud muudatuse.

Tulevase tunni tegelikku viga või veafunktsiooni selgitust ei tohi kasutada selle sama tunni parandaja suunamiseks. Reaalajas suunamine vajab prognoosihetkel kättesaadavat rühmakuuluvuse mudelit; alternatiiv on hoida meetod täielikult hilisema diagnoosi tööriistana. SHAP-panused on sisendite ja mudeli deterministlik teisendus, mistõttu võimalik eelis tuleb paremast esitusest või valikust, mitte uue vaatlusinfo tekkimisest.

Selgitusruumis rühmitamine on juba tuntud [haiguste alamtüüpide leidmisest](https://arxiv.org/abs/1912.08755), samuti on olemas [mudeli nõrkade alarühmade leidmine mõjuesituste kaudu](https://arxiv.org/abs/2312.04712). Meie uudsuse kandidaat oleks selle ühendamine elektrikoormuse **veadiagnoosi, ajaliselt korrektse parandusmälu ja võrdse eelarvega valikuvõrdlusega**. Otsing ei tõenda, et sellist kombinatsiooni pole varem avaldatud.

**4. Teadaolevate sisendivigadega diagnostikakatse — tugevaim viis kontrollida, kas diagnoos tabab midagi tegelikku**

Loomulike koormusvigade puhul pole meil üldjuhul teada tegelikku põhjust. Seetõttu saab praegu kontrollida kaardi arvude õigsust, kuid palju raskem on hinnata diagnoosi õigsust.

Looksime eraldi katsekeskkonna päris koormusandmete koopial: lisame mõneks perioodiks koormusmõõtmise viivituse, nihutame kalendritunnuse ajajoondust, tekitame ilmakanali nihke või piirame kasutatavat hooajalist ajalugu. Sihiks olev tegelik koormus säilib. Tunnused tuleb häiritud lähteandmetest uuesti arvutada, et üksikud viitajad ja jooksva akna tunnused ei läheks omavahel vastuollu. Lisada ka puhtad kontrolljuhtumid.

Meetod näeb probleemi ja oma tavapäraseid tõendeid, kuid ei näe sekkumise märgendit. Hinnata, kas ta leiab kahjustatud andmekanali, pakub piiratud eelarves mõistliku paranduse ja jätab puhta juhtumi rahule. Testis kasutada ka diagnoosi arendamisel nägemata kestusi, tugevusi ja vähemalt üht mitme vea kombinatsiooni. Kõiki paranduskandidaate ei tohi kujundada nii, et iga häiring annab ilmselgelt ühe ette kirjutatud vastuse.

Seda katset tuleks nimetada kontrollitud sisendihäirete diagnostikaks, mitte tõendiks loomulike prognoosivigade põhjuslikust lahendamisest. Ka kogu mudeli viga ei pruugi tuleneda lisatud häiringust. Tulemused esitada loomulike andmete tulemustest eraldi. Üldine selgituste diagnostikavõime testimine on olemas [Adebayo jt töös](https://arxiv.org/abs/2011.05429); valdkondlik parandusvaliku katsekeskkond võiks olla meie panuse üks osa.

**5. Paranduse väärtuse hindamine ja põhjendatud muutmata jätmine**

Praegune vastuvõtureegel vaatab üldist ja sihtolukorra MAE-d, parandaja aga töötab kõigil tundidel. Pakun nende ühendamist: valida arendusandmetel, millal parandus annab piisava tõendi kasust, ja millal jätta algprognoos alles. Inimesele esitataks kasu kõrval ka olukorrad, mida muudatus võib halvendada.

Kõigepealt võrrelda alati rakendatavat parandust, lihtsat teadaoleval koormusmuutusel põhinevat suunamist ja prognoosihetke kontekstist õpitud pehmet osakaalu. Siis testida, kas selgituste lisamine parandab sama keeruka suunaja otsuseid. Hindamiseks kasutada kogu ajajoone prognoose: ainult meetodi enda valitud lihtsate tundide MAE võib jätta eksitava mulje. Parandusest loobumisel tuleb samuti väljastada algprognoos ning arvestada selle viga.

Valiku ebakindlust saab hinnata eraldi arendusplokkidel. Konformaalne ennustus võib aidata kirjeldada tulevase prognoosivea ebakindlust, kuid see ei anna iseenesest garantiid, et mudelimuudatus parandab viga. Ka aegrea alarühmade katvust ei saa automaatselt järeldada üldisest katvusest. Kui see haru lisada, on asjakohane lähtekoht [Gibbs ja Candès, 2024](https://www.jmlr.org/papers/v25/22-1218.html).

Võimalik teaduslik väärtus on väiksem kahjulike muudatuste osakaal sama või väiksema üldvea juures. Eesmärk pole sundida meetodit alati midagi muutma.

**6. Inimene valib tegeliku paranduse ning tema valikut hinnatakse tulevastel andmetel**

Olemasolev piloot mõõdab kaartidest arusaamist ja järgmise sammu põhjendatust. Uus ettepanek laiendaks seda parandusvaliku uuringuks; see ei kuulu praeguses aruandes kirjeldatud valmis piloodi lubatud järelduste hulka.

Ülesandes annaksime sama diagnoosijuhtumi kohta kas veatabeli, veatabeli koos praeguse kaardiga või veafunktsiooni selgituse ja varasemate paranduste tõenditega kaardi. Osaleja valib ühe paranduse või jätab mudeli muutmata. Valikule vastava mudeli hilisem prognoosiviga on objektiivne tulemus. Eelnevalt treenitud kandidaatide tulemusi võib kasutada peidetud hindamiseks, et osalejad ei peaks treenimist ootama; neid tulemusi ei näidata valiku tegemisel.

Põhinäitaja võiks olla valitud paranduse kasu ja erinevus parimast lubatud kandidaadist, koos kahjulike valikute osakaaluga. Aeg ja kindlus on lisamõõdikud. Sama inimene ei peaks nägema sama juhtumit eri esitustes; esitusjärjekord ja juhtumite raskus tasakaalustada. Sõltumatute osalejate arv jääb osalejate arvuks, mitte vastuste arvuks.

See seoks pealkirja human-centered osa otseselt mudeli parandamise tulemusega. Väike piloot annaks esmase, piiratud tõendi; tugevat üldistusväidet vajav valim tuleb kavandada eraldi.

**7. Kaks odavat lisavõrdlust enne keerukama arhitektuuri ehitamist**

Esimene on õppida järgmise tunni koormuse asemel selle muutust: `delta = y_(t+1) − y_t`, väljundiks `y_t + delta_hat`. See sobib praeguse kasvul ja langusel nähtava suunatud vea uurimiseks. Võrrelda samu tunnuseid, parameetrite otsingu eelarvet ning otsest sihtmärki. Ümberkirjutus ise pole uus meetod ja võib ka halvendada tulemust.

Teine on võrrelda paisuvat treeningajalugu fikseeritud viimase 12 või 24 kuu aknaga ning ajaliselt kahanevate kaaludega. Akna valik toimub ainult arendusandmetel. See kontrollib, kas osa nõrkusest tuleneb vananenud koormusrežiimist. Kui lihtne ajakaalumine lahendab sama probleemi, peab XAI-põhine lahendus selle tugeva kontrolliga võistlema.

Need on täpsuse kontrollkatsed, millest ei tasu eraldi esmakordsusväidet teha. Samuti aitavad need hinnata, kui palju keerukust magistritöö keskne meetod tegelikult vajab.

**Milline osa on kirjanduse järgi juba olemas?**

| Varasem töö | Mõju meie uudsusväitele |
|---|---|
| [Robin jt, 2024](https://cris.technion.ac.il/en/publications/evaluating-model-performance-through-a-user-centric-explainable-f/) | Panama, GBR, PFI, SHAP ja kasutaja vajadustele kohandatud selgitused on väga lähedane lähtekoht. Kontrollisin abstrakti ja bibliokirjet, mitte täisteksti. |
| [Park, Lee ja Kim, 2026](https://file.techscience.com/files/onlinefirst/2026/4.7/TSP_CMC_79734/TSP_CMC_79734.pdf) | SHAP-iga suunatud interaktsioonitunnused ja kordustreening on juba olemas. Vaatasin PDF-i meetodit; vaatamata pealkirja short-term sõnale kasutab kirjeldatud andmestik kuu tasemele agregeerimist. |
| [Improving Load Forecasting With Feature Selection via XAI and Expert-Guided Prompting, 2025](https://ieeexplore.ieee.org/document/11270832/) | SHAP/LIME ning LLM-i kaudu lisatud eksperditeadmise kombinatsioon pole enam uus üldidee. Kasutasin indekseeritud abstrakti; otse avades oli täisteksti ligipääs piiratud. |
| [Collaborative short-term load forecasting…, 2026](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2026.1966347/full) | Väiksem veaparandaja ja suunatud paranduse tugevus on juba uuritud. |
| [Chen jt, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9349278/) | Mudeli veafunktsiooni selgitamine ja mitmest mudelist koosneva ahela selgitamine pole meie leiutis. |
| [Schulz jt, 2019/2020](https://arxiv.org/abs/1912.08755), [Wang jt, 2023](https://arxiv.org/abs/2312.04712) | Selgituste või mõjuesituste abil rühmade ja mudeli nõrkuste leidmisel on olemas lähtealused. |
| [Adebayo jt, 2020](https://arxiv.org/abs/2011.05429) | Selgituse diagnostilist kasu peab eraldi kontrollima; atribuutide kuvamine ei tõenda mudelivea leidmist. |

Otsing oli sihitud, mitte süstemaatiline kirjanduse ülevaade. Avaldamata või muude terminitega kirjeldatud lähedased kombinatsioonid võivad jääda leidmata. Esmakordsuse asemel on praegu põhjendatud rääkida uuritavast kombinatsioonist ja rangemast empiirilisest hindamisest.

**Soovitatav katsete järjekord ja piiratud maht**

| Samm | Töö | Millise küsimuse lahendab? |
|---|---|---|
| A | Taasta sama sisendiga tugevad Ridge'i ja HistGBR-i kontrollid; proovi ainult varasemate vigadega ajaparandust ja koormusmuutuse sihtmärki. | Kui palju parandab lihtne prognoosimismeetod? |
| B | Võrdle veapõhist, prognoosi-SHAP-i ja veafunktsiooni selgitusega parandusvalikut. | Kas selgituse tüüp parandab valikut? |
| C | Lisa ühes eraldi võrdluses varasemate paranduste mälu. | Kas analoogid annavad lisakasu? |
| D | Kontrollitud sisendihäired ning puhtad juhtumid. | Kas diagnoos ja muutmata jätmine tabavad teadaolevaid probleeme? |
| E | Fikseeritud meetodi kontroll uuel andmestikul ja võimaluse korral päris inimese parandusvaliku piloot. | Kas tulemus kandub üle ning aitab inimest? |

Esimeses keskse meetodi katses piisaks umbes 6–8 sisuliselt erinevast paranduskandidaadist, muutmata jätmise võimalusest ning eelarvetest 1, 2 ja 4. Kandidaadid ja nende lubatud seadistuste arv tuleb enne hindamist lukustada; eelarve peab sisaldama kogu valikuprotsessis kasutatud mudelite treenimist. Kasutada ka juhuslikku kandidaatide järjestust kordustega, mille hindamistulemusi ei kasutata parima juhuseemne valimiseks. Pärast kõigi kandidaatide hindamist saab tagasivaateliselt arvutada parima lubatud paranduse võrdluspiiri, kuid seda ei tohi esitada päris kasutuses kättesaadava meetodina.

Lisaks täpsusele salvestada diagnoosi, SHAP-i, treenimise ja inimese otsuse ajakulu. Selgituskomponentide eemaldamiskatsed teha ükshaaval; praeguse koondskoori ja tunnuserühmade eemaldamiskatsete põhjal ei saa SHAP-i, PFI, analoogide või kaartide üksikpanust eraldada. Negatiivse kontrollina võib algoritmilises katses segada selgitusi võrreldava ajakonteksti sees ja kontrollida, kas soovitused ning kasu üldse muutuvad.

Kogu senine Panama periood jääb arenduseks ja uurivaks hindamiseks. Uus juhuseeme ega 2020 ümberjaotamine ei anna vaatamata testi. Teise süsteemi kandidaat võiks olla Eesti tarbimine [Eleringi ametlikust andmeallikast](https://www.elering.ee/elektri-tarbimine-ja-tootmine). Selle sobivus tuleb enne katse alustamist kontrollida: ajasamm, puuduvad väärtused, ühikud, kohalik kalender, suveaeg ja info avaldamise viivitus. Eleringi avalike andmete olemasolu ei tähenda, et kõik Panama tunnused oleksid seal samal kujul saadaval. Teine riik kontrollib meetodi ülekantavust; sama süsteemi hilisemad andmed kontrolliksid selle tulevast toimimist.

Lõpphindamisel raporteerida üldine MAE/RMSE, kiire kasvu ja languse MAE eraldi, ette määratud halvima olukorra viga, märginihe ning kahjulike muudatuste osakaal. Kiire kasvu 72 tundi ei ole automaatselt 72 sõltumatut sündmust. Võrdluste ebakindlus arvutada ajasõltuvust arvestades; uuriva otsingu kogu mõju ei kao mitmese testimise parandusega.

**Milline tulemus oleks märkimisväärne?**

Kõige veenvam tulemus oleks näidata uuel andmestikul või puutumata perioodil, et veafunktsiooni selgituse ja parandusajaloo kombinatsioon leiab sama katsete arvuga paremaid parandusi, vähendab probleemse olukorra viga ja tekitab vähem kahjulikke muudatusi kui tugev veapõhine valik. Kui lisaks valivad inimesed selle abil parema paranduse või oskavad põhjendatumalt muutmata jätta, toetaks see otseselt kogu töö pealkirja.

Väiksem, kuid samuti sisukas panus oleks kontrollitav diagnostikakatsekeskkond, milles näidatakse täpselt, milliste vigade puhul XAI aitab ja milliste puhul ei aita. Üldise MAE paranemine, praktiline abi inimesele ja meetodi uudsus on kolm eraldi väidet; igaüks vajab oma tõendit.
