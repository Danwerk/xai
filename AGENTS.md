# Projekti tööpõhimõtted

Arenda seda magistritöö praktilist osa lihtsa, märkmikukeskse
uurimisprojektina. Lähtu uurimisküsimustest, andmetest ja kasutaja
ülesandest. Kohanda analüüsimeetodeid ja katsete ülesehitust vastavalt
tegelikule vajadusele.

## Projekti struktuur

- Peamine töö toimub nummerdatud Jupyter notebook'ides projekti juurkaustas.
- data/ sisaldab sisendandmeid ja vajaduse korral töödeldud andmeid.
- utils/ sisaldab jagatud abifunktsioone, sealhulgas visuaalse stiili abilisi.
- scripts/ sisaldab vajaduspõhiseid käivitamis- ja abiskripte.
- README.md annab projektist lühikese ülevaate.
- Kasuta olemasolevaid faile ja nimetusi, kui need sobivad ülesandega.
- Lisa uusi faile ja kaustu ainult sisulise vajaduse korral.
- Hoia lahendus uurimisprojekti mahu ja keerukusega proportsioonis.

## Töö jaotamine notebook'ideks

- Jaga töö sisulisteks etappideks.
- Võimalikud etapid on andmete uurimine, ettevalmistus, tunnuste loomine,
  meetodite katsetamine, võrdlemine ja tulemuste analüüs.
- Vali etapid uurimisküsimuse järgi; kõik projektid ei vaja kõiki etappe.
- Kasuta nummerdatud ja sisulisi failinimesid,
  näiteks 00_data_analysis.ipynb ja 01_data_pipeline.ipynb.
- Lisa uus notebook eraldi uurimisküsimuse või katseetapi jaoks.
- Sama etapi täiendused tee olemasolevasse notebook'i.
- Säilita olulised vahekatsed ja nende põhjal tehtud valikute põhjendused.
- Ära loo tühje notebook'e ainult etteantud jaotuse täitmiseks.

## Notebook'i ülesehitus

- Hoia eesmärk, metoodika, kood, tulemused ja tõlgendus koos.
- Alusta eesmärgi või uurimisküsimuse lühikese kirjeldusega.
- Kirjelda analüüsi mõistmiseks vajalikke sisendandmeid ja sõltuvusi.
- Hoia olulised seadistused nähtaval.
- Jaga analüüs Markdowni pealkirjade ja lühikeste selgitustega sammudeks.
- Kuva olulised tabelid, mõõdikud ja graafikud otse notebook'is.
- Selgita tulemuste tähendust, piiranguid ja tehtud valikuid.
- Hoia põhianalüüs käivitatavana ülevalt alla värskes kernelis.
- Ära tugine peidetud kerneliolekule ega kasuta salvestatud
  lahtriväljundeid arvutuste sisendina.

## Koodi paigutamine

- Notebook võib sisaldada kogu oma analüüsiks vajalikku koodi ja funktsioone.
- Tõsta utils/ alla loogika, mida kasutatakse mitmes notebook'is
  või mille eraldamine parandab selgelt loetavust.
- Väldi sama arvutusloogika kopeerimist eri failidesse.
- Lisa scripts/ alla skript konkreetse vajaduse korral.
- Skriptiga tehtud analüüsi eesmärk ja tulemused peavad olema
  vastavas notebook'is arusaadavalt esitatud.
- Väldi tarbetuid klasse, abstraktsioone ja konfiguratsioonikihte.
- Kasuta projekti suhtes määratud failiteid ja väldi arvutipõhiseid
  absoluutseid failiteid.

## Andmed

- Hoia projekti sisendandmeid data/ kaustas, kui ülesandes pole määratud teisiti.
- Säilita algandmed muutmata.
- Vajaduse korral salvesta töödeldud andmed data/ alla,
  et järgmised notebook'id saaksid neid kasutada.
- Kirjelda andmeallikaid ja olulisi töötlussamme vastavas notebook'is.
- Too vajaduse korral välja ühikud, ajaperiood, puuduvad väärtused
  ja analüüsi mõjutavad andmepiirangud.
- Välista mahukad või privaatsed kohalikud andmed Gitist .gitignore abil.

## Analüüs ja katsed

- Lähtu uurimisküsimusest ning vali sellele sobivad meetodid ja mõõdikud.
- Kirjelda katse eesmärk ning võrreldavate variantide erinevused.
- Kasuta sobivuse korral lihtsat võrdlusmeetodit või võrdlusmudelit.
- Hoia võrdlustingimused kooskõlas või põhjenda nende erinevusi.
- Mudelite sobitamisel erista õppimiseks, valikuks ja lõplikuks
  hindamiseks kasutatavaid andmeid ülesandele sobival viisil.
- Väldi andmeleket; sobita õpitavad eeltöötlused treeningandmetel.
- Ajaliste andmete puhul arvesta ajalise järjestuse ja prognoosihetkel
  tegelikult kättesaadava teabega.
- Juhuslikkust kasutavates katsetes määra võimaluse korral juhuseeme.
- Säilita olulised positiivsed, negatiivsed ja neutraalsed tulemused.
- Põhjenda meetodite ja seadistuste valikuid saadud tõenditega.
- Erista mõõdetud tulemus, tõlgendus ja oletus.
- Järelduste ulatus peab vastama katsekorraldusele ja andmetele.

## Tulemuste hoidmine

- Hoia katsetulemused notebook'ides.
- Kuva mõõdikud, tabelid ja graafikud koos nende tõlgendusega.
- Salvesta notebook koos oluliste väljunditega.
- Ära loo results/ kausta.
- Ära salvesta eraldi tulemuste tabeleid, mudelifaile,
  treeningu kontrollpunkte ega graafikute ekspordifaile,
  kui kasutaja seda ei palu.
- Vajadusel arvuta tulemused ja treeni mudelid uuesti.
- Ära käivita mahukaid arvutusi pelgalt teksti või paigutuse muutmise tõttu.
- Andmete ettevalmistuse väljundtabelid võivad paikneda data/ all;
  katsete mõõdikud ja võrdlustulemused jäävad notebook'idesse.
- Vajaduse korral märgi notebook'is, millised tulemused valiti lõputöösse.

## Graafikute ja tulemuste visuaalne stiil

- Kasuta graafikute ja skeemide kujundamisel olemasolevat plotting.py moodulit.
- Vaikimisi kasuta with plot_style("taltech"): konteksti.
- Loo joonis ja selle elemendid stiilikonteksti sees.
- Lähtu TALTECH_COLORS värvipaletist ja TALTECH_FONT fondimäärangust.
- Hoia samade mudelite, kategooriate ja tunnuserühmade värvid
  võrreldavatel joonistel ühtsed.
- Kasuta sobivuse korral mooduli olemasolevaid joonistusabilisi.
- Ära dubleeri notebook'ides stiilisõnastikke.
- Vajaduse korral lisa korduvkasutatav joonistusabi plotting.py faili.
- Kohanda suurusi ja paigutust plot_style() overrides kaudu.
- Vali graafikutüüp ja värviskaala andmete tähenduse ning loetavuse järgi.
  Kui TalTechi põhivärvid ei sobi näiteks pideva või lahkneva skaala jaoks,
  kasuta sisuliselt sobivat skaalat ja säilita ülejäänud üldkujundus.
- Kohanda võimaluse korral ka teiste visualiseerimisteekide joonised
  sama üldise stiiliga.
- Vaikimisi ära lisa graafikule üldpealkirja.
- Kirjelda graafiku eesmärki notebook'i eelnevas Markdowni tekstis.
- Lõputöös lisatakse joonise number ja kirjeldav allkiri dokumendis.
- Säilita telgede nimetused, ühikud, legendid ja vajalikud annotatsioonid.
- Hoia arvude täpsus selge ja võrreldavatel joonistel järjepidev.
- Mitme paneeliga joonistel kasuta vajaduse korral lühikesi
  paneelipealkirju.
- Lisa üldpealkiri siis, kui kasutaja seda soovib või joonise
  eraldiseisev kasutus seda vajab.
- Vormindatud tulemuste tabelites järgi sama kujundusloogikat;
  tavalised andmetabelite väljundid võivad jääda vaikekujule.
- Kuva joonised notebook'is ja ekspordi neid ainult kasutaja soovil.

## Testid ja koodi kontrollimine

- Ära loo eraldi testifaile ega tests/ kausta.
- Ära lisa testiraamistikke ega kirjuta automaatteste,
  kui kasutaja neid eraldi ei küsi.
- Kontrolli muudatusi vajaduse korral olemasoleva notebook'i
  asjakohaste lahtrite käivitamisega.
- Notebook'is võivad olla analüüsiks vajalikud andmete ja arvutuste
  kontrollid, näiteks mõõtmete, puuduvate väärtuste ja ühikute ülevaatus.
- Vali kontrolli ulatus muudatuse mõju järgi.
- Ütle selgelt, mida käivitati ja mida ei käivitatud.
- Ära esita varasemaid salvestatud väljundeid värske käivituse tulemustena.

## Dokumentatsioon

- Hoia README.md lühike ja ülevaatlik.
- Kirjelda seal projekti eesmärki, keskkonna käivitamist
  ja peamiste notebook'ide ning kaustade rolle.
- Uuenda README-d siis, kui need põhiandmed muutuvad.
- Hoia andmete kirjeldused, metoodika, katseseadistused,
  tulemused ja valikute põhjendused vastavates notebook'ides.
- Väldi sama info kordamist README-s ja notebook'ides.
- Ära loo eraldi dokumentatsioonifaile ilma kasutaja soovita.

## Projekti keel

- Kasuta notebook'ide pealkirjades, Markdowni selgitustes,
  tulemuste tõlgendustes ja järeldustes inglise keelt.
- Kasuta inglise keelt koodi kommentaarides, docstring'ides,
  muutujate ja funktsioonide nimedes ning failinimedes.
- Graafikute teljed, legendid, annotatsioonid ja paneelipealkirjad
  ning tulemuste tabelite kuvatekstid peavad olema inglise keeles.
- Kirjuta README.md inglise keeles.
- Säilita lähteandmete algsed veerunimed ja väärtused;
  vajaduse korral lisa analüüsiks ingliskeelsed vasted.
- Kasutajaga suhtle tema kasutatavas keeles.