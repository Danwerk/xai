"""Loo rollipõhised kaardid ja täitmata piloodimaterjalid päris juhtumitest."""

from datetime import datetime, timezone
from os.path import relpath
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.card_views import build_card_view, render_card_view, evidence_id, ROLE_NAMES
from src.common import latest_run, read_json, sha256, write_csv, write_json, manifest_core_sha256
from src.evaluation import audit_card


def main():
    parent = latest_run()
    output = ROOT / "results/card_studies" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    index = []
    sources = {}
    for path in sorted((parent / "cards").glob("card_*/evidence.json")):
        card = read_json(path)
        if not audit_card(card)["all_pass"]:
            raise ValueError(f"Lähtekaardi arvud ei läbinud kontrolli: {path.name}")
        sources[str(path.relative_to(ROOT))] = sha256(path)
        case_id = path.parent.name
        source = Path(relpath(path.parent, output)).as_posix()
        index.append(
            {
                "case_id": case_id,
                "target_time": card["target_time"],
                "selection_reason": card.get("selection_reason", "Varasema käivituse juhtum."),
                "has_hypothesis": card["diagnosis"]["hypothesis"] is not None,
                "evidence_id": evidence_id(card),
                "user_card": f"{source}/user.md",
                "developer_card": f"{source}/developer.md",
                "technical_check": "passed",
            }
        )
    make_study_materials(output, pd.DataFrame(index))
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "study_status": "materials_ready_no_participants",
            "parent_run": str(parent),
            "parent_manifest_sha256": sha256(parent / "manifest.json"),
            "parent_core_sha256": manifest_core_sha256(parent / "manifest.json"),
            "source_sha256": sources,
            "artifact_sha256": {
                str(path.relative_to(output)): sha256(path)
                for path in output.rglob("*")
                if path.is_file()
            },
        },
    )
    write_json(
        ROOT / "results/latest_card_study.json",
        {"path": str(output), "manifest_sha256": sha256(output / "manifest.json")},
    )
    print(f"Kaardid ja piloodimaterjalid valmis: {output}")


def make_study_materials(output, index):
    hypotheses = index.loc[index.has_hypothesis, "case_id"].head(2).tolist()
    abstentions = index.loc[~index.has_hypothesis, "case_id"].head(2).tolist()
    if len(hypotheses) < 2 or len(abstentions) < 2:
        raise ValueError(
            "Kasutajakatseks on vaja kaht muudatuse ettepanekuga ja kaht ettepanekuta juhtumit."
        )
    write_csv(output / "card_index.csv", index)
    links = [
        f"- {row.case_id}: [kasutaja]({row.user_card}), [arendaja]({row.developer_card}) — "
        + ("muudatuse ettepanek" if row.has_hypothesis else "muudatust ei soovitata")
        for row in index.itertuples()
    ]
    (output / "README.md").write_text(
        "# Sama juhtum, kaks selgitust\n\nKaardid kirjeldavad 2018. aasta prognoose. Kasutajakatset pole veel tehtud.\n\n"
        + "\n".join(links)
        + "\n\nKaartide lingid viitavad põhikatse väljunditele. Siin on eraldi ainult katses kasutatavad lihtsad loendid.\n\nKasutajakatse ülesanded on [juhendis](study/instructions.md). Vastusefailid on tühjad. Näited on valitud kindlate olukordade uurimiseks, mitte juhuslikult.\n",
        encoding="utf-8",
    )
    cases = [hypotheses[0], abstentions[0], hypotheses[1], abstentions[1]]
    by_case = index.set_index("case_id")
    for case_id in cases:
        evidence = (output / by_case.loc[case_id, "user_card"]).parent / "evidence.json"
        card = read_json(evidence)
        folder = output / "cards" / case_id
        folder.mkdir(parents=True, exist_ok=True)
        link = Path(relpath(evidence, folder)).as_posix()
        for role in ROLE_NAMES:
            (folder / f"{role}_plain.md").write_text(
                render_card_view(build_card_view(card, role), "plain", evidence_link=link),
                encoding="utf-8",
            )
    assignments = []
    for role in ROLE_NAMES:
        for rotation in range(4):
            order = cases[rotation:] + cases[:rotation]
            for variant, layouts in enumerate(
                (
                    ("structured", "plain", "plain", "structured"),
                    ("plain", "structured", "structured", "plain"),
                )
            ):
                for position, (case_id, layout) in enumerate(zip(order, layouts), start=1):
                    assignments.append(
                        {
                            "role": role,
                            "sequence": 2 * rotation + variant + 1,
                            "position": position,
                            "case_id": case_id,
                            "layout": layout,
                            "material": (
                                f"cards/{case_id}/{role}_plain.md"
                                if layout == "plain"
                                else by_case.loc[case_id, f"{role}_card"]
                            ),
                        }
                    )
    write_csv(output / "study/assignments.csv", pd.DataFrame(assignments))
    write_csv(
        output / "study/responses.csv",
        pd.DataFrame(
            columns=[
                "participant_id",
                "role",
                "sequence",
                "position",
                "case_id",
                "layout",
                "error_description",
                "known_vs_later",
                "causal_claim_assessment",
                "next_step_and_reason",
                "seconds",
                "confidence_1_5",
                "clarity_1_5",
            ]
        ),
    )
    write_csv(
        output / "study/ratings.csv",
        pd.DataFrame(
            columns=[
                "participant_id",
                "case_id",
                "rater_id",
                "facts_0_2",
                "timing_0_2",
                "causality_0_2",
                "next_step_0_2",
                "comment",
            ]
        ),
    )
    instructions = """# Kasutajakatse juhend

Materjalid on valmis. Osalejate vastuseid veel pole.

Plaanis on kaheksa osalejat: neli prognoosi kasutajat ja neli mudeli arendajat. Igaüks lahendab neli juhtumit. Kahes pakutakse mudelimuudatust ja kahes ei pakuta. Osaleja näeb kahte juhtumit kaardina ning kahte sama rolli samade faktide loendina. Kaheksa osalejat on väikese prooviuuringu praktiline siht; vajalikku valimi suurust pole statistiliselt arvutatud.

Enne ülesandeid küsi osaleja töö ja prognoosivigade uurimiseks vajaliku info kohta. Selgita kõigile ühtemoodi prognoosi, vea ja SHAP-panuse tähendust. Ära harjuta katses kasutatavate juhtumite vastuseid.

Määra osaleja roll tema töö järgi. Vali enne esimest ülesannet järjekord failist assignments.csv. Jaga järjekorrad kummaski rollirühmas juhuslikult ja võimalikult võrdselt. Osaleja näeb iga juhtumit ühe korra, ainult talle määratud kujul. Ära anna talle kaartide sisukorda ega teisi vaateid.

Iga juhtumi kohta küsi:

1. Mis suunas ja kui palju prognoos eksis?
2. Mis oli prognoosi tegemise ajal teada ja mis selgus hiljem? Too üks näide kummastki.
3. Kas esitatud info tõendab vea põhjust? Põhjenda ühe lausega.
4. Mida teeksid järgmisena ja miks? Too üks toetav asjaolu ja üks info piirang. Kasutaja võib kirjeldada kontrollimist või arendaja poole pöördumist. Arendaja võib pakkuda katset või põhjendada, miks ta mudelit praegu ei muudaks.

Märgi kulunud aeg sekundites. Palu hinnata vastuse kindlust ja teksti selgust skaalal 1–5, kus 1 on madalaim ja 5 kõrgeim hinnang. Vastused kirjutatakse oma sõnadega; need ei pea kordama kaardi soovitust.

Osalemine on vabatahtlik ja katkestamist ei pea põhjendama. Ära kogu vastustega nime ega tööandja andmeid. Korraldaja küsib osalemiseks teadliku nõusoleku eraldi ning kontrollib enne värbamist asutuse uuringukorda. Vastused salvesta faili responses.csv, kasutades osalejatunnust.

Võrdle kaardi ja loendi tulemusi esmalt iga osaleja puhul, seejärel kummaski rollirühmas. Põhiline on vastuste õigsus ja järgmise sammu põhjendus. Aeg, selgus ja kindlus on lisatulemused. Märgi ka juhud, kus vastus on vale, kuid osaleja on selles kindel. Ühe inimese nelja vastust ei käsitleta nelja sõltumatu osalejana.

See prooviuuring hindab info arusaadavust ja ülesannete sobivust. Väike valim ei tõenda üldist kasutatavuse paranemist. Samuti ei mõõdeta, kas osaleja valitud muudatus parandaks prognoosi: selleks tuleks muudatused eraldi läbi katsetada. Rollirühmade erinevus ei tõenda rollile kohandamise kasu, sest rühmades on erinevad inimesed ja ülesanded.
"""
    (output / "study/instructions.md").write_text(instructions, encoding="utf-8")
    rubric = """# Hindajate juhend

Kaks hindajat hindavad vastuseid iseseisvalt, teadmata, kas osaleja nägi kaarti või loendit. Peida nende koopias veerud layout ja sequence. Arvude kontrollimiseks kasuta faili evidence.json. Säilita mõlema hindaja algsed hinded; lahkarvamused arutatakse läbi pärast nende salvestamist.

Iga mõõde saab 0, 1 või 2 punkti. 0 = vale või puudub; 1 = osaliselt õige; 2 = korrektne ja nõutud põhjendusega.

- Faktid: vea suund ja suurus on õiged, arvestades arvude ümardamist.
- Info ajaline kättesaadavus: enne prognoosi teada olnud sisend eristatakse hiljem selgunud koormusest või aasta koondandmetest.
- Vea põhjus: SHAP-panust ei peeta tõendatud vea põhjuseks. Vastus nimetab asjakohase piirangu.
- Järgmine samm: sobib osaleja rolliga ja on esitatud info põhjal põhjendatud. Mudeli muutmata jätmine võib olla õige valik. Kaardi soovitusega ei pea nõustuma, kui põhjendus sobib andmetega.

Liida nelja vastuseosa punktid: kokku 0–8 iga juhtumi kohta. Tulemusi käsitletakse kummaski rollirühmas eraldi. Kommentaarid, aeg ja vastuse kindlus on lisatulemused. Ühe inimese vastused ei ole sõltumatud osalejad.

Enne hindamist täpsusta juhend näidisvastustega ja kontrolli hindajate kooskõla. Hilisemad muudatused dokumenteeri; prooviuuringu järel muudetud juhendit ei esitata algse juhendina.
"""
    (output / "study/rubric.md").write_text(rubric, encoding="utf-8")


if __name__ == "__main__":
    main()
