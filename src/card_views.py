"""Sama juhtumi selgitus prognoosi kasutajale ja mudeli arendajale."""

from datetime import datetime
import hashlib
import json

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .common import ROOT, write_json
from .labels import EXPERIMENT_NAMES, feature_label

ROLE_NAMES = {"user": "Prognoosi kasutaja", "developer": "Mudeli arendaja"}
TEMPLATE = Environment(
    loader=FileSystemLoader(ROOT / "templates"), undefined=StrictUndefined, autoescape=False
).get_template("role_card.md.j2")


def number(value, decimals=3):
    return f"{value:.{decimals}f}".replace(".", ",")


def display_time(value):
    return datetime.fromisoformat(str(value)).strftime("%d.%m.%Y %H:%M")


def evidence_id(card):
    encoded = json.dumps(card, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_card_view(card, role):
    diagnosis = card["diagnosis"]
    experiment = diagnosis["experiment"]
    has_hypothesis = diagnosis["hypothesis"] is not None
    common = {
        key: card[key]
        for key in ("origin_time", "target_time", "prediction", "actual", "error", "absolute_error")
    }
    common.update(hypothesis_available=has_hypothesis, experiment=experiment)
    sections = []

    def section(title, facts):
        sections.append(
            {
                "title": title,
                "facts": [{"label": label, "text": text} for label, text in facts],
            }
        )

    direction = "suurem" if card["error"] > 0 else "väiksem" if card["error"] < 0 else "võrdne"
    facts = [
        (
            "Aeg",
            f"Prognoos tehti {display_time(card['origin_time'])}. Prognoositav tund: {display_time(card['target_time'])}. Panama kohalik aeg.",
        ),
        (
            "Prognoos ja tegelik koormus",
            f"Prognoos oli {number(card['prediction'])}, tegelik koormus {number(card['actual'])}.",
        ),
        (
            "Viga",
            (
                f"Prognoosi ja tegeliku koormuse vahe: {number(card['error'])}. Vea suurus: {number(card['absolute_error'])}. Prognoos oli tegelikust {direction}."
                if card["error"]
                else "Prognoos ja tegelik koormus olid võrdsed. Viga oli 0."
            ),
        ),
        (
            "Olukord",
            "; ".join(card["scenarios"]).replace(" (tagasivaateline)", "") + ".",
        ),
    ]
    if card.get("selection_reason"):
        facts.append(("Miks see näide valiti", card["selection_reason"]))
    section("Mis juhtus?", facts)

    patterns = card["observed_patterns"]
    section(
        "Mida teati enne ja mis selgus hiljem?",
        [
            (
                "Enne prognoosi",
                f"Viimase teadaoleva tunni koormusmuutus oli {number(patterns['known_ramp'])}. Varem mõõdetud temperatuur oli {number(patterns['known_temperature'], 2)} °C; ilmaandmed on vähemalt 24 tunni vanused.",
            ),
            (
                "Pärast prognoosi",
                f"Järgmise tunni tegelik koormusmuutus oli {number(patterns['realized_ramp'])}. Seda mudel prognoosi tehes ei teadnud.",
            ),
            (
                "Analüüsi aeg",
                "Olukorra viga ja võimalikku muudatust hinnati kogu 2018. aasta andmetel. See info polnud prognoosi tegemisel veel täielikult teada.",
            ),
        ],
    )

    contributions = card["shap_contributions"]
    shown = contributions[:3] if role == "user" else contributions
    facts = [
        (
            "Kuidas panuseid lugeda",
            "Prognoos saadakse lähteväärtuse ja SHAP-panuste liitmisel. Positiivne panus tõstab ja negatiivne langetab prognoosi lähteväärtuse suhtes. See ei tõenda vea põhjust. Arvud on ümardatud.",
        ),
        ("Lähteväärtus", number(card["base_value"]) + "."),
    ]
    for item in shown:
        feature = item["feature"]
        detail = f"Panus {number(item['shap'])}."
        if role == "developer":
            detail += f" Tunnus `{feature}`, väärtus {number(item['value'])}."
        facts.append((feature_label(feature, simple=role == "user"), detail))
    if role == "user":
        facts.append(
            (
                "Ülejäänud tunnused kokku",
                number(sum(item["shap"] for item in contributions[3:])) + ".",
            )
        )
    section("Kuidas mudel prognoosini jõudis?", facts)

    hypothesis = "Selle info põhjal mudelimuudatust ei soovitata. Vea põhjus jääb selgitamata."
    action = "Kontrolli lähteandmeid ja seda, kas sarnased vead korduvad."
    if has_hypothesis:
        hypothesis = (
            f"Võimalik lisainfo mudelile: {EXPERIMENT_NAMES[experiment]}. Kasu tuleb katsega kontrollida."
            if role == "user"
            else diagnosis["hypothesis"]
        )
        action = (
            "Anna juhtum koos kaardiga arendajale uurimiseks. Kaart üksi ei anna alust prognoosi käsitsi muuta."
            if role == "user"
            else f"Kontrollkatse: lisa samale GBR-mudelile ainult tunnuserühm `{experiment}`. Võrdle samadel tundidel üldist ja selle olukorra viga. Vali muudatus 2019. aasta andmetel; 2020. aasta andmed jäta lõpphindamiseks."
        )
    facts = [("Katsetatav muudatus" if has_hypothesis else "Soovitus", hypothesis)]
    if has_hypothesis:
        facts.append(
            (
                "Sarnaste olukordade vead",
                f"2018. aastal oli selles olukorras {diagnosis['cohort_n']} tundi. Keskmine absoluutviga (MAE) oli tavapäraste tundide omast {number(diagnosis['MAE_minus_ordinary'])} võrra suurem.",
            )
        )
    facts.append(("Järgmine samm", action))
    section("Mida edasi teha?", facts)

    history = card["historical_cases"]
    facts = [
        (
            "Leitud juhtumid",
            (
                f"Leiti {len(history)} varasemat sarnast juhtumit, mille tulemused olid prognoosi tegemisel teada. Sarnasus ei tõenda sama vea põhjust."
                if history
                else "Varasemaid sarnaseid juhtumeid ei leitud."
            ),
        )
    ]
    if role == "developer" and history:
        facts.append(
            (
                "Kauguse tähendus",
                "Kaugus on arvutatud tunnuste põhjal. Väiksem väärtus tähendab suuremat sarnasust.",
            )
        )
        for previous in history:
            facts.append(
                (
                    display_time(previous["target_time"]),
                    f"Prognoos {number(previous['prediction'])}, tegelik koormus {number(previous['target_next_hour'])}, viga {number(previous['error'])}. Kaugus: {number(previous['distance'])}.",
                )
            )
    section("Varasemad sarnased juhtumid", facts)
    section(
        "Mida arvestada?",
        [
            (
                "Ühikud",
                "Koormus, vead ja SHAP-panused on algandmete ühikus. Ühiku tähendus pole kinnitatud.",
            ),
            (
                "Selgituse piirid",
                "Kaart ei näita prognoosi usaldusvahemikku ega muudatuse edu tõenäosust. Seotud tunnused võivad mõjutada SHAP-panuste jaotust.",
            ),
            (
                "Andmete kättesaadavus",
                "Eeldame koormuse kohest saadavust. Ilmaandmed on mõõtmiste põhjal hiljem arvutatud, mitte prognoositud. Nende avaldamisaeg pole teada.",
            ),
        ],
    )
    return {
        "role": role,
        "role_name": ROLE_NAMES[role],
        "evidence_id": evidence_id(card),
        "common": common,
        "sections": sections,
    }


def render_card_view(view, layout="structured", evidence_link="evidence.json"):
    if layout not in ("structured", "plain"):
        raise ValueError("Vali kaardi esitus: structured või plain.")
    return TEMPLATE.render(view=view, layout=layout, evidence_link=evidence_link)


def save_role_cards(card, folder):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "evidence.json", card)
    for role in ROLE_NAMES:
        (folder / f"{role}.md").write_text(
            render_card_view(build_card_view(card, role)), encoding="utf-8"
        )


def render_repair_audit(audit):
    """Render the paired correction/gain ledger inside the main thesis report."""
    from .labels import REPAIR_GROUP_NAMES

    def n(value):
        return f"{value:.3f}".replace(".", ",")

    lines = [
        f"**Juhtum {audit['target_time']}.** Tegelik koormus {n(audit['actual'])}; "
        f"algprognoos {n(audit['baseline'])}; parandatud prognoos {n(audit['prediction'])}.",
        f"Absoluutviga {n(audit['absolute_error_before'])} → {n(audit['absolute_error_after'])}; "
        f"vea vähenemine {n(audit['gain'])}. Kõik arvud on vastava andmestiku koormusühikus.",
        "",
        "| Paranduse osa | Panus prognoosi muutusse | Vea vähenemise Shapley-jaotus |",
        "|---|---:|---:|",
    ]
    for item in sorted(audit["items"], key=lambda x: abs(x["gain"]), reverse=True):
        lines.append(
            f"| {REPAIR_GROUP_NAMES.get(item['group'], item['group'])} | "
            f"{n(item['correction'])} | {n(item['gain'])} |"
        )
    lines.extend(
        [
            "",
            "Esimese arvveeru positiivne väärtus suurendab prognoosi. "
            "Teise positiivne väärtus tähendab sellele osale jaotatud absoluutvea vähenemist; "
            "negatiivne väärtus tähendab jaotatud kahju. Üksiku osa eemaldamise mõju võib erineda.",
            audit["limitations"],
        ]
    )
    return "\n\n".join(lines[:2]) + "\n\n" + "\n".join(lines[2:])
