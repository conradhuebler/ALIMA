"""Publication form of the work, read off the input text - Claude Generated.

Deterministic, no model, no network, no Qt. ``detect_publication_form`` looks
for a phrase in which the input text describes ITSELF as a thesis
("Zusammenfassung der Ergebnisse der Dissertation …", "Die vorliegende
Dissertation …", the title-page formula "… zur Erlangung des akademischen
Grades Doktor … vorgelegt von"). The bare word "Dissertation" is no evidence:
a text can be about theses, count them or cite one.

Every rule lives in ``RULES``; a rule names its form, where in the text it
may match (``scope``) and its pattern. The first rule in table order that
matches wins, so the table order is the order of strength. A form only exists
here when its notation is a form notation in
``classification_systems.FORM_NOTATIONS``: that registry has no Bachelor thesis
and no Habilitation, so neither has a rule.

``apply_publication_form`` puts a detection into a result: the form keyword
into ``form_keywords`` (once), the notation to the front of the
classifications (inserted, or moved there when already assigned). Without a
detection both lists come back as they were given.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple

#: A rule with ``scope == "head"`` must start within this many characters of
#: the (whitespace-collapsed) text: the span of a title page or of the first
#: sentences of a summary.
HEAD_CHARS = 600
#: Longest evidence string; a longer match is cut and ends with "…".
EVIDENCE_MAX = 120
#: How far back the citation guard looks for "basiert auf", "based on", ….
GUARD_CHARS = 60

#: Marker on a classification entry this module inserted.
ORIGIN_PUBLICATION_FORM = "publication_form"
#: Provenance text on such an entry and in the displays.
ORIGIN_LABEL = "Form (aus dem Eingabetext)"

#: The forms a rule can name. ``notation`` must be in ``FORM_NOTATIONS``
#: (checked by a test); the form keyword is the RSWK form heading of a thesis.
FORMS: Dict[str, Dict[str, Any]] = {
    "dissertation": {
        "label": "Dissertation",
        "notation": ("DK", "378.245"),
        "form_keyword": "Hochschulschrift",
    },
    "diplom_master": {
        "label": "Diplom- oder Masterarbeit",
        "notation": ("DK", "378.244"),
        "form_keyword": "Hochschulschrift",
    },
}

# Text immediately before a match that turns it into a reference to another
# text ("Der Beitrag basiert auf der Dissertation „…“ von Frau M. Sc. …",
# "This paper is based on a thesis submitted for the degree of …"), with up to
# three words between the trigger and the match.
_CITATION_GUARD = re.compile(
    r"(?:\b(?:basier\w*|beruh\w*|stütz\w*)\s+(?:sich\s+)?auf"
    r"|\bauf\s+(?:der\s+)?Grundlage"
    r"|\bim\s+Rahmen"
    r"|\bAuszug\s+aus"
    r"|\bentstand\w*\s+(?:aus|im)"
    r"|\bbased\s+on|\bderived\s+from|\bpart\s+of|\badapted\s+from|\bdrawn\s+from)"
    r"\s+(?:\S+\s+){0,3}$",
    re.IGNORECASE,
)

_QUOTE_OPEN = "„\"»«‚‹“"
_QUOTE_CLOSE = "“”\"«»‘‹›"
_HONORIFIC = (
    r"(?:Herrn|Frau|Dipl\.|M\.\s?Sc\.|M\.\s?A\.|M\.\s?Eng\.|B\.\s?Sc\.|B\.\s?A\.|Mag\.)"
)
_GAP = r".{0,250}?"
_SUBMITTED_BY = r"\b(?:vorgelegt|eingereicht)\s+von\b"
_SUMMARY_OF = (
    r"(?:Zusammenfassung|Kurzfassung)\s+(?:der\s+(?:wesentlichen\s+)?Ergebnisse\s+)?"
    r"(?:der|meiner|dieser)\s+"
)


@dataclass(frozen=True)
class FormRule:
    """One phrase that names the form of the present work.

    ``scope``: ``start`` (the text begins with it), ``head`` (it starts within
    ``HEAD_CHARS``) or ``anywhere``. ``guarded``: the match is dropped when the
    words before it make it a reference to another text (``_CITATION_GUARD``).
    """

    name: str
    form: str
    scope: str
    pattern: Pattern[str]
    guarded: bool = False


def _rx(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


#: The rule table, strongest first: title-page formulas, a summary that names
#: itself at the start of the text, a quoted title with the candidate's
#: name, the self-reference "vorliegende …" / "this …".
RULES: Tuple[FormRule, ...] = (
    # Title page: "Von der Fakultät … genehmigte DISSERTATION zur Erlangung des …"
    FormRule("titelei_genehmigte_dissertation", "dissertation", "head",
             _rx(r"\bgenehmigte\s+(?:Inaugural[- ]?)?Dissertation\s+zur\s+Erlangung\s+de[rs]\b")),
    # Title page: "… zur Erlangung des akademischen Grades Doktor-Ingenieur … vorgelegt von"
    # ("Dr. habil." is a Habilitation, which has no form notation).
    FormRule("titelei_doktorgrad_vorgelegt", "dissertation", "head",
             _rx(r"\bzur\s+Erlangung\s+(?:des|der)\s+(?:akademischen\s+)?"
                 r"(?:Grades\s+(?:eines\s+|einer\s+)?(?:Doktor|Dr\.)(?!\s*habil)|Doktorgrades"
                 r"|Doktorwürde|Würde\s+eines\s+Doktors)" + _GAP + _SUBMITTED_BY)),
    # Title page, English: "A thesis submitted … for the degree of Doctor of Philosophy"
    FormRule("thesis_degree_doctor", "dissertation", "head",
             _rx(r"\b(?:thesis|dissertation)\s+(?:submitted\b" + _GAP + r"\s)?"
                 r"for\s+the\s+degree\s+of\s+(?:Doctor|Ph\.?\s?D)"),
             guarded=True),
    FormRule("titelei_diplom_master_vorgelegt", "diplom_master", "head",
             _rx(r"\bzur\s+Erlangung\s+des\s+(?:akademischen\s+)?Grades\s+(?:eines\s+|einer\s+)?"
                 r"(?:Diplom|Master)" + _GAP + _SUBMITTED_BY)),
    FormRule("thesis_degree_master", "diplom_master", "head",
             _rx(r"\b(?:thesis|dissertation)\s+(?:submitted\b" + _GAP + r"\s)?"
                 r"for\s+the\s+degree\s+of\s+Master\b"),
             guarded=True),
    # A summary that names itself: "Zusammenfassung der Ergebnisse der Dissertation …"
    FormRule("zusammenfassung_dissertation", "dissertation", "start",
             _rx(_SUMMARY_OF + r"(?:Dissertation(?:sschrift)?|Doktorarbeit|Promotionsschrift)\b")),
    FormRule("zusammenfassung_diplom_master", "diplom_master", "start",
             _rx(_SUMMARY_OF + r"(?:Diplomarbeit|Masterarbeit|Master-Arbeit|Masterthesis)\b")),
    # "Dissertation „Titel“ von Herrn M. Sc. …"
    FormRule("dissertation_titel_von", "dissertation", "head",
             _rx(r"\b(?:Dissertation(?:sschrift)?|Doktorarbeit)\s+[" + _QUOTE_OPEN + r"][^"
                 + _QUOTE_OPEN + _QUOTE_CLOSE + r"]{3,300}[" + _QUOTE_CLOSE + r"]\s+von\s+"
                 + _HONORIFIC),
             guarded=True),
    FormRule("diplom_master_titel_von", "diplom_master", "head",
             _rx(r"\b(?:Diplomarbeit|Masterarbeit|Master-Arbeit|Masterthesis)\s+[" + _QUOTE_OPEN
                 + r"][^" + _QUOTE_OPEN + _QUOTE_CLOSE + r"]{3,300}[" + _QUOTE_CLOSE
                 + r"]\s+von\s+" + _HONORIFIC),
             guarded=True),
    # Self-reference: "Die vorliegende Dissertation …", "In this dissertation …"
    FormRule("vorliegende_dissertation", "dissertation", "anywhere",
             _rx(r"\bvorliegenden?\s+(?:Dissertation(?:sschrift)?|Doktorarbeit|Promotionsschrift"
                 r"|Promotionsarbeit)\b")),
    FormRule("this_dissertation", "dissertation", "anywhere",
             _rx(r"\b(?:this|the\s+present)\s+(?:(?:doctoral|Ph\.?\s?D\.?)\s+(?:thesis|dissertation)"
                 r"|dissertation)\b")),
    FormRule("vorliegende_diplom_master", "diplom_master", "anywhere",
             _rx(r"\bvorliegenden?\s+(?:Diplomarbeit|Masterarbeit|Master-Arbeit|Masterthesis)\b")),
    FormRule("this_master_thesis", "diplom_master", "anywhere",
             _rx(r"\b(?:this|the\s+present)\s+(?:master'?s?|master’s)\s+thesis\b")),
)


def _normalize(text: Any) -> str:
    """NFC, every run of whitespace (line breaks of a title page) one space."""
    return " ".join(unicodedata.normalize("NFC", str(text or "")).split())


def _lead(text: str) -> int:
    """Index of the first letter or digit: a ``start`` rule may follow quotes,
    dashes or a bullet, nothing else."""
    for index, char in enumerate(text):
        if char.isalnum():
            return index
    return len(text)


def _evidence(text: str) -> str:
    return text if len(text) <= EVIDENCE_MAX else text[: EVIDENCE_MAX - 1].rstrip() + "…"


def _match(rule: FormRule, text: str) -> Optional[re.Match]:
    if rule.scope == "start":
        return rule.pattern.match(text, _lead(text))
    if rule.scope == "head":
        found = rule.pattern.search(text, 0, HEAD_CHARS + 400)
        return found if found and found.start() < HEAD_CHARS else None
    return rule.pattern.search(text)


def _cited(text: str, start: int) -> bool:
    return bool(_CITATION_GUARD.search(text[max(0, start - GUARD_CHARS):start]))


def detect_publication_form(text: Any) -> Optional[Dict[str, Any]]:
    """The form the input text gives itself, or ``None``.

    Returns ``{form, label, notation: (system, code), form_keyword, evidence,
    rule}``; ``evidence`` is the matched passage (whitespace collapsed, at most
    ``EVIDENCE_MAX`` characters), ``rule`` the name of the rule in ``RULES``.
    The first matching rule in table order wins; there is one result at most.
    - Claude Generated
    """
    norm = _normalize(text)
    if not norm:
        return None
    for rule in RULES:
        found = _match(rule, norm)
        if found is None:
            continue
        if rule.guarded and _cited(norm, found.start()):
            continue
        form = FORMS[rule.form]
        return {
            "form": rule.form,
            "label": form["label"],
            "notation": tuple(form["notation"]),
            "form_keyword": form["form_keyword"],
            "evidence": _evidence(found.group(0)),
            "rule": rule.name,
        }
    return None


# ── into the result ───────────────────────────────────────────────────────


def origin_text(label: str) -> str:
    """``Form: Dissertation (aus dem Eingabetext)``; without a label the bare
    ``ORIGIN_LABEL``. The provenance the displays show. - Claude Generated"""
    label = str(label or "").strip()
    return f"Form: {label} (aus dem Eingabetext)" if label else ORIGIN_LABEL


def notation_display(detection: Dict[str, Any]) -> str:
    """``DK 378.245`` of a detection (``notation`` as tuple or list)."""
    system, code = (list(detection.get("notation") or []) + ["", ""])[:2]
    return f"{system} {code}".strip()


def _entry_key(entry: Any) -> Tuple[str, str]:
    """``(system, code)`` of a classification entry (dict or string), the
    system from the prefix, else ``system``/``type``, else DK."""
    from .classification_systems import split_classification_code

    if isinstance(entry, dict):
        raw = str(entry.get("code") or entry.get("notation") or "").strip()
        fallback = str(entry.get("system") or entry.get("type") or "").strip().upper()
    else:
        raw, fallback = str(entry or "").strip(), ""
    system, code = split_classification_code(raw)
    return (system or fallback or "DK"), re.sub(r"\s+", "", code).upper()


def _keyword_key(item: Any) -> str:
    """Comparison form of a keyword: without a GND-ID tail, NFC, case-folded."""
    if isinstance(item, dict):
        item = item.get("keyword") or item.get("title") or ""
    text = str(item or "")
    if "(" in text and ")" in text:
        text = text.split("(")[0]
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def apply_publication_form(
    detection: Optional[Dict[str, Any]],
    form_keywords: Optional[List[Any]],
    classifications: Optional[List[Any]],
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """``(form_keywords, classifications)`` with the detection applied.

    No detection: both inputs are returned as they are (same objects). With
    one: the form keyword is appended as ``{"keyword": …}`` unless a form
    keyword of the same spelling is there; the notation goes first, either the
    entry already assigned (moved, otherwise unchanged) or a new entry
    ``{code, type, rank: additional, origin, reason}`` with ``origin`` =
    ``ORIGIN_PUBLICATION_FORM``. New lists; the inputs are not mutated.
    - Claude Generated
    """
    if not detection:
        return form_keywords, classifications

    keywords = list(form_keywords or [])
    term = str(detection.get("form_keyword") or "").strip()
    if term and _keyword_key(term) not in {_keyword_key(item) for item in keywords}:
        keywords.append({"keyword": term})

    entries = list(classifications or [])
    system, code = (list(detection.get("notation") or []) + ["", ""])[:2]
    if not code:
        return keywords, entries
    wanted = (str(system).upper(), re.sub(r"\s+", "", str(code)).upper())
    index = next((i for i, entry in enumerate(entries) if _entry_key(entry) == wanted), None)
    if index is not None:
        entry = entries.pop(index)
    else:
        entry = {
            "code": f"{system} {code}",
            "type": system,
            "rank": "additional",
            "origin": ORIGIN_PUBLICATION_FORM,
            "reason": f"{ORIGIN_LABEL}: „{detection.get('evidence') or ''}“",
        }
    return keywords, [entry] + entries
