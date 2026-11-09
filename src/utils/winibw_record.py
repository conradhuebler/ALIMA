"""The WinIBW entry of a result, built in one place - Claude Generated.

Every output that writes catalogue lines (result card in GUI and webapp,
CLI ``show-protocol``, ``show-protocol --format k10plus``, the review tab,
the ``marc`` export and ``results.winibw`` in the JSON export) calls
``build_winibw_record``. It reads a state object (``KeywordAnalysisState``) or
an export payload dict (``{"results": {...}}`` or the results dict itself) and
is deterministic: no LLM, no network, no Qt.

Layout of the entry::

    5550 <term 1 of chain 1>
    5550 <term 2 of chain 1>
    5550 $A<ISIL>

    5551 <term 1 of chain 2>
    5551 $A<ISIL>

    5552 <single keyword>
    5553 <single keyword>

    1131 <form keyword>

    6700 DK <notation>
    6700 <RVK notation>

* Chains start at 5550, one field number per chain for all its terms, closed by
  ``<nr> $A<ISIL>`` (``institution.chain_end_marker``; ``$ADE-105`` for the UB
  Freiberg); one blank line between chains and between the sections. Without
  an ISIL the closing line is left out and ``notes`` says so.
* Single keywords (final keywords in no chain, not a form keyword, each once)
  get the next field numbers, one each, without the closing line.
* Form keywords (``form_keywords``) go to 1131, never to 555x; a form keyword
  inside a chain is taken out of the chain.
* DK as ``6700 DK <notation>``, RVK as ``6700 <notation>`` without prefix. Other
  systems get no line and are listed in ``skipped``.
* With ``geo_sachdb`` in the state (fields ``terms``/``notations``): a term with
  exactly one local heading (or exactly one of kind Sachschlagwort) that has a
  PPN is written ``<nr> !PPN!``; an assigned notation whose local entries carry
  exactly one PPN is written ``6700 !PPN!``. Everything else stays text.

NOT VERIFIED IN WINIBW: the field 1131 for form keywords, the last subject field
number (``WINIBW_SUBJECT_LAST_TAG``), "one chain takes one field number", that
WinIBW accepts a term as text instead of a PPN and resolves it, and that blank
lines between the fields are harmless.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: First field number of the subject headings.
WINIBW_SUBJECT_FIRST_TAG = 5550
#: Last field number a subject heading may get. 5559 is the bound the K10+
#: export used before (``generate_k10plus_lines``); whether WinIBW allows more
#: (5599) is NOT VERIFIED. A chain or single keyword beyond it gets no line and
#: is listed in ``overflow``. Read at call time, so this name is the one place
#: to change it.
WINIBW_SUBJECT_LAST_TAG = 5559
#: The closing line of a keyword chain, ``<nr> $A<ISIL>``, comes from the
#: institution setting (``institution.chain_end_marker``). - Claude Generated
CHAIN_END_NOT_SET_NOTE = (
    "Ketten ohne Abschlusszeile ($A<ISIL>): Einrichtung nicht gesetzt "
    "(ISIL unter Einstellungen ▸ Datenquellen ▸ Einrichtung)"
)
#: Field of the form keywords (Lehrbuch, …). NOT VERIFIED IN WINIBW.
WINIBW_FORM_TAG = "1131"
#: Field of the notations (DK, RVK).
WINIBW_CLASSIFICATION_TAG = "6700"

ORIGIN_CHAIN = "chain"
ORIGIN_CHAIN_END = "chain_end"
ORIGIN_SINGLE = "single"
ORIGIN_FORM = "form"

#: Systems the entry writes; others go to ``skipped``.
_WRITTEN_SYSTEMS = ("DK", "RVK")
_RVK_SHAPE = re.compile(r"[A-Z]{2} ?[0-9]")
_UNKNOWN_SYSTEMS = ("", "UNKNOWN")


# ── input normalisation ───────────────────────────────────────────────────


def strip_gnd_id(keyword: Any) -> str:
    """``"Wort (123456-7)"`` / ``"Wort (GND-ID: 123)"`` → ``"Wort"``. - Claude Generated

    Cuts at the first ``(`` when the term carries a closing ``)`` too, as the
    K10+ export always did. A heading whose own name contains parentheses
    loses that part as well.
    """
    text = str(keyword or "")
    if "(" in text and ")" in text:
        return text.split("(")[0].strip()
    return text.strip()


def _term_key(text: Any) -> str:
    """Comparison form of a term: NFC, case-folded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", str(text or "")).casefold().split())


def _compact(notation: Any) -> str:
    return re.sub(r"\s+", "", str(notation or "")).upper()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """``obj[name]`` for a dict, ``obj.name`` otherwise; falsy values → ``default``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name)
    else:
        value = getattr(obj, name, None)
    return value if value else default


def _chains(raw_chains: Iterable[Any]) -> List[List[str]]:
    """Chains as lists of terms (``{"chain": [...]}`` or plain lists; scalars dropped)."""
    chains: List[List[str]] = []
    for entry in list(raw_chains or []):
        if isinstance(entry, dict):
            terms = entry.get("chain") or []
        elif isinstance(entry, (list, tuple)):
            terms = list(entry)
        else:
            continue
        cleaned = [strip_gnd_id(term) for term in terms if str(term).strip()]
        cleaned = [term for term in cleaned if term]
        if cleaned:
            chains.append(cleaned)
    return chains


def _results(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("results"), dict):
        return payload["results"]
    return payload


def chains_from_payload(payload: Dict[str, Any]) -> List[List[str]]:
    """Keyword chains of an export payload: ``results.keyword_chains``, else
    ``results.final_llm_call_details.keyword_chains``. - Claude Generated"""
    results = _results(payload)
    raw = results.get("keyword_chains") or []
    if not raw:
        details = results.get("final_llm_call_details") or {}
        if isinstance(details, dict):
            raw = details.get("keyword_chains") or []
    return _chains(raw)


def _keyword_texts(items: Iterable[Any]) -> List[str]:
    out = []
    for item in list(items or []):
        if isinstance(item, dict):
            item = item.get("keyword") or item.get("title") or ""
        term = strip_gnd_id(item)
        if term:
            out.append(term)
    return out


def record_inputs(source: Any) -> Dict[str, Any]:
    """What the entry is built from, for a state object or an export payload.

    Returns ``{chains, keywords, form_keywords, classifications, geo}``. The
    chain source order matches ``extract_results_from_analysis_state``
    (``final_llm_analysis.keyword_chains`` first, then ``keyword_chains``), so a
    state and its export payload give the same entry. - Claude Generated
    """
    if isinstance(source, dict):
        results = _results(source)
        details = results.get("final_llm_call_details") or {}
        keywords = results.get("final_keywords") or []
        if not keywords and isinstance(details, dict):
            keywords = details.get("extracted_gnd_keywords") or []
        return {
            "chains": chains_from_payload(source),
            "keywords": _keyword_texts(keywords),
            "form_keywords": _keyword_texts(results.get("form_keywords") or []),
            "classifications": list(results.get("classifications")
                                    or results.get("dk_classifications") or []),
            "geo": results.get("geo_sachdb") if isinstance(results.get("geo_sachdb"), dict) else {},
        }
    final_llm = _field(source, "final_llm_analysis")
    raw_chains = _field(final_llm, "keyword_chains", []) or _field(source, "keyword_chains", [])
    geo = _field(source, "geo_sachdb", {})
    return {
        "chains": _chains(raw_chains),
        "keywords": _keyword_texts(_field(final_llm, "extracted_gnd_keywords", [])),
        "form_keywords": _keyword_texts(_field(source, "form_keywords", [])),
        "classifications": list(_field(source, "classification_entries", [])
                                or _field(source, "dk_classifications", [])),
        "geo": geo if isinstance(geo, dict) else {},
    }


def split_notation(item: Any) -> Tuple[str, str]:
    """``(system, code)`` of a classification string or dict. - Claude Generated

    An entry without a known system is RVK when it has the RVK shape (two
    capitals, digits: ``AR 18100``), otherwise DK, the default the K10+ export
    always used. A ``" - label"`` tail of a legacy string is dropped.
    """
    from .classification_systems import split_classification_code

    if isinstance(item, dict):
        system = str(item.get("system") or "").strip().upper()
        code = str(item.get("code") or item.get("notation") or "").strip()
        if not code:
            system, code = split_classification_code(str(item.get("display") or ""))
    else:
        text = str(item or "").strip()
        if " - " in text:
            text = text.split(" - ")[0].strip()
        system, code = split_classification_code(text)
    if system in _UNKNOWN_SYSTEMS:
        prefixed, rest = split_classification_code(code)
        if prefixed:
            system, code = prefixed, rest
    elif code.upper().startswith(system + " "):
        code = code[len(system) + 1:].strip()
    if system in _UNKNOWN_SYSTEMS:
        system = "RVK" if _RVK_SHAPE.match(code) else "DK"
    return system, code


def classification_line(system: str, code: str, ppn: str = "") -> Optional[str]:
    """The 6700 line of a notation: ``6700 !PPN!`` with a local PPN, otherwise
    ``6700 DK <code>`` for DK and ``6700 <code>`` for RVK; ``None`` for other
    systems. - Claude Generated"""
    if ppn:
        return f"{WINIBW_CLASSIFICATION_TAG} !{ppn}!"
    system = str(system or "").upper()
    if system == "DK":
        return f"{WINIBW_CLASSIFICATION_TAG} DK {code}"
    if system == "RVK":
        return f"{WINIBW_CLASSIFICATION_TAG} {code}"
    return None


# ── local subject database (geo_sachdb) ───────────────────────────────────


def _geo_maps(geo: Dict[str, Any]) -> Tuple[Dict[str, Dict], Dict[Tuple[str, str], Dict]]:
    """Per-term and per-notation local results of ``geo_sachdb``; empty for
    results saved before ``terms``/``notations`` existed."""
    terms = {}
    for item in geo.get("terms") or []:
        if isinstance(item, dict) and item.get("keyword"):
            terms.setdefault(_term_key(item["keyword"]), item)
    notations = {}
    for item in geo.get("notations") or []:
        if isinstance(item, dict) and item.get("notation"):
            key = (str(item.get("system") or "").upper(), _compact(item["notation"]))
            notations.setdefault(key, item)
    return terms, notations


def _local_text(record: Dict[str, Any]) -> str:
    parts = [str(record.get("label") or "").strip(), str(record.get("kind") or "").strip()]
    return " · ".join(part for part in parts if part)


def _geo_term(term: str, terms: Dict[str, Dict]) -> Tuple[str, Optional[Dict], str]:
    """``(ppn, local record, note)`` for a 555x term."""
    item = terms.get(_term_key(term))
    if not item:
        return "", None, ""
    status = item.get("status")
    record = item.get("record") if isinstance(item.get("record"), dict) else None
    if status == "unique" and item.get("ppn"):
        return str(item["ppn"]), record, ""
    if status == "ambiguous":
        count = len(item.get("records") or [])
        return "", None, f"lokal mehrdeutig ({count} Sätze), als Text"
    if status == "no_ppn" and record:
        return "", record, "lokaler Satz ohne PPN, als Text"
    return "", None, ""


def _geo_notation(system: str, code: str, notations: Dict) -> Tuple[str, Optional[Dict], str]:
    """``(ppn, local entry, note)`` for an assigned notation."""
    item = notations.get((system, _compact(code)))
    if not item:
        return "", None, ""
    local = [n for n in (item.get("local") or []) if isinstance(n, dict)]
    status = item.get("status")
    if status == "unique" and item.get("ppn"):
        entry = next((n for n in local if str(n.get("ppn") or "") == str(item["ppn"])), None)
        return str(item["ppn"]), entry, ""
    if status == "ambiguous":
        return "", None, f"lokal mehrdeutig ({len(local)} Einträge mit verschiedener PPN), als Text"
    if status == "no_ppn":
        return "", local[0] if local else None, "lokaler Eintrag ohne PPN"
    return "", None, ""


# ── the entry ─────────────────────────────────────────────────────────────


def build_winibw_record(source: Any) -> Dict[str, Any]:
    """The WinIBW entry of a result (state object or export payload).

    Returns ``{lines, text, entries, overflow, skipped, last_tag}``:

    * ``lines``: the entry line by line, blank lines (``""``) included;
      ``text`` the same joined with ``\\n``, the pure copy block.
    * ``entries``: one dict per non-blank line, in order: ``line``, ``tag``,
      ``origin`` (``chain``/``chain_end``/``single``/``form``/``DK``/``RVK``),
      ``term`` (the result's term or notation), ``chain`` (1-based, chains
      only), ``local`` (the local heading or notation entry whose PPN the line
      uses, or that was found without PPN), ``note``, ``also`` (further terms
      that gave the same line).
    * ``overflow``: chains and single keywords left without a field number
      because the numbers passed ``WINIBW_SUBJECT_LAST_TAG``; ``{origin, term,
      chain?}``. Nothing is put on the last number instead.
    * ``skipped``: notations of a system without a WinIBW field rule.
    * ``notes`` (only when there is one): what the entry lacks for a reason
      outside the result, i.e. the chain closing line without an ISIL.

    - Claude Generated
    """
    from .institution import chain_end_marker

    data = record_inputs(source)
    last_tag = int(WINIBW_SUBJECT_LAST_TAG)
    chain_end = chain_end_marker()
    geo_terms, geo_notations = _geo_maps(data["geo"] or {})

    form_keys: Dict[str, str] = {}
    for term in data["form_keywords"]:
        form_keys.setdefault(_term_key(term), term)
    form_from_chain: Dict[str, List[int]] = {}

    sections: List[List[List[Dict[str, Any]]]] = []  # section → groups → entries
    overflow: List[Dict[str, Any]] = []
    by_value: Dict[str, Dict[str, Any]] = {}
    tag = WINIBW_SUBJECT_FIRST_TAG

    def subject_entry(term: str, origin: str, number: int, chain: Optional[int]) -> Dict[str, Any]:
        ppn, local, note = _geo_term(term, geo_terms)
        value = f"!{ppn}!" if ppn else term
        entry: Dict[str, Any] = {"line": f"{number} {value}", "tag": str(number), "origin": origin,
                                 "term": term, "local": dict(local) if local else None, "note": note}
        if chain is not None:
            entry["chain"] = chain
        return entry

    # Chains: one field number each, closed by $A<ISIL> when an ISIL is set.
    chain_groups: List[List[Dict[str, Any]]] = []
    chain_terms: set = set()
    chain_no = 0
    for raw_chain in data["chains"]:
        terms = []
        for term in raw_chain:
            key = _term_key(term)
            if key in form_keys:
                form_from_chain.setdefault(key, []).append(chain_no + 1)
                continue
            terms.append(term)
        if not terms:
            continue
        chain_no += 1
        chain_terms.update(_term_key(term) for term in terms)
        if tag > last_tag:
            overflow.append({"origin": ORIGIN_CHAIN, "term": " → ".join(terms), "chain": chain_no})
            continue
        group = [subject_entry(term, ORIGIN_CHAIN, tag, chain_no) for term in terms]
        for entry in group:
            by_value.setdefault(entry["line"].split(" ", 1)[1], entry)
        if chain_end:
            group.append({"line": f"{tag} {chain_end}", "tag": str(tag), "origin": ORIGIN_CHAIN_END,
                          "term": "", "chain": chain_no, "local": None, "note": ""})
        chain_groups.append(group)
        tag += 1
    if chain_groups:
        sections.append(chain_groups)

    # Single keywords: in no chain, not a form keyword, each once.
    singles: List[Dict[str, Any]] = []
    seen = set(chain_terms) | set(form_keys)
    for term in data["keywords"]:
        key = _term_key(term)
        if key in seen:
            continue
        seen.add(key)
        if tag > last_tag:
            overflow.append({"origin": ORIGIN_SINGLE, "term": term})
            continue
        entry = subject_entry(term, ORIGIN_SINGLE, tag, None)
        value = entry["line"].split(" ", 1)[1]
        if value in by_value:
            # Another term already gave this line (same local PPN).
            by_value[value].setdefault("also", []).append(term)
            continue
        by_value[value] = entry
        singles.append(entry)
        tag += 1
    if singles:
        sections.append([singles])

    # Form keywords: 1131, as text.
    forms: List[Dict[str, Any]] = []
    for key, term in form_keys.items():
        chains_of = form_from_chain.get(key)
        note = ""
        if chains_of:
            note = "aus Kette " + ", ".join(str(n) for n in chains_of) + " genommen"
        forms.append({"line": f"{WINIBW_FORM_TAG} {term}", "tag": WINIBW_FORM_TAG, "origin": ORIGIN_FORM,
                      "term": term, "local": None, "note": note})
    if forms:
        sections.append([forms])

    # Notations: 6700, DK with prefix, RVK without.
    notations: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    seen_codes = set()
    for item in data["classifications"]:
        system, code = split_notation(item)
        if not code or (system, _compact(code)) in seen_codes:
            continue
        seen_codes.add((system, _compact(code)))
        if system not in _WRITTEN_SYSTEMS:
            skipped.append({"system": system, "code": code, "reason": "kein WinIBW-Feld festgelegt"})
            continue
        ppn, local, note = _geo_notation(system, code, geo_notations)
        notations.append({"line": classification_line(system, code, ppn), "tag": WINIBW_CLASSIFICATION_TAG,
                          "origin": system, "term": f"{system} {code}", "local": dict(local) if local else None,
                          "note": note})
    if notations:
        sections.append([notations])

    lines: List[str] = []
    entries: List[Dict[str, Any]] = []
    for section in sections:
        for group in section:
            if lines:
                lines.append("")
            for entry in group:
                lines.append(entry["line"])
                entries.append(entry)
    record = {
        "lines": lines,
        "text": "\n".join(lines),
        "entries": entries,
        "overflow": overflow,
        "skipped": skipped,
        "last_tag": last_tag,
    }
    if chain_groups and not chain_end:
        record["notes"] = [CHAIN_END_NOT_SET_NOTE]
    return record


# ── description for the displays ──────────────────────────────────────────


def winibw_entry_parts(entry: Dict[str, Any]) -> Tuple[str, str, str]:
    """``(line, description, note)`` of one entry. - Claude Generated

    The result card and the CLI both describe a line through this, so they
    say the same: description = the result's term or notation when the line
    does not already show it (a PPN line, an RVK line without prefix), note =
    where the line comes from (chain, single, form), the local heading or
    entry whose PPN it uses, and the remark on ambiguous or PPN-less local hits.
    """
    if not isinstance(entry, dict):
        return str(entry or ""), "", ""
    line = str(entry.get("line") or "")
    origin = entry.get("origin")
    chain = entry.get("chain")
    parts: List[str] = []
    if origin == ORIGIN_CHAIN:
        parts.append(f"Kette {chain}")
    elif origin == ORIGIN_CHAIN_END:
        parts.append(f"Abschluss Kette {chain}")
    elif origin == ORIGIN_SINGLE:
        parts.append("Einzelschlagwort")
    elif origin == ORIGIN_FORM:
        parts.append("Formschlagwort")
    local = entry.get("local")
    if isinstance(local, dict):
        text = _local_text(local)
        if text:
            parts.append(f"lokal: {text}")
    if entry.get("note"):
        parts.append(str(entry["note"]))
    if entry.get("also"):
        parts.append("auch für " + ", ".join(str(t) for t in entry["also"]))
    term = str(entry.get("term") or "")
    # A text line already names its term; only a PPN (or the RVK line without
    # prefix) needs it spelt out.
    if term and line.split(" ", 1)[-1] == term:
        term = ""
    return line, term, " · ".join(parts)


def winibw_entry_text(entry: Dict[str, Any]) -> str:
    """One entry as plain text: ``<line> = <description> · <note>``. - Claude Generated"""
    line, description, note = winibw_entry_parts(entry)
    text = f"{line} = {description}" if description else line
    return f"{text} · {note}" if note else text


def winibw_notes(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(label, text)`` rows for what the entry left out: subjects past the
    last field number, notations without a field rule and the ``notes`` of the
    record (chain closing line without ISIL). - Claude Generated"""
    rows: List[Tuple[str, str]] = []
    for note in record.get("notes") or []:
        rows.append(("Hinweis", str(note)))
    overflow = [item for item in (record.get("overflow") or []) if isinstance(item, dict)]
    if overflow:
        def _item(item: Dict[str, Any]) -> str:
            if item.get("origin") == ORIGIN_CHAIN:
                return f"Kette {item.get('chain')}: {item.get('term')}"
            return str(item.get("term") or "")

        rows.append((f"Ohne Feldnummer ({record.get('last_tag')} erreicht)",
                     "; ".join(_item(item) for item in overflow)))
    skipped = [item for item in (record.get("skipped") or []) if isinstance(item, dict)]
    if skipped:
        rows.append(("Nicht übernommen",
                     "; ".join(f"{item.get('system')} {item.get('code')} ({item.get('reason')})"
                               for item in skipped)))
    return rows
