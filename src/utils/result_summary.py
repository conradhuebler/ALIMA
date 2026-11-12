"""Readable sections of a finished result, for the chat - Claude Generated

The chat agent used to get a title and the start of the abstract; everything
else had to be fetched with tools, and the tool bridge dropped the shelf place,
the local subject-database hits, the core keywords and the DK titles. A model
asked to justify the indexing then rebuilt it from the abstract and invented
the meaning of DK numbers. Here the result is turned into plain-text sections
once, from the same state the result card is built from (so the chat and the
card cannot disagree):

* :func:`result_sections`: ``{name: text}``, empty sections left out; they go
  into ``SharedContext.extra["result_sections"]`` for the tool
  ``get_result_section``;
* :func:`result_summary_text`: the lean part that is always put into the chat context
  (keywords, chains with reasons, notations with designation, shelf place, one line on
  the local hits); the rest is fetched with the tool;
* :func:`chat_context_for_state`: title, abstract start and that summary.

Qt-free.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: Upper bound of the always-present summary (characters). The sections stay
#: complete behind the tool.
SUMMARY_MAX_CHARS = 9000

#: Sections in the order they are listed; the key is what ``get_result_section`` takes.
SECTION_ORDER = (
    "ergebnis", "ketten", "klassifikation", "dk_auswertung", "aufstellung",
    "lokal", "quellen", "winibw", "schlagworte", "suche", "bericht",
)

SECTION_HELP = {
    "ergebnis": "die Ergebniskarte: Kern, Schlagworte, Ketten, DK, Aufstellung, WinIBW, lokale Treffer",
    "ketten": "Schlagwortketten mit der Begründung je Kette",
    "klassifikation": "DK-Notationen mit Rang und Bezeichnung",
    "dk_auswertung": "Auswertung der Katalog-Notationen",
    "aufstellung": "Aufstellungsort mit Begründung",
    "lokal": "Treffer der lokalen Sacherschließung und ihre Zuordnungen",
    "quellen": "welche Quellen in diesem Lauf wie oft befragt wurden",
    "winibw": "der WinIBW-Eintrag",
    "schlagworte": "die finalen GND-Schlagworte",
    "suche": "gesuchte Begriffe und Zahl der GND-Treffer je Begriff",
    "bericht": "der Bericht des Workflows",
}


def _get(state: Any, name: str, default: Any = None) -> Any:
    value = getattr(state, name, None)
    return value if value else default


def _chains(state: Any) -> List[Any]:
    chains = list(_get(state, "keyword_chains", []) or [])
    if not chains:
        final = getattr(state, "final_llm_analysis", None)
        chains = list(getattr(final, "keyword_chains", []) or []) if final is not None else []
    return chains


def _chains_text(state: Any) -> str:
    lines = []
    for i, entry in enumerate(_chains(state), 1):
        if isinstance(entry, dict):
            terms = entry.get("chain") or entry.get("terms") or []
            reason = str(entry.get("reason") or "").strip()
        else:
            terms, reason = entry, ""
        if isinstance(terms, str):
            terms = [terms]
        line = f"{i}. " + " → ".join(str(t) for t in terms)
        if reason:
            line += f" ({reason})"
        lines.append(line)
    return "\n".join(lines)


def _classification_text(state: Any) -> str:
    """Notations with rank and, where the result has one, the designation.

    Designations come from the local subject database's hits; a notation
    without one says so, so a reader (or model) does not supply a meaning of
    its own. - Claude Generated
    """
    from src.utils.pipeline_formatters import geo_sachdb_hits

    labels = {}
    for term, text in geo_sachdb_hits(dict(_get(state, "geo_sachdb", {}) or {})):
        if term.startswith(("DK ", "RVK ", "DDC ")):
            labels.setdefault(term, re.sub(r"\s*·\s*PPN\s*\S+\s*$", "", text))
    lines = []
    for entry in _get(state, "classification_entries", []) or []:
        if not isinstance(entry, dict):
            continue
        shown = str(entry.get("display") or entry.get("code") or "")
        rank = entry.get("rank") or ""
        label = labels.get(shown) or labels.get(f"{entry.get('system') or ''} {entry.get('code') or ''}".strip())
        lines.append(shown + (f" [{rank}]" if rank else "")
                     + (f": {label}" if label else ": (keine Bezeichnung im Ergebnis)"))
    return "\n".join(lines)


def _shelf_text(state: Any) -> str:
    shelf = _get(state, "shelf", {}) or {}
    place = shelf.get("placement") if isinstance(shelf, dict) else None
    if not isinstance(place, dict) or not place:
        return ""
    head = " · ".join(str(place[k]) for k in ("notation", "label") if place.get(k))
    lines = [head or "Aufstellung"]
    for key, label in (("location", "Standort"), ("group", "Gruppe"),
                       ("reason", "Zuordnung"), ("selected_reason", "Begründung")):
        if place.get(key):
            lines.append(f"{label}: {place[key]}")
    return "\n".join(lines)


def _local_text(state: Any) -> str:
    from src.utils.pipeline_formatters import geo_sachdb_hits, geo_sachdb_proposals

    geo = dict(_get(state, "geo_sachdb", {}) or {})
    lines = [f"{term} → {text}" for term, text in geo_sachdb_hits(geo)]
    missing = [str(t) for t in (geo.get("not_found") or []) if t]
    if missing:
        lines.append("Ohne lokales Schlagwort: " + ", ".join(missing))
    proposals = geo_sachdb_proposals(geo)
    if proposals:
        lines.append(f"Weitere lokale DK-Vorschläge: {len(proposals)}")
    prior = _get(state, "geo_prior", {}) or {}
    for cand in (prior.get("candidates") or []) if isinstance(prior, dict) else []:
        if isinstance(cand, dict):
            label = cand.get("label") or ""
            via = ", ".join(cand.get("via_keywords") or [])
            lines.append(f"Vorgabe für die Klassifikation: {cand.get('notation') or cand.get('code')}"
                         + (f" {label}" if label else "") + (f" (über {via})" if via else ""))
    return "\n".join(lines)


def _sources_text(state: Any) -> str:
    rows = []
    for row in _get(state, "source_usage", []) or []:
        if isinstance(row, dict):
            rows.append(str(row.get("text") or row.get("label") or row.get("source") or ""))
    return "\n".join(r for r in rows if r)


def _keywords_text(state: Any) -> str:
    final = getattr(state, "final_llm_analysis", None)
    keywords = list(getattr(final, "extracted_gnd_keywords", []) or []) if final is not None else []
    core = [str(k) for k in (_get(state, "core_keywords", []) or [])]
    out = []
    if core:
        out.append("Kern: " + " · ".join(core))
    if keywords:
        out.append("Schlagworte: " + ", ".join(str(k) for k in keywords))
    return "\n".join(out)


def _search_text(state: Any) -> str:
    rows = []
    for sr in _get(state, "search_results", []) or []:
        term = getattr(sr, "search_term", "") or ""
        results = getattr(sr, "results", {}) or {}
        if term:
            rows.append(f"{term}: {len(results)} Treffer")
    return "\n".join(rows)


def _card_text(state: Any) -> str:
    from src.utils.pipeline_formatters import PipelineResultFormatter

    return str(PipelineResultFormatter.format_final_result_card_html(state)[1] or "").strip()


def _auswertung_text(state: Any) -> str:
    from src.utils.pipeline_formatters import PipelineResultFormatter

    return str(PipelineResultFormatter.format_dk_auswertung_card_html(state)[1] or "").strip()


def _winibw_text(state: Any) -> str:
    from src.utils.winibw_record import build_winibw_record

    return str(build_winibw_record(state).get("text") or "").strip()


_BUILDERS = {
    "ergebnis": _card_text,
    "ketten": _chains_text,
    "klassifikation": _classification_text,
    "dk_auswertung": _auswertung_text,
    "aufstellung": _shelf_text,
    "lokal": _local_text,
    "quellen": _sources_text,
    "winibw": _winibw_text,
    "schlagworte": _keywords_text,
    "suche": _search_text,
    "bericht": lambda s: str(getattr(s, "report_markdown", "") or "").strip(),
}


def result_sections(state: Any) -> Dict[str, str]:
    """``{section: text}`` of a result state; sections without content are left out.

    A section that fails to build is left out and logged: one broken section
    must not take the chat's whole result away. - Claude Generated
    """
    if state is None:
        return {}
    from src.utils.error_visibility import log_caught
    import logging

    logger = logging.getLogger(__name__)
    out: Dict[str, str] = {}
    for name in SECTION_ORDER:
        try:
            text = _BUILDERS[name](state)
        except Exception as exc:  # noqa: BLE001 - one section must not drop the others
            log_caught(logger, exc, f"Ergebnisabschnitt {name} für den Chat")
            continue
        if text and str(text).strip():
            out[name] = str(text).strip()
    return out


def result_summary_text(sections: Dict[str, str], max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """The always-present summary: keywords, chains with their reasons, notations
    with designation, shelf place with reason and a line on the local hits, plus
    the names of the sections the tool can return. The WinIBW entry and the long
    local list stay behind ``get_result_section``. Cut at ``max_chars`` with a
    note. - Claude Generated"""
    if not sections:
        return ""
    parts = []
    for key, head in (("schlagworte", "Schlagworte"), ("ketten", "Schlagwortketten mit Begründung"),
                      ("klassifikation", "Klassifikation"), ("aufstellung", "Aufstellung")):
        if sections.get(key):
            body = sections[key]
            parts.append(body if key == "schlagworte" else f"{head}:\n{body}")
    local = sections.get("lokal", "")
    if local:
        n = sum(1 for line in local.splitlines() if "→" in line and not line.startswith(("Vorgabe", "Ohne", "Weitere")))
        missing = next((l for l in local.splitlines() if l.startswith("Ohne lokales Schlagwort")), "")
        parts.append(f"Lokale Sacherschließung: {n} Treffer (Einzelheiten: Abschnitt lokal)"
                     + (f"\n{missing}" if missing else ""))
    if not parts and sections.get("ergebnis"):
        parts.append(sections["ergebnis"])
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[gekürzt, vollständig mit get_result_section]"
    names = [n for n in SECTION_ORDER if n in sections]
    text += "\n\nWeitere Abschnitte (Werkzeug get_result_section): " + ", ".join(names)
    return text


def chat_context_for_state(state: Any, sections: Optional[Dict[str, str]] = None,
                           abstract_chars: int = 500) -> str:
    """Title, abstract start and the result summary: the chat's context line.

    ``sections`` may be passed in when already built (the shared context holds
    them); otherwise they are built here. - Claude Generated
    """
    if state is None:
        return ""
    parts = []
    title = getattr(state, "working_title", "") or ""
    if title:
        parts.append(f"Titel: {title}")
    abstract = getattr(state, "original_abstract", "") or ""
    if abstract:
        cut = abstract[:abstract_chars] + ("…" if len(abstract) > abstract_chars else "")
        parts.append(f"Abstract: {cut}")
    secs = sections if sections is not None else result_sections(state)
    summary = result_summary_text(secs)
    if summary:
        parts.append(
            "Ergebnis der Erschließung dieses Werks (maßgeblich für Fragen dazu):\n" + summary
        )
    return "\n".join(parts)
