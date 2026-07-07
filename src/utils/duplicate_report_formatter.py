"""duplicate_report_formatter.py - Claude Generated.

Markdown table renderer for the ``title_list_search`` workflow's duplicate
analysis. Turns ``extra.duplicate_analysis`` (parsed LLM JSON, single-shot or
chunk-merged) into a human-readable GFM table — the raw JSON blob dumped
into the log/output was the explicit complaint this replaces.

Pure functions, no LLM/registry/Qt dependency — kept separate from
``deterministic_functions.py`` so it stays independently unit-testable.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

# Order matters: this is also the order status counts are listed in the
# summary line. - Claude Generated
_STATUS_LABELS = {
    "duplicate": "Duplikat",
    "likely_duplicate": "wahrscheinlich Duplikat",
    "different_edition": "neue Auflage",
    "related_work": "Sekundärliteratur zum Werk",
    "new": "neu",
    "no_match": "nicht im Bestand",
}

# Only these statuses mean `matches[]` actually describes the wishlist
# title itself (or, for related_work, genuinely relevant literature about
# it) — worth showing edition/link info for. "new" means the LLM decided
# the hit(s) are a different, unrelated work entirely, and "no_match" has
# no hits — neither should surface year/publisher/links as if they
# belonged to the wishlist item. - Claude Generated
_STATUSES_WITH_RELEVANT_MATCHES = {
    "duplicate", "likely_duplicate", "different_edition", "related_work",
}

# Statuses meaning "the exact work is already sufficiently held" — no
# acquisition needed. Everything else (different_edition, related_work, new,
# no_match) means the wishlist title itself isn't fully covered yet, even
# though the reasons differ (older edition held / only secondary literature
# held / a same-named-but-different work was found / nothing at all).
# Reported gap: the per-status breakdown alone forced the reader to manually
# add up which of the 6 numbers actually meant "needs action" — this drives
# the explicit rollup line instead. - Claude Generated
_ALREADY_COVERED_STATUSES = {"duplicate", "likely_duplicate"}


def _escape_cell(text: Any) -> str:
    """Escape characters that would silently break a Markdown table row."""
    s = str(text or "")
    s = s.replace("|", "\\|")
    s = s.replace("\n", " ").replace("\r", " ")
    return s.strip()


def _format_editions_column(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return "—"
    parts = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        year = m.get("year") or "?"
        pub = m.get("publisher") or m.get("publication") or ""
        text = f"{year} ({pub})" if pub else str(year)
        parts.append(_escape_cell(text))
    return "<br>".join(parts) if parts else "—"


def _format_link_column(matches: List[Dict[str, Any]], url_field: str, label: str) -> str:
    links = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        url = m.get(url_field)
        if url:
            links.append(f"[{label}]({url})")
    return "<br>".join(links) if links else "—"


def _format_authors(authors: Any) -> str:
    if isinstance(authors, list):
        joined = ", ".join(str(a) for a in authors if a)
    else:
        joined = str(authors or "")
    return _escape_cell(joined) if joined else "—"


def _format_isbn(isbn: Any) -> str:
    return _escape_cell(isbn) if isbn else "—"


def _extract_year(match: Dict[str, Any]) -> Optional[int]:
    """Best-effort 4-digit year from a match's `year` field, for sorting.

    Tolerates non-numeric/messy values (e.g. an unparsed date string) —
    returns None rather than raising, so a single bad match can't break the
    "latest edition" lookup for the whole row. - Claude Generated
    """
    year = match.get("year")
    if year is None:
        return None
    digits = "".join(ch for ch in str(year) if ch.isdigit())
    if len(digits) < 4:
        return None
    try:
        return int(digits[:4])
    except ValueError:
        return None


def _latest_match(matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The match with the highest parseable year, or None if none has one."""
    best: Optional[Dict[str, Any]] = None
    best_year = -1
    for m in matches:
        if not isinstance(m, dict):
            continue
        year = _extract_year(m)
        if year is not None and year > best_year:
            best_year, best = year, m
    return best


def _format_latest_edition_column(matches: List[Dict[str, Any]]) -> str:
    """The most recent edition already held, so a librarian can tell at a
    glance whether the wishlist's year is actually newer than what's in the
    catalog — without having to parse the full "Gefundene Auflage(n)" list. - Claude Generated
    """
    latest = _latest_match(matches)
    if not latest:
        return "—"
    year = latest.get("year") or "?"
    pub = latest.get("publisher") or latest.get("publication") or ""
    text = _escape_cell(f"{year} ({pub})" if pub else str(year))
    url = latest.get("web_url")
    return f"[{text}]({url})" if url else text


def _normalize_isbn(isbn: Any) -> str:
    """Strip hyphens/whitespace and uppercase for exact-match comparison —
    the wishlist ISBN may have hyphens the catalog record doesn't (or vice
    versa). Empty string in, empty string out (never a false match)."""
    return "".join(ch for ch in str(isbn or "") if ch.isalnum()).upper()


def apply_isbn_duplicate_override(analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministically force ``status="duplicate"`` on exact ISBN identity.

    The ``analyze_duplicates`` prompt already states this rule ("ISBN-
    Identität ist hinreichender Duplikat-Beweis"), but an LLM can apply a
    prose rule inconsistently across runs/models — this makes the one
    genuinely unambiguous case in the whole classification deterministic
    instead of trusting the model to apply it every time, without touching
    the necessarily judgment-based rest (title/author/year similarity
    without an ISBN still goes through the LLM as before). Every override
    is recorded in ``reasoning``, never silent. - Claude Generated

    Returns:
        ``{"analysis": [...], "overridden_count": N}``
    """
    result: List[Dict[str, Any]] = []
    overridden = 0
    for entry in analysis or []:
        if not isinstance(entry, dict):
            result.append(entry)
            continue
        entry = dict(entry)
        input_metadata = entry.get("input_metadata") or {}
        wishlist_isbn = _normalize_isbn(
            input_metadata.get("isbn") if isinstance(input_metadata, dict) else ""
        )
        matches = entry.get("matches") or []
        has_exact_isbn_match = wishlist_isbn and any(
            isinstance(m, dict) and _normalize_isbn(m.get("isbn")) == wishlist_isbn
            for m in matches
        )
        if has_exact_isbn_match and entry.get("status") != "duplicate":
            overridden += 1
            note = (
                f"[Deterministisch korrigiert: exakte ISBN-Übereinstimmung "
                f"({wishlist_isbn}) ⇒ Duplikat] "
            )
            entry["reasoning"] = note + str(entry.get("reasoning", "") or "")
            entry["status"] = "duplicate"
        result.append(entry)
    return {"analysis": result, "overridden_count": overridden}


def format_duplicate_report_markdown(analysis: List[Dict[str, Any]]) -> str:
    """Build a GFM Markdown table + summary from ``extra.duplicate_analysis``.

    Computes the status counts itself rather than trusting an LLM-supplied
    summary string, so the report is correct whether ``analyze_duplicates``
    ran single-shot or chunked (chunk-merging only keeps the declared
    ``merge_key`` list, so a separately-requested summary field would
    silently vanish once chunking is enabled). - Claude Generated

    Args:
        analysis: the parsed ``analysis`` list — each entry roughly
            ``{"input_title", "status", "matches": [...], "reasoning"}``.

    Returns:
        A Markdown string: one summary line, blank line, then a GFM table
        with one row per wishlist title.
    """
    analysis = analysis or []
    counts = Counter(str(a.get("status", "no_match")) for a in analysis if isinstance(a, dict))
    summary = ", ".join(
        f"{counts.get(status, 0)} {label}" for status, label in _STATUS_LABELS.items()
    )
    already_covered = sum(counts.get(s, 0) for s in _ALREADY_COVERED_STATUSES)
    needs_action = len(analysis) - already_covered

    lines = [
        f"**Zusammenfassung:** {needs_action} von {len(analysis)} Titeln zur Erwerbung "
        f"prüfen/empfehlenswert, {already_covered} bereits ausreichend im Bestand "
        f"(Duplikat/wahrscheinlich Duplikat).",
        f"Im Detail: {summary}.",
        "",
        "| Wunschlisten-Titel | Autor(en) | ISBN | Status | Gefundene Auflage(n) | "
        "Neueste Auflage im Bestand | Katalog-Link | Volltext-Link | Begründung |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in analysis:
        if not isinstance(entry, dict):
            continue
        title = _escape_cell(entry.get("input_title", ""))
        input_metadata = entry.get("input_metadata") or {}
        if not isinstance(input_metadata, dict):
            input_metadata = {}
        authors = _format_authors(input_metadata.get("authors"))
        isbn = _format_isbn(input_metadata.get("isbn"))
        status_raw = str(entry.get("status", ""))
        status = _STATUS_LABELS.get(status_raw, status_raw or "?")
        matches = entry.get("matches") or []
        if not isinstance(matches, list):
            matches = []
        if status_raw in _STATUSES_WITH_RELEVANT_MATCHES:
            editions = _format_editions_column(matches)
            latest_edition = _format_latest_edition_column(matches)
            catalog_links = _format_link_column(matches, "web_url", "Katalog")
            fulltext_links = _format_link_column(matches, "resource_url", "Volltext")
        else:
            # status == "new" means the LLM explicitly determined these
            # matches are a DIFFERENT, unrelated work — not an edition of the
            # wishlist title (matches[] for "no_match" is empty anyway).
            # Showing that work's year/publisher/links in this row would
            # misleadingly read as "the wishlist title is already held". - Claude Generated
            editions = latest_edition = catalog_links = fulltext_links = "—"
        reasoning = _escape_cell(entry.get("reasoning", ""))
        lines.append(
            f"| {title} | {authors} | {isbn} | {status} | {editions} | "
            f"{latest_edition} | {catalog_links} | {fulltext_links} | {reasoning} |"
        )
    return "\n".join(lines)
