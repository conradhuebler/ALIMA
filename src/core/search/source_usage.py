"""Per-run overview of the search sources: requests, cache, network, hits.

Claude Generated.

For every source (the plugin instance id, the key ``source_health`` uses) this
module counts what the terms asked of it in the current run came to:

* ``cache``: answered from a local cache (GND mapping cache, raw response
  cache, DK title cache, lookup cache, the precomputed DK files of Libero);
* ``network``: at least one request went to the server for the term (a failed
  request included, which also counts under ``errors``);
* ``skipped``: not asked because ``source_health`` had marked the source down;
* ``cache_misses``: no entry in the state "Nur Cache" (no request);
* ``errors``: the term failed (a failed request, or no ISIL set where the
  source needs one).

Lookup tools (``rvk_*``, ``k10plus_resolve``, ``geo_sachdb_*`` …) are counted
per call as ``cache`` or as a plain call: whether a plugin asked a server or
read a local file is not known at the tool handler.

``requests`` is the number of terms asked. ``hits`` counts the keywords that
got at least one result (GND keyword search, DK step); ``records`` the title
records delivered (title searches). ``down`` is set when ``source_health``
marked the source down during the run.

The counting sits where the cache/network decision is taken: the provider base
(``SuggesterBackedProvider``), the mapping cache (``CachingProvider``), the DK
extractors (``BiblioClient``, ``FincCatalogClient``, ``MarcXmlClient``,
``PicaDkExtractor``), the title-record tool handlers, ``search_titles`` and the
lookup tool handler. A client built without an instance id (``source`` None) is
not counted.

State is process-wide (one dict, one lock), like ``run_timing`` and
``source_health``; it is reset together with ``source_health`` at a run start
(``source_health.begin_run`` and the outermost ``source_health.run_scope``), so
the overview covers the same run as the skip. Qt-free.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

#: Origins of one term - Claude Generated
CACHE = "cache"
NETWORK = "network"
SKIPPED = "skipped"
CACHE_MISS = "cache_miss"
NOT_CONFIGURED = "not_configured"
#: A lookup call that did not come from the cache; whether it reached a
#: server or a local file is not known (lookup tools only).
CALL = "call"

#: What a term produced: keywords with hits, title records, a lookup answer.
KIND_KEYWORDS = "keywords"
KIND_RECORDS = "records"
KIND_LOOKUP = "lookup"

#: Title of the block in the result card, the CLI and the chat line.
TITLE = "Quellen dieses Laufs"

#: Rows above this number are shown collapsed in the result card.
COLLAPSE_ABOVE = 5


@dataclass
class _Usage:
    requests: int = 0
    cache: int = 0
    network: int = 0
    hits: int = 0
    records: int = 0
    errors: int = 0
    skipped: int = 0
    cache_misses: int = 0
    down: bool = False
    kinds: Set[str] = field(default_factory=set)


_lock = threading.Lock()
_state: Dict[str, _Usage] = {}
_labels: Dict[str, str] = {}


def reset() -> None:
    """Forget the counts (start of a run); labels stay - Claude Generated"""
    with _lock:
        _state.clear()


def set_label(source: Optional[str], label: Optional[str]) -> None:
    """Name shown for ``source`` (the instance label, else the type label).

    Set by ``factory.build_provider``. - Claude Generated
    """
    if source and label:
        with _lock:
            _labels[str(source)] = str(label)


def label_of(source: str) -> str:
    """The display name of ``source`` (its id when none is known) - Claude Generated"""
    with _lock:
        return _labels.get(source) or source


def _entry(source: str) -> _Usage:
    entry = _state.get(source)
    if entry is None:
        entry = _state[source] = _Usage()
    return entry


def record(
    source: Optional[str],
    origin: str,
    *,
    kind: str = KIND_KEYWORDS,
    hit: bool = False,
    records: int = 0,
    error: bool = False,
) -> None:
    """Count one term asked of ``source`` - Claude Generated

    ``origin`` is one of :data:`CACHE`, :data:`NETWORK`, :data:`SKIPPED`,
    :data:`CACHE_MISS`, :data:`NOT_CONFIGURED`, :data:`CALL`. ``hit`` (a keyword with at
    least one result) and ``records`` (title records) count only for an
    answered term. A ``None`` source is ignored.
    """
    if not source:
        return
    with _lock:
        entry = _entry(str(source))
        entry.requests += 1
        entry.kinds.add(kind)
        if origin == CACHE:
            entry.cache += 1
        elif origin == NETWORK:
            entry.network += 1
        elif origin == SKIPPED:
            entry.skipped += 1
        elif origin == CACHE_MISS:
            entry.cache_misses += 1
        elif origin == NOT_CONFIGURED:
            entry.errors += 1
        if error and origin in (CACHE, NETWORK, CALL):
            entry.errors += 1
        if not error and origin in (CACHE, NETWORK, CALL):
            if hit:
                entry.hits += 1
            entry.records += max(0, int(records or 0))


def origin_of_error(message: Any) -> Optional[str]:
    """The origin a per-term error message stands for, ``None`` for a plain
    failure: the source-down, cache-only and institution markers - Claude Generated"""
    from src.core.search.cache_only import is_cache_miss
    from src.core.search.source_health import is_source_down

    if is_source_down(message):
        return SKIPPED
    if is_cache_miss(message):
        return CACHE_MISS
    try:
        from src.utils.institution import is_not_set
    except Exception:  # noqa: BLE001 - the counter must not break a search
        return None
    return NOT_CONFIGURED if is_not_set(message) else None


def record_terms(
    source: Optional[str],
    terms: Iterable[str],
    *,
    errors: Optional[Mapping[str, Any]] = None,
    hits: Optional[Mapping[str, Any]] = None,
    records: Optional[Mapping[str, Any]] = None,
    cache_only: bool = False,
    kind: str = KIND_KEYWORDS,
) -> None:
    """Count a batch of terms from a provider result - Claude Generated

    ``errors`` maps a term to its per-term error message (markers decide the
    origin, any other message is a failed request); ``hits`` maps a term to its
    result (truthy = a hit); ``records`` maps a term to its record count.
    ``cache_only``: the answers came from the cache without a request.
    """
    if not source:
        return
    errors = errors or {}
    answered = CACHE if cache_only else NETWORK
    for term in terms:
        message = errors.get(term)
        if message:
            origin = origin_of_error(message)
            if origin is None:
                record(source, answered, kind=kind, error=True)
            else:
                record(source, origin, kind=kind)
            continue
        count = int((records or {}).get(term) or 0)
        record(source, answered, kind=kind,
               hit=bool((hits or {}).get(term)) or count > 0, records=count)


def mark_down(source: Optional[str]) -> None:
    """``source_health`` marked ``source`` down in this run - Claude Generated"""
    if not source:
        return
    with _lock:
        _entry(str(source)).down = True


def rows() -> List[Dict[str, Any]]:
    """The overview, one dict per source with a request or a skip, in the
    order the sources were first used - Claude Generated"""
    with _lock:
        items = [(source, entry) for source, entry in _state.items()
                 if entry.requests or entry.down]
        labels = dict(_labels)
    out: List[Dict[str, Any]] = []
    for source, entry in items:
        row = {
            "source": source,
            "label": labels.get(source) or source,
            "requests": entry.requests,
            "cache": entry.cache,
            "network": entry.network,
            "hits": entry.hits,
            "records": entry.records,
            "errors": entry.errors,
            "skipped": entry.skipped,
            "cache_misses": entry.cache_misses,
            "down": entry.down,
            "kinds": sorted(entry.kinds),
        }
        row["text"] = row_text(row)
        out.append(row)
    return out


def row_text(row: Mapping[str, Any]) -> str:
    """One line per source, e.g. ``finc · 38 Anfragen · 31 Cache · 7 Netz ·
    3 Treffer-Schlagwörter · 0 Fehler`` - Claude Generated"""
    from src.core.search.source_health import FAILURE_THRESHOLD

    label = str(row.get("label") or row.get("source") or "?")
    requests = int(row.get("requests") or 0)
    skipped = int(row.get("skipped") or 0)
    down_text = f"übersprungen nach {FAILURE_THRESHOLD} Verbindungsfehlern"
    if row.get("down") and requests == skipped:
        return f"{label} · {down_text}" + (f" ({skipped} Anfragen)" if skipped else "")
    kinds = set(row.get("kinds") or [])
    if kinds == {KIND_LOOKUP}:
        # Lookup tools: cache or a call of the plugin (server or local file).
        return " · ".join([label, f"{requests} Anfragen", f"{int(row.get('cache') or 0)} Cache",
                           f"{int(row.get('errors') or 0)} Fehler"])
    parts = [label, f"{requests} Anfragen", f"{int(row.get('cache') or 0)} Cache",
             f"{int(row.get('network') or 0)} Netz"]
    if KIND_RECORDS in kinds:
        parts.append(f"{int(row.get('records') or 0)} Datensätze")
    if KIND_KEYWORDS in kinds or not kinds:
        parts.append(f"{int(row.get('hits') or 0)} Treffer-Schlagwörter")
    parts.append(f"{int(row.get('errors') or 0)} Fehler")
    misses = int(row.get("cache_misses") or 0)
    if misses:
        parts.append(f"{misses} nicht im Cache")
    if row.get("down"):
        parts.append(f"{skipped} {down_text}" if skipped else down_text)
    elif skipped:
        parts.append(f"{skipped} übersprungen")
    return " · ".join(parts)


def row_lines(source_rows: Optional[Iterable[Mapping[str, Any]]]) -> List[str]:
    """The text of each stored row (rebuilt when a row carries none) - Claude Generated"""
    out = []
    for row in source_rows or []:
        if not isinstance(row, Mapping):
            continue
        out.append(str(row.get("text") or row_text(row)))
    return out


def summary_line(source_rows: Optional[Iterable[Mapping[str, Any]]] = None) -> str:
    """``"Quellen dieses Laufs: <row> | <row>"``, empty without rows - Claude Generated"""
    lines = row_lines(rows() if source_rows is None else source_rows)
    return f"{TITLE}: " + " | ".join(lines) if lines else ""


def attach_to(context: Any) -> None:
    """Store the current rows in ``context.extra["source_usage"]`` (read by
    ``SharedContext.to_keyword_analysis_state``); best-effort - Claude Generated"""
    try:
        extra = getattr(context, "extra", None)
        if isinstance(extra, dict):
            extra["source_usage"] = rows()
    except Exception:  # noqa: BLE001 - the overview must never break a run
        pass


def announce() -> str:
    """Send :func:`summary_line` as ``state.notice`` (GUI log, webapp, CLI
    stderr) when a source was asked; returns the line. Used at the end of a
    chat turn. Best-effort - Claude Generated"""
    line = summary_line()
    if not line:
        return ""
    try:
        from src.core.state_bus import AlimaStateBus

        AlimaStateBus().emit_event("state.notice", {"text": line, "level": "info"})
    except Exception as exc:  # noqa: BLE001 - a missing bus must not break a run
        import logging

        logging.getLogger(__name__).debug(f"source overview notice not emitted: {exc}")
    return line
