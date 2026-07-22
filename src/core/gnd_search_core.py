"""Shared GND-search aggregation primitives - Claude Generated.

Both the classic pipeline (``SearchCLI`` / ``PipelineStepExecutor.execute_gnd_search``)
and the agentic v4 path (``deterministic_functions.gnd_batch_search`` /
``catalog_multi_search``) already run their searches through the *same* engine —
``MetaSuggester`` with mapping-first caching. What used to be duplicated was the
**aggregation layer** on top of that engine: merging a keyword/title entry
(max hit-count + union of GND/DDC/DK codes), folding per-source results into a
single pool, and ranking that pool.

This module is the single home for those primitives. It has **no** dependency on
``pipeline_utils``, ``deterministic_functions`` or ``search_cli`` so either core can
import it without a cycle.

⚠️ **Count-Landmine (see ``src/core/CLAUDE.md``):** the per-entry ``count`` and the
``source_count`` produced here drive the agentic ranking
(``selection_chunks`` → ``selection``) and the classic chunk ordering. Do **not**
change how ``count`` is derived (it must stay ``max`` across confirmations, never a
sum) — a naive max/merge change collapses *chunk* and *final* into the same set.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.utils.classification_systems import normalize_classifications

logger = logging.getLogger(__name__)


def merge_code_entry(
    target: Dict[str, Any],
    source: Dict[str, Any],
    code_fields: Sequence[str],
    count_field: str = "count",
    display_count_field: Optional[str] = None,
    classifications_field: Optional[str] = None,
) -> None:
    """Merge one search entry into another in place — the shared merge-atom.

    Used by both the classic nested merge (``SearchCLI.merge_results`` /
    ``search.service`` cross-source merge, where the code fields are ``set``s) and
    the agentic pool merge (``merge_into_pool``, where they are ``list``s). The
    container type of ``target[field]`` is preserved: ``set`` → ``set.update``;
    ``list`` → order-preserving dedup append.

    ``count`` is combined with ``max`` (never summed) — this is the count-semantics
    the selection/chunking ranking relies on; see the module docstring.

    ``display_count_field`` (opt-in) max-merges the display-only F-4 count across
    sources — the single-merge home for what ``MetaSuggester._merge_suggester_results``
    did inline. Only carried when a source provides it (display-only, never ranked).

    ``classifications_field`` (opt-in, WP-D1; entry shape since WP-D2) merges the
    canonical ``{system: [{code, count?, origin}]}`` dict per equal-rank system
    via ``classification_systems.merge_classifications``: dedup by ``code``,
    ``count`` **max-merged like the entry count above** (never summed), and
    ``origin`` keeping the stronger claim (authority over co-occurrence).
    Unlike ``code_fields`` there is no set/list duality here — entries are dicts,
    so every container is an ordered list. - Claude Generated
    """
    if count_field:
        target[count_field] = max(
            target.get(count_field, 0) or 0, source.get(count_field, 0) or 0
        )
    if display_count_field:
        dc = source.get(display_count_field)
        if dc is not None:
            cur = target.get(display_count_field)
            target[display_count_field] = int(dc) if cur is None else max(int(cur), int(dc))
    for field in code_fields:
        new_vals: Iterable[Any] = source.get(field) or []
        existing = target.get(field)
        target[field] = _merge_codes(existing, new_vals)
    if classifications_field:
        src_cls = source.get(classifications_field) or {}
        if src_cls:
            # merge_classifications builds a new dict throughout (copy-on-write):
            # pool inserts are shallow ``dict(entry)`` copies, so mutating the
            # stored dict in place would leak into the source entry.
            from src.utils.classification_systems import merge_classifications

            target[classifications_field] = merge_classifications(
                target.get(classifications_field), src_cls
            )


def _merge_codes(existing: Any, new_vals: Iterable[Any]) -> Any:
    """Union ``new_vals`` into ``existing``, preserving the container type:
    ``set`` → ``set.update``; ``list`` → order-preserving dedup append. When the
    target has no container yet (sparse ``classifications``), the SOURCE type
    wins — a set source must not degrade to a nondeterministically ordered list.
    - Claude Generated"""
    if isinstance(existing, set):
        existing.update(new_vals)
        return existing
    if existing is None and isinstance(new_vals, set):
        return set(new_vals)
    merged: List[Any] = list(existing or [])
    seen: Set[Any] = set(merged)
    for value in new_vals:
        if value not in seen:
            merged.append(value)
            seen.add(value)
    return merged


def merge_into_pool(
    pool: Dict[str, Dict[str, Any]],
    new_data: Dict[str, Dict[str, Any]],
) -> None:
    """Fold a per-source title→entry map into the running ``pool`` (keyed by
    lower-cased title). Unions ``gnd_ids`` and per-system ``classifications``
    codes, keeps the max count.
    """
    for title, entry in new_data.items():
        key = title.lower()
        if key in pool:
            existing = pool[key]
            merge_code_entry(
                existing, entry, code_fields=("gnd_ids",),
                classifications_field="classifications",
            )
            _merge_display_count(existing, entry)
            if existing.get("gnd_ids") and not existing.get("gnd_id"):
                existing["gnd_id"] = existing["gnd_ids"][0]
        else:
            pool[key] = dict(entry)


def _merge_display_count(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Max-merge the display-only ``display_count`` across sources (F-4).

    Each side's "real" count is its ``display_count`` if set, else its pool
    ``count``. We only annotate ``target`` when the result exceeds the pool count,
    so the entry shape is unchanged in the common all-live (no-cache) case. This is
    display metadata only — ``rank_pool`` never reads it.
    """
    t_eff = target.get("display_count")
    if t_eff is None:
        t_eff = target.get("count", 0) or 0
    s_eff = source.get("display_count")
    if s_eff is None:
        s_eff = source.get("count", 0) or 0
    best = max(int(t_eff), int(s_eff))
    if best > (target.get("count", 0) or 0):
        target["display_count"] = best


def pool_entry_from_reduced(kw_title: str, kw_data: Dict[str, Any]) -> Dict[str, Any]:
    """THE nested→pool ingestion point: build a canonical pool entry from one
    reduced suggester keyword payload (contract v2). Since the F-1 collapse
    (WP-D1) this converts *representation* (nested sets / serialized lists →
    pool lists, ``gnd_id`` convenience), not names — there is no rename layer
    anymore."""
    gnd_ids = [str(g) for g in kw_data.get("gnd_ids", []) if g]
    # Canonical classifications (WP-D1 shape, WP-D2 entries):
    # {system: [{code, count?, origin}]}, systems equal-rank, only non-empty
    # carried. ``normalize_classifications`` accepts every producer form —
    # entries, or bare code sets/lists from producers with no evidence to report
    # — and normalises the system spelling, so this stays THE single ingestion
    # point rather than growing per-source branches.
    classifications = normalize_classifications(kw_data.get("classifications"))
    entry = {
        "title": kw_title,
        "gnd_ids": gnd_ids,
        "gnd_id": gnd_ids[0] if gnd_ids else "",
        "classifications": classifications,
        "count": kw_data.get("count", 0),
        "description": "",
        "synonyms": [],
    }
    # F-4: carry the display-only count when present (mapping-cache hits). It rides
    # into ``gnd_entries`` for the display layer but is NEVER read by ``rank_pool``
    # — see the count-landmine note in the module docstring. Absent when no cache.
    display_count = kw_data.get("display_count")
    if display_count is not None:
        entry["display_count"] = display_count
    return entry


def merge_authority_ddc(entries: Iterable[Dict[str, Any]], ddcs_by_gid: Dict[str, Any]) -> int:
    """Merge the local GND store's authority DDC onto pool entries, by GND-ID.

    Shared by the classic and agentic search paths. Each pool entry carries the
    ``gnd_ids`` of its subject; the store holds an authority DDC per GND-ID
    (``ddcs_by_gid``, the ``gnd_entries.ddcs`` TEXT column served by
    ``get_gnd_facts_batch`` / ``get_gnd_batch``). Without this the whole
    authority path is dead-ended: the store is filled (migration + DNB
    enrichment) and served, but nobody reads its DDC back onto the subjects, so
    every pool classification stays ``cooccurrence`` and a term like "Cadmium"
    never gets its authority DDC.

    Merges from EVERY gnd_id of an entry (merged spellings can each carry one).
    ``origin="authority"`` outranks the co-occurrence harvest on the same code
    (``merge_classifications`` handles precedence). In place; returns how many
    entries gained a DDC. - Claude Generated
    """
    from src.utils.classification_systems import merge_classifications, parse_stored_ddcs

    touched = 0
    for entry in entries:
        gained = False
        for gid in entry.get("gnd_ids", []) or []:
            ddc_entries = parse_stored_ddcs(ddcs_by_gid.get(gid))
            if ddc_entries:
                entry["classifications"] = merge_classifications(
                    entry.get("classifications"), {"DDC": ddc_entries}
                )
                gained = True
        touched += gained
    return touched


def parse_batch_response(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Parse a SWB/Lobid batch-search JSON response into a title→entry pool.

    Accepts the serialized tool response (``{"results": {term: {title: {...}}}}``)
    either as a JSON string or already-decoded dict.
    """
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        logger.warning(f"parse_batch_response: could not parse response: {exc}")
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for term_results in (data.get("results", {}) or {}).values():
        if not isinstance(term_results, dict):
            continue
        for kw_title, kw_data in term_results.items():
            if not isinstance(kw_data, dict):
                continue
            entry = pool_entry_from_reduced(kw_title, kw_data)
            if not entry["gnd_ids"] and not kw_title:
                continue
            out[kw_title] = entry
    return out


def parse_batch_response_with_terms(
    raw: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Like :func:`parse_batch_response` but also returns, per title, the list of
    search terms that produced it (drives per-keyword GUI display).
    """
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as exc:
        logger.warning(f"parse_batch_response_with_terms: could not parse response: {exc}")
        return {}, {}

    out: Dict[str, Dict[str, Any]] = {}
    terms_per_title: Dict[str, List[str]] = {}
    for term, term_results in (data.get("results", {}) or {}).items():
        if not isinstance(term_results, dict):
            continue
        for kw_title, kw_data in term_results.items():
            if not isinstance(kw_data, dict):
                continue
            entry = pool_entry_from_reduced(kw_title, kw_data)
            if not entry["gnd_ids"] and not kw_title:
                continue
            out[kw_title] = entry
            terms_per_title.setdefault(kw_title, []).append(term)
    return out, terms_per_title


def rank_pool(
    pool: Dict[str, Dict[str, Any]],
    src_index: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    """Attach source provenance and return pool entries ranked for selection.

    Each entry gets ``sources`` (sorted source names that confirmed the title) and
    ``source_count`` (= ``len(sources)``). Entries are ranked by
    ``(source_count, count)`` descending — multi-source confirmations first, then
    hit count. This ordering feeds the agentic ``selection_chunks`` → ``selection``
    flow; see the module docstring's count-landmine note before changing it.
    """
    for key, entry in pool.items():
        entry["sources"] = sorted(src_index.get(key, set()))
        entry["source_count"] = len(entry["sources"])
    return sorted(
        pool.values(),
        key=lambda e: (e.get("source_count", 0), e.get("count", 0)),
        reverse=True,
    )
