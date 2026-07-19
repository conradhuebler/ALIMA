"""Counter + provenance aggregation over the raw response cache (WP2 P4).

Operator model: the free-form agent can consume raw JSON directly; the *pipeline*
needs the data *aufbereitet* — the reduced ``{subject: …}`` view **plus** counter
statistics and provenance (which sources confirmed a keyword, how often). That
reduction is this integrable tool, sitting on top of the WP2 raw cache so **raw is
the single source of truth**.

Pipeline: for each (source, term) read the cached raw response, apply the source's
pure ``transform(raw)`` (``LobidSuggester``/``SWBSuggester``/``BiblioSuggester``),
fold the per-source reduced views into one pool via
:mod:`src.core.gnd_search_core` (max count, union of GND/DDC/DK codes, source
provenance), and rank.

⚠️ **Count-landmine:** derived entries are a *cache read*, so — exactly like
``CachingProvider._items_from_cache`` — the pool ``count`` is forced to ``1`` and
the real Häufigkeit rides in ``display_count`` (never read by ranking). This keeps
the pool byte-compatible with the mapping-first read path.

The per-source transforms are **injected** (``transform_by_source``) so this module
has no dependency on the suggester layer — callers wire the suggesters they already
built.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..gnd_search_core import merge_into_pool, parse_batch_response, rank_pool

logger = logging.getLogger(__name__)

# {source: transform(raw_dict) -> {subject: {count, gnd_ids, classifications}}}
TransformMap = Dict[str, Callable[[Dict[str, Any]], Dict[str, Dict[str, Any]]]]


def default_aggregate_from_raw() -> bool:
    """Global default for the raw-first pool read path (``SystemConfig.aggregate_from_raw``).

    Both pipelines fall back to this when no explicit per-call override is given, so
    an operator can flip the P4 convergence from config/GUI. Any config-load failure
    defaults to True (converged). - Claude Generated
    """
    try:
        from src.utils.config_manager import ConfigManager

        cfg = ConfigManager().load_config()
        return bool(getattr(cfg.system_config, "aggregate_from_raw", True))
    except Exception:
        return True


def _reduced_from_mapping(
    ukm: Any, source: str, term: str, max_age_hours: Optional[int]
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fallback reduced view from the mapping index when raw is unavailable.

    Raw can be absent even with a fresh mapping — a size-capped/pruned raw blob or
    pre-WP2 data. Dropping those terms would silently lose results, so we
    reconstruct the reduced view from the mapping exactly like
    ``CachingProvider._items_from_cache`` (dedup by ``gnd_entries`` title, real
    count from ``gnd_counts``). Returns None on miss/stale. - Claude Generated
    """
    try:
        mapping = ukm.get_search_mapping(term, source)
    except Exception:
        return None
    if mapping is None:
        return None
    if max_age_hours is not None and not ukm._raw_is_fresh(
        getattr(mapping, "last_updated", None), max_age_hours
    ):
        return None
    gnd_counts = getattr(mapping, "gnd_counts", {}) or {}
    by_title: Dict[str, Dict[str, Any]] = {}
    for gnd_id in getattr(mapping, "found_gnd_ids", []) or []:
        fact = ukm.get_gnd_fact(gnd_id)
        if not fact:
            continue
        title = fact.title
        cnt = gnd_counts.get(gnd_id)
        if title in by_title:
            by_title[title]["gnd_ids"].add(gnd_id)
            if cnt is not None:
                by_title[title]["count"] = max(by_title[title]["count"], int(cnt))
        else:
            by_title[title] = {
                "count": int(cnt) if cnt is not None else 1,
                "gnd_ids": {gnd_id},
                "classifications": {},
            }
    return by_title or None


def _to_cache_hit_shape(reduced: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Rewrite a reduced view to mapping-cache-hit semantics (count-landmine).

    Pool ``count`` becomes ``1``; the source's real count moves to
    ``display_count`` (display-only, never ranked). Mirrors
    ``CachingProvider._items_from_cache``. - Claude Generated
    """
    out: Dict[str, Dict[str, Any]] = {}
    for subject, data in reduced.items():
        real = data.get("count", 0) or 0
        entry = dict(data)
        entry["display_count"] = real
        entry["count"] = 1
        out[subject] = entry
    return out


def aggregate_gnd_results(
    terms: Sequence[str],
    sources: Sequence[str],
    ukm: Any,
    transform_by_source: TransformMap,
    *,
    params_by_source: Optional[Dict[str, Dict[str, Any]]] = None,
    max_age_hours: Optional[int] = 24,
) -> Dict[str, Any]:
    """Build the ranked GND pool (counter + provenance) from the raw cache.

    Returns ``{"pool": [ranked entries], "sources": [...], "missing":
    {source: [terms with no cached raw]}, "terms_map": {title: [terms]}}``.
    Sources with no injected transform are skipped. Never raises on a single bad
    blob (it is skipped). Raw is the single source of truth. - Claude Generated
    """
    pool: Dict[str, Dict[str, Any]] = {}
    src_index: Dict[str, set] = {}
    terms_map: Dict[str, set] = {}
    missing: Dict[str, List[str]] = {}
    params_by_source = params_by_source or {}

    for source in sources:
        transform = transform_by_source.get(source)
        if transform is None:
            continue
        params = params_by_source.get(source, {})
        for term in terms:
            cached = ukm.get_raw_response(source, term, params, max_age_hours=max_age_hours)
            reduced: Optional[Dict[str, Dict[str, Any]]] = None
            if cached:
                try:
                    reduced = transform(json.loads(cached["raw_json"]))
                except Exception as exc:  # bad blob must not sink the aggregation
                    logger.warning("transform failed for '%s' (%s): %s", term, source, exc)
                    reduced = None
            if reduced is None:
                # Raw absent/stale/unparsable → fall back to the mapping index so
                # size-capped / pruned / pre-WP2 terms are not silently dropped.
                reduced = _reduced_from_mapping(ukm, source, term, max_age_hours)
            if reduced is None:
                missing.setdefault(source, []).append(term)
                continue
            new_data = parse_batch_response({"results": {term: _to_cache_hit_shape(reduced)}})
            for title in new_data:
                src_index.setdefault(title.lower(), set()).add(source)
                terms_map.setdefault(title, set()).add(term)
            merge_into_pool(pool, new_data)

    ranked = rank_pool(pool, src_index)
    return {
        "pool": ranked,
        "sources": list(sources),
        "missing": missing,
        "terms_map": {title: sorted(t) for title, t in terms_map.items()},
    }


def nested_from_aggregate(agg: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Reshape an aggregate result to the classic nested ``{term:{title:{...}}}`` view.

    Inverts ``terms_map`` (title→terms) so each search term maps to its confirmed
    titles with the reduced fields in the canonical shape (``gnd_ids`` /
    ``classifications`` as fresh sets per term, ``count``, optional
    ``display_count``). This is the classic GUI/CLI/Webapp contract, now derived
    from the raw-first pool. - Claude Generated
    """
    by_title = {e.get("title"): e for e in agg.get("pool", [])}
    nested: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for title, terms in (agg.get("terms_map") or {}).items():
        entry = by_title.get(title)
        if entry is None:
            continue
        cls = entry.get("classifications") or {}
        for term in terms:
            reduced = {
                "count": entry.get("count", 1),
                "gnd_ids": set(entry.get("gnd_ids", [])),
                "classifications": {
                    system: set(codes) for system, codes in cls.items() if codes
                },
            }
            if entry.get("display_count") is not None:
                reduced["display_count"] = entry["display_count"]
            nested.setdefault(term, {})[title] = reduced
    return nested
