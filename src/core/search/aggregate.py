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

# {source: transform(raw_dict) -> {subject: {count, gndid, ddc, dk}}}
TransformMap = Dict[str, Callable[[Dict[str, Any]], Dict[str, Dict[str, Any]]]]


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
    {source: [terms with no cached raw]}}``. Sources with no injected transform
    are skipped. Never raises on a single bad blob (it is skipped). Raw is the
    single source of truth. - Claude Generated
    """
    pool: Dict[str, Dict[str, Any]] = {}
    src_index: Dict[str, set] = {}
    missing: Dict[str, List[str]] = {}
    params_by_source = params_by_source or {}

    for source in sources:
        transform = transform_by_source.get(source)
        if transform is None:
            continue
        params = params_by_source.get(source, {})
        for term in terms:
            cached = ukm.get_raw_response(source, term, params, max_age_hours=max_age_hours)
            if not cached:
                missing.setdefault(source, []).append(term)
                continue
            try:
                raw = json.loads(cached["raw_json"])
            except (ValueError, TypeError):
                continue
            try:
                reduced = transform(raw)
            except Exception as exc:  # a bad blob must not sink the whole aggregation
                logger.warning("transform failed for '%s' (%s): %s", term, source, exc)
                continue
            new_data = parse_batch_response({"results": {term: _to_cache_hit_shape(reduced)}})
            for title in new_data:
                src_index.setdefault(title.lower(), set()).add(source)
            merge_into_pool(pool, new_data)

    ranked = rank_pool(pool, src_index)
    return {"pool": ranked, "sources": list(sources), "missing": missing}
