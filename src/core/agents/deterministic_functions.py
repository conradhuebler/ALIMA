"""Deterministic functions for Workflow v4 - Claude Generated.

Non-LLM functions registered for ``DeterministicStep``. Each function is
exposed via :func:`register_tool_fn` so YAML workflows can dispatch to
them by name.

Available functions:
    * ``gnd_batch_search`` — SWB + Lobid batch search with local DB enrichment
    * ``dk_classification_twophase`` — DK data collection + single-pass LLM

These are extracted from the legacy ``SearchAgent`` and
``ClassificationAgent`` but decoupled from the SubAgent hierarchy.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.agents.registry import register_tool_fn
from src.core.agents.tool_providers import DKDataProvider

logger = logging.getLogger(__name__)


# ============================================================
# gnd_batch_search — SWB + Lobid + local enrichment
# ============================================================

def _merge_into_pool(
    pool: Dict[str, Dict[str, Any]],
    new_data: Dict[str, Dict[str, Any]],
) -> None:
    """Union GND IDs / DDC / DK codes across pool entries."""
    for title, entry in new_data.items():
        key = title.lower()
        if key in pool:
            existing = pool[key]
            ids = set(existing.get("gnd_ids", [])) | set(entry.get("gnd_ids", []))
            existing["gnd_ids"] = list(ids)
            if ids and not existing.get("gnd_id"):
                existing["gnd_id"] = next(iter(ids))
            existing["ddc_codes"] = list(
                set(existing.get("ddc_codes", [])) | set(entry.get("ddc_codes", []))
            )
            existing["dk_codes"] = list(
                set(existing.get("dk_codes", [])) | set(entry.get("dk_codes", []))
            )
            existing["count"] = max(existing.get("count", 0), entry.get("count", 0))
        else:
            pool[key] = dict(entry)


def _parse_batch_response(raw: str) -> Dict[str, Dict[str, Any]]:
    """Parse SWB/Lobid batch-search JSON response into keyword-level pool."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.warning(f"gnd_batch_search: could not parse response: {e}")
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for term_results in (data.get("results", {}) or {}).values():
        if not isinstance(term_results, dict):
            continue
        for kw_title, kw_data in term_results.items():
            if not isinstance(kw_data, dict):
                continue
            gnd_ids = [str(g) for g in kw_data.get("gndid", []) if g]
            if not gnd_ids and not kw_title:
                continue
            out[kw_title] = {
                "title": kw_title,
                "gnd_ids": gnd_ids,
                "gnd_id": gnd_ids[0] if gnd_ids else "",
                "ddc_codes": list(kw_data.get("ddc", [])),
                "dk_codes": list(kw_data.get("dk", [])),
                "count": kw_data.get("count", 0),
                "description": "",
                "synonyms": [],
            }
    return out


@register_tool_fn("gnd_batch_search")
def gnd_batch_search(
    keywords: List[str],
    *,
    tool_registry: Any = None,
    context: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    sources: Optional[List[str]] = None,
    enrich_from_local_db: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Batch GND search via SWB + Lobid + local DB enrichment.

    Args:
        keywords: list of search terms (strings).
        tool_registry: CachingToolRegistry (injected by DeterministicStep).
        context: SharedContext — merges entries into ``context.gnd_entries``.
        stream_callback: optional progress sink.
        sources: subset of ["swb","lobid"] (default: both).
        enrich_from_local_db: call get_gnd_batch for description/synonyms.

    Returns:
        ``{"entries": [...], "search_terms": [...], "tool_calls": N}``
    """
    if tool_registry is None:
        raise RuntimeError("gnd_batch_search requires tool_registry (caching)")

    if config:
        sources = sources or config.get("sources")
        enrich_from_local_db = config.get("enrich_from_local_db", enrich_from_local_db)

    sources = sources or ["swb", "lobid"]
    source_tools = {"swb": "search_swb", "lobid": "search_lobid"}

    # Accept both ["term1","term2"] and [{"term":"t1"}, {"keyword":"t2"}, {"title":"t3"}].
    # Non-empty dict keys tried in order: term > keyword > title > label.
    def _coerce(k: Any) -> Optional[str]:
        if isinstance(k, str):
            return k or None
        if isinstance(k, dict):
            for field in ("term", "keyword", "title", "label"):
                v = k.get(field)
                if isinstance(v, str) and v:
                    return v
        return None

    coerced = [_coerce(k) for k in (keywords or [])]
    keywords = list(dict.fromkeys(k for k in coerced if k))
    if not keywords:
        return {"entries": [], "search_terms": [], "tool_calls": 0}

    if stream_callback:
        stream_callback(
            f"\n🔍 gnd_batch_search: {len(keywords)} keywords × {len(sources)} sources\n"
        )

    pool: Dict[str, Dict[str, Any]] = {}
    tool_calls = 0

    for src in sources:
        tool_name = source_tools.get(src)
        if not tool_name:
            logger.warning(f"gnd_batch_search: unknown source '{src}'")
            continue
        try:
            raw = tool_registry.execute(tool_name, {"terms": keywords})
            tool_calls += 1
            data = _parse_batch_response(raw)
            _merge_into_pool(pool, data)
            if stream_callback:
                stream_callback(f"  🌐 {src}: {len(data)} hits\n")
        except Exception as e:
            logger.warning(f"gnd_batch_search: {tool_name} failed: {e}")

    if enrich_from_local_db and pool:
        all_ids: Set[str] = set()
        for entry in pool.values():
            all_ids.update(entry.get("gnd_ids", []))
        if all_ids:
            try:
                raw = tool_registry.execute("get_gnd_batch", {"gnd_ids": list(all_ids)})
                tool_calls += 1
                ed = json.loads(raw) if isinstance(raw, str) else raw
                enrich = {
                    gid: {
                        "description": e.get("description", ""),
                        "synonyms": e.get("synonyms", []),
                    }
                    for gid, e in (ed.get("entries") or {}).items()
                }
                for entry in pool.values():
                    for gid in entry.get("gnd_ids", []):
                        r = enrich.get(gid)
                        if not r:
                            continue
                        if r.get("description") and not entry.get("description"):
                            entry["description"] = r["description"]
                        if r.get("synonyms") and not entry.get("synonyms"):
                            entry["synonyms"] = r["synonyms"]
                        break
            except Exception as e:
                logger.warning(f"gnd_batch_search: get_gnd_batch enrichment failed: {e}")

    entries: List[Dict[str, Any]] = list(pool.values())

    if context is not None and hasattr(context, "gnd_entries"):
        existing_titles = {e.get("title", "").lower() for e in context.gnd_entries}
        for entry in entries:
            t = entry.get("title", "").lower()
            if t and t not in existing_titles:
                context.gnd_entries.append(entry)
                existing_titles.add(t)

    if stream_callback:
        enriched = sum(1 for e in entries if e.get("description"))
        stream_callback(
            f"✅ {len(entries)} entries ({enriched} enriched), {tool_calls} tool calls\n"
        )

    return {
        "entries": entries,
        "search_terms": keywords,
        "tool_calls": tool_calls,
    }


# ============================================================
# dk_classification_twophase — DK data collection + LLM call
# ============================================================

@register_tool_fn("dk_data_collect")
def dk_data_collect(
    *,
    tool_registry: Any = None,
    context: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    max_keywords: int = 30,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase-1: batch-collect DK data (cache + catalog + GND entries).

    No LLM call — pure tool dispatch via DKDataProvider.

    Returns:
        ``{"dk_entries": [...], "ddc_from_gnd": [...], "formatted_prompt": "...", "tool_calls": N, "has_data": bool}``
    """
    if tool_registry is None or context is None:
        raise RuntimeError("dk_data_collect requires tool_registry and context")

    if config:
        max_keywords = config.get("max_keywords", max_keywords)

    provider = DKDataProvider(tool_registry, context)
    result = provider.collect(max_keywords=max_keywords)

    if stream_callback:
        stream_callback(
            f"📚 DK data: {len(result.dk_entries)} entries, "
            f"{len(result.ddc_from_gnd)} GND-DDC, "
            f"{result.tool_calls} tool calls\n"
        )

    return {
        "dk_entries": result.dk_entries,
        "ddc_from_gnd": result.ddc_from_gnd,
        "formatted_prompt": result.format_for_prompt(max_entries=30),
        "tool_calls": result.tool_calls,
        "has_data": result.has_data,
    }


# ============================================================
# Classification-result post-processing
# ============================================================

@register_tool_fn("build_dk_search_results")
def build_dk_search_results(
    dk_entries: Optional[List[Dict]] = None,
    dk_classifications: Optional[List[Dict]] = None,
    *,
    context: Any = None,
    config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Merge collected DK cache entries with LLM-assigned classifications.

    Writes to ``context.dk_search_results`` for GUI consumption.
    """
    dk_entries = dk_entries or []
    dk_classifications = dk_classifications or []
    results: List[Dict[str, Any]] = []

    seen: Set[str] = set()
    for e in dk_entries:
        code = e.get("dk", "")
        if code and code not in seen:
            seen.add(code)
            results.append({
                "keyword": e.get("keyword", ""),
                "dk": code,
                "title": e.get("title", ""),
                "count": e.get("count", 0),
                "classification_type": e.get("classification_type", "DK"),
            })

    for cls in dk_classifications:
        code = cls.get("code", "")
        if code:
            results.append({
                "keyword": "",
                "dk": code,
                "title": cls.get("title", ""),
                "count": int(cls.get("confidence", 0) * 100),
                "classification_type": "DK",
                "reasoning": cls.get("reason", cls.get("reasoning", "")),
            })

    if context is not None and hasattr(context, "dk_search_results"):
        context.dk_search_results = results

    return {"results": results, "count": len(results)}


# ============================================================
# catalog_multi_search — Free-query multi-source lookup
# ============================================================

@register_tool_fn("catalog_multi_search")
def catalog_multi_search(
    queries: List[str],
    *,
    tool_registry: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    sources: Optional[List[str]] = None,
    enrich_from_local_db: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Multi-source free-text catalog search (SWB + Lobid + catalog).

    Unlike ``gnd_batch_search`` (which targets keyword lookups for the
    pipeline), this fn also hits the catalog SOAP/SRU and returns a
    result-oriented structure suitable for end-user display.

    Returns:
        ``{"hits": [...], "queries": [...], "tool_calls": N}``.
        Each hit: ``{title, gnd_ids, gnd_id, ddc_codes, dk_codes, count,
        description, synonyms, sources}``.
    """
    if tool_registry is None:
        raise RuntimeError("catalog_multi_search requires tool_registry")

    if config:
        sources = sources or config.get("sources")
        enrich_from_local_db = config.get("enrich_from_local_db", enrich_from_local_db)

    sources = sources or ["swb", "lobid", "catalog"]
    src_tools = {"swb": "search_swb", "lobid": "search_lobid", "catalog": "search_catalog"}

    queries = list(dict.fromkeys(q for q in (queries or []) if q))
    if not queries:
        return {"hits": [], "queries": [], "tool_calls": 0}

    if stream_callback:
        stream_callback(
            f"\n🔎 catalog_multi_search: {len(queries)} queries × {len(sources)} sources\n"
        )

    pool: Dict[str, Dict[str, Any]] = {}
    src_index: Dict[str, Set[str]] = {}
    tool_calls = 0

    for src in sources:
        tool = src_tools.get(src)
        if not tool:
            logger.warning(f"catalog_multi_search: unknown source '{src}'")
            continue
        try:
            raw = tool_registry.execute(tool, {"terms": queries})
            tool_calls += 1
            data = _parse_batch_response(raw)
            for key, entry in data.items():
                k = key.lower()
                src_index.setdefault(k, set()).add(src)
            _merge_into_pool(pool, data)
            if stream_callback:
                stream_callback(f"  🌐 {src}: {len(data)} hits\n")
        except Exception as e:
            logger.warning(f"catalog_multi_search: {tool} failed: {e}")

    if enrich_from_local_db and pool:
        all_ids: Set[str] = set()
        for e in pool.values():
            all_ids.update(e.get("gnd_ids", []))
        if all_ids:
            try:
                raw = tool_registry.execute("get_gnd_batch", {"gnd_ids": list(all_ids)})
                tool_calls += 1
                ed = json.loads(raw) if isinstance(raw, str) else raw
                lookup = ed.get("entries") or {}
                for entry in pool.values():
                    for gid in entry.get("gnd_ids", []):
                        r = lookup.get(gid)
                        if not r:
                            continue
                        if r.get("description") and not entry.get("description"):
                            entry["description"] = r["description"]
                        if r.get("synonyms") and not entry.get("synonyms"):
                            entry["synonyms"] = r["synonyms"]
                        break
            except Exception as e:
                logger.warning(f"catalog_multi_search: enrichment failed: {e}")

    hits: List[Dict[str, Any]] = []
    for key, entry in pool.items():
        entry["sources"] = sorted(src_index.get(key, set()))
        hits.append(entry)
    hits.sort(key=lambda e: e.get("count", 0), reverse=True)

    if stream_callback:
        stream_callback(f"✅ {len(hits)} unique hits, {tool_calls} tool calls\n")

    return {"hits": hits, "queries": queries, "tool_calls": tool_calls}


# ============================================================
# gnd_entry_lookup — Best-match single-keyword GND lookup
# ============================================================

@register_tool_fn("gnd_entry_lookup")
def gnd_entry_lookup(
    keyword: str,
    *,
    tool_registry: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    min_results: int = 3,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Search local GND for ``keyword``, return best-match entry.

    Uses ``search_gnd`` (already ranks by relevance); picks first entry.
    Returns empty fields if nothing found.
    """
    if tool_registry is None:
        raise RuntimeError("gnd_entry_lookup requires tool_registry")

    if config:
        min_results = config.get("min_results", min_results)

    if not keyword:
        return {"found": False, "keyword": keyword, "entry": {}, "tool_calls": 0}

    tool_calls = 0
    try:
        raw = tool_registry.execute("search_gnd", {"term": keyword, "min_results": min_results})
        tool_calls += 1
    except Exception as e:
        logger.warning(f"gnd_entry_lookup: search_gnd failed: {e}")
        return {"found": False, "keyword": keyword, "entry": {}, "tool_calls": tool_calls}

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        data = {}

    entries = data.get("entries") or []
    if not entries:
        if stream_callback:
            stream_callback(f"  ∅ no GND hit for '{keyword}'\n")
        return {"found": False, "keyword": keyword, "entry": {}, "tool_calls": tool_calls}

    best = entries[0]
    if stream_callback:
        stream_callback(f"  ✔ '{keyword}' → {best.get('title')} ({best.get('gnd_id')})\n")

    return {
        "found": True,
        "keyword": keyword,
        "gnd_id": best.get("gnd_id", ""),
        "title": best.get("title", ""),
        "description": best.get("description", ""),
        "synonyms": list(best.get("synonyms", []) or []),
        "ddcs": list(best.get("ddcs", []) or []),
        "entry": best,
        "alternatives": entries[1:5],
        "tool_calls": tool_calls,
    }


# ============================================================
# extract_gnd_related — Pure synonym/hierarchy extractor
# ============================================================

@register_tool_fn("extract_gnd_related")
def extract_gnd_related(
    entry: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Split a GND entry's synonyms/description into related-term buckets.

    No tool calls. Pure structural mapping so downstream LLM steps get
    stable vocabulary.
    """
    entry = entry or {}
    syns = [s for s in (entry.get("synonyms") or []) if s]
    ddcs = [d for d in (entry.get("ddcs") or []) if d]
    return {
        "synonyms": syns,
        "ddcs": ddcs,
        "title": entry.get("title", ""),
        "gnd_id": entry.get("gnd_id", ""),
        "description": entry.get("description", ""),
    }


# ============================================================
# gnd_batch_metadata — Batch metadata with optional Lobid fallback
# ============================================================

@register_tool_fn("gnd_batch_metadata")
def gnd_batch_metadata(
    gnd_ids: List[str],
    *,
    tool_registry: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    lobid_fallback: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bulk-fetch GND metadata via local DB, optional Lobid fallback.

    Returns:
        ``{"entries": {gnd_id: {title, description, synonyms, ddcs}},
           "missing": [...], "count": N, "tool_calls": N}``
    """
    if tool_registry is None:
        raise RuntimeError("gnd_batch_metadata requires tool_registry")

    if config:
        lobid_fallback = config.get("lobid_fallback", lobid_fallback)

    gnd_ids = [g for g in (gnd_ids or []) if g]
    if not gnd_ids:
        return {"entries": {}, "missing": [], "count": 0, "tool_calls": 0}

    if stream_callback:
        stream_callback(f"\n📦 gnd_batch_metadata: {len(gnd_ids)} IDs\n")

    tool_calls = 0
    entries: Dict[str, Dict[str, Any]] = {}

    try:
        raw = tool_registry.execute("get_gnd_batch", {"gnd_ids": gnd_ids})
        tool_calls += 1
        data = json.loads(raw) if isinstance(raw, str) else raw
        entries = dict(data.get("entries") or {})
    except Exception as e:
        logger.warning(f"gnd_batch_metadata: get_gnd_batch failed: {e}")

    missing = [g for g in gnd_ids if g not in entries]

    if lobid_fallback and missing:
        if stream_callback:
            stream_callback(f"  ↪ lobid fallback for {len(missing)} IDs\n")
        for gid in list(missing):
            try:
                raw = tool_registry.execute("search_lobid", {"terms": [gid]})
                tool_calls += 1
                data = json.loads(raw) if isinstance(raw, str) else raw
                for _term, kws in (data.get("results") or {}).items():
                    for title, kw in (kws or {}).items():
                        if gid in (kw.get("gndid") or []):
                            entries[gid] = {
                                "title": title,
                                "description": "",
                                "synonyms": [],
                                "ddcs": list(kw.get("ddc", [])),
                            }
                            missing.remove(gid)
                            break
                    if gid in entries:
                        break
            except Exception as e:
                logger.debug(f"gnd_batch_metadata: lobid fallback for {gid} failed: {e}")

    if stream_callback:
        stream_callback(
            f"✅ {len(entries)} entries, {len(missing)} missing, {tool_calls} tool calls\n"
        )

    return {
        "entries": entries,
        "missing": missing,
        "count": len(entries),
        "tool_calls": tool_calls,
    }


def register_all() -> None:
    """Idempotent registration — importing this module is enough."""
    # Registration happens at import time via decorators.
    logger.debug("deterministic_functions: registered")


register_all()
