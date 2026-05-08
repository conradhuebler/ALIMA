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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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


def _parse_batch_response_with_terms(raw: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    """Parse SWB/Lobid response into pool + track which search term found each title."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.warning(f"gnd_batch_search: could not parse response: {e}")
        return {}, {}

    out: Dict[str, Dict[str, Any]] = {}
    terms_per_title: Dict[str, List[str]] = {}
    for term, term_results in (data.get("results", {}) or {}).items():
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
            terms_per_title.setdefault(kw_title, []).append(term)
    return out, terms_per_title


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
    if config:
        source_tools = config.get("source_tool_map", source_tools)

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
        kw_preview = ", ".join(keywords[:8])
        more = f" … +{len(keywords)-8}" if len(keywords) > 8 else ""
        stream_callback(
            f"\n🔍 gnd_batch_search: {len(keywords)} keywords × {len(sources)} sources\n"
            f"   Suche: {kw_preview}{more}\n"
        )

    pool: Dict[str, Dict[str, Any]] = {}
    tool_calls = 0
    # Track which search term found which titles (for per-keyword GUI display)
    entries_per_keyword: Dict[str, List[str]] = {}

    for src in sources:
        tool_name = source_tools.get(src)
        if not tool_name:
            logger.warning(f"gnd_batch_search: unknown source '{src}'")
            continue
        try:
            raw = tool_registry.execute(tool_name, {"terms": keywords})
            tool_calls += 1
            data, terms_map = _parse_batch_response_with_terms(raw)
            _merge_into_pool(pool, data)
            # Track term-to-title mapping for per-keyword display
            for title, terms in terms_map.items():
                for term in terms:
                    entries_per_keyword.setdefault(term, []).append(title)
            # Log per-keyword hit counts
            if stream_callback:
                stream_callback(f"  🌐 {src}: {len(data)} hits\n")
                for idx, kw in enumerate(keywords, 1):
                    hits = [t for t, terms in terms_map.items() if kw in terms]
                    if hits:
                        stream_callback(
                            f"    [{idx}/{len(keywords)}] '{kw}': {len(hits)} Treffer\n"
                        )
                    else:
                        stream_callback(
                            f"    [{idx}/{len(keywords)}] '{kw}': ∅\n"
                        )
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

    # Store per-keyword term-to-title mapping for correct UI display
    if context is not None and hasattr(context, "gnd_entries_per_keyword"):
        existing = context.gnd_entries_per_keyword or {}
        for term, titles in entries_per_keyword.items():
            current = set(existing.get(term, []))
            current.update(titles)
            existing[term] = list(current)
        context.gnd_entries_per_keyword = existing

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

    provider = DKDataProvider(tool_registry, context, stream_callback=stream_callback)
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
        "formatted_prompt": result.format_for_prompt(max_entries=60),
        "tool_calls": result.tool_calls,
        "has_data": result.has_data,
    }


def _parse_keyword_string(kw: str) -> Tuple[str, str]:
    """Parse 'Term (GND-ID: id)' into (term, gnd_id)."""
    import re
    m = re.search(r"\(GND-ID:\s*([^)]+)\)", kw)
    if m:
        term = kw[:m.start()].strip()
        return term, m.group(1).strip()
    return kw.strip(), ""


# ============================================================
# dk_search_agentic — classic execute_dk_search wrapper
# ============================================================

def _build_dk_keywords(context: Any, max_keywords: int) -> List[str]:
    """Build keyword strings for execute_dk_search from SharedContext.

    Priority: extra.final_keywords → selected_keywords → extracted_keywords.
    Formats dicts as "Term (GND-ID: id)" so execute_dk_search GND-validation
    recognises them.
    """
    keywords: List[str] = []
    seen: set = set()

    def _add(term: str, gnd_id: str = "") -> None:
        t = term.strip()
        if not t or t.lower() in seen:
            return
        seen.add(t.lower())
        keywords.append(f"{t} (GND-ID: {gnd_id})" if gnd_id else t)

    final_kws = (getattr(context, "extra", None) or {}).get("final_keywords") or []
    for kw in final_kws[:max_keywords]:
        if isinstance(kw, dict):
            _add(kw.get("keyword", "") or kw.get("title", ""), kw.get("gnd_id", ""))
        elif isinstance(kw, str):
            # Parse "Term (GND-ID: id)" format
            term, gid = _parse_keyword_string(kw)
            _add(term, gid)

    if not keywords:
        for kw in (getattr(context, "selected_keywords", None) or [])[:max_keywords]:
            if isinstance(kw, dict):
                _add(kw.get("title", "") or kw.get("keyword", ""), kw.get("gnd_id", ""))
            elif isinstance(kw, str):
                term, gid = _parse_keyword_string(kw)
                _add(term, gid)

    if not keywords:
        for kw in (getattr(context, "extracted_keywords", None) or [])[:max_keywords]:
            _add(str(kw) if not isinstance(kw, str) else kw)

    return keywords


@register_tool_fn("dk_search_agentic")
def dk_search_agentic(
    *,
    context: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    max_keywords: int = 30,
    config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Classic per-keyword catalog DK search for the agentic pipeline.

    Wraps ``PipelineStepExecutor.execute_dk_search`` so the agentic pipeline
    uses identical catalog search logic (per-keyword BiblioClient/MarcXmlClient,
    RVK validation, deduplication) as the classic rigid pipeline.

    Returns:
        ``{"dk_entries": [...], "dk_search_results": [...], "statistics": {...},
           "formatted_prompt": "...", "has_data": bool}``

    Stores keyword-centric ``dk_search_results`` on ``context`` for GUI display.
    """
    if context is None:
        raise RuntimeError("dk_search_agentic requires context")

    if config:
        max_keywords = config.get("max_keywords", max_keywords)

    keywords = _build_dk_keywords(context, max_keywords)
    if not keywords:
        if stream_callback:
            stream_callback("⚠️ dk_search_agentic: no keywords available — skipping\n")
        return {
            "dk_entries": [],
            "dk_search_results": [],
            "statistics": {},
            "formatted_prompt": "",
            "has_data": False,
        }

    if stream_callback:
        preview = ", ".join(keywords[:5])
        more = f" … +{len(keywords) - 5}" if len(keywords) > 5 else ""
        stream_callback(
            f"\n🔍 dk_search_agentic: {len(keywords)} keywords → catalog DK search\n"
            f"   {preview}{more}\n"
        )

    # execute_dk_search expects (msg, step_id) callback — adapt single-arg agentic callback
    def _dk_cb(msg: str, step_id: str = None) -> None:
        if stream_callback:
            stream_callback(msg)

    try:
        from src.utils.pipeline_utils import PipelineStepExecutor, PipelineResultFormatter
        from src.utils.config_manager import ConfigManager

        config_manager = ConfigManager()
        executor = PipelineStepExecutor(
            alima_manager=None,
            cache_manager=None,
            logger=logger,
            config_manager=config_manager,
        )
        dk_result = executor.execute_dk_search(
            keywords=keywords,
            stream_callback=_dk_cb,
            strict_gnd_validation=True,  # keywords formatted as "Term (GND-ID: id)" — validated
        )
    except Exception as exc:
        logger.error(f"dk_search_agentic: execute_dk_search failed: {exc}")
        if stream_callback:
            stream_callback(f"❌ DK-Katalogsuche fehlgeschlagen: {exc}\n")
        return {
            "dk_entries": [],
            "dk_search_results": [],
            "statistics": {},
            "formatted_prompt": "",
            "has_data": False,
        }

    classifications = dk_result.get("classifications", [])
    keyword_results = dk_result.get("keyword_results", [])
    statistics = dk_result.get("statistics", {})

    # Aggregate catalog stats for MetaAgent visibility
    total_titles = sum(len(c.get("titles", [])) for c in classifications)
    total_unique_notations = len(classifications)
    total_keywords_searched = statistics.get("total_keywords_searched", len(keyword_results))
    top_notations = [
        {
            "code": c.get("dk", ""),
            "type": c.get("type", c.get("classification_type", "DK")),
            "count": c.get("count", 0),
            "titles": c.get("titles", [])[:5],  # cap for serialization
            "title_count": len(c.get("titles", [])),
        }
        for c in classifications[:10]
    ]
    catalog_stats = {
        "total_titles": total_titles,
        "total_unique_notations": total_unique_notations,
        "total_keywords_searched": total_keywords_searched,
        "top_notations": top_notations,
        "deduplication": statistics.get("deduplication_stats", {}),
    }
    if hasattr(context, "dk_catalog_stats"):
        context.dk_catalog_stats = catalog_stats

    # Store keyword-centric results for GUI transparency
    if hasattr(context, "dk_search_results"):
        context.dk_search_results = keyword_results

    try:
        from src.utils.pipeline_utils import PipelineResultFormatter
        formatted_prompt = PipelineResultFormatter.format_dk_results_for_prompt(classifications)
    except Exception as exc:
        logger.warning(f"dk_search_agentic: format_dk_results_for_prompt failed: {exc}")
        formatted_prompt = ""

    if stream_callback:
        stream_callback(
            f"✅ dk_search_agentic: {len(classifications)} DK entries, "
            f"{len(keyword_results)} keyword-centric results\n"
            f"   📊 {total_titles} Titel, {total_unique_notations} Notationen\n"
        )

    return {
        "dk_entries": classifications,
        "dk_search_results": keyword_results,
        "statistics": statistics,
        "formatted_prompt": formatted_prompt,
        "has_data": bool(classifications),
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
            titles = e.get("titles", [])
            # Fallback: if no titles list but single title field exists
            if not titles and e.get("title"):
                titles = [e.get("title")]
            results.append({
                "keyword": e.get("keyword", ""),
                "dk": code,
                "title": e.get("title", titles[0] if titles else ""),
                "titles": titles,
                "title_count": len(titles),
                "count": e.get("count", 0),
                "classification_type": e.get("classification_type", "DK"),
            })

    for cls in dk_classifications:
        code = cls.get("code", "")
        if code:
            cls_title = cls.get("title", "")
            results.append({
                "keyword": "",
                "dk": code,
                "title": cls_title,
                "titles": [cls_title] if cls_title else [],
                "title_count": 1 if cls_title else 0,
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
    search_type: str = "kw",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Multi-source free-text catalog search (SWB + Lobid + catalog).

    Unlike ``gnd_batch_search`` (which targets keyword lookups for the
    pipeline), this fn also hits the catalog SOAP/SRU and returns a
    result-oriented structure suitable for end-user display.

    Args:
        search_type: ``"kw"`` (default, subject/keyword), ``"title"`` (title-only
            lookup across all backends), ``"freetext"`` (anyword).

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
        search_type = config.get("search_type", search_type)

    sources = sources or ["swb", "lobid", "catalog"]
    src_tools = {"swb": "search_swb", "lobid": "search_lobid", "catalog": "search_catalog"}
    if config:
        src_tools = config.get("source_tool_map", src_tools)

    queries = list(dict.fromkeys(q for q in (queries or []) if q))
    if not queries:
        return {"hits": [], "queries": [], "tool_calls": 0}

    if stream_callback:
        q_preview = ", ".join(queries[:8])
        q_more = f" … +{len(queries)-8}" if len(queries) > 8 else ""
        stream_callback(
            f"\n🔎 catalog_multi_search: {len(queries)} queries × "
            f"{len(sources)} sources (search_type={search_type})\n"
            f"   Suche: {q_preview}{q_more}\n"
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
            raw = tool_registry.execute(
                tool, {"terms": queries, "search_type": search_type}
            )
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
# catalog_title_search — Title-only bibliographic lookup
# ============================================================

@register_tool_fn("catalog_title_search")
def catalog_title_search(
    queries: List[str],
    *,
    tool_registry: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    search_type: str = "title",
    max_results: int = 25,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bibliographic-only catalog search (no GND/SWB/Lobid enrichment).

    Takes a list of queries (typically book titles extracted by an LLM)
    and returns raw catalog hits per query. Intended for the
    ``title_list_search`` workflow. GND enrichment, if desired, must be
    requested separately via ``gnd_batch_search`` in a follow-up step.

    Args:
        queries: List of query strings.
        search_type: Libero use-code or alias (see
            :meth:`BiblioClient.search_titles`). Default ``"title"``.
        max_results: Maximum records per query.

    Returns:
        ``{"hits": [...], "queries": [...], "tool_calls": N}``.
        Each hit: ``{query, rsn, title, authors, year, dk_codes,
        rvk_codes, ddc_codes, subjects, mab_subjects}``.
    """
    if tool_registry is None:
        raise RuntimeError("catalog_title_search requires tool_registry")

    if config:
        search_type = config.get("search_type", search_type)
        max_results = config.get("max_results", max_results)

    # Accept both ["title1","title2"] and [{"title":"...","authors":[...],"isbn":"..."}, ...].
    # For title-mode search, the "title" field drives the query.
    def _coerce(q: Any) -> Optional[str]:
        if isinstance(q, str):
            return q or None
        if isinstance(q, dict):
            for field in ("title", "term", "keyword", "label"):
                v = q.get(field)
                if isinstance(v, str) and v:
                    return v
        return None

    queries = list(dict.fromkeys(c for c in (_coerce(q) for q in (queries or [])) if c))
    if not queries:
        return {"hits": [], "queries": [], "tool_calls": 0}

    if stream_callback:
        q_preview = ", ".join(queries[:5])
        q_more = f" … +{len(queries)-5}" if len(queries) > 5 else ""
        stream_callback(
            f"\n🔎 catalog_title_search: {len(queries)} queries "
            f"(search_type={search_type}, max={max_results})\n"
            f"   Suche: {q_preview}{q_more}\n"
        )

    hits: List[Dict[str, Any]] = []
    tool_calls = 0

    try:
        raw = tool_registry.execute(
            "search_catalog_titles",
            {
                "terms": queries,
                "search_type": search_type,
                "max_results": max_results,
            },
        )
        tool_calls += 1
        data = json.loads(raw) if isinstance(raw, str) else raw
        if "error" in data:
            logger.warning(f"catalog_title_search: {data['error']}")
            return {"hits": [], "queries": queries, "tool_calls": tool_calls}

        per_query = data.get("results", {}) or {}
        for query, records in per_query.items():
            if stream_callback:
                stream_callback(f"  📚 '{query}': {len(records)} hits\n")
            for rec in records:
                hits.append({"query": query, **rec})
    except Exception as e:
        logger.warning(f"catalog_title_search: search_catalog_titles failed: {e}")

    if stream_callback:
        stream_callback(f"✅ {len(hits)} total records, {tool_calls} tool calls\n")

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
