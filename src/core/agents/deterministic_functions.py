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
from src.core.gnd_search_core import (
    merge_into_pool,
    parse_batch_response,
    parse_batch_response_with_terms,
    rank_pool,
)

logger = logging.getLogger(__name__)


# ============================================================
# gnd_batch_search — SWB + Lobid + local enrichment
# ============================================================

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
        return {"entries": [], "search_terms": [], "tool_calls": 0, "source_errors": {}}

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
    # Per-source hard failures (whole source unusable) - Claude Generated
    source_errors: Dict[str, str] = {}
    attempted_sources: List[str] = []
    # Which sources confirmed each title — drives source_count ranking - Claude Generated
    src_index: Dict[str, Set[str]] = {}

    for src in sources:
        tool_name = source_tools.get(src)
        if not tool_name:
            logger.warning(f"gnd_batch_search: unknown source '{src}'")
            continue
        attempted_sources.append(src)
        try:
            raw = tool_registry.execute(tool_name, {"terms": keywords})
            tool_calls += 1
            payload = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(payload, dict) and payload.get("error"):
                # Whole-source failure (tool error / suggester unavailable) - Claude Generated
                source_errors[src] = str(payload["error"])
                logger.warning(f"gnd_batch_search: {tool_name} failed: {payload['error']}")
                if stream_callback:
                    stream_callback(f"  ❌ {src}: Quelle fehlgeschlagen — {payload['error']}\n")
                continue
            # Per-term failures recorded by the suggester (partial outage) - Claude Generated
            term_errors = (payload.get("errors") or {}) if isinstance(payload, dict) else {}
            if term_errors and stream_callback:
                failed_terms = ", ".join(sorted(term_errors)[:5])
                more = f" … +{len(term_errors)-5}" if len(term_errors) > 5 else ""
                stream_callback(
                    f"  ⚠️ {src}: {len(term_errors)} Teilfehler ({failed_terms}{more}) — "
                    f"leere Treffer dafür sind NICHT bestätigt\n"
                )
            data, terms_map = parse_batch_response_with_terms(payload)
            for key in data:
                src_index.setdefault(key.lower(), set()).add(src)
            merge_into_pool(pool, data)
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
            source_errors[src] = str(e)
            logger.warning(f"gnd_batch_search: {tool_name} failed: {e}")
            if stream_callback:
                stream_callback(f"  ❌ {src}: Quelle fehlgeschlagen — {e}\n")

    # Fail loudly instead of continuing with a silently empty pool - Claude Generated
    if attempted_sources and len(source_errors) == len(attempted_sources):
        raise RuntimeError(
            f"gnd_batch_search: alle Quellen fehlgeschlagen: {source_errors}"
        )
    if not pool and source_errors:
        raise RuntimeError(
            f"gnd_batch_search: keine GND-Treffer und Quellfehler aufgetreten "
            f"(Ergebnis unvollständig): {source_errors}"
        )
    if not pool and stream_callback:
        stream_callback(
            f"⚠️ gnd_batch_search: 0 Treffer für {len(keywords)} Keywords "
            f"(keine Quellfehler — echte Nulltreffer)\n"
        )

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

    # Attach source provenance and rank (multi-source confirmations first, then
    # hit count) — the ordering the selection prompt relies on; see
    # gnd_search_core.rank_pool and its count-landmine note. - Claude Generated
    entries: List[Dict[str, Any]] = rank_pool(pool, src_index)

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
        "source_errors": source_errors,
    }


# ============================================================
# finc_subject_harvest — keyword-step finc title + subject harvest
# ============================================================

@register_tool_fn("finc_subject_harvest")
def finc_subject_harvest(
    keywords: List[str],
    *,
    tool_registry: Any = None,
    context: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Harvest finc catalog titles + reconcile their subjects into the GND pool.

    Opt-in via ``CatalogConfig.finc_harvest_enabled``. For each extracted
    keyword it runs a finc Subject search, stores the title records + DK/RVK
    facet distribution on ``context.extra['finc_harvest']`` (reusable by the DK
    step), and reconciles the records' free-text subjects against the LOCAL GND
    cache (``search_gnd``). Matched subjects become GND-validated pool entries
    merged into ``context.gnd_entries`` (same shape as ``gnd_batch_search``),
    giving the selection LLM more catalog-grounded candidates; titles already in
    the pool gain ``"finc"`` as a confirming source (boosting their rank). finc
    subjects carry no GND-IDs of their own, so subjects that don't reconcile are
    dropped from the pool (kept only as harvest provenance). - Claude Generated

    Returns ``{"entries":[...], "harvested_terms":[...], "subjects_reconciled":N,
    "merged_added":N, "tool_calls":N, "enabled":bool}``.
    """
    if tool_registry is None:
        raise RuntimeError("finc_subject_harvest requires tool_registry")

    max_records = 20
    max_subjects = 60
    if config:
        max_records = config.get("max_records", max_records)
        max_subjects = config.get("max_subjects", max_subjects)

    # Gate: opt-in. Explicit config 'enabled' wins; else read the catalog config.
    enabled = bool(config.get("enabled")) if config and "enabled" in config else None
    if enabled is None:
        try:
            from src.utils.config_manager import ConfigManager
            cat_cfg = ConfigManager().get_catalog_config()
            enabled = bool(getattr(cat_cfg, "finc_harvest_enabled", False))
        except Exception as e:
            logger.debug(f"finc_subject_harvest: config read failed: {e}")
            enabled = False
    if not enabled:
        return {"entries": [], "harvested_terms": [], "subjects_reconciled": 0,
                "merged_added": 0, "tool_calls": 0, "enabled": False}

    keywords = list(dict.fromkeys(
        k for k in (keywords or []) if isinstance(k, str) and k.strip()
    ))
    if not keywords:
        return {"entries": [], "harvested_terms": [], "subjects_reconciled": 0,
                "merged_added": 0, "tool_calls": 0, "enabled": True}

    if stream_callback:
        stream_callback(
            f"\n📚 finc_subject_harvest: {len(keywords)} Keywords → finc Titel + "
            f"Schlagwort-Abgleich gegen GND-Cache\n"
        )

    tool_calls = 0
    subject_freq: Dict[str, int] = {}
    harvest_store: Dict[str, Any] = {}

    for kw in keywords:
        try:
            raw = tool_registry.execute("search_finc", {
                "terms": [kw], "search_type": "subject",
                "limit": max_records, "facets": ["udk_raw_de105", "rvk_facet"],
            })
            tool_calls += 1
        except Exception as e:
            logger.warning(f"finc_subject_harvest: search_finc failed for '{kw}': {e}")
            continue
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            # finc unconfigured / hard error — no point looping further. - Claude Generated
            logger.info(f"finc_subject_harvest: {data['error']}")
            if stream_callback:
                stream_callback(f"  ⚠️ finc nicht verfügbar: {data['error']}\n")
            break
        entry = (data.get("results") or {}).get(kw) or {}
        records = entry.get("records", []) or []
        harvest_store[kw] = {
            "records": [{"id": r.get("id"), "title": r.get("title")}
                        for r in records if r.get("id")],
            "dk_dist": (entry.get("facets", {}) or {}).get("udk_raw_de105", []),
            "rvk_dist": (entry.get("facets", {}) or {}).get("rvk_facet", []),
        }
        for r in records:
            for grp in r.get("subjects", []) or []:
                for subj in (grp if isinstance(grp, list) else [grp]):
                    s = (subj or "").strip() if isinstance(subj, str) else ""
                    if s:
                        subject_freq[s] = subject_freq.get(s, 0) + 1
        if stream_callback:
            stream_callback(f"  🌐 '{kw}': {len(records)} Titel\n")

    # Store harvest on context for the DK step / GUI transparency
    if context is not None and isinstance(getattr(context, "extra", None), dict):
        context.extra["finc_harvest"] = harvest_store

    # Reconcile the most frequent subjects against the local GND cache
    top_subjects = sorted(subject_freq.items(), key=lambda kv: kv[1], reverse=True)[:max_subjects]
    new_entries: List[Dict[str, Any]] = []
    reconciled = 0
    for subj, freq in top_subjects:
        try:
            raw = tool_registry.execute("search_gnd", {"term": subj, "min_results": 1})
            tool_calls += 1
        except Exception as e:
            logger.debug(f"finc_subject_harvest: search_gnd failed for '{subj}': {e}")
            continue
        gdata = json.loads(raw) if isinstance(raw, str) else raw
        gentries = (gdata or {}).get("entries") or []
        if not gentries:
            continue
        best = gentries[0]
        gid = best.get("gnd_id", "")
        title = best.get("title", "")
        if not title:
            continue
        reconciled += 1
        new_entries.append({
            "title": title,
            "gnd_ids": [gid] if gid else [],
            "gnd_id": gid,
            "ddc_codes": list(best.get("ddcs", []) or []),
            "dk_codes": [],
            "count": freq,
            "description": best.get("description", "") or "",
            "synonyms": list(best.get("synonyms", []) or []),
            "sources": ["finc"],
            "source_count": 1,
        })

    # Merge into context.gnd_entries: confirm existing titles (add 'finc' source)
    # or append new finc-reconciled entries. - Claude Generated
    merged_added = 0
    if context is not None and hasattr(context, "gnd_entries"):
        index = {(e.get("title") or "").lower(): e for e in context.gnd_entries}
        for ne in new_entries:
            key = ne["title"].lower()
            existing = index.get(key)
            if existing is not None:
                srcs = set(existing.get("sources", []) or [])
                if "finc" not in srcs:
                    srcs.add("finc")
                    existing["sources"] = sorted(srcs)
                    existing["source_count"] = len(srcs)
            else:
                context.gnd_entries.append(ne)
                index[key] = ne
                merged_added += 1

    if stream_callback:
        stream_callback(
            f"✅ finc_subject_harvest: {reconciled} Schlagworte gegen GND-Cache "
            f"abgeglichen, {merged_added} neue Pool-Einträge, {tool_calls} tool calls\n"
        )

    return {
        "entries": new_entries,
        "harvested_terms": keywords,
        "subjects_reconciled": reconciled,
        "merged_added": merged_added,
        "tool_calls": tool_calls,
        "enabled": True,
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
# verify_final_keywords — classic GND-pool verification wrapper
# ============================================================

@register_tool_fn("verify_final_keywords")
def verify_final_keywords(
    *,
    context: Any = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Verify selection-LLM keywords against the GND search pool - Claude Generated

    Classic-pipeline parity: the rigid pipeline verifies LLM-selected keywords
    against the pool from the search step (GND-ID match → text match → DB
    fallback via ``search_gnd_by_title``) and re-attaches authoritative
    GND-IDs. The agentic pipeline previously trusted the LLM output verbatim,
    so dropped/hallucinated GND-IDs silently shrank the strict-validated DK
    search. Reuses ``verify_keywords_against_gnd_pool`` from pipeline_utils.

    Reads ``extra.final_keywords`` (fallback: ``selected_keywords``) and
    ``context.gnd_entries``; writes the verified list back to
    ``extra.final_keywords``. Unverifiable keywords are logged, not silent.

    Returns:
        ``{"verified_keywords": [{"keyword","gnd_id"}], "rejected": [...],
           "stats": {...}}``
    """
    if context is None:
        raise RuntimeError("verify_final_keywords requires context")

    def _to_string(kw: Any) -> Optional[str]:
        if isinstance(kw, dict):
            term = (kw.get("keyword") or kw.get("title") or "").strip()
            gid = (kw.get("gnd_id") or "").strip()
            if not term:
                return None
            return f"{term} (GND-ID: {gid})" if gid else term
        if isinstance(kw, str) and kw.strip():
            return kw.strip()
        return None

    extra = getattr(context, "extra", None) or {}
    raw_keywords = extra.get("final_keywords") or getattr(context, "selected_keywords", None) or []
    extracted = [s for s in (_to_string(kw) for kw in raw_keywords) if s]

    if not extracted:
        if stream_callback:
            stream_callback("⚠️ verify_final_keywords: keine Keywords zu verifizieren\n")
        return {"verified_keywords": [], "rejected": [], "stats": {"total_extracted": 0}}

    # Build pool strings "Title (GND-ID: id)" from the search step entries
    pool: List[str] = []
    for entry in getattr(context, "gnd_entries", None) or []:
        if not isinstance(entry, dict):
            continue
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        gnd_ids = entry.get("gnd_ids") or ([entry.get("gnd_id")] if entry.get("gnd_id") else [])
        for gid in gnd_ids:
            if gid:
                pool.append(f"{title} (GND-ID: {gid})")

    # Classic callback signature is (msg, step_id) — adapt the agentic one
    def _cb(msg: str, step_id: str = None) -> None:
        if stream_callback:
            stream_callback(msg)

    knowledge_manager = None
    try:
        from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
        knowledge_manager = UnifiedKnowledgeManager()
    except Exception as exc:
        logger.warning(f"verify_final_keywords: UnifiedKnowledgeManager unavailable: {exc}")

    from src.utils.pipeline_utils import verify_keywords_against_gnd_pool
    result = verify_keywords_against_gnd_pool(
        extracted_keywords=extracted,
        gnd_pool_keywords=pool,
        stream_callback=_cb,
        step_id="verify_keywords",
        knowledge_manager=knowledge_manager,
    )

    # Back to the dict shape the selection step produces ({keyword, gnd_id})
    verified_keywords: List[Dict[str, str]] = []
    seen: set = set()
    for kw in result.get("verified", []):
        term, gid = _parse_keyword_string(kw)
        key = (gid or term).lower()
        if not term or key in seen:
            continue
        seen.add(key)
        verified_keywords.append({"keyword": term, "gnd_id": gid})

    if hasattr(context, "extra"):
        context.extra["final_keywords"] = verified_keywords

    return {
        "verified_keywords": verified_keywords,
        "rejected": result.get("rejected", []),
        "stats": result.get("stats", {}),
    }


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
    dk_frequency_threshold: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Classic per-keyword catalog DK search for the agentic pipeline.

    Wraps ``PipelineStepExecutor.execute_dk_search`` so the agentic pipeline
    uses identical catalog search logic (per-keyword BiblioClient/MarcXmlClient,
    RVK validation, deduplication) as the classic rigid pipeline. The
    classification prompt text is built via the shared
    ``prepare_dk_classification_context`` (frequency filter, title filter,
    RVK guardrail) and RVK anchors are derived like in the classic pipeline
    (heuristic path — no LLM available here) - Claude Generated

    Returns:
        ``{"dk_entries": [...], "dk_search_results": [...], "statistics": {...},
           "formatted_prompt": "...", "has_data": bool}``

    Stores keyword-centric ``dk_search_results`` on ``context`` for GUI display.
    """
    if context is None:
        raise RuntimeError("dk_search_agentic requires context")

    rvk_inline = True
    if config:
        max_keywords = config.get("max_keywords", max_keywords)
        dk_frequency_threshold = config.get("dk_frequency_threshold", dk_frequency_threshold)
        # rvk_inline=False ⇒ no inline RVK anchor/API work here; RVK is surfaced
        # on demand by the classification LLM via the rvk_lookup tool. - Claude Generated
        rvk_inline = config.get("rvk_inline", True)

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
        from src.utils.pipeline_utils import PipelineStepExecutor
        from src.utils.config_manager import ConfigManager

        config_manager = ConfigManager()
        executor = PipelineStepExecutor(
            alima_manager=None,
            cache_manager=None,
            logger=logger,
            config_manager=config_manager,
        )
        # Classic-parity: derive RVK anchors from the same keywords.
        # alima_manager is None → heuristic fallback inside. Skipped entirely
        # when rvk_inline is False (RVK handled via rvk_lookup tool). - Claude Generated
        rvk_anchor_keywords = None
        if rvk_inline:
            try:
                rvk_anchor_keywords = executor._derive_rvk_anchor_keywords(
                    keywords,
                    original_abstract=getattr(context, "abstract", "") or "",
                    stream_callback=_dk_cb,
                )
            except Exception as exc:
                logger.warning(f"dk_search_agentic: RVK anchor derivation failed: {exc}")
                rvk_anchor_keywords = None
        dk_result = executor.execute_dk_search(
            keywords=keywords,
            rvk_anchor_keywords=rvk_anchor_keywords,
            rvk_enabled=rvk_inline,
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

    # Build the classification prompt via the SHARED classic preparation:
    # frequency threshold (DK only), title filter, institution-library RVK
    # filter, RVK guardrail — identical context in both pipeline modes - Claude Generated
    try:
        from src.utils.pipeline_defaults import DEFAULT_DK_FREQUENCY_THRESHOLD
        threshold = (
            dk_frequency_threshold
            if dk_frequency_threshold is not None
            else DEFAULT_DK_FREQUENCY_THRESHOLD
        )
        prep = executor.prepare_dk_classification_context(
            classifications,
            original_abstract=getattr(context, "abstract", "") or "",
            dk_frequency_threshold=threshold,
            rvk_anchor_keywords=rvk_anchor_keywords,
            stream_callback=_dk_cb,
            include_rvk=rvk_inline,
        )
        formatted_prompt = prep["catalog_text"] if prep["results_with_titles"] else ""
        if hasattr(context, "extra") and rvk_inline:
            # RVK candidate maps for potential downstream post-processing
            context.extra["rvk_allowed_standard"] = prep["allowed_standard_rvk_map"]
            context.extra["rvk_allowed_nonstandard"] = prep["allowed_nonstandard_rvk_map"]
    except Exception as exc:
        logger.warning(f"dk_search_agentic: prepare_dk_classification_context failed: {exc}")
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
        # has_data gates the classification step (condition in YAML):
        # only True if the prompt actually carries catalog context - Claude Generated
        "has_data": bool(formatted_prompt),
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

    finc integration (resolved June 2026): finc is NOT folded into this fn.
    Because finc returns full VuFind records (not the aggregated keyword shape
    this fn produces), it is integrated where its strengths fit instead — opt-in
    and gated by ``CatalogConfig``:
      - keyword step: ``finc_subject_harvest`` (finc_harvest_enabled) reconciles
        finc record subjects against the local GND cache into the pool;
      - DK step: ``PipelineStepExecutor.execute_dk_search`` (finc_dk_enabled)
        reads per-title ``udk_raw_de105``/``rvk_facet`` via ``FincCatalogClient``.
    ``catalog_multi_search`` stays swb/lobid/catalog. - Claude Generated

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
            data = parse_batch_response(raw)
            for key, entry in data.items():
                k = key.lower()
                src_index.setdefault(k, set()).add(src)
            merge_into_pool(pool, data)
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
