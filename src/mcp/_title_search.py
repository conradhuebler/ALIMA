"""Agent tool ``search_titles``: one title search over the title sources - Claude Generated.

Generated when at least one active search instance declares ``TITLE_RECORDS``
(finc, catalog, sru, kvk, or a plugin). Per term the sources are asked in the
order of the plugin list; the first one with at least one record answers.

Per source and term the status is one of

* ``ok``: records came back, this source answers the term;
* ``empty``: the source answered with no record, the next one is asked;
* ``down``: skipped by ``source_health`` (no request);
* ``cache_miss``: state "Nur Cache" without an entry (no request);
* ``unsupported``: the source does not know the search axis, or is not
  available (no endpoint/token), with ``reason``;
* ``error``: the request failed, with ``error``.

The sources are the factory-built providers of the instances
(``ToolRegistry._provider_for``), searched through ``search(TITLE_RECORDS)``,
the same path as the single title tools take for a provider without
suggester and for the state "Nur Cache". The single tools stay as they are.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.mcp.mcp_types import ToolDefinition

logger = logging.getLogger(__name__)

TOOL_NAME = "search_titles"
#: The axes of the tool; each source maps them (``title_search_axes``).
AXES = ["kw", "title", "author", "subject", "isbn"]
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS_LIMIT = 50


def title_instances(registry: Any) -> List[Any]:
    """Active search instances whose type declares TITLE_RECORDS, in list order
    - Claude Generated"""
    from src.core.search.provider import SearchCapability
    from src.core.search.registry import PROVIDER_REGISTRY

    out = []
    for inst in registry._search_instances():
        cls = PROVIDER_REGISTRY.get(getattr(inst, "provider_id", ""))
        if cls is None:
            continue
        if SearchCapability.TITLE_RECORDS in getattr(cls, "capabilities", set()):
            out.append(inst)
    return out


def _label(inst: Any) -> str:
    from src.core.search.registry import PROVIDER_REGISTRY

    cls = PROVIDER_REGISTRY.get(getattr(inst, "provider_id", ""))
    return getattr(inst, "label", "") or getattr(cls, "label", "") or inst.instance_id


def _axis_for(inst: Any, axis: str) -> Optional[str]:
    """The source's own search type for ``axis``; ``None`` when it has none.
    A type without ``title_search_axes`` gets the axis unchanged. - Claude Generated"""
    from src.core.search.registry import PROVIDER_REGISTRY

    cls = PROVIDER_REGISTRY.get(getattr(inst, "provider_id", ""))
    axes = getattr(cls, "title_search_axes", None)
    if axes is None:
        return axis
    return axes.get(axis)


def build_title_search_tool(registry: Any) -> Optional[Tuple[ToolDefinition, Any]]:
    """``(ToolDefinition, handler)`` of ``search_titles``, ``None`` without a
    title source - Claude Generated"""
    instances = title_instances(registry)
    if not instances:
        return None
    order = ", ".join(f"{_label(inst)} [{inst.instance_id}]" for inst in instances)
    description = (
        "Search titles across all active title sources at once. Per term the "
        f"sources are asked in the order of the plugin list: {order}. The first "
        "source with at least one record answers the term; a source with no record "
        "for it, one that is skipped after connection failures, one in cache-only "
        "mode without a cache entry, one that is not configured, and one without "
        "the search axis are passed over, and the next source is asked. Use it to "
        "find a work by title, author, subject or ISBN when it does not matter which "
        "catalogue answers. Use the single tools (search_finc, "
        "search_catalog_titles, search_k10plus, search_sru, search_kvk) for options "
        "of one source (facets, filters, availability, local data) or to compare "
        "sources. Per term: `records` (as the answering source delivers them, each "
        "with `source` = the instance id), `answered_by` (instance id, null when no "
        "source had a record), `tried` (each source asked, in order, with `status` "
        "ok / empty / down / cache_miss / unsupported / error and `reason` or "
        "`error`), and `total` / `truncated` / `notice` when the answering source "
        "reports them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more search terms, each searched independently.",
            },
            "search_type": {
                "type": "string",
                "enum": list(AXES),
                "default": "kw",
                "description": "kw = all words; title; author; subject; isbn.",
            },
            "max_results": {
                "type": "integer",
                "default": DEFAULT_MAX_RESULTS,
                "minimum": 1,
                "maximum": MAX_RESULTS_LIMIT,
                "description": "Records per term from the answering source (default 10, at most 50).",
            },
        },
        "required": ["terms"],
    }

    def handler(terms, search_type="kw", max_results=None, **_ignore):
        return json.dumps(
            search_titles(registry, list(terms or []), search_type=search_type,
                          max_results=max_results, instances=instances),
            ensure_ascii=False, default=str,
        )

    return ToolDefinition(name=TOOL_NAME, description=description, parameters=parameters), handler


def _clamp(max_results: Any) -> int:
    try:
        value = int(max_results) if max_results is not None else DEFAULT_MAX_RESULTS
    except (TypeError, ValueError):
        value = DEFAULT_MAX_RESULTS
    return max(1, min(MAX_RESULTS_LIMIT, value))


def search_titles(registry: Any, terms: List[str], *, search_type: str = "kw",
                  max_results: Any = None, instances: Optional[List[Any]] = None) -> Dict[str, Any]:
    """The result of ``search_titles`` as a dict - Claude Generated"""
    axis = str(search_type or "kw").strip().lower()
    if axis not in AXES:
        return {"error": f"unknown search_type {search_type!r} (known: {', '.join(AXES)})"}
    limit = _clamp(max_results)
    if instances is None:
        instances = title_instances(registry)
    results: Dict[str, Any] = {}
    for term in terms:
        results[term] = _search_term(registry, instances, term, axis, limit)
    return {"search_type": axis, "sources": [i.instance_id for i in instances],
            "results": results}


def _search_term(registry: Any, instances: List[Any], term: str, axis: str,
                 limit: int) -> Dict[str, Any]:
    tried: List[Dict[str, Any]] = []
    for inst in instances:
        entry, answer = _ask(registry, inst, term, axis, limit)
        tried.append(entry)
        if answer is not None:
            out = {"records": answer["records"], "answered_by": inst.instance_id, "tried": tried}
            for key in ("total", "truncated", "notice"):
                if answer.get(key) is not None:
                    out[key] = answer[key]
            return out
    return {"records": [], "answered_by": None, "tried": tried}


def _ask(registry: Any, inst: Any, term: str, axis: str,
         limit: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``(tried entry, answer)`` of one source for one term; ``answer`` is set
    only for ``ok`` - Claude Generated"""
    from src.core.search import source_usage
    from src.core.search.provider import SearchCapability

    sid = inst.instance_id
    own_axis = _axis_for(inst, axis)
    if own_axis is None:
        return {"source": sid, "status": "unsupported",
                "reason": f"Suchart {axis} nicht unterstützt"}, None
    try:
        provider = registry._provider_for(inst, cache=False)
        if hasattr(provider, "is_available") and not provider.is_available():
            return {"source": sid, "status": "unsupported", "reason": "nicht verfügbar"}, None
    except Exception as exc:  # noqa: BLE001 - one source must not break the search
        from src.utils.error_visibility import log_caught

        log_caught(logger, exc, "search_titles provider", detail=sid)
        return {"source": sid, "status": "error", "error": str(exc)}, None
    cache_only = getattr(provider, "cache_only", False) is True
    try:
        res = provider.search(SearchCapability.TITLE_RECORDS, [term], search_type=own_axis,
                              max_results=limit, limit=limit)
    except Exception as exc:  # noqa: BLE001 - one source must not break the search
        from src.utils.error_visibility import log_caught

        log_caught(logger, exc, "search_titles search", detail=f"{sid}: {term}")
        source_usage.record(sid, source_usage.CACHE if cache_only else source_usage.NETWORK,
                            kind=source_usage.KIND_RECORDS, error=True)
        return {"source": sid, "status": "error", "error": str(exc)}, None
    message = (res.errors or {}).get(term)
    records = [it.record for it in res.per_term.get(term, []) if it.record is not None]
    source_usage.record_terms(sid, [term], errors=res.errors, records={term: len(records)},
                              cache_only=cache_only, kind=source_usage.KIND_RECORDS)
    if message:
        origin = source_usage.origin_of_error(message)
        if origin == source_usage.SKIPPED:
            return {"source": sid, "status": "down", "reason": str(message)}, None
        if origin == source_usage.CACHE_MISS:
            return {"source": sid, "status": "cache_miss", "reason": str(message)}, None
        return {"source": sid, "status": "error", "error": str(message)}, None
    if not records:
        return {"source": sid, "status": "empty"}, None
    records = records[:limit]
    if hasattr(provider, "catalog_web_record_base"):
        base = registry._finc_record_base(provider)
        if base:
            records = [dict(r, web_url=f"{base}/{r['id']}")
                       if not r.get("web_url") and r.get("id") else r for r in records]
    records = [{**r, "source": sid} for r in records]
    meta = (res.per_term_meta or {}).get(term) or {}
    answer: Dict[str, Any] = {"records": records, "total": meta.get("total"),
                              "truncated": True if meta.get("truncated") else None}
    notice_fn = getattr(provider, "tool_notice", None)
    if callable(notice_fn):
        try:
            answer["notice"] = notice_fn(res, tools=registry._known_tool_names()) or None
        except Exception as exc:  # noqa: BLE001 - a notice must not break the result
            from src.utils.error_visibility import log_caught

            log_caught(logger, exc, "search_titles notice", detail=sid)
    entry = {"source": sid, "status": "ok", "records": len(records)}
    return entry, answer
