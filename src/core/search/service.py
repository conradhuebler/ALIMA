"""Unified GND-keyword search service - Claude Generated.

The single provider-path entry point for a GND-keyword search. It replaces the
three hand-wired construction sites that each built providers their own way:

* classic ``SearchCLI`` (a fresh ``MetaSuggester`` per source, per call),
* the MCP ``ToolRegistry`` primaries (``MetaSuggester`` + direct suggesters),
* the GUI ``find_keywords`` standalone/manual search (direct ``MetaSuggester``).

All three now build providers through :func:`search.factory.build_provider` from
the authoritative :class:`PluginInstanceConfig` list, run the capability-based
``search(GND_KEYWORDS, ...)``, and merge into the legacy
``{term: {keyword: {count, gnd_ids, classifications, display_count?}}}`` shape via the single
:func:`gnd_search_core.merge_code_entry` atom.

Two read modes, same signature (parity with the retired ``MetaSuggester`` /
``SearchCLI`` pair):

* **live/merge** (``aggregate_from_raw=False``) — search each provider (mapping-first
  cached), cross-merge the reduced views.
* **raw-first** (``aggregate_from_raw=True``, the default) — run the search for its
  side effect (the fetch seam writes the verbatim response to the WP2
  ``search_response_cache``), then derive the reduced view from raw via
  :func:`aggregate.aggregate_gnd_results` (raw-first, mapping fallback). Byte-compatible
  with ``SearchCLI.search_from_raw``.

This module is Qt-free (``progress`` replaces the ``currentTerm`` signal).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..gnd_search_core import merge_code_entry
from .aggregate import (
    aggregate_gnd_results,
    default_aggregate_from_raw,
    nested_from_aggregate,
)
from .factory import build_provider
from .provider import SearchCapability, raw_cache_params_for
from .registry import get_provider

logger = logging.getLogger(__name__)

NestedResults = Dict[str, Dict[str, Dict[str, Any]]]

# Zero-config web defaults (oGND = lobid) used only when the config is unreadable —
# never silently search nothing. Configured blueprints (catalog/finc) are added by
# the operator's plugin instances, not baked in here. - Claude Generated
_DEFAULT_GND_PROVIDER_IDS: List[str] = ["lobid", "swb"]

CATEGORY = "search_provider"


# --------------------------------------------------------------------------- #
# Instance resolution
# --------------------------------------------------------------------------- #
def _is_gnd_capable(provider_id: str) -> bool:
    try:
        return SearchCapability.GND_KEYWORDS in getattr(
            get_provider(provider_id), "capabilities", set()
        )
    except KeyError:
        return False


def _expand_ids(ids: Union[str, Sequence[str]]) -> List[str]:
    """Normalise an id / id-list to a de-duplicated lower-cased list (drops ``all``,
    which the caller resolves to the full enabled set). - Claude Generated"""
    if isinstance(ids, str):
        ids = [ids]
    out: List[str] = []
    for pid in ids:
        p = str(pid).lower()
        if p == "all":
            continue
        if p not in out:
            out.append(p)
    return out


def resolve_gnd_instances(
    ids: Optional[Union[str, Sequence[str]]] = None,
    *,
    config: Any = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Any]:
    """Resolve the :class:`PluginInstanceConfig`s to search.

    ``ids=None`` or an ``"all"`` entry → every enabled GND-keyword instance in the
    config (the authoritative enable/disable gate). A specific id list → one primary
    instance per id, synthesising a default instance only for ids the config knows
    nothing about (no-config fallback); an id that exists but is *disabled* is
    respected (skipped), so this is the single gate.

    ``overrides`` (``{provider_id: {setting_key: value}}``) overlays non-empty values
    onto the resolved instance settings — the classic path uses it to inject an
    explicit catalog token/URL without going through the ``CatalogConfig`` mirror.
    - Claude Generated
    """
    from src.utils.config_models import PluginInstanceConfig

    overrides = overrides or {}

    if config is None:
        try:
            from src.utils.config_manager import ConfigManager

            config = ConfigManager().load_config()
        except Exception:
            config = None

    def _overlay(inst: Any) -> Any:
        ov = overrides.get(inst.provider_id)
        if not ov:
            return inst
        merged = dict(inst.settings or {})
        for key, value in ov.items():
            if value not in (None, ""):
                merged[key] = value
        return replace(inst, settings=merged)

    want_all = (
        ids is None
        or (isinstance(ids, str) and ids.lower() == "all")
        or (isinstance(ids, (list, tuple)) and any(str(i).lower() == "all" for i in ids))
    )

    if want_all:
        if config is not None:
            try:
                out = [
                    _overlay(inst)
                    for inst in config.enabled_instances_for(CATEGORY)
                    if _is_gnd_capable(inst.provider_id)
                ]
                if out:
                    return out
            except Exception:
                logger.debug("resolve_gnd_instances: enabled_instances_for failed", exc_info=True)
        ids = list(_DEFAULT_GND_PROVIDER_IDS)

    resolved: List[Any] = []
    for pid in _expand_ids(ids):
        inst = None
        if config is not None:
            try:
                inst = config.primary_instance(CATEGORY, pid)
            except Exception:
                inst = None
        if inst is not None:
            resolved.append(_overlay(inst))
            continue
        if config is not None and _config_has_provider(config, pid):
            # Present but disabled → respect the operator's disable.
            logger.debug("resolve_gnd_instances: provider '%s' present but disabled", pid)
            continue
        settings = {k: v for k, v in (overrides.get(pid) or {}).items() if v not in (None, "")}
        resolved.append(
            PluginInstanceConfig(
                instance_id=pid,
                category=CATEGORY,
                provider_id=pid,
                enabled=True,
                is_primary=True,
                settings=settings,
            )
        )
    return resolved


def _config_has_provider(config: Any, provider_id: str) -> bool:
    try:
        return any(p.provider_id == provider_id for p in config.instances_for(CATEGORY))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Provider construction + suggester access
# --------------------------------------------------------------------------- #
def build_gnd_providers(
    instances: Sequence[Any],
    *,
    cache: bool = True,
    ukm: Any = None,
    max_age_hours: int = 24,
    cache_raw: "bool | None" = None,
) -> Dict[str, Any]:
    """Build ``{provider_id: provider}`` for the given instances via the factory.

    Keyed by ``provider.id`` (the source name) so error keys and raw-cache source
    ids match the legacy ``MetaSuggester`` contract. A later instance of the same
    type overwrites an earlier one (the classic path passes one per type). - Claude Generated
    """
    providers: Dict[str, Any] = {}
    for inst in instances:
        try:
            provider = build_provider(
                inst, cache=cache, ukm=ukm, max_age_hours=max_age_hours, cache_raw=cache_raw
            )
        except KeyError:
            logger.warning(
                "Unknown GND provider type '%s' (instance '%s'), skipping",
                getattr(inst, "provider_id", "?"),
                getattr(inst, "instance_id", "?"),
            )
            continue
        providers[getattr(provider, "id", getattr(inst, "provider_id", "?"))] = provider
    return providers


def underlying_suggester(provider: Any) -> Any:
    """Return the wrapped ``BaseSuggester`` for a built provider (unwrapping a
    ``CachingProvider``), or ``None``. Builds the suggester lazily (no network call).

    Used for the raw-first ``transform`` and the MCP non-default passthrough that
    the ``ProviderResult`` surface does not expose. - Claude Generated"""
    inner = getattr(provider, "inner", provider)  # unwrap CachingProvider
    return getattr(inner, "suggester", None)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _merge_term(target: Dict[str, Dict[str, Any]], kw_map: Dict[str, Dict[str, Any]]) -> None:
    """Merge one provider's per-term keyword map into ``target`` (in place).

    Fresh entries copy count + fresh code sets (+ display_count when present);
    collisions go through the shared ``merge_code_entry`` atom (max count, union
    codes, max display_count). - Claude Generated"""
    for kw, data in kw_map.items():
        if kw not in target:
            entry: Dict[str, Any] = {
                "count": data.get("count", 1),
                "gnd_ids": set(data.get("gnd_ids", set()) or set()),
                "classifications": {
                    system: set(codes)
                    for system, codes in (data.get("classifications") or {}).items()
                    if codes
                },
            }
            if data.get("display_count") is not None:
                entry["display_count"] = data["display_count"]
            target[kw] = entry
        else:
            merge_code_entry(
                target[kw], data, code_fields=("gnd_ids",),
                display_count_field="display_count",
                classifications_field="classifications",
            )


def _build_one(inst: Any, *, cache: bool, ukm: Any, max_age_hours: int) -> Tuple[str, Any]:
    """Build one provider, returning ``(source_id, provider)``. Raises on failure so
    the caller records a per-term source error (parity with the old per-type
    ``try/except`` in ``SearchCLI``). - Claude Generated"""
    provider = build_provider(inst, cache=cache, ukm=ukm, max_age_hours=max_age_hours)
    return getattr(provider, "id", getattr(inst, "provider_id", "?")), provider


def search_gnd_keywords(
    terms: Sequence[str],
    instances: Sequence[Any],
    *,
    cache: bool = True,
    aggregate_from_raw: Optional[bool] = None,
    ukm: Any = None,
    max_age_hours: int = 24,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[NestedResults, Dict[str, str]]:
    """Search the given provider instances for GND keywords.

    Returns ``(nested_results, errors)`` where ``errors`` maps ``"<source>:<term>"``
    → a source-failure message (an empty result paired with an error entry means
    "source failed", NOT "no match" — mirrors ``BaseSuggester.last_errors``).
    - Claude Generated
    """
    terms = list(terms)
    if aggregate_from_raw is None:
        aggregate_from_raw = default_aggregate_from_raw()
    if ukm is None:
        from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

        ukm = UnifiedKnowledgeManager()

    if aggregate_from_raw:
        return _search_from_raw(
            terms, instances, cache=cache, ukm=ukm, max_age_hours=max_age_hours, progress=progress
        )

    combined: NestedResults = {t: {} for t in terms}
    errors: Dict[str, str] = {}
    for inst in instances:
        source = getattr(inst, "provider_id", "?")
        try:
            source, provider = _build_one(inst, cache=cache, ukm=ukm, max_age_hours=max_age_hours)
            result = provider.search(SearchCapability.GND_KEYWORDS, terms, progress=progress)
        except Exception as exc:
            logger.error("Error searching with %s provider: %s", source, exc)
            for term in terms:
                errors[f"{source}:{term}"] = str(exc)
            continue
        per_term = result.to_gnd_keywords()
        for term in terms:
            _merge_term(combined[term], per_term.get(term, {}))
        for term, message in (result.errors or {}).items():
            errors.setdefault(f"{source}:{term}", message)
    return combined, errors


def _search_from_raw(
    terms: List[str],
    instances: Sequence[Any],
    *,
    cache: bool,
    ukm: Any,
    max_age_hours: int,
    progress: Optional[Callable[[str], None]],
) -> Tuple[NestedResults, Dict[str, str]]:
    """Live-fetch (populating the raw cache) then derive the nested view from raw.

    Same convergence as ``SearchCLI.search_from_raw``: the per-provider search still
    runs (it performs the fetch that writes raw through the provider seam and
    surfaces source failures), but the reduced view is rebuilt from
    ``search_response_cache`` (single source of truth) via the shared aggregate
    engine (raw-first, mapping fallback). - Claude Generated"""
    errors: Dict[str, str] = {}
    transform_by_source: Dict[str, Any] = {}
    params_by_source: Dict[str, Dict[str, Any]] = {}
    ok_sources: List[str] = []

    for inst in instances:
        source = getattr(inst, "provider_id", "?")
        try:
            source, provider = _build_one(inst, cache=cache, ukm=ukm, max_age_hours=max_age_hours)
            result = provider.search(SearchCapability.GND_KEYWORDS, terms, progress=progress)
        except Exception as exc:
            logger.error("Error searching with %s provider: %s", source, exc)
            for term in terms:
                errors[f"{source}:{term}"] = str(exc)
            continue
        for term, message in (result.errors or {}).items():
            errors.setdefault(f"{source}:{term}", message)
        transform = getattr(underlying_suggester(provider), "transform", None)
        if transform is not None:
            transform_by_source[source] = transform
            params_by_source[source] = raw_cache_params_for(source)
            ok_sources.append(source)

    agg = aggregate_gnd_results(
        terms, ok_sources, ukm, transform_by_source,
        params_by_source=params_by_source, max_age_hours=max_age_hours,
    )
    return nested_from_aggregate(agg), errors
