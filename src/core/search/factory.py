"""Config → provider factory + the search plugin-category adapter - Claude Generated.

The *single* place that turns a configured instance into a runtime provider. It
replaces the three hand-wired config→provider sites (``MetaSuggester.__init__``,
``ToolRegistry._init_suggesters``, ``pipeline_utils.execute_dk_search`` — Debt
D-1/D-2/D-4): each now asks the factory for providers built from
:class:`PluginInstanceConfig`s.

Also defines :class:`SearchProviderCategory`, the adapter that lets the generic
plugin framework (settings UI, directory loader) treat search providers as one
plugin category. It self-registers on import (this module is imported by
``src/core/search/__init__.py``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from src.core.plugins.category import PluginCategory, PluginTypeMeta, register_category

from .caching import CachingProvider
from .provider import SearchCapability
from .registry import get_provider, list_providers, register_provider

if TYPE_CHECKING:  # pragma: no cover
    from src.utils.config_models import PluginInstanceConfig

logger = logging.getLogger(__name__)

CATEGORY = "search_provider"


def _global_response_cache_enabled() -> bool:
    """Read the ``SystemConfig.enable_response_cache`` master switch (default True).

    Best-effort — any config-load failure defaults to enabled. - Claude Generated
    """
    try:
        from src.utils.config_manager import ConfigManager

        config = ConfigManager().load_config()
        return bool(getattr(config.system_config, "enable_response_cache", True))
    except Exception:
        return True


def build_provider(
    instance: "PluginInstanceConfig",
    *,
    cache: bool = False,
    ukm: Any = None,
    max_age_hours: int = 24,
    force_update: bool = False,
    cache_raw: "bool | None" = None,
) -> Any:
    """Construct the provider for one instance.

    ``cache=True`` wraps GND-keyword providers in :class:`CachingProvider` (the
    mapping-first cache), matching what ``MetaSuggester`` did inline.

    ``cache_raw`` (WP2 raw-first) is injected onto the suggester-backed provider
    *below* any ``CachingProvider`` wrapper (the dual-write happens at the
    fetch seam). ``None`` ⇒ read the global master switch.
    """
    cls = get_provider(instance.provider_id)
    # Secret settings may be overridden per env var (ALIMA_PLUGIN_<ID>_<KEY>) —
    # runtime-only, never persisted. - Claude Generated
    from src.core.plugins.schema import apply_env_overrides, warn_operator_urls

    warn_operator_urls(cls, instance)

    fields = cls.config_fields() if hasattr(cls, "config_fields") else []
    settings = apply_env_overrides(instance.instance_id, instance.settings, fields)
    provider = cls(**settings)
    if cache_raw is None:
        cache_raw = _global_response_cache_enabled()
    try:
        provider._cache_raw = cache_raw
        provider._ukm_ref = ukm
    except Exception:
        pass
    if cache and SearchCapability.GND_KEYWORDS in getattr(cls, "capabilities", set()):
        provider = CachingProvider(
            provider, ukm=ukm, max_age_hours=max_age_hours, force_update=force_update
        )
    return provider


def build_enabled(
    instances: List["PluginInstanceConfig"],
    *,
    cache_gnd: bool = True,
    ukm: Any = None,
    max_age_hours: int = 24,
) -> Dict[str, Any]:
    """Build ``{instance_id: provider}`` for all enabled instances (skips unknown types)."""
    out: Dict[str, Any] = {}
    cache_raw = _global_response_cache_enabled()  # resolve once, not per instance
    for inst in instances:
        if not getattr(inst, "enabled", True):
            continue
        try:
            out[inst.instance_id] = build_provider(
                inst, cache=cache_gnd, ukm=ukm, max_age_hours=max_age_hours,
                cache_raw=cache_raw,
            )
        except KeyError:
            logger.warning("Unknown search provider type '%s', skipping instance '%s'",
                           inst.provider_id, inst.instance_id)
    return out


def enabled_gnd_provider_ids(
    config: Any = None, *, available_only: bool = False
) -> "list | None":
    """Provider *types* (unique) with an enabled GND-keyword instance.

    Used to gate the classic keyword search by the Plugins-tab enable/disable
    state (the agentic path is already gated via per-instance tool generation).
    ``available_only=True`` additionally drops instances whose availability-gating
    settings are unsatisfied (e.g. catalog without a token) — the source-selector
    UIs use it. Returns ``None`` when the config can not be read, so callers keep
    their own default list instead of silently searching nothing. - Claude Generated
    """
    try:
        if config is None:
            from src.utils.config_manager import ConfigManager

            config = ConfigManager().load_config()
        from src.core.plugins.schema import availability_ok

        ids = []
        for inst in config.enabled_instances_for("search_provider"):
            try:
                cls = get_provider(inst.provider_id)
            except KeyError:
                continue
            if SearchCapability.GND_KEYWORDS not in getattr(cls, "capabilities", set()):
                continue
            if available_only:
                fields = cls.config_fields() if hasattr(cls, "config_fields") else []
                if not availability_ok(fields, inst.settings):
                    continue
            if inst.provider_id not in ids:
                ids.append(inst.provider_id)
        return ids
    except Exception:
        return None


def primary_settings(
    config: Any, provider_id: str, *, enabled_only: bool = True
) -> Dict[str, Any]:
    """Settings of the primary instance of ``provider_id``, over its declared defaults.

    The read-side counterpart of :func:`build_provider` for the handful of call
    sites that need a single setting rather than a built provider — the successor
    of the ``CatalogConfig`` mirror reads (WP P7). Returns ``{}`` when no such
    instance exists or the config can not be read.

    ``enabled_only`` picks the semantics deliberately:

    * ``True`` (default) — *source gates*: a disabled instance yields ``{}``, so
      disabling a plugin stops the feature (Search parity, WP P5).
    * ``False`` — *policy* settings that merely live on a plugin but are not gated
      by it (the DK step's strict-GND flag, the OPAC web bases). This is what the
      mirror did: ``derive_search_mirrors`` copied from the primary regardless of
      its enable state.

    ``ConfigField`` defaults are layered underneath, replacing the dataclass
    defaults the mirror used to supply for unset fields. - Claude Generated
    """
    try:
        if config is None:
            from src.utils.config_manager import ConfigManager

            config = ConfigManager().load_config()
        inst = config.primary_instance(CATEGORY, provider_id, include_disabled=not enabled_only)
    except Exception as e:
        logger.debug(f"primary_settings({provider_id}) config read failed: {e}")
        return {}
    if inst is None:
        return {}
    try:
        from src.core.plugins.schema import defaults

        cls = get_provider(provider_id)
        out = defaults(cls.config_fields() if hasattr(cls, "config_fields") else [])
    except KeyError:  # unregistered type (e.g. a plugin that is no longer installed)
        out = {}
    out.update(inst.settings or {})
    return out


def set_primary_settings(config: Any, provider_id: str, settings: Dict[str, Any]) -> None:
    """Merge ``settings`` into the primary instance of ``provider_id`` - Claude Generated.

    The write-side counterpart of :func:`primary_settings`, for the setup wizards:
    they assemble an ``AlimaConfig`` from scratch and used to hand their catalog
    values to the ``CatalogConfig`` mirror, relying on ``save_config`` to lift them
    into instances. Keys are the plugin's own ``ConfigField`` keys (``token``,
    ``catalog_details``, …), not the legacy attribute names. Seeds the built-in
    search instances first, so a fresh config ends up with the full set.
    """
    from src.utils.plugin_migration import ensure_search_instances

    ensure_search_instances(config.plugins)
    inst = config.primary_instance(CATEGORY, provider_id, include_disabled=True)
    if inst is None:
        logger.warning(f"set_primary_settings: no {provider_id} instance to write to")
        return
    inst.settings = {**(inst.settings or {}), **settings}


def catalog_web_bases(config: Any = None) -> "tuple[str, str]":
    """``(web_record_url, web_search_url)`` for OPAC links — catalog before finc.

    Both the ``catalog`` and the ``finc`` plugin declare a ``catalog_web_record_url``
    (finc records link into the same OPAC). They used to mirror onto *one*
    ``CatalogConfig`` field, where dict order silently let finc win — so a catalog
    URL set without a finc one produced no links at all. Precedence is explicit
    here: the catalog instance owns the OPAC base, finc fills in only when catalog
    has none. Policy read (``enabled_only=False``): a link base stays valid for
    rendering old results even when the source is switched off. - Claude Generated
    """
    cat = primary_settings(config, "catalog", enabled_only=False)
    finc = primary_settings(config, "finc", enabled_only=False)
    record = (cat.get("catalog_web_record_url") or finc.get("catalog_web_record_url") or "")
    return str(record or ""), str(cat.get("catalog_web_search_url") or "")


def gnd_tool_name(provider_id: str) -> "str | None":
    """The generated GND-keyword tool name for a provider id — its GND_KEYWORDS
    ``ProviderToolSpec`` name (e.g. ``lobid`` → ``search_lobid``). A copied plugin
    keeps the original tool name (deploy_poc only rewrites the id), which is what
    lets a requested built-in id resolve to its copy. - Claude Generated"""
    try:
        cls = get_provider(provider_id)
    except KeyError:
        return None
    for spec in (cls.mcp_tool_specs() if hasattr(cls, "mcp_tool_specs") else []):
        if getattr(spec, "capability", None) == SearchCapability.GND_KEYWORDS:
            return spec.name
    return None


def resolve_gnd_source_tools(requested: Any, config: Any = None):
    """Resolve requested GND source ids to the enabled providers backing them.

    Returns ``(provider_ids, {provider_id: tool_name})`` — the agentic replacement
    for the hardcoded ``{"swb": "search_swb", "lobid": "search_lobid"}`` map. Both
    the ids and their tool names come from the enabled GND providers, so a copied
    or renamed plugin works. A requested built-in id whose class is registered but
    *disabled* (own-plugins POC) resolves to the enabled provider sharing its tool
    name (``swb`` → ``search_swb`` → ``poc_swb``). Empty ``requested`` → every
    enabled GND provider, in order.

    Returns ``None`` when the config can't be read (caller keeps its legacy
    default rather than searching nothing); ``([], {})`` when the config is
    readable but no GND provider is enabled (honour "all disabled"). Mirrors
    :func:`enabled_gnd_provider_ids`' ``None``-vs-``[]`` discipline. - Claude Generated
    """
    ids = enabled_gnd_provider_ids(config)
    if ids is None:
        return None
    pid_to_tool: dict = {}
    tool_to_pid: dict = {}
    for pid in ids:
        tn = gnd_tool_name(pid)
        if tn:
            pid_to_tool[pid] = tn
            tool_to_pid.setdefault(tn, pid)
    if not requested:
        return list(pid_to_tool.keys()), pid_to_tool
    out_ids: list = []
    out_map: dict = {}
    for req in requested:
        pid = req if req in pid_to_tool else tool_to_pid.get(gnd_tool_name(req))
        if pid and pid not in out_map:
            out_ids.append(pid)
            out_map[pid] = pid_to_tool[pid]
    return out_ids, out_map


# Built-in provider ids — lets a *custom* CLASSIFICATION plugin take precedence
# over the Libero default while preserving the finc→SRU→Libero order for the
# built-ins. - Claude Generated
_BUILTIN_PROVIDER_IDS = {"lobid", "swb", "catalog", "finc", "sru", "gnd_local"}


def _call_dk_extractor(provider: Any, *, logger_: Any, stream_callback: Any) -> Any:
    """Call ``provider.dk_extractor`` tolerating extractors that don't accept the
    optional logger/stream kwargs (built-ins accept+ignore them). - Claude Generated"""
    try:
        return provider.dk_extractor(logger_=logger_, stream_callback=stream_callback)
    except TypeError:
        return provider.dk_extractor()


def _dk_extractor_from_instance(
    inst: Any, *, debug: bool, logger_: Any, stream_callback: Any,
    require_available: bool = True,
) -> Any:
    """Build ``inst``'s provider through the factory and return its DK extractor.

    ``None`` when the instance is missing, unbuildable, declares no
    ``dk_extractor``, or (when ``require_available``) is unavailable. ``debug`` is
    threaded into the instance settings so the extractor's client keeps its
    verbose mode. - Claude Generated
    """
    if inst is None:
        return None
    try:
        if debug:
            from dataclasses import replace
            inst = replace(inst, settings={**(inst.settings or {}), "debug": True})
        provider = build_provider(inst)
    except Exception:
        logger.warning("Failed to build DK provider '%s'",
                       getattr(inst, "provider_id", "?"), exc_info=True)
        return None
    if not hasattr(provider, "dk_extractor"):
        return None
    if require_available:
        try:
            if hasattr(provider, "is_available") and not provider.is_available():
                return None
        except Exception:
            pass
    return _call_dk_extractor(provider, logger_=logger_, stream_callback=stream_callback)


def _custom_classification_extractor(
    config: Any, *, logger_: Any, stream_callback: Any, debug: bool = False,
) -> Any:
    """First enabled *non-built-in* CLASSIFICATION-capable provider's DK extractor.

    The extension point behind the classic DK step: a library without finc/Libero
    can ship its own DK/RVK catalog plugin (declaring ``CLASSIFICATION`` + a
    ``dk_extractor()``) and it is picked up here — no core change. ``None`` when
    there is no such plugin (or the config can't be read). - Claude Generated
    """
    if config is None:
        return None
    try:
        instances = config.enabled_instances_for("search_provider")
    except Exception:
        return None
    for inst in instances:
        pid = getattr(inst, "provider_id", "")
        if pid in _BUILTIN_PROVIDER_IDS:
            continue
        try:
            cls = get_provider(pid)
        except KeyError:
            continue
        if SearchCapability.CLASSIFICATION not in getattr(cls, "capabilities", set()):
            continue
        ext = _dk_extractor_from_instance(
            inst, debug=debug, logger_=logger_, stream_callback=stream_callback
        )
        if ext is not None:
            return ext
    return None


def _sru_selected_for_dk(sru_inst: Any, catalog_inst: Any) -> bool:
    """Whether SRU is the operator's chosen DK backend.

    The new explicit knob is the sru instance's own ``dk_enabled`` (symmetric to
    finc). For configs migrated before that field existed we still honour the
    legacy ``catalog_type == 'marcxml_sru'`` on the catalog instance (incl. the
    ``'auto'`` heuristic: SRU when a preset/base_url is configured) so those
    libraries keep their DK backend. ``catalog_type`` is thereby vestigial —
    read only as a transition fallback, dropped with the mirror in WP P7.
    - Claude Generated
    """
    if sru_inst is not None and (getattr(sru_inst, "settings", None) or {}).get("dk_enabled"):
        return True
    cat_type = str(((getattr(catalog_inst, "settings", None) or {}).get("catalog_type") or "")).strip()
    if cat_type == "marcxml_sru":
        return True
    if cat_type == "auto" and sru_inst is not None:
        s = getattr(sru_inst, "settings", None) or {}
        return bool(s.get("preset") or s.get("base_url"))
    return False


def resolve_dk_extractor(
    *,
    config: Any = None,
    logger_: Any = None,
    stream_callback: Any = None,
    debug: bool = False,
) -> Any:
    """Resolve the DK/RVK extractor for the classic DK step from the enabled
    CLASSIFICATION-capable search providers — the D-4 hand-wired site.

    Every backend is built through the factory from its own instance settings
    (WP P4), replacing the former 15-kwarg ``CatalogConfig`` wall + the
    ``catalog_type`` if-elif. Precedence preserves the operator's June-2026 order:

    1. **finc** — opt-in via the finc instance's ``dk_enabled``;
    2. **custom plugin** — any enabled non-built-in provider declaring CLASSIFICATION;
    3. **SRU / MARC-XML** — opt-in via the sru instance's ``dk_enabled``
       (legacy ``catalog_type == 'marcxml_sru'`` still honoured, see
       :func:`_sru_selected_for_dk`);
    4. **Libero SOAP** — the catalog instance, the default.

    Returns an object implementing the shared
    ``extract_dk_classifications_for_keywords`` contract. - Claude Generated
    """
    if config is None:
        try:
            from src.utils.config_manager import ConfigManager
            config = ConfigManager().load_config()
        except Exception:
            config = None

    by_id: dict = {}
    if config is not None:
        try:
            for inst in config.enabled_instances_for("search_provider"):
                by_id.setdefault(getattr(inst, "provider_id", ""), inst)
        except Exception:
            pass

    common = dict(debug=debug, logger_=logger_, stream_callback=stream_callback)

    # 1. finc — opt-in.
    finc_inst = by_id.get("finc")
    if finc_inst is not None and (getattr(finc_inst, "settings", None) or {}).get("dk_enabled"):
        ext = _dk_extractor_from_instance(finc_inst, **common)
        if ext is not None:
            return ext
    # 2. Custom (non-built-in) CLASSIFICATION plugin.
    ext = _custom_classification_extractor(
        config, logger_=logger_, stream_callback=stream_callback, debug=debug
    )
    if ext is not None:
        return ext
    # 3. SRU — opt-in (new dk_enabled knob or the legacy catalog_type fallback).
    if _sru_selected_for_dk(by_id.get("sru"), by_id.get("catalog")):
        ext = _dk_extractor_from_instance(by_id.get("sru"), **common)
        if ext is not None:
            return ext
    # 4. Libero catalog — default (built unconditionally; empty token → web scraping).
    ext = _dk_extractor_from_instance(by_id.get("catalog"), require_available=False, **common)
    if ext is not None:
        return ext
    # No catalog instance configured — build the registry default.
    prov = get_provider("catalog")(debug=debug)
    return _call_dk_extractor(prov, logger_=logger_, stream_callback=stream_callback)


class SearchProviderCategory(PluginCategory):
    """Bridges ``PROVIDER_REGISTRY`` to the generic plugin framework."""

    name = CATEGORY

    def list_types(self) -> List[str]:
        return list_providers()

    def type_meta(self, type_id: str) -> PluginTypeMeta:
        from src.core.plugins.schema import PluginDoc, cache_field

        cls = get_provider(type_id)
        caps = sorted(c.value for c in getattr(cls, "capabilities", set()))
        fields = cls.config_fields() if hasattr(cls, "config_fields") else []
        doc = cls.doc() if hasattr(cls, "doc") else PluginDoc()
        return PluginTypeMeta(
            type_id=type_id,
            label=getattr(cls, "label", type_id),
            category=self.name,
            # Standard per-plugin cache toggle appended to every provider's form.
            config_fields=list(fields) + [cache_field()],
            capabilities=caps,
            builtin=True,
            doc=doc,
        )

    def build(self, instance: "PluginInstanceConfig") -> Any:
        return build_provider(instance)

    def register_code_type(self, cls: type) -> str:
        register_provider(cls)
        return getattr(cls, "id")


register_category(SearchProviderCategory())
