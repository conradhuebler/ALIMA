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

# Operator-URL warnings already emitted this process (anti-spam: build_provider
# runs per search operation). - Claude Generated
_warned_operator_urls: set = set()


def _warn_operator_urls(cls: type, instance: "PluginInstanceConfig") -> None:
    """Log net_guard posture-(a) warnings for URL-kind settings, once each - Claude Generated"""
    try:
        from src.core.plugins.schema import URL
        from src.utils.net_guard import check_operator_url

        fields = cls.config_fields() if hasattr(cls, "config_fields") else []
        for fld in fields:
            if getattr(fld, "kind", None) != URL:
                continue
            value = str((instance.settings or {}).get(fld.key) or "")
            for msg in check_operator_url(value):
                key = (instance.instance_id, fld.key, msg)
                if key not in _warned_operator_urls:
                    _warned_operator_urls.add(key)
                    logger.warning("Plugin '%s': %s", instance.instance_id, msg)
    except Exception:
        pass


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
    _warn_operator_urls(cls, instance)
    # Secret settings may be overridden per env var (ALIMA_PLUGIN_<ID>_<KEY>) —
    # runtime-only, never persisted. - Claude Generated
    from src.core.plugins.schema import apply_env_overrides

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


def enabled_gnd_provider_ids(config: Any = None) -> "list | None":
    """Provider *types* (unique) with an enabled GND-keyword instance.

    Used to gate the classic keyword search by the Plugins-tab enable/disable
    state (the agentic path is already gated via per-instance tool generation).
    Returns ``None`` when the config can not be read, so callers keep their own
    default list instead of silently searching nothing. - Claude Generated
    """
    try:
        if config is None:
            from src.utils.config_manager import ConfigManager

            config = ConfigManager().load_config()
        ids = []
        for inst in config.enabled_instances_for("search_provider"):
            try:
                cls = get_provider(inst.provider_id)
            except KeyError:
                continue
            if SearchCapability.GND_KEYWORDS in getattr(cls, "capabilities", set()):
                if inst.provider_id not in ids:
                    ids.append(inst.provider_id)
        return ids
    except Exception:
        return None


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


def _custom_classification_extractor(config: Any, *, logger_: Any, stream_callback: Any) -> Any:
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
        try:
            provider = build_provider(inst)
        except Exception:
            logger.warning("Failed to build custom DK provider '%s'", pid, exc_info=True)
            continue
        if not hasattr(provider, "dk_extractor"):
            continue
        try:
            if hasattr(provider, "is_available") and not provider.is_available():
                continue
        except Exception:
            pass
        return _call_dk_extractor(provider, logger_=logger_, stream_callback=stream_callback)
    return None


def resolve_dk_extractor(
    *,
    config: Any = None,
    logger_: Any = None,
    stream_callback: Any = None,
    debug: bool = False,
    finc_base_url: str = "",
    finc_web_record_url: str = "",
    finc_institution_filter: str = "",
    finc_timeout: int = 30,
    finc_default_limit: int = 50,
    finc_dk_enabled: bool = False,
    catalog_type: str = "libero_soap",
    sru_preset: str = "",
    sru_base_url: str = "",
    sru_max_records: int = 50,
    catalog_token: str = "",
    catalog_search_url: str = "",
    catalog_details_url: str = "",
    catalog_web_search_url: str = "",
    catalog_web_record_url: str = "",
) -> Any:
    """Resolve the DK/RVK extractor for the classic DK step from the active
    CLASSIFICATION-capable search providers — the D-4 hand-wired site this
    module's docstring names.

    Precedence (preserves the operator's June-2026 order, now capability-driven
    instead of an if-elif over client classes):

    1. **finc** — opt-in (``finc_dk_enabled`` + a real ``finc_base_url``);
    2. **custom plugin** — any enabled non-built-in provider declaring
       ``CLASSIFICATION`` (the extensibility point for other libraries);
    3. **SRU / MARC-XML** — when ``catalog_type == "marcxml_sru"``;
    4. **Libero SOAP** — default.

    Returns an object implementing the shared
    ``extract_dk_classifications_for_keywords`` contract, built via the provider
    layer so it is byte-equivalent to the former direct client construction. - Claude Generated
    """
    # 1. finc (opt-in DK source).
    if finc_dk_enabled and isinstance(finc_base_url, str) and finc_base_url.strip():
        prov = get_provider("finc")(
            base_url=finc_base_url,
            web_record_url=finc_web_record_url or "",
            institution_filter=finc_institution_filter or "",
            timeout=int(finc_timeout or 30),
            default_limit=int(finc_default_limit or 50),
        )
        return _call_dk_extractor(prov, logger_=logger_, stream_callback=stream_callback)
    # 2. Custom (non-built-in) CLASSIFICATION plugin.
    ext = _custom_classification_extractor(config, logger_=logger_, stream_callback=stream_callback)
    if ext is not None:
        return ext
    # 3. SRU / MARC-XML.
    if catalog_type == "marcxml_sru":
        prov = get_provider("sru")(
            preset=sru_preset or "",
            base_url=(sru_base_url if not sru_preset else ""),
            max_records=int(sru_max_records or 50),
            debug=debug,
        )
        return _call_dk_extractor(prov, logger_=logger_, stream_callback=stream_callback)
    # 4. Libero SOAP (default).
    prov = get_provider("catalog")(
        token=catalog_token or "",
        catalog_search_url=catalog_search_url or "",
        catalog_details=catalog_details_url or "",
        catalog_web_search_url=catalog_web_search_url or "",
        catalog_web_record_url=catalog_web_record_url or "",
        debug=debug,
    )
    return _call_dk_extractor(prov, logger_=logger_, stream_callback=stream_callback)


class SearchProviderCategory(PluginCategory):
    """Bridges ``PROVIDER_REGISTRY`` to the generic plugin framework."""

    name = CATEGORY

    def list_types(self) -> List[str]:
        return list_providers()

    def type_meta(self, type_id: str) -> PluginTypeMeta:
        from src.core.plugins.schema import PluginDoc

        cls = get_provider(type_id)
        caps = sorted(c.value for c in getattr(cls, "capabilities", set()))
        fields = cls.config_fields() if hasattr(cls, "config_fields") else []
        doc = cls.doc() if hasattr(cls, "doc") else PluginDoc()
        return PluginTypeMeta(
            type_id=type_id,
            label=getattr(cls, "label", type_id),
            category=self.name,
            config_fields=list(fields),
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
