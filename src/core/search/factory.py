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
