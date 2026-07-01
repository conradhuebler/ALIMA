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


def build_provider(
    instance: "PluginInstanceConfig",
    *,
    cache: bool = False,
    ukm: Any = None,
    max_age_hours: int = 24,
    force_update: bool = False,
) -> Any:
    """Construct the provider for one instance.

    ``cache=True`` wraps GND-keyword providers in :class:`CachingProvider` (the
    mapping-first cache), matching what ``MetaSuggester`` did inline.
    """
    cls = get_provider(instance.provider_id)
    provider = cls(**dict(instance.settings or {}))
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
    for inst in instances:
        if not getattr(inst, "enabled", True):
            continue
        try:
            out[inst.instance_id] = build_provider(
                inst, cache=cache_gnd, ukm=ukm, max_age_hours=max_age_hours
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
