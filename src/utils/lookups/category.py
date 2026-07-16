"""Lookup plugin category adapter - Claude Generated.

Bridges the ``LOOKUP_REGISTRY`` to the generic plugin framework (settings UI,
directory loader), mirroring ``SearchProviderCategory`` / ``InputSourceCategory``.
Self-registers on import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from src.core.plugins.category import PluginCategory, PluginTypeMeta, register_category

from .registry import get_lookup, list_lookups, register_lookup

if TYPE_CHECKING:  # pragma: no cover
    from src.utils.config_models import PluginInstanceConfig

CATEGORY = "lookup"


class LookupCategory(PluginCategory):
    name = CATEGORY

    def list_types(self) -> List[str]:
        return list_lookups()

    def type_meta(self, type_id: str) -> PluginTypeMeta:
        from src.core.plugins.schema import PluginDoc, cache_field

        cls = get_lookup(type_id)
        fields = cls.config_fields() if hasattr(cls, "config_fields") else []
        doc = cls.doc() if hasattr(cls, "doc") else PluginDoc()
        return PluginTypeMeta(
            type_id=type_id,
            label=getattr(cls, "label", type_id),
            category=self.name,
            config_fields=list(fields) + [cache_field()],
            capabilities=[],
            builtin=True,
            doc=doc,
        )

    def build(self, instance: "PluginInstanceConfig") -> Any:
        cls = get_lookup(instance.provider_id)
        from src.core.plugins.schema import apply_env_overrides, warn_operator_urls

        warn_operator_urls(cls, instance)
        fields = cls.config_fields() if hasattr(cls, "config_fields") else []
        settings = apply_env_overrides(instance.instance_id, instance.settings, fields)
        # A LookupProvider takes **config, so operator settings must reach it. A
        # TypeError here means the plugin's __init__ doesn't conform — surface it
        # instead of silently dropping the settings via cls(). - Claude Generated
        return cls(**settings)

    def register_code_type(self, cls: type) -> str:
        register_lookup(cls)
        return getattr(cls, "id")


register_category(LookupCategory())
