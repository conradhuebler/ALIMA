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


def lookup_effect(cls: Any) -> "tuple[str, str]":
    """``(effect, effect_multi)`` of a lookup type for the settings form.

    Agent tools come from the type's ``mcp_tool_specs`` (one set per enabled
    instance); every other caller goes through ``build_lookup``, which takes the
    first enabled instance. - Claude Generated
    """
    try:
        has_tools = bool(cls.mcp_tool_specs()) if hasattr(cls, "mcp_tool_specs") else False
    except Exception:
        has_tools = False
    if has_tools:
        effect = ("Nachschlagen: Werkzeug für den Agenten; Programmteile, die es direkt "
                  "aufrufen, nutzen die erste aktive Instanz.")
        multi = ("Mehrere Instanzen dieses Typs: der Agent erhält für jede aktive Instanz "
                 "eigene Werkzeuge; direkte Aufrufe nutzen die erste aktive in der Liste.")
    else:
        effect = "Nachschlagen: Programmteile, die es direkt aufrufen, nutzen die erste aktive Instanz."
        multi = "Mehrere Instanzen dieses Typs: die erste aktive Instanz in der Liste gilt."
    return effect, multi


class LookupCategory(PluginCategory):
    name = CATEGORY

    def list_types(self) -> List[str]:
        return list_lookups()

    def type_meta(self, type_id: str) -> PluginTypeMeta:
        from src.core.plugins.schema import PluginDoc, cache_field

        cls = get_lookup(type_id)
        fields = cls.config_fields() if hasattr(cls, "config_fields") else []
        doc = cls.doc() if hasattr(cls, "doc") else PluginDoc()
        effect, effect_multi = lookup_effect(cls)
        return PluginTypeMeta(
            type_id=type_id,
            label=getattr(cls, "label", type_id),
            category=self.name,
            config_fields=list(fields) + [cache_field()],
            capabilities=[],
            builtin=True,
            doc=doc,
            effect=effect,
            effect_multi=effect_multi,
        )

    def type_class(self, type_id: str) -> Any:
        """The registered lookup class (read only) - Claude Generated"""
        return get_lookup(type_id)

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
