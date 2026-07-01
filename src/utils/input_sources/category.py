"""Input-source plugin-category adapter - Claude Generated.

Bridges ``INPUT_SOURCE_REGISTRY`` to the generic plugin framework so input sources
appear in the settings UI and can be added from the plugin directory, exactly like
search providers. Self-registers on import (see this package's ``__init__``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from src.core.plugins.category import PluginCategory, PluginTypeMeta, register_category

from .registry import get_input_source, list_input_sources, register_input_source

if TYPE_CHECKING:  # pragma: no cover
    from src.utils.config_models import PluginInstanceConfig

CATEGORY = "input_source"


class InputSourceCategory(PluginCategory):
    name = CATEGORY

    def list_types(self) -> List[str]:
        return list_input_sources()

    def type_meta(self, type_id: str) -> PluginTypeMeta:
        from src.core.plugins.schema import PluginDoc

        cls = get_input_source(type_id)
        fields = cls.config_fields() if hasattr(cls, "config_fields") else []
        doc = cls.doc() if hasattr(cls, "doc") else PluginDoc()
        return PluginTypeMeta(
            type_id=type_id,
            label=getattr(cls, "label", type_id),
            category=self.name,
            config_fields=list(fields),
            capabilities=[],
            builtin=True,
            doc=doc,
        )

    def build(self, instance: "PluginInstanceConfig") -> Any:
        cls = get_input_source(instance.provider_id)
        try:
            return cls(**dict(instance.settings or {}))
        except TypeError:
            return cls()

    def register_code_type(self, cls: type) -> str:
        register_input_source(cls)
        return getattr(cls, "id")


register_category(InputSourceCategory())
