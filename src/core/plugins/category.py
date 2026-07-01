"""Plugin-category abstraction - Claude Generated.

A *category* is one kind of extension point (search providers, input sources, …).
Each category already owns a concrete registry elsewhere in the codebase
(``PROVIDER_REGISTRY`` for search, ``INPUT_SOURCE_REGISTRY`` for input). A
:class:`PluginCategory` adapter bridges the generic plugin machinery (manifest
parsing, directory loading, the settings UI) to that concrete registry so the
framework never has to special-case a category.

Adapters live next to their category's code (e.g. the search adapter in
``src/core/search/``) and self-register here on import via
:func:`register_category`, mirroring how providers self-register.

This module is **Qt-free** and does not import any concrete category, so it stays
a small shared hub with no import cycles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from .schema import ConfigField, PluginDoc

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import edge
    from src.utils.config_models import PluginInstanceConfig


@dataclass
class PluginTypeMeta:
    """Static description of one plugin *type* within a category.

    ``config_fields`` is the declarative schema (drives the settings form +
    availability gating). ``capabilities`` are category-specific tags (search:
    the ``SearchCapability`` values; input: the handled input kinds). ``doc`` is
    the plugin's natural-language self-description (what it does + input/output).
    """

    type_id: str
    label: str
    category: str
    config_fields: List[ConfigField] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    builtin: bool = True
    doc: PluginDoc = field(default_factory=PluginDoc)


class PluginCategory(ABC):
    """Adapter binding the generic plugin layer to one concrete registry."""

    name: str = ""

    @abstractmethod
    def list_types(self) -> List[str]:
        """Registered type ids in this category (built-ins + loaded plugins)."""

    @abstractmethod
    def type_meta(self, type_id: str) -> PluginTypeMeta:
        """Static metadata for ``type_id``. Raises ``KeyError`` if unknown."""

    @abstractmethod
    def build(self, instance: "PluginInstanceConfig") -> Any:
        """Construct the runtime object for a configured instance.

        Implementations read ``instance.provider_id`` (the type) and
        ``instance.settings`` (values keyed by ``ConfigField.key``).
        """

    def register_code_type(self, cls: type) -> str:
        """Register a Tier-2 code plugin's class into the concrete registry.

        Returns the type id it registered under. Categories that do not support
        code plugins may leave this unimplemented.
        """
        raise NotImplementedError(
            f"Category '{self.name}' does not support code plugins"
        )

    def has_type(self, type_id: str) -> bool:
        return type_id in self.list_types()

    def config_fields(self, type_id: str) -> List[ConfigField]:
        return self.type_meta(type_id).config_fields


PLUGIN_CATEGORY_REGISTRY: Dict[str, PluginCategory] = {}


def register_category(adapter: PluginCategory) -> PluginCategory:
    """Register a category adapter under its ``name`` (idempotent by identity)."""
    name = getattr(adapter, "name", None)
    if not name or not isinstance(name, str):
        raise ValueError(
            f"Category adapter {type(adapter).__name__} must define a non-empty `name`"
        )
    existing = PLUGIN_CATEGORY_REGISTRY.get(name)
    if existing is not None and existing is not adapter and type(existing) is not type(adapter):
        raise ValueError(
            f"Plugin category '{name}' already registered to {type(existing).__name__}"
        )
    PLUGIN_CATEGORY_REGISTRY[name] = adapter
    return adapter


def get_category(name: str) -> PluginCategory:
    """Look up a category adapter by name. Raises ``KeyError`` if absent."""
    if name not in PLUGIN_CATEGORY_REGISTRY:
        raise KeyError(
            f"Unknown plugin category '{name}'. Registered: {sorted(PLUGIN_CATEGORY_REGISTRY)}"
        )
    return PLUGIN_CATEGORY_REGISTRY[name]


def list_categories() -> List[str]:
    return sorted(PLUGIN_CATEGORY_REGISTRY)


def _reset_for_tests() -> None:
    """Clear the category registry — used only by tests that register fakes."""
    PLUGIN_CATEGORY_REGISTRY.clear()
