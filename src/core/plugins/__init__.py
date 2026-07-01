"""Generic, category-agnostic plugin framework - Claude Generated.

Shared machinery for ALIMA's extension points: a declarative config schema
(:mod:`.schema`), a category abstraction bridging to the concrete registries
(:mod:`.category`), directory-plugin manifests (:mod:`.manifest`), a two-tier
loader (:mod:`.loader`), and static security checks for code plugins
(:mod:`.security`).

Concrete categories (search providers, input sources, …) register a
:class:`PluginCategory` adapter here; nothing in this package imports a concrete
category, so it stays a cycle-free hub. See ``docs/plugin_system.md``.
"""

from __future__ import annotations

from .category import (
    PLUGIN_CATEGORY_REGISTRY,
    PluginCategory,
    PluginTypeMeta,
    get_category,
    list_categories,
    register_category,
)
from .manifest import ManifestError, PluginManifest, load_manifest_file, parse_manifest
from .schema import (
    BOOL,
    CHOICE,
    ConfigField,
    INT,
    PluginDoc,
    SECRET,
    TEXT,
    URL,
    availability_ok,
    coerce_settings,
    defaults,
)

__all__ = [
    # schema
    "ConfigField",
    "PluginDoc",
    "TEXT",
    "SECRET",
    "INT",
    "BOOL",
    "URL",
    "CHOICE",
    "defaults",
    "coerce_settings",
    "availability_ok",
    # category
    "PluginCategory",
    "PluginTypeMeta",
    "PLUGIN_CATEGORY_REGISTRY",
    "register_category",
    "get_category",
    "list_categories",
    # manifest
    "PluginManifest",
    "ManifestError",
    "parse_manifest",
    "load_manifest_file",
]
