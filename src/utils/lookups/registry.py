"""Registry + contract for lookup plugins - Claude Generated.

A *lookup* is a small external-API/authority query that returns codes/records for
a value — e.g. an RVK-classification API search or notation validation. It is the
third plugin category alongside search providers (``src/core/search/``) and input
sources (``src/utils/input_sources/``): a module-level ``id -> class`` dict
populated by ``@register_lookup`` so plugins self-register on import.

Lookups differ from search providers (which return rankable GND/title records into
the pipeline pool) and input sources (which turn a reference into extracted text):
a lookup returns authority data (classification codes, ancestors, validation) and
is exposed purely as an agent tool. Each declares ``config_fields`` (per-instance
config), ``doc`` (self-description) and ``mcp_tool_specs`` (the tools it exposes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.plugins.schema import ConfigField

LOOKUP_REGISTRY: Dict[str, type] = {}


@dataclass
class LookupToolSpec:
    """One MCP tool a lookup plugin exposes.

    ``method`` is the plugin method the generated handler calls with the tool
    arguments (filtered to ``parameters``). ``cache_key_param`` names the argument
    used as the raw-response cache key; empty ⇒ not cached. - Claude Generated
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    method: str
    provider_id: str = ""  # stamped by the registry when collecting specs
    cacheable: bool = True
    cache_key_param: str = ""


def register_lookup(cls: type) -> type:
    """Class decorator: register a lookup plugin by its class-level ``id``."""
    lid = getattr(cls, "id", None)
    if not lid or not isinstance(lid, str):
        raise ValueError(f"Lookup {cls.__name__} must define a non-empty string `id`")
    if lid in LOOKUP_REGISTRY and LOOKUP_REGISTRY[lid] is not cls:
        raise ValueError(
            f"Lookup id '{lid}' already registered to {LOOKUP_REGISTRY[lid].__name__}"
        )
    LOOKUP_REGISTRY[lid] = cls
    return cls


def get_lookup(lid: str) -> type:
    if lid not in LOOKUP_REGISTRY:
        raise KeyError(f"Unknown lookup '{lid}'. Registered: {sorted(LOOKUP_REGISTRY)}")
    return LOOKUP_REGISTRY[lid]


def list_lookups() -> List[str]:
    return sorted(LOOKUP_REGISTRY)


def lookup_tool_specs() -> List[LookupToolSpec]:
    """Collect every registered lookup's tool specs, each stamped with its
    ``provider_id`` (the plugin id). - Claude Generated"""
    out: List[LookupToolSpec] = []
    for lid, cls in LOOKUP_REGISTRY.items():
        specs = cls.mcp_tool_specs() if hasattr(cls, "mcp_tool_specs") else []
        for spec in specs:
            spec.provider_id = lid
            out.append(spec)
    return out


def _reset_for_tests() -> None:
    LOOKUP_REGISTRY.clear()
