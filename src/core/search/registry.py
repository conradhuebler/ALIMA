"""Single registry for search providers - Claude Generated.

Mirrors the ``@register_step`` / ``@register_tool_fn`` idiom in
``src/core/agents/registry.py``: a module-level dict of ``id -> provider class``,
populated by the ``@register_provider`` class decorator so providers self-register
on import (side-effect imports live in ``src/core/search/providers/__init__.py``).

This replaces the two parallel registries the old layer had to keep in sync (the
``SuggesterType`` enum *and* the MCP tool registry). Both the orchestrator and the
tool layer enumerate this one registry. See ``docs/search_provider_plugins.md``.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .provider import _DEFAULT_RAW_CACHE_PARAM_KEYS, SearchCapability, SearchProvider

PROVIDER_REGISTRY: Dict[str, Type[SearchProvider]] = {}


def register_provider(cls: Type[SearchProvider]) -> Type[SearchProvider]:
    """Class decorator: register a provider by its class-level ``id``.

    Re-registering the same name to a *different* class raises ``ValueError``
    (guards against silent shadowing), matching the agents registry.
    """
    pid = getattr(cls, "id", None)
    if not pid or not isinstance(pid, str):
        raise ValueError(f"Provider {cls.__name__} must define a non-empty string `id`")
    if pid in PROVIDER_REGISTRY and PROVIDER_REGISTRY[pid] is not cls:
        raise ValueError(
            f"Provider id '{pid}' already registered to {PROVIDER_REGISTRY[pid].__name__}"
        )
    PROVIDER_REGISTRY[pid] = cls
    return cls


def get_provider(pid: str) -> Type[SearchProvider]:
    """Look up a provider class by its registered ``id``.

    Raises:
        KeyError: if ``pid`` is not registered.
    """
    if pid not in PROVIDER_REGISTRY:
        raise KeyError(
            f"Unknown provider '{pid}'. Registered: {sorted(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[pid]


def list_providers() -> List[str]:
    """Return the sorted list of registered provider ids."""
    return sorted(PROVIDER_REGISTRY)


def providers_for_capability(capability: SearchCapability) -> List[str]:
    """Return sorted ids of providers that declare ``capability``."""
    return sorted(
        pid
        for pid, cls in PROVIDER_REGISTRY.items()
        if capability in getattr(cls, "capabilities", set())
    )


def raw_cache_param_keys(source: str) -> tuple:
    """Return the WP2 raw-cache key params a source declares - Claude Generated.

    ``source`` is a cache-source label; unregistered labels (``catalog_titles``)
    and providers declaring nothing get the default. Read off the class, so a
    copied plugin dir keeps its own key shape instead of colliding on the default.
    """
    cls = PROVIDER_REGISTRY.get(source)
    if cls is None:
        return _DEFAULT_RAW_CACHE_PARAM_KEYS
    return getattr(cls, "raw_cache_param_keys", _DEFAULT_RAW_CACHE_PARAM_KEYS)


def provider_tool_specs() -> List:
    """Collect the MCP ``ProviderToolSpec``s declared by all registered providers.

    The MCP tool layer enumerates these to *generate* ToolDefinitions + handlers
    (no hand-written per-source schema/handler). Providers without specs (e.g.
    gnd_local, which is a knowledge tool, not a library tool) are skipped.
    """
    specs: List = []
    for pid in sorted(PROVIDER_REGISTRY):
        cls = PROVIDER_REGISTRY[pid]
        fn = getattr(cls, "mcp_tool_specs", None)
        if callable(fn):
            for spec in fn():
                spec.provider_id = pid  # tag with owning provider for config gating
                specs.append(spec)
    return specs


def _reset_for_tests() -> None:
    """Clear the registry — used only by tests that register fakes."""
    PROVIDER_REGISTRY.clear()
