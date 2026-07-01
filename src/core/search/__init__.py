"""Capability-based search-provider plugin system - Claude Generated.

The single standard + registry for ALIMA search sources (lobid, swb, catalog,
finc, local GND). Importing this package self-registers all built-in providers.
See ``docs/search_provider_plugins.md``.
"""

from .provider import (
    ProviderResult,
    ProviderToolSpec,
    ResultItem,
    SearchCapability,
    SearchProvider,
)
from .registry import (
    PROVIDER_REGISTRY,
    get_provider,
    list_providers,
    provider_tool_specs,
    providers_for_capability,
    register_provider,
)

# Side-effect import: register all built-in providers.
from . import providers as _providers  # noqa: F401,E402

# Side-effect import: the config→provider factory + search plugin-category adapter
# (self-registers ``SearchProviderCategory`` into the generic plugin framework).
from .factory import build_enabled, build_provider, enabled_gnd_provider_ids  # noqa: F401,E402

__all__ = [
    "ProviderResult",
    "ProviderToolSpec",
    "ResultItem",
    "SearchCapability",
    "SearchProvider",
    "PROVIDER_REGISTRY",
    "get_provider",
    "list_providers",
    "provider_tool_specs",
    "providers_for_capability",
    "register_provider",
    "build_provider",
    "build_enabled",
    "enabled_gnd_provider_ids",
]
