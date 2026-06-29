"""Capability-based search-provider plugin system - Claude Generated.

The single standard + registry for ALIMA search sources (lobid, swb, catalog,
finc, local GND). Importing this package self-registers all built-in providers.
See ``docs/search_provider_plugins.md``.
"""

from .provider import (
    ProviderResult,
    ResultItem,
    SearchCapability,
    SearchProvider,
)
from .registry import (
    PROVIDER_REGISTRY,
    get_provider,
    list_providers,
    providers_for_capability,
    register_provider,
)

# Side-effect import: register all built-in providers.
from . import providers as _providers  # noqa: F401,E402

__all__ = [
    "ProviderResult",
    "ResultItem",
    "SearchCapability",
    "SearchProvider",
    "PROVIDER_REGISTRY",
    "get_provider",
    "list_providers",
    "providers_for_capability",
    "register_provider",
]
