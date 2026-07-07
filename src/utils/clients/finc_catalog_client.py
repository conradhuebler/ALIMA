"""Compatibility re-export — the canonical ``FincCatalogClient`` now lives with
the finc search-provider plugin (its CLASSIFICATION-capability backend).

The implementation moved to
``src/core/search/providers/finc/finc_catalog_client.py`` (July 2026): it is the
finc-specific DK/RVK extractor and belongs with the plugin it backs, so the
whole ``providers/finc/`` directory stays copyable/self-contained. The classic
DK step reaches it via the finc provider's ``dk_extractor()`` (capability
resolver), not by importing this path directly.

This shim keeps the historical ``src.utils.clients.finc_catalog_client`` import
path working (tests / any legacy reader). Single source of truth — no
divergence. - Claude Generated
"""

from src.core.search.providers.finc.finc_catalog_client import FincCatalogClient  # noqa: F401

__all__ = ["FincCatalogClient"]
