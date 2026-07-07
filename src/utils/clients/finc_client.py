"""Compatibility re-export — the canonical ``FincClient`` now lives with the
finc search-provider plugin so that directory is self-contained/copyable.

The implementation moved to ``src/core/search/providers/finc/finc_client.py``
(July 2026): it is finc-specific code and belongs with the plugin it backs,
which lets the whole ``providers/finc/`` directory be copied out as a
standalone plugin (imported there as ``from .finc_client import FincClient``).

This shim preserves the historical ``src.utils.clients.finc_client`` import
path for the core-pipeline consumers that legitimately use the client outside
the plugin (``finc_catalog_client``, ``clients/__init__``, tests). There is a
single source of truth — no divergence. - Claude Generated
"""

from src.core.search.providers.finc.finc_client import FincClient  # noqa: F401

__all__ = ["FincClient"]
