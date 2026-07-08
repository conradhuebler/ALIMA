"""Lookup plugin category — small external-API/authority queries as agent tools.

Importing the package self-registers the category adapter + every built-in lookup
plugin (side-effect imports below), mirroring ``src/core/search`` and
``src/utils/input_sources``. - Claude Generated
"""

from .registry import (  # noqa: F401
    LOOKUP_REGISTRY,
    LookupToolSpec,
    get_lookup,
    list_lookups,
    lookup_tool_specs,
    register_lookup,
)
from . import category  # noqa: F401 — registers LookupCategory
from . import rvk  # noqa: F401 — registers the rvk_api lookup
from . import k10plus  # noqa: F401 — registers the k10plus lookup

__all__ = [
    "LOOKUP_REGISTRY",
    "LookupToolSpec",
    "get_lookup",
    "list_lookups",
    "lookup_tool_specs",
    "register_lookup",
]
