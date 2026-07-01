"""Input sources — self-registering plugin category - Claude Generated.

Turns raw input references (text, file, PDF, image, URL, DOI) into extracted text
via a registry (mirroring ``src/core/search``). Importing this package registers
every built-in source and the :class:`InputSourceCategory` adapter. See
``docs/plugin_system.md``.
"""

from __future__ import annotations

from .registry import (
    INPUT_SOURCE_REGISTRY,
    InputSource,
    InputToolSpec,
    get_input_source,
    list_input_sources,
    register_input_source,
)

# Side-effect imports: register built-in sources.
from . import builtin as _builtin  # noqa: F401,E402
from . import url_fetch as _url_fetch  # noqa: F401,E402
from . import doi as _doi  # noqa: F401,E402

# Side-effect import: register the category adapter.
from . import category as _category  # noqa: F401,E402

__all__ = [
    "INPUT_SOURCE_REGISTRY",
    "InputSource",
    "InputToolSpec",
    "get_input_source",
    "list_input_sources",
    "register_input_source",
]
