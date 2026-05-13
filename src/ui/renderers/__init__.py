"""Renderer plugin package - Claude Generated (WP10 P-α).

Importing this package auto-registers all bundled renderers (analog to
``src/core/agents/__init__.py``). Slot vocabulary follows WP3
(``slot:<snake_case>``).
"""
from .base import BaseRenderer  # noqa: F401
from .registry import (  # noqa: F401
    FALLBACK_SLOT,
    RENDERER_REGISTRY,
    get_renderer,
    list_renderers,
    register_renderer,
)

# Auto-register bundled renderers. Each import triggers the
# ``@register_renderer`` decorator side-effect.
from . import raw_json  # noqa: F401,E402
from . import dk_table  # noqa: F401,E402
from . import gnd_pool  # noqa: F401,E402
from . import keyword_chains  # noqa: F401,E402
from . import duplicate_table  # noqa: F401,E402
from . import title_list  # noqa: F401,E402

__all__ = [
    "BaseRenderer",
    "FALLBACK_SLOT",
    "RENDERER_REGISTRY",
    "get_renderer",
    "list_renderers",
    "register_renderer",
]
