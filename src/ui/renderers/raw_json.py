"""raw_json fallback renderer - Claude Generated (WP10 P-α).

Renderer of last resort. WP4 Sek 5.4. Auto-registered via package
``__init__.py`` import so ``get_renderer(fallback="slot:raw_json")``
never raises ``KeyError``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import BaseRenderer
from .registry import register_renderer


def _escape_html(text: str) -> str:
    """Minimal HTML-entity escape for ``<``, ``>``, ``&``."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@register_renderer("slot:raw_json")
class RawJsonRenderer(BaseRenderer):
    """Pretty-printed JSON in a monospaced ``<pre>``-block.

    Inherits :meth:`render_qt` and :meth:`render_cli` defaults from
    :class:`BaseRenderer`.
    """

    output_slot = "slot:raw_json"

    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        return (
            "<pre style='background:#1e1e1e; color:#dcdcdc;"
            " padding:8px; font-family:monospace;"
            f" white-space:pre-wrap;'>{_escape_html(body)}</pre>"
        )
