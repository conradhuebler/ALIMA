"""Abstract base for slot renderers - Claude Generated (WP10 P-α).

Mirror to ``BaseStep`` in ``src/core/agents/steps/base_step.py``.
Subclasses register via ``@register_renderer(slot)`` in
``src/ui/renderers/registry.py``.

Three render methods, per-frontend (WP4 Sek 6):
  * :meth:`render_html`  - canonical, abstract.
  * :meth:`render_qt`    - default: QTextEdit with setHtml(html).
  * :meth:`render_cli`   - default: json.dumps.

Subclass attribute :attr:`output_slot` MUST match the WP3 vocabulary
(``"slot:<snake_case>"``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseRenderer(ABC):
    """Renderer for a WP3 output-slot."""

    output_slot: str = ""
    data_schema: Any = None

    @abstractmethod
    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        """Return self-contained HTML (no external CSS/JS)."""

    def render_qt(self, data: Any, context: Optional[Dict] = None):
        """Return a QWidget. Default: QTextEdit (read-only) with HTML body."""
        from PyQt6.QtWidgets import QTextEdit

        widget = QTextEdit()
        widget.setReadOnly(True)
        widget.setHtml(self.render_html(data, context))
        return widget

    def render_cli(self, data: Any, context: Optional[Dict] = None) -> str:
        """Return plain-text representation. Default: pretty-printed JSON."""
        import json

        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
