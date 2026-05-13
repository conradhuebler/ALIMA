"""KeywordChains renderer - Claude Generated (WP10 P-β).

Migration of chains-rendering from
``src/ui/agentic_context_widget.py:417-441`` (``_chains``-Methode).

Input slot: ``slot:keyword_chains`` (WP3 Sek 2).

Input data shape::

    [
        {"chain": ["Cadmium", "Ökotoxikologie"], "reason": "…"},
        …
    ]

``context`` may carry::

    {"truncated": int, "label_prefix": "keyword_chains"}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseRenderer
from .registry import register_renderer


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@register_renderer("slot:keyword_chains")
class KeywordChainsRenderer(BaseRenderer):
    """Schlagwortketten mit Begründung."""

    output_slot = "slot:keyword_chains"

    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        if not isinstance(data, list) or not data:
            return ""

        ctx = context or {}
        items = list(data)
        truncated = int(ctx.get("truncated", 0) or 0)
        # Detect inline truncation sentinel (shared_context.py:18-99 pattern)
        if items and isinstance(items[-1], dict) and "_truncated" in items[-1]:
            truncated = int(items[-1]["_truncated"])
            items = items[:-1]

        label_prefix = str(ctx.get("label_prefix", "keyword_chains"))
        trunc_text = (
            f" <span style='color:#ff79c6;'>+{truncated} more</span>"
            if truncated
            else ""
        )

        rows: List[str] = [
            f"<div style='color:#8be9fd; margin-top:4px;'>{_escape_html(label_prefix)} "
            f"({len(items)}{trunc_text})</div>"
        ]
        for ch in items:
            if not isinstance(ch, dict):
                continue
            chain = ch.get("chain") or []
            reason = ch.get("reason", "")
            chain_str = " → ".join(str(c) for c in chain)
            rows.append(
                f"<div><span style='color:#f8f8f2'>• "
                f"{_escape_html(chain_str)}</span>"
                f" <span style='color:#6272a4'>{_escape_html(reason)}</span></div>"
            )
        return "".join(rows)

    def render_qt(self, data: Any, context: Optional[Dict] = None):
        from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

        items = list(data or [])
        tree = QTreeWidget()
        tree.setHeaderLabels(["Kette", "Begründung"])

        for ch in items:
            if not isinstance(ch, dict):
                continue
            chain = " → ".join(str(c) for c in (ch.get("chain") or []))
            reason = str(ch.get("reason", ""))
            QTreeWidgetItem(tree, [chain, reason])

        return tree

    def render_cli(self, data: Any, context: Optional[Dict] = None) -> str:
        items = list(data or [])
        if not items:
            return "(no chains)"
        lines: List[str] = []
        for ch in items:
            if not isinstance(ch, dict):
                continue
            chain = " → ".join(str(c) for c in (ch.get("chain") or []))
            reason = ch.get("reason", "")
            lines.append(f"  • {chain}    [{reason}]")
        return "\n".join(lines)
