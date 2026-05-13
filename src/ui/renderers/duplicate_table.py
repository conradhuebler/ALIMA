"""DuplicateTable renderer - Claude Generated (WP10 P-β).

New renderer for the ``title_list_search`` workflow output. WP4 Sek 5.3.
No HEAD source: this is the first UI render for ``extra.duplicate_analysis``.

Input slot: ``slot:duplicate_table`` (WP3 Sek 2).

Input data shape::

    [
        {
            "input_title": "Cadmium in soils",
            "status": "duplicate" | "likely_duplicate"
                    | "different_edition" | "new" | "no_match",
            "matches": [{"rsn": "123", "title": "…"}],
            "reasoning": "ISBN match",
        },
        …
    ]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseRenderer
from .registry import register_renderer

COLUMNS = ["Status", "Eingabe", "#Matches", "Begründung"]

_STATUS_COLOR = {
    "duplicate": "#dc3545",         # red
    "likely_duplicate": "#fd7e14",  # orange
    "different_edition": "#ffc107", # yellow
    "new": "#28a745",               # green
    "no_match": "#6c757d",          # grey
}


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@register_renderer("slot:duplicate_table")
class DuplicateTableRenderer(BaseRenderer):
    """Duplikat-Analyse mit Status-Spalte."""

    output_slot = "slot:duplicate_table"

    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        rows: List[Dict[str, Any]] = list(data or [])
        if not rows:
            return "<p><i>(keine Duplikatanalyse)</i></p>"

        head_cells = "".join(
            f"<th style='text-align:left; padding:4px 8px;'>{_escape_html(c)}</th>"
            for c in COLUMNS
        )
        body_parts: List[str] = []
        for row in rows:
            status = str(row.get("status", "no_match"))
            color = _STATUS_COLOR.get(status, "#888")
            matches = row.get("matches") or []
            body_parts.append(
                "<tr>"
                f"<td style='color:{color}; font-weight:bold; padding:4px 8px; border-top:1px solid #444;'>{_escape_html(status)}</td>"
                f"<td style='padding:4px 8px; border-top:1px solid #444;'>{_escape_html(row.get('input_title', ''))}</td>"
                f"<td style='padding:4px 8px; border-top:1px solid #444;'>{len(matches)}</td>"
                f"<td style='font-size:9pt; color:#666; padding:4px 8px; border-top:1px solid #444;'>"
                f"{_escape_html(row.get('reasoning', ''))}</td>"
                "</tr>"
            )
        return (
            "<table style='border-collapse:collapse; font-family:Arial,sans-serif;'>"
            f"<thead><tr>{head_cells}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"
        )

    def render_qt(self, data: Any, context: Optional[Dict] = None):
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

        rows: List[Dict[str, Any]] = list(data or [])
        table = QTableWidget(len(rows), len(COLUMNS))
        table.setHorizontalHeaderLabels(COLUMNS)

        for row_idx, row in enumerate(rows):
            status = str(row.get("status", "no_match"))
            color = _STATUS_COLOR.get(status, "#888")
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor(color))
            table.setItem(row_idx, 0, status_item)
            table.setItem(row_idx, 1, QTableWidgetItem(str(row.get("input_title", ""))))
            matches = row.get("matches") or []
            table.setItem(row_idx, 2, QTableWidgetItem(str(len(matches))))
            table.setItem(row_idx, 3, QTableWidgetItem(str(row.get("reasoning", ""))))

        table.resizeColumnsToContents()
        return table

    def render_cli(self, data: Any, context: Optional[Dict] = None) -> str:
        rows: List[Dict[str, Any]] = list(data or [])
        if not rows:
            return "(no duplicates)"
        lines = [f"{'STATUS':<20} TITLE"]
        for row in rows:
            lines.append(
                f"{str(row.get('status', 'no_match')):<20} "
                f"{row.get('input_title', '')}"
            )
        return "\n".join(lines)
