"""GndPool renderer - Claude Generated (WP10 P-β).

Migration of search-results table from
``src/ui/analysis_review_tab.py:827-857`` (``populate_search_results_table``).

Input slot: ``slot:gnd_pool`` (WP3 Sek 2).

Input data shape (per row, normalized for renderer)::

    [
        {
            "search_term": "Cadmium",
            "keyword": "Cadmium",
            "count": 42,
            "gnd_id": "4007249-3",
        },
        …
    ]

The 4 columns match ``analysis_review_tab.py`` column headers (L254-256):
``["Suchbegriff", "Keyword", "Count", "GND-ID"]``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseRenderer
from .registry import register_renderer

COLUMNS = ["Suchbegriff", "Keyword", "Count", "GND-ID"]


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort: accept either the raw gnd_pool-slot shape (title +
    gnd_id) or the AnalysisReviewTab-normalized shape (search_term +
    keyword + count + gnd_id)."""
    if "search_term" in row or "keyword" in row:
        return {
            "search_term": str(row.get("search_term", "")),
            "keyword": str(row.get("keyword", "")),
            "count": int(row.get("count", 0)),
            "gnd_id": str(row.get("gnd_id", "")),
        }
    # raw gnd_pool shape: {gnd_id, title, count, …}
    return {
        "search_term": "",
        "keyword": str(row.get("title", "")),
        "count": int(row.get("count", 0)),
        "gnd_id": str(row.get("gnd_id", "")),
    }


@register_renderer("slot:gnd_pool")
class GndPoolRenderer(BaseRenderer):
    """GND-Suche / Pool-Tabelle."""

    output_slot = "slot:gnd_pool"

    # -- Qt -----------------------------------------------------------------

    def render_qt(self, data: Any, context: Optional[Dict] = None):
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

        rows = [_normalize_row(r) for r in (data or [])]
        table = QTableWidget(len(rows), len(COLUMNS))
        table.setHorizontalHeaderLabels(COLUMNS)

        for row_idx, row in enumerate(rows):
            table.setItem(row_idx, 0, QTableWidgetItem(row["search_term"]))
            table.setItem(row_idx, 1, QTableWidgetItem(row["keyword"]))
            table.setItem(row_idx, 2, QTableWidgetItem(str(row["count"])))
            table.setItem(row_idx, 3, QTableWidgetItem(row["gnd_id"]))

        table.resizeColumnsToContents()
        return table

    def fill_table(self, table, data: Any) -> None:
        """Populate an *existing* QTableWidget in place (keeps layout
        references stable). Used by AnalysisReviewTab integration."""
        from PyQt6.QtWidgets import QTableWidgetItem

        rows = [_normalize_row(r) for r in (data or [])]
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            table.setItem(row_idx, 0, QTableWidgetItem(row["search_term"]))
            table.setItem(row_idx, 1, QTableWidgetItem(row["keyword"]))
            table.setItem(row_idx, 2, QTableWidgetItem(str(row["count"])))
            table.setItem(row_idx, 3, QTableWidgetItem(row["gnd_id"]))
        table.resizeColumnsToContents()

    # -- HTML ---------------------------------------------------------------

    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        rows = [_normalize_row(r) for r in (data or [])]
        if not rows:
            return "<table><tr><td><i>(keine GND-Treffer)</i></td></tr></table>"

        head = "".join(
            f"<th style='text-align:left; padding:4px 8px;'>{_escape_html(c)}</th>"
            for c in COLUMNS
        )
        body_parts: List[str] = []
        for row in rows:
            cells = [
                row["search_term"],
                row["keyword"],
                str(row["count"]),
                row["gnd_id"],
            ]
            body_parts.append(
                "<tr>"
                + "".join(
                    f"<td style='padding:4px 8px; border-top:1px solid #444;'>{_escape_html(c)}</td>"
                    for c in cells
                )
                + "</tr>"
            )
        return (
            "<table style='border-collapse:collapse; font-family:Arial,sans-serif;'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"
        )

    # -- CLI ----------------------------------------------------------------

    def render_cli(self, data: Any, context: Optional[Dict] = None) -> str:
        rows = [_normalize_row(r) for r in (data or [])]
        if not rows:
            return "(no GND hits)"
        lines = ["\t".join(COLUMNS)]
        for row in rows:
            lines.append(
                "\t".join(
                    [
                        row["search_term"],
                        row["keyword"],
                        str(row["count"]),
                        row["gnd_id"],
                    ]
                )
            )
        return "\n".join(lines)
