"""TitleList renderer - Claude Generated (WP10 P-β).

New renderer for the ``title_list_search`` workflow output (WP3 Sek 2).
No HEAD source.

Input slot: ``slot:title_list``.

Input data shape::

    [
        {
            "title": "Cadmium in soils",
            "authors": ["Müller, K."],
            "year": 2024,
            "isbn": "978-3-…",
            "dk_codes": ["57.62"],
            "rvk_codes": [],
        },
        …
    ]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseRenderer
from .registry import register_renderer

COLUMNS = ["Titel", "Autoren", "Jahr", "ISBN", "DK / RVK"]


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_authors(authors: Any) -> str:
    if isinstance(authors, list):
        return "; ".join(str(a) for a in authors)
    return str(authors or "")


def _format_codes(row: Dict[str, Any]) -> str:
    dks = row.get("dk_codes") or []
    rvks = row.get("rvk_codes") or []
    parts: List[str] = []
    if dks:
        parts.append("DK: " + ", ".join(str(c) for c in dks))
    if rvks:
        parts.append("RVK: " + ", ".join(str(c) for c in rvks))
    return " · ".join(parts)


@register_renderer("slot:title_list")
class TitleListRenderer(BaseRenderer):
    """Bibliographische Titelliste."""

    output_slot = "slot:title_list"

    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        rows: List[Dict[str, Any]] = list(data or [])
        if not rows:
            return "<p><i>(keine Titel)</i></p>"

        cards: List[str] = []
        for row in rows:
            title = _escape_html(row.get("title", ""))
            authors = _escape_html(_format_authors(row.get("authors")))
            year = row.get("year")
            isbn = _escape_html(str(row.get("isbn", "")))
            codes = _escape_html(_format_codes(row))

            meta_lines: List[str] = []
            if authors:
                meta_lines.append(f"<dt>Autoren</dt><dd>{authors}</dd>")
            if year:
                meta_lines.append(f"<dt>Jahr</dt><dd>{year}</dd>")
            if isbn:
                meta_lines.append(f"<dt>ISBN</dt><dd>{isbn}</dd>")
            if codes:
                meta_lines.append(f"<dt>Klassifikation</dt><dd>{codes}</dd>")

            cards.append(
                "<article style='border-left:3px solid #4a86e8; padding:6px 10px; margin:6px 0;'>"
                f"<h3 style='margin:0; font-size:11pt;'>{title}</h3>"
                f"<dl style='font-size:9pt; margin:4px 0;'>{''.join(meta_lines)}</dl>"
                "</article>"
            )
        return "<div>" + "".join(cards) + "</div>"

    def render_qt(self, data: Any, context: Optional[Dict] = None):
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

        rows: List[Dict[str, Any]] = list(data or [])
        table = QTableWidget(len(rows), len(COLUMNS))
        table.setHorizontalHeaderLabels(COLUMNS)

        for row_idx, row in enumerate(rows):
            table.setItem(row_idx, 0, QTableWidgetItem(str(row.get("title", ""))))
            table.setItem(row_idx, 1, QTableWidgetItem(_format_authors(row.get("authors"))))
            year = row.get("year")
            table.setItem(row_idx, 2, QTableWidgetItem(str(year) if year else ""))
            table.setItem(row_idx, 3, QTableWidgetItem(str(row.get("isbn", ""))))
            table.setItem(row_idx, 4, QTableWidgetItem(_format_codes(row)))

        table.resizeColumnsToContents()
        return table

    def render_cli(self, data: Any, context: Optional[Dict] = None) -> str:
        rows: List[Dict[str, Any]] = list(data or [])
        if not rows:
            return "(no titles)"
        lines: List[str] = []
        for row in rows:
            authors = _format_authors(row.get("authors"))
            year = row.get("year") or ""
            codes = _format_codes(row)
            line = f"{row.get('title', '')} — {authors} ({year})"
            if codes:
                line += f"  [{codes}]"
            lines.append(line)
        return "\n".join(lines)
