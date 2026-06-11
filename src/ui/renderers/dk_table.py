"""DkTable renderer - Claude Generated (WP10 P-β).

Migration of DK-classification HTML render block from
``src/ui/analysis_review_tab.py:670-732`` (color-divs + ``<ol>``-titles)
and the Top-10 QTableWidget from ``populate_dk_statistics`` (L957-1008).

Input slot: ``slot:dk_table`` (WP3 Sek 2).

Input data shape::

    [
        {
            "dk": "57.62",
            "type": "DK" | "RVK",        # optional, default "DK"
            "count": 7,                    # total occurrences
            "titles": ["Schwermetalle …"], # zero or more catalog titles
            "keywords": ["Cadmium", …],    # optional, used in QTable column
        },
        …
    ]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseRenderer
from .registry import register_renderer


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@register_renderer("slot:dk_table")
class DkTableRenderer(BaseRenderer):
    """DK/RVK-Klassifikations-Tabelle."""

    output_slot = "slot:dk_table"

    # -- HTML (verbatim from analysis_review_tab.py:670-732) ---------------

    def render_html(self, data: Any, context: Optional[Dict] = None) -> str:
        from src.ui.styles import get_confidence_style

        rows: List[Dict[str, Any]] = list(data or [])
        if not rows:
            return "<html><body><p>Keine DK/RVK-Klassifikationen vorhanden</p></body></html>"

        parts: List[str] = ["<html><body style='font-family: Arial, sans-serif;'>"]

        for idx, row in enumerate(rows, 1):
            dk_code = row.get("dk", "")
            titles = list(row.get("titles") or [])
            total_count = int(row.get("count", len(titles)))

            color, bg_color, _, _ = get_confidence_style(total_count)

            parts.append(
                f"<div style='background-color: {bg_color}; padding: 12px; margin-bottom: 8px; "
                f"border-left: 4px solid {color}; border-radius: 4px;'>"
                f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                f"<h2 style='color: {color}; margin: 0; font-size: 14pt;'>#{idx} {_escape_html(dk_code)}</h2>"
            )

            if total_count > 0:
                confidence_bar = "\U0001f7e9" * min(5, (total_count // 10) + 1)
                parts.append(
                    f"<span style='color: {color}; font-weight: bold; font-size: 10pt;'>{confidence_bar} {total_count}</span>"
                )
            parts.append("</div>")

            if total_count > 0:
                parts.append(
                    f"<p style='color: {color}; margin: 5px 0 0 0; font-size: 9pt; opacity: 0.8;'>"
                    f"\U0001f4da Katalogisiert in {total_count} Titel{'n' if total_count != 1 else ''}</p>"
                )
            parts.append("</div>")

            if titles:
                # Collapsible title list: the full set lives inside <details>
                # so long lists stay readable (no 3/5-title truncation). Needs a
                # <details>-capable view (QWebEngineView), not QTextEdit. - Claude Generated
                parts.append("<div style='padding-left: 20px; margin-bottom: 20px;'>")
                parts.append(
                    f"<details><summary style='cursor: pointer; color: {color}; "
                    f"font-size: 9pt; font-weight: bold;'>\U0001f4d6 {len(titles)} Titel anzeigen</summary>"
                )
                parts.append("<ol style='font-size: 9pt; line-height: 1.6;'>")
                for title in titles:
                    parts.append(f"<li>{_escape_html(str(title))}</li>")
                parts.append("</ol>")
                if total_count > len(titles):
                    parts.append(
                        f"<p style='color: #888; font-style: italic; font-size: 9pt;'>"
                        f"... und {total_count - len(titles)} weitere Titel</p>"
                    )
                parts.append("</details></div>")
            else:
                parts.append(
                    "<div style='padding-left: 20px; margin-bottom: 20px;'>"
                    "<p style='color: #888; font-style: italic; font-size: 9pt;'>Keine Titel gefunden</p>"
                    "</div>"
                )

        parts.append("</body></html>")
        return "".join(parts)

    # -- Qt (verbatim from populate_dk_statistics L957-1008) ---------------

    def render_qt(self, data: Any, context: Optional[Dict] = None):
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

        from src.ui.styles import get_confidence_style

        rows: List[Dict[str, Any]] = list(data or [])
        table = QTableWidget(len(rows), 6)
        table.setHorizontalHeaderLabels(
            ["Rank", "DK Code", "Type", "Count", "Keywords", "Confidence"]
        )

        for row_idx, item in enumerate(rows):
            rank_item = QTableWidgetItem(str(row_idx + 1))
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row_idx, 0, rank_item)

            table.setItem(row_idx, 1, QTableWidgetItem(item.get("dk", "unknown")))

            type_item = QTableWidgetItem(item.get("type", "DK"))
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row_idx, 2, type_item)

            count_item = QTableWidgetItem(str(item.get("count", 0)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            table.setItem(row_idx, 3, count_item)

            keywords = list(item.get("keywords") or [])
            kw_display = ", ".join(keywords[:3])
            if len(keywords) > 3:
                kw_display += f" (+{len(keywords) - 3} more)"
            table.setItem(row_idx, 4, QTableWidgetItem(kw_display))

            unique_titles = item.get("unique_titles", item.get("count", 0))
            text_color, bg_color, label, bar = get_confidence_style(unique_titles)
            conf_item = QTableWidgetItem(f"{bar} {label}")
            conf_item.setBackground(QColor(bg_color))
            table.setItem(row_idx, 5, conf_item)

        table.resizeColumnsToContents()
        return table

    # -- CLI ----------------------------------------------------------------

    def render_cli(self, data: Any, context: Optional[Dict] = None) -> str:
        rows: List[Dict[str, Any]] = list(data or [])
        if not rows:
            return "(no DK classifications)"

        lines = [f"{'DK':<10} {'Type':<4} {'Count':>5}  Titles"]
        for row in rows:
            titles = row.get("titles") or []
            preview = "; ".join(str(t) for t in titles[:2])
            lines.append(
                f"{row.get('dk', ''):<10} "
                f"{row.get('type', 'DK'):<4} "
                f"{row.get('count', 0):>5}  "
                f"{preview[:60]}"
            )
        return "\n".join(lines)
