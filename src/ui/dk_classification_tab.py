"""
Dedicated DK-Zuordnung Tab for ALIMA - Claude Generated
Displays DK/RVK classification results with statistics and transparency.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLabel, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QColor
import logging
from datetime import datetime
from typing import List, Dict, Any

from .styles import (
    get_main_stylesheet,
    get_status_label_styles,
    get_confidence_style,
    LAYOUT,
    COLORS,
)

class DkClassificationTab(QWidget):
    """Tab for displaying and analyzing DK/RVK classification results - Claude Generated"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface - Claude Generated"""
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"])
        layout.setSpacing(LAYOUT["spacing"])

        # Status Label (consistent with other tabs)
        self.status_label = QLabel("Status: Bereit")
        self.status_label.setStyleSheet(get_status_label_styles()["info"])
        layout.addWidget(self.status_label)

        # 1. Summary Header
        summary_group = QGroupBox("Zusammenfassung")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_label = QLabel("Führen Sie die Pipeline aus, um DK-Klassifikationen zu erhalten.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(summary_group)

        # 2. Main Content Splitter
        content_splitter = QSplitter(Qt.Orientation.Vertical)

        # Results View (Scrollable HTML)
        results_group = QGroupBox("📚 DK/RVK-Klassifikationen & Transparenz")
        results_layout = QVBoxLayout(results_group)
        self.results_view = QTextEdit()
        self.results_view.setReadOnly(True)
        self.results_view.setPlaceholderText("Ergebnisse werden nach Abschluss von Schritt 6 angezeigt...")
        results_layout.addWidget(self.results_view)
        content_splitter.addWidget(results_group)

        # Statistics Section
        stats_group = QGroupBox("📊 Analyse-Statistiken")
        stats_layout = QVBoxLayout(stats_group)

        # Deduplication Stats
        self.dedup_label = QLabel()
        self.dedup_label.setTextFormat(Qt.TextFormat.RichText)
        stats_layout.addWidget(self.dedup_label)

        # Top 10 Table
        stats_layout.addWidget(QLabel("<b>Top 10 Häufigste Klassifikationen im Katalog-Pool:</b>"))
        self.top10_table = QTableWidget()
        self.top10_table.setColumnCount(4)
        self.top10_table.setHorizontalHeaderLabels([
            "DK Code", "Vorkommen", "Keywords", "Konfidenz"
        ])
        self.top10_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.top10_table.horizontalHeader().setStretchLastSection(True)
        self.top10_table.setMinimumHeight(200)
        stats_layout.addWidget(self.top10_table)

        content_splitter.addWidget(stats_group)

        # Initial sizes: Results get more space
        content_splitter.setSizes([600, 300])
        layout.addWidget(content_splitter)

    @pyqtSlot(object)
    def update_data(self, analysis_state):
        """Update the tab with new analysis results - Claude Generated"""
        if not analysis_state:
            return

        # 1. Update Summary Label and Status
        working_title = getattr(analysis_state, 'working_title', 'Unbenannte Analyse')
        timestamp = datetime.now().strftime('%d.%m.%Y %H:%M')
        dk_count = len(analysis_state.dk_classifications) if analysis_state.dk_classifications else 0

        self.status_label.setText(f"Status: {dk_count} Klassifikationen geladen")
        self.status_label.setStyleSheet(get_status_label_styles()["success"])

        summary_html = f"<h3>{working_title}</h3>"
        summary_html += f"<p><b>Pipeline-Stand:</b> {timestamp}</p>"
        summary_html += f"<p>Es wurden <b>{dk_count} finale DK/RVK-Klassifikationen</b> ermittelt.</p>"
        self.summary_label.setText(summary_html)

        # 2. Update Results View
        if analysis_state.dk_classifications:
            dk_search_results = analysis_state.dk_search_results_flattened or []
            html_display = self._format_dk_classifications_with_titles(
                analysis_state.dk_classifications,
                dk_search_results
            )
            self.results_view.setHtml(html_display)
        else:
            self.results_view.setPlainText("Keine DK/RVK-Klassifikationen generiert.")

        # 3. Update Statistics
        if analysis_state.dk_statistics:
            stats = analysis_state.dk_statistics

            # Deduplication summary
            dedup = stats.get("deduplication_stats", {})
            if dedup:
                dedup_html = (
                    f"<b>Deduplizierung:</b> {dedup.get('original_count', 0)} → "
                    f"<b>{stats.get('total_classifications', 0)}</b> "
                    f"(Rate: <b>{dedup.get('deduplication_rate', '0%')}</b>)"
                )
                self.dedup_label.setText(dedup_html)

            # Top 10 Table
            most_frequent = stats.get("most_frequent", [])
            self.top10_table.setRowCount(len(most_frequent))

            for row, item in enumerate(most_frequent):
                # DK Code
                self.top10_table.setItem(row, 0, QTableWidgetItem(item.get('dk', 'unknown')))

                # Count
                count_item = QTableWidgetItem(str(item.get('count', 0)))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
                self.top10_table.setItem(row, 1, count_item)

                # Keywords
                keywords = item.get('keywords', [])
                self.top10_table.setItem(row, 2, QTableWidgetItem(', '.join(keywords[:3])))

                # Confidence (Color-coded) - Claude Generated
                unique_titles = item.get('unique_titles', item.get('count', 0))
                text_color, bg_color, label, bar = get_confidence_style(unique_titles)

                conf_item = QTableWidgetItem(f"{bar} {label}")
                conf_item.setBackground(QColor(bg_color))
                self.top10_table.setItem(row, 3, conf_item)

            self.top10_table.resizeColumnsToContents()
        else:
            self.dedup_label.setText("Keine Statistikdaten verfügbar.")
            self.top10_table.setRowCount(0)

    def _format_dk_classifications_with_titles(
        self,
        dk_classifications: List[str],
        dk_search_results: List[Dict[str, Any]]
    ) -> str:
        """Format DK classifications with catalog titles using HTML - Claude Generated"""
        html_parts = []
        html_parts.append("<html><body style='font-family: Arial, sans-serif;'>")

        for idx, dk_code in enumerate(dk_classifications, 1):
            titles, total_count = self._get_titles_for_dk_code(dk_code, dk_search_results)

            # Color-coding based on frequency (confidence) - Claude Generated
            color, bg_color, _, _ = get_confidence_style(total_count)

            # Header with background
            html_parts.append(
                f"<div style='background-color: {bg_color}; padding: 12px; margin-bottom: 8px; "
                f"border-left: 4px solid {color}; border-radius: 4px;'>"
                f"<h2 style='color: {color}; margin: 0; font-size: 13pt;'>#{idx} {dk_code}</h2>"
            )

            if total_count > 0:
                confidence_bar = "🟩" * min(5, (total_count // 10) + 1)
                html_parts.append(
                    f"<p style='color: {color}; font-weight: bold; margin: 5px 0 2px 0;'>"
                    f"{confidence_bar} {total_count} Katalog-Treffer</p>"
                )
            html_parts.append("</div>")

            # Title list
            if titles:
                html_parts.append("<ol style='font-size: 9pt; padding-left: 25px;'>")
                for title in titles[:15]:  # Limit to 15 titles for display
                    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    html_parts.append(f"<li>{safe_title}</li>")
                html_parts.append("</ol>")

                if total_count > 15:
                    html_parts.append(f"<p style='color: #888; font-style: italic; padding-left: 15px;'>... und {total_count - 15} weitere Titel</p>")
            else:
                html_parts.append("<p style='color: #888; font-style: italic; padding-left: 15px;'>Keine Katalog-Titel für diesen Code im aktuellen Such-Pool gefunden.</p>")

            html_parts.append("<br>")

        html_parts.append("</body></html>")
        return "".join(html_parts)

    def _get_titles_for_dk_code(
        self,
        dk_code: str,
        dk_search_results: List[Dict[str, Any]]
    ) -> tuple[list, int]:
        """Extract titles for a specific DK code - Claude Generated"""
        if not dk_search_results:
            return ([], 0)

        normalized_code = dk_code.replace("DK ", "").strip()

        for result in dk_search_results:
            result_code = str(result.get("dk", "")).strip()
            if result_code == normalized_code:
                titles = result.get("titles", [])
                return (titles, len(titles))

        return ([], 0)
