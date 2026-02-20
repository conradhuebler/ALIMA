"""
UBCatalogTab - Standalone UB Catalog Search Tab for ALIMA
Executes the dk_search pipeline step: searches UB catalog for GND keywords
and aggregates DK/RVK classifications with statistics.
Claude Generated
"""

import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLabel, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter,
    QPushButton, QProgressBar, QMessageBox, QSpinBox,
    QTreeWidget, QTreeWidgetItem,
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor

from .styles import (
    get_main_stylesheet,
    get_button_styles,
    get_confidence_style,
    LAYOUT,
)


# ============================================================================
# Worker Class
# ============================================================================

class DkSearchWorker(QThread):
    """Thin worker that delegates to PipelineStepExecutor.execute_dk_search - Claude Generated"""

    result_ready   = pyqtSignal(dict)   # full pipeline result dict
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)    # stream_callback messages

    def __init__(self, keywords: list, executor, max_results: int = 40):
        super().__init__()
        self.keywords    = keywords
        self.executor    = executor
        self.max_results = max_results
        self.logger      = logging.getLogger(__name__)

    def run(self):
        try:
            def _cb(msg, step_id="dk_search"):
                self.status_updated.emit(str(msg))

            # Read catalog token from config (same pattern as pipeline_manager.py) - Claude Generated
            catalog_token = ""
            catalog_search_url = ""
            catalog_details_url = ""
            try:
                from ..utils.config_manager import ConfigManager
                catalog_config = ConfigManager().get_catalog_config()
                catalog_token = getattr(catalog_config, "catalog_token", "") or ""
                catalog_search_url = getattr(catalog_config, "catalog_search_url", "") or ""
                catalog_details_url = getattr(catalog_config, "catalog_details_url", "") or ""
            except Exception as cfg_err:
                self.logger.debug(f"Could not read catalog config: {cfg_err}")

            result = self.executor.execute_dk_search(
                keywords=self.keywords,
                stream_callback=_cb,
                max_results=self.max_results,
                catalog_token=catalog_token,
                catalog_search_url=catalog_search_url,
                catalog_details_url=catalog_details_url,
                strict_gnd_validation=False,   # UI accepts plain keywords without GND-IDs
            )
            if not isinstance(result, dict):
                result = {"classifications": result, "statistics": {}, "keyword_results": []}
            self.result_ready.emit(result)
        except Exception as e:
            self.logger.error(f"DkSearchWorker failed: {e}", exc_info=True)
            self.error_occurred.emit(str(e))


# ============================================================================
# UB Search Panel Widget
# ============================================================================

class UBSearchPanel(QWidget):
    """UB Katalog Search Panel - uses pipeline executor for DK search - Claude Generated"""

    result_ready = pyqtSignal(dict)  # Emitted after results are displayed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_worker = None
        self._executor = None
        self._current_result = {}
        self.setup_ui()

    def set_executor(self, executor):
        """Inject the PipelineStepExecutor - Claude Generated"""
        self._executor = executor

    def setup_ui(self):
        self.setStyleSheet(get_main_stylesheet())
        btn_styles = get_button_styles()

        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["spacing"])
        layout.setContentsMargins(0, 0, 0, 0)

        # Input Area
        input_group = QGroupBox("Suche")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(LAYOUT["inner_spacing"])

        # Keywords label
        input_layout.addWidget(QLabel("Keywords (kommagetrennt):"))

        self.keywords_input = QTextEdit()
        self.keywords_input.setPlaceholderText("Keywords eingeben...")
        self.keywords_input.setMaximumHeight(60)
        self.keywords_input.setFont(QFont("Segoe UI", LAYOUT["input_font_size"]))
        input_layout.addWidget(self.keywords_input)

        # Button row: [Suchen] [progress_bar, stretch] [Max.:] [spinbox]
        self.search_button = QPushButton("Suchen")
        self.search_button.setStyleSheet(btn_styles["primary"])
        self.search_button.clicked.connect(self.start_search)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # indeterminate

        self.num_results = QSpinBox()
        self.num_results.setRange(1, 60)
        self.num_results.setValue(40)
        self.num_results.setMinimumWidth(60)

        button_row = QHBoxLayout()
        button_row.addWidget(self.search_button)
        button_row.addWidget(self.progress_bar, 1)
        button_row.addWidget(QLabel("Max.:"))
        button_row.addWidget(self.num_results)
        input_layout.addLayout(button_row)

        # Results Splitter (horizontal: Klassifikationen | Zugehörige Titel)
        self.results_splitter = QSplitter(Qt.Orientation.Horizontal)

        # TreeView for classifications
        tree_container = QGroupBox("Klassifikationen")
        tree_layout = QVBoxLayout(tree_container)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Klassifikation", "Treffer"])
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.tree_widget.setSortingEnabled(True)
        tree_layout.addWidget(self.tree_widget)
        self.results_splitter.addWidget(tree_container)

        # Detail view for selected classification
        detail_container = QGroupBox("Zugehörige Titel")
        detail_layout = QVBoxLayout(detail_container)
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        detail_layout.addWidget(self.detail_view)
        self.results_splitter.addWidget(detail_container)

        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 2)

        # Vertical splitter: input (30%) | results (70%) — wie GND-Suche [300, 700]
        panel_splitter = QSplitter(Qt.Orientation.Vertical)
        panel_splitter.addWidget(input_group)
        panel_splitter.addWidget(self.results_splitter)
        panel_splitter.setSizes([300, 700])
        panel_splitter.setStretchFactor(0, 0)
        panel_splitter.setStretchFactor(1, 1)

        layout.addWidget(panel_splitter)

    def set_keywords(self, keywords: str):
        """Set keywords in the search input"""
        self.keywords_input.setPlainText(keywords)

    def start_search(self):
        if self._executor is None:
            QMessageBox.warning(self, "Fehler", "Kein Pipeline-Executor verfügbar.")
            return

        keywords = [k.strip() for k in self.keywords_input.toPlainText().split(",") if k.strip()]
        if not keywords:
            QMessageBox.warning(self, "Eingabe", "Bitte mindestens ein Stichwort eingeben.")
            return

        self.search_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.detail_view.setHtml("<p><b>Suche läuft…</b></p>")

        self.current_worker = DkSearchWorker(
            keywords=list(dict.fromkeys(keywords)),
            executor=self._executor,
            max_results=self.num_results.value(),
        )
        self.current_worker.result_ready.connect(self._on_worker_done)
        self.current_worker.status_updated.connect(self._append_status)
        self.current_worker.error_occurred.connect(self._handle_error)
        self.current_worker.finished.connect(self.search_finished)
        self.current_worker.start()

    def _on_worker_done(self, result: dict):
        """Handle completed search result - Claude Generated"""
        self._current_result = result
        classifications = result.get("classifications", [])
        self._populate_tree(classifications)
        # Clear status messages and show a hint / auto-select first item
        if classifications:
            self.detail_view.setHtml(
                "<p style='color:gray'><i>Klassifikation in der linken Liste anklicken, "
                "um zugehörige Titel anzuzeigen.</i></p>"
            )
            # Auto-select and display the top result
            first = self.tree_widget.topLevelItem(0)
            if first:
                self.tree_widget.setCurrentItem(first)
                self.on_tree_item_clicked(first, 0)
        else:
            self.detail_view.setHtml("<p><i>Keine Klassifikationen gefunden.</i></p>")
        self.result_ready.emit(result)

    def _populate_tree(self, classifications: list):
        """Fill tree widget from pipeline classifications list - Claude Generated"""
        self.tree_widget.clear()
        for cls in classifications:  # already sorted by count desc from pipeline
            cls_type = cls.get("classification_type", cls.get("type", "DK"))
            dk_code  = cls.get("dk", "")
            key      = f"{cls_type} {dk_code}".strip()
            count    = cls.get("count", 0)
            item     = QTreeWidgetItem([key, str(count)])
            item.setData(0, Qt.ItemDataRole.UserRole, cls)
            self.tree_widget.addTopLevelItem(item)

    def on_tree_item_clicked(self, item, col):
        """Show titles and matched keywords for the selected classification - Claude Generated"""
        cls = item.data(0, Qt.ItemDataRole.UserRole)
        if not cls:
            return
        titles   = cls.get("titles", [])
        keywords = cls.get("matched_keywords", []) or cls.get("keywords", [])
        count    = cls.get("count", 0)
        html = [f"<h3>{item.text(0)} &nbsp;({count}×)</h3>"]
        if keywords:
            html.append(f"<p><b>Keywords:</b> {', '.join(keywords)}</p>")
        if titles:
            html.append(f"<h4>Beispieltitel ({len(titles)}):</h4><ul>")
            for t in titles[:20]:
                # Escape HTML special chars to prevent broken markup
                t_safe = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html.append(f"<li>{t_safe}</li>")
            html.append("</ul>")
            if len(titles) > 20:
                html.append(f"<p><i>… und {len(titles)-20} weitere</i></p>")
        else:
            html.append(
                "<p><i>Keine Titel im Cache – "
                "Suche wurde möglicherweise über Web-Fallback ohne Titeldetails durchgeführt.</i></p>"
            )
        self.detail_view.setHtml("".join(html))

    def _append_status(self, msg: str):
        """Append progress message to detail view - Claude Generated"""
        self.detail_view.append(msg.rstrip())

    def _handle_error(self, error_message: str):
        self.logger.error(f"DK search error: {error_message}")
        self.detail_view.append(f"<b style='color:red'>Fehler:</b> {error_message}")
        QMessageBox.critical(self, "Fehler", error_message)

    def search_finished(self):
        self.search_button.setEnabled(True)
        self.progress_bar.setVisible(False)


# ============================================================================
# DK Statistics Panel Widget
# ============================================================================

class DKStatisticsPanel(QWidget):
    """Panel for displaying DK classification statistics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["spacing"])
        layout.setContentsMargins(0, 0, 0, 0)

        # Deduplication Stats
        self.dedup_label = QLabel()
        self.dedup_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.dedup_label)

        # Top 10 Table
        layout.addWidget(QLabel("<b>Top 10 Most Frequent Classifications:</b>"))
        self.top10_table = QTableWidget()
        self.top10_table.setColumnCount(4)
        self.top10_table.setHorizontalHeaderLabels([
            "DK Code", "Occurrences", "Keywords", "Confidence"
        ])
        self.top10_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.top10_table.horizontalHeader().setStretchLastSection(True)
        self.top10_table.setMinimumHeight(150)
        layout.addWidget(self.top10_table)

        # Clear button
        clear_btn = QPushButton("Clear Statistics")
        clear_btn.setStyleSheet(get_button_styles()["secondary"])
        clear_btn.clicked.connect(self.clear_statistics)
        layout.addWidget(clear_btn)

    def update_statistics(self, dk_statistics: dict):
        """Update the statistics panel with new data"""
        if not dk_statistics:
            self.dedup_label.setText("No statistics available.")
            self.top10_table.setRowCount(0)
            return

        # Deduplication summary
        dedup = dk_statistics.get("deduplication_stats", {})
        if dedup:
            dedup_html = (
                f"<b>Deduplication:</b> {dedup.get('original_count', 0)} → "
                f"<b>{dk_statistics.get('total_classifications', 0)}</b> "
                f"(Rate: <b>{dedup.get('deduplication_rate', '0%')}</b>)"
            )
            self.dedup_label.setText(dedup_html)

        # Top 10 Table
        most_frequent = dk_statistics.get("most_frequent", [])
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

            # Confidence (Color-coded)
            unique_titles = item.get('unique_titles', item.get('count', 0))
            text_color, bg_color, label, bar = get_confidence_style(unique_titles)

            conf_item = QTableWidgetItem(f"{bar} {label}")
            conf_item.setBackground(QColor(bg_color))
            self.top10_table.setItem(row, 3, conf_item)

        self.top10_table.resizeColumnsToContents()

    def clear_statistics(self):
        self.dedup_label.setText("")
        self.top10_table.setRowCount(0)


# ============================================================================
# UB Catalog Tab - Standalone tab for dk_search pipeline step
# ============================================================================

class UBCatalogTab(QWidget):
    """
    Standalone UB Catalog Search Tab.
    Executes the dk_search pipeline step: searches UB catalog for GND keywords,
    aggregates DK/RVK classifications, shows statistics, and forwards results
    to DK-Analyse for LLM classification.
    Claude Generated
    """

    search_completed = pyqtSignal(list)  # Emits results_flattened for DK-Analyse

    def __init__(self, pipeline_manager=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._pipeline_manager = pipeline_manager
        self.setup_ui()
        if pipeline_manager is not None and hasattr(pipeline_manager, 'pipeline_executor'):
            self.ub_search_panel.set_executor(pipeline_manager.pipeline_executor)

    def setup_ui(self):
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["spacing"])
        layout.setContentsMargins(
            LAYOUT["margin"], LAYOUT["margin"],
            LAYOUT["margin"], LAYOUT["margin"]
        )

        # UBSearchPanel (full search functionality)
        self.ub_search_panel = UBSearchPanel()
        self.ub_search_panel.result_ready.connect(self._on_search_results)

        # DKStatisticsPanel (below search)
        stats_group = QGroupBox("DK Classification Statistics")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(
            LAYOUT["margin"], LAYOUT["margin"],
            LAYOUT["margin"], LAYOUT["margin"]
        )
        self.stats_panel = DKStatisticsPanel()
        stats_layout.addWidget(self.stats_panel)

        # Vertical splitter: search+results (700) | stats (300) — analog zu GND-Suche
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.ub_search_panel)
        splitter.addWidget(stats_group)
        splitter.setSizes([700, 300])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter)

    @pyqtSlot(object)
    def update_from_pipeline(self, analysis_state):
        """Auto-fill GND keywords from pipeline results - Claude Generated"""
        if not analysis_state:
            return
        if hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
            gnd_keywords = analysis_state.final_llm_analysis.extracted_gnd_keywords
            if gnd_keywords:
                if isinstance(gnd_keywords, list):
                    keywords_str = ", ".join(gnd_keywords)
                else:
                    keywords_str = str(gnd_keywords)
                self.ub_search_panel.set_keywords(keywords_str)

    def update_keywords(self, keywords: str):
        """Backward-compatibility wrapper for set_keywords on the search panel"""
        self.ub_search_panel.set_keywords(keywords)

    @pyqtSlot(dict)
    def _on_search_results(self, result: dict):
        """
        Handle pipeline search result: update stats panel and emit search_completed.
        The pipeline result dict already has the correct format for both consumers.
        Claude Generated
        """
        statistics      = result.get("statistics", {})
        classifications = result.get("classifications", [])

        if statistics:
            self.stats_panel.update_statistics(statistics)

        if classifications:
            # classifications list already in pipeline format expected by DK-Analyse
            self.search_completed.emit(classifications)
