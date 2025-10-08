from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QTextEdit,
    QLabel,
    QPushButton,
    QScrollArea,
    QGroupBox,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
)  # Claude Generated - Removed QFileDialog (now handled by AnalysisPersistence)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..core.data_models import KeywordAnalysisState, LlmKeywordAnalysis
from ..utils.pipeline_utils import AnalysisPersistence


class AnalysisReviewTab(QWidget):
    """Tab for reviewing and exporting analysis results - Claude Generated (Refactored)"""

    # Signals
    keywords_selected = pyqtSignal(str)
    abstract_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_analysis: Optional[KeywordAnalysisState] = None  # Claude Generated - Now uses KeywordAnalysisState

        # Batch review mode - Claude Generated
        self.batch_mode = False
        self.batch_results: List[tuple[str, KeywordAnalysisState]] = []  # (filename, state)

        self.setup_ui()

    def receive_analysis_data(
        self, abstract_text: str, keywords: str = "", analysis_result: str = ""
    ):
        """Receive analysis data from AbstractTab - Claude Generated (Refactored)"""
        # Create KeywordAnalysisState for unified data handling
        keyword_list = keywords.split(", ") if keywords else []

        # Create minimal LlmKeywordAnalysis from analysis result if provided
        final_llm_analysis = None
        if analysis_result:
            final_llm_analysis = LlmKeywordAnalysis(
                task_name="abstract_analysis",
                model_used="unknown",
                provider_used="unknown",
                prompt_template="",
                filled_prompt="",
                temperature=0.7,
                seed=None,
                response_full_text=analysis_result,
                extracted_gnd_keywords=keyword_list,
                extracted_gnd_classes=[]
            )

        self.current_analysis = KeywordAnalysisState(
            original_abstract=abstract_text,
            initial_keywords=keyword_list,
            search_suggesters_used=["auto_transfer"],
            initial_gnd_classes=[],
            search_results=[],
            initial_llm_call_details=None,
            final_llm_analysis=final_llm_analysis,
            timestamp=datetime.now().isoformat()
        )

        # Update UI
        self.populate_analysis_data()
        self.populate_detail_tabs()

        # Enable buttons
        self.export_button.setEnabled(True)
        self.use_keywords_button.setEnabled(True)
        self.use_abstract_button.setEnabled(True)

        self.logger.info("Analysis data received from AbstractTab")

    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Control buttons
        button_layout = QHBoxLayout()

        self.load_button = QPushButton("Analyse laden")
        self.load_button.clicked.connect(self.load_analysis)
        button_layout.addWidget(self.load_button)

        # Batch mode toggle button - Claude Generated
        self.batch_toggle_button = QPushButton("📋 Batch-Ansicht")
        self.batch_toggle_button.clicked.connect(self.toggle_batch_mode)
        self.batch_toggle_button.setCheckable(True)
        button_layout.addWidget(self.batch_toggle_button)

        self.export_button = QPushButton("Als JSON exportieren")
        self.export_button.clicked.connect(self.export_analysis)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)

        self.use_keywords_button = QPushButton("Keywords in Suche verwenden")
        self.use_keywords_button.clicked.connect(self.use_keywords_in_search)
        self.use_keywords_button.setEnabled(False)
        button_layout.addWidget(self.use_keywords_button)

        self.use_abstract_button = QPushButton("Abstract in Analyse verwenden")
        self.use_abstract_button.clicked.connect(self.use_abstract_in_analysis)
        self.use_abstract_button.setEnabled(False)
        button_layout.addWidget(self.use_abstract_button)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Batch review table - Claude Generated (initially hidden)
        self.setup_batch_table()
        main_layout.addWidget(self.batch_table_widget)

        # Main content splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Steps overview
        self.steps_tree = QTreeWidget()
        self.steps_tree.setHeaderLabels(["Analyse-Schritte", "Status"])
        self.steps_tree.itemClicked.connect(self.on_step_selected)
        self.steps_tree.setMaximumWidth(300)
        self.main_splitter.addWidget(self.steps_tree)

        # Right side: Details
        self.details_tabs = QTabWidget()
        self.main_splitter.addWidget(self.details_tabs)

        # Initialize detail tabs
        self.init_detail_tabs()

        # Set splitter sizes
        self.main_splitter.setSizes([300, 700])

        main_layout.addWidget(self.main_splitter)

    def init_detail_tabs(self):
        """Initialize the detail tabs"""
        # Original Abstract tab
        self.abstract_text = QTextEdit()
        self.abstract_text.setReadOnly(True)
        font = self.abstract_text.font()
        font.setPointSize(11)
        self.abstract_text.setFont(font)
        self.details_tabs.addTab(self.abstract_text, "Original Abstract")

        # Initial Keywords tab
        self.initial_keywords_text = QTextEdit()
        self.initial_keywords_text.setReadOnly(True)
        font = self.initial_keywords_text.font()
        font.setPointSize(11)
        self.initial_keywords_text.setFont(font)
        self.details_tabs.addTab(self.initial_keywords_text, "Initial Keywords")

        # Search Results tab
        self.search_results_table = QTableWidget()
        self.search_results_table.setColumnCount(4)
        self.search_results_table.setHorizontalHeaderLabels(
            ["Suchbegriff", "Keyword", "Count", "GND-ID"]
        )
        self.search_results_table.horizontalHeader().setStretchLastSection(True)
        self.details_tabs.addTab(self.search_results_table, "Such-Ergebnisse")

        # GND Compliant Keywords tab
        self.gnd_keywords_text = QTextEdit()
        self.gnd_keywords_text.setReadOnly(True)
        font = self.gnd_keywords_text.font()
        font.setPointSize(11)
        self.gnd_keywords_text.setFont(font)
        self.details_tabs.addTab(self.gnd_keywords_text, "GND-konforme Keywords")

        # Final Analysis tab
        self.final_analysis_text = QTextEdit()
        self.final_analysis_text.setReadOnly(True)
        font = self.final_analysis_text.font()
        font.setPointSize(11)
        self.final_analysis_text.setFont(font)
        self.details_tabs.addTab(self.final_analysis_text, "Finale Analyse")

        # Statistics tab
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        font = self.stats_text.font()
        font.setPointSize(11)
        self.stats_text.setFont(font)
        self.details_tabs.addTab(self.stats_text, "Statistiken")

    def load_analysis(self):
        """Load analysis from JSON file - Claude Generated (Refactored)"""
        # Use centralized AnalysisPersistence
        state = AnalysisPersistence.load_with_dialog(parent_widget=self)

        if state:  # User selected and loaded successfully
            self.current_analysis = state
            self.populate_analysis_data()
            self.populate_detail_tabs()

            # Enable buttons
            self.export_button.setEnabled(True)
            self.use_keywords_button.setEnabled(True)
            self.use_abstract_button.setEnabled(True)

            self.logger.info("Analysis loaded successfully")

    def populate_analysis_data(self):
        """Populate the UI with analysis data"""
        if not self.current_analysis:
            return

        # Clear existing data
        self.steps_tree.clear()

        # Populate steps tree
        self.populate_steps_tree()

        # Populate detail tabs
        self.populate_detail_tabs()

    def populate_steps_tree(self):
        """Populate the steps tree widget - Claude Generated (Refactored)"""
        if not self.current_analysis:
            return

        # Root item
        root = QTreeWidgetItem(self.steps_tree)
        root.setText(0, "Keyword-Analyse")
        root.setText(1, "Abgeschlossen")

        # Original Abstract
        abstract_item = QTreeWidgetItem(root)
        abstract_item.setText(0, "Original Abstract")
        abstract_len = len(self.current_analysis.original_abstract or "")
        abstract_item.setText(1, f"{abstract_len} Zeichen")
        abstract_item.setData(0, Qt.ItemDataRole.UserRole, "original_abstract")

        # Initial Keywords
        keywords_count = len(self.current_analysis.initial_keywords)
        keywords_item = QTreeWidgetItem(root)
        keywords_item.setText(0, "Initial Keywords")
        keywords_item.setText(1, f"{keywords_count} Keywords")
        keywords_item.setData(0, Qt.ItemDataRole.UserRole, "initial_keywords")

        # Search Results
        search_results_count = len(self.current_analysis.search_results)
        search_item = QTreeWidgetItem(root)
        search_item.setText(0, "Such-Ergebnisse")
        search_item.setText(1, f"{search_results_count} Begriffe")
        search_item.setData(0, Qt.ItemDataRole.UserRole, "search_results")

        # GND Keywords (extract from final_llm_analysis if available)
        gnd_keywords_count = 0
        if self.current_analysis.final_llm_analysis:
            gnd_keywords_count = len(self.current_analysis.final_llm_analysis.extracted_gnd_keywords)
        gnd_item = QTreeWidgetItem(root)
        gnd_item.setText(0, "GND-konforme Keywords")
        gnd_item.setText(1, f"{gnd_keywords_count} Keywords")
        gnd_item.setData(0, Qt.ItemDataRole.UserRole, "gnd_keywords")

        # Final Analysis
        final_item = QTreeWidgetItem(root)
        final_item.setText(0, "Finale LLM-Analyse")
        if self.current_analysis.final_llm_analysis:
            model_used = self.current_analysis.final_llm_analysis.model_used
            final_item.setText(1, f"Model: {model_used}")
        else:
            final_item.setText(1, "Model: N/A")
        final_item.setData(0, Qt.ItemDataRole.UserRole, "final_analysis")

        # Statistics
        stats_item = QTreeWidgetItem(root)
        stats_item.setText(0, "Statistiken")
        stats_item.setText(1, "Zusammenfassung")
        stats_item.setData(0, Qt.ItemDataRole.UserRole, "statistics")

        # Expand all
        self.steps_tree.expandAll()

    def populate_detail_tabs(self):
        """Populate the detail tabs with data - Claude Generated (Refactored)"""
        if not self.current_analysis:
            return

        # Original Abstract
        self.abstract_text.setPlainText(self.current_analysis.original_abstract or "")

        # Initial Keywords
        self.initial_keywords_text.setPlainText("\n".join(self.current_analysis.initial_keywords))

        # Search Results
        self.populate_search_results_table()

        # GND Compliant Keywords (from final_llm_analysis)
        gnd_keywords_list = []
        if self.current_analysis.final_llm_analysis:
            gnd_keywords_list = self.current_analysis.final_llm_analysis.extracted_gnd_keywords
        self.gnd_keywords_text.setPlainText("\n".join(gnd_keywords_list))

        # Final Analysis
        if self.current_analysis.final_llm_analysis:
            llm = self.current_analysis.final_llm_analysis
            final_text = f"Model: {llm.model_used}\n"
            final_text += f"Provider: {llm.provider_used}\n"
            final_text += f"Task: {llm.task_name}\n"
            final_text += f"Temperature: {llm.temperature}\n\n"
            final_text += "Response:\n"
            final_text += llm.response_full_text
            self.final_analysis_text.setPlainText(final_text)
        else:
            self.final_analysis_text.setPlainText("Keine LLM-Analyse verfügbar")

        # Statistics
        self.populate_statistics()

    def populate_search_results_table(self):
        """Populate the search results table - Claude Generated (Refactored)"""
        if not self.current_analysis:
            return

        search_results = self.current_analysis.search_results

        # Count total results
        total_results = 0
        for result in search_results:
            total_results += len(result.results)

        self.search_results_table.setRowCount(total_results)

        row = 0
        for result in search_results:
            search_term = result.search_term

            for keyword, data in result.results.items():
                self.search_results_table.setItem(row, 0, QTableWidgetItem(search_term))
                self.search_results_table.setItem(row, 1, QTableWidgetItem(keyword))
                self.search_results_table.setItem(
                    row, 2, QTableWidgetItem(str(data.get("count", 0)))
                )
                gnd_ids = data.get("gndid", set())
                gnd_id = list(gnd_ids)[0] if gnd_ids else ""
                self.search_results_table.setItem(row, 3, QTableWidgetItem(str(gnd_id)))
                row += 1

        # Resize columns
        self.search_results_table.resizeColumnsToContents()

    def populate_statistics(self):
        """Populate statistics tab - Claude Generated (Refactored)"""
        if not self.current_analysis:
            return

        stats_text = "=== Analyse-Statistiken ===\n\n"

        # Basic stats
        abstract_len = len(self.current_analysis.original_abstract or "")
        stats_text += f"Original Abstract: {abstract_len} Zeichen\n"

        initial_keywords_count = len(self.current_analysis.initial_keywords)
        stats_text += f"Initial Keywords: {initial_keywords_count} Keywords\n"

        search_results = self.current_analysis.search_results
        total_search_results = sum(len(r.results) for r in search_results)
        stats_text += f"Such-Ergebnisse: {total_search_results} Ergebnisse für {len(search_results)} Begriffe\n"

        gnd_keywords_count = 0
        if self.current_analysis.final_llm_analysis:
            gnd_keywords_count = len(self.current_analysis.final_llm_analysis.extracted_gnd_keywords)
        stats_text += f"GND-konforme Keywords: {gnd_keywords_count} Keywords\n"

        # Search suggesters used
        suggesters = self.current_analysis.search_suggesters_used
        stats_text += f"Verwendete Suggester: {', '.join(suggesters)}\n"

        # GND classes
        initial_gnd_classes = self.current_analysis.initial_gnd_classes
        stats_text += f"Initial GND-Klassen: {len(initial_gnd_classes)} Klassen\n"

        # Final analysis info
        if self.current_analysis.final_llm_analysis:
            llm = self.current_analysis.final_llm_analysis
            stats_text += f"\n=== Finale Analyse ===\n"
            stats_text += f"Model: {llm.model_used}\n"
            stats_text += f"Provider: {llm.provider_used}\n"
            stats_text += f"Task: {llm.task_name}\n"
            stats_text += f"Temperature: {llm.temperature}\n"
            stats_text += f"Seed: {llm.seed or 'N/A'}\n"

            response_text = llm.response_full_text
            stats_text += f"Response Länge: {len(response_text)} Zeichen\n"

            extracted_keywords = llm.extracted_gnd_keywords
            stats_text += f"Extrahierte Keywords: {len(extracted_keywords)} Keywords\n"

            extracted_classes = llm.extracted_gnd_classes
            stats_text += f"Extrahierte GND-Klassen: {len(extracted_classes)} Klassen\n"

        self.stats_text.setPlainText(stats_text)

    def on_step_selected(self, item, column):
        """Handle step selection in tree"""
        step_type = item.data(0, Qt.ItemDataRole.UserRole)

        if step_type == "original_abstract":
            self.details_tabs.setCurrentIndex(0)
        elif step_type == "initial_keywords":
            self.details_tabs.setCurrentIndex(1)
        elif step_type == "search_results":
            self.details_tabs.setCurrentIndex(2)
        elif step_type == "gnd_keywords":
            self.details_tabs.setCurrentIndex(3)
        elif step_type == "final_analysis":
            self.details_tabs.setCurrentIndex(4)
        elif step_type == "statistics":
            self.details_tabs.setCurrentIndex(5)

    def export_analysis(self):
        """Export current analysis to JSON - Claude Generated (Refactored)"""
        if not self.current_analysis:
            QMessageBox.warning(
                self, "Warnung", "Keine Analyse zum Exportieren vorhanden."
            )
            return

        # Use centralized AnalysisPersistence
        file_path = AnalysisPersistence.save_with_dialog(
            state=self.current_analysis,
            parent_widget=self,
            default_filename=f"analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if file_path:
            self.logger.info(f"Analysis exported to {file_path}")

    def use_keywords_in_search(self):
        """Use GND compliant keywords in search tab - Claude Generated (Refactored)"""
        if not self.current_analysis:
            return

        # Get keywords from final_llm_analysis
        gnd_keywords_list = []
        if self.current_analysis.final_llm_analysis:
            gnd_keywords_list = self.current_analysis.final_llm_analysis.extracted_gnd_keywords

        keywords_text = ", ".join(gnd_keywords_list)

        self.keywords_selected.emit(keywords_text)
        QMessageBox.information(
            self, "Info", "Keywords wurden an die Suche übertragen."
        )

    def use_abstract_in_analysis(self):
        """Use original abstract in analysis tab - Claude Generated (Refactored)"""
        if not self.current_analysis:
            return

        original_abstract = self.current_analysis.original_abstract or ""

        self.abstract_selected.emit(original_abstract)
        QMessageBox.information(
            self, "Info", "Abstract wurde an die Analyse übertragen."
        )

    # Claude Generated - DELETED: create_analysis_export() and export_current_gui_state()
    # These methods are now obsolete - use AnalysisPersistence.save_with_dialog() instead

    # ==================== Batch Review Mode Methods - Claude Generated ====================

    def setup_batch_table(self):
        """Setup the batch review table widget - Claude Generated"""
        self.batch_table_widget = QWidget()
        batch_layout = QVBoxLayout(self.batch_table_widget)

        # Info label
        info_label = QLabel("📋 Batch-Ergebnisse - Klicken Sie auf eine Zeile, um Details anzuzeigen")
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        batch_layout.addWidget(info_label)

        # Table
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels([
            "Status", "Quelle", "Keywords", "Datum", "Aktionen"
        ])
        self.batch_table.horizontalHeader().setStretchLastSection(False)
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        self.batch_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.batch_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.batch_table.cellDoubleClicked.connect(self.on_batch_row_double_clicked)

        batch_layout.addWidget(self.batch_table)

        # Initially hidden
        self.batch_table_widget.setVisible(False)

    def toggle_batch_mode(self):
        """Toggle between single and batch view - Claude Generated"""
        self.batch_mode = not self.batch_mode

        # Update UI visibility
        self.batch_table_widget.setVisible(self.batch_mode)
        self.main_splitter.setVisible(not self.batch_mode)

        # Update button text
        if self.batch_mode:
            self.batch_toggle_button.setText("📄 Einzelansicht")
            self.batch_toggle_button.setChecked(True)
        else:
            self.batch_toggle_button.setText("📋 Batch-Ansicht")
            self.batch_toggle_button.setChecked(False)

        self.logger.info(f"Batch mode {'enabled' if self.batch_mode else 'disabled'}")

    def load_batch_directory(self, directory: str):
        """Load all JSON files from directory - Claude Generated"""
        from pathlib import Path
        from ..utils.pipeline_utils import PipelineJsonManager

        json_files = list(Path(directory).glob("*.json"))

        # Filter out the .batch_state.json file
        json_files = [f for f in json_files if f.name != ".batch_state.json"]

        if not json_files:
            self.logger.warning(f"No JSON files found in {directory}")
            return

        self.batch_results = []
        for json_file in json_files:
            try:
                state = PipelineJsonManager.load_analysis_state(str(json_file))
                self.batch_results.append((json_file.name, state))
            except Exception as e:
                self.logger.error(f"Failed to load {json_file}: {e}")

        self.populate_batch_table()
        self.logger.info(f"Loaded {len(self.batch_results)} batch results from {directory}")

        # Automatically switch to batch mode
        if not self.batch_mode:
            self.toggle_batch_mode()

    def populate_batch_table(self):
        """Fill batch table with loaded results - Claude Generated"""
        self.batch_table.setRowCount(len(self.batch_results))

        for row, (filename, state) in enumerate(self.batch_results):
            # Status
            status_icon = "✅" if state.final_llm_analysis else "⚠️"
            status_item = QTableWidgetItem(status_icon)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.batch_table.setItem(row, 0, status_item)

            # Source
            source_item = QTableWidgetItem(filename)
            self.batch_table.setItem(row, 1, source_item)

            # Keywords count
            keyword_count = 0
            if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                keyword_count = len(state.final_llm_analysis.extracted_gnd_keywords)
            keyword_item = QTableWidgetItem(str(keyword_count))
            keyword_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.batch_table.setItem(row, 2, keyword_item)

            # Date
            date_str = state.timestamp or ""
            if date_str:
                try:
                    # Format datetime for display
                    from datetime import datetime
                    dt = datetime.fromisoformat(date_str)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            date_item = QTableWidgetItem(date_str)
            date_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.batch_table.setItem(row, 3, date_item)

            # Actions button
            view_btn = QPushButton("View")
            view_btn.clicked.connect(lambda checked, r=row: self.view_batch_result(r))
            self.batch_table.setCellWidget(row, 4, view_btn)

    def on_batch_row_double_clicked(self, row: int, column: int):
        """Handle double-click on batch table row - Claude Generated"""
        self.view_batch_result(row)

    def view_batch_result(self, row: int):
        """View detailed result from batch table - Claude Generated"""
        if row < 0 or row >= len(self.batch_results):
            return

        filename, state = self.batch_results[row]
        self.logger.info(f"Viewing batch result: {filename}")

        # Load the state into current analysis
        self.current_analysis = state

        # Switch to single view
        if self.batch_mode:
            self.toggle_batch_mode()

        # Populate the detail views
        self.populate_analysis_data()
        self.populate_detail_tabs()

        # Enable buttons
        self.export_button.setEnabled(True)
        self.use_keywords_button.setEnabled(True)
        self.use_abstract_button.setEnabled(True)
