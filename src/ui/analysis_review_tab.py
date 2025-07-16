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
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional


class AnalysisReviewTab(QWidget):
    """Tab for reviewing and exporting analysis results"""

    # Signals
    keywords_selected = pyqtSignal(str)
    abstract_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_analysis = None
        self.setup_ui()

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
        """Load analysis from JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Analyse laden", "", "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.current_analysis = json.load(f)

            self.populate_analysis_data()
            self.export_button.setEnabled(True)
            self.use_keywords_button.setEnabled(True)
            self.use_abstract_button.setEnabled(True)

            self.logger.info(f"Analysis loaded from {file_path}")

        except Exception as e:
            QMessageBox.critical(
                self, "Fehler", f"Fehler beim Laden der Analyse:\n{str(e)}"
            )
            self.logger.error(f"Error loading analysis: {e}")

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
        """Populate the steps tree widget"""
        if not self.current_analysis:
            return

        # Root item
        root = QTreeWidgetItem(self.steps_tree)
        root.setText(0, "Keyword-Analyse")
        root.setText(1, "Abgeschlossen")

        # Original Abstract
        abstract_item = QTreeWidgetItem(root)
        abstract_item.setText(0, "Original Abstract")
        abstract_item.setText(
            1, f"{len(self.current_analysis.get('original_abstract', ''))} Zeichen"
        )
        abstract_item.setData(0, Qt.ItemDataRole.UserRole, "original_abstract")

        # Initial Keywords
        initial_keywords = self.current_analysis.get("initial_keywords", [])
        keywords_item = QTreeWidgetItem(root)
        keywords_item.setText(0, "Initial Keywords")
        keywords_item.setText(1, f"{len(initial_keywords)} Keywords")
        keywords_item.setData(0, Qt.ItemDataRole.UserRole, "initial_keywords")

        # Search Results
        search_results = self.current_analysis.get("search_results", [])
        search_item = QTreeWidgetItem(root)
        search_item.setText(0, "Such-Ergebnisse")
        search_item.setText(1, f"{len(search_results)} Begriffe")
        search_item.setData(0, Qt.ItemDataRole.UserRole, "search_results")

        # GND Compliant Keywords
        gnd_keywords = self.current_analysis.get("gnd_compliant_keywords", [])
        gnd_item = QTreeWidgetItem(root)
        gnd_item.setText(0, "GND-konforme Keywords")
        gnd_item.setText(1, f"{len(gnd_keywords)} Keywords")
        gnd_item.setData(0, Qt.ItemDataRole.UserRole, "gnd_keywords")

        # Final Analysis
        final_analysis = self.current_analysis.get("final_llm_analysis", {})
        final_item = QTreeWidgetItem(root)
        final_item.setText(0, "Finale LLM-Analyse")
        final_item.setText(1, f"Model: {final_analysis.get('model_used', 'N/A')}")
        final_item.setData(0, Qt.ItemDataRole.UserRole, "final_analysis")

        # Statistics
        stats_item = QTreeWidgetItem(root)
        stats_item.setText(0, "Statistiken")
        stats_item.setText(1, "Zusammenfassung")
        stats_item.setData(0, Qt.ItemDataRole.UserRole, "statistics")

        # Expand all
        self.steps_tree.expandAll()

    def populate_detail_tabs(self):
        """Populate the detail tabs with data"""
        if not self.current_analysis:
            return

        # Original Abstract
        original_abstract = self.current_analysis.get("original_abstract", "")
        self.abstract_text.setPlainText(original_abstract)

        # Initial Keywords
        initial_keywords = self.current_analysis.get("initial_keywords", [])
        self.initial_keywords_text.setPlainText("\n".join(initial_keywords))

        # Search Results
        self.populate_search_results_table()

        # GND Compliant Keywords
        gnd_keywords = self.current_analysis.get("gnd_compliant_keywords", [])
        gnd_text = "\n".join(
            [f"{kw.get('keyword', '')} ({kw.get('gnd_id', '')})" for kw in gnd_keywords]
        )
        self.gnd_keywords_text.setPlainText(gnd_text)

        # Final Analysis
        final_analysis = self.current_analysis.get("final_llm_analysis", {})
        final_text = f"Model: {final_analysis.get('model_used', 'N/A')}\n"
        final_text += f"Provider: {final_analysis.get('provider_used', 'N/A')}\n"
        final_text += f"Task: {final_analysis.get('task_name', 'N/A')}\n\n"
        final_text += "Response:\n"
        final_text += final_analysis.get("response_full_text", "")
        self.final_analysis_text.setPlainText(final_text)

        # Statistics
        self.populate_statistics()

    def populate_search_results_table(self):
        """Populate the search results table"""
        search_results = self.current_analysis.get("search_results", [])

        # Count total results
        total_results = 0
        for result in search_results:
            total_results += len(result.get("results", {}))

        self.search_results_table.setRowCount(total_results)

        row = 0
        for result in search_results:
            search_term = result.get("search_term", "")
            results = result.get("results", {})

            for keyword, data in results.items():
                self.search_results_table.setItem(row, 0, QTableWidgetItem(search_term))
                self.search_results_table.setItem(row, 1, QTableWidgetItem(keyword))
                self.search_results_table.setItem(
                    row, 2, QTableWidgetItem(str(data.get("count", 0)))
                )
                gnd_ids = data.get("gndid", [])
                gnd_id = gnd_ids[0] if gnd_ids else ""
                self.search_results_table.setItem(row, 3, QTableWidgetItem(gnd_id))
                row += 1

        # Resize columns
        self.search_results_table.resizeColumnsToContents()

    def populate_statistics(self):
        """Populate statistics tab"""
        if not self.current_analysis:
            return

        stats_text = "=== Analyse-Statistiken ===\n\n"

        # Basic stats
        original_abstract = self.current_analysis.get("original_abstract", "")
        stats_text += f"Original Abstract: {len(original_abstract)} Zeichen\n"

        initial_keywords = self.current_analysis.get("initial_keywords", [])
        stats_text += f"Initial Keywords: {len(initial_keywords)} Keywords\n"

        search_results = self.current_analysis.get("search_results", [])
        total_search_results = sum(len(r.get("results", {})) for r in search_results)
        stats_text += f"Such-Ergebnisse: {total_search_results} Ergebnisse für {len(search_results)} Begriffe\n"

        gnd_keywords = self.current_analysis.get("gnd_compliant_keywords", [])
        stats_text += f"GND-konforme Keywords: {len(gnd_keywords)} Keywords\n"

        # Search suggesters used
        suggesters = self.current_analysis.get("search_suggesters_used", [])
        stats_text += f"Verwendete Suggester: {', '.join(suggesters)}\n"

        # GND classes
        initial_gnd_classes = self.current_analysis.get("initial_gnd_classes", [])
        stats_text += f"Initial GND-Klassen: {len(initial_gnd_classes)} Klassen\n"

        # Final analysis info
        final_analysis = self.current_analysis.get("final_llm_analysis", {})
        if final_analysis:
            stats_text += f"\n=== Finale Analyse ===\n"
            stats_text += f"Model: {final_analysis.get('model_used', 'N/A')}\n"
            stats_text += f"Provider: {final_analysis.get('provider_used', 'N/A')}\n"
            stats_text += f"Task: {final_analysis.get('task_name', 'N/A')}\n"
            stats_text += f"Temperature: {final_analysis.get('temperature', 'N/A')}\n"
            stats_text += f"Seed: {final_analysis.get('seed', 'N/A')}\n"

            response_text = final_analysis.get("response_full_text", "")
            stats_text += f"Response Länge: {len(response_text)} Zeichen\n"

            extracted_keywords = final_analysis.get("extracted_gnd_keywords", [])
            stats_text += f"Extrahierte Keywords: {len(extracted_keywords)} Keywords\n"

            extracted_classes = final_analysis.get("extracted_gnd_classes", [])
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
        """Export current analysis to JSON"""
        if not self.current_analysis:
            QMessageBox.warning(
                self, "Warnung", "Keine Analyse zum Exportieren vorhanden."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Analyse exportieren",
            f"analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)",
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.current_analysis, f, indent=2, ensure_ascii=False)

            QMessageBox.information(
                self, "Erfolg", f"Analyse erfolgreich exportiert nach:\n{file_path}"
            )
            self.logger.info(f"Analysis exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Exportieren:\n{str(e)}")
            self.logger.error(f"Error exporting analysis: {e}")

    def use_keywords_in_search(self):
        """Use GND compliant keywords in search tab"""
        if not self.current_analysis:
            return

        gnd_keywords = self.current_analysis.get("gnd_compliant_keywords", [])
        keywords_text = ", ".join(
            [f"{kw.get('keyword', '')} ({kw.get('gnd_id', '')})" for kw in gnd_keywords]
        )

        self.keywords_selected.emit(keywords_text)
        QMessageBox.information(
            self, "Info", "Keywords wurden an die Suche übertragen."
        )

    def use_abstract_in_analysis(self):
        """Use original abstract in analysis tab"""
        if not self.current_analysis:
            return

        original_abstract = self.current_analysis.get("original_abstract", "")

        self.abstract_selected.emit(original_abstract)
        QMessageBox.information(
            self, "Info", "Abstract wurde an die Analyse übertragen."
        )

    def create_analysis_export(
        self,
        abstract: str,
        keywords: str,
        search_results: dict,
        final_keywords: str,
        gnd_classes: str,
    ) -> dict:
        """Create analysis export from current GUI state"""
        export_data = {
            "original_abstract": abstract,
            "initial_keywords": keywords.split(", ") if keywords else [],
            "search_suggesters_used": ["gui"],
            "initial_gnd_classes": gnd_classes.split("|") if gnd_classes else [],
            "search_results": [],
            "gnd_compliant_keywords": [],
            "export_timestamp": datetime.now().isoformat(),
            "export_source": "gui",
        }

        # Process search results if provided
        if search_results:
            for term, results in search_results.items():
                if isinstance(results, dict):
                    search_result = {"search_term": term, "results": results}
                    export_data["search_results"].append(search_result)

        # Process final keywords
        if final_keywords:
            for keyword_line in final_keywords.split("\n"):
                if "(" in keyword_line and ")" in keyword_line:
                    keyword = keyword_line.split("(")[0].strip()
                    gnd_id = keyword_line.split("(")[1].split(")")[0].strip()
                    export_data["gnd_compliant_keywords"].append(
                        {"keyword": keyword, "gnd_id": gnd_id}
                    )

        return export_data

    def export_current_gui_state(
        self,
        abstract: str,
        keywords: str,
        search_results: dict,
        final_keywords: str,
        gnd_classes: str = "",
    ):
        """Export current GUI state as analysis JSON"""
        export_data = self.create_analysis_export(
            abstract, keywords, search_results, final_keywords, gnd_classes
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "GUI-Zustand exportieren",
            f"gui_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)",
        )

        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            QMessageBox.information(
                self, "Erfolg", f"GUI-Zustand erfolgreich exportiert nach:\n{file_path}"
            )
            self.logger.info(f"GUI state exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Exportieren:\n{str(e)}")
            self.logger.error(f"Error exporting GUI state: {e}")
