from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
    QApplication,
    QProgressBar,
    QSpinBox,
    QFrame,
    QSizePolicy,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QIcon
from typing import Dict, List, Optional
import logging
import re
import sys
import json
from pathlib import Path

from ..utils.suggesters.meta_suggester import MetaSuggester, SuggesterType
from ..core.search_cli import SearchCLI
from ..core.pipeline_manager import PipelineManager, PipelineStep, PipelineConfig
from ..utils.config_models import PipelineStepConfig, PipelineMode
from .workers import PipelineWorker
from .styles import (
    get_main_stylesheet,
    get_button_styles,
    get_status_label_styles,
    get_scaled_font,
    LAYOUT,
    COLORS,
)


class SearchTab(QWidget):
    """
    Tab für die unified GND-Schlagwortsuche
    Unterstützt verschiedene Backends: Lobid, SWB und lokaler Katalog
    """

    # Signale
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    search_completed = pyqtSignal(dict)
    keywords_found = pyqtSignal(str)
    keywords_exact = pyqtSignal(str)
    selection_changed = pyqtSignal(dict)  # Emits modified_selections - Claude Generated

    def __init__(
        self,
        cache_manager,
        parent=None,
        config_file: Path = Path.home() / ".alima_config.json",
        alima_manager=None,
        pipeline_manager=None,
    ):
        super().__init__(parent)
        self.cache_manager = cache_manager
        self.alima_manager = alima_manager  # Add AlimaManager for PipelineManager - Claude Generated
        self.current_results = None
        self.current_gnd_id = None
        self.logger = logging.getLogger(__name__)
        self.result_list = []
        self.katalog_keywords = []
        self.gnd_ids = []  # Liste der gefundenen GND-IDs
        self.unkown_terms = []

        self.catalog_token = ""
        self.catalog_search_url = ""
        self.catalog_details = ""

        # Pipeline integration state tracking - Claude Generated
        self.original_pipeline_state = None
        self.current_display_state = None
        self.modified_selections = {}  # {gnd_id: 'selected'/'deselected'}
        self.manual_additions = []  # List of manually added GND entries
        self.has_unsaved_changes = False
        self.cache_status = {}  # Cache for GND entry status (cache/new/outdated)

        # Use injected central PipelineManager instead of creating redundant instance - Claude Generated
        self.pipeline_manager = pipeline_manager
        if not self.pipeline_manager:
            self.logger.warning("No PipelineManager provided - pipeline integration disabled")

        # Lade den Katalog-Token aus der Konfigurationsdatei
        self.config_file = config_file
        self._load_catalog_token()

        self.init_ui()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche des Such-Tabs"""
        # Use main stylesheet
        self.setStyleSheet(get_main_stylesheet())
        btn_styles = get_button_styles()

        # Hauptlayout mit Abständen
        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["spacing"])
        layout.setContentsMargins(
            LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"], LAYOUT["margin"]
        )

        # ========= Kontrolleiste oben (wie bei AbstractTab) =========
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(0, 0, 0, 5)

        # Status-Label
        self.status_label = QLabel("Aktueller Status: Bereit")
        self.status_label.setStyleSheet(get_status_label_styles()["info"])
        control_bar.addWidget(self.status_label)

        control_bar.addStretch(1)

        # Fortschrittsanzeige
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        self.progressBar.setTextVisible(True)
        self.progressBar.setFormat("Verarbeite... %p%")
        self.progressBar.setFixedWidth(200)
        control_bar.addWidget(self.progressBar)

        layout.addLayout(control_bar)

        # ========= Transparency Section (Collapsible) =========
        self.transparency_group = QGroupBox("📋 Pipeline-Mapping")
        self.transparency_group.setCheckable(True)
        self.transparency_group.setChecked(False)  # Collapsed by default
        transparency_layout = QVBoxLayout(self.transparency_group)
        transparency_layout.setContentsMargins(10, 10, 10, 10)
        transparency_layout.setSpacing(5)

        self.transparency_text = QTextEdit()
        self.transparency_text.setReadOnly(True)
        self.transparency_text.setMaximumHeight(250)
        self.transparency_text.setPlaceholderText(
            "Mapping-Details werden nach Pipeline-Ausführung hier angezeigt..."
        )
        transparency_layout.addWidget(self.transparency_text)

        #layout.addWidget(self.transparency_group) # TODO -> bevor restoring, make it functional

        # Splitter between search input and results
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # ========= Suchbereich =========
        search_widget = QWidget()
        search_box_layout = QVBoxLayout(search_widget)
        search_box_layout.setContentsMargins(0, 0, 0, 0)
        search_box_layout.setSpacing(LAYOUT["spacing"])

        search_group = QGroupBox("Schlagwortsuche")
        search_layout = QVBoxLayout(search_group)
        search_layout.setSpacing(LAYOUT["inner_spacing"])
        search_layout.setContentsMargins(10, 20, 10, 10)

        # Hauptsuchfeld mit Beschreibung
        search_header = QLabel("Suchbegriffe:")
        search_header.setFont(get_scaled_font(bold=True))
        search_layout.addWidget(search_header)

        self.search_input = QTextEdit()
        self.search_input.setPlaceholderText(
            "Suchbegriffe (durch Komma getrennt oder in Anführungszeichen für exakte Phrasen)"
        )
        self.search_input.setMaximumHeight(100)
        self.search_input.setFont(get_scaled_font(size_delta=+1))
        search_layout.addWidget(self.search_input)

        # Suchoptionen-Bereich
        options_frame = QFrame()
        options_frame.setStyleSheet(
            f"background-color: {COLORS['background_dark']}; border-radius: 6px; padding: 4px;"
        )
        options_layout = QHBoxLayout(options_frame)

        # Checkboxen für die verschiedenen Suchquellen
        sources_label = QLabel("Suchquellen:")
        sources_label.setFont(get_scaled_font(size_delta=-1, bold=True))
        options_layout.addWidget(sources_label)

        self.lobid_button = QCheckBox("Lobid")
        self.lobid_button.setChecked(True)
        self.lobid_button.setToolTip("Suche in der Lobid-API (empfohlen)")
        options_layout.addWidget(self.lobid_button)

        self.swb_button = QCheckBox("SWB")
        self.swb_button.setChecked(True)
        self.swb_button.setToolTip("Suche im Südwestdeutschen Bibliotheksverbund")
        options_layout.addWidget(self.swb_button)

        self.catalog_button = QCheckBox("Katalog")
        self.catalog_button.setChecked(False)
        self.catalog_button.setToolTip("Suche im lokalen Katalog (falls verfügbar)")
        if self.catalog_token != "":
            options_layout.addWidget(self.catalog_button)

        options_layout.addStretch(1)

        # Ergebnisanzahl-Steuerung
        results_label = QLabel("Max. Ergebnisse:")
        results_label.setFont(get_scaled_font(size_delta=-1, bold=True))
        #options_layout.addWidget(results_label)

        self.num_results = QSpinBox()
        self.num_results.setRange(1, 50)
        self.num_results.setValue(10)
        self.num_results.setToolTip(
            "Maximale Anzahl zu verarbeitender Ergebnisse pro Quelle"
        )
        #options_layout.addWidget(self.num_results)

        search_layout.addWidget(options_frame)

        # Suchbutton
        self.search_button = QPushButton("Suche starten")
        self.search_button.setStyleSheet(btn_styles["primary"])
        self.search_button.clicked.connect(self.perform_search)
        self.search_button.setShortcut("Ctrl+Return")
        search_layout.addWidget(self.search_button)

        search_box_layout.addWidget(search_group)
        main_splitter.addWidget(search_widget)

        # ========= Ergebnisbereich =========
        results_container = QWidget()
        results_container_layout = QVBoxLayout(results_container)
        results_container_layout.setContentsMargins(0, 0, 0, 0)
        results_container_layout.setSpacing(LAYOUT["spacing"])

        # Obere Sektion: Ergebnistabelle und Details
        results_group = QGroupBox("Suchergebnisse") # TODO -> Table doesn't show the correct field (Begriff -> N/A, GND-ID -> name of keyword ) examine why bevor solving
        results_box_layout = QVBoxLayout(results_group)
        results_box_layout.setSpacing(LAYOUT["inner_spacing"])
        results_box_layout.setContentsMargins(10, 20, 10, 10)

        upper_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Ergebnistabelle
        table_frame = QWidget()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_header = QLabel("Gefundene Schlagwörter:")
        table_header.setFont(get_scaled_font(bold=True))
        table_layout.addWidget(table_header)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Begriff", "GND-ID", "Häufigkeit", "Ähnlichkeit", "Status"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)

        table_layout.addWidget(self.results_table)
        upper_splitter.addWidget(table_frame)

        # Details-Sektion
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_header = QLabel("GND-Details:")
        details_header.setFont(get_scaled_font(bold=True))
        details_layout.addWidget(details_header)

        self.details_display = QTextEdit()
        self.details_display.setReadOnly(True)
        details_layout.addWidget(self.details_display)

        self.update_button = QPushButton("Eintrag aktualisieren")
        self.update_button.setStyleSheet(btn_styles["secondary"])
        self.update_button.clicked.connect(self.update_selected_entry)
        details_layout.addWidget(self.update_button)

        upper_splitter.addWidget(details_widget)
        upper_splitter.setSizes([600, 400])

        results_box_layout.addWidget(upper_splitter)

        # ========= Manual Search Panel (Collapsible) - Claude Generated =========
        self.manual_search_group = QGroupBox("➕ Manuelle Nachsuche")
        self.manual_search_group.setCheckable(False)
        self.manual_search_group.setVisible(False)  # Hidden by default
        manual_layout = QHBoxLayout(self.manual_search_group)
        manual_layout.setContentsMargins(10, 10, 10, 10)

        manual_layout.addWidget(QLabel("Zusätzlicher Suchbegriff:"))
        self.manual_search_input = QTextEdit()
        self.manual_search_input.setPlaceholderText("Begriff für manuelle Suche eingeben...")
        self.manual_search_input.setMaximumHeight(60)
        manual_layout.addWidget(self.manual_search_input)

        self.manual_search_button = QPushButton("Suchen")
        self.manual_search_button.setStyleSheet(btn_styles["primary"])
        self.manual_search_button.clicked.connect(self.perform_manual_search)
        manual_layout.addWidget(self.manual_search_button)

        results_box_layout.addWidget(self.manual_search_group)

        # ========= Action Buttons - Claude Generated =========
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)

        self.toggle_manual_button = QPushButton("🔧 Manuelle Nachsuche")
        self.toggle_manual_button.setCheckable(True)
        self.toggle_manual_button.setStyleSheet(btn_styles["secondary"])
        self.toggle_manual_button.setToolTip(
            "Aktiviert manuelle Suche für zusätzliche GND-Schlagwörter, "
            "die die Pipeline nicht gefunden hat"
        )
        self.toggle_manual_button.toggled.connect(self.manual_search_group.setVisible)
        actions_layout.addWidget(self.toggle_manual_button)

        actions_layout.addStretch(1)

        # Keep existing update button, renamed for clarity
        self.update_button.setText("🔄 DNB-Sync Selected")
        self.update_button.setToolTip(
            "Aktualisiert alle ausgewählten (✅) GND-Einträge mit aktuellen Daten von der DNB. "
            "Holt DDC-Klassifikationen und GND-Systematiken."
        )
        actions_layout.addWidget(self.update_button)

        self.save_changes_button = QPushButton("💾 Änderungen Speichern")
        self.save_changes_button.setStyleSheet(btn_styles["accent"])
        self.save_changes_button.setEnabled(False)
        self.save_changes_button.setToolTip(
            "Speichert Ihre Änderungen an der GND-Auswahl zurück in den Analyse-Status. "
            "Änderungen werden dann in anderen Tabs sichtbar."
        )
        self.save_changes_button.clicked.connect(self.save_changes)
        actions_layout.addWidget(self.save_changes_button)

        results_box_layout.addLayout(actions_layout)

        results_container_layout.addWidget(results_group)

        main_splitter.addWidget(results_container)
        main_splitter.setSizes([300, 700])

        layout.addWidget(main_splitter)

        # Verbinde Signals
        self.results_table.itemSelectionChanged.connect(self.show_details)
        self.results_table.itemDoubleClicked.connect(self.on_result_double_clicked)  # Claude Generated

    def refresh_styles(self):
        """Re-apply styles after theme change — Claude Generated"""
        from .styles import get_main_stylesheet, get_status_label_styles
        self.setStyleSheet(get_main_stylesheet())
        if hasattr(self, 'status_label'):
            self.status_label.setStyleSheet(get_status_label_styles()["info"])

    def _load_catalog_token(self):
        """
        Lädt den Katalog-Token aus der Konfigurationsdatei.

        Returns:
            str: Katalog-Token oder einen leeren String, wenn nicht gefunden
        """
        default_token = ""  # Fallback-Token wenn nicht gefunden

        try:
            if not self.config_file.exists():
                self.logger.debug(
                    f"Konfigurationsdatei nicht gefunden: {self.config_file}"
                )
                return default_token

            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Versuche, den Katalog-Token aus verschiedenen möglichen Stellen zu laden
            if "catalog_token" in config:
                self.catalog_token = config["catalog_token"]

            if "catalog_search_url" in config:
                self.catalog_search_url = config["catalog_search_url"]

            if "catalog_details" in config:
                self.catalog_details = config["catalog_details"]

        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Katalog-Tokens: {str(e)}")
            return default_token

    def perform_search(self):
        """Führt die Suche mit den ausgewählten Quellen durch - Claude Generated"""
        self.logger.info("=== perform_search() called ===")

        # UI-Updates vor der Suche
        self.search_button.setEnabled(False)
        self.status_label.setText("Suche wird durchgeführt...")
        self.results_table.setRowCount(0)  # Tabelle leeren
        self.result_list.clear()
        self.gnd_ids.clear()
        self.details_display.clear()
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        QApplication.processEvents()

        try:
            # Eingabetext verarbeiten
            text = self.search_input.toPlainText().strip()
            self.logger.info(f"Search input text: '{text}'")

            # Keine Suchbegriffe vorhanden
            if not text:
                self.status_label.setText(
                    "Bitte geben Sie mindestens einen Suchbegriff ein."
                )
                self.status_label.setStyleSheet(get_status_label_styles()["warning"])
                self.search_button.setEnabled(True)
                self.progressBar.setVisible(False)
                return

            # Extrahiere Suchbegriffe
            search_terms = self.extract_search_terms(text)

            # Keine gültigen Suchbegriffe
            if not search_terms:
                self.status_label.setText("Keine gültigen Suchbegriffe gefunden.")
                self.status_label.setStyleSheet(get_status_label_styles()["warning"])
                self.search_button.setEnabled(True)
                self.progressBar.setVisible(False)
                return

            # Bestimme die zu verwendenden Suggester-Typen
            suggester_types = []
            if self.lobid_button.isChecked():
                suggester_types.append(SuggesterType.LOBID)
            if self.swb_button.isChecked():
                suggester_types.append(SuggesterType.SWB)
            if self.catalog_button.isChecked():
                suggester_types.append(SuggesterType.CATALOG)

            # Wenn keine Quelle ausgewählt wurde, Lobid als Standard verwenden
            if not suggester_types:
                self.logger.warning(
                    "Keine Suchquelle ausgewählt, verwende Lobid als Standard."
                )
                suggester_types.append(SuggesterType.LOBID)

            self.logger.info(f"Selected suggester types: {suggester_types}")
            self.logger.info(f"Search terms: {search_terms}")

            # Direct suggester usage for standalone search - Claude Generated
            # This is simpler and faster than using the full pipeline
            self.progressBar.setMaximum(100)
            self.progressBar.setValue(10)

            # Determine single suggester type
            # MetaSuggester expects a single SuggesterType, not a list
            # IMPORTANT: Don't use ALL to avoid triggering catalog DK lookups outside pipeline
            if len(suggester_types) == 1:
                selected_type = suggester_types[0]
            elif SuggesterType.LOBID in suggester_types and SuggesterType.SWB in suggester_types:
                # Both Lobid and SWB: Use Lobid as primary, manually merge SWB below
                selected_type = SuggesterType.LOBID
                self.use_swb_fallback = True
            elif SuggesterType.LOBID in suggester_types:
                selected_type = SuggesterType.LOBID
                self.use_swb_fallback = False
            elif SuggesterType.SWB in suggester_types:
                selected_type = SuggesterType.SWB
                self.use_swb_fallback = False
            else:
                selected_type = SuggesterType.LOBID  # Fallback
                self.use_swb_fallback = False

            self.logger.info(f"Using suggester type: {selected_type}, SWB fallback: {self.use_swb_fallback}")

            # Create MetaSuggester with selected type
            self.logger.info("Creating MetaSuggester...")
            meta_suggester = MetaSuggester(
                suggester_type=selected_type
            )
            self.logger.info("MetaSuggester created successfully")

            # Perform search (search() expects a list of terms)
            self.status_label.setText(f"Suche nach {len(search_terms)} Begriff(en)...")
            QApplication.processEvents()

            self.progressBar.setValue(30)
            self.logger.info("Starting search...")
            combined_results = meta_suggester.search(search_terms)
            self.logger.info(f"Search completed, got {len(combined_results)} results")

            # If both Lobid and SWB selected, merge SWB results
            if self.use_swb_fallback and SuggesterType.SWB in suggester_types:
                self.progressBar.setValue(50)
                self.logger.info("Adding SWB results...")
                swb_suggester = MetaSuggester(suggester_type=SuggesterType.SWB)
                swb_results = swb_suggester.search(search_terms)

                # Merge results
                for term, term_results in swb_results.items():
                    if term not in combined_results:
                        combined_results[term] = {}
                    combined_results[term].update(term_results)

                self.logger.info(f"After SWB merge: {len(combined_results)} results")

            self.progressBar.setValue(80)

            # Process and display results
            self.process_search_results(combined_results)

        except Exception as e:
            self.logger.error(f"Search error: {str(e)}", exc_info=True)
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.handle_error(str(e))
        finally:
            self.search_button.setEnabled(True)
            self.progressBar.setVisible(False)

    def on_search_completed(self, step: PipelineStep):
        """Handle search completion with PipelineStep integration - Claude Generated"""
        try:
            # Extract search results from PipelineStep output_data
            if step.output_data and hasattr(step.output_data, 'search_results'):
                results = step.output_data.search_results
            elif step.output_data and isinstance(step.output_data, dict):
                results = step.output_data.get('search_results', step.output_data)
            else:
                self.logger.warning(f"No search results in step output: {step.output_data}")
                results = {}

            # Process results using existing method
            self.process_search_results(results)

        except Exception as e:
            self.logger.error(f"Error processing search results: {e}")
            self.handle_error(str(e))

    def on_search_error(self, step: PipelineStep, error_message: str):
        """Handle search error with PipelineStep integration - Claude Generated"""
        error_details = f"Suchfehler bei Schritt {step.step_id}: {error_message}"
        if step.error_message:
            error_details += f"\nZusätzliche Informationen: {step.error_message}"

        self.logger.error(error_details)
        self.handle_error(error_details)

    def process_search_results(self, results: dict):
        """Processes the search results from the CLI and displays them."""
        self.process_results(results)

        # UI-Updates nach der Suche
        self.search_button.setEnabled(True)
        self.status_label.setText(
            f"Suche abgeschlossen - {len(self.flat_results)} Ergebnisse gefunden"
        )
        self.status_label.setStyleSheet(get_status_label_styles()["success"])
        self.progressBar.setVisible(False)
        QApplication.processEvents()

    def handle_error(self, error_message: str):
        """Handles errors from the search worker."""
        self.logger.error(f"Search error: {error_message}")
        self.status_label.setText(f"Fehler: {error_message}")
        self.status_label.setStyleSheet(get_status_label_styles()["error"])
        self.search_button.setEnabled(True)
        self.progressBar.setVisible(False)

    def current_term_update(self, term):
        """Aktualisiert die Fortschrittsanzeige bei Verarbeitung eines Terms"""
        if not self.progressBar.isVisible():
            self.progressBar.setVisible(True)

        self.progressBar.setValue(self.progressBar.value() + 1)
        self.status_label.setText(f"Verarbeite: {term}")
        QApplication.processEvents()  # UI aktualisieren

    def extract_search_terms(self, text):
        """Extrahiert Suchbegriffe aus dem eingegebenen Text"""
        # Extrahiere Begriffe, die in Anführungszeichen stehen
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, text)

        # Entferne die extrahierten Begriffe aus dem ursprünglichen Text
        remaining_text = re.sub(quoted_pattern, "", text)

        # Teile den verbleibenden Text nach Kommas auf
        remaining_terms = [
            term.strip() for term in remaining_text.split(",") if term.strip()
        ]

        # Kombiniere beide Listen
        search_terms = quoted_matches + remaining_terms

        return search_terms

    def merge_results(self, combined_results, new_results):
        """Führt neue Ergebnisse mit den bereits vorhandenen zusammen"""
        for search_term, term_results in new_results.items():
            # Initialisiere den Eintrag für diesen Suchterm, falls noch nicht vorhanden
            if search_term not in combined_results:
                combined_results[search_term] = {}

            # Füge für jedes Schlagwort die Daten hinzu oder aktualisiere sie
            for keyword, data in term_results.items():
                if keyword not in combined_results[search_term]:
                    # Neues Schlagwort hinzufügen
                    combined_results[search_term][keyword] = data.copy()
                else:
                    # Bestehendes Schlagwort aktualisieren
                    existing_data = combined_results[search_term][keyword]

                    # Count aktualisieren (Maximum verwenden)
                    existing_data["count"] = max(
                        existing_data["count"], data.get("count", 0)
                    )

                    # Sets vereinigen
                    existing_data["gndid"].update(data.get("gndid", set()))
                    existing_data["ddc"].update(data.get("ddc", set()))
                    existing_data["dk"].update(data.get("dk", set()))

    def process_results(self, results):
        """Verarbeitet die Suchergebnisse und stellt sie dar"""
        self.logger.info(f"Verarbeite Ergebnisse: {len(results)} Suchbegriffe gefunden")

        # Initialize flat_results if not exists
        if not hasattr(self, "flat_results"):
            self.flat_results = []
        else:
            self.flat_results.clear()

        for search_term, term_results in results.items():
            self.logger.info(f"Verarbeite Suchbegriff: {search_term}, Anzahl Ergebnisse: {len(term_results) if term_results else 0}")

            # Handle case where term_results is a list instead of dict
            if isinstance(term_results, list):
                self.logger.info(
                    f"term_results is a list with {len(term_results)} items"
                )
                for i, item in enumerate(term_results[:5]):  # Debug first 5 items
                    self.logger.info(f"Item {i}: {item}")
                    if isinstance(item, dict):
                        # Extract keyword and data from the item
                        keyword = item.get("label", item.get("title", ""))
                        gnd_id = item.get("gnd_id", item.get("gndid", ""))
                        count = item.get("count", 1)

                        self.logger.info(
                            f"Extracted: keyword='{keyword}', gnd_id='{gnd_id}', count={count}"
                        )

                        if keyword:
                            self.logger.info(f"Verarbeite Schlagwort: {keyword}")
                            # Bestimme die Beziehung zum Suchbegriff
                            relation = self.determine_relation(keyword, search_term)

                            # Speichere die GND-ID für spätere Verwendung
                            if gnd_id:
                                self.gnd_ids.append(gnd_id)
                            else:
                                self.logger.warning(f"No GND-ID found for keyword: {keyword}")
                                self.unkown_terms.append(keyword)
                                # Skip entries without GND-ID
                                continue

                            # Füge zur flachen Liste hinzu
                            if gnd_id:
                                self.flat_results.append(
                                    (gnd_id, keyword, count, relation, search_term)
                                )
                            else:
                                # Wenn keine GND-ID vorhanden ist, aber ein Schlagwort gefunden wurde
                                # Füge es trotzdem zur Liste hinzu
                                self.logger.debug(
                                    f"Keine GND-ID gefunden für: {keyword}"
                                )

            else:
                # Original dict handling code
                for keyword, data in term_results.items():
                    self.logger.debug(f"Verarbeite Schlagwort: {keyword}, Data: {data}")
                    # Bestimme die Beziehung zum Suchbegriff
                    relation = self.determine_relation(keyword, search_term)

                    # Ermittle GND-ID (erste aus dem Set oder leer)
                    gnd_id = next(iter(data.get("gndid", [])), "")
                    self.logger.debug(f"  Extracted GND-ID: {gnd_id}")

                    # Anzahl der Treffer
                    count = data.get("count", 1)

                    # Speichere die GND-ID für spätere Verwendung
                    if gnd_id:
                        self.gnd_ids.append(gnd_id)
                    else:
                        self.logger.warning(f"No GND-ID found for keyword: {keyword}")
                        self.unkown_terms.append(keyword)
                        # Continue anyway, the code below will handle missing GND-ID

                    # Füge zur flachen Liste hinzu
                    if gnd_id:
                        self.flat_results.append(
                            (gnd_id, keyword, count, relation, search_term)
                        )
                    else:
                        # Wenn keine GND-ID vorhanden ist, aber ein Schlagwort gefunden wurde
                        # Füge es trotzdem zur Liste hinzu
                        self.logger.debug(f"Keine GND-ID gefunden für: {keyword}")

        # Sortiere Ergebnisse nach Relation und dann nach Count
        sorted_results = sorted(self.flat_results, key=lambda x: (x[3], -x[2]))

        self.logger.info(f"Gefundene GND-Einträge: {len(sorted_results)}")

        # Zeige Ergebnisse in der Tabelle an
        self.display_results(sorted_results)

        # Generiere Initial-Prompt
        self.generate_initial_prompt(sorted_results)

    def determine_relation(self, keyword: str, search_term: str) -> int:
        """Determine relationship between keyword and search term - Claude Generated

        Returns:
            0: Exact match
            1: Similar/partial match
            2: Different/no match
        """
        keyword_lower = keyword.lower()
        search_lower = search_term.lower()

        if keyword_lower == search_lower:
            return 0  # Exakt
        elif search_lower in keyword_lower or keyword_lower in search_lower:
            return 1  # Ähnlich
        else:
            return 2  # Unterschiedlich

    def generate_initial_prompt(self, sorted_results):
        """Generate initial prompt from results - Stub for compatibility - Claude Generated
        Pipeline now handles prompt generation.
        """
        pass

    def update_database_entry(self, gnd_id, title, data):
        """Aktualisiert oder erstellt einen Datenbankeintrag für eine GND-ID"""
        if not self.cache_manager.gnd_entry_exists(gnd_id):
            # Lege neuen Eintrag an
            self.cache_manager.insert_gnd_entry(gnd_id, title=title)

    def display_results(self, sorted_results):
        """Zeigt die Suchergebnisse in der Tabelle an"""
        self.results_table.setRowCount(0)  # Tabelle leeren

        # Relation-Symbole
        relation_symbols = ["=", "≈", "≠"]
        relation_colors = [QColor("#4caf50"), QColor("#ff9800"), QColor("#9e9e9e")]

        for gnd_id, term, count, relation, search_term in sorted_results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            # Füge Daten in die Tabelle ein
            term_item = QTableWidgetItem(term)
            gnd_item = QTableWidgetItem(gnd_id)
            count_item = QTableWidgetItem(str(count))
            rel_item = QTableWidgetItem(relation_symbols[relation])

            # Setze Farben basierend auf Relation
            rel_item.setForeground(relation_colors[relation])
            rel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Füge Tooltip für Kontextinformationen hinzu
            term_item.setToolTip(f"Suchbegriff: {search_term}")
            gnd_item.setToolTip("Klicken für Details")

            # Setze Items in die Tabelle
            self.results_table.setItem(row, 0, term_item)
            self.results_table.setItem(row, 1, gnd_item)
            self.results_table.setItem(row, 2, count_item)
            self.results_table.setItem(row, 3, rel_item)

    def update_selected_entry(self):
        """Batch DNB sync for all selected/displayed entries - Claude Generated"""
        from .workers import DNBSyncWorker

        # Collect GND-IDs: either selected entries or all displayed entries with ✅
        selected_gnd_ids = []

        for row in range(self.results_table.rowCount()):
            begriff_item = self.results_table.item(row, 0)
            if begriff_item and begriff_item.text().startswith("✅"):
                gnd_id = self.results_table.item(row, 1).text()
                selected_gnd_ids.append(gnd_id)

        if not selected_gnd_ids:
            self.status_label.setText("Keine ausgewählten Einträge zum Synchronisieren")
            self.status_label.setStyleSheet(get_status_label_styles()["warning"])
            return

        # Disable button and show progress
        self.update_button.setEnabled(False)
        self.update_button.setText(f"Synchronisiere {len(selected_gnd_ids)} Einträge...")
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(100)

        self.status_label.setText(f"Starte DNB-Sync für {len(selected_gnd_ids)} Einträge...")
        self.status_label.setStyleSheet(get_status_label_styles()["info"])
        QApplication.processEvents()

        # Start worker thread
        self.sync_worker = DNBSyncWorker(selected_gnd_ids, self.cache_manager)
        self.sync_worker.progress.connect(self.progressBar.setValue)
        self.sync_worker.finished.connect(self.on_sync_finished)
        self.sync_worker.start()

    def on_sync_finished(self, success_count, error_count):
        """Handle DNB sync completion - Claude Generated"""
        total = success_count + error_count

        # Re-enable button
        self.update_button.setEnabled(True)
        self.update_button.setText("🔄 DNB-Sync Selected")
        self.progressBar.setVisible(False)

        # Update status
        if error_count == 0:
            self.status_label.setText(f"DNB-Sync erfolgreich: {success_count} Einträge aktualisiert")
            self.status_label.setStyleSheet(get_status_label_styles()["success"])
        else:
            self.status_label.setText(
                f"DNB-Sync abgeschlossen: {success_count} erfolgreich, {error_count} Fehler"
            )
            self.status_label.setStyleSheet(get_status_label_styles()["warning"])

        # Refresh details if an entry is currently selected
        if self.results_table.selectedItems():
            self.show_details()

        self.logger.info(f"DNB sync completed: {success_count}/{total} successful")

    def update_entry(self, gnd_id: str):
        """Aktualisiert einen GND-Eintrag mit Daten aus der DNB"""
        from ..core.dnb_utils import get_dnb_classification

        try:
            # Status aktualisieren
            self.status_label.setText(f"Aktualisiere GND-Eintrag: {gnd_id}")
            self.status_label.setStyleSheet(get_status_label_styles()["info"])

            # Fortschritt anzeigen
            if self.progressBar.isVisible():
                self.progressBar.setValue(20)

            QApplication.processEvents()

            # Hole DNB-Klassifikation
            dnb_class = get_dnb_classification(gnd_id)

            # Fortschritt anzeigen
            if self.progressBar.isVisible():
                self.progressBar.setValue(60)

            if dnb_class and dnb_class.get("status") == "success":
                # Extrahiere Daten
                term = dnb_class.get("preferred_name", "")

                # DDCs extrahieren und formatieren
                ddc_list = dnb_class.get("ddc", [])
                ddc = ";".join(f"{d['code']}({d['determinancy']})" for d in ddc_list)

                # GND-Kategorien extrahieren
                gnd_category = dnb_class.get("gnd_subject_categories", [])
                gnd_category = ";".join(gnd_category)

                # Allgemeine Kategorie
                category = dnb_class.get("category", "")

                # Aktualisiere den Eintrag in der Datenbank
                self.cache_manager.update_gnd_entry(
                    gnd_id,
                    title=term,
                    ddcs=ddc,
                    gnd_systems=gnd_category,
                    classification=category,
                )

                # Status aktualisieren
                self.status_label.setText(f"GND-Eintrag aktualisiert: {gnd_id}")
                self.status_label.setStyleSheet(get_status_label_styles()["success"])

            else:
                # Fehlermeldung
                error_msg = (
                    dnb_class.get("error_message", "Keine Daten erhalten")
                    if dnb_class
                    else "Keine Daten erhalten"
                )
                self.logger.error(f"Fehler bei GND {gnd_id}: {error_msg}")
                self.status_label.setText(f"Fehler bei GND {gnd_id}: {error_msg}")
                self.status_label.setStyleSheet(get_status_label_styles()["error"])

            # Fortschritt anzeigen
            if self.progressBar.isVisible():
                self.progressBar.setValue(100)

        except Exception as e:
            self.logger.error(f"Fehler bei Update von GND {gnd_id}: {str(e)}")
            self.error_occurred.emit(f"Fehler bei Update von GND {gnd_id}: {str(e)}")

    def show_details(self):
        """Zeigt Details für den ausgewählten Eintrag in der Detailansicht"""
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            return

        # Hole Daten aus der ausgewählten Zeile
        row = selected_items[0].row()
        gnd_id = self.results_table.item(row, 1).text()

        # Speichere die aktuelle GND-ID für spätere Verwendung
        self.current_gnd_id = gnd_id

        if gnd_id and self.cache_manager.gnd_entry_exists(gnd_id):
            # Hole Eintrag aus der Datenbank
            gnd_entry = self.cache_manager.get_gnd_entry_by_id(gnd_id)

            # Formatierter Text mit HTML-Styling
            details = f"""<html>
            <body style='font-family: Arial, sans-serif;'>
            <h3>Details für GND-ID: {gnd_id}</h3>
            <p><b>Titel:</b> {gnd_entry.get('title', 'N/A')}</p>
            """

            # Beschreibung hinzufügen, wenn vorhanden
            if gnd_entry.get("description"):
                details += f"<p><b>Beschreibung:</b> {gnd_entry.get('description')}</p>"

            # DDC-Klassifikationen mit Formatierung anzeigen
            if gnd_entry.get("ddcs"):
                details += "<p><b>DDC-Klassifikationen:</b><ul>"
                for ddc in gnd_entry.get("ddcs", "").split(";"):
                    if ddc.strip():
                        details += f"<li>{ddc}</li>"
                details += "</ul></p>"
            else:
                details += "<p><b>DDC:</b> Keine Klassifikation verfügbar</p>"

            # DK-Klassifikationen
            if gnd_entry.get("dks"):
                details += "<p><b>DK-Klassifikationen:</b><ul>"
                for dk in gnd_entry.get("dks", "").split(";"):
                    if dk.strip():
                        details += f"<li>{dk}</li>"
                details += "</ul></p>"

            # Synonyme
            if gnd_entry.get("synonyms"):
                details += "<p><b>Synonyme:</b><ul>"
                for syn in gnd_entry.get("synonyms", "").split(";"):
                    if syn.strip():
                        details += f"<li>{syn}</li>"
                details += "</ul></p>"

            # Metadaten für Datum
            details += f"""
            <p style='font-size: 0.9em; color: #666;'>
            <b>Erstellt am:</b> {gnd_entry.get('created_at', 'N/A')}<br>
            <b>Zuletzt aktualisiert:</b> {gnd_entry.get('updated_at', 'N/A')}
            </p>
            """

            details += "</body></html>"

            # Zeige Details in der Detailansicht
            self.details_display.setHtml(details)
        else:
            self.details_display.setHtml(
                f"<html><body><h3>Keine Details verfügbar für GND-ID: {gnd_id}</h3></body></html>"
            )

    def update_search_field(self, keywords):
        """Aktualisiert das Suchfeld mit den gegebenen Schlüsselwörtern"""
        self.search_input.setText(keywords)

    def display_search_results(self, results: Dict) -> None:
        """
        Display search results from pipeline - Claude Generated
        This method allows the SearchTab to act as a viewer for pipeline results

        Args:
            results: Dictionary of search results from pipeline search step
        """
        self.logger.info(f"Displaying pipeline search results: {len(results)} terms")

        # Clear existing results and UI state
        self.results_table.setRowCount(0)
        self.result_list.clear()
        self.gnd_ids.clear()
        self.details_display.clear()

        # Process and display the results using existing logic
        self.process_results(results)

        # Update status
        self.status_label.setText(
            f"Pipeline-Ergebnisse angezeigt - {len(getattr(self, 'flat_results', []))} Ergebnisse"
        )
        self.status_label.setStyleSheet(get_status_label_styles()["success"])

        self.logger.info("Pipeline search results displayed successfully")

    @pyqtSlot(object)
    def update_data(self, analysis_state):
        """Receive pipeline results and display with transparency - Claude Generated

        This slot is called when the pipeline completes, transforming the SearchTab
        into a post-processing view that shows:
        1. Which init-keywords led to which GND entries (mapping transparency)
        2. Which GND entries were selected in final_keywords (usage transparency)
        3. Which entries came from cache vs. newly fetched (cache status)

        Args:
            analysis_state: KeywordAnalysisState object with pipeline results
        """
        if not analysis_state:
            self.logger.warning("update_data called with empty analysis_state")
            self.status_label.setText("Keine Pipeline-Ergebnisse verfügbar")
            self.status_label.setStyleSheet(get_status_label_styles()["warning"])
            self.transparency_text.setHtml(
                "<p style='color: orange;'><b>Keine Pipeline-Daten geladen</b></p>"
                "<p>Führen Sie die Pipeline aus, um GND-Suchergebnisse zu sehen.</p>"
            )
            return

        try:
            # Store original state for tracking modifications
            self.original_pipeline_state = analysis_state
            self.current_display_state = analysis_state
            self.modified_selections = {}
            self.manual_additions = []
            self.has_unsaved_changes = False

            # Extract data from analysis_state
            initial_keywords = analysis_state.initial_keywords or []
            search_results = analysis_state.search_results or []
            final_keywords = []
            if analysis_state.final_llm_analysis:
                final_keywords = analysis_state.final_llm_analysis.extracted_gnd_keywords or []

            self.logger.info(
                f"Processing pipeline data: {len(initial_keywords)} init keywords, "
                f"{len(search_results)} search results, {len(final_keywords)} final keywords"
            )

            # Handle empty search results
            if not search_results:
                self.status_label.setText("Pipeline hat keine GND-Einträge gefunden")
                self.status_label.setStyleSheet(get_status_label_styles()["warning"])
                self.transparency_text.setHtml(
                    "<p style='color: orange;'><b>Keine Suchergebnisse</b></p>"
                    "<p>Die Pipeline hat keine GND-Schlagwörter gefunden. "
                    "Verwenden Sie die manuelle Nachsuche unten.</p>"
                )
                self.results_table.setRowCount(0)
                return

            # Query cache status for all GND entries
            self._query_cache_status(search_results)

            # Build mapping table (init-keyword → GND entries)
            self._build_mapping_table(initial_keywords, search_results, final_keywords)

            # Display results with transparency overlays
            self._display_pipeline_results(search_results, final_keywords)

            # Update status with helpful hint about double-click
            total_entries = sum(len(sr.results) for sr in search_results)
            self.status_label.setText(
                f"✅ Pipeline-Ergebnisse: {total_entries} GND-Einträge "
                f"({len(final_keywords)} ausgewählt) | 💡 Doppelklick zum Umschalten"
            )
            self.status_label.setStyleSheet(get_status_label_styles()["success"])

            self.logger.info("Pipeline results successfully loaded in SearchTab")

        except Exception as e:
            self.logger.error(f"Error in update_data: {e}", exc_info=True)
            self.status_label.setText(f"Fehler beim Laden der Pipeline-Ergebnisse: {str(e)}")
            self.status_label.setStyleSheet(get_status_label_styles()["error"])

    def _query_cache_status(self, search_results: List) -> None:
        """Query cache status for each GND entry - Claude Generated

        Populates self.cache_status with GND-ID → status mapping:
        - 'cache': Entry exists in DB and is recent (< 90 days)
        - 'outdated': Entry exists but is old (>= 90 days)
        - 'new': Entry was newly fetched (not in cache or no timestamp)

        Args:
            search_results: List of SearchResult objects from pipeline
        """
        from datetime import datetime, timedelta

        self.cache_status = {}

        for search_result in search_results:
            for gnd_id in search_result.results.keys():
                try:
                    # Check if entry exists in cache
                    entry = self.cache_manager.get_gnd_entry_by_id(gnd_id)

                    if entry and entry.get('updated_at'):
                        # Parse timestamp and check age
                        try:
                            updated_str = entry.get('updated_at')
                            # Handle different date formats
                            if 'T' in updated_str:
                                updated_dt = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
                            else:
                                updated_dt = datetime.strptime(updated_str, '%Y-%m-%d %H:%M:%S')

                            age_days = (datetime.now() - updated_dt.replace(tzinfo=None)).days

                            if age_days >= 90:
                                self.cache_status[gnd_id] = 'outdated'
                            else:
                                self.cache_status[gnd_id] = 'cache'
                        except (ValueError, AttributeError) as e:
                            self.logger.debug(f"Could not parse date for {gnd_id}: {e}")
                            self.cache_status[gnd_id] = 'cache'  # Assume cache if exists
                    elif entry:
                        # Entry exists but no timestamp
                        self.cache_status[gnd_id] = 'cache'
                    else:
                        # No entry found - newly fetched
                        self.cache_status[gnd_id] = 'new'

                except Exception as e:
                    self.logger.debug(f"Error checking cache for {gnd_id}: {e}")
                    self.cache_status[gnd_id] = 'new'

        self.logger.info(f"Cache status queried for {len(self.cache_status)} entries")

    def _build_mapping_table(self, initial_keywords: List[str], search_results: List,
                             final_keywords: List[str]) -> None:
        """Build mapping from init-keywords to GND entries - Claude Generated

        Creates HTML representation showing which pipeline keywords led to which
        GND entries, and which were ultimately selected by the LLM.

        Args:
            initial_keywords: List of initial keywords from pipeline
            search_results: List of SearchResult objects
            final_keywords: List of GND-IDs selected in final analysis
        """
        mapping_text = "<h3>Pipeline-Mapping (Initial Keywords → GND-Einträge)</h3>"
        mapping_text += "<p style='color: gray; font-size: 0.9em;'>"
        mapping_text += "Zeigt welche Pipeline-Keywords zu welchen GND-Einträgen führten:</p>"

        if not initial_keywords:
            mapping_text += "<p style='color: orange;'><b>Keine Initial-Keywords gefunden</b></p>"
            # Store for transparency section (to be added in Phase 4)
            self.mapping_html = mapping_text
            return

        for init_kw in initial_keywords:
            # Find search results for this keyword
            matching_results = [sr for sr in search_results if sr.search_term == init_kw]

            if not matching_results:
                continue

            mapping_text += f"<p style='margin-top: 12px;'><b>🔍 \"{init_kw}\"</b>:</p><ul style='margin-left: 20px;'>"

            for sr in matching_results:
                for gnd_id, entry_data in sr.results.items():
                    label = entry_data.get('label', 'N/A')
                    is_used = gnd_id in final_keywords

                    if is_used:
                        status = "✅ <span style='color: green; font-weight: bold;'>[Verwendet]</span>"
                    else:
                        status = "<span style='color: gray;'>[Nicht verwendet]</span>"

                    mapping_text += f"<li>{label} (GND:{gnd_id}) {status}</li>"

            mapping_text += "</ul>"

        # Display in transparency section - Claude Generated
        self.transparency_text.setHtml(mapping_text)
        self.logger.debug("Mapping table built and displayed successfully")

    def _display_pipeline_results(self, search_results: List, final_keywords: List[str]) -> None:
        """Display search results with transparency indicators - Claude Generated

        Shows results table with columns:
        - Checkbox (✅ for selected, ☐ for available)
        - Begriff (term name)
        - GND-ID
        - Init-Kw (which initial keyword found this)
        - Status (cache indicator: 💾/🌐/⚠️)
        - Häufigkeit (frequency count)

        Selected entries get light green background highlighting.

        Args:
            search_results: List of SearchResult objects from pipeline
            final_keywords: List of GND-IDs selected in final_keywords
        """
        self.results_table.setRowCount(0)

        # Status icons
        status_icons = {
            'cache': '💾',
            'new': '🌐',
            'outdated': '⚠️'
        }

        status_tooltips = {
            'cache': 'Aus Datenbank-Cache',
            'new': 'Neu geholt von Lobid/SWB',
            'outdated': 'Cache älter als 90 Tage'
        }

        for search_result in search_results:
            init_keyword = search_result.search_term

            for gnd_id, entry_data in search_result.results.items():
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)

                # Determine if selected in final_keywords
                is_selected = gnd_id in final_keywords

                # Column 0: Begriff
                begriff = entry_data.get('label', entry_data.get('title', 'N/A'))
                begriff_item = QTableWidgetItem(begriff)
                if is_selected:
                    begriff_item.setFont(get_scaled_font(bold=True))
                    begriff_item.setForeground(QColor("#2e7d32"))  # Dark green
                    begriff_item.setText(f"✅ {begriff}")
                self.results_table.setItem(row, 0, begriff_item)

                # Column 1: GND-ID
                gnd_item = QTableWidgetItem(gnd_id)
                self.results_table.setItem(row, 1, gnd_item)

                # Column 2: Häufigkeit
                count = entry_data.get('count', 0)
                count_item = QTableWidgetItem(str(count))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, 2, count_item)

                # Column 3: Ähnlichkeit (relation - keep existing column for compatibility)
                relation = self.determine_relation(begriff, init_keyword)
                relation_symbols = ["=", "≈", "≠"]
                relation_colors = [QColor("#4caf50"), QColor("#ff9800"), QColor("#9e9e9e")]
                rel_item = QTableWidgetItem(relation_symbols[relation])
                rel_item.setForeground(relation_colors[relation])
                rel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # Add init-keyword as tooltip for transparency
                init_kw_display = init_keyword[:30] + "..." if len(init_keyword) > 30 else init_keyword
                rel_item.setToolTip(f"Initial-Keyword: {init_keyword}")
                self.results_table.setItem(row, 3, rel_item)

                # Column 4: Status (cache indicator - transparency feature)
                status = self.cache_status.get(gnd_id, 'new')
                status_item = QTableWidgetItem(status_icons[status])
                status_item.setToolTip(status_tooltips[status])
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(row, 4, status_item)

                # Row highlighting for selected entries
                if is_selected:
                    for col in range(5):
                        item = self.results_table.item(row, col)
                        if item:
                            item.setBackground(QColor("#e8f5e9"))  # Light green

        self.logger.info(f"Displayed {self.results_table.rowCount()} pipeline results")

    def perform_manual_search(self):
        """Perform manual GND search and add results to display - Claude Generated

        Allows expert users to search for additional GND keywords that the pipeline
        might have missed. Results are added with a special 🔧 indicator.
        """
        search_term = self.manual_search_input.toPlainText().strip()
        if not search_term:
            self.status_label.setText("Bitte einen Suchbegriff eingeben")
            self.status_label.setStyleSheet(get_status_label_styles()["warning"])
            return

        try:
            self.status_label.setText(f"Manuelle Suche: {search_term}")
            self.status_label.setStyleSheet(get_status_label_styles()["info"])
            self.manual_search_button.setEnabled(False)
            QApplication.processEvents()

            # Direct suggester usage - simpler than pipeline
            search_terms = self.extract_search_terms(search_term)

            # Use Lobid + SWB for manual searches (NOT Catalog to avoid DK lookups)
            lobid_suggester = MetaSuggester(suggester_type=SuggesterType.LOBID)
            all_results = lobid_suggester.search(search_terms)

            # Also search SWB and merge
            swb_suggester = MetaSuggester(suggester_type=SuggesterType.SWB)
            swb_results = swb_suggester.search(search_terms)

            # Merge SWB results into Lobid results
            for term, term_results in swb_results.items():
                if term not in all_results:
                    all_results[term] = {}
                all_results[term].update(term_results)

            added_count = 0
            for term, results in all_results.items():
                if results:
                    # Add results to display with manual indicator
                    for keyword, data in results.items():
                        # Extract GND-ID
                        gnd_ids = data.get('gndid', set())
                        gnd_id = next(iter(gnd_ids), None)

                        if not gnd_id:
                            continue

                        # Check if already in display
                        already_exists = False
                        for row in range(self.results_table.rowCount()):
                            existing_id = self.results_table.item(row, 1).text()
                            if existing_id == gnd_id:
                                already_exists = True
                                break

                        if not already_exists:
                            # Add to table with manual indicator
                            row = self.results_table.rowCount()
                            self.results_table.insertRow(row)

                            begriff_item = QTableWidgetItem(f"🔧 {keyword}")
                            begriff_item.setForeground(QColor("#1976d2"))  # Blue for manual
                            self.results_table.setItem(row, 0, begriff_item)

                            gnd_item = QTableWidgetItem(gnd_id)
                            self.results_table.setItem(row, 1, gnd_item)

                            count_item = QTableWidgetItem(str(data.get('count', 0)))
                            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                            self.results_table.setItem(row, 2, count_item)

                            rel_item = QTableWidgetItem("🔧")
                            rel_item.setToolTip("Manuell hinzugefügt")
                            rel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                            self.results_table.setItem(row, 3, rel_item)

                            status_item = QTableWidgetItem("🆕")
                            status_item.setToolTip("Manuelle Addition")
                            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                            self.results_table.setItem(row, 4, status_item)

                            # Highlight row
                            for col in range(5):
                                item = self.results_table.item(row, col)
                                if item:
                                    item.setBackground(QColor("#e3f2fd"))  # Light blue

                            # Track as manual addition
                            self.manual_additions.append({
                                'gnd_id': gnd_id,
                                'label': keyword,
                                'search_term': term
                            })
                            added_count += 1

            if added_count > 0:
                self.status_label.setText(f"Manuelle Suche: {added_count} neue Einträge hinzugefügt")
                self.status_label.setStyleSheet(get_status_label_styles()["success"])
                self.has_unsaved_changes = True
                self.save_changes_button.setEnabled(True)
            else:
                self.status_label.setText("Manuelle Suche: Keine neuen Einträge gefunden")
                self.status_label.setStyleSheet(get_status_label_styles()["info"])

            self.manual_search_input.clear()

        except Exception as e:
            self.logger.error(f"Manual search error: {e}", exc_info=True)
            self.status_label.setText(f"Fehler bei manueller Suche: {str(e)}")
            self.status_label.setStyleSheet(get_status_label_styles()["error"])
        finally:
            self.manual_search_button.setEnabled(True)

    def save_changes(self):
        """Save modified selections back to analysis state - Claude Generated

        Emits selection_changed signal with:
        - modified: Dict of GND-IDs with 'selected'/'deselected' status changes
        - manual: List of manually added entries
        """
        if not (self.modified_selections or self.manual_additions):
            return

        # Emit signal to MainWindow
        self.selection_changed.emit({
            'modified': self.modified_selections,
            'manual': self.manual_additions
        })

        self.status_label.setText(
            f"Änderungen gespeichert: {len(self.modified_selections)} geändert, "
            f"{len(self.manual_additions)} manuell hinzugefügt"
        )
        self.status_label.setStyleSheet(get_status_label_styles()["success"])
        self.has_unsaved_changes = False
        self.save_changes_button.setEnabled(False)

        self.logger.info(
            f"GND selection changes saved: {len(self.modified_selections)} modified, "
            f"{len(self.manual_additions)} manual additions"
        )

    def on_result_double_clicked(self, item):
        """Toggle selection status when row is double-clicked - Claude Generated

        Allows users to manually adjust which GND entries should be included
        in the final keywords. Tracks changes for saving back to analysis state.
        """
        if not self.original_pipeline_state:
            # No pipeline data loaded, ignore
            return

        row = item.row()
        gnd_id = self.results_table.item(row, 1).text()
        begriff_item = self.results_table.item(row, 0)
        begriff_text = begriff_item.text()

        # Get original selection status
        final_keywords = []
        if self.original_pipeline_state.final_llm_analysis:
            final_keywords = self.original_pipeline_state.final_llm_analysis.extracted_gnd_keywords or []

        was_originally_selected = gnd_id in final_keywords

        # Determine current display status
        is_currently_selected = begriff_text.startswith("✅")

        # Toggle selection
        if is_currently_selected:
            # Deselect
            new_text = begriff_text.replace("✅ ", "")
            begriff_item.setText(new_text)
            begriff_item.setFont(get_scaled_font())
            begriff_item.setForeground(QColor("#000000"))  # Black

            # Remove highlighting
            for col in range(5):
                cell_item = self.results_table.item(row, col)
                if cell_item:
                    # Check if manual addition (light blue)
                    if cell_item.background().color() == QColor("#e3f2fd"):
                        continue  # Keep manual addition highlighting
                    cell_item.setBackground(QColor("#ffffff"))  # White

            # Track modification
            if was_originally_selected:
                self.modified_selections[gnd_id] = 'deselected'
            else:
                # Was not originally selected, and we're deselecting - remove from modifications
                if gnd_id in self.modified_selections:
                    del self.modified_selections[gnd_id]

        else:
            # Select
            if not begriff_text.startswith("✅"):
                new_text = f"✅ {begriff_text.replace('🔧 ', '')}"  # Remove manual indicator if present
                begriff_item.setText(new_text)
            begriff_item.setFont(get_scaled_font(bold=True))
            begriff_item.setForeground(QColor("#2e7d32"))  # Dark green

            # Add highlighting
            for col in range(5):
                cell_item = self.results_table.item(row, col)
                if cell_item:
                    cell_item.setBackground(QColor("#e8f5e9"))  # Light green

            # Track modification
            if not was_originally_selected:
                self.modified_selections[gnd_id] = 'selected'
            else:
                # Was originally selected, and we're selecting again - remove from modifications
                if gnd_id in self.modified_selections:
                    del self.modified_selections[gnd_id]

        # Update save button state
        if self.modified_selections or self.manual_additions:
            self.has_unsaved_changes = True
            self.save_changes_button.setEnabled(True)
        else:
            self.has_unsaved_changes = False
            self.save_changes_button.setEnabled(False)

        self.logger.debug(f"Toggled selection for {gnd_id}: {self.modified_selections.get(gnd_id, 'no change')}")
