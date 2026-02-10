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
    QTabWidget,
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
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
    LAYOUT,
    COLORS,
)


class GNDSystemFilterWidget(QGroupBox):
    """
    Widget zum Filtern nach GND-Systematiken.
    Erlaubt das manuelle Hinzufügen/Entfernen von GND-Systematiken für die Filterung.
    """

    def __init__(self, parent=None, cache_manager=None):
        super().__init__("GND-Systematik-Filter", parent)
        self.cache_manager = cache_manager
        self.selected_systems = set()  # Aktuell ausgewählte GND-Systematiken

        self.init_ui()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche des GND-Systematik-Filters"""
        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["inner_spacing"])
        layout.setContentsMargins(10, 20, 10, 10)

        # Suchbereich für GND-Systematiken
        search_layout = QHBoxLayout()

        self.system_input = QTextEdit()
        self.system_input.setPlaceholderText(
            "GND-Systematik eingeben (z.B. '7.12' für Musik)"
        )
        self.system_input.setMaximumHeight(60)
        search_layout.addWidget(self.system_input)

        buttons_layout = QVBoxLayout()
        btn_styles = get_button_styles()

        self.add_button = QPushButton("Hinzufügen")
        self.add_button.setStyleSheet(btn_styles["secondary"])
        self.add_button.clicked.connect(self.add_system)
        buttons_layout.addWidget(self.add_button)

        self.clear_button = QPushButton("Zurücksetzen")
        self.clear_button.setStyleSheet(btn_styles["secondary"])
        self.clear_button.clicked.connect(self.clear_systems)
        buttons_layout.addWidget(self.clear_button)

        search_layout.addLayout(buttons_layout)
        layout.addLayout(search_layout)

        # Tabelle für ausgewählte GND-Systematiken
        self.systems_table = QTableWidget()
        self.systems_table.setColumnCount(3)
        self.systems_table.setHorizontalHeaderLabels(
            ["GND-Systematik", "Beschreibung", "Aktion"]
        )
        self.systems_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.systems_table.setAlternatingRowColors(True)
        self.systems_table.verticalHeader().setVisible(False)

        layout.addWidget(self.systems_table)

        # Knopf zur Anwendung der Filter
        self.apply_button = QPushButton("Filter anwenden")
        self.apply_button.setStyleSheet(btn_styles["primary"])
        self.apply_button.clicked.connect(self.apply_filter)
        layout.addWidget(self.apply_button)

    def add_system(self, system=None):
        """Fügt eine GND-Systematik zur Filterliste hinzu"""
        if system is None or isinstance(system, bool):
            system = self.system_input.toPlainText().strip()

        if not system:
            return

        # Prüfe, ob diese Systematik bereits hinzugefügt wurde
        if system in self.selected_systems:
            return

        # Füge zur Menge hinzu
        self.selected_systems.add(system)

        # Füge zur Tabelle hinzu
        row = self.systems_table.rowCount()
        self.systems_table.insertRow(row)

        # Erstelle Items für die Tabelle
        system_item = QTableWidgetItem(system)
        self.systems_table.setItem(row, 0, system_item)

        # Beschreibung (könnte aus einer Datenbank kommen)
        description = "empty"
        desc_item = QTableWidgetItem(description)
        self.systems_table.setItem(row, 1, desc_item)

        # Lösch-Button
        delete_btn = QPushButton("Entfernen")
        delete_btn.setStyleSheet(get_button_styles()["error"])
        delete_btn.clicked.connect(lambda _, r=row, s=system: self.remove_system(r, s))
        self.systems_table.setCellWidget(row, 2, delete_btn)

        # Leere das Eingabefeld
        self.system_input.clear()

    def remove_system(self, row, system):
        """Entfernt eine GND-Systematik aus der Filterliste"""
        self.systems_table.removeRow(row)
        self.selected_systems.remove(system)

    def clear_systems(self):
        """Entfernt alle GND-Systematiken aus der Filterliste"""
        self.systems_table.setRowCount(0)
        self.selected_systems.clear()

    def apply_filter(self):
        """Wendet die GND-Systematik-Filter an und generiert eine gefilterte Liste von Schlagwörtern"""
        # Implementierung der Filterlogik hier
        if not self.selected_systems:
            return

        # Diese Methode sollte die Hauptsuche informieren, nach welchen GND-Systematiken gefiltert werden soll
        # Signal könnte hier emittiert werden
        pass

    def get_selected_systems(self):
        """Gibt die aktuell ausgewählten GND-Systematiken zurück"""
        return list(self.selected_systems)


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
        search_header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        search_layout.addWidget(search_header)

        self.search_input = QTextEdit()
        self.search_input.setPlaceholderText(
            "Suchbegriffe (durch Komma getrennt oder in Anführungszeichen für exakte Phrasen)"
        )
        self.search_input.setMaximumHeight(100)
        self.search_input.setFont(QFont("Segoe UI", LAYOUT["input_font_size"]))
        search_layout.addWidget(self.search_input)

        # Suchoptionen-Bereich
        options_frame = QFrame()
        options_frame.setStyleSheet(
            f"background-color: {COLORS['background_dark']}; border-radius: 6px; padding: 4px;"
        )
        options_layout = QHBoxLayout(options_frame)

        # Checkboxen für die verschiedenen Suchquellen
        sources_label = QLabel("Suchquellen:")
        sources_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
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
        results_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        options_layout.addWidget(results_label)

        self.num_results = QSpinBox()
        self.num_results.setRange(1, 50)
        self.num_results.setValue(10)
        self.num_results.setToolTip(
            "Maximale Anzahl zu verarbeitender Ergebnisse pro Quelle"
        )
        options_layout.addWidget(self.num_results)

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
        results_group = QGroupBox("Suchergebnisse")
        results_box_layout = QVBoxLayout(results_group)
        results_box_layout.setSpacing(LAYOUT["inner_spacing"])
        results_box_layout.setContentsMargins(10, 20, 10, 10)

        upper_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Ergebnistabelle
        table_frame = QWidget()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_header = QLabel("Gefundene Schlagwörter:")
        table_header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        table_layout.addWidget(table_header)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Begriff", "GND-ID", "Häufigkeit", "Ähnlichkeit", "GND-Systematik"]
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
        details_header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
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

        # ========= Filter-Bereich mit Tabs =========
        filter_group = QGroupBox("Filter- und Generierungsoptionen")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(LAYOUT["inner_spacing"])
        filter_layout.setContentsMargins(10, 20, 10, 10)

        self.filter_tabs = QTabWidget()

        # DDC-Filter Tab
        ddc_widget = QWidget()
        ddc_layout = QVBoxLayout(ddc_widget)
        ddc_layout.setSpacing(LAYOUT["inner_spacing"])
        ddc_layout.setContentsMargins(10, 10, 10, 10)

        ddc_frame = QFrame()
        ddc_frame.setStyleSheet(
            f"background-color: {COLORS['background_dark']}; border-radius: 6px; padding: 4px;"
        )
        ddc_grid = QGridLayout(ddc_frame)
        ddc_grid.setSpacing(8)

        ddc_label = QLabel("Einzubeziehende DDC-Klassen:")
        ddc_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        ddc_grid.addWidget(ddc_label, 0, 0, 1, 5)

        # DDC Checkboxes
        self.ddc_checks = []
        ddc_labels = [
            "1xx (Phil.)", "2xx (Rel.)", "3xx (Sozial.)", "4xx (Spr.)", "5xx (Nat.)",
            "6xx (Tech.)", "7xx (Kunst)", "8xx (Lit.)", "9xx (Gesch.)", "X (Sonstige)"
        ]
        for i, label in enumerate(ddc_labels):
            check = QCheckBox(label)
            check.setChecked(True)
            self.ddc_checks.append(check)
            ddc_grid.addWidget(check, 1 if i < 5 else 2, i % 5)

        # Compatibility names
        self.ddc1_check, self.ddc2_check, self.ddc3_check, self.ddc4_check, self.ddc5_check = self.ddc_checks[:5]
        self.ddc6_check, self.ddc7_check, self.ddc8_check, self.ddc9_check, self.ddcX_check = self.ddc_checks[5:]

        ddc_layout.addWidget(ddc_frame)

        self.ddc_regenerate_button = QPushButton("DDC-gefilterte Schlagwörter")
        self.ddc_regenerate_button.setStyleSheet(btn_styles["accent"])
        ddc_layout.addWidget(self.ddc_regenerate_button)

        self.filter_tabs.addTab(ddc_widget, "DDC-Filter")

        # GND-Systematik Tab
        self.gnd_filter_widget = GNDSystemFilterWidget(self, self.cache_manager)
        self.filter_tabs.addTab(self.gnd_filter_widget, "GND-Systematik")

        filter_layout.addWidget(self.filter_tabs)

        self.regenerate_button = QPushButton("Gefilterte Schlagwörter generieren")
        self.regenerate_button.setStyleSheet(btn_styles["accent"])
        filter_layout.addWidget(self.regenerate_button)

        results_box_layout.addWidget(filter_group)
        results_container_layout.addWidget(results_group)

        main_splitter.addWidget(results_container)
        main_splitter.setSizes([300, 700])

        layout.addWidget(main_splitter)

        # Verbinde Signals
        self.results_table.itemSelectionChanged.connect(self.show_details)

    def _load_catalog_token(self):
        """
        Lädt den Katalog-Token aus der Konfigurationsdatei.

        Returns:
            str: Katalog-Token oder einen leeren String, wenn nicht gefunden
        """
        default_token = ""  # Fallback-Token wenn nicht gefunden

        try:
            if not self.config_file.exists():
                self.logger.warning(
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

            # Use PipelineManager integration if available - Claude Generated
            if self.pipeline_manager:
                # 1. Create ad-hoc PipelineConfig for search step only
                adhoc_config = PipelineConfig()
                adhoc_config.auto_advance = False  # Important: We want only the search step

                # 2. Configure search suggesters (convert enum to string)
                suggester_names = []
                for sug_type in suggester_types:
                    if sug_type == SuggesterType.LOBID:
                        suggester_names.append("lobid")
                    elif sug_type == SuggesterType.SWB:
                        suggester_names.append("swb")
                    elif sug_type == SuggesterType.CATALOG:
                        suggester_names.append("catalog")

                adhoc_config.search_suggesters = suggester_names

                # 3. Prepare input as comma-separated keywords
                input_text = ", ".join(search_terms)

                # 4. Use centralized PipelineWorker with PipelineManager
                self.pipeline_worker = PipelineWorker(
                    pipeline_manager=self.pipeline_manager,
                    input_text=input_text,
                    input_type="keywords"  # Indicate we're starting with keywords
                )

                # Set the ad-hoc configuration
                self.pipeline_worker.pipeline_manager.set_config(adhoc_config)

                # 5. Connect callbacks to handle search results
                self.pipeline_worker.step_completed.connect(self.on_search_completed)
                self.pipeline_worker.step_error.connect(self.on_search_error)

                # Update status when search actually starts
                self.status_label.setText("Verbindung zu Suchservices...")
                self.status_label.setStyleSheet(get_status_label_styles()["info"])

                self.pipeline_worker.start()
            else:
                # Single Source of Truth: Only PipelineManager allowed - Claude Generated
                error_msg = "PipelineManager nicht verfügbar - Suche kann nicht durchgeführt werden"
                self.logger.error(error_msg)
                self.handle_error(error_msg)
                return

        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            self.handle_error(str(e))

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
            self.logger.info(f"Verarbeite Suchbegriff: {search_term}")

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
                                self.logger.info(f"Get from DB: {keyword}")
                                entry = self.cache_manager.get_gnd_keyword(keyword)
                                if entry:
                                    # Verwende GND-ID aus Cache
                                    gnd_id = entry["gnd_id"]
                                else:
                                    self.unkown_terms.append(keyword)

                                self.logger.debug(
                                    f"Verwende GND-ID aus Cache: {gnd_id}"
                                )
                                if gnd_id:
                                    self.gnd_ids.append(gnd_id)

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
                    self.logger.info(f"Verarbeite Schlagwort: {keyword}")
                    # Bestimme die Beziehung zum Suchbegriff
                    relation = self.determine_relation(keyword, search_term)

                    # Ermittle GND-ID (erste aus dem Set oder leer)
                    gnd_id = next(iter(data.get("gndid", [])), "")

                    # Anzahl der Treffer
                    count = data.get("count", 1)

                    # Speichere die GND-ID für spätere Verwendung
                    if gnd_id:
                        self.gnd_ids.append(gnd_id)
                    else:
                        self.logger.info(f"Get from DB: {keyword}")
                        entry = self.cache_manager.get_gnd_keyword(keyword)
                        if entry:
                            # Verwende GND-ID aus Cache
                            gnd_id = entry["gnd_id"]
                        else:
                            self.unkown_terms.append(keyword)

                        self.logger.debug(f"Verwende GND-ID aus Cache: {gnd_id}")
                        if gnd_id:
                            self.gnd_ids.append(gnd_id)

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
        """Aktualisiert den aktuell ausgewählten Eintrag"""
        # Hole den ausgewählten GND-Eintrag aus der Tabelle
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            self.status_label.setText("Kein Eintrag ausgewählt.")
            self.status_label.setStyleSheet(get_status_label_styles()["warning"])
            return

        # Hole die GND-ID aus der ausgewählten Zeile
        row = selected_items[0].row()
        gnd_id = self.results_table.item(row, 1).text()

        # Aktualisiere den Eintrag
        if gnd_id:
            self.update_button.setEnabled(False)
            self.update_button.setText("Wird aktualisiert...")
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)
            self.progressBar.setMaximum(100)
            self.progressBar.setValue(10)
            QApplication.processEvents()

            self.update_entry(gnd_id)

            self.update_button.setEnabled(True)
            self.update_button.setText("Eintrag aktualisieren")
            self.progressBar.setVisible(False)

            # Zeige aktualisierte Details an
            self.show_details()

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
            gnd_entry = self.cache_manager.get_gnd_entry(gnd_id)

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

    def set_gnd_systematic(self, systematic: str):
        """
        Setzt die GND-Systematik für die Filterung.

        Args:
            systematic: String mit '|'-separierten GND-Systematiken
        """
        if systematic:
            # Split by | and filter out empty strings
            systematics = [sys.strip() for sys in systematic.split("|") if sys.strip()]

            # Clear existing filters first
            self.gnd_filter_widget.clear_systems()

            # Add each systematic to filter
            for sys in systematics:
                self.gnd_filter_widget.add_system(sys)

            # Apply the filter
            self.gnd_filter_widget.apply_filter()
        else:
            self.gnd_filter_widget.clear_systems()

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
