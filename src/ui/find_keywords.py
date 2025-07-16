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
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QIcon
from typing import Dict, List, Optional
import logging
import re
import sys
import json
from pathlib import Path

from ..core.suggesters.meta_suggester import MetaSuggester, SuggesterType


class SearchWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, search_cli, search_terms: List[str], suggester_types: List[SuggesterType]):
        super().__init__()
        self.search_cli = search_cli
        self.search_terms = search_terms
        self.suggester_types = suggester_types

    def run(self):
        try:
            results = self.search_cli.search(self.search_terms, self.suggester_types)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


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
        layout.setSpacing(8)
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

        self.add_button = QPushButton("Hinzufügen")
        self.add_button.clicked.connect(self.add_system)
        buttons_layout.addWidget(self.add_button)

        self.clear_button = QPushButton("Zurücksetzen")
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
        self.apply_button.clicked.connect(self.apply_filter)
        layout.addWidget(self.apply_button)

    def add_system(self):
        """Fügt eine GND-Systematik zur Filterliste hinzu"""
        system = self.system_input.toPlainText().strip()
        if not system:
            return
        self.add_system(system)

    def add_system(self, system):
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
        search_engine,
        cache_manager,
        parent=None,
        config_file: Path = Path.home() / ".alima_config.json",
    ):
        super().__init__(parent)
        self.search_engine = search_engine
        self.cache_manager = cache_manager
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

        # Lade den Katalog-Token aus der Konfigurationsdatei
        self.config_file = config_file
        self._load_catalog_token()

        self.init_ui()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche des Such-Tabs"""
        # Hauptlayout mit Abständen
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        # Farbdefinitionen für das UI (identisch mit AbstractTab)
        primary_color = "#4a86e8"  # Blau
        secondary_color = "#6aa84f"  # Grün
        accent_color = "#f1c232"  # Gold
        bg_light = "#f8f9fa"  # Hell-Grau
        text_color = "#333333"  # Dunkelgrau
        error_color = "#e74c3c"  # Rot

        # Globale QSS-Stile (identisch mit AbstractTab)
        self.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-top: 12px;
                background-color: {bg_light};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {bg_light};
            }}
            QPushButton {{
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                background-color: {primary_color};
                color: white;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #3a76d8;
            }}
            QPushButton:pressed {{
                background-color: #2a66c8;
            }}
            QLabel {{
                color: {text_color};
            }}
            QCheckBox {{
                color: {text_color};
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QTableWidget {{
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }}
            QTextEdit {{
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                font-size: 11pt;
            }}
        """
        )

        # ========= Kontrolleiste oben (wie bei AbstractTab) =========
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(0, 0, 0, 5)

        # Status-Label (ersetzt den separaten Status-Bereich)
        self.status_label = QLabel("Aktueller Status: Bereit")
        self.status_label.setStyleSheet("font-weight: bold;")
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

        # ========= Suchbereich =========
        search_group = QGroupBox("Schlagwortsuche")
        search_layout = QVBoxLayout(search_group)
        search_layout.setSpacing(8)
        search_layout.setContentsMargins(10, 20, 10, 10)

        # Hauptsuchfeld mit Beschreibung
        search_header = QLabel("Suchbegriffe:")
        search_header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        search_layout.addWidget(search_header)

        self.search_input = QTextEdit()
        self.search_input.setPlaceholderText(
            "Suchbegriffe (durch Komma getrennt oder in Anführungszeichen für exakte Phrasen)"
        )
        self.search_input.setMaximumHeight(100)  # Höhe begrenzen
        search_layout.addWidget(self.search_input)

        # Suchoptionen-Bereich
        options_frame = QFrame()
        options_frame.setStyleSheet(
            f"background-color: #e8f0fe; border-radius: 6px; padding: 8px;"
        )
        options_layout = QHBoxLayout(options_frame)

        # Checkboxen für die verschiedenen Suchquellen mit verbesserten Labels
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
        self.num_results.setMinimum(1)
        self.num_results.setMaximum(50)
        self.num_results.setValue(10)
        self.num_results.setToolTip(
            "Maximale Anzahl zu verarbeitender Ergebnisse pro Quelle"
        )
        self.num_results.setStyleSheet("background-color: white; padding: 2px;")
        options_layout.addWidget(self.num_results)

        search_layout.addWidget(options_frame)

        # Suchbutton (hervorgehoben)
        self.search_button = QPushButton("Suche starten")
        self.search_button.setMinimumHeight(40)
        self.search_button.clicked.connect(self.perform_search)
        self.search_button.setShortcut("Ctrl+Return")
        search_layout.addWidget(self.search_button)

        layout.addWidget(search_group)

        # ========= Ergebnisbereich =========
        results_splitter = QSplitter(Qt.Orientation.Vertical)

        # Separator-Linie vor den Ergebnissen (wie bei AbstractTab)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(
            "background-color: #cccccc; height: 2px; margin: 8px 0px;"
        )
        layout.addWidget(separator)

        # Obere Sektion: Ergebnistabelle und Details
        results_group = QGroupBox("Suchergebnisse")
        results_box_layout = QVBoxLayout(results_group)
        results_box_layout.setSpacing(8)
        results_box_layout.setContentsMargins(10, 20, 10, 10)

        upper_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Ergebnistabelle mit verbessertem Styling
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
        self.results_table.setStyleSheet(
            """
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
        """
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)  # Verstecke Row-Header

        table_layout.addWidget(self.results_table)
        upper_splitter.addWidget(table_frame)

        # Details-Sektion mit besserer Formatierung
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_header = QLabel("GND-Details:")
        details_header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        details_layout.addWidget(details_header)

        self.details_display = QTextEdit()
        self.details_display.setReadOnly(True)
        self.details_display.setStyleSheet(
            """
            QTextEdit {
                background-color: #f9f9f9;
                line-height: 1.3;
            }
        """
        )
        details_layout.addWidget(self.details_display)

        self.update_button = QPushButton("Eintrag aktualisieren")
        self.update_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {secondary_color};
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: #5a9840;
            }}
        """
        )
        self.update_button.clicked.connect(self.update_selected_entry)
        details_layout.addWidget(self.update_button)

        upper_splitter.addWidget(details_widget)
        upper_splitter.setSizes([600, 400])

        results_box_layout.addWidget(upper_splitter)

        # Wo vorher der DDC-Filter-Bereich war, fügen wir ein Tab-Widget ein
        # ========= Filter-Bereich mit Tabs =========
        filter_group = QGroupBox("Filteroptionen für Schlagwortgenerierung")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(8)
        filter_layout.setContentsMargins(10, 20, 10, 10)

        # Tab-Widget für DDC und GND-Systematik
        self.filter_tabs = QTabWidget()

        # DDC-Tab (bestehender Code)
        ddc_widget = QWidget()
        ddc_layout = QVBoxLayout(ddc_widget)
        ddc_layout.setSpacing(8)
        ddc_layout.setContentsMargins(10, 10, 10, 10)

        # DDC-Checkboxen in einem schönen Grid-Layout
        ddc_frame = QFrame()
        ddc_frame.setStyleSheet(
            f"background-color: #f0f8ff; border-radius: 6px; padding: 8px;"
        )
        ddc_grid = QGridLayout(ddc_frame)
        ddc_grid.setSpacing(10)

        # Erste Zeile: Beschriftung
        ddc_label = QLabel("Einzubeziehende DDC-Klassen:")
        ddc_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        ddc_grid.addWidget(ddc_label, 0, 0, 1, 5)

        # DDC-Checkboxen in 2 Reihen
        self.ddc1_check = QCheckBox("DDC 1xx (Philosophie)")
        self.ddc1_check.setChecked(True)
        ddc_grid.addWidget(self.ddc1_check, 1, 0)

        self.ddc2_check = QCheckBox("DDC 2xx (Religion)")
        self.ddc2_check.setChecked(True)
        ddc_grid.addWidget(self.ddc2_check, 1, 1)

        self.ddc3_check = QCheckBox("DDC 3xx (Sozialwiss.)")
        self.ddc3_check.setChecked(True)
        ddc_grid.addWidget(self.ddc3_check, 1, 2)

        self.ddc4_check = QCheckBox("DDC 4xx (Sprache)")
        self.ddc4_check.setChecked(True)
        ddc_grid.addWidget(self.ddc4_check, 1, 3)

        self.ddc5_check = QCheckBox("DDC 5xx (Naturwiss.)")
        self.ddc5_check.setChecked(True)
        ddc_grid.addWidget(self.ddc5_check, 1, 4)

        self.ddc6_check = QCheckBox("DDC 6xx (Technik)")
        self.ddc6_check.setChecked(True)
        ddc_grid.addWidget(self.ddc6_check, 2, 0)

        self.ddc7_check = QCheckBox("DDC 7xx (Kunst)")
        self.ddc7_check.setChecked(True)
        ddc_grid.addWidget(self.ddc7_check, 2, 1)

        self.ddc8_check = QCheckBox("DDC 8xx (Literatur)")
        self.ddc8_check.setChecked(True)
        ddc_grid.addWidget(self.ddc8_check, 2, 2)

        self.ddc9_check = QCheckBox("DDC 9xx (Geschichte)")
        self.ddc9_check.setChecked(True)
        ddc_grid.addWidget(self.ddc9_check, 2, 3)

        self.ddcX_check = QCheckBox("DDC X (Sonstige)")
        self.ddcX_check.setChecked(True)
        ddc_grid.addWidget(self.ddcX_check, 2, 4)

        ddc_layout.addWidget(ddc_frame)

        # Button zur DDC-Schlagwortgenerierung
        self.ddc_regenerate_button = QPushButton(
            "Nach DDC gefilterte Schlagwörter generieren"
        )
        self.ddc_regenerate_button.setToolTip(
            "Erzeugt eine gefilterte Liste von Schlagwörtern basierend auf DDC-Klassen"
        )
        self.ddc_regenerate_button.setMinimumHeight(40)
        self.ddc_regenerate_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {accent_color};
                color: black;
            }}
            QPushButton:hover {{
                background-color: #e1b222;
            }}
        """
        )
        self.ddc_regenerate_button.clicked.connect(self.generate_ddc_prompt)
        ddc_layout.addWidget(self.ddc_regenerate_button)

        # Zum Tab hinzufügen
        self.filter_tabs.addTab(ddc_widget, "DDC-Filter")

        # GND-Systematik-Tab (neu)
        self.gnd_filter_widget = GNDSystemFilterWidget(self, self.cache_manager)
        self.filter_tabs.addTab(self.gnd_filter_widget, "GND-Systematik-Filter")

        # Verbinde Button mit Funktion
        self.gnd_filter_widget.apply_button.clicked.connect(self.generate_gnd_prompt)

        # Tab-Widget zum Layout hinzufügen
        filter_layout.addWidget(self.filter_tabs)

        # Füge die gesamte Filter-Gruppe zum Layout hinzu
        results_box_layout.addWidget(filter_group)

        # Button zur Schlagwortgenerierung
        self.regenerate_button = QPushButton("Gefilterte Schlagwörter generieren")
        self.regenerate_button.setToolTip(
            "Erzeugt eine gefilterte Liste von Schlagwörtern basierend auf DDC-Klassen"
        )
        self.regenerate_button.setMinimumHeight(40)
        self.regenerate_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {accent_color};
                color: black;
            }}
            QPushButton:hover {{
                background-color: #e1b222;
            }}
        """
        )
        self.regenerate_button.clicked.connect(self.generate_ddc_prompt)
        filter_layout.addWidget(self.regenerate_button)

        # Füge Filter-Bereich zur Ergebnisgruppe hinzu
        results_box_layout.addWidget(filter_group)

        # Füge die gesamte Ergebnisgruppe zum Layout hinzu
        layout.addWidget(results_group)

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
            # Option 1: Direkt im Hauptbereich
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
        """Führt die Suche mit den ausgewählten Quellen durch"""
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

        # Eingabetext verarbeiten
        text = self.search_input.toPlainText().strip()

        # Keine Suchbegriffe vorhanden
        if not text:
            self.status_label.setText(
                "Bitte geben Sie mindestens einen Suchbegriff ein."
            )
            self.search_button.setEnabled(True)
            self.progressBar.setVisible(False)
            return

        # Extrahiere Suchbegriffe
        search_terms = self.extract_search_terms(text)

        # Keine gültigen Suchbegriffe
        if not search_terms:
            self.status_label.setText("Keine gültigen Suchbegriffe gefunden.")
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

        self.search_worker = SearchWorker(self.search_engine.search_cli, search_terms, suggester_types)
        self.search_worker.finished.connect(self.process_search_results)
        self.search_worker.error.connect(self.handle_error)
        self.search_worker.start()

    def process_search_results(self, results: dict):
        """Processes the search results from the CLI and displays them."""
        self.process_results(results)
        self.finalise_catalog_search(self.unkown_terms)

        # UI-Updates nach der Suche
        self.search_button.setEnabled(True)
        self.status_label.setText(
            f"Suche abgeschlossen - {len(self.flat_results)} Ergebnisse gefunden"
        )
        self.progressBar.setVisible(False)
        QApplication.processEvents()

    def handle_error(self, error_message: str):
        """Handles errors from the search worker."""
        self.logger.error(f"Search error: {error_message}")
        self.status_label.setText(f"Fehler: {error_message}")
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

        for search_term, term_results in results.items():
            self.logger.info(f"Verarbeite Suchbegriff: {search_term}")
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

    def finalise_catalog_search(self, keywords):
        """Im Katalog werden keine GND-IDs gespeichert, und wenn diese noch nicht im Cache sind, müssen die noch von der SWB
        abgerufen werden, das machen wir hiermit
        """
        if not keywords:
            return

        try:
            suggester = MetaSuggester(
                suggester_type=SuggesterType.SWB, debug=False, catalog_token=""
            )
            suggester.currentTerm.connect(self.current_term_update)
            results = suggester.search(keywords)
            self.process_results(results)
        except Exception as e:
            self.logger.error(f"Fehler bei SWB-Abfrage für unbekannte Terme: {str(e)}")

    def determine_relation(self, keyword, search_term):
        """Bestimmt die Beziehung zwischen Schlagwort und Suchbegriff"""
        # 0: exakt, 1: ähnlich, 2: verschieden
        if keyword.lower() == search_term.lower():
            return 0  # exakt
        elif (
            search_term.lower() in keyword.lower()
            or keyword.lower() in search_term.lower()
        ):
            return 1  # ähnlich
        else:
            return 2  # verschieden

    def update_database_entry(self, gnd_id, title, data):
        """Aktualisiert oder erstellt einen Datenbankeintrag für eine GND-ID"""
        if not self.cache_manager.gnd_entry_exists(gnd_id):
            # Lege neuen Eintrag an
            self.cache_manager.insert_gnd_entry(gnd_id, title=title)

        # Bei neuen Einträgen oder relevanten Einträgen aktualisiere die Daten

        # vorübergehend nicht durchführend
        # entry = self.cache_manager.get_gnd_entry(gnd_id)
        # if entry and entry["created_at"] == entry["updated_at"]:
        # Dieser Eintrag wurde noch nicht aktualisiert
        #    if (
        #        title != gnd_id
        #    ):  # Ignoriere Einträge, bei denen der Titel die GND-ID ist
        #        self.update_entry(gnd_id)

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

            entry = self.cache_manager.get_gnd_entry(gnd_id)
            if entry:
                gnd_system_item = QTableWidgetItem(entry.get("gnd_systems"))
                self.results_table.setItem(row, 4, gnd_system_item)

    def generate_initial_prompt(self, sorted_results):
        """Generiert einen initialen Prompt basierend auf den Suchergebnissen"""
        initial_prompt = []

        for gnd_id, term, count, relation, search_term in sorted_results:
            # Füge nur exakte und ähnliche Begriffe hinzu
            if term != gnd_id:  # Ignoriere, wenn Term = GND-ID
                list_item = f"{term} ({gnd_id})"
                if list_item not in initial_prompt:
                    initial_prompt.append(list_item)

        # Sende Signal mit den gefundenen Schlagwörtern
        if initial_prompt:
            self.keywords_found.emit(", ".join(initial_prompt))

    def generate_ddc_prompt(self):
        """Generiert einen Prompt basierend auf DDC-Filterung"""
        filtered_items = []

        for gnd_id in self.gnd_ids:
            gnd_entry = self.cache_manager.get_gnd_entry(gnd_id)
            if not gnd_entry:
                continue

            ddcs = gnd_entry.get("ddcs", "").split(";")
            include = False

            # Prüfe für jede DDC, ob sie den Filter-Kriterien entspricht
            for ddc in ddcs:
                if not ddc:
                    continue

                # Extrahiere den DDC-Code (entferne Determinancy in Klammern)
                ddc_code = ddc.split("(")[0].strip()

                # Prüfe, ob das erste Zeichen den Filtern entspricht
                if ddc_code and ddc_code[0].isdigit():
                    ddc_first = int(ddc_code[0])

                    # Prüfe gegen alle DDC-Filter
                    if (
                        (self.ddc1_check.isChecked() and ddc_first == 1)
                        or (self.ddc2_check.isChecked() and ddc_first == 2)
                        or (self.ddc3_check.isChecked() and ddc_first == 3)
                        or (self.ddc4_check.isChecked() and ddc_first == 4)
                        or (self.ddc5_check.isChecked() and ddc_first == 5)
                        or (self.ddc6_check.isChecked() and ddc_first == 6)
                        or (self.ddc7_check.isChecked() and ddc_first == 7)
                        or (self.ddc8_check.isChecked() and ddc_first == 8)
                        or (self.ddc9_check.isChecked() and ddc_first == 9)
                    ):
                        include = True
                        break
                elif self.ddcX_check.isChecked():
                    # Für nicht-numerische DDC-Codes oder wenn alle zugelassen sind
                    include = True
                    break

            # Wenn mindestens eine DDC den Kriterien entspricht, füge den Begriff hinzu
            if include:
                list_item = f"{gnd_entry['title']} ({gnd_id})"
                filtered_items.append(list_item)

        # Sende Signal mit den gefilterten Schlagwörtern
        if filtered_items:
            self.keywords_found.emit(", ".join(filtered_items))
            self.status_label.setText(
                f"{len(filtered_items)} gefilterte Schlagwörter generiert"
            )
        else:
            self.status_label.setText("Keine Schlagwörter entsprechen den DDC-Filtern")

    # Neue Funktion für GND-Systematik-Filterung hinzufügen
    def generate_gnd_prompt(self):
        """Generiert einen Prompt basierend auf GND-Systematik-Filterung"""
        selected_systems = self.gnd_filter_widget.get_selected_systems()

        if not selected_systems:
            self.status_label.setText("Keine GND-Systematiken ausgewählt.")
            return

        filtered_items = []

        for gnd_id in self.gnd_ids:
            gnd_entry = self.cache_manager.get_gnd_entry(gnd_id)
            if not gnd_entry:
                continue

            # GND-Systematiken sind mit | getrennt
            gnd_systems = gnd_entry.get("gnd_systems", "").split("|")
            include = False

            # Prüfe, ob mindestens eine der GND-Systematiken den Filterkriterien entspricht
            for system in gnd_systems:
                system = system.strip()
                if not system:
                    continue

                # Vergleiche mit ausgewählten Systemen (Präfixvergleich)
                for selected in selected_systems:
                    if system.startswith(selected):
                        include = True
                        break

                if include:
                    break

            # Wenn mindestens eine GND-Systematik den Kriterien entspricht, füge den Begriff hinzu
            if include:
                list_item = f"{gnd_entry['title']} ({gnd_id})"
                filtered_items.append(list_item)

        # Sende Signal mit den gefilterten Schlagwörtern
        if filtered_items:
            self.keywords_found.emit(", ".join(filtered_items))
            self.status_label.setText(
                f"{len(filtered_items)} nach GND-Systematik gefilterte Schlagwörter generiert"
            )
        else:
            self.status_label.setText(
                "Keine Schlagwörter entsprechen den GND-Systematik-Filtern"
            )

    def update_selected_entry(self):
        """Aktualisiert den aktuell ausgewählten Eintrag"""
        # Hole den ausgewählten GND-Eintrag aus der Tabelle
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            self.status_label.setText("Kein Eintrag ausgewählt.")
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

            else:
                # Fehlermeldung
                error_msg = (
                    dnb_class.get("error_message", "Keine Daten erhalten")
                    if dnb_class
                    else "Keine Daten erhalten"
                )
                self.logger.error(f"Fehler bei GND {gnd_id}: {error_msg}")
                self.status_label.setText(f"Fehler bei GND {gnd_id}: {error_msg}")

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
