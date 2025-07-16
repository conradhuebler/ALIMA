from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QMenuBar,
    QMenu,
    QStatusBar,
    QLabel,
    QPushButton,
    QDialog,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QFileDialog,
    QProgressDialog,
    QTextEdit,
)
import requests
import gzip
import os
import tempfile
from pathlib import Path
from PyQt6.QtCore import Qt, QSettings, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QCursor
import re
import os
import sys
import subprocess
import datetime
from typing import Optional, Dict

from .find_keywords import SearchTab
from .abstract_tab import AbstractTab
from .settings_dialog import SettingsDialog
from ..core.search_engine import SearchEngine
from ..core.cache_manager import CacheManager
from ..core.gndparser import GNDParser
from ..core.gitupdate import GitUpdateWorker
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from ..core.alima_manager import AlimaManager

from ..utils.config import Config
from .crossref_tab import CrossrefTab
from .analysis_review_tab import AnalysisReviewTab
from .ubsearch_tab import UBSearchTab
from .tablewidget import TableWidget
from .image_analysis_tab import ImageAnalysisTab
import logging


class CommitSelectorDialog(QDialog):
    def __init__(self, parent=None, repo_path=None):
        super().__init__(parent)
        self.repo_path = repo_path or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.selected_commit = None
        # Mindestdatum für Commits: 26. Mai 2025
        self.min_allowed_date = datetime.datetime(
            2025,
            5,
            26,
            14,
            7,
            16,
            tzinfo=datetime.timezone(datetime.timedelta(hours=2)),
        )

        self.init_ui()
        self.load_commits()

    def init_ui(self):
        self.setWindowTitle("Commit auswählen")
        self.setMinimumWidth(600)

        layout = QVBoxLayout(self)

        # Informationstext
        info_label = QLabel(
            "Wählen Sie einen spezifischen Commit oder Branch/Tag aus, zu dem gewechselt werden soll. "
            f"Aus Sicherheitsgründen werden nur Commits ab dem {self.min_allowed_date.strftime('%d.%m.%Y')} angezeigt."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-weight: bold; color: #CC0000;")
        layout.addWidget(info_label)

        # Auswahl zwischen Commit, Branch oder Tag
        self.selection_type = QComboBox()
        self.selection_type.addItems(["Branch", "Tag", "Commit"])
        self.selection_type.currentIndexChanged.connect(self.update_selection_list)
        layout.addWidget(self.selection_type)

        # Auswahlliste für Branches, Tags oder Commits
        self.commit_list = QComboBox()
        self.commit_list.setEditable(True)  # Erlaubt manuelle Eingabe
        layout.addWidget(self.commit_list)

        # Commit-Details anzeigen
        self.details_label = QLabel("Commit-Details:")
        layout.addWidget(self.details_label)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(200)
        layout.addWidget(self.details_text)

        self.commit_list.currentTextChanged.connect(self.show_commit_details)

        # Buttons
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("Auswählen")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setEnabled(False)  # Standardmäßig deaktiviert

        cancel_button = QPushButton("Abbrechen")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def load_commits(self):
        self.update_selection_list()

    def check_commit_date(self, commit_ref):
        """
        Überprüft, ob ein Commit nach dem Mindestdatum erstellt wurde.

        Returns:
            tuple: (bool, datetime) - (Ist gültig?, Commit-Datum)
        """
        try:
            # Commit-Datum abrufen
            result = subprocess.run(
                ["git", "show", "-s", "--format=%aD", commit_ref],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            commit_date_str = result.stdout.strip()

            # Konvertiere das Datum-String in ein datetime-Objekt
            commit_date = datetime.datetime.strptime(
                commit_date_str, "%a, %d %b %Y %H:%M:%S %z"
            )

            # Prüfe, ob das Commit-Datum nach dem Mindestdatum liegt
            is_valid = commit_date >= self.min_allowed_date

            return is_valid, commit_date

        except Exception:
            return False, None

    def get_filtered_commits(self, command, extract_func=None):
        """
        Führt einen Git-Befehl aus und filtert die Ergebnisse nach Datum.

        Args:
            command: Git-Befehlszeile als Liste
            extract_func: Optional Funktion zur Extraktion des Commit-Refs

        Returns:
            list: Liste der gültigen Commits/Branches/Tags
        """
        try:
            result = subprocess.run(
                command, cwd=self.repo_path, check=True, capture_output=True, text=True
            )
            all_items = result.stdout.strip().split("\n")

            valid_items = []
            for item in all_items:
                if not item.strip():
                    continue

                # Falls nötig, extrahiere die Commit-Referenz
                if extract_func:
                    commit_ref = extract_func(item)
                else:
                    commit_ref = item

                # Prüfe das Datum
                is_valid, _ = self.check_commit_date(commit_ref)
                if is_valid:
                    valid_items.append(item)

            return valid_items

        except subprocess.CalledProcessError:
            return []

    def update_selection_list(self):
        selection_type = self.selection_type.currentText()
        self.commit_list.clear()

        try:
            if selection_type == "Branch":
                # Lokale Branches laden und nach Datum filtern
                local_branches = self.get_filtered_commits(
                    ["git", "branch", "--format=%(refname:short)"]
                )

                # Remote Branches laden und nach Datum filtern
                remote_branches = self.get_filtered_commits(
                    ["git", "branch", "-r", "--format=%(refname:short)"]
                )

                all_branches = local_branches + remote_branches
                self.commit_list.addItems(
                    sorted([branch for branch in all_branches if branch.strip()])
                )

            elif selection_type == "Tag":
                # Tags laden und nach Datum filtern
                tags = self.get_filtered_commits(["git", "tag"])
                self.commit_list.addItems(sorted([tag for tag in tags if tag.strip()]))

            elif selection_type == "Commit":
                # Letzte 50 Commits laden und nach Datum filtern
                commits = self.get_filtered_commits(
                    ["git", "log", "-50", "--pretty=format:%h - %s (%an, %ar)"],
                    lambda x: x.split(" - ")[0].strip(),
                )
                self.commit_list.addItems(
                    [commit for commit in commits if commit.strip()]
                )

        except Exception as e:
            self.details_text.setText(f"Fehler beim Laden der Auswahl: {str(e)}")

        # Initialzustand setzen
        if self.commit_list.count() > 0:
            self.commit_list.setCurrentIndex(0)
        else:
            self.details_text.setText(
                f"Keine Einträge gefunden, die nach dem Mindestdatum ({self.min_allowed_date.strftime('%d.%m.%Y')}) erstellt wurden."
            )

    def show_commit_details(self, commit_ref):
        if not commit_ref:
            self.details_text.clear()
            self.ok_button.setEnabled(False)
            return

        # Bei Commits nur den Hash extrahieren
        if self.selection_type.currentText() == "Commit" and " - " in commit_ref:
            commit_ref = commit_ref.split(" - ")[0].strip()

        try:
            # Prüfe das Datum
            is_valid, commit_date = self.check_commit_date(commit_ref)

            if not is_valid:
                self.details_text.setText(
                    f"FEHLER: Der ausgewählte Commit wurde vor dem erlaubten Mindestdatum "
                    f"({self.min_allowed_date.strftime('%d.%m.%Y')}) erstellt.\n\n"
                    f"Commit-Datum: {commit_date.strftime('%d.%m.%Y %H:%M') if commit_date else 'Unbekannt'}"
                )
                self.ok_button.setEnabled(False)
                return

            # Commit-Details anzeigen
            result = subprocess.run(
                ["git", "show", "--stat", commit_ref],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            details = result.stdout.strip()

            date_info = f"Commit-Datum: {commit_date.strftime('%d.%m.%Y %H:%M')}\n\n"
            self.details_text.setText(date_info + details)

            # Aktiviere den OK-Button
            self.ok_button.setEnabled(True)

        except subprocess.CalledProcessError:
            self.details_text.setText(f"Ungültige Referenz: {commit_ref}")
            self.ok_button.setEnabled(False)
        except Exception as e:
            self.details_text.setText(
                f"Fehler beim Anzeigen der Commit-Details: {str(e)}"
            )
            self.ok_button.setEnabled(False)

    def get_selected_commit(self):
        commit_ref = self.commit_list.currentText()

        # Bei Commits nur den Hash extrahieren
        if self.selection_type.currentText() == "Commit" and " - " in commit_ref:
            commit_ref = commit_ref.split(" - ")[0].strip()

        # Führe eine letzte Prüfung durch, um sicherzustellen, dass das Datum passt
        is_valid, _ = self.check_commit_date(commit_ref)
        if not is_valid:
            return None

        return commit_ref


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("TUBAF", "Alima")
        # self.config = Config.get_instance()
        # Initialisiere Core-Komponenten
        self.cache_manager = CacheManager()
        self.search_engine = SearchEngine(self.cache_manager)
        self.logger = logging.getLogger(__name__)
        self.ollama_url_default = self.settings.value("ollama_url", "http://localhost")
        self.ollama_port_default = self.settings.value("ollama_port", "11434")

        # Instantiate core services
        self.llm_service = LlmService(
            ollama_url=self.ollama_url_default,
            ollama_port=self.ollama_port_default,
        )
        self.llm = self.llm_service  # Assign llm here
        self.prompt_service = PromptService(
            "prompts.json"
        )  # Pass the path to prompts.json
        self.alima_manager = AlimaManager(
            llm_service=self.llm_service,
            prompt_service=self.prompt_service,
            logger=self.logger,  # Pass logger to manager
        )

        self.available_models = {}
        self.available_providers = []

        self.init_ui()
        self.load_settings()
        self.load_models_and_providers()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        self.setWindowTitle("AlIma")
        self.setGeometry(100, 100, 1200, 800)

        # Zentrales Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Menüleiste
        self.create_menu_bar()

        # Tab-Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tabs erstellen
        self.search_tab = SearchTab(cache_manager=self.cache_manager)

        self.crossref_tab = CrossrefTab()

        # Pass alima_manager and llm_service to AbstractTab
        # Pass alima_manager and llm_service to AbstractTab
        self.abstract_tab = AbstractTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            main_window=self,
        )
        # self.crossref_tab.result_abstract.connect(self.abstract_tab.set_abstract)
        # self.crossref_tab.result_keywords.connect(self.abstract_tab.set_keywords)
        self.abstract_tab.template_name = "abstract_analysis"  # This might be removed later if task selection is fully dynamic
        self.abstract_tab.set_task("abstract")  # Set initial task

        # Pass alima_manager and llm_service to AbstractTab
        self.analyse_keywords = AbstractTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            main_window=self,
        )
        self.analyse_keywords.template_name = (
            "results_verification"  # This might be removed later
        )
        self.search_tab.keywords_found.connect(self.analyse_keywords.set_keywords)
        self.abstract_tab.abstract_changed.connect(self.analyse_keywords.set_abstract)
        self.analyse_keywords.need_keywords = True
        self.analyse_keywords.final_list.connect(self.update_gnd_keywords)
        self.analyse_keywords.set_task("keywords")

        self.ub_search_tab = UBSearchTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            main_window=self,
        )

        self.abstract_tab.final_list.connect(self.search_tab.update_search_field)
        self.abstract_tab.gnd_systematic.connect(self.search_tab.set_gnd_systematic)

        self.analyse_keywords.final_list.connect(self.ub_search_tab.update_keywords)

        self.crossref_tab.result_abstract.connect(self.ub_search_tab.set_abstract)
        self.abstract_tab.abstract_changed.connect(self.ub_search_tab.set_abstract)

        # Analysis Review Tab
        self.analysis_review_tab = AnalysisReviewTab()
        self.analysis_review_tab.keywords_selected.connect(
            self.search_tab.update_search_field
        )
        self.analysis_review_tab.abstract_selected.connect(
            self.abstract_tab.set_abstract
        )

        # Image Analysis Tab
        self.image_analysis_tab = ImageAnalysisTab(
            llm_service=self.llm_service,
            main_window=self
        )
        self.image_analysis_tab.text_extracted.connect(
            self.abstract_tab.set_abstract
        )

        self.tabs.addTab(self.crossref_tab, "Crossref DOI Lookup")
        self.tabs.addTab(self.image_analysis_tab, "Bilderkennung")
        self.tabs.addTab(self.abstract_tab, "Abstract-Analyse")
        self.tabs.addTab(self.search_tab, "GND-Suche")
        self.tabs.addTab(self.analyse_keywords, "Verifikation")
        self.tabs.addTab(self.ub_search_tab, "UB Suche")
        self.tabs.addTab(self.analysis_review_tab, "Analyse-Review")

        # Statusleiste
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel()
        self.status_bar.addWidget(self.status_label)

        # Cache-Info Widget
        self.ollama_url = QLineEdit()  # Eingabefeld für Ollama URL
        self.ollama_url.setText(self.ollama_url_default)
        self.ollama_url.textChanged.connect(
            lambda: self.llm.set_ollama_url(self.ollama_url.text())
        )
        self.ollama_port = QLineEdit()  # Eingabefeld für Ollama Port
        self.ollama_port.setText("11434")
        self.ollama_port.textChanged.connect(
            lambda: self.llm.set_ollama_port(self.ollama_port.text())
        )

        # Layout für Ollama-Einstellungen
        ollama_layout = QHBoxLayout()
        ollama_layout.addWidget(QLabel("Ollama URL:"))
        ollama_layout.addWidget(self.ollama_url)
        ollama_layout.addWidget(QLabel("Ollama Port:"))
        ollama_layout.addWidget(self.ollama_port)
        ollama_widget = QWidget()
        ollama_widget.setLayout(ollama_layout)
        # self.cache_info = QLabel()
        self.status_bar.addPermanentWidget(ollama_widget)

    def load_models_and_providers(self):
        """Loads all available models and providers."""
        self.available_providers = self.llm_service.get_available_providers()
        for provider in self.available_providers:
            self.available_models[provider] = self.llm_service.get_available_models(
                provider
            )

        # Pass the loaded models and providers to the tabs
        self.abstract_tab.set_models_and_providers(
            self.available_models, self.available_providers
        )
        self.analyse_keywords.set_models_and_providers(
            self.available_models, self.available_providers
        )
        self.ub_search_tab.set_models_and_providers(
            self.available_models, self.available_providers
        )

    @pyqtSlot(str)
    def update_gnd_keywords(self, keywords):
        self.logger.info(keywords)
        """
        Extrahiert GND-Schlagworte aus einem Text.
        
        Args:
            text (str): Der Text, der die GND-Einträge enthält
            
        Returns:
            list: Liste der GND-Schlagworte ohne IDs
        """
        self.logger.info(keywords)
        try:
            # Suche den Abschnitt mit den GND-Einträgen
            if "Schlagworte OGND Eintrage:" not in keywords:
                return []

            # Extrahiere den relevanten Teil des Textes
            gnd_section = keywords.split("Schlagworte OGND Eintrage:")[1].split(
                "FEHLENDE KONZEPTE:"
            )[0]

            # Extrahiere die Schlagworte (alles vor der URL)
            gnd_terms = []
            for line in gnd_section.split(","):
                if "(https://" in line:
                    term = line.split("(https://")[0].strip().replace('"', "")

                    self.logger.info(term)
                    if term:
                        gnd_terms.append(term)
            self.logger.info(gnd_terms)
            self.ub_search_tab.update_keywords(keywords)
            return gnd_terms

        except Exception as e:
            print(f"Fehler beim Extrahieren der GND-Terme: {str(e)}")
            return []

    @pyqtSlot(str)
    def update_search_field(self, keywords):
        """
        Extrahiert Keywords aus einem String, behandelt geklammerte Terme und schützt Slashes.

        Args:
            keyword_string (str): String mit kommagetrennten Keywords in Anführungszeichen

        Returns:
            list: Liste von extrahierten Keywords
            list: Liste von extrahierten Klammer-Termen
        """

        # Entferne Leerzeichen am Anfang und Ende
        keyword_string = keywords.strip()

        # Wenn der String mit [ beginnt und mit ] endet, entferne diese
        if keyword_string.startswith("[") and keyword_string.endswith("]"):
            keyword_string = keyword_string[1:-1]

        # Regulärer Ausdruck für das Matching
        # Matches entweder:
        # 1. Terme in Anführungszeichen die Slash enthalten
        # 2. Terme in Anführungszeichen mit Klammern
        # 3. Normale Terme in Anführungszeichen
        pattern = r'"([^"]*?/[^"]*?)"|"([^"]*?\([^)]+?\)[^"]*?)"|"([^"]*?)"'

        matches = re.finditer(pattern, keyword_string)

        keywords = []
        bracketed_terms = []

        for match in matches:
            # Wenn es ein Slash-Term ist
            if match.group(1):
                term = match.group(1).replace("/", "\\/")
                keywords.append(f'"{term}"')
            # Wenn es ein Term mit Klammern ist
            elif match.group(2):
                term = match.group(2)
                # Extrahiere den Klammerinhalt
                bracket_content = re.findall(r"\(([^)]+)\)", term)
                # Füge den Hauptterm zu den Keywords hinzu
                main_term = re.sub(r"\s*\([^)]+\)", "", term).strip()
                if main_term:
                    keywords.append(main_term)
                # Füge die Klammerterme zur separaten Liste hinzu
                bracketed_terms.extend([f'"{term}"' for term in bracket_content])
            # Wenn es ein normaler Term ist
            elif match.group(3):
                keywords.append(f'"{match.group(3)}"')

        result = keywords + bracketed_terms
        self.search_tab.update_search_field(", ".join(result))

    def show_settings(self):
        """Öffnet den Einstellungsdialog"""
        try:
            # Übergebe beide Konfigurationen getrennt
            dialog = SettingsDialog(ui_settings=self.settings, parent=self)
            if dialog.exec():
                # Einstellungen wurden gespeichert
                self.load_settings()
        except Exception as e:
            self.logger.error(f"Fehler beim Öffnen der Einstellungen: {e}")
            QMessageBox.critical(
                self, "Fehler", f"Fehler beim Öffnen der Einstellungen: {str(e)}"
            )

    def load_settings(self):
        """Lädt die gespeicherten Einstellungen"""
        # Fenster-Geometrie
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        self.tabs.setCurrentIndex(1)

        self.ollama_url.setText(self.settings.value("ollama_url", "http://localhost"))
        self.llm.set_ollama_url(self.ollama_url.text())
        self.ollama_port.setText(self.settings.value("ollama_port", "11434"))
        self.llm.set_ollama_port(self.ollama_port.text())

    def save_settings(self):
        """Speichert die aktuellen Einstellungen"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("ollama_url", self.ollama_url.text())
        self.settings.setValue("ollama_port", self.ollama_port.text())

    def export_results(self):
        """Exportiert die aktuellen Suchergebnisse"""
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, "export_results"):
            current_tab.export_results()
        else:
            self.status_label.setText("Export nicht verfügbar für diesen Tab")

    def export_current_analysis(self):
        """Export current analysis state from GUI"""
        try:
            # Get abstract from abstract tab
            abstract = self.abstract_tab.abstract_edit.toPlainText()

            # Get keywords from keywords tab
            keywords = self.analyse_keywords.keywords_edit.toPlainText()

            # Get search results from search tab
            search_results = {}
            if hasattr(self.search_tab, "flat_results"):
                for (
                    gnd_id,
                    term,
                    count,
                    relation,
                    search_term,
                ) in self.search_tab.flat_results:
                    if search_term not in search_results:
                        search_results[search_term] = {}
                    search_results[search_term][term] = {
                        "count": count,
                        "gndid": [gnd_id],
                        "ddc": [],
                        "dk": [],
                    }

            # Get final keywords from verification tab
            final_keywords = self.analyse_keywords.keywords_edit.toPlainText()

            # Get GND classes if available
            gnd_classes = ""
            if hasattr(self.analyse_keywords, "gnd_systematic"):
                gnd_classes = getattr(self.analyse_keywords, "gnd_systematic", "")

            # Export using analysis review tab
            self.analysis_review_tab.export_current_gui_state(
                abstract, keywords, search_results, final_keywords, gnd_classes
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Error exporting analysis: {str(e)}"
            )
            self.logger.error(f"Export error: {e}")

    def import_results(self):
        """Importiert gespeicherte Suchergebnisse"""
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, "import_results"):
            current_tab.import_results()
        else:
            self.status_label.setText("Import nicht verfügbar für diesen Tab")

    def show_about(self):
        """Zeigt den Über-Dialog"""
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("Über AlIma")
        layout = QVBoxLayout(about_dialog)

        # Über-Text
        about_text = QLabel(
            "AlIma - Sacherschließung mit LLMs\nVersion 1.0\n\n"
            "Entwickelt von Conrad Hübler\n"
            "TU Freiberg\n"
            "Lizenz: LGPL-3.0 license \n"
            "GitHub: https://github.com/conradhuebler/ALIMA"
        )
        layout.addWidget(about_text)

        # Schließen-Button
        close_button = QPushButton("Schließen")
        close_button.clicked.connect(about_dialog.close)
        layout.addWidget(close_button)

        about_dialog.exec()

    def show_help(self):
        """Zeigt den Hilfe-Dialog"""

    def show_prompt_editor(self):
        """Öffnet den Prompt-Editor-Dialog"""
        from .prompt_editor_dialog import PromptEditorDialog

        editor = PromptEditorDialog(self)
        editor.exec()

    def closeEvent(self, event):
        """Wird beim Schließen des Fensters aufgerufen"""
        self.save_settings()
        event.accept()

    def update_status(self, message: str):
        """Aktualisiert die Statusleiste"""
        self.status_label.setText(message)

    def show_error(self, message: str):
        """Zeigt eine Fehlermeldung"""

    def import_gnd_database(self):
        """Importiert die GND-Datenbank aus einer lokalen Datei oder lädt sie von DNB herunter"""
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QFileDialog
        from PyQt6.QtCore import Qt, QThread, pyqtSignal
        from PyQt6.QtGui import QCursor
        import os

        # Ask user for source
        reply = QMessageBox.question(
            self,
            "GND-Datenbank Import",
            "Möchten Sie eine lokale Datei auswählen oder die aktuelle GND-Datenbank von DNB herunterladen und speichern?",
            QMessageBox.StandardButton.Open
            | QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        xml_file_path = None

        # Set wait cursor
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))

        try:
            if reply == QMessageBox.StandardButton.Open:
                # Restore cursor for file dialog
                QApplication.restoreOverrideCursor()

                # Select local file
                filename, _ = QFileDialog.getOpenFileName(
                    self,
                    "GND-Datenbank auswählen",
                    "",
                    "XML/GZ-Dateien (*.xml *.xml.gz *.gz)",
                )

                if filename:
                    # Set wait cursor again
                    QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
                    xml_file_path = self._handle_file_extraction(filename)

            elif reply == QMessageBox.StandardButton.Save:
                # Download from DNB
                xml_file_path = self._download_and_extract_gnd()

            else:
                # Cancel
                QApplication.restoreOverrideCursor()
                return

            if xml_file_path and os.path.exists(xml_file_path):
                try:
                    # Create progress dialog for parsing
                    progress = QProgressDialog(
                        "Importiere GND-Datenbank...", "Abbrechen", 0, 0, self
                    )
                    progress.setWindowModality(Qt.WindowModality.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.show()

                    # Process events to show progress dialog
                    QApplication.processEvents()

                    parser = GNDParser(self.cache_manager)
                    self.logger.info(f"Importiere GND-Datenbank: {xml_file_path}")

                    # Connect parser progress signals if available
                    if hasattr(parser, "progress_updated"):
                        parser.progress_updated.connect(progress.setValue)
                        parser.status_updated.connect(progress.setLabelText)

                    parser.process_file(xml_file_path)

                    progress.close()
                    QMessageBox.information(
                        self, "Erfolg", "GND-Datenbank erfolgreich importiert!"
                    )

                except Exception as e:
                    progress.close()
                    QMessageBox.critical(
                        self, "Fehler", f"Fehler beim Importieren: {str(e)}"
                    )
            else:
                QMessageBox.warning(
                    self, "Fehler", "Keine gültige Datei zum Importieren gefunden."
                )

        finally:
            # Always restore cursor
            QApplication.restoreOverrideCursor()

    def _handle_file_extraction(self, filename: str) -> str:
        """Behandelt die Extraktion von gz-Dateien"""
        import gzip
        import tempfile
        from pathlib import Path
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt

        if filename.endswith(".gz"):
            # Extract gz file
            try:
                # Create progress dialog for extraction
                progress = QProgressDialog("Entpacke Datei...", "Abbrechen", 0, 0, self)
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()
                QApplication.processEvents()

                # Create temporary file for extracted content
                temp_dir = tempfile.mkdtemp()
                extracted_filename = Path(filename).stem  # Remove .gz extension
                temp_xml_path = os.path.join(temp_dir, extracted_filename)

                self.logger.info(f"Extrahiere {filename} nach {temp_xml_path}")

                with gzip.open(filename, "rb") as gz_file:
                    with open(temp_xml_path, "wb") as xml_file:
                        # Read in chunks to allow for progress updates
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            if progress.wasCanceled():
                                progress.close()
                                return None

                            chunk = gz_file.read(chunk_size)
                            if not chunk:
                                break
                            xml_file.write(chunk)
                            QApplication.processEvents()

                progress.close()
                return temp_xml_path

            except Exception as e:
                if "progress" in locals():
                    progress.close()
                self.logger.error(f"Fehler beim Extrahieren der gz-Datei: {str(e)}")
                QMessageBox.critical(
                    self, "Fehler", f"Fehler beim Extrahieren: {str(e)}"
                )
                return None
        else:
            # File is already XML
            return filename

    def _download_and_extract_gnd(self) -> str:
        """Lädt die GND-Datenbank von DNB herunter und extrahiert sie"""
        import requests
        import gzip
        import tempfile
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt

        url = "https://data.dnb.de/GND/authorities-gnd-sachbegriff_dnbmarc.mrc.xml.gz"

        try:
            # Create progress dialog
            progress = QProgressDialog(
                "Lade GND-Datenbank herunter...", "Abbrechen", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            # Download file
            self.logger.info(f"Lade GND-Datenbank herunter von: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get("content-length", 0))

            # Create temporary files
            temp_dir = tempfile.mkdtemp()
            temp_gz_path = os.path.join(temp_dir, "gnd_data.xml.gz")
            temp_xml_path = os.path.join(temp_dir, "gnd_data.xml")

            # Download with progress
            downloaded = 0
            with open(temp_gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if progress.wasCanceled():
                        progress.close()
                        return None

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress.setValue(
                            int((downloaded / total_size) * 50)
                        )  # 50% for download

                    QApplication.processEvents()

            progress.setLabelText("Entpacke Datenbank...")
            progress.setValue(50)
            QApplication.processEvents()

            # Extract gz file
            self.logger.info(f"Entpacke {temp_gz_path} nach {temp_xml_path}")
            with gzip.open(temp_gz_path, "rb") as gz_file:
                with open(temp_xml_path, "wb") as xml_file:
                    xml_file.write(gz_file.read())

            progress.setValue(100)
            progress.close()

            # Clean up gz file
            os.remove(temp_gz_path)

            self.logger.info(
                f"GND-Datenbank erfolgreich heruntergeladen und entpackt: {temp_xml_path}"
            )
            return temp_xml_path

        except requests.RequestException as e:
            if "progress" in locals():
                progress.close()
            self.logger.error(f"Fehler beim Herunterladen: {str(e)}")
            QMessageBox.critical(
                self, "Download-Fehler", f"Fehler beim Herunterladen: {str(e)}"
            )
            return None
        except Exception as e:
            if "progress" in locals():
                progress.close()
            self.logger.error(f"Fehler beim Verarbeiten der Datei: {str(e)}")
            QMessageBox.critical(self, "Fehler", f"Fehler beim Verarbeiten: {str(e)}")
            return None

    # In der MainWindow Klasse - füge folgende Methoden hinzu

    def create_menu_bar(self):
        """Erstellt die Menüleiste"""
        menubar = self.menuBar()

        # Datei-Menü
        file_menu = menubar.addMenu("&Datei")

        # Export-Aktion
        export_action = file_menu.addAction("&Exportieren...")
        export_action.triggered.connect(self.export_results)

        # Export Analysis
        export_analysis_action = file_menu.addAction(
            "&Aktuelle Verschlagwortung als JSON..."
        )
        export_analysis_action.triggered.connect(self.export_current_analysis)

        # Import-Aktion
        import_action = file_menu.addAction("&Importieren...")
        import_action.triggered.connect(self.import_gnd_database)

        file_menu.addSeparator()

        # Beenden-Aktion
        exit_action = file_menu.addAction("&Beenden")
        exit_action.triggered.connect(self.close)

        # Bearbeiten-Menü
        edit_menu = menubar.addMenu("&Bearbeiten")

        # Einstellungen-Aktion
        settings_action = edit_menu.addAction("&Einstellungen")
        settings_action.triggered.connect(self.show_settings)

        # Prompt-Konfiguration-Aktion
        prompt_config_action = edit_menu.addAction("&Prompt-Konfiguration")
        prompt_config_action.triggered.connect(self.show_prompt_editor)

        # Cache-Menü
        cache_menu = menubar.addMenu("&Cache")

        # Update-Menü hinzufügen/aktualisieren
        update_menu = menubar.addMenu("&Updates")

        # Nach Updates suchen
        check_update_action = update_menu.addAction("Nach &Updates suchen")
        check_update_action.triggered.connect(self.check_for_updates)

        # NEUE OPTION: Zu spezifischem Commit wechseln
        specific_commit_action = update_menu.addAction(
            "Zu &spezifischem Commit wechseln"
        )
        specific_commit_action.triggered.connect(self.checkout_specific_commit)

        # Hilfe-Menü
        help_menu = menubar.addMenu("&Hilfe")

        # Über-Dialog
        about_action = help_menu.addAction("Ü&ber")
        about_action.triggered.connect(self.show_about)

        # Hilfe-Dialog
        help_action = help_menu.addAction("&Hilfe")
        help_action.triggered.connect(self.show_help)

    def checkout_specific_commit(self):
        """Öffnet einen Dialog zur Auswahl eines spezifischen Commits"""
        dialog = CommitSelectorDialog(self)
        if dialog.exec():
            target_commit = dialog.get_selected_commit()
            if not target_commit:
                return

            reply = QMessageBox.question(
                self,
                "Zu spezifischem Commit wechseln",
                f"Möchten Sie wirklich zu '{target_commit}' wechseln? Dies kann zu Programminstabilität führen, "
                "wenn der ausgewählte Commit nicht mit der aktuellen Version kompatibel ist.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Starte den Update-Prozess mit spezifischem Commit
                self._start_update_process(target_commit=target_commit)

    def check_for_updates(self):
        """Prüft auf Updates und installiert sie bei Bedarf"""
        self.logger.info("Prüfe auf Updates...")
        self._start_update_process()

    def _start_update_process(self, target_commit=None):
        """Startet den Update-Prozess mit optionalem Ziel-Commit"""
        # Erstelle den Progress-Dialog
        if target_commit:
            message = f"Wechsle zu Commit: {target_commit}..."
        else:
            message = "Prüfe auf Updates..."

        progress = QProgressDialog(message, "Abbrechen", 0, 0, self)
        progress.setWindowTitle("Software-Update")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setCancelButton(None)  # Entferne den Abbrechen-Button
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.show()

        # Erstelle und starte den Worker
        self.update_worker = GitUpdateWorker(target_commit=target_commit)
        self.update_worker.update_progress.connect(
            lambda msg: progress.setLabelText(msg)
        )
        self.update_worker.update_finished.connect(
            lambda success, msg: self.update_completed(success, msg, progress)
        )
        self.update_worker.start()

    def update_completed(self, success, message, progress_dialog):
        """Wird aufgerufen, wenn der Update-Prozess abgeschlossen ist"""
        progress_dialog.close()

        if success:
            QMessageBox.information(self, "Update Status", message)
            if "bereits auf dem neuesten Stand" not in message:
                reply = QMessageBox.question(
                    self,
                    "Neustart erforderlich",
                    "Für die Anwendung der Updates ist ein Neustart erforderlich. Jetzt neu starten?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.save_settings()
                    # Starte das Programm neu
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
        else:
            QMessageBox.warning(self, "Update-Fehler", message)
