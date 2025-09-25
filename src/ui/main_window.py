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
from .comprehensive_settings_dialog import ComprehensiveSettingsDialog
from ..core.search_engine import SearchEngine
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..core.gndparser import GNDParser
from ..core.gitupdate import GitUpdateWorker
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from ..core.alima_manager import AlimaManager
from ..utils.config_manager import ConfigManager


from ..utils.config import Config
from .crossref_tab import CrossrefTab
from .analysis_review_tab import AnalysisReviewTab
from .ubsearch_tab import UBSearchTab
from .tablewidget import TableWidget
from .image_analysis_tab import ImageAnalysisTab
from .styles import get_main_stylesheet
from .global_status_bar import GlobalStatusBar
from .pipeline_tab import PipelineTab
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
        self.cache_manager = UnifiedKnowledgeManager()
        self.search_engine = SearchEngine(self.cache_manager)
        self.logger = logging.getLogger(__name__)
        self.ollama_url_default = self.settings.value("ollama_url", "http://localhost")
        self.ollama_port_default = self.settings.value("ollama_port", "11434")

        self.config_manager = ConfigManager(logger=self.logger)

        # Instantiate core services with lazy initialization for faster GUI startup - Claude Generated
        self.llm_service = LlmService(
            config_manager=self.config_manager, # Pass config manager
            lazy_initialization=True,  # Don't test providers during GUI startup
        )
        self.llm = self.llm_service  # Assign llm here
        self.prompt_service = PromptService(
            "prompts.json"
        )  # Pass the path to prompts.json
        self.alima_manager = AlimaManager(
            llm_service=self.llm_service,
            prompt_service=self.prompt_service,
            config_manager=self.config_manager, # Pass config manager
            logger=self.logger,  # Pass logger to manager
        )

        self.available_models = {}
        self.available_providers = []

        self.init_ui()
        self.load_settings()
        # Don't load models during startup - do it on demand - Claude Generated
        # self.load_models_and_providers()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        self.setWindowTitle("ALIMA - Automatisierte Schlagwortgenerierung")
        self.setGeometry(100, 100, 1400, 900)

        # Apply main stylesheet
        self.setStyleSheet(get_main_stylesheet())

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
        self.search_tab = SearchTab(
            cache_manager=self.cache_manager,
            alima_manager=self.alima_manager
        )

        self.crossref_tab = CrossrefTab()

        # Pass alima_manager and llm_service to AbstractTab
        # Pass alima_manager and llm_service to AbstractTab
        self.abstract_tab = AbstractTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            cache_manager=self.cache_manager,
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
            cache_manager=self.cache_manager,
            main_window=self,
        )
        self.analyse_keywords.template_name = (
            "results_verification"  # This might be removed later
        )
        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert - Claude Generated
        # self.search_tab.keywords_found.connect(self.analyse_keywords.set_keywords)
        # self.abstract_tab.abstract_changed.connect(self.analyse_keywords.set_abstract)
        self.analyse_keywords.need_keywords = True
        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert - Claude Generated
        # self.analyse_keywords.final_list.connect(self.update_gnd_keywords)
        self.analyse_keywords.set_task("keywords")

        self.ub_search_tab = UBSearchTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            main_window=self,
        )

        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert - Claude Generated
        # self.abstract_tab.final_list.connect(self.search_tab.update_search_field)
        # self.abstract_tab.gnd_systematic.connect(self.search_tab.set_gnd_systematic)

        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert - Claude Generated
        # self.analyse_keywords.final_list.connect(self.ub_search_tab.update_keywords)

        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert - Claude Generated
        # self.crossref_tab.result_abstract.connect(self.ub_search_tab.set_abstract)
        # self.abstract_tab.abstract_changed.connect(self.ub_search_tab.set_abstract)

        # Analysis Review Tab
        self.analysis_review_tab = AnalysisReviewTab()
        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert - Claude Generated
        # self.analysis_review_tab.keywords_selected.connect(
        #     self.search_tab.update_search_field
        # )
        # self.analysis_review_tab.abstract_selected.connect(
        #     self.abstract_tab.set_abstract
        # )

        # Connect AbstractTab analysis completion to AnalysisReviewTab - Claude Generated
        self.abstract_tab.analysis_completed.connect(
            self.analysis_review_tab.receive_analysis_data
        )

        # Image Analysis Tab
        self.image_analysis_tab = ImageAnalysisTab(
            llm_service=self.llm_service, main_window=self
        )
        # OBSOLET: Datenfluss wird jetzt vom PipelineManager gesteuert - Claude Generated
        # self.image_analysis_tab.text_extracted.connect(self.abstract_tab.set_abstract)

        # Pipeline Tab - Claude Generated
        self.pipeline_tab = PipelineTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            cache_manager=self.cache_manager,
            main_window=self,
        )

        # Connect pipeline events to global status bar
        self.pipeline_tab.pipeline_started.connect(
            lambda: self.global_status_bar.update_pipeline_status("Pipeline", "running")
        )
        self.pipeline_tab.pipeline_completed.connect(
            lambda: self.global_status_bar.update_pipeline_status(
                "Pipeline", "completed"
            )
        )
        
        # Connect pipeline results to specialized tab viewer methods - Claude Generated
        self.pipeline_tab.search_results_ready.connect(self.search_tab.display_search_results)
        self.pipeline_tab.metadata_ready.connect(self.crossref_tab.display_metadata)
        # Note: analysis_results_ready connection would be added when AbstractTab viewer method is implemented

        # Add Pipeline tab first
        self.tabs.addTab(self.pipeline_tab, "🚀 Pipeline")

        # Individual tabs
        self.tabs.addTab(self.crossref_tab, "Crossref DOI Lookup")
        self.tabs.addTab(self.image_analysis_tab, "Bilderkennung")
        self.tabs.addTab(self.abstract_tab, "Abstract-Analyse")
        self.tabs.addTab(self.search_tab, "GND-Suche")
        self.tabs.addTab(self.analyse_keywords, "Verifikation")
        self.tabs.addTab(self.ub_search_tab, "UB Suche")
        self.tabs.addTab(self.analysis_review_tab, "Analyse-Review")

        # Globale Statusleiste
        self.global_status_bar = GlobalStatusBar()
        self.setStatusBar(self.global_status_bar)

        # Initialize status bar with services
        self.global_status_bar.set_services(self.llm_service, self.cache_manager)

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
        self.global_status_bar.addPermanentWidget(ollama_widget)

    def ensure_models_and_providers_loaded(self):
        """Ensure models and providers are loaded on-demand - Claude Generated"""
        if not self.available_providers:
            self.load_models_and_providers()

    def load_models_and_providers(self):
        """Loads all available models and providers - Claude Generated"""
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
        """Öffnet den umfassenden Einstellungsdialog - Claude Generated"""
        try:
            dialog = ComprehensiveSettingsDialog(parent=self)
            dialog.config_changed.connect(self._on_config_changed)
            if dialog.exec():
                # Einstellungen wurden gespeichert
                self.load_settings()
                # Aktualisiere alle Komponenten mit neuer Konfiguration
                self._refresh_components()
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

        self.tabs.setCurrentIndex(0)

        self.ollama_url.setText(self.settings.value("ollama_url", "http://localhost"))
        self.llm.set_ollama_url(self.ollama_url.text())
        self.ollama_port.setText(self.settings.value("ollama_port", "11434"))
        self.llm.set_ollama_port(self.ollama_port.text())

    def save_settings(self):
        """Speichert die aktuellen Einstellungen"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("ollama_url", self.ollama_url.text())
        self.settings.setValue("ollama_port", self.ollama_port.text())

    def _on_config_changed(self):
        """Handle configuration changes from comprehensive settings dialog - Claude Generated"""
        self.logger.info("Configuration changed, refreshing components")
        self._refresh_components()

    def _refresh_components(self):
        """Refresh all components with new configuration - Claude Generated"""
        try:
            # Refresh LLM service configuration
            if hasattr(self, 'llm'):
                # LLM service will automatically reload config on next use
                pass
            
            # Refresh cache/database connections  
            if hasattr(self, 'cache_manager'):
                # Database connections will be recreated with new config
                pass
                
            # Update global status bar
            if hasattr(self, 'global_status_bar'):
                # Status bar will automatically update on next refresh
                pass
                
            # Notify tabs about configuration changes
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                if hasattr(tab, 'on_config_changed'):
                    tab.on_config_changed()
                    
            self.logger.info("Component refresh completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error refreshing components: {e}")

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
                    
                    # Console Progress Output - Claude Generated
                    print("🔄 Starte GND-Datenbank Import...")
                    print(f"📁 Datei: {xml_file_path}")

                    # Connect parser progress signals if available
                    if hasattr(parser, "progress_updated"):
                        parser.progress_updated.connect(progress.setValue)
                        parser.status_updated.connect(progress.setLabelText)
                        
                        # Also connect to console output - Claude Generated
                        def console_progress(value):
                            if value > 0:
                                print(f"📊 Fortschritt: {value}%")
                        
                        def console_status(status):
                            print(f"ℹ️ Status: {status}")
                            
                        parser.progress_updated.connect(console_progress)
                        parser.status_updated.connect(console_status)

                    print("⚙️ Verarbeite XML-Daten...")
                    parser.process_file(xml_file_path)
                    print("✅ GND-Import erfolgreich abgeschlossen!")

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

    def load_analysis_state_from_file(self):
        """
        Öffnet einen Datei-Dialog, um einen JSON-Analyse-Zustand zu laden
        und die UI damit zu befüllen - Claude Generated
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from ..utils.pipeline_utils import PipelineJsonManager
        from ..core.data_models import KeywordAnalysisState

        # Datei-Dialog öffnen
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Analyse-Zustand laden",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_name:
            return  # Benutzer hat abgebrochen

        try:
            # 1. JSON-Datei laden und parsen
            self.logger.info(f"Loading analysis state from: {file_name}")
            state = PipelineJsonManager.load_analysis_state(file_name)

            # 2. Daten an die Tabs verteilen
            self.populate_all_tabs_from_state(state)

            # 3. Erfolgsmeldung
            self.global_status_bar.show_message("✅ Analyse-Zustand erfolgreich geladen.", 5000)

            # 4. Zur Pipeline-Ansicht wechseln für Übersicht
            self.tabs.setCurrentWidget(self.pipeline_tab)

            self.logger.info("Analysis state successfully loaded and distributed to tabs")

        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Analyse-Zustands: {e}")
            QMessageBox.critical(
                self,
                "Ladefehler",
                f"Die Datei konnte nicht geladen werden:\n\n{str(e)}"
            )

    def populate_all_tabs_from_state(self, state):
        """
        Verteilt die Daten aus einem KeywordAnalysisState-Objekt
        an alle relevanten UI-Tabs - Claude Generated
        """
        from ..core.data_models import KeywordAnalysisState

        self.logger.info("Distributing analysis state data to all tabs...")

        # Helper function to convert search results format
        def convert_search_results_to_dict(results_list):
            """Convert List[SearchResult] to Dict format expected by UI"""
            search_dict = {}
            for search_result in results_list:
                search_dict[search_result.search_term] = search_result.results
            return search_dict

        # Helper function to extract GND keywords for display
        def extract_gnd_keywords_from_search_results(search_results):
            """Extract formatted GND keywords from search results"""
            gnd_keywords = []
            for search_result in search_results:
                for keyword, data in search_result.results.items():
                    gnd_ids = data.get("gndid", [])
                    for gnd_id in gnd_ids:
                        # Get GND title from cache if available
                        gnd_title = self.cache_manager.get_gnd_title_by_id(gnd_id)
                        if gnd_title:
                            formatted_keyword = f"{gnd_title} (GND-ID: {gnd_id})"
                        else:
                            formatted_keyword = f"{keyword} (GND-ID: {gnd_id})"
                        gnd_keywords.append(formatted_keyword)
            return gnd_keywords

        try:
            # 1. 🚀 Pipeline Tab - Complete workflow overview
            if hasattr(self.pipeline_tab, 'unified_input') and state.original_abstract:
                self.pipeline_tab.unified_input.set_text(state.original_abstract)
                # Show visual indicators for loaded state
                if hasattr(self.pipeline_tab, 'show_loaded_state_indicator'):
                    self.pipeline_tab.show_loaded_state_indicator(state)
                self.logger.info("✅ Pipeline tab populated with original abstract and state indicators")

            # 2. 📄 Abstract-Analyse Tab - Initial analysis playground
            if state.original_abstract:
                self.abstract_tab.set_abstract(state.original_abstract)
                # Show LLM details if available and set loaded analysis context
                if state.initial_llm_call_details:
                    llm_details = state.initial_llm_call_details
                    # Add loaded analysis info to results area if available
                    if hasattr(self.abstract_tab, 'results_text'):
                        loaded_info = (
                            f"📁 Geladene Analyse:\n"
                            f"Provider: {llm_details.provider_used}\n"
                            f"Model: {llm_details.model_used}\n"
                            f"Task: {llm_details.task_name}\n"
                            f"Temperature: {llm_details.temperature}\n\n"
                            f"Extrahierte Keywords:\n{', '.join(state.initial_keywords)}\n\n"
                            f"Original LLM Response:\n{llm_details.response_full_text}"
                        )
                        self.abstract_tab.results_text.setPlainText(loaded_info)
                    self.logger.info(f"📊 Initial LLM analysis used: {llm_details.provider_used}/{llm_details.model_used}")
                self.logger.info("✅ Abstract tab populated with original text and analysis context")

            # 3. 🔍 GND-Suche Tab - Search results and new searches
            if state.initial_keywords:
                # Fill search input with initial keywords
                keywords_text = ", ".join(state.initial_keywords)
                self.search_tab.update_search_field(keywords_text)

                # Display search results if available
                if state.search_results:
                    search_results_dict = convert_search_results_to_dict(state.search_results)
                    self.search_tab.display_search_results(search_results_dict)
                    self.logger.info(f"✅ Search tab populated with {len(state.search_results)} search result sets")

            # 4. ✅ Verifikation Tab - Final analysis and GND pool
            if state.original_abstract:
                self.analyse_keywords.set_abstract(state.original_abstract)

                # Provide GND keyword pool for verification
                if state.search_results:
                    gnd_keywords = extract_gnd_keywords_from_search_results(state.search_results)
                    if gnd_keywords:
                        self.analyse_keywords.set_keywords("\n".join(gnd_keywords))
                        self.logger.info(f"✅ Verification tab populated with {len(gnd_keywords)} GND keywords")

            # 5. 📊 Analyse-Review Tab - Complete results and export
            if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                final_keywords = ", ".join(state.final_llm_analysis.extracted_gnd_keywords)
                full_response = state.final_llm_analysis.response_full_text

                self.analysis_review_tab.receive_analysis_data(
                    state.original_abstract or "",
                    final_keywords,
                    full_response
                )
                self.logger.info("✅ Analysis review tab populated with final results")

            # 6. 🏛️ UB Suche Tab - Keywords for library catalog search
            if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                final_keywords = ", ".join(state.final_llm_analysis.extracted_gnd_keywords)
                if hasattr(self.ub_search_tab, 'update_keywords'):
                    self.ub_search_tab.update_keywords(final_keywords)
                if hasattr(self.ub_search_tab, 'set_abstract') and state.original_abstract:
                    self.ub_search_tab.set_abstract(state.original_abstract)
                self.logger.info("✅ UB search tab populated with final keywords")

            # 7. 🖼️ Bilderkennung Tab - Show OCR details if input was image
            # Note: Currently we don't have image source info in KeywordAnalysisState
            # This could be enhanced in future versions

            # 8. Show summary in status bar
            total_keywords = len(state.final_llm_analysis.extracted_gnd_keywords) if state.final_llm_analysis else len(state.initial_keywords)
            summary_message = f"📁 Geladen: {total_keywords} Schlagwörter aus {len(state.search_results)} Suchvorgängen"
            self.global_status_bar.show_message(summary_message, 10000)

            self.logger.info(f"🎯 Analysis state distribution complete: {summary_message}")

        except Exception as e:
            self.logger.error(f"Error distributing analysis state: {e}")
            raise  # Re-raise to be handled by calling method

    def collect_current_gui_state(self):
        """
        Sammelt den aktuellen Zustand der GUI-Tabs zu einem KeywordAnalysisState-Objekt
        für Export oder Vergleichszwecke - Claude Generated
        """
        from ..core.data_models import KeywordAnalysisState, LlmKeywordAnalysis, SearchResult
        from ..utils.pipeline_utils import PipelineJsonManager

        try:
            self.logger.info("Collecting current GUI state for export...")

            # 1. Sammle Abstract/Input Text
            original_abstract = ""
            if hasattr(self.pipeline_tab, 'unified_input'):
                original_abstract = self.pipeline_tab.unified_input.get_text() or ""
            elif hasattr(self.abstract_tab, 'abstract_text'):
                original_abstract = self.abstract_tab.abstract_text.toPlainText() or ""

            # 2. Sammle initiale Keywords (aus Abstract Tab oder Search Tab Input)
            initial_keywords = []
            if hasattr(self.search_tab, 'search_field') and self.search_tab.search_field.text():
                keywords_text = self.search_tab.search_field.text()
                initial_keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]

            # 3. Sammle Suchergebnisse (aus Search Tab)
            search_results = []
            if hasattr(self.search_tab, 'flat_results') and self.search_tab.flat_results:
                # Convert flat results back to SearchResult format
                # This is a simplified conversion - in a full implementation we'd need more logic
                search_results.append(SearchResult(
                    search_term="current_search",
                    results={"gui_results": {"count": len(self.search_tab.flat_results)}}
                ))

            # 4. Sammle finale Keywords (aus Analysis Review Tab oder Verification Tab)
            final_keywords = []
            final_response = ""
            if hasattr(self.analysis_review_tab, 'keywords_text') and self.analysis_review_tab.keywords_text.toPlainText():
                keywords_text = self.analysis_review_tab.keywords_text.toPlainText()
                final_keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            if hasattr(self.analysis_review_tab, 'response_text') and self.analysis_review_tab.response_text.toPlainText():
                final_response = self.analysis_review_tab.response_text.toPlainText()

            # 5. Erstelle LLM Analysis Objekt für finale Analyse
            final_llm_analysis = None
            if final_keywords or final_response:
                final_llm_analysis = LlmKeywordAnalysis(
                    task_name="gui_collection",
                    model_used="unknown",  # Could be enhanced to track current model
                    provider_used="unknown",  # Could be enhanced to track current provider
                    prompt_template="",
                    filled_prompt="",
                    temperature=0.7,
                    seed=None,
                    response_full_text=final_response,
                    extracted_gnd_keywords=final_keywords,
                    extracted_gnd_classes=[]
                )

            # 6. Erstelle KeywordAnalysisState
            state = KeywordAnalysisState(
                original_abstract=original_abstract,
                initial_keywords=initial_keywords,
                search_suggesters_used=["gui_collection"],
                initial_gnd_classes=[],
                search_results=search_results,
                initial_llm_call_details=None,
                final_llm_analysis=final_llm_analysis,
                pipeline_step_completed="gui_collected"
            )

            self.logger.info(f"GUI state collected: {len(initial_keywords)} initial keywords, {len(final_keywords)} final keywords")
            return state

        except Exception as e:
            self.logger.error(f"Error collecting GUI state: {e}")
            raise

    def export_current_gui_state(self):
        """
        Exportiert den aktuellen GUI-Zustand als JSON-Datei - Claude Generated
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from ..utils.pipeline_utils import PipelineJsonManager

        try:
            # 1. Sammle aktuellen GUI-Zustand
            state = self.collect_current_gui_state()

            # 2. Datei-Dialog für Export
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "GUI-Zustand exportieren",
                f"gui_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON Files (*.json);;All Files (*)"
            )

            if not file_name:
                return  # Benutzer hat abgebrochen

            # 3. Export durchführen
            PipelineJsonManager.save_analysis_state(state, file_name)

            # 4. Erfolgsmeldung
            self.global_status_bar.show_message("✅ GUI-Zustand erfolgreich exportiert.", 5000)
            QMessageBox.information(
                self,
                "Export erfolgreich",
                f"GUI-Zustand wurde erfolgreich exportiert:\n{file_name}"
            )

            self.logger.info(f"GUI state successfully exported to: {file_name}")

        except Exception as e:
            self.logger.error(f"Error exporting GUI state: {e}")
            QMessageBox.critical(
                self,
                "Export-Fehler",
                f"Der GUI-Zustand konnte nicht exportiert werden:\n\n{str(e)}"
            )

    def compare_analysis_states(self):
        """
        Ermöglicht den Vergleich von zwei Analysis-States - Claude Generated
        Diese Funktion ist eine Grundlage für erweiterte Analytics Features
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QTextEdit, QPushButton

        try:
            # 1. Erste Datei auswählen
            file1, _ = QFileDialog.getOpenFileName(
                self,
                "Erste Analyse-Datei auswählen",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            if not file1:
                return

            # 2. Zweite Datei auswählen
            file2, _ = QFileDialog.getOpenFileName(
                self,
                "Zweite Analyse-Datei auswählen",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            if not file2:
                return

            # 3. Beide Dateien laden
            from ..utils.pipeline_utils import PipelineJsonManager
            state1 = PipelineJsonManager.load_analysis_state(file1)
            state2 = PipelineJsonManager.load_analysis_state(file2)

            # 4. Einfachen Vergleich erstellen
            comparison_text = self._create_state_comparison(state1, state2, file1, file2)

            # 5. Vergleichsdialog anzeigen
            dialog = QDialog(self)
            dialog.setWindowTitle("Analysis-State Vergleich")
            dialog.setMinimumSize(800, 600)

            layout = QVBoxLayout(dialog)
            text_widget = QTextEdit()
            text_widget.setPlainText(comparison_text)
            text_widget.setReadOnly(True)
            layout.addWidget(text_widget)

            close_button = QPushButton("Schließen")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)

            dialog.exec()

        except Exception as e:
            self.logger.error(f"Error comparing analysis states: {e}")
            QMessageBox.critical(
                self,
                "Vergleichsfehler",
                f"Die Analysis-States konnten nicht verglichen werden:\n\n{str(e)}"
            )

    def _create_state_comparison(self, state1, state2, file1, file2):
        """
        Erstellt einen textuellen Vergleich zwischen zwei Analysis-States - Claude Generated
        """
        from pathlib import Path

        comparison_lines = []
        comparison_lines.append("=== ANALYSIS-STATE VERGLEICH ===\n")
        comparison_lines.append(f"Datei 1: {Path(file1).name}")
        comparison_lines.append(f"Datei 2: {Path(file2).name}")
        comparison_lines.append("\n" + "="*50 + "\n")

        # Vergleiche Basis-Informationen
        comparison_lines.append("📄 ABSTRACT/INPUT:")
        if state1.original_abstract != state2.original_abstract:
            comparison_lines.append("  ❌ UNTERSCHIEDLICH")
            comparison_lines.append(f"  Datei 1: {len(state1.original_abstract or '')} Zeichen")
            comparison_lines.append(f"  Datei 2: {len(state2.original_abstract or '')} Zeichen")
        else:
            comparison_lines.append("  ✅ IDENTISCH")

        # Vergleiche Keywords
        comparison_lines.append("\n🔑 INITIALE KEYWORDS:")
        keywords1 = set(state1.initial_keywords)
        keywords2 = set(state2.initial_keywords)

        if keywords1 == keywords2:
            comparison_lines.append("  ✅ IDENTISCH")
        else:
            comparison_lines.append("  ❌ UNTERSCHIEDLICH")
            only_in_1 = keywords1 - keywords2
            only_in_2 = keywords2 - keywords1
            both = keywords1 & keywords2

            if both:
                comparison_lines.append(f"  🤝 Gemeinsam ({len(both)}): {', '.join(sorted(both))}")
            if only_in_1:
                comparison_lines.append(f"  📁 Nur in Datei 1 ({len(only_in_1)}): {', '.join(sorted(only_in_1))}")
            if only_in_2:
                comparison_lines.append(f"  📁 Nur in Datei 2 ({len(only_in_2)}): {', '.join(sorted(only_in_2))}")

        # Vergleiche finale Keywords
        comparison_lines.append("\n🎯 FINALE KEYWORDS:")
        final1 = set(state1.final_llm_analysis.extracted_gnd_keywords) if state1.final_llm_analysis else set()
        final2 = set(state2.final_llm_analysis.extracted_gnd_keywords) if state2.final_llm_analysis else set()

        if final1 == final2:
            comparison_lines.append("  ✅ IDENTISCH")
        else:
            comparison_lines.append("  ❌ UNTERSCHIEDLICH")
            only_in_1 = final1 - final2
            only_in_2 = final2 - final1
            both = final1 & final2

            if both:
                comparison_lines.append(f"  🤝 Gemeinsam ({len(both)}): {', '.join(sorted(both))}")
            if only_in_1:
                comparison_lines.append(f"  📁 Nur in Datei 1 ({len(only_in_1)}): {', '.join(sorted(only_in_1))}")
            if only_in_2:
                comparison_lines.append(f"  📁 Nur in Datei 2 ({len(only_in_2)}): {', '.join(sorted(only_in_2))}")

        # Vergleiche LLM-Details
        comparison_lines.append("\n🤖 LLM-DETAILS:")
        if state1.final_llm_analysis and state2.final_llm_analysis:
            llm1 = state1.final_llm_analysis
            llm2 = state2.final_llm_analysis

            if llm1.provider_used != llm2.provider_used:
                comparison_lines.append(f"  Provider: {llm1.provider_used} vs {llm2.provider_used}")
            if llm1.model_used != llm2.model_used:
                comparison_lines.append(f"  Model: {llm1.model_used} vs {llm2.model_used}")
            if llm1.temperature != llm2.temperature:
                comparison_lines.append(f"  Temperature: {llm1.temperature} vs {llm2.temperature}")
        elif state1.final_llm_analysis:
            comparison_lines.append("  📁 Nur Datei 1 hat LLM-Details")
        elif state2.final_llm_analysis:
            comparison_lines.append("  📁 Nur Datei 2 hat LLM-Details")
        else:
            comparison_lines.append("  ❌ Keine LLM-Details in beiden Dateien")

        comparison_lines.append("\n" + "="*50)
        comparison_lines.append("💡 Tipp: Verwende die Tabs zur detaillierteren Analyse der geladenen Daten!")

        return "\n".join(comparison_lines)

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
            
            # Console Progress Output - Claude Generated
            print("🌐 Starte DNB-Download...")
            print(f"📡 URL: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                print(f"📦 Dateigröße: {total_size / (1024*1024):.1f} MB")

            # Create temporary files
            temp_dir = tempfile.mkdtemp()
            temp_gz_path = os.path.join(temp_dir, "gnd_data.xml.gz")
            temp_xml_path = os.path.join(temp_dir, "gnd_data.xml")

            # Download with progress
            downloaded = 0
            last_console_percent = 0
            
            print("⬇️ Download läuft...")
            with open(temp_gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if progress.wasCanceled():
                        print("❌ Download abgebrochen")
                        progress.close()
                        return None

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        download_percent = int((downloaded / total_size) * 50)
                        progress.setValue(download_percent)  # 50% for download
                        
                        # Console progress every 10% - Claude Generated
                        console_percent = (downloaded / total_size) * 100
                        if console_percent - last_console_percent >= 10:
                            print(f"📊 Download: {console_percent:.0f}%")
                            last_console_percent = console_percent

                    QApplication.processEvents()

            progress.setLabelText("Entpacke Datenbank...")
            progress.setValue(50)
            QApplication.processEvents()

            # Extract gz file
            print("📦 Entpacke GZ-Datei...")
            self.logger.info(f"Entpacke {temp_gz_path} nach {temp_xml_path}")
            with gzip.open(temp_gz_path, "rb") as gz_file:
                with open(temp_xml_path, "wb") as xml_file:
                    xml_file.write(gz_file.read())

            print("✅ Download und Entpackung abgeschlossen")
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

    def import_lobid_dnb_data(self):
        """Importiert DNB/GND-Daten über LobidSuggester mit Progress - Claude Generated"""
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QVBoxLayout, QDialog, QLabel, QTextEdit, QPushButton
        from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
        from PyQt6.QtGui import QCursor, QFont
        from ..core.suggesters.lobid_suggester import LobidSuggester
        from pathlib import Path
        import time
        
        class LobidImportWorker(QThread):
            """Worker thread for Lobid DNB import - Claude Generated"""
            progress_updated = pyqtSignal(str)  # Progress message
            finished_successfully = pyqtSignal(int)  # Number of entries imported
            error_occurred = pyqtSignal(str)  # Error message
            
            def __init__(self, force_download=False, debug=False):
                super().__init__()
                self.force_download = force_download
                self.debug = debug
                
            def run(self):
                try:
                    self.progress_updated.emit("🔄 Initialisiere Lobid-Suggester...")
                    
                    # Create LobidSuggester instance for DNB import
                    data_dir = Path("data") / "lobid"
                    lobid_suggester = LobidSuggester(data_dir=data_dir, debug=self.debug)
                    
                    self.progress_updated.emit(f"📁 Datenverzeichnis: {data_dir}")
                    self.progress_updated.emit(f"🔄 Erzwungener Download: {self.force_download}")
                    
                    start_time = time.time()
                    
                    if self.force_download or not (data_dir / "subjects.json").exists():
                        self.progress_updated.emit("⬇️ Lade GND-Sachbegriffe von DNB herunter...")
                        
                    # Use the prepare method which handles download and processing
                    lobid_suggester.prepare(force_gnd_download=self.force_download)
                    
                    elapsed = time.time() - start_time
                    self.progress_updated.emit(f"✅ DNB-Import abgeschlossen in {elapsed:.2f} Sekunden")
                    
                    # Show some statistics
                    if lobid_suggester.gnd_subjects:
                        subject_count = len(lobid_suggester.gnd_subjects)
                        self.progress_updated.emit(f"📊 Importierte {subject_count:,} GND-Sachbegriff-Einträge")
                        
                        # Show sample entries
                        sample_entries = list(lobid_suggester.gnd_subjects.items())[:3]
                        self.progress_updated.emit("📋 Beispiel-Einträge:")
                        for gnd_id, title in sample_entries:
                            self.progress_updated.emit(f"   {gnd_id}: {title}")
                        
                        self.finished_successfully.emit(subject_count)
                    else:
                        self.error_occurred.emit("Keine GND-Sachbegriffe gefunden")
                        
                except Exception as e:
                    self.error_occurred.emit(f"Import-Fehler: {str(e)}")
        
        # Ask user for import options
        reply = QMessageBox.question(
            self,
            "Lobid DNB Import",
            "Möchten Sie die GND-Sachbegriffe von DNB herunterladen?\n\n"
            "Dies kann einige Minuten dauern, da die Daten heruntergeladen und verarbeitet werden müssen.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Check if data already exists
        data_dir = Path("data") / "lobid" 
        subjects_file = data_dir / "subjects.json"
        force_download = False
        
        if subjects_file.exists():
            force_reply = QMessageBox.question(
                self,
                "Daten bereits vorhanden",
                f"GND-Daten wurden bereits gefunden in:\n{subjects_file}\n\n"
                "Möchten Sie die Daten trotzdem neu herunterladen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            force_download = (force_reply == QMessageBox.StandardButton.Yes)
        
        # Create progress dialog
        class ImportProgressDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Lobid DNB Import")
                self.setModal(True)
                self.resize(600, 400)
                
                layout = QVBoxLayout()
                
                self.status_label = QLabel("Starte Import...")
                font = QFont()
                font.setBold(True)
                self.status_label.setFont(font)
                layout.addWidget(self.status_label)
                
                self.progress_text = QTextEdit()
                self.progress_text.setReadOnly(True)
                self.progress_text.setFont(QFont("Consolas", 9))
                layout.addWidget(self.progress_text)
                
                self.cancel_button = QPushButton("Abbrechen")
                self.cancel_button.clicked.connect(self.reject)
                layout.addWidget(self.cancel_button)
                
                self.setLayout(layout)
                
            def add_progress_message(self, message: str):
                self.progress_text.append(message)
                # Auto-scroll to bottom
                scrollbar = self.progress_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                
            def set_status(self, status: str):
                self.status_label.setText(status)
        
        # Create and show progress dialog
        progress_dialog = ImportProgressDialog(self)
        progress_dialog.show()
        
        # Create and start worker thread
        self.import_worker = LobidImportWorker(force_download=force_download, debug=True)
        
        # Connect worker signals
        self.import_worker.progress_updated.connect(progress_dialog.add_progress_message)
        self.import_worker.progress_updated.connect(progress_dialog.set_status)
        
        def on_import_finished(entry_count):
            progress_dialog.cancel_button.setText("Schließen")
            progress_dialog.add_progress_message(f"\n🎉 Import erfolgreich abgeschlossen!")
            progress_dialog.set_status(f"✅ {entry_count:,} Einträge importiert")
            
            # Update cache statistics if available
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.update_status()
                
        def on_import_error(error_message):
            progress_dialog.cancel_button.setText("Schließen")
            progress_dialog.add_progress_message(f"\n❌ Fehler: {error_message}")
            progress_dialog.set_status("❌ Import fehlgeschlagen")
            
            QMessageBox.critical(self, "Import-Fehler", f"Fehler beim Import:\n{error_message}")
        
        self.import_worker.finished_successfully.connect(on_import_finished)
        self.import_worker.error_occurred.connect(on_import_error)
        
        # Handle cancel button
        def on_cancel():
            if self.import_worker.isRunning():
                progress_dialog.set_status("🛑 Import wird abgebrochen...")
                progress_dialog.add_progress_message("🛑 Benutzer hat Import abgebrochen")
                self.import_worker.terminate()
                self.import_worker.wait(3000)  # Wait max 3 seconds
            progress_dialog.accept()
            
        progress_dialog.rejected.connect(on_cancel)
        
        # Start the import
        self.import_worker.start()
        
        # Show dialog and wait for completion
        progress_dialog.exec()

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

        # Analyse-Zustand laden - Claude Generated
        load_state_action = file_menu.addAction("&Analyse-Zustand laden...")
        load_state_action.triggered.connect(self.load_analysis_state_from_file)

        # GUI-Zustand exportieren - Claude Generated
        export_gui_state_action = file_menu.addAction("&GUI-Zustand exportieren...")
        export_gui_state_action.triggered.connect(self.export_current_gui_state)

        # Analysis-States vergleichen - Claude Generated
        compare_states_action = file_menu.addAction("&Analysis-States vergleichen...")
        compare_states_action.triggered.connect(self.compare_analysis_states)

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
