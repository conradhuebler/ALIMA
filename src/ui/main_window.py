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
    QProgressBar,
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
from ..core.pipeline_manager import PipelineManager
from ..utils.config_manager import ConfigManager


from ..utils.config import Config
from .crossref_tab import CrossrefTab
from .analysis_review_tab import AnalysisReviewTab
from .ubsearch_tab import UBSearchTab
from .tablewidget import TableWidget, DatabaseViewerDialog
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

        self.config_manager = ConfigManager(logger=self.logger)

        # Check for first run and offer configuration import - Claude Generated
        self.check_first_run()

        # Load prompts path from config - Claude Generated
        config = self.config_manager.load_config()
        prompts_path = config.system_config.prompts_path

        # Instantiate core services with lazy initialization for faster GUI startup - Claude Generated
        self.llm_service = LlmService(
            config_manager=self.config_manager, # Pass config manager
            lazy_initialization=True,  # Don't test providers during GUI startup
        )
        self.llm = self.llm_service  # Assign llm here
        self.prompt_service = PromptService(prompts_path)
        self.alima_manager = AlimaManager(
            llm_service=self.llm_service,
            prompt_service=self.prompt_service,
            config_manager=self.config_manager, # Pass config manager
            logger=self.logger,  # Pass logger to manager
        )

        # Create central PipelineManager for all tabs - Claude Generated
        self.pipeline_manager = PipelineManager(
            alima_manager=self.alima_manager,
            cache_manager=self.cache_manager,
            logger=self.logger,
            config_manager=self.config_manager
        )

        self.available_models = {}
        self.available_providers = []
        self.gnd_import_worker = None  # Track GND import worker - Claude Generated

        self.init_ui()
        self.load_settings()

        # Check for pending GND import from first-start wizard - Claude Generated
        self.check_pending_gnd_import()

        # Setup reactive provider status connections - Claude Generated
        self.setup_provider_status_connections()

        # Don't load models during startup - use ProviderStatusService instead - Claude Generated
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
            alima_manager=self.alima_manager,
            pipeline_manager=self.pipeline_manager
        )

        self.crossref_tab = CrossrefTab()

        # Pass alima_manager and llm_service to AbstractTab
        # Pass alima_manager and llm_service to AbstractTab
        self.abstract_tab = AbstractTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            cache_manager=self.cache_manager,
            pipeline_manager=self.pipeline_manager,
            main_window=self,
        )
        # self.crossref_tab.result_abstract.connect(self.abstract_tab.set_abstract)
        # self.crossref_tab.result_keywords.connect(self.abstract_tab.set_keywords)
        self.abstract_tab.template_name = "abstract_analysis"  # This might be removed later if task selection is fully dynamic
        self.abstract_tab.set_task("initialisation")  # Set initial task (pipeline step name) - Claude Generated

        # Pass alima_manager and llm_service to AbstractTab
        self.analyse_keywords = AbstractTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            cache_manager=self.cache_manager,
            pipeline_manager=self.pipeline_manager,
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

        # TAB ISOLATION: Live-Analysen bleiben im jeweiligen Tab - Claude Generated
        # Pipeline-Ergebnisse werden weiterhin via on_pipeline_results_ready() verteilt
        # TODO: Implement "Update Pipeline"-Button for manual data transfer from tab to pipeline
        # self.abstract_tab.analysis_completed.connect(
        #     self.analysis_review_tab.receive_analysis_data
        # )

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
            pipeline_manager=self.pipeline_manager,
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

        # Connect pipeline results to specialized tabs - Claude Generated
        # Central distribution via on_pipeline_results_ready slot
        self.pipeline_tab.pipeline_results_ready.connect(self.on_pipeline_results_ready)

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

    def get_provider_info(self):
        """Get cached provider information from ProviderStatusService - Claude Generated"""
        if hasattr(self.alima_manager, 'provider_status_service') and self.alima_manager.provider_status_service:
            return self.alima_manager.provider_status_service.get_all_provider_info()
        else:
            # Fallback if service not available
            self.logger.warning("ProviderStatusService not available, returning empty provider info")
            return {}

    def get_available_providers(self):
        """Get list of available providers from cached status - Claude Generated"""
        provider_info = self.get_provider_info()
        return [name for name, info in provider_info.items() if info.get('reachable', False)]

    def get_available_models(self, provider_name: str):
        """Get available models for a provider from cached status - Claude Generated"""
        if hasattr(self.alima_manager, 'provider_status_service') and self.alima_manager.provider_status_service:
            return self.alima_manager.provider_status_service.get_available_models(provider_name)
        else:
            return []

    def update_tabs_with_provider_info(self):
        """Update tabs with current provider information - Claude Generated"""
        try:
            provider_info = self.get_provider_info()
            available_providers = self.get_available_providers()

            # Build available_models dict for backward compatibility
            available_models = {}
            for provider_name in available_providers:
                available_models[provider_name] = self.get_available_models(provider_name)

            # Update tabs with current provider info
            if hasattr(self, 'abstract_tab'):
                self.abstract_tab.set_models_and_providers(available_models, available_providers)
            if hasattr(self, 'analyse_keywords'):
                self.analyse_keywords.set_models_and_providers(available_models, available_providers)
            if hasattr(self, 'ub_search_tab'):
                self.ub_search_tab.set_models_and_providers(available_models, available_providers)

            self.logger.info(f"Updated tabs with {len(available_providers)} providers")

        except Exception as e:
            self.logger.error(f"Error updating tabs with provider info: {e}")

    def setup_provider_status_connections(self):
        """Connect to ProviderStatusService signals for reactive updates - Claude Generated"""
        if hasattr(self.alima_manager, 'provider_status_service') and self.alima_manager.provider_status_service:
            # Connect to status updates for automatic UI refresh
            self.alima_manager.provider_status_service.status_updated.connect(
                self.update_tabs_with_provider_info
            )
            self.logger.info("Connected to ProviderStatusService signals")
        else:
            self.logger.warning("ProviderStatusService not available for signal connections")

        # REMOVED: Central Ollama signal management - hardcoded connections caused app hangs - Claude Generated
        # Signal connections removed to prevent deadlock/hang issues
        # Ollama provider changes now handled through unified provider system
        self.logger.info("Ollama signal connections disabled to prevent app hangs")

    @pyqtSlot(object)
    def on_pipeline_results_ready(self, analysis_state):
        """Central slot for distributing pipeline results to specialized tabs - Claude Generated"""
        self.logger.info("Distributing pipeline results to specialized tabs")

        # 1. Abstract-Analyse Tab: Set abstract + initial keywords + LLM response
        if analysis_state.original_abstract:
            self.abstract_tab.set_abstract(analysis_state.original_abstract)

        if analysis_state.initial_keywords:
            # Robust type handling for keywords - Claude Generated
            if isinstance(analysis_state.initial_keywords, list):
                keywords_str = ", ".join(analysis_state.initial_keywords)
            else:
                # Handle case where it's already a string
                keywords_str = str(analysis_state.initial_keywords)
            self.abstract_tab.set_keywords(keywords_str)

        if hasattr(analysis_state, 'initial_llm_call_details') and analysis_state.initial_llm_call_details:
            llm_response = analysis_state.initial_llm_call_details.response_full_text
            if llm_response:
                self.abstract_tab.display_llm_response(llm_response)

        # 2. GND-Suche Tab: Set initial keywords + display search results
        if analysis_state.initial_keywords:
            # Type-safe join - Claude Generated (Fix for string parsing bug)
            if isinstance(analysis_state.initial_keywords, list):
                keywords_str = ", ".join(analysis_state.initial_keywords)
            else:
                keywords_str = str(analysis_state.initial_keywords)
            self.search_tab.update_search_field(keywords_str)

        if analysis_state.search_results:
            # Handle both Dict and List[SearchResult] formats - Claude Generated
            if isinstance(analysis_state.search_results, dict):
                # Already in correct format
                self.search_tab.display_search_results(analysis_state.search_results)
            else:
                # Convert List[SearchResult] to Dict format
                search_results_dict = {
                    sr.search_term: sr.results
                    for sr in analysis_state.search_results
                }
                self.search_tab.display_search_results(search_results_dict)

        # 3. Verifikation Tab (analyse_keywords): Set abstract + GND keywords + final LLM response
        if analysis_state.original_abstract:
            self.analyse_keywords.set_abstract(analysis_state.original_abstract)

        # Extract GND keywords from search_results
        if analysis_state.search_results:
            gnd_keywords = []
            # Handle both Dict and List[SearchResult] formats - Claude Generated
            if isinstance(analysis_state.search_results, dict):
                # Dict format: {search_term: {keyword: data}}
                for results in analysis_state.search_results.values():
                    for keyword, data in results.items():
                        gnd_ids = data.get("gndid", set())
                        for gnd_id in gnd_ids:
                            gnd_keywords.append(f"{keyword} (GND-ID: {gnd_id})")
            else:
                # List[SearchResult] format
                for search_result in analysis_state.search_results:
                    for keyword, data in search_result.results.items():
                        gnd_ids = data.get("gndid", set())
                        for gnd_id in gnd_ids:
                            gnd_keywords.append(f"{keyword} (GND-ID: {gnd_id})")
            if gnd_keywords:
                self.analyse_keywords.set_keywords("\n".join(gnd_keywords))

        # Display final LLM analysis if available
        if hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
            if hasattr(analysis_state.final_llm_analysis, 'response_full_text'):
                self.analyse_keywords.display_llm_response(
                    analysis_state.final_llm_analysis.response_full_text
                )

        # 4. UB Suche Tab: Set final extracted GND keywords
        if hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
            if hasattr(analysis_state.final_llm_analysis, 'extracted_gnd_keywords'):
                final_keywords = analysis_state.final_llm_analysis.extracted_gnd_keywords
                if final_keywords:
                    # Format keywords properly for UB search
                    if isinstance(final_keywords, list):
                        keywords_text = "\n".join(final_keywords)
                    else:
                        keywords_text = str(final_keywords)
                    self.ub_search_tab.update_keywords(keywords_text)

        # 5. 📊 Analyse-Review Tab - Sende finale Ergebnisse
        # Pass data if we have EITHER keywords analysis OR DK classifications - Claude Generated
        if analysis_state.final_llm_analysis or analysis_state.dk_classifications:
            # Extract keywords if available
            keywords_text = ""
            analysis_result = ""
            if analysis_state.final_llm_analysis:
                # Type-safe join - Claude Generated (Fix for string parsing bug)
                final_kw = analysis_state.final_llm_analysis.extracted_gnd_keywords
                keywords_text = ", ".join(final_kw) if isinstance(final_kw, list) else str(final_kw)
                analysis_result = analysis_state.final_llm_analysis.response_full_text

            self.analysis_review_tab.receive_analysis_data(
                abstract_text=analysis_state.original_abstract or "",
                keywords=keywords_text,
                analysis_result=analysis_result,
                dk_classifications=analysis_state.dk_classifications,
                dk_search_results=analysis_state.dk_search_results
            )
            self.logger.info("✅ Analysis review tab populated with pipeline results (keywords and/or DK).")

        # 6. 📚 DK-Klassifikation (Optional) - Claude Generated
        # Note: DK search results and classifications are handled by show_loaded_state_indicator()
        # when called from populate_all_tabs_from_state(). For live pipeline results, they are
        # displayed directly in pipeline_tab via on_step_completed() callbacks.
        if analysis_state.dk_search_results:
            self.logger.info(f"DK search results available: {len(analysis_state.dk_search_results)} entries")
        if analysis_state.dk_classifications:
            self.logger.info(f"DK classifications available: {len(analysis_state.dk_classifications)} entries")

        self.logger.info("Pipeline results successfully distributed to all tabs")

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
            self.logger.error(f"Fehler beim Extrahieren der GND-Terme: {str(e)}")
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
            dialog = ComprehensiveSettingsDialog(alima_manager=self.alima_manager, parent=self)
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

    def save_settings(self):
        """Speichert die aktuellen Einstellungen"""
        self.settings.setValue("geometry", self.saveGeometry())

    def _on_config_changed(self):
        """Handle configuration changes from comprehensive settings dialog - Claude Generated"""
        self.logger.info("Configuration changed, refreshing components")
        self._refresh_components()

    def _refresh_components(self):
        """Refresh all components with new configuration - Claude Generated"""
        try:
            self.logger.info("Starting component refresh after configuration change...")

            # 1. Reload LLM service configuration and reinitialize providers
            if hasattr(self, 'llm_service'):
                self.logger.info("Reloading LLM service providers...")
                self.llm_service.reload_providers()
                self.logger.info("✓ LLM service providers reloaded")

            # 2. Refresh provider status to update reachability and available models
            if hasattr(self, 'llm_service'):
                self.logger.info("Refreshing provider status...")
                self.llm_service.refresh_all_provider_status()
                self.logger.info("✓ Provider status refreshed")

            # 3. Reload pipeline configuration to use updated provider preferences
            if hasattr(self, 'pipeline_manager'):
                self.logger.info("Reloading pipeline configuration...")
                self.pipeline_manager.reload_config()
                self.logger.info("✓ Pipeline configuration reloaded")

            # 4. Update tabs with new provider information
            self.logger.info("Updating tabs with provider information...")
            self.update_tabs_with_provider_info()
            self.logger.info("✓ Tabs updated with provider info")

            # 5. Update global status bar with new provider and cache info
            if hasattr(self, 'global_status_bar'):
                self.logger.info("Updating global status bar...")
                self.global_status_bar.update_provider_info()
                self.global_status_bar.update_cache_status()
                self.logger.info("✓ Global status bar updated")

            # 6. Notify tabs about configuration changes (custom handlers)
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                if hasattr(tab, 'on_config_changed'):
                    tab_name = self.tabs.tabText(i)
                    self.logger.info(f"Notifying tab '{tab_name}' about config change...")
                    tab.on_config_changed()

            self.logger.info("✅ Component refresh completed successfully - configuration is now active")

            # Show user feedback
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.show_temporary_message(
                    "✅ Einstellungen erfolgreich übernommen",
                    5000
                )

        except Exception as e:
            self.logger.error(f"Error refreshing components: {e}", exc_info=True)
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.show_temporary_message(
                    f"⚠️ Fehler beim Übernehmen der Einstellungen: {str(e)}",
                    10000
                )

    # Claude Generated - DELETED: export_results() and export_current_analysis()
    # export_results: Inconsistent tab-specific export removed
    # export_current_analysis: Obsolete - now use analysis_review_tab.current_analysis directly

    def import_results(self):
        """Importiert gespeicherte Suchergebnisse"""
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, "import_results"):
            current_tab.import_results()
        else:
            self.global_status_bar.show_temporary_message("Import nicht verfügbar für diesen Tab", 3000)

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

    def show_database_viewer(self):
        """Open database viewer dialog - Claude Generated"""
        try:
            # Get current database configuration
            config_manager = ConfigManager()
            database_config = config_manager.get_database_config()

            # Create and execute modal dialog (automatic memory management)
            dialog = DatabaseViewerDialog(database_config, self)
            dialog.exec()  # Modal dialog with automatic cleanup

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Öffnen des Datenbank-Viewers:\n{str(e)}"
            )

    def clear_search_cache(self):
        """Clear search mappings cache (with confirmation) - Claude Generated"""
        try:
            # Confirmation dialog
            reply = QMessageBox.question(
                self,
                "Such-Cache leeren?",
                "Alle gespeicherten Suchergebnisse werden gelöscht.\n\n"
                "GND-Einträge und Klassifikationen bleiben erhalten.\n\n"
                "Fortfahren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No  # Default to No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Get singleton instance and clear cache - Claude Generated (Enhanced error handling)
                try:
                    from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
                    uk_manager = UnifiedKnowledgeManager.get_instance()
                    success, message = uk_manager.clear_search_cache()

                    if success:
                        QMessageBox.information(
                            self,
                            "Erfolg",
                            message
                        )
                        self.logger.info(f"Search cache cleared successfully: {message}")
                    else:
                        QMessageBox.warning(
                            self,
                            "Warnung",
                            message
                        )
                        self.logger.warning(f"Cache clear warning: {message}")

                except Exception as cache_error:
                    # FIX: Better error handling for cache clear operations - Claude Generated
                    error_msg = f"Fehler beim Leeren des Caches:\n{str(cache_error)}\n\nVersuchen Sie später erneut."
                    QMessageBox.critical(self, "Fehler", error_msg)
                    self.logger.error(f"Cache clear failed with exception: {cache_error}", exc_info=True)

        except Exception as e:
            # FIX: Catch outer exceptions (dialog, etc.) - Claude Generated
            error_msg = f"Unerwarteter Fehler:\n{str(e)}"
            QMessageBox.critical(self, "Fehler", error_msg)
            self.logger.error(f"Unexpected error in clear_search_cache: {e}", exc_info=True)

    def cleanup_malformed_entries(self):
        """Clean up malformed classification entries (count>0 but no titles) - Claude Generated (Ultra-Deep Fix)"""
        try:
            # Confirmation dialog
            reply = QMessageBox.question(
                self,
                "Malformed Einträge bereinigen?",
                "Entfernt DK-Einträge mit count>0 aber ohne Titel.\n\n"
                "Diese Einträge können Live-Suchen blockieren.\n\n"
                "Fortfahren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No  # Default to No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Get singleton instance and cleanup - Claude Generated
                try:
                    from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
                    uk_manager = UnifiedKnowledgeManager.get_instance()
                    success, message = uk_manager.cleanup_malformed_classifications()

                    if success:
                        QMessageBox.information(
                            self,
                            "Erfolg",
                            message
                        )
                        self.logger.info(f"Malformed entries cleaned: {message}")
                    else:
                        QMessageBox.warning(
                            self,
                            "Warnung",
                            message
                        )
                        self.logger.warning(f"Cleanup warning: {message}")

                except Exception as cleanup_error:
                    error_msg = f"Fehler beim Bereinigen:\n{str(cleanup_error)}\n\nVersuchen Sie später erneut."
                    QMessageBox.critical(self, "Fehler", error_msg)
                    self.logger.error(f"Cleanup failed with exception: {cleanup_error}", exc_info=True)

        except Exception as e:
            error_msg = f"Unerwarteter Fehler:\n{str(e)}"
            QMessageBox.critical(self, "Fehler", error_msg)
            self.logger.error(f"Unexpected error in cleanup_malformed_entries: {e}", exc_info=True)

    def show_batch_processing_dialog(self):
        """Open batch processing dialog - Claude Generated"""
        try:
            from .batch_processing_dialog import BatchProcessingDialog

            dialog = BatchProcessingDialog(
                alima_manager=self.alima_manager,
                cache_manager=self.cache_manager,
                config_manager=self.config_manager,
                logger=self.logger,
                parent=self
            )
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Öffnen des Batch Processing Dialogs:\n{str(e)}"
            )
            self.logger.error(f"Failed to open batch processing dialog: {e}")

    def load_batch_results(self):
        """Load and review batch processing results - Claude Generated"""
        try:
            # Ask user to select output directory
            directory = QFileDialog.getExistingDirectory(
                self,
                "Batch-Ergebnisse laden",
                "",
                QFileDialog.Option.ShowDirsOnly
            )

            if not directory:
                return

            # Find all JSON files in directory
            from pathlib import Path
            json_files = list(Path(directory).glob("*.json"))

            # Filter out the .batch_state.json file
            json_files = [f for f in json_files if f.name != ".batch_state.json"]

            if not json_files:
                QMessageBox.warning(
                    self,
                    "Keine Ergebnisse",
                    f"Keine JSON-Dateien in {directory} gefunden."
                )
                return

            # Switch to analysis review tab
            for i in range(self.tabs.count()):
                if isinstance(self.tabs.widget(i), type(self.analysis_review_tab)):
                    self.tabs.setCurrentIndex(i)
                    break

            # Load batch directory into analysis review tab - Claude Generated
            self.analysis_review_tab.load_batch_directory(directory)

            self.update_status(f"Loaded {len(json_files)} batch result(s) from {directory}")

            QMessageBox.information(
                self,
                "Batch-Ergebnisse geladen",
                f"Geladen: {len(json_files)} Ergebnis-Dateien aus {directory}\n\n"
                f"Verwenden Sie die Batch-Tabelle um einzelne Ergebnisse anzuzeigen."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Laden der Batch-Ergebnisse:\n{str(e)}"
            )
            self.logger.error(f"Failed to load batch results: {e}")

    def closeEvent(self, event):
        """Wird beim Schließen des Fensters aufgerufen"""
        self.save_settings()
        event.accept()

    def update_status(self, message: str):
        """Aktualisiert die Statusleiste - Claude Generated (Fixed AttributeError)"""
        self.global_status_bar.show_temporary_message(message, 3000)

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
                    self.logger.info("🔄 Starte GND-Datenbank Import...")
                    self.logger.info(f"📁 Datei: {xml_file_path}")

                    # Connect parser progress signals if available
                    if hasattr(parser, "progress_updated"):
                        parser.progress_updated.connect(progress.setValue)
                        parser.status_updated.connect(progress.setLabelText)

                        # Also connect to console output - Claude Generated
                        def console_progress(value):
                            if value > 0:
                                self.logger.info(f"📊 Fortschritt: {value}%")

                        def console_status(status):
                            self.logger.info(f"ℹ️ Status: {status}")

                        parser.progress_updated.connect(console_progress)
                        parser.status_updated.connect(console_status)

                    self.logger.info("⚙️ Verarbeite XML-Daten...")
                    parser.process_file(xml_file_path)
                    self.logger.info("✅ GND-Import erfolgreich abgeschlossen!")

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
            self.global_status_bar.show_temporary_message("✅ Analyse-Zustand erfolgreich geladen.", 5000)

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

        Uses on_pipeline_results_ready() for core distribution to avoid duplication.
        """
        from ..core.data_models import KeywordAnalysisState

        self.logger.info("Distributing analysis state data to all tabs...")

        try:
            # Use the centralized distribution logic for base tabs - Claude Generated
            self.on_pipeline_results_ready(state)

            # Add loaded analysis to tab-local history - Claude Generated
            self.abstract_tab.add_external_analysis_to_history(state)
            self.analyse_keywords.add_external_analysis_to_history(state)
            self.logger.info("Added loaded analysis to tab histories")

            # Additional loading-specific UI enhancements - Claude Generated

            # 1. 🚀 Pipeline Tab - Show loaded state indicators
            if hasattr(self.pipeline_tab, 'unified_input') and state.original_abstract:
                self.pipeline_tab.unified_input.set_text_directly(
                    state.original_abstract,
                    "Geladen aus JSON"
                )
                if hasattr(self.pipeline_tab, 'show_loaded_state_indicator'):
                    self.pipeline_tab.show_loaded_state_indicator(state)
                self.logger.info("✅ Pipeline tab: loaded state indicators shown")

            # 5. 📊 Analyse-Review Tab - Complete results and export
            if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                # Type-safe join - Claude Generated (Fix for string parsing bug)
                final_kw = state.final_llm_analysis.extracted_gnd_keywords
                final_keywords = ", ".join(final_kw) if isinstance(final_kw, list) else str(final_kw)
                full_response = state.final_llm_analysis.response_full_text

                self.analysis_review_tab.receive_analysis_data(
                    state.original_abstract or "",
                    final_keywords,
                    full_response,
                    state.dk_classifications,
                    state.dk_search_results
                )
                self.logger.info("✅ Analysis review tab populated with final results")

            # 6. 🏛️ UB Suche Tab - Keywords for library catalog search
            if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                # Type-safe join - Claude Generated (Fix for string parsing bug)
                final_kw = state.final_llm_analysis.extracted_gnd_keywords
                final_keywords = ", ".join(final_kw) if isinstance(final_kw, list) else str(final_kw)
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
            self.global_status_bar.show_temporary_message(summary_message, 10000)

            self.logger.info(f"🎯 Analysis state distribution complete: {summary_message}")

        except Exception as e:
            self.logger.error(f"Error distributing analysis state: {e}")
            raise  # Re-raise to be handled by calling method

    # Claude Generated - DELETED: collect_current_gui_state()
    # This UI-scraping method is now obsolete and removed.
    # Reason: Single Source of Truth architecture - we now use analysis_review_tab.current_analysis directly.
    # All export operations now access the canonical data object instead of scraping UI elements.

    def export_current_gui_state(self):
        """
        Exportiert den aktuellen Analyse-Zustand aus dem Analyse-Review-Tab.
        Claude Generated - Refactored to use Single Source of Truth (analysis_review_tab.current_analysis)
        """
        from ..utils.pipeline_utils import AnalysisPersistence

        # 1. Prüfe, ob der Analyse-Review-Tab ein gültiges Ergebnis hat
        if self.analysis_review_tab and self.analysis_review_tab.current_analysis:
            # 2. Das "echte" Datenobjekt direkt holen (Single Source of Truth)
            state_to_save = self.analysis_review_tab.current_analysis
            self.logger.info("Exporting current analysis state from AnalysisReviewTab.")

            # 3. Den bewährten Speicher-Dialog mit dem korrekten Objekt aufrufen
            file_path = AnalysisPersistence.save_with_dialog(
                state=state_to_save,
                parent_widget=self,
                default_filename=f"analysis_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            if file_path:
                self.global_status_bar.show_temporary_message("✅ Analyse-Zustand erfolgreich exportiert.", 5000)
                self.logger.info(f"Analysis state successfully exported to: {file_path}")
        else:
            # 4. Fehlerbehandlung, wenn keine Daten zum Speichern da sind
            self.logger.warning("Export triggered, but no analysis state available in Review-Tab.")
            QMessageBox.information(
                self,
                "Keine Daten zum Speichern",
                "Es ist keine abgeschlossene Analyse vorhanden, die gespeichert werden könnte.\n\n"
                "Bitte führen Sie zuerst eine Analyse durch oder laden Sie ein Ergebnis in den 'Analyse-Review'-Tab."
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
            self.logger.info("🌐 Starte DNB-Download...")
            self.logger.info(f"📡 URL: {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                self.logger.info(f"📦 Dateigröße: {total_size / (1024*1024):.1f} MB")

            # Create temporary files
            temp_dir = tempfile.mkdtemp()
            temp_gz_path = os.path.join(temp_dir, "gnd_data.xml.gz")
            temp_xml_path = os.path.join(temp_dir, "gnd_data.xml")

            # Download with progress
            downloaded = 0
            last_console_percent = 0

            self.logger.info("⬇️ Download läuft...")
            with open(temp_gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if progress.wasCanceled():
                        self.logger.info("❌ Download abgebrochen")
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
                            self.logger.info(f"📊 Download: {console_percent:.0f}%")
                            last_console_percent = console_percent

                    QApplication.processEvents()

            progress.setLabelText("Entpacke Datenbank...")
            progress.setValue(50)
            QApplication.processEvents()

            # Extract gz file
            self.logger.info("📦 Entpacke GZ-Datei...")
            self.logger.info(f"Entpacke {temp_gz_path} nach {temp_xml_path}")
            with gzip.open(temp_gz_path, "rb") as gz_file:
                with open(temp_xml_path, "wb") as xml_file:
                    xml_file.write(gz_file.read())

            self.logger.info("✅ Download und Entpackung abgeschlossen")
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

    def import_configuration(self):
        """Öffnet den Konfigurations-Import-Dialog - Claude Generated"""
        try:
            from .import_config_dialog import ImportConfigDialog

            dialog = ImportConfigDialog(
                config_manager=self.config_manager,
                parent=self
            )

            if dialog.exec():
                # Erfolgreicher Import
                QMessageBox.information(
                    self,
                    "Import erfolgreich",
                    "Konfiguration wurde erfolgreich importiert.\n\n"
                    "Die Anwendung wird neu gestartet, um die neuen Einstellungen zu laden."
                )

                # Restart application
                self.restart_application()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Importieren der Konfiguration:\n{str(e)}"
            )
            self.logger.error(f"Configuration import error: {e}", exc_info=True)

    def export_configuration(self):
        """Exportiert die aktuelle Konfiguration - Claude Generated"""
        try:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Konfiguration exportieren",
                "",
                QFileDialog.Option.ShowDirsOnly
            )

            if not directory:
                return

            success, message = self.config_manager.export_configuration(directory)

            if success:
                QMessageBox.information(
                    self,
                    "Export erfolgreich",
                    f"Konfiguration erfolgreich exportiert nach:\n{directory}"
                )
                self.logger.info(f"Configuration exported to: {directory}")
            else:
                QMessageBox.warning(self, "Export fehlgeschlagen", message)
                self.logger.warning(f"Configuration export failed: {message}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Exportieren:\n{str(e)}"
            )
            self.logger.error(f"Configuration export error: {e}", exc_info=True)

    def restart_application(self):
        """Startet ALIMA neu - Claude Generated"""
        try:
            self.logger.info("Restarting application...")
            self.save_settings()

            # Restart Python process
            python = sys.executable
            os.execl(python, python, *sys.argv)

        except Exception as e:
            self.logger.error(f"Application restart failed: {e}")
            QMessageBox.warning(
                self,
                "Neustart fehlgeschlagen",
                "Die Anwendung konnte nicht automatisch neu gestartet werden.\n"
                "Bitte starten Sie ALIMA manuell neu."
            )

    def check_first_run(self):
        """Prüft ob dies der erste Start ist und bietet Config-Import an - Claude Generated"""
        try:
            config = self.config_manager.load_config()

            # Check if first-run check is disabled via config flag - Claude Generated
            if config.system_config.skip_first_run_check:
                self.logger.info("✓ First-run check disabled via config flag (skip_first_run_check=true)")
                return

            # Prüfe ob Config "leer" ist
            is_empty_config = self._is_empty_config(config)

            if is_empty_config:
                self.logger.info("🔔 First run detected - showing config import dialog")
                self.show_first_run_dialog()
            else:
                self.logger.info("✓ Configuration appears valid, skipping first-run dialog")

        except Exception as e:
            self.logger.warning(f"⚠️ First-run check failed: {e}", exc_info=True)

    def _is_empty_config(self, config) -> bool:
        """Prüft ob Config leer ist - Claude Generated"""

        # Prüfung 0: config.json existiert im Config-Verzeichnis? - Claude Generated
        config_file_exists = self.config_manager.config_file.exists()
        self.logger.info(f"  🔍 First-run check - config.json exists: {config_file_exists} (path={self.config_manager.config_file})")

        if not config_file_exists:
            self.logger.info("  ✗ config.json missing - definite first-run condition")
            return True  # Definit First-Run

        # Prüfung 1: Keine Provider konfiguriert
        no_providers = len(config.unified_config.providers) == 0
        self.logger.info(f"  🔍 no_providers: {no_providers} (provider_count={len(config.unified_config.providers)})")

        # Prüfung 2: prompts.json existiert nicht
        prompts_missing = not Path(config.system_config.prompts_path).exists()
        self.logger.info(f"  🔍 prompts_missing: {prompts_missing} (path={config.system_config.prompts_path})")

        # Prüfung 3: Datenbank existiert nicht
        # UNIFIED: Use database_config.sqlite_path as single source of truth - Claude Generated
        db_missing = not Path(config.database_config.sqlite_path).exists()
        self.logger.info(f"  🔍 db_missing: {db_missing} (path={config.database_config.sqlite_path})")

        # First-Run wenn mindestens 2 von 3 fehlen
        missing_count = sum([no_providers, prompts_missing, db_missing])
        self.logger.info(f"  🔍 missing_count: {missing_count}/3")

        is_empty = missing_count >= 2
        if is_empty:
            self.logger.info("  ✗ Configuration is empty - first-run condition detected")
        else:
            self.logger.info("  ✓ Configuration appears valid")

        return is_empty

    def show_first_run_dialog(self):
        """Zeigt First-Run Dialog mit Import-Option - Claude Generated"""

        reply = QMessageBox.question(
            self,
            "🎉 Willkommen bei ALIMA!",
            "Dies scheint der erste Start von ALIMA zu sein.\n\n"
            "Möchten Sie eine bestehende Konfiguration importieren?\n\n"
            "• JA: Bestehende Konfiguration aus einem Verzeichnis importieren\n"
            "• NEIN: Mit der Standardkonfiguration starten",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Öffne Import-Dialog
            self.import_configuration()

    def check_pending_gnd_import(self):
        """Check if GND was downloaded in wizard and offer background import - Claude Generated"""
        import json
        from pathlib import Path

        marker_file = Path(tempfile.gettempdir()) / "alima_gnd_pending.json"

        if not marker_file.exists():
            return

        try:
            marker_data = json.loads(marker_file.read_text())
            xml_path = marker_data.get('xml_path')

            if not xml_path or not Path(xml_path).exists():
                marker_file.unlink()
                return

            # Ask user if they want to start background import
            reply = QMessageBox.question(
                self,
                "GND-Import",
                "Eine GND-Datenbank wurde im Setup heruntergeladen.\n\n"
                "Möchten Sie den Import jetzt im Hintergrund starten?\n"
                "(Sie können währenddessen weiterarbeiten)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._start_background_gnd_import(xml_path)

            # Remove marker regardless of choice
            marker_file.unlink()

        except Exception as e:
            self.logger.error(f"Error checking pending GND import: {str(e)}")
            if marker_file.exists():
                marker_file.unlink()

    def _start_background_gnd_import(self, xml_file_path: str):
        """Start non-blocking GND import in background (StatusBar-based) - Claude Generated"""
        try:
            from ..utils.gnd_import_worker import GNDImportWorker

            self.logger.info(f"Starting background GND import: {xml_file_path}")

            # Create status bar widgets - Claude Generated (non-blocking UI)
            self.gnd_import_status_label = QLabel("🔄 GND-Import: Starte...")

            self.gnd_import_progress_bar = QProgressBar()
            self.gnd_import_progress_bar.setMaximum(100)
            self.gnd_import_progress_bar.setMinimumWidth(200)
            self.gnd_import_progress_bar.setMaximumHeight(20)

            self.gnd_import_cancel_btn = QPushButton("Abbrechen")
            self.gnd_import_cancel_btn.setMaximumWidth(80)
            self.gnd_import_cancel_btn.setMaximumHeight(20)

            # Add to status bar
            statusbar = self.statusBar()
            statusbar.addWidget(self.gnd_import_status_label)
            statusbar.addWidget(self.gnd_import_progress_bar)
            statusbar.addWidget(self.gnd_import_cancel_btn)

            # Create worker
            self.gnd_import_worker = GNDImportWorker(xml_file_path, self.cache_manager)

            # Connect signals
            self.gnd_import_worker.progress_updated.connect(self._update_gnd_progress)
            self.gnd_import_worker.status_updated.connect(self._update_gnd_status)
            self.gnd_import_worker.finished_successfully.connect(self._on_gnd_import_complete)
            self.gnd_import_worker.error_occurred.connect(self._on_gnd_import_error)

            # Connect cancel button
            self.gnd_import_cancel_btn.clicked.connect(self.gnd_import_worker.cancel)

            # Show notification
            QMessageBox.information(
                self,
                "GND-Import gestartet",
                "✅ GND-Import läuft im Hintergrund.\n\n"
                "Sie können ALIMA normal nutzen.\n"
                "Fortschritt wird in der Statusleiste angezeigt.",
                QMessageBox.StandardButton.Ok
            )

            # Start worker
            self.gnd_import_worker.start()

        except Exception as e:
            self.logger.error(f"Error starting GND import: {str(e)}", exc_info=True)
            QMessageBox.warning(
                self,
                "GND-Import Fehler",
                f"Fehler beim Starten des GND-Imports:\n{str(e)}"
            )

    def _update_gnd_progress(self, current: int, total: int):
        """Update GND import progress bar - Claude Generated"""
        if total > 0 and hasattr(self, 'gnd_import_progress_bar'):
            percent = int((current / total) * 100)
            self.gnd_import_progress_bar.setValue(percent)
            if hasattr(self, 'gnd_import_status_label'):
                self.gnd_import_status_label.setText(
                    f"🔄 GND-Import: {current:,} / {total:,} ({percent}%)"
                )

    def _update_gnd_status(self, status_msg: str):
        """Update GND import status message - Claude Generated"""
        if hasattr(self, 'gnd_import_status_label'):
            self.gnd_import_status_label.setText(f"🔄 {status_msg}")

    def _on_gnd_import_complete(self, count: int):
        """Handle successful GND import completion - Claude Generated"""
        self.logger.info(f"GND import completed successfully: {count:,} entries")

        # Remove status bar widgets - Claude Generated
        statusbar = self.statusBar()
        if hasattr(self, 'gnd_import_status_label'):
            statusbar.removeWidget(self.gnd_import_status_label)
            statusbar.removeWidget(self.gnd_import_progress_bar)
            statusbar.removeWidget(self.gnd_import_cancel_btn)

            # Cleanup references
            self.gnd_import_status_label.deleteLater()
            self.gnd_import_progress_bar.deleteLater()
            self.gnd_import_cancel_btn.deleteLater()

        # Show success message in status bar
        statusbar.showMessage(
            f"✅ GND-Import abgeschlossen: {count:,} Einträge importiert",
            10000  # Show for 10 seconds
        )

    def _on_gnd_import_error(self, error_msg: str):
        """Handle GND import error - Claude Generated"""
        self.logger.error(f"GND import failed: {error_msg}")

        # Remove status bar widgets - Claude Generated
        statusbar = self.statusBar()
        if hasattr(self, 'gnd_import_status_label'):
            statusbar.removeWidget(self.gnd_import_status_label)
            statusbar.removeWidget(self.gnd_import_progress_bar)
            statusbar.removeWidget(self.gnd_import_cancel_btn)

            # Cleanup references
            self.gnd_import_status_label.deleteLater()
            self.gnd_import_progress_bar.deleteLater()
            self.gnd_import_cancel_btn.deleteLater()

        # Show error in status bar
        statusbar.showMessage(f"❌ GND-Import fehlgeschlagen: {error_msg}", 15000)

    # In der MainWindow Klasse - füge folgende Methoden hinzu

    def create_menu_bar(self):
        """Erstellt die Menüleiste - Claude Generated (Reorganized)"""
        menubar = self.menuBar()

        # ========== Datei-Menü (Workflow-fokussiert) ==========
        file_menu = menubar.addMenu("&Datei")

        # Analyse-Zustand laden - Claude Generated
        load_state_action = file_menu.addAction("📂 &Analyse-Zustand laden...")
        load_state_action.triggered.connect(self.load_analysis_state_from_file)

        # Analyse-Zustand speichern - Claude Generated (Refactored to use unified persistence)
        save_state_action = file_menu.addAction("💾 Analyse-Zustand &speichern...")
        save_state_action.triggered.connect(self.export_current_gui_state)

        file_menu.addSeparator()

        # Konfiguration importieren - Claude Generated
        import_config_action = file_menu.addAction("📥 &Konfiguration importieren...")
        import_config_action.triggered.connect(self.import_configuration)

        # Konfiguration exportieren - Claude Generated
        export_config_action = file_menu.addAction("📤 Konfiguration &exportieren...")
        export_config_action.triggered.connect(self.export_configuration)

        file_menu.addSeparator()

        # Beenden-Aktion
        exit_action = file_menu.addAction("🚪 &Beenden")
        exit_action.triggered.connect(self.close)

        # ========== Extras/Tools-Menü (Datenbank und Debug) ==========
        tools_menu = menubar.addMenu("E&xtras")

        # GND-Datenbank importieren (moved from Datei)
        import_action = tools_menu.addAction("📥 &GND-Datenbank importieren...")
        import_action.triggered.connect(self.import_gnd_database)

        # Database viewer action - Claude Generated
        db_viewer_action = tools_menu.addAction("📊 &Datenbank-Viewer")
        db_viewer_action.triggered.connect(self.show_database_viewer)

        # Clear search cache action - Claude Generated
        clear_cache_action = tools_menu.addAction("🗑️ Such-&Cache leeren...")
        clear_cache_action.triggered.connect(self.clear_search_cache)

        # Cleanup malformed entries action - Claude Generated (Ultra-Deep Fix)
        cleanup_action = tools_menu.addAction("🧹 &Malformed Einträge bereinigen...")
        cleanup_action.triggered.connect(self.cleanup_malformed_entries)

        tools_menu.addSeparator()

        # Batch processing actions - Claude Generated
        batch_process_action = tools_menu.addAction("📦 &Batch Processing...")
        batch_process_action.triggered.connect(self.show_batch_processing_dialog)

        batch_review_action = tools_menu.addAction("📋 Batch-Ergebnisse &laden...")
        batch_review_action.triggered.connect(self.load_batch_results)

        tools_menu.addSeparator()

        # Analysis-States vergleichen - Claude Generated
        compare_states_action = tools_menu.addAction("🔍 Analysis-States &vergleichen...")
        compare_states_action.triggered.connect(self.compare_analysis_states)

        # GUI-Zustand exportieren - Claude Generated (marked as debug)
        export_gui_state_action = tools_menu.addAction("🐛 GUI-Debug: Zustand &exportieren...")
        export_gui_state_action.triggered.connect(self.export_current_gui_state)

        # ========== Bearbeiten-Menü ==========
        edit_menu = menubar.addMenu("&Bearbeiten")

        # Einstellungen-Aktion
        settings_action = edit_menu.addAction("⚙️ &Einstellungen")
        settings_action.triggered.connect(self.show_settings)

        # Prompt-Konfiguration-Aktion
        prompt_config_action = edit_menu.addAction("📝 &Prompt-Konfiguration")
        prompt_config_action.triggered.connect(self.show_prompt_editor)

        # ========== Update-Menü ==========
        update_menu = menubar.addMenu("&Updates")

        # Nach Updates suchen
        check_update_action = update_menu.addAction("🔄 Nach &Updates suchen")
        check_update_action.triggered.connect(self.check_for_updates)

        # NEUE OPTION: Zu spezifischem Commit wechseln
        specific_commit_action = update_menu.addAction(
            "🎯 Zu &spezifischem Commit wechseln"
        )
        specific_commit_action.triggered.connect(self.checkout_specific_commit)

        # ========== Hilfe-Menü ==========
        help_menu = menubar.addMenu("&Hilfe")

        # Über-Dialog
        about_action = help_menu.addAction("ℹ️ Ü&ber")
        about_action.triggered.connect(self.show_about)

        # Hilfe-Dialog
        help_action = help_menu.addAction("❓ &Hilfe")
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
