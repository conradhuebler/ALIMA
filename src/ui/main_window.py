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
    QSizePolicy,
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
from ..utils.pipeline_utils import PipelineResultFormatter
from ..utils.pipeline_defaults import get_autosave_dir


# Legacy config import removed - using unified config system now
from .crossref_tab import CrossrefTab
from .analysis_review_tab import AnalysisReviewTab
from .dk_analysis_unified_tab import DkAnalysisUnifiedTab
from .ub_catalog_tab import UBCatalogTab
from .tablewidget import TableWidget, DatabaseViewerDialog
from .image_analysis_tab import ImageAnalysisTab
from .styles import get_main_stylesheet, set_dark_mode, set_font_size, get_font_size
from .global_status_bar import GlobalStatusBar
from .pipeline_tab import PipelineTab
from .comparison_tab import ComparisonTab
# dk_classification_tab and dk_analysis_tab replaced by dk_analysis_unified_tab
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
        # self.config = UnifiedProviderConfig()  # Legacy config reference removed
        # Initialisiere Core-Komponenten
        self.cache_manager = UnifiedKnowledgeManager()
        self.search_engine = SearchEngine(self.cache_manager)
        self.logger = logging.getLogger(__name__)

        # === DIAGNOSTIC: DB-Status nach Wizard-Start ===
        # Problem: Nach Wizard findet Schlagwortsuche keine Treffer
        try:
            db_info = self.cache_manager.db_manager.get_database_info()
            self.logger.info(f"✅ DB verbunden: {db_info['type']} - is_open={db_info.get('is_open', False)}")
            # Prüfe Tabellen-Inhalte
            try:
                gnd_count = self.cache_manager.db_manager.fetch_scalar("SELECT COUNT(*) FROM gnd_entries")
                self.logger.info(f"📊 gnd_entries: {gnd_count} Einträge")
                mappings_count = self.cache_manager.db_manager.fetch_scalar("SELECT COUNT(*) FROM search_mappings")
                self.logger.info(f"📊 search_mappings: {mappings_count} Einträge")
            except Exception as e:
                self.logger.warning(f"⚠️ Konnte Tabellen nicht prüfen: {e}")
        except Exception as e:
            self.logger.error(f"❌ DB-Initialisierung fehlgeschlagen: {e}")
        # === ENDE DIAGNOSTIC ===

        self.config_manager = ConfigManager(logger=self.logger)

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
        self._dark_mode = False  # Current theme state — Claude Generated
        self._batch_dialog = None  # Singleton batch processing dialog - Claude Generated

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
        # Responsive window size: 85% of screen, max 1400x900, centered - Claude Generated
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            w = min(int(available.width() * 0.85), 1400)
            h = min(int(available.height() * 0.85), 900)
            x = available.x() + (available.width() - w) // 2
            y = available.y() + (available.height() - h) // 2
            self.setGeometry(x, y, w, h)
            # Note: setMaximumSize removed - it breaks maximize button functionality
            # The size policy "Ignored" on streaming widgets prevents window expansion instead
        else:
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
        # Prevent tab sizeHint changes from resizing the main window - Claude Generated
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
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

        # Unified DK Analysis Tab - combines DK-Zuordnung, DK-Statistik, and UB-Suche - Claude Generated
        self.dk_analysis_unified_tab = DkAnalysisUnifiedTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            cache_manager=self.cache_manager,
            pipeline_manager=self.pipeline_manager,
            main_window=self,
        )

        # UB Catalog Tab - standalone dk_search step - Claude Generated
        self.ub_catalog_tab = UBCatalogTab(pipeline_manager=self.pipeline_manager, parent=self)

        # Backward compatibility aliases - Claude Generated
        self.dk_analysis_tab = self.dk_analysis_unified_tab
        self.dk_classification_tab = self.dk_analysis_unified_tab
        self.ub_search_tab = self.ub_catalog_tab

        # Pipeline Tab - Claude Generated
        self.pipeline_tab = PipelineTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            cache_manager=self.cache_manager,
            pipeline_manager=self.pipeline_manager,
            main_window=self,
        )

        # Comparison Tab - Claude Generated
        self.comparison_tab = ComparisonTab(main_window=self)

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

        # Intermediate step distribution for better live feedback - Claude Generated
        self.pipeline_tab.analysis_results_ready.connect(self.on_intermediate_analysis_ready)

        # Connect to dedicated DK classification tab
        self.pipeline_tab.pipeline_results_ready.connect(self.dk_classification_tab.update_data)

        # Auto-fill UB-Katalog with GND keywords from pipeline - Claude Generated
        self.pipeline_tab.pipeline_results_ready.connect(self.ub_catalog_tab.update_from_pipeline)

        # Forward UB catalog search results to DK-Analyse for LLM input - Claude Generated
        self.ub_catalog_tab.search_completed.connect(self.dk_analysis_unified_tab.receive_catalog_results)

        # Connect to SearchTab for GND post-processing transparency - Claude Generated
        self.pipeline_tab.pipeline_results_ready.connect(self.search_tab.update_data)
        self.search_tab.selection_changed.connect(self.on_search_selection_changed)

        # Update window title when pipeline completes - Claude Generated
        self.pipeline_tab.pipeline_results_ready.connect(self.on_pipeline_title_update)

        # Keep comparison tab current with latest pipeline result - Claude Generated
        self.pipeline_tab.pipeline_results_ready.connect(self.comparison_tab.load_from_current)

        # Add Pipeline tab first
        self.tabs.addTab(self.pipeline_tab, "🚀 Pipeline")

        # Individual tabs with icons and proper naming
        self.tabs.addTab(self.crossref_tab, "🌐 DOI")
        self.tabs.addTab(self.image_analysis_tab, "📷 Bild")
        self.tabs.addTab(self.abstract_tab, "📝 Abstract")
        self.tabs.addTab(self.search_tab, "🔍 GND-Suche")
        self.tabs.addTab(self.analyse_keywords, "✅ Verifikation")
        self.tabs.addTab(self.ub_catalog_tab, "📚 UB-Katalog")        # NEW - Claude Generated
        self.tabs.addTab(self.dk_analysis_unified_tab, "📊 Klassifikationen")
        self.tabs.addTab(self.analysis_review_tab, "📊 Review")

        # Comparison tab - initially hidden until comparison is loaded - Claude Generated
        self._comparison_tab_idx = self.tabs.addTab(self.comparison_tab, "🔍 Vergleich")
        self.tabs.setTabVisible(self._comparison_tab_idx, False)
        self.comparison_tab.comparison_loaded.connect(self._show_comparison_tab)

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
            if hasattr(self, 'dk_analysis_tab'):
                self.dk_analysis_tab.set_models_and_providers(available_models, available_providers)
            if hasattr(self, 'ub_search_tab') and hasattr(self.ub_search_tab, 'set_models_and_providers'):
                self.ub_search_tab.set_models_and_providers(available_models, available_providers)

            self.logger.debug(f"Updated tabs with {len(available_providers)} providers")

        except Exception as e:
            self.logger.error(f"Error updating tabs with provider info: {e}")

    def setup_provider_status_connections(self):
        """Connect to ProviderStatusService signals for reactive updates - Claude Generated"""
        if hasattr(self.alima_manager, 'provider_status_service') and self.alima_manager.provider_status_service:
            # Connect to status updates for automatic UI refresh
            self.alima_manager.provider_status_service.status_updated.connect(
                self.update_tabs_with_provider_info
            )
            self.logger.debug("Connected to ProviderStatusService signals")
        else:
            self.logger.warning("ProviderStatusService not available for signal connections")

        # REMOVED: Central Ollama signal management - hardcoded connections caused app hangs - Claude Generated
        # Signal connections removed to prevent deadlock/hang issues
        # Ollama provider changes now handled through unified provider system
        self.logger.debug("Ollama signal connections disabled to prevent app hangs")

    @pyqtSlot(object)
    def on_pipeline_results_ready(self, analysis_state):
        """Central slot for distributing pipeline results to specialized tabs - Claude Generated"""
        self.logger.debug("Distributing pipeline results to specialized tabs")

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

        # Ensure history is updated during live pipeline runs - Claude Generated
        # Pass the correct step result for each tab to avoid cross-tab contamination - Claude Generated
        if analysis_state:
            init_text = (
                analysis_state.initial_llm_call_details.response_full_text
                if hasattr(analysis_state, 'initial_llm_call_details') and analysis_state.initial_llm_call_details
                else None
            )
            self.abstract_tab.add_external_analysis_to_history(analysis_state, result_text=init_text)
            self.logger.debug("Updated AbstractTab history with initialisation result")

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
                keywords_text = analysis_state.final_llm_analysis.response_full_text
                self.analyse_keywords.display_llm_response(keywords_text)
                # Add to history with the correct (keywords step) result text - Claude Generated
                self.analyse_keywords.add_external_analysis_to_history(analysis_state, result_text=keywords_text)

        # 4. UB-Katalog Tab: GND keywords auto-filled via pipeline_results_ready signal
        # (wired to ub_catalog_tab.update_from_pipeline — Claude Generated)

        # 5. 📚 DK-Zuordnung & DK-Statistik Tab - Claude Generated
        if analysis_state.original_abstract:
            self.dk_analysis_tab.set_abstract(analysis_state.original_abstract)

        # Distribute flattened DK results to dk_analysis_tab for display
        if hasattr(analysis_state, 'dk_search_results_flattened') and analysis_state.dk_search_results_flattened:
            # Use flattened (DK-centric) results with titles for display
            self.dk_analysis_tab.set_keywords(analysis_state.dk_search_results_flattened)
        elif hasattr(analysis_state, 'dk_search_results') and analysis_state.dk_search_results:
            # Fallback to keyword-centric if flattened not available
            self.dk_analysis_tab.set_keywords(analysis_state.dk_search_results)

        if hasattr(analysis_state, 'dk_llm_analysis') and analysis_state.dk_llm_analysis:
            dk_text = analysis_state.dk_llm_analysis.response_full_text
            self.dk_analysis_tab.display_llm_response(dk_text)
            # Pass DK-specific result text so history shows classification, not keywords - Claude Generated
            self.dk_analysis_tab.add_external_analysis_to_history(analysis_state, result_text=dk_text)

        if hasattr(analysis_state, 'classifications') and analysis_state.classifications:
            self.dk_classification_tab.update_data(analysis_state)
            self.logger.info("✅ DK statistics and analysis tabs populated with pipeline results.")

        # 6. 📊 Analyse-Review Tab - Sende finale Ergebnisse (lossless) - Claude Generated
        if analysis_state.final_llm_analysis or analysis_state.classifications:
            self.analysis_review_tab.receive_full_state(analysis_state)
            self.logger.info("✅ Analysis review tab populated with full pipeline state (lossless).")

        # 6. 📚 DK-Klassifikation (Optional) - Claude Generated
        # Note: DK search results and classifications are handled by show_loaded_state_indicator()
        # when called from populate_all_tabs_from_state(). For live pipeline results, they are
        # displayed directly in pipeline_tab via on_step_completed() callbacks.
        if analysis_state.dk_search_results:
            self.logger.info(f"DK search results available: {len(analysis_state.dk_search_results)} entries")
        if analysis_state.classifications:
            self.logger.info(f"Classifications available: {len(analysis_state.classifications)} entries")

        self.logger.info("Pipeline results successfully distributed to all tabs")

        # Auto-navigate to DK Classification tab if results are available - Claude Generated
        if hasattr(analysis_state, 'classifications') and analysis_state.classifications:
            # Find index of DK classification tab
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "📊 Klassifikationen":
                    self.tabs.setCurrentIndex(i)
                    break

    @pyqtSlot(object)
    def on_intermediate_analysis_ready(self, analysis_result):
        """Update AbstractTab during live pipeline run - Claude Generated"""
        if not analysis_result:
            return

        self.logger.info("Updating AbstractTab with intermediate pipeline result")

        # 1. Route intermediate result to the correct tab based on task_name - Claude Generated
        task_name = getattr(analysis_result, 'task_name', None)
        if task_name in ["dk_class", "dk_classification"]:
            target_tab = self.dk_analysis_tab
        elif task_name in ["keywords", "rephrase", "keywords_chunked"]:
            target_tab = self.analyse_keywords
        else:
            target_tab = self.abstract_tab

        if hasattr(analysis_result, 'full_text'):
            target_tab.display_llm_response(analysis_result.full_text)
        elif hasattr(analysis_result, 'response_full_text'):
            target_tab.display_llm_response(analysis_result.response_full_text)

        # 2. Update keywords if available
        # Don't overwrite dk_analysis_tab keywords — its keywords_edit holds
        # DK search results as INPUT for analysis, not as output display - Claude Generated
        if target_tab is not self.dk_analysis_tab:
            if hasattr(analysis_result, 'matched_keywords') and analysis_result.matched_keywords:
                keywords_str = ", ".join(analysis_result.matched_keywords.keys())
                target_tab.set_keywords(keywords_str)
            elif hasattr(analysis_result, 'extracted_gnd_keywords') and analysis_result.extracted_gnd_keywords:
                keywords_str = ", ".join(analysis_result.extracted_gnd_keywords)
                target_tab.set_keywords(keywords_str)
            elif hasattr(analysis_result, 'extracted_gnd_classes') and analysis_result.extracted_gnd_classes:
                classes_str = ", ".join(analysis_result.extracted_gnd_classes)
                target_tab.set_keywords(classes_str)

    def update_window_title(self, arbeitstitel: str = None):
        """Update window title with optional work title - Claude Generated"""
        if arbeitstitel:
            self.setWindowTitle(f"ALIMA - {arbeitstitel}")
        else:
            self.setWindowTitle("ALIMA - Automatisierte Schlagwortgenerierung")

    @pyqtSlot(object)
    def on_pipeline_title_update(self, analysis_state):
        """Update window title with working title from analysis state - Claude Generated"""
        if analysis_state and hasattr(analysis_state, 'working_title') and analysis_state.working_title:
            self.update_window_title(analysis_state.working_title)
            self.logger.info(f"Window title updated: {analysis_state.working_title}")

    @pyqtSlot(dict)
    def on_search_selection_changed(self, changes):
        """Handle GND selection changes from SearchTab - Claude Generated

        Receives user modifications to GND keyword selection and propagates
        them to AnalysisReviewTab for final export integration.

        Args:
            changes: Dict with 'modified' (selection changes) and 'manual' (additions)
        """
        modified = changes.get('modified', {})
        manual = changes.get('manual', [])

        self.logger.info(
            f"GND selection changed: {len(modified)} modifications, {len(manual)} manual additions"
        )

        # TODO: Update analysis state with modifications
        # For now, just log the changes and show a notification
        if modified or manual:
            total_changes = len(modified) + len(manual)
            self.global_status_bar.show_notification(
                f"✅ GND-Auswahl aktualisiert: {total_changes} Änderungen",
                duration=3000
            )

        # Future enhancement: Propagate to AnalysisReviewTab
        # if hasattr(self, 'analysis_review_tab'):
        #     self.analysis_review_tab.update_gnd_selection(changes)

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
            self.ub_catalog_tab.update_keywords(keywords)
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

        # Restore pipeline splitter states - Claude Generated
        if hasattr(self, 'pipeline_tab'):
            self.pipeline_tab.restore_splitter_state(self.settings)

        # Load dark mode preference — Claude Generated
        from src.alima_gui import is_system_dark_mode
        app = QApplication.instance()
        saved = self.settings.value("dark_mode")
        if saved is not None:
            dark = saved == "true" or saved is True
        else:
            dark = is_system_dark_mode(app)
        self.apply_theme(dark)

        # Load font size preference — Claude Generated
        saved_fs = self.settings.value("font_size", None)
        if saved_fs is None and self.config_manager:
            try:
                saved_fs = self.config_manager.get_config().ui_config.font_size
            except Exception:
                saved_fs = 10
        if saved_fs is not None:
            self.apply_font_size(int(saved_fs))

    def save_settings(self):
        """Speichert die aktuellen Einstellungen"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("dark_mode", self._dark_mode)

        # Save pipeline splitter states - Claude Generated
        if hasattr(self, 'pipeline_tab'):
            self.pipeline_tab.save_splitter_state(self.settings)

    def apply_theme(self, dark: bool):
        """Switch between dark and light theme — Claude Generated"""
        from src.alima_gui import _apply_dark_app_palette, _apply_light_app_palette
        self._dark_mode = dark
        set_dark_mode(dark)
        app = QApplication.instance()
        if dark:
            _apply_dark_app_palette(app)
        else:
            _apply_light_app_palette(app)
        self.setStyleSheet(get_main_stylesheet())
        self._reapply_tab_stylesheets()
        label = "☀️ Helles Design" if dark else "🌙 Dunkles Design"
        if hasattr(self, '_theme_action'):
            self._theme_action.setText(label)
        self.settings.setValue("dark_mode", dark)

    def apply_font_size(self, pt: int) -> None:
        """Set global font size and refresh all stylesheets. — Claude Generated"""
        set_font_size(pt)
        self.setStyleSheet(get_main_stylesheet())
        self._reapply_tab_stylesheets()
        self.settings.setValue("font_size", pt)
        if self.config_manager:
            try:
                self.config_manager.get_config().ui_config.font_size = pt
                self.config_manager.save_config()
            except Exception as e:
                self.logger.debug(f"Could not persist font_size to config: {e}")

    def _reapply_tab_stylesheets(self):
        """Re-apply stylesheets to all tabs after theme change — Claude Generated"""
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if tab and hasattr(tab, 'refresh_styles'):
                try:
                    tab.refresh_styles()
                except Exception as e:
                    self.logger.debug(f"refresh_styles on tab {i} failed: {e}")

    def _toggle_theme(self):
        """Toggle between dark and light theme — Claude Generated"""
        self.apply_theme(not self._dark_mode)

    def _set_font_size_from_menu(self, pt: int) -> None:
        """Apply font size from menu and update checkmarks. — Claude Generated"""
        self.apply_font_size(pt)
        if hasattr(self, "_font_size_menu"):
            for action in self._font_size_menu.actions():
                try:
                    action.setChecked(action.text() == f"{pt} pt")
                except Exception:
                    pass

    def _on_config_changed(self):
        """Handle configuration changes from comprehensive settings dialog - Claude Generated"""
        self.logger.info("Configuration changed, refreshing components")
        # Apply font size if it changed in the settings dialog — Claude Generated
        try:
            new_fs = self.config_manager.get_config().ui_config.font_size
            if new_fs != get_font_size():
                self.apply_font_size(new_fs)
        except Exception:
            pass
        self._refresh_components()

    def _refresh_components(self):
        """Refresh all components with new configuration - Claude Generated"""
        try:
            # 1. Reload LLM service configuration and reinitialize providers
            if hasattr(self, 'llm_service'):
                self.llm_service.reload_providers()

            # 2. Refresh provider status to update reachability and available models
            if hasattr(self, 'llm_service'):
                self.llm_service.refresh_all_provider_status()

            # 3. Reload pipeline configuration to use updated provider preferences
            if hasattr(self, 'pipeline_manager'):
                self.pipeline_manager.reload_config()

            # 4. Update tabs with new provider information
            self.update_tabs_with_provider_info()

            # 5. Update global status bar with new provider and cache info
            if hasattr(self, 'global_status_bar'):
                self.global_status_bar.update_provider_info()
                self.global_status_bar.update_cache_status()

            # 6. Notify tabs about configuration changes (custom handlers)
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)
                if hasattr(tab, 'on_config_changed'):
                    tab.on_config_changed()

            self.logger.info("Configuration refreshed (providers, pipeline, tabs, status bar)")

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
        """Open batch processing dialog (non-modal singleton) - Claude Generated"""
        try:
            # Reuse existing dialog or create new one - Claude Generated
            if self._batch_dialog is None:
                from .batch_processing_dialog import BatchProcessingDialog
                self._batch_dialog = BatchProcessingDialog(
                    alima_manager=self.alima_manager,
                    cache_manager=self.cache_manager,
                    config_manager=self.config_manager,
                    logger=self.logger,
                    pipeline_tab=self.pipeline_tab,
                    parent=self
                )

            # Show non-modal (user can continue working while batch runs)
            self._batch_dialog.show()
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()

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
                str(get_autosave_dir(self.config_manager)),
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
                    str(Path.home() / "Downloads"),
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
            str(get_autosave_dir(self.config_manager)),
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

            # 5. 📊 Analyse-Review Tab - already populated by on_pipeline_results_ready() above

            # 6. 📚 UB-Katalog Tab - Keywords for library catalog search
            if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                # Type-safe join - Claude Generated (Fix for string parsing bug)
                final_kw = state.final_llm_analysis.extracted_gnd_keywords
                final_keywords = ", ".join(final_kw) if isinstance(final_kw, list) else str(final_kw)
                self.ub_catalog_tab.update_keywords(final_keywords)
                self.logger.info("✅ UB-Katalog tab populated with final keywords")

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

            # 3. Use working_title for filename if available - Claude Generated
            if hasattr(state_to_save, 'working_title') and state_to_save.working_title:
                default_filename = f"{state_to_save.working_title}.json"
                self.logger.info(f"Using working_title for export: {default_filename}")
            else:
                default_filename = f"analysis_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.logger.info(f"No working_title, using timestamp: {default_filename}")

            # 4. Den bewährten Speicher-Dialog mit dem korrekten Objekt aufrufen
            file_path = AnalysisPersistence.save_with_dialog(
                state=state_to_save,
                parent_widget=self,
                default_filename=default_filename
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

    def _open_comparison_tab(self):
        """Show and switch to the comparison tab - Claude Generated"""
        self.tabs.setTabVisible(self._comparison_tab_idx, True)
        self.tabs.setCurrentIndex(self._comparison_tab_idx)

    def _show_comparison_tab(self):
        """Signal handler: make comparison tab visible and switch to it - Claude Generated"""
        self.tabs.setTabVisible(self._comparison_tab_idx, True)
        self.tabs.setCurrentIndex(self._comparison_tab_idx)

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

        # Beenden-Aktion
        exit_action = file_menu.addAction("🚪 &Beenden")
        exit_action.triggered.connect(self.close)

        # ========== Extras/Tools-Menü (Datenbank und Debug) ==========
        tools_menu = menubar.addMenu("E&xtras")

        # Dark/light theme toggle — Claude Generated
        self._theme_action = tools_menu.addAction("🌙 Dunkles Design")
        self._theme_action.triggered.connect(self._toggle_theme)

        # Font size submenu — Claude Generated
        font_menu = tools_menu.addMenu("🔠 Schriftgröße")
        for pt in [8, 9, 10, 11, 12, 13, 14, 16]:
            action = font_menu.addAction(f"{pt} pt")
            action.setCheckable(True)
            action.setChecked(pt == get_font_size())
            action.triggered.connect(lambda checked, size=pt: self._set_font_size_from_menu(size))
        self._font_size_menu = font_menu

        tools_menu.addSeparator()

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

        # Erschließungsvergleich - Claude Generated
        compare_states_action = tools_menu.addAction("🔍 Erschließungs&vergleich...")
        compare_states_action.triggered.connect(self._open_comparison_tab)

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
