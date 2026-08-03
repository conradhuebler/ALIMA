from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QMessageBox,
    QSizePolicy,
    QDockWidget,
)
from PyQt6.QtCore import Qt, QSettings, QTimer
import logging

from .find_keywords import SearchTab
from .abstract_tab import AbstractTab
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from ..core.alima_manager import AlimaManager
from ..core.pipeline_manager import PipelineManager
from ..core.state_bus import AlimaStateBus
from ..utils.config_manager import ConfigManager

# Legacy config import removed - using unified config system now
from .analysis_review_tab import AnalysisReviewTab
from .dk_analysis_unified_tab import DkAnalysisUnifiedTab
from .ub_catalog_tab import UBCatalogTab
from .image_analysis_tab import ImageAnalysisTab
from .styles import get_main_stylesheet
from .global_status_bar import GlobalStatusBar
from .pipeline_tab import PipelineTab
from .comparison_tab import ComparisonTab
from .agentic_context_widget import AgenticContextWidget
from ._main_window_results import MainWindowResultsMixin
from ._main_window_settings import MainWindowSettingsMixin
from ._main_window_data import MainWindowDataMixin
from ._main_window_menu import MainWindowMenuMixin

# P-δ.5a: ChatWidget retired — chat lives inside PipelineChatPanel (embedded in
# the PipelineTab right-side panel). dk_classification_tab / dk_analysis_tab
# replaced by dk_analysis_unified_tab. - Claude Generated


class MainWindow(
    MainWindowResultsMixin,
    MainWindowSettingsMixin,
    MainWindowDataMixin,
    MainWindowMenuMixin,
    QMainWindow,
):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("TUBAF", "Alima")
        # self.config = UnifiedProviderConfig()  # Legacy config reference removed
        # Initialisiere Core-Komponenten
        self.cache_manager = UnifiedKnowledgeManager()
        self.logger = logging.getLogger(__name__)

        # Bus-driven result-tab refresh (Konvergenz Pipeline/Agent) - Claude Generated
        # True while a classical GUI pipeline run owns tab distribution via the
        # pipeline_results_ready signal; the AlimaStateBus completion handler
        # then skips to avoid double-distribution. Agent-driven runs leave this
        # False, so the bus handler distributes for them.
        self._classical_active = False
        # op-tags collected from coalesced state.changed events before a refresh.
        self._pending_ops: set = set()

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

        # Agentic context dock – floatable/dockable, hidden until agentic mode is active - Claude Generated
        self.agentic_context_widget = AgenticContextWidget()
        self.agentic_dock = QDockWidget("🤖 Agentic Kontext", self)
        self.agentic_dock.setWidget(self.agentic_context_widget)
        self.agentic_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.agentic_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.agentic_dock.hide()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.agentic_dock)

        # P-δ.5a: chat_dock retired — chat input + history live inside
        # PipelineChatPanel which is embedded in PipelineTab's right
        # splitter. Auto-load of analysis_state happens via
        # pipeline_tab.stream_widget.load_context (PipelineChatPanel API).

        # Menüleiste
        self.create_menu_bar()

        # Tab-Widget
        self.tabs = QTabWidget()
        # Prevent tab sizeHint changes from resizing the main window - Claude Generated
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        main_layout.addWidget(self.tabs)

        # Tabs erstellen — UB-Katalog zuerst, er wird als Ergebnis-Reiter in
        # den vereinheitlichten Such-Tab eingebettet (Aug 4; der frühere
        # SearchTabUnified-Combo-Umschalter ist entfernt) - Claude Generated
        self.ub_catalog_tab = UBCatalogTab(pipeline_manager=self.pipeline_manager, parent=self)
        self.search_tab = SearchTab(
            cache_manager=self.cache_manager,
            alima_manager=self.alima_manager,
            pipeline_manager=self.pipeline_manager,
            ub_catalog_tab=self.ub_catalog_tab,
        )

        # Pass alima_manager and llm_service to AbstractTab
        self.abstract_tab = AbstractTab(
            alima_manager=self.alima_manager,
            llm_service=self.llm_service,
            cache_manager=self.cache_manager,
            pipeline_manager=self.pipeline_manager,
            main_window=self,
        )
        self.abstract_tab.template_name = "abstract_analysis"  # This might be removed later if task selection is fully dynamic
        self.abstract_tab.set_task("initialisation")  # Set initial task (pipeline step name) - Claude Generated

        # P-θ.2: Verifikations-Tab gemergt in abstract_tab via Task-Switcher.
        # `analyse_keywords` ist Alias für abstract_tab — bestehender
        # Routing-Code referenziert ihn weiterhin.
        self.analyse_keywords = self.abstract_tab

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
        # Mark classical-pipeline distribution active so the bus completion
        # handler defers to the pipeline_results_ready signal path - Claude Generated
        self.pipeline_tab.pipeline_started.connect(self._on_classical_pipeline_started)
        self.pipeline_tab.pipeline_completed.connect(
            lambda: self.global_status_bar.update_pipeline_status(
                "Pipeline", "completed"
            )
        )

        # Connect pipeline results to specialized tabs - Claude Generated
        # Central distribution via on_pipeline_results_ready slot
        self.pipeline_tab.pipeline_results_ready.connect(self.on_pipeline_results_ready)

        # P-δ.5a: PipelineChatPanel (inside PipelineTab) receives the
        # pipeline analysis state for chat-context loading. No floating
        # chat_dock anymore — load_context lives on the panel itself.
        self.pipeline_tab.pipeline_results_ready.connect(
            self.pipeline_tab.stream_widget.load_context
        )

        # Intermediate step distribution for better live feedback - Claude Generated
        self.pipeline_tab.analysis_results_ready.connect(self.on_intermediate_analysis_ready)

        # Connect to dedicated DK classification tab
        self.pipeline_tab.pipeline_results_ready.connect(self.dk_classification_tab.update_data)

        # (Auto-fill der UB-Keywords läuft jetzt über die GETEILTE Sucheingabe:
        # SearchTab.update_data füllt sie aus den finalen Pipeline-Keywords.)

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
        self.tabs.addTab(self.image_analysis_tab, "📷 Bild")
        self.tabs.addTab(self.abstract_tab, "📝 Manuelle Analyse")

        # Vereinheitlichter Such-Tab (Aug 4): GND + UB-Katalog teilen sich EIN
        # Suchfeld; der UB-Katalog ist ein Ergebnis-Reiter im SearchTab.
        # self.ub_catalog_tab bleibt als Referenz gültig (eingebettet).
        self.tabs.addTab(self.search_tab, "🔍 Suche")
        self.tabs.addTab(self.dk_analysis_unified_tab, "📊 Klassifikationen")
        self.tabs.addTab(self.analysis_review_tab, "📊 Review")

        # Comparison tab - initially hidden until comparison is loaded - Claude Generated
        self._comparison_tab_idx = self.tabs.addTab(self.comparison_tab, "🔍 Vergleich")
        self.tabs.setTabVisible(self._comparison_tab_idx, False)
        self.comparison_tab.comparison_loaded.connect(self._show_comparison_tab)

        # Konvergenz Pipeline/Agent: result tabs become bus-driven so they also
        # refresh when the chat agent (not just the classical pipeline) produces
        # results. Coalesce bursts of state.changed into one re-render. - Claude Generated
        self._rerender_timer = QTimer(self)
        self._rerender_timer.setSingleShot(True)
        self._rerender_timer.setInterval(150)
        self._rerender_timer.timeout.connect(self._flush_rerender)
        try:
            bus = AlimaStateBus()
            bus.subscribe("state.pipeline_completed", self._on_bus_pipeline_completed)
            bus.subscribe("state.changed", self._on_bus_state_changed)
        except Exception:
            self.logger.exception("MainWindow: AlimaStateBus subscribe failed")

        # Globale Statusleiste
        self.global_status_bar = GlobalStatusBar()
        self.setStatusBar(self.global_status_bar)

        # Initialize status bar with services
        self.global_status_bar.set_services(self.llm_service, self.cache_manager)

        # Show DB fallback notice if SQLite was used because MySQL/MariaDB driver missing
        if getattr(self.cache_manager, 'db_fallback_notice', None):
            self.global_status_bar.show_temporary_message(
                self.cache_manager.db_fallback_notice, 8000
            )

        # Connect pipeline_tab agentic signals to dock - Claude Generated
        self.pipeline_tab.agentic_context_updated.connect(
            self.agentic_context_widget.on_context_updated
        )
        self.pipeline_tab.agentic_mode_changed.connect(self._on_agentic_mode_changed)
        self.pipeline_tab.agentic_workflow_built.connect(
            self.agentic_context_widget.build_panels
        )

        # P-θ.4: one-time tab-consolidation banner.
        QTimer.singleShot(0, self._maybe_show_ptheta_banner)

    def _maybe_show_ptheta_banner(self) -> None:
        """Show the WP10 P-θ tab-consolidation banner once per install. Claude Generated."""
        if not self.config_manager:
            return
        try:
            config = self.config_manager.load_config()
            if getattr(config.ui_config, "ptheta_banner_seen", False):
                return
            QMessageBox.information(
                self,
                "GUI konsolidiert (WP10 P-θ)",
                "Die ALIMA-Tabs wurden auf 7 reduziert:\n\n"
                "• 🌐 Crossref entfernt — Pipeline-DOI-Input deckt den Bedarf.\n"
                "• 📝 Abstract + ✅ Verifikation → „Manuelle Analyse"
                " mit Task-Switcher in der Top-Leiste.\n"
                "• 🔍 GND-Suche + 📚 UB-Katalog → „Suche"
                " mit Quellen-Picker (GND/SWB/Lobid oder UB-Katalog/DK).\n"
                "• 🧬 Workflow-Picker jetzt prominent im Pipeline-Header.\n\n"
                "Details: docs/legacy/migration_roadmap.md — §P-θ.",
            )
            config.ui_config.ptheta_banner_seen = True
            self.config_manager.save_config(config)
        except Exception as exc:
            self.logger.debug(f"P-θ banner skipped: {exc}")

    def _on_agentic_mode_changed(self, enabled: bool) -> None:
        """Refresh agentic context panels on mode change - Claude Generated.

        Dock visibility is intentionally NOT toggled here: the agentic context
        dock is shown only on request via View ▸ 🤖 Agentic Kontext. This only
        keeps the panel data consistent with the selected pipeline mode.
        """
        if enabled:
            self.agentic_context_widget.reset()
        else:
            self.agentic_context_widget.clear_panels()

    def _show_agentic_dock(self) -> None:
        """Show the agentic context dock on demand and (re)build its panels
        from the currently selected workflow - Claude Generated."""
        try:
            if hasattr(self, "pipeline_tab"):
                self.pipeline_tab._rebuild_agentic_panels()
        except Exception as exc:  # noqa: BLE001
            self.logger.debug(f"Agentic dock panel rebuild skipped: {exc}")
        self.agentic_dock.show()
        self.agentic_dock.raise_()

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

    def closeEvent(self, event):
        """Wird beim Schließen des Fensters aufgerufen"""
        self.save_settings()
        event.accept()

    def update_status(self, message: str):
        """Aktualisiert die Statusleiste - Claude Generated (Fixed AttributeError)"""
        self.global_status_bar.show_temporary_message(message, 3000)

    def show_error(self, message: str):
        """Zeigt eine Fehlermeldung"""

