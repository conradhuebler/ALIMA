"""
Pipeline Tab - Vertical pipeline UI for ALIMA workflow
Claude Generated - Orchestrates the complete analysis pipeline in a chat-like interface
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QSplitter,
    QFrame,
    QComboBox,
    QSizePolicy,
    QSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from typing import Optional, Dict, List
import logging
from datetime import datetime

from ..core.pipeline_manager import PipelineManager, PipelineStep
from ..core.alima_manager import AlimaManager
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..llm.llm_service import LlmService
from .unified_input_widget import UnifiedInputWidget
from .pipeline_chat_panel import PipelineChatPanel
from .pipeline_step_widget import PipelineStepWidget
from ._pipeline_tab_control import PipelineTabControlMixin
from ._pipeline_tab_events import PipelineTabEventsMixin
from ._pipeline_tab_ui import PipelineTabUiMixin
from .workers import PipelineWorker


class PipelineTab(
    PipelineTabUiMixin,
    PipelineTabEventsMixin,
    PipelineTabControlMixin,
    QWidget,
):
    """Main pipeline tab with vertical workflow - Claude Generated"""

    # Signals
    pipeline_started = pyqtSignal(str)  # pipeline_id
    pipeline_completed = pyqtSignal()
    step_selected = pyqtSignal(str)  # step_id

    # Signals for pipeline result emission to other tabs - Claude Generated
    search_results_ready = pyqtSignal(dict)  # For SearchTab.display_search_results()
    analysis_results_ready = pyqtSignal(object)  # For AbstractTab analysis results
    pipeline_results_ready = pyqtSignal(object)  # Complete analysis_state for distribution - Claude Generated

    # Agentic dock signals – routed through MainWindow to QDockWidget - Claude Generated
    agentic_context_updated = pyqtSignal(str, dict)   # step_name, snapshot
    agentic_mode_changed = pyqtSignal(bool)            # enabled
    agentic_workflow_built = pyqtSignal(object)        # WorkflowDef

    # Agentic step ID → classical step_widget key + tab index - Claude Generated
    _AGENTIC_STEP_MAP = {
        "extraction": ("initialisation", 1),
        "search": ("search", 2),
        "selection_chunks": ("search", 2),
        "selection": ("keywords", 3),
        "verify_keywords": ("keywords", 3),
        "classification": ("dk_classification", 5),
        "dk_collect": ("dk_search", 4),
        "dk_postprocess": ("dk_classification", 5),
    }

    @staticmethod
    def _dk_class_codes(dk_classifications) -> list:
        """Convert dk_classifications (List[Dict] or List[str]) to flat code list - Claude Generated"""
        codes = []
        for cls in dk_classifications:
            if isinstance(cls, dict):
                codes.append(cls.get("code", str(cls)))
            else:
                codes.append(str(cls))
        return codes

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService,
        cache_manager: UnifiedKnowledgeManager,
        pipeline_manager: PipelineManager,
        main_window=None,
        parent=None,
    ):
        super().__init__(parent)
        self.alima_manager = alima_manager
        self.llm_service = llm_service
        self.cache_manager = cache_manager
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)

        # Use injected central PipelineManager instead of creating redundant instance - Claude Generated
        self.pipeline_manager = pipeline_manager

        # Pipeline worker for background execution
        self.pipeline_worker: Optional[PipelineWorker] = None

        # Pipeline timing tracking
        self.step_start_times: Dict[str, datetime] = {}
        self.pipeline_start_time: Optional[datetime] = None
        self.current_running_step: Optional[str] = None

        # Live timer for duration updates
        self.duration_update_timer = QTimer()
        self.duration_update_timer.timeout.connect(self.update_current_step_duration)
        self.duration_update_timer.setInterval(
            500
        )  # Update every 500ms (elapsed time doesn't need sub-second precision) - Claude Generated

        # UI components
        self.step_widgets: Dict[str, PipelineStepWidget] = {}
        self.unified_input: Optional[UnifiedInputWidget] = None

        # Input state
        self.current_input_text: str = ""
        self.current_source_info: str = ""

        self.setup_ui()


    def update_current_step_duration(self):
        """Update the duration of the currently running step in the status label - Claude Generated"""
        if (
            self.current_running_step
            and self.current_running_step in self.step_start_times
        ):
            duration_seconds = (
                datetime.now() - self.step_start_times[self.current_running_step]
            ).total_seconds()

            step_name = {
                "input": "Input",
                "initialisation": "Initialisierung",
                "search": "Suche",
                "keywords": "Schlagworte",
                "dk_search": "DK-Katalog-Suche",
                "dk_classification": "DK-Klassifikation",
            }.get(self.current_running_step, self.current_running_step.title())

            if hasattr(self, "pipeline_status_label"):
                self.pipeline_status_label.setText(f"▶ {step_name} ({duration_seconds:.1f}s)")

    def setup_ui(self):
        """Setup the pipeline UI - Claude Generated"""
        # Prevent sizeHint propagation to MainWindow to avoid automatic window resizing - Claude Generated
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Compact toolbar with primary actions - Claude Generated
        self.create_toolbar(main_layout)

        # Main pipeline area (control header moved to compact widget)
        self.setup_pipeline_area(main_layout)

    def setup_pipeline_area(self, main_layout):
        """Setup main pipeline area with streaming feedback - Claude Generated"""
        # Create a main splitter for pipeline steps and streaming - Claude Generated
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(True)

        # Left side: Pipeline steps as vertical tabs (directly in main_splitter, no steps_splitter) - Claude Generated
        self.pipeline_tabs = QTabWidget()
        self.pipeline_tabs.setTabPosition(QTabWidget.TabPosition.West)
        self.pipeline_tabs.setTabShape(QTabWidget.TabShape.Rounded)
        self.pipeline_tabs.setMinimumWidth(300)  # Reduced for small screens - Claude Generated

        # Set tab width to be smaller
        self.pipeline_tabs.setStyleSheet(
            self.pipeline_tabs.styleSheet()
            + """
            QTabBar::tab {
                min-width: 80px;  /* Reduced tab width */
                max-width: 120px;
            }
        """
        )

        # Enhanced tab styling
        self.pipeline_tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: 1px solid #ddd;
                background: white;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
            QTabWidget::tab-bar {{
                alignment: left;
            }}
            QTabBar::tab {{
                background: #f5f5f5;
                border: 1px solid #ddd;
                #padding: 12px 8px; # do not set padding to avoid increasing tab height/width
                margin-bottom: 2px;
                border-top-left-radius: 4px;
                border-bottom-left-radius: 4px;
                min-width: 100px;
            }}
            QTabBar::tab:selected {{
                background: #2196f3;
                color: white;
                border-right: none;
            }}
            QTabBar::tab:hover:!selected {{
                background: #e3f2fd;
            }}
        """
        )

        # Create pipeline step tabs - direkt in main_splitter - Claude Generated
        self.create_pipeline_step_tabs()
        self.main_splitter.addWidget(self.pipeline_tabs)

        # Right side: Unified PipelineChatPanel (P-δ.5a) — pipeline log on
        # top, chat input on the bottom, both rendered into the same area.
        self.stream_widget = PipelineChatPanel(
            llm_service=self.llm_service,
            prompt_service=getattr(self.alima_manager, "prompt_service", None),
            pipeline_manager=self.pipeline_manager,
            mcp_registry=getattr(self, "mcp_registry", None),
            parent=self,
        )

        # Connect streaming widget signals
        self.stream_widget.cancel_pipeline.connect(self.reset_pipeline)
        self.stream_widget.abort_generation_requested.connect(self.on_abort_current_step_requested)  # Claude Generated

        self.main_splitter.addWidget(self.stream_widget)

        # Initial split: 65% tabs (dominant when idle), 35% stream - Claude Generated
        self.main_splitter.setStretchFactor(0, 65)
        self.main_splitter.setStretchFactor(1, 35)
        self.main_splitter.setSizes([650, 350])

        main_layout.addWidget(self.main_splitter)

    def _get_task_provider_model(self, task_name: str) -> tuple[str, str]:
        """
        Get provider and model from task preferences configuration.
        Falls back to sensible defaults if not configured.
        Claude Generated Fix for persistent settings
        """
        try:
            from ..utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            if config and hasattr(config, 'unified_config') and config.unified_config:
                task_prefs = getattr(config.unified_config, 'task_preferences', {})
                if task_name in task_prefs:
                    task_data = task_prefs[task_name]
                    if hasattr(task_data, 'model_priority') and task_data.model_priority:
                        first_pref = task_data.model_priority[0]
                        if isinstance(first_pref, dict):
                            provider = first_pref.get('provider_name', '')
                            model = first_pref.get('model_name', '')
                        else:
                            provider = getattr(first_pref, 'provider_name', '')
                            model = getattr(first_pref, 'model_name', '')
                        self.logger.debug(f"Loaded task preference for {task_name}: {provider}/{model}")
                        return provider, model
        except Exception as e:
            self.logger.warning(f"Could not load task preference for {task_name}: {e}")

        # No task preference found — return empty strings so the pipeline uses
        # global_provider_override or the first available provider from step_configs.
        return "", ""

    def create_pipeline_step_tabs(self):
        """Create pipeline step tabs - Claude Generated"""
        # Step 1: Input
        input_step = PipelineStep(
            step_id="input", name="📥 SCHRITT 1: INPUT", status="pending"
        )
        input_widget = self.create_input_step_widget()
        input_step_widget = PipelineStepWidget(input_step)
        input_step_widget.set_content(input_widget)
        self.step_widgets["input"] = input_step_widget
        self.pipeline_tabs.addTab(input_step_widget, "📥 Input & Datenquellen")

        # Step 2: Initialisation
        # Get provider/model from task preferences - Claude Generated Fix
        init_provider, init_model = self._get_task_provider_model("initialisation")
        initialisation_step = PipelineStep(
            step_id="initialisation",
            name="🔤 SCHRITT 2: INITIALISIERUNG",
            status="pending",
            provider=init_provider,
            model=init_model,
        )
        initialisation_widget = self.create_initialisation_step_widget()
        initialisation_step_widget = PipelineStepWidget(initialisation_step)
        initialisation_step_widget.set_content(initialisation_widget)
        self.step_widgets["initialisation"] = initialisation_step_widget
        self.pipeline_tabs.addTab(initialisation_step_widget, "🔤 Schlagwort-Extraktion")

        # Step 3: Search
        search_step = PipelineStep(
            step_id="search", name="🔍 SCHRITT 3: GND-SUCHE", status="pending"
        )
        search_widget = self.create_search_step_widget()
        search_step_widget = PipelineStepWidget(search_step)
        search_step_widget.set_content(search_widget)
        self.step_widgets["search"] = search_step_widget
        self.pipeline_tabs.addTab(search_step_widget, "🔍 GND-Recherche")

        # Step 4: Keywords (Verbale Erschließung)
        # Get provider/model from task preferences - Claude Generated Fix
        kw_provider, kw_model = self._get_task_provider_model("keywords")
        keywords_step = PipelineStep(
            step_id="keywords",
            name="✅ SCHRITT 4: SCHLAGWORTE",
            status="pending",
            provider=kw_provider,
            model=kw_model,
        )
        keywords_widget = self.create_keywords_step_widget()
        keywords_step_widget = PipelineStepWidget(keywords_step)
        keywords_step_widget.set_content(keywords_widget)
        self.step_widgets["keywords"] = keywords_step_widget
        self.pipeline_tabs.addTab(keywords_step_widget, "✅ Schlagwort-Verifikation")

        # Step 5: DK Search (catalog search)
        dk_search_step = PipelineStep(
            step_id="dk_search",
            name="📊 SCHRITT 5: DK-KATALOG-SUCHE",
            status="pending",
        )
        dk_search_widget = self.create_dk_search_step_widget()
        dk_search_step_widget = PipelineStepWidget(dk_search_step)
        dk_search_step_widget.set_content(dk_search_widget)
        self.step_widgets["dk_search"] = dk_search_step_widget
        self.pipeline_tabs.addTab(dk_search_step_widget, "📊 Katalog-Recherche")

        # Step 6: DK Classification (LLM analysis)
        dk_provider, dk_model = self._get_task_provider_model("dk_classification")
        dk_classification_step = PipelineStep(
            step_id="dk_classification",
            name="📚 SCHRITT 6: DK-KLASSIFIKATION",
            status="pending",
            provider=dk_provider,
            model=dk_model,
        )
        dk_classification_widget = self.create_dk_classification_step_widget()
        dk_classification_step_widget = PipelineStepWidget(dk_classification_step)
        dk_classification_step_widget.set_content(dk_classification_widget)
        self.step_widgets["dk_classification"] = dk_classification_step_widget
        self.pipeline_tabs.addTab(dk_classification_step_widget, "📚 DK/RVK-Klassifikation")

    def create_toolbar(self, main_layout):
        """Create compact toolbar with primary pipeline actions - Claude Generated"""
        self.toolbar_frame = QFrame()
        self.toolbar_frame.setFixedHeight(44)
        self.toolbar_frame.setStyleSheet(
            "QFrame { background: #f8f9fa; border-bottom: 1px solid #dee2e6; }"
        )

        tb_layout = QHBoxLayout(self.toolbar_frame)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(6)

        # Auto-pipeline button
        self.auto_pipeline_button = QPushButton("🚀 Auto-Pipeline")
        self.auto_pipeline_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 5px 14px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #ccc; }
            """
        )
        self.auto_pipeline_button.clicked.connect(self.start_auto_pipeline)
        tb_layout.addWidget(self.auto_pipeline_button)

        # Stop button (initially hidden, shown only when pipeline is running)
        self.stop_pipeline_button = QPushButton("⏹️ Stop")
        self.stop_pipeline_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #da190b; }
            QPushButton:disabled { background-color: #e57373; }
            """
        )
        self.stop_pipeline_button.setVisible(False)
        self.stop_pipeline_button.clicked.connect(self.on_stop_pipeline_requested)
        tb_layout.addWidget(self.stop_pipeline_button)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedHeight(24)
        sep.setStyleSheet("color: #ccc;")
        tb_layout.addWidget(sep)

        # Secondary actions
        self.load_json_button = QPushButton("📁 JSON laden")
        self.load_json_button.setToolTip("Pipeline-State aus JSON-Datei laden")
        self.load_json_button.clicked.connect(self.load_json_state)
        tb_layout.addWidget(self.load_json_button)

        config_btn = QPushButton("⚙️ Config")
        config_btn.setToolTip("Pipeline-Konfiguration öffnen (per-Step Modell/Prompt/Parameter)")
        config_btn.clicked.connect(self.show_pipeline_config)
        tb_layout.addWidget(config_btn)

        reset_btn = QPushButton("🔄 Reset")
        reset_btn.setToolTip("Pipeline zurücksetzen")
        reset_btn.clicked.connect(self.reset_pipeline)
        tb_layout.addWidget(reset_btn)

        # Shared combobox stylesheet — light background, blue selection
        _combo_css = (
            "QComboBox { padding: 3px 6px; border: 1px solid #ccc; "
            "border-radius: 3px; font-size: 11px; }"
            "QComboBox QAbstractItemView { background-color: #ffffff; color: #333333; "
            "selection-background-color: #1976d2; selection-color: white; border: 1px solid #ccc; }"
        )

        # Workflow picker
        workflow_sep = QFrame()
        workflow_sep.setFrameShape(QFrame.Shape.VLine)
        workflow_sep.setFixedHeight(24)
        workflow_sep.setStyleSheet("color: #ccc;")
        tb_layout.addWidget(workflow_sep)

        workflow_label = QLabel("🧬 Workflow:")
        workflow_label.setStyleSheet("color: #555; font-weight: bold;")
        tb_layout.addWidget(workflow_label)
        self.agentic_workflow_label = workflow_label  # kept for compat

        self.workflow_combo = QComboBox()
        self.workflow_combo.setMinimumWidth(180)
        self.workflow_combo.setMaximumWidth(260)
        self.workflow_combo.setToolTip(
            "Pipeline-Auswahl. Bestimmt zugleich die Ausführungsart:\n"
            "• „Klassische Pipeline (nicht agentisch)“ → sequenzielle Pipeline.\n"
            "• alle übrigen Einträge → agentischer Workflow (YAML).\n"
            "Voreinstellung: ALIMA v5.1. Legacy-Workflows nach dem Trenner."
        )
        self.workflow_combo.setStyleSheet(_combo_css)
        self._populate_workflow_combo()
        tb_layout.addWidget(self.workflow_combo)

        # LLM override — always visible in toolbar
        llm_sep = QFrame()
        llm_sep.setFrameShape(QFrame.Shape.VLine)
        llm_sep.setFixedHeight(24)
        llm_sep.setStyleSheet("color: #ccc;")
        tb_layout.addWidget(llm_sep)

        llm_label = QLabel("🤖 LLM:")
        llm_label.setStyleSheet("color: #555;")
        tb_layout.addWidget(llm_label)

        # Shared provider+model picker (Phase 5): replaces the combined
        # "provider | model" combo. The empty placeholder means "-- Standard --"
        # (use the configured per-step defaults / task preferences). Non-editable:
        # native text rendering, and the clean model name lives in UserRole.
        from .provider_model_selector import ProviderModelSelector
        self.global_override_selector = ProviderModelSelector(
            allow_empty=True,
            editable_model=False,
            empty_provider_label="-- Standard --",
            empty_model_label="(Auto)",
        )
        self.global_override_selector.setToolTip(
            "Provider/Modell für alle LLM-Schritte.\n"
            "\"-- Standard --\" = Aus Konfiguration/Task-Präferenzen"
        )
        self.global_override_selector.set_combo_style(_combo_css)
        self.global_override_selector.provider_combo.setFixedWidth(150)
        self.global_override_selector.model_combo.setFixedWidth(190)
        self._populate_global_override_combo()
        tb_layout.addWidget(self.global_override_selector)

        # Global thinking override for all LLM steps - Claude Generated
        think_label = QLabel("Thinking:")
        think_label.setStyleSheet("padding-left: 8px;")
        tb_layout.addWidget(think_label)
        self.global_think_combo = QComboBox()
        self.global_think_combo.addItems(["Standard", "An", "Aus"])
        self.global_think_combo.setToolTip(
            "Thinking/Reasoning für alle LLM-Schritte überschreiben.\n"
            "Standard = pro Modell/Task konfigurierter Wert\n"
            "An = think=true · Aus = think=false"
        )
        self.global_think_combo.setStyleSheet(_combo_css)
        self.global_think_combo.setFixedWidth(110)
        tb_layout.addWidget(self.global_think_combo)

        # Token budget for the agentic steps - Claude Generated
        budget_label = QLabel("Budget:")
        budget_label.setStyleSheet("padding-left: 8px;")
        tb_layout.addWidget(budget_label)
        self.global_max_tokens_spin = QSpinBox()
        self.global_max_tokens_spin.setRange(0, 131072)
        self.global_max_tokens_spin.setSingleStep(2048)
        self.global_max_tokens_spin.setValue(0)
        self.global_max_tokens_spin.setSpecialValueText("Standard")
        self.global_max_tokens_spin.setToolTip(
            "max_tokens für alle agentischen LLM-Schritte.\n"
            "Standard = Wert aus dem Workflow-YAML (meist 4096)\n"
            "Ein Reasoning-Modell verbraucht dieses Budget im Denkkanal, "
            "bevor die Antwort beginnt; der Kanal wächst mit dem Budget mit.\n"
            "Wirkt nur agentisch — die klassische Pipeline kennt kein Budget."
        )
        self.global_max_tokens_spin.setFixedWidth(110)
        tb_layout.addWidget(self.global_max_tokens_spin)

        # Agentic vs. classic is driven by the workflow_combo selection
        # (see _on_workflow_changed). The agentic context dock is shown only
        # on request via View ▸ 🤖 Agentic Kontext — Claude Generated.

        # Fixed gap right after the override selector, then a stretch so the
        # status/timer label sits at the far right.
        tb_layout.addSpacing(16)
        tb_layout.addStretch()

        # Pipeline status label (shows the live step timer during a run)
        self.pipeline_status_label = QLabel("Bereit")
        self.pipeline_status_label.setStyleSheet("color: #666; padding-left: 8px;")
        tb_layout.addWidget(self.pipeline_status_label)

        main_layout.addWidget(self.toolbar_frame)

    def _populate_global_override_combo(self):
        """Populate the global LLM override picker from the enabled providers.

        Safe to call again on config changes; the per-provider model list is
        loaded lazily from the shared cache by the selector. - Claude Generated
        """
        try:
            from ..utils.config_manager import ConfigManager
            unified_config = ConfigManager().get_unified_config()
            names = [p.name for p in unified_config.get_enabled_providers()]
            # refresh=False: the "-- Standard --" placeholder is the default pick.
            self.global_override_selector.set_providers(names, refresh=False)
        except Exception as e:
            self.logger.error(f"Error populating LLM override selector: {e}")

    def jump_to_step(self, step_id: str):
        """Jump to specific pipeline step - Claude Generated"""
        for i in range(self.pipeline_tabs.count()):
            widget = self.pipeline_tabs.widget(i)
            if (
                isinstance(widget, PipelineStepWidget)
                and widget.step.step_id == step_id
            ):
                self.pipeline_tabs.setCurrentIndex(i)
                break

