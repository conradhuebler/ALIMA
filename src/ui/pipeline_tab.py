"""
Pipeline Tab - Vertical pipeline UI for ALIMA workflow
Claude Generated - Orchestrates the complete analysis pipeline in a chat-like interface
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QScrollArea,
    QGroupBox,
    QLabel,
    QPushButton,
    QProgressBar,
    QTabWidget,
    QTextEdit,
    QSplitter,
    QFrame,
    QComboBox,
    QSpinBox,
    QSlider,
    QMessageBox,
    QLineEdit,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot, QThread
from PyQt6.QtGui import QFont, QPalette, QPixmap, QColor
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import json
from pathlib import Path

from ..core.pipeline_manager import PipelineManager, PipelineStep, PipelineConfig
from ..utils.pipeline_utils import PipelineResultFormatter
from .pipeline_config_dialog import PipelineConfigDialog
from ..core.alima_manager import AlimaManager
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..llm.llm_service import LlmService
from .image_analysis_tab import ImageAnalysisTab
from .unified_input_widget import UnifiedInputWidget
from .pipeline_chat_panel import PipelineChatPanel
from .workers import PipelineWorker


class PipelineStepWidget(QFrame):
    """Widget representing a single pipeline step - Claude Generated"""

    step_clicked = pyqtSignal(str)  # step_id

    def __init__(self, step: PipelineStep, parent=None):
        super().__init__(parent)
        self.step = step
        self.setup_ui()

    def setup_ui(self):
        """Setup the step widget UI - Claude Generated"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)

        # Prevent sizeHint propagation from child QTextEdits - Claude Generated
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)

        # Header with step name and status
        header_layout = QHBoxLayout()

        # Status icon
        from .styles import get_scaled_font
        self.status_label = QLabel()
        self.status_label.setFont(get_scaled_font(size_delta=+4, bold=True))
        header_layout.addWidget(self.status_label)

        # Step name
        name_label = QLabel(self.step.name)
        name_label.setFont(get_scaled_font(size_delta=+2, bold=True))
        self._name_label = name_label  # keep ref for refresh_styles
        header_layout.addWidget(name_label)

        header_layout.addStretch()

        # Enhanced Provider/Model info with task preference indicators - Claude Generated
        self.provider_model_label = QLabel()
        self.provider_model_label.setStyleSheet("color: #666;")
        self._update_provider_model_display()
        header_layout.addWidget(self.provider_model_label)

        layout.addLayout(header_layout)

        # Content area (initially empty)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 10, 0, 0)
        layout.addWidget(self.content_widget)

        # Now that all UI elements are created, update the display - Claude Generated
        self.update_status_display()

    def _update_provider_model_display(self):
        return
        
        """Update provider/model display with task preference indicators - Claude Generated"""
        # Safety check: ensure the label exists before updating - Claude Generated
        if not hasattr(self, 'provider_model_label') or not self.provider_model_label:
            return

        if not self.step.provider or not self.step.model:
            # Check if this is an LLM step that should have provider/model info
            llm_steps = ["initialisation", "keywords", "dk_classification"]
            if self.step.step_id in llm_steps:
                self.provider_model_label.setText("⚠️ No provider configured")
                self.provider_model_label.setStyleSheet("color: #ff9800; font-style: italic;")
            else:
                # Non-LLM steps (like search) don't need provider info
                self.provider_model_label.setText("No LLM required")
                self.provider_model_label.setStyleSheet("color: #666; font-style: italic;")
            return

        # Build display text with visual indicators
        display_parts = []
        style_color = "#666"

        # Check if this looks like a task preference (basic heuristic)
        task_preference_indicators = []
        if hasattr(self.step, 'selection_reason') and self.step.selection_reason:
            if "task preference" in self.step.selection_reason.lower():
                task_preference_indicators.append("⭐")
                style_color = "#2e7d32"  # Green for task preferences
            elif "provider" in self.step.selection_reason.lower():
                task_preference_indicators.append("🔧")
                style_color = "#1976d2"  # Blue for provider preferences
            elif "fallback" in self.step.selection_reason.lower():
                task_preference_indicators.append("🔄")
                style_color = "#ff9800"  # Orange for fallbacks

        # Format provider/model display
        indicator_prefix = "".join(task_preference_indicators)
        if indicator_prefix:
            display_parts.append(f"{indicator_prefix} {self.step.provider}/{self.step.model}")
        else:
            display_parts.append(f"{self.step.provider}/{self.step.model}")

        # Add compact selection reason if available
        if hasattr(self.step, 'selection_reason') and self.step.selection_reason:
            reason_short = self.step.selection_reason.replace("task preference", "TP").replace("provider preferences", "PP")
            if len(reason_short) < 30:  # Only show if compact enough
                display_parts.append(f"({reason_short})")

        display_text = " ".join(display_parts)
        self.provider_model_label.setText(display_text)
        self.provider_model_label.setStyleSheet(f"color: {style_color};")

        # Set tooltip with full details
        tooltip_parts = [f"Provider: {self.step.provider}", f"Model: {self.step.model}"]
        if hasattr(self.step, 'selection_reason') and self.step.selection_reason:
            tooltip_parts.append(f"Source: {self.step.selection_reason}")
        self.provider_model_label.setToolTip("\n".join(tooltip_parts))

    def update_status_display(self):
        """Update visual status indicator - Claude Generated"""
        if self.step.status == "pending":
            self.status_label.setText("▷")
            self.status_label.setStyleSheet("color: #999;")
            self.setStyleSheet(
                "QFrame { border-color: #ddd; background-color: #fafafa; }"
            )

        elif self.step.status == "running":
            self.status_label.setText("▶")
            self.status_label.setStyleSheet("color: #2196f3;")
            self.setStyleSheet(
                "QFrame { border-color: #2196f3; background-color: #e3f2fd; }"
            )

        elif self.step.status == "completed":
            self.status_label.setText("✓")
            self.status_label.setStyleSheet("color: #4caf50;")
            self.setStyleSheet(
                "QFrame { border-color: #4caf50; background-color: #e8f5e8; }"
            )

        elif self.step.status == "error":
            self.status_label.setText("✗")
            self.status_label.setStyleSheet("color: #d32f2f;")
            self.setStyleSheet(
                "QFrame { border-color: #d32f2f; background-color: #ffebee; }"
            )

        # Always update provider/model display when status changes - Claude Generated
        self._update_provider_model_display()

    def set_content(self, content_widget: QWidget):
        """Set the content widget for this step - Claude Generated"""
        # Clear existing content
        for i in reversed(range(self.content_layout.count())):
            child = self.content_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        # Add new content
        self.content_layout.addWidget(content_widget)

    def update_step_data(self, step: PipelineStep):
        """Update step data and refresh display - Claude Generated"""
        self.step = step
        self.update_status_display()

    def refresh_styles(self):
        """Re-apply fonts after global font-size change. — Claude Generated"""
        from .styles import get_scaled_font
        if hasattr(self, "_name_label"):
            self._name_label.setFont(get_scaled_font(size_delta=+2, bold=True))


class PipelineTab(QWidget):
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
        "selection": ("keywords", 3),
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

        # Load catalog configuration
        self.catalog_token, self.catalog_search_url, self.catalog_details_url = self._load_catalog_config()

        # Use injected central PipelineManager instead of creating redundant instance - Claude Generated
        self.pipeline_manager = pipeline_manager
        
        # Update pipeline config with catalog settings
        self._update_pipeline_config_with_catalog_settings()

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
            "Workflow für Agent-Modus (v4 YAMLs aus workflows/).\n"
            "alima_classic: 4-Step ALIMA-Pipeline (default).\n"
            "Wird nur ausgeführt, wenn '🤖 Agentic' aktiv ist."
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

        self.global_override_combo = QComboBox()
        self.global_override_combo.setMinimumWidth(180)
        self.global_override_combo.setMaximumWidth(300)
        self.global_override_combo.setToolTip(
            "Provider/Modell für alle LLM-Schritte.\n"
            "\"-- Standard --\" = Aus Konfiguration/Task-Präferenzen"
        )
        self.global_override_combo.setStyleSheet(_combo_css)
        self._populate_global_override_combo()
        tb_layout.addWidget(self.global_override_combo)

        # Agentic mode controls
        agentic_sep = QFrame()
        agentic_sep.setFrameShape(QFrame.Shape.VLine)
        agentic_sep.setFixedHeight(24)
        agentic_sep.setStyleSheet("color: #ccc;")
        tb_layout.addWidget(agentic_sep)

        self.agentic_mode_checkbox = QCheckBox("🤖 Agentic")
        self.agentic_mode_checkbox.setToolTip(
            "Verwendet LLM-gesteuerte Agenten mit MCP-Tools statt sequenzieller Pipeline.\n"
            "⚠️ Experimentell: Erhöht Token-Nutzung um ca. 3x"
        )
        self.agentic_mode_checkbox.setChecked(False)
        self.agentic_mode_checkbox.stateChanged.connect(self.on_agentic_mode_toggled)
        tb_layout.addWidget(self.agentic_mode_checkbox)

        tb_layout.addStretch()

        # Pipeline status label
        self.pipeline_status_label = QLabel("Bereit")
        self.pipeline_status_label.setStyleSheet("color: #666; padding-left: 8px;")
        tb_layout.addWidget(self.pipeline_status_label)

        main_layout.addWidget(self.toolbar_frame)

    def _populate_global_override_combo(self):
        """Populate LLM model selector with available provider/model pairs - Claude Generated"""
        try:
            self.global_override_combo.clear()
            self.global_override_combo.addItem("-- Standard --", None)
            from ..utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            unified_config = config_manager.get_unified_config()
            for provider in unified_config.get_enabled_providers():
                models = getattr(provider, 'available_models', []) or []
                if not models and getattr(provider, 'preferred_model', None):
                    models = [provider.preferred_model]
                for model in models:
                    self.global_override_combo.addItem(f"{provider.name} | {model}", f"{provider.name}|{model}")
        except Exception as e:
            self.logger.error(f"Error populating LLM combo: {e}")

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

    def create_input_step_widget(self) -> QWidget:
        """Create unified input step widget - Claude Generated"""
        # Create unified input widget
        self.unified_input = UnifiedInputWidget(
            llm_service=self.llm_service, alima_manager=self.alima_manager
        )

        # Connect signals
        self.unified_input.text_ready.connect(self.on_input_text_ready)
        self.unified_input.input_cleared.connect(self.on_input_cleared)

        return self.unified_input

    def on_input_text_ready(self, text: str, source_info: str):
        """Handle ready input text - Claude Generated"""
        self.logger.info(f"Input text ready: {len(text)} chars from {source_info}")

        # Update the input step
        input_step = self._get_step_by_id("input")
        if input_step:
            input_step.output_data = {
                "text": text,
                "source_info": source_info,
                "timestamp": datetime.now().isoformat(),
            }
            input_step.status = "completed"

            # Update step widget
            if "input" in self.step_widgets:
                self.step_widgets["input"].update_step_data(input_step)

        # Store text for pipeline
        self.current_input_text = text
        self.current_source_info = source_info
        # Capture source type/data from input widget - Claude Generated
        self.current_input_type = getattr(self.unified_input, 'current_source_type', 'text')
        self.current_input_source = getattr(self.unified_input, 'current_source_data', '')

    def on_input_cleared(self):
        """Handle input clearing - Claude Generated"""
        self.current_input_text = ""
        self.current_source_info = ""
        self.current_input_type = "text"  # Reset source tracking - Claude Generated
        self.current_input_source = ""

        # Reset input step
        input_step = self._get_step_by_id("input")
        if input_step:
            input_step.status = "pending"
            input_step.output_data = None

            if "input" in self.step_widgets:
                self.step_widgets["input"].update_step_data(input_step)

    def _get_step_by_id(self, step_id: str) -> Optional[PipelineStep]:
        """Get step by ID - Claude Generated"""
        for step_widget in self.step_widgets.values():
            if step_widget.step.step_id == step_id:
                return step_widget.step
        return None

    def _create_text_result_widget(
        self, label_text: str, placeholder: str, min_height: int = 80, max_height: int = 300
    ) -> tuple[QWidget, QTextEdit]:
        """
        Helper method to create standardized text result widgets.
        Returns tuple of (widget, text_edit) for consistent layout.
        Claude Generated
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(4)  # Reduziert von default - Claude Generated
        layout.setContentsMargins(0, 0, 0, 0)  # Keine extra margins

        # Label kompakter gestylt
        label = QLabel(label_text)
        label.setStyleSheet(
            "font-weight: bold; color: #555; padding: 2px;"
        )
        label.setMaximumHeight(18)  # Explizite Height
        label.setWordWrap(False)  # Keine Zeilenumbrüche

        # Results area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        #text_edit.setMinimumHeight(min_height)
        #text_edit.setMaximumHeight(max_height)
        text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        text_edit.setPlaceholderText(placeholder)

        layout.addWidget(label, 0)  # Stretch 0
        layout.addWidget(text_edit, 1)  # Stretch 1

        return widget, text_edit

    def create_initialisation_step_widget(self) -> QWidget:
        """Create initialisation step widget - Claude Generated"""
        widget, self.initialisation_result = self._create_text_result_widget(
            label_text="Extrahierte freie Schlagworte:",
            placeholder="Freie Schlagworte werden hier angezeigt..."
        )
        return widget

    def create_search_step_widget(self) -> QWidget:
        """Create search step widget — sortable GND-hit table with selection filter - Claude Generated

        Shows every catalog hit with a GND-ID (Begriff / GND-ID / Häufigkeit /
        Auswahl). A checkbox hides the entries that were deselected during the
        keyword chunking/selection step.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        label = QLabel("GND-Suchergebnisse:")
        label.setStyleSheet("font-weight: bold; color: #555; padding: 2px;")
        label.setMaximumHeight(18)
        header.addWidget(label, 0)
        header.addStretch(1)
        self.search_show_selected_only = QCheckBox("Nur ausgewählte anzeigen")
        self.search_show_selected_only.setToolTip(
            "Im Chunking abgewählte GND-Treffer ausblenden"
        )
        self.search_show_selected_only.stateChanged.connect(self._filter_gnd_hits)
        header.addWidget(self.search_show_selected_only, 0)
        layout.addLayout(header, 0)

        self.search_results_table = QTableWidget()
        self.search_results_table.setColumnCount(4)
        self.search_results_table.setHorizontalHeaderLabels(
            ["Begriff", "GND-ID", "Häufigkeit", "Auswahl"]
        )
        self.search_results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.search_results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.search_results_table.setSortingEnabled(True)
        self.search_results_table.verticalHeader().setVisible(False)
        hh = self.search_results_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.search_results_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.search_results_table, 1)

        # Raw state for re-filtering / re-marking selection - Claude Generated
        self.search_raw_rows = []
        self.search_selected_ids = set()
        self.search_selected_labels = set()
        return widget

    def _populate_gnd_hits(self, search_results, selected=None) -> None:
        """Fill the GND-Recherche table from search_results (+ optional selection).

        ``search_results`` may be the classic dict form, a List[SearchResult], or
        agentic ``gnd_entries`` — PipelineResultFormatter.flatten_gnd_hits handles
        all three. ``selected`` (final keyword list) marks which hits survived the
        chunking step; pass None to leave the current selection untouched. - Claude Generated
        """
        if not hasattr(self, "search_results_table"):
            return
        self.search_raw_rows = PipelineResultFormatter.flatten_gnd_hits(search_results)
        if selected is not None:
            self.search_selected_ids, self.search_selected_labels = (
                PipelineResultFormatter.extract_selected_gnd_keys(selected)
            )
        self._render_gnd_hits_table()

    def _mark_gnd_selection(self, selected) -> None:
        """Re-mark which existing GND rows are selected (final keywords known) - Claude Generated"""
        if not hasattr(self, "search_results_table") or not self.search_raw_rows:
            return
        self.search_selected_ids, self.search_selected_labels = (
            PipelineResultFormatter.extract_selected_gnd_keys(selected)
        )
        self._render_gnd_hits_table()

    def _render_gnd_hits_table(self) -> None:
        """Render search_raw_rows into the table, highlighting selected hits - Claude Generated"""
        table = getattr(self, "search_results_table", None)
        if table is None:
            return
        table.setSortingEnabled(False)
        table.setRowCount(0)
        ids = self.search_selected_ids
        labels = self.search_selected_labels
        have_selection = bool(ids or labels)

        for row in self.search_raw_rows:
            is_selected = (row["gnd_id"] in ids) or (
                row["begriff"].lower() in labels
            )
            r = table.rowCount()
            table.insertRow(r)

            begriff_item = QTableWidgetItem(row["begriff"])
            gnd_item = QTableWidgetItem(row["gnd_id"])
            count_item = QTableWidgetItem()
            count_item.setData(Qt.ItemDataRole.DisplayRole, int(row.get("count", 0)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            sel_item = QTableWidgetItem(
                "✅" if is_selected else ("" if have_selection else "—")
            )
            sel_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if is_selected:
                for it in (begriff_item, gnd_item, count_item, sel_item):
                    it.setForeground(QColor("#2e7d32"))
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
            if row.get("search_terms"):
                begriff_item.setToolTip(
                    "Gefunden über: " + ", ".join(row["search_terms"])
                )
            # Stash selection flag for the visibility filter.
            begriff_item.setData(Qt.ItemDataRole.UserRole, is_selected)

            table.setItem(r, 0, begriff_item)
            table.setItem(r, 1, gnd_item)
            table.setItem(r, 2, count_item)
            table.setItem(r, 3, sel_item)

        table.setSortingEnabled(True)
        self._filter_gnd_hits()

    def _filter_gnd_hits(self) -> None:
        """Hide deselected rows when 'Nur ausgewählte anzeigen' is checked - Claude Generated"""
        table = getattr(self, "search_results_table", None)
        if table is None or not hasattr(self, "search_show_selected_only"):
            return
        only_selected = self.search_show_selected_only.isChecked()
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            is_sel = bool(item.data(Qt.ItemDataRole.UserRole)) if item else False
            table.setRowHidden(r, only_selected and not is_sel)

    def create_keywords_step_widget(self) -> QWidget:
        """Create keywords step widget (Verbale Erschließung) - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Finale Schlagworte ────────────────────────────────────────────────
        kw_label = QLabel("Finale GND-Schlagworte:")
        kw_label.setStyleSheet("font-weight: bold; color: #555; padding: 2px;")
        kw_label.setMaximumHeight(18)
        layout.addWidget(kw_label, 0)

        self.keywords_result = QTextEdit()
        self.keywords_result.setReadOnly(True)
        self.keywords_result.setPlaceholderText("Finale Schlagworte werden hier angezeigt...")
        self.keywords_result.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.keywords_result, 1)

        # ── Schlagwortketten ─────────────────────────────────────────────────
        chains_label = QLabel("Schlagwortketten (mit Verifikation):")
        chains_label.setStyleSheet("font-weight: bold; color: #555; padding: 2px;")
        chains_label.setMaximumHeight(18)
        layout.addWidget(chains_label, 0)

        self.keyword_chains_result = QTextEdit()
        self.keyword_chains_result.setReadOnly(True)
        self.keyword_chains_result.setPlaceholderText("Schlagwortketten erscheinen nach Abschluss der Verschlagwortung...")
        self.keyword_chains_result.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.keyword_chains_result, 2)

        return widget

    def create_dk_search_step_widget(self) -> QWidget:
        """Create DK search step with splitter for controls/results - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ═══ Splitter zwischen Controls und Results ═══
        self.dk_search_splitter = QSplitter(Qt.Orientation.Vertical)
        self.dk_search_splitter.setChildrenCollapsible(True)  # Allow collapse

        # Top: Controls (Config + Filter)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(8)

        # Config Section (kompakter)
        config_header = QLabel("⚙️ Katalog-Such-Konfiguration")
        config_header.setStyleSheet("font-weight: bold; color: #555;")
        controls_layout.addWidget(config_header)

        # Kompakte Grid-Layout statt 3 separate Rows
        config_grid = QGridLayout()
        config_grid.setSpacing(8)

        # Row 0: Max Results + Frequency (nebeneinander)
        config_grid.addWidget(QLabel("Max. Ergebnisse:"), 0, 0)
        self.dk_search_max_results = QSpinBox()
        self.dk_search_max_results.setRange(5, 100)
        from ..utils.pipeline_defaults import DEFAULT_DK_MAX_RESULTS
        self.dk_search_max_results.setValue(DEFAULT_DK_MAX_RESULTS)
        self.dk_search_max_results.setToolTip("Max. Katalog-Suchergebnisse pro Keyword")
        config_grid.addWidget(self.dk_search_max_results, 0, 1)

        config_grid.addWidget(QLabel("Min. Häufigkeit:"), 0, 2)
        self.dk_frequency_threshold = QSpinBox()
        self.dk_frequency_threshold.setRange(1, 50)
        from ..utils.pipeline_defaults import DEFAULT_DK_FREQUENCY_THRESHOLD
        self.dk_frequency_threshold.setValue(DEFAULT_DK_FREQUENCY_THRESHOLD)
        self.dk_frequency_threshold.setToolTip("Nur Klassifikationen mit >= N Vorkommen")
        config_grid.addWidget(self.dk_frequency_threshold, 0, 3)

        config_grid.setColumnStretch(4, 1)  # Push to left
        #controls_layout.addLayout(config_grid)

        # Row 1: Force Update Checkbox
        from PyQt6.QtWidgets import QCheckBox
        self.force_update_checkbox = QCheckBox("Katalog-Cache ignorieren")
        self.force_update_checkbox.setToolTip(
            "Erzwingt Live-Suche im Katalog und ignoriert gecachte Ergebnisse."
        )
        self.force_update_checkbox.setChecked(False)
        #controls_layout.addWidget(self.force_update_checkbox)

        # Filter Section (kompakter)
        filter_header = QLabel("🔍 Ergebnisse filtern")
        filter_header.setStyleSheet("font-weight: bold; color: #555;")
        controls_layout.addWidget(filter_header)

        # Filter Grid
        filter_grid = QGridLayout()
        filter_grid.setSpacing(8)

        # Row 0: Search + Clear + Mode + Count
        filter_grid.addWidget(QLabel("Suchen:"), 0, 0)
        self.dk_search_filter_input = QLineEdit()
        self.dk_search_filter_input.setPlaceholderText("Filter eingeben...")
        self.dk_search_filter_input.textChanged.connect(self._filter_dk_search_results)
        filter_grid.addWidget(self.dk_search_filter_input, 0, 1, 1, 2)  # Span 2 cols

        clear_filter_btn = QPushButton("×")
        clear_filter_btn.setMaximumWidth(30)
        clear_filter_btn.setToolTip("Filter löschen")
        clear_filter_btn.clicked.connect(lambda: self.dk_search_filter_input.clear())
        filter_grid.addWidget(clear_filter_btn, 0, 3)

        filter_grid.addWidget(QLabel("Modus:"), 0, 4)
        self.dk_filter_mode = QComboBox()
        self.dk_filter_mode.addItems(["Alle", "Titel", "Klassifikationscodes", "Keywords"])
        self.dk_filter_mode.setStyleSheet(
            "QComboBox { padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px; }"
            "QComboBox QAbstractItemView { background-color: #2b2b2b; color: #ccc; "
            "selection-background-color: #005fcc; selection-color: white; border: 1px solid #ccc; }"
        )
        self.dk_filter_mode.currentTextChanged.connect(self._filter_dk_search_results)
        filter_grid.addWidget(self.dk_filter_mode, 0, 5)

        self.dk_filter_count_label = QLabel("")
        self.dk_filter_count_label.setStyleSheet("color: #666;")
        filter_grid.addWidget(self.dk_filter_count_label, 0, 6)

        filter_grid.setColumnStretch(7, 1)  # Push to left
        controls_layout.addLayout(filter_grid)

        controls_layout.addStretch()  # Push controls to top
        self.dk_search_splitter.addWidget(controls_widget)

        # Bottom: Results
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(5, 5, 5, 5)

        results_header = QLabel("📊 Katalog-Suchergebnisse")
        results_header.setStyleSheet("font-weight: bold; color: #555;")
        results_layout.addWidget(results_header)

        self.dk_search_raw_data = []  # Store for filtering

        self.dk_search_results = QTextEdit()
        self.dk_search_results.setReadOnly(True)
        self.dk_search_results.setMinimumHeight(80)
        # KEIN setMaximumHeight mehr! - Claude Generated
        self.dk_search_results.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dk_search_results.setPlaceholderText(
            "Katalog-Suchergebnisse für DK/RVK-Klassifikationen..."
        )
        results_layout.addWidget(self.dk_search_results)
        self.dk_search_splitter.addWidget(results_widget)

        # Splitter ratio: 25% controls, 75% results
        self.dk_search_splitter.setStretchFactor(0, 1)
        self.dk_search_splitter.setStretchFactor(1, 3)
        self.dk_search_splitter.setSizes([120, 360])  # Initial

        layout.addWidget(self.dk_search_splitter)
        # ═══ END Splitter ═══

        return widget

    def create_dk_classification_step_widget(self) -> QWidget:
        """Create DK classification step widget with splitter between input/results - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ═══ Splitter zwischen Input und Results ═══
        self.dk_classification_splitter = QSplitter(Qt.Orientation.Vertical)
        self.dk_classification_splitter.setChildrenCollapsible(False)

        # Top: Input Summary
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(5, 5, 5, 5)

        # Header statt GroupBox
        input_header = QLabel("📥 Eingangsdaten für LLM-Klassifikation")
        input_header.setStyleSheet("font-weight: bold; color: #555;")
        input_layout.addWidget(input_header)

        self.dk_input_summary = QTextEdit()
        self.dk_input_summary.setReadOnly(True)
        self.dk_input_summary.setMinimumHeight(60)  # Reduziert von 80 - Claude Generated
        # KEIN setMaximumHeight mehr! - Claude Generated
        self.dk_input_summary.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dk_input_summary.setPlaceholderText(
            "Zusammenfassung der Katalog-Suchergebnisse für LLM..."
        )
        input_layout.addWidget(self.dk_input_summary)
        self.dk_classification_splitter.addWidget(input_widget)

        # Bottom: Results Display
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(5, 5, 5, 5)

        # Header statt GroupBox
        results_header = QLabel("✅ Finale DK/RVK-Klassifikationen")
        results_header.setStyleSheet("font-weight: bold; color: #555;")
        results_layout.addWidget(results_header)

        self.dk_classification_results = QTextEdit()
        self.dk_classification_results.setReadOnly(True)
        self.dk_classification_results.setMinimumHeight(60)  # Reduziert von 80 - Claude Generated
        # KEIN setMaximumHeight mehr! - Claude Generated
        self.dk_classification_results.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.dk_classification_results.setPlaceholderText(
            "Finale DK/RVK-Klassifikationen vom LLM werden hier angezeigt...\n"
            "Format: DK 666.76, RVK Q12, RVK QC 130, ..."
        )
        results_layout.addWidget(self.dk_classification_results)
        self.dk_classification_splitter.addWidget(results_widget)

        # Splitter ratio: 30% input, 70% results (results wichtiger)
        self.dk_classification_splitter.setStretchFactor(0, 3)
        self.dk_classification_splitter.setStretchFactor(1, 7)
        self.dk_classification_splitter.setSizes([100, 250])  # Initial

        layout.addWidget(self.dk_classification_splitter)
        # ═══ END Splitter ═══

        # Statistics Label (bleibt)
        self.dk_compact_stats = QLabel()
        self.dk_compact_stats.setWordWrap(True)
        self.dk_compact_stats.setTextFormat(Qt.TextFormat.RichText)
        self.dk_compact_stats.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.dk_compact_stats)

        return widget

    def save_splitter_state(self, settings):
        """Save splitter positions to QSettings - Claude Generated"""
        if hasattr(self, 'main_splitter'):
            settings.setValue("pipeline/main_splitter", self.main_splitter.saveState())

    def restore_splitter_state(self, settings):
        """Restore splitter positions from QSettings - Claude Generated"""
        state = settings.value("pipeline/main_splitter")
        if state and hasattr(self, 'main_splitter'):
            self.main_splitter.restoreState(state)

    def _filter_dk_search_results(self):
        """Filter displayed DK search results based on search input - Claude Generated"""
        if not hasattr(self, 'dk_search_raw_data') or not self.dk_search_raw_data:
            return

        filter_text = self.dk_search_filter_input.text().strip().lower()
        filter_mode = self.dk_filter_mode.currentText()

        # Ohne Filter: alle Ergebnisse anzeigen
        if not filter_text:
            self._display_dk_search_results(self.dk_search_raw_data)
            self.dk_filter_count_label.setText("")
            return

        # Filter anwenden
        filtered_results = []
        for result in self.dk_search_raw_data:
            dk_code = result.get("dk", "").lower()
            titles = [t.lower() for t in result.get("titles", [])]
            keywords = [k.lower() for k in result.get("keywords", [])]

            match = False
            if filter_mode == "Alle":
                match = (filter_text in dk_code or
                        any(filter_text in title for title in titles) or
                        any(filter_text in kw for kw in keywords))
            elif filter_mode == "Titel":
                match = any(filter_text in title for title in titles)
            elif filter_mode == "Klassifikationscodes":
                match = filter_text in dk_code
            elif filter_mode == "Keywords":
                match = any(filter_text in kw for kw in keywords)

            if match:
                filtered_results.append(result)

        self._display_dk_search_results(filtered_results)
        self.dk_filter_count_label.setText(
            f"Zeige {len(filtered_results)} von {len(self.dk_search_raw_data)} Ergebnissen"
        )

    def _display_dk_search_results(self, results: List[Dict[str, Any]]):
        """Display DK search results with formatting - Claude Generated

        Formatting delegates to ``PipelineResultFormatter.format_dk_search_results_text``
        (shared with the Agentic-Chat); this method keeps only the widget-side
        empty-state handling.
        """
        if not results:
            self.dk_search_results.setPlainText(
                "Keine Ergebnisse gefunden" if hasattr(self, 'dk_search_filter_input')
                and self.dk_search_filter_input.text()
                else "Keine DK/RVK-Klassifikationen gefunden"
            )
            return

        self.dk_search_results.setPlainText(
            PipelineResultFormatter.format_dk_search_results_text(results)
        )

    def _format_dk_classifications_with_titles(
        self,
        dk_classifications: List[str],
        dk_search_results: List[Dict[str, Any]],
        max_titles_per_code: int = 5
    ) -> str:
        """Format final classifications with catalog titles using HTML - Claude Generated

        Delegates to the shared ``PipelineResultFormatter`` (single source of
        truth, also used by the Agentic-Chat). The shared formatter returns an
        HTML fragment; the Pipeline-Tab wraps it in a body with the Arial base
        font for ``QTextEdit.setHtml``.
        """
        fragment = PipelineResultFormatter.format_dk_classifications_html(
            dk_classifications, dk_search_results, max_titles_per_code
        )
        if not dk_classifications:
            return fragment
        return (
            "<html><body style='font-family: Arial, sans-serif;'>"
            f"{fragment}</body></html>"
        )

    @staticmethod
    def _split_classification_code(classification: str) -> tuple[str, str]:
        """Split a prefixed classification string into (system, code)."""
        return PipelineResultFormatter.split_classification_code(classification)

    def _get_titles_for_dk_code(
        self,
        dk_code: str,
        dk_search_results: List[Dict[str, Any]]
    ) -> tuple[list, int]:
        """Extract titles for a specific classification code - Claude Generated"""
        return PipelineResultFormatter.get_titles_for_dk_code(dk_code, dk_search_results)

    def start_auto_pipeline(self):
        """Start the automatic pipeline in background thread - Claude Generated"""
        # Get input text — prefer confirmed value, fall back to whatever is in text_display.
        # This handles the case where the user pastes text directly without clicking "Text verwenden". - Claude Generated
        input_text = getattr(self, "current_input_text", "")
        if not input_text and hasattr(self, 'unified_input'):
            input_text = self.unified_input.text_display.toPlainText().strip()

        if not input_text:
            QMessageBox.warning(
                self,
                "Keine Eingabe",
                "Bitte wählen Sie eine Eingabequelle und stellen Sie Text bereit.",
            )
            return

        # Apply LLM model selection + DK config - Claude Generated
        self._apply_global_override_from_gui()
        self._update_dk_config_from_gui()

        # Stop any existing worker
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.quit()
            self.pipeline_worker.wait()

        # Reset streaming widget for new pipeline
        if hasattr(self, "stream_widget"):
            self.stream_widget.reset_for_new_pipeline()

        # Update status and button visibility - Claude Generated
        self.pipeline_status_label.setText("Pipeline läuft...")
        self.auto_pipeline_button.setEnabled(False)
        self.stop_pipeline_button.setVisible(True)

        # Get force_update flag from checkbox - Claude Generated
        force_update = getattr(self, 'force_update_checkbox', None)
        force_update_enabled = force_update.isChecked() if force_update else False

        # Determine source metadata for working title - Claude Generated
        # Priority: cached value from last text_ready > widget's current_source_type > doi_url_input field
        input_type = getattr(self, 'current_input_type', 'text')
        input_source = getattr(self, 'current_input_source', '')
        if input_type == 'text' and hasattr(self, 'unified_input'):
            # Read fresh from widget — set by extract_text() on last DOI/PDF/image resolution
            widget_type = getattr(self.unified_input, 'current_source_type', 'text')
            widget_data = getattr(self.unified_input, 'current_source_data', '')
            if widget_type != 'text' and widget_data:
                input_type = widget_type
                input_source = widget_data
            else:
                # Last fallback: read the DOI input field directly
                doi_val = self.unified_input.doi_url_input.text().strip()
                if doi_val:
                    input_type = 'url' if doi_val.startswith(('http://', 'https://')) else 'doi'
                    input_source = doi_val

        # Rebuild agentic-context panels from the selected workflow so the
        # widget reflects the *current* workflow instead of the hardcoded
        # 5-step pipeline view. Only relevant when agentic mode is on.
        self._rebuild_agentic_panels()

        # Create and start worker thread - Claude Generated
        self.pipeline_worker = PipelineWorker(
            self.pipeline_manager, input_text,
            input_type=input_type,
            input_source=input_source,
            force_update=force_update_enabled
        )

        # Connect worker signals
        self.pipeline_worker.step_started.connect(self.on_step_started)
        self.pipeline_worker.step_completed.connect(self.on_step_completed)
        self.pipeline_worker.step_error.connect(self.on_step_error)
        self.pipeline_worker.pipeline_error.connect(self.on_pipeline_error)  # Claude Generated
        self.pipeline_worker.pipeline_completed.connect(self.on_pipeline_completed)
        self.pipeline_worker.stream_token.connect(self.on_llm_stream_token)
        self.pipeline_worker.aborted.connect(self.on_pipeline_aborted)  # Claude Generated
        self.pipeline_worker.repetition_detected.connect(self.on_repetition_detected)  # Claude Generated (2026-02-17)
        # Forward agentic context updates to MainWindow dock via signal - Claude Generated
        self.pipeline_worker.agentic_context_updated.connect(self.agentic_context_updated)
        # Also update classical step tabs from agentic snapshots in real-time - Claude Generated
        self.pipeline_worker.agentic_context_updated.connect(self._on_agentic_step_snapshot)

        # Start the worker
        self.pipeline_worker.start()

        # Emit pipeline started signal
        self.pipeline_started.emit("pipeline_thread")

        # NOTE: the chat panel's "🚀 Pipeline gestartet" line is driven by the
        # state.pipeline_started bus event (with the real pipeline UUID). The
        # previous explicit stream_widget.on_pipeline_started("pipeline_thread")
        # call here produced a duplicate started-banner with a placeholder ID. - Claude Generated

    def _apply_global_override_from_gui(self):
        """Apply LLM model selection to pipeline config (global_provider/model_override) - Claude Generated"""
        if not hasattr(self, 'global_override_combo'):
            return
        override_data = self.global_override_combo.currentData()
        config = self.pipeline_manager.config
        if not config:
            return
        if override_data:
            provider, model = PipelineConfig.parse_override_string(override_data)
            config.global_provider_override = provider
            config.global_model_override = model
            self.logger.info(f"🤖 LLM selected: {provider}/{model}")
        else:
            config.global_provider_override = None
            config.global_model_override = None

    def _update_dk_config_from_gui(self):
        """
        Update DK pipeline configuration from GUI widgets - Claude Generated
        Applies current GUI spinner values to PipelineManager configuration
        """
        if not hasattr(self.pipeline_manager, 'config') or not self.pipeline_manager.config:
            return

        config = self.pipeline_manager.config

        # Update dk_search step config
        if 'dk_search' in config.step_configs:
            dk_search_config = config.step_configs['dk_search']
            if hasattr(self, 'dk_search_max_results'):
                dk_search_config.custom_params['max_results'] = self.dk_search_max_results.value()

        # Update dk_classification step config
        if 'dk_classification' in config.step_configs:
            dk_classification_config = config.step_configs['dk_classification']
            if hasattr(self, 'dk_frequency_threshold'):
                dk_classification_config.custom_params['dk_frequency_threshold'] = self.dk_frequency_threshold.value()

        self.logger.info(
            f"✅ DK config updated from GUI: max_results={self.dk_search_max_results.value()}, "
            f"frequency_threshold={self.dk_frequency_threshold.value()}"
        )

    def show_pipeline_config(self):
        """Show pipeline configuration dialog - Claude Generated"""
        prompt_service = None
        if hasattr(self.alima_manager, "prompt_service"):
            prompt_service = self.alima_manager.prompt_service

        # Get config_manager for provider preferences integration - Claude Generated
        config_manager = getattr(self.alima_manager, 'config_manager', None) or getattr(self.llm_service, 'config_manager', None)
        
        dialog = PipelineConfigDialog(
            llm_service=self.llm_service,
            prompt_service=prompt_service,
            current_config=self.pipeline_manager.config,
            config_manager=config_manager,
            parent=self,
        )
        dialog.config_saved.connect(self.on_config_saved)
        dialog.exec()

    def on_config_saved(self, config: PipelineConfig):
        """Handle saved pipeline configuration - Claude Generated"""
        self.pipeline_manager.set_config(config)

        # Update step widgets to reflect new configuration
        self.update_step_display_from_config()

        QMessageBox.information(
            self,
            "Konfiguration gespeichert",
            "Pipeline-Konfiguration wurde erfolgreich aktualisiert!",
        )

    def load_json_state(self):
        """Load pipeline state from JSON file - Claude Generated"""
        if self.main_window and hasattr(self.main_window, 'load_analysis_state_from_file'):
            self.main_window.load_analysis_state_from_file()
        else:
            self.logger.error("Cannot load JSON: MainWindow not available")

    def update_step_display_from_config(self):
        """Update step widgets based on current configuration - Claude Generated"""
        config = self.pipeline_manager.config

        # Update provider/model display for each step
        for step_id, step_widget in self.step_widgets.items():
            if step_id in config.step_configs:
                step_config = config.step_configs[step_id]

                # Handle both dict and PipelineStepConfig objects - Claude Generated
                if isinstance(step_config, dict):
                    provider = step_config.get("provider") or ""
                    model = step_config.get("model") or ""
                    enabled = step_config.get("enabled", True)
                else:
                    provider = step_config.provider or ""
                    model = step_config.model or ""
                    enabled = step_config.enabled

                # Update step data
                step_widget.step.provider = provider
                step_widget.step.model = model

                # ENHANCED: Add task preference information - Claude Generated
                selection_reason = self._determine_selection_reason(step_id, provider, model)
                step_widget.step.selection_reason = selection_reason

                # Update display (visual styling based on enabled state)
                if not enabled:
                    step_widget.setStyleSheet("QFrame { opacity: 0.5; }")
                else:
                    step_widget.setStyleSheet("")

                step_widget.update_status_display()

    def _determine_selection_reason(self, step_id: str, provider: str, model: str) -> str:
        """Determine why this provider/model was selected for the step - Claude Generated"""
        try:
            # Get config manager from pipeline manager
            config_manager = getattr(self.pipeline_manager, 'config_manager', None)
            if not config_manager:
                return "unknown"

            # Load current config to check task preferences
            config = config_manager.load_config()
            if not config or not hasattr(config, 'task_preferences'):
                return "fallback"

            # Map step_id to task name for task_preferences lookup
            task_name_mapping = {
                "initialisation": "initialisation",
                "keywords": "keywords",
                "dk_classification": "dk_class",
                "image_text_extraction": "image_text_extraction"
            }

            task_name = task_name_mapping.get(step_id)
            if not task_name or task_name not in config.unified_config.task_preferences:
                return "provider preferences" if provider else "default"

            # Check if this provider/model matches task preferences
            task_data = config.unified_config.task_preferences[task_name]
            model_priority = task_data.model_priority if task_data else []

            for rank, priority_entry in enumerate(model_priority, 1):
                candidate_provider = priority_entry.get("provider_name")
                candidate_model = priority_entry.get("model_name")

                if candidate_provider == provider and candidate_model == model:
                    return f"task preference #{rank}"

            # Check chunked preferences
            chunked_priorities = task_data.chunked_model_priority if task_data and task_data.chunked_model_priority else []
            for rank, priority_entry in enumerate(chunked_priorities, 1):
                candidate_provider = priority_entry.get("provider_name")
                candidate_model = priority_entry.get("model_name")

                if candidate_provider == provider and candidate_model == model:
                    return f"chunked preference #{rank}"

            # If we have provider/model but it's not in task preferences
            if provider and model:
                return "provider preferences"
            else:
                return "fallback"

        except Exception as e:
            return f"error: {str(e)[:20]}"

    def reset_pipeline(self):
        """Reset pipeline to initial state - Claude Generated"""
        # Stop any running worker
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.quit()
            self.pipeline_worker.wait()

        self.pipeline_manager.reset_pipeline()

        # Reset timing tracking
        self.step_start_times.clear()
        self.pipeline_start_time = None
        self.current_running_step = None
        self.duration_update_timer.stop()

        # Reset all step widgets
        for step_widget in self.step_widgets.values():
            step_widget.step.status = "pending"
            step_widget.update_status_display()

        # Clear results
        if hasattr(self, "initialisation_result"):
            self.initialisation_result.clear()
        if hasattr(self, "search_results_table"):
            self.search_results_table.setRowCount(0)
            self.search_raw_rows = []
            self.search_selected_ids = set()
            self.search_selected_labels = set()
        if hasattr(self, "keywords_result"):
            self.keywords_result.clear()
        # DK-related widgets - Claude Generated (Fixed widget names)
        if hasattr(self, "dk_classification_results"):
            self.dk_classification_results.clear()
        if hasattr(self, "dk_search_results"):
            self.dk_search_results.clear()
        if hasattr(self, "dk_input_summary"):
            self.dk_input_summary.clear()

        # Reset DK filter controls - Claude Generated
        if hasattr(self, "dk_search_filter_input"):
            self.dk_search_filter_input.clear()
        if hasattr(self, "dk_filter_mode"):
            self.dk_filter_mode.setCurrentIndex(0)  # "Alle Felder"
        if hasattr(self, "dk_filter_count_label"):
            self.dk_filter_count_label.setText("")
        if hasattr(self, "dk_search_raw_data"):
            self.dk_search_raw_data = []

        # Reset status and button states - Claude Generated
        self.pipeline_status_label.setStyleSheet("")
        self.pipeline_status_label.setText("Bereit für Pipeline-Start")
        self.auto_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setVisible(False)

        # Reset stream widget completely - Claude Generated
        if hasattr(self, "stream_widget"):
            self.stream_widget.reset_for_new_pipeline()

    def on_config_changed(self):
        """Handle configuration changes - Claude Generated (Webcam Feature)"""
        self.logger.debug("Pipeline tab: Handling config change")

    def _populate_workflow_combo(self):
        """Populate workflow combo from discovered v4 YAMLs - Claude Generated.

        Scans ``DEFAULT_SEARCH_PATHS`` for workflow files and lists each by
        stem (label shows version). The legacy hardcoded v3 names were
        removed in the Phase 5 cleanup.
        """
        try:
            from src.core.agents.workflow_loader import (
                DEFAULT_SEARCH_PATHS,
                load_workflow,
            )
            self.workflow_combo.clear()
            seen: set = set()
            for base in DEFAULT_SEARCH_PATHS:
                if not base.exists() or not base.is_dir():
                    continue
                for path in sorted(base.glob("*.yaml")):
                    key = path.resolve()
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        wf = load_workflow(path, strict=False)
                        label = f"{path.stem} (v{wf.version})"
                    except Exception:
                        continue
                    self.workflow_combo.addItem(label, path.stem)
            if self.workflow_combo.count() == 0:
                self.workflow_combo.addItem("alima_classic", "alima_classic")
            self.logger.debug(f"Workflow combo populated with {self.workflow_combo.count()} workflows")
            # Wire once — guard against duplicate connects on repopulate.
            try:
                self.workflow_combo.currentIndexChanged.disconnect(
                    self._on_workflow_changed
                )
            except (TypeError, RuntimeError):
                pass
            self.workflow_combo.currentIndexChanged.connect(
                self._on_workflow_changed
            )
        except Exception as e:
            self.logger.error(f"Error populating workflow combo: {e}")

    # Workflow-specific hints for the UnifiedInputWidget. Keys are YAML stems.
    WORKFLOW_HINTS = {
        "title_list_search": (
            "💡 Titel eingeben – eine pro Zeile oder im Fließtext "
            "(LLM extrahiert strukturierte Titel)."
        ),
        "catalog_search": (
            "💡 Suchbegriffe kommagetrennt oder als Liste – werden parallel "
            "gegen SWB / Lobid / Katalog geschickt."
        ),
    }

    def _on_workflow_changed(self, _index: int) -> None:
        """Apply workflow selection: update config + hint text - Claude Generated."""
        if not hasattr(self, "workflow_combo"):
            return
        workflow_name = self.workflow_combo.currentData()
        if workflow_name and self.pipeline_manager and self.pipeline_manager.config:
            self.pipeline_manager.config.workflow_name = workflow_name

        hint = self.WORKFLOW_HINTS.get(workflow_name or "", "")
        if hasattr(self, "unified_input"):
            self.unified_input.set_hint(hint)

        # Keep the context widget panels in sync with the selected workflow.
        self._rebuild_agentic_panels()

    def _rebuild_agentic_panels(self) -> None:
        """Load active workflow YAML and emit workflow def to MainWindow dock.

        No-op when agentic mode is off or the workflow can't be found.
        Failures are logged but never abort the pipeline start.
        """
        if not self.pipeline_manager or not self.pipeline_manager.config:
            return
        if not self.pipeline_manager.config.enable_agentic_mode:
            return

        try:
            from src.core.agents.workflow_loader import (
                find_workflow_file,
                load_workflow,
            )

            wf_name = self.pipeline_manager.config.workflow_name or "alima_classic"
            wf_path = find_workflow_file(wf_name)
            if wf_path is None:
                self.logger.warning(
                    f"Agentic dock: workflow '{wf_name}' not found — panels not rebuilt"
                )
                return
            wf_def = load_workflow(wf_path, strict=False)
            self.agentic_workflow_built.emit(wf_def)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Agentic panel rebuild failed: {e}")

    def on_agentic_mode_toggled(self, state):
        """Handle agentic mode checkbox toggle - Claude Generated"""
        enabled = state == Qt.CheckState.Checked.value

        self.logger.info(f"Agentic mode {'enabled' if enabled else 'disabled'}")

        # Notify MainWindow dock, adjust splitter - Claude Generated
        self.agentic_mode_changed.emit(enabled)
        w = self.main_splitter.width() or 1000
        if enabled:
            # More space for stream; context lives in external dock
            self.main_splitter.setSizes([int(w * 0.45), int(w * 0.55)])
            # Preview workflow panels in dock before first run
            self._rebuild_agentic_panels()
        else:
            self.main_splitter.setSizes([int(w * 0.65), int(w * 0.35)])

        # Update pipeline configuration
        if self.pipeline_manager and self.pipeline_manager.config:
            self.pipeline_manager.config.enable_agentic_mode = enabled
            if enabled:
                # Set workflow if selected
                workflow_name = self.workflow_combo.currentData()
                if workflow_name:
                    self.pipeline_manager.config.workflow_name = workflow_name
                    self.logger.info(f"Workflow set to: {workflow_name}")

                # Show warning about experimental feature
                if self.main_window and hasattr(self.main_window, "global_status_bar"):
                    self.main_window.global_status_bar.show_temporary_message(
                        "🤖 Agentic Modus aktiviert - Experimentell!", 3000
                    )
            else:
                if self.main_window and hasattr(self.main_window, "global_status_bar"):
                    self.main_window.global_status_bar.show_temporary_message(
                        "Agentic Modus deaktiviert", 2000
                    )

    @pyqtSlot(object)
    def on_step_started(self, step: PipelineStep):
        """Handle step started event - Claude Generated"""
        if step.step_id in self.step_widgets:
            self.step_widgets[step.step_id].update_step_data(step)

        # Track step start time
        self.step_start_times[step.step_id] = datetime.now()
        if self.pipeline_start_time is None:
            self.pipeline_start_time = datetime.now()

        # Start live duration updates for this step
        self.current_running_step = step.step_id
        self.duration_update_timer.start()

        # Update global status bar with current provider info
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(step, "provider") and hasattr(step, "model"):
                self.main_window.global_status_bar.update_provider_info(
                    step.provider, step.model
                )
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    step.name, "running"
                )
            if hasattr(self.main_window.global_status_bar, "pipeline_progress"):
                self.main_window.global_status_bar.pipeline_progress.show()

        self.pipeline_status_label.setText(f"Schritt läuft: {step.name}")

        # Auto-jump to current step tab
        if hasattr(self, "pipeline_tabs"):
            self.jump_to_step(step.step_id)

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_step_started(step)

    @pyqtSlot(object)
    def on_step_completed(self, step: PipelineStep):
        """Handle step completed event - Claude Generated"""
        if step.step_id in self.step_widgets:
            self.step_widgets[step.step_id].update_step_data(step)

        # Stop live duration updates for this step
        if self.current_running_step == step.step_id:
            self.duration_update_timer.stop()
            self.current_running_step = None

        # Update global status bar
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    step.name, "completed"
                )

        # Update result displays
        if step.step_id == "initialisation" and step.output_data:
            free_keywords = step.output_data.get("keywords", "")
            self.logger.debug(f"Initialisation step output_data: {step.output_data}")
            self.logger.debug(f"Extracted free keywords: '{free_keywords}'")
            if hasattr(self, "initialisation_result"):
                # keywords is a string, not a list
                self.initialisation_result.setPlainText(free_keywords)
                self.logger.debug(
                    f"Set initialisation_result text to: '{free_keywords}'"
                )

            # Display working title after initialisation - Claude Generated
            if (self.pipeline_manager.current_analysis_state and
                hasattr(self.pipeline_manager.current_analysis_state, 'working_title') and
                self.pipeline_manager.current_analysis_state.working_title):
                working_title = self.pipeline_manager.current_analysis_state.working_title

                # Set working title in stream widget for log filename - Claude Generated
                if hasattr(self, 'stream_widget') and self.stream_widget:
                    self.stream_widget.set_working_title(working_title)

                self.logger.info(f"Displaying working title: {working_title}")
        elif step.step_id == "search" and step.output_data:
            # Populate the GND-hit table from the full search_results (not the
            # text-reduced gnd_treffer). Selection is marked later when the
            # keywords step finishes. - Claude Generated
            state = (
                self.pipeline_manager.current_analysis_state
                if self.pipeline_manager
                else None
            )
            search_results = getattr(state, "search_results", None) if state else None
            if search_results:
                self._populate_gnd_hits(search_results)

        elif step.step_id == "keywords" and step.output_data:
            final_keywords = step.output_data.get("final_keywords", "")
            self.logger.debug(f"Keywords step output_data: {step.output_data}")
            self.logger.debug(f"Final keywords: '{final_keywords}'")
            # Normalise to list for cross-check below - Claude Generated
            if isinstance(final_keywords, list):
                final_keywords_list = final_keywords
                final_keywords_text = "\n".join(final_keywords)
            else:
                final_keywords_text = str(final_keywords)
                final_keywords_list = [l.strip() for l in final_keywords_text.splitlines() if l.strip()]
            if hasattr(self, "keywords_result"):
                self.keywords_result.setPlainText(final_keywords_text)
                self.logger.debug(
                    f"Set keywords_result text to: '{final_keywords_text}'"
                )

            # ── Schlagwortketten mit Verifikation anzeigen ─────────────────── Claude Generated
            if hasattr(self, "keyword_chains_result"):
                llm_analysis = step.output_data.get("llm_analysis")
                chains = llm_analysis.keyword_chains if llm_analysis else []
                self._render_keyword_chains(chains, final_keywords_list)

            # Mark which GND-Recherche hits survived the selection step - Claude Generated
            self._mark_gnd_selection(final_keywords_list)

        elif step.step_id == "dk_search" and step.output_data:
            # Display DK search results with counts and titles - Claude Generated (Enhanced with filtering)
            # Use flattened DK-centric format for display (backward compatibility fallback to original)
            dk_search_results = step.output_data.get("dk_search_results_flattened",
                                                      step.output_data.get("dk_search_results", []))
            if hasattr(self, "dk_search_results"):
                if dk_search_results:
                    # Store raw data for filtering
                    self.dk_search_raw_data = dk_search_results

                    # Display results (will respect any active filter)
                    self._display_dk_search_results(dk_search_results)

                    # Update filter count if filter is active
                    if (hasattr(self, 'dk_search_filter_input') and
                        self.dk_search_filter_input.text().strip()):
                        self._filter_dk_search_results()
                else:
                    self.dk_search_raw_data = []
                    self.dk_search_results.setPlainText("Keine DK/RVK-Klassifikationen gefunden")

        elif step.step_id == "dk_classification" and step.output_data:
            # Display final DK classification results from LLM - Claude Generated
            dk_classifications = step.output_data.get("dk_classifications", [])
            if hasattr(self, "dk_classification_results"):
                if dk_classifications:
                    # Get dk_search_results from previous step for title display
                    dk_search_results = step.output_data.get("dk_search_results_flattened", [])

                    # Generate HTML display with titles
                    html_display = self._format_dk_classifications_with_titles(
                        dk_classifications,
                        dk_search_results
                    )
                    self.dk_classification_results.setHtml(html_display)
                else:
                    self.dk_classification_results.setPlainText("Keine DK/RVK-Klassifikationen generiert")

                # Also update the input summary with search data from previous step
                if hasattr(self, "dk_input_summary"):
                    search_data = step.output_data.get("dk_search_summary", "")
                    if search_data:
                        self.dk_input_summary.setPlainText(search_data)
                    else:
                        self.dk_input_summary.setPlainText("Katalog-Suchergebnisse für LLM-Analyse")

                # Update compact stats - Claude Generated
                if hasattr(self, "dk_compact_stats"):
                    stats = step.output_data.get("statistics")
                    if stats:
                        total = stats.get("total_classifications", 0)
                        dedup = stats.get("deduplication_stats", {})
                        orig = dedup.get("original_count", 0)
                        rate = dedup.get("deduplication_rate", "0%")
                        self.dk_compact_stats.setText(
                            f"📊 <b>Klassifikations-Statistik:</b> {orig} Katalogtreffer → <b>{total}</b> unikale Klassifikationen "
                            f"(Deduplizierungsrate: {rate})"
                        )

        # End any active streaming for this step
        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_step_completed(step)
        
        # Emit results to other tabs based on step type - Claude Generated
        self._emit_step_results_to_tabs(step)

    @pyqtSlot(object, str)
    def on_step_error(self, step: PipelineStep, error_message: str):
        """Handle step error event - Claude Generated"""
        if step.step_id in self.step_widgets:
            self.step_widgets[step.step_id].update_step_data(step)

        # Stop live duration updates for this step
        if self.current_running_step == step.step_id:
            self.duration_update_timer.stop()
            self.current_running_step = None

        # Update global status bar
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    step.name, "error"
                )

        self.pipeline_status_label.setText(f"Fehler: {step.name}")

        # End any active streaming for this step
        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_step_error(step, error_message)

        QMessageBox.critical(
            self,
            "Pipeline-Fehler",
            f"Fehler in Schritt '{step.name}':\n{error_message}",
        )

        # Re-enable start button
        self.auto_pipeline_button.setEnabled(True)

    @pyqtSlot(str)
    def on_pipeline_error(self, error_message: str):
        """Handle pipeline-level failure that escaped step handling - Claude Generated

        Without this the worker thread dies silently and the UI stays in
        "Processing…" forever (see workers.py PipelineWorker.run).
        """
        if self.current_running_step:
            self.duration_update_timer.stop()
            self.current_running_step = None

        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    "Pipeline", "error"
                )

        self.pipeline_status_label.setText("Pipeline-Fehler")

        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        QMessageBox.critical(
            self,
            "Pipeline-Fehler",
            f"Die Pipeline ist mit einem Fehler abgebrochen:\n{error_message}",
        )

        self.auto_pipeline_button.setEnabled(True)

    @pyqtSlot(object)
    def on_pipeline_completed(self, analysis_state):
        """Handle pipeline completion - Claude Generated"""
        # Stop any running timer
        self.duration_update_timer.stop()
        self.current_running_step = None

        self.pipeline_status_label.setText("Pipeline abgeschlossen ✓")
        self.auto_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setVisible(False)
        self.pipeline_completed.emit()

        # Stop status bar timer and progress
        if self.main_window and hasattr(self.main_window, "global_status_bar"):
            if hasattr(self.main_window.global_status_bar, "pipeline_progress"):
                self.main_window.global_status_bar.pipeline_progress.hide()
            if hasattr(self.main_window.global_status_bar, "update_pipeline_status"):
                self.main_window.global_status_bar.update_pipeline_status(
                    "Pipeline", "completed"
                )

        # Propagate working_title to stream widget (agentic: no on_step_completed fires)
        if analysis_state and hasattr(analysis_state, 'working_title') and analysis_state.working_title:
            working_title = analysis_state.working_title
            if hasattr(self, 'stream_widget'):
                self.stream_widget.set_working_title(working_title)

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_pipeline_completed(analysis_state)

        # Emit complete analysis_state for distribution to specialized tabs - Claude Generated
        if analysis_state:
            self.pipeline_results_ready.emit(analysis_state)
            # Sync classical step tabs from agentic result (no step_completed fires in agentic mode) - Claude Generated
            if (self.pipeline_manager and self.pipeline_manager.config
                    and self.pipeline_manager.config.enable_agentic_mode):
                self._sync_classical_tabs_from_state(analysis_state)

        # Optional: Auto-save after completion - Claude Generated
        if hasattr(analysis_state, 'working_title') and analysis_state.working_title:
            from ..utils.pipeline_utils import export_analysis_state_to_file
            from ..utils.pipeline_defaults import get_autosave_dir

            auto_save_dir = get_autosave_dir(getattr(self, 'config_manager', None))
            auto_save_dir.mkdir(parents=True, exist_ok=True)

            auto_save_file = auto_save_dir / f"{analysis_state.working_title}.json"
            try:
                export_analysis_state_to_file(analysis_state, str(auto_save_file))
                self.logger.info(f"✅ Auto-saved pipeline result to: {auto_save_file}")
            except Exception as e:
                self.logger.warning(f"Auto-save failed: {e}")

        QMessageBox.information(
            self,
            "Pipeline abgeschlossen",
            "Die komplette Analyse-Pipeline wurde erfolgreich abgeschlossen!",
        )

    def _render_keyword_chains(self, chains: list, final_keywords_source) -> None:
        """Render Schlagwortketten with green/red verification into keyword_chains_result - Claude Generated"""
        if not hasattr(self, "keyword_chains_result"):
            return
        if not chains:
            self.keyword_chains_result.setPlainText("Keine Schlagwortketten in LLM-Antwort gefunden.")
            return

        import re as _re

        def _norm(kw: str) -> str:
            return _re.sub(r"\s*\(GND-ID:[^)]*\)", "", kw).strip().lower()

        def _esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Normalize final keywords – accepts list[str] or list[dict{keyword}]
        final_set: set = set()
        for item in (final_keywords_source or []):
            if isinstance(item, dict):
                final_set.add(_norm(item.get("keyword", "")))
            else:
                final_set.add(_norm(str(item)))

        blocks = []
        for c in chains:
            parts = c.get("chain", [])
            reason = c.get("reason", "")
            kw_html_parts = []
            all_present = True
            for kw in parts:
                if _norm(kw) in final_set:
                    kw_html_parts.append(f'<span style="color:#4caf50;font-weight:bold">{_esc(kw)}</span>')
                else:
                    kw_html_parts.append(f'<span style="color:#f44336;font-weight:bold">{_esc(kw)} ✗</span>')
                    all_present = False
            arrow = '<span style="color:#888"> → </span>'
            status_color = "#4caf50" if all_present else "#ff9800"
            block = (
                f'<p style="margin:4px 0 0 0">'
                f'<span style="color:{status_color};font-weight:bold">{"✓" if all_present else "⚠"} </span>'
                f'{arrow.join(kw_html_parts)}</p>'
            )
            if reason:
                block += f'<p style="margin:1px 0 6px 16px;color:#aaa;font-style:italic">{_esc(reason)}</p>'
            else:
                block += '<p style="margin:0 0 6px 0"></p>'
            blocks.append(block)

        self.keyword_chains_result.setHtml(
            '<html><body style="font-family:monospace">' + "".join(blocks) + "</body></html>"
        )

    @pyqtSlot(str, dict)
    def _on_agentic_step_snapshot(self, step_id: str, snapshot: dict) -> None:
        """Update classical step-tab widgets from agentic step completion snapshots - Claude Generated.

        Mapping: extraction→initialisation, search→search, selection→keywords,
                 classification→dk_classification, dk_collect→dk_search, dk_postprocess→dk_classification
        """
        status = snapshot.get("_step_status", "")
        if status not in ("completed", "error"):
            return  # skip running/pending intermediate emissions

        # Update PipelineStepWidget status indicator (▷▶✓✗) - Claude Generated
        mapping = self._AGENTIC_STEP_MAP.get(step_id)
        if mapping:
            widget_key, _ = mapping
            step_widget = self.step_widgets.get(widget_key)
            if step_widget:
                step_widget.step.status = status
                step_widget.update_status_display()

        try:
            if step_id == "extraction":
                keywords = snapshot.get("extracted_keywords", [])
                if keywords and hasattr(self, "initialisation_result"):
                    text = "\n".join(keywords) if isinstance(keywords, list) else str(keywords)
                    self.initialisation_result.setPlainText(text)
                working_title = snapshot.get("working_title", "")
                if working_title:
                    if hasattr(self, "stream_widget"):
                        self.stream_widget.set_working_title(working_title)

            elif step_id == "search":
                gnd_entries = snapshot.get("gnd_entries", [])
                if gnd_entries and hasattr(self, "search_results_table"):
                    self._populate_gnd_hits(gnd_entries)

            elif step_id == "selection":
                final_kws = snapshot.get("extra", {}).get("final_keywords", [])
                if final_kws and hasattr(self, "keywords_result"):
                    lines = []
                    for kw in final_kws:
                        if isinstance(kw, dict):
                            lines.append(f"{kw.get('keyword', '')} (GND-ID: {kw.get('gnd_id', '')})")
                        else:
                            lines.append(str(kw))
                    self.keywords_result.setPlainText("\n".join(lines))
                chains = snapshot.get("keyword_chains", [])
                if chains:
                    self._render_keyword_chains(chains, final_kws)
                # Mark which GND-Recherche hits survived selection - Claude Generated
                if final_kws:
                    self._mark_gnd_selection(final_kws)

            elif step_id == "classification":
                # classification step produces dk_classifications before dk_postprocess - Claude Generated
                dk_class = snapshot.get("dk_classifications", [])
                dk_results = snapshot.get("dk_search_results", [])
                if dk_class and hasattr(self, "dk_classification_results"):
                    codes = self._dk_class_codes(dk_class)
                    html_display = self._format_dk_classifications_with_titles(
                        codes, dk_results
                    )
                    self.dk_classification_results.setHtml(html_display)

            elif step_id == "dk_collect":
                dk_results = snapshot.get("dk_search_results", [])
                if dk_results and hasattr(self, "dk_search_results"):
                    self.dk_search_raw_data = dk_results
                    self._display_dk_search_results(dk_results)

            elif step_id == "dk_postprocess":
                dk_results = snapshot.get("dk_search_results", [])
                dk_class = snapshot.get("dk_classifications", [])
                if dk_results and hasattr(self, "dk_search_results"):
                    self.dk_search_raw_data = dk_results
                    self._display_dk_search_results(dk_results)
                if dk_class and hasattr(self, "dk_classification_results"):
                    codes = self._dk_class_codes(dk_class)
                    html_display = self._format_dk_classifications_with_titles(
                        codes, dk_results
                    )
                    self.dk_classification_results.setHtml(html_display)

        except Exception as e:
            self.logger.warning(f"_on_agentic_step_snapshot({step_id}) failed: {e}")

    def _sync_classical_tabs_from_state(self, state) -> None:
        """Populate classical step-tab widgets from analysis_state after agentic run.

        In agentic mode step_completed never fires, so this fills the same widgets
        that on_step_completed() would normally update. - Claude Generated
        """
        try:
            # Init tab: extracted keywords
            if state.initial_keywords and hasattr(self, "initialisation_result"):
                self.initialisation_result.setPlainText("\n".join(state.initial_keywords))

            # Search tab: show ALL GND hits (not just search terms), with the
            # final selection marked so deselected hits can be filtered. - Claude Generated
            if state.search_results and hasattr(self, "search_results_table"):
                selected = (
                    state.final_llm_analysis.extracted_gnd_keywords
                    if state.final_llm_analysis
                    else None
                )
                self._populate_gnd_hits(state.search_results, selected=selected)

            # Keywords tab: final GND keywords
            if hasattr(self, "keywords_result"):
                final_kws = []
                if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
                    final_kws = state.final_llm_analysis.extracted_gnd_keywords
                elif state.initial_keywords:
                    final_kws = state.initial_keywords
                if final_kws:
                    text = "\n".join(final_kws) if isinstance(final_kws, list) else str(final_kws)
                    self.keywords_result.setPlainText(text)

            # DK search + classification tabs.
            # In agentic mode the rich DK-centric catalog data (real titles +
            # counts) lives in state.dk_search_results — written by
            # build_dk_search_results / dk_postprocess and identical to what the
            # per-step snapshot shows during the run. state.dk_search_results_flattened
            # is only a thin structure derived from the final classifications
            # (titles = DK label, count = confidence×100), which clears the
            # Katalog-Recherche view and drops the title list. Prefer the rich
            # source; fall back to flattened only if it is empty. - Claude Generated
            dk_rich = PipelineResultFormatter.select_dk_title_source(
                getattr(state, "dk_search_results", None),
                getattr(state, "dk_search_results_flattened", None),
            )

            if dk_rich and hasattr(self, "dk_search_results"):
                self.dk_search_raw_data = dk_rich
                self._display_dk_search_results(dk_rich)

            # DK classification tab — state.dk_classifications may be List[Dict]
            if state.dk_classifications and hasattr(self, "dk_classification_results"):
                codes = self._dk_class_codes(state.dk_classifications)
                html_display = self._format_dk_classifications_with_titles(
                    codes,
                    dk_rich,
                )
                self.dk_classification_results.setHtml(html_display)

            # DK compact stats
            if state.dk_statistics and hasattr(self, "dk_compact_stats"):
                stats = state.dk_statistics
                total = stats.get("total_classifications", 0)
                dedup = stats.get("deduplication_stats", {})
                orig = dedup.get("original_count", 0)
                rate = dedup.get("deduplication_rate", "0%")
                self.dk_compact_stats.setText(
                    f"📊 <b>Klassifikations-Statistik:</b> {orig} Katalogtreffer → "
                    f"<b>{total}</b> unikale Klassifikationen (Deduplizierungsrate: {rate})"
                )
        except Exception as e:
            self.logger.warning(f"_sync_classical_tabs_from_state failed: {e}")

    def on_abort_current_step_requested(self):
        """Abort only the current LLM generation; pipeline continues - Claude Generated"""
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.logger.info("User requested step-only abort (pipeline continues)")
            self.pipeline_worker.abort_current_step()

    def on_stop_pipeline_requested(self):
        """Handle stop button click - Claude Generated"""
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.logger.info("User requested pipeline stop")
            self.stop_pipeline_button.setEnabled(False)
            self.stop_pipeline_button.setText("⏹ Stopping...")
            self.pipeline_status_label.setText("Beende Pipeline...")
            self.pipeline_worker.request_stop()

    @pyqtSlot()
    def on_pipeline_aborted(self):
        """Handle pipeline abort signal - Claude Generated"""
        self.logger.info("Pipeline aborted by user")

        # Stop any running timer
        self.duration_update_timer.stop()
        self.current_running_step = None

        # Reset button states
        self.auto_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setText("⏹️ Stop")
        self.stop_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setVisible(False)

        # Update status
        self.pipeline_status_label.setText("Pipeline abgebrochen")
        self.pipeline_status_label.setStyleSheet(
            "color: #FF9800; font-weight: bold; padding: 5px; "
            "background-color: #FFF3E0; border: 1px solid #FFB74D; border-radius: 3px;"
        )

        # End any active streaming
        if hasattr(self, "stream_widget") and self.stream_widget.is_streaming:
            self.stream_widget.end_llm_streaming()

        # Note: Removed QMessageBox - status label provides sufficient feedback - Claude Generated

    @pyqtSlot(str, str)
    def on_llm_stream_token(self, token: str, step_id: str):
        """Handle streaming LLM token - Claude Generated"""
        self.logger.debug(f"Received streaming token for {step_id}: '{token[:20]}...'")
        if hasattr(self, "stream_widget"):
            # Start streaming line if not already started
            if not self.stream_widget.is_streaming:
                self.logger.debug(f"Starting streaming for step {step_id}")
                self.stream_widget.start_llm_streaming(step_id)

            # Add the token to the streaming display
            self.stream_widget.add_streaming_token(token, step_id)

            # End streaming if we get a final token (this would need refinement based on actual LLM response patterns)
            # For now, we'll leave the line open and let the step completion handle ending

    def on_repetition_detected(self, result, suggestions: list, grace_period: bool, resolved: bool, grace_seconds: float):
        """Handle repetition detection from LLM - Claude Generated (2026-02-17)

        Args:
            result: RepetitionResult object (None if resolved)
            suggestions: List of parameter variation suggestions
            grace_period: True if grace period active
            resolved: True if repetition resolved during grace period
            grace_seconds: Grace period duration in seconds
        """
        if resolved:
            # Repetition resolved - hide warning
            self.stream_widget.hide_repetition_warning(resolved=True)
        elif result:
            # Repetition detected - show warning
            detection_type = result.detection_type
            details = result.details
            self.stream_widget.show_repetition_warning(
                detection_type=detection_type,
                details=details,
                suggestions=suggestions,
                grace_period=grace_period,
                grace_seconds=grace_seconds
            )

    def _load_catalog_config(self) -> tuple[str, str, str]:
        """Load catalog configuration from ConfigManager - Claude Generated"""
        # Initialize default values
        catalog_token = ""
        catalog_search_url = ""
        catalog_details_url = ""

        try:
            from ..utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            catalog_config = config_manager.get_catalog_config()

            # Access dataclass attributes directly (not dictionary .get())
            catalog_token = catalog_config.catalog_token
            catalog_search_url = catalog_config.catalog_search_url
            catalog_details_url = catalog_config.catalog_details_url

            if catalog_token:
                self.logger.debug(f"Loaded catalog token from config (length: {len(catalog_token)})")
            else:
                self.logger.warning("No catalog token found in config")

        except Exception as e:
            self.logger.error(f"Error loading catalog config: {e}")

        return catalog_token, catalog_search_url, catalog_details_url
    
    def _update_pipeline_config_with_catalog_settings(self):
        """Update pipeline config with loaded catalog settings - Claude Generated"""
        config = self.pipeline_manager.config
        
        # Update DK search step configuration
        if "dk_search" in config.step_configs:
            # Store catalog settings in step config custom parameters
            dk_search_config = config.step_configs["dk_search"]
            dk_search_config.custom_params.update({
                "catalog_token": self.catalog_token,
                "catalog_search_url": self.catalog_search_url,
                "catalog_details_url": self.catalog_details_url,
            })

        # Also update DK classification step if it exists
        if "dk_classification" in config.step_configs:
            dk_classification_config = config.step_configs["dk_classification"]
            dk_classification_config.custom_params.update({
                "catalog_token": self.catalog_token,
                "catalog_search_url": self.catalog_search_url,
                "catalog_details_url": self.catalog_details_url,
            })
        
        self.logger.debug(f"Updated pipeline config with catalog settings (token present: {bool(self.catalog_token)})")

    def _emit_step_results_to_tabs(self, step: PipelineStep) -> None:
        """
        Emit pipeline step results to appropriate tab viewer methods - Claude Generated
        
        Args:
            step: Completed pipeline step with results
        """
        if not step.output_data:
            return
            
        try:
            # Emit search results to SearchTab
            if step.step_id == "search" and "search_results" in step.output_data:
                search_results = step.output_data["search_results"]
                self.logger.debug(f"Emitting search results to SearchTab: {len(search_results)} terms")
                self.search_results_ready.emit(search_results)
            
            # Emit keyword analysis results to AbstractTab (and DkAnalysisTab)
            elif step.step_id in ["initialisation", "keywords", "dk_classification"]:
                if "analysis_result" in step.output_data:
                    analysis_result = step.output_data["analysis_result"]
                    self.logger.debug(f"Emitting {step.step_id} analysis results to AbstractTab")
                    self.analysis_results_ready.emit(analysis_result)
                elif "llm_analysis" in step.output_data:
                    llm_analysis = step.output_data["llm_analysis"]
                    self.logger.debug(f"Emitting {step.step_id} LLM analysis results to Tabs")
                    # We reuse the same signal, as AbstractTab can handle LlmKeywordAnalysis too
                    # (Need to ensure AbstractTab's slot can handle both or we wrap it)
                    self.analysis_results_ready.emit(llm_analysis)
                
        except Exception as e:
            self.logger.error(f"Error emitting step results to tabs: {e}")

    def show_loaded_state_indicator(self, state):
        """
        Display visual indicators for loaded analysis state - Claude Generated
        Shows which pipeline steps have data from the loaded JSON
        """
        try:
            # Add visual indicator in pipeline status
            loaded_steps = []

            if state.original_abstract:
                loaded_steps.append("Input")
            if state.initial_keywords:
                loaded_steps.append("Initialisierung")
            if state.search_results:
                loaded_steps.append("Suche")
            if state.final_llm_analysis:
                loaded_steps.append("Schlagworte")
            if state.classifications:
                loaded_steps.append("Klassifikation")

            if loaded_steps:
                loaded_info = " → ".join(loaded_steps)
                self.pipeline_status_label.setText(f"📁 Geladener Zustand: {loaded_info}")
                self.pipeline_status_label.setStyleSheet(
                    "color: #2E7D32; font-weight: bold; padding: 5px; "
                    "background-color: #E8F5E8; border: 1px solid #4CAF50; border-radius: 3px;"
                )

                # Populate results displays with loaded data
                if state.initial_keywords and hasattr(self, 'initialisation_result'):
                    # Type-safe join - Claude Generated (Fix for string parsing bug)
                    keywords_text = (", ".join(state.initial_keywords)
                                     if isinstance(state.initial_keywords, list)
                                     else str(state.initial_keywords))
                    self.initialisation_result.setPlainText(f"📁 Geladene Keywords:\n{keywords_text}")

                if state.search_results and hasattr(self, 'search_results_table'):
                    selected = (
                        state.final_llm_analysis.extracted_gnd_keywords
                        if state.final_llm_analysis
                        else None
                    )
                    self._populate_gnd_hits(state.search_results, selected=selected)

                if state.final_llm_analysis and hasattr(self, 'keywords_result'):
                    # Type-safe join - Claude Generated (Fix for string parsing bug)
                    final_kw = state.final_llm_analysis.extracted_gnd_keywords
                    final_keywords = (", ".join(final_kw)
                                      if isinstance(final_kw, list)
                                      else str(final_kw))
                    self.keywords_result.setPlainText(f"📁 Finale Schlagwörter:\n{final_keywords}")

                # DK Classification Results Display - Claude Generated (Enhanced with titles)
                if state.classifications and hasattr(self, 'dk_classification_results'):
                    html_display = self._format_dk_classifications_with_titles(
                        state.classifications,
                        state.dk_search_results_flattened  # flattened format has {dk, titles} at top level
                    )
                    self.dk_classification_results.setHtml(
                        f"<div style='background: #E8F5E8; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                        f"<strong>📁 Geladene Klassifikationen (DK/RVK)</strong>"
                        f"</div>{html_display}"
                    )

                # DK Search Results Display - Claude Generated (Enhanced for filtering)
                if state.dk_search_results and hasattr(self, 'dk_search_results'):
                    # Store raw data for filtering
                    self.dk_search_raw_data = state.dk_search_results

                    # Display results using display method
                    self._display_dk_search_results(state.dk_search_results)

                    # Add loaded indicator prefix
                    current_text = self.dk_search_results.toPlainText()
                    self.dk_search_results.setPlainText(
                        f"📁 Geladene DK-Suchergebnisse:\n\n{current_text}"
                    )

            self.logger.info(f"Pipeline tab updated with loaded state indicators: {loaded_steps}")

        except Exception as e:
            self.logger.error(f"Error showing loaded state indicator: {e}")
