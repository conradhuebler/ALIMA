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
from PyQt6.QtGui import QFont, QPalette, QPixmap
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import json
from pathlib import Path

from ..core.pipeline_manager import PipelineManager, PipelineStep, PipelineConfig
from .pipeline_config_dialog import PipelineConfigDialog
from ..core.alima_manager import AlimaManager
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..llm.llm_service import LlmService
from .crossref_tab import CrossrefTab
from .image_analysis_tab import ImageAnalysisTab
from .unified_input_widget import UnifiedInputWidget
from .pipeline_stream_widget import PipelineStreamWidget
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
        self.status_label = QLabel()
        header_layout.addWidget(self.status_label)

        # Step name
        name_label = QLabel(self.step.name)
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_label.setFont(name_font)
        header_layout.addWidget(name_label)

        header_layout.addStretch()

        # Enhanced Provider/Model info with task preference indicators - Claude Generated
        self.provider_model_label = QLabel()
        self.provider_model_label.setStyleSheet("color: #666; font-size: 10px;")
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
                self.provider_model_label.setStyleSheet("color: #ff9800; font-size: 10px; font-style: italic;")
            else:
                # Non-LLM steps (like search) don't need provider info
                self.provider_model_label.setText("No LLM required")
                self.provider_model_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
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
        self.provider_model_label.setStyleSheet(f"color: {style_color}; font-size: 10px;")

        # Set tooltip with full details
        tooltip_parts = [f"Provider: {self.step.provider}", f"Model: {self.step.model}"]
        if hasattr(self.step, 'selection_reason') and self.step.selection_reason:
            tooltip_parts.append(f"Source: {self.step.selection_reason}")
        self.provider_model_label.setToolTip("\n".join(tooltip_parts))

    def update_status_display(self):
        """Update visual status indicator - Claude Generated"""
        if self.step.status == "pending":
            self.status_label.setText("▷")
            self.status_label.setStyleSheet(
                "color: #999; font-size: 16px; font-weight: bold;"
            )
            self.setStyleSheet(
                "QFrame { border-color: #ddd; background-color: #fafafa; }"
            )

        elif self.step.status == "running":
            self.status_label.setText("▶")
            self.status_label.setStyleSheet(
                "color: #2196f3; font-size: 16px; font-weight: bold;"
            )
            self.setStyleSheet(
                "QFrame { border-color: #2196f3; background-color: #e3f2fd; }"
            )

        elif self.step.status == "completed":
            self.status_label.setText("✓")
            self.status_label.setStyleSheet(
                "color: #4caf50; font-size: 16px; font-weight: bold;"
            )
            self.setStyleSheet(
                "QFrame { border-color: #4caf50; background-color: #e8f5e8; }"
            )

        elif self.step.status == "error":
            self.status_label.setText("✗")
            self.status_label.setStyleSheet(
                "color: #d32f2f; font-size: 16px; font-weight: bold;"
            )
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


class PipelineTab(QWidget):
    """Main pipeline tab with vertical workflow - Claude Generated"""

    # Signals
    pipeline_started = pyqtSignal(str)  # pipeline_id
    pipeline_completed = pyqtSignal()
    step_selected = pyqtSignal(str)  # step_id

    # Signals for pipeline result emission to other tabs - Claude Generated
    search_results_ready = pyqtSignal(dict)  # For SearchTab.display_search_results()
    metadata_ready = pyqtSignal(dict)       # For CrossrefTab.display_metadata()
    analysis_results_ready = pyqtSignal(object)  # For AbstractTab analysis results
    pipeline_results_ready = pyqtSignal(object)  # Complete analysis_state for distribution - Claude Generated

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

        # Working title label - Claude Generated
        self.title_label: Optional[QLabel] = None

        # Input state
        self.current_input_text: str = ""
        self.current_source_info: str = ""

        self.setup_ui()

        # Synchronize iterative search checkbox with config - Claude Generated
        self._sync_iterative_search_checkbox()

    def _sync_iterative_search_checkbox(self):
        """Sync checkbox and spinbox state with pipeline config - Claude Generated"""
        try:
            if self.pipeline_manager and self.pipeline_manager.config:
                keywords_config = self.pipeline_manager.config.get_step_config("keywords")
                if keywords_config:
                    # Sync checkbox
                    if hasattr(self, 'iterative_search_checkbox'):
                        enabled = getattr(keywords_config, 'enable_iterative_refinement', False)
                        self.iterative_search_checkbox.blockSignals(True)
                        self.iterative_search_checkbox.setChecked(enabled)
                        self.iterative_search_checkbox.blockSignals(False)
                        self.logger.debug(f"Synced iterative search checkbox: {enabled}")

                    # Sync max iterations spinbox
                    if hasattr(self, 'max_iterations_spin'):
                        max_iter = getattr(keywords_config, 'max_refinement_iterations', 2)
                        self.max_iterations_spin.blockSignals(True)
                        self.max_iterations_spin.setValue(max_iter)
                        self.max_iterations_spin.blockSignals(False)
                        self.logger.debug(f"Synced max iterations: {max_iter}")
        except Exception as e:
            self.logger.error(f"Error syncing iterative search controls: {e}")

    def _populate_global_override_combo(self):
        """Populate the global override combo with available provider/model pairs - Claude Generated"""
        try:
            self.global_override_combo.clear()
            self.global_override_combo.addItem("-- Standard --", None)

            from ..utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            unified_config = config_manager.get_unified_config()
            enabled_providers = unified_config.get_enabled_providers()

            for provider in enabled_providers:
                provider_name = provider.name
                models = getattr(provider, 'available_models', []) or []
                if not models and getattr(provider, 'preferred_model', None):
                    models = [provider.preferred_model]

                for model in models:
                    display = f"{provider_name} | {model}"
                    data = f"{provider_name}|{model}"
                    self.global_override_combo.addItem(display, data)

            self.logger.debug(f"Override combo populated: {self.global_override_combo.count() - 1} models")
        except Exception as e:
            self.logger.error(f"Error populating override combo: {e}")

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

        # Collapsible advanced panel (model override + iterative search) - Claude Generated
        self.create_advanced_panel(main_layout)

        # Working title display and override field - Claude Generated
        title_widget = QWidget()
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(5, 5, 5, 5)
        title_layout.setSpacing(3)

        # Title label (always visible, empty until title generated) - Claude Generated
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 11px; color: #555555; padding: 2px;")
        title_layout.addWidget(self.title_label)

        # Title override field - Claude Generated
        from PyQt6.QtWidgets import QLineEdit
        self.title_override_field = QLineEdit()
        self.title_override_field.setPlaceholderText("Optional: Arbeitstitel überschreiben")
        self.title_override_field.setStyleSheet("font-size: 10px; padding: 6px; border: 1px solid #ddd; border-radius: 3px;")
        self.title_override_field.returnPressed.connect(self.on_title_override_changed)
        self.title_override_field.editingFinished.connect(self.on_title_override_changed)
        title_layout.addWidget(self.title_override_field)

        # Reserve enough space for both widgets to prevent window resize - Claude Generated
        # Use fixed size policy to prevent dynamic height changes that would resize the main window
        title_widget.setFixedHeight(70)  # Tight fit: label (11px font + 2px padding) + field (10px font + 6px padding*2) + margins (5+5) + spacing (3)
        title_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # Widget visible from start (label hidden until title generated) - Claude Generated
        self.title_widget = title_widget
        main_layout.addWidget(title_widget)

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

        # Right side: Live streaming widget
        self.stream_widget = PipelineStreamWidget()

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
                            provider = first_pref.get('provider_name', 'openai_compatible')
                            model = first_pref.get('model_name', 'gpt-4')
                        else:
                            provider = getattr(first_pref, 'provider_name', 'openai_compatible')
                            model = getattr(first_pref, 'model_name', 'gpt-4')
                        self.logger.debug(f"Loaded task preference for {task_name}: {provider}/{model}")
                        return provider, model
        except Exception as e:
            self.logger.warning(f"Could not load task preference for {task_name}: {e}")
        
        # Default fallback
        return "openai_compatible", "gpt-4"

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
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #ccc; }
            """
        )
        self.auto_pipeline_button.clicked.connect(self.start_auto_pipeline)
        tb_layout.addWidget(self.auto_pipeline_button)

        # Stop button (initially hidden, shown only when pipeline is running) - Claude Generated
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
                font-size: 12px;
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

        # Secondary actions (compact) - Claude Generated
        self.load_json_button = QPushButton("📁 JSON laden")
        self.load_json_button.setToolTip("Pipeline-State aus JSON-Datei laden")
        self.load_json_button.clicked.connect(self.load_json_state)
        tb_layout.addWidget(self.load_json_button)

        config_btn = QPushButton("⚙️ Config")
        config_btn.setToolTip("Pipeline-Konfiguration öffnen")
        config_btn.clicked.connect(self.show_pipeline_config)
        tb_layout.addWidget(config_btn)

        reset_btn = QPushButton("🔄 Reset")
        reset_btn.setToolTip("Pipeline zurücksetzen")
        reset_btn.clicked.connect(self.reset_pipeline)
        tb_layout.addWidget(reset_btn)

        # Advanced toggle button - Claude Generated
        self.advanced_toggle_button = QPushButton("▼ Erweitert")
        self.advanced_toggle_button.setToolTip("Modell-Override und Iterative Suche einblenden")
        self.advanced_toggle_button.setCheckable(True)
        self.advanced_toggle_button.setStyleSheet(
            """
            QPushButton { background: transparent; border: 1px solid #ccc;
                          padding: 3px 8px; border-radius: 3px; font-size: 11px; color: #555; }
            QPushButton:hover { background: #e9ecef; }
            QPushButton:checked { background: #e3f2fd; border-color: #90caf9; color: #1976d2; }
            """
        )
        self.advanced_toggle_button.clicked.connect(self.toggle_advanced_panel)
        tb_layout.addWidget(self.advanced_toggle_button)

        tb_layout.addStretch()

        # Mode indicator - Claude Generated
        self.mode_indicator_label = QLabel()
        self._update_mode_indicator()
        tb_layout.addWidget(self.mode_indicator_label)

        # Pipeline status label - Claude Generated
        self.pipeline_status_label = QLabel("Bereit")
        self.pipeline_status_label.setStyleSheet("font-size: 11px; color: #666; padding-left: 8px;")
        tb_layout.addWidget(self.pipeline_status_label)

        main_layout.addWidget(self.toolbar_frame)

    def create_advanced_panel(self, main_layout):
        """Create collapsible advanced options panel (hidden by default) - Claude Generated"""
        self.advanced_frame = QFrame()
        self.advanced_frame.setFixedHeight(38)
        self.advanced_frame.setStyleSheet(
            "QFrame { background: #f0f4f8; border-bottom: 1px solid #dee2e6; }"
        )
        self.advanced_frame.setVisible(False)

        adv_layout = QHBoxLayout(self.advanced_frame)
        adv_layout.setContentsMargins(8, 4, 8, 4)
        adv_layout.setSpacing(10)

        # Global Model Override ComboBox - Claude Generated
        override_label = QLabel("🔬 Modell-Override:")
        override_label.setStyleSheet("font-size: 11px; color: #555;")
        override_label.setToolTip("Erzwingt Provider/Modell für alle LLM-Steps")
        adv_layout.addWidget(override_label)

        self.global_override_combo = QComboBox()
        self.global_override_combo.setMinimumWidth(180)
        self.global_override_combo.setMaximumWidth(300)
        self.global_override_combo.setToolTip(
            "Globaler Override: Erzwingt Provider/Modell für alle LLM-Steps\n"
            "(Initialisation, Keywords, DK-Klassifikation)\n\n"
            "\"-- Standard --\" = Normale Provider-Auswahl"
        )
        self.global_override_combo.setStyleSheet(
            "QComboBox { padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px; font-size: 11px; }"
        )
        self._populate_global_override_combo()
        adv_layout.addWidget(self.global_override_combo)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFixedHeight(20)
        sep2.setStyleSheet("color: #ccc;")
        adv_layout.addWidget(sep2)

        # Iterative Search Controls - Claude Generated
        self.iterative_search_checkbox = QCheckBox("🔄 Iterative GND-Suche")
        self.iterative_search_checkbox.setToolTip(
            "Wenn aktiviert: Automatische Suche nach fehlenden Konzepten\n"
            "mit GND-Pool-Erweiterung über mehrere Iterationen.\n\n"
            "⚠️ Erhöht Token-Nutzung um ca. 2-3x\n"
            "⏱️ Verlängert Analysezeit um 30-70 Sekunden"
        )
        self.iterative_search_checkbox.setStyleSheet(
            "QCheckBox { font-weight: bold; color: #0066cc; font-size: 11px; }"
            "QCheckBox::indicator { width: 16px; height: 16px; }"
        )
        adv_layout.addWidget(self.iterative_search_checkbox)

        iterations_label = QLabel("Max:")
        iterations_label.setStyleSheet("color: #666; font-size: 10px;")
        adv_layout.addWidget(iterations_label)

        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(1, 5)
        self.max_iterations_spin.setValue(2)
        self.max_iterations_spin.setFixedWidth(48)
        self.max_iterations_spin.setEnabled(False)
        self.max_iterations_spin.setToolTip("Max. Iterationen (1-5)")
        adv_layout.addWidget(self.max_iterations_spin)

        # Connect signals
        self.iterative_search_checkbox.stateChanged.connect(self.on_iterative_search_toggled)
        self.iterative_search_checkbox.toggled.connect(self.max_iterations_spin.setEnabled)
        self.max_iterations_spin.valueChanged.connect(self.on_max_iterations_changed)

        adv_layout.addStretch()
        main_layout.addWidget(self.advanced_frame)

    def toggle_advanced_panel(self):
        """Toggle visibility of the advanced options panel - Claude Generated"""
        visible = not self.advanced_frame.isVisible()
        self.advanced_frame.setVisible(visible)
        self.advanced_toggle_button.setText("▲ Erweitert" if visible else "▼ Erweitert")

    def _update_mode_indicator(self):
        """Update mode indicator to show current pipeline mode - Claude Generated"""
        try:
            # Get the overall pipeline mode by checking if most steps use Smart Mode
            config = self.pipeline_manager.config
            if not hasattr(config, 'step_configs') or not config.step_configs:
                self.mode_indicator_label.setText("🤖 Smart Mode")
                self.mode_indicator_label.setStyleSheet("color: #2e7d32; font-size: 11px; font-weight: bold;")
                self.mode_indicator_label.setToolTip("Pipeline Mode: Smart (automatic provider/model selection)")
                return

            # Count configuration types across LLM steps (baseline vs override)
            llm_steps = ["initialisation", "keywords", "dk_classification"]
            config_counts = {"baseline": 0, "override": 0}

            for step_id in llm_steps:
                if step_id in config.step_configs:
                    step_config = config.step_configs[step_id]
                    # In baseline + override architecture: check if provider/model are explicitly set
                    if step_config.provider and step_config.model:
                        config_counts["override"] += 1
                    else:
                        config_counts["baseline"] += 1
                else:
                    config_counts["baseline"] += 1  # Default to smart baseline

            # Determine dominant configuration type
            dominant_config = max(config_counts, key=config_counts.get)

            # Set configuration indicator based on dominant type
            if dominant_config == "baseline" or config_counts["override"] == 0:
                self.mode_indicator_label.setText("🤖 Smart Baseline")
                self.mode_indicator_label.setStyleSheet("color: #2e7d32; font-size: 11px; font-weight: bold;")
                self.mode_indicator_label.setToolTip("Configuration: Smart Baseline (automatic provider/model selection)")
            elif config_counts["baseline"] == 0:
                self.mode_indicator_label.setText("⚙️ Full Override")
                self.mode_indicator_label.setStyleSheet("color: #d32f2f; font-size: 11px; font-weight: bold;")
                self.mode_indicator_label.setToolTip("Configuration: Full Override (all steps manually configured)")
            else:  # mixed
                self.mode_indicator_label.setText("🔧 Mixed Config")
                self.mode_indicator_label.setStyleSheet("color: #1976d2; font-size: 11px; font-weight: bold;")
                self.mode_indicator_label.setToolTip(f"Configuration: Mixed (baseline: {config_counts['baseline']}, override: {config_counts['override']})")

        except Exception as e:
            self.logger.error(f"Error updating mode indicator: {e}")
            # Fallback to Smart Mode
            self.mode_indicator_label.setText("🤖 Smart Mode")
            self.mode_indicator_label.setStyleSheet("color: #2e7d32; font-size: 11px; font-weight: bold;")
            self.mode_indicator_label.setToolTip("Pipeline Mode: Smart (automatic provider/model selection)")

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
            "font-weight: bold; font-size: 11px; color: #555; padding: 2px;"
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
        """Create search step widget - Claude Generated"""
        # TODO - erweitern um Tabelle mit Ergebnissen (Freies Schlagwort für Suche, GND-Schlagworte, Häufigkeiten) und Filtermöglichkeiten; vergleichbar mit standalone-tab für GND-Suche
        widget, self.search_result = self._create_text_result_widget(
            label_text="GND-Suchergebnisse:",
            placeholder="Suchergebnisse werden hier angezeigt..."
        )
        return widget

    def create_keywords_step_widget(self) -> QWidget:
        """Create keywords step widget (Verbale Erschließung) - Claude Generated"""
        # TODO - erweitern um Tabelle mit Ergebnissen ggf anpassen an Verschlagwortungsrelevante Dinge, welche Schlagworte ignoriert wurde ...
        widget, self.keywords_result = self._create_text_result_widget(
            label_text="Finale GND-Schlagworte:",
            placeholder="Finale Schlagworte werden hier angezeigt..."
        )
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
        config_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #555;")
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
        self.dk_frequency_threshold.setToolTip("Nur DK-Codes mit >= N Vorkommen")
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
        filter_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #555;")
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
        self.dk_filter_mode.addItems(["Alle", "Titel", "DK-Codes", "Keywords"])
        self.dk_filter_mode.currentTextChanged.connect(self._filter_dk_search_results)
        filter_grid.addWidget(self.dk_filter_mode, 0, 5)

        self.dk_filter_count_label = QLabel("")
        self.dk_filter_count_label.setStyleSheet("color: #666; font-size: 10px;")
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
        results_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #555;")
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
        input_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #555;")
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
        results_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #555;")
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
        self.dk_compact_stats.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
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
            elif filter_mode == "DK-Codes":
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
        """Display DK search results with formatting - Claude Generated"""
        if not results:
            self.dk_search_results.setPlainText(
                "Keine Ergebnisse gefunden" if hasattr(self, 'dk_search_filter_input')
                and self.dk_search_filter_input.text()
                else "Keine DK/RVK-Klassifikationen gefunden"
            )
            return

        result_lines = []
        for result in results:
            dk_code = result.get("dk", "")
            count = result.get("count", 0)
            titles = result.get("titles", [])
            keywords = result.get("keywords", [])
            classification_type = result.get("classification_type", "DK")

            if not titles or count == 0:
                continue

            sample_titles = titles[:3]
            titles_text = " | ".join(sample_titles)
            if len(titles) > 3:
                titles_text += f" | ... (und {len(titles) - 3} weitere)"

            result_line = (
                f"{classification_type}: {dk_code} (Häufigkeit: {count})\n"
                f"Beispieltitel: {titles_text}\n"
                f"Keywords: {', '.join(keywords)}\n"
            )
            result_lines.append(result_line)

        self.dk_search_results.setPlainText("\n".join(result_lines))

    def _format_dk_classifications_with_titles(
        self,
        dk_classifications: List[str],
        dk_search_results: List[Dict[str, Any]],
        max_titles_per_code: int = 5
    ) -> str:
        """Format DK classifications with catalog titles using HTML - Claude Generated"""
        if not dk_classifications:
            return "Keine DK/RVK-Klassifikationen generiert"

        html_parts = []
        html_parts.append("<html><body style='font-family: Arial, sans-serif;'>")

        for idx, dk_code in enumerate(dk_classifications, 1):
            titles, total_count = self._get_titles_for_dk_code(dk_code, dk_search_results)

            # Color-coding based on frequency (confidence)
            if total_count > 50:
                color, bg_color = "#2d5016", "#d4edda"  # Dark green
            elif total_count > 20:
                color, bg_color = "#0c5460", "#d1ecf1"  # Teal
            else:
                color, bg_color = "#664d03", "#fff3cd"  # Brown/Orange

            # Header with confidence indicator
            html_parts.append(
                f"<div style='background-color: {bg_color}; padding: 12px; margin-bottom: 8px; "
                f"border-left: 4px solid {color}; border-radius: 4px;'>"
                f"<h2 style='color: {color}; margin: 0; font-size: 14pt;'>#{idx} {dk_code}</h2>"
            )

            if total_count > 0:
                confidence_bar = "🟩" * min(5, (total_count // 10) + 1)
                html_parts.append(
                    f"<p style='color: {color}; font-weight: bold; margin: 5px 0 2px 0;'>"
                    f"{confidence_bar} {total_count} Katalog-Treffer</p>"
                    f"<p style='color: {color}; font-size: 9pt; opacity: 0.8; margin: 0 0 10px 0;'>"
                    f"📚 Diese Klassifikation wurde in {total_count} Titel{'n' if total_count != 1 else ''} gefunden.</p>"
                )
            html_parts.append("</div>")

            # Title list
            if titles:
                html_parts.append("<ol style='font-size: 9pt; padding-left: 30px;'>")
                for title in titles[:max_titles_per_code]:
                    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    html_parts.append(f"<li>{safe_title}</li>")
                html_parts.append("</ol>")

                if total_count > max_titles_per_code:
                    html_parts.append(f"<p style='color: #888; font-style: italic; padding-left: 20px;'>... und {total_count - max_titles_per_code} weitere Titel</p>")

        html_parts.append("</body></html>")
        return "".join(html_parts)

    def _get_titles_for_dk_code(
        self,
        dk_code: str,
        dk_search_results: List[Dict[str, Any]]
    ) -> tuple[list, int]:
        """Extract titles for a specific DK code - Claude Generated"""
        if not dk_search_results:
            return ([], 0)

        normalized_code = dk_code.replace("DK ", "").strip()

        for result in dk_search_results:
            result_code = str(result.get("dk", "")).strip()
            if result_code == normalized_code:
                titles = result.get("titles", [])
                return (titles[:50], len(titles))  # Max 50 for display

        return ([], 0)

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

        # Apply global model override from combo box - Claude Generated
        self._apply_global_override_from_gui()

        # Update DK configuration from GUI widgets - Claude Generated
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
        self.pipeline_worker.pipeline_completed.connect(self.on_pipeline_completed)
        self.pipeline_worker.stream_token.connect(self.on_llm_stream_token)
        self.pipeline_worker.aborted.connect(self.on_pipeline_aborted)  # Claude Generated
        self.pipeline_worker.repetition_detected.connect(self.on_repetition_detected)  # Claude Generated (2026-02-17)

        # Start the worker
        self.pipeline_worker.start()

        # Emit pipeline started signal
        self.pipeline_started.emit("pipeline_thread")

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_pipeline_started("pipeline_thread")

    def _apply_global_override_from_gui(self):
        """Apply global provider/model override from combo box to pipeline config - Claude Generated"""
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
            config.apply_global_override()
            self.logger.info(f"🔬 GUI override applied: {provider}/{model}")
        else:
            # "-- Standard --" selected, clear any previous override
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

        # Update mode indicator to reflect new configuration
        self._update_mode_indicator()

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

        # Update mode indicator to reflect configuration changes
        self._update_mode_indicator()

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
        if hasattr(self, "search_result"):
            self.search_result.clear()
        if hasattr(self, "keywords_result"):
            self.keywords_result.clear()
        # DK-related widgets - Claude Generated (Fixed widget names)
        if hasattr(self, "dk_classification_results"):
            self.dk_classification_results.clear()
        if hasattr(self, "dk_search_results"):
            self.dk_search_results.clear()
        if hasattr(self, "dk_input_summary"):
            self.dk_input_summary.clear()

        # Reset title display - Claude Generated
        if hasattr(self, "title_label"):
            self.title_label.clear()
        if hasattr(self, "title_override_field"):
            self.title_override_field.clear()
            self.title_override_field.setPlaceholderText("Optional: Arbeitstitel überschreiben")

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

    def on_iterative_search_toggled(self, state):
        """Handle iterative search checkbox toggle - Claude Generated"""
        enabled = state == Qt.CheckState.Checked.value

        self.logger.debug(f"Iterative search toggled: {enabled}")

        # Update pipeline configuration
        if self.pipeline_manager and self.pipeline_manager.config:
            keywords_config = self.pipeline_manager.config.get_step_config("keywords")
            if keywords_config:
                keywords_config.enable_iterative_refinement = enabled

                # Show visual feedback
                if enabled:
                    max_iter = self.max_iterations_spin.value() if hasattr(self, 'max_iterations_spin') else 2
                    self.logger.info(f"✅ Iterative GND-Suche aktiviert (max. {max_iter} Iterationen)")
                    # Show info in status bar if available
                    if self.main_window and hasattr(self.main_window, "global_status_bar"):
                        self.main_window.global_status_bar.show_temporary_message(
                            f"🔄 Iterative GND-Suche aktiviert (max. {max_iter})", 3000
                        )
                else:
                    self.logger.info("❌ Iterative GND-Suche deaktiviert")
                    if self.main_window and hasattr(self.main_window, "global_status_bar"):
                        self.main_window.global_status_bar.show_temporary_message(
                            "Iterative GND-Suche deaktiviert", 3000
                        )

    def on_max_iterations_changed(self, value):
        """Handle max iterations spinbox change - Claude Generated"""
        self.logger.debug(f"Max iterations changed: {value}")

        # Update pipeline configuration
        if self.pipeline_manager and self.pipeline_manager.config:
            keywords_config = self.pipeline_manager.config.get_step_config("keywords")
            if keywords_config:
                keywords_config.max_refinement_iterations = value

                # Show visual feedback if enabled
                if getattr(keywords_config, 'enable_iterative_refinement', False):
                    if self.main_window and hasattr(self.main_window, "global_status_bar"):
                        self.main_window.global_status_bar.show_temporary_message(
                            f"Max. Iterationen: {value}", 2000
                        )

        # Update webcam frame visibility in unified input widget
        if hasattr(self, 'unified_input') and self.unified_input:
            self.unified_input._update_webcam_frame_visibility()
            self.logger.debug("Webcam frame visibility updated")

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

                # Update title label text
                self.title_label.setText(f"📋 {working_title}")

                # Always update override field with current title - Claude Generated
                self.title_override_field.clear()
                self.title_override_field.setPlaceholderText(f"Current: {working_title}")

                # Set working title in stream widget for log filename - Claude Generated
                if hasattr(self, 'stream_widget') and self.stream_widget:
                    self.stream_widget.set_working_title(working_title)

                self.logger.info(f"Displaying working title: {working_title}")
        elif step.step_id == "search" and step.output_data:
            gnd_treffer = step.output_data.get("gnd_treffer", [])
            # if hasattr(self, 'search_result'):
            if gnd_treffer:
                self.search_result.setPlainText("\n".join(gnd_treffer))
            else:
                self.search_result.setPlainText("Keine GND-Treffer gefunden")

        elif step.step_id == "keywords" and step.output_data:
            final_keywords = step.output_data.get("final_keywords", "")
            self.logger.debug(f"Keywords step output_data: {step.output_data}")
            self.logger.debug(f"Final keywords: '{final_keywords}'")
            if hasattr(self, "keywords_result"):
                # Handle both string and list formats
                if isinstance(final_keywords, list):
                    final_keywords_text = "\n".join(final_keywords)
                else:
                    final_keywords_text = str(final_keywords)
                self.keywords_result.setPlainText(final_keywords_text)
                self.logger.debug(
                    f"Set keywords_result text to: '{final_keywords_text}'"
                )

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
                            f"📊 <b>DK-Statistik:</b> {orig} Katalogtreffer → <b>{total}</b> unikale Klassifikationen "
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

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_pipeline_completed(analysis_state)

        # Emit complete analysis_state for distribution to specialized tabs - Claude Generated
        if analysis_state:
            self.pipeline_results_ready.emit(analysis_state)

        # Optional: Auto-save after completion - Claude Generated
        if hasattr(analysis_state, 'working_title') and analysis_state.working_title:
            from ..utils.pipeline_utils import PipelineJsonManager
            from ..utils.pipeline_defaults import get_autosave_dir

            auto_save_dir = get_autosave_dir(getattr(self, 'config_manager', None))
            auto_save_dir.mkdir(parents=True, exist_ok=True)

            auto_save_file = auto_save_dir / f"{analysis_state.working_title}.json"
            try:
                PipelineJsonManager.save_analysis_state(analysis_state, str(auto_save_file))
                self.logger.info(f"✅ Auto-saved pipeline result to: {auto_save_file}")
            except Exception as e:
                self.logger.warning(f"Auto-save failed: {e}")

        QMessageBox.information(
            self,
            "Pipeline abgeschlossen",
            "Die komplette Analyse-Pipeline wurde erfolgreich abgeschlossen!",
        )

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

    def on_title_override_changed(self):
        """Handle title override field changes - Claude Generated"""
        override_text = self.title_override_field.text().strip()

        if override_text and self.pipeline_manager.current_analysis_state:
            # Update working_title in analysis state with override
            self.pipeline_manager.current_analysis_state.working_title = override_text
            self.title_label.setText(f"📋 {override_text}")

            # Update MainWindow title if available
            if self.main_window and hasattr(self.main_window, 'update_window_title'):
                self.main_window.update_window_title(override_text)

            self.logger.info(f"Title override applied: {override_text}")

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
            
            # Emit DOI resolution results to CrossrefTab
            elif step.step_id == "input" and step.output_data.get("source_info", "").startswith("DOI"):
                # If input was from DOI resolution, emit metadata if available
                if "metadata" in step.output_data:
                    metadata = step.output_data["metadata"]
                    self.logger.debug("Emitting DOI metadata to CrossrefTab")
                    self.metadata_ready.emit(metadata)
            
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
            if state.dk_classifications:
                loaded_steps.append("DK-Klassifikation")

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

                if state.search_results and hasattr(self, 'search_result'):
                    search_count = len(state.search_results)
                    total_results = sum(len(sr.results) for sr in state.search_results)
                    # Also list the GND terms found (same as gnd_treffer during live runs)
                    gnd_terms = []
                    for sr in state.search_results:
                        gnd_terms.extend(sr.results.keys())
                    gnd_terms_text = "\n".join(sorted(set(gnd_terms)))
                    self.search_result.setPlainText(
                        f"📁 Geladene Suchergebnisse:\n{search_count} Suchvorgänge mit {total_results} Ergebnissen\n\n"
                        f"{gnd_terms_text}"
                    )

                if state.final_llm_analysis and hasattr(self, 'keywords_result'):
                    # Type-safe join - Claude Generated (Fix for string parsing bug)
                    final_kw = state.final_llm_analysis.extracted_gnd_keywords
                    final_keywords = (", ".join(final_kw)
                                      if isinstance(final_kw, list)
                                      else str(final_kw))
                    self.keywords_result.setPlainText(f"📁 Finale Schlagwörter:\n{final_keywords}")

                # DK Classification Results Display - Claude Generated (Enhanced with titles)
                if state.dk_classifications and hasattr(self, 'dk_classification_results'):
                    html_display = self._format_dk_classifications_with_titles(
                        state.dk_classifications,
                        state.dk_search_results_flattened  # flattened format has {dk, titles} at top level
                    )
                    self.dk_classification_results.setHtml(
                        f"<div style='background: #E8F5E8; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                        f"<strong>📁 Geladene DK-Klassifikationen</strong>"
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
