"""
Pipeline Tab - Vertical pipeline UI for ALIMA workflow
Claude Generated - Orchestrates the complete analysis pipeline in a chat-like interface
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
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

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Now that all UI elements are created, update the display - Claude Generated
        self.update_status_display()

    def _update_provider_model_display(self):
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
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate

        elif self.step.status == "completed":
            self.status_label.setText("✓")
            self.status_label.setStyleSheet(
                "color: #4caf50; font-size: 16px; font-weight: bold;"
            )
            self.setStyleSheet(
                "QFrame { border-color: #4caf50; background-color: #e8f5e8; }"
            )
            self.progress_bar.setVisible(False)

        elif self.step.status == "error":
            self.status_label.setText("✗")
            self.status_label.setStyleSheet(
                "color: #d32f2f; font-size: 16px; font-weight: bold;"
            )
            self.setStyleSheet(
                "QFrame { border-color: #d32f2f; background-color: #ffebee; }"
            )
            self.progress_bar.setVisible(False)

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
            100
        )  # Update every 100ms for smooth display

        # UI components
        self.step_widgets: Dict[str, PipelineStepWidget] = {}
        self.unified_input: Optional[UnifiedInputWidget] = None

        # Input state
        self.current_input_text: str = ""
        self.current_source_info: str = ""

        self.setup_ui()

    def update_current_step_duration(self):
        """Update the duration of the currently running step - Claude Generated"""
        if (
            self.current_running_step
            and self.current_running_step in self.step_start_times
            and hasattr(self, "step_progress_labels")
            and self.current_running_step in self.step_progress_labels
        ):

            # Calculate current duration
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

            # Update the label with live duration
            self.step_progress_labels[self.current_running_step].setText(
                f"▶ {step_name} ({duration_seconds:.1f}s)"
            )

    def setup_ui(self):
        """Setup the pipeline UI - Claude Generated"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main pipeline area (control header moved to compact widget)
        self.setup_pipeline_area(main_layout)

    def setup_pipeline_area(self, main_layout):
        """Setup main pipeline area with streaming feedback - Claude Generated"""
        # Create a main splitter for pipeline steps and streaming
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Pipeline steps as vertical tabs
        steps_splitter = QSplitter(Qt.Orientation.Vertical)

        self.pipeline_tabs = QTabWidget()
        self.pipeline_tabs.setTabPosition(QTabWidget.TabPosition.West)
        self.pipeline_tabs.setTabShape(QTabWidget.TabShape.Rounded)
        self.pipeline_tabs.setMinimumWidth(500)  # Reduced from 600

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
            """
            QTabWidget::pane {
                border: 1px solid #ddd;
                background: white;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background: #f5f5f5;
                border: 1px solid #ddd;
                padding: 12px 8px;
                margin: 2px;
                min-width: 15px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #2196f3;
                color: white;
                border-color: #1976d2;
            }
            QTabBar::tab:hover {
                background: #e3f2fd;
            }
        """
        )

        # Create pipeline step tabs
        self.create_pipeline_step_tabs()
        steps_splitter.addWidget(self.pipeline_tabs)

        # Add compact pipeline control/progress widget below steps
        control_widget = self.create_compact_pipeline_control()
        steps_splitter.addWidget(control_widget)

        # Set initial sizes for steps area (smaller control area, more space for steps)
        steps_splitter.setSizes([450, 100])
        main_splitter.addWidget(steps_splitter)

        # Right side: Live streaming widget
        self.stream_widget = PipelineStreamWidget()

        # Connect streaming widget signals
        self.stream_widget.cancel_pipeline.connect(self.reset_pipeline)
        # self.stream_widget.pause_pipeline.connect(self.pause_pipeline)  # TODO: Implement pause

        main_splitter.addWidget(self.stream_widget)

        # Set splitter sizes (more space for streaming widget)
        main_splitter.setSizes([500, 700])

        main_layout.addWidget(main_splitter)

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
        initialisation_step = PipelineStep(
            step_id="initialisation",
            name="🔤 SCHRITT 2: INITIALISIERUNG",
            status="pending",
            provider="gemini",
            model="gemini-1.5-flash",
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
        keywords_step = PipelineStep(
            step_id="keywords",
            name="✅ SCHRITT 4: SCHLAGWORTE",
            status="pending",
            provider="gemini",
            model="gemini-1.5-flash",
        )
        keywords_widget = self.create_keywords_step_widget()
        keywords_step_widget = PipelineStepWidget(keywords_step)
        keywords_step_widget.set_content(keywords_widget)
        self.step_widgets["keywords"] = keywords_step_widget
        self.pipeline_tabs.addTab(keywords_step_widget, "✅ Schlagwort-Verifikation")

        # Step 5: DK Search (catalog search)
        dk_search_step = PipelineStep(
            step_id="dk_search",
            name="📊 SCHRITT 5: DK-KATALOG-SUCHE (Optional)",
            status="pending",
        )
        dk_search_widget = self.create_dk_search_step_widget()
        dk_search_step_widget = PipelineStepWidget(dk_search_step)
        dk_search_step_widget.set_content(dk_search_widget)
        self.step_widgets["dk_search"] = dk_search_step_widget
        self.pipeline_tabs.addTab(dk_search_step_widget, "📊 Katalog-Recherche")

        # Step 6: DK Classification (LLM analysis)
        dk_classification_step = PipelineStep(
            step_id="dk_classification",
            name="📚 SCHRITT 6: DK-KLASSIFIKATION (Optional)",
            status="pending",
        )
        dk_classification_widget = self.create_dk_classification_step_widget()
        dk_classification_step_widget = PipelineStepWidget(dk_classification_step)
        dk_classification_step_widget.set_content(dk_classification_widget)
        self.step_widgets["dk_classification"] = dk_classification_step_widget
        self.pipeline_tabs.addTab(dk_classification_step_widget, "📚 DK/RVK-Klassifikation")

    def create_compact_pipeline_control(self) -> QWidget:
        """Create compact pipeline control with progress and buttons - Claude Generated"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Top row: Pipeline buttons
        buttons_layout = QHBoxLayout()

        # Auto-pipeline button
        self.auto_pipeline_button = QPushButton("🚀 Auto-Pipeline")
        self.auto_pipeline_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """
        )
        self.auto_pipeline_button.clicked.connect(self.start_auto_pipeline)
        buttons_layout.addWidget(self.auto_pipeline_button)

        # Load JSON button - Claude Generated
        self.load_json_button = QPushButton("📁 JSON laden")
        self.load_json_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """
        )
        self.load_json_button.clicked.connect(self.load_json_state)
        self.load_json_button.setToolTip("Pipeline-State aus JSON-Datei laden")
        buttons_layout.addWidget(self.load_json_button)

        # Mode indicator
        self.mode_indicator_label = QLabel()
        self._update_mode_indicator()
        buttons_layout.addWidget(self.mode_indicator_label)

        # Configuration button
        config_button = QPushButton("⚙️ Config")
        config_button.setMaximumWidth(80)
        config_button.clicked.connect(self.show_pipeline_config)
        buttons_layout.addWidget(config_button)

        # Reset button
        reset_button = QPushButton("🔄 Reset")
        reset_button.setMaximumWidth(80)
        reset_button.clicked.connect(self.reset_pipeline)
        buttons_layout.addWidget(reset_button)

        # Stop button - Claude Generated
        self.stop_pipeline_button = QPushButton("⏹️ Stop")
        self.stop_pipeline_button.setMaximumWidth(80)
        self.stop_pipeline_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """
        )
        self.stop_pipeline_button.setVisible(False)  # Hidden until pipeline starts
        self.stop_pipeline_button.clicked.connect(self.on_stop_pipeline_requested)
        buttons_layout.addWidget(self.stop_pipeline_button)

        # Pause button (TODO: implement)
        pause_button = QPushButton("⏸️ Pause")
        pause_button.setMaximumWidth(80)
        pause_button.setEnabled(
            False
        )  # TODO: Enable when pause functionality is implemented
        buttons_layout.addWidget(pause_button)

        buttons_layout.addStretch()

        # Pipeline status
        self.pipeline_status_label = QLabel("Bereit für Pipeline-Start")
        self.pipeline_status_label.setStyleSheet(
            "font-weight: bold; color: #666; font-size: 11px;"
        )
        buttons_layout.addWidget(self.pipeline_status_label)

        layout.addLayout(buttons_layout)

        # Bottom area: Pipeline progress with timestamps
        progress_group = QGroupBox("Pipeline-Fortschritt")
        progress_group.setStyleSheet(
            "QGroupBox { font-size: 11px; font-weight: bold; }"
        )
        progress_layout = QHBoxLayout(
            progress_group
        )  # Horizontal layout for compactness
        progress_layout.setSpacing(10)

        self.step_progress_labels = {}
        steps = ["input", "initialisation", "search", "keywords", "dk_search", "dk_classification"]

        for i, step_id in enumerate(steps):
            step_name = {
                "input": "Input",
                "initialisation": "Initialisierung", 
                "search": "Suche",
                "keywords": "Schlagworte",
                "dk_search": "DK-Katalog-Suche",
                "dk_classification": "DK-Klassifikation",
            }[step_id]

            step_label = QLabel(f"{i+1}. {step_name}: ⏳")
            step_label.setStyleSheet(
                "padding: 3px; border-radius: 3px; font-size: 10px;"
            )
            step_label.setMinimumWidth(120)
            self.step_progress_labels[step_id] = step_label
            progress_layout.addWidget(step_label)

        layout.addWidget(progress_group)

        return control_widget

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

    def on_input_cleared(self):
        """Handle input clearing - Claude Generated"""
        self.current_input_text = ""
        self.current_source_info = ""

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

    def create_initialisation_step_widget(self) -> QWidget:
        """Create initialisation step widget - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Results area
        self.initialisation_result = QTextEdit()
        self.initialisation_result.setReadOnly(True)
        self.initialisation_result.setMinimumHeight(120)  # Increased from 100
        self.initialisation_result.setMaximumHeight(200)  # Allow more space
        self.initialisation_result.setPlaceholderText(
            "Freie Schlagworte werden hier angezeigt..."
        )
        layout.addWidget(QLabel("Extrahierte freie Schlagworte:"))
        layout.addWidget(self.initialisation_result)

        return widget

    def create_search_step_widget(self) -> QWidget:
        """Create search step widget - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Search results area
        self.search_result = QTextEdit()
        self.search_result.setReadOnly(True)
        self.search_result.setMinimumHeight(120)  # Increased from 100
        # self.search_result.setMaximumHeight(200)  # Allow more space
        self.search_result.setPlaceholderText("Suchergebnisse werden hier angezeigt...")
        layout.addWidget(QLabel("GND-Suchergebnisse:"))
        layout.addWidget(self.search_result)

        return widget

    def create_keywords_step_widget(self) -> QWidget:
        """Create keywords step widget (Verbale Erschließung) - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Keywords results
        self.keywords_result = QTextEdit()
        self.keywords_result.setReadOnly(True)
        self.keywords_result.setMinimumHeight(120)  # Increased from 100
        self.keywords_result.setMaximumHeight(200)  # Allow more space
        self.keywords_result.setPlaceholderText(
            "Finale Schlagworte werden hier angezeigt..."
        )
        layout.addWidget(QLabel("Finale GND-Schlagworte:"))
        layout.addWidget(self.keywords_result)

        return widget

    def create_dk_search_step_widget(self) -> QWidget:
        """Create DK search step widget for catalog search results - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Search configuration section
        config_group = QGroupBox("Katalog-Such-Konfiguration")
        config_layout = QVBoxLayout(config_group)
        
        # Max results control
        max_results_layout = QHBoxLayout()
        max_results_layout.addWidget(QLabel("Max. Ergebnisse:"))
        self.dk_search_max_results = QSpinBox()
        self.dk_search_max_results.setRange(5, 100)
        from ..utils.pipeline_defaults import DEFAULT_DK_MAX_RESULTS
        self.dk_search_max_results.setValue(DEFAULT_DK_MAX_RESULTS)
        self.dk_search_max_results.setToolTip("Max. Katalog-Suchergebnisse pro Keyword")
        max_results_layout.addWidget(self.dk_search_max_results)
        max_results_layout.addStretch()
        config_layout.addLayout(max_results_layout)

        # Frequency threshold control - Claude Generated
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Min. Häufigkeit:"))
        self.dk_frequency_threshold = QSpinBox()
        self.dk_frequency_threshold.setRange(1, 50)
        from ..utils.pipeline_defaults import DEFAULT_DK_FREQUENCY_THRESHOLD
        self.dk_frequency_threshold.setValue(DEFAULT_DK_FREQUENCY_THRESHOLD)
        self.dk_frequency_threshold.setToolTip("Nur DK-Codes mit >= N Vorkommen im Katalog werden an LLM übergeben")
        freq_layout.addWidget(self.dk_frequency_threshold)
        freq_layout.addStretch()
        config_layout.addLayout(freq_layout)

        # Force update checkbox - Claude Generated
        from PyQt6.QtWidgets import QCheckBox
        self.force_update_checkbox = QCheckBox("Katalog-Cache ignorieren")
        self.force_update_checkbox.setToolTip(
            "Erzwingt Live-Suche im Katalog und ignoriert gecachte Ergebnisse.\n"
            "Neue Titel werden mit bestehenden zusammengeführt (keine Ersetzung)."
        )
        self.force_update_checkbox.setChecked(False)
        config_layout.addWidget(self.force_update_checkbox)

        layout.addWidget(config_group)

        # Search results display
        results_group = QGroupBox("Katalog-Suchergebnisse")
        results_layout = QVBoxLayout(results_group)
        
        self.dk_search_results = QTextEdit()
        self.dk_search_results.setReadOnly(True)
        self.dk_search_results.setMinimumHeight(200)
        self.dk_search_results.setPlaceholderText(
            "Katalog-Suchergebnisse für DK/RVK-Klassifikationen werden hier angezeigt...\n"
            "Format: DK: 666.76 (Häufigkeit: 3) | Beispieltitel: ... | Keywords: ..."
        )
        results_layout.addWidget(self.dk_search_results)
        layout.addWidget(results_group)

        return widget

    def create_dk_classification_step_widget(self) -> QWidget:
        """Create DK classification step widget for LLM analysis - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Input data summary
        input_group = QGroupBox("Eingangsdaten für LLM-Klassifikation")
        input_layout = QVBoxLayout(input_group)
        
        self.dk_input_summary = QTextEdit()
        self.dk_input_summary.setReadOnly(True)
        self.dk_input_summary.setMaximumHeight(100)
        self.dk_input_summary.setPlaceholderText(
            "Zusammenfassung der Katalog-Suchergebnisse für LLM..."
        )
        input_layout.addWidget(self.dk_input_summary)
        layout.addWidget(input_group)

        # Final classification results
        results_group = QGroupBox("Finale DK/RVK-Klassifikationen")
        results_layout = QVBoxLayout(results_group)
        
        self.dk_classification_results = QTextEdit()
        self.dk_classification_results.setReadOnly(True)
        self.dk_classification_results.setMinimumHeight(150)
        self.dk_classification_results.setPlaceholderText(
            "Finale DK/RVK-Klassifikationen vom LLM werden hier angezeigt...\n"
            "Format: DK 666.76, RVK Q12, RVK QC 130, ..."
        )
        results_layout.addWidget(self.dk_classification_results)
        layout.addWidget(results_group)

        return widget

    def start_auto_pipeline(self):
        """Start the automatic pipeline in background thread - Claude Generated"""
        # Get input text from unified input widget
        input_text = getattr(self, "current_input_text", "")

        if not input_text:
            QMessageBox.warning(
                self,
                "Keine Eingabe",
                "Bitte wählen Sie eine Eingabequelle und stellen Sie Text bereit.",
            )
            return

        # Update DK configuration from GUI widgets - Claude Generated
        self._update_dk_config_from_gui()

        # Stop any existing worker
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.quit()
            self.pipeline_worker.wait()

        # Reset streaming widget for new pipeline
        if hasattr(self, "stream_widget"):
            self.stream_widget.reset_for_new_pipeline()

        # Update status
        self.pipeline_status_label.setText("Pipeline läuft...")
        self.auto_pipeline_button.setEnabled(False)
        self.stop_pipeline_button.setVisible(True)  # Show stop button - Claude Generated
        self.stop_pipeline_button.setEnabled(True)

        # Get force_update flag from checkbox - Claude Generated
        force_update = getattr(self, 'force_update_checkbox', None)
        force_update_enabled = force_update.isChecked() if force_update else False

        # Create and start worker thread - Claude Generated (added force_update parameter)
        self.pipeline_worker = PipelineWorker(
            self.pipeline_manager, input_text, "text", force_update=force_update_enabled
        )

        # Connect worker signals
        self.pipeline_worker.step_started.connect(self.on_step_started)
        self.pipeline_worker.step_completed.connect(self.on_step_completed)
        self.pipeline_worker.step_error.connect(self.on_step_error)
        self.pipeline_worker.pipeline_completed.connect(self.on_pipeline_completed)
        self.pipeline_worker.stream_token.connect(self.on_llm_stream_token)
        self.pipeline_worker.aborted.connect(self.on_pipeline_aborted)  # Claude Generated

        # Start the worker
        self.pipeline_worker.start()

        # Emit pipeline started signal
        self.pipeline_started.emit("pipeline_thread")

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_pipeline_started("pipeline_thread")

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

        # Reset progress labels
        if hasattr(self, "step_progress_labels"):
            steps = ["input", "initialisation", "search", "keywords", "dk_search", "dk_classification"]
            for i, step_id in enumerate(steps):
                if step_id in self.step_progress_labels:
                    # Claude Generated - Fixed step name mapping with fallback for DK steps
                    step_name = {
                        "input": "Input",
                        "initialisation": "Initialisierung",
                        "search": "Suche",
                        "keywords": "Schlagworte",
                        "dk_search": "DK-Katalog-Suche",
                        "dk_classification": "DK-Klassifikation",
                    }.get(step_id, step_id.title())  # Use .get() with fallback

                    self.step_progress_labels[step_id].setText(
                        f"{i+1}. {step_name}: ⏳"
                    )
                    self.step_progress_labels[step_id].setStyleSheet(
                        "padding: 3px; border-radius: 3px; font-size: 10px;"
                    )

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

        # Reset status
        self.pipeline_status_label.setText("Bereit für Pipeline-Start")
        self.auto_pipeline_button.setEnabled(True)

    def on_config_changed(self):
        """Handle configuration changes - Claude Generated (Webcam Feature)"""
        self.logger.info("Pipeline tab: Handling config change")

        # Update webcam frame visibility in unified input widget
        if hasattr(self, 'unified_input') and self.unified_input:
            self.unified_input._update_webcam_frame_visibility()
            self.logger.info("Webcam frame visibility updated")

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

        # Update progress label with running indicator
        if (
            hasattr(self, "step_progress_labels")
            and step.step_id in self.step_progress_labels
        ):
            step_name = {
                "input": "Input",
                "initialisation": "Initialisierung", 
                "search": "Suche",
                "keywords": "Schlagworte",
                "dk_search": "DK-Katalog-Suche",
                "dk_classification": "DK-Klassifikation",
            }.get(step.step_id, step.step_id.title())

            self.step_progress_labels[step.step_id].setText(f"▶ {step_name} (0.0s)")
            self.step_progress_labels[step.step_id].setStyleSheet(
                "background: #e3f2fd; padding: 4px; border-radius: 3px; color: #1976d2;"
            )

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

        # Update progress label with final duration
        if (
            hasattr(self, "step_progress_labels")
            and step.step_id in self.step_progress_labels
        ):
            step_name = {
                "input": "Input",
                "initialisation": "Initialisierung", 
                "search": "Suche",
                "keywords": "Schlagworte",
                "dk_search": "DK-Katalog-Suche",
                "dk_classification": "DK-Klassifikation",
            }.get(step.step_id, step.step_id.title())

            # Calculate duration
            duration_text = "?"
            if step.step_id in self.step_start_times:
                duration_seconds = (
                    datetime.now() - self.step_start_times[step.step_id]
                ).total_seconds()
                duration_text = f"{duration_seconds:.1f}s"

            self.step_progress_labels[step.step_id].setText(
                f"✓ {step_name} ({duration_text})"
            )
            self.step_progress_labels[step.step_id].setStyleSheet(
                "background: #e8f5e8; padding: 4px; border-radius: 3px; color: #2e7d32;"
            )

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
            # Display DK search results with counts and titles - Claude Generated
            dk_search_results = step.output_data.get("dk_search_results", [])
            if hasattr(self, "dk_search_results") and dk_search_results:
                # Format aggregated results for display
                result_lines = []
                for result in dk_search_results:
                    dk_code = result.get("dk", "")
                    count = result.get("count", 0)
                    titles = result.get("titles", [])
                    keywords = result.get("keywords", [])
                    classification_type = result.get("classification_type", "DK")

                    # FIX: Skip DKs with no titles or count=0 - Claude Generated
                    if not titles or count == 0:
                        continue

                    # Show titles (up to 3 per entry)
                    sample_titles = titles[:3]
                    titles_text = " | ".join(sample_titles)
                    if len(titles) > 3:
                        titles_text += f" | ... (und {len(titles) - 3} weitere)"

                    result_line = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nBeispieltitel: {titles_text}\nKeywords: {', '.join(keywords)}\n"
                    result_lines.append(result_line)
                
                self.dk_search_results.setPlainText("\n".join(result_lines))
            elif hasattr(self, "dk_search_results"):
                self.dk_search_results.setPlainText("Keine DK/RVK-Klassifikationen gefunden")

        elif step.step_id == "dk_classification" and step.output_data:
            # Display final DK classification results from LLM - Claude Generated
            dk_classifications = step.output_data.get("dk_classifications", [])
            if hasattr(self, "dk_classification_results") and dk_classifications:
                # Format final classifications
                classifications_text = "\n".join(dk_classifications)
                self.dk_classification_results.setPlainText(classifications_text)
                
                # Also update the input summary with search data from previous step
                if hasattr(self, "dk_input_summary"):
                    search_data = step.output_data.get("dk_search_summary", "")
                    if search_data:
                        self.dk_input_summary.setPlainText(search_data)
                    else:
                        self.dk_input_summary.setPlainText("Katalog-Suchergebnisse für LLM-Analyse")
            elif hasattr(self, "dk_classification_results"):
                self.dk_classification_results.setPlainText("Keine DK/RVK-Klassifikationen generiert")

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

        # Update progress label with final duration
        if (
            hasattr(self, "step_progress_labels")
            and step.step_id in self.step_progress_labels
        ):
            step_name = {
                "input": "Input",
                "initialisation": "Initialisierung", 
                "search": "Suche",
                "keywords": "Schlagworte",
                "dk_search": "DK-Katalog-Suche",
                "dk_classification": "DK-Klassifikation",
            }.get(step.step_id, step.step_id.title())

            # Calculate duration
            duration_text = "?"
            if step.step_id in self.step_start_times:
                duration_seconds = (
                    datetime.now() - self.step_start_times[step.step_id]
                ).total_seconds()
                duration_text = f"{duration_seconds:.1f}s"

            self.step_progress_labels[step.step_id].setText(
                f"✗ {step_name} ({duration_text})"
            )
            self.step_progress_labels[step.step_id].setStyleSheet(
                "background: #ffebee; padding: 4px; border-radius: 3px; color: #c62828;"
            )

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
        self.stop_pipeline_button.setVisible(False)  # Hide stop button - Claude Generated
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

        QMessageBox.information(
            self,
            "Pipeline abgeschlossen",
            "Die komplette Analyse-Pipeline wurde erfolgreich abgeschlossen!",
        )

    def on_stop_pipeline_requested(self):
        """Handle stop button click - Claude Generated"""
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.logger.info("User requested pipeline stop")
            self.stop_pipeline_button.setEnabled(False)
            self.stop_pipeline_button.setText("⏹️ Stopping...")
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
        self.stop_pipeline_button.setVisible(False)
        self.stop_pipeline_button.setText("⏹️ Stop")
        self.stop_pipeline_button.setEnabled(True)

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
                self.logger.info(f"Loaded catalog token from config (length: {len(catalog_token)})")
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
        
        self.logger.info(f"Updated pipeline config with catalog settings (token present: {bool(self.catalog_token)})")

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
                self.logger.info(f"Emitting search results to SearchTab: {len(search_results)} terms")
                self.search_results_ready.emit(search_results)
            
            # Emit DOI resolution results to CrossrefTab
            elif step.step_id == "input" and step.output_data.get("source_info", "").startswith("DOI"):
                # If input was from DOI resolution, emit metadata if available
                if "metadata" in step.output_data:
                    metadata = step.output_data["metadata"]
                    self.logger.info("Emitting DOI metadata to CrossrefTab")
                    self.metadata_ready.emit(metadata)
            
            # Emit keyword analysis results to AbstractTab 
            elif step.step_id in ["initialisation", "keywords"] and "analysis_result" in step.output_data:
                analysis_result = step.output_data["analysis_result"]
                self.logger.info(f"Emitting {step.step_id} analysis results to AbstractTab")
                self.analysis_results_ready.emit(analysis_result)
                
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
                    self.search_result.setPlainText(
                        f"📁 Geladene Suchergebnisse:\n{search_count} Suchvorgänge mit {total_results} Ergebnissen"
                    )

                if state.final_llm_analysis and hasattr(self, 'keywords_result'):
                    # Type-safe join - Claude Generated (Fix for string parsing bug)
                    final_kw = state.final_llm_analysis.extracted_gnd_keywords
                    final_keywords = (", ".join(final_kw)
                                      if isinstance(final_kw, list)
                                      else str(final_kw))
                    self.keywords_result.setPlainText(f"📁 Finale Schlagwörter:\n{final_keywords}")

                # DK Classification Results Display - Claude Generated (Fixed widget name)
                if state.dk_classifications and hasattr(self, 'dk_classification_results'):
                    # Type-safe join - Claude Generated (Fix for string parsing bug)
                    dk_text = (", ".join(state.dk_classifications)
                               if isinstance(state.dk_classifications, list)
                               else str(state.dk_classifications))
                    self.dk_classification_results.setPlainText(f"📁 DK-Klassifikationen:\n{dk_text}")

                # DK Search Results Display - Claude Generated (New logic for loaded states)
                if state.dk_search_results and hasattr(self, 'dk_search_results'):
                    # Format search results like in on_step_completed (lines 1217-1239)
                    result_lines = []
                    for result in state.dk_search_results:
                        dk_code = result.get("dk", "")
                        count = result.get("count", 0)
                        titles = result.get("titles", [])
                        keywords = result.get("keywords", [])
                        classification_type = result.get("classification_type", "DK")

                        # Show titles (up to 3 per entry)
                        sample_titles = titles[:3]
                        titles_text = " | ".join(sample_titles)
                        if len(titles) > 3:
                            titles_text += f" | ... (und {len(titles) - 3} weitere)"

                        result_line = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nBeispieltitel: {titles_text}\nKeywords: {', '.join(keywords)}\n"
                        result_lines.append(result_line)

                    self.dk_search_results.setPlainText(f"📁 Geladene DK-Suchergebnisse:\n\n" + "\n".join(result_lines))

            self.logger.info(f"Pipeline tab updated with loaded state indicators: {loaded_steps}")

        except Exception as e:
            self.logger.error(f"Error showing loaded state indicator: {e}")
