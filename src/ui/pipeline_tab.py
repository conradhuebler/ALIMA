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


class PipelineWorker(QThread):
    """Worker thread for pipeline execution - Claude Generated"""

    # Signals for pipeline events
    step_started = pyqtSignal(object)  # PipelineStep
    step_completed = pyqtSignal(object)  # PipelineStep
    step_error = pyqtSignal(object, str)  # PipelineStep, error_message
    pipeline_completed = pyqtSignal(object)  # analysis_state
    stream_token = pyqtSignal(str, str)  # token, step_id

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        input_text: str,
        input_type: str = "text",
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.input_text = input_text
        self.input_type = input_type
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute pipeline in background thread - Claude Generated"""
        try:
            # Set up callbacks to emit signals
            self.pipeline_manager.set_callbacks(
                step_started=self.step_started.emit,
                step_completed=self.step_completed.emit,
                step_error=self.step_error.emit,
                pipeline_completed=self.pipeline_completed.emit,
                stream_callback=self.stream_token.emit,
            )

            # Start pipeline
            pipeline_id = self.pipeline_manager.start_pipeline(
                self.input_text, self.input_type
            )
            self.logger.info(f"Pipeline {pipeline_id} completed in worker thread")

        except Exception as e:
            self.logger.error(f"Pipeline worker error: {e}")
            # Emit error signal if needed


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
        self.update_status_display()
        header_layout.addWidget(self.status_label)

        # Step name
        name_label = QLabel(self.step.name)
        name_font = QFont()
        name_font.setPointSize(12)
        name_font.setBold(True)
        name_label.setFont(name_font)
        header_layout.addWidget(name_label)

        header_layout.addStretch()

        # Provider/Model info
        if self.step.provider and self.step.model:
            provider_label = QLabel(f"{self.step.provider} ({self.step.model})")
            provider_label.setStyleSheet("color: #666; font-size: 10px;")
            header_layout.addWidget(provider_label)

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

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService,
        cache_manager: UnifiedKnowledgeManager,
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
        
        # Pipeline manager
        self.pipeline_manager = PipelineManager(
            alima_manager=alima_manager, cache_manager=cache_manager, logger=self.logger
        )
        
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
        self.unified_input = UnifiedInputWidget(llm_service=self.llm_service)

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
        self.dk_search_max_results.setValue(20)
        max_results_layout.addWidget(self.dk_search_max_results)
        max_results_layout.addStretch()
        config_layout.addLayout(max_results_layout)
        
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

        # Create and start worker thread
        self.pipeline_worker = PipelineWorker(self.pipeline_manager, input_text, "text")

        # Connect worker signals
        self.pipeline_worker.step_started.connect(self.on_step_started)
        self.pipeline_worker.step_completed.connect(self.on_step_completed)
        self.pipeline_worker.step_error.connect(self.on_step_error)
        self.pipeline_worker.pipeline_completed.connect(self.on_pipeline_completed)
        self.pipeline_worker.stream_token.connect(self.on_llm_stream_token)

        # Start the worker
        self.pipeline_worker.start()

        # Emit pipeline started signal
        self.pipeline_started.emit("pipeline_thread")

        # Notify streaming widget
        if hasattr(self, "stream_widget"):
            self.stream_widget.on_pipeline_started("pipeline_thread")

    def show_pipeline_config(self):
        """Show pipeline configuration dialog - Claude Generated"""
        prompt_service = None
        if hasattr(self.alima_manager, "prompt_service"):
            prompt_service = self.alima_manager.prompt_service

        dialog = PipelineConfigDialog(
            llm_service=self.llm_service,
            prompt_service=prompt_service,
            current_config=self.pipeline_manager.config,
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

    def update_step_display_from_config(self):
        """Update step widgets based on current configuration - Claude Generated"""
        config = self.pipeline_manager.config

        # Update provider/model display for each step
        for step_id, step_widget in self.step_widgets.items():
            if step_id in config.step_configs:
                step_config = config.step_configs[step_id]
                provider = step_config.get("provider", "")
                model = step_config.get("model", "")
                enabled = step_config.get("enabled", True)

                # Update step data
                step_widget.step.provider = provider
                step_widget.step.model = model

                # Update display (visual styling based on enabled state)
                if not enabled:
                    step_widget.setStyleSheet("QFrame { opacity: 0.5; }")
                else:
                    step_widget.setStyleSheet("")

                step_widget.update_status_display()

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
                    step_name = {
                        "input": "Input",
                        "initialisation": "Initialisierung",
                        "search": "Suche",
                        "keywords": "Schlagworte",
                        "classification": "Klassifikation",
                    }[step_id]

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
        if hasattr(self, "classification_result"):
            self.classification_result.clear()

        # Reset status
        self.pipeline_status_label.setText("Bereit für Pipeline-Start")
        self.auto_pipeline_button.setEnabled(True)

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

        QMessageBox.information(
            self,
            "Pipeline abgeschlossen",
            "Die komplette Analyse-Pipeline wurde erfolgreich abgeschlossen!",
        )

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
        try:
            from ..utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            catalog_config = config_manager.get_catalog_config()
            
            catalog_token = catalog_config.get("catalog_token", "")
            catalog_search_url = catalog_config.get("catalog_search_url", "")
            catalog_details_url = catalog_config.get("catalog_details_url", "")
            
            if catalog_token:
                self.logger.info(f"Loaded catalog token from config (length: {len(catalog_token)})")
            else:
                self.logger.warning("No catalog token found in config")
            #else:
            #    self.logger.warning(f"Config file not found: {config_file}")
                
        except Exception as e:
            self.logger.error(f"Error loading catalog config: {e}")
        
        return catalog_token, catalog_search_url, catalog_details_url
    
    def _update_pipeline_config_with_catalog_settings(self):
        """Update pipeline config with loaded catalog settings - Claude Generated"""
        config = self.pipeline_manager.config
        
        # Update DK search step configuration
        if "dk_search" in config.step_configs:
            config.step_configs["dk_search"].update({
                "catalog_token": self.catalog_token,
                "catalog_search_url": self.catalog_search_url,
                "catalog_details_url": self.catalog_details_url,
            })
            
        # Also update DK classification step if it exists
        if "dk_classification" in config.step_configs:
            config.step_configs["dk_classification"].update({
                "catalog_token": self.catalog_token,
                "catalog_search_url": self.catalog_search_url,
                "catalog_details_url": self.catalog_details_url,
            })
        
        self.logger.info(f"Updated pipeline config with catalog settings (token present: {bool(self.catalog_token)})")
