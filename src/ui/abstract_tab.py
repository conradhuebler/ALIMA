"""
AbstractTab - Restructured for improved chunking workflow
Changes: AI config moved right, QSplitter for prompt/result, chunk navigation
"""

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QMessageBox,
    QProgressBar,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QGridLayout,
    QFrame,
    QToolButton,
    QGroupBox,
    QFileDialog,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QTabWidget,
    QStatusBar,
)
from PyQt6.QtCore import (
    Qt,
    pyqtSlot,
    pyqtSignal,
    QThread,
    QObject,
    QSize,
    QPoint,
    QRect,
    QRegularExpression,
)
from PyQt6.QtGui import (
    QIcon,
    QFont,
    QColor,
    QPalette,
    QPixmap,
    QAction,
    QSyntaxHighlighter,
    QTextCharFormat,
    QFont,
)
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QSize

from ..llm.llm_service import LlmService
from ..utils.config import Config, ConfigSection, AIConfig
from ..llm.prompt_service import PromptService
from ..core.alima_manager import AlimaManager
from ..core.data_models import AbstractData, AnalysisResult, KeywordAnalysisState
from ..core.pipeline_manager import PipelineManager, PipelineStep, PipelineConfig
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..utils.config_models import PipelineStepConfig, PipelineMode
from .workers import PipelineWorker

from pathlib import Path
import os
import json
import logging
import re
import tempfile
import PyPDF2
import threading
import time
from typing import List, Tuple, Dict, Optional
import uuid


class AbstractTab(QWidget):
    """
    Modernized AbstractTab with improved chunking workflow:
    - AI config moved to right side for better space utilization
    - QSplitter for prompt/result display
    - Chunk navigation through results list
    - Combined results handling
    """

    # Signals
    keywords_extracted = pyqtSignal(str)
    abstract_changed = pyqtSignal(str)
    final_list = pyqtSignal(str)
    gnd_systematic = pyqtSignal(str)
    analysis_completed = pyqtSignal(
        str, str, str
    )  # abstract, keywords, analysis_result

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService,
        cache_manager: UnifiedKnowledgeManager,
        pipeline_manager: PipelineManager,
        main_window: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        # Inject dependencies
        self.alima_manager = alima_manager
        self.llm = llm_service  # Renamed from self.llm_service to self.llm for consistency with existing code
        self.cache_manager = cache_manager
        self.prompt_manager = (
            self.alima_manager.prompt_service
        )  # Access prompt_service via alima_manager
        self.main_window = main_window

        # Use injected central PipelineManager instead of creating redundant instance - Claude Generated
        self.pipeline_manager = pipeline_manager

        self.need_keywords = False
        self.logger = logging.getLogger(__name__)
        self.task: Optional[str] = None
        self.chosen_model: str = "default"
        self.available_models: Dict[str, List[str]] = {}
        self.results_history = []  # Store results history
        self.is_analysis_running = False  # Track analysis state
        self.input_widget_visible = True  # Track input widget visibility

        # Signal connections moved to central MainWindow management - Claude Generated
        # self.llm.ollama_url_updated.connect(self.on_ollama_url_updated)
        # self.llm.ollama_port_updated.connect(self.on_ollama_port_updated)

        # Set up the UI
        self.setup_ui()

    # ======== DEPRECATED METHODS - Moved to central MainWindow management ========
    def on_ollama_url_updated(self):
        """DEPRECATED: Ollama URL updates now handled centrally in MainWindow - Claude Generated"""
        pass

    def on_ollama_port_updated(self):
        """DEPRECATED: Ollama Port updates now handled centrally in MainWindow - Claude Generated"""
        pass

    def set_task(self, task: str):
        """Set the task type for model recommendations and update UI."""
        self.task = task
        self.logger.info(f"Task set to: {self.task}")
        index = self.task_selector_combo.findText(task)
        if index >= 0:
            self.task_selector_combo.setCurrentIndex(index)
        else:
            self.task_selector_combo.addItem(task)
            self.task_selector_combo.setCurrentText(task)
            if self.task_selector_combo.count() == 1:
                self.populate_prompt_selector()

        # Show PDF button for initialisation task (formerly "abstract") - Claude Generated
        self.pdf_button.setVisible(self.task in ["abstract", "initialisation"])

    def update_models(self, provider: str):
        """Update available models when provider changes."""
        self.model_combo.clear()
        if provider in self.available_models:
            self.model_combo.addItems(self.available_models[provider])

    def setup_ui(self):
        """Set up the user interface with restructured layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ======== Control Bar ========
        control_bar = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Generiere Antwort... %p%")
        control_bar.addWidget(self.progress_bar)
        main_layout.addLayout(control_bar)

        # ======== Input and Config Side by Side ========
        input_config_splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT SIDE: Input Group (Abstract/Keywords)
        self.input_group = QGroupBox("Eingabe")
        input_layout = QVBoxLayout(self.input_group)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Abstract / Text:"))
        header_layout.addStretch(1)
        self.pdf_button = QPushButton("PDF importieren")
        self.pdf_button.clicked.connect(self.import_pdf)
        header_layout.addWidget(self.pdf_button)
        input_layout.addLayout(header_layout)

        self.abstract_edit = QTextEdit()
        # Increase font size
        font = self.abstract_edit.font()
        font.setPointSize(11)
        self.abstract_edit.setFont(font)
        input_layout.addWidget(self.abstract_edit)

        input_layout.addWidget(QLabel("Vorhandene Keywords (optional):"))
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setMaximumHeight(80)
        # Increase font size
        font = self.keywords_edit.font()
        font.setPointSize(11)
        self.keywords_edit.setFont(font)
        input_layout.addWidget(self.keywords_edit)

        input_config_splitter.addWidget(self.input_group)

        # RIGHT SIDE: AI Configuration and Chunking

        config_widget = QWidget()
        config_main_layout = QVBoxLayout(config_widget)

        self.config_group = QGroupBox("KI-Konfiguration")
        config_layout = QHBoxLayout(self.config_group)  # Changed from QVBoxLayout

        # LEFT: Prompt + Parameter Tabs
        left_config_layout = QVBoxLayout()
        self.config_tabs = QTabWidget()

        # -- Prompt Tab --
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)

        prompt_selection_group = QGroupBox("Prompt-Auswahl")
        prompt_selection_layout = QGridLayout(prompt_selection_group)
        prompt_selection_layout.addWidget(QLabel("Task:"), 0, 0)
        self.task_selector_combo = QComboBox()
        self.task_selector_combo.currentIndexChanged.connect(self.on_task_selected)
        prompt_selection_layout.addWidget(self.task_selector_combo, 0, 1)
        prompt_selection_layout.addWidget(QLabel("Prompt:"), 1, 0)
        self.prompt_selector_combo = QComboBox()
        self.prompt_selector_combo.currentIndexChanged.connect(self.on_prompt_selected)
        prompt_selection_layout.addWidget(self.prompt_selector_combo, 1, 1)
        prompt_layout.addWidget(prompt_selection_group)

        prompt_layout.addWidget(QLabel("Prompt-Vorlage:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(150)
        prompt_layout.addWidget(self.prompt_edit)

        prompt_layout.addWidget(QLabel("System-Prompt (optional):"))
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setMinimumHeight(120)
        prompt_layout.addWidget(self.system_prompt_edit)

        self.config_tabs.addTab(prompt_tab, "Prompt")

        # -- Parameters Tab --
        params_tab = QWidget()
        params_layout = QGridLayout(params_tab)

        # Temperature
        params_layout.addWidget(QLabel("Temperatur:"), 0, 0)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_spinbox.setValue(v / 100.0)
        )
        params_layout.addWidget(self.temp_slider, 0, 1)
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setDecimals(2)
        self.temp_spinbox.setRange(0.0, 1.0)
        self.temp_spinbox.setSingleStep(0.01)
        self.temp_spinbox.valueChanged.connect(
            lambda v: self.temp_slider.setValue(int(v * 100))
        )
        params_layout.addWidget(self.temp_spinbox, 0, 2)

        # Top-P
        params_layout.addWidget(QLabel("Top-P:"), 1, 0)
        self.p_value_slider = QSlider(Qt.Orientation.Horizontal)
        self.p_value_slider.setRange(0, 100)
        self.p_value_slider.valueChanged.connect(
            lambda v: self.p_value_spinbox.setValue(v / 100.0)
        )
        params_layout.addWidget(self.p_value_slider, 1, 1)
        self.p_value_spinbox = QDoubleSpinBox()
        self.p_value_spinbox.setDecimals(2)
        self.p_value_spinbox.setRange(0.0, 1.0)
        self.p_value_spinbox.setSingleStep(0.01)
        self.p_value_spinbox.valueChanged.connect(
            lambda v: self.p_value_slider.setValue(int(v * 100))
        )
        params_layout.addWidget(self.p_value_spinbox, 1, 2)

        # Seed
        params_layout.addWidget(QLabel("Seed:"), 2, 0)
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999999999)
        self.seed_spinbox.setValue(0)
        params_layout.addWidget(self.seed_spinbox, 2, 1, 1, 2)

        self.config_tabs.addTab(params_tab, "Parameter")
        left_config_layout.addWidget(self.config_tabs)

        # RIGHT: Provider & Chunking Controls
        right_config_layout = QVBoxLayout()

        provider_model_group = QGroupBox("Provider & Modell")
        provider_model_layout = QGridLayout(provider_model_group)
        provider_model_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        # Initialize with placeholder - will be populated by ProviderStatusService - Claude Generated
        self.provider_combo.addItem("Loading providers...")
        self.provider_combo.currentTextChanged.connect(self.update_models)
        provider_model_layout.addWidget(self.provider_combo, 0, 1)
        provider_model_layout.addWidget(QLabel("Modell:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.set_model)
        provider_model_layout.addWidget(self.model_combo, 1, 1)
        right_config_layout.addWidget(provider_model_group)

        chunk_group = QGroupBox("Chunking-Kontrolle (optional)")
        chunk_layout = QVBoxLayout(chunk_group)
        self.enable_chunk_abstract = QCheckBox("Abstract-Chunking")
        self.abstract_chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self.abstract_chunk_slider.setEnabled(False)
        self.enable_chunk_abstract.toggled.connect(
            self.abstract_chunk_slider.setEnabled
        )
        chunk_layout.addWidget(self.enable_chunk_abstract)
        chunk_layout.addWidget(self.abstract_chunk_slider)

        self.enable_chunk_keywords = QCheckBox("Keywords-Chunking")
        self.keyword_chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self.keyword_chunk_slider.setEnabled(False)
        self.enable_chunk_keywords.toggled.connect(self.keyword_chunk_slider.setEnabled)
        chunk_layout.addWidget(self.enable_chunk_keywords)
        chunk_layout.addWidget(self.keyword_chunk_slider)
        right_config_layout.addWidget(chunk_group)

        # Combine both columns
        config_layout.addLayout(left_config_layout, stretch=3)
        config_layout.addLayout(right_config_layout, stretch=2)

        # Add the group to main layout
        config_main_layout.addWidget(self.config_group)

        input_config_splitter.addWidget(config_widget)

        # ======== Analysis Button Area (inside input widget) ========
        analysis_button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyse starten")
        self.analyze_button.clicked.connect(self.start_analysis)
        analysis_button_layout.addWidget(self.analyze_button)

        # Cancel button (initially hidden)
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setStyleSheet(
            "QPushButton { background-color: #d32f2f; color: white; }"
        )
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setVisible(False)
        analysis_button_layout.addWidget(self.cancel_button)

        analysis_button_layout.addStretch()

        # Status label for analysis feedback
        self.status_label = QLabel("Bereit")
        self.status_label.setStyleSheet("QLabel { color: #4caf50; font-weight: bold; }")
        analysis_button_layout.addWidget(self.status_label)

        # Create input widget that includes input/config and analysis buttons (this will be hidden/shown)
        self.input_widget = QWidget()
        input_widget_layout = QVBoxLayout(self.input_widget)
        input_widget_layout.addWidget(input_config_splitter)
        input_widget_layout.addLayout(analysis_button_layout)

        # ======== Control Button Area (always visible) ========
        control_button_layout = QHBoxLayout()

        # Toggle view button (always visible)
        self.toggle_input_button = QPushButton("Eingabe ausblenden")
        self.toggle_input_button.clicked.connect(self.toggle_input_visibility)
        control_button_layout.addWidget(self.toggle_input_button)

        # Auto-hide during streaming checkbox
        self.auto_hide_checkbox = QCheckBox("Auto-Ausblenden beim Streaming")
        self.auto_hide_checkbox.setChecked(True)
        control_button_layout.addWidget(self.auto_hide_checkbox)

        control_button_layout.addStretch()

        # ======== Main Content Area ========
        # Container for control buttons and splitter
        main_container = QWidget()
        main_container_layout = QVBoxLayout(main_container)

        # Add control buttons (always visible)
        main_container_layout.addLayout(control_button_layout)

        # Create main splitter
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Add input widget to splitter
        self.main_splitter.addWidget(self.input_widget)

        # ======== Results Area ========
        results_widget = QWidget()
        results_main_layout = QVBoxLayout(results_widget)

        self.results_group = QGroupBox("Analyseergebnis")
        results_layout = QHBoxLayout(self.results_group)

        # Results text area with enhanced styling
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        # Increase font size
        font = self.results_edit.font()
        font.setPointSize(11)
        self.results_edit.setFont(font)
        # Add better styling for streaming text
        self.results_edit.setStyleSheet(
            """
            QTextEdit {
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                padding: 8px;
                background-color: #fafafa;
            }
            QTextEdit:focus {
                border-color: #2196f3;
            }
        """
        )
        results_layout.addWidget(self.results_edit)

        # Results navigation sidebar
        nav_widget = QWidget()
        nav_widget.setMaximumWidth(250)
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.addWidget(QLabel("Ausgaben-Historie:"))

        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.load_result_from_history)
        nav_layout.addWidget(self.results_list)

        clear_button = QPushButton("Historie löschen")
        clear_button.clicked.connect(self.clear_results_history)
        nav_layout.addWidget(clear_button)

        results_layout.addWidget(nav_widget)

        results_main_layout.addWidget(self.results_group)
        self.main_splitter.addWidget(results_widget)

        # Set initial splitter sizes (input area smaller)
        self.main_splitter.setSizes([300, 700])

        # Add splitter to main container
        main_container_layout.addWidget(self.main_splitter)

        # Add main container to layout
        main_layout.addWidget(main_container)

        self.update_models(self.provider_combo.currentText())
        self.populate_task_selector()

    def load_result_from_history(self, item):
        """Load a result from the history list - Claude Generated (Enhanced for full state restoration)"""
        try:
            # DEFENSIVE: Clear only abstract and results, preserve keywords (user input) - Claude Generated
            self.abstract_edit.clear()
            # self.keywords_edit.clear()  # REMOVED - preserve user input
            self.results_edit.clear()

            # Try to get full state object from item data - Claude Generated
            state = item.data(Qt.ItemDataRole.UserRole)

            if state and isinstance(state, KeywordAnalysisState):
                # Full state restoration - Claude Generated
                self.logger.info("Restoring full analysis state from history")

                # 1. Restore abstract
                if state.original_abstract:
                    self.abstract_edit.setPlainText(state.original_abstract)

                # 2. DO NOT restore keywords - preserve user input - Claude Generated
                # (Keywords from analysis are visible in results_edit)

                # 3. Restore LLM response in results area
                result_text = ""
                if state.final_llm_analysis:
                    result_text = state.final_llm_analysis.response_full_text
                elif state.initial_llm_call_details:
                    result_text = state.initial_llm_call_details.response_full_text

                # DEBUG: Log what we're restoring - Claude Generated
                self.logger.info(f"History restore: result_text length = {len(result_text) if result_text else 0}")
                self.logger.info(f"History restore: has final_llm = {state.final_llm_analysis is not None}")
                self.logger.info(f"History restore: has initial_llm = {state.initial_llm_call_details is not None}")

                if result_text:
                    self.results_edit.setPlainText(result_text)
                    self.logger.info("✅ Results successfully restored")
                else:
                    self.logger.warning("⚠️ No result text to restore - state may be incomplete")

                self.logger.info("Successfully restored analysis state from history")
            else:
                # Fallback: Old string-based mechanism for backward compatibility - Claude Generated
                index = self.results_list.row(item)
                if 0 <= index < len(self.results_history):
                    result_data = self.results_history[index]
                    # Only restore results, not keywords (preserve user input) - Claude Generated
                    self.results_edit.setPlainText(result_data["result"])
                    self.logger.info("Restored result from old history format (text only)")

        except Exception as e:
            self.logger.error(f"Error loading result from history: {e}", exc_info=True)

    def clear_results_history(self):
        """Clear the results history."""
        self.results_history.clear()
        self.results_list.clear()
        self.results_edit.clear()

    def add_result_to_history(self, result_text, prompt_info="", state_object: Optional[KeywordAnalysisState] = None):
        """Add a result to the history - Claude Generated (Extended for full state storage)"""
        import datetime

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Create history entry
        history_entry = {
            "timestamp": timestamp,
            "result": result_text,
            "prompt": prompt_info,
        }

        self.results_history.append(history_entry)

        # Add to UI list
        display_text = f"{timestamp} - {len(result_text)} chars"
        if prompt_info:
            display_text += f" ({prompt_info[:30]}...)"

        item = QListWidgetItem(display_text)

        # Store full state object if provided - Claude Generated
        if state_object:
            item.setData(Qt.ItemDataRole.UserRole, state_object)

        self.results_list.addItem(item)

        # Auto-select newest entry for visual consistency - Claude Generated
        self.results_list.setCurrentItem(item)

        # Keep only last 10 results
        if len(self.results_history) > 10:
            self.results_history.pop(0)
            self.results_list.takeItem(0)

    def add_external_analysis_to_history(self, state: KeywordAnalysisState):
        """Add externally loaded analysis to tab history - Claude Generated"""
        import datetime

        # Extract display information from state
        timestamp_display = datetime.datetime.now().strftime("%H:%M:%S")

        # Count keywords
        keyword_count = 0
        if state.final_llm_analysis and state.final_llm_analysis.extracted_gnd_keywords:
            keyword_count = len(state.final_llm_analysis.extracted_gnd_keywords)
        elif state.initial_keywords:
            keyword_count = len(state.initial_keywords)

        # Create display info
        prompt_info = f"Loaded: {keyword_count} keywords"

        # Extract result text
        result_text = ""
        if state.final_llm_analysis:
            result_text = state.final_llm_analysis.response_full_text
        elif state.initial_llm_call_details:
            result_text = state.initial_llm_call_details.response_full_text
        else:
            result_text = f"Abstract: {state.original_abstract[:200]}..." if state.original_abstract else "No content"

        # Add to history with full state object
        self.add_result_to_history(result_text, prompt_info, state_object=state)

        # Ensure the loaded entry is visually selected - Claude Generated
        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(self.results_list.count() - 1)

        self.logger.info(f"Added external analysis to history: {keyword_count} keywords")

    def _create_state_from_current_analysis(self, llm_analysis=None, keywords_list=None) -> KeywordAnalysisState:
        """Create KeywordAnalysisState from current tab state - Claude Generated"""
        import datetime
        from ..core.data_models import LlmKeywordAnalysis

        # Get current abstract and keywords
        current_abstract = self.abstract_edit.toPlainText().strip()
        current_keywords = self.keywords_edit.toPlainText().strip()

        # Parse keywords into list
        if keywords_list:
            initial_keywords = keywords_list
        elif current_keywords:
            initial_keywords = [kw.strip() for kw in current_keywords.split(",") if kw.strip()]
        else:
            initial_keywords = []

        # Create state object
        state = KeywordAnalysisState(
            original_abstract=current_abstract,
            initial_keywords=initial_keywords,
            search_suggesters_used=["live_analysis"],
            initial_gnd_classes=[],
            search_results=[],
            initial_llm_call_details=llm_analysis,
            final_llm_analysis=llm_analysis,
            timestamp=datetime.datetime.now().isoformat()
        )

        return state

    def populate_task_selector(self):
        self.task_selector_combo.clear()
        tasks = self.prompt_manager.get_available_tasks()
        self.task_selector_combo.addItems(tasks)

    def on_task_selected(self, index):
        if index < 0:
            return
        self.task = self.task_selector_combo.currentText()
        self.populate_prompt_selector()

    def populate_prompt_selector(self):
        self.prompt_selector_combo.clear()
        if not self.task:
            return
        prompts = self.prompt_manager.get_prompts_for_task(self.task)
        for i, prompt_set in enumerate(prompts):
            self.prompt_selector_combo.addItem(f"Prompt Set {i+1}", userData=i)
        if self.prompt_selector_combo.count() > 0:
            self.prompt_selector_combo.setCurrentIndex(0)
            self.on_prompt_selected(0)

    def on_prompt_selected(self, index):
        if index < 0:
            return
        prompt_set_index = self.prompt_selector_combo.itemData(index)
        if prompt_set_index is None:
            return
        prompt_set = self.prompt_manager.get_prompts_for_task(self.task)[
            prompt_set_index
        ]

        # Safely get prompt template (index 0)
        prompt_template_text = prompt_set[0] if len(prompt_set) > 0 else ""
        self.prompt_edit.setPlainText(prompt_template_text)

        # Safely get system prompt (index 1)
        system_prompt_text = prompt_set[1] if len(prompt_set) > 1 else ""
        self.system_prompt_edit.setPlainText(system_prompt_text)

        # Safely get and set parameters
        # Temperature (index 2)
        temperature_value = (
            float(prompt_set[2]) if len(prompt_set) > 2 else 0.7
        )  # Default to 0.7
        self.temp_slider.setValue(int(temperature_value * 100))
        self.temp_spinbox.setValue(temperature_value)

        # P-value (index 3)
        p_value = float(prompt_set[3]) if len(prompt_set) > 3 else 0.1  # Default to 0.1
        self.p_value_slider.setValue(int(p_value * 100))
        self.p_value_spinbox.setValue(p_value)

        # Seed (index 5)
        seed_value = int(prompt_set[5]) if len(prompt_set) > 5 else 0  # Default to 0
        self.seed_spinbox.setValue(seed_value)

        # Model (index 4)
        if len(prompt_set) > 4 and isinstance(prompt_set[4], list) and prompt_set[4]:
            self.chosen_model = prompt_set[4][0]
            model_index = self.model_combo.findText(self.chosen_model)
            if model_index >= 0:
                self.model_combo.setCurrentIndex(model_index)

    def set_model(self, model_name: str):
        self.chosen_model = model_name

    def start_analysis(self):
        """Start analysis with PipelineManager integration - Claude Generated"""
        abstract_text = self.abstract_edit.toPlainText().strip()
        keywords_text = self.keywords_edit.toPlainText().strip()

        if not abstract_text:
            QMessageBox.warning(
                self, "Fehlende Eingabe", "Bitte geben Sie einen Text ein."
            )
            return

        # Set analysis running state
        self.is_analysis_running = True

        # Update UI for analysis state
        self.analyze_button.setVisible(False)
        self.cancel_button.setVisible(True)
        self.status_label.setText("Analyse läuft...")
        self.status_label.setStyleSheet("QLabel { color: #ff9800; font-weight: bold; }")
        self.progress_bar.setVisible(True)
        self.results_edit.clear()

        # Auto-hide input area if checkbox is checked
        if self.auto_hide_checkbox.isChecked():
            self.hide_input_during_streaming()

        # Add analysis start message to results
        self.results_edit.setPlainText("🔄 Analyse gestartet...\n\n")

        # 1. Create ad-hoc PipelineConfig for single step execution - Claude Generated
        adhoc_config = PipelineConfig()
        adhoc_config.auto_advance = False  # Important: We want only one step

        # 2. Create step configuration for the chosen task - Claude Generated
        step_config = PipelineStepConfig(
            step_id=self.task,
            provider=self.provider_combo.currentText(),
            model=self.model_combo.currentText(),
            task=self.task,
            temperature=self.temp_spinbox.value(),
            top_p=self.p_value_spinbox.value(),
            custom_params={
                'prompt_template': self.prompt_edit.toPlainText().strip(),
                'system_prompt': self.system_prompt_edit.toPlainText().strip(),
                'use_chunking_abstract': self.enable_chunk_abstract.isChecked(),
                'abstract_chunk_size': self.abstract_chunk_slider.value(),
                'use_chunking_keywords': self.enable_chunk_keywords.isChecked(),
                'keyword_chunk_size': self.keyword_chunk_slider.value(),
                'seed': self.seed_spinbox.value(),
            }
        )

        # 3. Set the ad-hoc configuration in PipelineConfig - Claude Generated
        adhoc_config.step_configs[self.task] = step_config

        # 4. Create input text with keywords if provided - Claude Generated
        input_text = abstract_text
        if keywords_text.strip():
            input_text = f"{abstract_text}\n\nExisting Keywords: {keywords_text}"

        # 5. Use SingleStepWorker for single step execution - Claude Generated
        from .workers import SingleStepWorker

        self.step_worker = SingleStepWorker(
            pipeline_manager=self.pipeline_manager,
            step_config=adhoc_config,
            input_data=input_text
        )

        # 6. Connect callbacks to handle PipelineStep objects - Claude Generated
        self.step_worker.step_completed.connect(self.on_analysis_completed)
        self.step_worker.step_error.connect(self.on_analysis_error)
        self.step_worker.stream_token.connect(self._update_results_text)

        # Update status when analysis actually starts
        self.status_label.setText("Verbindung zu LLM...")

        self.step_worker.start()

    def on_analysis_completed(self, step: PipelineStep):
        """Handle analysis completion with PipelineStep integration - Claude Generated"""

        # DEBUG: Log step.output_data structure - Claude Generated
        self.logger.info("=" * 50)
        self.logger.info("🔍 DEBUG: on_analysis_completed called")
        self.logger.info(f"step.output_data type: {type(step.output_data)}")
        self.logger.info(f"step.output_data is None: {step.output_data is None}")
        if step.output_data:
            self.logger.info(f"step.output_data attributes: {dir(step.output_data)}")
            if hasattr(step.output_data, '__dict__'):
                self.logger.info(f"step.output_data.__dict__: {step.output_data.__dict__}")
            if isinstance(step.output_data, dict):
                self.logger.info(f"step.output_data keys: {list(step.output_data.keys())}")

        # Get currently displayed text (from streaming) - Claude Generated
        current_results_text = self.results_edit.toPlainText()
        self.logger.info(f"📝 Current results_edit text length: {len(current_results_text)}")
        self.logger.info("=" * 50)

        # Reset analysis state
        self.is_analysis_running = False

        # Update UI for completion
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.cancel_button.setVisible(False)
        self.status_label.setText("Analyse abgeschlossen ✓")
        self.status_label.setStyleSheet("QLabel { color: #4caf50; font-weight: bold; }")

        # Restore input area if it was auto-hidden during streaming
        self.restore_input_after_streaming()

        # Extract data from PipelineStep output_data - Claude Generated
        if step.output_data and hasattr(step.output_data, 'analysis_result'):
            result = step.output_data.analysis_result

            # Extract keywords list
            keywords_only = ", ".join(result.matched_keywords.keys())
            keywords_list = list(result.matched_keywords.keys())

            # Create state object for history - Claude Generated
            from ..core.data_models import LlmKeywordAnalysis
            minimal_llm_analysis = LlmKeywordAnalysis(
                task_name=self.task or "analysis",
                model_used=self.model_combo.currentText(),
                provider_used=self.provider_combo.currentText(),
                prompt_template=self.prompt_edit.toPlainText().strip(),
                filled_prompt="",
                temperature=self.temp_spinbox.value(),
                seed=self.seed_spinbox.value(),
                response_full_text=result.full_text,
                extracted_gnd_keywords=keywords_list
            )
            state_object = self._create_state_from_current_analysis(llm_analysis=minimal_llm_analysis, keywords_list=keywords_list)

            # Add result to history with full state - Claude Generated
            self.add_result_to_history(result.full_text, self.task or "Analysis", state_object=state_object)
            self.results_edit.setPlainText(result.full_text)

            self.final_list.emit(keywords_only)
            self.gnd_systematic.emit(result.gnd_systematic)

            # Send analysis data to AnalysisReviewTab - Claude Generated
            current_abstract = self.abstract_edit.toPlainText().strip()
            self.analysis_completed.emit(current_abstract, keywords_only, result.full_text)

        # Handle dict-based output_data with llm_analysis (from pipeline keywords step) - Claude Generated
        elif step.output_data and isinstance(step.output_data, dict) and 'llm_analysis' in step.output_data:
            llm_analysis = step.output_data['llm_analysis']

            # Create state object for history - Claude Generated
            keywords_list = llm_analysis.extracted_gnd_keywords if llm_analysis.extracted_gnd_keywords else []
            state_object = self._create_state_from_current_analysis(llm_analysis=llm_analysis, keywords_list=keywords_list)

            # Display the full LLM response (already streamed, just keep it visible)
            self.add_result_to_history(llm_analysis.response_full_text, self.task or "Analysis", state_object=state_object)
            # DON'T overwrite - the text is already there from streaming
            # self.results_edit.setPlainText(llm_analysis.response_full_text)

            # Extract keywords
            keywords_only = ", ".join(llm_analysis.extracted_gnd_keywords) if llm_analysis.extracted_gnd_keywords else ""
            if keywords_only:
                self.final_list.emit(keywords_only)

            # Send data to AnalysisReviewTab
            current_abstract = self.abstract_edit.toPlainText().strip()
            self.analysis_completed.emit(current_abstract, keywords_only, llm_analysis.response_full_text)

        else:
            # Fallback: step.output_data is None, but text was streamed - Claude Generated
            # Get the text that was streamed to results_edit instead
            output_text = current_results_text if current_results_text else "No output data"

            self.logger.info(f"🔧 Fallback: Using streamed text from results_edit ({len(output_text)} chars)")

            # Create minimal LLM analysis object for fallback case - Claude Generated
            from ..core.data_models import LlmKeywordAnalysis
            minimal_llm_analysis = LlmKeywordAnalysis(
                task_name=self.task or "analysis",
                model_used=self.model_combo.currentText(),
                provider_used=self.provider_combo.currentText(),
                prompt_template=self.prompt_edit.toPlainText().strip(),
                filled_prompt="",
                temperature=self.temp_spinbox.value(),
                seed=self.seed_spinbox.value(),
                response_full_text=output_text,
                extracted_gnd_keywords=[]
            )

            # Create state object WITH llm_analysis - Claude Generated
            state_object = self._create_state_from_current_analysis(llm_analysis=minimal_llm_analysis)

            self.add_result_to_history(output_text, self.task or "Analysis", state_object=state_object)
            # DON'T overwrite results_edit - text is already there from streaming - Claude Generated
            # self.results_edit.setPlainText(output_text)

            # Send basic data to AnalysisReviewTab
            current_abstract = self.abstract_edit.toPlainText().strip()
            self.analysis_completed.emit(current_abstract, "", output_text)

        # Reset status after 3 seconds
        QApplication.instance().processEvents()
        import threading

        def reset_status():
            time.sleep(3)
            if not self.is_analysis_running:  # Only reset if no new analysis started
                self.status_label.setText("Bereit")
                self.status_label.setStyleSheet(
                    "QLabel { color: #4caf50; font-weight: bold; }"
                )

        threading.Thread(target=reset_status, daemon=True).start()

    def on_analysis_error(self, error_message: str):
        """Handle analysis error - Claude Generated"""
        # Reset analysis state
        self.is_analysis_running = False

        # Update UI for error state
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.cancel_button.setVisible(False)
        self.status_label.setText("Fehler ✗")
        self.status_label.setStyleSheet("QLabel { color: #d32f2f; font-weight: bold; }")

        # Restore input area if it was auto-hidden during streaming
        self.restore_input_after_streaming()

        # Show error in results area
        error_details = f"❌ Fehler bei der Analyse:\n{error_message}"
        self.results_edit.setPlainText(error_details)

        # Show error dialog
        QMessageBox.critical(self, "Analyse-Fehler", error_message)

        # Reset status after 5 seconds
        import threading

        def reset_status():
            time.sleep(5)
            if not self.is_analysis_running:
                self.status_label.setText("Bereit")
                self.status_label.setStyleSheet(
                    "QLabel { color: #4caf50; font-weight: bold; }"
                )

        threading.Thread(target=reset_status, daemon=True).start()

    def import_pdf(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "PDF importieren", "", "PDF Files (*.pdf)"
        )
        if file_name:
            try:
                with open(file_name, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "".join(page.extract_text() for page in reader.pages)
                    self.abstract_edit.setPlainText(text)
            except Exception as e:
                QMessageBox.critical(self, "Fehler beim PDF-Import", str(e))

    def set_abstract(self, abstract: str):
        """Sets the abstract text in the abstract_edit QTextEdit."""
        self.abstract_edit.setPlainText(abstract)

    def set_keywords(self, keywords: str):
        """Sets the keywords text in the keywords_edit QTextEdit."""
        self.keywords_edit.setPlainText(keywords)

    def display_llm_response(self, response_text: str):
        """Display LLM response in results area - Claude Generated"""
        self.results_edit.setPlainText(response_text)

    def _update_results_text(self, text_chunk: str, step_id: str = None):
        """Appends text chunks to the results_edit QTextEdit with enhanced feedback - Claude Generated"""
        # Update status to show streaming is active
        if self.is_analysis_running:
            self.status_label.setText("Streaming...")
            self.status_label.setStyleSheet(
                "QLabel { color: #2196f3; font-weight: bold; }"
            )

        # Clear the initial "Analysis started" message on first token
        current_text = self.results_edit.toPlainText()
        if current_text.startswith("🔄 Analyse gestartet..."):
            self.results_edit.clear()

        # Append new text
        self.results_edit.insertPlainText(text_chunk)

        # Auto-scroll to bottom to follow streaming text
        cursor = self.results_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.results_edit.setTextCursor(cursor)
        self.results_edit.ensureCursorVisible()

    def set_models_and_providers(
        self, models: Dict[str, List[str]], providers: List[str]
    ):
        """Sets the available models and providers."""
        self.available_models = models
        self.provider_combo.clear()
        self.provider_combo.addItems(providers)
        self.update_models(self.provider_combo.currentText())

    def cancel_analysis(self):
        """Cancel the running analysis - Claude Generated"""
        if hasattr(self, "step_worker") and self.step_worker.isRunning():
            self.step_worker.terminate()
            self.step_worker.wait()

        # Reset UI state
        self.is_analysis_running = False
        self.progress_bar.setVisible(False)
        self.analyze_button.setVisible(True)
        self.cancel_button.setVisible(False)
        self.status_label.setText("Abgebrochen")
        self.status_label.setStyleSheet("QLabel { color: #ff9800; font-weight: bold; }")

        # Restore input area if it was auto-hidden during streaming
        self.restore_input_after_streaming()

        # Add cancellation message to results
        self.results_edit.setPlainText("⏹️ Analyse abgebrochen")

        # Reset status after 3 seconds
        import threading

        def reset_status():
            time.sleep(3)
            if not self.is_analysis_running:
                self.status_label.setText("Bereit")
                self.status_label.setStyleSheet(
                    "QLabel { color: #4caf50; font-weight: bold; }"
                )

        threading.Thread(target=reset_status, daemon=True).start()

    def toggle_input_visibility(self):
        """Toggle input widget visibility - Claude Generated"""
        if self.input_widget_visible:
            # Hide input widget completely
            self.input_widget.setVisible(False)
            self.toggle_input_button.setText("Eingabe einblenden")
            self.input_widget_visible = False
            # Give all space to results (widget is hidden, so splitter adjusts automatically)
        else:
            # Show input widget
            self.input_widget.setVisible(True)
            self.toggle_input_button.setText("Eingabe ausblenden")
            self.input_widget_visible = True
            # Restore balanced view
            self.main_splitter.setSizes([300, 700])

    def hide_input_during_streaming(self):
        """Hide input completely during streaming - Claude Generated"""
        if self.input_widget_visible:
            self.input_widget.setVisible(False)
            self.toggle_input_button.setText("Eingabe einblenden")
            # Store original state to restore later, but don't change input_widget_visible
            # so user can still manually toggle
            self.input_was_visible_before_streaming = True
        else:
            self.input_was_visible_before_streaming = False

    def restore_input_after_streaming(self):
        """Restore input visibility after streaming if auto-hide was used - Claude Generated"""
        if (
            hasattr(self, "input_was_visible_before_streaming")
            and self.input_was_visible_before_streaming
        ):
            if self.auto_hide_checkbox.isChecked():
                # Restore input widget
                self.input_widget.setVisible(True)
                self.toggle_input_button.setText("Eingabe ausblenden")
                self.input_widget_visible = True
                self.main_splitter.setSizes([300, 700])

    def auto_adjust_layout_for_streaming(self):
        """Automatically adjust layout for optimal streaming view - Claude Generated"""
        if self.input_widget_visible:
            # Minimize input area when streaming starts (fallback if not hidden completely)
            self.main_splitter.setSizes([150, 850])
