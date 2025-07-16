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
from ..core.data_models import AbstractData, AnalysisResult

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


class AnalysisWorker(QThread):
    finished = pyqtSignal(AnalysisResult)
    error = pyqtSignal(str)
    new_token = pyqtSignal(str)

    def __init__(self, alima_manager: AlimaManager, abstract_data: AbstractData, task: str, model: str,
                 use_chunking_abstract: bool, abstract_chunk_size: int,
                 use_chunking_keywords: bool, keyword_chunk_size: int,
                 prompt_template: Optional[str] = None, temperature: float = 0.7, p_value: float = 0.1, seed: int = 0, system_prompt: Optional[str] = ""):
        super().__init__()
        self.alima_manager = alima_manager
        self.abstract_data = abstract_data
        self.task = task
        self.model = model
        self.use_chunking_abstract = use_chunking_abstract
        self.abstract_chunk_size = abstract_chunk_size
        self.use_chunking_keywords = use_chunking_keywords
        self.keyword_chunk_size = keyword_chunk_size
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.p_value = p_value
        self.seed = seed
        self.system_prompt = system_prompt

    def run(self):
        try:
            def stream_callback(token):
                self.new_token.emit(token)

            result = self.alima_manager.analyze_abstract(
                self.abstract_data,
                self.task,
                self.model,
                self.use_chunking_abstract,
                self.abstract_chunk_size,
                self.use_chunking_keywords,
                self.keyword_chunk_size,
                prompt_template=self.prompt_template,
                stream_callback=stream_callback,
                temperature=self.temperature,
                p_value=self.p_value,
                seed=self.seed,
                system=self.system_prompt
            )
            self.finished.emit(result.analysis_result)
        except Exception as e:
            self.error.emit(str(e))


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

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService,
        main_window: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        # Inject dependencies
        self.alima_manager = alima_manager
        self.llm = llm_service # Renamed from self.llm_service to self.llm for consistency with existing code
        self.prompt_manager = self.alima_manager.prompt_service # Access prompt_service via alima_manager
        self.main_window = main_window

        self.need_keywords = False
        self.logger = logging.getLogger(__name__)
        self.task: Optional[str] = None
        self.chosen_model: str = "default"
        self.available_models: Dict[str, List[str]] = {}

        self.llm.ollama_url_updated.connect(self.on_ollama_url_updated)
        self.llm.ollama_port_updated.connect(self.on_ollama_port_updated)

        # Set up the UI
        self.setup_ui()

    # ======== UNCHANGED METHODS - Keep existing functionality ========
    def on_ollama_url_updated(self):
        """Handle Ollama URL update."""
        if self.main_window:
            self.main_window.load_models_and_providers()

    def on_ollama_port_updated(self):
        """Handle Ollama Port update."""
        if self.main_window:
            self.main_window.load_models_and_providers()

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

        self.pdf_button.setVisible(self.task == "abstract")

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
        input_layout.addWidget(self.abstract_edit)

        input_layout.addWidget(QLabel("Vorhandene Keywords (optional):"))
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setMaximumHeight(80)
        input_layout.addWidget(self.keywords_edit)

        input_config_splitter.addWidget(self.input_group)

        # RIGHT SIDE: AI Configuration and Chunking
        config_widget = QWidget()
        config_main_layout = QVBoxLayout(config_widget)

        self.config_group = QGroupBox("KI-Konfiguration")
        config_layout = QVBoxLayout(self.config_group)

        # Create Tab Widget
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
        self.system_prompt_edit.setMinimumHeight(80)
        prompt_layout.addWidget(self.system_prompt_edit)
        
        self.config_tabs.addTab(prompt_tab, "Prompt")

        # -- Parameters Tab --
        params_tab = QWidget()
        params_layout = QGridLayout(params_tab)

        # Temperature
        params_layout.addWidget(QLabel("Temperatur:"), 0, 0)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_spinbox.setValue(v / 100.0))
        params_layout.addWidget(self.temp_slider, 0, 1)
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setDecimals(2)
        self.temp_spinbox.setRange(0.0, 1.0)
        self.temp_spinbox.setSingleStep(0.01)
        self.temp_spinbox.valueChanged.connect(lambda v: self.temp_slider.setValue(int(v * 100)))
        params_layout.addWidget(self.temp_spinbox, 0, 2)

        # Top-P
        params_layout.addWidget(QLabel("Top-P:"), 1, 0)
        self.p_value_slider = QSlider(Qt.Orientation.Horizontal)
        self.p_value_slider.setRange(0, 100)
        self.p_value_slider.valueChanged.connect(lambda v: self.p_value_spinbox.setValue(v / 100.0))
        params_layout.addWidget(self.p_value_slider, 1, 1)
        self.p_value_spinbox = QDoubleSpinBox()
        self.p_value_spinbox.setDecimals(2)
        self.p_value_spinbox.setRange(0.0, 1.0)
        self.p_value_spinbox.setSingleStep(0.01)
        self.p_value_spinbox.valueChanged.connect(lambda v: self.p_value_slider.setValue(int(v * 100)))
        params_layout.addWidget(self.p_value_spinbox, 1, 2)

        # Seed
        params_layout.addWidget(QLabel("Seed:"), 2, 0)
        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 999999999)
        self.seed_spinbox.setValue(0)
        params_layout.addWidget(self.seed_spinbox, 2, 1, 1, 2)

        self.config_tabs.addTab(params_tab, "Parameter")
        
        config_layout.addWidget(self.config_tabs)

        # Provider/Model Selection
        provider_model_group = QGroupBox("Provider & Modell")
        provider_model_layout = QGridLayout(provider_model_group)
        provider_model_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        provider_model_layout.addWidget(self.provider_combo, 0, 1)
        provider_model_layout.addWidget(QLabel("Modell:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.set_model)
        provider_model_layout.addWidget(self.model_combo, 1, 1)
        config_layout.addWidget(provider_model_group)

        config_main_layout.addWidget(self.config_group)

        chunk_group = QGroupBox("Chunking-Kontrolle (optional)")
        chunk_layout = QVBoxLayout(chunk_group)
        self.enable_chunk_abstract = QCheckBox("Abstract-Chunking")
        self.abstract_chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self.abstract_chunk_slider.setEnabled(False)
        self.enable_chunk_abstract.toggled.connect(self.abstract_chunk_slider.setEnabled)
        chunk_layout.addWidget(self.enable_chunk_abstract)
        chunk_layout.addWidget(self.abstract_chunk_slider)

        self.enable_chunk_keywords = QCheckBox("Keywords-Chunking")
        self.keyword_chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self.keyword_chunk_slider.setEnabled(False)
        self.enable_chunk_keywords.toggled.connect(self.keyword_chunk_slider.setEnabled)
        chunk_layout.addWidget(self.enable_chunk_keywords)
        chunk_layout.addWidget(self.keyword_chunk_slider)
        config_main_layout.addWidget(chunk_group)

        input_config_splitter.addWidget(config_widget)
        main_layout.addWidget(input_config_splitter)

        # ======== Button Area ========
        button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyse starten")
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)
        main_layout.addLayout(button_layout)

        # ======== Results Area ========
        self.results_group = QGroupBox("Analyseergebnis")
        results_layout = QVBoxLayout(self.results_group)
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        results_layout.addWidget(self.results_edit)
        main_layout.addWidget(self.results_group)

        self.update_models(self.provider_combo.currentText())
        self.populate_task_selector()

    def populate_task_selector(self):
        self.task_selector_combo.clear()
        tasks = self.prompt_manager.get_available_tasks()
        self.task_selector_combo.addItems(tasks)

    def on_task_selected(self, index):
        if index < 0: return
        self.task = self.task_selector_combo.currentText()
        self.populate_prompt_selector()

    def populate_prompt_selector(self):
        self.prompt_selector_combo.clear()
        if not self.task: return
        prompts = self.prompt_manager.get_prompts_for_task(self.task)
        for i, prompt_set in enumerate(prompts):
            self.prompt_selector_combo.addItem(f"Prompt Set {i+1}", userData=i)
        if self.prompt_selector_combo.count() > 0:
            self.prompt_selector_combo.setCurrentIndex(0)
            self.on_prompt_selected(0)

    def on_prompt_selected(self, index):
        if index < 0: return
        prompt_set_index = self.prompt_selector_combo.itemData(index)
        if prompt_set_index is None: return
        prompt_set = self.prompt_manager.get_prompts_for_task(self.task)[prompt_set_index]
        
        # Safely get prompt template (index 0)
        prompt_template_text = prompt_set[0] if len(prompt_set) > 0 else ""
        self.prompt_edit.setPlainText(prompt_template_text)

        # Safely get system prompt (index 1)
        system_prompt_text = prompt_set[1] if len(prompt_set) > 1 else ""
        self.system_prompt_edit.setPlainText(system_prompt_text)

        # Safely get and set parameters
        # Temperature (index 2)
        temperature_value = float(prompt_set[2]) if len(prompt_set) > 2 else 0.7 # Default to 0.7
        self.temp_slider.setValue(int(temperature_value * 100))
        self.temp_spinbox.setValue(temperature_value)

        # P-value (index 3)
        p_value = float(prompt_set[3]) if len(prompt_set) > 3 else 0.1 # Default to 0.1
        self.p_value_slider.setValue(int(p_value * 100))
        self.p_value_spinbox.setValue(p_value)

        # Seed (index 5)
        seed_value = int(prompt_set[5]) if len(prompt_set) > 5 else 0 # Default to 0
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
        abstract_text = self.abstract_edit.toPlainText().strip()
        keywords_text = self.keywords_edit.toPlainText().strip()

        if not abstract_text:
            QMessageBox.warning(self, "Fehlende Eingabe", "Bitte geben Sie einen Text ein.")
            return

        self.progress_bar.setVisible(True)
        self.results_edit.clear()

        abstract_data = AbstractData(abstract=abstract_text, keywords=keywords_text)
        use_chunking_abstract = self.enable_chunk_abstract.isChecked()
        abstract_chunk_size = self.abstract_chunk_slider.value()
        use_chunking_keywords = self.enable_chunk_keywords.isChecked()
        keyword_chunk_size = self.keyword_chunk_slider.value()
        use_chunking_abstract = self.enable_chunk_abstract.isChecked()
        abstract_chunk_size = self.abstract_chunk_slider.value()
        use_chunking_keywords = self.enable_chunk_keywords.isChecked()
        keyword_chunk_size = self.keyword_chunk_slider.value()

        prompt_template = self.prompt_edit.toPlainText().strip()
        system_prompt = self.system_prompt_edit.toPlainText().strip()
        temperature = self.temp_spinbox.value()
        p_value = self.p_value_spinbox.value()
        seed = self.seed_spinbox.value()

        self.analysis_worker = AnalysisWorker(
            self.alima_manager, abstract_data, self.task, self.chosen_model,
            use_chunking_abstract, abstract_chunk_size, use_chunking_keywords, keyword_chunk_size,
            prompt_template=prompt_template, temperature=temperature, p_value=p_value, seed=seed, system_prompt=system_prompt
        )
        self.analysis_worker.new_token.connect(self._update_results_text)
        self.analysis_worker.finished.connect(self.on_analysis_completed)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.start()

    def on_analysis_completed(self, result: AnalysisResult):
        self.progress_bar.setVisible(False)
        self.results_edit.setPlainText(result.full_text)
        self.final_list.emit(str(result.matched_keywords))
        self.gnd_systematic.emit(result.gnd_systematic)

    def on_analysis_error(self, error_message: str):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analyse-Fehler", error_message)

    def import_pdf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "PDF importieren", "", "PDF Files (*.pdf)")
        if file_name:
            try:
                with open(file_name, 'rb') as f:
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

    def _update_results_text(self, text_chunk: str):
        """Appends text chunks to the results_edit QTextEdit."""
        self.results_edit.insertPlainText(text_chunk)

    def set_models_and_providers(self, models: Dict[str, List[str]], providers: List[str]):
        """Sets the available models and providers."""
        self.available_models = models
        self.provider_combo.clear()
        self.provider_combo.addItems(providers)
        self.update_models(self.provider_combo.currentText())

    

    

    

    
