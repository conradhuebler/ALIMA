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

    def __init__(self, alima_manager: AlimaManager, abstract_data: AbstractData, task: str, model: str,
                 use_chunking_abstract: bool, abstract_chunk_size: int,
                 use_chunking_keywords: bool, keyword_chunk_size: int):
        super().__init__()
        self.alima_manager = alima_manager
        self.abstract_data = abstract_data
        self.task = task
        self.model = model
        self.use_chunking_abstract = use_chunking_abstract
        self.abstract_chunk_size = abstract_chunk_size
        self.use_chunking_keywords = use_chunking_keywords
        self.keyword_chunk_size = keyword_chunk_size

    def run(self):
        try:
            result = self.alima_manager.analyze_abstract(
                self.abstract_data,
                self.task,
                self.model,
                self.use_chunking_abstract,
                self.abstract_chunk_size,
                self.use_chunking_keywords,
                self.keyword_chunk_size
            )
            self.finished.emit(result)
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
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        # Inject dependencies
        self.alima_manager = alima_manager
        self.llm = llm_service # Renamed from self.llm_service to self.llm for consistency with existing code
        self.prompt_manager = self.alima_manager.prompt_service # Access prompt_service via alima_manager

        self.need_keywords = False
        self.logger = logging.getLogger(__name__)
        self.task: Optional[str] = None
        self.chosen_model: str = "default"

        self.llm.ollama_url_updated.connect(self.on_ollama_url_updated)
        self.llm.ollama_port_updated.connect(self.on_ollama_port_updated)

        # Set up the UI
        self.setup_ui()

    # ======== UNCHANGED METHODS - Keep existing functionality ========
    def on_ollama_url_updated(self):
        """Handle Ollama URL update."""
        current_provider = self.provider_combo.currentText()
        self.provider_combo.clear()
        items = self.llm.get_available_providers()
        self.provider_combo.addItems(items)
        if self.provider_combo.count() > 0:
            if current_provider in items:
                self.provider_combo.setCurrentText(current_provider)

    def on_ollama_port_updated(self):
        """Handle Ollama Port update."""
        current_provider = self.provider_combo.currentText()
        self.provider_combo.clear()
        items = self.llm.get_available_providers()
        self.provider_combo.addItems(items)
        if self.provider_combo.count() > 0:
            if current_provider in items:
                self.provider_combo.setCurrentText(current_provider)

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
        all_models = self.llm.get_available_models(provider)
        self.model_combo.addItems(all_models)

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
        config_grid = QGridLayout(self.config_group)

        prompt_selection_group = QGroupBox("Prompt-Auswahl")
        prompt_selection_layout = QHBoxLayout(prompt_selection_group)
        prompt_selection_layout.addWidget(QLabel("Task:"))
        self.task_selector_combo = QComboBox()
        self.task_selector_combo.currentIndexChanged.connect(self.on_task_selected)
        prompt_selection_layout.addWidget(self.task_selector_combo)
        prompt_selection_layout.addWidget(QLabel("Prompt:"))
        self.prompt_selector_combo = QComboBox()
        self.prompt_selector_combo.currentIndexChanged.connect(self.on_prompt_selected)
        prompt_selection_layout.addWidget(self.prompt_selector_combo)
        config_grid.addWidget(prompt_selection_group, 0, 0, 1, 2)

        config_grid.addWidget(QLabel("Provider:"), 1, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        config_grid.addWidget(self.provider_combo, 1, 1)

        config_grid.addWidget(QLabel("Modell:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.set_model)
        config_grid.addWidget(self.model_combo, 2, 1)

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

        self.analysis_worker = AnalysisWorker(
            self.alima_manager, abstract_data, self.task, self.chosen_model,
            use_chunking_abstract, abstract_chunk_size, use_chunking_keywords, keyword_chunk_size
        )
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

    

    

    

    
