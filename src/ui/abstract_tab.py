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

from ..llm.llm_interface import LLMInterface
from ..utils.config import Config, ConfigSection, AIConfig
from ..llm.prompt_manager import PromptManager

from pathlib import Path
import os
import json
import logging
import re
import tempfile
import PyPDF2
import threading
import time
from typing import List, Tuple, Dict
import uuid


class KeywordHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for predefined keywords and GND numbers."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Define the highlighting format
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setFontWeight(QFont.Weight.Bold)

        # Initialize empty keywords dictionary
        self.keywords_gnd = {}
        self.patterns = []

    def _build_patterns(self):
        """Build regex patterns from keywords and GND numbers."""
        self.patterns = []

        if not self.keywords_gnd:
            return

        # Add keyword patterns (case-insensitive word boundaries)
        for keyword in self.keywords_gnd.keys():
            pattern = QRegularExpression(
                f"\\b{QRegularExpression.escape(keyword)}\\b",
                QRegularExpression.PatternOption.CaseInsensitiveOption,
            )
            self.patterns.append(pattern)

        # Add GND number patterns (exact matches in parentheses)
        for gnd_number in self.keywords_gnd.values():
            pattern = QRegularExpression(
                f"\\({QRegularExpression.escape(gnd_number)}\\)"
            )
            self.patterns.append(pattern)

    def highlightBlock(self, text):
        """Apply highlighting to a block of text."""
        for pattern in self.patterns:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(
                    match.capturedStart(), match.capturedLength(), self.keyword_format
                )

    def set_keywords_from_database(self, keywords_dict):
        """Update keywords from database dictionary."""
        self.keywords_gnd = keywords_dict.copy()
        self._build_patterns()
        self.rehighlight()

    def clear_keywords(self):
        """Clear all keywords."""
        self.keywords_gnd.clear()
        self.patterns.clear()
        self.rehighlight()


class ChunkResultItem:
    """Data structure for chunk results"""

    def __init__(
        self,
        chunk_id: int,
        prompt: str,
        result: str,
        chunk_type: str,
        chunk_info: str = "",
    ):
        self.chunk_id = chunk_id
        self.prompt = prompt
        self.result = result
        self.chunk_type = chunk_type  # "single", "abstract", "keywords", "combined"
        self.chunk_info = (
            chunk_info  # Additional info like "Lines 1-10" or "Keywords 1-20"
        )


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
        parent=None,
        recommendations_file: Path = Path(__file__).parent.parent.parent
        / "model_recommendations.json",
        llm: LLMInterface = None,
    ):
        super().__init__(parent)
        # Initialize components
        self.llm = llm
        self.need_keywords = False
        self.logger = logging.getLogger(__name__)
        self.template_name = ""
        self.recommended_models = []
        self.current_template = ""
        self.model_descriptions = {}
        self.recommendations_file = recommendations_file
        self.propmpt_file = Path(__file__).parent.parent.parent / "prompts.json"
        self.promptmanager = PromptManager(self.propmpt_file)
        self.task = ""
        self.tmp_task = ""
        self.system = ""
        self.generated_response = ""
        self.pdf_metadata = {}
        self.pdf_path = None
        self.chosen_model = "default"
        self.matched_keywords = {}
        self.current_request_id = None

        # Load configurations
        self.load_recommendations()
        self.set_model_recommendations("default")
        self.required = []

        # New chunking workflow attributes
        self.chunk_results: List[ChunkResultItem] = []
        self.current_processing_chunks = []
        self.current_chunk_index = 0
        self.is_processing_chunks = False
        self.combined_result = ""

        # Legacy chunking attributes (keep for compatibility)
        self.total_chunks = 0

        # UI state
        self.is_compact_mode = False

        # Connect LLM signals
        self.llm.text_received.connect(self.on_text_received)
        self.llm.generation_finished.connect(self.on_generation_finished)
        self.llm.generation_error.connect(self.on_generation_error)
        self.llm.generation_cancelled.connect(self.on_generation_cancelled)
        self.llm.ollama_url_updated.connect(self.on_ollama_url_updated)
        self.llm.ollama_port_updated.connect(self.on_ollama_port_updated)

        self.keywords = ""
        self.abstract = ""

        # Set up the UI
        self.setup_ui()
        self.setup_animations()

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

    def load_recommendations(self):
        """Load model recommendations from the JSON file."""
        try:
            if not self.recommendations_file.exists():
                self.logger.warning(
                    f"Recommendations file not found: {self.recommendations_file}"
                )
                self.create_default_recommendations()
                return

            with open(self.recommendations_file, "r", encoding="utf-8") as f:
                self.recommendations = json.load(f)

            self.logger.info(
                f"Successfully loaded recommendations from {self.recommendations_file}"
            )

        except Exception as e:
            self.logger.error(f"Error loading recommendations: {e}")
            self.create_default_recommendations()

    def create_default_recommendations(self):
        """Create default recommendations if file is missing."""
        self.recommendations = {"default": {"recommended": {}, "descriptions": {}}}
        self.logger.info("Created default recommendations")

    def set_task(self, task: str):
        """Set the task type for model recommendations."""
        self.task = task
        self.recommended_models = self.promptmanager.get_available_models(task)
        self.logger.info(f"Available models for {task}: {self.recommended_models}")
        self.update_models(self.provider_combo.currentText())
        self.pdf_button.setVisible(self.task == "abstract")

    def set_model_recommendations(self, use_case: str):
        """Set model recommendations based on the use case."""
        pass  # Handled by prompt manager now

    def update_models(self, provider: str):
        """Update available models when provider changes."""
        self.model_combo.clear()
        all_models = self.llm.get_available_models(provider)
        recommended_available = []

        if self.recommended_models:
            recommended_group = "↳ Empfohlene Modelle"
            self.model_combo.addItem(recommended_group)
            recommended_available = [
                model for model in self.recommended_models if model in all_models
            ]

            for model in recommended_available:
                self.model_combo.addItem(f"  {model}")
                idx = self.model_combo.count() - 1
                description = self.model_descriptions.get(provider, {}).get(model, "")
                self.model_combo.setItemData(
                    idx, description, Qt.ItemDataRole.ToolTipRole
                )

            if recommended_available and len(all_models) > len(recommended_available):
                self.model_combo.addItem("↳ Weitere verfügbare Modelle")

            other_models = [
                model for model in all_models if model not in recommended_available
            ]
            for model in other_models:
                self.model_combo.addItem(f"  {model}")
        else:
            self.model_combo.addItems(all_models)

        if recommended_available:
            self.model_combo.setCurrentText(f"  {recommended_available[0]}")
            self.set_model(recommended_available[0])
        elif all_models:
            self.model_combo.setCurrentText(all_models[0])
            self.set_model(all_models[0])

    def setup_ui(self):
        """Set up the user interface with restructured layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Color definitions
        primary_color = "#4a86e8"
        secondary_color = "#6aa84f"
        accent_color = "#f1c232"
        bg_light = "#f8f9fa"
        text_color = "#333333"

        # Global styles
        self.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-top: 12px;
                background-color: {bg_light};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {bg_light};
            }}
            QPushButton {{
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                background-color: {primary_color};
                color: white;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #3a76d8; }}
            QPushButton:pressed {{ background-color: #2a66c8; }}
            QLabel {{ color: {text_color}; }}
            QComboBox {{
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }}
            QTextEdit {{
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                font-size: 11pt;
            }}
        """
        )

        # ======== Control Bar ========
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(0, 0, 0, 5)

        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setFixedWidth(120)
        self.cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #cccccc;
                color: #666666;
            }
        """
        )
        self.cancel_button.clicked.connect(self.cancel_analysis)
        control_bar.addWidget(self.cancel_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Generiere Antwort... %p%")
        control_bar.addWidget(self.progress_bar)

        self.toggle_sections_btn = QPushButton("▼ Eingabebereich")
        self.toggle_sections_btn.setFixedWidth(140)
        self.toggle_sections_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f0f0f0; 
                color: #555;
                font-weight: normal;
                border: 1px solid #ddd;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """
        )
        self.toggle_sections_btn.clicked.connect(self.toggle_compact_mode)
        control_bar.addWidget(self.toggle_sections_btn)

        main_layout.addLayout(control_bar)

        # ======== Collapsible Container ========
        self.collapsible_container = QWidget()
        collapsible_layout = QVBoxLayout(self.collapsible_container)
        collapsible_layout.setContentsMargins(0, 0, 0, 0)
        collapsible_layout.setSpacing(12)

        # ======== NEW LAYOUT: Input and Config Side by Side ========
        input_config_splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT SIDE: Input Group (Abstract/Keywords)
        self.input_group = QGroupBox("Eingabe")
        input_layout = QVBoxLayout(self.input_group)
        input_layout.setSpacing(8)
        input_layout.setContentsMargins(10, 20, 10, 10)

        # Abstract section with PDF import
        header_layout = QHBoxLayout()
        abstract_label = QLabel("Abstract / Text:")
        abstract_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header_layout.addWidget(abstract_label)
        header_layout.addStretch(1)

        self.pdf_button = QPushButton("PDF importieren")
        self.pdf_button.setToolTip("Text aus einer PDF-Datei importieren")
        self.pdf_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {secondary_color};
                padding: 6px 12px;
            }}
            QPushButton:hover {{ background-color: #5a9840; }}
        """
        )
        self.pdf_button.setMaximumWidth(150)
        self.pdf_button.clicked.connect(self.import_pdf)
        header_layout.addWidget(self.pdf_button)

        input_layout.addLayout(header_layout)

        # PDF info frame
        self.pdf_info_frame = QFrame()
        self.pdf_info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.pdf_info_frame.setStyleSheet(
            "background-color: #e8f0fe; border-radius: 6px; padding: 8px;"
        )
        self.pdf_info_frame.setVisible(False)

        pdf_info_layout = QHBoxLayout(self.pdf_info_frame)
        self.pdf_metadata_label = QLabel()
        self.pdf_metadata_label.setWordWrap(True)
        pdf_info_layout.addWidget(self.pdf_metadata_label, 3)

        self.pdf_preview_label = QLabel()
        self.pdf_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pdf_preview_label.setMinimumSize(100, 100)
        self.pdf_preview_label.setMaximumSize(120, 160)
        pdf_info_layout.addWidget(self.pdf_preview_label, 0)

        self.close_pdf_button = QPushButton("×")
        self.close_pdf_button.setToolTip("PDF-Informationen ausblenden")
        self.close_pdf_button.setFixedSize(24, 24)
        self.close_pdf_button.setStyleSheet(
            """
            QPushButton {
                background-color: #dddddd;
                border-radius: 12px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #cccccc; }
        """
        )
        self.close_pdf_button.clicked.connect(
            lambda: self.pdf_info_frame.setVisible(False)
        )
        pdf_info_layout.addWidget(self.close_pdf_button, 0, Qt.AlignmentFlag.AlignTop)

        input_layout.addWidget(self.pdf_info_frame)

        # Abstract text field
        self.abstract_edit = QTextEdit()
        self.abstract_edit.setPlaceholderText(
            "Geben Sie hier den zu analysierenden Text ein oder importieren Sie eine PDF..."
        )
        self.abstract_edit.textChanged.connect(self.update_input)
        self.abstract_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.abstract_edit.setMinimumHeight(120)
        input_layout.addWidget(self.abstract_edit)

        # Keywords section
        keywords_header = QHBoxLayout()
        keywords_label = QLabel(
            "Vorhandene Keywords (optional):"
            if not self.need_keywords
            else "Es müssen zwingend OGND-Keywords angebeben werden:"
        )
        keywords_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        keywords_header.addWidget(keywords_label)

        self.extract_keywords_btn = QPushButton("Keywords aus PDF")
        self.extract_keywords_btn.setToolTip(
            "Keywords aus den PDF-Metadaten extrahieren"
        )
        self.extract_keywords_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {accent_color};
                color: #333;
                padding: 4px 8px;
            }}
            QPushButton:hover {{ background-color: #e1b222; }}
        """
        )
        self.extract_keywords_btn.setMaximumWidth(140)
        self.extract_keywords_btn.setVisible(False)
        self.extract_keywords_btn.clicked.connect(self.extract_keywords_from_pdf)
        keywords_header.addWidget(self.extract_keywords_btn)

        input_layout.addLayout(keywords_header)

        self.keywords_edit = QTextEdit()
        self.keywords_edit.setPlaceholderText(
            "Fügen Sie hier bereits vorhandene Keywords ein ..."
        )
        self.keywords_edit.textChanged.connect(self.update_input)
        self.keywords_edit.setMaximumHeight(80)
        input_layout.addWidget(self.keywords_edit)

        # Add input group to splitter
        input_config_splitter.addWidget(self.input_group)

        # RIGHT SIDE: AI Configuration and Chunking
        config_widget = QWidget()
        config_main_layout = QVBoxLayout(config_widget)
        config_main_layout.setContentsMargins(0, 0, 0, 0)

        # AI Configuration
        self.config_group = QGroupBox("KI-Konfiguration")
        config_grid = QGridLayout(self.config_group)
        config_grid.setSpacing(8)
        config_grid.setContentsMargins(10, 20, 10, 10)

        # Provider and Model
        config_grid.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        config_grid.addWidget(self.provider_combo, 0, 1)

        config_grid.addWidget(QLabel("Modell:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(self.set_model)
        config_grid.addWidget(self.model_combo, 1, 1)

        # Temperature
        self.temperature_label = QLabel("Temperatur: 0.00")
        config_grid.addWidget(self.temperature_label, 2, 0)

        self.ki_temperature = QSlider(Qt.Orientation.Horizontal)
        self.ki_temperature.setRange(0, 100)
        self.ki_temperature.setValue(0)
        self.ki_temperature.setTickInterval(10)
        self.ki_temperature.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ki_temperature.valueChanged.connect(self.update_temperature_label)
        config_grid.addWidget(self.ki_temperature, 2, 1)

        # Seed
        config_grid.addWidget(QLabel("Seed:"), 3, 0)
        self.ki_seed = QSpinBox()
        self.ki_seed.setRange(0, 1000000000)
        self.ki_seed.setValue(1)
        self.ki_seed.setToolTip(
            "0 = zufällig, andere Werte für reproduzierbare Ergebnisse"
        )
        config_grid.addWidget(self.ki_seed, 3, 1)

        config_main_layout.addWidget(self.config_group)

        # Chunking Configuration
        chunk_group = QGroupBox("Chunking-Kontrolle (optional)")
        chunk_layout = QVBoxLayout(chunk_group)

        # Abstract Chunking
        abstract_chunk_layout = QHBoxLayout()
        self.enable_chunk_abstract = QCheckBox("Abstract-Chunking")
        self.enable_chunk_abstract.setToolTip(
            "Teilt den Abstract in kleinere Abschnitte auf"
        )
        self.abstract_chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self.abstract_chunk_slider.setMinimum(1)
        self.abstract_chunk_slider.setMaximum(500)
        self.abstract_chunk_slider.setValue(100)
        self.abstract_chunk_slider.setEnabled(False)
        self.abstract_chunk_display = QLabel("100")
        self.abstract_chunk_display.setMinimumWidth(30)

        abstract_chunk_layout.addWidget(self.enable_chunk_abstract)
        abstract_chunk_layout.addWidget(QLabel("Zeilen:"))
        abstract_chunk_layout.addWidget(self.abstract_chunk_slider)
        abstract_chunk_layout.addWidget(self.abstract_chunk_display)
        chunk_layout.addLayout(abstract_chunk_layout)

        # Keywords Chunking
        keyword_chunk_layout = QHBoxLayout()
        self.enable_chunk_keywords = QCheckBox("Keywords-Chunking")
        self.enable_chunk_keywords.setToolTip(
            "Teilt die Keywords in kleinere Abschnitte auf"
        )
        self.keyword_chunk_slider = QSlider(Qt.Orientation.Horizontal)
        self.keyword_chunk_slider.setMinimum(1)
        self.keyword_chunk_slider.setMaximum(10000)
        self.keyword_chunk_slider.setValue(500)
        self.keyword_chunk_slider.setEnabled(False)
        self.keyword_chunk_display = QLabel("500")
        self.keyword_chunk_display.setMinimumWidth(30)

        keyword_chunk_layout.addWidget(self.enable_chunk_keywords)
        keyword_chunk_layout.addWidget(QLabel("Keywords:"))
        keyword_chunk_layout.addWidget(self.keyword_chunk_slider)
        keyword_chunk_layout.addWidget(self.keyword_chunk_display)
        chunk_layout.addLayout(keyword_chunk_layout)

        config_main_layout.addWidget(chunk_group)

        # Connect chunking signals
        self.enable_chunk_abstract.toggled.connect(self.on_chunking_enabled)
        self.enable_chunk_keywords.toggled.connect(self.on_chunking_enabled)
        self.abstract_chunk_slider.valueChanged.connect(self.on_abstract_chunk_changed)
        self.keyword_chunk_slider.valueChanged.connect(self.on_keyword_chunk_changed)

        # Add spacing
        config_main_layout.addStretch()

        # Add config widget to splitter
        input_config_splitter.addWidget(config_widget)

        # Set splitter proportions (input takes more space)
        input_config_splitter.setSizes([400, 300])

        collapsible_layout.addWidget(input_config_splitter)

        # ======== Prompt Display (unchanged) ========
        self.prompt_frame = QFrame()
        prompt_layout = QVBoxLayout(self.prompt_frame)
        prompt_layout.setSpacing(4)

        prompt_header = QHBoxLayout()
        self.show_prompt_btn = QPushButton("Prompt anzeigen ▼")
        self.show_prompt_btn.setCheckable(True)
        self.show_prompt_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f0f0f0;
                color: #555;
                font-weight: normal;
                border: 1px solid #ddd;
                padding: 4px 8px;
            }
            QPushButton:checked { background-color: #e0e0e0; }
        """
        )
        self.run_prompt_btn = QPushButton("Prompt ausführen")
        self.run_prompt_btn.setToolTip("Führt den Prompt aus")
        self.run_prompt_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {secondary_color};
                color: white;
                padding: 6px 12px;
            }}
            QPushButton:hover {{ background-color: #5a9840; }}
        """
        )
        self.run_prompt_btn.setMaximumWidth(140)
        self.run_prompt_btn.setVisible(False)
        self.run_prompt_btn.clicked.connect(self.run_prompt)

        prompt_header.addWidget(self.show_prompt_btn)
        prompt_header.addStretch(1)
        prompt_header.addWidget(self.run_prompt_btn)

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Generierter Prompt wird hier angezeigt...")
        self.prompt.setReadOnly(False)
        self.prompt.setMaximumHeight(150)
        self.prompt.setVisible(False)
        prompt_layout.addWidget(self.prompt)
        prompt_layout.addLayout(prompt_header)

        self.show_prompt_btn.toggled.connect(
            lambda checked: (
                self.prompt.setVisible(checked),
                self.run_prompt_btn.setVisible(checked),
            )
        )
        self.show_prompt_btn.toggled.connect(
            lambda checked: self.show_prompt_btn.setText(
                "Prompt ausblenden ▲" if checked else "Prompt anzeigen ▼"
            )
        )

        collapsible_layout.addWidget(self.prompt_frame)

        # ======== Button Area ========
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.analyze_button = QPushButton("Analyse starten")
        self.analyze_button.setToolTip("Startet die KI-gestützte Analyse des Textes")
        self.analyze_button.setMinimumHeight(40)
        self.analyze_button.setShortcut("Ctrl+Return")
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)

        self.clear_button = QPushButton("Zurücksetzen")
        self.clear_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e0e0e0;
                color: #333;
            }
            QPushButton:hover { background-color: #d0d0d0; }
        """
        )
        self.clear_button.clicked.connect(self.clear_fields)
        button_layout.addWidget(self.clear_button)

        collapsible_layout.addLayout(button_layout)

        main_layout.addWidget(self.collapsible_container)

        # ======== Separator ========
        self.separator_frame = QFrame()
        self.separator_frame.setFrameShape(QFrame.Shape.HLine)
        self.separator_frame.setStyleSheet(
            """
            QFrame {
                background-color: #cccccc;
                height: 2px;
                margin: 8px 0px;
            }
        """
        )
        main_layout.addWidget(self.separator_frame)

        # ======== NEW RESULTS AREA: QSplitter with Prompt/Result + Navigation ========
        self.results_group = QGroupBox("Analyseergebnis")
        results_layout = QVBoxLayout(self.results_group)
        results_layout.setSpacing(8)
        results_layout.setContentsMargins(10, 20, 10, 10)

        # Main results splitter: Left=Content, Right=Navigation
        self.main_results_splitter = QSplitter(Qt.Orientation.Horizontal)
        results_layout.addWidget(self.main_results_splitter)

        # LEFT SIDE: Content splitter (Prompt above, Result below)
        self.content_splitter = QSplitter(Qt.Orientation.Vertical)

        # Prompt display area
        prompt_display_widget = QWidget()
        prompt_display_layout = QVBoxLayout(prompt_display_widget)
        prompt_display_layout.setContentsMargins(0, 0, 0, 0)

        prompt_header_label = QLabel("📝 Aktueller Prompt:")
        prompt_header_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        prompt_display_layout.addWidget(prompt_header_label)

        self.prompt_display = QTextEdit()
        self.prompt_display.setReadOnly(True)
        self.prompt_display.setPlaceholderText(
            "Hier wird der Prompt für den ausgewählten Chunk angezeigt..."
        )
        self.prompt_display.setStyleSheet(
            """
            QTextEdit {
                background-color: #f8f9fa;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
        """
        )
        prompt_display_layout.addWidget(self.prompt_display)

        self.content_splitter.addWidget(prompt_display_widget)

        # Result display area
        result_display_widget = QWidget()
        result_display_layout = QVBoxLayout(result_display_widget)
        result_display_layout.setContentsMargins(0, 0, 0, 0)

        result_header_label = QLabel("🤖 KI-Antwort:")
        result_header_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        result_display_layout.addWidget(result_header_label)

        # In setup_ui method, after creating results_edit:
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setPlaceholderText("Hier erscheinen die Analyseergebnisse...")
        self.results_edit.setStyleSheet(
            """
            QTextEdit {
                font-size: 12pt;
                line-height: 1.3;
            }
        """
        )

        # Apply the highlighter
        self.keyword_highlighter = KeywordHighlighter(self.results_edit.document())

        result_display_layout.addWidget(self.results_edit)

        self.content_splitter.addWidget(result_display_widget)

        # Set content splitter proportions (30% prompt, 70% result)
        self.content_splitter.setSizes([200, 500])

        self.main_results_splitter.addWidget(self.content_splitter)

        # RIGHT SIDE: Navigation and History
        navigation_widget = QWidget()
        navigation_layout = QVBoxLayout(navigation_widget)
        navigation_layout.setContentsMargins(0, 0, 0, 0)

        nav_header_label = QLabel("📋 Chunks & Ergebnisse:")
        nav_header_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        navigation_layout.addWidget(nav_header_label)

        self.results_list = QListWidget()
        self.results_list.setToolTip(
            "Klicken Sie auf einen Chunk, um Prompt und Ergebnis anzuzeigen"
        )
        navigation_layout.addWidget(self.results_list)

        # Navigation buttons
        nav_buttons = QHBoxLayout()

        self.copy_button = QPushButton("📋 Kopieren")
        self.copy_button.setToolTip("Kopiert das ausgewählte Ergebnis")
        self.copy_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {secondary_color};
                padding: 4px 8px;
            }}
            QPushButton:hover {{ background-color: #5a9840; }}
        """
        )
        self.copy_button.clicked.connect(self.copy_result)
        nav_buttons.addWidget(self.copy_button)

        self.delete_button = QPushButton("🗑️ Löschen")
        self.delete_button.setToolTip("Löscht den ausgewählten Eintrag")
        self.delete_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e74c3c;
                padding: 4px 8px;
            }
            QPushButton:hover { background-color: #c0392b; }
        """
        )
        self.delete_button.clicked.connect(self.delete_result)
        nav_buttons.addWidget(self.delete_button)

        navigation_layout.addLayout(nav_buttons)

        navigation_widget.setMaximumWidth(250)
        self.main_results_splitter.addWidget(navigation_widget)

        # Set main splitter proportions
        self.main_results_splitter.setSizes([700, 250])

        main_layout.addWidget(self.results_group, 1)

        # Connect navigation signal
        self.results_list.itemClicked.connect(self.show_chunk_result)

        # Initialize models
        self.update_models(self.provider_combo.currentText())

    # ======== NEW CHUNK NAVIGATION METHODS ========
    def show_chunk_result(self, item):
        """Display selected chunk's prompt and result."""
        chunk_item = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(chunk_item, ChunkResultItem):
            self.prompt_display.setPlainText(chunk_item.prompt)
            self.results_edit.setPlainText(chunk_item.result)

            # Extract and emit signals for compatibility
            try:
                keywords = self.extract_keywords(chunk_item.result)
                if keywords:
                    self.keywords_extracted.emit(keywords)

                gnd_system = self.extract_gnd_system(chunk_item.result)
                if gnd_system:
                    self.gnd_systematic.emit(gnd_system)

            except Exception as e:
                self.logger.error(f"Error extracting data from chunk result: {e}")

    def add_chunk_result_to_list(self, chunk_item: ChunkResultItem):
        """Add a chunk result to the navigation list."""
        display_text = f"🧩 Chunk {chunk_item.chunk_id + 1}"
        if chunk_item.chunk_info:
            display_text += f" ({chunk_item.chunk_info})"

        if chunk_item.chunk_type == "combined":
            display_text = f"📋 Zusammenfassung aller Chunks"
        elif chunk_item.chunk_type == "single":
            display_text = f"📄 Einzelanalyse"

        list_item = QListWidgetItem(display_text)
        list_item.setData(Qt.ItemDataRole.UserRole, chunk_item)
        list_item.setToolTip(f"Typ: {chunk_item.chunk_type}\n{chunk_item.chunk_info}")

        self.results_list.addItem(list_item)
        self.chunk_results.append(chunk_item)

    # ======== MODIFIED ANALYSIS METHODS ========
    def start_analysis(self):
        """Start analysis with new chunk workflow."""
        if not self.abstract_edit.toPlainText().strip():
            self.handle_error("Fehlende Eingabe", "Bitte geben Sie einen Text ein.")
            return

        self.toggle_compact_mode(force_state=True)
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)

        # Reset states
        self.chunk_results.clear()
        # We want all results be stored for the runtime
        #self.results_list.clear()
        self.prompt_display.clear()
        self.results_edit.clear()
        self.generated_response = ""

        # Activate cancel button
        self.cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton:hover { background-color: #c0392b; }
        """
        )

        # Determine analysis type
        if (
            self.enable_chunk_keywords.isChecked()
            or self.enable_chunk_abstract.isChecked()
        ):
            self.start_chunked_analysis()
        else:
            self.start_single_analysis()

    def run_prompt(self):
        """Start analysis with new chunk workflow."""

        self.toggle_compact_mode(force_state=True)
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)

        # Reset states
        self.chunk_results.clear()
        # We want all results be stored for the runtime
        #self.results_list.clear()
        self.prompt_display.clear()
        self.results_edit.clear()
        self.generated_response = ""

        # Activate cancel button
        self.cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton:hover { background-color: #c0392b; }
        """
        )
        model = self.model_combo.currentText().strip()
        provider = self.provider_combo.currentText()
        prompt = self.prompt.toMarkdown().strip()
        self.prompt_display.setPlainText(prompt)
        # Create single chunk item for tracking
        self.current_single_chunk = ChunkResultItem(
            chunk_id=0,
            prompt=prompt,
            result="",
            chunk_type="single",
            chunk_info="Einzelanalyse",
        )

        self.current_request_id = str(uuid.uuid4())
        try:
            self.llm.generate_response(
                provider=provider,
                model=model,
                prompt=prompt,
                request_id=self.current_request_id,
                temperature=self.ki_temperature.value() / 100,
                seed=self.ki_seed.value() if self.ki_seed.value() > 0 else None,
                system=self.system,
                stream=True,
            )
        except Exception as e:
            self.handle_error("Fehler bei der Anfrage", str(e))
            self.analysis_completed()

    def start_single_analysis(self):
        """Start normal single analysis with new result handling."""
        self.is_processing_chunks = False
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Generiere Antwort...")

        model = self.model_combo.currentText().strip()
        provider = self.provider_combo.currentText()
        prompt = self.set_input()
        self.prompt_display.setPlainText(prompt)
        # Create single chunk item for tracking
        self.current_single_chunk = ChunkResultItem(
            chunk_id=0,
            prompt=prompt,
            result="",
            chunk_type="single",
            chunk_info="Einzelanalyse",
        )
        self.current_request_id = str(uuid.uuid4())

        try:
            self.llm.generate_response(
                provider=provider,
                model=model,
                prompt=prompt,
                request_id=self.current_request_id,
                temperature=self.ki_temperature.value() / 100,
                seed=self.ki_seed.value() if self.ki_seed.value() > 0 else None,
                system=self.system,
                stream=True,
            )
        except Exception as e:
            self.handle_error("Fehler bei der Anfrage", str(e))
            self.analysis_completed()

    def start_chunked_analysis(self):
        """Start chunked analysis with new workflow."""
        abstract_text = self.abstract_edit.toPlainText().strip()
        keywords_text = self.keywords_edit.toPlainText().strip()

        self.is_processing_chunks = True
        self.tmp_task = self.task
        self.logger.info(f"Starting chunked analysis for task: {self.task}")
        if self.task == "keywords":
            self.task = "keywords_chunked"
            self.logger.info(f"Switching to: {self.task}")
            self.set_input()
        chunks_to_process = []

        # Create chunks based on settings
        if self.enable_chunk_abstract.isChecked():
            abstract_chunks = self.chunk_abstract_by_lines(
                abstract_text, self.abstract_chunk_slider.value()
            )
            for i, chunk in enumerate(abstract_chunks):
                chunk_info = f"Zeilen {i * self.abstract_chunk_slider.value() + 1}-{min((i + 1) * self.abstract_chunk_slider.value(), len(abstract_text.split('\\n')))}"
                chunks_to_process.append(
                    {
                        "type": "abstract",
                        "abstract": chunk,
                        "keywords": keywords_text,
                        "info": chunk_info,
                    }
                )

        if self.enable_chunk_keywords.isChecked():
            keyword_chunks = self.chunk_keywords_by_comma(
                keywords_text, self.keyword_chunk_slider.value()
            )
            for i, chunk in enumerate(keyword_chunks):
                chunk_info = f"Keywords {i * self.keyword_chunk_slider.value() + 1}-{min((i + 1) * self.keyword_chunk_slider.value(), len(keywords_text.split(',')))}"
                chunks_to_process.append(
                    {
                        "type": "keywords",
                        "abstract": abstract_text,
                        "keywords": chunk,
                        "info": chunk_info,
                    }
                )

        # If both chunking types enabled, create combined chunks
        if (
            self.enable_chunk_abstract.isChecked()
            and self.enable_chunk_keywords.isChecked()
        ):
            chunks_to_process = []
            abstract_chunks = self.chunk_abstract_by_lines(
                abstract_text, self.abstract_chunk_slider.value()
            )
            keyword_chunks = self.chunk_keywords_by_comma(
                keywords_text, self.keyword_chunk_slider.value()
            )

            for i, abstract_chunk in enumerate(abstract_chunks):
                for j, keyword_chunk in enumerate(keyword_chunks):
                    chunk_info = f"Abstract {i+1}, Keywords {j+1}"
                    chunks_to_process.append(
                        {
                            "type": "both",
                            "abstract": abstract_chunk,
                            "keywords": keyword_chunk,
                            "info": chunk_info,
                        }
                    )

        self.current_processing_chunks = chunks_to_process
        self.current_chunk_index = 0
        self.total_chunks = len(chunks_to_process)

        self.progress_bar.setRange(0, self.total_chunks)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Chunk %v von {self.total_chunks}")

        # Start processing first chunk
        self.process_next_chunk()

    def process_next_chunk(self):
        """Process next chunk in the queue."""
        if self.current_chunk_index >= len(self.current_processing_chunks):
            self.create_combined_result()
            return

        chunk_data = self.current_processing_chunks[self.current_chunk_index]
        self.progress_bar.setValue(self.current_chunk_index)

        # Create chunk prompt
        chunk_prompt = self.create_chunk_prompt_from_data(chunk_data)

        # Create chunk item for tracking
        self.current_chunk_item = ChunkResultItem(
            chunk_id=self.current_chunk_index,
            prompt=chunk_prompt,
            result="",
            chunk_type=chunk_data["type"],
            chunk_info=chunk_data["info"],
        )

        # Process chunk
        model = self.model_combo.currentText().strip()
        provider = self.provider_combo.currentText()
        self.prompt_display.setPlainText(chunk_prompt)
        self.logger.info(self.matched_keywords)
        self.current_request_id = str(uuid.uuid4())
        try:
            self.llm.generate_response(
                provider=provider,
                model=model,
                prompt=chunk_prompt,
                request_id=self.current_request_id,
                temperature=self.ki_temperature.value() / 100,
                seed=self.ki_seed.value() if self.ki_seed.value() > 0 else None,
                system=self.system,
                stream=True,
            )
        except Exception as e:
            self.handle_error("Fehler bei Chunk-Verarbeitung", str(e))
            self.analysis_completed()

    def create_combined_result(self):
        """Create combined result from all chunks."""
        combined_text = "=== ZUSAMMENFASSUNG ALLER CHUNKS ===\\n\\n"

        for i, chunk_result in enumerate(self.chunk_results):
            combined_text += f"--- Chunk {i + 1} ({chunk_result.chunk_info}) ---\\n"
            combined_text += chunk_result.result + "\\n\\n"

        # Create combined result item
        combined_item = ChunkResultItem(
            chunk_id=len(self.chunk_results),
            prompt="Zusammenfassung aller verarbeiteten Chunks",
            result=combined_text,
            chunk_type="combined",
            chunk_info=f"{len(self.chunk_results)} Chunks kombiniert",
        )

        self.add_chunk_result_to_list(combined_item)

        # Select the combined result
        self.results_list.setCurrentRow(self.results_list.count() - 1)
        self.show_chunk_result(self.results_list.currentItem())

        self.analysis_completed()
        self.create_last_analysis()

    def create_last_analysis(self):
        """Process the last analysis."""
        combined_text = "=== ENDANALYSE ALLER CHUNKS ===\\n\\n"

        self.is_processing_chunks = False
        self.task = self.tmp_task
        self.set_input()
        self.progress_bar.setValue(self.current_chunk_index)

        # Create chunk prompt
        prompt = ""
        template = self.current_template
        variables = {
            "abstract": self.abstract,
            "keywords": ", ".join(self.extract_and_match_keywords_from_results()),
        }

        try:
            prompt = template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing variable in template: {e}")
            return f"Fehler im Template: Variable {e} fehlt"

        # Create combined result item
        combined_item = ChunkResultItem(
            chunk_id=len(self.chunk_results),
            prompt=prompt,
            result=combined_text,
            chunk_type="combined",
            chunk_info=f"{len(self.chunk_results)} Chunks kombiniert und analysiert",
        )
        self.add_chunk_result_to_list(combined_item)

        # Process chunk
        model = self.model_combo.currentText().strip()
        provider = self.provider_combo.currentText()
        self.prompt_display.setPlainText(prompt)

        self.current_request_id = str(uuid.uuid4())
        try:
            self.llm.generate_response(
                provider=provider,
                model=model,
                prompt=prompt,
                request_id=self.current_request_id,
                temperature=self.ki_temperature.value() / 100,
                seed=self.ki_seed.value() if self.ki_seed.value() > 0 else None,
                system=self.system,
                stream=True,
            )
        except Exception as e:
            self.handle_error("Fehler bei Chunk-Verarbeitung", str(e))

    def extract_and_match_keywords_from_results(self) -> Dict[str, str]:
        """
        Extract final_list sections from all results and match against keyword database.

        Returns:
            Dict[str, str]: Dictionary of matched keywords with their GND numbers
        """
        matched_keywords = {}

        # Get current keywords database from keywords_edit
        keywords_dict = self.parse_keywords_from_list(self.keywords_edit.toPlainText())

        if not keywords_dict:
            self.logger.warning("No keywords database available for matching")
            return matched_keywords

        # Collect all results text
        all_results_text = ""

        # Add current display result
        if self.results_edit.toPlainText():
            all_results_text += self.results_edit.toPlainText() + "\n\n"

        # Add all chunk results
        for chunk_result in self.chunk_results:
            all_results_text += chunk_result.result + "\n\n"

        # Extract final_list sections using regex
        import re

        final_list_pattern = r"<final_list>\s*(.*?)\s*</final_list>"
        final_list_matches = re.findall(
            final_list_pattern, all_results_text, re.DOTALL | re.IGNORECASE
        )

        self.logger.info(f"Found {len(final_list_matches)} final_list sections")

        # Process each final_list section
        for i, final_list_content in enumerate(final_list_matches):
            self.logger.info(
                f"Processing final_list {i+1}: {final_list_content[:100]}..."
            )

            # Split by pipe separator and clean up
            result_terms = [
                term.strip() for term in final_list_content.split("|") if term.strip()
            ]

            # Match against keyword database
            for term in result_terms:
                # Direct keyword match (case-insensitive)
                for keyword, gnd_id in keywords_dict.items():
                    if term.lower() == keyword.lower():
                        matched_keywords[keyword] = gnd_id
                        self.logger.info(
                            f"Direct match: '{term}' -> {keyword} ({gnd_id})"
                        )
                        break
                    # Partial match (term contains keyword or vice versa)
                    elif (
                        keyword.lower() in term.lower()
                        or term.lower() in keyword.lower()
                    ) and len(term) > 3:
                        matched_keywords[keyword] = gnd_id
                        self.logger.info(
                            f"Partial match: '{term}' -> {keyword} ({gnd_id})"
                        )
                        break

                # Check if term is a GND ID
                if re.match(r"\d{7}-\d", term):
                    for keyword, gnd_id in keywords_dict.items():
                        if gnd_id == term:
                            matched_keywords[keyword] = gnd_id
                            self.logger.info(f"GND match: '{term}' -> {keyword}")
                            break

        self.logger.info(f"Total matched keywords: {len(matched_keywords)}")
        return matched_keywords

    def get_matched_keywords_formatted(self) -> str:
        """
        Get matched keywords in the same format as input.

        Returns:
            str: Formatted string like "Keyword1 (GND-1), Keyword2 (GND-2)"
        """
        matched = self.extract_and_match_keywords_from_results()

        if not matched:
            return ""

        formatted_pairs = [
            f"{keyword} ({gnd_id})" for keyword, gnd_id in matched.items()
        ]
        return ", ".join(formatted_pairs)

    def update_matched_keywords_display(self):
        """Update UI with matched keywords (optional display method)."""
        matched_keywords = self.extract_and_match_keywords_from_results()

        if matched_keywords:
            formatted_text = self.get_matched_keywords_formatted()
            self.logger.info(f"Matched keywords: {formatted_text}")

            # Emit signal for other components
            self.keywords_extracted.emit(formatted_text)

            # Store in instance variable for access
            self.matched_keywords_from_results = matched_keywords

            return matched_keywords
        else:
            self.logger.info("No keywords matched from results")
            self.matched_keywords_from_results = {}
            return {}

    def on_generation_finished(self, request_id: str, message: str):
        """Handle completion with keyword matching."""
        if request_id != self.current_request_id:
            return
        self.logger.info(f"Generation finished: {message}")

        if hasattr(self, "current_single_chunk") and not self.is_processing_chunks:
            # Single analysis completion
            self.current_single_chunk.result = self.generated_response
            self.add_chunk_result_to_list(self.current_single_chunk)

            # Auto-select the result
            self.results_list.setCurrentRow(0)
            self.show_chunk_result(self.results_list.item(0))

            # Extract and match keywords from results
            self.update_matched_keywords_display()
            self.logger.info(self.matched_keywords)
            self.analysis_completed()

        elif self.is_processing_chunks and hasattr(self, "current_chunk_item"):
            # Chunk processing completion
            self.current_chunk_item.result = self.generated_response
            self.add_chunk_result_to_list(self.current_chunk_item)

            # Move to next chunk
            self.current_chunk_index += 1
            self.generated_response = ""

            if self.current_chunk_index < len(self.current_processing_chunks):
                self.process_next_chunk()
                self.update_matched_keywords_display()
                self.logger.info(self.matched_keywords)

            else:
                if self.tmp_task == "keywords":
                    self.task = "keywords"

                # Extract and match keywords from all results
                self.update_matched_keywords_display()
                self.create_combined_result()
                self.logger.info(self.matched_keywords)

    # ======== MODIFIED RESPONSE HANDLING ========
    def on_text_received(self, request_id: str, text_chunk: str):
        """Handle streaming text with new display system."""
        if request_id != self.current_request_id:
            return
        # Display in current result area
        cursor = self.results_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text_chunk)
        self.results_edit.setTextCursor(cursor)
        self.results_edit.ensureCursorVisible()
        self.generated_response += text_chunk
        QApplication.processEvents()

    # ======== KEEP EXISTING HELPER METHODS (unchanged) ========
    def chunk_abstract_by_lines(self, text: str, lines_per_chunk: int) -> List[str]:
        """Split abstract text into chunks by line count."""
        lines = text.split("\\n")
        chunks = []

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i : i + lines_per_chunk]
            chunks.append("\\n".join(chunk_lines))

        self.logger.info(
            f"Split abstract into {len(chunks)} chunks of {lines_per_chunk} lines each"
        )
        return chunks

    def chunk_keywords_by_comma(
        self, keywords_text: str, keywords_per_chunk: int
    ) -> List[str]:
        """Split keywords into chunks by comma-separated count."""
        keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
        chunks = []

        for i in range(0, len(keywords), keywords_per_chunk):
            chunk_keywords = keywords[i : i + keywords_per_chunk]
            chunks.append(", ".join(chunk_keywords))

        self.logger.info(
            f"Split keywords into {len(chunks)} chunks of {keywords_per_chunk} keywords each"
        )
        return chunks

    def create_chunk_prompt_from_data(self, chunk_data: dict) -> str:
        """Create prompt from chunk data."""
        template = self.current_template
        variables = {
            "abstract": chunk_data.get("abstract", ""),
            "keywords": chunk_data.get("keywords", "Keine Keywords vorhanden"),
        }

        try:
            prompt = template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing variable in template: {e}")
            return f"Fehler im Template: Variable {e} fehlt"

        return prompt

    # ======== KEEP ALL OTHER EXISTING METHODS UNCHANGED ========
    def toggle_compact_mode(self, force_state=None):
        """Toggle between compact and full mode."""
        if force_state is not None:
            if self.is_compact_mode == force_state:
                return
            self.is_compact_mode = force_state
        else:
            self.is_compact_mode = not self.is_compact_mode

        if self.is_compact_mode:
            self.toggle_sections_btn.setText("▼ Eingabebereich")
        else:
            self.toggle_sections_btn.setText("▲ Eingabebereich")

        self.collapsible_container.setVisible(not self.is_compact_mode)

    def set_model(self, model_text):
        """Set the current model and update configuration."""
        model = model_text.strip()
        if model.startswith("↳"):
            return
        self.chosen_model = model
        self.logger.info(f"Setting model to: '{model}'")
        try:
            config = self.promptmanager.get_prompt_config(self.task, self.chosen_model)
            temp_value = int(config.get("temp", 0.7) * 100)
            self.ki_temperature.setValue(temp_value)
            self.current_template = config.get("prompt", "")
            self.system = config.get("system", "")
            self.set_input()
        except Exception as e:
            self.logger.error(f"Error setting model config: {e}")

    def update_temperature_label(self, value):
        """Update temperature label to show actual value."""
        self.temperature_label.setText(f"Temperatur: {value/100:.2f}")

    def parse_keywords_from_list(self, keywords_string: str) -> Dict[str, str]:
        """Parse keywords from formatted string like 'Keyword (GND-Number), ...'"""
        keywords_dict = {}

        if not keywords_string.strip():
            return keywords_dict

        # Split by comma and process each entry
        entries = [entry.strip() for entry in keywords_string.split(",")]

        for entry in entries:
            if "(" in entry and ")" in entry:
                # Extract keyword and GND number
                keyword = entry.split("(")[0].strip()
                gnd_match = entry.split("(")[1].split(")")[0].strip()

                if keyword and gnd_match:
                    keywords_dict[keyword] = gnd_match

        return keywords_dict

    def update_highlighter_from_keywords(self):
        """Update highlighter with current keywords from keywords_edit."""
        keywords_text = self.keywords_edit.toPlainText()
        keywords_dict = self.parse_keywords_from_list(keywords_text)

        if hasattr(self, "keyword_highlighter") and keywords_dict:
            self.keyword_highlighter.set_keywords_from_database(keywords_dict)
            self.logger.info(f"Updated highlighter with {len(keywords_dict)} keywords")

    def update_input(self):
        """Update the prompt when input changes."""
        self.prompt.setPlainText(self.set_input())
        self.abstract_changed.emit(self.abstract_edit.toPlainText().strip())

        # Update highlighter when keywords change
        self.update_highlighter_from_keywords()

    def set_input(self):
        """Prepare prompt using template and current input values."""
        config = self.promptmanager.get_prompt_config(self.task, self.chosen_model)
        temp_value = int(config.get("temp", 0.7) * 100)
        self.ki_temperature.setValue(temp_value)
        self.current_template = config.get("prompt", "")
        self.system = config.get("system", "")
        template = self.current_template
        variables = {
            "abstract": self.abstract_edit.toPlainText().strip(),
            "keywords": (
                self.keywords_edit.toPlainText().strip()
                if self.keywords_edit.toPlainText().strip()
                else "Keine Keywords vorhanden"
            ),
        }

        try:
            prompt = template.format(**variables)
        except KeyError as e:
            return f"Fehler im Template: Variable {e} fehlt"
        return prompt

    def extract_keywords(self, text):
        """Extract keywords from response text."""
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        match = re.search(r"<final_list>(.*?)</final_list>", cleaned_text, re.DOTALL)
        if not match:
            return ""

        keywords = match.group(1).split("|")
        quoted_keywords = [
            f'"{keyword.strip()}"' for keyword in keywords if keyword.strip()
        ]
        result = ", ".join(quoted_keywords)
        self.final_list.emit(result)
        return result

    def extract_gnd_system(self, text: str) -> str:
        """Extract GND system from response."""
        try:
            cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            match = re.search(r"<class>(.*?)</class>", cleaned_text, re.DOTALL)
            if not match:
                return ""
            ognd_system = match.group(1).strip()
            self.gnd_systematic.emit(ognd_system)
            return ognd_system
        except Exception as e:
            self.logger.error(f"Error extracting class content: {str(e)}")
            return ""

    def copy_result(self):
        """Copy current result to clipboard."""
        item = self.results_list.currentItem()
        if not item:
            return

        chunk_item = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(chunk_item, ChunkResultItem):
            text_to_copy = chunk_item.result
            try:
                extracted = self.extract_keywords(text_to_copy)
                if extracted:
                    text_to_copy = extracted
            except:
                pass

            clipboard = QApplication.clipboard()
            clipboard.setText(text_to_copy)
            QToolTip.showText(
                self.copy_button.mapToGlobal(QPoint(0, -30)),
                "In die Zwischenablage kopiert",
                self.copy_button,
                QRect(),
                2000,
            )

    def delete_result(self):
        """Delete selected result."""
        current_row = self.results_list.currentRow()
        if current_row == -1:
            return

        reply = QMessageBox.question(
            self,
            "Bestätigung",
            "Möchten Sie diesen Eintrag wirklich löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove from list and internal storage
            removed_item = self.results_list.takeItem(current_row)
            chunk_item = removed_item.data(Qt.ItemDataRole.UserRole)

            if chunk_item in self.chunk_results:
                self.chunk_results.remove(chunk_item)

            if self.results_list.count() == 0:
                self.results_edit.clear()
                self.prompt_display.clear()

    def analysis_completed(self):
        """Cleanup after analysis completion."""
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #cccccc;
                color: #666666;
            }
        """
        )
        self.is_processing_chunks = False

    def cancel_analysis(self):
        """Cancel running analysis."""
        current_style = self.cancel_button.styleSheet()
        if "background-color: #e74c3c" in current_style:
            try:
                self.llm.cancel_generation(reason="user_requested")
            except Exception as e:
                self.logger.error(f"Error canceling generation: {e}")
                self.analysis_completed()

    def on_generation_error(self, request_id: str, error_message: str):
        """Handle generation errors."""
        if request_id != self.current_request_id:
            return
        self.handle_error("Generierungsfehler", error_message)
        self.analysis_completed()

    def on_generation_cancelled(self, request_id: str):
        """Handle generation cancellation."""
        if request_id != self.current_request_id:
            return
        if self.generated_response:
            self.results_edit.setPlainText(
                f"{self.generated_response}\n\n[ABGEBROCHEN]"
            )
        else:
            self.results_edit.setPlainText("[ABGEBROCHEN]")
        self.analysis_completed()

    def handle_error(self, title: str, message: str):
        """Show error dialog."""
        QMessageBox.critical(self, title, message)
        self.logger.error(f"{title}: {message}")

    def set_ui_enabled(self, enabled: bool):
        """Enable/disable UI elements."""
        self.analyze_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.abstract_edit.setEnabled(enabled)
        self.keywords_edit.setEnabled(enabled)
        self.provider_combo.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.ki_temperature.setEnabled(enabled)
        self.ki_seed.setEnabled(enabled)
        self.pdf_button.setEnabled(enabled)
        self.abstract_chunk_slider.setEnabled(
            enabled and self.enable_chunk_abstract.isChecked()
        )
        self.keyword_chunk_slider.setEnabled(
            enabled and self.enable_chunk_keywords.isChecked()
        )

    def clear_fields(self):
        """Reset all fields."""
        if (
            self.abstract_edit.toPlainText()
            or self.keywords_edit.toPlainText()
            or self.results_edit.toPlainText()
            or len(self.chunk_results) > 0
        ):
            reply = QMessageBox.question(
                self,
                "Bestätigung",
                "Möchten Sie wirklich alle Felder zurücksetzen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.abstract_edit.clear()
                self.keywords_edit.clear()
                self.results_edit.clear()
                self.prompt_display.clear()
                #self.results_list.clear()
                self.chunk_results.clear()
                self.pdf_info_frame.setVisible(False)
                self.pdf_path = None

    def setup_animations(self):
        """Set up animations for UI effects."""
        self.container_animation = QPropertyAnimation(
            self.collapsible_container, b"maximumHeight"
        )
        self.container_animation.setDuration(300)
        self.container_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_keywords(self, keywords):
        """Set keywords in input field."""
        self.keywords = keywords
        self.keywords_edit.setPlainText(self.keywords)
        self.update_input()
        # Update highlighter with new keywords
        self.update_highlighter_from_keywords()

    def set_abstract(self, abstract):
        """Set abstract in input field."""
        self.abstract = abstract
        self.abstract_edit.setPlainText(self.abstract)
        self.update_input()

    def on_chunking_enabled(self, enabled: bool):
        """Handle chunking enable/disable."""
        self.abstract_chunk_slider.setEnabled(enabled)
        self.keyword_chunk_slider.setEnabled(enabled)

    def on_abstract_chunk_changed(self, value: int):
        """Handle abstract chunk size changes."""
        self.abstract_chunk_display.setText(str(value))

    def on_keyword_chunk_changed(self, value: int):
        """Handle keyword chunk size changes."""
        self.keyword_chunk_display.setText(str(value))

    # ======== PDF IMPORT METHODS (unchanged) ========
    def import_pdf(self):
        """Import text from PDF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "PDF-Datei öffnen", "", "PDF-Dateien (*.pdf)"
        )
        if not file_path:
            return

        self.pdf_path = file_path
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Verarbeite PDF...")
        self.progress_bar.setRange(0, 0)

        try:
            with open(self.pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                metadata = {
                    "title": info.title if info and hasattr(info, "title") else "",
                    "author": info.author if info and hasattr(info, "author") else "",
                    "subject": (
                        info.subject if info and hasattr(info, "subject") else ""
                    ),
                    "keywords": (
                        info.keywords if info and hasattr(info, "keywords") else ""
                    ),
                    "pages": len(pdf_reader.pages),
                    "filename": os.path.basename(self.pdf_path),
                }

                text = ""
                max_pages = min(50, len(pdf_reader.pages))
                for i in range(max_pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    text += page_text + "\\n\\n"

            self.handle_pdf_content(text, metadata)
        except Exception as e:
            self.handle_error("Fehler bei der PDF-Verarbeitung", str(e))

        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)

    def handle_pdf_content(self, text, metadata):
        """Handle extracted PDF content."""
        self.pdf_metadata = metadata
        metadata_text = f"<b>PDF-Informationen:</b><br>"
        if metadata.get("title"):
            metadata_text += f"<b>Titel:</b> {metadata['title']}<br>"
        if metadata.get("author"):
            metadata_text += f"<b>Autor:</b> {metadata['author']}<br>"
        metadata_text += f"<b>Seiten:</b> {metadata.get('pages', '?')}<br>"
        metadata_text += f"<b>Datei:</b> {os.path.basename(self.pdf_path)}"

        self.pdf_metadata_label.setText(metadata_text)
        self.pdf_info_frame.setVisible(True)
        self.abstract_edit.setPlainText(text)
        has_keywords = bool(metadata.get("keywords") or metadata.get("subject"))
        self.extract_keywords_btn.setVisible(has_keywords)

    def extract_keywords_from_pdf(self):
        """Extract keywords from PDF metadata."""
        if not self.pdf_metadata:
            return

        keywords = []
        for field in ["keywords", "subject"]:
            if self.pdf_metadata.get(field):
                kw = self.pdf_metadata.get(field, "")
                if isinstance(kw, str):
                    for separator in [",", ";", "\\n", " - "]:
                        if separator in kw:
                            keywords.extend(
                                [k.strip() for k in kw.split(separator) if k.strip()]
                            )
                            break
                    else:
                        keywords.append(kw.strip())

        unique_keywords = list(set(keywords))
        if not unique_keywords:
            QMessageBox.information(
                self,
                "Keine Keywords",
                "Es konnten keine Keywords aus den PDF-Metadaten extrahiert werden.",
            )
            return

        current_kws = self.keywords_edit.toPlainText()
        if current_kws.strip():
            new_kws = current_kws + "\\n" + "\\n".join(unique_keywords)
            self.keywords_edit.setPlainText(new_kws)
        else:
            self.keywords_edit.setPlainText("\\n".join(unique_keywords))

        QMessageBox.information(
            self,
            "Keywords importiert",
            f"{len(unique_keywords)} Keywords wurden extrahiert.",
        )
