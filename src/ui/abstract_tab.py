from PyQt6.QtWidgets import (QApplication,
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QMessageBox, QProgressBar,
    QSlider, QSpinBox, QCheckBox, QComboBox, QSplitter, QListWidget,
    QListWidgetItem, QGridLayout, QFrame, QToolButton, QGroupBox,
    QFileDialog, QScrollArea, QSizePolicy, QToolTip, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThread, QObject, QSize, QPoint, QRect
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette, QPixmap, QAction
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

class AbstractTab(QWidget):
    """
    Ein modernisiertes Tab für die Analyse von Abstracts und Extraktion von Keywords mittels LLMs.
    
    Mit ausblendbaren Bereichen für Eingabe und Konfiguration, die nach der Antwortgenerierung
    eingefahren werden, um mehr Platz für die Ergebnisanzeige zu bieten.
    """
    # Signals
    keywords_extracted = pyqtSignal(str)
    abstract_changed = pyqtSignal(str)
    final_list = pyqtSignal(str)

    def __init__(self, parent=None, recommendations_file: Path = Path(__file__).parent.parent.parent / "model_recommendations.json"):
        super().__init__(parent)
        # Initialize components
        self.llm = LLMInterface()
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
        self.system = ""
        self.generated_response = ""
        self.pdf_metadata = {}
        self.pdf_path = None
        
        # Load configurations
        self.load_recommendations()
        self.set_model_recommendations("default")
        self.required = []

        # Neue Attribute für ausblendbare Bereiche
        self.is_compact_mode = False
        self.input_group = None
        self.config_group = None
        self.prompt_frame = None
        self.results_group = None
        
        # Verbinde LLM-Signale mit Slots
        self.llm.text_received.connect(self.on_text_received)
        self.llm.generation_finished.connect(self.on_generation_finished)
        self.llm.generation_error.connect(self.on_generation_error)
        self.llm.generation_cancelled.connect(self.on_generation_cancelled)
        
        # Set up the UI
        self.setup_ui()
        self.setup_animations()

    def load_recommendations(self):
        """Load model recommendations from the JSON file."""
        try:
            if not self.recommendations_file.exists():
                self.logger.warning(f"Recommendations file not found: {self.recommendations_file}")
                self.create_default_recommendations()
                return

            with open(self.recommendations_file, 'r', encoding='utf-8') as f:
                self.recommendations = json.load(f)
                
            self.logger.info(f"Successfully loaded recommendations from {self.recommendations_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading recommendations: {e}")
            self.create_default_recommendations()

    def create_default_recommendations(self):
        """Create default recommendations if file is missing."""
        self.recommendations = {
            "default": {
                "recommended": {},
                "descriptions": {}
            }
        }
        self.logger.info("Created default recommendations")

    def load_prompts(self):
        """Load prompt templates from the JSON file."""
        try:
            if not self.propmpt_file.exists():
                self.logger.warning(f"Prompt file not found: {self.propmpt_file}")
                return

            with open(self.propmpt_file, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
                
            self.logger.info(f"Successfully loaded prompts from {self.propmpt_file}")
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")

    def set_task(self, task: str):
        """Set the task type for model recommendations."""
        self.task = task
        self.logger.info(f"Set task to {task}")
        
        # Hole verfügbare Modelle vom PromptManager
        self.recommended_models = self.promptmanager.get_available_models(task)
        self.logger.info(f"Available models for {task}: {self.recommended_models}")
        
        # UI aktualisieren
        self.update_models(self.provider_combo.currentText())
        self.pdf_button.setVisible(self.task == "abstract")

    def set_model_recommendations(self, use_case: str):
        """Set model recommendations based on the use case."""
        pass  # Handled by prompt manager now

    def update_models(self, provider: str):
        """Update available models when provider changes."""
        self.model_combo.clear()
        
        # Get all available models
        all_models = self.llm.get_available_models(provider)
        recommended_available = []

        self.logger.info(f"Recommended models for {provider}: {self.recommended_models}")
        if self.recommended_models:
            # Add recommended models first
            recommended_group = "↳ Empfohlene Modelle"
            self.model_combo.addItem(recommended_group)
            recommended_available = [model for model in self.recommended_models if model in all_models]
            
            for model in recommended_available:
                self.model_combo.addItem(f"  {model}")
                # Set tooltip with description
                idx = self.model_combo.count() - 1
                description = self.model_descriptions.get(provider, {}).get(model, "")
                self.model_combo.setItemData(idx, description, Qt.ItemDataRole.ToolTipRole)
            
            if recommended_available and len(all_models) > len(recommended_available):
                self.model_combo.addItem("↳ Weitere verfügbare Modelle")
            
            # Add remaining models
            other_models = [model for model in all_models if model not in recommended_available]
            for model in other_models:
                self.model_combo.addItem(f"  {model}")

        else:
            # If no recommendations, add all models
            self.model_combo.addItems(all_models)

        # Select the first recommended model if available
        if recommended_available:
            self.model_combo.setCurrentText(f"  {recommended_available[0]}")
            self.set_model(recommended_available[0])
        elif all_models:
            # Otherwise select the first available model
            self.model_combo.setCurrentText(all_models[0])
            self.set_model(all_models[0])

    def setup_ui(self):
        """Set up the user interface components with a modern design and collapsible sections."""
        # Hauptlayout mit Abständen
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Farbdefinitionen für das UI
        primary_color = "#4a86e8"    # Blau
        secondary_color = "#6aa84f"  # Grün
        accent_color = "#f1c232"     # Gold
        bg_light = "#f8f9fa"         # Hell-Grau
        text_color = "#333333"       # Dunkelgrau
        
        # Globale QSS-Stile
        self.setStyleSheet(f"""
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
            QPushButton:hover {{
                background-color: #3a76d8;
            }}
            QPushButton:pressed {{
                background-color: #2a66c8;
            }}
            QLabel {{
                color: {text_color};
            }}
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
        """)
        
        # ======== ABBRUCH-BUTTON - Immer sichtbar im oberen Bereich ========
        # Neue Kontrolleiste für Abbruch und Ein-/Ausblenden
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(0, 0, 0, 5)  # Schmale Leiste
        
        # Abbruch-Button
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.setFixedWidth(120)  # Feste Breite
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #cccccc;  /* Inaktiv-Grau */
                color: #666666;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        control_bar.addWidget(self.cancel_button)
        
        # Fortschrittsanzeige
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Generiere Antwort... %p%")
        control_bar.addWidget(self.progress_bar)
        
        # Toggle-Button
        self.toggle_sections_btn = QPushButton("▼ Eingabebereich")
        self.toggle_sections_btn.setFixedWidth(140)  # Feste Breite
        self.toggle_sections_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0; 
                color: #555;
                font-weight: normal;
                border: 1px solid #ddd;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.toggle_sections_btn.clicked.connect(self.toggle_compact_mode)
        control_bar.addWidget(self.toggle_sections_btn)
        
        # Zur Hauptlayout hinzufügen
        main_layout.addLayout(control_bar)

        # ======== Container für Eingabe- und Konfigurationsbereich ========
        self.collapsible_container = QWidget()
        collapsible_layout = QVBoxLayout(self.collapsible_container)
        collapsible_layout.setContentsMargins(0, 0, 0, 0)
        collapsible_layout.setSpacing(12)
        
        # ======== Eingabebereich ========
        self.input_group = QGroupBox("Eingabe")
        input_layout = QVBoxLayout(self.input_group)
        input_layout.setSpacing(8)
        input_layout.setContentsMargins(10, 20, 10, 10)

        # Header mit PDF-Import
        header_layout = QHBoxLayout()
        
        abstract_label = QLabel("Abstract / Text:")
        abstract_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header_layout.addWidget(abstract_label)
        
        header_layout.addStretch(1)
        
        # PDF-Import Button mit Icon und Farbe
        self.pdf_button = QPushButton("PDF importieren")
        self.pdf_button.setToolTip("Text aus einer PDF-Datei importieren")
        self.pdf_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {secondary_color};
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: #5a9840;
            }}
        """)
        self.pdf_button.setMaximumWidth(150)
        self.pdf_button.clicked.connect(self.import_pdf)
        header_layout.addWidget(self.pdf_button)
        
        input_layout.addLayout(header_layout)

        # PDF-Metadaten und Vorschau (initial unsichtbar)
        self.pdf_info_frame = QFrame()
        self.pdf_info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.pdf_info_frame.setStyleSheet(f"background-color: #e8f0fe; border-radius: 6px; padding: 8px;")
        self.pdf_info_frame.setVisible(False)
        
        pdf_info_layout = QHBoxLayout(self.pdf_info_frame)
        
        # Metadaten-Bereich
        self.pdf_metadata_label = QLabel()
        self.pdf_metadata_label.setWordWrap(True)
        pdf_info_layout.addWidget(self.pdf_metadata_label, 3)
        
        # Vorschaubild
        self.pdf_preview_label = QLabel()
        self.pdf_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pdf_preview_label.setMinimumSize(100, 100)
        self.pdf_preview_label.setMaximumSize(120, 160)
        pdf_info_layout.addWidget(self.pdf_preview_label, 0)
        
        # PDF schließen Button
        self.close_pdf_button = QPushButton("×")
        self.close_pdf_button.setToolTip("PDF-Informationen ausblenden")
        self.close_pdf_button.setFixedSize(24, 24)
        self.close_pdf_button.setStyleSheet("""
            QPushButton {
                background-color: #dddddd;
                border-radius: 12px;
                font-weight: bold;
                font-size: 16px;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #cccccc;
            }
        """)
        self.close_pdf_button.clicked.connect(lambda: self.pdf_info_frame.setVisible(False))
        pdf_info_layout.addWidget(self.close_pdf_button, 0, Qt.AlignmentFlag.AlignTop)
        
        input_layout.addWidget(self.pdf_info_frame)
        
        # Abstract-Textfeld
        self.abstract_edit = QTextEdit()
        self.abstract_edit.setPlaceholderText("Geben Sie hier den zu analysierenden Text ein oder importieren Sie eine PDF...")
        self.abstract_edit.textChanged.connect(self.update_input)
        self.abstract_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.abstract_edit.setMinimumHeight(120)
        input_layout.addWidget(self.abstract_edit)

        # Keywords-Eingabe mit flexiblem Layout
        keywords_header = QHBoxLayout()
        
        keywords_label = QLabel("Vorhandene Keywords (optional):" if not self.need_keywords 
                             else "Es müssen zwingend OGND-Keywords angebeben werden:")
        keywords_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        keywords_header.addWidget(keywords_label)
        
        # Keywords aus PDF extrahieren Button (nur sichtbar, wenn PDF geladen ist)
        self.extract_keywords_btn = QPushButton("Keywords aus PDF")
        self.extract_keywords_btn.setToolTip("Keywords aus den PDF-Metadaten extrahieren")
        self.extract_keywords_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {accent_color};
                color: #333;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: #e1b222;
            }}
        """)
        self.extract_keywords_btn.setMaximumWidth(140)
        self.extract_keywords_btn.setVisible(False)
        self.extract_keywords_btn.clicked.connect(self.extract_keywords_from_pdf)
        keywords_header.addWidget(self.extract_keywords_btn)
        
        input_layout.addLayout(keywords_header)
        
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setPlaceholderText("Fügen Sie hier bereits vorhandene Keywords ein ...")
        self.keywords_edit.textChanged.connect(self.update_input)
        self.keywords_edit.setMaximumHeight(80)
        input_layout.addWidget(self.keywords_edit)
        
        # Füge den Eingabebereich zum Layout hinzu
        collapsible_layout.addWidget(self.input_group)
        
        # ======== Prompt-Bereich (als ausklappbares Element) ========
        self.prompt_frame = QFrame()
        prompt_layout = QVBoxLayout(self.prompt_frame)
        prompt_layout.setSpacing(4)
        
        # Header mit Toggle-Button
        prompt_header = QHBoxLayout()
        
        self.show_prompt_btn = QPushButton("Prompt anzeigen ▼")
        self.show_prompt_btn.setCheckable(True)
        self.show_prompt_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #555;
                font-weight: normal;
                border: 1px solid #ddd;
                padding: 4px 8px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
            }
        """)
        
        prompt_header.addWidget(self.show_prompt_btn)
        prompt_header.addStretch(1)
        
        prompt_layout.addLayout(prompt_header)
        
        # Prompt-Textfeld (initial ausgeblendet)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Generierter Prompt wird hier angezeigt...")
        self.prompt.setReadOnly(True)
        self.prompt.setMaximumHeight(150)
        self.prompt.setVisible(False)
        prompt_layout.addWidget(self.prompt)
        
        # Connect toggle-Button zum Ein-/Ausblenden des Prompts
        self.show_prompt_btn.toggled.connect(lambda checked: self.prompt.setVisible(checked))
        self.show_prompt_btn.toggled.connect(lambda checked: self.show_prompt_btn.setText("Prompt ausblenden ▲" if checked else "Prompt anzeigen ▼"))
        
        collapsible_layout.addWidget(self.prompt_frame)

        # ======== KI-Konfiguration ========
        self.config_group = QGroupBox("KI-Konfiguration")
        config_grid = QGridLayout(self.config_group)
        config_grid.setSpacing(10)
        config_grid.setContentsMargins(10, 20, 10, 10)

        # Provider und Modell in einer Zeile
        config_grid.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        config_grid.addWidget(self.provider_combo, 0, 1)

        config_grid.addWidget(QLabel("Modell:"), 0, 2)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(self.set_model)
        config_grid.addWidget(self.model_combo, 0, 3, 1, 2)

        # Temperatur-Slider
        self.temperature_label = QLabel("Temperatur: 0.00")
        config_grid.addWidget(self.temperature_label, 1, 0)
        
        self.ki_temperature = QSlider(Qt.Orientation.Horizontal)
        self.ki_temperature.setRange(0, 100)
        self.ki_temperature.setValue(0)
        self.ki_temperature.setTickInterval(10)
        self.ki_temperature.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ki_temperature.valueChanged.connect(self.update_temperature_label)
        config_grid.addWidget(self.ki_temperature, 1, 1, 1, 3)

        # Seed-Eingabe
        config_grid.addWidget(QLabel("Seed:"), 1, 4)
        self.ki_seed = QSpinBox()
        self.ki_seed.setRange(0, 1000000000)
        self.ki_seed.setValue(1)
        self.ki_seed.setToolTip("0 = zufällig, andere Werte für reproduzierbare Ergebnisse")
        config_grid.addWidget(self.ki_seed, 1, 5)

        collapsible_layout.addWidget(self.config_group)

        # ======== Button-Bereich ========
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # Analyse-Button
        self.analyze_button = QPushButton("Analyse starten")
        self.analyze_button.setToolTip("Startet die KI-gestützte Analyse des Textes")
        self.analyze_button.setMinimumHeight(40)
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)
        
        # Zurücksetzen-Button
        self.clear_button = QPushButton("Zurücksetzen")
        self.clear_button.setToolTip("Löscht alle Eingaben und Ergebnisse")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #333;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
        self.clear_button.clicked.connect(self.clear_fields)
        button_layout.addWidget(self.clear_button)
        
        collapsible_layout.addLayout(button_layout)
        
        # Füge den Collapsible Container zum Hauptlayout hinzu
        main_layout.addWidget(self.collapsible_container)
        
        # ======== Trennlinie zwischen Eingabe und Ergebnisbereich ========
        self.separator_frame = QFrame()
        self.separator_frame.setFrameShape(QFrame.Shape.HLine)
        self.separator_frame.setStyleSheet("""
            QFrame {
                background-color: #cccccc;
                height: 2px;
                margin: 8px 0px;
            }
        """)
        main_layout.addWidget(self.separator_frame)

        # ======== Ergebnisbereich ========
        self.results_group = QGroupBox("Analyseergebnis")
        results_layout = QVBoxLayout(self.results_group)
        results_layout.setSpacing(8)
        results_layout.setContentsMargins(10, 20, 10, 10)

        self.result_splitter = QSplitter(Qt.Orientation.Horizontal)
        results_layout.addWidget(self.result_splitter)
        
        # Hauptergebnisbereich
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setMinimumHeight(200)
        self.results_edit.setPlaceholderText("Hier erscheinen die Analyseergebnisse...")
        self.results_edit.setStyleSheet("""
            QTextEdit {
                font-size: 12pt;
                line-height: 1.3;
            }
        """)
        self.result_splitter.addWidget(self.results_edit)

        # Verlaufsbereich
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(0, 0, 0, 0)
        
        history_label = QLabel("Verlauf:")
        history_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        history_layout.addWidget(history_label)
        
        self.results_list = QListWidget()
        self.results_list.setToolTip("Klicken Sie auf einen Eintrag, um das Ergebnis anzuzeigen")
        history_layout.addWidget(self.results_list)
        
        history_buttons = QHBoxLayout()
        self.copy_button = QPushButton("Kopieren")
        self.copy_button.setToolTip("Kopiert das ausgewählte Ergebnis in die Zwischenablage")
        self.copy_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {secondary_color};
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: #5a9840;
            }}
        """)
        self.copy_button.clicked.connect(self.copy_result)
        history_buttons.addWidget(self.copy_button)
        
        self.delete_button = QPushButton("Löschen")
        self.delete_button.setToolTip("Löscht den ausgewählten Verlaufseintrag")
        self.delete_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.delete_button.clicked.connect(self.delete_result)
        history_buttons.addWidget(self.delete_button)
        
        history_layout.addLayout(history_buttons)
        
        # Einstellungen für Verlaufsbereich
        history_widget.setMaximumWidth(200)
        self.result_splitter.addWidget(history_widget)
        
        # Splitter-Proportionen setzen
        self.result_splitter.setSizes([700, 300])
        
        main_layout.addWidget(self.results_group, 1)  # Mit Stretch-Faktor 1

        # Verbinde Signale
        self.results_list.itemClicked.connect(self.show_result)
        
        # Initial models update
        self.update_models(self.provider_combo.currentText())

    def toggle_compact_mode(self, force_state=None):
        """
        Wechselt zwischen kompaktem Modus (nur Ergebnis) und vollständigem Modus.
        
        Args:
            force_state: Optional - wenn angegeben, wird der Zustand auf diesen Wert gesetzt
                        (True für kompakt, False für vollständig)
        """
        # Wenn ein Zustand erzwungen wird, diesen setzen
        if force_state is not None:
            # Wenn Zustand bereits gleich ist, nichts tun
            if self.is_compact_mode == force_state:
                return
            self.is_compact_mode = force_state
        else:
            # Sonst Zustand umschalten
            self.is_compact_mode = not self.is_compact_mode
        
        # Button-Text aktualisieren
        if self.is_compact_mode:
            self.toggle_sections_btn.setText("▼ Eingabebereich")
        else:
            self.toggle_sections_btn.setText("▲ Eingabebereich")
        
        # Einfaches Ein-/Ausblenden ohne Animation
        self.collapsible_container.setVisible(not self.is_compact_mode)
        
        # Fokussierung anpassen
        if self.is_compact_mode:
            self.results_edit.setFocus()
        else:
            self.abstract_edit.setFocus()

    def set_model(self, model_text):
        """Set the current model and update configuration."""
        # Strip leading whitespace from model name (from indentation in combobox)
        model = model_text.strip()
        
        # Skip header items
        if model.startswith("↳"):
            return
            
        self.logger.info(f"Setting model to: '{model}'")
        
        try:
            config = self.promptmanager.get_prompt_config(self.task, model)
            self.logger.info(f"Prompt config for {model}: {config}")
            
            # Update temperature slider
            temp_value = int(config.get("temp", 0.7) * 100)
            self.ki_temperature.setValue(temp_value)
            
            # Save template and system prompt
            self.current_template = config.get("prompt", "")
            self.system = config.get("system", "")
            
            # Update UI
            self.set_input()
        except Exception as e:
            self.logger.error(f"Error setting model config: {e}")

    def update_temperature_label(self, value):
        """Update temperature label to show actual value."""
        self.temperature_label.setText(f"Temperatur: {value/100:.2f}")

    def import_pdf(self):
        """Import text from a PDF file using PyPDF2."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "PDF-Datei öffnen",
            "",
            "PDF-Dateien (*.pdf)"
        )
        
        if not file_path:
            return
            
        self.pdf_path = file_path
        
        # UI-Feedback
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat("Verarbeite PDF...")
        self.progress_bar.setRange(0, 0)
        
        # Direktes Verarbeiten der PDF ohne Thread
        try:
            # PDF öffnen
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Metadaten extrahieren
                info = pdf_reader.metadata
                metadata = {
                    'title': info.title if info and hasattr(info, 'title') else '',
                    'author': info.author if info and hasattr(info, 'author') else '',
                    'subject': info.subject if info and hasattr(info, 'subject') else '',
                    'keywords': info.keywords if info and hasattr(info, 'keywords') else '',
                    'pages': len(pdf_reader.pages),
                    'filename': os.path.basename(self.pdf_path)
                }
                
                # Text extrahieren (bis zu maximal 5 Seiten oder 5000 Zeichen)
                text = ""
                max_pages = min(5, len(pdf_reader.pages))
                
                for i in range(max_pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    text += page_text + "\n\n"
                    
                    if len(text) > 5000:
                        text = text[:5000] + "...[gekürzt]"
                        break
            
            # UI mit extrahiertem Text aktualisieren
            self.handle_pdf_content(text, metadata)
            
        except Exception as e:
            self.handle_error("Fehler bei der PDF-Verarbeitung", str(e))
            
        # UI-Feedback zurücksetzen
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)

    def handle_pdf_content(self, text, metadata):
        """Handle extracted PDF content."""
        self.pdf_metadata = metadata
        
        # Metadaten-Text formatieren
        metadata_text = f"<b>PDF-Informationen:</b><br>"
        if metadata.get('title'):
            metadata_text += f"<b>Titel:</b> {metadata['title']}<br>"
            
        if metadata.get('author'):
            metadata_text += f"<b>Autor:</b> {metadata['author']}<br>"
            
        metadata_text += f"<b>Seiten:</b> {metadata.get('pages', '?')}<br>"
        metadata_text += f"<b>Datei:</b> {os.path.basename(self.pdf_path)}"
        
        # UI aktualisieren
        self.pdf_metadata_label.setText(metadata_text)
        self.pdf_info_frame.setVisible(True)
        self.abstract_edit.setPlainText(text)
        
        # Keywords-Button aktivieren, falls Keywords verfügbar sind
        has_keywords = bool(metadata.get('keywords') or metadata.get('subject'))
        self.extract_keywords_btn.setVisible(has_keywords)

    def extract_keywords_from_pdf(self):
        """Extract keywords from PDF metadata."""
        if not self.pdf_metadata:
            return
            
        keywords = []
        
        # Aus Keywords-Feld extrahieren
        if self.pdf_metadata.get('keywords'):
            kw = self.pdf_metadata.get('keywords', '')

            # Wenn es ein String ist
            if isinstance(kw, str):
                # Verschiedene Trennzeichen probieren
                for separator in [',', ';', '\n', ' - ']:
                    if separator in kw:
                        keywords.extend([k.strip() for k in kw.split(separator) if k.strip()])
                        break
                else:
                    # Falls kein Separator gefunden wurde
                    keywords.append(kw.strip())
                    
        # Aus Subject-Feld extrahieren
        if self.pdf_metadata.get('subject'):
            subj = self.pdf_metadata.get('subject', '')
            
            # Wenn es ein String ist
            if isinstance(subj, str):
                # Verschiedene Trennzeichen probieren
                for separator in [',', ';', '\n', ' - ']:
                    if separator in subj:
                        keywords.extend([k.strip() for k in subj.split(separator) if k.strip()])
                        break
                else:
                    # Falls kein Separator gefunden wurde
                    keywords.append(subj.strip())
                    
        # Duplikate entfernen
        unique_keywords = list(set(keywords))
        
        if not unique_keywords:
            QMessageBox.information(self, "Keine Keywords", 
                              "Es konnten keine Keywords aus den PDF-Metadaten extrahiert werden.")
            return
            
        # Keywords in Textfeld einfügen
        current_kws = self.keywords_edit.toPlainText()
        
        if current_kws.strip():
            # Zu vorhandenen Keywords hinzufügen
            new_kws = current_kws + "\n" + "\n".join(unique_keywords)
            self.keywords_edit.setPlainText(new_kws)
        else:
            # Neue Keywords einfügen
            self.keywords_edit.setPlainText("\n".join(unique_keywords))
            
        QMessageBox.information(self, "Keywords importiert",
                          f"{len(unique_keywords)} Keywords wurden aus den PDF-Metadaten extrahiert.")

    def start_analysis(self):
        """Start the analysis of the abstract."""
        # Validierung
        if not self.abstract_edit.toPlainText().strip():
            self.handle_error("Fehlende Eingabe", "Bitte geben Sie einen Text ein.")
            return
            
        # Eingabebereich einklappen
        self.toggle_compact_mode(force_state=True)
        
        # UI vorbereiten
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # Abbruch-Button aktivieren und Farbe ändern
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;  /* Rot für aktiven Cancel-Button */
                color: white;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)

        # Bisherige Ergebnisse löschen
        self.results_edit.clear()
        self.generated_response = ""
        
        # Model ohne Leerzeichen
        model = self.model_combo.currentText().strip()
        
        # LLM-Anfrage direkt starten (kein Thread mehr)
        try:
            self.llm.generate_response(
                provider=self.provider_combo.currentText(),
                model=model,
                prompt=self.prompt.toPlainText(),
                temperature=self.ki_temperature.value() / 100,
                seed=self.ki_seed.value() if self.ki_seed.value() > 0 else None,
                system=self.system,
                stream=True  # Immer mit Streaming
            )
            # Die Antwort wird über die verbundenen Signals empfangen
        except Exception as e:
            self.handle_error("Fehler bei der Anfrage", str(e))
            self.analysis_completed()

    def on_text_received(self, text_chunk):
        """Empfängt Text-Chunks während des Streamings."""
        # Füge Text zum Ergebnis hinzu
        cursor = self.results_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text_chunk)
        self.results_edit.setTextCursor(cursor)
        
        # Stelle sicher, dass Text sichtbar ist
        self.results_edit.ensureCursorVisible()
        
        # Gesamte Antwort aktualisieren
        self.generated_response += text_chunk
        
        # UI-Update erzwingen
        QApplication.processEvents()

    def on_generation_finished(self, message):
        """Wird aufgerufen, wenn die Generierung abgeschlossen ist."""
        self.logger.info(f"Generation finished: {message}")
        
        # Zum Verlauf hinzufügen
        item = QListWidgetItem(f"{self.provider_combo.currentText()}: {self.model_combo.currentText().strip()}")
        item.setToolTip(self.prompt.toPlainText())
        item.setData(Qt.ItemDataRole.UserRole, self.generated_response)
        self.results_list.addItem(item)
        self.results_list.setCurrentItem(item)
        
        # Extrahiere Keywords aus der Antwort
        try:
            keywords = self.extract_keywords(self.generated_response)
            self.keywords_extracted.emit(keywords)
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
        
        # UI zurücksetzen
        self.analysis_completed()

    def on_generation_error(self, error_message):
        """Wird aufgerufen, wenn ein Fehler während der Generierung auftritt."""
        self.handle_error("Generierungsfehler", error_message)
        self.analysis_completed()

    def on_generation_cancelled(self):
        """Wird aufgerufen, wenn die Generierung abgebrochen wurde."""
        self.logger.info("Generation cancelled")
        
        # Abbruch-Text anzeigen
        if self.generated_response:
            self.results_edit.setPlainText(f"{self.generated_response}\n\n[ABGEBROCHEN]")
        else:
            self.results_edit.setPlainText("[ABGEBROCHEN]")
        
        # UI zurücksetzen
        self.analysis_completed()

    def cancel_analysis(self):
        """Bricht die laufende Analyse ab."""
        # Aktuelle Farbe prüfen, um zu sehen, ob generiert wird
        current_style = self.cancel_button.styleSheet()
        if "background-color: #e74c3c" in current_style:  # Aktiver Zustand
            try:
                # Abbruch bei der LLM-Interface anfordern
                self.llm.cancel_generation(reason="user_requested")
                self.logger.info("Cancellation requested")
            except Exception as e:
                self.logger.error(f"Error canceling generation: {e}")
                self.analysis_completed()

    def analysis_completed(self):
        """Bereinigt nach Abschluss der Analyse."""
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)
        
        # Abbruch-Button zurück auf inaktiven Stil setzen
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #cccccc;  /* Inaktiv-Grau */
                color: #666666;
            }
        """)

    def update_input(self):
        """Aktualisiert den Prompt, wenn sich die Eingabe ändert."""
        self.prompt.setPlainText(self.set_input())
        self.abstract_changed.emit(self.abstract_edit.toPlainText().strip())

    def set_input(self):
        """
        Bereitet den Prompt unter Verwendung des Templates und aktueller Eingabewerte vor.
        
        Returns:
            str: Der formatierte Prompt.
        """
        template = self.current_template
        self.logger.debug(f"Preparing prompt with template: {template}")
          
        # Variablen vorbereiten
        variables = {
            "abstract": self.abstract_edit.toPlainText().strip(),
            "keywords": self.keywords_edit.toPlainText().strip() if self.keywords_edit.toPlainText().strip() else "Keine Keywords vorhanden"
        }
        
        try:
            # Prompt erstellen
            prompt = template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing variable in template: {e}")
            return f"Fehler im Template: Variable {e} fehlt"
            
        return prompt    

    def extract_keywords(self, text):
        """
        Extrahiert Keywords aus dem Antworttext mit Regex.
        
        Args:
            text: Der zu durchsuchende Text
            
        Returns:
            str: Komma-getrennte Keywords in Anführungszeichen
        """
        # Versuche, Keywords zwischen <final_list> Tags zu finden
        match = re.search(r'<final_list>(.*?)</final_list>', text, re.DOTALL)
        if not match:
            self.logger.warning("No <final_list> tags found in response")
            return ""
            
        keywords = match.group(1).split("|")
        self.logger.info(f"Extracted keywords: {keywords}")
        
        # Formatiere Keywords mit Anführungszeichen
        quoted_keywords = [f'"{keyword.strip()}"' for keyword in keywords if keyword.strip()]
        result = ', '.join(quoted_keywords)
        
        # Sende Signal mit Ergebnis
        self.final_list.emit(result)
        return result
    
    def handle_error(self, title: str, message: str):
        """Zeigt einen Fehlerdialog an."""
        QMessageBox.critical(self, title, message)
        self.logger.error(f"{title}: {message}")

    def set_ui_enabled(self, enabled: bool):
        """Aktiviert/deaktiviert UI-Elemente während der Verarbeitung."""
        self.analyze_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.abstract_edit.setEnabled(enabled)
        self.keywords_edit.setEnabled(enabled)
        self.provider_combo.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.ki_temperature.setEnabled(enabled)
        self.ki_seed.setEnabled(enabled)
        self.pdf_button.setEnabled(enabled)

    def clear_fields(self):
        """Setzt alle Felder zurück."""
        if self.abstract_edit.toPlainText() or self.keywords_edit.toPlainText() or self.results_edit.toPlainText():
            reply = QMessageBox.question(
                self,
                "Bestätigung",
                "Möchten Sie wirklich alle Felder zurücksetzen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.abstract_edit.clear()
                self.keywords_edit.clear()
                self.results_edit.clear()
                self.pdf_info_frame.setVisible(False)
                self.pdf_path = None
                self.extract_keywords_btn.setVisible(False)

    def copy_result(self):
        """Kopiert das aktuelle Ergebnis in die Zwischenablage."""
        # Aktuelles Element holen
        item = self.results_list.currentItem()
        if not item:
            return
            
        # Elementdaten holen
        response = item.data(Qt.ItemDataRole.UserRole)
        
        # Versuche, Keywords zu extrahieren, sonst vollständigen Text verwenden
        try:
            text_to_copy = self.extract_keywords(response)
            if not text_to_copy:
                text_to_copy = response
        except:
            text_to_copy = response
            
        # In Zwischenablage kopieren
        clipboard = QApplication.clipboard()
        clipboard.setText(text_to_copy)
        
        # Bestätigung mit Ausblendeffekt anzeigen
        QToolTip.showText(self.copy_button.mapToGlobal(QPoint(0, -30)), 
                         "In die Zwischenablage kopiert", self.copy_button, QRect(), 2000)

    def delete_result(self):
        """Löscht das ausgewählte Ergebnis aus dem Verlauf."""
        # Aktuelle Zeile holen
        current_row = self.results_list.currentRow()
        if current_row == -1:
            return
            
        # Bestätigung anfordern
        reply = QMessageBox.question(
            self,
            "Bestätigung",
            "Möchten Sie diesen Eintrag wirklich löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Element entfernen
            self.results_list.takeItem(current_row)
            
            # Ergebnisbereich leeren, wenn es das letzte Element war
            if self.results_list.count() == 0:
                self.results_edit.clear()

    def setup_animations(self):
        """Konfiguriert Animationen für Ein- und Ausblendeffekte."""
        self.container_animation = QPropertyAnimation(self.collapsible_container, b"maximumHeight")
        self.container_animation.setDuration(300)  # 300 ms für die Animation
        self.container_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def show_result(self, item):
        """Zeigt das ausgewählte Ergebnis aus dem Verlauf an."""
        if item:
            response = item.data(Qt.ItemDataRole.UserRole)
            self.results_edit.setPlainText(response)
            
            try:
                keywords = self.extract_keywords(response)
                self.keywords_extracted.emit(keywords)
            except Exception as e:
                self.logger.error(f"Error extracting keywords from history: {e}")

    def set_keywords(self, keywords):
        """Setzt die Keywords im Eingabefeld."""
        self.keywords_edit.setPlainText(keywords)
        self.update_input()

    def set_abstract(self, abstract):
        """Setzt den Abstract im Eingabefeld."""
        self.abstract_edit.setPlainText(abstract)
        self.update_input()
        
    def prompt_generated(self, prompt):
        """Setzt den Prompt-Text."""
        self.prompt.setPlainText(prompt)
