"""
Pipeline Configuration Dialog - Konfiguration für Pipeline-Schritte
Claude Generated - Ermöglicht die Konfiguration von Provider und Modellen für jeden Pipeline-Schritt
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QTextEdit,
    QTabWidget,
    QWidget,
    QMessageBox,
    QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QMetaObject, Q_ARG
from PyQt6.QtGui import QFont
from typing import Dict, List, Any, Optional
import json
import logging

from ..core.pipeline_manager import PipelineConfig
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService


class SearchStepConfigWidget(QWidget):
    """Widget für die Konfiguration des GND-Suchschritts - Claude Generated"""

    def __init__(self, step_name: str, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.step_id = "search"
        self.setup_ui()

    def setup_ui(self):
        """Setup der UI für Search-Konfiguration - Claude Generated"""
        layout = QVBoxLayout(self)

        # Step Name Header
        header_label = QLabel(self.step_name)
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Suggester Selection
        suggester_group = QGroupBox("Suchprovider")
        suggester_layout = QVBoxLayout(suggester_group)

        # Available suggesters
        self.lobid_checkbox = QCheckBox("Lobid (Deutsche Nationalbibliothek)")
        self.lobid_checkbox.setChecked(True)
        suggester_layout.addWidget(self.lobid_checkbox)

        self.swb_checkbox = QCheckBox("SWB (Südwestdeutscher Bibliotheksverbund)")
        self.swb_checkbox.setChecked(True)
        suggester_layout.addWidget(self.swb_checkbox)

        self.catalog_checkbox = QCheckBox("Lokaler Katalog")
        self.catalog_checkbox.setChecked(False)
        suggester_layout.addWidget(self.catalog_checkbox)

        layout.addWidget(suggester_group)

        # Enable/Disable for this step
        self.enabled_checkbox = QCheckBox("Schritt aktivieren")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        suggester_layout.addWidget(self.enabled_checkbox)

        layout.addStretch()

    def on_enabled_changed(self, enabled: bool):
        """Enable/disable step configuration - Claude Generated"""
        for widget in self.findChildren(QWidget):
            if widget != self.enabled_checkbox:
                widget.setEnabled(enabled)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration - Claude Generated"""
        suggesters = []
        if self.lobid_checkbox.isChecked():
            suggesters.append("lobid")
        if self.swb_checkbox.isChecked():
            suggesters.append("swb")
        if self.catalog_checkbox.isChecked():
            suggesters.append("catalog")

        return {
            "step_id": self.step_id,
            "enabled": self.enabled_checkbox.isChecked(),
            "suggesters": suggesters,
        }

    def set_config(self, config: Dict[str, Any]):
        """Set configuration - Claude Generated"""
        if "enabled" in config:
            self.enabled_checkbox.setChecked(config["enabled"])

        if "suggesters" in config:
            suggesters = config["suggesters"]
            self.lobid_checkbox.setChecked("lobid" in suggesters)
            self.swb_checkbox.setChecked("swb" in suggesters)
            self.catalog_checkbox.setChecked("catalog" in suggesters)


class PipelineStepConfigWidget(QWidget):
    """Widget für die Konfiguration eines Pipeline-Schritts - Claude Generated"""

    def __init__(
        self,
        step_name: str,
        step_id: str,
        llm_service: LlmService,
        prompt_service: PromptService = None,
        parent=None,
    ):
        super().__init__(parent)
        self.step_name = step_name
        self.step_id = step_id
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.logger = logging.getLogger(__name__)
        self.setup_ui()

    def setup_ui(self):
        """Setup der UI für Step-Konfiguration - Claude Generated"""
        layout = QVBoxLayout(self)

        # Step Name Header
        header_label = QLabel(self.step_name)
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Provider & Model Selection
        provider_group = QGroupBox("LLM-Einstellungen")
        provider_layout = QGridLayout(provider_group)

        # Provider Dropdown
        provider_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        providers = self.llm_service.get_available_providers()
        self.provider_combo.addItems(providers)
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(self.provider_combo, 0, 1)

        # Model Dropdown
        provider_layout.addWidget(QLabel("Modell:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.load_prompt_settings)
        provider_layout.addWidget(self.model_combo, 1, 1)

        # Task Selection (for LLM steps)
        if self.step_id in ["initialisation", "keywords"]:
            provider_layout.addWidget(QLabel("Task:"), 2, 0)
            self.task_combo = QComboBox()

            # Define available tasks based on step
            if self.step_id == "initialisation":
                # Initial keyword extraction tasks
                available_tasks = ["initialisation", "keywords", "rephrase"]
            else:  # keywords step (final analysis)
                # Final keyword analysis tasks
                available_tasks = ["keywords", "rephrase", "keywords_chunked"]

            self.task_combo.addItems(available_tasks)
            self.task_combo.currentTextChanged.connect(self.on_task_changed)
            provider_layout.addWidget(self.task_combo, 2, 1)

        # Enable/Disable for this step
        self.enabled_checkbox = QCheckBox("Schritt aktivieren")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        provider_layout.addWidget(self.enabled_checkbox, 3, 0, 1, 2)

        layout.addWidget(provider_group)

        # Parameter Settings
        params_group = QGroupBox("Parameter")
        params_layout = QGridLayout(params_group)

        # Temperature
        params_layout.addWidget(QLabel("Temperatur:"), 0, 0)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_spinbox.setValue(v / 100.0)
        )
        params_layout.addWidget(self.temp_slider, 0, 1)

        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.0, 1.0)
        self.temp_spinbox.setValue(0.7)
        self.temp_spinbox.setDecimals(2)
        self.temp_spinbox.setSingleStep(0.01)
        self.temp_spinbox.valueChanged.connect(
            lambda v: self.temp_slider.setValue(int(v * 100))
        )
        params_layout.addWidget(self.temp_spinbox, 0, 2)

        # Top-P
        params_layout.addWidget(QLabel("Top-P:"), 1, 0)
        self.p_slider = QSlider(Qt.Orientation.Horizontal)
        self.p_slider.setRange(0, 100)
        self.p_slider.setValue(10)
        self.p_slider.valueChanged.connect(lambda v: self.p_spinbox.setValue(v / 100.0))
        params_layout.addWidget(self.p_slider, 1, 1)

        self.p_spinbox = QDoubleSpinBox()
        self.p_spinbox.setRange(0.0, 1.0)
        self.p_spinbox.setValue(0.1)
        self.p_spinbox.setDecimals(2)
        self.p_spinbox.setSingleStep(0.01)
        self.p_spinbox.valueChanged.connect(
            lambda v: self.p_slider.setValue(int(v * 100))
        )
        params_layout.addWidget(self.p_spinbox, 1, 2)

        layout.addWidget(params_group)

        # Keyword Chunking Parameters (only for keywords step)
        if self.step_id == "keywords":
            chunking_group = QGroupBox("Keyword Chunking")
            chunking_layout = QGridLayout(chunking_group)

            # Chunking Threshold
            chunking_layout.addWidget(QLabel("Chunking-Schwellwert:"), 0, 0)
            self.chunking_threshold_spinbox = QSpinBox()
            self.chunking_threshold_spinbox.setRange(100, 2000)
            self.chunking_threshold_spinbox.setValue(500)
            self.chunking_threshold_spinbox.setSuffix(" Keywords")
            self.chunking_threshold_spinbox.setToolTip(
                "Anzahl Keywords ab der Chunking aktiviert wird"
            )
            chunking_layout.addWidget(self.chunking_threshold_spinbox, 0, 1)

            # Chunking Task
            chunking_layout.addWidget(QLabel("Chunking-Task:"), 1, 0)
            self.chunking_task_combo = QComboBox()
            self.chunking_task_combo.addItems(["keywords_chunked", "rephrase"])
            self.chunking_task_combo.setCurrentText("keywords_chunked")
            self.chunking_task_combo.setToolTip("Task für Chunk-Verarbeitung")
            chunking_layout.addWidget(self.chunking_task_combo, 1, 1)

            layout.addWidget(chunking_group)

        # Debug: Add test button to load prompt settings
        if self.step_id in ["initialisation", "keywords"]:
            test_button = QPushButton("Test: Load Prompt Settings")
            test_button.clicked.connect(self.test_load_prompt_settings)
            layout.addWidget(test_button)

        # Custom Prompt (if applicable)
        if self.step_id in ["initialisation", "keywords", "classification"]:
            prompt_group = QGroupBox("Custom Prompts (optional)")
            prompt_layout = QVBoxLayout(prompt_group)

            # Main prompt
            prompt_layout.addWidget(QLabel("Haupt-Prompt:"))
            self.custom_prompt = QTextEdit()
            self.custom_prompt.setMaximumHeight(80)
            self.custom_prompt.setPlaceholderText("Leer lassen für Standard-Prompt...")
            prompt_layout.addWidget(self.custom_prompt)

            # System prompt
            prompt_layout.addWidget(QLabel("System-Prompt:"))
            self.system_prompt = QTextEdit()
            self.system_prompt.setMaximumHeight(60)
            self.system_prompt.setPlaceholderText(
                "Leer lassen für Standard-System-Prompt..."
            )
            prompt_layout.addWidget(self.system_prompt)

            layout.addWidget(prompt_group)

        # Initialize models for default provider and load initial prompt settings
        self.on_provider_changed(self.provider_combo.currentText())

        # Load initial prompt settings after UI is fully set up
        if hasattr(self, "task_combo"):
            # Try immediate loading first
            self.load_prompt_settings()
            # Also set a timer as backup
            QTimer.singleShot(100, self.load_prompt_settings)

    def on_provider_changed(self, provider: str):
        """Handle provider change - Claude Generated"""
        self.model_combo.clear()
        models = self.llm_service.get_available_models(provider)
        if models:
            self.model_combo.addItems(models)
        # Load prompt settings when provider changes
        self.load_prompt_settings()

    def on_task_changed(self, task: str):
        """Handle task change - load settings from prompts.json - Claude Generated"""
        self.load_prompt_settings()

    def load_prompt_settings(self):
        """Load prompt settings from prompts.json for current model/task - Claude Generated"""
        if not self.prompt_service:
            self.logger.warning("No prompt_service available for loading settings")
            return
        if not hasattr(self, "task_combo"):
            self.logger.warning("No task_combo available for loading settings")
            return

        try:
            current_task = self.task_combo.currentText()
            current_model = self.model_combo.currentText()

            self.logger.info(
                f"Loading prompt settings for task='{current_task}', model='{current_model}'"
            )
            self.logger.info(
                f"PromptService available: {self.prompt_service is not None}"
            )

            if current_task and current_model:
                # Get prompt config for this task and model
                prompt_config = self.prompt_service.get_prompt_config(
                    current_task, current_model
                )
                self.logger.info(f"Found prompt config: {prompt_config is not None}")

                if prompt_config:
                    self.logger.info(
                        f"Config details: temp={getattr(prompt_config, 'temp', 'N/A')}, p_value={getattr(prompt_config, 'p_value', 'N/A')}"
                    )

                if prompt_config:
                    # Update temperature and top_p from prompt config
                    if (
                        hasattr(prompt_config, "temp")
                        and prompt_config.temp is not None
                    ):
                        temp_value = float(prompt_config.temp)
                        self.logger.info(f"Setting temperature to {temp_value}")

                        # Use blockSignals to prevent recursion, more reliable than disconnect/reconnect
                        self.temp_spinbox.blockSignals(True)
                        self.temp_slider.blockSignals(True)

                        self.temp_spinbox.setValue(temp_value)
                        self.temp_slider.setValue(int(temp_value * 100))

                        self.temp_spinbox.blockSignals(False)
                        self.temp_slider.blockSignals(False)

                        # Force repaint to ensure UI is updated
                        self.temp_spinbox.repaint()
                        self.temp_slider.repaint()

                        # Also try processEvents to ensure immediate update
                        from PyQt6.QtWidgets import QApplication

                        QApplication.processEvents()

                        self.logger.info(
                            f"UI updated: spinbox={self.temp_spinbox.value()}, slider={self.temp_slider.value()}"
                        )

                    if (
                        hasattr(prompt_config, "p_value")
                        and prompt_config.p_value is not None
                    ):
                        p_value = float(prompt_config.p_value)
                        self.logger.info(f"Setting p_value to {p_value}")

                        # Use blockSignals to prevent recursion, more reliable than disconnect/reconnect
                        self.p_spinbox.blockSignals(True)
                        self.p_slider.blockSignals(True)

                        self.p_spinbox.setValue(p_value)
                        self.p_slider.setValue(int(p_value * 100))

                        self.p_spinbox.blockSignals(False)
                        self.p_slider.blockSignals(False)

                        # Force repaint to ensure UI is updated
                        self.p_spinbox.repaint()
                        self.p_slider.repaint()

                        # Also try processEvents to ensure immediate update
                        from PyQt6.QtWidgets import QApplication

                        QApplication.processEvents()

                        self.logger.info(
                            f"UI updated: p_spinbox={self.p_spinbox.value()}, p_slider={self.p_slider.value()}"
                        )

                    # Update custom prompt if available
                    if hasattr(self, "custom_prompt") and hasattr(
                        prompt_config, "prompt"
                    ):
                        # Don't overwrite if user has custom text, just show in placeholder
                        if not self.custom_prompt.toPlainText().strip():
                            self.custom_prompt.setPlaceholderText(
                                f"Standard-Prompt für {current_task}"
                            )

                    # Update system prompt if available
                    if hasattr(self, "system_prompt") and hasattr(
                        prompt_config, "system"
                    ):
                        # Don't overwrite if user has custom text, just show in placeholder
                        if not self.system_prompt.toPlainText().strip():
                            self.system_prompt.setPlaceholderText(
                                f"Standard-System-Prompt für {current_task}"
                            )

                    # Log the loaded settings for debugging
                    if hasattr(self, "logger"):
                        self.logger.info(
                            f"Loaded prompt settings for {current_task}/{current_model}: temp={getattr(prompt_config, 'temp', 'N/A')}, p={getattr(prompt_config, 'p_value', 'N/A')}"
                        )

        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Could not load prompt settings: {e}")

    def test_load_prompt_settings(self):
        """Test method to manually trigger prompt loading - Claude Generated"""
        self.logger.info("=== MANUAL TEST: Loading prompt settings ===")
        self.logger.info(
            f"Current values: temp={self.temp_spinbox.value()}, p={self.p_spinbox.value()}"
        )

        # Test 1: Force set test values to verify UI works
        self.logger.info("Test 1: Setting test values manually...")

        self.temp_spinbox.blockSignals(True)
        self.temp_slider.blockSignals(True)
        self.p_spinbox.blockSignals(True)
        self.p_slider.blockSignals(True)

        self.temp_spinbox.setValue(0.25)
        self.temp_slider.setValue(25)
        self.p_spinbox.setValue(0.1)
        self.p_slider.setValue(10)

        self.temp_spinbox.blockSignals(False)
        self.temp_slider.blockSignals(False)
        self.p_spinbox.blockSignals(False)
        self.p_slider.blockSignals(False)

        # Force UI update
        self.temp_spinbox.repaint()
        self.temp_slider.repaint()
        self.p_spinbox.repaint()
        self.p_slider.repaint()

        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()

        self.logger.info(
            f"After manual set: temp={self.temp_spinbox.value()}, p={self.p_spinbox.value()}"
        )

        # Test 2: Try loading from prompt service
        self.logger.info("Test 2: Loading from prompt service...")
        self.load_prompt_settings()

    def on_enabled_changed(self, enabled: bool):
        """Enable/disable step configuration - Claude Generated"""
        # Enable/disable all child widgets
        for widget in self.findChildren(QWidget):
            if widget != self.enabled_checkbox:
                widget.setEnabled(enabled)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration - Claude Generated"""
        config = {
            "step_id": self.step_id,
            "enabled": self.enabled_checkbox.isChecked(),
            "provider": self.provider_combo.currentText(),
            "model": self.model_combo.currentText(),
            "temperature": self.temp_spinbox.value(),
            "top_p": self.p_spinbox.value(),
        }

        # Add task if available
        if hasattr(self, "task_combo"):
            config["task"] = self.task_combo.currentText()

        # Add custom prompt if available - this overrides defaults from prompts.json
        if hasattr(self, "custom_prompt"):
            custom_text = self.custom_prompt.toPlainText().strip()
            if custom_text:
                config["prompt_template"] = custom_text
            # If no custom prompt, use the loaded prompt from prompts.json
            elif self.prompt_service and hasattr(self, "task_combo"):
                try:
                    current_task = self.task_combo.currentText()
                    current_model = self.model_combo.currentText()
                    if current_task and current_model:
                        prompt_config = self.prompt_service.get_prompt_config(
                            current_task, current_model
                        )
                        if prompt_config and hasattr(prompt_config, "prompt"):
                            config["prompt_template"] = prompt_config.prompt
                except Exception as e:
                    pass  # Fall back to default prompt

        # Add system prompt if available - this overrides defaults from prompts.json
        if hasattr(self, "system_prompt"):
            system_text = self.system_prompt.toPlainText().strip()
            if system_text:
                config["system_prompt"] = system_text
            # If no custom system prompt, use the loaded system prompt from prompts.json
            elif self.prompt_service and hasattr(self, "task_combo"):
                try:
                    current_task = self.task_combo.currentText()
                    current_model = self.model_combo.currentText()
                    if current_task and current_model:
                        prompt_config = self.prompt_service.get_prompt_config(
                            current_task, current_model
                        )
                        if prompt_config and hasattr(prompt_config, "system"):
                            config["system_prompt"] = prompt_config.system
                except Exception as e:
                    pass  # Fall back to default system prompt

        # Add keyword chunking parameters if available (keywords step only)
        if hasattr(self, "chunking_threshold_spinbox"):
            config["keyword_chunking_threshold"] = (
                self.chunking_threshold_spinbox.value()
            )
        if hasattr(self, "chunking_task_combo"):
            config["chunking_task"] = self.chunking_task_combo.currentText()

        return config

    def set_config(self, config: Dict[str, Any]):
        """Set configuration - Claude Generated"""
        if "enabled" in config:
            self.enabled_checkbox.setChecked(config["enabled"])

        if "provider" in config:
            index = self.provider_combo.findText(config["provider"])
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)

        if "model" in config:
            index = self.model_combo.findText(config["model"])
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

        if "temperature" in config:
            temp_value = config["temperature"]
            self.temp_spinbox.setValue(temp_value)
            # Update slider to match
            self.temp_slider.setValue(int(temp_value * 100))

        if "top_p" in config:
            p_value = config["top_p"]
            self.p_spinbox.setValue(p_value)
            # Update slider to match
            self.p_slider.setValue(int(p_value * 100))

        if "task" in config and hasattr(self, "task_combo"):
            index = self.task_combo.findText(config["task"])
            if index >= 0:
                self.task_combo.setCurrentIndex(index)

        if "prompt_template" in config and hasattr(self, "custom_prompt"):
            self.custom_prompt.setPlainText(config["prompt_template"])
        elif "custom_prompt" in config and hasattr(self, "custom_prompt"):
            self.custom_prompt.setPlainText(config["custom_prompt"])

        if "system_prompt" in config and hasattr(self, "system_prompt"):
            self.system_prompt.setPlainText(config["system_prompt"])

        # Set keyword chunking parameters if available (keywords step only)
        if "keyword_chunking_threshold" in config and hasattr(
            self, "chunking_threshold_spinbox"
        ):
            self.chunking_threshold_spinbox.setValue(
                config["keyword_chunking_threshold"]
            )

        if "chunking_task" in config and hasattr(self, "chunking_task_combo"):
            index = self.chunking_task_combo.findText(config["chunking_task"])
            if index >= 0:
                self.chunking_task_combo.setCurrentIndex(index)

        # Load prompt settings after config is set (with delay to ensure UI is updated)
        if hasattr(self, "task_combo") and not (
            "temperature" in config or "top_p" in config
        ):
            # Only load prompt settings if temperature/top_p weren't explicitly set in config
            QTimer.singleShot(50, self.load_prompt_settings)


class PipelineConfigDialog(QDialog):
    """Dialog für Pipeline-Konfiguration - Claude Generated"""

    config_saved = pyqtSignal(object)  # PipelineConfig

    def __init__(
        self,
        llm_service: LlmService,
        prompt_service: PromptService = None,
        current_config: Optional[PipelineConfig] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.current_config = current_config
        self.step_widgets = {}
        self.logger = logging.getLogger(__name__)
        self.setup_ui()

        if current_config:
            self.load_config(current_config)

    def setup_ui(self):
        """Setup der Dialog UI - Claude Generated"""
        self.setWindowTitle("Pipeline-Konfiguration")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)

        # Header
        header_label = QLabel("🚀 Pipeline-Konfiguration")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        description_label = QLabel(
            "Konfigurieren Sie Provider, Modelle und Parameter für jeden Pipeline-Schritt:"
        )
        description_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(description_label)

        # Main content with tabs for each step
        self.tab_widget = QTabWidget()

        # Define pipeline steps (using official step names from CLAUDE.md)
        pipeline_steps = [
            ("initialisation", "🔤 Initialisierung"),
            ("search", "🔍 Suche"),
            ("keywords", "✅ Schlagworte"),
            ("classification", "📚 Klassifikation"),
        ]

        # Create tab for each step
        for step_id, step_name in pipeline_steps:
            if step_id == "search":
                # Search step uses special SearchStepConfigWidget
                search_widget = SearchStepConfigWidget(step_name)
                self.step_widgets[step_id] = search_widget
                self.tab_widget.addTab(search_widget, step_name)
            else:
                step_widget = PipelineStepConfigWidget(
                    step_name, step_id, self.llm_service, self.prompt_service
                )
                self.step_widgets[step_id] = step_widget
                self.tab_widget.addTab(step_widget, step_name)

        layout.addWidget(self.tab_widget)

        # Global Settings
        global_group = QGroupBox("Globale Einstellungen")
        global_layout = QVBoxLayout(global_group)

        # Auto-advance option
        self.auto_advance_checkbox = QCheckBox("Automatisch zum nächsten Schritt")
        self.auto_advance_checkbox.setChecked(True)
        self.auto_advance_checkbox.setToolTip(
            "Pipeline läuft automatisch durch alle Schritte"
        )
        global_layout.addWidget(self.auto_advance_checkbox)

        # Stop on error option
        self.stop_on_error_checkbox = QCheckBox("Bei Fehler stoppen")
        self.stop_on_error_checkbox.setChecked(True)
        self.stop_on_error_checkbox.setToolTip("Pipeline stoppt bei ersten Fehler")
        global_layout.addWidget(self.stop_on_error_checkbox)

        layout.addWidget(global_group)

        # Buttons
        button_layout = QHBoxLayout()

        # Preset buttons
        preset_button = QPushButton("📋 Preset laden")
        preset_button.clicked.connect(self.load_preset)
        button_layout.addWidget(preset_button)

        save_preset_button = QPushButton("💾 Als Preset speichern")
        save_preset_button.clicked.connect(self.save_preset)
        button_layout.addWidget(save_preset_button)

        button_layout.addStretch()

        # Standard dialog buttons
        cancel_button = QPushButton("Abbrechen")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        save_button = QPushButton("Speichern")
        save_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        save_button.clicked.connect(self.save_config)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)

    def load_config(self, config: PipelineConfig):
        """Load existing configuration - Claude Generated"""
        try:
            # Load step configurations
            for step_id, step_widget in self.step_widgets.items():
                if step_id in config.step_configs:
                    step_widget.set_config(config.step_configs[step_id])
                elif step_id == "search":
                    # Load search suggesters from PipelineConfig
                    search_config = {"suggesters": config.search_suggesters}
                    step_widget.set_config(search_config)

            # Load global settings
            self.auto_advance_checkbox.setChecked(config.auto_advance)
            self.stop_on_error_checkbox.setChecked(config.stop_on_error)

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            QMessageBox.warning(
                self, "Fehler", f"Fehler beim Laden der Konfiguration: {e}"
            )

    def save_config(self):
        """Save current configuration - Claude Generated"""
        try:
            # Collect step configurations
            step_configs = {}
            search_suggesters = ["lobid", "swb"]  # Default

            for step_id, step_widget in self.step_widgets.items():
                config = step_widget.get_config()
                if step_id == "search" and "suggesters" in config:
                    # Extract suggesters for PipelineConfig
                    search_suggesters = config["suggesters"]
                step_configs[step_id] = config

                # Debug: Log what we're saving for each step
                task = config.get("task", "N/A")
                enabled = config.get("enabled", "N/A")
                self.logger.info(
                    f"Saving step '{step_id}': task='{task}', enabled={enabled}"
                )

            # Create PipelineConfig with search_suggesters
            config = PipelineConfig(
                auto_advance=self.auto_advance_checkbox.isChecked(),
                stop_on_error=self.stop_on_error_checkbox.isChecked(),
                step_configs=step_configs,
                search_suggesters=search_suggesters,
            )

            self.config_saved.emit(config)
            self.accept()

        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern: {e}")

    def load_preset(self):
        """Load a configuration preset - Claude Generated"""
        # TODO: Implement preset loading from file
        QMessageBox.information(
            self, "Preset laden", "Preset-Funktion wird implementiert..."
        )

    def save_preset(self):
        """Save current configuration as preset - Claude Generated"""
        # TODO: Implement preset saving to file
        QMessageBox.information(
            self, "Preset speichern", "Preset-Speichern wird implementiert..."
        )
