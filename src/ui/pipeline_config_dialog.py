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
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from typing import Dict, List, Any, Optional
import json
import logging

from ..core.pipeline_manager import PipelineConfig
from ..llm.llm_service import LlmService


class PipelineStepConfigWidget(QWidget):
    """Widget für die Konfiguration eines Pipeline-Schritts - Claude Generated"""

    def __init__(
        self, step_name: str, step_id: str, llm_service: LlmService, parent=None
    ):
        super().__init__(parent)
        self.step_name = step_name
        self.step_id = step_id
        self.llm_service = llm_service
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
        provider_layout.addWidget(self.model_combo, 1, 1)

        # Enable/Disable for this step
        self.enabled_checkbox = QCheckBox("Schritt aktivieren")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        provider_layout.addWidget(self.enabled_checkbox, 2, 0, 1, 2)

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

        # Custom Prompt (if applicable)
        if self.step_id in ["keywords", "verification", "classification"]:
            prompt_group = QGroupBox("Custom Prompt (optional)")
            prompt_layout = QVBoxLayout(prompt_group)

            self.custom_prompt = QTextEdit()
            self.custom_prompt.setMaximumHeight(100)
            self.custom_prompt.setPlaceholderText("Leer lassen für Standard-Prompt...")
            prompt_layout.addWidget(self.custom_prompt)

            layout.addWidget(prompt_group)

        # Initialize models for default provider
        self.on_provider_changed(self.provider_combo.currentText())

    def on_provider_changed(self, provider: str):
        """Handle provider change - Claude Generated"""
        self.model_combo.clear()
        models = self.llm_service.get_available_models(provider)
        if models:
            self.model_combo.addItems(models)

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

        # Add custom prompt if available
        if hasattr(self, "custom_prompt"):
            custom_text = self.custom_prompt.toPlainText().strip()
            if custom_text:
                config["custom_prompt"] = custom_text

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
            self.temp_spinbox.setValue(config["temperature"])

        if "top_p" in config:
            self.p_spinbox.setValue(config["top_p"])

        if "custom_prompt" in config and hasattr(self, "custom_prompt"):
            self.custom_prompt.setPlainText(config["custom_prompt"])


class PipelineConfigDialog(QDialog):
    """Dialog für Pipeline-Konfiguration - Claude Generated"""

    config_saved = pyqtSignal(object)  # PipelineConfig

    def __init__(
        self,
        llm_service: LlmService,
        current_config: Optional[PipelineConfig] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_service = llm_service
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

        # Define pipeline steps
        pipeline_steps = [
            ("keywords", "🔤 Keyword-Extraktion"),
            ("search", "🔍 GND-Suche"),
            ("verification", "✅ Verifikation"),
            ("classification", "📚 Klassifikation"),
        ]

        # Create tab for each step
        for step_id, step_name in pipeline_steps:
            if step_id == "search":
                # Search step doesn't need LLM config
                search_widget = QWidget()
                search_layout = QVBoxLayout(search_widget)
                search_layout.addWidget(
                    QLabel("GND-Suche verwendet keine LLM-Konfiguration.")
                )
                search_layout.addWidget(
                    QLabel(
                        "Die Suche wird automatisch über konfigurierte Suggester durchgeführt."
                    )
                )
                search_layout.addStretch()
                self.tab_widget.addTab(search_widget, step_name)
            else:
                step_widget = PipelineStepConfigWidget(
                    step_name, step_id, self.llm_service
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
            for step_id, step_widget in self.step_widgets.items():
                step_configs[step_id] = step_widget.get_config()

            # Create PipelineConfig
            config = PipelineConfig(
                auto_advance=self.auto_advance_checkbox.isChecked(),
                stop_on_error=self.stop_on_error_checkbox.isChecked(),
                step_configs=step_configs,
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
