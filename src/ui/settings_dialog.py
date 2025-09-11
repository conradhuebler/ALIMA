from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QScrollArea,
    QListWidget,
    QInputDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QSettings
from ..llm.prompt_service import PromptService
from ..utils.config import Config, ConfigSection, AIProvider
import logging
from typing import Dict, Any


class SettingsDialog(QDialog):
    def __init__(self, ui_settings: QSettings, parent=None):
        """
        Initialisiert den Einstellungsdialog.

        Args:
            ui_settings: QSettings für UI-bezogene Einstellungen
            app_config: Config-Instanz für Anwendungseinstellungen
            parent: Das übergeordnete Widget
        """
        super().__init__(parent)
        self.ui_settings = ui_settings  # Für UI-Einstellungen (Fenstergröße etc.)
        self.config = Config.get_instance()  # Für unsere eigenen Konfigurationen
        self.logger = logging.getLogger(__name__)
        self.prompt_manager = PromptService("/home/conrad/src/ALIMA/prompts.json")
        self.setWindowTitle("Einstellungen")
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        self.setWindowTitle("Einstellungen")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Tabs für verschiedene Einstellungsbereiche
        self.tabs = QTabWidget()

        # AI-Einstellungen
        self.ai_tab = self.create_ai_tab()
        self.tabs.addTab(self.ai_tab, "KI-Dienste")

        # Füge Prompts-Tab hinzu
        self.prompts_tab = self.create_prompts_tab()
        self.tabs.addTab(self.prompts_tab, "Prompts")

        # Such-Einstellungen
        self.search_tab = self.create_search_tab()
        self.tabs.addTab(self.search_tab, "Suche")

        # Cache-Einstellungen
        self.cache_tab = self.create_cache_tab()
        self.tabs.addTab(self.cache_tab, "Cache")

        # UI-Einstellungen
        self.ui_tab = self.create_ui_tab()
        self.tabs.addTab(self.ui_tab, "Oberfläche")
        
        # Provider-Preferences-Tab - Claude Generated
        self.provider_tab = self.create_provider_tab()
        self.tabs.addTab(self.provider_tab, "LLM-Provider")

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Speichern")
        self.save_button.clicked.connect(self.save_settings)

        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.clicked.connect(self.reject)

        self.reset_button = QPushButton("Zurücksetzen")
        self.reset_button.clicked.connect(self.reset_settings)

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

    def create_ai_tab(self) -> QWidget:
        """Erstellt den Tab für KI-Einstellungen"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Provider-Auswahl
        provider_group = QGroupBox("KI-Provider")
        provider_layout = QFormLayout()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems([p.value for p in AIProvider])
        provider_layout.addRow("Provider:", self.provider_combo)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        provider_layout.addRow("API-Key:", self.api_key_input)

        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)

        # Modell-Einstellungen
        model_group = QGroupBox("Modell-Parameter")
        model_layout = QFormLayout()

        self.model_combo = QComboBox()
        model_layout.addRow("Modell:", self.model_combo)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        model_layout.addRow("Temperature:", self.temperature_spin)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Verbinde Provider-Änderung mit Modell-Update
        self.provider_combo.currentTextChanged.connect(self.update_model_list)

        layout.addStretch()
        return tab

    def create_search_tab(self) -> QWidget:
        """Erstellt den Tab für Such-Einstellungen"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Allgemeine Sucheinstellungen
        search_group = QGroupBox("Sucheinstellungen")
        search_layout = QFormLayout()

        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(10, 10000)
        self.max_results_spin.setSingleStep(10)
        search_layout.addRow("Maximale Ergebnisse:", self.max_results_spin)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        search_layout.addRow("Timeout (Sekunden):", self.timeout_spin)

        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        layout.addStretch()
        return tab

    def create_cache_tab(self) -> QWidget:
        """Erstellt den Tab für Cache-Einstellungen"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Cache-Einstellungen
        cache_group = QGroupBox("Cache-Konfiguration")
        cache_layout = QFormLayout()

        self.cache_enabled_check = QCheckBox()
        cache_layout.addRow("Cache aktiviert:", self.cache_enabled_check)

    def create_ui_tab(self) -> QWidget:
        """Erstellt den Tab für UI-Einstellungen"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Darstellung
        display_group = QGroupBox("Darstellung")
        display_layout = QFormLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System", "Hell", "Dunkel"])
        display_layout.addRow("Theme:", self.theme_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        display_layout.addRow("Schriftgröße:", self.font_size_spin)

        self.tooltips_check = QCheckBox()
        display_layout.addRow("Tooltips anzeigen:", self.tooltips_check)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        layout.addStretch()
        return tab

    def create_prompts_tab(self) -> QWidget:
        """Erstellt den Tab für Prompt-Templates"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        prompt_editor_group = QGroupBox("Prompt-Konfiguration")
        prompt_editor_layout = QVBoxLayout(prompt_editor_group)

        open_editor_button = QPushButton("Prompt-Editor öffnen")
        open_editor_button.clicked.connect(self.open_prompt_editor)
        prompt_editor_layout.addWidget(open_editor_button)

        layout.addWidget(prompt_editor_group)
        layout.addStretch()
        return tab

    def create_provider_tab(self) -> QWidget:
        """Erstellt den Tab für Provider-Preferences - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        provider_group = QGroupBox("LLM-Provider-Einstellungen")
        provider_layout = QVBoxLayout(provider_group)

        info_label = QLabel(
            "Hier können Sie universelle LLM-Provider-Präferenzen für alle Tasks konfigurieren.\n"
            "Diese Einstellungen gelten für Textanalyse, Bilderkennung, Klassifikation und alle anderen KI-Aufgaben."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; margin: 5px; }")
        provider_layout.addWidget(info_label)

        open_preferences_button = QPushButton("Provider-Einstellungen öffnen")
        open_preferences_button.clicked.connect(self.open_provider_preferences)
        open_preferences_button.setStyleSheet(
            "QPushButton { "
            "background-color: #4CAF50; "
            "color: white; "
            "border: none; "
            "padding: 10px 20px; "
            "font-weight: bold; "
            "border-radius: 5px; "
            "} "
            "QPushButton:hover { "
            "background-color: #45a049; "
            "}"
        )
        provider_layout.addWidget(open_preferences_button)

        layout.addWidget(provider_group)
        layout.addStretch()
        return tab

    def open_prompt_editor(self):
        """Öffnet den Prompt-Editor-Dialog"""
        from .prompt_editor_dialog import PromptEditorDialog

        editor = PromptEditorDialog(self)
        editor.exec()
        # After closing the editor, reload settings if necessary
        self.load_settings()

    def open_provider_preferences(self):
        """Öffnet den Provider-Preferences-Dialog - Claude Generated"""
        from .provider_preferences_dialog import ProviderPreferencesDialog

        dialog = ProviderPreferencesDialog(self)
        dialog.preferences_changed.connect(self.on_provider_preferences_changed)
        dialog.exec()

    def on_provider_preferences_changed(self):
        """Handler für Provider-Preferences-Änderungen - Claude Generated"""
        # Reload settings if necessary
        self.load_settings()
        # Could emit signal to parent to refresh components if needed

    def update_model_list(self):
        """Aktualisiert die Liste der verfügbaren Modelle basierend auf dem ausgewählten Provider"""
        current_provider = self.provider_combo.currentText().lower()
        provider_config = self.config.get_ai_provider_config(current_provider)

        self.model_combo.clear()
        if provider_config and "models" in provider_config:
            self.model_combo.addItems(provider_config["models"])

            # Setze das Standard-Modell
            default_model = provider_config.get("default_model")
            if default_model:
                index = self.model_combo.findText(default_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)

    def load_settings(self):
        """Lädt die aktuellen Einstellungen in den Dialog"""
        try:
            # AI-Einstellungen
            ai_config = self.config.get_section(ConfigSection.AI)

            if ai_config:
                # Konvertiere AIProvider Enum zu String
                self.provider_combo.setCurrentText(ai_config.provider)
                self.api_key_input.setText(ai_config.api_key)  # Korrigierter Name
                self.temperature_spin.setValue(ai_config.temperature)
                self.update_model_list()

            # Cache-Einstellungen
            cache_config = self.config.get_section(ConfigSection.CACHE)
            if cache_config:
                self.cache_enabled_check.setChecked(cache_config.enabled)

            # Such-Einstellungen
            search_config = self.config.get_section(ConfigSection.SEARCH)
            if search_config:
                self.max_results_spin.setValue(search_config.max_results)
                self.timeout_spin.setValue(search_config.timeout)

            # UI-Einstellungen
            ui_config = self.config.get_section(ConfigSection.UI)
            if ui_config:
                self.theme_combo.setCurrentText(ui_config.theme)
                self.font_size_spin.setValue(ui_config.font_size)

            # Debug-Einstellungen
            general_config = self.config.get_section(ConfigSection.GENERAL)

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Einstellungen: {e}")
            QMessageBox.critical(
                self,
                "Fehler",
                f"Die Einstellungen konnten nicht geladen werden: {str(e)}",
            )

    def save_settings(self):
        """Speichert die aktuellen Einstellungen"""
        try:
            # AI-Einstellungen
            ai_config = self.config.get_section(ConfigSection.AI)
            if ai_config:
                ai_config.provider = AIProvider(self.provider_combo.currentText())
                ai_config.api_key = self.api_key_input.text()  # Korrigierter Name
                ai_config.temperature = self.temperature_spin.value()

            # Cache-Einstellungen
            cache_config = self.config.get_section(ConfigSection.CACHE)
            if cache_config:
                cache_config.enabled = self.cache_enabled_check.isChecked()

            # Such-Einstellungen
            search_config = self.config.get_section(ConfigSection.SEARCH)
            if search_config:
                search_config.max_results = self.max_results_spin.value()
                search_config.timeout = self.timeout_spin.value()

            # UI-Einstellungen
            ui_config = self.config.get_section(ConfigSection.UI)
            if ui_config:
                ui_config.theme = self.theme_combo.currentText()
                ui_config.font_size = self.font_size_spin.value()

            # Debug-Einstellungen
            general_config = self.config.get_section(ConfigSection.GENERAL)

            # Speichere die Konfiguration
            self.logger.debug("Versuche Konfiguration zu speichern...")
            self.config.save_config()
            self.logger.debug("save_config() wurde aufgerufen")
            self.logger.info("Einstellungen erfolgreich gespeichert")
            QMessageBox.information(
                self, "Erfolg", "Die Einstellungen wurden gespeichert."
            )

        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Einstellungen: {e}")
            QMessageBox.critical(
                self,
                "Fehler",
                f"Die Einstellungen konnten nicht gespeichert werden: {str(e)}",
            )

    def reset_settings(self):
        """Setzt alle Einstellungen auf Standardwerte zurück"""
        if (
            QMessageBox.question(
                self,
                "Einstellungen zurücksetzen",
                "Möchten Sie wirklich alle Einstellungen auf die Standardwerte zurücksetzen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        ):
            for section in ConfigSection:
                self.config.reset_section(section)
            self.load_settings()

    def clear_cache(self):
        """Leert den Cache"""
        if (
            QMessageBox.question(
                self,
                "Cache leeren",
                "Möchten Sie wirklich den gesamten Cache leeren?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        ):
            try:
                cache_config = self.config.get_section(ConfigSection.CACHE)
                if cache_config and os.path.exists(cache_config.db_path):
                    os.remove(cache_config.db_path)
                QMessageBox.information(self, "Erfolg", "Cache wurde geleert.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Fehler", f"Fehler beim Leeren des Cache:\n{str(e)}"
                )
