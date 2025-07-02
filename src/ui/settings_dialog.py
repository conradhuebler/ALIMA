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
)
from PyQt6.QtCore import Qt, QSettings
from ..llm.prompt_manager import PromptManager
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
        self.prompt_manager = PromptManager("/home/conrad/src/ALIMA/prompts.json")
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

        # Template-Liste
        templates_group = QGroupBox("Verfügbare Templates")
        templates_layout = QVBoxLayout()

        self.templates_table = QTableWidget()
        self.templates_table.setColumnCount(3)
        self.templates_table.setHorizontalHeaderLabels(
            ["Task", "Modell", "Beschreibung"]
        )
        self.templates_table.horizontalHeader().setStretchLastSection(True)
        templates_layout.addWidget(self.templates_table)

        # Buttons
        button_layout = QHBoxLayout()
        self.edit_template_button = QPushButton("Bearbeiten")
        self.edit_template_button.clicked.connect(self.edit_template)
        self.add_template_button = QPushButton("Hinzufügen")
        self.add_template_button.clicked.connect(self.add_template)
        self.delete_template_button = QPushButton("Löschen")
        self.delete_template_button.clicked.connect(self.delete_template)

        button_layout.addWidget(self.edit_template_button)
        button_layout.addWidget(self.add_template_button)
        button_layout.addWidget(self.delete_template_button)
        templates_layout.addLayout(button_layout)

        templates_group.setLayout(templates_layout)
        layout.addWidget(templates_group)

        # Prompt-Bearbeitungsbereich
        self.prompt_edit_group = QGroupBox("Prompt bearbeiten")
        self.prompt_edit_group.setVisible(False)  # Zunächst versteckt
        prompt_edit_layout = QVBoxLayout(self.prompt_edit_group)

        form_layout = QFormLayout()
        self.edit_task_combo = QComboBox()
        form_layout.addRow("Task:", self.edit_task_combo)
        self.edit_model_combo = QComboBox()
        form_layout.addRow("Modell:", self.edit_model_combo)
        self.edit_description_input = QLineEdit()
        form_layout.addRow("Beschreibung:", self.edit_description_input)
        self.edit_temperature_spin = QDoubleSpinBox()
        self.edit_temperature_spin.setRange(0.0, 2.0)
        self.edit_temperature_spin.setSingleStep(0.1)
        form_layout.addRow("Temperatur:", self.edit_temperature_spin)
        self.edit_p_value_spin = QDoubleSpinBox()
        self.edit_p_value_spin.setRange(0.0, 1.0)
        self.edit_p_value_spin.setSingleStep(0.1)
        form_layout.addRow("P-Wert:", self.edit_p_value_spin)
        self.edit_seed_spin = QSpinBox()
        self.edit_seed_spin.setRange(0, 1000000000)
        form_layout.addRow("Seed:", self.edit_seed_spin)

        prompt_edit_layout.addLayout(form_layout)

        self.edit_prompt_text = QTextEdit()
        self.edit_prompt_text.setPlaceholderText("Prompt-Text hier eingeben...")
        prompt_edit_layout.addWidget(self.edit_prompt_text)

        self.edit_system_text = QTextEdit()
        self.edit_system_text.setPlaceholderText("System-Prompt hier eingeben...")
        prompt_edit_layout.addWidget(self.edit_system_text)

        save_cancel_layout = QHBoxLayout()
        self.save_prompt_button = QPushButton("Speichern")
        self.save_prompt_button.clicked.connect(self.save_template)
        self.cancel_prompt_button = QPushButton("Abbrechen")
        self.cancel_prompt_button.clicked.connect(self.cancel_edit)
        save_cancel_layout.addWidget(self.save_prompt_button)
        save_cancel_layout.addWidget(self.cancel_prompt_button)
        prompt_edit_layout.addLayout(save_cancel_layout)

        layout.addWidget(self.prompt_edit_group)

        self.update_templates_table()
        return tab

    def update_templates_table(self):
        """Aktualisiert die Template-Tabelle"""
        self.templates_table.setRowCount(0)
        for task in self.prompt_manager.get_available_tasks():
            for model in self.prompt_manager.get_available_models(task):
                row = self.templates_table.rowCount()
                self.templates_table.insertRow(row)
                self.templates_table.setItem(row, 0, QTableWidgetItem(task))
                self.templates_table.setItem(row, 1, QTableWidgetItem(model))
                # Description is not directly available in prompt_manager, so leave empty or derive if possible
                self.templates_table.setItem(row, 2, QTableWidgetItem(""))

    def edit_template(self):
        """Öffnet den Bereich zum Bearbeiten eines Templates"""
        current_row = self.templates_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie ein Template aus.")
            return

        self.current_edit_mode = "edit"
        self.current_edit_task = self.templates_table.item(current_row, 0).text()
        self.current_edit_model = self.templates_table.item(current_row, 1).text()

        config = self.prompt_manager.get_prompt_config(self.current_edit_task, self.current_edit_model)

        self.edit_task_combo.clear()
        self.edit_task_combo.addItem(self.current_edit_task)
        self.edit_task_combo.setEnabled(False)

        self.edit_model_combo.clear()
        self.edit_model_combo.addItem(self.current_edit_model)
        self.edit_model_combo.setEnabled(False)

        self.edit_prompt_text.setPlainText(config["prompt"])
        self.edit_system_text.setPlainText(config["system"])
        self.edit_temperature_spin.setValue(config["temp"])
        self.edit_p_value_spin.setValue(config["p-value"])
        # Seed is not directly in get_prompt_config, need to adjust PromptManager or add it
        # For now, set a default or retrieve if possible
        self.edit_seed_spin.setValue(0) # Placeholder

        self.prompt_edit_group.setVisible(True)

    def add_template(self):
        """Öffnet den Bereich zum Hinzufügen eines neuen Templates"""
        self.current_edit_mode = "add"
        self.current_edit_task = ""
        self.current_edit_model = ""

        self.edit_task_combo.clear()
        self.edit_task_combo.addItems(self.prompt_manager.get_available_tasks())
        self.edit_task_combo.setEnabled(True)

        self.edit_model_combo.clear()
        self.edit_model_combo.setEditable(True)
        self.edit_model_combo.setEnabled(True)

        self.edit_prompt_text.clear()
        self.edit_system_text.clear()
        self.edit_temperature_spin.setValue(0.7)
        self.edit_p_value_spin.setValue(0.9)
        self.edit_seed_spin.setValue(0)

        self.prompt_edit_group.setVisible(True)

    def delete_template(self):
        """Löscht das ausgewählte Template"""
        current_row = self.templates_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie ein Template aus.")
            return

        task = self.templates_table.item(current_row, 0).text()
        model = self.templates_table.item(current_row, 1).text()

        reply = QMessageBox.question(
            self,
            "Bestätigung",
            f"Möchten Sie das Template für Task '{task}' und Modell '{model}' wirklich löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.prompt_manager.delete_prompt_config(task, model)
                self.prompt_manager.save_config()
                self.update_templates_table()
                QMessageBox.information(self, "Erfolg", "Template erfolgreich gelöscht.")
            except Exception as e:
                QMessageBox.critical(self, "Fehler", f"Fehler beim Löschen des Templates: {e}")

    def save_template(self):
        """Speichert das bearbeitete/neue Template"""
        task = self.edit_task_combo.currentText()
        model = self.edit_model_combo.currentText()
        prompt_text = self.edit_prompt_text.toPlainText()
        system_text = self.edit_system_text.toPlainText()
        temperature = self.edit_temperature_spin.value()
        p_value = self.edit_p_value_spin.value()
        seed = self.edit_seed_spin.value()

        if not task or not model or not prompt_text:
            QMessageBox.warning(self, "Warnung", "Bitte füllen Sie alle erforderlichen Felder aus (Task, Modell, Prompt-Text).")
            return

        new_prompt_data = {
            "prompt": prompt_text,
            "system": system_text,
            "temp": temperature,
            "p-value": p_value,
            "models": [model], # Always save as a list
            "seed": seed
        }

        try:
            if self.current_edit_mode == "edit":
                self.prompt_manager.update_prompt_config(task, self.current_edit_model, new_prompt_data)
            elif self.current_edit_mode == "add":
                self.prompt_manager.add_prompt_config(task, new_prompt_data)
            
            self.prompt_manager.save_config()
            self.update_templates_table()
            self.prompt_edit_group.setVisible(False)
            QMessageBox.information(self, "Erfolg", "Template erfolgreich gespeichert.")
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern des Templates: {e}")

    def cancel_edit(self):
        """Bricht die Bearbeitung ab und versteckt den Bearbeitungsbereich"""
        self.prompt_edit_group.setVisible(False)

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
