from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QGroupBox, QFormLayout, QComboBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QTextEdit
)
from PyQt6.QtCore import Qt, QSettings
from ..utils.config import Config, ConfigSection, AIProvider
from ..utils.prompt_templates import PromptTemplate, PromptConfig
from typing import Dict, Any
import os
import logging


class PromptEditDialog(QDialog):
    """Dialog zum Bearbeiten eines einzelnen Prompts"""
    def __init__(self, template: PromptTemplate, parent=None):
        super().__init__(parent)
        self.template = template
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"Prompt bearbeiten: {self.template.name}")
        self.setModal(True)
        self.resize(800, 600)

        layout = QVBoxLayout(self)

        # Beschreibung
        desc_layout = QFormLayout()
        self.desc_input = QLineEdit(self.template.description)
        desc_layout.addRow("Beschreibung:", self.desc_input)
        layout.addLayout(desc_layout)

        # Modell
        model_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gemini-1.5-flash", "gemini-pro", "gemini-ultra"
        ])  # Hier könnten wir die verfügbaren Modelle dynamisch laden
        self.model_combo.setCurrentText(self.template.model)
        model_layout.addRow("Modell:", self.model_combo)
        layout.addLayout(model_layout)

        # Template
        template_group = QGroupBox("Template")
        template_layout = QVBoxLayout()
        self.template_edit = QTextEdit(self.template.template)
        template_layout.addWidget(self.template_edit)
        
        # Variablen-Info
        variables_label = QLabel(
            "Verfügbare Variablen: " + 
            ", ".join([f"{{{var}}}" for var in self.template.required_variables])
        )
        template_layout.addWidget(variables_label)
        
        template_group.setLayout(template_layout)
        layout.addWidget(template_group)

        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Speichern")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Abbrechen")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

    def get_updated_template(self) -> PromptTemplate:
        """Gibt das aktualisierte Template zurück"""
        return PromptTemplate(
            name=self.template.name,
            description=self.desc_input.text(),
            template=self.template_edit.toPlainText(),
            required_variables=self.template.required_variables,
            model=self.model_combo.currentText()
        )


class SettingsDialog(QDialog):
    def __init__(self, ui_settings: QSettings,  parent=None):
        """
        Initialisiert den Einstellungsdialog.
        
        Args:
            ui_settings: QSettings für UI-bezogene Einstellungen
            app_config: Config-Instanz für Anwendungseinstellungen
            parent: Das übergeordnete Widget
        """
        super().__init__(parent)
        self.ui_settings = ui_settings  # Für UI-Einstellungen (Fenstergröße etc.)
        self.config = Config.get_instance()        # Für unsere eigenen Konfigurationen
        self.logger = logging.getLogger(__name__)
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
        
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.1)
        model_layout.addRow("Top-P:", self.top_p_spin)
        
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
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 100.0)
        self.threshold_spin.setSingleStep(0.1)
        search_layout.addRow("Standard-Schwellenwert (%):", self.threshold_spin)
        
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(10, 10000)
        self.max_results_spin.setSingleStep(10)
        search_layout.addRow("Maximale Ergebnisse:", self.max_results_spin)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        search_layout.addRow("Timeout (Sekunden):", self.timeout_spin)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # API-Einstellungen
        api_group = QGroupBox("API-Einstellungen")
        api_layout = QFormLayout()
        
        self.api_url_input = QLineEdit()
        api_layout.addRow("API-URL:", self.api_url_input)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(10, 1000)
        api_layout.addRow("Batch-Größe:", self.batch_size_spin)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

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
        
        self.cache_age_spin = QSpinBox()
        self.cache_age_spin.setRange(1, 168)  # 1 Stunde bis 1 Woche
        cache_layout.addRow("Maximales Alter (Stunden):", self.cache_age_spin)
        
        self.cache_path_input = QLineEdit()
        cache_layout.addRow("Cache-Pfad:", self.cache_path_input)
        
        self.max_entries_spin = QSpinBox()
        self.max_entries_spin.setRange(100, 100000)
        cache_layout.addRow("Maximale Einträge:", self.max_entries_spin)
        
        self.compression_check = QCheckBox()
        cache_layout.addRow("Komprimierung:", self.compression_check)
        
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        # Cache-Statistiken
        stats_group = QGroupBox("Cache-Statistiken")
        stats_layout = QFormLayout()
        
        self.stats_label = QLabel()
        stats_layout.addRow("Aktuelle Nutzung:", self.stats_label)
        
        self.clear_cache_button = QPushButton("Cache leeren")
        self.clear_cache_button.clicked.connect(self.clear_cache)
        stats_layout.addRow("", self.clear_cache_button)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        layout.addStretch()
        return tab

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

        # Verhalten
        behavior_group = QGroupBox("Verhalten")
        behavior_layout = QFormLayout()
        
        self.save_position_check = QCheckBox()
        behavior_layout.addRow("Fensterposition speichern:", self.save_position_check)
        
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(1, 60)
        behavior_layout.addRow("Automatisch speichern (Min.):", self.autosave_spin)
        
        self.recent_searches_spin = QSpinBox()
        self.recent_searches_spin.setRange(5, 50)
        behavior_layout.addRow("Anzahl letzter Suchen:", self.recent_searches_spin)
        
        behavior_group.setLayout(behavior_layout)
        layout.addWidget(behavior_group)

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
        self.templates_table.setHorizontalHeaderLabels([
            "Name", "Beschreibung", "Modell"
        ])
        self.templates_table.horizontalHeader().setStretchLastSection(True)
        templates_layout.addWidget(self.templates_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.edit_template_button = QPushButton("Bearbeiten")
        self.edit_template_button.clicked.connect(self.edit_template)
        #self.reset_template_button = QPushButton("Zurücksetzen")
        #self.reset_template_button.clicked.connect(self.reset_template)
        
        button_layout.addWidget(self.edit_template_button)
        #button_layout.addWidget(self.reset_template_button)
        templates_layout.addLayout(button_layout)
        
        templates_group.setLayout(templates_layout)
        layout.addWidget(templates_group)

        self.update_templates_table()
        return tab

    def update_templates_table(self):
        """Aktualisiert die Template-Tabelle"""
        self.templates_table.setRowCount(0)
        prompt_config = self.config.get_section(ConfigSection.PROMPTS)
        
        if not prompt_config:
            return

        for template in prompt_config.templates.values():
            row = self.templates_table.rowCount()
            self.templates_table.insertRow(row)
            self.templates_table.setItem(row, 0, QTableWidgetItem(template.name))
            self.templates_table.setItem(row, 1, QTableWidgetItem(template.description))
            self.templates_table.setItem(row, 2, QTableWidgetItem(template.model))

    def edit_template(self):
        """Öffnet den Dialog zum Bearbeiten eines Templates"""
        current_row = self.templates_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie ein Template aus.")
            return

        template_name = self.templates_table.item(current_row, 0).text()
        prompt_config = self.config.get_section(ConfigSection.PROMPTS)
        template = prompt_config.templates.get(template_name)

        if template:
            dialog = PromptEditDialog(template, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                updated_template = dialog.get_updated_template()
                prompt_config.templates[template_name] = updated_template
                self.update_templates_table()

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
            #    self.cache_max_age_spin.setValue(cache_config.max_age)
            #    self.cache_max_size_spin.setValue(cache_config.max_size)

            # Such-Einstellungen
            search_config = self.config.get_section(ConfigSection.SEARCH)
            if search_config:
                self.max_results_spin.setValue(search_config.max_results)
            #    self.min_score_spin.setValue(search_config.min_score)
                self.timeout_spin.setValue(search_config.timeout)

            # UI-Einstellungen
            ui_config = self.config.get_section(ConfigSection.UI)
            if ui_config:
                self.theme_combo.setCurrentText(ui_config.theme)
                self.font_size_spin.setValue(ui_config.font_size)

            # Debug-Einstellungen
            general_config = self.config.get_section(ConfigSection.GENERAL)
            #if general_config:
            #    self.debug_mode_check.setChecked(general_config.debug)

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Einstellungen: {e}")
            QMessageBox.critical(
                self,
                "Fehler",
                f"Die Einstellungen konnten nicht geladen werden: {str(e)}"
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
            #    cache_config.max_age = self.cache_max_age_spin.value()
            #    cache_config.max_size = self.cache_max_size_spin.value()

            # Such-Einstellungen
            search_config = self.config.get_section(ConfigSection.SEARCH)
            if search_config:
                search_config.max_results = self.max_results_spin.value()
            #    search_config.min_score = self.min_score_spin.value()
                search_config.timeout = self.timeout_spin.value()

            # UI-Einstellungen
            ui_config = self.config.get_section(ConfigSection.UI)
            if ui_config:
                ui_config.theme = self.theme_combo.currentText()
                ui_config.font_size = self.font_size_spin.value()

            # Debug-Einstellungen
            general_config = self.config.get_section(ConfigSection.GENERAL)
            #if general_config:
            #    general_config.debug = self.debug_mode_check.isChecked()

            # Speichere die Konfiguration
            self.logger.debug("Versuche Konfiguration zu speichern...")
            self.config.save_config()
            self.logger.debug("save_config() wurde aufgerufen")
            self.logger.info("Einstellungen erfolgreich gespeichert")
            QMessageBox.information(self, "Erfolg", "Die Einstellungen wurden gespeichert.")

        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Einstellungen: {e}")
            QMessageBox.critical(
                self,
                "Fehler",
                f"Die Einstellungen konnten nicht gespeichert werden: {str(e)}"
            )

    def reset_settings(self):
        """Setzt alle Einstellungen auf Standardwerte zurück"""
        if QMessageBox.question(
            self,
            "Einstellungen zurücksetzen",
            "Möchten Sie wirklich alle Einstellungen auf die Standardwerte zurücksetzen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            for section in ConfigSection:
                self.config.reset_section(section)
            self.load_settings()

    def clear_cache(self):
        """Leert den Cache"""
        if QMessageBox.question(
            self,
            "Cache leeren",
            "Möchten Sie wirklich den gesamten Cache leeren?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            try:
                cache_config = self.config.get_section(ConfigSection.CACHE)
                if cache_config and os.path.exists(cache_config.db_path):
                    os.remove(cache_config.db_path)
                QMessageBox.information(self, "Erfolg", "Cache wurde geleert.")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Fehler",
                    f"Fehler beim Leeren des Cache:\n{str(e)}"
                )
