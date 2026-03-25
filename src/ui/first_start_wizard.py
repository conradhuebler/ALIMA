#!/usr/bin/env python3
"""
ALIMA First-Start Setup Wizard
Guide new users through initial configuration (LLM provider and GND database)
Claude Generated
"""

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QRadioButton, QButtonGroup, QComboBox, QPushButton, QProgressDialog,
    QMessageBox, QGroupBox, QSpinBox, QCheckBox, QFileDialog, QTextEdit, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap
from pathlib import Path
import logging
from typing import Optional, List

from ..utils.config_manager import ConfigManager
from ..utils.setup_utils import (
    OllamaConnectionValidator, APIKeyValidator, GNDDatabaseDownloader,
    ConfigurationBuilder, PromptValidator, SetupResult
)
from ..utils.preset_loader import PresetLoader, InstitutionPresets


logger = logging.getLogger(__name__)


class FirstStartWizard(QWizard):
    """Main wizard for first-start setup - Claude Generated"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ALIMA Setup-Assistent")
        self.setWindowIcon(QIcon("assets/alima.png"))
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # Load institution presets (empty if no alima_presets.json found) - Claude Generated
        self._presets = PresetLoader.load()

        # Initialize pages
        self.welcome_page = WelcomePage(presets=self._presets)
        self.addPage(self.welcome_page)

        self.directory_page = DirectorySetupPage()
        self.addPage(self.directory_page)

        self.llm_page = LLMSetupPage(presets=self._presets)
        self.addPage(self.llm_page)

        self.model_page = ModelSelectionPage(presets=self._presets)
        self.addPage(self.model_page)

        self.database_page = DatabaseSetupPage(presets=self._presets)
        self.addPage(self.database_page)

        self.catalog_page = CatalogSetupPage(presets=self._presets)
        self.addPage(self.catalog_page)

        self.summary_page = SummaryPage()
        self.addPage(self.summary_page)

        # Store configuration results
        self.config = None

    def accept(self):
        """Handle wizard completion - Claude Generated"""
        try:
            # Get LLM provider configuration from the LLM page
            provider_type = getattr(self.llm_page, 'provider_type', 'ollama')
            provider_name = getattr(self.llm_page, 'provider_name', 'ALIMA Provider')
            base_url = getattr(self.llm_page, 'base_url', 'http://localhost:11434')
            api_key = getattr(self.llm_page, 'api_key', '')
            models = getattr(self.llm_page, 'available_models', [])

            # Get task model selections from model page - Claude Generated
            task_model_selections = getattr(
                self.model_page, 'task_model_selections', {}
            ) if hasattr(self, 'model_page') else {}

            # If no models were tested, use default model list
            if not models:
                if provider_type == 'ollama':
                    models = ['mistral']  # Default model
                elif provider_type == 'openai_compatible':
                    models = ['unknown']
                elif provider_type == 'gemini':
                    models = ['gemini-1.5-flash']
                elif provider_type == 'anthropic':
                    models = ['claude-3-5-haiku-20241022']

            # Build configuration with task selections - Claude Generated
            from ..utils.setup_utils import ConfigurationBuilder
            self.config = ConfigurationBuilder.create_initial_config(
                provider_type=provider_type,
                provider_name=provider_name,
                base_url=base_url,
                api_key=api_key,
                models=models,
                task_model_selections=task_model_selections
            )

            # Apply directory settings from wizard - Claude Generated
            root_dir = self.directory_page.root_dir_input.text().strip()
            if root_dir:
                root = Path(root_dir)
                self.config.system_config.cache_dir = str(root / "cache")
                self.config.system_config.data_dir = str(root / "data")
                self.config.system_config.temp_dir = str(root / "tmp")
                self.config.system_config.autosave_dir = str(root / "results")

            # Mark first run as completed
            self.config.system_config.first_run_completed = True

            # Save catalog SOAP configuration - Claude Generated
            from ..utils.config_models import CatalogConfig
            self.config.catalog_config = CatalogConfig(
                catalog_search_url=self.catalog_page.soap_search_url.text().strip(),
                catalog_details_url=self.catalog_page.soap_details_url.text().strip(),
                catalog_token=self.catalog_page.catalog_token.text().strip(),
                catalog_web_search_url=self.catalog_page.web_search_url.text().strip(),
                catalog_web_record_url=self.catalog_page.web_record_url.text().strip(),
            )

            # Save configuration
            config_manager = ConfigManager()
            config_manager.save_config(self.config)

            # Apply database configuration from database_page - Claude Generated
            if hasattr(self, 'database_page'):
                db_config = self.database_page.get_database_config()
                if db_config.get('db_type') == 'sqlite':
                    if db_config.get('sqlite_path'):
                        self.config.database_config.sqlite_path = db_config['sqlite_path']
                elif db_config.get('db_type') in ('mariadb', 'mysql'):
                    self.config.database_config.db_type = 'mariadb'
                    self.config.database_config.host = db_config.get('host', 'localhost')
                    self.config.database_config.port = db_config.get('port', 3306)
                    self.config.database_config.database = db_config.get('database', 'alima_knowledge')
                    self.config.database_config.username = db_config.get('username', 'alima')
                    self.config.database_config.password = db_config.get('password', '')

            # Handle GND actions - Claude Generated
            gnd_action = self.database_page.get_gnd_action() if hasattr(self, 'database_page') else 'skip'

            # Handle GND download for post-wizard background import
            if gnd_action == 'download' and hasattr(self.database_page, 'downloaded_xml_path') and self.database_page.downloaded_xml_path:
                import json
                import tempfile

                marker_file = Path(tempfile.gettempdir()) / "alima_gnd_pending.json"
                marker_data = {
                    'xml_path': self.database_page.downloaded_xml_path,
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                }
                marker_file.write_text(json.dumps(marker_data))
                logger.info(f"GND import marker saved: {marker_file}")

                QMessageBox.information(
                    self,
                    "GND-Import läuft im Hintergrund",
                    "✅ Setup abgeschlossen!\n\n"
                    "Die GND-Datenbank wird im Hintergrund importiert (~5-10 Min).\n\n"
                    "Sie können ALIMA sofort nutzen. Lobid-API wird als Fallback "
                    "verwendet, bis der Import fertig ist."
                )

            # Handle direct DB file import
            elif gnd_action == 'import' and self.database_page.gnd_file_input.text():
                self._import_database_file(self.database_page.gnd_file_input.text(), self.config)

            logger.info(f"First-start wizard completed: provider={provider_type}, models={len(models)}")
            super().accept()

        except Exception as e:
            logger.error(f"Error completing wizard: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Setup Error", f"Error saving configuration:\n{str(e)}")

    def _import_database_file(self, source_path: str, config) -> None:
        """Copy an existing SQLite database file to the ALIMA database location - Claude Generated"""
        import shutil
        try:
            source = Path(source_path)
            target = Path(config.database_config.sqlite_path)
            target.parent.mkdir(parents=True, exist_ok=True)

            # Basic SQLite validity check
            import sqlite3
            try:
                conn = sqlite3.connect(str(source))
                conn.execute("SELECT 1")
                conn.close()
            except sqlite3.DatabaseError:
                QMessageBox.warning(
                    self,
                    "Ungültige Datenbankdatei",
                    f"Die Datei ist keine gültige SQLite-Datenbank:\n{source_path}"
                )
                return

            shutil.copy2(str(source), str(target))
            logger.info(f"Database imported: {source} -> {target}")
            QMessageBox.information(
                self,
                "Datenbank importiert",
                f"✅ Datenbank erfolgreich importiert!\n\n"
                f"Quelle: {source_path}\n"
                f"Ziel: {target}"
            )
        except Exception as e:
            logger.error(f"Database import failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Import fehlgeschlagen",
                f"Die Datenbankdatei konnte nicht importiert werden:\n{str(e)}"
            )


class DirectorySetupPage(QWizardPage):
    """Directory setup page for choosing root data directory - Claude Generated"""

    def __init__(self):
        super().__init__()
        self.setTitle("Datenverzeichnisse")
        self.setSubTitle("Wählen Sie ein Wurzelverzeichnis für ALIMA-Daten")

        layout = QVBoxLayout()

        # Root directory selection
        root_group = QGroupBox("Wurzelverzeichnis")
        root_layout = QFormLayout()

        root_dir_layout = QHBoxLayout()
        self.root_dir_input = QLineEdit()
        self.root_dir_input.setText(str(Path.home() / "Documents" / "Alima"))
        root_dir_layout.addWidget(self.root_dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_root_dir)
        root_dir_layout.addWidget(browse_btn)
        root_layout.addRow("Wurzelverzeichnis:", root_dir_layout)

        root_group.setLayout(root_layout)
        layout.addWidget(root_group)

        # Subdirectory preview
        preview_group = QGroupBox("Unterverzeichnisse (werden automatisch angelegt)")
        preview_layout = QFormLayout()

        self.cache_label = QLabel()
        self.data_label = QLabel()
        self.temp_label = QLabel()
        self.results_label = QLabel()

        preview_layout.addRow("Cache:", self.cache_label)
        preview_layout.addRow("Daten:", self.data_label)
        preview_layout.addRow("Temp:", self.temp_label)
        preview_layout.addRow("Ergebnisse:", self.results_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        hint = QLabel("ℹ️ Verzeichnisse werden beim ersten Bedarf automatisch angelegt.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch()
        self.setLayout(layout)

        # Connect update
        self.root_dir_input.textChanged.connect(self._update_preview)
        self._update_preview(self.root_dir_input.text())

    def _browse_root_dir(self):
        """Browse for root directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Wurzelverzeichnis wählen", self.root_dir_input.text()
        )
        if directory:
            self.root_dir_input.setText(directory)

    def _update_preview(self, root_text: str):
        """Update subdirectory preview labels - Claude Generated"""
        root = Path(root_text) if root_text.strip() else Path.home() / "Documents" / "Alima"
        self.cache_label.setText(str(root / "cache"))
        self.data_label.setText(str(root / "data"))
        self.temp_label.setText(str(root / "tmp"))
        self.results_label.setText(str(root / "results"))


class WelcomePage(QWizardPage):
    """Welcome page introducing ALIMA - Claude Generated"""

    def __init__(self, presets: 'InstitutionPresets | None' = None):
        super().__init__()
        self.setTitle("Willkommen bei ALIMA")
        self.setSubTitle("Automatische Bibliotheksindexierung und Metadatenanalyse")

        layout = QVBoxLayout()

        # Title
        title = QLabel("Willkommen bei ALIMA!")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        # Description
        description = QLabel(
            "ALIMA hilft Ihnen bei der Analyse von Bibliotheksmaterialien und der Extraktion von Metadaten mit KI.\n\n"
            "Dieser Assistent führt Sie durch:\n"
            "1. Einrichtung eines LLM-Anbieters (lokal oder Cloud)\n"
            "2. Optionaler Download der GND-Normdaten\n"
            "3. Überprüfung Ihrer Konfiguration\n\n"
            "Los geht's!"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Institution preset banner - Claude Generated
        if presets and presets.institution_name:
            preset_banner = QLabel(
                f"🏛️ Konfiguriert für: {presets.institution_name}\n"
                "Vorausgefüllte Einstellungen können überschrieben werden."
            )
            preset_banner.setStyleSheet(
                "background-color: #dbeafe; color: #1e3a5f; "
                "padding: 8px; border-radius: 5px; border: 1px solid #93c5fd;"
            )
            preset_banner.setWordWrap(True)
            layout.addWidget(preset_banner)

        # Info box
        info = QLabel(
            "💡 Tipp: Sie können diese Einstellungen jederzeit im Anwendungsmenü ändern."
        )
        info.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        self.setLayout(layout)

    def validatePage(self) -> bool:
        """Page validation - Claude Generated"""
        return True


class LLMSetupPage(QWizardPage):
    """LLM Provider configuration page - Claude Generated"""

    def __init__(self, presets: 'InstitutionPresets | None' = None):
        super().__init__()
        self._presets = presets  # Store for initializePage - Claude Generated
        self.setTitle("LLM-Anbieter einrichten")
        self.setSubTitle("Wählen Sie aus, wo Ihre KI-Modelle laufen sollen")

        # Store values from current configuration
        self.provider_type = "ollama"
        self.provider_name = "My Ollama Server"
        self.base_url = "http://localhost:11434"
        self.api_key = ""
        self.available_models = []

        layout = QVBoxLayout()

        # Provider selection
        self.provider_group = QButtonGroup()
        provider_box = QGroupBox("LLM-Anbieter auswählen")
        provider_layout = QVBoxLayout()

        self.ollama_radio = QRadioButton("🔹 Ollama (Lokaler Server) - Empfohlen")
        self.ollama_radio.setChecked(True)
        self.ollama_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.ollama_radio, 0)
        provider_layout.addWidget(self.ollama_radio)

        self.openai_radio = QRadioButton("🔷 OpenAI-Compatible API")
        self.openai_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.openai_radio, 1)
        provider_layout.addWidget(self.openai_radio)

        self.gemini_radio = QRadioButton("🟡 Google Gemini API")
        self.gemini_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.gemini_radio, 2)
        provider_layout.addWidget(self.gemini_radio)

        self.anthropic_radio = QRadioButton("🔴 Anthropic Claude API")
        self.anthropic_radio.toggled.connect(self._on_provider_changed)
        self.provider_group.addButton(self.anthropic_radio, 3)
        provider_layout.addWidget(self.anthropic_radio)

        provider_box.setLayout(provider_layout)
        layout.addWidget(provider_box)

        # Configuration fields (dynamic based on provider)
        config_box = QGroupBox("Konfiguration")
        self.config_layout = QVBoxLayout()

        # Ollama - Host and Port
        self.ollama_host_label = QLabel("Host:")
        self.ollama_host_input = QLineEdit()
        self.ollama_host_input.setText("localhost")
        self.config_layout.addWidget(self.ollama_host_label)
        self.config_layout.addWidget(self.ollama_host_input)

        self.ollama_port_label = QLabel("Port:")
        self.ollama_port_input = QSpinBox()
        self.ollama_port_input.setValue(11434)
        self.ollama_port_input.setRange(1, 65535)
        self.config_layout.addWidget(self.ollama_port_label)
        self.config_layout.addWidget(self.ollama_port_input)

        # API Key (for cloud providers)
        self.api_key_label = QLabel("API-Schlüssel:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setVisible(False)
        self.api_key_label.setVisible(False)
        self.config_layout.addWidget(self.api_key_label)
        self.config_layout.addWidget(self.api_key_input)

        # Base URL (for OpenAI-compatible)
        self.base_url_label = QLabel("API Basis-URL:")
        self.base_url_input = QLineEdit()
        self.base_url_input.setText("https://api.openai.com/v1")
        self.base_url_input.setToolTip("OpenAI: https://api.openai.com/v1 | Lokal: http://localhost:8000/v1")
        self.base_url_input.setVisible(False)
        self.base_url_label.setVisible(False)
        self.config_layout.addWidget(self.base_url_label)
        self.config_layout.addWidget(self.base_url_input)

        # Auto-test timer for input changes - Claude Generated
        self._auto_test_timer = QTimer()
        self._auto_test_timer.setSingleShot(True)
        self._auto_test_timer.timeout.connect(self._auto_test_connection)

        # Connect input changes to trigger auto-test (with debounce)
        self.ollama_host_input.textChanged.connect(self._schedule_auto_test)
        self.ollama_port_input.valueChanged.connect(self._schedule_auto_test)
        self.api_key_input.textChanged.connect(self._schedule_auto_test)
        self.base_url_input.textChanged.connect(self._schedule_auto_test)

        config_box.setLayout(self.config_layout)
        layout.addWidget(config_box)

        # Test connection button
        self.test_button = QPushButton("🧪 Verbindung testen")
        self.test_button.clicked.connect(self._test_connection)
        layout.addWidget(self.test_button)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Preset hint label (hidden until presets applied) - Claude Generated
        self._preset_hint = QLabel("🏛️ (Institutions-Standard)")
        self._preset_hint.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        self._preset_hint.setVisible(False)
        layout.addWidget(self._preset_hint)

        layout.addStretch()
        self.setLayout(layout)

        # Apply institution presets after all widgets are created - Claude Generated
        if presets and presets.has_llm():
            self._apply_presets(presets)

    def _apply_presets(self, presets: 'InstitutionPresets') -> None:
        """Pre-fill LLM fields from institution presets - Claude Generated"""
        provider_map = {
            'ollama': self.ollama_radio,
            'openai_compatible': self.openai_radio,
            'gemini': self.gemini_radio,
            'anthropic': self.anthropic_radio,
        }
        if presets.llm_provider_type in provider_map:
            provider_map[presets.llm_provider_type].setChecked(True)

        if presets.llm_base_url:
            provider = presets.llm_provider_type
            if provider in ('ollama', ''):
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(presets.llm_base_url)
                    host = parsed.hostname or presets.llm_base_url
                    port = parsed.port or 11434
                    self.ollama_host_input.setText(host)
                    self.ollama_port_input.setValue(port)
                except Exception:
                    self.ollama_host_input.setText(presets.llm_base_url)
            elif provider == 'openai_compatible':
                self.base_url_input.setText(presets.llm_base_url)

        if presets.llm_api_key:
            self.api_key_input.setText(presets.llm_api_key)

        if presets.llm_provider_name:
            self.provider_name = presets.llm_provider_name

        self._preset_hint.setVisible(True)
        logger.debug(f"LLM preset applied: provider={presets.llm_provider_type}")

    def initializePage(self):
        """Auto-test LLM connection if presets have valid config - Claude Generated"""
        super().initializePage()
        # Auto-test if we have presets with URL
        if self._presets and self._presets.has_llm() and self._presets.llm_base_url:
            # Delay test slightly to let UI settle
            QTimer.singleShot(500, self._auto_test_connection)

    def _auto_test_connection(self):
        """Silent connection test without blocking UI - Claude Generated"""
        # Only test if not already tested
        if self.available_models:
            return

        # Check if we have enough input to test
        selected = self.provider_group.checkedId()
        can_test = False

        if selected == 0:  # Ollama
            host = self.ollama_host_input.text().strip()
            can_test = bool(host)  # At least a host
        elif selected == 1:  # OpenAI-compatible
            url = self.base_url_input.text().strip()
            can_test = bool(url)
        elif selected in [2, 3]:  # Gemini or Anthropic
            api_key = self.api_key_input.text().strip()
            can_test = len(api_key) >= 10  # Reasonable API key length

        if can_test:
            self._test_connection()

    def _schedule_auto_test(self):
        """Schedule auto-test after user stops typing (1.5s debounce) - Claude Generated"""
        # Cancel any pending test
        self._auto_test_timer.stop()
        # Schedule new test in 1.5 seconds
        self._auto_test_timer.start(1500)

    def _on_provider_changed(self):
        """Update UI when provider selection changes - Claude Generated (fixed visibility)"""
        selected = self.provider_group.checkedId()

        # Ollama settings
        self.ollama_host_label.setVisible(selected == 0)
        self.ollama_host_input.setVisible(selected == 0)
        self.ollama_port_label.setVisible(selected == 0)
        self.ollama_port_input.setVisible(selected == 0)

        # OpenAI-compatible settings (selected == 1)
        self.base_url_label.setVisible(selected == 1)
        self.base_url_input.setVisible(selected == 1)

        # Cloud API settings (Gemini=2, Anthropic=3, OpenAI-Compatible=1)
        self.api_key_label.setVisible(selected in [1, 2, 3])
        self.api_key_input.setVisible(selected in [1, 2, 3])

        # Update API Key label text for clarity - Claude Generated
        if selected == 1:
            self.api_key_label.setText("API-Schlüssel (optional für Kompatibilitätsmodus):")
        elif selected in [2, 3]:
            self.api_key_label.setText("API-Schlüssel:")

        self.status_label.setText("")

    def _test_connection(self):
        """Test LLM provider connection - Claude Generated"""
        selected = self.provider_group.checkedId()

        if selected == 0:  # Ollama
            host = self.ollama_host_input.text()
            port = self.ollama_port_input.value()
            result = OllamaConnectionValidator.test_native(host, port)
            self.provider_type = "ollama"
            self.base_url = f"http://{host}:{port}"

        elif selected == 1:  # OpenAI-compatible
            base_url = self.base_url_input.text()
            api_key = self.api_key_input.text()
            result = OllamaConnectionValidator.test_openai_compatible(base_url, api_key)
            self.provider_type = "openai_compatible"
            self.base_url = base_url
            self.api_key = api_key

        elif selected == 2:  # Gemini
            api_key = self.api_key_input.text()
            result = APIKeyValidator.validate_gemini(api_key)
            self.provider_type = "gemini"
            self.api_key = api_key

        elif selected == 3:  # Anthropic
            api_key = self.api_key_input.text()
            result = APIKeyValidator.validate_anthropic(api_key)
            self.provider_type = "anthropic"
            self.api_key = api_key

        else:
            result = SetupResult(False, "Unknown provider")

        # Display result
        if result.success:
            self.status_label.setText(f"✅ {result.message}")
            self.status_label.setStyleSheet("color: green;")
            self.available_models = result.data or []
            self.provider_name = self.provider_type  # BUGFIX: Use actual provider type for consistency
        else:
            self.status_label.setText(f"❌ {result.message}")
            self.status_label.setStyleSheet("color: red;")

    def validatePage(self) -> bool:
        """Validate LLM configuration before proceeding - Claude Generated"""
        # Allow proceeding even without testing - user can test later
        if not self.available_models:
            # Warn but don't block - "Yes" as default for easier progression
            msgBox = QMessageBox(self)
            msgBox.setIcon(QMessageBox.Icon.Warning)
            msgBox.setWindowTitle("Keine Verbindung getestet")
            msgBox.setText("Sie haben die LLM-Verbindung noch nicht getestet.\n\n"
                          "Trotzdem fortfahren? (Sie können die Verbindung nach dem Setup testen)")
            msgBox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msgBox.setDefaultButton(QMessageBox.StandardButton.Yes)
            msgBox.button(QMessageBox.StandardButton.Yes).setText("Ja, fortfahren")
            msgBox.button(QMessageBox.StandardButton.No).setText("Zurück")
            return msgBox.exec() == QMessageBox.StandardButton.Yes
        return True


class ModelSelectionPage(QWizardPage):
    """Model selection for different pipeline tasks - Claude Generated"""

    def __init__(self, presets=None):
        super().__init__()
        self.setTitle("Modell-Auswahl für Pipeline-Schritte")
        self.setSubTitle("Wählen Sie für jeden Schritt das optimale Modell")

        # Store selections {task_type: model_name}
        self.task_model_selections = {}
        self._presets = presets

        layout = QVBoxLayout()

        # Explanation
        explanation = QLabel(
            "Verschiedene Pipeline-Schritte haben unterschiedliche Anforderungen:\n"
            "• Initialisation & Keywords: Benötigen Reasoning-Fähigkeiten\n"
            "• Classification: Benötigt strukturiertes Denken\n"
            "• Vision: Benötigt Bildverständnis\n\n"
            "Sie können für jeden Schritt das optimale Modell wählen.\n"
            "Oder nutzen Sie den Button unten um ein Modell für alle zu verwenden."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Task selection form
        self.task_combos = {}

        # Use shared LLM task configuration from config_models - Claude Generated
        # Single source of truth for consistent task lists across all wizards/dialogs
        from ..utils.config_models import LLM_TASK_DISPLAY_INFO

        form_layout = QFormLayout()

        for task_type, task_label, task_desc in LLM_TASK_DISPLAY_INFO:
            # Use the enum's value (lower‑case) as the identifier – this aligns with the
            # configuration builder (setup_utils) which expects ``task_type.value`` as the key.
            task_key = task_type.value
            combo = QComboBox()
            combo.setMinimumWidth(300)
            combo.setToolTip(task_desc)

            label = QLabel(f"{task_label}:")
            label.setToolTip(task_desc)

            form_layout.addRow(label, combo)
            self.task_combos[task_key] = combo

        layout.addLayout(form_layout)

        # "Use same model for all" button
        self.use_same_btn = QPushButton("🔄 Gleiches Modell für alle Tasks verwenden")
        self.use_same_btn.clicked.connect(self._use_same_for_all)
        layout.addWidget(self.use_same_btn)

        layout.addStretch()
        self.setLayout(layout)

    def initializePage(self):
        """Populate combo boxes with available models - Claude Generated"""
        wizard = self.wizard()
        available_models = getattr(wizard.llm_page, 'available_models', [])

        if not available_models:
            # Use provider-specific default models when no test was done
            provider_type = getattr(wizard.llm_page, 'provider_type', 'ollama')
            available_models = self._get_default_models(provider_type)

        if not available_models:
            # Still no models - show placeholder
            for combo in self.task_combos.values():
                combo.addItem("(Keine Modelle verfügbar)")
            return

        # Populate all combos
        for task_key, combo in self.task_combos.items():
            combo.clear()
            combo.addItems(available_models)

            # Set intelligent defaults based on task
            if 'vision' in task_key:
                # Prefer vision models for vision tasks
                vision_models = [m for m in available_models
                               if 'vision' in m.lower() or 'llava' in m.lower()]
                if vision_models:
                    combo.setCurrentText(vision_models[0])
            else:
                # Use first available model as default
                combo.setCurrentIndex(0)

        # Apply preset models after populating combos - Claude Generated
        if self._presets and self._presets.has_models():
            self._apply_presets(self._presets)

    def _get_default_models(self, provider_type: str) -> list:
        """Get default models for provider type - Claude Generated"""
        defaults = {
            'ollama': ['mistral', 'llama3', 'llama2'],
            'gemini': ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro'],
            'anthropic': ['claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022'],
            'openai_compatible': ['gpt-4', 'gpt-3.5-turbo'],
        }
        return defaults.get(provider_type, [])

    def _apply_presets(self, presets) -> None:
        """Pre-select model combos from institution preset values - Claude Generated"""
        for task_key, combo in self.task_combos.items():
            model = presets.llm_task_models.get(task_key) or presets.llm_default_model
            if not model:
                continue
            idx = combo.findText(model)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _use_same_for_all(self):
        """Set all tasks to use the same model - Claude Generated"""
        if not self.task_combos:
            return

        # Get first combo's selection
        first_model = list(self.task_combos.values())[0].currentText()

        # Apply to all
        for combo in self.task_combos.values():
            combo.setCurrentText(first_model)

        QMessageBox.information(
            self,
            "Modelle eingestellt",
            f"✅ Alle Tasks verwenden jetzt: {first_model}"
        )

    def validatePage(self) -> bool:
        """Save selections - Claude Generated

        STRICT VALIDATION: Reject auto-select, use only explicit model names
        """
        from PyQt6.QtWidgets import QMessageBox

        # Clear previous selections
        self.task_model_selections = {}

        # Validate each task has a REAL model selected (not auto-select)
        for task_key, combo in self.task_combos.items():
            model = combo.currentText()

            # Check for invalid selections
            if not model:
                QMessageBox.warning(
                    self,
                    "Modell erforderlich",
                    f"Bitte wählen Sie ein Modell für {task_key} aus."
                )
                return False

            # Reject placeholder text
            if "(Keine Modelle" in model or "(No models" in model:
                QMessageBox.warning(
                    self,
                    "Modell erforderlich",
                    f"Keine Modelle verfügbar für {task_key}. Bitte überprüfen Sie die LLM-Verbindung."
                )
                return False

            # CRITICAL: Reject "auto-select" and "default" - Wizard MUST use explicit models
            model_lower = model.lower()
            if model_lower in ["auto-select", "(auto-select)", "default", "auto"]:
                QMessageBox.warning(
                    self,
                    "Automatische Auswahl nicht erlaubt",
                    f"Bitte wählen Sie ein spezifisches Modell für {task_key},\n"
                    f"nicht automatische Auswahl (Auto-select)."
                )
                return False

            # Valid selection - save it
            self.task_model_selections[task_key] = model

        return True


class DatabaseSetupPage(QWizardPage):
    """Unified database and GND configuration page - Claude Generated

    Merges SQLite/MariaDB database configuration with GND download options.
    Presets can pre-configure database connection and skip GND download if data exists.
    """

    def __init__(self, presets: Optional[InstitutionPresets] = None):
        super().__init__()
        self._presets = presets
        self.setTitle("Datenbank-Konfiguration")
        self.setSubTitle("Wählen Sie die Datenbank für ALIMA und GND-Normdaten")

        # Track downloaded file for post-wizard import
        self.downloaded_xml_path = None
        self._connection_test_result = None

        layout = QVBoxLayout()

        # ===== DATABASE TYPE SECTION =====
        db_type_group = QGroupBox("Datenbank-Typ")
        db_type_layout = QVBoxLayout()

        # Radio buttons for database type
        self.sqlite_radio = QRadioButton("SQLite (Lokal)")
        self.sqlite_radio.setChecked(True)
        self.mariadb_radio = QRadioButton("MariaDB/MySQL (Server)")
        db_type_layout.addWidget(self.sqlite_radio)
        db_type_layout.addWidget(self.mariadb_radio)

        db_type_group.setLayout(db_type_layout)
        layout.addWidget(db_type_group)

        # ===== SQLITE CONFIGURATION =====
        self.sqlite_group = QGroupBox("SQLite-Konfiguration")
        sqlite_layout = QFormLayout()

        sqlite_path_layout = QHBoxLayout()
        self.sqlite_path_input = QLineEdit()
        self.sqlite_path_input.setPlaceholderText("Pfad zur alima_knowledge.db")
        sqlite_path_layout.addWidget(self.sqlite_path_input)
        sqlite_browse_btn = QPushButton("Durchsuchen...")
        sqlite_browse_btn.clicked.connect(self._browse_sqlite_path)
        sqlite_path_layout.addWidget(sqlite_browse_btn)
        sqlite_layout.addRow("Datenbank-Pfad:", sqlite_path_layout)

        self.sqlite_group.setLayout(sqlite_layout)
        layout.addWidget(self.sqlite_group)

        # ===== MARIADB CONFIGURATION =====
        self.mariadb_group = QGroupBox("MariaDB/MySQL-Konfiguration")
        mariadb_layout = QFormLayout()

        # Host and Port
        host_layout = QHBoxLayout()
        self.host_input = QLineEdit()
        self.host_input.setPlaceholderText("localhost")
        self.host_input.setText("localhost")
        host_layout.addWidget(self.host_input)
        host_layout.addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(3306)
        host_layout.addWidget(self.port_input)
        mariadb_layout.addRow("Host:", host_layout)

        # Database name
        self.database_input = QLineEdit()
        self.database_input.setText("alima_knowledge")
        mariadb_layout.addRow("Datenbank:", self.database_input)

        # Username and Password
        self.username_input = QLineEdit()
        self.username_input.setText("alima")
        mariadb_layout.addRow("Benutzer:", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        mariadb_layout.addRow("Passwort:", self.password_input)

        self.mariadb_group.setLayout(mariadb_layout)
        self.mariadb_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.mariadb_group)

        # ===== SHARED DATABASE TEST BUTTON =====
        # Placed after both SQLite and MariaDB groups, visible for both
        test_layout = QHBoxLayout()
        self.test_db_btn = QPushButton("🔍 Datenbank testen & GND prüfen")
        self.test_db_btn.clicked.connect(self._test_database_connection)
        test_layout.addWidget(self.test_db_btn)
        test_layout.addStretch()
        layout.addLayout(test_layout)

        # Connection status (shared for both SQLite and MariaDB)
        self.connection_status = QLabel("")
        layout.addWidget(self.connection_status)

        # Radio button connections
        self.sqlite_radio.toggled.connect(self._update_db_type_visibility)

        # ===== GND DATA SECTION =====
        gnd_group = QGroupBox("GND-Normdaten (Schlagwörter)")
        gnd_layout = QVBoxLayout()

        explanation = QLabel(
            "Die GND (Gemeinsame Normdatei) enthält deutsche Schlagwörter.\n"
            "Optionen:\n"
            "• Download: ~300 MB, Import läuft im Hintergrund\n"
            "• Import: Bestehende .db-Datei kopieren\n"
            "• Überspringen: Daten werden bei Bedarf nachgeladen"
        )
        explanation.setWordWrap(True)
        gnd_layout.addWidget(explanation)

        # GND options
        self.gnd_download_radio = QRadioButton("📥 GND-Datenbank herunterladen")
        self.gnd_import_radio = QRadioButton("🗄️ Bestehende Datenbankdatei importieren")
        self.gnd_skip_radio = QRadioButton("⏭️ Überspringen")
        self.gnd_preset_radio = QRadioButton("✅ Preset-Datenbank verwenden (bereits konfiguriert)")
        self.gnd_preset_radio.setVisible(False)  # Only visible when preset has DB

        gnd_layout.addWidget(self.gnd_download_radio)
        gnd_layout.addWidget(self.gnd_import_radio)
        gnd_layout.addWidget(self.gnd_skip_radio)
        gnd_layout.addWidget(self.gnd_preset_radio)

        self.gnd_download_radio.setChecked(True)

        # GND file selector
        self.gnd_file_layout = QHBoxLayout()
        self.gnd_file_layout.addWidget(QLabel("Datei:"))
        self.gnd_file_input = QLineEdit()
        self.gnd_file_input.setPlaceholderText("Pfad zur .db oder .xml.gz Datei")
        self.gnd_file_layout.addWidget(self.gnd_file_input)
        gnd_file_btn = QPushButton("Durchsuchen...")
        gnd_file_btn.clicked.connect(self._browse_gnd_file)
        self.gnd_file_layout.addWidget(gnd_file_btn)
        gnd_layout.addLayout(self.gnd_file_layout)

        # GND status
        self.gnd_status_label = QLabel("")
        gnd_layout.addWidget(self.gnd_status_label)

        # GND download button
        self.gnd_download_btn = QPushButton("⬇️ Download starten")
        self.gnd_download_btn.clicked.connect(self._download_gnd)
        gnd_layout.addWidget(self.gnd_download_btn)

        gnd_group.setLayout(gnd_layout)
        layout.addWidget(gnd_group)

        layout.addStretch()
        self.setLayout(layout)

        # Apply presets if available
        if presets:
            self._apply_presets(presets)

        self._update_gnd_file_visibility()
        self.gnd_download_radio.toggled.connect(self._update_gnd_file_visibility)
        self.gnd_import_radio.toggled.connect(self._update_gnd_file_visibility)

    def _update_db_type_visibility(self):
        """Show/hide database type specific fields"""
        is_sqlite = self.sqlite_radio.isChecked()
        self.sqlite_group.setVisible(is_sqlite)
        self.mariadb_group.setVisible(not is_sqlite)

    def initializePage(self):
        """Auto-test database connection if presets have config - Claude Generated"""
        super().initializePage()
        # Auto-test if we have preset database config
        if self._presets and self._presets.has_database_config():
            # Delay test slightly to let UI settle
            QTimer.singleShot(500, self._auto_test_database)

    def _auto_test_database(self):
        """Silent database test - Claude Generated"""
        # Only test if not already tested
        if hasattr(self, '_connection_test_result') and self._connection_test_result:
            return
        self._test_database_connection()

    def _update_gnd_file_visibility(self):
        """Show/hide GND file selector based on selection"""
        show_file = self.gnd_import_radio.isChecked()
        for i in range(self.gnd_file_layout.count()):
            widget = self.gnd_file_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(show_file)

    def _browse_sqlite_path(self):
        """Browse for SQLite database path"""
        from ..utils.config_models import get_default_db_path
        default_path = get_default_db_path()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "SQLite-Datenbank speichern",
            str(default_path),
            "SQLite-Datenbank (*.db);;Alle Dateien (*)"
        )
        if file_path:
            self.sqlite_path_input.setText(file_path)

    def _browse_gnd_file(self):
        """Browse for GND file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "GND-Datei auswählen",
            "",
            "SQLite-Datenbank (*.db);;XML-Dateien (*.xml);;Gzip-Dateien (*.gz);;Alle Dateien (*)"
        )
        if file_path:
            self.gnd_file_input.setText(file_path)

    def _test_database_connection(self):
        """Test database connection and check for GND data - Claude Generated (unified for SQLite/MariaDB)"""
        from PyQt6.QtCore import QCoreApplication
        from ..utils.qt_plugin_setup import setup_qt_plugin_paths
        from ..core.database_manager import DatabaseManager
        from ..utils.config_models import DatabaseConfig, get_default_db_path

        # Ensure QCoreApplication exists
        if not QCoreApplication.instance():
            QCoreApplication([])

        setup_qt_plugin_paths()

        # Build config based on selected database type
        if self.sqlite_radio.isChecked():
            db_path = self.sqlite_path_input.text().strip()
            if not db_path:
                db_path = str(get_default_db_path())
            config = DatabaseConfig(
                db_type='sqlite',
                sqlite_path=db_path
            )
        else:
            config = DatabaseConfig(
                db_type='mariadb',
                host=self.host_input.text().strip() or 'localhost',
                port=self.port_input.value(),
                database=self.database_input.text().strip() or 'alima_knowledge',
                username=self.username_input.text().strip() or 'alima',
                password=self.password_input.text()
            )

        try:
            db = DatabaseManager(config, "wizard_test")
            success, message = db.test_connection()

            if not success:
                self.connection_status.setText(f"❌ {message}")
                self.connection_status.setStyleSheet("color: red;")
                self._connection_test_result = False
                db.close_connection()
                return

            # Connection successful - check for GND data
            gnd_count = 0
            try:
                # Check if gnd_entries table exists and has data
                result = db.fetch_scalar("SELECT COUNT(*) FROM gnd_entries")
                if result:
                    gnd_count = int(result)
            except Exception:
                pass  # Table doesn't exist yet - that's fine

            db.close_connection()

            # Build status message
            if gnd_count > 0:
                status_msg = f"✅ Verbunden - {gnd_count:,} GND-Einträge gefunden"
                self.connection_status.setStyleSheet("color: green;")

                # Auto-select "Skip" since GND data already exists
                self.gnd_skip_radio.setChecked(True)
                self.gnd_status_label.setText(f"✅ GND-Daten bereits vorhanden ({gnd_count:,} Einträge)")
                self.gnd_status_label.setStyleSheet("color: green;")
            else:
                status_msg = "✅ Verbunden - Keine GND-Daten"
                self.connection_status.setStyleSheet("color: #FF8C00;")  # Orange for warning

                # Suggest download if no GND data
                self.gnd_status_label.setText("ℹ️ GND-Daten nicht vorhanden - Download empfohlen")
                self.gnd_status_label.setStyleSheet("color: #FF8C00;")

            self.connection_status.setText(status_msg)
            self._connection_test_result = True

        except Exception as e:
            self.connection_status.setText(f"❌ Fehler: {str(e)}")
            self.connection_status.setStyleSheet("color: red;")
            self._connection_test_result = False

    def _download_gnd(self):
        """Download GND database"""
        if not self.gnd_download_radio.isChecked():
            return

        from PyQt6.QtWidgets import QApplication

        progress = QProgressDialog(
            "GND-Datenbank wird heruntergeladen...", "Abbrechen", 0, 100, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)

        try:
            def progress_callback(percent):
                progress.setValue(percent)
                QApplication.processEvents()

            result = GNDDatabaseDownloader.download(progress_callback)
            progress.close()

            if result.success:
                self.downloaded_xml_path = result.data
                self.gnd_status_label.setText("✅ GND-Datenbank heruntergeladen")
                self.gnd_status_label.setStyleSheet("color: green;")
                self.gnd_download_btn.setEnabled(False)
            else:
                self.gnd_status_label.setText(f"❌ Download fehlgeschlagen: {result.message}")
                self.gnd_status_label.setStyleSheet("color: red;")
        except Exception as e:
            progress.close()
            self.gnd_status_label.setText(f"❌ Fehler: {str(e)}")
            self.gnd_status_label.setStyleSheet("color: red;")

    def _apply_presets(self, presets: InstitutionPresets):
        """Apply database presets if available"""
        if not presets or not presets.has_database_config():
            return

        if presets.db_type == 'sqlite':
            self.sqlite_radio.setChecked(True)
            if presets.db_sqlite_path:
                self.sqlite_path_input.setText(presets.db_sqlite_path)
        elif presets.db_type in ('mariadb', 'mysql'):
            self.mariadb_radio.setChecked(True)
            if presets.db_host:
                self.host_input.setText(presets.db_host)
            if presets.db_port:
                self.port_input.setValue(presets.db_port)
            if presets.db_database:
                self.database_input.setText(presets.db_database)
            if presets.db_username:
                self.username_input.setText(presets.db_username)
            if presets.db_password:
                self.password_input.setText(presets.db_password)

        # If preset has preloaded GND data, show preset option
        if presets.gnd_data_preloaded:
            self.gnd_preset_radio.setVisible(True)
            self.gnd_preset_radio.setChecked(True)
            self.gnd_status_label.setText("✅ Preset-Datenbank enthält bereits GND-Daten")
            self.gnd_status_label.setStyleSheet("color: green;")

    def get_database_config(self) -> dict:
        """Get database configuration from the form"""
        if self.sqlite_radio.isChecked():
            return {
                'db_type': 'sqlite',
                'sqlite_path': self.sqlite_path_input.text().strip() or None
            }
        else:
            return {
                'db_type': 'mariadb',
                'host': self.host_input.text().strip() or 'localhost',
                'port': self.port_input.value(),
                'database': self.database_input.text().strip() or 'alima_knowledge',
                'username': self.username_input.text().strip() or 'alima',
                'password': self.password_input.text()
            }

    def get_gnd_action(self) -> str:
        """Get GND action: 'download', 'import', 'skip', or 'preset'"""
        if self.gnd_download_radio.isChecked():
            return 'download'
        elif self.gnd_import_radio.isChecked():
            return 'import'
        elif self.gnd_preset_radio.isChecked():
            return 'preset'
        else:
            return 'skip'

    def validatePage(self) -> bool:
        """Validate before proceeding - Claude Generated (updated for DatabaseSetupPage)"""
        # Check database path for SQLite
        if self.sqlite_radio.isChecked():
            path = self.sqlite_path_input.text().strip()
            if path:
                # Ensure parent directory exists
                try:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(self, "Pfadfehler", f"Kann Verzeichnis nicht erstellen:\n{str(e)}")
                    return False

        # Check GND action validation
        if self.gnd_download_radio.isChecked():
            if "✅" not in self.gnd_status_label.text():
                # User selected download but hasn't done it yet - offer clear choices
                msgBox = QMessageBox(self)
                msgBox.setIcon(QMessageBox.Icon.Question)
                msgBox.setWindowTitle("GND-Datenbank noch nicht heruntergeladen")
                msgBox.setText("Sie haben 'Download' ausgewählt, aber noch nicht gestartet.")
                msgBox.setInformativeText(
                    "Optionen:\n"
                    "• Jetzt herunterladen: Import läuft nach Setup im Hintergrund (~5-10 Min)\n"
                    "• Überspringen: Lobid-API lädt Daten bei Bedarf nach (langsamer)\n"
                    "• Zurück: Entscheidung überdenken\n\n"
                    "Hinweis: Die Datenbank wird für Verifikation in beiden Fällen verwendet."
                )
                msgBox.setStandardButtons(
                    QMessageBox.StandardButton.Retry |      # "Jetzt herunterladen"
                    QMessageBox.StandardButton.Ignore |     # "Überspringen"
                    QMessageBox.StandardButton.Cancel       # "Zurück"
                )
                msgBox.setDefaultButton(QMessageBox.StandardButton.Retry)

                # Customize button labels
                msgBox.button(QMessageBox.StandardButton.Retry).setText("Jetzt herunterladen")
                msgBox.button(QMessageBox.StandardButton.Ignore).setText("Überspringen")
                msgBox.button(QMessageBox.StandardButton.Cancel).setText("Zurück")

                result = msgBox.exec()

                if result == QMessageBox.StandardButton.Retry:
                    # User wants to download now - trigger download and stay on page
                    self._download_gnd()
                    return False  # Stay on page to show download progress
                elif result == QMessageBox.StandardButton.Ignore:
                    # User explicitly wants to skip despite selecting download option
                    return True  # Proceed to next page without download
                else:  # Cancel
                    # User wants to go back and reconsider
                    return False  # Stay on page

        elif self.gnd_import_radio.isChecked():
            if not self.gnd_file_input.text():
                QMessageBox.warning(self, "Datei erforderlich",
                                    "Bitte wählen Sie eine GND-Datei aus.")
                return False
            if not Path(self.gnd_file_input.text()).exists():
                QMessageBox.warning(self, "Datei nicht gefunden",
                                    "Die ausgewählte Datei wurde nicht gefunden.")
                return False

        elif self.gnd_skip_radio.isChecked():
            # Skip confirmation - only ask if GND data wasn't detected
            # If status shows GND data found, no need to confirm
            gnd_status = self.gnd_status_label.text()
            if "✅" in gnd_status and "GND" in gnd_status:
                # GND data was detected - proceed without confirmation
                return True

            # No GND data detected - confirm skip
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("GND-Download überspringen?")
            msgBox.setText("GND-Download überspringen?")
            msgBox.setInformativeText(
                "Wenn Sie überspringen:\n"
                "• Lobid-API lädt Daten bei Bedarf nach (langsamer beim ersten Zugriff)\n"
                "• Volle Funktionalität bleibt erhalten\n"
                "• Sie können später über Tools → GND-Datenbank importieren nachholen\n\n"
                "Hinweis: Die Datenbank wird auch mit Lobid für Verifikation verwendet."
            )
            msgBox.setStandardButtons(
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.Cancel
            )
            msgBox.setDefaultButton(QMessageBox.StandardButton.Cancel)

            # Customize button text
            msgBox.button(QMessageBox.StandardButton.Yes).setText("Überspringen")
            msgBox.button(QMessageBox.StandardButton.Cancel).setText("Zurück")

            result = msgBox.exec()
            # Yes = skip, Cancel = stay on page
            return result == QMessageBox.StandardButton.Yes

        return True


class CatalogSetupPage(QWizardPage):
    """Optional Libero SOAP catalog configuration page - Claude Generated"""

    def __init__(self, presets: 'InstitutionPresets | None' = None):
        super().__init__()
        self.setTitle("Katalog-Konfiguration")
        self.setSubTitle("Optional: Verbindung zum Bibliothekskatalog einrichten")

        layout = QVBoxLayout()

        explanation = QLabel(
            "Für die DK-Analyse und UB-Suche kann ein lokaler Bibliothekskatalog "
            "(Libero SOAP) eingebunden werden. Dieser Schritt ist optional — "
            "lassen Sie die Felder leer, um ihn zu überspringen."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        soap_box = QGroupBox("Libero SOAP-Zugang")
        form = QFormLayout()

        self.soap_search_url = QLineEdit()
        self.soap_search_url.setPlaceholderText(
            "https://katalog.ub.example.de/libero/LiberoWebServices.CatalogueSearcher.cls"
        )
        form.addRow("SOAP Search URL:", self.soap_search_url)

        self.soap_details_url = QLineEdit()
        self.soap_details_url.setPlaceholderText(
            "https://katalog.ub.example.de/libero/LiberoWebServices.LibraryAPI.cls"
        )
        form.addRow("SOAP Details URL:", self.soap_details_url)

        self.catalog_token = QLineEdit()
        self.catalog_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.catalog_token.setPlaceholderText("Auth-Token (falls erforderlich)")
        form.addRow("Token:", self.catalog_token)

        self.web_search_url = QLineEdit()
        self.web_search_url.setPlaceholderText(
            "https://katalog.ub.example.de/Search/Results"
        )
        self.web_search_url.setToolTip("Web-Frontend-URL für Scraping-Fallback (optional)")
        form.addRow("Web Search URL:", self.web_search_url)

        self.web_record_url = QLineEdit()
        self.web_record_url.setPlaceholderText(
            "https://katalog.ub.example.de/Record/"
        )
        self.web_record_url.setToolTip("Web-Frontend-Record-URL für Scraping-Fallback (optional)")
        form.addRow("Web Record URL:", self.web_record_url)

        soap_box.setLayout(form)
        layout.addWidget(soap_box)

        # Token login button row - Claude Generated
        token_btn_layout = QHBoxLayout()
        self.token_login_btn = QPushButton("🔑 Token erstellen...")
        self.token_login_btn.clicked.connect(self._fetch_token_dialog)
        token_btn_layout.addWidget(self.token_login_btn)
        token_btn_layout.addStretch()
        layout.addLayout(token_btn_layout)

        # Connection test row
        test_layout = QHBoxLayout()
        self.test_button = QPushButton("🧪 Verbindung testen")
        self.test_button.clicked.connect(self._test_connection)
        test_layout.addWidget(self.test_button)
        test_layout.addStretch()
        layout.addLayout(test_layout)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

        # Apply institution presets after all widgets are created - Claude Generated
        if presets and presets.has_catalog():
            self._apply_presets(presets)

    def _apply_presets(self, presets: 'InstitutionPresets') -> None:
        """Pre-fill catalog fields from institution presets - Claude Generated"""
        if presets.catalog_soap_search_url:
            self.soap_search_url.setText(presets.catalog_soap_search_url)
        if presets.catalog_soap_details_url:
            self.soap_details_url.setText(presets.catalog_soap_details_url)
        if presets.catalog_token:
            self.catalog_token.setText(presets.catalog_token)
        if presets.catalog_web_search_url:
            self.web_search_url.setText(presets.catalog_web_search_url)
        if presets.catalog_web_record_url:
            self.web_record_url.setText(presets.catalog_web_record_url)
        logger.debug("Catalog preset applied")

    def _fetch_token_dialog(self):
        """Open Libero login dialog and write token into token field - Claude Generated"""
        url = self.soap_search_url.text().strip()
        if not url:
            QMessageBox.warning(self, "URL fehlt",
                                "Bitte zuerst eine SOAP Search URL eintragen.")
            return
        from .libero_login_dialog import LiberoLoginDialog
        dialog = LiberoLoginDialog(soap_url=url, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.catalog_token.setText(dialog.token)

    def _test_connection(self):
        """Non-blocking HTTP reachability check - Claude Generated"""
        url = self.soap_search_url.text().strip()
        if not url:
            self.status_label.setText("⚠️  Bitte erst eine SOAP Search URL eingeben.")
            self.status_label.setStyleSheet("color: orange;")
            return

        self.test_button.setEnabled(False)
        self.status_label.setText("🔄 Teste Verbindung...")
        self.status_label.setStyleSheet("color: gray;")

        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            import requests
            r = requests.head(url, timeout=5)
            # SOAP server returns 400/405/500 for HEAD (expects POST) — still reachable
            if r.status_code in [200, 400, 405, 500]:
                self.status_label.setText(
                    f"✅ Server antwortet (HTTP {r.status_code}) — SOAP-Endpunkt erreichbar."
                )
                self.status_label.setStyleSheet("color: green;")
            else:
                self.status_label.setText(
                    f"⚠️  Server antwortete mit HTTP {r.status_code}. URL prüfen."
                )
                self.status_label.setStyleSheet("color: orange;")
        except Exception as exc:
            self.status_label.setText(f"❌ Verbindung fehlgeschlagen: {exc}")
            self.status_label.setStyleSheet("color: red;")
        finally:
            self.test_button.setEnabled(True)

    def validatePage(self) -> bool:
        """All fields optional — always valid - Claude Generated"""
        return True


class SummaryPage(QWizardPage):
    """Configuration summary page - Claude Generated"""

    def __init__(self):
        super().__init__()
        self.setTitle("Setup abgeschlossen!")
        self.setSubTitle("Überprüfen Sie Ihre ALIMA-Konfiguration")

        layout = QVBoxLayout()

        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

        # Completion message
        completion = QLabel(
            "✅ Ihre ALIMA-Konfiguration ist bereit!\n\n"
            "Sie können jetzt die Anwendung starten und mit der Analyse von Bibliotheksmaterialien beginnen."
        )
        completion.setWordWrap(True)
        layout.addWidget(completion)

        layout.addStretch()
        self.setLayout(layout)

    def initializePage(self):
        """Initialize with configuration summary - Claude Generated (enhanced with database config)"""
        # Get wizard to access configuration pages
        wizard = self.wizard()

        # Build database summary from database_page - Claude Generated
        db_type = "SQLite (Lokal)" if wizard.database_page.sqlite_radio.isChecked() else "MariaDB/MySQL (Server)"
        if wizard.database_page.sqlite_radio.isChecked():
            db_path = wizard.database_page.sqlite_path_input.text().strip()
            db_summary = f"Typ: {db_type}\n   Pfad: {db_path or '(Standard)'}"
        else:
            db_host = wizard.database_page.host_input.text().strip()
            db_name = wizard.database_page.database_input.text().strip()
            db_summary = f"Typ: {db_type}\n   Server: {db_host}\n   Datenbank: {db_name}"

        # Build GND summary from database_page
        gnd_action = wizard.database_page.get_gnd_action()
        gnd_option = "Download von DNB"
        if gnd_action == 'skip':
            gnd_option = "Überspringen (Lobid-API verwenden)"
        elif gnd_action == 'import':
            gnd_option = f"Datei importieren: {wizard.database_page.gnd_file_input.text()}"
        elif gnd_action == 'preset':
            gnd_option = "Preset-Datenbank (bereits konfiguriert)"

        # Build model selections summary - Claude Generated
        model_summary = ""
        if hasattr(wizard, 'model_page') and wizard.model_page.task_model_selections:
            model_summary = "\n🎯 Modell-Zuordnungen:\n"
            for task, model in wizard.model_page.task_model_selections.items():
                task_label = task.replace('_', ' ').title()
                model_summary += f"   {task_label}: {model}\n"
        else:
            model_summary = "\n🎯 Modell-Zuordnungen: Standard"

        # Build catalog summary - Claude Generated
        catalog_token = wizard.catalog_page.catalog_token.text().strip()
        catalog_url = wizard.catalog_page.soap_search_url.text().strip()
        if catalog_url:
            catalog_summary = f"Konfiguriert ✅ ({catalog_url[:50]}{'…' if len(catalog_url) > 50 else ''})"
        elif catalog_token:
            catalog_summary = "Konfiguriert ✅ (nur Token, URLs leer)"
        else:
            catalog_summary = "Übersprungen (kein SOAP konfiguriert)"

        summary = f"""
ALIMA Konfiguration - Zusammenfassung
══════════════════════════════════════

🤖 LLM-Anbieter:
   Typ: {wizard.llm_page.provider_type.title()}
   Name: {wizard.llm_page.provider_name}
   URL: {wizard.llm_page.base_url or "(keine URL)"}
   Verfügbare Modelle: {len(wizard.llm_page.available_models)}
{model_summary}

💾 Datenbank:
   {db_summary}

📚 GND-Datenbank:
   Option: {gnd_option}

🏛️ Katalog-SOAP:
   {catalog_summary}

✅ Setup abgeschlossen!
   Klicken Sie auf "Fertig stellen" um ALIMA zu starten.
        """
        self.summary_text.setText(summary)
