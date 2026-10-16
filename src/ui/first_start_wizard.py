#!/usr/bin/env python3
"""
ALIMA First-Start Setup Wizard
Guide new users through initial configuration (LLM provider and GND database)
Claude Generated
"""

from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QRadioButton, QButtonGroup, QComboBox, QPushButton, QProgressDialog,
    QMessageBox, QGroupBox, QSpinBox, QCheckBox, QFileDialog, QTextEdit, QDialog,
    QCompleter
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

    #: Sections the wizard actually asks about. Everything else on an existing
    #: config is left alone. - Claude Generated
    _WIZARD_OWNED_PROVIDER_FIELDS = (
        "providers", "preferred_provider", "preferred_model",
        "pipeline_default_provider", "pipeline_default_model",
        "vision_default_provider", "vision_default_model",
        "gemini_api_key", "anthropic_api_key",
        "gemini_preferred_model", "anthropic_preferred_model",
    )

    def _apply_onto_existing(self, wizard_config):
        """Merge the wizard's answers into the configuration already on disk.

        Returns ``wizard_config`` unchanged on a genuine first run (nothing to
        preserve). Otherwise the loaded config is the base and only the sections
        the wizard asks about are overwritten. - Claude Generated
        """
        try:
            existing = ConfigManager().load_config()
        except Exception as e:
            logger.warning(f"Bestehende Konfiguration nicht lesbar, nutze Wizard-Stand: {e}")
            return wizard_config

        if not getattr(existing.system_config, "first_run_completed", False):
            return wizard_config

        logger.info("Wizard auf bestehender Konfiguration: nur die abgefragten Bereiche werden ersetzt")
        for field in self._WIZARD_OWNED_PROVIDER_FIELDS:
            if hasattr(wizard_config.unified_config, field):
                setattr(
                    existing.unified_config, field,
                    getattr(wizard_config.unified_config, field),
                )
        return existing

    def accept(self):
        """Handle wizard completion - Claude Generated"""
        try:
            # Get LLM provider configuration from the LLM page
            provider_type = getattr(self.llm_page, 'provider_type', 'ollama')
            provider_name = getattr(self.llm_page, 'provider_name', 'ALIMA Provider')
            base_url = getattr(self.llm_page, 'base_url', 'http://localhost:11434')
            api_key = getattr(self.llm_page, 'api_key', '')
            models = getattr(self.llm_page, 'available_models', [])

            # Get pipeline/vision model selections from model page - Claude Generated
            pipeline_model = getattr(self.model_page, 'pipeline_model', '') if hasattr(self, 'model_page') else ''
            vision_model = getattr(self.model_page, 'vision_model', '') if hasattr(self, 'model_page') else ''

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

            # Build the config the wizard's answers describe…
            from ..utils.setup_utils import ConfigurationBuilder
            wizard_config = ConfigurationBuilder.create_initial_config(
                provider_type=provider_type,
                provider_name=provider_name,
                base_url=base_url,
                api_key=api_key,
                models=models,
                pipeline_model=pipeline_model,
                vision_model=vision_model,
            )

            # …and, on a machine that is already set up, apply it *onto* the
            # existing configuration instead of replacing it. create_initial_config
            # returns a fresh AlimaConfig, and save_config writes it wholesale — so
            # re-running the wizard used to wipe every plugin instance's settings
            # (finc endpoint, sru preset, the webindex fields, k10plus, dnb),
            # chat_config, ui_config, repetition_config, the Tier-2 plugin
            # approvals, the installed bundles and every additional LLM provider.
            # That was tolerable while --wizard was the only way in; it is not
            # once the wizard is reachable from the menu. - Claude Generated
            self.config = self._apply_onto_existing(wizard_config)

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

            # Save catalog SOAP configuration onto the catalog instance (WP P7:
            # was the CatalogConfig mirror, which save_config lifted into the
            # instances). - Claude Generated
            from ..core.search.factory import set_plugin_settings
            set_plugin_settings(self.config, "catalog", {
                "catalog_search_url": self.catalog_page.soap_search_url.text().strip(),
                "catalog_details": self.catalog_page.soap_details_url.text().strip(),
                "token": self.catalog_page.catalog_token.text().strip(),
                "catalog_web_search_url": self.catalog_page.web_search_url.text().strip(),
                "catalog_web_record_url": self.catalog_page.web_record_url.text().strip(),
            })

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

            # Save configuration (after database config is applied!)
            config_manager = ConfigManager()
            config_manager.save_config(self.config)

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
        self._preset_hint.setStyleSheet("color: gray; font-style: italic;")
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
    """Model selection for pipeline + vision - Claude Generated

    Used to ask six questions, one per classic-pipeline task. That per-task
    choice (``task_preferences``) only ever acted as a dead-on-arrival first
    entry of a fallback chain nothing else walked, and three of the six tasks
    had no step to land on at all (see ``docs/settings_cleanup_phase1.md``).
    Retired Oct 2026 to the two scopes that actually vary: the pipeline
    default (every classic step) and the vision default (image/OCR, a
    capability question rather than a pipeline step).
    """

    def __init__(self, presets=None):
        super().__init__()
        self.setTitle("Modell-Auswahl")
        self.setSubTitle("Welches Modell für die Pipeline, welches für Bilder/OCR?")

        self.pipeline_model: str = ""
        self.vision_model: str = ""
        self._presets = presets

        layout = QVBoxLayout()

        explanation = QLabel(
            "Das Pipeline-Modell läuft auf jedem Schritt der klassischen Pipeline "
            "(Initialisierung, Schlagworte, Klassifikation).\n"
            "Das Vision-Modell übernimmt Bild- und OCR-Analyse - das braucht "
            "Bildverständnis, nicht jedes Modell kann das.\n\n"
            "Oder nutzen Sie den Button unten um ein Modell für beides zu verwenden."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form_layout = QFormLayout()

        def _make_combo() -> QComboBox:
            combo = QComboBox()
            combo.setMinimumWidth(300)
            combo.setEditable(True)
            completer = QCompleter()
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            combo.setCompleter(completer)
            return combo

        self.pipeline_combo = _make_combo()
        self.pipeline_combo.setToolTip("Initialisation, Keywords, Classification, DK Classification")
        pipeline_label = QLabel("🔧 Pipeline-Modell:")
        pipeline_label.setToolTip(self.pipeline_combo.toolTip())
        form_layout.addRow(pipeline_label, self.pipeline_combo)

        self.vision_combo = _make_combo()
        self.vision_combo.setToolTip("Bild-/OCR-Analyse")
        vision_label = QLabel("👁️ Vision-Modell:")
        vision_label.setToolTip(self.vision_combo.toolTip())
        form_layout.addRow(vision_label, self.vision_combo)

        layout.addLayout(form_layout)

        self.use_same_btn = QPushButton("🔄 Gleiches Modell für beide verwenden")
        self.use_same_btn.clicked.connect(self._use_same_for_both)
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
            for combo in (self.pipeline_combo, self.vision_combo):
                combo.addItem("(Keine Modelle verfügbar)")
            return

        for combo in (self.pipeline_combo, self.vision_combo):
            combo.clear()
            combo.addItems(available_models)

        self.pipeline_combo.setCurrentIndex(0)

        # Prefer a vision-sounding model for the vision combo's default
        vision_models = [m for m in available_models
                         if 'vision' in m.lower() or 'llava' in m.lower()]
        if vision_models:
            self.vision_combo.setCurrentText(vision_models[0])
        else:
            self.vision_combo.setCurrentIndex(0)

        # Apply preset models after populating combos - Claude Generated
        if self._presets and self._presets.has_models():
            self._apply_presets(self._presets)

    def _get_default_models(self, provider_type: str) -> list:
        """Get default models for provider type - Claude Generated"""
        # Return empty list to avoid hardcoded models; will show placeholder if no models available
        return []

    def _apply_presets(self, presets) -> None:
        """Pre-select model combos from institution preset values - Claude Generated"""
        pipeline_model = presets.pipeline_model()
        if pipeline_model:
            idx = self.pipeline_combo.findText(pipeline_model)
            if idx >= 0:
                self.pipeline_combo.setCurrentIndex(idx)

        vision_model = presets.vision_model()
        if vision_model:
            idx = self.vision_combo.findText(vision_model)
            if idx >= 0:
                self.vision_combo.setCurrentIndex(idx)

    def _use_same_for_both(self):
        """Set the vision combo to the pipeline combo's selection - Claude Generated"""
        model = self.pipeline_combo.currentText()
        self.vision_combo.setCurrentText(model)

        QMessageBox.information(
            self,
            "Modelle eingestellt",
            f"✅ Beide verwenden jetzt: {model}"
        )

    def validatePage(self) -> bool:
        """Save selections - Claude Generated

        STRICT VALIDATION: Reject auto-select, use only explicit model names
        """
        from PyQt6.QtWidgets import QMessageBox

        self.pipeline_model = ""
        self.vision_model = ""

        for attr, combo, label in (
            ("pipeline_model", self.pipeline_combo, "Pipeline-Modell"),
            ("vision_model", self.vision_combo, "Vision-Modell"),
        ):
            model = combo.currentText()

            if not model:
                QMessageBox.warning(
                    self,
                    "Modell erforderlich",
                    f"Bitte wählen Sie ein {label} aus."
                )
                return False

            if "(Keine Modelle" in model or "(No models" in model:
                QMessageBox.warning(
                    self,
                    "Modell erforderlich",
                    f"Keine Modelle verfügbar für {label}. Bitte überprüfen Sie die LLM-Verbindung."
                )
                return False

            model_lower = model.lower()
            if model_lower in ["auto-select", "(auto-select)", "default", "auto"]:
                QMessageBox.warning(
                    self,
                    "Automatische Auswahl nicht erlaubt",
                    f"Bitte wählen Sie ein spezifisches {label},\n"
                    f"nicht automatische Auswahl (Auto-select)."
                )
                return False

            setattr(self, attr, model)

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
        if hasattr(wizard, 'model_page') and (wizard.model_page.pipeline_model or wizard.model_page.vision_model):
            model_summary = "\n🎯 Modell-Zuordnungen:\n"
            if wizard.model_page.pipeline_model:
                model_summary += f"   Pipeline: {wizard.model_page.pipeline_model}\n"
            if wizard.model_page.vision_model:
                model_summary += f"   Vision: {wizard.model_page.vision_model}\n"
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
