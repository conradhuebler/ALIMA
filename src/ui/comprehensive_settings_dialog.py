#!/usr/bin/env python3
"""
Comprehensive Settings Dialog for ALIMA
Combines database settings, LLM configuration, and system settings.
Claude Generated
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QTextEdit, QLabel, QGroupBox, QScrollArea,
    QMessageBox, QFileDialog, QProgressDialog, QGridLayout,
    QSplitter, QListWidget, QListWidgetItem, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView  # Claude Generated
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QIcon, QPalette
import json
import logging
import getpass
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy

from ..utils.config_manager import ConfigManager, AlimaConfig, DatabaseConfig, CatalogConfig, SystemConfig
from ..utils.config_models import UnifiedProvider
from .unified_provider_tab import UnifiedProviderTab
from ..utils.config_models import TaskPreference, TaskType


class DatabaseTestWorker(QThread):
    """Worker thread for database connection testing - Claude Generated"""
    test_completed = pyqtSignal(bool, str)
    
    def __init__(self, config_manager: ConfigManager):
        super().__init__()
        self.config_manager = config_manager
        
    def run(self):
        success, message = self.config_manager.test_database_connection()
        self.test_completed.emit(success, message)


class ComprehensiveSettingsDialog(QDialog):
    """Comprehensive settings dialog combining all ALIMA configurations - Claude Generated"""

    config_changed = pyqtSignal()
    task_preferences_changed = pyqtSignal()  # Forward task preference changes - Claude Generated
    
    def __init__(self, alima_manager=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        self.alima_manager = alima_manager  # For ProviderStatusService access - Claude Generated

        # Implement Unit of Work pattern - Claude Generated (Refactoring)
        self.original_config = self.config_manager.load_config()
        self.config_to_edit = deepcopy(self.original_config)

        self.setWindowTitle("ALIMA Settings")
        self.setModal(True)
        self.resize(900, 700)

        self._setup_ui()
        self._load_current_settings()
    
    def _setup_ui(self):
        """Setup the user interface - Claude Generated"""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.database_tab = self._create_database_tab()
        self.unified_provider_tab = UnifiedProviderTab(
            self.config_to_edit.unified_config,
            self.config_to_edit,
            self.config_manager,  # ✅ Pass ConfigManager for persistence operations
            self.alima_manager,
            self
        )
        self.catalog_tab = self._create_catalog_tab()
        self.system_tab = self._create_system_tab()
        self.about_tab = self._create_about_tab()
        
        # Connect unified provider tab signals
        self.unified_provider_tab.config_changed.connect(self.config_changed)
        self.unified_provider_tab.task_preferences_changed.connect(self.task_preferences_changed)  # Forward task preference changes - Claude Generated
        
        # Add tabs
        self.tab_widget.addTab(self.database_tab, "🗄️ Database")
        self.tab_widget.addTab(self.unified_provider_tab, "🚀 Providers & Models")  # Claude Generated - Unified Tab
        self.tab_widget.addTab(self.catalog_tab, "📚 Catalog")
        
        # Task Preferences are now integrated into the unified provider tab
        
        self.tab_widget.addTab(self.system_tab, "⚙️ System")
        self.tab_widget.addTab(self.about_tab, "ℹ️ About")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()

        button_layout.addStretch()

        self.export_preset_button = QPushButton("📤 Als Preset exportieren")
        self.export_preset_button.setToolTip("Aktuelle Einstellungen als alima_presets.json exportieren")
        self.export_preset_button.clicked.connect(self._export_as_preset)

        self.save_button = QPushButton("💾 Save & Close")
        self.save_button.clicked.connect(self._save_and_close)

        self.cancel_button = QPushButton("❌ Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.export_preset_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _create_database_tab(self) -> QWidget:
        """Create database configuration tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Database type selection
        type_group = QGroupBox("Database Type")
        type_layout = QFormLayout()
        
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["sqlite", "mysql", "mariadb"])
        self.db_type_combo.currentTextChanged.connect(self._on_db_type_changed)
        type_layout.addRow("Database Type:", self.db_type_combo)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # SQLite configuration
        self.sqlite_group = QGroupBox("SQLite Configuration")
        sqlite_layout = QFormLayout()
        
        self.sqlite_path = QLineEdit()
        sqlite_browse_layout = QHBoxLayout()
        sqlite_browse_layout.addWidget(self.sqlite_path)
        sqlite_browse_button = QPushButton("Browse...")
        sqlite_browse_button.clicked.connect(self._browse_sqlite_path)
        sqlite_browse_layout.addWidget(sqlite_browse_button)
        
        sqlite_layout.addRow("Database Path:", sqlite_browse_layout)
        
        self.sqlite_group.setLayout(sqlite_layout)
        layout.addWidget(self.sqlite_group)
        
        # MySQL configuration
        self.mysql_group = QGroupBox("MySQL/MariaDB Configuration")
        mysql_layout = QFormLayout()
        
        self.mysql_host = QLineEdit()
        mysql_layout.addRow("Host:", self.mysql_host)
        
        self.mysql_port = QSpinBox()
        self.mysql_port.setRange(1, 65535)
        self.mysql_port.setValue(3306)
        mysql_layout.addRow("Port:", self.mysql_port)
        
        self.mysql_database = QLineEdit()
        mysql_layout.addRow("Database:", self.mysql_database)
        
        self.mysql_username = QLineEdit()
        mysql_layout.addRow("Username:", self.mysql_username)
        
        self.mysql_password = QLineEdit()
        self.mysql_password.setEchoMode(QLineEdit.EchoMode.Password)
        mysql_layout.addRow("Password:", self.mysql_password)
        
        self.mysql_charset = QComboBox()
        self.mysql_charset.addItems(["utf8mb4", "utf8", "latin1"])
        mysql_layout.addRow("Charset:", self.mysql_charset)
        
        self.mysql_ssl_disabled = QCheckBox("Disable SSL")
        mysql_layout.addRow("SSL Settings:", self.mysql_ssl_disabled)
        
        self.mysql_group.setLayout(mysql_layout)
        layout.addWidget(self.mysql_group)
        
        # Connection settings
        conn_group = QGroupBox("Connection Settings")
        conn_layout = QFormLayout()
        
        self.connection_timeout = QSpinBox()
        self.connection_timeout.setRange(5, 300)
        self.connection_timeout.setSuffix(" seconds")
        conn_layout.addRow("Connection Timeout:", self.connection_timeout)
        
        self.auto_create_tables = QCheckBox("Automatically create database tables")
        self.auto_create_tables.setChecked(True)
        conn_layout.addRow("Auto-Setup:", self.auto_create_tables)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # Database-specific buttons
        db_button_layout = QHBoxLayout()

        self.db_test_button = QPushButton("🔧 Test Connection")
        self.db_test_button.clicked.connect(self._test_database_connection)

        self.db_reset_button = QPushButton("↺ Reset Database Settings")
        self.db_reset_button.clicked.connect(self._reset_database_to_defaults)

        db_button_layout.addWidget(self.db_test_button)
        db_button_layout.addWidget(self.db_reset_button)
        db_button_layout.addStretch()

        layout.addLayout(db_button_layout)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    
    def _create_catalog_tab(self) -> QWidget:
        """Create catalog configuration tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Catalog Type Selection
        type_group = QGroupBox("Catalog Type")
        type_layout = QFormLayout()
        
        self.catalog_type_combo = QComboBox()
        self.catalog_type_combo.addItems(["libero_soap", "marcxml_sru", "auto"])
        self.catalog_type_combo.setToolTip(
            "libero_soap: Original Libero SOAP API (requires token)\n"
            "marcxml_sru: Standard MARC XML via SRU protocol (DNB, K10plus, etc.)\n"
            "auto: Automatically detect based on configuration"
        )
        self.catalog_type_combo.currentTextChanged.connect(self._on_catalog_type_changed)
        type_layout.addRow("Catalog Type:", self.catalog_type_combo)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Libero SOAP settings (original)
        self.libero_group = QGroupBox("Libero SOAP Configuration")
        libero_layout = QFormLayout()
        
        self.catalog_token = QLineEdit()
        self.catalog_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.catalog_token.setToolTip("API token for Libero SOAP catalog access")
        libero_layout.addRow("Catalog Token:", self.catalog_token)

        self.libero_token_btn = QPushButton("🔑 Token erstellen...")
        self.libero_token_btn.clicked.connect(self._fetch_libero_token_dialog)
        libero_layout.addRow("", self.libero_token_btn)

        self.catalog_search_url = QLineEdit()
        self.catalog_search_url.setToolTip("SOAP search endpoint URL (e.g., https://libero.ub.example.de/libero/LiberoWebServices.CatalogueSearcher.cls)")
        libero_layout.addRow("SOAP Search URL:", self.catalog_search_url)

        self.catalog_details_url = QLineEdit()
        self.catalog_details_url.setToolTip("SOAP details endpoint URL (e.g., https://libero.ub.example.de/libero/LiberoWebServices.LibraryAPI.cls)")
        libero_layout.addRow("SOAP Details URL:", self.catalog_details_url)

        self.catalog_web_search_url = QLineEdit()
        self.catalog_web_search_url.setToolTip("Web frontend search URL for web-scraping fallback (e.g., https://katalog.ub.example.de/Search/Results). Leave empty to disable web fallback.")
        libero_layout.addRow("Web Search URL:", self.catalog_web_search_url)

        self.catalog_web_record_url = QLineEdit()
        self.catalog_web_record_url.setToolTip("Web frontend record base URL for web-scraping fallback (e.g., https://katalog.ub.example.de/Record/). Leave empty to disable web fallback.")
        libero_layout.addRow("Web Record URL:", self.catalog_web_record_url)

        self.libero_group.setLayout(libero_layout)
        layout.addWidget(self.libero_group)
        
        # MARC XML / SRU settings
        self.sru_group = QGroupBox("MARC XML / SRU Configuration")
        sru_layout = QFormLayout()
        
        # SRU Preset selector
        self.sru_preset_combo = QComboBox()
        self.sru_preset_combo.addItems(["", "dnb", "loc", "gbv", "swb", "k10plus"])
        self.sru_preset_combo.setToolTip(
            "Select a preset catalog or leave empty for custom URL:\n"
            "• dnb: Deutsche Nationalbibliothek\n"
            "• loc: Library of Congress\n"
            "• gbv: GBV Gemeinsamer Bibliotheksverbund\n"
            "• swb: SWB Südwestdeutscher Bibliotheksverbund\n"
            "• k10plus: K10plus (GBV + SWB combined)"
        )
        self.sru_preset_combo.currentTextChanged.connect(self._on_sru_preset_changed)
        sru_layout.addRow("SRU Preset:", self.sru_preset_combo)
        
        # Custom SRU URL (for custom endpoints)
        self.sru_base_url = QLineEdit()
        self.sru_base_url.setPlaceholderText("e.g., https://services.dnb.de/sru/dnb")
        self.sru_base_url.setToolTip("SRU endpoint URL (only needed if not using a preset)")
        sru_layout.addRow("SRU Base URL:", self.sru_base_url)
        
        self.sru_database = QLineEdit()
        self.sru_database.setPlaceholderText("e.g., dnb")
        self.sru_database.setToolTip("SRU database name (optional, depends on endpoint)")
        sru_layout.addRow("SRU Database:", self.sru_database)
        
        self.sru_schema = QComboBox()
        self.sru_schema.addItems(["marcxml", "MARC21-xml"])
        self.sru_schema.setToolTip("Record schema format")
        sru_layout.addRow("Record Schema:", self.sru_schema)
        
        self.sru_max_records = QSpinBox()
        self.sru_max_records.setRange(1, 500)
        self.sru_max_records.setValue(50)
        self.sru_max_records.setToolTip("Maximum records per search (default: 50)")
        sru_layout.addRow("Max Records:", self.sru_max_records)
        
        self.sru_group.setLayout(sru_layout)
        layout.addWidget(self.sru_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()
        
        self.strict_gnd_validation = QCheckBox()
        self.strict_gnd_validation.setChecked(True)
        self.strict_gnd_validation.setToolTip(
            "When enabled, only GND-validated keywords are used in DK search (recommended).\n"
            "When disabled, plain text keywords are included if GND validation fails."
        )
        advanced_layout.addRow("Strict GND Validation:", self.strict_gnd_validation)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        
        # Initialize visibility based on default catalog type
        self._on_catalog_type_changed(self.catalog_type_combo.currentText())
        
        return widget
    
    def _fetch_libero_token_dialog(self):
        """Open Libero login dialog and write token into token field - Claude Generated"""
        from PyQt6.QtWidgets import QMessageBox, QDialog
        url = self.catalog_search_url.text().strip()
        if not url:
            QMessageBox.warning(self, "URL fehlt",
                                "Bitte zuerst eine Search URL eintragen.")
            return
        from .libero_login_dialog import LiberoLoginDialog
        dialog = LiberoLoginDialog(soap_url=url, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.catalog_token.setText(dialog.token)

    def _on_catalog_type_changed(self, catalog_type: str):
        """Handle catalog type selection changes - Claude Generated"""
        is_libero = catalog_type == "libero_soap"
        is_sru = catalog_type == "marcxml_sru"
        is_auto = catalog_type == "auto"
        
        # Show/hide relevant groups
        self.libero_group.setVisible(is_libero or is_auto)
        self.sru_group.setVisible(is_sru or is_auto)
    
    def _on_sru_preset_changed(self, preset: str):
        """Handle SRU preset selection - disable custom URL fields when preset is selected - Claude Generated"""
        use_preset = bool(preset)
        self.sru_base_url.setEnabled(not use_preset)
        self.sru_database.setEnabled(not use_preset)
        if use_preset:
            self.sru_base_url.setPlaceholderText(f"Using preset: {preset}")
            self.sru_database.setPlaceholderText(f"Using preset: {preset}")
        else:
            self.sru_base_url.setPlaceholderText("e.g., https://services.dnb.de/sru/dnb")
            self.sru_database.setPlaceholderText("e.g., dnb")
    
    def _create_system_tab(self) -> QWidget:
        """Create system configuration tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # System settings
        system_group = QGroupBox("System Settings")
        system_layout = QFormLayout()
        
        self.debug_mode = QCheckBox("Enable debug mode")
        system_layout.addRow("Debug:", self.debug_mode)

        # Webcam input option - Claude Generated
        self.enable_webcam_input = QCheckBox("📷 Enable webcam capture in Pipeline tab")
        self.enable_webcam_input.setToolTip("Enable/disable the webcam button for capturing images directly from camera")
        # Note: Saved on dialog close via _get_config_from_ui() - Claude Generated
        system_layout.addRow("Webcam:", self.enable_webcam_input)

        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        system_layout.addRow("Log Level:", self.log_level)
        
        # Cache Directory - Claude Generated
        cache_layout = QHBoxLayout()
        self.cache_dir = QLineEdit()
        cache_layout.addWidget(self.cache_dir)
        cache_browse = QPushButton("Browse...")
        cache_browse.clicked.connect(self._browse_cache_dir)
        cache_layout.addWidget(cache_browse)
        system_layout.addRow("Cache Directory:", cache_layout)

        # Data Directory - Claude Generated
        data_layout = QHBoxLayout()
        self.data_dir = QLineEdit()
        data_layout.addWidget(self.data_dir)
        data_browse = QPushButton("Browse...")
        data_browse.clicked.connect(self._browse_data_dir)
        data_layout.addWidget(data_browse)
        system_layout.addRow("Data Directory:", data_layout)

        # Temp Directory - Claude Generated
        temp_layout = QHBoxLayout()
        self.temp_dir = QLineEdit()
        temp_layout.addWidget(self.temp_dir)
        temp_browse = QPushButton("Browse...")
        temp_browse.clicked.connect(self._browse_temp_dir)
        temp_layout.addWidget(temp_browse)
        system_layout.addRow("Temp Directory:", temp_layout)

        # Autosave / batch output directory - Claude Generated
        autosave_layout = QHBoxLayout()
        self.autosave_dir = QLineEdit()
        self.autosave_dir.setToolTip("Verzeichnis für automatisch gespeicherte Pipeline-Ergebnisse und Batch-Output")
        autosave_layout.addWidget(self.autosave_dir)
        autosave_browse = QPushButton("Browse...")
        autosave_browse.clicked.connect(self._browse_autosave_dir)
        autosave_layout.addWidget(autosave_browse)
        system_layout.addRow("Autosave-Verzeichnis:", autosave_layout)

        system_group.setLayout(system_layout)
        layout.addWidget(system_group)

        # DOI resolver settings - Claude Generated
        doi_group = QGroupBox("🔗 DOI-Auflösung")
        doi_layout = QFormLayout()

        self.contact_email = QLineEdit()
        self.contact_email.setPlaceholderText("z.B. name@institution.de")
        self.contact_email.setToolTip(
            "E-Mail für API-Polite-Pools (Crossref, OpenAlex). Ermöglicht höhere Rate Limits."
        )
        doi_layout.addRow("Kontakt-E-Mail:", self.contact_email)

        self.doi_use_crossref = QCheckBox("Crossref (empfohlen – breite Zeitschriften-Abdeckung)")
        self.doi_use_crossref.setChecked(True)
        doi_layout.addRow("", self.doi_use_crossref)

        self.doi_use_openalex = QCheckBox("OpenAlex (Fallback – gute Abstract-Abdeckung)")
        self.doi_use_openalex.setChecked(True)
        doi_layout.addRow("", self.doi_use_openalex)

        self.doi_use_datacite = QCheckBox("DataCite (Fallback – Datensätze, Berichte, nicht-Zeitschriften)")
        self.doi_use_datacite.setChecked(True)
        doi_layout.addRow("", self.doi_use_datacite)

        doi_group.setLayout(doi_layout)
        layout.addWidget(doi_group)

        # Repetition Detection settings - Claude Generated
        repetition_group = QGroupBox("🔄 Repetition Detection")
        repetition_layout = QFormLayout()

        self.repetition_enabled = QCheckBox("Enable repetition detection")
        self.repetition_enabled.setToolTip("Detect when LLM falls into repetitive output loops")
        repetition_layout.addRow("Detection:", self.repetition_enabled)

        self.repetition_auto_abort = QCheckBox("Auto-abort on detection")
        self.repetition_auto_abort.setToolTip("Automatically stop generation when repetition is detected")
        repetition_layout.addRow("Auto-Abort:", self.repetition_auto_abort)

        self.repetition_ngram_threshold = QSpinBox()
        self.repetition_ngram_threshold.setRange(3, 20)
        self.repetition_ngram_threshold.setToolTip("Number of phrase repetitions before triggering (higher = more lenient)")
        repetition_layout.addRow("N-gram Threshold:", self.repetition_ngram_threshold)

        self.repetition_min_text = QSpinBox()
        self.repetition_min_text.setRange(100, 5000)
        self.repetition_min_text.setSingleStep(100)
        self.repetition_min_text.setSuffix(" chars")
        self.repetition_min_text.setToolTip("Minimum text length before checking starts")
        repetition_layout.addRow("Min Text Length:", self.repetition_min_text)

        self.repetition_char_threshold = QSpinBox()
        self.repetition_char_threshold.setRange(20, 200)
        self.repetition_char_threshold.setToolTip("Consecutive identical characters before triggering (e.g., '!!!')")
        repetition_layout.addRow("Char Repeat Threshold:", self.repetition_char_threshold)

        # Grace period setting - Claude Generated (2026-02-17)
        self.repetition_grace_period = QDoubleSpinBox()
        self.repetition_grace_period.setRange(0.0, 10.0)
        self.repetition_grace_period.setSingleStep(0.5)
        self.repetition_grace_period.setValue(2.0)
        self.repetition_grace_period.setDecimals(1)
        self.repetition_grace_period.setSuffix(" s")
        self.repetition_grace_period.setToolTip(
            "Wartezeit vor Auto-Abbruch bei Wiederholungen\n"
            "0 = sofortiger Abbruch (alte Verhalten)\n"
            "2-5 = empfohlen für False-Positive-Vermeidung"
        )
        repetition_layout.addRow("Grace Period:", self.repetition_grace_period)

        repetition_group.setLayout(repetition_layout)
        layout.addWidget(repetition_group)

        # Configuration scope
        scope_group = QGroupBox("Save Configuration To")
        scope_layout = QVBoxLayout()
        
        self.scope_project = QCheckBox("Project (./alima_config.json)")
        self.scope_user = QCheckBox("User (OS-specific user directory)")
        self.scope_system = QCheckBox("System (OS-specific system directory)")
        
        self.scope_user.setChecked(True)  # Default to user scope
        
        scope_layout.addWidget(self.scope_project)
        scope_layout.addWidget(self.scope_user)
        scope_layout.addWidget(self.scope_system)
        
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group)
        
        # Configuration paths info
        paths_group = QGroupBox("Configuration Paths")
        paths_layout = QVBoxLayout()
        
        config_info = self.config_manager.get_config_info()
        paths_text = f"""
<b>Operating System:</b> {config_info['os']}<br>
<b>Project:</b> {config_info['project_config']}<br>
<b>User:</b> {config_info['user_config']}<br>
<b>System:</b> {config_info['system_config']}<br>
        """.strip()
        
        paths_label = QLabel(paths_text)
        paths_label.setWordWrap(True)
        paths_layout.addWidget(paths_label)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_about_tab(self) -> QWidget:
        """Create about tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # About info
        about_text = """
        <h2>ALIMA - AI-Powered Literature Analysis</h2>
        
        <h3>Configuration Management</h3>
        <p>This settings dialog manages all ALIMA configuration including:</p>
        <ul>
        <li><b>Database:</b> SQLite and MySQL/MariaDB support</li>
        <li><b>LLM Providers:</b> API keys for Gemini, Claude, OpenAI, and others</li>
        <li><b>Library Catalog:</b> Integration with library catalog systems</li>
        <li><b>Prompts:</b> Customizable AI prompts for different analysis tasks</li>
        <li><b>System:</b> Debug settings and directory configurations</li>
        </ul>
        
        <h3>Cross-Platform Support</h3>
        <p>Configuration files are stored in OS-appropriate locations:</p>
        <ul>
        <li><b>Linux:</b> ~/.config/alima/ (XDG specification)</li>
        <li><b>macOS:</b> ~/Library/Application Support/ALIMA/</li>
        <li><b>Windows:</b> %APPDATA%\\ALIMA\\</li>
        </ul>
        
        <h3>Priority System</h3>
        <p>Configuration sources in order of priority:</p>
        <ol>
        <li><b>Project:</b> ./alima_config.json (highest)</li>
        <li><b>User:</b> OS-specific user directory</li>
        <li><b>System:</b> OS-specific system directory</li>
        <li><b>Legacy:</b> ~/.alima_config.json (lowest)</li>
        </ol>
        
        <p><i>Generated by ALIMA Configuration Manager</i></p>
        """
        
        about_label = QLabel(about_text)
        about_label.setWordWrap(True)
        about_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(about_label)
        scroll_area.setWidgetResizable(True)
        
        layout.addWidget(scroll_area)
        widget.setLayout(layout)
        return widget

    def _load_current_settings(self):
        """Load current settings into UI elements - Claude Generated"""
        config = self.config_to_edit
        
        # Database settings
        self.db_type_combo.setCurrentText(config.database.db_type)
        self.sqlite_path.setText(config.database.sqlite_path)
        self.mysql_host.setText(config.database.host)
        self.mysql_port.setValue(config.database.port)
        self.mysql_database.setText(config.database.database)
        self.mysql_username.setText(config.database.username)
        self.mysql_password.setText(config.database.password)
        self.mysql_charset.setCurrentText(config.database.charset)
        self.mysql_ssl_disabled.setChecked(config.database.ssl_disabled)
        self.connection_timeout.setValue(config.database.connection_timeout)
        self.auto_create_tables.setChecked(config.database.auto_create_tables)
        
        # LLM settings are now handled by the unified provider tab
        # Note: Static provider API keys (Gemini, Anthropic) are managed through the unified provider system
        
        # Provider lists are now managed by the unified provider tab
        # Dynamic provider population is handled automatically by the UnifiedProviderTab
        
        # Catalog settings
        self.catalog_type_combo.setCurrentText(config.catalog.catalog_type)
        self.catalog_token.setText(config.catalog.catalog_token)
        self.catalog_search_url.setText(config.catalog.catalog_search_url)
        self.catalog_details_url.setText(config.catalog.catalog_details_url)
        self.catalog_web_search_url.setText(getattr(config.catalog, 'catalog_web_search_url', ''))
        self.catalog_web_record_url.setText(getattr(config.catalog, 'catalog_web_record_url', ''))
        
        # SRU settings
        self.sru_preset_combo.setCurrentText(config.catalog.sru_preset)
        self.sru_base_url.setText(config.catalog.sru_base_url)
        self.sru_database.setText(config.catalog.sru_database)
        self.sru_schema.setCurrentText(config.catalog.sru_schema)
        self.sru_max_records.setValue(config.catalog.sru_max_records)
        self.strict_gnd_validation.setChecked(config.catalog.strict_gnd_validation_for_dk_search)
        
        # Trigger visibility update
        self._on_catalog_type_changed(config.catalog.catalog_type)
        self._on_sru_preset_changed(config.catalog.sru_preset)
        
        # System settings
        self.debug_mode.setChecked(config.system_config.debug)
        self.log_level.setCurrentText(config.system_config.log_level)
        self.cache_dir.setText(config.system_config.cache_dir)
        self.data_dir.setText(config.system_config.data_dir)
        self.temp_dir.setText(config.system_config.temp_dir)
        self.autosave_dir.setText(config.system_config.autosave_dir)

        # DOI resolver settings - Claude Generated
        self.contact_email.setText(getattr(config.system_config, 'contact_email', ''))
        self.doi_use_crossref.setChecked(getattr(config.system_config, 'doi_use_crossref', True))
        self.doi_use_openalex.setChecked(getattr(config.system_config, 'doi_use_openalex', True))
        self.doi_use_datacite.setChecked(getattr(config.system_config, 'doi_use_datacite', True))

        # UI settings - Claude Generated
        self.enable_webcam_input.setChecked(config.ui_config.enable_webcam_input)

        # Repetition Detection settings - Claude Generated
        rep_config = getattr(config, 'repetition_config', None)
        if rep_config:
            self.repetition_enabled.setChecked(rep_config.enabled)
            self.repetition_auto_abort.setChecked(rep_config.auto_abort)
            self.repetition_ngram_threshold.setValue(rep_config.ngram_threshold)
            self.repetition_min_text.setValue(rep_config.min_text_length)
            self.repetition_char_threshold.setValue(rep_config.char_repeat_threshold)
            self.repetition_grace_period.setValue(getattr(rep_config, 'grace_period_seconds', 2.0))  # Claude Generated (2026-02-17)
        else:
            # Use defaults
            self.repetition_enabled.setChecked(True)
            self.repetition_auto_abort.setChecked(True)
            self.repetition_ngram_threshold.setValue(8)
            self.repetition_min_text.setValue(1000)
            self.repetition_char_threshold.setValue(80)
            self.repetition_grace_period.setValue(2.0)  # Claude Generated (2026-02-17)

        # Update UI based on database type
        self._on_db_type_changed(config.database.db_type)

    def _refresh_all_providers_status(self):
        """Refresh reachability status for all providers - Claude Generated"""
        try:
            # Show progress dialog
            progress = QProgressDialog("Refreshing provider status...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # Create temporary LLM service to test providers
            from ..llm.llm_service import LlmService
            temp_service = LlmService(lazy_initialization=True)
            temp_service.config_manager.config = self.config_to_edit
            
            # Refresh all provider status
            status_results = temp_service.refresh_all_provider_status()
            
            progress.close()
            
            # Show results
            result_text = "Provider Status Check Results:\\n\\n"
            for provider_name, is_reachable in status_results.items():
                status_icon = "✅" if is_reachable else "❌"
                status_text = "Reachable" if is_reachable else "Unreachable"
                
                # Get detailed status info
                status_info = temp_service.get_provider_status(provider_name)
                latency_info = f" ({status_info.get('latency_ms', 0):.1f}ms)" if is_reachable else ""
                
                result_text += f"{status_icon} {provider_name}: {status_text}{latency_info}\\n"
                
                if not is_reachable and status_info.get('error'):
                    result_text += f"   Error: {status_info['error']}\\n"
            
            QMessageBox.information(
                self,
                "🔄 Provider Status Refresh Complete",
                result_text
            )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(
                self,
                "❌ Status Refresh Failed",
                f"Failed to refresh provider status:\\n\\n{str(e)}"
            )

    def _on_db_type_changed(self, db_type: str):
        """Handle database type change - Claude Generated"""
        if db_type == "sqlite":
            self.sqlite_group.setVisible(True)
            self.mysql_group.setVisible(False)
        else:
            self.sqlite_group.setVisible(False)
            self.mysql_group.setVisible(True)

    def _browse_autosave_dir(self):
        """Browse for autosave directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Autosave-Verzeichnis wählen", self.autosave_dir.text()
        )
        if directory:
            self.autosave_dir.setText(directory)

    def _browse_cache_dir(self):
        """Browse for cache directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Cache-Verzeichnis wählen", self.cache_dir.text()
        )
        if directory:
            self.cache_dir.setText(directory)

    def _browse_data_dir(self):
        """Browse for data directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Daten-Verzeichnis wählen", self.data_dir.text()
        )
        if directory:
            self.data_dir.setText(directory)

    def _browse_temp_dir(self):
        """Browse for temp directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Temp-Verzeichnis wählen", self.temp_dir.text()
        )
        if directory:
            self.temp_dir.setText(directory)

    def _browse_sqlite_path(self):
        """Browse for SQLite database path - Claude Generated"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Select SQLite Database", 
            self.sqlite_path.text(),
            "SQLite Database (*.db *.sqlite *.sqlite3);;All Files (*)"
        )
        if file_path:
            self.sqlite_path.setText(file_path)

    def _test_database_connection(self):
        """Test database connection - Claude Generated"""
        # Update config with current settings
        config = self._get_config_from_ui()

        # Temporarily update config manager
        self.config_manager._config = config

        # Create and start test worker
        self.db_test_button.setEnabled(False)
        self.db_test_button.setText("Testing...")

        self.db_test_worker = DatabaseTestWorker(self.config_manager)
        self.db_test_worker.test_completed.connect(self._on_db_test_completed)
        self.db_test_worker.start()
    
    @pyqtSlot(bool, str)
    def _on_db_test_completed(self, success: bool, message: str):
        """Handle database test completion - Claude Generated"""
        self.db_test_button.setEnabled(True)
        self.db_test_button.setText("🔧 Test Connection")

        if success:
            QMessageBox.information(self, "Database Test", f"✅ {message}")
        else:
            QMessageBox.warning(self, "Database Test", f"❌ {message}")

    def _reset_database_to_defaults(self):
        """Reset only database settings to defaults - Claude Generated"""
        reply = QMessageBox.question(
            self,
            "Reset Database Settings",
            "Are you sure you want to reset database settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Reset only database-related UI elements to defaults
            self.db_type_combo.setCurrentText("sqlite")
            self.sqlite_path.setText("alima_knowledge.db")
            self.mysql_host.setText("localhost")
            self.mysql_port.setValue(3306)
            self.mysql_database.setText("")
            self.mysql_username.setText("")
            self.mysql_password.setText("")
            self.mysql_charset.setText("utf8mb4")
            self.mysql_ssl_disabled.setChecked(False)
            self.connection_timeout.setValue(30)
            self.auto_create_tables.setChecked(True)

            # Trigger UI updates
            self._on_db_type_changed("sqlite")

            QMessageBox.information(self, "Reset Complete", "Database settings have been reset to defaults.")

    def _reset_to_defaults(self):
        """Reset all settings to defaults - Claude Generated"""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.config_to_edit = AlimaConfig()  # Create default config
            self._load_current_settings()
    
    def _get_config_from_ui(self) -> AlimaConfig:
        """Extract configuration from UI elements - Claude Generated (Refactoring)"""
        # Use the Unit of Work copy and update it with UI values
        config = self.config_to_edit
        
        # Database configuration - Claude Generated fix for expanded DatabaseConfig
        config.database_config = DatabaseConfig(
            db_type=self.db_type_combo.currentText(),
            sqlite_path=self.sqlite_path.text(),
            host=self.mysql_host.text(),
            port=self.mysql_port.value(),
            database=self.mysql_database.text(),
            username=self.mysql_username.text(),
            password=self.mysql_password.text(),
            charset=self.mysql_charset.currentText(),
            ssl_disabled=self.mysql_ssl_disabled.isChecked(),
            connection_timeout=self.connection_timeout.value(),
            auto_create_tables=self.auto_create_tables.isChecked()
        )
        
        # Unified provider configuration - use current configuration as-is
        # Provider settings are managed through the unified provider system
        # No need to reconstruct LLMConfig - unified_config is already properly managed
        
        # 🔍 DEBUG: Log unified provider preferred models - Claude Generated
        for provider in config.unified_config.providers:
            self.logger.critical(f"🔍 GET_CONFIG_FROM_UI_UNIFIED: {provider.name} ({provider.provider_type}).preferred_model='{provider.preferred_model}'")
        
        # Catalog configuration - Claude Generated fix for expanded config structure
        config.catalog_config = CatalogConfig(
            catalog_type=self.catalog_type_combo.currentText(),
            catalog_token=self.catalog_token.text(),
            catalog_search_url=self.catalog_search_url.text(),
            catalog_details_url=self.catalog_details_url.text(),
            catalog_web_search_url=self.catalog_web_search_url.text(),
            catalog_web_record_url=self.catalog_web_record_url.text(),
            sru_base_url=self.sru_base_url.text(),
            sru_database=self.sru_database.text(),
            sru_schema=self.sru_schema.currentText(),
            sru_preset=self.sru_preset_combo.currentText(),
            sru_max_records=self.sru_max_records.value(),
            strict_gnd_validation_for_dk_search=self.strict_gnd_validation.isChecked()
        )

        # System configuration - Claude Generated fix for expanded config structure
        config.system_config = SystemConfig(
            debug=self.debug_mode.isChecked(),
            log_level=self.log_level.currentText(),
            cache_dir=self.cache_dir.text(),
            data_dir=self.data_dir.text(),
            temp_dir=self.temp_dir.text(),
            autosave_dir=self.autosave_dir.text(),
            contact_email=self.contact_email.text().strip(),
            doi_use_crossref=self.doi_use_crossref.isChecked(),
            doi_use_openalex=self.doi_use_openalex.isChecked(),
            doi_use_datacite=self.doi_use_datacite.isChecked(),
            # Preserve wizard/system flags that have no UI controls - Claude Generated
            prompts_path=config.system_config.prompts_path,
            first_run_completed=config.system_config.first_run_completed,
            skip_first_run_check=config.system_config.skip_first_run_check
        )

        # UI configuration - Claude Generated (Webcam Feature)
        from ..utils.config_models import UIConfig, RepetitionDetectionConfig
        config.ui_config = UIConfig(
            enable_webcam_input=self.enable_webcam_input.isChecked()
        )

        # Repetition Detection configuration - Claude Generated
        config.repetition_config = RepetitionDetectionConfig(
            enabled=self.repetition_enabled.isChecked(),
            auto_abort=self.repetition_auto_abort.isChecked(),
            ngram_threshold=self.repetition_ngram_threshold.value(),
            min_text_length=self.repetition_min_text.value(),
            char_repeat_threshold=self.repetition_char_threshold.value(),
            grace_period_seconds=self.repetition_grace_period.value(),  # Claude Generated (2026-02-17)
            # Keep other values at defaults or from existing config
            ngram_size=getattr(config.repetition_config, 'ngram_size', 6) if hasattr(config, 'repetition_config') else 6,
            window_size=getattr(config.repetition_config, 'window_size', 300) if hasattr(config, 'repetition_config') else 300,
            window_similarity_threshold=getattr(config.repetition_config, 'window_similarity_threshold', 0.90) if hasattr(config, 'repetition_config') else 0.90,
            min_windows=getattr(config.repetition_config, 'min_windows', 4) if hasattr(config, 'repetition_config') else 4,
            check_interval=getattr(config.repetition_config, 'check_interval', 200) if hasattr(config, 'repetition_config') else 200,
            show_suggestions=getattr(config.repetition_config, 'show_suggestions', True) if hasattr(config, 'repetition_config') else True,
        )

        # Task preferences are already up-to-date in config_to_edit from UnifiedProviderTab - Claude Generated (Refactoring)

        return config
    
    def _export_as_preset(self):
        """Export current settings as alima_presets.json - Claude Generated"""
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        import json

        # Sync UnifiedProviderTab UI state → unified_config before reading - Claude Generated
        # The visible preferred_provider_combo lives in unified_provider_tab, not in this dialog.
        # _update_config_from_ui() writes that combo's value to unified_config.preferred_provider.
        self.unified_provider_tab._update_config_from_ui()

        # Ask for institution name
        institution_name, ok = QInputDialog.getText(
            self, "Preset exportieren", "Institutions-Name:",
            text=self.config_to_edit.unified_config.institution_name
                 if hasattr(self.config_to_edit.unified_config, 'institution_name') else ""
        )
        if not ok:
            return

        # Find the preferred/default provider - Claude Generated
        providers = getattr(self.config_to_edit.unified_config, 'providers', [])
        preferred_name = getattr(self.config_to_edit.unified_config, 'preferred_provider', '')
        provider = next(
            (p for p in providers if p.name == preferred_name and getattr(p, 'enabled', True)),
            None
        ) or next((p for p in providers if getattr(p, 'enabled', True)), None)

        llm_block: dict = {}
        if provider:
            llm_block["provider_type"] = provider.provider_type
            if provider.base_url:
                llm_block["base_url"] = provider.base_url
            if provider.api_key:
                llm_block["api_key"] = provider.api_key
            if getattr(provider, 'preferred_model', ''):
                llm_block["default_model"] = provider.preferred_model

        # Build task_models from task_preferences
        task_preferences = getattr(self.config_to_edit.unified_config, 'task_preferences', {})
        task_models: dict = {}
        for task_key, pref in task_preferences.items():
            priority = getattr(pref, 'model_priority', [])
            if priority:
                first = priority[0]
                model_name = first.get('model_name', '') if isinstance(first, dict) else ''
                if model_name:
                    task_models[task_key] = model_name
        if task_models:
            llm_block["task_models"] = task_models

        # Build catalog block
        cat_cfg = getattr(self.config_to_edit, 'catalog_config', None)
        catalog_block: dict = {}
        if cat_cfg:
            if getattr(cat_cfg, 'catalog_search_url', ''):
                catalog_block["soap_search_url"] = cat_cfg.catalog_search_url
            if getattr(cat_cfg, 'catalog_details_url', ''):
                catalog_block["soap_details_url"] = cat_cfg.catalog_details_url
            if getattr(cat_cfg, 'catalog_token', ''):
                catalog_block["token"] = cat_cfg.catalog_token

        preset_data: dict = {}
        if institution_name:
            preset_data["institution_name"] = institution_name
        if llm_block:
            preset_data["llm"] = llm_block
        if catalog_block:
            preset_data["catalog"] = catalog_block

        # File save dialog – default to project root
        from ..utils.path_utils import get_project_root
        default_path = str(get_project_root() / "alima_presets.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Preset speichern", default_path,
            "JSON-Dateien (*.json);;Alle Dateien (*)"
        )
        if not path:
            return

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(
                self, "Preset exportiert",
                f"✅ Preset gespeichert:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"Konnte Preset nicht speichern:\n{e}")

    def _save_and_close(self):
        """Save configuration and close dialog - Claude Generated"""
        try:
            # Task preferences are now handled by the unified provider tab automatically
            
            # Provider preferences are handled by the unified provider tab and Unit of Work pattern - Claude Generated (Refactoring)
            
            # Get configuration from UI
            config = self._get_config_from_ui()
            
            # Determine scope
            scope = "user"  # Default
            if self.scope_project.isChecked():
                scope = "project"
            elif self.scope_system.isChecked():
                scope = "system"
            
            # Save configuration
            success = self.config_manager.save_config(config, scope)
            if not success:
                QMessageBox.critical(self, "Save Error", "Failed to save configuration!")
                return

            # Emit signal and close
            self.config_changed.emit()
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving configuration: {str(e)}")

    def _create_task_preferences_tab(self) -> QWidget:
        """Create task preferences management tab - Claude Generated"""
        widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(15)
        
        # Left side: Task list
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Task categories header
        tasks_header = QLabel("📋 Verfügbare Tasks")
        tasks_header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        left_layout.addWidget(tasks_header)
        
        # Tasks list widget
        self.tasks_list = QListWidget()
        self.tasks_list.setMinimumWidth(250)
        self.tasks_list.setMaximumWidth(300)
        self.tasks_list.currentItemChanged.connect(self._on_task_selected)
        
        # Populate tasks from prompts.json and pipeline tasks
        self._populate_tasks_list()
        
        left_layout.addWidget(self.tasks_list)
        left_widget.setLayout(left_layout)
        
        # Right side: Model priority configuration
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Right side header
        config_header = QLabel("⚙️ Modell-Prioritäten konfigurieren")
        config_header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        right_layout.addWidget(config_header)
        
        # Selected task info
        self.selected_task_label = QLabel("Wählen Sie einen Task aus der Liste")
        self.selected_task_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        right_layout.addWidget(self.selected_task_label)
        
        # Chunked checkbox (only for applicable tasks)
        self.chunked_checkbox = QCheckBox("Spezielle Modelle für große Texte (Chunked)")
        self.chunked_checkbox.setVisible(False)
        self.chunked_checkbox.stateChanged.connect(self._on_chunked_toggled)
        right_layout.addWidget(self.chunked_checkbox)
        
        # Model priority list (main)
        priority_label = QLabel("Modell-Priorität:")
        priority_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        right_layout.addWidget(priority_label)
        
        # Model list with drag and drop
        self.model_priority_list = QListWidget()
        self.model_priority_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.model_priority_list.setMinimumHeight(200)
        right_layout.addWidget(self.model_priority_list)
        
        # Chunked model priority list (optional)
        self.chunked_priority_label = QLabel("Chunked-Modell-Priorität:")
        self.chunked_priority_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.chunked_priority_label.setVisible(False)
        right_layout.addWidget(self.chunked_priority_label)
        
        self.chunked_model_priority_list = QListWidget()
        self.chunked_model_priority_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.chunked_model_priority_list.setMinimumHeight(150)
        self.chunked_model_priority_list.setVisible(False)
        right_layout.addWidget(self.chunked_model_priority_list)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        add_model_btn = QPushButton("➕ Modell hinzufügen")
        add_model_btn.clicked.connect(self._add_model_to_priority)
        button_layout.addWidget(add_model_btn)
        
        remove_model_btn = QPushButton("➖ Modell entfernen")
        remove_model_btn.clicked.connect(self._remove_model_from_priority)
        button_layout.addWidget(remove_model_btn)
        
        button_layout.addStretch()
        
        reset_task_btn = QPushButton("🔄 Task zurücksetzen")
        reset_task_btn.clicked.connect(self._reset_task_to_default)
        button_layout.addWidget(reset_task_btn)
        
        right_layout.addLayout(button_layout)
        right_layout.addStretch()
        
        right_widget.setLayout(right_layout)
        
        # Add splitter for resizable panes
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 500])  # Give more space to right side
        
        main_layout.addWidget(splitter)
        widget.setLayout(main_layout)
        return widget
    
    def _populate_tasks_list(self):
        """Populate the tasks list with pipeline and vision tasks - Claude Generated"""
        self.tasks_list.clear()

        # Define task lists before try block for scope availability
        pipeline_tasks = ["initialisation", "keywords", "classification"]
        vision_tasks = ["vision"]  # Maps to image_text_extraction internally

        # Pipeline tasks section
        pipeline_header = QListWidgetItem("🔥 Pipeline Tasks")
        pipeline_header.setFlags(pipeline_header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        pipeline_header.setBackground(QPalette().alternateBase())
        pipeline_header.setFont(QFont("", -1, QFont.Weight.Bold))
        self.tasks_list.addItem(pipeline_header)

        for task in pipeline_tasks:
            item = QListWidgetItem(f"  📋 {task}")
            item.setData(Qt.ItemDataRole.UserRole, {"task_name": task, "category": "pipeline"})
            self.tasks_list.addItem(item)

        # Vision tasks section
        vision_header = QListWidgetItem("👁️ Vision Tasks")
        vision_header.setFlags(vision_header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        vision_header.setBackground(QPalette().alternateBase())
        vision_header.setFont(QFont("", -1, QFont.Weight.Bold))
        self.tasks_list.addItem(vision_header)

        for task in vision_tasks:
            item = QListWidgetItem(f"  👁️ {task} (Bilderkennung)")
            item.setData(Qt.ItemDataRole.UserRole, {"task_name": task, "category": "vision"})
            self.tasks_list.addItem(item)

        # Load additional tasks from prompts.json
        other_tasks = []
        try:
            from ..llm.prompt_service import PromptService
            import os
            prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts.json')
            
            if os.path.exists(prompts_path):
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    import json
                    prompts_data = json.load(f)
                
                for task_name in prompts_data.keys():
                    if (task_name not in pipeline_tasks and 
                        task_name not in vision_tasks and
                        not task_name.startswith('_')):
                        other_tasks.append(task_name)
        
        except Exception as e:
            self.logger.warning(f"Could not load additional tasks from prompts.json: {e}")
        
        if other_tasks:
            other_header = QListWidgetItem("🔧 Weitere Tasks")
            other_header.setFlags(other_header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            other_header.setBackground(QPalette().alternateBase())
            other_header.setFont(QFont("", -1, QFont.Weight.Bold))
            self.tasks_list.addItem(other_header)
            
            for task in other_tasks:
                item = QListWidgetItem(f"  🔧 {task}")
                item.setData(Qt.ItemDataRole.UserRole, {"task_name": task, "category": "other"})
                self.tasks_list.addItem(item)
    
    def _on_task_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle task selection change - Claude Generated"""
        if not current or not current.data(Qt.ItemDataRole.UserRole):
            self.selected_task_label.setText("Wählen Sie einen Task aus der Liste")
            self.chunked_checkbox.setVisible(False)
            self.chunked_priority_label.setVisible(False)
            self.chunked_model_priority_list.setVisible(False)
            self.model_priority_list.clear()
            self.chunked_model_priority_list.clear()
            return
        
        task_data = current.data(Qt.ItemDataRole.UserRole)
        task_name = task_data["task_name"]
        category = task_data["category"]
        
        self.selected_task_label.setText(f"Task: {task_name} ({category})")
        
        # Show chunked options for applicable tasks
        chunked_applicable = task_name in ["keywords", "initialisation"] or category == "pipeline"
        self.chunked_checkbox.setVisible(chunked_applicable)
        
        # Load current model priorities for this task
        self._load_task_model_priorities(task_name)
    
    def _load_task_model_priorities(self, task_name: str):
        """Load model priorities for the selected task - Claude Generated"""
        self.model_priority_list.clear()
        self.chunked_model_priority_list.clear()

        try:
            # Get unified config from current edit state (not from disk)
            unified_config = self.config_to_edit.unified_config

            # Get model priority for this task
            model_priority = unified_config.get_model_priority_for_task(task_name, is_chunked=False)
            
            # Populate main priority list
            for model_config in model_priority:
                provider_name = model_config["provider_name"]
                model_name = model_config["model_name"]
                item_text = f"{provider_name}: {model_name}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, model_config)
                self.model_priority_list.addItem(item)
            
            # Check if task has chunked support
            if task_name in unified_config.task_preferences:
                task_pref = unified_config.task_preferences[task_name]
                if hasattr(task_pref, 'chunked_model_priority') and task_pref.chunked_model_priority:
                    self.chunked_checkbox.setChecked(True)
                    self._on_chunked_toggled(True)
                    
                    # Populate chunked priority list
                    for model_config in task_pref.chunked_model_priority:
                        provider_name = model_config["provider_name"]
                        model_name = model_config["model_name"]
                        item_text = f"{provider_name}: {model_name}"
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.ItemDataRole.UserRole, model_config)
                        self.chunked_model_priority_list.addItem(item)
                else:
                    self.chunked_checkbox.setChecked(False)
                    self._on_chunked_toggled(False)
            
        except Exception as e:
            self.logger.error(f"Error loading task model priorities: {e}")
            QMessageBox.warning(self, "Load Error", f"Could not load model priorities for task '{task_name}':\n{str(e)}")
    
    def _on_chunked_toggled(self, checked: bool):
        """Handle chunked checkbox toggle - Claude Generated"""
        self.chunked_priority_label.setVisible(checked)
        self.chunked_model_priority_list.setVisible(checked)
    
    def _add_model_to_priority(self):
        """Add model to priority list - Claude Generated"""
        current_item = self.tasks_list.currentItem()
        if not current_item or not current_item.data(Qt.ItemDataRole.UserRole):
            QMessageBox.information(self, "No Task Selected", "Please select a task first.")
            return
        
        # Create dialog for model selection (pass working copy, not disk config)
        dialog = ModelSelectionDialog(self.config_to_edit.unified_config, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            provider_name, model_name = dialog.get_selected_model()
            if provider_name and model_name:
                model_config = {"provider_name": provider_name, "model_name": model_name}
                item_text = f"{provider_name}: {model_name}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, model_config)
                
                # Add to appropriate list based on chunked checkbox
                if self.chunked_checkbox.isChecked() and self.chunked_checkbox.isVisible():
                    # Ask which list to add to
                    reply = QMessageBox.question(
                        self, "Add to Which List?", 
                        "Add to standard priority list or chunked priority list?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.model_priority_list.addItem(item)
                    else:
                        self.chunked_model_priority_list.addItem(item)
                else:
                    self.model_priority_list.addItem(item)
    
    def _remove_model_from_priority(self):
        """Remove selected model from priority list - Claude Generated"""
        # Try main list first
        current_item = self.model_priority_list.currentItem()
        if current_item:
            row = self.model_priority_list.row(current_item)
            self.model_priority_list.takeItem(row)
            return
        
        # Try chunked list
        current_item = self.chunked_model_priority_list.currentItem()
        if current_item:
            row = self.chunked_model_priority_list.row(current_item)
            self.chunked_model_priority_list.takeItem(row)
            return
        
        QMessageBox.information(self, "No Selection", "Please select a model to remove.")
    
    def _reset_task_to_default(self):
        """Reset task to default model priorities - Claude Generated"""
        current_item = self.tasks_list.currentItem()
        if not current_item or not current_item.data(Qt.ItemDataRole.UserRole):
            QMessageBox.information(self, "No Task Selected", "Please select a task first.")
            return
        
        task_data = current_item.data(Qt.ItemDataRole.UserRole)
        task_name = task_data["task_name"]
        
        reply = QMessageBox.question(
            self, "Reset Task", 
            f"Reset task '{task_name}' to default model priorities?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Remove task from unified config (will fall back to defaults) - Claude Generated (Refactoring)
                if task_name in self.config_to_edit.unified_config.task_preferences:
                    del self.config_to_edit.unified_config.task_preferences[task_name]
                
                # Reload priorities
                self._load_task_model_priorities(task_name)
                QMessageBox.information(self, "Reset Complete", f"Task '{task_name}' reset to default priorities.")
                
            except Exception as e:
                self.logger.error(f"Error resetting task: {e}")
                QMessageBox.critical(self, "Reset Error", f"Could not reset task '{task_name}':\n{str(e)}")


class ModelSelectionDialog(QDialog):
    """Dialog for selecting provider and model - Claude Generated"""

    def __init__(self, config_manager_or_unified_config, parent=None):
        super().__init__(parent)
        # Support both config_manager (legacy) and UnifiedProviderConfig (new) - Claude Generated
        from ..utils.config_models import UnifiedProviderConfig
        if isinstance(config_manager_or_unified_config, UnifiedProviderConfig):
            self.unified_config = config_manager_or_unified_config
        else:
            # Legacy: config_manager - load from disk
            self.config_manager = config_manager_or_unified_config
            self.unified_config = self.config_manager.get_unified_config()

        self.setWindowTitle("Modell auswählen")
        self.setModal(True)
        self.resize(400, 300)

        self.setup_ui()
        self.load_providers()
    
    def setup_ui(self):
        """Setup dialog UI - Claude Generated"""
        layout = QVBoxLayout()
        
        # Provider selection
        provider_label = QLabel("Provider auswählen:")
        layout.addWidget(provider_label)
        
        self.provider_combo = QComboBox()
        self.provider_combo.currentTextChanged.connect(self.load_models)
        layout.addWidget(self.provider_combo)
        
        # Model selection
        model_label = QLabel("Modell auswählen:")
        layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        layout.addWidget(self.model_combo)
        
        # Custom model input
        custom_label = QLabel("Oder eigenen Modellnamen eingeben:")
        layout.addWidget(custom_label)
        
        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText("z.B. custom-model:latest")
        layout.addWidget(self.custom_model_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("Abbrechen")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def load_providers(self):
        """Load available providers - Claude Generated"""
        self.provider_combo.clear()

        try:
            # Add common providers
            providers = ["ollama", "gemini", "openai", "anthropic"]

            # Add configured providers from unified config (use stored instance, not disk)
            for provider in self.unified_config.get_enabled_providers():
                if provider.name not in providers:
                    providers.append(provider.name)
            
            self.provider_combo.addItems(providers)
            
        except Exception as e:
            # Fallback to basic providers
            self.provider_combo.addItems(["ollama", "gemini", "openai", "anthropic"])
    
    def load_models(self, provider_name: str):
        """Load models for selected provider - Claude Generated"""
        self.model_combo.clear()
        
        if not provider_name:
            return
        
        try:
            # Add common models based on provider
            if provider_name == "ollama":
                models = ["cogito:14b", "cogito:32b", "llama3.2:latest", "mistral:latest"]
            elif provider_name == "gemini":
                models = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
            elif provider_name == "openai":
                models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
            elif provider_name == "anthropic":
                models = ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"]
            else:
                models = ["default"]
            
            self.model_combo.addItems(models)
            
        except Exception:
            # Fallback
            self.model_combo.addItems(["default"])
    
    def get_selected_model(self):
        """Get selected provider and model - Claude Generated"""
        provider = self.provider_combo.currentText()
        
        # Use custom model if provided
        custom_model = self.custom_model_input.text().strip()
        if custom_model:
            model = custom_model
        else:
            model = self.model_combo.currentText()
        
        return provider, model
    


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = ComprehensiveSettingsDialog()
    dialog.show()
    sys.exit(app.exec())