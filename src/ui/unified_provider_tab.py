#!/usr/bin/env python3
"""
Unified Provider Tab - Consolidates LLM Provider and Provider Preferences configuration
Replaces fragmented LLM + Provider Preferences tabs with a single, coherent interface.
Claude Generated
"""

import logging
from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox, QFormLayout,
    QLineEdit, QPushButton, QLabel, QComboBox, QSpinBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QProgressBar, QTextEdit,
    QSplitter, QFrame, QScrollArea, QGridLayout, QDialog, QDialogButtonBox,
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QIcon, QPalette

from ..utils.config_manager import ConfigManager, OpenAICompatibleProvider, OllamaProvider, ProviderDetectionService
from ..utils.config_models import (
    UnifiedProviderConfig,
    UnifiedProvider,
    TaskPreference,
    TaskType as UnifiedTaskType,
    AlimaConfig
)
from ..utils.model_capabilities import get_chunking_threshold  # For per-model chunking UI - Claude Generated


class ProviderTestWorker(QThread):
    """Worker thread for testing provider connections - Claude Generated"""
    
    test_completed = pyqtSignal(str, bool, str)  # provider_name, success, message
    
    def __init__(self, provider: UnifiedProvider):
        super().__init__()
        self.provider = provider
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Test provider connection - Claude Generated"""
        try:
            provider_name = self.provider.name
            
            if self.provider.provider_type == "ollama":
                # Test Ollama connection
                import requests
                base_url = self.provider.get_base_url()
                response = requests.get(f"{base_url}/api/tags", timeout=10)
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_count = len(models)
                    self.test_completed.emit(provider_name, True, f"Connected successfully. {model_count} models available.")
                else:
                    self.test_completed.emit(provider_name, False, f"HTTP {response.status_code}")
                    
            elif self.provider.provider_type == "openai_compatible":
                # Test OpenAI-compatible connection
                api_key = self.provider.api_key
                if not api_key:
                    self.test_completed.emit(provider_name, False, "No API key configured")
                    return
                
                # For now, just validate configuration
                self.test_completed.emit(provider_name, True, "Configuration valid (no test call)")
                
            elif self.provider.provider_type in ["gemini", "anthropic"]:
                # Test static provider API keys
                api_key = self.provider.api_key
                if not api_key:
                    self.test_completed.emit(provider_name, False, "No API key configured")
                else:
                    self.test_completed.emit(provider_name, True, "API key configured (no test call)")
            
            else:
                self.test_completed.emit(provider_name, False, f"Unknown provider type: {self.provider.provider_type}")
                
        except Exception as e:
            self.test_completed.emit(self.provider.name, False, f"Test failed: {str(e)}")


class TaskModelSelectionDialog(QDialog):
    """Dialog for selecting provider and model for task-specific preferences - Claude Generated"""

    def __init__(self, config_manager, task_name=None, current_model_info=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.task_name = task_name
        self.current_model_info = current_model_info or {}
        self._pending_cache_update = None  # Stores (provider_name, models) for later save
        self.status_label = None  # Will be set in _setup_ui

        self.setWindowTitle("Select Provider and Model")
        self.setModal(True)
        self.resize(450, 350)

        self._setup_ui()
        self._load_providers()
    
    def _setup_ui(self):
        """Setup dialog UI with real provider/model detection - Claude Generated"""
        layout = QVBoxLayout()
        
        # Provider selection
        provider_group = QGroupBox("🌐 Select Provider")
        provider_layout = QFormLayout(provider_group)
        
        self.provider_combo = QComboBox()
        self.provider_combo.currentTextChanged.connect(self._load_models)
        provider_layout.addRow("Available Providers:", self.provider_combo)
        layout.addWidget(provider_group)
        
        # Model selection
        model_group = QGroupBox("🎯 Select Model")
        model_layout = QVBoxLayout(model_group)

        model_info_label = QLabel("Choose a model, type to filter/search, or press Enter to validate:")
        model_layout.addWidget(model_info_label)

        # Make model combo editable for quick filtering/typing - Claude Generated
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)  # Don't add custom items
        # Enable substring filtering - Claude Generated
        from PyQt6.QtWidgets import QCompleter
        self.model_combo.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        # Validate selected model exists when losing focus or pressing Enter - Claude Generated
        self.model_combo.lineEdit().editingFinished.connect(self._validate_model_selection)
        model_layout.addWidget(self.model_combo)

        layout.addWidget(model_group)

        # Status label for model detection feedback - Claude Generated
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { color: #666; font-style: italic; margin: 5px; }")
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("✅ OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        button_layout.addWidget(ok_button)
        
        cancel_button = QPushButton("❌ Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _load_providers(self):
        """Load real available providers using detection service - Claude Generated"""
        self.provider_combo.clear()
        
        try:
            detection_service = ProviderDetectionService()
            available_providers = detection_service.get_available_providers()
            
            if not available_providers:
                self.provider_combo.addItem("No providers available")
                return
            
            self.provider_combo.addItems(available_providers)
            
        except Exception as e:
            # Fallback to basic providers if detection fails
            fallback_providers = ["ollama", "gemini", "openai", "anthropic"]
            self.provider_combo.addItems(fallback_providers)
    
    def _load_models(self, provider_name: str):
        """Load models using two-tier strategy: detected → cached - Claude Generated

        TIER 1: Try live detection from provider
        TIER 2: Use cached models from config if detection fails
        """
        self.model_combo.clear()
        self.model_combo.setEnabled(True)

        if not provider_name or provider_name == "No providers available":
            return

        models_to_display = []
        detection_success = False

        # TIER 1: Try live detection
        try:
            detection_service = ProviderDetectionService(self.config_manager)
            detected_models = detection_service.get_available_models(provider_name)

            if detected_models:
                models_to_display = detected_models
                detection_success = True
                self.status_label.setText(f"✅ {len(detected_models)} models detected from {provider_name}")

        except Exception as e:
            # Detection failed, will try TIER 2
            pass

        # TIER 2: Use cached models from config (if TIER 1 failed)
        if not models_to_display and self.config_manager:
            cached_models = self._get_cached_models_from_config(provider_name)

            if cached_models:
                models_to_display = cached_models
                self.status_label.setText(f"⚠️ Provider offline - showing {len(cached_models)} cached models")

        # Populate dropdown (no visual distinction between detected and cached)
        if models_to_display:
            for model in models_to_display:
                self.model_combo.addItem(model, model)

            # Pre-select current model if available
            self._preselect_current_model()
        else:
            # No models available at all
            self.model_combo.addItem("No models available", None)
            self.model_combo.setEnabled(False)
            self.status_label.setText(f"❌ No models available for {provider_name}")

        # Store detected models for later caching (only on dialog OK)
        if detection_success:
            self._pending_cache_update = (provider_name, models_to_display)

    def _get_cached_models_from_config(self, provider_name: str) -> list:
        """Get models from config cache for offline use - Claude Generated"""
        if not self.config_manager:
            return []

        try:
            config = self.config_manager.load_config()
            for provider in config.unified_config.providers:
                if provider.name.lower() == provider_name.lower():
                    # Return available_models if populated
                    if provider.available_models:
                        return provider.available_models
                    # Fallback to preferred_model if set
                    if provider.preferred_model:
                        return [provider.preferred_model]
        except Exception:
            pass

        return []

    def _update_provider_cached_models(self, provider_name: str, models: list) -> None:
        """Update provider's cached model list in config - Claude Generated

        Called only when dialog is accepted (OK button), not during detection.
        """
        if not self.config_manager or not models:
            return

        try:
            config = self.config_manager.load_config()
            for provider in config.unified_config.providers:
                if provider.name.lower() == provider_name.lower():
                    provider.available_models = models
                    self.config_manager.save_config(config)
                    break
        except Exception:
            pass

    def _preselect_current_model(self) -> None:
        """Pre-select the currently configured model in dropdown - Claude Generated"""
        if not self.current_model_info:
            return

        current_provider = self.current_model_info.get('provider')
        current_model = self.current_model_info.get('model')

        # Check if current provider matches selected provider
        if current_provider and current_provider.lower() == self.provider_combo.currentText().lower():
            # Try to find and select current model
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == current_model:
                    self.model_combo.setCurrentIndex(i)
                    break

    def _validate_model_selection(self) -> None:
        """Validate that typed model name exists in available models - Claude Generated"""
        current_text = self.model_combo.currentText().strip()

        if not current_text:
            return  # Empty is OK (will show as "No models available" handling)

        # Check if text matches any available model (case-insensitive)
        model_found = False
        for i in range(self.model_combo.count()):
            item_text = self.model_combo.itemText(i)
            if item_text.lower() == current_text.lower():
                # Found exact match - select it
                self.model_combo.setCurrentIndex(i)
                model_found = True
                self.status_label.setText(f"✅ Model '{current_text}' verified")
                break

        if not model_found and current_text != "No models available":
            # Model doesn't exist - show error and reset
            self.status_label.setText(
                f"❌ Model '{current_text}' not found in {self.provider_combo.currentText()}"
            )
            self.status_label.setStyleSheet(
                "QLabel { color: #d32f2f; font-style: italic; margin: 5px; }"
            )
            # Clear the input
            self.model_combo.lineEdit().clear()
            self.status_label.setStyleSheet("QLabel { color: #666; font-style: italic; margin: 5px; }")

    def accept(self) -> None:
        """Override accept to update cached models before closing - Claude Generated

        Only saves detected models to config when user confirms (OK button).
        """
        # Update config cache if we have pending detected models
        if self._pending_cache_update:
            provider_name, models = self._pending_cache_update
            self._update_provider_cached_models(provider_name, models)

        super().accept()
    
    def get_selected_model(self):
        """Get selected provider and model - Claude Generated"""
        provider = self.provider_combo.currentText()

        if provider == "No providers available":
            return None, None

        model = self.model_combo.currentData()

        if not model:
            return None, None

        return provider, model
    
    def set_default_from_global_preferences(self, provider_preferences):
        """Set dialog defaults from global provider preferences - Claude Generated"""
        try:
            # Set preferred provider as default
            preferred_provider = provider_preferences.preferred_provider
            provider_index = self.provider_combo.findText(preferred_provider)
            if provider_index >= 0:
                self.provider_combo.setCurrentIndex(provider_index)
                self._on_provider_changed()  # Trigger model list update
                
                # Set preferred model for this provider
                preferred_model = provider_preferences.preferred_models.get(preferred_provider)
                if preferred_model:
                    model_index = -1
                    for i in range(self.model_combo.count()):
                        if self.model_combo.itemData(i) == preferred_model:
                            model_index = i
                            break
                    
                    if model_index >= 0:
                        self.model_combo.setCurrentIndex(model_index)
                    else:
                        # Model not found in dropdown, fallback to auto-select
                        self.model_combo.setCurrentIndex(0)  # "(Auto-select)" is always first
                        
        except Exception as e:
            # Silent fallback - don't interrupt user workflow
            pass


class ProviderEditDialog(QDialog):
    """Dialog for editing provider configuration - Claude Generated"""
    
    def __init__(self, provider: Optional[UnifiedProvider] = None, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.is_editing = provider is not None
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("Edit Provider" if self.is_editing else "Add Provider")
        self.setModal(True)
        self.resize(500, 400)
        
        self._setup_ui()
        
        if self.is_editing:
            self._load_provider_data()
    
    def _setup_ui(self):
        """Setup the dialog UI - Claude Generated"""
        layout = QVBoxLayout(self)
        
        # Provider Type Selection
        type_group = QGroupBox("Provider Type")
        type_layout = QFormLayout(type_group)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["ollama", "openai_compatible", "gemini", "anthropic"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addRow("Type:", self.type_combo)
        
        layout.addWidget(type_group)
        
        # Basic Configuration
        basic_group = QGroupBox("Basic Configuration")
        basic_layout = QFormLayout(basic_group)
        
        self.name_edit = QLineEdit()
        self.description_edit = QLineEdit()
        self.enabled_checkbox = QCheckBox()
        self.enabled_checkbox.setChecked(True)
        self.preferred_model_edit = QLineEdit()
        self.preferred_model_edit.setPlaceholderText("(leer = automatisch)")
        self.preferred_model_edit.setToolTip("Standard-Modell für diesen Provider (leer lassen für automatische Auswahl)")

        basic_layout.addRow("Name:", self.name_edit)
        basic_layout.addRow("Description:", self.description_edit)
        basic_layout.addRow("Enabled:", self.enabled_checkbox)
        basic_layout.addRow("Preferred Model:", self.preferred_model_edit)
        
        layout.addWidget(basic_group)
        
        # Connection Configuration (dynamic based on type)
        self.connection_group = QGroupBox("Connection Configuration")
        self.connection_layout = QFormLayout(self.connection_group)
        layout.addWidget(self.connection_group)

        # Warning label for untested providers
        self.warning_label = QLabel()
        self.warning_label.setStyleSheet("color: orange; padding: 5px;")
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Initialize connection fields
        self._on_type_changed(self.type_combo.currentText())
    
    def _on_type_changed(self, provider_type: str):
        """Update connection configuration fields based on provider type - Claude Generated"""
        # Clear existing fields
        while self.connection_layout.count():
            child = self.connection_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add type-specific fields
        if provider_type == "ollama":
            self.host_edit = QLineEdit("localhost")
            self.port_spinbox = QSpinBox()
            self.port_spinbox.setRange(1, 65535)
            self.port_spinbox.setValue(11434)
            self.ssl_checkbox = QCheckBox()
            self.api_key_edit = QLineEdit()
            
            self.connection_layout.addRow("Host:", self.host_edit)
            self.connection_layout.addRow("Port:", self.port_spinbox)
            self.connection_layout.addRow("Use SSL:", self.ssl_checkbox)
            self.connection_layout.addRow("API Key (optional):", self.api_key_edit)
            
        elif provider_type == "openai_compatible":
            self.base_url_edit = QLineEdit()
            self.api_key_edit = QLineEdit()
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            
            self.connection_layout.addRow("Base URL:", self.base_url_edit)
            self.connection_layout.addRow("API Key:", self.api_key_edit)
            
        elif provider_type in ["gemini", "anthropic"]:
            self.api_key_edit = QLineEdit()
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

            self.connection_layout.addRow("API Key:", self.api_key_edit)

        # Show warning for untested providers
        if hasattr(self, 'warning_label'):
            if provider_type == "gemini":
                self.warning_label.setText(
                    "⚠️ Google Gemini-Provider ist nicht getestet und benötigt Nachbearbeitung. "
                    "API-Integration funktioniert möglicherweise nicht vollständig."
                )
                self.warning_label.show()
            elif provider_type == "anthropic":
                self.warning_label.setText(
                    "⚠️ Anthropic/Claude-Provider ist nicht getestet und benötigt Nachbearbeitung. "
                    "API-Integration funktioniert möglicherweise nicht vollständig."
                )
                self.warning_label.show()
            else:
                self.warning_label.hide()
    
    def _load_provider_data(self):
        """Load provider data into form fields - Claude Generated"""
        if not self.provider:
            return
        
        self.name_edit.setText(self.provider.name)
        self.description_edit.setText(self.provider.description)
        self.enabled_checkbox.setChecked(self.provider.enabled)
        self.preferred_model_edit.setText(self.provider.preferred_model or "")
        self.type_combo.setCurrentText(self.provider.provider_type)

        # Load connection configuration from provider attributes
        if self.provider.provider_type == "ollama":
            if hasattr(self, 'host_edit'):
                self.host_edit.setText(self.provider.host or "localhost")
            if hasattr(self, 'port_spinbox'):
                self.port_spinbox.setValue(self.provider.port or 11434)
            if hasattr(self, 'ssl_checkbox'):
                self.ssl_checkbox.setChecked(self.provider.use_ssl or False)
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(self.provider.api_key or "")

        elif self.provider.provider_type == "openai_compatible":
            if hasattr(self, 'base_url_edit'):
                self.base_url_edit.setText(self.provider.base_url or "")
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(self.provider.api_key or "")

        elif self.provider.provider_type in ["gemini", "anthropic"]:
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(self.provider.api_key or "")
    
    def get_provider_data(self) -> UnifiedProvider:
        """Extract provider data from form fields - Claude Generated"""
        # Build connection config based on type
        connection_config = {}
        provider_type = self.type_combo.currentText()
        
        if provider_type == "ollama":
            connection_config = {
                "host": getattr(self, 'host_edit', QLineEdit()).text() or "localhost",
                "port": getattr(self, 'port_spinbox', QSpinBox()).value() or 11434,
                "use_ssl": getattr(self, 'ssl_checkbox', QCheckBox()).isChecked(),
                "api_key": getattr(self, 'api_key_edit', QLineEdit()).text(),
                "connection_type": "native_client"
            }
        elif provider_type == "openai_compatible":
            connection_config = {
                "base_url": getattr(self, 'base_url_edit', QLineEdit()).text(),
                "api_key": getattr(self, 'api_key_edit', QLineEdit()).text()
            }
        elif provider_type in ["gemini", "anthropic"]:
            connection_config = {
                "api_key": getattr(self, 'api_key_edit', QLineEdit()).text()
            }

        return UnifiedProvider(
            name=self.name_edit.text() or f"New {provider_type.title()} Provider",
            provider_type=provider_type,
            enabled=self.enabled_checkbox.isChecked(),
            api_key=connection_config.get("api_key", ""),
            base_url=connection_config.get("base_url", ""),
            preferred_model=self.preferred_model_edit.text().strip(),
            description=self.description_edit.text(),
            # Ollama specific fields
            host=connection_config.get("host", ""),
            port=connection_config.get("port", 11434),
            use_ssl=connection_config.get("use_ssl", False),
            connection_type=connection_config.get("connection_type", "native_client")
        )


class ModelFetchWorker(QThread):
    """Background worker to fetch model lists for all providers without blocking the UI - Claude Generated"""

    models_fetched = pyqtSignal(dict)  # provider_name -> [model_names]

    def __init__(self, detection_service, providers: list):
        super().__init__()
        self.detection_service = detection_service
        self.providers = providers
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Fetch models for each provider sequentially in background - Claude Generated"""
        result = {}
        for prov in self.providers:
            try:
                result[prov] = self.detection_service.get_available_models(prov)
            except Exception as e:
                self.logger.warning(f"ModelFetchWorker: failed to fetch models for {prov}: {e}")
                result[prov] = []
        self.models_fetched.emit(result)


class UnifiedProviderTab(QWidget):
    """
    Unified Provider Configuration Tab - Claude Generated
    Consolidates LLM Provider and Provider Preferences into single interface
    """

    config_changed = pyqtSignal()
    task_preferences_changed = pyqtSignal()  # New signal for task preference changes - Claude Generated

    # Class constant: maps task name strings to TaskType enums (includes legacy aliases) - Claude Generated
    TASK_TYPE_MAPPING = {
        'INITIALISATION': UnifiedTaskType.INITIALISATION,
        'KEYWORDS': UnifiedTaskType.KEYWORDS,
        'CLASSIFICATION': UnifiedTaskType.CLASSIFICATION,
        'DK_CLASSIFICATION': UnifiedTaskType.DK_CLASSIFICATION,
        'VISION': UnifiedTaskType.VISION,
        'CHUNKED_PROCESSING': UnifiedTaskType.CHUNKED_PROCESSING,
        # Legacy lowercase aliases
        'initialisation': UnifiedTaskType.INITIALISATION,
        'keywords': UnifiedTaskType.KEYWORDS,
        'classification': UnifiedTaskType.CLASSIFICATION,
        'dk_classification': UnifiedTaskType.DK_CLASSIFICATION,
        'vision': UnifiedTaskType.VISION,
        'chunked': UnifiedTaskType.CHUNKED_PROCESSING,
        'chunked_processing': UnifiedTaskType.CHUNKED_PROCESSING,
        # Legacy prompt-specific names
        'rephrase': UnifiedTaskType.KEYWORDS,
        'image_text_extraction': UnifiedTaskType.VISION,
        'keywords_chunked': UnifiedTaskType.CHUNKED_PROCESSING,
        'extract_initial_keywords': UnifiedTaskType.INITIALISATION,
    }
    
    def __init__(self, unified_config: UnifiedProviderConfig, alima_config: AlimaConfig,
                 config_manager: ConfigManager, alima_manager=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Store all dependencies - Claude Generated
        self.unified_config = unified_config
        self.config = alima_config
        self.config_manager = config_manager  # ✅ ConfigManager for persistence operations
        self.alima_manager = alima_manager  # Access to ProviderStatusService - Claude Generated

        # CRITICAL FIX: Add explicit task tracking to prevent cross-contamination - Claude Generated
        self.current_editing_task = None  # Track which task is currently being edited
        self.task_ui_dirty = False  # Track if current task UI has unsaved changes

        self._setup_ui()
        self._load_configuration()

        # Setup reactive provider status connections - Claude Generated
        self._setup_provider_status_connections()
    
    def _setup_ui(self):
        """Setup the unified provider tab UI - Claude Generated"""
        layout = QVBoxLayout(self)

        # Main content in tabs
        self.main_tabs = QTabWidget()
        layout.addWidget(self.main_tabs)
        
        # Provider Management Tab
        self.providers_tab = self._create_providers_tab()
        self.main_tabs.addTab(self.providers_tab, "🔧 Provider Management")
        
        # Task Preferences Tab  
        self.preferences_tab = self._create_preferences_tab()
        self.main_tabs.addTab(self.preferences_tab, "🎯 Task Preferences")
        
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.test_all_button = QPushButton("🧪 Test All Providers")
        self.test_all_button.clicked.connect(self._test_all_providers)

        button_layout.addWidget(self.test_all_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def _create_providers_tab(self) -> QWidget:
        """Create provider management tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Unified provider table - Cols: Name | Type | Status | Preferred Model | API Key
        self.provider_table = QTableWidget()
        self.provider_table.setColumnCount(5)
        self.provider_table.setHorizontalHeaderLabels([
            "Name", "Type", "Status", "Preferred Model", "API Key"
        ])
        self.provider_table.verticalHeader().setVisible(False)
        self.provider_table.verticalHeader().setDefaultSectionSize(40)

        header = self.provider_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Status
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Preferred Model
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # API Key

        layout.addWidget(self.provider_table)

        # Provider control buttons
        provider_button_layout = QHBoxLayout()

        self.add_provider_button = QPushButton("➕ Add Provider")
        self.add_provider_button.clicked.connect(self._add_provider)

        self.edit_provider_button = QPushButton("✏️ Edit Provider")
        self.edit_provider_button.clicked.connect(self._edit_provider)

        self.remove_provider_button = QPushButton("🗑️ Remove Provider")
        self.remove_provider_button.clicked.connect(self._remove_provider)

        self.refresh_models_button = QPushButton("🔄 Refresh Models")
        self.refresh_models_button.clicked.connect(self._refresh_models)

        provider_button_layout.addWidget(self.add_provider_button)
        provider_button_layout.addWidget(self.edit_provider_button)
        provider_button_layout.addWidget(self.remove_provider_button)
        provider_button_layout.addWidget(self.refresh_models_button)
        provider_button_layout.addStretch()
        layout.addLayout(provider_button_layout)

        # Global default provider – separate row, clearly labelled
        default_provider_layout = QHBoxLayout()
        default_provider_layout.addWidget(QLabel("Default Provider:"))
        self.preferred_provider_combo = QComboBox(widget)
        self.preferred_provider_combo.setMinimumWidth(180)
        self.preferred_provider_combo.currentTextChanged.connect(self._update_config_from_ui)
        default_provider_layout.addWidget(self.preferred_provider_combo)
        default_provider_layout.addStretch()
        layout.addLayout(default_provider_layout)

        return widget

    def _create_preferences_tab(self) -> QWidget:
        """Create task preferences tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Task-specific preferences - Enhanced with advanced task management - Claude Generated
        task_group = QGroupBox("🎯 Task-Specific Model Preferences")
        task_main_layout = QVBoxLayout(task_group)
        
        # Create splitter for task management
        task_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Task categories and selection
        left_task_widget = QWidget()
        left_task_layout = QVBoxLayout(left_task_widget)
        
        task_categories_label = QLabel("📋 Available Tasks")
        task_categories_label.setStyleSheet("font-weight: bold; padding: 5px;")
        left_task_layout.addWidget(task_categories_label)
        
        self.task_categories_list = QListWidget()
        self.task_categories_list.setMinimumWidth(280)
        self.task_categories_list.setMaximumWidth(350)
        self.task_categories_list.currentItemChanged.connect(self._on_task_category_selected)
        left_task_layout.addWidget(self.task_categories_list)
        
        # Right side: Model priority configuration  
        right_task_widget = QWidget()
        right_task_layout = QVBoxLayout(right_task_widget)
        
        config_header_label = QLabel("⚙️ Model Priority Configuration")
        config_header_label.setStyleSheet("font-weight: bold; padding: 5px;")
        right_task_layout.addWidget(config_header_label)
        
        # Selected task info
        self.selected_task_info_label = QLabel("Select a task from the categories")
        self.selected_task_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        right_task_layout.addWidget(self.selected_task_info_label)
        
        # Standard model priority
        priority_label = QLabel("Model Priority:")
        priority_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        right_task_layout.addWidget(priority_label)
        
        self.task_model_priority_list = QListWidget()
        from PyQt6.QtWidgets import QAbstractItemView
        self.task_model_priority_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.task_model_priority_list.setMinimumHeight(150)
        # CRITICAL FIX: Add event handler to save priority changes after drag & drop - Claude Generated
        self.task_model_priority_list.model().rowsMoved.connect(self._on_priority_list_reordered)
        self.task_model_priority_list.currentItemChanged.connect(self._on_priority_item_selected)
        right_task_layout.addWidget(self.task_model_priority_list)

        # Per-entry chunking threshold - Claude Generated
        self.selected_model_chunking_row = QHBoxLayout()
        self.selected_model_chunking_label = QLabel("Chunking:")
        self.selected_model_chunking_spinbox = QSpinBox()
        self.selected_model_chunking_spinbox.setRange(0, 2000)
        self.selected_model_chunking_spinbox.setValue(0)
        self.selected_model_chunking_spinbox.setSuffix(" Kw")
        self.selected_model_chunking_spinbox.setSpecialValueText("Auto")
        self.selected_model_chunking_spinbox.setToolTip(
            "Chunking-Schwellwert für dieses Modell.\n"
            "Auto (0) = Erkennung basierend auf Modell.\n"
            "Große Modelle (>30B): ~1000, Mittel (13B): ~500, Klein (<7B): ~200-300"
        )
        self.selected_model_chunking_spinbox.valueChanged.connect(self._on_selected_model_chunking_changed)
        self.selected_model_chunking_row.addWidget(self.selected_model_chunking_label)
        self.selected_model_chunking_row.addWidget(self.selected_model_chunking_spinbox)
        self.selected_model_chunking_label.setVisible(False)
        self.selected_model_chunking_spinbox.setVisible(False)
        right_task_layout.addLayout(self.selected_model_chunking_row)

        # Thinking toggle per model+task - Claude Generated
        self.selected_model_think_row = QHBoxLayout()
        self.selected_model_think_label = QLabel("Thinking:")
        self.selected_model_think_combo = QComboBox()
        self.selected_model_think_combo.addItems(["Default (Anbieter)", "Aktiviert", "Deaktiviert"])
        self.selected_model_think_combo.setToolTip(
            "Thinking/CoT-Modus für dieses Modell in dieser Task.\n"
            "Default = Anbieter-Einstellung wird verwendet\n"
            "Aktiviert = think=true\nDeaktiviert = think=false"
        )
        self.selected_model_think_combo.currentIndexChanged.connect(self._on_selected_model_think_changed)
        self.selected_model_think_row.addWidget(self.selected_model_think_label)
        self.selected_model_think_row.addWidget(self.selected_model_think_combo)
        self.selected_model_think_label.setVisible(False)
        self.selected_model_think_combo.setVisible(False)
        right_task_layout.addLayout(self.selected_model_think_row)

        # Task management buttons
        task_button_layout = QHBoxLayout()
        
        add_task_model_btn = QPushButton("➕ Add Model")
        add_task_model_btn.clicked.connect(self._add_model_to_task_priority)
        task_button_layout.addWidget(add_task_model_btn)

        # Bulk add model to all tasks (except vision) - Claude Generated
        bulk_add_task_model_btn = QPushButton("➕➕ Add to All Tasks")
        bulk_add_task_model_btn.setToolTip("Add model to all text-based tasks (excludes vision)")
        bulk_add_task_model_btn.clicked.connect(self._bulk_add_model_to_tasks)
        task_button_layout.addWidget(bulk_add_task_model_btn)

        remove_task_model_btn = QPushButton("➖ Remove Model")
        remove_task_model_btn.clicked.connect(self._remove_model_from_task_priority)
        task_button_layout.addWidget(remove_task_model_btn)

        task_button_layout.addStretch()
        
        reset_task_btn = QPushButton("🔄 Reset Task")
        reset_task_btn.clicked.connect(self._reset_selected_task_to_defaults)
        task_button_layout.addWidget(reset_task_btn)
        
        right_task_layout.addLayout(task_button_layout)

        # Add widgets to splitter
        task_splitter.addWidget(left_task_widget)
        task_splitter.addWidget(right_task_widget)
        task_splitter.setSizes([300, 500])  # Give more space to right side
        
        task_main_layout.addWidget(task_splitter)
        layout.addWidget(task_group)
        
        return widget
    
    
    def _load_configuration(self):
        """Load unified configuration into UI - Claude Generated"""
        try:
            # Cache detection service and model lists to avoid repeated network calls - Claude Generated
            self._detection_service = ProviderDetectionService()
            self._cached_providers = self._detection_service.get_available_providers()
            self._cached_models = {}  # will be populated by background worker - Claude Generated
            self._start_background_model_fetch()

            # Check if we need to migrate from legacy configuration
            if not self.unified_config.providers and self.config:
                self.logger.info("No unified providers found, attempting migration from legacy config")
                self._migrate_from_legacy_config()
            
            # Load providers into table
            self._populate_provider_table()
            
            # Load global preferences
            self._populate_global_preferences()
            
            # Load model preferences
            self._populate_model_preferences()
            
            # Load task preferences
            self._populate_task_preferences()

            # CRITICAL FIX: Initialize task editing state properly - Claude Generated
            self._initialize_task_editing_state()

        except Exception as e:
            self.logger.error(f"Error loading unified provider configuration: {e}")
            QMessageBox.critical(self, "Loading Error", f"Failed to load configuration:\n\n{str(e)}")
    
    def _migrate_from_legacy_config(self):
        """Migrate from legacy LLMConfig to UnifiedProviderConfig - Claude Generated (Refactoring)"""
        try:
            # Create unified config from legacy data using the already loaded config
            self.unified_config = UnifiedProviderConfig.from_legacy_config(
                self.config.unified_config,
                self.unified_config  # Use existing unified_config instead of bridge
            )

            # Migration occurs in memory only - parent dialog will handle save
            self.config_changed.emit()

            self.logger.info(f"Successfully migrated {len(self.unified_config.providers)} providers from legacy config")

        except Exception as e:
            self.logger.error(f"Failed to migrate from legacy config: {e}")
            # Create default config if migration fails
            self.unified_config = UnifiedProviderConfig()
            self.config_changed.emit()
    
    def _start_background_model_fetch(self):
        """Start background worker to fetch model lists without blocking the UI - Claude Generated"""
        if not self._cached_providers:
            return
        self._model_fetch_worker = ModelFetchWorker(self._detection_service, list(self._cached_providers))
        self._model_fetch_worker.models_fetched.connect(self._on_models_fetched)
        self._model_fetch_worker.start()
        self.logger.debug("Started background model fetch for all providers")

    def _on_models_fetched(self, models: dict):
        """Slot called when background model fetch completes - Claude Generated"""
        try:
            self._cached_models.update(models)
            self._populate_model_preferences()
            self.logger.info(f"Background model fetch complete: {list(models.keys())}")
        except Exception as e:
            self.logger.error(f"Error applying fetched models: {e}")

    def _get_all_providers(self):
        """Get all providers from unified config + LLM config - Claude Generated"""
        all_providers = list(self.unified_config.providers)  # Start with unified providers
        existing_names = [p.name.lower() for p in all_providers]

        # Add Gemini from LLM config if configured and not already present
        gemini_conditions = [
            hasattr(self.config.unified_config, 'gemini_api_key'),
            self.config.unified_config.gemini_api_key,
            self.config.unified_config.gemini_api_key.strip() if self.config.unified_config.gemini_api_key else False,
            self.config.unified_config.gemini_api_key != "your_gemini_api_key_here" if self.config.unified_config.gemini_api_key else False,
            "gemini" not in existing_names
        ]

        if all(gemini_conditions):
            gemini_provider = UnifiedProvider(
                name="gemini",
                provider_type="gemini",
                api_key=self.config.unified_config.gemini_api_key,
                enabled=True,
                description="Gemini from LLM configuration"
            )
            all_providers.append(gemini_provider)

        # Add Anthropic from LLM config if configured and not already present
        anthropic_conditions = [
            hasattr(self.config.unified_config, 'anthropic_api_key'),
            self.config.unified_config.anthropic_api_key,
            self.config.unified_config.anthropic_api_key.strip() if self.config.unified_config.anthropic_api_key else False,
            self.config.unified_config.anthropic_api_key != "your_anthropic_api_key_here" if self.config.unified_config.anthropic_api_key else False,
            "anthropic" not in existing_names
        ]

        if all(anthropic_conditions):
            anthropic_provider = UnifiedProvider(
                name="anthropic",
                provider_type="anthropic",
                api_key=self.config.unified_config.anthropic_api_key,
                enabled=True,
                description="Anthropic from LLM configuration"
            )
            all_providers.append(anthropic_provider)

        return all_providers

    def _populate_provider_table(self):
        """Populate the unified provider table (status + preferred model) - Claude Generated"""
        providers = self._get_all_providers()
        self.provider_table.setRowCount(len(providers))

        # Get cached provider status from service (non-blocking)
        provider_status_cache = {}
        if (self.alima_manager and
            hasattr(self.alima_manager, 'provider_status_service') and
            self.alima_manager.provider_status_service):
            provider_status_cache = self.alima_manager.provider_status_service.get_all_provider_info()
            self.logger.debug(f"Using cached status for {len(provider_status_cache)} providers")
        else:
            self.logger.warning("ProviderStatusService not available, using fallback display")

        for row, provider in enumerate(providers):
            # Col 0: Name
            name_item = QTableWidgetItem(provider.name)
            if not provider.enabled:
                name_item.setForeground(QPalette().color(QPalette.ColorRole.PlaceholderText))
            self.provider_table.setItem(row, 0, name_item)

            # Col 1: Type
            self.provider_table.setItem(row, 1, QTableWidgetItem(provider.provider_type.title()))

            # Col 2: Status – derive from cache
            status_text = "Unknown"
            available_models: list = self._cached_models.get(provider.name, list(provider.available_models))

            cached_status = provider_status_cache.get(provider.name)
            if cached_status:
                is_reachable = cached_status.get('reachable', False)
                cached_models = cached_status.get('models', [])
                error_message = cached_status.get('error_message')
                if is_reachable:
                    status_text = "✅ Available"
                    if cached_models:
                        available_models = cached_models
                        provider.available_models = cached_models
                else:
                    status_text = "❌ Offline"
                    if error_message:
                        status_text = f"❌ {error_message[:25]}"
            elif provider_status_cache:
                status_text = "⏳ Testing..."
            else:
                status_text = "📝 Configured" if available_models else "⚠️ Unknown"

            self.provider_table.setItem(row, 2, QTableWidgetItem(status_text))

            # Col 3: Preferred Model – plain text (edit via provider dialog)
            preferred_model = self._get_preferred_model_from_config(provider.name, self.config)
            self.provider_table.setItem(row, 3, QTableWidgetItem(preferred_model or "(Auto)"))

            # Col 4: API Key Status
            api_key_status = self._get_api_key_status(provider)
            api_key_item = QTableWidgetItem(api_key_status)
            if "✅" in api_key_status:
                api_key_item.setForeground(QPalette().color(QPalette.ColorRole.Text))
            elif "❌" in api_key_status:
                api_key_item.setForeground(QPalette().color(QPalette.ColorRole.PlaceholderText))
            else:
                api_key_item.setForeground(QPalette().color(QPalette.ColorRole.Mid))
            self.provider_table.setItem(row, 4, api_key_item)

    def _get_api_key_status(self, provider) -> str:
        """Get API key status for a provider - Claude Generated"""
        try:
            # For Ollama providers, API key is not required
            if provider.provider_type == "ollama":
                return "➖ N/A"

            # For API-based providers, check connection_config for api_key first
            if provider.provider_type in ["gemini", "anthropic", "openai_compatible"]:
                api_key = provider.api_key or ""

                # Fallback to legacy LLM config for Gemini/Anthropic if no unified config
                if not api_key or api_key.strip() == "":
                    if provider.provider_type == "gemini":
                        api_key = getattr(self.config.unified_config, 'gemini', '')
                    elif provider.provider_type == "anthropic":
                        api_key = getattr(self.config.unified_config, 'anthropic', '')

                if api_key and api_key.strip() and api_key != "your_api_key_here":
                    # Check if API key looks like a placeholder
                    placeholder_indicators = ["your_", "_here", "api_key", "token", "key_here"]
                    is_placeholder = any(indicator in api_key.lower() for indicator in placeholder_indicators)

                    if is_placeholder:
                        return "❌ Placeholder"
                    else:
                        # Show partial key for security (first 8 chars + ...)
                        if len(api_key) > 8:
                            partial_key = api_key[:8] + "..."
                            return f"✅ {partial_key}"
                        else:
                            return "✅ Configured"
                else:
                    return "❌ Missing"

            # For unknown provider types
            return "❓ Unknown"

        except Exception as e:
            self.logger.warning(f"Error checking API key status for {provider.name}: {e}")
            return "❓ Error"

    def _populate_global_preferences(self):
        """Populate global preferences - Claude Generated"""
        # Update preferred provider combo
        self.preferred_provider_combo.clear()
        provider_names = [p.name for p in self.unified_config.providers if p.enabled]
        self.preferred_provider_combo.addItems(provider_names)
        
        # Set current selection
        if self.unified_config.preferred_provider in provider_names:
            self.preferred_provider_combo.setCurrentText(self.unified_config.preferred_provider)

    def _populate_model_preferences(self):
        """Delegates to _populate_provider_table (tables are now merged) - Claude Generated"""
        self._populate_provider_table()
    
    def _on_model_chunking_changed(self, provider: str, model: str, value: int):
        """Handle per-model chunking threshold change - Claude Generated"""
        try:
            if not model:
                # No specific model selected, can't save per-model setting
                self.logger.debug(f"No model selected for {provider}, skipping chunking threshold save")
                return

            if value > 0:
                # Set specific threshold
                self.unified_config.set_chunking_threshold(provider, model, value)
                self.logger.info(f"Set chunking threshold for {provider}/{model}: {value}")
                self._show_save_toast(f"📊 {provider}/{model}: {value} Keywords")
            else:
                # Remove threshold (revert to auto-detect)
                self.unified_config.remove_chunking_threshold(provider, model)
                auto_value = get_chunking_threshold(provider, model, config_manager=self.config_manager)
                self.logger.info(f"Removed chunking threshold for {provider}/{model} (auto: {auto_value})")
                self._show_save_toast(f"📊 {provider}/{model}: Auto ({auto_value})")

            # Emit config changed signal
            self.config_changed.emit()

        except Exception as e:
            self.logger.error(f"Error updating model chunking threshold: {e}")

    def _get_preferred_model_from_config(self, provider: str, config) -> str:
        """Get preferred model from direct provider configuration - Claude Generated"""
        try:
            # Check static providers
            if provider == "gemini":
                return config.unified_config.gemini_preferred_model or ""
            elif provider == "anthropic":
                return config.unified_config.anthropic_preferred_model or ""
            
            # Check providers in unified provider list
            for unified_provider in config.unified_config.providers:
                if unified_provider.name == provider:
                    return unified_provider.preferred_model or ""
            
            return ""
        except Exception as e:
            self.logger.warning(f"Error getting preferred model for {provider}: {e}")
            return ""
    
    # DEPRECATED: _get_available_prompt_tasks() removed
    # Now uses shared LLM_TASK_DISPLAY_INFO constant from config_models
    # This ensures consistency with wizards and prevents loading non-configurable tasks from prompts.json

    def _get_current_model_for_task(self, task_name: str) -> tuple:
        """Get currently configured provider and model for a task - Claude Generated

        Returns:
            (provider_name, model_name) or (None, None) if not configured
        """
        if not task_name or not self.config:
            return None, None

        # Check task preferences
        task_prefs = self.config.unified_config.task_preferences.get(task_name)
        if task_prefs and task_prefs.model_priority:
            first_entry = task_prefs.model_priority[0]
            return first_entry.get("provider_name"), first_entry.get("model_name")

        return None, None

    def _populate_task_preferences(self):
        """Populate task categories and load task-specific model preferences - Enhanced - Claude Generated"""
        self._populate_task_categories_list()
    
    def _populate_task_categories_list(self):
        """Populate task categories with the 6 configurable LLM tasks - Claude Generated

        Uses shared LLM_TASK_DISPLAY_INFO constant from config_models for consistency
        with wizards and to prevent loading non-configurable tasks from prompts.json.
        """
        from ..utils.config_models import LLM_TASK_DISPLAY_INFO

        self.task_categories_list.clear()

        # LLM Tasks section header
        llm_header = QListWidgetItem("🔥 Konfigurierbare LLM-Tasks")
        llm_header.setFlags(llm_header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        llm_header.setBackground(QPalette().alternateBase())
        llm_header.setFont(QFont("", -1, QFont.Weight.Bold))
        self.task_categories_list.addItem(llm_header)

        # Add the 6 configurable LLM tasks from shared constant
        for task_type, icon_label, description in LLM_TASK_DISPLAY_INFO:
            # Use the enum value (lowercase, e.g. "initialisation") as task_name
            task_name = task_type.value
            item = QListWidgetItem(f"  {icon_label}")
            item.setData(Qt.ItemDataRole.UserRole, {
                "task_name": task_name,
                "category": "llm_task",
                "description": description
            })
            item.setToolTip(description)
            self.task_categories_list.addItem(item)
    
    def _on_task_category_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle task category selection change - Claude Generated"""
        # Batch all layout changes to prevent visual jumping - Claude Generated
        self.setUpdatesEnabled(False)
        try:
            self._on_task_category_selected_inner(current, previous)
        finally:
            self.setUpdatesEnabled(True)

    def _on_task_category_selected_inner(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Inner implementation of task category selection - Claude Generated"""
        # Save previous task's changes in memory (no signals/toast) - Claude Generated
        if previous and previous.data(Qt.ItemDataRole.UserRole) and self.current_editing_task:
            # Preserve the previous task preferences in memory only
            previous_task_data = previous.data(Qt.ItemDataRole.UserRole)
            previous_task_name = previous_task_data["task_name"]
            self.logger.debug(f"Preserving preferences for previous task: {previous_task_name}")
            self._save_current_task_preferences(explicit_task_name=previous_task_name, emit_signals=False)

        # Preserve current task changes in memory before clearing - Claude Generated
        if self.current_editing_task and self.task_ui_dirty:
            # Save current task in memory only (no toast/signals for task switches)
            self.logger.debug(f"Preserving changes for {self.current_editing_task} before clearing selection")
            self._save_current_task_preferences(explicit_task_name=self.current_editing_task, emit_signals=False)

        self.current_editing_task = None
        self.task_ui_dirty = False

        if not current or not current.data(Qt.ItemDataRole.UserRole):
            self.selected_task_info_label.setText("Select a task from the categories")
            self.selected_model_chunking_label.setVisible(False)
            self.selected_model_chunking_spinbox.setVisible(False)
            self.selected_model_think_label.setVisible(False)
            self.selected_model_think_combo.setVisible(False)
            self.task_model_priority_list.clear()
            return
        
        task_data = current.data(Qt.ItemDataRole.UserRole)
        task_name = task_data["task_name"]
        category = task_data["category"]

        # CRITICAL FIX: Use safe task switching to prevent contamination - Claude Generated
        self._safe_task_switch(task_name)

        self.selected_task_info_label.setText(f"Task: {task_name} ({category})")
        self.logger.info(f"Now editing task: {task_name}")

        # Hide per-entry chunking and thinking until an item is selected
        self.selected_model_chunking_label.setVisible(False)
        self.selected_model_chunking_spinbox.setVisible(False)
        self.selected_model_think_label.setVisible(False)
        self.selected_model_think_combo.setVisible(False)

        # Load current model priorities for this task
        self._load_task_specific_model_priorities(task_name)
    
    def _load_task_specific_model_priorities(self, task_name: str):
        """Load model priorities for the selected task using detection service - Claude Generated"""
        # CRITICAL FIX: Clear UI state and reset dirty flag when loading new task - Claude Generated
        self.task_model_priority_list.clear()
        self.task_ui_dirty = False  # Loading fresh data, UI is now clean

        try:
            # Get model priority for this task from working copy (not disk) - Claude Generated
            if task_name in self.unified_config.task_preferences:
                # Task has specific preferences - validate and use them
                task_pref_data = self.unified_config.task_preferences[task_name]
                raw_model_priority = task_pref_data.model_priority if task_pref_data else []
                model_priority = self._validate_and_filter_model_priority(raw_model_priority)
            else:
                # Task has no specific preferences - create intelligent defaults from global provider preferences
                model_priority = self._create_task_defaults_from_global_preferences()
            
            # Populate main priority list using cached provider/model data - Claude Generated
            available_providers = self._cached_providers

            for model_config in model_priority:
                provider_name = model_config["provider_name"]
                model_name = model_config["model_name"]

                # Validate provider is available
                if provider_name in available_providers:
                    available_models = self._cached_models.get(provider_name, [])

                    # Use "Auto-select" if model is "default" or not available
                    display_model = model_name
                    if model_name == "default" or (available_models and model_name not in available_models):
                        display_model = "(Auto-select)"

                    # Include chunking info in display text - Claude Generated
                    chunking_val = self.unified_config.get_chunking_threshold(provider_name, model_name)
                    chunking_suffix = f" [{chunking_val} Kw]" if chunking_val else " [Auto]"
                    # Include think indicator if set - Claude Generated
                    think_val = model_config.get("think")
                    think_suffix = " [think=on]" if think_val is True else (" [think=off]" if think_val is False else "")
                    item_text = f"{provider_name}: {display_model}{chunking_suffix}{think_suffix}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, model_config)
                    self.task_model_priority_list.addItem(item)


        except Exception as e:
            self.logger.error(f"Error loading task-specific model priorities: {e}")
            QMessageBox.warning(self, "Load Error", f"Could not load model priorities for task '{task_name}':\n{str(e)}")
    
    def _on_priority_item_selected(self, current, previous):
        """Handle selection change in task_model_priority_list - show per-entry chunking/think - Claude Generated"""
        if not current or not current.data(Qt.ItemDataRole.UserRole):
            self.selected_model_chunking_label.setVisible(False)
            self.selected_model_chunking_spinbox.setVisible(False)
            self.selected_model_think_label.setVisible(False)
            self.selected_model_think_combo.setVisible(False)
            return

        model_config = current.data(Qt.ItemDataRole.UserRole)
        provider_name = model_config.get("provider_name", "")
        model_name = model_config.get("model_name", "")

        # Show chunking controls only for keywords task - Claude Generated
        is_keywords_task = self.current_editing_task == "keywords"
        self.selected_model_chunking_label.setVisible(is_keywords_task)
        self.selected_model_chunking_spinbox.setVisible(is_keywords_task)

        if is_keywords_task:
            # Load current chunking value (block signals to avoid triggering save)
            self.selected_model_chunking_spinbox.blockSignals(True)
            threshold = self.unified_config.get_chunking_threshold(provider_name, model_name)
            self.selected_model_chunking_spinbox.setValue(threshold if threshold else 0)
            self.selected_model_chunking_spinbox.blockSignals(False)

        # Show think controls for all tasks - Claude Generated
        self.selected_model_think_label.setVisible(True)
        self.selected_model_think_combo.setVisible(True)

        # Load current think value (block signals to avoid triggering save)
        self.selected_model_think_combo.blockSignals(True)
        think_val = model_config.get("think")
        if think_val is None:
            self.selected_model_think_combo.setCurrentIndex(0)
        elif think_val:
            self.selected_model_think_combo.setCurrentIndex(1)
        else:
            self.selected_model_think_combo.setCurrentIndex(2)
        self.selected_model_think_combo.blockSignals(False)

    def _on_selected_model_chunking_changed(self, value: int):
        """Handle per-entry chunking spinbox change - save to unified config - Claude Generated"""
        current_item = self.task_model_priority_list.currentItem()
        if not current_item or not current_item.data(Qt.ItemDataRole.UserRole):
            return

        model_config = current_item.data(Qt.ItemDataRole.UserRole)
        provider_name = model_config.get("provider_name", "")
        model_name = model_config.get("model_name", "")

        if value == 0:
            self.unified_config.remove_chunking_threshold(provider_name, model_name)
        else:
            self.unified_config.set_chunking_threshold(provider_name, model_name, value)

        # Update display text of selected item
        self._refresh_priority_item_text(current_item)

        # Emit config changed to trigger save
        self.config_changed.emit()
        self._show_save_toast(f"✅ Chunking for {provider_name}:{model_name} saved")

    def _refresh_priority_item_text(self, item: QListWidgetItem):
        """Refresh the display text of a priority list item from its stored model_config - Claude Generated"""
        model_config = item.data(Qt.ItemDataRole.UserRole)
        if not model_config:
            return
        provider_name = model_config.get("provider_name", "")
        model_name = model_config.get("model_name", "")
        available_models = self._cached_models.get(provider_name, [])
        display_model = model_name
        if model_name == "default" or (available_models and model_name not in available_models):
            display_model = "(Auto-select)"
        chunking_val = self.unified_config.get_chunking_threshold(provider_name, model_name)
        chunking_suffix = f" [{chunking_val} Kw]" if chunking_val else " [Auto]"
        think_val = model_config.get("think")
        think_suffix = " [think=on]" if think_val is True else (" [think=off]" if think_val is False else "")
        item.setText(f"{provider_name}: {display_model}{chunking_suffix}{think_suffix}")

    def _on_selected_model_think_changed(self, index: int):
        """Handle think combo change - save to item data and config - Claude Generated"""
        current_item = self.task_model_priority_list.currentItem()
        if not current_item or not current_item.data(Qt.ItemDataRole.UserRole):
            return

        model_config = current_item.data(Qt.ItemDataRole.UserRole)
        think_map = {0: None, 1: True, 2: False}
        model_config["think"] = think_map[index]
        current_item.setData(Qt.ItemDataRole.UserRole, model_config)
        self._refresh_priority_item_text(current_item)

        if self.current_editing_task:
            self.task_ui_dirty = True
            self._save_current_task_preferences(explicit_task_name=self.current_editing_task)

    def _add_model_to_task_priority(self):
        """Add model to task priority list using real provider/model detection - Claude Generated"""
        current_item = self.task_categories_list.currentItem()
        if not current_item or not current_item.data(Qt.ItemDataRole.UserRole):
            QMessageBox.information(self, "No Task Selected", "Please select a task first.")
            return

        # CRITICAL FIX: Lock the currently selected task to prevent cross-contamination - Claude Generated
        task_data = current_item.data(Qt.ItemDataRole.UserRole)
        selected_task_name = task_data["task_name"]
        selected_task_display = task_data.get("display_name", selected_task_name.replace('_', ' ').title())

        # CRITICAL FIX: Validate that we're adding to the correct task - Claude Generated
        if self.current_editing_task != selected_task_name:
            self.logger.error(f"Task mismatch: current_editing_task='{self.current_editing_task}' vs selected='{selected_task_name}'")
            QMessageBox.warning(
                self, "Task Mismatch Error",
                f"Internal error: Cannot add model to '{selected_task_name}' while editing '{self.current_editing_task}'. "
                f"Please re-select the task and try again."
            )
            return

        # Extract current model from task preferences - Claude Generated
        current_provider, current_model = self._get_current_model_for_task(selected_task_name)
        current_model_info = {
            'provider': current_provider,
            'model': current_model
        } if current_provider else None

        # Create enhanced model selection dialog with task-specific context - Claude Generated
        dialog = TaskModelSelectionDialog(
            config_manager=self.config_manager,
            task_name=selected_task_name,
            current_model_info=current_model_info,
            parent=self
        )

        # ENHANCEMENT: Set window title to show which task is being modified - Claude Generated
        dialog.setWindowTitle(f"Add Model for Task: {selected_task_display}")

        # Add task-specific information to the dialog - Claude Generated
        if hasattr(dialog, 'layout'):
            # Insert task info label at the top of the dialog
            task_info_label = QLabel(f"🎯 Adding model preference for: <b>{selected_task_display}</b> ({selected_task_name})")
            task_info_label.setStyleSheet("background-color: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 10px;")
            task_info_label.setWordWrap(True)
            dialog.layout().insertWidget(0, task_info_label)

        # TODO: Update ProviderSelectorDialog to use UnifiedProviderConfig
        # dialog.set_default_from_global_preferences(self.unified_config)

        # CRITICAL FIX: Verify the task selection hasn't changed during dialog - Claude Generated
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Double-check that the task selection is still valid
            current_item_verify = self.task_categories_list.currentItem()
            if (not current_item_verify or
                not current_item_verify.data(Qt.ItemDataRole.UserRole) or
                current_item_verify.data(Qt.ItemDataRole.UserRole).get("task_name") != selected_task_name):

                QMessageBox.warning(
                    self, "Task Selection Changed",
                    f"Task selection changed during model selection. "
                    f"Please select '{selected_task_display}' again and retry."
                )
                return

            provider_name, model_name = dialog.get_selected_model()
            if provider_name and model_name:
                model_config = {"provider_name": provider_name, "model_name": model_name}

                # Display model name or "Auto-select" for default
                display_model = "(Auto-select)" if model_name == "default" else model_name
                item_text = f"{provider_name}: {display_model}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, model_config)

                # Log which task is being modified for debugging - Claude Generated
                self.logger.info(f"Adding model preference to task '{selected_task_name}': {provider_name}/{model_name}")
                
                self.task_model_priority_list.addItem(item)
                
                # CRITICAL FIX: Mark UI as dirty and save immediately using explicit task name - Claude Generated
                self.task_ui_dirty = True
                self._save_current_task_preferences(explicit_task_name=selected_task_name)

    def _bulk_add_model_to_tasks(self):
        """Add model to all text-based tasks (excludes vision) - Claude Generated"""
        # Show model selection dialog
        dialog = TaskModelSelectionDialog(
            config_manager=self.config_manager,
            task_name="bulk",  # Generic task indicator
            current_model_info=None,
            parent=self
        )
        dialog.setWindowTitle("Add Model to All Text Tasks")

        if dialog.exec() == QDialog.DialogCode.Accepted:
            provider_name, model_name = dialog.get_selected_model()
            if not provider_name or not model_name:
                return

            # Define text-based tasks (exclude vision)
            text_tasks = [
                "initialisation",
                "keywords",
                "rephrase",
                "dk_classification",
                "rvk_classification",
                "ddc_classification"
            ]

            # Confirm bulk operation
            reply = QMessageBox.question(
                self,
                "Confirm Bulk Add",
                f"Add model '{provider_name}/{model_name}' to {len(text_tasks)} text-based tasks?\n\n"
                f"Tasks: {', '.join(text_tasks)}\n\n"
                f"This will append to existing preferences for each task.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Save current task to restore later
            original_task = self.current_editing_task

            # Add model to each text task
            added_count = 0
            for task_name in text_tasks:
                try:
                    # Find task in list
                    task_item = None
                    for i in range(self.task_categories_list.count()):
                        item = self.task_categories_list.item(i)
                        if item.data(Qt.ItemDataRole.UserRole):
                            item_task_name = item.data(Qt.ItemDataRole.UserRole).get("task_name")
                            if item_task_name == task_name:
                                task_item = item
                                break

                    if not task_item:
                        self.logger.warning(f"Task '{task_name}' not found in task list")
                        continue

                    # Select the task (loads its preferences)
                    self.task_categories_list.setCurrentItem(task_item)

                    # Create model item
                    display_model = "(Auto-select)" if model_name == "default" else model_name
                    item_text = f"{provider_name}: {display_model}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, {
                        "provider_name": provider_name,
                        "model_name": model_name
                    })

                    # Add to priority list
                    self.task_model_priority_list.addItem(item)

                    # Mark dirty and save
                    self.task_ui_dirty = True
                    self._save_current_task_preferences(explicit_task_name=task_name)

                    added_count += 1
                    self.logger.info(f"Bulk add: Added {provider_name}/{model_name} to {task_name}")

                except Exception as e:
                    self.logger.error(f"Failed to add model to task '{task_name}': {e}")
                    continue

            # Restore original task selection
            if original_task:
                for i in range(self.task_categories_list.count()):
                    item = self.task_categories_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole):
                        if item.data(Qt.ItemDataRole.UserRole).get("task_name") == original_task:
                            self.task_categories_list.setCurrentItem(item)
                            break

            # Show result
            QMessageBox.information(
                self,
                "Bulk Add Complete",
                f"Successfully added model to {added_count}/{len(text_tasks)} tasks."
            )

    def _remove_model_from_task_priority(self):
        """Remove selected model from task priority list - Claude Generated"""
        current_item = self.task_model_priority_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a model to remove.")
            return
        row = self.task_model_priority_list.row(current_item)
        self.task_model_priority_list.takeItem(row)
        # CRITICAL FIX: Mark UI as dirty and save with explicit task name - Claude Generated
        if self.current_editing_task:
            self.task_ui_dirty = True
            self._save_current_task_preferences(explicit_task_name=self.current_editing_task)
        else:
            self.logger.warning("No current editing task for remove operation")

    def _reset_selected_task_to_defaults(self):
        """Reset selected task to default model priorities - Claude Generated"""
        current_item = self.task_categories_list.currentItem()
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
                # Remove task from config.unified_config.task_preferences (will fall back to defaults) - Claude Generated
                if task_name in self.config.unified_config.task_preferences:
                    del self.config.unified_config.task_preferences[task_name]

                    # Configuration changed in memory - parent dialog will handle save
                    self.config_changed.emit()

                    # CRITICAL FIX: Update current editing task state after reset - Claude Generated
                    if self.current_editing_task == task_name:
                        self.task_ui_dirty = False  # UI will be reloaded with defaults
                
                # Reload priorities
                self._load_task_specific_model_priorities(task_name)
                QMessageBox.information(self, "Reset Complete", f"Task '{task_name}' reset to default priorities.")
                
            except Exception as e:
                self.logger.error(f"Error resetting task: {e}")
                QMessageBox.critical(self, "Reset Error", f"Could not reset task '{task_name}':\n{str(e)}")
    
    
    def _add_provider(self):
        """Add a new provider - Claude Generated"""
        dialog = ProviderEditDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_provider = dialog.get_provider_data()
            
            # Check for duplicate names
            existing_names = [p.name for p in self.unified_config.providers]
            if new_provider.name in existing_names:
                QMessageBox.warning(self, "Duplicate Name", 
                                  f"A provider with name '{new_provider.name}' already exists.")
                return
            
            # Add to configuration and persist immediately so reload picks it up
            self.unified_config.providers.append(new_provider)
            self.config_manager.save_config(self.config)
            self._load_configuration()
            self.config_changed.emit()
            
            self.logger.info(f"Added new provider: {new_provider.name} ({new_provider.provider_type})")
    
    def _edit_provider(self):
        """Edit selected provider - Claude Generated"""
        current_row = self.provider_table.currentRow()
        if current_row < 0:
            QMessageBox.information(self, "No Selection", "Please select a provider to edit.")
            return
        
        provider = self.unified_config.providers[current_row]
        dialog = ProviderEditDialog(provider, parent=self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated_provider = dialog.get_provider_data()
            self.unified_config.providers[current_row] = updated_provider
            self.config_manager.save_config(self.config)
            self._load_configuration()
            self.config_changed.emit()

            self.logger.info(f"Updated provider: {updated_provider.name}")
    
    def _remove_provider(self):
        """Remove selected provider - Claude Generated"""
        current_row = self.provider_table.currentRow()
        if current_row < 0:
            QMessageBox.information(self, "No Selection", "Please select a provider to remove.")
            return
        
        provider = self.unified_config.providers[current_row]
        
        reply = QMessageBox.question(
            self, "Confirm Removal",
            f"Are you sure you want to remove provider '{provider.name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.unified_config.providers[current_row]
            self.config_manager.save_config(self.config)
            self._load_configuration()
            self.config_changed.emit()

            self.logger.info(f"Removed provider: {provider.name}")
    
    def _refresh_models(self):
        """Trigger background provider refresh via ProviderStatusService - Claude Generated"""
        if (not self.alima_manager or
            not hasattr(self.alima_manager, 'provider_status_service') or
            not self.alima_manager.provider_status_service):
            QMessageBox.warning(
                self, "Service Unavailable",
                "ProviderStatusService not available. Cannot refresh provider models."
            )
            return

        try:
            # Trigger force refresh in background (non-blocking)
            # Signal handlers (_on_provider_tested) will update _cached_models when results arrive
            self.alima_manager.provider_status_service.refresh_all(force=True)

            # Show immediate feedback
            QMessageBox.information(
                self, "Refresh Started",
                "Provider model refresh started in background.\n\n"
                "The table will update automatically as providers are tested.\n"
                "This may take a few moments for all providers."
            )

            self.logger.info("Triggered background provider refresh via ProviderStatusService")

        except Exception as e:
            self.logger.error(f"Error starting provider refresh: {e}")
            QMessageBox.critical(self, "Refresh Error", f"Failed to start provider refresh:\n\n{str(e)}")
    
    def _test_all_providers(self):
        """Test connections to all enabled providers via ProviderStatusService - Claude Generated"""
        # Delegate to _refresh_models which already uses ProviderStatusService
        self._refresh_models()
    
    def _show_save_toast(self, message: str, duration: int = 2000, error: bool = False):
        """Show toast notification via main window's global status bar - Claude Generated"""
        try:
            # Navigate up the parent hierarchy to find the main window with global_status_bar
            current_widget = self
            main_window = None
            
            # Search up to 5 levels for the main window
            for _ in range(5):
                if current_widget is None:
                    break
                    
                # Check if this widget has a global_status_bar attribute (main window)
                if hasattr(current_widget, 'global_status_bar'):
                    main_window = current_widget
                    break
                    
                current_widget = current_widget.parent()
            
            # Show the toast notification
            if main_window and hasattr(main_window, 'global_status_bar'):
                main_window.global_status_bar.show_temporary_message(message, duration)
            else:
                # Fallback: Log the message if we can't find the global status bar
                log_level = "error" if error else "info"
                getattr(self.logger, log_level)(f"Toast notification (no status bar found): {message}")
                
        except Exception as e:
            self.logger.error(f"Error showing toast notification: {e}")

    def _setup_provider_status_connections(self):
        """Setup signal connections to ProviderStatusService for reactive updates - Claude Generated"""
        if (self.alima_manager and
            hasattr(self.alima_manager, 'provider_status_service') and
            self.alima_manager.provider_status_service):

            # Connect to status updates for automatic table refresh + model preferences - Claude Generated
            self.alima_manager.provider_status_service.status_updated.connect(
                self._populate_provider_table
            )
            self.alima_manager.provider_status_service.status_updated.connect(
                self._populate_model_preferences
            )

            # Connect to individual provider tests for live updates
            self.alima_manager.provider_status_service.provider_tested.connect(
                self._on_provider_tested
            )

            self.logger.info("Connected to ProviderStatusService signals for reactive updates")
        else:
            self.logger.warning("ProviderStatusService not available for signal connections")

    def _on_provider_tested(self, provider_name: str, provider_info: dict):
        """Handle individual provider test completion - Claude Generated"""
        try:
            # Update local model cache from fresh provider_info so dropdowns stay current - Claude Generated
            fresh_models = provider_info.get('models', [])
            if fresh_models:
                self._cached_models[provider_name] = fresh_models

            # Refresh provider table and model-preferences dropdowns
            self._populate_provider_table()
            self._populate_model_preferences()

            # Log the update
            status = provider_info.get('status', 'unknown')
            model_count = len(fresh_models) if fresh_models else provider_info.get('model_count', 0)
            self.logger.debug(f"Provider {provider_name} tested: {status} ({model_count} models)")

        except Exception as e:
            self.logger.error(f"Error handling provider test result for {provider_name}: {e}")
    
    def _update_config_from_ui(self):
        """Update configuration object from UI state - Claude Generated"""
        # Update global preferences
        self.unified_config.preferred_provider = self.preferred_provider_combo.currentText()

        # CRITICAL FIX: Only save task preferences if we have an explicit task and no UI conflicts - Claude Generated
        if self.current_editing_task and not self.task_ui_dirty:
            # Only save if UI state is clean to prevent contamination
            self.logger.debug(f"Global save: saving clean task preferences for {self.current_editing_task}")
            self._save_current_task_preferences(explicit_task_name=self.current_editing_task)
        elif self.current_editing_task and self.task_ui_dirty:
            self.logger.warning(f"Global save: skipping task preferences save for {self.current_editing_task} due to dirty UI state")

        # NOTE: Task preferences are managed in self.config.unified_config.task_preferences
        # They are already updated by individual UI operations
        # No additional UI → config sync needed here
    
    def _save_current_task_preferences(self, explicit_task_name: str = None, emit_signals: bool = True):
        """Save current task priority lists to ProviderPreferences - Claude Generated"""
        try:
            # CRITICAL FIX: Always use explicit task name or current_editing_task for isolation - Claude Generated
            if explicit_task_name:
                task_name = explicit_task_name
                self.logger.info(f"Using explicit task name for save: '{task_name}'")
            elif self.current_editing_task:
                task_name = self.current_editing_task
                self.logger.info(f"Using current_editing_task for save: '{task_name}'")
            else:
                # Fallback to UI selection, but warn about potential contamination
                current_item = self.task_categories_list.currentItem()
                if not current_item:
                    self.logger.warning("No task selected and no explicit task name provided - skipping save")
                    return

                task_data = current_item.data(Qt.ItemDataRole.UserRole)
                if not task_data:
                    self.logger.warning("No task data available - skipping save")
                    return
                task_name = task_data["task_name"]
                self.logger.warning(f"Fallback to UI selection for save: '{task_name}' - potential contamination risk!")
            
            # Extract model priorities from task_model_priority_list
            model_priority = []
            for i in range(self.task_model_priority_list.count()):
                item = self.task_model_priority_list.item(i)
                model_config = item.data(Qt.ItemDataRole.UserRole)
                if model_config:
                    model_priority.append(model_config)
            
            # Chunked model priority removed from UI - kept as None for backward compat
            chunked_model_priority = None

            # CRITICAL FIX: Add validation before saving to prevent cross-contamination - Claude Generated
            if explicit_task_name and explicit_task_name != task_name:
                self.logger.error(f"Task name mismatch: explicit='{explicit_task_name}' vs resolved='{task_name}' - aborting save")
                self._show_save_toast(f"❌ Save aborted: task name mismatch", error=True)
                return

            # Get existing TaskPreference to preserve settings
            existing_pref = self.config.unified_config.task_preferences.get(task_name)

            # Determine task type - try mapping, fallback to extracting from enum values
            if task_name in self.TASK_TYPE_MAPPING:
                task_type = self.TASK_TYPE_MAPPING[task_name]
            else:
                # Try to find matching task by enum value
                try:
                    task_type = UnifiedTaskType(task_name)
                except ValueError:
                    self.logger.warning(f"Unknown task name '{task_name}', using INITIALISATION as fallback")
                    task_type = UnifiedTaskType.INITIALISATION

            # Chunking is now per-model via unified_config.model_chunking_thresholds
            # Preserve existing task-level chunking_threshold for backward compat - Claude Generated
            chunking_threshold = existing_pref.chunking_threshold if existing_pref else None

            # Create proper TaskPreference object
            task_preference = TaskPreference(
                task_type=task_type,
                model_priority=model_priority,
                chunked_model_priority=chunked_model_priority,
                allow_fallback=existing_pref.allow_fallback if existing_pref else True,
                chunking_threshold=chunking_threshold,
            )

            # Save proper TaskPreference object
            self.config.unified_config.task_preferences[task_name] = task_preference

            # Mark UI as clean after successful change - Claude Generated
            self.task_ui_dirty = False

            # Conditionally emit signals and show toast (only for explicit user saves, not task switches)
            if emit_signals:
                # Configuration changed in memory - parent dialog will handle save
                self.config_changed.emit()

                # Show toast notification for task preference save - Claude Generated
                self._show_save_toast(f"✅ {task_name} preferences saved")

                # Emit signal to notify other components about task preference changes - Claude Generated
                self.task_preferences_changed.emit()

            self.logger.info(f"Task preferences {'saved' if emit_signals else 'updated in memory'} for '{task_name}': {len(model_priority)} models, chunked: {chunked_model_priority is not None}")
            
        except Exception as e:
            self._show_save_toast(f"❌ Save failed: {str(e)[:30]}", error=True)
            self.logger.warning(f"Failed to save current task preferences: {e}")

    def _initialize_task_editing_state(self):
        """Initialize the task editing state to prevent contamination - Claude Generated"""
        try:
            # Clear any initial task selection state
            self.current_editing_task = None
            self.task_ui_dirty = False

            # Ensure task list is clear initially
            if hasattr(self, 'task_model_priority_list'):
                self.task_model_priority_list.clear()

            self.logger.info("Task editing state initialized successfully")

        except Exception as e:
            self.logger.warning(f"Failed to initialize task editing state: {e}")

    def _safe_task_switch(self, new_task_name: str):
        """Safely switch to a new task with proper cleanup - Claude Generated"""
        try:
            # Save current task if it exists and has changes
            if self.current_editing_task and self.task_ui_dirty:
                self.logger.info(f"Auto-saving changes for {self.current_editing_task} before switching to {new_task_name}")
                self._save_current_task_preferences(explicit_task_name=self.current_editing_task)

            # Clear UI state
            if hasattr(self, 'task_model_priority_list'):
                self.task_model_priority_list.clear()

            # Update tracking variables
            self.current_editing_task = new_task_name
            self.task_ui_dirty = False

            self.logger.info(f"Successfully switched to task: {new_task_name}")

        except Exception as e:
            self.logger.error(f"Error during safe task switch to {new_task_name}: {e}")
            # Reset to safe state
            self.current_editing_task = None
            self.task_ui_dirty = False
    
    def _on_priority_list_reordered(self):
        """Handle priority list reordering via drag & drop - Claude Generated"""
        if self.current_editing_task:
            self.logger.info(f"Priority list reordered for task: {self.current_editing_task}")
            # Mark UI as dirty and save immediately
            self.task_ui_dirty = True
            self._save_current_task_preferences(explicit_task_name=self.current_editing_task)
            # Show visual feedback for the save
            self._show_save_toast(f"🔄 {self.current_editing_task} priority updated")
        else:
            self.logger.warning("Priority list reordered but no current editing task")

    def _create_task_defaults_from_global_preferences(self) -> List[Dict[str, str]]:
        """Create intelligent task default priorities from global provider preferences - Claude Generated"""
        try:
            # Start with provider_priority order
            model_priority = []
            
            for provider in self.unified_config.provider_priority:
                # Skip disabled providers
                if provider in self.unified_config.disabled_providers:
                    continue
                
                # Get preferred model for this provider (if configured)
                # TODO: Implement preferred_models in UnifiedProviderConfig
                preferred_model = "auto"  # Disabled until proper implementation
                
                # Add to model priority
                model_priority.append({
                    "provider_name": provider,
                    "model_name": preferred_model
                })
            
            self.logger.debug(f"Created task defaults from global preferences: {len(model_priority)} providers")
            return model_priority
            
        except Exception as e:
            self.logger.warning(f"Failed to create task defaults from global preferences: {e}")
            # Fallback to basic provider priority
            return [{"provider_name": p, "model_name": "auto"} for p in self.unified_config.provider_priority]
    
    def _validate_and_filter_model_priority(self, model_priority: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate model_priority entries against available models and filter invalid ones - Claude Generated"""
        try:
            available_providers = self._cached_providers

            validated_priority = []
            removed_entries = []

            for entry in model_priority:
                provider_name = entry.get("provider_name")
                model_name = entry.get("model_name")

                # Skip entries with missing data
                if not provider_name or not model_name:
                    removed_entries.append(f"Incomplete entry: {entry}")
                    continue

                # Provider offline? Still show it (user should see configured preferences)
                if provider_name not in available_providers:
                    self.logger.debug(f"Provider {provider_name} offline, but keeping preference visible")
                    validated_priority.append(entry)
                    continue

                # Special case: "auto" is always valid
                if model_name == "auto" or model_name == "default":
                    validated_priority.append(entry)
                    continue

                # Check if model is available for this provider (from cache)
                try:
                    available_models = self._cached_models.get(provider_name, [])
                    if available_models and model_name in available_models:
                        validated_priority.append(entry)
                    else:
                        removed_entries.append(f"{provider_name}/{model_name} (model not found)")
                except Exception as e:
                    # Provider detection failed, keep entry but mark as unverified
                    validated_priority.append(entry)
                    self.logger.warning(f"Could not verify model {provider_name}/{model_name}: {e}")
            
            # Log removed entries for user awareness
            if removed_entries:
                self.logger.info(f"Filtered out invalid task preference entries: {removed_entries}")
            
            return validated_priority
            
        except Exception as e:
            self.logger.warning(f"Model validation failed, using raw priority: {e}")
            return model_priority  # Return original if validation fails
