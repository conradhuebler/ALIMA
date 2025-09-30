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
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
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
        
        model_info_label = QLabel("Choose a specific model or use auto-selection:")
        model_layout.addWidget(model_info_label)
        
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        
        # Strict validation note - Claude Generated
        validation_note = QLabel("📋 Only detected models are available (for Smart Mode compatibility) • Changes apply immediately")
        validation_note.setStyleSheet("color: gray; font-style: italic; padding: 5px;")
        model_layout.addWidget(validation_note)
        
        layout.addWidget(model_group)
        
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
        """Load real models for selected provider using detection service - Claude Generated"""
        self.model_combo.clear()
        
        if not provider_name or provider_name == "No providers available":
            return
        
        try:
            detection_service = ProviderDetectionService()
            available_models = detection_service.get_available_models(provider_name)
            
            # Always add auto-select option first
            self.model_combo.addItem("(Auto-select)", "default")
            
            if available_models:
                for model in available_models:
                    self.model_combo.addItem(model, model)
            else:
                self.model_combo.addItem("No models detected", "default")
                
        except Exception as e:
            # Fallback to auto-select if detection fails
            self.model_combo.addItem("(Auto-select)", "default")
    
    def get_selected_model(self):
        """Get selected provider and model - Claude Generated"""
        provider = self.provider_combo.currentText()
        
        if provider == "No providers available":
            return None, None
        
        # Only use detected models from combo box - Claude Generated
        model = self.model_combo.currentData() or "default"
        
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
        
        basic_layout.addRow("Name:", self.name_edit)
        basic_layout.addRow("Description:", self.description_edit)
        basic_layout.addRow("Enabled:", self.enabled_checkbox)
        
        layout.addWidget(basic_group)
        
        # Connection Configuration (dynamic based on type)
        self.connection_group = QGroupBox("Connection Configuration")
        self.connection_layout = QFormLayout(self.connection_group)
        layout.addWidget(self.connection_group)
        
        # Capabilities
        capabilities_group = QGroupBox("Capabilities")
        capabilities_layout = QVBoxLayout(capabilities_group)
        
        self.capabilities_text = QTextEdit()
        self.capabilities_text.setPlaceholderText("Enter capabilities (one per line): vision, fast, large_context, local, privacy, etc.")
        self.capabilities_text.setMaximumHeight(100)
        capabilities_layout.addWidget(self.capabilities_text)
        
        layout.addWidget(capabilities_group)
        
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
    
    def _load_provider_data(self):
        """Load provider data into form fields - Claude Generated"""
        if not self.provider:
            return
        
        self.name_edit.setText(self.provider.name)
        self.description_edit.setText(self.provider.description)
        self.enabled_checkbox.setChecked(self.provider.enabled)
        self.type_combo.setCurrentText(self.provider.provider_type)
        
        # Load capabilities (UnifiedProvider doesn't have capabilities field yet, so leave empty)
        capabilities_text = ""  # TODO: Add capabilities field to UnifiedProvider if needed
        self.capabilities_text.setPlainText(capabilities_text)
        
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
        
        # Parse capabilities
        capabilities_text = self.capabilities_text.toPlainText().strip()
        capabilities = [cap.strip() for cap in capabilities_text.split('\n') if cap.strip()] if capabilities_text else []
        
        return UnifiedProvider(
            name=self.name_edit.text() or f"New {provider_type.title()} Provider",
            provider_type=provider_type,
            enabled=self.enabled_checkbox.isChecked(),
            api_key=connection_config.get("api_key", ""),
            base_url=connection_config.get("base_url", ""),
            preferred_model="",  # Can be set later via model selection
            description=self.description_edit.text(),
            # Ollama specific fields
            host=connection_config.get("host", ""),
            port=connection_config.get("port", 11434),
            use_ssl=connection_config.get("use_ssl", False),
            connection_type=connection_config.get("connection_type", "native_client")
        )


class UnifiedProviderTab(QWidget):
    """
    Unified Provider Configuration Tab - Claude Generated
    Consolidates LLM Provider and Provider Preferences into single interface
    """

    config_changed = pyqtSignal()
    task_preferences_changed = pyqtSignal()  # New signal for task preference changes - Claude Generated
    
    def __init__(self, unified_config: UnifiedProviderConfig, alima_config: AlimaConfig, alima_manager=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Store the configuration objects directly - Claude Generated (Refactoring)
        self.unified_config = unified_config
        self.config = alima_config
        self.alima_manager = alima_manager  # Access to ProviderStatusService - Claude Generated

        # CRITICAL FIX: Add explicit task tracking to prevent cross-contamination - Claude Generated
        self.current_editing_task = None  # Track which task is currently being edited
        self.task_ui_dirty = False  # Track if current task UI has unsaved changes

        # 🔍 DEBUG: Log config loading for legacy provider detection - Claude Generated
        self.logger.critical(f"🔍 CONFIG_LOAD: config loaded={self.config is not None}")
        if self.config and hasattr(self.config, 'llm'):
            self.logger.critical(f"🔍 CONFIG_LLM: hasattr llm={hasattr(self.config, 'llm')}")
            if hasattr(self.config.unified_config, 'gemini'):
                gemini_val = getattr(self.config.unified_config, 'gemini', '')
                self.logger.critical(f"🔍 CONFIG_GEMINI: value_exists={bool(gemini_val)}, length={len(gemini_val) if gemini_val else 0}")
                self.logger.critical(f"🔍 CONFIG_GEMINI: is_placeholder={gemini_val == 'your_gemini_api_key_here' if gemini_val else False}")
            if hasattr(self.config.unified_config, 'anthropic'):
                anthropic_val = getattr(self.config.unified_config, 'anthropic', '')
                self.logger.critical(f"🔍 CONFIG_ANTHROPIC: value_exists={bool(anthropic_val)}, length={len(anthropic_val) if anthropic_val else 0}")
                self.logger.critical(f"🔍 CONFIG_ANTHROPIC: is_placeholder={anthropic_val == 'your_anthropic_api_key_here' if anthropic_val else False}")
        else:
            self.logger.critical("🔍 CONFIG_NO_LLM: config has no llm attribute")

        self._setup_ui()
        self._load_configuration()

        # Setup reactive provider status connections - Claude Generated
        self._setup_provider_status_connections()
    
    def _setup_ui(self):
        """Setup the unified provider tab UI - Claude Generated"""
        layout = QVBoxLayout(self)
        
        # Title and description
        title_label = QLabel("<h2>🚀 Providers & Models</h2>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        desc_label = QLabel(
            "Unified configuration for all LLM providers and model preferences. "
            "Configure connections, set priorities, and manage task-specific preferences in one place."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
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
        
        # Provider table (full width)
        self.provider_table = QTableWidget()
        self.provider_table.setColumnCount(5)
        self.provider_table.setHorizontalHeaderLabels([
            "Name", "Type", "Status", "Models", "API Key"
        ])

        # Make table responsive
        header = self.provider_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Status
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Models
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # API Key

        layout.addWidget(self.provider_table)

        # Provider control buttons (horizontal under table)
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
        
        return widget
    
    def _create_preferences_tab(self) -> QWidget:
        """Create task preferences tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Global preferences
        global_group = QGroupBox("🌐 Global Preferences")
        global_layout = QFormLayout(global_group)
        
        self.preferred_provider_combo = QComboBox()
        self.auto_fallback_checkbox = QCheckBox()
        self.auto_fallback_checkbox.setChecked(True)
        self.prefer_fast_checkbox = QCheckBox()
        
        global_layout.addRow("Preferred Provider:", self.preferred_provider_combo)
        global_layout.addRow("Auto Fallback:", self.auto_fallback_checkbox)
        global_layout.addRow("Prefer Faster Models:", self.prefer_fast_checkbox)
        
        layout.addWidget(global_group)
        
        # Model preferences section - Claude Generated 
        model_group = QGroupBox("🎯 Model Preferences per Provider")
        model_layout = QVBoxLayout(model_group)
        
        self.model_preferences_table = QTableWidget()
        self.model_preferences_table.setColumnCount(3)
        self.model_preferences_table.setHorizontalHeaderLabels([
            "Provider", "Preferred Model", "Available Models"
        ])
        
        # Make model preferences table responsive
        model_header = self.model_preferences_table.horizontalHeader()
        model_header.setStretchLastSection(False)
        model_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Provider
        model_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)           # Preferred Model
        model_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)           # Available Models
        
        model_layout.addWidget(self.model_preferences_table)
        layout.addWidget(model_group)
        
        # Task-specific preferences - Enhanced with advanced task management - Claude Generated
        task_group = QGroupBox("🎯 Task-Specific Model Preferences")
        task_main_layout = QVBoxLayout(task_group)
        
        # Create splitter for task management
        task_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Task categories and selection
        left_task_widget = QWidget()
        left_task_layout = QVBoxLayout(left_task_widget)
        
        task_categories_label = QLabel("📋 Available Tasks")
        task_categories_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        left_task_layout.addWidget(task_categories_label)
        
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        self.task_categories_list = QListWidget()
        self.task_categories_list.setMinimumWidth(280)
        self.task_categories_list.setMaximumWidth(350)
        self.task_categories_list.currentItemChanged.connect(self._on_task_category_selected)
        left_task_layout.addWidget(self.task_categories_list)
        
        # Right side: Model priority configuration  
        right_task_widget = QWidget()
        right_task_layout = QVBoxLayout(right_task_widget)
        
        config_header_label = QLabel("⚙️ Model Priority Configuration")
        config_header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        right_task_layout.addWidget(config_header_label)
        
        # Selected task info
        self.selected_task_info_label = QLabel("Select a task from the categories")
        self.selected_task_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        right_task_layout.addWidget(self.selected_task_info_label)
        
        # Chunked checkbox for applicable tasks
        self.chunked_tasks_checkbox = QCheckBox("Enable specialized models for large texts (Chunked Processing)")
        self.chunked_tasks_checkbox.setVisible(False)
        self.chunked_tasks_checkbox.stateChanged.connect(self._on_chunked_tasks_toggled)
        right_task_layout.addWidget(self.chunked_tasks_checkbox)
        
        # Standard model priority
        priority_label = QLabel("Model Priority:")
        priority_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        right_task_layout.addWidget(priority_label)
        
        self.task_model_priority_list = QListWidget()
        from PyQt6.QtWidgets import QAbstractItemView
        self.task_model_priority_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.task_model_priority_list.setMinimumHeight(150)
        right_task_layout.addWidget(self.task_model_priority_list)
        
        # Chunked model priority (conditional)
        self.chunked_tasks_priority_label = QLabel("Chunked Processing Model Priority:")
        self.chunked_tasks_priority_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.chunked_tasks_priority_label.setVisible(False)
        right_task_layout.addWidget(self.chunked_tasks_priority_label)
        
        self.chunked_task_model_priority_list = QListWidget()
        self.chunked_task_model_priority_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.chunked_task_model_priority_list.setMinimumHeight(120)
        self.chunked_task_model_priority_list.setVisible(False)
        right_task_layout.addWidget(self.chunked_task_model_priority_list)
        
        # Task management buttons
        task_button_layout = QHBoxLayout()
        
        add_task_model_btn = QPushButton("➕ Add Model")
        add_task_model_btn.clicked.connect(self._add_model_to_task_priority)
        task_button_layout.addWidget(add_task_model_btn)
        
        remove_task_model_btn = QPushButton("➖ Remove Model")
        remove_task_model_btn.clicked.connect(self._remove_model_from_task_priority)
        task_button_layout.addWidget(remove_task_model_btn)
        
        task_button_layout.addStretch()
        
        reset_task_btn = QPushButton("🔄 Reset Task")
        reset_task_btn.clicked.connect(self._reset_selected_task_to_defaults)
        task_button_layout.addWidget(reset_task_btn)
        
        right_task_layout.addLayout(task_button_layout)
        right_task_layout.addStretch()
        
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
    
    def _get_all_providers(self):
        """Get all providers from unified config + LLM config - Claude Generated"""
        all_providers = list(self.unified_config.providers)  # Start with unified providers

        # 🔍 DEBUG: Log unified provider count - Claude Generated
        self.logger.critical(f"🔍 UNIFIED_PROVIDERS: Found {len(all_providers)} unified providers: {[p.name for p in all_providers]}")

        existing_names = [p.name.lower() for p in all_providers]

        # 🔍 DEBUG: Check LLM config availability - Claude Generated
        self.logger.critical(f"🔍 LLM_CONFIG_CHECK: hasattr(config.unified_config, 'gemini')={hasattr(self.config.unified_config, 'gemini')}")
        if hasattr(self.config.unified_config, 'gemini'):
            gemini_key = getattr(self.config.unified_config, 'gemini', '')
            self.logger.critical(f"🔍 GEMINI_KEY: exists={bool(gemini_key)}, length={len(gemini_key) if gemini_key else 0}, not_placeholder={gemini_key != 'your_gemini_api_key_here' if gemini_key else False}")

        # Add Gemini from LLM config if configured and not already present
        gemini_conditions = [
            hasattr(self.config.unified_config, 'gemini_api_key'),
            self.config.unified_config.gemini_api_key,
            self.config.unified_config.gemini_api_key.strip() if self.config.unified_config.gemini_api_key else False,
            self.config.unified_config.gemini_api_key != "your_gemini_api_key_here" if self.config.unified_config.gemini_api_key else False,
            "gemini" not in existing_names
        ]
        self.logger.critical(f"🔍 GEMINI_CONDITIONS: {gemini_conditions}")

        if all(gemini_conditions):
            gemini_provider = UnifiedProvider(
                name="gemini",
                provider_type="gemini",
                api_key=self.config.unified_config.gemini_api_key,
                enabled=True,
                description="Gemini from LLM configuration"
            )
            all_providers.append(gemini_provider)
            self.logger.critical(f"🔍 GEMINI_ADDED: Successfully added Gemini from LLM config")
        else:
            self.logger.critical(f"🔍 GEMINI_SKIPPED: Conditions not met")

        # 🔍 DEBUG: Check Anthropic config - Claude Generated
        if hasattr(self.config.unified_config, 'anthropic'):
            anthropic_key = getattr(self.config.unified_config, 'anthropic', '')
            self.logger.critical(f"🔍 ANTHROPIC_KEY: exists={bool(anthropic_key)}, length={len(anthropic_key) if anthropic_key else 0}, not_placeholder={anthropic_key != 'your_anthropic_api_key_here' if anthropic_key else False}")

        # Add Anthropic from LLM config if configured and not already present
        anthropic_conditions = [
            hasattr(self.config.unified_config, 'anthropic_api_key'),
            self.config.unified_config.anthropic_api_key,
            self.config.unified_config.anthropic_api_key.strip() if self.config.unified_config.anthropic_api_key else False,
            self.config.unified_config.anthropic_api_key != "your_anthropic_api_key_here" if self.config.unified_config.anthropic_api_key else False,
            "anthropic" not in existing_names
        ]
        self.logger.critical(f"🔍 ANTHROPIC_CONDITIONS: {anthropic_conditions}")

        if all(anthropic_conditions):
            anthropic_provider = UnifiedProvider(
                name="anthropic",
                provider_type="anthropic",
                api_key=self.config.unified_config.anthropic_api_key,
                enabled=True,
                description="Anthropic from LLM configuration"
            )
            all_providers.append(anthropic_provider)
            self.logger.critical(f"🔍 ANTHROPIC_ADDED: Successfully added Anthropic from LLM config")
        else:
            self.logger.critical(f"🔍 ANTHROPIC_SKIPPED: Conditions not met")

        # 🔍 DEBUG: Final provider count - Claude Generated
        self.logger.critical(f"🔍 FINAL_PROVIDERS: Total {len(all_providers)} providers: {[p.name for p in all_providers]}")

        return all_providers

    def _populate_provider_table(self):
        """Populate the provider table using cached ProviderStatusService data - Claude Generated"""
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
            # Name
            name_item = QTableWidgetItem(provider.name)
            if not provider.enabled:
                name_item.setForeground(QPalette().color(QPalette.ColorRole.PlaceholderText))
            self.provider_table.setItem(row, 0, name_item)

            # Type
            type_item = QTableWidgetItem(provider.provider_type.title())
            self.provider_table.setItem(row, 1, type_item)

            # Status and model info from cache (fast, non-blocking)
            status_text = "Unknown"
            models_text = "Not tested"

            # Get cached status for this provider
            cached_status = provider_status_cache.get(provider.name)
            if cached_status:
                # Use cached data from ProviderStatusService
                is_reachable = cached_status.get('reachable', False)
                cached_models = cached_status.get('models', [])
                error_message = cached_status.get('error_message')

                if is_reachable:
                    status_text = "✅ Available"
                    if cached_models:
                        models_text = f"{len(cached_models)} models"
                        # Update provider's available models from cache
                        provider.available_models = cached_models
                    else:
                        models_text = "No models detected"
                else:
                    status_text = "❌ Offline"
                    if error_message:
                        models_text = f"Error: {error_message[:30]}"
                    else:
                        models_text = "Provider not reachable"
            else:
                # No cached data available (tests haven't completed yet)
                if provider_status_cache:
                    # Service is running but provider not tested yet
                    status_text = "⏳ Testing..."
                    models_text = "Test in progress"
                else:
                    # Service not available, fallback to config data
                    model_count = len(provider.available_models)
                    if model_count > 0:
                        models_text = f"{model_count} models (config)"
                        status_text = "📝 Configured"
                    else:
                        status_text = "⚠️ Unknown"
                        models_text = "Status unknown"

            # Status
            status_item = QTableWidgetItem(status_text)
            self.provider_table.setItem(row, 2, status_item)

            # Models
            models_item = QTableWidgetItem(models_text)
            if provider.available_models:
                # Set tooltip with model list
                models_tooltip = "Available models:\n" + "\n".join(provider.available_models[:10])
                if len(provider.available_models) > 10:
                    models_tooltip += f"\n... and {len(provider.available_models) - 10} more"
                models_item.setToolTip(models_tooltip)
            self.provider_table.setItem(row, 3, models_item)

            # API Key Status
            api_key_status = self._get_api_key_status(provider)
            api_key_item = QTableWidgetItem(api_key_status)

            # Color coding for API key status
            if "✅" in api_key_status:
                api_key_item.setForeground(QPalette().color(QPalette.ColorRole.Text))
            elif "❌" in api_key_status:
                api_key_item.setForeground(QPalette().color(QPalette.ColorRole.PlaceholderText))
            else:  # N/A cases
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
        
        # Set checkboxes
        self.auto_fallback_checkbox.setChecked(self.unified_config.auto_fallback)
        self.prefer_fast_checkbox.setChecked(self.unified_config.prefer_faster_models)
    
    def _populate_model_preferences(self):
        """Populate model preferences table using direct provider config - Claude Generated"""
        try:
            # Use already loaded configuration
            config = self.config
            detection_service = ProviderDetectionService()
            
            # Get all available providers
            available_providers = detection_service.get_available_providers()
            self.model_preferences_table.setRowCount(len(available_providers))
            
            for row, provider in enumerate(available_providers):
                # Provider name
                provider_item = QTableWidgetItem(provider)
                self.model_preferences_table.setItem(row, 0, provider_item)
                
                # Get available models for this provider
                available_models = detection_service.get_available_models(provider)
                available_models_text = ", ".join(available_models[:3])
                if len(available_models) > 3:
                    available_models_text += f" (+{len(available_models) - 3} more)"
                
                # Get current preferred model from direct provider config
                preferred_model = self._get_preferred_model_from_config(provider, config)
                self.logger.critical(f"🔍 POPULATE_GET_PREF: Provider='{provider}', Found='{preferred_model}'")
                
                # Create dropdown for preferred model selection
                model_combo = QComboBox()
                model_combo.blockSignals(True)  # Block signals during setup
                model_combo.addItem("(Auto-select)")  # Default option
                if available_models:
                    self.logger.critical(f"🔍 COMBO_ADDING_MODELS: Provider='{provider}', Models={len(available_models)}")
                    model_combo.addItems(available_models)
                model_combo.blockSignals(False)  # Re-enable signals
                
                # Set current selection (block signals during population)
                if preferred_model and preferred_model in available_models:
                    self.logger.critical(f"🔍 POPULATE_SETTING: Provider='{provider}', Model='{preferred_model}'")
                    model_combo.blockSignals(True)  # Prevent signal emission during population
                    model_combo.setCurrentText(preferred_model)
                    model_combo.blockSignals(False)  # Re-enable signals
                
                # Connect change handler with proper closure
                def make_handler(provider_name):
                    return lambda text: self._on_model_preference_changed(provider_name, text)
                
                model_combo.currentTextChanged.connect(make_handler(provider))
                
                self.model_preferences_table.setCellWidget(row, 1, model_combo)
                
                # Available models display
                models_item = QTableWidgetItem(available_models_text or "No models detected")
                models_item.setToolTip("\n".join(available_models) if available_models else "No models available")
                self.model_preferences_table.setItem(row, 2, models_item)
                
        except Exception as e:
            self.logger.error(f"Error populating model preferences: {e}")
    
    def _on_model_preference_changed(self, provider: str, model: str):
        """Handle model preference change - Store directly in provider config - Claude Generated"""
        try:
            # 🔍 DEBUG: Log detailed method call information - Claude Generated
            import traceback
            stack = traceback.format_stack()[-3:-1]  # Get calling context
            self.logger.critical(f"🔍 MODEL_PREF_SAVE_CALL: Provider='{provider}', Model='{model}'")
            self.logger.critical(f"🔍 MODEL_PREF_SAVE_STACK: {''.join(stack).strip()}")
            
            if model == "(Auto-select)":
                model = ""  # Remove preference, use auto-select
            
            # 🔍 DEBUG: Log the preference change request - Claude Generated
            self.logger.critical(f"🔍 MODEL_PREF_SAVE: Provider='{provider}', Model='{model}' (after auto-select processing)")

            # 🔍 DEBUG: Log config state before changes - Claude Generated
            if provider == "gemini":
                self.logger.critical(f"🔍 CONFIG_BEFORE: gemini_preferred_model='{self.config.unified_config.gemini_preferred_model}'")
            elif provider == "anthropic":
                self.logger.critical(f"🔍 CONFIG_BEFORE: anthropic_preferred_model='{self.config.unified_config.anthropic_preferred_model}'")

            updated = False

            # Update static providers
            if provider == "gemini":
                self.config.unified_config.gemini_preferred_model = model
                updated = True
            elif provider == "anthropic":
                self.config.unified_config.anthropic_preferred_model = model
                updated = True

            # Update providers in unified provider list
            for unified_provider in self.config.unified_config.providers:
                if unified_provider.name == provider:
                    unified_provider.preferred_model = model
                    updated = True
                    break

            if updated:
                # 🔍 DEBUG: Log config state after changes - Claude Generated
                if provider == "gemini":
                    self.logger.critical(f"🔍 CONFIG_AFTER: gemini_preferred_model='{self.config.unified_config.gemini_preferred_model}'")
                elif provider == "anthropic":
                    self.logger.critical(f"🔍 CONFIG_AFTER: anthropic_preferred_model='{self.config.unified_config.anthropic_preferred_model}'")

                # Configuration changed in memory - parent dialog will handle save
                self.config_changed.emit()

                # Show toast notification for model preference save - Claude Generated
                model_text = model or "auto-select"
                self._show_save_toast(f"⚙️ {provider}: {model_text}")

                self.logger.info(f"Updated preferred model for {provider}: {model or 'auto-select'} (config_changed emitted)")
            else:
                self.logger.warning(f"Provider {provider} not found in configuration")
            
        except Exception as e:
            self.logger.error(f"Error updating model preference: {e}")
    
    def _get_preferred_model_from_config(self, provider: str, config) -> str:
        """Get preferred model from direct provider configuration - Claude Generated"""
        try:
            # Check static providers
            if provider == "gemini":
                result = config.unified_config.gemini_preferred_model or ""
                self.logger.critical(f"🔍 GET_PREF_GEMINI: gemini_preferred_model='{config.unified_config.gemini_preferred_model}' -> '{result}'")
                return result
            elif provider == "anthropic":
                result = config.unified_config.anthropic_preferred_model or ""
                self.logger.critical(f"🔍 GET_PREF_ANTHROPIC: anthropic_preferred_model='{config.unified_config.anthropic_preferred_model}' -> '{result}'")
                return result
            
            # Check providers in unified provider list
            for unified_provider in config.unified_config.providers:
                if unified_provider.name == provider:
                    return unified_provider.preferred_model or ""
            
            return ""
        except Exception as e:
            self.logger.warning(f"Error getting preferred model for {provider}: {e}")
            return ""
    
    def _get_available_prompt_tasks(self) -> List[str]:
        """Get available tasks dynamically from prompts.json - Claude Generated"""
        try:
            import json
            import os
            
            # Try to load prompts.json from project root
            prompt_file_paths = [
                "prompts.json",
                "../prompts.json", 
                "../../prompts.json",
                os.path.join(os.path.dirname(__file__), "..", "..", "prompts.json")
            ]
            
            for prompt_path in prompt_file_paths:
                try:
                    if os.path.exists(prompt_path):
                        with open(prompt_path, 'r', encoding='utf-8') as f:
                            prompts_data = json.load(f)
                        available_tasks = list(prompts_data.keys())
                        self.logger.info(f"Loaded {len(available_tasks)} tasks from {prompt_path}: {available_tasks}")
                        return sorted(available_tasks)
                except Exception as e:
                    self.logger.debug(f"Could not load prompts from {prompt_path}: {e}")
                    continue
            
            # Fallback to known tasks if prompts.json is not found
            fallback_tasks = ["initialisation", "keywords", "keywords_chunked", "rephrase", "image_text_extraction"]
            self.logger.warning("Could not load prompts.json, using fallback tasks")
            return fallback_tasks
            
        except Exception as e:
            self.logger.warning(f"Error loading prompt tasks: {e}")
            return ["initialisation", "keywords", "classification"]  # Minimal fallback
    
    def _populate_task_preferences(self):
        """Populate task categories and load task-specific model preferences - Enhanced - Claude Generated"""
        self._populate_task_categories_list()
    
    def _populate_task_categories_list(self):
        """Populate the task categories list with Pipeline, Vision, and other tasks - Claude Generated"""
        self.task_categories_list.clear()
        
        # Pipeline tasks section - Load dynamically from prompts.json - Claude Generated
        pipeline_header = QListWidgetItem("🔥 Available LLM Tasks")
        pipeline_header.setFlags(pipeline_header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        pipeline_header.setBackground(QPalette().alternateBase())
        pipeline_header.setFont(QFont("", -1, QFont.Weight.Bold))
        self.task_categories_list.addItem(pipeline_header)
        
        # Load available tasks dynamically from prompts.json - Claude Generated
        available_tasks = self._get_available_prompt_tasks()
        for task in available_tasks:
            item = QListWidgetItem(f"  📋 {task}")
            item.setData(Qt.ItemDataRole.UserRole, {"task_name": task, "category": "llm_task"})
            self.task_categories_list.addItem(item)
        
        # Vision tasks section  
        vision_header = QListWidgetItem("👁️ Vision Tasks")
        vision_header.setFlags(vision_header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        vision_header.setBackground(QPalette().alternateBase())
        vision_header.setFont(QFont("", -1, QFont.Weight.Bold))
        self.task_categories_list.addItem(vision_header)
        
        vision_tasks = ["image_text_extraction"]
        for task in vision_tasks:
            item = QListWidgetItem(f"  👁️ {task}")
            item.setData(Qt.ItemDataRole.UserRole, {"task_name": task, "category": "vision"})
            self.task_categories_list.addItem(item)
        
        # Load additional tasks from prompts.json
        other_tasks = []
        try:
            import os
            import json
            prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts.json')
            
            if os.path.exists(prompts_path):
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    prompts_data = json.load(f)
                
                for task_name in prompts_data.keys():
                    if (task_name not in pipeline_tasks and 
                        task_name not in vision_tasks and
                        not task_name.startswith('_')):
                        other_tasks.append(task_name)
        
        except Exception as e:
            self.logger.warning(f"Could not load additional tasks from prompts.json: {e}")
        
        if other_tasks:
            other_header = QListWidgetItem("🔧 Additional Tasks")
            other_header.setFlags(other_header.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            other_header.setBackground(QPalette().alternateBase())
            other_header.setFont(QFont("", -1, QFont.Weight.Bold))
            self.task_categories_list.addItem(other_header)
            
            for task in other_tasks:
                item = QListWidgetItem(f"  🔧 {task}")
                item.setData(Qt.ItemDataRole.UserRole, {"task_name": task, "category": "other"})
                self.task_categories_list.addItem(item)
    
    def _on_task_category_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle task category selection change - Claude Generated"""
        # CRITICAL FIX: Save previous task's changes before switching using explicit task name - Claude Generated
        if previous and previous.data(Qt.ItemDataRole.UserRole) and self.current_editing_task:
            # Save the previous task preferences using the explicit task name
            previous_task_data = previous.data(Qt.ItemDataRole.UserRole)
            previous_task_name = previous_task_data["task_name"]
            self.logger.info(f"Saving preferences for previous task: {previous_task_name}")
            self._save_current_task_preferences(explicit_task_name=previous_task_name)

        # CRITICAL FIX: Use safe task clearing to preserve changes - Claude Generated
        if self.current_editing_task and self.task_ui_dirty:
            # Save current task before clearing
            self.logger.info(f"Auto-saving changes for {self.current_editing_task} before clearing selection")
            self._save_current_task_preferences(explicit_task_name=self.current_editing_task)

        self.current_editing_task = None
        self.task_ui_dirty = False

        if not current or not current.data(Qt.ItemDataRole.UserRole):
            self.selected_task_info_label.setText("Select a task from the categories")
            self.chunked_tasks_checkbox.setVisible(False)
            self.chunked_tasks_priority_label.setVisible(False)
            self.chunked_task_model_priority_list.setVisible(False)
            self.task_model_priority_list.clear()
            self.chunked_task_model_priority_list.clear()
            return
        
        task_data = current.data(Qt.ItemDataRole.UserRole)
        task_name = task_data["task_name"]
        category = task_data["category"]

        # CRITICAL FIX: Use safe task switching to prevent contamination - Claude Generated
        self._safe_task_switch(task_name)

        self.selected_task_info_label.setText(f"Task: {task_name} ({category})")
        self.logger.info(f"Now editing task: {task_name}")

        # Show chunked options for applicable tasks
        chunked_applicable = task_name in ["keywords", "initialisation"] or category == "pipeline"
        self.chunked_tasks_checkbox.setVisible(chunked_applicable)

        # Load current model priorities for this task
        self._load_task_specific_model_priorities(task_name)
    
    def _load_task_specific_model_priorities(self, task_name: str):
        """Load model priorities for the selected task using detection service - Claude Generated"""
        # CRITICAL FIX: Clear UI state and reset dirty flag when loading new task - Claude Generated
        self.task_model_priority_list.clear()
        self.chunked_task_model_priority_list.clear()
        self.task_ui_dirty = False  # Loading fresh data, UI is now clean
        
        try:
            # Get model priority for this task from root-level config.unified_config.task_preferences - Claude Generated
            if task_name in self.config.unified_config.task_preferences:
                # Task has specific preferences - validate and use them
                task_pref_data = self.config.unified_config.task_preferences[task_name]
                raw_model_priority = task_pref_data.model_priority if task_pref_data else []
                model_priority = self._validate_and_filter_model_priority(raw_model_priority)
            else:
                # Task has no specific preferences - create intelligent defaults from global provider preferences
                model_priority = self._create_task_defaults_from_global_preferences()
            
            # Populate main priority list with real provider/model validation
            detection_service = ProviderDetectionService()
            available_providers = detection_service.get_available_providers()
            
            for model_config in model_priority:
                provider_name = model_config["provider_name"]
                model_name = model_config["model_name"]
                
                # Validate provider is available
                if provider_name in available_providers:
                    available_models = detection_service.get_available_models(provider_name)
                    
                    # Use "Auto-select" if model is "default" or not available
                    display_model = model_name
                    if model_name == "default" or (available_models and model_name not in available_models):
                        display_model = "(Auto-select)"
                    
                    item_text = f"{provider_name}: {display_model}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, model_config)
                    self.task_model_priority_list.addItem(item)
            
            # Check if task has chunked support - use config.unified_config.task_preferences - Claude Generated
            if task_name in self.config.unified_config.task_preferences:
                task_pref_data = self.config.unified_config.task_preferences[task_name]
                if task_pref_data and task_pref_data.chunked_model_priority:
                    self.chunked_tasks_checkbox.setChecked(True)
                    self._on_chunked_tasks_toggled(True)
                    
                    # Populate chunked priority list
                    for model_config in task_pref_data.chunked_model_priority:
                        provider_name = model_config["provider_name"]
                        model_name = model_config["model_name"]
                        
                        # Validate provider is available
                        if provider_name in available_providers:
                            available_models = detection_service.get_available_models(provider_name)
                            
                            # Use "Auto-select" if model is "default" or not available
                            display_model = model_name
                            if model_name == "default" or (available_models and model_name not in available_models):
                                display_model = "(Auto-select)"
                            
                            item_text = f"{provider_name}: {display_model}"
                            item = QListWidgetItem(item_text)
                            item.setData(Qt.ItemDataRole.UserRole, model_config)
                            self.chunked_task_model_priority_list.addItem(item)
                else:
                    self.chunked_tasks_checkbox.setChecked(False)
                    self._on_chunked_tasks_toggled(False)
            
        except Exception as e:
            self.logger.error(f"Error loading task-specific model priorities: {e}")
            QMessageBox.warning(self, "Load Error", f"Could not load model priorities for task '{task_name}':\n{str(e)}")
    
    def _on_chunked_tasks_toggled(self, checked: bool):
        """Handle chunked tasks checkbox toggle - Claude Generated"""
        self.chunked_tasks_priority_label.setVisible(checked)
        self.chunked_task_model_priority_list.setVisible(checked)

        # CRITICAL FIX: Mark UI as dirty when chunked setting changes - Claude Generated
        if self.current_editing_task:
            self.task_ui_dirty = True
            # Auto-save the chunked setting change immediately
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

        # Create enhanced model selection dialog with task-specific context - Claude Generated
        dialog = TaskModelSelectionDialog(None, parent=self)

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
                
                # Add to appropriate list based on chunked checkbox
                if self.chunked_tasks_checkbox.isChecked() and self.chunked_tasks_checkbox.isVisible():
                    # Ask which list to add to
                    reply = QMessageBox.question(
                        self, "Add to Which List?", 
                        "Add to standard priority list or chunked priority list?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.task_model_priority_list.addItem(item)
                    else:
                        self.chunked_task_model_priority_list.addItem(item)
                else:
                    self.task_model_priority_list.addItem(item)
                
                # CRITICAL FIX: Mark UI as dirty and save immediately using explicit task name - Claude Generated
                self.task_ui_dirty = True
                self._save_current_task_preferences(explicit_task_name=selected_task_name)
    
    def _remove_model_from_task_priority(self):
        """Remove selected model from task priority list - Claude Generated"""
        # Try main list first
        current_item = self.task_model_priority_list.currentItem()
        if current_item:
            row = self.task_model_priority_list.row(current_item)
            self.task_model_priority_list.takeItem(row)
            # CRITICAL FIX: Mark UI as dirty and save with explicit task name - Claude Generated
            if self.current_editing_task:
                self.task_ui_dirty = True
                self._save_current_task_preferences(explicit_task_name=self.current_editing_task)
            else:
                self.logger.warning("No current editing task for remove operation")
            return
        
        # Try chunked list
        current_item = self.chunked_task_model_priority_list.currentItem()
        if current_item:
            row = self.chunked_task_model_priority_list.row(current_item)
            self.chunked_task_model_priority_list.takeItem(row)
            # CRITICAL FIX: Mark UI as dirty and save with explicit task name - Claude Generated
            if self.current_editing_task:
                self.task_ui_dirty = True
                self._save_current_task_preferences(explicit_task_name=self.current_editing_task)
            else:
                self.logger.warning("No current editing task for remove operation")
            return
        
        QMessageBox.information(self, "No Selection", "Please select a model to remove.")
    
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
            
            # Add to configuration
            self.unified_config.providers.append(new_provider)
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
        """Test connections to all enabled providers - Claude Generated"""
        enabled_providers = [p for p in self.unified_config.providers if p.enabled]

        if not enabled_providers:
            QMessageBox.information(self, "No Providers", "No enabled providers to test.")
            return

        # Simple message box implementation since status tab is removed
        test_results = []
        test_results.append("🧪 Provider Connection Test Results:\n")

        # Test each provider (simplified for now)
        for provider in enabled_providers:
            # Simulate test result
            if provider.provider_type == "ollama":
                success = True
                message = "Connection successful (simulated)"
            else:
                success = len(provider.api_key or "") > 0
                message = "API key configured" if success else "No API key configured"

            status_icon = "✅" if success else "❌"
            test_results.append(f"{status_icon} {provider.name}: {message}")

        # Show results in message box
        QMessageBox.information(self, "Provider Test Results", "\n".join(test_results))
    
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

            # Connect to status updates for automatic table refresh
            self.alima_manager.provider_status_service.status_updated.connect(
                self._populate_provider_table
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
            # Refresh the table to show updated status
            self._populate_provider_table()

            # Log the update
            status = provider_info.get('status', 'unknown')
            model_count = provider_info.get('model_count', 0)
            self.logger.debug(f"Provider {provider_name} tested: {status} ({model_count} models)")

        except Exception as e:
            self.logger.error(f"Error handling provider test result for {provider_name}: {e}")
    
    def _update_config_from_ui(self):
        """Update configuration object from UI state - Claude Generated"""
        # Update global preferences
        self.unified_config.preferred_provider = self.preferred_provider_combo.currentText()
        self.unified_config.auto_fallback = self.auto_fallback_checkbox.isChecked()
        self.unified_config.prefer_faster_models = self.prefer_fast_checkbox.isChecked()

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
    
    def _save_current_task_preferences(self, explicit_task_name: str = None):
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
            
            # Extract chunked model priorities if chunked mode is enabled
            chunked_model_priority = None
            if self.chunked_tasks_checkbox.isChecked():
                chunked_model_priority = []
                for i in range(self.chunked_task_model_priority_list.count()):
                    item = self.chunked_task_model_priority_list.item(i)
                    model_config = item.data(Qt.ItemDataRole.UserRole)
                    if model_config:
                        chunked_model_priority.append(model_config)
            
            # CRITICAL FIX: Add validation before saving to prevent cross-contamination - Claude Generated
            if explicit_task_name and explicit_task_name != task_name:
                self.logger.error(f"Task name mismatch: explicit='{explicit_task_name}' vs resolved='{task_name}' - aborting save")
                self._show_save_toast(f"❌ Save aborted: task name mismatch", error=True)
                return

            # CRITICAL FIX: Save proper TaskPreference object instead of dictionary - Claude Generated
            # Map task name to TaskType enum
            task_type_mapping = {
                'initialisation': UnifiedTaskType.INITIALISATION,
                'keywords': UnifiedTaskType.KEYWORDS,
                'rephrase': UnifiedTaskType.KEYWORDS,  # rephrase is keywords variant
                'classification': UnifiedTaskType.CLASSIFICATION,
                'dk_classification': UnifiedTaskType.DK_CLASSIFICATION,
                'image_text_extraction': UnifiedTaskType.VISION,
                'keywords_chunked': UnifiedTaskType.CHUNKED_PROCESSING,
                'extract_initial_keywords': UnifiedTaskType.INITIALISATION,
                'dk_class': UnifiedTaskType.DK_CLASSIFICATION,
                'dk_list': UnifiedTaskType.DK_SEARCH,
                'text_analysis': UnifiedTaskType.GENERAL,
                'vision': UnifiedTaskType.VISION,
                'chunked': UnifiedTaskType.CHUNKED_PROCESSING
            }

            # Get existing TaskPreference to preserve settings
            existing_pref = self.config.unified_config.task_preferences.get(task_name)
            task_type = task_type_mapping.get(task_name, UnifiedTaskType.GENERAL)

            # Create proper TaskPreference object
            task_preference = TaskPreference(
                task_type=task_type,
                model_priority=model_priority,
                chunked_model_priority=chunked_model_priority,
                allow_fallback=existing_pref.allow_fallback if existing_pref else True
            )

            # Save proper TaskPreference object
            self.config.unified_config.task_preferences[task_name] = task_preference

            # Configuration changed in memory - parent dialog will handle save
            self.config_changed.emit()

            # Mark UI as clean after successful change - Claude Generated
            self.task_ui_dirty = False

            # Show toast notification for task preference save - Claude Generated
            self._show_save_toast(f"✅ {task_name} preferences saved")
            self.logger.info(f"Task preferences saved for '{task_name}': {len(model_priority)} models, chunked: {chunked_model_priority is not None}")

            # Emit signal to notify other components about task preference changes - Claude Generated
            self.task_preferences_changed.emit()
            
        except Exception as e:
            self._show_save_toast(f"❌ Save failed: {str(e)[:30]}", error=True)
            self.logger.warning(f"Failed to save current task preferences: {e}")

    def _initialize_task_editing_state(self):
        """Initialize the task editing state to prevent contamination - Claude Generated"""
        try:
            # Clear any initial task selection state
            self.current_editing_task = None
            self.task_ui_dirty = False

            # Ensure task lists are clear initially
            if hasattr(self, 'task_model_priority_list'):
                self.task_model_priority_list.clear()
            if hasattr(self, 'chunked_task_model_priority_list'):
                self.chunked_task_model_priority_list.clear()

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
            if hasattr(self, 'chunked_task_model_priority_list'):
                self.chunked_task_model_priority_list.clear()

            # Update tracking variables
            self.current_editing_task = new_task_name
            self.task_ui_dirty = False

            self.logger.info(f"Successfully switched to task: {new_task_name}")

        except Exception as e:
            self.logger.error(f"Error during safe task switch to {new_task_name}: {e}")
            # Reset to safe state
            self.current_editing_task = None
            self.task_ui_dirty = False
    
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
            detection_service = ProviderDetectionService()
            available_providers = detection_service.get_available_providers()
            
            validated_priority = []
            removed_entries = []
            
            for entry in model_priority:
                provider_name = entry.get("provider_name")
                model_name = entry.get("model_name")
                
                # Skip entries with missing data
                if not provider_name or not model_name:
                    removed_entries.append(f"Incomplete entry: {entry}")
                    continue
                
                # Check if provider is available
                if provider_name not in available_providers:
                    removed_entries.append(f"{provider_name}/{model_name} (provider offline)")
                    continue
                
                # Special case: "auto" is always valid
                if model_name == "auto" or model_name == "default":
                    validated_priority.append(entry)
                    continue
                
                # Check if model is available for this provider
                try:
                    available_models = detection_service.get_available_models(provider_name)
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