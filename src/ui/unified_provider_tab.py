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
    QSplitter, QFrame, QScrollArea, QGridLayout, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QIcon, QPalette

from ..utils.config_manager import ConfigManager, OpenAICompatibleProvider, OllamaProvider
from ..utils.unified_provider_config import (
    UnifiedProviderConfig, 
    UnifiedProvider, 
    TaskPreference, 
    TaskType as UnifiedTaskType,
    get_unified_config_manager
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
            
            if self.provider.type == "ollama":
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
                    
            elif self.provider.type == "openai_compatible":
                # Test OpenAI-compatible connection
                api_key = self.provider.connection_config.get("api_key", "")
                if not api_key:
                    self.test_completed.emit(provider_name, False, "No API key configured")
                    return
                
                # For now, just validate configuration
                self.test_completed.emit(provider_name, True, "Configuration valid (no test call)")
                
            elif self.provider.type in ["gemini", "anthropic"]:
                # Test static provider API keys
                api_key = self.provider.connection_config.get("api_key", "")
                if not api_key:
                    self.test_completed.emit(provider_name, False, "No API key configured")
                else:
                    self.test_completed.emit(provider_name, True, "API key configured (no test call)")
            
            else:
                self.test_completed.emit(provider_name, False, f"Unknown provider type: {self.provider.type}")
                
        except Exception as e:
            self.test_completed.emit(self.provider.name, False, f"Test failed: {str(e)}")


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
        self.type_combo.setCurrentText(self.provider.type)
        
        # Load capabilities
        capabilities_text = "\n".join(self.provider.capabilities)
        self.capabilities_text.setPlainText(capabilities_text)
        
        # Load connection configuration
        config = self.provider.connection_config
        
        if self.provider.type == "ollama":
            if hasattr(self, 'host_edit'):
                self.host_edit.setText(config.get("host", "localhost"))
            if hasattr(self, 'port_spinbox'):
                self.port_spinbox.setValue(config.get("port", 11434))
            if hasattr(self, 'ssl_checkbox'):
                self.ssl_checkbox.setChecked(config.get("use_ssl", False))
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(config.get("api_key", ""))
                
        elif self.provider.type == "openai_compatible":
            if hasattr(self, 'base_url_edit'):
                self.base_url_edit.setText(config.get("base_url", ""))
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(config.get("api_key", ""))
                
        elif self.provider.type in ["gemini", "anthropic"]:
            if hasattr(self, 'api_key_edit'):
                self.api_key_edit.setText(config.get("api_key", ""))
    
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
            type=provider_type,
            connection_config=connection_config,
            capabilities=capabilities,
            enabled=self.enabled_checkbox.isChecked(),
            description=self.description_edit.text()
        )


class UnifiedProviderTab(QWidget):
    """
    Unified Provider Configuration Tab - Claude Generated
    Consolidates LLM Provider and Provider Preferences into single interface
    """
    
    config_changed = pyqtSignal()
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Get unified config manager
        self.unified_config_manager = get_unified_config_manager(config_manager)
        self.unified_config = self.unified_config_manager.get_unified_config()
        
        self._setup_ui()
        self._load_configuration()
        
        # Auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save)
        self.auto_save_timer.start(30000)  # Auto-save every 30 seconds
    
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
        
        # Status & Testing Tab
        self.status_tab = self._create_status_tab()
        self.main_tabs.addTab(self.status_tab, "📊 Status & Testing")
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.test_all_button = QPushButton("🧪 Test All Providers")
        self.test_all_button.clicked.connect(self._test_all_providers)
        
        self.reset_button = QPushButton("🔄 Reset to Defaults") 
        self.reset_button.clicked.connect(self._reset_to_defaults)
        
        self.save_button = QPushButton("💾 Save Configuration")
        self.save_button.clicked.connect(self._save_configuration)
        
        button_layout.addWidget(self.test_all_button)
        button_layout.addStretch()
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
    
    def _create_providers_tab(self) -> QWidget:
        """Create provider management tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Provider list and controls
        list_layout = QHBoxLayout()
        
        # Provider table
        self.provider_table = QTableWidget()
        self.provider_table.setColumnCount(6)
        self.provider_table.setHorizontalHeaderLabels([
            "Name", "Type", "Status", "Models", "Capabilities", "Actions"
        ])
        
        # Make table responsive
        header = self.provider_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Type  
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Status
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Models
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)           # Capabilities
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Actions
        
        list_layout.addWidget(self.provider_table)
        
        # Control buttons
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setMaximumWidth(150)
        
        self.add_provider_button = QPushButton("➕ Add Provider")
        self.add_provider_button.clicked.connect(self._add_provider)
        
        self.edit_provider_button = QPushButton("✏️ Edit Provider")
        self.edit_provider_button.clicked.connect(self._edit_provider)
        
        self.remove_provider_button = QPushButton("🗑️ Remove Provider")
        self.remove_provider_button.clicked.connect(self._remove_provider)
        
        self.refresh_models_button = QPushButton("🔄 Refresh Models")
        self.refresh_models_button.clicked.connect(self._refresh_models)
        
        controls_layout.addWidget(self.add_provider_button)
        controls_layout.addWidget(self.edit_provider_button)
        controls_layout.addWidget(self.remove_provider_button)
        controls_layout.addWidget(self.refresh_models_button)
        controls_layout.addStretch()
        
        list_layout.addWidget(controls_widget)
        layout.addLayout(list_layout)
        
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
        
        # Task-specific preferences
        task_group = QGroupBox("🎯 Task-Specific Preferences")
        task_layout = QVBoxLayout(task_group)
        
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(4)
        self.task_table.setHorizontalHeaderLabels([
            "Task Type", "Preferred Providers", "Performance", "Allow Fallback"
        ])
        
        task_layout.addWidget(self.task_table)
        layout.addWidget(task_group)
        
        return widget
    
    def _create_status_tab(self) -> QWidget:
        """Create status and testing tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Status overview
        status_group = QGroupBox("📊 Provider Status Overview")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        
        status_layout.addWidget(self.status_text)
        layout.addWidget(status_group)
        
        # Test results
        test_group = QGroupBox("🧪 Connection Tests")
        test_layout = QVBoxLayout(test_group)
        
        self.test_progress = QProgressBar()
        self.test_progress.setVisible(False)
        
        self.test_results_text = QTextEdit()
        self.test_results_text.setReadOnly(True)
        
        test_layout.addWidget(self.test_progress)
        test_layout.addWidget(self.test_results_text)
        layout.addWidget(test_group)
        
        return widget
    
    def _load_configuration(self):
        """Load unified configuration into UI - Claude Generated"""
        try:
            # Check if we need to migrate from legacy configuration
            if not self.unified_config.providers and self.config_manager:
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
            
            # Update status
            self._update_status_overview()
            
        except Exception as e:
            self.logger.error(f"Error loading unified provider configuration: {e}")
            QMessageBox.critical(self, "Loading Error", f"Failed to load configuration:\n\n{str(e)}")
    
    def _migrate_from_legacy_config(self):
        """Migrate from legacy LLMConfig to UnifiedProviderConfig - Claude Generated"""
        try:
            # Load current ALIMA config
            alima_config = self.config_manager.load_config()
            
            # Get provider preferences for migration
            provider_preferences = self.config_manager.get_provider_preferences()
            
            # Create unified config from legacy data
            self.unified_config = UnifiedProviderConfig.from_legacy_config(
                alima_config.llm, 
                provider_preferences
            )
            
            # Update the unified config manager
            self.unified_config_manager.save_unified_config(self.unified_config)
            
            self.logger.info(f"Successfully migrated {len(self.unified_config.providers)} providers from legacy config")
            
        except Exception as e:
            self.logger.error(f"Failed to migrate from legacy config: {e}")
            # Create default config if migration fails
            self.unified_config = UnifiedProviderConfig()
    
    def _populate_provider_table(self):
        """Populate the provider table with current providers - Claude Generated"""
        providers = self.unified_config.providers
        self.provider_table.setRowCount(len(providers))
        
        # Get provider detection service for live model detection
        detection_service = None
        if self.config_manager:
            try:
                from ..utils.config_manager import ProviderDetectionService
                detection_service = ProviderDetectionService(self.config_manager)
            except Exception as e:
                self.logger.warning(f"Could not initialize provider detection service: {e}")
        
        for row, provider in enumerate(providers):
            # Name
            name_item = QTableWidgetItem(provider.name)
            if not provider.enabled:
                name_item.setForeground(QPalette().color(QPalette.ColorRole.PlaceholderText))
            self.provider_table.setItem(row, 0, name_item)
            
            # Type
            type_item = QTableWidgetItem(provider.type.title())
            self.provider_table.setItem(row, 1, type_item)
            
            # Status and live model detection
            status_text = "Unknown"
            models_text = "Not detected"
            
            if detection_service:
                try:
                    # Check if provider is available
                    available_providers = detection_service.get_available_providers()
                    is_available = provider.name in available_providers or provider.type in available_providers
                    
                    if is_available:
                        status_text = "✅ Available"
                        
                        # Try to detect models
                        detected_models = detection_service.get_available_models(provider.name)
                        if not detected_models:
                            detected_models = detection_service.get_available_models(provider.type)
                        
                        if detected_models:
                            # Update provider's available models
                            provider.available_models = detected_models
                            models_text = f"{len(detected_models)} models"
                            
                            # Show first few models in tooltip
                            models_preview = ", ".join(detected_models[:5])
                            if len(detected_models) > 5:
                                models_preview += f" (+{len(detected_models) - 5} more)"
                        else:
                            models_text = "No models detected"
                    else:
                        status_text = "❌ Unavailable"
                        models_text = "Provider not available"
                        
                except Exception as e:
                    self.logger.debug(f"Error detecting models for {provider.name}: {e}")
                    status_text = "⚠️ Detection failed"
            else:
                # Fallback to stored models
                model_count = len(provider.available_models)
                if model_count > 0:
                    models_text = f"{model_count} models (cached)"
                    status_text = "📝 Configured"
            
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
            
            # Capabilities
            capabilities_text = ", ".join(provider.capabilities[:3])  # Show first 3
            if len(provider.capabilities) > 3:
                capabilities_text += f" (+{len(provider.capabilities) - 3} more)"
            capabilities_item = QTableWidgetItem(capabilities_text)
            self.provider_table.setItem(row, 4, capabilities_item)
            
            # Actions - placeholder (buttons will be added later)
            actions_item = QTableWidgetItem("Edit | Test")
            self.provider_table.setItem(row, 5, actions_item)
    
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
            # Load current configuration
            config = self.config_manager.load_config()
            detection_service = self.config_manager.get_provider_detection_service()
            
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
                
            # Load current configuration
            config = self.config_manager.load_config()
            
            # 🔍 DEBUG: Log config state before changes - Claude Generated
            if provider == "gemini":
                self.logger.critical(f"🔍 CONFIG_BEFORE: gemini_preferred_model='{config.llm.gemini_preferred_model}'")
            elif provider == "anthropic":
                self.logger.critical(f"🔍 CONFIG_BEFORE: anthropic_preferred_model='{config.llm.anthropic_preferred_model}'")
            
            updated = False
            
            # Update static providers
            if provider == "gemini":
                config.llm.gemini_preferred_model = model
                updated = True
            elif provider == "anthropic":
                config.llm.anthropic_preferred_model = model
                updated = True
            
            # Update OpenAI-compatible providers
            for openai_provider in config.llm.openai_compatible_providers:
                if openai_provider.name == provider:
                    openai_provider.preferred_model = model
                    updated = True
                    break
            
            # Update Ollama providers
            for ollama_provider in config.llm.ollama_providers:
                if ollama_provider.name == provider:
                    ollama_provider.preferred_model = model
                    updated = True
                    break
            
            if updated:
                # 🔍 DEBUG: Log config state after changes - Claude Generated
                if provider == "gemini":
                    self.logger.critical(f"🔍 CONFIG_AFTER: gemini_preferred_model='{config.llm.gemini_preferred_model}'")
                elif provider == "anthropic":
                    self.logger.critical(f"🔍 CONFIG_AFTER: anthropic_preferred_model='{config.llm.anthropic_preferred_model}'")
                
                # Save configuration directly
                save_success = self.config_manager.save_config(config)
                self.logger.critical(f"🔍 CONFIG_SAVE_SUCCESS: {save_success}")
                
                # DON'T emit config_changed for model preferences - it causes reload cycle
                # self.config_changed.emit()  # <-- DISABLED TO PREVENT RELOAD CYCLE
                self.logger.info(f"Updated preferred model for {provider}: {model or 'auto-select'} (no UI reload)")
            else:
                self.logger.warning(f"Provider {provider} not found in configuration")
            
        except Exception as e:
            self.logger.error(f"Error updating model preference: {e}")
    
    def _get_preferred_model_from_config(self, provider: str, config) -> str:
        """Get preferred model from direct provider configuration - Claude Generated"""
        try:
            # Check static providers
            if provider == "gemini":
                result = config.llm.gemini_preferred_model or ""
                self.logger.critical(f"🔍 GET_PREF_GEMINI: gemini_preferred_model='{config.llm.gemini_preferred_model}' -> '{result}'")
                return result
            elif provider == "anthropic":
                result = config.llm.anthropic_preferred_model or ""
                self.logger.critical(f"🔍 GET_PREF_ANTHROPIC: anthropic_preferred_model='{config.llm.anthropic_preferred_model}' -> '{result}'")
                return result
            
            # Check OpenAI-compatible providers
            for openai_provider in config.llm.openai_compatible_providers:
                if openai_provider.name == provider:
                    return openai_provider.preferred_model or ""
            
            # Check Ollama providers
            for ollama_provider in config.llm.ollama_providers:
                if ollama_provider.name == provider:
                    return ollama_provider.preferred_model or ""
            
            return ""
        except Exception as e:
            self.logger.warning(f"Error getting preferred model for {provider}: {e}")
            return ""
    
    def _populate_task_preferences(self):
        """Populate task preferences table - Claude Generated"""
        task_types = list(UnifiedTaskType)
        self.task_table.setRowCount(len(task_types))
        
        for row, task_type in enumerate(task_types):
            task_pref = self.unified_config.get_task_preference(task_type)
            
            # Task Type
            task_item = QTableWidgetItem(task_type.value.replace('_', ' ').title())
            self.task_table.setItem(row, 0, task_item)
            
            # Preferred Providers
            providers_text = ", ".join(task_pref.preferred_providers[:2])
            if len(task_pref.preferred_providers) > 2:
                providers_text += f" (+{len(task_pref.preferred_providers) - 2} more)"
            providers_item = QTableWidgetItem(providers_text)
            self.task_table.setItem(row, 1, providers_item)
            
            # Performance Preference
            perf_item = QTableWidgetItem(task_pref.performance_preference.title())
            self.task_table.setItem(row, 2, perf_item)
            
            # Allow Fallback
            fallback_item = QTableWidgetItem("Yes" if task_pref.allow_fallback else "No")
            self.task_table.setItem(row, 3, fallback_item)
    
    def _update_status_overview(self):
        """Update status overview text - Claude Generated"""
        enabled_providers = [p for p in self.unified_config.providers if p.enabled]
        disabled_providers = [p for p in self.unified_config.providers if not p.enabled]
        
        status_text = f"📊 Configuration Overview:\n\n"
        status_text += f"• Total Providers: {len(self.unified_config.providers)}\n"
        status_text += f"• Enabled Providers: {len(enabled_providers)}\n" 
        status_text += f"• Disabled Providers: {len(disabled_providers)}\n"
        status_text += f"• Preferred Provider: {self.unified_config.preferred_provider}\n"
        status_text += f"• Auto Fallback: {'Enabled' if self.unified_config.auto_fallback else 'Disabled'}\n\n"
        
        if enabled_providers:
            status_text += "🟢 Enabled Providers:\n"
            for provider in enabled_providers:
                model_count = len(provider.available_models)
                status_text += f"  • {provider.name} ({provider.type}) - {model_count} models\n"
        
        if disabled_providers:
            status_text += "\n🔴 Disabled Providers:\n"
            for provider in disabled_providers:
                status_text += f"  • {provider.name} ({provider.type})\n"
        
        self.status_text.setPlainText(status_text)
    
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
            
            self.logger.info(f"Added new provider: {new_provider.name} ({new_provider.type})")
    
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
        """Refresh model lists for all providers - Claude Generated"""
        if not self.config_manager:
            QMessageBox.warning(self, "No Config Manager", "No configuration manager available for model detection.")
            return
        
        try:
            from ..utils.config_manager import ProviderDetectionService
            detection_service = ProviderDetectionService(self.config_manager)
            
            updated_count = 0
            
            for provider in self.unified_config.providers:
                if not provider.enabled:
                    continue
                    
                try:
                    # Try to detect models for this provider
                    detected_models = detection_service.get_available_models(provider.name)
                    if not detected_models:
                        detected_models = detection_service.get_available_models(provider.type)
                    
                    if detected_models:
                        old_count = len(provider.available_models)
                        provider.available_models = detected_models
                        new_count = len(detected_models)
                        
                        if new_count != old_count:
                            updated_count += 1
                            self.logger.info(f"Updated {provider.name}: {old_count} → {new_count} models")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to refresh models for {provider.name}: {e}")
            
            # Refresh the UI
            self._populate_provider_table()
            self._update_status_overview()
            
            # Show result
            if updated_count > 0:
                QMessageBox.information(
                    self, "Models Refreshed", 
                    f"Successfully refreshed models for {updated_count} provider(s).\n\n"
                    f"Check the provider table for updated model counts."
                )
            else:
                QMessageBox.information(
                    self, "No Updates", 
                    "No model changes detected. All providers have current model lists."
                )
            
        except Exception as e:
            self.logger.error(f"Error refreshing models: {e}")
            QMessageBox.critical(self, "Refresh Error", f"Failed to refresh models:\n\n{str(e)}")
    
    def _test_all_providers(self):
        """Test connections to all enabled providers - Claude Generated"""
        enabled_providers = [p for p in self.unified_config.providers if p.enabled]
        
        if not enabled_providers:
            QMessageBox.information(self, "No Providers", "No enabled providers to test.")
            return
        
        self.test_progress.setVisible(True)
        self.test_progress.setMaximum(len(enabled_providers))
        self.test_progress.setValue(0)
        
        self.test_results_text.clear()
        self.test_results_text.append("🧪 Starting provider connection tests...\n")
        
        # Test each provider (simplified for now)
        for i, provider in enumerate(enabled_providers):
            self.test_results_text.append(f"Testing {provider.name} ({provider.type})...")
            
            # Simulate test result
            if provider.type == "ollama":
                success = True
                message = "Connection successful (simulated)"
            else:
                success = len(provider.connection_config.get("api_key", "")) > 0
                message = "API key configured" if success else "No API key configured"
            
            status_icon = "✅" if success else "❌"
            self.test_results_text.append(f"{status_icon} {provider.name}: {message}")
            
            self.test_progress.setValue(i + 1)
        
        self.test_results_text.append("\n🏁 Testing completed.")
        QTimer.singleShot(3000, lambda: self.test_progress.setVisible(False))
    
    def _reset_to_defaults(self):
        """Reset configuration to defaults - Claude Generated"""
        reply = QMessageBox.question(
            self, "Reset Configuration",
            "Are you sure you want to reset all provider configuration to defaults?\n\nThis will remove all custom providers and preferences.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.unified_config = UnifiedProviderConfig()  # Create new default config
            self._load_configuration()
            self.config_changed.emit()
            
            QMessageBox.information(self, "Reset Complete", "Configuration has been reset to defaults.")
    
    def _save_configuration(self):
        """Save unified configuration - Claude Generated"""
        try:
            # Update config from UI
            self._update_config_from_ui()
            
            # Save via unified config manager
            success = self.unified_config_manager.save_unified_config(self.unified_config)
            
            if success:
                QMessageBox.information(self, "Save Successful", "Provider configuration has been saved successfully.")
                self.config_changed.emit()
            else:
                QMessageBox.warning(self, "Save Failed", "Failed to save provider configuration. Please check the logs.")
                
        except Exception as e:
            self.logger.error(f"Error saving unified provider configuration: {e}")
            QMessageBox.critical(self, "Save Error", f"Error saving configuration:\n\n{str(e)}")
    
    def _update_config_from_ui(self):
        """Update configuration object from UI state - Claude Generated"""
        # Update global preferences
        self.unified_config.preferred_provider = self.preferred_provider_combo.currentText()
        self.unified_config.auto_fallback = self.auto_fallback_checkbox.isChecked()
        self.unified_config.prefer_faster_models = self.prefer_fast_checkbox.isChecked()
        
        # Task preferences updates would go here
        # For now, they're managed separately
    
    def _auto_save(self):
        """Auto-save configuration periodically - Claude Generated"""
        try:
            self._update_config_from_ui()
            self.unified_config_manager.save_unified_config(self.unified_config)
            self.logger.debug("Auto-saved unified provider configuration")
        except Exception as e:
            self.logger.warning(f"Auto-save failed: {e}")