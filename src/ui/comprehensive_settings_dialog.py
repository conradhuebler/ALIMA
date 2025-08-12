#!/usr/bin/env python3
"""
Comprehensive Settings Dialog for ALIMA
Combines prompt configuration, database settings, LLM configuration, and all other settings.
Claude Generated
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, 
    QFormLayout, QLineEdit, QSpinBox, QComboBox, QCheckBox,
    QPushButton, QTextEdit, QLabel, QGroupBox, QScrollArea,
    QMessageBox, QFileDialog, QProgressDialog, QGridLayout,
    QSplitter, QListWidget, QListWidgetItem, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QIcon
import json
import logging
import getpass
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.config_manager import ConfigManager, AlimaConfig, DatabaseConfig, LLMConfig, CatalogConfig, SystemConfig, OpenAICompatibleProvider, OllamaProvider
from ..llm.prompt_service import PromptService


class OllamaConnectionTestWorker(QThread):
    """Asynchronous worker for testing Ollama connections without blocking GUI - Claude Generated"""
    
    # Signals for communication with main thread
    test_completed = pyqtSignal(bool, str, list)  # success, message, models
    progress_updated = pyqtSignal(str)  # progress message
    
    def __init__(self, provider: OllamaProvider):
        super().__init__()
        self.provider = provider
        self._should_cancel = False
        
    def cancel(self):
        """Cancel the connection test - Claude Generated"""
        self._should_cancel = True
        
    def run(self):
        """Run connection test in background thread - Claude Generated"""
        try:
            if self.provider.connection_type == "openai_compatible":
                self._test_openai_compatible()
            else:  # native_client
                self._test_native()
        except Exception as e:
            self.test_completed.emit(False, f"Unexpected error: {str(e)}", [])
            
    def _test_openai_compatible(self):
        """Test OpenAI-compatible connection - Claude Generated"""
        try:
            import openai
            
            self.progress_updated.emit("Creating OpenAI client...")
            if self._should_cancel:
                return
            
            # Create temporary OpenAI client with timeout
            client_params = {
                "base_url": self.provider.base_url,
                "api_key": self.provider.api_key if self.provider.api_key else "no-key-required",
                "timeout": 10.0  # 10 second timeout
            }
            
            client = openai.OpenAI(**client_params)
            
            self.progress_updated.emit("Fetching model list...")
            if self._should_cancel:
                return
            
            # Try to get models with timeout
            models_response = client.models.list()
            models = [model.id for model in models_response.data]
            
            if self._should_cancel:
                return
            
            self.test_completed.emit(True, "Connection successful", models)
            
        except ImportError:
            self.test_completed.emit(False, "OpenAI package not installed. Install with: pip install openai", [])
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                self.test_completed.emit(False, f"Authentication failed: {error_msg}\nPlease check your API key configuration.", [])
            elif "404" in error_msg or "not found" in error_msg.lower():
                self.test_completed.emit(False, f"Server doesn't support OpenAI-compatible API: {error_msg}\nTry switching to 'native_client' connection type.", [])
            else:
                self.test_completed.emit(False, f"Connection failed: {error_msg}\nPlease check host, port, and network connectivity.", [])
    
    def _test_native(self):
        """Test native Ollama connection - Claude Generated"""
        try:
            import ollama
            
            self.progress_updated.emit("Creating native Ollama client...")
            if self._should_cancel:
                return
            
            # Create temporary native client with timeout
            client_params = {
                "host": self.provider.base_url,
                "timeout": 10.0  # 10 second timeout
            }
            if self.provider.api_key:
                client_params["headers"] = {"Authorization": self.provider.api_key}
            
            client = ollama.Client(**client_params)
            
            self.progress_updated.emit("Fetching model list...")
            if self._should_cancel:
                return
            
            # Try to get models using native client
            models_response = client.list()
            models = [model["name"] for model in models_response.get("models", [])]
            
            if self._should_cancel:
                return
            
            self.test_completed.emit(True, "Connection successful", models)
            
        except ImportError:
            self.test_completed.emit(False, "Ollama package not installed. Install with: pip install ollama", [])
        except Exception as e:
            self.test_completed.emit(False, f"Connection failed: {str(e)}\nPlease check host, port, and network connectivity.", [])


class OllamaProviderEditorDialog(QDialog):
    """Dialog for adding/editing Ollama providers - Claude Generated"""
    
    def __init__(self, parent=None, provider: OllamaProvider = None):
        super().__init__(parent)
        self.provider = provider
        self.is_edit_mode = provider is not None
        
        self.setWindowTitle("Edit Ollama Provider" if self.is_edit_mode else "Add Ollama Provider")
        self.setModal(True)
        self.resize(500, 400)
        
        self._setup_ui()
        self._load_provider_data()
        
    def _setup_ui(self):
        """Setup Ollama provider editor UI - Claude Generated"""
        layout = QVBoxLayout()
        
        # Provider form
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., 'localhost', 'work_server', 'cloud_instance'")
        form_layout.addRow("Provider Name (Alias):", self.name_edit)
        
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("e.g., 'localhost', '192.168.1.100', 'ollama.example.com'")
        form_layout.addRow("Host:", self.host_edit)
        
        self.port_edit = QSpinBox()
        self.port_edit.setRange(1, 65535)
        self.port_edit.setValue(11434)
        form_layout.addRow("Port:", self.port_edit)
        
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Optional: API key for authenticated access")
        form_layout.addRow("API Key:", self.api_key_edit)
        
        self.use_ssl_checkbox = QCheckBox("Use HTTPS instead of HTTP")
        form_layout.addRow("SSL:", self.use_ssl_checkbox)
        
        # Ollama always uses native_client - no user selection needed - Claude Generated
        # self.connection_type_combo = QComboBox()
        # self.connection_type_combo.addItems(["openai_compatible", "native_client"])
        # self.connection_type_combo.setCurrentText("native_client")
        # form_layout.addRow("Connection Type:", self.connection_type_combo)
        
        self.enabled_checkbox = QCheckBox("Provider enabled")
        self.enabled_checkbox.setChecked(True)
        form_layout.addRow("Status:", self.enabled_checkbox)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        self.description_edit.setPlaceholderText("Optional description for this provider")
        form_layout.addRow("Description:", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        test_btn = QPushButton("🔧 Test Connection")
        test_btn.clicked.connect(self._test_connection)
        button_layout.addWidget(test_btn)
        
        button_layout.addStretch()
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _load_provider_data(self):
        """Load provider data into form fields - Claude Generated"""
        if self.provider:
            self.name_edit.setText(self.provider.name)
            self.host_edit.setText(self.provider.host)
            self.port_edit.setValue(self.provider.port)
            self.api_key_edit.setText(self.provider.api_key)
            self.use_ssl_checkbox.setChecked(self.provider.use_ssl)
            # self.connection_type_combo.setCurrentText(self.provider.connection_type)  # Always native_client - Claude Generated
            self.enabled_checkbox.setChecked(self.provider.enabled)
            self.description_edit.setPlainText(self.provider.description)
        else:
            # Set default values for new provider - Claude Generated
            self.host_edit.setText("https://ollama.com")  # Official Ollama server
            self.port_edit.setValue(443)  # HTTPS port  
            self.use_ssl_checkbox.setChecked(True)  # Enable HTTPS for official server
    
    def _test_connection(self):
        """Test connection to this Ollama provider with ping test first - Claude Generated"""
        try:
            provider = self.get_provider()
            
            # Show progress dialog
            progress = QProgressDialog(f"Testing {provider.name} connection...", "Cancel", 0, 0, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.show()
            
            # First, do a simple ping test
            progress.setLabelText(f"Pinging {provider.name} server...")
            from ..llm.llm_service import LlmService
            temp_service = LlmService(lazy_initialization=True)
            config = temp_service.config_manager.load_config()
            config.llm.ollama_providers = [provider]
            temp_service.config_manager.config = config
            
            ping_result = temp_service.ping_test_provider(provider.name, timeout=5.0)
            
            if not ping_result['reachable']:
                progress.close()
                QMessageBox.warning(
                    self,
                    "🔌 Server Unreachable",
                    f"Could not reach {provider.name} server:\n\n"
                    f"Error: {ping_result.get('error', 'Unknown error')}\n"
                    f"Method: {ping_result.get('method', 'unknown')}\n\n"
                    f"Please check:\n"
                    f"• Server is running\n"
                    f"• Host and port are correct\n"
                    f"• Network connectivity"
                )
                return
            
            # Show ping success
            progress.setLabelText(f"Server reachable ({ping_result['latency_ms']:.1f}ms). Testing connection...")
            
            # Proceed with full connection test
            if provider.connection_type == "openai_compatible":
                self._test_openai_compatible_connection(provider, progress)
            else:  # native_client
                self._test_native_connection(provider, progress)
                
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(
                self,
                "❌ Connection Test Failed", 
                f"Failed to test connection:\n\n{str(e)}"
            )
    
    def _test_openai_compatible_connection(self, provider: OllamaProvider, progress):
        """Test OpenAI-compatible Ollama connection - Claude Generated"""
        try:
            import openai
            
            # Create temporary OpenAI client
            client_params = {
                "base_url": provider.base_url,
                "api_key": provider.api_key if provider.api_key else "no-key-required"
            }
            
            client = openai.OpenAI(**client_params)
            progress.setLabelText("Getting available models...")
            
            # Try to get models
            models_response = client.models.list()
            models = [model.id for model in models_response.data]
            
            progress.close()
            
            if models:
                model_list = "\n".join(f"• {model}" for model in models[:15])
                if len(models) > 15:
                    model_list += f"\n... and {len(models) - 15} more models"
                
                QMessageBox.information(
                    self,
                    "✅ Connection Successful",
                    f"Ollama provider '{provider.name}' connection successful!\n\n"
                    f"Host: {provider.host}:{provider.port}\n"
                    f"Type: {provider.connection_type}\n"
                    f"SSL: {'Yes' if provider.use_ssl else 'No'}\n\n"
                    f"Available models ({len(models)} total):\n{model_list}"
                )
            else:
                QMessageBox.information(
                    self,
                    "✅ Connection Successful",
                    f"Ollama provider '{provider.name}' connection successful!\n\n"
                    f"Host: {provider.host}:{provider.port}\n"
                    f"No models available (server may be starting up)"
                )
                
        except Exception as e:
            progress.close()
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                QMessageBox.warning(
                    self,
                    "🔑 Authentication Required",
                    f"Connection failed due to authentication:\n\n{error_msg}\n\n"
                    f"Please check your API key configuration."
                )
            else:
                QMessageBox.warning(
                    self,
                    "⚠️ Connection Failed",
                    f"Failed to connect to Ollama provider '{provider.name}':\n\n{error_msg}\n\n"
                    f"Please check host, port, and network connectivity."
                )
    
    def _test_native_connection(self, provider: OllamaProvider, progress):
        """Test native Ollama client connection - Claude Generated"""
        try:
            import ollama
            
            # Create temporary native client
            client_params = {"host": provider.base_url}
            if provider.api_key:
                client_params["headers"] = {"Authorization": provider.api_key}
            
            client = ollama.Client(**client_params)
            progress.setLabelText("Getting available models...")
            
            # Try to get models using native client
            models_response = client.list()
            models = [model["name"] for model in models_response.get("models", [])]
            
            progress.close()
            
            if models:
                model_list = "\n".join(f"• {model}" for model in models[:15])
                if len(models) > 15:
                    model_list += f"\n... and {len(models) - 15} more models"
                
                QMessageBox.information(
                    self,
                    "✅ Connection Successful",
                    f"Ollama native provider '{provider.name}' connection successful!\n\n"
                    f"Host: {provider.host}:{provider.port}\n"
                    f"Type: {provider.connection_type}\n\n"
                    f"Available models ({len(models)} total):\n{model_list}"
                )
            else:
                QMessageBox.information(
                    self,
                    "✅ Connection Successful",
                    f"Ollama native provider '{provider.name}' connection successful!\n\n"
                    f"Host: {provider.host}:{provider.port}\n"
                    f"No models available (server may be starting up)"
                )
                
        except ImportError:
            progress.close()
            QMessageBox.critical(
                self,
                "❌ Missing Dependency",
                f"Native Ollama client requires the 'ollama' Python package.\n\n"
                f"Install with: pip install ollama"
            )
        except Exception as e:
            progress.close()
            QMessageBox.warning(
                self,
                "⚠️ Connection Failed",
                f"Failed to connect to Ollama native provider '{provider.name}':\n\n{str(e)}\n\n"
                f"Please check host, port, and network connectivity."
            )
    
    def get_provider(self) -> OllamaProvider:
        """Get provider object from form data - Claude Generated"""
        return OllamaProvider(
            name=self.name_edit.text(),
            host=self.host_edit.text(),
            port=self.port_edit.value(),
            api_key=self.api_key_edit.text(),
            use_ssl=self.use_ssl_checkbox.isChecked(),
            connection_type="native_client",  # Always use native client for Ollama - Claude Generated
            enabled=self.enabled_checkbox.isChecked(),
            description=self.description_edit.toPlainText()
        )


class ProviderEditorDialog(QDialog):
    """Dialog for adding/editing OpenAI-compatible providers - Claude Generated"""
    
    def __init__(self, parent=None, provider: OpenAICompatibleProvider = None):
        super().__init__(parent)
        self.provider = provider
        self.is_edit_mode = provider is not None
        
        self.setWindowTitle("Edit Provider" if self.is_edit_mode else "Add Provider")
        self.setModal(True)
        self.resize(500, 400)
        
        self._setup_ui()
        self._load_provider_data()
        
    def _setup_ui(self):
        """Setup provider editor UI - Claude Generated"""
        layout = QVBoxLayout()
        
        # Provider form
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        form_layout.addRow("Provider Name:", self.name_edit)
        
        self.base_url_edit = QLineEdit()
        form_layout.addRow("Base URL:", self.base_url_edit)
        
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addRow("API Key:", self.api_key_edit)
        
        self.enabled_checkbox = QCheckBox("Provider enabled")
        self.enabled_checkbox.setChecked(True)
        form_layout.addRow("Status:", self.enabled_checkbox)
        
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        form_layout.addRow("Description:", self.description_edit)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        test_btn = QPushButton("🔧 Test Connection")
        test_btn.clicked.connect(self._test_connection)
        button_layout.addWidget(test_btn)
        
        button_layout.addStretch()
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _load_provider_data(self):
        """Load provider data into form fields - Claude Generated"""
        if self.provider:
            self.name_edit.setText(self.provider.name)
            self.base_url_edit.setText(self.provider.base_url)
            self.api_key_edit.setText(self.provider.api_key)
            self.enabled_checkbox.setChecked(self.provider.enabled)
            self.description_edit.setPlainText(self.provider.description)
        else:
            # Set default values for new provider
            self.base_url_edit.setText("https://api.example.com/v1")
    
    def _test_connection(self):
        """Test provider connection with ping test first and show available models - Claude Generated"""
        # Basic validation
        if not self.name_edit.text() or not self.base_url_edit.text():
            QMessageBox.warning(self, "Validation Error", "Name and Base URL are required!")
            return
            
        # Create temporary provider for testing
        test_provider = self.get_provider()
        
        # Show progress dialog
        progress = QProgressDialog("Testing connection...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        try:
            # Import LlmService for testing
            from ..llm.llm_service import LlmService
            
            # First, do a simple ping test
            progress.setLabelText(f"Pinging {test_provider.name} server...")
            temp_service = LlmService(lazy_initialization=True)
            config = temp_service.config_manager.load_config()
            config.llm.openai_compatible_providers = [test_provider]
            temp_service.config_manager.config = config
            
            ping_result = temp_service.ping_test_provider(test_provider.name, timeout=5.0)
            
            if not ping_result['reachable']:
                progress.close()
                QMessageBox.warning(
                    self,
                    "🔌 Server Unreachable",
                    f"Could not reach {test_provider.name} server:\n\n"
                    f"Error: {ping_result.get('error', 'Unknown error')}\n"
                    f"Method: {ping_result.get('method', 'unknown')}\n\n"
                    f"Please check:\n"
                    f"• Server is running\n"
                    f"• Base URL is correct\n"
                    f"• Network connectivity"
                )
                return
            
            # Show ping success and continue with API test
            progress.setLabelText(f"Server reachable ({ping_result['latency_ms']:.1f}ms). Testing API connection...")
            
            # Create temporary config manager and LLM service
            temp_config_manager = ConfigManager()
            
            # Save original config for restoration
            original_config = temp_config_manager.load_config()
            
            # Create temporary config with test provider
            temp_config = ConfigManager().load_config()
            temp_config.llm.add_provider(test_provider)
            
            # Save temporarily
            temp_config_manager.save_config(temp_config, "user")
            
            # Create new LLM service with temporary config
            llm_service = LlmService()
            llm_service.reload_providers()
            
            progress.setLabelText("Getting available models...")
            
            # Get available models
            models = llm_service.get_available_models(test_provider.name.lower().replace(" ", "_"))
            
            progress.close()
            
            if models:
                # Show success dialog with models
                model_list = "\n".join(f"• {model}" for model in models[:10])  # Show first 10 models
                if len(models) > 10:
                    model_list += f"\n... and {len(models) - 10} more models"
                
                QMessageBox.information(
                    self, 
                    "✅ Connection Successful", 
                    f"Provider '{test_provider.name}' connected successfully!\n\n"
                    f"Available models ({len(models)} total):\n{model_list}"
                )
                
                # Update models in test provider (optional)
                test_provider.models = models
                self._update_form_from_provider(test_provider)
            else:
                QMessageBox.warning(
                    self,
                    "⚠️ Connection Warning",
                    f"Connected to '{test_provider.name}' but no models found.\n"
                    "This might indicate an authentication issue or empty model list."
                )
            
            # Restore original configuration
            temp_config_manager.save_config(original_config, "user")
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "❌ Connection Failed", 
                f"Failed to connect to '{test_provider.name}':\n\n{str(e)}\n\n"
                "Please check:\n"
                "• Base URL is correct\n"
                "• API key is valid\n" 
                "• Provider is accessible"
            )
            
            # Ensure we restore original configuration even on error
            try:
                temp_config_manager.save_config(original_config, "user")
                from ..llm.llm_service import LlmService
                llm_service = LlmService()
                llm_service.reload_providers()
            except Exception as cleanup_error:
                self.logger.warning(f"Error during cleanup: {cleanup_error}")
                pass  # Best effort cleanup
    
    def _update_form_from_provider(self, provider: OpenAICompatibleProvider):
        """Update form fields from provider data - Claude Generated"""
        self.name_edit.setText(provider.name)
        self.base_url_edit.setText(provider.base_url)
        self.api_key_edit.setText(provider.api_key)
        self.enabled_checkbox.setChecked(provider.enabled)
        self.description_edit.setPlainText(provider.description)
        
        # Add models info to description if available
        if provider.models:
            current_desc = provider.description
            models_info = f"\n\nAvailable models ({len(provider.models)}): {', '.join(provider.models[:5])}"
            if len(provider.models) > 5:
                models_info += f" and {len(provider.models) - 5} more..."
            
            if "Available models" not in current_desc:
                self.description_edit.setPlainText(current_desc + models_info)
    
    def get_provider(self) -> OpenAICompatibleProvider:
        """Get provider from form data - Claude Generated"""
        return OpenAICompatibleProvider(
            name=self.name_edit.text().strip(),
            base_url=self.base_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            enabled=self.enabled_checkbox.isChecked(),
            description=self.description_edit.toPlainText().strip()
        )


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
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        
        # Load current configuration
        self.current_config = self.config_manager.load_config()
        
        # Load prompts for prompt editing
        self.prompts_file = Path("prompts.json")
        self.prompt_data = self._load_prompts()
        
        self.setWindowTitle("ALIMA Settings")
        self.setModal(True)
        self.resize(900, 700)
        
        self._setup_ui()
        self._load_current_settings()
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts.json file - Claude Generated"""
        try:
            if self.prompts_file.exists():
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load prompts: {e}")
            return {}
    
    def _save_prompts(self, prompt_data: Dict[str, Any]) -> bool:
        """Save prompts.json file - Claude Generated"""
        try:
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save prompts: {e}")
            return False
    
    def _setup_ui(self):
        """Setup the user interface - Claude Generated"""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.database_tab = self._create_database_tab()
        self.llm_tab = self._create_llm_tab()
        self.catalog_tab = self._create_catalog_tab()
        self.prompts_tab = self._create_prompts_tab()
        self.system_tab = self._create_system_tab()
        self.about_tab = self._create_about_tab()
        
        # Add tabs
        self.tab_widget.addTab(self.database_tab, "🗄️ Database")
        self.tab_widget.addTab(self.llm_tab, "🤖 LLM Providers") 
        self.tab_widget.addTab(self.catalog_tab, "📚 Catalog")
        self.tab_widget.addTab(self.prompts_tab, "📝 Prompts")
        self.tab_widget.addTab(self.system_tab, "⚙️ System")
        self.tab_widget.addTab(self.about_tab, "ℹ️ About")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.test_button = QPushButton("🔧 Test Connection")
        self.test_button.clicked.connect(self._test_database_connection)
        
        self.reset_button = QPushButton("↶ Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        
        self.save_button = QPushButton("💾 Save & Close")
        self.save_button.clicked.connect(self._save_and_close)
        
        self.cancel_button = QPushButton("❌ Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
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
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_llm_tab(self) -> QWidget:
        """Create LLM providers configuration tab with dynamic provider management - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Static Provider API Keys
        static_group = QGroupBox("Static Provider API Keys")
        static_layout = QFormLayout()
        
        self.gemini_key = QLineEdit()
        self.gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        static_layout.addRow("Google Gemini:", self.gemini_key)
        
        self.anthropic_key = QLineEdit()
        self.anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        static_layout.addRow("Anthropic Claude:", self.anthropic_key)
        
        static_group.setLayout(static_layout)
        layout.addWidget(static_group)
        
        # OpenAI-Compatible Providers
        openai_group = QGroupBox("OpenAI-Compatible Providers")
        openai_layout = QVBoxLayout()
        
        # Provider management buttons
        provider_buttons = QHBoxLayout()
        
        add_provider_btn = QPushButton("➕ Add Provider")
        add_provider_btn.clicked.connect(self._add_openai_provider)
        provider_buttons.addWidget(add_provider_btn)
        
        edit_provider_btn = QPushButton("✏️ Edit Provider")
        edit_provider_btn.clicked.connect(self._edit_openai_provider)
        provider_buttons.addWidget(edit_provider_btn)
        
        delete_provider_btn = QPushButton("🗑️ Delete Provider")
        delete_provider_btn.clicked.connect(self._delete_openai_provider)
        provider_buttons.addWidget(delete_provider_btn)
        
        # Add provider status refresh button - Claude Generated
        refresh_status_btn = QPushButton("🔄 Refresh All Status")
        refresh_status_btn.clicked.connect(self._refresh_all_providers_status)
        provider_buttons.addWidget(refresh_status_btn)
        
        provider_buttons.addStretch()
        openai_layout.addLayout(provider_buttons)
        
        # Provider list
        self.providers_list = QListWidget()
        self.providers_list.itemDoubleClicked.connect(self._edit_openai_provider)
        openai_layout.addWidget(self.providers_list)
        
        openai_group.setLayout(openai_layout)
        layout.addWidget(openai_group)
        
        # Multi-Instance Ollama Configuration - Claude Generated
        ollama_group = QGroupBox("Ollama Providers")
        ollama_layout = QVBoxLayout()
        
        # Provider management buttons
        ollama_buttons = QHBoxLayout()
        
        add_ollama_btn = QPushButton("➕ Add Ollama Provider")
        add_ollama_btn.clicked.connect(self._add_ollama_provider)
        ollama_buttons.addWidget(add_ollama_btn)
        
        edit_ollama_btn = QPushButton("✏️ Edit Provider")
        edit_ollama_btn.clicked.connect(self._edit_ollama_provider)
        ollama_buttons.addWidget(edit_ollama_btn)
        
        delete_ollama_btn = QPushButton("🗑️ Delete Provider")
        delete_ollama_btn.clicked.connect(self._delete_ollama_provider)
        ollama_buttons.addWidget(delete_ollama_btn)
        
        ollama_buttons.addStretch()
        ollama_layout.addLayout(ollama_buttons)
        
        # Provider list
        self.ollama_providers_list = QListWidget()
        self.ollama_providers_list.itemDoubleClicked.connect(self._edit_ollama_provider)
        ollama_layout.addWidget(self.ollama_providers_list)
        
        # Test connection button for Ollama
        ollama_test_layout = QHBoxLayout()
        self.ollama_test_btn = QPushButton("🔧 Test Selected Provider")
        self.ollama_test_btn.clicked.connect(self._test_ollama_connection)
        ollama_test_layout.addWidget(self.ollama_test_btn)
        ollama_test_layout.addStretch()
        ollama_layout.addLayout(ollama_test_layout)
        
        ollama_group.setLayout(ollama_layout)
        layout.addWidget(ollama_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_catalog_tab(self) -> QWidget:
        """Create catalog configuration tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Catalog settings
        catalog_group = QGroupBox("Library Catalog Configuration")
        catalog_layout = QFormLayout()
        
        self.catalog_token = QLineEdit()
        self.catalog_token.setEchoMode(QLineEdit.EchoMode.Password)
        catalog_layout.addRow("Catalog Token:", self.catalog_token)
        
        self.catalog_search_url = QLineEdit()
        catalog_layout.addRow("Search URL:", self.catalog_search_url)
        
        self.catalog_details_url = QLineEdit()
        catalog_layout.addRow("Details URL:", self.catalog_details_url)
        
        catalog_group.setLayout(catalog_layout)
        layout.addWidget(catalog_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_prompts_tab(self) -> QWidget:
        """Create prompts configuration tab - Claude Generated"""
        widget = QWidget()
        layout = QHBoxLayout()
        
        # Left side: Prompt task list
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Prompt Tasks:"))
        
        self.prompts_list = QListWidget()
        self.prompts_list.currentItemChanged.connect(self._on_prompt_selected)
        left_layout.addWidget(self.prompts_list)
        
        # Buttons for prompt management
        prompt_buttons = QHBoxLayout()
        
        add_prompt_btn = QPushButton("Add Task")
        add_prompt_btn.clicked.connect(self._add_prompt_task)
        prompt_buttons.addWidget(add_prompt_btn)
        
        del_prompt_btn = QPushButton("Delete Task") 
        del_prompt_btn.clicked.connect(self._delete_prompt_task)
        prompt_buttons.addWidget(del_prompt_btn)
        
        left_layout.addLayout(prompt_buttons)
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(250)
        
        # Right side: Prompt editor
        self.prompt_editor = QTextEdit()
        self.prompt_editor.setFont(QFont("Monaco", 10))
        self.prompt_editor.textChanged.connect(self._on_prompt_edited)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self.prompt_editor)
        splitter.setSizes([250, 650])
        
        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget
    
    def _create_system_tab(self) -> QWidget:
        """Create system configuration tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # System settings
        system_group = QGroupBox("System Settings")
        system_layout = QFormLayout()
        
        self.debug_mode = QCheckBox("Enable debug mode")
        system_layout.addRow("Debug:", self.debug_mode)
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        system_layout.addRow("Log Level:", self.log_level)
        
        self.cache_dir = QLineEdit()
        system_layout.addRow("Cache Directory:", self.cache_dir)
        
        self.data_dir = QLineEdit()
        system_layout.addRow("Data Directory:", self.data_dir)
        
        self.temp_dir = QLineEdit()
        system_layout.addRow("Temp Directory:", self.temp_dir)
        
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
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
<b>Legacy:</b> {config_info['legacy_config']}
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
        config = self.current_config
        
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
        
        # LLM settings (only static providers)
        self.gemini_key.setText(config.llm.gemini)
        self.anthropic_key.setText(config.llm.anthropic)
        
        # Load Ollama providers - Claude Generated
        # (The Ollama providers are loaded in _load_ollama_providers_list)
        
        # Catalog settings
        self.catalog_token.setText(config.catalog.catalog_token)
        self.catalog_search_url.setText(config.catalog.catalog_search_url)
        self.catalog_details_url.setText(config.catalog.catalog_details_url)
        
        # System settings
        self.debug_mode.setChecked(config.system.debug)
        self.log_level.setCurrentText(config.system.log_level)
        self.cache_dir.setText(config.system.cache_dir)
        self.data_dir.setText(config.system.data_dir)
        self.temp_dir.setText(config.system.temp_dir)
        
        # Update UI based on database type
        self._on_db_type_changed(config.database.db_type)
        
        # Load prompts
        self._load_prompts_list()
        
        # Load OpenAI-compatible providers
        self._load_providers_list()
        
        # Load Ollama providers
        self._load_ollama_providers_list()
    
    def _load_providers_list(self):
        """Load OpenAI-compatible providers into the list widget - Claude Generated"""
        self.providers_list.clear()
        for provider in self.current_config.llm.openai_compatible_providers:
            status = "✅" if provider.enabled else "❌"
            item_text = f"{status} {provider.name} - {provider.base_url}"
            if provider.description:
                item_text += f" ({provider.description})"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, provider)
            self.providers_list.addItem(item)
    
    def _add_openai_provider(self):
        """Add new OpenAI-compatible provider - Claude Generated"""
        dialog = ProviderEditorDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                new_provider = dialog.get_provider()
                
                # Check if provider name already exists
                if self.current_config.llm.get_provider_by_name(new_provider.name):
                    QMessageBox.warning(self, "Duplicate Name", 
                                      f"A provider with name '{new_provider.name}' already exists!")
                    return
                
                # Add provider to configuration
                self.current_config.llm.add_provider(new_provider)
                self._load_providers_list()
                
            except ValueError as e:
                QMessageBox.critical(self, "Invalid Provider", f"Error creating provider: {str(e)}")
    
    def _edit_openai_provider(self):
        """Edit selected OpenAI-compatible provider - Claude Generated"""
        current_item = self.providers_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a provider to edit.")
            return
            
        provider = current_item.data(Qt.ItemDataRole.UserRole)
        dialog = ProviderEditorDialog(parent=self, provider=provider)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                updated_provider = dialog.get_provider()
                
                # Check if name changed and conflicts with existing
                if (updated_provider.name != provider.name and 
                    self.current_config.llm.get_provider_by_name(updated_provider.name)):
                    QMessageBox.warning(self, "Duplicate Name", 
                                      f"A provider with name '{updated_provider.name}' already exists!")
                    return
                
                # Update provider in configuration
                # Remove old and add updated
                self.current_config.llm.remove_provider(provider.name)
                self.current_config.llm.add_provider(updated_provider)
                self._load_providers_list()
                
            except ValueError as e:
                QMessageBox.critical(self, "Invalid Provider", f"Error updating provider: {str(e)}")
    
    def _delete_openai_provider(self):
        """Delete selected OpenAI-compatible provider - Claude Generated"""
        current_item = self.providers_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a provider to delete.")
            return
            
        provider = current_item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self,
            "Delete Provider",
            f"Are you sure you want to delete the provider '{provider.name}'?\n\n"
            f"Base URL: {provider.base_url}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.current_config.llm.remove_provider(provider.name)
            self._load_providers_list()
    
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
            temp_service.config_manager.config = self.current_config
            
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
    
    def _load_prompts_list(self):
        """Load prompts into the list widget - Claude Generated"""
        self.prompts_list.clear()
        for task_name in self.prompt_data.keys():
            if not task_name.startswith('_'):  # Skip metadata fields
                item = QListWidgetItem(task_name)
                self.prompts_list.addItem(item)
    
    def _on_db_type_changed(self, db_type: str):
        """Handle database type change - Claude Generated"""
        if db_type == "sqlite":
            self.sqlite_group.setVisible(True)
            self.mysql_group.setVisible(False)
        else:
            self.sqlite_group.setVisible(False)
            self.mysql_group.setVisible(True)
    
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
    
    def _on_prompt_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle prompt selection change - Claude Generated"""
        if not current:
            self.prompt_editor.clear()
            return
            
        task_name = current.text()
        task_data = self.prompt_data.get(task_name, {})
        
        # Format the prompt data for editing
        formatted_data = json.dumps(task_data, indent=2, ensure_ascii=False)
        self.prompt_editor.setPlainText(formatted_data)
    
    def _on_prompt_edited(self):
        """Handle prompt editor text changes - Claude Generated"""
        current_item = self.prompts_list.currentItem()
        if not current_item:
            return
            
        task_name = current_item.text()
        try:
            # Parse the edited JSON
            edited_data = json.loads(self.prompt_editor.toPlainText())
            self.prompt_data[task_name] = edited_data
        except json.JSONDecodeError:
            # Invalid JSON, don't update the data
            pass
    
    def _add_prompt_task(self):
        """Add new prompt task - Claude Generated"""
        # Simple dialog for task name
        from PyQt6.QtWidgets import QInputDialog
        task_name, ok = QInputDialog.getText(self, "New Prompt Task", "Enter task name:")
        
        if ok and task_name and task_name not in self.prompt_data:
            # Create default prompt structure
            self.prompt_data[task_name] = {
                "fields": ["prompt", "system", "temp", "p-value", "model", "seed"],
                "required": ["abstract"],
                "prompts": [
                    [
                        "Enter your prompt here...",
                        "You are a helpful assistant.",
                        "0.7",
                        "0.1",
                        ["default"],
                        "0"
                    ]
                ]
            }
            self._load_prompts_list()
    
    def _delete_prompt_task(self):
        """Delete selected prompt task - Claude Generated"""
        current_item = self.prompts_list.currentItem()
        if not current_item:
            return
            
        task_name = current_item.text()
        reply = QMessageBox.question(
            self,
            "Delete Prompt Task",
            f"Are you sure you want to delete the '{task_name}' prompt task?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.prompt_data[task_name]
            self._load_prompts_list()
            self.prompt_editor.clear()
    
    def _test_database_connection(self):
        """Test database connection - Claude Generated"""
        # Update config with current settings
        config = self._get_config_from_ui()
        
        # Temporarily update config manager
        self.config_manager._config = config
        
        # Create and start test worker
        self.test_button.setEnabled(False)
        self.test_button.setText("Testing...")
        
        self.db_test_worker = DatabaseTestWorker(self.config_manager)
        self.db_test_worker.test_completed.connect(self._on_db_test_completed)
        self.db_test_worker.start()
    
    @pyqtSlot(bool, str)
    def _on_db_test_completed(self, success: bool, message: str):
        """Handle database test completion - Claude Generated"""
        self.test_button.setEnabled(True)
        self.test_button.setText("🔧 Test Connection")
        
        if success:
            QMessageBox.information(self, "Database Test", f"✅ {message}")
        else:
            QMessageBox.warning(self, "Database Test", f"❌ {message}")
    
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
            self.current_config = AlimaConfig()  # Create default config
            self._load_current_settings()
    
    def _get_config_from_ui(self) -> AlimaConfig:
        """Extract configuration from UI elements - Claude Generated"""
        config = AlimaConfig()
        
        # Database configuration
        config.database = DatabaseConfig(
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
        
        # LLM configuration with flexible Ollama - Claude Generated
        # OllamaConfig no longer needed - using OllamaProvider instead - Claude Generated
        
        # Use current Ollama providers from configuration (they're managed through the provider list)
        config.llm = LLMConfig(
            gemini=self.gemini_key.text(),
            anthropic=self.anthropic_key.text(),
            openai_compatible_providers=self.current_config.llm.openai_compatible_providers,
            ollama_providers=self.current_config.llm.ollama_providers,
            ollama_host=self.current_config.llm.ollama_host,  # Legacy field
            ollama_port=self.current_config.llm.ollama_port   # Legacy field
        )
        
        # Catalog configuration
        config.catalog = CatalogConfig(
            catalog_token=self.catalog_token.text(),
            catalog_search_url=self.catalog_search_url.text(),
            catalog_details_url=self.catalog_details_url.text()
        )
        
        # System configuration
        config.system = SystemConfig(
            debug=self.debug_mode.isChecked(),
            log_level=self.log_level.currentText(),
            cache_dir=self.cache_dir.text(),
            data_dir=self.data_dir.text(),
            temp_dir=self.temp_dir.text()
        )
        
        return config
    
    def _save_and_close(self):
        """Save configuration and close dialog - Claude Generated"""
        try:
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
            
            # Save prompts
            if not self._save_prompts(self.prompt_data):
                QMessageBox.warning(self, "Save Warning", "Configuration saved but failed to save prompts!")
            
            # Emit signal and close
            self.config_changed.emit()
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving configuration: {str(e)}")
    
    # Ollama Provider Management Methods - Claude Generated
    
    def _add_ollama_provider(self):
        """Add new Ollama provider - Claude Generated"""
        dialog = OllamaProviderEditorDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                new_provider = dialog.get_provider()
                
                # Check if provider name already exists
                if self.current_config.llm.get_ollama_provider_by_name(new_provider.name):
                    QMessageBox.warning(self, "Duplicate Name", 
                                      f"An Ollama provider with name '{new_provider.name}' already exists!")
                    return
                
                # Add provider to configuration
                self.current_config.llm.add_ollama_provider(new_provider)
                self._load_ollama_providers_list()
                
            except ValueError as e:
                QMessageBox.critical(self, "Invalid Provider", f"Error creating Ollama provider: {str(e)}")
    
    def _edit_ollama_provider(self):
        """Edit selected Ollama provider - Claude Generated"""
        current_item = self.ollama_providers_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select an Ollama provider to edit.")
            return
            
        provider = current_item.data(Qt.ItemDataRole.UserRole)
        dialog = OllamaProviderEditorDialog(parent=self, provider=provider)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                updated_provider = dialog.get_provider()
                
                # Check if name changed and conflicts with existing
                if (updated_provider.name != provider.name and 
                    self.current_config.llm.get_ollama_provider_by_name(updated_provider.name)):
                    QMessageBox.warning(self, "Duplicate Name", 
                                      f"An Ollama provider with name '{updated_provider.name}' already exists!")
                    return
                
                # Update provider in configuration
                self.current_config.llm.remove_ollama_provider(provider.name)
                self.current_config.llm.add_ollama_provider(updated_provider)
                self._load_ollama_providers_list()
                
            except ValueError as e:
                QMessageBox.critical(self, "Invalid Provider", f"Error updating Ollama provider: {str(e)}")
    
    def _delete_ollama_provider(self):
        """Delete selected Ollama provider - Claude Generated"""
        current_item = self.ollama_providers_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select an Ollama provider to delete.")
            return
            
        provider = current_item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self,
            "Delete Ollama Provider",
            f"Are you sure you want to delete the Ollama provider '{provider.name}'?\n\n"
            f"Host: {provider.host}:{provider.port}\n"
            f"Type: {provider.connection_type}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.current_config.llm.remove_ollama_provider(provider.name)
            self._load_ollama_providers_list()
    
    def _load_ollama_providers_list(self):
        """Load Ollama providers into the list widget - Claude Generated"""
        self.ollama_providers_list.clear()
        for provider in self.current_config.llm.ollama_providers:
            item_text = provider.display_name
            if provider.description:
                item_text += f" - {provider.description}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, provider)
            self.ollama_providers_list.addItem(item)
    
    def _test_ollama_connection(self):
        """Test connection for selected Ollama provider using async worker - Claude Generated"""
        current_item = self.ollama_providers_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select an Ollama provider to test.")
            return
            
        provider = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Disable test button to prevent multiple simultaneous tests
        self.ollama_test_btn.setEnabled(False)
        self.ollama_test_btn.setText("Testing...")
        
        # Create progress dialog (non-blocking)
        self.connection_progress = QProgressDialog(f"Testing {provider.name} connection...", "Cancel", 0, 0, self)
        self.connection_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.connection_progress.canceled.connect(self._cancel_connection_test)
        self.connection_progress.show()
        
        # Create and start async worker
        self.connection_worker = OllamaConnectionTestWorker(provider)
        self.connection_worker.test_completed.connect(self._on_connection_test_completed)
        self.connection_worker.progress_updated.connect(self._on_connection_progress_updated)
        self.connection_worker.start()
    
    def _cancel_connection_test(self):
        """Cancel ongoing connection test - Claude Generated"""
        if hasattr(self, 'connection_worker') and self.connection_worker:
            self.connection_worker.cancel()
            self.connection_worker.wait(1000)  # Wait up to 1 second for graceful shutdown
            if self.connection_worker.isRunning():
                self.connection_worker.terminate()  # Force terminate if needed
            self.connection_worker = None
        
        self._reset_connection_test_ui()
    
    def _on_connection_progress_updated(self, message: str):
        """Update progress dialog with current status - Claude Generated"""
        if hasattr(self, 'connection_progress') and self.connection_progress:
            self.connection_progress.setLabelText(message)
    
    def _on_connection_test_completed(self, success: bool, message: str, models: list):
        """Handle completed connection test - Claude Generated"""
        # Clean up progress dialog
        if hasattr(self, 'connection_progress') and self.connection_progress:
            self.connection_progress.close()
            self.connection_progress = None
        
        # Clean up worker
        if hasattr(self, 'connection_worker') and self.connection_worker:
            self.connection_worker.deleteLater()
            self.connection_worker = None
        
        # Reset UI
        self._reset_connection_test_ui()
        
        # Show results
        if success:
            provider = self.ollama_providers_list.currentItem().data(Qt.ItemDataRole.UserRole)
            
            if models:
                model_list = "\n".join(f"• {model}" for model in models[:15])
                if len(models) > 15:
                    model_list += f"\n... and {len(models) - 15} more models"
                
                QMessageBox.information(
                    self,
                    "✅ Connection Successful",
                    f"Ollama provider '{provider.name}' connection successful!\n\n"
                    f"Host: {provider.host}:{provider.port}\n"
                    f"Type: {provider.connection_type}\n"
                    f"SSL: {'Yes' if provider.use_ssl else 'No'}\n\n"
                    f"Available models ({len(models)} total):\n{model_list}"
                )
            else:
                QMessageBox.information(
                    self,
                    "✅ Connection Successful",
                    f"Ollama provider '{provider.name}' connection successful!\n\n"
                    f"Host: {provider.host}:{provider.port}\n"
                    f"No models available (server may be starting up)"
                )
        else:
            # Show error based on message content
            if "not installed" in message:
                QMessageBox.critical(
                    self,
                    "❌ Missing Dependency",
                    message
                )
            elif "Authentication failed" in message:
                QMessageBox.warning(
                    self,
                    "🔑 Authentication Required",
                    message
                )
            elif "doesn't support OpenAI-compatible API" in message:
                QMessageBox.warning(
                    self,
                    "⚠️ Server Not Compatible",
                    message
                )
            else:
                QMessageBox.warning(
                    self,
                    "⚠️ Connection Failed",
                    message
                )
    
    def _reset_connection_test_ui(self):
        """Reset connection test UI elements - Claude Generated"""
        self.ollama_test_btn.setEnabled(True)
        self.ollama_test_btn.setText("🔧 Test Selected Provider")


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = ComprehensiveSettingsDialog()
    dialog.show()
    sys.exit(app.exec())