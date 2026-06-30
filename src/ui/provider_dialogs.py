#!/usr/bin/env python3
"""Provider configuration dialogs — Claude Generated.

Extracted from ``unified_provider_tab.py`` (F-5 god-file split, June 30) so the
tab module holds only the tab itself:

* :class:`TaskModelSelectionDialog` — pick provider + model for one pipeline task
  (async model detection via the shared ``ModelLoadWorker``).
* :class:`ProviderEditDialog` — add/edit a single LLM provider's connection config.

Both are self-contained ``QDialog``s (no coupling to the tab beyond the Qt parent);
the tab instantiates them and reads results via their getters.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
)

from ..utils.config_manager import ProviderDetectionService
from ..utils.config_models import UnifiedProvider
from .workers import ModelLoadWorker


class TaskModelSelectionDialog(QDialog):
    """Dialog for selecting provider and model for task-specific preferences - Claude Generated"""

    def __init__(self, config_manager, task_name=None, current_model_info=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.task_name = task_name
        self.current_model_info = current_model_info or {}
        self._pending_cache_update = None  # Stores (provider_name, models) for later save
        self.status_label = None  # Will be set in _setup_ui
        # Async model detection (F-7): keep strong refs to in-flight workers until
        # they finish, and remember the provider whose load is current so late
        # results for a switched-away provider are ignored. - Claude Generated
        self._model_workers = set()
        self._pending_provider = None

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
        """Kick off async model detection (two-tier: live → cached) - Claude Generated

        TIER 1 (live detection) runs off the UI thread via ModelLoadWorker so
        switching the provider no longer freezes the dialog; the result is handled
        in :meth:`_on_models_loaded`, which falls back to TIER 2 (cached) when the
        live list is empty.
        """
        self.model_combo.clear()
        self.model_combo.setEnabled(True)

        if not provider_name or provider_name == "No providers available":
            self._pending_provider = None
            return

        # Loading placeholder while the (network) detection runs in the background.
        self._pending_provider = provider_name
        self.model_combo.addItem("⏳ Lade Modelle…", None)
        self.model_combo.setEnabled(False)
        if self.status_label:
            self.status_label.setText(f"⏳ Detecting models from {provider_name}…")

        detection_service = ProviderDetectionService(self.config_manager)
        worker = ModelLoadWorker(detection_service, provider_name, force=True)
        worker.fetched.connect(self._on_models_loaded)
        worker.finished.connect(lambda w=worker: self._retire_model_worker(w))
        self._model_workers.add(worker)
        worker.start()

    def _retire_model_worker(self, worker) -> None:
        """Drop the strong ref once the worker thread has finished - Claude Generated"""
        self._model_workers.discard(worker)
        worker.deleteLater()

    def _on_models_loaded(self, provider_name: str, detected_models: list):
        """Populate the model combo when detection finishes (main thread) - Claude Generated

        TIER 2 cached fallback runs here when the live list is empty. Late results
        for a provider the user has since switched away from are ignored.
        """
        if provider_name != self._pending_provider:
            return

        self.model_combo.clear()
        self.model_combo.setEnabled(True)

        models_to_display = []
        detection_success = False

        # TIER 1 result (live detection, fetched off-thread)
        if detected_models:
            models_to_display = sorted(detected_models, key=lambda s: s.lower())
            detection_success = True
            if self.status_label:
                self.status_label.setText(f"✅ {len(models_to_display)} models detected from {provider_name}")

        # TIER 2: cached models from config (if live detection returned nothing)
        if not models_to_display and self.config_manager:
            cached_models = self._get_cached_models_from_config(provider_name)
            if cached_models:
                models_to_display = sorted(cached_models, key=lambda s: s.lower())
                if self.status_label:
                    self.status_label.setText(f"⚠️ Provider offline - showing {len(models_to_display)} cached models")

        # Populate dropdown (no visual distinction between detected and cached)
        if models_to_display:
            for model in models_to_display:
                self.model_combo.addItem(model, model)
            self._preselect_current_model()
        else:
            self.model_combo.addItem("No models available", None)
            self.model_combo.setEnabled(False)
            if self.status_label:
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
        except Exception as e:
            self.logger.warning(f"Could not read cached models for provider: {e}")

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
        except Exception as e:
            self.logger.warning(f"Could not persist models for provider '{provider_name}': {e}")

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
