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
    QSplitter, QFrame, QScrollArea, QGridLayout, QDialog,
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPalette

from ..utils.config_manager import ConfigManager, OpenAICompatibleProvider, OllamaProvider, ProviderDetectionService
from ..utils.config_models import (
    UnifiedProviderConfig,
    UnifiedProvider,
    AlimaConfig
)
from ..utils.model_capabilities import get_chunking_threshold  # For per-model chunking UI - Claude Generated
from .workers import ModelLoadWorker  # Shared model-list loader (F-7) - Claude Generated
from .provider_dialogs import TaskModelSelectionDialog, ProviderEditDialog  # F-5 split - Claude Generated
from .task_preferences_widget import TaskPreferencesWidget  # F-5 split - Claude Generated


class UnifiedProviderTab(QWidget):
    """
    Unified Provider Configuration Tab - Claude Generated
    Consolidates LLM Provider and Provider Preferences into single interface
    """

    config_changed = pyqtSignal()
    task_preferences_changed = pyqtSignal()  # New signal for task preference changes - Claude Generated

    
    def __init__(self, unified_config: UnifiedProviderConfig, alima_config: AlimaConfig,
                 config_manager: ConfigManager, alima_manager=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Store all dependencies - Claude Generated
        self.unified_config = unified_config
        self.config = alima_config
        self.config_manager = config_manager  # ✅ ConfigManager for persistence operations
        self.alima_manager = alima_manager  # Access to ProviderStatusService - Claude Generated
        # Task-preference editing state now lives in TaskPreferencesWidget (F-5).

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
        self.preferences_widget = TaskPreferencesWidget(
            self.unified_config, self.config, self.config_manager, self.alima_manager, self
        )
        self.preferences_widget.config_changed.connect(self.config_changed)
        self.preferences_widget.task_preferences_changed.connect(self.task_preferences_changed)
        self.preferences_widget.save_toast.connect(
            lambda msg, err: self._show_save_toast(msg, error=err)
        )
        self.preferences_tab = self.preferences_widget
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

        # Default provider/model rows — shared ProviderModelSelector widgets.
        from .provider_model_selector import ProviderModelSelector

        provider_label_width = 120

        def _make_selector_row(label_text: str, selector: "ProviderModelSelector", tooltip: str):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(provider_label_width)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl)
            selector.setToolTip(tooltip)
            row.addWidget(selector, 1)
            # Live writeback on USER changes only. The selector stays silent during
            # programmatic population (set_selection/set_providers), so the immediate-
            # save architecture persists the user's pick without a late async
            # selectionChanged clobbering it. Save-time harvest is the safety net.
            selector.selectionChanged.connect(lambda *_: self._on_default_selection_changed())
            layout.addLayout(row)

        # General default — central fallback for pipeline & chat (always concrete).
        self.preferred_selector = ProviderModelSelector()
        _make_selector_row(
            "General Default:", self.preferred_selector,
            "Central default provider/model. Used as fallback when no "
            "pipeline-specific or agentic-specific default is set.")

        # Pipeline default — overrides the general default (may be left unset).
        self.pipeline_default_selector = ProviderModelSelector(
            allow_empty=True, empty_provider_label="(Use general default)")
        _make_selector_row(
            "Pipeline Default:", self.pipeline_default_selector,
            "Default provider/model for the classic pipeline. "
            "Leave empty to use the general default.")

        # Agentic default — overrides the pipeline default (may be left unset).
        self.agentic_default_selector = ProviderModelSelector(
            allow_empty=True, empty_provider_label="(Use pipeline default)")
        _make_selector_row(
            "Agentic Default:", self.agentic_default_selector,
            "Default provider/model for agentic workflows and the chat agent. "
            "Leave empty to use the pipeline default.")

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
            
            # Load task preferences (delegated to TaskPreferencesWidget) - Claude Generated
            self.preferences_widget.set_available_models(self._cached_providers, self._cached_models)
            self.preferences_widget.load()

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
        self._model_fetch_worker = ModelLoadWorker(self._detection_service, list(self._cached_providers))
        self._model_fetch_worker.models_fetched.connect(self._on_models_fetched)
        self._model_fetch_worker.start()
        self.logger.debug("Started background model fetch for all providers")

    def _on_models_fetched(self, models: dict):
        """Slot called when background model fetch completes - Claude Generated"""
        try:
            self._cached_models.update(models)
            self._populate_model_preferences()
            self.preferences_widget.set_available_models(self._cached_providers, self._cached_models)
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
        """Populate the default-provider/model selectors from config - Claude Generated"""
        provider_names = [p.name for p in self.unified_config.providers if p.enabled]
        uc = self.unified_config
        for selector in (self.preferred_selector,
                         self.pipeline_default_selector,
                         self.agentic_default_selector):
            selector.set_providers(provider_names, refresh=False)
        # set_selection triggers the single model load per selector (no throwaway
        # load for the index-0 provider first). The shared selector loads models
        # live (cached); pipeline/agentic keep the "(Use … default)" placeholder.
        self.preferred_selector.set_selection(uc.preferred_provider, uc.preferred_model)
        self.pipeline_default_selector.set_selection(
            uc.pipeline_default_provider, uc.pipeline_default_model)
        self.agentic_default_selector.set_selection(
            uc.agentic_default_provider, uc.agentic_default_model)

    def _on_default_selection_changed(self):
        """A default provider/model selector was changed by the user → write it to
        config and mark dirty. Programmatic population is silent (the selector only
        emits on user edits), so this never fires during load. Claude Generated."""
        self._update_config_from_ui()
        self.config_changed.emit()

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
        # Update global preferences from the shared selectors. The pipeline/agentic
        # selectors return ("", "") when their "(Use … default)" placeholder is
        # chosen, i.e. unset = fall back to the wider default.
        prov, model = self.preferred_selector.get_selection()
        # General default is always a concrete provider (no placeholder). Never
        # clobber it with a transient empty read (e.g. mid-repopulation triggered
        # by another selector's change) — that silently wiped preferred_provider
        # while pipeline/agentic (where "" is a valid "unset") survived.
        if prov:
            self.unified_config.preferred_provider = prov
            self.unified_config.preferred_model = model

        prov, model = self.pipeline_default_selector.get_selection()
        self.unified_config.pipeline_default_provider = prov
        self.unified_config.pipeline_default_model = model

        prov, model = self.agentic_default_selector.get_selection()
        self.unified_config.agentic_default_provider = prov
        self.unified_config.agentic_default_model = model

        # Task preferences save delegated to TaskPreferencesWidget (clean-state guarded) - Claude Generated
        self.preferences_widget.save_if_clean()

        # NOTE: Task preferences are managed in self.config.unified_config.task_preferences
        # They are already updated by individual UI operations
        # No additional UI → config sync needed here
