#!/usr/bin/env python3
"""
Unified Provider Tab - Consolidates LLM Provider and Provider Preferences configuration
Replaces fragmented LLM + Provider Preferences tabs with a single, coherent interface.
Claude Generated
"""

from copy import deepcopy
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QDialog,
)
from PyQt6.QtCore import pyqtSignal

from ..utils.config_manager import ConfigManager, ProviderDetectionService
from ..utils.error_visibility import log_caught
from ..utils.config_models import (
    UnifiedProviderConfig,
    UnifiedProvider,
    AlimaConfig
)
from ..utils.model_capabilities import get_chunking_threshold  # For per-model chunking UI - Claude Generated
from .workers import ModelLoadWorker  # Shared model-list loader (F-7) - Claude Generated
from .provider_dialogs import ProviderEditDialog  # F-5 split - Claude Generated
from .settings import ProviderAccordion, ResolutionView


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

        # Providers must reach disk as soon as they change: the model list comes
        # from LlmService, whose client map and provider lookup both read the
        # *saved* config (llm_service.py:945 and :960). A provider that lives
        # only in this dialog's editing copy can therefore never load its models.
        # The snapshot is what "Cancel" restores. - Claude Generated
        self._providers_at_open = deepcopy(list(self.unified_config.providers))

        self._setup_ui()
        self._load_configuration()

        # Setup reactive provider status connections - Claude Generated
        self._setup_provider_status_connections()
    
    def _persist_providers(self) -> None:
        """Write the provider list to disk, and nothing else - Claude Generated.

        Deliberately not ``save_config(self.config)``: that object is the whole
        dialog's editing copy, so saving it would publish unsaved plugin, chat and
        system edits too — and ``save_config`` caches what it wrote in the
        ConfigManager singleton, putting them in front of the running application.
        Loading the saved config and replacing only ``providers`` keeps the write
        and the singleton limited to what actually changed.
        """
        try:
            saved = self.config_manager.load_config()
            saved.unified_config.providers = list(self.unified_config.providers)
            self.config_manager.save_config(saved)
        except Exception as e:
            self.logger.error(f"Provider konnten nicht gespeichert werden: {e}")

    def restore_providers_at_open(self) -> bool:
        """Undo provider add/edit/remove. Returns True if anything was written.

        Called by the settings dialog on Cancel. - Claude Generated
        """
        current = list(self.unified_config.providers)
        if [p.name for p in current] == [p.name for p in self._providers_at_open] and all(
            a == b for a, b in zip(current, self._providers_at_open)
        ):
            return False
        self.unified_config.providers = deepcopy(self._providers_at_open)
        self._persist_providers()
        return True

    def _setup_ui(self):
        """Setup the unified provider tab UI - Claude Generated"""
        layout = QVBoxLayout(self)

        # No tabs of its own: this widget is one page of the settings sidebar, and
        # a tab bar in here would put a second navigation level back on the page.
        # The "Task Preferences" tab that used to sit next to the providers is
        # gone with the per-task model choice it edited. - Claude Generated
        self.providers_tab = self._create_providers_tab()
        layout.addWidget(self.providers_tab)

        # Action buttons
        button_layout = QHBoxLayout()
        
        # "🧪 Test All Providers" sat here and called _refresh_models() verbatim —
        # two buttons, one action. The remaining one is next to the provider list,
        # where it belongs. - Claude Generated
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _create_providers_tab(self) -> QWidget:
        """Create provider management tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # First on the page on purpose: this is not an input but the consequence
        # of the ones below it. Which of the four defaults applies used to be
        # invisible. - Claude Generated
        self.resolution_view = ResolutionView()
        self.resolution_view.selection_changed.connect(self._on_scope_selection)
        layout.addWidget(self.resolution_view)

        # Providers as an accordion. Three to six entries with six fields each is
        # a list, not a table: a table has to squeeze the endpoint and the key
        # state into columns and then needs a dialog for whatever did not fit.
        # - Claude Generated
        self.provider_accordion = ProviderAccordion()
        self.provider_accordion.edit_requested.connect(self._edit_provider_by_name)
        self.provider_accordion.remove_requested.connect(self._remove_provider_by_name)
        self.provider_accordion.models_requested.connect(self._refresh_models_for)
        self.provider_accordion.chunking_changed.connect(self._set_chunking)
        self.provider_accordion.chunking_removed.connect(self._drop_chunking)
        layout.addWidget(self.provider_accordion)

        provider_button_layout = QHBoxLayout()
        self.add_provider_button = QPushButton("➕ Anbieter hinzufügen…")
        self.add_provider_button.clicked.connect(self._add_provider)
        self.refresh_models_button = QPushButton("🔄 Alle Anbieter prüfen")
        self.refresh_models_button.clicked.connect(self._refresh_models)
        provider_button_layout.addWidget(self.add_provider_button)
        provider_button_layout.addWidget(self.refresh_models_button)
        provider_button_layout.addStretch()
        layout.addLayout(provider_button_layout)

        return widget

    def _refresh_effective_defaults(self) -> None:
        """Re-render the resolution table from the current config - Claude Generated."""
        view = getattr(self, "resolution_view", None)
        if view is not None:
            view.refresh(self.unified_config)

    
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
            

            self._refresh_effective_defaults()

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
        """Render the provider accordion. Name kept: several call sites use it."""
        self.provider_accordion.refresh(
            list(self.unified_config.providers),
            models=dict(self._cached_models or {}),
            thresholds=dict(getattr(self.unified_config, "model_chunking_thresholds", {}) or {}),
        )

    # ---- accordion callbacks, addressed by provider name ----------------

    def _provider_index(self, name: str) -> int:
        for i, provider in enumerate(self.unified_config.providers):
            if provider.name == name:
                return i
        return -1

    def _edit_provider_by_name(self, name: str) -> None:
        index = self._provider_index(name)
        if index < 0:
            return
        dialog = ProviderEditDialog(self.unified_config.providers[index], parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.unified_config.providers[index] = dialog.get_provider_data()
            self._persist_providers()
            self._load_configuration()
            self.config_changed.emit()

    def _remove_provider_by_name(self, name: str) -> None:
        index = self._provider_index(name)
        if index < 0:
            return
        reply = QMessageBox.question(
            self, "Anbieter entfernen",
            f"Anbieter '{name}' entfernen?\n\n"
            "'Cancel' macht die Änderung rückgängig.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        del self.unified_config.providers[index]
        self._persist_providers()
        self._load_configuration()
        self.config_changed.emit()

    def _refresh_models_for(self, name: str) -> None:
        """Fetch one provider's model list on demand - Claude Generated.

        Per provider rather than for all of them: the dialog must not spend four
        network calls before it is usable, and a dead endpoint must not hang it.
        """
        try:
            models = self._detection_service.get_available_models(name, force_check=True)
            self._cached_models[name] = list(models or [])
            self._populate_provider_table()
            self._show_save_toast(f"{name}: {len(models or [])} Modelle")
        except Exception as e:
            log_caught(self.logger, e, f"Modell-Abfrage für {name}")
            self._show_save_toast(f"{name}: Abfrage fehlgeschlagen", error=True)

    def _set_chunking(self, provider: str, model: str, value: int) -> None:
        if not model:
            return
        self.unified_config.set_chunking_threshold(provider, model, int(value))
        self.config_changed.emit()

    def _drop_chunking(self, provider: str, model: str) -> None:
        self.unified_config.remove_chunking_threshold(provider, model)
        self._populate_provider_table()
        self.config_changed.emit()

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

    #: Which config fields a scope owns. - Claude Generated
    SCOPE_FIELDS = {
        "general": ("preferred_provider", "preferred_model"),
        "pipeline": ("pipeline_default_provider", "pipeline_default_model"),
        "agentic": ("agentic_default_provider", "agentic_default_model"),
        "vision": ("vision_default_provider", "vision_default_model"),
    }

    def _populate_global_preferences(self):
        """Hand the enabled providers to the pickers inside the table.

        The three separate "General / Pipeline / Agentic Default" rows that used
        to sit below the provider list are gone: they offered the same four values
        the table already shows, once as input and once as result. The table is
        now both. - Claude Generated
        """
        # The table derives the provider list itself on refresh; pushing it from
        # here ran before the pickers existed. - Claude Generated
        self._refresh_effective_defaults()

    def _on_scope_selection(self, scope: str, provider: str, model: str) -> None:
        """A row's picker changed → write that scope's own default. - Claude Generated"""
        fields = self.SCOPE_FIELDS.get(scope)
        if not fields:
            return
        setattr(self.unified_config, fields[0], provider or "")
        setattr(self.unified_config, fields[1], model or "")
        self._refresh_effective_defaults()
        self.config_changed.emit()

    def _on_default_selection_changed(self):
        """A default provider/model selector was changed by the user → write it to
        config and mark dirty. Programmatic population is silent (the selector only
        emits on user edits), so this never fires during load. Claude Generated."""
        self._update_config_from_ui()
        self._refresh_effective_defaults()
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
            
            self.unified_config.providers.append(new_provider)
            self._persist_providers()
            self._load_configuration()
            self.config_changed.emit()
            
            self.logger.info(f"Added new provider: {new_provider.name} ({new_provider.provider_type})")
    
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
        # Harvest the pickers inside the resolution table. Each row writes its own
        # scope live via _on_scope_selection; this is the safety net for a pick
        # that never emitted. The general default must stay concrete — everything
        # else falls back to it — so an empty read is never written there. A
        # transient empty read used to wipe preferred_provider while the other
        # scopes (where "" legitimately means "inherit") survived.
        # - Claude Generated
        for scope, (prov_field, model_field) in self.SCOPE_FIELDS.items():
            selector = self.resolution_view._selectors.get(scope)
            if selector is None:
                continue
            prov, model = selector.get_selection()
            if scope == "general" and not prov:
                continue
            setattr(self.unified_config, prov_field, prov or "")
            setattr(self.unified_config, model_field, model or "")


        # NOTE: Task preferences are managed in self.config.unified_config.task_preferences
        # They are already updated by individual UI operations
        # No additional UI → config sync needed here
