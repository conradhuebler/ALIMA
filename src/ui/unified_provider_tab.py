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
    TaskPreference,
    TaskType as UnifiedTaskType,
    AlimaConfig
)
from ..utils.model_capabilities import get_chunking_threshold  # For per-model chunking UI - Claude Generated
from .workers import ModelLoadWorker  # Shared model-list loader (F-7) - Claude Generated
from .provider_dialogs import TaskModelSelectionDialog, ProviderEditDialog  # F-5 split - Claude Generated


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
