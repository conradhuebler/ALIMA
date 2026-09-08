#!/usr/bin/env python3
"""Per-step pipeline config widgets — Claude Generated.

Extracted from ``pipeline_config_dialog.py`` (F-5 god-file split, June 30):

* :class:`SearchStepConfigWidget` — config for the GND search step.
* :class:`HybridStepConfigWidget` — Smart/Advanced/Expert per-step provider+model
  config (async model detection via the shared ``ModelLoadWorker``).

Self-contained ``QWidget``s; ``PipelineConfigDialog`` instantiates them and
reads/writes their config. No coupling back to the dialog.
"""

import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QCompleter, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QTextEdit,
    QVBoxLayout, QWidget,
)

from ..utils.config_models import PipelineStepConfig, TaskType as UnifiedTaskType
from ..utils.smart_provider_selector import SmartProviderSelector
from ..utils.pipeline_config_parser import PipelineConfigParser
from .workers import ModelLoadWorker

# Baseline-source styling constants (moved with HybridStepConfigWidget) - Claude Generated
STYLE_TASK_PREFERENCE = "background-color: #e8f5e9; color: #2e7d32; font-weight: bold;"
STYLE_PROVIDER_PREFERENCE = "background-color: #e3f2fd; color: #1976d2;"
STYLE_OVERRIDE = ""  # Default styling


class SearchStepConfigWidget(QWidget):
    """Widget für die Konfiguration des GND-Suchschritts - Claude Generated"""

    def __init__(self, step_name: str, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.step_id = "search"
        self.setup_ui()

    def setup_ui(self):
        """Setup der UI für Search-Konfiguration - Claude Generated"""
        layout = QVBoxLayout(self)

        # Step Name Header
        header_label = QLabel(self.step_name)
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Suggester Selection
        suggester_group = QGroupBox("Suchprovider")
        suggester_layout = QVBoxLayout(suggester_group)

        # Available suggesters
        self.lobid_checkbox = QCheckBox("Lobid (Deutsche Nationalbibliothek)")
        self.lobid_checkbox.setChecked(True)
        suggester_layout.addWidget(self.lobid_checkbox)

        self.swb_checkbox = QCheckBox("SWB (Südwestdeutscher Bibliotheksverbund)")
        self.swb_checkbox.setChecked(True)
        suggester_layout.addWidget(self.swb_checkbox)

        self.catalog_checkbox = QCheckBox("Lokaler Katalog")
        self.catalog_checkbox.setChecked(False)
        suggester_layout.addWidget(self.catalog_checkbox)

        layout.addWidget(suggester_group)

        # Enable/Disable for this step
        self.enabled_checkbox = QCheckBox("Schritt aktivieren")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        suggester_layout.addWidget(self.enabled_checkbox)

        layout.addStretch()

    def on_enabled_changed(self, enabled: bool):
        """Enable/disable step configuration - Claude Generated"""
        for widget in self.findChildren(QWidget):
            if widget != self.enabled_checkbox:
                widget.setEnabled(enabled)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration - Claude Generated"""
        suggesters = []
        if self.lobid_checkbox.isChecked():
            suggesters.append("lobid")
        if self.swb_checkbox.isChecked():
            suggesters.append("swb")
        if self.catalog_checkbox.isChecked():
            suggesters.append("catalog")

        return {
            "step_id": self.step_id,
            "enabled": self.enabled_checkbox.isChecked(),
            "suggesters": suggesters,
        }

    def set_config(self, config: Dict[str, Any]):
        """Set configuration - Claude Generated"""
        if "enabled" in config:
            self.enabled_checkbox.setChecked(config["enabled"])

        if "suggesters" in config:
            suggesters = config["suggesters"]
            self.lobid_checkbox.setChecked("lobid" in suggesters)
            self.swb_checkbox.setChecked("swb" in suggesters)
            self.catalog_checkbox.setChecked("catalog" in suggesters)


class HybridStepConfigWidget(QWidget):
    """
    Hybrid Mode Step Configuration Widget - Claude Generated
    Supports Smart/Advanced/Expert modes for pipeline step configuration
    """
    
    config_changed = pyqtSignal()
    
    def __init__(self, step_name: str, step_id: str, 
                 config_manager=None, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.step_id = step_id
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        # Async model detection (F-7): strong refs to in-flight workers + the
        # provider whose load is current + an optional model to restore once
        # populated (for refresh / config-load). - Claude Generated
        self._model_workers = set()
        self._pending_provider = None
        self._pending_restore_model = None
        
        # Initialize with default step config - Claude Generated
        self.step_config = PipelineStepConfig(
            step_id=step_id,
            task_type=self._get_default_task_type(step_id)
        )
        
        self.setup_ui()
        self._update_task_type_display()
        self._update_ui_for_mode()

        # Initialize with preferred provider/model from settings after UI is set up - Claude Generated
        self._initialize_with_preferred_settings()

        # Update smart mode preview after initialization - Claude Generated
        self._update_smart_preview()

    def _update_task_type_display(self):
        """Update task type display with auto-derived type - Claude Generated"""
        task_type_text = self.step_config.task_type.value.title().replace('_', ' ') if self.step_config.task_type else "General"
        self.task_type_label.setText(f"{task_type_text} (automatic)")
        self.task_type_label.setToolTip(f"Task type automatically derived from step ID: {self.step_id}")

    def _get_available_tasks_for_step(self) -> List[str]:
        """Get appropriate task options for the current pipeline step - Claude Generated

        Uses unified PipelineConfigParser for consistent validation across CLI and GUI
        """
        # Use unified parser for consistent step-aware task selection
        parser = PipelineConfigParser()
        valid_tasks = parser.get_valid_tasks_for_step(self.step_id)

        # If no tasks defined for this step, provide fallback
        if not valid_tasks:
            # Fallback to common tasks if step is not in mapping
            valid_tasks = ["keywords"]
            self.logger.debug(f"Step '{self.step_id}' not in parser mapping, using fallback task: {valid_tasks}")

        return valid_tasks

    def _populate_task_combo(self):
        """Populate task combo with step-appropriate tasks and auto-select - Claude Generated"""
        # Get available tasks for this step
        available_tasks = self._get_available_tasks_for_step()

        # Clear and populate combo box
        self.task_combo.clear()
        if not available_tasks:
            self.task_combo.addItem("No tasks available")
            self.task_combo.setEnabled(False)
            return

        # Add available tasks
        self.task_combo.addItems(available_tasks)
        self.task_combo.setEnabled(True)

        # Auto-select the most appropriate task for this step
        preferred_task = self._get_preferred_task_for_step()
        if preferred_task and preferred_task in available_tasks:
            index = self.task_combo.findText(preferred_task)
            if index >= 0:
                self.task_combo.setCurrentIndex(index)
                self.logger.info(f"Auto-selected task '{preferred_task}' for step '{self.step_id}'")
        else:
            # Fallback to first available task
            if available_tasks:
                self.task_combo.setCurrentIndex(0)
                self.logger.info(f"Fallback selected task '{available_tasks[0]}' for step '{self.step_id}'")

    def _get_preferred_task_for_step(self) -> str:
        """Get the preferred/default task for the current step - Claude Generated"""
        # Map steps to their most logical default tasks
        step_preferred_task = {
            "initialisation": "initialisation",
            "keywords": "keywords",
            "classification": "classification",
            "dk_classification": "dk_classification",
            "image_text_extraction": "image_text_extraction"
        }

        return step_preferred_task.get(self.step_id, "keywords")

    def _load_task_preferences_direct(self) -> tuple[Optional[str], Optional[str], str]:
        """
        Load task preferences directly from config.unified_config.task_preferences - Claude Generated
        Returns: (provider_name, model_name, selection_reason)
        """
        if not self.config_manager:
            return None, None, "no config manager"

        try:
            # Load current config with force refresh to ensure latest data - Claude Generated
            config = self.config_manager.load_config(force_reload=True)
            if not config:
                return None, None, "no config loaded"
            if not hasattr(config, 'task_preferences'):
                return None, None, "config has no task_preferences attribute"
            if not config.unified_config.task_preferences:
                return None, None, "task_preferences is empty"

            # CRITICAL DEBUG: Log available task preferences - Claude Generated
            available_tasks = list(config.unified_config.task_preferences.keys())
            self.logger.info(f"🔍 TASK_PREFS_AVAILABLE: {available_tasks} for step_id '{self.step_id}'")

            # Map step_id to task name for task_preferences lookup - Claude Generated
            task_name_mapping = {
                "initialisation": "initialisation",
                "keywords": "keywords",
                "classification": "classification",
                "dk_classification": "dk_classification",  # FIXED: Match the actual key in config
                "image_text_extraction": "image_text_extraction"
            }

            task_name = task_name_mapping.get(self.step_id)
            if not task_name:
                return None, None, f"no task mapping for step '{self.step_id}'"

            # Get task preferences from config
            task_data = config.unified_config.task_preferences.get(task_name)
            model_priority = task_data.model_priority if task_data else []

            # CRITICAL DEBUG: Log task preference lookup - Claude Generated
            self.logger.info(f"🔍 TASK_PREF_LOOKUP: step_id='{self.step_id}' -> task_name='{task_name}' -> found={task_name in config.unified_config.task_preferences}")
            if task_name in config.unified_config.task_preferences:
                self.logger.info(f"🔍 TASK_PREF_DATA: {task_data}")

            if not model_priority:
                return None, None, f"no model_priority for task '{task_name}' (task_data: {task_data})"

            # Get provider detection service for availability checking
            try:
                from ..utils.config_manager import ProviderDetectionService
                detection_service = ProviderDetectionService(self.config_manager)
            except Exception as e:
                self.logger.warning(f"Could not initialize provider detection service: {e}")
                detection_service = None

            # Try each model in priority order
            for rank, priority_entry in enumerate(model_priority, 1):
                candidate_provider = priority_entry.get("provider_name")
                candidate_model = priority_entry.get("model_name")

                if not candidate_provider or not candidate_model:
                    continue

                # Check if provider is available
                if detection_service:
                    try:
                        available_providers = detection_service.get_available_providers()
                        if candidate_provider not in available_providers:
                            self.logger.debug(f"Provider '{candidate_provider}' not available for {task_name}")
                            continue

                        # Check if model is available
                        available_models = detection_service.get_available_models(candidate_provider)
                        if candidate_model in available_models:
                            # Exact match found
                            selection_reason = f"task preference #{rank} ({task_name})"
                            self.logger.info(f"✅ Direct task preference match: {candidate_provider}/{candidate_model} - {selection_reason}")
                            return candidate_provider, candidate_model, selection_reason

                        # Try fuzzy matching
                        fuzzy_match = self._find_fuzzy_model_match(candidate_model, available_models)
                        if fuzzy_match:
                            selection_reason = f"task preference #{rank} (fuzzy: '{candidate_model}' → '{fuzzy_match}')"
                            self.logger.info(f"✅ Fuzzy task preference match: {candidate_provider}/{fuzzy_match} - {selection_reason}")
                            return candidate_provider, fuzzy_match, selection_reason

                        self.logger.debug(f"Model '{candidate_model}' not available in {candidate_provider}")

                    except Exception as e:
                        self.logger.warning(f"Error checking availability for {candidate_provider}/{candidate_model}: {e}")
                        continue
                else:
                    # No detection service available, return first preference
                    selection_reason = f"task preference #{rank} (unchecked)"
                    self.logger.info(f"📝 Using unchecked task preference: {candidate_provider}/{candidate_model}")
                    return candidate_provider, candidate_model, selection_reason

            # No usable preferences found
            return None, None, f"no usable preferences in {len(model_priority)} entries for '{task_name}'"

        except Exception as e:
            self.logger.error(f"Error loading task preferences directly: {e}")
            return None, None, f"error: {str(e)}"
    
    def _get_default_task_type(self, step_id: str) -> UnifiedTaskType:
        """Get default task type for pipeline step - Claude Generated"""
        task_mapping = {
            "input": UnifiedTaskType.INPUT,
            "initialisation": UnifiedTaskType.INITIALISATION,
            "search": UnifiedTaskType.SEARCH,
            "keywords": UnifiedTaskType.KEYWORDS,
            "classification": UnifiedTaskType.CLASSIFICATION,
            "dk_search": UnifiedTaskType.DK_SEARCH,
            "dk_classification": UnifiedTaskType.DK_CLASSIFICATION,
            "image_text_extraction": UnifiedTaskType.VISION
        }
        return task_mapping.get(step_id, UnifiedTaskType.GENERAL)
    
    def setup_ui(self):
        """Setup the hybrid mode UI - Claude Generated"""
        layout = QVBoxLayout(self)
        
        # Step Header
        header_label = QLabel(f"📋 {self.step_name}")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        # Note: Mode selection removed in baseline + override architecture
        # Smart configuration is always the baseline, Advanced/Expert are override editors
        
        # Smart Mode Configuration
        self.smart_group = QGroupBox("🤖 Smart Configuration")
        smart_layout = QGridLayout(self.smart_group)
        
        # Task Type (readonly - auto-derived from step_id)
        smart_layout.addWidget(QLabel("Task Type:"), 0, 0)
        self.task_type_label = QLabel()
        self.task_type_label.setStyleSheet("color: #666; font-style: italic;")
        smart_layout.addWidget(self.task_type_label, 0, 1)

        # Smart preview
        self.smart_preview_label = QLabel("🎯 Will auto-select optimal provider/model")
        self.smart_preview_label.setStyleSheet("color: #666; font-style: italic;")
        smart_layout.addWidget(self.smart_preview_label, 1, 0, 1, 2)

        # Edit Preferences Button (only shown when task preferences are detected) - Claude Generated
        self.edit_preferences_button = QPushButton("⚙️ Edit Task Preferences")
        self.edit_preferences_button.setStyleSheet("""
            QPushButton {
                background-color: #e3f2fd;
                border: 1px solid #2196f3;
                border-radius: 4px;
                padding: 6px 12px;
                color: #1976d2;
            }
            QPushButton:hover {
                background-color: #bbdefb;
            }
        """)
        self.edit_preferences_button.clicked.connect(self._open_task_preferences)
        self.edit_preferences_button.setVisible(False)  # Hidden by default
        smart_layout.addWidget(self.edit_preferences_button, 2, 0, 1, 2)

        # Color Legend for Baseline Settings - Claude Generated
        legend_label = QLabel(
            "📋 <b>Configuration Legend:</b><br>"
            "<span style='color: #2e7d32; background-color: #e8f5e9; padding: 2px 4px;'>🟢 Task Preference</span> = "
            "From Settings → Task Preferences (highest priority)<br>"
            "<span style='color: #1976d2; background-color: #e3f2fd; padding: 2px 4px;'>🔵 Provider Default</span> = "
            "From Settings → Provider Settings<br>"
            "<span style='padding: 2px 4px;'>⚪ Override/Default</span> = Manual selection or system default"
        )
        legend_label.setWordWrap(True)
        legend_label.setStyleSheet("color: #666; margin-top: 8px; padding: 8px; background-color: #f5f5f5; border-radius: 4px;")
        smart_layout.addWidget(legend_label, 3, 0, 1, 2)

        layout.addWidget(self.smart_group)
        
        # Manual Configuration (Advanced/Expert)
        self.manual_group = QGroupBox("⚙️ Manual Configuration")
        manual_layout = QGridLayout(self.manual_group)
        
        # Provider Selection
        manual_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        manual_layout.addWidget(self.provider_combo, 0, 1)
        
        # Model Selection
        manual_layout.addWidget(QLabel("Model:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_manual_config_changed)
        manual_layout.addWidget(self.model_combo, 1, 1)
        self.model_combo.setEditable(True)
        from PyQt6.QtWidgets import QCompleter
        self.model_combo.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)

        # Task/Prompt Selection
        manual_layout.addWidget(QLabel("Prompt Task:"), 2, 0)
        self.task_combo = QComboBox()
        self._populate_task_combo()  # Populate with step-appropriate tasks - Claude Generated
        self.task_combo.currentTextChanged.connect(self._on_manual_config_changed)
        manual_layout.addWidget(self.task_combo, 2, 1)
        
        layout.addWidget(self.manual_group)
        
        # Expert Parameters (only visible in Expert mode)
        self.expert_group = QGroupBox("🔬 Expert Parameters")
        expert_layout = QGridLayout(self.expert_group)
        
        # Temperature
        expert_layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setRange(0.0, 2.0)
        self.temperature_spinbox.setSingleStep(0.1)
        self.temperature_spinbox.setValue(0.7)
        self.temperature_spinbox.valueChanged.connect(self._on_expert_config_changed)
        expert_layout.addWidget(self.temperature_spinbox, 0, 1)
        
        # Top-P
        expert_layout.addWidget(QLabel("Top-P:"), 1, 0)
        self.top_p_spinbox = QDoubleSpinBox()
        self.top_p_spinbox.setRange(0.0, 1.0)
        self.top_p_spinbox.setSingleStep(0.05)
        self.top_p_spinbox.setValue(0.1)
        self.top_p_spinbox.valueChanged.connect(self._on_expert_config_changed)
        expert_layout.addWidget(self.top_p_spinbox, 1, 1)
        
        # Max Tokens
        expert_layout.addWidget(QLabel("Max Tokens:"), 2, 0)
        self.max_tokens_spinbox = QSpinBox()
        self.max_tokens_spinbox.setRange(1, 8192)
        self.max_tokens_spinbox.setValue(2048)
        self.max_tokens_spinbox.valueChanged.connect(self._on_expert_config_changed)
        expert_layout.addWidget(self.max_tokens_spinbox, 2, 1)

        # Thinking Mode (Ollama/OpenAI-compat think flag) - Claude Generated
        expert_layout.addWidget(QLabel("Thinking Mode:"), 3, 0)
        self.think_checkbox = QCheckBox("Enable thinking (think=true)")
        self.think_checkbox.setTristate(True)
        self.think_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)  # PartiallyChecked = not set (use default)
        self.think_checkbox.setToolTip(
            "Controls thinking/CoT mode for Ollama and compatible providers.\n"
            "Partially checked (─) = use provider default (think not sent)\n"
            "Checked (✓) = think=true\n"
            "Unchecked (☐) = think=false"
        )
        self.think_checkbox.stateChanged.connect(self._on_expert_config_changed)
        expert_layout.addWidget(self.think_checkbox, 3, 1)

        layout.addWidget(self.expert_group)

        # Expert Mode Prompt Editing (only visible in Expert mode) - Claude Generated
        self.prompt_editing_group = QGroupBox("📝 Prompt Editing")
        prompt_layout = QVBoxLayout(self.prompt_editing_group)

        # System Prompt Editor
        prompt_layout.addWidget(QLabel("System Prompt:"))
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setMaximumHeight(100)
        self.system_prompt_edit.setPlaceholderText("System prompt will be loaded from prompts.json...")
        self.system_prompt_edit.textChanged.connect(self._on_expert_config_changed)
        prompt_layout.addWidget(self.system_prompt_edit)

        # User Prompt Editor
        prompt_layout.addWidget(QLabel("User Prompt:"))
        self.user_prompt_edit = QTextEdit()
        self.user_prompt_edit.setMaximumHeight(100)
        self.user_prompt_edit.setPlaceholderText("User prompt template will be loaded from prompts.json...")
        self.user_prompt_edit.textChanged.connect(self._on_expert_config_changed)
        prompt_layout.addWidget(self.user_prompt_edit)

        layout.addWidget(self.prompt_editing_group)

        # Testing and Validation
        test_layout = QHBoxLayout()
        
        self.validate_button = QPushButton("🔍 Validate Configuration")
        self.validate_button.clicked.connect(self._validate_configuration)
        
        self.test_button = QPushButton("🧪 Test Configuration")
        self.test_button.clicked.connect(self._test_configuration)
        
        test_layout.addWidget(self.validate_button)
        test_layout.addWidget(self.test_button)
        test_layout.addStretch()
        
        layout.addLayout(test_layout)
        
        # Status/Results
        self.status_label = QLabel("✅ Configuration ready")
        self.status_label.setStyleSheet("color: green;")
        layout.addWidget(self.status_label)
        
        # Provider/model combos will be initialized after preferred settings are loaded
        
        # Add refresh button for providers
        refresh_layout = QHBoxLayout()
        refresh_button = QPushButton("🔄 Refresh Providers")
        refresh_button.clicked.connect(self._refresh_providers)
        refresh_layout.addWidget(refresh_button)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)
    
    
    def _update_ui_for_mode(self):
        """Update UI visibility - all groups always visible in baseline + override architecture - Claude Generated"""
        # In new architecture: all groups are always visible
        # Smart group shows baseline (read-only), Advanced/Expert are override editors
        self.smart_group.setVisible(True)      # Baseline display
        if hasattr(self, 'manual_group'):
            self.manual_group.setVisible(True)    # Override editor
        if hasattr(self, 'expert_group'):
            self.expert_group.setVisible(True)    # Override editor
        if hasattr(self, 'prompt_editing_group'):
            self.prompt_editing_group.setVisible(True)  # Override editor

        # Update status
        self.status_label.setText("📋 Baseline + Override Configuration")
        self.status_label.setStyleSheet("color: blue;")
    
    def _populate_providers(self):
        """Populate provider and model combos - Claude Generated"""
        if not self.config_manager:
            return
        
        try:
            from ..utils.config_manager import ProviderDetectionService
            detection_service = ProviderDetectionService(self.config_manager)
            providers = detection_service.get_available_providers()
            
            self.provider_combo.clear()
            self.provider_combo.addItems(providers)
            
            if providers:
                # Priority: use provider from step_config if already set, otherwise from settings - Claude Generated
                provider_to_select = None
                
                # 1. Check if step_config already has a provider set
                current_step_provider = getattr(self.step_config, 'provider', None)
                if current_step_provider and current_step_provider in providers:
                    provider_to_select = current_step_provider
                    self.logger.info(f"Using provider from step config: {current_step_provider}")
                else:
                    # 2. Try to use preferred provider from settings
                    preferred_provider = self._get_preferred_provider_from_settings()
                    if preferred_provider and preferred_provider in providers:
                        provider_to_select = preferred_provider
                        self.logger.info(f"Auto-selected preferred provider from settings: {preferred_provider}")
                    else:
                        # 3. Fallback to first provider
                        provider_to_select = providers[0]
                        self.logger.info(f"Using first available provider: {providers[0]}")
                
                # Set the selected provider
                if provider_to_select:
                    index = self.provider_combo.findText(provider_to_select)
                    if index >= 0:
                        # Block the signal so only the explicit call below loads
                        # the models once (the connected slot would double it).
                        self.provider_combo.blockSignals(True)
                        self.provider_combo.setCurrentIndex(index)
                        self.provider_combo.blockSignals(False)
                    self._on_provider_changed(provider_to_select)
                
        except Exception as e:
            self.logger.error(f"Could not populate providers: {e}")

            # Clear combo and add error indicator
            self.provider_combo.clear()
            self.provider_combo.addItem("❌ Error loading providers")
            self.provider_combo.setEnabled(False)

            # Clear models combo and disable it
            self.model_combo.clear()
            self.model_combo.addItem("❌ Providers not available")
            self.model_combo.setEnabled(False)

            # Show error in task type label as feedback to user
            if hasattr(self, 'task_type_label'):
                self.task_type_label.setText("❌ Provider detection failed - check settings")
                self.task_type_label.setStyleSheet("color: #d32f2f; font-weight: bold;")

            self.logger.error(f"Provider detection failed. User must check configuration before using this step.")
    
    def _refresh_providers(self):
        """Refresh provider and model lists with current status - Claude Generated"""
        current_provider = self.provider_combo.currentText()
        current_model = self.model_combo.currentText()
        
        # Re-populate providers
        self._populate_providers()
        
        # Try to restore previous selections
        if current_provider:
            index = self.provider_combo.findText(current_provider)
            if index >= 0:
                self.provider_combo.blockSignals(True)
                self.provider_combo.setCurrentIndex(index)
                self.provider_combo.blockSignals(False)
                # The model is restored asynchronously once the new list is
                # populated (see _on_provider_models_loaded).
                self._on_provider_changed(current_provider, restore_model=current_model or None)
        
        # Update status
        self._validate_configuration()
        self.logger.info("Provider list refreshed")

    def _populate_model_combo_with_styling(self, provider: str, models: List[str]) -> Optional[str]:
        """
        Populate model combo with visual baseline highlighting - Claude Generated

        Args:
            provider: Provider name
            models: List of available models

        Returns:
            The model that should be selected (preferred model)
        """
        self.model_combo.clear()

        if not models:
            return None

        # Sort models alphabetically (case-insensitive)
        models = sorted(models, key=lambda s: s.lower())

        # Get baseline information for this provider
        try:
            task_pref_provider, task_pref_model, task_reason = self._load_task_preferences_direct()
        except Exception as e:
            self.logger.debug(f"No task preferences: {e}")
            task_pref_provider, task_pref_model, task_reason = None, None, ""

        try:
            provider_pref_model = self._get_preferred_model_for_provider(provider)
        except Exception as e:
            self.logger.debug(f"No provider preference: {e}")
            provider_pref_model = None

        model_to_select = None

        # Add each model with individual styling
        for model in models:
            # Determine baseline source for this specific model
            baseline_source = self._get_baseline_source(provider, model)

            # Determine display text and styling
            if baseline_source == 'task_preference':
                # Task Preference: Green background + star icon
                display_text = f"⭐ {model}"
                background_color = QColor("#e8f5e9")  # Light green
                text_color = QColor("#2e7d32")  # Dark green
                tooltip = f"⭐ Task Preference for '{self.step_id}'\nSource: Settings → Task Preferences\nReason: {task_reason}"

                # This is the model to select (highest priority)
                if not model_to_select:
                    model_to_select = model

            elif baseline_source == 'provider_preference':
                # Provider Preference: Blue background + diamond icon
                display_text = f"💎 {model}"
                background_color = QColor("#e3f2fd")  # Light blue
                text_color = QColor("#1976d2")  # Dark blue
                tooltip = f"💎 Provider Preferred Model\nSource: Settings → Provider Settings → {provider}"

                # Select if no task preference exists
                if not model_to_select:
                    model_to_select = model

            else:
                # Normal model: no special styling
                display_text = model
                background_color = None
                text_color = None
                tooltip = None

            # Add item to combo
            self.model_combo.addItem(display_text)
            index = self.model_combo.count() - 1

            # Set item data roles for styling
            if background_color:
                self.model_combo.setItemData(index, QBrush(background_color), Qt.ItemDataRole.BackgroundRole)
            if text_color:
                self.model_combo.setItemData(index, QBrush(text_color), Qt.ItemDataRole.ForegroundRole)
            if tooltip:
                self.model_combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)

            # Store the clean model name (without icon) as user data
            self.model_combo.setItemData(index, model, Qt.ItemDataRole.UserRole)

        # Fallback: select first model if no preferred model found
        if not model_to_select and models:
            model_to_select = models[0]

        return model_to_select

    def _on_provider_changed(self, provider: str, restore_model: Optional[str] = None):
        """Handle provider change — detect models off the UI thread - Claude Generated

        Live model detection runs in the shared ModelLoadWorker so switching the
        provider no longer freezes the dialog; the result is handled in
        :meth:`_on_provider_models_loaded`. ``restore_model`` (refresh / config-load)
        is selected once the list is populated, mirroring the former synchronous
        post-call restore.
        """
        if not provider:
            return
        if not self.config_manager:
            self._on_manual_config_changed()
            return

        self._pending_provider = provider
        self._pending_restore_model = restore_model
        self.model_combo.clear()
        self.model_combo.addItem("⏳ Lade Modelle…", None)
        self.model_combo.setEnabled(False)

        from ..utils.config_manager import ProviderDetectionService
        detection_service = ProviderDetectionService(self.config_manager)
        worker = ModelLoadWorker(detection_service, provider, force=True)
        worker.fetched.connect(self._on_provider_models_loaded)
        worker.finished.connect(lambda w=worker: self._retire_model_worker(w))
        self._model_workers.add(worker)
        worker.start()

    def _retire_model_worker(self, worker) -> None:
        """Drop the strong ref once the worker thread has finished - Claude Generated"""
        self._model_workers.discard(worker)
        worker.deleteLater()

    def _on_provider_models_loaded(self, provider: str, models: list):
        """Populate the model combo when detection finishes (main thread) - Claude Generated"""
        # Ignore late results for a provider the user has since switched away from.
        if provider != self._pending_provider:
            return
        restore_model = self._pending_restore_model
        self._pending_restore_model = None
        self.model_combo.setEnabled(True)

        try:
            if models:
                self.logger.debug(f"🔍AVAILABLE_MODELS: provider='{provider}', models={models[:5]}{'...' if len(models) > 5 else ''} (total: {len(models)})")

                # Populate combo with visual styling for baseline models
                model_to_select = self._populate_model_combo_with_styling(provider, models)

                # Handle fuzzy matching if exact model not found
                if model_to_select and model_to_select not in models:
                    fuzzy_match = self._find_fuzzy_model_match(model_to_select, models)
                    if fuzzy_match:
                        model_to_select = fuzzy_match
                        self.logger.info(f"Using fuzzy match '{fuzzy_match}' for '{model_to_select}'")

                # Set the selected model in combo (search by UserRole data, not display text)
                if model_to_select:
                    for i in range(self.model_combo.count()):
                        stored_model = self.model_combo.itemData(i, Qt.ItemDataRole.UserRole)
                        if stored_model == model_to_select:
                            self.model_combo.setCurrentIndex(i)
                            self.logger.debug(f"🔍MODEL_SELECTED: '{model_to_select}' at index {i}")
                            break

                # Apply combo-box level styling based on selected model
                if model_to_select:
                    baseline_source = self._get_baseline_source(provider, model_to_select)

                    tooltip_parts = []
                    if baseline_source == 'task_preference':
                        tooltip_parts.append(f"🟢 Task Preference: {provider} / {model_to_select}")
                        tooltip_parts.append(f"Source: Settings → Task Preferences → {self.step_id}")
                    elif baseline_source == 'provider_preference':
                        tooltip_parts.append(f"🔵 Provider Preference: {provider} / {model_to_select}")
                        tooltip_parts.append("Source: Settings → Provider Settings")
                    else:
                        tooltip_parts.append(f"Model: {model_to_select}")
                    self.model_combo.setToolTip("\n".join(tooltip_parts))

                    if baseline_source == 'task_preference':
                        self.model_combo.setStyleSheet(STYLE_TASK_PREFERENCE)
                        self.provider_combo.setStyleSheet(STYLE_TASK_PREFERENCE)
                    elif baseline_source == 'provider_preference':
                        self.model_combo.setStyleSheet(STYLE_PROVIDER_PREFERENCE)
                        self.provider_combo.setStyleSheet(STYLE_PROVIDER_PREFERENCE)
                    else:
                        self.model_combo.setStyleSheet(STYLE_OVERRIDE)
                        self.provider_combo.setStyleSheet(STYLE_OVERRIDE)
            else:
                # Fallback models
                fallback_models = {
                    "ollama": ["cogito:32b", "cogito:14b", "llama3:8b"],
                    "gemini": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
                    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                    "anthropic": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"]
                }
                self.model_combo.clear()
                for model in fallback_models.get(provider, ["default-model"]):
                    self.model_combo.addItem(model)
                    self.model_combo.setItemData(self.model_combo.count() - 1, model, Qt.ItemDataRole.UserRole)

            # Restore a caller-requested model (refresh / config-load) — mirrors the
            # former synchronous post-call findText/setCurrentText restore.
            if restore_model:
                idx = self.model_combo.findText(restore_model)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
                else:
                    self.model_combo.setCurrentText(restore_model)
        except Exception as e:
            self.logger.warning(f"Could not load models for {provider}: {e}")

        self._on_manual_config_changed()
    
    def _get_preferred_model_for_provider(self, provider: str) -> Optional[str]:
        """Get preferred model for provider with Task Preference priority - Claude Generated"""
        try:
            if not self.config_manager:
                return None

            # 🔍 DEBUG: Log pipeline config dialog preference request - Claude Generated
            self.logger.debug(f"🔍PIPELINE_DIALOG_PREF_REQUEST: provider='{provider}'")

            # TIER 1: Check Task Preferences first (highest priority) - Claude Generated
            try:
                task_pref_provider, task_pref_model, reason = self._load_task_preferences_direct()
                if task_pref_provider == provider and task_pref_model:
                    self.logger.info(f"🎯 Using model from task preferences for {provider}: {task_pref_model} ({reason})")
                    return task_pref_model
            except Exception as e:
                self.logger.debug(f"No task preference model for {provider}: {e}")

            # TIER 2: the provider's own preferred model - Claude Generated
            # Read from the unified provider list. This block used to walk
            # ``unified_config.openai_compatible_providers`` / ``.ollama_providers``,
            # attributes the class has not had since the provider unification, so
            # it raised AttributeError on every call, was swallowed by the outer
            # handler ("Error getting preferred model for X") and no provider
            # ever contributed a preferred model.
            unified = self.config_manager.load_config(force_reload=True).unified_config

            entry = unified.get_provider_by_name(provider)
            if entry is None:
                # Fuzzy fallback for name variants (e.g. "LLMachine/Ollama").
                for candidate in unified.providers:
                    if self._provider_names_match(candidate.name, provider):
                        entry = candidate
                        break
            if entry is not None and entry.preferred_model:
                self.logger.debug(
                    f"🔍PIPELINE_DIALOG_FOUND: '{provider}' -> '{entry.preferred_model}'"
                )
                return entry.preferred_model

            # Legacy single-provider fields, still written for these two.
            if provider == "gemini":
                return unified.gemini_preferred_model or None
            if provider == "anthropic":
                return unified.anthropic_preferred_model or None

            self.logger.debug(f"🔍PIPELINE_DIALOG_FOUND: '{provider}' -> None (not found)")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting preferred model for {provider}: {e}")
            return None
    
    def _find_fuzzy_model_match(self, preferred_model: str, available_models: List[str]) -> Optional[str]:
        """Find fuzzy match for model name in available models - Claude Generated"""
        if not preferred_model or not available_models:
            return None
        
        preferred_lower = preferred_model.lower()
        
        # 1. Try partial name matching (e.g., "cogito:8b" -> "cogito:*")
        base_name = preferred_lower.split(':')[0]  # Extract base name before ':'
        for model in available_models:
            if model.lower().startswith(base_name):
                self.logger.debug(f"🔍FUZZY_MATCH: '{preferred_model}' -> '{model}' (base name match)")
                return model
        
        # 2. Try tag-flexible matching (e.g., "model:8b" -> "model:latest")  
        if ':' in preferred_lower:
            base_part = preferred_lower.split(':')[0]
            for model in available_models:
                if ':' in model.lower() and model.lower().split(':')[0] == base_part:
                    self.logger.debug(f"🔍FUZZY_MATCH: '{preferred_model}' -> '{model}' (tag flexible match)")
                    return model
        
        # 3. Try substring matching for complex model names
        for model in available_models:
            if base_name in model.lower() or model.lower() in preferred_lower:
                self.logger.debug(f"🔍FUZZY_MATCH: '{preferred_model}' -> '{model}' (substring match)")
                return model
        
        return None
    
    def _task_type_for_step(self) -> UnifiedTaskType:
        """The TaskType for this step, ``GENERAL`` when the id is not one.

        The step ids are the enum's values. Both call sites used to go through
        ``SmartTaskType.from_pipeline_step(...).to_unified_task_type()``, and
        neither method exists — ``SmartTaskType`` *is* ``UnifiedTaskType``,
        imported twice under two names. Every call raised, so the legacy task
        preference fallback and the smart preview were dead.
        - Claude Generated
        """
        try:
            return UnifiedTaskType(self.step_id)
        except ValueError:
            return UnifiedTaskType.GENERAL

    def _provider_names_match(self, config_name: str, requested_name: str) -> bool:
        """Check if provider names match with fuzzy logic for common variations - Claude Generated"""
        # Normalize names for comparison
        config_normalized = config_name.lower().replace(' ', '').replace('-', '').replace('/', '').replace('_', '')
        requested_normalized = requested_name.lower().replace(' ', '').replace('-', '').replace('/', '').replace('_', '')
        
        # Direct match after normalization
        if config_normalized == requested_normalized:
            return True
        
        # Check if one contains the other (e.g., "LLMachine/Ollama" contains "ollama")
        if 'ollama' in config_normalized and 'ollama' in requested_normalized:
            return True
            
        return False
    
    def _get_preferred_provider_from_settings(self) -> Optional[str]:
        """Get preferred provider with Task Preference priority - Claude Generated"""
        try:
            if not self.config_manager:
                return None

            # TIER 1: Check Task Preferences first (highest priority) - Claude Generated
            try:
                task_pref_provider, _, reason = self._load_task_preferences_direct()
                if task_pref_provider:
                    self.logger.info(f"🎯 Using provider from task preferences: {task_pref_provider} ({reason})")
                    return task_pref_provider
            except Exception as e:
                self.logger.debug(f"No task preference provider available: {e}")

            # TIER 2: Fallback to global preferred_provider
            unified_config = self.config_manager.get_unified_config()
            if unified_config.preferred_provider:
                self.logger.info(f"📋 Using global preferred provider: {unified_config.preferred_provider}")
                return unified_config.preferred_provider

            return None

        except Exception as e:
            self.logger.warning(f"Error getting preferred provider from settings: {e}")
            return None

    def _get_baseline_source(self, provider: str, model: str) -> str:
        """
        Determine the source of baseline configuration for provider/model combo - Claude Generated

        Args:
            provider: Provider name to check
            model: Model name to check

        Returns:
            'task_preference': From config.unified_config.task_preferences (highest priority)
            'provider_preference': From provider's preferred_model setting
            'none': Not a baseline setting (user override or default)
        """
        try:
            if not self.config_manager:
                return 'none'

            config = self.config_manager.load_config()

            # Map step_id to task_name for task_preferences lookup - Claude Generated
            task_name_mapping = {
                "initialisation": "initialisation",
                "keywords": "keywords",
                "classification": "classification",
                "dk_classification": "dk_classification",
                "image_text_extraction": "image_text_extraction"
            }
            task_name = task_name_mapping.get(self.step_id, "")

            # TIER 1: Check task preferences (highest priority)
            if task_name and task_name in config.unified_config.task_preferences:
                task_data = config.unified_config.task_preferences[task_name]
                model_priorities = task_data.model_priority if task_data else []

                for priority_entry in model_priorities:
                    if (priority_entry.get("provider_name") == provider and
                        priority_entry.get("model_name") == model):
                        return 'task_preference'

            # TIER 2: Check provider preferred model
            unified_config = config.unified_config
            provider_obj = unified_config.get_provider_by_name(provider)
            if provider_obj and provider_obj.preferred_model:
                if provider_obj.preferred_model == model:
                    return 'provider_preference'

            return 'none'

        except Exception as e:
            self.logger.warning(f"Error determining baseline source: {e}")
            return 'none'

    def _initialize_with_preferred_settings(self):
        """Initialize step config with Task Preference enhanced defaults - Claude Generated"""
        try:
            if not self.config_manager:
                # If no config manager, just populate providers with defaults
                self._populate_providers()
                return

            # ENHANCED INITIALIZATION WITH DIRECT TASK PREFERENCES
            selected_provider = None
            selected_model = None
            selection_reason = "unknown"

            # 1. HIGHEST PRIORITY: Direct Task Preference Loading (NEW) - Claude Generated
            try:
                selected_provider, selected_model, selection_reason = self._load_task_preferences_direct()

                if selected_provider and selected_model:
                    self.logger.info(f"✅ Initialized {self.step_id} via direct task preferences: {selected_provider}/{selected_model} - {selection_reason}")
                else:
                    self.logger.debug(f"⚠️ Direct task preference loading failed for {self.step_id}: {selection_reason}")

            except Exception as e:
                self.logger.warning(f"Error in direct task preference loading: {e}")

            # 2. FALLBACK: Legacy SmartProviderSelector task preferences (if direct loading failed)
            if not selected_provider:
                try:
                    smart_selector = SmartProviderSelector(self.config_manager)
                    if hasattr(smart_selector, 'unified_config') and smart_selector.unified_config:
                        # Map step to task type
                        task_pref = smart_selector.unified_config.get_task_preference(
                            self._task_type_for_step()
                        )

                        # Get first available provider/model from task preferences
                        for priority_entry in task_pref.model_priority:
                            candidate_provider = priority_entry.get("provider_name")
                            candidate_model = priority_entry.get("model_name")

                            if candidate_provider and candidate_model:
                                # Verify provider is available
                                if smart_selector._is_provider_available(candidate_provider):
                                    available_models = smart_selector.provider_detection_service.get_available_models(candidate_provider)
                                    if candidate_model in available_models:
                                        selected_provider = candidate_provider
                                        selected_model = candidate_model
                                        rank = task_pref.model_priority.index(priority_entry) + 1
                                        selection_reason = f"legacy task preference (rank {rank})"
                                        break
                                    else:
                                        # Try fuzzy matching
                                        fuzzy_match = smart_selector._find_fuzzy_model_match(candidate_model, available_models)
                                        if fuzzy_match:
                                            selected_provider = candidate_provider
                                            selected_model = fuzzy_match
                                            selection_reason = f"legacy task preference via fuzzy match ('{candidate_model}' -> '{fuzzy_match}')"
                                            break

                        if selected_provider and selected_model:
                            self.logger.info(f"🔄 Initialized {self.step_id} via {selection_reason}: {selected_provider}/{selected_model}")
                except Exception as e:
                    self.logger.warning(f"Failed to use legacy task preferences for initialization: {e}")

            # 3. FALLBACK: Legacy provider preferences
            if not selected_provider:
                preferred_provider = self._get_preferred_provider_from_settings()
                if preferred_provider:
                    preferred_model = self._get_preferred_model_for_provider(preferred_provider)
                    if preferred_model:
                        selected_provider = preferred_provider
                        selected_model = preferred_model
                        selection_reason = "provider preferences"
                        self.logger.info(f"📝 Initialized {self.step_id} via {selection_reason}: {preferred_provider}/{preferred_model}")

            # 4. Apply the selected configuration to step_config
            if selected_provider:
                self.step_config.provider = selected_provider
                if selected_model:
                    self.step_config.model = selected_model
                # Store selection reason for UI display
                self.step_config.selection_reason = selection_reason

            # Now populate providers with the preferred settings in place
            self._populate_providers()

        except Exception as e:
            self.logger.warning(f"Error initializing with enhanced preferred settings: {e}")
            # Fallback to basic provider population
            self._populate_providers()
    
    def _on_smart_config_changed(self):
        """Handle smart configuration changes - Claude Generated"""
        # Task type is auto-derived, no user input needed
        # Update preview with smart selection
        self._update_smart_preview()
        self.config_changed.emit()
    
    def _open_task_preferences(self):
        """Open the comprehensive settings dialog to edit task preferences - Claude Generated"""
        try:
            # Navigate up the parent hierarchy to find the main window
            main_window = self.parent()
            while main_window and not hasattr(main_window, 'show_settings'):
                main_window = main_window.parent()

            if main_window and hasattr(main_window, 'show_settings'):
                # Close this dialog first
                self.accept()
                # Open settings dialog with focus on provider tab
                main_window.show_settings()
                # TODO: Add way to focus on specific task in the UnifiedProviderTab
            else:
                QMessageBox.information(
                    self,
                    "Settings Access",
                    "Please open the Settings dialog from the main menu to edit task preferences.\n\n"
                    "Go to: Settings → Providers & Models → Task Preferences tab"
                )

        except Exception as e:
            self.logger.error(f"Error opening task preferences: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open task preferences:\n\n{str(e)}"
            )

    def _update_smart_preview(self):
        """Update smart mode preview with Task Preference integration - Claude Generated"""
        try:
            if self.config_manager:
                smart_selector = SmartProviderSelector(self.config_manager)
                prefer_fast = False  # Smart mode uses balanced approach
                
                # Map pipeline step_id to task_name for task_preferences lookup - Claude Generated
                task_name_mapping = {
                    "input": "",                     # No LLM required
                    "initialisation": "initialisation",
                    "search": "",                    # No LLM required
                    "keywords": "keywords",
                    "classification": "classification",
                    "dk_search": "",                 # No LLM required
                    "dk_classification": "dk_classification",  # FIXED: Consistent mapping
                    "image_text_extraction": "image_text_extraction"
                }
                task_name = task_name_mapping.get(self.step_id, "")
                smart_task_type = self._task_type_for_step()

                # Get smart selection with task preference integration - Claude Generated
                selection = smart_selector.select_provider(
                    task_type=smart_task_type, 
                    prefer_fast=prefer_fast,
                    step_id=self.step_id,
                    task_name=task_name  # This enables task preference hierarchy
                )
                
                # Enhanced preview with task-specific information
                preview_parts = []
                
                # Base selection info
                preview_parts.append(f"🎯 **{selection.provider}** / **{selection.model}**")
                
                # Task type indicator
                task_type_display = smart_task_type.value.replace('_', ' ').title()
                preview_parts.append(f"📋 Task: {task_type_display}")
                
                # Selection reason analysis
                selection_indicators = []
                
                # ENHANCED: Check if task preference was used (root-level config.task_preferences) - Claude Generated
                try:
                    if hasattr(smart_selector, 'config') and smart_selector.config and task_name:
                        # CRITICAL DEBUG: Log task preference availability - Claude Generated
                        self.logger.info(f"🔍 SMART_PREVIEW_TASK_CHECK: step_id='{self.step_id}' -> task_name='{task_name}' -> available_tasks={list(smart_selector.config.unified_config.task_preferences.keys())}")

                        # Check if task has specific preferences in config.unified_config.task_preferences
                        if task_name in smart_selector.config.unified_config.task_preferences:
                            task_data = smart_selector.config.unified_config.task_preferences[task_name]
                            model_priorities = task_data.model_priority if task_data else []

                            # CRITICAL DEBUG: Log found task preference data - Claude Generated
                            self.logger.info(f"🔍 SMART_PREVIEW_FOUND_PREFS: task='{task_name}' -> priorities={model_priorities}")

                            # Check if this provider/model combo is in task preferences
                            task_pref_found = False
                            for priority_entry in model_priorities:
                                if (priority_entry.get("provider_name") == selection.provider and
                                    priority_entry.get("model_name") == selection.model):
                                    rank = model_priorities.index(priority_entry) + 1
                                    total_prefs = len(model_priorities)
                                    confidence = "High" if rank == 1 else "Medium" if rank <= 2 else "Low"
                                    selection_indicators.append(f"⭐ Task preference #{rank}/{total_prefs}")
                                    selection_indicators.append(f"🏆 Confidence: {confidence}")
                                    task_pref_found = True
                                    break

                            # Check for chunked preferences if applicable
                            if not task_pref_found and 'chunked_model_priority' in task_data:
                                chunked_priorities = task_data.chunked_model_priority if task_data else None
                                if chunked_priorities:
                                    for priority_entry in chunked_priorities:
                                        if (priority_entry.get("provider_name") == selection.provider and
                                            priority_entry.get("model_name") == selection.model):
                                            rank = chunked_priorities.index(priority_entry) + 1
                                            total_prefs = len(chunked_priorities)
                                            selection_indicators.append(f"⭐ Chunked preference #{rank}/{total_prefs}")
                                            selection_indicators.append(f"🧩 Chunked mode")
                                            task_pref_found = True
                                            break

                            # Add task preference summary info - Claude Generated
                            if task_pref_found:
                                # Show how many total preferences are configured for this task
                                chunked_count = len(task_data.chunked_model_priority) if task_data and task_data.chunked_model_priority else 0
                                if chunked_count > 0:
                                    selection_indicators.append(f"📊 Total preferences: {len(model_priorities)} standard, {chunked_count} chunked")
                                else:
                                    selection_indicators.append(f"📊 Total preferences: {len(model_priorities)}")
                            else:
                                # Show that task has preferences but this isn't one of them
                                if model_priorities:
                                    selection_indicators.append(f"⚠️ Not in {len(model_priorities)} task preferences")
                        else:
                            # No task preferences configured for this task
                            selection_indicators.append("📝 No task preferences configured")
                    
                    # Fallback analysis if no task preference matched
                    if not selection_indicators:
                        # Check if provider config was used. Own lookup, not a
                        # private method of SmartProviderSelector — that one
                        # (``_get_preferred_model_from_config``) does not exist,
                        # so this branch always ended in the handler below.
                        # - Claude Generated
                        preferred_model = self._get_preferred_model_for_provider(selection.provider)
                        if preferred_model and selection.model == preferred_model:
                            selection_indicators.append("🔧 Provider config")
                        elif prefer_fast and any(indicator in selection.model.lower() for indicator in ['flash', 'mini', 'haiku', 'turbo']):
                            selection_indicators.append("⚡ Speed optimized")
                        elif selection.fallback_used:
                            selection_indicators.append("🔄 Fallback used")
                        else:
                            selection_indicators.append("✅ Auto-selected")
                            
                except Exception as e:
                    self.logger.warning(f"Failed to analyze selection reason: {e}")
                    selection_indicators.append("✅ Smart selection")
                
                # Performance info
                if hasattr(selection, 'selection_time'):
                    selection_indicators.append(f"⏱️ {selection.selection_time*1000:.1f}ms")
                
                preview_parts.extend(selection_indicators)
                
                # Combine preview text
                preview_text = " • ".join(preview_parts)
                self.smart_preview_label.setText(preview_text)
                self.smart_preview_label.setStyleSheet("color: green; font-style: italic; font-weight: bold;")

                # Show/hide Edit Preferences button based on task preference usage - Claude Generated
                has_task_preferences = any("Task preference" in indicator for indicator in selection_indicators)
                self.edit_preferences_button.setVisible(has_task_preferences)
            
        except Exception as e:
            error_msg = f"⚠️ Preview unavailable: {str(e)}"
            self.smart_preview_label.setText(error_msg)
            self.smart_preview_label.setStyleSheet("color: orange; font-style: italic;")
            self.edit_preferences_button.setVisible(False)  # Hide button on error - Claude Generated
            self.logger.warning(f"Smart preview update failed: {e}")
    
    def _on_manual_config_changed(self):
        """Handle manual configuration changes with icon prefix handling - Claude Generated"""
        self.step_config.provider = self.provider_combo.currentText()

        # The clean model name lives in UserRole (set for every item by
        # _populate_model_combo_with_styling); the display text may carry a ⭐/💎
        # decoration that must never leak into the value. A free-typed entry
        # (editable combo, no matching item) carries no decoration, so
        # currentText() is already clean. This replaces a brittle hardcoded
        # .replace() strip that broke whenever a new decoration icon appeared.
        # - Claude Generated
        current_index = self.model_combo.currentIndex()
        if current_index >= 0:
            clean_model = self.model_combo.itemData(current_index, Qt.ItemDataRole.UserRole)
            self.step_config.model = clean_model or self.model_combo.currentText()
        else:
            self.step_config.model = self.model_combo.currentText()

        self.step_config.task = self.task_combo.currentText()
        self.config_changed.emit()
    
    def _on_expert_config_changed(self):
        """Handle expert parameter changes - Claude Generated"""
        self.step_config.temperature = self.temperature_spinbox.value()
        self.step_config.top_p = self.top_p_spinbox.value()
        self.step_config.max_tokens = self.max_tokens_spinbox.value()
        # Think checkbox: PartiallyChecked = None (use default), Checked = True, Unchecked = False
        state = self.think_checkbox.checkState()
        if state == Qt.CheckState.PartiallyChecked:
            self.step_config.think = None
        elif state == Qt.CheckState.Checked:
            self.step_config.think = True
        else:
            self.step_config.think = False
        self.config_changed.emit()
    
    def _validate_configuration(self):
        """Validate current configuration - Claude Generated"""
        # In baseline + override architecture, validate based on whether configuration is complete
        if not self.step_config.provider or not self.step_config.model:
            self.status_label.setText("✅ Using smart defaults (will auto-select optimal provider/model)")
            self.status_label.setStyleSheet("color: green;")
            return

        # Validate manual overrides using SmartProviderSelector
        if self.config_manager:
            try:
                smart_selector = SmartProviderSelector(self.config_manager)
                validation = smart_selector.validate_manual_choice(
                    self.step_config.provider,
                    self.step_config.model
                )

                if validation["valid"]:
                    self.status_label.setText("✅ Manual override configuration is valid")
                    self.status_label.setStyleSheet("color: green;")
                else:
                    issues = "; ".join(validation["issues"])
                    self.status_label.setText(f"❌ Issues: {issues}")
                    self.status_label.setStyleSheet("color: red;")

            except Exception as e:
                self.status_label.setText(f"⚠️ Validation error: {str(e)}")
                self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText("✅ Manual override configuration set")
            self.status_label.setStyleSheet("color: green;")
    
    def _test_configuration(self):
        """Test current configuration - Claude Generated"""
        # This would perform an actual test call to the provider
        self.status_label.setText("🧪 Testing functionality to be implemented")
        self.status_label.setStyleSheet("color: blue;")
    
    def get_step_config(self) -> PipelineStepConfig:
        """Get current step configuration - Claude Generated"""
        return self.step_config
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration in legacy format for compatibility - Claude Generated

        Now uses unified PipelineConfigParser for validation across CLI and GUI
        """
        parser = PipelineConfigParser()

        # Build configuration from widget values
        config = {
            "step_id": self.step_config.step_id,
            "enabled": True,  # HybridStepConfig is always enabled
            "provider": self.step_config.provider or "",
            "model": self.step_config.model or "",
            "task": self.step_config.task or "",
            "temperature": getattr(self.step_config, 'temperature', 0.7),
            "top_p": getattr(self.step_config, 'top_p', 0.1),
        }

        # Validate critical parameters using unified parser
        if config.get("task"):
            is_valid, error_msg = parser.validate_parameter(
                self.step_id, "task", config["task"]
            )
            if not is_valid:
                self.logger.warning(f"Task validation failed: {error_msg}")
                # Still return config but log the issue
                config["validation_warning"] = error_msg

        if config.get("temperature") is not None:
            if not parser.validate_temperature(config["temperature"]):
                self.logger.warning(f"Temperature validation failed: {config['temperature']}")
                config["validation_warning"] = f"Temperature out of range: {config['temperature']}"

        if config.get("top_p") is not None:
            if not parser.validate_top_p(config["top_p"]):
                self.logger.warning(f"Top_p validation failed: {config['top_p']}")
                config["validation_warning"] = f"Top_p out of range: {config['top_p']}"

        # Add task type information as metadata
        config["task_type"] = self.step_config.task_type.value if self.step_config.task_type else None

        return config
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration from legacy dict format - Claude Generated"""
        # Convert legacy dict to PipelineStepConfig
        from ..utils.config_models import TaskType as UnifiedTaskType

        # Map task_type if available
        task_type = None
        if "task_type" in config and config["task_type"]:
            try:
                task_type = UnifiedTaskType(config["task_type"])
            except ValueError:
                self.logger.warning(f"Unknown task type: {config.get('task_type')}")

        # Create PipelineStepConfig from legacy dict
        step_config = PipelineStepConfig(
            step_id=config.get("step_id", self.step_id),
            task_type=task_type,
            provider=config.get("provider"),
            model=config.get("model"),
            task=config.get("task")
        )
        
        # Set additional parameters if available
        if hasattr(step_config, 'temperature'):
            step_config.temperature = config.get("temperature", 0.7)
        if hasattr(step_config, 'top_p'):
            step_config.top_p = config.get("top_p", 0.1)
        
        # Apply the configuration
        self.set_step_config(step_config)
    
    def set_step_config(self, config: PipelineStepConfig):
        """Set step configuration and update UI - Claude Generated"""
        self.step_config = config
        
        # Note: Mode radios removed in baseline + override architecture
        
        # Update task type display (readonly)
        if config.task_type:
            self._update_task_type_display()
        
        # Update manual mode controls
        if config.provider:
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentText(config.provider)
            self.provider_combo.blockSignals(False)
            # Trigger provider change; the saved model is restored asynchronously
            # once the model list is populated.
            self._on_provider_changed(config.provider, restore_model=config.model or None)
        elif config.model:
            self.model_combo.setCurrentText(config.model)
        if config.task:
            self.task_combo.setCurrentText(config.task)
        
        # Update expert mode controls
        if config.temperature is not None:
            self.temperature_spinbox.setValue(config.temperature)
        if config.top_p is not None:
            self.top_p_spinbox.setValue(config.top_p)
        if config.max_tokens is not None:
            self.max_tokens_spinbox.setValue(config.max_tokens)

        self._update_ui_for_mode()

        # Update smart mode preview after configuration change - Claude Generated
        self._update_smart_preview()
