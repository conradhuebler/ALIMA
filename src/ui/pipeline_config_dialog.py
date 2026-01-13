"""
Pipeline Configuration Dialog - Konfiguration für Pipeline-Schritte
Claude Generated - Ermöglicht die Konfiguration von Provider und Modellen für jeden Pipeline-Schritt
"""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QTextEdit,
    QTabWidget,
    QWidget,
    QMessageBox,
    QSplitter,
    QRadioButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QMetaObject, Q_ARG
from PyQt6.QtGui import QFont, QBrush, QColor
from typing import Dict, List, Any, Optional
import json
import logging

from ..core.pipeline_manager import PipelineConfig
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from ..utils.config_models import (
    PipelineStepConfig,
    TaskType as UnifiedTaskType
)
from ..utils.smart_provider_selector import SmartProviderSelector, TaskType as SmartTaskType
from ..utils.pipeline_config_parser import PipelineConfigParser
from ..utils.pipeline_config_builder import PipelineConfigBuilder

# Styling constants for baseline highlighting - Claude Generated
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
                font-size: 12px;
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
        legend_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 8px; padding: 8px; background-color: #f5f5f5; border-radius: 4px;")
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
                        self.provider_combo.setCurrentIndex(index)
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
                self.provider_combo.setCurrentIndex(index)
                self._on_provider_changed(current_provider)
                
                # Try to restore model selection
                if current_model:
                    model_index = self.model_combo.findText(current_model)
                    if model_index >= 0:
                        self.model_combo.setCurrentIndex(model_index)
        
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

    def _on_provider_changed(self, provider: str):
        """Handle provider change with visual baseline highlighting - Claude Generated"""
        if not provider:
            return

        try:
            if self.config_manager:
                from ..utils.config_manager import ProviderDetectionService
                detection_service = ProviderDetectionService(self.config_manager)
                models = detection_service.get_available_models(provider)

                if models:
                    # 🔍 DEBUG: Log available models - Claude Generated
                    self.logger.critical(f"🔍 AVAILABLE_MODELS: provider='{provider}', models={models[:5]}{'...' if len(models) > 5 else ''} (total: {len(models)})")

                    # Populate combo with visual styling for baseline models - Claude Generated
                    model_to_select = self._populate_model_combo_with_styling(provider, models)

                    # Handle fuzzy matching if exact model not found
                    if model_to_select and model_to_select not in models:
                        fuzzy_match = self._find_fuzzy_model_match(model_to_select, models)
                        if fuzzy_match:
                            model_to_select = fuzzy_match
                            self.logger.info(f"Using fuzzy match '{fuzzy_match}' for '{model_to_select}'")

                    # Set the selected model in combo (search by UserRole data, not display text)
                    if model_to_select:
                        # Find index by matching UserRole data (clean model name)
                        for i in range(self.model_combo.count()):
                            stored_model = self.model_combo.itemData(i, Qt.ItemDataRole.UserRole)
                            if stored_model == model_to_select:
                                self.model_combo.setCurrentIndex(i)
                                self.logger.critical(f"🔍 MODEL_SELECTED: '{model_to_select}' at index {i}")
                                break

                    # Apply combo-box level styling based on selected model
                    if model_to_select:
                        baseline_source = self._get_baseline_source(provider, model_to_select)

                        # Build combo tooltip
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

                        # Apply combobox-level styling - Claude Generated
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
        except Exception as e:
            self.logger.warning(f"Could not load models for {provider}: {e}")
        
        self._on_manual_config_changed()
    
    def _get_preferred_model_for_provider(self, provider: str) -> Optional[str]:
        """Get preferred model for provider with Task Preference priority - Claude Generated"""
        try:
            if not self.config_manager:
                return None

            # 🔍 DEBUG: Log pipeline config dialog preference request - Claude Generated
            self.logger.critical(f"🔍 PIPELINE_DIALOG_PREF_REQUEST: provider='{provider}'")

            # TIER 1: Check Task Preferences first (highest priority) - Claude Generated
            try:
                task_pref_provider, task_pref_model, reason = self._load_task_preferences_direct()
                if task_pref_provider == provider and task_pref_model:
                    self.logger.info(f"🎯 Using model from task preferences for {provider}: {task_pref_model} ({reason})")
                    return task_pref_model
            except Exception as e:
                self.logger.debug(f"No task preference model for {provider}: {e}")

            # TIER 2: Check Provider Settings - Claude Generated
            # Force reload to ensure we get latest saved config - Claude Generated
            config = self.config_manager.load_config(force_reload=True)
            
            # 🔍 DEBUG: Log what pipeline dialog sees in loaded config - Claude Generated
            self.logger.critical(f"🔍 PIPELINE_CONFIG_LOAD: gemini_preferred='{config.unified_config.gemini_preferred_model}', anthropic_preferred='{config.unified_config.anthropic_preferred_model}'")
            self.logger.critical(f"🔍 PIPELINE_CONFIG_LOAD: openai_providers_count={len(config.unified_config.openai_compatible_providers)}, ollama_providers_count={len(config.unified_config.ollama_providers)}")
            
            # Check static providers
            if provider == "gemini":
                preferred = config.unified_config.gemini_preferred_model or None
                self.logger.critical(f"🔍 PIPELINE_DIALOG_FOUND: gemini -> '{preferred}'")
                return preferred
            elif provider == "anthropic":
                preferred = config.unified_config.anthropic_preferred_model or None
                self.logger.critical(f"🔍 PIPELINE_DIALOG_FOUND: anthropic -> '{preferred}'")
                return preferred
            
            # Check OpenAI-compatible providers
            for openai_provider in config.unified_config.openai_compatible_providers:
                self.logger.critical(f"🔍 PIPELINE_CHECKING_OPENAI: '{openai_provider.name}'.preferred_model='{openai_provider.preferred_model}' vs requested '{provider}'")
                if openai_provider.name == provider:
                    preferred = openai_provider.preferred_model or None
                    self.logger.critical(f"🔍 PIPELINE_DIALOG_FOUND: openai_compatible '{provider}' -> '{preferred}'")
                    return preferred
            
            # Check Ollama providers - with fuzzy matching - Claude Generated
            for ollama_provider in config.unified_config.ollama_providers:
                self.logger.critical(f"🔍 PIPELINE_CHECKING_OLLAMA: '{ollama_provider.name}' vs requested '{provider}'")
                
                # Direct name match
                if ollama_provider.name == provider:
                    preferred = ollama_provider.preferred_model or None
                    self.logger.critical(f"🔍 PIPELINE_DIALOG_FOUND: ollama '{provider}' -> '{preferred}' (exact)")
                    return preferred
                
                # Fuzzy matching for provider name variations
                if self._provider_names_match(ollama_provider.name, provider):
                    preferred = ollama_provider.preferred_model or None
                    self.logger.critical(f"🔍 PIPELINE_DIALOG_FOUND: ollama '{provider}' -> '{preferred}' (fuzzy: '{ollama_provider.name}')")
                    return preferred
            
            self.logger.critical(f"🔍 PIPELINE_DIALOG_FOUND: '{provider}' -> None (not found)")
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
                self.logger.critical(f"🔍 FUZZY_MATCH: '{preferred_model}' -> '{model}' (base name match)")
                return model
        
        # 2. Try tag-flexible matching (e.g., "model:8b" -> "model:latest")  
        if ':' in preferred_lower:
            base_part = preferred_lower.split(':')[0]
            for model in available_models:
                if ':' in model.lower() and model.lower().split(':')[0] == base_part:
                    self.logger.critical(f"🔍 FUZZY_MATCH: '{preferred_model}' -> '{model}' (tag flexible match)")
                    return model
        
        # 3. Try substring matching for complex model names
        for model in available_models:
            if base_name in model.lower() or model.lower() in preferred_lower:
                self.logger.critical(f"🔍 FUZZY_MATCH: '{preferred_model}' -> '{model}' (substring match)")
                return model
        
        return None
    
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
                        smart_task_type = SmartTaskType.from_pipeline_step(self.step_id, "")
                        unified_task_type = smart_task_type.to_unified_task_type()

                        # Get task preference
                        task_pref = smart_selector.unified_config.get_task_preference(unified_task_type)

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
                
                # Map step_id to SmartTaskType for enhanced task detection
                smart_task_type = SmartTaskType.from_pipeline_step(self.step_id, "")
                
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
                        # Check if provider config was used
                        preferred_model = smart_selector._get_preferred_model_from_config(selection.provider)
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

        # Get clean model name from UserRole (without icon prefix) - Claude Generated
        current_index = self.model_combo.currentIndex()
        if current_index >= 0:
            clean_model = self.model_combo.itemData(current_index, Qt.ItemDataRole.UserRole)
            if clean_model:
                self.step_config.model = clean_model
            else:
                # Fallback: remove icon prefix manually
                display_text = self.model_combo.currentText()
                self.step_config.model = display_text.replace("⭐ ", "").replace("💎 ", "")
        else:
            self.step_config.model = self.model_combo.currentText()

        self.step_config.task = self.task_combo.currentText()
        self.config_changed.emit()
    
    def _on_expert_config_changed(self):
        """Handle expert parameter changes - Claude Generated"""
        self.step_config.temperature = self.temperature_spinbox.value()
        self.step_config.top_p = self.top_p_spinbox.value()
        self.step_config.max_tokens = self.max_tokens_spinbox.value()
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
            self.provider_combo.setCurrentText(config.provider)
            # Trigger provider change to populate models including preferred model
            self._on_provider_changed(config.provider)
        if config.model:
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


class PipelineStepConfigWidget(QWidget):
    """Widget für die Konfiguration eines Pipeline-Schritts - Claude Generated"""

    def __init__(
        self,
        step_name: str,
        step_id: str,
        llm_service: LlmService,
        prompt_service: PromptService = None,
        parent=None,
    ):
        super().__init__(parent)
        self.step_name = step_name
        self.step_id = step_id
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.logger = logging.getLogger(__name__)
        self.config = {}  # Store the config for this step
        self.setup_ui()

    def setup_ui(self):
        """Setup der UI für Step-Konfiguration - Claude Generated"""
        layout = QVBoxLayout(self)

        # Step Name Header
        header_label = QLabel(self.step_name)
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Provider & Model Selection
        provider_group = QGroupBox("LLM-Einstellungen")
        provider_layout = QGridLayout(provider_group)

        # Provider Dropdown
        provider_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.provider_combo = QComboBox()
        providers = self.llm_service.get_available_providers()
        self.provider_combo.addItems(providers)
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(self.provider_combo, 0, 1)

        # Model Dropdown
        provider_layout.addWidget(QLabel("Modell:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.load_prompt_settings)
        provider_layout.addWidget(self.model_combo, 1, 1)

        # Task Selection (for LLM steps)
        if self.step_id in ["initialisation", "keywords"]:
            provider_layout.addWidget(QLabel("Task:"), 2, 0)
            self.task_combo = QComboBox()

            # Define available tasks based on step
            if self.step_id == "initialisation":
                # Initial keyword extraction tasks
                available_tasks = ["initialisation", "keywords", "rephrase"]
            else:  # keywords step (final analysis)
                # Final keyword analysis tasks - Claude Generated
                # NOTE: "keywords_chunked" is only for chunking_task, NOT main task!
                available_tasks = ["keywords", "rephrase"]

            self.task_combo.addItems(available_tasks)
            self.task_combo.currentTextChanged.connect(self.on_task_changed)
            provider_layout.addWidget(self.task_combo, 2, 1)

        # Enable/Disable for this step
        self.enabled_checkbox = QCheckBox("Schritt aktivieren")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        provider_layout.addWidget(self.enabled_checkbox, 3, 0, 1, 2)

        layout.addWidget(provider_group)

        # Parameter Settings
        params_group = QGroupBox("Parameter")
        params_layout = QGridLayout(params_group)

        # Temperature
        params_layout.addWidget(QLabel("Temperatur:"), 0, 0)
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_spinbox.setValue(v / 100.0)
        )
        params_layout.addWidget(self.temp_slider, 0, 1)

        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.0, 1.0)
        self.temp_spinbox.setValue(0.7)
        self.temp_spinbox.setDecimals(2)
        self.temp_spinbox.setSingleStep(0.01)
        self.temp_spinbox.valueChanged.connect(
            lambda v: self.temp_slider.setValue(int(v * 100))
        )
        params_layout.addWidget(self.temp_spinbox, 0, 2)

        # Top-P
        params_layout.addWidget(QLabel("Top-P:"), 1, 0)
        self.p_slider = QSlider(Qt.Orientation.Horizontal)
        self.p_slider.setRange(0, 100)
        self.p_slider.setValue(10)
        self.p_slider.valueChanged.connect(lambda v: self.p_spinbox.setValue(v / 100.0))
        params_layout.addWidget(self.p_slider, 1, 1)

        self.p_spinbox = QDoubleSpinBox()
        self.p_spinbox.setRange(0.0, 1.0)
        self.p_spinbox.setValue(0.1)
        self.p_spinbox.setDecimals(2)
        self.p_spinbox.setSingleStep(0.01)
        self.p_spinbox.valueChanged.connect(
            lambda v: self.p_slider.setValue(int(v * 100))
        )
        params_layout.addWidget(self.p_spinbox, 1, 2)

        layout.addWidget(params_group)

        # DK Classification Parameters (only for dk_classification step) - Claude Generated
        if self.step_id == "dk_classification":
            dk_group = QGroupBox("DK Klassifikation")
            dk_layout = QGridLayout(dk_group)
            
            # DK Frequency Threshold
            dk_layout.addWidget(QLabel("Häufigkeits-Schwellenwert:"), 0, 0)
            self.dk_frequency_spinbox = QSpinBox()
            self.dk_frequency_spinbox.setMinimum(1)
            self.dk_frequency_spinbox.setMaximum(100)
            self.dk_frequency_spinbox.setValue(10)  # Default value
            self.dk_frequency_spinbox.setSuffix(" Vorkommen")
            self.dk_frequency_spinbox.setToolTip(
                "Mindest-Häufigkeit für DK-Klassifikationen.\n"
                "Nur Klassifikationen mit ≥ N Vorkommen im Katalog\n"
                "werden an das LLM weitergegeben.\n\n"
                "Niedrigere Werte = mehr Ergebnisse\n"
                "Höhere Werte = weniger, aber relevantere Ergebnisse"
            )
            dk_layout.addWidget(self.dk_frequency_spinbox, 0, 1)
            
            layout.addWidget(dk_group)

        # Keyword Chunking Parameters (only for keywords step)
        if self.step_id == "keywords":
            chunking_group = QGroupBox("Keyword Chunking")
            chunking_layout = QGridLayout(chunking_group)

            # Chunking Threshold
            chunking_layout.addWidget(QLabel("Chunking-Schwellwert:"), 0, 0)
            self.chunking_threshold_spinbox = QSpinBox()
            self.chunking_threshold_spinbox.setRange(100, 2000)
            self.chunking_threshold_spinbox.setValue(500)
            self.chunking_threshold_spinbox.setSuffix(" Keywords")
            self.chunking_threshold_spinbox.setToolTip(
                "Anzahl Keywords ab der Chunking aktiviert wird"
            )
            chunking_layout.addWidget(self.chunking_threshold_spinbox, 0, 1)

            # Chunking Task
            chunking_layout.addWidget(QLabel("Chunking-Task:"), 1, 0)
            self.chunking_task_combo = QComboBox()
            self.chunking_task_combo.addItems(["keywords_chunked", "rephrase"])
            self.chunking_task_combo.setCurrentText("keywords_chunked")
            self.chunking_task_combo.setToolTip("Task für Chunk-Verarbeitung")
            chunking_layout.addWidget(self.chunking_task_combo, 1, 1)

            layout.addWidget(chunking_group)

        # Debug: Add test button to load prompt settings
        if self.step_id in ["initialisation", "keywords"]:
            test_button = QPushButton("Test: Load Prompt Settings")
            test_button.clicked.connect(self.test_load_prompt_settings)
            layout.addWidget(test_button)

        # Custom Prompt (if applicable)
        if self.step_id in ["initialisation", "keywords", "dk_classification"]:
            prompt_group = QGroupBox("Custom Prompts (optional)")
            prompt_layout = QVBoxLayout(prompt_group)

            # Main prompt
            prompt_layout.addWidget(QLabel("Haupt-Prompt:"))
            self.custom_prompt = QTextEdit()
            self.custom_prompt.setMaximumHeight(80)
            self.custom_prompt.setPlaceholderText("Leer lassen für Standard-Prompt...")
            prompt_layout.addWidget(self.custom_prompt)

            # System prompt
            prompt_layout.addWidget(QLabel("System-Prompt:"))
            self.system_prompt = QTextEdit()
            self.system_prompt.setMaximumHeight(60)
            self.system_prompt.setPlaceholderText(
                "Leer lassen für Standard-System-Prompt..."
            )
            prompt_layout.addWidget(self.system_prompt)

            layout.addWidget(prompt_group)

        # Initialize models for default provider and load initial prompt settings
        self.on_provider_changed(self.provider_combo.currentText())

        # Load initial prompt settings after UI is fully set up
        if hasattr(self, "task_combo"):
            # Try immediate loading first
            self.load_prompt_settings()
            # Also set a timer as backup
            QTimer.singleShot(100, self.load_prompt_settings)

    def on_provider_changed(self, provider: str):
        """Handle provider change - Claude Generated"""
        self.model_combo.clear()
        models = self.llm_service.get_available_models(provider)
        if models:
            self.model_combo.addItems(models)
            
            # --- START QUICK FIX ---
            # Try to re-select the model that was saved in the config
            saved_model = self.config.get("model")
            if saved_model:
                index = self.model_combo.findText(saved_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
            # --- END QUICK FIX ---

        # Load prompt settings when provider changes
        self.load_prompt_settings()

    def on_task_changed(self, task: str):
        """Handle task change - load settings from prompts.json - Claude Generated"""
        self.load_prompt_settings()

    def load_prompt_settings(self):
        """Load prompt settings from prompts.json for current model/task - Claude Generated"""
        if not self.prompt_service:
            self.logger.warning("No prompt_service available for loading settings")
            return
        if not hasattr(self, "task_combo"):
            self.logger.warning("No task_combo available for loading settings")
            return

        try:
            current_task = self.task_combo.currentText()
            current_model = self.model_combo.currentText()

            self.logger.info(
                f"Loading prompt settings for task='{current_task}', model='{current_model}'"
            )
            self.logger.info(
                f"PromptService available: {self.prompt_service is not None}"
            )

            if current_task and current_model:
                # Get prompt config for this task and model
                prompt_config = self.prompt_service.get_prompt_config(
                    current_task, current_model
                )
                self.logger.info(f"Found prompt config: {prompt_config is not None}")

                if prompt_config:
                    self.logger.info(
                        f"Config details: temp={getattr(prompt_config, 'temp', 'N/A')}, p_value={getattr(prompt_config, 'p_value', 'N/A')}"
                    )

                if prompt_config:
                    # Update temperature and top_p from prompt config
                    if (
                        hasattr(prompt_config, "temp")
                        and prompt_config.temp is not None
                    ):
                        temp_value = float(prompt_config.temp)
                        self.logger.info(f"Setting temperature to {temp_value}")

                        # Use blockSignals to prevent recursion, more reliable than disconnect/reconnect
                        self.temp_spinbox.blockSignals(True)
                        self.temp_slider.blockSignals(True)

                        self.temp_spinbox.setValue(temp_value)
                        self.temp_slider.setValue(int(temp_value * 100))

                        self.temp_spinbox.blockSignals(False)
                        self.temp_slider.blockSignals(False)

                        # Force repaint to ensure UI is updated
                        self.temp_spinbox.repaint()
                        self.temp_slider.repaint()

                        # Also try processEvents to ensure immediate update
                        from PyQt6.QtWidgets import QApplication

                        QApplication.processEvents()

                        self.logger.info(
                            f"UI updated: spinbox={self.temp_spinbox.value()}, slider={self.temp_slider.value()}"
                        )

                    if (
                        hasattr(prompt_config, "p_value")
                        and prompt_config.p_value is not None
                    ):
                        p_value = float(prompt_config.p_value)
                        self.logger.info(f"Setting p_value to {p_value}")

                        # Use blockSignals to prevent recursion, more reliable than disconnect/reconnect
                        self.p_spinbox.blockSignals(True)
                        self.p_slider.blockSignals(True)

                        self.p_spinbox.setValue(p_value)
                        self.p_slider.setValue(int(p_value * 100))

                        self.p_spinbox.blockSignals(False)
                        self.p_slider.blockSignals(False)

                        # Force repaint to ensure UI is updated
                        self.p_spinbox.repaint()
                        self.p_slider.repaint()

                        # Also try processEvents to ensure immediate update
                        from PyQt6.QtWidgets import QApplication

                        QApplication.processEvents()

                        self.logger.info(
                            f"UI updated: p_spinbox={self.p_spinbox.value()}, p_slider={self.p_slider.value()}"
                        )

                    # Update custom prompt if available
                    if hasattr(self, "custom_prompt") and hasattr(
                        prompt_config, "prompt"
                    ):
                        # Don't overwrite if user has custom text, just show in placeholder
                        if not self.custom_prompt.toPlainText().strip():
                            self.custom_prompt.setPlaceholderText(
                                f"Standard-Prompt für {current_task}"
                            )

                    # Update system prompt if available
                    if hasattr(self, "system_prompt") and hasattr(
                        prompt_config, "system"
                    ):
                        # Don't overwrite if user has custom text, just show in placeholder
                        if not self.system_prompt.toPlainText().strip():
                            self.system_prompt.setPlaceholderText(
                                f"Standard-System-Prompt für {current_task}"
                            )

                    # Log the loaded settings for debugging
                    if hasattr(self, "logger"):
                        self.logger.info(
                            f"Loaded prompt settings for {current_task}/{current_model}: temp={getattr(prompt_config, 'temp', 'N/A')}, p={getattr(prompt_config, 'p_value', 'N/A')}"
                        )

        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Could not load prompt settings: {e}")

    def test_load_prompt_settings(self):
        """Test method to manually trigger prompt loading - Claude Generated"""
        self.logger.info("=== MANUAL TEST: Loading prompt settings ===")
        self.logger.info(
            f"Current values: temp={self.temp_spinbox.value()}, p={self.p_spinbox.value()}"
        )

        # Test 1: Force set test values to verify UI works
        self.logger.info("Test 1: Setting test values manually...")

        self.temp_spinbox.blockSignals(True)
        self.temp_slider.blockSignals(True)
        self.p_spinbox.blockSignals(True)
        self.p_slider.blockSignals(True)

        self.temp_spinbox.setValue(0.25)
        self.temp_slider.setValue(25)
        self.p_spinbox.setValue(0.1)
        self.p_slider.setValue(10)

        self.temp_spinbox.blockSignals(False)
        self.temp_slider.blockSignals(False)
        self.p_spinbox.blockSignals(False)
        self.p_slider.blockSignals(False)

        # Force UI update
        self.temp_spinbox.repaint()
        self.temp_slider.repaint()
        self.p_spinbox.repaint()
        self.p_slider.repaint()

        from PyQt6.QtWidgets import QApplication

        QApplication.processEvents()

        self.logger.info(
            f"After manual set: temp={self.temp_spinbox.value()}, p={self.p_spinbox.value()}"
        )

        # Test 2: Try loading from prompt service
        self.logger.info("Test 2: Loading from prompt service...")
        self.load_prompt_settings()

    def on_enabled_changed(self, enabled: bool):
        """Enable/disable step configuration - Claude Generated"""
        # Enable/disable all child widgets
        for widget in self.findChildren(QWidget):
            if widget != self.enabled_checkbox:
                widget.setEnabled(enabled)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration - Claude Generated"""
        config = {
            "step_id": self.step_id,
            "enabled": self.enabled_checkbox.isChecked(),
            "provider": self.provider_combo.currentText(),
            "model": self.model_combo.currentText(),
            "temperature": self.temp_spinbox.value(),
            "top_p": self.p_spinbox.value(),
        }

        # Add task if available
        if hasattr(self, "task_combo"):
            config["task"] = self.task_combo.currentText()

        # Add custom prompt if available - this overrides defaults from prompts.json
        if hasattr(self, "custom_prompt"):
            custom_text = self.custom_prompt.toPlainText().strip()
            if custom_text:
                config["prompt_template"] = custom_text
            # If no custom prompt, use the loaded prompt from prompts.json
            elif self.prompt_service and hasattr(self, "task_combo"):
                try:
                    current_task = self.task_combo.currentText()
                    current_model = self.model_combo.currentText()
                    if current_task and current_model:
                        prompt_config = self.prompt_service.get_prompt_config(
                            current_task, current_model
                        )
                        if prompt_config and hasattr(prompt_config, "prompt"):
                            config["prompt_template"] = prompt_config.prompt
                except Exception as e:
                    pass  # Fall back to default prompt

        # Add system prompt if available - this overrides defaults from prompts.json
        if hasattr(self, "system_prompt"):
            system_text = self.system_prompt.toPlainText().strip()
            if system_text:
                config["system_prompt"] = system_text
            # If no custom system prompt, use the loaded system prompt from prompts.json
            elif self.prompt_service and hasattr(self, "task_combo"):
                try:
                    current_task = self.task_combo.currentText()
                    current_model = self.model_combo.currentText()
                    if current_task and current_model:
                        prompt_config = self.prompt_service.get_prompt_config(
                            current_task, current_model
                        )
                        if prompt_config and hasattr(prompt_config, "system"):
                            config["system_prompt"] = prompt_config.system
                except Exception as e:
                    pass  # Fall back to default system prompt

        # Add keyword chunking parameters if available (keywords step only)
        if hasattr(self, "chunking_threshold_spinbox"):
            config["keyword_chunking_threshold"] = (
                self.chunking_threshold_spinbox.value()
            )
        if hasattr(self, "chunking_task_combo"):
            config["chunking_task"] = self.chunking_task_combo.currentText()
            
        # Add DK classification parameters if available (dk_classification step only) - Claude Generated
        if hasattr(self, "dk_frequency_spinbox"):
            config["dk_frequency_threshold"] = self.dk_frequency_spinbox.value()

        return config

    def set_config(self, config: Dict[str, Any]):
        """Set configuration - Claude Generated"""
        self.config = config
        if "enabled" in config:
            self.enabled_checkbox.setChecked(config["enabled"])

        if "provider" in config:
            index = self.provider_combo.findText(config["provider"])
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)

        if "model" in config:
            index = self.model_combo.findText(config["model"])
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

        if "temperature" in config:
            temp_value = config["temperature"]
            self.temp_spinbox.setValue(temp_value)
            # Update slider to match
            self.temp_slider.setValue(int(temp_value * 100))

        if "top_p" in config:
            p_value = config["top_p"]
            self.p_spinbox.setValue(p_value)
            # Update slider to match
            self.p_slider.setValue(int(p_value * 100))

        if "task" in config and hasattr(self, "task_combo"):
            index = self.task_combo.findText(config["task"])
            if index >= 0:
                self.task_combo.setCurrentIndex(index)

        if "prompt_template" in config and hasattr(self, "custom_prompt"):
            self.custom_prompt.setPlainText(config["prompt_template"])
        elif "custom_prompt" in config and hasattr(self, "custom_prompt"):
            self.custom_prompt.setPlainText(config["custom_prompt"])

        if "system_prompt" in config and hasattr(self, "system_prompt"):
            self.system_prompt.setPlainText(config["system_prompt"])

        # Set keyword chunking parameters if available (keywords step only)
        if "keyword_chunking_threshold" in config and hasattr(
            self, "chunking_threshold_spinbox"
        ):
            self.chunking_threshold_spinbox.setValue(
                config["keyword_chunking_threshold"]
            )

        if "chunking_task" in config and hasattr(self, "chunking_task_combo"):
            index = self.chunking_task_combo.findText(config["chunking_task"])
            if index >= 0:
                self.chunking_task_combo.setCurrentIndex(index)
                
        # Set DK classification parameters if available (dk_classification step only) - Claude Generated
        if "dk_frequency_threshold" in config and hasattr(self, "dk_frequency_spinbox"):
            self.dk_frequency_spinbox.setValue(config["dk_frequency_threshold"])

        # Load prompt settings after config is set (with delay to ensure UI is updated)
        if hasattr(self, "task_combo") and not (
            "temperature" in config or "top_p" in config
        ):
            # Only load prompt settings if temperature/top_p weren't explicitly set in config
            QTimer.singleShot(50, self.load_prompt_settings)


class PipelineConfigDialog(QDialog):
    """Dialog für Pipeline-Konfiguration - Claude Generated"""

    config_saved = pyqtSignal(object)  # PipelineConfig

    def __init__(
        self,
        llm_service: LlmService,
        prompt_service: PromptService = None,
        current_config: Optional[PipelineConfig] = None,
        config_manager=None,
        parent=None,
    ):
        super().__init__(parent)
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.current_config = current_config
        self.config_manager = config_manager
        self.step_widgets = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize SmartProviderSelector for intelligent defaults - Claude Generated
        self.smart_selector = None
        if config_manager:
            try:
                from ..utils.smart_provider_selector import SmartProviderSelector
                self.smart_selector = SmartProviderSelector(config_manager)
                self.logger.info("PipelineConfigDialog initialized with SmartProviderSelector")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SmartProviderSelector: {e}")
        
        self.setup_ui()

        # Load SmartProvider-based config if no explicit config provided - Claude Generated
        if not current_config and config_manager:
            try:
                from ..core.pipeline_manager import PipelineConfig
                smart_config = PipelineConfig.create_from_provider_preferences(config_manager)
                self.load_config(smart_config)
                self.logger.info("Loaded configuration from Provider Preferences")
            except Exception as e:
                self.logger.warning(f"Failed to load SmartProvider config: {e}")
        elif current_config:
            self.load_config(current_config)

    def setup_ui(self):
        """Setup der Dialog UI - Claude Generated"""
        self.setWindowTitle("Pipeline-Konfiguration")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)

        # Header
        header_label = QLabel("🚀 Pipeline-Konfiguration")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        description_label = QLabel(
            "Konfigurieren Sie Provider, Modelle und Parameter für jeden Pipeline-Schritt:"
        )
        description_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(description_label)

        # Main content with tabs for each step
        self.tab_widget = QTabWidget()

        # Define pipeline steps (using official step names from CLAUDE.md)
        pipeline_steps = [
            ("initialisation", "🔤 Initialisierung"),
            ("search", "🔍 Suche"),
            ("keywords", "✅ Schlagworte"),
            ("dk_classification", "📚 DK-Klassifikation"),
        ]

        # Create tab for each step
        for step_id, step_name in pipeline_steps:
            if step_id == "search":
                # Search step uses special SearchStepConfigWidget
                search_widget = SearchStepConfigWidget(step_name)
                self.step_widgets[step_id] = search_widget
                self.tab_widget.addTab(search_widget, step_name)
            else:
                # Use HybridStepConfigWidget for LLM steps to show Smart/Advanced/Expert modes - Claude Generated
                step_widget = HybridStepConfigWidget(
                    step_name=step_name, 
                    step_id=step_id, 
                    config_manager=self.config_manager,
                    parent=self
                )
                self.step_widgets[step_id] = step_widget
                self.tab_widget.addTab(step_widget, step_name)

        layout.addWidget(self.tab_widget)

        # Global Settings
        global_group = QGroupBox("Globale Einstellungen")
        global_layout = QVBoxLayout(global_group)

        # Auto-advance option
        self.auto_advance_checkbox = QCheckBox("Automatisch zum nächsten Schritt")
        self.auto_advance_checkbox.setChecked(True)
        self.auto_advance_checkbox.setToolTip(
            "Pipeline läuft automatisch durch alle Schritte"
        )
        global_layout.addWidget(self.auto_advance_checkbox)

        # Stop on error option
        self.stop_on_error_checkbox = QCheckBox("Bei Fehler stoppen")
        self.stop_on_error_checkbox.setChecked(True)
        self.stop_on_error_checkbox.setToolTip("Pipeline stoppt bei ersten Fehler")
        global_layout.addWidget(self.stop_on_error_checkbox)

        layout.addWidget(global_group)

        # Buttons
        button_layout = QHBoxLayout()

        # Preset buttons
        preset_button = QPushButton("📋 Preset laden")
        preset_button.clicked.connect(self.load_preset)
        button_layout.addWidget(preset_button)

        save_preset_button = QPushButton("💾 Als Preset speichern")
        save_preset_button.clicked.connect(self.save_preset)
        button_layout.addWidget(save_preset_button)
        
        # Save as provider preferences button - Claude Generated
        save_as_preferences_button = QPushButton("🎯 Als Standardeinstellung speichern")
        save_as_preferences_button.setToolTip("Speichert die aktuellen Provider-Einstellungen als universelle Standardwerte für alle ALIMA-Funktionen")
        save_as_preferences_button.clicked.connect(self.save_as_provider_preferences)
        button_layout.addWidget(save_as_preferences_button)

        button_layout.addStretch()

        # Standard dialog buttons
        cancel_button = QPushButton("Abbrechen")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        save_button = QPushButton("Speichern")
        save_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        save_button.clicked.connect(self.save_config)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)

    def load_config(self, config: PipelineConfig):
        """Load existing configuration - Claude Generated"""
        try:
            # Load step configurations
            for step_id, step_widget in self.step_widgets.items():
                if step_id in config.step_configs:
                    step_config = config.step_configs[step_id]
                    
                    # Handle both PipelineStepConfig objects and dict formats
                    if isinstance(step_config, dict):
                        # Already a dict (e.g., search step stored as dict)
                        if step_id == "search":
                            # Search step uses suggesters format
                            search_config = {"suggesters": step_config.get("suggesters", config.search_suggesters)}
                            step_widget.set_config(search_config)
                        else:
                            # Other steps stored as dict - use directly
                            step_widget.set_config(step_config)
                    else:
                        # Convert PipelineStepConfig to dict format for widget compatibility
                        config_dict = {
                            'step_id': step_config.step_id,
                            'enabled': step_config.enabled,
                            'provider': step_config.provider or '',
                            'model': step_config.model or '',
                            'task': step_config.task or '',
                            'temperature': step_config.temperature or 0.7,
                            'top_p': step_config.top_p or 0.1,
                            'max_tokens': step_config.max_tokens
                        }
                        step_widget.set_config(config_dict)
                elif step_id == "search":
                    # Load search suggesters from PipelineConfig
                    search_config = {"suggesters": config.search_suggesters}
                    step_widget.set_config(search_config)

            # Load global settings
            self.auto_advance_checkbox.setChecked(config.auto_advance)
            self.stop_on_error_checkbox.setChecked(config.stop_on_error)

        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            QMessageBox.warning(
                self, "Fehler", f"Fehler beim Laden der Konfiguration: {e}"
            )

    def _dict_to_pipeline_step_config(self, config_dict: dict, step_id: str) -> PipelineStepConfig:
        """Convert dict config to PipelineStepConfig object - Claude Generated"""
        # Handle special case for search step (no LLM params)
        if step_id == "search":
            # Return dict as-is for search (it doesn't use PipelineStepConfig)
            return config_dict

        # Extract fields that PipelineStepConfig expects
        return PipelineStepConfig(
            step_id=step_id,
            enabled=config_dict.get("enabled", True),
            provider=config_dict.get("provider"),
            model=config_dict.get("model"),
            task=config_dict.get("task"),
            temperature=config_dict.get("temperature"),
            top_p=config_dict.get("top_p"),
            max_tokens=config_dict.get("max_tokens"),
            seed=config_dict.get("seed"),
            custom_params=config_dict.get("custom_params", {}),
            task_type=config_dict.get("task_type"),
        )

    def save_config(self):
        """Save configuration using baseline + override pattern - Claude Generated"""
        try:
            # Step 1: Create smart baseline configuration
            if self.config_manager:
                # Use smart provider preferences as baseline
                baseline_config = PipelineConfig.create_from_provider_preferences(self.config_manager)
            else:
                # Fallback to default configuration
                baseline_config = PipelineConfig()

            # Step 2: Apply UI overrides for each step
            step_configs = {}
            search_suggesters = ["lobid", "swb"]  # Default

            for step_id, step_widget in self.step_widgets.items():
                if step_id == "search":
                    # Handle search step (no LLM configuration)
                    config = step_widget.get_config()
                    if "suggesters" in config:
                        search_suggesters = config["suggesters"]
                    step_configs[step_id] = config
                else:
                    # Handle LLM steps with baseline + override logic
                    widget_config = step_widget.get_config()

                    # Check if user made manual selections (overrides)
                    has_provider_override = widget_config.get("provider") and widget_config["provider"] != ""
                    has_model_override = widget_config.get("model") and widget_config["model"] != ""

                    if has_provider_override or has_model_override:
                        # User made manual selections -> apply as overrides
                        step_configs[step_id] = widget_config
                        self.logger.info(f"Step '{step_id}': applying UI overrides (provider={widget_config.get('provider')}, model={widget_config.get('model')})")
                    else:
                        # No manual selections -> use baseline (smart selection)
                        # Create minimal config that will trigger smart selection
                        step_configs[step_id] = {
                            "step_id": step_id,
                            "enabled": widget_config.get("enabled", True),
                            "provider": None,  # Will use smart selection
                            "model": None      # Will use smart selection
                        }
                        self.logger.info(f"Step '{step_id}': using smart baseline (no overrides)")

            # Step 3: Convert dict configs to PipelineStepConfig objects - Claude Generated
            step_configs_converted = {}
            for step_id, config_data in step_configs.items():
                if isinstance(config_data, dict):
                    step_configs_converted[step_id] = self._dict_to_pipeline_step_config(config_data, step_id)
                else:
                    # Already a PipelineStepConfig object
                    step_configs_converted[step_id] = config_data

            # Step 4: Create final configuration with converted objects
            final_config = PipelineConfig(
                auto_advance=self.auto_advance_checkbox.isChecked(),
                stop_on_error=self.stop_on_error_checkbox.isChecked(),
                step_configs=step_configs_converted,
                search_suggesters=search_suggesters,
            )

            self.logger.info("Configuration saved using baseline + override pattern")
            self.config_saved.emit(final_config)
            self.accept()

        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            QMessageBox.critical(self, "Fehler", f"Fehler beim Speichern: {e}")

    def refresh_from_settings(self):
        """Refresh all step widgets from updated settings/task preferences - Claude Generated"""
        try:
            self.logger.info("🔄 Refreshing pipeline configuration from updated settings")

            # Refresh each step widget's provider/model selection
            for step_id, widget in self.step_widgets.items():
                try:
                    # Re-initialize with updated preferences
                    widget._initialize_with_preferred_settings()

                    # Update UI displays
                    widget._update_smart_preview()
                    widget._validate_configuration()

                    self.logger.debug(f"✅ Refreshed {step_id} step widget")

                except Exception as e:
                    self.logger.warning(f"Error refreshing {step_id} step widget: {e}")

            self.logger.info("✅ Pipeline configuration refresh completed")

        except Exception as e:
            self.logger.error(f"Error refreshing pipeline configuration: {e}")
            # Show user-friendly notification
            QMessageBox.information(
                self,
                "Settings Update",
                "Pipeline configuration has been updated to reflect the latest settings changes."
            )

    def load_preset(self):
        """Load a configuration preset - Claude Generated"""
        # TODO: Implement preset loading from file
        QMessageBox.information(
            self, "Preset laden", "Preset-Funktion wird implementiert..."
        )

    def save_preset(self):
        """Save current configuration as preset - Claude Generated"""
        # TODO: Implement preset saving to file
        QMessageBox.information(
            self, "Preset speichern", "Preset-Speichern wird implementiert..."
        )
    
    def save_as_provider_preferences(self):
        """Save current pipeline configuration as universal provider preferences - Claude Generated"""
        if not self.config_manager:
            QMessageBox.warning(
                self, "Konfiguration nicht verfügbar", 
                "ConfigManager ist nicht verfügbar. Provider-Einstellungen können nicht gespeichert werden."
            )
            return
            
        try:
            # Get current configuration from UI
            current_config = self.get_config()
            
            # Extract provider preferences from pipeline config
            unified_config = self.config_manager.get_unified_config()
            
            # Update provider preferences based on pipeline step configurations
            step_configs = current_config.step_configs
            
            # Determine the most frequently used provider as preferred
            provider_counts = {}
            for step_config in step_configs.values():
                if step_config.provider and step_config.enabled:
                    provider = step_config.provider
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            if provider_counts:
                # Set most used provider as preferred
                most_used_provider = max(provider_counts, key=provider_counts.get)
                unified_config.preferred_provider = most_used_provider

                # Update provider priority based on usage
                sorted_providers = sorted(provider_counts.keys(), key=provider_counts.get, reverse=True)
                # Keep existing priority for unused providers, append at end
                existing_priority = unified_config.provider_priority[:]
                new_priority = sorted_providers[:]
                for provider in existing_priority:
                    if provider not in new_priority:
                        new_priority.append(provider)
                unified_config.provider_priority = new_priority
            
            # Update task-specific overrides based on pipeline config
            if 'initialisation' in step_configs and step_configs['initialisation'].enabled:
                # Fast text provider for initialization
                init_provider = step_configs['initialisation'].provider
                if init_provider:
                    # TODO: Implement task-specific provider overrides in UnifiedProviderConfig
                    pass  # Disabled until proper implementation
                    
            if 'keywords' in step_configs and step_configs['keywords'].enabled:
                # Quality text provider for final analysis
                keywords_provider = step_configs['keywords'].provider
                if keywords_provider:
                    # TODO: Implement task-specific provider overrides in UnifiedProviderConfig
                    pass  # Disabled until proper implementation
                    
            if 'dk_classification' in step_configs and step_configs['dk_classification'].enabled:
                # Classification-specific provider
                classification_provider = step_configs['dk_classification'].provider
                if classification_provider:
                    # TODO: Implement task-specific provider overrides in UnifiedProviderConfig
                    pass  # Disabled until proper implementation
            
            # Update preferred models per provider
            for step_config in step_configs.values():
                if step_config.provider and step_config.model and step_config.enabled:
                    provider = step_config.provider
                    model = step_config.model
                    if provider and model:
                        # TODO: Implement preferred_models in UnifiedProviderConfig
                        pass  # Disabled until proper implementation
            
            # TODO: Implement validation in UnifiedProviderConfig if needed
            # if self.smart_selector:
            #     validation_issues = unified_config.validate_preferences(self.smart_selector.provider_detection_service)
            # TODO: Re-implement validation block when UnifiedProviderConfig supports validation\n            if False:  # Disabled: any(validation_issues.values()):
                    # Show validation issues but allow saving
                    issues_text = ""
                    for category, issues in validation_issues.items():
                        if issues:
                            category_name = category.replace('_', ' ').title()
                            issues_text += f"**{category_name}:**\n"
                            for issue in issues[:3]:  # Show first 3 issues
                                issues_text += f"  • {issue}\n"
                            if len(issues) > 3:
                                issues_text += f"  • ... und {len(issues) - 3} weitere\n"
                            issues_text += "\n"
                    
                    reply = QMessageBox.question(
                        self,
                        "Konfigurationsvalidierung",
                        f"⚠️ Einige Provider-Einstellungen haben Probleme:\n\n{issues_text}"
                        f"Möchten Sie trotzdem speichern? (Auto-Cleanup wird durchgeführt)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    
                    if reply == QMessageBox.StandardButton.No:
                        return
                    
                    # Perform auto-cleanup
                    # TODO: Implement cleanup in UnifiedProviderConfig
                    # cleanup_report = unified_config.auto_cleanup(self.smart_selector.provider_detection_service)
                    cleanup_report = {}
                    if cleanup_report and any(cleanup_report.values()):
                        self.logger.info("Auto-cleanup performed during provider preferences save")
            
            # Save updated config directly
            self.config_manager.save_config()
            
            # Success message with summary
            success_message = "✅ Provider-Einstellungen erfolgreich gespeichert!\n\n"
            success_message += f"📋 Bevorzugter Provider: {unified_config.preferred_provider}\n"
            success_message += f"🎯 Provider-Priorität: {', '.join(unified_config.provider_priority[:3])}"
            if len(unified_config.provider_priority) > 3:
                success_message += f" (+{len(unified_config.provider_priority) - 3} weitere)"
            success_message += f"\n🚀 Konfiguration erfolgreich gespeichert\n\n"
            success_message += "Diese Einstellungen werden jetzt als Standardwerte für alle ALIMA-Funktionen verwendet."
            
            QMessageBox.information(self, "Erfolgreich gespeichert", success_message)
            
        except Exception as e:
            self.logger.error(f"Error saving provider preferences: {e}")
            QMessageBox.critical(
                self, "Fehler beim Speichern", 
                f"Fehler beim Speichern der Provider-Einstellungen:\n\n{str(e)}"
            )
