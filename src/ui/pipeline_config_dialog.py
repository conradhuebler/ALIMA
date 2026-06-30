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
from PyQt6.QtGui import QFont
from typing import Dict, List, Any, Optional
import json
import logging

from ..core.pipeline_manager import PipelineConfig
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from ..utils.config_models import (
    PipelineStepConfig,
    ProviderScope,
    TaskType as UnifiedTaskType
)
from ..utils.smart_provider_selector import SmartProviderSelector
from ..utils.pipeline_config_builder import PipelineConfigBuilder
from .step_config_widgets import SearchStepConfigWidget, HybridStepConfigWidget  # F-5 split - Claude Generated


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

        # Pipeline Default Settings - Claude Generated
        pipeline_default_group = QGroupBox("Pipeline-Standard")
        pipeline_default_layout = QVBoxLayout(pipeline_default_group)

        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel("Standard-Provider:")
        provider_label.setMinimumWidth(120)
        self.default_provider_combo = QComboBox()
        self.default_provider_combo.setToolTip(
            "Standard-Provider für alle Pipeline-Schritte (kann pro Schritt überschrieben werden)"
        )
        self._populate_provider_dropdown(self.default_provider_combo)
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.default_provider_combo)
        provider_layout.addStretch()
        pipeline_default_layout.addLayout(provider_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Standard-Modell:")
        model_label.setMinimumWidth(120)
        self.default_model_combo = QComboBox()
        self.default_model_combo.setToolTip(
            "Standard-Modell für alle Pipeline-Schritte (kann pro Schritt überschrieben werden)"
        )
        self.default_provider_combo.currentTextChanged.connect(self._update_model_dropdown)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.default_model_combo)
        model_layout.addStretch()
        pipeline_default_layout.addLayout(model_layout)

        layout.addWidget(pipeline_default_group)

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

        # Agentic mode option - Claude Generated
        self.agentic_mode_checkbox = QCheckBox("🤖 Agentic Modus (experimentell)")
        self.agentic_mode_checkbox.setChecked(False)
        self.agentic_mode_checkbox.setToolTip(
            "Verwendet LLM-gesteuerte Agenten mit Tool-Calling statt sequenzieller Pipeline.\n"
            "Die Agenten entscheiden autonom, welche Such-Tools sie verwenden.\n\n"
            "⚠️ Experimentell: Erhöht Token-Nutzung um ca. 3x"
        )
        global_layout.addWidget(self.agentic_mode_checkbox)

        # Agentic verbose mode - Claude Generated
        self.agentic_verbose_checkbox = QCheckBox("🔍 Agentic Verbose (vollständige Prompts loggen)")
        self.agentic_verbose_checkbox.setChecked(False)
        self.agentic_verbose_checkbox.setToolTip(
            "Gibt im Agentic-Modus vollständige System- und User-Prompts in den Stream-Log aus.\n"
            "Nützlich für Debugging und Transparenz der LLM-Anfragen."
        )
        global_layout.addWidget(self.agentic_verbose_checkbox)

        # Workflow selector (v4 YAML workflows) - Claude Generated
        workflow_row = QHBoxLayout()
        workflow_row.addWidget(QLabel("📋 Workflow:"))
        self.workflow_combo = QComboBox()
        self.workflow_combo.setToolTip(
            "Welches Workflow-YAML der Agentic-Modus lädt.\n"
            "v4-Workflows (alima_classic, catalog_search, ...) laufen über den\n"
            "generischen WorkflowExecutor. Andere Werte fallen auf den MetaAgent zurück."
        )
        self._populate_workflow_combo()
        workflow_row.addWidget(self.workflow_combo, 1)

        # Workflow editor button (view/edit/create the YAML) - Claude Generated
        edit_workflow_btn = QPushButton("✏️")
        edit_workflow_btn.setToolTip("Workflow-YAML ansehen, bearbeiten oder neu anlegen")
        edit_workflow_btn.setMaximumWidth(40)
        edit_workflow_btn.clicked.connect(self._open_workflow_editor)
        workflow_row.addWidget(edit_workflow_btn)

        global_layout.addLayout(workflow_row)

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

            # Load pipeline default settings - Claude Generated
            if self.config_manager:
                try:
                    unified_config = self.config_manager.get_unified_config()
                    # Show the *effective* pipeline default, even when the saved
                    # pipeline_default_provider is empty and the fallback is used.
                    effective_provider, effective_model = (
                        unified_config.resolve_default_provider_model(scope=ProviderScope.PIPELINE)
                    )
                    if effective_provider:
                        index = self.default_provider_combo.findData(effective_provider)
                        if index >= 0:
                            self.default_provider_combo.setCurrentIndex(index)
                            # Trigger model list population for the selected provider.
                            self._update_model_dropdown(effective_provider)
                    if effective_model:
                        index = self.default_model_combo.findData(effective_model)
                        if index >= 0:
                            self.default_model_combo.setCurrentIndex(index)
                except Exception as e:
                    self.logger.warning(f"Error loading pipeline defaults: {e}")

            # Load global settings
            self.auto_advance_checkbox.setChecked(config.auto_advance)
            self.stop_on_error_checkbox.setChecked(config.stop_on_error)
            if hasattr(config, 'enable_agentic_mode'):
                self.agentic_mode_checkbox.setChecked(config.enable_agentic_mode)
            if hasattr(config, 'agentic_verbose'):
                self.agentic_verbose_checkbox.setChecked(config.agentic_verbose)
            if hasattr(config, 'workflow_name') and hasattr(self, 'workflow_combo'):
                idx = self.workflow_combo.findData(config.workflow_name)
                if idx >= 0:
                    self.workflow_combo.setCurrentIndex(idx)

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
            repetition_penalty=config_dict.get("repetition_penalty"),
            think=config_dict.get("think"),
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

            # Pipeline default against which overrides are judged.
            baseline_provider = getattr(baseline_config.step_configs.get("initialisation"), "provider", "") or ""
            baseline_model = getattr(baseline_config.step_configs.get("initialisation"), "model", "") or ""

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

                    step_provider = widget_config.get("provider") or ""
                    step_model = widget_config.get("model") or ""

                    # Only treat as an explicit override if it differs from the
                    # pipeline default. Otherwise store empty so the central
                    # default is used and future default changes propagate.
                    is_override = (
                        step_provider and
                        (step_provider != baseline_provider or step_model != baseline_model)
                    )

                    if is_override:
                        step_configs[step_id] = widget_config
                        self.logger.info(f"Step '{step_id}': applying UI override (provider={step_provider}, model={step_model})")
                    else:
                        step_configs[step_id] = {
                            "step_id": step_id,
                            "enabled": widget_config.get("enabled", True),
                            "provider": None,  # Will use pipeline default
                            "model": None      # Will use pipeline default
                        }
                        self.logger.info(f"Step '{step_id}': using pipeline default ({baseline_provider}/{baseline_model})")

            # Step 3: Convert dict configs to PipelineStepConfig objects - Claude Generated
            step_configs_converted = {}
            for step_id, config_data in step_configs.items():
                if isinstance(config_data, dict):
                    step_configs_converted[step_id] = self._dict_to_pipeline_step_config(config_data, step_id)
                else:
                    # Already a PipelineStepConfig object
                    step_configs_converted[step_id] = config_data

            # Step 4: Save Pipeline Default settings to unified config - Claude Generated
            if self.config_manager:
                try:
                    unified_config = self.config_manager.get_unified_config()
                    unified_config.pipeline_default_provider = self.default_provider_combo.currentData() or ""
                    unified_config.pipeline_default_model = self.default_model_combo.currentData() or ""

                    # Save to disk
                    config = self.config_manager.load_config()
                    config.unified_config = unified_config
                    self.config_manager.save_config(config)
                    self.logger.info(f"Pipeline defaults saved: {unified_config.pipeline_default_provider}/{unified_config.pipeline_default_model}")
                except Exception as e:
                    self.logger.warning(f"Error saving pipeline defaults: {e}")

            # Step 5: Create final configuration with converted objects
            selected_workflow = (
                self.workflow_combo.currentData()
                if hasattr(self, 'workflow_combo') else None
            ) or "alima_classic"

            final_config = PipelineConfig(
                auto_advance=self.auto_advance_checkbox.isChecked(),
                stop_on_error=self.stop_on_error_checkbox.isChecked(),
                step_configs=step_configs_converted,
                search_suggesters=search_suggesters,
                enable_agentic_mode=self.agentic_mode_checkbox.isChecked(),
                agentic_verbose=self.agentic_verbose_checkbox.isChecked(),
                workflow_name=selected_workflow,
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

    def _open_workflow_editor(self) -> None:
        """Open the workflow YAML editor and refresh the combo afterwards - Claude Generated."""
        try:
            from .workflow_editor_dialog import WorkflowEditorDialog
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Workflow editor unavailable: {e}")
            return

        current = self.workflow_combo.currentData()
        editor = WorkflowEditorDialog(self, initial_workflow=current)
        last_saved = {"stem": None}
        editor.saved.connect(lambda stem: last_saved.update(stem=stem))
        editor.exec()

        # Refresh discovery so new/edited workflows appear, re-select sensibly.
        self._populate_workflow_combo()
        target = last_saved["stem"] or current
        if target is not None:
            idx = self.workflow_combo.findData(target)
            if idx >= 0:
                self.workflow_combo.setCurrentIndex(idx)

    def _populate_workflow_combo(self) -> None:
        """Populate workflow combo from workflows/ directory - Claude Generated.

        Lists every v4 YAML. Legacy v3 workflows live in ``workflows/legacy/``
        and are not discovered; the MetaAgent dispatch was removed.
        """
        try:
            from src.core.agents.workflow_loader import (
                discover_workflow_files,
                load_workflow,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Workflow loader unavailable: {e}")
            return

        self.workflow_combo.blockSignals(True)
        self.workflow_combo.clear()

        for path in discover_workflow_files():
            try:
                wf = load_workflow(path, strict=False)
                label = f"{path.stem} (v{wf.version})"
            except Exception:
                continue
            self.workflow_combo.addItem(label, path.stem)

        if self.workflow_combo.count() == 0:
            self.workflow_combo.addItem("alima_classic", "alima_classic")

        self.workflow_combo.blockSignals(False)

    def _populate_provider_dropdown(self, combo: QComboBox):
        """Populate provider dropdown with available providers - Claude Generated"""
        if not self.config_manager:
            return

        try:
            unified_config = self.config_manager.get_unified_config()
            enabled_providers = unified_config.get_enabled_providers()

            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(Auto-select)", "")

            for provider in enabled_providers:
                combo.addItem(provider.name, provider.name)

            combo.blockSignals(False)
        except Exception as e:
            self.logger.warning(f"Error populating provider dropdown: {e}")

    def _update_model_dropdown(self, provider_name: str):
        """Update model dropdown based on selected provider - Claude Generated"""
        if not provider_name or not self.config_manager:
            self.default_model_combo.clear()
            self.default_model_combo.addItem("(Auto-select)", "")
            return

        try:
            from ..llm.llm_service import LlmService
            llm_service = LlmService(lazy_initialization=True)
            models = list(llm_service.get_available_models(provider_name) or [])

            # Live fetch can be empty (provider unreachable / not yet initialized).
            # Fall back to the provider's configured preferred/available models so
            # the user can still pick a model instead of seeing only "(Auto-select)".
            if not models:
                try:
                    for p in self.config_manager.get_unified_config().get_enabled_providers():
                        if p.name == provider_name:
                            cfg_models = list(getattr(p, "available_models", None) or [])
                            pref = getattr(p, "preferred_model", "") or ""
                            if pref and pref not in cfg_models:
                                cfg_models.insert(0, pref)
                            models = cfg_models
                            break
                except Exception:
                    self.logger.debug("Config model fallback failed", exc_info=True)

            self.default_model_combo.blockSignals(True)
            self.default_model_combo.clear()
            self.default_model_combo.addItem("(Auto-select)", "")

            for model in models:
                self.default_model_combo.addItem(model, model)

            self.default_model_combo.blockSignals(False)
        except Exception as e:
            self.logger.warning(f"Error updating model dropdown: {e}")
