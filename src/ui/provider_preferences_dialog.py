#!/usr/bin/env python3
"""
Provider Preferences Dialog - LLM Provider Configuration UI
Allows users to configure universal LLM provider preferences for all tasks.
Claude Generated
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QListWidget, QListWidgetItem, QCheckBox, QSpinBox,
    QPushButton, QLabel, QLineEdit, QMessageBox, QTabWidget,
    QWidget, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSplitter
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

from typing import Dict, List, Optional
import logging

from ..utils.config_manager import ConfigManager, ProviderPreferences
from ..utils.smart_provider_selector import SmartProviderSelector, TaskType


class ProviderPreferencesDialog(QDialog):
    """Dialog for configuring LLM provider preferences - Claude Generated"""
    
    preferences_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.config_manager = ConfigManager()
        
        # Load current preferences
        self.preferences = self.config_manager.get_provider_preferences()
        self.original_preferences = None  # For cancel functionality
        
        self.setWindowTitle("LLM Provider Preferences")
        self.setModal(True)
        self.resize(800, 600)
        
        self.init_ui()
        self.load_preferences()
        
    def init_ui(self):
        """Initialize the user interface - Claude Generated"""
        layout = QVBoxLayout(self)
        
        # Create tab widget for organization
        self.tabs = QTabWidget()
        
        # General Provider Tab
        self.general_tab = self.create_general_tab()
        self.tabs.addTab(self.general_tab, "General Preferences")
        
        # Task-Specific Tab
        self.task_tab = self.create_task_specific_tab() 
        self.tabs.addTab(self.task_tab, "Task-Specific")
        
        # Models Tab
        self.models_tab = self.create_models_tab()
        self.tabs.addTab(self.models_tab, "Preferred Models")
        
        # Performance Tab
        self.performance_tab = self.create_performance_tab()
        self.tabs.addTab(self.performance_tab, "Performance")
        
        layout.addWidget(self.tabs)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.test_button = QPushButton("Test Current Settings")
        self.test_button.clicked.connect(self.test_settings)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.save_button = QPushButton("Save & Apply")
        self.save_button.clicked.connect(self.save_and_apply)
        self.save_button.setDefault(True)
        
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
    
    def create_general_tab(self) -> QWidget:
        """Create the general preferences tab - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Provider Priority Group
        priority_group = QGroupBox("Provider Priority")
        priority_layout = QVBoxLayout(priority_group)
        
        # Preferred provider selection
        preferred_layout = QFormLayout()
        self.preferred_provider_combo = QComboBox()
        self.preferred_provider_combo.addItems([
            "ollama", "gemini", "anthropic", "openai", "chatai"
        ])
        preferred_layout.addRow("Preferred Provider:", self.preferred_provider_combo)
        priority_layout.addLayout(preferred_layout)
        
        # Provider priority list
        priority_label = QLabel("Fallback Priority Order:")
        priority_label.setFont(QFont("", 10, QFont.Weight.Bold))
        priority_layout.addWidget(priority_label)
        
        # Create splitter for priority list and buttons
        priority_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.priority_list = QListWidget()
        self.priority_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        priority_splitter.addWidget(self.priority_list)
        
        # Priority control buttons
        priority_buttons_widget = QWidget()
        priority_buttons_layout = QVBoxLayout(priority_buttons_widget)
        
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(self.move_priority_up)
        
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(self.move_priority_down)
        
        priority_buttons_layout.addWidget(self.move_up_button)
        priority_buttons_layout.addWidget(self.move_down_button)
        priority_buttons_layout.addStretch()
        
        priority_splitter.addWidget(priority_buttons_widget)
        priority_layout.addWidget(priority_splitter)
        
        layout.addWidget(priority_group)
        
        # Disabled Providers Group
        disabled_group = QGroupBox("Disabled Providers")
        disabled_layout = QVBoxLayout(disabled_group)
        
        disabled_label = QLabel("Select providers to disable completely:")
        disabled_layout.addWidget(disabled_label)
        
        # Create checkboxes for each provider
        self.disabled_checkboxes = {}
        providers = ["gemini", "anthropic", "openai", "ollama", "chatai"]
        
        for provider in providers:
            checkbox = QCheckBox(provider.title())
            self.disabled_checkboxes[provider] = checkbox
            disabled_layout.addWidget(checkbox)
        
        layout.addWidget(disabled_group)
        
        # Fallback Settings Group
        fallback_group = QGroupBox("Fallback Behavior")
        fallback_layout = QFormLayout(fallback_group)
        
        self.auto_fallback_checkbox = QCheckBox()
        fallback_layout.addRow("Enable Auto-Fallback:", self.auto_fallback_checkbox)
        
        self.fallback_timeout_spin = QSpinBox()
        self.fallback_timeout_spin.setRange(5, 300)
        self.fallback_timeout_spin.setSuffix(" seconds")
        fallback_layout.addRow("Fallback Timeout:", self.fallback_timeout_spin)
        
        layout.addWidget(fallback_group)
        
        layout.addStretch()
        return tab
    
    def create_task_specific_tab(self) -> QWidget:
        """Create the task-specific overrides tab - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Task-Specific Overrides Group
        task_group = QGroupBox("Task-Specific Provider Overrides")
        task_layout = QFormLayout(task_group)
        
        # Vision tasks
        self.vision_provider_combo = QComboBox()
        self.vision_provider_combo.addItems(["(use general preference)", "gemini", "openai", "anthropic", "ollama"])
        task_layout.addRow("Vision/Image Tasks:", self.vision_provider_combo)
        
        # Text-only tasks
        self.text_provider_combo = QComboBox()
        self.text_provider_combo.addItems(["(use general preference)", "ollama", "gemini", "anthropic", "openai", "chatai"])
        task_layout.addRow("Text-Only Tasks:", self.text_provider_combo)
        
        # Classification tasks
        self.classification_provider_combo = QComboBox()
        self.classification_provider_combo.addItems(["(use general preference)", "gemini", "anthropic", "openai", "ollama", "chatai"])
        task_layout.addRow("Classification Tasks:", self.classification_provider_combo)
        
        layout.addWidget(task_group)
        
        # Task Capabilities Information
        info_group = QGroupBox("Provider Capabilities Reference")
        info_layout = QVBoxLayout(info_group)
        
        capabilities_text = """
        <b>Vision Capabilities:</b>
        • Gemini: Excellent vision, fast processing
        • OpenAI (GPT-4o): High-quality image analysis
        • Anthropic (Claude): Good vision, detailed analysis
        • Ollama (LLaVA): Local vision models, privacy-focused
        
        <b>Text Processing Strengths:</b>
        • Ollama: Local processing, privacy, custom models
        • Gemini: Fast, good for German content
        • Anthropic: Thoughtful analysis, good reasoning
        • OpenAI: General purpose, function calling
        • ChatAI: Academic focus, German language support
        
        <b>Classification Tasks:</b>
        • Gemini: Fast classification, good for libraries
        • Anthropic: Detailed reasoning, hierarchy understanding
        • OpenAI: Consistent classification, good accuracy
        """
        
        capabilities_label = QLabel(capabilities_text)
        capabilities_label.setWordWrap(True)
        capabilities_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        info_layout.addWidget(capabilities_label)
        
        layout.addWidget(info_group)
        layout.addStretch()
        return tab
    
    def create_models_tab(self) -> QWidget:
        """Create the preferred models configuration tab - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Preferred Models Group
        models_group = QGroupBox("Preferred Models per Provider")
        models_layout = QVBoxLayout(models_group)
        
        # Create table for model preferences
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(2)
        self.models_table.setHorizontalHeaderLabels(["Provider", "Preferred Model"])
        
        # Configure table appearance
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        self.models_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.models_table.setAlternatingRowColors(True)
        
        # Populate with providers
        providers = ["ollama", "gemini", "anthropic", "openai", "chatai"]
        self.models_table.setRowCount(len(providers))
        
        self.model_inputs = {}
        for i, provider in enumerate(providers):
            # Provider name (read-only)
            provider_item = QTableWidgetItem(provider.title())
            provider_item.setFlags(provider_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.models_table.setItem(i, 0, provider_item)
            
            # Model input field
            model_input = QLineEdit()
            model_input.setPlaceholderText(f"Enter preferred model for {provider}")
            self.models_table.setCellWidget(i, 1, model_input)
            self.model_inputs[provider] = model_input
        
        models_layout.addWidget(self.models_table)
        
        # Model suggestions
        suggestions_layout = QHBoxLayout()
        suggestions_label = QLabel("Quick Fill:")
        suggestions_layout.addWidget(suggestions_label)
        
        self.quality_button = QPushButton("Quality Models")
        self.quality_button.clicked.connect(self.fill_quality_models)
        self.quality_button.setToolTip("Fill with high-quality models optimized for accuracy")
        
        self.speed_button = QPushButton("Speed Models") 
        self.speed_button.clicked.connect(self.fill_speed_models)
        self.speed_button.setToolTip("Fill with fast models optimized for speed")
        
        suggestions_layout.addWidget(self.quality_button)
        suggestions_layout.addWidget(self.speed_button)
        suggestions_layout.addStretch()
        
        models_layout.addLayout(suggestions_layout)
        layout.addWidget(models_group)
        
        layout.addStretch()
        return tab
    
    def create_performance_tab(self) -> QWidget:
        """Create the performance preferences tab - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Performance Preferences Group
        perf_group = QGroupBox("Performance Preferences")
        perf_layout = QFormLayout(perf_group)
        
        self.prefer_faster_checkbox = QCheckBox()
        self.prefer_faster_checkbox.setToolTip("Prioritize speed over quality when selecting models")
        perf_layout.addRow("Prefer Faster Models:", self.prefer_faster_checkbox)
        
        # Future feature - cost management
        self.max_cost_input = QLineEdit()
        self.max_cost_input.setPlaceholderText("Not implemented yet")
        self.max_cost_input.setEnabled(False)
        perf_layout.addRow("Max Cost per Request:", self.max_cost_input)
        
        layout.addWidget(perf_group)
        
        # Provider Statistics Group (if SmartProviderSelector is available)
        stats_group = QGroupBox("Provider Performance Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        try:
            selector = SmartProviderSelector()
            stats = selector.get_provider_stats()
            
            # Create table for statistics
            stats_table = QTableWidget()
            stats_table.setColumnCount(4)
            stats_table.setHorizontalHeaderLabels([
                "Provider", "Avg Response Time", "Total Requests", "Availability"
            ])
            
            stats_table.setRowCount(len(stats))
            for i, (provider, provider_stats) in enumerate(stats.items()):
                # Provider name
                stats_table.setItem(i, 0, QTableWidgetItem(provider.title()))
                
                # Average response time
                avg_time = provider_stats.get("average_response_time")
                time_text = f"{avg_time:.2f}s" if avg_time else "N/A"
                stats_table.setItem(i, 1, QTableWidgetItem(time_text))
                
                # Total requests
                total_requests = provider_stats.get("total_requests", 0)
                stats_table.setItem(i, 2, QTableWidgetItem(str(total_requests)))
                
                # Availability
                is_available = provider_stats.get("is_available", False)
                availability_text = "✅ Available" if is_available else "❌ Unavailable"
                stats_table.setItem(i, 3, QTableWidgetItem(availability_text))
            
            # Configure table
            header = stats_table.horizontalHeader()
            for col in range(stats_table.columnCount()):
                header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
            
            stats_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            stats_table.setAlternatingRowColors(True)
            stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            
            stats_layout.addWidget(stats_table)
            
            # Reset stats button
            reset_stats_button = QPushButton("Reset Performance Statistics")
            reset_stats_button.clicked.connect(self.reset_performance_stats)
            stats_layout.addWidget(reset_stats_button)
            
        except Exception as e:
            error_label = QLabel(f"Could not load performance statistics: {str(e)}")
            error_label.setStyleSheet("color: red;")
            stats_layout.addWidget(error_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        return tab
    
    def load_preferences(self):
        """Load current preferences into UI controls - Claude Generated"""
        try:
            # Store original preferences for cancel functionality
            import copy
            self.original_preferences = copy.deepcopy(self.preferences)
            
            # General tab
            self.preferred_provider_combo.setCurrentText(self.preferences.preferred_provider)
            
            # Load priority list
            self.priority_list.clear()
            for provider in self.preferences.provider_priority:
                self.priority_list.addItem(provider)
            
            # Load disabled providers
            for provider, checkbox in self.disabled_checkboxes.items():
                checkbox.setChecked(provider in self.preferences.disabled_providers)
            
            # Fallback settings
            self.auto_fallback_checkbox.setChecked(self.preferences.auto_fallback)
            self.fallback_timeout_spin.setValue(self.preferences.fallback_timeout)
            
            # Task-specific tab
            self.vision_provider_combo.setCurrentText(
                self.preferences.vision_provider or "(use general preference)"
            )
            self.text_provider_combo.setCurrentText(
                self.preferences.text_provider or "(use general preference)"
            )
            self.classification_provider_combo.setCurrentText(
                self.preferences.classification_provider or "(use general preference)"
            )
            
            # Models tab
            for provider, input_widget in self.model_inputs.items():
                preferred_model = self.preferences.preferred_models.get(provider, "")
                input_widget.setText(preferred_model)
            
            # Performance tab
            self.prefer_faster_checkbox.setChecked(self.preferences.prefer_faster_models)
            
        except Exception as e:
            self.logger.error(f"Error loading preferences: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load preferences: {str(e)}")
    
    def save_preferences(self):
        """Save UI settings to preferences object - Claude Generated"""
        try:
            # General settings
            self.preferences.preferred_provider = self.preferred_provider_combo.currentText()
            
            # Priority list
            self.preferences.provider_priority = []
            for i in range(self.priority_list.count()):
                item = self.priority_list.item(i)
                self.preferences.provider_priority.append(item.text())
            
            # Disabled providers
            self.preferences.disabled_providers = []
            for provider, checkbox in self.disabled_checkboxes.items():
                if checkbox.isChecked():
                    self.preferences.disabled_providers.append(provider)
            
            # Fallback settings
            self.preferences.auto_fallback = self.auto_fallback_checkbox.isChecked()
            self.preferences.fallback_timeout = self.fallback_timeout_spin.value()
            
            # Task-specific settings
            vision_text = self.vision_provider_combo.currentText()
            self.preferences.vision_provider = None if vision_text == "(use general preference)" else vision_text
            
            text_text = self.text_provider_combo.currentText()
            self.preferences.text_provider = None if text_text == "(use general preference)" else text_text
            
            classification_text = self.classification_provider_combo.currentText()
            self.preferences.classification_provider = None if classification_text == "(use general preference)" else classification_text
            
            # Preferred models
            self.preferences.preferred_models = {}
            for provider, input_widget in self.model_inputs.items():
                model_text = input_widget.text().strip()
                if model_text:
                    self.preferences.preferred_models[provider] = model_text
            
            # Performance settings
            self.preferences.prefer_faster_models = self.prefer_faster_checkbox.isChecked()
            
        except Exception as e:
            self.logger.error(f"Error saving preferences: {e}")
            raise
    
    def move_priority_up(self):
        """Move selected item up in priority list - Claude Generated"""
        current_row = self.priority_list.currentRow()
        if current_row > 0:
            item = self.priority_list.takeItem(current_row)
            self.priority_list.insertItem(current_row - 1, item)
            self.priority_list.setCurrentRow(current_row - 1)
    
    def move_priority_down(self):
        """Move selected item down in priority list - Claude Generated"""
        current_row = self.priority_list.currentRow()
        if current_row < self.priority_list.count() - 1 and current_row >= 0:
            item = self.priority_list.takeItem(current_row)
            self.priority_list.insertItem(current_row + 1, item)
            self.priority_list.setCurrentRow(current_row + 1)
    
    def fill_quality_models(self):
        """Fill model inputs with quality-focused models - Claude Generated"""
        quality_models = {
            "ollama": "cogito:32b",
            "gemini": "gemini-2.0-flash",
            "anthropic": "claude-3-5-sonnet",
            "openai": "gpt-4o",
            "chatai": "gpt-4o"
        }
        
        for provider, model in quality_models.items():
            if provider in self.model_inputs:
                self.model_inputs[provider].setText(model)
    
    def fill_speed_models(self):
        """Fill model inputs with speed-focused models - Claude Generated"""
        speed_models = {
            "ollama": "cogito:14b",
            "gemini": "gemini-1.5-flash",
            "anthropic": "claude-3-haiku",
            "openai": "gpt-4o-mini",
            "chatai": "gpt-4o-mini"
        }
        
        for provider, model in speed_models.items():
            if provider in self.model_inputs:
                self.model_inputs[provider].setText(model)
    
    def test_settings(self):
        """Test the current provider settings - Claude Generated"""
        try:
            # Save current UI state to preferences temporarily
            self.save_preferences()
            
            # Update config manager with current preferences
            success = self.config_manager.update_provider_preferences(self.preferences)
            
            if not success:
                QMessageBox.critical(self, "Test Failed", "Failed to apply preferences for testing.")
                return
            
            # Test SmartProviderSelector with current settings
            selector = SmartProviderSelector(self.config_manager)
            
            test_results = []
            
            # Test different task types
            task_types = [
                (TaskType.GENERAL, "General Text Processing"),
                (TaskType.VISION, "Vision/Image Analysis"),
                (TaskType.TEXT, "Text-Only Processing"),
                (TaskType.CLASSIFICATION, "Classification Tasks")
            ]
            
            for task_type, task_name in task_types:
                try:
                    selection = selector.select_provider(task_type=task_type)
                    test_results.append(f"✅ {task_name}: {selection.provider} with {selection.model}")
                    if selection.fallback_used:
                        test_results.append(f"    (Used fallback after {selection.total_attempts} attempts)")
                except Exception as e:
                    test_results.append(f"❌ {task_name}: Failed - {str(e)}")
            
            # Show results
            results_text = "\\n".join(test_results)
            QMessageBox.information(
                self, 
                "Provider Test Results", 
                f"Test Results:\\n\\n{results_text}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Test Error", 
                f"Error testing provider settings:\\n\\n{str(e)}"
            )
    
    def reset_performance_stats(self):
        """Reset performance statistics - Claude Generated"""
        try:
            selector = SmartProviderSelector()
            selector.reset_performance_tracking()
            
            QMessageBox.information(
                self, 
                "Statistics Reset", 
                "Provider performance statistics have been reset."
            )
            
            # Refresh the performance tab
            self.tabs.removeTab(3)
            self.performance_tab = self.create_performance_tab()
            self.tabs.insertTab(3, self.performance_tab, "Performance")
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Reset Error", 
                f"Error resetting statistics:\\n\\n{str(e)}"
            )
    
    def reset_to_defaults(self):
        """Reset all preferences to defaults - Claude Generated"""
        result = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all provider preferences to defaults?\\n\\nThis will discard all your current settings.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            # Create new default preferences
            self.preferences = ProviderPreferences()
            self.load_preferences()
    
    def save_and_apply(self):
        """Save and apply the preferences - Claude Generated"""
        try:
            # Save UI state to preferences
            self.save_preferences()
            
            # Update config manager
            success = self.config_manager.update_provider_preferences(self.preferences)
            
            if success:
                self.preferences_changed.emit()
                QMessageBox.information(
                    self, 
                    "Settings Saved", 
                    "Provider preferences have been saved and applied successfully."
                )
                self.accept()
            else:
                QMessageBox.critical(
                    self, 
                    "Save Failed", 
                    "Failed to save provider preferences. Please check the logs for details."
                )
                
        except Exception as e:
            self.logger.error(f"Error saving preferences: {e}")
            QMessageBox.critical(
                self, 
                "Save Error", 
                f"Error saving preferences:\\n\\n{str(e)}"
            )
    
    def reject(self):
        """Cancel and restore original preferences - Claude Generated"""
        if self.original_preferences:
            # Restore original preferences
            self.config_manager.update_provider_preferences(self.original_preferences)
        
        super().reject()