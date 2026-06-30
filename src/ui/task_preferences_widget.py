#!/usr/bin/env python3
"""Task-specific model-preference editor — Claude Generated.

Extracted from ``unified_provider_tab.py`` (F-5 god-file split, June 30): the
"🎯 Task Preferences" sub-tab and its model-priority editing logic now live here
as a self-contained widget. ``UnifiedProviderTab`` embeds it, pushes the available
provider/model lists via :meth:`set_available_models`, calls :meth:`load` /
:meth:`save_if_clean`, and forwards the widget's ``config_changed`` /
``task_preferences_changed`` / ``save_toast`` signals.

Coupling is config + cached models only (no provider-table access).
"""

import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from ..utils.config_models import TaskPreference, TaskType as UnifiedTaskType
from ..utils.model_capabilities import get_chunking_threshold


class TaskPreferencesWidget(QWidget):
    """Per-task model-priority editor (the Task Preferences sub-tab)."""

    config_changed = pyqtSignal()
    task_preferences_changed = pyqtSignal()
    save_toast = pyqtSignal(str, bool)  # message, is_error

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

    def __init__(self, unified_config, alima_config, config_manager,
                 alima_manager=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.unified_config = unified_config
        self.config = alima_config
        self.config_manager = config_manager
        self.alima_manager = alima_manager
        # Task editing state (was on UnifiedProviderTab) - Claude Generated
        self.current_editing_task = None
        self.task_ui_dirty = False
        # Provider/model lists are pushed by the host tab (it runs the bg fetch).
        self._cached_providers = []
        self._cached_models = {}
        self._setup_ui()

    # -- public interface ---------------------------------------------------
    def set_available_models(self, providers, models) -> None:
        """Receive the available provider/model lists from the host tab."""
        self._cached_providers = list(providers or [])
        self._cached_models = dict(models or {})

    def load(self) -> None:
        """Populate task categories + initialise editing state."""
        self._populate_task_preferences()
        self._initialize_task_editing_state()

    def save_if_clean(self) -> None:
        """Persist the current task's prefs on a global save, if UI state is clean."""
        if self.current_editing_task and not self.task_ui_dirty:
            self.logger.debug(f"Global save: saving clean task preferences for {self.current_editing_task}")
            self._save_current_task_preferences(explicit_task_name=self.current_editing_task)
        elif self.current_editing_task and self.task_ui_dirty:
            self.logger.warning(f"Global save: skipping task preferences for {self.current_editing_task} (dirty UI)")

    def _show_save_toast(self, message: str, duration: int = 2000, error: bool = False) -> None:
        """Forward toast requests to the host tab via signal - Claude Generated"""
        self.save_toast.emit(message, error)

    def _setup_ui(self) -> None:
        """Build the task-preferences UI (adapted from the former _create_preferences_tab)."""
        layout = QVBoxLayout(self)
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
