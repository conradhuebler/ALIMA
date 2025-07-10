import sys
import json
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QComboBox,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QGroupBox,
    QDoubleSpinBox,
    QSplitter,
    QGridLayout,
    QFormLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QListWidgetItem,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon


class PromptEditorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.config_data = None
        self.current_file = "prompts.json"
        self.current_task = None
        self.current_prompt_set_index = None
        self._dirty = False # Track unsaved changes
        self.load_config()

    def initUI(self):
        self.setWindowTitle("Prompt Configuration Editor")
        self.setGeometry(100, 100, 1200, 800)

        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create toolbar
        toolbar_layout = QHBoxLayout()
        main_layout.addLayout(toolbar_layout)

        # File operations
        self.new_btn = QPushButton("New")
        self.new_btn.clicked.connect(self.new_config)
        toolbar_layout.addWidget(self.new_btn)

        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_config)
        toolbar_layout.addWidget(self.open_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_config)
        toolbar_layout.addWidget(self.save_btn)

        self.save_as_btn = QPushButton("Save As")
        self.save_as_btn.clicked.connect(self.save_config_as)
        toolbar_layout.addWidget(self.save_as_btn)

        toolbar_layout.addStretch()

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Left panel - Tree view
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Task tree
        self.task_tree = QTreeWidget()
        self.task_tree.setHeaderLabel("Tasks & Prompts")
        self.task_tree.setMinimumWidth(250)
        self.task_tree.itemClicked.connect(self.on_tree_item_clicked)
        left_layout.addWidget(self.task_tree)

        # Task operations
        task_btns_layout = QHBoxLayout()
        self.add_task_btn = QPushButton("Add Task")
        self.add_task_btn.clicked.connect(self.add_task)
        task_btns_layout.addWidget(self.add_task_btn)

        self.remove_task_btn = QPushButton("Remove Task")
        self.remove_task_btn.clicked.connect(self.remove_task)
        task_btns_layout.addWidget(self.remove_task_btn)
        left_layout.addLayout(task_btns_layout)

        # Prompt set operations
        prompt_btns_layout = QHBoxLayout()
        self.add_prompt_btn = QPushButton("Add Prompt Set")
        self.add_prompt_btn.clicked.connect(self.add_prompt_set)
        prompt_btns_layout.addWidget(self.add_prompt_btn)

        self.remove_prompt_btn = QPushButton("Remove Prompt Set")
        self.remove_prompt_btn.clicked.connect(self.remove_prompt_set)
        prompt_btns_layout.addWidget(self.remove_prompt_btn)
        left_layout.addLayout(prompt_btns_layout)

        splitter.addWidget(left_panel)

        # Right panel - Editor
        self.editor_panel = QScrollArea()
        self.editor_panel.setWidgetResizable(True)
        self.editor_content = QWidget()
        self.editor_layout = QVBoxLayout(self.editor_content)
        self.editor_panel.setWidget(self.editor_content)

        # Display "No config loaded" message
        self.no_config_label = QLabel(
            "No configuration loaded.\nUse New or Open to start editing."
        )
        self.no_config_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_config_label.setFont(QFont("Arial", 14))
        self.editor_layout.addWidget(self.no_config_label)

        splitter.addWidget(self.editor_panel)

        # Set initial state
        self.update_ui_state(False)

        # Set the splitter proportions
        splitter.setSizes([300, 900])

    def update_ui_state(self, config_loaded):
        """Update the UI based on whether a config is loaded"""
        self.save_btn.setEnabled(config_loaded)
        self.save_as_btn.setEnabled(config_loaded)
        self.add_task_btn.setEnabled(config_loaded)
        self.remove_task_btn.setEnabled(config_loaded and self.current_task is not None)
        self.add_prompt_btn.setEnabled(config_loaded and self.current_task is not None)
        self.remove_prompt_btn.setEnabled(
            config_loaded and self.current_prompt_set_index is not None
        )

    def _prompt_to_save_on_action(self) -> bool:
        """
        Prompts the user to save changes if the editor is dirty.
        Returns True if the action can proceed (changes saved/discarded or no changes),
        False if the action is cancelled.
        """
        if not self._dirty:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )

        if reply == QMessageBox.StandardButton.Save:
            self.save_config()
            return not self._dirty  # Return True only if save was successful
        elif reply == QMessageBox.StandardButton.Discard:
            return True
        else:
            return False

    def new_config(self):
        """Create a new empty configuration"""
        if not self._prompt_to_save_on_action():
            return
        self.config_data = {
            "abstract": {
                "fields": ["prompt", "system", "temp", "p-value", "model"],
                "required": ["abstract", "keywords"],
                "prompts": [],
            },
            "keywords": {},
            "dk_list": {},
            "dk_class": {},
        }
        self.current_file = None
        self.refresh_task_tree()
        self.update_ui_state(True)
        self.setWindowTitle("Prompt Configuration Editor - New Config")
        self._dirty = False

    def open_config(self):
        """Open a configuration file"""
        if not self._prompt_to_save_on_action():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Configuration File", "", "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            self.current_file = file_path
            self.load_config()

    def load_config(self):
        """Load the configuration from the current file"""
        if self.current_file and os.path.exists(self.current_file):
            try:
                with open(self.current_file, "r", encoding="utf-8") as file:
                    self.config_data = json.load(file)
                self.refresh_task_tree()
                self.update_ui_state(True)
                self.setWindowTitle(
                    f"Prompt Configuration Editor - {os.path.basename(self.current_file)}"
                )
                self._dirty = False
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to open configuration file: {str(e)}"
                )
        else:
            self.new_config()

    def save_config(self):
        """Save the current configuration"""
        if self.current_file:
            self.save_to_file(self.current_file)
        else:
            self.save_config_as()

    def save_config_as(self):
        """Save the configuration to a new file"""
        # No need to prompt for saving here, as save_to_file will handle it
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration File", "", "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            self.save_to_file(file_path)
            self.current_file = file_path
            self.setWindowTitle(
                f"Prompt Configuration Editor - {os.path.basename(file_path)}"
            )

    def save_to_file(self, file_path):
        """Save the configuration data to the specified file"""
        # First, collect any data from the current editor if it's open
        self.collect_current_editor_data()

        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(self.config_data, file, indent=4, ensure_ascii=False)
            self._dirty = False
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save configuration file: {str(e)}"
            )

    def refresh_task_tree(self):
        """Refresh the task tree with the current configuration"""
        self.task_tree.clear()

        if not self.config_data:
            return

        # Add each task to the tree
        for task_name, task_data in self.config_data.items():
            task_item = QTreeWidgetItem(self.task_tree, [task_name])
            task_item.setData(
                0, Qt.ItemDataRole.UserRole, {"type": "task", "name": task_name}
            )

            # Add prompt sets under each task
            if "prompts" in task_data and isinstance(task_data["prompts"], list):
                for idx, prompt_set in enumerate(task_data["prompts"]):
                    models = (
                        prompt_set[4]
                        if len(prompt_set) > 4 and isinstance(prompt_set[4], list)
                        else []
                    )
                    models_str = ", ".join(models) if models else "No models"
                    prompt_item = QTreeWidgetItem(
                        task_item, [f"Prompt Set {idx+1} ({models_str})"]
                    )
                    prompt_item.setData(
                        0,
                        Qt.ItemDataRole.UserRole,
                        {"type": "prompt_set", "task": task_name, "index": idx},
                    )

        self.task_tree.expandAll()

    def on_tree_item_clicked(self, item, column):
        """Handle tree item selection"""
        # Save any changes from the current editor before switching
        self.collect_current_editor_data()

        item_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_data:
            return

        item_type = item_data.get("type")

        if item_type == "task":
            task_name = item_data.get("name")
            self.current_task = task_name
            self.current_prompt_set_index = None
            self.load_task_editor(task_name)

        elif item_type == "prompt_set":
            task_name = item_data.get("task")
            prompt_idx = item_data.get("index")
            self.current_task = task_name
            self.current_prompt_set_index = prompt_idx
            self.load_prompt_set_editor(task_name, prompt_idx)

        self.update_ui_state(True)

    def add_task(self):
        """Add a new task to the configuration"""
        task_name, ok = QInputDialog.getText(
            self, "Add Task", "Enter the name of the new task:"
        )

        if ok and task_name:
            if task_name in self.config_data:
                QMessageBox.warning(
                    self, "Warning", f"Task '{task_name}' already exists."
                )
                return

            self.config_data[task_name] = {
                "fields": ["prompt", "system", "temp", "p-value", "model"],
                "required": [],
                "prompts": [],
            }

            self.refresh_task_tree()

            # Find and select the new task
            for i in range(self.task_tree.topLevelItemCount()):
                item = self.task_tree.topLevelItem(i)
                if item.text(0) == task_name:
                    self.task_tree.setCurrentItem(item)
                    self.on_tree_item_clicked(item, 0)
                    break
            self._dirty = True

    def remove_task(self):
        """Remove the current task from the configuration"""
        if not self.current_task:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the task '{self.current_task}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.config_data[self.current_task]
            self.current_task = None
            self.current_prompt_set_index = None
            self.refresh_task_tree()
            self.clear_editor()
            self.update_ui_state(True)
            self._dirty = True

    def add_prompt_set(self):
        """Add a new prompt set to the current task"""
        if not self.current_task:
            return

        # Ensure the task has the 'prompts' key and it's a list
        if "prompts" not in self.config_data[self.current_task]:
            self.config_data[self.current_task]["prompts"] = []

        # Create a new prompt set with default values
        new_prompt_set = [
            "Enter your prompt here...",  # Prompt
            "Enter your system prompt here...",  # System prompt
            "0.25",  # Temperature
            "0.1",  # P-value
            ["default"],  # Models
        ]

        self.config_data[self.current_task]["prompts"].append(new_prompt_set)

        # Refresh the tree and select the new prompt set
        self.refresh_task_tree()
        self.current_prompt_set_index = (
            len(self.config_data[self.current_task]["prompts"]) - 1
        )

        # Find and select the new prompt set
        task_items = self.task_tree.findItems(
            self.current_task, Qt.MatchFlag.MatchExactly, 0
        )
        if task_items:
            task_item = task_items[0]
            if task_item.childCount() > 0:
                new_item = task_item.child(self.current_prompt_set_index)
                self.task_tree.setCurrentItem(new_item)
                self.on_tree_item_clicked(new_item, 0)
        self._dirty = True

    def remove_prompt_set(self):
        """Remove the current prompt set from the current task"""
        if not self.current_task or self.current_prompt_set_index is None:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete Prompt Set {self.current_prompt_set_index + 1}?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove the prompt set
            self.config_data[self.current_task]["prompts"].pop(
                self.current_prompt_set_index
            )

            # Reset the current prompt set index and refresh
            self.current_prompt_set_index = None
            self.refresh_task_tree()
            self.clear_editor()
            self.update_ui_state(True)
            self._dirty = True

    def clear_editor(self):
        """Clear the editor panel"""
        # Remove all widgets except the no_config_label
        for i in reversed(range(self.editor_layout.count())):
            widget = self.editor_layout.itemAt(i).widget()
            if widget and widget is not self.no_config_label:
                widget.setParent(None)  # This deletes the widget

        # Explicitly set attributes to None after widgets are removed
        self.fields_edit = None
        self.required_edit = None
        self.prompt_edit = None
        self.system_edit = None
        self.temp_spinbox = None
        self.p_value_spinbox = None
        self.models_list = None

        # Show the no config label if no config is loaded
        if not self.config_data:
            self.no_config_label.setVisible(True)

    def load_task_editor(self, task_name):
        """Load the editor for a task"""
        self.clear_editor()
        self.no_config_label.setVisible(False)

        task_data = self.config_data[task_name]

        # Create task editor form
        form_group = QGroupBox(f"Edit Task: {task_name}")
        form_layout = QFormLayout()

        # Fields
        fields_label = QLabel("Fields (comma separated):")
        self.fields_edit = QLineEdit()
        if "fields" in task_data:
            self.fields_edit.setText(", ".join(task_data["fields"]))
        form_layout.addRow(fields_label, self.fields_edit)

        # Required fields
        required_label = QLabel("Required fields (comma separated):")
        self.required_edit = QLineEdit()
        if "required" in task_data:
            self.required_edit.setText(", ".join(task_data["required"]))
        form_layout.addRow(required_label, self.required_edit)

        form_group.setLayout(form_layout)
        self.editor_layout.addWidget(form_group)

        # Add a save button
        save_btn = QPushButton("Apply Changes")
        save_btn.clicked.connect(self.save_task_data)
        self.editor_layout.addWidget(save_btn)

        # Add stretch to push everything to the top
        self.editor_layout.addStretch()

    def save_task_data(self):
        """Save the task data from the editor"""
        if not self.current_task:
            return

        # Get fields and required fields
        fields_text = self.fields_edit.text().strip()
        fields = [field.strip() for field in fields_text.split(",") if field.strip()]

        required_text = self.required_edit.text().strip()
        required = [
            field.strip() for field in required_text.split(",") if field.strip()
        ]

        # Update the config data
        self.config_data[self.current_task]["fields"] = fields
        self.config_data[self.current_task]["required"] = required

        self._dirty = True

    def load_prompt_set_editor(self, task_name, prompt_idx):
        """Load the editor for a prompt set"""
        self.clear_editor()
        self.no_config_label.setVisible(False)

        prompt_set = self.config_data[task_name]["prompts"][prompt_idx]

        # Create scroll area for the prompt editor
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)

        # Prompt
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setMinimumHeight(200)
        if len(prompt_set) > 0:
            self.prompt_edit.setPlainText(prompt_set[0])
        prompt_layout.addWidget(self.prompt_edit)
        prompt_group.setLayout(prompt_layout)
        editor_layout.addWidget(prompt_group)

        # System prompt
        system_group = QGroupBox("System Prompt")
        system_layout = QVBoxLayout()
        self.system_edit = QTextEdit()
        self.system_edit.setMinimumHeight(150)
        if len(prompt_set) > 1:
            self.system_edit.setPlainText(prompt_set[1])
        system_layout.addWidget(self.system_edit)
        system_group.setLayout(system_layout)
        editor_layout.addWidget(system_group)

        # Temperature and P-value
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()

        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.0, 2.0)
        self.temp_spinbox.setSingleStep(0.01)
        self.temp_spinbox.setDecimals(2)
        if len(prompt_set) > 2:
            try:
                self.temp_spinbox.setValue(float(prompt_set[2]))
            except (ValueError, TypeError):
                self.temp_spinbox.setValue(0.25)
        else:
            self.temp_spinbox.setValue(0.25)
        params_layout.addRow("Temperature:", self.temp_spinbox)

        self.p_value_spinbox = QDoubleSpinBox()
        self.p_value_spinbox.setRange(0.0, 1.0)
        self.p_value_spinbox.setSingleStep(0.01)
        self.p_value_spinbox.setDecimals(2)
        if len(prompt_set) > 3:
            try:
                self.p_value_spinbox.setValue(float(prompt_set[3]))
            except (ValueError, TypeError):
                self.p_value_spinbox.setValue(0.1)
        else:
            self.p_value_spinbox.setValue(0.1)
        params_layout.addRow("P-value:", self.p_value_spinbox)

        params_group.setLayout(params_layout)
        editor_layout.addWidget(params_group)

        # Models
        models_group = QGroupBox("Models")
        models_layout = QVBoxLayout()

        self.models_list = QListWidget()
        if len(prompt_set) > 4 and isinstance(prompt_set[4], list):
            for model in prompt_set[4]:
                self.models_list.addItem(model)
        models_layout.addWidget(self.models_list)

        models_buttons_layout = QHBoxLayout()
        self.add_model_btn = QPushButton("Add Model")
        self.add_model_btn.clicked.connect(self.add_model)
        models_buttons_layout.addWidget(self.add_model_btn)

        self.remove_model_btn = QPushButton("Remove Model")
        self.remove_model_btn.clicked.connect(self.remove_model)
        models_buttons_layout.addWidget(self.remove_model_btn)

        models_layout.addLayout(models_buttons_layout)
        models_group.setLayout(models_layout)
        editor_layout.addWidget(models_group)

        # Apply button
        save_btn = QPushButton("Apply Changes")
        save_btn.clicked.connect(self.save_prompt_set_data)
        editor_layout.addWidget(save_btn)

        # Add the editor widget to the scroll area
        self.editor_layout.addWidget(editor_widget)

    def add_model(self):
        """Add a model to the models list"""
        model_name, ok = QInputDialog.getText(
            self, "Add Model", "Enter the model name:"
        )

        if ok and model_name:
            self.models_list.addItem(model_name)

    def remove_model(self):
        """Remove the selected model from the models list"""
        current_item = self.models_list.currentItem()
        if current_item:
            row = self.models_list.row(current_item)
            self.models_list.takeItem(row)

    def save_prompt_set_data(self):
        """Save the prompt set data from the editor"""
        if not self.current_task or self.current_prompt_set_index is None:
            return

        # Collect data from the editor
        prompt = self.prompt_edit.toPlainText()
        system = self.system_edit.toPlainText()
        temp = str(self.temp_spinbox.value())
        p_value = str(self.p_value_spinbox.value())

        # Get models from the list widget
        models = []
        for i in range(self.models_list.count()):
            models.append(self.models_list.item(i).text())

        # Update the prompt set
        self.config_data[self.current_task]["prompts"][
            self.current_prompt_set_index
        ] = [prompt, system, temp, p_value, models]

        # Update the tree item text to reflect the models
        task_items = self.task_tree.findItems(
            self.current_task, Qt.MatchFlag.MatchExactly, 0
        )
        if task_items and task_items[0].childCount() > self.current_prompt_set_index:
            models_str = ", ".join(models) if models else "No models"
            task_items[0].child(self.current_prompt_set_index).setText(
                0, f"Prompt Set {self.current_prompt_set_index + 1} ({models_str})"
            )

        self._dirty = True

    def collect_current_editor_data(self):
        """Collect data from the current editor if it's open"""
        if not self.config_data:
            return

        # If editing a task and the widgets are still valid and visible
        if (
            hasattr(self, "fields_edit")
            and self.fields_edit is not None
            and self.fields_edit.parent() is not None
        ):
            self.save_task_data()

        # If editing a prompt set and the widgets are still valid and visible
        if (
            self.current_prompt_set_index is not None
            and hasattr(self, "prompt_edit")
            and self.prompt_edit is not None
            and self.prompt_edit.parent() is not None
        ):
            self.save_prompt_set_data()

    def closeEvent(self, event):
        """Handle close event, prompting to save if changes are unsaved."""
        if self._dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Save:
                self.save_config()
                if self._dirty:  # If save failed, don't close
                    event.ignore()
                else:
                    event.accept()
            elif reply == QMessageBox.StandardButton.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
