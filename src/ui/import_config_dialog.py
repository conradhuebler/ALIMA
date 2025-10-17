#!/usr/bin/env python3
"""
ALIMA Configuration Import Dialog - Claude Generated
Provides a graphical interface for importing ALIMA configurations from directories.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QCheckBox,
    QTextEdit,
    QProgressBar,
    QMessageBox,
    QWidget,
    QFormLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont


class ImportConfigWorker(QThread):
    """Worker thread for configuration import - Claude Generated"""

    progress_updated = pyqtSignal(str)  # Progress message
    import_completed = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, source_dir: str, create_backup: bool, config_manager):
        super().__init__()
        self.source_dir = source_dir
        self.create_backup = create_backup
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Execute configuration import in background thread - Claude Generated"""
        try:
            self.progress_updated.emit("🔍 Validiere Quellverzeichnis...")

            # Call ConfigManager.import_configuration()
            success, message = self.config_manager.import_configuration(
                self.source_dir,
                create_backup=self.create_backup
            )

            self.import_completed.emit(success, message)

        except Exception as e:
            error_msg = f"Import-Fehler: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.import_completed.emit(False, error_msg)


class ImportConfigDialog(QDialog):
    """Configuration Import Dialog with multi-step workflow - Claude Generated"""

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.source_dir = None
        self.import_worker = None

        self.setWindowTitle("ALIMA Konfiguration importieren")
        self.setModal(True)
        self.resize(700, 600)

        self.init_ui()

    def init_ui(self):
        """Initialize UI components - Claude Generated"""
        main_layout = QVBoxLayout(self)

        # === TAB WIDGET ===
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Quellauswahl
        self.create_source_selection_tab()

        # Tab 2: Vorschau
        self.create_preview_tab()

        # Tab 3: Import-Fortschritt
        self.create_progress_tab()

        # === BUTTONS ===
        button_layout = QHBoxLayout()

        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        button_layout.addStretch()

        self.back_button = QPushButton("◀ Zurück")
        self.back_button.clicked.connect(self.go_to_previous_tab)
        self.back_button.setEnabled(False)
        button_layout.addWidget(self.back_button)

        self.next_button = QPushButton("Weiter ▶")
        self.next_button.clicked.connect(self.go_to_next_tab)
        button_layout.addWidget(self.next_button)

        main_layout.addLayout(button_layout)

    def create_source_selection_tab(self):
        """Create tab for source directory selection - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Info label
        info_label = QLabel(
            "Wählen Sie das Verzeichnis aus, das Ihre ALIMA Konfigurationsdateien enthält.\n"
            "Erforderlich: config.json\n"
            "Optional: prompts.json, *.db Dateien"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Directory selection
        select_layout = QHBoxLayout()
        self.source_path_label = QLineEdit()
        self.source_path_label.setReadOnly(True)
        self.source_path_label.setPlaceholderText("Kein Verzeichnis ausgewählt...")
        select_layout.addWidget(self.source_path_label)

        browse_button = QPushButton("📁 Durchsuchen...")
        browse_button.clicked.connect(self.browse_source_directory)
        select_layout.addWidget(browse_button)

        layout.addLayout(select_layout)

        # File status
        layout.addSpacing(20)
        layout.addWidget(QLabel("📋 Datei-Status:"))

        self.file_status_widget = QTextEdit()
        self.file_status_widget.setReadOnly(True)
        self.file_status_widget.setMinimumHeight(200)
        layout.addWidget(self.file_status_widget)

        layout.addStretch()

        self.tabs.addTab(tab, "1️⃣  Quellauswahl")

    def create_preview_tab(self):
        """Create tab for import options and preview - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Backup option
        self.backup_checkbox = QCheckBox(
            "☑ Backup der aktuellen Konfiguration erstellen"
        )
        self.backup_checkbox.setChecked(True)
        layout.addWidget(self.backup_checkbox)

        # Target directory info
        layout.addSpacing(20)
        target_label = QLabel("📂 Zielverzeichnis:")
        layout.addWidget(target_label)

        config_dir = str(self.config_manager.config_file.parent)
        self.target_dir_label = QLineEdit()
        self.target_dir_label.setText(config_dir)
        self.target_dir_label.setReadOnly(True)
        layout.addWidget(self.target_dir_label)

        # Preview of files to import
        layout.addSpacing(20)
        preview_label = QLabel("📋 Dateien zum Importieren:")
        layout.addWidget(preview_label)

        self.preview_widget = QTextEdit()
        self.preview_widget.setReadOnly(True)
        self.preview_widget.setMinimumHeight(250)
        layout.addWidget(self.preview_widget)

        layout.addStretch()

        self.tabs.addTab(tab, "2️⃣  Vorschau & Optionen")

    def create_progress_tab(self):
        """Create tab for import progress - Claude Generated"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        status_label = QLabel("Import-Status:")
        font = QFont()
        font.setBold(True)
        status_label.setFont(font)
        layout.addWidget(status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setFont(QFont("Consolas", 9))
        self.progress_text.setMinimumHeight(300)
        layout.addWidget(self.progress_text)

        layout.addStretch()

        self.tabs.addTab(tab, "3️⃣  Import-Fortschritt")

    @pyqtSlot()
    def browse_source_directory(self):
        """Handle directory selection - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "ALIMA Konfigurationsverzeichnis wählen",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self.source_dir = directory
            self.source_path_label.setText(directory)
            self.validate_source_directory()

    def validate_source_directory(self):
        """Validate source directory and show file status - Claude Generated"""
        if not self.source_dir:
            return

        try:
            from ..utils.config_validator import ConfigValidator

            validator = ConfigValidator(self.logger)
            info = validator.get_config_directory_info(self.source_dir)

            # Build status text
            status_lines = []

            status_lines.append("✅ config.json" if info.has_config else "❌ config.json (FEHLT)")
            if info.has_config:
                status_lines.append(f"   └─ {info.config_size / 1024:.1f} KB")
                if info.config_valid:
                    status_lines.append(f"   └─ Version: {info.config_version}")

            status_lines.append(
                "✅ prompts.json" if info.has_prompts else "⚠️  prompts.json (optional)"
            )
            if info.has_prompts:
                status_lines.append(f"   └─ {info.prompts_size / 1024:.1f} KB")

            status_lines.append(
                "✅ alima_knowledge.db" if info.has_database else "⚠️  Database (optional)"
            )
            if info.has_database:
                size_mb = info.database_size / (1024 * 1024)
                status_lines.append(f"   └─ {size_mb:.1f} MB")
                status_lines.append(f"   └─ Tabellen: {info.database_tables}")

            if info.errors:
                status_lines.append("\n❌ Fehler:")
                for error in info.errors:
                    status_lines.append(f"   • {error}")

            self.file_status_widget.setText("\n".join(status_lines))

            # Update preview when moving to next tab
            self.update_preview()

            # Enable next button if config.json exists
            self.next_button.setEnabled(info.has_config)

        except Exception as e:
            error_msg = f"Validierungsfehler: {str(e)}"
            self.file_status_widget.setText(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            self.next_button.setEnabled(False)

    def update_preview(self):
        """Update preview tab with file information - Claude Generated"""
        if not self.source_dir:
            return

        try:
            from ..utils.config_validator import ConfigValidator

            validator = ConfigValidator(self.logger)
            info = validator.get_config_directory_info(self.source_dir)

            preview_lines = ["📋 Folgende Dateien werden kopiert:\n"]

            if info.has_config:
                size_kb = info.config_size / 1024
                preview_lines.append(f"✅ config.json ({size_kb:.1f} KB)")

            if info.has_prompts:
                size_kb = info.prompts_size / 1024
                preview_lines.append(f"✅ prompts.json ({size_kb:.1f} KB)")

            if info.has_database:
                size_mb = info.database_size / (1024 * 1024)
                preview_lines.append(f"✅ alima_knowledge.db ({size_mb:.1f} MB)")

            preview_lines.append(f"\n📂 Zielverzeichnis:")
            preview_lines.append(str(self.config_manager.config_file.parent))

            if self.backup_checkbox.isChecked():
                preview_lines.append(f"\n📁 Backup wird erstellt:")
                preview_lines.append("   ✅ config_backup_<timestamp>.json")
                if info.has_prompts:
                    preview_lines.append("   ✅ prompts_backup_<timestamp>.json")
                if info.has_database:
                    preview_lines.append("   ✅ alima_knowledge_backup_<timestamp>.db")

            self.preview_widget.setText("\n".join(preview_lines))

        except Exception as e:
            self.preview_widget.setText(f"❌ Fehler bei der Vorschau: {str(e)}")

    @pyqtSlot()
    def go_to_next_tab(self):
        """Move to next tab or start import - Claude Generated"""
        current_tab = self.tabs.currentIndex()

        if current_tab == 0:
            # Tab 1 -> Tab 2: Validate source directory
            if not self.source_dir:
                QMessageBox.warning(
                    self,
                    "Verzeichnis erforderlich",
                    "Bitte wählen Sie zunächst ein Verzeichnis aus."
                )
                return

            self.tabs.setCurrentIndex(1)
            self.back_button.setEnabled(True)
            self.next_button.setText("Importieren ✓")

        elif current_tab == 1:
            # Tab 2 -> Tab 3: Start import
            self.tabs.setCurrentIndex(2)
            self.back_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.start_import()

    @pyqtSlot()
    def go_to_previous_tab(self):
        """Move to previous tab - Claude Generated"""
        current_tab = self.tabs.currentIndex()

        if current_tab == 1:
            self.tabs.setCurrentIndex(0)
            self.back_button.setEnabled(False)
            self.next_button.setText("Weiter ▶")

    def start_import(self):
        """Start the configuration import process - Claude Generated"""
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress

            # Create and start worker
            self.import_worker = ImportConfigWorker(
                self.source_dir,
                self.backup_checkbox.isChecked(),
                self.config_manager
            )

            self.import_worker.progress_updated.connect(self.on_progress_message)
            self.import_worker.import_completed.connect(self.on_import_completed)

            self.import_worker.start()

        except Exception as e:
            error_msg = f"Import konnte nicht gestartet werden: {str(e)}"
            self.progress_text.append(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            self.next_button.setEnabled(True)
            self.cancel_button.setEnabled(True)

    @pyqtSlot(str)
    def on_progress_message(self, message: str):
        """Handle progress message from worker - Claude Generated"""
        self.progress_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.progress_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(bool, str)
    def on_import_completed(self, success: bool, message: str):
        """Handle import completion - Claude Generated"""
        self.progress_bar.setVisible(False)

        if success:
            self.progress_text.append("")
            self.progress_text.append("=" * 50)
            self.progress_text.append("✅ IMPORT ERFOLGREICH!")
            self.progress_text.append("=" * 50)

            self.cancel_button.setText("Schließen")
            self.cancel_button.setEnabled(True)
            self.next_button.setEnabled(False)

            # Auto-accept after 2 seconds
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(2000, self.accept)

        else:
            self.progress_text.append("")
            self.progress_text.append("=" * 50)
            self.progress_text.append(f"❌ IMPORT FEHLGESCHLAGEN")
            self.progress_text.append(message)
            self.progress_text.append("=" * 50)

            self.cancel_button.setEnabled(True)
            self.next_button.setEnabled(True)

            QMessageBox.critical(self, "Import-Fehler", message)
