"""
Batch Processing Dialog - GUI for batch processing multiple sources
Claude Generated - Provides user-friendly interface for batch workflow
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QTabWidget, QLineEdit, QFileDialog, QListWidget, QListWidgetItem,
    QCheckBox, QGroupBox, QFormLayout, QTextEdit, QProgressBar,
    QMessageBox, QComboBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import os

from ..utils.batch_processor import BatchProcessor, BatchSource, BatchSourceResult, SourceType
from ..utils.pipeline_utils import PipelineStepExecutor
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..core.alima_manager import AlimaManager
from ..core.pipeline_manager import PipelineConfig
from .workers import StoppableWorker


class BatchProcessingWorker(StoppableWorker):
    """Background worker for batch processing - Claude Generated

    Extends StoppableWorker to support graceful batch processing interruption.
    """

    # Signals
    source_started = pyqtSignal(str, int, int)  # source_name, current, total
    source_completed = pyqtSignal(str, bool, str)  # source_name, success, message
    batch_completed = pyqtSignal(dict)  # summary dict
    progress_updated = pyqtSignal(int)  # progress percentage
    error_occurred = pyqtSignal(str)  # error message
    pipeline_token = pyqtSignal(str, str)  # token, step_id - Claude Generated

    def __init__(
        self,
        batch_processor: BatchProcessor,
        batch_file: Optional[str] = None,
        resume_state: Optional[str] = None,
        pipeline_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.batch_processor = batch_processor
        self.batch_file = batch_file
        self.resume_state = resume_state
        self.pipeline_config = pipeline_config
        self.logger = logging.getLogger(__name__)

    def cancel(self):
        """Cancel batch processing (backward compatible with previous implementation) - Claude Generated"""
        self.request_stop()

    def run(self):
        """Execute batch processing in background - Claude Generated"""
        try:
            # Setup callbacks
            def on_source_start(source, current, total):
                self.check_interruption()
                self.source_started.emit(source.source_value, current, total)
                self.progress_updated.emit(int((current - 1) / total * 100))

            def on_source_complete(result):
                self.check_interruption()
                self.source_completed.emit(
                    result.source.source_value,
                    result.success,
                    result.error_message or "Success"
                )

            def on_batch_complete(results):
                summary = self.batch_processor.get_batch_summary()
                self.batch_completed.emit(summary)
                self.progress_updated.emit(100)

            def on_pipeline_stream(token: str, step_id: str = ""):
                # Emit token to GUI via signal (thread-safe) - Claude Generated
                self.pipeline_token.emit(token, step_id or "")

            self.batch_processor.on_source_start = on_source_start
            self.batch_processor.on_source_complete = on_source_complete
            self.batch_processor.on_batch_complete = on_batch_complete
            self.batch_processor.stream_callback = on_pipeline_stream

            # Execute batch
            self.batch_processor.process_batch_file(
                batch_file=self.batch_file,
                pipeline_config=self.pipeline_config,
                resume_state=self.resume_state
            )

        except InterruptedError as e:
            self.logger.info("Batch processing interrupted by user")
            self.aborted.emit()
            self.error_occurred.emit(str(e))
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            self.error_occurred.emit(f"Batch processing error: {str(e)}")


class BatchProcessingDialog(QDialog):
    """Dialog for configuring and executing batch processing - Claude Generated"""

    def __init__(
        self,
        alima_manager: AlimaManager,
        cache_manager: UnifiedKnowledgeManager,
        config_manager,
        logger: Optional[logging.Logger] = None,
        pipeline_tab=None,
        parent=None
    ):
        super().__init__(parent)
        self.alima_manager = alima_manager
        self.cache_manager = cache_manager
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline_tab = pipeline_tab  # Optional PipelineTab for live display - Claude Generated

        self.worker: Optional[BatchProcessingWorker] = None
        self.batch_sources: List[BatchSource] = []

        self.setWindowTitle("Batch Processing - ALIMA")
        self.setMinimumSize(800, 600)
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface - Claude Generated"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("📦 Batch Processing")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Tab widget for input methods
        self.tabs = QTabWidget()

        # Tab 0: Batch File
        self.batch_file_tab = self._create_batch_file_tab()
        self.tabs.addTab(self.batch_file_tab, "📄 Batch File")

        # Tab 1: Direct File Selection - Claude Generated
        self.files_tab = self._create_files_tab()
        self.tabs.addTab(self.files_tab, "📎 Dateien")

        # Tab 2: Directory Scan
        self.directory_tab = self._create_directory_scan_tab()
        self.tabs.addTab(self.directory_tab, "📁 Directory Scan")

        layout.addWidget(self.tabs)

        # Output configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout()

        # Default to the central autosave dir from config - Claude Generated
        from ..utils.pipeline_defaults import get_autosave_dir
        _default_output = str(get_autosave_dir(self.config_manager))
        self.output_dir_input = QLineEdit(_default_output)
        output_dir_button = QPushButton("Browse...")
        output_dir_button.clicked.connect(self._browse_output_dir)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(output_dir_button)
        output_layout.addRow("Output Directory:", output_dir_layout)

        self.continue_on_error_checkbox = QCheckBox("Continue processing on error")
        self.continue_on_error_checkbox.setChecked(True)
        output_layout.addRow("Error Handling:", self.continue_on_error_checkbox)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Progress section
        self.progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_log = QTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setMaximumHeight(150)
        progress_layout.addWidget(self.progress_log)

        self.progress_group.setLayout(progress_layout)
        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.start_button = QPushButton("▶ Start Batch Processing")
        self.start_button.clicked.connect(self._start_batch_processing)
        self.start_button.setMinimumHeight(40)
        button_layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("⏹ Cancel")
        self.cancel_button.clicked.connect(self._cancel_batch_processing)
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.cancel_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_batch_file_tab(self) -> QWidget:
        """Create the batch file input tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Select a text file containing sources to process.\n"
            "Format: TYPE:VALUE (one per line)\n"
            "Supported types: DOI, PDF, TXT, IMG, URL"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # File selection
        file_layout = QHBoxLayout()
        self.batch_file_input = QLineEdit()
        self.batch_file_input.setPlaceholderText("Select batch file...")
        file_layout.addWidget(self.batch_file_input)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_batch_file)
        file_layout.addWidget(browse_button)

        layout.addLayout(file_layout)

        # Preview button
        preview_button = QPushButton("Preview Sources")
        preview_button.clicked.connect(self._preview_batch_file)
        layout.addWidget(preview_button)

        # Source preview list
        self.batch_preview_list = QListWidget()
        layout.addWidget(self.batch_preview_list)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_files_tab(self) -> QWidget:
        """Create the direct file selection tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Einzelne Dateien direkt auswählen und verarbeiten.\n"
            "Unterstützte Formate: TXT, PDF, PNG, JPG/JPEG"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Button row
        btn_layout = QHBoxLayout()
        add_files_btn = QPushButton("📂 Dateien auswählen...")
        add_files_btn.clicked.connect(self._add_files)
        btn_layout.addWidget(add_files_btn)

        remove_btn = QPushButton("Entfernen")
        remove_btn.clicked.connect(self._remove_selected_files)
        btn_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Liste leeren")
        clear_btn.clicked.connect(self._clear_files_list)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # File list with checkboxes
        self.files_list = QListWidget()
        self.files_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.files_list)

        widget.setLayout(layout)
        return widget

    def _add_files(self):
        """Open file dialog and add selected files to the list - Claude Generated"""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Dateien auswählen",
            "",
            "Unterstützte Dateien (*.txt *.pdf *.png *.jpg *.jpeg);;Alle Dateien (*)"
        )
        for filepath in filepaths:
            # Avoid duplicates
            existing = [self.files_list.item(i).data(Qt.ItemDataRole.UserRole)
                        for i in range(self.files_list.count())]
            if filepath in existing:
                continue
            item = QListWidgetItem(os.path.basename(filepath))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, filepath)
            item.setToolTip(filepath)
            self.files_list.addItem(item)

    def _remove_selected_files(self):
        """Remove selected items from the files list - Claude Generated"""
        for item in self.files_list.selectedItems():
            self.files_list.takeItem(self.files_list.row(item))

    def _clear_files_list(self):
        """Clear all items from the files list - Claude Generated"""
        self.files_list.clear()

    def _create_directory_scan_tab(self) -> QWidget:
        """Create the directory scan tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Directory selection
        dir_layout = QHBoxLayout()
        self.scan_dir_input = QLineEdit()
        self.scan_dir_input.setPlaceholderText("Select directory to scan...")
        dir_layout.addWidget(self.scan_dir_input)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_scan_directory)
        dir_layout.addWidget(browse_button)

        layout.addLayout(dir_layout)

        # Filter configuration
        filter_group = QGroupBox("File Filters")
        filter_layout = QVBoxLayout()

        # File type filter
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("File Types:"))
        self.file_types_input = QLineEdit("*.pdf; *.txt; *.png; *.jpg")
        type_layout.addWidget(self.file_types_input)
        filter_layout.addLayout(type_layout)

        # Recursive checkbox
        self.recursive_checkbox = QCheckBox("Include subdirectories")
        self.recursive_checkbox.setChecked(True)
        filter_layout.addWidget(self.recursive_checkbox)

        # Name pattern
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Name Pattern:"))
        self.name_pattern_input = QLineEdit()
        self.name_pattern_input.setPlaceholderText("Optional (e.g., report_*.pdf)")
        pattern_layout.addWidget(self.name_pattern_input)
        filter_layout.addLayout(pattern_layout)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Scan button
        scan_button = QPushButton("Scan Directory")
        scan_button.clicked.connect(self._scan_directory)
        layout.addWidget(scan_button)

        # Preview list
        preview_label = QLabel("Found Files (uncheck to exclude):")
        layout.addWidget(preview_label)

        self.scan_preview_list = QListWidget()
        self.scan_preview_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self.scan_preview_list)

        widget.setLayout(layout)
        return widget

    def _browse_batch_file(self):
        """Browse for batch file - Claude Generated"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Batch File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if filepath:
            self.batch_file_input.setText(filepath)

    def _browse_scan_directory(self):
        """Browse for directory to scan - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Scan"
        )
        if directory:
            self.scan_dir_input.setText(directory)

    def _browse_output_dir(self):
        """Browse for output directory - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        if directory:
            self.output_dir_input.setText(directory)

    def _preview_batch_file(self):
        """Preview sources from batch file - Claude Generated"""
        filepath = self.batch_file_input.text().strip()
        if not filepath:
            QMessageBox.warning(self, "No File", "Please select a batch file first.")
            return

        if not os.path.exists(filepath):
            QMessageBox.warning(self, "File Not Found", f"File not found: {filepath}")
            return

        try:
            from ..utils.batch_processor import BatchSourceParser
            parser = BatchSourceParser(self.logger)
            sources = parser.parse_batch_file(filepath)

            self.batch_preview_list.clear()
            if not sources:
                # Informative hint when no valid entries found - Claude Generated
                hint = QListWidgetItem(
                    "Keine Quellen gefunden. Erwartet: Einträge im Format TYP:WERT\n"
                    "Beispiele:\n"
                    "  TXT:/pfad/abstract.txt\n"
                    "  PDF:/pfad/paper.pdf\n"
                    "  DOI:10.1234/example\n"
                    "  IMG:/pfad/bild.png\n"
                    "  URL:https://example.com\n"
                    "\nHinweis: Bare DOIs (10.XXXX/...) und doi.org-URLs werden automatisch erkannt."
                )
                hint.setForeground(Qt.GlobalColor.darkRed)
                self.batch_preview_list.addItem(hint)
            else:
                for source in sources:
                    item_text = f"{source.source_type.value}: {source.source_value}"
                    if source.custom_name:
                        item_text += f" ({source.custom_name})"
                    self.batch_preview_list.addItem(item_text)

            self.batch_sources = sources
            self.logger.info(f"Previewed {len(sources)} sources from batch file")

        except Exception as e:
            QMessageBox.critical(self, "Parse Error", f"Failed to parse batch file:\n{str(e)}")

    def _scan_directory(self):
        """Scan directory for files - Claude Generated"""
        directory = self.scan_dir_input.text().strip()
        if not directory:
            QMessageBox.warning(self, "No Directory", "Please select a directory first.")
            return

        if not os.path.exists(directory):
            QMessageBox.warning(self, "Directory Not Found", f"Directory not found: {directory}")
            return

        try:
            # Parse file type patterns
            patterns = [p.strip() for p in self.file_types_input.text().split(';')]
            recursive = self.recursive_checkbox.isChecked()
            name_pattern = self.name_pattern_input.text().strip()

            # Scan directory
            found_files = []
            path = Path(directory)

            for pattern in patterns:
                pattern = pattern.replace('*', '')  # Remove wildcards
                if recursive:
                    found_files.extend(path.rglob(f"*{pattern}"))
                else:
                    found_files.extend(path.glob(f"*{pattern}"))

            # Filter by name pattern if specified
            if name_pattern:
                import fnmatch
                found_files = [f for f in found_files if fnmatch.fnmatch(f.name, name_pattern)]

            # Populate preview list with checkboxes
            self.scan_preview_list.clear()
            for filepath in found_files:
                item = QListWidgetItem(str(filepath))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(Qt.ItemDataRole.UserRole, str(filepath))
                self.scan_preview_list.addItem(item)

            self.logger.info(f"Found {len(found_files)} files in directory")

        except Exception as e:
            QMessageBox.critical(self, "Scan Error", f"Failed to scan directory:\n{str(e)}")

    def _start_batch_processing(self):
        """Start batch processing - Claude Generated"""
        # Validate configuration
        output_dir = self.output_dir_input.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please specify an output directory.")
            return

        # Get sources based on active tab - Claude Generated
        tab_index = self.tabs.currentIndex()
        if tab_index == 0:  # Batch File tab
            batch_file = self.batch_file_input.text().strip()
            if not batch_file:
                QMessageBox.warning(self, "No Batch File", "Please select a batch file.")
                return
            sources_input = ("file", batch_file)
        elif tab_index == 1:  # Dateien tab (direct file selection)
            checked_files = []
            for i in range(self.files_list.count()):
                item = self.files_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    filepath = item.data(Qt.ItemDataRole.UserRole)
                    checked_files.append(filepath)

            if not checked_files:
                QMessageBox.warning(self, "Keine Dateien", "Bitte mindestens eine Datei auswählen.")
                return

            sources_input = ("files", checked_files)
        else:  # Directory Scan tab (index 2)
            # Collect checked files
            checked_files = []
            for i in range(self.scan_preview_list.count()):
                item = self.scan_preview_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    filepath = item.data(Qt.ItemDataRole.UserRole)
                    checked_files.append(filepath)

            if not checked_files:
                QMessageBox.warning(self, "No Files Selected", "Please select at least one file.")
                return

            sources_input = ("files", checked_files)

        # Create output directory if needed
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create PipelineStepExecutor
        executor = PipelineStepExecutor(
            alima_manager=self.alima_manager,
            cache_manager=self.cache_manager,
            logger=self.logger,
            config_manager=self.config_manager
        )

        # Create BatchProcessor
        batch_processor = BatchProcessor(
            pipeline_executor=executor,
            cache_manager=self.cache_manager,
            output_dir=output_dir,
            logger=self.logger,
            continue_on_error=self.continue_on_error_checkbox.isChecked()
        )

        # Get pipeline configuration
        try:
            pipeline_config = PipelineConfig.create_from_provider_preferences(self.config_manager)
        except:
            pipeline_config = PipelineConfig()

        # Convert to dict
        pipeline_config_dict = {
            "step_configs": {
                step_id: {
                    "provider": step_config.provider,
                    "model": step_config.model,
                    "enabled": step_config.enabled,
                }
                for step_id, step_config in pipeline_config.step_configs.items()
            }
        }

        # Handle different source inputs
        if sources_input[0] == "file":
            batch_file = sources_input[1]
            resume_state = None
        else:  # files list
            # Create temporary batch file
            import tempfile
            temp_batch = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            for filepath in sources_input[1]:
                # Determine source type
                ext = Path(filepath).suffix.lower()
                if ext == '.pdf':
                    source_type = 'PDF'
                elif ext in ['.txt', '.text']:
                    source_type = 'TXT'
                elif ext in ['.png', '.jpg', '.jpeg']:
                    source_type = 'IMG'
                else:
                    continue
                temp_batch.write(f"{source_type}:{filepath}\n")
            temp_batch.close()
            batch_file = temp_batch.name
            resume_state = None

        # Create worker thread
        self.worker = BatchProcessingWorker(
            batch_processor=batch_processor,
            batch_file=batch_file,
            resume_state=resume_state,
            pipeline_config=pipeline_config_dict
        )

        # Connect signals
        self.worker.source_started.connect(self._on_source_started)
        self.worker.source_completed.connect(self._on_source_completed)
        self.worker.batch_completed.connect(self._on_batch_completed)
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.error_occurred.connect(self._on_error_occurred)
        self.worker.pipeline_token.connect(self._on_pipeline_token)

        # Route tokens to PipelineStreamWidget if available - Claude Generated
        if self.pipeline_tab and hasattr(self.pipeline_tab, 'on_llm_stream_token'):
            self.worker.pipeline_token.connect(self.pipeline_tab.on_llm_stream_token)
            self.worker.source_started.connect(self._notify_pipeline_tab_source_started)
            self.worker.source_completed.connect(self._notify_pipeline_tab_source_completed)
            self.worker.batch_completed.connect(self._notify_pipeline_tab_batch_completed)

        # Update UI
        self.progress_group.setVisible(True)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.tabs.setEnabled(False)
        self.output_dir_input.setEnabled(False)

        # Start worker
        self.worker.start()
        self.logger.info("Batch processing started")

        # Switch main window to pipeline tab so output is visible - Claude Generated
        if self.pipeline_tab:
            main_win = self.parent()
            if main_win and hasattr(main_win, 'tabs'):
                main_win.tabs.setCurrentWidget(self.pipeline_tab)

    def _cancel_batch_processing(self):
        """Cancel batch processing - Claude Generated"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Cancel Batch",
                "Are you sure you want to cancel batch processing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.cancel()
                self.progress_log.append("\n⚠️  Cancelling batch processing...")

    @pyqtSlot(str, int, int)
    def _on_source_started(self, source_name: str, current: int, total: int):
        """Handle source started signal - Claude Generated"""
        self.progress_log.append(f"\n▶ [{current}/{total}] Processing: {source_name}")

    @pyqtSlot(str, bool, str)
    def _on_source_completed(self, source_name: str, success: bool, message: str):
        """Handle source completed signal - Claude Generated"""
        if success:
            self.progress_log.append(f"  ✅ Completed: {source_name}")
        else:
            self.progress_log.append(f"  ❌ Failed: {source_name} - {message}")

    @pyqtSlot(dict)
    def _on_batch_completed(self, summary: Dict[str, Any]):
        """Handle batch completed signal - Claude Generated"""
        self.progress_bar.setValue(100)
        self.progress_log.append(f"\n{'='*60}")
        self.progress_log.append("🎉 Batch Processing Complete!")
        self.progress_log.append(f"{'='*60}")
        self.progress_log.append(f"Total: {summary['total_sources']}")
        self.progress_log.append(f"Successful: {summary['successful']}")
        self.progress_log.append(f"Failed: {summary['failed']}")
        self.progress_log.append(f"Success Rate: {summary['success_rate']:.1f}%")
        self.progress_log.append(f"Output: {summary['output_dir']}")

        # Re-enable UI
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.tabs.setEnabled(True)
        self.output_dir_input.setEnabled(True)

        QMessageBox.information(
            self,
            "Batch Complete",
            f"Batch processing completed!\n\n"
            f"Successful: {summary['successful']}\n"
            f"Failed: {summary['failed']}\n"
            f"Results saved to: {summary['output_dir']}"
        )

    @pyqtSlot(int)
    def _on_progress_updated(self, progress: int):
        """Handle progress update signal - Claude Generated"""
        self.progress_bar.setValue(progress)

    @pyqtSlot(str, str)
    def _on_pipeline_token(self, token: str, step_id: str):
        """Append LLM streaming tokens to progress log - Claude Generated"""
        cursor = self.progress_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.progress_log.setTextCursor(cursor)
        self.progress_log.ensureCursorVisible()

    @pyqtSlot(str, int, int)
    def _notify_pipeline_tab_source_started(self, source_name: str, current: int, total: int):
        """Forward source-start event to PipelineStreamWidget - Claude Generated"""
        if not self.pipeline_tab:
            return
        sw = getattr(self.pipeline_tab, 'stream_widget', None)
        if sw is None:
            return
        sw.reset_for_new_pipeline()
        sw.on_pipeline_started(f"batch_{current}")
        sw.add_pipeline_message(f"[{current}/{total}] Batch: {source_name}", "info")
        sw.start_llm_streaming("initialisation")

    @pyqtSlot(str, bool, str)
    def _notify_pipeline_tab_source_completed(self, source_name: str, success: bool, message: str):
        """Forward source-complete event to PipelineStreamWidget - Claude Generated"""
        if not self.pipeline_tab:
            return
        sw = getattr(self.pipeline_tab, 'stream_widget', None)
        if sw is None:
            return
        if sw.is_streaming:
            sw.end_llm_streaming()
        status = "✅ Abgeschlossen" if success else f"❌ Fehler: {message}"
        sw.add_pipeline_message(status, "success" if success else "error")

    @pyqtSlot(dict)
    def _notify_pipeline_tab_batch_completed(self, summary: dict):
        """Forward batch-complete event to PipelineStreamWidget - Claude Generated"""
        if not self.pipeline_tab:
            return
        sw = getattr(self.pipeline_tab, 'stream_widget', None)
        if sw is None:
            return
        sw.add_pipeline_message(
            f"Batch abgeschlossen — {summary.get('successful', 0)}/{summary.get('total_sources', 0)} erfolgreich",
            "success"
        )

    @pyqtSlot(str)
    def _on_error_occurred(self, error_message: str):
        """Handle error signal - Claude Generated"""
        self.progress_log.append(f"\n❌ Error: {error_message}")

        # Re-enable UI
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.tabs.setEnabled(True)
        self.output_dir_input.setEnabled(True)

        QMessageBox.critical(self, "Batch Error", error_message)
