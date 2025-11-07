"""
Pipeline Stream Widget - Live feedback for pipeline execution
Claude Generated - Kontinuierliches Textfenster mit Streaming-Updates
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QGroupBox,
    QProgressBar,
    QSplitter,
    QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QThread
from PyQt6.QtGui import QFont, QTextCursor, QColor, QPalette
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import json

from ..core.pipeline_manager import PipelineStep


class PipelineStreamWidget(QWidget):
    """Kontinuierliches Streaming-Widget für Pipeline-Feedback - Claude Generated"""

    # Signals for user interaction
    cancel_pipeline = pyqtSignal()
    pause_pipeline = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_step_id: Optional[str] = None
        self.is_streaming: bool = False
        self.step_start_times: Dict[str, datetime] = {}

        self.setup_ui()

    def setup_ui(self):
        """Setup der Stream UI - Claude Generated"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Main streaming area
        self.create_streaming_area(layout)

    def create_streaming_area(self, layout):
        """Create main streaming text area - Claude Generated"""
        stream_group = QGroupBox("📝 Pipeline-Aktivität (Live)")
        stream_layout = QVBoxLayout(stream_group)

        # Main streaming text widget
        self.stream_text = QTextEdit()
        self.stream_text.setReadOnly(True)
        self.stream_text.setMinimumHeight(300)

        # Enhanced styling for readability
        font = QFont("Consolas", 10)  # Monospace font for structured output
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.stream_text.setFont(font)

        self.stream_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;
                color: #f8f8f2;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            QScrollBar:vertical {
                background: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
        """
        )

        stream_layout.addWidget(self.stream_text)

        # Stream controls
        controls_layout = QHBoxLayout()

        self.auto_scroll_checkbox = QCheckBox("🔄 Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_scroll_checkbox)

        controls_layout.addStretch()

        self.clear_button = QPushButton("🗑️ Leeren")
        self.clear_button.clicked.connect(self.clear_stream)
        controls_layout.addWidget(self.clear_button)

        self.save_log_button = QPushButton("💾 Log speichern")
        self.save_log_button.clicked.connect(self.save_stream_log)
        controls_layout.addWidget(self.save_log_button)

        stream_layout.addLayout(controls_layout)
        layout.addWidget(stream_group)

    def add_pipeline_message(
        self, message: str, level: str = "info", step_id: Optional[str] = None
    ):
        """Add message to pipeline stream - Claude Generated"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color coding based on message level
        color_map = {
            "info": "#f8f8f2",  # White
            "success": "#50fa7b",  # Green
            "warning": "#f1fa8c",  # Yellow
            "error": "#ff5555",  # Red
            "step": "#8be9fd",  # Cyan
            "stream": "#bd93f9",  # Purple
        }

        color = color_map.get(level, "#f8f8f2")

        # Format message with step context
        if step_id:
            formatted_message = f"<span style='color: #6272a4;'>[{timestamp}]</span> <span style='color: {color}; font-weight: bold;'>[{step_id.upper()}]</span> <span style='color: {color};'>{message}</span>"
        else:
            formatted_message = f"<span style='color: #6272a4;'>[{timestamp}]</span> <span style='color: {color};'>{message}</span>"

        # Add to stream
        self.stream_text.append(formatted_message)

        # Auto-scroll if enabled
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def add_streaming_token(self, token: str, step_id: str):
        """Add streaming token to current line - Claude Generated"""
        # Get current cursor position
        cursor = self.stream_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Insert token with streaming color - preserve whitespace
        from html import escape

        escaped_token = escape(token).replace(" ", "&nbsp;").replace("\n", "<br>")
        cursor.insertHtml(f"<span style='color: #bd93f9;'>{escaped_token}</span>")

        # Auto-scroll if enabled
        if self.auto_scroll_checkbox.isChecked():
            self.auto_scroll_to_bottom()

    def start_streaming_line(self, step_id: str, prefix: str = ""):
        """Start a new streaming line - Claude Generated"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_prefix = f"<span style='color: #6272a4;'>[{timestamp}]</span> <span style='color: #8be9fd; font-weight: bold;'>[{step_id.upper()}]</span> <span style='color: #bd93f9;'>{prefix}"

        self.stream_text.append(formatted_prefix)
        self.is_streaming = True

    def end_streaming_line(self):
        """End current streaming line - Claude Generated"""
        if self.is_streaming:
            cursor = self.stream_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml("</span>")
            self.is_streaming = False

    def auto_scroll_to_bottom(self):
        """Auto-scroll to bottom of text area - Claude Generated"""
        scrollbar = self.stream_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(object)
    def on_pipeline_started(self, pipeline_id: str):
        """Handle pipeline start - Claude Generated"""
        self.add_pipeline_message("🚀 Pipeline gestartet", "step")
        self.add_pipeline_message(f"Pipeline ID: {pipeline_id}", "info")

        self.pipeline_start_time = datetime.now()

    @pyqtSlot(object)
    def on_step_started(self, step: PipelineStep):
        """Handle step start - Claude Generated"""
        self.current_step_id = step.step_id
        self.step_start_times[step.step_id] = datetime.now()

        # Add to stream
        self.add_pipeline_message(
            f"▶ Starte Schritt: {step.name}", "step", step.step_id
        )
        if step.provider and step.model:
            self.add_pipeline_message(
                f"Verwende {step.provider}/{step.model}", "info", step.step_id
            )

    @pyqtSlot(object)
    def on_step_completed(self, step: PipelineStep):
        """Handle step completion - Claude Generated"""
        duration = "unbekannt"
        if step.step_id in self.step_start_times:
            duration_seconds = (
                datetime.now() - self.step_start_times[step.step_id]
            ).total_seconds()
            duration = f"{duration_seconds:.1f}s"

        self.add_pipeline_message(
            f"✅ Schritt abgeschlossen in {duration}", "success", step.step_id
        )

        # Show results if available
        if step.output_data:
            if step.step_id == "keywords" and "keywords" in step.output_data:
                keywords = step.output_data["keywords"]
                self.add_pipeline_message(
                    f"Gefunden: {len(keywords)} Keywords", "info", step.step_id
                )
                self.add_pipeline_message(
                    f"Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}",
                    "info",
                    step.step_id,
                )

            elif step.step_id == "search" and "search_results" in step.output_data:
                count = step.output_data["search_results"]
                self.add_pipeline_message(
                    f"Gefunden: {count} GND-Einträge", "info", step.step_id
                )

            elif (
                step.step_id == "verification"
                and "verified_keywords" in step.output_data
            ):
                verified = step.output_data["verified_keywords"]
                self.add_pipeline_message(
                    f"Verifiziert: {len(verified)} Keywords", "info", step.step_id
                )

            elif step.step_id == "dk_search" and "dk_search_results" in step.output_data:
                # Display DK search results with sample titles - Claude Generated
                dk_results = step.output_data["dk_search_results"]
                self._display_dk_search_results(dk_results, step.step_id)

    def _display_dk_search_results(self, dk_results: List[Dict[str, Any]], step_id: str):
        """Display DK search results with sample titles - Claude Generated

        Args:
            dk_results: List of DK search result dictionaries, each containing:
                - "dk": DK classification code
                - "titles": List of catalog titles with this classification
                - "count": Total count of titles
            step_id: Pipeline step ID for message formatting
        """
        if not dk_results:
            self.add_pipeline_message("Keine DK-Klassifikationen gefunden", "info", step_id)
            return

        # Group results by DK code and count total titles
        total_titles = 0
        dk_summary = []

        for result in dk_results:
            if isinstance(result, dict):
                dk_code = result.get("dk", "unknown")
                titles = result.get("titles", [])
                total_count = result.get("count", len(titles))

                if titles:
                    total_titles += total_count
                    # Get first 3 titles as sample
                    sample_titles = titles[:3]
                    dk_summary.append((dk_code, total_count, sample_titles))

        # Display summary
        self.add_pipeline_message(
            f"🔍 DK-Suche: {len(dk_summary)} Klassifikationen gefunden ({total_titles} Titel)",
            "info",
            step_id,
        )

        # Display each DK code with sample titles - Claude Generated (Fixed None-handling)
        for dk_code, total_count, sample_titles in dk_summary:
            # Filter None values and convert to strings - Claude Generated
            filtered_titles = [str(t).strip() for t in sample_titles if t is not None and str(t).strip()]

            if not filtered_titles:
                # No valid titles available
                titles_display = "(keine Titel verfügbar)"
            else:
                titles_preview = " | ".join(filtered_titles)
                # Add ellipsis if there are more titles than shown in preview
                if total_count > len(filtered_titles):
                    titles_display = f"{titles_preview} ... (+{total_count - len(filtered_titles)} weitere)"
                else:
                    titles_display = titles_preview

            # Truncate long title preview to fit in one line
            if len(titles_display) > 100:
                titles_display = titles_display[:97] + "..."

            self.add_pipeline_message(
                f"  📊 DK {dk_code} ({total_count} Titel): {titles_display}",
                "info",
                step_id,
            )

    @pyqtSlot(object, str)
    def on_step_error(self, step: PipelineStep, error_message: str):
        """Handle step error - Claude Generated"""
        self.add_pipeline_message(
            f"❌ Fehler in Schritt: {step.name}", "error", step.step_id
        )
        self.add_pipeline_message(
            f"Fehlermeldung: {error_message}", "error", step.step_id
        )

    @pyqtSlot(object)
    def on_pipeline_completed(self, analysis_state):
        """Handle pipeline completion - Claude Generated"""
        total_duration = "unbekannt"
        if hasattr(self, "pipeline_start_time"):
            total_seconds = (datetime.now() - self.pipeline_start_time).total_seconds()
            total_duration = f"{total_seconds:.1f}s"

        self.add_pipeline_message(
            f"🎉 Pipeline vollständig abgeschlossen in {total_duration}!", "success"
        )

    @pyqtSlot(str)
    def on_llm_token_received(self, token: str):
        """Handle streaming LLM token - Claude Generated"""
        if self.current_step_id:
            self.add_streaming_token(token, self.current_step_id)

    def start_llm_streaming(self, step_id: str):
        """Start LLM streaming for a step - Claude Generated"""
        self.start_streaming_line(step_id, "LLM Antwort: ")

    def end_llm_streaming(self):
        """End LLM streaming - Claude Generated"""
        self.end_streaming_line()

    def clear_stream(self):
        """Clear stream content - Claude Generated"""
        self.stream_text.clear()
        self.add_pipeline_message("Stream geleert", "info")

    def save_stream_log(self):
        """Save stream log to file - Claude Generated"""
        from PyQt6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Pipeline-Log speichern",
            f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)",
        )

        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    # Get plain text version
                    plain_text = self.stream_text.toPlainText()
                    f.write(f"ALIMA Pipeline Log - {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(plain_text)

                self.add_pipeline_message(f"Log gespeichert: {filename}", "success")
            except Exception as e:
                self.add_pipeline_message(f"Fehler beim Speichern: {e}", "error")

    def reset_for_new_pipeline(self):
        """Reset widget for new pipeline - Claude Generated"""
        self.current_step_id = None
        self.step_start_times.clear()
        self.is_streaming = False
