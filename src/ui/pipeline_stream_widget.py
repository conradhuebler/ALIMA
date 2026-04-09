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
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QThread
from PyQt6.QtGui import QFont, QTextCursor, QColor, QPalette
from typing import Optional, Dict, Any, List
import logging
import time
from datetime import datetime
import json

from ..core.pipeline_manager import PipelineStep


class PipelineStreamWidget(QWidget):
    """Kontinuierliches Streaming-Widget für Pipeline-Feedback - Claude Generated"""

    # Signals for user interaction
    cancel_pipeline = pyqtSignal()
    pause_pipeline = pyqtSignal()
    retry_with_variations = pyqtSignal(dict)  # Retry with parameter variations - Claude Generated
    abort_generation_requested = pyqtSignal()  # Immediate abort of running generation - Claude Generated

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_step_id: Optional[str] = None
        self.is_streaming: bool = False
        self.step_start_times: Dict[str, datetime] = {}
        self.current_working_title: Optional[str] = None  # For log filename - Claude Generated
        self.current_suggestions: List[Dict] = []  # Current retry suggestions - Claude Generated
        self._last_scroll_time: float = 0.0  # Throttle auto-scroll to max 20/sec - Claude Generated

        self.setup_ui()

        # Size policy: Vertical Ignored to prevent sizeHint propagation to window - Claude Generated
        # Both this widget AND stream_text must have Ignored to prevent window expansion on streaming
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,   # Horizontal: darf wachsen
            QSizePolicy.Policy.Ignored      # Vertical: sizeHint ignorieren (kritisch für Multi-Monitor!)
        )

    def setup_ui(self):
        """Setup der Stream UI - Claude Generated"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main streaming area (header + text)
        self.create_streaming_area(layout)

        # Repetition warning panel at bottom (initially hidden) - Claude Generated
        self.create_repetition_warning_panel(layout)

    def create_streaming_area(self, layout):
        """Create main streaming text area with compact header - Claude Generated"""
        # Compact header row (replaces QGroupBox) - Claude Generated
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(6, 3, 6, 3)
        header_layout.setSpacing(6)

        title_label = QLabel("📝 Live")
        title_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.auto_scroll_checkbox = QCheckBox("Auto-scroll")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("font-size: 10px; color: #888;")
        header_layout.addWidget(self.auto_scroll_checkbox)

        self.clear_button = QPushButton("🗑️")
        self.clear_button.setFixedSize(26, 22)
        self.clear_button.setToolTip("Stream leeren")
        self.clear_button.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #555; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background: #333; }"
        )
        self.clear_button.clicked.connect(self.clear_stream)
        header_layout.addWidget(self.clear_button)

        self.save_log_button = QPushButton("💾")
        self.save_log_button.setFixedSize(26, 22)
        self.save_log_button.setToolTip("Log speichern")
        self.save_log_button.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #555; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background: #333; }"
        )
        self.save_log_button.clicked.connect(self.save_stream_log)
        header_layout.addWidget(self.save_log_button)

        layout.addLayout(header_layout)

        # Main streaming text widget
        self.stream_text = QTextEdit()
        self.stream_text.setReadOnly(True)
        self.stream_text.setMinimumHeight(200)

        self.stream_text.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored
        )

        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.stream_text.setFont(font)

        self.stream_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;
                color: #f8f8f2;
                border: none;
                border-top: 1px solid #333;
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

        layout.addWidget(self.stream_text)

    # Green resolved stylesheet - Claude Generated
    _STYLE_WARNING_GREEN = """
        QFrame {
            background-color: #1b3a1f;
            border: 1px solid #4caf50;
            border-radius: 3px;
            padding: 1px;
        }
        QLabel { color: #a5d6a7; }
        QPushButton {
            background-color: #2e7d32;
            color: #fff;
            border: none;
            border-radius: 3px;
            padding: 1px 5px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #43a047; }
    """

    # Orange warning stylesheet (reused in show/hide) - Claude Generated
    _STYLE_WARNING_ORANGE = """
        QFrame {
            background-color: #3d2a00;
            border: 1px solid #ff9800;
            border-radius: 3px;
            padding: 1px;
        }
        QLabel { color: #ffcc80; }
        QPushButton {
            background-color: #ff9800;
            color: #1e1e1e;
            border: none;
            border-radius: 3px;
            padding: 1px 5px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #ffb74d; }
    """

    # Transparent stylesheet for hidden warning panel (no visual footprint) - Claude Generated
    _STYLE_WARNING_HIDDEN = """
        QFrame { background: transparent; border: none; padding: 0; }
        QLabel { color: transparent; }
        QPushButton { background: transparent; border: none; color: transparent; }
    """

    def create_repetition_warning_panel(self, layout):
        """Create compact single-line repetition warning bar - Claude Generated"""
        self.repetition_warning_frame = QFrame()
        self.repetition_warning_frame.setFixedHeight(28)  # Always reserves 28px, no layout shift - Claude Generated
        self._warning_style_state = "hidden"  # Track style state to avoid redundant setStyleSheet - Claude Generated
        self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_HIDDEN)

        bar_layout = QHBoxLayout(self.repetition_warning_frame)
        bar_layout.setContentsMargins(6, 1, 4, 1)
        bar_layout.setSpacing(6)

        self.warning_icon_label = QLabel("⚠️")
        self.warning_icon_label.setStyleSheet("font-size: 11px;")
        bar_layout.addWidget(self.warning_icon_label)

        self.warning_title_label = QLabel("Wiederholung erkannt")
        self.warning_title_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #ff9800;")
        bar_layout.addWidget(self.warning_title_label)

        self.warning_details_label = QLabel("")
        self.warning_details_label.setWordWrap(False)
        self.warning_details_label.setStyleSheet("color: #ffe0b2; font-size: 10px;")
        bar_layout.addWidget(self.warning_details_label, 1)  # stretch fills remaining space

        self.countdown_label = QLabel("")
        self.countdown_label.setStyleSheet("color: #fff; font-weight: bold; font-size: 10px;")
        self.countdown_label.setVisible(False)
        bar_layout.addWidget(self.countdown_label)

        self.suggestions_button_layout = QHBoxLayout()
        self.suggestions_button_layout.setSpacing(3)
        bar_layout.addLayout(self.suggestions_button_layout)

        self.abort_now_button = QPushButton("🛑 Abbrechen")
        self.abort_now_button.setStyleSheet(
            "background-color: #d32f2f; color: white; font-size: 10px; font-weight: bold;"
            " border-radius: 3px; padding: 1px 5px;"
        )
        self.abort_now_button.clicked.connect(self._on_abort_requested)
        bar_layout.addWidget(self.abort_now_button)

        self.continue_button = QPushButton("Fortfahren")
        self.continue_button.setStyleSheet(
            "background-color: #555; color: #ccc; font-size: 10px; padding: 1px 5px;"
        )
        self.continue_button.clicked.connect(self.hide_repetition_warning)
        bar_layout.addWidget(self.continue_button)

        self.dismiss_warning_button = QPushButton("✕")
        self.dismiss_warning_button.setFixedSize(18, 18)
        self.dismiss_warning_button.setStyleSheet("background-color: transparent; color: #ff9800; padding: 0;")
        self.dismiss_warning_button.clicked.connect(self.hide_repetition_warning)
        bar_layout.addWidget(self.dismiss_warning_button)

        # Grace period countdown timer (created once, reused) - Claude Generated
        self.grace_timer = QTimer(self)
        self.grace_timer.timeout.connect(self._update_countdown)
        self.grace_period_end = 0.0

        layout.addWidget(self.repetition_warning_frame)

        # Start with children hidden (panel is transparent placeholder) - Claude Generated
        self._set_warning_children_visible(False)

    def _set_warning_children_visible(self, visible: bool):
        """Show/hide warning panel content without changing frame size - Claude Generated
        Frame stays at fixed 28px. When hidden, children are invisible and frame is transparent.
        """
        if not visible and self._warning_style_state != "hidden":
            self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_HIDDEN)
            self._warning_style_state = "hidden"
        for child in self.repetition_warning_frame.findChildren(QWidget):
            child.setVisible(visible)

    def show_repetition_warning(self, detection_type: str, details: str, suggestions: List[Dict],
                                 grace_period: bool = False, grace_seconds: float = 2.0):
        """Show repetition warning with optional countdown - Claude Generated (2026-02-17)

        Args:
            detection_type: Type of repetition detected
            details: Detection details
            suggestions: Parameter variation suggestions
            grace_period: If True, show countdown timer
            grace_seconds: Grace period duration in seconds
        """
        self.current_suggestions = suggestions

        # Reset to orange warning state (in case previous run left it green) - Claude Generated
        if self._warning_style_state != "orange":
            self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_ORANGE)
            self._warning_style_state = "orange"
        self.warning_icon_label.setText("⚠️")
        self.warning_title_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #ff9800;")
        self.continue_button.setVisible(True)

        # Set warning title based on detection type
        type_labels = {
            "char_pattern": "Zeichenwiederholung erkannt",
            "ngram": "Phrasenwiederholung erkannt",
            "window_similarity": "Textblock-Wiederholung erkannt",
        }
        self.warning_title_label.setText(type_labels.get(detection_type, "Wiederholung erkannt"))
        self.warning_details_label.setText(details)

        # Handle grace period countdown - Claude Generated (2026-02-17)
        if grace_period:
            self.grace_period_end = time.time() + grace_seconds
            self.grace_timer.stop()
            self.grace_timer.start(200)  # Update every 200ms (reused timer) - Claude Generated

            self.countdown_label.setText(f"⏳ {grace_seconds:.1f}s")
            self.countdown_label.setVisible(True)
        else:
            self.countdown_label.setVisible(False)
            self.grace_timer.stop()

        # Clear existing suggestion buttons
        while self.suggestions_button_layout.count():
            item = self.suggestions_button_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add suggestion buttons (compact, max 3 to save space in bar)
        for i, suggestion in enumerate(suggestions[:3]):
            button = QPushButton(suggestion.get("label", f"Option {i+1}"))
            button.setToolTip(suggestion.get("description", ""))
            button.setStyleSheet("font-size: 10px; padding: 1px 4px;")

            params = suggestion.get("params", {})

            def make_handler(p):
                return lambda: self._on_suggestion_clicked(p)

            button.clicked.connect(make_handler(params))
            self.suggestions_button_layout.addWidget(button)

        # Show the warning panel (children visible + orange style)
        self._set_warning_children_visible(True)

        # NOTE: Don't log to pipeline stream during streaming - it fragments the LLM output!
        # The warning panel already shows this information visually - Claude Generated (2026-02-17)

    def _update_countdown(self):
        """Update countdown timer display - Claude Generated (2026-02-17)"""
        remaining = self.grace_period_end - time.time()
        if remaining > 0:
            self.countdown_label.setText(f"⏳ {remaining:.1f}s")
        else:
            self.countdown_label.setText("⏳ …")
            self.grace_timer.stop()

    def hide_repetition_warning(self, resolved: bool = False):
        """Hide or resolve the repetition warning panel - Claude Generated

        Args:
            resolved: If True, switch to green success state (panel stays visible).
                      If False, actually hide the panel (explicit user dismiss or reset).
        """
        self.grace_timer.stop()
        self.countdown_label.setVisible(False)

        if resolved:
            # Switch to green "resolved" state – panel stays visible - Claude Generated
            if self._warning_style_state != "green":
                self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_GREEN)
                self._warning_style_state = "green"
            self.warning_icon_label.setText("✅")
            self.warning_title_label.setText("Wiederholung behoben – Generation läuft weiter")
            self.warning_title_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #4caf50;")
            self.warning_details_label.setText("")

            # Clear suggestion buttons
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # Hide "Trotzdem fortfahren" button, keep abort button visible
            self.continue_button.setVisible(False)

            # Panel stays visible – no setVisible(False) call
        else:
            # Explicit dismiss or pipeline reset: hide children + transparent frame - Claude Generated
            self._set_warning_children_visible(False)
            self.warning_icon_label.setText("⚠️")
            self.warning_title_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #ff9800;")
            self.continue_button.setVisible(True)

    def _on_abort_requested(self):
        """Handle immediate abort button click - Claude Generated"""
        self.hide_repetition_warning()
        self.abort_generation_requested.emit()

    def _on_suggestion_clicked(self, params: Dict):
        """Handle suggestion button click - Claude Generated"""
        self.hide_repetition_warning()
        self.retry_with_variations.emit(params)
        self.add_pipeline_message(
            f"🔄 Retry mit Parametern: {params}",
            "info",
            self.current_step_id
        )

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
        """Auto-scroll to bottom of text area (throttled to max 20/sec) - Claude Generated"""
        now = time.time()
        if now - self._last_scroll_time < 0.05:  # 50ms minimum gap
            return
        self._last_scroll_time = now
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
        # P1.5: Enhanced provider/model feedback - Claude Generated
        if step.provider and step.model:
            self.add_pipeline_message(
                f"✓ Verwende: {step.provider} / {step.model}", "success", step.step_id
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
            if step.step_id == "keywords" and ("keywords" in step.output_data or "final_keywords" in step.output_data):
                keywords = step.output_data.get("final_keywords", step.output_data.get("keywords", []))
                self.add_pipeline_message(
                    f"Gefunden: {len(keywords)} Keywords", "info", step.step_id
                )
                self.add_pipeline_message(
                    f"Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}",
                    "info",
                    step.step_id,
                )

                # Show GND verification summary - Claude Generated
                verification = step.output_data.get("verification")
                if verification and isinstance(verification, dict):
                    stats = verification.get("stats", {})
                    verified_count = stats.get("verified_count", 0)
                    total = stats.get("total_extracted", 0)
                    rejected = verification.get("rejected", [])
                    self.add_pipeline_message(
                        f"✅ {verified_count}/{total} Keywords GND-verifiziert",
                        "success", step.step_id,
                    )
                    if rejected:
                        rejected_names = [r.split('(')[0].strip() for r in rejected]
                        self.add_pipeline_message(
                            f"⚠️ {len(rejected)} Keywords ohne GND-Pool-Treffer entfernt: {', '.join(rejected_names)}",
                            "warning", step.step_id,
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
        """Display DK search results with status feedback (titles removed to reduce clutter) - Claude Generated

        Args:
            dk_results: List of keyword-centric result dictionaries:
                - "keyword": GND keyword with ID
                - "source": "cache" or "live"
                - "search_time_ms": Search time in milliseconds
                - "classifications": List of DK codes with details
            step_id: Pipeline step ID for message formatting
        """
        if not dk_results:
            self.add_pipeline_message("Keine Klassifikationen (DK/RVK) gefunden", "info", step_id)
            return

        # Calculate total statistics
        total_keywords = len(dk_results)
        total_classifications = sum(len(r.get("classifications", [])) for r in dk_results)
        cache_count = sum(1 for r in dk_results if r.get("source") == "cache")
        live_count = total_keywords - cache_count
        success_count = sum(1 for r in dk_results if r.get("classifications"))

        # Display summary header with status overview - Claude Generated (Enhanced feedback)
        self.add_pipeline_message(
            f"🔍 Klassifikationssuche: {total_keywords} Keywords → {success_count} erfolgreich → {total_classifications} Klassifikationen",
            "info",
            step_id,
        )

        if cache_count > 0 or live_count > 0:
            self.add_pipeline_message(
                f"   📦 Cache: {cache_count} | 🔍 Live: {live_count}",
                "debug",
                step_id,
            )

        # Display per-keyword results with success/failure indicators - Claude Generated (No titles)
        for keyword_result in dk_results:
            keyword = keyword_result.get("keyword", "unknown")
            source = keyword_result.get("source", "unknown")
            search_time = keyword_result.get("search_time_ms", 0)
            classifications = keyword_result.get("classifications", [])

            # Determine status icon
            if classifications:
                status_icon = "✅"
                classification_count = len(classifications)
                msg_type = "info"
                status_text = f"{classification_count} Klassifikationen"
            else:
                status_icon = "⚠️"
                msg_type = "warning"
                status_text = "Keine Klassifikationen"

            # Display keyword with status indicator
            source_icon = "📦" if source == "cache" else "🔍"
            timing_text = f"({search_time:.1f}ms)" if search_time > 0 else ""

            self.add_pipeline_message(
                f"{status_icon} {source_icon} {keyword} - {status_text} {timing_text}",
                msg_type,
                step_id,
            )

            # Display DK codes summary (not individual codes to avoid spam)
            if classifications:
                total_dk = len(classifications)
                self.add_pipeline_message(
                    f"   ✓ {total_dk} Klassifikationen (DK/RVK) gefunden",
                    "debug",
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

        # Show result summary for agentic pipeline (no per-step signals are emitted)
        if analysis_state and hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
            kw_list = analysis_state.final_llm_analysis.extracted_gnd_keywords or []
            if kw_list:
                self.add_pipeline_message(
                    f"📌 {len(kw_list)} GND-Schlagworte ausgewählt: "
                    f"{', '.join(kw_list[:5])}{'...' if len(kw_list) > 5 else ''}",
                    "success"
                )
        if analysis_state and analysis_state.dk_classifications:
            dk_codes = analysis_state.dk_classifications
            self.add_pipeline_message(
                f"🏷 DK-Klassifikationen: {', '.join(dk_codes[:5])}{'...' if len(dk_codes) > 5 else ''}",
                "success"
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

        # Use working_title for log filename if available - Claude Generated
        if self.current_working_title:
            default_filename = f"{self.current_working_title}_log.txt"
        else:
            default_filename = f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        from pathlib import Path
        docs_dir = Path.home() / "Documents"
        if not docs_dir.exists():
            docs_dir = Path.home()
        default_path = str(docs_dir / default_filename)

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Pipeline-Log speichern",
            default_path,
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

    def set_working_title(self, working_title: str):
        """Set working title for log filename - Claude Generated"""
        self.current_working_title = working_title
        self.logger.info(f"Stream widget: working_title set to '{working_title}'")

    def reset_for_new_pipeline(self):
        """Reset widget for new pipeline - Claude Generated"""
        self.current_step_id = None
        self.step_start_times.clear()
        self.is_streaming = False
        self.current_working_title = None  # Reset title for new pipeline - Claude Generated

        # Clear stream content and hide repetition warning
        self.clear_stream()
        self.hide_repetition_warning()
