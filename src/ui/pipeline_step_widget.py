"""Single-step display widget for the pipeline tab. Claude Generated.

Extracted verbatim from ``pipeline_tab.py`` (F-5 god-file split). A standalone
``QFrame`` representing one pipeline step (status icon, name, provider/model
label, content area); no coupling back to ``PipelineTab`` — the tab imports and
instantiates it.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.pipeline_manager import PipelineStep
from .styles import set_style_prop, set_tone


class PipelineStepWidget(QFrame):
    """Widget representing a single pipeline step - Claude Generated"""

    step_clicked = pyqtSignal(str)  # step_id

    def __init__(self, step: PipelineStep, parent=None):
        super().__init__(parent)
        self.step = step
        self.setup_ui()

    def setup_ui(self):
        """Setup the step widget UI - Claude Generated"""
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        # Frame colours per state come from the main stylesheet
        # (QFrame#pipelineStep[state=...]); scoped by object name so they do
        # not bleed into the labels and text fields inside.
        self.setObjectName("pipelineStep")

        # Prevent sizeHint propagation from child QTextEdits - Claude Generated
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)

        # Header with step name and status
        header_layout = QHBoxLayout()

        # Status icon
        from .styles import get_scaled_font
        self.status_label = QLabel()
        self.status_label.setFont(get_scaled_font(size_delta=+4, bold=True))
        header_layout.addWidget(self.status_label)

        # Step name
        name_label = QLabel(self.step.name)
        name_label.setFont(get_scaled_font(size_delta=+2, bold=True))
        self._name_label = name_label  # keep ref for refresh_styles
        header_layout.addWidget(name_label)

        header_layout.addStretch()

        # Enhanced Provider/Model info with task preference indicators - Claude Generated
        self.provider_model_label = QLabel()
        set_tone(self.provider_model_label, "muted")
        self._update_provider_model_display()
        header_layout.addWidget(self.provider_model_label)

        layout.addLayout(header_layout)

        # Content area (initially empty)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 10, 0, 0)
        layout.addWidget(self.content_widget)

        # Now that all UI elements are created, update the display - Claude Generated
        self.update_status_display()

    def _update_provider_model_display(self):
        return

        """Update provider/model display with task preference indicators - Claude Generated"""
        # Safety check: ensure the label exists before updating - Claude Generated
        if not hasattr(self, 'provider_model_label') or not self.provider_model_label:
            return

        if not self.step.provider or not self.step.model:
            # Check if this is an LLM step that should have provider/model info
            llm_steps = ["initialisation", "keywords", "dk_classification"]
            if self.step.step_id in llm_steps:
                self.provider_model_label.setText("⚠️ No provider configured")
                set_tone(self.provider_model_label, "warning")
                self.provider_model_label.setStyleSheet("font-style: italic;")
            else:
                # Non-LLM steps (like search) don't need provider info
                self.provider_model_label.setText("No LLM required")
                set_tone(self.provider_model_label, "muted")
                self.provider_model_label.setStyleSheet("font-style: italic;")
            return

        # Build display text with visual indicators
        display_parts = []
        tone = "muted"

        # Check if this looks like a task preference (basic heuristic)
        task_preference_indicators = []
        if hasattr(self.step, 'selection_reason') and self.step.selection_reason:
            if "task preference" in self.step.selection_reason.lower():
                task_preference_indicators.append("⭐")
                tone = "success"  # task preferences
            elif "provider" in self.step.selection_reason.lower():
                task_preference_indicators.append("🔧")
                tone = "info"  # provider preferences
            elif "fallback" in self.step.selection_reason.lower():
                task_preference_indicators.append("🔄")
                tone = "warning"  # fallbacks

        # Format provider/model display
        indicator_prefix = "".join(task_preference_indicators)
        if indicator_prefix:
            display_parts.append(f"{indicator_prefix} {self.step.provider}/{self.step.model}")
        else:
            display_parts.append(f"{self.step.provider}/{self.step.model}")

        # Add compact selection reason if available
        if hasattr(self.step, 'selection_reason') and self.step.selection_reason:
            reason_short = self.step.selection_reason.replace("task preference", "TP").replace("provider preferences", "PP")
            if len(reason_short) < 30:  # Only show if compact enough
                display_parts.append(f"({reason_short})")

        display_text = " ".join(display_parts)
        self.provider_model_label.setText(display_text)
        set_tone(self.provider_model_label, tone)

        # Set tooltip with full details
        tooltip_parts = [f"Provider: {self.step.provider}", f"Model: {self.step.model}"]
        if hasattr(self.step, 'selection_reason') and self.step.selection_reason:
            tooltip_parts.append(f"Source: {self.step.selection_reason}")
        self.provider_model_label.setToolTip("\n".join(tooltip_parts))

    def update_status_display(self):
        """Update visual status indicator - Claude Generated"""
        # Symbol + colour per state: the symbol carries the meaning, the
        # colour repeats it.
        status = self.step.status
        if status == "pending":
            self.status_label.setText("▷")
            set_tone(self.status_label, "muted")
        elif status == "running":
            self.status_label.setText("▶")
            set_tone(self.status_label, "info")
        elif status == "completed":
            self.status_label.setText("✓")
            set_tone(self.status_label, "success")
        elif status == "error":
            self.status_label.setText("✗")
            set_tone(self.status_label, "error")
        if status in ("pending", "running", "completed", "error"):
            set_style_prop(self, "state", status)

        # Always update provider/model display when status changes - Claude Generated
        self._update_provider_model_display()

    def set_content(self, content_widget: QWidget):
        """Set the content widget for this step - Claude Generated"""
        # Clear existing content
        for i in reversed(range(self.content_layout.count())):
            child = self.content_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

        # Add new content
        self.content_layout.addWidget(content_widget)

    def update_step_data(self, step: PipelineStep):
        """Update step data and refresh display - Claude Generated"""
        self.step = step
        self.update_status_display()

    def refresh_styles(self):
        """Re-apply fonts after global font-size change. — Claude Generated"""
        from .styles import get_scaled_font
        if hasattr(self, "_name_label"):
            self._name_label.setFont(get_scaled_font(size_delta=+2, bold=True))
