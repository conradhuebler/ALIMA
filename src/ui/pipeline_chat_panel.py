"""PipelineChatPanel — unified pipeline log + chat dock (WP10 P-δ.5a).

Claude Generated.

Replaces the previously separate ``PipelineStreamWidget`` (right-side
panel in ``PipelineTab``) and the floating ``ChatWidget`` (``chat_dock``).
Single widget hosting a shared log area that renders:

- Pipeline step events (▶/✅/❌, durations, GND-verification stats, DK
  catalog results, repetition warnings).
- LLM streaming tokens (purple) for pipeline LLM calls.
- User chat turns (WhatsApp-style right-aligned green bubble).
- Assistant chat turns (left-aligned grey bubble, streaming-capable
  cell-cursor).
- Tool-call markers — from both the chat-agent (direct signal) and the
  agentic-pipeline ``AgentLoop`` (via ``AlimaStateBus`` events).
- Status messages from the AgentLoop status channel.

The chat input lives at the bottom of the same panel. Operator can chat
about pipeline state in-place; future P-ζ lets the agent itself drive
the pipeline and "talk to itself" in the same log.

Public API preserved from ``PipelineStreamWidget`` so existing callers
in ``pipeline_tab.py`` / ``pipeline_manager.py`` need no signature
changes.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QKeyEvent
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# NB: WebLogView (which imports QWebEngineView) is imported lazily where the
# log widget is built, not at module level. QWebEngineView MUST be imported
# before the QApplication is created — that ordering is guaranteed by the
# explicit early import in alima_gui.py. Keeping this import lazy also lets the
# test suite swap in a lightweight stub. - Claude Generated

from ..core.chat_prompts import (
    DEFAULT_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_system_prompt,
    get_user_prompt_template,
    detect_mode,
)
from ..core.headless_agent import resolve_provider_model
from ..core.pipeline_manager import PipelineStep
from ..core.state_bus import AlimaStateBus
from ..utils.pipeline_utils import PipelineResultFormatter
from .chat_agent_worker import ChatAgentWorker
from .chat_session import ChatSession
from .chat_tools import build_chat_toolset
from .styles import (
    LAYOUT,
    get_button_styles,
    get_main_stylesheet,
    get_scaled_font,
)


# ----------------------------------------------------------------------
# System-Prompt edit dialog (was previously in chat_widget.py).
# ----------------------------------------------------------------------


class SystemPromptDialog(QDialog):
    """Small dialog for editing the assistant system prompt."""

    def __init__(self, current_prompt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ System-Prompt bearbeiten")
        self.setMinimumSize(500, 300)
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        m = LAYOUT.get("margin", 12)
        layout.setContentsMargins(m, m, m, m)
        layout.setSpacing(LAYOUT.get("spacing", 8))

        info = QLabel(
            "Dieser Prompt definiert die Rolle des Assistenten. "
            "Änderungen wirken sich sofort auf die nächste Nachricht aus."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info)

        self.editor = QTextEdit()
        self.editor.setPlainText(current_prompt)
        self.editor.setFont(get_scaled_font(size_delta=-1, monospace=False))
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_prompt(self) -> str:
        return self.editor.toPlainText().strip()


# ----------------------------------------------------------------------
# ChatInputEdit — multi-line QTextEdit, Enter sends, Shift+Enter newline.
# ----------------------------------------------------------------------


class ChatInputEdit(QTextEdit):
    """Multi-line chat input. Enter = submit, Shift+Enter = newline."""

    submit = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submit.emit()
            return
        super().keyPressEvent(event)


# ----------------------------------------------------------------------
# PipelineChatPanel
# ----------------------------------------------------------------------


class PipelineChatPanel(QWidget):
    """Unified pipeline-log + chat panel (WP10 P-δ.5a)."""

    # -- Pipeline signals (preserved from PipelineStreamWidget) ---------
    cancel_pipeline = pyqtSignal()
    pause_pipeline = pyqtSignal()
    retry_with_variations = pyqtSignal(dict)
    abort_generation_requested = pyqtSignal()

    # -- Chat signal ----------------------------------------------------
    message_sent = pyqtSignal(str)

    # ------------------------------------------------------------------
    # Default prompts (chat-agent).
    # ------------------------------------------------------------------

    # Default chat-agent prompts now live in src/core/chat_prompts.py
    # (Qt-free, shared with headless CLI/HTTP frontends). Aliased onto the
    # class so the existing self.DEFAULT_SYSTEM_PROMPT / self.USER_PROMPT_TEMPLATE
    # references keep working. RHS resolves to the module-level import.
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE

    # NB (Phase F): ``_STATUS_SKIP_PREFIXES`` removed. The status-line
    # duplication in ``agent_loop.py`` is now suppressed at the source
    # when ``on_tool_call`` is wired (the bus/hook already conveys the
    # call/result). The filter existed because of that duplication —
    # no longer needed.

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm_service=None,
        prompt_service=None,
        pipeline_manager=None,
        mcp_registry=None,
        parent=None,
    ):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        # Chat-side dependencies
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.pipeline_manager = pipeline_manager
        if mcp_registry is None:
            from src.mcp.tool_registry import ToolRegistry
            mcp_registry = ToolRegistry()
            mcp_registry.register_all_tools()
        self.mcp_registry = mcp_registry

        # Pipeline-side state
        self.current_step_id: Optional[str] = None
        self.step_start_times: Dict[str, datetime] = {}
        self.current_working_title: Optional[str] = None
        self.current_suggestions: List[Dict] = []
        # Agentic input/prompt collapsible blocks: prompt_id → (tool_id, base_meta)
        self._prompt_blocks: Dict[str, str] = {}
        self._prompt_meta: Dict[str, str] = {}

        # Chat-side state
        self.system_prompt: str = self.DEFAULT_SYSTEM_PROMPT
        self.session: ChatSession = ChatSession()
        self.current_worker: Optional[ChatAgentWorker] = None
        self.current_context: str = ""
        self.working_title: str = ""
        self._current_render_model: str = ""
        self._typing_dots: int = 0
        self._typing_model: str = ""
        self._last_tool_call_id: Optional[str] = None
        self._bus_tool_call_ids: Dict[str, str] = {}
        # Classic-pipeline step → tool-block bridge (step_id → renderer tool_id).
        self._step_tool_call_ids: Dict[str, str] = {}
        # True while a bus pipeline-step block is open ("running" received,
        # no terminal event yet). Used to route status messages correctly.
        self._pipeline_step_open: bool = False
        # Post-(F) polish: status-line accumulator for the open
        # pipeline-step tool block. While a step is "running", incoming
        # status messages (per-keyword search progress, etc.) are
        # captured here and rendered as the block's expanded body when
        # the step completes — no more duplicate markers under the
        # collapsed block.
        self._open_step_status: List[str] = []

        # P-ε: cross-thread bridge for mutation tool confirmations.
        from src.ui.chat_tools.proposal_gateway import ProposalGateway
        self.proposal_gateway = ProposalGateway(self)
        self.proposal_gateway.proposal_requested.connect(
            self._render_proposal_bubble
        )

        self.setup_ui()

        # Unified renderer — all QTextBrowser manipulation lives here.
        from .unified_message_renderer import UnifiedMessageRenderer
        self._renderer = UnifiedMessageRenderer(
            self.stream_text,
            self.auto_scroll_checkbox,
        )
        # P-δ.5: wire catalog web-OPAC base URL so the renderer can turn
        # <<CAT:rsn|text>> markers into clickable links. Degrades cleanly
        # (markers are reduced to plain text) if config is missing.
        try:
            from src.utils.config_manager import ConfigManager
            from urllib.parse import urlparse
            cat_cfg = ConfigManager().get_catalog_config()
            web_base = getattr(cat_cfg, "catalog_web_record_url", "") or ""
            self._renderer.set_catalog_web_base(web_base)
            # Derive catalog host (scheme+netloc) for ext-link classification.
            if web_base:
                p = urlparse(web_base)
                if p.scheme and p.netloc:
                    self._renderer.set_catalog_host(f"{p.scheme}://{p.netloc}")
        except Exception:
            pass  # feature disabled silently — see _replace_cat_markers

        # Size policy: vertical Ignored to prevent sizeHint propagation
        # to window (multi-monitor safety, inherited from PipelineStreamWidget).
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )

        # AlimaStateBus subscriptions:
        # - state.changed → refresh SharedContext snapshot for chat tools.
        # - tool.called / tool.result → render markers for agentic pipeline
        #   and chat-agent (Phase D unified producer).
        # - state.pipeline_step → render pipeline-step tool blocks.
        # - state.pipeline_started / state.pipeline_completed → previously
        #   dead events; now bound to the panel's slot logic (Phase F).
        try:
            bus = AlimaStateBus()
            bus.subscribe("state.changed", self._on_state_changed)
            bus.subscribe("tool.called", self._on_bus_tool_called)
            bus.subscribe("tool.result", self._on_bus_tool_result)
            bus.subscribe("state.pipeline_step", self._on_bus_pipeline_step)
            bus.subscribe("state.pipeline_prompt", self._on_bus_pipeline_prompt)
            bus.subscribe(
                "state.pipeline_prompt_done", self._on_bus_pipeline_prompt_done
            )
            bus.subscribe(
                "state.pipeline_started",
                lambda p: self.on_pipeline_started(p.get("pipeline_id", "")),
            )
            bus.subscribe(
                "state.pipeline_completed",
                lambda p: self._on_bus_pipeline_completed(p),
            )
        except Exception:
            self.logger.exception("PipelineChatPanel: AlimaStateBus subscribe failed")

    @property
    def is_streaming(self) -> bool:
        """Backward-compat for external callers (pipeline_tab.py)."""
        return self._renderer._is_streaming

    @property
    def _last_scroll_time(self) -> float:
        return self._renderer._last_scroll_time

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def setup_ui(self):
        self.setStyleSheet(get_main_stylesheet())
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Single consolidated header --------------------------------
        header = QFrame()
        header.setStyleSheet(
            "QFrame { background-color: #2d2d2d; border-bottom: 1px solid #444; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        title_label = QLabel("📝 Pipeline + 💬 Chat")
        title_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        header_layout.addWidget(title_label)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setMaximumWidth(280)
        self.model_combo.setStyleSheet(
            "QComboBox { font-size: 10px; padding: 2px 6px; border: 1px solid #555; "
            "border-radius: 3px; background-color: #3d3d3d; color: #ccc; }"
            "QComboBox QAbstractItemView { background-color: #2b2b2b; color: #ccc; "
            "selection-background-color: #005fcc; selection-color: white; border: 1px solid #555; }"
        )
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        header_layout.addWidget(self.model_combo)
        self.model_combo.setEditable(True)
        from PyQt6.QtWidgets import QCompleter
        self.model_combo.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)

        self.persist_combo_toggle = QCheckBox("💾")
        self.persist_combo_toggle.setChecked(False)
        self.persist_combo_toggle.setStyleSheet("color: #aaa; font-size: 10px;")
        self.persist_combo_toggle.setToolTip(
            "Bei Combo-Wechsel als ChatConfig-Default speichern."
        )
        header_layout.addWidget(self.persist_combo_toggle)

        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet(
            "color: #8be9fd; font-size: 10px; padding-left: 4px;"
        )
        header_layout.addWidget(self.model_status_label)

        header_layout.addStretch()

        # Compact icon-style controls.
        icon_btn_style = (
            "QPushButton { background: transparent; border: 1px solid #555; "
            "border-radius: 3px; color: #ccc; font-size: 10px; padding: 2px 6px; }"
            "QPushButton:hover { background: #3d3d3d; }"
        )

        self.system_prompt_btn = QPushButton("⚙️")
        self.system_prompt_btn.setFixedSize(26, 22)
        self.system_prompt_btn.setStyleSheet(icon_btn_style)
        self.system_prompt_btn.setToolTip("System-Prompt bearbeiten")
        self.system_prompt_btn.clicked.connect(self.show_system_prompt_dialog)
        header_layout.addWidget(self.system_prompt_btn)

        self.reset_toggle = QCheckBox("🔄 Reset")
        self.reset_toggle.setChecked(True)
        self.reset_toggle.setStyleSheet("color: #aaa; font-size: 10px;")
        self.reset_toggle.setToolTip(
            "Chat-Verlauf bei jedem neuen Pipeline-Lauf leeren."
        )
        header_layout.addWidget(self.reset_toggle)

        self.autonomous_toggle = QCheckBox("🤖 Autonom")
        self.autonomous_toggle.setStyleSheet("color: #aaa; font-size: 10px;")
        self.autonomous_toggle.setToolTip(
            "Mutationen ohne Rückfrage anwenden (ChatConfig.autonomous_pipeline). "
            "Default off — Agent fragt vor jeder Änderung."
        )
        # Initial state from ChatConfig.
        try:
            cfg = self._get_chat_config()
            self.autonomous_toggle.setChecked(
                bool(getattr(cfg, "autonomous_pipeline", False))
            )
        except Exception:
            pass
        self.autonomous_toggle.stateChanged.connect(
            self._on_autonomous_toggle_changed
        )
        header_layout.addWidget(self.autonomous_toggle)

        self.auto_scroll_checkbox = QCheckBox("⬇")
        self.auto_scroll_checkbox.setChecked(True)
        self.auto_scroll_checkbox.setStyleSheet("color: #aaa; font-size: 10px;")
        self.auto_scroll_checkbox.setToolTip("Auto-scroll Log")
        header_layout.addWidget(self.auto_scroll_checkbox)

        self.clear_button = QPushButton("🗑️")
        self.clear_button.setFixedSize(26, 22)
        self.clear_button.setStyleSheet(icon_btn_style)
        self.clear_button.setToolTip("Log leeren")
        self.clear_button.clicked.connect(self.clear_stream)
        header_layout.addWidget(self.clear_button)

        self.save_log_button = QPushButton("💾")
        self.save_log_button.setFixedSize(26, 22)
        self.save_log_button.setStyleSheet(icon_btn_style)
        self.save_log_button.setToolTip("Log speichern")
        self.save_log_button.clicked.connect(self.save_stream_log)
        header_layout.addWidget(self.save_log_button)

        self.cancel_btn = QPushButton("⏹ Abbrechen")
        self.cancel_btn.setStyleSheet(get_button_styles().get("danger", ""))
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setToolTip("Aktuelle Generierung abbrechen")
        self.cancel_btn.clicked.connect(self.cancel_generation)
        header_layout.addWidget(self.cancel_btn)

        outer.addWidget(header)

        # --- Vertical splitter between log + input (user-draggable) ----
        self.body_splitter = QSplitter(Qt.Orientation.Vertical)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(4)
        self.body_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #2d2d2d; }"
            "QSplitter::handle:hover { background-color: #555; }"
        )

        # Top half = log area container (stream_text + warning + typing).
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)

        # --- Main log area (QWebEngineView-backed unified renderer) ---
        # Collapsible blocks are native <details>; mutation:// and http(s)://
        # link clicks are routed back here via link_clicked (replaces the old
        # QTextBrowser.anchorClicked). - Claude Generated
        from .styles import get_font_size
        from .web_log_view import WebLogView

        self.stream_text = WebLogView(base_font_pt=get_font_size())
        self.stream_text.link_clicked.connect(self._on_anchor_clicked)
        self.stream_text.setMinimumHeight(80)
        self.stream_text.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        log_layout.addWidget(self.stream_text)

        # --- Repetition-warning panel (pipeline-only feature) ---
        self.create_repetition_warning_panel(log_layout)

        # --- Typing indicator (chat-only) ---
        self.typing_label = QLabel("")
        self.typing_label.setStyleSheet(
            "color: #8be9fd; font-size: 9pt; font-style: italic; "
            "padding: 2px 12px; background-color: #1e1e1e; "
            "border-top: 1px solid #2a2a2a;"
        )
        self.typing_label.setVisible(False)
        log_layout.addWidget(self.typing_label)
        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(400)
        self._typing_timer.timeout.connect(self._tick_typing)

        self.body_splitter.addWidget(log_container)

        # --- Chat input frame (expandable QTextEdit) -------------------
        input_frame = QFrame()
        input_frame.setStyleSheet(
            "QFrame { background-color: #2d2d2d; border-top: 1px solid #444; }"
        )
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(6, 4, 6, 4)
        input_layout.setSpacing(6)

        self.input_field = ChatInputEdit()
        self.input_field.setPlaceholderText(
            "Frage stellen — Enter = senden, Shift+Enter = neue Zeile"
        )
        self.input_field.setStyleSheet(
            "QTextEdit { background-color: #3d3d3d; color: #e0e0e0; "
            "border: 1px solid #555; border-radius: 4px; padding: 4px 8px; "
            "font-size: 11pt; }"
            "QTextEdit:focus { border: 1px solid #8be9fd; }"
        )
        self.input_field.setFont(get_scaled_font(size_delta=0))
        fm = self.input_field.fontMetrics()
        line_h = fm.lineSpacing()
        self.input_field.setMinimumHeight(line_h * 3 + 14)
        self.input_field.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.input_field.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.input_field.submit.connect(self.send_message)
        input_layout.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Senden")
        self.send_btn.setStyleSheet(get_button_styles().get("primary", ""))
        self.send_btn.setDefault(True)
        self.send_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(
            self.send_btn,
            alignment=Qt.AlignmentFlag.AlignBottom,
        )

        self.body_splitter.addWidget(input_frame)

        # Default ratio: log gets ~80%, input ~20% — both grow with window.
        self.body_splitter.setStretchFactor(0, 8)
        self.body_splitter.setStretchFactor(1, 2)
        self.body_splitter.setSizes([800, 200])

        outer.addWidget(self.body_splitter, stretch=1)

        # Initial model-status paint.
        self._refresh_model_status()

    # ==================================================================
    # Repetition-warning panel (verbatim port from PipelineStreamWidget)
    # ==================================================================

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

    _STYLE_WARNING_HIDDEN = """
        QFrame { background: transparent; border: none; padding: 0; }
        QLabel { color: transparent; }
        QPushButton { background: transparent; border: none; color: transparent; }
    """

    def create_repetition_warning_panel(self, layout):
        self.repetition_warning_frame = QFrame()
        self.repetition_warning_frame.setFixedHeight(28)
        self._warning_style_state = "hidden"
        self._last_shown_detection_type = ""
        self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_HIDDEN)

        bar = QHBoxLayout(self.repetition_warning_frame)
        bar.setContentsMargins(6, 1, 4, 1)
        bar.setSpacing(6)

        self.warning_icon_label = QLabel("⚠️")
        bar.addWidget(self.warning_icon_label)

        self.warning_title_label = QLabel("Wiederholung erkannt")
        self.warning_title_label.setStyleSheet("font-weight: bold; color: #ff9800;")
        bar.addWidget(self.warning_title_label)

        self.warning_details_label = QLabel("")
        self.warning_details_label.setWordWrap(False)
        self.warning_details_label.setStyleSheet("color: #ffe0b2;")
        bar.addWidget(self.warning_details_label, 1)

        self.countdown_label = QLabel("")
        self.countdown_label.setStyleSheet("color: #fff; font-weight: bold;")
        self.countdown_label.setVisible(False)
        bar.addWidget(self.countdown_label)

        self.suggestions_button_layout = QHBoxLayout()
        self.suggestions_button_layout.setSpacing(3)
        bar.addLayout(self.suggestions_button_layout)

        self.abort_now_button = QPushButton("🛑 Abbrechen")
        self.abort_now_button.setStyleSheet(
            "background-color: #d32f2f; color: white; font-weight: bold;"
            " border-radius: 3px; padding: 1px 5px;"
        )
        self.abort_now_button.clicked.connect(self._on_abort_requested)
        bar.addWidget(self.abort_now_button)

        self.continue_button = QPushButton("Fortfahren")
        self.continue_button.setStyleSheet(
            "background-color: #555; color: #ccc; padding: 1px 5px;"
        )
        self.continue_button.clicked.connect(self.hide_repetition_warning)
        bar.addWidget(self.continue_button)

        self.dismiss_warning_button = QPushButton("✕")
        self.dismiss_warning_button.setFixedSize(18, 18)
        self.dismiss_warning_button.setStyleSheet(
            "background-color: transparent; color: #ff9800; padding: 0;"
        )
        self.dismiss_warning_button.clicked.connect(self.hide_repetition_warning)
        bar.addWidget(self.dismiss_warning_button)

        self.grace_timer = QTimer(self)
        self.grace_timer.timeout.connect(self._update_countdown)
        self.grace_period_end = 0.0

        layout.addWidget(self.repetition_warning_frame)

    def show_repetition_warning(
        self,
        detection_type: str,
        details: str,
        suggestions: List[Dict],
        grace_period: bool = False,
        grace_seconds: float = 2.0,
    ):
        self.current_suggestions = suggestions
        already_showing = (
            self._warning_style_state == "orange"
            and self._last_shown_detection_type == detection_type
        )
        if self._warning_style_state != "orange":
            self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_ORANGE)
            self._warning_style_state = "orange"
        self._last_shown_detection_type = detection_type

        if not already_showing:
            self.warning_icon_label.setText("⚠️")
            self.warning_title_label.setStyleSheet(
                "font-weight: bold; color: #ff9800;"
            )
            self.continue_button.setStyleSheet(
                "background-color: #555; color: #ccc; padding: 1px 5px;"
            )

            type_labels = {
                "char_pattern": "Zeichenwiederholung erkannt",
                "ngram": "Phrasenwiederholung erkannt",
                "window_similarity": "Textblock-Wiederholung erkannt",
            }
            self.warning_title_label.setText(
                type_labels.get(detection_type, "Wiederholung erkannt")
            )

            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            for i, suggestion in enumerate(suggestions[:3]):
                button = QPushButton(suggestion.get("label", f"Option {i+1}"))
                button.setToolTip(suggestion.get("description", ""))
                button.setStyleSheet("padding: 1px 4px;")
                params = suggestion.get("params", {})

                def make_handler(p):
                    return lambda: self._on_suggestion_clicked(p)

                button.clicked.connect(make_handler(params))
                self.suggestions_button_layout.addWidget(button)

        self.warning_details_label.setText(details)

        if grace_period and not already_showing:
            self.grace_period_end = time.time() + grace_seconds
            self.grace_timer.stop()
            self.grace_timer.start(200)
            self.countdown_label.setText(f"⏳ {grace_seconds:.1f}s")
        elif not grace_period:
            self.countdown_label.setText("")
            self.grace_timer.stop()

    def _update_countdown(self):
        remaining = self.grace_period_end - time.time()
        if remaining > 0:
            self.countdown_label.setText(f"⏳ {remaining:.1f}s")
        else:
            self.countdown_label.setText("⏳ …")
            self.grace_timer.stop()

    def hide_repetition_warning(self, resolved: bool = False):
        self.grace_timer.stop()
        self.countdown_label.setText("")
        self._last_shown_detection_type = ""

        if resolved:
            if self._warning_style_state != "green":
                self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_GREEN)
                self._warning_style_state = "green"
            self.warning_icon_label.setText("✅")
            self.warning_title_label.setText(
                "Wiederholung behoben – Generation läuft weiter"
            )
            self.warning_title_label.setStyleSheet(
                "font-weight: bold; color: #4caf50;"
            )
            self.warning_details_label.setText("")
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.continue_button.setStyleSheet(
                "background: transparent; border: none; color: transparent;"
            )
        else:
            if self._warning_style_state != "hidden":
                self.repetition_warning_frame.setStyleSheet(self._STYLE_WARNING_HIDDEN)
                self._warning_style_state = "hidden"
            self.warning_icon_label.setText("")
            self.warning_title_label.setText("")
            self.warning_details_label.setText("")
            self.countdown_label.setText("")
            self.continue_button.setStyleSheet(
                "background: transparent; border: none; color: transparent;"
            )
            while self.suggestions_button_layout.count():
                item = self.suggestions_button_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def _on_abort_requested(self):
        self.hide_repetition_warning()
        self.abort_generation_requested.emit()

    def _on_suggestion_clicked(self, params: Dict):
        self.hide_repetition_warning()
        self.retry_with_variations.emit(params)
        self.add_pipeline_message(
            f"🔄 Retry mit Parametern: {params}",
            "info",
            self.current_step_id,
        )

    # ==================================================================
    # Pipeline rendering API (preserved from PipelineStreamWidget)
    # ==================================================================

    def add_pipeline_message(
        self,
        message: str,
        level: str = "info",
        step_id: Optional[str] = None,
    ):
        self._renderer.render_pipeline_log(message, level, step_id)

    def add_streaming_token(self, token: str, step_id: str):
        # The SYSTEM/USER prompt dump is rendered as a collapsible 📥 Input
        # block via the state.pipeline_prompt bus event — drop the inline
        # duplicate so it isn't shown twice. - Claude Generated
        if token.lstrip().startswith("--- SYSTEM ---"):
            return
        self._renderer.render_streaming_token(token, step_id)

    def start_streaming_line(self, step_id: str, prefix: str = ""):
        self._renderer.start_streaming_line(step_id, prefix)

    def end_streaming_line(self):
        self._renderer.end_streaming_line()

    def auto_scroll_to_bottom(self):
        self._renderer.auto_scroll_to_bottom()

    @pyqtSlot(object)
    def on_pipeline_started(self, pipeline_id: str):
        self.add_pipeline_message("🚀 Pipeline gestartet", "step")
        self.add_pipeline_message(f"Pipeline ID: {pipeline_id}", "info")
        self.pipeline_start_time = datetime.now()

    @pyqtSlot(object)
    def on_step_started(self, step: PipelineStep):
        self.current_step_id = step.step_id
        self.step_start_times[step.step_id] = datetime.now()
        args: Dict[str, Any] = {"name": step.name}
        if step.provider and step.model:
            args["provider"] = f"{step.provider}/{step.model}"
        tool_id = self._renderer.render_tool_call(f"pipeline.{step.step_id}", args)
        self._step_tool_call_ids[step.step_id] = tool_id

    def _build_step_summary(self, step: "PipelineStep", duration: str) -> str:
        """Return a multiline summary string for a completed pipeline step."""
        lines = [f"✅ Abgeschlossen in {duration}"]
        if not step.output_data:
            return "\n".join(lines)

        if step.step_id == "keywords" and (
            "keywords" in step.output_data or "final_keywords" in step.output_data
        ):
            keywords = step.output_data.get(
                "final_keywords", step.output_data.get("keywords", [])
            )
            lines.append(f"Gefunden: {len(keywords)} Keywords")
            if keywords:
                preview = ", ".join(keywords[:5]) + ("..." if len(keywords) > 5 else "")
                lines.append(f"Keywords: {preview}")
            verification = step.output_data.get("verification")
            if verification and isinstance(verification, dict):
                stats = verification.get("stats", {})
                verified_count = stats.get("verified_count", 0)
                total = stats.get("total_extracted", 0)
                rejected = verification.get("rejected", [])
                lines.append(f"✅ {verified_count}/{total} Keywords GND-verifiziert")
                if rejected:
                    rejected_names = [r.split("(")[0].strip() for r in rejected]
                    lines.append(
                        f"⚠️ {len(rejected)} Keywords ohne GND-Pool-Treffer entfernt: "
                        + ", ".join(rejected_names)
                    )

        elif step.step_id == "search" and "search_results" in step.output_data:
            count = step.output_data["search_results"]
            lines.append(f"Gefunden: {count} GND-Einträge")

        elif step.step_id == "verification" and "verified_keywords" in step.output_data:
            verified = step.output_data["verified_keywords"]
            lines.append(f"Verifiziert: {len(verified)} Keywords")

        elif step.step_id == "dk_search" and "dk_search_results" in step.output_data:
            lines.append(self._format_dk_search_results(step.output_data["dk_search_results"]))

        return "\n".join(lines)

    @pyqtSlot(object)
    def on_step_completed(self, step: PipelineStep):
        duration = "unbekannt"
        if step.step_id in self.step_start_times:
            duration_seconds = (
                datetime.now() - self.step_start_times[step.step_id]
            ).total_seconds()
            duration = f"{duration_seconds:.1f}s"

        summary = self._build_step_summary(step, duration)
        tool_id = self._step_tool_call_ids.pop(step.step_id, None)
        if tool_id:
            self._renderer.render_tool_result(tool_id, summary, status="success")
        else:
            # Fallback: no tool block was opened for this step (e.g. step fired
            # before the panel was ready), emit as flat log lines.
            self.add_pipeline_message(
                f"✅ Schritt abgeschlossen in {duration}", "success", step.step_id
            )

        # Render the per-DK-code catalog-research result identically to the
        # Pipeline-Tab (shared formatter), in addition to the keyword-timing
        # summary kept in the collapsible tool block above. - Claude Generated
        if step.step_id == "dk_search" and step.output_data:
            self._render_dk_search_card(step.output_data)

    def _render_dk_search_card(self, output_data: Dict[str, Any]) -> None:
        """Render per-DK-code catalog-research results as a card (shared formatter).

        WP12: the card HTML is produced by the shared
        ``PipelineResultFormatter.format_dk_search_card_html`` so the GUI and the
        webapp emit byte-identical chrome from one source.
        """
        html, text = PipelineResultFormatter.format_dk_search_card_html(output_data)
        if html:
            self._renderer.render_html_block(html, kind="dk_search", plain_text=text)

    def _format_dk_search_results(self, dk_results: List[Dict[str, Any]]) -> str:
        """Build a summary string for DK search results (tool-block body)."""
        if not dk_results:
            return "Keine Klassifikationen (DK/RVK) gefunden"

        total_keywords = len(dk_results)
        total_classifications = sum(
            len(r.get("classifications", [])) for r in dk_results
        )
        cache_count = sum(1 for r in dk_results if r.get("source") == "cache")
        live_count = total_keywords - cache_count
        success_count = sum(1 for r in dk_results if r.get("classifications"))

        lines = [
            f"🔍 Klassifikationssuche: {total_keywords} Keywords → "
            f"{success_count} erfolgreich → {total_classifications} Klassifikationen",
        ]
        if cache_count > 0 or live_count > 0:
            lines.append(f"   📦 Cache: {cache_count} | 🔍 Live: {live_count}")

        for keyword_result in dk_results:
            keyword = keyword_result.get("keyword", "unknown")
            source = keyword_result.get("source", "unknown")
            search_time = keyword_result.get("search_time_ms", 0)
            classifications = keyword_result.get("classifications", [])
            status_icon = "✅" if classifications else "⚠️"
            status_text = f"{len(classifications)} Klassifikationen" if classifications else "Keine Klassifikationen"
            source_icon = "📦" if source == "cache" else "🔍"
            timing_text = f"({search_time:.1f}ms)" if search_time > 0 else ""
            lines.append(f"{status_icon} {source_icon} {keyword} - {status_text} {timing_text}")

        return "\n".join(lines)

    @pyqtSlot(object, str)
    def on_step_error(self, step: PipelineStep, error_message: str):
        tool_id = self._step_tool_call_ids.pop(step.step_id, None)
        if tool_id:
            self._renderer.render_tool_result(tool_id, error_message, status="error")
        else:
            self.add_pipeline_message(
                f"❌ Fehler in Schritt: {step.name}", "error", step.step_id
            )
            self.add_pipeline_message(error_message, "error", step.step_id)

    @pyqtSlot(object)
    def on_pipeline_completed(self, analysis_state):
        total_duration = "unbekannt"
        if hasattr(self, "pipeline_start_time"):
            total_seconds = (
                datetime.now() - self.pipeline_start_time
            ).total_seconds()
            total_duration = f"{total_seconds:.1f}s"
        self.add_pipeline_message(
            f"\U0001f389 Pipeline vollständig abgeschlossen in {total_duration}!",
            "success",
        )

        if (
            analysis_state
            and hasattr(analysis_state, "final_llm_analysis")
            and analysis_state.final_llm_analysis
        ):
            kw_list = analysis_state.final_llm_analysis.extracted_gnd_keywords or []
            if kw_list:
                kw_display = ", ".join(kw_list)
                self.add_pipeline_message(
                    f"\U0001f4cc {len(kw_list)} GND-Schlagworte ausgewählt:\n{kw_display}",
                    "success",
                )
            response_text = (
                analysis_state.final_llm_analysis.response_full_text or ""
            )
            if (
                "Schlagwortketten" in response_text
                or "schlagwortketten" in response_text.lower()
            ):
                chain_lines = [
                    line
                    for line in response_text.split("\n")
                    if "→" in line or "->" in line
                ]
                if chain_lines:
                    self.add_pipeline_message(
                        "\U0001f517 Schlagwortketten:\n" + "\n".join(chain_lines[:10]),
                        "success",
                    )

        if analysis_state and getattr(analysis_state, "dk_classifications", None):
            # WP12: the rich colour-coded card (confidence + per-code titles) is
            # built by the shared formatter so the GUI and webapp render the
            # identical chrome from one source.
            card_html, dk_codes_text = (
                PipelineResultFormatter.format_dk_classifications_card_html(analysis_state)
            )
            if card_html:
                self.add_pipeline_message("\U0001f3f7 DK-Klassifikationen:", "success")
                self._renderer.render_html_block(
                    card_html, kind="dk_classifications", plain_text=dk_codes_text
                )

        if (
            analysis_state
            and hasattr(analysis_state, "rvk_provenance")
            and analysis_state.rvk_provenance
        ):
            rvk_count = len(analysis_state.rvk_provenance)
            if rvk_count:
                self.add_pipeline_message(
                    f"\U0001f4d6 {rvk_count} RVK-Klassifikationen zugeordnet",
                    "success",
                )

        # Auto-load chat context for the just-finished pipeline.
        try:
            self.load_context(analysis_state)
        except Exception:
            self.logger.exception("PipelineChatPanel: load_context after pipeline failed")

    @pyqtSlot(str)
    def on_llm_token_received(self, token: str):
        if self.current_step_id:
            self.add_streaming_token(token, self.current_step_id)

    def start_llm_streaming(self, step_id: str):
        self.start_streaming_line(step_id, "LLM Antwort: ")

    def end_llm_streaming(self):
        self.end_streaming_line()

    def clear_stream(self):
        self._renderer.clear()
        self.add_pipeline_message("Stream geleert", "info")

    def save_stream_log(self):
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        if self.current_working_title:
            default_filename = f"{self.current_working_title}_log.txt"
        else:
            default_filename = (
                f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

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
                    # WebLogView has no toPlainText(); reconstruct the log from
                    # the renderer's message history (completed messages).
                    plain_text = "\n".join(
                        entry.content for entry in self._renderer.history
                    )
                    f.write(f"ALIMA Pipeline Log - {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(plain_text)
                self.add_pipeline_message(
                    f"Log gespeichert: {filename}", "success"
                )
            except Exception as e:
                self.add_pipeline_message(f"Fehler beim Speichern: {e}", "error")

    def set_working_title(self, working_title: str):
        self.current_working_title = working_title
        self.logger.info(
            f"PipelineChatPanel: working_title set to '{working_title}'"
        )

    def refresh_styles(self):
        if hasattr(self, "stream_text"):
            from .styles import get_font_size
            self.stream_text.set_font_pt(get_font_size())

    def reset_for_new_pipeline(self):
        self.current_step_id = None
        self.step_start_times.clear()
        self.current_working_title = None
        self.clear_stream()
        self.hide_repetition_warning()
        self._bus_tool_call_ids.clear()
        self._prompt_blocks.clear()
        self._prompt_meta.clear()
        self._last_tool_call_id = None
        if self.reset_toggle.isChecked():
            self.session.reset()
            self.current_context = ""
            self.working_title = ""
            self._renderer.clear()

    # ==================================================================
    # Chat-side rendering & lifecycle (ported from ChatWidget)
    # ==================================================================

    # -- Typing indicator ------------------------------------------------

    def _show_typing(self, model_label: str) -> None:
        self._typing_model = model_label or "…"
        self._typing_dots = 0
        self._tick_typing()
        self.typing_label.setVisible(True)
        self._typing_timer.start()

    def _tick_typing(self) -> None:
        self._typing_dots = (self._typing_dots % 3) + 1
        dots = "●" * self._typing_dots + "○" * (3 - self._typing_dots)
        self.typing_label.setText(f"🤖 {self._typing_model}  {dots}")

    def _hide_typing(self) -> None:
        self._typing_timer.stop()
        self.typing_label.setVisible(False)
        self.typing_label.setText("")

    # -- Model resolution & combo persistence ----------------------------

    def _refresh_model_status(self) -> None:
        try:
            provider, model = self._resolve_provider_model()
        except Exception:
            provider, model = "", ""
        if provider and model:
            self.model_status_label.setText(f"→ {provider} | {model}")
        else:
            self.model_status_label.setText("→ (kein Modell)")

    @pyqtSlot(int)
    def _on_model_combo_changed(self, _idx: int) -> None:
        self._refresh_model_status()
        if self.persist_combo_toggle.isChecked():
            self._persist_combo_to_chat_config()

    @pyqtSlot(int)
    def _on_autonomous_toggle_changed(self, _state: int) -> None:
        """Persist autonomous-mode toggle to ChatConfig + chat-side state."""
        new_value = self.autonomous_toggle.isChecked()
        try:
            from ..utils.config_manager import ConfigManager
            cm = ConfigManager()
            # chat_config lives on AlimaConfig (load_config), not the unified
            # config. Mutate the SAME object we then save, or the change is lost.
            full = cm.load_config()
            chat_cfg = getattr(full, "chat_config", None)
            if chat_cfg is None:
                return
            chat_cfg.autonomous_pipeline = new_value
            cm.save_config(full, preserve_unified=True)
            msg = "aktiv" if new_value else "aus"
            self._append_system_message(f"🤖 Autonom-Modus: {msg}")
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: persist autonomous_pipeline failed"
            )

    def _persist_combo_to_chat_config(self) -> None:
        data = self.model_combo.currentData()
        if not data:
            return
        try:
            provider, model = data.split("|", 1)
        except ValueError:
            return
        try:
            from ..utils.config_manager import ConfigManager

            cm = ConfigManager()
            # Mutate the AlimaConfig.chat_config we actually persist (the unified
            # config has no chat_config — reading it dropped the save silently).
            full = cm.load_config()
            chat_cfg = getattr(full, "chat_config", None)
            if chat_cfg is None:
                return
            chat_cfg.default_provider = provider
            chat_cfg.default_model = model
            cm.save_config(full, preserve_unified=True)
            self._append_system_message(
                f"💾 Chat-Default gespeichert: {provider} | {model}"
            )
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: persist default model failed"
            )

    def _populate_model_combo(self):
        try:
            self.model_combo.clear()
            self.model_combo.addItem("-- Auto --", None)
            from ..utils.config_manager import ConfigManager

            config_manager = ConfigManager()
            unified_config = config_manager.get_unified_config()
            for provider in unified_config.get_enabled_providers():
                models = getattr(provider, "available_models", []) or []
                if not models and getattr(provider, "preferred_model", None):
                    models = [provider.preferred_model]
                # Sort models alphabetically by model name (case-insensitive)
                models = sorted(models, key=lambda s: s.lower())
                for model in models:
                    self.model_combo.addItem(
                        f"{provider.name} | {model}",
                        f"{provider.name}|{model}",
                    )
        except Exception as e:
            self.logger.error(f"Error populating model combo: {e}")

    def _resolve_provider_model(self) -> tuple[str, str]:
        # Shared chain (CLI/HTTP/GUI): combo override → ChatConfig default →
        # pipeline global override → pipeline step default (carries the
        # first-enabled-provider fallback) → llm_service.current.
        ov_provider = ov_model = None
        override_data = self.model_combo.currentData()
        if override_data:
            try:
                ov_provider, ov_model = override_data.split("|", 1)
            except ValueError:
                ov_provider = ov_model = None
        provider, model = resolve_provider_model(
            ov_provider, ov_model,
            chat_config=self._get_chat_config(),
            pipeline_manager=self.pipeline_manager,
            llm_service=self.llm_service,
        )
        if provider and model:
            return provider, model
        # Last resort: first initialized LLM client + its first listed model.
        try:
            clients = getattr(self.llm_service, "clients", {}) if self.llm_service else {}
            if clients:
                provider = list(clients.keys())[0]
                sp = getattr(self.llm_service, "supported_providers", {})
                models = sp.get(provider, {}).get("models", [])
                if models:
                    return provider, models[0]
        except Exception:
            pass
        return "", ""

    def _get_chat_config(self):
        try:
            from ..utils.config_manager import ConfigManager

            # chat_config is a field of AlimaConfig (load_config), not the
            # UnifiedProviderConfig — reading the latter always yielded None and
            # silently dropped saved chat defaults + autonomous flag.
            chat_cfg = getattr(ConfigManager().load_config(), "chat_config", None)
            if chat_cfg is not None:
                return chat_cfg
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: ChatConfig lookup failed"
            )
        from ..utils.config_models import ChatConfig

        return ChatConfig()

    # -- Context loading -------------------------------------------------

    def load_context(self, analysis_state) -> None:
        if self.reset_toggle.isChecked():
            # Don't wipe the pipeline log; only reset chat session state.
            self.session.reset()
            self.current_context = ""
            self.working_title = ""
            self._renderer._assistant_block_open = False
            self._renderer._assistant_cell_cursor = None

        self.session.reset()

        if analysis_state is None:
            self.current_context = ""
            self.session.last_shared_context = None
            self._append_system_message(
                "ℹ️ Kein Pipeline-Kontext geladen. Chat funktioniert trotzdem — "
                "stelle einfach eine Frage."
            )
            return

        parts = []
        if hasattr(analysis_state, "working_title") and analysis_state.working_title:
            self.working_title = analysis_state.working_title
            parts.append(f"Titel: {analysis_state.working_title}")
        if (
            hasattr(analysis_state, "original_abstract")
            and analysis_state.original_abstract
        ):
            abstract = analysis_state.original_abstract
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            parts.append(f"Abstract: {abstract}")
        self.current_context = "\n".join(parts) or "(kein Titel/Abstract)"

        ctx = self._shared_context_from_analysis_state(analysis_state)
        self.session.last_shared_context = ctx

        kw_count = len(getattr(ctx, "extracted_keywords", []) or []) if ctx else 0
        if not kw_count and ctx:
            kw_count = len(getattr(ctx, "initial_keywords", []) or [])
        dk_count = len(getattr(ctx, "dk_classifications", []) or []) if ctx else 0
        self._append_system_message(
            f"✅ Kontext geladen: {self.working_title or 'Unbenannt'}"
            f" ({kw_count} Keywords, {dk_count} DK-Codes, Tools aktiv)"
        )

    @staticmethod
    def _shared_context_from_analysis_state(state) -> Optional[object]:
        # P-ζ: bridge promoted to SharedContext.from_keyword_analysis_state.
        # Wrapper retained one cycle for any external caller; remove afterwards.
        try:
            from src.core.agents.shared_context import SharedContext
            return SharedContext.from_keyword_analysis_state(state)
        except Exception:
            return None

    # -- Send / cancel / worker callbacks --------------------------------

    def send_message(self):
        text = self.input_field.toPlainText().strip()
        if not text:
            return
        if self.current_worker and self.current_worker.isRunning():
            return

        self._append_user_message(text)
        self.input_field.clear()

        # Mode-aware prompt assembly
        mode = detect_mode(text, self.current_context)
        effective_system_prompt = self.system_prompt or build_system_prompt(mode=mode)
        effective_user_template = get_user_prompt_template(mode)
        user_prompt = effective_user_template.format(
            context=self.current_context or "(kein Werk geladen)",
            user_message=text,
        )

        provider, model = self._resolve_provider_model()
        if not provider or not model:
            self._append_system_message(
                "⚠️ Kein LLM-Provider konfiguriert. Bitte in Pipeline-Einstellungen "
                "ein Modell wählen."
            )
            return

        self._refresh_shared_context()
        chat_config = self._get_chat_config()
        kb_manager = None
        if self.pipeline_manager is not None:
            kb_manager = (
                getattr(self.pipeline_manager, "unified_knowledge_manager", None)
                or getattr(self.pipeline_manager, "knowledge_manager", None)
            )
        try:
            tool_registry = build_chat_toolset(
                session=self.session,
                chat_config=chat_config,
                mcp_registry=self.mcp_registry,
                pipeline_manager=self.pipeline_manager,
                kb_manager=kb_manager,
                proposal_gateway=self.proposal_gateway,
            )
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: build_chat_toolset failed"
            )
            self._append_system_message(
                "❌ Tool-Setup fehlgeschlagen — siehe Log."
            )
            return

        self._set_ui_running(True)
        self.current_worker = ChatAgentWorker(
            llm_service=self.llm_service,
            tool_registry=tool_registry,
            system_prompt=effective_system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            temperature=getattr(chat_config, "temperature", 0.5),
            max_iterations=30,
            timeout_seconds=getattr(chat_config, "timeout_seconds", 600),
            history=list(self.session.messages[-6:]),
        )
        self.current_worker.token_received.connect(self._on_token)
        self.current_worker.status_message.connect(self._on_status_message)
        # NB (Phase D): tool calls ride AlimaStateBus, not direct signals.
        self.current_worker.generation_finished.connect(self._on_finished)
        self.current_worker.generation_error.connect(self._on_error)
        self._current_render_model = f"{provider} | {model}"
        self._refresh_model_status()
        self._show_typing(self._current_render_model)
        self.current_worker.start()

        self.message_sent.emit(text)

    def cancel_generation(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.request_stop()
            self._append_system_message("⏹ Generation abgebrochen.")
            self._set_ui_running(False)

    @pyqtSlot(str)
    def _on_token(self, token: str):
        if not self._renderer._assistant_block_open:
            self._hide_typing()
            self._open_assistant_message(self._current_render_model)
            self._renderer._assistant_block_open = True
        self._append_assistant_token(token)

    @pyqtSlot(str)
    def _on_status_message(self, line: str):
        text = (line or "").strip()
        if not text:
            return
        # While a bus pipeline-step block is open, accumulate status lines
        # into the block body instead of emitting separate messages.
        if self._pipeline_step_open:
            self._open_step_status.append(text)
            return
        # No open step → dim log line (chat-agent status, LLM progress, etc.).
        self._renderer.render_pipeline_log(text, "debug")

    @pyqtSlot(object)
    def _on_finished(self, result):
        try:
            # Merge full conversation (user + tool calls + tool results + assistant)
            for msg in getattr(result, "messages", []) or []:
                self.session.messages.append(dict(msg))
            final = getattr(result, "content", "") or ""
            if final and not self._renderer._assistant_block_open:
                self._hide_typing()
                self._open_assistant_message(self._current_render_model)
                self._renderer._assistant_block_open = True
                self._append_assistant_token(final)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: failed to append assistant turn"
            )
        self._hide_typing()
        self._finalize_assistant_message()
        self._set_ui_running(False)

    @pyqtSlot(str)
    def _on_error(self, error: str):
        self._hide_typing()
        self._append_system_message(f"❌ Fehler: {error}")
        self._renderer._assistant_block_open = False
        self._renderer._assistant_cell_cursor = None
        self._set_ui_running(False)
        self.logger.error(f"PipelineChatPanel: generation error: {error}")

    # -- Bus subscriptions for agentic-pipeline tool events --------------

    def _on_state_changed(self, _diff: dict) -> None:
        self._refresh_shared_context()

    def _on_bus_tool_called(self, payload: dict) -> None:
        try:
            name = payload.get("name", "") or "tool"
            args = payload.get("arguments", {}) or {}
            tool_id = self._renderer.render_tool_call(name, args)
            bus_id = payload.get("id")
            if bus_id:
                self._bus_tool_call_ids[bus_id] = tool_id
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus tool.called rendering failed"
            )

    def _on_bus_pipeline_prompt(self, payload: dict) -> None:
        """Render the agentic step input (SYSTEM+USER prompt) as a collapsed,
        timestamped block instead of inline streamed text. - Claude Generated"""
        try:
            prompt_id = str(payload.get("prompt_id", "") or "")
            step_id = payload.get("step_id", "") or "?"
            kind = payload.get("kind", "input") or "input"
            system = payload.get("system", "") or ""
            user = payload.get("user", "") or ""
            ts = str(payload.get("timestamp", "") or "")
            ts_hms = ts.split("T")[-1] if "T" in ts else ts
            provider = payload.get("provider", "") or ""
            model = payload.get("model", "") or ""

            meta_parts = []
            if ts_hms:
                meta_parts.append(ts_hms)
            pm = "/".join(p for p in (provider, model) if p)
            if pm:
                meta_parts.append(pm)
            if hasattr(self, "pipeline_start_time"):
                elapsed = (datetime.now() - self.pipeline_start_time).total_seconds()
                meta_parts.append(f"(+{elapsed:.1f}s)")
            meta = "  ".join(meta_parts)

            if kind == "reflection":
                icon, title = "🔍", f"Reflexion '{step_id}'"
            else:
                icon, title = "📥", f"Input '{step_id}'"

            body = f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}"
            # Close any open streaming line so the collapsible sits on its own.
            if self._renderer._is_streaming:
                self._renderer.end_streaming_line()
            tool_id = self._renderer.render_collapsible(
                title, body, collapsed=True, icon=icon, meta=meta,
            )
            if prompt_id:
                self._prompt_blocks[prompt_id] = tool_id
                self._prompt_meta[prompt_id] = meta
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: _on_bus_pipeline_prompt failed"
            )

    def _on_bus_pipeline_prompt_done(self, payload: dict) -> None:
        """Append the LLM-call duration (⏱ Ys) to a rendered input block. - Claude Generated"""
        try:
            prompt_id = str(payload.get("prompt_id", "") or "")
            dur = payload.get("duration_s")
            tool_id = self._prompt_blocks.get(prompt_id)
            if tool_id is None or dur is None:
                return
            base = self._prompt_meta.get(prompt_id, "")
            meta = f"{base}  ⏱ {float(dur):.1f}s" if base else f"⏱ {float(dur):.1f}s"
            self._renderer.update_collapsible_meta(tool_id, meta)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: _on_bus_pipeline_prompt_done failed"
            )

    @staticmethod
    def _extract_urls_from_json(data) -> "set[str]":
        """Recursively collect all http(s) URL strings from a parsed JSON value."""
        urls: "set[str]" = set()
        if isinstance(data, str):
            if data.startswith(("http://", "https://")):
                urls.add(data)
        elif isinstance(data, dict):
            for v in data.values():
                urls |= PipelineChatPanel._extract_urls_from_json(v)
        elif isinstance(data, (list, tuple)):
            for item in data:
                urls |= PipelineChatPanel._extract_urls_from_json(item)
        return urls

    def _on_bus_tool_result(self, payload: dict) -> None:
        try:
            result = payload.get("result", "") or ""
            # Register any URLs from tool results so the renderer won't
            # flag them as external (e.g. DOIs from finc records). - Claude Generated
            try:
                import json as _json
                self._renderer.add_trusted_urls(
                    self._extract_urls_from_json(_json.loads(result))
                )
            except Exception:
                pass
            bus_id = payload.get("id")
            tool_id = self._bus_tool_call_ids.pop(bus_id, None) if bus_id else None
            if tool_id:
                # P-B: cache-hit branch appends a 📦 badge to the result
                # text so users can distinguish fast cache returns from
                # live calls. Renderer forwards the result verbatim.
                if payload.get("cache_hit"):
                    preview = result.strip().replace("\n", " ")
                    if len(preview) > 80:
                        preview = preview[:80] + "…"
                    result = f"📦 cache: {preview}"
                # Phase E: bus producers emit "ok" (CachingToolRegistry) but
                # the renderer's internal status enum is "success" / "error".
                # Normalize here so the ✓/✗ icon picks the right glyph.
                raw_status = payload.get("status") or "success"
                status = "success" if raw_status == "ok" else raw_status
                self._renderer.render_tool_result(
                    tool_id, result, status=status,
                )
            else:
                # Fallback: orphan result (bus id unknown — late or dropped call).
                preview = result.strip().replace("\n", " ")
                if len(preview) > 120:
                    preview = preview[:120] + "…"
                self._append_system_message(f"↳ {preview}")
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus tool.result rendering failed"
            )

    def _on_bus_pipeline_completed(self, payload: dict) -> None:
        """Render a system message when ``state.pipeline_completed`` fires.

        Phase F: the legacy ``pipeline_completed_callback`` still runs
        elsewhere, but the bus event was previously dead (no
        subscriber). Now the panel reacts to it so the chat log
        acknowledges the end of a pipeline run consistently.
        """
        try:
            label = "✅ Pipeline abgeschlossen"
            workflow = (payload or {}).get("workflow")
            if workflow:
                label += f" ({workflow})"
            self._renderer.render_system_message(label)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus state.pipeline_completed render failed"
            )

    def _on_bus_pipeline_step(self, payload: dict) -> None:
        """Render step-progress as collapsible tool-call block (was: legacy marker).

        Same visual treatment as real ``tool.called``/``tool.result`` events so
        that steps triggered via ``run_pipeline`` / ``rerun_step`` look
        identical to native tool calls.
        """
        try:
            status = payload.get("status", "")
            step_id = payload.get("step_id", "") or "?"
            name = payload.get("name", "") or step_id
            tool_name = f"pipeline.{step_id}"
            args = {
                "step": step_id,
                "name": name,
                "tool": payload.get("tool", ""),
            }
            if status == "running":
                self._last_tool_call_id = self._renderer.render_tool_call(
                    tool_name, args
                )
                self._open_step_status = []
                self._pipeline_step_open = True
                return
            # Terminal: success when explicitly "completed", error otherwise.
            result_status = "success" if status == "completed" else "error"
            # Prefer the accumulated status lines (per-keyword search
            # progress, etc.) as the block's body when present. Fall
            # back to a short summary line if the step emitted no
            # status messages.
            # Post-(F) polish: if no status lines accumulated, fall back
            # to a short summary line. Use ``getattr`` so the handler
            # is robust against an older stub that doesn't define the
            # accumulator attribute yet.
            if getattr(self, "_open_step_status", None):
                result_text = "\n".join(self._open_step_status)
            else:
                result_text = f"{status or 'done'}: {name}"
            self._open_step_status = []
            self._pipeline_step_open = False
            if self._last_tool_call_id:
                self._renderer.render_tool_result(
                    self._last_tool_call_id, result_text, status=result_status
                )
                self._last_tool_call_id = None
            else:
                # Orphan completion (no prior running) → still render a block.
                tid = self._renderer.render_tool_call(tool_name, args)
                self._renderer.render_tool_result(
                    tid, result_text, status=result_status
                )
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: bus state.pipeline_step rendering failed"
            )

    def _refresh_shared_context(self) -> None:
        try:
            if self.pipeline_manager is None:
                return
            ctx = getattr(self.pipeline_manager, "last_shared_context", None)
            if ctx is not None:
                self.session.last_shared_context = ctx
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: shared-context refresh failed"
            )

    # -- UI helpers ------------------------------------------------------

    def _set_ui_running(self, running: bool):
        self.send_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.input_field.setEnabled(not running)
        if running:
            self.input_field.setPlaceholderText("Antwort wird generiert...")
        else:
            self.input_field.setPlaceholderText(
                "Frage zu den Pipeline-Ergebnissen stellen..."
            )
            self.input_field.setFocus()

    # -- Chat bubble & marker rendering (delegated to UnifiedMessageRenderer) --

    def _append_user_message(self, text: str):
        self._renderer.render_user_bubble(text)

    def _append_tool_marker(self, text: str):
        """Backwards-compat shim. Prefer ``renderer.render_tool_call`` for new code."""
        self._renderer.render_tool_marker(text)

    def _append_system_message(self, text: str):
        self._renderer.render_system_message(text)

    def _open_assistant_message(self, model_label: str):
        self._renderer.open_assistant_bubble(model_label)

    def _append_assistant_token(self, token: str):
        self._renderer.append_assistant_token(token)

    def _finalize_assistant_message(self):
        self._renderer.finalize_assistant_bubble()

    def _append_html(self, html: str):
        self._renderer.append_raw_html(html)

    # -- P-ε: mutation-proposal inline bubble --------------------------

    @pyqtSlot(int, str, dict)
    def _render_proposal_bubble(
        self, audit_id: int, tool_name: str, payload: dict
    ) -> None:
        """Render a clickable confirmation bubble for a mutation proposal."""
        self._renderer.render_proposal_bubble(audit_id, tool_name, payload)

    @pyqtSlot(QUrl)
    def _on_anchor_clicked(self, url: QUrl) -> None:
        """Route ``mutation://``, ``tool://`` and external ``http(s)://`` link clicks."""
        scheme = url.scheme()
        if scheme == "mutation":
            self._handle_mutation_link(url)
        elif scheme == "tool":
            self._handle_tool_link(url)
        elif scheme in ("http", "https"):
            # P-δ.5: external catalog/web links from <<CAT:rsn|…>> markers.
            # setOpenExternalLinks(False) is set on the text browser, so we
            # have to drive the open ourselves via QDesktopServices.
            QDesktopServices.openUrl(url)

    def _handle_mutation_link(self, url: QUrl) -> None:
        host_part = url.host()
        path_part = url.path().lstrip("/")
        try:
            audit_id = int(host_part)
        except (TypeError, ValueError):
            return
        action = path_part.lower()
        if action not in ("accept", "reject"):
            return
        accepted = action == "accept"
        try:
            self.proposal_gateway.resolve_decision(audit_id, accepted)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: resolve_decision failed"
            )
            return
        status = "✓ Akzeptiert" if accepted else "✗ Abgelehnt"
        color = "#50fa7b" if accepted else "#ff5555"
        self._append_html(
            f'<div style="margin: 2px 24px; color: {color}; font-size: 9pt;">'
            f'{status} (#audit_{audit_id})</div>'
        )

    def _handle_tool_link(self, url: QUrl) -> None:
        """Toggle collapsible tool-call block.

        URL format: ``tool://toggle/<tool_id>``.
        Host = "toggle", Path = "<tool_id>".
        """
        action = url.host()
        tool_id = url.path().lstrip("/")
        if action != "toggle" or not tool_id:
            return
        try:
            self._renderer.toggle_tool_call(tool_id)
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: toggle_tool_call failed"
            )

    @staticmethod
    def _escape_html(text: str) -> str:
        from .unified_message_renderer import UnifiedMessageRenderer
        return UnifiedMessageRenderer._escape_html(text)

    @staticmethod
    def _format_tool_args(args: dict) -> str:
        from .unified_message_renderer import UnifiedMessageRenderer
        return UnifiedMessageRenderer._format_tool_args(args)

    # -- System-prompt dialog --------------------------------------------

    def show_system_prompt_dialog(self):
        dialog = SystemPromptDialog(self.system_prompt, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_prompt = dialog.get_prompt()
            if new_prompt:
                self.system_prompt = new_prompt
                self._append_system_message("✅ System-Prompt aktualisiert.")
