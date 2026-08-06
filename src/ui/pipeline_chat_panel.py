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
from datetime import datetime
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
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
)
from ..core.state_bus import AlimaStateBus
from ._chat_panel_bus import BusEventMixin
from ._chat_panel_chat_agent import ChatAgentMixin
from ._chat_panel_pipeline_log import PipelineLogMixin
from .chat_agent_worker import ChatAgentWorker
from .chat_input_widgets import ChatInputEdit, SystemPromptDialog
from .chat_session import ChatSession
from .repetition_warning_bar import RepetitionWarningBar
from .styles import (
    get_button_styles,
    get_main_stylesheet,
    get_scaled_font,
)


# ----------------------------------------------------------------------
# PipelineChatPanel
# ----------------------------------------------------------------------
# SystemPromptDialog + ChatInputEdit now live in chat_input_widgets.py
# (F-5 god-file split); imported above. - Claude Generated


class PipelineChatPanel(PipelineLogMixin, ChatAgentMixin, BusEventMixin, QWidget):
    """Unified pipeline-log + chat panel (WP10 P-δ.5a).

    Behavior is split across mixins (F-5 god-file split):
    - ``PipelineLogMixin`` (_chat_panel_pipeline_log.py) — pipeline-side log.
    - ``ChatAgentMixin`` (_chat_panel_chat_agent.py) — chat send/cancel/worker,
      model resolution, typing indicator, context loading, running/stopping UI.
    - ``BusEventMixin`` (_chat_panel_bus.py) — AlimaStateBus event handlers.
    This class keeps construction (``__init__`` incl. the ``bus.subscribe``
    wiring), ``setup_ui``, the repetition-bar delegators, and the render
    delegators / link handlers below. - Claude Generated
    """

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
        # Agentic input/prompt collapsible blocks: prompt_id → (tool_id, base_meta)
        self._prompt_blocks: Dict[str, str] = {}
        self._prompt_meta: Dict[str, str] = {}

        # Chat-side state
        self.system_prompt: str = self.DEFAULT_SYSTEM_PROMPT
        self.session: ChatSession = ChatSession()
        self.current_worker: Optional[ChatAgentWorker] = None
        # True between a cancel request and the worker actually finishing, so the
        # UI stays locked (no premature 'ready' / silently-dropped sends). - Claude Generated
        self._stopping: bool = False
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
        # (markers are reduced to plain text) if config is missing. Shared with
        # the webapp via configure_catalog (Claude Generated).
        try:
            from src.core.search.factory import catalog_web_bases
            self._renderer.configure_catalog(*catalog_web_bases())
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
            bus.subscribe("state.notice", self._on_bus_notice)
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

        # Shared provider+model picker (Phase 5): replaces the single combined
        # "provider | model" combo. Model lists come from the shared TTL cache and
        # refresh per provider; the clean model name lives in UserRole so display
        # decoration can never leak into the value (the old phantom-model bug).
        from .provider_model_selector import ProviderModelSelector
        # Non-editable: a read-only model combo renders its text natively via the
        # QComboBox color (no embedded QLineEdit that ignores the dark theme), and
        # the model list is anyway fully covered by detection — free-text entry is
        # not needed in the chat header.
        self.provider_selector = ProviderModelSelector(
            allow_empty=True,
            editable_model=False,
            empty_provider_label="-- Auto --",
            empty_model_label="(Auto)",
        )
        _combo_qss = (
            "QComboBox { font-size: 10px; padding: 2px 6px; border: 1px solid #555; "
            "border-radius: 3px; background-color: #3d3d3d; color: #ccc; }"
            # Make the drop-down affordance visible: the custom dark theme replaces
            # the native rendering, so style the button area + draw a light arrow.
            "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; "
            "width: 18px; border-left: 1px solid #555; background-color: #4a4a4a; "
            "border-top-right-radius: 3px; border-bottom-right-radius: 3px; }"
            "QComboBox::down-arrow { width: 0; height: 0; border-left: 4px solid transparent; "
            "border-right: 4px solid transparent; border-top: 5px solid #ccc; }"
            "QComboBox QAbstractItemView { background-color: #2b2b2b; color: #ccc; "
            "selection-background-color: #005fcc; selection-color: white; border: 1px solid #555; }"
        )
        # set_combo_style (not setStyleSheet directly): the selector's validation
        # pass composes onto this base, so the dark theme survives model loading.
        self.provider_selector.set_combo_style(_combo_qss)
        # Fixed widths: a long model name must not resize the header. The closed
        # combo clips; the popup view gets a generous minimum so the full names
        # stay readable when the dropdown is open.
        self.provider_selector.provider_combo.setFixedWidth(130)
        self.provider_selector.model_combo.setFixedWidth(210)
        self.provider_selector.model_combo.view().setMinimumWidth(320)
        self._populate_model_combo()
        self.provider_selector.selectionChanged.connect(self._on_model_selection_changed)
        header_layout.addWidget(self.provider_selector)

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

        # Antwortsprache umschalten (Deutsch/Englisch) - Claude Generated
        self.chat_language = "de"
        self.language_btn = QPushButton("DE")
        self.language_btn.setFixedSize(30, 22)
        self.language_btn.setStyleSheet(icon_btn_style)
        self.language_btn.setToolTip("Antwortsprache umschalten (Deutsch/Englisch)")
        self.language_btn.clicked.connect(self._toggle_chat_language)
        header_layout.addWidget(self.language_btn)

        # Thinking/Reasoning des Chat-Agenten überschreiben - Claude Generated
        self.chat_think_combo = QComboBox()
        self.chat_think_combo.addItems(["🧠 Auto", "🧠 An", "🧠 Aus"])
        self.chat_think_combo.setStyleSheet(
            "QComboBox { color: #ccc; font-size: 10px; padding: 2px 6px; "
            "border: 1px solid #555; border-radius: 3px; }"
        )
        self.chat_think_combo.setToolTip(
            "Thinking/Reasoning des Chat-Agenten.\n"
            "Auto = Modell-Default · An = think=true · Aus = think=false"
        )
        header_layout.addWidget(self.chat_think_combo)

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

        # --- Transient status strip (Chat-UX 7/9) -----------------------
        # One-line, overwrite-in-place home for chat-config/status echoes
        # (🧭 Modus, 💾 Default gespeichert, 🤖 Autonom, ✅ Kontext) that used
        # to spam the conversation stream as system messages.
        self.status_strip = QLabel("")
        self.status_strip.setVisible(False)
        self.status_strip.setStyleSheet(
            "QLabel { background-color: #262626; color: #9aa5b1;"
            " font-size: 8pt; padding: 2px 8px;"
            " border-bottom: 1px solid #444; }"
        )
        outer.addWidget(self.status_strip)

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

        # --- Repetition-warning bar (pipeline-only feature) ---
        # Extracted to repetition_warning_bar.py (F-5). The bar owns its state
        # machine + signals; the panel re-wires them. - Claude Generated
        self.repetition_bar = RepetitionWarningBar()
        self.repetition_bar.abort_requested.connect(self._on_repetition_abort)
        self.repetition_bar.retry_requested.connect(self._on_repetition_retry)
        log_layout.addWidget(self.repetition_bar)

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
        from ..utils.i18n import t
        self.input_field.setPlaceholderText(t("chat.input.placeholder"))
        # Right padding reserves room for the floating send button so text
        # never flows underneath it. Claude Generated.
        self.input_field.setStyleSheet(
            "QTextEdit { background-color: #3d3d3d; color: #e0e0e0; "
            "border: 1px solid #555; border-radius: 10px; "
            "padding: 8px 52px 8px 12px; font-size: 11pt; }"
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

        # Round send button floating in the field's bottom-right corner.
        self.send_btn = QPushButton("➤", self.input_field)
        self.send_btn.setToolTip("Senden (Enter)")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedSize(36, 36)
        self.send_btn.setDefault(True)
        self.send_btn.setStyleSheet(
            "QPushButton { background-color: #8be9fd; color: #1e1e1e; "
            "border: none; border-radius: 18px; font-size: 15pt; "
            "font-weight: bold; padding-bottom: 2px; }"
            "QPushButton:hover { background-color: #a4f0ff; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.send_btn.clicked.connect(self.send_message)
        self.input_field.set_overlay_button(self.send_btn, margin=8)

        self.body_splitter.addWidget(input_frame)

        # Default ratio: log gets ~80%, input ~20% — both grow with window.
        self.body_splitter.setStretchFactor(0, 8)
        self.body_splitter.setStretchFactor(1, 2)
        self.body_splitter.setSizes([800, 200])

        outer.addWidget(self.body_splitter, stretch=1)

        # Initial model-status paint.
        self._refresh_model_status()

    # ==================================================================
    # Repetition-warning bar — extracted to repetition_warning_bar.py (F-5).
    # The panel keeps thin delegators (public API used by pipeline_tab.py) and
    # re-wires the bar's signals to its own. - Claude Generated
    # ==================================================================

    def show_repetition_warning(
        self,
        detection_type: str,
        details: str,
        suggestions: List[Dict],
        grace_period: bool = False,
        grace_seconds: float = 2.0,
    ):
        self.repetition_bar.show_warning(
            detection_type, details, suggestions, grace_period, grace_seconds
        )

    def hide_repetition_warning(self, resolved: bool = False):
        self.repetition_bar.hide_warning(resolved)

    def _on_repetition_abort(self):
        self.abort_generation_requested.emit()

    def _on_repetition_retry(self, params: Dict):
        self.retry_with_variations.emit(params)
        self.add_pipeline_message(
            f"🔄 Retry mit Parametern: {params}",
            "info",
            self.current_step_id,
        )

    # ==================================================================
    # Chat-side rendering & lifecycle → ChatAgentMixin
    # (_chat_panel_chat_agent.py); bus handlers → BusEventMixin
    # (_chat_panel_bus.py). Render delegators + link handlers remain below.
    # - Claude Generated
    # ==================================================================

    # -- Chat bubble & marker rendering (delegated to UnifiedMessageRenderer) --

    def _toggle_chat_language(self):
        """Toggle the chat reply language DE↔EN - Claude Generated"""
        self.chat_language = "en" if self.chat_language == "de" else "de"
        self.language_btn.setText(self.chat_language.upper())

    def _get_chat_think_override(self):
        """Read the chat thinking combo → None/True/False - Claude Generated"""
        if not hasattr(self, "chat_think_combo"):
            return None
        return {0: None, 1: True, 2: False}.get(self.chat_think_combo.currentIndex())

    def _append_user_message(self, text: str):
        self._renderer.render_user_bubble(text)

    def set_status_strip(self, text: str) -> None:
        """Transient status line under the header — kept out of the
        conversation stream (Chat-UX 7/9). Empty text hides the strip."""
        if not hasattr(self, "status_strip"):
            return
        self.status_strip.setText(text)
        self.status_strip.setVisible(bool(text))

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
                from ..utils.i18n import t as _t
                self._append_system_message(_t("chat.system_prompt_updated"))
