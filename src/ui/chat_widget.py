"""Chat Widget — Conversational interface for agentic pipeline results - Claude Generated.

Dock-Widget für MainWindow. Nimmt KeywordAnalysisState als Kontext,
chatet mit LLM über Pipeline-Ergebnisse. Prompts sind Teil des Widgets
(nicht prompts.json).
"""

import logging
from typing import Optional, List, Dict

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import (
    QTextCursor,
    QColor,
    QTextBlockFormat,
    QTextTableFormat,
    QTextTableCellFormat,
    QTextLength,
)
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QSplitter,
    QSizePolicy,
    QFrame,
)

from .chat_agent_worker import ChatAgentWorker
from .chat_session import ChatSession
from .chat_tools import build_chat_toolset
from .styles import get_main_stylesheet, get_scaled_font, get_button_styles, LAYOUT
from src.core.state_bus import AlimaStateBus


class SystemPromptDialog(QDialog):
    """Small dialog for editing the system prompt - Claude Generated"""

    def __init__(self, current_prompt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ System-Prompt bearbeiten")
        self.setMinimumSize(500, 300)
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            LAYOUT.get("margin", 12), LAYOUT.get("margin", 12),
            LAYOUT.get("margin", 12), LAYOUT.get("margin", 12),
        )
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
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_prompt(self) -> str:
        return self.editor.toPlainText().strip()


class ChatWidget(QWidget):
    """Chat interface for discussing agentic pipeline results - Claude Generated"""

    # Signals
    message_sent = pyqtSignal(str)  # user_message

    # Prompts are part of the agent definition, not prompts.json - Claude Generated
    DEFAULT_SYSTEM_PROMPT = (
        "Du bist Experte für Bibliothekswissenschaft und Sacherschließung "
        "(RSWK, GND, DDC/DK) und arbeitest als Assistent in der ALIMA-Pipeline. "
        "Antworte präzise, fachlich korrekt, auf Deutsch.\n\n"
        "WICHTIG — Tool-Use-Regeln (zwingend):\n"
        "- Du hast Tools für ALLE Pipeline-Daten: Keywords, Keyword-Ketten,\n"
        "  DK-Klassifikationen, GND-Einträge, fehlende Konzepte.\n"
        "- Rufe IMMER zuerst `list_available_data` auf, um zu sehen welche\n"
        "  Daten vorliegen — außer der Nutzer stellt nur eine Begrüßung\n"
        "  oder eine allgemeine Frage ohne Bezug zum konkreten Werk.\n"
        "- Nenne KEINE GND-ID, KEINEN DK-Code, KEINE Keyword-Anzahl,\n"
        "  KEINE Schlagwortkette aus dem Gedächtnis. Hole sie mit dem\n"
        "  passenden Tool (`get_keywords`, `get_keyword_chains`,\n"
        "  `get_dk_classifications`, `validate_gnd_term`).\n"
        "- Wenn ein Tool 0 Treffer zurückgibt, sage das ehrlich. Erfinde\n"
        "  keine Begriffe als 'GND-Vorschläge'. Markiere eigene Vorschläge\n"
        "  explizit als unverifiziert.\n"
        "- Halluzinationen kosten Vertrauen. Lieber kurz und korrekt als\n"
        "  ausführlich und erfunden."
    )

    USER_PROMPT_TEMPLATE = (
        "Aktuelles Werk: {context}\n\n"
        "Die vollständigen Pipeline-Daten (Keywords, GND-Einträge, "
        "DK-Codes, Schlagwortketten, fehlende Konzepte) hole dir bei "
        "Bedarf via Tool-Calls. Beginne ggf. mit `list_available_data`.\n\n"
        "Nutzer-Frage: {user_message}"
    )

    def __init__(
        self,
        llm_service,
        prompt_service=None,
        pipeline_manager=None,
        mcp_registry=None,
        parent=None,
    ):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.pipeline_manager = pipeline_manager
        self.mcp_registry = mcp_registry

        # State — P-δ.3: replace flat messages list with ChatSession.
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.session: ChatSession = ChatSession()
        self.current_worker: Optional[ChatAgentWorker] = None
        self.current_context: str = ""
        self.working_title: str = ""

        self.setup_ui()

        # P-δ.3: subscribe to state.changed so tool calls / pipeline runs
        # surface new SharedContext snapshots into the chat session.
        try:
            AlimaStateBus().subscribe("state.changed", self._on_state_changed)
        except Exception:
            self.logger.exception("ChatWidget: AlimaStateBus subscription failed")

    def setup_ui(self):
        """Build chat widget UI - Claude Generated"""
        self.setStyleSheet(get_main_stylesheet())
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Header ---
        header = QFrame()
        header.setStyleSheet("QFrame { background-color: #2d2d2d; border-bottom: 1px solid #444; }")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(8)

        title = QLabel("💬 ALIMA Chat")
        title.setFont(get_scaled_font(size_delta=+2, bold=True))
        title.setStyleSheet("color: #e0e0e0;")
        header_layout.addWidget(title)

        # Model selector - Claude Generated
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setMaximumWidth(280)
        self.model_combo.setStyleSheet(
            "QComboBox { font-size: 10px; padding: 2px 6px; border: 1px solid #555; "
            "border-radius: 3px; background-color: #3d3d3d; color: #ccc; }"
        )
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._refresh_model_status)
        header_layout.addWidget(self.model_combo)

        # Live indicator for the *resolved* provider/model (Combo may be
        # "Auto" → show what ChatConfig / fallback actually selects).
        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet(
            "color: #8be9fd; font-size: 10px; padding-left: 4px;"
        )
        self.model_status_label.setToolTip(
            "Aktuell verwendetes Modell (Auflösung: Combo → ChatConfig → "
            "Pipeline-Override → LlmService → erstes verfügbares)."
        )
        header_layout.addWidget(self.model_status_label)

        header_layout.addStretch()

        # System prompt button
        self.system_prompt_btn = QPushButton("⚙️ System-Prompt")
        self.system_prompt_btn.setStyleSheet(
            "QPushButton { font-size: 10px; padding: 3px 8px; border: 1px solid #555; "
            "border-radius: 3px; background-color: #3d3d3d; color: #ccc; }"
            "QPushButton:hover { background-color: #4d4d4d; }"
        )
        self.system_prompt_btn.setToolTip("System-Prompt für den Assistenten bearbeiten")
        self.system_prompt_btn.clicked.connect(self.show_system_prompt_dialog)
        header_layout.addWidget(self.system_prompt_btn)

        # Reset toggle
        self.reset_toggle = QCheckBox("🔄 Bei neuer Pipeline zurücksetzen")
        self.reset_toggle.setChecked(True)
        self.reset_toggle.setStyleSheet("color: #aaa; font-size: 10px;")
        self.reset_toggle.setToolTip(
            "Wenn aktiviert, wird der Chat bei jedem neuen Pipeline-Lauf geleert."
        )
        header_layout.addWidget(self.reset_toggle)

        main_layout.addWidget(header)

        # --- Chat History ---
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setFont(get_scaled_font(size_delta=-1, monospace=True))
        self.history.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "border: none; padding: 8px; }"
        )
        self.history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.history)

        # --- Input Area ---
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background-color: #2d2d2d; border-top: 1px solid #444; }")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 6, 8, 6)
        input_layout.setSpacing(6)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Frage zu den Pipeline-Ergebnissen stellen...")
        self.input_field.setStyleSheet(
            "QLineEdit { background-color: #3d3d3d; color: #e0e0e0; "
            "border: 1px solid #555; border-radius: 4px; padding: 6px 10px; "
            "font-size: 11pt; }"
            "QLineEdit:focus { border: 1px solid #8be9fd; }"
        )
        self.input_field.setFont(get_scaled_font(size_delta=0))
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Senden")
        self.send_btn.setStyleSheet(get_button_styles().get("primary", ""))
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        self.cancel_btn = QPushButton("Abbrechen")
        self.cancel_btn.setStyleSheet(get_button_styles().get("danger", ""))
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_generation)
        input_layout.addWidget(self.cancel_btn)

        # --- Typing indicator (WhatsApp-style animated dots) ---
        self.typing_label = QLabel("")
        self.typing_label.setStyleSheet(
            "color: #8be9fd; font-size: 9pt; font-style: italic; "
            "padding: 2px 12px; background-color: #1e1e1e; "
            "border-top: 1px solid #2a2a2a;"
        )
        self.typing_label.setVisible(False)
        main_layout.addWidget(self.typing_label)

        main_layout.addWidget(input_frame)

        # Track whether the current assistant turn has opened its
        # rendering block. False between turns; True from first token until
        # finalize. Also store the cell-cursor inside the assistant
        # bubble's table cell so streaming tokens land inside the bubble.
        self._assistant_block_open: bool = False
        self._assistant_cell_cursor: Optional[QTextCursor] = None
        self._current_render_model: str = ""

        # Typing-indicator animation
        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(400)
        self._typing_timer.timeout.connect(self._tick_typing)
        self._typing_dots: int = 0
        self._typing_model: str = ""

        # Initial paint of model status label.
        self._refresh_model_status()

    def _show_typing(self, model_label: str) -> None:
        """Show animated typing indicator below history."""
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

    def _refresh_model_status(self) -> None:
        """Update header label with the currently-resolved provider/model."""
        try:
            provider, model = self._resolve_provider_model()
        except Exception:
            provider, model = "", ""
        if provider and model:
            self.model_status_label.setText(f"→ {provider} | {model}")
        else:
            self.model_status_label.setText("→ (kein Modell)")

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def load_context(self, analysis_state) -> None:
        """Load pipeline results as chat context - Claude Generated.

        Strategy (changed in δ.3 follow-up): the user-prompt dump is kept
        deliberately small — only Title + truncated Abstract. All other
        data (keywords, chains, DK-codes, missing concepts) is exposed to
        the agent via session tools so the LLM has to look them up.
        This prevents passive context dumping that crowds the prompt and
        encourages tool-grounded answers (Anti-Halluc-Block).

        Also builds a ``SharedContext`` from ``KeywordAnalysisState`` so
        that the alima_tools layer (``get_keywords``,
        ``get_dk_classifications``, ``get_keyword_chains``,
        ``list_available_data`` …) is available even for classic
        (non-agentic) pipeline runs.
        """
        if self.reset_toggle.isChecked():
            self._clear_history()

        self.session.reset()

        if analysis_state is None:
            self.current_context = ""
            self.session.last_shared_context = None
            self._append_system_message(
                "ℹ️ Kein Pipeline-Kontext geladen. Chat funktioniert trotzdem — "
                "stelle einfach eine Frage."
            )
            return

        # --- Build slim user-prompt context (header only) ---
        parts = []
        if hasattr(analysis_state, "working_title") and analysis_state.working_title:
            self.working_title = analysis_state.working_title
            parts.append(f"Titel: {analysis_state.working_title}")
        if hasattr(analysis_state, "original_abstract") and analysis_state.original_abstract:
            abstract = analysis_state.original_abstract
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            parts.append(f"Abstract: {abstract}")
        self.current_context = "\n".join(parts) or "(kein Titel/Abstract)"

        # --- Build SharedContext for tool layer ---
        ctx = self._shared_context_from_analysis_state(analysis_state)
        self.session.last_shared_context = ctx

        # Status line for the operator.
        kw_count = len(getattr(ctx, "extracted_keywords", []) or [])
        if not kw_count:
            kw_count = len(getattr(ctx, "initial_keywords", []) or [])
        dk_count = len(getattr(ctx, "dk_classifications", []) or [])
        self._append_system_message(
            f"✅ Kontext geladen: {self.working_title or 'Unbenannt'}"
            f" ({kw_count} Keywords, {dk_count} DK-Codes, Tools aktiv)"
        )
        self.logger.info(
            f"ChatWidget: SharedContext built — {kw_count} keywords, {dk_count} DK codes"
        )

    @staticmethod
    def _shared_context_from_analysis_state(state) -> Optional[object]:
        """Adapter: build a ``SharedContext`` from a ``KeywordAnalysisState``.

        Maps every relevant field so the alima_tools layer can answer
        questions about ALL pipeline data — not just the curated
        final-LLM output. Returns None on import / construction failure
        (tools then degrade to empty registry, no crash).
        """
        try:
            from src.core.agents.shared_context import SharedContext
        except Exception:
            return None

        ctx = SharedContext()
        ctx.working_title = getattr(state, "working_title", "") or ""
        ctx.abstract = getattr(state, "original_abstract", "") or ""
        ctx.initial_keywords = list(getattr(state, "initial_keywords", []) or [])

        final = getattr(state, "final_llm_analysis", None)
        if final is not None:
            ctx.extracted_keywords = list(
                getattr(final, "extracted_gnd_keywords", []) or []
            )
            ctx.keyword_chains = list(getattr(final, "keyword_chains", []) or [])
            ctx.missing_concepts = list(getattr(final, "missing_concepts", []) or [])
            verification = getattr(final, "verification", None)
            if verification:
                ctx.extra["verification"] = verification
            extracted_classes = getattr(final, "extracted_gnd_classes", None) or []
            if extracted_classes:
                ctx.extra["extracted_gnd_classes"] = list(extracted_classes)

        # --- GND search results from the Pipeline search step ---
        # KeywordAnalysisState.search_results = List[SearchResult] where
        # SearchResult.results is {gnd_id: {"title": ..., "system": ..., ...}}.
        # Flatten into ctx.gnd_entries (list of dicts with gnd_id key) and
        # build ctx.gnd_entries_per_keyword for term-scoped lookup.
        gnd_entries = []
        gnd_per_kw: dict = {}
        seen_ids = set()
        for sr in getattr(state, "search_results", []) or []:
            term = getattr(sr, "search_term", "") or ""
            results = getattr(sr, "results", {}) or {}
            titles: list = []
            for gnd_id, info in results.items():
                if not isinstance(info, dict):
                    info = {"value": info}
                title = info.get("title") or info.get("label") or ""
                titles.append(title or gnd_id)
                if gnd_id in seen_ids:
                    continue
                seen_ids.add(gnd_id)
                gnd_entries.append({"gnd_id": gnd_id, **info})
            if term and titles:
                gnd_per_kw[term] = titles
        ctx.gnd_entries = gnd_entries
        ctx.gnd_entries_per_keyword = gnd_per_kw

        # --- Final DK classifications (curated) ---
        raw_dk = getattr(state, "dk_classifications", []) or []
        normalised_dk = []
        for cls in raw_dk:
            if isinstance(cls, dict):
                normalised_dk.append(cls)
            else:
                normalised_dk.append({"code": str(cls)})
        ctx.dk_classifications = normalised_dk

        # --- DK catalog-search results from the catalog step ---
        ctx.dk_search_results = list(getattr(state, "dk_search_results", []) or [])
        stats = getattr(state, "dk_statistics", None)
        if stats:
            ctx.dk_catalog_stats = dict(stats)

        # --- Final keyword shortlist (if pipeline stores one) ---
        final_keywords = getattr(state, "final_keywords", None)
        if final_keywords:
            ctx.extra["final_keywords"] = list(final_keywords)

        return ctx

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_message(self):
        """Send user message and start LLM generation - Claude Generated.

        P-δ.3: spawns a ``ChatAgentWorker`` (multi-turn agent loop with
        tool support) instead of the legacy single-shot ChatWorker.
        """
        text = self.input_field.text().strip()
        if not text:
            return

        if self.current_worker and self.current_worker.isRunning():
            return  # Already running

        # Display user message
        self._append_user_message(text)
        self.input_field.clear()
        self.session.append("user", text)

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            context=self.current_context,
            user_message=text,
        )

        # Determine provider/model
        provider, model = self._resolve_provider_model()
        if not provider or not model:
            self._append_system_message(
                "⚠️ Kein LLM-Provider konfiguriert. Bitte in Pipeline-Einstellungen ein Modell wählen."
            )
            return

        # P-δ.3: refresh SharedContext snapshot from pipeline_manager and
        # build the per-turn toolset bound to this session.
        self._refresh_shared_context()
        chat_config = self._get_chat_config()
        try:
            tool_registry = build_chat_toolset(
                session=self.session,
                chat_config=chat_config,
                mcp_registry=self.mcp_registry,
            )
        except Exception:
            self.logger.exception("ChatWidget: build_chat_toolset failed")
            self._append_system_message("❌ Tool-Setup fehlgeschlagen — siehe Log.")
            return

        # Start worker
        self._set_ui_running(True)
        self.current_worker = ChatAgentWorker(
            llm_service=self.llm_service,
            tool_registry=tool_registry,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            temperature=getattr(chat_config, "temperature", 0.5),
            max_iterations=getattr(chat_config, "max_iterations", 10),
        )
        self.current_worker.token_received.connect(self._on_token)
        self.current_worker.status_message.connect(self._on_status_message)
        self.current_worker.tool_called.connect(self._on_tool_called)
        self.current_worker.tool_result.connect(self._on_tool_result)
        self.current_worker.generation_finished.connect(self._on_finished)
        self.current_worker.generation_error.connect(self._on_error)
        self._current_render_model = f"{provider} | {model}"
        self._refresh_model_status()
        self._show_typing(self._current_render_model)
        self.current_worker.start()

        self.message_sent.emit(text)

    def cancel_generation(self):
        """Cancel running LLM generation - Claude Generated"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.request_stop()
            self._append_system_message("⏹ Generation abgebrochen.")
            self._set_ui_running(False)

    @pyqtSlot(str)
    def _on_token(self, token: str):
        """Receive streaming token from ChatAgentWorker - Claude Generated"""
        if not self._assistant_block_open:
            self._hide_typing()
            self._open_assistant_message(self._current_render_model)
            self._assistant_block_open = True
        self._append_assistant_token(token)

    # Status-message prefixes whose info is already rendered by the
    # ``on_tool_call`` / ``on_tool_result`` hook handlers. Skipping them
    # here avoids duplicate inline markers per tool dispatch.
    _STATUS_SKIP_PREFIXES = ("🔧", "✓", "💭")

    @pyqtSlot(str)
    def _on_status_message(self, line: str):
        """AgentLoop progress (🔄/✅/…). Render as faded inline marker so
        the main token stream stays clean but the operator can still see
        what's happening. Dedup against hook-rendered tool markers."""
        text = (line or "").strip()
        if not text:
            return
        if text.startswith(self._STATUS_SKIP_PREFIXES):
            return
        self._append_tool_marker(text)

    @pyqtSlot(str, dict)
    def _on_tool_called(self, name: str, args: dict):
        """Inline marker before a tool dispatch (P-δ.3, plain-text)."""
        args_preview = self._format_tool_args(args)
        self._append_tool_marker(f"🔧 {name}({args_preview})")

    @pyqtSlot(str, str)
    def _on_tool_result(self, name: str, result_str: str):
        """Inline marker after a tool dispatch (P-δ.3, plain-text)."""
        preview = result_str.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        self._append_tool_marker(f"↳ {preview}")

    @pyqtSlot(object)
    def _on_finished(self, result):
        """Generation finished - Claude Generated.

        P-δ.3: ``result`` is an ``AgentResult``; persist its final content
        into the session transcript and, if streaming did not actually
        emit tokens (Ollama/OpenAI disable streaming when tools are
        active — see ``_generate_*_with_tools``), render the full content
        now so the operator sees the answer.
        """
        try:
            final = getattr(result, "content", "") or ""
            if final:
                self.session.append("assistant", final)
                if not self._assistant_block_open:
                    self._hide_typing()
                    self._open_assistant_message(self._current_render_model)
                    self._assistant_block_open = True
                    self._append_assistant_token(final)
        except Exception:
            self.logger.exception("ChatWidget: failed to append assistant turn")
        self._hide_typing()
        self._finalize_assistant_message()
        self._set_ui_running(False)
        self.logger.info("ChatWidget: generation finished")

    @pyqtSlot(str)
    def _on_error(self, error: str):
        """Generation error - Claude Generated"""
        self._hide_typing()
        self._append_system_message(f"❌ Fehler: {error}")
        self._assistant_block_open = False
        self._assistant_cell_cursor = None
        self._set_ui_running(False)
        self.logger.error(f"ChatWidget: generation error: {error}")

    def _on_state_changed(self, diff: dict) -> None:
        """AlimaStateBus subscriber — refresh SharedContext snapshot."""
        self._refresh_shared_context()

    def _refresh_shared_context(self) -> None:
        """Pull the latest SharedContext from PipelineManager into session.

        Only overwrites when the pipeline-manager actually exposes a
        non-None SharedContext — classic-pipeline mode never populates
        that field, so we must not clobber the locally-built ctx from
        ``load_context``.
        """
        try:
            if self.pipeline_manager is None:
                return
            ctx = getattr(self.pipeline_manager, "last_shared_context", None)
            if ctx is not None:
                self.session.last_shared_context = ctx
        except Exception:
            self.logger.exception("ChatWidget: shared-context refresh failed")

    def _get_chat_config(self):
        """Return ``ChatConfig`` from unified config, or a default instance."""
        try:
            from ..utils.config_manager import ConfigManager
            cfg = ConfigManager().get_unified_config()
            chat_cfg = getattr(cfg, "chat_config", None)
            if chat_cfg is not None:
                return chat_cfg
        except Exception:
            self.logger.exception("ChatWidget: ChatConfig lookup failed")
        from ..utils.config_models import ChatConfig
        return ChatConfig()

    @staticmethod
    def _format_tool_args(args: dict) -> str:
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            sv = repr(v)
            if len(sv) > 40:
                sv = sv[:40] + "…"
            parts.append(f"{k}={sv}")
        joined = ", ".join(parts)
        if len(joined) > 80:
            joined = joined[:80] + "…"
        return joined

    # ------------------------------------------------------------------
    # UI Helpers
    # ------------------------------------------------------------------

    def _set_ui_running(self, running: bool):
        """Toggle UI between running and idle state - Claude Generated"""
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

    def _clear_history(self):
        """Clear chat history - Claude Generated"""
        self.history.clear()
        self.session.reset()
        self.current_context = ""
        self.working_title = ""

    # ------------------------------------------------------------------
    # Message rendering (colored blocks in QTextEdit via HTML)
    # ------------------------------------------------------------------

    def _append_user_message(self, text: str):
        """WhatsApp-style user bubble: right-aligned greenish bubble (~65%)."""
        self._insert_bubble(
            text,
            align=Qt.AlignmentFlag.AlignRight,
            width_percent=65,
            bg_color="#005c4b",
            fg_color="#e9edef",
        )

    def _insert_bubble(
        self,
        text: str,
        *,
        align: Qt.AlignmentFlag,
        width_percent: int,
        bg_color: str,
        fg_color: str,
    ) -> None:
        """Insert a fully-rendered chat bubble via QTextTable."""
        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.history.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        table_fmt = QTextTableFormat()
        table_fmt.setCellPadding(8)
        table_fmt.setCellSpacing(0)
        table_fmt.setBorder(0)
        table_fmt.setWidth(QTextLength(QTextLength.Type.PercentageLength, width_percent))
        table_fmt.setAlignment(align)
        table = cursor.insertTable(1, 1, table_fmt)
        cell = table.cellAt(0, 0)
        cell_fmt = QTextTableCellFormat()
        cell_fmt.setBackground(QColor(bg_color))
        cell.setFormat(cell_fmt)
        body = self._escape_html(text).replace("\n", "<br>")
        cell.firstCursorPosition().insertHtml(
            f'<span style="color: {fg_color}; font-size: 10pt;">{body}</span>'
        )
        # Move main cursor after the table for subsequent inserts.
        end_cursor = self.history.textCursor()
        end_cursor.movePosition(QTextCursor.MoveOperation.End)
        end_cursor.insertBlock(QTextBlockFormat())
        scrollbar = self.history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_tool_marker(self, text: str):
        """Inline tool-call / tool-result marker (P-δ.3, plain monospace line)."""
        html = (
            f'<div style="margin: 2px 0 2px 8px; '
            f'font-family: monospace; font-size: 9pt; color: #888;">'
            f'{self._escape_html(text)}</div>'
        )
        self._append_html(html)

    def _append_system_message(self, text: str):
        """Add system/info message (centered, green) - Claude Generated"""
        html = (
            f'<div style="text-align: center; margin: 4px 0;">'
            f'<span style="color: #4caf50; font-size: 9pt; font-style: italic;">'
            f'{self._escape_html(text)}</span></div>'
        )
        self._append_html(html)

    def _open_assistant_message(self, model_label: str):
        """Start a fresh assistant turn — WhatsApp-style left-aligned bubble.

        Creates a ``QTextTable`` with one cell whose cursor is kept in
        ``self._assistant_cell_cursor`` so streaming tokens land inside
        the bubble (not in the document flow outside it).
        """
        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.history.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        # Model header — faint italic line above the bubble.
        cursor.insertHtml(
            f'<span style="color: #8be9fd; font-size: 9pt; font-style: italic;">'
            f'🤖 {self._escape_html(model_label or "Modell")}'
            f'</span>'
        )
        cursor.insertBlock(QTextBlockFormat())
        # Open the bubble table (left-aligned, ~75% width).
        table_fmt = QTextTableFormat()
        table_fmt.setCellPadding(8)
        table_fmt.setCellSpacing(0)
        table_fmt.setBorder(0)
        table_fmt.setWidth(QTextLength(QTextLength.Type.PercentageLength, 75))
        table_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        table = cursor.insertTable(1, 1, table_fmt)
        cell = table.cellAt(0, 0)
        cell_fmt = QTextTableCellFormat()
        cell_fmt.setBackground(QColor("#202c33"))
        cell.setFormat(cell_fmt)
        self._assistant_cell_cursor = cell.firstCursorPosition()
        scrollbar = self.history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_assistant_token(self, token: str):
        """Append a streaming token inside the open assistant bubble cell."""
        if self._assistant_cell_cursor is None:
            return
        html = self._escape_html(token).replace("\n", "<br>")
        self._assistant_cell_cursor.insertHtml(
            f'<span style="color: #e9edef; font-size: 10pt;">{html}</span>'
        )
        scrollbar = self.history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _finalize_assistant_message(self):
        """Close the current assistant turn and ensure clean separation."""
        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock(QTextBlockFormat())
        self._assistant_block_open = False
        self._assistant_cell_cursor = None
        scrollbar = self.history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_html(self, html: str):
        """Append a block-level HTML chunk and scroll to bottom.

        Forces a paragraph boundary with a *default* block format so
        consecutive entries (system msg / user bubble / tool marker /
        status) don't inherit the narrow ``rightMargin`` of the previous
        assistant body block.
        """
        cursor = self.history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.history.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        cursor.insertHtml(html)
        scrollbar = self.history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters - Claude Generated"""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    # ------------------------------------------------------------------
    # System prompt dialog
    # ------------------------------------------------------------------

    def show_system_prompt_dialog(self):
        """Show dialog to edit system prompt - Claude Generated"""
        dialog = SystemPromptDialog(self.system_prompt, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_prompt = dialog.get_prompt()
            if new_prompt:
                self.system_prompt = new_prompt
                self.logger.info("ChatWidget: system prompt updated")
                self._append_system_message("✅ System-Prompt aktualisiert.")

    # ------------------------------------------------------------------
    # Provider / model resolution
    # ------------------------------------------------------------------

    def _populate_model_combo(self):
        """Fill model combo with available provider/model pairs - Claude Generated"""
        try:
            self.model_combo.clear()
            self.model_combo.addItem("-- Auto --", None)
            from ..utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            unified_config = config_manager.get_unified_config()
            for provider in unified_config.get_enabled_providers():
                models = getattr(provider, 'available_models', []) or []
                if not models and getattr(provider, 'preferred_model', None):
                    models = [provider.preferred_model]
                for model in models:
                    self.model_combo.addItem(f"{provider.name} | {model}", f"{provider.name}|{model}")
        except Exception as e:
            self.logger.error(f"Error populating model combo: {e}")

    def _resolve_provider_model(self) -> tuple[str, str]:
        """Resolve LLM provider/model — P-δ.3 priority order:

        1. Explicit combo selection (operator override)
        2. ``ChatConfig.default_provider/model`` (chat-scoped config)
        3. Pipeline-config global override (legacy fallback)
        4. LlmService last-used (compat)
        5. First-available provider (last-ditch)
        """
        # 1. Explicit selection from combo
        override_data = self.model_combo.currentData()
        if override_data:
            provider, model = override_data.split("|", 1)
            if provider and model:
                return provider, model

        # 2. ChatConfig defaults (P-δ.3 — chat-scoped, independent of pipeline)
        try:
            chat_cfg = self._get_chat_config()
            provider = getattr(chat_cfg, "default_provider", "") or ""
            model = getattr(chat_cfg, "default_model", "") or ""
            if provider and model:
                return provider, model
        except Exception:
            self.logger.exception("ChatWidget: ChatConfig provider lookup failed")

        # 3. Pipeline-config global override (legacy)
        if self.pipeline_manager and hasattr(self.pipeline_manager, "config"):
            cfg = self.pipeline_manager.config
            provider = getattr(cfg, "global_provider_override", None)
            model = getattr(cfg, "global_model_override", None)
            if provider and model:
                return provider, model

        # 4. Fallback: llm_service current (last used)
        provider = getattr(self.llm_service, "current_provider", None)
        model = getattr(self.llm_service, "current_model", None)
        if provider and model:
            return provider, model

        # 4. Fallback: first available provider from llm_service - Claude Generated
        try:
            clients = getattr(self.llm_service, "clients", {})
            if clients:
                provider = list(clients.keys())[0]
                sp = getattr(self.llm_service, "supported_providers", {})
                pinfo = sp.get(provider, {})
                models = pinfo.get("models", [])
                if models:
                    return provider, models[0]
        except Exception:
            pass

        return "", ""
