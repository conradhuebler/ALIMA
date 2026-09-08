"""Chat-agent mixin for PipelineChatPanel. Claude Generated.

Extracted from ``pipeline_chat_panel.py`` (F-5 god-file split): the former
``ChatWidget`` behavior — typing indicator, provider/model resolution,
pipeline-context loading, the send/cancel/worker lifecycle, the shared-context
refresh, and the running/stopping UI state.

The panel has no provider/model picker of its own: the pipeline toolbar hands
its pick over through ``set_llm_override`` and ``model_status_label`` shows what
that resolves to.

``ChatAgentMixin`` is mixed into ``PipelineChatPanel`` (which provides
``__init__``, the widgets referenced here — ``input_field``, ``send_btn``,
``cancel_btn``, ``typing_label``, ``model_status_label`` — ``self._renderer``,
``self.session``, and the render delegators such as ``_append_system_message`` /
``_open_assistant_message``); it is not a standalone widget.
"""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import pyqtSlot

from ..core.chat_prompts import (
    CHAT_HISTORY_WINDOW,
    apply_chat_directives,
    build_chat_system_prompt,
    detect_mode,
    get_user_prompt_template,
    resolve_prompt_compact,
)
from ..core.headless_agent import resolve_provider_model
from ..utils.error_visibility import log_caught
from ..utils.i18n import t
from .chat_agent_worker import ChatAgentWorker
from .chat_tools import build_chat_toolset


class ChatAgentMixin:
    """Chat-side rendering & lifecycle (ported from ChatWidget)."""

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

    @pyqtSlot(str, str, str)
    def _on_agent_model_switch(self, provider: str, model: str, reason: str) -> None:
        """Announce a model the agent picked for itself. **UI thread only.**

        Reached via ``model_switch_requested``; calling it directly from the
        chat worker thread aborts the process, because it renders into the log
        and writes a QLabel. A switch that only shows up in the tool log is a
        change nobody notices, so it belongs in the conversation.
        - Claude Generated
        """
        suffix = f" — {reason}" if reason else ""
        try:
            self._append_system_message(
                f"🤖 Modellwechsel ab der nächsten Antwort: {provider} | {model}{suffix}"
            )
            self._refresh_model_status()
        except Exception:
            self.logger.exception("PipelineChatPanel: announcing model switch failed")

    def set_llm_override(self, provider: str, model: str) -> None:
        """Adopt the pipeline toolbar's LLM pick for the chat as well.

        Called by the embedding pipeline tab whenever its selector changes. An
        empty pair means "-- Standard --" and falls through to the configured
        chat default. - Claude Generated
        """
        self._llm_override = (provider or "", model or "")
        session = getattr(self, "session", None)
        if session is not None and hasattr(session, "clear_requested_model"):
            # The operator acted — that outranks whatever the agent picked.
            session.clear_requested_model()
        self._refresh_model_status()

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
            state = t("chat.state.on") if new_value else t("chat.state.off")
            self.set_status_strip(t("chat.status.autonomous", state=state))
        except Exception:
            self.logger.exception(
                "PipelineChatPanel: persist autonomous_pipeline failed"
            )

    def refresh_providers(self) -> None:
        """Re-resolve after a settings change (provider added or removed).

        Called by the embedding pipeline tab's on_config_changed. - Claude Generated
        """
        self._refresh_model_status()

    def _build_tool_registry(self, chat_config) -> Optional[Any]:
        """Assemble this turn's toolset, or None when that failed.

        Its own method so the cross-thread contract below is testable without
        building the widget: ``on_model_switch`` must be the **signal's**
        ``emit``, never the slot. The tool runs inside ``ChatAgentWorker``'s
        QThread, and calling the slot from there writes into the log view and a
        QLabel from the wrong thread, which aborts the process (SIGTRAP).
        - Claude Generated
        """
        kb_manager = None
        if self.pipeline_manager is not None:
            kb_manager = (
                getattr(self.pipeline_manager, "unified_knowledge_manager", None)
                or getattr(self.pipeline_manager, "knowledge_manager", None)
            )
        try:
            return build_chat_toolset(
                session=self.session,
                chat_config=chat_config,
                mcp_registry=self.mcp_registry,
                pipeline_manager=self.pipeline_manager,
                kb_manager=kb_manager,
                proposal_gateway=self.proposal_gateway,
                llm_service=self.llm_service,
                on_model_switch=self.model_switch_requested.emit,
            )
        except Exception:
            self.logger.exception("PipelineChatPanel: build_chat_toolset failed")
            return None

    def _resolve_provider_model(self) -> tuple[str, str]:
        # Shared chain (CLI/HTTP/GUI): explicit override → ChatConfig default →
        # pipeline global override → unified agentic default → pipeline default →
        # general default → first enabled provider → llm_service.current.
        # The explicit override is the pipeline toolbar's pick, handed over by
        # the embedding tab; "-- Standard --" yields ("", "") and falls through.
        # A model the agent asked for (only reachable with
        # ChatConfig.allow_model_switch on) wins over the toolbar pick until the
        # operator touches the toolbar again — otherwise the switch would be
        # undone by the very control it was meant to adapt. - Claude Generated
        ov_provider, ov_model = getattr(self, "_llm_override", ("", ""))
        session = getattr(self, "session", None)
        req_provider = str(getattr(session, "requested_provider", "") or "")
        req_model = str(getattr(session, "requested_model", "") or "")
        if req_provider and req_model:
            ov_provider, ov_model = req_provider, req_model
        provider, model = resolve_provider_model(
            ov_provider or None, ov_model or None,
            pipeline_manager=self.pipeline_manager,
            llm_service=self.llm_service,
        )
        if provider and model:
            return provider, model
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

    def _registered_tool_names(self):
        """Tool names the agent actually has, for the system prompt.

        The search_* tools are generated per enabled plugin instance, so the prompt
        must not recommend a disabled one. ``None`` on failure → the prompt keeps
        its static tool list rather than claiming nothing is available.
        - Claude Generated"""
        try:
            reg = getattr(self, "mcp_registry", None)
            if reg is not None and hasattr(reg, "get_tool_names"):
                return set(reg.get_tool_names())
        except Exception:
            self.logger.exception("PipelineChatPanel: tool-name lookup failed")
        return None

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
        self.set_status_strip(
            t(
                "chat.status.context_loaded",
                title=self.working_title or t("chat.untitled"),
                kw=kw_count,
                dk=dk_count,
            )
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

        # Resolve provider/model + config first — needed to pick the prompt tier.
        provider, model = self._resolve_provider_model()
        if not provider or not model:
            self._append_system_message(t("chat.no_provider"))
            return

        chat_config = self._get_chat_config()

        # Mode-aware prompt assembly + tier + language/history directives - Claude Generated
        mode = detect_mode(text, self.current_context)
        self.set_status_strip(t("chat.status.mode", mode=mode))
        history = list(self.session.messages[-CHAT_HISTORY_WINDOW:])
        history_truncated = len(self.session.messages) > len(history)
        compact = resolve_prompt_compact(
            getattr(chat_config, "system_prompt_tier", "auto"), model
        )
        # Personal rules scoped to the chat come in here (shared with the
        # headless runner). - Claude Generated
        base_system_prompt = build_chat_system_prompt(
            self.system_prompt,
            mode=mode,
            compact=compact,
            institution_context=getattr(chat_config, "institution_context", ""),
            available_tools=self._registered_tool_names(),
        )
        effective_system_prompt = apply_chat_directives(
            base_system_prompt,
            language=self.chat_language,
            history_truncated=history_truncated,
        )
        effective_user_template = get_user_prompt_template(mode)
        user_prompt = effective_user_template.format(
            context=self.current_context or "(kein Werk geladen)",
            user_message=text,
        )

        self._refresh_shared_context()
        tool_registry = self._build_tool_registry(chat_config)
        if tool_registry is None:
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
            max_tokens=getattr(chat_config, "max_tokens", 4096),
            max_iterations=30,
            timeout_seconds=getattr(chat_config, "timeout_seconds", 600),
            history=history,
            think=self._get_chat_think_override(),
        )
        self._chat_tokens_seen = False  # duplicate guard for _on_finished - Claude Generated
        self.current_worker.token_received.connect(self._on_token)
        self.current_worker.thinking_received.connect(self._on_thinking_token)
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
        # Only SIGNAL the worker; do NOT flip the UI to 'ready' here. The worker
        # is still winding down (finishing the in-flight tool/LLM call), and
        # send_message() drops new sends while it runs — so a premature 'ready'
        # state would silently swallow the next message. The lifecycle returns to
        # ready in _on_finished/_on_error, the single owners of that transition.
        # - Claude Generated
        if self.current_worker and self.current_worker.isRunning() and not self._stopping:
            self._stopping = True
            self.current_worker.request_stop()
            self._append_system_message("⏹ Abbruch angefordert … (warte auf Worker)")
            self._set_ui_stopping()

    @pyqtSlot(str)
    def _on_token(self, token: str):
        self._chat_tokens_seen = True
        if not self._renderer._assistant_block_open:
            self._hide_typing()
            self._open_assistant_message(self._current_render_model)
            self._renderer._assistant_block_open = True
        self._append_assistant_token(token)

    @pyqtSlot(str)
    def _on_thinking_token(self, text: str):
        """Route thinking/reasoning content into the 💭 collapsible - Claude Generated"""
        try:
            self._hide_typing()  # thinking arriving means the model is alive
            self._renderer.append_thinking(text)
        except Exception as e:
            log_caught(self.logger, e, "chat thinking render")

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
        # No open step → logger only (Chat-UX 7/9): worker status / LLM
        # progress used to leak into the chat surface as dim debug lines;
        # tool activity is already visible via the collapsible blocks.
        self.logger.debug("chat-agent status: %s", text)

    @pyqtSlot(object)
    def _on_finished(self, result):
        try:
            # Merge full conversation (user + tool calls + tool results + assistant)
            for msg in getattr(result, "messages", []) or []:
                self.session.messages.append(dict(msg))
            final = getattr(result, "content", "") or ""
            # _chat_tokens_seen: with per-iteration bubbles the last bubble is
            # already finalized here — re-rendering `final` after a streamed
            # run would duplicate the prose (e.g. after cancel). - Claude Generated
            if (
                final
                and not self._renderer._assistant_block_open
                and not getattr(self, "_chat_tokens_seen", False)
            ):
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
        if self._stopping:
            self._append_system_message(t("chat.cancelled"))
        self._set_ui_running(False)

    @pyqtSlot(str)
    def _on_error(self, error: str):
        self._hide_typing()
        # WP12 §9.3: shared red error chrome instead of a bare system line.
        self._renderer.render_error_block(t("chat.error_title"), error)
        self._renderer._assistant_block_open = False
        self._renderer._assistant_cell_cursor = None
        self._set_ui_running(False)
        self.logger.error(f"PipelineChatPanel: generation error: {error}")

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
        # Single owner of the running↔ready transition. Always clears the
        # transient 'stopping' state and re-arms the cancel button. - Claude Generated
        self._stopping = False
        self.send_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.cancel_btn.setEnabled(True)
        self.input_field.setEnabled(not running)
        if running:
            self.input_field.setPlaceholderText(t("chat.input.generating"))
        else:
            # Same text as the initial placeholder — the ready state used to
            # show a different wording (Chat-UX 7/9).
            self.input_field.setPlaceholderText(t("chat.input.placeholder"))
            self.input_field.setFocus()

    def _set_ui_stopping(self):
        """Transient state after a cancel request: keep send locked and show the
        cancel button disabled until the worker actually finishes, so the UI never
        claims 'ready' (and silently drops the next send) mid-stop. Claude Generated."""
        self.send_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.cancel_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        self.input_field.setPlaceholderText(t("chat.input.stopping"))
