"""ChatAgentWorker — multi-turn agent worker for the chat dock.

WP10 P-δ.3. Claude Generated.

Replaces the legacy single-shot ``ChatWorker``. Wraps
:class:`src.core.agent_loop.AgentLoop` and exposes Qt signals so the
chat widget can render streaming tokens, tool-call markers, tool
results, and a final ``AgentResult``.

Threading model: QThread (same as the worker it replaces). The thread
holds a ``threading.Event`` so ``request_stop()`` is honoured at the
next AgentLoop iteration boundary (max one extra LLM round-trip
latency).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.agent_loop import AgentLoop


logger = logging.getLogger(__name__)


class ChatAgentWorker(QThread):
    """Run a multi-turn agent conversation in a background Qt thread."""

    # Signals
    token_received = pyqtSignal(str)          # real LLM token (final answer stream)
    status_message = pyqtSignal(str)          # AgentLoop progress line (🔄/✅/⚠️/…)
    tool_called = pyqtSignal(str, dict)       # (tool_name, arguments)
    tool_result = pyqtSignal(str, str)        # (tool_name, result_json_str)
    generation_finished = pyqtSignal(object)  # AgentResult
    generation_error = pyqtSignal(str)

    def __init__(
        self,
        llm_service: Any,
        tool_registry: Any,
        system_prompt: str,
        user_prompt: str,
        provider: str,
        model: str,
        *,
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        max_iterations: int = 30,
        timeout_seconds: int = 600,
        seed: Optional[int] = None,
        tools: Optional[list] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._llm_service = llm_service
        self._tool_registry = tool_registry
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._seed = seed
        # tools=None ⇒ AgentLoop treats as "no tools";
        # tools=[]   ⇒ AgentLoop returns all registered tools.
        # We default to "all", matching the chat-tools registry contract.
        self._tools = tools if tools is not None else []
        self._history = history
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Signal the agent loop to stop at the next iteration boundary."""
        self._stop_event.set()
        try:
            self._llm_service.cancel_generation(reason="user_requested")
        except Exception:  # cancel_generation may not exist on every backend
            logger.debug("cancel_generation raised; ignoring", exc_info=True)

    # ------------------------------------------------------------------

    def _on_stream(self, token: str) -> None:
        if not token:
            return
        self.token_received.emit(token)

    def _on_status(self, line: str) -> None:
        if not line:
            return
        logger.debug("AgentLoop status: %s", line.strip())
        self.status_message.emit(line)

    def _on_tool_call(self, tc: Any) -> None:
        try:
            self.tool_called.emit(tc.name, dict(tc.arguments or {}))
        except Exception:
            logger.exception("Failed to emit tool_called signal")

    def _on_tool_result(self, name: str, result_str: str) -> None:
        try:
            self.tool_result.emit(name, result_str)
        except Exception:
            logger.exception("Failed to emit tool_result signal")

    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            loop = AgentLoop(
                llm_service=self._llm_service,
                tool_registry=self._tool_registry,
                max_iterations=self._max_iterations,
                timeout_seconds=self._timeout_seconds,
                stream_callback=self._on_stream,
                status_callback=self._on_status,
                on_tool_call=self._on_tool_call,
                on_tool_result=self._on_tool_result,
                should_stop=self._stop_event.is_set,
            )
            result = loop.run(
                system_prompt=self._system_prompt,
                user_prompt=self._user_prompt,
                tools=self._tools,
                provider=self._provider,
                model=self._model,
                temperature=self._temperature,
                top_p=self._top_p,
                max_tokens=self._max_tokens,
                seed=self._seed,
                conversation_history=self._history,
            )
            self.generation_finished.emit(result)
        except Exception as exc:
            logger.exception("ChatAgentWorker error")
            self.generation_error.emit(str(exc))
