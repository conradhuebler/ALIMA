"""Headless agent driver — shared by the CLI (`alima agent`) and the HTTP
(`POST /agent/run`) frontends.

Claude Generated (P-ι).

Drives the same ``AgentLoop`` + chat toolset the GUI ``ChatAgentWorker`` uses,
without any PyQt6 dependency. The GUI worker emits Qt signals; here the same
``AgentLoop`` callbacks are plain Python callables the caller supplies (print
to stdout, push to an SSE queue, …).

Cancel support reuses the mechanism the P-ζ pipeline tools already expect: a
``_stop_event`` attribute on the *current thread* (see ``_resolve_should_stop``
in ``src/ui/chat_tools/pipeline.py``). ``StoppableAgentThread`` sets that
attribute on itself so a pipeline started mid-run is interruptible, and the
same event drives ``AgentLoop(should_stop=…)``.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Dict, Optional

from src.core.agent_loop import AgentLoop
from src.core.chat_prompts import DEFAULT_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.core.data_models import AgentResult

logger = logging.getLogger(__name__)


def resolve_provider_model(
    explicit_provider: Optional[str],
    explicit_model: Optional[str],
    *,
    chat_config: Any = None,
    pipeline_manager: Any = None,
    llm_service: Any = None,
) -> "tuple[str, str]":
    """Resolve (provider, model) for a headless agent run.

    Precedence (mirrors PipelineChatPanel + adds the pipeline general default):
      1. explicit args / request body
      2. ``ChatConfig.default_provider/model``
      3. ``pipeline_manager.config.global_provider_override/model``
      4. pipeline general default — ``config.step_configs[*].provider/model``
         (populated from ``pipeline_default_provider`` / first enabled provider
         by ``PipelineConfig.create_from_provider_preferences``). This is the
         "real" default since provider/model are configured per pipeline step,
         not as a standalone ChatConfig default.
      5. ``llm_service.current_provider/model``
    """
    if explicit_provider and explicit_model:
        return explicit_provider, explicit_model

    p = getattr(chat_config, "default_provider", "") or ""
    m = getattr(chat_config, "default_model", "") or ""
    if p and m:
        return p, m

    cfg = getattr(pipeline_manager, "config", None)
    if cfg is not None:
        p = getattr(cfg, "global_provider_override", None) or ""
        m = getattr(cfg, "global_model_override", None) or ""
        if p and m:
            return p, m
        step_configs = getattr(cfg, "step_configs", None) or {}
        # Prefer LLM-bearing steps; fall back to any populated step.
        ordered = ["keywords", "initialisation", "dk_classification"]
        ordered += [k for k in step_configs if k not in ordered]
        for key in ordered:
            sc = step_configs.get(key)
            sp = getattr(sc, "provider", "") or ""
            sm = getattr(sc, "model", "") or ""
            if sp and sm:
                return sp, sm

    p = getattr(llm_service, "current_provider", None) or ""
    m = getattr(llm_service, "current_model", None) or ""
    if p and m:
        return p, m

    return "", ""


def _resolve_kb_manager(pipeline_manager: Any) -> Any:
    """Pull the knowledge manager off the pipeline.

    ``PipelineManager`` stores it as ``cache_manager`` (a
    ``UnifiedKnowledgeManager`` — a singleton). The ``*knowledge_manager``
    fallbacks mirror the GUI's lookup names for forward-compat.
    """
    if pipeline_manager is None:
        return None
    return (
        getattr(pipeline_manager, "unified_knowledge_manager", None)
        or getattr(pipeline_manager, "knowledge_manager", None)
        or getattr(pipeline_manager, "cache_manager", None)
    )


class HeadlessAgentRunner:
    """Builds a session-scoped chat toolset and runs the agent loop.

    Services are injected (the CLI / HTTP layer bootstraps them) — this class
    never touches ConfigManager so it stays trivially testable with a mock
    ``llm_service``.
    """

    def __init__(
        self,
        *,
        llm_service: Any,
        pipeline_manager: Any = None,
        kb_manager: Any = None,
        mcp_registry: Any = None,
        chat_config: Any = None,
        gateway: Any = None,
        system_prompt: Optional[str] = None,
        max_iterations: Optional[int] = None,
        timeout_seconds: int = 600,
    ) -> None:
        self.llm_service = llm_service
        self.pipeline_manager = pipeline_manager
        self.kb_manager = kb_manager if kb_manager is not None else _resolve_kb_manager(pipeline_manager)
        if mcp_registry is None:
            from src.mcp.tool_registry import ToolRegistry
            mcp_registry = ToolRegistry()
            mcp_registry.register_all_tools()
        self.mcp_registry = mcp_registry
        self.chat_config = chat_config
        self.gateway = gateway
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout_seconds = timeout_seconds
        cfg_iter = getattr(chat_config, "max_iterations", None)
        self.max_iterations = max_iterations or cfg_iter or 30

    def run(
        self,
        user_message: str,
        *,
        provider: str,
        model: str,
        context_str: str = "",
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        shared_context: Any = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[Any], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> AgentResult:
        """Run one agent turn and return its :class:`AgentResult`.

        ``context_str`` describes the current work (title/abstract snippet);
        empty means "no work loaded". ``shared_context`` (optional) seeds
        ``ChatSession.last_shared_context`` so generic SharedContext tools see
        pipeline data without a fresh tool call.
        """
        # Local import: src.ui.chat_tools is Qt-free at module level (verified),
        # but kept lazy so importing this module never reaches into src/ui
        # unless a run actually happens.
        from src.ui.chat_session import ChatSession
        from src.ui.chat_tools import build_chat_toolset

        session = ChatSession()
        if shared_context is not None:
            session.last_shared_context = shared_context
        elif self.pipeline_manager is not None:
            session.last_shared_context = getattr(
                self.pipeline_manager, "last_shared_context", None
            )

        registry = build_chat_toolset(
            session=session,
            chat_config=self.chat_config,
            mcp_registry=self.mcp_registry,
            pipeline_manager=self.pipeline_manager,
            kb_manager=self.kb_manager,
            proposal_gateway=self.gateway,
        )

        if temperature is None:
            temperature = float(getattr(self.chat_config, "temperature", 0.5) or 0.5)

        loop = AgentLoop(
            llm_service=self.llm_service,
            tool_registry=registry,
            max_iterations=self.max_iterations,
            timeout_seconds=self.timeout_seconds,
            stream_callback=on_token,
            status_callback=on_status,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            should_stop=should_stop,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context_str or "(kein Werk geladen)",
            user_message=user_message,
        )

        result = loop.run(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            tools=[],  # all registered chat tools
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_history=conversation_history,
        )
        return result


class StoppableAgentThread(threading.Thread):
    """Runs ``target`` on a worker thread that exposes ``_stop_event``.

    The ``_stop_event`` attribute is what the P-ζ pipeline tools read off the
    current thread to cancel a mid-run pipeline (``_resolve_should_stop``).
    ``request_stop()`` also asks the LlmService to abort an in-flight
    generation so cancel latency stays sub-iteration.
    """

    def __init__(self, target: Callable[[Callable[[], bool]], Any], *, llm_service: Any = None) -> None:
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._target = target
        self._llm_service = llm_service
        self.result: Any = None
        self.error: Optional[BaseException] = None

    def run(self) -> None:
        try:
            self.result = self._target(self._stop_event.is_set)
        except BaseException as exc:  # noqa: BLE001 — propagate to caller
            self.error = exc
            logger.exception("StoppableAgentThread target raised")

    def request_stop(self) -> None:
        self._stop_event.set()
        if self._llm_service is not None:
            try:
                self._llm_service.cancel_generation(reason="headless_cancel")
            except Exception:
                logger.debug("cancel_generation raised; ignoring", exc_info=True)
