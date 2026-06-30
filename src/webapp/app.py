"""
ALIMA Webapp - FastAPI Backend
Claude Generated - Pipeline widget as web interface
"""

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import subprocess
import sys

# Add project root to sys.path BEFORE importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Import ALIMA Pipeline components - Claude Generated
# PipelineManager used by the agent-runner helpers below; PipelineConfig kept for
# re-export (tests patch the class method appmod.PipelineConfig.*). - Claude Generated
from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.utils.config_manager import ConfigManager
from src.utils.qt_plugin_setup import setup_qt_plugin_paths, get_available_sql_drivers
from src.webapp.result_serialization import (
    build_export_payload as _build_export_payload,
    ensure_json_serializable as _ensure_json_serializable,
    extract_results_from_analysis_state as _extract_results_from_analysis_state,
)
from src.core.agents.workflow_loader import (
    find_workflow_file,
    load_workflow,
)
# Shared webapp helpers (session_io.py, F-6 split): cleanup used by the lifespan,
# _parse_think_override by the chat endpoint that remain here. - Claude Generated
from src.webapp.session_io import (
    cleanup_old_autosaves,
    _parse_think_override,
)
# APIRouter modules extracted from this file (F-6 split). _discover_workflows is
# re-exported for the unit test that imports it via src.webapp.app. - Claude Generated
from src.webapp.routers import workflows as workflows_router
from src.webapp.routers import models as models_router
from src.webapp.routers import sessions as sessions_router
from src.webapp.routers import export as export_router
from src.webapp.routers import websocket as websocket_router
from src.webapp.routers import analysis as analysis_router
from src.webapp.routers.workflows import _discover_workflows
# run_analysis re-exported for tests that call appmod.run_analysis directly
# (heavy deps mocked via the analysis-router namespace). - Claude Generated
from src.webapp.routers.analysis import run_analysis

# Setup logging - Claude Generated: shared setup (console + alima_webapp.log),
# verbosity still configurable via LOG_LEVEL env var (DEBUG → level 2)
from src.utils.logging_utils import setup_logging

_env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
setup_logging(level=2 if _env_level == "DEBUG" else 1, log_file="alima_webapp.log")
logger = logging.getLogger(__name__)

# Shared webapp state (sessions registry, Session model, lazy AppContext).
# Extracted to session_state.py (F-6 split); imported after setup_logging so the
# module's "Auto-save directory" startup log line is captured. Re-exported for
# tests that import/patch these via ``src.webapp.app``. - Claude Generated
from src.webapp.session_state import (
    AUTOSAVE_ENABLED,
    AUTOSAVE_MAX_AGE_HOURS,
    WEBSOCKET_TIMEOUT_SECONDS,
    WEBSOCKET_HEARTBEAT_INTERVAL,
    AUTOSAVE_DIR,
    sessions,
    AppContext,
    Session,
)
# WP12 render bridge (Session-buffer transport + StateBus→renderer subscriber),
# extracted to render_bridge.py (F-6 split); re-exported for tests that import
# these via ``src.webapp.app``. - Claude Generated
from src.webapp.render_bridge import (
    _HeadlessAutoScroll,
    WebSocketRenderTransport,
    _build_session_renderer,
    _SessionBusSubscriber,
)


class _SuppressSessionPolling(logging.Filter):
    """Filter out high-frequency GET /api/session/{id} polling from access logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("GET /api/session/" in msg and "200" in msg)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Auto-save configuration (AUTOSAVE_ENABLED / AUTOSAVE_MAX_AGE_HOURS /
# WEBSOCKET_TIMEOUT_SECONDS / WEBSOCKET_HEARTBEAT_INTERVAL) now lives in
# session_state.py and is imported above. - Claude Generated

# Lifespan context manager replaces deprecated on_event - Claude Generated
@asynccontextmanager
async def lifespan(app):
    """Startup and shutdown lifecycle - Claude Generated"""
    # Suppress session-polling spam AFTER uvicorn has configured its loggers
    logging.getLogger("uvicorn.access").addFilter(_SuppressSessionPolling())
    logger.info("Starting ALIMA Webapp...")

    try:
        startup_config = ConfigManager(logger=logger).load_config()
        db_cfg = startup_config.database_config
        if db_cfg.db_type.lower() in {"sqlite", "sqlite3"}:
            logger.warning(
                "ALIMA webapp is configured to use SQLite (%s). "
                "This is acceptable for development or light single-process use, "
                "but MariaDB/MySQL is recommended for concurrent multi-user webapp "
                "access and simultaneous CLI usage.",
                db_cfg.sqlite_path,
            )
    except Exception as e:
        logger.warning(f"Could not validate database configuration during startup: {e}")

    # Setup Qt plugin paths for SQL drivers - Claude Generated
    setup_qt_plugin_paths()
    drivers = get_available_sql_drivers()
    if drivers:
        logger.info(f"Available SQL drivers: {', '.join(drivers)}")
    else:
        logger.warning("No SQL drivers found - database operations may fail")

    logger.info(f"Auto-Save: {'Enabled' if AUTOSAVE_ENABLED else 'Disabled'} | Timeout: {WEBSOCKET_TIMEOUT_SECONDS}s | Cleanup: {AUTOSAVE_MAX_AGE_HOURS}h")
    cleanup_old_autosaves()
    logger.info("Webapp initialization complete")
    yield
    logger.info("Shutting down ALIMA Webapp...")
    logger.info("Webapp shutdown complete")


app = FastAPI(title="ALIMA Webapp", description="Pipeline widget as web interface", lifespan=lifespan)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files BEFORE routes - Claude Generated
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup Jinja2 templates for dynamic HTML generation - Claude Generated (2026-01-13)
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# APIRouter modules extracted from this file (F-6 split). Mounted here; their
# paths are unique so registration order does not affect matching. - Claude Generated
app.include_router(workflows_router.router)
app.include_router(models_router.router)
app.include_router(sessions_router.router)
app.include_router(export_router.router)
app.include_router(websocket_router.router)
app.include_router(analysis_router.router)

# Session registry (``sessions``), the ``Session`` model and the lazy
# ``AppContext`` live in session_state.py; the WP12 render bridge classes
# (_HeadlessAutoScroll / WebSocketRenderTransport / _build_session_renderer /
# _SessionBusSubscriber) live in render_bridge.py. Both imported above (F-6
# split). - Claude Generated


@app.get("/")
async def root():
    """Redirect '/' to '/webapp' and set a session cookie."""
    # Create a response that performs the redirect
    response = RedirectResponse(url="/webapp", status_code=301)

    # Set a cookie with a unique session ID (if not already present)
    # Using UUID4 for guaranteed uniqueness
    session_id = uuid.uuid4().hex
    response.set_cookie(
        key="SESSION_ID",
        value=session_id,
        path="/webapp",
        httponly=True,
        secure=True,
        samesite="Lax"
    )
    return response


@app.get("/webapp")
async def get_webapp(request: Request, session: str = None) -> HTMLResponse:
    """Serve webapp with session ID injected - Claude Generated (2026-01-13)

    Each browser tab gets its own HTML page with unique session ID.
    This prevents DOM ID conflicts when multiple tabs are open.

    Usage:
        /webapp              → Creates new session
        /webapp?session=abc  → Uses existing session ID
    """
    if not session:
        session = str(uuid.uuid4())

    # Ensure the session exists in server state so a tab opened with
    # ?session=... can immediately poll/chat without a separate create call.
    # If an existing ID was passed, we keep its state; otherwise we seed it.
    if session not in sessions:
        sessions[session] = Session(session)
        logger.info(f"Created session from webapp route: {session}")

    # Render template with injected session ID
    return templates.TemplateResponse(
        "webapp.html",
        {"request": request, "session_id": session}
    )


# Session lifecycle endpoints (create/get/clear/cancel/abort_step) now live
# in routers/sessions.py (mounted via app.include_router above). - Claude Generated


# GET /api/models + POST /api/models/refresh now live in routers/models.py
# (mounted via app.include_router above). - Claude Generated


# _parse_think_override now lives in session_io.py (shared analysis+agent). - Claude Generated


def _log_chat_turn_safe(runner, session_id, provider, model, message, result, req,
                        tool_log=None) -> None:
    """Best-effort chat-turn logging to the configured SQLite DB - Claude Generated.

    No-op unless ``chat_config.session_log_db`` is set. Never raises. ``tool_log``
    (full args+results from the callbacks) is preferred over the truncated
    ``result.tool_log`` so the DB record is complete for later analysis.
    """
    try:
        chat_config = getattr(runner, "chat_config", None)
        db_path = getattr(chat_config, "session_log_db", "") if chat_config else ""
        if not db_path:
            return
        from src.utils.chat_session_logger import log_chat_turn
        cm = AppContext().get_services()['config_manager']
        cfg_file = getattr(cm, "config_file", None)
        config_dir = str(cfg_file.parent) if cfg_file else None
        log_chat_turn(
            db_path,
            session_id=session_id,
            provider=provider,
            model=model,
            user_message=message,
            response=getattr(result, "content", "") or "",
            tool_log=tool_log if tool_log is not None else getattr(result, "tool_log", None),
            mode=(getattr(req, "mode", None) or "auto"),
            language=(getattr(req, "language", None) or "de"),
            iterations=getattr(result, "iterations", 0),
            stop_reason=getattr(result, "stop_reason", None),
            error=getattr(result, "error", None),
            config_dir=config_dir,
        )
    except Exception as e:
        logger.warning(f"chat-turn logging skipped: {e}")


# POST /api/analyze/{id} now lives in routers/analysis.py. - Claude Generated


# POST /api/input/{id} now lives in routers/analysis.py. - Claude Generated




# Session serialization + auto-save helpers (make_json_serializable,
# sanitize_filename, _autosave_session_state, cleanup_old_autosaves) now live
# in session_io.py and are imported above. - Claude Generated



# GET /api/queue/status now lives in routers/models.py. - Claude Generated


# WS /ws/{id} (live progress) now lives in routers/websocket.py. - Claude Generated


# GET /api/export/{id} now lives in routers/export.py. - Claude Generated


# Workflow discovery + GET /api/workflows now live in
# routers/workflows.py (mounted via app.include_router above). - Claude Generated

class AgentRunRequest(BaseModel):
    """Body for POST /agent/run — Claude Generated (P-ι)."""
    input: Dict[str, Any] = {}
    prompt: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_iterations: Optional[int] = None
    mode: Optional[str] = None  # "verschlagwortung" | "suche" | "general" | "auto"
    autonomous: bool = False
    stream: bool = True


class ChatMessageRequest(BaseModel):
    """Body for POST /api/session/{id}/chat — Claude Generated."""
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    mode: Optional[str] = None  # auto | verschlagwortung | suche | general
    language: Optional[str] = None  # "de" (default) | "en" — reply language
    think: Optional[str] = None  # "default" | "on" | "off" — thinking override


def _agent_context_from_input(input_data: Dict[str, Any]) -> str:
    """Resolve a work-context string from the request's ``input`` block.

    Accepts ``{"abstract": "..."}`` / ``{"text": "..."}`` directly, or
    ``{"doi": "10.x/y"}`` which is resolved to text. Empty ⇒ no work loaded.
    """
    text = input_data.get("abstract") or input_data.get("text") or ""
    if not text and input_data.get("doi"):
        from src.utils.doi_resolver import resolve_input_to_text
        success, resolved, error = resolve_input_to_text(input_data["doi"], logger)
        if not success:
            raise ValueError(f"DOI resolution failed: {error}")
        text = resolved or ""
    return (text[:1500] + "…") if len(text) > 1500 else text


def _build_agent_runner(req: AgentRunRequest):
    """Per-request HeadlessAgentRunner with an isolated PipelineManager.

    A fresh PipelineManager avoids cross-request pipeline-state collisions
    while still sharing the AlimaManager (LLM concurrency semaphore) and the
    singleton knowledge DB.
    """
    from src.core.headless_agent import HeadlessAgentRunner, resolve_provider_model
    from src.core.headless_gateway import AutoRejectGateway
    from src.utils.config_models import ChatConfig

    services = AppContext().get_services()
    cm = services['config_manager']
    try:
        # chat_config lives on AlimaConfig (load_config), not the unified config.
        chat_config = cm.load_config().chat_config
    except Exception:
        chat_config = ChatConfig()

    pm = PipelineManager(
        alima_manager=services['alima_manager'],
        cache_manager=services['cache_manager'],
        logger=logger,
        config_manager=cm,
    )

    gateway = None
    if req.autonomous:
        chat_config.autonomous_pipeline = True
    else:
        gateway = AutoRejectGateway()

    provider, model = resolve_provider_model(
        req.provider, req.model,
        chat_config=chat_config, pipeline_manager=pm, llm_service=services['llm_service'],
    )
    if not provider or not model:
        raise ValueError(
            "No provider/model — set in request body, ChatConfig defaults, "
            "or pipeline_default_provider/model in config"
        )

    runner = HeadlessAgentRunner(
        llm_service=services['llm_service'],
        pipeline_manager=pm,
        chat_config=chat_config,
        gateway=gateway,
        mode=req.mode or "auto",
        max_iterations=req.max_iterations,
    )
    return runner, pm, provider, model


def _agent_done_payload(result, pipeline_manager, provider: str, model: str, autonomous: bool) -> dict:
    state = getattr(pipeline_manager, "current_analysis_state", None)
    results = _extract_results_from_analysis_state(state) if state else None
    payload = _build_export_payload(
        session_id="http-agent-" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        created_at=datetime.now().isoformat(),
        status="completed",
        current_step=None,
        input_data=None,
        results=results,
    )
    payload["agent"] = {
        "final_content": result.content,
        "iterations": result.iterations,
        "tool_log": result.tool_log,
        "provider": provider,
        "model": model,
        "autonomous": autonomous,
    }
    return payload


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    """Run the headless chat-agent. SSE stream (default) or single JSON.

    Permission: non-autonomous requests cannot prompt over SSE, so any
    confirmation-gated op is auto-rejected (AutoRejectGateway). Set
    ``autonomous: true`` to let the agent apply mutations / start pipelines.
    """
    import queue as _queue
    from src.core.headless_agent import StoppableAgentThread

    try:
        runner, pm, provider, model = _build_agent_runner(req)
        context_str = _agent_context_from_input(req.input or {})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user_message = req.prompt or (
        "Analysiere das vorliegende Werk und schlage GND-Schlagwörter sowie "
        "DK-Klassifikationen vor." if context_str else
        "Was kannst du tun? Liste verfügbare Daten und Werkzeuge."
    )
    services = AppContext().get_services()

    def _run(should_stop):
        return runner.run(
            user_message,
            provider=provider,
            model=model,
            context_str=context_str,
            temperature=req.temperature,
            on_token=lambda t: _q.put(("token", t)),
            on_status=lambda s: _q.put(("status", s)),
            on_tool_call=lambda tc: _q.put(("tool_call", {"name": tc.name, "arguments": tc.arguments})),
            on_tool_result=lambda name, res: _q.put(("tool_result", {"name": name, "result": res[:500]})),
            should_stop=should_stop,
        )

    # --- non-streaming: collect synchronously off the event loop ----------
    if not req.stream:
        _q = _queue.Queue()  # discarded; callbacks fire but we ignore them

        def _blocking():
            th = StoppableAgentThread(_run, llm_service=services['llm_service'])
            th.start()
            th.join()
            if th.error:
                raise th.error
            return th.result

        result = await asyncio.get_event_loop().run_in_executor(None, _blocking)
        return JSONResponse(_agent_done_payload(result, pm, provider, model, req.autonomous))

    # --- streaming: SSE -------------------------------------------------
    _q = _queue.Queue()

    def event_stream():
        def _sse(etype: str, data: Any) -> str:
            return f"data: {json.dumps({'type': etype, 'data': data}, ensure_ascii=False, default=str)}\n\n"

        thread = StoppableAgentThread(
            _wrap_agent_target(_run, _q),
            llm_service=services['llm_service'],
        )
        thread.start()
        try:
            while True:
                etype, data = _q.get()
                if etype == "__result__":
                    yield _sse("done", _agent_done_payload(data, pm, provider, model, req.autonomous))
                    break
                if etype == "__error__":
                    yield _sse("error", str(data))
                    break
                yield _sse(etype, data)
        finally:
            thread.request_stop()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _wrap_agent_target(run_callable, q):
    """Run the agent, routing the result or any exception onto the queue."""
    def _target(should_stop):
        try:
            result = run_callable(should_stop)
            q.put(("__result__", result))
        except BaseException as exc:  # noqa: BLE001
            q.put(("__error__", exc))
    return _target


def _shared_context_from_analysis_state(state) -> Optional[Any]:
    """Bridge a PipelineManager analysis_state into a SharedContext for chat tools.

    Returns None if the bridge cannot be built (e.g. state missing). - Claude Generated
    """
    if state is None:
        return None
    try:
        from src.core.agents.shared_context import SharedContext
        return SharedContext.from_keyword_analysis_state(state)
    except Exception as e:
        logger.debug(f"Could not build SharedContext from analysis state: {e}")
        return None


def _build_session_agent_runner(session: "Session", req: ChatMessageRequest):
    """Per-session HeadlessAgentRunner wired to the session's pipeline context. - Claude Generated"""
    from src.core.headless_agent import HeadlessAgentRunner, resolve_provider_model
    from src.core.headless_gateway import AutoRejectGateway
    from src.utils.config_models import ChatConfig

    services = AppContext().get_services()
    cm = services['config_manager']
    try:
        chat_config = cm.load_config().chat_config
    except Exception:
        chat_config = ChatConfig()

    # Reuse the session's pipeline manager reference if a pipeline ran here;
    # otherwise create an isolated one so chat tools still see a PM.
    pm = session.pipeline_manager_ref
    if pm is None:
        pm = PipelineManager(
            alima_manager=services['alima_manager'],
            cache_manager=services['cache_manager'],
            logger=logger,
            config_manager=cm,
        )
        # Seed the isolated PM with the session analysis state so chat tools
        # can answer questions about the just-finished pipeline results.
        if session.current_analysis_state is not None:
            pm.current_analysis_state = session.current_analysis_state

    # Chat-agent in the webapp session is never autonomous in round 1:
    # destructive mutations require explicit confirmation via the UI.
    gateway = AutoRejectGateway()

    # Prefer explicit request values, then the configured chat default
    # (ChatConfig.default_provider/model — a config-only webapp setting), then
    # the session's last effective pipeline provider/model, then the usual
    # unified-config fallbacks inside resolve_provider_model. - Claude Generated
    chat_provider = req.provider or chat_config.default_provider or session.last_provider
    chat_model = req.model or chat_config.default_model or session.last_model
    provider, model = resolve_provider_model(
        chat_provider, chat_model,
        chat_config=chat_config, pipeline_manager=pm, llm_service=services['llm_service'],
    )
    if not provider or not model:
        raise ValueError(
            "No provider/model — set in request body, ChatConfig defaults, "
            "or pipeline_default_provider/model in config"
        )

    runner = HeadlessAgentRunner(
        llm_service=services['llm_service'],
        pipeline_manager=pm,
        chat_config=chat_config,
        gateway=gateway,
        mode=req.mode or "auto",
        max_iterations=30,
    )
    return runner, pm, provider, model


@app.post("/api/session/{session_id}/chat")
async def session_chat(session_id: str, req: ChatMessageRequest) -> dict:
    """Send a chat message within a session and stream the agent turn via WebSocket.

    The assistant reply is rendered into the same session render_buffer that
    pipelines use, so the unified log/chat panel shows one continuous conversation.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    try:
        runner, pm, provider, model = _build_session_agent_runner(session, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Persist user turn in session history.
    session.chat_history.append({"role": "user", "content": message})

    # Renderer writes into the same append-only buffer the WebSocket broadcasts.
    session_renderer = _build_session_renderer(session)
    # Phase 4: bridge AlimaStateBus events (extra tool calls, pipeline state)
    # during the chat turn into the same render buffer.
    bus_subscriber = _SessionBusSubscriber(session_renderer)
    bus_subscriber.subscribe()

    session_renderer.render_user_bubble(message)
    session_renderer.show_typing(f"{provider} | {model}")
    session_renderer.open_assistant_bubble(f"{provider} | {model}")

    # Keep the WebSocket alive while the chat agent is streaming. The WS loop
    # terminates as soon as the session leaves running/idle, so after a pipeline
    # run the connection would otherwise drop before any chat tokens arrive.
    session.status = "running"

    # Derive work context from the session's pipeline result, if any.
    context_str = ""
    state = getattr(pm, "current_analysis_state", None) or session.current_analysis_state
    shared_context = _shared_context_from_analysis_state(state)
    if state is not None:
        wt = getattr(state, "working_title", "") or ""
        ab = getattr(state, "original_abstract", "") or ""
        parts = []
        if wt:
            parts.append(f"Titel: {wt}")
        if ab:
            parts.append(f"Abstract: {ab[:500]}{'…' if len(ab) > 500 else ''}")
        context_str = "\n".join(parts)

    services = AppContext().get_services()

    # Fresh turn → clear any stale cancel flag from a previous run.
    session.abort_requested = False

    # Track open tool-call ids so on_tool_result can close the matching block.
    _open_tool_ids: list = []
    # Full tool calls (args + untruncated results) for the chat-session DB log,
    # which the rendered preview (res[:2000]) would otherwise lose. - Claude Generated
    _tool_calls_full: list = []

    def run_chat_turn(should_stop):
        try:
            def _on_tool_call(tc):
                tool_id = session_renderer.render_tool_call(
                    tc.name, tc.arguments or {}
                )
                _open_tool_ids.append(tool_id)
                _tool_calls_full.append({
                    "name": getattr(tc, "name", ""),
                    "arguments": dict(getattr(tc, "arguments", {}) or {}),
                    "result": None,
                })

            def _on_tool_result(_name, res):
                # Trust tool-result URLs so pre-formatted GND/catalog links
                # aren't flagged external (GUI parity). Claude Generated.
                try:
                    import json as _json
                    from src.core.url_utils import extract_urls_from_json
                    session_renderer.add_trusted_urls(
                        extract_urls_from_json(_json.loads(res))
                    )
                except Exception:
                    pass
                # Record the full result for the DB log (calls/results are
                # sequential, so the open entry is the most recent one).
                for entry in reversed(_tool_calls_full):
                    if entry["result"] is None:
                        entry["result"] = res
                        break
                tool_id = _open_tool_ids.pop() if _open_tool_ids else None
                if tool_id:
                    session_renderer.render_tool_result(
                        tool_id, res[:2000], status="success"
                    )
                else:
                    session_renderer.render_system_message(
                        f"↳ {_name}: {res[:120]}"
                    )

            # Last-N window of *prior* turns (exclude the just-appended current
            # message; AgentLoop re-adds it as the user prompt). history_truncated
            # tells the model older turns exist. - Claude Generated
            from src.core.chat_prompts import CHAT_HISTORY_WINDOW
            prior_history = list(session.chat_history)[:-1]
            recent_history = prior_history[-CHAT_HISTORY_WINDOW:]
            history_truncated = len(prior_history) > len(recent_history)

            result = runner.run(
                message,
                provider=provider,
                model=model,
                context_str=context_str,
                temperature=req.temperature,
                max_tokens=getattr(runner.chat_config, "max_tokens", 4096),
                conversation_history=recent_history,
                shared_context=shared_context,
                think=_parse_think_override(req.think),
                language=(req.language or "de"),
                history_truncated=history_truncated,
                on_token=lambda t: session_renderer.append_assistant_token(t),
                on_status=lambda s: session_renderer.render_pipeline_log(s, "debug"),
                on_tool_call=_on_tool_call,
                on_tool_result=_on_tool_result,
                should_stop=should_stop,
            )
            # Persist assistant turn and tool log in session history.
            content = getattr(result, "content", "") or ""
            if content:
                session.chat_history.append({"role": "assistant", "content": content})
            session_renderer.finalize_assistant_bubble()

            # Config-only chat-session DB logging (no UI). Pass the full
            # tool calls+results captured via the callbacks (untruncated). - Claude Generated
            _log_chat_turn_safe(
                runner, session_id, provider, model, message, result, req,
                tool_log=_tool_calls_full,
            )
            if should_stop():
                session_renderer.render_system_message("⏹ Chat abgebrochen")
            else:
                session_renderer.render_system_message("✅ Assistant-Antwort abgeschlossen")
            session.status = "idle"
            return result
        except Exception as e:
            logger.exception("Session chat turn failed")
            session_renderer.render_system_message(f"❌ Chat-Fehler: {e}")
            session.status = "error"
            session.error_message = str(e)
            raise
        finally:
            session_renderer.hide_typing()
            session.chat_thread = None
            try:
                bus_subscriber.unsubscribe()
            except Exception:
                logger.exception("Failed to unsubscribe chat bus subscriber")

    # Run on a StoppableAgentThread so /cancel can abort the turn (parity with
    # /agent/run and the GUI ChatAgentWorker). The thread exposes _stop_event,
    # which nested pipeline tools also read for mid-run cancellation. Callbacks
    # fire on the worker thread and append to the lock-protected render buffer.
    from src.core.headless_agent import StoppableAgentThread
    chat_thread = StoppableAgentThread(
        run_chat_turn, llm_service=services['llm_service']
    )
    session.chat_thread = chat_thread
    chat_thread.start()

    return {
        "session_id": session_id,
        "status": "chat_started",
        "provider": provider,
        "model": model,
    }


# GET /api/session/{id}/recover now lives in routers/sessions.py. - Claude Generated


# run_analysis now lives in routers/analysis.py. - Claude Generated


# run_input_extraction now lives in routers/analysis.py. - Claude Generated


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint - Claude Generated"""
    return {"status": "ok", "active_sessions": len(sessions)}


if __name__ == "__main__":
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
