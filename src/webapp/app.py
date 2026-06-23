"""
ALIMA Webapp - FastAPI Backend
Claude Generated - Pipeline widget as web interface
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import subprocess
import sys

# Add project root to sys.path BEFORE importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import yaml

# Import ALIMA Pipeline components - Claude Generated
from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.core.alima_manager import AlimaManager
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.utils.config_manager import ConfigManager
from src.utils.doi_resolver import UnifiedResolver, _get_doi_config, format_doi_metadata, resolve_input_to_text
from src.utils.pipeline_utils import PipelineJsonManager, PipelineResultFormatter
from src.utils.qt_plugin_setup import setup_qt_plugin_paths, get_available_sql_drivers
from src.webapp.result_serialization import (
    build_export_payload as _build_export_payload,
    ensure_json_serializable as _ensure_json_serializable,
    extract_results_from_analysis_state as _extract_results_from_analysis_state,
    prepare_results_for_export as _prepare_results_for_export,
)
from src.core.agents.workflow_loader import (
    DEFAULT_SEARCH_PATHS,
    find_workflow_file,
    load_workflow,
)

# Setup logging - Claude Generated: shared setup (console + alima_webapp.log),
# verbosity still configurable via LOG_LEVEL env var (DEBUG → level 2)
from src.utils.logging_utils import setup_logging

_env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
setup_logging(level=2 if _env_level == "DEBUG" else 1, log_file="alima_webapp.log")
logger = logging.getLogger(__name__)


class _SuppressSessionPolling(logging.Filter):
    """Filter out high-frequency GET /api/session/{id} polling from access logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("GET /api/session/" in msg and "200" in msg)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Auto-Save Configuration - Claude Generated (2026-01-06)
# These settings control the auto-save and recovery system
AUTOSAVE_ENABLED = True  # Enable/disable auto-save system
AUTOSAVE_MAX_AGE_HOURS = 24  # Auto-cleanup files older than this (hours)
WEBSOCKET_TIMEOUT_SECONDS = 1800  # WebSocket idle timeout (30 minutes = 1800s)
WEBSOCKET_HEARTBEAT_INTERVAL = 5  # Heartbeat interval in seconds (5s)

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

# Auto-save directory for session recovery - Claude Generated
AUTOSAVE_DIR = Path(tempfile.gettempdir()) / "alima_webapp_autosave"
AUTOSAVE_DIR.mkdir(exist_ok=True)
logger.info(f"Auto-save directory: {AUTOSAVE_DIR}")

# Store active sessions and their results
sessions: dict = {}


class AppContext:
    """Global application context with lazy-initialized services - Claude Generated"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init_services(self):
        """Initialize ALIMA services on first use"""
        if self._initialized:
            return

        logger.info("Initializing ALIMA services...")

        # Step 1: ConfigManager (load config.json)
        self.config_manager = ConfigManager(logger=logger)
        config = self.config_manager.load_config()
        prompts_path = config.system_config.prompts_path

        # Step 2: LlmService (with lazy initialization for webapp responsiveness)
        self.llm_service = LlmService(
            config_manager=self.config_manager,
            lazy_initialization=True
        )

        # Step 3: PromptService (load prompts.json)
        self.prompt_service = PromptService(prompts_path, logger=logger)

        # Step 4: AlimaManager (core business logic)
        self.alima_manager = AlimaManager(
            llm_service=self.llm_service,
            prompt_service=self.prompt_service,
            config_manager=self.config_manager,
            logger=logger
        )

        # Step 5: UnifiedKnowledgeManager (singleton database)
        self.cache_manager = UnifiedKnowledgeManager()

        # Step 6: PipelineManager (pipeline orchestration)
        self.pipeline_manager = PipelineManager(
            alima_manager=self.alima_manager,
            cache_manager=self.cache_manager,
            logger=logger,
            config_manager=self.config_manager
        )

        AppContext._initialized = True
        logger.info("✅ ALIMA services initialized")

    def get_services(self):
        """Get or initialize services"""
        if not self._initialized:
            self.init_services()
        return {
            'config_manager': self.config_manager,
            'llm_service': self.llm_service,
            'prompt_service': self.prompt_service,
            'alima_manager': self.alima_manager,
            'cache_manager': self.cache_manager,
            'pipeline_manager': self.pipeline_manager
        }


class Session:
    """Represents an analysis session - Claude Generated"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now().isoformat()
        self.status = "idle"  # idle, running, completed, error
        self.current_step = None
        self.current_step_status = None  # 'running' or 'completed' - Claude Generated
        self.input_data = None
        self.results = {}
        self.error_message = None
        self.process = None
        self.temp_files = []
        self.streaming_buffer = {}  # Buffer for streaming tokens by step_id - Claude Generated
        self.streaming_buffer_sent_count = {}  # Track how many tokens sent per step - Claude Generated
        self._streaming_lock = threading.Lock()  # Thread-safe access to streaming buffers - Claude Generated
        # WP12: append-only render-event log for the shared chrome (DK/GND
        # cards). Broadcast over the WS (per-connection replay on reconnect) and
        # surfaced to polling clients via render_buffer_sent_count. - Claude Generated
        self.render_buffer = []
        self.render_buffer_sent_count = 0  # cursor for polling clients
        self.abort_requested = False  # Flag to signal pipeline abort - Claude Generated
        # Auto-save support - Claude Generated
        self.autosave_path = AUTOSAVE_DIR / f"session_{session_id}.json"
        self.autosave_enabled = AUTOSAVE_ENABLED  # Use global config
        self.autosave_failed = False
        self.autosave_timestamp = None  # Last auto-save timestamp for status indicator
        self.current_analysis_state = None  # Reference to PipelineManager state
        self.working_title = None  # Working title from initialisation step - Claude Generated
        self.dk_search_progress = None  # DK search progress info (current/total/percent) - Claude Generated
        self.pipeline_manager_ref = None  # Reference for step-abort - Claude Generated
        self.chat_thread = None  # Running chat-agent StoppableAgentThread (for cancel) - Claude Generated
        self.workflow_name = None  # Selected workflow for this session - Claude Generated
        self.last_provider: Optional[str] = None  # Effective provider from last pipeline run - Claude Generated
        self.last_model: Optional[str] = None  # Effective model from last pipeline run - Claude Generated
        self.chat_history: list = []  # Chat-agent conversation history - Claude Generated

    def add_temp_file(self, path: str):
        """Track temporary files for cleanup - Claude Generated"""
        self.temp_files.append(path)

    def add_streaming_token(self, token: str, step_id: str):
        """Add token to streaming buffer - Thread-safe - Claude Generated"""
        with self._streaming_lock:
            if step_id not in self.streaming_buffer:
                self.streaming_buffer[step_id] = []
            self.streaming_buffer[step_id].append(token)

    def get_and_clear_streaming_buffer(self) -> dict:
        """Get all buffered tokens and clear - Thread-safe - Claude Generated"""
        with self._streaming_lock:
            result = dict(self.streaming_buffer)
            self.streaming_buffer.clear()
            self.streaming_buffer_sent_count.clear()
            return result

    def get_new_streaming_tokens(self) -> dict:
        """Get only newly added tokens since last retrieval - Thread-safe - Claude Generated"""
        with self._streaming_lock:
            result = {}
            for step_id, tokens in self.streaming_buffer.items():
                sent_count = self.streaming_buffer_sent_count.get(step_id, 0)
                new_tokens = tokens[sent_count:]
                if new_tokens:
                    result[step_id] = new_tokens
                    self.streaming_buffer_sent_count[step_id] = len(tokens)
            return result

    def append_render_event(self, event: dict):
        """Append a WP12 render event to the per-session log - Thread-safe.

        Stamps each event with a monotonic ``seq`` (its buffer index) so a
        client can dedup across WS-reconnect replay / polling re-delivery.
        """
        with self._streaming_lock:
            event = {**event, "seq": len(self.render_buffer)}
            self.render_buffer.append(event)

    def get_render_events_since(self, index: int):
        """Return (events_since_index, new_length) - for WS replay/incremental."""
        with self._streaming_lock:
            return list(self.render_buffer[index:]), len(self.render_buffer)

    def get_new_render_events(self) -> list:
        """Return render events not yet sent to a polling client - Thread-safe."""
        with self._streaming_lock:
            new = list(self.render_buffer[self.render_buffer_sent_count:])
            self.render_buffer_sent_count = len(self.render_buffer)
            return new

    def clear(self):
        """Complete session reset - clear all data - Claude Generated"""
        self.status = "idle"
        self.current_step = None
        self.current_step_status = None
        self.input_data = None
        self.results = {}
        self.error_message = None
        with self._streaming_lock:  # Thread-safe buffer clearing - Claude Generated
            self.streaming_buffer.clear()
            self.streaming_buffer_sent_count.clear()  # Reset token tracking - Claude Generated
            self.render_buffer.clear()  # WP12: reset render-event log - Claude Generated
            self.render_buffer_sent_count = 0
        self.abort_requested = False
        self.workflow_name = None
        self.chat_history = []
        self.cleanup()
        logger.info(f"Session {self.session_id} cleared")

    def cleanup(self):
        """Clean up temporary files - Claude Generated"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not cleanup {temp_file}: {e}")
        self.temp_files.clear()


class _HeadlessAutoScroll:
    """Minimal QCheckBox stand-in for the headless UnifiedMessageRenderer (WP12).

    The renderer's only use of the checkbox is ``isChecked()`` plus a
    ``toggled.connect(...)`` wire-up; neither matters server-side.
    """

    class _Signal:
        def connect(self, *_args, **_kwargs):  # noqa: D401
            pass

    def __init__(self):
        self.toggled = self._Signal()

    def isChecked(self) -> bool:  # noqa: N802 (Qt-style name)
        return False


class WebSocketRenderTransport:
    """RenderTransport that appends render events to a Session buffer (WP12).

    The WebSocket handler broadcasts the buffer to connected clients with replay
    on reconnect. Control ops (autoscroll/scroll) are server-side no-ops.
    """

    def __init__(self, session: "Session"):
        self._session = session

    def send(self, event: dict) -> None:
        self._session.append_render_event(event)

    def set_autoscroll(self, enabled: bool) -> None:
        pass

    def scroll_to_bottom(self) -> None:
        pass


def _build_session_renderer(session: "Session"):
    """Per-session UnifiedMessageRenderer wired to a WebSocket render transport.

    WP12: the webapp drives the *same* producer as the GUI (single source of the
    chrome), but the events flow over the WebSocket instead of runJavaScript.
    The import is lazy so QtWidgets is only pulled in when a pipeline runs.
    """
    from src.ui.unified_message_renderer import UnifiedMessageRenderer

    renderer = UnifiedMessageRenderer(
        WebSocketRenderTransport(session), _HeadlessAutoScroll()
    )
    # Webapp parity with the GUI panel: wire the catalog web-OPAC base + hosts so
    # <<CAT:rsn|…>> markers become links here too. Degrades silently. Claude Generated.
    try:
        from src.utils.config_manager import ConfigManager
        renderer.configure_catalog_from_config(
            ConfigManager().get_catalog_config()
        )
    except Exception:
        pass
    return renderer


class _SessionBusSubscriber:
    """Session-local AlimaStateBus subscriber for agentic pipeline + chat logs.

    Bridges ``tool.called``, ``tool.result`` and ``state.pipeline_*`` bus events
    into the per-session :class:`UnifiedMessageRenderer`. The webapp has no Qt
    event loop, so the bus falls back to direct handler dispatch; this class is
    therefore instantiated per-run and unsubscribed when the run ends to avoid
    leaking handlers on the singleton bus.

    Claude Generated (Phase 4).
    """

    def __init__(self, renderer):
        self._renderer = renderer
        self._bus = None
        # Pipeline-step block state
        self._step_tool_id: Optional[str] = None
        self._open_step_status: List[str] = []
        self._pipeline_step_open: bool = False
        # Agentic prompt collapsible state
        self._prompt_blocks: Dict[str, str] = {}
        self._prompt_meta: Dict[str, str] = {}
        # Bus tool-call id -> renderer tool-call id
        self._bus_tool_ids: Dict[str, str] = {}

        # Store bound handlers so subscribe/unsubscribe pairs match by identity.
        self._on_tool_called = self._handle_tool_called
        self._on_tool_result = self._handle_tool_result
        self._on_pipeline_step = self._handle_pipeline_step
        self._on_pipeline_prompt = self._handle_pipeline_prompt
        self._on_pipeline_prompt_done = self._handle_pipeline_prompt_done
        self._on_pipeline_completed = self._handle_pipeline_completed
        self._on_pipeline_started = self._handle_pipeline_started

    def subscribe(self) -> None:
        from src.core.state_bus import AlimaStateBus

        self._bus = AlimaStateBus()
        self._bus.subscribe("tool.called", self._on_tool_called)
        self._bus.subscribe("tool.result", self._on_tool_result)
        self._bus.subscribe("state.pipeline_step", self._on_pipeline_step)
        self._bus.subscribe("state.pipeline_prompt", self._on_pipeline_prompt)
        self._bus.subscribe(
            "state.pipeline_prompt_done", self._on_pipeline_prompt_done
        )
        self._bus.subscribe("state.pipeline_completed", self._on_pipeline_completed)
        self._bus.subscribe("state.pipeline_started", self._on_pipeline_started)

    def unsubscribe(self) -> None:
        if self._bus is None:
            return
        try:
            self._bus.unsubscribe("tool.called", self._on_tool_called)
            self._bus.unsubscribe("tool.result", self._on_tool_result)
            self._bus.unsubscribe("state.pipeline_step", self._on_pipeline_step)
            self._bus.unsubscribe("state.pipeline_prompt", self._on_pipeline_prompt)
            self._bus.unsubscribe(
                "state.pipeline_prompt_done", self._on_pipeline_prompt_done
            )
            self._bus.unsubscribe(
                "state.pipeline_completed", self._on_pipeline_completed
            )
            self._bus.unsubscribe("state.pipeline_started", self._on_pipeline_started)
        except Exception:
            logger.exception("SessionBusSubscriber unsubscribe failed")
        finally:
            self._bus = None
            self._step_tool_id = None
            self._open_step_status.clear()
            self._pipeline_step_open = False
            self._prompt_blocks.clear()
            self._prompt_meta.clear()
            self._bus_tool_ids.clear()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_tool_called(self, payload: Dict[str, Any]) -> None:
        bus_id = (payload or {}).get("id") or ""
        name = (payload or {}).get("name") or ""
        args = (payload or {}).get("arguments") or {}
        tool_id = self._renderer.render_tool_call(name, args)
        if bus_id:
            self._bus_tool_ids[bus_id] = tool_id

    def _handle_tool_result(self, payload: Dict[str, Any]) -> None:
        bus_id = (payload or {}).get("id") or ""
        result = (payload or {}).get("result") or ""
        raw_status = (payload or {}).get("status") or "success"
        status = "success" if raw_status == "ok" else raw_status

        # Register tool-result URLs as trusted so pre-formatted GND/catalog links
        # aren't flagged as external (GUI parity). Claude Generated.
        try:
            import json as _json
            from src.core.url_utils import extract_urls_from_json
            self._renderer.add_trusted_urls(
                extract_urls_from_json(_json.loads(result))
            )
        except Exception:
            pass

        if (payload or {}).get("cache_hit"):
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"
            result = f"📦 cache: {preview}"

        tool_id = self._bus_tool_ids.pop(bus_id, None) if bus_id else None
        if tool_id:
            self._renderer.render_tool_result(
                tool_id, result, status=status or "success"
            )
        else:
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "…"
            self._renderer.render_system_message(f"↳ {preview}")

    def _handle_pipeline_step(self, payload: Dict[str, Any]) -> None:
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
            self._step_tool_id = self._renderer.render_tool_call(tool_name, args)
            self._open_step_status.clear()
            self._pipeline_step_open = True
            return

        result_status = "success" if status == "completed" else "error"
        if self._open_step_status:
            result_text = "\n".join(self._open_step_status)
        else:
            result_text = f"{status or 'done'}: {name}"
        self._open_step_status.clear()
        self._pipeline_step_open = False

        if self._step_tool_id:
            self._renderer.render_tool_result(
                self._step_tool_id, result_text, status=result_status
            )
            self._step_tool_id = None
        else:
            tid = self._renderer.render_tool_call(tool_name, args)
            self._renderer.render_tool_result(tid, result_text, status=result_status)

    def _handle_pipeline_prompt(self, payload: Dict[str, Any]) -> None:
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
        meta = "  ".join(meta_parts)

        if kind == "reflection":
            icon, title = "🔍", f"Reflexion '{step_id}'"
        else:
            icon, title = "📥", f"Input '{step_id}'"

        body = f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}"
        # Close any open stream block so the collapsible sits on its own line.
        self._renderer.end_streaming_line()
        tool_id = self._renderer.render_collapsible(
            title, body, collapsed=True, icon=icon, meta=meta
        )
        if prompt_id:
            self._prompt_blocks[prompt_id] = tool_id
            self._prompt_meta[prompt_id] = meta

    def _handle_pipeline_prompt_done(self, payload: Dict[str, Any]) -> None:
        prompt_id = str(payload.get("prompt_id", "") or "")
        dur = payload.get("duration_s")
        tool_id = self._prompt_blocks.get(prompt_id)
        if tool_id is None or dur is None:
            return
        base = self._prompt_meta.get(prompt_id, "")
        meta = f"{base}  ⏱ {float(dur):.1f}s" if base else f"⏱ {float(dur):.1f}s"
        self._renderer.update_collapsible_meta(tool_id, meta)

    def _handle_pipeline_completed(self, payload: Dict[str, Any]) -> None:
        label = "✅ Pipeline abgeschlossen"
        workflow = (payload or {}).get("workflow")
        if workflow:
            label += f" ({workflow})"
        self._renderer.render_system_message(label)

    def _handle_pipeline_started(self, payload: Dict[str, Any]) -> None:
        pid = (payload or {}).get("pipeline_id", "") or ""
        self._renderer.render_system_message(
            f"🚀 Pipeline gestartet{f' ({pid[:8]})' if pid else ''}"
        )


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


@app.post("/api/session")
async def create_session() -> dict:
    """Create a new analysis session - Claude Generated"""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = Session(session_id)
    logger.info(f"Created session: {session_id}")
    return {"session_id": session_id, "status": "created"}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get session status - Claude Generated"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    # Get only new streaming tokens since last retrieval - Claude Generated
    # This prevents tokens from being lost when session is still running
    if session.status == "running":
        streaming_tokens = session.get_new_streaming_tokens()
    else:
        # Session finished, return all remaining unsent tokens
        streaming_tokens = session.get_and_clear_streaming_buffer()

    return {
        "session_id": session.session_id,
        "status": session.status,
        "current_step": session.current_step,
        "created_at": session.created_at,
        "results": _prepare_results_for_export(session.results, validate_rvk=False),
        "error_message": session.error_message,
        "streaming_tokens": streaming_tokens,  # Include for polling clients
        "render_events": session.get_new_render_events(),  # WP12: shared chrome
    }


@app.post("/api/session/{session_id}/clear")
async def clear_session(session_id: str) -> dict:
    """Clear session state and reset for new analysis - Claude Generated"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    session.clear()

    return {
        "session_id": session_id,
        "status": "cleared",
        "message": "Session cleared and reset"
    }


@app.post("/api/session/{session_id}/cancel")
async def cancel_session(session_id: str) -> dict:
    """Request cancellation of running pipeline - Claude Generated"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    if session.status == "running":
        session.abort_requested = True
        # Stop a running chat-agent turn (aborts the AgentLoop + in-flight LLM
        # generation). Pipeline runs read abort_requested separately. - Claude Generated
        chat_thread = getattr(session, "chat_thread", None)
        if chat_thread is not None:
            try:
                chat_thread.request_stop()
            except Exception:
                logger.exception("Failed to request chat-thread stop")
        logger.info(f"Cancellation requested for session {session_id}")
        return {
            "session_id": session_id,
            "status": "cancel_requested",
            "message": "Cancellation requested"
        }
    else:
        return {
            "session_id": session_id,
            "status": session.status,
            "message": "Session is not running"
        }


@app.post("/api/session/{session_id}/abort_step")
async def abort_current_step_endpoint(session_id: str) -> dict:
    """Abort only the current LLM generation; pipeline continues - Claude Generated"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    pm = session.pipeline_manager_ref  # Local ref to avoid race condition
    if session.status == "running" and pm is not None:
        pm.abort_current_step()
        logger.info(f"Step-abort requested for session {session_id}")
        return {"session_id": session_id, "status": "step_abort_requested",
                "message": "Current LLM step will be aborted; pipeline continues"}
    return {"session_id": session_id, "status": session.status,
            "message": "No active LLM step to abort"}


@app.get("/api/models")
async def get_available_models() -> list:
    """Get available provider/model combinations for override dropdown - Claude Generated

    Live-detects models per provider via ProviderDetectionService (shared 300s TTL
    cache, same source as the Qt6 GUI), instead of reading the runtime-only
    ``provider.available_models`` field that is never populated on the webapp
    backend. Falls back to the persisted list / preferred model when detection
    yields nothing (e.g. provider unreachable) so the dropdown is never empty.
    """
    try:
        app_context = AppContext()
        services = app_context.get_services()
        config_manager = services['config_manager']
        detection = config_manager.get_provider_detection_service()
        unified_config = config_manager.get_unified_config()
        enabled_providers = unified_config.get_enabled_providers()

        models = []
        for provider in enabled_providers:
            provider_name = provider.name
            try:
                available = detection.get_available_models(provider_name) or []
            except Exception as e:
                logger.warning(f"Model detection failed for {provider_name}: {e}")
                available = []
            if not available:
                available = list(getattr(provider, 'available_models', []) or [])
            if not available and getattr(provider, 'preferred_model', None):
                available = [provider.preferred_model]
            for model in available:
                models.append({
                    "provider": provider_name,
                    "model": model,
                    "value": f"{provider_name}|{model}"
                })
        return models
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []


@app.post("/api/models/refresh")
async def refresh_models() -> list:
    """Re-read config.json + re-detect provider models, then return the fresh list - Claude Generated

    Mirrors the Qt6 GUI's "Refresh Models" / config-change path
    (``MainWindow._refresh_components``): force-reload config from disk, rebuild
    the generation provider clients, and rebuild the detection service while
    clearing its model cache. Lets the webapp pick up providers/models added
    externally (Qt6 GUI or a config.json edit) without a server restart.
    """
    try:
        app_context = AppContext()
        services = app_context.get_services()
        config_manager = services['config_manager']
        config_manager.load_config(force_reload=True)
        try:
            services['llm_service'].reload_providers()
        except Exception as e:
            logger.warning(f"llm_service.reload_providers failed: {e}")
        config_manager.get_provider_detection_service().reload()
    except Exception as e:
        logger.error(f"Error refreshing models: {e}")
    return await get_available_models()


def _parse_think_override(value: Optional[str]) -> Optional[bool]:
    """Map a 'default'|'on'|'off' thinking override string to None/True/False - Claude Generated."""
    if not value:
        return None
    v = value.strip().lower()
    if v in ("on", "true", "1", "yes", "an"):
        return True
    if v in ("off", "false", "0", "no", "aus"):
        return False
    return None  # "default" / unknown → leave per-step/task value


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


@app.post("/api/analyze/{session_id}")
async def start_analysis(
    session_id: str,
    input_type: str = Form(...),  # "text", "doi", "pdf", "img"
    content: Optional[str] = Form(None),  # For text/doi
    file: Optional[UploadFile] = File(None),  # For pdf/img
    global_override: Optional[str] = Form(None),  # "provider|model" override - Claude Generated
    think_override: Optional[str] = Form(None),  # "default"|"on"|"off" thinking override - Claude Generated
    source_type: Optional[str] = Form(None),   # Original source type for filename metadata - Claude Generated
    source_value: Optional[str] = Form(None),  # DOI/URL/filename for working title - Claude Generated
    workflow: Optional[str] = Form(None),  # Workflow stem or __classic__ - Claude Generated
) -> dict:
    """Start pipeline analysis - Direct execution with LLM queueing - Claude Generated (2026-01-13)"""
    # Auto-create session if not exists (for /webapp route with injected sessionId) - Claude Generated (2026-01-13)
    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
        logger.info(f"Auto-created session {session_id} for /api/analyze")

    session = sessions[session_id]

    if session.status == "running":
        raise HTTPException(status_code=400, detail="Analysis already running")

    session.input_data = {"type": input_type, "content": content}

    # READ FILE CONTENTS IMMEDIATELY before creating background task - Claude Generated (Defensive)
    # This prevents "read of closed file" error that occurs when UploadFile is passed to background task
    file_contents = None
    filename = None
    if file:
        try:
            file_contents = await file.read()
            filename = file.filename
            if not file_contents:
                raise HTTPException(status_code=400, detail="File is empty")
            logger.info(f"File read successfully: {len(file_contents)} bytes")
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    session.status = "running"
    if workflow:
        session.workflow_name = workflow
        logger.info(f"Session {session_id} workflow set to: {workflow}")

    # Start analysis in background with file contents, not the UploadFile object
    asyncio.create_task(run_analysis(session_id, input_type, content, file_contents, filename, global_override, source_type, source_value, workflow, think_override))

    return {"session_id": session_id, "status": "started"}


@app.post("/api/input/{session_id}")
async def process_input_only(
    session_id: str,
    input_type: str = Form(...),  # "text", "doi", "pdf", "img"
    content: Optional[str] = Form(None),  # For text/doi
    file: Optional[UploadFile] = File(None),  # For pdf/img
) -> dict:
    """Process only the input step (text extraction/OCR) - Claude Generated"""

    # Auto-create session if not exists (for /webapp route with injected sessionId) - Claude Generated (2026-01-13)
    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
        logger.info(f"Auto-created session {session_id} for /api/input")

    session = sessions[session_id]

    if session.status == "running":
        raise HTTPException(status_code=400, detail="Analysis already running")

    session.input_data = {"type": input_type, "content": content}

    # READ FILE CONTENTS IMMEDIATELY before creating background task - Claude Generated (Defensive)
    file_contents = None
    if file:
        try:
            file_contents = await file.read()
            if not file_contents:
                raise HTTPException(status_code=400, detail="File is empty")
            logger.info(f"File read successfully: {len(file_contents)} bytes")
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    session.status = "running"

    # Start input-only processing in background - Claude Generated
    asyncio.create_task(run_input_extraction(session_id, input_type, content, file_contents, file.filename if file else None))

    return {"session_id": session_id, "status": "started", "mode": "input_extraction"}


def make_json_serializable(obj):
    """Convert sets and other non-JSON types to JSON-serializable equivalents - Claude Generated"""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    return obj


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize filename for HTTP headers and cross-platform safety - Claude Generated

    Args:
        filename: Original filename (may contain unicode, special chars)
        max_length: Maximum filename length (default: 100)

    Returns:
        ASCII-safe filename suitable for Content-Disposition header
    """
    if not filename:
        return "alima_analysis"

    # Normalize unicode (e.g., ü → u)
    normalized = unicodedata.normalize('NFKD', filename)
    # Remove non-ASCII characters
    ascii_safe = normalized.encode('ASCII', 'ignore').decode('ASCII')
    # Replace invalid filename characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', ascii_safe)
    # Collapse multiple underscores/spaces into single underscore
    sanitized = re.sub(r'[_\s]+', '_', sanitized).strip('_ ')
    # Truncate and ensure we have a valid result
    result = sanitized[:max_length].rstrip('_')
    return result if result else "alima_analysis"


def _autosave_session_state(session: Session):
    """Auto-save session state to JSON after each pipeline step - Claude Generated"""

    if not session.autosave_enabled or not session.current_analysis_state:
        return

    try:
        # Save analysis state using existing PipelineJsonManager
        PipelineJsonManager.save_analysis_state(
            session.current_analysis_state,
            str(session.autosave_path)
        )

        # Update timestamp for status indicator - Claude Generated
        session.autosave_timestamp = datetime.now().isoformat()

        # Save metadata for recovery UI
        metadata = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_step": session.current_step,
            "status": session.status,
            "autosave_timestamp": session.autosave_timestamp,
        }

        metadata_path = session.autosave_path.with_suffix('.meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Auto-saved session {session.session_id} after step '{session.current_step}'")

    except Exception as e:
        logger.error(f"Auto-save failed for session {session.session_id}: {e}")
        session.autosave_failed = True
        # Don't raise - auto-save is best-effort, shouldn't block pipeline


def cleanup_old_autosaves(max_age_hours: int = None):
    """Remove auto-save files older than max_age_hours - Claude Generated"""

    if max_age_hours is None:
        max_age_hours = AUTOSAVE_MAX_AGE_HOURS  # Use global config

    try:
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0

        for file_path in AUTOSAVE_DIR.glob("session_*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                # Remove JSON file
                file_path.unlink()

                # Remove metadata file
                meta_path = file_path.with_suffix('.meta.json')
                if meta_path.exists():
                    meta_path.unlink()

                cleaned_count += 1
                logger.debug(f"Cleaned up old autosave: {file_path.name}")

        if cleaned_count > 0:
            logger.info(f"✓ Cleaned up {cleaned_count} old auto-save files (>{max_age_hours}h)")

    except Exception as e:
        logger.error(f"Cleanup error: {e}")



@app.get("/api/queue/status")
async def get_queue_status() -> dict:
    """Get combined LLM + Pipeline queue status - Claude Generated (2026-01-13)"""
    # Get LLM stats from AppContext.pipeline_manager
    app_context = AppContext()
    try:
        llm_stats = app_context.pipeline_manager.get_llm_queue_status()
    except Exception as e:
        logger.warning(f"Could not get LLM queue status: {e}")
        llm_stats = {
            "active_llm_requests": 0,
            "pending_llm_requests": 0,
            "max_concurrent": 3,
            "total_completed": 0,
            "avg_duration_seconds": 0.0
        }

    # Pipeline stats (simplified)
    pipeline_stats = {
        "active_pipelines": len([s for s in sessions.values() if s.status == "running"]),
        "queued_sessions": 0  # No longer queueing at pipeline level
    }

    # Determine overall status
    status = "healthy"
    if llm_stats["pending_llm_requests"] > 10:
        status = "busy"

    return {
        "llm": llm_stats,
        "pipeline": pipeline_stats,
        "status": status
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket for live progress updates - Claude Generated"""

    if session_id not in sessions:
        await websocket.close(code=1008, reason="Session not found")
        return

    await websocket.accept()
    session = sessions[session_id]
    logger.info(f"WebSocket connected for session {session_id}")

    try:
        last_step = None
        idle_count = 0
        render_sent = 0  # WP12: per-connection cursor → full replay on (re)connect
        # Use configurable timeout (count in 0.5s intervals)
        max_idle = WEBSOCKET_TIMEOUT_SECONDS * 2  # Claude Generated (config-based)

        # Heartbeat mechanism for long-running pipelines - Claude Generated
        # Use configurable heartbeat interval (count in 0.5s intervals)
        heartbeat_interval = WEBSOCKET_HEARTBEAT_INTERVAL * 2  # Claude Generated (config-based)
        heartbeat_counter = 0

        while True:
            # Check if analysis is complete
            if session.status not in ["running", "idle"]:
                logger.info(f"Session {session_id} status changed to {session.status}")
                # Flush any remaining render events (e.g. the final DK card). - WP12
                final_render, render_sent = session.get_render_events_since(render_sent)
                # Send final update with JSON-serializable results
                await websocket.send_json({
                    "type": "complete",
                    "status": session.status,
                    "results": make_json_serializable(
                        _prepare_results_for_export(session.results, validate_rvk=False)
                    ),
                    "error": session.error_message,
                    "current_step": session.current_step,
                    "render_events": final_render,
                })
                break

            # Increment and send heartbeat periodically - Claude Generated
            heartbeat_counter += 1
            if heartbeat_counter >= heartbeat_interval:
                heartbeat_counter = 0
                await websocket.send_json({
                    "type": "heartbeat",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "current_step": session.current_step
                })

            # Always send status update (every 500ms) - Claude Generated
            # Include streaming tokens buffered since last update
            # Use get_new_streaming_tokens to avoid losing tokens during long runs - Claude Generated
            if session.status == "running":
                streaming_tokens = session.get_new_streaming_tokens()
            else:
                streaming_tokens = session.get_and_clear_streaming_buffer()

            # WP12: new render events since this connection last saw them.
            new_render, render_sent = session.get_render_events_since(render_sent)

            await websocket.send_json({
                "type": "status",
                "status": session.status,
                "current_step": session.current_step,
                "current_step_status": session.current_step_status,  # 'running' or 'completed' - Claude Generated
                "results": make_json_serializable(
                    _prepare_results_for_export(session.results, validate_rvk=False)
                ),
                "streaming_tokens": make_json_serializable(streaming_tokens),  # Dict[step_id -> List[tokens]]
                "render_events": new_render,  # WP12: shared chrome events
                "autosave_timestamp": session.autosave_timestamp,  # For status indicator - Claude Generated
                "dk_search_progress": session.dk_search_progress,  # DK search progress info - Claude Generated
            })

            # Track idle time (no step change)
            if session.current_step == last_step:
                idle_count += 1
            else:
                idle_count = 0
                last_step = session.current_step
                logger.info(f"Step changed: {session.current_step}")

            # Timeout if idle too long
            if idle_count > max_idle:
                logger.warning(f"Session {session_id} idle timeout")
                await websocket.send_json({
                    "type": "error",
                    "error": "Analysis timeout",
                })
                break

            # Wait before next update
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}", exc_info=True)


@app.get("/api/export/{session_id}")
async def export_results(session_id: str, format: str = "json") -> FileResponse:
    """Export analysis results - supports partial and complete exports - Claude Generated"""

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Allow export even if results are empty (partial state) - Claude Generated (2026-01-06)
    # User can download current progress at any time

    if format == "json":
        status_suffix = "complete" if session.status == "completed" else "partial"

        # Create temporary JSON file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            dir=tempfile.gettempdir()
        )

        export_data = _build_export_payload(
            session_id=session.session_id,
            created_at=session.created_at,
            status=session.status,
            current_step=session.current_step,
            input_data=session.input_data,
            results=session.results,
            autosave_timestamp=session.autosave_timestamp,
            validate_rvk=True,
        )

        json.dump(export_data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()

        session.add_temp_file(temp_file.name)

        # Filename includes working title if available - Claude Generated
        if session.working_title:
            safe_title = sanitize_filename(session.working_title)
            filename = f"{safe_title}.json"
            logger.info(f"📥 Export filename from working_title: '{session.working_title}' → '{filename}'")
        else:
            # Fallback: use session ID and status indicator - Claude Generated (2026-01-06)
            filename = f"alima_analysis_{session.session_id}_{status_suffix}.json"
            logger.warning(f"⚠️ No working_title, using fallback filename: {filename} (session.working_title={session.working_title})")

        return FileResponse(
            temp_file.name,
            filename=filename,
            media_type="application/json"
        )

    raise HTTPException(status_code=400, detail=f"Format not supported: {format}")


# Workflow list order mirrors the Qt6 pipeline tab picker.
_WORKFLOW_ORDER = [
    "alima_v51",
    "__classic__",
    "alima",
    "alima_classic_v51",
    "alima_classic",
    "title_list_search",
    "catalog_search",
    "synonym_expansion",
    "batch_metadata",
]


def _extract_workflow_steps(data: dict) -> list:
    """Reduce a workflow YAML's ``steps:`` block to [{id, label}, …] for the
    frontend pipeline-stepper. Skips non-dict / id-less entries. - Claude Generated
    """
    steps = []
    for entry in data.get("steps", []) or []:
        if not isinstance(entry, dict):
            continue
        step_id = entry.get("id")
        if not step_id:
            continue
        steps.append({"id": str(step_id), "label": str(entry.get("name") or step_id)})
    return steps


# Canonical classic-pipeline steps — authoritative source is
# PipelineManager.step_definitions (src/core/pipeline_manager.py); order matches
# _create_pipeline_steps. German labels for webapp consistency. - Claude Generated
_CLASSIC_STEPS = [
    {"id": "input", "label": "Eingabe"},
    {"id": "initialisation", "label": "Schlagwörter"},
    {"id": "search", "label": "GND-Suche"},
    {"id": "keywords", "label": "Prüfung"},
    {"id": "dk_search", "label": "DK-Suche"},
    {"id": "dk_classification", "label": "Klassifikation"},
]


def _discover_workflows() -> tuple[dict, dict, dict]:
    """Scan DEFAULT_SEARCH_PATHS for v4 YAML workflows.

    Returns (root_stem → version, legacy_stem → version, stem → steps[]). - Claude Generated
    """
    root: dict = {}
    legacy: dict = {}
    steps_by_stem: dict = {}
    seen: set = set()

    for base in DEFAULT_SEARCH_PATHS:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.glob("*.yaml"), key=lambda p: str(p.name)):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                version = str(data.get("version", "?"))
                root[path.stem] = version
                steps_by_stem[path.stem] = _extract_workflow_steps(data)
            except Exception as e:
                logger.warning(f"Could not read workflow {path}: {e}")
        legacy_dir = base / "legacy"
        if legacy_dir.is_dir():
            for path in sorted(legacy_dir.glob("*.yaml"), key=lambda p: str(p.name)):
                key = path.resolve()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    with open(path, encoding="utf-8") as fh:
                        data = yaml.safe_load(fh) or {}
                    version = str(data.get("version", "?"))
                    legacy[path.stem] = version
                    steps_by_stem[path.stem] = _extract_workflow_steps(data)
                except Exception as e:
                    logger.warning(f"Could not read legacy workflow {path}: {e}")

    return root, legacy, steps_by_stem


@app.get("/api/workflows")
async def get_available_workflows() -> list:
    """Get available pipeline/agentic workflows for the workflow dropdown. - Claude Generated"""
    try:
        root, legacy, steps_by_stem = _discover_workflows()

        def _label(stem: str) -> str:
            ver = root.get(stem)
            return f"{stem} (v{ver})" if ver else stem

        items = []
        added: set = set()

        if "alima_v51" in root:
            items.append({
                "label": f"⭐ ALIMA v5.1 — agentisch (v{root['alima_v51']})",
                "value": "alima_v51",
                "agentic": True,
                "steps": steps_by_stem.get("alima_v51", []),
            })
            added.add("alima_v51")

        items.append({
            "label": "Klassische Pipeline (nicht agentisch)",
            "value": "__classic__",
            "agentic": False,
            "steps": _CLASSIC_STEPS,
        })
        added.add("__classic__")

        for stem in _WORKFLOW_ORDER:
            if stem in added or stem not in root:
                continue
            items.append({"label": _label(stem), "value": stem, "agentic": True,
                          "steps": steps_by_stem.get(stem, [])})
            added.add(stem)

        for stem in sorted(root):
            if stem in added:
                continue
            items.append({"label": _label(stem), "value": stem, "agentic": True,
                          "steps": steps_by_stem.get(stem, [])})
            added.add(stem)

        if legacy:
            items.append({"label": "───────────────", "value": "__separator__",
                          "agentic": False, "steps": []})
            for stem in sorted(legacy):
                items.append({
                    "label": f"{stem} (legacy v{legacy[stem]})",
                    "value": stem,
                    "agentic": True,
                    "steps": steps_by_stem.get(stem, []),
                })

        return items
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        return []


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


@app.get("/api/session/{session_id}/recover")
async def recover_session(session_id: str) -> dict:
    """Recover results from auto-saved state after timeout - Claude Generated"""

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Check if auto-save exists
    if not session.autosave_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No auto-saved state available for this session"
        )

    try:
        # Load from auto-saved JSON using existing PipelineJsonManager
        analysis_state = PipelineJsonManager.load_analysis_state(str(session.autosave_path))

        # Reconstruct results using shared helper
        session.results = _extract_results_from_analysis_state(analysis_state)
        session.status = "recovered"
        session.current_analysis_state = analysis_state

        # Read metadata
        metadata = {}
        metadata_path = session.autosave_path.with_suffix('.meta.json')
        if metadata_path.exists():
            with open(metadata_path, encoding='utf-8') as f:
                metadata = json.load(f)

        logger.info(f"✓ Successfully recovered session {session_id} from auto-save")

        return {
            "session_id": session_id,
            "status": "recovered",
            "results": make_json_serializable(
                _prepare_results_for_export(session.results, validate_rvk=False)
            ),
            "metadata": metadata,
            "message": "Results recovered successfully"
        }

    except json.JSONDecodeError as e:
        logger.error(f"Corrupted auto-save file for session {session_id}: {e}")
        raise HTTPException(
            status_code=422,
            detail="Auto-save file is corrupted and cannot be recovered"
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Auto-save file not found"
        )
    except Exception as e:
        logger.error(f"Recovery failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Recovery failed: {str(e)}"
        )


async def run_analysis(
    session_id: str,
    input_type: str,
    content: Optional[str],
    file_contents: Optional[bytes],
    filename: Optional[str],
    global_override: Optional[str] = None,
    source_type: Optional[str] = None,   # Original source type for working title - Claude Generated
    source_value: Optional[str] = None,  # DOI/URL/filename for working title - Claude Generated
    workflow: Optional[str] = None,  # Workflow stem or __classic__ - Claude Generated
    think_override: Optional[str] = None,  # "default"|"on"|"off" thinking override - Claude Generated
):
    """Execute pipeline analysis with direct PipelineManager - Claude Generated"""

    session = sessions[session_id]
    session.status = "running"

    # WP12: single shared producer for the DK/GND chrome (same renderer the GUI
    # uses); events are buffered on the session and broadcast over the WebSocket.
    session_renderer = _build_session_renderer(session)
    # Phase 4: bridge AlimaStateBus events (tool calls, pipeline steps/prompts)
    # into the same render buffer. Subscriber is local to this run.
    bus_subscriber = _SessionBusSubscriber(session_renderer)

    try:
        bus_subscriber.subscribe()
        # Resolve input to text - Claude Generated
        input_text = None

        if input_type == "text" and content:
            input_text = content
        elif input_type == "doi" and content:
            logger.info(f"Resolving DOI: {content}")
            def _resolve_doi():
                cfg = _get_doi_config()
                resolver = UnifiedResolver(logger,
                    contact_email=cfg['contact_email'],
                    use_crossref=cfg['use_crossref'],
                    use_openalex=cfg['use_openalex'],
                    use_datacite=cfg['use_datacite'],
                )
                success, metadata, text_result = resolver.resolve(content)
                return format_doi_metadata(metadata, text_result or "") if success else None
            input_text = await asyncio.to_thread(_resolve_doi)
        elif input_type == "pdf" and file_contents:
            # Save and extract from PDF - Claude Generated (File contents already read)
            try:
                suffix = ".pdf" if filename and filename.endswith(".pdf") else ".pdf"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                logger.info(f"Extracting text from PDF: {temp_file.name} ({len(file_contents)} bytes)")
                input_text = await asyncio.to_thread(resolve_input_to_text, temp_file.name)
            except Exception as e:
                logger.error(f"PDF processing error: {e}")
                raise
        elif input_type == "img" and file_contents:
            # Save and extract from image - Claude Generated (File contents already read)
            try:
                # Determine extension from filename or default to jpg
                suffix = ""
                if filename:
                    if filename.lower().endswith(".png"):
                        suffix = ".png"
                    elif filename.lower().endswith(".jpeg"):
                        suffix = ".jpeg"
                    else:
                        suffix = ".jpg"
                else:
                    suffix = ".jpg"

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                logger.info(f"Analyzing image: {temp_file.name} ({len(file_contents)} bytes)")
                input_text = await asyncio.to_thread(resolve_input_to_text, temp_file.name)
            except Exception as e:
                logger.error(f"Image processing error: {e}")
                raise
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if not input_text:
            raise ValueError("Could not extract text from input")

        logger.info(f"Input text extracted ({len(input_text)} chars)")
        session.input_data = {"type": input_type, "text_preview": input_text[:100]}

        # Get or initialize services (singleton pattern - Claude Generated)
        app_context = AppContext()
        services = app_context.get_services()

        config_manager = services['config_manager']

        # Create a NEW PipelineManager for this session to prevent cross-session contamination - Claude Generated (2026-01-13)
        # This ensures each concurrent analysis has its own isolated pipeline state
        pipeline_manager = PipelineManager(
            alima_manager=services['alima_manager'],
            cache_manager=services['cache_manager'],
            config_manager=config_manager
        )
        logger.info(f"Created new PipelineManager for session {session_id}")

        # Create pipeline config from preferences
        pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)

        # Apply global override if provided - Claude Generated
        think_val = _parse_think_override(think_override)
        if global_override or think_val is not None:
            if global_override:
                provider, model = PipelineConfig.parse_override_string(global_override)
                pipeline_config.global_provider_override = provider
                pipeline_config.global_model_override = model
            pipeline_config.global_think_override = think_val
            pipeline_config.apply_global_override()
            logger.info(
                f"🔬 Webapp global override applied: "
                f"{pipeline_config.global_provider_override}/{pipeline_config.global_model_override} "
                f"think={think_val}"
            )

        # Set config on pipeline manager (was missing - config was built but never applied)
        pipeline_manager.set_config(pipeline_config)

        # Remember the effective provider/model so the session chat agent can fall
        # back to the same credentials after the pipeline manager is discarded.
        eff_provider = pipeline_config.global_provider_override
        eff_model = pipeline_config.global_model_override
        if not eff_provider:
            init_cfg = pipeline_config.step_configs.get("initialisation")
            if init_cfg is not None:
                eff_provider = getattr(init_cfg, "provider", None) or ""
                eff_model = getattr(init_cfg, "model", None) or ""
        session.last_provider = eff_provider or None
        session.last_model = eff_model or None

        # Configure agentic mode when a non-classic workflow is requested - Claude Generated
        if workflow and workflow != "__classic__":
            pipeline_config.enable_agentic_mode = True
            pipeline_config.workflow_name = workflow
            logger.info(f"🧬 Agentic mode enabled for workflow: {workflow}")
        else:
            pipeline_config.enable_agentic_mode = False
            pipeline_config.workflow_name = None
            logger.info("🔒 Classic (non-agentic) pipeline mode selected")

        # Define callbacks for live updates - Claude Generated
        def on_step_started(step):
            session.current_step = step.step_id
            session.current_step_status = 'running'  # Claude Generated
            logger.info(f"Step started: {step.step_id}")

        def on_step_completed(step):
            session.current_step = step.step_id
            session.current_step_status = 'completed'  # Claude Generated
            logger.info(f"Step completed: {step.step_id}")

            # WP12: emit the DK/RVK catalog-research card as a render event so
            # the webapp shows the identical chrome the GUI does.
            if step.step_id == "dk_search" and step.output_data:
                try:
                    html, plain = PipelineResultFormatter.format_dk_search_card_html(
                        step.output_data
                    )
                    if html:
                        session_renderer.render_html_block(
                            html, kind="dk_search", plain_text=plain
                        )
                except Exception:
                    logger.exception("WP12: dk_search card emission failed")

            # Sync analysis state reference so autosave has access - Claude Generated
            # Must be set here because start_pipeline() hasn't returned yet when callbacks fire
            session.current_analysis_state = pipeline_manager.current_analysis_state

            # Update working title after initialisation - Claude Generated
            if step.step_id == "initialisation":
                if pipeline_manager.current_analysis_state and hasattr(pipeline_manager.current_analysis_state, 'working_title'):
                    wt = pipeline_manager.current_analysis_state.working_title
                    logger.debug(f"Working title from analysis state: '{wt}'")
                    session.working_title = wt
                    if not session.results:
                        session.results = {}
                    session.results['working_title'] = wt
                    logger.info(f"Session working title set: {wt}")
                else:
                    logger.warning("No working_title available after initialisation")

            # Add delay after LLM steps to allow WebSocket to fetch buffered tokens - Claude Generated
            llm_steps = ["initialisation", "keywords", "dk_classification"]
            if step.step_id in llm_steps:
                import time
                time.sleep(0.7)  # 700ms = 500ms poll + 200ms margin
                logger.debug(f"Waited 700ms for streaming token transmission after {step.step_id}")

            # Auto-save after each step completion - Claude Generated
            if session.autosave_enabled:
                try:
                    _autosave_session_state(session)
                except Exception as e:
                    logger.error(f"Auto-save error (continuing): {e}")

        def on_step_error(step, error_msg):
            session.current_step = step.step_id
            session.error_message = error_msg
            logger.error(f"Step error: {step.step_id}: {error_msg}")

        def on_agentic_context(step_id, snapshot):
            """Mirror agentic step progress into the session for the frontend
            pipeline-stepper. Agentic workflows don't use step_started_callback;
            they report per-step completion via context snapshots (running
            snapshots are skipped upstream). - Claude Generated"""
            try:
                sid = step_id or (snapshot or {}).get("_step_id")
                if not sid:
                    return
                session.current_step = sid
                snap_status = (snapshot or {}).get("_step_status") or "completed"
                session.current_step_status = (
                    "error" if snap_status == "error" else "completed"
                )
            except Exception:
                logger.debug("agentic context step update failed", exc_info=True)

        def on_pipeline_completed(analysis_state):
            logger.info(f"Pipeline completed, storing results")

            # Sync analysis state reference so autosave has access - Claude Generated
            session.current_analysis_state = analysis_state

            # WP12: emit the final DK-classifications card as a render event
            # (buffered before status flips to "completed", so the WS picks it
            # up in the final message).
            try:
                html, plain = PipelineResultFormatter.format_dk_classifications_card_html(
                    analysis_state
                )
                if html:
                    session_renderer.render_html_block(
                        html, kind="dk_classifications", plain_text=plain
                    )
            except Exception:
                logger.exception("WP12: dk_classifications card emission failed")

            # Reintroduced RVK-Analytik: frequency Auswertung + RVK provenance
            # tables as a shared render card (classic pipeline only). - Claude Generated
            try:
                html, plain = PipelineResultFormatter.format_dk_auswertung_card_html(
                    analysis_state
                )
                if html:
                    session_renderer.render_html_block(
                        html, kind="dk_statistics", plain_text=plain
                    )
            except Exception:
                logger.exception("WP12: dk_auswertung card emission failed")

            # Use shared extraction helper (DRY principle) - Claude Generated
            session.results = _prepare_results_for_export(
                _extract_results_from_analysis_state(analysis_state),
                validate_rvk=True,
            )

            # Synchronize session.working_title with session.results['working_title'] - Claude Generated
            if session.results.get('working_title'):
                session.working_title = session.results['working_title']
                logger.info(f"✅ Synchronized session.working_title from results: {session.working_title}")
            else:
                logger.warning(f"⚠️ No working_title in results, session.working_title remains: {session.working_title}")

            # Log summary
            final_keywords = session.results.get("final_keywords", [])
            dk_classifications = session.results.get("dk_classifications", [])
            initial_keywords = session.results.get("initial_keywords", [])
            logger.info(f"Extracted results - keywords: {len(final_keywords)}, classifications: {len(dk_classifications)}, initial: {len(initial_keywords)}")

            # Wait for WebSocket to send ALL remaining streaming tokens - Claude Generated
            # WebSocket sends updates every 500ms, so wait at least 600ms to ensure final tokens are sent
            import time
            time.sleep(0.6)

            total_tokens = sum(len(t) for t in session.streaming_buffer.values()) if session.streaming_buffer else 0
            logger.info(f"Waited 600ms for final streaming tokens to be sent (buffer has {total_tokens} total tokens)")

            session.status = "completed"
            session.current_step = "classification"

            # Final auto-save - Claude Generated
            if session.autosave_enabled:
                try:
                    _autosave_session_state(session)
                except Exception as e:
                    logger.error(f"Final auto-save error: {e}")

        def on_stream_token(token: str, step_id: str = ""):
            """Handle token streaming - buffer tokens for WebSocket - Claude Generated"""
            # Check for abort request - Claude Generated
            if session.abort_requested:
                raise Exception("Pipeline execution cancelled by user")

            # Extract DK search progress if present - Claude Generated
            # Pattern: [N/M] (P%) Suche 'keyword'...
            if step_id == "dk_search":
                progress_match = re.match(r'\[(\d+)/(\d+)\]\s*\((\d+)%\)', token)
                if progress_match:
                    current = int(progress_match.group(1))
                    total = int(progress_match.group(2))
                    percent = int(progress_match.group(3))
                    session.dk_search_progress = {
                        "current": current,
                        "total": total,
                        "percent": percent
                    }

            # Buffer tokens by step for periodic transmission via WebSocket
            if step_id:
                session.add_streaming_token(token, step_id)
            logger.debug(f"Token [{step_id}]: {token[:30] if len(token) > 30 else token}...")

        # Run pipeline in background thread - Claude Generated
        def execute_pipeline():
            try:
                # Check for abort before starting - Claude Generated
                if session.abort_requested:
                    raise Exception("Pipeline execution cancelled by user")

                # Set up callbacks
                pipeline_manager.set_callbacks(
                    step_started=on_step_started,
                    step_completed=on_step_completed,
                    step_error=on_step_error,
                    pipeline_completed=on_pipeline_completed,
                    stream_callback=on_stream_token,
                    agentic_context=on_agentic_context,
                )

                # Store reference and wire interrupt callback for step-abort - Claude Generated
                session.pipeline_manager_ref = pipeline_manager
                if hasattr(pipeline_manager, 'set_interrupt_flag'):
                    import threading
                    pipeline_manager.set_interrupt_flag(
                        threading.Lock(),
                        lambda: session.abort_requested
                    )

                # Determine effective source type/value for working title BEFORE start_pipeline runs.
                # start_pipeline executes the pipeline synchronously, so overriding state afterwards is too late.
                # source_type/source_value come from JS when the text was pre-extracted (DOI resolved in browser). - Claude Generated
                if source_type and source_type != 'text' and source_value:
                    effective_input_type = source_type      # e.g. 'doi'
                    effective_input_source = source_value   # e.g. '10.1007/...'
                elif input_type in ("doi", "url"):
                    effective_input_type = input_type
                    effective_input_source = content
                else:
                    effective_input_type = input_type
                    effective_input_source = filename or None

                logger.info(f"Starting pipeline: input_type={input_type}, effective_type={effective_input_type}, source={effective_input_source}")
                pipeline_id = pipeline_manager.start_pipeline(
                    input_text,
                    input_type=effective_input_type,
                    input_source=effective_input_source,
                )

                # Store analysis state reference for auto-save - Claude Generated
                session.current_analysis_state = pipeline_manager.current_analysis_state

                logger.info(f"Pipeline {pipeline_id} started with input_type={input_type}")

            except Exception as e:
                logger.error(f"Pipeline execution error: {str(e)}", exc_info=True)
                session.status = "error"
                session.error_message = str(e)
            finally:
                session.pipeline_manager_ref = None  # Clear reference after pipeline ends - Claude Generated

        # Run in executor to avoid blocking
        await asyncio.to_thread(execute_pipeline)

    except Exception as e:
        logger.error(f"Analysis setup error: {str(e)}", exc_info=True)
        session.status = "error"
        session.error_message = str(e)
    finally:
        # Phase 4: remove session-local bus handlers before cleanup.
        try:
            bus_subscriber.unsubscribe()
        except Exception:
            logger.exception("Failed to unsubscribe session bus subscriber")
        # Cleanup
        if session_id in sessions:
            session.cleanup()


async def run_input_extraction(
    session_id: str,
    input_type: str,
    content: Optional[str],
    file_contents: Optional[bytes],
    filename: Optional[str],
):
    """Execute only the input extraction step (text extraction/OCR) - Claude Generated"""

    session = sessions[session_id]
    session.status = "running"

    try:
        # Use execute_input_extraction from pipeline_utils (same as pipeline does) - Claude Generated
        from src.utils.pipeline_utils import execute_input_extraction

        def stream_callback_wrapper(message: str):
            """Wrap stream callback for live progress - Claude Generated"""
            # Check for abort before updating - Claude Generated
            if session.abort_requested:
                raise Exception("Pipeline execution cancelled by user")

            session.current_step = "input"
            # Use existing add_streaming_token method - correct parameter order: (token, step_id) - Claude Generated
            session.add_streaming_token(message, "input")
            logger.info(f"[Stream] {message}")

        def execute_extraction():
            # Check for abort before starting - Claude Generated
            if session.abort_requested:
                raise Exception("Pipeline execution cancelled by user")

            # Prepare input source and normalize input_type - Claude Generated
            input_source = None
            normalized_input_type = input_type  # Will change for doi->text after resolution

            if input_type == "text" and content:
                input_source = content
            elif input_type == "doi" and content:
                # Resolve DOI/URL to text first - Claude Generated
                logger.info(f"Resolving DOI/URL: {content}")
                cfg = _get_doi_config()
                resolver = UnifiedResolver(logger,
                    contact_email=cfg['contact_email'],
                    use_crossref=cfg['use_crossref'],
                    use_openalex=cfg['use_openalex'],
                    use_datacite=cfg['use_datacite'],
                )
                success, metadata, text_result = resolver.resolve(content)
                if not success:
                    raise ValueError(f"DOI resolution failed: {text_result}")
                text_content = format_doi_metadata(metadata, text_result or "")
                if not text_content:
                    raise ValueError("DOI resolution returned no content")
                input_source = text_content
                normalized_input_type = "text"  # Now treat as text
                logger.info(f"✅ DOI resolved to {len(text_content)} characters")
            elif input_type == "pdf" and file_contents:
                # Save PDF temporarily - Claude Generated
                suffix = ".pdf"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                input_source = temp_file.name
                logger.info(f"Saved PDF to {temp_file.name}")
            elif input_type == "img" and file_contents:
                # Save image temporarily - Claude Generated
                suffix = ""
                if filename:
                    if filename.lower().endswith(".png"):
                        suffix = ".png"
                    elif filename.lower().endswith(".jpeg"):
                        suffix = ".jpeg"
                    else:
                        suffix = ".jpg"
                else:
                    suffix = ".jpg"

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(file_contents)
                temp_file.close()
                session.add_temp_file(temp_file.name)
                input_source = temp_file.name
                logger.info(f"Saved image to {temp_file.name}")
            else:
                raise ValueError(f"Invalid input: type={input_type}")

            # Get LLM service via AppContext - Claude Generated
            app_context = AppContext()
            services = app_context.get_services()
            llm_service = services['llm_service']

            # Call execute_input_extraction with normalized input_type - Claude Generated
            logger.info(f"Executing input extraction with input_type={normalized_input_type} from {str(input_source)[:50]}...")
            extracted_text, source_info, extraction_method = execute_input_extraction(
                llm_service=llm_service,
                input_source=input_source,
                input_type=normalized_input_type if normalized_input_type != "img" else "image",  # pipeline uses "image" not "img"
                stream_callback=stream_callback_wrapper,
                logger=logger,
            )

            return extracted_text, source_info, extraction_method

        # Run extraction in executor to avoid blocking - Claude Generated
        extracted_text, source_info, extraction_method = await asyncio.to_thread(execute_extraction)

        # Store extracted text in results - Claude Generated
        session.results = {
            "original_abstract": extracted_text,
            "input_type": input_type,
            "input_mode": "extraction_only",
            "source_info": source_info,
            "extraction_method": extraction_method,
        }

        logger.info(f"✅ Input extraction completed: {extraction_method} - {len(extracted_text)} characters")
        session.current_step = "input"
        session.status = "completed"

    except Exception as e:
        logger.error(f"Input extraction error: {str(e)}", exc_info=True)
        session.status = "error"
        session.error_message = str(e)
    finally:
        # Cleanup
        if session_id in sessions:
            session.cleanup()


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session - Claude Generated"""
    if session_id in sessions:
        session = sessions[session_id]
        session.cleanup()
        del sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint - Claude Generated"""
    return {"status": "ok", "active_sessions": len(sessions)}


if __name__ == "__main__":
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
