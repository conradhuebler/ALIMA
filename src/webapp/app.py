"""
ALIMA Webapp - FastAPI Backend
Claude Generated - Pipeline widget as web interface
"""

import logging
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path BEFORE importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import ALIMA Pipeline components - Claude Generated
# PipelineConfig kept only for re-export (tests patch appmod.PipelineConfig.*). - Claude Generated
from src.core.pipeline_manager import PipelineConfig
from src.core.state_bus import set_direct_dispatch
from src.utils.config_manager import ConfigManager
from src.utils.qt_plugin_setup import setup_qt_plugin_paths, get_available_sql_drivers
# cleanup_old_autosaves (session_io.py, F-6 split) is used by the lifespan. - Claude Generated
from src.webapp.session_io import cleanup_old_autosaves
# APIRouter modules extracted from this file (F-6 split). _discover_workflows is
# re-exported for the unit test that imports it via src.webapp.app. - Claude Generated
from src.webapp.routers import workflows as workflows_router
from src.webapp.routers import models as models_router
from src.webapp.routers import sessions as sessions_router
from src.webapp.routers import export as export_router
from src.webapp.routers import websocket as websocket_router
from src.webapp.routers import analysis as analysis_router
from src.webapp.routers import agent as agent_router
from src.webapp.routers.workflows import _discover_workflows
# Re-exports for tests that import/call these via src.webapp.app: run_analysis
# (analysis); ChatMessageRequest + _build_session_agent_runner (agent). - Claude Generated
from src.webapp.routers.analysis import run_analysis
from src.webapp.routers.agent import ChatMessageRequest, _build_session_agent_runner

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

    # No Qt event loop runs here, but DatabaseManager creates a QCoreApplication
    # for QtSql — so AlimaStateBus would queue every worker-thread event for a
    # loop that never spins, and all bus-driven log chrome (pipeline steps,
    # agentic prompts, tool calls) would vanish. Deliver synchronously instead;
    # the webapp's subscribers only append to a lock-protected buffer. - Claude Generated
    set_direct_dispatch(True)

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
app.include_router(agent_router.router)

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

    # Render template with injected session ID + UI-chrome i18n catalog
    from src.utils.i18n import catalog_for_js

    return templates.TemplateResponse(
        "webapp.html",
        {"request": request, "session_id": session, "i18n_js": catalog_for_js()}
    )


# Session lifecycle endpoints (create/get/clear/cancel/abort_step) now live
# in routers/sessions.py (mounted via app.include_router above). - Claude Generated


# GET /api/models + POST /api/models/refresh now live in routers/models.py
# (mounted via app.include_router above). - Claude Generated


# _parse_think_override now lives in session_io.py (shared analysis+agent). - Claude Generated


# Agent endpoints (/agent/run, /api/session/{id}/chat) + their runner-builders
# and request models now live in routers/agent.py (mounted via
# app.include_router above). - Claude Generated


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint - Claude Generated"""
    return {"status": "ok", "active_sessions": len(sessions)}


if __name__ == "__main__":
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
