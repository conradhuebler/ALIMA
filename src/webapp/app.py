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
from typing import Optional
from datetime import datetime
import subprocess
import sys

# Add project root to sys.path BEFORE importing src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Import ALIMA Pipeline components - Claude Generated
from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.core.alima_manager import AlimaManager
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.utils.config_manager import ConfigManager
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.pipeline_utils import PipelineJsonManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Auto-Save Configuration - Claude Generated (2026-01-06)
# These settings control the auto-save and recovery system
AUTOSAVE_ENABLED = True  # Enable/disable auto-save system
AUTOSAVE_MAX_AGE_HOURS = 24  # Auto-cleanup files older than this (hours)
WEBSOCKET_TIMEOUT_SECONDS = 1800  # WebSocket idle timeout (30 minutes = 1800s)
WEBSOCKET_HEARTBEAT_INTERVAL = 5  # Heartbeat interval in seconds (5s)

app = FastAPI(title="ALIMA Webapp", description="Pipeline widget as web interface")

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
        self.input_data = None
        self.results = {}
        self.error_message = None
        self.process = None
        self.temp_files = []
        self.streaming_buffer = {}  # Buffer for streaming tokens by step_id - Claude Generated
        self.streaming_buffer_sent_count = {}  # Track how many tokens sent per step - Claude Generated
        self._streaming_lock = threading.Lock()  # Thread-safe access to streaming buffers - Claude Generated
        self.abort_requested = False  # Flag to signal pipeline abort - Claude Generated
        # Auto-save support - Claude Generated
        self.autosave_path = AUTOSAVE_DIR / f"session_{session_id}.json"
        self.autosave_enabled = AUTOSAVE_ENABLED  # Use global config
        self.autosave_failed = False
        self.autosave_timestamp = None  # Last auto-save timestamp for status indicator
        self.current_analysis_state = None  # Reference to PipelineManager state
        self.working_title = None  # Working title from initialisation step - Claude Generated
        self.dk_search_progress = None  # DK search progress info (current/total/percent) - Claude Generated

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

    def clear(self):
        """Complete session reset - clear all data - Claude Generated"""
        self.status = "idle"
        self.current_step = None
        self.input_data = None
        self.results = {}
        self.error_message = None
        with self._streaming_lock:  # Thread-safe buffer clearing - Claude Generated
            self.streaming_buffer.clear()
            self.streaming_buffer_sent_count.clear()  # Reset token tracking - Claude Generated
        self.abort_requested = False
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
        "results": session.results,
        "error_message": session.error_message,
        "streaming_tokens": streaming_tokens,  # Include for polling clients
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


@app.post("/api/analyze/{session_id}")
async def start_analysis(
    session_id: str,
    input_type: str = Form(...),  # "text", "doi", "pdf", "img"
    content: Optional[str] = Form(None),  # For text/doi
    file: Optional[UploadFile] = File(None),  # For pdf/img
) -> dict:
    """Start pipeline analysis - Direct execution with LLM queueing - Claude Generated (2026-01-13)"""
    # Auto-create session if not exists (for /webapp route with injected sessionId) - Claude Generated (2026-01-13)
    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
        logger.info(f"Auto-created session {session_id} for /api/analyze")

    session = sessions[session_id]

    if session.status == "running":
        raise HTTPException(status_code=400, detail="Analysis already running")

    session.status = "running"
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

    # Start analysis in background with file contents, not the UploadFile object
    asyncio.create_task(run_analysis(session_id, input_type, content, file_contents, filename))

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

    session.status = "running"
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


def _extract_results_from_analysis_state(analysis_state) -> dict:
    """Extract results dict from KeywordAnalysisState - shared logic for callback and recovery - Claude Generated"""

    # Extract final GND keywords from LLM analysis
    final_keywords = []
    if hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
        final_keywords = getattr(analysis_state.final_llm_analysis, 'extracted_gnd_keywords', [])

    # Extract DK classifications
    dk_classifications = getattr(analysis_state, 'dk_classifications', [])

    # Extract initial keywords
    initial_keywords = getattr(analysis_state, 'initial_keywords', [])

    # Extract original abstract
    original_abstract = getattr(analysis_state, 'original_abstract', '')

    # Extract working title - Claude Generated
    working_title = getattr(analysis_state, 'working_title', '')

    # Extract search results
    search_results = getattr(analysis_state, 'search_results', [])

    # Extract DK search results
    dk_search_results = getattr(analysis_state, 'dk_search_results', [])

    # Serialize search_results properly
    serialized_search_results = []
    if search_results:
        for result in search_results:
            try:
                search_term = getattr(result, 'search_term', '') if hasattr(result, 'search_term') else ''
                result_data = getattr(result, 'results', {}) if hasattr(result, 'results') else {}

                serialized_search_results.append({
                    "search_term": search_term,
                    "results": result_data if isinstance(result_data, dict) else {}
                })
            except Exception as e:
                logger.warning(f"Error serializing search result: {e}")
                continue

    # Serialize initial_llm_call_details
    initial_llm_details = None
    try:
        if hasattr(analysis_state, 'initial_llm_call_details') and analysis_state.initial_llm_call_details:
            llm_call = analysis_state.initial_llm_call_details
            initial_llm_details = {
                "response_full_text": getattr(llm_call, 'response_full_text', '') if hasattr(llm_call, 'response_full_text') else '',
                "provider": getattr(llm_call, 'provider', '') if hasattr(llm_call, 'provider') else '',
                "model": getattr(llm_call, 'model', '') if hasattr(llm_call, 'model') else '',
                "extracted_keywords": getattr(llm_call, 'extracted_keywords', []) if hasattr(llm_call, 'extracted_keywords') else [],
                "extracted_gnd_keywords": getattr(llm_call, 'extracted_gnd_keywords', []) if hasattr(llm_call, 'extracted_gnd_keywords') else [],
                "token_count": getattr(llm_call, 'token_count', 0) if hasattr(llm_call, 'token_count') else 0,
            }
    except Exception as e:
        logger.warning(f"Error serializing initial_llm_call_details: {e}")
        initial_llm_details = None

    # Serialize final_llm_analysis
    final_llm_details = None
    try:
        if hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
            llm_call = analysis_state.final_llm_analysis
            final_llm_details = {
                "response_full_text": getattr(llm_call, 'response_full_text', '') if hasattr(llm_call, 'response_full_text') else '',
                "provider": getattr(llm_call, 'provider', '') if hasattr(llm_call, 'provider') else '',
                "model": getattr(llm_call, 'model', '') if hasattr(llm_call, 'model') else '',
                "extracted_keywords": getattr(llm_call, 'extracted_keywords', []) if hasattr(llm_call, 'extracted_keywords') else [],
                "extracted_gnd_keywords": getattr(llm_call, 'extracted_gnd_keywords', []) if hasattr(llm_call, 'extracted_gnd_keywords') else [],
                "token_count": getattr(llm_call, 'token_count', 0) if hasattr(llm_call, 'token_count') else 0,
            }
    except Exception as e:
        logger.warning(f"Error serializing final_llm_analysis: {e}")
        final_llm_details = None

    # Convert sets to lists for JSON serialization
    def ensure_json_serializable(value):
        """Recursively convert non-JSON-serializable types"""
        if isinstance(value, set):
            return list(value)
        elif isinstance(value, dict):
            return {k: ensure_json_serializable(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [ensure_json_serializable(v) for v in value]
        return value

    return {
        # Input & Keywords & Title
        "original_abstract": original_abstract,
        "working_title": working_title,  # Claude Generated
        "initial_keywords": ensure_json_serializable(initial_keywords),
        "final_keywords": ensure_json_serializable(final_keywords),

        # GND/SWB Search Results
        "search_results": ensure_json_serializable(serialized_search_results),

        # DK Classification Results
        "dk_classifications": ensure_json_serializable(dk_classifications),
        "dk_search_results": ensure_json_serializable(dk_search_results),

        # LLM Analysis Details
        "initial_llm_call_details": ensure_json_serializable(initial_llm_details),
        "final_llm_call_details": ensure_json_serializable(final_llm_details),

        # Pipeline Metadata
        "pipeline_metadata": {
            "search_suggesters_used": ensure_json_serializable(getattr(analysis_state, 'search_suggesters_used', [])),
            "initial_gnd_classes": ensure_json_serializable(getattr(analysis_state, 'initial_gnd_classes', [])),
            "has_final_llm_analysis": bool(final_llm_details),
            "has_initial_llm_analysis": bool(initial_llm_details),
        }
    }


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


@app.on_event("startup")
async def startup_event():
    """Initialize webapp on startup - Claude Generated"""
    logger.info("🚀 Starting ALIMA Webapp...")
    logger.info(f"Auto-Save: {'Enabled' if AUTOSAVE_ENABLED else 'Disabled'} | Timeout: {WEBSOCKET_TIMEOUT_SECONDS}s | Cleanup: {AUTOSAVE_MAX_AGE_HOURS}h")
    cleanup_old_autosaves()  # Uses global config
    logger.info("✓ Webapp initialization complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown webapp gracefully - Claude Generated"""
    logger.info("🛑 Shutting down ALIMA Webapp...")
    logger.info("✓ Webapp shutdown complete")


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
                # Send final update with JSON-serializable results
                await websocket.send_json({
                    "type": "complete",
                    "status": session.status,
                    "results": make_json_serializable(session.results),
                    "error": session.error_message,
                    "current_step": session.current_step,
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

            await websocket.send_json({
                "type": "status",
                "status": session.status,
                "current_step": session.current_step,
                "results": make_json_serializable(session.results),
                "streaming_tokens": make_json_serializable(streaming_tokens),  # Dict[step_id -> List[tokens]]
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
        # Determine if this is a partial or complete export
        is_complete = (session.status == "completed")
        status_suffix = "complete" if is_complete else "partial"

        # Create temporary JSON file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            dir=tempfile.gettempdir()
        )

        # Enhanced export data with status and metadata - Claude Generated
        export_data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "exported_at": datetime.now().isoformat(),
            "status": session.status,
            "current_step": session.current_step,
            "is_complete": is_complete,
            "input": session.input_data,
            "results": session.results,
            "autosave_timestamp": session.autosave_timestamp,
        }

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
            "results": make_json_serializable(session.results),
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
):
    """Execute pipeline analysis with direct PipelineManager - Claude Generated"""

    session = sessions[session_id]
    session.status = "running"

    try:
        # Resolve input to text - Claude Generated
        input_text = None

        if input_type == "text" and content:
            input_text = content
        elif input_type == "doi" and content:
            logger.info(f"Resolving DOI: {content}")
            input_text = await asyncio.to_thread(resolve_input_to_text, content)
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
        from src.core.pipeline_manager import PipelineManager
        pipeline_manager = PipelineManager(
            alima_manager=services['alima_manager'],
            cache_manager=services['cache_manager'],
            config_manager=config_manager
        )
        logger.info(f"Created new PipelineManager for session {session_id}")

        # Create pipeline config from preferences
        pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)

        # Define callbacks for live updates - Claude Generated
        def on_step_started(step):
            session.current_step = step.step_id
            logger.info(f"Step started: {step.step_id}")

        def on_step_completed(step):
            session.current_step = step.step_id
            logger.info(f"Step completed: {step.step_id} (type: {type(step.step_id)}, repr: {repr(step.step_id)})")

            # Update working title after initialisation - Claude Generated
            logger.info(f"Checking if step_id matches 'initialisation': {step.step_id} == 'initialisation' -> {step.step_id == 'initialisation'}")

            if step.step_id == "initialisation":
                logger.info(f"✓ Initialisation step completed, checking for working_title...")
                if pipeline_manager.current_analysis_state:
                    logger.info(f"✓ Analysis state exists")
                    if hasattr(pipeline_manager.current_analysis_state, 'working_title'):
                        wt = pipeline_manager.current_analysis_state.working_title
                        logger.info(f"✓ Found working_title attribute: '{wt}'")
                        session.working_title = wt
                        # Also update session.results for WebSocket transmission - Claude Generated
                        if not session.results:
                            session.results = {}
                        session.results['working_title'] = wt
                        logger.info(f"✅ Session working title set to: {session.working_title}")
                        logger.info(f"✅ Session results updated with working_title: {session.results.get('working_title')}")
                    else:
                        logger.warning("⚠️ Analysis state has no working_title attribute")
                else:
                    logger.warning("⚠️ No current_analysis_state available")
            else:
                logger.info(f"ℹ️ Not initialisation step, skipping working_title update")

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

        def on_pipeline_completed(analysis_state):
            logger.info(f"Pipeline completed, storing results")

            # Use shared extraction helper (DRY principle) - Claude Generated
            session.results = _extract_results_from_analysis_state(analysis_state)

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
            # Use info level for visibility (same as input extraction) - Claude Generated
            logger.info(f"Token [{step_id}]: {token[:30] if len(token) > 30 else token}...")

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
                )

                # Start pipeline with correct input_type - Claude Generated (FIX: Use actual input_type, not always "text")
                # This ensures PDFs use LLM-OCR and images use vision models
                logger.info(f"Starting pipeline execution with input_type={input_type}")
                pipeline_id = pipeline_manager.start_pipeline(
                    input_text,
                    input_type=input_type,  # FIXED: Use actual input_type (pdf, img, text, doi) not hardcoded "text"
                )

                # Store analysis state reference for auto-save - Claude Generated
                session.current_analysis_state = pipeline_manager.current_analysis_state

                logger.info(f"Pipeline {pipeline_id} started with input_type={input_type}")

            except Exception as e:
                logger.error(f"Pipeline execution error: {str(e)}", exc_info=True)
                session.status = "error"
                session.error_message = str(e)

        # Run in executor to avoid blocking
        await asyncio.to_thread(execute_pipeline)

    except Exception as e:
        logger.error(f"Analysis setup error: {str(e)}", exc_info=True)
        session.status = "error"
        session.error_message = str(e)
    finally:
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
                success, text_content, error = resolve_input_to_text(content, logger)
                if not success or not text_content:
                    error_msg = error or "Could not resolve DOI/URL"
                    raise ValueError(f"DOI resolution failed: {error_msg}")
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
