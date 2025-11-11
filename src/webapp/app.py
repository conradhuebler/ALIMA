"""
ALIMA Webapp - FastAPI Backend
Claude Generated - Pipeline widget as web interface
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import subprocess
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Import ALIMA Pipeline components - Claude Generated
from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.core.alima_manager import AlimaManager
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import ConfigManager
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.pipeline_utils import PipelineJsonManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

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

# Store active sessions and their results
sessions: dict = {}


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

    def add_temp_file(self, path: str):
        """Track temporary files for cleanup - Claude Generated"""
        self.temp_files.append(path)

    def cleanup(self):
        """Clean up temporary files - Claude Generated"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not cleanup {temp_file}: {e}")


@app.get("/")
async def root():
    """Serve main page - Claude Generated"""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


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
    return {
        "session_id": session.session_id,
        "status": session.status,
        "current_step": session.current_step,
        "created_at": session.created_at,
        "results": session.results,
        "error_message": session.error_message,
    }


@app.post("/api/analyze/{session_id}")
async def start_analysis(
    session_id: str,
    input_type: str = Form(...),  # "text", "doi", "pdf", "img"
    content: Optional[str] = Form(None),  # For text/doi
    file: Optional[UploadFile] = File(None),  # For pdf/img
) -> dict:
    """Start pipeline analysis - Claude Generated"""

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if session.status == "running":
        raise HTTPException(status_code=400, detail="Analysis already running")

    session.status = "running"
    session.input_data = {"type": input_type, "content": content}

    # Start analysis in background
    asyncio.create_task(run_analysis(session_id, input_type, content, file))

    return {"session_id": session_id, "status": "started"}


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
        max_idle = 600  # 5 minutes of inactivity = timeout (600 * 0.5s)

        while True:
            # Check if analysis is complete
            if session.status not in ["running", "idle"]:
                logger.info(f"Session {session_id} status changed to {session.status}")
                # Send final update
                await websocket.send_json({
                    "type": "complete",
                    "status": session.status,
                    "results": session.results,
                    "error": session.error_message,
                    "current_step": session.current_step,
                })
                break

            # Always send status update (every 500ms)
            await websocket.send_json({
                "type": "status",
                "status": session.status,
                "current_step": session.current_step,
                "results": session.results,
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
    """Export analysis results - Claude Generated"""

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if not session.results:
        raise HTTPException(status_code=400, detail="No results available")

    if format == "json":
        # Create temporary JSON file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            dir=tempfile.gettempdir()
        )

        export_data = {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "input": session.input_data,
            "results": session.results,
        }

        json.dump(export_data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()

        session.add_temp_file(temp_file.name)

        return FileResponse(
            temp_file.name,
            filename=f"alima_analysis_{session.session_id}.json",
            media_type="application/json"
        )

    raise HTTPException(status_code=400, detail=f"Format not supported: {format}")


async def run_analysis(
    session_id: str,
    input_type: str,
    content: Optional[str],
    file: Optional[UploadFile],
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
        elif input_type == "pdf" and file:
            # Save and extract from PDF
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            session.add_temp_file(temp_file.name)
            logger.info(f"Extracting text from PDF: {temp_file.name}")
            input_text = await asyncio.to_thread(resolve_input_to_text, temp_file.name)
        elif input_type == "img" and file:
            # Save and extract from image
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            session.add_temp_file(temp_file.name)
            logger.info(f"Analyzing image: {temp_file.name}")
            input_text = await asyncio.to_thread(resolve_input_to_text, temp_file.name)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if not input_text:
            raise ValueError("Could not extract text from input")

        logger.info(f"Input text extracted ({len(input_text)} chars)")
        session.input_data = {"type": input_type, "text_preview": input_text[:100]}

        # Initialize Pipeline Components - Claude Generated
        config_manager = ConfigManager()
        alima_manager = AlimaManager(config_manager)
        pipeline_manager = PipelineManager(alima_manager, config_manager)
        cache_manager = UnifiedKnowledgeManager()

        # Create pipeline config from preferences
        pipeline_config = PipelineConfig.create_from_provider_preferences(config_manager)

        # Define callbacks for live updates - Claude Generated
        def on_step_started(step):
            session.current_step = step.step_id
            logger.info(f"Step started: {step.step_id}")

        def on_step_completed(step):
            session.current_step = step.step_id
            logger.info(f"Step completed: {step.step_id}")

        def on_step_error(step, error_msg):
            session.current_step = step.step_id
            session.error_message = error_msg
            logger.error(f"Step error: {step.step_id}: {error_msg}")

        def on_pipeline_completed(analysis_state):
            logger.info(f"Pipeline completed, storing results")
            # Extract results from analysis state
            session.results = {
                "keywords": analysis_state.final_gnd_schlagworte if hasattr(analysis_state, 'final_gnd_schlagworte') else [],
                "dk_classification": analysis_state.classifications if hasattr(analysis_state, 'classifications') else [],
                "analysis_state": analysis_state,
            }
            session.status = "completed"
            session.current_step = "classification"

        def on_stream_token(token: str, step_id: str = ""):
            """Handle token streaming - Claude Generated"""
            logger.debug(f"Token [{step_id}]: {token[:30]}...")

        # Run pipeline in background thread - Claude Generated
        def execute_pipeline():
            try:
                # Set up callbacks
                pipeline_manager.set_callbacks(
                    step_started=on_step_started,
                    step_completed=on_step_completed,
                    step_error=on_step_error,
                    pipeline_completed=on_pipeline_completed,
                    stream_callback=on_stream_token,
                )

                # Start pipeline
                logger.info("Starting pipeline execution...")
                pipeline_id = pipeline_manager.start_pipeline(
                    input_text,
                    input_type="text",
                )
                logger.info(f"Pipeline {pipeline_id} started")

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
