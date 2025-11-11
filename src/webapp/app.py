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

    try:
        while session.status in ["running"]:
            # Send current state
            await websocket.send_json({
                "type": "status",
                "status": session.status,
                "current_step": session.current_step,
                "results": session.results,
            })

            # Wait before next update
            await asyncio.sleep(0.5)

        # Final update
        await websocket.send_json({
            "type": "complete",
            "status": session.status,
            "results": session.results,
            "error": session.error_message,
        })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")


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
    """Execute pipeline analysis via CLI - Claude Generated"""

    session = sessions[session_id]

    try:
        # Prepare input for CLI
        cli_args = [sys.executable, "src/alima_cli.py", "pipeline"]

        if input_type == "text" and content:
            cli_args.extend(["--input-text", content])
        elif input_type == "doi" and content:
            cli_args.extend(["--doi", content])
        elif input_type == "pdf" and file:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            session.add_temp_file(temp_file.name)
            cli_args.extend(["--pdf", temp_file.name])
        elif input_type == "img" and file:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            contents = await file.read()
            temp_file.write(contents)
            temp_file.close()
            session.add_temp_file(temp_file.name)
            cli_args.extend(["--image", temp_file.name])
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        # Output to temp JSON file
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        output_file.close()
        session.add_temp_file(output_file.name)
        cli_args.extend(["--output-json", output_file.name])

        # Run CLI command
        session.current_step = "initialisation"

        logger.info(f"Running: {' '.join(cli_args)}")

        process = await asyncio.create_subprocess_exec(
            *cli_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )

        session.process = process
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace')
            logger.error(f"CLI error: {error_msg}")
            session.status = "error"
            session.error_message = error_msg
            return

        # Read results from output file
        if os.path.exists(output_file.name):
            with open(output_file.name, 'r', encoding='utf-8') as f:
                session.results = json.load(f)

        session.status = "completed"
        session.current_step = "classification"

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
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
