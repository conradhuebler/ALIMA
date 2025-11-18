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

    def add_temp_file(self, path: str):
        """Track temporary files for cleanup - Claude Generated"""
        self.temp_files.append(path)

    def add_streaming_token(self, token: str, step_id: str):
        """Add token to streaming buffer - Claude Generated"""
        if step_id not in self.streaming_buffer:
            self.streaming_buffer[step_id] = []
        self.streaming_buffer[step_id].append(token)

    def get_and_clear_streaming_buffer(self) -> dict:
        """Get all buffered tokens and clear - Claude Generated"""
        result = dict(self.streaming_buffer)
        self.streaming_buffer.clear()
        return result

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
    # Get streaming tokens for polling clients
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

    # READ FILE CONTENTS IMMEDIATELY before creating background task - Claude Generated (Defensive)
    # This prevents "read of closed file" error that occurs when UploadFile is passed to background task
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

    # Start analysis in background with file contents, not the UploadFile object
    asyncio.create_task(run_analysis(session_id, input_type, content, file_contents, file.filename if file else None))

    return {"session_id": session_id, "status": "started"}


@app.post("/api/input/{session_id}")
async def process_input_only(
    session_id: str,
    input_type: str = Form(...),  # "text", "doi", "pdf", "img"
    content: Optional[str] = Form(None),  # For text/doi
    file: Optional[UploadFile] = File(None),  # For pdf/img
) -> dict:
    """Process only the input step (text extraction/OCR) - Claude Generated"""

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

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

            # Always send status update (every 500ms) - Claude Generated
            # Include streaming tokens buffered since last update
            streaming_tokens = session.get_and_clear_streaming_buffer()

            await websocket.send_json({
                "type": "status",
                "status": session.status,
                "current_step": session.current_step,
                "results": session.results,
                "streaming_tokens": streaming_tokens,  # Dict[step_id -> List[tokens]]
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
        pipeline_manager = services['pipeline_manager']

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
            # Extract results from analysis state (same as GUI) - Claude Generated

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

            # Extract search results
            search_results = getattr(analysis_state, 'search_results', [])

            # Extract DK search results
            dk_search_results = getattr(analysis_state, 'dk_search_results', [])

            # Store formatted results for JSON export
            session.results = {
                "original_abstract": original_abstract,
                "initial_keywords": initial_keywords,
                "final_keywords": final_keywords,
                "dk_classifications": dk_classifications,
                "dk_search_results": dk_search_results,
                "search_results_count": len(search_results) if isinstance(search_results, list) else 0,
                "full_analysis_state": {
                    "has_final_llm_analysis": bool(hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis),
                }
            }

            # Log extracted results
            logger.info(f"Extracted results - keywords: {len(final_keywords)}, classifications: {len(dk_classifications)}, initial: {len(initial_keywords)}")

            session.status = "completed"
            session.current_step = "classification"

        def on_stream_token(token: str, step_id: str = ""):
            """Handle token streaming - buffer tokens for WebSocket - Claude Generated"""
            # Buffer tokens by step for periodic transmission via WebSocket
            if step_id:
                session.add_streaming_token(token, step_id)
            logger.debug(f"Token [{step_id}]: {token[:30] if len(token) > 30 else token}...")

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

                # Start pipeline with correct input_type - Claude Generated (FIX: Use actual input_type, not always "text")
                # This ensures PDFs use LLM-OCR and images use vision models
                logger.info(f"Starting pipeline execution with input_type={input_type}")
                pipeline_id = pipeline_manager.start_pipeline(
                    input_text,
                    input_type=input_type,  # FIXED: Use actual input_type (pdf, img, text, doi) not hardcoded "text"
                )
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
            session.current_step = "input"
            if hasattr(session, 'streaming_buffer'):
                session.streaming_buffer['input'].append(message)
            logger.info(f"[Stream] {message}")

        def execute_extraction():
            # Prepare input source - Claude Generated
            input_source = None

            if input_type == "text" and content:
                input_source = content
            elif input_type == "doi" and content:
                input_source = content
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

            # Call execute_input_extraction (same as pipeline does) - Claude Generated
            logger.info(f"Executing input extraction with input_type={input_type} from {input_source[:50]}...")
            extracted_text, source_info, extraction_method = execute_input_extraction(
                llm_service=llm_service,
                input_source=input_source,
                input_type=input_type if input_type != "img" else "image",  # pipeline uses "image" not "img"
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
