"""Session-lifecycle endpoints for the webapp (F-6 split).

Claude Generated — moved verbatim from ``app.py``. Create / poll / clear /
cancel / abort-step / recover / delete a session. Shares the ``sessions``
registry (session_state.py) with every other router.
"""

import json
import logging
import uuid

from fastapi import APIRouter, HTTPException

from src.utils.pipeline_utils import PipelineJsonManager
from src.webapp.result_serialization import (
    extract_results_from_analysis_state as _extract_results_from_analysis_state,
    prepare_results_for_export as _prepare_results_for_export,
)
from src.webapp.session_io import make_json_serializable
from src.webapp.session_state import Session, sessions

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/session")
async def create_session() -> dict:
    """Create a new analysis session - Claude Generated"""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = Session(session_id)
    logger.info(f"Created session: {session_id}")
    return {"session_id": session_id, "status": "created"}


@router.get("/api/session/{session_id}")
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

    response = {
        "session_id": session.session_id,
        "status": session.status,
        "current_step": session.current_step,
        "created_at": session.created_at,
        "error_message": session.error_message,
        "streaming_tokens": streaming_tokens,  # Include for polling clients
        "render_events": session.get_new_render_events(),  # WP12: shared chrome
        # Full-pipeline results intentionally omitted (can be large) — polling
        # clients fetch them via /api/export/{id}.
    }
    # Exception: the extract-only flow (input_type doi/url/pdf/img) polls THIS
    # endpoint for the extracted text and expects `results.original_abstract`.
    # Those results are small, so include them here; without this the frontend
    # reads `undefined` and reports "Keine Textextraktion möglich" even though
    # extraction succeeded server-side. - Claude Generated
    if isinstance(session.results, dict) and session.results.get("input_mode") == "extraction_only":
        response["results"] = make_json_serializable(session.results)
    return response


@router.post("/api/session/{session_id}/clear")
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


@router.post("/api/session/{session_id}/cancel")
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


@router.post("/api/session/{session_id}/abort_step")
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


@router.get("/api/session/{session_id}/recover")
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


@router.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session - Claude Generated"""
    if session_id in sessions:
        session = sessions[session_id]
        session.cleanup()
        del sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")
