"""Load a saved result into a session: ``POST /api/session/{id}/load`` - Claude Generated.

Validation and rendering live in ``src/webapp/result_loading.py``; this module
only decides when the session takes the result over. A loaded result replaces
what the session showed (render log, results, title, chat history), the session
reports ``status: completed`` with ``loaded_from`` set, and the answer carries
the render events, so the browser fills its tabs without a WebSocket.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.webapp.result_loading import MAX_RESULT_FILE_BYTES, ResultFileError, load_result_file
from src.webapp.session_io import make_json_serializable
from src.webapp.session_state import Session, sessions

logger = logging.getLogger(__name__)

router = APIRouter()

_RUNNING = "Während eines Laufs kann kein Ergebnis geladen werden."


@router.post("/api/session/{session_id}/load")
async def load_result(session_id: str, file: Optional[UploadFile] = File(None)) -> dict:
    """Validate an uploaded result file and show it in the session - Claude Generated"""
    if session_id not in sessions:
        sessions[session_id] = Session(session_id)
        logger.info(f"Auto-created session {session_id} for /load")
    session = sessions[session_id]

    if session.status == "running":
        raise HTTPException(status_code=409, detail=_RUNNING)
    if file is None:
        raise HTTPException(status_code=400, detail="Keine Datei übermittelt.")

    # One byte past the limit is enough to know it is too large.
    raw = await file.read(MAX_RESULT_FILE_BYTES + 1)
    try:
        loaded = await asyncio.to_thread(load_result_file, raw, file.filename)
    except ResultFileError as exc:
        logger.info(f"Session {session_id}: result file rejected ({exc.status}): {exc.message}")
        raise HTTPException(status_code=exc.status, detail=exc.message)

    if session.status == "running":  # a run started while the file was read
        raise HTTPException(status_code=409, detail=_RUNNING)

    replaced = bool(session.render_buffer or session.results)
    session.start_new_result()
    session.chat_history = []
    session.abort_requested = False
    session.status = "completed"
    session.current_step = "classification"
    session.input_data = loaded["input_data"]
    session.results = loaded["results"]
    session.current_analysis_state = loaded["state"]
    session.working_title = getattr(loaded["state"], "working_title", None) or None
    session.loaded_from = loaded["filename"]
    for event in loaded["events"]:
        session.append_render_event(event)
    events, _ = session.get_render_events_since(0)

    logger.info(
        f"Session {session_id}: result loaded from {loaded['filename']!r} "
        f"({loaded['format']}, {len(events)} render events, replaced previous: {replaced})"
    )
    return {
        "status": "loaded",
        "session_id": session_id,
        "filename": loaded["filename"],
        "format": loaded["format"],
        "working_title": session.working_title,
        "ignored_fields": loaded["ignored_fields"],
        "replaced_previous": replaced,
        "render_events": events,
        "results": make_json_serializable(session.results),
    }
