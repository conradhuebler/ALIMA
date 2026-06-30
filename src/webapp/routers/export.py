"""Result-export endpoint for the webapp (F-6 split).

Claude Generated — moved verbatim from ``app.py``. ``GET /api/export/{id}``
streams the (partial or complete) analysis result as a JSON download.
"""

import json
import logging
import tempfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.webapp.result_serialization import build_export_payload as _build_export_payload
from src.webapp.session_io import sanitize_filename
from src.webapp.session_state import sessions

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/export/{session_id}")
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
