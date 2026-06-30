"""Shared session serialization + auto-save helpers for the webapp (F-6 split).

Claude Generated — moved verbatim from ``app.py``. These helpers cross router
boundaries (export, websocket, recover, analysis) and the app lifespan, so they
live in one shared module rather than inside a single router.
"""

import json
import logging
import re
import unicodedata
from datetime import datetime
from typing import Optional

from src.utils.pipeline_utils import PipelineJsonManager
from src.webapp.session_state import AUTOSAVE_DIR, AUTOSAVE_MAX_AGE_HOURS, Session

logger = logging.getLogger(__name__)


def make_json_serializable(obj):
    """Convert sets/tuples to JSON-serializable equivalents - Claude Generated.

    Thin wrapper over the canonical ``PipelineJsonManager.convert_sets_to_lists``.
    """
    return PipelineJsonManager.convert_sets_to_lists(obj)


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
