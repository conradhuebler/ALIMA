"""Live-progress WebSocket endpoint for the webapp (F-6 split).

Claude Generated — moved verbatim from ``app.py``. Streams pipeline status,
buffered LLM tokens and WP12 render events to a connected client, with a
heartbeat and idle-timeout. Reads the shared ``sessions`` registry.
"""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.webapp.result_serialization import (
    prepare_results_for_export as _prepare_results_for_export,
)
from src.webapp.session_io import make_json_serializable
from src.webapp.session_state import (
    WEBSOCKET_HEARTBEAT_INTERVAL,
    WEBSOCKET_TIMEOUT_SECONDS,
    sessions,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/{session_id}")
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
                # Flush the remaining streaming tokens FIRST as a small, standalone frame
                # (tokens buffered since the last 500ms poll). Delivering it ahead of the
                # large `complete` frame means it survives even if a reverse proxy delays
                # or truncates that bigger frame. Since Chat-UX 5/9 the browser renders
                # tokens from the seq-deduped render events, NOT from these frames — the
                # frames stay for the polling API and external consumers; the `complete`
                # frame still carries empty streaming_tokens for symmetry. - Claude Generated
                final_tokens = session.get_and_clear_streaming_buffer()
                if final_tokens:
                    await websocket.send_json({
                        "type": "status",
                        "status": session.status,
                        "current_step": session.current_step,
                        "current_step_status": session.current_step_status,
                        "streaming_tokens": make_json_serializable(final_tokens),
                    })
                # Flush any remaining render events (e.g. the final DK card). - WP12
                final_render, render_sent = session.get_render_events_since(render_sent)
                # Send final update with JSON-serializable results (tokens already sent above).
                await websocket.send_json({
                    "type": "complete",
                    "status": session.status,
                    "streaming_tokens": {},
                    "results": make_json_serializable(
                        _prepare_results_for_export(session.results, validate_rvk=False)
                    ),
                    "error": session.error_message,
                    "current_step": session.current_step,
                    "render_events": final_render,
                })
                # Graceful close so a reverse proxy flushes the final frame(s) before the
                # socket is torn down — an abrupt close right after send can drop the last
                # frame through mod_proxy_wstunnel. - Claude Generated
                try:
                    await websocket.close()
                except Exception:
                    pass
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
