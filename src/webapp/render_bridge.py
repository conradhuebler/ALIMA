"""WP12 render bridge for the webapp — Session-buffer render transport + the
AlimaStateBus → UnifiedMessageRenderer subscriber.

Claude Generated — extracted verbatim from ``app.py`` during the F-6 god-file
split. ``Session`` is referenced only as a string forward-ref (TYPE_CHECKING
import) so there is no runtime dependency on session_state and no import cycle.
Heavy Qt/state-bus deps stay lazily imported inside the methods. ``app.py``
re-exports these names for backward compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from src.webapp.session_state import Session

logger = logging.getLogger(__name__)


class _HeadlessAutoScroll:
    """Minimal QCheckBox stand-in for the headless UnifiedMessageRenderer (WP12).

    The renderer's only use of the checkbox is ``isChecked()`` plus a
    ``toggled.connect(...)`` wire-up; neither matters server-side.
    """

    class _Signal:
        def connect(self, *_args, **_kwargs):  # noqa: D401
            pass

    def __init__(self):
        self.toggled = self._Signal()

    def isChecked(self) -> bool:  # noqa: N802 (Qt-style name)
        return False


class WebSocketRenderTransport:
    """RenderTransport that appends render events to a Session buffer (WP12).

    The WebSocket handler broadcasts the buffer to connected clients with replay
    on reconnect. Control ops (autoscroll/scroll) are server-side no-ops.
    """

    def __init__(self, session: "Session"):
        self._session = session

    def send(self, event: dict) -> None:
        self._session.append_render_event(event)

    def set_autoscroll(self, enabled: bool) -> None:
        pass

    def scroll_to_bottom(self) -> None:
        pass


def _build_session_renderer(session: "Session"):
    """Per-session UnifiedMessageRenderer wired to a WebSocket render transport.

    WP12: the webapp drives the *same* producer as the GUI (single source of the
    chrome), but the events flow over the WebSocket instead of runJavaScript.
    The import is lazy so QtWidgets is only pulled in when a pipeline runs.
    """
    from src.ui.unified_message_renderer import UnifiedMessageRenderer

    renderer = UnifiedMessageRenderer(
        WebSocketRenderTransport(session), _HeadlessAutoScroll()
    )
    # Webapp parity with the GUI panel: wire the catalog web-OPAC base + hosts so
    # <<CAT:rsn|…>> markers become links here too. Degrades silently. Claude Generated.
    try:
        from src.core.search.factory import catalog_web_bases
        renderer.configure_catalog(*catalog_web_bases())
    except Exception:
        pass
    return renderer


class _SessionBusSubscriber:
    """Session-local AlimaStateBus subscriber for agentic pipeline + chat logs.

    Bridges ``tool.called``, ``tool.result`` and ``state.pipeline_*`` bus events
    into the per-session :class:`UnifiedMessageRenderer`. The webapp has no Qt
    event loop, so the bus falls back to direct handler dispatch; this class is
    therefore instantiated per-run and unsubscribed when the run ends to avoid
    leaking handlers on the singleton bus.

    Claude Generated (Phase 4).
    """

    def __init__(self, renderer):
        self._renderer = renderer
        self._bus = None
        # Pipeline-step block state
        self._step_tool_id: Optional[str] = None
        self._open_step_status: List[str] = []
        self._pipeline_step_open: bool = False
        # Agentic prompt collapsible state
        self._prompt_blocks: Dict[str, str] = {}
        self._prompt_meta: Dict[str, str] = {}
        # Bus tool-call id -> renderer tool-call id
        self._bus_tool_ids: Dict[str, str] = {}

        # Store bound handlers so subscribe/unsubscribe pairs match by identity.
        self._on_tool_called = self._handle_tool_called
        self._on_tool_result = self._handle_tool_result
        self._on_pipeline_step = self._handle_pipeline_step
        self._on_pipeline_prompt = self._handle_pipeline_prompt
        self._on_pipeline_prompt_done = self._handle_pipeline_prompt_done
        self._on_pipeline_completed = self._handle_pipeline_completed
        self._on_pipeline_started = self._handle_pipeline_started

    def subscribe(self) -> None:
        from src.core.state_bus import AlimaStateBus

        self._bus = AlimaStateBus()
        self._bus.subscribe("tool.called", self._on_tool_called)
        self._bus.subscribe("tool.result", self._on_tool_result)
        self._bus.subscribe("state.pipeline_step", self._on_pipeline_step)
        self._bus.subscribe("state.pipeline_prompt", self._on_pipeline_prompt)
        self._bus.subscribe(
            "state.pipeline_prompt_done", self._on_pipeline_prompt_done
        )
        self._bus.subscribe("state.pipeline_completed", self._on_pipeline_completed)
        self._bus.subscribe("state.pipeline_started", self._on_pipeline_started)

    def unsubscribe(self) -> None:
        if self._bus is None:
            return
        try:
            self._bus.unsubscribe("tool.called", self._on_tool_called)
            self._bus.unsubscribe("tool.result", self._on_tool_result)
            self._bus.unsubscribe("state.pipeline_step", self._on_pipeline_step)
            self._bus.unsubscribe("state.pipeline_prompt", self._on_pipeline_prompt)
            self._bus.unsubscribe(
                "state.pipeline_prompt_done", self._on_pipeline_prompt_done
            )
            self._bus.unsubscribe(
                "state.pipeline_completed", self._on_pipeline_completed
            )
            self._bus.unsubscribe("state.pipeline_started", self._on_pipeline_started)
        except Exception:
            logger.exception("SessionBusSubscriber unsubscribe failed")
        finally:
            self._bus = None
            self._step_tool_id = None
            self._open_step_status.clear()
            self._pipeline_step_open = False
            self._prompt_blocks.clear()
            self._prompt_meta.clear()
            self._bus_tool_ids.clear()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_tool_called(self, payload: Dict[str, Any]) -> None:
        bus_id = (payload or {}).get("id") or ""
        name = (payload or {}).get("name") or ""
        args = (payload or {}).get("arguments") or {}
        tool_id = self._renderer.render_tool_call(name, args)
        if bus_id:
            self._bus_tool_ids[bus_id] = tool_id

    def _handle_tool_result(self, payload: Dict[str, Any]) -> None:
        bus_id = (payload or {}).get("id") or ""
        result = (payload or {}).get("result") or ""
        raw_status = (payload or {}).get("status") or "success"
        status = "success" if raw_status == "ok" else raw_status

        # Register tool-result URLs as trusted so pre-formatted GND/catalog links
        # aren't flagged as external (GUI parity). Claude Generated.
        try:
            import json as _json
            from src.core.url_utils import extract_urls_from_json
            self._renderer.add_trusted_urls(
                extract_urls_from_json(_json.loads(result))
            )
        except Exception:
            pass

        if (payload or {}).get("cache_hit"):
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"
            result = f"📦 cache: {preview}"

        tool_id = self._bus_tool_ids.pop(bus_id, None) if bus_id else None
        if tool_id:
            self._renderer.render_tool_result(
                tool_id, result, status=status or "success"
            )
        else:
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "…"
            self._renderer.render_system_message(f"↳ {preview}")

    def _handle_pipeline_step(self, payload: Dict[str, Any]) -> None:
        status = payload.get("status", "")
        step_id = payload.get("step_id", "") or "?"
        name = payload.get("name", "") or step_id
        tool_name = f"pipeline.{step_id}"
        args = {
            "step": step_id,
            "name": name,
            "tool": payload.get("tool", ""),
        }

        if status == "running":
            self._step_tool_id = self._renderer.render_tool_call(tool_name, args)
            self._open_step_status.clear()
            self._pipeline_step_open = True
            return

        result_status = "success" if status == "completed" else "error"
        if self._open_step_status:
            result_text = "\n".join(self._open_step_status)
        else:
            result_text = f"{status or 'done'}: {name}"
        self._open_step_status.clear()
        self._pipeline_step_open = False

        if self._step_tool_id:
            self._renderer.render_tool_result(
                self._step_tool_id, result_text, status=result_status
            )
            self._step_tool_id = None
        else:
            tid = self._renderer.render_tool_call(tool_name, args)
            self._renderer.render_tool_result(tid, result_text, status=result_status)

    def _handle_pipeline_prompt(self, payload: Dict[str, Any]) -> None:
        prompt_id = str(payload.get("prompt_id", "") or "")
        step_id = payload.get("step_id", "") or "?"
        kind = payload.get("kind", "input") or "input"
        system = payload.get("system", "") or ""
        user = payload.get("user", "") or ""
        ts = str(payload.get("timestamp", "") or "")
        ts_hms = ts.split("T")[-1] if "T" in ts else ts
        provider = payload.get("provider", "") or ""
        model = payload.get("model", "") or ""

        meta_parts = []
        if ts_hms:
            meta_parts.append(ts_hms)
        pm = "/".join(p for p in (provider, model) if p)
        if pm:
            meta_parts.append(pm)
        meta = "  ".join(meta_parts)

        if kind == "reflection":
            icon, title = "🔍", f"Reflexion '{step_id}'"
        else:
            icon, title = "📥", f"Input '{step_id}'"

        body = f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}"
        # Close any open stream block so the collapsible sits on its own line.
        self._renderer.end_streaming_line()
        tool_id = self._renderer.render_collapsible(
            title, body, collapsed=True, icon=icon, meta=meta
        )
        if prompt_id:
            self._prompt_blocks[prompt_id] = tool_id
            self._prompt_meta[prompt_id] = meta

    def _handle_pipeline_prompt_done(self, payload: Dict[str, Any]) -> None:
        prompt_id = str(payload.get("prompt_id", "") or "")
        dur = payload.get("duration_s")
        tool_id = self._prompt_blocks.get(prompt_id)
        if tool_id is None or dur is None:
            return
        base = self._prompt_meta.get(prompt_id, "")
        meta = f"{base}  ⏱ {float(dur):.1f}s" if base else f"⏱ {float(dur):.1f}s"
        self._renderer.update_collapsible_meta(tool_id, meta)

    def _handle_pipeline_completed(self, payload: Dict[str, Any]) -> None:
        label = "✅ Pipeline abgeschlossen"
        workflow = (payload or {}).get("workflow")
        if workflow:
            label += f" ({workflow})"
        self._renderer.render_system_message(label)

    def _handle_pipeline_started(self, payload: Dict[str, Any]) -> None:
        pid = (payload or {}).get("pipeline_id", "") or ""
        self._renderer.render_system_message(
            f"🚀 Pipeline gestartet{f' ({pid[:8]})' if pid else ''}"
        )
