"""Render-event protocol for the unified render layer (WP12). Claude Generated.

A small, versioned, **JSON-serializable** event vocabulary that decouples the
single producer (:class:`~src.ui.unified_message_renderer.UnifiedMessageRenderer`)
from the frontend transport. Each event ``type`` maps 1:1 to a render function
in the shared ``src/webapp/static/alima_render.js`` DOM dispatcher, so the GUI
(``WebLogView.runJavaScript``) and the webapp (WebSocket broadcast) render the
identical chrome from the identical event stream.

Design constraints (see ``docs/wp12_unified_render_layer.md`` §6):

* **Append-only + idempotent per ``id``** — a webapp client can replay the
  buffered event stream on reconnect without corrupting the DOM.
* **Content-only** — scroll/autoscroll/font are transport *control* methods,
  not events, so the vocabulary stays purely about rendered content.
* **Ignorable types** — GUI-only chrome (mutation proposals) is tagged so a
  Tier-3 frontend (webapp) can drop it. ``block`` events carry an optional
  ``kind`` for semantic filtering (``pipeline_log`` / ``html_block`` /
  ``system`` / ``user_bubble`` / ``tool_marker`` / ``proposal``).

This module is intentionally Qt-free so the webapp backend can import it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Protocol version — bump on a breaking change to the event shape so a client
# can detect a mismatch. Carried out-of-band by the transport, not per-event.
PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# Event type constants (each maps to one alima_render.js function)
# ---------------------------------------------------------------------------
BLOCK = "block"                          # appendBlock(html)
COLLAPSIBLE = "collapsible"              # appendCollapsible(id, summary, body, open)
COLLAPSIBLE_UPDATE = "collapsible_update"  # updateCollapsible(id, summary, body)
ASSISTANT_OPEN = "assistant_open"        # openAssistant(header)
ASSISTANT_TOKEN = "assistant_token"      # appendToken(text)
ASSISTANT_FINALIZE = "assistant_finalize"  # finalizeAssistant(html)
STREAM_OPEN = "stream_open"              # openStreamBlock(id, summary)
STREAM_TOKEN = "stream_token"            # appendStreamBlock(text)
STREAM_CLOSE = "stream_close"            # closeStreamBlock(id, summary, collapse)
CLEAR = "clear"                          # clearLog()
TYPING = "typing"                        # showTyping(model) / hideTyping()

# Semantic ``kind`` tags for BLOCK events (optional; for frontend filtering).
KIND_PIPELINE_LOG = "pipeline_log"
KIND_HTML_BLOCK = "html_block"
KIND_SYSTEM = "system"
KIND_USER_BUBBLE = "user_bubble"
KIND_TOOL_MARKER = "tool_marker"
KIND_PROPOSAL = "proposal"  # GUI-only (mutation:// anchors) — webapp ignores it

# Block kinds a Tier-3 frontend (webapp) may safely drop.
GUI_ONLY_KINDS = frozenset({KIND_PROPOSAL})


# ---------------------------------------------------------------------------
# Builders (return plain dicts)
# ---------------------------------------------------------------------------
def block(html: str, kind: Optional[str] = None) -> Dict[str, Any]:
    ev: Dict[str, Any] = {"type": BLOCK, "html": html}
    if kind:
        ev["kind"] = kind
    return ev


def collapsible(
    block_id: str, summary: str, body: str, open_: bool, kind: Optional[str] = None
) -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "type": COLLAPSIBLE,
        "id": block_id,
        "summary": summary,
        "body": body or "",
        "open": bool(open_),
    }
    if kind:
        # Additive (no PROTOCOL_VERSION bump): stale clients ignore the extra
        # field. "error" switches on the red error chrome. - Claude Generated
        ev["kind"] = kind
    return ev


def collapsible_update(
    block_id: str, summary: str, body: str, kind: Optional[str] = None
) -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "type": COLLAPSIBLE_UPDATE,
        "id": block_id,
        "summary": summary,
        "body": body or "",
    }
    if kind:
        ev["kind"] = kind
    return ev


def assistant_open(header: str) -> Dict[str, Any]:
    return {"type": ASSISTANT_OPEN, "header": header}


def assistant_token(text: str) -> Dict[str, Any]:
    return {"type": ASSISTANT_TOKEN, "text": text}


def assistant_finalize(html: str) -> Dict[str, Any]:
    return {"type": ASSISTANT_FINALIZE, "html": html}


def stream_open(block_id: str, summary: str) -> Dict[str, Any]:
    return {"type": STREAM_OPEN, "id": block_id, "summary": summary}


def stream_token(text: str) -> Dict[str, Any]:
    return {"type": STREAM_TOKEN, "text": text}


def stream_close(block_id: str, summary: str, collapse: bool = True) -> Dict[str, Any]:
    return {
        "type": STREAM_CLOSE,
        "id": block_id,
        "summary": summary,
        "collapse": bool(collapse),
    }


def clear() -> Dict[str, Any]:
    return {"type": CLEAR}


def typing(model: Optional[str] = None, active: bool = True) -> Dict[str, Any]:
    return {"type": TYPING, "model": model or "", "active": bool(active)}


# ---------------------------------------------------------------------------
# Transport interface + a test/util double
# ---------------------------------------------------------------------------
try:  # Protocol is stdlib (3.8+); guard for very old interpreters only.
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class RenderTransport(Protocol):
        """A sink for render events. Implementations: ``WebLogViewTransport``
        (GUI) and the webapp's per-session WebSocket transport."""

        def send(self, event: Dict[str, Any]) -> None: ...
except ImportError:  # pragma: no cover
    RenderTransport = object  # type: ignore


class MockTransport:
    """Records every emitted event. Used by renderer protocol tests and by the
    webapp's headless adapter scaffolding. Qt-free."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def send(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    # -- control passthroughs (no-ops; control is not part of the event log) --
    def set_autoscroll(self, enabled: bool) -> None:
        pass

    def scroll_to_bottom(self) -> None:
        pass

    # -- test helpers --------------------------------------------------------
    def types(self) -> List[str]:
        return [e.get("type") for e in self.events]

    def of_type(self, type_: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("type") == type_]
