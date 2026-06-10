"""WebLogViewTransport — GUI render-event transport (WP12). Claude Generated.

Maps the JSON render events emitted by
:class:`~src.ui.unified_message_renderer.UnifiedMessageRenderer` onto the
existing :class:`~src.ui.web_log_view.WebLogView` public API (which issues the
``page().runJavaScript(...)`` calls against the shared ``alima_render.js``
dispatcher). This is the GUI half of the two-transport design; the webapp half
broadcasts the same events over its WebSocket.

``WebLogView`` is unchanged — this adapter is the only new GUI-side piece.
Control operations (autoscroll / scroll / font) are passed through directly
rather than modelled as events (the event vocabulary is content-only).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.core import render_events as ev

logger = logging.getLogger(__name__)


class WebLogViewTransport:
    """Render-event sink that drives a :class:`WebLogView`."""

    def __init__(self, web_view: Any) -> None:
        self._view = web_view

    def send(self, event: Dict[str, Any]) -> None:
        t = event.get("type")
        v = self._view
        if t == ev.BLOCK:
            v.append_block(event["html"])
        elif t == ev.COLLAPSIBLE:
            v.append_collapsible(
                event["id"], event["summary"], event.get("body", ""), event.get("open", False)
            )
        elif t == ev.COLLAPSIBLE_UPDATE:
            v.update_collapsible(event["id"], event["summary"], event.get("body", ""))
        elif t == ev.ASSISTANT_OPEN:
            v.open_assistant(event["header"])
        elif t == ev.ASSISTANT_TOKEN:
            v.append_token(event["text"])
        elif t == ev.ASSISTANT_FINALIZE:
            v.finalize_assistant(event["html"])
        elif t == ev.STREAM_OPEN:
            v.open_stream_block(event["id"], event["summary"])
        elif t == ev.STREAM_TOKEN:
            v.append_stream_block(event["text"])
        elif t == ev.STREAM_CLOSE:
            v.close_stream_block(event["id"], event["summary"], event.get("collapse", True))
        elif t == ev.CLEAR:
            v.clear_log()
        else:
            logger.warning("WebLogViewTransport: unknown render event type %r", t)

    # -- control passthroughs (not part of the event vocabulary) -------------
    def set_autoscroll(self, enabled: bool) -> None:
        self._view.set_autoscroll(enabled)

    def scroll_to_bottom(self) -> None:
        self._view.scroll_to_bottom()

    def set_font_pt(self, pt: int) -> None:
        self._view.set_font_pt(pt)
