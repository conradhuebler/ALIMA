"""WebLogView — QWebEngineView-backed log/chat surface. Claude Generated.

Replaces the single ``QTextBrowser`` + ``QTextCursor`` surgery that
``UnifiedMessageRenderer`` previously drove. Rendering is now plain HTML/CSS
in a real browser engine:

* **Collapsible blocks** are native ``<details>/<summary>`` — the toggle is
  100 % browser-side, so expanding/collapsing never re-renders a block and
  never desyncs while tokens stream in elsewhere (the regression this fixes).
* **Streaming** appends text nodes into one isolated ``<div>`` — sibling
  ``<details>`` state is untouched.
* **Markdown** is rendered once on finalize (Python ``markdown`` lib) and
  injected as innerHTML.

Link clicks (``mutation://…``, ``http(s)://…``) are routed back to Python via
:meth:`_LogPage.acceptNavigationRequest` → the :attr:`link_clicked` signal, so
the panel keeps its existing ``_on_anchor_clicked`` router. ``<details>``
toggles need no Python round-trip, so no QWebChannel bridge is required.

All public methods must be called from the UI thread (guaranteed today via Qt
signals). JS string arguments are passed through ``json.dumps`` so quotes,
newlines and unicode are escaped safely.
"""
from __future__ import annotations

import json
import logging
import time
from typing import List

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView


logger = logging.getLogger(__name__)


def _js_str(value: str) -> str:
    """Return ``value`` as a safe JS string literal."""
    return json.dumps(value if value is not None else "")


# Dark-theme scaffold. Colours ported from the former QTextBrowser stylesheet
# (#1e1e1e / #f8f8f2) and the bubble palette (#202c33 / #005c4b).
_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
html, body {{
  margin: 0; padding: 8px;
  background: #1e1e1e; color: #f8f8f2;
  font-family: 'Consolas', 'Monaco', monospace; font-size: {fs}pt;
}}
#log {{ display: flex; flex-direction: column; }}
.block {{ margin: 1px 0; word-wrap: break-word; }}
details {{ margin: 2px 0; }}
summary {{
  cursor: pointer; color: #888; font-family: monospace; font-size: 9pt;
  list-style: none; outline: none;
}}
summary::-webkit-details-marker {{ display: none; }}
summary::before {{ content: '\\25B6 '; color: #8be9fd; }}
details[open] > summary::before {{ content: '\\25BC '; color: #8be9fd; }}
.tc-body {{
  margin: 2px 0 6px 14px; font-family: monospace; font-size: 9pt;
  color: #a8a8a8; white-space: pre-wrap; word-wrap: break-word;
}}
.sl-body {{ color: #cdc6f0; font-size: 9.5pt; }}  /* live LLM stream text */
.assistant {{ margin: 6px 0; }}
.ahdr {{ color: #8be9fd; font-size: 9pt; font-style: italic; margin-bottom: 2px; }}
.stream, .rendered {{
  background: #202c33; border-radius: 6px; padding: 8px; color: #e9edef;
  max-width: 75%; word-wrap: break-word;
}}
.stream {{ white-space: pre-wrap; }}
.rendered table {{ border-collapse: collapse; margin: 4px 0; }}
.rendered td, .rendered th {{ border: 1px solid #444; padding: 2px 6px; }}
.rendered pre {{ background: #1a2228; padding: 6px; border-radius: 4px; overflow-x: auto; }}
.rendered code {{ font-family: monospace; }}
a {{ color: #5af; }}
::-webkit-scrollbar {{ width: 12px; }}
::-webkit-scrollbar-track {{ background: #2d2d2d; }}
::-webkit-scrollbar-thumb {{ background: #555; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: #777; }}
</style></head>
<body><div id="log"></div>
<script>
window.__autoscroll = true;
var curStream = null;       // live assistant bubble target
var curStreamBlock = null;  // live <details> stream-block body target

function maybeScroll() {{
  if (window.__autoscroll) {{ window.scrollTo(0, document.body.scrollHeight); }}
}}
function _log() {{ return document.getElementById('log'); }}

function appendBlock(html) {{
  var d = document.createElement('div');
  d.className = 'block';
  d.innerHTML = html;
  _log().appendChild(d);
  maybeScroll();
}}
function appendCollapsible(id, summary, body, open) {{
  var det = document.createElement('details');
  det.id = id;
  if (open) det.open = true;
  var s = document.createElement('summary');
  s.innerHTML = summary;
  var b = document.createElement('div');
  b.className = 'tc-body';
  b.innerHTML = body || '';
  det.appendChild(s);
  det.appendChild(b);
  _log().appendChild(det);
  maybeScroll();
}}
function updateCollapsible(id, summary, body) {{
  var det = document.getElementById(id);
  if (!det) return;
  var s = det.querySelector('summary');
  if (s) s.innerHTML = summary;
  var b = det.querySelector('.tc-body');
  if (b) b.innerHTML = body || '';
  maybeScroll();
}}
function openAssistant(header) {{
  var wrap = document.createElement('div');
  wrap.className = 'assistant';
  var h = document.createElement('div');
  h.className = 'ahdr';
  h.innerHTML = header;
  var s = document.createElement('div');
  s.className = 'stream';
  wrap.appendChild(h);
  wrap.appendChild(s);
  _log().appendChild(wrap);
  curStream = s;
  maybeScroll();
}}
function appendToken(text) {{
  if (!curStream) return;
  curStream.appendChild(document.createTextNode(text));
  maybeScroll();
}}
function finalizeAssistant(html) {{
  if (!curStream) return;
  curStream.className = 'rendered';
  curStream.innerHTML = html;
  curStream = null;
  maybeScroll();
}}
function openStreamBlock(id, summary) {{
  // A <details>, OPEN while the LLM streams, that collapses on close.
  var det = document.createElement('details');
  det.id = id;
  det.open = true;
  var s = document.createElement('summary');
  s.innerHTML = summary;
  var b = document.createElement('div');
  b.className = 'tc-body sl-body';
  det.appendChild(s);
  det.appendChild(b);
  _log().appendChild(det);
  curStreamBlock = b;
  maybeScroll();
}}
function appendStreamBlock(text) {{
  if (!curStreamBlock) return;
  curStreamBlock.appendChild(document.createTextNode(text));
  maybeScroll();
}}
function closeStreamBlock(id, summary, collapse) {{
  var det = document.getElementById(id);
  if (det) {{
    if (summary) {{ var s = det.querySelector('summary'); if (s) s.innerHTML = summary; }}
    det.open = !collapse;
  }}
  curStreamBlock = null;
  maybeScroll();
}}
function clearLog() {{ _log().innerHTML = ''; curStream = null; curStreamBlock = null; }}
</script></body></html>"""


class _LogPage(QWebEnginePage):
    """Page that routes link clicks back to Python instead of navigating."""

    link_clicked = pyqtSignal(QUrl)

    def acceptNavigationRequest(  # noqa: N802 (Qt override)
        self, url: QUrl, nav_type: "QWebEnginePage.NavigationType", is_main_frame: bool
    ) -> bool:
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            # mutation:// (proposal accept/reject) and http(s):// (catalog
            # links) are handled by the panel; <details> toggles never reach
            # here. Block the actual navigation.
            self.link_clicked.emit(url)
            return False
        return True


class WebLogView(QWidget):
    """QWebEngineView wrapper exposing a thin append/stream/collapse API."""

    link_clicked = pyqtSignal(QUrl)

    def __init__(self, *, base_font_pt: int = 10, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self._view = QWebEngineView(self)
        self._page = _LogPage(self._view)
        self._page.link_clicked.connect(self.link_clicked)
        self._view.setPage(self._page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._view)

        # JS calls issued before the page finishes loading are queued.
        self._ready = False
        self._pending: List[str] = []
        self._last_scroll_time = 0.0
        self._view.loadFinished.connect(self._on_load_finished)

        self._view.setHtml(
            _HTML_TEMPLATE.format(fs=max(8, int(base_font_pt))),
            QUrl("about:blank"),
        )

    # -- JS plumbing -----------------------------------------------------

    def _on_load_finished(self, ok: bool) -> None:
        self._ready = True
        if not ok:
            self.logger.warning("WebLogView scaffold failed to load")
        pending, self._pending = self._pending, []
        for code in pending:
            self._page.runJavaScript(code)

    def _run_js(self, code: str) -> None:
        if self._ready:
            self._page.runJavaScript(code)
        else:
            self._pending.append(code)

    # -- Public rendering API -------------------------------------------

    def append_block(self, html: str) -> None:
        self._run_js(f"appendBlock({_js_str(html)});")

    def append_collapsible(
        self, block_id: str, summary_html: str, body_html: str, open_: bool
    ) -> None:
        self._run_js(
            f"appendCollapsible({_js_str(block_id)}, {_js_str(summary_html)}, "
            f"{_js_str(body_html)}, {str(bool(open_)).lower()});"
        )

    def update_collapsible(
        self, block_id: str, summary_html: str, body_html: str
    ) -> None:
        self._run_js(
            f"updateCollapsible({_js_str(block_id)}, {_js_str(summary_html)}, "
            f"{_js_str(body_html)});"
        )

    def open_assistant(self, header_html: str) -> None:
        self._run_js(f"openAssistant({_js_str(header_html)});")

    def append_token(self, text: str) -> None:
        self._run_js(f"appendToken({_js_str(text)});")

    def finalize_assistant(self, html: str) -> None:
        self._run_js(f"finalizeAssistant({_js_str(html)});")

    def open_stream_block(self, block_id: str, summary_html: str) -> None:
        """Open an expanded ``<details>`` stream block; tokens append to its body."""
        self._run_js(
            f"openStreamBlock({_js_str(block_id)}, {_js_str(summary_html)});"
        )

    def append_stream_block(self, text: str) -> None:
        self._run_js(f"appendStreamBlock({_js_str(text)});")

    def close_stream_block(
        self, block_id: str, summary_html: str, collapse: bool = True
    ) -> None:
        """Update the summary and collapse (or keep open) the stream block."""
        self._run_js(
            f"closeStreamBlock({_js_str(block_id)}, {_js_str(summary_html)}, "
            f"{str(bool(collapse)).lower()});"
        )

    def clear_log(self) -> None:
        self._run_js("clearLog();")

    # -- Auto-scroll -----------------------------------------------------

    def set_autoscroll(self, enabled: bool) -> None:
        self._run_js(f"window.__autoscroll = {str(bool(enabled)).lower()};")

    def set_font_pt(self, pt: int) -> None:
        """Update the base font size (pt) of the log document."""
        self._run_js(f"document.body.style.fontSize = '{max(8, int(pt))}pt';")

    def scroll_to_bottom(self) -> None:
        """Throttled scroll (max 20 Hz) to limit runJavaScript churn."""
        now = time.time()
        if now - self._last_scroll_time < 0.05:
            return
        self._last_scroll_time = now
        self._run_js("maybeScroll();")
