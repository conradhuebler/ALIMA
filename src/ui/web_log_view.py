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
from pathlib import Path
from typing import List

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView


logger = logging.getLogger(__name__)


def _js_str(value: str) -> str:
    """Return ``value`` as a safe JS string literal."""
    return json.dumps(value if value is not None else "")


# WP12: the render chrome (theme CSS + DOM dispatcher JS) lives in the shared
# static assets so the GUI and the webapp render identically from one source.
# The GUI inlines them at construction (lowest-risk load path in QWebEngine —
# no file:///baseUrl/CORS concerns); the webapp serves the same files. Font
# size is driven by the --alima-fs CSS custom property. - Claude Generated
_RENDER_ASSET_DIR = Path(__file__).resolve().parent.parent / "webapp" / "static"


def _load_render_asset(name: str) -> str:
    """Read a shared render asset (alima_render.{css,js}) as text."""
    try:
        return (_RENDER_ASSET_DIR / name).read_text(encoding="utf-8")
    except OSError:
        logger.error("WebLogView: could not load render asset %s", name, exc_info=True)
        return ""


# Document-level chrome that is GUI-only (the whole WebLogView document is the
# render surface). The shared alima_render.css is scoped under #log so it can be
# safely loaded into the multi-element webapp page; these page-level rules are
# not, so they live here. - Claude Generated
_DOC_CHROME_CSS = """
html, body {{ margin: 0; padding: 0; background: #1e1e1e; --alima-fs: {fs}pt; }}
::-webkit-scrollbar {{ width: 12px; }}
::-webkit-scrollbar-track {{ background: #2d2d2d; }}
::-webkit-scrollbar-thumb {{ background: #555; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: #777; }}
"""

_SCAFFOLD_TEMPLATE = (
    "<!DOCTYPE html>\n"
    '<html><head><meta charset="utf-8"><style>\n{doc_css}\n{css}\n</style></head>\n'
    '<body><div id="log"></div>\n'
    "<script>window.__alimaI18n = {i18n};</script>\n"
    "<script>\n{js}\n</script></body></html>"
)


def _build_scaffold_html(base_font_pt: int) -> str:
    """Assemble the WebLogView document from the shared CSS + JS assets."""
    from ..utils.i18n import catalog_for_js

    fs = max(8, int(base_font_pt))
    return _SCAFFOLD_TEMPLATE.format(
        doc_css=_DOC_CHROME_CSS.format(fs=fs),
        css=_load_render_asset("alima_render.css"),
        js=_load_render_asset("alima_render.js"),
        i18n=json.dumps(catalog_for_js()),
        fs=fs,
    )


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

    # No createWindow() override: all links rendered into this view are
    # first-party HTML (UnifiedMessageRenderer/its markdown/marker helpers),
    # none of which set target="_blank" — see unified_message_renderer.py.
    # A target="_blank" anchor would trigger createWindow() instead of
    # acceptNavigationRequest (QWebEngineView is a real browser engine and
    # tries to open an actual new tab), which isn't wired to anything here
    # and would silently do nothing — that's the actual bug fixed by NOT
    # emitting target="_blank" in the first place, rather than trying to
    # intercept the new-window path (a urlChanged/temp-page idiom was tried
    # and discarded: it double-fired in testing). - Claude Generated


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
            _build_scaffold_html(base_font_pt),
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
        """Update the base font size (pt) of the log document (--alima-fs var)."""
        self._run_js(
            f"document.body.style.setProperty('--alima-fs', '{max(8, int(pt))}pt');"
        )

    def scroll_to_bottom(self) -> None:
        """Throttled scroll (max 20 Hz) to limit runJavaScript churn."""
        now = time.time()
        if now - self._last_scroll_time < 0.05:
            return
        self._last_scroll_time = now
        self._run_js("maybeScroll();")
