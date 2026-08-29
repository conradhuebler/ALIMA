"""UnifiedMessageRenderer — single renderer for pipeline log + chat bubbles.

Claude Generated.

Builds HTML for every message role and pushes it into a :class:`WebLogView`
(QWebEngineView). Collapsible blocks are native ``<details>/<summary>`` — the
toggle is browser-side, so it never re-renders a block and never desyncs while
tokens stream in elsewhere (the regression the previous QTextCursor approach
suffered). Streaming appends text nodes into an isolated node; markdown is
rendered once on finalize.

All rendering methods must be called from the UI thread (guaranteed today via
Qt signals).

Uses MessageEntry for history tracking of *completed* messages only; streaming
tokens are renderer state until the line / bubble is finalised.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from html import escape as html_escape
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from PyQt6.QtWidgets import QCheckBox

from src.core import render_events as ev
from src.utils.i18n import t
from .message_entry import MessageEntry, MessageRole

if TYPE_CHECKING:
    # Type-only import. WebLogView pulls in QWebEngineView, which must be
    # imported before QApplication; the runtime import happens in alima_gui.py
    # and PipelineChatPanel (module level). The renderer only receives a
    # WebLogView instance, so it needs the name for annotations only.
    from .web_log_view import WebLogView
    from src.core.render_events import RenderTransport


# ----------------------------------------------------------------------
# Module-level style constants
# ----------------------------------------------------------------------

# Log levels with a dedicated CSS class in alima_render.css (Chat-UX 6/9:
# colors live in the stylesheet as --alima-* variables so the webapp light
# theme can restyle them; the renderer emits classes, not inline hex).
_PIPELINE_LOG_LEVELS = {"info", "success", "warning", "error", "step", "stream", "debug"}


class UnifiedMessageRenderer:
    """Render messages of all roles into a shared :class:`WebLogView`."""

    # Coalescing window for live thinking chunks (see append_thinking).
    _THINKING_FLUSH_S = 0.04

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        transport: "RenderTransport | WebLogView",
        auto_scroll_checkbox: QCheckBox,
    ):
        # WP12: the renderer emits JSON render events to a transport instead of
        # calling a WebLogView directly. For back-compat the historical callers
        # pass a WebLogView — auto-wrap it in a WebLogViewTransport. A caller may
        # also pass a transport directly (webapp / tests).
        if hasattr(transport, "send"):
            self.transport = transport
        else:
            from .render_transport import WebLogViewTransport
            self.transport = WebLogViewTransport(transport)
        self.auto_scroll_checkbox = auto_scroll_checkbox
        self.logger = logging.getLogger(__name__)

        # Mirror the checkbox into the transport so JS auto-scroll honours it.
        try:
            self.transport.set_autoscroll(auto_scroll_checkbox.isChecked())
            auto_scroll_checkbox.toggled.connect(self.transport.set_autoscroll)
        except Exception:
            self.logger.debug("auto-scroll checkbox wiring skipped", exc_info=True)

        # History of completed messages (streaming tokens excluded).
        self.history: List[MessageEntry] = []

        # Pipeline streaming state. The live LLM stream is rendered as an
        # expanded <details> block (replaces the old flat inline line) that
        # collapses to a one-line preview when the step ends — so everything
        # streamed is visible live, then folded away.
        self._is_streaming = False
        self._stream_block_id: Optional[str] = None
        self._stream_text = ""
        self._stream_title = ""

        # Assistant bubble streaming state. ``_assistant_cell_cursor`` is kept
        # only as an "open" sentinel (truthy while a bubble is open, None when
        # closed) for backward-compat with PipelineChatPanel, which sets it to
        # None to force-close a bubble. It no longer holds a QTextCursor.
        self._assistant_block_open = False
        self._assistant_cell_cursor: Optional[bool] = None
        self._current_assistant_text = ""

        # Auto-scroll throttle (kept for PipelineChatPanel introspection).
        self._last_scroll_time = 0.0

        # Collapsible block state (id -> dict). ``expanded`` is a server-side
        # mirror only; the real open/closed state lives in the native
        # <details> element and is owned by the user.
        self._tool_call_id = 0
        self._tool_calls: Dict[str, Dict[str, Any]] = {}
        # Live thinking: buffered chunks + when they were last flushed.
        self._thinking_pending: str = ""

        # Thinking block state: one collapsed 💭 collapsible per reasoning
        # segment; closed when the answer (or a tool block) follows. Updates
        # are throttled — a per-token collapsible_update would resend the
        # whole accumulated body each time. - Claude Generated
        self._thinking_block_id: Optional[str] = None
        self._thinking_last_update = 0.0

        # Phase E: bus id → renderer tool_id bridge (for subscribe/unsubscribe).
        self._bus_id_to_tool_id: Dict[str, str] = {}

        # P-δ.5: <<CAT:rsn|text>> marker → clickable catalog link. The web
        # base is set by the panel via configure_catalog(); empty default
        # disables the feature (markers reduced to display text).
        self._catalog_web_base: str = ""
        self._catalog_hosts: set = set()
        # URLs seen in tool results this turn — exempt from ext-link flagging.
        self._trusted_urls: set = set()
        # Authority hosts (GND/SWB) are always trusted — pre-formatted GND/SWB
        # URLs must never be flagged as suspicious external links, even when they
        # didn't arrive via a tool result. Claude Generated.
        self._authority_hosts: set = {
            "https://d-nb.info",
            "https://swb.bsz-bw.de",
        }

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_catalog_web_base(self, url: str) -> None:
        """Set the catalog web-OPAC base URL used to render <<CAT:rsn|…>> markers.

        Example: ``set_catalog_web_base("https://katalog.ub.tu-freiberg.de/Record/")``
        makes ``<<CAT:364641185|Chemoinformatics>>`` render as a link to
        ``https://katalog.ub.tu-freiberg.de/Record/0-364641185``.

        Pass an empty string to disable the feature (markers are reduced to
        their display text only — no broken links).
        """
        self._catalog_web_base = (url or "").rstrip("/")

    def set_catalog_host(self, host: str) -> None:
        """Set (replace) the primary catalog hostname for link classification. Claude Generated."""
        self._catalog_hosts = {(host or "").rstrip("/")} if host else set()

    def add_catalog_host(self, host: str) -> None:
        """Register an additional catalog hostname (e.g. from finc_web_record_url). Claude Generated."""
        h = (host or "").rstrip("/")
        if h:
            self._catalog_hosts.add(h)

    def add_trusted_urls(self, urls: Iterable[str]) -> None:
        """Register URLs from tool results as trusted for link classification.

        Trusted URLs are not flagged as ext-link even when they don't belong
        to the configured catalog host (e.g. DOIs or publisher URLs returned
        by finc records). The whitelist is cleared on each new user turn via
        render_user_bubble(). Claude Generated.
        """
        for u in urls:
            if u and isinstance(u, str) and u.startswith(("http://", "https://")):
                self._trusted_urls.add(u)

    def configure_catalog(self, web_record_url: str = "", web_search_url: str = "") -> None:
        """Wire the catalog web-OPAC base + host(s) for ``<<CAT:rsn|…>>`` markers.

        Claude Generated. Takes the resolved URLs rather than a config object, so
        the GUI panel and the webapp share one call shape (``factory.catalog_web_bases``
        does the resolving). Empty URLs degrade silently — markers reduce to plain
        text. The GND/SWB authority hosts are always trusted regardless (see
        ``__init__``).
        """
        from urllib.parse import urlparse

        web_base = web_record_url or ""
        self.set_catalog_web_base(web_base)
        for url in (web_base, web_search_url or ""):
            if url:
                p = urlparse(url)
                if p.scheme and p.netloc:
                    self.add_catalog_host(f"{p.scheme}://{p.netloc}")

    # ------------------------------------------------------------------
    # Pipeline log rendering
    # ------------------------------------------------------------------

    def render_pipeline_log(
        self,
        message: str,
        level: str = "info",
        step_id: Optional[str] = None,
    ) -> None:
        """Append a timestamped, colour-coded pipeline log line."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        lvl = level if level in _PIPELINE_LOG_LEVELS else "info"

        if step_id:
            formatted = (
                f"<span class='ts'>[{timestamp}]</span> "
                f"<span class='log-lvl--{lvl} log-step'>[{step_id.upper()}]</span> "
                f"<span class='log-lvl--{lvl}'>{self._escape_html(message)}</span>"
            )
        else:
            formatted = (
                f"<span class='ts'>[{timestamp}]</span> "
                f"<span class='log-lvl--{lvl}'>{self._escape_html(message)}</span>"
            )

        self.transport.send(ev.block(formatted, kind=ev.KIND_PIPELINE_LOG))
        self._touch_scroll()

        self.history.append(
            MessageEntry(
                role=MessageRole.PIPELINE_LOG,
                content=message,
                metadata={"level": level, "step_id": step_id},
            )
        )

    def render_streaming_token(self, token: str, step_id: str) -> None:
        """Append a streamed LLM token to the open (expanded) stream block."""
        if not self._is_streaming:
            return
        self._stream_text += token
        self.transport.send(ev.stream_token(token))
        self._touch_scroll()

    def start_streaming_line(self, step_id: str, prefix: str = "") -> None:
        """Open an expanded ``<details>`` stream block for live LLM output.

        It stays open while tokens stream in and is collapsed (with a short text
        preview) by :meth:`end_streaming_line` — so everything streamed is
        visible live and then folded away. - Claude Generated
        """
        self._tool_call_id += 1
        self._stream_block_id = f"sl_{self._tool_call_id}"
        self._stream_text = ""
        # Omit the [TAG] bracket when step_id is empty.
        tag = f"[{step_id.upper()}] " if step_id else ""
        self._stream_title = f"💬 {tag}{prefix}".strip()
        summary = self._stream_summary_html(
            self._stream_title, datetime.now().strftime("%H:%M:%S"), preview=""
        )
        self.transport.send(ev.stream_open(self._stream_block_id, summary))
        self._is_streaming = True

    def end_streaming_line(self) -> None:
        """Collapse the open stream block, keeping a one-line text preview."""
        if not self._is_streaming:
            return
        preview = " ".join((self._stream_text or "").split())
        if len(preview) > 90:
            preview = preview[:90] + "…"
        summary = self._stream_summary_html(
            self._stream_title, datetime.now().strftime("%H:%M:%S"), preview
        )
        self.transport.send(
            ev.stream_close(self._stream_block_id, summary, collapse=True)
        )
        self._is_streaming = False
        self._stream_block_id = None
        self._stream_text = ""
        self._stream_title = ""

    def _stream_summary_html(self, title: str, timestamp: str, preview: str) -> str:
        """Header for the live/collapsed stream block (no arrow — native marker)."""
        prev = (
            f' <span class="sl-preview">— {self._escape_html(preview)}</span>'
            if preview
            else ""
        )
        return (
            f'<span class="sl-title">{self._escape_html(title)}</span>'
            f' <span class="ts">[{timestamp}]</span>'
            f"{prev}"
        )

    # ------------------------------------------------------------------
    # Chat bubble rendering
    # ------------------------------------------------------------------

    def render_user_bubble(self, text: str) -> None:
        """Right-aligned green WhatsApp-style bubble."""
        self._trusted_urls.clear()  # new turn → discard previous tool-result URLs
        body = self._escape_html(text)
        html = (
            '<div class="user-bubble">'
            '<span class="user-bubble-inner">'
            f"{body}</span></div>"
        )
        self.transport.send(ev.block(html, kind=ev.KIND_USER_BUBBLE))
        self.history.append(
            MessageEntry(
                role=MessageRole.USER_BUBBLE,
                content=text,
            )
        )

    def open_assistant_bubble(self, model_label: str) -> None:
        """Open a left-aligned grey assistant bubble with model label."""
        self._close_thinking_block()
        self._current_assistant_text = ""
        header = f'🤖 {self._escape_html(model_label or t("render.assistant.model_fallback"))}'
        self.transport.send(ev.assistant_open(header))
        self._assistant_block_open = True
        self._assistant_cell_cursor = True  # "open" sentinel (back-compat)
        self._touch_scroll()

    def append_assistant_token(self, token: str) -> None:
        """Stream a token into the open assistant bubble."""
        self._close_thinking_block()
        if not self._assistant_block_open:
            return
        self._current_assistant_text += token
        self.transport.send(ev.assistant_token(token))
        self._touch_scroll()

    def finalize_assistant_bubble(self) -> None:
        """Post-render Markdown and close the assistant bubble.

        Idempotent: without an open bubble this is a no-op (no event, no
        empty history entry) — drivers may call it defensively. - Claude Generated
        """
        self._close_thinking_block()
        if not self._assistant_block_open:
            return
        md_html = ""
        if self._current_assistant_text:
            try:
                from markdown_it import MarkdownIt

                # P-δ.5: replace <<CAT:rsn|display>> and <<CLINK:url|display>>
                # markers with clickable anchors BEFORE the markdown pass.
                render_text = self._replace_cat_markers(self._current_assistant_text)
                render_text = self._replace_clink_markers(render_text)
                _md = MarkdownIt("commonmark", {"breaks": True}).enable("table")
                md_html = _md.render(render_text)
                md_html = self._classify_links(md_html)
                md_html = f'<span class="md-body">{md_html}</span>'
            except Exception:
                # Keep raw text if markdown fails.
                md_html = (
                    '<span class="md-body md-body--raw">'
                    f"{self._escape_html(self._current_assistant_text)}</span>"
                )

        self.transport.send(ev.assistant_finalize(md_html))

        self.history.append(
            MessageEntry(
                role=MessageRole.ASSISTANT_BUBBLE,
                content=self._current_assistant_text,
                metadata={"rendered_markdown": True},
            )
        )

        self._assistant_block_open = False
        self._assistant_cell_cursor = None
        self._current_assistant_text = ""
        self._touch_scroll()

    def _replace_cat_markers(self, text: str) -> str:
        """Replace ``<<CAT:rsn|display>>`` with an HTML anchor.

        Claude Generated (P-δ.5). When ``self._catalog_web_base`` is empty or
        the RSN is non-numeric the marker is reduced to the display text so
        users never see a broken anchor. Display text is HTML-escaped to avoid
        injection of arbitrary HTML by the LLM (prompt-injection mitigation).
        """
        if not text or "<<CAT:" not in text:
            return text

        def _sub(match: "re.Match[str]") -> str:
            rsn = match.group(1).strip()
            display = html_escape(match.group(2).strip(), quote=True)
            if not rsn.isdigit() or not self._catalog_web_base:
                return display  # feature disabled or malformed → display only
            url = f"{self._catalog_web_base}/0-{rsn}"
            return (
                # No target="_blank": that routes through QWebEnginePage
                # .createWindow() instead of acceptNavigationRequest(), which
                # isn't wired to anything here — the click would silently do
                # nothing. Same-window clicks are already correctly
                # intercepted and opened via QDesktopServices. - Claude Generated
                f'<a href="{url}" rel="noopener noreferrer" '
                f'class="cat-anchor">'
                f"{display}</a>"
            )

        # `|` is the separator; `[^|]+?` is non-greedy on display text.
        return re.sub(r"<<CAT:([^|\n]+)\|([^|\n]+)>>", _sub, text)

    def _replace_clink_markers(self, text: str) -> str:
        """Replace ``<<CLINK:url|display>>`` with a clickable HTML anchor.

        Accepts any http(s) URL directly — no base-URL config needed.
        Intended for catalog records whose ``web_url`` is already a full URL
        (finc, Libero OPAC). Display text is HTML-escaped (prompt-injection
        mitigation). Non-http(s) URLs degrade to display text only. Claude Generated.
        """
        if not text or "<<CLINK:" not in text:
            return text

        def _sub(match: "re.Match[str]") -> str:
            url = match.group(1).strip()
            display = html_escape(match.group(2).strip(), quote=True)
            if not url.startswith(("http://", "https://")):
                return display
            return (
                # No target="_blank" — see _replace_cat_markers. - Claude Generated
                f'<a href="{url}" rel="noopener noreferrer" '
                f'class="cat-anchor">'
                f"{display}</a>"
            )

        # \\? makes the backslash before | optional: inside Markdown tables the
        # LLM escapes | as \| to avoid splitting cells, so the marker arrives
        # as <<CLINK:url\|title>>. Lazy +? stops the URL group cleanly before
        # the optional backslash so the captured URL has no trailing \.
        return re.sub(r"<<CLINK:([^|\n]+?)\\?\|([^|\n]+?)>>", _sub, text)

    def _classify_links(self, html: str) -> str:
        """Add class='ext-link' to any <a href> that does not belong to the
        configured local catalog host or a trusted/authority host. External links
        are rendered in a warning colour by the CSS. Authority hosts (GND/SWB),
        catalog hosts and trusted tool URLs are exempt. No-op when no catalog host
        is configured (no basis to tell local from external). Claude Generated.
        """
        if not html or "<a " not in html or not self._catalog_hosts:
            return html

        hosts = self._catalog_hosts
        authority = self._authority_hosts
        trusted = self._trusted_urls

        def _sub(match: "re.Match[str]") -> str:
            href = match.group(1)
            after = match.group(2)
            # No target="_blank": that was meant to keep ALIMA's own tab
            # focused, but target="_blank" routes through QWebEnginePage
            # .createWindow() instead of acceptNavigationRequest() — which
            # isn't wired to anything in this app, so the click silently did
            # nothing. Same-window clicks are already correctly intercepted
            # and opened via QDesktopServices.openUrl(). - Claude Generated

            # Local catalog URL → mark as cat-link (book icon via CSS)
            if any(href.startswith(h) for h in hosts):
                if 'class="' in after:
                    after = after.replace('class="', 'class="cat-link ', 1)
                else:
                    after = f' class="cat-link"{after}'
                return f'<a href="{href}"{after}>'
            # GND/SWB authority host → trusted, no warning icon
            if any(href.startswith(h) for h in authority):
                return f'<a href="{href}"{after}>'
            # Trusted URL from tool result → keep as-is (no warning, no icon)
            if href in trusted:
                return f'<a href="{href}"{after}>'
            # Invented external link — inject warning class (↗ icon via CSS)
            if 'class="' in after:
                after = after.replace('class="', 'class="ext-link ', 1)
            else:
                after = f' class="ext-link"{after}'
            return f'<a href="{href}"{after}>'

        return re.sub(r'<a href="([^"]*)"([^>]*)>', _sub, html)

    # ------------------------------------------------------------------
    # Segmentation + thinking block
    # ------------------------------------------------------------------

    def _segment_break(self) -> None:
        """Close an open thinking block and assistant bubble - Claude Generated

        Called before appending a collapsible/tool/error block so it lands
        *below* the already-streamed prose in the DOM. Without this, later
        tokens keep flowing into a bubble that sits visually above blocks
        which chronologically followed it.
        """
        self._close_thinking_block()
        if self._assistant_block_open:
            self.finalize_assistant_bubble()

    def append_thinking(self, text: str) -> None:
        """Stream thinking/reasoning text into a live 💭 block - Claude Generated

        Opens the block **expanded** on first call (finalizing an open assistant
        bubble first — a thinking segment must not append below an open bubble),
        appends each chunk as it arrives, and folds the block away in
        :meth:`_close_thinking_block`, which fires on the next answer token,
        segment break or finalize. Same shape as the pipeline stream block:
        visible while it happens, out of the way afterwards.

        Chunks go through ``collapsible_append`` (a text node each) instead of
        re-rendering the whole body: the body rewrite had to be throttled to
        ~0.7s, which is what made the thinking lag behind the model. Appends are
        coalesced over ``_THINKING_FLUSH_S`` — a reasoning channel delivers
        hundreds of chunks per turn (measured: 555 in 4.3s on nemotron-3.5) and
        each one costs the Qt view a ``runJavaScript`` round trip. 40 ms is
        below what an eye resolves and cuts that by an order of magnitude.
        """
        if not text:
            return
        if self._thinking_block_id is None:
            if self._assistant_block_open:
                self.finalize_assistant_bubble()
            self._tool_call_id += 1
            tool_id = f"tc_{self._tool_call_id}"
            self._thinking_block_id = tool_id
            self._tool_calls[tool_id] = {
                "name": t("render.thinking"),
                "args": None,
                "args_preview": "",
                "duration_s": None,
                "result": "",
                "expanded": True,
                "status": "success",
                "kind": "collapsible",
                "icon": "💭",
                "meta": "",
            }
            self.transport.send(
                ev.collapsible(
                    tool_id,
                    self._tool_summary_html(tool_id),
                    "",
                    True,
                    kind="thinking",
                )
            )
            self._thinking_last_update = 0.0
        tc = self._tool_calls[self._thinking_block_id]
        tc["result"] = (tc.get("result") or "") + text
        self._thinking_pending += text
        now = time.monotonic()
        if now - self._thinking_last_update >= self._THINKING_FLUSH_S:
            self._flush_thinking_appends()
        self._touch_scroll()

    def _flush_thinking_appends(self) -> None:
        """Send the buffered thinking chunks as one append. - Claude Generated"""
        if not self._thinking_pending or not self._thinking_block_id:
            return
        pending, self._thinking_pending = self._thinking_pending, ""
        self._thinking_last_update = time.monotonic()
        self.transport.send(ev.collapsible_append(self._thinking_block_id, pending))

    def _send_thinking_update(self, open_: Optional[bool] = None) -> None:
        """Push the accumulated thinking body to the frontend - Claude Generated"""
        tool_id = self._thinking_block_id
        if not tool_id:
            return
        self.transport.send(
            ev.collapsible_update(
                tool_id,
                self._tool_summary_html(tool_id),
                self._tool_body_html(tool_id),
                kind="thinking",
                open_=open_,
            )
        )

    def close_thinking(self) -> None:
        """Fold the live 💭 block away — public entry for bus consumers.

        The agentic steps signal the end of a turn explicitly; inside a chat the
        block also closes on the next answer token or tool call. - Claude Generated
        """
        self._close_thinking_block()

    def _close_thinking_block(self) -> None:
        """Final body update + history entry, then reset state - Claude Generated"""
        if not self._thinking_block_id:
            return
        tool_id = self._thinking_block_id
        tc = self._tool_calls.get(tool_id, {})
        body = tc.get("result") or ""
        self._flush_thinking_appends()
        tc["meta"] = t("render.summary.chars", n=len(body))
        tc["expanded"] = False
        # Final body (formatted, replacing the raw appended chunks) AND the fold.
        self._send_thinking_update(open_=False)
        self._thinking_block_id = None
        self.history.append(
            MessageEntry(
                role=MessageRole.TOOL_MARKER,
                content=f"💭 {tc.get('name', '')}",
                metadata={"kind": "thinking"},
            )
        )

    # ------------------------------------------------------------------
    # Collapsible tool calls (native <details>)
    # ------------------------------------------------------------------

    def render_tool_call(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        duration_s: Optional[float] = None,
    ) -> str:
        """Render a collapsible tool-call block. Returns the tool_call_id."""
        self._segment_break()
        self._tool_call_id += 1
        tool_id = f"tc_{self._tool_call_id}"
        args_preview = self._format_tool_args(args)

        self._tool_calls[tool_id] = {
            "name": name,
            "args": args,
            "args_preview": args_preview,
            "duration_s": duration_s,
            "result": None,
            "expanded": False,
            "status": "running",  # running | success | error
        }

        self.transport.send(
            ev.collapsible(
                tool_id,
                self._tool_summary_html(tool_id),
                self._tool_body_html(tool_id),
                False,
            )
        )
        self._touch_scroll()

        self.history.append(
            MessageEntry(
                role=MessageRole.TOOL_MARKER,
                content=f"🔧 {name}({args_preview})",
                metadata={"tool_name": name, "tool_call_id": tool_id},
            )
        )
        return tool_id

    def render_tool_result(self, tool_id: str, result: str, status: str = "success") -> None:
        """Attach a result to an existing tool call and update the block."""
        tc = self._tool_calls.get(tool_id)
        if tc is None:
            # Fallback: orphan result (no matching open call).
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "…"
            self.render_system_message(f"⚠ orphan result: {preview}")
            return
        tc["result"] = result
        tc["status"] = status
        self.transport.send(
            ev.collapsible_update(
                tool_id,
                self._tool_summary_html(tool_id),
                self._tool_body_html(tool_id),
                kind="error" if status == "error" else None,
            )
        )
        self._touch_scroll()

    def render_collapsible(
        self,
        title: str,
        body: str,
        *,
        collapsed: bool = True,
        icon: str = "📄",
        meta: str = "",
    ) -> str:
        """Render a generic collapsible block (e.g. the agentic input prompt).

        ``meta`` is shown next to the header (e.g. timestamp / duration) and
        can be updated later via :meth:`update_collapsible_meta`. Returns the
        block id. - Claude Generated
        """
        self._segment_break()
        self._tool_call_id += 1
        tool_id = f"tc_{self._tool_call_id}"
        self._tool_calls[tool_id] = {
            "name": title,
            "args": None,
            "args_preview": "",
            "duration_s": None,
            "result": body,
            "expanded": not collapsed,
            "status": "success",
            "kind": "collapsible",
            "icon": icon,
            "meta": meta,
        }
        self.transport.send(
            ev.collapsible(
                tool_id,
                self._tool_summary_html(tool_id),
                self._tool_body_html(tool_id),
                not collapsed,
            )
        )
        self._touch_scroll()
        self.history.append(
            MessageEntry(
                role=MessageRole.TOOL_MARKER,
                content=f"{icon} {title}",
                metadata={"kind": "collapsible"},
            )
        )
        return tool_id

    def render_error_block(self, title: str, error_text: str) -> str:
        """Open (uncollapsed) red error block — the shared error chrome for
        failed chat turns and step failures (WP12 §9.3). Returns the block
        id. - Claude Generated"""
        self._segment_break()
        self._tool_call_id += 1
        tool_id = f"tc_{self._tool_call_id}"
        summary = (
            f'<span class="rc-error-title">❌ {self._escape_html(title)}</span>'
        )
        self.transport.send(
            ev.collapsible(
                tool_id,
                summary,
                self._escape_html(error_text or ""),
                True,
                kind="error",
            )
        )
        self._touch_scroll()
        self.history.append(
            MessageEntry(
                role=MessageRole.SYSTEM_MESSAGE,
                content=f"❌ {title}: {error_text}",
                metadata={"kind": "error"},
            )
        )
        return tool_id

    def update_collapsible_meta(self, tool_id: str, meta: str) -> None:
        """Update the header meta (timestamp/duration) of a collapsible block."""
        tc = self._tool_calls.get(tool_id)
        if tc is None or tc.get("kind") != "collapsible":
            return
        tc["meta"] = meta
        self.transport.send(
            ev.collapsible_update(
                tool_id,
                self._tool_summary_html(tool_id),
                self._tool_body_html(tool_id),
            )
        )

    def _tool_summary_html(self, tool_id: str) -> str:
        """Summary (header) HTML for a collapsible block. No arrow — the native
        ``<details>`` marker provides it (styled via CSS)."""
        tc = self._tool_calls[tool_id]
        if tc.get("kind") == "collapsible":
            title = self._escape_html(tc.get("name", ""))
            icon = tc.get("icon", "📄")
            meta = tc.get("meta", "")
            meta_html = (
                f' <span class="tc-meta">{self._escape_html(meta)}</span>'
                if meta
                else ""
            )
            return f'<span class="tc-title">{icon} {title}</span>{meta_html}'

        name = self._escape_html(tc["name"])
        args_preview = self._escape_html(tc["args_preview"])
        duration_str = f"  ({tc['duration_s']:.1f}s)" if tc.get("duration_s") else ""
        status_icon = (
            "⏳" if tc["status"] == "running"
            else ("✓" if tc["status"] == "success" else "✗")
        )
        # Result summary in the always-visible (collapsed) line so tool activity
        # is reviewable without expanding each block. - Claude Generated
        result_summary = self._result_summary(tc.get("result") or "")
        result_html = (
            f' <span class="tc-result">→ {self._escape_html(result_summary)}</span>'
            if result_summary
            else ""
        )
        return (
            f'<span class="tc-line">'
            f"🔧 {name}({args_preview})  {status_icon}{result_html}{duration_str}</span>"
        )

    @staticmethod
    def _result_summary(result: str) -> str:
        """One-line summary of a tool result for the collapsed header - Claude Generated.

        Counts records for the common finc/catalog shapes (a JSON list, or a
        dict-of-lists keyed per search term) so the collapsed line shows e.g.
        ``→ 12 Treffer``; falls back to a size hint for non-JSON results.
        """
        if not result:
            return ""
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            n = len(result)
            if n < 1024:
                return t("render.summary.chars", n=n)
            return t("render.summary.kb", n=f"{n / 1024:.1f}")
        if isinstance(parsed, list):
            return t("render.summary.hits", n=len(parsed))
        if isinstance(parsed, dict):
            list_vals = [v for v in parsed.values() if isinstance(v, list)]
            if list_vals and len(list_vals) == len(parsed):
                return t("render.summary.hits", n=sum(len(v) for v in list_vals))
            if "error" in parsed:
                return t("render.summary.error")
            return t("render.summary.fields", n=len(parsed))
        return ""

    def _tool_body_html(self, tool_id: str) -> str:
        """Body HTML for a collapsible block (escaped, pre-wrapped by .tc-body)."""
        tc = self._tool_calls[tool_id]
        result = tc.get("result") or ""
        if not result:
            return ""
        return self._escape_html(result)

    def render_system_message(self, text: str) -> None:
        """Centered italic green status line."""
        html = (
            f'<div class="system-message">'
            f"{self._escape_html(text)}</div>"
        )
        self.transport.send(ev.block(html, kind=ev.KIND_SYSTEM))
        self.history.append(
            MessageEntry(
                role=MessageRole.SYSTEM_MESSAGE,
                content=text,
            )
        )

    def show_typing(self, model_label: str = "") -> None:
        """Emit a typing indicator event."""
        self.transport.send(ev.typing(model_label, active=True))

    def hide_typing(self) -> None:
        """Emit a typing-hide event."""
        self.transport.send(ev.typing(active=False))

    def render_markdown_block(
        self, markdown_text: str, *, kind: Optional[str] = None
    ) -> None:
        """Render a Markdown string (e.g. a GFM table) as a trusted HTML block.

        For content produced by ALIMA's own formatters — not raw LLM/user
        text, so no CAT/CLINK marker handling is needed here (that's
        :meth:`finalize_assistant_bubble`'s job for the interactive chat
        path). E.g. the ``title_list_search`` workflow's duplicate-check
        report was previously only visible as unrendered pipe-table text in
        the pipeline log; this renders it as an actual ``<table>``. Falls
        back to escaped preformatted text if markdown-it fails, so a
        rendering hiccup can't break the run. - Claude Generated
        """
        if not markdown_text:
            return
        try:
            from markdown_it import MarkdownIt

            md_html = MarkdownIt("commonmark", {"breaks": True}).enable("table").render(markdown_text)
            html = f'<div class="md-body">{md_html}</div>'
        except Exception:
            html = (
                '<pre class="md-body md-body--raw">'
                f"{self._escape_html(markdown_text)}</pre>"
            )
        self.render_html_block(html, kind=kind, plain_text=markdown_text)

    def render_html_block(
        self, html: str, *, kind: Optional[str] = None, plain_text: str = ""
    ) -> None:
        """Append a pre-formatted, trusted HTML block (e.g. a DK/RVK result card).

        The caller must pass already-sanitised HTML — the shared
        ``PipelineResultFormatter`` escapes catalog titles. ``plain_text`` is
        recorded in history for export; ``kind`` tags the card (e.g.
        ``"dk_classifications"``, ``"dk_search"``).
        """
        if not html:
            return
        self.transport.send(ev.block(html, kind=ev.KIND_HTML_BLOCK))
        self.history.append(
            MessageEntry(
                role=MessageRole.RESULT_CARD,
                content=plain_text or html,
                metadata={"kind": kind},
            )
        )

    def render_proposal_bubble(
        self, audit_id: int, tool_name: str, payload: Dict[str, Any]
    ) -> None:
        """Clickable mutation-proposal block with accept / reject anchors.

        ``mutation://`` anchors are intercepted by ``WebLogView`` navigation
        handling and routed back to the panel.
        """
        title_map = {
            "propose_keyword_replacement": "🔁 Vorschlag: Keyword ersetzen",
            "propose_dk_change": "🏷️ Vorschlag: DK-Klassifikation ändern",
        }
        title = title_map.get(tool_name, f"⚠️ Mutations-Vorschlag: {tool_name}")

        if tool_name == "propose_keyword_replacement":
            old = self._escape_html(str(payload.get("old", "")))
            new = self._escape_html(str(payload.get("new", "")))
            gnd = payload.get("gnd_id") or ""
            gnd_str = (
                f" <span style='color: #888;'>(GND-ID: {self._escape_html(gnd)})</span>"
                if gnd
                else ""
            )
            diff_html = f"<b>{old}</b> → <b>{new}</b>{gnd_str}"
        elif tool_name == "propose_dk_change":
            code = self._escape_html(str(payload.get("code", "")))
            action = str(payload.get("action", ""))
            verb = "hinzufügen" if action == "add" else "entfernen"
            diff_html = f"<b>{code}</b> ({verb})"
        else:
            diff_html = self._escape_html(str(payload))

        reason = self._escape_html(str(payload.get("reason", "") or "—"))
        accept_href = f"mutation://{audit_id}/accept"
        reject_href = f"mutation://{audit_id}/reject"

        html = (
            f'<div style="margin: 6px 12px; padding: 10px; '
            f'background-color: #2d3142; border-left: 3px solid #ffb86c; '
            f'border-radius: 4px;">'
            f'<div style="color: #ffb86c; font-weight: bold; font-size: 10pt;">{title}</div>'
            f'<div style="color: #f8f8f2; margin-top: 4px;">{diff_html}</div>'
            f'<div style="color: #888; font-size: 9pt; margin-top: 4px;">'
            f"Begründung: {reason}</div>"
            f'<div style="margin-top: 8px;">'
            f'<a href="{accept_href}" style="color: #50fa7b; '
            f'text-decoration: none; padding: 4px 10px; '
            f'border: 1px solid #50fa7b; border-radius: 3px; '
            f'margin-right: 8px;">✓ Akzeptieren</a>'
            f'<a href="{reject_href}" style="color: #ff5555; '
            f'text-decoration: none; padding: 4px 10px; '
            f'border: 1px solid #ff5555; border-radius: 3px;">✗ Ablehnen</a>'
            f'<span style="color: #555; font-size: 8pt; margin-left: 8px;">'
            f"#audit_{audit_id}</span>"
            f"</div></div>"
        )
        self.transport.send(ev.block(html, kind=ev.KIND_PROPOSAL))
        self.history.append(
            MessageEntry(
                role=MessageRole.PROPOSAL_BUBBLE,
                content=f"{tool_name}: {payload}",
                metadata={"audit_id": audit_id, "tool_name": tool_name},
            )
        )

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _append_html(self, html: str) -> None:
        """Append a trusted HTML block at the end of the log."""
        self.transport.send(ev.block(html, kind=ev.KIND_HTML_BLOCK))
        self._touch_scroll()

    def append_raw_html(self, html: str) -> None:
        """Public escape-hatch for external callers."""
        self._append_html(html)

    # ------------------------------------------------------------------
    # Auto-scroll
    # ------------------------------------------------------------------

    def auto_scroll_to_bottom(self) -> None:
        """Scroll to bottom (JS honours the auto-scroll checkbox)."""
        self._touch_scroll()

    def _touch_scroll(self) -> None:
        self._last_scroll_time = time.time()
        if self.auto_scroll_checkbox.isChecked():
            self.transport.scroll_to_bottom()

    # ------------------------------------------------------------------
    # Clear / reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the log and reset all open streaming / bubble state."""
        self.transport.send(ev.clear())
        self._is_streaming = False
        self._stream_block_id = None
        self._stream_text = ""
        self._stream_title = ""
        self._assistant_block_open = False
        self._assistant_cell_cursor = None
        self._current_assistant_text = ""
        self._tool_call_id = 0
        self._tool_calls.clear()
        # History is intentionally preserved; call sites can clear it
        # explicitly if desired.

    def clear_history(self) -> None:
        """Reset the message history list."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Phase E: bus subscription (opt-in consumer)
    # ------------------------------------------------------------------

    def subscribe(self) -> None:
        """Subscribe to ``tool.called`` / ``tool.result`` on AlimaStateBus.

        This is the *embedder* bus consumer: mini-log renderers without a
        panel (``image_analysis_tab``, ``analysis_review_tab``) use it to show
        tool activity. It deliberately covers only tool events — pipeline
        step/prompt chrome is the job of the two full consumers (GUI
        ``BusEventMixin``, webapp ``_SessionBusSubscriber``; see the
        ownership note in ``_chat_panel_bus.py``). Bound methods are stored
        on ``self`` so ``unsubscribe`` can find the *same* object — the bus
        compares handlers by identity.
        """
        try:
            from src.core.state_bus import AlimaStateBus
            bus = AlimaStateBus()
            self._bus_handler_tool_called = self._on_bus_tool_called
            self._bus_handler_tool_result = self._on_bus_tool_result
            bus.subscribe("tool.called", self._bus_handler_tool_called)
            bus.subscribe("tool.result", self._bus_handler_tool_result)
        except Exception:
            self.logger.exception("UnifiedMessageRenderer.subscribe failed")

    def unsubscribe(self) -> None:
        """Reverse of :meth:`subscribe`. Idempotent and exception-safe."""
        try:
            from src.core.state_bus import AlimaStateBus
            bus = AlimaStateBus()
            tool_called = getattr(self, "_bus_handler_tool_called", None)
            tool_result = getattr(self, "_bus_handler_tool_result", None)
            if tool_called is not None:
                bus.unsubscribe("tool.called", tool_called)
            if tool_result is not None:
                bus.unsubscribe("tool.result", tool_result)
        except Exception:
            self.logger.exception("UnifiedMessageRenderer.unsubscribe failed")
        finally:
            self._bus_id_to_tool_id.clear()

    def _on_bus_tool_called(self, payload: Dict[str, Any]) -> None:
        """Bridge bus ``tool.called`` → ``render_tool_call``."""
        bus_id = (payload or {}).get("id") or ""
        name = (payload or {}).get("name") or ""
        args = (payload or {}).get("arguments") or {}
        tool_id = self.render_tool_call(name, args)
        if bus_id:
            self._bus_id_to_tool_id[bus_id] = tool_id

    def _on_bus_tool_result(self, payload: Dict[str, Any]) -> None:
        """Bridge bus ``tool.result`` → ``render_tool_result``.

        Bus producers use ``"ok"`` / ``"error"``; the renderer uses
        ``"success"`` / ``"error"`` for the icon. Normalize ``"ok"`` →
        ``"success"``.
        """
        if not hasattr(self, "_bus_id_to_tool_id"):
            self._bus_id_to_tool_id = {}
        bus_id = (payload or {}).get("id") or ""
        result = (payload or {}).get("result") or ""
        raw_status = (payload or {}).get("status") or "success"
        status = "success" if raw_status == "ok" else raw_status
        tool_id = self._bus_id_to_tool_id.pop(bus_id, None) if bus_id else None

        if (payload or {}).get("cache_hit"):
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"
            result = f"📦 cache: {preview}"

        if tool_id:
            self.render_tool_result(tool_id, result, status=status or "success")
        else:
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "…"
            self.render_system_message(f"↳ {preview}")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _format_tool_args(args: Optional[Dict[str, Any]]) -> str:
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            if isinstance(v, list) and v:
                # A crude repr()-then-truncate on a long list (e.g. `terms`
                # with dozens of book titles) used to cut off mid-way
                # through the FIRST element and hide that there even were
                # more — showing "N: item1, item2, …" is actually
                # informative instead of a near-empty fragment. - Claude Generated
                preview = ", ".join(
                    (str(item)[:30] + "…") if len(str(item)) > 30 else str(item)
                    for item in v[:2]
                )
                more = ", …" if len(v) > 2 else ""
                sv = f"[{len(v)}: {preview}{more}]"
            else:
                sv = repr(v)
                if len(sv) > 40:
                    sv = sv[:40] + "…"
            parts.append(f"{k}={sv}")
        joined = ", ".join(parts)
        if len(joined) > 120:
            joined = joined[:120] + "…"
        return joined
