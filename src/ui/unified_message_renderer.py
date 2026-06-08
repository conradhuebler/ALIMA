"""UnifiedMessageRenderer — single renderer for pipeline log + chat bubbles.

Claude Generated.

Encapsulates every QTextBrowser manipulation that was previously scattered
across PipelineChatPanel.  All rendering methods must be called from the UI
thread (guaranteed today via Qt signals).

Uses MessageEntry for history tracking of *completed* messages only; streaming
tokens are renderer state until the line / bubble is finalised.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from html import escape as html_escape
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QTextBlockFormat,
    QTextCursor,
    QTextLength,
    QTextTableCellFormat,
    QTextTableFormat,
)
from PyQt6.QtWidgets import QCheckBox, QTextBrowser

from .message_entry import MessageEntry, MessageRole


# ----------------------------------------------------------------------
# Module-level style constants
# ----------------------------------------------------------------------

_PIPELINE_COLOR_MAP = {
    "info": "#f8f8f2",
    "success": "#50fa7b",
    "warning": "#f1fa8c",
    "error": "#ff5555",
    "step": "#8be9fd",
    "stream": "#bd93f9",
    "debug": "#6272a4",
}


class UnifiedMessageRenderer:
    """Render messages of all roles into a shared QTextBrowser."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, text_browser: QTextBrowser, auto_scroll_checkbox: QCheckBox):
        self.text_browser = text_browser
        self.auto_scroll_checkbox = auto_scroll_checkbox
        self.logger = logging.getLogger(__name__)

        # History of completed messages (streaming tokens excluded).
        self.history: List[MessageEntry] = []

        # Pipeline streaming state
        self._is_streaming = False

        # Assistant bubble streaming state
        self._assistant_block_open = False
        self._assistant_cell_cursor: Optional[QTextCursor] = None
        self._current_assistant_text = ""

        # Auto-scroll throttle
        self._last_scroll_time = 0.0

        # Tool-call toggle state (id -> expanded bool).
        self._tool_call_id = 0
        self._tool_calls: Dict[str, Dict[str, Any]] = {}
        self._tool_call_blocks: Dict[str, int] = {}  # tool_id -> userState marker

        # Phase E: bus id → renderer tool_id bridge (for subscribe/unsubscribe).
        self._bus_id_to_tool_id: Dict[str, str] = {}

        # P-δ.5: <<CAT:rsn|text>> marker → clickable catalog link. The web
        # base is set by the panel from CatalogConfig.catalog_web_record_url;
        # empty default disables the feature (markers are stripped, leaving
        # only the display text — see _replace_cat_marker in finalize).
        self._catalog_web_base: str = ""

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
        color = _PIPELINE_COLOR_MAP.get(level, "#f8f8f2")

        if step_id:
            formatted = (
                f"<span style='color: #6272a4;'>[{timestamp}]</span> "
                f"<span style='color: {color}; font-weight: bold;'>[{step_id.upper()}]</span> "
                f"<span style='color: {color};'>{self._escape_html(message)}</span>"
            )
        else:
            formatted = (
                f"<span style='color: #6272a4;'>[{timestamp}]</span> "
                f"<span style='color: {color};'>{self._escape_html(message)}</span>"
            )

        self.text_browser.append(formatted)
        self.auto_scroll_to_bottom()

        self.history.append(
            MessageEntry(
                role=MessageRole.PIPELINE_LOG,
                content=message,
                metadata={"level": level, "step_id": step_id},
            )
        )

    def render_streaming_token(self, token: str, step_id: str) -> None:
        """Insert a single purple streaming token inline."""
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        escaped = html_escape(token).replace(" ", "&nbsp;").replace("\n", "<br>")
        cursor.insertHtml(f"<span style='color: #bd93f9;'>{escaped}</span>")
        self.auto_scroll_to_bottom()

    def start_streaming_line(self, step_id: str, prefix: str = "") -> None:
        """Open a streaming line with timestamp + (optional) step tag."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Omit the [TAG] bracket when step_id is empty — orchestration messages
        # without a step id previously rendered an ugly empty "[]". - Claude Generated
        tag = (
            f"<span style='color: #8be9fd; font-weight: bold;'>[{step_id.upper()}]</span> "
            if step_id
            else ""
        )
        formatted_prefix = (
            f"<span style='color: #6272a4;'>[{timestamp}]</span> "
            f"{tag}"
            f"<span style='color: #bd93f9;'>{self._escape_html(prefix)}"
        )
        self.text_browser.append(formatted_prefix)
        self._is_streaming = True

    def end_streaming_line(self) -> None:
        """Close the currently open streaming span."""
        if self._is_streaming:
            cursor = self.text_browser.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml("</span>")
            self._is_streaming = False

    # ------------------------------------------------------------------
    # Chat bubble rendering
    # ------------------------------------------------------------------

    def render_user_bubble(self, text: str) -> None:
        """Right-aligned green WhatsApp-style bubble."""
        self._insert_bubble(
            text,
            align=Qt.AlignmentFlag.AlignRight,
            width_percent=65,
            bg_color="#005c4b",
            fg_color="#e9edef",
        )
        self.history.append(
            MessageEntry(
                role=MessageRole.USER_BUBBLE,
                content=text,
            )
        )

    def open_assistant_bubble(self, model_label: str) -> None:
        """Open a left-aligned grey assistant bubble with model label."""
        self._current_assistant_text = ""
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.text_browser.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        cursor.insertHtml(
            f'<span style="color: #8be9fd; font-size: 9pt; font-style: italic;">'
            f'🤖 {self._escape_html(model_label or "Modell")}'
            f"</span>"
        )
        cursor.insertBlock(QTextBlockFormat())
        table_fmt = QTextTableFormat()
        table_fmt.setCellPadding(8)
        table_fmt.setCellSpacing(0)
        table_fmt.setBorder(0)
        table_fmt.setWidth(QTextLength(QTextLength.Type.PercentageLength, 75))
        table_fmt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        table = cursor.insertTable(1, 1, table_fmt)
        cell = table.cellAt(0, 0)
        cell_fmt = QTextTableCellFormat()
        cell_fmt.setBackground(QColor("#202c33"))
        cell.setFormat(cell_fmt)
        self._assistant_cell_cursor = cell.firstCursorPosition()
        self._assistant_block_open = True
        self.auto_scroll_to_bottom()

    def append_assistant_token(self, token: str) -> None:
        """Stream a token into the open assistant bubble."""
        if self._assistant_cell_cursor is None:
            return
        self._current_assistant_text += token
        html = self._escape_html(token).replace("\n", "<br>").replace(" ", "&nbsp;")
        self._assistant_cell_cursor.insertHtml(
            f'<span style="color: #e9edef; font-size: 10pt;">{html}</span>'
        )
        self.auto_scroll_to_bottom()

    def finalize_assistant_bubble(self) -> None:
        """Post-render Markdown and close the assistant bubble."""
        if (
            self._assistant_cell_cursor is not None
            and self._current_assistant_text
        ):
            try:
                import markdown

                # P-δ.5: replace <<CAT:rsn|display>> markers with clickable
                # anchors BEFORE the markdown pass. Run on the raw text (not
                # HTML-escaped) so the marker regex stays readable. Markdown
                # then leaves the inserted <a> tags alone (they're already
                # valid HTML, and python-markdown passes inline HTML through
                # by default). If catalog_web_base is empty or the RSN is
                # non-numeric, the marker is reduced to the display text so
                # the user sees a clean message instead of a broken link.
                render_text = self._replace_cat_markers(self._current_assistant_text)

                md_html = markdown.markdown(
                    render_text,
                    extensions=["extra", "nl2br"],
                )
                cursor = self._assistant_cell_cursor
                cursor.movePosition(
                    QTextCursor.MoveOperation.Start,
                    QTextCursor.MoveMode.MoveAnchor,
                )
                cursor.movePosition(
                    QTextCursor.MoveOperation.End,
                    QTextCursor.MoveMode.KeepAnchor,
                )
                cursor.removeSelectedText()
                cursor.insertHtml(
                    f'<span style="color: #e9edef; font-size: 10pt;">{md_html}</span>'
                )
            except Exception:
                pass  # Keep raw text if markdown fails

        cursor = self.text_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock(QTextBlockFormat())

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
        self.auto_scroll_to_bottom()

    def _replace_cat_markers(self, text: str) -> str:
        """Replace ``<<CAT:rsn|display>>`` with an HTML anchor.

        Claude Generated (P-δ.5). The regex matches the literal angle-bracket
        marker (no leading/trailing whitespace inside the brackets). When
        ``self._catalog_web_base`` is empty or the RSN is non-numeric the
        marker is reduced to the display text so users never see a broken
        anchor — the worst case is "no link", not "404 link".

        Display text is HTML-escaped to avoid injection of arbitrary HTML
        by the LLM (a prompt-injection mitigation — never trust LLM output
        as raw HTML).
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
                f'<a href="{url}" style="color: #5af; text-decoration: underline;">'
                f"{display}</a>"
            )

        # `|` is the separator; `[^|]+?` is non-greedy on display text.
        # Allow multi-digit RSNs and most display chars; disallow newlines.
        return re.sub(r"<<CAT:([^|\n]+)\|([^|\n]+)>>", _sub, text)

    # ------------------------------------------------------------------
    # Collapsible tool calls
    # ------------------------------------------------------------------

    def render_tool_call(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        duration_s: Optional[float] = None,
    ) -> str:
        """Render a collapsible tool-call block.  Returns the tool_call_id."""
        self._tool_call_id += 1
        tool_id = f"tc_{self._tool_call_id}"
        args_preview = self._format_tool_args(args)
        duration_str = f"  ({duration_s:.1f}s)" if duration_s else ""

        self._tool_calls[tool_id] = {
            "name": name,
            "args": args,
            "args_preview": args_preview,
            "duration_s": duration_s,
            "result": None,
            "expanded": False,
            "status": "running",  # running | success | error
        }

        html = self._tool_call_html(tool_id)
        self._append_html(html)

        # Mark the last block with userState so we can re-render it later.
        last_block = self.text_browser.document().lastBlock()
        if last_block.isValid():
            last_block.setUserState(self._tool_call_id)
            self._tool_call_blocks[tool_id] = self._tool_call_id

        self.history.append(
            MessageEntry(
                role=MessageRole.TOOL_MARKER,
                content=f"🔧 {name}({args_preview})",
                metadata={"tool_name": name, "tool_call_id": tool_id},
            )
        )
        return tool_id

    def render_tool_result(self, tool_id: str, result: str, status: str = "success") -> None:
        """Attach a result to an existing tool call and re-render the block."""
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
        self._rerender_tool_call_block(tool_id)

    def toggle_tool_call(self, tool_id: str) -> bool:
        """Toggle expanded state.  Returns new expanded value."""
        tc = self._tool_calls.get(tool_id)
        if tc is None:
            return False
        tc["expanded"] = not tc["expanded"]
        self._rerender_tool_call_block(tool_id)
        return tc["expanded"]

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

        Reuses the tool-call toggle/re-render machinery (anchor
        ``tool://toggle/<id>``), so ``PipelineChatPanel._handle_tool_link``
        toggles it without changes. ``meta`` is shown next to the header (e.g.
        timestamp / duration) and can be updated later via
        :meth:`update_collapsible_meta`. Returns the block id. - Claude Generated
        """
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
        self._append_html(self._collapsible_html(tool_id))
        last_block = self.text_browser.document().lastBlock()
        if last_block.isValid():
            last_block.setUserState(self._tool_call_id)
            self._tool_call_blocks[tool_id] = self._tool_call_id
        self.history.append(
            MessageEntry(
                role=MessageRole.TOOL_MARKER,
                content=f"{icon} {title}",
                metadata={"kind": "collapsible"},
            )
        )
        return tool_id

    def update_collapsible_meta(self, tool_id: str, meta: str) -> None:
        """Update the header meta (timestamp/duration) of a collapsible block."""
        tc = self._tool_calls.get(tool_id)
        if tc is None or tc.get("kind") != "collapsible":
            return
        tc["meta"] = meta
        self._rerender_tool_call_block(tool_id)

    def _collapsible_html(self, tool_id: str) -> str:
        """Inline HTML for a generic collapsible block (single QTextBlock)."""
        tc = self._tool_calls[tool_id]
        title = self._escape_html(tc.get("name", ""))
        icon = tc.get("icon", "📄")
        meta = tc.get("meta", "")
        result = tc.get("result") or ""
        arrow = "▼" if tc["expanded"] else "▶"
        meta_html = (
            f' <span style="color: #6272a4; font-size: 9pt;">'
            f'{self._escape_html(meta)}</span>'
            if meta
            else ""
        )
        header = (
            f'<a href="tool://toggle/{tool_id}" '
            f'style="color: #8be9fd; text-decoration: none; font-family: monospace; font-size: 9pt;">'
            f'{arrow} {icon} {title}</a>{meta_html}'
        )
        if tc["expanded"] and result:
            escaped = self._escape_html(result)
            return (
                f'{header}<br>'
                f'<span style="font-family: monospace; font-size: 9pt; color: #a8a8a8; '
                f'white-space: pre-wrap; word-wrap: break-word;">{escaped}</span>'
            )
        return header

    def _tool_call_html(self, tool_id: str) -> str:
        """Generate inline HTML for a single tool-call block.

        Must stay inline (no <div> / <pre>) so the entire tool call lives in
        one QTextBlock and can be replaced in-place via QTextCursor.
        """
        tc = self._tool_calls[tool_id]
        if tc.get("kind") == "collapsible":
            return self._collapsible_html(tool_id)
        name = self._escape_html(tc["name"])
        args_preview = self._escape_html(tc["args_preview"])
        duration_str = f"  ({tc['duration_s']:.1f}s)" if tc.get("duration_s") else ""
        status_icon = "⏳" if tc["status"] == "running" else ("✓" if tc["status"] == "success" else "✗")
        arrow = "▼" if tc["expanded"] else "▶"
        result = tc.get("result") or ""

        header = (
            f'<a href="tool://toggle/{tool_id}" '
            f'style="color: #888; text-decoration: none; font-family: monospace; font-size: 9pt;">'
            f'{arrow} 🔧 {name}({args_preview})  {status_icon}{duration_str}</a>'
        )

        if tc["expanded"] and result:
            escaped_result = self._escape_html(result)
            return (
                f'{header}<br>'
                f'<span style="font-family: monospace; font-size: 9pt; color: #a8a8a8; '
                f'white-space: pre-wrap; word-wrap: break-word;">'
                f'{escaped_result}</span>'
            )
        else:
            return header

    def _rerender_tool_call_block(self, tool_id: str) -> None:
        """Re-render a single tool-call block in-place using QTextCursor."""
        marker_id = self._tool_call_blocks.get(tool_id)
        if marker_id is None:
            return

        doc = self.text_browser.document()
        block = doc.begin()
        while block.isValid():
            if block.userState() == marker_id:
                cursor = self.text_browser.textCursor()
                cursor.setPosition(block.position())
                cursor.movePosition(
                    QTextCursor.MoveOperation.EndOfBlock,
                    QTextCursor.MoveMode.KeepAnchor,
                )
                cursor.removeSelectedText()
                cursor.insertHtml(self._tool_call_html(tool_id))
                return
            block = block.next()

        self.logger.warning(f"tool call block {tool_id} not found for re-render")

    # ------------------------------------------------------------------
    # Legacy marker (plain text, no collapse)
    # ------------------------------------------------------------------

    def render_tool_marker(self, text: str, tool_name: Optional[str] = None) -> None:
        """Monospace grey line for tool calls / results / step progress.

        .. deprecated::
            Prefer :meth:`render_tool_call` / :meth:`render_tool_result` for
            new code — they give collapsible blocks, status icons, and
            history entries consistent with the agentic tool-call pathway.
            ``render_tool_marker`` remains only for status echoes that are
            not tool-driven and for the ``PipelineChatPanel`` shim.
        """
        html = (
            f'<div style="margin: 2px 0 2px 8px; '
            f'font-family: monospace; font-size: 9pt; color: #888;">'
            f"{self._escape_html(text)}</div>"
        )
        self._append_html(html)
        self.history.append(
            MessageEntry(
                role=MessageRole.TOOL_MARKER,
                content=text,
                metadata={"tool_name": tool_name},
            )
        )

    def render_system_message(self, text: str) -> None:
        """Centered italic green status line."""
        html = (
            f'<div style="text-align: center; margin: 4px 0;">'
            f'<span style="color: #4caf50; font-size: 9pt; font-style: italic;">'
            f"{self._escape_html(text)}</span></div>"
        )
        self._append_html(html)
        self.history.append(
            MessageEntry(
                role=MessageRole.SYSTEM_MESSAGE,
                content=text,
            )
        )

    def render_html_block(
        self, html: str, *, kind: Optional[str] = None, plain_text: str = ""
    ) -> None:
        """Append a pre-formatted, trusted HTML block (e.g. a DK/RVK result card).

        Unlike the inline tool-result block (:meth:`_tool_call_html`, which is
        HTML-escaped and re-rendered in place), this block is appended once and
        never re-rendered, so block-level HTML (``<div>``, ``<ol>``, ``<h2>``) is
        allowed. The caller must pass already-sanitised HTML — the shared
        ``PipelineResultFormatter`` escapes catalog titles.

        ``plain_text`` is recorded in history for export; ``kind`` tags the card
        (e.g. ``"dk_classifications"``, ``"dk_search"``).
        """
        if not html:
            return
        self._append_html(html)
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
        """Clickable mutation-proposal block with accept / reject anchors."""
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
            f'Begründung: {reason}</div>'
            f'<div style="margin-top: 8px;">'
            f'<a href="{accept_href}" style="color: #50fa7b; '
            f'text-decoration: none; padding: 4px 10px; '
            f'border: 1px solid #50fa7b; border-radius: 3px; '
            f'margin-right: 8px;">✓ Akzeptieren</a>'
            f'<a href="{reject_href}" style="color: #ff5555; '
            f'text-decoration: none; padding: 4px 10px; '
            f'border: 1px solid #ff5555; border-radius: 3px;">✗ Ablehnen</a>'
            f'<span style="color: #555; font-size: 8pt; margin-left: 8px;">'
            f'#audit_{audit_id}</span>'
            f"</div></div>"
        )
        self._append_html(html)
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

    def _insert_bubble(
        self,
        text: str,
        *,
        align: Qt.AlignmentFlag,
        width_percent: int,
        bg_color: str,
        fg_color: str,
    ) -> None:
        """Insert a QTextTable bubble with the given alignment and colours."""
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.text_browser.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        table_fmt = QTextTableFormat()
        table_fmt.setCellPadding(8)
        table_fmt.setCellSpacing(0)
        table_fmt.setBorder(0)
        table_fmt.setWidth(
            QTextLength(QTextLength.Type.PercentageLength, width_percent)
        )
        table_fmt.setAlignment(align)
        table = cursor.insertTable(1, 1, table_fmt)
        cell = table.cellAt(0, 0)
        cell_fmt = QTextTableCellFormat()
        cell_fmt.setBackground(QColor(bg_color))
        cell.setFormat(cell_fmt)
        body = self._escape_html(text).replace("\n", "<br>")
        cell.firstCursorPosition().insertHtml(
            f'<span style="color: {fg_color}; font-size: 10pt;">{body}</span>'
        )
        end_cursor = self.text_browser.textCursor()
        end_cursor.movePosition(QTextCursor.MoveOperation.End)
        end_cursor.insertBlock(QTextBlockFormat())
        self.auto_scroll_to_bottom()

    def _append_html(self, html: str) -> None:
        """Insert raw HTML at the end of the document."""
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.text_browser.document().isEmpty():
            cursor.insertBlock(QTextBlockFormat())
        cursor.insertHtml(html)
        self.auto_scroll_to_bottom()

    def append_raw_html(self, html: str) -> None:
        """Public escape-hatch for external callers."""
        self._append_html(html)

    # ------------------------------------------------------------------
    # Auto-scroll
    # ------------------------------------------------------------------

    def auto_scroll_to_bottom(self) -> None:
        """Throttled scroll-to-bottom (max 20 Hz)."""
        if not self.auto_scroll_checkbox.isChecked():
            return
        now = time.time()
        if now - self._last_scroll_time < 0.05:
            return
        self._last_scroll_time = now
        scrollbar = self.text_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------------
    # Clear / reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the document and reset all open streaming / bubble state."""
        self.text_browser.clear()
        self._is_streaming = False
        self._assistant_block_open = False
        self._assistant_cell_cursor = None
        self._current_assistant_text = ""
        self._tool_call_id = 0
        self._tool_calls.clear()
        self._tool_call_blocks.clear()
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

        Phase E: lets a renderer instance (e.g. a mini-log in
        AnalysisReviewTab / ImageAnalysisTab) participate in the same
        event flow as PipelineChatPanel without manual Qt-signal wiring.
        Each renderer keeps its own ``tool_id`` counter and id mapping
        — renderers do not share state.

        Bound methods are stored on ``self`` so ``unsubscribe`` can find
        the *same* object — bare ``self._on_bus_tool_called`` access on
        a second call yields a *new* bound-method object that the bus
        cannot match (it compares by identity). Storing once is the
        only way to ensure subscribe/unsubscribe pair up.
        """
        try:
            from src.core.state_bus import AlimaStateBus
            bus = AlimaStateBus()
            # Bind once, reuse the same object on unsubscribe.
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
            # Must use the same object stored in ``subscribe`` —
            # re-binding ``self._on_bus_tool_called`` here would yield
            # a different object identity.
            tool_called = getattr(self, "_bus_handler_tool_called", None)
            tool_result = getattr(self, "_bus_handler_tool_result", None)
            if tool_called is not None:
                bus.unsubscribe("tool.called", tool_called)
            if tool_result is not None:
                bus.unsubscribe("tool.result", tool_result)
        except Exception:
            self.logger.exception("UnifiedMessageRenderer.unsubscribe failed")
        finally:
            # Drop any pending open-call mapping so stale ids don't leak.
            self._bus_id_to_tool_id.clear()

    def _on_bus_tool_called(self, payload: Dict[str, Any]) -> None:
        """Bridge bus ``tool.called`` → ``render_tool_call``.

        Maps the bus ``id`` (UUID-string from any producer) to a
        renderer-local ``tool_id`` so the matching ``tool.result`` can
        attach to the same block.
        """
        bus_id = (payload or {}).get("id") or ""
        name = (payload or {}).get("name") or ""
        args = (payload or {}).get("arguments") or {}
        tool_id = self.render_tool_call(name, args)
        if bus_id:
            self._bus_id_to_tool_id[bus_id] = tool_id

    def _on_bus_tool_result(self, payload: Dict[str, Any]) -> None:
        """Bridge bus ``tool.result`` → ``render_tool_result``.

        Mirrors the panel's bus handler: cache hits get the 📦 badge,
        and the status field is propagated. Falls back to a plain
        marker when no matching open call exists.

        Bus producers use ``"ok"`` / ``"error"`` (CachingToolRegistry,
        classic shim). The renderer internally uses ``"success"`` /
        ``"error"`` for the icon. Normalize ``"ok"`` → ``"success"``.
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
            sv = repr(v)
            if len(sv) > 40:
                sv = sv[:40] + "…"
            parts.append(f"{k}={sv}")
        joined = ", ".join(parts)
        if len(joined) > 80:
            joined = joined[:80] + "…"
        return joined
