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
        """Open a streaming line with timestamp + step tag."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_prefix = (
            f"<span style='color: #6272a4;'>[{timestamp}]</span> "
            f"<span style='color: #8be9fd; font-weight: bold;'>[{step_id.upper()}]</span> "
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

                md_html = markdown.markdown(
                    self._current_assistant_text,
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
            # Fallback: plain marker if id unknown.
            preview = (result or "").strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:120] + "…"
            self.render_tool_marker(f"↳ {preview}")
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

    def _tool_call_html(self, tool_id: str) -> str:
        """Generate inline HTML for a single tool-call block.

        Must stay inline (no <div> / <pre>) so the entire tool call lives in
        one QTextBlock and can be replaced in-place via QTextCursor.
        """
        tc = self._tool_calls[tool_id]
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

        Kept for callers that do not need collapsible blocks (e.g. bus
        pipeline-step progress markers).
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
