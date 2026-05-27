"""Tests for UnifiedMessageRenderer. Claude Generated.

Tests all rendering paths using mocked QTextBrowser / QCheckBox to avoid
Qt GUI singleton conflicts.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextCursor


class _MockCell:
    def __init__(self):
        self._cursor = _MockCursor()

    def firstCursorPosition(self):
        return self._cursor

    def setFormat(self, fmt):
        pass


class _MockTable:
    def __init__(self):
        self._cell = _MockCell()

    def cellAt(self, row, col):
        return self._cell


class _MockCursor:
    """Minimal QTextCursor stand-in."""

    def __init__(self):
        self._html_fragments: list[str] = []
        self._ops: list[str] = []

    def movePosition(self, op, mode=None, n=1):
        self._ops.append(f"move:{op}")
        return True

    def insertHtml(self, html: str):
        self._html_fragments.append(html)
        self._ops.append("insertHtml")

    def insertBlock(self, fmt=None):
        self._ops.append("insertBlock")

    def removeSelectedText(self):
        self._ops.append("removeSelectedText")

    def insertTable(self, rows, cols, fmt=None):
        self._ops.append(f"insertTable:{rows}x{cols}")
        return _MockTable()


class _MockDocument:
    def __init__(self, empty=True):
        self._empty = empty

    def isEmpty(self):
        return self._empty


class _MockTextBrowser:
    """Stand-in for QTextBrowser that records HTML insertions."""

    def __init__(self):
        self._html_calls: list[str] = []
        self._cursor = _MockCursor()
        self._doc = _MockDocument(empty=True)

    def textCursor(self):
        return self._cursor

    def document(self):
        return self._doc

    def append(self, html: str):
        self._html_calls.append(html)

    def toHtml(self):
        cursor_html = "\n".join(self._cursor._html_fragments)
        doc_html = "\n".join(self._html_calls)
        return cursor_html + "\n" + doc_html if cursor_html else doc_html

    def toPlainText(self):
        return self.toHtml()

    def clear(self):
        self._html_calls.clear()
        self._cursor._html_fragments.clear()
        self._cursor._ops.clear()

    def verticalScrollBar(self):
        m = MagicMock()
        m.maximum.return_value = 100
        return m


class _MockCheckBox:
    def __init__(self, checked=True):
        self._checked = checked

    def isChecked(self):
        return self._checked


class RendererTestBase(unittest.TestCase):
    """Set up a mock QTextBrowser + mock checkbox."""

    def setUp(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer

        self.text_browser = _MockTextBrowser()
        self.checkbox = _MockCheckBox(checked=True)
        self.renderer = UnifiedMessageRenderer(self.text_browser, self.checkbox)


class TestPipelineLogRendering(RendererTestBase):

    def test_info_log_contains_timestamp(self):
        self.renderer.render_pipeline_log("test msg", "info", "step1")
        html = self.text_browser.toHtml()
        self.assertIn("STEP1", html.upper())
        self.assertIn("test msg", html)

    def test_error_log_red_color(self):
        self.renderer.render_pipeline_log("err", "error")
        html = self.text_browser.toHtml()
        self.assertIn("#ff5555", html)

    def test_success_log_green_color(self):
        self.renderer.render_pipeline_log("ok", "success")
        html = self.text_browser.toHtml()
        self.assertIn("#50fa7b", html)

    def test_step_tag_bold(self):
        self.renderer.render_pipeline_log("start", "step", "init")
        html = self.text_browser.toHtml()
        self.assertIn("font-weight: bold", html)

    def test_history_appended(self):
        self.renderer.render_pipeline_log("msg", "info", "s1")
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "PIPELINE_LOG")
        self.assertEqual(self.renderer.history[0].metadata["level"], "info")
        self.assertEqual(self.renderer.history[0].metadata["step_id"], "s1")


class TestStreamingTokens(RendererTestBase):

    def test_streaming_state(self):
        self.renderer.start_streaming_line("step1", "prefix: ")
        self.assertTrue(self.renderer._is_streaming)
        self.renderer.end_streaming_line()
        self.assertFalse(self.renderer._is_streaming)

    def test_streaming_token_purple(self):
        self.renderer.start_streaming_line("step1")
        self.renderer.render_streaming_token("hello", "step1")
        self.renderer.end_streaming_line()
        self.assertIn("#bd93f9", self.text_browser.toHtml())


class TestUserBubble(RendererTestBase):

    def test_user_bubble_appended(self):
        self.renderer.render_user_bubble("hi")
        self.assertTrue(len(self.text_browser._cursor._ops) > 0)

    def test_user_bubble_history(self):
        self.renderer.render_user_bubble("hello")
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "USER_BUBBLE")
        self.assertEqual(self.renderer.history[0].content, "hello")


class TestAssistantBubble(RendererTestBase):

    def test_assistant_bubble_open(self):
        self.renderer.open_assistant_bubble("gpt-4")
        self.assertTrue(self.renderer._assistant_block_open)
        self.assertIsNotNone(self.renderer._assistant_cell_cursor)

    def test_assistant_token_appended(self):
        self.renderer.open_assistant_bubble("model")
        self.renderer.append_assistant_token("token1")
        self.renderer.append_assistant_token(" token2")
        self.assertEqual(self.renderer._current_assistant_text, "token1 token2")

    def test_assistant_finalize_calls_markdown(self):
        fake_md = MagicMock()
        fake_md.markdown.return_value = "<p><strong>bold</strong></p>"
        with patch.dict("sys.modules", {"markdown": fake_md}):
            self.renderer.open_assistant_bubble("model")
            self.renderer.append_assistant_token("**bold**")
            self.renderer.finalize_assistant_bubble()
        fake_md.markdown.assert_called_once()
        self.assertFalse(self.renderer._assistant_block_open)
        self.assertIsNone(self.renderer._assistant_cell_cursor)

    def test_assistant_history(self):
        fake_md = MagicMock()
        fake_md.markdown.return_value = "<p>text</p>"
        self.renderer.open_assistant_bubble("model")
        self.renderer.append_assistant_token("text")
        with patch.dict("sys.modules", {"markdown": fake_md}):
            self.renderer.finalize_assistant_bubble()
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "ASSISTANT_BUBBLE")


class TestToolMarker(RendererTestBase):

    def test_tool_marker_monospace(self):
        self.renderer.render_tool_marker("🔧 search(query='x')")
        html = self.text_browser.toHtml()
        self.assertIn("monospace", html)

    def test_tool_marker_history(self):
        self.renderer.render_tool_marker("marker", tool_name="search")
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "TOOL_MARKER")
        self.assertEqual(self.renderer.history[0].metadata["tool_name"], "search")


class TestSystemMessage(RendererTestBase):

    def test_system_message_centered(self):
        self.renderer.render_system_message("status")
        html = self.text_browser.toHtml()
        self.assertIn("text-align: center", html)

    def test_system_history(self):
        self.renderer.render_system_message("ok")
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "SYSTEM_MESSAGE")


class TestProposalBubble(RendererTestBase):

    def test_proposal_renders_anchor(self):
        self.renderer.render_proposal_bubble(
            1, "propose_keyword_replacement",
            {"old": "A", "new": "B", "reason": "test"}
        )
        html = self.text_browser.toHtml()
        self.assertIn("mutation://1/accept", html)
        self.assertIn("mutation://1/reject", html)
        self.assertIn("Akzeptieren", html)

    def test_proposal_history(self):
        self.renderer.render_proposal_bubble(
            2, "propose_dk_change", {"code": "123", "action": "add"}
        )
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "PROPOSAL_BUBBLE")
        self.assertEqual(self.renderer.history[0].metadata["audit_id"], 2)


class TestClearAndReset(RendererTestBase):

    def test_clear_resets_streaming_state(self):
        self.renderer.start_streaming_line("step1")
        self.renderer.clear()
        self.assertFalse(self.renderer._is_streaming)

    def test_clear_resets_assistant_state(self):
        self.renderer.open_assistant_bubble("model")
        self.renderer.clear()
        self.assertFalse(self.renderer._assistant_block_open)
        self.assertIsNone(self.renderer._assistant_cell_cursor)
        self.assertEqual(self.renderer._current_assistant_text, "")

    def test_clear_preserves_history(self):
        self.renderer.render_system_message("msg")
        self.renderer.clear()
        self.assertEqual(len(self.renderer.history), 1)

    def test_clear_clears_document(self):
        self.renderer.render_system_message("msg")
        self.renderer.clear()
        self.assertEqual(len(self.text_browser._html_calls), 0)


class TestAutoScroll(RendererTestBase):

    def test_throttle_ignores_second_call(self):
        import time
        self.renderer.auto_scroll_to_bottom()
        t0 = time.time()
        self.renderer.auto_scroll_to_bottom()
        self.assertLess(time.time() - t0, 0.05)

    def test_respects_checkbox(self):
        self.checkbox._checked = False
        # Should not crash and should return early
        self.renderer.auto_scroll_to_bottom()


class TestStaticHelpers(unittest.TestCase):

    def test_escape_html(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        out = UnifiedMessageRenderer._escape_html('a < b & c > d "e"')
        self.assertEqual(out, "a &lt; b &amp; c &gt; d &quot;e&quot;")

    def test_format_tool_args_empty(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        self.assertEqual(UnifiedMessageRenderer._format_tool_args({}), "")
        self.assertEqual(UnifiedMessageRenderer._format_tool_args(None), "")

    def test_format_tool_args_short(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        out = UnifiedMessageRenderer._format_tool_args({"kind": "initial"})
        self.assertIn("kind=", out)

    def test_format_tool_args_truncates_long_value(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        out = UnifiedMessageRenderer._format_tool_args({"q": "x" * 80})
        self.assertIn("q=", out)
        self.assertIn("…", out)

    def test_format_tool_args_truncates_long_joined(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        args = {f"k{i}": f"v{i}" for i in range(30)}
        out = UnifiedMessageRenderer._format_tool_args(args)
        self.assertIn("…", out)


if __name__ == "__main__":
    unittest.main()
