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


class _MockBlock:
    def __init__(self, user_state=-1, valid=True, position=0):
        self._user_state = user_state
        self._valid = valid
        self._position = position
        self._next = None
        self._html = ""

    def isValid(self):
        return self._valid

    def userState(self):
        return self._user_state

    def setUserState(self, state):
        self._user_state = state

    def next(self):
        return self._next if self._next else _MockBlock(valid=False)

    def position(self):
        return self._position


class _MockDocument:
    def __init__(self, empty=True):
        self._blocks: list[_MockBlock] = []
        self._position_counter = 0
        # Qt documents always have at least one empty block
        self.add_block()

    def isEmpty(self):
        return all(not b._html for b in self._blocks)

    def begin(self):
        return self._blocks[0] if self._blocks else _MockBlock(valid=False)

    def lastBlock(self):
        return self._blocks[-1] if self._blocks else _MockBlock(valid=False)

    def add_block(self):
        block = _MockBlock(position=self._position_counter)
        self._position_counter += 100
        if self._blocks:
            self._blocks[-1]._next = block
        self._blocks.append(block)
        return block


class _MockCursor:
    """Minimal QTextCursor stand-in with block-aware insertHtml."""

    def __init__(self, text_browser=None):
        self._html_fragments: list[str] = []
        self._ops: list[str] = []
        self._block = None
        self._text_browser = text_browser

    def block(self):
        return self._block if self._block else _MockBlock(valid=False)

    def movePosition(self, op, mode=None, n=1):
        self._ops.append(f"move:{op}")
        return True

    def setPosition(self, pos, mode=None):
        self._ops.append(f"setPosition:{pos}")
        if self._text_browser:
            for b in self._text_browser._doc._blocks:
                if b._position == pos:
                    self._block = b
                    break

    def insertHtml(self, html: str):
        self._ops.append("insertHtml")
        if self._block and self._text_browser:
            self._block._html += html
        elif self._text_browser and self._text_browser._doc._blocks:
            # No current block but document has blocks - use last block
            self._block = self._text_browser._doc._blocks[-1]
            self._block._html += html
        else:
            self._html_fragments.append(html)

    def insertBlock(self, fmt=None):
        self._ops.append("insertBlock")
        if self._text_browser:
            self._block = self._text_browser._doc.add_block()

    def removeSelectedText(self):
        self._ops.append("removeSelectedText")
        if self._block:
            self._block._html = ""

    def insertTable(self, rows, cols, fmt=None):
        self._ops.append(f"insertTable:{rows}x{cols}")
        return _MockTable()


class _MockTextBrowser:
    """Stand-in for QTextBrowser that records HTML insertions."""

    def __init__(self):
        self._html_calls: list[str] = []
        self._doc = _MockDocument(empty=True)
        self._cursor = _MockCursor(text_browser=self)

    def textCursor(self):
        return self._cursor

    def document(self):
        return self._doc

    def append(self, html: str):
        self._html_calls.append(html)

    def toHtml(self):
        block_html = "\n".join(b._html for b in self._doc._blocks)
        cursor_html = "\n".join(self._cursor._html_fragments)
        doc_html = "\n".join(self._html_calls)
        parts = [p for p in [block_html, cursor_html, doc_html] if p]
        return "\n".join(parts)

    def toPlainText(self):
        return self.toHtml()

    def clear(self):
        self._html_calls.clear()
        self._cursor._html_fragments.clear()
        self._cursor._ops.clear()
        self._doc._blocks.clear()
        self._cursor._block = None

    def setHtml(self, html: str):
        # For re-render tests: parse the HTML back into our mock state.
        self._html_calls = [html]
        self._cursor._html_fragments.clear()
        self._cursor._ops.clear()
        self._doc._blocks.clear()
        self._cursor._block = None

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


class TestCollapsibleToolCall(RendererTestBase):

    def test_tool_call_renders_collapsed(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        html = self.text_browser.toHtml()
        self.assertIn("▶", html)
        self.assertIn("search", html)
        self.assertIn(f"tool://toggle/{tid}", html)

    def test_tool_call_returns_id(self):
        tid = self.renderer.render_tool_call("search", {})
        self.assertTrue(tid.startswith("tc_"))

    def test_tool_result_attaches(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        self.renderer.render_tool_result(tid, '{"hits": 5}')
        html = self.text_browser.toHtml()
        # After result, status should be success (✓)
        self.assertIn("✓", html)

    def test_tool_toggle_expands(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        self.renderer.render_tool_result(tid, '{"hits": 5}')
        expanded = self.renderer.toggle_tool_call(tid)
        self.assertTrue(expanded)
        html = self.text_browser.toHtml()
        self.assertIn("▼", html)
        self.assertIn("hits", html)

    def test_tool_toggle_collapses(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        self.renderer.render_tool_result(tid, '{"hits": 5}')
        self.renderer.toggle_tool_call(tid)  # expand
        collapsed = self.renderer.toggle_tool_call(tid)  # collapse again
        self.assertFalse(collapsed)
        html = self.text_browser.toHtml()
        self.assertIn("▶", html)

    def test_tool_call_history(self):
        self.renderer.render_tool_call("search", {"q": "x"})
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "TOOL_MARKER")
        self.assertEqual(self.renderer.history[0].metadata["tool_name"], "search")

    def test_unknown_tool_id_fallback(self):
        self.renderer.render_tool_result("nonexistent", "result")
        html = self.text_browser.toHtml()
        self.assertIn("↳", html)


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


class TestBusSubscription(unittest.TestCase):
    """Phase E: ``subscribe()`` / ``unsubscribe()`` + bus event bridge."""

    def setUp(self):
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.instance() or QCoreApplication([])

        from src.core import state_bus as state_bus_mod
        from src.ui.unified_message_renderer import UnifiedMessageRenderer

        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()

        self.text_browser = _MockTextBrowser()
        self.checkbox = _MockCheckBox(checked=True)
        self.renderer = UnifiedMessageRenderer(self.text_browser, self.checkbox)

    def tearDown(self):
        from src.core import state_bus as state_bus_mod
        state_bus_mod.reset()

    def test_subscribe_unsubscribe_round_trip(self):
        """``subscribe()`` registers handlers; ``unsubscribe()`` removes them."""
        self.renderer.subscribe()
        # Both event types are now subscribed on the bus.
        called_subs = [s for s in self.bus._subscriptions if s[0] == "tool.called"]
        result_subs = [s for s in self.bus._subscriptions if s[0] == "tool.result"]
        self.assertGreater(len(called_subs), 0)
        self.assertGreater(len(result_subs), 0)

        self.renderer.unsubscribe()
        # After unsubscribe, our specific handlers are gone.
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        for event, handler, _slot in self.bus._subscriptions:
            if event != "tool.called":
                continue
            qualname = getattr(handler, "__qualname__", "")
            self.assertFalse(
                qualname.startswith("UnifiedMessageRenderer._on_bus_tool_called"),
                f"renderer should have unsubscribed, found {qualname}",
            )

    def test_bus_events_forward_to_renderer(self):
        """``tool.called`` opens a tool block; ``tool.result`` attaches."""
        self.renderer.subscribe()
        try:
            self.bus.emit_event("tool.called", {
                "name": "search_gnd",
                "arguments": {"term": "Bibliothek"},
                "id": "tc_bus1",
            })
            # A tool call block was rendered.
            self.assertEqual(len(self.renderer.history), 1)
            self.assertEqual(self.renderer.history[0].metadata["tool_name"], "search_gnd")
            # The id was mapped.
            self.assertIn("tc_bus1", self.renderer._bus_id_to_tool_id)

            self.bus.emit_event("tool.result", {
                "name": "search_gnd",
                "result": '{"hits": 3}',
                "id": "tc_bus1",
                "status": "ok",
            })
            # Tool block updated: status is "success" (✓ rendered).
            html = self.text_browser.toHtml()
            self.assertIn("✓", html)
            # Mapping consumed.
            self.assertNotIn("tc_bus1", self.renderer._bus_id_to_tool_id)
        finally:
            self.renderer.unsubscribe()

    def test_cache_hit_badge_propagates(self):
        """A ``cache_hit=True`` result rewrites the result string with 📦."""
        self.renderer.subscribe()
        try:
            self.bus.emit_event("tool.called", {
                "name": "search_gnd",
                "arguments": {"term": "x"},
                "id": "tc_cache",
            })
            self.bus.emit_event("tool.result", {
                "name": "search_gnd",
                "result": '{"hits": 0}',
                "id": "tc_cache",
                "cache_hit": True,
                "status": "ok",
            })
            # Inspect the tool call's stored result — it should carry 📦.
            tool_id = "tc_1"  # renderer's local id; first call → tc_1
            tc = self.renderer._tool_calls[tool_id]
            self.assertIn("📦 cache", tc["result"])
        finally:
            self.renderer.unsubscribe()

    def test_orphan_result_renders_marker(self):
        """A ``tool.result`` without a matching ``tool.called`` falls back
        to a plain marker (no exception, no crash)."""
        self.renderer.subscribe()
        try:
            self.bus.emit_event("tool.result", {
                "name": "lone",
                "result": "lone result",
                "id": "no-match",
                "status": "ok",
            })
            html = self.text_browser.toHtml()
            self.assertIn("↳", html)
        finally:
            self.renderer.unsubscribe()


class TestCatalogMarkerReplacement(RendererTestBase):
    """Tests for P-δ.5: <<CAT:rsn|display>> → clickable anchor."""

    WEB_BASE = "https://katalog.ub.tu-freiberg.de/Record/"

    def test_marker_replaced_with_anchor(self):
        """Valid marker is converted to an <a href=…0-{rsn}> with display text."""
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers("Vor <<CAT:12345|Titel>> nach")
        self.assertIn('href="https://katalog.ub.tu-freiberg.de/Record/0-12345"', out)
        self.assertIn(">Titel</a>", out)
        # Marker syntax itself must be gone.
        self.assertNotIn("<<CAT:", out)
        # Surrounding text preserved.
        self.assertTrue(out.startswith("Vor "))
        self.assertTrue(out.endswith(" nach"))

    def test_invalid_rsn_passes_through_as_text(self):
        """Non-numeric RSN → marker is reduced to display text (no anchor)."""
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers("Siehe <<CAT:abc|Mein Titel>>")
        self.assertNotIn("<a ", out, "No anchor should be emitted for invalid RSN")
        self.assertNotIn("<<CAT:", out)
        self.assertIn("Mein Titel", out)

    def test_empty_web_base_disables_feature(self):
        """Without a configured base, markers reduce to display text."""
        # Do NOT call set_catalog_web_base — default is "".
        out = self.renderer._replace_cat_markers("X <<CAT:12345|Titel>> Y")
        self.assertNotIn("<a ", out, "Feature must be off when no base URL set")
        self.assertNotIn("<<CAT:", out)
        self.assertIn("Titel", out)

    def test_multiple_markers_all_replaced(self):
        """Multiple markers in one text are all replaced independently."""
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers(
            "<<CAT:111|A>> und <<CAT:222|B>>"
        )
        self.assertIn("Record/0-111", out)
        self.assertIn("Record/0-222", out)
        self.assertIn(">A</a>", out)
        self.assertIn(">B</a>", out)

    def test_html_in_display_is_escaped(self):
        """Display text is HTML-escaped to prevent LLM-injected XSS."""
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers('<<CAT:1|<script>x</script>>>')
        # The raw <script> tag must be escaped, not embedded.
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_no_marker_passthrough(self):
        """Text without markers is returned unchanged."""
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        text = "Just plain text, no marker here."
        self.assertEqual(self.renderer._replace_cat_markers(text), text)


if __name__ == "__main__":
    unittest.main()
