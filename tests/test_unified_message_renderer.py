"""Tests for UnifiedMessageRenderer. Claude Generated.

The renderer now drives a :class:`WebLogView` (QWebEngineView) instead of a
QTextBrowser. These tests use a lightweight ``_MockWebLogView`` that records the
HTML strings the renderer emits, so assertions target the emitted markup
(colours, text, history) without needing a browser engine. Collapsibles are
native ``<details>`` — the body is always present in the DOM; the ``open`` flag
controls initial expansion, and toggling is a server-side mirror only.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class _MockWebLogView:
    """Records every HTML fragment the renderer pushes into the page."""

    def __init__(self):
        self.blocks: list[str] = []
        self.collapsibles: dict[str, dict] = {}
        self.assistant_header: str | None = None
        self.assistant_tokens: list[str] = []
        self.assistant_final: str | None = None
        self.stream_blocks: dict[str, dict] = {}
        self._cur_stream: str | None = None
        self.autoscroll = True

    # -- append / stream API mirrored from WebLogView --------------------
    def append_block(self, html: str) -> None:
        self.blocks.append(html)

    def append_collapsible(self, block_id, summary, body, open_, kind=None) -> None:
        self.collapsibles[block_id] = {
            "summary": summary, "body": body or "", "open": bool(open_),
            "kind": kind,
        }

    def update_collapsible(self, block_id, summary, body, kind=None) -> None:
        prev = self.collapsibles.get(block_id, {})
        self.collapsibles[block_id] = {
            "summary": summary, "body": body or "", "open": prev.get("open", False),
            "kind": kind or prev.get("kind"),
        }

    def open_assistant(self, header: str) -> None:
        self.assistant_header = header
        self.assistant_tokens = []
        self.assistant_final = None

    def append_token(self, text: str) -> None:
        self.assistant_tokens.append(text)

    def finalize_assistant(self, html: str) -> None:
        self.assistant_final = html

    def open_stream_block(self, block_id, summary) -> None:
        self.stream_blocks[block_id] = {
            "summary": summary, "tokens": [], "collapsed": False
        }
        self._cur_stream = block_id

    def append_stream_block(self, text) -> None:
        if self._cur_stream is not None:
            self.stream_blocks[self._cur_stream]["tokens"].append(text)

    def close_stream_block(self, block_id, summary, collapse=True) -> None:
        b = self.stream_blocks.setdefault(
            block_id, {"summary": summary, "tokens": [], "collapsed": False}
        )
        b["summary"] = summary
        b["collapsed"] = bool(collapse)
        self._cur_stream = None

    def clear_log(self) -> None:
        self.blocks.clear()
        self.collapsibles.clear()
        self.assistant_header = None
        self.assistant_tokens = []
        self.assistant_final = None
        self.stream_blocks.clear()
        self._cur_stream = None

    def set_autoscroll(self, enabled: bool) -> None:
        self.autoscroll = enabled

    def scroll_to_bottom(self) -> None:
        pass

    def set_font_pt(self, pt: int) -> None:
        pass

    # -- test helper -----------------------------------------------------
    def to_html(self) -> str:
        parts: list[str] = list(self.blocks)
        for tc in self.collapsibles.values():
            parts.append(tc["summary"])
            parts.append(tc["body"])  # native <details>: body always in DOM
        if self.assistant_header:
            parts.append(self.assistant_header)
        parts.extend(self.assistant_tokens)
        if self.assistant_final:
            parts.append(self.assistant_final)
        for blk in self.stream_blocks.values():
            parts.append(blk["summary"])
            parts.extend(blk["tokens"])
        return "\n".join(p for p in parts if p)


class _MockCheckBox:
    def __init__(self, checked=True):
        self._checked = checked
        self.toggled = MagicMock()  # .connect() used by the renderer ctor

    def isChecked(self):
        return self._checked


class RendererTestBase(unittest.TestCase):
    """Set up a mock WebLogView + mock checkbox."""

    def setUp(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer

        self.view = _MockWebLogView()
        self.checkbox = _MockCheckBox(checked=True)
        self.renderer = UnifiedMessageRenderer(self.view, self.checkbox)


class TestPipelineLogRendering(RendererTestBase):

    def test_info_log_contains_timestamp(self):
        self.renderer.render_pipeline_log("test msg", "info", "step1")
        html = self.view.to_html()
        self.assertIn("STEP1", html.upper())
        self.assertIn("test msg", html)

    def test_error_log_level_class(self):
        # Colors live in alima_render.css (--alima-*); the renderer emits
        # semantic classes (Chat-UX 6/9).
        self.renderer.render_pipeline_log("err", "error")
        self.assertIn("log-lvl--error", self.view.to_html())

    def test_success_log_level_class(self):
        self.renderer.render_pipeline_log("ok", "success")
        self.assertIn("log-lvl--success", self.view.to_html())

    def test_unknown_level_falls_back_to_info(self):
        self.renderer.render_pipeline_log("odd", "nonsense")
        self.assertIn("log-lvl--info", self.view.to_html())

    def test_step_tag_class(self):
        self.renderer.render_pipeline_log("start", "step", "init")
        self.assertIn("log-step", self.view.to_html())

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

    def test_streaming_block_opens_expanded(self):
        """While streaming the block is open (expanded); tokens go to its body."""
        self.renderer.start_streaming_line("step1", "LLM: ")
        self.renderer.render_streaming_token("hello keywords", "step1")
        blk = next(iter(self.view.stream_blocks.values()))
        self.assertFalse(blk["collapsed"])                  # open during stream
        self.assertIn("hello keywords", "".join(blk["tokens"]))
        self.assertIn("sl-title", self.view.to_html())      # styled title class

    def test_streaming_block_collapses_with_preview(self):
        """On end the block collapses and the summary keeps a text preview."""
        self.renderer.start_streaming_line("step1", "LLM: ")
        self.renderer.render_streaming_token("erste keywords hier", "step1")
        self.renderer.end_streaming_line()
        blk = next(iter(self.view.stream_blocks.values()))
        self.assertTrue(blk["collapsed"])                   # folded away
        self.assertIn("erste keywords hier", blk["summary"])  # preview retained

    def test_token_ignored_without_open_block(self):
        """A stray token with no open stream block is a no-op (no crash)."""
        self.renderer.render_streaming_token("orphan", "step1")
        self.assertEqual(self.view.stream_blocks, {})


class TestUserBubble(RendererTestBase):

    def test_user_bubble_appended(self):
        self.renderer.render_user_bubble("hi")
        self.assertEqual(len(self.view.blocks), 1)
        self.assertIn("hi", self.view.to_html())

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
        self.assertIn("gpt-4", self.view.assistant_header)

    def test_assistant_token_appended(self):
        self.renderer.open_assistant_bubble("model")
        self.renderer.append_assistant_token("token1")
        self.renderer.append_assistant_token(" token2")
        self.assertEqual(self.renderer._current_assistant_text, "token1 token2")
        self.assertEqual(self.view.assistant_tokens, ["token1", " token2"])

    def test_assistant_finalize_calls_markdown(self):
        self.renderer.open_assistant_bubble("model")
        self.renderer.append_assistant_token("**bold**")
        self.renderer.finalize_assistant_bubble()
        # markdown_it renders **bold** → <strong>bold</strong>
        self.assertIn("bold", self.view.assistant_final or "")
        self.assertIn("<strong>", self.view.assistant_final or "")
        self.assertFalse(self.renderer._assistant_block_open)
        self.assertIsNone(self.renderer._assistant_cell_cursor)

    def test_assistant_history(self):
        self.renderer.open_assistant_bubble("model")
        self.renderer.append_assistant_token("text")
        self.renderer.finalize_assistant_bubble()
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "ASSISTANT_BUBBLE")


class TestCollapsibleToolCall(RendererTestBase):
    """Native <details>: no toggle anchor, no arrow glyph; the body is always
    in the DOM and the ``open`` flag controls initial expansion."""

    def test_tool_call_renders_collapsed(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        tc = self.view.collapsibles[tid]
        self.assertIn("search", tc["summary"])
        self.assertFalse(tc["open"])
        # No legacy toggle anchor / arrow — collapse is native.
        self.assertNotIn("tool://toggle", self.view.to_html())

    def test_tool_call_returns_id(self):
        tid = self.renderer.render_tool_call("search", {})
        self.assertTrue(tid.startswith("tc_"))

    def test_tool_result_attaches(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        self.renderer.render_tool_result(tid, '{"hits": 5}')
        # After result the summary carries the success icon and the body holds
        # the result text.
        self.assertIn("✓", self.view.collapsibles[tid]["summary"])
        self.assertIn("hits", self.view.collapsibles[tid]["body"])

    def test_tool_call_history(self):
        self.renderer.render_tool_call("search", {"q": "x"})
        self.assertEqual(len(self.renderer.history), 1)
        self.assertEqual(self.renderer.history[0].role.name, "TOOL_MARKER")
        self.assertEqual(self.renderer.history[0].metadata["tool_name"], "search")

    def test_error_result_marks_kind_error(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        self.renderer.render_tool_result(tid, "boom", status="error")
        self.assertEqual(self.view.collapsibles[tid]["kind"], "error")
        self.assertIn("✗", self.view.collapsibles[tid]["summary"])

    def test_success_result_has_no_kind(self):
        tid = self.renderer.render_tool_call("search", {"q": "x"})
        self.renderer.render_tool_result(tid, '{"hits": 5}')
        self.assertIsNone(self.view.collapsibles[tid]["kind"])

    def test_render_error_block_open_red(self):
        tid = self.renderer.render_error_block("Chat-Fehler", "Timeout after 30s")
        block = self.view.collapsibles[tid]
        self.assertEqual(block["kind"], "error")
        self.assertTrue(block["open"])
        self.assertIn("❌", block["summary"])
        self.assertIn("Timeout after 30s", block["body"])

    def test_unknown_tool_id_fallback(self):
        self.renderer.render_tool_result("nonexistent", "result")
        # Orphan result renders as a system message block.
        self.assertIn("orphan result", self.view.to_html())


class TestSystemMessage(RendererTestBase):

    def test_system_message_centered(self):
        self.renderer.render_system_message("status")
        self.assertIn('class="system-message"', self.view.to_html())

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
        html = self.view.to_html()
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


class TestHtmlBlock(RendererTestBase):

    def test_html_block_appended(self):
        self.renderer.render_html_block("<div>#1 DK 614.7</div>", kind="dk_classifications")
        self.assertIn("#1 DK 614.7", self.view.to_html())

    def test_html_block_history(self):
        self.renderer.render_html_block(
            "<div>card</div>", kind="dk_search", plain_text="DK 614.7"
        )
        self.assertEqual(len(self.renderer.history), 1)
        entry = self.renderer.history[0]
        self.assertEqual(entry.role.name, "RESULT_CARD")
        self.assertEqual(entry.metadata["kind"], "dk_search")
        self.assertEqual(entry.content, "DK 614.7")

    def test_empty_html_block_is_noop(self):
        self.renderer.render_html_block("")
        self.assertEqual(len(self.renderer.history), 0)


class TestMarkdownBlock(RendererTestBase):
    """render_markdown_block: Markdown → actual HTML (e.g. a real <table>),
    not the unrendered pipe-table text the GUI pipeline log previously showed
    for workflow reports (title_list_search's duplicate-check table). - Claude Generated"""

    def test_table_renders_as_html_table(self):
        md = "| A | B |\n|---|---|\n| x | y |"
        self.renderer.render_markdown_block(md, kind="workflow_report")
        html = self.view.to_html()
        self.assertIn("<table>", html)
        self.assertIn("<td>x</td>", html)

    def test_link_renders_as_anchor(self):
        md = "[Katalog](https://katalog.example/Record/0-1)"
        self.renderer.render_markdown_block(md)
        html = self.view.to_html()
        self.assertIn('<a href="https://katalog.example/Record/0-1"', html)
        self.assertIn(">Katalog</a>", html)

    def test_empty_markdown_is_noop(self):
        self.renderer.render_markdown_block("")
        self.assertEqual(len(self.renderer.history), 0)

    def test_history_records_kind_and_plain_text(self):
        md = "**Zusammenfassung:** 1 neu"
        self.renderer.render_markdown_block(md, kind="workflow_report")
        self.assertEqual(len(self.renderer.history), 1)
        entry = self.renderer.history[0]
        self.assertEqual(entry.metadata["kind"], "workflow_report")
        self.assertEqual(entry.content, md)

    def test_markdown_it_failure_falls_back_to_escaped_text(self):
        md = "<script>alert(1)</script> plain text"
        with patch("markdown_it.MarkdownIt.render", side_effect=RuntimeError("boom")):
            self.renderer.render_markdown_block(md)
        html = self.view.to_html()
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)


class TestCollapsible(RendererTestBase):

    def test_collapsible_collapsed_state(self):
        tid = self.renderer.render_collapsible(
            "Input 'classification'", "SECRET BODY", collapsed=True, meta="14:23:01"
        )
        tc = self.view.collapsibles[tid]
        self.assertIn("Input 'classification'", tc["summary"])
        self.assertIn("14:23:01", tc["summary"])
        self.assertFalse(tc["open"])           # starts collapsed
        self.assertIn("SECRET BODY", tc["body"])  # body present, just collapsed

    def test_collapsible_expanded_shows_body(self):
        tid = self.renderer.render_collapsible(
            "Input 'x'", "VISIBLE BODY", collapsed=False
        )
        self.assertTrue(self.view.collapsibles[tid]["open"])
        self.assertIn("VISIBLE BODY", self.view.collapsibles[tid]["body"])

    def test_update_collapsible_meta(self):
        tid = self.renderer.render_collapsible("t", "b", collapsed=True, meta="14:00:00")
        self.renderer.update_collapsible_meta(tid, "14:00:00  ⏱ 3.4s")
        self.assertIn("⏱ 3.4s", self.view.collapsibles[tid]["summary"])

    def test_collapsible_history_entry(self):
        self.renderer.render_collapsible("Title", "body")
        self.assertEqual(self.renderer.history[-1].metadata.get("kind"), "collapsible")


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
        self.assertEqual(len(self.view.blocks), 0)


class TestAutoScroll(RendererTestBase):

    def test_does_not_crash(self):
        self.renderer.auto_scroll_to_bottom()

    def test_respects_checkbox(self):
        self.checkbox._checked = False
        # Should not crash and should not scroll.
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

        self.view = _MockWebLogView()
        self.checkbox = _MockCheckBox(checked=True)
        self.renderer = UnifiedMessageRenderer(self.view, self.checkbox)

    def tearDown(self):
        from src.core import state_bus as state_bus_mod
        state_bus_mod.reset()

    def test_subscribe_unsubscribe_round_trip(self):
        self.renderer.subscribe()
        called_subs = [s for s in self.bus._subscriptions if s[0] == "tool.called"]
        result_subs = [s for s in self.bus._subscriptions if s[0] == "tool.result"]
        self.assertGreater(len(called_subs), 0)
        self.assertGreater(len(result_subs), 0)

        self.renderer.unsubscribe()
        for event, handler, _slot in self.bus._subscriptions:
            if event != "tool.called":
                continue
            qualname = getattr(handler, "__qualname__", "")
            self.assertFalse(
                qualname.startswith("UnifiedMessageRenderer._on_bus_tool_called"),
                f"renderer should have unsubscribed, found {qualname}",
            )

    def test_bus_events_forward_to_renderer(self):
        self.renderer.subscribe()
        try:
            self.bus.emit_event("tool.called", {
                "name": "search_gnd",
                "arguments": {"term": "Bibliothek"},
                "id": "tc_bus1",
            })
            self.assertEqual(len(self.renderer.history), 1)
            self.assertEqual(self.renderer.history[0].metadata["tool_name"], "search_gnd")
            self.assertIn("tc_bus1", self.renderer._bus_id_to_tool_id)

            self.bus.emit_event("tool.result", {
                "name": "search_gnd",
                "result": '{"hits": 3}',
                "id": "tc_bus1",
                "status": "ok",
            })
            self.assertIn("✓", self.view.to_html())
            self.assertNotIn("tc_bus1", self.renderer._bus_id_to_tool_id)
        finally:
            self.renderer.unsubscribe()

    def test_cache_hit_badge_propagates(self):
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
            tool_id = "tc_1"  # renderer's local id; first call → tc_1
            tc = self.renderer._tool_calls[tool_id]
            self.assertIn("📦 cache", tc["result"])
        finally:
            self.renderer.unsubscribe()

    def test_orphan_result_renders_marker(self):
        self.renderer.subscribe()
        try:
            self.bus.emit_event("tool.result", {
                "name": "lone",
                "result": "lone result",
                "id": "no-match",
                "status": "ok",
            })
            self.assertIn("↳", self.view.to_html())
        finally:
            self.renderer.unsubscribe()


class TestCatalogMarkerReplacement(RendererTestBase):
    """Tests for P-δ.5: <<CAT:rsn|display>> → clickable anchor."""

    WEB_BASE = "https://katalog.ub.tu-freiberg.de/Record/"

    def test_marker_replaced_with_anchor(self):
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers("Vor <<CAT:12345|Titel>> nach")
        self.assertIn('href="https://katalog.ub.tu-freiberg.de/Record/0-12345"', out)
        self.assertIn(">Titel</a>", out)
        self.assertNotIn("<<CAT:", out)
        self.assertTrue(out.startswith("Vor "))
        self.assertTrue(out.endswith(" nach"))

    def test_invalid_rsn_passes_through_as_text(self):
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers("Siehe <<CAT:abc|Mein Titel>>")
        self.assertNotIn("<a ", out, "No anchor should be emitted for invalid RSN")
        self.assertNotIn("<<CAT:", out)
        self.assertIn("Mein Titel", out)

    def test_empty_web_base_disables_feature(self):
        out = self.renderer._replace_cat_markers("X <<CAT:12345|Titel>> Y")
        self.assertNotIn("<a ", out, "Feature must be off when no base URL set")
        self.assertNotIn("<<CAT:", out)
        self.assertIn("Titel", out)

    def test_multiple_markers_all_replaced(self):
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers(
            "<<CAT:111|A>> und <<CAT:222|B>>"
        )
        self.assertIn("Record/0-111", out)
        self.assertIn("Record/0-222", out)
        self.assertIn(">A</a>", out)
        self.assertIn(">B</a>", out)

    def test_html_in_display_is_escaped(self):
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        out = self.renderer._replace_cat_markers('<<CAT:1|<script>x</script>>>')
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_no_marker_passthrough(self):
        self.renderer.set_catalog_web_base(self.WEB_BASE)
        text = "Just plain text, no marker here."
        self.assertEqual(self.renderer._replace_cat_markers(text), text)


class TestCLinkMarkerReplacement(RendererTestBase):
    """Tests for <<CLINK:url|display>> → clickable anchor (Claude Generated)."""

    def test_https_url_replaced_with_anchor(self):
        out = self.renderer._replace_clink_markers(
            "Vor <<CLINK:https://katalog.example.org/Record/0-123|Chemie>> nach"
        )
        self.assertIn('href="https://katalog.example.org/Record/0-123"', out)
        self.assertIn(">Chemie</a>", out)
        self.assertNotIn("<<CLINK:", out)
        self.assertTrue(out.startswith("Vor "))
        self.assertTrue(out.endswith(" nach"))

    def test_http_url_also_accepted(self):
        out = self.renderer._replace_clink_markers(
            "<<CLINK:http://katalog.example.org/Record/0-99|Titel>>"
        )
        self.assertIn('href="http://katalog.example.org/Record/0-99"', out)
        self.assertIn(">Titel</a>", out)

    def test_non_http_url_rejected(self):
        out = self.renderer._replace_clink_markers(
            "<<CLINK:ftp://evil.example/x|Anzeige>>"
        )
        self.assertNotIn("<a ", out)
        self.assertNotIn("<<CLINK:", out)
        self.assertIn("Anzeige", out)

    def test_html_in_display_is_escaped(self):
        out = self.renderer._replace_clink_markers(
            "<<CLINK:https://example.org/1|<script>x</script>>>"
        )
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_no_marker_passthrough(self):
        text = "Just plain text, no CLINK here."
        self.assertEqual(self.renderer._replace_clink_markers(text), text)

    def test_markdown_table_backslash_pipe_separator(self):
        # In Markdown tables | must be escaped as \| inside a cell.
        # The regex must handle <<CLINK:url\|title>> and produce a clean URL
        # (no trailing backslash in href).
        out = self.renderer._replace_clink_markers(
            "<<CLINK:https://katalog.example.org/Record/0-123\\|Quantenchemie>>"
        )
        self.assertIn('href="https://katalog.example.org/Record/0-123"', out)
        self.assertNotIn("\\", out.split("href=")[1].split('"')[1])  # no \ in href
        self.assertIn(">Quantenchemie</a>", out)

    def test_both_cat_and_clink_in_same_text(self):
        self.renderer.set_catalog_web_base("https://katalog.example.org/Record/")
        out = self.renderer._replace_cat_markers(
            "A: <<CAT:12345|Libero-Titel>> B: <<CLINK:https://finc.example.org/Record/0-9|finc-Titel>>"
        )
        out = self.renderer._replace_clink_markers(out)
        self.assertIn("Record/0-12345", out)
        self.assertIn("href=\"https://finc.example.org/Record/0-9\"", out)
        self.assertIn(">Libero-Titel</a>", out)
        self.assertIn(">finc-Titel</a>", out)


class TestClassifyLinks(RendererTestBase):
    """_classify_links: catalog links get cat-link, external links get ext-link. Claude Generated."""

    def test_local_link_gets_cat_link_class(self):
        self.renderer.set_catalog_host("https://katalog.example.org")
        html = '<a href="https://katalog.example.org/Record/0-123">Titel</a>'
        out = self.renderer._classify_links(html)
        self.assertIn("cat-link", out)
        self.assertNotIn("ext-link", out)
        self.assertIn('href="https://katalog.example.org/Record/0-123"', out)

    def test_external_link_gets_warning_class(self):
        self.renderer.set_catalog_host("https://katalog.example.org")
        html = '<a href="https://www.google.com/search?q=test">Google</a>'
        out = self.renderer._classify_links(html)
        self.assertIn("ext-link", out)

    def test_no_catalog_host_passthrough(self):
        # When no host is configured, all links pass through unchanged.
        html = '<a href="https://www.google.com/">Google</a>'
        out = self.renderer._classify_links(html)
        self.assertEqual(html, out)

    def test_doi_from_finc_classified_external_without_trust(self):
        self.renderer.set_catalog_host("https://katalog.example.org")
        html = '<a href="https://doi.org/10.1234/test">DOI-Link</a>'
        out = self.renderer._classify_links(html)
        self.assertIn("ext-link", out)

    def test_doi_trusted_via_tool_result_not_flagged(self):
        self.renderer.set_catalog_host("https://katalog.example.org")
        self.renderer.add_trusted_urls(["https://doi.org/10.1234/test"])
        html = '<a href="https://doi.org/10.1234/test">DOI-Link</a>'
        out = self.renderer._classify_links(html)
        self.assertNotIn("ext-link", out)

    def test_trusted_urls_cleared_on_new_user_message(self):
        self.renderer.set_catalog_host("https://katalog.example.org")
        self.renderer.add_trusted_urls(["https://doi.org/10.1234/test"])
        self.renderer.render_user_bubble("neue Frage")
        html = '<a href="https://doi.org/10.1234/test">DOI-Link</a>'
        out = self.renderer._classify_links(html)
        self.assertIn("ext-link", out)


class TestRenderEventEmission(unittest.TestCase):
    """WP12: the renderer emits a versioned JSON render-event stream to an
    injected transport (instead of calling WebLogView directly). These tests
    drive a Qt-free ``MockTransport`` and assert the event shapes that
    ``alima_render.js`` consumes."""

    def setUp(self):
        import json
        from src.core.render_events import MockTransport
        from src.ui.unified_message_renderer import UnifiedMessageRenderer

        self.json = json
        self.transport = MockTransport()
        self.checkbox = _MockCheckBox(checked=True)
        # Passing a transport (has .send) must be used as-is, not re-wrapped.
        self.renderer = UnifiedMessageRenderer(self.transport, self.checkbox)
        self.assertIs(self.renderer.transport, self.transport)

    def _last(self, type_):
        evs = self.transport.of_type(type_)
        self.assertTrue(evs, f"no {type_} event emitted; got {self.transport.types()}")
        return evs[-1]

    def test_pipeline_log_emits_block_with_kind(self):
        self.renderer.render_pipeline_log("hello world", "info", "init")
        ev = self._last("block")
        self.assertEqual(ev["kind"], "pipeline_log")
        self.assertIn("hello world", ev["html"])

    def test_user_bubble_kind(self):
        self.renderer.render_user_bubble("hi there")
        self.assertEqual(self._last("block")["kind"], "user_bubble")

    def test_proposal_is_gui_only_kind(self):
        from src.core import render_events as re_mod
        self.renderer.render_proposal_bubble(
            7, "propose_dk_change", {"code": "614.7", "action": "add"}
        )
        ev = self._last("block")
        self.assertEqual(ev["kind"], re_mod.KIND_PROPOSAL)
        self.assertIn(ev["kind"], re_mod.GUI_ONLY_KINDS)  # webapp may drop it

    def test_tool_call_emits_collapsible(self):
        tid = self.renderer.render_tool_call("search_gnd", {"term": "x"})
        ev = self._last("collapsible")
        self.assertEqual(ev["id"], tid)
        self.assertFalse(ev["open"])
        self.assertIn("search_gnd", ev["summary"])

    def test_tool_result_emits_collapsible_update(self):
        tid = self.renderer.render_tool_call("search_gnd", {"term": "x"})
        self.renderer.render_tool_result(tid, '{"hits": 3}', status="success")
        ev = self._last("collapsible_update")
        self.assertEqual(ev["id"], tid)
        self.assertIn("hits", ev["body"])

    def test_stream_block_lifecycle(self):
        self.renderer.start_streaming_line("keywords", "LLM: ")
        self.renderer.render_streaming_token("alpha beta", "keywords")
        self.renderer.end_streaming_line()
        opened = self._last("stream_open")
        token = self._last("stream_token")
        closed = self._last("stream_close")
        self.assertEqual(token["text"], "alpha beta")
        self.assertEqual(opened["id"], closed["id"])
        self.assertTrue(closed["collapse"])
        self.assertIn("alpha beta", closed["summary"])  # preview retained

    def test_assistant_bubble_lifecycle(self):
        self.renderer.open_assistant_bubble("gpt-4")
        self.renderer.append_assistant_token("**b**")
        self.renderer.finalize_assistant_bubble()
        self.assertIn("gpt-4", self._last("assistant_open")["header"])
        self.assertEqual(self._last("assistant_token")["text"], "**b**")
        self.assertIn("b", self._last("assistant_finalize")["html"])

    def test_clear_emits_clear_event(self):
        self.renderer.render_system_message("x")
        self.renderer.clear()
        self.assertEqual(self.transport.types()[-1], "clear")

    def test_all_events_are_json_serializable(self):
        self.renderer.render_pipeline_log("msg", "step", "init")
        self.renderer.render_user_bubble("u")
        self.renderer.render_tool_call("t", {"a": 1})
        self.renderer.render_html_block("<div>card</div>", kind="dk_search")
        self.renderer.start_streaming_line("keywords")
        self.renderer.render_streaming_token("tok", "keywords")
        self.renderer.end_streaming_line()
        # Round-trips without error and stays a flat list of dicts.
        dumped = self.json.dumps(self.transport.events)
        self.assertIsInstance(self.json.loads(dumped), list)
        for ev in self.transport.events:
            self.assertIn("type", ev)


class TestWebLogViewTransportMapping(unittest.TestCase):
    """The GUI transport maps each render event onto the WebLogView API."""

    def setUp(self):
        from src.ui.render_transport import WebLogViewTransport
        self.view = _MockWebLogView()
        self.transport = WebLogViewTransport(self.view)

    def test_block_maps_to_append_block(self):
        from src.core import render_events as ev
        self.transport.send(ev.block("<b>x</b>", kind=ev.KIND_SYSTEM))
        self.assertEqual(self.view.blocks, ["<b>x</b>"])

    def test_collapsible_round_trip(self):
        from src.core import render_events as ev
        self.transport.send(ev.collapsible("tc_1", "sum", "body", True))
        self.assertIn("tc_1", self.view.collapsibles)
        self.assertTrue(self.view.collapsibles["tc_1"]["open"])
        self.transport.send(ev.collapsible_update("tc_1", "sum2", "body2"))
        self.assertEqual(self.view.collapsibles["tc_1"]["summary"], "sum2")

    def test_stream_events_round_trip(self):
        from src.core import render_events as ev
        self.transport.send(ev.stream_open("sl_1", "header"))
        self.transport.send(ev.stream_token("abc"))
        self.transport.send(ev.stream_close("sl_1", "header — abc", collapse=True))
        blk = self.view.stream_blocks["sl_1"]
        self.assertIn("abc", "".join(blk["tokens"]))
        self.assertTrue(blk["collapsed"])

    def test_clear_maps_to_clear_log(self):
        from src.core import render_events as ev
        self.transport.send(ev.block("x"))
        self.transport.send(ev.clear())
        self.assertEqual(self.view.blocks, [])

    def test_unknown_event_type_is_ignored(self):
        # Forward-compat: an unknown type must not raise.
        self.transport.send({"type": "future_event", "foo": "bar"})


class TestLinkClassification(RendererTestBase):
    """GND/SWB authority links + catalog config wiring. Claude Generated."""

    RECORD_URL = "https://katalog.ub.tu-freiberg.de/Record/"
    SEARCH_URL = "https://katalog.ub.tu-freiberg.de/Search"

    def setUp(self):
        super().setUp()
        # Realistic GUI/webapp state: a catalog host is configured, so link
        # classification is active (no-op otherwise).
        self.renderer.configure_catalog(self.RECORD_URL, self.SEARCH_URL)

    def _finalize(self, markdown_text: str) -> str:
        self.renderer.open_assistant_bubble("m")
        self.renderer.append_assistant_token(markdown_text)
        self.renderer.finalize_assistant_bubble()
        return self.view.assistant_final or ""

    def test_gnd_link_not_flagged_external(self):
        html = self._finalize("[Quantenchemie](https://d-nb.info/gnd/4047979-1)")
        self.assertIn('href="https://d-nb.info/gnd/4047979-1"', html)
        self.assertNotIn("ext-link", html)

    def test_swb_link_not_flagged_external(self):
        html = self._finalize(
            "[Begriff](https://swb.bsz-bw.de/DB=2.104/PPNSET?PPN=106192760&INDEXSET=21)"
        )
        self.assertIn("swb.bsz-bw.de", html)
        self.assertNotIn("ext-link", html)

    def test_unknown_external_link_is_flagged(self):
        html = self._finalize("[Suche](https://www.google.com/search?q=x)")
        self.assertIn("ext-link", html)

    def test_trusted_tool_url_not_flagged(self):
        self.renderer.add_trusted_urls({"https://doi.org/10.1/xyz"})
        html = self._finalize("[Paper](https://doi.org/10.1/xyz)")
        self.assertNotIn("ext-link", html)

    def test_configure_catalog(self):
        self.renderer.configure_catalog(self.RECORD_URL, self.SEARCH_URL)
        self.assertEqual(
            self.renderer._catalog_web_base,
            "https://katalog.ub.tu-freiberg.de/Record",
        )
        self.assertIn(
            "https://katalog.ub.tu-freiberg.de", self.renderer._catalog_hosts
        )

    def test_configure_catalog_without_urls_disables_links(self):
        # No catalog configured (factory.catalog_web_bases returns ("", "") when
        # nothing is set or the config is unreadable): must not raise, and the
        # marker feature stays off. - Claude Generated
        self.renderer.configure_catalog("", "")
        self.assertEqual(self.renderer._catalog_web_base, "")
        html = self._finalize("Treffer: <<CAT:25515640|Quantenchemie>>")
        self.assertNotIn("href=", html)
        self.assertIn("Quantenchemie", html)

    def test_catalog_marker_links_after_config(self):
        self.renderer.configure_catalog(self.RECORD_URL, "")
        html = self._finalize("Treffer: <<CAT:25515640|Quantenchemie>>")
        self.assertIn(
            'href="https://katalog.ub.tu-freiberg.de/Record/0-25515640"', html
        )
        self.assertIn("cat-link", html)


if __name__ == "__main__":
    unittest.main()
