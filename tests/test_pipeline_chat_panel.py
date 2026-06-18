"""Tests for PipelineChatPanel (P-δ.5a unified widget). Claude Generated.

Avoid instantiating the full widget (heavy QApplication hierarchy +
conflicts with QCoreApplication-only tests in the same suite). Instead
exercise pure-logic helpers as unbound methods on a minimal stand-in.

Covered:
1. ``_shared_context_from_analysis_state`` maps a KeywordAnalysisState into
   a SharedContext with gnd_entries, dk_classifications, missing concepts.
2. ``_format_tool_args`` truncates long args sensibly.
3. ``_on_bus_tool_called`` / ``_on_bus_tool_result`` append markers via
   the panel's ``_append_tool_marker`` (intercepted).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.data_models import (
    KeywordAnalysisState,
    LlmKeywordAnalysis,
    SearchResult,
)
from src.ui.pipeline_chat_panel import PipelineChatPanel
from src.ui.unified_message_renderer import UnifiedMessageRenderer


def _make_stub_panel() -> SimpleNamespace:
    """Minimal panel-like object with intercepted renderer output."""
    markers: list[str] = []
    tool_calls: dict = {}
    tool_results: list[tuple] = []
    tool_call_counter = {"n": 0}

    class FakeRenderer:
        def render_tool_marker(self, text, tool_name=None):
            markers.append(text)
        def render_tool_call(self, name, args):
            tool_call_counter["n"] += 1
            tcid = f"tc_{tool_call_counter['n']}"
            tool_calls[tcid] = {"name": name, "args": args}
            return tcid
        def render_tool_result(self, tool_id, result, status="success"):
            tool_results.append((tool_id, result, status))

    system_messages: list[str] = []

    stub = SimpleNamespace(
        logger=MagicMock(),
        _append_tool_marker=markers.append,
        _append_system_message=system_messages.append,
        _renderer=FakeRenderer(),
        _bus_tool_call_ids={},
        _last_tool_call_id=None,
        _pipeline_step_open=False,
        _open_step_status=[],
    )
    stub.markers = markers
    stub.system_messages = system_messages
    stub.tool_calls = tool_calls
    stub.tool_results = tool_results
    stub._format_tool_args = PipelineChatPanel._format_tool_args
    stub._on_bus_tool_called = PipelineChatPanel._on_bus_tool_called.__get__(stub)
    stub._on_bus_tool_result = PipelineChatPanel._on_bus_tool_result.__get__(stub)
    stub._on_bus_pipeline_step = PipelineChatPanel._on_bus_pipeline_step.__get__(stub)
    return stub


class TestSharedContextAdapter(unittest.TestCase):

    def test_maps_all_relevant_fields(self):
        state = KeywordAnalysisState(
            original_abstract="Cadmium toxicity overview.",
            initial_keywords=["Cadmium", "Phytoremediation"],
            search_suggesters_used=["lobid"],
            working_title="Cadmium Study",
            search_results=[
                SearchResult(
                    search_term="Cadmium",
                    results={"4029259-9": {"title": "Cadmium", "system": "gnd"}},
                ),
            ],
            final_llm_analysis=LlmKeywordAnalysis(
                task_name="final",
                model_used="m",
                provider_used="p",
                prompt_template="",
                filled_prompt="",
                temperature=0.5,
                seed=None,
                response_full_text="",
                extracted_gnd_keywords=["Cadmium"],
                missing_concepts=["Speziation"],
            ),
            dk_classifications=[{"code": "631.811", "title": "Pflanzen"}],
            dk_search_results=[{"keyword": "Cadmium", "hits": 12}],
        )
        ctx = PipelineChatPanel._shared_context_from_analysis_state(state)
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.working_title, "Cadmium Study")
        self.assertEqual(ctx.extracted_keywords, ["Cadmium"])
        self.assertEqual(ctx.missing_concepts, ["Speziation"])
        self.assertEqual(len(ctx.gnd_entries), 1)
        self.assertEqual(ctx.gnd_entries[0]["gnd_id"], "4029259-9")
        self.assertEqual(ctx.gnd_entries_per_keyword["Cadmium"], ["Cadmium"])
        self.assertEqual(ctx.dk_classifications[0]["code"], "631.811")
        self.assertEqual(ctx.dk_search_results[0]["keyword"], "Cadmium")


class TestFormatToolArgs(unittest.TestCase):

    def test_empty_dict_returns_empty_string(self):
        self.assertEqual(PipelineChatPanel._format_tool_args({}), "")

    def test_short_args_render_inline(self):
        self.assertEqual(
            PipelineChatPanel._format_tool_args({"kind": "initial"}),
            "kind='initial'",
        )

    def test_long_value_truncated(self):
        out = PipelineChatPanel._format_tool_args({"q": "x" * 80})
        self.assertIn("q='", out)
        self.assertIn("…", out)


class TestBusToolHandlers(unittest.TestCase):

    def test_bus_tool_called_renders_tool_call(self):
        stub = _make_stub_panel()
        stub._on_bus_tool_called(
            {"name": "get_keywords", "arguments": {"kind": "initial"}, "id": "t1"}
        )
        self.assertIn("t1", stub._bus_tool_call_ids)

    def test_bus_tool_result_renders_preview(self):
        stub = _make_stub_panel()
        stub._on_bus_tool_result({"name": "get_keywords", "result": "ok"})
        # No matching tool_call id → fallback system message (not a legacy marker)
        self.assertEqual(len(stub.markers), 0)
        self.assertEqual(len(stub.system_messages), 1)
        self.assertIn("ok", stub.system_messages[0])

    def test_bus_tool_result_truncates_long_payload(self):
        stub = _make_stub_panel()
        stub._on_bus_tool_result({"name": "x", "result": "y" * 200})
        self.assertEqual(len(stub.markers), 0)
        self.assertEqual(len(stub.system_messages), 1)
        self.assertTrue(stub.system_messages[0].endswith("…"))


class TestPipelineStepBusAsToolBlock(unittest.TestCase):
    """``state.pipeline_step`` renders as collapsible tool-call block."""

    def test_running_step_renders_tool_call(self):
        stub = _make_stub_panel()
        stub._on_bus_pipeline_step(
            {"status": "running", "step_id": "search", "name": "Search", "tool": "run_pipeline"}
        )
        # No legacy marker, exactly one tool call opened.
        self.assertEqual(stub.markers, [])
        self.assertEqual(len(stub.tool_calls), 1)
        tc = next(iter(stub.tool_calls.values()))
        self.assertEqual(tc["name"], "pipeline.search")
        self.assertEqual(tc["args"]["step"], "search")
        self.assertEqual(tc["args"]["tool"], "run_pipeline")
        # Panel tracks the open id for the upcoming completion event.
        self.assertIsNotNone(stub._last_tool_call_id)

    def test_completed_step_attaches_success_result(self):
        stub = _make_stub_panel()
        stub._on_bus_pipeline_step(
            {"status": "running", "step_id": "search", "name": "Search"}
        )
        running_id = stub._last_tool_call_id
        stub._on_bus_pipeline_step(
            {"status": "completed", "step_id": "search", "name": "Search"}
        )
        self.assertEqual(len(stub.tool_results), 1)
        tool_id, result_text, status = stub.tool_results[0]
        self.assertEqual(tool_id, running_id)
        self.assertEqual(status, "success")
        self.assertIn("completed", result_text)
        # Id is cleared after attach.
        self.assertIsNone(stub._last_tool_call_id)

    def test_error_step_attaches_error_result(self):
        stub = _make_stub_panel()
        stub._on_bus_pipeline_step(
            {"status": "running", "step_id": "search", "name": "Search"}
        )
        running_id = stub._last_tool_call_id
        stub._on_bus_pipeline_step(
            {"status": "error", "step_id": "search", "name": "Search"}
        )
        self.assertEqual(len(stub.tool_results), 1)
        tool_id, result_text, status = stub.tool_results[0]
        self.assertEqual(tool_id, running_id)
        self.assertEqual(status, "error")
        self.assertIn("error", result_text)

    def test_orphan_completion_renders_block(self):
        stub = _make_stub_panel()
        # No prior running event.
        stub._on_bus_pipeline_step(
            {"status": "completed", "step_id": "verify", "name": "Verify"}
        )
        # Should still render a complete tool-call + result pair.
        self.assertEqual(len(stub.tool_calls), 1)
        self.assertEqual(len(stub.tool_results), 1)
        tool_id, result_text, status = stub.tool_results[0]
        self.assertIn(tool_id, stub.tool_calls)
        self.assertEqual(status, "success")
        self.assertIn("completed", result_text)

    def test_deterministic_tool_call_renders_with_cache_badge(self):
        """Phase B: ``cache_hit`` payload triggers 📦 badge in result text."""
        stub = _make_stub_panel()
        # 1) Open a tool-call block via the bus.
        stub._on_bus_tool_called(
            {"name": "search_gnd", "arguments": {"term": "x"}, "id": "abc"}
        )
        # 2) Result event with cache_hit=True.
        stub._on_bus_tool_result(
            {"name": "search_gnd", "result": '[{"id": 1}]', "id": "abc", "cache_hit": True}
        )
        # The result_text captured by FakeRenderer carries the badge.
        self.assertEqual(len(stub.tool_results), 1)
        _, result_text, _ = stub.tool_results[0]
        self.assertIn("📦 cache", result_text)
        # And the id-keyed map is consumed.
        self.assertNotIn("abc", stub._bus_tool_call_ids)

    def test_classic_pipeline_renders_as_tool_block(self):
        """Phase C: classic.* tool events render as collapsible blocks."""
        stub = _make_stub_panel()
        # Simulate the two events a classic step emits.
        stub._on_bus_tool_called({
            "name": "classic.keywords", "arguments": {"task": "keywords"}, "id": "k1",
        })
        stub._on_bus_tool_result({
            "name": "classic.keywords", "result": "ok", "id": "k1", "status": "ok",
        })
        self.assertEqual(len(stub.tool_calls), 1)
        tc = next(iter(stub.tool_calls.values()))
        self.assertEqual(tc["name"], "classic.keywords")
        self.assertEqual(tc["args"]["task"], "keywords")
        # Result event matched the id and attached a status.
        # Phase E normalization: bus "ok" → renderer-internal "success"
        # so the ✓ icon picks the right glyph.
        self.assertEqual(len(stub.tool_results), 1)
        _, _, status = stub.tool_results[0]
        self.assertEqual(status, "success")


class TestStepStatusAccumulator(unittest.TestCase):
    """Post-(F) polish: status messages emitted while a pipeline-step
    tool block is open are absorbed into the block's body instead of
    rendered as separate markers after the collapsed block.
    """

    def _make_stub(self) -> SimpleNamespace:
        markers: list[str] = []
        tool_calls: dict = {}
        tool_results: list[tuple] = []
        tool_call_counter = {"n": 0}

        pipeline_logs: list[tuple] = []

        class FakeRenderer:
            def render_tool_marker(self, text, tool_name=None):
                markers.append(text)
            def render_pipeline_log(self, text, level="info", step_id=None):
                pipeline_logs.append((text, level))
            def render_tool_call(self, name, args):
                tool_call_counter["n"] += 1
                tcid = f"tc_{tool_call_counter['n']}"
                tool_calls[tcid] = {"name": name, "args": args}
                return tcid
            def render_tool_result(self, tool_id, result, status="success"):
                tool_results.append((tool_id, result, status))

        stub = SimpleNamespace(
            logger=MagicMock(),
            _renderer=FakeRenderer(),
            _bus_tool_call_ids={},
            _last_tool_call_id=None,
            _pipeline_step_open=False,
            _open_step_status=[],
            markers=markers,
            pipeline_logs=pipeline_logs,
            tool_calls=tool_calls,
            tool_results=tool_results,
            _append_tool_marker=markers.append,
        )
        # Bind real panel methods.
        stub._on_status_message = PipelineChatPanel._on_status_message.__get__(stub)
        stub._on_bus_pipeline_step = PipelineChatPanel._on_bus_pipeline_step.__get__(stub)
        return stub

    def test_status_lines_during_step_become_block_body(self):
        """``🔍 Suche 'Cadmium'…`` etc. accumulate into the open block."""
        stub = self._make_stub()
        # 1) Step starts → opens a tool block.
        stub._on_bus_pipeline_step({
            "status": "running", "step_id": "search", "name": "GND Search",
        })
        self.assertIsNotNone(stub._last_tool_call_id)
        # 2) Three search status lines arrive while the step is open.
        stub._on_status_message("🔍 Suche 'Cadmium'...")
        stub._on_status_message("    ✓ 'Cadmium': 115 Treffer")
        stub._on_status_message("🔍 Suche 'Phytoremediation'...")
        # 3) Step completes → the accumulator becomes the block body.
        stub._on_bus_pipeline_step({
            "status": "completed", "step_id": "search", "name": "GND Search",
        })
        # No extra markers were emitted.
        self.assertEqual(stub.markers, [])
        # Exactly one tool-call / one tool-result, body = joined status.
        self.assertEqual(len(stub.tool_calls), 1)
        self.assertEqual(len(stub.tool_results), 1)
        block_id = list(stub.tool_calls.keys())[0]
        tool_id, result_text, status = stub.tool_results[0]
        self.assertEqual(tool_id, block_id)
        self.assertIn("Cadmium", result_text)
        self.assertIn("115 Treffer", result_text)
        self.assertIn("Phytoremediation", result_text)
        self.assertEqual(status, "success")
        # Accumulator reset for the next step.
        self.assertEqual(stub._open_step_status, [])

    def test_status_outside_step_still_emits_log(self):
        """Status lines without an open step go to dim pipeline log
        (LLM streaming tokens, chat-agent status, etc.) — not legacy markers."""
        stub = self._make_stub()
        stub._on_status_message("🔄 Tool-Call 1/30")
        self.assertEqual(stub.markers, [])
        self.assertEqual(len(stub.pipeline_logs), 1)
        self.assertEqual(stub.pipeline_logs[0][0], "🔄 Tool-Call 1/30")
        self.assertEqual(stub.pipeline_logs[0][1], "debug")

    def test_step_with_no_status_uses_fallback_summary(self):
        """If a step emits no status lines, the old summary text is used."""
        stub = self._make_stub()
        stub._on_bus_pipeline_step({
            "status": "running", "step_id": "initialisation", "name": "Init",
        })
        # No status messages arrive.
        stub._on_bus_pipeline_step({
            "status": "completed", "step_id": "initialisation", "name": "Init",
        })
        self.assertEqual(len(stub.tool_results), 1)
        _, result_text, _ = stub.tool_results[0]
        # Fallback: short "completed: <name>" line.
        self.assertIn("completed", result_text)
        self.assertIn("Init", result_text)


class TestPipelineLifecycleEvents(unittest.TestCase):
    """Phase F: ``state.pipeline_started`` / ``state.pipeline_completed``
    events now have real consumers in PipelineChatPanel (previously dead).

    The handler implementations live on the real panel, but to avoid
    instantiating the full widget we exercise the *handler functions*
    directly: the panel wires them as bus subscribers in ``__init__``.
    """

    def test_on_bus_pipeline_completed_renders_system_message(self):
        """The handler appends a system message via the renderer."""
        from src.ui.pipeline_chat_panel import PipelineChatPanel

        # Minimal stand-in with a renderer that records system messages.
        messages: list[str] = []
        stub = SimpleNamespace(
            _renderer=SimpleNamespace(
                render_system_message=lambda text: messages.append(text)
            )
        )
        handler = PipelineChatPanel._on_bus_pipeline_completed.__get__(stub)
        handler({"workflow": "alima_classic"})
        self.assertEqual(len(messages), 1)
        self.assertIn("abgeschlossen", messages[0])
        self.assertIn("alima_classic", messages[0])

    def test_on_bus_pipeline_completed_handles_empty_payload(self):
        """Empty payload (legacy emitter) must not raise."""
        from src.ui.pipeline_chat_panel import PipelineChatPanel

        messages: list[str] = []
        stub = SimpleNamespace(
            _renderer=SimpleNamespace(
                render_system_message=lambda text: messages.append(text)
            )
        )
        handler = PipelineChatPanel._on_bus_pipeline_completed.__get__(stub)
        # Must not raise.
        handler({})
        self.assertEqual(len(messages), 1)


class TestCancelLifecycle(unittest.TestCase):
    """Cancel must not flip the UI to 'ready' until the worker finishes. Claude Generated."""

    def _stub(self, running=True, stopping=False):
        stub = SimpleNamespace(
            current_worker=MagicMock(),
            _stopping=stopping,
            send_btn=MagicMock(),
            cancel_btn=MagicMock(),
            input_field=MagicMock(),
            system_messages=[],
        )
        stub.current_worker.isRunning.return_value = running
        stub._append_system_message = stub.system_messages.append
        stub._set_ui_stopping = lambda: PipelineChatPanel._set_ui_stopping(stub)
        stub._set_ui_running = lambda r: PipelineChatPanel._set_ui_running(stub, r)
        return stub

    def test_cancel_signals_worker_and_locks_ui(self):
        stub = self._stub()
        PipelineChatPanel.cancel_generation(stub)
        stub.current_worker.request_stop.assert_called_once()
        self.assertTrue(stub._stopping)
        # UI does NOT go to 'ready': send hidden, input + cancel disabled
        # (so a second click can't re-fire request_stop).
        stub.send_btn.setVisible.assert_called_with(False)
        stub.input_field.setEnabled.assert_called_with(False)
        stub.cancel_btn.setEnabled.assert_called_with(False)

    def test_cancel_is_idempotent_while_stopping(self):
        stub = self._stub(stopping=True)
        PipelineChatPanel.cancel_generation(stub)
        stub.current_worker.request_stop.assert_not_called()

    def test_finish_after_cancel_returns_to_ready(self):
        stub = self._stub(running=False, stopping=True)
        stub.session = SimpleNamespace(messages=[])
        stub._renderer = SimpleNamespace(_assistant_block_open=False)
        stub._hide_typing = MagicMock()
        stub._finalize_assistant_message = MagicMock()
        result = SimpleNamespace(messages=[], content="")
        PipelineChatPanel._on_finished(stub, result)
        # _set_ui_running(False) is the single owner that clears 'stopping'
        # and re-arms the cancel button.
        self.assertFalse(stub._stopping)
        stub.cancel_btn.setEnabled.assert_called_with(True)
        self.assertTrue(any("Abgebrochen" in m for m in stub.system_messages))


if __name__ == "__main__":
    unittest.main()
