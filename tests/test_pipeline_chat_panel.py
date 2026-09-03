"""Tests for PipelineChatPanel (P-δ.5a unified widget). Claude Generated.

Avoid instantiating the full widget (heavy QApplication hierarchy +
conflicts with QCoreApplication-only tests in the same suite). Instead
exercise pure-logic helpers as unbound methods on a minimal stand-in.

Covered:
1. ``_shared_context_from_analysis_state`` maps a KeywordAnalysisState into
   a SharedContext with gnd_entries, dk_classifications, missing concepts.
2. ``_format_tool_args`` truncates long args sensibly.
3. ``_on_bus_tool_called`` / ``_on_bus_tool_result`` render collapsible
   tool blocks via the renderer (intercepted).
4. ``_on_anchor_clicked`` routes a proposal accept/reject link to the gateway
   and leaves ordinary links to the external-open branch.
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

    def test_long_list_shows_count_and_preview_not_raw_repr_cutoff(self):
        # Real reported bug: search_finc(terms=[30 long titles]) rendered as
        # "terms=['The Fraying Bonds of Peace – Economic …" — a crude
        # repr()-truncation that hid there even were more titles. - Claude Generated
        terms = [f"Book Title Number {i} With Some Extra Words" for i in range(30)]
        out = PipelineChatPanel._format_tool_args({"terms": terms, "search_type": "title", "limit": 5})
        self.assertIn("terms=[30:", out)
        self.assertIn("search_type='title'", out)

    def test_short_list_shown_fully_without_ellipsis_marker(self):
        out = PipelineChatPanel._format_tool_args({"terms": ["A", "B"]})
        self.assertEqual(out, "terms=[2: A, B]")

    def test_empty_list_does_not_crash(self):
        out = PipelineChatPanel._format_tool_args({"terms": []})
        self.assertIn("terms=", out)


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

    def test_status_outside_step_goes_to_logger_only(self):
        """Status lines without an open step no longer leak into the chat
        surface (Chat-UX 7/9) — they go to the logger; tool activity is
        already visible via the collapsible blocks."""
        stub = self._make_stub()
        stub._on_status_message("🔄 Tool-Call 1/30")
        self.assertEqual(stub.markers, [])
        self.assertEqual(stub.pipeline_logs, [])
        stub.logger.debug.assert_called_once()

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
        self.assertTrue(any("abgebrochen" in m.lower() for m in stub.system_messages))


class TestPipelineLogSummaries(unittest.TestCase):
    """Pure-string helpers moved to PipelineLogMixin (F-5 split). Reachable as
    PipelineChatPanel.<name> via the MRO; exercised on a minimal stub."""

    def _stub(self) -> SimpleNamespace:
        stub = SimpleNamespace()
        stub._format_dk_search_results = (
            PipelineChatPanel._format_dk_search_results.__get__(stub)
        )
        stub._build_step_summary = PipelineChatPanel._build_step_summary.__get__(stub)
        return stub

    def test_build_step_summary_keywords_with_verification(self):
        stub = self._stub()
        step = SimpleNamespace(
            step_id="keywords",
            output_data={
                "final_keywords": ["Cadmium", "Phytoremediation"],
                "verification": {
                    "stats": {"verified_count": 1, "total_extracted": 2},
                    "rejected": ["Foo (no hit)"],
                },
            },
        )
        out = stub._build_step_summary(step, "1.2s")
        self.assertIn("Abgeschlossen in 1.2s", out)
        self.assertIn("Gefunden: 2 Keywords", out)
        self.assertIn("1/2 Keywords GND-verifiziert", out)
        self.assertIn("Foo", out)

    def test_build_step_summary_search(self):
        stub = self._stub()
        step = SimpleNamespace(step_id="search", output_data={"search_results": 42})
        out = stub._build_step_summary(step, "0.5s")
        self.assertIn("Gefunden: 42 GND-Einträge", out)

    def test_build_step_summary_no_output(self):
        stub = self._stub()
        step = SimpleNamespace(step_id="search", output_data=None)
        self.assertEqual(stub._build_step_summary(step, "9s"), "✅ Abgeschlossen in 9s")

    def test_format_dk_search_results_counts(self):
        stub = self._stub()
        out = stub._format_dk_search_results(
            [
                {"keyword": "A", "source": "cache", "classifications": [{"c": 1}]},
                {"keyword": "B", "source": "live", "classifications": []},
            ]
        )
        self.assertIn("2 Keywords", out)
        self.assertIn("1 erfolgreich", out)
        self.assertIn("📦 Cache: 1 | 🔍 Live: 1", out)

    def test_format_dk_search_results_empty(self):
        stub = self._stub()
        self.assertEqual(
            stub._format_dk_search_results([]),
            "Keine Klassifikationen (DK/RVK) gefunden",
        )


class TestOnPipelineCompletedReportMarkdown(unittest.TestCase):
    """on_pipeline_completed must render extra.report_markdown (surfaced via
    KeywordAnalysisState.report_markdown) as an actual HTML table — the bug
    reported live: the GUI chat log couldn't render the workflow's Markdown
    table, only unrendered pipe-text. - Claude Generated"""

    def _stub(self) -> SimpleNamespace:
        markdown_calls: list = []

        class FakeRenderer:
            def render_markdown_block(self, markdown_text, *, kind=None):
                markdown_calls.append((markdown_text, kind))
            def render_html_block(self, html, *, kind=None, plain_text=""):
                pass
            def render_pipeline_log(self, message, level="info", step_id=None):
                # render_pipeline_result emits the completion/keyword lines
                # through the renderer, no longer via add_pipeline_message.
                pass

        stub = SimpleNamespace(
            logger=MagicMock(),
            _renderer=FakeRenderer(),
            add_pipeline_message=MagicMock(),
            load_context=MagicMock(),
        )
        stub.markdown_calls = markdown_calls
        stub.on_pipeline_completed = PipelineChatPanel.on_pipeline_completed.__get__(stub)
        return stub

    def _make_state(self, report_markdown=""):
        return KeywordAnalysisState(
            original_abstract="", initial_keywords=[], search_suggesters_used=[],
            report_markdown=report_markdown,
        )

    def test_report_markdown_present_renders_as_markdown_block(self):
        stub = self._stub()
        state = self._make_state("| Titel | Status |\n|---|---|\n| A | neu |")
        stub.on_pipeline_completed(state)
        self.assertEqual(len(stub.markdown_calls), 1)
        text, kind = stub.markdown_calls[0]
        self.assertIn("| A | neu |", text)
        self.assertEqual(kind, "workflow_report")

    def test_report_markdown_empty_renders_nothing(self):
        stub = self._stub()
        state = self._make_state("")
        stub.on_pipeline_completed(state)
        self.assertEqual(stub.markdown_calls, [])

    def test_none_analysis_state_does_not_raise(self):
        stub = self._stub()
        stub.on_pipeline_completed(None)
        self.assertEqual(stub.markdown_calls, [])


if __name__ == "__main__":
    unittest.main()


class TestProposalDecision(unittest.TestCase):
    """The confirmation must reach the gateway exactly once.

    Two regressions this pins: the decision surface used to be an anchor in the
    chat log, which the QWebEngineView never delivered while it used a custom
    scheme; and once it did work, it stayed clickable, so one question could be
    answered repeatedly.
    """

    def _panel(self):
        from unittest.mock import MagicMock

        panel = SimpleNamespace(
            proposal_gateway=MagicMock(),
            logger=MagicMock(),
            _append_html=lambda html: None,
        )
        panel._handle_mutation_link = (
            lambda *a, **kw: PipelineChatPanel._handle_mutation_link(panel, *a, **kw)
        )
        return panel

    def test_accept_resolves_the_decision(self):
        panel = self._panel()
        PipelineChatPanel._on_proposal_decided(panel, 42, True)
        panel.proposal_gateway.resolve_decision.assert_called_once_with(42, True)

    def test_reject_resolves_the_decision(self):
        panel = self._panel()
        PipelineChatPanel._on_proposal_decided(panel, 7, False)
        panel.proposal_gateway.resolve_decision.assert_called_once_with(7, False)

    def test_a_catalog_link_still_opens_externally(self):
        from PyQt6.QtCore import QUrl

        panel = self._panel()
        with unittest.mock.patch(
            "src.ui.pipeline_chat_panel.QDesktopServices.openUrl"
        ) as open_url:
            PipelineChatPanel._on_anchor_clicked(
                panel, QUrl("https://katalog.ub.tu-freiberg.de/Record/0-025515640")
            )
        panel.proposal_gateway.resolve_decision.assert_not_called()
        open_url.assert_called_once()


class TestProposalBar(unittest.TestCase):
    """The bar is a one-shot: it answers once and then has nothing to answer."""

    # The suite bootstrap creates a QApplication but keeps no reference, so by
    # the time this class runs ``QCoreApplication.instance()`` can be None
    # again — and constructing a QWidget then aborts the process instead of
    # raising. Hold the app and every widget for the whole class.
    _kept: list = []

    def _bar(self):
        from PyQt6.QtWidgets import QApplication, QWidget
        from src.ui.proposal_bar import ProposalBar

        if not self._kept:
            self._kept.append(QApplication.instance() or QApplication(["alima-tests"]))
            self._kept.append(QWidget())
        bar = ProposalBar(parent=self._kept[1])
        self._kept.append(bar)
        return bar

    def test_hidden_until_a_proposal_arrives(self):
        bar = self._bar()
        self.assertFalse(bar.isVisible())
        self.assertIsNone(bar.pending_audit_id)

    def test_shows_the_rule_wording_condition_and_scope(self):
        bar = self._bar()
        bar.show_proposal(
            1, "propose_rule",
            {"text": "Regeltext.", "applies_when": "bei X",
             "scope": "* × selection", "reason": "weil"},
        )
        self.assertEqual(bar.pending_audit_id, 1)
        self.assertIn("Regeltext.", bar.body_label.text())
        self.assertIn("bei X", bar.body_label.text())
        self.assertIn("selection", bar.meta_label.text())
        self.assertIn("weil", bar.meta_label.text())

    def test_a_decision_is_emitted_once_and_the_bar_closes(self):
        bar = self._bar()
        seen = []
        bar.decided.connect(lambda aid, ok: seen.append((aid, ok)))
        bar.show_proposal(5, "propose_rule", {"text": "R"})

        bar.accept_btn.click()
        self.assertEqual(seen, [(5, True)])
        self.assertIsNone(bar.pending_audit_id)

        # Clicking again must not answer the same question a second time.
        bar.accept_btn.click()
        bar.reject_btn.click()
        self.assertEqual(seen, [(5, True)])

    def test_reject_emits_false(self):
        bar = self._bar()
        seen = []
        bar.decided.connect(lambda aid, ok: seen.append((aid, ok)))
        bar.show_proposal(9, "delete_rule", {"text": "R"})
        bar.reject_btn.click()
        self.assertEqual(seen, [(9, False)])

    def test_an_unknown_tool_still_gets_a_title(self):
        bar = self._bar()
        bar.show_proposal(1, "something_new", {"a": "b"})
        self.assertTrue(bar.title_label.text())
