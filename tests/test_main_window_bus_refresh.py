"""Tests for MainWindow bus-driven result-tab refresh (Konvergenz Pipeline/Agent).
Claude Generated.

The real MainWindow needs a DB + services + QApplication, so — following the
pattern in ``test_pipeline_chat_panel.py`` — the new bus handlers are exercised
as unbound methods bound to a lightweight stub with MagicMock tabs. This
verifies the routing/guard logic (agent-driven completion distributes, classical
runs defer, incremental mutations re-render without appending history) without
constructing the full window.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core import state_bus
from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis
from src.ui.main_window import MainWindow


def _make_state(**over) -> KeywordAnalysisState:
    base = dict(
        original_abstract="abstract",
        initial_keywords=["A", "B"],
        search_suggesters_used=["lobid"],
    )
    base.update(over)
    return KeywordAnalysisState(**base)


def _make_stub(state=None, classical=False) -> SimpleNamespace:
    pm = SimpleNamespace(current_analysis_state=state)
    stub = SimpleNamespace(
        logger=MagicMock(),
        pipeline_manager=pm,
        _classical_active=classical,
        _pending_ops=set(),
        _rerender_timer=MagicMock(),
        on_pipeline_results_ready=MagicMock(),
        on_pipeline_title_update=MagicMock(),
        # In the real window dk_analysis_tab and dk_classification_tab alias the
        # same widget; kept separate here so the no-history assertion (the
        # history-appending update_data must not be used on the incremental
        # path) is meaningful.
        dk_classification_tab=MagicMock(),
        dk_analysis_tab=MagicMock(),
        ub_catalog_tab=MagicMock(),
        search_tab=MagicMock(),
        comparison_tab=MagicMock(),
        analysis_review_tab=MagicMock(),
        _rerender_result_tabs=MagicMock(),
    )
    stub._on_bus_pipeline_completed = MainWindow._on_bus_pipeline_completed.__get__(stub)
    stub._on_bus_state_changed = MainWindow._on_bus_state_changed.__get__(stub)
    stub._flush_rerender = MainWindow._flush_rerender.__get__(stub)
    return stub


class TestBusPipelineCompleted(unittest.TestCase):
    def setUp(self):
        state_bus.reset()

    def tearDown(self):
        state_bus.reset()

    def test_distributes_for_agent_driven_completion(self):
        state = _make_state()
        stub = _make_stub(state=state, classical=False)
        stub._on_bus_pipeline_completed({})
        stub.on_pipeline_results_ready.assert_called_once_with(state)
        stub.dk_classification_tab.update_data.assert_called_once_with(state)
        stub.ub_catalog_tab.update_from_pipeline.assert_called_once_with(state)
        stub.search_tab.update_data.assert_called_once_with(state)
        stub.comparison_tab.load_from_current.assert_called_once_with(state)
        stub.on_pipeline_title_update.assert_called_once_with(state)

    def test_skips_when_classical_active(self):
        state = _make_state()
        stub = _make_stub(state=state, classical=True)
        stub._on_bus_pipeline_completed({})
        stub.on_pipeline_results_ready.assert_not_called()
        stub.search_tab.update_data.assert_not_called()

    def test_noop_without_state(self):
        stub = _make_stub(state=None, classical=False)
        stub._on_bus_pipeline_completed({})
        stub.on_pipeline_results_ready.assert_not_called()


class TestBusStateChanged(unittest.TestCase):
    def setUp(self):
        state_bus.reset()

    def tearDown(self):
        state_bus.reset()

    def test_coalesces_into_single_rerender(self):
        state = _make_state()
        stub = _make_stub(state=state)
        stub._on_bus_state_changed({"op": "keyword_replacement"})
        stub._on_bus_state_changed({"op": "keyword_addition"})
        # The debounce timer is (re)started on each event...
        self.assertEqual(stub._rerender_timer.start.call_count, 2)
        self.assertEqual(
            stub._pending_ops, {"keyword_replacement", "keyword_addition"}
        )
        # ...and the actual re-render happens once on flush.
        stub._flush_rerender()
        stub._rerender_result_tabs.assert_called_once()
        args, kwargs = stub._rerender_result_tabs.call_args
        self.assertIs(args[0], state)
        self.assertEqual(
            kwargs.get("ops"), {"keyword_replacement", "keyword_addition"}
        )
        self.assertEqual(stub._pending_ops, set())

    def test_flush_noop_without_state(self):
        stub = _make_stub(state=None)
        stub._on_bus_state_changed({"op": "x"})
        stub._flush_rerender()
        stub._rerender_result_tabs.assert_not_called()


class TestRerenderIdempotent(unittest.TestCase):
    """The incremental re-render must update data views but NOT append history."""

    def setUp(self):
        state_bus.reset()

    def tearDown(self):
        state_bus.reset()

    def test_updates_data_views_without_history_append(self):
        dk_llm = LlmKeywordAnalysis(
            task_name="dk", model_used="m", provider_used="p",
            prompt_template="", filled_prompt="", temperature=0.0, seed=None,
            response_full_text="DK response",
        )
        state = _make_state(
            dk_search_results_flattened=[{"code": "004", "title": "Informatik"}],
            dk_llm_analysis=dk_llm,
        )
        stub = _make_stub(state=state)
        # Exercise the REAL re-render for this test.
        stub._rerender_result_tabs = MainWindow._rerender_result_tabs.__get__(stub)

        stub._rerender_result_tabs(state)

        stub.search_tab.update_data.assert_called_once_with(state)
        stub.ub_catalog_tab.update_from_pipeline.assert_called_once_with(state)
        stub.dk_analysis_tab.set_keywords.assert_called_once_with(
            state.dk_search_results_flattened
        )
        stub.dk_analysis_tab.display_llm_response.assert_called_once_with("DK response")
        stub.analysis_review_tab.receive_full_state.assert_called_once_with(state)
        stub.comparison_tab.load_from_current.assert_called_once_with(state)
        stub.on_pipeline_title_update.assert_called_once_with(state)
        # Crucial: the history-appending DK update_data must NOT be used here.
        stub.dk_classification_tab.update_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
