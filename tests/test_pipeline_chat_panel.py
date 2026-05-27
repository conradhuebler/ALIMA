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
    stub = SimpleNamespace(
        logger=MagicMock(),
        _append_tool_marker=markers.append,
        _renderer=SimpleNamespace(render_tool_marker=markers.append),
    )
    stub.markers = markers
    stub._format_tool_args = PipelineChatPanel._format_tool_args
    stub._on_bus_tool_called = PipelineChatPanel._on_bus_tool_called.__get__(stub)
    stub._on_bus_tool_result = PipelineChatPanel._on_bus_tool_result.__get__(stub)
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

    def test_bus_tool_called_renders_marker(self):
        stub = _make_stub_panel()
        stub._on_bus_tool_called(
            {"name": "get_keywords", "arguments": {"kind": "initial"}, "id": "t1"}
        )
        self.assertEqual(len(stub.markers), 1)
        self.assertIn("get_keywords", stub.markers[0])
        self.assertIn("🔧", stub.markers[0])

    def test_bus_tool_result_renders_preview(self):
        stub = _make_stub_panel()
        stub._on_bus_tool_result({"name": "get_keywords", "result": "ok"})
        self.assertEqual(stub.markers, ["↳ ok"])

    def test_bus_tool_result_truncates_long_payload(self):
        stub = _make_stub_panel()
        stub._on_bus_tool_result({"name": "x", "result": "y" * 200})
        self.assertEqual(len(stub.markers), 1)
        self.assertTrue(stub.markers[0].endswith("…"))


if __name__ == "__main__":
    unittest.main()
