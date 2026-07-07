"""Tests for v4-agent foundation classes (``SharedContext`` + tool cache).

The legacy MetaAgent/SubAgent tests that lived here were removed in the
Phase 5 cleanup. Replacement coverage for the v4 workflow system lives in
``tests/test_agents_v2.py`` (registry, steps, executor, YAML loader, PoC
workflows, CLI dispatch).
"""

import json
import unittest
from unittest.mock import MagicMock

from PyQt6.QtCore import QCoreApplication

# pyqtSignal needs a QApplication before any bus-based test runs.
_qapp = QCoreApplication.instance() or QCoreApplication([])

from src.core import state_bus as state_bus_mod
from src.core.agents.shared_context import SharedContext, ToolResultCache
from src.core.agents.sub_agents import CachingToolRegistry


def make_shared_context(abstract: str = "Test abstract über Bibliothekswissenschaft.") -> SharedContext:
    return SharedContext(
        abstract=abstract,
        initial_keywords=["Bibliothek", "Erschließung"],
        provider="test_provider",
        model="test_model",
        temperature=0.5,
    )


def make_mock_tool_registry() -> MagicMock:
    registry = MagicMock()
    registry.get_tool_names.return_value = ["search_gnd", "search_swb", "search_lobid"]
    registry.get_tool_schemas.return_value = []
    registry.execute.return_value = json.dumps({"results": []})
    return registry


class TestSharedContext(unittest.TestCase):
    def test_step_result_storage_and_retrieval(self):
        ctx = make_shared_context()
        ctx.set_step_result("extraction", {"keywords": ["Bibliothek"]}, quality=0.9)
        result = ctx.get_step_result("extraction")
        self.assertEqual(result, {"keywords": ["Bibliothek"]})
        self.assertAlmostEqual(ctx.quality_scores["extraction"], 0.9)

    def test_get_missing_step_returns_none(self):
        ctx = make_shared_context()
        self.assertIsNone(ctx.get_step_result("nonexistent_step"))

    def test_conversation_memory(self):
        ctx = make_shared_context()
        ctx.add_message("user", "Analysiere diesen Text.")
        ctx.add_message("assistant", "Hier sind die Keywords.")
        recent = ctx.get_recent_messages(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["role"], "assistant")

    def test_to_keyword_analysis_state(self):
        ctx = make_shared_context()
        ctx.working_title = "Autor_Thema_2024"
        ctx.extracted_keywords = ["Bibliothek", "Katalog"]
        ctx.selected_keywords = [{"gnd_id": "4006278-9", "title": "Bibliothek"}]
        ctx.gnd_entries = [
            {"title": "Bibliothek", "gnd_id": "4006278-9", "gnd_ids": ["4006278-9"], "ddc_codes": ["020"], "count": 5},
            {"title": "Katalog", "gnd_id": "4145769-0", "gnd_ids": ["4145769-0"], "ddc_codes": ["025.3"], "count": 3},
        ]
        ctx.dk_classifications = [{"code": "02", "title": "Bibliothekswesen", "confidence": 0.9}]
        ctx.keyword_chains = [
            {"chain": ["Bibliothek", "Katalog"], "reason": "Verwandte Begriffe"},
        ]
        ctx.rvk_classifications = [{"code": "AN", "title": "Bibliothekswesen", "confidence": 0.8}]

        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.original_abstract, ctx.abstract)
        self.assertEqual(state.working_title, "Autor_Thema_2024")
        self.assertEqual(state.dk_classifications, ["02"])
        self.assertTrue(len(state.search_results) > 0)
        self.assertIn("020", state.initial_gnd_classes)
        self.assertTrue(len(state.dk_search_results_flattened) > 0)
        self.assertIn("AN", state.rvk_provenance)
        self.assertIn("Schlagwortketten", state.final_llm_analysis.response_full_text)

    def test_to_keyword_analysis_state_carries_report_markdown(self):
        # title_list_search's render_report step writes extra.report_markdown;
        # to_keyword_analysis_state() must surface it so the GUI's
        # on_pipeline_completed can render it as an HTML table. - Claude Generated
        ctx = make_shared_context()
        ctx.extra = {"report_markdown": "| Title | Status |\n|---|---|\n| A | new |"}
        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.report_markdown, "| Title | Status |\n|---|---|\n| A | new |")

    def test_to_keyword_analysis_state_report_markdown_defaults_empty(self):
        ctx = make_shared_context()
        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.report_markdown, "")

    def test_get_summary_returns_correct_counts(self):
        ctx = make_shared_context()
        ctx.extracted_keywords = ["A", "B", "C"]
        ctx.gnd_entries = [{"gnd_id": "123"}]
        summary = ctx.get_summary()
        self.assertEqual(summary["extracted_keywords_count"], 3)
        self.assertEqual(summary["gnd_entries_count"], 1)


class TestToolResultCache(unittest.TestCase):
    def test_cache_miss_returns_none(self):
        cache = ToolResultCache()
        self.assertIsNone(cache.get("search_gnd", {"term": "Bibliothek"}))

    def test_cache_set_and_get(self):
        cache = ToolResultCache()
        cache.set("search_gnd", {"term": "Bibliothek"}, "cached_result")
        self.assertEqual(cache.get("search_gnd", {"term": "Bibliothek"}), "cached_result")

    def test_cache_stats_hit_rate(self):
        cache = ToolResultCache()
        cache.set("search_gnd", {"term": "Bibliothek"}, "result")
        cache.get("search_gnd", {"term": "Bibliothek"})
        cache.get("search_gnd", {"term": "Katalog"})
        stats = cache.get_stats()
        self.assertEqual(stats["total_hits"], 1)
        self.assertEqual(stats["total_misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_cache_clear(self):
        cache = ToolResultCache()
        cache.set("search_gnd", {"term": "test"}, "result")
        cache.clear()
        self.assertIsNone(cache.get("search_gnd", {"term": "test"}))
        self.assertEqual(cache.get_stats()["cache_size"], 0)


class TestCachingToolRegistry(unittest.TestCase):
    def test_second_call_hits_cache(self):
        cache = ToolResultCache()
        inner = make_mock_tool_registry()
        inner.execute.return_value = json.dumps({"results": [{"id": "1"}]})
        caching = CachingToolRegistry(inner, cache)

        result1 = caching.execute("search_gnd", {"term": "Bibliothek"})
        result2 = caching.execute("search_gnd", {"term": "Bibliothek"})

        self.assertEqual(result1, result2)
        inner.execute.assert_called_once()

    def test_different_args_are_separate_cache_entries(self):
        cache = ToolResultCache()
        inner = make_mock_tool_registry()
        inner.execute.side_effect = lambda t, a: json.dumps({"term": a.get("term")})
        caching = CachingToolRegistry(inner, cache)

        r1 = caching.execute("search_gnd", {"term": "Bibliothek"})
        r2 = caching.execute("search_gnd", {"term": "Katalog"})

        self.assertNotEqual(r1, r2)
        self.assertEqual(inner.execute.call_count, 2)

    def test_cache_disabled_always_executes(self):
        cache = ToolResultCache()
        inner = make_mock_tool_registry()
        inner.execute.return_value = "result"
        caching = CachingToolRegistry(inner, cache, cache_enabled=False)

        caching.execute("search_gnd", {"term": "test"})
        caching.execute("search_gnd", {"term": "test"})

        self.assertEqual(inner.execute.call_count, 2)


class TestCachingToolRegistryBusEmission(unittest.TestCase):
    """Phase B: deterministic tool calls emit ``tool.called`` + ``tool.result``."""

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()

    def tearDown(self):
        state_bus_mod.reset()

    def _setup(self, inner_return: str = '{"hits": 2}'):
        cache = ToolResultCache()
        inner = make_mock_tool_registry()
        inner.execute.return_value = inner_return
        return CachingToolRegistry(inner, cache)

    def test_emits_called_and_result_with_matching_id(self):
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        caching = self._setup()
        caching.execute("search_gnd", {"term": "Bibliothek"})

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["name"], "search_gnd")
        self.assertEqual(called[0]["arguments"], {"term": "Bibliothek"})
        self.assertTrue(called[0]["id"].startswith("tc_"))
        self.assertEqual(len(results), 1)
        # Result re-uses the called event's id.
        self.assertEqual(results[0]["id"], called[0]["id"])
        self.assertEqual(results[0]["cache_hit"], False)
        self.assertEqual(results[0]["status"], "ok")

    def test_cache_hit_emits_with_cache_hit_true(self):
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        caching = self._setup()
        caching.execute("search_gnd", {"term": "Bibliothek"})
        caching.execute("search_gnd", {"term": "Bibliothek"})

        self.assertEqual(len(called), 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["cache_hit"], False)
        self.assertEqual(results[1]["cache_hit"], True)
        # Different ids per call (Phase A: each execute gets a new id).
        self.assertNotEqual(results[0]["id"], results[1]["id"])

    def test_cache_disabled_path_still_emits(self):
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        cache = ToolResultCache()
        inner = make_mock_tool_registry()
        inner.execute.return_value = "{}"
        caching = CachingToolRegistry(inner, cache, cache_enabled=False)

        caching.execute("search_gnd", {"term": "x"})

        self.assertEqual(len(called), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["cache_hit"], False)

    def test_execute_error_emits_with_ok_status_legacy_compat(self):
        """P-B (documented limitation): legacy ``_execute_tool`` swallows
        exceptions and returns a JSON error payload, so the bus emission
        reports ``status='ok'`` (the result string was returned, even
        though its content is an error). A future polish item could
        re-raise from ``_execute_tool`` to surface the error status.
        """
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        cache = ToolResultCache()
        inner = make_mock_tool_registry()
        inner.execute.side_effect = RuntimeError("boom")
        caching = CachingToolRegistry(inner, cache)

        result = caching.execute("search_gnd", {"term": "x"})
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("boom", parsed["error"])

        # Both events fire.
        self.assertEqual(len(called), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], called[0]["id"])
        # Documented limitation: status stays 'ok' because _execute_tool
        # converts the exception to a JSON result string.
        self.assertEqual(results[0]["status"], "ok")

    def test_inner_registry_does_not_emit(self):
        """Regression guard: ToolRegistry.execute (raw) does not emit events."""
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        # The raw inner registry, not CachingToolRegistry.
        inner = make_mock_tool_registry()
        inner.execute.return_value = "{}"
        inner.execute("search_gnd", {"term": "x"})

        self.assertEqual(called, [])
        self.assertEqual(results, [])


class TestSharedContextSerialization(unittest.TestCase):
    def _make_populated_context(self) -> SharedContext:
        ctx = make_shared_context()
        ctx.working_title = "Autor_Thema_2024"
        ctx.extracted_keywords = ["Bibliothek", "Katalog", "GND"]
        ctx.gnd_entries = [{"gnd_id": "4006278-9", "title": "Bibliothek", "ddc_codes": ["020"]}]
        ctx.selected_keywords = [{"gnd_id": "4006278-9", "title": "Bibliothek"}]
        ctx.keyword_chains = [{"chain": ["Bibliothek", "Katalog"], "reason": "Verwandt"}]
        ctx.dk_classifications = [{"code": "02", "title": "Bibliothekswesen", "confidence": 0.9}]
        ctx.rvk_classifications = [{"code": "AN", "title": "Bibliothekswesen", "confidence": 0.8}]
        ctx.set_step_result("extraction", {"keywords": ["Bibliothek"]}, quality=0.8)
        return ctx

    def test_to_dict_contains_all_fields(self):
        ctx = self._make_populated_context()
        d = ctx.to_dict()
        self.assertEqual(d["abstract"], ctx.abstract)
        self.assertEqual(d["working_title"], "Autor_Thema_2024")
        self.assertEqual(d["extracted_keywords"], ["Bibliothek", "Katalog", "GND"])
        self.assertEqual(len(d["gnd_entries"]), 1)
        self.assertEqual(len(d["selected_keywords"]), 1)
        self.assertEqual(len(d["dk_classifications"]), 1)
        self.assertIn("extraction", d["step_results"])

    def test_from_dict_restores_state(self):
        ctx = self._make_populated_context()
        d = ctx.to_dict()
        restored = SharedContext.from_dict(d)

        self.assertEqual(restored.abstract, ctx.abstract)
        self.assertEqual(restored.working_title, "Autor_Thema_2024")
        self.assertEqual(restored.extracted_keywords, ["Bibliothek", "Katalog", "GND"])
        self.assertEqual(len(restored.gnd_entries), 1)
        self.assertEqual(len(restored.selected_keywords), 1)
        self.assertAlmostEqual(restored.quality_scores.get("extraction"), 0.8)

    def test_roundtrip_via_dict(self):
        ctx = self._make_populated_context()
        restored = SharedContext.from_dict(ctx.to_dict())
        self.assertEqual(ctx.to_dict(), restored.to_dict())

    def test_save_and_load_file(self):
        import tempfile
        import os
        ctx = self._make_populated_context()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            ctx.save_to_file(path)
            self.assertTrue(os.path.exists(path))
            loaded = SharedContext.load_from_file(path)
            self.assertEqual(loaded.working_title, "Autor_Thema_2024")
            self.assertEqual(loaded.extracted_keywords, ctx.extracted_keywords)
        finally:
            os.unlink(path)

    def test_from_dict_fresh_cache(self):
        ctx = self._make_populated_context()
        ctx.tool_result_cache.set("search_gnd", {"term": "test"}, "result")
        restored = SharedContext.from_dict(ctx.to_dict())
        self.assertIsNone(restored.tool_result_cache.get("search_gnd", {"term": "test"}))


if __name__ == "__main__":
    unittest.main()
