"""Tests for the ALIMA Agent System (MetaAgent + SubAgents).

Test strategy:
- Mock AgentLoop.run() to return pre-defined AgentResult objects
- Mock ToolRegistry.execute() for tool-based agents
- No Qt, no real DB, no real LLM calls

See CLAUDE.md test maintenance rules.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.core.agents.shared_context import SharedContext, ToolResultCache
from src.core.agents.sub_agents import (
    KeywordExtractionAgent,
    SearchAgent,
    KeywordSelectionAgent,
    ClassificationAgent,
    CachingToolRegistry,
)
from src.core.data_models import AgentResult


def make_agent_result(content: str) -> AgentResult:
    """Helper: create a mock AgentResult."""
    return AgentResult(content=content, tool_log=[], iterations=1)


def make_shared_context(abstract: str = "Test abstract über Bibliothekswissenschaft.") -> SharedContext:
    """Helper: create a SharedContext without any DB or Qt dependencies."""
    return SharedContext(
        abstract=abstract,
        initial_keywords=["Bibliothek", "Erschließung"],
        provider="test_provider",
        model="test_model",
        temperature=0.5,
    )


def make_mock_tool_registry() -> MagicMock:
    """Helper: create a mock ToolRegistry."""
    registry = MagicMock()
    registry.get_tool_names.return_value = ["search_gnd", "search_swb", "search_lobid"]
    registry.get_tool_schemas.return_value = []
    registry.execute.return_value = json.dumps({"results": []})
    return registry


class TestSharedContext(unittest.TestCase):
    """Tests for SharedContext data management."""

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
        ctx.dk_classifications = [{"code": "02", "title": "Bibliothekswesen", "confidence": 0.9}]

        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.original_abstract, ctx.abstract)
        self.assertEqual(state.working_title, "Autor_Thema_2024")
        self.assertEqual(state.dk_classifications, ["02"])

    def test_get_summary_returns_correct_counts(self):
        ctx = make_shared_context()
        ctx.extracted_keywords = ["A", "B", "C"]
        ctx.gnd_entries = [{"gnd_id": "123"}]
        summary = ctx.get_summary()
        self.assertEqual(summary["extracted_keywords_count"], 3)
        self.assertEqual(summary["gnd_entries_count"], 1)


class TestToolResultCache(unittest.TestCase):
    """Tests for ToolResultCache caching and statistics."""

    def test_cache_miss_returns_none(self):
        cache = ToolResultCache()
        result = cache.get("search_gnd", {"term": "Bibliothek"})
        self.assertIsNone(result)

    def test_cache_set_and_get(self):
        cache = ToolResultCache()
        cache.set("search_gnd", {"term": "Bibliothek"}, "cached_result")
        result = cache.get("search_gnd", {"term": "Bibliothek"})
        self.assertEqual(result, "cached_result")

    def test_cache_stats_hit_rate(self):
        cache = ToolResultCache()
        cache.set("search_gnd", {"term": "Bibliothek"}, "result")
        cache.get("search_gnd", {"term": "Bibliothek"})  # hit
        cache.get("search_gnd", {"term": "Katalog"})     # miss
        stats = cache.get_stats()
        self.assertEqual(stats["total_hits"], 1)
        self.assertEqual(stats["total_misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_cache_clear(self):
        cache = ToolResultCache()
        cache.set("search_gnd", {"term": "test"}, "result")
        cache.clear()
        self.assertIsNone(cache.get("search_gnd", {"term": "test"}))
        stats = cache.get_stats()
        self.assertEqual(stats["cache_size"], 0)


class TestCachingToolRegistry(unittest.TestCase):
    """Tests for CachingToolRegistry deduplication."""

    def test_second_call_hits_cache(self):
        cache = ToolResultCache()
        inner = make_mock_tool_registry()
        inner.execute.return_value = json.dumps({"results": [{"id": "1"}]})
        caching = CachingToolRegistry(inner, cache)

        result1 = caching.execute("search_gnd", {"term": "Bibliothek"})
        result2 = caching.execute("search_gnd", {"term": "Bibliothek"})

        self.assertEqual(result1, result2)
        inner.execute.assert_called_once()  # Only one real execution

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


class TestKeywordExtractionAgent(unittest.TestCase):
    """Tests for KeywordExtractionAgent (pure LLM, no tools)."""

    def _run_agent_with_response(self, response_content: str) -> dict:
        """Helper: run extraction agent with mocked LLM response."""
        ctx = make_shared_context()
        registry = make_mock_tool_registry()
        agent = KeywordExtractionAgent(
            shared_context=ctx,
            llm_service=MagicMock(),
            tool_registry=registry,
        )
        mock_result = make_agent_result(response_content)
        with patch("src.core.agents.sub_agents.base_sub_agent.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            result = agent.execute()
        return result, ctx

    def test_successful_json_response(self):
        response = json.dumps({
            "title": "Maier_Bibliothek_2024",
            "keywords": ["Bibliothekswissenschaft", "Katalogisierung", "GND"],
        })
        result, ctx = self._run_agent_with_response(response)
        self.assertTrue(result.success)
        self.assertIn("Bibliothekswissenschaft", ctx.extracted_keywords)
        self.assertEqual(ctx.working_title, "Maier_Bibliothek_2024")

    def test_malformed_response_graceful_fallback(self):
        response = "Hier sind die Keywords: Bibliothek, Erschließung. Kein JSON."
        result, ctx = self._run_agent_with_response(response)
        # Should still succeed (graceful fallback)
        self.assertTrue(result.success)
        # raw_output fallback key
        self.assertIn("raw_output", result.data)

    def test_json_in_markdown_block(self):
        response = """```json
{"title": "Test_Titel", "keywords": ["Schlagwort1", "Schlagwort2"]}
```"""
        result, ctx = self._run_agent_with_response(response)
        self.assertTrue(result.success)
        self.assertIn("Schlagwort1", ctx.extracted_keywords)

    def test_no_tools_used(self):
        agent = KeywordExtractionAgent(
            shared_context=make_shared_context(),
            llm_service=MagicMock(),
            tool_registry=make_mock_tool_registry(),
        )
        self.assertEqual(agent.get_available_tools(), [])

    def test_context_updated_after_execution(self):
        response = json.dumps({
            "title": "Autor_Katalog",
            "keywords": ["Erschließung", "Metadaten"],
        })
        result, ctx = self._run_agent_with_response(response)
        # Context should be populated
        self.assertTrue(len(ctx.extracted_keywords) > 0)
        self.assertTrue(len(ctx.working_title) > 0)


class TestSearchAgent(unittest.TestCase):
    """Tests for SearchAgent (GND/SWB search with tools)."""

    def _run_search_agent(self, response_content: str) -> tuple:
        ctx = make_shared_context()
        ctx.extracted_keywords = ["Bibliothek", "Katalog"]
        registry = make_mock_tool_registry()
        agent = SearchAgent(
            shared_context=ctx,
            llm_service=MagicMock(),
            tool_registry=registry,
        )
        mock_result = make_agent_result(response_content)
        with patch("src.core.agents.sub_agents.base_sub_agent.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            result = agent.execute()
        return result, ctx

    def test_search_agent_has_tools(self):
        agent = SearchAgent(
            shared_context=make_shared_context(),
            llm_service=MagicMock(),
            tool_registry=make_mock_tool_registry(),
        )
        tools = agent.get_available_tools()
        self.assertTrue(len(tools) > 0)
        self.assertIn("search_gnd", tools)

    def test_gnd_entries_stored_in_context(self):
        # LLMs typically wrap JSON in markdown blocks; _extract_json handles this
        gnd_data = {
            "gnd_entries": [
                {"gnd_id": "4006278-9", "title": "Bibliothek", "ddc_codes": ["020"]},
                {"gnd_id": "4030351-5", "title": "Katalog", "ddc_codes": ["025"]},
            ],
            "search_terms_used": ["Bibliothek"],
            "coverage_assessment": "gut",
        }
        response = f"```json\n{json.dumps(gnd_data)}\n```"
        result, ctx = self._run_search_agent(response)
        self.assertTrue(result.success)
        self.assertTrue(len(ctx.gnd_entries) >= 2)

    def test_malformed_response_still_succeeds(self):
        result, ctx = self._run_search_agent("Keine GND-Einträge gefunden.")
        self.assertTrue(result.success)


class TestMetaAgentIntegration(unittest.TestCase):
    """Integration tests for MetaAgent orchestration."""

    def _make_meta_agent(self):
        from src.core.agents.meta_agent import MetaAgent
        llm_service = MagicMock()
        return MetaAgent(llm_service=llm_service, config_manager=None)

    def test_get_step_order_returns_strings(self):
        agent = self._make_meta_agent()
        # Load default steps
        agent._get_default_steps()
        agent.workflow_steps = agent._get_default_steps()
        order = agent.get_step_order()
        self.assertIsInstance(order, list)
        self.assertEqual(order, ["extraction", "search", "selection", "classification"])

    def test_assess_quality_empty_context(self):
        from src.core.agents.meta_agent import MetaAgent
        agent = MetaAgent(llm_service=MagicMock(), config_manager=None)
        ctx = make_shared_context()
        score = agent.assess_quality(ctx)
        self.assertEqual(score, 0.0)

    def test_assess_quality_with_results(self):
        from src.core.agents.meta_agent import MetaAgent
        agent = MetaAgent(llm_service=MagicMock(), config_manager=None)
        ctx = make_shared_context()
        ctx.extracted_keywords = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        ctx.selected_keywords = [{"gnd_id": "1", "title": "A"}, {"gnd_id": "2", "title": "B"},
                                  {"gnd_id": "3", "title": "C"}, {"gnd_id": "4", "title": "D"},
                                  {"gnd_id": "5", "title": "E"}]
        score = agent.assess_quality(ctx)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_full_pipeline_with_mocked_agents(self):
        """Integration test: MetaAgent runs all 4 stages with mocked SubAgent execution."""
        from src.core.agents.meta_agent import MetaAgent, MetaAgentConfig

        extraction_result = AgentResult(
            content=json.dumps({"title": "Test_2024", "keywords": ["Bibliothek"]}),
            tool_log=[], iterations=1
        )
        search_result = AgentResult(
            content=json.dumps({
                "gnd_entries": [{"gnd_id": "4006278-9", "title": "Bibliothek"}],
                "search_terms_used": ["Bibliothek"],
            }),
            tool_log=[], iterations=1
        )
        selection_result = AgentResult(
            content=json.dumps({
                "selected_keywords": [{"gnd_id": "4006278-9", "title": "Bibliothek"}],
                "missing_concepts": [],
            }),
            tool_log=[], iterations=1
        )
        classification_result = AgentResult(
            content=json.dumps({
                "dk_classifications": [{"code": "02", "title": "Bibliothekswesen", "confidence": 0.9}],
                "rvk_classifications": [],
                "reasoning": "Hauptthema ist Bibliothekswesen.",
            }),
            tool_log=[], iterations=1
        )

        results_cycle = iter([extraction_result, search_result, selection_result, classification_result])

        with patch("src.core.agents.sub_agents.base_sub_agent.AgentLoop") as MockLoop:
            MockLoop.return_value.run.side_effect = lambda **kwargs: next(results_cycle)
            with patch.object(MetaAgent, "__init__", lambda self, llm_service, config_manager, stream_callback=None: None):
                agent = MetaAgent.__new__(MetaAgent)
                agent.llm_service = MagicMock()
                agent.config_manager = None
                agent.stream_callback = None
                agent.logger = __import__("logging").getLogger("test")
                agent.tool_registry = MagicMock()
                agent.tool_registry.register_all_tools = MagicMock()
                agent.workflow_steps = []

            agent.workflow_steps = agent._get_default_steps()

            config = MetaAgentConfig(provider="ollama", model="mistral", enable_classification=True)
            state = agent.execute(abstract="Test abstract.", config=config)

        self.assertIsNotNone(state)
        self.assertEqual(state.original_abstract, "Test abstract.")


class TestSharedContextSerialization(unittest.TestCase):
    """Tests for SharedContext JSON serialization (warm-start support)."""

    def _make_populated_context(self) -> SharedContext:
        ctx = make_shared_context()
        ctx.working_title = "Autor_Thema_2024"
        ctx.extracted_keywords = ["Bibliothek", "Katalog", "GND"]
        ctx.gnd_entries = [{"gnd_id": "4006278-9", "title": "Bibliothek", "ddc_codes": ["020"]}]
        ctx.selected_keywords = [{"gnd_id": "4006278-9", "title": "Bibliothek"}]
        ctx.dk_classifications = [{"code": "02", "title": "Bibliothekswesen", "confidence": 0.9}]
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
        import tempfile, os
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
        # Cache is NOT persisted — fresh cache on restore
        self.assertIsNone(restored.tool_result_cache.get("search_gnd", {"term": "test"}))


class TestMetaAgentSingleStep(unittest.TestCase):
    """Tests for MetaAgent single-step execution and dependency validation."""

    def _make_meta_agent(self):
        from src.core.agents.meta_agent import MetaAgent
        agent = MetaAgent.__new__(MetaAgent)
        agent.llm_service = MagicMock()
        agent.config_manager = None
        agent.stream_callback = None
        agent.logger = __import__("logging").getLogger("test")
        agent.tool_registry = MagicMock()
        agent.tool_registry.register_all_tools = MagicMock()
        agent.workflow_steps = agent._get_default_steps()
        return agent

    def test_validate_dependencies_extraction_needs_nothing(self):
        agent = self._make_meta_agent()
        ctx = make_shared_context()
        # Should not raise
        agent._validate_dependencies("extraction", ctx)

    def test_validate_dependencies_search_needs_keywords(self):
        agent = self._make_meta_agent()
        ctx = make_shared_context()
        ctx.extracted_keywords = []  # Missing
        with self.assertRaises(ValueError) as cm:
            agent._validate_dependencies("search", ctx)
        self.assertIn("extracted_keywords", str(cm.exception))

    def test_validate_dependencies_search_ok_with_keywords(self):
        agent = self._make_meta_agent()
        ctx = make_shared_context()
        ctx.extracted_keywords = ["Bibliothek"]
        agent._validate_dependencies("search", ctx)  # Should not raise

    def test_validate_dependencies_selection_needs_gnd_entries(self):
        agent = self._make_meta_agent()
        ctx = make_shared_context()
        ctx.extracted_keywords = ["Bibliothek"]
        ctx.gnd_entries = []  # Missing
        with self.assertRaises(ValueError) as cm:
            agent._validate_dependencies("selection", ctx)
        self.assertIn("gnd_entries", str(cm.exception))

    def test_validate_dependencies_classification_needs_selected_keywords(self):
        agent = self._make_meta_agent()
        ctx = make_shared_context()
        ctx.selected_keywords = []  # Missing
        with self.assertRaises(ValueError) as cm:
            agent._validate_dependencies("classification", ctx)
        self.assertIn("selected_keywords", str(cm.exception))

    def test_execute_unknown_step_id_raises(self):
        from src.core.agents.meta_agent import MetaAgent, MetaAgentConfig
        agent = self._make_meta_agent()
        ctx = make_shared_context()
        with self.assertRaises(ValueError) as cm:
            agent.execute(abstract="test", step_id="nonexistent", input_context=ctx)
        self.assertIn("nonexistent", str(cm.exception))

    def test_execute_single_step_uses_input_context(self):
        """Single-step run with warm-start context updates context in place."""
        from src.core.agents.meta_agent import MetaAgent, MetaAgentConfig

        ctx = make_shared_context()
        ctx.extracted_keywords = ["Bibliothek", "Katalog"]
        ctx.gnd_entries = [{"gnd_id": "4006278-9", "title": "Bibliothek"}]

        response = json.dumps({
            "selected_keywords": [{"gnd_id": "4006278-9", "title": "Bibliothek"}],
            "missing_concepts": [],
        })
        mock_result = make_agent_result(f"```json\n{response}\n```")

        agent = self._make_meta_agent()
        with patch("src.core.agents.sub_agents.base_sub_agent.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            state = agent.execute(
                abstract=ctx.abstract,
                step_id="selection",
                input_context=ctx,
                config=MetaAgentConfig(provider="test", model="test"),
            )

        self.assertIsNotNone(state)
        # Context was updated by the selection agent
        self.assertTrue(len(ctx.selected_keywords) > 0)

    def test_input_context_provider_overridden_by_config(self):
        """Provider/model from MetaAgentConfig takes precedence over saved context."""
        from src.core.agents.meta_agent import MetaAgent, MetaAgentConfig

        ctx = make_shared_context()
        ctx.provider = "old_provider"
        ctx.model = "old_model"

        agent = self._make_meta_agent()
        # Just test that execute() applies the new config (no LLM call needed)
        # We test by checking the context is updated before any step runs
        config = MetaAgentConfig(provider="new_provider", model="new_model")

        # Patch _execute_step to avoid LLM calls, just check context state
        with patch.object(agent, "_execute_step") as mock_step:
            from src.core.agents.meta_agent import PipelineStepResult
            mock_step.return_value = PipelineStepResult(
                step_name="extraction", agent_name="test",
                success=True, duration_seconds=0.1, data={}
            )
            agent.execute(abstract="test", step_id="extraction",
                          input_context=ctx, config=config)

        self.assertEqual(ctx.provider, "new_provider")
        self.assertEqual(ctx.model, "new_model")


if __name__ == "__main__":
    unittest.main()
