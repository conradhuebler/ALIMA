"""Tests for the rvk_lookup MCP tool + dk_search_agentic rvk_inline gating.

Claude Generated. RVK is no longer collected inline in dk_collect (no RVK-API
calls there); the classification LLM pulls it on demand via the rvk_lookup tool
(alima_v51 always, alima_v51_105 only for WiWi). These tests cover:

 1. rvk_lookup is registered and exposes a well-formed schema.
 2. rvk_lookup returns a ranked candidate list with formatted notations.
 3. rvk_lookup short-circuits on empty keywords (no executor work).
 4. rvk_lookup feeds dk_codes through _build_dk_semantic_profile.
 5. dk_search_agentic with rvk_inline=False disables all inline RVK work.
 6. dk_search_agentic default (no flag) keeps inline RVK enabled.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.mcp.tool_registry import ToolRegistry


def _mock_executor() -> MagicMock:
    """Executor whose RVK helpers return canned, deterministic data."""
    ex = MagicMock()
    ex._derive_rvk_anchor_keywords.return_value = ["Marketing (GND-ID: 123)"]
    ex.execute_notation_search.return_value = {
        "classifications": [{"dk": "QP 340", "classification_type": "RVK"}],
        "keyword_results": [],
        "statistics": {},
    }
    ex.prepare_notation_classification_context.return_value = {
        "results_with_titles": [{"dk": "QP 340", "classification_type": "RVK"}],
        "catalog_text": "...",
        "allowed_standard_rvk_map": {"QP 340": "RVK QP 340"},
        "allowed_nonstandard_rvk_map": {},
        "rvk_source_map": {},
        "selected_rvk_meta": {},
    }
    ex._build_dk_semantic_profile.return_value = "DK 650 | Management"
    ex._build_rvk_scoring_shortlist.return_value = [
        {
            "dk": "QP 340",
            "label": "Marketing",
            "ancestor_path": "Wirtschaftswissenschaften > Marketing",
            "rvk_validation_status": "standard",
            "source": "rvk_api",
            "count": 5,
            "_anchor_hit_count": 2,
            "_score": 88,
        }
    ]
    return ex


class TestRvkLookupRegistration(unittest.TestCase):
    def test_registered_with_schema(self):
        reg = ToolRegistry()
        reg.register_all_tools()
        self.assertIn("rvk_lookup", reg.get_tool_names())
        schema = reg.get_tool_schemas(["rvk_lookup"])[0]
        self.assertEqual(schema["name"], "rvk_lookup")
        self.assertIn("keywords", schema["parameters"]["properties"])
        self.assertEqual(schema["parameters"]["required"], ["keywords"])


class TestRvkLookupHandler(unittest.TestCase):
    def setUp(self):
        # config_manager set → handler skips the ConfigManager() import path
        self.reg = ToolRegistry(config_manager=MagicMock())

    def test_returns_ranked_candidates(self):
        ex = _mock_executor()
        with patch("src.utils.pipeline_utils.PipelineStepExecutor", return_value=ex):
            payload = json.loads(
                self.reg._handle_rvk_lookup(
                    keywords=["Marketing (GND-ID: 123)"],
                    abstract="Ein Werk über Marketing und Vertrieb.",
                )
            )
        self.assertEqual(payload["count"], 1)
        cand = payload["rvk"][0]
        self.assertEqual(cand["notation"], "RVK QP 340")
        self.assertEqual(cand["validation_status"], "standard")
        self.assertEqual(cand["source"], "rvk_api")
        self.assertEqual(cand["anchor_hits"], 2)
        self.assertEqual(cand["score"], 88)
        # rvk_enabled must be True for the tool's own catalog pass;
        # strict_gnd_validation must be False so plain LLM-supplied terms
        # (no "(GND-ID: …)") are not all filtered out.
        _, kwargs = ex.execute_notation_search.call_args
        self.assertTrue(kwargs["rvk_enabled"])
        self.assertFalse(kwargs["strict_gnd_validation"])
        _, prep_kwargs = ex.prepare_notation_classification_context.call_args
        self.assertTrue(prep_kwargs["include_rvk"])

    def test_empty_keywords_short_circuits(self):
        with patch("src.utils.pipeline_utils.PipelineStepExecutor") as exec_cls:
            payload = json.loads(self.reg._handle_rvk_lookup(keywords=[]))
        self.assertEqual(payload, {"rvk": [], "count": 0})
        exec_cls.assert_not_called()

    def test_passes_real_cache_manager_to_executor(self):
        # WP P3: the executor must receive a real cache_manager (was None) so the
        # RVK search/validate calls share the WP2 raw cache with the classic
        # pipeline + the rvk_search/rvk_validate agent tools. - Claude Generated
        ex = _mock_executor()
        km = MagicMock()
        self.reg._knowledge_manager = km
        with patch("src.utils.pipeline_utils.PipelineStepExecutor", return_value=ex) as exec_cls:
            self.reg._handle_rvk_lookup(keywords=["Marketing (GND-ID: 123)"], abstract="x")
        _, kwargs = exec_cls.call_args
        self.assertIs(kwargs["cache_manager"], km)

    def test_dk_codes_feed_semantic_profile(self):
        ex = _mock_executor()
        with patch("src.utils.pipeline_utils.PipelineStepExecutor", return_value=ex):
            self.reg._handle_rvk_lookup(
                keywords=["Marketing (GND-ID: 123)"],
                abstract="abstract",
                dk_codes=["DK 658.8"],
            )
        ex._build_dk_semantic_profile.assert_called_once()
        called_codes = ex._build_dk_semantic_profile.call_args[0][0]
        self.assertEqual(called_codes, ["DK 658.8"])
        # The DK profile must reach the scorer's abstract argument
        scoring_abstract = ex._build_rvk_scoring_shortlist.call_args[0][1]
        self.assertIn("DK-Profil", scoring_abstract)


class TestDkSearchAgenticRvkInline(unittest.TestCase):
    """dk_search_agentic gating of inline RVK via config.rvk_inline."""

    def _run(self, config):
        from src.core.agents import deterministic_functions as df

        ex = _mock_executor()
        ex.execute_notation_search.return_value = {
            "classifications": [{"dk": "650", "classification_type": "DK"}],
            "keyword_results": [],
            "statistics": {},
        }
        ex.prepare_notation_classification_context.return_value = {
            "results_with_titles": [{"dk": "650"}],
            "catalog_text": "DK 650",
            "allowed_standard_rvk_map": {},
            "allowed_nonstandard_rvk_map": {},
            "rvk_source_map": {},
            "selected_rvk_meta": {},
        }
        ctx = SimpleNamespace(
            abstract="Marketing-Werk",
            extra={"final_keywords": ["Marketing (GND-ID: 123)"]},
            dk_catalog_stats=None,
            dk_search_results=None,
        )
        with patch("src.utils.pipeline_utils.PipelineStepExecutor", return_value=ex), \
             patch("src.utils.config_manager.ConfigManager", return_value=MagicMock()):
            df.dk_search_agentic(context=ctx, config=config)
        return ex

    def test_rvk_inline_false_disables_rvk(self):
        ex = self._run({"rvk_inline": False})
        ex._derive_rvk_anchor_keywords.assert_not_called()
        _, dk_kwargs = ex.execute_notation_search.call_args
        self.assertFalse(dk_kwargs["rvk_enabled"])
        _, prep_kwargs = ex.prepare_notation_classification_context.call_args
        self.assertFalse(prep_kwargs["include_rvk"])

    def test_default_keeps_rvk_inline(self):
        ex = self._run({})
        ex._derive_rvk_anchor_keywords.assert_called_once()
        _, dk_kwargs = ex.execute_notation_search.call_args
        self.assertTrue(dk_kwargs["rvk_enabled"])
        _, prep_kwargs = ex.prepare_notation_classification_context.call_args
        self.assertTrue(prep_kwargs["include_rvk"])


class TestRealRvkGating(unittest.TestCase):
    """Exercise the REAL pipeline_utils gating (no executor mock, no network)."""

    def _executor(self):
        from src.utils.pipeline_utils import PipelineStepExecutor
        # config_manager=None → no SmartProviderSelector; pure data transforms.
        return PipelineStepExecutor(
            alima_manager=None, cache_manager=None, logger=None, config_manager=None
        )

    def _sample_results(self):
        return [
            {"dk": "650", "classification_type": "DK", "count": 3, "titles": ["BWL"]},
            {
                "dk": "QP 340", "classification_type": "RVK", "count": 2,
                "source": "rvk_api", "rvk_validation_status": "standard",
                "label": "Marketing", "ancestor_path": "Wirtschaft > Marketing",
                "matched_keywords": ["Marketing"],
            },
        ]

    def test_strip_rvk_helper(self):
        from src.utils.pipeline_utils import PipelineStepExecutor

        kw_results = [
            {"keyword": "Marketing", "classifications": [
                {"dk": "650", "classification_type": "DK"},
                {"dk": "QP 340", "classification_type": "RVK"},
            ]},
        ]
        out = PipelineStepExecutor._strip_rvk_from_keyword_results(kw_results)
        types = [c["classification_type"] for c in out[0]["classifications"]]
        self.assertEqual(types, ["DK"])

    def test_prepare_include_rvk_true_keeps_rvk(self):
        prep = self._executor().prepare_notation_classification_context(
            self._sample_results(), original_abstract="Marketing", include_rvk=True
        )
        self.assertEqual(len(prep["allowed_standard_rvk_map"]), 1)
        self.assertIn("WICHTIG FÜR RVK", prep["catalog_text"])
        types = {
            str(r.get("classification_type", "DK")).upper()
            for r in prep["results_with_titles"]
        }
        self.assertIn("RVK", types)

    def test_dk_semantic_profile_uses_dk_and_ddc(self):
        ex = self._executor()
        candidates = [
            {"dk": "330", "classification_type": "DDC", "titles": ["Wirtschaft"],
             "matched_keywords": ["Wirtschaftskrise"], "count": 4},
            {"dk": "316.42", "classification_type": "DK", "titles": ["Wandel"],
             "matched_keywords": ["Sozialer Wandel"], "count": 2},
            {"dk": "QP 340", "classification_type": "RVK", "titles": ["x"]},
        ]
        profile = ex._build_dk_semantic_profile(["DDC 330", "DK 316.42"], candidates)
        self.assertIn("DDC 330", profile)
        self.assertIn("DK 316.42", profile)
        # RVK must never appear in the DK/DDC semantic profile
        self.assertNotIn("QP 340", profile)

    def test_prepare_include_rvk_false_drops_rvk(self):
        prep = self._executor().prepare_notation_classification_context(
            self._sample_results(), original_abstract="Marketing", include_rvk=False
        )
        self.assertEqual(prep["allowed_standard_rvk_map"], {})
        self.assertNotIn("WICHTIG FÜR RVK", prep["catalog_text"])
        types = {
            str(r.get("classification_type", "DK")).upper()
            for r in prep["results_with_titles"]
        }
        self.assertNotIn("RVK", types)


class TestWorkflowYamlWiring(unittest.TestCase):
    """Both RVK-tool workflows must carry the identical non-prompt wiring."""

    def _load(self, name):
        import yaml
        with open(f"workflows/{name}", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return {s["id"]: s for s in data["steps"]}

    def test_v51_and_freiberg_wired_for_rvk_tool(self):
        for name in ("alima_v51.yaml", "alima_v51_105.yaml"):
            steps = self._load(name)
            with self.subTest(workflow=name):
                self.assertFalse(steps["dk_collect"]["config"]["rvk_inline"])
                cls = steps["classification"]
                self.assertEqual(cls["tools"], ["rvk_lookup"])
                self.assertGreater(cls["llm"]["max_iterations"], 1)
                self.assertIn("final_keywords", cls["inputs"])


if __name__ == "__main__":
    unittest.main()
