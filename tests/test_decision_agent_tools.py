"""Der Agent kann das Entscheidungsmodell benutzen. Claude Generated.

Die Werkzeuge ``decide`` und ``rank_by_relevance`` stehen im MCP-Katalog nur,
solange die Rolle gesetzt ist; der Chat (GUI, CLI, HTTP) bekommt sie über den
MCP-Adapter, Workflow-Schritte über das Preset ``decision``. Gegen den
Fake-Server aus ``test_decision_prefilter``.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from src.mcp.tool_registry import ToolRegistry
from src.ui.chat_tools.mcp_adapter import build_mcp_chat_tools
from src.utils.decision_tools import (
    MAX_DECIDE_STATE_CHARS,
    MAX_RANK_CANDIDATES,
    run_decide,
    run_rank,
)
from tests.test_decision_prefilter import _Fake, _ServerCase

NOUL = {"type": "noul", "instructions": "Does it concern cadmium?",
        "criteria": {"true": "Yes.", "false": "No."}}
CHOICE = {"type": "choice", "instructions": "Which area?",
          "criteria": {"soil": "Soil science.", "water": "Hydrology."}}


class RunDecideTest(_ServerCase):
    def test_passes_the_probabilities_through(self):
        out = run_decide(self.config(), "Cadmium im Boden",
                         {"on_topic": NOUL, "area": CHOICE})
        self.assertNotIn("error", out)
        self.assertEqual(out["model"], "d1")
        self.assertGreater(out["answers"]["on_topic"]["noul"], 0.5)
        self.assertEqual(out["answers"]["area"]["choice"], "soil")
        self.assertEqual(_Fake.calls[0]["n"], 2)

    def test_object_state_is_accepted(self):
        out = run_decide(self.config(), {"title": "Cadmium"}, {"q": NOUL})
        self.assertNotIn("error", out)

    def test_no_role_means_a_readable_error_and_no_call(self):
        cfg = self.config(decision_default_provider="", decision_default_model="")
        out = run_decide(cfg, "x", {"q": NOUL})
        self.assertIn("Entscheidungsmodell", out["error"])
        self.assertEqual(_Fake.calls, [])

    def test_bad_input_is_refused_before_any_call(self):
        cfg = self.config()
        cases = [
            ("x", {}),                                                   # keine Fragen
            ("", {"q": NOUL}),                                           # leerer State
            ("x" * (MAX_DECIDE_STATE_CHARS + 1), {"q": NOUL}),           # State zu lang
            ("x", {"bad name": NOUL}),                                   # Fragename
            ("x", {"q": {**NOUL, "type": "score"}}),                     # nicht unterstützter Typ
            ("x", {"q": {**NOUL, "criteria": {"yes": "a", "no": "b"}}}), # noul-Kriterien
            ("x", {"q": {**NOUL, "instructions": " "}}),                 # ohne Frage
            ("x", {f"q{i}": NOUL for i in range(65)}),                   # zu viele
        ]
        for state, qs in cases:
            self.assertIn("error", run_decide(cfg, state, qs), (state[:10], list(qs)[:2]))
        self.assertEqual(_Fake.calls, [])

    def test_a_host_error_comes_back_as_error_text(self):
        _Fake.status_override = 400
        self.assertIn("error", run_decide(self.config(), "x", {"q": NOUL}))


class RunRankTest(_ServerCase):
    CANDS = [{"keyword": ("Cadmium %d" % i) if i % 10 == 0 else f"Boden {i}",
              "gnd_id": str(900 + i), "count": i} for i in range(120)]

    def test_returns_the_best_first_with_scores_and_extra_fields(self):
        out = run_rank(self.config(), "Cadmium im Boden", self.CANDS, 5)
        self.assertNotIn("error", out)
        self.assertEqual(len(out["ranked"]), 5)
        self.assertTrue(all(r["keyword"].startswith("Cadmium") for r in out["ranked"]))
        self.assertTrue(all(r["score"] > 0.5 and r["gnd_id"] for r in out["ranked"]))
        self.assertEqual((out["n_in"], out["n_out"]), (120, 5))

    def test_strings_work_and_duplicates_count_once(self):
        out = run_rank(self.config(), "t", ["Cadmium", "Cadmium", "Boden"], 10)
        self.assertEqual([r["keyword"] for r in out["ranked"]], ["Cadmium", "Boden"])
        self.assertEqual(out["n_in"], 2)

    def test_top_k_is_clamped(self):
        self.assertEqual(len(run_rank(self.config(), "t", self.CANDS, 0)["ranked"]), 1)
        self.assertEqual(len(run_rank(self.config(), "t", self.CANDS, "x")["ranked"]), 30)
        self.assertEqual(len(run_rank(self.config(), "t", self.CANDS, 9999)["ranked"]), 120)

    def test_refusals(self):
        cfg = self.config()
        self.assertIn("error", run_rank(cfg, "", ["a"]))
        self.assertIn("error", run_rank(cfg, "t", []))
        self.assertIn("error", run_rank(cfg, "t", ["a"] * (MAX_RANK_CANDIDATES + 1)))
        self.assertIn("error", run_rank(
            self.config(decision_default_provider="", decision_default_model=""), "t", ["a"]))
        self.assertEqual(_Fake.calls, [])


class RegistryTest(_ServerCase):
    @staticmethod
    def alima(unified):
        # Der Registry-Bau liest weitere Teile der Konfiguration (Plugins); eine
        # echte AlimaConfig statt eines Stubs, wie in den anderen Registry-Tests.
        from src.utils.config_models import AlimaConfig

        ac = AlimaConfig()
        ac.unified_config = unified
        return ac

    def registry(self, cfg):
        cm = MagicMock()
        cm.load_config.return_value = self.alima(cfg)
        return ToolRegistry(config_manager=cm), cm

    def test_tools_exist_only_while_the_role_is_set(self):
        reg, cm = self.registry(self.config())
        reg.register_all_tools()
        self.assertIn("decide", reg.get_tool_names())
        self.assertIn("rank_by_relevance", reg.get_tool_names())

        cm.load_config.return_value = self.alima(
            self.config(decision_default_provider="", decision_default_model=""))
        reg.refresh()
        self.assertNotIn("decide", reg.get_tool_names())
        self.assertNotIn("rank_by_relevance", reg.get_tool_names())

    def test_a_registry_without_a_readable_config_offers_neither(self):
        reg = ToolRegistry(config_manager=MagicMock(load_config=MagicMock(side_effect=OSError)))
        reg.register_all_tools()
        self.assertNotIn("decide", reg.get_tool_names())

    def test_execute_returns_json(self):
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        out = json.loads(reg.execute("rank_by_relevance", {
            "text": "Cadmium im Boden", "candidates": ["Cadmium", "Boden"], "top_k": 1}))
        self.assertEqual(out["ranked"][0]["keyword"], "Cadmium")
        out = json.loads(reg.execute("decide", {"state": "Cadmium", "questions": {"q": NOUL}}))
        self.assertGreater(out["answers"]["q"]["noul"], 0.5)

    def test_the_chat_gets_both_tools_through_the_mcp_adapter(self):
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        names = {t.name for t in build_mcp_chat_tools(no_cache_writes=True, mcp_registry=reg)}
        self.assertTrue({"decide", "rank_by_relevance"} <= names)

    def _pool(self, n=120):
        return json.dumps({"pool": [
            {"title": ("Cadmium %d" % i) if i % 10 == 0 else f"Boden {i}",
             "gnd_ids": [str(700 + i)], "count": i, "sources": ["lobid"]}
            for i in range(n)], "sources": ["lobid"], "missing": {}, "terms_map": {}})

    def test_rank_takes_the_pool_from_the_search_terms(self):
        """Der Agent tippt den Pool nicht ab: die Suchbegriffe genügen."""
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        seen = {}

        def fake_aggregate(terms, **kw):
            seen["terms"] = terms
            return self._pool()

        reg._handle_aggregate_gnd_results = fake_aggregate
        out = json.loads(reg.execute("rank_by_relevance", {
            "text": "Cadmium im Boden", "from_search_terms": ["Cadmium", "Itai-Itai"], "top_k": 5}))
        self.assertEqual(seen["terms"], ["Cadmium", "Itai-Itai"])
        self.assertNotIn("error", out)
        self.assertEqual(out["n_in"], 120)
        self.assertTrue(all(r["keyword"].startswith("Cadmium") for r in out["ranked"]))
        self.assertTrue(all(r["gnd_id"] and "count" in r for r in out["ranked"]))

    def test_rank_from_search_terms_explains_an_empty_or_failed_pool(self):
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        reg._handle_aggregate_gnd_results = lambda terms, **kw: json.dumps(
            {"pool": [], "missing": {"lobid": ["Cadmium"]}})
        out = json.loads(reg.execute("rank_by_relevance", {"text": "t", "from_search_terms": ["Cadmium"]}))
        self.assertIn("search_*", out["error"])
        reg._handle_aggregate_gnd_results = lambda terms, **kw: json.dumps({"pool": [], "error": "keine Quelle"})
        out = json.loads(reg.execute("rank_by_relevance", {"text": "t", "from_search_terms": ["Cadmium"]}))
        self.assertEqual(out["error"], "keine Quelle")
        self.assertEqual(_Fake.calls, [])

    def test_unlabeled_pool_entries_are_not_scored_and_the_count_is_reported(self):
        """Einträge, deren Titel nur die GND-ID ist, bekamen mittlere Scores wie Rauschen."""
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        pool = json.loads(self._pool(30))
        pool["pool"] += [{"title": "4014613-3", "gnd_ids": ["4014613-3"], "count": 1},
                         {"title": "35439-9", "gnd_ids": ["35439-9"], "count": 1}]
        pool["pool"][0]["display_count"] = 42
        reg._handle_aggregate_gnd_results = lambda terms, **kw: json.dumps(pool)
        out = json.loads(reg.execute("rank_by_relevance", {
            "text": "Cadmium", "from_search_terms": ["x"], "top_k": 40}))
        self.assertEqual(out["skipped_unlabeled"], 2)
        self.assertEqual(out["n_in"], 30)
        self.assertNotIn("4014613-3", [r["keyword"] for r in out["ranked"]])
        # Katalog-Häufigkeit darf nicht im Ergebnis stehen: kopiert der Agent sie als
        # ``count`` in select_from_gnd_pool, führt die Auswahl mit "Krankheit" und "Medizin".
        self.assertTrue(all("display_count" not in r for r in out["ranked"]))

    def test_a_pool_of_only_ids_is_an_error(self):
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        reg._handle_aggregate_gnd_results = lambda terms, **kw: json.dumps(
            {"pool": [{"title": "4014613-3", "gnd_ids": ["4014613-3"]}]})
        out = json.loads(reg.execute("rank_by_relevance", {"text": "t", "from_search_terms": ["x"]}))
        self.assertIn("nur GND-IDs", out["error"])

    def test_the_result_reports_the_input_tokens_the_host_counted(self):
        """Bei einem bezahlten Anbieter ist das die Zahl, auf die es ankommt."""
        out = run_rank(self.config(), "Cadmium", [f"Begriff {i}" for i in range(70)], 5)
        self.assertEqual(out["input_tokens"], 100 * 70)

    def test_select_from_gnd_pool_gets_an_llm_service_when_none_was_injected(self):
        """GUI-Chat und Headless-Runner bauen ihr Registry ohne LlmService; das Werkzeug
        meldete dort immer "requires LLM service"."""
        from unittest.mock import MagicMock as MM

        llm = MM()
        llm.alima_config.unified_config.resolve_default_provider_model.return_value = ("p", "m")
        llm.generate_response.return_value = '{"keywords": [{"keyword": "Cadmium", "gnd_id": "1"}]}'
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        self.assertIsNone(reg._llm_service)
        reg._get_llm_service = lambda: llm
        out = json.loads(reg.execute("select_from_gnd_pool", {
            "abstract": "Cadmium", "gnd_entries": [{"keyword": "Cadmium", "gnd_id": "1"}]}))
        self.assertNotIn("requires LLM service", json.dumps(out))
        self.assertTrue(llm.generate_response.called)
        # the result names the model that selected (the settings default, not the toolbar)
        self.assertEqual((out["provider"], out["model"]), ("p", "m"))
        self.assertEqual(llm.generate_response.call_args.kwargs["model"], "m")

    def test_the_registry_builders_hand_the_llm_service_on(self):
        """Quelltext-Wächter für die zwei Stellen, die ein Registry selbst bauen."""
        import pathlib

        for path in ("src/ui/pipeline_chat_panel.py", "src/core/headless_agent.py"):
            src = pathlib.Path(path).read_text(encoding="utf-8")
            self.assertIn("ToolRegistry(llm_service=llm_service)", src, path)

    def test_without_candidates_or_terms_the_error_names_both_ways(self):
        reg, _ = self.registry(self.config())
        reg.register_all_tools()
        out = json.loads(reg.execute("rank_by_relevance", {"text": "t"}))
        self.assertIn("from_search_terms", out["error"])

    def test_get_workflow_tells_the_agent_that_the_prefilter_is_active(self):
        """Der Agent hielt die Pipeline für nicht anpassbar und baute sie von Hand nach."""
        reg, cm = self.registry(self.config())
        reg.register_all_tools()
        steps = {s["id"]: s for s in json.loads(reg.execute("get_workflow", {"name": "alima_classic"}))["steps"]}
        self.assertEqual(steps["selection_chunks"]["decision_prefilter"], {"top_k": 80, "active": True})
        self.assertNotIn("decision_prefilter", steps["extraction"])

        cm.load_config.return_value = self.alima(
            self.config(decision_default_provider="", decision_default_model=""))
        steps = {s["id"]: s for s in json.loads(reg.execute("get_workflow", {"name": "alima_classic"}))["steps"]}
        self.assertFalse(steps["selection_chunks"]["decision_prefilter"]["active"])

    def test_the_tool_descriptions_say_the_pipeline_uses_it_by_itself(self):
        from src.mcp import tool_schemas
        from src.ui.chat_tools.pipeline import RunPipelineTool

        self.assertIn("decision model", tool_schemas.EXECUTE_WORKFLOW.description)
        self.assertIn("decision model", RunPipelineTool.description)
        self.assertIn("from_search_terms", tool_schemas.RANK_BY_RELEVANCE.description)
        self.assertEqual(tool_schemas.RANK_BY_RELEVANCE.parameters["required"], ["text"])

    def test_the_decision_preset_names_exactly_the_two_tools(self):
        reg, _ = self.registry(self.config())
        self.assertEqual(reg.get_preset("decision"), ["decide", "rank_by_relevance"])

    def test_the_tool_schemas_are_valid_json_schema_objects(self):
        from src.mcp import tool_schemas

        for td in (tool_schemas.DECIDE, tool_schemas.RANK_BY_RELEVANCE):
            json.dumps(td.to_schema())
            self.assertEqual(td.parameters["type"], "object")
            for req in td.parameters["required"]:
                self.assertIn(req, td.parameters["properties"])


if __name__ == "__main__":
    unittest.main()
