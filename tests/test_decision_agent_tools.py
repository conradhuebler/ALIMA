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
