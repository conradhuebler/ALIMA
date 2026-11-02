"""Der Vorfilter an seinen zwei Einhängepunkten. Claude Generated.

Agentisch: ``LLMAgentStep._run_chunked`` (Opt-in im Workflow *und* Rolle
gesetzt). Klassisch: ``PipelineStepExecutor.execute_final_keyword_analysis``
(Rolle gesetzt, nur wenn ohnehin gechunkt würde). Der Entscheidungs-Host ist der
Fake-Server aus ``test_decision_prefilter``.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.llm_agent_step import LLMAgentStep
from src.core.agents.workflow_loader import StepConfig
from src.core.agent_loop import AgentResult
from tests.test_decision_prefilter import _Fake, _ServerCase

POOL = [
    {"title": ("Cadmium %d" % i) if i % 10 == 0 else f"Boden {i}",
     "gnd_id": str(1000 + i), "count": 300 - i}
    for i in range(300)
]


def _step(prefilter) -> LLMAgentStep:
    chunking = {"enabled": True, "chunk_field": "keywords", "chunk_size": 500,
                "chunk_fields": ["title", "gnd_id"], "merge_key": "keywords",
                "dedup_field": "keyword"}
    if prefilter is not None:
        chunking["decision_prefilter"] = prefilter
    raw = {
        "system_prompt": "sys", "user_prompt": "{abstract}\n{keywords}",
        "inputs": {"abstract": "${abstract}", "keywords": "${gnd_entries}"},
        "outputs": {}, "llm": {"max_iterations": 1}, "chunking": chunking,
    }
    cfg = StepConfig(id="selection_chunks", type="llm_agent",
                     inputs=raw["inputs"], outputs={}, raw=raw)
    return LLMAgentStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())


class AgenticHookTest(_ServerCase):
    def run_step(self, prefilter, cfg):
        seen: list = []

        def run(**kw):
            seen.append(kw["user_prompt"])
            return AgentResult(content='{"keywords": []}', tool_log=[], iterations=1)

        loop = MagicMock()
        loop.run.side_effect = run
        ctx = SharedContext(abstract="Schwermetall im Boden", provider="p", model="m")
        step = _step(prefilter)
        fake_cm = MagicMock()
        fake_cm.return_value.get_unified_config.return_value = cfg
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=loop), \
                patch("src.utils.config_manager.ConfigManager", fake_cm):
            out = step._run_chunked(step.config.raw, {"abstract": ctx.abstract, "keywords": POOL},
                                    step.config.raw["chunking"], ctx)
        return seen, ctx, out

    def test_enabled_and_role_set_cuts_the_pool_before_chunking(self):
        seen, ctx, _ = self.run_step({"enabled": True, "top_k": 30}, self.config())
        self.assertEqual(len(seen), 1)                       # ein Chunk statt 300 Einträge
        self.assertEqual(seen[0].count("Cadmium"), 30)
        self.assertNotIn('"title": "Boden', seen[0])
        rec = ctx.extra["decision_prefilter"]
        self.assertEqual((rec["n_in"], rec["n_out"], rec["model"]), (300, 30, "d1"))
        self.assertEqual(len(rec["scores"]), 300)

    def test_the_record_survives_serialisation_of_the_context(self):
        _, ctx, _ = self.run_step({"enabled": True, "top_k": 30}, self.config())
        again = json.loads(json.dumps(ctx.to_dict()))
        self.assertEqual(again["extra"]["decision_prefilter"]["n_out"], 30)

    def test_role_off_changes_nothing(self):
        cfg = self.config(decision_default_provider="", decision_default_model="")
        seen, ctx, _ = self.run_step({"enabled": True, "top_k": 30}, cfg)
        self.assertIn('"title": "Boden', seen[0])
        self.assertNotIn("decision_prefilter", ctx.extra)
        self.assertEqual(_Fake.calls, [])

    def test_workflow_without_the_block_changes_nothing(self):
        for block in (None, {"enabled": False, "top_k": 30}):
            _Fake.calls = []
            seen, ctx, _ = self.run_step(block, self.config())
            self.assertIn('"title": "Boden', seen[0])
            self.assertEqual(_Fake.calls, [])

    def test_a_broken_host_falls_back_to_the_full_pool(self):
        cfg = self.config()
        cfg.get_provider_by_name("srv").base_url = "http://127.0.0.1:9"
        seen, ctx, _ = self.run_step({"enabled": True, "top_k": 30}, cfg)
        self.assertIn('"title": "Boden', seen[0])
        self.assertNotIn("decision_prefilter", ctx.extra)

    def test_shipped_workflows_carry_the_opt_in_block(self):
        import yaml
        for name in ("alima", "alima_classic", "alima_classic_v51", "alima_v51", "alima_v51_105"):
            with open(f"workflows/{name}.yaml", encoding="utf-8") as fh:
                wf = yaml.safe_load(fh)
            step = next(s for s in wf["steps"] if s["id"] == "selection_chunks")
            self.assertEqual(step["chunking"]["decision_prefilter"], {"enabled": True, "top_k": 80}, name)


class ClassicHookTest(_ServerCase):
    def run_classic(self, cfg, threshold=50, **kw):
        from types import SimpleNamespace

        from src.utils.pipeline_utils import PipelineStepExecutor

        pool = {"suchbegriff": {t["title"]: {"count": t["count"], "gnd_ids": {t["gnd_id"]}}
                                for t in POOL}}
        cm = MagicMock()
        cm.get_unified_config.return_value = cfg
        cache = MagicMock()
        cache.get_gnd_facts_batch.return_value = {
            t["gnd_id"]: SimpleNamespace(title=t["title"], synonyms="") for t in POOL}
        ex = PipelineStepExecutor(MagicMock(), cache, logger=MagicMock(), config_manager=cm)
        seen: dict = {}

        def single(**k):
            seen["single"] = list(k["gnd_compliant_keywords"])
            seen["single_text"] = k["gnd_keywords_text"]
            return seen["single"], [], MagicMock()

        def chunked(**k):
            seen["chunked"] = list(k["gnd_compliant_keywords"])
            return seen["chunked"], [], MagicMock()

        with patch.object(ex, "_resolve_provider_smart", return_value=("p", "m")), \
                patch.object(ex, "_execute_single_keyword_analysis", side_effect=single), \
                patch.object(ex, "_execute_chunked_keyword_analysis", side_effect=chunked):
            ex.execute_final_keyword_analysis(
                "Schwermetall im Boden", pool, keyword_chunking_threshold=threshold, **kw)
        return ex, seen

    def test_over_threshold_pool_is_cut_to_top_k_and_recorded(self):
        ex, seen = self.run_classic(self.config(), threshold=50)
        self.assertEqual(len(seen["chunked"]), 80)           # 80 > 50: weiter gechunkt, aber kleiner
        hot = [k for k in seen["chunked"] if k.startswith("Cadmium")]
        self.assertEqual(len(hot), 30)                       # alle 30 heißen Begriffe bleiben
        self.assertEqual(ex.last_decision_prefilter["n_in"], 300)
        self.assertEqual(ex.last_decision_prefilter["n_out"], 80)

    def test_a_cut_below_the_threshold_takes_the_single_pass_with_rebuilt_text(self):
        ex, seen = self.run_classic(self.config(), threshold=100)
        self.assertNotIn("chunked", seen)
        self.assertEqual(len(seen["single"]), 80)
        self.assertEqual(seen["single_text"].count("\n"), 80)
        self.assertNotIn("Boden 299", seen["single_text"])

    def test_zero_switches_it_off(self):
        ex, seen = self.run_classic(self.config(), decision_prefilter_top_k=0)
        self.assertEqual(_Fake.calls, [])
        self.assertIsNone(ex.last_decision_prefilter)
        self.assertEqual(len(seen["chunked"]), 300)

    def test_role_off_changes_nothing(self):
        cfg = self.config(decision_default_provider="", decision_default_model="")
        ex, seen = self.run_classic(cfg)
        self.assertEqual(_Fake.calls, [])
        self.assertIsNone(ex.last_decision_prefilter)
        self.assertEqual(len(seen["chunked"]), 300)

    def test_below_threshold_needs_no_call(self):
        ex, seen = self.run_classic(self.config(), threshold=1000)
        self.assertEqual(_Fake.calls, [])
        self.assertIsNone(ex.last_decision_prefilter)
        self.assertEqual(len(seen["single"]), 300)


if __name__ == "__main__":
    unittest.main()
