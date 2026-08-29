"""Operator token-budget override for agentic steps - Claude Generated.

The workflow YAML names ``llm.max_tokens`` per step (4096 in both v5.1
workflows). ``PipelineConfig.global_max_tokens_override`` outranks that value so
the budget can be raised without editing two workflow files in five places
each; absent, every step keeps exactly what its YAML says.

Why it matters: a reasoning model spends this budget on its thinking channel
before the answer starts (measured with ``scripts/probe_thinking.py``), so the
budget is the knob an operator reaches for when steps come back empty.
"""

import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.base_step import StepConfig
from src.core.agents.steps.llm_agent_step import LLMAgentStep
from src.core.data_models import AgentResponse, StopReason
from src.core.pipeline_manager import PipelineConfig


def _llm_service(content: str = '{"keywords": ["k"]}'):
    svc = MagicMock()
    svc.generate_with_tools.return_value = AgentResponse(
        content=content, tool_calls=[], stop_reason=StopReason.END_TURN
    )
    return svc


def _step(llm_block: dict) -> tuple:
    cfg = StepConfig(
        id="extraction",
        type="llm_agent",
        raw={
            "llm": {"max_iterations": 1, **llm_block},
            "system_prompt": "s",
            "user_prompt": "u",
            "tools": [],
        },
    )
    svc = _llm_service()
    return LLMAgentStep(cfg, llm_service=svc, tool_registry=MagicMock()), svc


def _budget_of(svc) -> int:
    return svc.generate_with_tools.call_args.kwargs["max_tokens"]


class TestBudgetPrecedence(unittest.TestCase):
    def test_operator_budget_outranks_the_yaml(self):
        step, svc = _step({"max_tokens": 4096})
        ctx = SharedContext(abstract="a", initial_keywords=[])
        ctx.max_tokens_override = 32768
        step.run(ctx)
        self.assertEqual(_budget_of(svc), 32768)

    def test_without_override_the_yaml_decides(self):
        step, svc = _step({"max_tokens": 4096})
        step.run(SharedContext(abstract="a", initial_keywords=[]))
        self.assertEqual(_budget_of(svc), 4096)

    def test_without_yaml_the_context_default_decides(self):
        step, svc = _step({})
        ctx = SharedContext(abstract="a", initial_keywords=[])
        ctx.max_tokens = 9000
        step.run(ctx)
        self.assertEqual(_budget_of(svc), 9000)


    def test_reflection_turn_uses_the_same_budget(self):
        """The MetaAgent's reflection runs on the same model and hits the same
        wall, so the override has to reach it too (its YAML says 2048)."""
        from src.core.agents.steps.reflection_step import ReflectionStep

        cfg = StepConfig(id="reflection", type="reflection", raw={"llm": {"max_tokens": 2048}})
        svc = _llm_service('{"status": "ok", "action": "finish"}')
        step = ReflectionStep(cfg, llm_service=svc, tool_registry=MagicMock())

        ctx = SharedContext(abstract="a", initial_keywords=[])
        step.run(ctx)
        self.assertEqual(_budget_of(svc), 2048)

        ctx.max_tokens_override = 32768
        step.run(ctx)
        self.assertEqual(_budget_of(svc), 32768)


class TestOverrideReachesTheContext(unittest.TestCase):
    def test_config_default_leaves_the_yaml_alone(self):
        """Absent by default: an unset budget must not silently lower one."""
        self.assertIsNone(PipelineConfig().global_max_tokens_override)

    def test_context_survives_a_warm_start_roundtrip(self):
        ctx = SharedContext(abstract="a", initial_keywords=[])
        ctx.max_tokens_override = 16384
        self.assertEqual(
            SharedContext.from_dict(ctx.to_dict()).max_tokens_override, 16384
        )

    def test_cli_flag_sets_the_config_override(self):
        from src.utils.pipeline_config_builder import PipelineConfigBuilder

        builder = MagicMock()
        builder.baseline = PipelineConfig()
        builder.apply_overrides.side_effect = lambda _o: builder.baseline

        cfg = PipelineConfigBuilder.parse_and_apply_cli_args(
            builder, argparse.Namespace(max_tokens=32768)
        )
        self.assertEqual(cfg.global_max_tokens_override, 32768)

    def test_cli_without_the_flag_leaves_it_unset(self):
        from src.utils.pipeline_config_builder import PipelineConfigBuilder

        builder = MagicMock()
        builder.baseline = PipelineConfig()
        builder.apply_overrides.side_effect = lambda _o: builder.baseline

        cfg = PipelineConfigBuilder.parse_and_apply_cli_args(
            builder, argparse.Namespace(max_tokens=None)
        )
        self.assertIsNone(cfg.global_max_tokens_override)


class TestShippedDefaults(unittest.TestCase):
    """The shipped budget, pinned.

    4096 was measured to lose the answer of a reasoning model outright, because
    the thinking channel spends the same budget (scripts/probe_thinking.py). The
    two v5.1 workflows therefore carry 32768 in every step, reflection included,
    and the code fallback matches. A step that drops back below that reopens the
    failure quietly, so the number is held here rather than in prose.
    """

    SHIPPED = 32768

    def _budgets(self, stem):
        import yaml

        path = Path(__file__).resolve().parent.parent / "workflows" / f"{stem}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        found = []

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == "max_tokens" and isinstance(v, int):
                        found.append(v)
                    else:
                        walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        return found

    def test_both_v51_workflows_carry_the_shipped_budget(self):
        for stem in ("alima_v51", "alima_v51_105"):
            budgets = self._budgets(stem)
            self.assertTrue(budgets, f"{stem}.yaml names no max_tokens at all")
            self.assertEqual(
                sorted(set(budgets)), [self.SHIPPED],
                f"{stem}.yaml has a step below the shipped budget: {sorted(set(budgets))}",
            )

    def test_the_two_workflows_agree(self):
        """Sync rule from workflows/CLAUDE.md: only the classification prompt
        may differ between v51 and v51_105."""
        self.assertEqual(self._budgets("alima_v51"), self._budgets("alima_v51_105"))

    def test_code_fallback_matches_the_workflows(self):
        ctx = SharedContext(abstract="a", initial_keywords=[])
        self.assertEqual(ctx.max_tokens, self.SHIPPED)


class TestWebappBudget(unittest.TestCase):
    """The webapp reaches the same config field as GUI and CLI."""

    def _client(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod

        return TestClient(appmod.app), appmod

    def _fake_pipeline_manager(self):
        class FakePM:
            def __init__(self, *a, **k):
                self.config = None
                self.current_analysis_state = None

            def set_config(self, cfg):
                self.config = cfg

            def set_callbacks(self, **cb):
                self._cb = cb

            def set_interrupt_flag(self, *a, **k):
                pass

            def start_pipeline(self, text, input_type=None, input_source=None):
                if self._cb.get("pipeline_completed"):
                    self._cb["pipeline_completed"](None)
                return "fake-pipeline-id"

        return FakePM()

    def _post(self, sid, data):
        from unittest import mock

        client, appmod = self._client()
        appmod.sessions[sid] = appmod.Session(sid)
        fake_pm = self._fake_pipeline_manager()
        try:
            with mock.patch("src.webapp.routers.analysis.PipelineManager", return_value=fake_pm), \
                 mock.patch("src.webapp.routers.analysis.resolve_input_to_text", return_value="text"), \
                 mock.patch("src.webapp.routers.analysis.AppContext") as mock_ctx:
                mock_ctx.return_value.get_services.return_value = {
                    "config_manager": MagicMock(),
                    "alima_manager": MagicMock(),
                    "cache_manager": MagicMock(),
                    "llm_service": MagicMock(),
                    "prompt_service": MagicMock(),
                    "pipeline_manager": fake_pm,
                }
                resp = client.post(f"/api/analyze/{sid}", data=data)
            self.assertEqual(resp.status_code, 200)
            return fake_pm
        finally:
            appmod.sessions.pop(sid, None)

    def test_form_field_reaches_the_pipeline_config(self):
        pm = self._post("budget-set", {
            "input_type": "text", "content": "abc", "max_tokens_override": "32768",
        })
        self.assertEqual(pm.config.global_max_tokens_override, 32768)

    def test_without_the_field_the_yaml_keeps_deciding(self):
        pm = self._post("budget-unset", {"input_type": "text", "content": "abc"})
        self.assertIsNone(pm.config.global_max_tokens_override)

    def test_garbage_is_no_override_not_an_error(self):
        """The field comes from a browser; a typo must not fail the analysis."""
        from src.webapp.session_io import _parse_max_tokens_override

        self.assertIsNone(_parse_max_tokens_override("viele"))
        self.assertIsNone(_parse_max_tokens_override("0"))
        self.assertIsNone(_parse_max_tokens_override("-5"))
        self.assertIsNone(_parse_max_tokens_override(""))
        self.assertEqual(_parse_max_tokens_override("3276800"), 131072)
        self.assertEqual(_parse_max_tokens_override(" 8192 "), 8192)

    def test_frontend_and_server_agree_on_the_field_name(self):
        """The select, the FormData key and the Form parameter are three places
        that must spell the same name; a rename in one is silent otherwise."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        html = (root / "src/webapp/templates/webapp.html").read_text(encoding="utf-8")
        js = (root / "src/webapp/static/app.js").read_text(encoding="utf-8")
        py = (root / "src/webapp/routers/analysis.py").read_text(encoding="utf-8")

        self.assertIn('id="max-tokens-override"', html)
        self.assertIn("getElementById('max-tokens-override')", js)
        self.assertIn("formData.append('max_tokens_override'", js)
        self.assertIn("max_tokens_override: Optional[str] = Form(None)", py)


if __name__ == "__main__":
    unittest.main()
