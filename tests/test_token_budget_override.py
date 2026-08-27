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


if __name__ == "__main__":
    unittest.main()
