"""Thinking control reaches the agentic pipeline.

The classic pipeline has honoured ``step_config.think`` for a long time; the
v4 agentic path dropped it between ``SharedContext`` and ``AgentLoop``, so the
GUI combo, the webapp toggle and ``--step-think`` had no effect there. On a
reasoning model that is the difference between an answer and an empty turn:
the thinking channel is charged against the same ``max_tokens`` budget.

Claude Generated.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.base_step import StepConfig
from src.core.agents.steps.llm_agent_step import LLMAgentStep
from src.core.data_models import AgentResult


def _agent_result(content: str) -> AgentResult:
    return AgentResult(content=content, tool_log=[], iterations=1, tokens_used=0)


def _make_step(llm_cfg=None) -> LLMAgentStep:
    raw = {
        "system_prompt": "sys",
        "user_prompt": "user {abstract}",
        "inputs": {"abstract": "${abstract}"},
        "outputs": {},
        "llm": {"max_iterations": 1, **(llm_cfg or {})},
    }
    cfg = StepConfig(
        id="extraction", type="llm_agent",
        inputs={"abstract": "${abstract}"}, outputs={}, raw=raw,
    )
    return LLMAgentStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())


class TestLLMAgentStepThink(unittest.TestCase):
    def _run_and_capture(self, context, llm_cfg=None):
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result('{"keywords": []}')
        with patch(
            "src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop
        ):
            _make_step(llm_cfg).execute(context)
        return fake_loop.run.call_args.kwargs

    def test_context_think_reaches_agent_loop(self):
        ctx = SharedContext(abstract="text", provider="p", model="m", think=False)
        self.assertIs(self._run_and_capture(ctx)["think"], False)

    def test_step_llm_think_wins_over_context(self):
        ctx = SharedContext(abstract="text", provider="p", model="m", think=False)
        kwargs = self._run_and_capture(ctx, llm_cfg={"think": True})
        self.assertIs(kwargs["think"], True)

    def test_unset_stays_none_provider_default(self):
        ctx = SharedContext(abstract="text", provider="p", model="m")
        self.assertIsNone(self._run_and_capture(ctx)["think"])


class TestSharedContextThinkRoundTrip(unittest.TestCase):
    def test_think_survives_serialization(self):
        ctx = SharedContext(abstract="a", think=False)
        restored = SharedContext.from_dict(ctx.to_dict())
        self.assertIs(restored.think, False)

    def test_default_is_none(self):
        self.assertIsNone(SharedContext.from_dict({}).think)


if __name__ == "__main__":
    unittest.main()
