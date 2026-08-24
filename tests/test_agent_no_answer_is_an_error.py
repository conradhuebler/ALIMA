"""A turn the loop had to answer for itself is not a successful step.

``AgentResult.error`` already carried the contract "content holds an error
string, NOT a model answer — callers must check this instead of treating the
run as successful". The empty-turn fallbacks broke it: they wrote a
loop-authored warning into ``content`` and left ``error`` unset, so
``LLMAgentStep`` parsed the warning as if it were the model's JSON, got ``{}``,
and reported ``success=True``. That is what let a workflow keep cycling on a
model that had answered nothing at all.

Claude Generated.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.core.agent_loop import AgentLoop
from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.base_step import StepConfig
from src.core.agents.steps.llm_agent_step import LLMAgentStep
from src.core.data_models import AgentResponse, AgentResult, StopReason, ToolCall


def _registry():
    reg = MagicMock()
    reg.get_tool_schemas.return_value = [
        {"name": "get_keywords", "description": "", "parameters": {}}
    ]
    reg.execute.return_value = '{"count": 0}'
    return reg


def _llm(responses):
    svc = MagicMock()
    it = iter(responses)
    last = [None]

    def _gen(**_kw):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    svc.generate_with_tools.side_effect = _gen
    return svc


def _run(responses, **kw):
    loop = AgentLoop(llm_service=_llm(responses), tool_registry=_registry(), **kw)
    return loop.run(
        system_prompt="s", user_prompt="u", tools=[], provider="p", model="m",
        max_tokens=256,
    )


class TestLoopMarksItsOwnDiagnostics(unittest.TestCase):
    def test_budget_exhausted_is_an_error(self):
        res = _run([AgentResponse(content="", tool_calls=[],
                                  stop_reason=StopReason.MAX_TOKENS)])
        self.assertIsNotNone(res.error)
        self.assertIn("Token-Budget", res.error)

    def test_silent_empty_turn_is_an_error(self):
        res = _run([AgentResponse(content="", tool_calls=[],
                                  stop_reason=StopReason.END_TURN)])
        self.assertIsNotNone(res.error)

    def test_reasoning_only_is_model_output_not_an_error(self):
        """The answer arrived in the wrong channel, but it IS the model
        talking — a caller may still find its JSON in there."""
        res = _run([AgentResponse(content="", tool_calls=[], reasoning='{"keywords": []}',
                                  stop_reason=StopReason.END_TURN)])
        self.assertIsNone(res.error)
        self.assertIn('{"keywords": []}', res.content)

    def test_a_real_answer_is_not_an_error(self):
        res = _run([AgentResponse(content="Fertig.", tool_calls=[],
                                  stop_reason=StopReason.END_TURN)])
        self.assertIsNone(res.error)

    def test_no_final_text_after_tool_calls_is_an_error(self):
        tc = ToolCall(id="t1", name="get_keywords", arguments={})
        res = _run(
            [AgentResponse(content="", tool_calls=[tc], stop_reason=StopReason.TOOL_USE)],
            max_iterations=1,
        )
        self.assertIsNotNone(res.error)


class TestStepFailsInsteadOfSucceedingEmpty(unittest.TestCase):
    def _step(self):
        raw = {
            "system_prompt": "sys",
            "user_prompt": "user {abstract}",
            "inputs": {"abstract": "${abstract}"},
            "outputs": {},
            "llm": {"max_iterations": 1},
        }
        cfg = StepConfig(id="extraction", type="llm_agent",
                         inputs={"abstract": "${abstract}"}, outputs={}, raw=raw)
        return LLMAgentStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())

    def _execute(self, agent_result):
        loop = MagicMock()
        loop.run.return_value = agent_result
        ctx = SharedContext(abstract="text", provider="p", model="m")
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=loop):
            return self._step().execute(ctx)

    def test_budget_message_fails_the_step(self):
        warning = "⚠️ Das Modell hat das Token-Budget (max_tokens=4096) aufgebraucht…"
        res = self._execute(AgentResult(
            content=warning, tool_log=[], iterations=1, tokens_used=0,
            error=warning, stop_reason="max_tokens",
        ))
        self.assertFalse(res.success)
        self.assertIn("Token-Budget", res.error)

    def test_a_parseable_answer_still_succeeds(self):
        res = self._execute(AgentResult(
            content='{"keywords": ["A"]}', tool_log=[], iterations=1, tokens_used=0,
        ))
        self.assertTrue(res.success)
        self.assertEqual(res.data["response"]["keywords"], ["A"])

    def test_a_legitimately_empty_answer_still_succeeds(self):
        """{"keywords": []} is a well-formed 'found nothing', not a failure."""
        res = self._execute(AgentResult(
            content='{"keywords": []}', tool_log=[], iterations=1, tokens_used=0,
        ))
        self.assertTrue(res.success)


class TestReflectionStepFailsLoudly(unittest.TestCase):
    def test_no_answer_fails_the_reflection_step(self):
        """MetaAgent has a documented fallback for a failed reflection; it had
        no fallback for a reflection that 'succeeded' with status=None."""
        from src.core.agents.steps.reflection_step import ReflectionStep

        cfg = StepConfig(id="reflection", type="reflection", inputs={}, outputs={}, raw={})
        step = ReflectionStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())
        warning = "⚠️ Das Modell hat keine Antwort geliefert"
        loop = MagicMock()
        loop.run.return_value = AgentResult(
            content=warning, tool_log=[], iterations=1, tokens_used=0, error=warning,
        )
        ctx = SharedContext(abstract="text", provider="p", model="m")
        with patch("src.core.agents.steps.reflection_step.AgentLoop", return_value=loop):
            res = step.execute(ctx)
        self.assertFalse(res.success)
        self.assertIn("Reflection LLM call failed", res.error)


if __name__ == "__main__":
    unittest.main()
