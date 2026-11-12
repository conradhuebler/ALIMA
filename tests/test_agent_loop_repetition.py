"""AgentLoop after a repetition stop, per ``repetition_action`` - Claude Generated

The provider side is faked: ``generate_with_tools`` returns a REPETITION
response (as ``RepetitionGuard`` produces it) and then a normal one. Pins what
each action does, the attempt cap, and the loop-local ``set_generation_params``
tool including its validation.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from src.core.agent_loop import AgentLoop, apply_generation_params
from src.core.data_models import AgentResponse, StopReason, ToolCall
from src.utils.config_models import RepetitionDetectionConfig

REP_INFO = {
    "channel": "answer", "rate": 0.017, "threshold": 0.06, "kept_chars": 18,
    "cut_chars": 4000, "target": "content", "detection_type": "low_information",
}


def _rep_response(content="Teil vor der Schleife") -> AgentResponse:
    return AgentResponse(content=content, stop_reason=StopReason.REPETITION, repetition=dict(REP_INFO))


class _FakeLLM:
    """Scripted generate_with_tools; records every call's kwargs (messages copied)."""

    def __init__(self, responses: List[AgentResponse]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_with_tools(self, **kwargs):
        rec = dict(kwargs)
        rec["messages"] = [dict(m) for m in kwargs["messages"]]
        self.calls.append(rec)
        return self.responses.pop(0)


def _loop(llm, action, max_retries=1, tools_registry=None, **kw):
    registry = tools_registry or MagicMock()
    if tools_registry is None:
        registry.get_tool_schemas.return_value = [{"name": "search_gnd", "parameters": {}}]
        registry.execute.return_value = "{}"
    cfg = RepetitionDetectionConfig(repetition_action=action, max_retries=max_retries)
    status: List[str] = []
    loop = AgentLoop(llm, registry, max_iterations=5, status_callback=status.append,
                     repetition_config=cfg, **kw)
    return loop, status, registry


class AbortActionTest(unittest.TestCase):
    def test_abort_ends_the_run_with_the_partial_text(self):
        llm = _FakeLLM([_rep_response()])
        loop, status, _ = _loop(llm, "abort")
        result = loop.run("sys", "frage", tools=["search_gnd"], provider="p", model="m")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.stop_reason, "repetition")
        self.assertEqual(result.content, "Teil vor der Schleife")
        self.assertIsNone(result.error, "partial model text is not a loop-authored error")
        self.assertEqual(result.repetition["channel"], "answer")
        self.assertTrue(any("Wiederholungsschleife" in s for s in status))
        self.assertFalse(any("Abbruch durch Operator" in s for s in status))

    def test_abort_without_text_does_not_trigger_the_empty_answer_nudge(self):
        """A thinking-channel loop leaves no answer text; that is not an empty turn."""
        llm = _FakeLLM([_rep_response("")])
        loop, status, _ = _loop(llm, "abort")
        result = loop.run("sys", "frage", tools=None, provider="p", model="m")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.stop_reason, "repetition")
        self.assertIsNone(result.error)


class RetryActionTest(unittest.TestCase):
    def test_retry_repeats_the_call_with_varied_sampling(self):
        llm = _FakeLLM([_rep_response(), AgentResponse(content="fertige Antwort")])
        loop, status, _ = _loop(llm, "retry")
        result = loop.run("sys", "frage", tools=["search_gnd"], provider="p", model="m",
                          temperature=0.3, top_p=0.5)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result.content, "fertige Antwort")
        self.assertEqual(result.stop_reason, "end_turn")
        first, second = llm.calls
        self.assertEqual((first["temperature"], first["top_p"]), (0.3, 0.5))
        self.assertGreater(second["temperature"], first["temperature"])
        self.assertGreaterEqual(second["top_p"], 0.9)
        # same request: the partial text is NOT added to the conversation
        self.assertEqual(first["messages"], second["messages"])
        self.assertEqual(result.iterations, 1, "a retry does not use up an iteration")

    def test_retry_cap_then_partial_result(self):
        llm = _FakeLLM([_rep_response("A"), _rep_response("B")])
        loop, _status, _ = _loop(llm, "retry", max_retries=1)
        result = loop.run("sys", "frage", tools=None, provider="p", model="m")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result.stop_reason, "repetition")
        self.assertEqual(result.content, "B")

    def test_zero_retries_behaves_like_abort(self):
        llm = _FakeLLM([_rep_response()])
        loop, _s, _ = _loop(llm, "retry", max_retries=0)
        result = loop.run("sys", "frage", tools=None, provider="p", model="m")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.stop_reason, "repetition")


class AgentActionTest(unittest.TestCase):
    def test_agent_gets_partial_text_and_a_note(self):
        llm = _FakeLLM([_rep_response(), AgentResponse(content="neu formuliert")])
        loop, status, _ = _loop(llm, "agent")
        result = loop.run("sys", "frage", tools=["search_gnd"], provider="p", model="m")
        self.assertEqual(result.content, "neu formuliert")
        second = llm.calls[1]
        msgs = second["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": "sys"}, "system prompt untouched")
        self.assertEqual(msgs[-2], {"role": "assistant", "content": "Teil vor der Schleife"})
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("Wiederholungsschleife", msgs[-1]["content"])
        self.assertIn("set_generation_params", msgs[-1]["content"])
        names = [t["name"] for t in second["tools"]]
        self.assertIn("set_generation_params", names)
        self.assertIn("search_gnd", names)
        first_names = [t["name"] for t in llm.calls[0]["tools"]]
        self.assertNotIn("set_generation_params", first_names, "only offered after a loop")

    def test_agent_note_in_a_tool_less_run_offers_the_param_tool(self):
        llm = _FakeLLM([_rep_response(""), AgentResponse(content="ok")])
        loop, _s, _ = _loop(llm, "agent")
        loop.run("sys", "frage", tools=None, provider="p", model="m")
        self.assertIsNone(llm.calls[0]["tools"])
        self.assertEqual([t["name"] for t in llm.calls[1]["tools"]], ["set_generation_params"])
        # nothing before the loop → no empty assistant turn
        self.assertEqual(llm.calls[1]["messages"][-2]["role"], "user")

    def test_agent_sets_generation_params_for_the_following_calls(self):
        call = ToolCall(id="t1", name="set_generation_params",
                        arguments={"temperature": 0.9, "repetition_penalty": 1.2, "max_tokens": 2048})
        llm = _FakeLLM([
            _rep_response(),
            AgentResponse(tool_calls=[call], stop_reason=StopReason.TOOL_USE),
            AgentResponse(content="fertig"),
        ])
        loop, _s, registry = _loop(llm, "agent")
        result = loop.run("sys", "frage", tools=["search_gnd"], provider="p", model="m",
                          temperature=0.3, max_tokens=4096)
        self.assertEqual(result.content, "fertig")
        registry.execute.assert_not_called()  # answered by the loop, not the shared registry
        last = llm.calls[-1]
        self.assertEqual(last["temperature"], 0.9)
        self.assertEqual(last["repetition_penalty"], 1.2)
        self.assertEqual(last["max_tokens"], 2048)
        tool_msg = [m for m in last["messages"] if m.get("role") == "tool"][-1]
        self.assertTrue(json.loads(tool_msg["content"])["ok"])

    def test_param_tool_is_not_dispatched_before_it_was_offered(self):
        call = ToolCall(id="t1", name="set_generation_params", arguments={"temperature": 1.5})
        llm = _FakeLLM([
            AgentResponse(tool_calls=[call], stop_reason=StopReason.TOOL_USE),
            AgentResponse(content="fertig"),
        ])
        loop, _s, registry = _loop(llm, "agent")
        loop.run("sys", "frage", tools=["search_gnd"], provider="p", model="m", temperature=0.3)
        registry.execute.assert_called_once()
        self.assertEqual(llm.calls[-1]["temperature"], 0.3)

    def test_agent_cap_ends_with_the_partial_result(self):
        llm = _FakeLLM([_rep_response("A"), _rep_response("B")])
        loop, _s, _ = _loop(llm, "agent", max_retries=1)
        result = loop.run("sys", "frage", tools=["search_gnd"], provider="p", model="m")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(result.stop_reason, "repetition")
        self.assertEqual(result.content, "B")


class LoopPassesRepetitionConfigTest(unittest.TestCase):
    def test_config_reaches_generate_with_tools(self):
        llm = _FakeLLM([AgentResponse(content="ok")])
        loop, _s, _ = _loop(llm, "abort")
        loop.run("sys", "frage", tools=None, provider="p", model="m")
        self.assertIs(llm.calls[0]["repetition_config"], loop.repetition_config)
        self.assertNotIn("repetition_penalty", llm.calls[0], "no override sent by default")

    def test_default_action_is_agent(self):
        loop = AgentLoop(MagicMock(), MagicMock())
        self.assertEqual(loop.repetition_config.repetition_action, "agent")


class SetGenerationParamsValidationTest(unittest.TestCase):
    def _gen(self):
        return {"temperature": 0.3, "top_p": 0.9, "max_tokens": 4096, "repetition_penalty": None}

    def test_valid_values_are_applied(self):
        gen = self._gen()
        out = apply_generation_params(gen, {"temperature": "0.8", "top_p": 0.95, "max_tokens": 8192.0})
        self.assertTrue(out["ok"])
        self.assertEqual(gen["temperature"], 0.8)
        self.assertEqual(gen["max_tokens"], 8192)
        self.assertIsInstance(gen["max_tokens"], int)

    def test_invalid_input_changes_nothing(self):
        cases = [
            {"temperature": 2.5},
            {"top_p": 0},
            {"repetition_penalty": 0.9},
            {"max_tokens": 100},
            {"max_tokens": 1000.5},
            {"temperature": True},
            {"temperature": "heiß"},
            {"temperature": float("nan")},
            {"seed": 3},
            {},
            {"temperature": 0.8, "top_p": 5},   # all or nothing
        ]
        for args in cases:
            with self.subTest(args=args):
                gen = self._gen()
                out = apply_generation_params(gen, args)
                self.assertIn("error", out)
                self.assertEqual(gen, self._gen())
        self.assertIn("error", apply_generation_params(self._gen(), "temperature=1"))


if __name__ == "__main__":
    unittest.main()
