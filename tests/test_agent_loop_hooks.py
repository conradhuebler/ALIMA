"""Tests for AgentLoop hooks added in WP10 P-δ.3. Claude Generated.

Two cases:
1. ``on_tool_call`` + ``on_tool_result`` fire exactly once per tool dispatch.
2. ``should_stop`` breaks the loop between iterations.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.core.agent_loop import AgentLoop
from src.core.data_models import AgentResponse, ToolCall


def _make_registry() -> MagicMock:
    """Mock ToolRegistry: schema-less, execute returns fixed JSON."""
    reg = MagicMock()
    reg.get_tool_schemas.return_value = [
        {"name": "get_keywords", "description": "", "parameters": {}}
    ]
    reg.execute.return_value = '{"count": 3, "keywords": ["A", "B", "C"]}'
    return reg


def _make_llm_service(responses):
    """Mock LlmService whose ``generate_with_tools`` returns each
    ``AgentResponse`` in order then raises ``StopIteration``-ish behavior
    by repeating the last one."""
    svc = MagicMock()
    it = iter(responses)
    last = [None]

    def _gen(**kwargs):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    svc.generate_with_tools.side_effect = _gen
    return svc


class TestAgentLoopHooks(unittest.TestCase):

    def test_on_tool_call_and_result_invoked_per_dispatch(self):
        """One tool-dispatch ⇒ on_tool_call called once, on_tool_result once."""
        tool_call = ToolCall(id="t1", name="get_keywords", arguments={"kind": "initial"})
        responses = [
            AgentResponse(content="", tool_calls=[tool_call]),
            AgentResponse(content="Done.", tool_calls=[]),  # final answer
        ]
        llm = _make_llm_service(responses)
        registry = _make_registry()

        call_log = []
        result_log = []

        loop = AgentLoop(
            llm_service=llm,
            tool_registry=registry,
            on_tool_call=lambda tc: call_log.append(tc),
            on_tool_result=lambda name, res: result_log.append((name, res)),
        )
        result = loop.run(
            system_prompt="sys",
            user_prompt="ask",
            tools=["get_keywords"],
            provider="ollama",
            model="m",
        )

        self.assertEqual(len(call_log), 1)
        self.assertEqual(call_log[0].name, "get_keywords")
        self.assertEqual(call_log[0].arguments, {"kind": "initial"})

        self.assertEqual(len(result_log), 1)
        name, result_str = result_log[0]
        self.assertEqual(name, "get_keywords")
        self.assertIn('"count": 3', result_str)

        self.assertEqual(result.content, "Done.")

    def test_should_stop_breaks_loop_between_iterations(self):
        """should_stop returning True after first call halts the loop."""
        tool_call = ToolCall(id="t1", name="get_keywords", arguments={})
        # Provide many tool-call responses; the loop should stop before
        # consuming most of them.
        responses = [
            AgentResponse(content="", tool_calls=[tool_call]) for _ in range(10)
        ]
        llm = _make_llm_service(responses)
        registry = _make_registry()

        stop_counter = {"calls": 0}

        def _should_stop() -> bool:
            stop_counter["calls"] += 1
            # Halt on the second invocation, so we get at least 1 iteration.
            return stop_counter["calls"] >= 2

        loop = AgentLoop(
            llm_service=llm,
            tool_registry=registry,
            max_iterations=10,
            should_stop=_should_stop,
        )
        loop.run(system_prompt="sys", user_prompt="ask", tools=[], provider="p", model="m")

        # Only one LLM call should have happened before should_stop returned True.
        self.assertEqual(llm.generate_with_tools.call_count, 1)


if __name__ == "__main__":
    unittest.main()
