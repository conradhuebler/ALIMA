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

    def test_status_callback_suppressed_when_on_tool_call_set(self):
        """Phase F: when ``on_tool_call`` is wired, status lines for the
        call/result are suppressed — the bus/hook already carries the
        same info, so the chat log no longer shows duplicates."""
        tool_call = ToolCall(id="t1", name="get_keywords", arguments={"kind": "initial"})

        # Case 1: ``on_tool_call`` set ⇒ status callback is silent.
        responses1 = [
            AgentResponse(content="", tool_calls=[tool_call]),
            AgentResponse(content="Done.", tool_calls=[]),
        ]
        llm1 = _make_llm_service(responses1)
        registry1 = _make_registry()
        status_log: list[str] = []
        loop = AgentLoop(
            llm_service=llm1,
            tool_registry=registry1,
            status_callback=status_log.append,
            on_tool_call=lambda tc: None,
        )
        loop.run(system_prompt="sys", user_prompt="ask", tools=["get_keywords"], provider="p", model="m")
        # The legacy "  🔧 …" / "    ✓ …" lines are gone.
        self.assertEqual([s for s in status_log if s.lstrip().startswith("🔧")], [])
        self.assertEqual([s for s in status_log if s.lstrip().startswith("✓")], [])

        # Case 2: ``on_tool_call`` NOT set ⇒ status callback still
        # receives the legacy lines (backwards-compat for headless CLI).
        # Fresh response list — case 1 already consumed the iterator.
        responses2 = [
            AgentResponse(content="", tool_calls=[tool_call]),
            AgentResponse(content="Done.", tool_calls=[]),
        ]
        llm2 = _make_llm_service(responses2)
        registry2 = _make_registry()
        status_log2: list[str] = []
        loop2 = AgentLoop(
            llm_service=llm2,
            tool_registry=registry2,
            status_callback=status_log2.append,
        )
        loop2.run(system_prompt="sys", user_prompt="ask", tools=["get_keywords"], provider="p", model="m")
        self.assertTrue(any(s.lstrip().startswith("🔧") for s in status_log2))
        self.assertTrue(any(s.lstrip().startswith("✓") for s in status_log2))


if __name__ == "__main__":
    unittest.main()
