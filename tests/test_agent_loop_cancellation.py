"""AgentLoop never checked an LLM generator's AgentResponse.stop_reason for
StopReason.CANCELLED - the one thing a provider generator (e.g.
_generate_openai_with_tools) returns when should_stop() fires mid-stream.

Consequence: a cancel landing before any content/tool_calls existed (the
model still "thinking", or the very first chunk not arrived yet) hit the
exact same code path as a genuinely empty turn - the one-time "write a
final answer" nudge fired a SECOND, non-streaming LLM call (no
stream_callback there, so should_stop couldn't interrupt it promptly
either) instead of stopping. From the chat UI this looked exactly like
clicking the cancel button did nothing: the stream froze, then a new,
uninterruptible generation silently started and ran to completion before
anything happened. Same gap for a cancel landing during the nudge call
itself, or during the max-iterations forced-conclusion call.

Fixed: an explicit stop_reason == CANCELLED check right after the main
call (before any Case 1/Case 2 branching), and after both forced-answer
call sites. - Claude Generated
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.core.agent_loop import AgentLoop
from src.core.data_models import AgentResponse, StopReason, ToolCall


def _make_registry() -> MagicMock:
    reg = MagicMock()
    reg.get_tool_schemas.return_value = [
        {"name": "get_keywords", "description": "", "parameters": {}}
    ]
    reg.execute.return_value = '{"count": 1}'
    return reg


def _make_llm_service(calls: list):
    """Records every generate_with_tools call's kwargs in ``calls`` and
    returns responses from ``calls.response_queue`` in order (repeats the
    last one once exhausted, matching _make_llm_service in
    test_agent_loop_hooks.py - a clean repeat instead of an IndexError keeps
    a wrong "nudge fired again" mutation's failure a readable assertion
    rather than a crash)."""
    svc = MagicMock()

    def _gen(**kwargs):
        calls.append(kwargs)
        queue = calls.response_queue
        return queue.pop(0) if len(queue) > 1 else queue[0]

    svc.generate_with_tools.side_effect = _gen
    return svc


class _Calls(list):
    """A plain list that also carries the queued responses, so the fake
    generate_with_tools above can pop from it without a second global."""
    response_queue: list = []


class CancelledMainCallStopsImmediatelyTest(unittest.TestCase):
    def test_no_nudge_fires_after_a_cancelled_empty_turn(self):
        calls = _Calls()
        calls.response_queue = [
            AgentResponse(content="", tool_calls=[], stop_reason=StopReason.CANCELLED),
        ]
        svc = _make_llm_service(calls)
        loop = AgentLoop(llm_service=svc, tool_registry=_make_registry(), max_iterations=5)

        result = loop.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")

        self.assertEqual(len(calls), 1, "a cancelled empty turn must not trigger the nudge")
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(result.content, "")

    def test_partial_content_already_streamed_is_kept_as_the_result(self):
        calls = _Calls()
        calls.response_queue = [
            AgentResponse(content="Hallo, ich helfe", tool_calls=[], stop_reason=StopReason.CANCELLED),
        ]
        svc = _make_llm_service(calls)
        loop = AgentLoop(llm_service=svc, tool_registry=_make_registry(), max_iterations=5)

        result = loop.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")

        self.assertEqual(len(calls), 1)
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertEqual(result.content, "Hallo, ich helfe")


class CancelledDuringNudgeTest(unittest.TestCase):
    def test_cancelled_nudge_is_not_reported_as_a_model_failure(self):
        """First call: genuinely empty turn (not cancelled) -> nudge fires.
        Second call (the nudge itself): cancelled."""
        calls = _Calls()
        calls.response_queue = [
            AgentResponse(content="", tool_calls=[], stop_reason=StopReason.END_TURN),
            AgentResponse(content="", tool_calls=[], stop_reason=StopReason.CANCELLED),
        ]
        svc = _make_llm_service(calls)
        loop = AgentLoop(llm_service=svc, tool_registry=_make_registry(), max_iterations=5)

        result = loop.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")

        self.assertEqual(len(calls), 2, "nudge must fire exactly once, not loop further")
        self.assertEqual(result.stop_reason, "cancelled")
        self.assertNotIn("keine Antwort geliefert", result.content)


class CancelledDuringMaxIterationsForcedCallTest(unittest.TestCase):
    def test_cancelled_forced_call_does_not_blame_the_model(self):
        """Every iteration calls a tool, exhausting max_iterations; the
        forced "give your final answer" call is then cancelled."""
        tc = ToolCall(id="t1", name="get_keywords", arguments={})
        calls = _Calls()
        calls.response_queue = [
            AgentResponse(content="", tool_calls=[tc], stop_reason=StopReason.TOOL_USE),
        ]

        def _gen(**kwargs):
            calls.append(kwargs)
            if kwargs.get("tools") == []:
                return AgentResponse(content="", tool_calls=[], stop_reason=StopReason.CANCELLED)
            return AgentResponse(content="", tool_calls=[tc], stop_reason=StopReason.TOOL_USE)

        svc = MagicMock()
        svc.generate_with_tools.side_effect = _gen
        loop = AgentLoop(llm_service=svc, tool_registry=_make_registry(), max_iterations=2)

        result = loop.run(system_prompt="s", user_prompt="u", tools=["get_keywords"], provider="p", model="m")

        self.assertEqual(result.stop_reason, "cancelled")
        self.assertNotIn("keine finale", result.content)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
