"""Tests for AgentLoop ↔ AlimaStateBus tool-event plumbing (P-δ.5a).

Claude Generated.

Verifies that ``LLMAgentStep._invoke_loop`` wires AgentLoop's
``on_tool_call``/``on_tool_result`` hooks to bus events ``tool.called``
and ``tool.result`` with the correct payload shape.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt6.QtCore import QCoreApplication

# pyqtSignal in AlimaStateBus needs an instance before the class body runs.
_qapp = QCoreApplication.instance() or QCoreApplication([])

from src.core import state_bus as state_bus_mod
from src.core.agent_loop import AgentLoop
from src.core.data_models import AgentResponse, ToolCall


def _make_registry() -> MagicMock:
    reg = MagicMock()
    reg.get_tool_schemas.return_value = [
        {"name": "get_keywords", "description": "", "parameters": {}}
    ]
    reg.execute.return_value = '{"count": 2, "keywords": ["A", "B"]}'
    return reg


def _make_llm_service(responses):
    svc = MagicMock()
    it = iter(responses)
    last = [None]

    def _gen(**_kwargs):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    svc.generate_with_tools.side_effect = _gen
    return svc


class TestToolSignalBus(unittest.TestCase):

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()

    def tearDown(self):
        state_bus_mod.reset()

    def test_agent_loop_hooks_emit_bus_events(self):
        """Hooks attached to AgentLoop forward tool dispatch to the bus."""
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        tc = ToolCall(id="t1", name="get_keywords", arguments={"kind": "initial"})
        responses = [
            AgentResponse(content="", tool_calls=[tc]),
            AgentResponse(content="done", tool_calls=[]),
        ]
        llm = _make_llm_service(responses)
        registry = _make_registry()

        def on_call(call):
            self.bus.emit_event(
                "tool.called",
                {
                    "name": call.name,
                    "arguments": dict(call.arguments or {}),
                    "id": call.id,
                },
            )

        def on_result(name, result):
            self.bus.emit_event(
                "tool.result", {"name": name, "result": result}
            )

        loop = AgentLoop(
            llm_service=llm,
            tool_registry=registry,
            on_tool_call=on_call,
            on_tool_result=on_result,
        )
        loop.run(
            system_prompt="sys",
            user_prompt="ask",
            tools=["get_keywords"],
            provider="ollama",
            model="m",
        )

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["name"], "get_keywords")
        self.assertEqual(called[0]["arguments"], {"kind": "initial"})
        self.assertEqual(called[0]["id"], "t1")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "get_keywords")
        self.assertIn('"count": 2', results[0]["result"])

    def test_subscriber_exception_does_not_break_dispatch(self):
        """One subscriber raising must not silence the others."""
        survived: list[dict] = []
        self.bus.subscribe(
            "tool.called", lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        self.bus.subscribe("tool.called", survived.append)

        self.bus.emit_event("tool.called", {"name": "x", "arguments": {}, "id": "1"})
        self.assertEqual(len(survived), 1)
        self.assertEqual(survived[0]["name"], "x")


if __name__ == "__main__":
    unittest.main()
