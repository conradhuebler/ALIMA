"""Tests for ChatAgentWorker (P-δ.3 + Phase D). Claude Generated.

Phase D removed the ``tool_called`` / ``tool_result`` Qt signals.
Tool-call events are now emitted on ``AlimaStateBus`` with the same
``id`` schema as the agentic and classic producers.

Cases:
1. ``test_tool_call_roundtrip_emits_bus_events`` — single tool call/result
   pair on the bus, ids match.
2. ``test_error_propagates_to_signal`` — AgentLoop.run raising propagates
   to ``generation_error`` (no ``generation_finished``). Unchanged from
   the Qt-signal era.
3. ``test_request_stop_sets_event`` — ``request_stop()`` sets the internal
   threading.Event so AgentLoop's ``should_stop`` callback returns True.
4. ``test_no_qt_tool_signals`` — regression guard: the legacy Qt signals
   do not exist any more.
5. ``test_emits_when_no_panel_subscribed`` — bus emissions are
   best-effort and do not raise when no consumer is connected.

Strategy: monkeypatch ``src.ui.chat_agent_worker.AgentLoop`` with a fake
class. Worker.run() is invoked directly on the main thread (no
QThread.start()) — this exercises all wiring without needing a Qt event
loop or pytest-qt. Bus events captured via ``state_bus.subscribe``.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from PyQt6.QtCore import QCoreApplication

# pyqtSignal needs a QApplication before any QObject is instantiated.
_qapp = QCoreApplication.instance() or QCoreApplication([])

from src.core import state_bus as state_bus_mod
from src.core.data_models import AgentResult, ToolCall
from src.ui import chat_agent_worker as caw_mod
from src.ui.chat_agent_worker import ChatAgentWorker


class _FakeAgentLoop:
    """Captures init kwargs and exposes hooks for tests to drive."""

    instances: list["_FakeAgentLoop"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.on_tool_call = kwargs.get("on_tool_call")
        self.on_tool_result = kwargs.get("on_tool_result")
        self.should_stop = kwargs.get("should_stop")
        self.stream_callback = kwargs.get("stream_callback")
        self._run_behavior = None
        _FakeAgentLoop.instances.append(self)

    def set_behavior(self, behavior):
        """behavior(self) -> AgentResult (or raises)."""
        self._run_behavior = behavior

    def run(self, **run_kwargs):
        self.run_kwargs = run_kwargs
        if self._run_behavior is None:
            return AgentResult(content="")
        return self._run_behavior(self)


def _install_fake_loop(test_case):
    """Patch AgentLoop in chat_agent_worker. Auto-restored on teardown."""
    _FakeAgentLoop.instances.clear()
    original = caw_mod.AgentLoop
    caw_mod.AgentLoop = _FakeAgentLoop
    test_case.addCleanup(lambda: setattr(caw_mod, "AgentLoop", original))


class TestChatAgentWorker(unittest.TestCase):

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()
        _install_fake_loop(self)
        self.llm_service = MagicMock()
        self.tool_registry = MagicMock()

    def tearDown(self):
        state_bus_mod.reset()

    def _make_worker(self, **overrides) -> ChatAgentWorker:
        defaults = dict(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            system_prompt="sys",
            user_prompt="ask",
            provider="ollama",
            model="m",
        )
        defaults.update(overrides)
        return ChatAgentWorker(**defaults)

    def _drive_with_behavior(self, worker: ChatAgentWorker, behavior) -> None:
        """Install a behavior on the next _FakeAgentLoop instance and run."""
        original_init = _FakeAgentLoop.__init__

        def init_with_behavior(self, **kw):
            original_init(self, **kw)
            self.set_behavior(behavior)

        _FakeAgentLoop.__init__ = init_with_behavior
        try:
            worker.run()
        finally:
            _FakeAgentLoop.__init__ = original_init
        QCoreApplication.processEvents()

    # ------------------------------------------------------------------
    # Phase D: bus emissions (replaces Qt-signal roundtrip test).
    # ------------------------------------------------------------------

    def test_tool_call_roundtrip_emits_bus_events(self):
        worker = self._make_worker()

        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        def behavior(loop: _FakeAgentLoop) -> AgentResult:
            loop.on_tool_call(ToolCall(id="t1", name="get_keywords", arguments={"kind": "initial"}))
            loop.on_tool_result("get_keywords", '{"count":3}')
            loop.stream_callback("Antwort")
            return AgentResult(content="Antwort", iterations=2)

        self._drive_with_behavior(worker, behavior)

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["name"], "get_keywords")
        self.assertEqual(called[0]["arguments"], {"kind": "initial"})
        self.assertEqual(called[0]["id"], "t1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "get_keywords")
        self.assertEqual(results[0]["result"], '{"count":3}')
        # Result re-uses the call's id.
        self.assertEqual(results[0]["id"], "t1")
        # _last_tool_call_id cleared so the next pair starts fresh.
        self.assertEqual(worker._last_tool_call_id, "")

    def test_thinking_received_signal_emitted(self):
        """on_thinking wiring: AgentLoop thinking content reaches the
        thinking_received Qt signal. Claude Generated."""
        worker = self._make_worker()
        received: list[str] = []
        worker.thinking_received.connect(received.append)

        def behavior(loop: _FakeAgentLoop) -> AgentResult:
            loop.kwargs["on_thinking"]("Ich überlege.")
            loop.stream_callback("Antwort")
            return AgentResult(content="Antwort")

        self._drive_with_behavior(worker, behavior)
        self.assertEqual(received, ["Ich überlege."])

    def test_emits_with_unified_id_when_tc_id_empty(self):
        """Phase A contract: when ToolCall.id is empty, worker fills via
        ``make_tool_call_id`` and the result event reuses the same id."""
        worker = self._make_worker()
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        def behavior(loop: _FakeAgentLoop) -> AgentResult:
            loop.on_tool_call(ToolCall(id="", name="search_gnd", arguments={"term": "x"}))
            loop.on_tool_result("search_gnd", "[]")
            return AgentResult(content="")

        self._drive_with_behavior(worker, behavior)

        self.assertEqual(len(called), 1)
        self.assertTrue(called[0]["id"].startswith("tc_"))
        self.assertNotEqual(called[0]["id"], "")
        self.assertEqual(results[0]["id"], called[0]["id"])

    def test_emits_when_no_panel_subscribed(self):
        """Bus emissions are best-effort: do not raise when nobody listens."""
        worker = self._make_worker()
        # After setUp().reset() the bus has no subscribers.

        def behavior(loop: _FakeAgentLoop) -> AgentResult:
            loop.on_tool_call(ToolCall(id="abc", name="t", arguments={}))
            loop.on_tool_result("t", "ok")
            return AgentResult(content="")

        # Must not raise.
        self._drive_with_behavior(worker, behavior)

    def test_no_qt_tool_signals(self):
        """Regression guard (Phase D): the legacy Qt signals must NOT
        exist on ChatAgentWorker any more — they were replaced by bus
        emissions to consolidate producers."""
        self.assertFalse(
            hasattr(ChatAgentWorker, "tool_called"),
            "ChatAgentWorker.tool_called was removed in Phase D; bus is the producer",
        )
        self.assertFalse(
            hasattr(ChatAgentWorker, "tool_result"),
            "ChatAgentWorker.tool_result was removed in Phase D; bus is the producer",
        )

    # ------------------------------------------------------------------
    # Unchanged: token / error / request_stop tests.
    # ------------------------------------------------------------------

    def test_error_propagates_to_signal(self):
        worker = self._make_worker()
        errors: list[str] = []
        finished: list[AgentResult] = []
        worker.generation_error.connect(lambda e: errors.append(e))
        worker.generation_finished.connect(lambda r: finished.append(r))

        def boom(loop: _FakeAgentLoop) -> AgentResult:
            raise RuntimeError("LLM down")

        self._drive_with_behavior(worker, boom)

        self.assertEqual(len(errors), 1)
        self.assertIn("LLM down", errors[0])
        self.assertEqual(finished, [])

    def test_request_stop_sets_event(self):
        worker = self._make_worker()

        worker.request_stop()

        self.assertTrue(worker._stop_event.is_set())
        self.llm_service.cancel_generation.assert_called_once_with(
            reason="user_requested"
        )

        captured: dict = {}

        def behavior(loop: _FakeAgentLoop) -> AgentResult:
            captured["should_stop_value"] = loop.should_stop()
            return AgentResult(content="")

        self._drive_with_behavior(worker, behavior)

        self.assertTrue(captured["should_stop_value"])


if __name__ == "__main__":
    unittest.main()
