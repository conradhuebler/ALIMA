"""Tests for ChatAgentWorker (WP10 P-δ.3). Claude Generated.

Three cases:
1. ``test_tool_call_roundtrip_emits_signals`` — tool_called / tool_result /
   token_received / generation_finished all fire correctly.
2. ``test_error_propagates_to_signal`` — AgentLoop.run raising propagates
   to ``generation_error`` (no ``generation_finished``).
3. ``test_request_stop_sets_event`` — ``request_stop()`` sets the internal
   threading.Event so AgentLoop's ``should_stop`` callback returns True.

Strategy: monkeypatch ``src.ui.chat_agent_worker.AgentLoop`` with a fake
class. Worker.run() is invoked directly on the main thread (no
QThread.start()) — this exercises all wiring without needing a Qt event
loop or pytest-qt. Signals are captured via direct slot-connect.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from PyQt6.QtCore import QCoreApplication

# pyqtSignal needs a QApplication before any QObject is instantiated.
_qapp = QCoreApplication.instance() or QCoreApplication([])

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
        _install_fake_loop(self)
        self.llm_service = MagicMock()
        self.tool_registry = MagicMock()

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

    # ------------------------------------------------------------------

    def test_tool_call_roundtrip_emits_signals(self):
        worker = self._make_worker()

        tokens: list[str] = []
        tool_calls: list[tuple[str, dict]] = []
        tool_results: list[tuple[str, str]] = []
        finished: list[AgentResult] = []
        errors: list[str] = []

        worker.token_received.connect(lambda t: tokens.append(t))
        worker.tool_called.connect(lambda n, a: tool_calls.append((n, a)))
        worker.tool_result.connect(lambda n, r: tool_results.append((n, r)))
        worker.generation_finished.connect(lambda r: finished.append(r))
        worker.generation_error.connect(lambda e: errors.append(e))

        def behavior(loop: _FakeAgentLoop) -> AgentResult:
            loop.on_tool_call(ToolCall(id="t1", name="get_keywords", arguments={"kind": "initial"}))
            loop.on_tool_result("get_keywords", '{"count":3}')
            loop.stream_callback("Antwort")
            return AgentResult(content="Antwort", iterations=2)

        _FakeAgentLoop.instances  # populated when worker.run() constructs the loop
        # Pre-set behavior on the next instance via a closure trick:
        # we capture the instance lazily inside `_run_behavior` of the first
        # instance created during worker.run().
        original_init = _FakeAgentLoop.__init__

        def init_with_behavior(self, **kw):
            original_init(self, **kw)
            self.set_behavior(behavior)

        _FakeAgentLoop.__init__ = init_with_behavior
        try:
            worker.run()  # direct call — no QThread.start()
        finally:
            _FakeAgentLoop.__init__ = original_init

        QCoreApplication.processEvents()

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0][0], "get_keywords")
        self.assertEqual(tool_calls[0][1], {"kind": "initial"})

        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0], ("get_keywords", '{"count":3}'))

        self.assertEqual(tokens, ["Antwort"])
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0].content, "Antwort")
        self.assertEqual(errors, [])

    def test_error_propagates_to_signal(self):
        worker = self._make_worker()
        errors: list[str] = []
        finished: list[AgentResult] = []
        worker.generation_error.connect(lambda e: errors.append(e))
        worker.generation_finished.connect(lambda r: finished.append(r))

        def boom(loop: _FakeAgentLoop) -> AgentResult:
            raise RuntimeError("LLM down")

        original_init = _FakeAgentLoop.__init__

        def init_with_behavior(self, **kw):
            original_init(self, **kw)
            self.set_behavior(boom)

        _FakeAgentLoop.__init__ = init_with_behavior
        try:
            worker.run()
        finally:
            _FakeAgentLoop.__init__ = original_init

        QCoreApplication.processEvents()

        self.assertEqual(len(errors), 1)
        self.assertIn("LLM down", errors[0])
        self.assertEqual(finished, [])

    def test_request_stop_sets_event(self):
        worker = self._make_worker()

        # request_stop should also try to call llm_service.cancel_generation;
        # confirm both side-effects.
        worker.request_stop()

        self.assertTrue(worker._stop_event.is_set())
        self.llm_service.cancel_generation.assert_called_once_with(
            reason="user_requested"
        )

        # When the worker hands `self._stop_event.is_set` to AgentLoop as
        # `should_stop`, it must return True after request_stop() was called.
        captured: dict = {}

        def behavior(loop: _FakeAgentLoop) -> AgentResult:
            captured["should_stop_value"] = loop.should_stop()
            return AgentResult(content="")

        original_init = _FakeAgentLoop.__init__

        def init_with_behavior(self, **kw):
            original_init(self, **kw)
            self.set_behavior(behavior)

        _FakeAgentLoop.__init__ = init_with_behavior
        try:
            worker.run()
        finally:
            _FakeAgentLoop.__init__ = original_init

        self.assertTrue(captured["should_stop_value"])


if __name__ == "__main__":
    unittest.main()
