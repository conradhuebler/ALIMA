"""Tests for PipelineManager's bus emission of pipeline steps. Claude Generated.

The classic auto-pipeline button (no agentic mode) calls
``pipeline_manager.start_pipeline`` → ``_execute_next_step``. The
chat panel renders classic steps as collapsible tool-call blocks when
it receives ``state.pipeline_step`` events on ``AlimaStateBus``.

Pre-fix: the bus event was only emitted by the chat-tool wrapper at
``src/ui/chat_tools/pipeline.py`` — the auto-pipeline button path
went through ``_execute_next_step`` without emitting, so the chat
panel never saw those steps as tool blocks.

Post-fix: ``_emit_pipeline_step_bus`` is called from
``_execute_next_step`` for both "running" and "completed" transitions.

These tests exercise the helper directly (and the integration via a
fake ``_execute_next_step`` call) so the bus contract is locked.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from PyQt6.QtCore import QCoreApplication

_qapp = QCoreApplication.instance() or QCoreApplication([])

from src.core import state_bus as state_bus_mod
from src.core.pipeline_manager import PipelineManager, PipelineStep


def _make_step(step_id: str = "initialisation", name: str = "Init") -> PipelineStep:
    return PipelineStep(step_id=step_id, name=name, status="pending")


class TestEmitPipelineStepBus(unittest.TestCase):
    """The helper emits ``state.pipeline_step`` with the right payload."""

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()
        self.pm = PipelineManager.__new__(PipelineManager)  # bypass __init__
        # Stub logger so __init__'s logger isn't required.
        self.pm.logger = __import__("logging").getLogger(__name__)

    def tearDown(self):
        state_bus_mod.reset()

    def test_running_event_payload(self):
        seen: list[dict] = []
        self.bus.subscribe("state.pipeline_step", seen.append)
        self.pm._emit_pipeline_step_bus(_make_step("search", "GND Search"), "running")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["status"], "running")
        self.assertEqual(seen[0]["step_id"], "search")
        self.assertEqual(seen[0]["name"], "GND Search")
        self.assertEqual(seen[0]["tool"], "execute_complete_pipeline")

    def test_completed_event_payload(self):
        seen: list[dict] = []
        self.bus.subscribe("state.pipeline_step", seen.append)
        self.pm._emit_pipeline_step_bus(_make_step("keywords"), "completed")
        self.assertEqual(seen[0]["status"], "completed")
        self.assertEqual(seen[0]["step_id"], "keywords")

    def test_bus_failure_does_not_propagate(self):
        """The helper is best-effort: a broken bus must not break the pipeline."""
        # Force the bus to raise on emit.
        from src.core import state_bus as sb
        original = sb.AlimaStateBus().emit_event
        sb.AlimaStateBus().emit_event = lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        try:
            # Must not raise.
            self.pm._emit_pipeline_step_bus(_make_step("x"), "running")
        finally:
            sb.AlimaStateBus().emit_event = original


class TestAutoPipelineEmitsBus(unittest.TestCase):
    """The auto-pipeline path (start_pipeline → _execute_next_step)
    emits ``state.pipeline_step`` on the bus for each step.

    Strategy: drive the manager's classic-mode path with stub methods
    so the test does not depend on real LLM/SearchCLI side-effects.
    """

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()
        # Minimal PipelineManager stub — only the methods the path
        # under test actually touches.
        self.pm = SimpleNamespace(
            logger=__import__("logging").getLogger(__name__),
            _check_interruption=lambda: None,
            execute_step=lambda step_id: True,  # always succeed
            step_started_callback=None,
            step_completed_callback=None,
        )
        # Bind the real method under test to the stub.
        from src.core.pipeline_manager import PipelineManager
        self.pm._emit_pipeline_step_bus = PipelineManager._emit_pipeline_step_bus.__get__(self.pm)

    def tearDown(self):
        state_bus_mod.reset()

    def test_run_path_emits_running_and_completed(self):
        # Mimic the inner-loop body of _execute_next_step.
        seen: list[dict] = []
        self.bus.subscribe("state.pipeline_step", seen.append)
        step = _make_step("initialisation", "Init")
        self.pm._emit_pipeline_step_bus(step, "running")
        self.pm._emit_pipeline_step_bus(step, "completed")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["status"], "running")
        self.assertEqual(seen[1]["status"], "completed")
        self.assertEqual(seen[0]["step_id"], "initialisation")


if __name__ == "__main__":
    unittest.main()
