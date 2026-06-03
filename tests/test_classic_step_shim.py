"""Tests for the classic-pipeline tool-call shim (P-C). Claude Generated.

Covers ``_run_classic_step`` and the helper emit functions in
:mod:`src.utils.pipeline_utils`. The shim wraps each step in
``execute_complete_pipeline`` so the chat log shows classic-pipeline work
identically to agentic-pipeline tool calls.

These tests do NOT exercise the full ``execute_complete_pipeline`` flow
(the existing ``test_pipeline_utils.py`` is broken — see Pre-existing
Failures in the task list). They cover the helper in isolation and
verify the bus emission contract.
"""
from __future__ import annotations

import unittest

from PyQt6.QtCore import QCoreApplication

# pyqtSignal needs a QApplication before any bus-based test runs.
_qapp = QCoreApplication.instance() or QCoreApplication([])

from src.core import state_bus as state_bus_mod
from src.utils.pipeline_utils import (
    _emit_classic_tool_call,
    _emit_classic_tool_result,
    _run_classic_step,
)


class TestClassicStepShim(unittest.TestCase):
    """P-C: classic-pipeline steps emit ``tool.called`` + ``tool.result``."""

    def setUp(self):
        state_bus_mod.reset()
        self.bus = state_bus_mod.AlimaStateBus()

    def tearDown(self):
        state_bus_mod.reset()

    def test_run_classic_step_emits_pair(self):
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        def step_fn(x, y=0):
            return x + y

        out = _run_classic_step("initialisation", {"task": "init"}, step_fn, 2, y=3)
        self.assertEqual(out, 5)
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["name"], "classic.initialisation")
        self.assertEqual(called[0]["arguments"], {"task": "init"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], called[0]["id"])
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["name"], "classic.initialisation")

    def test_run_classic_step_error_emits_error_status(self):
        called: list[dict] = []
        results: list[dict] = []
        self.bus.subscribe("tool.called", called.append)
        self.bus.subscribe("tool.result", results.append)

        def step_fn():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            _run_classic_step("keywords", {"x": 1}, step_fn)

        self.assertEqual(len(called), 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["id"], called[0]["id"])

    def test_emit_classic_tool_call_format(self):
        tc_id = _emit_classic_tool_call("search", {"keywords_count": 3})
        self.assertTrue(tc_id.startswith("tc_"))

    def test_emit_classic_tool_result_uses_same_id(self):
        results: list[dict] = []
        self.bus.subscribe("tool.result", results.append)
        _emit_classic_tool_result("tc_abc", "dk_search", "ok")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "tc_abc")
        self.assertEqual(results[0]["name"], "classic.dk_search")


if __name__ == "__main__":
    unittest.main()
