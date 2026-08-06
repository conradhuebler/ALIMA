"""Webapp pipeline run: the log must show the run, not only its result.

Claude Generated. Regression for the operator report "man sieht nix von der
pipeline": bus-driven chrome (pipeline started / per-step collapsibles) never
reached the session render buffer because the events were queued for a Qt loop
the webapp never runs. Drives ``run_analysis`` with a stub PipelineManager that
emits on the bus from a worker thread, exactly like the real one.
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication

from src.core import state_bus as sb


class _FakeStep(SimpleNamespace):
    pass


class _FakePipelineManager:
    """Mimics the real manager: bus events + direct callbacks, off-thread."""

    instances: list = []

    def __init__(self, *a, **k):
        self.current_analysis_state = SimpleNamespace(
            final_llm_analysis=SimpleNamespace(
                extracted_gnd_keywords=["Cadmium"],
                response_full_text="",
            ),
            dk_classifications=None,
            report_markdown="",
            working_title="Testwerk",
        )
        self.config = None
        self._cb = {}
        _FakePipelineManager.instances.append(self)

    def set_config(self, config):
        self.config = config

    def set_callbacks(self, **kwargs):
        self._cb = kwargs

    def set_interrupt_flag(self, *a, **k):
        pass

    def start_pipeline(self, input_text, input_type="text", input_source=None):
        from src.core.state_bus import AlimaStateBus

        bus = AlimaStateBus()
        bus.emit_event("state.pipeline_started", {"pipeline_id": "abcdef123456"})
        for step_id, name in (("initialisation", "Initialisierung"), ("keywords", "Keywords")):
            step = _FakeStep(step_id=step_id, name=name, status="running", output_data=None)
            bus.emit_event(
                "state.pipeline_step",
                {"tool": "execute_complete_pipeline", "step_id": step_id,
                 "name": name, "status": "running"},
            )
            if self._cb.get("step_started"):
                self._cb["step_started"](step)
            if self._cb.get("stream_callback"):
                self._cb["stream_callback"]("Token ", step_id)
            step.status = "completed"
            bus.emit_event(
                "state.pipeline_step",
                {"tool": "execute_complete_pipeline", "step_id": step_id,
                 "name": name, "status": "completed"},
            )
            if self._cb.get("step_completed"):
                self._cb["step_completed"](step)
        if self._cb.get("pipeline_completed"):
            self._cb["pipeline_completed"](self.current_analysis_state)
        return "pipeline-1"


class TestWebappRunChrome(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # QCoreApplication without exec(): DatabaseManager creates one in the
        # real process, which is what made queued bus events disappear.
        self._app = QCoreApplication.instance() or QCoreApplication(sys.argv)
        sb.reset()
        sb.set_direct_dispatch(True)
        _FakePipelineManager.instances.clear()

    def tearDown(self):
        sb.set_direct_dispatch(False)
        sb.reset()

    async def _run(self, workflow: str):
        from src.webapp import app as appmod
        from src.webapp.routers import analysis as anmod

        sid = f"test-run-chrome-{workflow}"
        session = appmod.Session(sid)
        appmod.sessions[sid] = session
        services = {
            "config_manager": mock.MagicMock(),
            "alima_manager": mock.MagicMock(),
            "cache_manager": mock.MagicMock(),
            "llm_service": mock.MagicMock(),
            "prompt_service": mock.MagicMock(),
            "pipeline_manager": mock.MagicMock(),
        }
        try:
            with mock.patch.object(anmod.AppContext, "get_services", return_value=services), \
                 mock.patch.object(anmod, "PipelineManager", _FakePipelineManager), \
                 mock.patch.object(anmod, "_prepare_results_for_export", return_value={}), \
                 mock.patch.object(anmod, "_extract_results_from_analysis_state", return_value={}), \
                 mock.patch.object(anmod, "_autosave_session_state", lambda *a, **k: None):
                await anmod.run_analysis(
                    sid, "text", "Ein Text über Cadmium.", None, None,
                    workflow=workflow,
                )
            return list(session.render_buffer)
        finally:
            appmod.sessions.pop(sid, None)

    def _kinds(self, events):
        return [e["type"] for e in events]

    async def test_classic_run_renders_step_chrome_and_result(self):
        events = await self._run("__classic__")
        types = self._kinds(events)
        # Per-step collapsibles from the bus bridge.
        self.assertGreaterEqual(types.count("collapsible"), 2)
        # Live LLM stream blocks (classic mirrors tokens into #log).
        self.assertIn("stream_open", types)
        self.assertIn("stream_token", types)
        # And the final result blocks.
        html = "\n".join(e.get("html", "") for e in events if e["type"] == "block")
        self.assertIn("Pipeline vollständig abgeschlossen", html)
        self.assertIn("GND-Schlagworte", html)

    async def test_agentic_run_renders_step_chrome(self):
        events = await self._run("alima_classic")
        types = self._kinds(events)
        # Agentic gets no token mirror by design — the bus chrome is all it has,
        # so losing it left the log empty until completion.
        self.assertGreaterEqual(types.count("collapsible"), 2)
        html = "\n".join(e.get("html", "") for e in events if e["type"] == "block")
        self.assertIn("Pipeline vollständig abgeschlossen", html)

    async def test_without_direct_dispatch_step_chrome_disappears(self):
        """Pins the failure mode: queued delivery drops every step block, so the
        log holds only what the direct callbacks rendered."""
        sb.set_direct_dispatch(False)
        events = await self._run("alima_classic")
        self.assertEqual(self._kinds(events).count("collapsible"), 0)
        html = "\n".join(e.get("html", "") for e in events if e["type"] == "block")
        self.assertIn("Pipeline vollständig abgeschlossen", html)  # result survives


if __name__ == "__main__":
    unittest.main()
