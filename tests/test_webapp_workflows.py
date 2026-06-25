"""Tests for webapp workflow/agentic support - Claude Generated."""
from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestWorkflowDiscovery(unittest.TestCase):
    """Unit tests for _discover_workflows / GET /api/workflows."""

    def test_discover_workflows_separates_root_and_legacy(self):
        from src.webapp.app import _discover_workflows

        with mock.patch("src.webapp.app.DEFAULT_SEARCH_PATHS", [mock.MagicMock()]):
            # Set up a fake directory tree
            base = mock.MagicMock()
            base.exists.return_value = True
            base.is_dir.return_value = True
            base.glob.side_effect = lambda pattern: {
                "*.yaml": [
                    mock.MagicMock(resolve=lambda: "wf1", stem="alima_v51", exists=lambda: True),
                    mock.MagicMock(resolve=lambda: "wf2", stem="alima", exists=lambda: True),
                ],
            }.get(pattern, [])
            (base / "legacy").is_dir.return_value = True
            (base / "legacy").glob.side_effect = lambda pattern: {
                "*.yaml": [
                    mock.MagicMock(resolve=lambda: "wf3", stem="old", exists=lambda: True),
                ],
            }.get(pattern, [])

            # Replace the first search path with our fake base
            with mock.patch("src.webapp.app.DEFAULT_SEARCH_PATHS", [base]):
                # The fake paths cannot be opened; mock both open() and yaml parsing.
                with mock.patch("builtins.open", mock.mock_open()), \
                     mock.patch("src.webapp.app.yaml.safe_load", return_value={"version": "5.1", "steps": []}):
                    root, legacy, steps_by_stem = _discover_workflows()

        self.assertIn("alima_v51", root)
        self.assertIn("alima", root)
        self.assertIn("old", legacy)
        self.assertIn("alima_v51", steps_by_stem)
        self.assertIn("alima", steps_by_stem)

    def test_workflows_endpoint_returns_list(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod

        client = TestClient(appmod.app)
        with mock.patch.object(appmod, "_discover_workflows", return_value=({
            "alima_v51": "5.1",
            "alima": "5.0",
        }, {"old": "4.0"}, {
            "alima_v51": [{"id": "extraction", "label": "Extraction"}],
            "alima": [],
            "old": [],
        })):
            resp = client.get("/api/workflows")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        values = [item["value"] for item in data if not item["value"].startswith("__")]
        self.assertIn("alima_v51", values)
        self.assertIn("__classic__", [item["value"] for item in data])
        # separator exists if legacy workflows present
        self.assertIn("__separator__", [item["value"] for item in data])


class TestAgenticAnalyzeParameter(unittest.TestCase):
    """Unit tests for POST /api/analyze/{id} workflow parameter wiring."""

    def _client(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod
        return TestClient(appmod.app), appmod

    def test_analyze_with_classic_workflow_does_not_enable_agentic(self):
        client, appmod = self._client()
        sid = "test-classic"
        appmod.sessions[sid] = appmod.Session(sid)
        try:
            fake_pm = self._fake_pipeline_manager(agentic=False, workflow=None)
            with mock.patch("src.webapp.app.PipelineManager", return_value=fake_pm), \
                 mock.patch("src.webapp.app.resolve_input_to_text", return_value="text"), \
                 mock.patch("src.webapp.app.AppContext") as mock_ctx:
                mock_ctx.return_value.get_services.return_value = {
                    "config_manager": mock.MagicMock(),
                    "alima_manager": mock.MagicMock(),
                    "cache_manager": mock.MagicMock(),
                    "llm_service": mock.MagicMock(),
                    "prompt_service": mock.MagicMock(),
                    "pipeline_manager": fake_pm,
                }
                resp = client.post(
                    f"/api/analyze/{sid}",
                    data={"input_type": "text", "content": "abc", "workflow": "__classic__"},
                )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "started")
            self.assertFalse(fake_pm.config.enable_agentic_mode)
            self.assertIsNone(fake_pm.config.workflow_name)
        finally:
            appmod.sessions.pop(sid, None)

    def test_analyze_with_yaml_workflow_enables_agentic(self):
        client, appmod = self._client()
        sid = "test-agentic"
        appmod.sessions[sid] = appmod.Session(sid)
        try:
            fake_pm = self._fake_pipeline_manager(agentic=False, workflow=None)
            with mock.patch("src.webapp.app.PipelineManager", return_value=fake_pm), \
                 mock.patch("src.webapp.app.resolve_input_to_text", return_value="text"), \
                 mock.patch("src.webapp.app.AppContext") as mock_ctx:
                mock_ctx.return_value.get_services.return_value = {
                    "config_manager": mock.MagicMock(),
                    "alima_manager": mock.MagicMock(),
                    "cache_manager": mock.MagicMock(),
                    "llm_service": mock.MagicMock(),
                    "prompt_service": mock.MagicMock(),
                    "pipeline_manager": fake_pm,
                }
                resp = client.post(
                    f"/api/analyze/{sid}",
                    data={"input_type": "text", "content": "abc", "workflow": "alima_v51"},
                )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "started")
            self.assertTrue(fake_pm.config.enable_agentic_mode)
            self.assertEqual(fake_pm.config.workflow_name, "alima_v51")
        finally:
            appmod.sessions.pop(sid, None)

    def _fake_pipeline_manager(self, agentic: bool, workflow: str | None):
        from types import SimpleNamespace

        config = SimpleNamespace(
            enable_agentic_mode=agentic,
            workflow_name=workflow,
            global_provider_override=None,
            global_model_override=None,
        )

        class FakePM:
            def __init__(self, *a, **k):
                self.config = config
                self.current_analysis_state = None

            def set_config(self, cfg):
                # Capture the config object that run_analysis actually configured
                # (with enable_agentic_mode / workflow_name set on it).
                self.config = cfg

            def set_callbacks(self, **cb):
                self._cb = cb

            def set_interrupt_flag(self, *a, **k):
                pass

            def start_pipeline(self, text, input_type=None, input_source=None):
                # Fire completion immediately so the task ends
                if self._cb.get("pipeline_completed"):
                    self._cb["pipeline_completed"](None)
                return "fake-pipeline-id"

        return FakePM()


if __name__ == "__main__":
    unittest.main()
