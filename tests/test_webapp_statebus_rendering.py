"""Tests for webapp session-local AlimaStateBus rendering (Phase 4)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _MockCheckBox:
    def __init__(self, checked=True):
        self._checked = checked
        self.toggled = mock.MagicMock()

    def isChecked(self):
        return self._checked


class TestSessionBusSubscriber(unittest.TestCase):
    """Unit tests for _SessionBusSubscriber event bridging."""

    def setUp(self):
        from src.core.render_events import MockTransport
        from src.core.state_bus import AlimaStateBus, reset
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        from src.webapp.app import _SessionBusSubscriber

        reset()
        self.transport = MockTransport()
        checkbox = _MockCheckBox(checked=True)
        self.renderer = UnifiedMessageRenderer(self.transport, checkbox)
        self.sub = _SessionBusSubscriber(self.renderer)
        self.bus = AlimaStateBus()

    def tearDown(self):
        from src.core.state_bus import reset

        self.sub.unsubscribe()
        reset()

    def _types(self):
        return self.transport.types()

    def _of_type(self, type_):
        return self.transport.of_type(type_)

    def test_tool_called_and_result_render_collapsible(self):
        self.sub.subscribe()
        self.bus.emit_event("tool.called", {
            "id": "tc_abc123",
            "name": "gnd_search",
            "arguments": {"query": "Chemie"},
        })
        self.bus.emit_event("tool.result", {
            "id": "tc_abc123",
            "result": "found 3",
            "status": "ok",
        })

        self.assertIn("collapsible", self._types())
        self.assertIn("collapsible_update", self._types())
        update = self._of_type("collapsible_update")[0]
        self.assertIn("found 3", update["body"])

    def test_tool_result_urls_registered_as_trusted(self):
        # Claude Generated - webapp parity with the GUI: pre-formatted GND/catalog
        # URLs in a tool result must be trusted so they aren't flagged external.
        import json as _json
        self.sub.subscribe()
        self.bus.emit_event("tool.called", {
            "id": "tc_1", "name": "search_gnd", "arguments": {"term": "X"},
        })
        self.bus.emit_event("tool.result", {
            "id": "tc_1",
            "result": _json.dumps(
                {"entries": [{"url": "https://d-nb.info/gnd/4047979-1"}]}
            ),
            "status": "ok",
        })
        self.assertIn(
            "https://d-nb.info/gnd/4047979-1", self.renderer._trusted_urls
        )

    def test_build_session_renderer_wires_catalog_config(self):
        from unittest import mock as _mock
        from src.webapp.app import _build_session_renderer, Session
        from src.utils.config_models import AlimaConfig, PluginInstanceConfig

        # The OPAC base comes from the catalog *instance* since WP P7.
        cfg = AlimaConfig()
        cfg.plugins = [PluginInstanceConfig(
            "catalog", "search_provider", "catalog", enabled=True, is_primary=True,
            settings={"catalog_web_record_url": "https://katalog.ub.tu-freiberg.de/Record/"},
        )]
        cm = _mock.MagicMock()
        cm.load_config.return_value = cfg
        with _mock.patch(
            "src.utils.config_manager.ConfigManager", return_value=cm
        ):
            r = _build_session_renderer(Session("s1"))
        self.assertEqual(
            r._catalog_web_base, "https://katalog.ub.tu-freiberg.de/Record"
        )

    def test_state_pipeline_step_running_then_completed(self):
        self.sub.subscribe()
        self.bus.emit_event("state.pipeline_step", {
            "status": "running",
            "step_id": "verify_keywords",
            "name": "verify keywords",
        })
        self.bus.emit_event("state.pipeline_step", {
            "status": "completed",
            "step_id": "verify_keywords",
            "name": "verify keywords",
        })

        events = self._of_type("collapsible")
        self.assertEqual(len(events), 1)
        self.assertIn("pipeline.verify_keywords", events[0]["summary"])
        updates = self._of_type("collapsible_update")
        self.assertEqual(len(updates), 1)
        # Terminal status renders as a success icon (✓) plus status text.
        self.assertIn("✓", updates[0]["summary"])
        self.assertIn("completed: verify keywords", updates[0]["body"])

    def test_state_pipeline_step_error_carries_error_text_and_kind(self):
        # WP12 §9.3: payload["error"] must reach the block body and the
        # update must carry kind="error" for the red chrome.
        self.sub.subscribe()
        self.bus.emit_event("state.pipeline_step", {
            "status": "running",
            "step_id": "classification",
            "name": "classification",
        })
        self.bus.emit_event("state.pipeline_step", {
            "status": "error",
            "step_id": "classification",
            "name": "classification",
            "error": "LLM timeout after 30s",
        })

        updates = self._of_type("collapsible_update")
        self.assertEqual(len(updates), 1)
        self.assertIn("✗", updates[0]["summary"])
        self.assertIn("LLM timeout after 30s", updates[0]["body"])
        self.assertEqual(updates[0].get("kind"), "error")

    def test_state_pipeline_prompt_renders_collapsible(self):
        self.sub.subscribe()
        self.bus.emit_event("state.pipeline_prompt", {
            "prompt_id": "p1",
            "step_id": "keywords",
            "system": "sys",
            "user": "usr",
            "timestamp": "2026-06-17T10:00:00",
            "provider": "ollama",
            "model": "llama3",
        })
        self.bus.emit_event("state.pipeline_prompt_done", {
            "prompt_id": "p1",
            "duration_s": 1.25,
        })

        events = self._of_type("collapsible")
        self.assertEqual(len(events), 1)
        self.assertIn("Input 'keywords'", events[0]["summary"])
        self.assertIn("SYSTEM", events[0]["body"])
        self.assertIn("USER", events[0]["body"])
        updates = self._of_type("collapsible_update")
        self.assertEqual(len(updates), 1)
        self.assertIn("⏱ 1.2s", updates[0]["summary"])

    def test_state_pipeline_completed_renders_system_message(self):
        self.sub.subscribe()
        self.bus.emit_event("state.pipeline_completed", {"workflow": "alima_v51"})

        blocks = self._of_type("block")
        self.assertTrue(any("Pipeline abgeschlossen" in e.get("html", "") for e in blocks))

    def test_unsubscribe_removes_handlers(self):
        self.sub.subscribe()
        self.sub.unsubscribe()
        # After unsubscribe the same events must not reach the renderer.
        before = len(self.transport.events)
        self.bus.emit_event("tool.called", {"id": "x", "name": "t", "arguments": {}})
        self.assertEqual(len(self.transport.events), before)


class TestAgenticAnalysisBusRendering(unittest.TestCase):
    """Integration: run_analysis subscribes and renders bus events."""

    def _client(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod

        return TestClient(appmod.app), appmod

    def test_agentic_analysis_emits_bus_tool_call_into_render_buffer(self):
        client, appmod = self._client()
        sid = "test-agentic-bus"
        appmod.sessions[sid] = appmod.Session(sid)

        from src.core.state_bus import AlimaStateBus, reset

        reset()
        try:
            from src.core.state_bus import AlimaStateBus

            fake_pm = self._fake_pipeline_manager()

            # Force the state bus to use direct dispatch (webapp has no Qt event loop),
            # otherwise a previously-created QApplication in the test process could
            # cause queued signals to be lost.
            from src.core.state_bus import _AlimaStateBus
            original_emit = _AlimaStateBus.emit_event

            def _direct_emit(self, event_type, diff):
                for event_filter, handler, _slot in self._subscriptions:
                    if event_filter != event_type:
                        continue
                    try:
                        handler(diff)
                    except Exception as exc:  # noqa: BLE001
                        pass

            _AlimaStateBus.emit_event = _direct_emit
            try:
                # Monkeypatch the PipelineManager name in the analysis router (where
                # run_analysis now lives) directly; mock.patch on the origin module can
                # miss because the module bound the name at import time and the
                # singleton may already be alive.
                from src.webapp.routers import analysis as analysismod
                original_pm = analysismod.PipelineManager
                analysismod.PipelineManager = lambda *a, **k: fake_pm
                with mock.patch("src.webapp.routers.analysis.resolve_input_to_text", return_value="text"), \
                     mock.patch("src.webapp.routers.analysis.AppContext") as mock_ctx:
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
                analysismod.PipelineManager = original_pm
                self.assertEqual(resp.status_code, 200)

                # Wait for the background thread to finish.
                import time
                for _ in range(100):
                    if appmod.sessions[sid].status != "running":
                        break
                    time.sleep(0.05)

                # The inline tool.called emitted inside start_pipeline should have
                # been rendered into the session buffer while the subscriber was active.
                render_types = [e.get("type") for e in appmod.sessions[sid].render_buffer]
                self.assertIn("collapsible", render_types)
            finally:
                _AlimaStateBus.emit_event = original_emit
        finally:
            appmod.sessions.pop(sid, None)
            reset()

    def test_classic_analysis_streams_llm_tokens_as_stream_blocks(self):
        # Chat-UX 5/9: classic-pipeline LLM tokens render as shared #log
        # stream blocks (stream_open/token/close) instead of raw
        # streaming_tokens frames.
        client, appmod = self._client()
        sid = "test-classic-stream"
        appmod.sessions[sid] = appmod.Session(sid)

        from src.core.state_bus import reset

        reset()
        try:
            from types import SimpleNamespace

            fake_pm = self._fake_pipeline_manager()

            def start_pipeline(text, input_type=None, input_source=None):
                cb = fake_pm._cb
                for token in ("Hallo ", "**Welt**"):
                    cb["stream_callback"](token, "keywords")
                cb["step_completed"](SimpleNamespace(step_id="keywords", output_data=None))
                if cb.get("pipeline_completed"):
                    cb["pipeline_completed"](None)
                return "fake-pipeline-id"

            fake_pm.start_pipeline = start_pipeline

            from src.webapp.routers import analysis as analysismod
            original_pm = analysismod.PipelineManager
            analysismod.PipelineManager = lambda *a, **k: fake_pm
            try:
                with mock.patch("src.webapp.routers.analysis.resolve_input_to_text", return_value="text"), \
                     mock.patch("src.webapp.routers.analysis.AppContext") as mock_ctx:
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
            finally:
                analysismod.PipelineManager = original_pm
            self.assertEqual(resp.status_code, 200)

            import time
            for _ in range(100):
                if appmod.sessions[sid].status != "running":
                    break
                time.sleep(0.05)

            buffer = appmod.sessions[sid].render_buffer
            types = [e.get("type") for e in buffer]
            self.assertIn("stream_open", types)
            self.assertIn("stream_token", types)
            self.assertIn("stream_close", types)
            tokens = "".join(
                e.get("text", "") for e in buffer if e.get("type") == "stream_token"
            )
            self.assertEqual(tokens, "Hallo **Welt**")
        finally:
            appmod.sessions.pop(sid, None)
            reset()

    def _fake_pipeline_manager(self):
        from types import SimpleNamespace

        config = SimpleNamespace(
            enable_agentic_mode=False,
            workflow_name=None,
            global_provider_override=None,
            global_model_override=None,
        )

        class FakePM:
            def __init__(self, *a, **k):
                self.config = config
                self.current_analysis_state = None

            def set_config(self, cfg):
                self.config = cfg

            def set_callbacks(self, **cb):
                self._cb = cb

            def set_interrupt_flag(self, *a, **k):
                pass

            def start_pipeline(self, text, input_type=None, input_source=None):
                # Emit a bus tool call while subscribed, then finish.
                from src.core.state_bus import AlimaStateBus

                bus = AlimaStateBus()
                bus.emit_event("tool.called", {
                    "id": "tc_inline123",
                    "name": "inline_tool",
                    "arguments": {"q": "x"},
                })
                if self._cb.get("pipeline_completed"):
                    self._cb["pipeline_completed"](None)
                return "fake-pipeline-id"

        return FakePM()


if __name__ == "__main__":
    unittest.main()
