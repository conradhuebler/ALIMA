"""Tests for webapp session chat-agent endpoint - Claude Generated."""
from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestSessionChatEndpoint(unittest.IsolatedAsyncioTestCase):
    """Unit tests for POST /api/session/{id}/chat."""

    def _client(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod
        return TestClient(appmod.app), appmod

    async def test_chat_endpoint_requires_message(self):
        client, appmod = self._client()
        sid = "test-chat-empty"
        appmod.sessions[sid] = appmod.Session(sid)
        try:
            resp = client.post(
                f"/api/session/{sid}/chat",
                json={"message": "  "},
            )
            self.assertEqual(resp.status_code, 400)
        finally:
            appmod.sessions.pop(sid, None)

    async def test_chat_runner_uses_session_last_provider_model_as_fallback(self):
        """When the request omits provider/model, the runner falls back to the
        session's last effective pipeline provider/model. - Claude Generated"""
        client, appmod = self._client()
        sid = "test-chat-fallback"
        session = appmod.Session(sid)
        session.last_provider = "ollama"
        session.last_model = "llama3"
        appmod.sessions[sid] = session

        try:
            req = appmod.ChatMessageRequest(message="hi")
            services = {
                "config_manager": mock.MagicMock(),
                "alima_manager": mock.MagicMock(),
                "cache_manager": mock.MagicMock(),
                "llm_service": mock.MagicMock(),
                "prompt_service": mock.MagicMock(),
                "pipeline_manager": mock.MagicMock(),
            }
            # No configured chat default → fall through to session.last_*.
            # (A real ChatConfig has empty-string defaults; a bare MagicMock
            # would look like a configured default and short-circuit.)
            chat_cfg = mock.MagicMock()
            chat_cfg.default_provider = ""
            chat_cfg.default_model = ""
            services["config_manager"].load_config.return_value.chat_config = chat_cfg
            services["config_manager"].get_unified_config.return_value.get_enabled_providers.return_value = []

            captured = {}

            def fake_resolve(provider, model, **kwargs):
                captured["provider_arg"] = provider
                captured["model_arg"] = model
                return ("resolved-provider", "resolved-model")

            with mock.patch.object(appmod.AppContext, "get_services", return_value=services), \
                 mock.patch("src.webapp.routers.agent.PipelineManager", return_value=mock.MagicMock()), \
                 mock.patch("src.core.headless_agent.HeadlessAgentRunner", return_value=mock.MagicMock()), \
                 mock.patch("src.core.headless_agent.resolve_provider_model", side_effect=fake_resolve):
                runner, pm, provider, model = appmod._build_session_agent_runner(session, req)

            self.assertEqual(captured.get("provider_arg"), "ollama")
            self.assertEqual(captured.get("model_arg"), "llama3")
            self.assertEqual(provider, "resolved-provider")
            self.assertEqual(model, "resolved-model")
        finally:
            appmod.sessions.pop(sid, None)

    async def test_chat_endpoint_starts_turn_and_appends_history(self):
        client, appmod = self._client()
        sid = "test-chat-turn"
        appmod.sessions[sid] = appmod.Session(sid)
        try:
            fake_result = mock.MagicMock()
            fake_result.content = "Hallo"
            fake_result.iterations = 1
            fake_result.tool_log = []

            class FakeRunner:
                def __init__(self, *a, **k):
                    pass

                def run(self, *a, **k):
                    # Capture callbacks and emit a token + tool call so the renderer
                    # writes into the session buffer.
                    on_token = k.get("on_token")
                    on_tool_call = k.get("on_tool_call")
                    on_tool_result = k.get("on_tool_result")
                    if on_token:
                        on_token("Hallo")
                    if on_tool_call:
                        tc = mock.MagicMock()
                        tc.name = "test_tool"
                        tc.arguments = {"x": 1}
                        on_tool_call(tc)
                    if on_tool_result:
                        on_tool_result("test_tool", "ok")
                    return fake_result

            services = {
                "config_manager": mock.MagicMock(),
                "alima_manager": mock.MagicMock(),
                "cache_manager": mock.MagicMock(),
                "llm_service": mock.MagicMock(),
                "prompt_service": mock.MagicMock(),
                "pipeline_manager": mock.MagicMock(),
            }
            services["config_manager"].load_config.return_value.chat_config = mock.MagicMock()
            services["config_manager"].get_unified_config.return_value.get_enabled_providers.return_value = []

            with mock.patch.object(appmod.AppContext, "get_services", return_value=services), \
                 mock.patch("src.core.headless_agent.HeadlessAgentRunner", FakeRunner), \
                 mock.patch("src.core.headless_agent.resolve_provider_model", return_value=("test", "model")):
                resp = client.post(
                    f"/api/session/{sid}/chat",
                    json={"message": "Wie geht es?"},
                )

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "chat_started")
            self.assertEqual(data["provider"], "test")
            self.assertEqual(data["model"], "model")

            # User message persisted before the runner started.
            self.assertEqual(
                appmod.sessions[sid].chat_history[0],
                {"role": "user", "content": "Wie geht es?"},
            )
        finally:
            appmod.sessions.pop(sid, None)


    async def test_chat_config_default_beats_session_last_provider(self):
        """ChatConfig.default_provider/model (config-only webapp setting) takes
        precedence over the session's last pipeline provider/model. - Claude Generated"""
        client, appmod = self._client()
        sid = "test-chat-default"
        session = appmod.Session(sid)
        session.last_provider = "ollama"
        session.last_model = "llama3"
        appmod.sessions[sid] = session

        try:
            req = appmod.ChatMessageRequest(message="hi")
            services = {
                "config_manager": mock.MagicMock(),
                "alima_manager": mock.MagicMock(),
                "cache_manager": mock.MagicMock(),
                "llm_service": mock.MagicMock(),
                "prompt_service": mock.MagicMock(),
                "pipeline_manager": mock.MagicMock(),
            }
            chat_cfg = mock.MagicMock()
            chat_cfg.default_provider = "LLMachine"
            chat_cfg.default_model = "north-mini-code-1.0:latest"
            services["config_manager"].load_config.return_value.chat_config = chat_cfg
            services["config_manager"].get_unified_config.return_value.get_enabled_providers.return_value = []

            captured = {}

            def fake_resolve(provider, model, **kwargs):
                captured["provider_arg"] = provider
                captured["model_arg"] = model
                return (provider, model)

            with mock.patch.object(appmod.AppContext, "get_services", return_value=services), \
                 mock.patch("src.webapp.routers.agent.PipelineManager", return_value=mock.MagicMock()), \
                 mock.patch("src.core.headless_agent.HeadlessAgentRunner", return_value=mock.MagicMock()), \
                 mock.patch("src.core.headless_agent.resolve_provider_model", side_effect=fake_resolve):
                runner, pm, provider, model = appmod._build_session_agent_runner(session, req)

            self.assertEqual(captured.get("provider_arg"), "LLMachine")
            self.assertEqual(captured.get("model_arg"), "north-mini-code-1.0:latest")
        finally:
            appmod.sessions.pop(sid, None)

    async def test_cancel_stops_running_chat_thread(self):
        # /cancel must abort a running chat turn (parity with /agent/run and the
        # GUI ChatAgentWorker), not only set the pipeline flag. - Claude Generated
        client, appmod = self._client()
        sid = "test-chat-cancel"
        session = appmod.Session(sid)
        session.status = "running"
        session.chat_thread = mock.MagicMock()
        appmod.sessions[sid] = session
        try:
            resp = client.post(f"/api/session/{sid}/cancel")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "cancel_requested")
            self.assertTrue(session.abort_requested)
            session.chat_thread.request_stop.assert_called_once()
        finally:
            appmod.sessions.pop(sid, None)

    async def test_cancel_when_idle_does_not_stop(self):
        client, appmod = self._client()
        sid = "test-chat-cancel-idle"
        session = appmod.Session(sid)
        session.status = "idle"
        session.chat_thread = mock.MagicMock()
        appmod.sessions[sid] = session
        try:
            resp = client.post(f"/api/session/{sid}/cancel")
            self.assertEqual(resp.status_code, 200)
            session.chat_thread.request_stop.assert_not_called()
        finally:
            appmod.sessions.pop(sid, None)


if __name__ == "__main__":
    unittest.main()
