"""WP12 — webapp render-event mechanics + WebSocket broadcast. Claude Generated.

Covers the server half of the unified render layer in ``src/webapp/app.py``:
the per-session append-only render buffer (with monotonic ``seq`` for client
dedup), the cursors used by the WS (replay) and polling clients, the
``WebSocketRenderTransport``, the shared ``UnifiedMessageRenderer`` producer,
and the WS ``complete`` broadcast.

No pipeline / LLM is run: render events are injected directly and the session
status is flipped, so the WS handler immediately emits its terminal message.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestSessionRenderBuffer(unittest.TestCase):
    def setUp(self):
        from src.webapp.app import Session
        self.Session = Session
        self.s = Session("rbuf")

    def test_seq_is_monotonic(self):
        self.s.append_render_event({"type": "block", "html": "a"})
        self.s.append_render_event({"type": "block", "html": "b"})
        self.assertEqual([e["seq"] for e in self.s.render_buffer], [0, 1])

    def test_get_since_returns_tail_and_length(self):
        for i in range(3):
            self.s.append_render_event({"type": "block", "html": str(i)})
        tail, n = self.s.get_render_events_since(1)
        self.assertEqual(n, 3)
        self.assertEqual([e["html"] for e in tail], ["1", "2"])

    def test_polling_cursor_only_returns_new(self):
        self.s.append_render_event({"type": "block", "html": "x"})
        first = self.s.get_new_render_events()
        self.assertEqual(len(first), 1)
        self.assertEqual(self.s.get_new_render_events(), [])  # nothing new
        self.s.append_render_event({"type": "block", "html": "y"})
        self.assertEqual(len(self.s.get_new_render_events()), 1)

    def test_clear_resets_buffer_and_cursor(self):
        self.s.append_render_event({"type": "block", "html": "x"})
        self.s.get_new_render_events()
        self.s.clear()
        self.assertEqual(self.s.render_buffer, [])
        self.assertEqual(self.s.render_buffer_sent_count, 0)
        # New events after clear start at seq 0 again.
        self.s.append_render_event({"type": "block", "html": "z"})
        self.assertEqual(self.s.render_buffer[0]["seq"], 0)


class TestWebSocketRenderTransportAndProducer(unittest.TestCase):
    def test_transport_send_appends_to_session(self):
        from src.webapp.app import Session, WebSocketRenderTransport
        s = Session("tx")
        t = WebSocketRenderTransport(s)
        t.send({"type": "block", "html": "hi"})
        t.set_autoscroll(True)   # control no-ops must not raise
        t.scroll_to_bottom()
        self.assertEqual(len(s.render_buffer), 1)
        self.assertEqual(s.render_buffer[0]["html"], "hi")

    def test_session_renderer_emits_html_block(self):
        from src.webapp.app import Session, _build_session_renderer
        s = Session("prod")
        r = _build_session_renderer(s)
        r.render_html_block("<div>#1 DK 614.7</div>", kind="dk_classifications",
                            plain_text="614.7")
        self.assertEqual(len(s.render_buffer), 1)
        ev = s.render_buffer[0]
        self.assertEqual(ev["type"], "block")
        self.assertEqual(ev["kind"], "html_block")
        self.assertIn("614.7", ev["html"])
        self.assertEqual(ev["seq"], 0)

    def test_dk_cards_via_shared_formatter(self):
        from src.webapp.app import Session, _build_session_renderer
        from src.utils.pipeline_utils import PipelineResultFormatter
        s = Session("cards")
        r = _build_session_renderer(s)
        out = {"dk_search_results_flattened": [
            {"dk": "614.7", "count": 12, "titles": ["T1", "T2"],
             "keywords": ["Umwelt"], "classification_type": "DK"},
        ]}
        html, plain = PipelineResultFormatter.format_dk_search_card_html(out)
        self.assertTrue(html)
        r.render_html_block(html, kind="dk_search", plain_text=plain)
        self.assertIn("Katalog-Recherche", s.render_buffer[0]["html"])


class TestWebSocketBroadcast(unittest.TestCase):
    """End-to-end: a completed session's render buffer is delivered over the WS."""

    def _client(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod
        return TestClient(appmod.app), appmod

    def test_complete_message_includes_render_events_with_seq(self):
        client, appmod = self._client()
        sid = "ws-complete"
        sess = appmod.Session(sid)
        appmod.sessions[sid] = sess
        try:
            # Inject chrome events, then mark the session complete so the WS
            # handler emits its terminal message on the first iteration.
            sess.append_render_event({"type": "block", "html": "<div>card</div>",
                                      "kind": "html_block"})
            sess.status = "completed"
            sess.current_step = "classification"

            with client.websocket_connect(f"/ws/{sid}") as ws:
                msg = ws.receive_json()
                self.assertEqual(msg["type"], "complete")
                self.assertIn("render_events", msg)
                evs = msg["render_events"]
                self.assertEqual(len(evs), 1)
                self.assertEqual(evs[0]["type"], "block")
                self.assertEqual(evs[0]["seq"], 0)
                self.assertIn("card", evs[0]["html"])
        finally:
            appmod.sessions.pop(sid, None)

    def test_reconnect_replays_full_buffer(self):
        client, appmod = self._client()
        sid = "ws-replay"
        sess = appmod.Session(sid)
        appmod.sessions[sid] = sess
        try:
            sess.append_render_event({"type": "block", "html": "a", "kind": "html_block"})
            sess.append_render_event({"type": "block", "html": "b", "kind": "html_block"})
            sess.status = "completed"

            # A fresh connection (cursor starts at 0) replays the whole buffer.
            for _ in range(2):
                with client.websocket_connect(f"/ws/{sid}") as ws:
                    msg = ws.receive_json()
                    self.assertEqual([e["seq"] for e in msg["render_events"]], [0, 1])
        finally:
            appmod.sessions.pop(sid, None)


class TestRunAnalysisEmitsCards(unittest.IsolatedAsyncioTestCase):
    """Drive the *real* run_analysis callback wiring (heavy deps mocked) to prove
    the DK/GND cards reach the session render buffer during a run."""

    async def test_dk_cards_emitted_through_callbacks(self):
        import types
        from unittest import mock
        from src.webapp import app as appmod

        sid = "run-emit"
        appmod.sessions[sid] = appmod.Session(sid)

        dk_flat = [{
            "dk": "614.7", "count": 10, "titles": ["T1", "T2"],
            "keywords": ["Umwelt"], "classification_type": "DK",
        }]
        dk_step = types.SimpleNamespace(
            step_id="dk_search", name="dk_search",
            output_data={"dk_search_results_flattened": dk_flat},
        )

        class FakeState:
            working_title = "Test"
            dk_classifications = ["614.7"]
            dk_search_results = dk_flat
            dk_search_results_flattened = dk_flat
            final_llm_analysis = None
            rvk_provenance = {}
        state = FakeState()

        class FakePM:
            def __init__(self, *a, **k):
                self.current_analysis_state = state

            def set_config(self, _c):
                pass

            def set_callbacks(self, **cb):
                self._cb = cb

            def set_interrupt_flag(self, *a, **k):
                pass

            def start_pipeline(self, text, input_type=None, input_source=None):
                self._cb["step_started"](dk_step)
                self._cb["step_completed"](dk_step)
                self._cb["pipeline_completed"](state)
                return "pid-1"

        services = {k: mock.MagicMock() for k in (
            "config_manager", "alima_manager", "cache_manager",
            "llm_service", "prompt_service", "pipeline_manager")}

        with mock.patch.object(appmod.AppContext, "get_services", return_value=services), \
             mock.patch.object(appmod.PipelineConfig, "create_from_provider_preferences",
                               return_value=mock.MagicMock()), \
             mock.patch("src.webapp.routers.analysis.PipelineManager", FakePM), \
             mock.patch("src.webapp.routers.analysis._autosave_session_state"):
            await appmod.run_analysis(sid, "text", "an abstract about environment", None, None)

        buf = appmod.sessions[sid].render_buffer
        htmls = " ".join(e.get("html", "") for e in buf)
        appmod.sessions.pop(sid, None)

        self.assertTrue(buf, "no render events were emitted during the run")
        self.assertIn("Katalog-Recherche", htmls)   # dk_search card
        self.assertIn("614.7", htmls)               # dk_classifications card
        # Every emitted event is a block with monotonic seq.
        self.assertEqual([e["seq"] for e in buf], list(range(len(buf))))


if __name__ == "__main__":
    unittest.main()
