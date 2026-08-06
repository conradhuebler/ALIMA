"""Rate-limit notices reach every log surface. Claude Generated.

A 429 wait used to be pushed into ``stream_callback``, i.e. rendered as part of
the model's answer — and in the webapp's agentic mode, where tokens are only
buffered, it was invisible entirely. It now goes on the state bus, which both
frontend bridges render as a warning log line.
"""
from __future__ import annotations

import os
import unittest
import unittest.mock
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.core import state_bus as sb
from src.core.render_events import MockTransport


class _RateLimited(Exception):
    """Looks like an HTTP 429 to _is_rate_limit_error."""

    def __init__(self):
        super().__init__("429 rate limit exceeded")
        self.status_code = 429


class TestRateLimitNotice(unittest.TestCase):

    def setUp(self):
        sb.reset()
        sb.set_direct_dispatch(False)
        self.bus = sb.AlimaStateBus()
        self.notices: list = []
        self.bus.subscribe("state.notice", self.notices.append)

    def tearDown(self):
        sb.reset()

    def test_retry_emits_notice_on_bus_not_into_the_token_stream(self):
        from src.llm import llm_service as svc

        calls = {"n": 0}

        def _fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RateLimited()
            return "ok"

        tokens: list = []
        with unittest.mock.patch.object(svc, "_RL_BASE_DELAY_S", 0.0), \
             unittest.mock.patch.object(svc.time, "sleep", lambda s: None):
            result = svc._retry_on_rate_limit(_fn, label="prov/model")

        self.assertEqual(result, "ok")
        self.assertEqual(tokens, [])  # never injected into the answer stream
        self.assertEqual(len(self.notices), 1)
        self.assertIn("Rate-Limit", self.notices[0]["text"])
        self.assertIn("prov/model", self.notices[0]["text"])
        self.assertEqual(self.notices[0]["level"], "warning")


class TestNoticeRendering(unittest.TestCase):
    """Both bus→renderer bridges must render it (K1 lockstep)."""

    def setUp(self):
        sb.reset()

    def tearDown(self):
        sb.reset()

    def _renderer(self):
        from src.ui.unified_message_renderer import UnifiedMessageRenderer

        checkbox = MagicMock()
        checkbox.isChecked.return_value = True
        transport = MockTransport()
        return UnifiedMessageRenderer(transport, checkbox), transport

    def test_webapp_bridge_renders_notice(self):
        from src.webapp.render_bridge import _SessionBusSubscriber

        renderer, transport = self._renderer()
        sub = _SessionBusSubscriber(renderer)
        sub._handle_notice({"text": "⏳ Rate-Limit erreicht", "level": "warning"})
        blocks = transport.of_type("block")
        self.assertEqual(len(blocks), 1)
        self.assertIn("Rate-Limit", blocks[0]["html"])
        self.assertIn("warning", blocks[0]["html"])

    def test_gui_bridge_renders_notice(self):
        from src.ui._chat_panel_bus import BusEventMixin

        renderer, transport = self._renderer()
        stub = MagicMock()
        stub._renderer = renderer
        BusEventMixin._on_bus_notice(stub, {"text": "⏳ Rate-Limit erreicht"})
        blocks = transport.of_type("block")
        self.assertEqual(len(blocks), 1)
        self.assertIn("Rate-Limit", blocks[0]["html"])

    def test_empty_notice_renders_nothing(self):
        from src.webapp.render_bridge import _SessionBusSubscriber

        renderer, transport = self._renderer()
        _SessionBusSubscriber(renderer)._handle_notice({"text": ""})
        self.assertEqual(transport.of_type("block"), [])


if __name__ == "__main__":
    unittest.main()
