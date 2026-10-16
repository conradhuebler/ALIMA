"""cancel_generation used to dispatch a server-side cancel attempt keyed by
literal provider-type strings ("openai"/"anthropic"/"chatai"/"comet") against
self.clients, which is keyed by the configured instance name (the same bug
class fixed for Anthropic/Gemini generation elsewhere this session) - and
even a correct key match would have hit AttributeError on every branch,
since none of the installed anthropic/openai/ollama clients expose a
cancel-style method. Removed rather than patched: real cancellation already
happens via cancel_requested/stream_running, checked inside every provider's
own streaming loop, independent of this dead dispatch. - Claude Generated
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import Mock

from src.llm.llm_service import LlmService


def _svc(provider="claude", request_id="req-1"):
    svc = LlmService.__new__(LlmService)
    svc.logger = logging.getLogger("test")
    svc.stream_running = True
    svc.cancel_requested = False
    svc.current_provider = provider
    svc.current_request_id = request_id
    svc.generation_cancelled = Mock()
    return svc


class CancelGenerationTest(unittest.TestCase):
    def test_sets_flags_and_emits_for_any_provider(self):
        for provider in ("claude", "ollama", "gemini", "my-openai-compatible"):
            svc = _svc(provider=provider)
            result = svc.cancel_generation(reason="user_requested")

            self.assertTrue(result)
            self.assertTrue(svc.cancel_requested)
            self.assertFalse(svc.stream_running)
            svc.generation_cancelled.emit.assert_called_once_with("user_requested")

    def test_returns_false_when_nothing_is_running(self):
        svc = _svc()
        svc.stream_running = False

        result = svc.cancel_generation()

        self.assertFalse(result)
        self.assertFalse(svc.cancel_requested)

    def test_the_dead_server_side_helpers_are_gone(self):
        for name in (
            "_cancel_openai_request",
            "_cancel_openai_compatible_request",
            "_cancel_ollama_request",
            "_cancel_anthropic_request",
        ):
            self.assertFalse(hasattr(LlmService, name), name)


if __name__ == "__main__":
    unittest.main()
