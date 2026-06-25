"""Tests for rate-limit (HTTP 429) retry handling in the tool-calling path.

When a provider returns 429 (e.g. Mistral free tier), the agentic workflow used
to abort the whole pipeline. The LLM service now waits + retries instead,
honouring the API's ``Retry-After`` hint when present.

Claude Generated.
"""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.data_models import AgentResponse, StopReason
from src.llm.llm_service import (
    LlmService,
    _is_rate_limit_error,
    _rate_limit_retry_after,
    _retry_on_rate_limit,
    _RL_MAX_RETRIES,
)


class _FakeResp:
    def __init__(self, status: int = 429, headers=None):
        self.status_code = status
        self.headers = headers or {}


class _RateLimitError(Exception):
    """Mimics an OpenAI/Mistral SDK RateLimitError (429 + response.headers)."""

    def __init__(self, msg: str, headers=None):
        super().__init__(msg)
        self.status_code = 429
        self.response = _FakeResp(429, headers)


# The exact payload the operator saw from Mistral.
_MISTRAL_429 = (
    "Error code: 429 - {'object': 'error', 'message': 'Rate limit exceeded', "
    "'type': 'rate_limited', 'param': None, 'code': '1300', 'raw_status_code': 429}"
)


class TestRateLimitDetection(unittest.TestCase):
    def test_detects_mistral_429(self):
        self.assertTrue(_is_rate_limit_error(_RateLimitError(_MISTRAL_429)))

    def test_detects_by_class_name(self):
        exc = type("ResourceExhausted", (Exception,), {})("quota metric exceeded")
        self.assertTrue(_is_rate_limit_error(exc))

    def test_detects_by_message_text(self):
        self.assertTrue(_is_rate_limit_error(Exception("Too Many Requests")))

    def test_server_error_is_not_rate_limit(self):
        exc = type("E", (Exception,), {"status_code": 500})("internal")
        self.assertFalse(_is_rate_limit_error(exc))

    def test_plain_value_error_is_not_rate_limit(self):
        self.assertFalse(_is_rate_limit_error(ValueError("bad input")))


class TestRetryAfterExtraction(unittest.TestCase):
    def test_reads_numeric_retry_after_header(self):
        exc = _RateLimitError("429", headers={"retry-after": "7"})
        self.assertEqual(_rate_limit_retry_after(exc), 7.0)

    def test_reads_http_date_retry_after_header(self):
        from email.utils import format_datetime
        from datetime import datetime, timezone, timedelta

        when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
        exc = _RateLimitError("429", headers={"Retry-After": when})
        delay = _rate_limit_retry_after(exc)
        self.assertTrue(25 <= delay <= 31, f"expected ~30s, got {delay}")

    def test_reads_gemini_retry_delay_duration(self):
        exc = type("Gem", (Exception,), {"retry_delay": SimpleNamespace(seconds=23)})("429")
        self.assertEqual(_rate_limit_retry_after(exc), 23.0)

    def test_reads_seconds_from_message(self):
        self.assertEqual(_rate_limit_retry_after(Exception("retry after 12 seconds")), 12.0)

    def test_none_when_no_hint(self):
        self.assertIsNone(_rate_limit_retry_after(Exception("Rate limit exceeded")))


class TestRetryLoop(unittest.TestCase):
    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _RateLimitError("429", headers={"retry-after": "0"})
            return "OK"

        logs = []
        result = _retry_on_rate_limit(flaky, label="t", status_cb=logs.append)
        self.assertEqual(result, "OK")
        self.assertEqual(calls["n"], 3)
        # Two waits → two status lines surfaced to the GUI/CLI log.
        self.assertEqual(sum("Rate-Limit" in s for s in logs), 2)

    def test_gives_up_after_max_retries(self):
        calls = {"n": 0}

        def always():
            calls["n"] += 1
            raise _RateLimitError("429", headers={"retry-after": "0"})

        with self.assertRaises(_RateLimitError):
            _retry_on_rate_limit(always, label="t")
        self.assertEqual(calls["n"], _RL_MAX_RETRIES + 1)

    def test_non_rate_limit_error_reraised_immediately(self):
        calls = {"n": 0}

        def boom():
            calls["n"] += 1
            raise ValueError("nope")

        with self.assertRaises(ValueError):
            _retry_on_rate_limit(boom, label="t")
        self.assertEqual(calls["n"], 1)  # no retry

    def test_should_stop_interrupts_wait(self):
        def slow_429():
            raise _RateLimitError("429", headers={"retry-after": "30"})

        t0 = time.time()
        with self.assertRaises(_RateLimitError):
            _retry_on_rate_limit(slow_429, label="t", should_stop=lambda: True)
        self.assertLess(time.time() - t0, 2.0, "should_stop must abort the wait promptly")


class TestGenerateWithToolsRetries(unittest.TestCase):
    """End-to-end: generate_with_tools() retries a transient 429 from the handler."""

    def _service(self):
        svc = MagicMock(spec=LlmService)
        svc.generate_with_tools = LlmService.generate_with_tools.__get__(svc, LlmService)
        svc._map_provider_name = lambda p: p
        svc._ensure_provider_initialized = lambda p: True
        svc.supported_providers = {
            "fake": {
                "generator": MagicMock(),
                "config": SimpleNamespace(provider_type="openai_compatible"),
            },
        }
        return svc

    def test_transient_429_is_retried_not_raised(self):
        svc = self._service()
        calls = {"n": 0}

        def handler(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _RateLimitError("429", headers={"retry-after": "0"})
            return AgentResponse(content="ok", tool_calls=[], stop_reason=StopReason.END_TURN)

        svc._generate_openai_with_tools = handler
        resp = svc.generate_with_tools(
            provider="fake", model="mistral-small-latest",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
        )
        self.assertEqual(resp.content, "ok")
        self.assertEqual(calls["n"], 2)  # first 429, second success


if __name__ == "__main__":
    unittest.main()
