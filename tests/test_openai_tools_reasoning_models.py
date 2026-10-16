"""OpenAI reasoning models (o1/o3/o4/gpt-5) reject temperature/top_p outright
on chat completions. The text path (_generate_openai_compatible) already knew
this, but only behind an ``if image`` gate; _generate_openai_with_tools (the
agentic/tool-calling path) had no guard at all, so a reasoning model called
through AgentLoop/MCP tools would 400 on every single call. Both methods now
share LlmService.STRICT_DEFAULT_SAMPLING_MODELS and drop temperature/top_p
unconditionally for a matching model name (no images in this path, so the
old ``if image`` gate never applied anyway). - Claude Generated
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.llm.llm_service import LlmService


def _make_svc():
    svc = LlmService.__new__(LlmService)
    svc.logger = MagicMock()
    svc._convert_messages_for_openai = lambda m: m

    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["params"] = kwargs
            message = SimpleNamespace(content="ok", tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])

    class _FakeChat:
        def __init__(self, completions):
            self.completions = completions

    svc.clients = {"fake": SimpleNamespace(chat=_FakeChat(_FakeCompletions()))}
    return svc, captured


def _call(svc, model):
    return svc._generate_openai_with_tools(
        provider="fake",
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stream_callback=None,  # non-streaming branch
    )


class ReasoningModelsLoseTemperatureAndTopPTest(unittest.TestCase):
    def test_o3_drops_temperature_and_top_p(self):
        svc, captured = _make_svc()
        _call(svc, model="o3-mini")

        self.assertNotIn("temperature", captured["params"])
        self.assertNotIn("top_p", captured["params"])

    def test_gpt5_drops_temperature_and_top_p(self):
        svc, captured = _make_svc()
        _call(svc, model="gpt-5.1")

        self.assertNotIn("temperature", captured["params"])
        self.assertNotIn("top_p", captured["params"])

    def test_a_regular_model_keeps_temperature_and_top_p(self):
        # Deliberately NOT "gpt-4o*" - that prefix is itself in
        # STRICT_DEFAULT_SAMPLING_MODELS.
        svc, captured = _make_svc()
        _call(svc, model="gpt-4-turbo")

        self.assertEqual(captured["params"]["temperature"], 0.7)
        self.assertEqual(captured["params"]["top_p"], 0.9)

    def test_other_params_are_unaffected(self):
        """Not a blanket strip - only temperature/top_p, nothing else."""
        svc, captured = _make_svc()
        _call(svc, model="o1")

        self.assertEqual(captured["params"]["model"], "o1")
        self.assertEqual(captured["params"]["max_tokens"], 256)


def _make_streaming_svc():
    """Same as _make_svc but returns chunks from the streaming branch
    (stream_callback is not None), which has its own params dict."""
    svc = LlmService.__new__(LlmService)
    svc.logger = MagicMock()
    svc._convert_messages_for_openai = lambda m: m

    captured = {}

    class _FakeStream:
        def __iter__(self):
            return iter([])

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["params"] = kwargs
            return _FakeStream()

    class _FakeChat:
        def __init__(self, completions):
            self.completions = completions

    svc.clients = {"fake": SimpleNamespace(chat=_FakeChat(_FakeCompletions()))}
    return svc, captured


class ReasoningModelsLoseThemInTheStreamingBranchTooTest(unittest.TestCase):
    """The streaming branch builds its own, separate params dict - both
    needed the guard independently."""

    def test_o3_drops_temperature_and_top_p_while_streaming(self):
        svc, captured = _make_streaming_svc()
        svc._generate_openai_with_tools(
            provider="fake",
            model="o3-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stream_callback=lambda t: None,
        )

        self.assertNotIn("temperature", captured["params"])
        self.assertNotIn("top_p", captured["params"])

    def test_a_regular_model_keeps_them_while_streaming(self):
        svc, captured = _make_streaming_svc()
        svc._generate_openai_with_tools(
            provider="fake",
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stream_callback=lambda t: None,
        )

        self.assertEqual(captured["params"]["temperature"], 0.7)
        self.assertEqual(captured["params"]["top_p"], 0.9)


if __name__ == "__main__":
    unittest.main()
