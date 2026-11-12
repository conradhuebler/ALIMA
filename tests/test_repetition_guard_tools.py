"""Repetition guard in ``LlmService.generate_with_tools`` - Claude Generated

A fake Ollama stream that runs into a loop: the guard closes the stream,
returns ``StopReason.REPETITION`` with the text before the loop, and leaves
operator cancel, disabled detection and non-streaming calls behaving as
described in ``generate_with_tools``.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.core.data_models import StopReason
from src.utils.config_models import RepetitionDetectionConfig

ROOT = Path(__file__).resolve().parent.parent
LOOP = (
    "Das Werk behandelt die Auswirkungen von Mikroplastik auf marine "
    "Ökosysteme und deren Nahrungsketten. "
) * 200


def _prefix() -> str:
    data = json.loads((ROOT / "examples" / "classic_result.json").read_text(encoding="utf-8"))
    return data["results"]["initial_llm_call_details"]["response_full_text"][:2000]


class _FakeStream:
    """Iterable Ollama stream that records how far it was read and whether it was closed."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for chunk in self._chunks:
            if self.closed:
                return
            self.consumed += 1
            yield chunk

    def close(self):
        self.closed = True


def _chunks(text: str, size: int = 23, field: str = "content"):
    out = []
    for i in range(0, len(text), size):
        out.append({"message": {field: text[i:i + size]}, "done_reason": ""})
    out.append({"message": {"content": ""}, "done_reason": "stop", "done": True})
    return out


def _service(stream=None, response=None):
    from src.llm.llm_service import LlmService

    svc = LlmService.__new__(LlmService)
    svc.logger = MagicMock()
    svc._map_provider_name = lambda p: p
    svc._ensure_provider_initialized = lambda p: True
    svc._convert_messages_for_ollama = lambda m: m
    svc.supported_providers = {"fake": {"config": SimpleNamespace(provider_type="ollama")}}
    captured = {}

    class _Client:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return stream if kwargs.get("stream") else response

    svc.clients = {"fake": _Client()}
    return svc, captured


CFG = RepetitionDetectionConfig(grace_period_seconds=0)


def _call(svc, cfg=CFG, stream_cb=..., should_stop=None, thinking_cb=None, **kw):
    received = []
    if stream_cb is ...:
        stream_cb = received.append
    with patch("src.core.state_bus.AlimaStateBus") as bus:
        resp = svc.generate_with_tools(
            provider="fake", model="m", messages=[{"role": "user", "content": "hi"}],
            tools=[], stream_callback=stream_cb, should_stop=should_stop,
            thinking_callback=thinking_cb, repetition_config=cfg, **kw,
        )
    return resp, received, bus


class GuardStopsAStreamingLoopTest(unittest.TestCase):
    def test_loop_stops_the_stream_and_keeps_the_text_before(self):
        """Gate: REPETITION, stream closed early, prefix kept, tail cut."""
        prefix = _prefix()
        stream = _FakeStream(_chunks(prefix + LOOP))
        svc, _ = _service(stream=stream)
        resp, received, bus = _call(svc)

        self.assertEqual(resp.stop_reason, StopReason.REPETITION)
        self.assertTrue(stream.closed, "stream must be closed, not just left unread")
        self.assertLess(stream.consumed, len(_chunks(prefix + LOOP)) // 2)
        self.assertTrue(resp.content.startswith(prefix), "text before the loop must survive")
        self.assertLess(len(resp.content), len(prefix) + 450)
        info = resp.repetition
        self.assertEqual(info["channel"], "answer")
        self.assertEqual(info["target"], "content")
        self.assertEqual(info["kept_chars"], len(resp.content))
        self.assertGreater(info["cut_chars"], 0)
        self.assertEqual(resp.tool_calls, [])
        # the notice went on the state bus (GUI log, webapp, CLI)
        events = [c.args[0] for c in bus.return_value.emit_event.call_args_list]
        self.assertIn("state.notice", events)

    def test_loop_free_stream_is_untouched(self):
        prefix = _prefix()
        stream = _FakeStream(_chunks(prefix))
        svc, _ = _service(stream=stream)
        resp, received, _bus = _call(svc)
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)
        self.assertEqual(resp.content, prefix)
        self.assertEqual("".join(received), prefix)
        self.assertIsNone(resp.repetition)

    def test_thinking_channel_loop_is_stopped(self):
        stream = _FakeStream(_chunks(LOOP, field="thinking"))
        svc, _ = _service(stream=stream)
        thoughts = []
        resp, _r, _b = _call(svc, thinking_cb=thoughts.append)
        self.assertEqual(resp.stop_reason, StopReason.REPETITION)
        self.assertEqual(resp.repetition["channel"], "thinking")
        self.assertEqual(resp.repetition["target"], "reasoning")
        self.assertLess(len(resp.reasoning), 1000)
        self.assertTrue(thoughts, "caller's thinking sink still receives the deltas")

    def test_thinking_measured_without_a_caller_sink(self):
        """Ollama: the guard adds its own thinking sink, so a loop there is stopped live."""
        stream = _FakeStream(_chunks(LOOP, field="thinking"))
        svc, _ = _service(stream=stream)
        resp, _r, _b = _call(svc, thinking_cb=None)
        self.assertEqual(resp.stop_reason, StopReason.REPETITION)
        self.assertTrue(stream.closed)

    def test_operator_cancel_stays_cancelled(self):
        stream = _FakeStream(_chunks(_prefix() + LOOP))
        svc, _ = _service(stream=stream)
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 3

        resp, _r, _b = _call(svc, should_stop=stop)
        self.assertEqual(resp.stop_reason, StopReason.CANCELLED)
        self.assertIsNone(resp.repetition)

    def test_operator_cancel_inside_the_grace_period_stays_cancelled(self):
        """The loop is at abort level but the grace period still runs when the
        operator stops: the stop is the operator's, not the guard's."""
        prefix = _prefix()
        stream = _FakeStream(_chunks(prefix + LOOP))
        svc, _ = _service(stream=stream)
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 250  # ~5700 characters: well inside the loop

        cfg = RepetitionDetectionConfig(grace_period_seconds=1000)
        resp, _r, _b = _call(svc, cfg=cfg, should_stop=stop)
        self.assertEqual(resp.stop_reason, StopReason.CANCELLED)
        self.assertIsNone(resp.repetition)
        self.assertGreater(len(resp.content), len(prefix) + 2000, "nothing cut on operator stop")

    def test_disabled_detection_passes_the_loop_through(self):
        stream = _FakeStream(_chunks(LOOP))
        svc, _ = _service(stream=stream)
        resp, received, _b = _call(svc, cfg=RepetitionDetectionConfig(enabled=False))
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)
        self.assertEqual(resp.content, LOOP)

    def test_non_streaming_answer_is_measured_after_the_fact(self):
        prefix = _prefix()
        response = {"message": {"content": prefix + LOOP}, "done_reason": "length"}
        svc, captured = _service(response=response)
        resp, _r, _b = _call(svc, stream_cb=None)
        self.assertFalse(captured.get("stream"), "no stream_callback → no streaming request")
        self.assertEqual(resp.stop_reason, StopReason.REPETITION)
        self.assertTrue(resp.content.startswith(prefix))
        self.assertLess(len(resp.content), len(prefix) + 450)

    def test_repetition_penalty_reaches_ollama_options(self):
        stream = _FakeStream(_chunks("kurz"))
        svc, captured = _service(stream=stream)
        _call(svc, repetition_penalty=1.3)
        self.assertEqual(captured["options"].get("repeat_penalty"), 1.3)


class RepetitionPenaltyOpenAITest(unittest.TestCase):
    def test_openai_tool_path_sends_repetition_penalty_in_extra_body(self):
        from src.llm.llm_service import LlmService

        svc = LlmService.__new__(LlmService)
        svc.logger = MagicMock()
        svc._convert_messages_for_openai = lambda m: m
        svc._apply_openai_think = lambda params, model: None
        captured = {}

        class _Completions:
            def create(self, **params):
                captured.update(params)
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )])

        svc.clients = {"fake": SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))}
        svc._generate_openai_with_tools(
            "fake", "m", [{"role": "user", "content": "hi"}], [], 0.5, 0.9, 128,
            repetition_penalty=1.2,
        )
        self.assertEqual(captured["extra_body"]["repetition_penalty"], 1.2)


if __name__ == "__main__":
    unittest.main()
