"""Native Ollama tool path: budget, truncation, reasoning channel.

Three things were missing on this path while every other provider had them:
``max_tokens`` was accepted and dropped (the request ran against Ollama's
unlimited ``num_predict=-1``), a truncated turn came back as ``END_TURN`` with
an empty answer, and the reasoning channel — a THIRD field name, ``thinking`` —
was not read at all.

Claude Generated.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.data_models import StopReason
from src.llm.llm_service import LlmService, _field


def _msg(content="", thinking="", tool_calls=None, as_dict=False):
    data = {"content": content, "thinking": thinking, "tool_calls": tool_calls}
    return data if as_dict else SimpleNamespace(**data)


def _response(message, done_reason="stop", as_dict=False):
    data = {"message": message, "done_reason": done_reason}
    return data if as_dict else SimpleNamespace(**data)


class _Service:
    """Bind the real method onto a stub carrying a fake ollama client."""

    def __init__(self, chat_return):
        self.svc = MagicMock(spec=LlmService)
        self.svc.logger = MagicMock()
        self.svc.current_think = None
        self.svc._convert_messages_for_ollama = lambda m: m
        self.client = MagicMock()
        self.client.chat.return_value = chat_return
        self.svc.clients = {"p": self.client}

    def call(self, *, max_tokens=4096, stream=False, tools=None):
        return LlmService._generate_ollama_native_with_tools(
            self.svc, "p", "m", [{"role": "user", "content": "u"}], tools or [],
            temperature=0.3, top_p=0.9, max_tokens=max_tokens,
            stream_callback=(lambda t: None) if stream else None,
        )


class TestFieldAccessor(unittest.TestCase):
    """The Ollama client returns pydantic models for some calls and dicts for
    others, and has switched between them across releases."""

    def test_reads_a_dict(self):
        self.assertEqual(_field({"a": 1}, "a"), 1)

    def test_reads_an_object(self):
        self.assertEqual(_field(SimpleNamespace(a=1), "a"), 1)

    def test_missing_and_none_both_give_the_default(self):
        self.assertEqual(_field({}, "a", "x"), "x")
        self.assertEqual(_field(SimpleNamespace(a=None), "a", "x"), "x")


class TestBudgetIsSent(unittest.TestCase):
    def test_max_tokens_becomes_num_predict(self):
        svc = _Service(_response(_msg(content="ok")))
        svc.call(max_tokens=777)
        self.assertEqual(svc.client.chat.call_args.kwargs["options"]["num_predict"], 777)

    def test_no_budget_no_cap(self):
        """max_tokens=0/None must not pin num_predict to 0 — that would answer
        with nothing at all."""
        svc = _Service(_response(_msg(content="ok")))
        svc.call(max_tokens=0)
        self.assertNotIn("num_predict", svc.client.chat.call_args.kwargs["options"])


class TestTruncationIsReported(unittest.TestCase):
    def test_length_is_max_tokens_not_end_turn(self):
        svc = _Service(_response(_msg(content="", thinking="halber Gedanke"), done_reason="length"))
        res = svc.call()
        self.assertEqual(res.stop_reason, StopReason.MAX_TOKENS)
        self.assertEqual(res.reasoning, "halber Gedanke")

    def test_truncation_wins_over_a_half_written_tool_call(self):
        tc = {"function": {"name": "search", "arguments": {"q": "x"}}}
        svc = _Service(_response(_msg(tool_calls=[tc]), done_reason="length"))
        self.assertEqual(svc.call().stop_reason, StopReason.MAX_TOKENS)

    def test_normal_stop_is_end_turn(self):
        svc = _Service(_response(_msg(content="fertig")))
        res = svc.call()
        self.assertEqual(res.stop_reason, StopReason.END_TURN)
        self.assertEqual(res.content, "fertig")

    def test_tool_calls_still_report_tool_use(self):
        tc = {"function": {"name": "search", "arguments": {"q": "x"}}}
        svc = _Service(_response(_msg(tool_calls=[tc])))
        res = svc.call()
        self.assertEqual(res.stop_reason, StopReason.TOOL_USE)
        self.assertEqual([t.name for t in res.tool_calls], ["search"])

    def test_dict_shaped_response_works_too(self):
        svc = _Service(_response(_msg(content="", thinking="denk", as_dict=True),
                                 done_reason="length", as_dict=True))
        res = svc.call()
        self.assertEqual(res.stop_reason, StopReason.MAX_TOKENS)
        self.assertEqual(res.reasoning, "denk")


class TestStreaming(unittest.TestCase):
    def _chunks(self, *chunks):
        return _Service(iter(chunks))

    def test_thinking_and_done_reason_from_the_stream(self):
        svc = self._chunks(
            _response(_msg(thinking="denk-"), done_reason=""),
            _response(_msg(thinking="weiter"), done_reason="length"),
        )
        res = svc.call(stream=True)
        self.assertEqual(res.reasoning, "denk-weiter")
        self.assertEqual(res.stop_reason, StopReason.MAX_TOKENS)

    def test_content_streams_and_ends_normally(self):
        svc = self._chunks(
            _response(_msg(content="Hal")),
            _response(_msg(content="lo"), done_reason="stop"),
        )
        res = svc.call(stream=True)
        self.assertEqual(res.content, "Hallo")
        self.assertEqual(res.stop_reason, StopReason.END_TURN)


if __name__ == "__main__":
    unittest.main()
