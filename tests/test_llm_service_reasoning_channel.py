"""Reasoning channel and think-toggle on OpenAI-compatible backends.

Two field dialects exist for the same thing: vLLM/SGLang/DeepSeek send
``reasoning_content``, Ollama's ``/v1`` endpoint and OpenRouter send
``reasoning``. Reading only one of them turns a thinking model's whole
answer into "empty content".

The think toggle has the same split: ``chat_template_kwargs.enable_thinking``
reaches vLLM, ``reasoning_effort`` reaches Ollama.

Claude Generated.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.data_models import StopReason
from src.llm.llm_service import LlmService, _extract_reasoning


def _delta(content=None, reasoning=None, reasoning_content=None, tool_calls=None):
    """Chunk delta carrying only the fields a given backend actually sends."""
    fields = {"content": content, "tool_calls": tool_calls}
    if reasoning is not None:
        fields["reasoning"] = reasoning
    if reasoning_content is not None:
        fields["reasoning_content"] = reasoning_content
    return SimpleNamespace(**fields)


def _chunk(delta, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


class TestExtractReasoning(unittest.TestCase):
    def test_reads_vllm_field(self):
        self.assertEqual(_extract_reasoning(_delta(reasoning_content="denk")), "denk")

    def test_reads_ollama_field(self):
        self.assertEqual(_extract_reasoning(_delta(reasoning="denk")), "denk")

    def test_absent_channel_is_empty_string(self):
        self.assertEqual(_extract_reasoning(_delta(content="x")), "")


class TestApplyOpenAIThink(unittest.TestCase):
    def _apply(self, model: str, think):
        svc = MagicMock(spec=LlmService)
        svc.logger = MagicMock()
        svc.current_think = think
        params: dict = {}
        LlmService._apply_openai_think(svc, params, model)
        return params

    def test_untouched_when_think_unset(self):
        self.assertEqual(self._apply("nemotron-3.5-lightning:latest", None), {})

    def test_off_reaches_both_dialects(self):
        params = self._apply("nemotron-3.5-lightning:latest", False)
        self.assertEqual(params["reasoning_effort"], "none")
        self.assertIs(
            params["extra_body"]["chat_template_kwargs"]["enable_thinking"], False
        )

    def test_on_reaches_both_dialects(self):
        params = self._apply("qwen3.5:latest", True)
        self.assertEqual(params["reasoning_effort"], "medium")
        self.assertIs(
            params["extra_body"]["chat_template_kwargs"]["enable_thinking"], True
        )

    def test_openai_reasoning_models_keep_effort_only(self):
        params = self._apply("gpt-5", False)
        self.assertEqual(params["reasoning_effort"], "minimal")
        self.assertNotIn("extra_body", params)


class TestOpenAIToolsReasoningCapture(unittest.TestCase):
    """The with-tools path must surface reasoning from either dialect and
    report a length-truncated turn as MAX_TOKENS."""

    def _service(self, create_return):
        svc = MagicMock(spec=LlmService)
        svc.logger = MagicMock()
        svc.current_think = None
        svc._convert_messages_for_openai = lambda m: m
        svc._apply_openai_think = lambda params, model: None
        client = MagicMock()
        client.chat.completions.create.return_value = create_return
        svc.clients = {"p": client}
        return svc

    def _call(self, svc, stream_callback):
        return LlmService._generate_openai_with_tools(
            svc, "p", "m", [{"role": "user", "content": "u"}], [],
            temperature=0.3, top_p=0.9, max_tokens=256,
            stream_callback=stream_callback,
        )

    def test_streaming_ollama_reasoning_and_length_stop(self):
        chunks = [
            _chunk(_delta(reasoning="denk-")),
            _chunk(_delta(reasoning="weiter"), finish_reason="length"),
        ]
        svc = self._service(iter(chunks))
        result = self._call(svc, stream_callback=lambda t: None)
        self.assertEqual(result.reasoning, "denk-weiter")
        self.assertEqual(result.content, "")
        self.assertEqual(result.stop_reason, StopReason.MAX_TOKENS)

    def test_streaming_vllm_reasoning(self):
        svc = self._service(iter([_chunk(_delta(reasoning_content="denk"))]))
        result = self._call(svc, stream_callback=lambda t: None)
        self.assertEqual(result.reasoning, "denk")

    def test_non_streaming_ollama_reasoning(self):
        message = SimpleNamespace(content="", reasoning="denk", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="length")]
        )
        svc = self._service(response)
        result = self._call(svc, stream_callback=None)
        self.assertEqual(result.reasoning, "denk")
        self.assertEqual(result.stop_reason, StopReason.MAX_TOKENS)


if __name__ == "__main__":
    unittest.main()
