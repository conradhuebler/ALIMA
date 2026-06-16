"""P-δ.5b — Streaming-with-Tools + Cancel-Latency tests. Claude Generated.

Verifies:
1. OpenAI text tokens stream live even when tools are in the request.
2. Tool-call deltas (index/id/name/arguments) are accumulated across chunks
   and assembled into correct ToolCall objects.
3. ``should_stop`` cancels the stream per-chunk and returns CANCELLED.
4. ``should_stop`` is forwarded from generate_with_tools → every sub-handler.
5. ``should_stop`` is forwarded from AgentLoop.run → generate_with_tools.
6. Ollama keeps stream=False when tools present (API limitation).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock

from src.core.agent_loop import AgentLoop
from src.core.data_models import AgentResponse, StopReason, ToolCall


# ---------------------------------------------------------------------------
# Helpers for building fake OpenAI streaming chunks
# ---------------------------------------------------------------------------

def _make_delta(
    content: str | None = None,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _make_tc_delta(
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=func)


def _make_chunk(
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = _make_delta(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def _make_svc(chunks: List[Any]) -> Any:
    """Build a minimal LlmService stub that returns `chunks` from the OpenAI streaming path."""
    from src.llm.llm_service import LlmService

    svc = LlmService.__new__(LlmService)
    svc.logger = MagicMock()
    svc._convert_messages_for_openai = lambda m: m

    class _FakeStream:
        def __init__(self, items):
            self._items = iter(items)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._items)

        def close(self):
            pass

    class _FakeCompletions:
        def __init__(self, stream):
            self._stream = stream

        def create(self, **_kwargs):
            return self._stream

    class _FakeChat:
        def __init__(self, completions):
            self.completions = completions

    stream = _FakeStream(chunks)
    svc.clients = {"fake": SimpleNamespace(chat=_FakeChat(_FakeCompletions(stream)))}
    return svc


def _make_response(content: str = "ok") -> AgentResponse:
    return AgentResponse(content=content, tool_calls=[], stop_reason=StopReason.END_TURN)


# ---------------------------------------------------------------------------
# Test class 1: OpenAI streaming with tools
# ---------------------------------------------------------------------------

class TestOpenAIStreamingWithTools(unittest.TestCase):

    def _call(self, svc, chunks_unused, tools, callback=None, should_stop=None):
        """Thin wrapper — chunks already baked into svc."""
        return svc._generate_openai_with_tools(
            provider="fake",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stream_callback=callback,
            should_stop=should_stop,
        )

    def test_text_only_streams_tokens(self):
        tokens: list[str] = []
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
        ]
        svc = _make_svc(chunks)
        resp = self._call(svc, chunks, tools=[], callback=tokens.append)
        self.assertEqual(tokens, ["Hello", " world"])
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)
        self.assertEqual(resp.content, "Hello world")

    def test_streaming_with_tools_accumulates_deltas(self):
        tokens: list[str] = []
        chunks = [
            _make_chunk(content="I will search"),
            _make_chunk(
                tool_calls=[_make_tc_delta(0, id="call_abc", name="search_gnd", arguments="")],
                finish_reason=None,
            ),
            _make_chunk(
                tool_calls=[_make_tc_delta(0, arguments='{"query":')],
            ),
            _make_chunk(
                tool_calls=[_make_tc_delta(0, arguments='"test"}')],
            ),
            _make_chunk(finish_reason="tool_calls"),
        ]
        svc = _make_svc(chunks)
        resp = self._call(
            svc, chunks,
            tools=[{"name": "search_gnd", "description": "", "parameters": {}}],
            callback=tokens.append,
        )
        self.assertEqual(tokens, ["I will search"])
        self.assertEqual(len(resp.tool_calls), 1)
        tc = resp.tool_calls[0]
        self.assertEqual(tc.name, "search_gnd")
        self.assertEqual(tc.id, "call_abc")
        self.assertEqual(tc.arguments, {"query": "test"})
        self.assertEqual(resp.stop_reason, StopReason.TOOL_USE)

    def test_streaming_multiple_tool_calls(self):
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, id="id0", name="tool_a", arguments='{"x":1}')]),
            _make_chunk(tool_calls=[_make_tc_delta(1, id="id1", name="tool_b", arguments='{"y":2}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        svc = _make_svc(chunks)
        resp = self._call(svc, chunks, tools=[{"name": "tool_a", "description": "", "parameters": {}},
                                              {"name": "tool_b", "description": "", "parameters": {}}],
                          callback=lambda t: None)
        self.assertEqual(len(resp.tool_calls), 2)
        self.assertEqual(resp.tool_calls[0].name, "tool_a")
        self.assertEqual(resp.tool_calls[0].arguments, {"x": 1})
        self.assertEqual(resp.tool_calls[1].name, "tool_b")
        self.assertEqual(resp.tool_calls[1].arguments, {"y": 2})

    def test_malformed_json_in_tool_args_uses_raw(self):
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, id="id0", name="t", arguments="not valid json")]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        svc = _make_svc(chunks)
        resp = self._call(svc, chunks, tools=[{"name": "t", "description": "", "parameters": {}}],
                          callback=lambda t: None)
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].arguments, {"raw": "not valid json"})

    def test_finish_reason_length_sets_max_tokens(self):
        chunks = [
            _make_chunk(content="partial"),
            _make_chunk(finish_reason="length"),
        ]
        svc = _make_svc(chunks)
        resp = self._call(svc, chunks, tools=[], callback=lambda t: None)
        self.assertEqual(resp.stop_reason, StopReason.MAX_TOKENS)


# ---------------------------------------------------------------------------
# Test class 2: should_stop cancellation
# ---------------------------------------------------------------------------

class TestShouldStopCancellation(unittest.TestCase):

    def test_should_stop_cancels_mid_stream_openai(self):
        tokens: list[str] = []
        chunks = [
            _make_chunk(content="first"),
            _make_chunk(content="second"),
            _make_chunk(finish_reason="stop"),
        ]
        svc = _make_svc(chunks)
        call_count = [0]

        def _stop():
            call_count[0] += 1
            return call_count[0] >= 2  # stop before 2nd token

        resp = svc._generate_openai_with_tools(
            provider="fake", model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.7, top_p=0.9, max_tokens=256,
            stream_callback=tokens.append,
            should_stop=_stop,
        )
        self.assertEqual(tokens, ["first"])
        self.assertEqual(resp.stop_reason, StopReason.CANCELLED)
        self.assertEqual(resp.tool_calls, [])

    def test_should_stop_none_does_not_affect_stream(self):
        tokens: list[str] = []
        chunks = [
            _make_chunk(content="a"),
            _make_chunk(content="b"),
            _make_chunk(finish_reason="stop"),
        ]
        svc = _make_svc(chunks)
        resp = svc._generate_openai_with_tools(
            provider="fake", model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.7, top_p=0.9, max_tokens=256,
            stream_callback=tokens.append,
            should_stop=None,
        )
        self.assertEqual(tokens, ["a", "b"])
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)

    def test_should_stop_after_ollama_blocking_call(self):
        """should_stop returning True after blocking Ollama .chat() → CANCELLED."""
        from src.llm.llm_service import LlmService

        svc = LlmService.__new__(LlmService)
        svc.logger = MagicMock()
        svc._convert_messages_for_ollama = lambda m: m

        class _FakeOllamaClient:
            def chat(self_inner, **kwargs):
                return {"message": {"content": "result", "tool_calls": []}}

        svc.clients = {"fake": _FakeOllamaClient()}

        resp = svc._generate_ollama_native_with_tools(
            provider="fake", model="cogito:14b",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "", "parameters": {}}],
            temperature=0.7, top_p=0.9, max_tokens=256,
            should_stop=lambda: True,
        )
        self.assertEqual(resp.stop_reason, StopReason.CANCELLED)


# ---------------------------------------------------------------------------
# Test class 3: should_stop forwarded through dispatch to every sub-handler
# ---------------------------------------------------------------------------

class TestShouldStopSignaturePropagation(unittest.TestCase):
    """generate_with_tools forwards should_stop kwarg to every sub-handler."""

    def setUp(self):
        from src.llm.llm_service import LlmService
        self.LlmService = LlmService

    _GEN_TO_TYPE = {
        "_generate_ollama_native": "ollama",
        "_generate_openai_compatible": "openai_compatible",
        "_generate_anthropic": "anthropic",
        "_generate_gemini": "gemini",
    }

    def _patched_service(self, generator_attr: str, sub_handler_attr: str):
        from types import SimpleNamespace
        svc = MagicMock(spec=self.LlmService)
        svc.generate_with_tools = self.LlmService.generate_with_tools.__get__(svc, self.LlmService)
        svc._map_provider_name = lambda p: p
        svc._ensure_provider_initialized = lambda p: True
        # Dispatch is by provider_type on the provider's config object.
        svc.supported_providers = {
            "fake": {
                "generator": getattr(svc, generator_attr),
                "config": SimpleNamespace(provider_type=self._GEN_TO_TYPE[generator_attr]),
            }
        }
        target = MagicMock(return_value=_make_response())
        setattr(svc, sub_handler_attr, target)
        return svc, target

    def _assert_should_stop_forwarded(self, generator_attr, handler_attr):
        svc, target = self._patched_service(generator_attr, handler_attr)
        cb = lambda: False
        svc.generate_with_tools(
            provider="fake", model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            should_stop=cb,
        )
        _, kwargs = target.call_args
        self.assertIs(kwargs.get("should_stop"), cb,
                      f"{handler_attr} did not receive should_stop kwarg")

    def test_ollama_receives_should_stop(self):
        self._assert_should_stop_forwarded(
            "_generate_ollama_native", "_generate_ollama_native_with_tools"
        )

    def test_openai_receives_should_stop(self):
        self._assert_should_stop_forwarded(
            "_generate_openai_compatible", "_generate_openai_with_tools"
        )

    def test_anthropic_receives_should_stop(self):
        self._assert_should_stop_forwarded(
            "_generate_anthropic", "_generate_anthropic_with_tools"
        )

    def test_gemini_receives_should_stop(self):
        self._assert_should_stop_forwarded(
            "_generate_gemini", "_generate_gemini_with_tools"
        )

    def test_fallback_receives_should_stop(self):
        svc = MagicMock(spec=self.LlmService)
        svc.generate_with_tools = self.LlmService.generate_with_tools.__get__(svc, self.LlmService)
        svc._map_provider_name = lambda p: p
        svc._ensure_provider_initialized = lambda p: True
        svc.supported_providers = {"fake": {"generator": MagicMock()}}
        target = MagicMock(return_value=_make_response())
        svc._generate_text_fallback_with_tools = target
        cb = lambda: False
        svc.generate_with_tools(
            provider="fake", model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            should_stop=cb,
        )
        _, kwargs = target.call_args
        self.assertIs(kwargs.get("should_stop"), cb)

    def test_anthropic_positional_arg_count_unchanged(self):
        """Anthropic handler still receives exactly 7 positional args (no seed, should_stop via kwarg)."""
        svc, target = self._patched_service("_generate_anthropic", "_generate_anthropic_with_tools")
        svc.generate_with_tools(
            provider="fake", model="claude-3-opus",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=42, should_stop=lambda: False,
        )
        args, _ = target.call_args
        self.assertEqual(len(args), 7,
                         f"anthropic must NOT receive seed; should_stop via kwarg; got args={args}")


# ---------------------------------------------------------------------------
# Test class 4: AgentLoop passes should_stop to generate_with_tools
# ---------------------------------------------------------------------------

class TestAgentLoopPassesShouldStop(unittest.TestCase):

    def _make_loop(self, should_stop=None):
        llm_service = MagicMock()
        llm_service.generate_with_tools.return_value = _make_response("done")
        registry = MagicMock()
        registry.get_tool_schemas.return_value = []
        loop = AgentLoop(
            llm_service=llm_service,
            tool_registry=registry,
            max_iterations=1,
            should_stop=should_stop,
        )
        return loop, llm_service

    def test_run_passes_should_stop_to_llm_service(self):
        cb = lambda: False
        loop, svc = self._make_loop(should_stop=cb)
        loop.run(
            system_prompt="sys", user_prompt="usr",
            tools=None, provider="fake", model="m",
            temperature=0.3, top_p=0.9, max_tokens=128,
        )
        kwargs = svc.generate_with_tools.call_args.kwargs
        self.assertIs(kwargs.get("should_stop"), cb)

    def test_run_default_should_stop_is_none(self):
        loop, svc = self._make_loop(should_stop=None)
        loop.run(
            system_prompt="sys", user_prompt="usr",
            tools=None, provider="fake", model="m",
            temperature=0.3, top_p=0.9, max_tokens=128,
        )
        kwargs = svc.generate_with_tools.call_args.kwargs
        self.assertIsNone(kwargs.get("should_stop"))


# ---------------------------------------------------------------------------
# Test class 5: Ollama streaming + tools (P-δ.5c — formerly the regression
# test that enforced stream=False; lifted now that Ollama SDK 0.6.1 supports
# stream=True with tools, emitting tool_calls atomically on the final chunk).
# ---------------------------------------------------------------------------


def _make_ollama_chunk(content: str = "", tool_calls=None, done: bool = False):
    """Build a fake Ollama ChatResponse-shaped chunk via SimpleNamespace."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=msg, done=done)


def _make_ollama_svc(chunks_or_response):
    """Build a minimal LlmService stub returning fake Ollama chunks (stream)
    or a fake non-streaming response dict."""
    from src.llm.llm_service import LlmService

    svc = LlmService.__new__(LlmService)
    svc.logger = MagicMock()
    svc._convert_messages_for_ollama = lambda m: m
    captured: dict = {}

    class _FakeOllamaClient:
        def chat(self_inner, **kwargs):
            captured.update(kwargs)
            if kwargs.get("stream"):
                return iter(chunks_or_response)
            return chunks_or_response

    svc.clients = {"fake": _FakeOllamaClient()}
    return svc, captured


class TestOllamaStreamingWithTools(unittest.TestCase):

    def _call(self, svc, tools, callback=None, should_stop=None):
        return svc._generate_ollama_native_with_tools(
            provider="fake", model="cogito:14b",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            temperature=0.7, top_p=0.9, max_tokens=256,
            stream_callback=callback,
            should_stop=should_stop,
        )

    def test_ollama_with_tools_streams_true_and_passes_tools(self):
        """P-δ.5c: stream_callback set → stream=True even when tools present,
        tools must be forwarded to the SDK call."""
        chunks = [
            _make_ollama_chunk(content="hi", done=False),
            _make_ollama_chunk(done=True),
        ]
        svc, captured = _make_ollama_svc(chunks)
        self._call(
            svc,
            tools=[{"name": "t", "description": "", "parameters": {}}],
            callback=lambda t: None,
        )
        self.assertTrue(
            captured.get("stream"),
            "Ollama must use stream=True when stream_callback provided",
        )
        self.assertIsNotNone(
            captured.get("tools"),
            "tools must be forwarded to the SDK call alongside stream=True",
        )

    def test_ollama_streaming_with_tools_streams_content(self):
        """Text tokens are delivered via callback while tools are in request."""
        tokens: list = []
        chunks = [
            _make_ollama_chunk(content="Hello"),
            _make_ollama_chunk(content=" "),
            _make_ollama_chunk(content="world"),
            _make_ollama_chunk(done=True),
        ]
        svc, _ = _make_ollama_svc(chunks)
        resp = self._call(
            svc,
            tools=[{"name": "t", "description": "", "parameters": {}}],
            callback=tokens.append,
        )
        self.assertEqual(tokens, ["Hello", " ", "world"])
        self.assertEqual(resp.content, "Hello world")
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.stop_reason, StopReason.END_TURN)

    def test_ollama_streaming_with_tools_captures_final_tool_calls(self):
        """tool_calls are extracted from the final (done=True) chunk."""
        chunks = [
            _make_ollama_chunk(content="searching..."),
            _make_ollama_chunk(
                tool_calls=[
                    SimpleNamespace(function=SimpleNamespace(
                        name="search_gnd", arguments={"query": "test"}
                    ))
                ],
                done=True,
            ),
        ]
        svc, _ = _make_ollama_svc(chunks)
        resp = self._call(
            svc,
            tools=[{"name": "search_gnd", "description": "", "parameters": {}}],
            callback=lambda t: None,
        )
        self.assertEqual(len(resp.tool_calls), 1)
        self.assertEqual(resp.tool_calls[0].name, "search_gnd")
        self.assertEqual(resp.tool_calls[0].arguments, {"query": "test"})
        self.assertEqual(resp.stop_reason, StopReason.TOOL_USE)

    def test_ollama_should_stop_per_chunk_cancels(self):
        """should_stop checked per chunk → cancel returns CANCELLED."""
        chunks = [
            _make_ollama_chunk(content="part1"),
            _make_ollama_chunk(content="part2"),
            _make_ollama_chunk(content="part3"),
            _make_ollama_chunk(done=True),
        ]
        svc, _ = _make_ollama_svc(chunks)
        call_count = {"n": 0}

        def stopper():
            call_count["n"] += 1
            return call_count["n"] >= 2  # cancel after first chunk processed

        tokens: list = []
        resp = self._call(
            svc,
            tools=[{"name": "t", "description": "", "parameters": {}}],
            callback=tokens.append,
            should_stop=stopper,
        )
        self.assertEqual(resp.stop_reason, StopReason.CANCELLED)
        # Streamed content collected before cancel must be preserved.
        self.assertEqual(resp.content, "part1")
        self.assertLess(len(tokens), 3, "stream should abort before all chunks consumed")


if __name__ == "__main__":
    unittest.main()
