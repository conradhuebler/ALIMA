"""P-η — Tests for seed-parameter propagation in tool-calling pathway.

Covers WP11 Sek 8 seed retrofit (7+1 sites). Per operator decision, Anthropic
is skipped (SDK does not accept seed); the dispatch must not crash but also
must not push seed through.

Claude Generated.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

from src.core.agent_loop import AgentLoop
from src.core.agents.base_shared_context import BaseSharedContext
from src.core.agents.shared_context import SharedContext
from src.core.data_models import AgentResponse, StopReason


def _make_response(content: str = "ok") -> AgentResponse:
    return AgentResponse(
        content=content,
        tool_calls=[],
        stop_reason=StopReason.END_TURN,
    )


class TestGenerateWithToolsSeedDispatch(unittest.TestCase):
    """generate_with_tools() forwards `seed` to the right sub-handler."""

    def setUp(self):
        from src.llm.llm_service import LlmService
        self.LlmService = LlmService

    def _patched_service(self, generator_attr: str, sub_handler_attr: str):
        """Build a service stub where dispatch sees `generator_attr` and we mock the matching `_with_tools` handler."""
        svc = MagicMock(spec=self.LlmService)
        # Bind the real generate_with_tools method onto the mock so dispatch runs.
        svc.generate_with_tools = self.LlmService.generate_with_tools.__get__(svc, self.LlmService)
        svc._map_provider_name = lambda p: p
        svc._ensure_provider_initialized = lambda p: True
        svc.supported_providers = {
            "fake": {"generator": getattr(svc, generator_attr)},
        }
        target = MagicMock(return_value=_make_response())
        setattr(svc, sub_handler_attr, target)
        return svc, target

    def test_ollama_handler_receives_seed(self):
        svc, target = self._patched_service("_generate_ollama_native", "_generate_ollama_native_with_tools")
        svc.generate_with_tools(
            provider="fake", model="cogito:14b",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=42,
        )
        args, kwargs = target.call_args
        # positional: provider, model, messages, tools, temperature, top_p, max_tokens, seed, stream_callback
        self.assertEqual(args[7], 42, f"seed should be 8th positional arg, got args={args}")

    def test_openai_handler_receives_seed(self):
        svc, target = self._patched_service("_generate_openai_compatible", "_generate_openai_with_tools")
        svc.generate_with_tools(
            provider="fake", model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=7,
        )
        args, _ = target.call_args
        self.assertEqual(args[7], 7)

    def test_gemini_handler_receives_seed(self):
        svc, target = self._patched_service("_generate_gemini", "_generate_gemini_with_tools")
        svc.generate_with_tools(
            provider="fake", model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=99,
        )
        # gemini handler has different signature (no provider positional)
        # model, messages, tools, temperature, top_p, max_tokens, seed, stream_callback
        args, _ = target.call_args
        self.assertEqual(args[6], 99)

    def test_anthropic_handler_does_not_receive_seed(self):
        """Anthropic SDK kennt kein seed — dispatch must omit it (P-η decision)."""
        svc, target = self._patched_service("_generate_anthropic", "_generate_anthropic_with_tools")
        svc.generate_with_tools(
            provider="fake", model="claude-3-opus",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=42,
        )
        args, _ = target.call_args
        # anthropic signature: model, messages, tools, temperature, top_p, max_tokens, stream_callback
        # length must be 7 positional + nothing for seed
        self.assertEqual(len(args), 7, f"anthropic handler must NOT receive seed; got args={args}")

    def test_text_fallback_handler_receives_seed(self):
        svc = MagicMock(spec=self.LlmService)
        svc.generate_with_tools = self.LlmService.generate_with_tools.__get__(svc, self.LlmService)
        svc._map_provider_name = lambda p: p
        svc._ensure_provider_initialized = lambda p: True
        # Generator that isn't any known type → fallback path
        svc.supported_providers = {"fake": {"generator": MagicMock()}}
        target = MagicMock(return_value=_make_response())
        svc._generate_text_fallback_with_tools = target
        svc.generate_with_tools(
            provider="fake", model="mystery-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=11,
        )
        args, _ = target.call_args
        self.assertEqual(args[7], 11)


class TestOllamaOptionsCarrySeed(unittest.TestCase):
    """`options['seed']` is populated when seed!=None in Ollama-native handler."""

    def test_ollama_options_includes_seed_when_set(self):
        from src.llm.llm_service import LlmService
        svc = LlmService.__new__(LlmService)
        svc.logger = MagicMock()
        svc.clients = {"fake": MagicMock()}
        svc._convert_messages_for_ollama = lambda m: m

        captured: Dict[str, Any] = {}

        class FakeClient:
            def chat(self_inner, **kwargs):
                captured.update(kwargs)
                return {"message": {"content": "ok"}, "done_reason": "stop"}

        svc.clients["fake"] = FakeClient()

        svc._generate_ollama_native_with_tools(
            provider="fake", model="cogito:14b",
            messages=[{"role": "user", "content": "x"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=42,
        )
        opts = captured.get("options", {})
        self.assertEqual(opts.get("seed"), 42)

    def test_ollama_options_omits_seed_when_none(self):
        from src.llm.llm_service import LlmService
        svc = LlmService.__new__(LlmService)
        svc.logger = MagicMock()
        svc.clients = {"fake": MagicMock()}
        svc._convert_messages_for_ollama = lambda m: m

        captured: Dict[str, Any] = {}

        class FakeClient:
            def chat(self_inner, **kwargs):
                captured.update(kwargs)
                return {"message": {"content": "ok"}, "done_reason": "stop"}

        svc.clients["fake"] = FakeClient()

        svc._generate_ollama_native_with_tools(
            provider="fake", model="cogito:14b",
            messages=[{"role": "user", "content": "x"}],
            tools=[], temperature=0.5, top_p=0.9, max_tokens=128,
            seed=None,
        )
        opts = captured.get("options", {})
        self.assertNotIn("seed", opts)


class TestAgentLoopSeedPropagation(unittest.TestCase):
    """AgentLoop.run(seed=...) forwards to LlmService.generate_with_tools()."""

    def test_run_passes_seed_to_llm_service(self):
        llm_service = MagicMock()
        llm_service.generate_with_tools.return_value = _make_response("done")
        registry = MagicMock()
        registry.get_tool_schemas.return_value = []

        loop = AgentLoop(llm_service=llm_service, tool_registry=registry, max_iterations=1)
        loop.run(
            system_prompt="sys", user_prompt="usr",
            tools=None,
            provider="fake", model="cogito:14b",
            temperature=0.3, top_p=0.9, max_tokens=128,
            seed=123,
        )
        kwargs = llm_service.generate_with_tools.call_args.kwargs
        self.assertEqual(kwargs.get("seed"), 123)

    def test_run_default_seed_is_none(self):
        llm_service = MagicMock()
        llm_service.generate_with_tools.return_value = _make_response("done")
        registry = MagicMock()
        registry.get_tool_schemas.return_value = []

        loop = AgentLoop(llm_service=llm_service, tool_registry=registry, max_iterations=1)
        loop.run(
            system_prompt="sys", user_prompt="usr",
            tools=None,
            provider="fake", model="cogito:14b",
            temperature=0.3, top_p=0.9, max_tokens=128,
        )
        kwargs = llm_service.generate_with_tools.call_args.kwargs
        self.assertIsNone(kwargs.get("seed"))


class TestSharedContextSeedRoundtrip(unittest.TestCase):
    """SharedContext.to_dict()/from_dict() preserve seed."""

    def test_base_shared_context_seed_roundtrip(self):
        ctx = BaseSharedContext(seed=42)
        d = ctx.to_dict()
        self.assertEqual(d.get("seed"), 42)
        restored = BaseSharedContext.from_dict(d)
        self.assertEqual(restored.seed, 42)

    def test_shared_context_seed_roundtrip(self):
        ctx = SharedContext(abstract="X", seed=99)
        d = ctx.to_dict()
        self.assertEqual(d.get("seed"), 99)
        restored = SharedContext.from_dict(d)
        self.assertEqual(restored.seed, 99)

    def test_shared_context_default_seed_is_none(self):
        ctx = SharedContext(abstract="X")
        self.assertIsNone(ctx.seed)


if __name__ == "__main__":
    unittest.main()
