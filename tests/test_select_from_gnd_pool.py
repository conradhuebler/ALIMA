"""_handle_select_from_gnd_pool's LLM call TypeError'd on every invocation.

``generate_response()`` was called with ``system_prompt=`` (the real kwarg is
``system``), no ``request_id`` (required, no default) and ``max_tokens=``
(the method has no such parameter at all) - confirmed via
``inspect.signature().bind()``. The TypeError was caught per-chunk and logged
as a routine warning, so the tool always returned ``keywords: []`` with no
visible error; the chat agent is told to use exactly this tool
(``chat_prompts.py``). - Claude Generated
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from src.mcp.tool_registry import ToolRegistry


def _llm(response_json='{"keywords": [{"keyword": "Anatomie", "gnd_id": "4"}]}'):
    llm = MagicMock()
    llm.alima_config.unified_config.resolve_default_provider_model.return_value = (
        "claude", "claude-opus-5-5",
    )
    llm.generate_response.return_value = response_json
    return llm


class SelectFromGndPoolActuallyCallsTheRealSignatureTest(unittest.TestCase):
    def test_a_real_bound_call_does_not_raise(self):
        """The regression test that would have caught the original bug:
        bind the exact kwargs the handler sends against the real method
        signature, not a mock that accepts anything."""
        from src.llm.llm_service import LlmService
        import inspect

        sig = inspect.signature(LlmService.generate_response)
        sig.bind(
            object(),
            provider="claude", model="claude-opus-5-5", prompt="x",
            request_id="r1", system="sys", temperature=0.01, stream=False,
        )  # must not raise TypeError

    def test_resolves_the_agentic_default_instead_of_empty_strings(self):
        llm = _llm()
        reg = ToolRegistry(llm_service=llm)
        reg._handle_select_from_gnd_pool(
            abstract="ein Abstract", gnd_entries=[{"keyword": "Anatomie", "gnd_id": "4"}],
        )
        kwargs = llm.generate_response.call_args.kwargs
        self.assertEqual(kwargs["provider"], "claude")
        self.assertEqual(kwargs["model"], "claude-opus-5-5")

    def test_passes_system_not_system_prompt(self):
        llm = _llm()
        reg = ToolRegistry(llm_service=llm)
        reg._handle_select_from_gnd_pool(
            abstract="x", gnd_entries=[{"keyword": "Anatomie", "gnd_id": "4"}],
        )
        kwargs = llm.generate_response.call_args.kwargs
        self.assertIn("system", kwargs)
        self.assertNotIn("system_prompt", kwargs)

    def test_passes_a_request_id(self):
        llm = _llm()
        reg = ToolRegistry(llm_service=llm)
        reg._handle_select_from_gnd_pool(
            abstract="x", gnd_entries=[{"keyword": "Anatomie", "gnd_id": "4"}],
        )
        kwargs = llm.generate_response.call_args.kwargs
        self.assertTrue(kwargs.get("request_id"))

    def test_does_not_pass_the_nonexistent_max_tokens_kwarg(self):
        llm = _llm()
        reg = ToolRegistry(llm_service=llm)
        reg._handle_select_from_gnd_pool(
            abstract="x", gnd_entries=[{"keyword": "Anatomie", "gnd_id": "4"}],
        )
        kwargs = llm.generate_response.call_args.kwargs
        self.assertNotIn("max_tokens", kwargs)

    def test_a_relevant_keyword_actually_comes_back(self):
        """End to end: the tool must produce a non-empty result when the LLM
        genuinely returns one - the old bug always returned empty."""
        llm = _llm()
        reg = ToolRegistry(llm_service=llm)
        result = json.loads(reg._handle_select_from_gnd_pool(
            abstract="x", gnd_entries=[{"keyword": "Anatomie", "gnd_id": "4"}],
        ))
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["keywords"][0]["keyword"], "Anatomie")

    def test_no_usable_provider_is_reported_not_silently_empty(self):
        llm = _llm()
        llm.alima_config.unified_config.resolve_default_provider_model.return_value = ("", "")
        reg = ToolRegistry(llm_service=llm)
        result = json.loads(reg._handle_select_from_gnd_pool(
            abstract="x", gnd_entries=[{"keyword": "Anatomie", "gnd_id": "4"}],
        ))
        self.assertIn("error", result)
        llm.generate_response.assert_not_called()


if __name__ == "__main__":
    unittest.main()
