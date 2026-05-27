"""Regression test for the Ollama multi-turn tool-call message conversion.

Claude Generated (P-ι follow-up).

Bug: ``_convert_messages_for_ollama`` passed the generic assistant tool-call
format (``{"id", "name", "arguments"}``) straight through. The Ollama SDK's
``Message.ToolCall`` model requires a nested ``function`` field, so the second
agent-loop iteration (which replays the assistant tool-call message) failed
with ``tool_calls.0.function Field required``. Surfaced first via the headless
CLI agent, which exercises Ollama multi-turn tool use end-to-end.

The method doesn't use ``self``, so it is called unbound with a dummy receiver
to avoid constructing a full LlmService.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    from src.llm.llm_service import LlmService
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    IMPORT_ERROR = exc


def _convert(messages):
    return LlmService._convert_messages_for_ollama(SimpleNamespace(), messages)


@unittest.skipIf(IMPORT_ERROR is not None, f"LlmService import failed: {IMPORT_ERROR}")
class TestOllamaMessageConversion(unittest.TestCase):
    def test_assistant_tool_calls_nested_under_function(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "list_available_data", "arguments": {"x": 1}}],
        }]
        out = _convert(msgs)
        tc = out[0]["tool_calls"][0]
        self.assertIn("function", tc)
        self.assertEqual(tc["function"]["name"], "list_available_data")
        self.assertEqual(tc["function"]["arguments"], {"x": 1})
        # Flat generic keys must be gone (they break the Ollama Message model).
        self.assertNotIn("name", tc)
        self.assertNotIn("id", tc)

    def test_arguments_json_string_parsed_to_dict(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "t", "arguments": '{"a": "b"}'}],
        }]
        tc = _convert(msgs)[0]["tool_calls"][0]
        self.assertEqual(tc["function"]["arguments"], {"a": "b"})

    def test_bad_arguments_string_falls_back_to_empty_dict(self):
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "t", "arguments": "not json"}],
        }]
        tc = _convert(msgs)[0]["tool_calls"][0]
        self.assertEqual(tc["function"]["arguments"], {})

    def test_tool_result_message_preserved(self):
        out = _convert([{"role": "tool", "content": '{"ok": true}', "name": "t"}])
        self.assertEqual(out[0]["role"], "tool")
        self.assertEqual(out[0]["content"], '{"ok": true}')

    def test_plain_messages_passthrough(self):
        out = _convert([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "answer"},
        ])
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant"])
        self.assertEqual(out[2]["content"], "answer")


if __name__ == "__main__":
    unittest.main()
