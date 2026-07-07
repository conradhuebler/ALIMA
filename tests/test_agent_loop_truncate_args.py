"""Claude Generated - Tests for agent_loop._truncate_args and the
result_full field on tool_log entries.

Covers a real reported readability bug: `search_finc(terms=[30 titles])`
rendered in logs as "terms=['The Fraying Bonds of Peace – Economic …" — a
crude json.dumps()-then-truncate that cut off mid-way through the FIRST
list element and hid that there even were more titles being searched.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.core.agent_loop import AgentLoop, _truncate_args
from src.core.data_models import AgentResponse, ToolCall


class TestTruncateArgs(unittest.TestCase):
    def test_empty_args(self):
        self.assertEqual(_truncate_args({}), "")
        self.assertEqual(_truncate_args(None), "")

    def test_scalar_args_unaffected(self):
        self.assertEqual(_truncate_args({"kind": "initial"}), "kind='initial'")

    def test_long_list_shows_count_and_preview(self):
        terms = [f"Book Title Number {i} With Some Extra Words And More" for i in range(30)]
        out = _truncate_args({"terms": terms, "search_type": "title", "limit": 5})
        self.assertIn("terms=[30:", out)
        self.assertIn("search_type='title'", out)
        self.assertIn("limit=5", out)

    def test_short_list_shown_fully(self):
        out = _truncate_args({"terms": ["A", "B"]})
        self.assertEqual(out, "terms=[2: A, B]")

    def test_long_individual_list_item_still_capped(self):
        long_title = "X" * 100
        out = _truncate_args({"terms": [long_title]})
        self.assertIn("…", out)
        self.assertLess(len(out), len(long_title))

    def test_overall_result_still_capped_by_max_len(self):
        # The smart list preview alone is already short (only shows the
        # first 2 items regardless of list length), so force truncation via
        # a very tight max_len rather than a huge list. - Claude Generated
        out = _truncate_args({"terms": ["Alpha", "Beta"], "search_type": "title"}, max_len=15)
        self.assertLessEqual(len(out), 18)  # 15 + "..."
        self.assertTrue(out.endswith("..."))


class TestToolLogResultFull(unittest.TestCase):
    """tool_log entries must carry the complete tool result (result_full),
    not just the 500-char result_preview, so deterministic steps can parse
    a tool's real output instead of trusting an LLM to retype it. - Claude Generated"""

    def test_tool_log_entry_has_result_full(self):
        registry = MagicMock()
        registry.get_tool_schemas.return_value = [
            {"name": "search_finc", "description": "", "parameters": {}}
        ]
        long_result = '{"hits": [' + ",".join(f'{{"id": {i}}}' for i in range(200)) + "]}"
        registry.execute.return_value = long_result

        llm_service = MagicMock()
        first = AgentResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="search_finc", arguments={"terms": ["A"]})],
        )
        second = AgentResponse(content="done")
        llm_service.generate_with_tools.side_effect = [first, second]

        loop = AgentLoop(llm_service=llm_service, tool_registry=registry, max_iterations=5)
        result = loop.run(
            system_prompt="sys", user_prompt="user",
            tools=["search_finc"], provider="test", model="test-model",
        )
        self.assertEqual(len(result.tool_log), 1)
        entry = result.tool_log[0]
        self.assertEqual(entry["result_full"], long_result)
        self.assertGreater(len(entry["result_full"]), len(entry["result_preview"]))
        self.assertEqual(entry["result_preview"], long_result[:500])


if __name__ == "__main__":
    unittest.main()
