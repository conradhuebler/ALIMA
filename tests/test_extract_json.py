"""Tests for src/core/agents/steps/llm_agent_step._extract_json. Claude Generated.

Covers the Mini-WP JSON-Strictness rewrite: thought-strip, solution-unwrap,
fence-unwrap, balanced-brace extraction.
"""

from __future__ import annotations

import unittest

from src.core.agents.steps.llm_agent_step import _extract_json


class TestExtractJson(unittest.TestCase):
    def test_plain_object(self):
        out = _extract_json('{"keywords": ["a", "b"]}')
        self.assertEqual(out, {"keywords": ["a", "b"]})

    def test_solution_wrapper(self):
        raw = (
            "<|begin_of_thought|>some reasoning here<|end_of_thought|>\n"
            '<|begin_of_solution|>{"keywords": ["x"]}<|end_of_solution|>'
        )
        self.assertEqual(_extract_json(raw), {"keywords": ["x"]})

    def test_markdown_fence(self):
        raw = '```json\n{"a": 1, "b": [2, 3]}\n```'
        self.assertEqual(_extract_json(raw), {"a": 1, "b": [2, 3]})

    def test_solution_plus_fence(self):
        raw = (
            "<|begin_of_thought|>thinking<|end_of_thought|>\n"
            '<|begin_of_solution|>\n```json\n{"k": "v"}\n```\n<|end_of_solution|>'
        )
        self.assertEqual(_extract_json(raw), {"k": "v"})

    def test_nested_object_full_capture(self):
        """Regression: old non-greedy regex pulled inner {...} instead of outer."""
        raw = (
            '{"keywords": [{"keyword": "Cadmium", "gnd_id": "4007249-3"}],'
            ' "missing_concepts": ["x"]}'
        )
        out = _extract_json(raw)
        self.assertEqual(len(out.get("keywords", [])), 1)
        self.assertEqual(out["keywords"][0]["gnd_id"], "4007249-3")
        self.assertEqual(out["missing_concepts"], ["x"])

    def test_cadmium_regression_shape(self):
        """Mirrors the live-run output that produced empty selected_keywords."""
        raw = (
            "<|begin_of_thought|>\n**ANALYSE**: ...\n<|end_of_thought|>\n\n"
            "<|begin_of_solution|>\n```json\n"
            '{\n  "keywords": [\n'
            '    {"keyword": "Cadmium", "gnd_id": "4007249-3"},\n'
            '    {"keyword": "Boden", "gnd_id": "4007348-8"}\n'
            "  ],\n"
            '  "keyword_chains": [],\n'
            '  "missing_concepts": []\n'
            "}\n```\n<|end_of_solution|>"
        )
        out = _extract_json(raw)
        self.assertEqual(len(out["keywords"]), 2)
        self.assertEqual(out["keywords"][1]["keyword"], "Boden")

    def test_string_with_braces_in_content(self):
        """Brace-counter must respect string literals."""
        raw = '{"note": "this } is in a string", "n": 1}'
        self.assertEqual(_extract_json(raw)["n"], 1)

    def test_escaped_quote_in_string(self):
        raw = '{"q": "he said \\"hi\\"", "ok": true}'
        out = _extract_json(raw)
        self.assertTrue(out["ok"])
        self.assertEqual(out["q"], 'he said "hi"')

    def test_garbage_returns_empty(self):
        self.assertEqual(_extract_json("no json here"), {})
        self.assertEqual(_extract_json(""), {})
        self.assertEqual(_extract_json("{ broken"), {})

    def test_list_top_level_wrapped_in_items(self):
        raw = '```json\n[{"a": 1}, {"a": 2}]\n```'
        out = _extract_json(raw)
        self.assertEqual(out, {"items": [{"a": 1}, {"a": 2}]})


if __name__ == "__main__":
    unittest.main()
