"""Claude Generated - Tests for LLMAgentStep's salvage/required/retry mechanism.

Covers the fix for the title_list_search silent-failure chain: a step whose
JSON parsing fails (prose instead of JSON, or truncation) must not silently
propagate empty data — it should recover via salvage where possible, retry
once with a stricter prompt when configured, and hard-fail when the
declared field is still empty and `required: true`.

Strategy: no Qt, no real DB, no real LLM — mock AgentLoop.run() to return
canned AgentResult objects, same pattern as test_agents_v2.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.base_step import StepConfig
from src.core.agents.steps.llm_agent_step import (
    LLMAgentStep,
    _apply_salvage,
    _salvage_title_triples,
)
from src.core.data_models import AgentResult

# Real freeform failure text from an actual title_list_search run (GWDG/gemma
# ignored the "JSON only" instruction and emitted a plain Title/Publisher-or-
# ISBN/Year line-triple listing instead). - Claude Generated
_REAL_FAILING_RUN_TEXT = """The Fraying Bonds of Peace – Economic Origins of the First World War
CUP
2026
Among the Waves of Globalisation – An Economic History of the World
Giappichelli Editore
2025
Schule des Sehens – Bilder von Giotto bis Warhol
978-3-7-7574-588-8
2019
The Market for Skill. Apprenticeship and Economic Growth in Early Modern England
PUP
2027"""


def _agent_result(content: str, stop_reason: str = "end_turn") -> AgentResult:
    return AgentResult(content=content, tool_log=[], iterations=1, stop_reason=stop_reason)


class TestSalvageTitleTriples(unittest.TestCase):
    def test_recovers_all_entries_from_real_failing_text(self):
        recovered = _salvage_title_triples(_REAL_FAILING_RUN_TEXT)
        self.assertEqual(len(recovered), 4)
        self.assertEqual(recovered[0]["title"], "The Fraying Bonds of Peace – Economic Origins of the First World War")
        self.assertEqual(recovered[0]["publisher"], "CUP")
        self.assertEqual(recovered[0]["year"], "2026")
        self.assertEqual(recovered[0]["isbn"], "")

    def test_isbn_line_recognized_as_isbn_not_publisher(self):
        recovered = _salvage_title_triples(_REAL_FAILING_RUN_TEXT)
        schule = next(r for r in recovered if r["title"].startswith("Schule des Sehens"))
        self.assertEqual(schule["isbn"], "978-3-7-7574-588-8")
        self.assertEqual(schule["publisher"], "")
        self.assertEqual(schule["year"], "2019")

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(_salvage_title_triples(""), [])
        self.assertEqual(_salvage_title_triples("just one line"), [])

    def test_does_not_match_when_no_trailing_year_line(self):
        # Two lines that look like a title+publisher but no bare-year third line.
        text = "Some Title\nSome Publisher\nNot a year at all"
        self.assertEqual(_salvage_title_triples(text), [])


class TestApplySalvageTypes(unittest.TestCase):
    def test_codes_type_still_recovers_dk_rvk(self):
        # Regression: explicit type: codes must behave exactly as before the
        # default-type change. - Claude Generated
        parsed = _apply_salvage(
            {}, "Die Klassifikation ist DK 504.064 und RVK WW 3350.",
            {"field": "classifications", "type": "codes"},
        )
        self.assertEqual(len(parsed["classifications"]), 2)

    def test_missing_type_performs_no_recovery(self):
        # New behavior: a salvage: block with only required/retry (no type)
        # must NOT fall back to code-salvage — it should be a pure
        # empty-field gate. - Claude Generated
        parsed = _apply_salvage(
            {}, "Die Klassifikation ist DK 504.064.",
            {"field": "catalog_hits", "required": True},
        )
        self.assertEqual(parsed, {})

    def test_title_triples_type_recovers_via_apply_salvage(self):
        parsed = _apply_salvage({}, _REAL_FAILING_RUN_TEXT, {"field": "titles", "type": "title_triples"})
        self.assertEqual(len(parsed["titles"]), 4)

    def test_does_not_overwrite_valid_json(self):
        parsed = _apply_salvage(
            {"titles": [{"title": "Already parsed"}]},
            _REAL_FAILING_RUN_TEXT,
            {"field": "titles", "type": "title_triples"},
        )
        self.assertEqual(parsed["titles"], [{"title": "Already parsed"}])


class TestRequiredAndRetry(unittest.TestCase):
    def _make_step(self, salvage_cfg, extra_raw=None, stream_callback=None):
        raw = {
            "system_prompt": "sys",
            "user_prompt": "user {abstract}",
            "inputs": {"abstract": "${abstract}"},
            "outputs": {},
            "llm": {"max_iterations": 1},
            "salvage": salvage_cfg,
        }
        if extra_raw:
            raw.update(extra_raw)
        cfg = StepConfig(
            id="extract_titles",
            type="llm_agent",
            inputs={"abstract": "${abstract}"},
            outputs={"extra.titles": "response.titles"},
            raw=raw,
        )
        return LLMAgentStep(
            cfg, llm_service=MagicMock(), tool_registry=MagicMock(),
            stream_callback=stream_callback,
        )

    def test_required_raises_when_field_stays_empty(self):
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result("just some prose, no json, no recognizable triples")
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "type": "title_triples", "required": True})
            result = step.execute(ctx)
        self.assertFalse(result.success)
        self.assertIn("required field 'titles'", result.error)

    def test_required_does_not_raise_when_salvage_recovers(self):
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result(_REAL_FAILING_RUN_TEXT)
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "type": "title_triples", "required": True})
            result = step.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(len(ctx.extra["titles"]), 4)

    def test_retry_invokes_loop_again_and_uses_successful_result(self):
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.side_effect = [
            _agent_result("unparseable prose with no triples"),
            _agent_result('{"titles": [{"title": "Recovered via retry", "year": "2026"}]}'),
        ]
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "type": "title_triples", "retry": True, "required": True})
            result = step.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(fake_loop.run.call_count, 2)
        self.assertEqual(ctx.extra["titles"][0]["title"], "Recovered via retry")

    def test_retry_stricter_prompt_mentions_the_field(self):
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.side_effect = [
            _agent_result("unparseable prose"),
            _agent_result('{"titles": []}'),
        ]
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "type": "title_triples", "retry": True})
            step.execute(ctx)
        retry_call_kwargs = fake_loop.run.call_args_list[1].kwargs
        self.assertIn("titles", retry_call_kwargs["system_prompt"])
        self.assertIn("kein gültiges JSON", retry_call_kwargs["system_prompt"])

    def test_max_tokens_truncation_warns_on_original_attempt(self):
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result(
            '{"titles": [{"title": "cut off mid', stop_reason="max_tokens"
        )
        messages = []
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "required": True}, stream_callback=messages.append)
            result = step.execute(ctx)
        self.assertFalse(result.success)
        self.assertTrue(any("max_tokens abgeschnitten" in m for m in messages), messages)

    def test_max_tokens_truncation_on_retry_attempt_also_warns(self):
        # The real bug this guards: a retry attempt that produces a
        # correctly-keyed {"titles": [...]} JSON but gets cut off mid-array
        # by max_tokens must be diagnosed as a truncation, not silently
        # collapse into the same generic "still empty" message as a genuine
        # formatting failure. - Claude Generated
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.side_effect = [
            _agent_result(""),  # first attempt: empty content (unrelated failure)
            _agent_result(
                '{"titles": [{"title": "The Economic History of Latin America"',
                stop_reason="max_tokens",
            ),
        ]
        messages = []
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step(
                {"field": "titles", "retry": True, "required": True},
                stream_callback=messages.append,
            )
            result = step.execute(ctx)
        self.assertFalse(result.success)
        self.assertIn("required field 'titles'", result.error)
        retry_warnings = [m for m in messages if "[retry]" in m and "max_tokens abgeschnitten" in m]
        self.assertTrue(retry_warnings, messages)

    def test_no_retry_without_config_stays_at_one_call(self):
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result("prose, no json")
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "type": "title_triples"})  # no retry, no required
            result = step.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(fake_loop.run.call_count, 1)

    def test_required_does_not_raise_on_legitimate_empty_list(self):
        # {"catalog_hits": []} is a well-formed "found nothing" answer (e.g.
        # no wishlist title exists in the catalog) — required: true must NOT
        # treat this the same as JSON parsing failing outright ({} with the
        # key missing). Regression for a real bug found during end-to-end
        # workflow simulation. - Claude Generated
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result('{"titles": []}')
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "required": True})
            result = step.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(ctx.extra["titles"], [])
        self.assertEqual(fake_loop.run.call_count, 1)  # no retry triggered either

    def test_retry_not_triggered_by_legitimate_empty_list(self):
        ctx = SharedContext(abstract="some text", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result('{"titles": []}')
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            step = self._make_step({"field": "titles", "retry": True, "required": True})
            result = step.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(fake_loop.run.call_count, 1)


class TestChunkedRequired(unittest.TestCase):
    def test_chunked_raises_when_a_chunk_produces_no_items(self):
        ctx = SharedContext(abstract="", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result("prose response, no json at all")
        raw = {
            "system_prompt": "sys",
            "user_prompt": "user",
            "inputs": {"items": "${extra.items}"},
            "outputs": {},
            "llm": {"max_iterations": 1},
            "salvage": {"field": "analysis", "required": True},
            "chunking": {
                "enabled": True,
                "chunk_size": 2,
                "chunk_field": "items",
                "merge_key": "analysis",
                "dedup_field": "input_title",
            },
        }
        cfg = StepConfig(
            id="analyze_duplicates", type="llm_agent",
            inputs={"items": "${extra.items}"},
            outputs={"extra.duplicate_analysis": "response.analysis"},
            raw=raw,
        )
        ctx.extra = {"items": [{"input_title": "A"}, {"input_title": "B"}]}
        step = LLMAgentStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            result = step.execute(ctx)
        self.assertFalse(result.success)
        self.assertIn("chunk 1/1", result.error)

    def test_chunked_succeeds_when_items_present(self):
        ctx = SharedContext(abstract="", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result(
            '{"analysis": [{"input_title": "A", "status": "new"}]}'
        )
        raw = {
            "system_prompt": "sys",
            "user_prompt": "user",
            "inputs": {"items": "${extra.items}"},
            "outputs": {},
            "llm": {"max_iterations": 1},
            "salvage": {"field": "analysis", "required": True},
            "chunking": {
                "enabled": True,
                "chunk_size": 2,
                "chunk_field": "items",
                "merge_key": "analysis",
                "dedup_field": "input_title",
            },
        }
        cfg = StepConfig(
            id="analyze_duplicates", type="llm_agent",
            inputs={"items": "${extra.items}"},
            outputs={"extra.duplicate_analysis": "response.analysis"},
            raw=raw,
        )
        ctx.extra = {"items": [{"input_title": "A"}]}
        step = LLMAgentStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            result = step.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(len(ctx.extra["duplicate_analysis"]), 1)

    def test_chunked_legitimate_empty_analysis_does_not_raise(self):
        # {"analysis": []} — the chunk's items were validly analyzed as "no
        # matches" — must NOT be treated as a parsing failure. Regression,
        # same class of bug as test_required_does_not_raise_on_legitimate_empty_list. - Claude Generated
        ctx = SharedContext(abstract="", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result('{"analysis": []}')
        raw = {
            "system_prompt": "sys",
            "user_prompt": "user",
            "inputs": {"items": "${extra.items}"},
            "outputs": {},
            "llm": {"max_iterations": 1},
            "salvage": {"field": "analysis", "required": True},
            "chunking": {
                "enabled": True,
                "chunk_size": 2,
                "chunk_field": "items",
                "merge_key": "analysis",
                "dedup_field": "input_title",
            },
        }
        cfg = StepConfig(
            id="analyze_duplicates", type="llm_agent",
            inputs={"items": "${extra.items}"},
            outputs={"extra.duplicate_analysis": "response.analysis"},
            raw=raw,
        )
        ctx.extra = {"items": [{"input_title": "A"}]}
        step = LLMAgentStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            result = step.execute(ctx)
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(ctx.extra["duplicate_analysis"], [])

    def test_chunked_max_tokens_truncation_warns(self):
        ctx = SharedContext(abstract="", provider="test", model="test-model")
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result(
            '{"analysis": [{"input_title": "A"', stop_reason="max_tokens"
        )
        raw = {
            "system_prompt": "sys",
            "user_prompt": "user",
            "inputs": {"items": "${extra.items}"},
            "outputs": {},
            "llm": {"max_iterations": 1},
            "salvage": {"field": "analysis", "required": True},
            "chunking": {
                "enabled": True,
                "chunk_size": 2,
                "chunk_field": "items",
                "merge_key": "analysis",
                "dedup_field": "input_title",
            },
        }
        cfg = StepConfig(
            id="analyze_duplicates", type="llm_agent",
            inputs={"items": "${extra.items}"},
            outputs={"extra.duplicate_analysis": "response.analysis"},
            raw=raw,
        )
        ctx.extra = {"items": [{"input_title": "A"}]}
        messages = []
        step = LLMAgentStep(
            cfg, llm_service=MagicMock(), tool_registry=MagicMock(),
            stream_callback=messages.append,
        )
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop):
            result = step.execute(ctx)
        self.assertFalse(result.success)
        self.assertTrue(any("max_tokens abgeschnitten" in m for m in messages), messages)


if __name__ == "__main__":
    unittest.main()
