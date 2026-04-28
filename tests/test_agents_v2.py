"""Tests for the v4 Workflow Agent System (registry, steps, loader, executor).

Strategy:
    - No Qt, no real DB, no real LLM
    - Mock AgentLoop.run() to return canned AgentResult objects
    - Use test-only registered steps/fns via registry._reset_for_tests() + manual register
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.agents import registry
from src.core.agents.context_path import (
    resolve_mapping,
    resolve_path,
    resolve_string,
    resolve_value,
)
from src.core.agents.registry import (
    STEP_REGISTRY,
    TOOL_FN_REGISTRY,
    get_step_class,
    get_tool_fn,
    list_steps,
    list_tool_fns,
    register_step,
    register_tool_fn,
)
from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.base_step import BaseStep, StepConfig, StepResult
from src.core.agents.steps.deterministic_step import DeterministicStep
from src.core.agents.steps.llm_agent_step import LLMAgentStep
from src.core.agents.workflow_executor import WorkflowExecutor
from src.core.agents.workflow_loader import load_workflow, parse_steps
from src.core.data_models import AgentResult


def _agent_result(content: str) -> AgentResult:
    return AgentResult(content=content, tool_log=[], iterations=1)


class TestContextPath(unittest.TestCase):
    def setUp(self):
        self.ctx = SharedContext(
            abstract="An abstract.",
            extracted_keywords=["a", "b", "c"],
            step_results={"search": {"gnd_entries": [{"title": "Foo", "gnd_id": "123"}]}},
        )
        self.ctx.extra = {"input": {"query": "hello"}}

    def test_resolve_typed_attr(self):
        self.assertEqual(resolve_path("abstract", self.ctx), "An abstract.")
        self.assertEqual(resolve_path("extracted_keywords", self.ctx), ["a", "b", "c"])

    def test_resolve_steps(self):
        self.assertEqual(
            resolve_path("steps.search.gnd_entries.0.title", self.ctx), "Foo"
        )

    def test_resolve_extra_and_input_alias(self):
        self.assertEqual(resolve_path("extra.input.query", self.ctx), "hello")
        self.assertEqual(resolve_path("input.query", self.ctx), "hello")

    def test_missing_path_raises(self):
        with self.assertRaises(KeyError):
            resolve_path("steps.missing.x", self.ctx)
        with self.assertRaises(KeyError):
            resolve_path("unknown_root", self.ctx)

    def test_resolve_string_replaces(self):
        out = resolve_string("A=${abstract} Q=${extra.input.query}", self.ctx)
        self.assertEqual(out, "A=An abstract. Q=hello")

    def test_resolve_string_missing_becomes_empty(self):
        out = resolve_string("X=${steps.nope.y}", self.ctx)
        self.assertEqual(out, "X=")

    def test_resolve_value_bare_placeholder_returns_native(self):
        # Bare ${...} expression should return the list, not a stringified JSON
        val = resolve_value("${extracted_keywords}", self.ctx)
        self.assertEqual(val, ["a", "b", "c"])

    def test_resolve_value_non_placeholder_passthrough(self):
        self.assertEqual(resolve_value("plain text", self.ctx), "plain text")

    def test_resolve_mapping(self):
        mapped = resolve_mapping(
            {"q": "${extra.input.query}", "all": "${extracted_keywords}"}, self.ctx
        )
        self.assertEqual(mapped, {"q": "hello", "all": ["a", "b", "c"]})


class TestRegistry(unittest.TestCase):
    def setUp(self):
        # Snapshot state so we don't clobber module-level built-ins between tests.
        self._step_snap = dict(STEP_REGISTRY)
        self._fn_snap = dict(TOOL_FN_REGISTRY)

    def tearDown(self):
        STEP_REGISTRY.clear()
        STEP_REGISTRY.update(self._step_snap)
        TOOL_FN_REGISTRY.clear()
        TOOL_FN_REGISTRY.update(self._fn_snap)

    def test_step_registry_lookup(self):
        self.assertIn("llm_agent", list_steps())
        self.assertIn("deterministic", list_steps())
        self.assertIs(get_step_class("llm_agent"), LLMAgentStep)
        self.assertIs(get_step_class("deterministic"), DeterministicStep)

    def test_register_new_step(self):
        @register_step("my_custom")
        class _MyStep(BaseStep):
            def run(self, context):
                return {"ok": True}

        self.assertIs(get_step_class("my_custom"), _MyStep)

    def test_duplicate_registration_raises(self):
        class _A(BaseStep):
            def run(self, context): return {}
        class _B(BaseStep):
            def run(self, context): return {}

        register_step("dup_test")(_A)
        with self.assertRaises(ValueError):
            register_step("dup_test")(_B)

    def test_register_tool_fn(self):
        @register_tool_fn("echo")
        def _echo(value):
            return {"echoed": value}

        fn = get_tool_fn("echo")
        self.assertEqual(fn(value=42), {"echoed": 42})
        self.assertIn("echo", list_tool_fns())

    def test_unknown_tool_fn_raises(self):
        with self.assertRaises(KeyError):
            get_tool_fn("not_registered")


class TestDeterministicStep(unittest.TestCase):
    """Test the deterministic-step plumbing without touching MCP."""

    def setUp(self):
        self._fn_snap = dict(TOOL_FN_REGISTRY)

    def tearDown(self):
        TOOL_FN_REGISTRY.clear()
        TOOL_FN_REGISTRY.update(self._fn_snap)

    def test_dispatch_with_inputs_and_config(self):
        @register_tool_fn("join_words")
        def _join(words, separator=", ", config=None):
            return {"joined": separator.join(words), "sep": config.get("sep") if config else None}

        ctx = SharedContext(abstract="x", extracted_keywords=["alpha", "beta"])
        cfg = StepConfig(
            id="joiner",
            type="deterministic",
            inputs={"words": "${extracted_keywords}"},
            outputs={"extra.joined_phrase": "result.joined"},
            raw={
                "function": "join_words",
                "config": {"sep": "|"},
                "inputs": {"words": "${extracted_keywords}"},
                "outputs": {"extra.joined_phrase": "result.joined"},
            },
        )
        step = DeterministicStep(cfg)
        result = step.execute(ctx)

        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(ctx.extra.get("joined_phrase"), "alpha, beta")
        # Step output is also stored under step_results[step_id] for ${steps.joiner.x}
        self.assertIn("joiner", ctx.step_results)

    def test_missing_function_errors(self):
        cfg = StepConfig(id="bad", type="deterministic", raw={})
        step = DeterministicStep(cfg)
        result = step.execute(SharedContext())
        self.assertFalse(result.success)
        self.assertIn("function", result.error)


class TestLLMAgentStep(unittest.TestCase):
    def test_llm_step_runs_with_placeholders(self):
        ctx = SharedContext(
            abstract="Bibliothek und Toxikologie.",
            provider="test",
            model="test-model",
        )

        # Mock AgentLoop at the module where LLMAgentStep imports it
        fake_loop = MagicMock()
        fake_loop.run.return_value = _agent_result(
            '```json\n{"keywords": ["Bibliothek", "Toxikologie"], "title": "T"}\n```'
        )

        with patch(
            "src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop
        ):
            cfg = StepConfig(
                id="extraction",
                type="llm_agent",
                inputs={"abstract": "${abstract}"},
                outputs={
                    "extracted_keywords": "response.keywords",
                    "working_title": "response.title",
                },
                raw={
                    "system_prompt": "You are a librarian.",
                    "user_prompt": "Analyse: {abstract}",
                    "inputs": {"abstract": "${abstract}"},
                    "outputs": {
                        "extracted_keywords": "response.keywords",
                        "working_title": "response.title",
                    },
                    "llm": {"temperature": 0.4, "max_tokens": 1024},
                },
            )
            step = LLMAgentStep(
                cfg,
                llm_service=MagicMock(),
                tool_registry=MagicMock(),
            )
            result = step.execute(ctx)

        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(ctx.extracted_keywords, ["Bibliothek", "Toxikologie"])
        self.assertEqual(ctx.working_title, "T")

        # Check the user prompt was rendered with {abstract}
        _, kwargs = fake_loop.run.call_args
        self.assertIn("Bibliothek und Toxikologie.", kwargs["user_prompt"])
        self.assertEqual(kwargs["temperature"], 0.4)

    def test_tool_preset_expansion(self):
        cfg = StepConfig(id="s", type="llm_agent", raw={"tools": {"preset": "gnd"}})
        step = LLMAgentStep(cfg)
        tools = step._resolve_tools({"preset": "gnd"})
        self.assertIn("search_gnd", tools)
        self.assertIn("get_gnd_entry", tools)

    def test_explicit_tools_override_preset(self):
        cfg = StepConfig(id="s", type="llm_agent", raw={})
        step = LLMAgentStep(cfg)
        tools = step._resolve_tools({"preset": "gnd", "explicit": ["search_swb"]})
        self.assertEqual(tools, ["search_swb"])


class TestWorkflowLoader(unittest.TestCase):
    def test_parse_valid_v4_yaml(self):
        yaml_src = textwrap.dedent("""
            name: "Test WF"
            version: "4.0"
            steps:
              - id: step1
                type: llm_agent
                system_prompt: "hi"
                inputs:
                  x: "${abstract}"
              - id: step2
                type: deterministic
                function: "join_words"
                inputs:
                  words: "${steps.step1.response.keywords}"
        """)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.yaml"
            path.write_text(yaml_src, encoding="utf-8")
            wf = load_workflow(path)

        self.assertEqual(wf.name, "Test WF")
        self.assertEqual(len(wf.steps), 2)
        self.assertEqual(wf.steps[0].id, "step1")
        self.assertEqual(wf.steps[1].type, "deterministic")
        self.assertTrue(wf.is_v4)

    def test_missing_type_raises(self):
        with self.assertRaises(ValueError):
            parse_steps([{"id": "x"}])

    def test_unknown_type_strict_raises(self):
        with self.assertRaises(ValueError):
            parse_steps([{"id": "x", "type": "unknown_type"}])

    def test_unknown_type_nonstrict_ok(self):
        steps = parse_steps([{"id": "x", "type": "unknown_type"}], strict=False)
        self.assertEqual(steps[0].type, "unknown_type")

    def test_duplicate_id_raises(self):
        with self.assertRaises(ValueError):
            parse_steps([
                {"id": "a", "type": "llm_agent"},
                {"id": "a", "type": "deterministic", "function": "f"},
            ])


class TestWorkflowExecutor(unittest.TestCase):
    def setUp(self):
        self._fn_snap = dict(TOOL_FN_REGISTRY)

    def tearDown(self):
        TOOL_FN_REGISTRY.clear()
        TOOL_FN_REGISTRY.update(self._fn_snap)

    def test_end_to_end_deterministic_only(self):
        """Two deterministic steps chained via ${steps.X.result.Y}."""
        @register_tool_fn("upper")
        def _upper(text):
            return {"upper": text.upper()}

        @register_tool_fn("wrap")
        def _wrap(text, config=None):
            marker = (config or {}).get("marker", "*")
            return {"wrapped": f"{marker}{text}{marker}"}

        yaml_src = textwrap.dedent("""
            name: "Pipe"
            version: "4.0"
            steps:
              - id: up
                type: deterministic
                function: "upper"
                inputs:
                  text: "${abstract}"
              - id: wr
                type: deterministic
                function: "wrap"
                inputs:
                  text: "${steps.up.result.upper}"
                config:
                  marker: "#"
                outputs:
                  extra.final: "result.wrapped"
        """)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.yaml"
            path.write_text(yaml_src, encoding="utf-8")
            wf = load_workflow(path)

        ctx = SharedContext(abstract="hello")
        executor = WorkflowExecutor()
        report = executor.run(wf, ctx)

        self.assertTrue(report.success, msg=report.error)
        self.assertEqual(len(report.step_results), 2)
        self.assertEqual(ctx.extra.get("final"), "#HELLO#")

    def test_only_step_filter(self):
        @register_tool_fn("inc")
        def _inc(value, config=None):
            return {"value": value + 1}

        from src.core.agents.steps.base_step import StepConfig
        cfg_a = StepConfig(
            id="a", type="deterministic",
            inputs={"value": "${extra.n}"},
            outputs={"extra.a_out": "result.value"},
            raw={"function": "inc", "inputs": {"value": "${extra.n}"},
                 "outputs": {"extra.a_out": "result.value"}},
        )
        cfg_b = StepConfig(
            id="b", type="deterministic",
            inputs={"value": "${extra.n}"},
            outputs={"extra.b_out": "result.value"},
            raw={"function": "inc", "inputs": {"value": "${extra.n}"},
                 "outputs": {"extra.b_out": "result.value"}},
        )

        from src.core.agents.workflow_loader import WorkflowDef
        wf = WorkflowDef(name="x", version="4.0", steps=[cfg_a, cfg_b], raw={"steps": []})
        ctx = SharedContext(extra={"n": 5})

        executor = WorkflowExecutor()
        report = executor.run(wf, ctx, only_step="b")
        self.assertEqual(len(report.step_results), 1)
        self.assertEqual(ctx.extra.get("b_out"), 6)
        self.assertNotIn("a_out", ctx.extra)


class TestSharedContextExtra(unittest.TestCase):
    def test_extra_persists_through_serialization(self):
        ctx = SharedContext(abstract="x", extra={"input": {"q": 1}, "other": [1, 2]})
        blob = ctx.to_dict()
        ctx2 = SharedContext.from_dict(blob)
        self.assertEqual(ctx2.extra, {"input": {"q": 1}, "other": [1, 2]})


class TestAlimaClassicMigration(unittest.TestCase):
    """Validate workflows/alima_classic.yaml loads + all referenced fns/steps registered."""

    def test_alima_classic_loads_and_resolves(self):
        # Side-effect imports register built-in fns + step types.
        from src.core.agents import steps as _steps  # noqa: F401
        from src.core.agents import deterministic_functions as _fns  # noqa: F401

        wf = load_workflow(Path("workflows/alima_classic.yaml"))
        self.assertEqual(wf.version, "4.0")
        step_ids = [s.id for s in wf.steps]
        self.assertEqual(
            step_ids,
            [
                "extraction", "search", "selection_chunks",
                "selection", "dk_collect", "classification", "dk_postprocess",
            ],
        )
        # Every step type resolves
        for s in wf.steps:
            get_step_class(s.type)
        # Deterministic fns referenced in YAML must be registered
        det_fns = {s.raw.get("function") for s in wf.steps if s.type == "deterministic"}
        self.assertEqual(det_fns, {"gnd_batch_search", "dk_search_agentic", "build_dk_search_results"})
        for fn in det_fns:
            self.assertIsNotNone(get_tool_fn(fn))

    def test_llm_chunked_selection(self):
        """Chunked LLMAgentStep slices input list and merges per-chunk responses."""
        ctx = SharedContext(
            abstract="Abstract zu X.",
            provider="p", model="m",
        )
        # 3 items × chunk_size=2 → 2 chunks. First chunk returns [A,B], second [C,A].
        # Expected merged (dedup_field=title): [A, B, C].
        fake_loop = MagicMock()
        fake_loop.run.side_effect = [
            _agent_result('```json\n{"keywords": [{"title": "A"}, {"title": "B"}]}\n```'),
            _agent_result('```json\n{"keywords": [{"title": "C"}, {"title": "A"}]}\n```'),
        ]
        with patch(
            "src.core.agents.steps.llm_agent_step.AgentLoop", return_value=fake_loop
        ):
            cfg = StepConfig(
                id="sel",
                type="llm_agent",
                inputs={"gnd_entries": "${extra.pool}"},
                outputs={"extra.picked": "response.keywords"},
                raw={
                    "system_prompt": "sys",
                    "user_prompt": "chunk {chunk_index}/{chunk_total}: {gnd_entries}",
                    "inputs": {"gnd_entries": "${extra.pool}"},
                    "outputs": {"extra.picked": "response.keywords"},
                    "chunking": {
                        "enabled": True,
                        "chunk_field": "gnd_entries",
                        "chunk_size": 2,
                        "merge_key": "keywords",
                        "dedup_field": "title",
                    },
                },
            )
            ctx.extra["pool"] = [
                {"title": "X1", "count": 10},
                {"title": "X2", "count": 5},
                {"title": "X3", "count": 1},
            ]
            step = LLMAgentStep(cfg, llm_service=MagicMock(), tool_registry=MagicMock())
            result = step.execute(ctx)

        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(fake_loop.run.call_count, 2)
        picked = ctx.extra.get("picked")
        self.assertEqual([p["title"] for p in picked], ["A", "B", "C"])

    def test_gnd_batch_search_fn(self):
        """gnd_batch_search: parses SWB/Lobid batch responses, merges pool, enriches."""
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("gnd_batch_search")

        tool_registry = MagicMock()

        def _exec(tool, args):
            if tool == "search_swb":
                return json.dumps({"results": {
                    "kw1": {"Titel1": {"gndid": ["123-4"], "count": 5, "ddc": ["540"], "dk": []}},
                }})
            if tool == "search_lobid":
                return json.dumps({"results": {
                    "kw1": {"Titel1": {"gndid": ["999-9"], "count": 3, "ddc": [], "dk": ["DK1"]}},
                    "kw2": {"Titel2": {"gndid": ["456-7"], "count": 2, "ddc": [], "dk": []}},
                }})
            if tool == "get_gnd_batch":
                return json.dumps({"entries": {
                    "123-4": {"description": "desc1", "synonyms": ["syn1"]},
                }})
            return "{}"

        tool_registry.execute.side_effect = _exec
        ctx = SharedContext(abstract="a")

        out = fn(
            keywords=["kw1", "kw2"],
            tool_registry=tool_registry,
            context=ctx,
        )
        titles = {e["title"] for e in out["entries"]}
        self.assertEqual(titles, {"Titel1", "Titel2"})
        # Title1 merged GND IDs from both sources
        t1 = next(e for e in out["entries"] if e["title"] == "Titel1")
        self.assertEqual(set(t1["gnd_ids"]), {"123-4", "999-9"})
        # Enriched description applied
        self.assertEqual(t1["description"], "desc1")
        # Context updated
        self.assertEqual(len(ctx.gnd_entries), 2)
        # 3 tool calls (swb, lobid, get_gnd_batch)
        self.assertEqual(out["tool_calls"], 3)


class TestPoCWorkflows(unittest.TestCase):
    """Phase-3 proof-of-concept workflows: catalog_search, synonym_expansion, batch_metadata."""

    @classmethod
    def setUpClass(cls):
        # Side-effect imports register built-in fns + step types.
        from src.core.agents import steps as _steps  # noqa: F401
        from src.core.agents import deterministic_functions as _fns  # noqa: F401

    def test_poc_workflows_load(self):
        """All 3 PoC YAMLs load + referenced fns/steps are registered."""
        for path in (
            "workflows/catalog_search.yaml",
            "workflows/synonym_expansion.yaml",
            "workflows/batch_metadata.yaml",
        ):
            wf = load_workflow(Path(path))
            self.assertEqual(wf.version, "4.0", msg=path)
            for s in wf.steps:
                get_step_class(s.type)
                if s.type == "deterministic":
                    get_tool_fn(s.raw.get("function"))

    def test_catalog_multi_search_fn(self):
        """catalog_multi_search: fan-out over SWB/Lobid/catalog, merged+ranked hits."""
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("catalog_multi_search")

        tool_registry = MagicMock()

        def _exec(tool, args):
            if tool == "search_swb":
                return json.dumps({"results": {
                    "q": {"Titel A": {"gndid": ["111"], "count": 10, "ddc": [], "dk": []}},
                }})
            if tool == "search_lobid":
                return json.dumps({"results": {
                    "q": {"Titel A": {"gndid": ["222"], "count": 3, "ddc": [], "dk": []},
                          "Titel B": {"gndid": ["333"], "count": 1, "ddc": [], "dk": []}},
                }})
            if tool == "search_catalog":
                return json.dumps({"results": {
                    "q": {"Titel C": {"gndid": ["444"], "count": 2, "ddc": [], "dk": []}},
                }})
            if tool == "get_gnd_batch":
                return json.dumps({"entries": {}})
            return "{}"

        tool_registry.execute.side_effect = _exec
        out = fn(queries=["q"], tool_registry=tool_registry)

        titles = {h["title"] for h in out["hits"]}
        self.assertEqual(titles, {"Titel A", "Titel B", "Titel C"})
        # Ranked by count desc → Titel A first (10)
        self.assertEqual(out["hits"][0]["title"], "Titel A")
        # Multi-source union on Titel A
        a = next(h for h in out["hits"] if h["title"] == "Titel A")
        self.assertEqual(set(a["gnd_ids"]), {"111", "222"})
        self.assertEqual(set(a["sources"]), {"swb", "lobid"})

    def test_gnd_entry_lookup_fn(self):
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("gnd_entry_lookup")

        tool_registry = MagicMock()
        tool_registry.execute.return_value = json.dumps({
            "term": "Nachhaltigkeit",
            "count": 2,
            "entries": [
                {"gnd_id": "4326464-5", "title": "Nachhaltigkeit",
                 "description": "desc", "synonyms": ["Sustainability"], "ddcs": ["333"]},
                {"gnd_id": "9999-9", "title": "Andere", "description": "", "synonyms": [], "ddcs": []},
            ],
        })

        out = fn(keyword="Nachhaltigkeit", tool_registry=tool_registry)
        self.assertTrue(out["found"])
        self.assertEqual(out["gnd_id"], "4326464-5")
        self.assertEqual(out["synonyms"], ["Sustainability"])
        self.assertEqual(len(out["alternatives"]), 1)

    def test_gnd_entry_lookup_empty(self):
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("gnd_entry_lookup")
        tool_registry = MagicMock()
        tool_registry.execute.return_value = json.dumps({"entries": []})
        out = fn(keyword="x", tool_registry=tool_registry)
        self.assertFalse(out["found"])
        self.assertEqual(out["entry"], {})

    def test_extract_gnd_related_fn(self):
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("extract_gnd_related")
        out = fn(entry={
            "gnd_id": "123", "title": "Foo", "description": "d",
            "synonyms": ["s1", "s2", ""], "ddcs": ["540", ""],
        })
        self.assertEqual(out["synonyms"], ["s1", "s2"])
        self.assertEqual(out["ddcs"], ["540"])
        self.assertEqual(out["gnd_id"], "123")

    def test_gnd_batch_metadata_fn_no_fallback(self):
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("gnd_batch_metadata")
        tool_registry = MagicMock()
        tool_registry.execute.return_value = json.dumps({
            "entries": {
                "4037944-9": {"title": "T1", "description": "d1", "synonyms": [], "ddcs": []},
            }
        })
        out = fn(gnd_ids=["4037944-9", "missing"], tool_registry=tool_registry)
        self.assertEqual(list(out["entries"]), ["4037944-9"])
        self.assertEqual(out["missing"], ["missing"])
        self.assertEqual(out["tool_calls"], 1)

    def test_gnd_batch_metadata_lobid_fallback(self):
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("gnd_batch_metadata")
        tool_registry = MagicMock()

        def _exec(tool, args):
            if tool == "get_gnd_batch":
                return json.dumps({"entries": {}})
            if tool == "search_lobid":
                return json.dumps({"results": {
                    args["terms"][0]: {"Recovered": {"gndid": [args["terms"][0]], "count": 1, "ddc": ["540"]}},
                }})
            return "{}"

        tool_registry.execute.side_effect = _exec
        out = fn(gnd_ids=["4053309-8"], tool_registry=tool_registry, lobid_fallback=True)
        self.assertIn("4053309-8", out["entries"])
        self.assertEqual(out["entries"]["4053309-8"]["title"], "Recovered")
        self.assertEqual(out["missing"], [])

    def test_gnd_batch_search_accepts_dicts(self):
        """Validate step in synonym_expansion feeds LLM candidate dicts straight in."""
        from src.core.agents.registry import get_tool_fn
        fn = get_tool_fn("gnd_batch_search")
        tool_registry = MagicMock()

        captured: list = []

        def _exec(tool, args):
            if tool in ("search_swb", "search_lobid"):
                captured.append(args["terms"])
                return json.dumps({"results": {}})
            return json.dumps({"entries": {}})

        tool_registry.execute.side_effect = _exec
        fn(
            keywords=[{"term": "Ökologie", "relation": "related"}, "Klima", {"title": "Umwelt"}],
            tool_registry=tool_registry,
            context=SharedContext(),
        )
        # Both sources received the same dict→str coerced list
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0], ["Ökologie", "Klima", "Umwelt"])
        self.assertEqual(captured[0], captured[1])


class TestCliWorkflowDispatch(unittest.TestCase):
    """End-to-end dispatch via ``workflow_cmd.handle_workflow``.

    Uses the real ``batch_metadata.yaml`` (deterministic-only) with a mocked
    tool registry so no network/DB is needed.
    """

    def _ns(self, **kw):
        import argparse
        return argparse.Namespace(**kw)

    def test_handle_workflow_batch_metadata(self):
        from src.cli.commands import workflow_cmd

        mock_registry = MagicMock()

        def _exec(tool, args):
            if tool == "get_gnd_batch":
                return json.dumps({"entries": {
                    "4037944-9": {
                        "id": "4037944-9",
                        "title": "Mathematik",
                        "description": "",
                        "synonyms": [],
                        "ddcs": ["510"],
                    },
                }})
            return "{}"

        mock_registry.execute.side_effect = _exec

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "report.json"
            args = self._ns(
                name="batch_metadata",
                input='{"gnd_ids": ["4037944-9"]}',
                input_file=None,
                output=str(out_path),
                provider="", model="", temperature=None,
                only_step=None, quiet=True,
            )
            with patch("src.cli.commands.workflow_cmd.create_caching_registry",
                       return_value=mock_registry):
                rc = workflow_cmd.handle_workflow(
                    args, config_manager=MagicMock(), llm_service=MagicMock(),
                    log=MagicMock(),
                )
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            report = json.loads(out_path.read_text())
            self.assertTrue(report["success"])
            self.assertEqual(len(report["steps"]), 1)
            extra = report["context"]["extra"]
            self.assertIn("metadata", extra)

    def test_handle_workflow_unknown_name(self):
        from src.cli.commands import workflow_cmd
        args = self._ns(
            name="does_not_exist_xyz",
            input=None, input_file=None, output=None,
            provider="", model="", temperature=None,
            only_step=None, quiet=True,
        )
        rc = workflow_cmd.handle_workflow(
            args, config_manager=MagicMock(), llm_service=MagicMock(),
            log=MagicMock(),
        )
        self.assertEqual(rc, 2)

    def test_handle_workflow_bad_input_json(self):
        from src.cli.commands import workflow_cmd
        args = self._ns(
            name="batch_metadata",
            input="not valid json{{{",
            input_file=None, output=None,
            provider="", model="", temperature=None,
            only_step=None, quiet=True,
        )
        rc = workflow_cmd.handle_workflow(
            args, config_manager=MagicMock(), llm_service=MagicMock(),
            log=MagicMock(),
        )
        self.assertEqual(rc, 2)

    def test_handle_workflows_list(self):
        from src.cli.commands import workflow_cmd
        rc = workflow_cmd.handle_workflows_list(self._ns(), MagicMock())
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
