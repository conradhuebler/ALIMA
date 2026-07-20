# tests/test_e2e_smoke.py
"""End-to-end smoke tests with mocked LLM service. Claude Generated (WP C).

Three previously untested core paths, each exercised end-to-end:

1. Classic pipeline (execute_complete_pipeline): initialisation → search →
   keywords with a mocked AlimaManager + fake SearchCLI → KeywordAnalysisState.
2. Agentic v4 workflow (alima_classic.yaml) through WorkflowExecutor with a
   mocked LlmService (LLMAgentStep + AgentLoop) and faked deterministic
   functions — all 7 steps run, report.success is True.
3. AgentLoop multi-turn: 2 tool calls + final answer over 3 LLM turns.

The LLM is mocked at the LlmService boundary (generate_with_tools /
analyze_abstract); network access is faked at the SearchCLI / deterministic-
function boundary. Everything in between is real code.
"""

import copy
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_models import (
    AbstractData,
    AgentResponse,
    AnalysisResult,
    PromptConfigData,
    TaskState,
    ToolCall,
)
from src.utils.pipeline_utils import PipelineStepExecutor

logging.disable(logging.CRITICAL)

ABSTRACT = "Limnologische Studien zur Seenkunde im Alpenraum."
GND_KEYWORD = "Limnologie (GND-ID: 4035769-7)"

# ONE reduced suggester payload (plugin contract v2) that both the classic and the
# agentic path ingest, so the WP-D1 P0 convergence test compares like with like.
# Deliberately non-trivial — a trivial fixture would pass even if a path dropped
# fields: - Claude Generated
#   * two classification systems (a single-system fixture hides a system-key drift)
#   * display_count != count (the exact shape of the 038738e counter bug: pool
#     ``count`` is the ranking placeholder, ``display_count`` the real Häufigkeit)
#   * two gnd_ids on one entry (exercises the merge/dedup + gnd_id-convenience path)
# Nested views carry sets, pool entries carry lists — an intentional, pinned
# difference (docs/wp_records_as_first_class.md "Not-a-bug"), so comparisons
# normalise containers and assert on values.
CONVERGENCE_HITS = {
    "Limnologie": {
        "count": 1,
        "display_count": 17,
        "gnd_ids": {"4035769-7", "4127654-7"},
        "classifications": {"DK": {"556.55"}, "DDC": {"551.48"}},
    },
    "Seenkunde": {
        "count": 1,
        "display_count": 4,
        "gnd_ids": {"4180168-1"},
        "classifications": {"DDC": {"551.48"}},
    },
}


def _task_state(task: str, full_text: str) -> TaskState:
    return TaskState(
        abstract_data=AbstractData(abstract=ABSTRACT, keywords=""),
        analysis_result=AnalysisResult(
            full_text=full_text, matched_keywords={}, gnd_systematic=""
        ),
        prompt_config=PromptConfigData(
            prompt="p", system="s", temp=0.7, p_value=0.9, models=["m"], seed=42
        ),
        status="completed",
        task_name=task,
        model_used="mock-model",
        provider_used="mock-provider",
    )


class _FakeSearchCLI:
    """Stands in for SearchCLI: canned GND hits, no network."""

    def __init__(self, *args, **kwargs):
        self.last_errors = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def search(self, search_terms, suggester_types):
        # Deep-copy per term: the pipeline may mutate the payloads in place, and a
        # shared fixture object would let one term's mutation leak into the next
        # (and into the agentic side of the convergence test). - Claude Generated
        return {
            term: copy.deepcopy(CONVERGENCE_HITS) for term in search_terms
        }

    # WP2 P4.4b: execute_gnd_search now calls search_from_raw by default; the
    # fake returns the same canned hits (raw-derivation is covered elsewhere).
    search_from_raw = search


class TestClassicPipelineEndToEnd(unittest.TestCase):
    """All classic steps chained: initialisation → search → keywords."""

    def _analyze_abstract(self, abstract_data=None, task="", **kwargs):
        if task == "initialisation":
            return _task_state(
                task, "<final_list>Limnologie | Seenkunde</final_list><class>31</class>"
            )
        # keywords (final analysis)
        return _task_state(task, f"<final_list>{GND_KEYWORD}</final_list>")

    def test_complete_pipeline_produces_state_with_keywords(self):
        from src.core.pipeline_manager import PipelineConfig

        alima_manager = Mock()
        alima_manager.analyze_abstract.side_effect = self._analyze_abstract
        cache_manager = Mock()
        cache_manager.get_gnd_facts_batch.return_value = {}

        executor = PipelineStepExecutor(
            alima_manager=alima_manager,
            cache_manager=cache_manager,
            logger=Mock(level=100),
        )

        config = PipelineConfig(
            step_configs={
                "initialisation": {"provider": "mock-provider", "model": "mock-model"},
                "keywords": {
                    "provider": "mock-provider",
                    "model": "mock-model",
                    # explicit threshold: skip model-capability auto-detection
                    "custom_params": {"keyword_chunking_threshold": 500},
                },
                "dk_classification": {"enabled": False},
            }
        )

        stream_lines = []
        with patch("src.utils.pipeline_utils.SearchCLI", _FakeSearchCLI):
            state = executor.execute_complete_pipeline(
                ABSTRACT,
                pipeline_config=config,
                stream_callback=lambda tok, sid: stream_lines.append((sid, tok)),
            )

        # Step 1: initial keywords extracted
        self.assertEqual(state.original_abstract, ABSTRACT)
        self.assertIn("Limnologie", state.initial_keywords)
        self.assertIn("31", state.initial_gnd_classes)
        # Step 2: search results present for the extracted keywords
        self.assertTrue(state.search_results)
        first_term_results = next(iter(state.search_results.values()))
        self.assertIn("Limnologie", first_term_results)
        # Step 3: final analysis produced GND-validated keywords
        self.assertIsNotNone(state.final_llm_analysis)
        self.assertTrue(state.final_llm_analysis.extracted_gnd_keywords)
        # Both LLM steps were actually called
        tasks_called = [
            c.kwargs.get("task") for c in alima_manager.analyze_abstract.call_args_list
        ]
        self.assertEqual(tasks_called, ["initialisation", "keywords"])
        # Streaming reached the caller for every step
        step_ids = {sid for sid, _ in stream_lines}
        self.assertTrue({"initialisation", "search", "keywords"} <= step_ids)


class TestAgenticWorkflowEndToEnd(unittest.TestCase):
    """alima_classic.yaml through WorkflowExecutor with mocked LLM + tools."""

    @staticmethod
    def _generate_with_tools(provider="", model="", messages=None, **kwargs):
        """Route canned JSON by the step's system prompt (order-independent)."""
        system = ""
        for m in messages or []:
            if m.get("role") == "system":
                system = m.get("content", "")
                break
        if "Selektiver GND-Experte" in system:
            content = json.dumps(
                {"keywords": [{"keyword": "Limnologie", "gnd_id": "4035769-7"}]}
            )
        elif "Schlagwortketten" in system:
            content = json.dumps({
                "keyword_chains": [{"chain": ["Limnologie"], "reason": "Kernthema"}],
                "missing_concepts": [],
                "final_keywords": [{"keyword": "Limnologie", "gnd_id": "4035769-7"}],
            })
        elif "Klassifikations-Experte" in system:
            content = json.dumps({
                "classifications": [{"code": "DK 556.55", "type": "DK"}],
                "analyse": "Limnologie.",
            })
        else:  # extraction
            content = json.dumps(
                {"title": "Alpen_Limnologie", "keywords": ["Limnologie", "Seenkunde"]}
            )
        return AgentResponse(content=content)

    @staticmethod
    def _fake_tool_fn(name):
        def gnd_batch_search(keywords=None, config=None, tool_registry=None,
                             context=None, stream_callback=None, **kw):
            return {"entries": [
                {"keyword": "Limnologie", "title": "Limnologie",
                 "gnd_id": "4035769-7", "count": 3},
            ]}

        def finc_subject_harvest(keywords=None, config=None, tool_registry=None,
                                 context=None, stream_callback=None, **kw):
            # Opt-in step; default-disabled no-op shape. - Claude Generated
            return {"entries": [], "harvested_terms": [], "subjects_reconciled": 0,
                    "merged_added": 0, "tool_calls": 0, "enabled": False}

        def verify_final_keywords(config=None, tool_registry=None, context=None,
                                  stream_callback=None, **kw):
            return {
                "verified_keywords": [{"keyword": "Limnologie", "gnd_id": "4035769-7"}],
                "rejected": [],
                "stats": {"total_extracted": 1, "verified_count": 1},
            }

        def dk_search_agentic(config=None, tool_registry=None, context=None,
                              stream_callback=None, **kw):
            return {
                "dk_entries": [{"code": "DK 556.55", "count": 2}],
                "formatted_prompt": "DK: 556.55 (Häufigkeit: 2) | Limnologie",
                "dk_search_results": [{"keyword": "Limnologie", "dk": "556.55"}],
            }

        def build_dk_search_results(dk_entries=None, dk_classifications=None,
                                    config=None, tool_registry=None, context=None,
                                    stream_callback=None, **kw):
            return {"results": [{"code": "DK 556.55", "titles": []}]}

        fns = {
            "gnd_batch_search": gnd_batch_search,
            "finc_subject_harvest": finc_subject_harvest,
            "verify_final_keywords": verify_final_keywords,
            "dk_search_agentic": dk_search_agentic,
            "build_dk_search_results": build_dk_search_results,
        }
        if name not in fns:
            raise KeyError(f"unexpected deterministic function: {name}")
        return fns[name]

    def test_alima_classic_workflow_runs_all_steps(self):
        from src.core.agents.shared_context import SharedContext
        from src.core.agents.workflow_executor import WorkflowExecutor
        from src.core.agents.workflow_loader import load_workflow

        workflow = load_workflow(
            Path(__file__).parent.parent / "workflows" / "alima_classic.yaml"
        )

        llm_service = Mock()
        llm_service.generate_with_tools.side_effect = self._generate_with_tools
        tool_registry = Mock()
        tool_registry.get_tool_schemas.return_value = []

        context = SharedContext(abstract=ABSTRACT)
        context.provider = "mock-provider"
        context.model = "mock-model"

        executor = WorkflowExecutor(
            llm_service=llm_service, tool_registry=tool_registry
        )
        with patch(
            "src.core.agents.steps.deterministic_step.get_tool_fn",
            side_effect=self._fake_tool_fn,
        ):
            report = executor.run(workflow, context)

        failed = [r.step_id for r in report.step_results if not r.success]
        self.assertTrue(report.success, f"failed steps: {failed}; error: {report.error}")
        self.assertEqual(len(report.step_results), 9)  # incl. opt-in finc_harvest step
        # Context carries results through the whole chain
        self.assertEqual(context.working_title, "Alpen_Limnologie")
        self.assertIn("Limnologie", context.extracted_keywords)
        self.assertTrue(context.gnd_entries)
        self.assertTrue(context.selected_keywords)
        self.assertTrue(context.keyword_chains)
        self.assertTrue(context.dk_classifications)
        self.assertTrue(context.dk_search_results)
        # 4 LLM steps → 4 generate_with_tools calls (1 iteration each, no chunk split)
        self.assertEqual(llm_service.generate_with_tools.call_count, 4)

    def test_workflow_reports_failing_llm_step(self):
        """LLM failure in one step surfaces in the report, not as a crash."""
        from src.core.agents.shared_context import SharedContext
        from src.core.agents.workflow_executor import WorkflowExecutor
        from src.core.agents.workflow_loader import load_workflow

        workflow = load_workflow(
            Path(__file__).parent.parent / "workflows" / "alima_classic.yaml"
        )
        llm_service = Mock()
        llm_service.generate_with_tools.side_effect = RuntimeError("provider down")
        tool_registry = Mock()
        tool_registry.get_tool_schemas.return_value = []

        context = SharedContext(abstract=ABSTRACT)
        executor = WorkflowExecutor(llm_service=llm_service, tool_registry=tool_registry)
        with patch(
            "src.core.agents.steps.deterministic_step.get_tool_fn",
            side_effect=self._fake_tool_fn,
        ):
            report = executor.run(workflow, context)

        self.assertFalse(report.success)


def _normalise_pool_payload(payload):
    """Reduce one persisted GND-pool payload to its comparable values.

    Container types are deliberately NOT unified across the two paths (sets in
    nested per-term views for merge-dedup, lists in pool entries for display
    order — pinned as "Not-a-bug" in docs/wp_records_as_first_class.md), so the
    comparison normalises containers and asserts on the VALUES of the four
    canonical fields. - Claude Generated
    """
    return {
        "count": payload.get("count"),
        "display_count": payload.get("display_count"),
        "gnd_ids": set(payload.get("gnd_ids") or []),
        "classifications": {
            system: set(codes)
            for system, codes in (payload.get("classifications") or {}).items()
            if codes
        },
    }


def _normalise_search_results(search_results):
    """``{title: payload}`` from either path's persisted ``search_results``.

    The two paths persist different CONTAINERS — classic keeps the nested dict
    ``{term: {title: …}}``, agentic builds ``List[SearchResult]``. That split
    predates WP-D1 and is out of its scope; this helper flattens both so the
    test gates the payload fields P0 actually unified. - Claude Generated
    """
    flat = {}
    if isinstance(search_results, dict):
        per_term = search_results.values()
    else:
        per_term = [getattr(r, "results", {}) or {} for r in (search_results or [])]
    for results in per_term:
        for title, payload in (results or {}).items():
            flat[title] = _normalise_pool_payload(payload)
    return flat


class TestClassicAgenticPoolConvergence(unittest.TestCase):
    """WP-D1 P0 gate: both paths persist the SAME canonical GND-pool payload.

    P0 collapsed F-1 into one vocabulary ``{count, gnd_ids,
    classifications{system: codes}, display_count?}`` end-to-end as a HARD CUT —
    no tolerant legacy readers. The risk that buys is a silent shape drift
    between the classic and the agentic persistence path; a bug of exactly that
    class lived here before (counter divergence, fixed in 038738e, where
    ``to_keyword_analysis_state`` rebuilt search_results but dropped
    count/display_count).

    Both sides ingest the same ``CONVERGENCE_HITS`` fixture through REAL
    production code — the classic pipeline via ``execute_complete_pipeline``,
    the agentic side via ``pool_entry_from_reduced`` (THE nested→pool ingestion
    point) plus ``SharedContext.to_keyword_analysis_state``. Nothing in the
    conversion is re-implemented in the test, so the comparison is not circular.

    Scope caveat (CLAUDE.md): this gates the persistence/transport SHAPE on
    mocked data. It does not prove live behaviour, and it does not test the
    aggregation engine itself — that is covered by tests/test_core_convergence.py
    and tests/test_gnd_search_core.py.
    """

    def _classic_search_results(self):
        from src.core.pipeline_manager import PipelineConfig

        alima_manager = Mock()
        alima_manager.analyze_abstract.side_effect = (
            TestClassicPipelineEndToEnd._analyze_abstract
        ).__get__(TestClassicPipelineEndToEnd())
        cache_manager = Mock()
        cache_manager.get_gnd_facts_batch.return_value = {}

        executor = PipelineStepExecutor(
            alima_manager=alima_manager,
            cache_manager=cache_manager,
            logger=Mock(level=100),
        )
        config = PipelineConfig(
            step_configs={
                "initialisation": {"provider": "mock-provider", "model": "mock-model"},
                "keywords": {
                    "provider": "mock-provider",
                    "model": "mock-model",
                    "custom_params": {"keyword_chunking_threshold": 500},
                },
                "dk_classification": {"enabled": False},
            }
        )
        with patch("src.utils.pipeline_utils.SearchCLI", _FakeSearchCLI):
            state = executor.execute_complete_pipeline(ABSTRACT, pipeline_config=config)
        return state.search_results

    def _agentic_search_results(self):
        from src.core.agents.shared_context import SharedContext
        from src.core.gnd_search_core import pool_entry_from_reduced

        context = SharedContext(abstract=ABSTRACT)
        context.extracted_keywords = ["Limnologie"]
        # THE real nested→pool ingestion point — the same function the agentic
        # search step feeds its suggester payloads through.
        context.gnd_entries = [
            pool_entry_from_reduced(title, copy.deepcopy(data))
            for title, data in CONVERGENCE_HITS.items()
        ]
        context.gnd_entries_per_keyword = {
            "Limnologie": list(CONVERGENCE_HITS.keys())
        }
        return context.to_keyword_analysis_state().search_results

    def test_both_paths_persist_identical_pool_payloads(self):
        classic = _normalise_search_results(self._classic_search_results())
        agentic = _normalise_search_results(self._agentic_search_results())

        self.assertTrue(classic, "classic path persisted no search_results")
        self.assertTrue(agentic, "agentic path persisted no search_results")
        self.assertEqual(
            set(classic), set(agentic),
            "the two paths persist different GND-pool titles",
        )
        for title in classic:
            self.assertEqual(
                classic[title], agentic[title],
                f"pool payload for {title!r} diverges between classic and agentic",
            )

    def test_fixture_would_catch_a_dropped_field(self):
        """The fixture must be able to fail — a trivial one silently cannot.

        Guards the gate itself: if a later edit flattens CONVERGENCE_HITS (all
        counts equal, one system, one gnd_id), the convergence assertion above
        would still pass while no longer detecting a dropped field. - Claude Generated
        """
        payloads = [_normalise_pool_payload(p) for p in CONVERGENCE_HITS.values()]
        self.assertTrue(
            any(p["display_count"] != p["count"] for p in payloads),
            "fixture cannot detect a dropped display_count (038738e bug class)",
        )
        systems = {s for p in payloads for s in p["classifications"]}
        self.assertGreaterEqual(
            len(systems), 2, "fixture cannot detect a classification system-key drift"
        )
        self.assertTrue(
            any(len(p["gnd_ids"]) > 1 for p in payloads),
            "fixture cannot detect a dropped secondary gnd_id",
        )


class TestAgentLoopMultiTurn(unittest.TestCase):
    """AgentLoop: tool call → tool result → tool call → tool result → final."""

    def test_three_turns_with_two_tool_calls(self):
        from src.core.agent_loop import AgentLoop

        responses = [
            AgentResponse(
                content="Ich suche zuerst.",
                tool_calls=[ToolCall(id="t1", name="search_gnd",
                                     arguments={"term": "Limnologie"})],
            ),
            AgentResponse(
                tool_calls=[ToolCall(id="t2", name="get_gnd_entry",
                                     arguments={"gnd_id": "4035769-7"})],
            ),
            AgentResponse(content="Fertig: Limnologie (4035769-7)."),
        ]
        llm_service = Mock()
        llm_service.generate_with_tools.side_effect = responses

        tool_registry = Mock()
        tool_registry.get_tool_schemas.return_value = [
            {"name": "search_gnd"}, {"name": "get_gnd_entry"}
        ]
        tool_registry.execute.return_value = json.dumps({"title": "Limnologie"})

        seen_calls, seen_results = [], []
        loop = AgentLoop(
            llm_service=llm_service,
            tool_registry=tool_registry,
            max_iterations=5,
            on_tool_call=lambda tc: seen_calls.append(tc.name),
            on_tool_result=lambda name, res: seen_results.append(name),
        )
        result = loop.run(
            system_prompt="Du bist ein Test-Agent.",
            user_prompt="Finde Limnologie.",
            tools=[],
            provider="mock", model="mock",
        )

        self.assertEqual(result.content, "Fertig: Limnologie (4035769-7).")
        self.assertEqual(result.iterations, 3)
        self.assertEqual([e["tool"] for e in result.tool_log],
                         ["search_gnd", "get_gnd_entry"])
        self.assertEqual(seen_calls, ["search_gnd", "get_gnd_entry"])
        self.assertEqual(seen_results, ["search_gnd", "get_gnd_entry"])
        self.assertEqual(tool_registry.execute.call_count, 2)
        # Tool results were fed back into the conversation
        third_call_messages = llm_service.generate_with_tools.call_args_list[2].kwargs["messages"]
        roles = [m["role"] for m in third_call_messages]
        self.assertEqual(roles.count("tool"), 2)


if __name__ == "__main__":
    unittest.main()
