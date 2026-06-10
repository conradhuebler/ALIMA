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
        return {
            term: {
                "Limnologie": {
                    "count": 3,
                    "gndid": {"4035769-7"},
                    "ddc": set(),
                    "dk": set(),
                }
            }
            for term in search_terms
        }


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
        self.assertEqual(len(report.step_results), 7)
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
