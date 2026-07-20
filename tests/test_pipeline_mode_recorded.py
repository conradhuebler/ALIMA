"""The exported state must name the pipeline that produced it - Claude Generated.

Comparing a classic against an agentic export used to require GUESSING the mode
from side effects — agentic runs happen to carry `keyword_chains` and a
`verification` block, classic ones do not. That is an accident of the current
step set, not a contract, so `pipeline_mode` is recorded explicitly (plus
`workflow_name` for agentic runs) and surfaced in `pipeline_metadata`.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.core.agents.shared_context import SharedContext
from src.core.data_models import KeywordAnalysisState
from src.webapp.result_serialization import extract_results_from_analysis_state


class TestAgenticStateIsLabelled(unittest.TestCase):
    def test_to_keyword_analysis_state_marks_agentic(self):
        ctx = SharedContext(abstract="Ein Abstract.")
        ctx.extracted_keywords = ["Limnologie"]
        state = ctx.to_keyword_analysis_state()
        self.assertEqual(state.pipeline_mode, "agentic")

    def test_workflow_name_travels_from_the_executor(self):
        """WorkflowExecutor.run names the workflow on the context."""
        from src.core.agents.workflow_executor import WorkflowExecutor

        ctx = SharedContext(abstract="Ein Abstract.")
        ctx.extracted_keywords = ["Limnologie"]
        workflow = Mock()
        workflow.name = "alima_classic"
        workflow.prompts = {}
        workflow.settings = {}
        workflow.steps = []

        executor = WorkflowExecutor(llm_service=Mock(), tool_registry=Mock())
        executor.run(workflow, ctx)

        self.assertEqual(ctx.workflow_name, "alima_classic")
        self.assertEqual(ctx.to_keyword_analysis_state().workflow_name, "alima_classic")

    def test_workflow_name_is_none_when_unnamed(self):
        """An unnamed context must export None, not an empty string."""
        ctx = SharedContext(abstract="A")
        ctx.extracted_keywords = ["X"]
        self.assertIsNone(ctx.to_keyword_analysis_state().workflow_name)


class TestClassicStateIsLabelled(unittest.TestCase):
    def test_execute_complete_pipeline_marks_classic(self):
        import tests.test_e2e_smoke as smoke
        from src.core.pipeline_manager import PipelineConfig
        from src.utils.pipeline_utils import PipelineStepExecutor

        alima_manager = Mock()
        alima_manager.analyze_abstract.side_effect = (
            smoke.TestClassicPipelineEndToEnd()._analyze_abstract
        )
        cache_manager = Mock()
        cache_manager.get_gnd_facts_batch.return_value = {}
        executor = PipelineStepExecutor(
            alima_manager=alima_manager, cache_manager=cache_manager, logger=Mock(level=100)
        )
        config = PipelineConfig(
            step_configs={
                "initialisation": {"provider": "p", "model": "m"},
                "keywords": {
                    "provider": "p", "model": "m",
                    "custom_params": {"keyword_chunking_threshold": 500},
                },
                "dk_classification": {"enabled": False},
            }
        )
        with patch("src.utils.pipeline_utils.SearchCLI", smoke._FakeSearchCLI):
            state = executor.execute_complete_pipeline(smoke.ABSTRACT, pipeline_config=config)

        self.assertEqual(state.pipeline_mode, "classic")
        self.assertIsNone(state.workflow_name)


class TestExportCarriesTheMode(unittest.TestCase):
    def test_pipeline_metadata_exposes_mode_and_workflow(self):
        state = KeywordAnalysisState(
            original_abstract="A",
            initial_keywords=[],
            search_suggesters_used=["lobid"],
            pipeline_mode="agentic",
            workflow_name="alima_classic",
        )
        meta = extract_results_from_analysis_state(state)["pipeline_metadata"]
        self.assertEqual(meta["pipeline_mode"], "agentic")
        self.assertEqual(meta["workflow_name"], "alima_classic")

    def test_older_states_without_the_field_export_none(self):
        """Exports written before this field existed must not break."""
        legacy = Mock(spec=["search_suggesters_used", "initial_gnd_classes"])
        legacy.search_suggesters_used = ["lobid"]
        legacy.initial_gnd_classes = []
        meta = extract_results_from_analysis_state(legacy)["pipeline_metadata"]
        self.assertIsNone(meta["pipeline_mode"])
        self.assertIsNone(meta["workflow_name"])


if __name__ == "__main__":
    unittest.main()
