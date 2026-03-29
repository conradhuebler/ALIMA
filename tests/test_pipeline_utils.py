import logging
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_models import (
    AbstractData,
    AnalysisResult,
    LlmKeywordAnalysis,
    PromptConfigData,
    TaskState,
)

try:
    from src.utils.pipeline_utils import PipelineStepExecutor
    PIPELINE_UTILS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    PipelineStepExecutor = None
    PIPELINE_UTILS_IMPORT_ERROR = exc


logging.disable(logging.CRITICAL)


@unittest.skipIf(
    PIPELINE_UTILS_IMPORT_ERROR is not None,
    f"Pipeline utilities dependencies unavailable: {PIPELINE_UTILS_IMPORT_ERROR}",
)
class TestPipelineStepExecutor(unittest.TestCase):
    def setUp(self):
        self.mock_alima_manager = Mock()
        self.mock_cache_manager = Mock()
        self.mock_logger = Mock()
        self.executor = PipelineStepExecutor(
            alima_manager=self.mock_alima_manager,
            cache_manager=self.mock_cache_manager,
            logger=self.mock_logger,
        )

    def test_execute_initial_keyword_extraction(self):
        mock_task_state = TaskState(
            abstract_data=AbstractData(abstract="This is a test abstract.", keywords=""),
            analysis_result=AnalysisResult(
                full_text="<keywords>Machine Learning, AI</keywords><class>004</class>",
                matched_keywords={},
                gnd_systematic="",
            ),
            prompt_config=PromptConfigData(prompt="Test prompt", system="System prompt"),
            status="completed",
            task_name="initialisation",
            model_used="test-model",
            provider_used="test-provider",
        )
        self.mock_alima_manager.analyze_abstract.return_value = mock_task_state

        keywords, gnd_classes, llm_analysis, llm_title = self.executor.execute_initial_keyword_extraction(
            abstract_text="This is a test abstract.",
            model="test-model",
            provider="test-provider",
            task="initialisation",
        )

        self.mock_alima_manager.analyze_abstract.assert_called_once()
        _, call_kwargs = self.mock_alima_manager.analyze_abstract.call_args
        self.assertEqual(call_kwargs["task"], "initialisation")
        self.assertEqual(call_kwargs["model"], "test-model")
        self.assertEqual(call_kwargs["provider"], "test-provider")
        self.assertEqual(keywords, ["Machine Learning", "AI"])
        self.assertEqual(gnd_classes, ["004"])
        self.assertIsInstance(llm_analysis, LlmKeywordAnalysis)
        self.assertIsNone(llm_title)

    def test_create_complete_analysis_state_converts_search_results(self):
        initial_analysis = Mock(spec=LlmKeywordAnalysis)
        final_analysis = Mock(spec=LlmKeywordAnalysis)

        state = self.executor.create_complete_analysis_state(
            original_abstract="Test abstract for full state creation.",
            initial_keywords=["initial", "keywords"],
            initial_gnd_classes=["001"],
            search_results={
                "term1": {"kw1": {"gndid": {"1"}}},
                "term2": {"kw2": {"gndid": {"2"}}},
            },
            initial_llm_analysis=initial_analysis,
            final_llm_analysis=final_analysis,
            suggesters_used=["lobid"],
        )

        self.assertEqual(state.original_abstract, "Test abstract for full state creation.")
        self.assertEqual(state.initial_keywords, ["initial", "keywords"])
        self.assertEqual(state.initial_gnd_classes, ["001"])
        self.assertEqual(state.search_suggesters_used, ["lobid"])
        self.assertEqual(len(state.search_results), 2)
        self.assertEqual(state.search_results[0].search_term, "term1")
        self.assertEqual(state.search_results[1].search_term, "term2")

    def test_execute_final_keyword_analysis_uses_batch_gnd_lookup(self):
        self.mock_cache_manager.get_gnd_facts_batch.return_value = {
            "4061694-5": SimpleNamespace(title="Umweltverschmutzung", synonyms=""),
            "4009274-4": SimpleNamespace(title="Cadmium", synonyms=""),
        }
        self.mock_alima_manager.analyze_abstract.return_value = TaskState(
            abstract_data=AbstractData(abstract="Abstract", keywords=""),
            analysis_result=AnalysisResult(
                full_text="<keywords>Umweltverschmutzung (GND-ID: 4061694-5), Cadmium (GND-ID: 4009274-4)</keywords><class>21.4</class>",
                matched_keywords={},
                gnd_systematic="",
            ),
            prompt_config=PromptConfigData(prompt="Prompt", system="System"),
            status="completed",
            task_name="keywords",
            model_used="final-model",
            provider_used="final-provider",
        )

        final_keywords, gnd_classes, llm_analysis = self.executor.execute_final_keyword_analysis(
            original_abstract="Abstract",
            search_results={
                "environmental pollution": {
                    "Umweltverschmutzung": {"count": 50, "gndid": {"4061694-5"}},
                    "Cadmium": {"count": 20, "gndid": {"4009274-4"}},
                }
            },
            model="final-model",
            provider="final-provider",
            task="keywords",
            keyword_chunking_threshold=999,
        )

        self.mock_cache_manager.get_gnd_facts_batch.assert_called_once()
        self.assertEqual(gnd_classes, ["21.4"])
        self.assertIsInstance(llm_analysis, LlmKeywordAnalysis)
        self.assertTrue(any("4061694-5" in kw for kw in final_keywords))


if __name__ == "__main__":
    unittest.main()
