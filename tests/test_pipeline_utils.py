# tests/test_pipeline_utils.py

import unittest
from unittest.mock import Mock, patch
import logging

# Add the project root to the Python path for src imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.pipeline_utils import PipelineStepExecutor
from src.core.data_models import AbstractData, TaskState, AnalysisResult, PromptConfigData

# Disable logging for tests
logging.disable(logging.CRITICAL)

class TestPipelineStepExecutor(unittest.TestCase):
    """Unit tests for the PipelineStepExecutor class."""

    def setUp(self):
        """Set up a fresh test environment before each test."""
        # Create mock objects for dependencies
        self.mock_alima_manager = Mock()
        self.mock_cache_manager = Mock()
        self.mock_logger = Mock()

        # Instantiate the class we are testing
        self.executor = PipelineStepExecutor(
            alima_manager=self.mock_alima_manager,
            cache_manager=self.mock_cache_manager,
            logger=self.mock_logger
        )

    def test_execute_initial_keyword_extraction(self):
        """Test the first step of the pipeline: initial keyword extraction."""
        # 1. Arrange: Define inputs and configure mock responses
        abstract_text = "This is a test abstract about machine learning."
        model = "test-model"
        provider = "test-provider"
        task = "initialisation"

        # Configure the mock AlimaManager to return a predictable result
        mock_analysis_result = AnalysisResult(
            full_text="<keywords>Machine Learning, AI</keywords><class>004</class>",
            matched_keywords={},
            gnd_systematic=""
        )
        mock_prompt_config = PromptConfigData(prompt="Test prompt", system="System prompt")
        mock_task_state = TaskState(
            abstract_data=AbstractData(abstract=abstract_text, keywords=""),
            analysis_result=mock_analysis_result,
            prompt_config=mock_prompt_config,
            status="completed",
            task_name=task,
            model_used=model,
            provider_used=provider
        )
        self.mock_alima_manager.analyze_abstract.return_value = mock_task_state

        # 2. Act: Call the method we are testing
        keywords, gnd_classes, llm_analysis, llm_title = self.executor.execute_initial_keyword_extraction(
            abstract_text=abstract_text,
            model=model,
            provider=provider,
            task=task
        )

        # 3. Assert: Check if the results are correct
        # Check if the mock was called correctly
        self.mock_alima_manager.analyze_abstract.assert_called_once()
        call_args, call_kwargs = self.mock_alima_manager.analyze_abstract.call_args
        self.assertEqual(call_kwargs['task'], task)
        self.assertEqual(call_kwargs['model'], model)
        self.assertEqual(call_kwargs['provider'], provider)
        self.assertEqual(call_args[0].abstract, abstract_text)

        # Check the processed output of our method
        self.assertEqual(keywords, ["Machine Learning", "AI"])
        self.assertEqual(gnd_classes, ["004"])
        self.assertIsNotNone(llm_analysis)
        self.assertEqual(llm_analysis.model_used, model)
        self.assertEqual(llm_analysis.extracted_gnd_keywords, ["Machine Learning", "AI"])

    @patch('src.core.search_cli.SearchCLI') # Patch the SearchCLI class
    def test_execute_gnd_search(self, MockSearchCLI):
        """Test the GND search step of the pipeline."""
        # 1. Arrange: Define inputs and configure mock responses
        keywords = ["Machine Learning", "Artificial Intelligence"]
        suggesters = ["lobid", "swb"]

        # Configure the mock SearchCLI instance
        mock_search_cli_instance = Mock()
        mock_search_cli_instance.search.return_value = {
            "Machine Learning": {
                "Maschinelles Lernen": {"count": 100, "gndid": {"4037877-9"}},
            },
            "Artificial Intelligence": {
                "Künstliche Intelligenz": {"count": 120, "gndid": {"4033597-0"}},
            },
        }
        MockSearchCLI.return_value = mock_search_cli_instance # Configure the mock class to return our mock instance

        # 2. Act: Call the method we are testing
        search_results = self.executor.execute_gnd_search(
            keywords=keywords,
            suggesters=suggesters
        )

        # 3. Assert: Check if the results are correct
        # Check if SearchCLI was instantiated correctly
        MockSearchCLI.assert_called_once_with(
            self.mock_cache_manager, # Should pass the cache manager
            catalog_token="", # Default values
            catalog_search_url="",
            catalog_details_url=""
        )
        # Check if the search method was called correctly
        mock_search_cli_instance.search.assert_called_once_with(
            search_terms=keywords,
            suggester_types=[Mock(spec=str), Mock(spec=str)] # We can't easily assert SuggesterType enum directly here
        )

        # Check the processed output of our method
        self.assertIn("Machine Learning", search_results)
        self.assertIn("Artificial Intelligence", search_results)
        self.assertIn("Maschinelles Lernen", search_results["Machine Learning"])
        self.assertIn("Künstliche Intelligenz", search_results["Artificial Intelligence"])
        self.assertEqual(search_results["Machine Learning"]["Maschinelles Lernen"]["gndid"], {"4037877-9"})
        self.assertEqual(search_results["Artificial Intelligence"]["Künstliche Intelligenz"]["gndid"], {"4033597-0"})

    def test_execute_final_keyword_analysis(self):
        """Test the final keyword analysis step of the pipeline."""
        # 1. Arrange: Define inputs and configure mock responses
        original_abstract = "This is an abstract about environmental pollution."
        search_results = {
            "environmental pollution": {
                "Umweltverschmutzung": {"count": 50, "gndid": {"4061694-5"}},
            },
            "cadmium contamination": {
                "Cadmium": {"count": 30, "gndid": {"4009274-4"}},
                "Kontamination": {"count": 20, "gndid": {"4032184-0"}},
            },
        }
        model = "final-model"
        provider = "final-provider"
        task = "keywords"

        # Mock AlimaManager response for final analysis
        mock_analysis_result = AnalysisResult(
            full_text="<keywords>Umweltverschmutzung (GND-ID: 4061694-5), Cadmium (GND-ID: 4009274-4)</keywords><class>21.4</class>",
            matched_keywords={},
            gnd_systematic=""
        )
        mock_prompt_config = Mock(spec=object, prompt="Final prompt", system="Final system")
        mock_task_state = TaskState(
            abstract_data=Mock(spec=object),
            analysis_result=mock_analysis_result,
            prompt_config=mock_prompt_config,
            status="completed",
            task_name=task,
            model_used=model,
            provider_used=provider
        )
        self.mock_alima_manager.analyze_abstract.return_value = mock_task_state

        # Mock cache_manager for GND title and synonyms
        self.mock_cache_manager.get_gnd_title_by_id.side_effect = lambda gnd_id:
            {
                "4061694-5": "Umweltverschmutzung",
                "4009274-4": "Cadmium",
                "4032184-0": "Kontamination",
            }.get(gnd_id, "")
        self.mock_cache_manager.get_gnd_synonyms_by_id.return_value = []

        # 2. Act: Call the method we are testing
        final_keywords, gnd_classes, llm_analysis = self.executor.execute_final_keyword_analysis(
            original_abstract=original_abstract,
            search_results=search_results,
            model=model,
            provider=provider,
            task=task
        )

        # 3. Assert: Check if the results are correct
        self.mock_alima_manager.analyze_abstract.assert_called_once()
        call_args, call_kwargs = self.mock_alima_manager.analyze_abstract.call_args
        self.assertEqual(call_kwargs['task'], task)
        self.assertEqual(call_kwargs['model'], model)
        self.assertEqual(call_kwargs['provider'], provider)
        self.assertEqual(call_args[0].abstract, original_abstract)

        expected_keywords = [
            "Umweltverschmutzung (GND-ID: 4061694-5)",
            "Cadmium (GND-ID: 4009274-4)"
        ]
        self.assertCountEqual(final_keywords, expected_keywords)
        self.assertEqual(gnd_classes, ["21.4"])
        self.assertIsNotNone(llm_analysis)
        self.assertEqual(llm_analysis.model_used, model)
        self.assertCountEqual(llm_analysis.extracted_gnd_keywords, expected_keywords)

    @patch('src.utils.clients.biblio_client.BiblioClient')
    def test_execute_dk_search(self, MockBiblioClient):
        """Test the DK search step of the pipeline."""
        # 1. Arrange: Define inputs and configure mock responses
        keywords = ["Umweltverschmutzung (GND-ID: 4061694-5)"]
        catalog_token = "test_token"
        catalog_search_url = "test_search_url"
        catalog_details_url = "test_details_url"

        # Configure the mock BiblioClient instance
        mock_biblio_client_instance = Mock()
        mock_biblio_client_instance.extract_dk_classifications_for_keywords.return_value = [
            {"dk": "614.7", "classification_type": "DK", "keyword": "Umweltverschmutzung"}
        ]
        MockBiblioClient.return_value = mock_biblio_client_instance

        # 2. Act: Call the method we are testing
        dk_search_results = self.executor.execute_dk_search(
            keywords=keywords,
            catalog_token=catalog_token,
            catalog_search_url=catalog_search_url,
            catalog_details_url=catalog_details_url,
        )

        # 3. Assert: Check if the results are correct
        MockBiblioClient.assert_called_once_with(
            token=catalog_token,
            debug=False # Default debug value
        )
        mock_biblio_client_instance.extract_dk_classifications_for_keywords.assert_called_once_with(
            keywords=["Umweltverschmutzung"], # Should clean the keyword
            max_results=50 # Default max_results
        )
        self.assertEqual(len(dk_search_results), 1)
        self.assertEqual(dk_search_results[0]["dk"], "614.7")

    def test_execute_dk_classification(self):
        """Test the DK classification step of the pipeline."""
        # 1. Arrange: Define inputs and configure mock responses
        original_abstract = "Abstract about environmental science."
        dk_search_results = [
            {"dk": "614.7", "classification_type": "DK", "count": 5, "titles": ["Title 1", "Title 2"], "keywords": ["Umwelt"]},
            {"dk": "QZ 123", "classification_type": "RVK", "count": 3, "titles": ["Title 3"], "keywords": ["Biologie"]},
        ]
        model = "dk-model"
        provider = "dk-provider"

        # Configure the mock AlimaManager to return a predictable result
        mock_analysis_result = AnalysisResult(
            full_text="<dk_class>DK 614.7, RVK QZ 123</dk_class>",
            matched_keywords={},
            gnd_systematic=""
        )
        mock_task_state = TaskState(
            abstract_data=Mock(spec=object),
            analysis_result=mock_analysis_result,
            prompt_config=Mock(spec=object),
            status="completed",
            task_name="dk_class",
            model_used=model,
            provider_used=provider
        )
        self.mock_alima_manager.analyze_abstract.return_value = mock_task_state

        # 2. Act: Call the method we are testing
        dk_classifications = self.executor.execute_dk_classification(
            original_abstract=original_abstract,
            dk_search_results=dk_search_results,
            model=model,
            provider=provider
        )

        # 3. Assert: Check if the results are correct
        self.mock_alima_manager.analyze_abstract.assert_called_once()
        call_args, call_kwargs = self.mock_alima_manager.analyze_abstract.call_args
        self.assertEqual(call_kwargs['task'], "dk_class")
        self.assertEqual(call_kwargs['model'], model)
        self.assertEqual(call_kwargs['provider'], provider)
        self.assertIn("614.7", call_args[0].keywords) # Check if formatted catalog data is in prompt

        self.assertEqual(dk_classifications, ["DK 614.7", "RVK QZ 123"])

    def test_create_complete_analysis_state(self):
        """Test the creation of the complete KeywordAnalysisState."""
        # 1. Arrange: Define all necessary input data
        original_abstract = "Test abstract for full state creation."
        initial_keywords = ["initial", "keywords"]
        initial_gnd_classes = ["001"]
        
        # Mock SearchResult objects
        mock_search_results_dict = {
            "term1": {"kw1": {"gndid": {"1"}}},
            "term2": {"kw2": {"gndid": {"2"}}},
        }
        
        # Mock LlmKeywordAnalysis objects
        mock_initial_llm_analysis = Mock(spec=LlmKeywordAnalysis)
        mock_final_llm_analysis = Mock(spec=LlmKeywordAnalysis)

        suggesters_used = ["lobid"]

        # 2. Act: Call the method we are testing
        analysis_state = self.executor.create_complete_analysis_state(
            original_abstract=original_abstract,
            initial_keywords=initial_keywords,
            initial_gnd_classes=initial_gnd_classes,
            search_results=mock_search_results_dict,
            initial_llm_analysis=mock_initial_llm_analysis,
            final_llm_analysis=mock_final_llm_analysis,
            suggesters_used=suggesters_used,
        )

        # 3. Assert: Check if the KeywordAnalysisState is correctly populated
        self.assertEqual(analysis_state.original_abstract, original_abstract)
        self.assertEqual(analysis_state.initial_keywords, initial_keywords)
        self.assertEqual(analysis_state.initial_gnd_classes, initial_gnd_classes)
        self.assertEqual(analysis_state.search_suggesters_used, suggesters_used)
        self.assertEqual(analysis_state.initial_llm_call_details, mock_initial_llm_analysis)
        self.assertEqual(analysis_state.final_llm_analysis, mock_final_llm_analysis)

        # Check search_results conversion
        self.assertEqual(len(analysis_state.search_results), 2)
        self.assertIsInstance(analysis_state.search_results[0], SearchResult)
        self.assertEqual(analysis_state.search_results[0].search_term, "term1")
        self.assertEqual(analysis_state.search_results[1].search_term, "term2")

if __name__ == '__main__':
    unittest.main()
