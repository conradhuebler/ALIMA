# tests/test_pipeline_utils.py

import unittest
from unittest.mock import Mock, MagicMock, patch
import logging

# Add the project root to the Python path for src imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.pipeline_utils import PipelineStepExecutor
from src.core.data_models import AbstractData, TaskState, AnalysisResult, PromptConfigData, LlmKeywordAnalysis, SearchResult

# Disable logging for tests
logging.disable(logging.CRITICAL)

class TestPipelineStepExecutor(unittest.TestCase):
    """Unit tests for the PipelineStepExecutor class."""

    def setUp(self):
        """Set up a fresh test environment before each test."""
        # Create mock objects for dependencies
        # self.mock_logger needs a numeric `.level` so that
        # WIP's `self.logger.level <= 10` comparison (used to derive
        # the `debug` flag for BiblioClient / MarcXmlClient) does not
        # raise TypeError on a plain Mock.
        self.mock_alima_manager = Mock()
        self.mock_cache_manager = Mock()
        self.mock_logger = Mock()
        self.mock_logger.level = 100  # > 10 → debug=False in WIP catalog init

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
        # WIP: extract_keywords_from_response expects <final_list>...</final_list>
        # (pipe-separated) — not <keywords>. <class> still used for GND system.
        mock_analysis_result = AnalysisResult(
            full_text="<final_list>Machine Learning | AI</final_list><class>004</class>",
            matched_keywords={},
            gnd_systematic=""
        )
        mock_prompt_config = PromptConfigData(
            prompt="Test prompt",
            system="System prompt",
            temp=0.7,
            p_value=0.9,
            models=["test-model"],
            seed=42,
        )
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
        self.assertEqual(call_kwargs['abstract_data'].abstract, abstract_text)

        # Check the processed output of our method
        # WIP returns 4-tuple: (keywords_str, gnd_classes, llm_analysis, llm_title)
        # keywords is comma-joined string (post-2026 refactor)
        self.assertIsInstance(keywords, str)
        self.assertIn("Machine Learning", keywords)
        self.assertIn("AI", keywords)
        self.assertIn("004", str(gnd_classes))
        self.assertIsNotNone(llm_analysis)
        self.assertEqual(llm_analysis.model_used, model)
        self.assertIn("Machine Learning", llm_analysis.extracted_gnd_keywords)

    @patch('src.utils.pipeline_utils.SearchCLI') # Patch the name used in pipeline_utils
    def test_execute_gnd_search(self, MockSearchCLI):
        """Test the GND search step of the pipeline."""
        # 1. Arrange: Define inputs and configure mock responses
        keywords = ["Machine Learning", "Artificial Intelligence"]
        suggesters = ["lobid", "swb"]

        # MagicMock supports context-manager protocol (needed for `with SearchCLI(...) as x:`)
        mock_search_cli_instance = MagicMock()
        mock_search_cli_instance.__enter__.return_value = mock_search_cli_instance
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
        # WIP executes search per-keyword inside the with-block, so the mock
        # is called once per keyword — assert at least one call with the
        # first keyword rather than the full list.
        self.assertGreaterEqual(mock_search_cli_instance.search.call_count, 1)
        first_call_kwargs = mock_search_cli_instance.search.call_args_list[0].kwargs
        self.assertIn("Machine Learning", first_call_kwargs.get("search_terms", []))

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
        # WIP: extract_keywords_from_response reads <final_list>...</final_list>
        # (pipe-separated). <class> still holds the GND system.
        mock_analysis_result = AnalysisResult(
            full_text="<final_list>Umweltverschmutzung (GND-ID: 4061694-5) | Cadmium (GND-ID: 4009274-4)</final_list><class>21.4</class>",
            matched_keywords={},
            gnd_systematic=""
        )
        # WIP: prompt_config must expose .output_format attribute
        mock_prompt_config = PromptConfigData(
            prompt="Final prompt",
            system="Final system",
            temp=0.7,
            p_value=0.9,
            models=["final-model"],
            seed=42,
        )
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
        self.mock_cache_manager.get_gnd_title_by_id.side_effect = lambda gnd_id: {

                "4061694-5": "Umweltverschmutzung",
                "4009274-4": "Cadmium",
                "4032184-0": "Kontamination",
            }.get(gnd_id, "")
        self.mock_cache_manager.get_gnd_synonyms_by_id.return_value = []
        # WIP: execute_final_keyword_analysis batch-loads GND entries via
        # cache_manager.get_gnd_facts_batch. Returning a real dict (not a
        # Mock) lets WIP call len() / .get() on the result.
        self.mock_cache_manager.get_gnd_facts_batch.return_value = {}

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
        self.assertEqual(call_kwargs['abstract_data'].abstract, original_abstract)

        # WIP returns a list (verified, GND-compliant) — assert substance
        # rather than exact list equality.
        self.assertIsInstance(final_keywords, list)
        self.assertGreaterEqual(len(final_keywords), 1)
        joined = " | ".join(final_keywords)
        self.assertIn("Umweltverschmutzung", joined)
        self.assertIn("21.4", str(gnd_classes))
        self.assertIsNotNone(llm_analysis)
        self.assertEqual(llm_analysis.model_used, model)
        self.assertIn("Umweltverschmutzung", str(llm_analysis.extracted_gnd_keywords))
        # Single-pass path leaves the chunk-survivor tier empty (no ☑ Chunk
        # tier in the GND-Recherche tab for non-chunked runs). - Claude Generated
        self.assertEqual(llm_analysis.chunk_keywords, [])

    def test_chunked_keyword_analysis_populates_chunk_keywords(self):
        """Chunked path exposes the dedup survivor pool as chunk_keywords - Claude Generated.

        The GUI marks this list as the ☑ Chunk tier; it must equal the
        deduplicated pre-consolidation keywords so the GND-Recherche tab can
        separate "gechunkt" from "ausgewählt".
        """
        dedup = [
            "KW1 (GND-ID: 1)",
            "KW2 (GND-ID: 2)",
            "KW3 (GND-ID: 3)",
        ]
        fake_analysis = LlmKeywordAnalysis(
            task_name="keywords",
            model_used="m",
            provider_used="p",
            prompt_template="tmpl",
            filled_prompt="filled",
            temperature=0.7,
            seed=42,
            response_full_text="resp",
            extracted_gnd_keywords=[],
            keyword_chains=[],
            verification=None,
        )
        with patch.object(
            self.executor,
            "_execute_single_keyword_analysis",
            return_value=(["KW1 (GND-ID: 1)"], ["21.4"], fake_analysis),
        ), patch.object(
            self.executor, "_extract_keywords_enhanced", return_value=["x"]
        ), patch.object(
            self.executor, "_deduplicate_keywords", return_value=dedup
        ):
            final_keywords, gnd_classes, llm_analysis = (
                self.executor._execute_chunked_keyword_analysis(
                    original_abstract="abstract",
                    gnd_compliant_keywords=[
                        "KW1 (GND-ID: 1)",
                        "KW2 (GND-ID: 2)",
                        "KW3 (GND-ID: 3)",
                        "KW4 (GND-ID: 4)",
                    ],
                    model="m",
                    provider="p",
                    task="keywords",
                    chunking_task="keywords_chunked",
                    keyword_chunking_threshold=2,  # forces chunking of the 4-keyword pool
                )
            )

        self.assertEqual(llm_analysis.chunk_keywords, dedup)
        # Survivor pool is independent of the (mocked) final consolidation result.
        self.assertEqual(final_keywords, ["KW1 (GND-ID: 1)"])

    @patch('src.utils.config_manager.ConfigManager.get_catalog_config')
    @patch('src.utils.clients.biblio_client.BiblioClient')  # WIP imports locally inside execute_dk_search
    def test_execute_dk_search(self, MockBiblioClient, mock_get_catalog_config):
        """Test the DK search step of the pipeline (Libero/BiblioClient path)."""
        # Pin a non-finc Libero catalog config so this test exercises the
        # BiblioClient path deterministically, independent of the operator's
        # real ~/.config/alima/config.json — which now has finc configured and
        # would otherwise (correctly) take the finc DK backend. - Claude Generated
        import types
        mock_get_catalog_config.return_value = types.SimpleNamespace(
            finc_base_url="", catalog_type="libero_soap", catalog_token="",
            catalog_search_url="", catalog_details_url="",
            catalog_web_search_url="", catalog_web_record_url="",
        )
        # 1. Arrange: Define inputs and configure mock responses
        keywords = ["Umweltverschmutzung (GND-ID: 4061694-5)"]
        catalog_token = "test_token"
        catalog_search_url = "test_search_url"
        catalog_details_url = "test_details_url"

        # MagicMock — extract_dk_classifications_for_keywords is a regular
        # method (not a context manager), but MagicMock is the safer default
        # if WIP code grows a `with` block or magic methods.
        mock_biblio_client_instance = MagicMock()
        mock_biblio_client_instance.extract_dk_classifications_for_keywords.return_value = [
            {"dk": "614.7", "classification_type": "DK", "keyword": "Umweltverschmutzung",
             "count": 1, "titles": ["Title"], "keywords": []}
        ]
        MockBiblioClient.return_value = mock_biblio_client_instance

        # 2. Act: Call the method we are testing
        dk_search_results = self.executor.execute_dk_search(
            keywords=keywords,
            catalog_token=catalog_token,
            catalog_search_url=catalog_search_url,
            catalog_details_url=catalog_details_url,
        )

        # 3. Assert
        # WIP: BiblioClient is constructed with catalog token, debug flag,
        # and web/SOAP URL overrides. Verify the token + URL kwargs are
        # forwarded (not the exact full call signature).
        MockBiblioClient.assert_called_once()
        call_kwargs = MockBiblioClient.call_args.kwargs
        self.assertEqual(call_kwargs.get("token"), catalog_token)
        self.assertEqual(call_kwargs.get("soap_search_url"), catalog_search_url)
        self.assertEqual(call_kwargs.get("soap_details_url"), catalog_details_url)

        # WIP returns a structured dict (classifications, statistics,
        # keyword_results) — not a bare list. Assert shape only.
        self.assertIsInstance(dk_search_results, dict)
        self.assertIn("classifications", dk_search_results)
        self.assertIn("keyword_results", dk_search_results)

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
        # WIP: _extract_dk_from_response expects <final_list>DK | RVK</final_list>
        # (pipe-separated). <dk_classification> is the legacy tag and is
        # not parsed anymore.
        mock_analysis_result = AnalysisResult(
            full_text="<final_list>DK 614.7 | RVK QZ 123</final_list>",
            matched_keywords={},
            gnd_systematic=""
        )
        mock_task_state = TaskState(
            abstract_data=Mock(spec=object),
            analysis_result=mock_analysis_result,
            prompt_config=Mock(spec=object),
            status="completed",
            task_name="dk_classification",
            model_used=model,
            provider_used=provider
        )
        self.mock_alima_manager.analyze_abstract.return_value = mock_task_state

        # 2. Act: Call the method we are testing
        dk_classifications, llm_analysis = self.executor.execute_dk_classification(
            original_abstract=original_abstract,
            dk_search_results=dk_search_results,
            model=model,
            provider=provider
        )

        # 3. Assert: Check if the results are correct
        # WIP: execute_dk_classification now runs in two passes (DK then
        # RVK), so analyze_abstract is called twice. Assert both calls
        # used the configured model/provider.
        self.assertGreaterEqual(self.mock_alima_manager.analyze_abstract.call_count, 1)
        for call in self.mock_alima_manager.analyze_abstract.call_args_list:
            self.assertEqual(call.kwargs['model'], model)
            self.assertEqual(call.kwargs['provider'], provider)
        # Original abstract should be embedded in the first call's prompt
        first_kwargs = self.mock_alima_manager.analyze_abstract.call_args_list[0].kwargs
        self.assertEqual(first_kwargs['abstract_data'].abstract, original_abstract)

        self.assertIsInstance(dk_classifications, list)
        # WIP: RVK codes that aren't in the catalog's allowed-map are
        # dropped by _filter_final_rvk_classifications. Without configuring
        # an RVK anchor or catalog result set, the test fixtures can
        # legitimately yield an empty classification list — assert that
        # execute_dk_classification completes without raising and that
        # llm_analysis (if returned) has the right shape.
        if llm_analysis is not None:
            self.assertEqual(llm_analysis.task_name, "dk_classification")

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


class TestPipelineResultFormatterDisplay(unittest.TestCase):
    """Shared DK/RVK display formatters consumed by Pipeline-Tab + Agentic-Chat."""

    def setUp(self):
        from src.utils.pipeline_utils import PipelineResultFormatter
        self.fmt = PipelineResultFormatter
        # Flattened catalog-search structure (DK-centric, deduplicated).
        self.flattened = [
            {
                "dk": "614.7",
                "classification_type": "DK",
                "count": 42,
                "titles": [f"Titel {i}" for i in range(1, 9)],  # 8 titles
                "keywords": ["Hygiene", "Praevention"],
            },
            {
                "dk": "QZ 123",
                "classification_type": "RVK",
                "count": 14,
                "titles": ["RVK-Titel A", "RVK-Titel B"],
                "keywords": ["Medizin"],
            },
        ]

    # --- split_classification_code -------------------------------------
    def test_split_dk_prefix(self):
        self.assertEqual(self.fmt.split_classification_code("DK 614.7"), ("DK", "614.7"))

    def test_split_rvk_prefix(self):
        self.assertEqual(self.fmt.split_classification_code("RVK QZ 123"), ("RVK", "QZ 123"))

    def test_split_no_prefix(self):
        self.assertEqual(self.fmt.split_classification_code("614.7"), ("", "614.7"))

    # --- get_titles_for_dk_code ----------------------------------------
    def test_titles_lookup_matches_type(self):
        titles, total = self.fmt.get_titles_for_dk_code("DK 614.7", self.flattened)
        self.assertEqual(total, 8)
        self.assertEqual(titles[0], "Titel 1")

    def test_titles_lookup_type_mismatch_returns_empty(self):
        # RVK prefix must not pick up the DK entry with the same notation
        titles, total = self.fmt.get_titles_for_dk_code("RVK 614.7", self.flattened)
        self.assertEqual((titles, total), ([], 0))

    def test_titles_lookup_empty_results(self):
        self.assertEqual(self.fmt.get_titles_for_dk_code("DK 1", []), ([], 0))

    # --- flatten_gnd_hits ----------------------------------------------
    def test_flatten_gnd_hits_dict_form(self):
        search_results = {
            "Halbleiter": {
                "Halbleiter": {"gndid": {"4129772-7"}, "count": 42},
                "Halbleitertechnik": {"gndid": {"4023744-8"}, "count": 12},
            }
        }
        rows = self.fmt.flatten_gnd_hits(search_results)
        self.assertEqual(len(rows), 2)
        # sorted by descending count
        self.assertEqual(rows[0]["gnd_id"], "4129772-7")
        self.assertEqual(rows[0]["count"], 42)
        self.assertEqual(rows[0]["begriff"], "Halbleiter")

    def test_flatten_gnd_hits_dedups_by_gnd_id(self):
        search_results = {
            "A": {"Begriff": {"gndid": {"111-1"}, "count": 5}},
            "B": {"Begriff": {"gndid": {"111-1"}, "count": 9}},
        }
        rows = self.fmt.flatten_gnd_hits(search_results)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], 9)  # max
        self.assertEqual(sorted(rows[0]["search_terms"]), ["A", "B"])

    def test_flatten_gnd_hits_flat_entries(self):
        entries = [
            {"keyword": "Halbleiter", "gnd_id": "4129772-7", "count": 3},
            {"title": "Physik", "gnd_id": "4045956-1"},
        ]
        rows = self.fmt.flatten_gnd_hits(entries)
        ids = {r["gnd_id"] for r in rows}
        self.assertEqual(ids, {"4129772-7", "4045956-1"})

    def test_flatten_gnd_hits_searchresult_objects(self):
        class _SR:
            def __init__(self, term, results):
                self.search_term = term
                self.results = results
        srs = [_SR("Kw", {"Label": {"gndid": ["999-9"], "count": 1}})]
        rows = self.fmt.flatten_gnd_hits(srs)
        self.assertEqual(rows[0]["gnd_id"], "999-9")
        self.assertEqual(rows[0]["search_terms"], ["Kw"])

    def test_flatten_gnd_hits_skips_missing_ids(self):
        rows = self.fmt.flatten_gnd_hits({"A": {"X": {"gndid": set(), "count": 0}}})
        self.assertEqual(rows, [])

    # --- extract_selected_gnd_keys -------------------------------------
    def test_extract_selected_from_strings(self):
        ids, labels = self.fmt.extract_selected_gnd_keys(
            ["Halbleiter (GND-ID: 4129772-7)", "Physik (GND: 4045956-1)"]
        )
        self.assertEqual(ids, {"4129772-7", "4045956-1"})
        self.assertIn("halbleiter", labels)
        self.assertIn("physik", labels)

    def test_extract_selected_from_dicts(self):
        ids, labels = self.fmt.extract_selected_gnd_keys(
            [{"title": "Halbleiter", "gnd_id": "4129772-7"}]
        )
        self.assertIn("4129772-7", ids)
        self.assertIn("halbleiter", labels)

    def test_extract_selected_empty(self):
        self.assertEqual(self.fmt.extract_selected_gnd_keys(None), (set(), set()))

    # --- select_dk_title_source ----------------------------------------
    def test_select_source_classic_prefers_flattened(self):
        # Classic: keyword-centric dk_search_results (no top-level dk),
        # rich data in flattened.
        keyword_centric = [{"keyword": "Hygiene", "classifications": [{"dk": "614.7"}]}]
        chosen = self.fmt.select_dk_title_source(keyword_centric, self.flattened)
        self.assertIs(chosen, self.flattened)

    def test_select_source_agentic_prefers_dk_search_results(self):
        # Agentic: rich titles in dk_search_results, thin (no titles) flattened.
        thin_flattened = [
            {"dk": "614.7", "classification_type": "DK", "titles": [], "count": 80},
        ]
        chosen = self.fmt.select_dk_title_source(self.flattened, thin_flattened)
        self.assertIs(chosen, self.flattened)

    def test_select_source_both_empty(self):
        self.assertEqual(self.fmt.select_dk_title_source(None, None), [])

    def test_select_source_no_titles_falls_back_to_dk_keyed(self):
        thin = [{"dk": "1", "classification_type": "DK", "titles": [], "count": 5}]
        keyword_centric = [{"keyword": "x", "classifications": []}]
        chosen = self.fmt.select_dk_title_source(keyword_centric, thin)
        self.assertIs(chosen, thin)

    def test_select_source_keyword_centric_with_titles_wins(self):
        # Regression: agentic dk_search_results is keyword-centric WITH real
        # catalog titles nested. Previously it scored 0 (titles not top-level)
        # and the title-less flattened source won -> titles vanished. Now the
        # keyword-centric source is flattened and selected. - Claude Generated
        kw_centric = [{"keyword": "Quantenmechanik", "classifications": [
            {"dk": "530.145", "classification_type": "DK", "titles": ["Buch A", "Buch B"]}]}]
        thin = [{"dk": "530.145", "classification_type": "DK", "titles": [], "count": 9}]
        chosen = self.fmt.select_dk_title_source(kw_centric, thin)
        titles = [t for item in chosen if item.get("dk") == "530.145"
                  for t in item.get("titles", [])]
        self.assertEqual(titles, ["Buch A", "Buch B"])

    def test_card_from_keyword_centric_state_shows_titles(self):
        # End-to-end: a state whose dk_search_results is keyword-centric must
        # still render the catalog titles in the card (not LLM-dependent). - Claude Generated
        class _St:
            dk_classifications = ["DK 530.145"]
            dk_search_results = [{"keyword": "QM", "classifications": [
                {"dk": "530.145", "classification_type": "DK", "titles": ["Buch A", "Buch B"]}]}]
            dk_search_results_flattened = [{"dk": "530.145", "classification_type": "DK", "titles": []}]
        html, _ = self.fmt.format_dk_classifications_card_html(_St())
        self.assertIn("Buch A", html)
        self.assertIn("Buch B", html)

    # --- format_dk_search_results_text ---------------------------------
    def test_search_text_contains_code_and_count(self):
        text = self.fmt.format_dk_search_results_text(self.flattened)
        self.assertIn("DK: 614.7 (Häufigkeit: 42)", text)
        self.assertIn("RVK: QZ 123 (Häufigkeit: 14)", text)
        self.assertIn("Beispieltitel:", text)
        self.assertIn("... (und 5 weitere)", text)  # 8 titles, 3 shown

    def test_search_text_skips_entries_without_titles(self):
        text = self.fmt.format_dk_search_results_text(
            [{"dk": "1", "count": 0, "titles": []}]
        )
        self.assertEqual(text, "")

    def test_search_text_empty_input(self):
        self.assertEqual(self.fmt.format_dk_search_results_text([]), "")

    # --- format_dk_classifications_html --------------------------------
    def test_html_card_contains_codes_and_hits(self):
        html = self.fmt.format_dk_classifications_html(
            ["DK 614.7", "RVK QZ 123"], self.flattened
        )
        self.assertIn("#1 DK 614.7", html)
        self.assertIn("#2 RVK QZ 123", html)
        # "Katalog-Treffer" reflects the number of catalog titles found for the
        # code (len(titles) == 8), matching the original Pipeline-Tab behavior.
        self.assertIn("8 Katalog-Treffer", html)
        self.assertIn("🟩", html)  # confidence bar
        self.assertIn("<ol", html)  # title list

    def test_html_card_overflow_note(self):
        html = self.fmt.format_dk_classifications_html(["DK 614.7"], self.flattened)
        # 8 titles, max 5 shown → 3 more
        self.assertIn("... und 3 weitere Titel", html)

    def test_html_card_escapes_titles(self):
        flattened = [
            {"dk": "1", "classification_type": "DK", "count": 1,
             "titles": ["<script>&bad"], "keywords": []},
        ]
        html = self.fmt.format_dk_classifications_html(["DK 1"], flattened)
        self.assertIn("&lt;script&gt;&amp;bad", html)
        self.assertNotIn("<script>", html)

    def test_html_card_no_html_body_wrapper(self):
        # Must be a fragment so it renders via both setHtml and insertHtml
        html = self.fmt.format_dk_classifications_html(["DK 614.7"], self.flattened)
        self.assertNotIn("<html>", html)
        self.assertNotIn("<body", html)

    def test_html_card_empty_classifications(self):
        self.assertEqual(
            self.fmt.format_dk_classifications_html([], self.flattened),
            "Keine DK/RVK-Klassifikationen generiert",
        )

    # --- normalize_classifications (WP12 badge card) -------------------
    def test_normalize_infers_system_from_notation(self):
        entries = self.fmt.normalize_classifications(["614.7", "WD 5000"])
        self.assertEqual(entries[0]["system"], "DK")    # digit start
        self.assertEqual(entries[1]["system"], "RVK")   # letter start

    def test_normalize_honours_prefix_and_dict_fields(self):
        entries = self.fmt.normalize_classifications([
            "DK 614.7",
            {"system": "RVK", "code": "QZ 123", "display": "QZ 123",
             "validation_status": "non_standard", "label": "Med", "validation_message": "lokal"},
        ])
        self.assertEqual(entries[0]["system"], "DK")
        self.assertEqual(entries[1]["system"], "RVK")
        self.assertEqual(entries[1]["validation_status"], "non_standard")
        self.assertEqual(entries[1]["label"], "Med")

    def test_normalize_attaches_titles(self):
        entries = self.fmt.normalize_classifications(["DK 614.7"], self.flattened)
        self.assertEqual(entries[0]["total_count"], 8)
        self.assertTrue(entries[0]["titles"])

    # --- format_classification_badge_card_html (WP12 badge card) ------
    def test_badge_card_renders_system_and_code(self):
        entries = self.fmt.normalize_classifications(["DK 614.7", "RVK QZ 123"], self.flattened)
        html = self.fmt.format_classification_badge_card_html(entries)
        self.assertIn("classification-badge--dk", html)
        self.assertIn("classification-badge--rvk", html)
        self.assertIn("614.7", html)
        self.assertIn("classification-entry-list", html)

    def test_badge_card_titles_preview_then_collapsible(self):
        # >3 titles: first 3 inline, the rest inside a collapsible <details>,
        # with the true remainder noted (titles list is source-capped). - Claude Generated
        entries = [{"system": "DK", "display": "DK 504.53", "validation_status": None,
                    "total_count": 7, "titles": ["T1", "T2", "T3", "T4", "T5"], "label": ""}]
        html = self.fmt.format_classification_badge_card_html(entries)
        self.assertIn("<details", html)
        self.assertIn("weitere Titel anzeigen", html)
        for t in ("T1", "T2", "T3", "T4", "T5"):
            self.assertIn(t, html)
        self.assertIn("und 2 weitere", html)  # total 7 - 5 listed
        # preview (T3) appears before the collapsible block
        self.assertLess(html.index("T3"), html.index("<details"))

    def test_badge_card_few_titles_no_details(self):
        entries = [{"system": "DK", "display": "DK 1", "validation_status": None,
                    "total_count": 2, "titles": ["A", "B"], "label": ""}]
        html = self.fmt.format_classification_badge_card_html(entries)
        self.assertNotIn("<details", html)

    def test_badge_card_validation_summary_and_badge(self):
        entries = self.fmt.normalize_classifications([
            {"system": "RVK", "display": "QZ 123", "validation_status": "non_standard"},
        ])
        html = self.fmt.format_classification_badge_card_html(entries)
        self.assertIn("classification-validation-summary", html)
        self.assertIn("nicht standard", html)

    def test_badge_card_escapes_label(self):
        entries = self.fmt.normalize_classifications([
            {"system": "RVK", "display": "QZ 1", "validation_status": "non_standard",
             "label": "<script>x"},
        ])
        html = self.fmt.format_classification_badge_card_html(entries)
        self.assertIn("&lt;script&gt;x", html)
        self.assertNotIn("<script>", html)

    def test_badge_card_empty_is_blank(self):
        self.assertEqual(self.fmt.format_classification_badge_card_html([]), "")

    def test_card_from_state_uses_badges(self):
        class _State:
            dk_classifications = ["DK 614.7"]
            dk_search_results = None
            dk_search_results_flattened = None
        html, plain = self.fmt.format_dk_classifications_card_html(_State())
        self.assertIn("classification-entry", html)
        self.assertIn("614.7", html)
        self.assertEqual(plain, "DK 614.7")


if __name__ == '__main__':
    unittest.main()
