"""
Pipeline Utils - Shared logic for CLI and GUI pipeline implementations
Claude Generated - Abstracts common pipeline operations and utilities
"""

import json
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

from ..core.data_models import (
    AbstractData,
    TaskState,
    AnalysisResult,
    KeywordAnalysisState,
    LlmKeywordAnalysis,
    SearchResult,
)
from ..core.search_cli import SearchCLI
from ..core.cache_manager import CacheManager
from ..core.suggesters.meta_suggester import SuggesterType
from ..core.processing_utils import (
    extract_keywords_from_response,
    extract_gnd_system_from_response,
)


class PipelineStepExecutor:
    """Shared pipeline step execution logic - Claude Generated"""

    def __init__(
        self,
        alima_manager,
        cache_manager: CacheManager,
        logger=None,
    ):
        self.alima_manager = alima_manager
        self.cache_manager = cache_manager
        self.logger = logger

    def execute_initial_keyword_extraction(
        self,
        abstract_text: str,
        model: str,
        provider: str = "ollama",
        task: str = "initialisation",
        stream_callback: Optional[callable] = None,
        **kwargs,
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis]:
        """Execute initial keyword extraction step - Claude Generated"""

        # Create abstract data
        abstract_data = AbstractData(abstract=abstract_text, keywords="")

        # Create a compatible stream callback for AlimaManager
        alima_stream_callback = None
        if stream_callback:

            def alima_stream_callback(token):
                # AlimaManager expects callback(token), we have callback(token, step_id)
                # So we call our callback with a default step_id
                stream_callback(token, kwargs.get("step_id", "initialisation"))

        # Filter out our custom parameters that AlimaManager doesn't expect
        alima_kwargs = {k: v for k, v in kwargs.items() if k not in ["step_id"]}

        # Execute analysis via AlimaManager
        task_state = self.alima_manager.analyze_abstract(
            abstract_data=abstract_data,
            task=task,
            model=model,
            provider=provider,
            stream_callback=alima_stream_callback,
            **alima_kwargs,
        )

        if task_state.status == "failed":
            raise ValueError(
                f"Initial keyword extraction failed: {task_state.analysis_result.full_text}"
            )

        # Extract keywords and GND classes from response
        keywords = extract_keywords_from_response(task_state.analysis_result.full_text)
        gnd_classes = extract_gnd_system_from_response(
            task_state.analysis_result.full_text
        )

        # Create analysis details
        llm_analysis = LlmKeywordAnalysis(
            task_name=task,
            model_used=model,
            provider_used=provider,
            prompt_template=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            filled_prompt=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            temperature=kwargs.get("temperature", 0.7),
            seed=kwargs.get("seed", 0),
            response_full_text=task_state.analysis_result.full_text,
            extracted_gnd_keywords=keywords,
            extracted_gnd_classes=gnd_classes,
        )

        return keywords, gnd_classes, llm_analysis

    def execute_gnd_search(
        self,
        keywords: List[str],
        suggesters: List[str] = None,
        stream_callback: Optional[callable] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute GND search step - Claude Generated"""

        if suggesters is None:
            suggesters = ["lobid", "swb"]

        # Create SearchCLI instance
        search_cli = SearchCLI(self.cache_manager)

        # Convert suggester names to types
        suggester_types = []
        for suggester_name in suggesters:
            try:
                suggester_types.append(SuggesterType[suggester_name.upper()])
            except KeyError:
                if self.logger:
                    self.logger.warning(f"Unknown suggester: {suggester_name}")

        # Convert keywords to list if needed
        if isinstance(keywords, str):
            keywords_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        else:
            keywords_list = keywords

        # Stream search progress if callback provided
        if stream_callback:
            stream_callback(
                f"Suche mit {len(keywords_list)} Keywords: {', '.join(keywords_list[:3])}{'...' if len(keywords_list) > 3 else ''}\n",
                "search",
            )
            stream_callback(
                f"Verwende Suggester: {', '.join([st.value for st in suggester_types])}\n",
                "search",
            )

        # Execute search
        search_results = search_cli.search(
            search_terms=keywords_list, suggester_types=suggester_types
        )

        # Stream completion info if callback provided
        if stream_callback:
            total_hits = sum(len(results) for results in search_results.values())
            stream_callback(
                f"Suche abgeschlossen: {total_hits} Treffer in {len(search_results)} Kategorien\n",
                "search",
            )

        return search_results

    def execute_final_keyword_analysis(
        self,
        original_abstract: str,
        search_results: Dict[str, Dict[str, Any]],
        model: str,
        provider: str = "ollama",
        task: str = "keywords",
        stream_callback: Optional[callable] = None,
        **kwargs,
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis]:
        """Execute final keyword analysis step - Claude Generated"""

        # Prepare GND search results for prompt
        gnd_keywords_text = ""
        gnd_compliant_keywords = []

        for results in search_results.values():
            for keyword, data in results.items():
                gnd_ids = data.get("gndid", set())
                for gnd_id in gnd_ids:
                    formatted_keyword = f"{keyword} (GND-ID: {gnd_id})"
                    gnd_keywords_text += formatted_keyword + "\n"
                    gnd_compliant_keywords.append(formatted_keyword)

        # Create abstract data with correct placeholder mapping
        abstract_data = AbstractData(
            abstract=original_abstract,  # This fills {abstract} placeholder
            keywords=gnd_keywords_text,  # This fills {keywords} placeholder
        )

        # Create a compatible stream callback for AlimaManager
        alima_stream_callback = None
        if stream_callback:

            def alima_stream_callback(token):
                # AlimaManager expects callback(token), we have callback(token, step_id)
                stream_callback(token, kwargs.get("step_id", "keywords"))

        # Filter out our custom parameters that AlimaManager doesn't expect
        alima_kwargs = {k: v for k, v in kwargs.items() if k not in ["step_id"]}

        # Execute final analysis
        task_state = self.alima_manager.analyze_abstract(
            abstract_data=abstract_data,
            task=task,
            model=model,
            provider=provider,
            stream_callback=alima_stream_callback,
            **alima_kwargs,
        )

        if task_state.status == "failed":
            raise ValueError(
                f"Final keyword analysis failed: {task_state.analysis_result.full_text}"
            )

        # Extract final keywords and classes
        extracted_keywords_all, extracted_keywords_exact = (
            extract_keywords_from_descriptive_text(
                task_state.analysis_result.full_text, gnd_compliant_keywords
            )
        )
        extracted_gnd_classes = extract_classes_from_descriptive_text(
            task_state.analysis_result.full_text
        )

        # Create final analysis details
        llm_analysis = LlmKeywordAnalysis(
            task_name=task,
            model_used=model,
            provider_used=provider,
            prompt_template=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            filled_prompt=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            temperature=kwargs.get("temperature", 0.7),
            seed=kwargs.get("seed", 0),
            response_full_text=task_state.analysis_result.full_text,
            extracted_gnd_keywords=extracted_keywords_exact,  # Use exact matches
            extracted_gnd_classes=extracted_gnd_classes,
        )

        return extracted_keywords_exact, extracted_gnd_classes, llm_analysis

    def create_complete_analysis_state(
        self,
        original_abstract: str,
        initial_keywords: List[str],
        initial_gnd_classes: List[str],
        search_results: Dict[str, Dict[str, Any]],
        initial_llm_analysis: LlmKeywordAnalysis,
        final_llm_analysis: LlmKeywordAnalysis,
        suggesters_used: List[str] = None,
    ) -> KeywordAnalysisState:
        """Create complete analysis state from pipeline results - Claude Generated"""

        if suggesters_used is None:
            suggesters_used = ["lobid", "swb"]

        # Convert search results to SearchResult objects
        search_result_objects = [
            SearchResult(search_term=term, results=results)
            for term, results in search_results.items()
        ]

        return KeywordAnalysisState(
            original_abstract=original_abstract,
            initial_keywords=initial_keywords,
            search_suggesters_used=suggesters_used,
            initial_gnd_classes=initial_gnd_classes,
            search_results=search_result_objects,
            initial_llm_call_details=initial_llm_analysis,
            final_llm_analysis=final_llm_analysis,
        )


def extract_keywords_from_descriptive_text(
    text: str, gnd_compliant_keywords: List[str]
) -> Tuple[List[str], List[str]]:
    """Extract keywords from LLM descriptive text - Claude Generated"""

    pattern = re.compile(r"\b([A-Za-zäöüÄÖÜß\s-]+?)\s*\((\d{7}-\d|\d{7}-\d{1,2})\)")
    matches = pattern.findall(text)

    all_extracted_keywords = []
    exact_matches = []

    # Convert gnd_compliant_keywords to set for faster lookup
    gnd_compliant_set = set(gnd_compliant_keywords)

    for keyword_part, gnd_id_part in matches:
        formatted_keyword = f"{keyword_part.strip()} ({gnd_id_part})"
        all_extracted_keywords.append(formatted_keyword)

        # Check if formatted keyword from LLM output is in gnd_compliant_keywords list
        if formatted_keyword in gnd_compliant_set:
            exact_matches.append(formatted_keyword)

    return all_extracted_keywords, exact_matches


def extract_keywords_from_descriptive_text_simple(
    text: str, gnd_compliant_keywords: List[str]
) -> List[str]:
    """Simplified keyword extraction using basic string containment - Claude Generated"""

    if not text or not gnd_compliant_keywords:
        return []

    matched_keywords = []
    text_lower = text.lower()

    for gnd_keyword in gnd_compliant_keywords:
        if "(" in gnd_keyword and ")" in gnd_keyword:
            # Extract clean keyword
            clean_keyword = gnd_keyword.split("(")[0].strip().lower()

            # Simple containment check
            if clean_keyword in text_lower:
                matched_keywords.append(gnd_keyword)

    return matched_keywords


def extract_classes_from_descriptive_text(text: str) -> List[str]:
    """Extract classification classes from LLM text - Claude Generated"""

    match = re.search(r"<class>(.*?)</class>", text)
    if match:
        classes_str = match.group(1)
        return [cls.strip() for cls in classes_str.split("|") if cls.strip()]
    return []


class PipelineJsonManager:
    """JSON serialization/deserialization for pipeline states - Claude Generated"""

    @staticmethod
    def task_state_to_dict(task_state: TaskState) -> dict:
        """Convert TaskState to dictionary for JSON serialization - Claude Generated"""
        task_state_dict = asdict(task_state)

        # Convert nested dataclasses to dicts if they exist
        if task_state_dict.get("abstract_data"):
            task_state_dict["abstract_data"] = asdict(task_state.abstract_data)
        if task_state_dict.get("analysis_result"):
            task_state_dict["analysis_result"] = asdict(task_state.analysis_result)
        if task_state_dict.get("prompt_config"):
            task_state_dict["prompt_config"] = asdict(task_state.prompt_config)

        return task_state_dict

    @staticmethod
    def convert_sets_to_lists(obj):
        """Convert sets to lists for JSON serialization - Claude Generated"""
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dict):
            return {
                k: PipelineJsonManager.convert_sets_to_lists(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [PipelineJsonManager.convert_sets_to_lists(elem) for elem in obj]
        return obj

    @staticmethod
    def save_analysis_state(analysis_state: KeywordAnalysisState, file_path: str):
        """Save KeywordAnalysisState to JSON file - Claude Generated"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    PipelineJsonManager.convert_sets_to_lists(asdict(analysis_state)),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
        except Exception as e:
            raise ValueError(f"Error saving analysis state to JSON: {e}")

    @staticmethod
    def load_analysis_state(file_path: str) -> KeywordAnalysisState:
        """Load KeywordAnalysisState from JSON file - Claude Generated"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return KeywordAnalysisState(**data)
        except FileNotFoundError:
            raise ValueError(f"Analysis state file not found: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading analysis state: {e}")

    @staticmethod
    def save_task_state(task_state: TaskState, file_path: str):
        """Save TaskState to JSON file - Claude Generated"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    PipelineJsonManager.task_state_to_dict(task_state),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
        except Exception as e:
            raise ValueError(f"Error saving task state to JSON: {e}")


class PipelineResultFormatter:
    """Format pipeline results for display - Claude Generated"""

    @staticmethod
    def format_search_results_for_display(
        search_results: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Format search results as list of strings for display - Claude Generated"""
        formatted_results = []

        for search_term, results in search_results.items():
            for keyword, data in results.items():
                gnd_ids = data.get("gndid", set())
                for gnd_id in gnd_ids:
                    formatted_results.append(f"{keyword} (GND: {gnd_id})")

        return formatted_results

    @staticmethod
    def format_keywords_for_prompt(search_results: Dict[str, Dict[str, Any]]) -> str:
        """Format search results as text for LLM prompt - Claude Generated"""
        search_results_text = ""

        for search_term, results in search_results.items():
            search_results_text += f"Search Term: {search_term}\n"
            for keyword, data in results.items():
                gnd_ids = ", ".join(data.get("gndid", [])) if data.get("gndid") else ""
                formatted_keyword = f"{keyword} ({gnd_ids})" if gnd_ids else keyword
                search_results_text += f"  - {formatted_keyword}\n"

        return search_results_text

    @staticmethod
    def get_gnd_compliant_keywords(
        search_results: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Extract GND-compliant keywords from search results - Claude Generated"""
        gnd_keywords = []

        for results in search_results.values():
            for keyword, data in results.items():
                gnd_ids = data.get("gndid", set())
                for gnd_id in gnd_ids:
                    gnd_keywords.append(f"{keyword} (GND-ID: {gnd_id})")

        return gnd_keywords


def execute_complete_pipeline(
    alima_manager,
    cache_manager: CacheManager,
    input_text: str,
    initial_model: str,
    final_model: str,
    provider: str = "ollama",
    suggesters: List[str] = None,
    stream_callback: Optional[callable] = None,
    logger=None,
    **kwargs,
) -> KeywordAnalysisState:
    """Execute complete pipeline from start to finish - Claude Generated"""

    if suggesters is None:
        suggesters = ["lobid", "swb"]

    executor = PipelineStepExecutor(alima_manager, cache_manager, logger)

    # Step 1: Initial keyword extraction
    if stream_callback:
        stream_callback("Starting initial keyword extraction...\n", "initialisation")

    initial_keywords, initial_gnd_classes, initial_llm_analysis = (
        executor.execute_initial_keyword_extraction(
            abstract_text=input_text,
            model=initial_model,
            provider=provider,
            task="initialisation",
            stream_callback=stream_callback,
            **kwargs,
        )
    )

    # Step 2: GND search
    if stream_callback:
        stream_callback("Starting GND search...\n", "search")

    search_results = executor.execute_gnd_search(
        keywords=initial_keywords,
        suggesters=suggesters,
        stream_callback=stream_callback,
    )

    # Step 3: Final keyword analysis
    if stream_callback:
        stream_callback("Starting final keyword analysis...\n", "keywords")

    final_keywords, final_gnd_classes, final_llm_analysis = (
        executor.execute_final_keyword_analysis(
            original_abstract=input_text,
            search_results=search_results,
            model=final_model,
            provider=provider,
            task="keywords",
            stream_callback=stream_callback,
            **kwargs,
        )
    )

    # Create complete analysis state
    analysis_state = executor.create_complete_analysis_state(
        original_abstract=input_text,
        initial_keywords=initial_keywords,
        initial_gnd_classes=initial_gnd_classes,
        search_results=search_results,
        initial_llm_analysis=initial_llm_analysis,
        final_llm_analysis=final_llm_analysis,
        suggesters_used=suggesters,
    )

    if stream_callback:
        stream_callback("Pipeline completed successfully!\n", "completion")

    return analysis_state
