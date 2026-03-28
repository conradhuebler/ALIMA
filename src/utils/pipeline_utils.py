"""
Pipeline Utils - Shared logic for CLI and GUI pipeline implementations
Claude Generated - Abstracts common pipeline operations and utilities
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime
from urllib.parse import urlparse  # For URL parsing in title builder - Claude Generated

logger = logging.getLogger(__name__)

from ..core.data_models import (
    AbstractData,
    TaskState,
    AnalysisResult,
    KeywordAnalysisState,
    LlmKeywordAnalysis,
    SearchResult,
)
from ..core.search_cli import SearchCLI
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from .suggesters.meta_suggester import SuggesterType
from ..core.processing_utils import (
    extract_keywords_from_response,
    extract_gnd_system_from_response,
    extract_title_from_response,  # For LLM title extraction - Claude Generated
    extract_missing_concepts_from_response,  # For iterative refinement - Claude Generated
)
from .smart_provider_selector import SmartProviderSelector
from .config_models import TaskType
from .pipeline_defaults import DEFAULT_DK_MAX_RESULTS, DEFAULT_DK_FREQUENCY_THRESHOLD


# Title Building Utilities - Claude Generated
def sanitize_for_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text for use in filenames - Claude Generated

    Args:
        text: Text to sanitize
        max_length: Maximum length (default 50)

    Returns:
        Sanitized filename-safe string
    """
    if not text:
        return "untitled"

    # Replace problematic characters with underscores
    # Windows forbidden: < > : " / \ | ? *
    # Also replace spaces, commas, periods
    sanitized = re.sub(r'[<>:"/\\|?*\s,.]', '_', text)

    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Trim underscores from start/end
    sanitized = sanitized.strip('_')

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    return sanitized or "untitled"


def build_working_title(
    llm_title: Optional[str],
    source_identifier: str,
    timestamp: Optional[str] = None,
    fallback_prefix: str = "analysis"
) -> str:
    """
    Build complete working title: {llm_title}_{source}_{timestamp} - Claude Generated

    Args:
        llm_title: Title extracted from LLM (can be None)
        source_identifier: DOI/filename/URL identifier
        timestamp: ISO timestamp (generated if None)
        fallback_prefix: Prefix when llm_title missing

    Returns:
        Complete working title string
    """
    # Generate timestamp if not provided
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        # Convert ISO to compact format if needed
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp = dt.strftime('%Y%m%d_%H%M%S')
        except:
            # If parsing fails, use as-is (might already be compact)
            pass

    # Build title components
    components = []

    # Component 1: LLM title or fallback
    if llm_title:
        components.append(sanitize_for_filename(llm_title, max_length=30))
    else:
        components.append(fallback_prefix)

    # Component 2: Source identifier
    source_clean = sanitize_for_filename(source_identifier, max_length=40)
    if source_clean and source_clean != "untitled":
        components.append(source_clean)

    # Component 3: Timestamp
    components.append(timestamp)

    # Combine with underscores
    return '_'.join(components)


def extract_source_identifier(
    input_type: str,
    input_value: str
) -> str:
    """
    Extract clean source identifier from input - Claude Generated

    Args:
        input_type: 'text', 'doi', 'pdf', 'img', 'url'
        input_value: The actual input value/path

    Returns:
        Clean source identifier string
    """
    if input_type == 'doi':
        # DOI: Use the DOI itself (already sanitized by build_working_title)
        return input_value

    elif input_type in ('pdf', 'img'):
        # File: Use basename without extension
        from pathlib import Path
        return Path(input_value).stem

    elif input_type == 'url':
        # URL: Extract domain
        parsed = urlparse(input_value)
        return parsed.netloc or 'url'

    else:  # 'text'
        # Text: Don't include text preview in identifier - Claude Generated
        return 'text'


class PipelineStepExecutor:
    """Shared pipeline step execution logic - Claude Generated"""

    def __init__(
        self,
        alima_manager,
        cache_manager: UnifiedKnowledgeManager,
        logger=None,
        config_manager=None,
    ):
        self.alima_manager = alima_manager
        self.cache_manager = cache_manager
        self.logger = logger
        self.config_manager = config_manager
        
        # Initialize SmartProviderSelector if config_manager available
        self.smart_selector = None
        if config_manager:
            try:
                self.smart_selector = SmartProviderSelector(config_manager)
                if logger:
                    logger.info("PipelineStepExecutor initialized with SmartProviderSelector")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to initialize SmartProviderSelector: {e}")
                    logger.info("Falling back to config-based provider selection")

    def _resolve_provider_smart(self, provider: str, model: str, task_type: str, prefer_fast: bool = False, task_name: str = None, step_id: str = None) -> tuple[str, str]:
        """Intelligent provider/model resolution with proper fallback chain - Claude Generated

        Priority Order (FIXED - Issue #2):
        1. Explicit UI parameters (highest priority) - user manual selection
        2. Task preferences from config - auto-selection based on task
        3. Config defaults - fallback provider configuration
        4. Detection service fallback - last resort
        """

        # ✅ PRIORITY 1: Explicit UI parameters (FIXED: was evaluating task preferences first)
        # User manually selected in UI combo boxes - MUST respect this!
        # Check for both non-empty and non-whitespace values
        if provider and provider.strip() and model and model.strip():
            if self.logger:
                self.logger.info(f"🎯 Using EXPLICIT UI selection: {provider}/{model} (overrides task preferences)")
            return provider, model

        # PRIORITY 2: SmartProviderSelector with task preferences (was priority 1)
        # Only use if no explicit UI selection provided
        if self.smart_selector:
            try:
                # Map string to TaskType enum
                task_type_mapping = {
                    "text": TaskType.TEXT,
                    "classification": TaskType.CLASSIFICATION,
                    "vision": TaskType.VISION
                }

                task_type_enum = task_type_mapping.get(task_type.lower(), TaskType.TEXT)

                selection = self.smart_selector.select_provider(
                    task_type=task_type_enum,
                    prefer_fast=prefer_fast,
                    task_name=task_name,
                    step_id=step_id
                )

                # Use SmartProvider selection
                final_provider = selection.provider
                final_model = selection.model

                # Enhanced logging to show task preference usage
                if self.logger:
                    if task_name:
                        self.logger.info(f"📋 Using task preference: {final_provider}/{final_model} (task: {task_name})")
                    else:
                        self.logger.info(f"⚙️ Using config default: {final_provider}/{final_model} (task_type: {task_type})")

                return final_provider, final_model

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"SmartProvider selection failed: {e}")

        # 3. Config-manager fallbacks (when SmartProvider unavailable)
        if self.config_manager:
            try:
                config = self.config_manager.load_config()

                # Try to get default provider/model from config
                if hasattr(config, 'llm') and hasattr(config.unified_config, 'default_provider'):
                    config_provider = provider or config.unified_config.default_provider
                    config_model = model or getattr(config.unified_config, 'default_model', None)

                    if config_provider and config_model:
                        if self.logger:
                            self.logger.info(f"Using config defaults: {config_provider}/{config_model}")
                        return config_provider, config_model

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Config fallback failed: {e}")

        # 4. System defaults (last resort only)
        # Use first available provider instead of hardcoded fallback - Claude Generated
        fallback_provider = provider or self._get_first_enabled_provider()

        # If no provider available at all, return None to signal configuration error
        if fallback_provider is None:
            if self.logger:
                self.logger.error("No providers configured. Please run first-start wizard or configure a provider.")
            return None, None

        # BUGFIX: Removed hardcoded task_defaults - use explicit model parameter or error
        # Model must come from SmartProvider or be explicitly provided
        if not model:
            error_msg = (
                f"No model specified for task type {task_type}. "
                f"Provider {fallback_provider} selected but model missing. "
                f"Check task preferences for '{task_name or task_type}' or provider configuration."
            )
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

        fallback_model = model

        if self.logger:
            self.logger.warning(f"Using system fallback provider: {fallback_provider}/{fallback_model} (no SmartProvider or Config available)")

        return fallback_provider, fallback_model

    def _get_first_enabled_provider(self) -> Optional[str]:
        """Get the first enabled provider name from config (any type) - Claude Generated"""
        try:
            if self.smart_selector and hasattr(self.smart_selector, 'config'):
                config = self.smart_selector.config
                # Get unified config and find first enabled provider (any type)
                if hasattr(config, 'unified_config') and config.unified_config:
                    enabled_providers = config.unified_config.get_enabled_providers()
                    if enabled_providers:
                        return enabled_providers[0].name

            # No providers available
            if self.logger:
                self.logger.error("No enabled providers found in configuration")
            return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get enabled provider: {e}")
            return None

    def _create_stream_callback_adapter(self, stream_callback: Optional[callable], step_id: str, debug: bool = False) -> Optional[callable]:
        """Centralized stream callback adapter creation - Claude Generated"""
        if not stream_callback:
            if debug and self.logger:
                self.logger.warning(f"⚠️ No stream callback provided for {step_id} step")
            return None

        if debug and self.logger:
            self.logger.info(f"🔄 Creating stream callback adapter for {step_id} step")

        def alima_stream_callback(token):
            try:
                if debug and self.logger:
                    self.logger.debug(f"📡 Stream token received: '{token[:50]}...', forwarding to step_id='{step_id}'")
                stream_callback(token, step_id)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Stream callback error: {e}")

        return alima_stream_callback

    def _filter_alima_kwargs(self, kwargs: Dict[str, Any], exclude_llm_params: bool = False) -> Dict[str, Any]:
        """Centralized parameter filtering for AlimaManager calls - Claude Generated"""
        excluded_params = ["step_id", "keyword_chunking_threshold", "chunking_task", "expand_synonyms", "dk_max_results", "dk_frequency_threshold"]

        # Some methods need to exclude LLM parameters that are handled separately
        if exclude_llm_params:
            excluded_params.extend(["top_p", "temperature"])

        return {k: v for k, v in kwargs.items() if k not in excluded_params}

    def execute_initial_keyword_extraction(
        self,
        abstract_text: str,
        model: str = None,
        provider: str = None,
        task: str = "initialisation",
        stream_callback: Optional[callable] = None,
        **kwargs,
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis, Optional[str]]:
        """Execute initial keyword extraction step with intelligent provider selection - Claude Generated"""

        # Intelligent provider selection using centralized method - Claude Generated
        provider, model = self._resolve_provider_smart(
            provider=provider,
            model=model,
            task_type="text",
            prefer_fast=True,  # Initial extraction can prioritize speed
            task_name=task,
            step_id="initialisation"
        )

        # Create abstract data
        abstract_data = AbstractData(abstract=abstract_text, keywords="")

        # Create stream callback adapter using centralized method - Claude Generated
        alima_stream_callback = self._create_stream_callback_adapter(
            stream_callback,
            kwargs.get("step_id", "initialisation"),
            debug=True
        )

        # Filter parameters using centralized method - Claude Generated
        alima_kwargs = self._filter_alima_kwargs(kwargs)

        # Execute analysis via AlimaManager - ENHANCED DEBUG - Claude Generated
        if self.logger:
            self.logger.info(f"🚀 Calling AlimaManager.analyze_abstract:")
            self.logger.info(f"   📋 task='{task}', model='{model}', provider='{provider}'")
            self.logger.info(f"   🔄 stream_callback={'✅ YES' if alima_stream_callback else '❌ NONE'}")
            self.logger.info(f"   ⚙️ kwargs={list(alima_kwargs.keys())}")

        task_state = self.alima_manager.analyze_abstract(
            abstract_data=abstract_data,
            task=task,
            model=model,
            provider=provider,
            stream_callback=alima_stream_callback,
            **alima_kwargs,
        )

        if self.logger:
            self.logger.info(f"📊 AlimaManager result: status='{task_state.status}'")
            if task_state.status == "failed":
                self.logger.error(f"❌ Analysis failed: {task_state.analysis_result.full_text}")
            else:
                response_preview = task_state.analysis_result.full_text[:100] if task_state.analysis_result.full_text else "NO RESPONSE"
                self.logger.info(f"✅ Analysis success: '{response_preview}...'")

        if task_state.status == "failed":
            raw_error = task_state.analysis_result.full_text
            error_msg = f"Initial keyword extraction failed: {raw_error}"
            if self.logger:
                self.logger.error(f"💥 PIPELINE_FAILURE: {error_msg}")
            # Stream a connection hint if it looks like a network error - Claude Generated
            if stream_callback and raw_error and any(kw in raw_error.lower() for kw in ("connect", "timeout", "network", "unreachable", "refused", "name or service")):
                stream_callback(f"\n🔌 Server nicht erreichbar – Verbindung prüfen ({provider})\n", kwargs.get("step_id", "initialisation"))
            raise ValueError(error_msg)

        # Extract keywords and GND classes from response
        # Pass output_format for JSON extraction - Claude Generated
        _output_format = getattr(task_state.prompt_config, 'output_format', None) if task_state.prompt_config else None
        keywords = extract_keywords_from_response(task_state.analysis_result.full_text, output_format=_output_format)
        gnd_classes = extract_gnd_system_from_response(
            task_state.analysis_result.full_text, output_format=_output_format
        )

        # Extract title from response - Claude Generated
        llm_title = extract_title_from_response(task_state.analysis_result.full_text, output_format=_output_format)

        if self.logger:
            if llm_title:
                self.logger.info(f"📝 Extracted LLM title: '{llm_title}'")
            else:
                self.logger.warning("⚠️ No <final_title> found in LLM response - will use fallback")

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

        return keywords, gnd_classes, llm_analysis, llm_title  # BREAKING CHANGE: Now returns 4-tuple - Claude Generated

    def execute_gnd_search(
        self,
        keywords: List[str],
        suggesters: List[str] = None,
        stream_callback: Optional[callable] = None,
        catalog_token: str = None,
        catalog_search_url: str = None,
        catalog_details_url: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute GND search step with automatic catalog detection - Claude Generated"""

        if suggesters is None:
            suggesters = ["lobid", "swb"]
            
            # Add catalog if available (no auto-detection, explicit configuration)
            # Catalog will be added via suggesters parameter in pipeline

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

        # Stream search progress if callback provided - Claude Generated
        # === DIAGNOSTIC: Log suggester configuration ===
        if self.logger:
            self.logger.info(f"🔍 execute_gnd_search: {len(keywords_list)} Keywords, Suggester: {[st.value for st in suggester_types]}")
        if stream_callback:
            stream_callback(
                f"Suche mit {len(keywords_list)} Keywords: {', '.join(keywords_list)}\n",
                "search",
            )
            stream_callback(
                f"Verwende Suggester: {', '.join([st.value for st in suggester_types])}\n",
                "search",
            )

        # Execute search per keyword for live progress updates - Claude Generated
        search_results = {}
        with SearchCLI(
            self.cache_manager,
            catalog_token=catalog_token or "",
            catalog_search_url=catalog_search_url or "",
            catalog_details_url=catalog_details_url or ""
        ) as search_cli:
            for keyword in keywords_list:
                if stream_callback:
                    stream_callback(f"🔍 Suche '{keyword}'...\n", "search")

                kw_results = search_cli.search(
                    search_terms=[keyword], suggester_types=suggester_types
                )

                # Merge into combined results
                for term, term_data in kw_results.items():
                    if term not in search_results:
                        search_results[term] = {}
                    search_results[term].update(term_data)

                if stream_callback:
                    total_hits = sum(
                        details.get('count', 0)
                        for details in search_results.get(keyword, {}).values()
                    )
                    stream_callback(f"    ✓ '{keyword}': {total_hits} Treffer\n", "search")

        # Post-process catalog results: validate subjects against cache and SWB
        if "catalog" in suggesters:
            search_results = self._validate_catalog_subjects(
                search_results, stream_callback
            )

        if stream_callback:
            stream_callback("--> Suche abgeschlossen.\n", "search")

        return search_results

    def execute_fallback_gnd_search(
        self,
        missing_concepts: List[str],
        existing_results: Dict[str, Dict[str, Any]],
        stream_callback: Optional[callable] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Search GND for missing concepts identified by LLM.
        Claude Generated

        Args:
            missing_concepts: List of concepts not covered by existing GND pool
            existing_results: Current search results to avoid duplicates
            stream_callback: Progress feedback callback

        Returns:
            Merged search results (existing + new)

        Strategy:
            1. Search GND for each missing concept
            2. Track concepts not found in GND
            3. Merge new results with existing results (union GND-IDs for duplicates)
            4. Return enriched search results dict
        """
        if stream_callback:
            stream_callback(
                f"\n🔍 Fallback-Suche für {len(missing_concepts)} fehlende Konzepte...\n",
                "keywords_refinement"
            )

        new_results = {}
        concepts_not_found = []

        for concept in missing_concepts:
            if stream_callback:
                stream_callback(f"  Suche: {concept}\n", "keywords_refinement")

            # Execute search via SearchCLI using context manager - Claude Generated
            try:
                from ..core.search_cli import SearchCLI

                with SearchCLI(self.cache_manager) as search_cli:
                    # Search with SWB and Lobid suggesters
                    search_result_dict = search_cli.search(
                        search_terms=[concept],
                        suggester_types=[SuggesterType.LOBID, SuggesterType.SWB]
                    )

                    # Extract results for this concept
                    if concept in search_result_dict and search_result_dict[concept]:
                        new_results[concept] = search_result_dict[concept]

                        # Count total GND-IDs found
                        total_gnd_ids = sum(
                            len(data.get("gndid", set()))
                            for data in search_result_dict[concept].values()
                        )

                        if stream_callback:
                            stream_callback(f"    ✓ {total_gnd_ids} GND-Einträge gefunden\n", "keywords_refinement")
                    else:
                        concepts_not_found.append(concept)
                        if stream_callback:
                            stream_callback(f"    ✗ Keine GND-Einträge gefunden\n", "keywords_refinement")
            except Exception as e:
                logger.warning(f"Fallback search failed for concept '{concept}': {e}")
                concepts_not_found.append(concept)
                if stream_callback:
                    stream_callback(f"    ✗ Suchfehler: {str(e)}\n", "keywords_refinement")

        # Merge with existing results using atomic rollback pattern - Claude Generated
        # Deep copy to prevent partial corruption on failure
        import copy
        merged_results = copy.deepcopy(existing_results)

        try:
            for concept, concept_data in new_results.items():
                # concept_data is: {keyword: {gndid: set(), ddc: set(), dk: set(), count: int}}
                for keyword, data in concept_data.items():
                    if concept in merged_results:
                        # Concept already exists as search term - merge at keyword level
                        if keyword in merged_results[concept]:
                            # Merge GND-IDs, DDC, and DK codes
                            merged_results[concept][keyword]["gndid"].update(data.get("gndid", set()))
                            merged_results[concept][keyword]["ddc"].update(data.get("ddc", set()))
                            merged_results[concept][keyword]["dk"].update(data.get("dk", set()))
                            # Update count
                            merged_results[concept][keyword]["count"] = max(
                                merged_results[concept][keyword].get("count", 0),
                                data.get("count", 0)
                            )
                        else:
                            # New keyword for existing search term - deep copy to isolate
                            merged_results[concept][keyword] = copy.deepcopy(data)
                    else:
                        # New search term entirely - deep copy to isolate
                        merged_results[concept] = copy.deepcopy(concept_data)
        except Exception as e:
            # Rollback: return original existing_results unchanged
            logger.error(f"Merge failed, rolling back to existing results: {e}")
            if stream_callback:
                stream_callback(f"⚠️  Merge-Fehler, verwende vorherige Ergebnisse: {str(e)}\n", "keywords_refinement")
            return existing_results

        if stream_callback:
            stream_callback(
                f"\n📊 Fallback-Ergebnis: {len(new_results)}/{len(missing_concepts)} Konzepte gefunden\n",
                "keywords_refinement"
            )
            if concepts_not_found and len(concepts_not_found) <= 5:
                stream_callback(
                    f"⚠️  Nicht gefunden: {', '.join(concepts_not_found)}\n",
                    "keywords_refinement"
                )
            elif concepts_not_found:
                stream_callback(
                    f"⚠️  Nicht gefunden: {len(concepts_not_found)} Konzepte\n",
                    "keywords_refinement"
                )

        return merged_results

    def execute_iterative_keyword_refinement(
        self,
        original_abstract: str,
        initial_search_results: Dict[str, Dict[str, Any]],
        model: str,
        provider: str,
        max_iterations: int = 2,
        stream_callback: Optional[callable] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[str], Dict[str, Any], LlmKeywordAnalysis]:
        """
        Iteratively refine keyword selection by searching for missing concepts.
        Claude Generated

        Process:
            1. Run initial keyword analysis
            2. Extract missing concepts from <missing_list>
            3. If missing concepts found AND iterations remaining:
               a. Search GND for missing concepts
               b. Merge results into GND pool
               c. Re-run keyword analysis
               d. Check for convergence
            4. Return final keywords + enriched state

        Args:
            original_abstract: The abstract text
            initial_search_results: Initial GND search results
            model: LLM model to use
            provider: LLM provider
            max_iterations: Maximum refinement iterations (default: 2)
            stream_callback: Progress callback
            checkpoint_path: Optional path prefix for checkpoint files (enables crash recovery)

        Returns:
            (final_keywords, iteration_metadata, final_llm_analysis)

        Convergence Conditions:
            - No missing concepts in LLM response → STOP (success)
            - Missing concepts identical to previous iteration → STOP (self-consistency)
            - GND search finds no new matches → STOP (no improvement possible)
            - Max iterations reached → STOP (timeout)
        """
        current_search_results = initial_search_results.copy()
        iteration_history = []
        previous_missing_concepts = []
        final_keywords = []
        final_llm_analysis = None

        def _save_checkpoint(iteration_num: int) -> None:
            """Save iteration checkpoint for crash recovery - Claude Generated"""
            if not checkpoint_path:
                return

            try:
                checkpoint_data = {
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_num,
                    "original_abstract": original_abstract,
                    "current_search_results": PipelineJsonManager.convert_sets_to_lists(current_search_results),
                    "iteration_history": iteration_history,
                    "final_keywords": final_keywords,
                    "model": model,
                    "provider": provider,
                    "max_iterations": max_iterations
                }

                checkpoint_file = f"{checkpoint_path}_iter{iteration_num}.json"
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

                logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
                # Don't fail the iteration if checkpoint fails

        # Retry configuration - Claude Generated
        MAX_RETRIES = 3
        RETRY_DELAY_SECONDS = 2

        for iteration in range(1, max_iterations + 1):
            if stream_callback:
                stream_callback(
                    f"\n{'='*60}\n🔄 Iteration {iteration}/{max_iterations}\n{'='*60}\n",
                    "keywords_refinement"
                )

            # 1. Execute keyword analysis with current GND pool
            # Wrapped in try-catch with retry logic for LLM failures - Claude Generated
            llm_analysis = None
            last_exception = None

            for retry in range(MAX_RETRIES):
                try:
                    final_keywords, _, llm_analysis = self.execute_final_keyword_analysis(
                        original_abstract=original_abstract,
                        search_results=current_search_results,
                        model=model,
                        provider=provider,
                        stream_callback=stream_callback,
                        **kwargs
                    )
                    break  # Success - exit retry loop
                except (TimeoutError, ConnectionError) as e:
                    # Transient errors - retry
                    last_exception = e
                    if retry < MAX_RETRIES - 1:
                        if stream_callback:
                            stream_callback(
                                f"⚠️  LLM-Fehler (Retry {retry + 1}/{MAX_RETRIES}): {str(e)}\n",
                                "keywords_refinement"
                            )
                        import time
                        time.sleep(RETRY_DELAY_SECONDS)
                    continue
                except ValueError as e:
                    # Parse errors or LLM refused - don't retry, use current state
                    last_exception = e
                    if stream_callback:
                        stream_callback(
                            f"⚠️  LLM-Analyse fehlgeschlagen: {str(e)}\n",
                            "keywords_refinement"
                        )
                    logger.warning(f"Iteration {iteration}: LLM analysis failed (non-retryable): {e}")
                    break
                except Exception as e:
                    # Unknown error - log and try to continue
                    last_exception = e
                    logger.error(f"Iteration {iteration}: Unexpected error in keyword analysis: {e}")
                    if retry < MAX_RETRIES - 1:
                        import time
                        time.sleep(RETRY_DELAY_SECONDS)
                    continue

            # Handle case where all retries failed - Claude Generated
            if llm_analysis is None:
                if stream_callback:
                    stream_callback(
                        f"❌ Iteration {iteration} abgebrochen: Alle LLM-Versuche fehlgeschlagen\n",
                        "keywords_refinement"
                    )
                logger.error(f"Iteration {iteration}: All {MAX_RETRIES} retries failed. Last error: {last_exception}")

                # Record failed iteration and stop
                iteration_data = {
                    "iteration": iteration,
                    "missing_concepts": [],
                    "keywords_selected": len(final_keywords) if final_keywords else 0,
                    "gnd_pool_size": len(current_search_results),
                    "convergence_reason": "llm_failure",
                    "error": str(last_exception)
                }
                iteration_history.append(iteration_data)
                _save_checkpoint(iteration)  # Save checkpoint after LLM failure
                break

            final_llm_analysis = llm_analysis

            # 2. Extract missing concepts from LLM response
            # Get output_format from prompt config for the task - Claude Generated
            _iter_output_format = None
            if self.alima_manager and hasattr(self.alima_manager, 'prompt_service'):
                _iter_pc = self.alima_manager.prompt_service.get_prompt_config("keywords", model)
                _iter_output_format = getattr(_iter_pc, 'output_format', None) if _iter_pc else None
            missing_concepts = extract_missing_concepts_from_response(
                llm_analysis.response_full_text, output_format=_iter_output_format
            )
            llm_analysis.missing_concepts = missing_concepts

            # 3. Record iteration data
            iteration_data = {
                "iteration": iteration,
                "missing_concepts": missing_concepts.copy(),
                "keywords_selected": len(final_keywords),
                "gnd_pool_size": len(current_search_results)
            }

            if stream_callback:
                stream_callback(
                    f"\n📋 Iteration {iteration} Ergebnis:\n"
                    f"  - Keywords: {len(final_keywords)}\n"
                    f"  - Fehlende Konzepte: {len(missing_concepts)}\n",
                    "keywords_refinement"
                )

            # 4. Check convergence conditions

            # Condition 1: No missing concepts
            if not missing_concepts:
                if stream_callback:
                    stream_callback(
                        "✓ Konvergenz erreicht: Keine fehlenden Konzepte\n",
                        "keywords_refinement"
                    )
                iteration_data["convergence_reason"] = "no_missing_concepts"
                iteration_history.append(iteration_data)
                _save_checkpoint(iteration)  # Save checkpoint after convergence
                break

            # Condition 2: Self-consistency (same missing concepts as before)
            if missing_concepts == previous_missing_concepts:
                if stream_callback:
                    stream_callback(
                        "✓ Konvergenz erreicht: Identische fehlende Konzepte\n",
                        "keywords_refinement"
                    )
                iteration_data["convergence_reason"] = "self_consistency"
                iteration_history.append(iteration_data)
                _save_checkpoint(iteration)  # Save checkpoint after self-consistency
                break

            # 5. Not last iteration? Search for missing concepts
            if iteration < max_iterations:
                enriched_results = self.execute_fallback_gnd_search(
                    missing_concepts=missing_concepts,
                    existing_results=current_search_results,
                    stream_callback=stream_callback,
                    **kwargs
                )

                # Calculate new keywords found
                new_count = len(enriched_results) - len(current_search_results)
                iteration_data["new_gnd_results"] = new_count

                # Condition 3: No new GND results
                if new_count == 0:
                    if stream_callback:
                        stream_callback(
                            "⚠️  Keine neuen GND-Einträge gefunden - Iteration beendet\n",
                            "keywords_refinement"
                        )
                    iteration_data["convergence_reason"] = "no_new_results"
                    iteration_history.append(iteration_data)
                    _save_checkpoint(iteration)  # Save checkpoint after no-new-results
                    break

                # Update for next iteration
                current_search_results = enriched_results
                previous_missing_concepts = missing_concepts.copy()
                iteration_history.append(iteration_data)
                _save_checkpoint(iteration)  # Save checkpoint after successful iteration
            else:
                # Condition 4: Max iterations reached
                iteration_data["convergence_reason"] = "max_iterations"
                iteration_history.append(iteration_data)
                _save_checkpoint(iteration)  # Save checkpoint after max iterations
                if stream_callback:
                    stream_callback(
                        f"⚠️  Maximale Iterationen ({max_iterations}) erreicht\n",
                        "keywords_refinement"
                    )

        # 6. Build enriched metadata
        state_metadata = {
            "total_iterations": len(iteration_history),
            "iteration_history": iteration_history,
            "final_gnd_pool_size": len(current_search_results),
            "convergence_achieved": any(
                it.get("convergence_reason") not in ["max_iterations", None]
                for it in iteration_history
            )
        }

        if stream_callback:
            stream_callback(
                f"\n{'='*60}\n"
                f"✅ Iterative Refinement abgeschlossen\n"
                f"  - Gesamt-Iterationen: {state_metadata['total_iterations']}\n"
                f"  - Konvergenz: {'✓ Ja' if state_metadata['convergence_achieved'] else '✗ Nein'}\n"
                f"  - Finale Keywords: {len(final_keywords)}\n"
                f"{'='*60}\n",
                "keywords_refinement"
            )

        return final_keywords, state_metadata, final_llm_analysis

    def _validate_catalog_subjects(
        self, 
        search_results: Dict[str, Dict[str, Any]], 
        stream_callback: Optional[callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Claude Generated - Validate catalog subjects against cache and SWB fallback.
        
        Catalog subjects don't have GND-IDs, so we need to:
        1. Check local cache for existing GND-IDs
        2. Use SWB fallback for unknown subjects
        """
        if stream_callback:
            stream_callback("Validiere Katalog-Schlagwörter gegen Cache und SWB...\n", "search")
        
        # Collect all catalog subjects without GND-IDs
        unknown_subjects = []
        catalog_subjects_found = 0
        
        for search_term, term_results in search_results.items():
            for subject, data in term_results.items():
                gnd_ids = data.get("gndid", set())
                if not gnd_ids:  # Subject from catalog without GND-ID
                    catalog_subjects_found += 1
                    
                    # Check cache first - Claude Generated - use new method to get all GND-IDs
                    cached_gnd_ids = self.cache_manager.get_all_gnd_ids_for_keyword(subject)
                    if cached_gnd_ids:
                        # Found in cache - add all GND-IDs
                        data["gndid"].update(cached_gnd_ids)
                        if self.logger:
                            self.logger.debug(f"Cache hit: {subject} -> {len(cached_gnd_ids)} GND-IDs")
                    else:
                        # Not in cache - mark for SWB lookup
                        unknown_subjects.append(subject)
        
        if stream_callback:
            stream_callback(f"Katalog-Subjects gefunden: {catalog_subjects_found}\n", "search")
            stream_callback(f"Cache-Treffer: {catalog_subjects_found - len(unknown_subjects)}\n", "search")
            stream_callback(f"SWB-Lookup erforderlich: {len(unknown_subjects)}\n", "search")
        
        # SWB fallback for unknown subjects
        if unknown_subjects:
            if stream_callback:
                stream_callback(f"Starte SWB-Fallback für {len(unknown_subjects)} unbekannte Subjects...\n", "search")
            
            # Claude Generated - Debug information for SWB fallback
            if self.logger:
                self.logger.info(f"SWB-Fallback: Searching for {len(unknown_subjects)} unknown subjects:")
                for i, subject in enumerate(unknown_subjects):  # Show all - Claude Generated
                    self.logger.info(f"  {i+1}. '{subject}'")
            
            try:
                # Use SWB suggester for validation with context manager - Claude Generated
                with SearchCLI(self.cache_manager) as swb_search_cli:
                    # Search unknown subjects via SWB
                    swb_results = swb_search_cli.search(
                        search_terms=unknown_subjects,
                        suggester_types=[SuggesterType.SWB]
                    )

                    # Claude Generated - Debug SWB results before merging
                    if self.logger:
                        total_swb_subjects = sum(len(term_results) for term_results in swb_results.values())
                        total_swb_gnd_ids = sum(
                            len(data.get("gndid", set()))
                            for term_results in swb_results.values()
                            for data in term_results.values()
                        )
                        self.logger.info(f"SWB-Ergebnisse: {total_swb_subjects} Subjects mit {total_swb_gnd_ids} GND-IDs gefunden")

                        # Show detailed results for first few terms
                        for i, (term, term_results) in enumerate(swb_results.items()):
                            if i >= 5:  # Limit to first 5 terms
                                break
                            self.logger.info(f"  SWB '{term}': {len(term_results)} Subjects gefunden")
                            for j, (subject, data) in enumerate(term_results.items()):
                                if j >= 3:  # Limit to first 3 subjects per term
                                    break
                                gnd_count = len(data.get("gndid", set()))
                                self.logger.info(f"    - '{subject}': {gnd_count} GND-IDs")

                    # Merge SWB results back into original results
                    self._merge_swb_validation_results(search_results, swb_results, stream_callback)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"SWB fallback failed: {e}")
                if stream_callback:
                    stream_callback(f"SWB-Fallback-Fehler: {str(e)}\n", "search")
        
        return search_results
    
    def _merge_swb_validation_results(
        self,
        original_results: Dict[str, Dict[str, Any]],
        swb_results: Dict[str, Dict[str, Any]],
        stream_callback: Optional[callable] = None
    ):
        """Claude Generated - Merge SWB validation results back into original catalog results"""
        
        swb_matches = 0
        processed_matches = set()  # Track processed combinations to avoid duplicates
        unmatched_swb_subjects = []  # Track subjects that couldn't be matched
        
        # Create lookup map for faster matching
        original_lookup = {}
        for search_term, term_results in original_results.items():
            for orig_keyword in term_results.keys():
                key = orig_keyword.lower()
                if key not in original_lookup:
                    original_lookup[key] = []
                original_lookup[key].append((search_term, orig_keyword))
        
        # Claude Generated - Debug original catalog subjects
        if self.logger:
            total_orig_subjects = sum(len(term_results) for term_results in original_results.values())
            self.logger.info(f"Merge: {total_orig_subjects} original catalog subjects to match against")
        
        for swb_term, swb_term_results in swb_results.items():
            for swb_keyword, swb_data in swb_term_results.items():
                swb_gnd_ids = swb_data.get("gndid", set())
                
                if swb_gnd_ids:
                    swb_key = swb_keyword.lower()
                    term_key = swb_term.lower()
                    
                    # Find matches in original results
                    matches = []
                    if swb_key in original_lookup:
                        matches.extend(original_lookup[swb_key])
                    if term_key in original_lookup and term_key != swb_key:
                        matches.extend(original_lookup[term_key])
                    
                    if matches:
                        # Add SWB subject as new entry instead of merging with catalog subject
                        for search_term, orig_keyword in matches:
                            match_id = f"{search_term}:{swb_keyword}"
                            if match_id not in processed_matches:
                                processed_matches.add(match_id)
                                
                                # Add SWB subject as separate entry with its proper name
                                if swb_keyword not in original_results[search_term]:
                                    original_results[search_term][swb_keyword] = {
                                        "count": swb_data.get("count", 1),
                                        "gndid": swb_gnd_ids.copy(),
                                        "ddc": swb_data.get("ddc", set()),
                                        "dk": swb_data.get("dk", set())
                                    }
                                    swb_matches += len(swb_gnd_ids)
                                    if self.logger:
                                        self.logger.info(f"SWB add: '{swb_keyword}' (+{len(swb_gnd_ids)} GND-IDs) [matched via '{orig_keyword}']")
                                else:
                                    # Update existing SWB subject entry
                                    existing_data = original_results[search_term][swb_keyword]
                                    old_count = len(existing_data["gndid"])
                                    existing_data["gndid"].update(swb_gnd_ids)
                                    existing_data["ddc"].update(swb_data.get("ddc", set()))
                                    existing_data["dk"].update(swb_data.get("dk", set()))
                                    new_count = len(existing_data["gndid"])
                                    
                                    if new_count > old_count:
                                        added_gnd_ids = new_count - old_count
                                        swb_matches += added_gnd_ids
                                        if self.logger:
                                            self.logger.info(f"SWB update: '{swb_keyword}' (+{added_gnd_ids} GND-IDs)")
                                break  # Only process first match to avoid duplicates
                    else:
                        # No match found - add as completely new subject for the search term
                        # Find the most appropriate search term (the one being searched)
                        target_term = swb_term if swb_term in original_results else list(original_results.keys())[0]
                        if swb_keyword not in original_results[target_term]:
                            original_results[target_term][swb_keyword] = {
                                "count": swb_data.get("count", 1), 
                                "gndid": swb_gnd_ids.copy(),
                                "ddc": swb_data.get("ddc", set()),
                                "dk": swb_data.get("dk", set())
                            }
                            swb_matches += len(swb_gnd_ids)
                            if self.logger:
                                self.logger.info(f"SWB new: '{swb_keyword}' (+{len(swb_gnd_ids)} GND-IDs) [no catalog match]")
                        unmatched_swb_subjects.append(f"{swb_keyword} ({len(swb_gnd_ids)} GND-IDs) -> added as new")
        
        # Claude Generated - Debug unmatched subjects
        if self.logger and unmatched_swb_subjects:
            self.logger.warning(f"SWB: {len(unmatched_swb_subjects)} subjects couldn't be matched to catalog:")
            for i, unmatched in enumerate(unmatched_swb_subjects):  # Show all - Claude Generated
                self.logger.warning(f"  {i+1}. {unmatched}")
        
        if stream_callback:
            stream_callback(f"SWB-Validierung: {swb_matches} neue GND-IDs zugeordnet\n", "search")

    def execute_final_keyword_analysis(
        self,
        original_abstract: str,
        search_results: Dict[str, Dict[str, Any]],
        model: str = None,
        provider: str = None,
        task: str = "keywords",
        stream_callback: Optional[callable] = None,
        keyword_chunking_threshold: Optional[int] = None,  # None = auto-detect based on model
        chunking_task: str = "keywords_chunked",
        expand_synonyms: bool = False,
        mode=None,  # <--- NEUER PARAMETER: Pipeline mode for PromptService
        **kwargs,
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis]:
        """Execute final keyword analysis step with intelligent provider selection - Claude Generated"""

        # Intelligent provider selection using centralized method - Claude Generated
        provider, model = self._resolve_provider_smart(
            provider=provider,
            model=model,
            task_type="text",
            prefer_fast=False,  # Final analysis should prioritize quality
            task_name=task,
            step_id="keywords"
        )

        # Auto-resolve chunking threshold based on model capabilities (Issue #1 fix) - Claude Generated
        from .model_capabilities import get_chunking_threshold

        resolved_threshold = get_chunking_threshold(
            provider=provider,
            model=model,
            explicit_override=keyword_chunking_threshold,
            config_manager=self.config_manager  # Pass config_manager for per-model lookup - Claude Generated
        )

        if self.logger:
            if keyword_chunking_threshold is not None and keyword_chunking_threshold > 0:
                source = "explicit"
            elif self.config_manager:
                # Check if per-model config was used
                try:
                    cfg = self.config_manager.get_unified_config()
                    cfg_threshold = cfg.get_chunking_threshold(provider, model)
                    source = "per-model config" if cfg_threshold else "auto-detected"
                except Exception:
                    source = "auto-detected"
            else:
                source = "auto-detected"
            self.logger.info(f"💡 Keyword chunking threshold: {resolved_threshold} ({source} for {provider}:{model})")

        keyword_chunking_threshold = resolved_threshold

        # Prepare GND search results for prompt
        gnd_keywords_text = ""
        gnd_compliant_keywords = []
        seen_keywords = set()  # Track added keywords to prevent duplicates - Claude Generated

        # Claude Generated - Phase 2 optimization: Batch load all GND entries
        # Collect all unique GND-IDs first
        all_gnd_ids = set()
        for results in search_results.values():
            for keyword, data in results.items():
                gnd_ids = data.get("gndid", set())
                all_gnd_ids.update(gnd_ids)

        # Batch query: retrieve all GND entries in optimized batches instead of N individual queries
        if all_gnd_ids:
            if self.logger:
                self.logger.info(f"🚀 Batch loading {len(all_gnd_ids)} GND entries (chunks of 50)")
            gnd_entries_cache = self.cache_manager.get_gnd_facts_batch(list(all_gnd_ids))
            if self.logger:
                self.logger.info(f"✅ Retrieved {len(gnd_entries_cache)}/{len(all_gnd_ids)} entries from database")
        else:
            gnd_entries_cache = {}

        for results in search_results.values():
            for keyword, data in results.items():
                gnd_ids = data.get("gndid", set())

                # Handle keywords without GND-IDs (user-provided plain text) - Claude Generated
                if not gnd_ids:
                    # Add keyword as plain text without GND notation
                    formatted_keyword = keyword
                    # Check for duplicates before adding - Claude Generated
                    if formatted_keyword not in seen_keywords:
                        seen_keywords.add(formatted_keyword)
                        gnd_keywords_text += formatted_keyword + "\n"
                        gnd_compliant_keywords.append(formatted_keyword)
                    continue

                # Process keywords WITH GND-IDs
                for gnd_id in gnd_ids:
                    # Claude Generated - Lookup from in-memory cache (no database query - Phase 2 optimization)
                    gnd_entry = gnd_entries_cache.get(gnd_id)

                    if gnd_entry and gnd_entry.title:
                        gnd_title = gnd_entry.title

                        # DEFENSIVE: Explicit validation before split() to prevent accessing invalid pointers - Claude Generated
                        synonyms_list = []
                        if gnd_entry.synonyms:
                            try:
                                # Validate it's actually a string before calling split()
                                if isinstance(gnd_entry.synonyms, str) and len(gnd_entry.synonyms) > 0:
                                    synonyms_list = [s.strip() for s in gnd_entry.synonyms.split(';') if s.strip()]
                            except Exception as syn_error:
                                if self.logger:
                                    self.logger.debug(f"Failed to parse synonyms for {gnd_id}: {syn_error}")
                                synonyms_list = []

                        # Check if we should expand synonyms and if this title is relevant
                        if expand_synonyms:
                            # Check if this title already appears in our keyword list
                            title_in_keywords = any(gnd_title.lower() in kw.lower() for kw in [keyword] + list(results.keys()))

                            if title_in_keywords and synonyms_list:
                                # Format with synonyms: "Limnologie (Seenkunde; Süßwasserbiologie) (GND-ID: 4035769-7)"
                                synonym_text = "; ".join(synonyms_list)
                                formatted_keyword = f"{gnd_title} ({synonym_text}) (GND-ID: {gnd_id})"
                            else:
                                # No synonyms available or title not in keywords
                                formatted_keyword = f"{gnd_title} (GND-ID: {gnd_id})"
                        else:
                            # No synonym expansion
                            formatted_keyword = f"{gnd_title} (GND-ID: {gnd_id})"

                        # Check for duplicates before adding - Claude Generated
                        if formatted_keyword not in seen_keywords:
                            seen_keywords.add(formatted_keyword)
                            gnd_keywords_text += formatted_keyword + "\n"
                            gnd_compliant_keywords.append(formatted_keyword)
                    else:
                        # Fallback to original keyword if GND title not found
                        formatted_keyword = f"{keyword} (GND-ID: {gnd_id})"

                        # Check for duplicates before adding - Claude Generated
                        if formatted_keyword not in seen_keywords:
                            seen_keywords.add(formatted_keyword)
                            gnd_keywords_text += formatted_keyword + "\n"
                            gnd_compliant_keywords.append(formatted_keyword)

        # Check if chunking is needed based on keyword count
        total_keywords = len(gnd_compliant_keywords)

        if total_keywords > keyword_chunking_threshold:
            if stream_callback:
                stream_callback(
                    f"Zu viele Keywords ({total_keywords} > {keyword_chunking_threshold}). Verwende Chunking-Logik.\n",
                    kwargs.get("step_id", "keywords"),
                )
            return self._execute_chunked_keyword_analysis(
                original_abstract=original_abstract,
                gnd_compliant_keywords=gnd_compliant_keywords,
                model=model,
                provider=provider,
                task=task,
                chunking_task=chunking_task,
                stream_callback=stream_callback,
                mode=mode,
                **kwargs,
            )
        else:
            if stream_callback:
                stream_callback(
                    f"Keywords unter Schwellenwert ({total_keywords} <= {keyword_chunking_threshold}). Normale Verarbeitung.\n",
                    kwargs.get("step_id", "keywords"),
                )
            return self._execute_single_keyword_analysis(
                original_abstract=original_abstract,
                gnd_keywords_text=gnd_keywords_text,
                gnd_compliant_keywords=gnd_compliant_keywords,
                model=model,
                provider=provider,
                task=task,
                stream_callback=stream_callback,
                mode=mode,
                **kwargs,
            )

        # DEAD CODE REMOVED - This section was unreachable due to early returns above - Claude Generated

    def _execute_single_keyword_analysis(
        self,
        original_abstract: str,
        gnd_keywords_text: str,
        gnd_compliant_keywords: List[str],
        model: str,
        provider: str,
        task: str,
        stream_callback: Optional[callable] = None,
        mode=None,
        full_gnd_pool_for_verification: List[str] = None,  # Claude Generated - for chunk verification
        **kwargs,
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis]:
        """Execute single keyword analysis without chunking - Claude Generated"""

        # Use full pool for verification if provided, else use chunk pool - Claude Generated
        verification_pool = full_gnd_pool_for_verification if full_gnd_pool_for_verification is not None else gnd_compliant_keywords

        # Create abstract data with correct placeholder mapping
        abstract_data = AbstractData(
            abstract=original_abstract,  # This fills {abstract} placeholder
            keywords=gnd_keywords_text,  # This fills {keywords} placeholder
        )

        # Create stream callback adapter using centralized method - Claude Generated
        alima_stream_callback = self._create_stream_callback_adapter(
            stream_callback,
            kwargs.get("step_id", "keywords")
        )

        # Filter parameters using centralized method - Claude Generated
        alima_kwargs = self._filter_alima_kwargs(kwargs)

        # Execute final analysis
        task_state = self.alima_manager.analyze_abstract(
            abstract_data=abstract_data,
            task=task,
            model=model,
            provider=provider,
            stream_callback=alima_stream_callback,
            mode=mode,  # <--- NEUER PARAMETER: Pass mode to AlimaManager
            **alima_kwargs,
        )

        if task_state.status == "failed":
            raw_error = task_state.analysis_result.full_text
            # Stream a connection hint if it looks like a network error - Claude Generated
            if stream_callback and raw_error and any(kw in raw_error.lower() for kw in ("connect", "timeout", "network", "unreachable", "refused", "name or service")):
                stream_callback(f"\n🔌 Server nicht erreichbar – Verbindung prüfen ({provider})\n", kwargs.get("step_id", "keywords"))
            raise ValueError(f"Final keyword analysis failed: {raw_error}")

        # Extract final keywords and classes
        # FIXED: Use all keywords (including plain text) for DK search - Claude Generated
        _output_format = getattr(task_state.prompt_config, 'output_format', None) if task_state.prompt_config else None
        all_keywords_including_plain, gnd_validated_only = (
            extract_keywords_from_descriptive_text(
                task_state.analysis_result.full_text, gnd_compliant_keywords,
                output_format=_output_format
            )
        )

        # Apply deduplication to ensure no duplicate keywords - Claude Generated
        final_keywords = self._deduplicate_keywords(
            [all_keywords_including_plain],  # Use ALL keywords, not just GND-validated
            gnd_compliant_keywords
        )

        # Verify keywords against GND pool - Claude Generated
        # Use verification_pool (full pool for chunks, or chunk pool for non-chunked)
        verification_result = verify_keywords_against_gnd_pool(
            extracted_keywords=final_keywords,
            gnd_pool_keywords=verification_pool,  # Claude Generated - use full pool for chunk verification
            stream_callback=stream_callback,
            step_id=kwargs.get("step_id", "keywords"),
            knowledge_manager=self.cache_manager,  # Pass for DB fallback verification
        )
        final_keywords = verification_result["verified"]

        extracted_gnd_classes = extract_classes_from_descriptive_text(
            task_state.analysis_result.full_text, output_format=_output_format
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
            extracted_gnd_keywords=final_keywords,  # Store verified keywords only
            extracted_gnd_classes=extracted_gnd_classes,
            verification=verification_result,  # Store verification details - Claude Generated
        )

        return final_keywords, extracted_gnd_classes, llm_analysis

    def _execute_chunked_keyword_analysis(
        self,
        original_abstract: str,
        gnd_compliant_keywords: List[str],
        model: str,
        provider: str,
        task: str,
        chunking_task: str,
        stream_callback: Optional[callable] = None,
        mode=None,
        **kwargs,
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis]:
        """Execute keyword analysis with chunking for large keyword sets - Claude Generated"""

        # Store full GND pool for verification across all chunks - Claude Generated
        # This ensures keywords verified in one chunk are accepted in all chunks
        full_gnd_pool = gnd_compliant_keywords

        # Calculate optimal chunk size for equal distribution
        total_keywords = len(gnd_compliant_keywords)

        # Get threshold from kwargs (it's passed from execute_final_keyword_analysis)
        threshold = kwargs.get("keyword_chunking_threshold", 500)

        # Determine number of chunks needed
        if total_keywords <= threshold * 1.5:
            # For moderate oversize: use 2 chunks
            num_chunks = 2
        else:
            # For large oversize: calculate based on threshold
            num_chunks = max(2, (total_keywords + threshold - 1) // threshold)

        # Calculate equal chunk size
        chunk_size = total_keywords // num_chunks
        remainder = total_keywords % num_chunks

        # Create chunks with equal distribution
        chunks = []
        start_idx = 0
        for i in range(num_chunks):
            # Add one extra keyword to first 'remainder' chunks to distribute remainder
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            chunks.append(
                gnd_compliant_keywords[start_idx : start_idx + current_chunk_size]
            )
            start_idx += current_chunk_size

        if stream_callback:
            chunk_sizes = [len(chunk) for chunk in chunks]
            stream_callback(
                f"Teile {total_keywords} Keywords in {num_chunks} gleichmäßige Chunks auf: {chunk_sizes}\n",
                kwargs.get("step_id", "keywords"),
            )

        # Process each chunk
        all_chunk_results = []
        combined_responses = []

        for i, chunk in enumerate(chunks):
            if stream_callback:
                stream_callback(
                    f"\n--- Chunk {i+1}/{len(chunks)} ({len(chunk)} Keywords) ---\n",
                    kwargs.get("step_id", "keywords"),
                )

            # Create keywords text for this chunk
            chunk_keywords_text = "\n".join(chunk)

            # Execute keyword selection for this chunk
            chunk_result = self._execute_single_keyword_analysis(
                original_abstract=original_abstract,
                gnd_keywords_text=chunk_keywords_text,
                gnd_compliant_keywords=chunk,
                model=model,
                provider=provider,
                task=chunking_task,  # Use chunking task (e.g., "keywords_chunked" or "rephrase")
                stream_callback=stream_callback,
                mode=mode,
                full_gnd_pool_for_verification=full_gnd_pool,  # Claude Generated - pass full pool
                **kwargs,
            )

            # Extract keywords with enhanced recognition
            chunk_keywords = self._extract_keywords_enhanced(
                chunk_result[2].response_full_text,
                chunk,
                stream_callback,
                chunk_id=f"Chunk {i+1}",
            )

            all_chunk_results.append(
                (chunk_keywords, chunk_result[1], chunk_result[2])
            )  # Use enhanced keywords
            combined_responses.append(
                chunk_result[2].response_full_text
            )  # LlmKeywordAnalysis.response_full_text

        # Deduplicate keywords from all chunks
        deduplicated_keywords = self._deduplicate_keywords(
            [
                result[0] for result in all_chunk_results
            ],  # extracted_keywords_exact from each chunk
            gnd_compliant_keywords,
        )

        # Combine GND classes from all chunks (simple concatenation, no deduplication needed)
        all_gnd_classes = []
        for result in all_chunk_results:
            all_gnd_classes.extend(result[1])  # extracted_gnd_classes

        if stream_callback:
            stream_callback(
                f"\n--- Deduplizierung abgeschlossen ---\n", kwargs.get("step_id", "keywords")
            )
            total_chunk_keywords = sum(len(r[0]) for r in all_chunk_results)
            stream_callback(
                f"Deduplizierte Keywords: {len(deduplicated_keywords)} aus {total_chunk_keywords} chunk-results\n",
                kwargs.get("step_id", "keywords"),
            )

            # Show current deduplicated list for debugging - Claude Generated
            if deduplicated_keywords:
                # Show all deduplicated keywords - Claude Generated
                preview_text = ", ".join(
                    [kw.split(" (GND-ID:")[0] for kw in deduplicated_keywords]
                )
                stream_callback(
                    f"Deduplizierte Liste: {preview_text}\n",
                    kwargs.get("step_id", "keywords"),
                )

        # Execute final keyword analysis with deduplicated results - Claude Generated
        # IMPORTANT: Uses 'task' (normal keywords prompt) NOT 'chunking_task' for proper Sacherschließung
        final_keywords_text = "\n".join(deduplicated_keywords)
        final_single_result = self._execute_single_keyword_analysis(
            original_abstract=original_abstract,
            gnd_keywords_text=final_keywords_text,
            gnd_compliant_keywords=deduplicated_keywords,
            model=model,
            provider=provider,
            task=task,  # Use normal keywords task, NOT chunking_task!
            stream_callback=stream_callback,
            mode=mode,
            full_gnd_pool_for_verification=full_gnd_pool,  # Claude Generated - pass full pool
            **kwargs,
        )

        # Use already verified keywords from _execute_single_keyword_analysis() - Claude Generated
        # No need to re-extract or re-verify since _execute_single_keyword_analysis() already does both
        final_keywords = final_single_result[0]  # Already verified keywords

        # Fallback: Konsolidierungs-Call abgebrochen oder leer, aber deduplizierte
        # Keywords aus Chunks sind valide → direkt verwenden.  - Claude Generated
        if not final_keywords and deduplicated_keywords:
            self.logger.info(
                f"⚠️ Consolidation returned empty – using {len(deduplicated_keywords)} "
                f"deduplicated chunk keywords directly"
            )
            final_keywords = deduplicated_keywords

        # Update the LlmKeywordAnalysis to include chunk information
        final_llm_analysis = LlmKeywordAnalysis(
            task_name=f"{task} (chunked)",
            model_used=model,
            provider_used=provider,
            prompt_template=final_single_result[2].prompt_template,
            filled_prompt=final_single_result[2].filled_prompt,
            temperature=kwargs.get("temperature", 0.7),
            seed=kwargs.get("seed", 0),
            response_full_text=final_single_result[2].response_full_text,  # Only final consolidation response
            extracted_gnd_keywords=final_keywords,  # Use verified keywords only - Claude Generated
            extracted_gnd_classes=final_single_result[1],
            chunk_responses=combined_responses,  # Store chunk responses separately - Claude Generated
            verification=final_single_result[2].verification,  # Use verification from _execute_single_keyword_analysis() - Claude Generated
        )

        return final_keywords, final_single_result[1], final_llm_analysis

    def _deduplicate_keywords(
        self, keyword_lists: List[List[str]], reference_keywords: List[str]
    ) -> List[str]:
        """Deduplicate keywords based on exact word or GND-ID matching - Claude Generated"""

        # Parse reference keywords to create lookup dictionaries
        word_to_gnd = {}  # word -> gnd_id
        gnd_to_word = {}  # gnd_id -> word

        for keyword in reference_keywords:
            # Parse format: "Keyword (GND-ID: 123456789)"
            match = re.match(r"^(.+?)\s*\(GND-ID:\s*([^)]+)\)$", keyword.strip())
            if match:
                word = match.group(1).strip()
                gnd_id = match.group(2).strip()
                word_to_gnd[word.lower()] = gnd_id
                gnd_to_word[gnd_id] = keyword  # Store full formatted keyword

        # Collect unique keywords
        seen_words = set()
        seen_gnd_ids = set()
        deduplicated = []

        for keyword_list in keyword_lists:
            for keyword in keyword_list:
                # Parse the keyword
                match = re.match(r"^(.+?)\s*\(GND-ID:\s*([^)]+)\)$", keyword.strip())
                if match:
                    word = match.group(1).strip()
                    gnd_id = match.group(2).strip()

                    # Deduplicate by GND-ID and by text - Claude Generated
                    # Same text with different GND-IDs: prefer pool version (authoritative)
                    word_lower = word.lower()

                    if gnd_id not in seen_gnd_ids:
                        if word_lower in seen_words:
                            # Same text, different GND-ID → prefer pool version - Claude Generated
                            if word_lower in word_to_gnd:
                                pool_gnd_id = word_to_gnd[word_lower]
                                if pool_gnd_id != gnd_id and self.logger:
                                    self.logger.warning(
                                        f"⚠️ Konflikt: '{word}' hat GND-ID {gnd_id}, "
                                        f"Pool hat {pool_gnd_id} - verwende Pool-Version"
                                    )
                                # Skip this duplicate text entry
                                continue
                            elif self.logger:
                                self.logger.debug(f"⚠️  Multiple GND-IDs for similar term: '{word}' ({gnd_id})")
                                continue  # Skip duplicate text with different GND-ID

                        seen_words.add(word_lower)
                        seen_gnd_ids.add(gnd_id)
                        deduplicated.append(keyword)
                else:
                    # Fallback for keywords without proper format
                    word_lower = keyword.strip().lower()
                    if word_lower not in seen_words:
                        seen_words.add(word_lower)
                        deduplicated.append(keyword)

        return deduplicated

    def _lookup_gnd_id_from_db(self, keyword_normalized: str) -> Optional[str]:
        """Lookup GND-ID for a keyword from database - Claude Generated

        Args:
            keyword_normalized: Normalized keyword text (lowercase, whitespace normalized)

        Returns:
            GND-ID string if found in database, None otherwise
        """
        if not self.cache_manager:
            return None

        try:
            # Try exact title match first
            from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
            ukm = self.cache_manager

            # Search by keyword/title
            # The UnifiedKnowledgeManager has a search_by_keywords method
            results = ukm.search_by_keywords([keyword_normalized], fuzzy_threshold=90)

            if results and len(results) > 0:
                # Get first result's GND-ID
                first_result = results[0]
                if isinstance(first_result, dict) and 'gnd_id' in first_result:
                    gnd_id = first_result['gnd_id']
                    return gnd_id

            return None

        except Exception as e:
            if self.logger:
                self.logger.debug(f"DB lookup error for '{keyword_normalized}': {e}")
            return None

    def _extract_keywords_enhanced(
        self,
        response_text: str,
        reference_keywords: List[str],
        stream_callback: Optional[callable] = None,
        chunk_id: str = "",
    ) -> List[str]:
        """Enhanced keyword extraction with exact string and GND-ID matching - Claude Generated"""

        if not response_text or not reference_keywords:
            return []

        # Parse reference keywords to create lookup dictionaries - Claude Generated
        # Use GND-ID as primary key (no overwriting), store multiple keywords per word
        word_to_full = {}  # clean_word_lower -> List[full_formatted_keywords]
        gnd_to_full = {}  # gnd_id -> full_formatted_keyword (unique by design)

        for keyword in reference_keywords:
            # Parse format: "Keyword (GND-ID: 123456789)"
            match = re.match(r"^(.+?)\s*\(GND-ID:\s*([^)]+)\)$", keyword.strip())
            if match:
                word = match.group(1).strip()
                gnd_id = match.group(2).strip()

                # Primary: Store by GND-ID (guaranteed unique, no overwriting)
                gnd_to_full[gnd_id] = keyword

                # Secondary: Store by word (as list to avoid overwriting)
                word_lower = word.lower()
                if word_lower not in word_to_full:
                    word_to_full[word_lower] = []
                word_to_full[word_lower].append(keyword)

        # Search for matches in response text
        found_keywords = []
        response_lower = response_text.lower()

        # Method 1: Search for exact keyword strings (now handling lists) - Claude Generated
        for clean_word, full_keywords_list in word_to_full.items():
            if clean_word in response_lower:
                # Iterate over list of keywords with same word
                for full_keyword in full_keywords_list:
                    if full_keyword not in found_keywords:
                        found_keywords.append(full_keyword)

        # Method 2: Search for GND-IDs
        for gnd_id, full_keyword in gnd_to_full.items():
            if gnd_id in response_text:  # GND-IDs are case-sensitive
                if full_keyword not in found_keywords:
                    found_keywords.append(full_keyword)

        # Debug output
        if stream_callback:
            stream_callback(
                f"{chunk_id} Keywords gefunden: {len(found_keywords)} aus {len(reference_keywords)} verfügbaren\n",
                "keywords",
            )
            if found_keywords:
                # Show all found keywords for debugging - Claude Generated
                preview_text = ", ".join(
                    [kw.split(" (GND-ID:")[0] for kw in found_keywords]
                )
                stream_callback(
                    f"{chunk_id} Aktuelle Liste: {preview_text}\n", "keywords"
                )

        return found_keywords

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

    def execute_dk_classification(
        self,
        original_abstract: str,
        dk_search_results: List[Dict[str, Any]],
        model: str = None,
        provider: str = None,
        stream_callback: Optional[callable] = None,
        dk_frequency_threshold: int = DEFAULT_DK_FREQUENCY_THRESHOLD,  # Claude Generated - Only pass classifications with >= N occurrences
        mode=None,  # <--- NEUER PARAMETER: Pipeline mode for PromptService
        **kwargs,
    ) -> Tuple[List[str], Optional["LlmKeywordAnalysis"]]:
        """
        Execute LLM-based DK classification using pre-fetched catalog search results with intelligent provider selection - Claude Generated

        Args:
            original_abstract: The original abstract text for analysis
            dk_search_results: List of DK classification results from catalog search
            model: LLM model to use for classification (optional - SmartProvider selection if None)
            provider: LLM provider (optional - SmartProvider selection if None)
            stream_callback: Optional callback for streaming progress updates
            dk_frequency_threshold: Minimum occurrence count for DK classifications to be included.
                                  Only classifications that appear >= this many times in the catalog
                                  will be passed to the LLM for analysis. Default: 10.
                                  This reduces prompt size and focuses on most relevant classifications.
            **kwargs: Additional parameters for LLM (temperature, top_p, etc.)

        Returns:
            Tuple containing:
            - List of selected DK classification codes
            - LlmKeywordAnalysis object with details of the LLM call

        Note:
            The frequency threshold helps manage large result sets by filtering out
            classifications that occur infrequently in the catalog, which are typically
            less relevant for the given abstract.
        """

        # Intelligent provider selection using centralized method - Claude Generated
        provider, model = self._resolve_provider_smart(
            provider=provider,
            model=model,
            task_type="classification",
            prefer_fast=False,  # Classification should prioritize accuracy
            task_name="classification",
            step_id="dk_classification"
        )

        if not dk_search_results:
            if stream_callback:
                stream_callback("Keine DK-Suchergebnisse vorhanden - DK-Klassifikation übersprungen\n", "dk_classification")
            return [], None

        if stream_callback:
            stream_callback(f"Starte DK-Klassifikation mit {len(dk_search_results)} Katalog-Einträgen\n", "dk_classification")

        # Filter results by frequency threshold - Claude Generated
        filtered_results = []
        low_frequency_count = 0
        
        for result in dk_search_results:
            classification_type = str(result.get("classification_type", result.get("type", "DK"))).upper()
            if classification_type == "RVK":
                filtered_results.append(result)
                continue

            # Check if result has frequency information and meets threshold
            if "count" in result:
                count = result.get("count", 0)
                if count >= dk_frequency_threshold:
                    filtered_results.append(result)
                else:
                    low_frequency_count += 1
            else:
                # Include results without count information (legacy format)
                filtered_results.append(result)
        
        if stream_callback:
            if low_frequency_count > 0:
                stream_callback(f"Filtere DK-Ergebnisse: {len(filtered_results)} Einträge mit ≥{dk_frequency_threshold} Vorkommen, {low_frequency_count} mit niedrigerer Häufigkeit ausgeschlossen\n", "dk_classification")
            else:
                stream_callback(f"Verwende alle {len(filtered_results)} DK-Einträge (keine Häufigkeits-Filterung nötig)\n", "dk_classification")

        # Filter out results without titles - Claude Generated
        results_with_titles = []
        titleless_count = 0

        for result in filtered_results:
            if result.get("source") in {"rvk_api", "rvk_gnd_index"}:
                if result.get("label") or result.get("ancestor_path"):
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
                continue

            # Check if result has titles (aggregated format)
            if "titles" in result:
                if result.get("titles") and any(t.strip() for t in result.get("titles", [])):
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
            # Check if result has source_title (individual format)
            elif "source_title" in result:
                if result.get("source_title", "").strip():
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
            # Check if result has title (legacy format)
            elif "title" in result:
                if result.get("title", "").strip():
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
            else:
                # No title field found - skip this result
                titleless_count += 1

        if stream_callback:
            if titleless_count > 0:
                stream_callback(f"⚠️ Filtere titel-lose Einträge: {len(results_with_titles)} mit Titeln, {titleless_count} ohne Titel ausgeschlossen\n", "dk_classification")
            else:
                stream_callback(f"✅ Alle {len(results_with_titles)} Einträge haben Titel\n", "dk_classification")

        if self.logger:
            self.logger.info(f"DK title filter: {len(results_with_titles)} with titles, {titleless_count} without titles excluded")

        allowed_standard_rvk_map = {}
        allowed_nonstandard_rvk_map = {}
        rvk_source_map = {}
        for result in results_with_titles:
            if str(result.get("classification_type", result.get("type", "DK"))).upper() != "RVK":
                continue

            raw_code = result.get("dk", "")
            if not raw_code:
                continue

            normalized = canonicalize_rvk_notation(raw_code)
            if not normalized:
                continue
            rvk_status = result.get("rvk_validation_status")
            rvk_source_map[normalized] = {
                "source": result.get("source", "catalog"),
                "status": rvk_status or "standard",
            }
            if rvk_status in {"non_standard", "validation_error"}:
                allowed_nonstandard_rvk_map[normalized] = f"RVK {normalized}"
            else:
                allowed_standard_rvk_map[normalized] = f"RVK {normalized}"

        # Format catalog results for LLM prompt with aggregated data - Claude Generated
        catalog_text = PipelineResultFormatter.format_dk_results_for_prompt(results_with_titles)

        if allowed_standard_rvk_map or allowed_nonstandard_rvk_map:
            rvk_guardrail = (
                "WICHTIG FUER RVK:\n"
                "- Erfinde niemals neue RVK-Notationen.\n"
            )
            if allowed_standard_rvk_map:
                rvk_guardrail += (
                    "- Verwende RVK nur aus den unten gelisteten standardisierten Kandidaten.\n"
                    "- Wenn keine standardisierte RVK thematisch passt, gib keine RVK aus.\n"
                )
            else:
                rvk_guardrail += (
                    "- Es liegen keine standardisierten RVK aus dem Katalog vor.\n"
                    "- Bevorzuge standardisierte RVK aus dem RVK-API-Fallback; nur wenn keine solche passt, darf eine explizit als nicht-standardisiert/lokal markierte RVK verwendet werden.\n"
                )
            rvk_guardrail += (
                "- Achte auf den Fachpfad und verwerfe Kandidaten mit unpassendem Oberbereich.\n\n"
            )
            catalog_text = rvk_guardrail + catalog_text

        # Create AbstractData for LLM call
        from ..core.data_models import AbstractData
        abstract_data = AbstractData(
            abstract=original_abstract,
            keywords=catalog_text  # Use catalog results as "keywords" for dk_class prompt
        )

        if stream_callback:
            stream_callback("Starte LLM-basierte DK-Klassifikation...\n", "dk_classification")

        # Create stream callback adapter using centralized method - Claude Generated
        alima_stream_callback = self._create_stream_callback_adapter(
            stream_callback,
            "dk_classification"
        )

        # Filter parameters using centralized method - Claude Generated
        alima_kwargs = self._filter_alima_kwargs(kwargs, exclude_llm_params=True)

        # Execute LLM classification
        try:
            task_state = self.alima_manager.analyze_abstract(
                abstract_data=abstract_data,
                task="dk_classification",
                model=model,
                provider=provider,
                stream_callback=alima_stream_callback,
                mode=mode,  # <--- NEUER PARAMETER: Pass mode to AlimaManager
                **alima_kwargs,
            )

            if task_state.status == "failed":
                if stream_callback:
                    stream_callback(f"LLM-Klassifikation fehlgeschlagen: {task_state.analysis_result.full_text}\n", "dk_classification")
                return [], None

            # Extract DK classifications from LLM response
            response_text = task_state.analysis_result.full_text
            # Pass output_format from prompt_config for JSON extraction - Claude Generated
            _output_format = getattr(task_state.prompt_config, 'output_format', None) if task_state.prompt_config else None
            llm_classifications = self._extract_dk_from_response(response_text, output_format=_output_format)
            llm_classifications = self._filter_final_rvk_classifications(
                llm_classifications,
                allowed_standard_rvk_map,
                allowed_nonstandard_rvk_map,
                stream_callback=stream_callback,
            )
            deterministic_rvk = self._select_final_rvk_candidates(
                results_with_titles,
                original_abstract,
            )
            llm_dk_only = [
                code for code in llm_classifications
                if not str(code or "").strip().upper().startswith("RVK ")
            ]
            llm_rvk_only = [
                code for code in llm_classifications
                if str(code or "").strip().upper().startswith("RVK ")
            ]

            max_total_classifications = 10
            max_dk_count = max(0, max_total_classifications - len(deterministic_rvk))
            dk_classifications = list(
                dict.fromkeys(llm_dk_only[:max_dk_count] + deterministic_rvk)
            )

            if stream_callback:
                if deterministic_rvk:
                    stream_callback(
                        (
                            "ℹ️ Finale RVK-Auswahl erfolgt deterministisch aus validierten Kandidaten: "
                            f"{', '.join(deterministic_rvk)}\n"
                        ),
                        "dk_classification",
                    )
                elif llm_rvk_only:
                    stream_callback(
                        "ℹ️ LLM-RVK-Auswahl wurde verworfen; es blieb kein deterministisch validierter RVK-Kandidat uebrig\n",
                        "dk_classification",
                    )

            # Extract analysis text from LLM response - Claude Generated
            from ..core.processing_utils import extract_analyse_text_from_response
            analyse_text = extract_analyse_text_from_response(response_text, output_format=_output_format)

            # Construct LlmKeywordAnalysis object for history/display - Claude Generated
            from ..core.data_models import LlmKeywordAnalysis
            llm_analysis = LlmKeywordAnalysis(
                task_name="dk_classification",
                model_used=task_state.model_used or model or "unknown",
                provider_used=task_state.provider_used or provider or "unknown",
                prompt_template=task_state.prompt_config.prompt if task_state.prompt_config else "",
                filled_prompt="", # We don't store the filled prompt to save space
                temperature=task_state.prompt_config.temp if task_state.prompt_config else 0.7,
                seed=task_state.prompt_config.seed if task_state.prompt_config else 0,
                response_full_text=response_text,
                extracted_gnd_classes=dk_classifications,
                analyse_text=analyse_text  # Analysis text from thought block or JSON - Claude Generated
            )

            final_rvk_sources = {
                "catalog_standard": 0,
                "catalog_nonstandard": 0,
                "rvk_gnd_index": 0,
                "rvk_api": 0,
            }
            if stream_callback:
                stream_callback(
                    f"DK-Klassifikation abgeschlossen: {len(dk_classifications)} Klassifikationscodes extrahiert\n",
                    "dk_classification",
                )
                for code in dk_classifications:
                    clean = str(code or "").strip()
                    if not clean.upper().startswith("RVK "):
                        continue
                    normalized = canonicalize_rvk_notation(clean[4:].strip())
                    source_info = rvk_source_map.get(normalized, {})
                    source = source_info.get("source", "catalog")
                    status = source_info.get("status", "standard")
                    if source == "rvk_gnd_index":
                        final_rvk_sources["rvk_gnd_index"] += 1
                    elif source == "rvk_api":
                        final_rvk_sources["rvk_api"] += 1
                    elif status in {"non_standard", "validation_error"}:
                        final_rvk_sources["catalog_nonstandard"] += 1
                    else:
                        final_rvk_sources["catalog_standard"] += 1

                source_parts = []
                if final_rvk_sources["catalog_standard"]:
                    source_parts.append(f"Katalog standard {final_rvk_sources['catalog_standard']}")
                if final_rvk_sources["catalog_nonstandard"]:
                    source_parts.append(f"Katalog lokal {final_rvk_sources['catalog_nonstandard']}")
                if final_rvk_sources["rvk_gnd_index"]:
                    source_parts.append(f"RVK-GND-Index {final_rvk_sources['rvk_gnd_index']}")
                if final_rvk_sources["rvk_api"]:
                    source_parts.append(f"RVK-API-Label {final_rvk_sources['rvk_api']}")
                if source_parts:
                    stream_callback(
                        f"ℹ️ Finale RVK-Quellen: {', '.join(source_parts)}\n",
                        "dk_classification",
                    )
            else:
                for code in dk_classifications:
                    clean = str(code or "").strip()
                    if not clean.upper().startswith("RVK "):
                        continue
                    normalized = canonicalize_rvk_notation(clean[4:].strip())
                    source_info = rvk_source_map.get(normalized, {})
                    source = source_info.get("source", "catalog")
                    status = source_info.get("status", "standard")
                    if source == "rvk_gnd_index":
                        final_rvk_sources["rvk_gnd_index"] += 1
                    elif source == "rvk_api":
                        final_rvk_sources["rvk_api"] += 1
                    elif status in {"non_standard", "validation_error"}:
                        final_rvk_sources["catalog_nonstandard"] += 1
                    else:
                        final_rvk_sources["catalog_standard"] += 1

            if llm_analysis is not None:
                setattr(llm_analysis, "rvk_provenance", final_rvk_sources)
            setattr(task_state, "rvk_provenance", final_rvk_sources)

            return dk_classifications, llm_analysis

        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM DK classification failed: {e}")
            if stream_callback:
                stream_callback(f"LLM-Klassifikation-Fehler: {str(e)}\n", "dk_classification")
            return [], None

    def _build_rvk_api_fallback_results(
        self,
        keywords: List[str],
        stream_callback: Optional[callable] = None,
        max_results_per_keyword: int = 6,
    ) -> List[Dict[str, Any]]:
        """Build keyword-centric RVK candidates from the official RVK API."""
        from .clients.rvk_api_client import RvkApiClient

        rvk_client = RvkApiClient()
        keyword_results = []

        for keyword in keywords:
            clean_keyword = canonicalize_keyword(keyword)
            try:
                candidates = rvk_client.search_keyword(clean_keyword, max_results=max_results_per_keyword)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"RVK API fallback failed for '{clean_keyword}': {exc}")
                if stream_callback:
                    stream_callback(f"  ⚠️ RVK API '{clean_keyword}': Fehler - {str(exc)}\n", "dk_search")
                continue

            if not candidates:
                continue

            classifications = []
            for candidate in candidates:
                classifications.append({
                    "dk": candidate["notation"],
                    "type": "RVK",
                    "classification_type": "RVK",
                    "count": 1,
                    "titles": [],
                    "matched_keywords": [clean_keyword],
                    "source": "rvk_api",
                    "label": candidate["label"],
                    "ancestor_path": candidate["ancestor_path"],
                    "register": candidate["register"],
                    "score": candidate["score"],
                    "branch_family": candidate["branch_family"],
                    "rvk_validation_status": "standard",
                    "validation_message": "",
                })

            keyword_results.append({
                "keyword": clean_keyword,
                "source": "rvk_api",
                "search_time_ms": 0.0,
                "classifications": classifications,
            })

            if stream_callback:
                stream_callback(
                    f"  ✅ RVK API {clean_keyword}: {len(classifications)} authority-backed Kandidaten\n",
                    "dk_search"
                )

        return keyword_results

    def _build_rvk_gnd_index_results(
        self,
        keyword_entries: List[Dict[str, str]],
        stream_callback: Optional[callable] = None,
        max_results_per_keyword: int = 6,
    ) -> List[Dict[str, Any]]:
        """Build RVK candidates from the official RVK MarcXML dump via GND links."""
        from .clients.rvk_marc_index import RvkMarcIndex

        index = RvkMarcIndex()
        if stream_callback:
            stream_callback("🔎 Nutze RVK MarcXML-GND-Index fuer standardisierte RVK-Kandidaten...\n", "dk_search")

        results = index.lookup_by_gnd_keywords(
            keyword_entries,
            max_results_per_keyword=max_results_per_keyword,
            progress_callback=(lambda msg: stream_callback(msg, "dk_search")) if stream_callback else None,
        )

        if stream_callback and results:
            total = sum(len(item.get("classifications", [])) for item in results)
            stream_callback(
                f"  ✅ RVK-GND-Index: {total} standardisierte Kandidaten fuer {len(results)} Keywords\n",
                "dk_search",
            )

        return results

    def _validate_catalog_rvk_candidates(
        self,
        keyword_results: List[Dict[str, Any]],
        stream_callback: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Validate catalog-derived RVK candidates and drop only obvious artifacts."""
        from .clients.rvk_api_client import RvkApiClient

        validation_cache: Dict[str, Dict[str, Any]] = {}
        cleaned_results: List[Dict[str, Any]] = []
        standard_count = 0
        nonstandard_count = 0
        artifact_count = 0
        validation_error_count = 0
        max_validation_candidates = 120
        unique_plausible_codes = []
        seen_plausible_codes = set()
        evidence_by_code: Dict[str, Dict[str, Any]] = {}

        for kw_result in keyword_results:
            keyword = kw_result.get("keyword", "")
            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", "DK"))).upper()
                if cls_type != "RVK":
                    continue

                normalized = canonicalize_rvk_notation(classification.get("dk", ""))
                if not normalized:
                    continue
                if not is_plausible_nonstandard_rvk(normalized):
                    validation_cache[normalized] = {
                        "status": "artifact",
                        "notation": normalized,
                        "message": "Implausible RVK notation pattern",
                    }
                    continue
                evidence = evidence_by_code.setdefault(
                    normalized,
                    {
                        "count": 0,
                        "keyword_hits": set(),
                        "title_hits": 0,
                    },
                )
                evidence["count"] += int(classification.get("count", 0) or 0)
                if keyword:
                    evidence["keyword_hits"].add(keyword)
                evidence["title_hits"] += len(classification.get("titles", []) or [])
                if normalized not in seen_plausible_codes:
                    seen_plausible_codes.add(normalized)
                    unique_plausible_codes.append(normalized)

        def _evidence_score(code: str) -> int:
            evidence = evidence_by_code.get(code, {})
            return (
                int(evidence.get("count", 0)) * 4
                + len(evidence.get("keyword_hits", set())) * 3
                + min(int(evidence.get("title_hits", 0)), 10)
            )

        unique_plausible_codes.sort(key=lambda code: (-_evidence_score(code), code))
        selected_codes = unique_plausible_codes[:max_validation_candidates]

        if selected_codes and stream_callback:
            stream_callback(
                (
                    f"🔎 Prüfe die {len(selected_codes)} stärksten von "
                    f"{len(unique_plausible_codes)} eindeutigen RVK-Kandidaten gegen die RVK-API...\n"
                    if len(selected_codes) < len(unique_plausible_codes)
                    else f"🔎 Prüfe {len(selected_codes)} eindeutige RVK-Kandidaten gegen die RVK-API...\n"
                ),
                "dk_search",
            )

        def _validate_code(code: str) -> Tuple[str, Dict[str, Any]]:
            client = RvkApiClient(timeout=4)
            return code, client.validate_notation(code)

        if selected_codes:
            max_workers = min(8, len(selected_codes))
            progress_step = max(10, len(selected_codes) // 5)
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_validate_code, code): code
                    for code in selected_codes
                }
                for future in as_completed(future_map):
                    code = future_map[future]
                    try:
                        normalized_code, validation = future.result()
                    except Exception as exc:
                        normalized_code = code
                        validation = {
                            "status": "validation_error",
                            "notation": code,
                            "label": "",
                            "register": [],
                            "ancestor_path": "",
                            "branch_family": "",
                            "message": str(exc),
                        }

                    validation_cache[normalized_code] = validation
                    completed += 1
                    if stream_callback and (completed == len(selected_codes) or completed % progress_step == 0):
                        stream_callback(
                            f"  ↳ RVK-API-Validierung: {completed}/{len(selected_codes)}\n",
                            "dk_search",
                        )

        for kw_result in keyword_results:
            cleaned_classifications = []

            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", "DK"))).upper()
                if cls_type != "RVK":
                    cleaned_classifications.append(classification)
                    continue

                normalized = canonicalize_rvk_notation(classification.get("dk", ""))
                if not normalized:
                    artifact_count += 1
                    continue

                validation = validation_cache.get(normalized)
                if validation is None:
                    validation = {
                        "status": "artifact",
                        "notation": normalized,
                        "message": "Niedrige Prioritaet - nicht gegen die RVK-API geprüft",
                    }
                    validation_cache[normalized] = validation

                status = validation.get("status", "")
                if status == "standard":
                    normalized = validation.get("notation") or normalized
                    updated = dict(classification)
                    updated["dk"] = normalized
                    updated["rvk_validation_status"] = "standard"
                    updated["validation_message"] = ""
                    if validation.get("label"):
                        updated["label"] = validation["label"]
                    if validation.get("ancestor_path"):
                        updated["ancestor_path"] = validation["ancestor_path"]
                    if validation.get("register"):
                        updated["register"] = validation["register"]
                    if validation.get("branch_family"):
                        updated["branch_family"] = validation["branch_family"]
                    cleaned_classifications.append(updated)
                    standard_count += 1
                    continue

                if status == "validation_error":
                    updated = dict(classification)
                    updated["dk"] = normalized
                    updated["rvk_validation_status"] = "validation_error"
                    updated["validation_message"] = validation.get("message", "")
                    cleaned_classifications.append(updated)
                    validation_error_count += 1
                    continue

                if is_plausible_nonstandard_rvk(normalized):
                    updated = dict(classification)
                    updated["dk"] = normalized
                    updated["rvk_validation_status"] = "non_standard"
                    updated["validation_message"] = validation.get("message", "Notation Not Found")
                    cleaned_classifications.append(updated)
                    nonstandard_count += 1
                    continue

                artifact_count += 1

            if cleaned_classifications:
                updated_result = dict(kw_result)
                updated_result["classifications"] = cleaned_classifications
                cleaned_results.append(updated_result)

        if stream_callback and (standard_count or nonstandard_count or artifact_count or validation_error_count):
            parts = []
            if standard_count:
                parts.append(f"{standard_count} standard")
            if nonstandard_count:
                parts.append(f"{nonstandard_count} nicht-standardisiert/lokal")
            if validation_error_count:
                parts.append(f"{validation_error_count} ungeprüft (API-Fehler)")
            if artifact_count:
                parts.append(f"{artifact_count} Artefakte verworfen")
            stream_callback(
                f"🔎 RVK-Prüfung Katalog: {', '.join(parts)}\n",
                "dk_search",
            )

        return cleaned_results

    def _inject_rvk_api_fallback(
        self,
        final_search_keywords: List[str],
        dk_search_results: List[Dict[str, Any]],
        gnd_keyword_entries: Optional[List[Dict[str, str]]] = None,
        stream_callback: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Add authority-backed RVK candidates if no standard RVK survived validation."""
        existing_standard_rvk = False
        for kw_result in dk_search_results:
            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", ""))).upper()
                rvk_status = classification.get("rvk_validation_status")
                if cls_type == "RVK" and rvk_status == "standard":
                    existing_standard_rvk = True
                    break
            if existing_standard_rvk:
                break

        if existing_standard_rvk:
            return dk_search_results

        if gnd_keyword_entries:
            if stream_callback:
                stream_callback(
                    "⚠️ Katalog lieferte keine standardisierten RVK-Kandidaten - versuche RVK-GND-Index\n",
                    "dk_search"
                )

            rvk_index_results = self._build_rvk_gnd_index_results(
                gnd_keyword_entries,
                stream_callback=stream_callback,
            )
            if rvk_index_results:
                if self.logger:
                    self.logger.info(
                        f"RVK GND index added {sum(len(item.get('classifications', [])) for item in rvk_index_results)} "
                        f"candidates across {len(rvk_index_results)} keywords"
                    )
                dk_search_results = dk_search_results + rvk_index_results
                return dk_search_results

        if stream_callback:
            stream_callback(
                "⚠️ RVK-GND-Index lieferte nichts - nutze offiziellen RVK-API-Label-Fallback\n",
                "dk_search"
            )

        rvk_api_results = self._build_rvk_api_fallback_results(
            final_search_keywords,
            stream_callback=stream_callback,
        )
        if not rvk_api_results:
            if stream_callback:
                stream_callback("  ⚠️ RVK-API-Fallback lieferte keine geeigneten Kandidaten\n", "dk_search")
            return dk_search_results

        if self.logger:
            self.logger.info(
                f"RVK API fallback added {sum(len(item.get('classifications', [])) for item in rvk_api_results)} "
                f"candidates across {len(rvk_api_results)} keywords"
            )
        return dk_search_results + rvk_api_results

    def _filter_final_rvk_classifications(
        self,
        classifications: List[str],
        allowed_standard_rvk_map: Dict[str, str],
        allowed_nonstandard_rvk_map: Dict[str, str],
        stream_callback: Optional[callable] = None,
    ) -> List[str]:
        """Prevent free-form RVK inference while allowing local RVK only as a fallback."""
        filtered = []
        dropped = []
        allowed_map = allowed_standard_rvk_map or allowed_nonstandard_rvk_map

        for code in classifications:
            clean = str(code or "").strip()
            if not clean:
                continue

            if clean.upper().startswith("RVK "):
                normalized = canonicalize_rvk_notation(clean[4:].strip())
                canonical = allowed_map.get(normalized)
                if canonical:
                    filtered.append(canonical)
                else:
                    dropped.append(clean)
                continue

            filtered.append(clean)

        deduplicated = list(dict.fromkeys(filtered))

        if dropped:
            if self.logger:
                self.logger.warning(f"Dropped {len(dropped)} non-authoritative RVK classifications: {dropped}")
            if stream_callback:
                preview = ", ".join(dropped[:4])
                if len(dropped) > 4:
                    preview += f", +{len(dropped) - 4} weitere"
                stream_callback(
                    f"⚠️ Verwerfe nicht-autorisierte RVK-Ausgaben des LLM: {preview}\n",
                    "dk_classification"
                )

        return deduplicated

    def _emit_rvk_source_diagnostics(
        self,
        keyword_results: List[Dict[str, Any]],
        stream_callback: Optional[callable] = None,
        step_id: str = "dk_search",
    ) -> None:
        """Emit a compact per-run summary of RVK candidate provenance."""
        if not stream_callback:
            return

        buckets: Dict[str, set] = {
            "catalog_standard": set(),
            "catalog_nonstandard": set(),
            "catalog_validation_error": set(),
            "gnd_index": set(),
            "rvk_api": set(),
        }

        for kw_result in keyword_results:
            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", ""))).upper()
                if cls_type != "RVK":
                    continue
                normalized = canonicalize_rvk_notation(classification.get("dk", ""))
                if not normalized:
                    continue
                source = classification.get("source", "")
                status = classification.get("rvk_validation_status")
                if source == "rvk_gnd_index":
                    buckets["gnd_index"].add(normalized)
                elif source == "rvk_api":
                    buckets["rvk_api"].add(normalized)
                elif status == "standard":
                    buckets["catalog_standard"].add(normalized)
                elif status == "validation_error":
                    buckets["catalog_validation_error"].add(normalized)
                elif status == "non_standard":
                    buckets["catalog_nonstandard"].add(normalized)

        parts = []
        if buckets["catalog_standard"]:
            parts.append(f"Katalog standard {len(buckets['catalog_standard'])}")
        if buckets["catalog_nonstandard"]:
            parts.append(f"Katalog lokal {len(buckets['catalog_nonstandard'])}")
        if buckets["catalog_validation_error"]:
            parts.append(f"Katalog ungeprüft {len(buckets['catalog_validation_error'])}")
        if buckets["gnd_index"]:
            parts.append(f"RVK-GND-Index {len(buckets['gnd_index'])}")
        if buckets["rvk_api"]:
            parts.append(f"RVK-API-Label {len(buckets['rvk_api'])}")

        if parts:
            stream_callback(f"ℹ️ RVK-Quellen: {', '.join(parts)}\n", step_id)

    def _score_rvk_candidate(self, candidate: Dict[str, Any], original_abstract: str) -> int:
        """Deterministically score validated RVK candidates."""
        source = candidate.get("source", "catalog")
        status = candidate.get("rvk_validation_status", "standard")
        matched_keywords = [
            canonicalize_keyword(keyword).lower()
            for keyword in (candidate.get("matched_keywords") or [])
            if canonicalize_keyword(keyword)
        ]
        label = str(candidate.get("label", "") or "")
        ancestor_path = str(candidate.get("ancestor_path", "") or "")
        register_entries = [str(item) for item in (candidate.get("register") or []) if str(item).strip()]
        titles = [str(item) for item in (candidate.get("titles") or []) if str(item).strip()]
        haystack = " ".join([label, ancestor_path, " ".join(register_entries)]).lower()
        abstract_text = str(original_abstract or "").lower()

        source_weight = {
            "rvk_gnd_index": 120,
            "rvk_api": 100,
            "catalog": 80,
        }.get(source, 60)
        if status == "non_standard":
            source_weight -= 40
        elif status == "validation_error":
            source_weight -= 60

        overlap_score = 0
        for keyword in matched_keywords:
            if keyword and keyword in haystack:
                overlap_score += 12
            elif keyword and keyword in abstract_text:
                overlap_score += 3

        specificity = len(re.sub(r"[^A-Z0-9]", "", str(candidate.get("dk", "")))) * 2
        count_score = min(int(candidate.get("count", 0) or 0), 30) * 3
        keyword_support = len(set(matched_keywords)) * 8
        title_support = min(len(titles), 10)
        register_support = min(len(register_entries), 8)

        return source_weight + overlap_score + specificity + count_score + keyword_support + title_support + register_support

    def _select_final_rvk_candidates(
        self,
        candidate_results: List[Dict[str, Any]],
        original_abstract: str,
        max_standard: int = 2,
        max_nonstandard: int = 1,
    ) -> List[str]:
        """Select final RVK deterministically from validated candidates."""
        standard_candidates = []
        nonstandard_candidates = []
        aggregated_candidates: Dict[str, Dict[str, Any]] = {}

        def _source_rank(source: str) -> int:
            return {
                "rvk_gnd_index": 3,
                "rvk_api": 2,
                "catalog": 1,
            }.get(source, 0)

        def _status_rank(status: str) -> int:
            return {
                "standard": 3,
                "non_standard": 2,
                "validation_error": 1,
            }.get(status, 0)

        for candidate in candidate_results:
            cls_type = str(candidate.get("classification_type", candidate.get("type", "DK"))).upper()
            if cls_type != "RVK":
                continue

            normalized = canonicalize_rvk_notation(candidate.get("dk", ""))
            if not normalized:
                continue

            current = aggregated_candidates.get(normalized)
            if current is None:
                current = dict(candidate)
                current["dk"] = normalized
                current["matched_keywords"] = list(candidate.get("matched_keywords", []) or [])
                current["titles"] = list(candidate.get("titles", []) or [])
                current["register"] = list(candidate.get("register", []) or [])
                current["count"] = int(candidate.get("count", 0) or 0)
                aggregated_candidates[normalized] = current
            else:
                current["count"] = int(current.get("count", 0) or 0) + int(candidate.get("count", 0) or 0)
                for field in ("matched_keywords", "titles", "register"):
                    existing_values = list(current.get(field, []) or [])
                    seen_values = set(existing_values)
                    for value in candidate.get(field, []) or []:
                        if value not in seen_values:
                            existing_values.append(value)
                            seen_values.add(value)
                    current[field] = existing_values

                if candidate.get("label") and not current.get("label"):
                    current["label"] = candidate.get("label")
                if candidate.get("ancestor_path") and not current.get("ancestor_path"):
                    current["ancestor_path"] = candidate.get("ancestor_path")
                if candidate.get("branch_family") and not current.get("branch_family"):
                    current["branch_family"] = candidate.get("branch_family")

                current_source = str(current.get("source", "catalog") or "catalog")
                incoming_source = str(candidate.get("source", "catalog") or "catalog")
                current_status = str(current.get("rvk_validation_status", "standard") or "standard")
                incoming_status = str(candidate.get("rvk_validation_status", "standard") or "standard")
                replace_source = _source_rank(incoming_source) > _source_rank(current_source)
                replace_status = _status_rank(incoming_status) > _status_rank(current_status)
                if replace_status or (incoming_status == current_status and replace_source):
                    current["source"] = incoming_source
                    current["rvk_validation_status"] = incoming_status
                    current["validation_message"] = candidate.get("validation_message", current.get("validation_message", ""))

        for enriched in aggregated_candidates.values():
            enriched["_score"] = self._score_rvk_candidate(enriched, original_abstract)
            enriched["_branch"] = str(enriched.get("branch_family", "") or "")

            status = enriched.get("rvk_validation_status", "standard")
            if status == "standard":
                standard_candidates.append(enriched)
            elif status in {"non_standard", "validation_error"}:
                nonstandard_candidates.append(enriched)

        def _sort_key(item: Dict[str, Any]):
            return (-int(item.get("_score", 0)), item.get("dk", ""))

        standard_candidates.sort(key=_sort_key)
        nonstandard_candidates.sort(key=_sort_key)

        def _pick_diverse(candidates: List[Dict[str, Any]], limit: int) -> List[str]:
            selected = []
            used_branches = set()
            seen_values = set()
            for candidate in candidates:
                value = f"RVK {candidate['dk']}"
                if value in seen_values:
                    continue
                branch = candidate.get("_branch", "")
                if branch and branch in used_branches:
                    continue
                selected.append(value)
                seen_values.add(value)
                if branch:
                    used_branches.add(branch)
                if len(selected) >= limit:
                    return selected
            for candidate in candidates:
                value = f"RVK {candidate['dk']}"
                if value in seen_values:
                    continue
                selected.append(value)
                seen_values.add(value)
                if len(selected) >= limit:
                    break
            return selected

        if standard_candidates:
            return _pick_diverse(standard_candidates, max_standard)
        return _pick_diverse(nonstandard_candidates, max_nonstandard)

    def _extract_dk_from_response(self, response_text: str, output_format: Optional[str] = None) -> List[str]:
        """Extract DK and RVK classifications from LLM response - Claude Generated

        JSON-first extraction if output_format == "json", then XML fallback.
        PRIMARY: Extract from <final_list> tag (like keywords extraction)
        FALLBACK: Use regex patterns only if <final_list> not found
        """
        import re

        # JSON-first extraction - Claude Generated
        if output_format != "xml":
            from ..core.json_response_parser import parse_json_response, extract_dk_from_json
            data = parse_json_response(response_text)
            if data:
                codes = extract_dk_from_json(data)
                if codes:
                    if self.logger:
                        self.logger.info(f"✅ JSON DK-Extraktion: {len(codes)} Klassifikationen")
                    return codes
            if self.logger:
                self.logger.warning("JSON DK-Parsing fehlgeschlagen, Fallback auf XML")

        classification_codes = []

        # PRIMARY METHOD: Extract from <final_list> tag (preferred and most reliable)
        final_list_match = re.search(r'<final_list>\s*(.*?)\s*</final_list>', response_text, re.DOTALL | re.IGNORECASE)

        if final_list_match:
            # Extract and split by pipe separator
            final_list_content = final_list_match.group(1).strip()
            raw_codes = [code.strip() for code in final_list_content.split('|') if code.strip()]

            if self.logger:
                self.logger.info(f"✅ Extracted {len(raw_codes)} classifications from <final_list>")

            # Parse each code (format: "DK 615.9" or "RVK QC 130")
            for code in raw_codes:
                code_clean = code.strip()
                code_upper = code_clean.upper()

                # Keep DK and RVK prefixes intact
                if code_upper.startswith('DK ') or code_upper.startswith('RVK '):
                    classification_codes.append(code_clean)
                elif re.match(r'^\d+(?:\.\d+)*$', code_clean):
                    # Plain number without prefix -> assume DK
                    classification_codes.append(f"DK {code_clean}")
                elif re.match(r'^[A-Z]{1,2}\s*\d+', code_clean):
                    # Letter-number pattern -> assume RVK
                    classification_codes.append(f"RVK {code_clean}")
                else:
                    # Unknown format, keep as-is and log - Claude Generated
                    if self.logger:
                        self.logger.debug(f"⚠️ Unknown format: '{code_clean}' (not DK/RVK prefixed, not number pattern)")
                    classification_codes.append(code_clean)

            if self.logger:
                self.logger.info(f"✅ Parsed {len(classification_codes)} valid classifications from <final_list>")

            return classification_codes

        # FALLBACK METHOD: Use regex patterns (legacy, less reliable)
        if self.logger:
            self.logger.warning("⚠️  No <final_list> found in DK response, using regex fallback (may produce false positives)")

        # Look for DK patterns explicitly prefixed with "DK" (not arbitrary numbers)
        dk_pattern = r'\bDK\s+(\d{1,3}(?:\.\d+)*)\b'
        dk_matches = re.findall(dk_pattern, response_text, re.IGNORECASE)

        for match in dk_matches:
            classification_codes.append(f"DK {match}")

        # Look for RVK patterns explicitly prefixed with "RVK"
        rvk_pattern = r'\bRVK\s+([A-Z]{1,2}\s*\d{1,4}(?:\s*[A-Z]*)?)\b'
        rvk_matches = re.findall(rvk_pattern, response_text, re.IGNORECASE)

        for match in rvk_matches:
            classification_codes.append(f"RVK {match.strip()}")

        if self.logger and not classification_codes:
            self.logger.warning("⚠️  No classifications found using regex fallback either")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(classification_codes))

    def _is_gnd_validated_keyword(self, keyword: str) -> bool:
        """
        Check if a keyword has been validated against GND system (contains GND-ID)
        Claude Generated - GND validation helper for strict filtering

        Args:
            keyword: Keyword string to validate

        Returns:
            True if keyword contains "(GND-ID:..." format, False for plain text keywords
        """
        return "(GND-ID:" in keyword

    def _flatten_keyword_centric_results(
        self, keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Flatten keyword-centric format to DK-centric with deduplication - Claude Generated

        The BiblioClient.extract_dk_classifications_for_keywords() returns keyword-centric format:
        [{"keyword": "Cadmium", "source": "cache", "classifications": [{"dk": "681.3", ...}]},
         {"keyword": "Halbleiter", "source": "...", "classifications": [{"dk": "681.3", ...}]}]

        This function DEDUPLICATES and merges classifications across keywords:
        - Merges identical DK codes from multiple keywords
        - Deduplicates titles across keywords
        - Sums frequency counts
        - Tracks which keywords led to each classification

        Args:
            keyword_results: List of keyword-centric results from BiblioClient

        Returns:
            Flattened and deduplicated list of DK-centric classification results
        """
        # Group classifications by "{type}:{code}" to detect and merge duplicates
        grouped = {}  # Key: "DK:681.3", Value: merged classification data

        for kw_result in keyword_results:
            keyword = kw_result.get("keyword", "unknown")
            classifications = kw_result.get("classifications", [])

            for cls in classifications:
                cls_type = cls.get("type") or cls.get("classification_type", "DK")
                cls_code = cls.get("dk", "")
                key = f"{cls_type}:{cls_code}"

                # Initialize group if first time seeing this classification
                if key not in grouped:
                    grouped[key] = {
                        "dk": cls_code,
                        "type": cls_type,
                        "classification_type": cls.get("classification_type", cls_type),
                        "titles": [],
                        "count": 0,
                        "matched_keywords": [],
                        "keyword_counts": {},
                        "source": cls.get("source"),
                        "label": cls.get("label"),
                        "ancestor_path": cls.get("ancestor_path"),
                        "register": list(cls.get("register", [])) if cls.get("register") else [],
                        "score": cls.get("score", 0),
                        "branch_family": cls.get("branch_family"),
                        "rvk_validation_status": cls.get("rvk_validation_status"),
                        "validation_message": cls.get("validation_message"),
                    }

                # Merge titles (deduplicate using set) - Claude Generated
                # Filter out placeholder titles from cache that should not be displayed
                title_set = set(grouped[key]["titles"])
                for title in cls.get("titles", []):
                    # Skip placeholder titles from classification cache - Claude Generated
                    if title.startswith("Cached Catalog Entry for RSN"):
                        continue
                    if title == "Cached Author":
                        continue
                    if title not in title_set:
                        grouped[key]["titles"].append(title)
                        title_set.add(title)

                # Sum counts from this keyword
                grouped[key]["count"] += cls.get("count", 0)
                grouped[key]["score"] = max(grouped[key].get("score", 0), cls.get("score", 0))

                # Track which keywords contributed to this classification
                if keyword not in grouped[key]["matched_keywords"]:
                    grouped[key]["matched_keywords"].append(keyword)
                grouped[key]["keyword_counts"][keyword] = cls.get("count", 0)
                for register_entry in cls.get("register", []) or []:
                    if register_entry not in grouped[key]["register"]:
                        grouped[key]["register"].append(register_entry)
                if cls.get("rvk_validation_status") and not grouped[key].get("rvk_validation_status"):
                    grouped[key]["rvk_validation_status"] = cls.get("rvk_validation_status")
                if cls.get("validation_message") and not grouped[key].get("validation_message"):
                    grouped[key]["validation_message"] = cls.get("validation_message")

        # Convert to list and sort by count (most frequent first)
        flattened = sorted(grouped.values(), key=lambda x: x["count"], reverse=True)

        # Log deduplication metrics
        original_count = sum(len(kr.get("classifications", [])) for kr in keyword_results)
        deduplicated_count = len(flattened)
        if original_count > deduplicated_count:
            logger.info(f"🔧 DK Deduplication: {original_count} → {deduplicated_count} (-{original_count - deduplicated_count} Duplikate entfernt)")

        return flattened

    def _calculate_dk_statistics(
        self,
        deduplicated_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for DK/RVK classifications - Claude Generated

        Tracks frequency, deduplication metrics, and keyword coverage for classifications.

        Args:
            deduplicated_results: Merged classification results from _flatten_keyword_centric_results()
            keyword_results: Original keyword-centric results from BiblioClient

        Returns:
            Statistics dictionary with frequency data and deduplication metrics
        """
        # Calculate basic metrics
        total_classifications = len(deduplicated_results)
        original_count = sum(len(kr.get("classifications", [])) for kr in keyword_results)
        duplicates_removed = original_count - total_classifications

        # Get top 10 most frequent classifications
        top_10 = sorted(deduplicated_results, key=lambda x: x["count"], reverse=True)[:10]

        # Build keyword coverage map (which keywords led to which DK codes)
        keyword_coverage = {}
        for result in deduplicated_results:
            for keyword in result.get("matched_keywords", []):
                if keyword not in keyword_coverage:
                    keyword_coverage[keyword] = []
                keyword_coverage[keyword].append(result["dk"])

        # Calculate frequency distribution (how many classifications have X occurrences)
        freq_dist = {}
        for result in deduplicated_results:
            count = result["count"]
            freq_dist[count] = freq_dist.get(count, 0) + 1

        # Estimated token savings (rough estimate: ~70 tokens per duplicate entry removed)
        estimated_token_savings = duplicates_removed * 70

        return {
            "total_classifications": total_classifications,
            "total_keywords_searched": len(keyword_results),
            "most_frequent": [
                {
                    "dk": r["dk"],
                    "type": r.get("type", r.get("classification_type", "DK")),
                    "count": r["count"],
                    "keywords": r.get("matched_keywords", []),
                    "unique_titles": len(r.get("titles", []))
                }
                for r in top_10
            ],
            "keyword_coverage": keyword_coverage,
            "frequency_distribution": freq_dist,
            "deduplication_stats": {
                "original_count": original_count,
                "duplicates_removed": duplicates_removed,
                "deduplication_rate": f"{duplicates_removed / original_count * 100:.1f}%" if original_count > 0 else "0%",
                "estimated_token_savings": estimated_token_savings
            }
        }

    def execute_dk_search(
        self,
        keywords: List[str],
        stream_callback: Optional[callable] = None,
        max_results: int = DEFAULT_DK_MAX_RESULTS,
        catalog_token: str = None,
        catalog_search_url: str = None,
        catalog_details_url: str = None,
        catalog_web_search_url: str = None,
        catalog_web_record_url: str = None,
        force_update: bool = False,  # Claude Generated
        strict_gnd_validation: bool = True,  # EXPERT OPTION: Allow disabling strict GND validation
    ) -> List[Dict[str, Any]]:
        """
        Execute catalog search for DK classification data - Claude Generated

        Args:
            keywords: List of keywords to search
            stream_callback: Optional callback for progress updates
            max_results: Maximum results per keyword
            catalog_token: Catalog API token
            catalog_search_url: Catalog search endpoint URL
            catalog_details_url: Catalog details endpoint URL
            force_update: If True, results will be merged with existing cache (used by store_classification_results)
            strict_gnd_validation: If True (default), only use GND-validated keywords. If False, include plain text keywords.

        Returns:
            List of classification results with titles, counts, and metadata
        """

        # Log force_update status - Claude Generated
        if force_update and self.logger:
            self.logger.info("⚠️ Force update enabled: new titles will be merged with existing")

        # Use provided catalog configuration or allow web fallback - Claude Generated
        if not catalog_token or not catalog_token.strip():
            if self.logger:
                self.logger.warning("No catalog token provided - BiblioClient will use web scraping fallback")
            if stream_callback:
                stream_callback("Kein Katalog-Token: Web-Fallback wird verwendet\n", "dk_search")
            catalog_token = ""  # Empty token triggers automatic web fallback

        # Initialize catalog client - supports BiblioClient or MarcXmlClient - Claude Generated
        try:
            # Determine catalog type from config
            try:
                from .config_manager import ConfigManager
                config_manager = ConfigManager()
                catalog_config = config_manager.get_catalog_config()
                catalog_type = getattr(catalog_config, 'catalog_type', 'libero_soap')
                if catalog_type == 'auto':
                    catalog_type = catalog_config.get_catalog_type() if hasattr(catalog_config, 'get_catalog_type') else 'libero_soap'
            except Exception as cfg_err:
                if self.logger:
                    self.logger.debug(f"Config load failed, using default: {cfg_err}")
                catalog_type = 'libero_soap'  # Default to original behavior
            
            if catalog_type == 'marcxml_sru':
                # Use MARC XML SRU client - Claude Generated
                from .clients.marcxml_client import MarcXmlClient
                sru_preset = getattr(catalog_config, 'sru_preset', '') if 'catalog_config' in dir() else ''
                sru_base_url = getattr(catalog_config, 'sru_base_url', '') if 'catalog_config' in dir() else ''
                sru_max_records = getattr(catalog_config, 'sru_max_records', 50) if 'catalog_config' in dir() else 50
                
                extractor = MarcXmlClient(
                    preset=sru_preset if sru_preset else '',
                    sru_base_url=sru_base_url if not sru_preset else '',
                    max_records=sru_max_records,
                    debug=self.logger.level <= 10 if self.logger else False
                )
                if self.logger:
                    self.logger.info(f"Using MARC XML SRU client (preset: {sru_preset or 'custom'})")
                if stream_callback:
                    stream_callback(f"Verwende MARC XML SRU ({sru_preset or sru_base_url})\n", "dk_search")
            else:
                # Use original Libero SOAP client
                from .clients.biblio_client import BiblioClient

                extractor = BiblioClient(
                    token=catalog_token or "",  # Ensure string, not None
                    debug=self.logger.level <= 10 if self.logger else False,
                    enable_web_fallback=True,  # Claude Generated - Explicitly enable web fallback
                    soap_search_url=catalog_search_url or "",  # Claude Generated - from CatalogConfig
                    soap_details_url=catalog_details_url or "",  # Claude Generated - from CatalogConfig
                    web_search_url=catalog_web_search_url or "",  # Claude Generated - from CatalogConfig
                    web_record_url=catalog_web_record_url or "",  # Claude Generated - from CatalogConfig
                )
                
        except Exception as e:
            error_msg = f"Catalog client initialization failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            if stream_callback:
                stream_callback(f"❌ Katalog-Initialisierung fehlgeschlagen: {str(e)}\n", "dk_search")
            # Return structured result with error info - Claude Generated
            return {
                "classifications": [],
                "statistics": {
                    "error": error_msg,
                    "initialization_failed": True,
                    "total_classifications": 0,
                    "total_keywords_searched": 0,
                    "most_frequent": [],
                    "keyword_coverage": {},
                    "frequency_distribution": {},
                    "deduplication_stats": {
                        "original_count": 0,
                        "duplicates_removed": 0,
                        "deduplication_rate": "0%",
                        "estimated_token_savings": 0
                    }
                },
                "keyword_results": []
            }

        # GND VALIDATION FILTERING - Claude Generated
        # Default (strict_gnd_validation=True): Only use keywords with validated GND-IDs
        #   This prevents irrelevant catalog titles from plain text keywords (e.g., "Molekül")
        # Expert mode (strict_gnd_validation=False): Include plain text keywords too
        gnd_validated_keywords = []
        gnd_keyword_entries = []
        plain_keywords = []

        for keyword in keywords:
            if self._is_gnd_validated_keyword(keyword):
                # Extract just the keyword part before (GND-ID:...)
                clean_keyword = keyword.split("(GND-ID:")[0].strip()
                gnd_validated_keywords.append(clean_keyword)
                gnd_id = extract_gnd_id(keyword)
                if gnd_id:
                    gnd_keyword_entries.append({
                        "keyword": clean_keyword,
                        "gnd_id": gnd_id,
                    })
            else:
                # Plain text keyword without GND-ID validation
                plain_keywords.append(keyword)

        # CRITICAL FIX: Deduplicate after GND-ID stripping - Claude Generated
        # Problem: Same keyword from different GND-IDs (e.g. "Cadmium (GND-ID: 123)" and "Cadmium (GND-ID: 456)")
        # both become "Cadmium" after stripping, leading to duplicate searches
        gnd_validated_keywords_before = len(gnd_validated_keywords)
        gnd_validated_keywords = deduplicate_canonical_keywords(gnd_validated_keywords)
        gnd_validated_keywords_after = len(gnd_validated_keywords)
        seen_gnd_ids = set()
        deduplicated_gnd_entries = []
        for entry in gnd_keyword_entries:
            gnd_id = entry.get("gnd_id")
            if not gnd_id or gnd_id in seen_gnd_ids:
                continue
            seen_gnd_ids.add(gnd_id)
            deduplicated_gnd_entries.append(entry)
        gnd_keyword_entries = deduplicated_gnd_entries

        if gnd_validated_keywords_before != gnd_validated_keywords_after:
            duplicates_removed = gnd_validated_keywords_before - gnd_validated_keywords_after
            if self.logger:
                self.logger.info(
                    f"🔧 GND Keywords Deduplication: {gnd_validated_keywords_before} → "
                    f"{gnd_validated_keywords_after} unique ({duplicates_removed} duplicates removed)"
                )

        # Decide which keywords to use based on strict_gnd_validation setting - Claude Generated
        if strict_gnd_validation:
            final_search_keywords = gnd_validated_keywords
            filtered_keywords = plain_keywords
        else:
            # Combine GND and plain keywords, then deduplicate to avoid searching "Keyword" twice
            combined_keywords = gnd_validated_keywords + plain_keywords
            combined_before = len(combined_keywords)
            final_search_keywords = deduplicate_canonical_keywords(combined_keywords)
            combined_after = len(final_search_keywords)

            if combined_before != combined_after:
                duplicates_removed = combined_before - combined_after
                if self.logger:
                    self.logger.info(
                        f"🔧 Combined Keywords Deduplication: {combined_before} → "
                        f"{combined_after} unique ({duplicates_removed} duplicates removed)"
                    )

            filtered_keywords = []

        # Log filtering results - Claude Generated: Enhanced user feedback
        if filtered_keywords and strict_gnd_validation:
            # Build list of excluded keyword texts for user feedback
            excluded_list = ", ".join([kw[:40] for kw in filtered_keywords[:5]])
            if len(filtered_keywords) > 5:
                excluded_list += f", +{len(filtered_keywords)-5} weitere"

            if self.logger:
                self.logger.info(
                    f"🔍 DK Search (strict GND mode): {len(gnd_validated_keywords)} GND-validated keywords used, "
                    f"{len(filtered_keywords)} plain keywords excluded: {excluded_list}"
                )
            if stream_callback:
                stream_callback(
                    f"⚠️ DK-Suche-Filter: {len(gnd_validated_keywords)} GND-validierte Keywords, "
                    f"{len(filtered_keywords)} ohne GND ausgeschlossen\n   Ausgeschlossen: {excluded_list}\n",
                    "dk_search"
                )

        # Handle edge case: all keywords filtered
        if not final_search_keywords:
            if self.logger:
                self.logger.warning(
                    f"⚠️ DK Search: All {len(keywords)} keywords lack GND validation - skipping catalog search"
                )
            if stream_callback:
                stream_callback(
                    "⚠️ Keine Keywords für DK-Suche vorhanden\n",
                    "dk_search"
                )
            # Return empty results in new format - Claude Generated Step 3
            return {
                "classifications": [],
                "statistics": {
                    "total_classifications": 0,
                    "total_keywords_searched": 0,
                    "most_frequent": [],
                    "keyword_coverage": {},
                    "frequency_distribution": {},
                    "deduplication_stats": {
                        "original_count": 0,
                        "duplicates_removed": 0,
                        "deduplication_rate": "0%",
                        "estimated_token_savings": 0
                    }
                },
                "keyword_results": []
            }

        if stream_callback:
            mode_info = "(strict GND mode)" if strict_gnd_validation else "(including plain keywords)"
            stream_callback(
                f"Suche Katalog-Einträge für {len(final_search_keywords)} Keywords {mode_info} (max {max_results})\n",
                "dk_search"
            )

        # Execute catalog search with Per-Keyword Feedback - Claude Generated QUICK-FIX
        # IMPORTANT: Loop over keywords individually to provide per-keyword status feedback
        # This replaces the old single-batch call with individual keyword searches
        try:
            from .clients.marcxml_client import MarcXmlClient

            dk_search_results = []
            success_count = 0
            failed_keywords = []

            # Process EACH keyword individually for detailed feedback
            for idx, keyword in enumerate(final_search_keywords, 1):
                # Check circuit breaker status before each search - Claude Generated
                if hasattr(extractor, 'get_circuit_breaker_status'):
                    cb_status = extractor.get_circuit_breaker_status()
                    if cb_status.get('open'):
                        remaining = cb_status.get('remaining_seconds', 60)
                        if stream_callback:
                            stream_callback(
                                f"⏳ Katalog-Server überlastet: Wartezeit {remaining}s...\n",
                                "dk_search"
                            )
                        if self.logger:
                            self.logger.warning(f"Circuit breaker open, waiting {remaining}s")
                        time.sleep(min(remaining + 1, 65))

                # Progress callback with percentage - Claude Generated
                if stream_callback:
                    progress_pct = int((idx / len(final_search_keywords)) * 100)
                    stream_callback(
                        f"[{idx}/{len(final_search_keywords)}] ({progress_pct}%) Suche '{keyword}'...\n",
                        "dk_search"
                    )

                try:
                    # Search THIS keyword only (not all keywords)
                    if isinstance(extractor, MarcXmlClient):
                        kw_results = extractor.extract_dk_classifications_for_keywords(
                            keywords=[keyword],  # Single keyword
                            max_results=max_results,
                        )
                    else:
                        kw_results = extractor.extract_dk_classifications_for_keywords(
                            keywords=[keyword],  # Single keyword
                            max_results=max_results,
                            force_update=force_update,
                        )

                    # Analyze result for THIS keyword
                    if kw_results and len(kw_results) > 0:
                        kw_result = kw_results[0]
                        classifications = kw_result.get("classifications", [])

                        if classifications:
                            # SUCCESS: Keyword found with classifications
                            success_count += 1
                            if stream_callback:
                                stream_callback(
                                    f"  ✅ {keyword}: {len(classifications)} Klassifikationen gefunden\n",
                                    "dk_search"
                                )
                            dk_search_results.append(kw_result)
                        else:
                            # PARTIAL: Keyword found but no classifications
                            failed_keywords.append((keyword, "no_results", "Keine Klassifikationen"))
                            if stream_callback:
                                stream_callback(
                                    f"  ⚠️ {keyword}: Keine Klassifikationen gefunden\n",
                                    "dk_search"
                                )
                    else:
                        # FAILURE: Keyword search completely failed
                        failed_keywords.append((keyword, "error", "Suche fehlgeschlagen"))
                        if stream_callback:
                            stream_callback(
                                f"  ❌ {keyword}: Suche fehlgeschlagen\n",
                                "dk_search"
                            )

                except Exception as kw_error:
                    # Individual keyword error
                    if self.logger:
                        self.logger.error(f"Error searching keyword '{keyword}': {kw_error}")
                    failed_keywords.append((keyword, "error", str(kw_error)))
                    if stream_callback:
                        stream_callback(
                            f"  ❌ {keyword}: Fehler - {str(kw_error)}\n",
                            "dk_search"
                        )

            # Summary callback with complete stats
            if stream_callback:
                stream_callback(
                    f"✅ DK-Suche abgeschlossen: {success_count}/{len(final_search_keywords)} erfolgreich\n",
                    "dk_search"
                )

                # List failed keywords if any
                if failed_keywords:
                    failed_list = ", ".join([k for k, _, _ in failed_keywords[:5]])
                    if len(failed_keywords) > 5:
                        failed_list += f", +{len(failed_keywords) - 5} weitere"

                    stream_callback(
                        f"⚠️ {len(failed_keywords)} fehlgeschlagen: {failed_list}\n",
                        "dk_search"
                    )

            dk_search_results = self._validate_catalog_rvk_candidates(
                dk_search_results,
                stream_callback=stream_callback,
            )
            dk_search_results = self._inject_rvk_api_fallback(
                final_search_keywords,
                dk_search_results,
                gnd_keyword_entries=gnd_keyword_entries,
                stream_callback=stream_callback,
            )
            self._emit_rvk_source_diagnostics(
                dk_search_results,
                stream_callback=stream_callback,
                step_id="dk_search",
            )

            # Deduplicate and flatten classifications - Claude Generated Step 3
            dk_search_results_flattened = self._flatten_keyword_centric_results(dk_search_results)

            # Calculate comprehensive statistics - Claude Generated Step 3
            dk_statistics = self._calculate_dk_statistics(dk_search_results_flattened, dk_search_results)

            # Return results in new format with statistics and transparency
            return {
                "classifications": dk_search_results_flattened,  # Deduplicated for LLM prompt
                "statistics": dk_statistics,                      # For display/diagnostics
                "keyword_results": dk_search_results              # Original keyword-centric for GUI transparency
            }

        except Exception as e:
            error_msg = f"DK catalog search failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            if stream_callback:
                stream_callback(f"❌ DK-Suche-Fehler: {str(e)}\n", "dk_search")

            # Return partial results if available - Claude Generated
            # Check if we collected any results before the error occurred
            if 'dk_search_results' in dir() and dk_search_results:
                if stream_callback:
                    stream_callback(
                        f"⚠️ Teilergebnisse: {len(dk_search_results)} Keywords erfolgreich vor Fehler\n",
                        "dk_search"
                    )
                dk_search_results = self._validate_catalog_rvk_candidates(
                    dk_search_results,
                    stream_callback=stream_callback,
                )
                dk_search_results = self._inject_rvk_api_fallback(
                    final_search_keywords,
                    dk_search_results,
                    gnd_keyword_entries=gnd_keyword_entries,
                    stream_callback=stream_callback,
                )
                self._emit_rvk_source_diagnostics(
                    dk_search_results,
                    stream_callback=stream_callback,
                    step_id="dk_search",
                )
                # Process partial results
                dk_search_results_flattened = self._flatten_keyword_centric_results(dk_search_results)
                dk_statistics = self._calculate_dk_statistics(dk_search_results_flattened, dk_search_results)
                dk_statistics["error"] = error_msg
                dk_statistics["partial_results"] = True
                return {
                    "classifications": dk_search_results_flattened,
                    "statistics": dk_statistics,
                    "keyword_results": dk_search_results
                }

            # Return empty results if no partial data
            return {
                "classifications": [],
                "statistics": {
                    "error": error_msg,
                    "total_classifications": 0,
                    "total_keywords_searched": 0,
                    "most_frequent": [],
                    "keyword_coverage": {},
                    "frequency_distribution": {},
                    "deduplication_stats": {
                        "original_count": 0,
                        "duplicates_removed": 0,
                        "deduplication_rate": "0%",
                        "estimated_token_savings": 0
                    }
                },
                "keyword_results": []
            }


    def execute_complete_pipeline(
        self,
        input_text: str,
        pipeline_config=None,
        stream_callback: Optional[callable] = None,
    ) -> "KeywordAnalysisState":
        """
        Execute a complete ALIMA pipeline synchronously without Qt dependencies.

        Chains: initialisation → search → keywords → (dk_classification if enabled)

        This is the preferred method for batch processing since it runs purely
        synchronously in any thread context (no QThread/event loop required).

        Claude Generated
        """
        from ..core.pipeline_manager import PipelineConfig

        config = pipeline_config or PipelineConfig()

        # Helpers to read per-step config
        def _step(step_id):
            return config.get_step_config(step_id) if hasattr(config, "get_step_config") else None

        def _cb(step_id):
            """Wrap stream_callback with step_id prefix for identification."""
            if not stream_callback:
                return None
            def _inner(token, sid=step_id):
                stream_callback(token, sid)
            return _inner

        # ── Step 1: initialisation ───────────────────────────────────────────
        init_cfg = _step("initialisation")
        if stream_callback:
            stream_callback(f"\n▶ [initialisation]\n", "initialisation")
        keywords, gnd_classes, init_analysis, llm_title = self.execute_initial_keyword_extraction(
            abstract_text=input_text,
            provider=init_cfg.provider if init_cfg else None,
            model=init_cfg.model if init_cfg else None,
            task=init_cfg.task or "initialisation" if init_cfg else "initialisation",
            stream_callback=_cb("initialisation"),
        )

        # ── Step 2: GND search ───────────────────────────────────────────────
        if stream_callback:
            stream_callback(f"\n▶ [search] {len(keywords)} keywords\n", "search")
        search_results = self.execute_gnd_search(
            keywords=keywords,
            suggesters=config.search_suggesters,
            stream_callback=_cb("search"),
        )

        # ── Step 3: keywords (final analysis) ────────────────────────────────
        kw_cfg = _step("keywords")
        if stream_callback:
            stream_callback(f"\n▶ [keywords]\n", "keywords")
        final_keywords, gnd_compliant, kw_analysis = self.execute_final_keyword_analysis(
            original_abstract=input_text,
            search_results=search_results,
            provider=kw_cfg.provider if kw_cfg else None,
            model=kw_cfg.model if kw_cfg else None,
            task=kw_cfg.task or "keywords" if kw_cfg else "keywords",
            stream_callback=_cb("keywords"),
        )

        # ── Build state ───────────────────────────────────────────────────────
        state = KeywordAnalysisState(
            original_abstract=input_text,
            initial_keywords=keywords,
            search_suggesters_used=config.search_suggesters,
            initial_gnd_classes=gnd_classes,
            search_results=[],
            final_llm_analysis=kw_analysis,
            working_title=llm_title,
        )

        # ── Step 4: DK classification (optional) ─────────────────────────────
        dk_cfg = _step("dk_classification")
        if dk_cfg and getattr(dk_cfg, "enabled", False):
            if stream_callback:
                stream_callback(f"\n▶ [dk_classification]\n", "dk_classification")
            try:
                dk_search = self.execute_dk_search(
                    keywords=final_keywords,
                    stream_callback=_cb("dk_classification"),
                )
                state.dk_search_results = dk_search.get("keyword_results", [])
                state.dk_search_results_flattened = dk_search.get("classifications", [])
                state.dk_statistics = dk_search.get("statistics")
                dk_classes, dk_analysis = self.execute_dk_classification(
                    original_abstract=input_text,
                    dk_search_results=dk_search.get("classifications", []),
                    provider=dk_cfg.provider if dk_cfg else None,
                    model=dk_cfg.model if dk_cfg else None,
                    stream_callback=_cb("dk_classification"),
                )
                state.dk_classifications = dk_classes
                state.dk_llm_analysis = dk_analysis
                state.rvk_provenance = getattr(dk_analysis, "rvk_provenance", {})
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"DK classification failed (non-fatal): {e}")

        return state


def verify_keywords_against_gnd_pool(
    extracted_keywords: List[str],
    gnd_pool_keywords: List[str],
    stream_callback: Optional[callable] = None,
    step_id: str = "keywords",
    knowledge_manager = None,  # Claude Generated - for DB fallback verification
) -> Dict[str, Any]:
    """Verify LLM-extracted keywords against the GND pool from step 2 - Claude Generated

    Args:
        extracted_keywords: Keywords extracted from LLM response
        gnd_pool_keywords: The gnd_compliant_keywords built from search results (step 2)
        stream_callback: Optional callback for live progress feedback
        step_id: Pipeline step ID for callback routing
        knowledge_manager: Optional UnifiedKnowledgeManager for DB fallback verification

    Returns:
        Dict with keys: verified (list), rejected (list), stats (dict)
    """
    import logging
    vlog = logging.getLogger(__name__)

    # Debug: Check if knowledge_manager is available
    vlog.info(f"🔍 Verifikation: knowledge_manager={'verfügbar' if knowledge_manager else 'NICHT verfügbar'}")

    if not gnd_pool_keywords:
        vlog.warning("⚠️ GND pool is empty - skipping verification")
        return {
            "verified": list(extracted_keywords),
            "rejected": [],
            "stats": {"total_extracted": len(extracted_keywords), "verified_count": len(extracted_keywords),
                       "rejected_count": 0, "pool_size": 0},
        }

    # Build lookup maps from pool
    gnd_id_to_pool = {}   # gnd_id -> full pool keyword
    text_lower_to_pool = {}  # keyword_text_lower -> full pool keyword

    for pool_kw in gnd_pool_keywords:
        # Extract GND-ID if present
        gnd_match = re.search(r'GND-ID:\s*([0-9X-]+)', pool_kw)
        if gnd_match:
            gnd_id_to_pool[gnd_match.group(1)] = pool_kw

        # Extract keyword text (before first parenthesis)
        kw_text = pool_kw.split('(')[0].strip().lower()
        if kw_text:
            text_lower_to_pool[kw_text] = pool_kw

    if stream_callback:
        stream_callback(
            f"\n🔍 Verifiziere {len(extracted_keywords)} Keywords gegen GND-Pool ({len(gnd_pool_keywords)} Einträge)...\n",
            step_id,
        )

    verified = []
    rejected = []

    for kw in extracted_keywords:
        matched_pool_kw = None

        # Extract GND-ID from extracted keyword (formats: "Keyword (1234567-8)" or "Keyword (GND-ID: 1234567-8)")
        gnd_id_match = re.search(r'GND-ID:\s*([0-9X-]+)', kw)
        if not gnd_id_match:
            gnd_id_match = re.search(r'\((\d{7,}-\d{1,2})\)', kw)

        if gnd_id_match:
            gnd_id = gnd_id_match.group(1)
            if gnd_id in gnd_id_to_pool:
                matched_pool_kw = gnd_id_to_pool[gnd_id]

        # Text match fallback
        if not matched_pool_kw:
            kw_text = kw.split('(')[0].strip().lower()
            if kw_text in text_lower_to_pool:
                matched_pool_kw = text_lower_to_pool[kw_text]

        # Database lookup fallback - Claude Generated
        # If keyword not in pool, search by text in database (ignore LLM's GND-ID)
        if not matched_pool_kw:
            if knowledge_manager:
                kw_text = kw.split('(')[0].strip()
                vlog.info(f"  🔍 DB-Suche nach '{kw_text}'")

                try:
                    # Search GND entries by title/synonyms - this is the authoritative source
                    results = knowledge_manager.search_gnd_by_title(kw_text, fuzzy_threshold=90)
                    vlog.info(f"  🔍 DB-Suche Ergebnisse: {len(results)} Treffer")

                    if results and len(results) > 0:
                        first_result = results[0]
                        if isinstance(first_result, dict) and 'gnd_id' in first_result:
                            db_gnd_id = first_result['gnd_id']
                            db_title = first_result.get('title', kw_text)
                            # Use GND-ID from database (authoritative)
                            matched_pool_kw = f"{db_title} (GND-ID: {db_gnd_id})"
                            vlog.info(f"  💾 DB-Treffer: '{kw_text}' → {db_title} (GND-ID: {db_gnd_id})")

                            # Check if we corrected an LLM error
                            if gnd_id_match:
                                llm_gnd_id = gnd_id_match.group(1)
                                if llm_gnd_id != db_gnd_id:
                                    vlog.info(f"  ✅ DB-verifiziert mit Korrektur: '{kw}' → '{matched_pool_kw}'")
                                    if stream_callback:
                                        stream_callback(f"  ✅ {kw_text} - DB-verifiziert (GND-ID korrigiert: {llm_gnd_id} → {db_gnd_id})\n", step_id)
                                else:
                                    vlog.info(f"  ✅ DB-verifiziert: '{matched_pool_kw}'")
                                    if stream_callback:
                                        stream_callback(f"  ✅ {kw_text} - DB-verifiziert\n", step_id)
                            else:
                                # LLM had no GND-ID, we added it from DB
                                vlog.info(f"  ✅ DB-verifiziert: '{kw}' → '{matched_pool_kw}'")
                                if stream_callback:
                                    stream_callback(f"  ✅ {kw_text} - DB-verifiziert (GND-ID ergänzt)\n", step_id)
                        else:
                            vlog.warning(f"  ⚠️ DB-Ergebnis hat keine GND-ID für '{kw_text}'")
                    else:
                        vlog.info(f"  ❌ Kein DB-Eintrag für '{kw_text}' gefunden")
                except Exception as e:
                    vlog.warning(f"  ⚠️ DB-Suche Fehler für '{kw_text}': {e}")
            else:
                # knowledge_manager not available
                kw_text = kw.split('(')[0].strip()
                vlog.warning(f"  ⚠️ DB-Lookup für '{kw_text}' nicht möglich - knowledge_manager fehlt")

        if matched_pool_kw:
            verified.append(matched_pool_kw)
            if matched_pool_kw != kw:  # Only log if it's from pool (not DB)
                vlog.info(f"  ✅ Verifiziert: {kw[:60]} → Pool: {matched_pool_kw[:60]}")
                if stream_callback:
                    short_name = matched_pool_kw.split('(')[0].strip()
                    stream_callback(f"  ✅ {short_name} - GND-verifiziert\n", step_id)
        else:
            rejected.append(kw)
            vlog.warning(f"  ❌ Abgelehnt: {kw[:60]} - nicht im GND-Pool")
            if stream_callback:
                short_name = kw.split('(')[0].strip()
                stream_callback(f"  ❌ {short_name} - nicht im GND-Pool, entfernt\n", step_id)

    stats = {
        "total_extracted": len(extracted_keywords),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "pool_size": len(gnd_pool_keywords),
    }

    if stream_callback:
        stream_callback(
            f"📊 Verifikation: {stats['verified_count']}/{stats['total_extracted']} GND-verifiziert\n",
            step_id,
        )

    vlog.info(
        f"📊 GND-Verifikation: {stats['verified_count']}/{stats['total_extracted']} verifiziert, "
        f"{stats['rejected_count']} abgelehnt"
    )

    return {"verified": verified, "rejected": rejected, "stats": stats}


def extract_keywords_from_descriptive_text(
    text: str, gnd_compliant_keywords: List[str], output_format: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Extract keywords from LLM descriptive text with robust fallback - Claude Generated

    Supports three formats (in priority order):
    0. JSON: Parse structured JSON response (if output_format == "json")
    1. Primary: "<final_list>…</final_list>" scoped – GND regex applied only inside tags
    2. Fallback: Full-text GND regex if no <final_list> tags are present

    Returns:
        Tuple[List[str], List[str]]:
            - all_keywords: ALL keywords (with and without GND-IDs) for DK search
            - exact_matches: ONLY GND-validated keywords for statistics/history

    Note: DK search uses all_keywords to ensure no LLM-identified keywords are lost.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Debug: Log input for diagnosis - Claude Generated
    logger.info(f"🔍 extract_keywords: input={len(text)} chars, available_gnd={len(gnd_compliant_keywords)}, format={output_format}")

    # JSON-first extraction - Claude Generated
    if output_format != "xml":
        from ..core.json_response_parser import parse_json_response, extract_keywords_from_json
        data = parse_json_response(text)
        if data:
            keywords_str = extract_keywords_from_json(data)
            if keywords_str:
                # Parse the comma-separated string into keyword entries
                all_keywords = []
                exact_matches = []
                gnd_compliant_set = set(gnd_compliant_keywords)

                for kw_entry in keywords_str.split(", "):
                    kw_entry = kw_entry.strip()
                    if not kw_entry:
                        continue
                    all_keywords.append(kw_entry)
                    # Check if this keyword matches any GND-compliant keyword
                    for gnd_kw in gnd_compliant_keywords:
                        kw_text = kw_entry.split("(")[0].strip().lower()
                        gnd_text = gnd_kw.split("(")[0].strip().lower()
                        if kw_text == gnd_text:
                            exact_matches.append(gnd_kw)
                            break

                logger.info(f"✅ JSON keyword extraction: {len(all_keywords)} total, {len(exact_matches)} GND-matched")
                return all_keywords, exact_matches
        logger.warning("JSON keyword extraction fehlgeschlagen, Fallback auf XML")

    # Remove <think> blocks (reasoning steps) before any parsing - Claude Generated
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Attempt to scope extraction to <final_list> content - Claude Generated
    final_list_match = re.search(r'<final_list>\s*(.*?)\s*</final_list>', clean_text, re.DOTALL | re.IGNORECASE)
    if final_list_match:
        extraction_scope = final_list_match.group(1)
        logger.info(f"✅ <final_list> found – scoping extraction to {len(extraction_scope)} chars")
    else:
        extraction_scope = clean_text
        logger.warning("⚠️ No <final_list> found – falling back to full text extraction")

    # PRIMARY METHOD: Regex for "Keyword (1234567-8)" format
    pattern = re.compile(r"\b([A-Za-zäöüÄÖÜß\s-]+?)\s*\((\d{7}-\d|\d{7}-\d{1,2})\)")
    matches = pattern.findall(extraction_scope)

    all_extracted_keywords = []
    exact_matches = []

    # Convert gnd_compliant_keywords to set for faster lookup
    gnd_compliant_set = set(gnd_compliant_keywords)

    if matches:
        logger.info(f"✅ Regex found {len(matches)} keyword matches")

        # Build lookup maps for flexible matching - Claude Generated
        gnd_id_lookup = {}  # gnd_id -> full_keyword
        text_lookup = {}    # keyword_text_lower -> full_keyword

        for gnd_kw in gnd_compliant_keywords:
            # Extract GND-ID if present (format: "Keyword (GND-ID: 1234567-8)")
            gnd_match = re.search(r'GND-ID:\s*([0-9-]+)', gnd_kw)
            if gnd_match:
                gnd_id = gnd_match.group(1)
                gnd_id_lookup[gnd_id] = gnd_kw

            # Extract keyword text (before first parenthesis)
            keyword_text = gnd_kw.split('(')[0].strip().lower()
            text_lookup[keyword_text] = gnd_kw

        logger.info(f"🔍 Built lookups: {len(gnd_id_lookup)} GND-IDs, {len(text_lookup)} text entries")

        # Match extracted keywords using dual strategy - Claude Generated
        for keyword_part, gnd_id_part in matches:
            formatted_keyword = f"{keyword_part.strip()} ({gnd_id_part})"
            all_extracted_keywords.append(formatted_keyword)

            matched = False

            # Strategy 1: GND-ID match (e.g. "1234567-8")
            if gnd_id_part in gnd_id_lookup:
                full_keyword = gnd_id_lookup[gnd_id_part]
                exact_matches.append(full_keyword)
                logger.info(f"  ✅ GND-ID match: '{keyword_part}' ({gnd_id_part}) → {full_keyword[:60]}")
                matched = True

            # Strategy 2: Text match (e.g. "cadmium")
            elif keyword_part.strip().lower() in text_lookup:
                full_keyword = text_lookup[keyword_part.strip().lower()]
                exact_matches.append(full_keyword)
                logger.info(f"  ✅ Text match: '{keyword_part}' → {full_keyword[:60]}")
                matched = True

            if not matched:
                logger.warning(f"  ❌ No match: '{keyword_part}' ({gnd_id_part})")

        logger.info(f"✅ Matched {len(exact_matches)} keywords from {len(matches)} regex matches")
        return all_extracted_keywords, exact_matches

    # FALLBACK METHOD: Parse <final_list> format - Claude Generated
    logger.warning("⚠️ Regex found NO matches - trying <final_list> fallback")

    if final_list_match:
        final_list_content = final_list_match.group(1).strip()
        logger.info(f"✅ Found <final_list>: {final_list_content[:100]}")

        # FIXME: Parser robustness - LLM sometimes returns comma-separated instead of pipe-separated keywords
        # Split by pipe separator (preferred), fall back to comma if needed - Claude Generated
        raw_keywords = [kw.strip() for kw in final_list_content.split('|') if kw.strip()]

        # Fallback: if pipe split yields only one keyword, try comma separator - Claude Generated
        if len(raw_keywords) == 1 and ',' in final_list_content:
            logger.warning(f"⚠️ Pipe separator yielded only 1 keyword, attempting comma fallback")
            raw_keywords = [kw.strip() for kw in final_list_content.split(',') if kw.strip()]
            logger.info(f"✅ Comma fallback: {len(raw_keywords)} keywords extracted")

        logger.info(f"✅ Extracted {len(raw_keywords)} raw keywords from <final_list>")

        # Build lookup map: keyword_text_lower -> full_gnd_keyword
        gnd_lookup = {}
        for gnd_kw in gnd_compliant_keywords:
            # Extract keyword text before (GND-ID: ...)
            if "(GND-ID:" in gnd_kw:
                # ROBUST: Normalize whitespace (multiple spaces, tabs, newlines) - Claude Generated
                keyword_text = " ".join(gnd_kw.split("(GND-ID:")[0].split()).lower()
                gnd_lookup[keyword_text] = gnd_kw

        # Match raw keywords against GND lookup - Claude Generated FIX
        unmatched_keywords = []  # FIX: Track keywords without GND matches for DK search
        for raw_kw in raw_keywords:
            # ROBUST: Normalize whitespace and strip GND-ID label if LLM added it literally - Claude Generated
            raw_kw_normalized = " ".join(raw_kw.split()).lower()

            # Fallback: If LLM returned "Keyword (GND-ID)" without actual ID, strip the label
            if raw_kw_normalized.endswith("(gnd-id)"):
                raw_kw_normalized = raw_kw_normalized[:-9].strip()
                logger.info(f"  ℹ️ Stripped literal '(GND-ID)' label from '{raw_kw}'")

            # Exact match
            if raw_kw_normalized in gnd_lookup:
                matched_gnd_kw = gnd_lookup[raw_kw_normalized]
                exact_matches.append(matched_gnd_kw)
                logger.info(f"  ✅ Matched '{raw_kw}' -> {matched_gnd_kw[:60]}")
            else:
                # Fuzzy match: check if raw_kw is contained in any GND keyword
                found = False
                for gnd_kw_text, full_gnd_kw in gnd_lookup.items():
                    if raw_kw_normalized in gnd_kw_text or gnd_kw_text in raw_kw_normalized:
                        exact_matches.append(full_gnd_kw)
                        logger.info(f"  ⚠️ Fuzzy matched '{raw_kw}' -> {full_gnd_kw[:60]}")
                        found = True
                        break

                if not found:
                    # Try database lookup for GND-ID before treating as plain keyword - Claude Generated
                    db_gnd_id = None
                    try:
                        from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
                        ukm = UnifiedKnowledgeManager()
                        results = ukm.search_by_keywords([raw_kw_normalized], fuzzy_threshold=90)
                        if results and len(results) > 0:
                            first_result = results[0]
                            if isinstance(first_result, dict) and 'gnd_id' in first_result:
                                db_gnd_id = first_result['gnd_id']
                    except Exception as db_err:
                        logger.debug(f"DB lookup error for '{raw_kw_normalized}': {db_err}")

                    if db_gnd_id:
                        # Found GND-ID in database - create proper GND keyword format
                        gnd_keyword = f"{raw_kw} (GND-ID: {db_gnd_id})"
                        exact_matches.append(gnd_keyword)
                        logger.info(f"  🔍 DB lookup matched '{raw_kw}' -> {db_gnd_id}")
                    else:
                        # FIXED: Keywords without GND validation are now INCLUDED in DK search - Claude Generated
                        # All LLM-identified keywords are used to ensure complete catalog coverage:
                        # - Keywords WITH GND-IDs: Full metadata from GND database
                        # - Keywords WITHOUT GND-IDs: Plain text search (user requirement: "das DARF nicht passieren")
                        # - Preserves LLM analysis intent while maximizing catalog search coverage
                        logger.info(f"  ℹ️ No GND match for '{raw_kw}' - using as plain keyword in DK search")
                        unmatched_keywords.append(raw_kw)  # Will be included in DK search via all_keywords

        # Combine GND-matched and plain keywords for complete DK search - Claude Generated FIX
        all_keywords = exact_matches + unmatched_keywords
        logger.info(f"✅ Fallback extraction: {len(exact_matches)} GND-matched + {len(unmatched_keywords)} plain keywords = {len(all_keywords)} total")
        return all_keywords, exact_matches  # Return combined list for DK search, GND-only list for history

    # NO EXTRACTION SUCCESSFUL
    logger.error("❌ NO keyword extraction successful (neither regex nor <final_list>)")
    logger.error(f"Text preview: {text[:300]}")
    return [], []


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


def extract_classes_from_descriptive_text(text: str, output_format: Optional[str] = None) -> List[str]:
    """Extract classification classes from LLM text - Claude Generated

    JSON-first extraction if output_format == "json", then XML fallback.
    """
    # JSON-first extraction - Claude Generated
    if output_format != "xml":
        from ..core.json_response_parser import parse_json_response, extract_gnd_classes_from_json
        data = parse_json_response(text)
        if data:
            classes = extract_gnd_classes_from_json(data)
            if classes:
                return classes

    match = re.search(r"<class>(.*?)</class>", text)
    if match:
        classes_str = match.group(1)
        return [cls.strip() for cls in classes_str.split("|") if cls.strip()]
    return []


def canonicalize_keyword(keyword: str) -> str:
    """Extract canonical keyword form by stripping GND-ID - Claude Generated

    Converts "Keyword (GND-ID: 1234567-8)" to "Keyword"
    Handles both formats: with and without GND-ID

    Args:
        keyword: Keyword potentially in format "Keyword (GND-ID: 1234567-8)" or plain "Keyword"

    Returns:
        Canonical form: "Keyword" (stripped of GND-ID and whitespace)

    Examples:
        "Cadmium (GND-ID: 4029921-1)" → "Cadmium"
        "Festkörper (GND-ID: 4016918-2)" → "Festkörper"
        "Molekül" → "Molekül"
    """
    if "(GND-ID:" in keyword:
        return keyword.split("(GND-ID:")[0].strip()
    return keyword.strip()


def extract_gnd_id(keyword: str) -> Optional[str]:
    """Extract GND-ID from a verified keyword string."""
    match = re.search(r"GND-ID:\s*([0-9X-]+)", str(keyword or ""))
    if match:
        return match.group(1).strip()
    return None


def canonicalize_rvk_notation(code: str) -> str:
    """Normalize RVK notation spacing for comparisons and canonical output."""
    clean = str(code or "").strip().upper()
    clean = re.sub(r"\s+", " ", clean)
    match = re.match(r"^([A-Z]{1,4})\s*([0-9].*)$", clean)
    if match:
        clean = f"{match.group(1)} {match.group(2).strip()}"
    return clean


def is_plausible_nonstandard_rvk(code: str) -> bool:
    """Allow local RVK variants, but reject obvious artifacts."""
    clean = canonicalize_rvk_notation(code)
    if not clean:
        return False
    if " - " in clean:
        return False
    return bool(re.match(r"^[A-Z]{1,4}\s[0-9][0-9A-Z./-]*$", clean))


def deduplicate_canonical_keywords(keywords: List[str]) -> List[str]:
    """Deduplicate keywords by canonical form (case-insensitive) - Claude Generated

    Removes duplicate keywords that differ only in their GND-ID or whitespace.
    Preserves first occurrence of each unique canonical keyword.
    Uses case-insensitive comparison to catch "Cadmium" vs "cadmium" duplicates.

    Args:
        keywords: List of keywords, may contain duplicates after GND-ID stripping
                 e.g., ["Cadmium", "Cadmium", "Festkörper"]

    Returns:
        Deduplicated list maintaining original order and formatting

    Examples:
        ["Cadmium", "Cadmium (GND-ID: 1234567-8)"] → ["Cadmium"]
        ["Informatik", "informatik"] → ["Informatik"]
    """
    seen = set()
    deduplicated = []

    for kw in keywords:
        canonical = canonicalize_keyword(kw).lower()
        if canonical not in seen:
            seen.add(canonical)
            deduplicated.append(kw)

    return deduplicated


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
    def convert_lists_to_sets(obj):
        """Convert known list fields back to sets after JSON loading - Claude Generated

        Enhanced to handle all known set fields: gndid, ddc, dk, missing_concepts
        """
        if isinstance(obj, dict):
            # Known set fields in search results and data models
            SET_FIELDS = {"gndid", "ddc", "dk", "missing_concepts"}

            result = {}
            for key, value in obj.items():
                if key in SET_FIELDS and isinstance(value, list):
                    # Convert known set fields back to sets
                    result[key] = set(value)
                elif isinstance(value, (dict, list)):
                    result[key] = PipelineJsonManager.convert_lists_to_sets(value)
                else:
                    result[key] = value
            return result
        elif isinstance(obj, list):
            return [PipelineJsonManager.convert_lists_to_sets(elem) for elem in obj]
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
        """Load KeywordAnalysisState from JSON file with deep object reconstruction - Claude Generated"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Detect and unwrap webapp session format - Claude Generated (Support both CLI/GUI and Webapp formats)
            if "session_id" in data and "results" in data:
                logger.info(f"Detected webapp session format (session_id: {data.get('session_id')}), unwrapping results...")
                # Extract the actual analysis state from webapp's session wrapper
                data = data["results"]

                # Map webapp field names to KeywordAnalysisState field names
                if "final_llm_call_details" in data and "final_llm_analysis" not in data:
                    logger.info("Mapping 'final_llm_call_details' → 'final_llm_analysis'")
                    data["final_llm_analysis"] = data.pop("final_llm_call_details")

                # Map webapp LLM field names to LlmKeywordAnalysis field names - Claude Generated
                def fix_llm_field_names(llm_dict):
                    """Fix field names in LLM analysis objects from webapp format"""
                    if not isinstance(llm_dict, dict):
                        return llm_dict

                    # Map provider → provider_used
                    if "provider" in llm_dict and "provider_used" not in llm_dict:
                        llm_dict["provider_used"] = llm_dict.pop("provider")

                    # Map model → model_used
                    if "model" in llm_dict and "model_used" not in llm_dict:
                        llm_dict["model_used"] = llm_dict.pop("model")

                    # Remove webapp-specific fields not in LlmKeywordAnalysis
                    for field in ["extracted_keywords", "token_count"]:
                        llm_dict.pop(field, None)

                    # Fill in missing required fields with defaults (webapp doesn't store these)
                    llm_dict.setdefault("task_name", "webapp-import")
                    llm_dict.setdefault("prompt_template", "")
                    llm_dict.setdefault("filled_prompt", "")
                    llm_dict.setdefault("temperature", 0.7)
                    llm_dict.setdefault("seed", None)

                    return llm_dict

                # Apply mappings to LLM analysis objects
                if isinstance(data.get("initial_llm_call_details"), dict):
                    data["initial_llm_call_details"] = fix_llm_field_names(data["initial_llm_call_details"])

                if isinstance(data.get("final_llm_analysis"), dict):
                    data["final_llm_analysis"] = fix_llm_field_names(data["final_llm_analysis"])

                # Remove webapp-specific fields not in KeywordAnalysisState
                removed_fields = []
                if "pipeline_metadata" in data:
                    data.pop("pipeline_metadata")
                    removed_fields.append("pipeline_metadata")
                if "final_keywords" in data:
                    data.pop("final_keywords")  # Redundant - extracted from final_llm_analysis
                    removed_fields.append("final_keywords")
                if "verification" in data:
                    data.pop("verification")  # Already nested inside final_llm_analysis
                    removed_fields.append("verification")

                if removed_fields:
                    logger.info(f"Removed webapp-specific fields: {', '.join(removed_fields)}")

                # Fill in missing required fields from webapp format - Claude Generated
                data.setdefault("search_suggesters_used", [])
                data.setdefault("initial_gnd_classes", [])
                data.setdefault("timestamp", datetime.now().isoformat())
                data.setdefault("pipeline_step_completed", "classification")
                data.setdefault("initial_llm_call_details", None)  # May not be in webapp export

            # Deep reconstruction of nested dataclass objects
            from ..core.data_models import SearchResult, LlmKeywordAnalysis

            # Reconstruct SearchResult objects
            if data.get("search_results"):
                reconstructed_search_results = []
                for item in data["search_results"]:
                    # Convert known list fields back to sets (e.g., gndid fields)
                    if "results" in item:
                        item["results"] = PipelineJsonManager.convert_lists_to_sets(item["results"])
                    reconstructed_search_results.append(SearchResult(**item))
                data["search_results"] = reconstructed_search_results

            # Reconstruct LlmKeywordAnalysis objects
            if data.get("initial_llm_call_details"):
                data["initial_llm_call_details"] = LlmKeywordAnalysis(**data["initial_llm_call_details"])

            if data.get("final_llm_analysis"):
                data["final_llm_analysis"] = LlmKeywordAnalysis(**data["final_llm_analysis"])

            if data.get("dk_llm_analysis"):
                data["dk_llm_analysis"] = LlmKeywordAnalysis(**data["dk_llm_analysis"])

            # Ensure list fields are actually lists - Claude Generated (Fix for string parsing bug)
            # This prevents "B, a, t, t, e, r, i, e" issue when JSON contains strings instead of lists
            if "initial_keywords" in data and isinstance(data["initial_keywords"], str):
                # Split comma-separated string back to list
                data["initial_keywords"] = [kw.strip() for kw in data["initial_keywords"].split(",") if kw.strip()]

            if data.get("final_llm_analysis") and hasattr(data["final_llm_analysis"], "extracted_gnd_keywords"):
                if isinstance(data["final_llm_analysis"].extracted_gnd_keywords, str):
                    kw_str = data["final_llm_analysis"].extracted_gnd_keywords
                    data["final_llm_analysis"].extracted_gnd_keywords = [kw.strip() for kw in kw_str.split(",") if kw.strip()]

            if "dk_classifications" in data and isinstance(data["dk_classifications"], str):
                data["dk_classifications"] = [dk.strip() for dk in data["dk_classifications"].split(",") if dk.strip()]

            return KeywordAnalysisState(**data)

        except FileNotFoundError:
            raise ValueError(f"Analysis state file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {file_path}. Error: {e}")
        except TypeError as e:
            raise ValueError(f"JSON structure incompatible with KeywordAnalysisState: {e}")
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
    def _filter_placeholder_titles(titles: List[str]) -> List[str]:
        """Filter out placeholder titles from classification cache - Claude Generated"""
        filtered = []
        for title in titles:
            # Skip placeholder titles that should not be shown to users or LLM
            if title.startswith("Cached Catalog Entry for RSN"):
                continue
            if title == "Cached Author":
                continue
            if not title.strip():  # Skip empty strings
                continue
            filtered.append(title)
        return filtered

    @staticmethod
    def format_dk_results_for_prompt(dk_results: List[Dict[str, Any]], max_results: int = 60) -> str:
        """Format DK/RVK results for LLM prompt or UI display - Claude Generated
        max_results caps total entries to prevent context-length overflow."""
        catalog_results = []
        for result in dk_results[:max_results]:
            # Handle keyword-centric format (fallback) - Claude Generated
            if "keyword" in result and "classifications" in result:
                classifications = result.get("classifications", [])
                for cl in classifications:
                    dk_code = cl.get("dk", "")
                    titles = cl.get("titles", [])
                    classification_type = cl.get("classification_type", "DK")

                    if dk_code:
                        # Filter placeholder titles - Claude Generated
                        filtered_titles = PipelineResultFormatter._filter_placeholder_titles(titles)
                        # Only show titles if available - Claude Generated
                        entry = f"{classification_type}: {dk_code}"
                        if filtered_titles:
                            title_text = " | ".join(filtered_titles[:5]) # Limit to 5 titles for keyword-centric
                            entry += f"\nBeispieltitel: {title_text}"
                        catalog_results.append(entry)
                continue

            # Handle aggregated format from _aggregate_dk_results
            if "dk" in result and "count" in result and "titles" in result:
                # Aggregated format with count and titles
                dk_code = result.get("dk", "")
                count = result.get("count", 0)
                titles = result.get("titles", [])
                matched_keywords = result.get("matched_keywords", [])
                classification_type = result.get("classification_type", "DK")
                source = result.get("source", "")
                label = result.get("label", "")
                ancestor_path = result.get("ancestor_path", "")
                register = result.get("register", [])
                rvk_validation_status = result.get("rvk_validation_status", "")
                validation_message = result.get("validation_message", "")

                if dk_code:
                    keyword_text = ", ".join(matched_keywords) if matched_keywords else "keine"
                    if classification_type == "RVK" and rvk_validation_status == "standard":
                        if source == "rvk_api":
                            source_text = "RVK API (autoritaetsbasiert)"
                        elif source == "rvk_gnd_index":
                            source_text = "RVK MarcXML-GND-Index (autoritaetsbasiert)"
                        else:
                            source_text = "Katalog (RVK-API-validiert)"
                        entry = f"{classification_type}: {dk_code}\nKeywords: {keyword_text}\nQuelle: {source_text}"
                        if label:
                            entry += f"\nBenennung: {label}"
                        if ancestor_path:
                            entry += f"\nFachpfad: {ancestor_path}"
                        if register:
                            entry += f"\nRegister: {', '.join(map(str, register[:6]))}"
                    elif source == "rvk_api":
                        entry = f"{classification_type}: {dk_code}\nKeywords: {keyword_text}\nQuelle: RVK API (autoritaetsbasiert)"
                        if label:
                            entry += f"\nBenennung: {label}"
                        if ancestor_path:
                            entry += f"\nFachpfad: {ancestor_path}"
                        if register:
                            entry += f"\nRegister: {', '.join(map(str, register[:6]))}"
                    elif classification_type == "RVK" and rvk_validation_status == "non_standard":
                        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nKeywords: {keyword_text}\nQuelle: Katalog (nicht-standardisiert/lokal)"
                        if validation_message:
                            entry += f"\nHinweis: {validation_message}"
                    elif classification_type == "RVK" and rvk_validation_status == "validation_error":
                        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nKeywords: {keyword_text}\nQuelle: Katalog (RVK-Validierung fehlgeschlagen)"
                        if validation_message:
                            entry += f"\nHinweis: {validation_message}"
                    else:
                        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nKeywords: {keyword_text}"
                    # Filter placeholder titles, cap at 5 to control prompt length - Claude Generated
                    filtered_titles = PipelineResultFormatter._filter_placeholder_titles(titles)
                    if filtered_titles:
                        title_text = " | ".join(filtered_titles[:5])
                        entry += f"\nBeispieltitel: {title_text}"
                    catalog_results.append(entry)
            elif "source_title" in result and "dk" in result:
                # Individual result format
                dk_code = result.get("dk", "")
                title = result.get("source_title", "")
                classification_type = result.get("classification_type", "DK")

                if dk_code and title:
                    entry = f"{classification_type}: {dk_code} | Titel: {title}"
                    catalog_results.append(entry)
            else:
                # Legacy format support
                title = result.get("title", "")
                subjects = result.get("subjects", [])
                dk_class = result.get("dk", [])
                rvk_class = result.get("rvk", [])

                if title:
                    entry = f"Titel: {title}"
                    if subjects:
                        entry += f" | Schlagworte: {', '.join(subjects)}"

                    # Handle DK
                    if isinstance(dk_class, str):
                        entry += f" | DK: {dk_class}"
                    elif isinstance(dk_class, list):
                        valid_dk = [dk for dk in dk_class if len(str(dk)) > 1]
                        if valid_dk:
                            entry += f" | DK: {', '.join(map(str, valid_dk))}"

                    # Handle RVK
                    if isinstance(rvk_class, str):
                        entry += f" | RVK: {rvk_class}"
                    elif isinstance(rvk_class, list):
                        entry += f" | RVK: {', '.join(map(str, rvk_class))}"

                    catalog_results.append(entry)

        return "\n".join(catalog_results)

    @staticmethod
    def parse_dk_results_from_text(text: str) -> List[Dict[str, Any]]:
        """Parse DK/RVK results back from formatted text into dictionary format - Claude Generated"""
        import re
        import logging
        logger = logging.getLogger(__name__)
        results = []

        if not text:
            return results

        # Regex for the main entry line: "[Type]: [DK Code] (Häufigkeit: [Count])"
        # We make it more flexible: optional spaces, optional prefix
        entry_pattern = r'(?:^|\n)\s*(DK|RVK):\s*([A-Z0-9.\-\s/]+?)\s*(?:\(Häufigkeit:\s*(\d+)\))'

        lines = text.split('\n')
        current_entry = None

        logger.debug(f"Parsing DK results from text ({len(text)} chars)")

        # First attempt: Try to parse the structured format with frequencies and titles
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a new entry line
            match = re.search(entry_pattern, line)
            if match:
                if current_entry:
                    results.append(current_entry)

                current_entry = {
                    "classification_type": match.group(1),
                    "dk": match.group(2).strip(),
                    "count": int(match.group(3)) if match.group(3) else 1,
                    "matched_keywords": [],
                    "titles": []
                }
            elif current_entry:
                # Check for Keywords line
                if "Keywords:" in line:
                    kw_text = line.split("Keywords:")[1].strip()
                    if kw_text and kw_text.lower() != "keine":
                        # Split by comma, semicolon or pipe
                        current_entry["matched_keywords"] = [k.strip() for k in re.split(r'[;,|]', kw_text) if k.strip()]

                # Check for Beispieltitel line
                elif "Beispieltitel:" in line:
                    title_text = line.split("Beispieltitel:")[1].strip()
                    if title_text:
                        current_entry["titles"] = [t.strip() for t in title_text.split("|") if t.strip()]

        # Don't forget the last entry
        if current_entry:
            results.append(current_entry)

        # Fallback: If no structured entries found, try simple comma-separated DK/RVK codes
        if not results:
            logger.debug("No structured DK entries found, trying simple fallback parser")
            # Look for things like "DK 614.7", "DK: 614.7", "RVK QZ 123"
            simple_pattern = r'(DK|RVK):?\s*([A-Z0-9.\-\s/]+?)(?=[,\n;]|$)'
            matches = re.finditer(simple_pattern, text)
            for match in matches:
                code = match.group(2).strip()
                # Basic validation: code shouldn't be too long and should have some digits/uppercase
                if 1 < len(code) < 30:
                    results.append({
                        "classification_type": match.group(1),
                        "dk": code,
                        "count": 1,
                        "matched_keywords": [],
                        "titles": []
                    })

        if results:
            logger.info(f"✅ Successfully parsed {len(results)} DK entries from text context")
        else:
            logger.warning("⚠️ No DK entries could be parsed from the provided text context")

        return results

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


def execute_input_extraction(
    llm_service,
    input_source: str,
    input_type: str = "auto",
    stream_callback: Optional[callable] = None,
    logger=None,
    **kwargs,
) -> Tuple[str, str, str]:
    """
    Extract text from various input sources (PDF, Image, Text) - Claude Generated
    
    Args:
        llm_service: LLM service instance for image OCR
        input_source: File path or text content
        input_type: "auto", "pdf", "image", "text", or "file"
        stream_callback: Callback for progress updates
        logger: Logger instance
        **kwargs: Additional parameters for LLM
        
    Returns:
        Tuple of (extracted_text, source_info, extraction_method)
    """
    import os
    import PyPDF2
    import tempfile
    from pathlib import Path
    
    if logger:
        logger.info(f"Starting input extraction: {input_source[:50]}... (type: {input_type})")
    
    # Auto-detect input type if not specified
    if input_type == "auto":
        if os.path.isfile(input_source):
            ext = Path(input_source).suffix.lower()
            if ext == ".pdf":
                input_type = "pdf" 
            elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"]:
                input_type = "image"
            else:
                input_type = "file"
        else:
            input_type = "text"
    
    # Handle different input types
    try:
        if input_type == "text":
            # Direct text input
            return input_source.strip(), "Direkter Text", "text"
            
        elif input_type == "file" and os.path.isfile(input_source):
            # Text file reading
            try:
                with open(input_source, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    filename = os.path.basename(input_source)
                    return text, f"Textdatei: {filename}", "file_read"
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252']:
                    try:
                        with open(input_source, 'r', encoding=encoding) as f:
                            text = f.read().strip()
                            filename = os.path.basename(input_source)
                            return text, f"Textdatei: {filename} ({encoding})", "file_read"
                    except UnicodeDecodeError:
                        continue
                raise Exception("Datei konnte nicht gelesen werden (Encoding-Problem)")
                
        elif input_type == "pdf":
            return _extract_from_pdf_pipeline(input_source, llm_service, stream_callback, logger)
            
        elif input_type == "image":
            return _extract_from_image_pipeline(input_source, llm_service, stream_callback, logger)
            
        else:
            raise Exception(f"Unbekannter Input-Typ: {input_type}")
            
    except Exception as e:
        error_msg = f"Input-Extraktion fehlgeschlagen: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise Exception(error_msg)


def _extract_from_pdf_pipeline(
    pdf_path: str, 
    llm_service,
    stream_callback: Optional[callable] = None,
    logger=None
) -> Tuple[str, str, str]:
    """Extract text from PDF with LLM fallback for pipeline - Claude Generated"""
    import os
    from pathlib import Path

    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 ist nicht installiert. Bitte mit 'pip install PyPDF2' installieren.")

    filename = os.path.basename(pdf_path)

    if stream_callback:
        stream_callback(f"📄 PDF wird gelesen: {filename}")

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text_parts = []
            
            for i, page in enumerate(reader.pages):
                if stream_callback:
                    stream_callback(f"📄 Seite {i+1} von {len(reader.pages)} wird verarbeitet...")
                page_text = page.extract_text()
                text_parts.append(page_text)
            
            full_text = "\\n\\n".join(text_parts).strip()
            
            # Text-Qualität prüfen
            quality_assessment = _assess_text_quality_pipeline(full_text)
            
            if quality_assessment['is_good']:
                # Direkter Text ist brauchbar
                source_info = f"PDF: {filename} ({len(reader.pages)} Seiten, Text extrahiert)"
                return full_text, source_info, "pdf_direct"
            else:
                # Text-Qualität schlecht, verwende LLM-OCR
                if stream_callback:
                    stream_callback(f"📄 Text-Qualität unzureichend ({quality_assessment['reason']}), starte OCR...")
                
                return _extract_pdf_with_llm_pipeline(pdf_path, filename, len(reader.pages), llm_service, stream_callback, logger)
                
    except Exception as e:
        raise Exception(f"PDF-Verarbeitung fehlgeschlagen: {str(e)}")


def _extract_pdf_with_llm_pipeline(
    pdf_path: str,
    filename: str, 
    page_count: int,
    llm_service,
    stream_callback: Optional[callable] = None,
    logger=None
) -> Tuple[str, str, str]:
    """Extract PDF using LLM OCR for pipeline - Claude Generated"""
    
    try:
        # Versuche pdf2image Import
        try:
            import pdf2image
        except ImportError:
            raise Exception("pdf2image-Bibliothek nicht verfügbar. Installieren Sie: pip install pdf2image")
        
        if stream_callback:
            stream_callback("📄 Konvertiere PDF für OCR-Analyse...")
        
        # Konvertiere PDF zu Bildern (max. erste 3 Seiten)
        images = pdf2image.convert_from_path(
            pdf_path,
            first_page=1,
            last_page=min(3, page_count),
            dpi=200
        )
        
        if not images:
            raise Exception("PDF konnte nicht zu Bildern konvertiert werden")
        
        # Speichere erstes Bild temporär
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            images[0].save(tmp_file.name, 'PNG')
            temp_image_path = tmp_file.name
        
        try:
            # Verwende LLM für OCR
            extracted_text, _, _ = _extract_from_image_pipeline(
                temp_image_path, 
                llm_service, 
                stream_callback, 
                logger
            )
            
            source_info = f"PDF (OCR): {filename} ({page_count} Seiten, per LLM analysiert)"
            return extracted_text, source_info, "pdf_llm_ocr"
            
        finally:
            # Cleanup temporäre Datei
            try:
                os.unlink(temp_image_path)
            except:
                pass
                
    except Exception as e:
        raise Exception(f"PDF-LLM-OCR fehlgeschlagen: {str(e)}")


def _extract_from_image_pipeline(
    image_path: str,
    llm_service, 
    stream_callback: Optional[callable] = None,
    logger=None
) -> Tuple[str, str, str]:
    """Extract text from image using LLM for pipeline - Claude Generated"""
    import uuid
    import os
    from pathlib import Path
    from ..llm.prompt_service import PromptService
    
    filename = os.path.basename(image_path)
    
    if stream_callback:
        stream_callback(f"🖼️ Analysiere Bild mit LLM: {filename}")
    
    try:
        # Lade Konfiguration und bestimme zuerst das tatsächliche Vision-Modell.
        # So können modellspezifische OCR-Prompts und Parameter geladen werden.
        from ..utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()

        # Check if a vision-capable model is configured for OCR.
        # The PyQt settings UI stores this under "vision", while some older/manual configs
        # may use the prompt-specific key "image_text_extraction".
        has_providers = False
        configured_task = None
        for task_name in ["image_text_extraction", "vision", "initialisation", "keywords"]:
            task_prefs = config.unified_config.task_preferences.get(task_name)
            if task_prefs and bool(task_prefs.model_priority):
                has_providers = True
                configured_task = task_name
                break

        if not has_providers:
            error_msg = (
                "❌ Kein Vision-Modell für Bilderkennung konfiguriert!\n"
                "Bitte in der Config file unter 'unified_config.task_preferences.vision' "
                "oder 'unified_config.task_preferences.image_text_extraction' "
                "einen 'model_priority'-Eintrag hinzufügen.\n"
                "Beispiel: 'model_priority': [{'provider_name': 'openai_compatible', 'model_name': 'gpt-4o'}]"
            )
            if logger:
                logger.error(error_msg)
            raise Exception(error_msg)
        elif logger:
            logger.info(f"Vision/OCR task configuration found via '{configured_task}'")

        # Bestimme besten Provider für Bilderkennung
        provider, model = _get_best_vision_provider_pipeline(llm_service, logger)

        if not provider:
            raise Exception("Kein Provider mit Bilderkennung verfügbar")

        prompts_path = config.system_config.prompts_path
        prompt_service = PromptService(prompts_path, logger)

        # Load OCR prompt for the actual selected model, not the generic default.
        prompt_config_data = prompt_service.get_prompt_config(
            task="image_text_extraction",
            model=model or "default"
        )

        if not prompt_config_data:
            raise Exception("OCR-Prompt 'image_text_extraction' nicht gefunden in prompts.json")

        # Konvertiere PromptConfigData zu Dictionary für Kompatibilität
        prompt_config = {
            'prompt': prompt_config_data.prompt,
            'system': prompt_config_data.system or '',
            'temperature': prompt_config_data.temp,
            'top_p': prompt_config_data.p_value,
            'seed': prompt_config_data.seed
        }

        if stream_callback:
            stream_callback(f"🖼️ Verwende {provider} ({model}) für Bilderkennung...")
        
        request_id = str(uuid.uuid4())

        # LLM-Aufruf für Bilderkennung mit Streaming - Claude Generated
        response = llm_service.generate_response(
            provider=provider,
            model=model,
            prompt=prompt_config['prompt'],
            system=prompt_config.get('system', ''),
            request_id=request_id,
            temperature=float(prompt_config.get('temperature', 0.1)),
            p_value=float(prompt_config.get('top_p', 0.1)),
            seed=prompt_config.get('seed'),
            image=image_path,
            stream=True,  # Enable streaming for live feedback - Claude Generated
            output_format="xml",  # OCR expects raw text, not JSON-mode structured output.
        )

        # Handle streaming response with live callback - Claude Generated
        extracted_text = ""
        if hasattr(response, "__iter__") and not isinstance(response, str):
            # Generator response with live streaming
            text_parts = []
            for chunk in response:
                chunk_text = ""
                if isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                elif hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                else:
                    chunk_text = str(chunk)

                # Send to live callback if available - Claude Generated
                if chunk_text and stream_callback:
                    stream_callback(chunk_text)

                text_parts.append(chunk_text)
            extracted_text = "".join(text_parts)
        else:
            extracted_text = str(response)

        error_markers = (
            "error with ",
            "error code:",
            "invalid_request_error",
            "unsupported_value",
            "unsupported value:",
        )
        lowered_response = extracted_text.strip().lower()
        if lowered_response and any(marker in lowered_response for marker in error_markers):
            raise Exception(extracted_text.strip())
        
        # Bereinige LLM-Output
        extracted_text = _clean_ocr_output_pipeline(extracted_text)
        
        if not extracted_text.strip():
            raise Exception("LLM konnte keinen Text im Bild erkennen")
        
        source_info = f"Bild (OCR): {filename}"
        return extracted_text, source_info, "image_llm_ocr"
        
    except Exception as e:
        raise Exception(f"Bild-LLM-OCR fehlgeschlagen: {str(e)}")


def _assess_text_quality_pipeline(text: str) -> Dict[str, Any]:
    """Assess quality of extracted PDF text for pipeline - Claude Generated"""
    if not text or len(text.strip()) == 0:
        return {'is_good': False, 'reason': 'Kein Text gefunden'}
    
    char_count = len(text)
    word_count = len(text.split())
    
    if char_count < 50:
        return {'is_good': False, 'reason': 'Text zu kurz'}
    
    if word_count > 0:
        avg_word_length = char_count / word_count
        if avg_word_length < 2 or avg_word_length > 20:
            return {'is_good': False, 'reason': 'Ungewöhnliche Wortlängen'}
    
    special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]') / len(text)
    if special_char_ratio > 0.3:
        return {'is_good': False, 'reason': 'Zu viele Sonderzeichen'}
    
    lines_with_content = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
    if len(lines_with_content) < max(1, word_count // 20):
        return {'is_good': False, 'reason': 'Text fragmentiert'}
        
    return {'is_good': True, 'reason': 'Text-Qualität ausreichend'}


def _get_best_vision_provider_pipeline(llm_service, logger=None) -> Tuple[Optional[str], Optional[str]]:
    """Get best available provider for vision tasks using SmartProviderSelector - Claude Generated"""
    try:
        from .smart_provider_selector import SmartProviderSelector
        from .config_models import TaskType
        from .config_manager import ConfigManager

        # Initialize ConfigManager for task_preferences access - Claude Generated
        config_manager = ConfigManager()
        selector = SmartProviderSelector(config_manager)
        selection = selector.select_provider(
            task_type=TaskType.VISION,
            prefer_fast=False,
            task_name="image_text_extraction"  # Task preferences define explicit provider - no capability filtering needed - Claude Generated
        )

        if logger:
            logger.info(f"SmartProviderSelector chose {selection.provider} with {selection.model} for vision task (fallback_used: {selection.fallback_used})")
        
        return selection.provider, selection.model
        
    except Exception as e:
        if logger:
            logger.warning(f"SmartProviderSelector failed, falling back to legacy selection: {e}")
        
        # Legacy fallback for compatibility
        vision_providers = [
            ("gemini", ["gemini-2.0-flash", "gemini-1.5-flash"]),
            ("openai", ["gpt-4o", "gpt-4-vision-preview"]),
            ("anthropic", ["claude-3-5-sonnet", "claude-3-opus"]),
            ("ollama", ["llava", "minicpm-v", "cogito:32b"])
        ]
        
        try:
            available_providers = llm_service.get_available_providers()
            
            for provider_name, preferred_models in vision_providers:
                if provider_name in available_providers:
                    try:
                        available_models = llm_service.get_available_models(provider_name)
                        
                        for preferred_model in preferred_models:
                            if preferred_model in available_models:
                                return provider_name, preferred_model
                        
                        if available_models:
                            return provider_name, available_models[0]
                            
                    except Exception as e:
                        if logger:
                            logger.warning(f"Error checking models for {provider_name}: {e}")
                        continue
            
            return None, None
            
        except Exception as e:
            if logger:
                logger.error(f"Error determining best vision provider: {e}")
            return None, None


def _clean_ocr_output_pipeline(text: str) -> str:
    """Clean OCR output from common LLM artifacts for pipeline - Claude Generated"""
    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Überspringe typische LLM-Metakommentare
        if any(phrase in line.lower() for phrase in [
            'hier ist der text',
            'der text lautet',
            'ich kann folgenden text erkennen',
            'das bild enthält folgenden text',
            'extracted text:',
            'ocr result:',
            'text erkannt:',
            'gefundener text:'
        ]):
            continue

        if line:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


class AnalysisPersistence:
    """
    Unified persistence interface for KeywordAnalysisState with Qt dialog integration.
    Eliminates code duplication across GUI components by providing a single API.
    Claude Generated
    """

    @staticmethod
    def save_with_dialog(
        state: "KeywordAnalysisState",
        parent_widget=None,
        default_filename: str = None
    ) -> Optional[str]:
        """
        Save KeywordAnalysisState with Qt file dialog.

        Args:
            state: KeywordAnalysisState object to save
            parent_widget: Qt parent widget for dialog (optional)
            default_filename: Default filename suggestion (optional)

        Returns:
            File path if saved successfully, None if cancelled or failed

        Claude Generated
        """
        try:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox
        except ImportError:
            raise ImportError("PyQt6 required for GUI dialogs. Use PipelineJsonManager directly for CLI.")

        # Generate default filename if not provided
        if not default_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f"analysis_state_{timestamp}.json"

        # Resolve full path using configured autosave directory
        from .pipeline_defaults import get_autosave_dir
        default_path = str(get_autosave_dir() / default_filename)

        # Open save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Analyse-Zustand speichern",
            default_path,
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return None  # User cancelled

        try:
            # Use PipelineJsonManager for actual save
            PipelineJsonManager.save_analysis_state(state, file_path)

            # Success notification
            if parent_widget:
                QMessageBox.information(
                    parent_widget,
                    "Erfolg",
                    f"Analyse-Zustand erfolgreich gespeichert:\n{file_path}"
                )

            return file_path

        except Exception as e:
            # Error notification
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Fehler",
                    f"Fehler beim Speichern:\n\n{str(e)}"
                )
            raise

    @staticmethod
    def load_with_dialog(parent_widget=None) -> Optional["KeywordAnalysisState"]:
        """
        Load KeywordAnalysisState with Qt file dialog.

        Args:
            parent_widget: Qt parent widget for dialog (optional)

        Returns:
            KeywordAnalysisState object if loaded successfully, None if cancelled or failed

        Claude Generated
        """
        try:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox
        except ImportError:
            raise ImportError("PyQt6 required for GUI dialogs. Use PipelineJsonManager directly for CLI.")

        # Resolve start directory using configured autosave directory
        from .pipeline_defaults import get_autosave_dir

        # Open load dialog
        file_path, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Analyse-Zustand laden",
            str(get_autosave_dir()),
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return None  # User cancelled

        try:
            # Use PipelineJsonManager for actual load
            state = PipelineJsonManager.load_analysis_state(file_path)

            # Success notification
            if parent_widget:
                QMessageBox.information(
                    parent_widget,
                    "Erfolg",
                    f"Analyse-Zustand erfolgreich geladen:\n{file_path}"
                )

            return state

        except Exception as e:
            # Error notification
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Fehler",
                    f"Fehler beim Laden:\n\n{str(e)}"
                )
            return None
