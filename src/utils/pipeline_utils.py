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
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from .suggesters.meta_suggester import SuggesterType
from ..core.processing_utils import (
    extract_keywords_from_response,
    extract_gnd_system_from_response,
)
from .smart_provider_selector import SmartProviderSelector, TaskType


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
        """Intelligent provider/model resolution with proper fallback chain - Claude Generated"""

        # 1. Explicit parameters (highest priority)
        if provider and model:
            if self.logger:
                self.logger.info(f"Using explicit provider/model: {provider}/{model}")
            return provider, model

        # 2. SmartProviderSelector (when available)
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

                # Use SmartProvider selection if no explicit provider/model given
                final_provider = provider or selection.provider
                final_model = model or selection.model

                if self.logger:
                    self.logger.info(f"SmartProvider selection: {final_provider}/{final_model} (task_type={task_type}, prefer_fast={prefer_fast})")

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
        # Use first available Ollama provider instead of non-existent "ollama" - Claude Generated
        fallback_provider = provider or self._get_first_available_ollama_provider()

        # Task-specific model defaults as absolute last resort
        task_defaults = {
            "text": "cogito:14b" if prefer_fast else "cogito:32b",
            "classification": "cogito:32b",
            "vision": "cogito:32b"
        }
        fallback_model = model or task_defaults.get(task_type.lower(), "cogito:14b")

        if self.logger:
            self.logger.warning(f"Using system fallback defaults: {fallback_provider}/{fallback_model} (no SmartProvider or Config available)")

        return fallback_provider, fallback_model

    def _get_first_available_ollama_provider(self) -> str:
        """Get the first available Ollama provider name from config - Claude Generated"""
        try:
            if self.smart_selector and hasattr(self.smart_selector, 'config'):
                config = self.smart_selector.config
                # Get unified config and find first enabled Ollama provider
                if hasattr(config, 'unified_config') and config.unified_config:
                    for provider in config.unified_config.providers:
                        if provider.provider_type == "ollama" and provider.enabled:
                            return provider.name
                # Fallback to legacy config if available
                elif hasattr(config, 'llm'):
                    enabled_ollama = config.unified_config.get_enabled_ollama_providers()
                    if enabled_ollama:
                        return enabled_ollama[0].name

            # Last resort: hardcoded fallback to localhost
            return "localhost"

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to get Ollama provider: {e}")
            return "localhost"

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
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis]:
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
            error_msg = f"Initial keyword extraction failed: {task_state.analysis_result.full_text}"
            if self.logger:
                self.logger.error(f"💥 PIPELINE_FAILURE: {error_msg}")
            raise ValueError(error_msg)

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
        catalog_token: str = None,
        catalog_search_url: str = None,
        catalog_details_url: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute GND search step with automatic catalog detection - Claude Generated"""

        if suggesters is None:
            suggesters = ["lobid", "swb"]
            
            # Add catalog if available (no auto-detection, explicit configuration)
            # Catalog will be added via suggesters parameter in pipeline

        # Create SearchCLI instance with catalog parameters
        search_cli = SearchCLI(
            self.cache_manager,
            catalog_token=catalog_token or "",
            catalog_search_url=catalog_search_url or "",
            catalog_details_url=catalog_details_url or ""
        )

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

        # Post-process catalog results: validate subjects against cache and SWB
        if "catalog" in suggesters:
            search_results = self._validate_catalog_subjects(
                search_results, stream_callback
            )

        # Stream completion info if callback provided
        if stream_callback:
            total_hits = sum(len(results) for results in search_results.values())
            stream_callback(
                f"Suche abgeschlossen: {total_hits} Treffer in {len(search_results)} Kategorien\n",
                "search",
            )

        return search_results

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
                for i, subject in enumerate(unknown_subjects[:10]):  # Show first 10
                    self.logger.info(f"  {i+1}. '{subject}'")
                if len(unknown_subjects) > 10:
                    self.logger.info(f"  ... und {len(unknown_subjects) - 10} weitere")
            
            try:
                # Use SWB suggester for validation
                swb_search_cli = SearchCLI(self.cache_manager)
                
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
            for i, unmatched in enumerate(unmatched_swb_subjects[:5]):  # Show first 5
                self.logger.warning(f"  {i+1}. {unmatched}")
            if len(unmatched_swb_subjects) > 5:
                self.logger.warning(f"  ... und {len(unmatched_swb_subjects) - 5} weitere")
        
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
        keyword_chunking_threshold: int = 500,
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

        # Prepare GND search results for prompt
        gnd_keywords_text = ""
        gnd_compliant_keywords = []

        for results in search_results.values():
            for keyword, data in results.items():
                gnd_ids = data.get("gndid", set())
                for gnd_id in gnd_ids:
                    # Claude Generated - Use GND title with optional synonym expansion
                    gnd_title = self.cache_manager.get_gnd_title_by_id(gnd_id)
                    if gnd_title:
                        # Check if we should expand synonyms and if this title is relevant
                        if expand_synonyms:
                            # Check if this title already appears in our keyword list
                            title_in_keywords = any(gnd_title.lower() in kw.lower() for kw in [keyword] + list(results.keys()))
                            
                            if title_in_keywords:
                                synonyms = self.cache_manager.get_gnd_synonyms_by_id(gnd_id)
                                if synonyms:
                                    # Format with synonyms: "Limnologie (Seenkunde; Süßwasserbiologie) (GND-ID: 4035769-7)"
                                    synonym_text = "; ".join(synonyms)
                                    formatted_keyword = f"{gnd_title} ({synonym_text}) (GND-ID: {gnd_id})"
                                else:
                                    # No synonyms available
                                    formatted_keyword = f"{gnd_title} (GND-ID: {gnd_id})"
                            else:
                                # Title not in keywords, don't expand synonyms
                                formatted_keyword = f"{gnd_title} (GND-ID: {gnd_id})"
                        else:
                            # No synonym expansion
                            formatted_keyword = f"{gnd_title} (GND-ID: {gnd_id})"
                            
                        gnd_keywords_text += formatted_keyword + "\n"
                        gnd_compliant_keywords.append(formatted_keyword)
                    else:
                        # Fallback to original keyword if GND title not found
                        formatted_keyword = f"{keyword} (GND-ID: {gnd_id})"
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
        **kwargs,
    ) -> Tuple[List[str], List[str], LlmKeywordAnalysis]:
        """Execute single keyword analysis without chunking - Claude Generated"""

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
            raise ValueError(
                f"Final keyword analysis failed: {task_state.analysis_result.full_text}"
            )

        # Extract final keywords and classes
        extracted_keywords_all, extracted_keywords_exact = (
            extract_keywords_from_descriptive_text(
                task_state.analysis_result.full_text, gnd_compliant_keywords
            )
        )
        
        # Apply deduplication to ensure no duplicate keywords - Claude Generated
        extracted_keywords_exact = self._deduplicate_keywords(
            [extracted_keywords_exact],  # Wrap in list for consistency with chunked version
            gnd_compliant_keywords
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
            extracted_gnd_keywords=extracted_keywords_exact,
            extracted_gnd_classes=extracted_gnd_classes,
        )

        return extracted_keywords_exact, extracted_gnd_classes, llm_analysis

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
                f"\n--- Finale Konsolidierung ---\n", kwargs.get("step_id", "keywords")
            )
            total_chunk_keywords = sum(len(r[0]) for r in all_chunk_results)
            stream_callback(
                f"Deduplizierte Keywords: {len(deduplicated_keywords)} aus {total_chunk_keywords} chunk-results\n",
                kwargs.get("step_id", "keywords"),
            )

            # Show current deduplicated list for debugging
            if deduplicated_keywords:
                preview = deduplicated_keywords[:5]
                if len(deduplicated_keywords) > 5:
                    preview_text = f"{', '.join([kw.split(' (GND-ID:')[0] for kw in preview])}... (+{len(deduplicated_keywords)-5} weitere)"
                else:
                    preview_text = ", ".join(
                        [kw.split(" (GND-ID:")[0] for kw in preview]
                    )
                stream_callback(
                    f"Deduplizierte Liste: {preview_text}\n",
                    kwargs.get("step_id", "keywords"),
                )

        # Execute final consolidation request with deduplicated results
        final_keywords_text = "\n".join(deduplicated_keywords)
        final_single_result = self._execute_single_keyword_analysis(
            original_abstract=original_abstract,
            gnd_keywords_text=final_keywords_text,
            gnd_compliant_keywords=deduplicated_keywords,
            model=model,
            provider=provider,
            task=task,  # Use original task for final analysis
            stream_callback=stream_callback,
            mode=mode,
            **kwargs,
        )

        # Extract final keywords with enhanced recognition
        final_keywords = self._extract_keywords_enhanced(
            final_single_result[2].response_full_text,
            deduplicated_keywords,
            stream_callback,
            chunk_id="Finale Auswahl",
        )

        # Update the LlmKeywordAnalysis to include chunk information
        final_llm_analysis = LlmKeywordAnalysis(
            task_name=f"{task} (chunked)",
            model_used=model,
            provider_used=provider,
            prompt_template=final_single_result[2].prompt_template,
            filled_prompt=final_single_result[2].filled_prompt,
            temperature=kwargs.get("temperature", 0.7),
            seed=kwargs.get("seed", 0),
            response_full_text=f"CHUNKED ANALYSIS ({len(chunks)} chunks):\n\n"
            + "\n\n---CHUNK SEPARATOR---\n\n".join(combined_responses)
            + f"\n\n---FINAL CONSOLIDATION---\n\n{final_single_result[2].response_full_text}",
            extracted_gnd_keywords=final_keywords,  # Use enhanced extraction results
            extracted_gnd_classes=final_single_result[1],
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

                    # Check for duplicates
                    word_lower = word.lower()
                    if word_lower not in seen_words and gnd_id not in seen_gnd_ids:
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

        # Parse reference keywords to create lookup dictionaries
        word_to_full = {}  # clean_word_lower -> full_formatted_keyword
        gnd_to_full = {}  # gnd_id -> full_formatted_keyword

        for keyword in reference_keywords:
            # Parse format: "Keyword (GND-ID: 123456789)"
            match = re.match(r"^(.+?)\s*\(GND-ID:\s*([^)]+)\)$", keyword.strip())
            if match:
                word = match.group(1).strip()
                gnd_id = match.group(2).strip()
                word_to_full[word.lower()] = keyword
                gnd_to_full[gnd_id] = keyword

        # Search for matches in response text
        found_keywords = []
        response_lower = response_text.lower()

        # Method 1: Search for exact keyword strings
        for clean_word, full_keyword in word_to_full.items():
            if clean_word in response_lower:
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
                # Show first few keywords for debugging
                preview = found_keywords[:5]
                if len(found_keywords) > 5:
                    preview_text = f"{', '.join([kw.split(' (GND-ID:')[0] for kw in preview])}... (+{len(found_keywords)-5} weitere)"
                else:
                    preview_text = ", ".join(
                        [kw.split(" (GND-ID:")[0] for kw in preview]
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
        dk_frequency_threshold: int = 10,  # Claude Generated - Only pass classifications with >= N occurrences
        mode=None,  # <--- NEUER PARAMETER: Pipeline mode for PromptService
        **kwargs,
    ) -> List[str]:
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
            List of selected DK classification codes
            
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
            return []

        if stream_callback:
            stream_callback(f"Starte DK-Klassifikation mit {len(dk_search_results)} Katalog-Einträgen\n", "dk_classification")

        # Filter results by frequency threshold - Claude Generated
        filtered_results = []
        low_frequency_count = 0
        
        for result in dk_search_results:
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

        # Format catalog results for LLM prompt with aggregated data
        catalog_results = []
        for result in filtered_results:
            # Handle aggregated format from _aggregate_dk_results
            if "dk" in result and "count" in result and "titles" in result:
                # Aggregated format with count and titles
                dk_code = result.get("dk", "")
                count = result.get("count", 0)
                titles = result.get("titles", [])
                keywords = result.get("keywords", [])
                classification_type = result.get("classification_type", "DK")
                avg_confidence = result.get("avg_confidence", 0.0)
                
                # Format with count and sample titles
                if dk_code:
                    # Show up to 3 sample titles to keep prompt manageable
                    sample_titles = titles # nope lets go with all
                    title_text = " | ".join(sample_titles)
                    #if len(titles) > 3:
                    #    title_text += f" | ... (und {len(titles) - 3} weitere)"
                    
                    entry = f"{classification_type}: {dk_code} (Häufigkeit: {count}) | Beispieltitel: {title_text}"
                    catalog_results.append(entry)
            elif "source_title" in result and "dk" in result:
                # Individual result format from extract_dk_classifications_for_keywords
                dk_code = result.get("dk", "")
                title = result.get("source_title", "")
                keyword = result.get("keyword", "")
                classification_type = result.get("classification_type", "DK")
                
                # Format individual classification
                if dk_code and title:
                    entry = f"{classification_type}: {dk_code} | Titel: {title}"
                    catalog_results.append(entry)
            else:
                # Legacy format support
                title = result.get("title", "")
                subjects = result.get("subjects", [])
                dk_classifications = result.get("dk", [])
                rvk_classifications = result.get("rvk", [])
                
                # Format legacy results
                if title:
                    entry = f"Titel: {title}"
                    if subjects:
                        entry += f" | Schlagworte: {', '.join(subjects)}"
                    if dk_classifications:
                        if isinstance(dk_classifications, str):
                            entry += f" | DK: {dk_classifications}"
                        elif isinstance(dk_classifications, list):
                            valid_dk = [dk for dk in dk_classifications if len(str(dk)) > 1]
                            if valid_dk:
                                entry += f" | DK: {', '.join(map(str, valid_dk))}"
                    if rvk_classifications:
                        if isinstance(rvk_classifications, str):
                            entry += f" | RVK: {rvk_classifications}"
                        elif isinstance(rvk_classifications, list):
                            entry += f" | RVK: {', '.join(map(str, rvk_classifications))}"
                    
                    catalog_results.append(entry)

        # Prepare data for LLM classification
        catalog_text = "\n".join(catalog_results)
        
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
                task="dk_class",
                model=model,
                provider=provider,
                stream_callback=alima_stream_callback,
                mode=mode,  # <--- NEUER PARAMETER: Pass mode to AlimaManager
                **alima_kwargs,
            )

            if task_state.status == "failed":
                if stream_callback:
                    stream_callback(f"LLM-Klassifikation fehlgeschlagen: {task_state.analysis_result.full_text}\n", "dk_classification")
                return []

            # Extract DK classifications from LLM response
            response_text = task_state.analysis_result.full_text
            dk_classifications = self._extract_dk_from_response(response_text)
            
            if stream_callback:
                stream_callback(f"DK-Klassifikation abgeschlossen: {len(dk_classifications)} DK-Codes extrahiert\n", "dk_classification")
                
            return dk_classifications
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM DK classification failed: {e}")
            if stream_callback:
                stream_callback(f"LLM-Klassifikation-Fehler: {str(e)}\n", "dk_classification")
            return []

    def _extract_dk_from_response(self, response_text: str) -> List[str]:
        """Extract DK and RVK classifications from LLM response - Claude Generated"""
        import re
        
        classification_codes = []
        
        # Look for DK patterns like "610.3", "004.5", etc.
        dk_pattern = r'\b\d{1,3}(?:\.\d+)*\b'
        dk_matches = re.findall(dk_pattern, response_text)
        
        # Filter to keep only valid DK codes (usually have 2-3 digits + optional decimal)
        for match in dk_matches:
            if len(match.split('.')[0]) >= 2:  # At least 2 digits before decimal
                classification_codes.append(f"DK {match}")
        
        # Look for RVK patterns like "Q12", "QC 130", "QB 910", etc.
        rvk_pattern = r'\b[A-Z]{1,2}\s*\d{1,4}(?:\s*[A-Z]*)?'
        rvk_matches = re.findall(rvk_pattern, response_text)
        
        # Filter RVK codes (especially Q* codes for economics)
        for match in rvk_matches:
            match_clean = match.strip()
            # Focus on Q codes for economics, but allow other RVK codes too
            if match_clean.startswith('Q') or len(match_clean) >= 3:
                classification_codes.append(f"RVK {match_clean}")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(classification_codes))

    def execute_dk_search(
        self,
        keywords: List[str],
        stream_callback: Optional[callable] = None,
        max_results: int = 20,
        catalog_token: str = None,
        catalog_search_url: str = None,
        catalog_details_url: str = None,
    ) -> List[Dict[str, Any]]:
        """Execute catalog search for DK classification data - Claude Generated"""

        # Use provided catalog configuration or skip
        if not catalog_token or not catalog_token.strip():
            if stream_callback:
                stream_callback("Kein Katalog-Token verfügbar - DK-Suche übersprungen\n", "dk_search")
            return []

        # Initialize BiblioExtractor
        try:
            from .clients.biblio_client import BiblioClient
            
            extractor = BiblioClient(
                token=catalog_token, 
                debug=self.logger.level <= 10 if self.logger else False
            )
            
            # Set URLs if available
            if catalog_search_url:
                extractor.SEARCH_URL = catalog_search_url
            if catalog_details_url:
                extractor.DETAILS_URL = catalog_details_url
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"BiblioExtractor initialization failed: {e}")  
            if stream_callback:
                stream_callback(f"BiblioExtractor-Fehler: {str(e)}\n", "dk_search")
            return []

        # Convert keywords to simple strings if they contain GND-IDs
        clean_keywords = []
        for keyword in keywords:
            if "(GND-ID:" in keyword:
                # Extract just the keyword part before (GND-ID:...)
                clean_keyword = keyword.split("(GND-ID:")[0].strip()
                clean_keywords.append(clean_keyword)
            else:
                clean_keywords.append(keyword)

        if stream_callback:
            stream_callback(f"Suche Katalog-Einträge für {len(clean_keywords)} Keywords (max {max_results})\n", "dk_search")

        # Execute catalog search
        try:
            dk_search_results = extractor.extract_dk_classifications_for_keywords(
                keywords=clean_keywords,
                max_results=max_results,
            )
            
            if stream_callback:
                stream_callback(f"DK-Suche abgeschlossen: {len(dk_search_results)} Katalog-Einträge gefunden\n", "dk_search")
                
            return dk_search_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"DK catalog search failed: {e}")
            if stream_callback:
                stream_callback(f"DK-Suche-Fehler: {str(e)}\n", "dk_search")
            return []


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
    def convert_lists_to_sets(obj):
        """Convert known list fields back to sets after JSON loading - Claude Generated"""
        if isinstance(obj, dict):
            # Known set fields in search results (e.g., gndid fields)
            result = {}
            for key, value in obj.items():
                if key == "gndid" and isinstance(value, list):
                    # Convert gndid lists back to sets
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
    import PyPDF2
    from pathlib import Path
    
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
        # Lade OCR-Prompt
        import os
        prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts.json')
        prompt_service = PromptService(prompts_path, logger)
        
        # Verwende image_text_extraction Task
        prompt_config_data = prompt_service.get_prompt_config(
            task="image_text_extraction",
            model="default"
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
        
        # Bestimme besten Provider für Bilderkennung
        provider, model = _get_best_vision_provider_pipeline(llm_service, logger)
        
        if not provider:
            raise Exception("Kein Provider mit Bilderkennung verfügbar")
        
        if stream_callback:
            stream_callback(f"🖼️ Verwende {provider} ({model}) für Bilderkennung...")
        
        request_id = str(uuid.uuid4())
        
        # LLM-Aufruf für Bilderkennung
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
            stream=False
        )
        
        # Handle verschiedene Response-Typen
        extracted_text = ""
        if hasattr(response, "__iter__") and not isinstance(response, str):
            # Generator response
            text_parts = []
            for chunk in response:
                if isinstance(chunk, str):
                    text_parts.append(chunk)
                elif hasattr(chunk, 'text'):
                    text_parts.append(chunk.text)
                elif hasattr(chunk, 'content'):
                    text_parts.append(chunk.content)
                else:
                    text_parts.append(str(chunk))
            extracted_text = "".join(text_parts)
        else:
            extracted_text = str(response)
        
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
        from .smart_provider_selector import SmartProviderSelector, TaskType
        
        selector = SmartProviderSelector()
        selection = selector.select_provider(
            task_type=TaskType.VISION,
            required_capabilities=["vision"],
            prefer_fast=False
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
