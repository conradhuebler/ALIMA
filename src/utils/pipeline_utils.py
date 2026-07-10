"""
Pipeline Utils - Shared logic for CLI and GUI pipeline implementations
Claude Generated - Abstracts common pipeline operations and utilities
"""

import json
import logging
import os
import re
import time
import html
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional, Set
import dataclasses
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
from ..core.search import SearchCapability, enabled_gnd_provider_ids, providers_for_capability
from ..core.processing_utils import (
    extract_keywords_from_response,
    extract_gnd_system_from_response,
    extract_title_from_response,  # For LLM title extraction - Claude Generated
    extract_missing_concepts_from_response,  # For iterative refinement - Claude Generated
)
from .smart_provider_selector import SmartProviderSelector
from .config_models import TaskType
from .pipeline_defaults import DEFAULT_DK_MAX_RESULTS, DEFAULT_DK_FREQUENCY_THRESHOLD
from .chunking import split_into_equal_chunks
# Input extraction moved to pipeline_input.py — re-exported for backward compat
from .pipeline_input import (
    execute_input_extraction,
    _extract_from_pdf_pipeline,
    _extract_pdf_with_llm_pipeline,
    _extract_from_image_pipeline,
    _assess_text_quality_pipeline,
    _get_best_vision_provider_pipeline,
    _clean_ocr_output_pipeline,
)
# Keyword/GND helpers moved to gnd_keyword_utils.py — re-exported (used by
# PipelineStepExecutor below and by external importers)
from .gnd_keyword_utils import (
    verify_keywords_against_gnd_pool,
    extract_keywords_from_descriptive_text,
    extract_classes_from_descriptive_text,
    canonicalize_keyword,
    extract_gnd_id,
    canonicalize_rvk_notation,
    is_plausible_nonstandard_rvk,
    deduplicate_canonical_keywords,
)
# Pure text/display/title helpers moved to pipeline_text_utils.py — re-exported
from .pipeline_text_utils import (
    repair_display_text,
    sanitize_for_filename,
    build_working_title,
    extract_source_identifier,
    flatten_keyword_centric_results,
)
# PipelineResultFormatter moved to pipeline_formatters.py — re-exported
from .pipeline_formatters import PipelineResultFormatter
# Persistence moved to pipeline_persistence.py — re-exported
from .pipeline_persistence import (
    PipelineJsonManager,
    AnalysisPersistence,
    export_analysis_state_to_file,
)


# ----------------------------------------------------------------------
# P-C: classic-pipeline tool-call shim — emits one ``tool.called`` /
# ``tool.result`` pair per PipelineStepExecutor step so the chat log
# shows classic-pipeline work the same way as agentic-pipeline work.
# Subscribers can filter by ``tool.name.startswith("classic.")``.
# ----------------------------------------------------------------------

def _emit_classic_tool_call(step_id: str, args: Dict[str, Any]) -> str:
    """Emit a ``tool.called`` event for a classic step. Returns the id."""
    from ..core.agents.sub_agents.caching_tool_registry import make_tool_call_id
    tc_id = make_tool_call_id()
    try:
        from ..core.state_bus import AlimaStateBus
        AlimaStateBus().emit_event("tool.called", {
            "name": f"classic.{step_id}",
            "arguments": args,
            "id": tc_id,
        })
    except Exception:
        # Bus is a monitoring channel — never break pipeline execution.
        pass
    return tc_id


def _emit_classic_tool_result(
    tc_id: str, step_id: str, status: str = "ok"
) -> None:
    """Emit a ``tool.result`` event for a classic step."""
    try:
        from ..core.state_bus import AlimaStateBus
        AlimaStateBus().emit_event("tool.result", {
            "name": f"classic.{step_id}",
            "result": status,
            "id": tc_id,
            "status": status,
        })
    except Exception:
        # Bus is a monitoring channel — never break pipeline execution - Claude Generated
        logging.getLogger(__name__).debug("tool.result bus emit failed", exc_info=True)


def _run_classic_step(
    step_id: str, args: Dict[str, Any], fn, *pos, **kwargs
):
    """Run a PipelineStepExecutor step with classic-pipeline bus emissions.

    P-C: emits one ``tool.called`` / ``tool.result`` pair around ``fn`` so the
    chat log shows classic-pipeline work identically to agentic-pipeline
    tool calls. The result event reports ``status='error'`` automatically
    if ``fn`` raises; the exception is re-raised unchanged.

    Used by :meth:`PipelineStepExecutor.execute_complete_pipeline` to wrap
    each of the 5 (or 6) step invocations in one place — keeps the
    individual ``execute_*`` methods free of try/finally noise.
    """
    tc_id = _emit_classic_tool_call(step_id, args)
    try:
        result = fn(*pos, **kwargs)
    except Exception:
        _emit_classic_tool_result(tc_id, step_id, "error")
        raise
    _emit_classic_tool_result(tc_id, step_id, "ok")
    return result


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

    def _alima_config_for_cache(self):
        """Load the AlimaConfig (or None) for per-plugin cache-setting lookups - Claude Generated.

        Used to gate raw-response caching of direct lookup calls (e.g. the RVK anchor)
        on the plugin's ``cache_responses`` setting. Best-effort; never raises.
        """
        cm = getattr(self, "config_manager", None)
        if cm is None:
            return None
        try:
            return cm.load_config()
        except Exception:
            return None

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

        # An unparseable LLM response must fail the step instead of silently
        # continuing the pipeline with 0 keywords ("success" with empty result
        # leads to wrong cataloguing decisions) - Claude Generated
        response_text = (task_state.analysis_result.full_text or "").strip()
        if not keywords and response_text:
            preview = response_text[:300]
            error_msg = (
                "Initial keyword extraction: LLM response could not be parsed "
                "(no JSON keywords and no <final_list> found). "
                f"Response preview: {preview!r}"
            )
            if self.logger:
                self.logger.error(f"💥 PIPELINE_FAILURE: {error_msg}")
            if stream_callback:
                stream_callback(
                    "\n❌ LLM-Antwort konnte nicht geparst werden – Schritt abgebrochen "
                    "(keine Schlagwörter extrahierbar)\n",
                    kwargs.get("step_id", "initialisation"),
                )
            raise ValueError(error_msg)

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
        aggregate_from_raw: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Execute GND search step with automatic catalog detection - Claude Generated

        WP2 P4.4b: by default the reduced ``{term:{title:{...}}}`` result is derived
        from the raw response cache (single source of truth) via
        ``SearchCLI.search_from_raw``. ``aggregate_from_raw=None`` (default) reads the
        ``SystemConfig.aggregate_from_raw`` toggle; pass an explicit bool to override.
        """
        if aggregate_from_raw is None:
            from src.core.search.aggregate import default_aggregate_from_raw
            aggregate_from_raw = default_aggregate_from_raw()

        if suggesters is None:
            suggesters = ["lobid", "swb"]
            
            # Add catalog if available (no auto-detection, explicit configuration)
            # Catalog will be added via suggesters parameter in pipeline

        # Convert suggester names to provider ids - Claude Generated
        valid_providers = set(providers_for_capability(SearchCapability.GND_KEYWORDS)) | {"all"}
        suggester_types = []
        for suggester_name in suggesters:
            pid = str(suggester_name).lower()
            if pid in valid_providers:
                suggester_types.append(pid)
            elif self.logger:
                self.logger.warning(f"Unknown suggester: {suggester_name}")

        # Honor the Plugins-tab enable/disable gate: expand "all", then keep only
        # provider types whose instance is enabled. ``None`` = config unreadable →
        # keep the requested list unchanged (no silent empty search). - Claude Generated
        enabled_ids = enabled_gnd_provider_ids()
        if enabled_ids is not None:
            expanded = []
            for pid in suggester_types:
                expanded.extend(["lobid", "swb", "catalog"] if pid == "all" else [pid])
            gated = [pid for pid in dict.fromkeys(expanded) if pid in enabled_ids]
            if not gated and enabled_ids:
                # None of the requested sources is enabled, but other GND
                # providers are — e.g. the built-ins were disabled and the
                # keyword search is served entirely by own/external plugins
                # (poc_lobid …). Use the enabled set instead of silently
                # searching nothing (the requested list holds only the retired
                # default ids that no live instance matches). - Claude Generated
                gated = list(enabled_ids)
                if self.logger:
                    self.logger.info(
                        "execute_gnd_search: requested %s all disabled → "
                        "falling back to enabled providers %s",
                        suggester_types, gated,
                    )
                if stream_callback:
                    stream_callback(
                        f"Angeforderte Quellen deaktiviert → nutze aktive Plugins: "
                        f"{', '.join(gated)}\n", "search"
                    )
            elif gated != suggester_types:
                dropped = [p for p in dict.fromkeys(expanded) if p not in enabled_ids]
                if dropped and self.logger:
                    self.logger.info(f"execute_gnd_search: skipping disabled providers {dropped}")
                if dropped and stream_callback:
                    stream_callback(
                        f"Übersprungene (deaktivierte) Quellen: {', '.join(dropped)}\n", "search"
                    )
            suggester_types = gated

        # Convert keywords to list if needed
        if isinstance(keywords, str):
            keywords_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        else:
            keywords_list = keywords

        # Stream search progress if callback provided - Claude Generated
        # === DIAGNOSTIC: Log suggester configuration ===
        if self.logger:
            self.logger.info(f"🔍 execute_gnd_search: {len(keywords_list)} Keywords, Suggester: {suggester_types}")
        if stream_callback:
            stream_callback(
                f"Suche mit {len(keywords_list)} Keywords: {', '.join(keywords_list)}\n",
                "search",
            )
            stream_callback(
                f"Verwende Suggester: {', '.join(suggester_types)}\n",
                "search",
            )

        # Execute search per keyword for live progress updates - Claude Generated
        search_results = {}
        source_failures: Dict[str, str] = {}  # "<suggester>:<term>" → message - Claude Generated
        with SearchCLI(
            self.cache_manager,
            catalog_token=catalog_token or "",
            catalog_search_url=catalog_search_url or "",
            catalog_details_url=catalog_details_url or ""
        ) as search_cli:
            for keyword in keywords_list:
                if stream_callback:
                    stream_callback(f"🔍 Suche '{keyword}'...\n", "search")

                search_fn = (
                    search_cli.search_from_raw if aggregate_from_raw else search_cli.search
                )
                kw_results = search_fn(
                    search_terms=[keyword], suggester_types=suggester_types
                )

                # Merge into combined results
                for term, term_data in kw_results.items():
                    if term not in search_results:
                        search_results[term] = {}
                    search_results[term].update(term_data)

                # Surface source failures: 0 hits after a failed source is
                # NOT a confirmed miss - Claude Generated
                if search_cli.last_errors:
                    source_failures.update(search_cli.last_errors)
                    failed_sources = sorted(
                        {key.split(":", 1)[0] for key in search_cli.last_errors}
                    )
                    if stream_callback:
                        stream_callback(
                            f"    ⚠️ Quelle(n) fehlgeschlagen für '{keyword}': "
                            f"{', '.join(failed_sources)}\n",
                            "search",
                        )

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

        if source_failures:
            if self.logger:
                self.logger.warning(
                    f"GND search finished with {len(source_failures)} source failure(s): "
                    f"{sorted(source_failures)}"
                )
            if stream_callback:
                stream_callback(
                    f"⚠️ Suche abgeschlossen, aber {len(source_failures)} Quellen-Fehler "
                    "(leere Treffer ggf. nicht aussagekräftig).\n",
                    "search",
                )
        elif stream_callback:
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
                        suggester_types=["lobid", "swb"]
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
                        suggester_types=["swb"]
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
        # Extract keyword chains from response for display in verification tab - Claude Generated
        from ..core.processing_utils import extract_keyword_chains_from_response
        _prompt_cfg = task_state.prompt_config
        _output_fmt = _prompt_cfg.output_format if _prompt_cfg else None
        extracted_chains = extract_keyword_chains_from_response(
            task_state.analysis_result.full_text, output_format=_output_fmt
        )

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
            keyword_chains=extracted_chains,  # Store parsed chains for UI display - Claude Generated
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

        # Split into equal chunks via the shared classic-semantics splitter
        # (single source of truth: src/utils/chunking.py) - Claude Generated
        chunks = split_into_equal_chunks(gnd_compliant_keywords, threshold)
        num_chunks = len(chunks)

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
        if stream_callback:
            stream_callback(
                (
                    "\n--- Konsolidierung der deduplizierten Keywords ---\n"
                    f"Starte finalen Konsolidierungslauf mit {len(deduplicated_keywords)} Keywords "
                    "(ohne Live-Token-Stream, um Wiederholungsschleifen in der Anzeige zu vermeiden)\n"
                ),
                kwargs.get("step_id", "keywords"),
            )
        final_single_result = self._execute_single_keyword_analysis(
            original_abstract=original_abstract,
            gnd_keywords_text=final_keywords_text,
            gnd_compliant_keywords=deduplicated_keywords,
            model=model,
            provider=provider,
            task=task,  # Use normal keywords task, NOT chunking_task!
            stream_callback=None,
            mode=mode,
            full_gnd_pool_for_verification=full_gnd_pool,  # Claude Generated - pass full pool
            **kwargs,
        )
        if stream_callback:
            stream_callback(
                (
                    f"Konsolidierung abgeschlossen: {len(final_single_result[0])} "
                    "verifizierte Keywords im Endergebnis\n"
                ),
                kwargs.get("step_id", "keywords"),
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
        # Keyword chains come from the final consolidation response - Claude Generated
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
            keyword_chains=final_single_result[2].keyword_chains,  # Chains from final consolidation - Claude Generated
            chunk_responses=combined_responses,  # Store chunk responses separately - Claude Generated
            chunk_keywords=list(deduplicated_keywords),  # Chunk-survivor pool (pre-consolidation) for the GND-Recherche chunk tier - Claude Generated
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

    def prepare_dk_classification_context(
        self,
        dk_search_results: List[Dict[str, Any]],
        original_abstract: str,
        dk_frequency_threshold: int = DEFAULT_DK_FREQUENCY_THRESHOLD,
        rvk_anchor_keywords: Optional[List[str]] = None,
        stream_callback=None,
        include_rvk: bool = True,
    ) -> Dict[str, Any]:
        """Filter and format DK search results for the classification prompt - Claude Generated

        Shared by the classic pipeline (execute_dk_classification) and the
        agentic pipeline (dk_search_agentic) so both build the classification
        context identically: frequency threshold (DK only, RVK exempt),
        institution-library RVK filter, title filter, RVK candidate maps and
        RVK guardrail text prepended to the formatted catalog excerpt.

        ``include_rvk=False`` strips every RVK-typed entry up front, so the
        formatted prompt and the candidate maps stay DK/DDC only. Used by
        workflows that surface RVK out-of-band via the ``rvk_lookup`` tool
        instead of inline in ``dk_collect`` - Claude Generated.

        Returns:
            Dict with results_with_titles, catalog_text, allowed_standard_rvk_map,
            allowed_nonstandard_rvk_map, rvk_source_map, selected_rvk_meta.
        """
        # Filter results by frequency threshold - Claude Generated
        filtered_results = []
        low_frequency_count = 0
        
        for result in dk_search_results:
            classification_type = str(result.get("classification_type", result.get("type", "DK"))).upper()
            # RVK is validated separately and exempt from the frequency filter;
            # DK and DDC are frequency-filtered below. When include_rvk is False
            # the caller handles RVK out-of-band (rvk_lookup tool) → drop it so
            # the prompt and candidate maps stay DK/DDC only. - Claude Generated
            if classification_type == "RVK":
                if include_rvk:
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
                stream_callback(f"Filtere Klassifikationen: {len(filtered_results)} Einträge mit ≥{dk_frequency_threshold} Vorkommen, {low_frequency_count} mit niedrigerer Häufigkeit ausgeschlossen\n", "dk_classification")
            else:
                stream_callback(f"Verwende alle {len(filtered_results)} Klassifikations-Einträge (keine Häufigkeits-Filterung nötig)\n", "dk_classification")

        # Filter out results without titles - Claude Generated
        results_with_titles = []
        titleless_count = 0
        institution_library_rvk_count = 0

        for result in filtered_results:
            classification_type = str(result.get("classification_type", result.get("type", "DK"))).upper()
            if (
                classification_type == "RVK"
                and str(result.get("source", "catalog") or "catalog") == "catalog"
                and self._is_institution_library_rvk(result)
                and not self._matches_specific_library_context(
                    result,
                    original_abstract,
                    result.get("matched_keywords", []) or result.get("keywords", []) or [],
                )
            ):
                institution_library_rvk_count += 1
                continue

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
            if institution_library_rvk_count > 0:
                stream_callback(
                    f"⚠️ Verwerfe {institution_library_rvk_count} katalogseitige RVK für einzelne Bibliotheken ohne Dokumentbezug\n",
                    "dk_classification",
                )

        if self.logger:
            self.logger.info(f"DK title filter: {len(results_with_titles)} with titles, {titleless_count} without titles excluded")

        allowed_standard_rvk_map = {}
        allowed_nonstandard_rvk_map = {}
        rvk_source_map = {}
        selected_rvk_meta = {}
        anchor_terms = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }
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
            matched_keywords = {
                canonicalize_keyword(str(item or ""))
                for item in (result.get("matched_keywords") or [])
                if canonicalize_keyword(str(item or ""))
            }
            selected_rvk_meta[normalized] = {
                "branch": str(result.get("branch_family", "") or ""),
                "depth": len([part for part in str(result.get("ancestor_path", "") or "").split(">") if part.strip()]),
                "anchor_hit_count": len(anchor_terms.intersection(matched_keywords)),
                "source": result.get("source", "catalog"),
            }
            if rvk_status in {"non_standard", "validation_error"}:
                allowed_nonstandard_rvk_map[normalized] = f"RVK {normalized}"
            else:
                allowed_standard_rvk_map[normalized] = f"RVK {normalized}"

        # Format catalog results for LLM prompt with aggregated data - Claude Generated
        catalog_text = PipelineResultFormatter.format_dk_results_for_prompt(results_with_titles)

        if allowed_standard_rvk_map or allowed_nonstandard_rvk_map:
            rvk_guardrail = (
                "WICHTIG FÜR RVK:\n"
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

        return {
            "results_with_titles": results_with_titles,
            "catalog_text": catalog_text,
            "allowed_standard_rvk_map": allowed_standard_rvk_map,
            "allowed_nonstandard_rvk_map": allowed_nonstandard_rvk_map,
            "rvk_source_map": rvk_source_map,
            "selected_rvk_meta": selected_rvk_meta,
        }

    def execute_dk_classification(
        self,
        original_abstract: str,
        dk_search_results: List[Dict[str, Any]],
        model: str = None,
        provider: str = None,
        stream_callback: Optional[callable] = None,
        dk_frequency_threshold: int = DEFAULT_DK_FREQUENCY_THRESHOLD,  # Claude Generated - Only pass classifications with >= N occurrences
        rvk_anchor_keywords: Optional[List[str]] = None,
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

        # Shared filtering/formatting with the agentic pipeline - Claude Generated
        prep = self.prepare_dk_classification_context(
            dk_search_results,
            original_abstract,
            dk_frequency_threshold=dk_frequency_threshold,
            rvk_anchor_keywords=rvk_anchor_keywords,
            stream_callback=stream_callback,
        )
        results_with_titles = prep["results_with_titles"]
        catalog_text = prep["catalog_text"]
        allowed_standard_rvk_map = prep["allowed_standard_rvk_map"]
        allowed_nonstandard_rvk_map = prep["allowed_nonstandard_rvk_map"]
        rvk_source_map = prep["rvk_source_map"]
        selected_rvk_meta = prep["selected_rvk_meta"]

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
            # Extract analysis text from LLM response - Claude Generated
            from ..core.processing_utils import extract_analyse_text_from_response
            analyse_text = extract_analyse_text_from_response(response_text, output_format=_output_format)

            llm_dk_only = [
                code for code in llm_classifications
                if not str(code or "").strip().upper().startswith("RVK ")
            ]
            llm_rvk_only = [
                code for code in llm_classifications
                if str(code or "").strip().upper().startswith("RVK ")
            ]

            rescored_rvk = self._select_final_rvk_with_dk_context(
                results_with_titles,
                original_abstract,
                llm_dk_only,
                rvk_anchor_keywords=rvk_anchor_keywords,
                model=model,
                provider=provider,
                stream_callback=stream_callback,
                mode=mode,
                llm_kwargs=alima_kwargs,
            )
            if rescored_rvk:
                llm_rvk_only = rescored_rvk

            pruned_llm_rvk = []
            for code in llm_rvk_only:
                normalized = canonicalize_rvk_notation(str(code or "")[4:].strip())
                meta = selected_rvk_meta.get(normalized, {})
                depth = int(meta.get("depth", 0) or 0)
                anchor_hit_count = int(meta.get("anchor_hit_count", 0) or 0)
                source = str(meta.get("source", "catalog") or "catalog")
                branch = str(meta.get("branch", "") or "")

                is_broad_catalog = source == "catalog" and depth <= 2 and anchor_hit_count <= 1
                has_more_specific_peer = any(
                    other_code != normalized
                    and str(other_meta.get("branch", "") or "") == branch
                    and int(other_meta.get("depth", 0) or 0) > depth
                    and int(other_meta.get("anchor_hit_count", 0) or 0) >= anchor_hit_count
                    for other_code, other_meta in selected_rvk_meta.items()
                )

                if is_broad_catalog and has_more_specific_peer:
                    if stream_callback:
                        stream_callback(
                            f"⚠️ Verwerfe zu allgemeine RVK-Auswahl: RVK {normalized}\n",
                            "dk_classification",
                        )
                    continue
                pruned_llm_rvk.append(code)

            llm_rvk_only = pruned_llm_rvk

            max_total_classifications = 10
            max_dk_count = max(0, max_total_classifications - len(llm_rvk_only))
            dk_classifications = list(
                dict.fromkeys(llm_dk_only[:max_dk_count] + llm_rvk_only)
            )

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
        from .lookups.cache import cached_call, lookup_cache_enabled
        from .lookups.resolve import build_lookup

        # Route through the `rvk_api` lookup plugin (single RVK construction path,
        # shared with the agent tool) so per-instance settings (timeout) are honored.
        # - Claude Generated
        _config = self._alima_config_for_cache()
        rvk_plugin = build_lookup(_config, "rvk_api")
        # Reuse the WP2 raw cache (F3): RVK API results are cached under the same
        # `rvk_search` key the agent tool uses, gated by the rvk_api plugin's
        # `cache_responses` setting. - Claude Generated
        _km = getattr(self, "cache_manager", None)
        _rvk_cache_on = lookup_cache_enabled(_config, "rvk_api")
        keyword_results = []

        for keyword in keywords:
            clean_keyword = canonicalize_keyword(keyword)
            try:
                candidates = cached_call(
                    _km, _rvk_cache_on, "rvk_search", clean_keyword,
                    {"max_results": max_results_per_keyword},
                    lambda kw=clean_keyword: rvk_plugin.search_keyword(kw, max_results=max_results_per_keyword)["results"],
                )
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
            stream_callback("🔎 Nutze RVK MarcXML-GND-Index für standardisierte RVK-Kandidaten...\n", "dk_search")

        results = index.lookup_by_gnd_keywords(
            keyword_entries,
            max_results_per_keyword=max_results_per_keyword,
            progress_callback=(lambda msg: stream_callback(msg, "dk_search")) if stream_callback else None,
        )

        if stream_callback and results:
            total = sum(len(item.get("classifications", [])) for item in results)
            stream_callback(
                f"  ✅ RVK-GND-Index: {total} standardisierte Kandidaten für {len(results)} Keywords\n",
                "dk_search",
            )

        return results

    def _validate_catalog_rvk_candidates(
        self,
        keyword_results: List[Dict[str, Any]],
        stream_callback: Optional[callable] = None,
        rvk_anchor_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Validate catalog-derived RVK candidates and drop only obvious artifacts."""
        from .lookups.cache import cached_call, lookup_cache_enabled
        from .lookups.resolve import build_lookup

        # Validate through the `rvk_api` lookup plugin (single RVK construction path,
        # shared with the agent tool). - Claude Generated
        _config = self._alima_config_for_cache()
        rvk_plugin = build_lookup(_config, "rvk_api")
        # Persist RVK notation validations in the WP2 raw cache (F3), sharing the
        # `rvk_validate` key with the agent tool. - Claude Generated
        _km = getattr(self, "cache_manager", None)
        _rvk_cache_on = lookup_cache_enabled(_config, "rvk_api")

        validation_cache: Dict[str, Dict[str, Any]] = {}
        cleaned_results: List[Dict[str, Any]] = []
        standard_count = 0
        nonstandard_count = 0
        artifact_count = 0
        validation_error_count = 0
        pruned_general_count = 0
        max_validation_candidates = 120
        max_anchor_candidates = 96
        max_exploration_candidates = 12
        max_per_anchor = 12
        max_per_branch = 18
        unique_plausible_codes = []
        seen_plausible_codes = set()
        evidence_by_code: Dict[str, Dict[str, Any]] = {}
        anchor_terms = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }

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

        def _branch_key(code: str) -> str:
            match = re.match(r"^([A-Z]{1,3})\s*", code)
            return match.group(1) if match else (code.split()[0] if code.split() else code)

        def _compact_rvk(code: str) -> str:
            return re.sub(r"[^A-Z0-9.]", "", str(code or "").upper())

        def _is_parent_like(parent_code: str, child_code: str) -> bool:
            parent_compact = _compact_rvk(parent_code)
            child_compact = _compact_rvk(child_code)
            return (
                bool(parent_compact)
                and bool(child_compact)
                and child_compact != parent_compact
                and child_compact.startswith(parent_compact)
                and len(child_compact) > len(parent_compact) + 1
            )

        candidate_meta = []
        for code in unique_plausible_codes:
            evidence = evidence_by_code.get(code, {})
            keyword_hits = {
                canonicalize_keyword(str(item or ""))
                for item in evidence.get("keyword_hits", set())
                if canonicalize_keyword(str(item or ""))
            }
            anchor_hits = sorted(anchor_terms.intersection(keyword_hits))
            specificity_bonus = min(len(code.replace(" ", "")), 12)
            keyword_hit_count = len(keyword_hits)
            count_value = int(evidence.get("count", 0))
            title_hit_count = int(evidence.get("title_hits", 0))
            score = (
                len(anchor_hits) * 120
                + keyword_hit_count * 15
                + count_value * 4
                + min(title_hit_count, 6)
                + specificity_bonus
            )
            candidate_meta.append({
                "code": code,
                "score": score,
                "anchor_hits": anchor_hits,
                "anchor_hit_count": len(anchor_hits),
                "branch": _branch_key(code),
                "keyword_hit_count": keyword_hit_count,
                "count_value": count_value,
                "title_hit_count": title_hit_count,
            })

        candidate_meta.sort(
            key=lambda item: (
                -int(item["anchor_hit_count"]),
                -int(item["score"]),
                item["code"],
            )
        )

        selected_codes: List[str] = []
        selected_set = set()
        branch_counts: Dict[str, int] = {}
        anchor_counts: Dict[str, int] = {anchor: 0 for anchor in anchor_terms}
        meta_by_code = {item["code"]: item for item in candidate_meta}

        def _is_strong_anchor_candidate(item: Dict[str, Any]) -> bool:
            if item["anchor_hit_count"] >= 2:
                return True
            if item["anchor_hit_count"] == 1:
                return (
                    int(item.get("keyword_hit_count", 0)) >= 2
                    or int(item.get("count_value", 0)) >= 8
                    or int(item.get("title_hit_count", 0)) >= 3
                )
            return False

        anchored_candidates = [
            item for item in candidate_meta
            if item["anchor_hit_count"] > 0 and (not anchor_terms or _is_strong_anchor_candidate(item))
        ]
        if anchor_terms and not anchored_candidates:
            anchored_candidates = [item for item in candidate_meta if item["anchor_hit_count"] > 0]
        exploratory_candidates = [item for item in candidate_meta if item["anchor_hit_count"] == 0]

        def _can_take_branch(item: Dict[str, Any]) -> bool:
            return branch_counts.get(item["branch"], 0) < max_per_branch

        def _select_item(item: Dict[str, Any]) -> None:
            code = item["code"]
            if code in selected_set:
                return
            selected_set.add(code)
            selected_codes.append(code)
            branch_counts[item["branch"]] = branch_counts.get(item["branch"], 0) + 1
            for anchor in item["anchor_hits"]:
                anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

        if anchored_candidates:
            for item in anchored_candidates:
                if len(selected_codes) >= max_anchor_candidates:
                    break
                if not _can_take_branch(item):
                    continue
                if item["anchor_hits"] and not any(anchor_counts.get(anchor, 0) < max_per_anchor for anchor in item["anchor_hits"]):
                    continue
                _select_item(item)

            for item in anchored_candidates:
                if len(selected_codes) >= max_anchor_candidates:
                    break
                if item["code"] in selected_set or not _can_take_branch(item):
                    continue
                _select_item(item)

        remaining_slots = max_validation_candidates - len(selected_codes)
        exploration_slots = min(max_exploration_candidates, max(0, remaining_slots))

        for item in exploratory_candidates:
            if exploration_slots <= 0 or len(selected_codes) >= max_validation_candidates:
                break
            if not _can_take_branch(item):
                continue
            _select_item(item)
            exploration_slots -= 1

        if not selected_codes:
            # Fallback: keep the strongest codes even if no anchor-balanced shortlist could be built.
            selected_codes = [item["code"] for item in candidate_meta[:max_validation_candidates]]

        selected_total = len(selected_codes)
        anchored_total = sum(1 for item in candidate_meta if item["code"] in set(selected_codes) and item["anchor_hit_count"] > 0)
        exploratory_total = selected_total - anchored_total

        if selected_codes and stream_callback:
            shortlist_parts = []
            if anchor_terms:
                shortlist_parts.append(f"{anchored_total} ankergestützt")
                if exploratory_total:
                    shortlist_parts.append(f"{exploratory_total} explorativ")
                shortlist_parts.append(f"{len(branch_counts)} Zweige")
            stream_callback(
                (
                    f"🔎 Prüfe {len(selected_codes)} eindeutige RVK-Kandidaten gegen die RVK-API"
                    + (f" ({', '.join(shortlist_parts)})" if shortlist_parts else "")
                    + "...\n"
                ),
                "dk_search",
            )

        def _validate_code(code: str) -> Tuple[str, Dict[str, Any]]:
            result = cached_call(
                _km, _rvk_cache_on, "rvk_validate", code, {},
                lambda c=code: rvk_plugin.validate_notation(c)["result"],
            )
            return code, result

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

        pruned_standard_codes = set()
        standard_validated_codes = [
            code for code in selected_codes
            if validation_cache.get(code, {}).get("status") == "standard"
        ]
        for parent_code in standard_validated_codes:
            parent_meta = meta_by_code.get(parent_code, {})
            parent_branch = parent_meta.get("branch", "")
            for child_code in standard_validated_codes:
                if child_code == parent_code:
                    continue
                child_meta = meta_by_code.get(child_code, {})
                if parent_branch and parent_branch != child_meta.get("branch", ""):
                    continue
                if not _is_parent_like(parent_code, child_code):
                    continue
                if (
                    int(child_meta.get("anchor_hit_count", 0)) >= int(parent_meta.get("anchor_hit_count", 0))
                    and int(child_meta.get("score", 0)) >= int(parent_meta.get("score", 0)) - 10
                ):
                    pruned_standard_codes.add(parent_code)
                    break

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
                    if normalized in pruned_standard_codes:
                        pruned_general_count += 1
                        continue
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

        if stream_callback and (standard_count or nonstandard_count or artifact_count or validation_error_count or pruned_general_count):
            parts = []
            if standard_count:
                parts.append(f"{standard_count} standard")
            if nonstandard_count:
                parts.append(f"{nonstandard_count} nicht-standardisiert/lokal")
            if validation_error_count:
                parts.append(f"{validation_error_count} ungeprüft (API-Fehler)")
            if pruned_general_count:
                parts.append(f"{pruned_general_count} allgemeine Elternknoten verworfen")
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
        covered_standard_keywords = set()
        for kw_result in dk_search_results:
            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", ""))).upper()
                rvk_status = classification.get("rvk_validation_status")
                if cls_type == "RVK" and rvk_status == "standard":
                    existing_standard_rvk = True
                    matched_keywords = classification.get("matched_keywords", []) or classification.get("keywords", []) or []
                    if not matched_keywords and kw_result.get("keyword"):
                        matched_keywords = [kw_result.get("keyword")]
                    for keyword in matched_keywords:
                        normalized = canonicalize_keyword(keyword)
                        if normalized:
                            covered_standard_keywords.add(normalized)
                    break

        fallback_keywords = deduplicate_canonical_keywords(final_search_keywords or [])
        uncovered_fallback_keywords = [
            keyword for keyword in fallback_keywords
            if canonicalize_keyword(keyword) not in covered_standard_keywords
        ]

        if existing_standard_rvk and not uncovered_fallback_keywords:
            return dk_search_results

        if gnd_keyword_entries:
            use_gnd_index = not existing_standard_rvk
            gnd_entries_for_fallback = list(gnd_keyword_entries or [])
            if use_gnd_index and stream_callback:
                stream_callback(
                    "⚠️ Katalog lieferte keine standardisierten RVK-Kandidaten - versuche RVK-GND-Index\n",
                    "dk_search"
                )
            if use_gnd_index:
                rvk_index_results = self._build_rvk_gnd_index_results(
                    gnd_entries_for_fallback,
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

        if existing_standard_rvk and uncovered_fallback_keywords and stream_callback:
            preview = ", ".join(uncovered_fallback_keywords[:6])
            if len(uncovered_fallback_keywords) > 6:
                preview += f", +{len(uncovered_fallback_keywords) - 6} weitere"
            stream_callback(
                f"ℹ️ Ergänze RVK-API-Fallback für nicht abgedeckte Anker/Promotionen: {preview}\n",
                "dk_search"
            )
        elif stream_callback:
            stream_callback(
                "⚠️ RVK-GND-Index lieferte nichts - nutze offiziellen RVK-API-Label-Fallback\n",
                "dk_search"
            )

        rvk_api_results = self._build_rvk_api_fallback_results(
            uncovered_fallback_keywords or fallback_keywords,
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

    @staticmethod
    def _rvk_significant_tokens(text: str) -> List[str]:
        """Extract simple content-bearing tokens for deterministic RVK ranking."""
        stopwords = {
            "und", "oder", "der", "die", "das", "des", "dem", "den", "ein", "eine", "einer",
            "eines", "im", "in", "am", "an", "auf", "mit", "ohne", "von", "vom", "zum", "zur",
            "fur", "fuer", "uber", "ueber", "unter", "zwischen", "nach", "vor", "bei", "aus",
            "zu", "ist", "sind", "war", "werden", "wird", "als", "auch", "nicht", "kein",
            "keine", "sehr", "mehr", "weniger", "durch", "gegen", "seit", "bis", "eines",
            "einem", "einen", "dieser", "diese", "dieses", "jene", "jener", "jenes",
            "text", "analyse", "geschichte",  # generic high-frequency tokens contribute little
        }
        normalized = canonicalize_keyword(str(text or "")).casefold()
        raw_tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]{4,}", normalized)
        return [token for token in raw_tokens if token not in stopwords]

    def _derive_rvk_anchor_keywords_heuristic(
        self,
        verified_keywords: List[str],
        llm_analysis: Optional["LlmKeywordAnalysis"] = None,
        max_anchors: int = 8,
    ) -> List[str]:
        """Derive a small thematic GND subset to drive RVK lookup and ranking."""
        if not verified_keywords:
            return []

        analysis_text = ""
        missing_concepts: List[str] = []
        keyword_chains: List[Dict[str, Any]] = []

        if llm_analysis:
            analysis_text = str(getattr(llm_analysis, "analyse_text", "") or "")
            response_text = str(getattr(llm_analysis, "response_full_text", "") or "")
            if not analysis_text and response_text:
                from ..core.processing_utils import extract_analyse_text_from_response
                analysis_text = extract_analyse_text_from_response(response_text) or ""
            missing_concepts = list(getattr(llm_analysis, "missing_concepts", []) or [])
            if not missing_concepts and response_text:
                from ..core.processing_utils import extract_missing_concepts_from_response
                missing_concepts = extract_missing_concepts_from_response(response_text)
            if response_text:
                from ..core.processing_utils import extract_keyword_chains_from_response
                keyword_chains = extract_keyword_chains_from_response(response_text)

        thematic_fragments = [analysis_text] + list(missing_concepts)
        for chain in keyword_chains:
            thematic_fragments.extend(chain.get("chain", []) or [])
            thematic_fragments.append(str(chain.get("reason", "") or ""))
        thematic_text = " ".join(fragment for fragment in thematic_fragments if fragment)
        thematic_tokens = set(self._rvk_significant_tokens(thematic_text))

        institutional_terms = {
            "bibliothek", "bibliotheken", "zeitung", "fernsehen", "massenmedien",
            "kommentar", "alltag", "rezeption", "stadt", "bild", "sohn",
        }

        scored_keywords = []
        for keyword in verified_keywords:
            clean_keyword = keyword.split("(GND-ID:")[0].strip()
            keyword_lower = canonicalize_keyword(clean_keyword).casefold()
            keyword_tokens = set(self._rvk_significant_tokens(clean_keyword))
            score = 0

            if analysis_text and keyword_lower and keyword_lower in canonicalize_keyword(analysis_text).casefold():
                score += 45

            for concept in missing_concepts:
                concept_lower = canonicalize_keyword(concept).casefold()
                if not concept_lower:
                    continue
                if keyword_lower == concept_lower:
                    score += 30
                elif keyword_lower and (keyword_lower in concept_lower or concept_lower in keyword_lower):
                    score += 18

            for chain in keyword_chains:
                chain_terms = [canonicalize_keyword(item).casefold() for item in (chain.get("chain", []) or [])]
                if keyword_lower and keyword_lower in chain_terms:
                    score += 26
                reason_text = canonicalize_keyword(str(chain.get("reason", "") or "")).casefold()
                if keyword_lower and keyword_lower in reason_text:
                    score += 10

            token_overlap = len(keyword_tokens.intersection(thematic_tokens))
            score += token_overlap * 9

            if keyword_tokens and keyword_tokens.issubset(institutional_terms) and score < 45:
                score -= 12

            scored_keywords.append((score, clean_keyword.casefold(), keyword))

        scored_keywords.sort(key=lambda item: (-item[0], item[1]))
        selected = [keyword for score, _, keyword in scored_keywords if score > 0][:max_anchors]

        if not selected:
            selected = verified_keywords[: min(max_anchors, len(verified_keywords))]

        return selected

    def _derive_promoted_rvk_terms(
        self,
        initial_keywords: Optional[List[str]] = None,
        search_results: Optional[Any] = None,
        llm_analysis: Optional["LlmKeywordAnalysis"] = None,
        max_terms: int = 3,
    ) -> Tuple[List[str], List[str]]:
        """Promote search-backed initial concepts that reappear as missing core concepts."""
        if not initial_keywords or not llm_analysis:
            return [], []

        if isinstance(initial_keywords, str):
            initial_keywords = [
                item.strip()
                for item in re.split(r"[\n,]+", initial_keywords)
                if item.strip()
            ]
        else:
            initial_keywords = [
                str(item).strip()
                for item in list(initial_keywords)
                if str(item).strip()
            ]

        if not initial_keywords:
            return [], []

        response_text = str(getattr(llm_analysis, "response_full_text", "") or "")
        missing_concepts = list(getattr(llm_analysis, "missing_concepts", []) or [])
        if not missing_concepts and response_text:
            from ..core.processing_utils import extract_missing_concepts_from_response
            missing_concepts = extract_missing_concepts_from_response(response_text)
        if not missing_concepts:
            return [], []

        search_support: Dict[str, bool] = {}
        observed_search_terms = set()
        if isinstance(search_results, dict):
            for term, results in search_results.items():
                normalized = canonicalize_keyword(term)
                if not normalized:
                    continue
                observed_search_terms.add(normalized)
                search_support[normalized] = bool(results)
        elif isinstance(search_results, list):
            for item in search_results:
                term = getattr(item, "search_term", None) or (item.get("search_term") if isinstance(item, dict) else "")
                results = getattr(item, "results", None) or (item.get("results") if isinstance(item, dict) else None)
                normalized = canonicalize_keyword(term)
                if normalized:
                    observed_search_terms.add(normalized)
                    search_support[normalized] = bool(results)

        promoted: List[str] = []
        seen = set()
        diagnostics: List[str] = []
        missing_canonical = [canonicalize_keyword(item) for item in missing_concepts if canonicalize_keyword(item)]

        for keyword in initial_keywords:
            clean_keyword = str(keyword or "").strip()
            normalized_keyword = canonicalize_keyword(clean_keyword)
            if not normalized_keyword:
                diagnostics.append(f"{clean_keyword or '(leer)'} -> ignoriert (nicht normalisierbar)")
                continue

            has_search_support = search_support.get(normalized_keyword)
            if not has_search_support and normalized_keyword not in observed_search_terms:
                diagnostics.append(f"{clean_keyword} -> verworfen (keine Suchunterstützung)")
                continue

            matched_concept = None
            for concept in missing_canonical:
                if (
                    normalized_keyword == concept
                    or normalized_keyword in concept
                    or concept in normalized_keyword
                ):
                    matched_concept = concept
                    if clean_keyword not in seen:
                        promoted.append(clean_keyword)
                        seen.add(clean_keyword)
                    break
            if matched_concept:
                diagnostics.append(f"{clean_keyword} -> gefördert (passt zu '{matched_concept}')")
            else:
                diagnostics.append(f"{clean_keyword} -> verworfen (kein Missing-Concept-Match)")
            if len(promoted) >= max_terms:
                break

        return promoted, diagnostics

    def _derive_rvk_anchor_keywords(
        self,
        verified_keywords: List[str],
        llm_analysis: Optional["LlmKeywordAnalysis"] = None,
        original_abstract: str = "",
        initial_keywords: Optional[List[str]] = None,
        search_results: Optional[Any] = None,
        max_anchors: int = 8,
        stream_callback: Optional[callable] = None,
        ) -> List[str]:
        """Derive RVK anchors with an LLM-first selection and heuristic fallback."""
        promoted_terms, promotion_diagnostics = self._derive_promoted_rvk_terms(
            initial_keywords=initial_keywords,
            search_results=search_results,
            llm_analysis=llm_analysis,
        )

        def _merge_promoted(selected_terms: List[str]) -> List[str]:
            merged = list(selected_terms or [])
            for term in promoted_terms:
                if term not in merged:
                    merged.append(term)
            return merged

        def _emit_promotion_log() -> None:
            if not stream_callback:
                return
            if promoted_terms:
                preview = ", ".join(promoted_terms[:6])
                stream_callback(
                    f"ℹ️ RVK-Promotion aus Initialbegriffen: {preview}\n",
                    "dk_search",
                )
                return
            stream_callback(
                "ℹ️ RVK-Promotion aus Initialbegriffen: keine passenden Kandidaten\n",
                "dk_search",
            )
            if promotion_diagnostics:
                preview = "; ".join(promotion_diagnostics[:4])
                if len(promotion_diagnostics) > 4:
                    preview += f"; +{len(promotion_diagnostics) - 4} weitere"
                stream_callback(
                    f"  ↳ {preview}\n",
                    "dk_search",
                )

        if not verified_keywords:
            _emit_promotion_log()
            return promoted_terms

        heuristic_selected = self._derive_rvk_anchor_keywords_heuristic(
            verified_keywords,
            llm_analysis=llm_analysis,
            max_anchors=max_anchors,
        )

        if not self.alima_manager:
            selected = _merge_promoted(heuristic_selected)
            _emit_promotion_log()
            return selected

        analysis_fragments = []
        if original_abstract:
            analysis_fragments.append(str(original_abstract))

        if llm_analysis:
            analysis_text = str(getattr(llm_analysis, "analyse_text", "") or "")
            if analysis_text:
                analysis_fragments.append(f"\nThematische Analyse:\n{analysis_text}")
            missing_concepts = list(getattr(llm_analysis, "missing_concepts", []) or [])
            if missing_concepts:
                analysis_fragments.append(
                    "\nFehlende Konzepte:\n" + ", ".join(str(item) for item in missing_concepts if str(item).strip())
                )

        abstract_for_selection = "\n".join(fragment for fragment in analysis_fragments if fragment).strip()
        if not abstract_for_selection:
            selected = _merge_promoted(heuristic_selected)
            _emit_promotion_log()
            return selected

        from ..core.data_models import AbstractData
        from ..core.json_response_parser import parse_json_response

        keyword_lines = "\n".join(verified_keywords)
        abstract_data = AbstractData(
            abstract=abstract_for_selection,
            keywords=keyword_lines,
        )

        provider = getattr(llm_analysis, "provider_used", None) if llm_analysis else None
        model = getattr(llm_analysis, "model_used", None) if llm_analysis else None

        try:
            task_state = self.alima_manager.analyze_abstract(
                abstract_data=abstract_data,
                task="rvk_anchor_selection",
                model=model,
                provider=provider,
                stream_callback=None,
            )
            if task_state.status == "failed":
                selected = _merge_promoted(heuristic_selected)
                _emit_promotion_log()
                return selected

            parsed = parse_json_response(task_state.analysis_result.full_text) or {}
            anchors = parsed.get("anchors", [])
            if not isinstance(anchors, list):
                selected = _merge_promoted(heuristic_selected)
                _emit_promotion_log()
                return selected

            keyword_by_gnd: Dict[str, str] = {}
            keyword_by_text: Dict[str, str] = {}
            for keyword in verified_keywords:
                clean_keyword = keyword.split("(GND-ID:")[0].strip()
                clean_key = canonicalize_keyword(clean_keyword)
                if clean_key:
                    keyword_by_text[clean_key] = keyword
                gnd_id = extract_gnd_id(keyword)
                if gnd_id:
                    keyword_by_gnd[gnd_id] = keyword

            selected = []
            seen = set()
            for item in anchors:
                if not isinstance(item, dict):
                    continue
                gnd_id = str(item.get("gnd_id", "") or "").strip()
                keyword_text = canonicalize_keyword(str(item.get("keyword", "") or "").strip())
                matched = None
                if gnd_id and gnd_id in keyword_by_gnd:
                    matched = keyword_by_gnd[gnd_id]
                elif keyword_text and keyword_text in keyword_by_text:
                    matched = keyword_by_text[keyword_text]
                if matched and matched not in seen:
                    selected.append(matched)
                    seen.add(matched)
                if len(selected) >= max_anchors:
                    break

            if selected:
                selected = _merge_promoted(selected)
                if stream_callback:
                    preview = ", ".join(keyword.split("(GND-ID:")[0].strip() for keyword in selected[:6])
                    if len(selected) > 6:
                        preview += f", +{len(selected) - 6} weitere"
                    stream_callback(
                        f"ℹ️ RVK-Anker (LLM): {preview}\n",
                        "dk_search",
                    )
                    _emit_promotion_log()
                return selected
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"LLM RVK anchor selection failed, falling back to heuristic: {exc}")

        selected = _merge_promoted(heuristic_selected)
        if stream_callback and selected:
            preview = ", ".join(keyword.split("(GND-ID:")[0].strip() for keyword in selected[:6])
            if len(selected) > 6:
                preview += f", +{len(selected) - 6} weitere"
            stream_callback(
                f"ℹ️ RVK-Anker (heuristisch): {preview}\n",
                "dk_search",
            )
            _emit_promotion_log()
        return selected

    def _rvk_domain_profile(self, abstract_text: str, matched_keywords: List[str]) -> Dict[str, int]:
        """Estimate thematic domain strength from the abstract and matched keywords."""
        combined_tokens = self._rvk_significant_tokens(
            " ".join([str(abstract_text or "")] + [str(item) for item in matched_keywords])
        )
        counts: Dict[str, int] = {}
        domain_terms = {
            "politics": {
                "politik", "politisch", "demokratie", "demokratisierung", "dissident",
                "kommunismus", "sozialismus", "totalitarismus", "verfolgung",
                "menschenrecht", "menschenrechte", "menschenrechtspolitik", "staat",
                "ost", "west", "konflikt", "krieg", "nato", "osze", "helsinki",
                "sicherheitspolitik", "weltpolitik", "international", "völkerrecht", "voelkerrecht",
            },
            "history": {
                "geschichte", "historisch", "osteuropa", "sowjetunion", "prager",
                "fruhling", "fruehling", "tschechoslowakei", "slowakei", "russisch",
                "ukrainisch", "kalter", "krieg", "biografie", "autobiografie",
            },
            "law": {
                "recht", "rechte", "grundrecht", "freiheitsrecht", "vertrag",
                "völkerrecht", "voelkerrecht", "konvention", "menschenrecht",
            },
            "media": {
                "zeitung", "fernsehen", "massenmedien", "propaganda", "offentliche", "oeffentliche",
            },
            "literature": {
                "literatur", "schriftsteller", "roman", "erzahlung", "erzaehlung",
                "autobiografie", "biografie",
            },
            "religion": {
                "christlich", "kirche", "theologie", "religion", "ethik",
            },
            "philosophy": {
                "philosophie", "ethik", "denken", "theorie",
            },
        }

        for domain, terms in domain_terms.items():
            counts[domain] = sum(1 for token in combined_tokens if token in terms)
        return counts

    def _rvk_branch_fit_score(
        self,
        candidate: Dict[str, Any],
        abstract_text: str,
        matched_keywords: List[str],
    ) -> int:
        """Score how well the candidate branch fits the text domain."""
        branch_text = " ".join(
            [
                str(candidate.get("label", "") or ""),
                str(candidate.get("ancestor_path", "") or ""),
                " ".join(str(item) for item in (candidate.get("register") or [])),
            ]
        )
        branch_tokens = set(self._rvk_significant_tokens(branch_text))
        domain_profile = self._rvk_domain_profile(abstract_text, matched_keywords)

        fit_score = 0
        domain_branch_terms = {
            "politics": {"politik", "politische", "internationale", "demokratie", "staat", "regierung", "konflikt"},
            "history": {"geschichte", "historische", "osteuropa", "sowjetunion", "zeitgeschichte"},
            "law": {"recht", "rechte", "vertrag", "völkerrecht", "voelkerrecht", "menschenrechte"},
            "media": {"medien", "presse", "kommunikation", "propaganda", "fernsehen"},
            "literature": {"literatur", "schriftsteller", "autobiograph", "biograph"},
            "religion": {"christliche", "theologie", "religion", "kirche"},
            "philosophy": {"philosophie", "ethik", "theorie"},
        }

        for domain, strength in domain_profile.items():
            if strength <= 0:
                continue
            hits = len(branch_tokens.intersection(domain_branch_terms.get(domain, set())))
            if hits:
                fit_score += min(strength, 4) * hits * 10

        # Penalize clearly mismatched major domains when the text strongly points elsewhere.
        if domain_profile.get("politics", 0) + domain_profile.get("history", 0) >= 3:
            if branch_tokens.intersection(domain_branch_terms["religion"]):
                fit_score -= 45
            if branch_tokens.intersection(domain_branch_terms["literature"]) and domain_profile.get("literature", 0) == 0:
                fit_score -= 20

        if domain_profile.get("law", 0) >= 2 and branch_tokens.intersection(domain_branch_terms["law"]):
            fit_score += 15

        if self._is_institution_library_rvk(candidate):
            if self._matches_specific_library_context(candidate, abstract_text, matched_keywords):
                fit_score += 10
            else:
                fit_score -= 120

        return fit_score

    @staticmethod
    def _is_institution_library_rvk(candidate: Dict[str, Any]) -> bool:
        """Detect RVK notations for single named libraries that are often catalog artifacts."""
        label = str(candidate.get("label", "") or "").lower()
        ancestor_path = str(candidate.get("ancestor_path", "") or "").lower()
        branch_text = f"{label} {ancestor_path}"
        return (
            "bibliothekswesen" in branch_text
            and (
                "einzelne bibliotheken" in branch_text
                or "einzelne deutsche bibliotheken" in branch_text
                or "bibliotheken d" in branch_text
            )
        )

    @staticmethod
    def _matches_specific_library_context(
        candidate: Dict[str, Any],
        abstract_text: str,
        matched_keywords: List[str],
    ) -> bool:
        """Keep single-library RVK only when the specific institution/location is really in the text."""
        context_text = " ".join(
            [
                str(abstract_text or "").lower(),
                " ".join(str(item or "").lower() for item in (matched_keywords or [])),
            ]
        )
        label = str(candidate.get("label", "") or "")
        distinctive_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", label)
            if token.lower() not in {
                "bibliothek",
                "bibliotheken",
                "landesbibliothek",
                "staats",
                "universitätsbibliothek",
                "universitaetsbibliothek",
                "sowie",
                "deutsche",
            }
        ]
        return any(token in context_text for token in distinctive_tokens)

    @staticmethod
    def _rvk_broadness_penalty(candidate: Dict[str, Any]) -> int:
        """Penalize overly broad or shallow RVK nodes."""
        ancestor_path = str(candidate.get("ancestor_path", "") or "")
        label = str(candidate.get("label", "") or "")
        register_entries = [str(item) for item in (candidate.get("register") or []) if str(item).strip()]
        notation = str(candidate.get("dk", "") or "")

        depth = len([part for part in ancestor_path.split(">") if part.strip()])
        label_tokens = re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", label)
        notation_core = re.sub(r"[^A-Z0-9]", "", notation)

        penalty = 0
        if depth <= 1:
            penalty += 60
        elif depth == 2:
            penalty += 35
        elif depth == 3:
            penalty += 15

        if len(label_tokens) <= 1 and len(register_entries) <= 1:
            penalty += 18
        if len(notation_core) <= 4:
            penalty += 12

        return penalty

    def _score_rvk_candidate(
        self,
        candidate: Dict[str, Any],
        original_abstract: str,
        rvk_anchor_keywords: Optional[List[str]] = None,
    ) -> int:
        """Deterministically score validated RVK candidates."""
        source = candidate.get("source", "catalog")
        status = candidate.get("rvk_validation_status", "standard")
        matched_keywords = [
            canonicalize_keyword(keyword).lower()
            for keyword in (candidate.get("matched_keywords") or [])
            if canonicalize_keyword(keyword)
        ]
        anchor_keywords = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip()).lower()
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }
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
        count_score = min(int(candidate.get("count", 0) or 0), 24) * 2
        keyword_support = len(set(matched_keywords)) * 12
        title_support = min(len(titles), 6)
        register_support = min(len(register_entries), 8)
        branch_fit = self._rvk_branch_fit_score(candidate, abstract_text, matched_keywords)
        broadness_penalty = self._rvk_broadness_penalty(candidate)
        anchor_match_bonus = 0
        if anchor_keywords:
            anchor_hits = len(anchor_keywords.intersection(set(matched_keywords)))
            if anchor_hits:
                anchor_match_bonus += anchor_hits * 24
            else:
                anchor_match_bonus -= 30

        return (
            source_weight
            + overlap_score
            + specificity
            + count_score
            + keyword_support
            + title_support
            + register_support
            + branch_fit
            + anchor_match_bonus
            - broadness_penalty
        )

    def _score_rvk_shortlist_with_llm(
        self,
        shortlist: List[Dict[str, Any]],
        original_abstract: str,
        model: Optional[str],
        provider: Optional[str],
        stream_callback: Optional[callable] = None,
        mode=None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Best-effort LLM scoring for a fixed RVK shortlist."""
        if not shortlist or not self.alima_manager:
            return {}

        from ..core.data_models import AbstractData
        from ..core.json_response_parser import parse_json_response

        shortlist_text = PipelineResultFormatter.format_dk_results_for_prompt(
            shortlist,
            max_results=len(shortlist),
        )
        abstract_data = AbstractData(
            abstract=original_abstract,
            keywords=shortlist_text,
        )

        try:
            task_state = self.alima_manager.analyze_abstract(
                abstract_data=abstract_data,
                task="rvk_scoring",
                model=model,
                provider=provider,
                stream_callback=None,
                mode=mode,
                **(llm_kwargs or {}),
            )
            if task_state.status == "failed":
                return {}

            parsed = parse_json_response(task_state.analysis_result.full_text) or {}
            scores = parsed.get("scores", [])
            results: Dict[str, Dict[str, Any]] = {}
            for item in scores:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code", "")).strip()
                normalized = canonicalize_rvk_notation(code[4:].strip() if code.upper().startswith("RVK ") else code)
                if not normalized:
                    continue
                thematic_fit = int(item.get("thematic_fit", 0) or 0)
                branch_fit = int(item.get("branch_fit", 0) or 0)
                specificity = int(item.get("specificity", 0) or 0)
                total = int(item.get("total_score", 0) or 0)
                if not total:
                    total = thematic_fit * 4 + branch_fit * 4 + specificity * 2
                results[normalized] = {
                    "thematic_fit": thematic_fit,
                    "branch_fit": branch_fit,
                    "specificity": specificity,
                    "total_score": total,
                    "reason": str(item.get("reason", "") or ""),
                }
            if stream_callback and results:
                stream_callback(
                    f"ℹ️ RVK-Shortlist per LLM bewertet: {len(results)} Kandidaten\n",
                    "dk_classification",
                )
            return results
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"RVK shortlist scoring failed: {exc}")
            if stream_callback:
                stream_callback(
                    f"⚠️ RVK-Shortlist-Scoring fehlgeschlagen, nutze Heuristik: {str(exc)}\n",
                    "dk_classification",
                )
            return {}

    def _build_dk_semantic_profile(
        self,
        selected_dk_codes: List[str],
        candidate_results: List[Dict[str, Any]],
        max_keywords: int = 6,
        max_titles: int = 3,
        max_codes: int = 6,
    ) -> str:
        """Build compact semantic hints from the selected DK classes."""
        if not selected_dk_codes or not candidate_results:
            return ""

        aggregated: Dict[str, Dict[str, Any]] = {}
        for candidate in candidate_results:
            # DK and DDC both contribute thematic hints; RVK is excluded
            # (it is what we are ranking). - Claude Generated
            cls_type = str(candidate.get("classification_type", candidate.get("type", "DK"))).upper()
            if cls_type not in ("DK", "DDC"):
                continue

            raw_code = str(candidate.get("dk", "") or "").strip()
            if not raw_code:
                continue

            prefix = "DDC" if cls_type == "DDC" else "DK"
            key = f"{prefix} {raw_code}"
            current = aggregated.setdefault(
                key,
                {
                    "matched_keywords": [],
                    "titles": [],
                    "count": 0,
                },
            )
            current["count"] += int(candidate.get("count", 0) or 0)

            seen_keywords = set(current["matched_keywords"])
            for keyword in (candidate.get("matched_keywords") or candidate.get("keywords") or []):
                clean_keyword = str(keyword or "").strip()
                if clean_keyword and clean_keyword not in seen_keywords:
                    current["matched_keywords"].append(clean_keyword)
                    seen_keywords.add(clean_keyword)

            seen_titles = set(current["titles"])
            for title in (candidate.get("titles") or []):
                clean_title = str(title or "").strip()
                if clean_title and clean_title not in seen_titles:
                    current["titles"].append(clean_title)
                    seen_titles.add(clean_title)

        lines = []
        for code in selected_dk_codes:
            clean_code = str(code or "").strip()
            if not clean_code:
                continue
            upper = clean_code.upper()
            if upper.startswith("RVK "):
                continue

            # Normalise the prefix (uppercase) so DK/DDC codes match the
            # aggregated keys; bare codes default to DK. - Claude Generated
            if upper.startswith("DK "):
                normalized = "DK " + clean_code[3:].strip()
            elif upper.startswith("DDC "):
                normalized = "DDC " + clean_code[4:].strip()
            else:
                normalized = f"DK {clean_code}"
            data = aggregated.get(normalized)
            if not data:
                continue

            keywords = ", ".join(
                cleaned
                for cleaned in (repair_display_text(item) for item in data.get("matched_keywords", [])[:max_keywords])
                if cleaned
            )
            titles = " | ".join(
                cleaned
                for cleaned in (repair_display_text(item) for item in data.get("titles", [])[:max_titles])
                if cleaned
            )
            parts = [normalized]
            if keywords:
                parts.append(f"Schlagworte: {keywords}")
            if titles:
                parts.append(f"Beispieltitel: {titles}")
            if data.get("count"):
                parts.append(f"Haeufigkeit: {int(data['count'])}")
            lines.append(" | ".join(parts))
            if len(lines) >= max_codes:
                break

        return "\n".join(lines)

    def _build_rvk_scoring_shortlist(
        self,
        candidate_results: List[Dict[str, Any]],
        original_abstract: str,
        rvk_anchor_keywords: Optional[List[str]] = None,
        max_standard: int = 8,
        max_nonstandard: int = 3,
    ) -> List[Dict[str, Any]]:
        """Build a compact validated RVK shortlist for the second-pass scorer."""
        aggregated: Dict[str, Dict[str, Any]] = {}
        anchor_keywords = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip()).lower()
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }

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

            current = aggregated.get(normalized)
            if current is None:
                current = dict(candidate)
                current["dk"] = normalized
                current["matched_keywords"] = list(candidate.get("matched_keywords", []) or candidate.get("keywords", []) or [])
                current["titles"] = list(candidate.get("titles", []) or [])
                current["register"] = list(candidate.get("register", []) or [])
                current["count"] = int(candidate.get("count", 0) or 0)
                aggregated[normalized] = current
                continue

            current["count"] = int(current.get("count", 0) or 0) + int(candidate.get("count", 0) or 0)
            for field in ("matched_keywords", "titles", "register"):
                existing = list(current.get(field, []) or [])
                seen = set(existing)
                incoming = candidate.get(field, []) or candidate.get("keywords", []) or []
                for value in incoming:
                    clean_value = str(value or "").strip()
                    if clean_value and clean_value not in seen:
                        existing.append(clean_value)
                        seen.add(clean_value)
                current[field] = existing

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

        standard_candidates = []
        nonstandard_candidates = []
        for candidate in aggregated.values():
            candidate["_score"] = self._score_rvk_candidate(
                candidate,
                original_abstract,
                rvk_anchor_keywords=rvk_anchor_keywords,
            )
            matched_keyword_set = {
                canonicalize_keyword(keyword).lower()
                for keyword in (candidate.get("matched_keywords") or [])
                if canonicalize_keyword(keyword)
            }
            candidate["_anchor_hit_count"] = len(anchor_keywords.intersection(matched_keyword_set))
            candidate["_source_rank"] = _source_rank(str(candidate.get("source", "catalog") or "catalog"))
            candidate["_status_rank"] = _status_rank(str(candidate.get("rvk_validation_status", "standard") or "standard"))
            status = str(candidate.get("rvk_validation_status", "standard") or "standard")
            if status == "standard":
                standard_candidates.append(candidate)
            elif status in {"non_standard", "validation_error"}:
                nonstandard_candidates.append(candidate)

        def _sort_key(item: Dict[str, Any]):
            return (
                -int(item.get("_anchor_hit_count", 0)),
                -int(item.get("_source_rank", 0)),
                -int(item.get("_status_rank", 0)),
                -int(item.get("_score", 0)),
                item.get("dk", ""),
            )

        standard_candidates.sort(key=_sort_key)
        nonstandard_candidates.sort(key=_sort_key)

        shortlist = standard_candidates[:max_standard]
        if not shortlist:
            shortlist = nonstandard_candidates[:max_nonstandard]
        elif len(shortlist) < max_standard:
            remaining = max_nonstandard
            for candidate in nonstandard_candidates:
                if remaining <= 0:
                    break
                shortlist.append(candidate)
                remaining -= 1

        return shortlist

    def _select_final_rvk_with_dk_context(
        self,
        candidate_results: List[Dict[str, Any]],
        original_abstract: str,
        selected_dk_codes: List[str],
        rvk_anchor_keywords: Optional[List[str]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        mode=None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Use a DK-guided second LLM pass to rank a fixed validated RVK shortlist."""
        shortlist = self._build_rvk_scoring_shortlist(
            candidate_results,
            original_abstract,
            rvk_anchor_keywords=rvk_anchor_keywords,
        )
        if not shortlist:
            return []

        dk_profile = self._build_dk_semantic_profile(selected_dk_codes, candidate_results)
        abstract_for_scoring = str(original_abstract or "").strip()
        if dk_profile:
            abstract_for_scoring += (
                "\n\nDK-Profil aus der bereits gewaehlten DK-Seite:\n"
                f"{dk_profile}\n"
                "Nutze dieses DK-Profil als zusaetzlichen thematischen Hinweis für die RVK-Auswahl."
            )

        if stream_callback:
            stream_callback(
                f"\n\nℹ️ RVK-Zweitranking mit DK-Profil: {len(shortlist)} Kandidaten\n",
                "dk_classification",
            )
            if dk_profile:
                stream_callback(
                    "DK-Profil für RVK-Zweitranking:\n"
                    + "\n".join(f"  {line}" for line in dk_profile.splitlines() if line.strip())
                    + "\n",
                    "dk_classification",
                )
            shortlist_lines = []
            for candidate in shortlist:
                line = f"  RVK {candidate['dk']}"
                if candidate.get("label"):
                    line += f" | {candidate['label']}"
                if candidate.get("ancestor_path"):
                    line += f" | Pfad: {candidate['ancestor_path']}"
                source = str(candidate.get("source", "catalog") or "catalog")
                status = str(candidate.get("rvk_validation_status", "standard") or "standard")
                line += f" | Quelle: {source}"
                if status != "standard":
                    line += f" ({status})"
                shortlist_lines.append(line)
            if shortlist_lines:
                stream_callback(
                    "RVK-Kandidaten für DK-basiertes Zweitranking:\n"
                    + "\n".join(shortlist_lines)
                    + "\n",
                    "dk_classification",
                )

        llm_scores = self._score_rvk_shortlist_with_llm(
            shortlist,
            abstract_for_scoring,
            model=model,
            provider=provider,
            stream_callback=stream_callback,
            mode=mode,
            llm_kwargs=llm_kwargs,
        )
        if not llm_scores:
            return []

        scored_candidates = []
        for candidate in shortlist:
            score = llm_scores.get(candidate["dk"])
            if not score:
                continue
            candidate_copy = dict(candidate)
            candidate_copy["_llm_total_score"] = int(score.get("total_score", 0) or 0)
            candidate_copy["_llm_thematic_fit"] = int(score.get("thematic_fit", 0) or 0)
            candidate_copy["_llm_branch_fit"] = int(score.get("branch_fit", 0) or 0)
            candidate_copy["_llm_specificity"] = int(score.get("specificity", 0) or 0)
            candidate_copy["_llm_reason"] = str(score.get("reason", "") or "")
            scored_candidates.append(candidate_copy)

        if not scored_candidates:
            return []

        standard_candidates = [
            candidate for candidate in scored_candidates
            if str(candidate.get("rvk_validation_status", "standard") or "standard") == "standard"
        ]
        nonstandard_candidates = [
            candidate for candidate in scored_candidates
            if str(candidate.get("rvk_validation_status", "standard") or "standard") in {"non_standard", "validation_error"}
        ]

        def _sort_key(item: Dict[str, Any]):
            return (
                -int(item.get("_llm_total_score", 0)),
                -int(item.get("_llm_thematic_fit", 0)),
                -int(item.get("_llm_branch_fit", 0)),
                -int(item.get("_llm_specificity", 0)),
                -int(item.get("_anchor_hit_count", 0)),
                -int(item.get("_score", 0)),
                item.get("dk", ""),
            )

        standard_candidates.sort(key=_sort_key)
        nonstandard_candidates.sort(key=_sort_key)

        if stream_callback:
            score_lines = []
            for candidate in sorted(scored_candidates, key=_sort_key):
                line = (
                    f"  RVK {candidate['dk']}: total={int(candidate.get('_llm_total_score', 0))}, "
                    f"thematisch={int(candidate.get('_llm_thematic_fit', 0))}, "
                    f"Pfad={int(candidate.get('_llm_branch_fit', 0))}, "
                    f"Spezifitaet={int(candidate.get('_llm_specificity', 0))}"
                )
                reason = str(candidate.get("_llm_reason", "") or "").strip()
                if reason:
                    line += f" | {reason}"
                score_lines.append(line)
            if score_lines:
                stream_callback(
                    "RVK-Bewertung aus DK-basiertem Zweitranking:\n"
                    + "\n".join(score_lines)
                    + "\n",
                    "dk_classification",
                )

        selected: List[str] = []
        selected_branches = set()
        for candidate in standard_candidates:
            branch = str(candidate.get("branch_family", "") or "")
            if branch and branch in selected_branches and len(selected) >= 1:
                continue
            selected.append(f"RVK {candidate['dk']}")
            if branch:
                selected_branches.add(branch)
            if len(selected) >= 2:
                break

        if not selected and nonstandard_candidates:
            selected.append(f"RVK {nonstandard_candidates[0]['dk']}")

        if stream_callback and selected:
            stream_callback(
                "RVK-Auswahl nach DK-basiertem Zweitranking:\n"
                + "\n".join(f"  {code}" for code in selected)
                + "\n",
                "dk_classification",
            )

        return selected

    def _select_final_rvk_candidates(
        self,
        candidate_results: List[Dict[str, Any]],
        original_abstract: str,
        max_standard: int = 2,
        max_nonstandard: int = 1,
        rvk_anchor_keywords: Optional[List[str]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        mode=None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
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

        anchor_keywords = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip()).lower()
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }

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
            enriched["_score"] = self._score_rvk_candidate(
                enriched,
                original_abstract,
                rvk_anchor_keywords=rvk_anchor_keywords,
            )
            enriched["_branch"] = str(enriched.get("branch_family", "") or "")
            matched_keyword_set = {
                canonicalize_keyword(keyword).lower()
                for keyword in (enriched.get("matched_keywords") or [])
                if canonicalize_keyword(keyword)
            }
            enriched["_anchor_hits"] = sorted(anchor_keywords.intersection(matched_keyword_set))
            enriched["_anchor_hit_count"] = len(enriched["_anchor_hits"])
            enriched["_source_rank"] = _source_rank(str(enriched.get("source", "catalog") or "catalog"))
            enriched["_status_rank"] = _status_rank(str(enriched.get("rvk_validation_status", "standard") or "standard"))

            status = enriched.get("rvk_validation_status", "standard")
            if status == "standard":
                standard_candidates.append(enriched)
            elif status in {"non_standard", "validation_error"}:
                nonstandard_candidates.append(enriched)

        def _sort_key(item: Dict[str, Any]):
            return (
                -int(item.get("_anchor_hit_count", 0)),
                -int(item.get("_source_rank", 0)),
                -int(item.get("_status_rank", 0)),
                -int(item.get("_score", 0)),
                item.get("dk", ""),
            )

        standard_candidates.sort(key=_sort_key)
        nonstandard_candidates.sort(key=_sort_key)

        def _prefilter_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not candidates:
                return []
            if anchor_keywords:
                anchored = [candidate for candidate in candidates if candidate.get("_anchor_hit_count", 0) > 0]
                if anchored:
                    candidates = anchored
            shortlist = candidates[:8]
            llm_scores = self._score_rvk_shortlist_with_llm(
                shortlist,
                original_abstract,
                model=model,
                provider=provider,
                stream_callback=stream_callback,
                mode=mode,
                llm_kwargs=llm_kwargs,
            )
            for candidate in shortlist:
                llm_score = llm_scores.get(candidate["dk"], {})
                candidate["_llm_total_score"] = int(llm_score.get("total_score", 0) or 0)
                candidate["_llm_reason"] = str(llm_score.get("reason", "") or "")
                candidate["_combined_score"] = (
                    int(candidate.get("_score", 0))
                    + int(candidate.get("_llm_total_score", 0)) * 12
                    + int(candidate.get("_anchor_hit_count", 0)) * 18
                )
            shortlist.sort(
                key=lambda item: (
                    -int(item.get("_combined_score", 0)),
                    -int(item.get("_anchor_hit_count", 0)),
                    item.get("dk", ""),
                )
            )
            return shortlist

        def _pick_diverse(candidates: List[Dict[str, Any]], limit: int) -> List[str]:
            selected = []
            selected_candidates: List[Dict[str, Any]] = []
            covered_anchors = set()
            remaining = list(candidates)

            while remaining and len(selected) < limit:
                best_idx = None
                best_value = None
                for idx, candidate in enumerate(remaining):
                    anchor_hits = set(candidate.get("_anchor_hits", []))
                    new_coverage = len(anchor_hits - covered_anchors)
                    overlap = len(anchor_hits.intersection(covered_anchors))
                    dynamic_score = (
                        int(candidate.get("_combined_score", candidate.get("_score", 0)))
                        + new_coverage * 45
                        - overlap * 15
                    )
                    branch = candidate.get("_branch", "")
                    if branch and any(selected_candidate.get("_branch", "") == branch for selected_candidate in selected_candidates):
                        dynamic_score -= 8
                    candidate_value = (dynamic_score, int(candidate.get("_anchor_hit_count", 0)), -idx)
                    if best_value is None or candidate_value > best_value:
                        best_value = candidate_value
                        best_idx = idx

                if best_idx is None:
                    break

                chosen = remaining.pop(best_idx)
                selected_candidates.append(chosen)
                selected.append(f"RVK {chosen['dk']}")
                covered_anchors.update(chosen.get("_anchor_hits", []))
            return selected

        shortlisted_standard = _prefilter_candidates(standard_candidates)
        shortlisted_nonstandard = _prefilter_candidates(nonstandard_candidates)

        if stream_callback:
            if shortlisted_standard:
                preview = ", ".join(f"RVK {item['dk']}" for item in shortlisted_standard[:5])
                stream_callback(
                    f"ℹ️ RVK-Shortlist (standard): {preview}\n",
                    "dk_classification",
                )
            elif shortlisted_nonstandard:
                preview = ", ".join(f"RVK {item['dk']}" for item in shortlisted_nonstandard[:5])
                stream_callback(
                    f"ℹ️ RVK-Shortlist (lokal): {preview}\n",
                    "dk_classification",
                )

        if shortlisted_standard:
            return _pick_diverse(shortlisted_standard, max_standard)
        return _pick_diverse(shortlisted_nonstandard, max_nonstandard)

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
        """Thin wrapper — logic lives module-level for reuse by formatters - Claude Generated"""
        return flatten_keyword_centric_results(keyword_results)

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

    @staticmethod
    def _strip_rvk_from_keyword_results(
        keyword_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop RVK-typed classifications from keyword-centric results - Claude Generated

        Used when ``rvk_enabled=False``: RVK is handled out-of-band (e.g. via
        the ``rvk_lookup`` tool), so neither the catalog excerpt nor the GUI
        transparency view should carry RVK entries.
        """
        cleaned: List[Dict[str, Any]] = []
        for entry in keyword_results or []:
            if not isinstance(entry, dict):
                cleaned.append(entry)
                continue
            classifications = entry.get("classifications")
            if isinstance(classifications, list):
                new_entry = dict(entry)
                new_entry["classifications"] = [
                    c for c in classifications
                    if str(
                        (c or {}).get("classification_type", (c or {}).get("type", "DK"))
                    ).upper() != "RVK"
                ]
                cleaned.append(new_entry)
            else:
                cleaned.append(entry)
        return cleaned

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
        rvk_anchor_keywords: Optional[List[str]] = None,
        rvk_enabled: bool = True,  # Claude Generated - False skips all RVK anchor/API work
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

        # Initialize catalog client - supports BiblioClient or MarcXmlClient - Claude Generated
        try:
            # Determine catalog type from config and fill missing parameters - Claude Generated
            try:
                from .config_manager import ConfigManager
                config_manager = ConfigManager()
                catalog_config = config_manager.get_catalog_config()
                catalog_type = getattr(catalog_config, 'catalog_type', 'libero_soap')
                if catalog_type == 'auto':
                    catalog_type = catalog_config.get_catalog_type() if hasattr(catalog_config, 'get_catalog_type') else 'libero_soap'

                # Fill missing catalog parameters from config — fixes dk_search_agentic
                # which calls execute_dk_search without passing catalog config.
                if not catalog_token or not catalog_token.strip():
                    catalog_token = getattr(catalog_config, 'catalog_token', '') or ''
                if not catalog_search_url or not catalog_search_url.strip():
                    catalog_search_url = getattr(catalog_config, 'catalog_search_url', '') or ''
                if not catalog_details_url or not catalog_details_url.strip():
                    catalog_details_url = getattr(catalog_config, 'catalog_details_url', '') or ''
                if not catalog_web_search_url or not catalog_web_search_url.strip():
                    catalog_web_search_url = getattr(catalog_config, 'catalog_web_search_url', '') or ''
                if not catalog_web_record_url or not catalog_web_record_url.strip():
                    catalog_web_record_url = getattr(catalog_config, 'catalog_web_record_url', '') or ''
            except Exception as cfg_err:
                if self.logger:
                    self.logger.debug(f"Config load failed, using default: {cfg_err}")
                catalog_type = 'libero_soap'  # Default to original behavior

            # Log catalog configuration status - Claude Generated
            # (skip the Libero token warning when finc is the active backend) - Claude Generated
            _finc_url_for_warn = getattr(catalog_config, 'finc_base_url', '') if 'catalog_config' in dir() else ''
            _finc_dk_active = (
                isinstance(_finc_url_for_warn, str) and bool(_finc_url_for_warn.strip())
                and bool(getattr(catalog_config, 'finc_dk_enabled', False) if 'catalog_config' in dir() else False)
            )
            if not _finc_dk_active and (not catalog_token or not catalog_token.strip()):
                if self.logger:
                    self.logger.warning("No catalog token provided - BiblioClient will use web scraping fallback")
                if stream_callback:
                    stream_callback("Kein Katalog-Token: Web-Fallback wird verwendet\n", "dk_search")
                catalog_token = ""  # Empty token triggers automatic web fallback

            # DK/RVK source resolved from the active CLASSIFICATION-capable search
            # providers (finc opt-in → custom plugin → SRU/Libero), replacing the
            # former hand-wired FincCatalogClient/MarcXmlClient/BiblioClient
            # if-elif (the D-4 site in search/factory.py). Precedence preserves
            # the operator's June-2026 order; any catalog plugin declaring the
            # CLASSIFICATION capability + a dk_extractor() becomes a DK source with
            # no change here. finc DK stays opt-in (finc_dk_enabled + a real URL).
            # All extractors share the extract_dk_classifications_for_keywords
            # contract, so the per-keyword loop below is unchanged. - Claude Generated
            from src.core.search.factory import resolve_dk_extractor

            try:
                _full_config = config_manager.load_config()
            except Exception:
                _full_config = None
            extractor = resolve_dk_extractor(
                config=_full_config,
                logger_=self.logger,
                stream_callback=(lambda m: stream_callback(m, "dk_search")) if stream_callback else None,
                debug=(self.logger.level <= 10) if self.logger else False,
                finc_base_url=getattr(catalog_config, 'finc_base_url', '') if 'catalog_config' in dir() else '',
                finc_web_record_url=getattr(catalog_config, 'finc_web_record_url', '') or '' if 'catalog_config' in dir() else '',
                finc_institution_filter=getattr(catalog_config, 'finc_institution_filter', '') or '' if 'catalog_config' in dir() else '',
                finc_timeout=getattr(catalog_config, 'finc_timeout', 30) or 30 if 'catalog_config' in dir() else 30,
                finc_default_limit=getattr(catalog_config, 'finc_default_limit', 50) or 50 if 'catalog_config' in dir() else 50,
                finc_dk_enabled=bool(getattr(catalog_config, 'finc_dk_enabled', False)) if 'catalog_config' in dir() else False,
                catalog_type=catalog_type,
                sru_preset=getattr(catalog_config, 'sru_preset', '') if 'catalog_config' in dir() else '',
                sru_base_url=getattr(catalog_config, 'sru_base_url', '') if 'catalog_config' in dir() else '',
                sru_max_records=getattr(catalog_config, 'sru_max_records', 50) if 'catalog_config' in dir() else 50,
                catalog_token=catalog_token,
                catalog_search_url=catalog_search_url,
                catalog_details_url=catalog_details_url,
                catalog_web_search_url=catalog_web_search_url,
                catalog_web_record_url=catalog_web_record_url,
            )
            # Preserve the old per-source progress line. - Claude Generated
            _dk_src_label = {
                "FincCatalogClient": "finc-Katalog (Titelliste + udk_raw pro Titel)",
                "MarcXmlClient": "MARC XML SRU",
                "BiblioClient": "Libero-Katalog",
            }.get(type(extractor).__name__, type(extractor).__name__)
            if self.logger:
                self.logger.info(f"DK source resolved: {type(extractor).__name__}")
            if stream_callback:
                stream_callback(f"Verwende {_dk_src_label} für DK-Suche\n", "dk_search")

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
        rvk_anchor_entries = gnd_keyword_entries
        rvk_anchor_search_keywords = list(gnd_validated_keywords)
        if rvk_anchor_keywords:
            anchor_term_lookup: Dict[str, str] = {}
            for keyword in rvk_anchor_keywords:
                clean_keyword = keyword.split("(GND-ID:")[0].strip()
                normalized_keyword = canonicalize_keyword(clean_keyword)
                if normalized_keyword and clean_keyword:
                    anchor_term_lookup[normalized_keyword] = clean_keyword
            normalized_anchor_ids = {
                extract_gnd_id(keyword)
                for keyword in rvk_anchor_keywords
                if extract_gnd_id(keyword)
            }
            normalized_anchor_terms = {
                canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
                for keyword in rvk_anchor_keywords
                if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
            }
            filtered_anchor_entries = [
                entry for entry in gnd_keyword_entries
                if (
                    entry.get("gnd_id") in normalized_anchor_ids
                    or canonicalize_keyword(entry.get("keyword", "")) in normalized_anchor_terms
                )
            ]
            filtered_anchor_keywords = [
                keyword for keyword in gnd_validated_keywords
                if canonicalize_keyword(keyword) in normalized_anchor_terms
            ]
            supplemental_anchor_terms = [
                anchor_term_lookup[normalized_term]
                for normalized_term in normalized_anchor_terms
                if normalized_term not in {
                    canonicalize_keyword(keyword) for keyword in filtered_anchor_keywords
                }
            ]
            if filtered_anchor_entries:
                rvk_anchor_entries = filtered_anchor_entries
            if filtered_anchor_keywords or supplemental_anchor_terms:
                rvk_anchor_search_keywords = deduplicate_canonical_keywords(
                    filtered_anchor_keywords + supplemental_anchor_terms
                )

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
            if rvk_enabled and (rvk_anchor_keywords or rvk_anchor_entries):
                rvk_anchor_preview_terms = []
                if rvk_anchor_keywords:
                    rvk_anchor_preview_terms = [
                        keyword.split("(GND-ID:")[0].strip()
                        for keyword in rvk_anchor_keywords
                        if keyword.split("(GND-ID:")[0].strip()
                    ]
                if not rvk_anchor_preview_terms:
                    rvk_anchor_preview_terms = [
                        entry.get("keyword", "") for entry in rvk_anchor_entries if entry.get("keyword")
                    ]
                rvk_anchor_preview = ", ".join(rvk_anchor_preview_terms[:6])
                if len(rvk_anchor_preview_terms) > 6:
                    rvk_anchor_preview += f", +{len(rvk_anchor_preview_terms) - 6} weitere"
                if rvk_anchor_preview:
                    stream_callback(
                        f"ℹ️ RVK-Ankerbegriffe: {rvk_anchor_preview}\n",
                        "dk_search"
                    )

        # Execute catalog search with Per-Keyword Feedback - Claude Generated QUICK-FIX
        # IMPORTANT: Loop over keywords individually to provide per-keyword status feedback
        # This replaces the old single-batch call with individual keyword searches
        try:
            import inspect
            # Not every extractor accepts force_update (MarcXmlClient and custom
            # DK plugins don't) — detect once and pass it only when supported, so
            # any CLASSIFICATION-capable provider's extractor works here. - Claude Generated
            try:
                _dk_accepts_force = "force_update" in inspect.signature(
                    extractor.extract_dk_classifications_for_keywords
                ).parameters
            except (TypeError, ValueError):
                _dk_accepts_force = False

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
                    # Search THIS keyword only (not all keywords).
                    _dk_kwargs = {"keywords": [keyword], "max_results": max_results}
                    if _dk_accepts_force:
                        _dk_kwargs["force_update"] = force_update
                    kw_results = extractor.extract_dk_classifications_for_keywords(**_dk_kwargs)

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

            if rvk_enabled:
                dk_search_results = self._validate_catalog_rvk_candidates(
                    dk_search_results,
                    stream_callback=stream_callback,
                    rvk_anchor_keywords=rvk_anchor_keywords,
                )
                dk_search_results = self._inject_rvk_api_fallback(
                    rvk_anchor_search_keywords or final_search_keywords,
                    dk_search_results,
                    gnd_keyword_entries=rvk_anchor_entries,
                    stream_callback=stream_callback,
                )
                self._emit_rvk_source_diagnostics(
                    dk_search_results,
                    stream_callback=stream_callback,
                    step_id="dk_search",
                )
            else:
                # RVK handled out-of-band (rvk_lookup tool) — no anchor
                # validation, no RVK-API calls, no RVK in the result. - Claude Generated
                dk_search_results = self._strip_rvk_from_keyword_results(dk_search_results)

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
                if rvk_enabled:
                    dk_search_results = self._validate_catalog_rvk_candidates(
                        dk_search_results,
                        stream_callback=stream_callback,
                        rvk_anchor_keywords=rvk_anchor_keywords,
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
                else:
                    dk_search_results = self._strip_rvk_from_keyword_results(dk_search_results)
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
        keywords, gnd_classes, init_analysis, llm_title = _run_classic_step(
            "initialisation",
            {
                "task": (init_cfg.task or "initialisation") if init_cfg else "initialisation",
                "provider": init_cfg.provider if init_cfg else None,
                "model": init_cfg.model if init_cfg else None,
            },
            self.execute_initial_keyword_extraction,
            abstract_text=input_text,
            provider=init_cfg.provider if init_cfg else None,
            model=init_cfg.model if init_cfg else None,
            task=init_cfg.task or "initialisation" if init_cfg else "initialisation",
            stream_callback=_cb("initialisation"),
        )

        # ── Step 2: GND search ───────────────────────────────────────────────
        if stream_callback:
            stream_callback(f"\n▶ [search] {len(keywords)} keywords\n", "search")
        search_results = _run_classic_step(
            "search",
            {"keywords_count": len(keywords), "suggesters": config.search_suggesters},
            self.execute_gnd_search,
            keywords=keywords,
            suggesters=config.search_suggesters,
            stream_callback=_cb("search"),
        )

        # ── Step 3: keywords (final analysis) ────────────────────────────────
        kw_cfg = _step("keywords")
        if stream_callback:
            stream_callback(f"\n▶ [keywords]\n", "keywords")
        final_keywords, gnd_compliant, kw_analysis = _run_classic_step(
            "keywords",
            {
                "task": (kw_cfg.task or "keywords") if kw_cfg else "keywords",
                "provider": kw_cfg.provider if kw_cfg else None,
                "model": kw_cfg.model if kw_cfg else None,
            },
            self.execute_final_keyword_analysis,
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
            search_results=search_results,
            final_llm_analysis=kw_analysis,
            working_title=llm_title,
        )

        # ── Step 4: DK classification (optional) ─────────────────────────────
        dk_cfg = _step("dk_classification")
        if dk_cfg and getattr(dk_cfg, "enabled", False):
            if stream_callback:
                stream_callback(f"\n▶ [dk_classification]\n", "dk_classification")
            try:
                rvk_anchor_keywords = self._derive_rvk_anchor_keywords(
                    final_keywords,
                    kw_analysis,
                    original_abstract=input_text,
                    initial_keywords=keywords,
                    search_results=search_results,
                    stream_callback=_cb("dk_classification"),
                )
                dk_search = _run_classic_step(
                    "dk_search",
                    {"keywords_count": len(final_keywords)},
                    self.execute_dk_search,
                    keywords=final_keywords,
                    rvk_anchor_keywords=rvk_anchor_keywords,
                    stream_callback=_cb("dk_classification"),
                )
                state.dk_search_results = dk_search.get("keyword_results", [])
                state.dk_search_results_flattened = dk_search.get("classifications", [])
                state.dk_statistics = dk_search.get("statistics")
                dk_classes, dk_analysis = _run_classic_step(
                    "dk_classification",
                    {
                        "provider": dk_cfg.provider if dk_cfg else None,
                        "model": dk_cfg.model if dk_cfg else None,
                    },
                    self.execute_dk_classification,
                    original_abstract=input_text,
                    dk_search_results=dk_search.get("classifications", []),
                    provider=dk_cfg.provider if dk_cfg else None,
                    model=dk_cfg.model if dk_cfg else None,
                    rvk_anchor_keywords=rvk_anchor_keywords,
                    stream_callback=_cb("dk_classification"),
                )
                state.dk_classifications = dk_classes
                state.dk_llm_analysis = dk_analysis
                state.rvk_provenance = getattr(dk_analysis, "rvk_provenance", {})
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"DK classification failed (non-fatal): {e}")

        return state

