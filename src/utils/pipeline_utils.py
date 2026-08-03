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
from ..core.gnd_search_core import merge_code_entry
from .classification_systems import normalize_classifications
from .error_visibility import log_caught
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
# DK/RVK classification methods extracted into behavioral mixins (July 19)
from ._pipeline_dk_steps import DkStepsMixin
from ._pipeline_rvk_scoring import RvkScoringMixin


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


class PipelineStepExecutor(DkStepsMixin, RvkScoringMixin):
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

        # Authority DDC from the local GND store, merged onto the pool subjects
        # by GND-ID — same read-back the agentic path does. Without it the
        # filled gnd_local store (migration + DNB enrichment) never reaches the
        # pool and every classification stays co-occurrence. - Claude Generated
        self._enrich_search_results_with_authority_ddc(search_results)

        return search_results

    def _enrich_search_results_with_authority_ddc(
        self, search_results: Dict[str, Dict[str, Any]]
    ) -> None:
        """Merge authority DDC (local GND store) onto the nested classic results.

        Flattens the ``{term: {title: entry}}`` view to the entry list the
        shared ``merge_authority_ddc`` atom expects. Best-effort: a store read
        failure must not fail the search. - Claude Generated
        """
        from src.core.gnd_search_core import merge_authority_ddc

        entries = [
            data
            for results in (search_results or {}).values()
            for data in (results or {}).values()
            if isinstance(data, dict)
        ]
        all_ids = {
            gid for e in entries for gid in (e.get("gnd_ids") or []) if gid
        }
        if not all_ids:
            return
        try:
            facts = self.cache_manager.get_gnd_facts_batch(list(all_ids))
            ddcs_by_gid = {
                gid: getattr(fact, "ddcs", "")
                for gid, fact in (facts or {}).items()
            }
            merge_authority_ddc(entries, ddcs_by_gid)
        except Exception as e:
            # Best-effort: authority enrichment must never fail the search.
            log_caught(self.logger, e, "authority-DDC enrichment")

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
                            len(data.get("gnd_ids", set()))
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
                # concept_data is: {keyword: {gnd_ids: set(), classifications: {system: set()}, count: int}}
                for keyword, data in concept_data.items():
                    if concept in merged_results:
                        # Concept already exists as search term - merge at keyword level
                        if keyword in merged_results[concept]:
                            # Merge GND-IDs, classifications, and count via the shared atom
                            merge_code_entry(
                                merged_results[concept][keyword], data,
                                code_fields=("gnd_ids",),
                                classifications_field="classifications",
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
                gnd_ids = data.get("gnd_ids", set())
                if not gnd_ids:  # Subject from catalog without GND-ID
                    catalog_subjects_found += 1
                    
                    # Check cache first - Claude Generated - use new method to get all GND-IDs
                    cached_gnd_ids = self.cache_manager.get_all_gnd_ids_for_keyword(subject)
                    if cached_gnd_ids:
                        # Found in cache - add all GND-IDs
                        data["gnd_ids"].update(cached_gnd_ids)
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
                            len(data.get("gnd_ids", set()))
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
                                gnd_count = len(data.get("gnd_ids", set()))
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
                swb_gnd_ids = swb_data.get("gnd_ids", set())
                
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
                                        "gnd_ids": swb_gnd_ids.copy(),
                                        "classifications": normalize_classifications(
                                            swb_data.get("classifications")
                                        ),
                                    }
                                    swb_matches += len(swb_gnd_ids)
                                    if self.logger:
                                        self.logger.info(f"SWB add: '{swb_keyword}' (+{len(swb_gnd_ids)} GND-IDs) [matched via '{orig_keyword}']")
                                else:
                                    # Update existing SWB subject entry
                                    existing_data = original_results[search_term][swb_keyword]
                                    old_count = len(existing_data["gnd_ids"])
                                    merge_code_entry(
                                        existing_data, swb_data,
                                        code_fields=("gnd_ids",), count_field="",
                                        classifications_field="classifications",
                                    )
                                    new_count = len(existing_data["gnd_ids"])
                                    
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
                                "gnd_ids": swb_gnd_ids.copy(),
                                "classifications": normalize_classifications(
                                    swb_data.get("classifications")
                                ),
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
                gnd_ids = data.get("gnd_ids", set())
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
                gnd_ids = data.get("gnd_ids", set())

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
            pipeline_mode="classic",
        )

    def execute_complete_pipeline(
        self,
        input_text: str,
        pipeline_config=None,
        stream_callback: Optional[callable] = None,
        input_record_classifications: Optional[Dict[str, Any]] = None,
    ) -> "KeywordAnalysisState":
        """
        Execute a complete ALIMA pipeline synchronously without Qt dependencies.

        Chains: initialisation → search → keywords → (dk_classification if enabled)

        This is the preferred method for batch processing since it runs purely
        synchronously in any thread context (no QThread/event loop required).

        ``input_record_classifications`` (WP-D1 P2): the input record's own
        classifications (canonical ``{SYSTEM: [{code, origin}]}``) when the
        input came from a bibliographic record — stored on the state and fed
        into dk_classification as priors.

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
            pipeline_mode="classic",
            input_record_classifications=input_record_classifications or {},
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
                    record_priors=input_record_classifications or None,
                )
                state.dk_classifications = dk_classes
                state.dk_llm_analysis = dk_analysis
                state.rvk_provenance = getattr(dk_analysis, "rvk_provenance", {})
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"DK classification failed (non-fatal): {e}")

        return state

