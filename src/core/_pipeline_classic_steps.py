"""Classic per-step executors for PipelineManager - Claude Generated.

Split out of ``pipeline_manager.py`` (WP cleanup D). Verbatim mixin extraction:
the methods stay on ``PipelineManager`` via MRO, so no call site changes.

The classic 5-step pipeline's step bodies — input, initialisation, search,
keywords, DK-search, DK-classification — plus the two search-result shape
converters they use. Each delegates the real work to ``PipelineStepExecutor``
(shared with CLI/GUI/webapp) and wraps it in this class's state + streaming.
They read back into the rest of the class (``_stream_callback_adapter``,
``_wrap_stream_callback_for_input``) via ``self`` — one class across two files.

``PipelineStep`` appears only in type annotations here; with
``from __future__ import annotations`` those stay strings, so there is no
runtime import back into ``pipeline_manager`` and no import cycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .data_models import KeywordAnalysisState, LlmKeywordAnalysis, SearchResult
from ..utils.pipeline_utils import (
    PipelineResultFormatter,
    build_working_title,
    execute_input_extraction,
    extract_source_identifier,
)
from ..utils.pipeline_defaults import (
    DEFAULT_DK_MAX_RESULTS,
    DEFAULT_DK_FREQUENCY_THRESHOLD,
)


class ClassicStepExecutorMixin:
    """The classic per-step executors. Mixed into :class:`PipelineManager`."""

    def _get_default_task_for_step(self, step_id: str) -> str:
        """Get default prompt task for a pipeline step - Claude Generated"""
        task_mapping = {
            "initialisation": "initialisation",
            "keywords": "keywords",
            "dk_classification": "dk_classification",
            "input": "input",
            "search": "search"
        }
        return task_mapping.get(step_id, "keywords")

    def _execute_input_step(self, step: PipelineStep) -> bool:
        """Execute input processing step with file support - Claude Generated"""
        
        # Check if input data specifies a file path that needs processing
        input_data = step.input_data or {}
        input_type = input_data.get("type", "text")
        
        if input_type in ["file", "pdf", "image", "auto", "isbn", "ppn"] and "file_path" in input_data:
            # File-based input processing (isbn/ppn carry the identifier in
            # file_path — same envelope, resolved by the input-source registry)
            try:
                if self.stream_callback:
                    self.stream_callback("🔄 Verarbeite Datei-Input...", "input")

                file_path = input_data["file_path"]
                record_sink = {}  # WP-D1 P2: bib_lookup deposits the BibRecord here - Claude Generated
                extracted_text, source_info, extraction_method = execute_input_extraction(
                    llm_service=self.llm_service,
                    input_source=file_path,
                    input_type=input_type,
                    stream_callback=self._wrap_stream_callback_for_input,
                    logger=self.logger,
                    record_sink=record_sink,
                )

                # Update analysis state with extracted text
                if not self.current_analysis_state:
                    self.logger.error("No analysis state available for file processing")
                    return False

                self.current_analysis_state.original_abstract = extracted_text

                # WP-D1 P2: input record's own classifications become priors - Claude Generated
                record = record_sink.get("record")
                if record is not None and getattr(record, "classifications", None):
                    self.current_analysis_state.input_record_classifications = record.classifications

                # Store extraction info
                if not hasattr(self.current_analysis_state, 'extraction_info'):
                    self.current_analysis_state.extraction_info = {}

                self.current_analysis_state.extraction_info.update({
                    "source": file_path,
                    "method": extraction_method,
                    "source_info": source_info,
                    "input_type": input_type
                })
                
                step.output_data = {
                    "text": extracted_text,
                    "source_info": source_info,
                    "extraction_method": extraction_method,
                    "file_path": file_path,
                    "processed": True,
                    "timestamp": datetime.now().isoformat(),
                }
                
                if self.stream_callback:
                    self.stream_callback(f"✅ {source_info}", "input")
                
                self.logger.info(f"File input processed: {extraction_method} - {len(extracted_text)} characters")
                return True
                
            except Exception as e:
                error_msg = f"File input processing failed: {str(e)}"
                self.logger.error(error_msg)
                
                if self.stream_callback:
                    self.stream_callback(f"❌ {error_msg}", "input")
                
                step.error_message = error_msg
                return False
        
        else:
            # Text-based input (traditional path)
            # Just verify we have text available
            if (
                not self.current_analysis_state
                or not self.current_analysis_state.original_abstract
            ):
                self.logger.warning("No input text available in analysis state")
                return False

            step.output_data = {
                "text": self.current_analysis_state.original_abstract,
                "processed": True,
                "timestamp": datetime.now().isoformat(),
            }
            
            self.logger.info(
                f"Text input completed with {len(self.current_analysis_state.original_abstract)} characters"
            )
            return True

    def _execute_initialisation_step(self, step: PipelineStep) -> bool:
        """Execute initialisation step using shared pipeline executor - Claude Generated"""
        if (
            not self.current_analysis_state
            or not self.current_analysis_state.original_abstract
        ):
            raise ValueError("No input text available for keyword extraction")

        # Get configuration for initialisation step
        step_config = self.config.get_step_config("initialisation")
        task = step_config.task or "initialisation"
        temperature = step_config.temperature or 0.7
        top_p = step_config.top_p or 0.1
        repetition_penalty = step_config.repetition_penalty
        think = step_config.think

        # Debug: Log the extracted configuration
        self.logger.info(
            f"Initialisation step config: task='{task}', temp={temperature}, top_p={top_p}"
        )

        # Debug: Check for system prompt
        system_prompt = getattr(step_config, 'system_prompt', None)
        if system_prompt:
            self.logger.info(
                f"Initialisation step has system_prompt: {len(system_prompt)} chars"
            )
        else:
            self.logger.info("Initialisation step has no system_prompt")

        # Create stream callback for UI feedback
        def stream_callback(token, step_id):
            if hasattr(self, "stream_callback") and self.stream_callback:
                self.stream_callback(token, step_id)

        self.logger.info(
            f"Starting initialisation with model {step.model} from provider {step.provider}"
        )

        # Show provider/model in GUI - Claude Generated
        if self.stream_callback:
            # Resolve actual provider/model for display in Smart Mode
            display_provider = step.provider or "Smart Mode"
            display_model = step.model or "Auto-Selected"

            # Try to get resolved values if auto-selection is needed
            if not step.provider or not step.model:
                try:
                    # Auto-select provider/model when not explicitly configured
                    # Use SmartProviderSelector to get the actual selection
                    selection = self.pipeline_executor.smart_selector.select_provider(
                        task_type="text",
                        prefer_fast=True,
                        task_name="initialisation",
                        step_id="initialisation"
                    )
                    display_provider = selection.provider
                    display_model = selection.model
                except Exception as e:
                    self.logger.debug(f"Could not resolve Smart Mode provider/model for display: {e}")

            self.stream_callback(f"🤖 Using {display_provider}/{display_model} for initial extraction\n", "initialisation")

        # Execute using shared pipeline executor
        try:
            # Clear step-only abort before starting LLM call - Claude Generated
            self._abort_step_event.clear()

            # Only pass parameters that AlimaManager.analyze_abstract() expects
            allowed_params = [
                "use_chunking_abstract",
                "abstract_chunk_size",
                "use_chunking_keywords",
                "keyword_chunk_size",
                "prompt_template",
            ]

            # Create filtered config from step_config attributes
            filtered_config = {}
            for param in allowed_params:
                value = getattr(step_config, param, None)
                if value is not None:
                    filtered_config[param] = value

            # Handle system_prompt -> system parameter mapping
            if hasattr(step_config, 'system_prompt') and step_config.system_prompt:
                filtered_config["system"] = step_config.system_prompt
                self.logger.info(
                    f"Initialisation: Mapped system_prompt to system parameter"
                )

            # Add repetition callback to filtered_config - Claude Generated (2026-02-17)
            if self.repetition_detected_callback:
                filtered_config["on_repetition_detected"] = self.repetition_detected_callback

            keywords, gnd_classes, llm_analysis, llm_title = (
                self.pipeline_executor.execute_initial_keyword_extraction(
                    abstract_text=self.current_analysis_state.original_abstract,
                    model=step.model,
                    provider=step.provider,
                    task=task,
                    stream_callback=stream_callback,
                    temperature=temperature,
                    p_value=top_p,
                    step_id=step.step_id,  # Pass step_id for proper callback handling
                    repetition_penalty=repetition_penalty,
                    think=think,
                    **filtered_config,  # Pass remaining config parameters
                )
            )

            # Update analysis state
            self.current_analysis_state.initial_keywords = keywords
            self.current_analysis_state.initial_gnd_classes = gnd_classes
            self.current_analysis_state.initial_llm_call_details = llm_analysis

            # Build and set working title - Claude Generated
            # Prefer official dataclass fields, fall back to extraction_info dict
            state = self.current_analysis_state
            if state.input_type is not None:
                source_value = state.source_value or state.original_abstract[:50]
                input_type_for_id = state.input_type
            elif hasattr(state, 'extraction_info') and state.extraction_info:
                source_value = state.extraction_info.get('source', 'text')
                input_type_for_id = state.extraction_info.get('input_type', 'text')
            else:
                source_value = 'text'
                input_type_for_id = 'text'

            # Extract clean source identifier
            source_id = extract_source_identifier(input_type_for_id, source_value)

            # Build complete working title
            working_title = build_working_title(
                llm_title=llm_title,
                source_identifier=source_id,
                timestamp=state.timestamp
            )

            # Set in analysis state
            state.working_title = working_title

            if self.logger:
                self.logger.info(f"📝 Generated working title: '{working_title}' (type={input_type_for_id})")

            step.output_data = {"keywords": keywords, "gnd_classes": gnd_classes}
            return True

        except ValueError as e:
            raise ValueError(f"Initialisation step failed: {e}")

    def _execute_search_step(self, step: PipelineStep) -> bool:
        """Execute GND search step using shared pipeline executor - Claude Generated"""
        if (
            not self.current_analysis_state
            or not self.current_analysis_state.initial_keywords
        ):
            raise ValueError("No keywords available for search")

        # Create stream callback for UI feedback
        def stream_callback(token, step_id):
            if hasattr(self, "stream_callback") and self.stream_callback:
                self.stream_callback(token, step_id)

        self.logger.info(
            f"Starting search with keywords: {self.current_analysis_state.initial_keywords}"
        )

        # Execute using shared pipeline executor
        try:
            search_results = self.pipeline_executor.execute_gnd_search(
                keywords=self.current_analysis_state.initial_keywords,
                suggesters=self.config.search_suggesters,
                stream_callback=stream_callback,
            )

            # Update analysis state - Convert Dict to List[SearchResult] for data model consistency
            self.current_analysis_state.search_results = self._convert_search_results_to_objects(search_results)

            self.logger.info(
                f"Search completed. Found {len(search_results)} result sets"
            )

            # Format results for display using shared formatter
            gnd_treffer = PipelineResultFormatter.format_search_results_for_display(
                search_results
            )

            step.output_data = {"gnd_treffer": gnd_treffer}
            return True

        except Exception as e:
            raise ValueError(f"Search step failed: {e}")

    def _execute_keywords_step(self, step: PipelineStep) -> bool:
        """Execute keywords step using shared pipeline executor - Claude Generated"""
        if not self.current_analysis_state:
            raise ValueError("No analysis state available for keywords step")

        # Allow empty search_results for single-step execution (user may provide text-only analysis) - Claude Generated
        if not self.current_analysis_state.search_results:
            self.logger.warning("No search results available - proceeding with text-only analysis")
            self.current_analysis_state.search_results = []  # Empty list (List[SearchResult]) to match data model

        # Get configuration for keywords step
        step_config = self.config.get_step_config("keywords")
        task = step_config.task or "keywords"
        temperature = step_config.temperature or 0.7
        top_p = step_config.top_p or 0.1
        repetition_penalty = step_config.repetition_penalty
        think = step_config.think

        # Debug: Log the extracted configuration
        self.logger.info(
            f"Keywords step config: task='{task}', temp={temperature}, top_p={top_p}"
        )
        self.logger.info(f"Full step_config: {step_config}")

        # Debug: Check for system prompt
        system_prompt = getattr(step_config, 'system_prompt', None)
        if system_prompt:
            self.logger.info(
                f"Keywords step has system_prompt: {len(system_prompt)} chars"
            )
        else:
            self.logger.info("Keywords step has no system_prompt")

        # Create stream callback for UI feedback
        def stream_callback(token, step_id):
            if hasattr(self, "stream_callback") and self.stream_callback:
                self.stream_callback(token, step_id)

        self.logger.info(f"Starting keywords step with task '{task}'")

        # Show provider/model in GUI - Claude Generated
        if self.stream_callback:
            # Resolve actual provider/model for display in Smart Mode
            display_provider = step.provider or "Smart Mode"
            display_model = step.model or "Auto-Selected"

            # Try to get resolved values if auto-selection is needed
            if not step.provider or not step.model:
                try:
                    # Auto-select provider/model when not explicitly configured
                    # Use SmartProviderSelector to get the actual selection
                    selection = self.pipeline_executor.smart_selector.select_provider(
                        task_type="text",
                        prefer_fast=False,
                        task_name="keywords",
                        step_id="keywords"
                    )
                    display_provider = selection.provider
                    display_model = selection.model
                except Exception as e:
                    self.logger.debug(f"Could not resolve Smart Mode provider/model for display: {e}")

            self.stream_callback(f"🤖 Using {display_provider}/{display_model} for final analysis\n", "keywords")

        # Execute using shared pipeline executor
        try:
            # Clear step-only abort before starting LLM call - Claude Generated
            self._abort_step_event.clear()

            # Only pass parameters that AlimaManager.analyze_abstract() expects
            # Note: prompt_template removed - let PromptService load correct prompt based on task
            allowed_params = [
                "use_chunking_abstract",
                "abstract_chunk_size",
                "use_chunking_keywords",
                "keyword_chunk_size",
                "keyword_chunking_threshold",
                "chunking_task",
            ]
            # Create filtered config from step_config attributes
            filtered_config = {}
            for param in allowed_params:
                value = getattr(step_config, param, None)
                if value is not None:
                    filtered_config[param] = value

            # Fix: keyword_chunking_threshold / chunking_task live in custom_params, not as top-level attrs
            for p in ["keyword_chunking_threshold", "chunking_task"]:
                if p in step_config.custom_params:
                    filtered_config[p] = step_config.custom_params[p]

            # TaskPreference override for chunking_threshold (keywords task only)
            if hasattr(self, 'config') and hasattr(self.config, 'step_configs'):
                try:
                    from ..utils.config_manager import ConfigManager
                    cm = ConfigManager()
                    cfg = cm.load_config()
                    task_pref = cfg.unified_config.task_preferences.get("keywords")
                    if task_pref and task_pref.chunking_threshold is not None:
                        if task_pref.chunking_threshold == 0:
                            # 0 means "Auto" → pass None to trigger auto-detect
                            filtered_config["keyword_chunking_threshold"] = None
                        else:
                            filtered_config["keyword_chunking_threshold"] = task_pref.chunking_threshold
                        self.logger.info(f"TaskPreference chunking_threshold override: {task_pref.chunking_threshold} → filtered={filtered_config.get('keyword_chunking_threshold')}")
                except Exception as e:
                    self.logger.debug(f"Could not load TaskPreference chunking_threshold: {e}")

            # Debug: Log what's actually in the filtered config - Claude Generated
            self.logger.info(f"Keywords step filtered_config: {filtered_config}")
            if "keyword_chunking_threshold" in filtered_config:
                self.logger.info(f"GUI Chunking threshold: {filtered_config['keyword_chunking_threshold']}")
            if "chunking_task" in filtered_config:
                self.logger.info(f"GUI Chunking task: {filtered_config['chunking_task']}")
            else:
                self.logger.warning("GUI: chunking_task missing from filtered_config!")

            # Handle system_prompt -> system parameter mapping
            if hasattr(step_config, 'system_prompt') and step_config.system_prompt:
                filtered_config["system"] = step_config.system_prompt
                self.logger.info(f"Keywords: Mapped system_prompt to system parameter")

            # Convert List[SearchResult] back to Dict for executor compatibility
            search_results_dict = self._convert_search_results_to_dict(
                self.current_analysis_state.search_results
            ) if self.current_analysis_state.search_results else {}

            # Check if iterative refinement is enabled - Claude Generated
            enable_iteration = getattr(step_config, 'enable_iterative_refinement', False)
            max_iterations = getattr(step_config, 'max_refinement_iterations', 2)

            if enable_iteration:
                # Iterative refinement path - Claude Generated
                self.logger.info(f"🔄 Iterative refinement enabled (max {max_iterations} iterations)")
                if self.stream_callback:
                    self.stream_callback(
                        f"🔄 Iterative Refinement aktiviert (max. {max_iterations} Iterationen)\n",
                        "keywords"
                    )

                final_keywords, iteration_metadata, llm_analysis = (
                    self.pipeline_executor.execute_iterative_keyword_refinement(
                        original_abstract=self.current_analysis_state.original_abstract,
                        initial_search_results=search_results_dict,
                        model=step.model,
                        provider=step.provider,
                        max_iterations=max_iterations,
                        stream_callback=stream_callback,
                        task=task,
                        temperature=temperature,
                        p_value=top_p,
                        step_id=step.step_id,
                        repetition_penalty=repetition_penalty,
                        think=think,
                        **filtered_config,
                    )
                )

                # Store iteration metadata in analysis state - Claude Generated
                self.current_analysis_state.refinement_iterations = iteration_metadata["iteration_history"]
                self.current_analysis_state.convergence_achieved = iteration_metadata["convergence_achieved"]
                self.current_analysis_state.max_iterations_reached = (
                    iteration_metadata["convergence_achieved"] == False and
                    len(iteration_metadata["iteration_history"]) >= max_iterations
                )

                self.logger.info(
                    f"✅ Iterative refinement completed: "
                    f"{iteration_metadata['total_iterations']} iterations, "
                    f"convergence={'achieved' if iteration_metadata['convergence_achieved'] else 'not achieved'}"
                )
            else:
                # Standard single-pass execution
                # Add repetition callback to filtered_config - Claude Generated (2026-02-17)
                if self.repetition_detected_callback:
                    filtered_config["on_repetition_detected"] = self.repetition_detected_callback

                final_keywords, _, llm_analysis = (
                    self.pipeline_executor.execute_final_keyword_analysis(
                        original_abstract=self.current_analysis_state.original_abstract,
                        search_results=search_results_dict,
                        model=step.model,
                        provider=step.provider,
                        task=task,
                        stream_callback=stream_callback,
                        temperature=temperature,
                        p_value=top_p,
                        step_id=step.step_id,  # Pass step_id for proper callback handling
                        repetition_penalty=repetition_penalty,
                        think=think,
                        **filtered_config,  # Pass remaining config parameters
                    )
                )

            # Update analysis state with final results
            self.current_analysis_state.final_llm_analysis = llm_analysis

            # Store keywords, LLM analysis, and verification data for UI display - Claude Generated
            step.output_data = {
                "final_keywords": final_keywords,
                "llm_analysis": llm_analysis,  # LlmKeywordAnalysis object with response_full_text
                "verification": llm_analysis.verification if llm_analysis and llm_analysis.verification else None,
            }

            # Debug: Log extracted keywords - Claude Generated
            self.logger.info(f"🔍 Keywords step completed: {len(final_keywords)} keywords extracted")
            if final_keywords:
                for i, kw in enumerate(final_keywords[:5], 1):
                    self.logger.info(f"  {i}. {kw[:80]}")
                if len(final_keywords) > 5:
                    self.logger.info(f"  ... und {len(final_keywords)-5} weitere")
            else:
                self.logger.warning("⚠️ NO KEYWORDS EXTRACTED! Check LLM response format.")
                if llm_analysis and llm_analysis.response_full_text:
                    self.logger.warning(f"LLM Response preview: {llm_analysis.response_full_text[:200]}")

            return True

        except ValueError as e:
            raise ValueError(f"Keywords step failed: {e}")

    def _execute_dk_search_step(self, step: PipelineStep) -> bool:
        """Execute DK search step using catalog search - Claude Generated"""
        try:
            # Get the final keywords from previous step
            previous_step = self._get_previous_step("keywords")
            if not previous_step or not previous_step.output_data:
                self.logger.warning("No keywords available for DK search")
                step.output_data = {"dk_search_results": []}
                return True
            
            final_keywords = previous_step.output_data.get("final_keywords", [])

            # Debug: Log keywords received from previous step - Claude Generated
            self.logger.info(f"🔍 DK Search received {len(final_keywords)} keywords from keywords step")
            if final_keywords:
                for i, kw in enumerate(final_keywords[:5], 1):
                    self.logger.info(f"  {i}. {kw[:80]}")
                if len(final_keywords) > 5:
                    self.logger.info(f"  ... und {len(final_keywords)-5} weitere")
            else:
                self.logger.error("❌ DK Search: NO KEYWORDS received from keywords step!")
                self.logger.error(f"previous_step.output_data keys: {list(previous_step.output_data.keys())}")

            # Use the shared pipeline executor for DK search
            step_config = self.config.get_step_config("dk_search")
            
            # DK-step policy from the catalog instance. A *policy* setting, not a
            # source gate → read even when the catalog source is disabled (finc/SRU
            # may be the DK backend). The endpoints/token are not read here: the
            # extractor builds itself (resolve_dk_extractor, WP P4). - Claude Generated
            from .search.factory import primary_settings
            _strict = primary_settings(None, "catalog", enabled_only=False).get(
                "strict_gnd_validation_for_dk_search"
            )
            strict_gnd_validation = True if _strict is None else bool(_strict)

            rvk_anchor_keywords = self.pipeline_executor._derive_rvk_anchor_keywords(
                final_keywords,
                self.current_analysis_state.final_llm_analysis if self.current_analysis_state else None,
                original_abstract=self.current_analysis_state.original_abstract if self.current_analysis_state else "",
                initial_keywords=self.current_analysis_state.initial_keywords if self.current_analysis_state else None,
                search_results=self.current_analysis_state.search_results if self.current_analysis_state else None,
                stream_callback=self._stream_callback_adapter,
            )
            dk_search_result = self.pipeline_executor.execute_dk_search(
                keywords=final_keywords,
                rvk_anchor_keywords=rvk_anchor_keywords,
                stream_callback=self._stream_callback_adapter,
                max_results=getattr(step_config, 'max_results', DEFAULT_DK_MAX_RESULTS),
                force_update=getattr(self, 'force_update', False),  # Claude Generated
                strict_gnd_validation=strict_gnd_validation,  # EXPERT OPTION - Claude Generated
            )

            # Extract components from new deduplication-aware format - Claude Generated Step 5
            flattened_results = dk_search_result.get("classifications", [])  # Deduplicated for LLM
            dk_statistics = dk_search_result.get("statistics", {})  # Statistics for display
            dk_search_results = dk_search_result.get("keyword_results", [])  # Keyword-centric for GUI

            # TRIPLE FORMAT ARCHITECTURE - Claude Generated
            # Three complementary formats for different purposes:
            # 1. Keyword-centric: Shows "Keyword X → DK Y, DK Z" (for user understanding/GUI)
            # 2. Deduplicated (flattened): Shows merged DKs across keywords (for LLM analysis)
            # 3. Statistics: Frequency, keyword coverage, deduplication metrics (for diagnostics)
            # - DK-centric: Groups DKs with all their source keywords (required for LLM prompt building)
            # Trade-off: Minor redundancy ↔ Clear separation of concerns and better UI/LLM data

            # Store all three formats - Claude Generated Step 5
            step.output_data = {
                "dk_search_results": dk_search_results,  # Keyword-centric: what each keyword found (GUI transparency)
                "dk_search_results_flattened": flattened_results,  # Deduplicated DK-centric: merged view (LLM analysis)
                "dk_statistics": dk_statistics  # Statistics: frequency, deduplication metrics, keyword coverage
            }

            # Log deduplication effectiveness - Claude Generated Step 5
            if dk_statistics:
                dedup_stats = dk_statistics.get("deduplication_stats", {})
                if dedup_stats.get("duplicates_removed", 0) > 0:
                    self.logger.info(
                        f"✅ DK Deduplication Summary: {dedup_stats.get('original_count', 0)} → "
                        f"{dk_statistics.get('total_classifications', 0)} classifications | "
                        f"~{dedup_stats.get('estimated_token_savings', 0)} tokens saved"
                    )

            # Transfer DK search results to analysis state - Claude Generated (Use deduplicated format for LLM)
            if self.current_analysis_state:
                self.current_analysis_state.dk_search_results = dk_search_results  # Keyword-centric for GUI
                self.current_analysis_state.dk_search_results_flattened = flattened_results  # Deduplicated for LLM
                self.current_analysis_state.dk_statistics = dk_statistics  # Statistics for display

            return True
            
        except Exception as e:
            self.logger.error(f"DK search step failed: {e}")
            step.error_message = str(e)
            step.output_data = {"dk_search_results": []}
            return False

    def _execute_dk_classification_step(self, step: PipelineStep) -> bool:
        """Execute DK classification step using LLM analysis - Claude Generated"""
        try:
            # Get DK search results from previous step or current analysis state - Claude Generated
            dk_search_results = []
            previous_step = self._get_previous_step("dk_search")
            if previous_step and previous_step.output_data:
                dk_search_results = previous_step.output_data.get("dk_search_results_flattened",
                                                                   previous_step.output_data.get("dk_search_results", []))

            if not dk_search_results and self.current_analysis_state:
                dk_search_results = getattr(self.current_analysis_state, 'dk_search_results_flattened', [])
                if dk_search_results:
                    self.logger.info(f"Using {len(dk_search_results)} DK results from analysis state (no completed dk_search step found)")

            if not dk_search_results:
                self.logger.warning("No DK search results available for classification")
                step.output_data = {"dk_classifications": []}
                return True

            # Get original abstract text - Claude Generated
            original_abstract = ""
            input_step = self._get_previous_step("input")
            if input_step and input_step.output_data:
                original_abstract = input_step.output_data.get("text", "")

            if not original_abstract and self.current_analysis_state:
                original_abstract = self.current_analysis_state.original_abstract
            
            # Use the shared pipeline executor for DK classification
            step_config = self.config.get_step_config("dk_classification")
            rvk_anchor_keywords = self.pipeline_executor._derive_rvk_anchor_keywords(
                self.current_analysis_state.final_llm_analysis.extracted_gnd_keywords if self.current_analysis_state and self.current_analysis_state.final_llm_analysis else [],
                self.current_analysis_state.final_llm_analysis if self.current_analysis_state else None,
                original_abstract=original_abstract,
                initial_keywords=self.current_analysis_state.initial_keywords if self.current_analysis_state else None,
                search_results=self.current_analysis_state.search_results if self.current_analysis_state else None,
                stream_callback=self._stream_callback_adapter,
            )

            # Prepare kwargs for DK classification - Claude Generated (2026-02-17)
            dk_kwargs = {
                "dk_search_results": dk_search_results,
                "original_abstract": original_abstract,
                "model": step.model or step_config.model or "cogito:32b",
                "provider": step.provider or step_config.provider or "ollama",
                "stream_callback": self._stream_callback_adapter,
                "temperature": step_config.temperature or 0.7,
                "top_p": step_config.top_p or 0.1,
                "dk_frequency_threshold": getattr(step_config, 'dk_frequency_threshold', DEFAULT_DK_FREQUENCY_THRESHOLD),
                "rvk_anchor_keywords": rvk_anchor_keywords,
                # WP-D1 P2: input record's own classifications as priors - Claude Generated
                "record_priors": getattr(self.current_analysis_state, "input_record_classifications", None) or None,
                "repetition_penalty": step_config.repetition_penalty,
                "think": step_config.think,
            }

            # Add repetition callback if available
            if self.repetition_detected_callback:
                dk_kwargs["on_repetition_detected"] = self.repetition_detected_callback

            dk_classifications, llm_analysis = self.pipeline_executor.execute_dk_classification(**dk_kwargs)

            # Prepare search summary for display
            search_summary_lines = []
            for result in dk_search_results[:5]:  # Show first 5 for summary
                dk_code = result.get("dk", "")
                count = result.get("count", 0)
                classification_type = result.get("classification_type", "DK")
                search_summary_lines.append(f"{classification_type}: {dk_code} (Häufigkeit: {count})")

            step.output_data = {
                "dk_classifications": dk_classifications,
                "llm_analysis": llm_analysis,  # Store the LlmKeywordAnalysis object - Claude Generated
                "dk_search_summary": "\n".join(search_summary_lines),
                "dk_search_results_flattened": dk_search_results  # ← Preserve for GUI display - Claude Generated
            }

            # Transfer DK classifications to analysis state - Claude Generated
            if self.current_analysis_state:
                self.current_analysis_state.dk_classifications = dk_classifications
                self.current_analysis_state.dk_llm_analysis = llm_analysis  # Store in analysis state - Claude Generated
                # IMPORTANT: Preserve dk_search_results from dk_search step (don't overwrite)
                # This ensures the title list is available in review tab even after dk_classification

            return True
            
        except Exception as e:
            self.logger.error(f"DK classification step failed: {e}")
            step.error_message = str(e)
            step.output_data = {"dk_classifications": []}
            return False

    def _get_previous_step(self, step_id: str) -> Optional[PipelineStep]:
        """Get the step with the given step_id from completed steps - Claude Generated"""
        for step in self.pipeline_steps:
            if step.step_id == step_id and step.status == "completed":
                return step
        return None

    def _convert_search_results_to_objects(
        self, search_results: Dict[str, Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Convert dict search results to SearchResult objects - Claude Generated

        This ensures consistency with the KeywordAnalysisState data model which expects
        a List[SearchResult], not a raw Dict. This fixes CLI/GUI JSON compatibility.

        Args:
            search_results: Dict mapping search_term to results dict

        Returns:
            List of SearchResult objects
        """
        from ..core.data_models import SearchResult
        return [
            SearchResult(search_term=term, results=results)
            for term, results in search_results.items()
        ]

    def _convert_search_results_to_dict(
        self, search_results: List[SearchResult]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert SearchResult objects back to dict format - Claude Generated

        PipelineStepExecutor expects Dict format for processing, but KeywordAnalysisState
        stores List[SearchResult] for proper data modeling. This converts back when needed.

        Args:
            search_results: List of SearchResult objects

        Returns:
            Dict mapping search_term to results dict
        """
        return {
            result.search_term: result.results
            for result in search_results
        }
