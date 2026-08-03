"""Single-step / resume execution axis of ``PipelineManager``.

Split out of ``pipeline_manager.py`` (WP cleanup D, F-15). Verbatim mixin
extraction: the methods stay on :class:`PipelineManager` via MRO, so no call
site changes.

``from __future__ import annotations`` keeps the ``PipelineConfig``/``PipelineStep``
type annotations as strings (no import cycle). ``PipelineStep`` is additionally
*instantiated* at runtime inside ``execute_single_step``; since it is defined in
``pipeline_manager`` (which imports this module), a top-level import would cycle —
so it is imported method-locally there (the one non-verbatim line of the split).
"""

from __future__ import annotations

from .data_models import KeywordAnalysisState
from ..utils.pipeline_utils import PipelineResultFormatter


class SingleStepExecutorMixin:
    """Ad-hoc single-step execution + resume-from-state. Mixed into :class:`PipelineManager`."""

    def execute_single_step(self, step_id: str, config: PipelineConfig, input_data: Optional[Any] = None) -> PipelineStep:
        """
        Execute a single pipeline step with ad-hoc configuration - Claude Generated
        Optimized for GUI tab single operations
        """
        from .pipeline_manager import PipelineStep  # local import avoids import cycle - Claude Generated

        step = None  # Initialize to None to avoid scope error in exception handler - Claude Generated

        try:
            # Set the configuration
            self.set_config(config)

            # Force recreation of pipeline steps from fresh config - Claude Generated
            # Bug: pipeline_steps may persist from previous executions with stale provider/model.
            # execute_step() only recreates them if empty, so clear here to ensure
            # _create_pipeline_steps() reads the just-set config with correct provider/model.
            self.pipeline_steps = []

            # Initialize analysis state with input data (for single step execution) - Claude Generated
            if input_data and isinstance(input_data, str):
                # Parse input_data for keywords step - Claude Generated
                abstract_text = input_data
                keywords_list = []
                mock_search_results = {}
                parsed_dk_results = []  # Safe default for non-DK steps - Claude Generated
                self.logger.debug(f"Initialized parsed_dk_results=[] for step_id='{step_id}'")

                # Check if keywords or DK results are embedded in input (format: "abstract\n\nExisting Keywords: ...")
                if "Existing Keywords:" in input_data:
                    parts = input_data.split("Existing Keywords:")
                    if len(parts) == 2:
                        abstract_text = parts[0].strip()
                        keywords_part = parts[1].strip()
                        self.logger.info(f"Found 'Existing Keywords:' marker in input (length: {len(keywords_part)})")

                        if step_id == "keywords":
                            # Parse keywords with GND-ID format - Claude Generated
                            import re
                            gnd_pattern = r"(.*?)\s*\(GND-ID:\s*([^)]+)\)"

                            keywords_list = []
                            mock_results = {"user_provided": {}}

                            # Split by both comma and newline to support different formats - Claude Generated
                            keyword_items = re.split(r'[,\n]+', keywords_part)

                            for kw in keyword_items:
                                kw = kw.strip()
                                if not kw:
                                    continue

                                # Try to extract GND-ID from format "Keyword (GND-ID: 123456)"
                                match = re.match(gnd_pattern, kw)
                                if match:
                                    keyword_text = match.group(1).strip()
                                    gnd_id = match.group(2).strip()

                                    # Lookup in knowledge_manager for additional data - Claude Generated
                                    gnd_title = self.cache_manager.get_gnd_title_by_id(gnd_id)
                                    final_keyword = gnd_title if gnd_title else keyword_text

                                    keywords_list.append(final_keyword)
                                    mock_results["user_provided"][final_keyword] = {
                                        "count": 1,
                                        "gnd_ids": {gnd_id},  # Real GND-ID from parsed text!
                                        "classifications": {},
                                    }
                                    self.logger.debug(f"Parsed GND keyword: '{final_keyword}' (GND-ID: {gnd_id})")
                                else:
                                    # Plain keyword without GND-ID
                                    keywords_list.append(kw)
                                    mock_results["user_provided"][kw] = {
                                        "count": 1,
                                        "gnd_ids": set(),  # No GND-ID
                                        "classifications": {},
                                    }
                                    self.logger.debug(f"Parsed plain keyword: '{kw}'")

                            mock_search_results = mock_results
                            self.logger.info(f"✅ Parsed {len(keywords_list)} keywords (with GND lookup) for keywords step")

                        elif step_id == "dk_classification":
                            # Parse DK results from formatted text - Claude Generated
                            self.logger.info(f"Attempting to parse DK results from context area...")
                            parsed_dk_results = PipelineResultFormatter.parse_dk_results_from_text(keywords_part)

                            # Validate parser always returns list - Claude Generated
                            if not isinstance(parsed_dk_results, list):
                                self.logger.warning(f"⚠️ parse_dk_results_from_text returned {type(parsed_dk_results).__name__} instead of list, using empty list")
                                parsed_dk_results = []

                            if parsed_dk_results:
                                self.logger.info(f"✅ Successfully parsed {len(parsed_dk_results)} DK results for dk_classification step")
                            else:
                                self.logger.warning(f"⚠️ No DK results parsed from context area. Context preview: {keywords_part[:100]}...")

                # Convert mock_search_results Dict to List[SearchResult] for data model consistency
                search_result_objects = self._convert_search_results_to_objects(mock_search_results)

                self.current_analysis_state = KeywordAnalysisState(
                    original_abstract=abstract_text,
                    initial_keywords=keywords_list,
                    search_suggesters_used=config.search_suggesters,
                    initial_gnd_classes=[],
                    search_results=search_result_objects,
                    initial_llm_call_details=None,
                    final_llm_analysis=None,
                    dk_search_results_flattened=parsed_dk_results  # ← NEW: Populate from parsed results
                )

                # Special case: create simulated previous steps for dk_classification - Claude Generated
                if step_id == "dk_classification":
                    input_step = PipelineStep(
                        step_id="input",
                        name="Input",
                        status="completed",
                        output_data={"text": abstract_text}
                    )
                    dk_search_step = PipelineStep(
                        step_id="dk_search",
                        name="DK Search",
                        status="completed",
                        output_data={
                            "dk_search_results_flattened": parsed_dk_results,
                            "dk_search_results": []
                        }
                    )
                    # Initialize pipeline_steps with these simulated steps
                    self.pipeline_steps = [input_step, dk_search_step]
                    self.logger.debug(f"Simulated completed steps for single-step execution: {[s.step_id for s in self.pipeline_steps]}")

                self.logger.info(f"✅ Initialized analysis state with {len(abstract_text)} characters for single step execution")

            # Create the target step
            step = PipelineStep(
                step_id=step_id,
                name=self.step_definitions.get(step_id, {}).get("name", step_id),
                input_data=input_data
            )

            # Ensure the target step is in pipeline_steps for _get_step_by_id - Claude Generated
            if self.pipeline_steps:
                # If we have simulated steps, add this one to the list
                self.pipeline_steps.append(step)
            else:
                # Otherwise initialize list with just this step
                # (Note: execute_step will recreate full list if it finds only 1 step or list empty)
                self.pipeline_steps = [step]

            # Get provider/model from config
            step_config = config.get_step_config(step_id)
            step.provider = step_config.provider
            step.model = step_config.model

            # Execute the step
            if self.step_started_callback:
                self.step_started_callback(step)

            # Use the existing execute_step logic - Claude Generated
            success = self.execute_step(step_id)

            if success:
                step.status = "completed"
                if self.step_completed_callback:
                    self.step_completed_callback(step)
            else:
                step.status = "error"
                if self.step_error_callback:
                    self.step_error_callback(step, step.error_message or "Unknown error")

            return step

        except Exception as e:
            # Handle case where step wasn't created yet - Claude Generated
            if step is None:
                step = PipelineStep(
                    step_id=step_id,
                    name=step_id,
                    status="error",
                    error_message=str(e)
                )
            else:
                step.status = "error"
                step.error_message = str(e)

            self.logger.error(f"Single step execution failed: {e}")

            if self.step_error_callback:
                self.step_error_callback(step, str(e))

            return step

    def resume_pipeline_from_state(self, analysis_state: KeywordAnalysisState):
        """Resume pipeline from existing analysis state - Claude Generated"""
        self.current_analysis_state = analysis_state

        # Determine which steps are complete based on available data
        completed_steps = []

        if analysis_state.original_abstract:
            completed_steps.append("input")

        if analysis_state.initial_keywords and analysis_state.initial_llm_call_details:
            completed_steps.append("initialisation")

        if analysis_state.search_results:
            completed_steps.append("search")

        if analysis_state.final_llm_analysis:
            completed_steps.append("keywords")

        self.logger.info(f"Resuming pipeline with completed steps: {completed_steps}")

        # Create steps and mark completed ones
        self.pipeline_steps = self._create_pipeline_steps(
            "text"
        )  # Default to text input

        for step in self.pipeline_steps:
            if step.step_id in completed_steps:
                step.status = "completed"
                # Set output data based on analysis state
                if step.step_id == "initialisation":
                    step.output_data = {
                        "keywords": analysis_state.initial_keywords,
                        "gnd_classes": analysis_state.initial_gnd_classes,
                    }
                elif step.step_id == "search":
                    # Format search results for display
                    search_dict = {}
                    for search_result in analysis_state.search_results:
                        search_dict[search_result.search_term] = search_result.results
                    gnd_treffer = (
                        PipelineResultFormatter.format_search_results_for_display(
                            search_dict
                        )
                    )
                    step.output_data = {"gnd_treffer": gnd_treffer}
                elif step.step_id == "keywords":
                    step.output_data = {
                        "final_keywords": analysis_state.final_llm_analysis.extracted_gnd_keywords
                    }

        # Set current step index to first incomplete step
        self.current_step_index = len(completed_steps)

        return completed_steps
