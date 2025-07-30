"""
Pipeline Manager - Orchestrates the complete ALIMA analysis pipeline
Claude Generated - Extends AlimaManager functionality for UI pipeline workflow
"""

from typing import Optional, Dict, Any, List, Callable
import logging
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .alima_manager import AlimaManager
from .data_models import (
    AbstractData,
    AnalysisResult,
    TaskState,
    KeywordAnalysisState,
    LlmKeywordAnalysis,
    SearchResult,
)
from .search_cli import SearchCLI
from .cache_manager import CacheManager
from .suggesters.meta_suggester import SuggesterType
from .processing_utils import (
    extract_keywords_from_response,
    extract_gnd_system_from_response,
)
from ..llm.llm_service import LlmService
from ..llm.prompt_service import PromptService
from ..utils.pipeline_utils import (
    PipelineStepExecutor,
    PipelineResultFormatter,
    PipelineJsonManager,
)


@dataclass
class PipelineStep:
    """Represents a single step in the analysis pipeline - Claude Generated"""

    step_id: str
    name: str
    status: str = "pending"  # pending, running, completed, error
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution - Claude Generated"""

    # Pipeline behavior
    auto_advance: bool = True
    stop_on_error: bool = True
    save_intermediate_results: bool = True

    # Step configurations - flexible dict structure
    step_configs: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "initialisation": {
                "step_id": "initialisation",
                "enabled": True,
                "provider": "ollama",
                "model": "cogito:14b",
                "temperature": 0.7,
                "top_p": 0.1,
                "task": "initialisation",
            },
            "keywords": {
                "step_id": "keywords",
                "enabled": True,
                "provider": "ollama",
                "model": "cogito:32b",
                "temperature": 0.7,
                "top_p": 0.1,
                "task": "keywords",
            },
            "classification": {
                "step_id": "classification",
                "enabled": False,
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": 0.7,
                "top_p": 0.1,
            },
        }
    )

    # Search config (no LLM needed)
    search_suggesters: List[str] = field(default_factory=lambda: ["lobid", "swb"])


class PipelineManager:
    """Manages the complete ALIMA analysis pipeline - Claude Generated"""

    def __init__(
        self,
        alima_manager: AlimaManager,
        cache_manager: CacheManager,
        logger: logging.Logger = None,
    ):
        self.alima_manager = alima_manager
        self.cache_manager = cache_manager
        self.logger = logger or logging.getLogger(__name__)

        # Initialize shared pipeline executor
        self.pipeline_executor = PipelineStepExecutor(
            alima_manager, cache_manager, logger
        )

        # Current pipeline state
        self.current_analysis_state: Optional[KeywordAnalysisState] = None
        self.pipeline_steps: List[PipelineStep] = []
        self.current_step_index: int = 0
        self.config: PipelineConfig = PipelineConfig()

        # Step definitions
        self.step_definitions = {
            "input": {"name": "Input Processing", "order": 1},
            "initialisation": {"name": "Keyword Extraction", "order": 2},
            "search": {"name": "GND Search", "order": 3},
            "keywords": {"name": "Result Verification", "order": 4},
            "classification": {"name": "Classification", "order": 5},
        }

        # Callbacks for UI updates
        self.step_started_callback: Optional[Callable] = None
        self.step_completed_callback: Optional[Callable] = None
        self.step_error_callback: Optional[Callable] = None
        self.pipeline_completed_callback: Optional[Callable] = None
        self.stream_callback: Optional[Callable] = (
            None  # Callback for LLM streaming tokens
        )

    def set_config(self, config: PipelineConfig):
        """Set pipeline configuration - Claude Generated"""
        self.config = config
        self.logger.info(f"Pipeline configuration updated: {config}")

    def set_callbacks(
        self,
        step_started: Optional[Callable] = None,
        step_completed: Optional[Callable] = None,
        step_error: Optional[Callable] = None,
        pipeline_completed: Optional[Callable] = None,
        stream_callback: Optional[Callable] = None,
    ):
        """Set callbacks for pipeline events - Claude Generated"""
        self.step_started_callback = step_started
        self.step_completed_callback = step_completed
        self.step_error_callback = step_error
        self.pipeline_completed_callback = pipeline_completed
        self.stream_callback = stream_callback

    def start_pipeline(self, input_text: str, input_type: str = "text") -> str:
        """Start a new pipeline execution - Claude Generated"""
        pipeline_id = str(uuid.uuid4())

        # Initialize pipeline state
        self.current_analysis_state = KeywordAnalysisState(
            original_abstract=input_text,  # Always store the text regardless of input type
            initial_keywords=[],
            search_suggesters_used=self.config.search_suggesters,
            initial_gnd_classes=[],
            search_results=[],
            initial_llm_call_details=None,
            final_llm_analysis=None,
        )

        # Create pipeline steps
        self.pipeline_steps = self._create_pipeline_steps(input_type)
        self.current_step_index = 0

        self.logger.info(
            f"Starting pipeline {pipeline_id} with {len(self.pipeline_steps)} steps"
        )

        # Start first step
        if self.config.auto_advance:
            self._execute_next_step()

        return pipeline_id

    def _create_pipeline_steps(self, input_type: str) -> List[PipelineStep]:
        """Create pipeline steps based on configuration - Claude Generated"""
        steps = []

        # Input step
        steps.append(
            PipelineStep(
                step_id="input",
                name=self.step_definitions["input"]["name"],
                input_data={"type": input_type},
            )
        )

        # Initialisation step (free keyword generation)
        initialisation_config = self.config.step_configs.get("initialisation", {})
        if initialisation_config.get("enabled", True):
            steps.append(
                PipelineStep(
                    step_id="initialisation",
                    name=self.step_definitions.get("initialisation", {}).get(
                        "name", "Initialisation"
                    ),
                    provider=initialisation_config.get("provider", "ollama"),
                    model=initialisation_config.get("model", "cogito:14b"),
                )
            )

        # Search step (always enabled, no LLM config needed)
        steps.append(
            PipelineStep(
                step_id="search",
                name=self.step_definitions["search"]["name"],
                input_data={"suggesters": self.config.search_suggesters},
            )
        )

        # Keywords step (Verbale Erschließung)
        keywords_config = self.config.step_configs.get("keywords", {})
        if keywords_config.get("enabled", True):
            steps.append(
                PipelineStep(
                    step_id="keywords",
                    name=self.step_definitions.get("keywords", {}).get(
                        "name", "Keywords"
                    ),
                    provider=keywords_config.get("provider", "ollama"),
                    model=keywords_config.get("model", "cogito:32b"),
                )
            )

        # Classification step (optional)
        classification_config = self.config.step_configs.get("classification", {})
        if classification_config.get("enabled", False):
            steps.append(
                PipelineStep(
                    step_id="classification",
                    name=self.step_definitions["classification"]["name"],
                    provider=classification_config.get("provider", "gemini"),
                    model=classification_config.get("model", "gemini-1.5-flash"),
                )
            )

        return steps

    def execute_step(self, step_id: str) -> bool:
        """Execute a specific pipeline step - Claude Generated"""
        step = self._get_step_by_id(step_id)
        if not step:
            self.logger.error(f"Step {step_id} not found")
            return False

        try:
            self.logger.info(f"Executing step: {step.name}")

            # Execute step based on type
            if step.step_id == "input":
                success = self._execute_input_step(step)
            elif step.step_id == "initialisation":
                success = self._execute_initialisation_step(step)
            elif step.step_id == "search":
                success = self._execute_search_step(step)
            elif step.step_id == "keywords":
                success = self._execute_keywords_step(step)
            elif step.step_id == "classification":
                success = self._execute_classification_step(step)
            else:
                raise ValueError(f"Unknown step type: {step.step_id}")

            return success

        except Exception as e:
            step.status = "error"
            step.error_message = str(e)
            self.logger.error(f"Error executing step {step.name}: {e}")

            if self.step_error_callback:
                self.step_error_callback(step, str(e))

            return False

    def _execute_input_step(self, step: PipelineStep) -> bool:
        """Execute input processing step - Claude Generated"""
        # Input step should have been completed by UI already
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
            f"Input step completed with {len(self.current_analysis_state.original_abstract)} characters"
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
        initialisation_config = self.config.step_configs.get("initialisation", {})
        task = initialisation_config.get("task", "initialisation")
        temperature = initialisation_config.get("temperature", 0.7)
        top_p = initialisation_config.get("top_p", 0.1)

        # Create stream callback for UI feedback
        def stream_callback(token, step_id):
            if hasattr(self, "stream_callback") and self.stream_callback:
                self.stream_callback(token, step_id)

        self.logger.info(
            f"Starting initialisation with model {step.model} from provider {step.provider}"
        )

        # Execute using shared pipeline executor
        try:
            # Only pass parameters that AlimaManager.analyze_abstract() expects
            allowed_params = [
                "use_chunking_abstract",
                "abstract_chunk_size",
                "use_chunking_keywords",
                "keyword_chunk_size",
                "prompt_template",
            ]
            filtered_config = {
                k: v for k, v in initialisation_config.items() if k in allowed_params
            }

            keywords, gnd_classes, llm_analysis = (
                self.pipeline_executor.execute_initial_keyword_extraction(
                    abstract_text=self.current_analysis_state.original_abstract,
                    model=step.model,
                    provider=step.provider,
                    task=task,
                    stream_callback=stream_callback,
                    temperature=temperature,
                    p_value=top_p,
                    step_id=step.step_id,  # Pass step_id for proper callback handling
                    **filtered_config,  # Pass remaining config parameters
                )
            )

            # Update analysis state
            self.current_analysis_state.initial_keywords = keywords
            self.current_analysis_state.initial_gnd_classes = gnd_classes
            self.current_analysis_state.initial_llm_call_details = llm_analysis

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

            # Update analysis state
            self.current_analysis_state.search_results = search_results

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

        if not self.current_analysis_state.search_results:
            raise ValueError("No search results available for keywords step")

        # Get configuration for keywords step
        keywords_config = self.config.step_configs.get("keywords", {})
        task = keywords_config.get("task", "keywords")
        temperature = keywords_config.get("temperature", 0.7)
        top_p = keywords_config.get("top_p", 0.1)

        # Create stream callback for UI feedback
        def stream_callback(token, step_id):
            if hasattr(self, "stream_callback") and self.stream_callback:
                self.stream_callback(token, step_id)

        self.logger.info(f"Starting keywords step with task '{task}'")

        # Execute using shared pipeline executor
        try:
            # Only pass parameters that AlimaManager.analyze_abstract() expects
            allowed_params = [
                "use_chunking_abstract",
                "abstract_chunk_size",
                "use_chunking_keywords",
                "keyword_chunk_size",
                "prompt_template",
            ]
            filtered_config = {
                k: v for k, v in keywords_config.items() if k in allowed_params
            }

            final_keywords, _, llm_analysis = (
                self.pipeline_executor.execute_final_keyword_analysis(
                    original_abstract=self.current_analysis_state.original_abstract,
                    search_results=self.current_analysis_state.search_results,
                    model=step.model,
                    provider=step.provider,
                    task=task,
                    stream_callback=stream_callback,
                    temperature=temperature,
                    p_value=top_p,
                    step_id=step.step_id,  # Pass step_id for proper callback handling
                    **filtered_config,  # Pass remaining config parameters
                )
            )

            # Update analysis state with final results
            self.current_analysis_state.final_llm_analysis = llm_analysis

            step.output_data = {"final_keywords": final_keywords}
            return True

        except ValueError as e:
            raise ValueError(f"Keywords step failed: {e}")

    def _execute_classification_step(self, step: PipelineStep) -> bool:
        """Execute classification step - Claude Generated"""
        # Placeholder for classification logic
        # This would integrate with UB-specific classification systems
        step.output_data = {"classifications": []}
        return True

    def _execute_next_step(self):
        """Execute the next step in the pipeline - Claude Generated"""
        self.logger.info(
            f"Executing next step: index {self.current_step_index} of {len(self.pipeline_steps)}"
        )

        if self.current_step_index < len(self.pipeline_steps):
            current_step = self.pipeline_steps[self.current_step_index]
            self.logger.info(
                f"Processing step: {current_step.step_id} (status: {current_step.status})"
            )

            if current_step.status == "pending":
                current_step.status = "running"
                if self.step_started_callback:
                    self.step_started_callback(current_step)

                success = self.execute_step(current_step.step_id)
                self.logger.info(
                    f"Step {current_step.step_id} completed with success: {success}"
                )

                if success:
                    current_step.status = "completed"
                    if self.step_completed_callback:
                        self.step_completed_callback(current_step)

            self.current_step_index += 1

            # Continue to next step if auto-advance is enabled
            if self.config.auto_advance:
                self.logger.info("Auto-advancing to next step")
                self._execute_next_step()
        else:
            # Pipeline completed
            self.logger.info("Pipeline completed - all steps finished")
            if self.pipeline_completed_callback:
                self.pipeline_completed_callback(self.current_analysis_state)

    def _get_step_by_id(self, step_id: str) -> Optional[PipelineStep]:
        """Get step by ID - Claude Generated"""
        for step in self.pipeline_steps:
            if step.step_id == step_id:
                return step
        return None

    def get_current_step(self) -> Optional[PipelineStep]:
        """Get currently executing step - Claude Generated"""
        if 0 <= self.current_step_index < len(self.pipeline_steps):
            return self.pipeline_steps[self.current_step_index]
        return None

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status - Claude Generated"""
        completed_steps = sum(
            1 for step in self.pipeline_steps if step.status == "completed"
        )
        failed_steps = sum(1 for step in self.pipeline_steps if step.status == "error")

        return {
            "total_steps": len(self.pipeline_steps),
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "current_step": self.current_step_index,
            "current_step_name": (
                self.get_current_step().name if self.get_current_step() else None
            ),
            "analysis_state": self.current_analysis_state,
        }

    def reset_pipeline(self):
        """Reset pipeline to initial state - Claude Generated"""
        self.current_analysis_state = None
        self.pipeline_steps = []
        self.current_step_index = 0
        self.logger.info("Pipeline reset")

    def save_analysis_state(self, file_path: str):
        """Save current analysis state to JSON file - Claude Generated"""
        if not self.current_analysis_state:
            raise ValueError("No analysis state available to save")

        try:
            PipelineJsonManager.save_analysis_state(
                self.current_analysis_state, file_path
            )
            self.logger.info(f"Analysis state saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving analysis state: {e}")
            raise

    def load_analysis_state(self, file_path: str) -> KeywordAnalysisState:
        """Load analysis state from JSON file - Claude Generated"""
        try:
            analysis_state = PipelineJsonManager.load_analysis_state(file_path)
            self.current_analysis_state = analysis_state
            self.logger.info(f"Analysis state loaded from {file_path}")
            return analysis_state
        except Exception as e:
            self.logger.error(f"Error loading analysis state: {e}")
            raise

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
