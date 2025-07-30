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
        """Execute initialisation step (free keyword generation) - Claude Generated"""
        if (
            not self.current_analysis_state
            or not self.current_analysis_state.original_abstract
        ):
            raise ValueError("No input text available for keyword extraction")

        # Create abstract data
        abstract_data = AbstractData(
            abstract=self.current_analysis_state.original_abstract, keywords=""
        )

        # Get configuration for initialisation step
        initialisation_config = self.config.step_configs.get("initialisation", {})
        task = initialisation_config.get(
            "task", "initialisation"
        )  # Use "initialisation" task for free keyword generation
        temperature = initialisation_config.get("temperature", 0.7)
        top_p = initialisation_config.get("top_p", 0.1)

        # Chunking configuration
        use_chunking_abstract = initialisation_config.get(
            "use_chunking_abstract", False
        )
        abstract_chunk_size = initialisation_config.get("abstract_chunk_size", 0)
        use_chunking_keywords = initialisation_config.get(
            "use_chunking_keywords", False
        )
        keyword_chunk_size = initialisation_config.get("keyword_chunk_size", 0)

        self.logger.info(
            f"Starting initialisation with model {step.model} from provider {step.provider}"
        )
        self.logger.debug(
            f"Input text for initialisation: {self.current_analysis_state.original_abstract}"
        )
        self.logger.debug(
            f"Using task: {task} with temperature: {temperature}, top_p: {top_p}"
        )

        # Create stream callback for UI feedback
        def stream_callback(token):
            if hasattr(self, "stream_callback") and self.stream_callback:
                self.logger.debug(
                    f"Streaming token for step {step.step_id}: '{token[:50]}...'"
                )
                self.stream_callback(token, step.step_id)

        # Execute analysis via AlimaManager
        task_state = self.alima_manager.analyze_abstract(
            abstract_data=abstract_data,
            task=task,
            model=step.model,
            use_chunking_abstract=use_chunking_abstract,
            abstract_chunk_size=abstract_chunk_size,
            use_chunking_keywords=use_chunking_keywords,
            keyword_chunk_size=keyword_chunk_size,
            temperature=temperature,
            p_value=top_p,
            stream_callback=stream_callback,
        )

        if task_state.status == "failed":
            raise ValueError(
                f"Keyword extraction failed: {task_state.analysis_result.full_text}"
            )

        # Extract keywords from response
        self.logger.info(
            f"LLM Response for keyword extraction: {task_state.analysis_result.full_text}"
        )
        keywords = extract_keywords_from_response(task_state.analysis_result.full_text)
        gnd_classes = extract_gnd_system_from_response(
            task_state.analysis_result.full_text
        )

        self.logger.info(f"Extracted keywords: {keywords}")
        self.logger.info(f"Extracted GND classes: {gnd_classes}")

        # Update analysis state
        self.current_analysis_state.initial_keywords = keywords
        self.current_analysis_state.initial_gnd_classes = gnd_classes
        self.current_analysis_state.initial_llm_call_details = LlmKeywordAnalysis(
            task_name=task,
            model_used=step.model,
            provider_used=step.provider,
            prompt_template=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            filled_prompt=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            temperature=temperature,
            seed=0,
            response_full_text=task_state.analysis_result.full_text,
            extracted_gnd_keywords=keywords,
            extracted_gnd_classes=gnd_classes,
        )

        step.output_data = {"keywords": keywords, "gnd_classes": gnd_classes}
        return True

    def _execute_search_step(self, step: PipelineStep) -> bool:
        """Execute GND search step - Claude Generated"""
        if (
            not self.current_analysis_state
            or not self.current_analysis_state.initial_keywords
        ):
            raise ValueError("No keywords available for search")

        self.logger.info(
            f"Starting search with keywords: {self.current_analysis_state.initial_keywords}"
        )

        # Use SearchCLI for search (reuse existing logic)
        search_cli = SearchCLI(self.cache_manager)

        # Convert suggester names to types
        suggester_types = []
        for suggester_name in self.config.search_suggesters:
            try:
                suggester_types.append(SuggesterType[suggester_name.upper()])
            except KeyError:
                self.logger.warning(f"Unknown suggester: {suggester_name}")

        # Convert keywords string to list for search
        if isinstance(self.current_analysis_state.initial_keywords, str):
            keywords_list = [
                kw.strip()
                for kw in self.current_analysis_state.initial_keywords.split(",")
                if kw.strip()
            ]
        else:
            keywords_list = self.current_analysis_state.initial_keywords

        self.logger.info(f"Converted keywords for search: {keywords_list}")
        self.logger.debug(f"Using suggesters: {[st.value for st in suggester_types]}")

        # Send search info to stream widget
        if hasattr(self, "stream_callback") and self.stream_callback:
            self.stream_callback(
                f"Suche mit {len(keywords_list)} Keywords: {', '.join(keywords_list[:3])}{'...' if len(keywords_list) > 3 else ''}\n",
                step.step_id,
            )
            self.stream_callback(
                f"Verwende Suggester: {', '.join([st.value for st in suggester_types])}\n",
                step.step_id,
            )

        # Execute search
        search_results = search_cli.search(
            search_terms=keywords_list, suggester_types=suggester_types
        )

        # Update analysis state
        self.current_analysis_state.search_results = search_results

        self.logger.info(f"Search completed. Found {len(search_results)} result sets")
        self.logger.debug(f"Search results keys: {list(search_results.keys())}")

        # Send search completion info to stream widget
        if hasattr(self, "stream_callback") and self.stream_callback:
            total_hits = sum(len(results) for results in search_results.values())
            self.stream_callback(
                f"Suche abgeschlossen: {total_hits} Treffer in {len(search_results)} Kategorien\n",
                step.step_id,
            )

        # Format search results with GND IDs for display
        gnd_treffer = []
        for results in search_results.values():
            for keyword, data in results.items():
                gnd_ids = data.get("gndid", set())
                for gnd_id in gnd_ids:
                    gnd_treffer.append(f"{keyword} (GND: {gnd_id})")

        step.output_data = {"gnd_treffer": gnd_treffer}  # Show all hits
        return True

    def _execute_keywords_step(self, step: PipelineStep) -> bool:
        """Execute keywords step (Verbale Erschließung) - Claude Generated"""
        if not self.current_analysis_state:
            raise ValueError("No analysis state available for keywords step")

        # Prepare the original abstract text for {abstract} placeholder
        original_abstract = self.current_analysis_state.original_abstract or ""

        # Prepare GND search results for {keywords} placeholder
        gnd_keywords_text = ""
        if self.current_analysis_state.search_results:
            # Format GND search results for the prompt
            gnd_keywords = []
            for results in self.current_analysis_state.search_results.values():
                for keyword, data in results.items():
                    gnd_ids = data.get("gndid", set())
                    for gnd_id in gnd_ids:
                        gnd_keywords.append(f"{keyword} (GND-ID: {gnd_id})")
            gnd_keywords_text = "\n".join(gnd_keywords)

        # Create abstract data with correct placeholder mapping
        # The prompt expects {abstract} and {keywords} placeholders
        # We put the original text in abstract and GND results in keywords
        abstract_data = AbstractData(
            abstract=original_abstract,  # This will fill {abstract} placeholder
            keywords=gnd_keywords_text,  # This will fill {keywords} placeholder
        )

        # Get configuration for keywords step
        keywords_config = self.config.step_configs.get("keywords", {})
        # Allow both "keywords" and "rephrase" tasks for this step
        task = keywords_config.get("task", "keywords")
        if task not in ["keywords", "rephrase"]:
            self.logger.warning(
                f"Unknown task '{task}' for keywords step, defaulting to 'keywords'"
            )
            task = "keywords"
        temperature = keywords_config.get("temperature", 0.7)
        top_p = keywords_config.get("top_p", 0.1)

        # Chunking configuration
        use_chunking_abstract = keywords_config.get("use_chunking_abstract", False)
        abstract_chunk_size = keywords_config.get("abstract_chunk_size", 0)
        use_chunking_keywords = keywords_config.get("use_chunking_keywords", False)
        keyword_chunk_size = keywords_config.get("keyword_chunk_size", 0)

        self.logger.info(f"Starting keywords step with task '{task}'")
        self.logger.debug(f"Abstract text length: {len(original_abstract)} chars")
        self.logger.debug(f"GND keywords text length: {len(gnd_keywords_text)} chars")
        self.logger.debug(f"Abstract preview: '{original_abstract[:200]}...'")
        self.logger.debug(f"GND keywords preview: '{gnd_keywords_text[:500]}...'")

        # Create stream callback for UI feedback
        def stream_callback(token):
            if hasattr(self, "stream_callback") and self.stream_callback:
                self.logger.debug(
                    f"Streaming token for step {step.step_id}: '{token[:50]}...'"
                )
                self.stream_callback(token, step.step_id)

        # Execute keywords step via AlimaManager
        task_state = self.alima_manager.analyze_abstract(
            abstract_data=abstract_data,
            task=task,
            model=step.model,
            use_chunking_abstract=use_chunking_abstract,
            abstract_chunk_size=abstract_chunk_size,
            use_chunking_keywords=use_chunking_keywords,
            keyword_chunk_size=keyword_chunk_size,
            temperature=temperature,
            p_value=top_p,
            stream_callback=stream_callback,
        )

        if task_state.status == "failed":
            raise ValueError(
                f"Keywords step failed: {task_state.analysis_result.full_text}"
            )

        # Update analysis state with final results
        final_keywords = extract_keywords_from_response(
            task_state.analysis_result.full_text
        )

        self.current_analysis_state.final_llm_analysis = LlmKeywordAnalysis(
            task_name=task,
            model_used=step.model,
            provider_used=step.provider,
            prompt_template=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            filled_prompt=(
                task_state.prompt_config.prompt if task_state.prompt_config else ""
            ),
            temperature=temperature,
            seed=0,
            response_full_text=task_state.analysis_result.full_text,
            extracted_gnd_keywords=final_keywords,
            extracted_gnd_classes=[],
        )

        step.output_data = {"final_keywords": final_keywords}
        return True

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
