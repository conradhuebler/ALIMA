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
from .unified_knowledge_manager import UnifiedKnowledgeManager
from ..utils.suggesters.meta_suggester import SuggesterType
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
    execute_input_extraction,
    build_working_title,  # For title generation - Claude Generated
    extract_source_identifier,  # For title generation - Claude Generated
)
from ..utils.smart_provider_selector import SmartProviderSelector
from ..utils.config_models import (
    UnifiedProviderConfig,
    PipelineMode,
    TaskType,
    PipelineStepConfig
)
from ..utils.pipeline_defaults import (
    DEFAULT_DK_MAX_RESULTS,
    DEFAULT_DK_FREQUENCY_THRESHOLD,
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
    """Configuration for pipeline execution with Hybrid Mode support - Claude Generated"""

    # Pipeline behavior
    auto_advance: bool = True
    stop_on_error: bool = True
    save_intermediate_results: bool = True

    # Step configurations
    step_configs: Dict[str, PipelineStepConfig] = field(default_factory=dict)  # Unified step configs


    # Search config (no LLM needed)
    search_suggesters: List[str] = field(default_factory=lambda: ["lobid", "swb"])

    # Global provider/model override for all LLM steps - Claude Generated
    global_provider_override: Optional[str] = None
    global_model_override: Optional[str] = None

    def __post_init__(self):
        """Initialize step configs with proper defaults - Claude Generated"""
        # Apply global override if set
        if self.global_provider_override or self.global_model_override:
            self.apply_global_override()

    @staticmethod
    def parse_override_string(override: str) -> tuple:
        """Parse override string into (provider, model) - Claude Generated

        Supported formats:
            provider|model      — explicit pipe separator
            provider/model      — slash separator (natural for provider/model)
            provider             — provider only, no model

        Examples:
            "gemini|gemini-2.0-flash"          → ("gemini", "gemini-2.0-flash")
            "openai_compatible/glm-4.6:cloud"  → ("openai_compatible", "glm-4.6:cloud")
            "ollama|cogito:14b"                → ("ollama", "cogito:14b")
            "gemini"                           → ("gemini", None)

        Returns:
            Tuple of (provider, model) where model may be None
        """
        if not override or not override.strip():
            return (None, None)

        override = override.strip()

        # Try pipe separator first (highest priority, unambiguous)
        if "|" in override:
            parts = override.split("|", 1)
            return (parts[0].strip(), parts[1].strip() if parts[1].strip() else None)

        # Try slash separator
        if "/" in override:
            parts = override.split("/", 1)
            return (parts[0].strip(), parts[1].strip() if parts[1].strip() else None)

        # No separator — provider only
        return (override, None)

    def apply_global_override(self):
        """Apply global provider/model override to all LLM steps - Claude Generated

        Overrides provider and/or model for initialisation, keywords, and dk_classification.
        Non-LLM steps (input, search, dk_search) are not affected.
        """
        llm_steps = ["initialisation", "keywords", "dk_classification"]
        logger = logging.getLogger(__name__)

        for step_id in llm_steps:
            if step_id not in self.step_configs:
                continue
            step_config = self.step_configs[step_id]
            if self.global_provider_override:
                step_config.provider = self.global_provider_override
            if self.global_model_override:
                step_config.model = self.global_model_override

        provider = self.global_provider_override or "(unchanged)"
        model = self.global_model_override or "(unchanged)"
        logger.info(f"🔬 Global override applied: {provider}/{model} → {llm_steps}")
    
    @classmethod
    def create_from_provider_preferences(cls, config_manager) -> 'PipelineConfig':
        """Create PipelineConfig from pipeline_default settings - Claude Generated

        Simplified provider selection strategy:
        - Uses pipeline_default_provider/model from unified config as baseline
        - Falls back to first available provider if no default set
        - Step overrides can be applied via task_preferences
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            unified_config = config_manager.get_unified_config()

            # Get pipeline default provider/model
            default_provider = unified_config.pipeline_default_provider
            default_model = unified_config.pipeline_default_model

            # Fallback: use first available provider if no default configured
            if not default_provider:
                enabled_providers = unified_config.get_enabled_providers()
                if enabled_providers:
                    first_provider = enabled_providers[0]
                    default_provider = first_provider.name
                    default_model = first_provider.preferred_model or ""
                    logger.info(f"No pipeline default set, using first provider: {default_provider}")
                else:
                    logger.warning("No enabled providers found")
                    return cls()

            logger.info(f"Pipeline Default: {default_provider}/{default_model}")

            # Create step configurations with pipeline default
            step_configs = {
                "initialisation": PipelineStepConfig(
                    step_id="initialisation",
                    task_type=TaskType.INITIALISATION,
                    enabled=True,
                    provider=default_provider,
                    model=default_model,
                    temperature=0.7,
                    top_p=0.1,
                    task="initialisation",
                ),
                "keywords": PipelineStepConfig(
                    step_id="keywords",
                    task_type=TaskType.KEYWORDS,
                    enabled=True,
                    provider=default_provider,
                    model=default_model,
                    temperature=0.7,
                    top_p=0.1,
                    task="keywords",
                    custom_params={
                        "keyword_chunking_threshold": 500,
                        "chunking_task": "keywords_chunked",
                    }
                ),
                "dk_search": PipelineStepConfig(
                    step_id="dk_search",
                    task_type=TaskType.DK_SEARCH,
                    enabled=True,
                    custom_params={
                        "max_results": DEFAULT_DK_MAX_RESULTS,
                        "catalog_token": "",
                        "catalog_search_url": None,
                        "catalog_details_url": None,
                    }
                ),
                "dk_classification": PipelineStepConfig(
                    step_id="dk_classification",
                    task_type=TaskType.DK_CLASSIFICATION,
                    enabled=True,
                    provider=default_provider,
                    model=default_model,
                    temperature=0.7,
                    top_p=0.1,
                    task="dk_classification",
                    custom_params={
                        "dk_frequency_threshold": DEFAULT_DK_FREQUENCY_THRESHOLD,
                    }
                ),
            }

            # Apply step overrides from task_preferences if any - Claude Generated
            for task_name, task_pref in unified_config.task_preferences.items():
                step_id = task_name.lower()  # e.g., "KEYWORDS" -> "keywords"
                if step_id in step_configs and task_pref.model_priority:
                    first_pref = task_pref.model_priority[0]
                    override_provider = first_pref.get("provider_name")
                    override_model = first_pref.get("model_name")
                    if override_provider and override_provider != "auto":
                        step_configs[step_id].provider = override_provider
                        step_configs[step_id].model = override_model or default_model
                        logger.info(f"Step override for {step_id}: {override_provider}/{override_model}")

            return cls(
                auto_advance=True,
                stop_on_error=True,
                save_intermediate_results=True,
                step_configs=step_configs,
                search_suggesters=["lobid", "swb"]
            )

        except Exception as e:
            logger.warning(f"Failed to create PipelineConfig: {e}")
            return cls()
    
    
    def get_step_config(self, step_id: str) -> PipelineStepConfig:
        """Get step configuration with fallback to defaults - Claude Generated"""
        if self.step_configs and step_id in self.step_configs:
            config = self.step_configs[step_id]

            # Handle both dict and PipelineStepConfig objects - Claude Generated
            if isinstance(config, dict):
                # Convert dict to PipelineStepConfig on-the-fly
                return PipelineStepConfig(
                    step_id=step_id,
                    enabled=config.get("enabled", True),
                    provider=config.get("provider"),
                    model=config.get("model"),
                    task=config.get("task"),
                    temperature=config.get("temperature"),
                    top_p=config.get("top_p"),
                    max_tokens=config.get("max_tokens"),
                    seed=config.get("seed"),
                    custom_params=config.get("custom_params", {}),
                    task_type=config.get("task_type"),
                )
            else:
                # Already a PipelineStepConfig object
                return config

        # Fallback: create default config
        return PipelineStepConfig(
            step_id=step_id,
            task_type=TaskType.GENERAL
        )
    
    
    def get_effective_config(self, step_id: str, config_manager=None) -> Dict[str, Any]:
        """
        Get effective configuration for a step - Claude Generated
        Returns dict compatible with existing pipeline execution logic
        """
        step_config = self.get_step_config(step_id)

        # Return configuration directly from step config
        return {
            "step_id": step_id,
            "enabled": step_config.enabled,
            "provider": step_config.provider,
            "model": step_config.model,
            "task": step_config.task or self._get_default_task_for_step(step_id),
            "temperature": step_config.temperature,
            "top_p": step_config.top_p,
            "max_tokens": step_config.max_tokens,
            "seed": step_config.seed,
            **step_config.custom_params
        }

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



class PipelineManager:
    """Manages the complete ALIMA analysis pipeline - Claude Generated"""

    def __init__(
        self,
        alima_manager: AlimaManager,
        cache_manager: UnifiedKnowledgeManager,
        logger: logging.Logger = None,
        config_manager=None,
    ):
        self.alima_manager = alima_manager
        self.cache_manager = cache_manager
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager

        # Initialize shared pipeline executor with intelligent provider selection - Claude Generated
        self.pipeline_executor = PipelineStepExecutor(
            alima_manager, cache_manager, logger, config_manager
        )

        # Current pipeline state
        self.current_analysis_state: Optional[KeywordAnalysisState] = None
        self.pipeline_steps: List[PipelineStep] = []
        self.current_step_index: int = 0
        
        # Initialize configuration - use SmartProvider preferences if available
        if config_manager:
            try:
                self.config: PipelineConfig = PipelineConfig.create_from_provider_preferences(config_manager)
                self.logger.info("Pipeline configuration initialized from Provider Preferences")
            except Exception as e:
                self.logger.warning(f"Failed to initialize from Provider Preferences, using default: {e}")
                self.config: PipelineConfig = PipelineConfig()
        else:
            self.config: PipelineConfig = PipelineConfig()
            self.logger.info("Pipeline configuration initialized with default settings (no ConfigManager provided)")

        # Pipeline initialized with baseline + override architecture
        self.logger.info("Pipeline configuration initialized with baseline + override architecture")

        # Step definitions
        self.step_definitions = {
            "input": {"name": "Input Processing", "order": 1},
            "initialisation": {"name": "Keyword Extraction", "order": 2},
            "search": {"name": "GND Search", "order": 3},
            "keywords": {"name": "Result Verification", "order": 4},
            "dk_search": {"name": "DK Search", "order": 5},
            "dk_classification": {"name": "DK Classification", "order": 6},
        }

        # Callbacks for UI updates
        self.step_started_callback: Optional[Callable] = None
        self.step_completed_callback: Optional[Callable] = None
        self.step_error_callback: Optional[Callable] = None
        self.pipeline_completed_callback: Optional[Callable] = None
        self.stream_callback: Optional[Callable] = (
            None  # Callback for LLM streaming tokens
        )
        self.repetition_detected_callback: Optional[Callable] = None  # Claude Generated (2026-02-17)

        # Interrupt handling with thread-safety - Claude Generated
        import threading
        self._interrupt_lock = threading.Lock()
        self._interrupt_check_func: Optional[Callable] = None
        self._is_interrupted = False
        self.logger.info("Pipeline manager initialized with thread-safe interrupt support")


    def set_config(self, config: PipelineConfig):
        """Set pipeline configuration - Claude Generated"""
        self.config = config
        self.logger.info(f"Pipeline configuration updated: {config}")

        # Migrate legacy "abstract" step to "initialisation" - Claude Generated
        if hasattr(config, 'step_configs') and config.step_configs and 'abstract' in config.step_configs:
            config.step_configs['initialisation'] = config.step_configs.pop('abstract')
            self.logger.info("✅ Migrated legacy 'abstract' step configuration to 'initialisation'")

        # Debug: Log modern step configurations in detail
        if hasattr(config, 'step_configs') and config.step_configs:
            for step_id, step_config in config.step_configs.items():
                # Handle both dict and PipelineStepConfig objects - Claude Generated
                if isinstance(step_config, dict):
                    provider = step_config.get("provider")
                    model = step_config.get("model")
                    enabled = step_config.get("enabled", True)
                else:
                    provider = step_config.provider
                    model = step_config.model
                    enabled = step_config.enabled

                config_status = "configured" if provider and model else "auto-selected"
                self.logger.info(f"Step '{step_id}': status={config_status}, enabled={enabled}")
        else:
            self.logger.info("No modern step configurations found")

    def reload_config(self):
        """Reload pipeline configuration from provider preferences - Claude Generated"""
        if not self.config_manager:
            self.logger.warning("Cannot reload config: no ConfigManager available")
            return

        try:
            self.logger.info("Reloading pipeline configuration from provider preferences...")
            new_config = PipelineConfig.create_from_provider_preferences(self.config_manager)
            self.set_config(new_config)
            self.logger.info("✅ Pipeline configuration reloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to reload pipeline configuration: {e}")

    def set_callbacks(
        self,
        step_started: Optional[Callable] = None,
        step_completed: Optional[Callable] = None,
        step_error: Optional[Callable] = None,
        pipeline_completed: Optional[Callable] = None,
        stream_callback: Optional[Callable] = None,
        repetition_detected: Optional[Callable] = None,  # Claude Generated (2026-02-17)
    ):
        """Set callbacks for pipeline events - Claude Generated"""
        self.step_started_callback = step_started
        self.step_completed_callback = step_completed
        self.step_error_callback = step_error
        self.pipeline_completed_callback = pipeline_completed
        self.stream_callback = stream_callback
        self.repetition_detected_callback = repetition_detected  # Claude Generated (2026-02-17)

    def set_interrupt_flag(self, lock, is_interrupted_func: Callable) -> None:
        """Set interrupt check function from worker - Claude Generated

        Called by PipelineWorker to provide interrupt checking capability.
        Allows the pipeline manager to check if the worker has been interrupted.

        Thread-safe implementation using internal lock.

        Args:
            lock: External threading lock (stored for reference, uses internal lock)
            is_interrupted_func: Callable that returns True if interrupted
        """
        with self._interrupt_lock:
            self._interrupt_check_func = is_interrupted_func
            self._external_lock = lock  # Store reference to external lock if needed
        self.logger.debug("Interrupt check function registered with PipelineManager (thread-safe)")

    def _check_interruption(self):
        """Check if pipeline should be interrupted - Claude Generated

        Thread-safe implementation using internal lock.

        Raises:
            InterruptedError: If interruption was requested
        """
        with self._interrupt_lock:
            check_func = self._interrupt_check_func

        if check_func and check_func():
            self.logger.info("Pipeline interruption detected")
            with self._interrupt_lock:
                self._is_interrupted = True
            raise InterruptedError("Pipeline interrupted by user")

    def start_pipeline(self, input_text: str, input_type: str = "text", input_source: str = None, force_update: bool = False) -> str:
        """Start a new pipeline execution - Claude Generated"""
        pipeline_id = str(uuid.uuid4())

        # Store force_update flag for use during pipeline execution - Claude Generated
        self.force_update = force_update
        if force_update:
            self.logger.info("⚠️ Force update enabled: catalog cache will be ignored")

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

        # Store source info for title generation - Claude Generated
        self.current_analysis_state.extraction_info = {
            "source": input_source or input_text[:50],  # Use source if provided, otherwise text preview
            "input_type": input_type,
            "method": "direct"
        }

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

    def start_pipeline_with_file(self, input_source: str, input_type: str = "auto") -> str:
        """Start pipeline with file input (PDF, Image) - Claude Generated"""
        pipeline_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"Starting file-based pipeline: {input_source} (type: {input_type})")
            
            # Extract text from file using pipeline utils
            if self.stream_callback:
                self.stream_callback("🔄 Starte Texterkennung...", "input")
            
            extracted_text, source_info, extraction_method = execute_input_extraction(
                llm_service=self.llm_service,
                input_source=input_source,
                input_type=input_type,
                stream_callback=self._wrap_stream_callback_for_input,
                logger=self.logger
            )
            
            if self.stream_callback:
                self.stream_callback(f"✅ {source_info}", "input")
                
            self.logger.info(f"Text extraction completed: {extraction_method} - {len(extracted_text)} characters")
            
            # Initialize pipeline state with extracted text
            self.current_analysis_state = KeywordAnalysisState(
                original_abstract=extracted_text,
                initial_keywords=[],
                search_suggesters_used=self.config.search_suggesters,
                initial_gnd_classes=[],
                search_results=[],
                initial_llm_call_details=None,
                final_llm_analysis=None,
            )
            
            # Store extraction info for pipeline tracking
            self.current_analysis_state.extraction_info = {
                "source": input_source,
                "method": extraction_method,
                "source_info": source_info,
                "input_type": input_type
            }
            
            # Create pipeline steps
            self.pipeline_steps = self._create_pipeline_steps("file")
            self.current_step_index = 0
            
            # Mark input step as completed since we already processed it
            if self.pipeline_steps and self.pipeline_steps[0].step_id == "input":
                self.pipeline_steps[0].status = "completed"
                self.pipeline_steps[0].output_data = {
                    "text": extracted_text,
                    "source_info": source_info,
                    "extraction_method": extraction_method,
                    "processed": True,
                    "timestamp": datetime.now().isoformat(),
                }
                # Advance to next step
                self.current_step_index = 1
            
            self.logger.info(f"File pipeline {pipeline_id} initialized with {len(extracted_text)} characters")
            
            # Start next step if auto-advance is enabled
            if self.config.auto_advance:
                self._execute_next_step()
                
            return pipeline_id
            
        except Exception as e:
            error_msg = f"File pipeline initialization failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self.stream_callback:
                self.stream_callback(f"❌ {error_msg}", "input")
                
            # Initialize with error state
            self.current_analysis_state = KeywordAnalysisState(
                original_abstract="",
                initial_keywords=[],
                search_suggesters_used=[],
                initial_gnd_classes=[],
                search_results=[],
                error_info=error_msg
            )
            raise Exception(error_msg)

    def _wrap_stream_callback_for_input(self, message: str):
        """Wrap stream callback for input extraction - Claude Generated"""
        if self.stream_callback:
            self.stream_callback(message, "input")

    def get_step_config(self, step_id: str) -> PipelineStepConfig:
        """Get step configuration with smart fallback - Claude Generated"""
        try:
            return self.config.get_step_config(step_id)
        except Exception as e:
            self.logger.warning(f"Failed to get step config for {step_id}: {e}")
            # Fallback to default configuration
            return PipelineStepConfig(
                step_id=step_id,
                task_type=TaskType.GENERAL
            )


    # _get_smart_mode_provider_model method removed - replaced by _resolve_smart_mode_for_step - Claude Generated

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
        initialisation_config = self.config.get_step_config("initialisation")
        if initialisation_config.enabled:
            # Read provider/model directly from configuration - Claude Generated
            steps.append(
                PipelineStep(
                    step_id="initialisation",
                    name=self.step_definitions.get("initialisation", {}).get(
                        "name", "Initialisation"
                    ),
                    provider=initialisation_config.provider,
                    model=initialisation_config.model,
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
        keywords_step_config = self.config.get_step_config("keywords")
        if keywords_step_config.enabled:
            # Read provider/model directly from configuration - Claude Generated
            steps.append(
                PipelineStep(
                    step_id="keywords",
                    name=self.step_definitions.get("keywords", {}).get(
                        "name", "Keywords"
                    ),
                    provider=keywords_step_config.provider,
                    model=keywords_step_config.model,
                )
            )

        # DK Search step (optional)
        dk_search_config = self.config.get_step_config("dk_search")
        if dk_search_config.enabled:
            steps.append(
                PipelineStep(
                    step_id="dk_search",
                    name=self.step_definitions["dk_search"]["name"],
                )
            )
            
        # DK Classification step (optional)
        dk_classification_config = self.config.get_step_config("dk_classification")
        if dk_classification_config.enabled:
            # Read provider/model directly from configuration - Claude Generated
            steps.append(
                PipelineStep(
                    step_id="dk_classification",
                    name=self.step_definitions["dk_classification"]["name"],
                    provider=dk_classification_config.provider,
                    model=dk_classification_config.model,
                )
            )

        return steps

    def execute_step(self, step_id: str) -> bool:
        """Execute a specific pipeline step - Claude Generated"""
        # Migrate legacy step names - Claude Generated
        STEP_ALIASES = {"abstract": "initialisation"}
        if step_id in STEP_ALIASES:
            original_step_id = step_id
            step_id = STEP_ALIASES[step_id]
            self.logger.info(f"✅ Migrated legacy step name '{original_step_id}' → '{step_id}'")

        # Auto-create pipeline steps if not exist (for individual step execution) - Claude Generated
        if not self.pipeline_steps:
            self.logger.info("⚙️ Creating pipeline steps for individual step execution")
            self.pipeline_steps = self._create_pipeline_steps("text")
            self.logger.info(f"✅ Created {len(self.pipeline_steps)} pipeline steps")

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
            elif step.step_id == "dk_search":
                success = self._execute_dk_search_step(step)
            elif step.step_id == "dk_classification":
                success = self._execute_dk_classification_step(step)
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
        """Execute input processing step with file support - Claude Generated"""
        
        # Check if input data specifies a file path that needs processing
        input_data = step.input_data or {}
        input_type = input_data.get("type", "text")
        
        if input_type in ["file", "pdf", "image", "auto"] and "file_path" in input_data:
            # File-based input processing
            try:
                if self.stream_callback:
                    self.stream_callback("🔄 Verarbeite Datei-Input...", "input")
                
                file_path = input_data["file_path"]
                extracted_text, source_info, extraction_method = execute_input_extraction(
                    llm_service=self.llm_service,
                    input_source=file_path,
                    input_type=input_type,
                    stream_callback=self._wrap_stream_callback_for_input,
                    logger=self.logger
                )
                
                # Update analysis state with extracted text
                if not self.current_analysis_state:
                    self.logger.error("No analysis state available for file processing")
                    return False
                
                self.current_analysis_state.original_abstract = extracted_text
                
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
                    **filtered_config,  # Pass remaining config parameters
                )
            )

            # Update analysis state
            self.current_analysis_state.initial_keywords = keywords
            self.current_analysis_state.initial_gnd_classes = gnd_classes
            self.current_analysis_state.initial_llm_call_details = llm_analysis

            # Build and set working title - Claude Generated
            if hasattr(self.current_analysis_state, 'extraction_info') and self.current_analysis_state.extraction_info:
                source_value = self.current_analysis_state.extraction_info.get('source', 'text')
                input_type = self.current_analysis_state.extraction_info.get('input_type', 'text')

                # Extract clean source identifier
                source_id = extract_source_identifier(input_type, source_value)

                # Build complete working title
                working_title = build_working_title(
                    llm_title=llm_title,
                    source_identifier=source_id,
                    timestamp=self.current_analysis_state.timestamp
                )

                # Set in analysis state
                self.current_analysis_state.working_title = working_title

                if self.logger:
                    self.logger.info(f"📝 Generated working title: '{working_title}'")
            else:
                # Fallback if no extraction info
                if self.logger:
                    self.logger.warning("⚠️ No extraction_info available, using minimal title")
                self.current_analysis_state.working_title = build_working_title(
                    llm_title=llm_title,
                    source_identifier='text',
                    timestamp=self.current_analysis_state.timestamp
                )

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
            
            # Get catalog configuration from global config if not in step config - Claude Generated
            try:
                from ..utils.config_manager import ConfigManager
                config_manager = ConfigManager()
                catalog_config = config_manager.get_catalog_config()

                catalog_token = getattr(step_config, 'catalog_token', '') or getattr(catalog_config, "catalog_token", "")
                catalog_search_url = getattr(step_config, 'catalog_search_url', '') or getattr(catalog_config, "catalog_search_url", "")
                catalog_details_url = getattr(step_config, 'catalog_details_url', '') or getattr(catalog_config, "catalog_details_url", "")
                strict_gnd_validation = getattr(catalog_config, "strict_gnd_validation_for_dk_search", True)

            except Exception as e:
                self.logger.warning(f"Failed to load catalog config: {e}")
                catalog_token = getattr(step_config, 'catalog_token', '')
                catalog_search_url = getattr(step_config, 'catalog_search_url', '')
                catalog_details_url = getattr(step_config, 'catalog_details_url', '')
                strict_gnd_validation = True

            dk_search_result = self.pipeline_executor.execute_dk_search(
                keywords=final_keywords,
                stream_callback=self._stream_callback_adapter,
                max_results=getattr(step_config, 'max_results', DEFAULT_DK_MAX_RESULTS),
                catalog_token=catalog_token,
                catalog_search_url=catalog_search_url,
                catalog_details_url=catalog_details_url,
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
                "repetition_penalty": step_config.repetition_penalty,
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

    def _stream_callback_adapter(self, token: str, step_id: str):
        """Adapter for stream callbacks - Claude Generated"""
        if self.stream_callback:
            self.stream_callback(token, step_id)

    def _execute_next_step(self):
        """Execute the next step in the pipeline - Claude Generated"""
        try:
            # Check for interruption before processing next step
            self._check_interruption()

            self.logger.info(
                f"Executing next step: index {self.current_step_index} of {len(self.pipeline_steps)}"
            )

            if self.current_step_index < len(self.pipeline_steps):
                current_step = self.pipeline_steps[self.current_step_index]
                self.logger.info(
                    f"Processing step: {current_step.step_id} (status: {current_step.status})"
                )

                if current_step.status == "pending":
                    # Check for interruption before starting step
                    self._check_interruption()

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

                            # NEW: Allow main thread time to process completion and display messages - Claude Generated
                            # This prevents output interleaving where next step's output appears before
                            # previous step's completion summary (especially critical for GUI event queue)
                            import time
                            completion_delay_steps = ["initialisation", "keywords", "dk_classification"]
                            if current_step.step_id in completion_delay_steps:
                                time.sleep(0.2)  # 200ms for main thread to process completion signals
                                self.logger.debug(f"✅ Delayed 200ms after {current_step.step_id} completion for UI processing")

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

        except InterruptedError:
            self.logger.info("Pipeline execution interrupted by user")
            # Save current state for resume functionality - Claude Generated
            current_step = self.pipeline_steps[self.current_step_index] if self.current_step_index < len(self.pipeline_steps) else None
            if current_step:
                self.logger.info(f"Pipeline interrupted at step: {current_step.step_id}")
            # Re-raise to let worker handle it
            raise

    def _get_step_by_id(self, step_id: str) -> Optional[PipelineStep]:
        """Get step by ID - Claude Generated"""
        for step in self.pipeline_steps:
            if step.step_id == step_id:
                return step
        return None


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

    def get_current_step(self) -> Optional[PipelineStep]:
        """Get currently executing step - Claude Generated"""
        if 0 <= self.current_step_index < len(self.pipeline_steps):
            return self.pipeline_steps[self.current_step_index]
        return None

    @property
    def is_running(self) -> bool:
        """Check if pipeline is currently running (any step has 'running' status) - Claude Generated"""
        return any(step.status == "running" for step in self.pipeline_steps)

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

    def execute_single_step(self, step_id: str, config: PipelineConfig, input_data: Optional[Any] = None) -> PipelineStep:
        """
        Execute a single pipeline step with ad-hoc configuration - Claude Generated
        Optimized for GUI tab single operations
        """
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
                                        "gndid": {gnd_id},  # Real GND-ID from parsed text!
                                        "ddc": set(),
                                        "dk": set()
                                    }
                                    self.logger.debug(f"Parsed GND keyword: '{final_keyword}' (GND-ID: {gnd_id})")
                                else:
                                    # Plain keyword without GND-ID
                                    keywords_list.append(kw)
                                    mock_results["user_provided"][kw] = {
                                        "count": 1,
                                        "gndid": set(),  # No GND-ID
                                        "ddc": set(),
                                        "dk": set()
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
