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
)
from ..utils.smart_provider_selector import SmartProviderSelector, TaskType
from ..utils.config_models import (
    UnifiedProviderConfig,
    PipelineMode,
    TaskType as UnifiedTaskType,
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

    def __post_init__(self):
        """Initialize step configs with proper defaults - Claude Generated"""
        # Ensure all step_configs are PipelineStepConfig objects
        # In the new approach, we only accept PipelineStepConfig objects directly
        pass
    
    @classmethod
    def create_from_provider_preferences(cls, config_manager) -> 'PipelineConfig':
        """Create PipelineConfig automatically from SmartProvider preferences - Claude Generated"""
        try:
            # Initialize SmartProviderSelector for intelligent provider selection
            smart_selector = SmartProviderSelector(config_manager)
            
            # Get optimal providers for different task types with proper task_name and step_id
            text_selection = smart_selector.select_provider(
                task_type=TaskType.TEXT,
                prefer_fast=False,
                task_name="keywords",
                step_id="keywords"
            )
            text_fast_selection = smart_selector.select_provider(
                task_type=TaskType.TEXT,
                prefer_fast=True,
                task_name="initialisation",
                step_id="initialisation"
            )
            classification_selection = smart_selector.select_provider(
                task_type=TaskType.CLASSIFICATION,
                prefer_fast=False,
                task_name="dk_class",
                step_id="dk_classification"
            )
            
            # Create step configurations based on intelligent selections using PipelineStepConfig objects
            from ..utils.config_models import PipelineStepConfig, TaskType as UnifiedTaskType

            step_configs = {
                "initialisation": PipelineStepConfig(
                    step_id="initialisation",
                    task_type=UnifiedTaskType.INITIALISATION,
                    enabled=True,
                    provider=text_fast_selection.provider,  # Fast provider for initial extraction
                    model=text_fast_selection.model,
                    temperature=0.7,
                    top_p=0.1,
                    task="initialisation",
                ),
                "keywords": PipelineStepConfig(
                    step_id="keywords",
                    task_type=UnifiedTaskType.KEYWORDS,
                    enabled=True,
                    provider=text_selection.provider,  # Quality provider for final analysis
                    model=text_selection.model,
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
                    task_type=UnifiedTaskType.DK_SEARCH,
                    enabled=True,
                    custom_params={
                        "max_results": DEFAULT_DK_MAX_RESULTS,
                        "catalog_token": "",  # Will be updated from config
                        "catalog_search_url": None,
                        "catalog_details_url": None,
                    }
                ),
                "dk_classification": PipelineStepConfig(
                    step_id="dk_classification",
                    task_type=UnifiedTaskType.DK_CLASSIFICATION,
                    enabled=True,
                    provider=classification_selection.provider,  # Classification-optimized provider
                    model=classification_selection.model,
                    temperature=0.7,
                    top_p=0.1,
                    task="dk_class",
                    custom_params={
                        "dk_frequency_threshold": DEFAULT_DK_FREQUENCY_THRESHOLD,
                    }
                ),
            }
            
            # Log provider selections
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Pipeline Config from Provider Preferences:")
            logger.info(f"  Initial extraction: {text_fast_selection.provider}/{text_fast_selection.model} (fast)")
            logger.info(f"  Keywords analysis: {text_selection.provider}/{text_selection.model} (quality)")
            logger.info(f"  Classification: {classification_selection.provider}/{classification_selection.model}")
            
            return cls(
                auto_advance=True,
                stop_on_error=True,
                save_intermediate_results=True,
                step_configs=step_configs,
                search_suggesters=["lobid", "swb"]
            )
            
        except Exception as e:
            # Fallback to default configuration if SmartProvider fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to create PipelineConfig from provider preferences: {e}")
            logger.info("Falling back to default hardcoded configuration")
            
            return cls()  # Return default configuration
    
    
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
            task_type=UnifiedTaskType.GENERAL
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
            "dk_classification": "dk_class",
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
                task_type=UnifiedTaskType.GENERAL
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
                    **filtered_config,  # Pass remaining config parameters
                )
            )

            # Update analysis state with final results
            self.current_analysis_state.final_llm_analysis = llm_analysis

            # Store both keywords and full LLM analysis for UI display - Claude Generated
            step.output_data = {
                "final_keywords": final_keywords,
                "llm_analysis": llm_analysis  # LlmKeywordAnalysis object with response_full_text
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
                
            except Exception as e:
                self.logger.warning(f"Failed to load catalog config: {e}")
                catalog_token = getattr(step_config, 'catalog_token', '')
                catalog_search_url = getattr(step_config, 'catalog_search_url', '')
                catalog_details_url = getattr(step_config, 'catalog_details_url', '')
            
            dk_search_results = self.pipeline_executor.execute_dk_search(
                keywords=final_keywords,
                stream_callback=self._stream_callback_adapter,
                max_results=getattr(step_config, 'max_results', DEFAULT_DK_MAX_RESULTS),
                catalog_token=catalog_token,
                catalog_search_url=catalog_search_url,
                catalog_details_url=catalog_details_url,
            )

            step.output_data = {"dk_search_results": dk_search_results}

            # Transfer DK search results to analysis state - Claude Generated
            if self.current_analysis_state:
                self.current_analysis_state.dk_search_results = dk_search_results

            return True
            
        except Exception as e:
            self.logger.error(f"DK search step failed: {e}")
            step.error_message = str(e)
            step.output_data = {"dk_search_results": []}
            return False

    def _execute_dk_classification_step(self, step: PipelineStep) -> bool:
        """Execute DK classification step using LLM analysis - Claude Generated"""
        try:
            # Get DK search results from previous step
            previous_step = self._get_previous_step("dk_search")
            if not previous_step or not previous_step.output_data:
                self.logger.warning("No DK search results available for classification")
                step.output_data = {"dk_classifications": []}
                return True
                
            dk_search_results = previous_step.output_data.get("dk_search_results", [])
            
            # Get original abstract text
            input_step = self._get_previous_step("input")
            original_abstract = ""
            if input_step and input_step.output_data:
                original_abstract = input_step.output_data.get("text", "")
            
            # Use the shared pipeline executor for DK classification
            step_config = self.config.get_step_config("dk_classification")
            dk_classifications = self.pipeline_executor.execute_dk_classification(
                dk_search_results=dk_search_results,
                original_abstract=original_abstract,
                model=step.model or step_config.model or "cogito:32b",
                provider=step.provider or step_config.provider or "ollama",
                stream_callback=self._stream_callback_adapter,
                temperature=step_config.temperature or 0.7,
                top_p=step_config.top_p or 0.1,
                dk_frequency_threshold=getattr(step_config, 'dk_frequency_threshold', DEFAULT_DK_FREQUENCY_THRESHOLD),  # Claude Generated
            )
            
            # Prepare search summary for display
            search_summary_lines = []
            for result in dk_search_results[:5]:  # Show first 5 for summary
                dk_code = result.get("dk", "")
                count = result.get("count", 0)
                classification_type = result.get("classification_type", "DK")
                search_summary_lines.append(f"{classification_type}: {dk_code} (Häufigkeit: {count})")

            step.output_data = {
                "dk_classifications": dk_classifications,
                "dk_search_summary": "\n".join(search_summary_lines)
            }

            # Transfer DK classifications to analysis state - Claude Generated
            if self.current_analysis_state:
                self.current_analysis_state.dk_classifications = dk_classifications

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


    def _get_default_task_for_step(self, step_id: str) -> str:
        """Get default prompt task for a pipeline step - Claude Generated"""
        task_mapping = {
            "initialisation": "initialisation",
            "keywords": "keywords",
            "dk_classification": "dk_class",
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

            # Initialize analysis state with input data (for single step execution) - Claude Generated
            if input_data and isinstance(input_data, str):
                # Parse input_data for keywords step - Claude Generated
                abstract_text = input_data
                keywords_list = []
                mock_search_results = {}

                # Check if keywords are embedded in input (format: "abstract\n\nExisting Keywords: kw1, kw2")
                if step_id == "keywords" and "Existing Keywords:" in input_data:
                    parts = input_data.split("Existing Keywords:")
                    if len(parts) == 2:
                        abstract_text = parts[0].strip()
                        keywords_part = parts[1].strip()

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
                )
                self.logger.info(f"✅ Initialized analysis state with {len(abstract_text)} characters for single step execution")

            # Create a single step
            step = PipelineStep(
                step_id=step_id,
                name=self.step_definitions.get(step_id, {}).get("name", step_id),
                input_data=input_data
            )

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
