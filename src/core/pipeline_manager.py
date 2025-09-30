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
    
    # Hybrid Mode Configuration - NEW
    default_mode: PipelineMode = PipelineMode.SMART
    step_configs: Dict[str, PipelineStepConfig] = field(default_factory=dict)  # Unified step configs


    # Search config (no LLM needed)
    search_suggesters: List[str] = field(default_factory=lambda: ["lobid", "swb"])

    def __post_init__(self):
        """Convert dictionary step_configs to PipelineStepConfig objects - Claude Generated"""
        # Convert any dictionary step configs to PipelineStepConfig objects
        converted_configs = {}
        for step_id, step_config in self.step_configs.items():
            if isinstance(step_config, dict):
                # Convert dictionary to PipelineStepConfig object
                converted_configs[step_id] = self._dict_to_pipeline_step_config(step_id, step_config)
            else:
                # Already a PipelineStepConfig object
                converted_configs[step_id] = step_config

        # Update step_configs with converted objects
        self.step_configs = converted_configs

    def _dict_to_pipeline_step_config(self, step_id: str, config_dict: dict) -> 'PipelineStepConfig':
        """Convert dictionary to PipelineStepConfig object - Claude Generated"""
        from ..utils.config_models import PipelineStepConfig, PipelineMode, TaskType

        # Determine mode based on config parameters
        mode = PipelineMode.SMART  # Default
        if 'mode' in config_dict:
            if isinstance(config_dict['mode'], str):
                mode = PipelineMode(config_dict['mode'])
            else:
                mode = config_dict['mode']
        elif config_dict.get('provider') and config_dict.get('model'):
            # Has explicit provider/model = Advanced mode
            mode = PipelineMode.ADVANCED
            if any(key in config_dict for key in ['temperature', 'top_p', 'seed', 'max_tokens']):
                # Has expert parameters = Expert mode
                mode = PipelineMode.EXPERT

        # Map step_id to task_type
        task_type_mapping = {
            'input': TaskType.INPUT,
            'initialisation': TaskType.INITIALISATION,
            'search': TaskType.SEARCH,
            'keywords': TaskType.KEYWORDS,
            'classification': TaskType.CLASSIFICATION,
            'dk_search': TaskType.DK_SEARCH,
            'dk_classification': TaskType.DK_CLASSIFICATION
        }
        task_type = task_type_mapping.get(step_id, TaskType.GENERAL)

        # Create PipelineStepConfig object with all supported parameters
        step_config = PipelineStepConfig(
            step_id=step_id,
            mode=mode,
            task_type=task_type,
            enabled=config_dict.get('enabled', True),
            provider=config_dict.get('provider'),
            model=config_dict.get('model'),
            task=config_dict.get('task'),
            temperature=config_dict.get('temperature'),
            top_p=config_dict.get('top_p'),
            max_tokens=config_dict.get('max_tokens'),
            seed=config_dict.get('seed'),
            timeout=config_dict.get('timeout')
        )

        # Add any custom parameters (unsupported parameters go here)
        custom_params = {}
        supported_params = {
            'step_id', 'mode', 'task_type', 'enabled', 'provider', 'model', 'task',
            'temperature', 'top_p', 'max_tokens', 'seed', 'timeout'
        }
        for key, value in config_dict.items():
            if key not in supported_params:
                custom_params[key] = value

        if custom_params:
            step_config.custom_params.update(custom_params)

        return step_config
    
    @classmethod
    def create_from_provider_preferences(cls, config_manager) -> 'PipelineConfig':
        """Create PipelineConfig automatically from SmartProvider preferences - Claude Generated"""
        try:
            # Initialize SmartProviderSelector for intelligent provider selection
            smart_selector = SmartProviderSelector(config_manager)
            
            # Get optimal providers for different task types
            text_selection = smart_selector.select_provider(TaskType.TEXT, prefer_fast=False)
            text_fast_selection = smart_selector.select_provider(TaskType.TEXT, prefer_fast=True) 
            classification_selection = smart_selector.select_provider(TaskType.CLASSIFICATION, prefer_fast=False)
            
            # Create step configurations based on intelligent selections
            step_configs = {
                "initialisation": {
                    "step_id": "initialisation",
                    "enabled": True,
                    "provider": text_fast_selection.provider,  # Fast provider for initial extraction
                    "model": text_fast_selection.model,
                    "temperature": 0.7,
                    "top_p": 0.1,
                    "task": "initialisation",
                },
                "keywords": {
                    "step_id": "keywords",
                    "enabled": True,
                    "provider": text_selection.provider,  # Quality provider for final analysis
                    "model": text_selection.model,
                    "temperature": 0.7,
                    "top_p": 0.1,
                    "task": "keywords",
                    "keyword_chunking_threshold": 500,
                    "chunking_task": "keywords_chunked",
                },
                "dk_search": {
                    "step_id": "dk_search",
                    "enabled": True,
                    "max_results": 20,
                    "catalog_token": "",  # Will be updated from config
                    "catalog_search_url": None,
                    "catalog_details_url": None,
                },
                "dk_classification": {
                    "step_id": "dk_classification",
                    "enabled": True,
                    "provider": classification_selection.provider,  # Classification-optimized provider
                    "model": classification_selection.model,
                    "temperature": 0.7,
                    "top_p": 0.1,
                    "task": "dk_class",
                    "dk_frequency_threshold": 10,
                },
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
    
    def initialize_hybrid_mode_configs(self, config_manager=None):
        """Initialize hybrid mode step configurations - Claude Generated"""
        if self.step_configs:
            return  # Already initialized

        # Create clean smart mode defaults for each step (no legacy migration)
        self.step_configs = {
            "input": PipelineStepConfig(
                step_id="input",
                mode=PipelineMode.SMART,
                task_type=UnifiedTaskType.GENERAL
            ),
            "initialisation": PipelineStepConfig(
                step_id="initialisation",
                mode=self.default_mode,
                task_type=UnifiedTaskType.INITIALISATION
            ),
            "search": PipelineStepConfig(
                step_id="search",
                mode=PipelineMode.SMART,  # Search doesn't use LLM
                task_type=UnifiedTaskType.GENERAL
            ),
            "keywords": PipelineStepConfig(
                step_id="keywords",
                mode=self.default_mode,
                task_type=UnifiedTaskType.KEYWORDS
            ),
            "dk_classification": PipelineStepConfig(
                step_id="dk_classification",
                mode=self.default_mode,
                task_type=UnifiedTaskType.CLASSIFICATION
            )
        }
    
    def get_step_config(self, step_id: str) -> PipelineStepConfig:
        """Get step configuration with fallback to defaults - Claude Generated"""
        if not self.step_configs:
            self.initialize_hybrid_mode_configs()
            
        if step_id in self.step_configs:
            return self.step_configs[step_id]
        
        # Fallback: create default smart mode config
        return PipelineStepConfig(
            step_id=step_id,
            mode=PipelineMode.SMART,
            task_type=UnifiedTaskType.GENERAL
        )
    
    def set_step_mode(self, step_id: str, mode: PipelineMode):
        """Set the mode for a specific step - Claude Generated"""
        if not self.step_configs:
            self.initialize_hybrid_mode_configs()
            
        if step_id in self.step_configs:
            self.step_configs[step_id].mode = mode
    
    def set_step_manual_config(self, step_id: str, provider: str = None, model: str = None, 
                              task: str = None, **expert_params):
        """Configure step for Advanced/Expert mode - Claude Generated"""
        if not self.step_configs:
            self.initialize_hybrid_mode_configs()
            
        if step_id not in self.step_configs:
            return False
            
        step_config = self.step_configs[step_id]
        
        # Set manual parameters
        if provider:
            step_config.provider = provider
        if model:
            step_config.model = model  
        if task:
            step_config.task = task
            
        # Handle expert parameters
        if expert_params:
            step_config.mode = PipelineMode.EXPERT
            if "temperature" in expert_params:
                step_config.temperature = expert_params["temperature"]
            if "top_p" in expert_params:
                step_config.top_p = expert_params["top_p"]
            if "max_tokens" in expert_params:
                step_config.max_tokens = expert_params["max_tokens"]
            
            # Store other custom parameters
            other_params = {k: v for k, v in expert_params.items() 
                          if k not in ["temperature", "top_p", "max_tokens"]}
            step_config.custom_params.update(other_params)
        elif provider or model or task:
            # Advanced mode if only provider/model/task specified
            step_config.mode = PipelineMode.ADVANCED
        
        return True
    
    def get_effective_config(self, step_id: str, config_manager=None) -> Dict[str, Any]:
        """
        Get effective configuration for a step, resolving Smart mode via SmartProviderSelector
        Returns dict compatible with existing pipeline execution logic - Claude Generated
        """
        step_config = self.get_step_config(step_id)
        
        if step_config.mode == PipelineMode.SMART:
            # Use SmartProviderSelector for automatic selection
            if config_manager:
                try:
                    from ..utils.smart_provider_selector import SmartProviderSelector
                    smart_selector = SmartProviderSelector(config_manager)
                    
                    # Smart mode uses balanced approach (not fast preference)
                    prefer_fast = False
                    
                    # Get smart selection
                    selection = smart_selector.select_provider(
                        task_type=TaskType.TEXT,  # Convert unified task type if needed
                        prefer_fast=prefer_fast,
                        task_name=self._get_default_task_for_step(step_id),
                        step_id=step_id
                    )
                    
                    # Return config dict
                    return {
                        "step_id": step_id,
                        "enabled": step_config.enabled,
                        "provider": selection.provider,
                        "model": selection.model,
                        "task": self._get_default_task_for_step(step_id),
                        "temperature": 0.7,  # Smart defaults
                        "top_p": 0.1,
                        "_smart_selection": True
                    }
                except Exception as e:
                    # Fallback to manual defaults if smart selection fails
                    pass
        
        elif step_config.is_manual_override():
            # Use manual configuration
            manual_config = step_config.get_manual_config()
            return {
                "step_id": step_id,
                "enabled": step_config.enabled,
                **manual_config,
                "_manual_override": True
            }
        
        # Ultimate fallback - use smart mode with intelligent defaults
        return {
            "step_id": step_id,
            "enabled": True,
            "provider": "auto",  # Will be resolved by Smart Mode
            "model": "auto",     # Will be resolved by Smart Mode
            "task": self._get_default_task_for_step(step_id),
            "temperature": 0.7,
            "top_p": 0.1,
            "_smart_fallback": True
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

    def is_using_smart_mode(self) -> bool:
        """Check if pipeline is primarily using smart mode - Claude Generated"""
        if not self.step_configs:
            return False

        smart_steps = sum(1 for config in self.step_configs.values()
                         if config.mode == PipelineMode.SMART)
        total_steps = len(self.step_configs)

        return smart_steps > total_steps // 2  # Majority are smart mode


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

        # Initialize hybrid mode configurations - Claude Generated
        self.config.initialize_hybrid_mode_configs(config_manager)
        self.logger.info(f"Hybrid mode initialized with default mode: {self.config.default_mode.value}")

        # 🔍 DEBUG: Log migration details after initialization
        self._debug_step_configs()

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

    def _debug_step_configs(self):
        """Debug logging for step configurations - Claude Generated"""
        self.logger.info("🔍 DEBUG: Modern Step Configuration Analysis")

        # Log current hybrid mode configs
        if hasattr(self.config, 'step_configs') and self.config.step_configs:
            for step_id, step_config in self.config.step_configs.items():
                # Handle both dictionary and PipelineStepConfig objects - Claude Generated
                if isinstance(step_config, dict):
                    # Dictionary format from UI
                    mode = step_config.get('mode', 'unknown')
                    provider = step_config.get('provider', 'unknown')
                    model = step_config.get('model', 'unknown')
                    enabled = step_config.get('enabled', True)
                    self.logger.info(f"🔍 STEP_CONFIG[{step_id}]: mode={mode}, provider={provider}, model={model}, enabled={enabled}")
                else:
                    # PipelineStepConfig object
                    mode_value = step_config.mode.value if hasattr(step_config.mode, 'value') else str(step_config.mode)
                    self.logger.info(f"🔍 STEP_CONFIG[{step_id}]: mode={mode_value}, provider={step_config.provider}, model={step_config.model}, enabled={step_config.enabled}")
        else:
            self.logger.info("🔍 NO_STEP_CONFIGS: step_configs not initialized")

    def set_config(self, config: PipelineConfig):
        """Set pipeline configuration - Claude Generated"""
        self.config = config
        self.logger.info(f"Pipeline configuration updated: {config}")

        # Debug: Log modern step configurations in detail
        if hasattr(config, 'step_configs') and config.step_configs:
            for step_id, step_config in config.step_configs.items():
                self.logger.info(f"Step '{step_id}': mode={step_config.mode.value}, enabled={step_config.enabled}")
        else:
            self.logger.info("No modern step configurations found")

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
            # Fallback to Smart mode
            return PipelineStepConfig(
                step_id=step_id,
                mode=PipelineMode.SMART,
                task_type=UnifiedTaskType.GENERAL
            )

    def _should_use_smart_mode(self, step_id: str) -> bool:
        """Check if a step should use Smart Mode for provider/model selection - Claude Generated"""
        self.logger.info(f"🔍 SMART_MODE_CHECK: {step_id}")
        try:
            step_config = self.get_step_config(step_id)
            result = step_config.mode == PipelineMode.SMART
            self.logger.info(f"🔍 SMART_MODE_RESULT: {step_id} mode={step_config.mode.value}, is_smart={result}")
            return result
        except Exception as e:
            self.logger.warning(f"🔍 SMART_MODE_ERROR: {step_id}: {e}")
            # Default to Smart Mode if configuration is unclear
            return True

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
            # Provider/model resolution moved to execution time via _resolve_smart_mode_for_step - Claude Generated
            steps.append(
                PipelineStep(
                    step_id="initialisation",
                    name=self.step_definitions.get("initialisation", {}).get(
                        "name", "Initialisation"
                    ),
                    provider=None,  # Resolved at execution time
                    model=None,     # Resolved at execution time
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
            # Provider/model resolution moved to execution time via _resolve_smart_mode_for_step - Claude Generated
            steps.append(
                PipelineStep(
                    step_id="keywords",
                    name=self.step_definitions.get("keywords", {}).get(
                        "name", "Keywords"
                    ),
                    provider=None,  # Resolved at execution time
                    model=None,     # Resolved at execution time
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
            # Provider/model resolution moved to execution time via _resolve_smart_mode_for_step - Claude Generated
            steps.append(
                PipelineStep(
                    step_id="dk_classification",
                    name=self.step_definitions["dk_classification"]["name"],
                    provider=None,  # Resolved at execution time
                    model=None,     # Resolved at execution time
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

            # Try to get resolved values if using Smart Mode
            if not step.provider or not step.model:
                try:
                    # Check if we're using Smart Mode by looking at config
                    if self._should_use_smart_mode("initialisation"):
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

            # Try to get resolved values if using Smart Mode
            if not step.provider or not step.model:
                try:
                    # Check if we're using Smart Mode by looking at config
                    if self._should_use_smart_mode("keywords"):
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
                max_results=getattr(step_config, 'max_results', 20),
                catalog_token=catalog_token,
                catalog_search_url=catalog_search_url,
                catalog_details_url=catalog_details_url,
            )
            
            step.output_data = {"dk_search_results": dk_search_results}
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
                dk_frequency_threshold=getattr(step_config, 'dk_frequency_threshold', 10),  # Claude Generated
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

                # Resolve Smart Mode provider/model BEFORE sending UI callback - Claude Generated
                self._resolve_smart_mode_for_step(current_step)

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

    def _resolve_smart_mode_for_step(self, step: PipelineStep):
        """Resolve Smart Mode provider/model for step before UI callback - Claude Generated"""
        # Skip steps that don't use LLM or already have provider/model set
        if step.step_id in ["input", "search", "dk_search"]:
            return  # No LLM needed for these steps

        if step.provider and step.model:
            return  # Already has provider/model (manual mode)

        try:
            # Get step configuration
            step_config = self.config.get_step_config(step.step_id)
            is_smart = step_config.mode == PipelineMode.SMART
            self.logger.info(f"🔍 RESOLVE_PROVIDER: {step.step_id} mode={step_config.mode.value}")

            # ALWAYS start with Smart defaults, even for Advanced/Expert modes
            smart_provider = None
            smart_model = None

            if step.step_id in ["initialisation", "keywords", "dk_classification"]:
                # Determine task type and parameters for SmartProviderSelector
                task_type_mapping = {
                    "initialisation": (TaskType.TEXT, True),   # (task_type, prefer_fast)
                    "keywords": (TaskType.TEXT, False),
                    "dk_classification": (TaskType.CLASSIFICATION, False)
                }

                task_type, prefer_fast = task_type_mapping[step.step_id]

                # Get smart selection as default for ALL modes
                # Use proper task name mapping (e.g., "dk_classification" step_id -> "dk_class" task_name)
                task_name = self._get_default_task_for_step(step.step_id)
                selection = self.pipeline_executor.smart_selector.select_provider(
                    task_type=task_type,
                    prefer_fast=prefer_fast,
                    task_name=task_name,
                    step_id=step.step_id
                )
                smart_provider = selection.provider
                smart_model = selection.model
                self.logger.info(f"🤖 Smart defaults for {step.step_id}: {smart_provider}/{smart_model}")

            # Apply mode-specific logic
            if is_smart:
                # Pure Smart Mode - use intelligent selection
                step.provider = smart_provider or self.pipeline_executor._get_first_available_ollama_provider()
                step.model = smart_model or "cogito:32b"
                self.logger.info(f"🤖 Smart Mode for {step.step_id}: {step.provider}/{step.model}")
            else:
                # Advanced/Expert Mode - start with Smart defaults, allow manual overrides
                step.provider = step_config.provider or smart_provider or self.pipeline_executor._get_first_available_ollama_provider()
                step.model = step_config.model or smart_model or "cogito:32b"
                mode_icon = "⚙️" if step_config.mode == PipelineMode.ADVANCED else "🔧"
                self.logger.info(f"{mode_icon} {step_config.mode.value} Mode for {step.step_id}: {step.provider}/{step.model} (smart defaults applied)")

        except Exception as e:
            self.logger.warning(f"Could not resolve provider/model for step {step.step_id}: {e}")

            # Intelligent fallback: try user's manual config for this step first - Claude Generated
            try:
                step_config = self.config.get_step_config(step.step_id)
                user_provider = step_config.provider
                user_model = step_config.model

                if user_provider and user_model:
                    step.provider = user_provider
                    step.model = user_model
                    self.logger.info(f"Fallback to user manual config for {step.step_id}: {user_provider}/{user_model}")
                else:
                    # Ultimate fallback: use first available configured ollama provider
                    fallback_provider = self.pipeline_executor._get_first_available_ollama_provider()
                    step.provider = fallback_provider
                    step.model = "cogito:32b"
                    self.logger.warning(f"Ultimate fallback to configured provider for {step.step_id}: {fallback_provider}/cogito:32b")

            except Exception as fallback_error:
                # If even user config fails, use first available configured ollama provider
                fallback_provider = self.pipeline_executor._get_first_available_ollama_provider()
                step.provider = fallback_provider
                step.model = "cogito:32b"
                self.logger.error(f"All fallbacks failed for {step.step_id}, using configured provider {fallback_provider}: {fallback_error}")

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
        try:
            # Set the configuration
            self.set_config(config)

            # Create a single step
            step = PipelineStep(
                step_id=step_id,
                name=self.step_definitions.get(step_id, {}).get("name", step_id),
                input_data=input_data
            )

            # Execute the step
            if self.step_started_callback:
                self.step_started_callback(step)

            success = self._execute_step(step)

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
