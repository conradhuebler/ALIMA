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
from ..utils.unified_provider_config import (
    UnifiedProviderConfig,
    PipelineMode, 
    TaskType as UnifiedTaskType,
    PipelineStepConfig,
    get_unified_config_manager
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
    step_configs_v2: Dict[str, PipelineStepConfig] = field(default_factory=dict)  # New unified step configs

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
                "keyword_chunking_threshold": 500,  # Claude Generated - Default chunking threshold
                "chunking_task": "keywords_chunked",  # Claude Generated - Task for chunk processing
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
                "provider": "ollama",
                "model": "cogito:32b",
                "temperature": 0.7,
                "top_p": 0.1,
                "task": "dk_class",
                "dk_frequency_threshold": 10,  # Claude Generated - Default threshold
            },
        }
    )

    # Search config (no LLM needed)
    search_suggesters: List[str] = field(default_factory=lambda: ["lobid", "swb"])
    
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
        if self.step_configs_v2:
            return  # Already initialized
            
        # Create smart mode defaults for each step
        self.step_configs_v2 = {
            "input": PipelineStepConfig(
                step_id="input",
                mode=PipelineMode.SMART,
                task_type=UnifiedTaskType.GENERAL,
                quality_preference="fast"
            ),
            "initialisation": PipelineStepConfig(
                step_id="initialisation", 
                mode=self.default_mode,
                task_type=UnifiedTaskType.TEXT_ANALYSIS,
                quality_preference="fast"  # Initial extraction can be fast
            ),
            "search": PipelineStepConfig(
                step_id="search",
                mode=PipelineMode.SMART,  # Search doesn't use LLM
                task_type=UnifiedTaskType.GENERAL
            ),
            "keywords": PipelineStepConfig(
                step_id="keywords",
                mode=self.default_mode,
                task_type=UnifiedTaskType.TEXT_ANALYSIS, 
                quality_preference="balanced"  # Final analysis should be quality
            ),
            "dk_classification": PipelineStepConfig(
                step_id="dk_classification",
                mode=self.default_mode,
                task_type=UnifiedTaskType.CLASSIFICATION,
                quality_preference="balanced"
            )
        }
        
        # Migrate from legacy step_configs if needed
        self._migrate_legacy_configs()
        
    def _migrate_legacy_configs(self):
        """Migrate legacy step_configs to new hybrid format - Claude Generated"""
        for step_id, legacy_config in self.step_configs.items():
            if step_id in self.step_configs_v2:
                step_config = self.step_configs_v2[step_id]
                
                # If legacy config has manual provider/model, switch to Advanced mode
                if legacy_config.get("provider") and legacy_config.get("model"):
                    step_config.mode = PipelineMode.ADVANCED
                    step_config.provider = legacy_config.get("provider")
                    step_config.model = legacy_config.get("model")
                    step_config.task = legacy_config.get("task")
                    
                    # Expert mode if has custom parameters
                    if any(key in legacy_config for key in ["temperature", "top_p", "max_tokens"]):
                        step_config.mode = PipelineMode.EXPERT
                        step_config.temperature = legacy_config.get("temperature")
                        step_config.top_p = legacy_config.get("top_p")
                        step_config.custom_params = {
                            k: v for k, v in legacy_config.items() 
                            if k not in ["step_id", "enabled", "provider", "model", "task", "temperature", "top_p"]
                        }
                
                # Copy enabled state
                step_config.enabled = legacy_config.get("enabled", True)
    
    def get_step_config(self, step_id: str) -> PipelineStepConfig:
        """Get step configuration with fallback to defaults - Claude Generated"""
        if not self.step_configs_v2:
            self.initialize_hybrid_mode_configs()
            
        if step_id in self.step_configs_v2:
            return self.step_configs_v2[step_id]
        
        # Fallback: create default smart mode config
        return PipelineStepConfig(
            step_id=step_id,
            mode=PipelineMode.SMART,
            task_type=UnifiedTaskType.GENERAL,
            quality_preference="balanced"
        )
    
    def set_step_mode(self, step_id: str, mode: PipelineMode):
        """Set the mode for a specific step - Claude Generated"""
        if not self.step_configs_v2:
            self.initialize_hybrid_mode_configs()
            
        if step_id in self.step_configs_v2:
            self.step_configs_v2[step_id].mode = mode
    
    def set_step_manual_config(self, step_id: str, provider: str = None, model: str = None, 
                              task: str = None, **expert_params):
        """Configure step for Advanced/Expert mode - Claude Generated"""
        if not self.step_configs_v2:
            self.initialize_hybrid_mode_configs()
            
        if step_id not in self.step_configs_v2:
            return False
            
        step_config = self.step_configs_v2[step_id]
        
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
                    
                    # Map quality preference to prefer_fast
                    prefer_fast = step_config.quality_preference == "fast"
                    
                    # Get smart selection
                    selection = smart_selector.select_provider(
                        task_type=TaskType.TEXT,  # Convert unified task type if needed
                        prefer_fast=prefer_fast
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
        
        # Fallback to legacy config or defaults
        if step_id in self.step_configs:
            return self.step_configs[step_id].copy()
        
        # Ultimate fallback
        return {
            "step_id": step_id,
            "enabled": True,
            "provider": "ollama",
            "model": "cogito:32b",
            "task": self._get_default_task_for_step(step_id),
            "temperature": 0.7,
            "top_p": 0.1
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
        if not self.step_configs_v2:
            return False
            
        smart_steps = sum(1 for config in self.step_configs_v2.values() 
                         if config.mode == PipelineMode.SMART)
        total_steps = len(self.step_configs_v2)
        
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

        # Debug: Log step configurations in detail
        for step_id, step_config in config.step_configs.items():
            task = step_config.get("task", "N/A")
            enabled = step_config.get("enabled", "N/A")
            self.logger.info(f"Step '{step_id}': task='{task}', enabled={enabled}")

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

        # DK Search step (optional)
        dk_search_config = self.config.step_configs.get("dk_search", {})
        if dk_search_config.get("enabled", True):
            steps.append(
                PipelineStep(
                    step_id="dk_search",
                    name=self.step_definitions["dk_search"]["name"],
                )
            )
            
        # DK Classification step (optional) 
        dk_classification_config = self.config.step_configs.get("dk_classification", {})
        if dk_classification_config.get("enabled", True):
            steps.append(
                PipelineStep(
                    step_id="dk_classification",
                    name=self.step_definitions["dk_classification"]["name"],
                    provider=dk_classification_config.get("provider", "ollama"),
                    model=dk_classification_config.get("model", "cogito:32b"),
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
        initialisation_config = self.config.step_configs.get("initialisation", {})
        task = initialisation_config.get("task", "initialisation")
        temperature = initialisation_config.get("temperature", 0.7)
        top_p = initialisation_config.get("top_p", 0.1)

        # Debug: Log the extracted configuration
        self.logger.info(
            f"Initialisation step config: task='{task}', temp={temperature}, top_p={top_p}"
        )

        # Debug: Check for system prompt
        system_prompt = initialisation_config.get("system_prompt")
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

            # Handle system_prompt -> system parameter mapping
            if "system_prompt" in initialisation_config:
                filtered_config["system"] = initialisation_config["system_prompt"]
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
        keywords_config = self.config.step_configs.get("keywords", {})
        task = keywords_config.get("task", "keywords")
        temperature = keywords_config.get("temperature", 0.7)
        top_p = keywords_config.get("top_p", 0.1)

        # Debug: Log the extracted configuration
        self.logger.info(
            f"Keywords step config: task='{task}', temp={temperature}, top_p={top_p}"
        )
        self.logger.info(f"Full keywords_config: {keywords_config}")

        # Debug: Check for system prompt
        system_prompt = keywords_config.get("system_prompt")
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
            filtered_config = {
                k: v for k, v in keywords_config.items() if k in allowed_params
            }

            # Debug: Log what's actually in the filtered config - Claude Generated
            self.logger.info(f"Keywords step filtered_config: {filtered_config}")
            if "keyword_chunking_threshold" in filtered_config:
                self.logger.info(f"GUI Chunking threshold: {filtered_config['keyword_chunking_threshold']}")
            if "chunking_task" in filtered_config:
                self.logger.info(f"GUI Chunking task: {filtered_config['chunking_task']}")
            else:
                self.logger.warning("GUI: chunking_task missing from filtered_config!")

            # Handle system_prompt -> system parameter mapping
            if "system_prompt" in keywords_config:
                filtered_config["system"] = keywords_config["system_prompt"]
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
            dk_search_config = self.config.step_configs.get("dk_search", {})
            
            # Get catalog configuration from global config if not in step config - Claude Generated
            try:
                from ..utils.config_manager import ConfigManager
                config_manager = ConfigManager()
                catalog_config = config_manager.get_catalog_config()
                
                catalog_token = dk_search_config.get("catalog_token") or catalog_config.get("catalog_token", "")
                catalog_search_url = dk_search_config.get("catalog_search_url") or catalog_config.get("catalog_search_url", "")
                catalog_details_url = dk_search_config.get("catalog_details_url") or catalog_config.get("catalog_details_url", "")
                
            except Exception as e:
                self.logger.warning(f"Failed to load catalog config: {e}")
                catalog_token = dk_search_config.get("catalog_token", "")
                catalog_search_url = dk_search_config.get("catalog_search_url", "")
                catalog_details_url = dk_search_config.get("catalog_details_url", "")
            
            dk_search_results = self.pipeline_executor.execute_dk_search(
                keywords=final_keywords,
                stream_callback=self._stream_callback_adapter,
                max_results=dk_search_config.get("max_results", 20),
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
            dk_classification_config = self.config.step_configs.get("dk_classification", {})
            dk_classifications = self.pipeline_executor.execute_dk_classification(
                dk_search_results=dk_search_results,
                original_abstract=original_abstract,
                model=step.model or dk_classification_config.get("model", "cogito:32b"),
                provider=step.provider or dk_classification_config.get("provider", "ollama"),
                stream_callback=self._stream_callback_adapter,
                temperature=dk_classification_config.get("temperature", 0.7),
                top_p=dk_classification_config.get("top_p", 0.1),
                dk_frequency_threshold=dk_classification_config.get("dk_frequency_threshold", 10),  # Claude Generated
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
