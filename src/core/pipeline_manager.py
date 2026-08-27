"""
Pipeline Manager - Orchestrates the complete ALIMA analysis pipeline
Claude Generated - Extends AlimaManager functionality for UI pipeline workflow
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import logging
import re as _re
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
    export_analysis_state_to_file,
    execute_input_extraction,
    build_working_title,  # For title generation - Claude Generated
    extract_source_identifier,  # For title generation - Claude Generated
)
from ..utils.smart_provider_selector import SmartProviderSelector
from ..utils.config_models import (
    UnifiedProviderConfig,
    PipelineMode,
    ProviderScope,
    TaskType,
    PipelineStepConfig
)
from ._pipeline_classic_steps import ClassicStepExecutorMixin
from ._pipeline_agentic import AgenticPipelineMixin
from ._pipeline_single_step import SingleStepExecutorMixin
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

    # Agentic mode: LLM-driven agents with MCP tools instead of sequential steps - Claude Generated
    enable_agentic_mode: bool = False
    agentic_max_iterations: int = 20
    agentic_quality_threshold: float = 0.6

    # Workflow configuration: Custom agent workflows - Claude Generated
    workflow_name: str = "alima_classic"  # v4 workflow to use when enable_agentic_mode=True
    custom_workflow_path: Optional[str] = None  # Path to custom workflow YAML/JSON file

    # Single-step agentic execution - Claude Generated
    agentic_step_id: Optional[str] = None         # If set, run only this step
    agentic_input_context_path: Optional[str] = None  # JSON file to load as warm-start context

    # Missing-concept feedback loop - Claude Generated
    agentic_missing_concept_search: bool = True   # Re-search after selection if concepts missing
    agentic_missing_concept_iterations: int = 1   # Max feedback rounds (prevents infinite loop)
    agentic_verbose: bool = False                  # Log full prompts to stream + logger

    # Search config (no LLM needed)
    search_suggesters: List[str] = field(default_factory=lambda: ["lobid", "swb"])

    # Global provider/model override for all LLM steps - Claude Generated
    global_provider_override: Optional[str] = None
    global_model_override: Optional[str] = None
    # Global thinking override for all LLM steps (None = leave per-step/task value) - Claude Generated
    global_think_override: Optional[bool] = None
    # Global token budget for all agentic LLM steps (None = leave the workflow
    # YAML's per-step ``llm.max_tokens``). Set, it OUTRANKS the YAML: it exists
    # so an operator can lift the budget without editing two workflow files in
    # five places each. Reaches the steps as ``SharedContext.max_tokens_override``;
    # the classic path takes no budget parameter at all. - Claude Generated
    global_max_tokens_override: Optional[int] = None

    def __post_init__(self):
        """Initialize step configs with proper defaults - Claude Generated"""
        # Apply global override if set
        if (self.global_provider_override or self.global_model_override
                or self.global_think_override is not None):
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
            if self.global_think_override is not None:
                step_config.think = self.global_think_override

        provider = self.global_provider_override or "(unchanged)"
        model = self.global_model_override or "(unchanged)"
        think = "(unchanged)" if self.global_think_override is None else self.global_think_override
        logger.info(f"🔬 Global override applied: {provider}/{model} think={think} → {llm_steps}")

    def has_explicit_step_override(self, step_id: str) -> bool:
        """True if the step's provider/model differs from the pipeline baseline.

        Used by the UI to decide whether a step is using the pipeline default or
        an explicit per-step override.  A saved step with empty provider/model is
        treated as *not* overridden.  Claude Generated (default-model cleanup).
        """
        if not self.step_configs or step_id not in self.step_configs:
            return False
        step_config = self.step_configs[step_id]
        if isinstance(step_config, dict):
            return bool(step_config.get("provider") or step_config.get("model"))
        return bool(step_config.provider or step_config.model)

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

            # Resolve pipeline default through the centralized hierarchy.
            # Falls back through pipeline_default -> preferred -> first enabled provider.
            default_provider, default_model = unified_config.resolve_default_provider_model(
                scope=ProviderScope.PIPELINE
            )

            if not default_provider:
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
                        logger.debug(f"Step override for {step_id}: {override_provider}/{override_model}")
                    # Apply think setting if explicitly set (None = default, not inherited) - Claude Generated
                    think_val = first_pref.get("think")
                    if think_val is not None:
                        step_configs[step_id].think = think_val
                        logger.debug(f"Think override for {step_id}: think={think_val}")

            # Resolve default workflow from system config; keep fallback for tests.
            default_workflow = "alima_v51"
            try:
                alima_cfg = config_manager.load_config()
                default_workflow = getattr(alima_cfg.system_config, "default_workflow", "alima_v51") or "alima_v51"
            except Exception:
                pass

            return cls(
                auto_advance=True,
                stop_on_error=True,
                save_intermediate_results=True,
                step_configs=step_configs,
                search_suggesters=["lobid", "swb"],
                workflow_name=default_workflow,
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



class PipelineManager(ClassicStepExecutorMixin, AgenticPipelineMixin, SingleStepExecutorMixin):
    """Manages the complete ALIMA analysis pipeline - Claude Generated"""

    def __init__(
        self,
        alima_manager: AlimaManager,
        cache_manager: UnifiedKnowledgeManager,
        logger: logging.Logger = None,
        config_manager=None,
    ):
        self.alima_manager = alima_manager
        self.llm_service = alima_manager.llm_service
        self.cache_manager = cache_manager
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager

        # Initialize shared pipeline executor with intelligent provider selection - Claude Generated
        self.pipeline_executor = PipelineStepExecutor(
            alima_manager, cache_manager, logger, config_manager
        )

        # Current pipeline state
        self.current_analysis_state: Optional[KeywordAnalysisState] = None
        # WP10 P-δ.1: retain last successful agentic SharedContext for
        # chat tools / single-step warm-start. Reset at start_pipeline().
        self.last_shared_context = None
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
                self._apply_default_workflow_from_config()
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
        self.agentic_context_callback: Optional[Callable] = None  # Claude Generated

        # Interrupt handling with thread-safety - Claude Generated
        import threading
        self._interrupt_lock = threading.Lock()
        self._interrupt_check_func: Optional[Callable] = None
        self._is_interrupted = False
        self._abort_step_event = threading.Event()  # Step-only abort, does not stop pipeline - Claude Generated
        self.logger.debug("Pipeline manager initialized with thread-safe interrupt support")

    def _apply_default_workflow_from_config(self):
        """Override PipelineConfig.workflow_name from SystemConfig.default_workflow."""
        if not self.config_manager:
            return
        try:
            alima_cfg = self.config_manager.load_config()
            default_workflow = getattr(alima_cfg.system_config, "default_workflow", None)
            if default_workflow:
                self.config.workflow_name = default_workflow
        except Exception:
            pass

    def set_config(self, config: PipelineConfig):
        """Set pipeline configuration - Claude Generated"""
        self.config = config
        self.logger.debug(f"Pipeline configuration updated: {config}")

        # Migrate legacy "abstract" step to "initialisation" - Claude Generated
        if hasattr(config, 'step_configs') and config.step_configs and 'abstract' in config.step_configs:
            config.step_configs['initialisation'] = config.step_configs.pop('abstract')
            self.logger.info("✅ Migrated legacy 'abstract' step configuration to 'initialisation'")

        # Log step configurations at debug level
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
                self.logger.debug(f"Step '{step_id}': status={config_status}, enabled={enabled}")
        else:
            self.logger.debug("No modern step configurations found")

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
        agentic_context: Optional[Callable] = None,  # Claude Generated
    ):
        """Set callbacks for pipeline events - Claude Generated"""
        self.step_started_callback = step_started
        self.step_completed_callback = step_completed
        self.step_error_callback = step_error
        self.pipeline_completed_callback = pipeline_completed
        self.stream_callback = stream_callback
        self.repetition_detected_callback = repetition_detected  # Claude Generated (2026-02-17)
        self.agentic_context_callback = agentic_context  # Claude Generated

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
        # Forward COMBINED check to AlimaManager streaming loop:
        # stops on full worker interrupt OR step-only abort (does not stop pipeline) - Claude Generated
        abort_event = self._abort_step_event

        def _step_abort_once():
            # Single-shot: triggers once and immediately clears itself.
            # Only the ONE currently-streaming LLM call is aborted;
            # subsequent chunk calls run normally.  - Claude Generated
            if abort_event.is_set():
                abort_event.clear()
                return True
            return False

        combined_check = lambda: is_interrupted_func() or _step_abort_once()
        self.alima_manager.set_interrupt_callback(combined_check)
        self.logger.debug("Interrupt check function registered with PipelineManager (thread-safe)")

    def _emit_pipeline_step_bus(self, step, status: str) -> None:
        """Emit a ``state.pipeline_step`` event on the AlimaStateBus.

        Called by ``_execute_next_step`` for both "running" and
        "completed" transitions so the chat panel can render a
        collapsible tool-call block for each classic pipeline step.
        Mirrors what the chat-tool wrapper at
        ``src/ui/chat_tools/pipeline.py`` does, but at the manager
        level — that way the auto-pipeline button (which bypasses
        the wrapper) gets the same treatment.

        Best-effort: any bus failure is swallowed so the pipeline
        keeps running.
        """
        try:
            from src.core.state_bus import AlimaStateBus
            payload = {
                "tool": "execute_complete_pipeline",
                "step_id": getattr(step, "step_id", "") or "",
                "name": getattr(step, "name", "") or "",
                "status": status,
            }
            if status == "error":
                payload["error"] = getattr(step, "error_message", "") or ""
            AlimaStateBus().emit_event("state.pipeline_step", payload)
        except Exception:
            self.logger.warning(
                "_emit_pipeline_step_bus failed", exc_info=True
            )

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

    def abort_current_step(self) -> None:
        """Abort only the current LLM generation without stopping the pipeline - Claude Generated

        Sets a step-level abort event that the streaming loop in AlimaManager
        checks via the combined interrupt callback.  The event is cleared
        automatically before the next LLM call so subsequent steps run normally.
        _check_interruption() (used at step boundaries) does NOT check this event,
        so the pipeline continues after the current generation is aborted.
        """
        self._abort_step_event.set()
        self.logger.info("🛑 Step-only abort requested – pipeline will continue after current generation")

    def start_pipeline(self, input_text: str, input_type: str = "text", input_source: str = None, force_update: bool = False) -> str:
        """Start a new pipeline execution - Claude Generated"""
        pipeline_id = str(uuid.uuid4())
        self.logger.info(f"🔵 [DEBUG] start_pipeline: {pipeline_id[:8]} agentic={getattr(self.config, 'enable_agentic_mode', '?')}")

        # Store force_update flag for use during pipeline execution - Claude Generated
        self.force_update = force_update
        if force_update:
            self.logger.info("⚠️ Force update enabled: catalog cache will be ignored")

        # WP10 P-δ.1: invalidate retained SharedContext + broadcast start.
        self.last_shared_context = None
        self.logger.info("🔵 [DEBUG] start_pipeline: before AlimaStateBus")
        try:
            from src.core.state_bus import AlimaStateBus
            AlimaStateBus().emit_event(
                "state.pipeline_started", {"pipeline_id": pipeline_id}
            )
        except Exception:
            self.logger.warning("state.pipeline_started emit failed", exc_info=True)
        self.logger.info("🔵 [DEBUG] start_pipeline: after AlimaStateBus")

        # Agentic mode: use MetaAgent instead of sequential steps - Claude Generated
        if self.config.enable_agentic_mode:
            return self._start_agentic_pipeline(pipeline_id, input_text, input_type, input_source)

        # Initialize pipeline state
        self.current_analysis_state = KeywordAnalysisState(
            original_abstract=input_text,  # Always store the text regardless of input type
            initial_keywords=[],
            search_suggesters_used=self.config.search_suggesters,
            initial_gnd_classes=[],
            search_results=[],
            initial_llm_call_details=None,
            final_llm_analysis=None,
            # Reached only when agentic mode is off — the agentic branch returns
            # above and labels its own state in to_keyword_analysis_state().
            pipeline_mode="classic",
        )

        # Store source info in official dataclass fields - Claude Generated
        self.current_analysis_state.input_type = input_type
        self.current_analysis_state.source_value = input_source or None
        # Keep extraction_info for backward compatibility with title generation
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

                    # Phase (post-F): emit ``state.pipeline_step`` on the
                    # bus so PipelineChatPanel renders a collapsible
                    # tool-call block for this step. The chat-tool
                    # wrapper at ``src/ui/chat_tools/pipeline.py`` does
                    # the same thing for the chat-driven classic path —
                    # moving the emit here means the auto-pipeline button
                    # (which doesn't go through the wrapper) gets the
                    # same visual treatment as agentic mode and the
                    # chat-driven path.
                    self._emit_pipeline_step_bus(current_step, "running")

                    success = self.execute_step(current_step.step_id)
                    self.logger.info(
                        f"Step {current_step.step_id} completed with success: {success}"
                    )

                    if success:
                        current_step.status = "completed"
                        if self.step_completed_callback:
                            self.step_completed_callback(current_step)

                        # Phase (post-F): mirror the started-event emit
                        # for the terminal state. Status "completed" is
                        # recognised by PipelineChatPanel as success.
                        self._emit_pipeline_step_bus(current_step, "completed")

                        # NEW: Allow main thread time to process completion and display messages - Claude Generated
                        # This prevents output interleaving where next step's output appears before
                        # previous step's completion summary (especially critical for GUI event queue)
                        import time
                        completion_delay_steps = ["initialisation", "keywords", "dk_classification"]
                        if current_step.step_id in completion_delay_steps:
                            time.sleep(0.2)  # 200ms for main thread to process completion signals
                            self.logger.debug(f"✅ Delayed 200ms after {current_step.step_id} completion for UI processing")
                    else:
                        # A failed step must stop the pipeline: auto-advancing
                        # past it would feed garbage into every following step
                        # and present the run as a success - Claude Generated
                        self._emit_pipeline_step_bus(current_step, "error")
                        if current_step.status != "error":
                            # Graceful failure (returned False without raising):
                            # execute_step has NOT notified anyone yet
                            current_step.status = "error"
                            if self.step_error_callback:
                                self.step_error_callback(
                                    current_step,
                                    current_step.error_message or "Step failed without error message",
                                )
                        self.logger.error(
                            f"Pipeline stopped at failed step '{current_step.step_id}': "
                            f"{current_step.error_message or 'no error message'}"
                        )
                        return

                self.current_step_index += 1

                # Continue to next step if auto-advance is enabled
                if self.config.auto_advance:
                    self.logger.info("Auto-advancing to next step")
                    self._execute_next_step()
            else:
                # Pipeline completed
                self.logger.info("Pipeline completed - all steps finished")
                # P-ζ: bridge classical result into SharedContext so chat tools
                # can read it. Agentic path already sets last_shared_context at
                # _finalize_v4_pipeline.
                if self.current_analysis_state is not None:
                    try:
                        from src.core.agents.shared_context import SharedContext
                        self.last_shared_context = SharedContext.from_keyword_analysis_state(
                            self.current_analysis_state
                        )
                    except Exception:
                        self.logger.exception("from_keyword_analysis_state failed")
                        self.last_shared_context = None
                try:
                    from src.core.state_bus import AlimaStateBus
                    AlimaStateBus().emit_event("state.pipeline_completed", {})
                except Exception:
                    self.logger.warning(
                        "state.pipeline_completed emit failed", exc_info=True
                    )
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
            export_analysis_state_to_file(
                self.current_analysis_state,
                file_path,
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
