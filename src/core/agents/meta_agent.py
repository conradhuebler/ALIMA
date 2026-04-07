"""MetaAgent - Intelligent Orchestrator for ALIMA Pipeline - Claude Generated

The MetaAgent coordinates specialized SubAgents to complete the ALIMA pipeline.
SubAgents are loaded from YAML workflow configuration, making the pipeline
fully customizable without code changes.

Key features:
- Shared tool result cache eliminates redundant searches
- SubAgents configured via YAML workflow
- Context passed between SubAgents
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field
import yaml

from src.core.agents.shared_context import SharedContext, ToolResultCache
from src.core.agents.sub_agents import (
    BaseSubAgent,
    SubAgentResult,
    KeywordExtractionAgent,
    SearchAgent,
    KeywordSelectionAgent,
    ClassificationAgent,
)
from src.core.data_models import KeywordAnalysisState
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# Mapping from agent type names to classes
AGENT_TYPE_MAP = {
    "extraction": KeywordExtractionAgent,
    "search": SearchAgent,
    "selection": KeywordSelectionAgent,
    "classification": ClassificationAgent,
}


@dataclass
class MetaAgentConfig:
    """Configuration for MetaAgent execution."""
    provider: str = ""
    model: str = ""
    temperature: float = 0.5
    max_tokens: int = 4096
    max_iterations: int = 20
    quality_threshold: float = 0.6
    enable_classification: bool = True
    skip_search_if_cached: bool = True
    workflow_name: str = "meta_agent_default"  # Name of workflow YAML to load


@dataclass
class PipelineStepResult:
    """Result from a pipeline step."""
    step_name: str
    agent_name: str
    success: bool
    duration_seconds: float
    data: Dict[str, Any]
    quality_score: float = 1.0
    error: Optional[str] = None


@dataclass
class SubAgentStepConfig:
    """Configuration for a SubAgent step."""
    id: str
    agent_class: Type[BaseSubAgent]
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    custom_system_prompt: Optional[str] = None
    custom_user_prompt: Optional[str] = None


class MetaAgent:
    """Intelligent orchestrator for the ALIMA pipeline.

    SubAgents are loaded from YAML workflow configuration.
    Coordinates SubAgents with shared context and tool caching.
    """

    # Default workflow search paths
    WORKFLOW_PATHS = [
        Path("workflows"),
        Path.home() / ".config" / "alima" / "workflows",
        Path(__file__).parent.parent.parent.parent / "workflows",
    ]

    def __init__(
        self,
        llm_service,
        config_manager=None,
        stream_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """Initialize MetaAgent.

        Args:
            llm_service: LLM service for generation
            config_manager: Optional config manager
            stream_callback: Optional callback for streaming output (text, step_id)
        """
        self.llm_service = llm_service
        self.config_manager = config_manager
        self.stream_callback = stream_callback
        self.logger = logging.getLogger(__name__)

        # Initialize tool registry
        self.tool_registry = ToolRegistry(config_manager=config_manager)
        self.tool_registry.register_all_tools()

        # Loaded workflow steps
        self.workflow_steps: List[SubAgentStepConfig] = []

    def load_workflow(self, workflow_name: str) -> List[SubAgentStepConfig]:
        """Load workflow configuration from YAML file.

        Args:
            workflow_name: Name of workflow (without .yaml extension)

        Returns:
            List of SubAgentStepConfig
        """
        # Find workflow file
        workflow_path = None
        for search_path in self.WORKFLOW_PATHS:
            candidate = search_path / f"{workflow_name}.yaml"
            if candidate.exists():
                workflow_path = candidate
                break

        if not workflow_path:
            self.logger.warning(f"Workflow '{workflow_name}' not found, using default pipeline")
            return self._get_default_steps()

        self.logger.info(f"Loading workflow from {workflow_path}")

        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            steps = []
            for step_config in config.get("pipeline", []):
                step_id = step_config.get("id", "")
                agent_type = step_config.get("id", "")  # Step ID maps to agent type

                # Map step ID to agent class
                agent_class = AGENT_TYPE_MAP.get(agent_type)
                if not agent_class:
                    self.logger.warning(f"Unknown agent type: {agent_type}, skipping")
                    continue

                steps.append(SubAgentStepConfig(
                    id=step_id,
                    agent_class=agent_class,
                    enabled=step_config.get("enabled", True),
                    depends_on=step_config.get("depends_on", []),
                    custom_system_prompt=step_config.get("system_prompt"),
                    custom_user_prompt=step_config.get("user_prompt_template"),
                ))

            self.workflow_steps = steps
            self.logger.info(f"Loaded {len(steps)} workflow steps")
            return steps

        except Exception as e:
            self.logger.error(f"Error loading workflow: {e}")
            return self._get_default_steps()

    def _get_default_steps(self) -> List[SubAgentStepConfig]:
        """Get default pipeline steps if workflow not found.

        Returns:
            List of default SubAgentStepConfig
        """
        return [
            SubAgentStepConfig(id="extraction", agent_class=KeywordExtractionAgent),
            SubAgentStepConfig(id="search", agent_class=SearchAgent),
            SubAgentStepConfig(id="selection", agent_class=KeywordSelectionAgent),
            SubAgentStepConfig(id="classification", agent_class=ClassificationAgent),
        ]

    # Dependencies each step requires to be pre-populated in SharedContext
    STEP_DEPENDENCIES: Dict[str, List[str]] = {
        "extraction": [],
        "search": ["extracted_keywords"],
        "selection": ["gnd_entries"],
        "classification": ["selected_keywords"],
    }

    def execute(
        self,
        abstract: str,
        initial_keywords: Optional[List[str]] = None,
        config: Optional[MetaAgentConfig] = None,
        input_type: str = "text",
        source_value: Optional[str] = None,
        step_id: Optional[str] = None,
        input_context: Optional[SharedContext] = None,
    ) -> KeywordAnalysisState:
        """Execute the ALIMA pipeline — either all steps or a single step.

        Args:
            abstract: The text to analyze
            initial_keywords: Optional pre-defined keywords
            config: Execution configuration
            input_type: Type of input (text, doi, etc.)
            source_value: Original source (DOI, file path, etc.)
            step_id: If set, run only this one step (e.g. "search", "selection").
                     Requires input_context to supply results from previous steps.
            input_context: Pre-populated SharedContext for warm-start / single-step
                           execution. Overrides abstract/initial_keywords when provided.

        Returns:
            KeywordAnalysisState with results from executed steps
        """
        start_time = time.time()
        config = config or MetaAgentConfig()

        # Load workflow if not already loaded
        if not self.workflow_steps:
            self.load_workflow(config.workflow_name)

        # Build or reuse context
        if input_context is not None:
            context = input_context
            # Apply LLM config from current run (may differ from saved context)
            context.provider = config.provider or context.provider
            context.model = config.model or context.model
            context.temperature = config.temperature
            context.max_tokens = config.max_tokens
            self.logger.info("Using provided input_context (warm-start)")
        else:
            context = SharedContext(
                abstract=abstract,
                initial_keywords=initial_keywords or [],
                input_type=input_type,
                source_value=source_value,
                provider=config.provider,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        mode_label = f"Step '{step_id}'" if step_id else "Full Pipeline"
        self.logger.info(f"Starting MetaAgent {mode_label} with {config.provider}/{config.model}")
        self._stream(f"\n{'='*50}\n🤖 MetaAgent — {mode_label}\n{'='*50}\n", "workflow")

        # Determine which steps to run
        if step_id is not None:
            steps_to_run = [s for s in self.workflow_steps if s.id == step_id]
            if not steps_to_run:
                raise ValueError(
                    f"Step '{step_id}' not found in workflow. "
                    f"Available: {[s.id for s in self.workflow_steps]}"
                )
            # Validate that required preceding data is available
            self._validate_dependencies(step_id, context)
        else:
            steps_to_run = self.workflow_steps

        # Track step results
        step_results: List[PipelineStepResult] = []

        # Create adapter for stream callback
        def step_stream(text: str):
            if self.stream_callback:
                self.stream_callback(text, "agent")

        # Execute selected steps
        for step_config in steps_to_run:
            step_name = step_config.id

            # Skip disabled steps
            if not step_config.enabled:
                self.logger.info(f"Skipping {step_name} (disabled in workflow)")
                continue

            # Skip classification if disabled
            if step_name == "classification" and not config.enable_classification:
                self.logger.info(f"Skipping {step_name} (classification disabled)")
                continue

            # Skip search if GND entries already present (full-pipeline only)
            if step_id is None and step_name == "search" and config.skip_search_if_cached:
                if len(context.gnd_entries) > 0:
                    self.logger.info(f"Skipping {step_name} (GND entries already in context)")
                    continue

            step_result = self._execute_step(
                step_name=step_name,
                agent_class=step_config.agent_class,
                context=context,
                stream_callback=step_stream,
                custom_system_prompt=step_config.custom_system_prompt,
                custom_user_prompt=step_config.custom_user_prompt,
            )
            step_results.append(step_result)

            if not step_result.success:
                self.logger.error(f"Pipeline step {step_name} failed: {step_result.error}")
                # Continue with next step if possible

        duration = time.time() - start_time
        self.logger.info(f"MetaAgent {mode_label} completed in {duration:.1f}s")
        self._stream(f"\n{'='*50}\n✅ {mode_label} abgeschlossen ({duration:.1f}s)\n{'='*50}\n", "workflow")

        cache_stats = context.tool_result_cache.get_stats()
        self.logger.info(
            f"Cache stats: {cache_stats['total_hits']} hits, "
            f"{cache_stats['total_misses']} misses, "
            f"hit rate: {cache_stats['hit_rate']:.1%}"
        )

        return context.to_keyword_analysis_state()

    def _validate_dependencies(self, step_id: str, context: SharedContext) -> None:
        """Validate that required preceding data is present for a single-step run.

        Args:
            step_id: The step to be executed
            context: Current SharedContext

        Raises:
            ValueError: If a required context field is missing
        """
        required = self.STEP_DEPENDENCIES.get(step_id, [])
        missing = []
        for field_name in required:
            value = getattr(context, field_name, None)
            if not value:
                missing.append(field_name)
        if missing:
            raise ValueError(
                f"Step '{step_id}' requires the following context fields to be populated "
                f"(from a previous run or input_context): {missing}"
            )

    def _execute_step(
        self,
        step_name: str,
        agent_class: type,
        context: SharedContext,
        stream_callback: Optional[Callable[[str], None]] = None,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
    ) -> PipelineStepResult:
        """Execute a single pipeline step.

        Args:
            step_name: Name of the step
            agent_class: SubAgent class to instantiate
            context: Shared context
            stream_callback: Optional stream callback
            custom_system_prompt: Optional custom system prompt from workflow
            custom_user_prompt: Optional custom user prompt from workflow

        Returns:
            PipelineStepResult with outcome
        """
        start_time = time.time()
        agent_name = agent_class.__name__

        self._stream(f"\n📍 Step: {step_name}\n", "workflow")
        self.logger.info(f"Executing step '{step_name}' with {agent_name}")

        try:
            # Create agent instance
            agent = agent_class(
                shared_context=context,
                llm_service=self.llm_service,
                tool_registry=self.tool_registry,
                stream_callback=stream_callback,
            )

            # Override prompts if custom prompts provided from workflow
            if custom_system_prompt:
                agent._system_prompt_override = custom_system_prompt
            if custom_user_prompt:
                agent._user_prompt_override = custom_user_prompt

            # Execute agent
            result = agent.execute()

            duration = time.time() - start_time

            self.logger.info(f"Step '{step_name}' completed in {duration:.1f}s")

            return PipelineStepResult(
                step_name=step_name,
                agent_name=agent_name,
                success=result.success,
                duration_seconds=duration,
                data=result.data,
                quality_score=result.quality_score,
                error=result.error,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Step '{step_name}' failed: {e}")

            return PipelineStepResult(
                step_name=step_name,
                agent_name=agent_name,
                success=False,
                duration_seconds=duration,
                data={},
                error=str(e),
            )

    def _stream(self, text: str, step_id: str = "workflow") -> None:
        """Stream output if callback is set.

        Args:
            text: Text to stream
            step_id: Step identifier for the callback
        """
        if self.stream_callback:
            self.stream_callback(text, step_id)

    def get_step_order(self) -> List[str]:
        """Get ordered list of pipeline step names."""
        return [step.id for step in self.workflow_steps]

    def assess_quality(self, context: SharedContext) -> float:
        """Assess overall pipeline quality.

        Args:
            context: Shared context with results

        Returns:
            Quality score from 0.0 to 1.0
        """
        scores = []

        # Keyword extraction quality
        if context.extracted_keywords:
            scores.append(min(len(context.extracted_keywords) / 10, 1.0))

        # GND coverage quality
        if context.selected_keywords:
            scores.append(min(len(context.selected_keywords) / 5, 1.0))

        # Classification quality
        if context.dk_classifications:
            # Weight by confidence
            weighted = sum(cls.get("confidence", 0.5) for cls in context.dk_classifications)
            scores.append(min(weighted / 2, 1.0))

        return sum(scores) / len(scores) if scores else 0.0


def create_meta_agent(
    llm_service,
    config_manager=None,
    stream_callback: Optional[Callable[[str, str], None]] = None,
) -> MetaAgent:
    """Factory function to create a MetaAgent instance.

    Args:
        llm_service: LLM service for generation
        config_manager: Optional config manager
        stream_callback: Optional stream callback

    Returns:
        Configured MetaAgent instance
    """
    return MetaAgent(
        llm_service=llm_service,
        config_manager=config_manager,
        stream_callback=stream_callback,
    )