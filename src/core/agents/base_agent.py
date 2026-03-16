"""Base agent class with self-reflection loop - Claude Generated

All ALIMA agents inherit from BaseAgent and implement:
- execute(): Core logic using AgentLoop with MCP tools
- self_validate(): Quality check on results
- _adjust_strategy_for_retry(): Parameter tweaks for next attempt
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

from src.core.data_models import AgentResult
from src.core.agent_loop import AgentLoop
from src.mcp.tool_registry import ToolRegistry


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent instance - Claude Generated"""
    provider: str = ""
    model: str = ""
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 4096
    max_self_reflection_iterations: int = 3
    max_tool_iterations: int = 20
    timeout_seconds: int = 300
    quality_threshold: float = 0.6


@dataclass
class QualityMetrics:
    """Quality assessment of an agent's output - Claude Generated"""
    score: float = 0.0  # 0.0 - 1.0
    coverage: float = 0.0
    relevance: float = 0.0
    diversity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passes_threshold(self) -> bool:
        return self.score >= 0.6  # Default threshold


class BaseAgent(ABC):
    """
    Abstract base agent with self-reflection loop - Claude Generated

    Lifecycle:
    1. execute() runs the AgentLoop with domain-specific tools and prompts
    2. self_validate() assesses output quality
    3. If quality < threshold, _adjust_strategy_for_retry() modifies params
    4. Repeat up to max_self_reflection_iterations
    """

    def __init__(
        self,
        llm_service,
        tool_registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()
        self.stream_callback = stream_callback
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique name for this agent type."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """System prompt defining the agent's role and capabilities."""
        ...

    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Tool names this agent can use."""
        ...

    @abstractmethod
    def build_user_prompt(self, input_data: Dict[str, Any], **kwargs) -> str:
        """Build the user prompt from input data."""
        ...

    @abstractmethod
    def parse_result(self, agent_result: AgentResult, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AgentResult into domain-specific structured output."""
        ...

    def self_validate(self, result: Dict[str, Any], input_data: Dict[str, Any]) -> QualityMetrics:
        """
        Assess quality of agent output. Override for domain-specific validation.
        Default: always passes.
        """
        return QualityMetrics(score=1.0, coverage=1.0, relevance=1.0, diversity=1.0)

    def _adjust_strategy_for_retry(self, metrics: QualityMetrics, config: AgentConfig, iteration: int) -> AgentConfig:
        """
        Adjust strategy based on quality metrics. Override for domain-specific adjustments.
        Default: increase temperature slightly.
        """
        new_config = AgentConfig(
            provider=config.provider,
            model=config.model,
            temperature=min(config.temperature + 0.1, 1.0),
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            max_self_reflection_iterations=config.max_self_reflection_iterations,
            max_tool_iterations=config.max_tool_iterations,
            timeout_seconds=config.timeout_seconds,
            quality_threshold=config.quality_threshold,
        )
        return new_config

    def execute_with_self_reflection(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Full execution with self-reflection loop.

        Returns domain-specific result dict (from parse_result).
        """
        config = self.config
        best_result = None
        best_metrics = QualityMetrics(score=0.0)

        for iteration in range(1, config.max_self_reflection_iterations + 1):
            self.logger.info(f"{self.agent_name} self-reflection iteration {iteration}/{config.max_self_reflection_iterations}")

            if self.stream_callback:
                self.stream_callback(f"\n{'='*40}\n🤖 {self.agent_name} - Durchlauf {iteration}\n")

            # Run the agent loop
            agent_loop = AgentLoop(
                llm_service=self.llm_service,
                tool_registry=self.tool_registry,
                max_iterations=config.max_tool_iterations,
                timeout_seconds=config.timeout_seconds,
                stream_callback=self.stream_callback,
            )

            user_prompt = self.build_user_prompt(input_data, iteration=iteration, **kwargs)

            agent_result = agent_loop.run(
                system_prompt=self.get_system_prompt(),
                user_prompt=user_prompt,
                tools=self.get_available_tools(),
                provider=config.provider,
                model=config.model,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
            agent_result.agent_name = self.agent_name

            # Parse into domain-specific result
            parsed = self.parse_result(agent_result, input_data)
            parsed["_agent_result"] = agent_result
            parsed["_iteration"] = iteration

            # Validate quality
            metrics = self.self_validate(parsed, input_data)
            parsed["_quality"] = {
                "score": metrics.score, "coverage": metrics.coverage,
                "relevance": metrics.relevance, "diversity": metrics.diversity,
                "details": metrics.details,
            }

            self.logger.info(
                f"{self.agent_name} iteration {iteration}: "
                f"score={metrics.score:.2f} coverage={metrics.coverage:.2f} "
                f"relevance={metrics.relevance:.2f}"
            )

            # Track best result
            if metrics.score > best_metrics.score:
                best_result = parsed
                best_metrics = metrics

            # Quality threshold met?
            if metrics.score >= config.quality_threshold:
                self.logger.info(f"{self.agent_name} passed quality threshold ({metrics.score:.2f} >= {config.quality_threshold})")
                if self.stream_callback:
                    self.stream_callback(f"\n✅ Qualität ausreichend (Score: {metrics.score:.2f})\n")
                return parsed

            # Adjust strategy for retry
            if iteration < config.max_self_reflection_iterations:
                config = self._adjust_strategy_for_retry(metrics, config, iteration)
                if self.stream_callback:
                    self.stream_callback(
                        f"\n🔄 Qualität nicht ausreichend (Score: {metrics.score:.2f}), "
                        f"Strategie wird angepasst...\n"
                    )

        # Return best result from all iterations
        self.logger.info(f"{self.agent_name} completed after {config.max_self_reflection_iterations} iterations, best score: {best_metrics.score:.2f}")
        if self.stream_callback:
            self.stream_callback(f"\n⚠️ Max Iterationen erreicht, bestes Ergebnis: Score {best_metrics.score:.2f}\n")
        return best_result or parsed
