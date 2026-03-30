"""Base SubAgent Class - Claude Generated

Abstract base class for specialized SubAgents in the MetaAgent architecture.
Each SubAgent handles a specific task in the ALIMA pipeline.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from src.core.agents.shared_context import SharedContext
from src.core.agents.sub_agents.caching_tool_registry import CachingToolRegistry
from src.core.agent_loop import AgentLoop
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    """Result from a SubAgent execution."""
    success: bool
    data: Dict[str, Any]
    quality_score: float = 1.0
    iterations: int = 1
    tool_calls: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "quality_score": self.quality_score,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "error": self.error,
        }


class BaseSubAgent(ABC):
    """Abstract base class for specialized SubAgents.

    Each SubAgent handles a specific task in the ALIMA pipeline:
    - KeywordExtractionAgent: Extract keywords from abstract
    - SearchAgent: Search GND/SWB catalogs
    - KeywordSelectionAgent: Select relevant GND entries
    - ClassificationAgent: Assign DK/RVK classifications

    SubAgents share:
    - SharedContext with tool result cache
    - Conversation memory
    - Previous step results
    """

    def __init__(
        self,
        shared_context: SharedContext,
        llm_service,
        tool_registry: ToolRegistry,
        config: Optional[Any] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize SubAgent.

        Args:
            shared_context: Shared context with cache and memory
            llm_service: LLM service for generation
            tool_registry: Tool registry (will be wrapped with caching)
            config: Optional agent configuration
            stream_callback: Optional callback for streaming output
        """
        self.context = shared_context
        self.llm_service = llm_service
        self.config = config
        self.stream_callback = stream_callback
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Wrap tool registry with caching
        self.caching_registry = CachingToolRegistry(
            tool_registry,
            shared_context.tool_result_cache,
        )

        # Prompt overrides from workflow
        self._system_prompt_override: Optional[str] = None
        self._user_prompt_override: Optional[str] = None

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Name of this agent for logging."""
        pass

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Return list of tool names this agent can use."""
        pass

    @abstractmethod
    def build_user_prompt(self) -> str:
        """Build the user prompt from context and previous results."""
        pass

    @abstractmethod
    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        """Parse LLM output into structured result."""
        pass

    def get_previous_context(self) -> str:
        """Get context from previous pipeline steps.

        Returns:
            Formatted string with previous step results
        """
        context_parts = []

        # Add initial keywords if available
        if self.context.initial_keywords:
            context_parts.append(f"Initial Keywords: {', '.join(self.context.initial_keywords[:20])}")

        # Add working title if available
        if self.context.working_title:
            context_parts.append(f"Working Title: {self.context.working_title}")

        # Add previous step results
        for step_name, result in self.context.step_results.items():
            if result and isinstance(result, dict):
                context_parts.append(f"[{step_name}]: {json.dumps(result, ensure_ascii=False)[:500]}...")

        if context_parts:
            return "\n".join(context_parts)
        return ""

    def _replace_placeholders(self, template: str) -> str:
        """Replace placeholders in template with context values.

        Handles placeholders like {abstract}, {keywords}, {gnd_entries}, etc.

        Args:
            template: Template string with placeholders

        Returns:
            Template with placeholders replaced by context values
        """
        result = template

        # Replace abstract
        if "{abstract}" in result:
            result = result.replace("{abstract}", self.context.abstract[:5000])

        # Replace keywords
        if "{keywords}" in result:
            keywords = self.context.extracted_keywords or self.context.initial_keywords
            result = result.replace("{keywords}", ", ".join(keywords[:50]))

        # Replace gnd_entries (as JSON string, truncated)
        if "{gnd_entries}" in result:
            import json
            gnd_json = json.dumps(self.context.gnd_entries[:30], ensure_ascii=False, indent=2)
            result = result.replace("{gnd_entries}", gnd_json)

        # Replace working title
        if "{title}" in result or "{working_title}" in result:
            result = result.replace("{title}", self.context.working_title or "")
            result = result.replace("{working_title}", self.context.working_title or "")

        # Replace selected keywords
        if "{selected_keywords}" in result:
            import json
            selected_json = json.dumps(self.context.selected_keywords[:30], ensure_ascii=False, indent=2)
            result = result.replace("{selected_keywords}", selected_json)

        # Replace previous context
        if "{previous_context}" in result:
            result = result.replace("{previous_context}", self.get_previous_context())

        return result

    def execute(self) -> SubAgentResult:
        """Execute this SubAgent.

        Returns:
            SubAgentResult with success status and data
        """
        self.logger.info(f"Starting {self.agent_name} execution")
        self.logger.debug(f"Context summary: {self.context.get_summary()}")

        # Get available tools for display
        tools = self.get_available_tools()
        tools_info = f" (Tools: {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''})" if tools else " (keine Tools)"

        if self.stream_callback:
            self.stream_callback(f"\n{'='*50}\n🤖 {self.agent_name}{tools_info}\n{'='*50}\n")

        try:
            # Build prompts - use overrides if provided from workflow
            system_prompt = self._system_prompt_override or self.get_system_prompt()
            user_prompt = self._user_prompt_override or self.build_user_prompt()

            # Replace placeholders in custom prompts
            if self._user_prompt_override:
                user_prompt = self._replace_placeholders(user_prompt)
            if self._system_prompt_override:
                system_prompt = self._replace_placeholders(system_prompt)

            # Show task summary (first line of system prompt)
            task_summary = system_prompt.split('\n')[0][:80]
            if self.stream_callback:
                self.stream_callback(f"📋 Aufgabe: {task_summary}\n")
                # Show first line of user prompt (truncated, without abstract)
                user_preview = user_prompt.split('\n')[0][:100]
                self.stream_callback(f"📝 Input: {user_preview}...\n")
                self.stream_callback(f"⏳ Sende an LLM ({self.context.provider}/{self.context.model})...\n")

            # Get available tools - None means no tools, empty list is still tools
            tools = self.get_available_tools()
            tool_schemas = self.caching_registry.get_tool_schemas(tools) if tools else None

            # Create agent loop for tool calling
            agent_loop = AgentLoop(
                llm_service=self.llm_service,
                tool_registry=self.caching_registry,
                max_iterations=20,
                timeout_seconds=300,
                stream_callback=self.stream_callback,
            )

            # Execute with LLM
            # Pass tools=None (not empty list) to indicate no tool calling
            result = agent_loop.run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools if tools else None,  # None disables tool calling
                provider=self.context.provider,
                model=self.context.model,
                temperature=self.context.temperature,
                max_tokens=self.context.max_tokens,
            )

            # Parse result
            parsed_data = self.parse_result(result.content)

            # Store in context
            self.context.set_step_result(
                self.agent_id,
                parsed_data,
                quality=1.0,  # Could calculate based on validation
            )

            # Update shared context with extracted data
            self._update_shared_context(parsed_data)

            # Add to conversation memory
            self.context.add_message("assistant", result.content[:1000])

            self.logger.info(f"{self.agent_name} completed successfully")

            return SubAgentResult(
                success=True,
                data=parsed_data,
                quality_score=1.0,
                iterations=result.iterations,
                tool_calls=len(result.tool_log) if hasattr(result, 'tool_log') else 0,
            )

        except Exception as e:
            self.logger.error(f"{self.agent_name} failed: {e}")
            if self.stream_callback:
                self.stream_callback(f"\n❌ {self.agent_name} Fehler: {e}\n")

            return SubAgentResult(
                success=False,
                data={},
                error=str(e),
            )

    def _update_shared_context(self, result_data: Dict[str, Any]) -> None:
        """Update shared context with results from this agent.

        Override in subclasses to update specific context fields.

        Args:
            result_data: Parsed result from this agent
        """
        pass

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM output.

        Handles markdown code blocks and nested JSON.

        Args:
            content: Raw LLM output

        Returns:
            Parsed JSON dict
        """
        import re

        # Try markdown code block first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        json_objects = re.findall(r'\{[^{}]*\}', content, re.DOTALL)
        for obj in reversed(json_objects):  # Start from end (usually final answer)
            try:
                parsed = json.loads(obj)
                if isinstance(parsed, dict) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue

        # Fallback: return raw content
        return {"raw_content": content}