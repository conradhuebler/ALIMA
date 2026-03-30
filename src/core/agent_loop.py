"""Generic provider-agnostic tool-calling agent loop - Claude Generated

Drives multi-turn LLM conversations with tool use, supporting any provider
that implements generate_with_tools() in LlmService.
"""
import logging
import time
import json
from typing import List, Dict, Any, Optional, Callable
from collections import Counter

from src.core.data_models import AgentResponse, AgentResult, ToolCall, ToolResult, StopReason
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    Generic agent loop: LLM decides → tools execute → results fed back → repeat.

    Works with any LlmService provider that supports generate_with_tools().
    Includes safety features: max iterations, diminishing-returns detection, timeout.
    """

    def __init__(
        self,
        llm_service,
        tool_registry: ToolRegistry,
        max_iterations: int = 20,
        timeout_seconds: int = 300,
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.stream_callback = stream_callback

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Optional[List[str]] = None,
        provider: str = "",
        model: str = "",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> AgentResult:
        """
        Execute a full agent run with tool-calling loop.

        Args:
            system_prompt: System instructions for the agent
            user_prompt: User's request
            tools: Tool names to make available (None = no tools, [] = all registered)
            provider: LLM provider name
            model: Model name
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Top-p sampling
            max_tokens: Max tokens per LLM call

        Returns:
            AgentResult with final content, tool log, and iteration count
        """
        # Get tool schemas - only if tools list is provided
        # None means no tools, empty list means all registered tools
        if tools is None:
            tool_schemas = None  # No tools available
        else:
            tool_schemas = self.tool_registry.get_tool_schemas(tools if tools else None)

        # Build initial messages
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tool_log: List[Dict[str, Any]] = []
        tool_call_counter = Counter()  # Track repeated tool calls
        start_time = time.time()
        final_content = ""

        for iteration in range(1, self.max_iterations + 1):
            # Timeout check
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                logger.warning(f"Agent timeout after {elapsed:.1f}s at iteration {iteration}")
                if self.stream_callback:
                    self.stream_callback(f"\n⏰ Agent-Timeout nach {elapsed:.0f}s\n")
                break

            # Call LLM with tools
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")
            if self.stream_callback:
                self.stream_callback(f"\n🔄 Iteration {iteration}: Warte auf LLM-Antwort...")

            try:
                response: AgentResponse = self.llm_service.generate_with_tools(
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream_callback=self.stream_callback,
                )
            except Exception as e:
                logger.error(f"LLM call failed at iteration {iteration}: {e}")
                if self.stream_callback:
                    self.stream_callback(f"\n❌ LLM-Fehler: {e}\n")
                final_content = f"Error: {e}"
                break

            # Case 1: LLM wants to call tools
            if response.has_tool_calls:
                # Show LLM's reasoning before tool calls (transparency)
                if response.content and self.stream_callback:
                    reasoning = response.content.strip()
                    if reasoning:
                        # Truncate long reasoning to first 200 chars
                        if len(reasoning) > 200:
                            reasoning = reasoning[:200] + "..."
                        self.stream_callback(f"\n💭 {reasoning}\n")

                # Append assistant message with tool calls to conversation
                assistant_msg = self._build_assistant_tool_message(response)
                messages.append(assistant_msg)

                # Execute each tool call
                for tc in response.tool_calls:
                    # Diminishing returns detection
                    call_key = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}"
                    tool_call_counter[call_key] += 1
                    if tool_call_counter[call_key] >= 3:
                        logger.warning(f"Tool '{tc.name}' called 3x with same args - forcing conclusion")
                        if self.stream_callback:
                            self.stream_callback(f"\n⚠️ Wiederholte Tool-Aufrufe erkannt, erzwinge Abschluss\n")
                        # Force conclusion by not providing more tool results
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({
                                "error": "Tool called repeatedly with identical arguments. "
                                         "Please provide your final answer based on results so far."
                            }),
                            "name": tc.name,
                        })
                        continue

                    logger.info(f"  🔧 Executing tool: {tc.name}({_truncate_args(tc.arguments)})")

                    # Show tool type for better transparency
                    tool_type = self._get_tool_type_label(tc.name)
                    if self.stream_callback:
                        args_preview = _truncate_args(tc.arguments, 60)
                        self.stream_callback(f"\n  🔧 {tool_type}: {tc.name}({args_preview})")

                    # Execute tool
                    tool_start = time.time()
                    result_str = self.tool_registry.execute(tc.name, tc.arguments)
                    tool_duration = time.time() - tool_start

                    # Log tool call
                    log_entry = {
                        "iteration": iteration,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": result_str[:500],
                        "duration_s": round(tool_duration, 2),
                    }
                    tool_log.append(log_entry)

                    # Show result summary for transparency
                    if self.stream_callback:
                        result_preview = result_str[:100] + "..." if len(result_str) > 100 else result_str
                        # Count results if it's a list
                        try:
                            parsed = json.loads(result_str)
                            if isinstance(parsed, list):
                                result_preview = f"{len(parsed)} Ergebnisse"
                            elif isinstance(parsed, dict):
                                result_preview = f"{len(parsed)} Einträge"
                        except:
                            pass
                        self.stream_callback(f"    ✓ {result_preview} ({tool_duration:.1f}s)")

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                        "name": tc.name,
                    })

                # If there was also text content, accumulate it
                if response.content:
                    final_content += response.content

                # Continue loop for next LLM turn
                continue

            # Case 2: LLM returned final text (no tool calls)
            final_content = response.content
            if self.stream_callback and final_content:
                self.stream_callback(f"\n✅ Fertig nach {iteration} Iterationen\n")
            logger.info(f"Agent completed after {iteration} iterations")
            break

        else:
            # max_iterations exhausted
            logger.warning(f"Agent hit max iterations ({self.max_iterations})")
            if self.stream_callback:
                self.stream_callback(f"\n⚠️ Maximum {self.max_iterations} Iterationen erreicht\n")

            # Force a final response without tools
            if not final_content:
                messages.append({
                    "role": "user",
                    "content": "Maximum iterations reached. Please provide your final answer now based on all information gathered so far.",
                })
                try:
                    forced = self.llm_service.generate_with_tools(
                        provider=provider, model=model,
                        messages=messages, tools=[],  # No tools = force text response
                        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                    )
                    final_content = forced.content
                except Exception:
                    final_content = "Agent reached maximum iterations without conclusion."

        return AgentResult(
            content=final_content,
            tool_log=tool_log,
            iterations=min(iteration, self.max_iterations) if 'iteration' in dir() else 0,
            tokens_used=0,  # TODO: Track from provider responses
        )

    def _get_tool_type_label(self, tool_name: str) -> str:
        """Get a human-readable label for the tool type.

        Returns:
            Label indicating the tool category (LLM, DB, Web, etc.)
        """
        # Database tools
        db_tools = {"get_gnd_entry", "get_gnd_batch", "get_dk_cache", "get_classification",
                    "get_search_cache", "list_pipeline_results", "get_pipeline_result"}
        # Web/API tools
        web_tools = {"search_gnd", "search_lobid", "search_swb", "search_catalog"}
        # Pipeline tools
        pipeline_tools = {"run_pipeline_step", "save_pipeline_result"}

        if tool_name in db_tools:
            return "🗄️ DB"
        elif tool_name in web_tools:
            return "🌐 Web"
        elif tool_name in pipeline_tools:
            return "⚙️ Pipeline"
        else:
            return "🔧 Tool"

    def _build_assistant_tool_message(self, response: AgentResponse) -> Dict[str, Any]:
        """Build assistant message containing tool calls for conversation history."""
        # Store in generic format - converters in LlmService will transform for each provider
        tool_calls_data = []
        for tc in response.tool_calls:
            tool_calls_data.append({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            })
        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": tool_calls_data,
        }


def _truncate_args(args: Dict[str, Any], max_len: int = 80) -> str:
    """Truncate arguments for logging."""
    s = json.dumps(args, ensure_ascii=False)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
