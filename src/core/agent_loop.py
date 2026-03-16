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
            tools: Tool names to make available (None = all registered)
            provider: LLM provider name
            model: Model name
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Top-p sampling
            max_tokens: Max tokens per LLM call

        Returns:
            AgentResult with final content, tool log, and iteration count
        """
        # Get tool schemas
        tool_schemas = self.tool_registry.get_tool_schemas(tools)

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
                self.stream_callback(f"\n🔄 Iteration {iteration}")

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
                    if self.stream_callback:
                        self.stream_callback(f"\n  🔧 {tc.name}()")

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
                self.stream_callback(f"\n{final_content}")
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

    def _build_assistant_tool_message(self, response: AgentResponse) -> Dict[str, Any]:
        """Build assistant message containing tool calls for conversation history."""
        # Format depends on provider, but we store in a generic format
        # that _convert_messages_for_* methods in LlmService can handle
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
