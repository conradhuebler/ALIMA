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
        max_iterations: int = 30,
        timeout_seconds: int = 600,
        stream_callback: Optional[Callable[[str], None]] = None,
        repeat_threshold: int = 3,
        tool_labels: Optional[Dict[str, str]] = None,
        on_tool_call: Optional[Callable[["ToolCall"], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.stream_callback = stream_callback
        self.repeat_threshold = repeat_threshold
        self.tool_labels = tool_labels or {}
        # P-δ.3 hooks: all default None for full backward compat with
        # LLMAgentStep / ReflectionStep / existing tests.
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.should_stop = should_stop
        # Status channel split: if status_callback is provided, all
        # progress / error / tool-dispatch status lines go there. Otherwise
        # they fall through to stream_callback (legacy behaviour for
        # LLMAgentStep/ReflectionStep which show them in PipelineChatPanel).
        self.status_callback = status_callback

    @property
    def _status_cb(self) -> Optional[Callable[[str], None]]:
        return self.status_callback or self.stream_callback

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
        seed: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
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
            seed: Optional sampling seed for reproducibility (None = non-deterministic)
            conversation_history: Previous user/assistant/tool messages to prepend.

        Returns:
            AgentResult with final content, tool log, and iteration count
        """
        # Get tool schemas - only if tools list is provided
        # None means no tools, empty list means all registered tools
        if tools is None:
            tool_schemas = None  # No tools available
        else:
            tool_schemas = self.tool_registry.get_tool_schemas(tools if tools else None)

        # Build initial messages: system + prior history + current user prompt
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "")
                if role in ("user", "assistant", "tool"):
                    messages.append(dict(msg))
        history_len = len(messages)  # offset before new user prompt
        messages.append({"role": "user", "content": user_prompt})

        tool_log: List[Dict[str, Any]] = []
        tool_call_counter = Counter()  # Track repeated tool calls
        start_time = time.time()
        final_content = ""

        for iteration in range(1, self.max_iterations + 1):
            # Cancel check (P-δ.3 hook). Latency = max one iteration.
            if self.should_stop and self.should_stop():
                logger.info(f"Agent loop stopped by should_stop callback at iteration {iteration}")
                if self._status_cb:
                    self._status_cb(f"\n⏹ Agent-Abbruch durch Operator\n")
                break

            # Timeout check
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                logger.warning(f"Agent timeout after {elapsed:.1f}s at iteration {iteration}")
                if self._status_cb:
                    self._status_cb(f"\n⏰ Agent-Timeout nach {elapsed:.0f}s\n")
                break

            # Call LLM with tools
            logger.info(f"Agent tool-call {iteration}/{self.max_iterations}")
            if self._status_cb and self.max_iterations > 1:
                self._status_cb(f"\n🔄 Tool-Call {iteration}/{self.max_iterations}: Warte auf LLM-Antwort...")

            try:
                response: AgentResponse = self.llm_service.generate_with_tools(
                    provider=provider,
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    stream_callback=self.stream_callback,
                    should_stop=self.should_stop,
                )
            except Exception as e:
                logger.error(f"LLM call failed at tool-call {iteration}: {e}")
                if self._status_cb:
                    self._status_cb(f"\n❌ LLM-Fehler: {e}\n")
                final_content = f"Error: {e}"
                break

            # Case 1: LLM wants to call tools
            if response.has_tool_calls:
                # Show LLM's reasoning before tool calls (transparency)
                if response.content and self._status_cb:
                    reasoning = response.content.strip()
                    if reasoning:
                        # Truncate long reasoning to first 200 chars
                        if len(reasoning) > 200:
                            reasoning = reasoning[:200] + "..."
                        self._status_cb(f"\n💭 {reasoning}\n")

                # Append assistant message with tool calls to conversation
                assistant_msg = self._build_assistant_tool_message(response)
                messages.append(assistant_msg)

                # Execute each tool call
                for tc in response.tool_calls:
                    # Diminishing returns detection
                    call_key = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}"
                    tool_call_counter[call_key] += 1
                    if tool_call_counter[call_key] >= self.repeat_threshold:
                        logger.warning(f"Tool '{tc.name}' called {self.repeat_threshold}x with same args - forcing conclusion")
                        if self._status_cb:
                            self._status_cb(f"\n⚠️ Wiederholte Tool-Aufrufe erkannt, erzwinge Abschluss\n")
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

                    # Show tool type for better transparency.
                    # Phase F: when ``on_tool_call`` is wired the bus/hook
                    # already conveys the call; suppress the duplicate
                    # status line so the chat log doesn't show the same
                    # info twice.
                    tool_type = self._get_tool_type_label(tc.name)
                    if self._status_cb and not self.on_tool_call:
                        args_preview = _truncate_args(tc.arguments, 60)
                        self._status_cb(f"\n  🔧 {tool_type}: {tc.name}({args_preview})")

                    # P-δ.3 hook: notify before tool dispatch
                    if self.on_tool_call:
                        try:
                            self.on_tool_call(tc)
                        except Exception:
                            logger.exception("on_tool_call hook raised")

                    # Execute tool
                    tool_start = time.time()
                    result_str = self.tool_registry.execute(tc.name, tc.arguments)
                    tool_duration = time.time() - tool_start

                    # P-δ.3 hook: notify after tool dispatch
                    if self.on_tool_result:
                        try:
                            self.on_tool_result(tc.name, result_str)
                        except Exception:
                            logger.exception("on_tool_result hook raised")

                    # Log tool call
                    log_entry = {
                        "iteration": iteration,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": result_str[:500],
                        "duration_s": round(tool_duration, 2),
                    }
                    tool_log.append(log_entry)

                    # Show result summary for transparency.
                    # Phase F: same as the call-line above — when
                    # ``on_tool_call`` is set, the hook/bus already carries
                    # the result; skip the duplicate.
                    if self._status_cb and not self.on_tool_call:
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
                        self._status_cb(f"    ✓ {result_preview} ({tool_duration:.1f}s)")

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
            if final_content:
                messages.append({"role": "assistant", "content": final_content})
            if self._status_cb and final_content and self.max_iterations > 1:
                self._status_cb(f"\n✅ Fertig nach {iteration} Tool-Calls\n")
            logger.info(f"Agent completed after {iteration} tool-calls")
            logger.debug(f"LLM response content:\n{final_content}")
            break

        else:
            # max_iterations exhausted
            logger.warning(f"Agent hit max tool-calls ({self.max_iterations})")
            if self._status_cb:
                self._status_cb(f"\n⚠️ Maximum {self.max_iterations} Tool-Calls erreicht\n")

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
                        seed=seed,
                    )
                    final_content = forced.content
                    if final_content:
                        messages.append({"role": "assistant", "content": final_content})
                except Exception:
                    final_content = "Agent reached maximum iterations without conclusion."

        # Extract only messages added during this run (new user prompt + tool calls + assistant)
        conv = [dict(m) for m in messages[history_len:]] if messages else []
        return AgentResult(
            content=final_content,
            tool_log=tool_log,
            iterations=min(iteration, self.max_iterations) if 'iteration' in dir() else 0,
            tokens_used=0,  # TODO: Track from provider responses
            messages=conv,
        )

    def _get_tool_type_label(self, tool_name: str) -> str:
        """Get a human-readable label for the tool type.

        Uses ``self.tool_labels`` if configured, otherwise falls back to
        built-in categories.
        """
        # User-configured labels take priority
        if tool_name in self.tool_labels:
            return self.tool_labels[tool_name]

        # Default categories
        db_tools = {"get_gnd_entry", "get_gnd_batch", "get_dk_cache", "get_classification",
                    "get_search_cache", "list_pipeline_results", "get_pipeline_result",
                    "store_search_result", "get_db_stats"}
        web_tools = {"search_gnd", "search_lobid", "search_swb", "search_catalog",
                     "search_catalog_titles", "resolve_doi"}
        pipeline_tools = {"run_pipeline_step", "save_pipeline_result",
                          "get_pipeline_keywords", "get_pipeline_abstract"}

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
