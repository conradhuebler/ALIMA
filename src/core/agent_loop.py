"""Generic provider-agnostic tool-calling agent loop - Claude Generated

Drives multi-turn LLM conversations with tool use, supporting any provider
that implements generate_with_tools() in LlmService.
"""
import logging
import re
import time
import json
from typing import List, Dict, Any, Optional, Callable
from collections import Counter

from src.core.data_models import AgentResponse, AgentResult, ToolCall, ToolResult, StopReason
from src.core.processing_utils import strip_think_tags
from src.mcp.tool_registry import ToolRegistry
from src.utils.error_visibility import log_caught

logger = logging.getLogger(__name__)


class ThinkStreamFilter:
    """Split a token stream into answer text and <think>…</think> content - Claude Generated

    Feeds text outside think tags to ``on_text`` and text inside to
    ``on_thinking``. Tags may arrive split across arbitrary token boundaries:
    the filter holds back the longest buffer suffix that is a prefix of the
    tag it is currently waiting for. ``flush()`` emits any held-back text to
    the current mode's sink (an unclosed ``<think>`` therefore stays thinking).
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(
        self,
        on_text: Optional[Callable[[str], None]],
        on_thinking: Optional[Callable[[str], None]],
    ):
        self.on_text = on_text
        self.on_thinking = on_thinking
        self._buf = ""
        self._in_think = False

    def feed(self, token: str) -> None:
        if not token:
            return
        self._buf += token
        while True:
            tag = self._CLOSE if self._in_think else self._OPEN
            idx = self._buf.find(tag)
            if idx != -1:
                self._emit(self._buf[:idx])
                self._buf = self._buf[idx + len(tag):]
                self._in_think = not self._in_think
                continue
            held = self._partial_tag_suffix(self._buf, tag)
            if len(self._buf) > held:
                self._emit(self._buf[: len(self._buf) - held])
                self._buf = self._buf[len(self._buf) - held:]
            return

    def flush(self) -> None:
        buf, self._buf = self._buf, ""
        self._emit(buf)

    def _emit(self, text: str) -> None:
        if not text:
            return
        sink = self.on_thinking if self._in_think else self.on_text
        if sink:
            sink(text)

    @staticmethod
    def _partial_tag_suffix(buf: str, tag: str) -> int:
        for k in range(min(len(buf), len(tag) - 1), 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0


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
        on_thinking: Optional[Callable[[str], None]] = None,
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
        # Thinking channel: when set, streamed <think>…</think> content and the
        # provider reasoning channel are routed here instead of into the answer
        # stream. None (default) keeps the token stream unfiltered. - Claude Generated
        self.on_thinking = on_thinking

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
        think: Optional[bool] = None,
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
        final_stop_reason: StopReason = StopReason.END_TURN  # tracked for diagnosis - Claude Generated
        nudged = False  # one-time "write the final answer" retry - Claude Generated
        # Set when the run produced no model answer — an LLM exception, or a
        # turn the loop had to answer for itself with a diagnostic message.
        # Both put loop-authored text into ``content``, so ``AgentResult.error``
        # is the only way a caller can tell that text apart from a model answer.
        # Leaving it unset made an empty turn look like a successful step.
        # See AgentResult.error. - Claude Generated
        run_error: Optional[str] = None

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
                self._status_cb(f"\n🔄 Iteration {iteration}/{self.max_iterations}: Warte auf LLM-Antwort...")

            # Fresh per-iteration filter: diverts streamed <think> content to
            # the thinking channel; without on_thinking the stream stays
            # untouched (pipeline paths). - Claude Generated
            think_filter: Optional[ThinkStreamFilter] = None
            stream_cb = self.stream_callback
            if self.on_thinking and self.stream_callback:
                think_filter = ThinkStreamFilter(self.stream_callback, self.on_thinking)
                stream_cb = think_filter.feed

            # Providers with a SEPARATE reasoning channel (Ollama, vLLM) used to
            # hand it over only as the finished string on the response, so the
            # thinking appeared in one lump after the turn while the inline
            # <think> dialect had been streaming live all along. This routes the
            # channel through the same sink, per chunk. - Claude Generated
            streamed_thinking = [False]

            def _thinking_cb(text: str) -> None:
                streamed_thinking[0] = True
                self.on_thinking(text)

            thinking_cb = _thinking_cb if self.on_thinking else None

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
                    stream_callback=stream_cb,
                    should_stop=self.should_stop,
                    think=think,
                    thinking_callback=thinking_cb,
                )
            except Exception as e:
                if think_filter:
                    think_filter.flush()
                logger.error(f"LLM call failed at tool-call {iteration}: {e}")
                if self._status_cb:
                    self._status_cb(f"\n❌ LLM-Fehler: {e}\n")
                final_content = f"Error: {e}"
                run_error = str(e)  # mark run as failed, not just oddly-worded - Claude Generated
                break
            if think_filter:
                think_filter.flush()

            # Provider reasoning channel → thinking block. Only when it did NOT
            # already arrive chunk-wise above (non-streaming call, or a provider
            # whose generator has no reasoning channel yet) — otherwise the whole
            # block would be appended a second time. May still cosmetically
            # duplicate streamed <think> content if a provider delivers both
            # dialects. - Claude Generated
            if self.on_thinking and not streamed_thinking[0] and getattr(response, "reasoning", ""):
                try:
                    self.on_thinking(response.reasoning)
                except Exception as e:
                    log_caught(logger, e, "on_thinking hook (reasoning channel)")

            # Case 1: LLM wants to call tools
            if response.has_tool_calls:
                # Show LLM's reasoning before tool calls (transparency).
                # When the content already streamed live, this excerpt is by
                # definition a duplicate → skip (mirrors the on_tool_call
                # suppression pattern below). - Claude Generated
                if response.content and self._status_cb and not self.stream_callback:
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
                    # Cancel check between tool calls in a batch: without this, a
                    # multi-tool response runs every queued tool before the loop
                    # re-checks at the iteration boundary, so cancel latency could
                    # span several tool calls. Bound it to one in-flight tool. The
                    # outer loop then breaks at the next boundary. - Claude Generated
                    if self.should_stop and self.should_stop():
                        logger.info("Agent loop stop requested mid tool-batch")
                        break

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
                        args_preview = _truncate_args(tc.arguments, 120)
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

                    # Log tool call. result_full carries the complete,
                    # untruncated tool result (result_preview stays capped at
                    # 500 chars for display) so deterministic steps can parse
                    # a tool's real output directly instead of trusting the
                    # LLM to correctly retype it into a final JSON answer —
                    # for a long list of tool results, that transcription is
                    # lossy and produces the exact "found it last time, not
                    # this time" inconsistency deterministic extraction
                    # avoids entirely. - Claude Generated
                    log_entry = {
                        "iteration": iteration,
                        "tool": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": result_str[:500],
                        "result_full": result_str,
                        "duration_s": round(tool_duration, 2),
                    }
                    tool_log.append(log_entry)

                    # The result went only to the status callback, so a run's
                    # log recorded that a tool ran but never what it returned —
                    # which left "did rvk_lookup propose this notation, or did
                    # the model?" unanswerable after the fact. Capped like the
                    # prompt dumps around it. - Claude Generated
                    logger.info(
                        f"  ↩️ {tc.name} → {result_str[:500]}"
                        + (f" … ({len(result_str)} chars total)" if len(result_str) > 500 else "")
                        + f" [{tool_duration:.1f}s]"
                    )

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
                        except (json.JSONDecodeError, TypeError):
                            pass  # keep generic preview - Claude Generated
                        self._status_cb(f"    ✓ {result_preview} ({tool_duration:.1f}s)")

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                        "name": tc.name,
                    })

                # If there was also text content, accumulate it. Separate
                # iterations with a blank line so markdown doesn't glue the
                # turns into one paragraph. - Claude Generated
                if response.content:
                    final_content += ("\n\n" if final_content else "") + response.content

                # Continue loop for next LLM turn
                continue

            # Case 2: LLM returned final text (no tool calls)
            final_content = response.content
            final_stop_reason = response.stop_reason

            # One-time nudge: small/code models often call tools but never write
            # a final answer (empty turn). Ask explicitly for a text answer (no
            # tools) before giving up. Skip if the model used its reasoning
            # channel (handled below). - Claude Generated
            if not final_content and not nudged and not getattr(response, "reasoning", ""):
                nudged = True
                if self._status_cb:
                    self._status_cb("\n📝 Modell ohne Text — fordere finale Antwort an…")
                messages.append({
                    "role": "user",
                    "content": (
                        "Bitte schreibe JETZT die finale Antwort als normalen Text "
                        "auf Deutsch, basierend auf den bisherigen Ergebnissen. "
                        "Rufe KEIN weiteres Tool auf."
                    ),
                })
                try:
                    forced = self.llm_service.generate_with_tools(
                        provider=provider, model=model, messages=messages, tools=[],
                        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                        seed=seed, think=think,
                    )
                    final_content = forced.content or getattr(forced, "reasoning", "")
                    final_stop_reason = forced.stop_reason
                    if final_content:
                        final_content = self._emit_final(final_content)
                except Exception:
                    logger.exception("final-answer nudge failed")

            if not final_content:
                # No text AND no tool calls — surface WHY instead of a silent
                # empty bubble. Prefer the reasoning channel; otherwise explain
                # via stop_reason. Stream it so both frontends show it live. - Claude Generated
                reasoning_text = getattr(response, "reasoning", "") or ""
                # Truncation is checked BEFORE the reasoning channel: a run cut
                # off by max_tokens leaves an unfinished train of thought, and
                # printing that as the answer hides why the answer is missing. - Claude Generated
                if response.stop_reason == StopReason.MAX_TOKENS:
                    # The advice depends on WHERE the budget went. Measured on
                    # the real alima_v51 extraction step (deepseek-v4-flash,
                    # scripts/probe_thinking.py): a reasoning channel grows with
                    # the budget it is given (6406 → 56886 characters between
                    # 4096 and 16384 max_tokens), so recommending a bigger
                    # budget there sends the operator down a road that ends in
                    # the same place, only slower. - Claude Generated
                    if reasoning_text:
                        budget = (
                            f" Davon entfielen {len(reasoning_text)} Zeichen auf den "
                            f"Reasoning-Kanal."
                        )
                        advice = (
                            " Bei einem Reasoning-Modell wirkt „Thinking: Aus\" "
                            "(think=false) sofort; ein größeres Budget hilft auch, "
                            "muss dafür aber deutlich größer sein, denn der "
                            "Denkkanal wächst mit (Toolbar „Budget\", CLI "
                            "--max-tokens)."
                        )
                    else:
                        budget = ""
                        advice = " max_tokens erhöhen oder die Eingabe kürzen."
                    final_content = (
                        f"⚠️ Das Modell hat das Token-Budget (max_tokens={max_tokens}) "
                        f"aufgebraucht, bevor eine Antwort kam.{budget}{advice}"
                    )
                    run_error = final_content
                elif reasoning_text:
                    # Model output, not a loop-authored diagnostic: the answer
                    # arrived in the wrong channel, but it IS the model talking.
                    # No run_error — a caller may still find its JSON in there.
                    final_content = (
                        "💭 (Modell antwortete nur im Reasoning-Kanal, keine "
                        "separate finale Antwort):\n\n" + reasoning_text
                    )
                else:
                    final_content = (
                        "⚠️ Das Modell hat keine Antwort geliefert "
                        "(leerer Inhalt, keine Tool-Calls)."
                    )
                    run_error = final_content
                if self.stream_callback:
                    self.stream_callback(final_content)
            if final_content:
                messages.append({"role": "assistant", "content": strip_think_tags(final_content)})
            if self._status_cb and final_content and self.max_iterations > 1:
                self._status_cb(
                    f"\n✅ Fertig nach {iteration} Iteration(en), "
                    f"{len(tool_log)} Tool-Call(s)\n"
                )
            logger.info(f"Agent completed after {iteration} iterations, {len(tool_log)} tool-calls")
            logger.debug(f"LLM response content:\n{final_content}")
            break

        else:
            # max_iterations exhausted
            logger.warning(f"Agent hit max iterations ({self.max_iterations})")
            if self._status_cb:
                self._status_cb(f"\n⚠️ Maximum {self.max_iterations} Iterationen erreicht\n")

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
                        seed=seed, think=think,
                    )
                    final_stop_reason = forced.stop_reason
                    final_content = forced.content or getattr(forced, "reasoning", "")
                    if final_content:
                        # Stream the forced answer too — otherwise it is the one
                        # final-content path the frontends never see live. - Claude Generated
                        final_content = self._emit_final(final_content)
                    if final_content:
                        messages.append({"role": "assistant", "content": final_content})
                except Exception:
                    final_content = "Agent reached maximum iterations without conclusion."
                    run_error = final_content
                # Some models (e.g. code models) call tools but never write a
                # final answer → don't end on a silent empty bubble. - Claude Generated
                if not final_content:
                    final_content = (
                        "⚠️ Das Modell hat nach mehreren Tool-Aufrufen keine finale "
                        "Textantwort geliefert. Dieses Modell schreibt im Tool-Modus "
                        "oft keinen Abschlusstext — ggf. ein anderes Chat-Modell wählen."
                    )
                    run_error = final_content
                    if self.stream_callback:
                        self.stream_callback(final_content)

        # Extract only messages added during this run (new user prompt + tool calls + assistant)
        conv = [dict(m) for m in messages[history_len:]] if messages else []
        return AgentResult(
            content=strip_think_tags(final_content) if final_content else final_content,
            tool_log=tool_log,
            iterations=min(iteration, self.max_iterations) if 'iteration' in dir() else 0,
            tokens_used=0,  # TODO: Track from provider responses
            messages=conv,
            error=run_error,
            stop_reason=getattr(final_stop_reason, "value", str(final_stop_reason)),
        )

    def _emit_final(self, text: str) -> str:
        """Deliver a non-streamed final answer to the frontends - Claude Generated

        Routes ``<think>`` parts to the thinking channel (if wired), streams the
        clean remainder, and returns it. May return "" if the answer was
        thinking-only — callers fall back to their empty-answer handling.
        """
        if self.on_thinking:
            thinking = "\n".join(
                part for part in re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
                if part.strip()
            )
            if thinking.strip():
                try:
                    self.on_thinking(thinking)
                except Exception as e:
                    log_caught(logger, e, "on_thinking hook (final answer)")
        clean = strip_think_tags(text).strip()
        if clean and self.stream_callback:
            self.stream_callback(clean)
        return clean

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
            "content": strip_think_tags(response.content or ""),
            "tool_calls": tool_calls_data,
        }


def _truncate_args(args: Dict[str, Any], max_len: int = 150) -> str:
    """Format tool-call arguments for logging.

    A list value (e.g. `terms` with dozens of book titles) gets a "N items:
    first, second, …" preview instead of json.dumps()-then-truncate, which
    used to cut off mid-way through the FIRST element and hide that there
    even were more — e.g. `search_finc(terms=['The Fraying Bonds of Peace –
    Economic …)` told an operator nothing about how many titles were
    actually being searched. - Claude Generated
    """
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, list) and v:
            preview = ", ".join(
                (str(item)[:30] + "…") if len(str(item)) > 30 else str(item)
                for item in v[:2]
            )
            more = ", …" if len(v) > 2 else ""
            parts.append(f"{k}=[{len(v)}: {preview}{more}]")
        else:
            parts.append(f"{k}={v!r}")
    s = ", ".join(parts)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
