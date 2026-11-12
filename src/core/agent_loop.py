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
from src.mcp.tool_schemas import SET_GENERATION_PARAMS
from src.utils.error_visibility import log_caught
from src.utils.repetition_detector import resolve_repetition_config, retry_sampling_params

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
        repetition_config: Optional[Any] = None,
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
        # Repetition settings (detection lives in generate_with_tools; what
        # happens after a REPETITION stop is decided here, see
        # _handle_repetition). Explicit config wins, else the service's
        # config manager, else the defaults. - Claude Generated
        self.repetition_config = resolve_repetition_config(
            repetition_config if repetition_config is not None
            else getattr(llm_service, "config_manager", None)
        )

    @property
    def _status_cb(self) -> Optional[Callable[[str], None]]:
        return self.status_callback or self.stream_callback

    def _generate(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools,
        gen: Dict[str, Any],
        seed: Optional[int],
        think: Optional[bool],
        stream_callback: Optional[Callable[[str], None]],
        thinking_callback: Optional[Callable[[str], None]],
    ) -> AgentResponse:
        """One generate_with_tools call with the run's current sampling - Claude Generated

        ``gen`` holds temperature/top_p/max_tokens/repetition_penalty for this
        run; set_generation_params and the ``retry`` action change it.
        """
        kwargs: Dict[str, Any] = dict(
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            max_tokens=gen["max_tokens"],
            seed=seed,
            stream_callback=stream_callback,
            should_stop=self.should_stop,
            think=think,
            thinking_callback=thinking_callback,
            repetition_config=self.repetition_config,
        )
        if gen.get("repetition_penalty") is not None:
            kwargs["repetition_penalty"] = gen["repetition_penalty"]
        return self.llm_service.generate_with_tools(**kwargs)

    def _handle_repetition(
        self,
        response: AgentResponse,
        messages: List[Dict[str, Any]],
        gen: Dict[str, Any],
        state: Dict[str, Any],
    ) -> str:
        """Decide what follows a REPETITION stop - Claude Generated

        ``repetition_action`` (``RepetitionDetectionConfig``):

        * ``retry`` — the same request again with varied sampling
          (``retry_sampling_params``).
        * ``agent`` — the text before the loop goes into the conversation as
          the assistant's turn, followed by a note (role ``user``: a mid-run
          ``system`` message would replace the system prompt in the Anthropic
          and Gemini converters) that the answer was stopped; the model decides
          how to go on and may call ``set_generation_params``.
        * ``abort`` — the run ends with the text before the loop.

        ``retry`` and ``agent`` together get at most ``max_retries`` attempts
        per run; after that the run ends as with ``abort``.
        Returns ``"retry"``, ``"agent"`` or ``"stop"``.
        """
        cfg = self.repetition_config
        info = response.repetition or {}
        where = "Denkkanal" if info.get("channel") == "thinking" else "Antwort"
        state["last"] = info
        action = cfg.repetition_action
        if action in ("retry", "agent") and state["attempts"] < cfg.max_retries:
            state["attempts"] += 1
            n, limit = state["attempts"], cfg.max_retries
            if action == "retry":
                gen.update(retry_sampling_params(gen["temperature"], gen["top_p"], n))
                logger.warning(f"Repetition stop ({where}) → retry {n}/{limit} with {gen}")
                if self._status_cb:
                    self._status_cb(
                        f"\n🔁 Wiederholungsschleife erkannt ({where}): neuer Versuch {n}/{limit} "
                        f"mit temperature={gen['temperature']}, top_p={gen['top_p']}\n"
                    )
                return "retry"
            partial = strip_think_tags(response.content or "").strip()
            if partial:
                messages.append({"role": "assistant", "content": partial})
            messages.append({"role": "user", "content": _repetition_note(info, bool(partial))})
            state["offer_params_tool"] = True
            logger.warning(f"Repetition stop ({where}) → agent decides ({n}/{limit})")
            if self._status_cb:
                self._status_cb(
                    f"\n🔁 Wiederholungsschleife erkannt ({where}): Antwort abgebrochen, "
                    f"der Agent entscheidet über das weitere Vorgehen ({n}/{limit})\n"
                )
            return "agent"
        logger.warning(f"Repetition stop ({where}) → run ends with partial result")
        if self._status_cb:
            self._status_cb(
                f"\n⛔ Wiederholungsschleife erkannt ({where}): Schritt endet mit dem Text "
                f"vor der Schleife ({info.get('kept_chars', 0)} Zeichen behalten, "
                f"{info.get('cut_chars', 0)} abgeschnitten)\n"
            )
        return "stop"

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
        # Sampling of this run; set_generation_params and the repetition
        # "retry" action change it for all following calls. - Claude Generated
        gen: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": None,
        }
        rep_state: Dict[str, Any] = {"attempts": 0, "last": None, "offer_params_tool": False}

        def _current_tools():
            # set_generation_params is offered from the first repetition stop
            # with action "agent" on — also in a run without other tools, so
            # the note's advice can be followed. - Claude Generated
            if not rep_state["offer_params_tool"]:
                return tool_schemas
            schema = SET_GENERATION_PARAMS.to_schema()
            base = list(tool_schemas or [])
            if all(t.get("name") != schema["name"] for t in base):
                base.append(schema)
            return base

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

            # One call, repeated in place when a repetition stop is answered
            # with "retry" or "agent" (no iteration is used up for it). A fresh
            # think filter per call: an aborted stream may end inside an open
            # <think>. - Claude Generated
            call_error: Optional[Exception] = None
            while True:
                # Fresh per-call filter: diverts streamed <think> content to
                # the thinking channel; without on_thinking the stream stays
                # untouched (pipeline paths). - Claude Generated
                think_filter: Optional[ThinkStreamFilter] = None
                stream_cb = self.stream_callback
                if self.on_thinking and self.stream_callback:
                    think_filter = ThinkStreamFilter(self.stream_callback, self.on_thinking)
                    stream_cb = think_filter.feed
                try:
                    response: AgentResponse = self._generate(
                        provider, model, messages, _current_tools(), gen, seed, think,
                        stream_cb, thinking_cb,
                    )
                except Exception as e:
                    if think_filter:
                        think_filter.flush()
                    call_error = e
                    break
                if think_filter:
                    think_filter.flush()
                if response.stop_reason != StopReason.REPETITION:
                    break
                if self._handle_repetition(response, messages, gen, rep_state) == "stop":
                    break

            if call_error is not None:
                e = call_error
                logger.error(f"LLM call failed at tool-call {iteration}: {e}")
                if self._status_cb:
                    self._status_cb(f"\n❌ LLM-Fehler: {e}\n")
                final_content = f"Error: {e}"
                run_error = str(e)  # mark run as failed, not just oddly-worded - Claude Generated
                break

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

            # A provider generator detects should_stop() mid-stream and returns
            # this instead of raising - but nothing here ever checked for it,
            # so a cancel hit before any content/tool_calls existed (streaming
            # not yet started, or a reasoning-only turn) fell through to the
            # "empty turn" handling below exactly like an ordinary empty
            # response: the one-time nudge fired a SECOND, non-streaming LLM
            # call (no stream_callback there either, so should_stop couldn't
            # even interrupt IT promptly) instead of stopping. Any content that
            # did stream already reached the UI via stream_callback per chunk -
            # nothing to re-emit here. - Claude Generated
            if response.stop_reason == StopReason.CANCELLED:
                logger.info(f"Agent loop: LLM call cancelled at iteration {iteration}")
                final_content = response.content or ""
                final_stop_reason = StopReason.CANCELLED
                if self._status_cb:
                    self._status_cb("\n⏹ Agent-Abbruch durch Operator\n")
                if final_content:
                    messages.append({"role": "assistant", "content": strip_think_tags(final_content)})
                break

            # Repetition stop that _handle_repetition did not turn into another
            # attempt: the run ends with the text before the loop. Not an
            # error (that text is the model's), but stop_reason "repetition"
            # tells the caller it is unfinished. - Claude Generated
            if response.stop_reason == StopReason.REPETITION:
                final_content = response.content or ""
                final_stop_reason = StopReason.REPETITION
                if final_content:
                    messages.append({"role": "assistant", "content": strip_think_tags(final_content)})
                break

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

                    # Execute tool. set_generation_params acts on this run's
                    # sampling, so the loop answers it itself instead of the
                    # (shared) registry. - Claude Generated
                    tool_start = time.time()
                    if tc.name == SET_GENERATION_PARAMS.name and rep_state["offer_params_tool"]:
                        result_str = json.dumps(
                            apply_generation_params(gen, tc.arguments), ensure_ascii=False
                        )
                    else:
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
                    forced = self._generate(
                        provider, model, messages, [], gen, seed, think, None, thinking_cb,
                    )
                    if forced.stop_reason == StopReason.CANCELLED:
                        final_stop_reason = StopReason.CANCELLED
                        if self._status_cb:
                            self._status_cb("\n⏹ Agent-Abbruch durch Operator\n")
                        break
                    if forced.stop_reason == StopReason.REPETITION:
                        rep_state["last"] = forced.repetition
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
                        f"⚠️ Das Modell hat das Token-Budget (max_tokens={gen['max_tokens']}) "
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
                cancelled = False
                try:
                    forced = self._generate(
                        provider, model, messages, [],  # No tools = force text response
                        gen, seed, think, None, thinking_cb,
                    )
                    cancelled = forced.stop_reason == StopReason.CANCELLED
                    if forced.stop_reason == StopReason.REPETITION:
                        rep_state["last"] = forced.repetition
                    final_stop_reason = forced.stop_reason
                    final_content = forced.content or getattr(forced, "reasoning", "")
                    if final_content and not cancelled:
                        # Stream the forced answer too — otherwise it is the one
                        # final-content path the frontends never see live. - Claude Generated
                        final_content = self._emit_final(final_content)
                    if final_content:
                        messages.append({"role": "assistant", "content": final_content})
                except Exception:
                    final_content = "Agent reached maximum iterations without conclusion."
                    run_error = final_content
                # Some models (e.g. code models) call tools but never write a
                # final answer → don't end on a silent empty bubble. A cancel
                # is not that case - don't overwrite it with a message blaming
                # the model for something the operator asked to stop. - Claude Generated
                if not final_content and not cancelled:
                    final_content = (
                        "⚠️ Das Modell hat nach mehreren Tool-Aufrufen keine finale "
                        "Textantwort geliefert. Dieses Modell schreibt im Tool-Modus "
                        "oft keinen Abschlusstext — ggf. ein anderes Chat-Modell wählen."
                    )
                    run_error = final_content
                    if self.stream_callback:
                        self.stream_callback(final_content)
                elif cancelled and self._status_cb:
                    self._status_cb("\n⏹ Agent-Abbruch durch Operator\n")

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
            repetition=rep_state["last"],
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


# Limits of set_generation_params (also stated in its tool description).
GENERATION_PARAM_LIMITS: Dict[str, tuple] = {
    "temperature": (0.0, 2.0),
    "top_p": (0.05, 1.0),
    "repetition_penalty": (1.0, 2.0),
    "max_tokens": (256, 32768),
}


def apply_generation_params(gen: Dict[str, Any], arguments: Any) -> Dict[str, Any]:
    """Validate and apply a set_generation_params call to ``gen`` - Claude Generated

    All or nothing: a single invalid value leaves ``gen`` untouched. Numbers
    may arrive as numeric strings (some models quote them); booleans are not
    numbers here. Returns the JSON-able tool result.
    """
    if not isinstance(arguments, dict):
        return {"error": "Argumente müssen ein Objekt sein", "allowed": sorted(GENERATION_PARAM_LIMITS)}
    errors: List[str] = []
    applied: Dict[str, Any] = {}
    unknown = sorted(k for k in arguments if k not in GENERATION_PARAM_LIMITS)
    if unknown:
        errors.append(f"unbekannte Parameter: {', '.join(unknown)}")
    for name, (lo, hi) in GENERATION_PARAM_LIMITS.items():
        if name not in arguments or arguments[name] is None:
            continue
        raw = arguments[name]
        if isinstance(raw, bool):
            errors.append(f"{name}: Zahl erwartet, nicht {raw!r}")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name}: Zahl erwartet, nicht {raw!r}")
            continue
        if value != value or not (lo <= value <= hi):  # NaN or out of range
            errors.append(f"{name}: {raw!r} liegt nicht in {lo}–{hi}")
            continue
        if name == "max_tokens":
            if value != int(value):
                errors.append(f"max_tokens: ganze Zahl erwartet, nicht {raw!r}")
                continue
            applied[name] = int(value)
        else:
            applied[name] = round(value, 3)
    if not errors and not applied:
        errors.append(
            "kein Parameter angegeben (erlaubt: " + ", ".join(sorted(GENERATION_PARAM_LIMITS)) + ")"
        )
    if errors:
        return {"error": "; ".join(errors), "unchanged": dict(gen)}
    gen.update(applied)
    return {
        "ok": True,
        "applied": applied,
        "now": dict(gen),
        "note": "Gilt für die folgenden Modellaufrufe dieses Laufs, nicht dauerhaft.",
    }


def _repetition_note(info: Dict[str, Any], partial_kept: bool) -> str:
    """The note a model gets after its answer was stopped by the guard - Claude Generated"""
    where = "im Denkkanal" if info.get("channel") == "thinking" else "in der Antwort"
    rate = info.get("rate")
    threshold = info.get("threshold")
    measured = (
        f" (Informationsrate {rate:.3f} unter {threshold:.3f})"
        if isinstance(rate, (int, float)) and isinstance(threshold, (int, float))
        else ""
    )
    kept = (
        "Der Text bis zum Beginn der Schleife steht oben als deine letzte Antwort."
        if partial_kept
        else "Vor der Schleife lag kein verwertbarer Antworttext."
    )
    return (
        f"[Systemhinweis] Deine letzte Antwort wurde automatisch abgebrochen: sie geriet "
        f"{where} in eine Wiederholungsschleife{measured}. {kept} Wiederhole dich nicht. "
        "Entscheide selbst, wie es weitergeht: setze die Antwort ohne Wiederholung fort, "
        "formuliere sie knapper neu, oder ändere mit dem Werkzeug set_generation_params "
        "die Sampling-Parameter (temperature, top_p, repetition_penalty, max_tokens) für "
        "die folgenden Aufrufe."
    )
