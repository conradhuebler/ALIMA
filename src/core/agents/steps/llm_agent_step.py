"""LLMAgentStep — generic LLM-driven step for Workflow v4 - Claude Generated.

Drives a single LLM call (with optional tool use via AgentLoop) configured
entirely from YAML.  No per-task Python subclass needed — the same class
powers extraction, selection, classification, ranking, or any future LLM
step, just with different YAML prompts.

YAML fields consumed (see workflows/*.yaml for examples)::

    - id: extraction
      type: llm_agent
      system_prompt: |
          Du bist ein präziser Bibliothekar ...
      user_prompt: |
          Analysiere den Abstract: {abstract}
      inputs:
          abstract: "${abstract}"
          keywords: "${steps.extraction.keywords}"
      outputs:
          extracted_keywords: "response.keywords"
          working_title: "response.title"
      tools:
          explicit: [search_swb, get_gnd_batch]
          preset: library          # OR a preset name; explicit wins
      llm:
          temperature: 0.5
          top_p: 0.9
          max_tokens: 4096
          max_iterations: 20

Placeholder handling:
    * ``${path}`` in the ``inputs:`` values is resolved against the
      SharedContext (see :mod:`src.core.agents.context_path`).
    * Resolved inputs are then substituted into ``{name}`` markers in the
      prompts — matching the existing ALIMA prompt style.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.agent_loop import AgentLoop
from src.core.agents.context_path import resolve_mapping
from src.core.agents.prompt_resolver import resolve_prompts
from src.core.agents.registry import register_step
from src.core.agents.steps.base_step import BaseStep, StepConfig
# Shared keyword-chunking split (classic-pipeline semantics) — single source of
# truth lives in src/utils/chunking.py; kept under the historic private name so
# existing imports/tests keep working. - Claude Generated
from src.utils.chunking import split_into_equal_chunks as _split_chunks_classic

logger = logging.getLogger(__name__)


# Built-in fallback presets when no ToolRegistry is available (e.g. tests).
# In production, presets are loaded from src/mcp/default_presets.yaml.
_TOOL_PRESETS_FALLBACK: Dict[str, List[str]] = {
    "library": ["search_gnd", "search_lobid", "search_swb", "get_search_cache"],
    "gnd": ["search_gnd", "get_gnd_entry", "get_gnd_batch"],
    "classification": [
        "get_dk_cache", "get_classification", "search_catalog",
        "rvk_search", "rvk_validate",
    ],
    "lookup": ["rvk_search", "rvk_validate", "k10plus_package", "dnb_classification"],
    "none": [],
}


@register_step("llm_agent")
class LLMAgentStep(BaseStep):
    """Configurable LLM step, YAML-driven.

    Supports optional chunking: when ``chunking.enabled`` is true in the
    step YAML, an input list is sliced into fixed-size chunks and the LLM
    is invoked once per chunk. Results are merged into a single response.

    Chunking YAML::

        chunking:
          enabled: true
          chunk_size: 350
          chunk_field: gnd_entries       # key in resolved_inputs (list)
          sort_by: count                 # optional, sort items desc before chunking
          merge_key: keywords            # response[merge_key] must be a list
          dedup_field: title             # dedupe merged items by this key (lower)
    """

    def run(self, context: Any) -> Dict[str, Any]:
        raw_cfg = self.config.raw or {}

        resolved_inputs = resolve_mapping(self.config.inputs, context)

        chunk_cfg = raw_cfg.get("chunking") or {}
        if chunk_cfg.get("enabled") and chunk_cfg.get("chunk_field"):
            return self._run_chunked(raw_cfg, resolved_inputs, chunk_cfg, context)
        return self._run_single(raw_cfg, resolved_inputs, context)

    # ------------------------------------------------------------------

    def _run_single(
        self,
        raw_cfg: Dict[str, Any],
        resolved_inputs: Dict[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        tool_names = self._resolve_tools(raw_cfg.get("tools"))
        params = self._llm_params(raw_cfg, context)
        system_prompt, user_prompt, params = self._resolve_prompts(
            raw_cfg, resolved_inputs, context, params
        )

        _emit_header(self.step_id, params, self.stream_callback)
        _prompt_id = uuid.uuid4().hex[:8]
        _emit_prompts(
            self.step_id, system_prompt, user_prompt, params,
            self.stream_callback, prompt_id=_prompt_id,
        )

        _t0 = time.monotonic()
        result = self._invoke_loop(system_prompt, user_prompt, tool_names, params)
        _emit_prompt_done(_prompt_id, self.step_id, time.monotonic() - _t0)
        if getattr(result, "error", None):
            # LLM hard failure — fail the step instead of parsing the error
            # string as if it were a model answer - Claude Generated
            raise RuntimeError(f"LLM call failed: {result.error}")

        # Log the raw response BEFORE parsing/salvage/retry/required, so that
        # if `required: true` raises below, the actual text that failed to
        # parse is still visible in the log/diagnostics instead of vanishing
        # along with the exception — otherwise a hard-fail is exactly as
        # opaque as the silent failure it replaces. - Claude Generated
        _log_response(self.step_id, result.content)

        salvage_cfg = raw_cfg.get("salvage")
        parsed, result = self._parse_with_salvage_and_retry(
            result, salvage_cfg, system_prompt, user_prompt, tool_names, params,
        )
        self._announce_field_count(salvage_cfg, parsed)

        return {
            "response": parsed,
            "response_text": result.content,
            "iterations": getattr(result, "iterations", 1),
            "tool_log": getattr(result, "tool_log", []),
        }

    def _parse_with_salvage_and_retry(
        self,
        result: Any,
        salvage_cfg: Optional[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
        tool_names: List[str],
        params: Dict[str, Any],
    ) -> tuple:
        """Parse the LLM response as JSON, apply salvage, and — per the step's
        YAML ``salvage:`` block — optionally retry once with a stricter prompt
        and/or hard-fail if the declared field is still empty.

        Every layer downstream (``BaseStep._write``, ``context_path.resolve_value``,
        deterministic step functions treating ``None``/``[]`` as "nothing to
        search") currently swallows a JSON-parse failure silently, so a step
        whose output is load-bearing for the rest of the workflow should
        declare ``required: true`` here rather than let empty data propagate
        unnoticed. Returns ``(parsed, result)`` — ``result`` is the retry's
        ``AgentResult`` if a retry happened and produced usable data, else the
        original. - Claude Generated
        """
        parsed = _extract_json(result.content)
        parsed = self._salvage_and_warn(parsed, result.content, salvage_cfg)
        field = salvage_cfg.get("field") if salvage_cfg else None
        # Presence, not truthiness: {"catalog_hits": []} is a legitimate,
        # well-formed "found nothing" answer (e.g. no wishlist title exists
        # in the catalog) and must NOT be treated the same as parsing failing
        # outright ({} with the key missing entirely). Conflating the two
        # would hard-fail a step for a perfectly valid empty result. - Claude Generated
        got_field = bool(field) and isinstance(parsed, dict) and field in parsed

        # Checked regardless of whether retry is configured — a step with
        # `required: true` but no `retry` (e.g. the search step's
        # catalog_hits) deserves the same truncation-vs-prose diagnosis
        # before it hard-fails below. - Claude Generated
        if salvage_cfg and not got_field:
            self._warn_if_max_tokens_truncated(self.step_id, parsed, result)

        if salvage_cfg and salvage_cfg.get("retry") and not got_field:
            retry_system = system_prompt + (
                "\n\nACHTUNG: Deine letzte Antwort enthielt kein gültiges JSON "
                f"für das Feld '{field}'. Antworte dieses Mal AUSSCHLIESSLICH "
                "mit dem angeforderten JSON-Objekt — kein Fließtext, kein "
                "Markdown-Codeblock, keine Erklärung davor oder danach."
            )
            if self.stream_callback:
                self.stream_callback(
                    f"🔁 '{self.step_id}': JSON-Schema verfehlt, Retry mit verschärftem Prompt …\n"
                )
            retry_result = self._invoke_loop(retry_system, user_prompt, tool_names, params)
            if not getattr(retry_result, "error", None):
                _log_response(f"{self.step_id}[retry]", retry_result.content)
                retry_parsed = _extract_json(retry_result.content)
                retry_parsed = self._salvage_and_warn(retry_parsed, retry_result.content, salvage_cfg)
                retry_got_field = (
                    bool(field) and isinstance(retry_parsed, dict) and field in retry_parsed
                )
                if retry_got_field:
                    parsed, result, got_field = retry_parsed, retry_result, True
                else:
                    # The retry itself can ALSO be cut off by max_tokens — a
                    # real observed case: the model produced a correctly-keyed
                    # {"titles": [...]} on retry (visible in the log, well
                    # past the point where prose-vs-JSON was the issue) but
                    # still failed the required-check because the response
                    # was truncated mid-array. Without this check that case
                    # silently collapsed into the same generic "still empty"
                    # error as a genuine formatting failure. - Claude Generated
                    self._warn_if_max_tokens_truncated(
                        f"{self.step_id}[retry]", retry_parsed, retry_result
                    )

        if salvage_cfg and salvage_cfg.get("required") and not got_field:
            raise RuntimeError(
                f"LLMAgentStep '{self.step_id}': required field '{field}' is still "
                f"empty after JSON parsing"
                + (" + salvage" if salvage_cfg.get("type") else "")
                + (" + retry" if salvage_cfg.get("retry") else "")
                + " — aborting instead of continuing with empty data"
            )
        return parsed, result

    def _warn_if_max_tokens_truncated(
        self, label: str, parsed: Dict[str, Any], result: Any
    ) -> None:
        """Distinguish "model answered in prose" from "model was cut off
        mid-JSON by max_tokens" — both collapse into the same empty
        ``parsed``, but operators need to fix different things (raise
        ``max_tokens`` vs. fix the prompt/model) depending on which
        occurred. Called for both the original attempt and (if it also
        failed) the retry attempt — a retry can be truncated too, and
        without checking there it silently looked like a generic parsing
        failure even when the raw response showed a correctly-keyed JSON
        object cut off partway through. - Claude Generated
        """
        if parsed or getattr(result, "stop_reason", "") != "max_tokens":
            return
        msg = (
            f"⚠️ '{label}': Antwort durch max_tokens abgeschnitten — "
            f"JSON unvollständig. max_tokens erhöhen oder chunking aktivieren.\n"
        )
        if self.stream_callback:
            self.stream_callback(msg)
        logger.warning(msg.strip())

    def _announce_field_count(
        self, salvage_cfg: Optional[Dict[str, Any]], parsed: Dict[str, Any]
    ) -> None:
        """Stream a one-line "✅ field: N entries" summary for any step that
        declares ``salvage.field`` and produced a list — cheap, generic
        transparency (e.g. "28 titles extracted", "24 catalog_hits found")
        visible right when that step finishes, instead of only discoverable
        at the very end via the final report or by reading raw JSON. Fires
        for every step with a declared salvage field, not just this
        workflow's. - Claude Generated
        """
        if not salvage_cfg or not self.stream_callback:
            return
        field = salvage_cfg.get("field")
        if not field or not isinstance(parsed, dict):
            return
        value = parsed.get(field)
        if isinstance(value, list):
            self.stream_callback(f"✅ '{self.step_id}': {len(value)} '{field}'-Eintrag(e)\n")

    def _salvage_and_warn(
        self,
        parsed: Dict[str, Any],
        raw_text: str,
        salvage_cfg: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply ``_apply_salvage`` and stream/log a notice when it actually
        recovered something (never fires on well-formed JSON). - Claude Generated
        """
        if not salvage_cfg:
            return parsed
        field = salvage_cfg.get("field")
        before = parsed.get(field) if isinstance(parsed, dict) else None
        parsed = _apply_salvage(parsed, raw_text, salvage_cfg)
        if not before and isinstance(parsed, dict) and parsed.get(field):
            n = len(parsed[field])
            if self.stream_callback:
                self.stream_callback(
                    f"🛟 Salvage: {n} '{field}' aus Rohtext gerettet "
                    f"(JSON-Schema vom Modell verfehlt)\n"
                )
            logger.warning(
                f"LLMAgentStep '{self.step_id}': salvaged {n} '{field}' "
                f"entries from non-JSON output"
            )
        return parsed

    def _run_chunked(
        self,
        raw_cfg: Dict[str, Any],
        resolved_inputs: Dict[str, Any],
        chunk_cfg: Dict[str, Any],
        context: Any,
    ) -> Dict[str, Any]:
        chunk_field = chunk_cfg["chunk_field"]
        items = resolved_inputs.get(chunk_field) or []
        if not isinstance(items, list):
            logger.warning(
                f"LLMAgentStep '{self.step_id}': chunk_field '{chunk_field}' is not a list; "
                "falling back to single-shot"
            )
            return self._run_single(raw_cfg, resolved_inputs, context)

        tool_names = self._resolve_tools(raw_cfg.get("tools"))
        params = self._llm_params(raw_cfg, context)

        # Classic-pipeline parity: chunk_size <= 0 / missing → auto-detect per
        # model via model_capabilities (same source as the rigid pipeline's
        # keyword_chunking_threshold; default 500) - Claude Generated
        chunk_size = int(chunk_cfg.get("chunk_size", 0) or 0)
        if chunk_size <= 0:
            chunk_size = self._auto_chunk_size(params)

        sort_by = chunk_cfg.get("sort_by")
        if sort_by:
            items = sorted(
                items,
                key=lambda e: (e or {}).get(sort_by, 0) if isinstance(e, dict) else 0,
                reverse=bool(chunk_cfg.get("sort_desc", True)),
            )

        # Project each dict to specified fields only — reduces LLM input size.
        # Done AFTER sorting so sort_by field is still available during sort.
        chunk_fields = chunk_cfg.get("chunk_fields")
        if chunk_fields and isinstance(chunk_fields, list):
            items = [
                {k: item[k] for k in chunk_fields if k in item}
                if isinstance(item, dict) else item
                for item in items
            ]

        chunks = _split_chunks_classic(items, chunk_size)
        total = len(chunks)
        merge_key = chunk_cfg.get("merge_key", "keywords")
        dedup_field = chunk_cfg.get("dedup_field", "title")
        salvage_cfg = raw_cfg.get("salvage")
        max_merged = chunk_cfg.get("max_merged")  # optional cap on merged output size

        provider = params.get("provider", "")
        model = params.get("model", "")
        temp = params.get("temperature", "?")
        header = (
            f"\n{'='*50}\n🤖 LLMAgent '{self.step_id}' "
            f"({provider}/{model}  temp={temp}) chunked: "
            f"{len(items)} items × {total} chunks × {chunk_size}\n{'='*50}\n"
        )
        if self.stream_callback:
            # Compact one-liner (full banner stays in the log). - Claude Generated
            self.stream_callback(
                f"🤖 {self.step_id} · {provider}/{model} · "
                f"{len(items)} Items in {total} Chunks\n"
            )
        logger.info(header.strip())

        merged: List[Dict[str, Any]] = []
        seen: set = set()
        per_chunk: List[Dict[str, Any]] = []
        total_iterations = 0

        for idx, chunk in enumerate(chunks, 1):
            chunk_inputs = dict(resolved_inputs)
            chunk_inputs[chunk_field] = chunk
            chunk_inputs["chunk_index"] = idx
            chunk_inputs["chunk_total"] = total

            system_prompt, user_prompt, chunk_params = self._resolve_prompts(
                raw_cfg, chunk_inputs, context, params
            )

            chunk_header = f"\n▶ Chunk {idx}/{total} ({len(chunk)} items)\n"
            if self.stream_callback:
                self.stream_callback(chunk_header)
            _chunk_prompt_id = uuid.uuid4().hex[:8]
            _emit_prompts(
                f"{self.step_id}[chunk {idx}/{total}]",
                system_prompt, user_prompt, chunk_params, self.stream_callback,
                prompt_id=_chunk_prompt_id,
            )

            _t0 = time.monotonic()
            result = self._invoke_loop(system_prompt, user_prompt, tool_names, chunk_params)
            _emit_prompt_done(
                _chunk_prompt_id, f"{self.step_id}[chunk {idx}/{total}]",
                time.monotonic() - _t0,
            )
            if getattr(result, "error", None):
                # LLM hard failure — fail the whole step (a missing chunk would
                # silently drop keywords) - Claude Generated
                raise RuntimeError(
                    f"LLM call failed in chunk {idx}/{total}: {result.error}"
                )
            parsed = _extract_json(result.content)
            parsed = self._salvage_and_warn(parsed, result.content, salvage_cfg)
            total_iterations += getattr(result, "iterations", 1)
            per_chunk.append({"index": idx, "response": parsed})
            _log_response(f"{self.step_id}[chunk {idx}/{total}]", result.content)
            self._warn_if_max_tokens_truncated(f"{self.step_id}[chunk {idx}/{total}]", parsed, result)

            chunk_items = parsed.get(merge_key, []) if isinstance(parsed, dict) else []
            # Presence, not truthiness — a chunk that legitimately analyzed
            # its items as "no matches for any of these" is a valid empty
            # list, distinct from JSON parsing failing outright (merge_key
            # missing entirely). See _parse_with_salvage_and_retry. - Claude Generated
            chunk_has_key = isinstance(parsed, dict) and merge_key in parsed
            if salvage_cfg and salvage_cfg.get("required") and not chunk_has_key:
                # Same silent-failure class as the single-shot path: a chunk
                # whose JSON parsing failed would otherwise just contribute 0
                # items with no trace, quietly dropping those titles/keywords
                # from the merged result. - Claude Generated
                raise RuntimeError(
                    f"LLMAgentStep '{self.step_id}': chunk {idx}/{total} produced "
                    f"no '{merge_key}' items after JSON parsing"
                    + (" + salvage" if salvage_cfg.get("type") else "")
                    + " — aborting instead of silently dropping this chunk's items"
                )
            if isinstance(chunk_items, list):
                for it in chunk_items:
                    key: Any
                    if isinstance(it, dict):
                        # The LLM frequently echoes the projected input field name
                        # (``title``) instead of the requested ``keyword``. Fall back
                        # across both so dedup is never silently disabled, and — when
                        # this is a keyword selection (dedup_field == "keyword") —
                        # canonicalise to ``keyword`` so downstream display/merge stay
                        # consistent (copy, never mutate the parsed response). - Claude Generated
                        key = str(
                            it.get(dedup_field) or it.get("keyword") or it.get("title") or ""
                        ).lower()
                        if (
                            dedup_field == "keyword"
                            and not it.get("keyword")
                            and it.get("title")
                        ):
                            it = {**it, "keyword": it["title"]}
                    else:
                        key = str(it).lower()
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    merged.append(it)

        if max_merged and len(merged) > max_merged:
            if self.stream_callback:
                self.stream_callback(
                    f"  ✂️ merged {len(merged)} → {max_merged} (max_merged limit)\n"
                )
            merged = merged[:max_merged]

        response = {merge_key: merged}
        if self.stream_callback:
            self.stream_callback(f"✅ '{self.step_id}': {len(merged)} '{merge_key}'-Eintrag(e) (gesamt)\n")
        return {
            "response": response,
            "response_text": json.dumps(response, ensure_ascii=False),
            "iterations": total_iterations,
            "tool_log": [],
            "per_chunk": per_chunk,
        }

    # ------------------------------------------------------------------

    def _resolve_prompts(
        self,
        raw_cfg: Dict[str, Any],
        resolved_inputs: Dict[str, Any],
        context: Any,
        params: Dict[str, Any],
    ) -> tuple:
        """Return (system_prompt, user_prompt, params).

        Uses centralized prompt_resolver for consistent override hierarchy:
        inline YAML → prompt_task → workflow-level prompts → defaults.
        """
        workflow_prompts = {}
        # Try to get workflow-level prompts from the execution context
        # (WorkflowExecutor could inject this, or we look it up via meta)
        if hasattr(context, "_workflow_prompts"):
            workflow_prompts = context._workflow_prompts or {}

        system, user, llm_override = resolve_prompts(
            raw_cfg=raw_cfg,
            resolved_inputs=resolved_inputs,
            context=context,
            default_system=raw_cfg.get("system_prompt", ""),
            default_user=raw_cfg.get("user_prompt", ""),
            workflow_prompts=workflow_prompts,
        )

        if llm_override and llm_override.get("temperature") is not None:
            updated = dict(params)
            updated["temperature"] = llm_override["temperature"]
            if llm_override.get("top_p") is not None:
                updated["top_p"] = llm_override["top_p"]
            return system, user, updated

        return system, user, params

    def _auto_chunk_size(self, params: Dict[str, Any]) -> int:
        """Resolve the chunk size like the classic pipeline - Claude Generated

        Delegates to ``model_capabilities.get_chunking_threshold`` (per-model
        config > pattern match > default 500) so the same model chunks
        identically in both pipeline modes.
        """
        provider = params.get("provider", "") or ""
        model = params.get("model", "") or ""
        config_manager = None
        try:
            from src.utils.config_manager import ConfigManager
            config_manager = ConfigManager()
        except Exception:
            pass
        try:
            from src.utils.model_capabilities import get_chunking_threshold
            size = int(get_chunking_threshold(provider, model, config_manager=config_manager) or 500)
        except Exception as exc:
            logger.warning(
                f"LLMAgentStep '{self.step_id}': chunk-size auto-detect failed: {exc} — using 500"
            )
            return 500
        logger.info(
            f"LLMAgentStep '{self.step_id}': auto chunk size {size} for {provider}:{model}"
        )
        return size

    def _llm_params(self, raw_cfg: Dict[str, Any], context: Any) -> Dict[str, Any]:
        llm_cfg = raw_cfg.get("llm", {}) or {}
        # seed resolution: step llm.seed > settings.seed (via context) > None
        seed_val = llm_cfg.get("seed", getattr(context, "seed", None))
        return {
            "temperature": llm_cfg.get("temperature", getattr(context, "temperature", 0.5)),
            "top_p": llm_cfg.get("top_p", 0.9),
            # Budget: operator override > step llm.max_tokens > context default.
            # The override is the one place that outranks the YAML on purpose —
            # without it the budget can only be changed by editing every step of
            # both v5.1 workflows. - Claude Generated
            "max_tokens": (
                getattr(context, "max_tokens_override", None)
                or llm_cfg.get("max_tokens", getattr(context, "max_tokens", 4096))
            ),
            # Thinking control: step llm.think > context.think > provider default.
            # A reasoning model spends its max_tokens budget on the thinking
            # channel first, so this is the lever that decides whether the
            # answer still fits into it. - Claude Generated
            "think": llm_cfg.get("think", getattr(context, "think", None)),
            "max_iterations": int(llm_cfg.get("max_iterations", 20)),
            "timeout_seconds": int(llm_cfg.get("timeout_seconds", 300)),
            "provider": getattr(context, "provider", "") or "",
            "model": getattr(context, "model", "") or "",
            "seed": seed_val,
            # AgentLoop's own default (3) only blocks on the 3rd identical
            # tool call — a 2nd identical call still executes. Configurable
            # per-step since a step with expensive/slow tools (e.g. a
            # sequential-per-term SOAP search) benefits from blocking on the
            # very first repeat, while a step that legitimately needs a few
            # retries with the same args (rare) can raise it back up. - Claude Generated
            "repeat_threshold": int(llm_cfg.get("repeat_threshold", 3)),
        }

    def _invoke_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_names: List[str],
        params: Dict[str, Any],
    ) -> Any:
        # P-δ.5a: route AgentLoop tool events through AlimaStateBus so the
        # PipelineChatPanel (and any future subscriber) can render them in
        # the unified log. No signature change on PipelineManager /
        # WorkflowExecutor required.
        try:
            from src.core.state_bus import AlimaStateBus
            from src.core.agents.sub_agents.caching_tool_registry import (
                make_tool_call_id,
            )
            _bus = AlimaStateBus()

            # P-A: closure cell carries the id of the in-flight tool call so
            # the result event can re-use the same id even though
            # ``on_tool_result(name, result_str)`` does not pass it explicitly.
            _id_cell: list = [None]

            def _emit_tool_called(tc):
                try:
                    _id_cell[0] = (
                        getattr(tc, "id", "") or make_tool_call_id()
                    )
                    _bus.emit_event(
                        "tool.called",
                        {
                            "name": getattr(tc, "name", ""),
                            "arguments": dict(getattr(tc, "arguments", {}) or {}),
                            "id": _id_cell[0],
                        },
                    )
                except Exception:
                    logger.debug("tool.called bus emit failed", exc_info=True)

            def _emit_tool_result(name, result):
                try:
                    _bus.emit_event(
                        "tool.result",
                        {
                            "name": name,
                            "result": result or "",
                            "id": _id_cell[0] or "",
                        },
                    )
                except Exception:
                    logger.debug("tool.result bus emit failed", exc_info=True)

            # Reasoning channel → same 💭 block the chat shows. Without an
            # on_thinking sink the AgentLoop leaves the stream untouched, which
            # is why the agentic run showed no thinking at all (and let an
            # inline <think> dialect run into the pipeline text). - Claude Generated
            def _emit_thinking(text):
                try:
                    _bus.emit_event(
                        "llm.thinking", {"text": text or "", "step_id": self.step_id}
                    )
                except Exception:
                    logger.debug("llm.thinking bus emit failed", exc_info=True)

            def _emit_thinking_done():
                try:
                    _bus.emit_event("llm.thinking_done", {"step_id": self.step_id})
                except Exception:
                    logger.debug("llm.thinking_done bus emit failed", exc_info=True)
        except Exception:
            _emit_tool_called = None
            _emit_tool_result = None
            _emit_thinking = None
            _emit_thinking_done = None

        loop = AgentLoop(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            max_iterations=params["max_iterations"],
            timeout_seconds=params["timeout_seconds"],
            stream_callback=self.stream_callback,
            on_tool_call=_emit_tool_called,
            on_tool_result=_emit_tool_result,
            repeat_threshold=params.get("repeat_threshold", 3),
            on_thinking=_emit_thinking,
        )
        result = loop.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tool_names if tool_names else None,
            provider=params["provider"],
            model=params["model"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            seed=params.get("seed"),
            think=params.get("think"),
        )
        # Fold the block away once the step is done. The renderer also closes it
        # on the next answer token or tool call; a step whose last output was
        # reasoning would otherwise leave it standing open. - Claude Generated
        if _emit_thinking_done is not None:
            _emit_thinking_done()
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _render(template: str, values: Dict[str, Any]) -> str:
        """Replace ``{name}`` markers with stringified values.

        Non-string values become JSON (dicts/lists) or ``str()`` otherwise.
        Missing markers pass through unchanged so prompts that use literal
        braces in examples don't break.
        """
        if not template:
            return ""

        def _stringify(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return str(v)

        out = template
        # Replace the longest names first to avoid partial overlaps (e.g. {k} vs {key}).
        for name in sorted(values.keys(), key=len, reverse=True):
            out = out.replace("{" + name + "}", _stringify(values[name]))
        return out

    def _resolve_tools(self, cfg: Any) -> List[str]:
        """Turn the ``tools:`` block into a flat list of tool names."""
        if cfg is None:
            return []
        if isinstance(cfg, list):
            return list(cfg)
        if isinstance(cfg, dict):
            explicit = cfg.get("explicit") or []
            if explicit:
                return list(explicit)
            preset_name = cfg.get("preset")
            if preset_name:
                # Try registry first (production), fall back to built-in dict (tests)
                if self.tool_registry is not None and hasattr(self.tool_registry, "get_preset"):
                    preset_tools = self.tool_registry.get_preset(preset_name)
                    if preset_tools:
                        return list(preset_tools)
                return list(_TOOL_PRESETS_FALLBACK.get(preset_name, []))
        return []


def _format_header(step_id: str, params: Dict[str, Any]) -> str:
    """Build the one-line `🤖 LLMAgent` header. Used by both _emit_header (always)
    and _emit_prompts (verbose-only)."""
    provider = params.get("provider", "") or "?"
    model = params.get("model", "") or "?"
    temp = params.get("temperature", "?")
    top_p = params.get("top_p", "?")
    return (
        f"\n{'='*50}\n🤖 LLMAgent '{step_id}' "
        f"({provider}/{model}  temp={temp}  top_p={top_p})\n{'='*50}\n"
    )


def _emit_header(
    step_id: str,
    params: Dict[str, Any],
    stream_callback: Optional[Any],
) -> None:
    """Emit a compact one-line step marker before the LLM call.

    The provider/model/params also appear in the collapsible 📥 Input block, so
    the old multi-line ``====`` banner was redundant noise in the GUI log. The
    full banner is still written to the logger for CLI/file diagnostics. - Claude Generated
    """
    provider = params.get("provider", "") or "?"
    model = params.get("model", "") or "?"
    one_liner = (
        f"\n🤖 {step_id} · {provider}/{model} "
        f"(temp={params.get('temperature', '?')}, top_p={params.get('top_p', '?')})\n"
    )
    if stream_callback:
        stream_callback(one_liner)
    logger.info(_format_header(step_id, params).strip())


def _emit_prompts(
    step_id: str,
    system_prompt: str,
    user_prompt: str,
    params: Dict[str, Any],
    stream_callback: Optional[Any],
    prompt_id: Optional[str] = None,
    kind: str = "input",
) -> None:
    """Log the full SYSTEM/USER prompt + publish it as a structured event.

    The prompt is intentionally NOT streamed to ``stream_callback`` anymore: in
    the GUI it duplicated (and spammed the log next to) the collapsible 📥 Input
    block, and in non-verbose mode the line-buffering filter fragmented it so it
    could not be suppressed reliably. CLI/file diagnostics keep it via
    ``logger.info``; the GUI renders it as a collapsible via the bus event. - Claude Generated
    """
    sys_block = f"--- SYSTEM ---\n{system_prompt}\n"
    usr_block = f"--- USER ---\n{user_prompt}\n{'='*50}\n"
    logger.info(sys_block + usr_block)

    # Publish the prompt as a structured event so the GUI can render it as a
    # collapsible, timestamped block. No-op without bus subscribers.
    if prompt_id:
        try:
            from datetime import datetime as _dt
            from src.core.state_bus import AlimaStateBus

            AlimaStateBus().emit_event(
                "state.pipeline_prompt",
                {
                    "prompt_id": prompt_id,
                    "step_id": step_id,
                    "kind": kind,
                    "system": system_prompt,
                    "user": user_prompt,
                    "provider": params.get("provider", ""),
                    "model": params.get("model", ""),
                    "timestamp": _dt.now().isoformat(timespec="seconds"),
                },
            )
        except Exception:
            logger.debug("pipeline_prompt bus emit failed", exc_info=True)


def _emit_prompt_done(prompt_id: Optional[str], step_id: str, duration_s: float) -> None:
    """Publish the LLM-call duration for a prompt block (paired with _emit_prompts). - Claude Generated"""
    if not prompt_id:
        return
    try:
        from src.core.state_bus import AlimaStateBus

        AlimaStateBus().emit_event(
            "state.pipeline_prompt_done",
            {"prompt_id": prompt_id, "step_id": step_id, "duration_s": float(duration_s)},
        )
    except Exception:
        logger.debug("pipeline_prompt_done bus emit failed", exc_info=True)


def _log_response(step_id: str, content: str) -> None:
    """Log LLM response to logger so it appears in the console independently of stream_callback."""
    if not content:
        logger.info(f"[{step_id}] LLM response: (empty)")
        return
    preview = content[:600]
    if len(content) > 600:
        preview += f"\n... ({len(content)} chars total)"
    logger.info(f"[{step_id}] LLM response:\n{preview}")


_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL
)
_SOLUTION_RE = re.compile(
    r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL
)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def _first_balanced_object(text: str) -> Optional[str]:
    """Return the first balanced top-level ``{...}`` substring in ``text``.

    Brace-counting parser that respects string literals + backslash escapes.
    Returns ``None`` if no balanced object is found.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if in_str:
            if c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# Salvage classification codes from non-JSON output. Some models
# (mistral/ministral) emit the DK codes as plain lines instead of the requested
# {"classifications": [...]} schema, which leaves the DK table empty. - Claude Generated
_SALVAGE_DK_RE = re.compile(r"\b(DK|DDC)\s+(\d[\d.\-/;:]*)", re.IGNORECASE)
_SALVAGE_RVK_RE = re.compile(r"\bRVK\s+([A-Z]{2}\s?\d[\d.,/\-]*)", re.IGNORECASE)


def _salvage_codes(raw_text: str) -> List[Dict[str, str]]:
    """Recover ``[{"code": "DK 504.064", "type": "DK"}, …]`` from raw text.

    Order-preserving, deduplicated. Returns ``[]`` when nothing matches.
    """
    out: List[Dict[str, str]] = []
    seen: set = set()
    for m in _SALVAGE_DK_RE.finditer(raw_text or ""):
        typ = m.group(1).upper()
        code = f"{typ} {m.group(2).strip().rstrip('.;:-/')}"
        if code not in seen:
            seen.add(code)
            out.append({"code": code, "type": typ})
    for m in _SALVAGE_RVK_RE.finditer(raw_text or ""):
        code = "RVK " + re.sub(r"\s+", " ", m.group(1).strip())
        if code not in seen:
            seen.add(code)
            out.append({"code": code, "type": "RVK"})
    return out


# Salvage title/publisher-or-isbn/year triples from non-JSON output. Observed
# real failure mode for extract_titles-style steps: the model ignores the
# JSON-only instruction and instead answers with a plain repeated
# "Title\nPublisher\nYear" listing (no braces at all, so _extract_json finds
# nothing). Verified against an actual failing run's raw output: 28/28 titles
# recovered correctly, including entries where the "publisher" line is
# actually an ISBN. - Claude Generated
_YEAR_LINE_RE = re.compile(r"^(19|20)\d{2}$")
_ISBN_LINE_RE = re.compile(r"^97[89][\d\-]{10,17}$")


def _salvage_title_triples(raw_text: str) -> List[Dict[str, Any]]:
    """Recover ``[{"title", "publisher", "isbn", "year", "authors": []}, …]``
    from a plain title/publisher-or-isbn/year line-triple listing.

    Best-effort heuristic, not a general-purpose parser: a line-triple only
    matches when the third line is a bare 4-digit year (1500-2099) and
    neither of the first two lines is itself a bare year. ``[]`` if nothing
    matches — never partially guesses.
    """
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    out: List[Dict[str, Any]] = []
    i = 0
    while i + 2 < len(lines):
        title, mid, year_line = lines[i], lines[i + 1], lines[i + 2]
        if (
            _YEAR_LINE_RE.match(year_line)
            and not _YEAR_LINE_RE.match(title)
            and not _YEAR_LINE_RE.match(mid)
        ):
            entry: Dict[str, Any] = {"title": title, "authors": [], "year": year_line}
            if _ISBN_LINE_RE.match(mid.replace(" ", "")):
                entry["isbn"] = mid
                entry["publisher"] = ""
            else:
                entry["isbn"] = ""
                entry["publisher"] = mid
            out.append(entry)
            i += 3
        else:
            i += 1
    return out


_SALVAGE_TYPES = {
    "codes": _salvage_codes,
    "title_triples": _salvage_title_triples,
}


def _apply_salvage(
    parsed: Dict[str, Any],
    raw_text: str,
    salvage_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recover a structured field from raw text when JSON extraction missed it.

    Opt-in via the step's YAML ``salvage: {field: ..., type: codes|title_triples}``
    block. Only fires when ``parsed[field]`` is empty, so well-formed JSON is
    never overwritten. ``type`` must be explicit — a ``salvage:`` block with
    only ``required``/``retry`` (no ``type``) performs no text-recovery, just
    the empty-field check/retry those keys control. - Claude Generated
    """
    if not salvage_cfg or not isinstance(salvage_cfg, dict):
        return parsed
    field = salvage_cfg.get("field")
    if not field:
        return parsed
    if isinstance(parsed, dict) and parsed.get(field):
        return parsed  # JSON parsing already produced it
    salvage_fn = _SALVAGE_TYPES.get(salvage_cfg.get("type"))
    if salvage_fn:
        recovered = salvage_fn(raw_text)
        if recovered:
            base = dict(parsed) if isinstance(parsed, dict) else {}
            base[field] = recovered
            return base
    return parsed


def _extract_json(content: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from an LLM response.

    Pipeline:
        1. Strip ``<|begin_of_thought|>…<|end_of_thought|>`` blocks.
        2. If ``<|begin_of_solution|>…<|end_of_solution|>`` present, narrow to
           that payload.
        3. If a markdown fence (``` ```json ... ``` ```) is present, narrow to
           the fence body.
        4. Balanced-brace extraction of the first top-level ``{...}``.

    Empty input or unparseable content returns ``{}``.
    """
    if not content:
        return {}

    text = _THOUGHT_RE.sub("", content)

    sol = _SOLUTION_RE.search(text)
    if sol:
        text = sol.group(1)

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)

    stripped = text.lstrip()
    if stripped.startswith("["):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, list):
                return {"items": obj}
        except json.JSONDecodeError:
            pass

    chunk = _first_balanced_object(text)
    if chunk:
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                return {"items": obj}
        except json.JSONDecodeError:
            pass

    return {}
