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
from typing import Any, Dict, List, Optional

from src.core.agent_loop import AgentLoop
from src.core.agents.context_path import resolve_mapping
from src.core.agents.prompt_resolver import resolve_prompts
from src.core.agents.registry import register_step
from src.core.agents.steps.base_step import BaseStep, StepConfig

logger = logging.getLogger(__name__)


# Built-in fallback presets when no ToolRegistry is available (e.g. tests).
# In production, presets are loaded from src/mcp/default_presets.yaml.
_TOOL_PRESETS_FALLBACK: Dict[str, List[str]] = {
    "library": ["search_gnd", "search_lobid", "search_swb", "get_search_cache"],
    "gnd": ["search_gnd", "get_gnd_entry", "get_gnd_batch"],
    "classification": ["get_dk_cache", "get_classification", "search_catalog"],
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
        _emit_prompts(self.step_id, system_prompt, user_prompt, params, self.stream_callback)

        result = self._invoke_loop(system_prompt, user_prompt, tool_names, params)
        parsed = _extract_json(result.content)

        _log_response(self.step_id, result.content)
        return {
            "response": parsed,
            "response_text": result.content,
            "iterations": getattr(result, "iterations", 1),
            "tool_log": getattr(result, "tool_log", []),
        }

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

        chunk_size = int(chunk_cfg.get("chunk_size", 350))
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

        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        total = len(chunks)

        tool_names = self._resolve_tools(raw_cfg.get("tools"))
        params = self._llm_params(raw_cfg, context)
        merge_key = chunk_cfg.get("merge_key", "keywords")
        dedup_field = chunk_cfg.get("dedup_field", "title")
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
            self.stream_callback(header)
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
            _emit_prompts(
                f"{self.step_id}[chunk {idx}/{total}]",
                system_prompt, user_prompt, chunk_params, self.stream_callback,
            )

            result = self._invoke_loop(system_prompt, user_prompt, tool_names, chunk_params)
            parsed = _extract_json(result.content)
            total_iterations += getattr(result, "iterations", 1)
            per_chunk.append({"index": idx, "response": parsed})
            _log_response(f"{self.step_id}[chunk {idx}/{total}]", result.content)

            chunk_items = parsed.get(merge_key, []) if isinstance(parsed, dict) else []
            if isinstance(chunk_items, list):
                for it in chunk_items:
                    key: Any
                    if isinstance(it, dict):
                        key = str(it.get(dedup_field, "")).lower()
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

    def _llm_params(self, raw_cfg: Dict[str, Any], context: Any) -> Dict[str, Any]:
        llm_cfg = raw_cfg.get("llm", {}) or {}
        # seed resolution: step llm.seed > settings.seed (via context) > None
        seed_val = llm_cfg.get("seed", getattr(context, "seed", None))
        return {
            "temperature": llm_cfg.get("temperature", getattr(context, "temperature", 0.5)),
            "top_p": llm_cfg.get("top_p", 0.9),
            "max_tokens": llm_cfg.get("max_tokens", getattr(context, "max_tokens", 4096)),
            "max_iterations": int(llm_cfg.get("max_iterations", 20)),
            "timeout_seconds": int(llm_cfg.get("timeout_seconds", 300)),
            "provider": getattr(context, "provider", "") or "",
            "model": getattr(context, "model", "") or "",
            "seed": seed_val,
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
                    pass

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
                    pass
        except Exception:
            _emit_tool_called = None
            _emit_tool_result = None

        loop = AgentLoop(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            max_iterations=params["max_iterations"],
            timeout_seconds=params["timeout_seconds"],
            stream_callback=self.stream_callback,
            on_tool_call=_emit_tool_called,
            on_tool_result=_emit_tool_result,
        )
        return loop.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tool_names if tool_names else None,
            provider=params["provider"],
            model=params["model"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            seed=params.get("seed"),
        )

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
    """Emit the lightweight `🤖 LLMAgent` header unconditionally so the
    operator always sees step + provider + model before the LLM call."""
    header = _format_header(step_id, params)
    if stream_callback:
        stream_callback(header)
    logger.info(header.strip())


def _emit_prompts(
    step_id: str,
    system_prompt: str,
    user_prompt: str,
    params: Dict[str, Any],
    stream_callback: Optional[Any],
) -> None:
    """Stream + log full SYSTEM/USER prompts (verbose mode only — caller gates)."""
    sys_block = f"--- SYSTEM ---\n{system_prompt}\n"
    usr_block = f"--- USER ---\n{user_prompt}\n{'='*50}\n"

    full = sys_block + usr_block
    if stream_callback:
        stream_callback(full)
    logger.info(full)


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
