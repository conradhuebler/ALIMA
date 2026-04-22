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
from src.core.agents.registry import register_step
from src.core.agents.steps.base_step import BaseStep, StepConfig

logger = logging.getLogger(__name__)


# Simple built-in presets. Agents can override via ``tools.explicit``.
TOOL_PRESETS: Dict[str, List[str]] = {
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
        system_prompt = self._render(raw_cfg.get("system_prompt", ""), resolved_inputs)
        user_prompt = self._render(raw_cfg.get("user_prompt", ""), resolved_inputs)
        tool_names = self._resolve_tools(raw_cfg.get("tools"))

        params = self._llm_params(raw_cfg, context)
        if self.stream_callback:
            self.stream_callback(
                f"\n{'='*50}\n🤖 LLMAgent '{self.step_id}' "
                f"({params['provider']}/{params['model']})\n{'='*50}\n"
            )

        result = self._invoke_loop(system_prompt, user_prompt, tool_names, params)
        parsed = _extract_json(result.content)

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

        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        total = len(chunks)

        tool_names = self._resolve_tools(raw_cfg.get("tools"))
        params = self._llm_params(raw_cfg, context)
        merge_key = chunk_cfg.get("merge_key", "keywords")
        dedup_field = chunk_cfg.get("dedup_field", "title")

        if self.stream_callback:
            self.stream_callback(
                f"\n{'='*50}\n🤖 LLMAgent '{self.step_id}' chunked: "
                f"{len(items)} items × {total} chunks × {chunk_size}\n{'='*50}\n"
            )

        merged: List[Dict[str, Any]] = []
        seen: set = set()
        per_chunk: List[Dict[str, Any]] = []
        total_iterations = 0

        for idx, chunk in enumerate(chunks, 1):
            chunk_inputs = dict(resolved_inputs)
            chunk_inputs[chunk_field] = chunk
            chunk_inputs["chunk_index"] = idx
            chunk_inputs["chunk_total"] = total

            system_prompt = self._render(raw_cfg.get("system_prompt", ""), chunk_inputs)
            user_prompt = self._render(raw_cfg.get("user_prompt", ""), chunk_inputs)

            if self.stream_callback:
                self.stream_callback(f"\n▶ Chunk {idx}/{total} ({len(chunk)} items)\n")

            result = self._invoke_loop(system_prompt, user_prompt, tool_names, params)
            parsed = _extract_json(result.content)
            total_iterations += getattr(result, "iterations", 1)
            per_chunk.append({"index": idx, "response": parsed})

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

        response = {merge_key: merged}
        return {
            "response": response,
            "response_text": json.dumps(response, ensure_ascii=False),
            "iterations": total_iterations,
            "tool_log": [],
            "per_chunk": per_chunk,
        }

    # ------------------------------------------------------------------

    def _llm_params(self, raw_cfg: Dict[str, Any], context: Any) -> Dict[str, Any]:
        llm_cfg = raw_cfg.get("llm", {}) or {}
        return {
            "temperature": llm_cfg.get("temperature", getattr(context, "temperature", 0.5)),
            "top_p": llm_cfg.get("top_p", 0.9),
            "max_tokens": llm_cfg.get("max_tokens", getattr(context, "max_tokens", 4096)),
            "max_iterations": int(llm_cfg.get("max_iterations", 20)),
            "timeout_seconds": int(llm_cfg.get("timeout_seconds", 300)),
            "provider": getattr(context, "provider", "") or "",
            "model": getattr(context, "model", "") or "",
        }

    def _invoke_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_names: List[str],
        params: Dict[str, Any],
    ) -> Any:
        loop = AgentLoop(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            max_iterations=params["max_iterations"],
            timeout_seconds=params["timeout_seconds"],
            stream_callback=self.stream_callback,
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

    @staticmethod
    def _resolve_tools(cfg: Any) -> List[str]:
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
                return list(TOOL_PRESETS.get(preset_name, []))
        return []


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json(content: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from an LLM response.

    Mirrors the existing BaseSubAgent._extract_json logic so migrated
    workflows keep working.
    """
    if not content:
        return {}

    m = _JSON_BLOCK_RE.search(content)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                return {"items": obj}
        except json.JSONDecodeError:
            pass

    # Fallback: last balanced JSON object in the text.
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", content, re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            continue

    return {}
