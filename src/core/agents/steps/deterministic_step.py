"""DeterministicStep — YAML-configured Python function call - Claude Generated.

Used for pipeline stages that are fixed Python logic (e.g. batch GND search,
local DB enrichment, DOI resolution) rather than LLM reasoning.  The function
is registered with :func:`src.core.agents.registry.register_tool_fn` and
referenced by name in the workflow YAML::

    - id: search
      type: deterministic
      function: gnd_batch_search
      inputs:
          keywords: "${steps.extraction.keywords}"
      config:
          sources: [swb, lobid]
      outputs:
          gnd_entries: "result.entries"

The resolved ``inputs`` mapping is passed to the function as keyword
arguments.  The ``config`` block is passed as the ``config`` kwarg
(optional on the function signature). The MCP tool registry and the
SharedContext are passed as ``tool_registry`` and ``context`` kwargs
when the function declares them, so deterministic functions can still
call MCP tools (e.g. ``search_swb``) without the LLM overhead.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict

from src.core.agents.context_path import resolve_mapping
from src.core.agents.registry import get_tool_fn, register_step
from src.core.agents.steps.base_step import BaseStep

logger = logging.getLogger(__name__)


@register_step("deterministic")
class DeterministicStep(BaseStep):
    """Dispatch a named Python function registered via ``register_tool_fn``."""

    def run(self, context: Any) -> Dict[str, Any]:
        raw_cfg = self.config.raw or {}
        fn_name = raw_cfg.get("function")
        if not fn_name:
            raise ValueError(
                f"Deterministic step '{self.step_id}' is missing required 'function' field"
            )

        fn = get_tool_fn(fn_name)

        kwargs = resolve_mapping(self.config.inputs, context)

        # Inject optional framework kwargs only if the function declares them.
        sig = inspect.signature(fn)
        if "config" in sig.parameters:
            kwargs.setdefault("config", raw_cfg.get("config", {}) or {})
        if "tool_registry" in sig.parameters:
            kwargs.setdefault("tool_registry", self.tool_registry)
        if "context" in sig.parameters:
            kwargs.setdefault("context", context)
        if "stream_callback" in sig.parameters:
            kwargs.setdefault("stream_callback", self.stream_callback)

        if self.stream_callback:
            self.stream_callback(
                f"\n{'='*50}\n⚙️  DeterministicStep '{self.step_id}' → {fn_name}\n{'='*50}\n"
            )

        raw = fn(**kwargs)

        # Normalise return value: always a dict under 'result'.
        if raw is None:
            result_dict = {}
        elif isinstance(raw, dict):
            result_dict = raw
        else:
            result_dict = {"value": raw}

        return {"result": result_dict}
