"""Centralized prompt resolution for agentic workflow steps.

Provides a single hierarchy that all YAML-configurable steps can use:

1. Inline YAML ``system_prompt`` / ``user_prompt`` in the step definition
2. ``prompt_task`` lookup via PromptService (prompts.json or prompts.yaml)
3. Workflow-level ``prompts:`` block (loaded into WorkflowDef)
4. Fallback to hardcoded defaults

Claude Generated
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_prompts(
    raw_cfg: Dict[str, Any],
    resolved_inputs: Dict[str, Any],
    context: Any,
    default_system: str = "",
    default_user: str = "",
    workflow_prompts: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Resolve system and user prompts for a step.

    Args:
        raw_cfg: Step's ``raw`` dict from YAML (may contain ``system_prompt``,
            ``user_prompt``, ``prompt_task``, ``llm``, etc.).
        resolved_inputs: Already-resolved input values (for ``{var}`` substitution).
        context: SharedContext — checked for ``prompt_service``.
        default_system: Fallback system prompt if nothing else matches.
        default_user: Fallback user prompt if nothing else matches.
        workflow_prompts: Optional dict from ``WorkflowDef.prompts`` top-level block.

    Returns:
        ``(system_prompt, user_prompt, llm_override_dict or None)``
    """
    llm_override: Optional[Dict[str, Any]] = None

    # --- 1. Inline YAML prompts ---
    inline_system = raw_cfg.get("system_prompt")
    inline_user = raw_cfg.get("user_prompt")
    if inline_system or inline_user:
        system = _render(inline_system or default_system, resolved_inputs)
        user = _render(inline_user or default_user, resolved_inputs)
        logger.debug("PromptResolver: using inline YAML prompts")
        return system, user, None

    # --- 2. prompt_task lookup via PromptService ---
    task = raw_cfg.get("prompt_task")
    if task:
        ps = getattr(context, "prompt_service", None)
        if ps is not None:
            try:
                model = getattr(context, "model", "")
                cfg = ps.get_prompt_config(task, model)
                if cfg:
                    system = _render(cfg.system or default_system, resolved_inputs)
                    user = _render(cfg.prompt or default_user, resolved_inputs)
                    llm_override = {
                        "temperature": float(cfg.temp),
                        "top_p": float(cfg.p_value),
                    }
                    if cfg.output_format:
                        llm_override["output_format"] = cfg.output_format
                    logger.debug(
                        f"PromptResolver: loaded from PromptService task='{task}' "
                        f"(temp={cfg.temp}, top_p={cfg.p_value})"
                    )
                    return system, user, llm_override
            except Exception as exc:
                logger.warning(
                    f"PromptResolver: PromptService lookup failed for task='{task}': {exc}"
                )

    # --- 3. Workflow-level prompts block ---
    if workflow_prompts and task and task in workflow_prompts:
        wp = workflow_prompts[task]
        if isinstance(wp, dict):
            system = _render(wp.get("system", default_system), resolved_inputs)
            user = _render(wp.get("prompt", default_user), resolved_inputs)
            llm_override = {}
            if "temperature" in wp:
                llm_override["temperature"] = float(wp["temperature"])
            if "top_p" in wp:
                llm_override["top_p"] = float(wp["top_p"])
            logger.debug(f"PromptResolver: loaded from workflow-level prompts block task='{task}'")
            return system, user, llm_override or None

    # --- 4. Fallback ---
    system = _render(default_system, resolved_inputs)
    user = _render(default_user, resolved_inputs)
    logger.debug("PromptResolver: using hardcoded/default prompts")
    return system, user, None


def _render(template: str, values: Dict[str, Any]) -> str:
    """Replace ``{name}`` markers with stringified values."""
    if not template:
        return ""

    def _stringify(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    out = template
    for name in sorted(values.keys(), key=len, reverse=True):
        out = out.replace("{" + name + "}", _stringify(values[name]))
    return out
