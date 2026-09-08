"""Centralized prompt resolution for agentic workflow steps.

Provides a single hierarchy that all YAML-configurable steps can use:

1. Inline YAML ``system_prompt`` / ``user_prompt`` in the step definition
2. ``prompt_task`` lookup via PromptService (prompts.json or prompts.yaml)
3. Workflow-level ``prompts:`` block (loaded into WorkflowDef)
4. Fallback to hardcoded defaults

Whichever branch wins, the operator's personal rules (``src/core/user_rules.py``)
are appended to the system prompt afterwards — after ``_render``, so braces in a
rule text stay literal.

Claude Generated
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def resolve_prompts(
    raw_cfg: Dict[str, Any],
    resolved_inputs: Dict[str, Any],
    context: Any,
    default_system: str = "",
    default_user: str = "",
    workflow_prompts: Optional[Dict[str, Any]] = None,
    step_id: str = "",
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
        step_id: The step's id, used to scope the operator's personal rules.
            Falls back to ``raw_cfg["id"]``.

    Returns:
        ``(system_prompt, user_prompt, llm_override_dict or None)``
    """
    llm_override: Optional[Dict[str, Any]] = None
    step_id = str(step_id or raw_cfg.get("id") or "")
    workflow = str(getattr(context, "workflow_name", "") or "")

    # --- 1. Inline YAML prompts ---
    inline_system = raw_cfg.get("system_prompt")
    inline_user = raw_cfg.get("user_prompt")
    if inline_system or inline_user:
        system = _render(inline_system or default_system, resolved_inputs)
        user = _render(inline_user or default_user, resolved_inputs)
        logger.debug("PromptResolver: using inline YAML prompts")
        return _with_user_rules(system, workflow, step_id, context), user, None

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
                    return (
                        _with_user_rules(system, workflow, step_id, context),
                        user,
                        llm_override,
                    )
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
            return (
                _with_user_rules(system, workflow, step_id, context),
                user,
                llm_override or None,
            )

    # --- 4. Fallback ---
    system = _render(default_system, resolved_inputs)
    user = _render(default_user, resolved_inputs)
    logger.debug("PromptResolver: using hardcoded/default prompts")
    return _with_user_rules(system, workflow, step_id, context), user, None


def _with_user_rules(system: str, workflow: str, step_id: str, context: Any) -> str:
    """Append the operator's personal rules for this workflow/step.

    Runs after ``_render`` on purpose, so ``{...}`` inside a rule text is never
    substituted. Without a matching rule the block is empty and ``system`` comes
    back unchanged — that byte-identity is pinned by a test.

    The rules that were actually injected are collected on the context
    (``applied_user_rules``) so the saved result can say which rules shaped it.
    - Claude Generated
    """
    from src.core.user_rules import STEP_REFLECTION, append_rules_block, rules_block_for

    if step_id == STEP_REFLECTION:
        # The reflection composes its own rules section (USER_RULES_INTRO plus,
        # on the last cycle, the production gate). Appending the generic block
        # here too printed every rule twice. - Claude Generated
        return system

    block, rules = rules_block_for(workflow=workflow, step=step_id)
    if not block:
        return system
    _record_applied(context, rules)
    logger.debug(
        f"PromptResolver: injected {len(rules)} user rule(s) "
        f"into step='{step_id or '?'}' workflow='{workflow or '?'}'"
    )
    return append_rules_block(system, block)


def _record_applied(context: Any, rules: Any) -> None:
    """Note the injected rules on the SharedContext (id + text, deduplicated)."""
    if context is None:
        return
    try:
        seen = getattr(context, "applied_user_rules", None)
        if seen is None:
            seen = []
            setattr(context, "applied_user_rules", seen)
        known = {entry.get("id") for entry in seen}
        for rule in rules:
            if rule.id not in known:
                seen.append({"id": rule.id, "text": rule.text})
                known.add(rule.id)
    except Exception as exc:  # a context without attribute support must not break the run
        from src.utils.error_visibility import log_caught

        log_caught(logger, exc, "prompt_resolver: recording applied rules")


#: A ``{name}`` marker. Names are identifiers, so JSON braces in a prompt
#: (``{\n  "status": …``) are never touched.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _render(template: str, values: Dict[str, Any]) -> str:
    """Replace ``{name}`` markers with stringified values, in a single pass.

    One pass, not repeated ``str.replace``: a substituted value must not itself
    be scanned for markers. The sequential version did exactly that, so a
    personal rule reading "nenne {dk_codes}" came back from the reflection
    prompt with the run's DK codes pasted into it — the module promises the
    opposite (``user_rules``: braces in a rule text stay literal).

    A marker with no value is left standing, as before. - Claude Generated
    """
    if not template:
        return ""

    def _stringify(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    def _replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        return _stringify(values[name])

    return _PLACEHOLDER_RE.sub(_replace, template)
