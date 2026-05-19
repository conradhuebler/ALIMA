"""StepFormBuilder for SingleStepDialog (P-γ). Claude Generated.

Pure-Python (Qt-free) translation of a step's ``inputs:`` YAML block into a
list of :class:`FormField` descriptors plus a prerequisite-resolver for
cascade-runs.

Algorithm follows WP5 §3 (`docs/single_step_model.md`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.agents.context_path import (
    PLACEHOLDER_RE,
    resolve_path,
    resolve_value,
)
from src.core.agents.workflow_loader import WorkflowDef
from src.core.agents.steps.base_step import StepConfig
from src.ui.forms.form_field import FormField


def _find_step(workflow: WorkflowDef, step_id: str) -> Tuple[int, StepConfig]:
    for idx, cfg in enumerate(workflow.steps):
        if cfg.id == step_id:
            return idx, cfg
    raise KeyError(f"step '{step_id}' not found in workflow '{workflow.name}'")


def _bare_path(expr: str) -> Optional[str]:
    """Return the path inside a bare ``${...}`` expression, else None."""
    if not isinstance(expr, str):
        return None
    stripped = expr.strip()
    m = PLACEHOLDER_RE.fullmatch(stripped)
    return m.group(1).strip() if m else None


def _collect_upstream_writes(
    workflow: WorkflowDef, target_idx: int
) -> Tuple[Set[str], Dict[str, str]]:
    """Collect roots of all output-targets written by steps before ``target_idx``.

    Returns:
        (roots, writer_map) where
            roots: set of root keys (e.g. ``"selected_keywords"``, ``"extra"``).
            writer_map: maps each fully-qualified output target to the
                        step_id that writes it (last writer wins).
    """
    roots: Set[str] = set()
    writer_map: Dict[str, str] = {}
    for cfg in workflow.steps[:target_idx]:
        for target in cfg.outputs.keys():
            root = target.split(".", 1)[0]
            roots.add(root)
            writer_map[target] = cfg.id
            writer_map[root] = cfg.id
    return roots, writer_map


def _classify(expr: Any, upstream_roots: Set[str]) -> str:
    """Classify an input expression as user_fill / derived / static."""
    path = _bare_path(expr) if isinstance(expr, str) else None
    if path is None:
        return "static"
    root = path.split(".", 1)[0]
    if root == "steps":
        return "derived"
    if root in upstream_roots:
        return "derived"
    return "user_fill"


def _has_value(context: Any, expr: str) -> bool:
    """True iff resolving ``expr`` returns a non-empty value."""
    path = _bare_path(expr)
    if path is None:
        return bool(expr)
    try:
        value = resolve_path(path, context)
    except KeyError:
        return False
    if value is None:
        return False
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return False
    return True


def _pick_widget(name: str, expr: str, kind: str) -> str:
    """Heuristic widget choice based on field name + kind."""
    if kind == "derived":
        return "readonly"
    lower = name.lower()
    if lower in ("abstract", "text", "description"):
        return "text"
    if "keyword" in lower or "entries" in lower or "concept" in lower:
        return "list"
    return "json"


def _writer_for_expr(expr: str, writer_map: Dict[str, str]) -> Optional[str]:
    """Find the step that writes the field referenced by ``expr``.

    Tries the full path first, then the root segment. Returns None if no
    upstream step writes this field (the caller should treat as user_fill).
    """
    path = _bare_path(expr)
    if path is None:
        return None
    if path.startswith("steps."):
        parts = path.split(".", 2)
        return parts[1] if len(parts) >= 2 else None
    if path in writer_map:
        return writer_map[path]
    root = path.split(".", 1)[0]
    return writer_map.get(root)


def build_step_form(
    workflow: WorkflowDef, step_id: str, context: Any
) -> List[FormField]:
    """Build the form-field list for a single step.

    Args:
        workflow: Parsed workflow definition.
        step_id: ID of the step to render.
        context: SharedContext (or compatible) used to resolve derived inputs.

    Returns:
        Ordered list of FormField (preserves YAML ``inputs:`` order). Static
        inputs are omitted from the form.
    """
    target_idx, step = _find_step(workflow, step_id)
    upstream_roots, writer_map = _collect_upstream_writes(workflow, target_idx)

    fields: List[FormField] = []
    for name, expr in (step.inputs or {}).items():
        kind = _classify(expr, upstream_roots)
        if kind == "static":
            continue

        present = _has_value(context, expr) if isinstance(expr, str) else False
        try:
            value = resolve_value(expr, context) if isinstance(expr, str) else expr
        except Exception:
            value = None

        widget = _pick_widget(name, expr, kind)
        writer = _writer_for_expr(expr, writer_map) if kind == "derived" else None

        fields.append(FormField(
            name=name,
            expr=expr,
            kind=kind,
            value=value if present else None,
            widget=widget,
            missing=(kind == "derived" and not present),
            writer_step_id=writer,
        ))

    return fields


def find_missing_prerequisites(
    workflow: WorkflowDef, step_id: str, context: Any
) -> List[str]:
    """Return the chain of step_ids needed to satisfy ``step_id`` against ``context``.

    Walks backwards from ``step_id``: any derived input whose value is missing
    triggers inclusion of its writer step; the writer's own missing inputs are
    resolved recursively. The result is topologically ordered by
    ``workflow.steps`` and always ends with ``step_id``.

    User-fill inputs that are missing are NOT cascaded (cascade-run cannot
    invent user data); they show up as ``FormField.missing`` in
    :func:`build_step_form` instead.

    Args:
        workflow: Parsed workflow definition.
        step_id: Target step.
        context: SharedContext used to check value presence.

    Returns:
        List of step_ids in execution order, e.g.
        ``["extraction", "search", "selection_chunks", ..., "classification"]``.
        Returns ``[step_id]`` when no derived prerequisite is missing.
    """
    target_idx, _ = _find_step(workflow, step_id)

    needed_steps: Set[str] = set()
    queue: List[str] = [step_id]

    while queue:
        current = queue.pop()
        idx, cfg = _find_step(workflow, current)
        upstream_roots, writer_map = _collect_upstream_writes(workflow, idx)

        for expr in (cfg.inputs or {}).values():
            if not isinstance(expr, str):
                continue
            kind = _classify(expr, upstream_roots)
            if kind != "derived":
                continue
            if _has_value(context, expr):
                continue
            writer = _writer_for_expr(expr, writer_map)
            if writer is None or writer == current:
                continue
            if writer not in needed_steps:
                needed_steps.add(writer)
                queue.append(writer)

        # Steps with empty `inputs:` (typical for deterministic steps) read
        # typed SharedContext fields directly. Their YAML does not document
        # these reads, so we fall back to `depends_on:` as a proxy.
        if not (cfg.inputs or {}):
            for dep in cfg.depends_on or []:
                if dep in needed_steps:
                    continue
                try:
                    dep_idx, _ = _find_step(workflow, dep)
                except KeyError:
                    continue
                if dep_idx >= idx:
                    continue
                needed_steps.add(dep)
                queue.append(dep)

    chain = [
        cfg.id
        for cfg in workflow.steps[: target_idx + 1]
        if cfg.id in needed_steps or cfg.id == step_id
    ]
    return chain


