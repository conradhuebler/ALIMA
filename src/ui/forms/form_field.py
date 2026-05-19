"""FormField dataclass for SingleStepDialog (P-γ). Claude Generated.

Each step input becomes one ``FormField`` describing how the UI should render
and treat it (user-editable, derived from upstream, missing prerequisite).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FormField:
    """Single form-field descriptor produced by :func:`build_step_form`.

    Attributes:
        name: Input key from the step's ``inputs:`` YAML block.
        expr: Raw placeholder expression (e.g. ``"${abstract}"``).
        kind: One of ``"user_fill"``, ``"derived"``, ``"static"``.
        value: Resolved value from context (``None`` if missing or not yet run).
        widget: Suggested widget hint — ``"text"``, ``"list"``, ``"json"``,
                ``"readonly"`` (derived fields), or ``"raw_json"`` (fallback).
        missing: True when the field's value cannot be resolved against the
                 current context (relevant for derived inputs whose upstream
                 step has not run yet).
        writer_step_id: For derived missing fields, the step that would
                        produce the value (used by cascade-run).
    """

    name: str
    expr: str
    kind: str
    value: Any = None
    widget: str = "json"
    missing: bool = False
    writer_step_id: Optional[str] = None
