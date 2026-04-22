"""Workflow v4 YAML Loader - Claude Generated.

Parses a v4 workflow file into a :class:`WorkflowDef` containing a list of
:class:`StepConfig` objects.  The returned steps are *config only* — they
are instantiated into concrete ``BaseStep`` objects by the executor.

Schema detection:
    * Has top-level ``steps:``  → v4
    * Has top-level ``pipeline:`` → v3 (handled by the legacy MetaAgent path)

The loader performs minimal validation:
    * ``steps`` is a non-empty list of dicts
    * Each step has ``id`` and ``type``
    * ``type`` exists in ``STEP_REGISTRY``

Detailed validation (tool names, function names, placeholder paths) happens
lazily at execution time to keep the loader independent of lazy-loaded
services.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.agents.registry import STEP_REGISTRY
from src.core.agents.steps.base_step import StepConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkflowDef:
    """Parsed workflow definition."""
    name: str
    version: str
    description: str = ""
    steps: List[StepConfig] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    context_init: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_v4(self) -> bool:
        """Workflows are v4 if they use the ``steps:`` top-level key."""
        return bool(self.steps) and "steps" in self.raw


DEFAULT_SEARCH_PATHS = [
    Path("workflows"),
    Path.home() / ".config" / "alima" / "workflows",
    Path(__file__).parent.parent.parent.parent / "workflows",
]


def find_workflow_file(
    name: str, search_paths: Optional[List[Path]] = None
) -> Optional[Path]:
    """Locate ``<name>.yaml`` in the standard search paths."""
    paths = search_paths or DEFAULT_SEARCH_PATHS
    for base in paths:
        candidate = base / f"{name}.yaml"
        if candidate.exists():
            return candidate
    return None


def is_v4_yaml(data: Dict[str, Any]) -> bool:
    """Return True if the parsed YAML uses the v4 ``steps:`` schema."""
    return isinstance(data, dict) and isinstance(data.get("steps"), list)


def load_workflow(
    path: Path | str, *, strict: bool = True
) -> WorkflowDef:
    """Load and parse a v4 workflow YAML file.

    Args:
        path: Path to the YAML file.
        strict: If True, unknown step ``type`` values raise ValueError.
                If False, unknown types survive (useful for tests that
                register plugins after loading).

    Returns:
        WorkflowDef

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the schema is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level mapping")

    if not is_v4_yaml(data):
        raise ValueError(
            f"{path}: not a v4 workflow (expected top-level 'steps:' list). "
            f"Use MetaAgent.load_workflow() for v3 'pipeline:' workflows."
        )

    steps = parse_steps(data["steps"], strict=strict, source=str(path))

    return WorkflowDef(
        name=str(data.get("name", path.stem)),
        version=str(data.get("version", "4.0")),
        description=str(data.get("description", "")),
        steps=steps,
        settings=data.get("settings", {}) or {},
        context_init=data.get("context_init", {}) or {},
        raw=data,
    )


def parse_steps(
    raw_steps: List[Dict[str, Any]], *, strict: bool = True, source: str = "<memory>"
) -> List[StepConfig]:
    """Convert the raw ``steps:`` list into :class:`StepConfig` objects."""
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{source}: 'steps' must be a non-empty list")

    out: List[StepConfig] = []
    seen_ids: set[str] = set()

    for i, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise ValueError(f"{source}: step #{i} must be a mapping")

        step_id = step.get("id")
        step_type = step.get("type")

        if not step_id:
            raise ValueError(f"{source}: step #{i} missing 'id'")
        if not step_type:
            raise ValueError(f"{source}: step '{step_id}' missing 'type'")
        if step_id in seen_ids:
            raise ValueError(f"{source}: duplicate step id '{step_id}'")
        seen_ids.add(step_id)

        if strict and step_type not in STEP_REGISTRY:
            raise ValueError(
                f"{source}: step '{step_id}' has unknown type '{step_type}'. "
                f"Registered: {sorted(STEP_REGISTRY)}"
            )

        out.append(StepConfig(
            id=step_id,
            type=step_type,
            enabled=bool(step.get("enabled", True)),
            depends_on=list(step.get("depends_on", []) or []),
            inputs=dict(step.get("inputs", {}) or {}),
            outputs=dict(step.get("outputs", {}) or {}),
            raw=dict(step),
        ))

    return out
