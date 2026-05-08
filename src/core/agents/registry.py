"""Plugin Registry for Workflow v4 Steps and Tool Functions - Claude Generated.

Two parallel registries:

* ``STEP_REGISTRY`` — maps ``type:`` name in YAML to a ``BaseStep`` subclass.
  Populated via ``@register_step("llm_agent")`` decorators.
* ``TOOL_FN_REGISTRY`` — maps ``function:`` name in YAML to a Python callable
  used by ``DeterministicStep``.  Populated via ``@register_tool_fn("name")``.

The registries intentionally live as module-level globals so new plugins can
import the decorators and attach their step/function classes without touching
the dispatch code.  All registered names must be unique — re-registering the
same name raises ``ValueError`` (guards against silent shadowing).

To enumerate what's available at runtime:

    from src.core.agents import registry
    print(registry.list_steps())
    print(registry.list_tool_fns())
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

STEP_REGISTRY: Dict[str, Type] = {}
TOOL_FN_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_step(name: str):
    """Decorator: register a ``BaseStep`` subclass under ``name``.

    Example::

        @register_step("llm_agent")
        class LLMAgentStep(BaseStep):
            ...
    """
    def _wrap(cls):
        if name in STEP_REGISTRY and STEP_REGISTRY[name] is not cls:
            raise ValueError(
                f"Step type '{name}' already registered to {STEP_REGISTRY[name].__name__}"
            )
        STEP_REGISTRY[name] = cls
        return cls
    return _wrap


def get_step_class(name: str) -> Type:
    """Look up a step class by its registered name.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in STEP_REGISTRY:
        raise KeyError(
            f"Unknown step type '{name}'. Registered: {sorted(STEP_REGISTRY)}"
        )
    return STEP_REGISTRY[name]


def list_steps() -> list[str]:
    """Return sorted list of registered step type names."""
    return sorted(STEP_REGISTRY)


def register_tool_fn(name: str):
    """Decorator: register a plain callable as a deterministic tool function.

    The wrapped function receives keyword-arguments matching the YAML
    ``inputs:`` mapping (after placeholder resolution) plus an optional
    ``config`` dict for static step configuration.  It must return a dict
    or a JSON-serialisable object.

    Example::

        @register_tool_fn("gnd_batch_search")
        def gnd_batch_search(keywords, config=None, registry=None):
            return {"entries": [...]}
    """
    def _wrap(fn):
        if name in TOOL_FN_REGISTRY and TOOL_FN_REGISTRY[name] is not fn:
            raise ValueError(
                f"Tool function '{name}' already registered to "
                f"{TOOL_FN_REGISTRY[name].__name__}"
            )
        TOOL_FN_REGISTRY[name] = fn
        return fn
    return _wrap


def get_tool_fn(name: str) -> Callable[..., Any]:
    """Look up a registered tool function by name."""
    if name not in TOOL_FN_REGISTRY:
        raise KeyError(
            f"Unknown tool function '{name}'. Registered: {sorted(TOOL_FN_REGISTRY)}"
        )
    return TOOL_FN_REGISTRY[name]


def list_tool_fns() -> list[str]:
    """Return sorted list of registered tool function names."""
    return sorted(TOOL_FN_REGISTRY)


def register_step_from_yaml(name: str, yaml_path: str) -> None:
    """Register a composite step type from a YAML file.

    The YAML defines a sub-workflow (list of steps) that the CompositeStep
    will execute.  This enables reusable step templates without Python code.
    """
    from pathlib import Path
    from .steps.composite_step import CompositeStep

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Step template not found: {yaml_path}")

    class _DynamicCompositeStep(CompositeStep):
        """Dynamically loaded composite step."""
        _template_path = str(path)

    # Re-registering the same name with the same class is idempotent
    if name in STEP_REGISTRY and STEP_REGISTRY[name] is not _DynamicCompositeStep:
        raise ValueError(
            f"Step type '{name}' already registered to {STEP_REGISTRY[name].__name__}"
        )
    STEP_REGISTRY[name] = _DynamicCompositeStep


def _reset_for_tests() -> None:
    """Clear both registries — used only in tests that register mock steps/fns."""
    STEP_REGISTRY.clear()
    TOOL_FN_REGISTRY.clear()
