"""Abstract BaseStep for Workflow v4 - Claude Generated.

Every step in a v4 workflow is a subclass of ``BaseStep``.  The MetaAgent
instantiates each step once per execution with its parsed YAML config, then
calls :meth:`execute` passing the SharedContext.

Subclasses:
    * ``LLMAgentStep``      — LLM with optional tool-calling (AgentLoop).
    * ``DeterministicStep`` — dispatches to a registered Python callable.

The design splits *what* a step does (subclass) from *how* it's wired
(SharedContext + YAML config).  Steps write their outputs through the
``outputs`` mapping into either ``context.step_results[step_id]`` or a
typed attribute on SharedContext, so downstream steps can read them via
``${steps.<id>.<field>}`` or ``${<typed_field>}``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StepConfig:
    """Parsed YAML configuration for a single step.

    The ``raw`` dict is the original YAML block, kept so subclasses can
    read step-type-specific keys (``system_prompt``, ``function``, ...).
    """
    id: str
    type: str
    enabled: bool = True
    description: str = ""  # human-readable step purpose (from YAML) - Claude Generated
    depends_on: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None  # when: expression
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Outcome of a single step execution."""
    step_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    # Diagnostics (optional):
    iterations: int = 1
    tool_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "success": self.success,
            "data": self.data,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
        }


class BaseStep(ABC):
    """Abstract base for all v4 workflow steps.

    Subclasses implement :meth:`run` and optionally override
    :meth:`apply_outputs`.  ``execute()`` is the single entry point used by
    the workflow executor — it handles the ``outputs`` mapping and any
    diagnostics the base class wants to emit.
    """

    def __init__(
        self,
        config: StepConfig,
        *,
        llm_service: Any = None,
        tool_registry: Any = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.stream_callback = stream_callback

    @property
    def step_id(self) -> str:
        return self.config.id

    # -- Subclass hooks -----------------------------------------------------

    @abstractmethod
    def run(self, context: Any) -> Dict[str, Any]:
        """Do the actual work.

        Args:
            context: SharedContext instance.

        Returns:
            A dict with raw step output.  Keys here are what the
            ``outputs:`` mapping can reference as ``response.<key>`` (for
            LLM agents) or ``result.<key>`` (for deterministic steps).

        Raises:
            Any exception — caller converts it to a failed StepResult.
        """
        raise NotImplementedError

    # -- Execution ---------------------------------------------------------

    def execute(self, context: Any) -> StepResult:
        """Run the step and apply output mapping to the shared context."""
        import time
        start = time.time()
        try:
            raw = self.run(context)
            self.apply_outputs(raw, context)
            return StepResult(
                step_id=self.step_id,
                success=True,
                data=raw,
                duration_seconds=time.time() - start,
            )
        except Exception as e:  # noqa: BLE001
            return StepResult(
                step_id=self.step_id,
                success=False,
                data={},
                duration_seconds=time.time() - start,
                error=str(e),
            )

    # -- Output mapping ----------------------------------------------------

    # Output mapping semantics:
    #   key  = destination on the context (dotted path, see below)
    #   val  = source path inside ``raw`` (dotted; ``result.foo`` / ``response.foo``
    #          are common prefixes, bare ``foo`` also accepted)
    # Destination forms:
    #   "some_field"       → SharedContext.<some_field> = value
    #   "steps.<id>.<k>"   → context.step_results[id][k] = value
    #   anything else      → stored under context.extra
    def apply_outputs(self, raw: Dict[str, Any], context: Any) -> None:
        """Write step outputs into the context.

        Always stores the raw dict at ``context.step_results[self.step_id]``
        so it is reachable as ``${steps.<id>.<key>}`` downstream. Then
        applies any explicit ``outputs:`` mapping for typed fields.
        """
        if not hasattr(context, "step_results"):
            return
        context.step_results[self.step_id] = raw

        for dest, source in (self.config.outputs or {}).items():
            value = _pick(raw, source)
            _write(context, self.step_id, dest, value)


# -- Helpers -----------------------------------------------------------------

def _pick(raw: Dict[str, Any], source: str) -> Any:
    """Resolve a source path inside a raw dict.

    The path is a plain dotted traversal:
        response.<k>  — raw["response"][k]          (for LLM agents)
        result.<k>    — raw["result"][k]            (for deterministic steps)
        <k>.<l>.<m>   — raw[k][l][m]
    Integer segments index into lists.
    """
    if not isinstance(source, str):
        return source
    cur: Any = raw
    for seg in source.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                idx = int(seg)
            except ValueError:
                return None
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            return None
    return cur


def _write(context: Any, step_id: str, dest: str, value: Any) -> None:
    """Write ``value`` to ``dest`` on the context.

    Destination forms:
      ``foo``            → ``setattr(context, "foo", value)`` if the attr exists
      ``steps.<id>.<k>`` → ``context.step_results[<id>][<k>] = value``
      ``extra.<path>``   → ``context.extra[<...>] = value`` (creates nested dicts)
      else              → stored under ``context.extra[<dest>]``.
    """
    if value is None:
        return
    parts = dest.split(".")
    root = parts[0]

    if root == "steps" and len(parts) >= 3:
        target_step = parts[1]
        key_path = parts[2:]
        bucket = context.step_results.setdefault(target_step, {})
        _set_nested(bucket, key_path, value)
        return

    if root == "extra":
        extra = getattr(context, "extra", None)
        if extra is None:
            # Target context has no extra dict — silently ignore.
            return
        _set_nested(extra, parts[1:] or ["_"], value)
        return

    # Single-segment → try typed field on context.
    if len(parts) == 1 and hasattr(context, dest):
        setattr(context, dest, value)
        return

    # Fallback: store in extra if available, else step_results.
    extra = getattr(context, "extra", None)
    if extra is not None:
        _set_nested(extra, parts, value)
    else:
        context.step_results.setdefault(step_id, {})[dest] = value


def _set_nested(d: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = d
    for seg in path[:-1]:
        if seg not in cur or not isinstance(cur[seg], dict):
            cur[seg] = {}
        cur = cur[seg]
    cur[path[-1]] = value
