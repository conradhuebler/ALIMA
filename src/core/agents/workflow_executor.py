"""Workflow v4 Executor - Claude Generated.

Takes a parsed :class:`WorkflowDef` and runs it against a SharedContext.
Instantiates each step via the plugin registry, handles dependency
validation, and collects per-step :class:`StepResult` objects.

Kept separate from MetaAgent so v4 workflows can be driven independently
(e.g. from CLI, tests, or non-ALIMA use cases) without pulling in the
v3 orchestration code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from src.core.agents.registry import get_step_class
from src.core.agents.steps.base_step import BaseStep, StepResult
from src.core.agents.workflow_loader import WorkflowDef

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """Summary of a workflow execution."""
    workflow_name: str
    success: bool
    duration_seconds: float
    step_results: List[StepResult] = field(default_factory=list)
    error: Optional[str] = None


class WorkflowExecutor:
    """Sequentially executes the steps in a :class:`WorkflowDef`.

    The executor is stateless across runs — each :meth:`run` call uses the
    provided context + workflow.  Steps are instantiated fresh per run.
    """

    def __init__(
        self,
        llm_service: Any = None,
        tool_registry: Any = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.stream_callback = stream_callback

    def run(
        self,
        workflow: WorkflowDef,
        context: Any,
        *,
        only_step: Optional[str] = None,
        stop_on_error: bool = False,
    ) -> ExecutionReport:
        """Execute all enabled steps in ``workflow`` against ``context``.

        Args:
            workflow: Parsed workflow definition.
            context: SharedContext instance (mutated in-place).
            only_step: If set, run only this step id (skips others).
                       Dependencies are NOT automatically run.
            stop_on_error: If True, abort on the first failing step.

        Returns:
            ExecutionReport with per-step results.
        """
        start = time.time()
        results: List[StepResult] = []
        overall_success = True
        error: Optional[str] = None

        steps = workflow.steps
        if only_step:
            steps = [s for s in steps if s.id == only_step]
            if not steps:
                return ExecutionReport(
                    workflow_name=workflow.name,
                    success=False,
                    duration_seconds=0.0,
                    error=f"Step '{only_step}' not found in workflow",
                )

        for cfg in steps:
            if not cfg.enabled:
                logger.info(f"Skipping '{cfg.id}' (disabled in workflow)")
                continue

            try:
                step_cls = get_step_class(cfg.type)
            except KeyError as e:
                overall_success = False
                error = str(e)
                if stop_on_error:
                    break
                continue

            step: BaseStep = step_cls(
                config=cfg,
                llm_service=self.llm_service,
                tool_registry=self.tool_registry,
                stream_callback=self.stream_callback,
            )

            if self.stream_callback:
                self.stream_callback(f"\n📍 Step: {cfg.id} ({cfg.type})\n")

            result = step.execute(context)
            results.append(result)

            if not result.success:
                overall_success = False
                error = result.error
                logger.error(f"Step '{cfg.id}' failed: {result.error}")
                if stop_on_error:
                    break

        return ExecutionReport(
            workflow_name=workflow.name,
            success=overall_success,
            duration_seconds=time.time() - start,
            step_results=results,
            error=error,
        )
