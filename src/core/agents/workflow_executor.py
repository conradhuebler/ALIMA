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
from typing import Any, Callable, Dict, List, Optional

from src.core.agents.conditional_engine import ConditionalEngine
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
        context_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.stream_callback = stream_callback
        # Emits (step_id, snapshot) per step. Called once for "running" before
        # execute() and once with final status after. Widget uses this for
        # the live agentic-context display.
        self.context_callback = context_callback

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

        # Name the driving workflow on the context so the exported state can say
        # which one produced it (same tolerance as the prompt injection below —
        # a context without attribute support must not break execution).
        # - Claude Generated
        try:
            context.workflow_name = getattr(workflow, "name", "") or ""
        except Exception:
            pass

        # Inject workflow prompts into context for step-level prompt resolution
        if hasattr(context, "_workflow_prompts"):
            context._workflow_prompts = workflow.prompts
        else:
            try:
                context._workflow_prompts = workflow.prompts
            except Exception:
                pass  # context without attribute support — prompts resolved from YAML defaults - Claude Generated

        # P-η: propagate settings.seed → context.seed when caller did not set one.
        # Per-step `llm.seed` still overrides via LLMAgentStep._llm_params().
        settings_seed = workflow.settings.get("seed") if isinstance(workflow.settings, dict) else None
        if settings_seed is not None and getattr(context, "seed", None) is None:
            try:
                context.seed = settings_seed
            except Exception:
                pass  # context without attribute support — per-step llm.seed still applies - Claude Generated

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

            # Evaluate conditional expression
            if cfg.condition:
                try:
                    should_run = ConditionalEngine.evaluate(cfg.condition, context)
                except Exception as exc:
                    # A broken condition must not silently skip or run the step:
                    # treat it as a step failure so it shows up in the report - Claude Generated
                    overall_success = False
                    error = f"Condition '{cfg.condition}' of step '{cfg.id}' failed to evaluate: {exc}"
                    logger.error(error, exc_info=True)
                    results.append(StepResult(
                        step_id=cfg.id,
                        success=False,
                        error=error,
                        duration_seconds=0.0,
                    ))
                    if stop_on_error:
                        break
                    continue
                if not should_run:
                    logger.info(f"Skipping '{cfg.id}' (condition '{cfg.condition}' is False)")
                    if self.stream_callback:
                        self.stream_callback(f"\n⏭️ Step: {cfg.id} — skipped (condition)\n")
                    results.append(StepResult(
                        step_id=cfg.id,
                        success=True,
                        data={"skipped": True, "reason": f"condition '{cfg.condition}' evaluated to False"},
                        duration_seconds=0.0,
                    ))
                    continue

            try:
                step_cls = get_step_class(cfg.type)
            except KeyError as e:
                overall_success = False
                error = str(e)
                if stop_on_error:
                    break
                continue

            try:
                step: BaseStep = step_cls(
                    config=cfg,
                    llm_service=self.llm_service,
                    tool_registry=self.tool_registry,
                    stream_callback=self.stream_callback,
                )
            except Exception as exc:
                # Constructor failures (bad config, missing service) must be
                # reported per step, not crash the workflow thread - Claude Generated
                logger.error(f"Step '{cfg.id}' could not be instantiated: {exc}", exc_info=True)
                overall_success = False
                error = f"{type(exc).__name__}: {exc}"
                results.append(StepResult(
                    step_id=cfg.id,
                    success=False,
                    error=error,
                    duration_seconds=0.0,
                ))
                if stop_on_error:
                    break
                continue

            if self.stream_callback:
                self.stream_callback(f"\n📍 Step: {cfg.id} ({cfg.type})\n")

            self._emit_snapshot(context, cfg, status="running")

            try:
                result = step.execute(context)
            except Exception as exc:
                # Step implementations should return StepResult(success=False),
                # but an escaping exception must not kill the whole workflow
                # thread unreported - Claude Generated
                logger.error(f"Step '{cfg.id}' raised: {exc}", exc_info=True)
                result = StepResult(
                    step_id=cfg.id,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_seconds=0.0,
                )
            results.append(result)

            self._emit_snapshot(
                context,
                cfg,
                status="completed" if result.success else "error",
                duration=result.duration_seconds,
                error=result.error,
            )

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

    def _emit_snapshot(
        self,
        context: Any,
        cfg: Any,
        *,
        status: str,
        duration: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Push a SharedContext snapshot to the UI context_callback.

        Swallows all exceptions so a flaky widget can never break workflow execution.
        To keep UI responsive, large lists are capped at MAX_SNAPSHOT_LIST_LEN
        and the snapshot is only pushed for completed/error steps (not running).
        """
        if self.context_callback is None:
            return
        # Skip "running" snapshots — the widget only needs the final state.
        if status == "running":
            return

        try:
            snap = context.to_dict() if hasattr(context, "to_dict") else {}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"context snapshot serialization failed: {e}")
            snap = {}

        # Per-field caps. The user-facing data tables (GND pool ~1000+,
        # DK catalog results) must arrive COMPLETE — a hard 50-entry cap made
        # the Pipeline-Tab silently show partial pools/title lists. Only
        # genuinely unbounded bookkeeping lists stay tightly capped.
        # - Claude Generated
        FIELD_CAPS = {
            "gnd_entries": 5000,
            "dk_search_results": 2000,
            "selected_keywords": 1000,
            "extracted_keywords": 500,
            "keyword_chains": 200,
            "dk_classifications": 200,
            "rvk_classifications": 200,
            "missing_concepts": 200,
            "execution_history": 50,
        }
        for key, cap in FIELD_CAPS.items():
            val = snap.get(key)
            if isinstance(val, list) and len(val) > cap:
                snap[key] = val[:cap]
                # add a sentinel so the widget knows data was truncated
                snap[key].append({"_truncated": len(val) - cap})

        snap["_step_id"] = cfg.id
        snap["_step_type"] = cfg.type
        snap["_step_description"] = getattr(cfg, "description", "") or ""
        snap["_step_status"] = status
        if duration is not None:
            snap["_step_duration"] = duration
        if error:
            snap["_step_error"] = error
        try:
            self.context_callback(cfg.id, snap)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"context_callback raised: {e}")
