"""CompositeStep — executes a sub-workflow against the same context.

Enables reusable sub-workflows without writing Python.  The YAML defines a
list of steps that run sequentially, sharing the parent context.

Claude Generated
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..registry import get_step_class
from ..workflow_executor import WorkflowExecutor
from ..workflow_loader import WorkflowDef, StepConfig
from .base_step import BaseStep

logger = logging.getLogger(__name__)


class CompositeStep(BaseStep):
    """Execute a nested list of steps defined in YAML."""

    def run(self, context: Any) -> Dict[str, Any]:
        raw_cfg = self.config.raw or {}
        sub_steps_raw = raw_cfg.get("steps", [])
        if not sub_steps_raw:
            logger.warning(f"CompositeStep '{self.step_id}': no sub-steps defined")
            return {"response": {}, "response_text": "", "iterations": 0}

        # Build StepConfig list
        sub_configs: List[StepConfig] = []
        for i, step in enumerate(sub_steps_raw):
            if not isinstance(step, dict):
                continue
            step_id = step.get("id", f"{self.step_id}_sub_{i}")
            step_type = step.get("type")
            if not step_type:
                continue
            sub_configs.append(StepConfig(
                id=step_id,
                type=step_type,
                enabled=bool(step.get("enabled", True)),
                depends_on=list(step.get("depends_on", []) or []),
                inputs=dict(step.get("inputs", {}) or {}),
                outputs=dict(step.get("outputs", {}) or {}),
                raw=dict(step),
            ))

        # Build a minimal WorkflowDef
        sub_workflow = WorkflowDef(
            name=f"{self.step_id}_composite",
            version="4.0",
            steps=sub_configs,
            raw={"steps": sub_steps_raw},
        )

        executor = WorkflowExecutor(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            stream_callback=self.stream_callback,
        )

        report = executor.run(sub_workflow, context)

        return {
            "response": {
                "success": report.success,
                "step_count": len(sub_configs),
                "step_results": {r.step_id: r.success for r in report.step_results},
            },
            "response_text": f"Composite '{self.step_id}': {len([r for r in report.step_results if r.success])}/{len(sub_configs)} steps succeeded",
            "iterations": 1,
        }
