"""MetaAgent must not run a step whose `depends_on` has not run yet.

The LLM planner names the next step freely and `WorkflowExecutor.run(only_step=…)`
executes whatever it is handed. On 2026-09-03 a Mistral planner jumped from
`selection_chunks` straight to `dk_collect`/`classification`; `selection` was
pulled in only at the end by the finish-veto. The classification prompt's
`${extra.final_keywords}` resolved to empty, so no keywords reached `rvk_lookup`
(zero RVK in the result) and `dk_collect` built its catalog pool from the coarse
chunk selection instead of the curated final keywords.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.agents import registry
from src.core.agents.meta_agent import MetaAgent
from src.core.agents.registry import register_step, register_tool_fn
from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.base_step import StepConfig
from src.core.agents.steps.deterministic_step import DeterministicStep
from src.core.agents.workflow_loader import WorkflowDef, load_workflow

REPO_ROOT = Path(__file__).resolve().parents[1]


def _chain_workflow() -> WorkflowDef:
    """The alima_v51 backbone, reduced to ids + depends_on."""
    steps = [
        StepConfig(id="extraction", type="deterministic", raw={"function": "noop"}),
        StepConfig(id="search", type="deterministic", depends_on=["extraction"],
                   raw={"function": "noop"}),
        StepConfig(id="selection_chunks", type="deterministic", depends_on=["search"],
                   raw={"function": "noop"}),
        StepConfig(id="selection", type="deterministic", depends_on=["selection_chunks"],
                   raw={"function": "noop"}),
        StepConfig(id="verify_keywords", type="deterministic", depends_on=["selection"],
                   raw={"function": "noop"}),
        StepConfig(id="dk_collect", type="deterministic", depends_on=["verify_keywords"],
                   raw={"function": "noop"}),
        StepConfig(id="classification", type="deterministic", depends_on=["dk_collect"],
                   raw={"function": "noop"}),
    ]
    return WorkflowDef(name="chain", version="1", steps=steps)


def _context(*steps_run: str) -> SharedContext:
    ctx = SharedContext(abstract="Test abstract")
    ctx.execution_history = [
        {"cycle": i + 1, "step": s, "success": True, "duration": 0.0, "error": None}
        for i, s in enumerate(steps_run)
    ]
    return ctx


class TestUnmetDependency(unittest.TestCase):
    """Unit tests for the dependency walk itself."""

    def test_returns_deepest_missing_prerequisite(self):
        wf = _chain_workflow()
        ctx = _context("extraction", "search", "selection_chunks")
        # classification ← dk_collect ← verify_keywords ← selection: the walk must
        # return the one that is actually runnable now, not the nearest dep.
        self.assertEqual(MetaAgent._unmet_dependency(wf, ctx, "classification"), "selection")
        self.assertEqual(MetaAgent._unmet_dependency(wf, ctx, "dk_collect"), "selection")

    def test_none_when_dependencies_satisfied(self):
        wf = _chain_workflow()
        ctx = _context("extraction", "search", "selection_chunks", "selection",
                       "verify_keywords", "dk_collect")
        self.assertIsNone(MetaAgent._unmet_dependency(wf, ctx, "classification"))

    def test_step_without_dependencies_is_never_redirected(self):
        wf = _chain_workflow()
        self.assertIsNone(MetaAgent._unmet_dependency(wf, _context(), "extraction"))

    def test_disabled_dependency_counts_as_satisfied(self):
        """A disabled dep can never run — redirecting to it would deadlock."""
        wf = _chain_workflow()
        for step in wf.steps:
            if step.id == "verify_keywords":
                step.enabled = False
        ctx = _context("extraction", "search", "selection_chunks", "selection")
        self.assertIsNone(MetaAgent._unmet_dependency(wf, ctx, "dk_collect"))

    def test_unknown_dependency_counts_as_satisfied(self):
        wf = WorkflowDef(name="x", version="1", steps=[
            StepConfig(id="a", type="deterministic", depends_on=["ghost"],
                       raw={"function": "noop"}),
        ])
        self.assertIsNone(MetaAgent._unmet_dependency(wf, _context(), "a"))

    def test_dependency_cycle_leaves_the_planner_choice_alone(self):
        """A cycle has no runnable prerequisite — no redirect, and no hang."""
        wf = WorkflowDef(name="x", version="1", steps=[
            StepConfig(id="a", type="deterministic", depends_on=["b"], raw={"function": "noop"}),
            StepConfig(id="b", type="deterministic", depends_on=["a"], raw={"function": "noop"}),
        ])
        self.assertIsNone(MetaAgent._unmet_dependency(wf, _context(), "a"))

    def test_real_alima_v51_graph_redirects_classification_to_selection(self):
        """Pins the production graph, not just the fixture."""
        wf = load_workflow(REPO_ROOT / "workflows" / "alima_v51.yaml", strict=False)
        ctx = _context("extraction", "search", "selection_chunks")
        self.assertEqual(MetaAgent._unmet_dependency(wf, ctx, "classification"), "selection")


class TestMetaAgentGate(unittest.TestCase):
    """The gate as it acts inside the PLAN→EXECUTE loop."""

    def setUp(self):
        self._saved_steps = dict(registry.STEP_REGISTRY)
        self._saved_fns = dict(registry.TOOL_FN_REGISTRY)
        registry._reset_for_tests()
        register_step("deterministic")(DeterministicStep)

        @register_tool_fn("noop")
        def _noop_fn(**_):
            return {"result": "done"}

    def tearDown(self):
        registry.STEP_REGISTRY.clear()
        registry.STEP_REGISTRY.update(self._saved_steps)
        registry.TOOL_FN_REGISTRY.clear()
        registry.TOOL_FN_REGISTRY.update(self._saved_fns)

    def test_premature_classification_runs_selection_instead(self):
        wf = _chain_workflow()
        ctx = _context("extraction", "search", "selection_chunks")
        agent = MetaAgent(llm_service=None, tool_registry=MagicMock(), max_cycles=1)

        with patch.object(MetaAgent, "_plan_next_step", return_value="classification"):
            agent.run(wf, ctx)

        self.assertEqual(ctx.execution_history[-1]["step"], "selection")

    def test_satisfied_step_is_executed_as_planned(self):
        wf = _chain_workflow()
        ctx = _context("extraction", "search", "selection_chunks", "selection",
                       "verify_keywords", "dk_collect")
        agent = MetaAgent(llm_service=None, tool_registry=MagicMock(), max_cycles=1)

        with patch.object(MetaAgent, "_plan_next_step", return_value="classification"):
            agent.run(wf, ctx)

        self.assertEqual(ctx.execution_history[-1]["step"], "classification")

    def test_gate_walks_the_chain_over_successive_cycles(self):
        """Repeated premature picks fill in the prerequisites in order."""
        wf = _chain_workflow()
        ctx = _context("extraction", "search", "selection_chunks")
        agent = MetaAgent(llm_service=None, tool_registry=MagicMock(), max_cycles=4)

        with patch.object(MetaAgent, "_plan_next_step", return_value="classification"):
            agent.run(wf, ctx)

        self.assertEqual(
            [h["step"] for h in ctx.execution_history],
            ["extraction", "search", "selection_chunks",
             "selection", "verify_keywords", "dk_collect", "classification"],
        )


if __name__ == "__main__":
    unittest.main()
