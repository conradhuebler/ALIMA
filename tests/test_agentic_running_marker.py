"""Agentic steps announce "running" so front ends can follow the agent. Claude Generated"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.core.agents.workflow_executor import WorkflowExecutor


def test_running_emits_a_small_marker_not_the_context():
    seen = []
    ex = WorkflowExecutor.__new__(WorkflowExecutor)
    ex.context_callback = lambda sid, snap: seen.append((sid, snap))
    cfg = SimpleNamespace(id="dk_collect", type="deterministic", description="d")
    ex._emit_snapshot(MagicMock(), cfg, status="running")
    assert seen == [("dk_collect", {
        "_step_id": "dk_collect", "_step_type": "deterministic",
        "_step_description": "d", "_step_status": "running"})]


def test_qt_tab_jumps_to_the_running_steps_tab():
    from src.ui._pipeline_tab_events import PipelineTabEventsMixin
    from src.ui.pipeline_tab import PipelineTab

    tab = MagicMock()
    tab._AGENTIC_STEP_MAP = PipelineTab._AGENTIC_STEP_MAP
    widget = MagicMock()
    tab.step_widgets = {"dk_search": widget}
    PipelineTabEventsMixin._on_agentic_step_snapshot(
        tab, "dk_collect", {"_step_status": "running"})
    tab.jump_to_step.assert_called_once_with("dk_search")
    assert widget.step.status == "running"
