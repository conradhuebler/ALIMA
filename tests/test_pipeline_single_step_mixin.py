"""Real-drive tests for SingleStepExecutorMixin (F-15 split) - Claude Generated

The only other test touching ``execute_single_step`` mocks it away, so nothing
in the suite executed the moved bodies. These tests drive the real methods on a
real ``PipelineManager`` (LLM/DB mocked, ``execute_step`` stubbed), which in
particular executes the method-local ``PipelineStep`` import — the one
non-verbatim line of the split, invisible to the opcode comparison.
"""

import logging
import unittest
from unittest.mock import MagicMock

from src.core.pipeline_manager import PipelineManager, PipelineConfig
from src.core.data_models import KeywordAnalysisState


def _make_pm() -> PipelineManager:
    pm = PipelineManager(
        alima_manager=MagicMock(),
        cache_manager=MagicMock(),
        logger=logging.getLogger("test_single_step_mixin"),
    )
    pm.execute_step = MagicMock(return_value=True)
    pm.cache_manager.get_gnd_title_by_id.return_value = None
    return pm


class TestExecuteSingleStep(unittest.TestCase):
    def test_keywords_step_parses_existing_keywords_with_gnd_ids(self):
        pm = _make_pm()
        step = pm.execute_single_step(
            "keywords",
            PipelineConfig(),
            "Ein Abstract.\n\nExisting Keywords: Chemie (GND-ID: 4009816-3), Boden",
        )
        self.assertEqual(step.status, "completed")
        self.assertEqual(step.step_id, "keywords")
        self.assertEqual(
            pm.current_analysis_state.initial_keywords, ["Chemie", "Boden"]
        )
        self.assertEqual(pm.current_analysis_state.original_abstract, "Ein Abstract.")
        pm.execute_step.assert_called_once_with("keywords")

    def test_dk_classification_simulates_previous_steps(self):
        pm = _make_pm()
        step = pm.execute_single_step(
            "dk_classification",
            PipelineConfig(),
            "Abstract.\n\nExisting Keywords: kein DK-Format",
        )
        self.assertEqual(step.status, "completed")
        self.assertEqual(
            [s.step_id for s in pm.pipeline_steps],
            ["input", "dk_search", "dk_classification"],
        )
        self.assertEqual(pm.pipeline_steps[0].status, "completed")

    def test_failure_before_step_creation_returns_error_step(self):
        pm = _make_pm()
        pm.set_config = MagicMock(side_effect=RuntimeError("kaputt"))
        step = pm.execute_single_step("keywords", PipelineConfig(), "text")
        self.assertEqual(step.status, "error")
        self.assertEqual(step.error_message, "kaputt")


class TestResumePipelineFromState(unittest.TestCase):
    def test_resume_with_abstract_only(self):
        pm = _make_pm()
        kas = KeywordAnalysisState(
            original_abstract="abc",
            initial_keywords=[],
            search_suggesters_used=[],
            initial_gnd_classes=[],
            search_results=[],
            initial_llm_call_details=None,
            final_llm_analysis=None,
        )
        completed = pm.resume_pipeline_from_state(kas)
        self.assertEqual(completed, ["input"])
        self.assertEqual(pm.current_step_index, 1)
        self.assertEqual(pm.pipeline_steps[0].status, "completed")


if __name__ == "__main__":
    unittest.main()
