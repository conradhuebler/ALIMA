# tests/test_error_visibility.py
"""Negative-path tests: failures must be visible, never silent. Claude Generated.

Covers the WP A ("Fehler sichtbar machen") guarantees:
- WorkflowExecutor survives a raising step and reports it as a failed StepResult.
- A broken `when:` condition is reported as a step failure, not silently skipped.
- An unparseable LLM response fails the initialisation step instead of
  continuing the pipeline with 0 keywords.
- SWB network failures are recorded in `last_errors` and are NOT cached
  as "no results".
- A failed classic pipeline step stops the pipeline (no auto-advance past it)
  and notifies the step_error callback exactly once.
"""

import logging
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.agents.steps.base_step import BaseStep, StepConfig, StepResult
from src.core.agents.workflow_executor import WorkflowExecutor
from src.core.agents.workflow_loader import WorkflowDef
from src.core.data_models import AbstractData, AnalysisResult, PromptConfigData, TaskState
from src.core.processing_utils import extract_keywords_from_response
from src.utils.pipeline_utils import PipelineStepExecutor

logging.disable(logging.CRITICAL)


class _RaisingStep(BaseStep):
    """Step whose execute() escapes with an exception.

    BaseStep.execute already converts run() exceptions into failed
    StepResults; overriding execute models a step that bypasses that
    safety net — the executor must survive it.
    """

    def run(self, context):  # pragma: no cover - never reached
        return {}

    def execute(self, context):
        raise RuntimeError("boom")


class _PlainContext:
    """Minimal context object; attribute assignment must work."""

    def to_dict(self):
        return {}


class TestWorkflowExecutorErrorPaths(unittest.TestCase):
    def test_raising_step_yields_failed_result_not_crash(self):
        workflow = WorkflowDef(
            name="t", version="4", steps=[StepConfig(id="s1", type="raising")]
        )
        executor = WorkflowExecutor()
        with patch(
            "src.core.agents.workflow_executor.get_step_class",
            return_value=_RaisingStep,
        ):
            report = executor.run(workflow, _PlainContext())

        self.assertFalse(report.success)
        self.assertEqual(len(report.step_results), 1)
        result = report.step_results[0]
        self.assertFalse(result.success)
        self.assertIn("RuntimeError", result.error)
        self.assertIn("boom", result.error)

    def test_broken_condition_is_reported_as_step_failure(self):
        workflow = WorkflowDef(
            name="t",
            version="4",
            steps=[StepConfig(id="s1", type="raising", condition="kaputt >")],
        )
        executor = WorkflowExecutor()
        with patch(
            "src.core.agents.workflow_executor.ConditionalEngine.evaluate",
            side_effect=ValueError("bad expression"),
        ):
            report = executor.run(workflow, _PlainContext())

        self.assertFalse(report.success)
        self.assertEqual(len(report.step_results), 1)
        result = report.step_results[0]
        self.assertFalse(result.success)
        self.assertIn("bad expression", result.error)


class TestUnparseableLlmResponseFailsStep(unittest.TestCase):
    """An LLM response without extractable keywords must fail the
    initialisation step instead of returning an empty 'success'."""

    def setUp(self):
        self.mock_alima_manager = Mock()
        self.mock_cache_manager = Mock()
        self.mock_logger = Mock()
        self.mock_logger.level = 100
        self.executor = PipelineStepExecutor(
            alima_manager=self.mock_alima_manager,
            cache_manager=self.mock_cache_manager,
            logger=self.mock_logger,
        )

    def _task_state(self, full_text: str) -> TaskState:
        return TaskState(
            abstract_data=AbstractData(abstract="abc", keywords=""),
            analysis_result=AnalysisResult(
                full_text=full_text, matched_keywords={}, gnd_systematic=""
            ),
            prompt_config=PromptConfigData(
                prompt="p", system="s", temp=0.7, p_value=0.9,
                models=["m"], seed=42,
            ),
            status="completed",
            task_name="initialisation",
            model_used="m",
            provider_used="p",
        )

    def test_unparseable_response_raises(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._task_state(
            "Sorry, as an AI model I cannot comply with this request."
        )
        with self.assertRaises(ValueError) as ctx:
            self.executor.execute_initial_keyword_extraction(
                abstract_text="abc", model="m", provider="p", task="initialisation"
            )
        self.assertIn("could not be parsed", str(ctx.exception))

    def test_parseable_response_still_succeeds(self):
        self.mock_alima_manager.analyze_abstract.return_value = self._task_state(
            "<final_list>Limnologie | Seenkunde</final_list><class>31</class>"
        )
        keywords, _classes, _analysis, _title = (
            self.executor.execute_initial_keyword_extraction(
                abstract_text="abc", model="m", provider="p", task="initialisation"
            )
        )
        self.assertIn("Limnologie", keywords)


class TestExtractKeywordsFromResponse(unittest.TestCase):
    def test_garbage_returns_empty_string(self):
        self.assertEqual(extract_keywords_from_response("no structure at all"), "")

    def test_final_list_is_extracted(self):
        out = extract_keywords_from_response("<final_list>A | B</final_list>")
        self.assertEqual(out, "A, B")


class TestSwbNetworkFailureNotCached(unittest.TestCase):
    """SWB outages must surface in last_errors and must never be persisted
    in the cache as 'no results for this term'."""

    def _make_suggester(self, tmpdir):
        from src.utils.suggesters.swb_suggester import SWBSuggester

        return SWBSuggester(data_dir=tmpdir, debug=False)

    def test_network_error_recorded_and_not_cached(self):
        import requests

        with tempfile.TemporaryDirectory() as tmpdir:
            suggester = self._make_suggester(tmpdir)
            with patch(
                "src.utils.suggesters.swb_suggester.requests.get",
                side_effect=requests.exceptions.ConnectionError("API down"),
            ):
                results = suggester.search(["Limnologie"])

            self.assertEqual(results["Limnologie"], {})
            self.assertIn("Limnologie", suggester.last_errors)
            self.assertIn("API down", suggester.last_errors["Limnologie"])
            # Outage must not be persisted as an (empty) cached result
            self.assertNotIn("Limnologie", suggester.cache)

    def test_successful_search_has_no_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suggester = self._make_suggester(tmpdir)
            fake_response = Mock()
            fake_response.text = "<html>nothing relevant</html>"
            fake_response.raise_for_status = Mock()
            with patch(
                "src.utils.suggesters.swb_suggester.requests.get",
                return_value=fake_response,
            ):
                suggester.search(["Limnologie"])
            self.assertEqual(suggester.last_errors, {})


class TestPipelineStopsOnFailedStep(unittest.TestCase):
    """_execute_next_step must stop on a failed step (no auto-advance) and
    notify step_error exactly once."""

    def _make_manager(self, execute_step):
        from src.core.pipeline_manager import PipelineManager, PipelineStep

        pm = PipelineManager.__new__(PipelineManager)  # bypass __init__
        pm.logger = logging.getLogger(__name__)
        pm.pipeline_steps = [
            PipelineStep(step_id="initialisation", name="Init", status="pending"),
            PipelineStep(step_id="search", name="Search", status="pending"),
        ]
        pm.current_step_index = 0
        pm.config = SimpleNamespace(auto_advance=True)
        pm.current_analysis_state = None
        pm.step_started_callback = None
        pm.step_completed_callback = None
        pm.pipeline_completed_callback = None
        pm.step_error_callback = Mock()
        pm._check_interruption = lambda: None
        pm._emit_pipeline_step_bus = Mock()
        pm.execute_step = execute_step
        return pm

    def test_graceful_false_stops_pipeline_and_notifies(self):
        pm = self._make_manager(execute_step=lambda step_id: False)
        pm._execute_next_step()

        self.assertEqual(pm.pipeline_steps[0].status, "error")
        pm.step_error_callback.assert_called_once()
        # No auto-advance past the failed step
        self.assertEqual(pm.current_step_index, 0)
        self.assertEqual(pm.pipeline_steps[1].status, "pending")
        pm._emit_pipeline_step_bus.assert_any_call(pm.pipeline_steps[0], "error")

    def test_exception_path_does_not_double_notify(self):
        # Simulate execute_step's own exception handling: it sets status=error,
        # notifies the callback itself and returns False.
        def execute_step(step_id):
            step = pm.pipeline_steps[0]
            step.status = "error"
            step.error_message = "LLM unreachable"
            pm.step_error_callback(step, step.error_message)
            return False

        pm = self._make_manager(execute_step=execute_step)
        pm._execute_next_step()

        # Exactly one notification (from execute_step), none from the else-branch
        self.assertEqual(pm.step_error_callback.call_count, 1)
        self.assertEqual(pm.current_step_index, 0)


if __name__ == "__main__":
    unittest.main()
