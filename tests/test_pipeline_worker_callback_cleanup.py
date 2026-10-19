"""PipelineWorker never restored PipelineManager's callbacks after finishing.

SingleStepWorker already guards against exactly this for stream_callback
alone (save old value, restore in finally - "so no other tab inherits our
signal"). PipelineWorker set all seven callbacks via set_callbacks() but
never reverted them, so pm.step_started_callback/step_completed_callback/
etc. stayed bound to THIS (now-finished) worker's Qt signals forever.

Consequence: the chat agent's run_pipeline/rerun_step tools
(src/ui/chat_tools/pipeline.py) run pm.start_pipeline()/execute_single_step()
synchronously on the ChatAgentWorker thread, and their _StatusForwarder
explicitly preserves-and-forwards "whatever callback is already registered"
so it doesn't stomp on an active consumer. Fed a stale dead-worker callback
instead of None, it re-emitted that worker's Qt signals during a run that
had nothing to do with it - firing this tab's on_step_started/
on_step_completed slots (status label, tab jump, the pipeline "Stop" button
visibility, ...) alongside the chat's own cancel button. From the user's
side: two cancel buttons where only one made sense, and the stale
"PipelineWorker.request_stop()"-wired one did nothing since no
PipelineWorker was actually running. - Claude Generated
"""
from __future__ import annotations

import unittest

from PyQt6.QtWidgets import QApplication

from src.ui.workers import PipelineWorker


def _qapp(keep: list):
    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class _FakePipelineManager:
    """Mimics only what PipelineWorker.run() touches on PipelineManager."""

    def __init__(self):
        self.step_started_callback = None
        self.step_completed_callback = None
        self.step_error_callback = None
        self.pipeline_completed_callback = None
        self.stream_callback = None
        self.repetition_detected_callback = None
        self.agentic_context_callback = None
        self.started_with = None

    def set_callbacks(self, step_started=None, step_completed=None, step_error=None,
                       pipeline_completed=None, stream_callback=None,
                       repetition_detected=None, agentic_context=None):
        self.step_started_callback = step_started
        self.step_completed_callback = step_completed
        self.step_error_callback = step_error
        self.pipeline_completed_callback = pipeline_completed
        self.stream_callback = stream_callback
        self.repetition_detected_callback = repetition_detected
        self.agentic_context_callback = agentic_context

    def set_interrupt_flag(self, *_a, **_kw):
        pass

    def start_pipeline(self, *_a, **_kw):
        self.started_with = (self.step_started_callback, self.stream_callback)
        return "fake-pipeline-id"


class PipelineWorkerRestoresCallbacksTest(unittest.TestCase):
    _kept: list = []

    def test_pre_existing_callbacks_are_restored_after_a_successful_run(self):
        _qapp(self._kept)
        pm = _FakePipelineManager()
        sentinel_started = object()
        sentinel_stream = object()
        pm.step_started_callback = sentinel_started
        pm.stream_callback = sentinel_stream

        worker = PipelineWorker(pm, "abstract text")
        worker.run()

        # During the run, the worker's OWN emit methods must have been in
        # place - otherwise step_started/stream_token signals never reach
        # the UI in the first place.
        self.assertIsNotNone(pm.started_with)
        self.assertIsNot(pm.started_with[0], sentinel_started)
        self.assertIsNot(pm.started_with[1], sentinel_stream)

        # ... but afterwards, the pre-existing callbacks must be back.
        self.assertIs(pm.step_started_callback, sentinel_started)
        self.assertIs(pm.stream_callback, sentinel_stream)

    def test_callbacks_are_restored_even_when_start_pipeline_raises(self):
        _qapp(self._kept)
        pm = _FakePipelineManager()
        sentinel_completed = object()
        pm.step_completed_callback = sentinel_completed

        def _boom(*_a, **_kw):
            raise RuntimeError("boom")

        pm.start_pipeline = _boom
        worker = PipelineWorker(pm, "abstract text")
        worker.run()

        self.assertIs(pm.step_completed_callback, sentinel_completed)

    def test_no_pre_existing_callback_restores_to_none(self):
        """The common case: nothing registered before this run - restoring
        must leave None, not the worker's own emit method."""
        _qapp(self._kept)
        pm = _FakePipelineManager()

        worker = PipelineWorker(pm, "abstract text")
        worker.run()

        self.assertIsNone(pm.step_started_callback)
        self.assertIsNone(pm.stream_callback)


if __name__ == "__main__":
    unittest.main()
