"""The pipeline thread owns the render subscription, not the request.

Claude Generated. ``run_analysis`` subscribes a ``_SessionBusSubscriber`` and
then runs the pipeline via ``asyncio.to_thread``. Its ``finally`` used to
unsubscribe — but that ``finally`` also fires when the awaiting task is
cancelled (client gone, server shutting down, TestClient tearing the request
down), while ``asyncio.to_thread`` keeps running. The pipeline then finished
with nobody rendering its bus events, and the session log stayed empty from
that point on.

Visible as the "flaky" ``test_agentic_analysis_emits_bus_tool_call_into_render_buffer``
(2 of 8 isolated runs red before the fix, 0 of 10 after): the very same race,
just with a shorter pipeline.

The interleaving is forced here rather than raced: the fake pipeline thread is
held open while the awaiting coroutine is cancelled, and the event it emits
afterwards must still be rendered.
"""
from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest import mock


class _Renderer:
    """Minimal renderer stand-in; records what the subscriber renders."""

    def __init__(self):
        self.calls: list = []

    def render_tool_call(self, name, args):
        self.calls.append(("tool_call", name))
        return "tc_1"

    def __getattr__(self, item):
        def _noop(*a, **k):
            return None
        return _noop


class PipelineThreadOwnsSubscriptionTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_cancelled_request_does_not_strip_the_running_pipeline(self):
        from src.core import state_bus
        from src.webapp.render_bridge import _SessionBusSubscriber
        from src.webapp.routers import analysis as analysismod

        state_bus.reset()
        renderer = _Renderer()
        subscriber = _SessionBusSubscriber(renderer)

        session = SimpleNamespace(
            status="idle", error_message=None, working_title=None,
            pipeline_manager_ref=None, current_analysis_state=None,
            render_buffer=[], cleanup=lambda: None, add_temp_file=lambda p: None,
            abort_requested=False, workflow=None,
        )

        in_pipeline = threading.Event()
        may_finish = threading.Event()
        seen_subscriptions: list = []

        class _FakePM:
            """Holds the pipeline open while the request is torn down."""

            def __init__(self, *a, **k):
                self.config = SimpleNamespace(
                    enable_agentic_mode=False, workflow_name=None,
                    global_provider_override=None, global_model_override=None,
                )
                self.current_analysis_state = None

            def set_config(self, cfg):
                self.config = cfg

            def set_callbacks(self, **cb):
                pass

            def set_interrupt_flag(self, *a, **k):
                pass

            def start_pipeline(self, text, input_type=None, input_source=None):
                in_pipeline.set()
                may_finish.wait(timeout=5)
                bus = state_bus.AlimaStateBus()
                seen_subscriptions.append(len(bus._subscriptions))
                bus.emit_event("tool.called", {"id": "t1", "name": "inline_tool",
                                               "arguments": {}})
                return "fake-id"

        async def _to_thread(fn, *a, **k):
            # Run the real ``execute_pipeline`` in a thread, then cancel the
            # awaiting coroutine while that thread is still in the pipeline.
            worker = threading.Thread(target=fn, args=a, kwargs=k, daemon=True)
            worker.start()
            in_pipeline.wait(timeout=5)
            raise asyncio.CancelledError()

        services = {k: mock.MagicMock() for k in (
            "config_manager", "alima_manager", "cache_manager", "llm_service",
            "prompt_service", "pipeline_manager")}

        with mock.patch.dict(analysismod.sessions, {"s1": session}, clear=False), \
             mock.patch.object(analysismod, "_build_session_renderer", return_value=renderer), \
             mock.patch.object(analysismod, "_SessionBusSubscriber", return_value=subscriber), \
             mock.patch.object(analysismod, "PipelineManager", _FakePM), \
             mock.patch.object(analysismod.asyncio, "to_thread", _to_thread), \
             mock.patch.object(analysismod, "AppContext") as ctx, \
             mock.patch.object(analysismod, "resolve_input_to_text", return_value="text"):
            ctx.return_value.get_services.return_value = services
            state_bus.set_direct_dispatch(True)
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await analysismod.run_analysis(
                        "s1", "text", "abc", None, None, workflow="alima_v51",
                    )

                # The request is gone; the pipeline is not. Its events must
                # still reach the renderer.
                may_finish.set()
                for _ in range(150):
                    if seen_subscriptions:
                        break
                    await asyncio.sleep(0.02)
            finally:
                state_bus.set_direct_dispatch(False)
                subscriber.unsubscribe()
                state_bus.reset()

        self.assertTrue(seen_subscriptions, "the pipeline thread never ran")
        self.assertGreater(
            seen_subscriptions[0], 0,
            "the cancelled request removed the subscription while the pipeline ran",
        )
        self.assertIn(("tool_call", "inline_tool"), renderer.calls)

    async def test_a_run_that_never_reaches_the_thread_cleans_up(self):
        """No leak in the other direction either.

        If the executor never picks the job up, nobody else would remove the
        handlers, and they would render into a dead session for the rest of the
        process' life.
        """
        from src.core import state_bus
        from src.webapp.render_bridge import _SessionBusSubscriber
        from src.webapp.routers import analysis as analysismod

        state_bus.reset()
        renderer = _Renderer()
        subscriber = _SessionBusSubscriber(renderer)
        session = SimpleNamespace(
            status="idle", error_message=None, working_title=None,
            pipeline_manager_ref=None, current_analysis_state=None,
            render_buffer=[], cleanup=lambda: None, add_temp_file=lambda p: None,
            abort_requested=False, workflow=None,
        )

        async def _to_thread(fn, *a, **k):
            raise asyncio.CancelledError()  # cancelled before the job starts

        with mock.patch.dict(analysismod.sessions, {"s2": session}, clear=False), \
             mock.patch.object(analysismod, "_build_session_renderer", return_value=renderer), \
             mock.patch.object(analysismod, "_SessionBusSubscriber", return_value=subscriber), \
             mock.patch.object(analysismod.asyncio, "to_thread", _to_thread), \
             mock.patch.object(analysismod, "AppContext"), \
             mock.patch.object(analysismod, "resolve_input_to_text", return_value="text"):
            with self.assertRaises(asyncio.CancelledError):
                await analysismod.run_analysis(
                    "s2", "text", "abc", None, None, workflow="alima_v51",
                )

        self.assertEqual(len(state_bus.AlimaStateBus()._subscriptions), 0,
                         "the subscriber leaked onto the shared bus")
        state_bus.reset()


if __name__ == "__main__":
    unittest.main()
