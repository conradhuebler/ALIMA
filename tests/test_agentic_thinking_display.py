"""Thinking of an agentic step is shown like the chat's - Claude Generated.

The chat has streamed its reasoning into a live 💭 block since the render-layer
work; the agentic pipeline showed nothing at all, because ``LLMAgentStep`` built
its ``AgentLoop`` without an ``on_thinking`` sink (and the loop leaves the stream
untouched without one). The step now emits ``llm.thinking`` per chunk and
``llm.thinking_done`` at the end of its turn; both StateBus consumers (GUI
mixin, webapp subscriber) map them onto the same renderer methods.
"""

import unittest
from unittest.mock import MagicMock

from src.core.agents.shared_context import SharedContext
from src.core.agents.steps.base_step import StepConfig
from src.core.agents.steps.llm_agent_step import LLMAgentStep
from src.core.data_models import AgentResponse, StopReason


class TestStepEmitsThinking(unittest.TestCase):
    """Producer side: the step puts the reasoning channel on the bus."""

    def setUp(self):
        from src.core.state_bus import AlimaStateBus, reset

        reset()
        self.bus = AlimaStateBus()
        self.seen = []
        self.bus.subscribe("llm.thinking", lambda p: self.seen.append(("chunk", p.get("text"))))
        self.bus.subscribe("llm.thinking_done", lambda p: self.seen.append(("done", p.get("step_id"))))

    def tearDown(self):
        from src.core.state_bus import reset

        reset()

    def _run_step(self, chunks):
        def _generate(**kw):
            cb = kw.get("thinking_callback")
            if cb:
                for c in chunks:
                    cb(c)
            return AgentResponse(
                content='{"keywords": ["k"]}', tool_calls=[], stop_reason=StopReason.END_TURN
            )

        svc = MagicMock()
        svc.generate_with_tools.side_effect = _generate
        cfg = StepConfig(
            id="extraction",
            type="llm_agent",
            raw={"llm": {"max_iterations": 1}, "system_prompt": "s", "user_prompt": "u", "tools": []},
        )
        LLMAgentStep(cfg, llm_service=svc, tool_registry=MagicMock()).run(
            SharedContext(abstract="a", initial_keywords=[])
        )

    def test_chunks_and_the_closing_signal_reach_the_bus(self):
        self._run_step(["den", "ken"])
        self.assertEqual(
            self.seen, [("chunk", "den"), ("chunk", "ken"), ("done", "extraction")]
        )

    def test_a_turn_without_reasoning_still_closes(self):
        """The done signal is unconditional: it must fold away a block that an
        earlier turn of the same step left open."""
        self._run_step([])
        self.assertEqual(self.seen, [("done", "extraction")])


class TestWebappSubscriberRendersThinking(unittest.TestCase):
    """Consumer side (webapp): bus → render events."""

    def setUp(self):
        from src.core.render_events import MockTransport
        from src.core.state_bus import AlimaStateBus, reset
        from src.ui.unified_message_renderer import UnifiedMessageRenderer
        from src.webapp.app import _SessionBusSubscriber

        reset()
        self.transport = MockTransport()
        self.renderer = UnifiedMessageRenderer(self.transport, MagicMock(isChecked=lambda: True))
        self.sub = _SessionBusSubscriber(self.renderer)
        self.bus = AlimaStateBus()
        self.sub.subscribe()

    def tearDown(self):
        from src.core.state_bus import reset

        self.sub.unsubscribe()
        reset()

    def test_block_opens_expanded_streams_and_folds_away(self):
        self.bus.emit_event("llm.thinking", {"text": "erst denken, "})
        self.bus.emit_event("llm.thinking", {"text": "dann antworten"})
        self.bus.emit_event("llm.thinking_done", {"step_id": "extraction"})

        opens = self.transport.of_type("collapsible")
        self.assertTrue(opens, "no 💭 block was opened")
        self.assertTrue(opens[0]["open"], "the block must open expanded, like the chat's")
        self.assertEqual(opens[0].get("kind"), "thinking")

        streamed = "".join(
            e["text"] for e in self.transport.of_type("collapsible_append")
        )
        updates = self.transport.of_type("collapsible_update")
        body = updates[-1]["body"] if updates else ""
        self.assertIn("erst denken", streamed + body)
        self.assertIn("dann antworten", streamed + body)

        self.assertFalse(updates[-1].get("open", True), "the block must fold away at the end")

    def test_empty_chunks_open_nothing(self):
        self.bus.emit_event("llm.thinking", {"text": ""})
        self.assertEqual(self.transport.of_type("collapsible"), [])


class TestGuiMixinMatchesTheWebappHandler(unittest.TestCase):
    """The two StateBus consumers must stay in lockstep (see _chat_panel_bus)."""

    def _panel(self):
        from src.ui._chat_panel_bus import BusEventMixin

        class _Panel(BusEventMixin):
            def __init__(self):
                self._renderer = MagicMock()
                self.logger = MagicMock()

        return _Panel()

    def test_chunk_goes_to_append_thinking(self):
        panel = self._panel()
        panel._on_bus_thinking({"text": "denken"})
        panel._renderer.append_thinking.assert_called_once_with("denken")

    def test_empty_chunk_is_ignored(self):
        panel = self._panel()
        panel._on_bus_thinking({"text": ""})
        panel._renderer.append_thinking.assert_not_called()

    def test_done_folds_the_block_away(self):
        panel = self._panel()
        panel._on_bus_thinking_done({})
        panel._renderer.close_thinking.assert_called_once()

    def test_panel_subscribes_to_both_events(self):
        """The handlers are wired in PipelineChatPanel.__init__, which needs a
        QApplication to build — read the source instead (same approach as
        tests/test_state_bus.py)."""
        import inspect

        from src.ui import pipeline_chat_panel

        source = inspect.getsource(pipeline_chat_panel)
        self.assertIn('bus.subscribe("llm.thinking", self._on_bus_thinking)', source)
        self.assertIn(
            'bus.subscribe("llm.thinking_done", self._on_bus_thinking_done)', source
        )


if __name__ == "__main__":
    unittest.main()
