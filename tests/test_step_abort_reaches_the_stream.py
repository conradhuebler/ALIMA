"""Der Step-Abbruch muss in der laufenden Generierung landen. Claude Generated.

Der klassische Pfad prüft den Abbruch pro Token und schließt den Generator
(``AlimaManager``). Der agentische reichte gar kein ``should_stop`` durch:
``LLMAgentStep`` baute den ``AgentLoop`` ohne, und damit sah keine der
Provider-Schleifen den Abbruch — er wirkte frühestens, wenn der Schritt von
selbst fertig war. Diese Tests pinnen die Kette
PipelineManager → WorkflowExecutor/MetaAgent → Step → AgentLoop und die
Einmal-Semantik des Ereignisses.
"""
from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class _Manager:
    """The abort surface of PipelineManager, without building the manager."""

    def __init__(self):
        from src.core.pipeline_manager import PipelineManager

        self._abort_step_event = threading.Event()
        self._interrupt_lock = threading.Lock()
        self._interrupt_check_func = None
        self.logger = MagicMock()
        self._step_abort_once = PipelineManager._step_abort_once.__get__(self)
        self.agentic_should_stop = PipelineManager.agentic_should_stop.__get__(self)
        self.abort_current_step = PipelineManager.abort_current_step.__get__(self)


class AbortSemantics(unittest.TestCase):
    def test_quiet_by_default(self):
        self.assertFalse(_Manager().agentic_should_stop())

    def test_an_abort_fires_once_and_clears_itself(self):
        manager = _Manager()
        manager.abort_current_step()
        self.assertTrue(manager.agentic_should_stop())
        # Single-shot: the NEXT call must run normally, or the abort would kill
        # every remaining step of the workflow.
        self.assertFalse(manager.agentic_should_stop())

    def test_a_worker_stop_keeps_firing(self):
        manager = _Manager()
        manager._interrupt_check_func = lambda: True
        self.assertTrue(manager.agentic_should_stop())
        self.assertTrue(manager.agentic_should_stop())

    def test_the_classic_path_consumes_the_same_event(self):
        from src.core.pipeline_manager import PipelineManager

        manager = _Manager()
        manager.alima_manager = MagicMock()
        PipelineManager.set_interrupt_flag(manager, None, lambda: False)
        combined = manager.alima_manager.set_interrupt_callback.call_args[0][0]
        manager.abort_current_step()
        self.assertTrue(combined())
        self.assertFalse(combined())


class WiringToTheLoop(unittest.TestCase):
    """Every layer between the manager and the provider loop has to carry it."""

    def _config(self, step_type="llm_agent"):
        from src.core.agents.steps.base_step import StepConfig

        return StepConfig(id="s", type=step_type, raw={"function": "noop"})

    def test_the_executor_hands_it_to_the_step(self):
        from src.core.agents.workflow_executor import WorkflowExecutor
        from src.core.agents.workflow_loader import WorkflowDef

        marker = lambda: True
        executor = WorkflowExecutor(should_stop=marker)
        seen = {}

        class _Step:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def execute(self, context):
                return SimpleNamespace(step_id="s", success=True, data={}, error=None,
                                       duration_seconds=0.0, iterations=1, tool_calls=0)

        workflow = WorkflowDef(name="w", version="1", steps=[self._config()])
        with patch("src.core.agents.workflow_executor.get_step_class", return_value=_Step):
            executor.run(workflow, SimpleNamespace(step_results={}, extra={}))
        self.assertIs(seen.get("should_stop"), marker)

    def test_the_step_hands_it_to_the_agent_loop(self):
        from src.core.agents.steps.llm_agent_step import LLMAgentStep

        marker = lambda: True
        step = LLMAgentStep(
            config=self._config(), llm_service=object(), should_stop=marker
        )
        captured = {}

        class _Loop:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, **kwargs):
                from src.core.data_models import AgentResult

                return AgentResult(content="{}")

        context = SimpleNamespace(
            abstract="a", extra={}, step_results={}, temperature=0.1, seed=None,
            top_p=0.9, max_tokens=100, provider="p", model="m",
        )
        with patch("src.core.agents.steps.llm_agent_step.AgentLoop", _Loop):
            try:
                step.run(context)
            except Exception:
                pass  # the fake answer may not satisfy the step's parsing
        self.assertIs(captured.get("should_stop"), marker)

    def test_the_meta_agent_forwards_it(self):
        from src.core.agents.meta_agent import MetaAgent

        marker = lambda: True
        agent = MetaAgent(should_stop=marker)
        self.assertIs(agent.should_stop, marker)
        self.assertIs(agent.executor.should_stop, marker)

    def test_a_composite_step_does_not_swallow_it(self):
        import inspect

        from src.core.agents.steps import composite_step

        source = inspect.getsource(composite_step)
        self.assertIn("should_stop=self.should_stop", source)

    def test_the_agentic_pipeline_passes_the_managers_check(self):
        import inspect

        from src.core import _pipeline_agentic

        source = inspect.getsource(_pipeline_agentic)
        # Both runners — with and without the MetaAgent loop.
        self.assertEqual(source.count("should_stop=self.agentic_should_stop"), 2)


class StoppingMidStream(unittest.TestCase):
    """The point of the wiring: the generation stops where it is."""

    def test_the_provider_loop_polls_it_per_chunk(self):
        """A stand-in for the provider loops, driven by the real manager check."""
        manager = _Manager()
        should_stop = manager.agentic_should_stop
        produced = []
        for index, chunk in enumerate("abcdefghij"):
            if index == 3:
                manager.abort_current_step()  # operator presses the button
            if should_stop():
                break
            produced.append(chunk)
        self.assertEqual("".join(produced), "abc")

    def test_the_next_call_is_not_affected(self):
        manager = _Manager()
        should_stop = manager.agentic_should_stop
        manager.abort_current_step()
        first = [c for c in "abc" if not should_stop()]
        second = [c for c in "abc" if not should_stop()]
        self.assertEqual(first, ["b", "c"])
        self.assertEqual(second, ["a", "b", "c"])


class TheStreamIsClosedNotJustAbandoned(unittest.TestCase):
    """Abbrechen heißt: die Anfrage zumachen.

    Nur aus der Schleife zu springen lässt die HTTP-Antwort offen, das Backend
    generiert weiter und die Antwort kostet trotzdem. - Claude Generated
    """

    class _Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            for _ in range(50):
                yield {"message": {"content": "x"}}

        def close(self):
            self.closed = True

    def _service(self, stream):
        from src.llm.llm_service import LlmService

        service = LlmService.__new__(LlmService)
        service.logger = MagicMock()
        service.clients = {"ollama": SimpleNamespace(chat=lambda **kw: stream)}
        return service

    def test_ollama_closes_its_stream_on_abort(self):
        from src.core.data_models import StopReason
        from src.llm.llm_service import LlmService

        stream = self._Stream()
        service = self._service(stream)
        calls = {"n": 0}

        def should_stop():
            calls["n"] += 1
            return calls["n"] > 3  # let three chunks through, then abort

        result = LlmService._generate_ollama_native_with_tools(
            service, "ollama", "m", [{"role": "user", "content": "hi"}], [],
            0.1, 0.9, 100, None, lambda _t: None, should_stop=should_stop,
        )
        self.assertEqual(result.stop_reason, StopReason.CANCELLED)
        self.assertTrue(stream.closed, "der Stream blieb offen — das Backend generiert weiter")
        self.assertEqual(result.content, "xxx")  # exactly what arrived before the abort

    def test_every_in_stream_abort_closes_or_leaves_a_context_manager(self):
        """The other providers, by inspection — they differ in mechanism.

        OpenAI closes explicitly, Anthropic returns out of a ``with`` block
        (which closes), Gemini/text-fallback check only after the call.
        """
        import inspect

        from src.llm import llm_service

        source = inspect.getsource(llm_service)
        # Both explicit closers are present …
        self.assertEqual(source.count("best-effort close on user abort"), 2)
        # … and the Anthropic stream is scoped by a context manager.
        self.assertIn("with self.clients[\"anthropic\"].messages.stream(**params) as stream:", source)


if __name__ == "__main__":
    unittest.main()
