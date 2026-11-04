"""Zeitmessung eines Laufs nach Art der Arbeit (utils.run_timing). Claude Generated."""

from __future__ import annotations

import threading
import time
import unittest

from src.utils import run_timing as rt


class ExclusiveAccountingTest(unittest.TestCase):
    def setUp(self):
        rt.begin_run()

    def test_nested_time_is_carved_out_of_the_outer_block(self):
        with rt.timed("retrieval"):
            with rt.timed("llm"):
                time.sleep(0.3)
        s = rt.finish_run()
        self.assertAlmostEqual(s["llm_s"], 0.3, delta=0.05)
        self.assertLess(s["retrieval_s"], 0.1)                        # nur der Rest, nicht 0,3
        self.assertEqual(s["calls"], {"llm": 1, "decision": 0, "retrieval": 1})

    def test_the_remainder_is_other_and_the_parts_add_up_to_the_total(self):
        time.sleep(0.05)
        with rt.timed("decision"):
            time.sleep(0.05)
        s = rt.finish_run()
        self.assertGreaterEqual(s["other_s"], 0.0)
        self.assertAlmostEqual(s["decision_s"] + s["other_s"] + s["llm_s"] + s["retrieval_s"], s["total_s"], delta=0.2)
        self.assertFalse(s["parallel"])

    def test_begin_run_resets(self):
        with rt.timed("llm"):
            time.sleep(0.02)
        rt.begin_run()
        self.assertEqual(rt.finish_run()["calls"]["llm"], 0)

    def test_an_exception_still_counts_and_propagates(self):
        with self.assertRaises(RuntimeError):
            with rt.timed("llm"):
                raise RuntimeError("x")
        self.assertEqual(rt.finish_run()["calls"]["llm"], 1)

    def test_unknown_category_is_refused(self):
        with self.assertRaises(ValueError):
            with rt.timed("netz"):
                pass

    def test_threads_are_added_and_flagged_as_parallel(self):
        def work():
            with rt.timed("decision"):
                time.sleep(0.3)

        ts = [threading.Thread(target=work) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        s = rt.finish_run()
        self.assertGreater(s["decision_s"], s["total_s"] + 0.2)       # 3 x 0,3 s in etwa 0,3 s
        self.assertTrue(s["parallel"])
        self.assertIn("parallele", rt.format_summary(s))

    def test_format_names_all_parts(self):
        text = rt.format_summary(rt.finish_run())
        for word in ("Gesamt", "LLM", "Entscheidung", "Abruf", "Sonstiges"):
            self.assertIn(word, text)

    def test_decorator_keeps_the_signature(self):
        import inspect

        @rt.timed_method("llm")
        def f(a, b=2):
            return a + b

        self.assertEqual(f(1, b=3), 4)
        self.assertEqual(list(inspect.signature(f).parameters), ["a", "b"])


class HooksTest(unittest.TestCase):
    """Die Messpunkte liegen dort, wo die Zeit anfällt."""

    def test_llm_service_generate_methods_are_timed(self):
        from src.llm.llm_service import LlmService

        for name in ("generate_response", "generate_with_tools"):
            self.assertTrue(hasattr(getattr(LlmService, name), "__wrapped__"), name)

    def test_decision_call_is_timed(self):
        from src.utils import decision_client

        self.assertTrue(hasattr(decision_client.systemone, "__wrapped__"))

    def test_tool_execution_counts_as_retrieval_and_a_model_inside_it_does_not(self):
        from src.mcp.tool_registry import ToolRegistry

        reg = ToolRegistry()

        def handler():
            with rt.timed("llm"):
                time.sleep(0.15)
            return "{}"

        reg._handlers["slow"] = handler
        rt.begin_run()
        reg.execute("slow", {})
        s = rt.finish_run()
        self.assertEqual(s["calls"]["retrieval"], 1)
        self.assertEqual(s["calls"]["llm"], 1)
        self.assertLess(s["retrieval_s"], 0.1)

    def test_deterministic_workflow_steps_count_as_retrieval(self):
        import yaml
        from unittest.mock import MagicMock

        from src.core.agents.steps.base_step import StepResult
        from src.core.agents.workflow_executor import WorkflowExecutor
        from src.core.agents.workflow_loader import WorkflowDef, StepConfig

        step_cfg = StepConfig(id="s1", type="deterministic", inputs={}, outputs={}, raw={})
        wf = WorkflowDef(name="t", version="1", description="", steps=[step_cfg])

        class _Step:
            def __init__(self, *a, **k):
                pass

            def execute(self, ctx):
                time.sleep(0.05)
                return StepResult(step_id="s1", success=True, outputs={}, duration_seconds=0.05)

        from unittest.mock import patch
        rt.begin_run()
        with patch("src.core.agents.workflow_executor.get_step_class", return_value=_Step):
            try:
                WorkflowExecutor(llm_service=MagicMock(), tool_registry=MagicMock()).run(wf, MagicMock())
            except Exception:
                pass  # nur die Zählung interessiert
        self.assertEqual(rt.finish_run()["calls"]["retrieval"], 1)

    def test_the_pipeline_manager_wires_begin_and_finish(self):
        import pathlib

        src = pathlib.Path("src/core/pipeline_manager.py").read_text(encoding="utf-8")
        self.assertIn("run_timing.begin_run()", src)
        self.assertIn('("input", "search", "dk_search")', src)
        self.assertIn("self._finish_run_timing()", src)
        ag = pathlib.Path("src/core/_pipeline_agentic.py").read_text(encoding="utf-8")
        self.assertIn("self._finish_run_timing()", ag)


if __name__ == "__main__":
    unittest.main()
