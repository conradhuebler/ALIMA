"""P-ι — Headless agent driver + gateway tests. Claude Generated.

Covers:

1. ``StdinProposalGateway`` accept / decline / non-interactive branches.
2. ``AutoRejectGateway`` always rejects.
3. ``_resolve_kb_manager`` falls back to ``cache_manager``.
4. ``HeadlessAgentRunner.run`` drives ``AgentLoop`` headless (no QApplication):
   immediate final answer + a tool-calling round-trip.
5. ``StoppableAgentThread`` exposes ``_stop_event`` and ``request_stop``.

No PyQt6 QApplication is created — the whole point of P-ι is that this path
runs without a GUI.
"""
from __future__ import annotations

import io
import threading
import unittest
from types import SimpleNamespace

from src.core.data_models import AgentResponse, ToolCall, StopReason
from src.core.headless_agent import (
    HeadlessAgentRunner,
    StoppableAgentThread,
    resolve_provider_model,
    _resolve_kb_manager,
)
from src.utils.config_models import ChatConfig
from src.core.headless_gateway import StdinProposalGateway, AutoRejectGateway


class _FakeTtyStdin:
    """Minimal stdin double that reports as a tty and returns a scripted line."""

    def __init__(self, line: str):
        self._line = line

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return self._line


class _ScriptedLLM:
    """LlmService double: returns a scripted sequence of AgentResponses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.cancelled_reason = None

    def generate_with_tools(self, **kwargs):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]

    def cancel_generation(self, reason: str = ""):
        self.cancelled_reason = reason


class TestStdinProposalGateway(unittest.TestCase):
    def test_accept(self):
        g = StdinProposalGateway(stream=io.StringIO(), stdin=_FakeTtyStdin("y\n"))
        out = g.request_decision(1, "run_pipeline", {"mode": "classic"})
        self.assertTrue(out["accepted"])

    def test_accept_ja(self):
        g = StdinProposalGateway(stream=io.StringIO(), stdin=_FakeTtyStdin("ja\n"))
        self.assertTrue(g.request_decision(1, "t", {})["accepted"])

    def test_decline(self):
        g = StdinProposalGateway(stream=io.StringIO(), stdin=_FakeTtyStdin("n\n"))
        out = g.request_decision(1, "propose_dk_change", {"code": "004"})
        self.assertFalse(out["accepted"])
        self.assertEqual(out["reject_reason"], "user_declined")

    def test_non_interactive_pipe(self):
        # StringIO.isatty() is False → never prompts, fail-safe reject.
        g = StdinProposalGateway(stream=io.StringIO(), stdin=io.StringIO("y\n"))
        out = g.request_decision(1, "t", {})
        self.assertFalse(out["accepted"])
        self.assertEqual(out["reject_reason"], "non_interactive")

    def test_eof(self):
        g = StdinProposalGateway(stream=io.StringIO(), stdin=_FakeTtyStdin(""))
        out = g.request_decision(1, "t", {})
        self.assertFalse(out["accepted"])
        self.assertEqual(out["reject_reason"], "non_interactive")


class TestAutoRejectGateway(unittest.TestCase):
    def test_always_rejects(self):
        out = AutoRejectGateway().request_decision(1, "run_pipeline", {})
        self.assertFalse(out["accepted"])
        self.assertEqual(out["reject_reason"], "non_interactive")


class TestResolveKbManager(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(_resolve_kb_manager(None))

    def test_cache_manager_fallback(self):
        sentinel = object()
        pm = SimpleNamespace(cache_manager=sentinel)
        self.assertIs(_resolve_kb_manager(pm), sentinel)

    def test_explicit_name_wins(self):
        ukm = object()
        cache = object()
        pm = SimpleNamespace(unified_knowledge_manager=ukm, cache_manager=cache)
        self.assertIs(_resolve_kb_manager(pm), ukm)


def _pm_with_steps(step_kv):
    """Fake pipeline_manager with config.step_configs of {key: (provider, model)}."""
    steps = {k: SimpleNamespace(provider=p, model=m) for k, (p, m) in step_kv.items()}
    return SimpleNamespace(config=SimpleNamespace(
        global_provider_override=None, global_model_override=None, step_configs=steps))


class TestResolveProviderModel(unittest.TestCase):
    """The chain has no chat-specific step any more.

    ``ChatConfig.default_provider/model`` used to sit above everything the
    operator can set, was written by a single control that has since been
    removed, and a value left over from an earlier session therefore decided
    every chat turn while no surface showed it. The parameter is gone.
    """

    def test_explicit_wins(self):
        self.assertEqual(resolve_provider_model("openai", "gpt-4"), ("openai", "gpt-4"))

    def test_the_chain_takes_no_chat_config_any_more(self):
        import inspect

        sig = inspect.signature(resolve_provider_model)
        self.assertNotIn("chat_config", sig.parameters)

    def test_a_leftover_chat_default_cannot_be_passed_in(self):
        # ChatConfig no longer has the fields at all, so an old config file
        # cannot revive the behaviour either.
        cfg = ChatConfig()
        self.assertFalse(hasattr(cfg, "default_provider"))
        self.assertFalse(hasattr(cfg, "default_model"))

    def test_global_override(self):
        pm = SimpleNamespace(config=SimpleNamespace(
            global_provider_override="gemini", global_model_override="flash", step_configs={}))
        self.assertEqual(resolve_provider_model(None, None, pipeline_manager=pm),
                         ("gemini", "flash"))

    def test_pipeline_step_default(self):
        pm = _pm_with_steps({"keywords": ("openai_compatible", "nemotron-3-nano:30b")})
        self.assertEqual(resolve_provider_model(None, None, pipeline_manager=pm),
                         ("openai_compatible", "nemotron-3-nano:30b"))

    def test_step_order_prefers_keywords(self):
        pm = _pm_with_steps({
            "initialisation": ("prov_init", "m_init"),
            "keywords": ("prov_kw", "m_kw"),
        })
        self.assertEqual(resolve_provider_model(None, None, pipeline_manager=pm),
                         ("prov_kw", "m_kw"))

    def test_skips_empty_steps(self):
        pm = _pm_with_steps({"keywords": ("", ""), "dk_classification": ("p", "m")})
        self.assertEqual(resolve_provider_model(None, None, pipeline_manager=pm), ("p", "m"))

    def test_llm_service_last_resort(self):
        llm = SimpleNamespace(current_provider="gemini", current_model="flash")
        pm = _pm_with_steps({})
        self.assertEqual(
            resolve_provider_model(None, None, pipeline_manager=pm, llm_service=llm),
            ("gemini", "flash"))

    def test_empty_when_nothing(self):
        self.assertEqual(resolve_provider_model(None, None), ("", ""))


class TestHeadlessAgentRunner(unittest.TestCase):
    def test_immediate_final_answer(self):
        llm = _ScriptedLLM([
            AgentResponse(content="Hallo, womit kann ich helfen?", tool_calls=[],
                          stop_reason=StopReason.END_TURN),
        ])
        runner = HeadlessAgentRunner(llm_service=llm, pipeline_manager=None)
        tokens = []
        res = runner.run("Hallo", provider="ollama", model="x",
                         on_token=tokens.append)
        self.assertEqual(res.content, "Hallo, womit kann ich helfen?")
        self.assertEqual(llm.calls, 1)

    def test_tool_call_roundtrip(self):
        llm = _ScriptedLLM([
            AgentResponse(content="", tool_calls=[
                ToolCall(id="c1", name="list_available_data", arguments={})],
                stop_reason=StopReason.TOOL_USE),
            AgentResponse(content="Keine Pipeline-Daten vorhanden.", tool_calls=[],
                          stop_reason=StopReason.END_TURN),
        ])
        runner = HeadlessAgentRunner(llm_service=llm, pipeline_manager=None)
        seen_tools = []
        res = runner.run("Was liegt vor?", provider="ollama", model="x",
                         on_tool_call=lambda tc: seen_tools.append(tc.name))
        self.assertEqual(res.content, "Keine Pipeline-Daten vorhanden.")
        self.assertIn("list_available_data", seen_tools)
        self.assertEqual(res.iterations, 2)

    def test_max_iterations_from_chat_config(self):
        chat_config = SimpleNamespace(max_iterations=7, temperature=0.3)
        runner = HeadlessAgentRunner(llm_service=_ScriptedLLM([
            AgentResponse(content="ok")]), chat_config=chat_config)
        self.assertEqual(runner.max_iterations, 7)

    def test_explicit_max_iterations_overrides_config(self):
        chat_config = SimpleNamespace(max_iterations=7)
        runner = HeadlessAgentRunner(llm_service=_ScriptedLLM([AgentResponse(content="ok")]),
                                     chat_config=chat_config, max_iterations=3)
        self.assertEqual(runner.max_iterations, 3)


class TestStoppableAgentThread(unittest.TestCase):
    def test_runs_target_with_should_stop(self):
        captured = {}

        def target(should_stop):
            captured["stop_callable"] = should_stop
            captured["stopped_initially"] = should_stop()
            return "done"

        th = StoppableAgentThread(target)
        th.start()
        th.join(timeout=5)
        self.assertEqual(th.result, "done")
        self.assertFalse(captured["stopped_initially"])
        self.assertTrue(hasattr(th, "_stop_event"))

    def test_request_stop_sets_event_and_cancels(self):
        llm = _ScriptedLLM([AgentResponse(content="x")])
        started = threading.Event()

        def target(should_stop):
            started.set()
            while not should_stop():
                pass
            return "stopped"

        th = StoppableAgentThread(target, llm_service=llm)
        th.start()
        self.assertTrue(started.wait(timeout=5))
        th.request_stop()
        th.join(timeout=5)
        self.assertEqual(th.result, "stopped")
        self.assertEqual(llm.cancelled_reason, "headless_cancel")

    def test_target_exception_captured(self):
        def target(should_stop):
            raise RuntimeError("boom")

        th = StoppableAgentThread(target)
        th.start()
        th.join(timeout=5)
        self.assertIsInstance(th.error, RuntimeError)


if __name__ == "__main__":
    unittest.main()
