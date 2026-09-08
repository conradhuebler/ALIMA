"""P-ι — CLI ``alima agent`` handler tests. Claude Generated.

Covers the pure helpers (input resolution, provider/model resolution) and the
``handle_agent`` flow with all heavy services patched out (no DB, no LLM):

1. ``_resolve_input``: doi / inline / file / none.
2. ``_resolve_provider_model`` precedence.
3. ``handle_agent`` picks ``StdinProposalGateway`` by default and ``None`` +
   ``autonomous_pipeline=True`` with ``--autonomous``; writes a result JSON
   containing the ``agent`` block.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.cli.commands import agent_cmd
from src.core.data_models import AgentResult
from src.core.headless_gateway import StdinProposalGateway
from src.utils.config_models import ChatConfig

_LOG = logging.getLogger("test")


def _args(**over):
    base = dict(
        doi=None, input=None, input_file=None, input_image=None,
        prompt=None, provider="ollama", model="cogito:14b",
        temperature=None, max_iterations=None, autonomous=False,
        output=None, quiet=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestResolveInput(unittest.TestCase):
    def test_inline(self):
        text, itype, src = agent_cmd._resolve_input(_args(input="Ein Abstract."), None, _LOG)
        self.assertEqual(text, "Ein Abstract.")
        self.assertEqual(itype, "text")

    def test_none(self):
        text, itype, src = agent_cmd._resolve_input(_args(), None, _LOG)
        self.assertEqual(text, "")
        self.assertEqual(itype, "none")

    def test_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Dateiinhalt.")
            path = f.name
        try:
            text, itype, src = agent_cmd._resolve_input(_args(input_file=path), None, _LOG)
            self.assertEqual(text, "Dateiinhalt.")
            self.assertEqual(itype, "text")
        finally:
            os.unlink(path)

    def test_doi_success(self):
        with mock.patch.object(agent_cmd, "resolve_input_to_text",
                               return_value=(True, "Aufgelöster Text", None)) as m:
            text, itype, src = agent_cmd._resolve_input(_args(doi="10.1/abc"), None, _LOG)
        self.assertEqual(text, "Aufgelöster Text")
        self.assertEqual(itype, "doi")
        m.assert_called_once()

    def test_doi_failure_raises(self):
        with mock.patch.object(agent_cmd, "resolve_input_to_text",
                               return_value=(False, None, "404")):
            with self.assertRaises(ValueError):
                agent_cmd._resolve_input(_args(doi="10.1/bad"), None, _LOG)


class TestResolveProviderModel(unittest.TestCase):
    def test_args_win(self):
        p, m = agent_cmd._resolve_provider_model(
            _args(provider="openai", model="gpt-4"), ChatConfig(), None, None)
        self.assertEqual((p, m), ("openai", "gpt-4"))

    def test_a_chat_config_no_longer_carries_a_default(self):
        # The chat-specific provider/model default is gone: it outranked the
        # settings and only one removed control ever wrote it.
        p, m = agent_cmd._resolve_provider_model(
            _args(provider=None, model=None), ChatConfig(), None, None)
        self.assertEqual((p, m), ("", ""))

    def test_llm_service_fallback(self):
        llm = SimpleNamespace(current_provider="gemini", current_model="flash")
        p, m = agent_cmd._resolve_provider_model(
            _args(provider=None, model=None), ChatConfig(), None, llm)
        self.assertEqual((p, m), ("gemini", "flash"))

    def test_empty_when_nothing(self):
        p, m = agent_cmd._resolve_provider_model(
            _args(provider=None, model=None), ChatConfig(), None, None)
        self.assertEqual((p, m), ("", ""))


class _FakeRunner:
    """Records constructor kwargs; run() returns a canned AgentResult."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeRunner.instances.append(self)

    def run(self, user_message, **kwargs):
        self.run_kwargs = kwargs
        self.user_message = user_message
        return AgentResult(content="Fertige Antwort.", iterations=2, tool_log=[])


class TestHandleAgent(unittest.TestCase):
    def setUp(self):
        _FakeRunner.instances = []
        self._cm = mock.MagicMock()
        # chat_config is read via load_config().chat_config.
        self._cm.load_config.return_value = SimpleNamespace(chat_config=ChatConfig())
        self._llm = SimpleNamespace(cancel_generation=lambda **k: None,
                                    current_provider=None, current_model=None)

    def _run(self, args):
        pm_instance = mock.MagicMock()
        pm_instance.current_analysis_state = None
        pm_instance.config = SimpleNamespace(
            global_provider_override=None, global_model_override=None, step_configs={})
        with mock.patch.object(agent_cmd, "AlimaManager"), \
             mock.patch.object(agent_cmd, "UnifiedKnowledgeManager"), \
             mock.patch.object(agent_cmd, "PipelineManager", return_value=pm_instance), \
             mock.patch.object(agent_cmd, "HeadlessAgentRunner", _FakeRunner):
            return agent_cmd.handle_agent(args, self._cm, self._llm, mock.MagicMock(), _LOG)

    def test_default_uses_stdin_gateway(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            rc = self._run(_args(input="Abstract.", output=out))
            self.assertEqual(rc, 0)
            runner = _FakeRunner.instances[0]
            self.assertIsInstance(runner.kwargs["gateway"], StdinProposalGateway)
            with open(out, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["agent"]["final_content"], "Fertige Antwort.")
            self.assertFalse(payload["agent"]["autonomous"])

    def test_autonomous_no_gateway_and_flag_set(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            rc = self._run(_args(input="Abstract.", autonomous=True, output=out))
            self.assertEqual(rc, 0)
            runner = _FakeRunner.instances[0]
            self.assertIsNone(runner.kwargs["gateway"])
            # chat_config passed to runner has autonomous flag flipped on.
            self.assertTrue(runner.kwargs["chat_config"].autonomous_pipeline)
            with open(out, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertTrue(payload["agent"]["autonomous"])

    def test_missing_provider_model_returns_2(self):
        self._cm.get_unified_config.return_value = SimpleNamespace(chat_config=ChatConfig())
        rc = self._run(_args(provider=None, model=None, input="x"))
        self.assertEqual(rc, 2)

    def test_default_prompt_when_context(self):
        rc = self._run(_args(input="Abstract.", output=None))
        self.assertEqual(rc, 0)
        runner = _FakeRunner.instances[0]
        self.assertIn("GND-Schlagwörter", runner.user_message)


if __name__ == "__main__":
    unittest.main()
