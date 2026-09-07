"""The chat's LLM pick comes from the pipeline toolbar. Claude Generated.

The chat panel used to carry its own provider/model picker while sitting inside
the pipeline tab, whose toolbar has one too. Two controls for the same decision,
and the chat's silently outranked the toolbar's: it entered the shared
resolution chain above ``ChatConfig`` *and* above
``PipelineConfig.global_*_override``. The picker is gone; the toolbar hands its
selection over via ``set_llm_override``.

Strategy unchanged from the previous module: never instantiate the real panel
(it builds a full QWidget hierarchy that conflicts with the QCoreApplication-only
setup elsewhere in the suite). The helpers are called as unbound methods on a
minimal stand-in carrying only the attributes they touch.
"""
from __future__ import annotations

import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ui.pipeline_chat_panel import PipelineChatPanel as ChatWidget
from src.utils.config_models import ChatConfig


def _make_stub(override=("", ""), chat_config=None) -> SimpleNamespace:
    """Minimal duck-typed panel carrying only what the helpers touch."""
    status_strip: list[str] = []
    labels: list[str] = []
    stub = SimpleNamespace(
        _llm_override=override,
        logger=logging.getLogger("test"),
        model_status_label=SimpleNamespace(setText=labels.append),
        pipeline_manager=None,
        llm_service=None,
        _append_system_message=lambda _m: None,
        set_status_strip=status_strip.append,
    )
    stub.status_strip_texts = status_strip
    stub.label_texts = labels
    stub._get_chat_config = lambda: chat_config or ChatConfig()
    for name in (
        "set_llm_override",
        "_resolve_provider_model",
        "_refresh_model_status",
        "_persist_combo_to_chat_config",
    ):
        setattr(stub, name, getattr(ChatWidget, name).__get__(stub))
    return stub


class TestOverrideHandover(unittest.TestCase):
    def test_set_llm_override_is_adopted(self):
        stub = _make_stub()
        stub.set_llm_override("ollama", "gemma4:31b-cloud")
        self.assertEqual(stub._llm_override, ("ollama", "gemma4:31b-cloud"))

    def test_standard_clears_the_override(self):
        # "-- Standard --" in the toolbar yields ("", "") — the chat must fall
        # back, not keep the previous pick.
        stub = _make_stub(override=("ollama", "gemma4:31b-cloud"))
        stub.set_llm_override("", "")
        self.assertEqual(stub._llm_override, ("", ""))

    def test_none_is_normalised_to_empty(self):
        stub = _make_stub()
        stub.set_llm_override(None, None)
        self.assertEqual(stub._llm_override, ("", ""))

    def test_the_status_label_is_refreshed_on_handover(self):
        stub = _make_stub()
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("ollama", "gemma4:31b-cloud"),
        ):
            stub.set_llm_override("ollama", "gemma4:31b-cloud")
        self.assertEqual(stub.label_texts[-1], "→ ollama | gemma4:31b-cloud")

    def test_no_model_says_so(self):
        stub = _make_stub()
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("", ""),
        ):
            stub._refresh_model_status()
        self.assertEqual(stub.label_texts[-1], "→ (kein Modell)")


class TestResolution(unittest.TestCase):
    def test_the_toolbar_pick_is_passed_as_the_explicit_override(self):
        stub = _make_stub(override=("ollama", "gemma4:31b-cloud"))
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("ollama", "gemma4:31b-cloud"),
        ) as resolve:
            stub._resolve_provider_model()
        args, _ = resolve.call_args
        self.assertEqual(args[:2], ("ollama", "gemma4:31b-cloud"))

    def test_standard_passes_none_so_the_chain_decides(self):
        stub = _make_stub(override=("", ""))
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("gwdg", "gemma-3-27b"),
        ) as resolve:
            provider, model = stub._resolve_provider_model()
        args, _ = resolve.call_args
        self.assertEqual(args[:2], (None, None))
        self.assertEqual((provider, model), ("gwdg", "gemma-3-27b"))

    def test_an_unresolvable_pair_comes_back_empty(self):
        stub = _make_stub()
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("ollama", ""),
        ):
            self.assertEqual(stub._resolve_provider_model(), ("", ""))

    def test_a_panel_without_the_attribute_still_resolves(self):
        # Defensive: the helper is also reachable before setup_ui has run.
        stub = _make_stub()
        del stub._llm_override
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("ollama", "m"),
        ) as resolve:
            stub._resolve_provider_model()
        self.assertEqual(resolve.call_args[0][:2], (None, None))


class TestPersistDefault(unittest.TestCase):
    """Writing the chat default now takes the resolved pair, not a combo."""

    def _run_persist(self, resolved):
        stub = _make_stub()
        cfg = SimpleNamespace(chat_config=ChatConfig())
        manager = MagicMock()
        manager.load_config.return_value = cfg
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=resolved,
        ), patch("src.utils.config_manager.ConfigManager", return_value=manager):
            stub._persist_combo_to_chat_config()
        return cfg, manager, stub

    def test_the_resolved_pair_is_written(self):
        cfg, manager, stub = self._run_persist(("ollama", "gemma4:31b-cloud"))
        self.assertEqual(cfg.chat_config.default_provider, "ollama")
        self.assertEqual(cfg.chat_config.default_model, "gemma4:31b-cloud")
        manager.save_config.assert_called_once()
        self.assertTrue(stub.status_strip_texts)

    def test_nothing_resolvable_writes_nothing(self):
        _, manager, _ = self._run_persist(("", ""))
        manager.save_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class _Session:
    """Minimal ChatSession stand-in."""

    def __init__(self):
        self.requested_provider = ""
        self.requested_model = ""

    def clear_requested_model(self):
        self.requested_provider = ""
        self.requested_model = ""


class _Llm:
    def __init__(self, catalogue):
        self._catalogue = catalogue

    def get_available_models(self, provider):
        return list(self._catalogue.get(provider, []))


class TestSwitchToolGate(unittest.TestCase):
    """The agent gets the tool only when the operator enabled it."""

    def _build(self, *, allow, llm_service):
        from src.ui.chat_tools import build_chat_toolset

        class S:
            session_id = "s1"
            shared_context = None
            messages = []

        cfg = SimpleNamespace(
            no_cache_writes=True, autonomous_pipeline=False, allow_model_switch=allow
        )
        registry = build_chat_toolset(
            session=S(),
            chat_config=cfg,
            mcp_registry=None,
            pipeline_manager=None,
            kb_manager=None,
            proposal_gateway=None,
            llm_service=llm_service,
        )
        return [s["name"] for s in registry.get_tool_schemas([])]

    def test_off_means_the_tool_is_not_even_offered(self):
        self.assertNotIn("switch_llm_model", self._build(allow=False, llm_service=_Llm({})))

    def test_on_registers_the_tool(self):
        self.assertIn("switch_llm_model", self._build(allow=True, llm_service=_Llm({})))

    def test_a_host_without_an_llm_service_does_not_get_it(self):
        # The headless runner builds a fresh session per call — a switch there
        # would be forgotten before it could apply.
        self.assertNotIn("switch_llm_model", self._build(allow=True, llm_service=None))


class TestSwitchTool(unittest.TestCase):
    CATALOGUE = {"ollama": ["gemma4:31b-cloud", "mistral-small"], "gwdg": ["gemma-3-27b"]}

    def _tool(self, on_switch=None):
        from src.ui.chat_tools.llm_switch import SwitchLlmModelTool

        tool = SwitchLlmModelTool(llm_service=_Llm(self.CATALOGUE), on_switch=on_switch)
        tool._catalogue = lambda: {k: list(v) for k, v in self.CATALOGUE.items()}
        return tool

    def test_a_valid_switch_is_recorded_on_the_session(self):
        session = _Session()
        seen = []
        out = json.loads(
            self._tool(on_switch=lambda *a: seen.append(a)).execute(
                session, provider="ollama", model="mistral-small", reason="schneller"
            )
        )
        self.assertEqual(out["status"], "switched")
        self.assertEqual(out["effective"], "next_turn")
        self.assertEqual(session.requested_model, "mistral-small")
        self.assertEqual(seen, [("ollama", "mistral-small", "schneller")])

    def test_case_is_normalised_to_the_providers_spelling(self):
        session = _Session()
        self._tool().execute(session, provider="OLLAMA", model="Mistral-Small")
        self.assertEqual(
            (session.requested_provider, session.requested_model),
            ("ollama", "mistral-small"),
        )

    def test_an_unknown_model_is_refused_and_nothing_is_recorded(self):
        session = _Session()
        out = json.loads(self._tool().execute(session, provider="ollama", model="gpt-9"))
        self.assertEqual(out["status"], "error")
        self.assertIn("available", out)
        self.assertEqual(session.requested_model, "")

    def test_an_unknown_provider_is_refused(self):
        session = _Session()
        out = json.loads(self._tool().execute(session, provider="acme", model="x"))
        self.assertEqual(out["status"], "error")
        self.assertEqual(session.requested_provider, "")

    def test_half_a_pick_is_refused(self):
        session = _Session()
        out = json.loads(self._tool().execute(session, provider="ollama"))
        self.assertEqual(out["status"], "error")
        self.assertEqual(session.requested_provider, "")

    def test_no_arguments_lists_what_is_available(self):
        out = json.loads(self._tool().execute(_Session()))
        self.assertEqual(out["status"], "ok")
        self.assertIn("ollama", out["available"])

    def test_a_provider_that_cannot_be_asked_says_so(self):
        from src.ui.chat_tools.llm_switch import SwitchLlmModelTool

        tool = SwitchLlmModelTool(llm_service=_Llm({}))
        tool._catalogue = lambda: {"ollama": []}
        out = json.loads(tool.execute(_Session(), provider="ollama", model="x"))
        self.assertIn("nicht abgefragt", out["message"])


class TestSwitchPrecedence(unittest.TestCase):
    """The agent's pick outranks the toolbar until the operator acts."""

    def _stub(self, session):
        stub = _make_stub(override=("ollama", "gemma4:31b-cloud"))
        stub.session = session
        return stub

    def test_a_requested_model_wins_over_the_toolbar(self):
        session = _Session()
        session.requested_provider, session.requested_model = "gwdg", "gemma-3-27b"
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("gwdg", "gemma-3-27b"),
        ) as resolve:
            self._stub(session)._resolve_provider_model()
        self.assertEqual(resolve.call_args[0][:2], ("gwdg", "gemma-3-27b"))

    def test_without_a_request_the_toolbar_decides(self):
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("ollama", "gemma4:31b-cloud"),
        ) as resolve:
            self._stub(_Session())._resolve_provider_model()
        self.assertEqual(resolve.call_args[0][:2], ("ollama", "gemma4:31b-cloud"))

    def test_touching_the_toolbar_clears_the_agents_pick(self):
        session = _Session()
        session.requested_provider, session.requested_model = "gwdg", "gemma-3-27b"
        stub = self._stub(session)
        with patch(
            "src.ui._chat_panel_chat_agent.resolve_provider_model",
            return_value=("ollama", "mistral-small"),
        ):
            stub.set_llm_override("ollama", "mistral-small")
        self.assertEqual(session.requested_model, "")


class TestSwitchCrossThreadContract(unittest.TestCase):
    """The switch announcement must cross threads via a signal.

    The tool runs inside ``ChatAgentWorker``'s QThread. The first version handed
    the panel method to the tool as a plain callback, so the announcement wrote
    into the log view and a QLabel from the worker thread — Qt aborts the whole
    process for that (SIGTRAP), which is what a real GUI run did the moment an
    actual switch happened. The listing call, which touches nothing, had worked.
    """

    def test_the_panel_declares_the_signal_with_three_strings(self):
        from src.ui.pipeline_chat_panel import PipelineChatPanel

        signal = getattr(PipelineChatPanel, "model_switch_requested", None)
        self.assertIsNotNone(signal, "the announcement needs a signal to cross on")
        self.assertEqual(signal.signatures, ("QString,QString,QString)",))

    def test_the_handler_documents_its_thread(self):
        from src.ui.pipeline_chat_panel import PipelineChatPanel

        doc = PipelineChatPanel._on_agent_model_switch.__doc__ or ""
        self.assertIn("UI thread only", doc)

    def test_the_toolset_gets_the_emit_not_the_method(self):
        import inspect

        from src.ui import _chat_panel_chat_agent as mod

        source = inspect.getsource(mod.ChatAgentMixin.send_message)
        self.assertIn("model_switch_requested.emit", source)
        self.assertNotIn("on_model_switch=self._on_agent_model_switch", source)
