"""Setup wizards: Ollama is stored natively, with the same input help as the
provider dialog.

Before: ``create_initial_config`` turned "ollama" into an openai_compatible
entry with "/v1" and the placeholder key "no-key-required" (a cloud key was
dropped), although the wizard had just tested the *native* client.
Claude Generated.
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from src.utils.setup_utils import ConfigurationBuilder, OllamaConnectionValidator, SetupResult


def _provider(**kw):
    cfg = ConfigurationBuilder.create_initial_config(
        provider_type="ollama", provider_name="x", models=["m1"], **kw
    )
    (p,) = cfg.unified_config.providers
    return cfg, p


class BuilderNativeOllamaTest(unittest.TestCase):
    def test_local_defaults(self):
        cfg, p = _provider(base_url="http://localhost:11434")
        self.assertEqual(
            (p.name, p.provider_type, p.host, p.port, p.use_ssl, p.api_key, p.enabled),
            ("ollama", "ollama", "localhost", 11434, False, "", True),
        )
        self.assertEqual(p.base_url, "http://localhost:11434")
        self.assertEqual(p.preferred_model, "m1")
        self.assertEqual(cfg.unified_config.pipeline_default_provider, "ollama")

    def test_no_placeholder_key_and_no_v1_suffix(self):
        _, p = _provider(base_url="http://localhost:11434")
        self.assertNotEqual(p.api_key, "no-key-required")
        self.assertFalse(p.base_url.endswith("/v1"))

    def test_remote_with_scheme_port_and_ssl(self):
        _, p = _provider(base_url="https://gpu.uni.de:8443")
        self.assertEqual((p.host, p.port, p.use_ssl), ("gpu.uni.de", 8443, True))

    def test_cloud_keeps_a_cleaned_key_and_gets_https_443(self):
        cfg, p = _provider(base_url="http://ollama.com:11434", api_key=" Bearer abc\n")
        self.assertEqual((p.name, p.host, p.port, p.use_ssl, p.api_key),
                         ("ollama_cloud", "ollama.com", 443, True, "abc"))
        self.assertEqual(p.base_url, "https://ollama.com")
        self.assertEqual(cfg.unified_config.pipeline_default_provider, "ollama_cloud")

    def test_missing_base_url_defaults_to_local(self):
        _, p = _provider()
        self.assertEqual((p.host, p.port), ("localhost", 11434))

    def test_other_provider_types_unchanged(self):
        cfg = ConfigurationBuilder.create_initial_config(
            provider_type="openai_compatible", provider_name="x",
            base_url="https://api.example/v1", api_key="k", models=["m"],
        )
        (p,) = cfg.unified_config.providers
        self.assertEqual((p.provider_type, p.base_url, p.api_key),
                         ("openai_compatible", "https://api.example/v1", "k"))


_kept: list = []


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


class GuiPageTest(unittest.TestCase):
    def _page(self):
        from src.ui.first_start_wizard import LLMSetupPage

        _qapp()
        page = LLMSetupPage(None)
        page._auto_test_timer.stop()
        _kept.append(page)
        return page

    def test_ssl_and_key_fields_exist_for_ollama_and_hint_is_live(self):
        page = self._page()
        page.ollama_ssl_input.setVisible(True)
        page.ollama_host_input.setText("ollama.com")
        self.assertIn("API-Key", page.ollama_hint.text())
        page.api_key_input.setText("SECRET-KEY-123")
        self.assertNotIn("SECRET-KEY-123", page.ollama_hint.text())
        self.assertNotIn("braucht einen API-Key", page.ollama_hint.text())

    def test_test_connection_passes_ssl_and_clean_key_and_stores_real_url(self):
        page = self._page()
        page.ollama_host_input.setText("gpu.uni.de")
        page.ollama_port_input.setValue(443)
        page.ollama_ssl_input.setChecked(True)
        page.api_key_input.setText(" Bearer abc ")
        seen = {}

        def fake(host, port, key, timeout=10, use_ssl=False):
            seen.update(host=host, port=port, key=key, ssl=use_ssl)
            return SetupResult(True, "ok", ["m1"])

        with patch.object(OllamaConnectionValidator, "test_native", staticmethod(fake)):
            page._test_connection()
        self.assertEqual(seen, dict(host="gpu.uni.de", port=443, key="abc", ssl=True))
        self.assertEqual(page.base_url, "https://gpu.uni.de:443")
        self.assertEqual(page.api_key, "abc")
        self.assertEqual(page.available_models, ["m1"])


class CliWizardTest(unittest.TestCase):
    def _wizard(self):
        from src.utils.cli_setup_wizard import CLISetupWizard

        w = CLISetupWizard.__new__(CLISetupWizard)
        w._presets = MagicMock(llm_provider_type="", llm_base_url="")
        w.api_key = ""
        return w

    def test_cloud_input_skips_ssl_question_and_keeps_key(self):
        w = self._wizard()
        seen = {}

        def fake(host, port, key, timeout=10, use_ssl=False):
            seen.update(host=host, port=port, key=key, ssl=use_ssl)
            return SetupResult(True, "ok", ["m1"])

        # Host, Port; kein SSL-Prompt für Cloud
        with patch("builtins.input", side_effect=["ollama.com", ""]), \
             patch("getpass.getpass", return_value="abc"), \
             patch.object(OllamaConnectionValidator, "test_native", staticmethod(fake)):
            w._setup_ollama()
        self.assertTrue(seen["ssl"])
        self.assertEqual((seen["host"], seen["key"]), ("ollama.com", "abc"))
        self.assertEqual((w.base_url, w.api_key), ("https://ollama.com", "abc"))

    def test_local_default_asks_ssl_and_builds_http_url(self):
        w = self._wizard()
        with patch("builtins.input", side_effect=["", "", ""]), \
             patch("getpass.getpass", return_value=""), \
             patch.object(OllamaConnectionValidator, "test_native",
                          staticmethod(lambda *a, **k: SetupResult(True, "ok", ["m"]))):
            w._setup_ollama()
        self.assertEqual(w.base_url, "http://localhost:11434")


class ValidatorUsesSslTest(unittest.TestCase):
    def test_test_native_connects_to_https_when_ssl_is_set(self):
        captured = {}

        class Spy:
            def __init__(self, **kw):
                captured.update(kw)

            def list(self):
                raise ConnectionError("Connection refused")

        with patch("ollama.Client", Spy):
            OllamaConnectionValidator.test_native("gpu.uni.de", 443, "k", use_ssl=True)
        self.assertEqual(captured["host"], "https://gpu.uni.de:443")
        self.assertEqual(captured["headers"], {"Authorization": "Bearer k"})


if __name__ == "__main__":
    unittest.main()


class PortDefaultTest(unittest.TestCase):
    """QSpinBox hat von Haus aus den Bereich 0..99: setValue(11434) vor setRange
    kappte den Port still auf 99 (Test lief gegen http://localhost:99)."""

    def test_wizard_port_field_starts_at_the_ollama_default(self):
        from src.ui.first_start_wizard import LLMSetupPage

        _qapp()
        page = LLMSetupPage(None)
        page._auto_test_timer.stop()
        _kept.append(page)
        self.assertEqual(page.ollama_port_input.value(), 11434)
        self.assertEqual(page.ollama_port_input.maximum(), 65535)

    def test_default_connection_test_targets_port_11434(self):
        from src.ui.first_start_wizard import LLMSetupPage

        _qapp()
        page = LLMSetupPage(None)
        page._auto_test_timer.stop()
        _kept.append(page)
        seen = {}

        def fake(host, port, key, timeout=10, use_ssl=False):
            seen.update(host=host, port=port)
            return SetupResult(True, "ok", ["m"])

        with patch.object(OllamaConnectionValidator, "test_native", staticmethod(fake)):
            page._test_connection()
        self.assertEqual((seen["host"], seen["port"]), ("localhost", 11434))
        self.assertEqual(page.base_url, "http://localhost:11434")


class GeneralDefaultIsSetExplicitlyTest(unittest.TestCase):
    """"Allgemein" (= Agentik = Chat) kam nie aus dem Wizard, sondern aus dem
    Rückfall "erster nutzbarer Anbieter" mit dessen erstem Modell; ein erneuter
    Lauf leerte einen vorhandenen Wert. Claude Generated."""

    def test_general_default_follows_the_chosen_pipeline_model(self):
        cfg = ConfigurationBuilder.create_initial_config(
            provider_type="ollama", provider_name="x", base_url="http://localhost:11434",
            models=["beam:cloud", "gemma3:12b"], pipeline_model="gemma3:12b",
        )
        u = cfg.unified_config
        self.assertEqual((u.preferred_provider, u.preferred_model), ("ollama", "gemma3:12b"))
        self.assertEqual(u.explain_default_provider_model(),
                         ("ollama", "gemma3:12b", "preferred"))

    def test_agentic_and_chat_resolve_to_the_same_choice(self):
        cfg = ConfigurationBuilder.create_initial_config(
            provider_type="ollama", provider_name="x", base_url="http://localhost:11434",
            models=["beam:cloud", "gemma3:12b"], pipeline_model="gemma3:12b",
        )
        u = cfg.unified_config
        self.assertEqual(u.resolve_default_provider_model(scope="agentic"),
                         u.resolve_default_provider_model(scope="general"))

    def test_a_rerun_on_an_existing_config_keeps_a_real_general_default(self):
        from src.ui.first_start_wizard import FirstStartWizard
        from src.utils.config_models import AlimaConfig, UnifiedProvider

        existing = AlimaConfig()
        existing.system_config.first_run_completed = True
        existing.unified_config.providers = [UnifiedProvider(name="alt", provider_type="ollama")]
        existing.unified_config.preferred_provider = "alt"
        existing.unified_config.preferred_model = "alt-modell"

        wizard_config = ConfigurationBuilder.create_initial_config(
            provider_type="ollama", provider_name="x", base_url="http://localhost:11434",
            models=["m1"], pipeline_model="m1",
        )
        cm = MagicMock()
        cm.load_config.return_value = existing
        wiz = FirstStartWizard.__new__(FirstStartWizard)
        with patch("src.ui.first_start_wizard.ConfigManager", return_value=cm):
            out = wiz._apply_onto_existing(wizard_config)
        # Der Wizard hat gewählt; Allgemein ist nicht leer und zeigt auf seine Wahl.
        self.assertEqual((out.unified_config.preferred_provider,
                          out.unified_config.preferred_model), ("ollama", "m1"))
