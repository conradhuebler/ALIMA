"""Ollama connection input: cleaning, hints, URL building, and the dialog.

Motivating failure: host ``ollama.com`` with the dialog defaults (port 11434,
no SSL) built ``http://ollama.com:11434`` and the API key went out in plain
text. Claude Generated.
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from src.utils import ollama_endpoint as oe


class CleanApiKeyTest(unittest.TestCase):
    def test_strips_whitespace_newlines_and_bearer(self):
        self.assertEqual(oe.clean_api_key("  abc \n"), "abc")
        self.assertEqual(oe.clean_api_key("Bearer abc"), "abc")
        self.assertEqual(oe.clean_api_key("bearer   abc"), "abc")
        self.assertEqual(oe.clean_api_key("ab c\td"), "abcd")
        self.assertEqual(oe.clean_api_key(""), "")
        self.assertEqual(oe.clean_api_key(None), "")

    def test_key_that_merely_contains_bearer_inside_is_kept(self):
        self.assertEqual(oe.clean_api_key("xbearerx"), "xbearerx")


class ParseHostTest(unittest.TestCase):
    def test_plain_host_unchanged(self):
        self.assertEqual(oe.parse_host("localhost", 11434, False), ("localhost", 11434, False))

    def test_scheme_sets_ssl_and_is_removed(self):
        self.assertEqual(oe.parse_host("https://gpu.uni.de", 11434, False),
                         ("gpu.uni.de", 11434, True))
        self.assertEqual(oe.parse_host("http://gpu.uni.de", 11434, True),
                         ("gpu.uni.de", 11434, False))

    def test_port_suffix_replaces_port_and_path_is_dropped(self):
        self.assertEqual(oe.parse_host("https://gpu.uni.de:8443/api/", 11434, False),
                         ("gpu.uni.de", 8443, True))
        self.assertEqual(oe.parse_host("gpu.uni.de:8080", 11434, False),
                         ("gpu.uni.de", 8080, False))

    def test_empty_and_whitespace(self):
        self.assertEqual(oe.parse_host("  ", 11434, False), ("", 11434, False))
        self.assertEqual(oe.parse_host("  localhost  ", 11434, False)[0], "localhost")


class BaseUrlTest(unittest.TestCase):
    def test_local(self):
        self.assertEqual(oe.base_url("localhost", 11434, False), "http://localhost:11434")

    def test_cloud_ignores_port_and_ssl(self):
        for args in (("ollama.com", 11434, False), ("https://ollama.com", 443, True),
                     ("www.ollama.com", 80, False), ("http://ollama.com:11434", 11434, False)):
            self.assertEqual(oe.base_url(*args), "https://ollama.com", args)

    def test_scheme_in_host_field_no_longer_doubles(self):
        self.assertEqual(oe.base_url("https://gpu.uni.de", 443, False), "https://gpu.uni.de:443")

    def test_lookalike_domain_is_not_cloud(self):
        self.assertFalse(oe.is_ollama_cloud_host("notollama.com"))
        self.assertFalse(oe.is_ollama_cloud_host("ollama.com.evil.org"))
        self.assertTrue(oe.is_ollama_cloud_host("api.ollama.com"))


def _texts(hs, level=None):
    return [h.text for h in hs if level is None or h.level == level]


class HintsTest(unittest.TestCase):
    def test_empty_host_is_the_only_error(self):
        hs = oe.hints("", 11434, False, "")
        self.assertEqual([h.level for h in hs], ["error"])

    def test_clean_local_setup_is_silent(self):
        self.assertEqual(oe.hints("localhost", 11434, False, ""), [])
        self.assertEqual(oe.hints("127.0.0.1", 11434, False, "k"), [])

    def test_cloud_without_key_warns_and_with_key_only_informs(self):
        self.assertTrue(_texts(oe.hints("ollama.com", 11434, False, ""), "warn"))
        self.assertEqual(_texts(oe.hints("ollama.com", 11434, False, "k"), "warn"), [])
        self.assertTrue(_texts(oe.hints("ollama.com", 11434, False, "k"), "info"))

    def test_port_443_without_ssl_and_ssl_on_11434(self):
        self.assertTrue(_texts(oe.hints("gpu.uni.de", 443, False, ""), "warn"))
        self.assertTrue(_texts(oe.hints("gpu.uni.de", 11434, True, ""), "warn"))

    def test_key_over_plain_http_to_public_name_warns_but_not_to_intranet(self):
        self.assertTrue(_texts(oe.hints("gpu.uni.de", 11434, False, "k"), "warn"))
        self.assertEqual(_texts(oe.hints("192.168.1.5", 11434, False, "k"), "warn"), [])
        self.assertEqual(_texts(oe.hints("llmachine", 11434, False, "k"), "warn"), [])
        self.assertEqual(_texts(oe.hints("gpu.uni.de", 443, True, "k"), "warn"), [])

    def test_pasted_url_and_dirty_key_are_announced(self):
        self.assertTrue(_texts(oe.hints("https://gpu.uni.de:8443", 11434, False, ""), "info"))
        self.assertTrue(_texts(oe.hints("localhost", 11434, False, "Bearer k"), "info"))

    def test_hints_never_echo_the_key(self):
        secret = "SECRET-KEY-123"
        for h in oe.hints("gpu.uni.de", 11434, False, secret):
            self.assertNotIn(secret, h.text)


# ---- dialog ----------------------------------------------------------------

_kept: list = []


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


class ProviderDialogFeedbackTest(unittest.TestCase):
    def _dlg(self):
        from src.ui.provider_dialogs import ProviderEditDialog

        app = _qapp()
        dlg = ProviderEditDialog()
        _kept.append(dlg)
        return dlg, app

    def test_ollama_key_field_is_masked(self):
        from PyQt6.QtWidgets import QLineEdit

        dlg, _ = self._dlg()
        self.assertEqual(dlg.api_key_edit.echoMode(), QLineEdit.EchoMode.Password)

    def test_empty_host_disables_ok_and_shows_error(self):
        dlg, _ = self._dlg()
        dlg.host_edit.setText("")
        self.assertFalse(dlg.ok_button.isEnabled())
        self.assertIn("Host fehlt", dlg.hint_label.text())
        dlg.host_edit.setText("localhost")
        self.assertTrue(dlg.ok_button.isEnabled())

    def test_switching_type_releases_ok_and_clears_ollama_hints(self):
        """Leerer Ollama-Host sperrt OK; der Wechsel auf einen Typ ohne Hostfeld
        muss das wieder freigeben."""
        dlg, _ = self._dlg()
        dlg.host_edit.setText("")
        self.assertFalse(dlg.ok_button.isEnabled())
        dlg.type_combo.setCurrentText("gemini")
        self.assertTrue(dlg.ok_button.isEnabled())
        self.assertTrue(dlg.hint_label.isHidden())

    def test_the_warning_dialog_can_be_built(self):
        """accept() öffnet bei Endpunkt-Warnungen eine QMessageBox; der Import
        fehlte, der Warnpfad endete in einem NameError und der Dialog blieb offen."""
        from src.ui import provider_dialogs

        self.assertTrue(hasattr(provider_dialogs, "QMessageBox"))

    def test_cloud_host_shows_hint_live(self):
        dlg, _ = self._dlg()
        dlg.host_edit.setText("ollama.com")
        self.assertIn("API-Key", dlg.hint_label.text())
        dlg.api_key_edit.setText("k")
        self.assertNotIn("braucht einen API-Key", dlg.hint_label.text())

    def test_hint_never_contains_the_key(self):
        dlg, _ = self._dlg()
        dlg.host_edit.setText("gpu.uni.de")
        dlg.api_key_edit.setText("SECRET-KEY-123")
        self.assertNotIn("SECRET-KEY-123", dlg.hint_label.text())

    def test_get_provider_data_cleans_input(self):
        dlg, _ = self._dlg()
        dlg.host_edit.setText("https://gpu.uni.de:8443/")
        dlg.api_key_edit.setText(" Bearer abc\n")
        p = dlg.get_provider_data()
        self.assertEqual((p.host, p.port, p.use_ssl, p.api_key),
                         ("gpu.uni.de", 8443, True, "abc"))

    def test_cloud_dialog_defaults_are_stored_as_https_443(self):
        dlg, _ = self._dlg()
        dlg.host_edit.setText("ollama.com")
        p = dlg.get_provider_data()
        self.assertEqual((p.host, p.port, p.use_ssl), ("ollama.com", 443, True))

    def test_test_button_reports_result_without_saving(self):
        from src.utils.setup_utils import OllamaConnectionValidator, SetupResult

        dlg, app = self._dlg()
        dlg.host_edit.setText("gpu.uni.de")
        dlg.api_key_edit.setText("k")
        calls = []

        def fake(host, port, key, timeout=10, use_ssl=False):
            calls.append((host, port, key, use_ssl))
            return SetupResult(True, "Verbindung erfolgreich. 3 Modelle gefunden.", ["a"])

        with patch.object(OllamaConnectionValidator, "test_native", staticmethod(fake)):
            dlg._run_connection_test()
            self.assertFalse(dlg.test_button.isEnabled())
            deadline = time.time() + 5
            while not dlg.test_button.isEnabled() and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
        self.assertEqual(calls, [("gpu.uni.de", 11434, "k", False)])
        self.assertIn("✅", dlg.test_result.text())
        self.assertTrue(dlg.test_button.isEnabled())

    def test_failed_test_is_shown_as_failure(self):
        from src.utils.setup_utils import OllamaConnectionValidator, SetupResult

        dlg, app = self._dlg()
        with patch.object(OllamaConnectionValidator, "test_native",
                          staticmethod(lambda *a, **k: SetupResult(False, "401", []))):
            dlg._run_connection_test()
            deadline = time.time() + 5
            while not dlg.test_button.isEnabled() and time.time() < deadline:
                app.processEvents()
                time.sleep(0.01)
        self.assertIn("❌", dlg.test_result.text())

    def test_test_button_hidden_for_types_without_a_test(self):
        dlg, _ = self._dlg()
        dlg.type_combo.setCurrentText("gemini")
        self.assertTrue(dlg.test_button.isHidden())
        dlg.type_combo.setCurrentText("ollama")
        self.assertFalse(dlg.test_button.isHidden())


class NativeValidatorMessagesTest(unittest.TestCase):
    """test_native turns server answers into messages a user can act on."""

    def _run(self, exc):
        from src.utils.setup_utils import OllamaConnectionValidator

        class Boom:
            def __init__(self, **kw):
                pass

            def list(self):
                raise exc

        with patch("ollama.Client", Boom):
            return OllamaConnectionValidator.test_native("ollama.com", 11434, "k")

    def test_401_names_the_key_and_the_real_url(self):
        r = self._run(Exception("Unauthorized (status code: 401)"))
        self.assertFalse(r.success)
        self.assertIn("Schlüssel", r.message)
        self.assertIn("https://ollama.com", r.message)

    def test_connection_error_names_host_port_ssl(self):
        r = self._run(ConnectionError("Connection refused"))
        self.assertIn("Host, Port und SSL", r.message)


if __name__ == "__main__":
    unittest.main()


class NoUntestedNoticeTest(unittest.TestCase):
    """Gemini und Anthropic laufen; der Dialog trägt keinen „nicht getestet“-Hinweis mehr."""

    def test_no_provider_type_shows_an_untested_notice(self):
        from src.ui.provider_dialogs import ProviderEditDialog

        _qapp()
        d = ProviderEditDialog()
        _kept.append(d)
        self.assertFalse(hasattr(d, "warning_label"))
        for ptype in ("gemini", "anthropic", "ollama", "openai_compatible"):
            d.type_combo.setCurrentText(ptype)
            self.assertNotIn("nicht getestet", d.hint_label.text())
