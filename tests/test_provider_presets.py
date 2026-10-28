"""Vorlagen für OpenAI-kompatible Anbieter (Dialog und Wizard). Claude Generated."""
from __future__ import annotations

import unittest
from urllib.parse import urlparse

from src.utils.provider_presets import (
    OPENAI_COMPATIBLE_PRESETS as PRESETS, preset_for_url,
)


class PresetDataTest(unittest.TestCase):
    def test_ids_labels_and_urls_are_unique(self):
        for field in ("id", "label", "base_url"):
            values = [getattr(p, field) for p in PRESETS]
            self.assertEqual(len(values), len(set(values)), field)

    def test_every_url_is_well_formed_and_without_trailing_slash(self):
        for p in PRESETS:
            u = urlparse(p.base_url)
            self.assertIn(u.scheme, ("http", "https"), p.id)
            self.assertTrue(u.hostname, p.id)
            self.assertFalse(p.base_url.endswith("/"), p.id)

    def test_remote_services_use_https_and_need_a_key_local_ones_do_not(self):
        for p in PRESETS:
            local = urlparse(p.base_url).hostname in ("localhost", "127.0.0.1")
            self.assertEqual(p.key_required, not local, p.id)
            if not local:
                self.assertEqual(urlparse(p.base_url).scheme, "https", p.id)

    def test_requested_services_are_present_and_the_gwdg_is_not(self):
        ids = {p.id for p in PRESETS}
        self.assertTrue({"openai", "mistral", "nvidia"} <= ids)
        blob = " ".join(p.base_url + p.label for p in PRESETS).lower()
        self.assertNotIn("gwdg", blob)
        self.assertNotIn("academiccloud", blob)

    def test_preset_for_url_ignores_case_and_trailing_slash(self):
        self.assertEqual(preset_for_url("https://API.mistral.ai/v1/").id, "mistral")
        self.assertIsNone(preset_for_url("https://example.org/v1"))
        self.assertIsNone(preset_for_url(""))
        self.assertIsNone(preset_for_url(None))


_kept: list = []


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


class DialogTest(unittest.TestCase):
    def _dlg(self):
        from src.ui.provider_dialogs import ProviderEditDialog

        _qapp()
        d = ProviderEditDialog()
        d.type_combo.setCurrentText("openai_compatible")
        _kept.append(d)
        return d

    def _choose(self, d, preset_id):
        idx = 1 + [p.id for p in PRESETS].index(preset_id)
        d.preset_combo.setCurrentIndex(idx)
        d._on_preset_chosen(idx)

    def test_choosing_a_preset_fills_url_name_and_description(self):
        d = self._dlg()
        self._choose(d, "mistral")
        self.assertEqual(d.base_url_edit.text(), "https://api.mistral.ai/v1")
        self.assertEqual(d.name_edit.text(), "mistral")
        self.assertTrue(d.description_edit.text())

    def test_existing_name_and_description_are_kept(self):
        d = self._dlg()
        d.name_edit.setText("mein-mistral")
        d.description_edit.setText("Arbeitskonto")
        self._choose(d, "mistral")
        self.assertEqual((d.name_edit.text(), d.description_edit.text()),
                         ("mein-mistral", "Arbeitskonto"))

    def test_remote_preset_without_key_warns_and_with_key_does_not(self):
        d = self._dlg()
        self._choose(d, "openai")
        self.assertIn("braucht einen API-Key", d.hint_label.text())
        d.api_key_edit.setText("sk-test")
        self.assertNotIn("braucht einen API-Key", d.hint_label.text())
        self.assertNotIn("sk-test", d.hint_label.text())

    def test_local_preset_needs_no_key(self):
        d = self._dlg()
        self._choose(d, "lmstudio")
        self.assertNotIn("braucht einen API-Key", d.hint_label.text())

    def test_combo_follows_a_typed_or_loaded_url(self):
        d = self._dlg()
        d.base_url_edit.setText("https://api.x.ai/v1")
        self.assertEqual(d.preset_combo.currentData(), "xai")
        d.base_url_edit.setText("https://eigener.server/v1")
        self.assertEqual(d.preset_combo.currentIndex(), 0)

    def test_saved_provider_data_contains_the_preset_url(self):
        d = self._dlg()
        self._choose(d, "nvidia")
        d.api_key_edit.setText(" nvapi-x \n")
        p = d.get_provider_data()
        self.assertEqual((p.name, p.provider_type, p.base_url, p.api_key),
                         ("nvidia", "openai_compatible",
                          "https://integrate.api.nvidia.com/v1", "nvapi-x"))


class WizardTest(unittest.TestCase):
    def _page(self):
        from src.ui.first_start_wizard import LLMSetupPage

        _qapp()
        page = LLMSetupPage(None)
        page._auto_test_timer.stop()
        _kept.append(page)
        return page

    def test_default_url_selects_the_openai_entry(self):
        page = self._page()
        self.assertEqual(page.base_url_input.text(), "https://api.openai.com/v1")
        self.assertEqual(page.openai_preset_combo.currentData(), "https://api.openai.com/v1")

    def test_choosing_a_preset_sets_the_url(self):
        page = self._page()
        idx = page.openai_preset_combo.findData("https://api.mistral.ai/v1")
        page.openai_preset_combo.setCurrentIndex(idx)
        page._on_openai_preset_chosen(idx)
        self.assertEqual(page.base_url_input.text(), "https://api.mistral.ai/v1")

    def test_own_url_entry_leaves_the_url_alone(self):
        page = self._page()
        page.base_url_input.setText("https://eigener.server/v1")
        self.assertEqual(page.openai_preset_combo.currentIndex(), 0)
        page._on_openai_preset_chosen(0)
        self.assertEqual(page.base_url_input.text(), "https://eigener.server/v1")


if __name__ == "__main__":
    unittest.main()
