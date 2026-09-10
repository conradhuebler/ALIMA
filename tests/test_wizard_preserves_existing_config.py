"""Re-running the setup wizard must not wipe an existing installation.

``FirstStartWizard.accept`` built a **fresh** ``AlimaConfig`` via
``ConfigurationBuilder.create_initial_config`` and handed it to ``save_config``,
which writes the whole object. ``preserve_unified`` rescues parts of
``unified_config`` only; everything else went through ``asdict``. On a configured
machine that destroyed:

* every plugin instance's settings except catalog (finc base URL, sru preset, the
  webindex fields, k10plus, dnb) — ``save_config`` re-seeds an empty
  ``config.plugins`` with defaults;
* ``chat_config`` (institution context), ``ui_config``, ``repetition_config``;
* ``approved_plugins`` (the Tier-2 code-plugin approvals) and
  ``installed_bundles``;
* every LLM provider except the one the wizard asks about.

There is no backup: ``_atomic_write_json`` does ``os.replace``.

That was tolerable while ``--wizard`` was the only way in — an explicit reset.
It is not, once the wizard is reachable from the menu, which is why this landed
first. - Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _configured_installation():
    """An AlimaConfig that looks like a machine in use."""
    from src.utils.config_models import (
        AlimaConfig,
        PluginInstanceConfig,
        UnifiedProvider,
    )

    cfg = AlimaConfig()
    cfg.system_config.first_run_completed = True
    cfg.system_config.contact_email = "ub@example.org"
    cfg.chat_config.institution_context = "UB Freiberg, Bergbau-Schwerpunkt"
    cfg.ui_config.ui_language = "en"
    cfg.repetition_config.ngram_threshold = 4
    cfg.approved_plugins = {"mein_plugin": "abc123"}
    cfg.installed_bundles = {"ub-freiberg": {"version": "2"}}
    cfg.plugins = [
        PluginInstanceConfig(
            "finc", "search_provider", "finc",
            settings={"base_url": "https://finc.example", "dk_enabled": True},
        ),
        PluginInstanceConfig(
            "webindex", "lookup", "webindex", settings={"base_url": "https://wi.example"}
        ),
    ]
    cfg.unified_config.providers = [
        UnifiedProvider(name="gwdg", provider_type="openai_compatible", enabled=True),
        UnifiedProvider(name="alt-ollama", provider_type="ollama", enabled=True),
    ]
    return cfg


class WizardMergesOntoAnExistingConfigTest(unittest.TestCase):
    def _merged(self, existing):
        from src.ui.first_start_wizard import FirstStartWizard
        from src.utils.config_models import AlimaConfig, UnifiedProvider

        wizard_config = AlimaConfig()
        wizard_config.unified_config.providers = [
            UnifiedProvider(name="neu", provider_type="ollama", enabled=True)
        ]
        wizard_config.unified_config.preferred_provider = "neu"

        cm = MagicMock()
        cm.load_config.return_value = existing
        wiz = FirstStartWizard.__new__(FirstStartWizard)
        with patch("src.ui.first_start_wizard.ConfigManager", return_value=cm):
            return wiz._apply_onto_existing(wizard_config)

    def test_plugin_settings_survive(self):
        out = self._merged(_configured_installation())
        finc = next(p for p in out.plugins if p.provider_id == "finc")
        self.assertEqual(finc.settings["base_url"], "https://finc.example")
        self.assertIs(finc.settings["dk_enabled"], True)

    def test_approvals_and_bundles_survive(self):
        out = self._merged(_configured_installation())
        self.assertEqual(out.approved_plugins, {"mein_plugin": "abc123"})
        self.assertIn("ub-freiberg", out.installed_bundles)

    def test_chat_ui_and_repetition_settings_survive(self):
        out = self._merged(_configured_installation())
        self.assertEqual(
            out.chat_config.institution_context, "UB Freiberg, Bergbau-Schwerpunkt"
        )
        self.assertEqual(out.ui_config.ui_language, "en")
        self.assertEqual(out.repetition_config.ngram_threshold, 4)

    def test_system_settings_survive(self):
        out = self._merged(_configured_installation())
        self.assertEqual(out.system_config.contact_email, "ub@example.org")

    def test_the_wizards_own_answers_are_applied(self):
        """Preserving must not mean "the wizard does nothing"."""
        out = self._merged(_configured_installation())
        self.assertEqual([p.name for p in out.unified_config.providers], ["neu"])
        self.assertEqual(out.unified_config.preferred_provider, "neu")

    def test_a_genuine_first_run_uses_the_wizard_config(self):
        """Nothing to preserve — the fresh config is the right answer there."""
        fresh = _configured_installation()
        fresh.system_config.first_run_completed = False
        out = self._merged(fresh)
        self.assertFalse(out.approved_plugins)
        self.assertEqual([p.name for p in out.unified_config.providers], ["neu"])

    def test_an_unreadable_config_does_not_block_setup(self):
        from src.ui.first_start_wizard import FirstStartWizard
        from src.utils.config_models import AlimaConfig

        wizard_config = AlimaConfig()
        cm = MagicMock()
        cm.load_config.side_effect = OSError("kaputt")
        wiz = FirstStartWizard.__new__(FirstStartWizard)
        with patch("src.ui.first_start_wizard.ConfigManager", return_value=cm):
            self.assertIs(wiz._apply_onto_existing(wizard_config), wizard_config)


class TheMenuEntryIsWiredTest(unittest.TestCase):
    """The entry point only became safe once the merge above existed."""

    def test_the_main_window_has_a_handler(self):
        from src.ui._main_window_settings import MainWindowSettingsMixin

        self.assertTrue(hasattr(MainWindowSettingsMixin, "show_setup_wizard"))

    def test_the_menu_calls_it(self):
        import inspect

        from src.ui import _main_window_menu

        self.assertIn(
            "self.show_setup_wizard", inspect.getsource(_main_window_menu)
        )


if __name__ == "__main__":
    unittest.main()
