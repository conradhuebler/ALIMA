"""The plugin settings form: choice handling and what it shows.

Two things this pins:

* A stored CHOICE value that is not among the declared choices used to leave the
  combo on index 0, and ``_flush_form`` wrote that back — so opening a plugin
  panel could silently change a setting on the next save. ``ConfigField.coerce``
  had the correct fallback (the declared default) all along; the form now uses it.
* The form shows the type's **capabilities**. They are what the runtime selects on
  (``enabled_gnd_provider_ids``, ``resolve_dk_extractor``); ``type_meta`` computed
  them all along and the form rendered prose and a since-removed "DK-Backend"
  combo instead.

- Claude Generated
"""

from __future__ import annotations

import unittest


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class ChoiceFieldFallsBackToTheDeclaredDefaultTest(unittest.TestCase):
    _kept: list = []

    def _widget_for(self, stored):
        from src.core.plugins.schema import CHOICE, ConfigField
        from src.ui.plugin_settings_tab import _CategoryPanel

        _qapp(self._kept)
        fld = ConfigField(
            key="mode", label="Modus", kind=CHOICE,
            choices=["fast", "safe", "off"], default="safe",
        )
        panel = _CategoryPanel.__new__(_CategoryPanel)
        return fld, _CategoryPanel._make_field_widget(panel, fld, stored, instance_id="x")

    def test_a_valid_value_is_selected(self):
        _fld, w = self._widget_for("off")
        self.assertEqual(w.currentData(), "off")

    def test_an_unknown_value_falls_back_to_the_default_not_the_first_choice(self):
        """The bug: "fast" is choices[0], "safe" is the declared default."""
        _fld, w = self._widget_for("was-mal-gültig-war")
        self.assertEqual(w.currentData(), "safe")

    def test_a_legacy_bool_falls_back_to_the_default(self):
        """cache_responses was a bool before it became auto/on/off."""
        _fld, w = self._widget_for(True)
        self.assertEqual(w.currentData(), "safe")

    def test_an_absent_value_uses_the_default(self):
        _fld, w = self._widget_for(None)
        self.assertEqual(w.currentData(), "safe")


class CapabilityLabelsCoverTheEnumTest(unittest.TestCase):
    """The label map must not fall behind SearchCapability."""

    def test_every_capability_has_a_label(self):
        from src.core.search.provider import SearchCapability
        from src.ui.plugin_settings_tab import _CAPABILITY_LABELS

        for cap in SearchCapability:
            self.assertIn(cap.value, _CAPABILITY_LABELS)

    def test_the_search_category_reports_capabilities(self):
        """Guard for the source the form reads: type_meta must carry them."""
        from src.core.search.factory import SearchProviderCategory

        meta = SearchProviderCategory().type_meta("lobid")
        self.assertIn("gnd_keywords", list(getattr(meta, "capabilities", []) or []))


if __name__ == "__main__":
    unittest.main()


class InstanceOfAnUnloadedTypeDoesNotCrashTheFormTest(unittest.TestCase):
    """Eine gespeicherte Instanz eines nicht geladenen Plugin-Typs (Code-Plugin
    gesperrt oder nicht freigegeben) ließ ``_build_form`` mit ``KeyError``
    scheitern; als Slot-Exception beendete das das Programm. - Claude Generated"""

    _kept: list = []

    def _panel(self):
        from src.utils.config_models import PluginInstanceConfig
        from src.ui.plugin_settings_tab import _CategoryPanel, _ensure_categories

        _qapp(self._kept)
        _ensure_categories()
        panel = _CategoryPanel("lookup")
        ghost = PluginInstanceConfig(
            instance_id="ghost", category="lookup", provider_id="not_loaded_plugin",
            label="Nicht geladen", enabled=True, settings={"db_path": "/x"},
        )
        panel.load([ghost])
        return panel, ghost

    def test_selecting_the_instance_does_not_raise(self):
        panel, _ = self._panel()
        panel.list.setCurrentRow(0)  # löst _on_row_changed aus

    def test_the_settings_survive_selecting_and_collecting(self):
        panel, _ = self._panel()
        panel.list.setCurrentRow(0)
        out = panel.collect()
        self.assertEqual(out[0].settings, {"db_path": "/x"})
        self.assertTrue(out[0].enabled)

    def test_switching_back_to_a_loaded_instance_works_after_the_ghost(self):
        from src.utils.config_models import PluginInstanceConfig

        panel, ghost = self._panel()
        real = PluginInstanceConfig(
            instance_id="k10plus", category="lookup", provider_id="k10plus",
            label="k10plus", enabled=True, settings={},
        )
        panel.load([ghost, real])
        panel.list.setCurrentRow(0)
        panel.list.setCurrentRow(1)
        panel.list.setCurrentRow(0)
        panel.collect()


class RescanKeepsTheDialogStateTest(unittest.TestCase):
    """Der Scan lädt eine frische Konfiguration. Ihr gespeicherter Stand (Code-
    Plugins aus) setzte den Haken zurück, mit OK wurde "aus" gespeichert.
    - Claude Generated"""

    _kept: list = []

    def _run_scan(self, *, checked, edit_label=None):
        from types import SimpleNamespace
        from unittest import mock

        from src.ui.plugin_settings_tab import PluginSettingsTab
        from src.utils.config_models import PluginInstanceConfig

        _qapp(self._kept)
        k10 = PluginInstanceConfig(instance_id="k10plus", category="lookup",
                                   provider_id="k10plus", label="k10plus",
                                   enabled=True, settings={})
        found = PluginInstanceConfig(instance_id="neu", category="lookup",
                                     provider_id="k10plus", label="neu",
                                     enabled=True, settings={})

        def fresh_config():
            return SimpleNamespace(
                plugins=[PluginInstanceConfig(**vars(k10))],
                system_config=SimpleNamespace(enable_code_plugins=False),
                approved_plugins={},
            )

        class FakeManager:
            plugins_dir = "/nonexistent"
            def load_config(self): return fresh_config()
            def save_config(self, config): pass

        def fake_discover(config, **kw):
            return SimpleNamespace(plugins=[], instances=[found])

        tab = PluginSettingsTab()
        tab.load(fresh_config())
        tab._code_plugins_cb.setChecked(checked)
        if edit_label is not None:
            panel = tab._panels["lookup"]
            panel.list.setCurrentRow(0)
            panel._label_edit.setText(edit_label)   # wie eine Eingabe im Formular
        with mock.patch("src.utils.config_manager.ConfigManager", FakeManager), \
             mock.patch("src.utils.plugin_discovery.discover_plugins", fake_discover), \
             mock.patch("src.ui.plugin_settings_tab.QMessageBox.information"):
            tab._rescan_plugins()
        return tab

    def test_the_checkbox_survives_the_scan(self):
        self.assertTrue(self._run_scan(checked=True)._code_plugins_cb.isChecked())

    def test_an_unchecked_box_stays_unchecked(self):
        self.assertFalse(self._run_scan(checked=False)._code_plugins_cb.isChecked())

    def test_unsaved_instance_edits_survive_and_new_instances_are_added(self):
        tab = self._run_scan(checked=True, edit_label="geändert")
        labels = {i.instance_id: i.label for i in tab._panels["lookup"].collect()}
        self.assertEqual(labels.get("k10plus"), "geändert")
        self.assertIn("neu", labels)
