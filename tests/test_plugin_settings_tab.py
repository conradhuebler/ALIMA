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
