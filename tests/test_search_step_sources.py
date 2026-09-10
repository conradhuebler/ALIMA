"""The classic search step offers the sources that actually exist.

``SearchStepConfigWidget`` had three hardcoded checkboxes — lobid, swb and
"Lokaler Katalog" — and ``get_config`` could only ever emit those three ids. Of
the four built-in GND-keyword providers, ``gnd_local`` was unreachable, and so
was every directory plugin. CLAUDE.md forbids hardcoded provider lists for
exactly this reason.

The list now comes from ``factory.enabled_gnd_provider_ids`` — the same gate
``pipeline_utils.execute_gnd_search`` intersects the request against — following
the pattern SearchTab adopted in WP-K5.

This is also what makes the ``is_primary`` removal honest: the classic path fixes
its plugins explicitly here, instead of a flag deciding invisibly.
- Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import patch


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class SearchStepSourcesComeFromThePluginSystemTest(unittest.TestCase):
    _kept: list = []

    def _widget(self, ids):
        from src.ui.step_config_widgets import SearchStepConfigWidget

        _qapp(self._kept)
        with patch(
            "src.core.search.factory.enabled_gnd_provider_ids", return_value=ids
        ):
            w = SearchStepConfigWidget("Suche")
        self._kept.append(w)
        return w

    def test_every_enabled_gnd_provider_gets_a_checkbox(self):
        w = self._widget(["lobid", "swb", "catalog", "gnd_local"])
        self.assertEqual(
            set(w.source_checkboxes), {"lobid", "swb", "catalog", "gnd_local"}
        )

    def test_gnd_local_is_reachable(self):
        """It never was: the three hardcoded boxes did not include it."""
        w = self._widget(["lobid", "gnd_local"])
        w.source_checkboxes["gnd_local"].setChecked(True)
        self.assertIn("gnd_local", w.get_config()["suggesters"])

    def test_an_own_plugin_is_offered(self):
        w = self._widget(["lobid", "poc_lobid"])
        self.assertIn("poc_lobid", w.source_checkboxes)

    def test_get_config_returns_the_checked_ids(self):
        w = self._widget(["lobid", "swb", "catalog"])
        w.source_checkboxes["lobid"].setChecked(True)
        w.source_checkboxes["swb"].setChecked(False)
        w.source_checkboxes["catalog"].setChecked(True)
        self.assertEqual(
            sorted(w.get_config()["suggesters"]), ["catalog", "lobid"]
        )

    def test_set_config_drops_a_source_that_no_longer_exists(self):
        """A stored id whose plugin is gone has no checkbox — same gate as runtime."""
        w = self._widget(["lobid", "swb"])
        w.set_config({"suggesters": ["lobid", "verschwundene_quelle"]})
        self.assertEqual(w.get_config()["suggesters"], ["lobid"])

    def test_an_unreadable_config_falls_back_to_the_defaults(self):
        """None means "cannot read", not "operator disabled everything"."""
        w = self._widget(None)
        self.assertEqual(set(w.source_checkboxes), {"lobid", "swb"})

    def test_an_explicit_empty_list_stays_empty(self):
        """The operator disabled every GND source; do not search anyway."""
        w = self._widget([])
        self.assertEqual(w.source_checkboxes, {})
        self.assertEqual(w.get_config()["suggesters"], [])

    def test_refresh_keeps_the_checked_state_of_known_sources(self):
        w = self._widget(["lobid", "swb"])
        w.source_checkboxes["swb"].setChecked(False)
        with patch(
            "src.core.search.factory.enabled_gnd_provider_ids",
            return_value=["lobid", "swb", "gnd_local"],
        ):
            w.refresh_sources()
        self.assertFalse(w.source_checkboxes["swb"].isChecked())
        self.assertTrue(w.source_checkboxes["lobid"].isChecked())


if __name__ == "__main__":
    unittest.main()
