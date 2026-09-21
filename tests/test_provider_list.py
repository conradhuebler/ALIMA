"""Providers as a plain list, and the model limits in one place.

An accordion was tried first and was wrong: five entries one wants to see at once
gain nothing from folding, and the fold hid mostly static text behind an
interaction. The operator said so after using it.

The model limits moved out of the per-provider card for the same reason: they are
rare — typically none — so one section listing every deviation beats five sections
that are empty. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.utils.config_models import UnifiedProvider


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class ProviderListTest(unittest.TestCase):
    _kept: list = []

    def _list(self, providers, models=None, thresholds=None):
        from src.ui.settings import ProviderList

        _qapp(self._kept)
        widget = ProviderList()
        widget.refresh(providers, models=models or {}, thresholds=thresholds or {})
        self._kept.append(widget)
        return widget

    def _p(self, name, **kw):
        return UnifiedProvider(name=name, provider_type=kw.pop("t", "ollama"), **kw)

    def test_every_provider_is_visible_at_once(self):
        widget = self._list([self._p(f"P{i}") for i in range(5)])
        self.assertEqual(widget.names(), [f"P{i}" for i in range(5)])

    def test_each_field_has_its_own_column(self):
        """Joining the fields with separators looked like formatting and was not:
        nothing lined up. Real columns, one value each."""
        widget = self._list(
            [self._p("GWDG", t="openai_compatible", base_url="https://x/v1")],
            models={"GWDG": ["m1", "m2"]},
        )
        self.assertEqual(
            widget.row_texts()[0], ("GWDG", "openai_compatible", "https://x/v1", "2", "")
        )

    def test_a_switched_off_provider_says_so(self):
        widget = self._list([self._p("A", enabled=False)])
        self.assertIn("aus", widget.row_texts()[0][4])

    def test_a_cloud_provider_without_a_key_says_so(self):
        """It is configured but cannot work; the row must not look healthy."""
        widget = self._list([self._p("G", t="gemini")])
        self.assertIn("kein Schlüssel", widget.row_texts()[0][4])

    def test_a_healthy_provider_has_an_empty_state_column(self):
        widget = self._list([self._p("A")])
        self.assertEqual(widget.row_texts()[0][4], "")

    def test_no_providers_says_what_that_means(self):
        widget = self._list([])
        self.assertIn("kein Anbieter", widget.row_texts()[0][0])
        self.assertEqual(widget.names(), [])

    def test_double_click_edits(self):
        widget = self._list([self._p("A")])
        seen = []
        widget.edit_requested.connect(seen.append)
        widget._on_double_click(widget.list.topLevelItem(0))
        self.assertEqual(seen, ["A"])

    def test_the_buttons_act_on_the_selection(self):
        widget = self._list([self._p("A"), self._p("B")])
        widget.select("B")
        seen = []
        widget.remove_requested.connect(seen.append)
        widget._emit_for_selection(widget.remove_requested)
        self.assertEqual(seen, ["B"])

    def test_buttons_needing_a_selection_are_disabled_without_one(self):
        widget = self._list([self._p("A")])
        widget.list.setCurrentItem(None)
        widget._update_buttons()
        for button in (widget.edit_button, widget.remove_button, widget.models_button):
            self.assertFalse(button.isEnabled())
        self.assertTrue(widget.add_button.isEnabled())

    def test_the_selection_survives_a_refresh(self):
        """Fetching one provider's models must not throw away what is selected."""
        widget = self._list([self._p("A"), self._p("B")])
        widget.select("B")
        widget.refresh([self._p("A"), self._p("B")], models={}, thresholds={})
        self.assertEqual(widget.selected_name(), "B")

    def test_the_list_shows_all_rows_without_scrolling(self):
        """Its height must cover the rows plus the header, not fall just short."""
        widget = self._list([self._p(f"P{i}") for i in range(5)])
        self.assertEqual(widget.list.minimumHeight(), widget.list.maximumHeight())
        needed = 5 * widget.list.sizeHintForRow(0)
        self.assertGreaterEqual(widget.list.maximumHeight(), needed)
        self.assertLess(widget.list.minimumHeight(), 250)


class ModelLimitsAreOneDeviationListTest(unittest.TestCase):
    _kept: list = []

    def _list(self, thresholds, providers=("A", "B"), models=None):
        from src.ui.settings import ProviderList

        _qapp(self._kept)
        widget = ProviderList()
        widget.refresh(
            [UnifiedProvider(name=n, provider_type="ollama") for n in providers],
            models=models or {n: ["m1", "m2"] for n in providers},
            thresholds=thresholds,
        )
        self._kept.append(widget)
        return widget

    def test_no_deviation_costs_one_folded_line(self):
        widget = self._list({})
        self.assertIn("keine Abweichung", widget.limits._button.text())
        self.assertFalse(widget.limits.is_expanded())

    def test_deviations_across_providers_are_counted_together(self):
        widget = self._list({"A": {"m1": 24000}, "B": {"m2": 8000}})
        self.assertIn("2 Abweichungen", widget.limits._button.text())

    def test_one_deviation_is_singular(self):
        self.assertIn("1 Abweichung", self._list({"A": {"m1": 24000}}).limits._button.text())

    def test_setting_a_limit_reports_provider_model_and_value(self):
        widget = self._list({})
        seen = []
        widget.chunking_changed.connect(lambda p, m, v: seen.append((p, m, v)))
        widget.limit_provider.setCurrentText("B")
        widget.limit_model.setCurrentText("m2")
        widget._add_limit()
        self.assertEqual(seen, [("B", "m2", 24000)])

    def test_the_model_choices_follow_the_chosen_provider(self):
        widget = self._list({}, models={"A": ["a1"], "B": ["b1", "b2"]})
        widget._refill_limit_models("B")
        offered = [widget.limit_model.itemText(i) for i in range(widget.limit_model.count())]
        self.assertEqual(offered, ["b1", "b2"])


if __name__ == "__main__":
    unittest.main()
