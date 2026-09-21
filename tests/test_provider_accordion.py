"""Providers as a list, and their model limits where they belong.

Three to six providers with six fields each is a list, not a table: a table has
to squeeze the endpoint and the key state into columns and then needs a dialog
for whatever did not fit.

The chunking threshold sits with the *model*, not with the task that triggers
it: it depends on the context window. Per task the same value would have to be
maintained once per task for the same model, and the copies would drift. Only
deviations from the automatic value are listed. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.utils.config_models import UnifiedProvider


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class ProviderAccordionTest(unittest.TestCase):
    _kept: list = []

    def _accordion(self, providers, models=None, thresholds=None):
        from src.ui.settings import ProviderAccordion

        _qapp(self._kept)
        acc = ProviderAccordion()
        acc.refresh(providers, models=models or {}, thresholds=thresholds or {})
        self._kept.append(acc)
        return acc

    def _p(self, name, **kw):
        return UnifiedProvider(name=name, provider_type=kw.pop("t", "ollama"), **kw)

    def test_one_card_per_provider(self):
        acc = self._accordion([self._p("A"), self._p("B")])
        self.assertEqual(sorted(acc.cards), ["A", "B"])

    def test_the_header_identifies_without_unfolding(self):
        acc = self._accordion([self._p("GWDG", t="openai_compatible")],
                              models={"GWDG": ["m1", "m2"]})
        header = acc.headers()[0]
        self.assertIn("GWDG", header)
        self.assertIn("openai_compatible", header)
        self.assertIn("2 Modelle", header)

    def test_a_switched_off_provider_says_so_in_the_header(self):
        acc = self._accordion([self._p("A", enabled=False)])
        self.assertIn("aus", acc.headers()[0])

    def test_cards_start_folded(self):
        acc = self._accordion([self._p("A")])
        self.assertFalse(acc.cards["A"].is_expanded())

    def test_refresh_replaces_instead_of_appending(self):
        acc = self._accordion([self._p("A")])
        acc.refresh([self._p("B")], models={}, thresholds={})
        self.assertEqual(list(acc.cards), ["B"])

    def test_no_providers_says_what_that_means(self):
        acc = self._accordion([])
        self.assertIn("kein Anbieter", acc._empty.text())
        self.assertTrue(acc._empty.isVisibleTo(acc))


class ModelLimitsAreADeviationListTest(unittest.TestCase):
    _kept: list = []

    def _card(self, thresholds, models=("m1", "m2")):
        from src.ui.settings import ProviderCard

        _qapp(self._kept)
        card = ProviderCard(
            UnifiedProvider(name="A", provider_type="ollama"),
            models=list(models), thresholds=dict(thresholds),
        )
        self._kept.append(card)
        return card

    def test_no_deviation_costs_one_line(self):
        card = self._card({})
        self.assertIn("keine Abweichung", card.limits._button.text())

    def test_a_single_deviation_is_counted_in_the_singular(self):
        card = self._card({"m1": 24000})
        self.assertIn("1 Abweichung", card.limits._button.text())

    def test_several_deviations_are_counted(self):
        card = self._card({"m1": 24000, "m2": 8000})
        self.assertIn("2 Abweichungen", card.limits._button.text())

    def test_the_limits_section_starts_folded(self):
        self.assertFalse(self._card({"m1": 24000}).limits.is_expanded())

    def test_setting_a_limit_reports_provider_model_and_value(self):
        card = self._card({})
        seen = []
        card.chunking_changed.connect(lambda p, m, v: seen.append((p, m, v)))
        card.add_model.setCurrentIndex(0)
        # the "Grenze setzen" button emits with the default starting value
        card.chunking_changed.emit("A", card.add_model.currentText(), 24000)
        self.assertEqual(seen, [("A", "m1", 24000)])

    def test_a_model_that_already_has_a_limit_is_not_offered_again(self):
        card = self._card({"m1": 24000})
        offered = [card.add_model.itemText(i) for i in range(card.add_model.count())]
        self.assertNotIn("m1", offered)
        self.assertIn("m2", offered)


if __name__ == "__main__":
    unittest.main()


class ItDoesNotBalloonTest(unittest.TestCase):
    """A folded card kept the vertical stretch its body asked for, so the cards
    ballooned and ran into each other. - Claude Generated"""

    _kept: list = []

    def _accordion(self, n=5):
        from src.ui.settings import ProviderAccordion

        _qapp(self._kept)
        acc = ProviderAccordion()
        acc.refresh(
            [UnifiedProvider(name=f"P{i}", provider_type="ollama") for i in range(n)],
            models={f"P{i}": [f"m{j}" for j in range(20)] for i in range(n)},
            thresholds={},
        )
        self._kept.append(acc)
        return acc

    def test_a_folded_card_is_one_line_tall(self):
        acc = self._accordion(1)
        card = next(iter(acc.cards.values()))
        self.assertLess(card.sizeHint().height(), 40, "zugeklappte Karte zu hoch")

    def test_a_card_never_claims_vertical_stretch(self):
        """The actual cause of the ballooning, asserted directly.

        sizeHint alone does not show it: a hidden body contributes nothing to the
        hint, while the size *policy* is what made the layout hand the card the
        leftover height.
        """
        from PyQt6.QtWidgets import QSizePolicy

        card = next(iter(self._accordion(1).cards.values()))
        self.assertEqual(
            card.sizePolicy().verticalPolicy(), QSizePolicy.Policy.Maximum
        )
        self.assertEqual(
            card.body().sizePolicy().verticalPolicy(), QSizePolicy.Policy.Maximum
        )

    def test_five_folded_cards_stay_compact(self):
        """Five providers must not fill the page before anything is opened."""
        self.assertLess(self._accordion(5).sizeHint().height(), 200)

    def test_unfolding_grows_only_that_card(self):
        acc = self._accordion(3)
        cards = list(acc.cards.values())
        before = [c.sizeHint().height() for c in cards]
        cards[0].expand()
        after = [c.sizeHint().height() for c in cards]
        self.assertGreater(after[0], before[0])
        self.assertEqual(after[1:], before[1:])
