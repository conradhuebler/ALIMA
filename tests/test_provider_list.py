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

    def _list(self, providers, models=None):
        from src.ui.settings import ProviderList

        _qapp(self._kept)
        widget = ProviderList()
        widget.refresh(providers, models=models or {})
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
        widget.refresh([self._p("A"), self._p("B")], models={})
        self.assertEqual(widget.selected_name(), "B")

    def test_the_list_shows_all_rows_without_scrolling(self):
        """Its height must cover the rows plus the header, not fall just short."""
        widget = self._list([self._p(f"P{i}") for i in range(5)])
        self.assertEqual(widget.list.minimumHeight(), widget.list.maximumHeight())
        needed = 5 * widget.list.sizeHintForRow(0)
        self.assertGreaterEqual(widget.list.maximumHeight(), needed)
        self.assertLess(widget.list.minimumHeight(), 250)


class ModelLimitsAreNoLongerEditedHereTest(unittest.TestCase):
    """The control is gone; the setting is not.

    ``model_capabilities.get_chunking_threshold`` still consults
    ``model_chunking_thresholds`` between an explicit call argument and the
    pattern match, and both the classic keyword step and every agentic step go
    through it. It is an escape hatch for a model the automatic detection gets
    wrong — no installation has ever needed one, so it stays reachable by hand in
    config.json instead of costing a section in the dialog. - Claude Generated
    """

    def test_the_provider_list_has_no_limits_section(self):
        from src.ui.settings import ProviderList

        _qapp([])
        self.assertFalse(hasattr(ProviderList, "limits"))

    def test_the_setting_is_still_read_at_runtime(self):
        from unittest.mock import MagicMock

        from src.utils.config_models import UnifiedProviderConfig
        from src.utils.model_capabilities import get_chunking_threshold

        cfg = UnifiedProviderConfig(providers=[])
        cfg.set_chunking_threshold("A", "m1", 4242)
        cm = MagicMock()
        cm.get_unified_config.return_value = cfg
        self.assertEqual(get_chunking_threshold("A", "m1", config_manager=cm), 4242)

    def test_an_explicit_argument_still_wins(self):
        from unittest.mock import MagicMock

        from src.utils.config_models import UnifiedProviderConfig
        from src.utils.model_capabilities import get_chunking_threshold

        cfg = UnifiedProviderConfig(providers=[])
        cfg.set_chunking_threshold("A", "m1", 4242)
        cm = MagicMock()
        cm.get_unified_config.return_value = cfg
        self.assertEqual(
            get_chunking_threshold("A", "m1", config_manager=cm, explicit_override=99),
            99,
        )


if __name__ == "__main__":
    unittest.main()


class LocalOllamaPresetTest(unittest.TestCase):
    """Vorkonfiguriertes lokales Ollama, per Häkchen aktivierbar. - Claude Generated"""

    _kept: list = []

    def test_preset_is_native_local_and_off(self):
        from src.utils.config_models import local_ollama_preset

        p = local_ollama_preset()
        self.assertEqual((p.provider_type, p.host, p.port, p.enabled),
                         ("ollama", "localhost", 11434, False))
        self.assertEqual(p.base_url, "http://localhost:11434")

    def test_detection_of_existing_local_native_ollama(self):
        from src.utils.config_models import has_local_native_ollama

        mk = lambda **kw: UnifiedProvider(name="x", **kw)
        self.assertFalse(has_local_native_ollama([]))
        self.assertTrue(has_local_native_ollama([mk(provider_type="ollama", host="localhost")]))
        self.assertTrue(has_local_native_ollama(
            [mk(provider_type="ollama", base_url="http://127.0.0.1:11434")]))
        # /v1-Endpunkt ist openai_compatible: zählt nicht als natives Ollama
        self.assertFalse(has_local_native_ollama(
            [mk(provider_type="openai_compatible", base_url="http://localhost:11434/v1")]))
        # Cloud-Ollama ist kein lokales
        self.assertFalse(has_local_native_ollama(
            [mk(provider_type="ollama", host="ollama.com")]))

    def _widget(self, providers, suggestions):
        from src.ui.settings import ProviderList

        _qapp(self._kept)
        w = ProviderList()
        w.refresh(providers, models={}, suggestions=suggestions)
        self._kept.append(w)
        return w

    def test_checkbox_reflects_enabled_and_suggestion_is_unchecked(self):
        from PyQt6.QtCore import Qt
        from src.utils.config_models import local_ollama_preset

        on = UnifiedProvider(name="A", provider_type="ollama", enabled=True)
        off = UnifiedProvider(name="B", provider_type="ollama", enabled=False)
        w = self._widget([on, off], [local_ollama_preset()])
        states = [w.list.topLevelItem(i).checkState(0) for i in range(3)]
        self.assertEqual(states, [Qt.CheckState.Checked, Qt.CheckState.Unchecked,
                                  Qt.CheckState.Unchecked])
        self.assertIn("vorkonfiguriert", w.row_texts()[2][4])

    def test_ticking_emits_name_and_state_but_refresh_does_not(self):
        from PyQt6.QtCore import Qt
        from src.utils.config_models import local_ollama_preset

        w = self._widget([], [local_ollama_preset()])
        seen = []
        w.enabled_toggled.connect(lambda n, e: seen.append((n, e)))
        w.refresh([], models={}, suggestions=[local_ollama_preset()])
        self.assertEqual(seen, [])
        w.list.topLevelItem(0).setCheckState(0, Qt.CheckState.Checked)
        self.assertEqual(seen, [("ollama_local", True)])
