"""Colour schemes ALIMA / Neutral / Desktop. - Claude Generated

``UIConfig.color_scheme`` selects the token tables in ``src/ui/styles.py``.
These tests hold the configuration contract (default, round trip, fallback),
the scheme switch in styles.py, the menu ↔ settings synchronisation through
the config, the disabled light/dark entry under "Desktop", and that a switch
leaves no colour of the previous scheme in widgets that set their own
stylesheet. Contrast of the tables: tests/test_design_tokens.py.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.ui import styles
from src.utils.config_models import AlimaConfig, UIConfig

_kept: list = []

BREEZE_DARK = dict(
    window="#31363b", window_text="#eff0f1", base="#232629", text="#eff0f1",
    button="#31363b", highlight="#3daee9", highlighted_text="#eff0f1",
    link="#1d99f3", mid="#4d5257", placeholder_text="#7f8c8d",
)


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


class _SchemeState(unittest.TestCase):
    """Restore the module-level scheme state after each test."""

    def setUp(self):
        self._scheme = styles.get_color_scheme()
        self._dark = styles.dark_mode_preference()
        self._desktop = (dict(styles._desktop_tables), styles._desktop_dark)

    def tearDown(self):
        styles.set_color_scheme(self._scheme)
        styles.set_dark_mode(self._dark)
        styles._desktop_tables.clear()
        styles._desktop_tables.update(self._desktop[0])
        styles._desktop_dark = self._desktop[1]


class UIConfigColorSchemeTest(unittest.TestCase):
    def test_default_is_alima(self):
        self.assertEqual(UIConfig().color_scheme, "alima")

    def test_known_values_are_kept_and_normalised(self):
        self.assertEqual(UIConfig(color_scheme="neutral").color_scheme, "neutral")
        self.assertEqual(UIConfig(color_scheme=" Desktop ").color_scheme, "desktop")

    def test_unknown_value_falls_back_with_a_log_entry(self):
        logging.disable(logging.NOTSET)
        with self.assertLogs("src.utils.config_models", level="WARNING") as caught:
            cfg = UIConfig(color_scheme="lila")
        self.assertEqual(cfg.color_scheme, "alima")
        self.assertIn("lila", "\n".join(caught.output))

    def _manager_on(self, path: Path):
        """The singleton pointed at a temporary file; ``_restore`` puts back
        both the path and the cached config (``save_config`` caches)."""
        from src.utils.config_manager import ConfigManager

        manager = ConfigManager()
        original = (manager.config_file, getattr(manager, "_config", None))
        manager.config_file = path
        return manager, original

    @staticmethod
    def _restore(manager, original):
        manager.config_file, manager._config = original

    def test_round_trip_through_the_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager, original = self._manager_on(Path(tmp) / "config.json")
            try:
                cfg = AlimaConfig()
                cfg.ui_config.color_scheme = "desktop"
                self.assertTrue(manager.save_config(cfg, preserve_unified=False))
                stored = json.loads((Path(tmp) / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(stored["ui_config"]["color_scheme"], "desktop")
                self.assertEqual(manager._load_config_from_file().ui_config.color_scheme, "desktop")
            finally:
                self._restore(manager, original)

    def test_unknown_value_in_the_file_loads_as_alima(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"config_version": "2.0",
                                        "ui_config": {"color_scheme": "lila", "font_size": 12}}),
                            encoding="utf-8")
            manager, original = self._manager_on(path)
            try:
                loaded = manager._load_config_from_file()
            finally:
                self._restore(manager, original)
        self.assertEqual(loaded.ui_config.color_scheme, "alima")
        self.assertEqual(loaded.ui_config.font_size, 12)  # the rest survives


class StylesSchemeSwitchTest(_SchemeState):
    def test_set_and_get(self):
        self.assertEqual(styles.set_color_scheme("neutral"), "neutral")
        self.assertEqual(styles.get_color_scheme(), "neutral")
        self.assertEqual(styles.set_color_scheme("unbekannt"), "alima")

    def test_alima_tables_are_the_unchanged_originals(self):
        styles.set_color_scheme("alima")
        styles.set_dark_mode(False)
        self.assertIs(styles.get_colors(), styles.COLORS)
        styles.set_dark_mode(True)
        self.assertIs(styles.get_colors(), styles.DARK_COLORS)

    def test_neutral_is_flat_and_free_of_logo_colours(self):
        styles.set_color_scheme("neutral")
        for dark in (False, True):
            styles.set_dark_mode(dark)
            c = styles.get_colors()
            self.assertEqual(styles._gradient(c), c["primary"] if not dark else c["grad_start"])
            alima = styles.DARK_COLORS if dark else styles.COLORS
            neutral_values = {v.lower() for v in c.values()}
            sheet = styles.get_main_stylesheet().lower()
            logo_only = {alima[k].lower() for k in ("primary", "grad_start", "grad_mid", "grad_end",
                                                    "accent_text", "indicator", "focus")}
            for value in logo_only - neutral_values:
                self.assertNotIn(value, sheet, f"ALIMA colour {value} in the Neutral stylesheet")

    def test_desktop_follows_the_system_not_the_preference(self):
        styles.set_desktop_palette(BREEZE_DARK, dark=True)
        styles.set_color_scheme("desktop")
        styles.set_dark_mode(False)
        self.assertTrue(styles.is_dark_mode())
        self.assertFalse(styles.dark_mode_preference())
        self.assertEqual(styles.get_colors()["background"], BREEZE_DARK["base"])
        styles.set_color_scheme("alima")
        self.assertFalse(styles.is_dark_mode())

    def test_desktop_without_a_snapshot_uses_neutral(self):
        styles._desktop_tables.clear()
        styles.set_color_scheme("desktop")
        styles._desktop_dark = False
        self.assertIs(styles.get_colors(), styles.NEUTRAL_COLORS)


class MenuAndSettingsSyncTest(_SchemeState):
    """Menu and settings page edit the same UIConfig.color_scheme."""

    def _menu_host(self):
        from PyQt6.QtWidgets import QMenuBar

        from src.ui._main_window_menu import MainWindowMenuMixin
        from src.ui._main_window_settings import MainWindowSettingsMixin

        _qapp()
        window = MagicMock()
        menubar = QMenuBar()
        _kept.append(menubar)
        window.menuBar.return_value = menubar
        MainWindowMenuMixin.create_menu_bar(window)
        cfg = AlimaConfig()
        window.config_manager.load_config.return_value = cfg
        window._dark_mode = False
        # the real switch logic on the stand-in
        for name in ("apply_color_scheme", "_sync_color_scheme_controls", "_update_theme_action"):
            setattr(window, name, getattr(MainWindowSettingsMixin, name).__get__(window))
        return window, cfg

    def test_menu_lists_the_three_schemes_with_the_active_one_ticked(self):
        styles.set_color_scheme("alima")
        window, _cfg = self._menu_host()
        self.assertEqual(list(window._scheme_actions), ["alima", "neutral", "desktop"])
        self.assertTrue(window._scheme_actions["alima"].isChecked())
        self.assertEqual(window._scheme_actions["desktop"].text(), "Desktop")

    def test_menu_switch_sets_style_config_and_ticks(self):
        styles.set_color_scheme("alima")
        window, cfg = self._menu_host()
        window._scheme_actions["neutral"].trigger()
        self.assertEqual(styles.get_color_scheme(), "neutral")
        self.assertEqual(cfg.ui_config.color_scheme, "neutral")
        window.config_manager.save_config.assert_called()
        self.assertTrue(window._scheme_actions["neutral"].isChecked())
        self.assertFalse(window._scheme_actions["alima"].isChecked())
        window.apply_theme.assert_called()

    def test_a_switch_from_elsewhere_moves_the_menu_tick(self):
        """The settings page (via config / load_settings) switches without
        touching the menu; the tick must follow."""
        styles.set_color_scheme("alima")
        window, _cfg = self._menu_host()
        window.apply_color_scheme("desktop")
        self.assertTrue(window._scheme_actions["desktop"].isChecked())
        self.assertFalse(window._scheme_actions["alima"].isChecked())

    def test_light_dark_entry_is_disabled_under_desktop_only(self):
        styles.set_desktop_palette(BREEZE_DARK, dark=True)
        styles.set_color_scheme("alima")
        window, _cfg = self._menu_host()
        window.apply_color_scheme("desktop")
        self.assertFalse(window._theme_action.isEnabled())
        self.assertIn("Desktop", window._theme_action.toolTip())
        window.apply_color_scheme("alima")
        self.assertTrue(window._theme_action.isEnabled())

    def test_settings_page_shows_and_writes_the_same_setting(self):
        from tests.test_settings_dialog_state import _DialogHarness

        class _Harness(_DialogHarness):
            def configure(self, cfg):
                cfg.ui_config.color_scheme = "neutral"

        dlg = _Harness()._dialog()
        self.assertEqual(dlg.color_scheme_combo.currentData(), "neutral")
        self.assertEqual(
            [dlg.color_scheme_combo.itemData(i) for i in range(dlg.color_scheme_combo.count())],
            ["alima", "neutral", "desktop"],
        )
        dlg.color_scheme_combo.setCurrentIndex(dlg.color_scheme_combo.findData("desktop"))
        self.assertEqual(dlg._get_config_from_ui().ui_config.color_scheme, "desktop")


class SwitchLeavesNoOldColoursTest(_SchemeState):
    """Widgets with their own stylesheet re-colour on the refresh walk."""

    HEX = re.compile(r"#[0-9a-fA-F]{6}\b")

    def _widgets(self):
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

        from src.ui.agentic_context_widget import AgenticStepPanel
        from src.ui.proposal_bar import ProposalBar
        from src.ui.repetition_warning_bar import RepetitionWarningBar

        _qapp()
        root = QWidget()
        lay = QVBoxLayout(root)
        bar = RepetitionWarningBar()
        bar.show_warning("ngram", "x", [], grace_period=False)
        panel = AgenticStepPanel("selection", "llm")
        panel.apply({"_step_status": "completed", "extra": {"a": 1}})
        label = QLabel("x")
        styles.set_tone(label, "warning")
        for w in (bar, ProposalBar(), panel, label):
            lay.addWidget(w)
        _kept.append(root)
        return root

    def _walk(self, root):
        from src.ui._main_window_settings import MainWindowSettingsMixin

        host = MagicMock()
        MainWindowSettingsMixin._walk_refresh_styles(host, root)

    def _colours(self, root):
        from PyQt6.QtWidgets import QWidget

        found = set()
        for w in [root] + root.findChildren(QWidget):
            found |= {h.lower() for h in self.HEX.findall(w.styleSheet())}
        return found

    def test_each_switch_drops_the_previous_schemes_colours(self):
        styles.set_desktop_palette(BREEZE_DARK, dark=True)
        styles.set_dark_mode(False)
        styles.set_color_scheme("alima")
        root = self._widgets()
        self.assertTrue(self._colours(root))  # the probe sees something
        previous = {v.lower() for v in styles.get_colors().values() if v.startswith("#")}
        for scheme in ("neutral", "desktop", "alima"):
            styles.set_color_scheme(scheme)
            self._walk(root)
            current = {v.lower() for v in styles.get_colors().values() if v.startswith("#")}
            stale = self._colours(root) & (previous - current)
            self.assertEqual(stale, set(), f"after switching to {scheme}")
            previous = current


if __name__ == "__main__":
    unittest.main()
