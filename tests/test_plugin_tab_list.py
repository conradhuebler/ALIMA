"""The instance list of the plugin tab - Claude Generated.

Pinned here (Nov 2026):

1. A three-state checkbox per row (An / Nur Cache / Aus) that is the same
   state as the form's "Zustand": a click moves it on, "Nur Cache" is skipped
   where the type does not offer it, an instance of a type that is not loaded
   cannot be switched, a status line names the new state.
2. The order of the rows is the order ``collect`` / ``apply_to`` write; moving a
   row (``model().moveRow``, the path of a drop, and Alt+Up/Down) loses nothing,
   duplicates nothing, keeps the selection and the unsaved form input.
3. An origin tag per row, derived on load (eingebaut, Verzeichnis, Code-Plugin
   freigegeben / gesperrt / nicht geladen, unbekannt).
4. The DK switch shows the effective state of an unstored key and writes
   nothing while unchanged; "DK n" tags and the line above the list follow the
   unsaved dialog state.
5. Round trip: loading a config shaped like the operator's and applying it
   unchanged leaves the plugin entries as they were.

Real widgets, offscreen (tests/__init__.py). HOME points at a temporary
directory; no test touches ``~/.config/alima`` or the network.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from src.utils.config_models import PluginInstanceConfig

_KEEP: list = []


def _app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["alima-tests"])
    _KEEP.append(app)
    return app


def _inst(pid, *, iid=None, category="search_provider", enabled=True, label="", **settings):
    return PluginInstanceConfig(
        instance_id=iid or pid, category=category, provider_id=pid,
        label=label, enabled=enabled, settings=dict(settings),
    )


def _write_manifest(root: Path, plugin_id: str, text: str, files=None) -> Path:
    d = root / plugin_id
    d.mkdir(parents=True)
    (d / "plugin.toml").write_text(text, encoding="utf-8")
    for name, src in (files or {}).items():
        (d / name).write_text(src, encoding="utf-8")
    return d


def _code_manifest(plugin_id, category="search_provider"):
    return (f'[plugin]\nid="{plugin_id}"\nlabel="Code {plugin_id}"\ncategory="{category}"\n'
            'type="code"\n[entry]\nmodule="provider.py"\nclass="P"\n')


def _declarative_manifest(plugin_id, kind, category="search_provider"):
    return (f'[plugin]\nid="{plugin_id}"\nlabel="Dekl {plugin_id}"\ncategory="{category}"\n'
            f'type="declarative"\nkind="{kind}"\n[settings]\nbase_url="https://f.example"\n')


class _Case(unittest.TestCase):
    def setUp(self):
        from PyQt6.QtWidgets import QApplication

        if not isinstance(_app(), QApplication):
            self.skipTest("a bare QCoreApplication exists; widgets cannot be built")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = patch.dict(os.environ, {"HOME": tmp.name})
        home.start()
        self.addCleanup(home.stop)
        self.tmp = Path(tmp.name)
        # PluginSettingsTab() builds a ConfigManager (preset group); it must not
        # outlive the temporary HOME, nor may an earlier one be reused here.
        from src.utils.config_manager import ConfigManager

        ConfigManager.reset()
        self.addCleanup(ConfigManager.reset)

    def panel(self, instances, category="search_provider", show=False):
        from PyQt6.QtWidgets import QApplication

        from src.ui.plugin_settings_tab import _CategoryPanel, _ensure_categories

        _ensure_categories()
        panel = _CategoryPanel(category)
        _KEEP.append(panel)
        panel.load(instances)
        if show:
            panel.resize(1000, 650)
            panel.show()
            QApplication.processEvents()
            self.addCleanup(panel.close)
        return panel

    @staticmethod
    def ids(panel):
        return [i.instance_id for i in panel.collect()]

    @staticmethod
    def texts(panel):
        return [panel.list.item(r).text() for r in range(panel.list.count())]

    @staticmethod
    def indicator(panel, row):
        from PyQt6.QtWidgets import QStyle, QStyleOptionViewItem

        lst = panel.list
        index = lst.model().index(row, 0)
        opt = QStyleOptionViewItem()
        lst.initViewItemOption(opt)
        opt.rect = lst.visualRect(index)
        lst.itemDelegate().initStyleOption(opt, index)
        rect = lst.style().subElementRect(
            QStyle.SubElement.SE_ItemViewItemCheckIndicator, opt, lst)
        return index, opt, rect

    def click_box(self, panel, row):
        """A real mouse click on the row's check indicator."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        _index, _opt, rect = self.indicator(panel, row)
        self.assertTrue(rect.isValid())
        QTest.mouseClick(panel.list.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, rect.center())


# ===========================================================================
# 1. Three-state checkbox
# ===========================================================================
class TestStateCycle(_Case):
    def test_next_state(self):
        from src.ui.plugin_settings_tab import (OFFER_BLOCKED, OFFER_CACHE_ONLY, OFFER_NONE,
                                                next_state)

        self.assertEqual([next_state(s, OFFER_CACHE_ONLY) for s in ("on", "cache_only", "off")],
                         ["cache_only", "off", "on"])
        for offer in (OFFER_NONE, OFFER_BLOCKED):
            self.assertEqual([next_state(s, offer) for s in ("on", "off")], ["off", "on"])

    def _states_by_click(self, panel, row, clicks):
        seen = []
        for _ in range(clicks):
            self.click_box(panel, row)
            out = panel.collect()[row]
            seen.append((out.enabled, out.settings.get("cache_only"),
                         panel.list.item(row).checkState().name))
        return seen

    def test_click_cycles_on_cache_only_off(self):
        panel = self.panel([_inst("lobid")], show=True)
        self.assertEqual(panel.list.item(0).checkState().name, "Checked")
        self.assertEqual(self._states_by_click(panel, 0, 3), [
            (True, True, "PartiallyChecked"),
            (False, True, "Unchecked"),
            (True, False, "Checked"),
        ])

    def test_click_skips_cache_only_where_the_type_has_none(self):
        for category, inst in (
            ("search_provider", _inst("gnd_local")),
            ("lookup", _inst("k10plus", category="lookup")),
            ("input_source", _inst("doi_datacite", category="input_source")),
        ):
            panel = self.panel([inst], category=category, show=True)
            seen = self._states_by_click(panel, 0, 2)
            self.assertEqual([s[0] for s in seen], [False, True], category)
            self.assertEqual([s[2] for s in seen], ["Unchecked", "Checked"], category)
            self.assertNotIn("cache_only", panel.collect()[0].settings, category)


class TestLockedRow(_Case):
    def _panel(self):
        ghost = _inst("not_loaded_plugin", iid="ghost", label="Geist", db_path="/x")
        return self.panel([_inst("lobid"), ghost], show=True)

    def test_unloaded_type_cannot_be_switched(self):
        from PyQt6.QtCore import Qt

        from src.ui.plugin_settings_tab import _CHECK_TIP_ROLE, _LOCKED_ROLE

        panel = self._panel()
        item = panel.list.item(1)
        self.assertFalse(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
        self.assertTrue(panel.list.item(0).flags() & Qt.ItemFlag.ItemIsUserCheckable)
        self.assertTrue(item.data(_LOCKED_ROLE))
        self.click_box(panel, 1)
        out = panel.collect()[1]
        self.assertTrue(out.enabled)
        self.assertEqual(out.settings, {"db_path": "/x"})
        self.assertEqual(item.checkState(), Qt.CheckState.Checked)
        tip = item.data(_CHECK_TIP_ROLE)
        self.assertTrue(tip.startswith("Nicht schaltbar."), tip)
        self.assertIn("nicht geladen", tip)

    def test_a_programmatic_change_is_reverted(self):
        from PyQt6.QtCore import Qt

        panel = self._panel()
        panel.list.item(1).setCheckState(Qt.CheckState.Unchecked)
        self.assertEqual(panel.list.item(1).checkState(), Qt.CheckState.Checked)
        self.assertTrue(panel.collect()[1].enabled)


class TestBoxAndFormAreOneState(_Case):
    def test_click_on_another_row(self):
        """Qt 6.10 switches the box without selecting the row: the form of the
        selected row stays as it is."""
        panel = self.panel([_inst("lobid"), _inst("finc", base_url="https://f")], show=True)
        self.click_box(panel, 1)
        out = panel.collect()[1]
        self.assertEqual((out.enabled, out.settings.get("cache_only")), (True, True))
        self.assertEqual(panel.status_label.text(), "finc: nur Cache (keine Abfragen im Netz)")
        if panel._current_row == 0:
            self.assertEqual(panel._state_combo.currentData(), "on")
            self.assertTrue(panel.collect()[0].enabled)
        else:  # a Qt that selects on press shows the switched row
            self.assertEqual(panel._state_combo.currentData(), "cache_only")
        panel.list.setCurrentRow(1)
        self.assertEqual(panel._state_combo.currentData(), "cache_only")

    def test_a_row_that_is_not_selected(self):
        from PyQt6.QtCore import Qt

        panel = self.panel([_inst("lobid"), _inst("swb")])
        panel.list.setCurrentRow(0)
        panel.list.item(1).setCheckState(Qt.CheckState.PartiallyChecked)  # what Qt sets
        self.assertEqual(panel._current_row, 0)
        self.assertEqual(panel._state_combo.currentData(), "on")      # row 0 untouched
        self.assertEqual(panel.collect()[1].settings.get("cache_only"), True)
        self.assertEqual(panel.status_label.text(), "swb: nur Cache (keine Abfragen im Netz)")

    def test_the_click_on_the_selected_row_moves_the_combo(self):
        panel = self.panel([_inst("lobid")], show=True)
        self.click_box(panel, 0)
        self.click_box(panel, 0)
        self.assertEqual(panel._state_combo.currentData(), "off")
        self.assertFalse(panel.collect()[0].enabled)
        self.assertEqual(panel.status_label.text(), "lobid: aus")

    def test_the_combo_moves_the_box(self):
        from PyQt6.QtCore import Qt

        panel = self.panel([_inst("lobid"), _inst("swb")])
        panel.list.setCurrentRow(1)
        panel._state_combo.setCurrentIndex(panel._state_combo.findData("off"))
        self.assertEqual(panel.list.item(1).checkState(), Qt.CheckState.Unchecked)
        self.assertIn("(aus)", panel.list.item(1).text())
        self.assertFalse(panel.collect()[1].enabled)
        self.assertEqual(panel.status_label.text(), "swb: aus")

    def test_the_status_line_is_emptied_on_a_row_change(self):
        panel = self.panel([_inst("lobid"), _inst("swb")])
        panel._state_combo.setCurrentIndex(panel._state_combo.findData("off"))
        self.assertEqual(panel.status_label.text(), "lobid: aus")
        panel.list.setCurrentRow(1)
        self.assertEqual(panel.status_label.text(), "")

    def test_the_status_line_uses_the_unsaved_name(self):
        panel = self.panel([_inst("lobid", label="Lobid")])
        panel._label_edit.setText("Lobid neu")
        panel._state_combo.setCurrentIndex(panel._state_combo.findData("off"))
        self.assertEqual(panel.status_label.text(), "Lobid neu: aus")
        self.assertTrue(panel.list.item(0).text().startswith("Lobid neu [lobid]"))

    def test_typing_a_name_updates_the_row(self):
        panel = self.panel([_inst("lobid", label="Lobid")])
        panel._label_edit.setText("Lobid neu")
        self.assertTrue(panel.list.item(0).text().startswith("Lobid neu [lobid]"))
        self.assertEqual(panel.collect()[0].label, "Lobid neu")

    def test_checkbox_tooltip_follows_the_state(self):
        from src.ui.plugin_settings_tab import _CHECK_TIP_ROLE

        panel = self.panel([_inst("lobid"), _inst("gnd_local")], show=True)
        self.assertEqual(panel.list.item(0).data(_CHECK_TIP_ROLE),
                         "An: die Instanz wird benutzt. Ein Klick schaltet auf „Nur Cache“.")
        self.assertTrue(panel.list.item(1).data(_CHECK_TIP_ROLE).endswith("„Aus“."))
        self.click_box(panel, 0)
        self.assertTrue(panel.list.item(0).data(_CHECK_TIP_ROLE).startswith("Nur Cache:"))

    def test_the_indicator_has_its_own_tooltip(self):
        from src.ui.plugin_settings_tab import _CHECK_TIP_ROLE

        panel = self.panel([_inst("lobid")], show=True)
        index, opt, rect = self.indicator(panel, 0)
        delegate = panel.list.itemDelegate()
        on_box = delegate.tooltip_at(panel.list, opt, index, rect.center())
        beside = delegate.tooltip_at(panel.list, opt, index, opt.rect.center())
        self.assertEqual(on_box, panel.list.item(0).data(_CHECK_TIP_ROLE))
        self.assertEqual(beside, panel.list.item(0).toolTip())
        self.assertIn("Eingebaut", beside)


class TestIndicatorColours(_Case):
    """The list checkbox is drawn from the design tokens; Fusion drew its frame
    from the palette role Dark, nearly invisible on the list in dark themes."""

    STATES = ("on", "cache_only", "off")

    def _tables(self):
        """(name, token table, list backgrounds) for every scheme and theme."""
        from PyQt6.QtGui import QPalette

        from src.alima_gui import _set_palette_roles
        from src.ui.styles import derive_desktop_colors, scheme_colors
        from tests.test_design_tokens import DESKTOP_PALETTES

        out = []
        for scheme in ("alima", "neutral"):
            for dark in (False, True):
                c = scheme_colors(scheme, dark)
                palette = QPalette()
                _set_palette_roles(palette, c, dark)
                base = palette.color(QPalette.ColorRole.Base).name()
                out.append((f"{scheme}/{'dunkel' if dark else 'hell'}", c, [base]))
        for name, (dark, roles) in DESKTOP_PALETTES.items():
            out.append((f"desktop/{name}", derive_desktop_colors(roles, dark), [roles["base"]]))
        return out

    def test_contrast_in_every_scheme(self):
        from src.ui.plugin_settings_tab import indicator_colors
        from src.ui.styles import contrast_ratio

        failures = []
        for name, c, bases in self._tables():
            for state in self.STATES:
                for selected in (False, True):
                    for locked in (False, True):
                        col = indicator_colors(c, state, selected=selected, locked=locked)
                        grounds = [c["primary"]] if selected else bases
                        for ground in grounds:
                            r = contrast_ratio(col["border"], ground)
                            if r < 3.0:
                                failures.append(f"{name} {state} sel={selected} lock={locked}: "
                                                f"Rand {col['border']} auf {ground} = {r:.2f}")
                        if col["mark"] is not None:
                            r = contrast_ratio(col["mark"], col["fill"])
                            if r < 3.0:
                                failures.append(f"{name} {state} sel={selected} lock={locked}: "
                                                f"Zeichen {col['mark']} auf {col['fill']} = {r:.2f}")
        self.assertEqual(failures, [])

    def test_the_colours_are_tokens(self):
        from src.ui.plugin_settings_tab import indicator_colors
        from src.ui.styles import get_colors

        c = get_colors()
        tokens = set(c.values())
        for state in self.STATES:
            for selected in (False, True):
                for locked in (False, True):
                    col = indicator_colors(c, state, selected=selected, locked=locked)
                    for key, value in col.items():
                        if value is not None:
                            self.assertIn(value, tokens, (state, selected, locked, key))
        self.assertEqual(indicator_colors(c, "off")["border"], c["border"])
        self.assertIsNone(indicator_colors(c, "off")["mark"])

    def _pixels(self, dark):
        from PyQt6.QtGui import QColor

        from src.ui import styles

        before = (styles.is_dark_mode(), styles.get_color_scheme())
        styles.set_color_scheme("alima")
        styles.set_dark_mode(dark)
        self.addCleanup(styles.set_color_scheme, before[1])
        self.addCleanup(styles.set_dark_mode, before[0])
        c = styles.get_colors()
        panel = self.panel([_inst("lobid"), _inst("kvk", enabled=False),
                            _inst("swb", cache_only=True)], show=True)
        image = panel.list.viewport().grab().toImage()

        def at(row, where):
            _index, _opt, rect = self.indicator(panel, row)
            point = {"left": (rect.left(), rect.center().y()),
                     "center": (rect.center().x(), rect.center().y())}[where]
            return image.pixelColor(*point).name()

        return c, at, QColor

    def _check_theme(self, dark):
        c, at, QColor = self._pixels(dark)
        self.assertEqual(at(1, "left"), QColor(c["border"]).name())       # Aus: frame
        self.assertEqual(at(1, "center"), QColor(c["background"]).name())
        self.assertEqual(at(2, "left"), QColor(c["border"]).name())       # Nur Cache
        self.assertEqual(at(2, "center"), QColor(c["primary"]).name())
        self.assertEqual(at(0, "left"), QColor(c["on_primary"]).name())   # selected row

    def test_dark_theme_draws_the_token_frame(self):
        self._check_theme(True)

    def test_light_theme_draws_the_token_frame(self):
        self._check_theme(False)


# ===========================================================================
# 2. Order
# ===========================================================================
class TestOrder(_Case):
    FOUR = ("lobid", "swb", "catalog", "finc")

    def _panel(self, show=False):
        return self.panel([_inst(p, label=p.upper()) for p in self.FOUR], show=show)

    def test_move_row_is_the_order_of_collect(self):
        from PyQt6.QtCore import QModelIndex

        panel = self._panel()
        panel.list.setCurrentRow(1)
        panel._label_edit.setText("ungespeichert")          # not flushed yet
        self.assertTrue(panel.list.model().moveRow(QModelIndex(), 1, QModelIndex(), 4))
        self.assertEqual(self.ids(panel), ["lobid", "catalog", "finc", "swb"])
        self.assertEqual(sorted(self.ids(panel)), sorted(self.FOUR))
        self.assertEqual([t.split(" [")[0] for t in self.texts(panel)],
                         ["LOBID", "CATALOG", "FINC", "ungespeichert"])
        self.assertEqual((panel._current_row, panel.list.currentRow()), (3, 3))
        self.assertEqual(panel._label_edit.text(), "ungespeichert")
        self.assertEqual([i.label for i in panel.collect()],
                         ["LOBID", "CATALOG", "FINC", "ungespeichert"])

    def test_moving_another_row_keeps_the_selected_instance(self):
        from PyQt6.QtCore import QModelIndex

        panel = self._panel()
        panel.list.setCurrentRow(2)                          # catalog
        panel.list.model().moveRow(QModelIndex(), 0, QModelIndex(), 4)
        self.assertEqual(self.ids(panel), ["swb", "catalog", "finc", "lobid"])
        self.assertEqual(panel._current_row, 1)
        self.assertEqual(panel.list.currentRow(), 1)
        self.assertEqual(panel._label_edit.text(), "CATALOG")
        panel.list.setCurrentRow(3)                          # flush catalog into the right row
        self.assertEqual([i.label for i in panel.collect()],
                         ["SWB", "CATALOG", "FINC", "LOBID"])

    def test_alt_arrows(self):
        from PyQt6.QtCore import Qt
        from PyQt6.QtTest import QTest

        panel = self._panel(show=True)
        panel.list.setCurrentRow(0)
        QTest.keyClick(panel.list, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(self.ids(panel), ["swb", "lobid", "catalog", "finc"])
        self.assertEqual(panel._current_row, 1)
        QTest.keyClick(panel.list, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(self.ids(panel), ["swb", "catalog", "lobid", "finc"])
        QTest.keyClick(panel.list, Qt.Key.Key_Up, Qt.KeyboardModifier.AltModifier)
        self.assertEqual(self.ids(panel), ["swb", "lobid", "catalog", "finc"])
        self.assertEqual((panel._current_row, panel.list.currentRow()), (1, 1))
        self.assertEqual(panel._label_edit.text(), "LOBID")
        # Without Alt the arrows only move the selection.
        QTest.keyClick(panel.list, Qt.Key.Key_Down)
        self.assertEqual(self.ids(panel), ["swb", "lobid", "catalog", "finc"])
        self.assertEqual(panel._current_row, 2)

    def test_a_new_load_does_not_take_the_previous_form(self):
        """The form of the previous list was written into the new instance at
        the old row on the first row change after ``load``."""
        panel = self.panel([_inst("lobid", label="A"), _inst("swb", label="B")])
        panel.list.setCurrentRow(1)
        panel._label_edit.setText("ungespeichert")
        panel.load([_inst("finc", label="F"), _inst("kvk", label="K")])
        self.assertEqual([i.label for i in panel.collect()], ["F", "K"])
        panel.list.setCurrentRow(1)
        self.assertEqual([i.label for i in panel.collect()], ["F", "K"])

    def test_the_ends_do_not_move(self):
        panel = self._panel()
        panel.list.setCurrentRow(0)
        panel.move_current(-1)
        panel.list.setCurrentRow(3)
        panel.move_current(1)
        self.assertEqual(self.ids(panel), list(self.FOUR))

    def test_dk_tags_and_line_follow_the_order(self):
        panel = self.panel([
            _inst("catalog", label="Katalog (Libero)", dk_enabled=True),
            _inst("finc", label="finc (VuFind)", base_url="https://f", dk_enabled=True)])
        self.assertIn("  ·  DK 1  ·  ", self.texts(panel)[0])
        self.assertIn("  ·  DK 2  ·  ", self.texts(panel)[1])
        self.assertTrue(panel._dk_label.text().startswith("DK-Quellen im Moment: Katalog"))
        panel.list.setCurrentRow(1)
        panel.move_current(-1)
        self.assertEqual(self.ids(panel), ["finc", "catalog"])
        self.assertIn("  ·  DK 1  ·  ", self.texts(panel)[0])
        self.assertTrue(self.texts(panel)[0].startswith("finc (VuFind)"))
        self.assertTrue(panel._dk_label.text().startswith("DK-Quellen im Moment: finc"))

    def test_apply_to_writes_the_new_order_and_keeps_the_categories(self):
        from types import SimpleNamespace

        from src.ui.plugin_settings_tab import PluginSettingsTab

        plugins = [
            _inst("doi_datacite", category="input_source"),
            _inst("k10plus", category="lookup"),
            _inst("lobid"), _inst("swb"), _inst("catalog"),
            _inst("x", category="not_managed_here"),
        ]
        config = SimpleNamespace(plugins=list(plugins), approved_plugins={},
                                 system_config=SimpleNamespace(enable_code_plugins=False))
        tab = PluginSettingsTab()
        _KEEP.append(tab)
        tab.load(config, plugins_dir=self.tmp / "plugins")
        panel = tab._panels["search_provider"]
        panel.list.setCurrentRow(2)
        panel.move_current(-1)
        panel.move_current(-1)
        tab.apply_to(config)
        self.assertEqual([(p.category, p.instance_id) for p in config.plugins], [
            ("input_source", "doi_datacite"), ("lookup", "k10plus"),
            ("search_provider", "catalog"), ("search_provider", "lobid"),
            ("search_provider", "swb"), ("not_managed_here", "x")])


# ===========================================================================
# 3. Origin
# ===========================================================================
class _Foreign:
    """A type registered from outside src/ and outside the plugin directory."""


class TestOrigin(_Case):
    def _origins(self, **kw):
        from src.ui.plugin_settings_tab import PluginOrigins

        return PluginOrigins(self.tmp / "plugins", **kw)

    def _search_class(self, pid):
        from src.core.search.registry import get_provider

        return get_provider(pid)

    def test_builtin(self):
        (self.tmp / "plugins").mkdir()
        o = self._origins().origin(_inst("lobid"), registered=True,
                                   type_cls=self._search_class("lobid"))
        self.assertEqual((o.kind, o.tag), ("builtin", "eingebaut"))
        self.assertNotIn("nicht gelesen", o.tooltip)

    def test_declarative_plugin_from_the_directory(self):
        _write_manifest(self.tmp / "plugins", "finc_zbw", _declarative_manifest("finc_zbw", "finc"))
        origins = self._origins()
        o = origins.origin(_inst("finc", iid="finc_zbw"), registered=True,
                           type_cls=self._search_class("finc"))
        self.assertEqual((o.kind, o.tag), ("directory", "Verzeichnis"))
        self.assertIn(str(self.tmp / "plugins" / "finc_zbw" / "plugin.toml"), o.tooltip)
        # The same type under another id is the built-in one.
        other = origins.origin(_inst("finc"), registered=True, type_cls=self._search_class("finc"))
        self.assertEqual(other.tag, "eingebaut")
        # A manifest of another kind does not claim the instance.
        wrong = origins.origin(_inst("lobid", iid="finc_zbw"), registered=True,
                               type_cls=self._search_class("lobid"))
        self.assertEqual(wrong.tag, "eingebaut")

    def test_code_plugin_loaded_through_the_loader(self):
        from src.core.plugins import loader
        from src.core.search.registry import PROVIDER_REGISTRY

        root = self.tmp / "plugins"
        _write_manifest(root, "tabtest_code", _code_manifest("tabtest_code"), {
            "provider.py": ("class P:\n    id = 'tabtest_code'\n    label = 'Testquelle'\n"
                            "    capabilities = set()\n"
                            "    @classmethod\n    def config_fields(cls):\n        return []\n"),
        })
        loader._reset_for_tests()
        self.addCleanup(loader._reset_for_tests)
        self.addCleanup(PROVIDER_REGISTRY.pop, "tabtest_code", None)
        approved = {}
        result = loader.discover(root, approved_plugins=approved,
                                 approve_cb=lambda *a: True, enable_code_plugins=True)
        self.assertEqual(result.plugins[0].status, "loaded", result.plugins[0].detail)
        cls = self._search_class("tabtest_code")
        self.assertTrue(cls.__module__.startswith("alima_plugin_"))
        o = self._origins(approved=approved, code_enabled=True).origin(
            _inst("tabtest_code"), registered=True, type_cls=cls)
        self.assertEqual((o.kind, o.tag), ("code", "Code-Plugin, freigegeben"))
        # In the panel: a normal, switchable row with the tag.
        panel = self.panel([_inst("tabtest_code")])
        panel.set_origins(self._origins(approved=approved, code_enabled=True))
        self.assertTrue(self.texts(panel)[0].endswith("  ·  Code-Plugin, freigegeben"))

    def _unloaded_code(self, **kw):
        _write_manifest(self.tmp / "plugins", "mylib_dk", _code_manifest("mylib_dk"),
                        {"provider.py": "class P:\n    id = 'mylib_dk'\n"})
        return self._origins(**kw).origin(_inst("mylib_dk"), registered=False)

    def test_code_plugin_while_code_plugins_are_off(self):
        o = self._unloaded_code(code_enabled=False)
        self.assertEqual((o.kind, o.tag), ("code", "Code-Plugin, gesperrt"))
        self.assertIn("Code-Plugins sind nicht erlaubt", o.tooltip)

    def test_code_plugin_without_approval(self):
        o = self._unloaded_code(code_enabled=True, approved={})
        self.assertEqual(o.tag, "Code-Plugin, gesperrt")
        self.assertIn("nicht freigegeben", o.tooltip)

    def test_code_plugin_changed_since_the_approval(self):
        o = self._unloaded_code(code_enabled=True, approved={"mylib_dk": "0" * 64})
        self.assertEqual(o.tag, "Code-Plugin, gesperrt")
        self.assertIn("seit der Freigabe geändert", o.tooltip)

    def test_approved_code_plugin_that_is_not_loaded(self):
        from types import SimpleNamespace

        from src.core.plugins.security import hash_dir

        _write_manifest(self.tmp / "plugins", "mylib_dk", _code_manifest("mylib_dk"),
                        {"provider.py": "class P:\n    id = 'mylib_dk'\n"})
        digest = hash_dir(self.tmp / "plugins" / "mylib_dk")
        scan = {"mylib_dk": SimpleNamespace(status="error", detail="ImportError: kaputt")}
        o = self._origins(code_enabled=True, approved={"mylib_dk": digest}, scan=scan).origin(
            _inst("mylib_dk"), registered=False)
        self.assertEqual((o.kind, o.tag), ("code", "Code-Plugin, nicht geladen"))
        self.assertIn("ImportError: kaputt", o.tooltip)

    def test_unknown_type_without_manifest(self):
        (self.tmp / "plugins").mkdir()
        o = self._origins().origin(_inst("weg"), registered=False)
        self.assertEqual((o.kind, o.tag), ("unknown", "unbekannt, nicht geladen"))

    def test_type_registered_from_elsewhere(self):
        o = self._origins().origin(_inst("x"), registered=True, type_cls=_Foreign)
        self.assertEqual((o.kind, o.tag), ("unknown", "unbekannt"))

    def test_directory_not_known(self):
        from src.ui.plugin_settings_tab import PluginOrigins

        o = PluginOrigins(None).origin(_inst("lobid"), registered=True,
                                       type_cls=self._search_class("lobid"))
        self.assertEqual(o.tag, "eingebaut")
        self.assertIn("Das Plugin-Verzeichnis wurde nicht gelesen", o.tooltip)

    def test_the_tab_shows_the_tags_and_follows_the_code_switch(self):
        from types import SimpleNamespace

        from src.ui.plugin_settings_tab import PluginSettingsTab, _CHECK_TIP_ROLE

        root = self.tmp / "plugins"
        _write_manifest(root, "finc_zbw", _declarative_manifest("finc_zbw", "finc"))
        _write_manifest(root, "mylib_dk", _code_manifest("mylib_dk"),
                        {"provider.py": "class P:\n    id = 'mylib_dk'\n"})
        config = SimpleNamespace(
            plugins=[_inst("lobid"), _inst("finc", iid="finc_zbw", base_url="https://f"),
                     _inst("mylib_dk", label="Meine DK")],
            approved_plugins={}, system_config=SimpleNamespace(enable_code_plugins=False))
        tab = PluginSettingsTab()
        _KEEP.append(tab)
        tab.load(config, plugins_dir=root)
        panel = tab._panels["search_provider"]
        texts = self.texts(panel)
        self.assertTrue(texts[0].endswith("  ·  eingebaut"), texts[0])
        self.assertTrue(texts[1].endswith("  ·  Verzeichnis"), texts[1])
        self.assertTrue(texts[2].endswith("  ·  Code-Plugin, gesperrt"), texts[2])
        self.assertIn("nicht erlaubt", panel.list.item(2).data(_CHECK_TIP_ROLE))
        tab._code_plugins_cb.setChecked(True)
        self.assertIn("nicht freigegeben", panel.list.item(2).data(_CHECK_TIP_ROLE))
        # The form of the unloaded instance names the reason too.
        panel.list.setCurrentRow(2)
        from PyQt6.QtWidgets import QLabel

        labels = [w.text() for w in panel.form_box.findChildren(QLabel)]
        self.assertTrue(any("nicht freigegeben" in t for t in labels), labels)


# ===========================================================================
# 4. DK switch, DK tags, line above the list
# ===========================================================================
class TestDkSwitch(_Case):
    def test_legacy_catalog_shows_on_and_writes_nothing(self):
        panel = self.panel([_inst("catalog", token="t"), _inst("sru", preset="dnb")])
        self.assertTrue(panel._field_widgets["dk_enabled"].isChecked())
        panel.list.setCurrentRow(1)
        panel.list.setCurrentRow(0)
        self.assertTrue(panel._field_widgets["dk_enabled"].isChecked())
        for out in panel.collect():
            self.assertNotIn("dk_enabled", out.settings, out.instance_id)

    def test_not_a_source_shows_off(self):
        panel = self.panel([_inst("catalog"), _inst("finc", base_url="https://f", dk_enabled=True)])
        self.assertFalse(panel._field_widgets["dk_enabled"].isChecked())
        self.assertNotIn("dk_enabled", panel.collect()[0].settings)

    def test_follows_another_row_switched_on_in_the_list(self):
        from PyQt6.QtCore import Qt

        panel = self.panel([_inst("catalog", label="Katalog (Libero)"),
                            _inst("sru", label="SRU", preset="dnb", dk_enabled=True,
                                  enabled=False)])
        self.assertTrue(panel._field_widgets["dk_enabled"].isChecked())
        panel.list.item(1).setCheckState(Qt.CheckState.Checked)   # sru: Aus → An
        self.assertEqual(panel._current_row, 0)
        self.assertFalse(panel._field_widgets["dk_enabled"].isChecked())
        self.assertTrue(panel._dk_label.text().startswith("DK-Quelle im Moment: SRU"))
        self.assertNotIn("dk_enabled", panel.collect()[0].settings)

    def test_a_stored_value_is_shown_as_stored(self):
        panel = self.panel([_inst("catalog", dk_enabled=False), _inst("finc", dk_enabled=False)])
        self.assertFalse(panel._field_widgets["dk_enabled"].isChecked())
        panel.list.setCurrentRow(1)
        self.assertFalse(panel._field_widgets["dk_enabled"].isChecked())
        self.assertEqual([o.settings["dk_enabled"] for o in panel.collect()], [False, False])

    def test_unsaved_switch_moves_the_tags_and_the_line(self):
        panel = self.panel([_inst("catalog", label="Katalog (Libero)"),
                            _inst("finc", label="finc (VuFind)", base_url="https://f")])
        self.assertIn("  ·  DK 1  ·  ", self.texts(panel)[0])
        panel.list.setCurrentRow(1)
        panel._field_widgets["dk_enabled"].setChecked(True)       # not saved yet
        texts = self.texts(panel)
        self.assertNotIn("DK ", texts[0])
        self.assertIn("  ·  DK 1  ·  ", texts[1])
        self.assertEqual(panel._dk_label.text(), "DK-Quelle im Moment: finc (VuFind) [finc]")

    def test_duplicate_and_remove_update_tags_and_line(self):
        panel = self.panel([_inst("finc", label="finc", base_url="https://f", dk_enabled=True)])
        self.assertEqual(panel._dk_label.text(), "DK-Quelle im Moment: finc [finc]")
        panel._duplicate_instance()
        self.assertEqual(panel._dk_label.text(),
                         "DK-Quellen im Moment: finc [finc]; dann finc (Kopie) [finc-2] "
                         "(Rückfall je Schlagwort)")
        self.assertIn("  ·  DK 2  ·  ", self.texts(panel)[1])
        panel._remove_instance()                                   # the copy is selected
        self.assertEqual(self.ids(panel), ["finc"])
        self.assertEqual(panel._dk_label.text(), "DK-Quelle im Moment: finc [finc]")

    def test_tag_tooltips(self):
        panel = self.panel([_inst("catalog", dk_enabled=True),
                            _inst("finc", base_url="https://f", dk_enabled=True)])
        self.assertIn("DK 2: Stelle 2 von 2 der DK-Kette", panel.list.item(1).toolTip())
        solo = self.panel([_inst("catalog")])
        self.assertIn("DK 1: die einzige DK-Quelle im Moment", solo.list.item(0).toolTip())


# ===========================================================================
# 5. Round trip with a config shaped like the operator's
# ===========================================================================
def _operator_plugins():
    def p(category, iid, label, enabled=True, **settings):
        return {"instance_id": iid, "category": category, "provider_id": iid, "label": label,
                "enabled": enabled, "usage_hint": "", "settings": settings}

    return [
        p("input_source", "doi_crossref", "DOI: Crossref", contact_email="a@b.example",
          timeout=30, cache_responses="auto"),
        p("input_source", "doi_openalex", "DOI: OpenAlex", contact_email="a@b.example"),
        p("input_source", "doi_datacite", "DOI: DataCite"),
        p("input_source", "url_fetch", "URL (Web-Scrape)"),
        p("lookup", "dnb", "DNB-Klassifikation (GND-RDF)", timeout=10, cache_responses="auto"),
        p("lookup", "k10plus", "K10plus (Paketsigel)"),
        p("lookup", "local_scheme", "Lokale Aufstellungssystematik"),
        p("lookup", "rvk_api", "RVK-API (Regensburg)"),
        p("lookup", "webindex", "Webseiten-Index (RAG)"),
        p("lookup", "geo_sachdb", "Lokale Sacherschließungsdatenbank (SQLite)",
          db_path="/nicht/da.db", max_results=10, cache_responses="auto"),
        p("search_provider", "lobid", "Lobid (GND/DNB)", page_size=30, cache_responses="auto"),
        p("search_provider", "swb", "SWB (BSZ)"),
        p("search_provider", "catalog", "Katalog (Libero)",
          catalog_search_url="https://libero.example/soap", token="T",
          strict_gnd_validation_for_dk_search=True, cache_only=True, cache_responses="auto"),
        p("search_provider", "finc", "finc (VuFind)", base_url="", web_record_url="",
          catalog_web_record_url="", institution_filter="", default_limit=20, timeout=30,
          dk_enabled=False, harvest_enabled=False, cache_only=True, cache_responses="auto"),
        p("search_provider", "sru", "SRU / MARC-XML"),
        p("search_provider", "gnd_local", "Lokale GND-DB"),
        p("search_provider", "kvk", "KVK (Verbundkataloge)", enabled=False),
        p("search_provider", "mylib_dk", "Meine DK", base="x"),
    ]


class TestRoundTrip(_Case):
    def _visit_every_row(self, tab):
        for panel in tab._panels.values():
            for row in range(panel.list.count()):
                panel.list.setCurrentRow(row)
            panel.list.setCurrentRow(0)

    def _plugins_dir(self):
        root = self.tmp / ".config" / "alima" / "plugins"
        _write_manifest(root, "mylib_dk", _code_manifest("mylib_dk"),
                        {"provider.py": "class P:\n    id = 'mylib_dk'\n"})
        return root

    def test_tab_round_trip_on_the_config_object(self):
        from src.ui.plugin_settings_tab import PluginSettingsTab
        from src.utils.config_models import AlimaConfig

        config = AlimaConfig()
        config.plugins = [PluginInstanceConfig(**d) for d in _operator_plugins()]
        config.system_config.enable_code_plugins = True
        config.approved_plugins = {"geo_sachdb": "f" * 64}
        before = [asdict(p) for p in config.plugins]
        tab = PluginSettingsTab()
        _KEEP.append(tab)
        tab.load(config, plugins_dir=self._plugins_dir())
        self._visit_every_row(tab)
        with patch("src.ui.plugin_settings_tab.QMessageBox.warning"):
            tab.apply_to(config)
        self.assertEqual([asdict(p) for p in config.plugins], before)
        self.assertTrue(config.system_config.enable_code_plugins)

    def test_config_json_is_unchanged_by_the_tab(self):
        from src.ui.plugin_settings_tab import PluginSettingsTab
        from src.utils.config_manager import ConfigManager

        path = self.tmp / ".config" / "alima" / "config.json"
        self._plugins_dir()
        path.write_text(json.dumps({
            "unified_config": {"providers": [], "gemini_api_key": "K"},
            "system_config": {"enable_code_plugins": True},
            "approved_plugins": {"geo_sachdb": "f" * 64},
            "plugins": _operator_plugins(),
        }), encoding="utf-8")

        def load_and_save(with_tab):
            ConfigManager.reset()
            try:
                cm = ConfigManager()
                cm.config_file = path
                config = cm.load_config(force_reload=True)
                if with_tab:
                    tab = PluginSettingsTab()
                    _KEEP.append(tab)
                    tab.load(config, plugins_dir=cm.plugins_dir)
                    self._visit_every_row(tab)
                    with patch("src.ui.plugin_settings_tab.QMessageBox.warning"):
                        tab.apply_to(config)
                cm.save_config(config)
                return json.loads(path.read_text(encoding="utf-8"))
            finally:
                ConfigManager.reset()

        plain = load_and_save(False)
        with_tab = load_and_save(True)
        self.assertEqual(with_tab["plugins"], plain["plugins"])
        self.assertEqual(with_tab.get("approved_plugins"), plain.get("approved_plugins"))
        self.assertEqual(with_tab["system_config"].get("enable_code_plugins"),
                         plain["system_config"].get("enable_code_plugins"))
        by_id = {p["instance_id"]: p for p in with_tab["plugins"]}
        self.assertNotIn("dk_enabled", by_id["catalog"]["settings"])
        self.assertIs(by_id["catalog"]["settings"]["cache_only"], True)
        self.assertEqual(by_id["finc"]["settings"]["base_url"], "")
        self.assertEqual(by_id["mylib_dk"]["settings"], {"base": "x"})


if __name__ == "__main__":
    unittest.main()
