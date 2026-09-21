"""The dialog frame: one navigation level, and search that reaches into folds.

The old dialog had tabs inside tabs — depth three. A sidebar does not invite a
second row the way a tab bar does when it runs out of space. Scrolling inside a
page replaces nesting. - Claude Generated
"""

from __future__ import annotations

import unittest


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class ShellTest(unittest.TestCase):
    _kept: list = []

    def _shell(self):
        from PyQt6.QtWidgets import QLabel, QWidget

        from src.ui.settings import SettingsIndex, SettingsShell

        _qapp(self._kept)
        index = SettingsIndex()
        shell = SettingsShell(index)
        for title in ("Sprachmodelle", "Erschließung", "Datenquellen",
                      "Daten und Speicher", "Anwendung"):
            page = QWidget()
            page.setLayout(__import__("PyQt6.QtWidgets", fromlist=["QVBoxLayout"]).QVBoxLayout())
            page.layout().addWidget(QLabel(title))
            shell.add_page(title, page)
        self._kept.append(shell)
        return shell, index

    def test_there_is_exactly_one_navigation_level(self):
        """No QTabWidget anywhere in the frame."""
        from PyQt6.QtWidgets import QTabWidget

        shell, _ = self._shell()
        self.assertEqual(shell.findChildren(QTabWidget), [])

    def test_all_five_pages_are_reachable(self):
        shell, _ = self._shell()
        self.assertEqual(shell.nav.count(), 5)
        self.assertEqual(shell.stack.count(), 5)

    def test_selecting_a_page_switches_the_stack(self):
        shell, _ = self._shell()
        shell.show_page("Anwendung")
        self.assertEqual(shell.stack.currentIndex(), 4)

    def test_a_badge_is_shown(self):
        shell, _ = self._shell()
        shell.set_page_badge("Datenquellen", "12/16")
        self.assertIn("12/16", shell.nav.item(2).text())

    def test_rebadging_replaces_instead_of_appending(self):
        """The badge has to be found on an item that already carries one.

        Naming the page by the item's full text would work once and then stop:
        the second call looks for "Datenquellen" and finds "Datenquellen  12/16".
        """
        shell, _ = self._shell()
        shell.set_page_badge("Datenquellen", "12/16")
        shell.set_page_badge("Datenquellen", "9/16")
        text = shell.nav.item(2).text()
        self.assertIn("9/16", text)
        self.assertNotIn("12/16", text)

    def test_a_badge_can_be_cleared(self):
        shell, _ = self._shell()
        shell.set_page_badge("Datenquellen", "12/16")
        shell.set_page_badge("Datenquellen", "")
        self.assertEqual(shell.nav.item(2).text(), "Datenquellen")

    def test_page_lookup_is_independent_of_the_badge(self):
        shell, _ = self._shell()
        shell.set_page_badge("Datenquellen", "12/16")
        shell.show_page("Datenquellen")
        self.assertEqual(shell.stack.currentIndex(), 2)


class SearchDrivesTheShellTest(unittest.TestCase):
    _kept: list = []

    def _shell(self):
        from PyQt6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

        from src.ui.settings import CollapsibleSection, SettingsIndex, SettingsShell

        _qapp(self._kept)
        index = SettingsIndex()
        shell = SettingsShell(index)

        page = QWidget()
        layout = QVBoxLayout(page)
        section = CollapsibleSection("Feinheiten der Verbindung")
        body = QVBoxLayout(section.body())
        field = QLineEdit()
        body.addWidget(field)
        layout.addWidget(section)
        shell.add_page("Daten und Speicher", page)
        shell.add_page("Anwendung", QWidget())

        index.add("Daten und Speicher", "Zeitlimit",
                  section="Feinheiten der Verbindung",
                  widget=field, reveal=section.expand)
        self._kept.extend([shell, section])
        return shell, section

    def test_typing_replaces_the_nav_with_results(self):
        shell, _ = self._shell()
        shell.search_field.setText("timeout")
        self.assertTrue(shell.results.isVisible() or shell.results.count() > 0)
        self.assertEqual(shell.results.count(), 1)

    def test_clearing_brings_the_nav_back(self):
        shell, _ = self._shell()
        shell.search_field.setText("timeout")
        shell.search_field.setText("")
        self.assertEqual(shell.results.count(), 0)

    def test_a_hit_opens_its_page_and_unfolds_its_section(self):
        """The property that makes folding acceptable."""
        shell, section = self._shell()
        self.assertFalse(section.is_expanded())
        shell.search_field.setText("timeout")
        shell._on_result(shell.results.item(0))
        self.assertEqual(shell.stack.currentIndex(), 0)
        self.assertTrue(section.is_expanded())

    def test_an_unknown_term_says_so_instead_of_showing_nothing(self):
        shell, _ = self._shell()
        shell.search_field.setText("quantenfeldtheorie")
        self.assertEqual(shell.results.count(), 1)
        self.assertIn("keine Treffer", shell.results.item(0).text())


class CollapsibleSectionTest(unittest.TestCase):
    _kept: list = []

    def test_it_starts_folded_and_keeps_its_title_visible(self):
        from src.ui.settings import CollapsibleSection

        _qapp(self._kept)
        s = CollapsibleSection("Modellgrenzen", subtitle="1 Abweichung")
        self._kept.append(s)
        self.assertFalse(s.is_expanded())
        self.assertIn("Modellgrenzen", s._button.text())
        self.assertIn("1 Abweichung", s._button.text())

    def test_expand_is_idempotent(self):
        from src.ui.settings import CollapsibleSection

        _qapp(self._kept)
        s = CollapsibleSection("Modellgrenzen")
        self._kept.append(s)
        s.expand()
        s.expand()
        self.assertTrue(s.is_expanded())


if __name__ == "__main__":
    unittest.main()
