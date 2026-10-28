"""Die senkrechten Schritt-Tabs der Pipeline sind so breit wie ihre Schrift.

Eine Mindestbreite (``min-width: 100px``) in der Tab-Regel machte die Leiste
136 statt 52 px breit. Am früheren Stand wirkte sie nie, weil eine ungültige
Zeile in derselben Regel den Qt-Parser aus dem Tritt brachte; mit sauberem
Stylesheet griff sie plötzlich. Claude Generated.
"""
from __future__ import annotations

import unittest

_kept: list = []


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


def _bar_width(stylesheet: str) -> int:
    from PyQt6.QtWidgets import QLabel, QTabWidget

    app = _qapp()
    tw = QTabWidget()
    tw.setTabPosition(QTabWidget.TabPosition.West)
    for title in ("Input_Datenquellen", "Schlagwort-Extraktion", "GND-Recherche"):
        tw.addTab(QLabel(title), title)
    tw.setStyleSheet(stylesheet)
    tw.resize(500, 400)
    tw.show()
    app.processEvents()
    width = tw.tabBar().tabRect(0).width()
    _kept.append(tw)
    return width


class PipelineTabsStayNarrowTest(unittest.TestCase):
    def _css(self) -> str:
        from src.ui.pipeline_tab import PipelineTab

        return PipelineTab._pipeline_tabs_css(None)

    def test_the_tab_rule_sets_no_minimum_width(self):
        self.assertNotIn("min-width", self._css())

    def test_the_bar_is_not_wider_than_the_same_tabs_without_the_rule(self):
        reference = _bar_width("")
        styled = _bar_width(self._css())
        self.assertLessEqual(styled, reference + 6, (styled, reference))

    def test_a_minimum_width_would_be_caught(self):
        """Der Test misst tatsächlich: mit der alten Regel wird die Leiste breit."""
        wide = _bar_width(self._css().replace(
            "QTabBar::tab {", "QTabBar::tab { min-width: 100px;", 1))
        self.assertGreater(wide, _bar_width("") + 30)

    def test_the_stylesheet_has_no_stray_non_declarations(self):
        """Zeilen wie '#padding: …' verwarfen früher die ganze Regel stillschweigend."""
        for line in self._css().splitlines():
            self.assertFalse(line.strip().startswith("#"), line)


if __name__ == "__main__":
    unittest.main()
