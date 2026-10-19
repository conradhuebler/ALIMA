"""Selecting a row in the provider list rendered white text on a gray
background instead of white-on-blue.

Root cause: ``QTreeWidget::item:selected`` sets a blue background + white
text, and a later ``QTreeWidget::item:hover`` rule set a light-gray
background with no matching text color. Clicking an item leaves the cursor
hovering over the very row it just selected, so both pseudo-states are true
at once - and in Qt Style Sheets, when two rules for the same target
conflict, the one declared LATER in the sheet wins per property. The hover
rule, declared after :selected, overrode just the background-color, leaving
:selected's white text with no matching background (confirmed by actually
rendering the widget and sampling the pixel under QT_QPA_PLATFORM=offscreen -
the default platform plugin picked outside that env var did not reproduce
the hover pseudo-state reliably enough to make a pixel-based test robust
across environments, hence the string-level check below instead). Fixed by
excluding :selected from the hover rule (``:hover:!selected``).
- Claude Generated
"""

from __future__ import annotations

import unittest

from src.ui.styles import get_main_stylesheet


class TreeWidgetHoverExcludesSelectedTest(unittest.TestCase):
    def test_hover_rule_excludes_selected(self):
        sheet = get_main_stylesheet()
        self.assertIn("QTreeWidget::item:hover:!selected", sheet)
        self.assertNotIn("QTreeWidget::item:hover {", sheet)


if __name__ == "__main__":
    unittest.main()
