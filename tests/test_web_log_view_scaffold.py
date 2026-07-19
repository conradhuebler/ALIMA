"""WebLogView scaffold contract - Claude Generated.

Guards the GUI-only flags injected into the QWebEngine document: the
same-window-links flag (without it, the shared _ensureLinksNewTab stamps
target="_blank" and GUI link clicks die in an unfulfilled Chromium popup
request — July 19 regression) and the i18n catalog.
"""

import unittest

try:
    from src.ui.web_log_view import _build_scaffold_html
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - headless import guard
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class WebLogViewScaffoldTest(unittest.TestCase):
    def test_scaffold_sets_same_window_links_flag(self):
        html = _build_scaffold_html(10)
        flag = "window.__alimaSameWindowLinks = true;"
        self.assertIn(flag, html)
        # The flag must be set BEFORE the shared JS is evaluated.
        self.assertLess(html.index(flag), html.index("_ensureLinksNewTab"))

    def test_scaffold_injects_i18n_catalog(self):
        html = _build_scaffold_html(10)
        self.assertIn("window.__alimaI18n = {", html)


if __name__ == "__main__":
    unittest.main()
