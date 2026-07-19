"""UI-chrome i18n catalog (src/utils/i18n.py) - Claude Generated.

Covers the fallback chain (active → de → key literal), placeholder
formatting, the browser-facing ``js.*`` subset, de/en catalog parity, and
tolerance for unknown languages (never raises).
"""

import unittest

from src.utils import i18n
from src.utils.i18n import catalog_for_js, get_language, set_language, t


class I18nTest(unittest.TestCase):
    def setUp(self):
        i18n._reset_for_tests()

    def tearDown(self):
        i18n._reset_for_tests()

    def test_default_language_is_german(self):
        self.assertEqual(get_language(), "de")
        self.assertEqual(t("render.summary.error"), "Fehler")

    def test_english_catalog(self):
        set_language("en")
        self.assertEqual(t("render.summary.error"), "Error")
        self.assertEqual(t("render.summary.hits", n=12), "12 hits")

    def test_placeholder_formatting(self):
        self.assertEqual(t("render.summary.chars", n=42), "42 Zeichen")

    def test_missing_placeholder_does_not_raise(self):
        # Wrong/missing kwargs degrade to the raw template, never crash.
        self.assertEqual(t("render.summary.chars"), "{n} Zeichen")
        self.assertEqual(t("render.summary.chars", wrong=1), "{n} Zeichen")

    def test_unknown_key_returns_key_literal(self):
        self.assertEqual(t("no.such.key"), "no.such.key")

    def test_unknown_language_falls_back_to_default(self):
        set_language("xx")
        self.assertEqual(t("render.summary.error"), "Fehler")

    def test_catalog_for_js_only_js_keys(self):
        catalog = catalog_for_js()
        self.assertTrue(catalog)
        self.assertTrue(all(k.startswith("js.") for k in catalog))
        self.assertIn("js.typing.suffix", catalog)

    def test_catalog_for_js_active_language_wins(self):
        set_language("en")
        self.assertEqual(catalog_for_js()["js.typing.suffix"], "is typing …")

    def test_de_en_catalog_parity(self):
        de = i18n._catalog("de")
        en = i18n._catalog("en")
        self.assertEqual(set(de), set(en), "locales/de.json and en.json diverged")


if __name__ == "__main__":
    unittest.main()
