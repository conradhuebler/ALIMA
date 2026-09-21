"""The search field is what makes folded sections acceptable.

A section that folds away is only defensible if the search reaches into it.
Sections are therefore named after their content ("Modellgrenzen"), never
"Erweitert", and every control registers itself with page and section.

The German↔English synonyms are load-bearing, not decoration: the interface is
German while the documentation of every service it talks to is English. Someone
who reads "set your API key" must find "API-Schlüssel". - Claude Generated
"""

from __future__ import annotations

import unittest

from src.ui.settings.search import SettingsIndex


def _index() -> SettingsIndex:
    ix = SettingsIndex()
    ix.add("Sprachmodelle", "API-Schlüssel", section="GWDG")
    ix.add("Sprachmodelle", "Basis-URL", section="GWDG")
    ix.add("Sprachmodelle", "Stückeln ab", section="Modellgrenzen")
    ix.add("Erschließung", "Standardablauf")
    ix.add("Datenquellen", "Antworten zwischenspeichern", section="lobid GND")
    ix.add("Daten und Speicher", "Zeitlimit", section="Feinheiten der Verbindung")
    ix.add("Daten und Speicher", "Datei auf diesem Rechner", section="Wissensdatenbank")
    ix.add("Anwendung", "Protokoll")
    ix.add("Anwendung", "Schriftgröße")
    return ix


class EnglishTermsFindGermanLabelsTest(unittest.TestCase):
    """Each of these is a term the service documentation uses."""

    def setUp(self):
        self.ix = _index()

    def _labels(self, needle):
        return [e.label for e in self.ix.search(needle)]

    def test_api_key(self):
        self.assertIn("API-Schlüssel", self._labels("api key"))

    def test_token(self):
        self.assertIn("API-Schlüssel", self._labels("token"))

    def test_timeout(self):
        self.assertIn("Zeitlimit", self._labels("timeout"))

    def test_cache(self):
        self.assertIn("Antworten zwischenspeichern", self._labels("cache"))

    def test_sqlite(self):
        self.assertIn("Datei auf diesem Rechner", self._labels("sqlite"))

    def test_debug_finds_the_log_level(self):
        """The support path: "stell auf Debug und schick mir die Logdatei"."""
        self.assertIn("Protokoll", self._labels("debug"))

    def test_font_size(self):
        self.assertIn("Schriftgröße", self._labels("font size"))


class TypingWithoutUmlautsWorksTest(unittest.TestCase):
    def test_schluessel_without_umlaut(self):
        self.assertIn("API-Schlüssel", [e.label for e in _index().search("schlussel")])

    def test_schriftgroesse_without_umlaut(self):
        self.assertIn("Schriftgröße", [e.label for e in _index().search("schriftgrosse")])

    def test_case_is_ignored(self):
        self.assertTrue(_index().search("ZEITLIMIT"))


class TheSearchReachesIntoFoldedSectionsTest(unittest.TestCase):
    """The whole point: a hit carries the path and a way to unfold it."""

    def test_a_hit_names_its_section(self):
        hit = _index().search("stückeln")[0]
        self.assertEqual(hit.path(), "Sprachmodelle › Modellgrenzen")

    def test_a_page_level_hit_has_no_section_in_its_path(self):
        hit = _index().search("standardablauf")[0]
        self.assertEqual(hit.path(), "Erschließung")

    def test_reveal_is_called_to_unfold(self):
        opened = []
        ix = SettingsIndex()
        ix.add("Sprachmodelle", "Stückeln ab", section="Modellgrenzen",
               reveal=lambda: opened.append(True))
        ix.search("stückeln")[0].reveal()
        self.assertEqual(opened, [True])


class RankingAndBoundsTest(unittest.TestCase):
    def test_a_label_match_beats_a_section_match(self):
        ix = SettingsIndex()
        ix.add("A", "Etwas anderes", section="Protokoll")
        ix.add("B", "Protokoll")
        self.assertEqual(ix.search("protokoll")[0].label, "Protokoll")

    def test_an_empty_needle_finds_nothing(self):
        self.assertEqual(_index().search(""), [])
        self.assertEqual(_index().search("   "), [])

    def test_an_unknown_term_finds_nothing(self):
        self.assertEqual(_index().search("quantenfeldtheorie"), [])

    def test_results_are_capped(self):
        ix = SettingsIndex()
        for i in range(50):
            ix.add("A", f"Protokoll {i}")
        self.assertEqual(len(ix.search("protokoll", limit=5)), 5)


if __name__ == "__main__":
    unittest.main()
