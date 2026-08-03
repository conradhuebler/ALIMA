"""Tests for SearchTab's pure helpers (WP-K5 rework) - Claude Generated.

The display defects these pin: the table used to show the pool ``count``
(ranking placeholder, 1 on cache hits — the F-4 "Häufigkeit zeigt 1" bug never
fixed in this tab) and no classifications at all.
"""

from __future__ import annotations

import unittest

from src.ui.find_keywords import (
    determine_relation,
    extract_search_terms,
    format_classifications_compact,
    preferred_display_count,
)


class TestExtractSearchTerms(unittest.TestCase):
    def test_commas_and_quoted_phrases(self):
        self.assertEqual(
            extract_search_terms('Boden, "saure Böden", Chemie'),
            ["saure Böden", "Boden", "Chemie"],
        )

    def test_empty_and_whitespace(self):
        self.assertEqual(extract_search_terms(""), [])
        self.assertEqual(extract_search_terms("  ,  , "), [])


class TestDetermineRelation(unittest.TestCase):
    def test_exact_similar_different(self):
        self.assertEqual(determine_relation("Boden", "boden"), 0)
        self.assertEqual(determine_relation("Bodenkunde", "Boden"), 1)
        self.assertEqual(determine_relation("Chemie", "Boden"), 2)


class TestPreferredDisplayCount(unittest.TestCase):
    def test_display_count_wins_over_pool_count(self):
        """The count landmine: pool count stays 1 on cache hits; the REAL
        Häufigkeit is display_count."""
        self.assertEqual(preferred_display_count({"count": 1, "display_count": 17}), 17)

    def test_falls_back_to_count(self):
        self.assertEqual(preferred_display_count({"count": 4}), 4)

    def test_garbage_is_zero(self):
        self.assertEqual(preferred_display_count({}), 0)
        self.assertEqual(preferred_display_count({"count": "x"}), 0)


class TestFormatClassificationsCompact(unittest.TestCase):
    _CLS = {
        "DDC": [{"code": "551.48", "origin": "authority"}],
        "RVK": [
            {"code": "WI 5000", "origin": "cooccurrence", "count": 3},
            {"code": "WI 4800", "origin": "cooccurrence", "count": 2},
            {"code": "WI 4700", "origin": "cooccurrence", "count": 1},
        ],
    }

    def test_compact_line_with_cap(self):
        self.assertEqual(
            format_classifications_compact(self._CLS, max_per_system=2),
            "DDC 551.48 · RVK WI 5000, WI 4800 (+1)",
        )

    def test_empty_and_none(self):
        self.assertEqual(format_classifications_compact({}), "")
        self.assertEqual(format_classifications_compact(None), "")

    def test_bare_codes_are_tolerated(self):
        """Plugins may emit bare code strings — the shared normaliser applies."""
        self.assertEqual(
            format_classifications_compact({"DK": ["546.43"]}), "DK 546.43"
        )


if __name__ == "__main__":
    unittest.main()
