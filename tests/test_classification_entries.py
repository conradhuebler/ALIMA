"""Canonical classification ENTRY semantics - Claude Generated (WP-D2, P0 revision).

WP-D1 P0 pinned ``classifications: {system: [codes]}``, implicitly treating a
classification as a FACT ("this concept has DDC X"). Harvesting lobid showed
that most classifications are not facts but weighted evidence ("RVK WI 4700
appeared in 13 catalogue records about this term") — and that the same field
would otherwise carry both, indistinguishably: an authority DDC from the GND
record next to a statistical co-occurrence.

So an entry is ``{code, count?, origin}``. These tests pin the two rules that
carry risk: the count merge (max, never sum — the 038738e landmine) and the
origin precedence.
"""

from __future__ import annotations

import unittest

from src.utils.classification_systems import (
    ORIGIN_AUTHORITY,
    ORIGIN_COOCCURRENCE,
    classification_entry,
    merge_classification_entries,
    merge_classifications,
    normalize_classifications,
)


class TestEntryConstruction(unittest.TestCase):
    def test_authority_entry_has_no_count(self):
        """An authority statement has no frequency; inventing 1 would be evidence."""
        entry = classification_entry("551.48", origin=ORIGIN_AUTHORITY)
        self.assertEqual(entry, {"code": "551.48", "origin": ORIGIN_AUTHORITY})
        self.assertNotIn("count", entry)

    def test_cooccurrence_entry_carries_its_count(self):
        entry = classification_entry("WI 4700", count=13)
        self.assertEqual(
            entry, {"code": "WI 4700", "count": 13, "origin": ORIGIN_COOCCURRENCE}
        )


class TestMergeRules(unittest.TestCase):
    def test_counts_are_maxed_never_summed(self):
        """Summing would inflate evidence when two sources saw the same records."""
        merged = merge_classification_entries(
            [classification_entry("WI 4700", count=13)],
            [classification_entry("WI 4700", count=6)],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["count"], 13)

    def test_authority_outranks_cooccurrence_for_the_same_code(self):
        merged = merge_classification_entries(
            [classification_entry("551.48", count=4)],
            [classification_entry("551.48", origin=ORIGIN_AUTHORITY)],
        )
        self.assertEqual(merged[0]["origin"], ORIGIN_AUTHORITY)
        # The evidence it also happens to have is kept, not discarded.
        self.assertEqual(merged[0]["count"], 4)

    def test_result_is_ordered_authority_then_strength(self):
        merged = merge_classification_entries(
            [
                classification_entry("AR 22480", count=1),
                classification_entry("WI 4700", count=13),
            ],
            [classification_entry("QQ 000", origin=ORIGIN_AUTHORITY)],
        )
        self.assertEqual([e["code"] for e in merged], ["QQ 000", "WI 4700", "AR 22480"])

    def test_merge_is_order_independent(self):
        a = [classification_entry("A", count=3)]
        b = [classification_entry("B", count=9)]
        self.assertEqual(
            merge_classification_entries(a, b), merge_classification_entries(b, a)
        )

    def test_systems_merge_independently_and_stay_equal_rank(self):
        merged = merge_classifications(
            {"RVK": [classification_entry("WI 4700", count=2)]},
            {
                "RVK": [classification_entry("WI 4700", count=8)],
                "DDC": [classification_entry("551.48", origin=ORIGIN_AUTHORITY)],
            },
        )
        self.assertEqual(merged["RVK"][0]["count"], 8)
        self.assertEqual(merged["DDC"][0]["origin"], ORIGIN_AUTHORITY)

    def test_merge_does_not_mutate_its_inputs(self):
        """Pool inserts are shallow copies — an in-place merge leaks across entries."""
        left = {"RVK": [classification_entry("A", count=1)]}
        snapshot = {"RVK": [dict(left["RVK"][0])]}
        merge_classifications(left, {"RVK": [classification_entry("B", count=5)]})
        self.assertEqual(left, snapshot)

    def test_empty_systems_are_dropped(self):
        self.assertEqual(merge_classifications({"RVK": []}, {"DDC": []}), {})


class TestNormalization(unittest.TestCase):
    def test_bare_code_containers_become_entries(self):
        """Producers with no evidence to report still fit the shape."""
        out = normalize_classifications({"ddc": {"551.48"}}, origin=ORIGIN_AUTHORITY)
        self.assertEqual(out, {"DDC": [{"code": "551.48", "origin": ORIGIN_AUTHORITY}]})

    def test_system_spelling_cannot_split_one_system(self):
        out = normalize_classifications({"rvk": ["A"], "RVK": ["B"]})
        self.assertEqual(set(out), {"RVK"})
        self.assertEqual([e["code"] for e in out["RVK"]], ["A", "B"])

    def test_unknown_system_is_dropped_not_guessed(self):
        self.assertEqual(normalize_classifications({"LBZ-Notationen": ["720"]}), {})

    def test_entry_shape_passes_through(self):
        out = normalize_classifications(
            {"RVK": [{"code": "WI 4700", "count": 13, "origin": ORIGIN_COOCCURRENCE}]}
        )
        self.assertEqual(out["RVK"][0]["count"], 13)

    def test_blank_codes_are_dropped(self):
        self.assertEqual(normalize_classifications({"DDC": ["", None, "  "]}), {})


if __name__ == "__main__":
    unittest.main()
