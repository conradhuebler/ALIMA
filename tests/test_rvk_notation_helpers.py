"""The RVK notation helpers, now reachable - Claude Generated (WP cleanup C).

These five lived nested inside method bodies, where no test could reach them —
``_is_parent_like`` decides whether one notation is treated as an ancestor of
another and therefore dropped from the shortlist, and ``_source_rank`` /
``_status_rank`` existed TWICE, byte-identical, in two different methods. They
were moved to module level verbatim (opcode sequences compared against the
pre-move versions) purely so this file can exist.

Characterisation, not specification: what the code does today, pinned.
"""

from __future__ import annotations

import unittest

from src.utils._pipeline_rvk_scoring import (
    _branch_key,
    _compact_rvk,
    _is_parent_like,
    _source_rank,
    _status_rank,
)


class TestCompactRvk(unittest.TestCase):
    def test_spaces_are_removed_and_case_normalised(self):
        self.assertEqual(_compact_rvk("wi 4700"), "WI4700")

    def test_dots_survive(self):
        """Dots carry hierarchy depth in RVK, so they must not be stripped."""
        self.assertEqual(_compact_rvk("WI 4700.5"), "WI4700.5")

    def test_punctuation_is_dropped(self):
        self.assertEqual(_compact_rvk("WI-4700/2"), "WI47002")

    def test_empty_and_none_are_empty(self):
        for value in ("", None):
            with self.subTest(value=value):
                self.assertEqual(_compact_rvk(value), "")


class TestBranchKey(unittest.TestCase):
    """The branch is used to cap how many codes one RVK area may contribute."""

    def test_leading_letters_are_the_branch(self):
        self.assertEqual(_branch_key("WI 4700"), "WI")
        self.assertEqual(_branch_key("QC 130"), "QC")

    def test_single_letter_branch(self):
        self.assertEqual(_branch_key("A 1"), "A")

    def test_code_without_letters_falls_back_to_the_first_token(self):
        self.assertEqual(_branch_key("12345"), "12345")

    def test_lowercase_is_not_recognised_as_a_branch(self):
        """CHARACTERISATION: the pattern is ``^([A-Z]{1,3})``, case-SENSITIVE.

        A lowercase notation therefore falls into the token fallback and its
        whole string becomes the branch key, so per-branch capping stops working
        for it. Callers normalise upstream today; this is a latent sharp edge.
        """
        self.assertEqual(_branch_key("wi 4700"), "wi")


class TestIsParentLike(unittest.TestCase):
    """Decides whether a broader notation is superseded by a narrower one."""

    def test_clear_parent_child(self):
        self.assertTrue(_is_parent_like("WI 4700", "WI 470012"))

    def test_dotted_child_counts(self):
        self.assertTrue(_is_parent_like("WI 4700", "WI 4700.5"))

    def test_identical_codes_are_not_parent_child(self):
        self.assertFalse(_is_parent_like("WI 4700", "WI 4700"))

    def test_different_branches_are_unrelated(self):
        self.assertFalse(_is_parent_like("WI 4700", "WN 4700"))

    def test_a_single_extra_character_is_not_enough(self):
        """CHARACTERISATION: the rule is ``len(child) > len(parent) + 1``.

        "WI 47001" is NOT treated as a child of "WI 4700" — one extra character
        does not qualify. Whether that matches RVK's actual hierarchy is a
        domain question; pinned here so a change to the threshold is deliberate.
        """
        self.assertFalse(_is_parent_like("WI 4700", "WI 47001"))
        self.assertTrue(_is_parent_like("WI 4700", "WI 470012"))

    def test_comparison_ignores_spacing_and_case(self):
        self.assertTrue(_is_parent_like("wi 4700", "WI470012"))

    def test_empty_operands_are_never_parent_like(self):
        self.assertFalse(_is_parent_like("", "WI 4700"))
        self.assertFalse(_is_parent_like("WI 4700", ""))

    def test_relation_is_directional(self):
        self.assertTrue(_is_parent_like("WI 4700", "WI 470012"))
        self.assertFalse(_is_parent_like("WI 470012", "WI 4700"))


class TestRankHelpers(unittest.TestCase):
    """One definition each now — they used to be duplicated across two methods."""

    def test_source_precedence(self):
        self.assertEqual(
            [_source_rank(s) for s in ("input_record", "rvk_gnd_index", "rvk_api", "catalog")],
            [4, 3, 2, 1],
        )

    def test_input_record_outranks_every_derived_source(self):
        """WP-D1 P2: the record's own classification is the catalog's statement
        about THIS document — no derived source may beat it."""
        for derived in ("rvk_gnd_index", "rvk_api", "catalog"):
            self.assertGreater(_source_rank("input_record"), _source_rank(derived))

    def test_unknown_source_ranks_last(self):
        self.assertEqual(_source_rank("etwas anderes"), 0)

    def test_status_precedence(self):
        self.assertEqual(
            [_status_rank(s) for s in ("standard", "non_standard", "validation_error")],
            [3, 2, 1],
        )

    def test_unknown_status_ranks_last(self):
        self.assertEqual(_status_rank(""), 0)


class TestNoDuplicateDefinitionsRemain(unittest.TestCase):
    def test_helpers_are_defined_exactly_once(self):
        """Guards the dedup: a re-nested copy would drift from this one."""
        import ast
        import pathlib

        tree = ast.parse(
            pathlib.Path("src/utils/_pipeline_rvk_scoring.py").read_text()
        )
        counts = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                counts[node.name] = counts.get(node.name, 0) + 1
        for name in ("_branch_key", "_compact_rvk", "_is_parent_like",
                     "_source_rank", "_status_rank"):
            with self.subTest(name=name):
                self.assertEqual(counts.get(name), 1, f"{name} is defined more than once")


class TestIsStrongAnchorCandidate(unittest.TestCase):
    """The keep-unconditionally rule, extracted from
    ``_validate_catalog_rvk_candidates`` (F-14) so it can be tested.

    Two anchor hits is always strong; one hit needs corroboration.
    """

    def setUp(self):
        from src.utils._pipeline_rvk_scoring import _is_strong_anchor_candidate
        self.strong = _is_strong_anchor_candidate

    def test_two_anchor_hits_is_always_strong(self):
        self.assertTrue(self.strong({"anchor_hit_count": 2}))
        self.assertTrue(self.strong({"anchor_hit_count": 5}))

    def test_zero_anchor_hits_is_never_strong(self):
        self.assertFalse(self.strong({"anchor_hit_count": 0, "count_value": 999}))

    def test_one_hit_needs_corroboration(self):
        """A lone anchor hit alone is not enough."""
        self.assertFalse(self.strong({"anchor_hit_count": 1}))
        self.assertFalse(self.strong(
            {"anchor_hit_count": 1, "keyword_hit_count": 1,
             "count_value": 7, "title_hit_count": 2}
        ))

    def test_one_hit_with_repeated_keywords_is_strong(self):
        self.assertTrue(self.strong({"anchor_hit_count": 1, "keyword_hit_count": 2}))

    def test_one_hit_with_high_count_is_strong(self):
        self.assertTrue(self.strong({"anchor_hit_count": 1, "count_value": 8}))

    def test_one_hit_with_several_titles_is_strong(self):
        self.assertTrue(self.strong({"anchor_hit_count": 1, "title_hit_count": 3}))

    def test_the_corroboration_thresholds_are_boundaries(self):
        """Just below each threshold is not strong; at it, is."""
        self.assertFalse(self.strong({"anchor_hit_count": 1, "count_value": 7}))
        self.assertTrue(self.strong({"anchor_hit_count": 1, "count_value": 8}))
        self.assertFalse(self.strong({"anchor_hit_count": 1, "title_hit_count": 2}))
        self.assertTrue(self.strong({"anchor_hit_count": 1, "title_hit_count": 3}))


class TestCanTakeBranch(unittest.TestCase):
    """The per-branch shortlist cap, extracted from
    ``_validate_catalog_rvk_candidates`` (F-14) — closure vars are now params."""

    def setUp(self):
        from src.utils._pipeline_rvk_scoring import _can_take_branch
        self.can = _can_take_branch

    def test_branch_with_room_can_take(self):
        self.assertTrue(self.can({"branch": "WI"}, {"WI": 3}, 5))

    def test_branch_at_cap_cannot_take(self):
        self.assertFalse(self.can({"branch": "WI"}, {"WI": 5}, 5))

    def test_unseen_branch_can_take(self):
        self.assertTrue(self.can({"branch": "RB"}, {"WI": 5}, 1))

    def test_over_cap_cannot_take(self):
        self.assertFalse(self.can({"branch": "WI"}, {"WI": 6}, 5))


if __name__ == "__main__":
    unittest.main()
