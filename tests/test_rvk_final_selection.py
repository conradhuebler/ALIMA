"""Characterization tests for _select_final_rvk_candidates - Claude Generated (F-13).

This is the FINAL RVK selection — what actually reaches the output a librarian
sees. It was untested (the one test naming the module mocks scoring away), and it
is exactly the "plausible but wrong" class CLAUDE.md warns about: a defect here
doesn't crash, it silently returns the wrong classification.

The method is deterministic except one LLM call (``_score_rvk_shortlist_with_llm``
inside the prefilter); mocking that to a no-op makes the whole selection
reproducible. These pin the current behavior so it can be changed deliberately.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.utils._pipeline_rvk_scoring import RvkScoringMixin


class _Host(RvkScoringMixin):
    def __init__(self):
        self.logger = Mock(level=100)
        # No LLM boost — selection then rests purely on the deterministic score
        # + anchor/diversity logic. Each shortlisted code gets an empty score.
        self._score_rvk_shortlist_with_llm = Mock(return_value={})


def _cand(code, *, status="standard", keywords=None, count=1, branch=None,
          ancestor="Root > Sub > Leaf", source="catalog"):
    return {
        "type": "RVK",
        "dk": code,
        "rvk_validation_status": status,
        "matched_keywords": keywords or [],
        "count": count,
        "branch_family": branch if branch is not None else code.split()[0],
        "ancestor_path": ancestor,
        "titles": [],
        "register": [],
        "source": source,
    }


class TestFinalRvkSelection(unittest.TestCase):
    def setUp(self):
        self.host = _Host()

    def _select(self, candidates, **kw):
        return self.host._select_final_rvk_candidates(
            candidates, kw.pop("abstract", "Ein Abstract über Limnologie."), **kw
        )

    def test_non_rvk_candidates_are_ignored(self):
        out = self._select([
            {"type": "DK", "dk": "530.1"},
            _cand("WI 4700"),
        ])
        self.assertEqual(out, ["RVK WI 4700"])

    def test_empty_input_yields_empty(self):
        self.assertEqual(self._select([]), [])

    def test_result_codes_are_rvk_prefixed(self):
        out = self._select([_cand("WI 4700")])
        self.assertEqual(out, ["RVK WI 4700"])

    def test_standard_is_preferred_over_nonstandard(self):
        """If any standard candidate exists, non-standard ones are not returned."""
        out = self._select([
            _cand("WI 4700", status="standard"),
            _cand("QQ 100", status="non_standard"),
        ], max_standard=2, max_nonstandard=1)
        self.assertEqual(out, ["RVK WI 4700"])

    def test_nonstandard_used_only_when_no_standard(self):
        out = self._select([
            _cand("QQ 100", status="non_standard"),
            _cand("QQ 200", status="validation_error"),
        ], max_standard=2, max_nonstandard=1)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("RVK QQ"))

    def test_duplicate_notation_is_aggregated_not_repeated(self):
        """Same code from two catalog hits → one entry (counts summed)."""
        out = self._select([
            _cand("WI 4700", count=3, keywords=["Limnologie"]),
            _cand("WI 4700", count=5, keywords=["Gewässer"]),
        ], max_standard=2)
        self.assertEqual(out, ["RVK WI 4700"])

    def test_anchor_matching_candidate_ranks_first(self):
        """A candidate matching an anchor keyword beats one that doesn't."""
        out = self._select(
            [
                _cand("QD 1000", keywords=["Nebensache"], branch="QD"),
                _cand("WI 4700", keywords=["Limnologie"], branch="WI"),
            ],
            max_standard=1,
            rvk_anchor_keywords=["Limnologie"],
        )
        self.assertEqual(out, ["RVK WI 4700"])

    def test_cap_is_respected(self):
        out = self._select(
            [_cand(f"W{i} {1000+i}", branch=f"W{i}") for i in range(5)],
            max_standard=2,
        )
        self.assertEqual(len(out), 2)

    def test_diversity_covers_distinct_anchors(self):
        """The selection spreads across anchors rather than piling on one.

        The primary driver is anchor COVERAGE (``new_coverage * 45``): given two
        candidates on the same anchor and one on a second anchor, the second
        anchor's candidate is taken to cover more ground. (The ``-8`` same-branch
        penalty is a finer tiebreaker on top of this; it is not isolated here —
        a mutation removing it alone does not change this outcome.)
        """
        out = self._select(
            [
                _cand("WI 4700", keywords=["Limnologie"], branch="WI"),
                _cand("WI 4800", keywords=["Limnologie"], branch="WI"),
                _cand("RB 1000", keywords=["Ökologie"], branch="RB"),
            ],
            max_standard=2,
            rvk_anchor_keywords=["Limnologie", "Ökologie"],
        )
        self.assertEqual(len(out), 2)
        # both anchors represented: the Ökologie candidate (RB) is included
        # alongside a Limnologie one, not two Limnologie ones.
        self.assertIn("RVK RB 1000", out)

    def test_llm_score_is_consulted(self):
        """The prefilter calls the LLM scorer; a high score can reorder."""
        self.host._score_rvk_shortlist_with_llm = Mock(return_value={
            "QD 1000": {"total_score": 100, "reason": "perfekt"},
        })
        out = self._select(
            [
                _cand("WI 4700", count=9, branch="WI"),
                _cand("QD 1000", count=1, branch="QD"),
            ],
            max_standard=1,
        )
        self.assertEqual(out, ["RVK QD 1000"])
        self.host._score_rvk_shortlist_with_llm.assert_called()


if __name__ == "__main__":
    unittest.main()
