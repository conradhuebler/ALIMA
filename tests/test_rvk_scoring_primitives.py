"""Characterisation tests for the RVK scoring primitives - Claude Generated.

``_pipeline_rvk_scoring.py`` (1992 lines) had NO test referencing it — and the
one test that names it (``test_rvk_lookup_tool``) MOCKS the scoring away. These
functions decide which classification a librarian is offered. A defect here does
not raise: it reorders the shortlist, and a plausible-looking but wrong notation
comes out on top. That is the failure mode CLAUDE.md calls the dangerous one.

Structural note: only 21 of the module's 40 functions are reachable at all. The
other 19 are nested inside methods (8 alone in
``_validate_catalog_rvk_candidates``), including the RVK hierarchy helpers
``_is_parent_like``/``_compact_rvk``/``_branch_key``. Those are untestable
without extracting them first — deliberately left for a separate decision rather
than refactored under the cover of "adding tests".
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.utils._pipeline_rvk_scoring import RvkScoringMixin


class _Host(RvkScoringMixin):
    def __init__(self):
        self.logger = Mock(level=100)


def _candidate(**kw):
    """A catalogue RVK candidate as the scorer receives it."""
    base = {"dk": "WI 4700", "label": "", "ancestor_path": "", "register": []}
    base.update(kw)
    return base


class TestSignificantTokens(unittest.TestCase):
    """Feeds the deterministic ranking — dropping a content word costs a match."""

    def setUp(self):
        self.host = _Host()

    def test_stopwords_are_removed(self):
        self.assertEqual(
            self.host._rvk_significant_tokens("Die Analyse von Cadmium in Boeden"),
            ["cadmium", "boeden"],
        )

    def test_short_tokens_are_dropped(self):
        """The pattern requires 4+ letters."""
        self.assertEqual(self.host._rvk_significant_tokens("Ion pH Blei"), ["blei"])

    def test_umlauts_are_kept_as_content(self):
        self.assertIn("gewässer", self.host._rvk_significant_tokens("Gewässer"))

    def test_digits_are_not_tokens(self):
        self.assertEqual(self.host._rvk_significant_tokens("Cadmium 2024 615.9"), ["cadmium"])

    def test_generic_words_are_treated_as_stopwords(self):
        """CHARACTERISATION: 'analyse'/'geschichte'/'text' are suppressed on purpose.

        Documented in the source as "generic high-frequency tokens contribute
        little" — worth knowing, because a genuinely historical work loses
        'geschichte' as a ranking signal.
        """
        self.assertEqual(self.host._rvk_significant_tokens("Geschichte und Analyse"), [])

    def test_empty_input_is_not_an_error(self):
        for value in ("", None):
            with self.subTest(value=value):
                self.assertEqual(self.host._rvk_significant_tokens(value), [])


class TestBroadnessPenalty(unittest.TestCase):
    """Pushes shallow/vague nodes down. The thresholds are pinned here."""

    def setUp(self):
        self.host = _Host()

    def test_shallower_nodes_are_penalised_harder(self):
        deep = self.host._rvk_broadness_penalty(
            _candidate(ancestor_path="A > B > C > D > E", label="Limnologie Gewässer",
                       register=["Seen", "Fluss"], dk="WI 4700")
        )
        shallow = self.host._rvk_broadness_penalty(
            _candidate(ancestor_path="A", label="Limnologie Gewässer",
                       register=["Seen", "Fluss"], dk="WI 4700")
        )
        self.assertGreater(shallow, deep)

    def test_depth_tiers_are_monotonic(self):
        def pen(path):
            return self.host._rvk_broadness_penalty(
                _candidate(ancestor_path=path, label="Limnologie Gewässer",
                           register=["a", "b"], dk="WI 4700")
            )
        self.assertEqual(
            [pen("A"), pen("A > B"), pen("A > B > C"), pen("A > B > C > D")],
            [60, 35, 15, 0],
        )

    def test_thin_label_and_register_add_a_penalty(self):
        rich = self.host._rvk_broadness_penalty(
            _candidate(ancestor_path="A > B > C > D", label="Limnologie Gewässer",
                       register=["Seen", "Fluss"], dk="WI 4700")
        )
        thin = self.host._rvk_broadness_penalty(
            _candidate(ancestor_path="A > B > C > D", label="Allgemein",
                       register=[], dk="WI 4700")
        )
        self.assertEqual(thin - rich, 18)

    def test_short_notation_adds_a_penalty(self):
        long_code = self.host._rvk_broadness_penalty(
            _candidate(ancestor_path="A > B > C > D", label="Limnologie Gewässer",
                       register=["a", "b"], dk="WI 47000")
        )
        short_code = self.host._rvk_broadness_penalty(
            _candidate(ancestor_path="A > B > C > D", label="Limnologie Gewässer",
                       register=["a", "b"], dk="WI 1")
        )
        self.assertEqual(short_code - long_code, 12)

    def test_empty_candidate_is_maximally_penalised(self):
        self.assertEqual(self.host._rvk_broadness_penalty(_candidate(dk="", label="")), 90)


class TestInstitutionLibraryDetection(unittest.TestCase):
    """Filters catalogue artefacts: notations for individual libraries."""

    def setUp(self):
        self.host = _Host()

    def test_single_library_branch_is_detected(self):
        self.assertTrue(self.host._is_institution_library_rvk(_candidate(
            label="Einzelne Bibliotheken",
            ancestor_path="Allgemeines > Buch- und Bibliothekswesen",
        )))

    def test_bibliothekswesen_alone_is_not_enough(self):
        """The general subject stays; only single-library nodes are artefacts."""
        self.assertFalse(self.host._is_institution_library_rvk(_candidate(
            label="Bibliothekswesen allgemein",
            ancestor_path="Allgemeines > Buch- und Bibliothekswesen",
        )))

    def test_unrelated_candidate_is_not_flagged(self):
        self.assertFalse(self.host._is_institution_library_rvk(_candidate(
            label="Limnologie", ancestor_path="Biologie > Ökologie"
        )))

    def test_matching_is_case_insensitive(self):
        self.assertTrue(self.host._is_institution_library_rvk(_candidate(
            label="EINZELNE BIBLIOTHEKEN", ancestor_path="BIBLIOTHEKSWESEN"
        )))

    def test_missing_fields_are_not_an_error(self):
        self.assertFalse(self.host._is_institution_library_rvk({}))


class TestDomainProfile(unittest.TestCase):
    """Maps the abstract onto coarse domains, used for branch-fit scoring."""

    def setUp(self):
        self.host = _Host()

    def test_profile_is_a_mapping_of_counts(self):
        profile = self.host._rvk_domain_profile(
            "Limnologische Studien zur Gewässerökologie und Biologie der Seen",
            ["Limnologie", "Ökologie"],
        )
        self.assertIsInstance(profile, dict)
        for value in profile.values():
            self.assertIsInstance(value, int)

    def test_empty_input_yields_an_empty_or_zero_profile(self):
        profile = self.host._rvk_domain_profile("", [])
        self.assertFalse(any(profile.values()), f"unexpected signal from nothing: {profile}")


if __name__ == "__main__":
    unittest.main()
