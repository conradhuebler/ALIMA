"""Characterisation tests for DK/RVK extraction from LLM responses - Claude Generated.

``_pipeline_dk_steps.py`` (1186 lines) had NO test referencing it. This function
is the sharpest edge in it: it turns an LLM's free text into the classifications
the pipeline reports. A miss here does not crash — it silently drops a
classification, or invents one from a number that happened to look like a code.

These are CHARACTERISATION tests: they pin what the code does today, so the
behaviour can be changed deliberately rather than by accident. Where the current
behaviour looks questionable it is marked, not quietly corrected.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.utils._pipeline_dk_steps import DkStepsMixin


class _Host(DkStepsMixin):
    """Minimal carrier: the mixin only needs a logger."""

    def __init__(self):
        self.logger = Mock(level=100)


class TestFinalListIsPreferred(unittest.TestCase):
    def setUp(self):
        self.host = _Host()

    def _extract(self, text):
        return self.host._extract_dk_from_response(text, output_format="xml")

    def test_prefixed_codes_pass_through(self):
        self.assertEqual(
            self._extract("<final_list>DK 615.9 | RVK QC 130</final_list>"),
            ["DK 615.9", "RVK QC 130"],
        )

    def test_bare_number_is_assumed_to_be_dk(self):
        self.assertEqual(self._extract("<final_list>615.9</final_list>"), ["DK 615.9"])

    def test_letter_number_is_assumed_to_be_rvk(self):
        self.assertEqual(self._extract("<final_list>QC 130</final_list>"), ["RVK QC 130"])

    def test_prose_around_the_tag_is_ignored(self):
        text = (
            "Ich schlage folgende Klassifikationen vor, weil DK 999.9 nicht passt:\n"
            "<final_list>DK 615.9</final_list>\nHoffe das hilft."
        )
        self.assertEqual(self._extract(text), ["DK 615.9"])

    def test_empty_segments_are_dropped(self):
        self.assertEqual(
            self._extract("<final_list>DK 615.9 |  | DK 504.53</final_list>"),
            ["DK 615.9", "DK 504.53"],
        )

    def test_tag_is_matched_case_insensitively(self):
        self.assertEqual(
            self._extract("<FINAL_LIST>DK 615.9</FINAL_LIST>"), ["DK 615.9"]
        )

    def test_unknown_format_is_kept_verbatim(self):
        """Deliberate: an unrecognised entry is passed on rather than dropped."""
        self.assertEqual(
            self._extract("<final_list>Chemie allgemein</final_list>"),
            ["Chemie allgemein"],
        )

    def test_lowercase_prefix_keeps_its_original_casing(self):
        """CHARACTERISATION: the prefix test upper-cases, the OUTPUT does not.

        "dk 615.9" is recognised as prefixed and stored as-is, so the emitted
        string is lowercase while a sibling from the same list is uppercase.
        Downstream ``split_classification_code`` matches case-insensitively, so
        this is survivable — but the pipeline does emit mixed casing.
        """
        self.assertEqual(self._extract("<final_list>dk 615.9</final_list>"), ["dk 615.9"])


class TestRegexFallback(unittest.TestCase):
    """Used only when no <final_list> is present; the code calls it less reliable."""

    def setUp(self):
        self.host = _Host()

    def _extract(self, text):
        return self.host._extract_dk_from_response(text, output_format="xml")

    def test_prefixed_codes_are_found_in_prose(self):
        self.assertEqual(
            self._extract("Passend sind DK 615.9 und RVK QC 130."),
            ["DK 615.9", "RVK QC 130"],
        )

    def test_bare_numbers_are_NOT_harvested(self):
        """The guard against inventing classifications from arbitrary numbers."""
        self.assertEqual(self._extract("Auf Seite 615.9 steht dazu nichts."), [])

    def test_duplicates_collapse_preserving_order(self):
        self.assertEqual(
            self._extract("DK 615.9, spaeter nochmal DK 615.9 und DK 504.53"),
            ["DK 615.9", "DK 504.53"],
        )

    def test_no_classifications_yields_empty_list(self):
        self.assertEqual(self._extract("Dazu kann ich nichts sagen."), [])

    def test_dk_code_longer_than_three_leading_digits_is_missed(self):
        """CHARACTERISATION — looks like a defect.

        The fallback pattern is ``\\bDK\\s+(\\d{1,3}(?:\\.\\d+)*)``: at most three
        digits before the first dot. A four-digit DK notation is therefore
        skipped entirely in the fallback path, silently. Pinned here rather than
        widened, because whether 4+ digit DK codes occur in this catalogue is a
        domain question, not a code one.
        """
        self.assertEqual(self._extract("Hier passt DK 1234.5 am besten."), [])


class TestJsonPathWins(unittest.TestCase):
    def test_json_response_short_circuits_the_xml_path(self):
        host = _Host()
        text = '{"classifications": [{"code": "DK 615.9"}]}'
        result = host._extract_dk_from_response(text)
        self.assertTrue(result, "JSON path produced nothing for a JSON response")

    def test_broken_json_falls_back_and_still_extracts(self):
        """The salvage direction: bad JSON must not lose a usable <final_list>."""
        host = _Host()
        text = '{"classifications": [ kaputt <final_list>DK 615.9</final_list>'
        self.assertEqual(host._extract_dk_from_response(text), ["DK 615.9"])


if __name__ == "__main__":
    unittest.main()
