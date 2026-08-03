"""Characterisation tests for batch input parsing and naming - Claude Generated.

``batch_processor.py`` (895 lines) had NO test referencing it — and that is where
the JSON-save crash lived unnoticed: every batch run reported per-item failures
instead of results. These tests cover the parts that decide what gets processed
and what the output is called, i.e. the places where a defect is silent rather
than loud.

Characterisation, not specification: current behaviour is pinned so it can be
changed deliberately. Questionable behaviour is marked, not corrected.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.batch_processor import (
    BatchSource,
    BatchSourceParser,
    BatchState,
    SourceType,
)


class TestLineParsing(unittest.TestCase):
    def setUp(self):
        self.parser = BatchSourceParser(logger=Mock())

    def _one(self, line):
        return self.parser._parse_line(line, 1)

    def test_explicit_type_prefix(self):
        src = self._one("DOI:10.1007/978-3-031-47390-6")
        self.assertEqual(src.source_type, SourceType.DOI)
        self.assertEqual(src.source_value, "10.1007/978-3-031-47390-6")

    def test_type_prefix_is_case_insensitive(self):
        self.assertEqual(self._one("doi:10.1/x").source_type, SourceType.DOI)

    def test_bare_doi_is_auto_detected(self):
        """A plain DOI without prefix is the common paste-from-browser case."""
        src = self._one("10.1007/978-3-031-47390-6")
        self.assertEqual(src.source_type, SourceType.DOI)

    def test_doi_org_url_is_reduced_to_the_doi(self):
        src = self._one("https://doi.org/10.1007/978-3-031-47390-6")
        self.assertEqual(src.source_type, SourceType.DOI)
        self.assertEqual(src.source_value, "10.1007/978-3-031-47390-6")

    def test_custom_name_after_semicolon(self):
        src = self._one("DOI:10.1/x;Mein Titel")
        self.assertEqual(src.custom_name, "Mein Titel")

    def test_step_overrides_are_parsed_as_json(self):
        src = self._one('DOI:10.1/x;Name;{"keywords": {"model": "m"}}')
        self.assertEqual(src.step_overrides, {"keywords": {"model": "m"}})

    def test_broken_override_json_warns_and_continues(self):
        """A typo in the third field must not lose the whole line."""
        src = self._one("DOI:10.1/x;Name;{kaputt")
        self.assertIsNone(src.step_overrides)
        self.assertEqual(src.source_type, SourceType.DOI)
        self.parser.logger.warning.assert_called()

    def test_missing_separator_is_rejected(self):
        with self.assertRaises(ValueError):
            self._one("irgendein freitext")

    def test_unknown_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self._one("BLAH:wert")

    def test_missing_file_warns_but_still_parses(self):
        """CHARACTERISATION: a nonexistent path yields a source, not an error.

        The batch then fails on that item at processing time. Deliberate or not,
        it means a typo'd path is reported late rather than at parse time.
        """
        src = self._one("PDF:/nicht/vorhanden.pdf")
        self.assertEqual(src.source_type, SourceType.PDF)
        self.parser.logger.warning.assert_called()


class TestBatchTextParsing(unittest.TestCase):
    def setUp(self):
        self.parser = BatchSourceParser(logger=Mock())

    def test_comments_and_blanks_are_skipped(self):
        sources = self.parser.parse_batch_text(
            "# Kommentar\n\nDOI:10.1/a\n   \nDOI:10.1/b\n"
        )
        self.assertEqual([s.source_value for s in sources], ["10.1/a", "10.1/b"])

    def test_a_bad_line_does_not_sink_the_good_ones(self):
        sources = self.parser.parse_batch_text("DOI:10.1/a\nkaputt\nDOI:10.1/b")
        self.assertEqual([s.source_value for s in sources], ["10.1/a", "10.1/b"])

    def test_line_numbers_are_recorded_for_diagnostics(self):
        sources = self.parser.parse_batch_text("# c\nDOI:10.1/a")
        self.assertEqual(sources[0].line_number, 2)


class TestSafeFilename(unittest.TestCase):
    def test_doi_slashes_become_underscores(self):
        src = BatchSource(source_type=SourceType.DOI, source_value="10.1007/978-3-031")
        self.assertEqual(src.get_safe_filename(), "10_1007_978-3-031.json")

    def test_custom_name_wins(self):
        src = BatchSource(
            source_type=SourceType.DOI, source_value="10.1/x", custom_name="Mein Titel"
        )
        self.assertEqual(src.get_safe_filename(), "Mein_Titel.json")

    def test_url_uses_the_host(self):
        src = BatchSource(
            source_type=SourceType.URL, source_value="https://example.org/a/b?c=d"
        )
        self.assertEqual(src.get_safe_filename(), "example_org.json")

    def test_file_path_uses_the_stem(self):
        src = BatchSource(source_type=SourceType.PDF, source_value="/tmp/Ein Buch.pdf")
        self.assertEqual(src.get_safe_filename(), "Ein_Buch.json")

    def test_umlauts_survive(self):
        """``str.isalnum()`` is unicode-aware, so German titles keep their shape.

        (I expected these to be stripped — they are not. The regex in
        ``_sanitize_filename`` lists äöüÄÖÜß explicitly, which reads as if the
        default were to drop them; ``\w`` would have kept them anyway.)
        """
        src = BatchSource(
            source_type=SourceType.DOI, source_value="x", custom_name="Öl und Wärme"
        )
        self.assertEqual(src.get_safe_filename(), "Öl_und_Wärme.json")


class TestSanitizeFilename(unittest.TestCase):
    def setUp(self):
        self.proc = object.__new__(
            __import__("src.utils.batch_processor", fromlist=["BatchProcessor"]).BatchProcessor
        )

    def test_umlauts_survive_here(self):
        self.assertEqual(self.proc._sanitize_filename("Öl und Wärme"), "Öl_und_Wärme")

    def test_punctuation_is_stripped(self):
        self.assertEqual(self.proc._sanitize_filename("Cadmium: Toxizität!"), "Cadmium_Toxizität")

    def test_truncated_at_max_length(self):
        self.assertEqual(len(self.proc._sanitize_filename("a" * 80, max_length=20)), 20)

    def test_empty_stays_empty(self):
        self.assertEqual(self.proc._sanitize_filename(""), "")


class TestPoorQualityHeuristic(unittest.TestCase):
    """Decides whether PDF text is bad enough to fall back to Vision-OCR."""

    def setUp(self):
        self.proc = object.__new__(
            __import__("src.utils.batch_processor", fromlist=["BatchProcessor"]).BatchProcessor
        )

    def test_clean_german_prose_is_good(self):
        self.assertFalse(
            self.proc._is_poor_quality_text(
                "Limnologische Studien zur Seenkunde im Alpenraum, mit Messreihen."
            )
        )

    def test_ocr_garbage_is_poor(self):
        self.assertTrue(self.proc._is_poor_quality_text("§$%&/()=?*#~|<>[]{}\\+^°"))

    def test_empty_is_poor(self):
        self.assertTrue(self.proc._is_poor_quality_text(""))

    def test_the_threshold_is_half_non_alphanumeric(self):
        """Pinning the boundary: 50% alnum+space is the cut."""
        self.assertFalse(self.proc._is_poor_quality_text("abcd" + "%%%"))   # 4/7 ≈ 0.57
        self.assertTrue(self.proc._is_poor_quality_text("abc" + "%%%%"))    # 3/7 ≈ 0.43

    def test_umlauts_count_as_alphanumeric(self):
        """str.isalnum() is unicode-aware — German text is not punished."""
        self.assertFalse(self.proc._is_poor_quality_text("Öl Wärme Größe Fluß"))


class TestIsbnPpnResolve(unittest.TestCase):
    """The ISBN/PPN branch of _resolve_source_to_text now runs over the shared
    BibRecord path (WP-D1 P1) instead of two hand-rolled formatter copies."""

    def setUp(self):
        import logging
        from src.utils.batch_processor import BatchProcessor

        self.proc = object.__new__(BatchProcessor)
        self.proc.logger = logging.getLogger("test_batch_isbn_ppn")

    _CLS = {"DK": [{"code": "556.55", "origin": "authority"}]}

    def _record(self):
        from src.core.bib_record import BibRecord

        return BibRecord(
            source="sru",
            title="Limnologie der Alpenseen",
            authors=["Müller, Anna"],
            publisher="Verlag X",
            subjects=["Seenkunde"],
            abstract="Studien zur Seenkunde.",
            classifications=dict(self._CLS),
        )

    def test_isbn_uses_isbn_index_and_returns_metadata(self):
        from src.utils.batch_processor import BatchSource, SourceType

        with patch("src.utils.input_sources.bib_lookup.lookup_bibrecord") as lookup:
            lookup.return_value = self._record()
            text, metadata = self.proc._resolve_source_to_text(
                BatchSource(SourceType.ISBN, "9780000000001")
            )
        self.assertEqual(lookup.call_args.kwargs["search_type"], "isbn")
        self.assertIn("Titel: Limnologie der Alpenseen", text)
        self.assertEqual(
            metadata,
            {
                "title": "Limnologie der Alpenseen",
                "authors": "Müller, Anna",
                "source": "ISBN",
                # WP-D1 P2: the record's own classifications travel along
                "classifications": self._CLS,
            },
        )

    def test_ppn_uses_keyword_index(self):
        from src.utils.batch_processor import BatchSource, SourceType

        with patch("src.utils.input_sources.bib_lookup.lookup_bibrecord") as lookup:
            lookup.return_value = self._record()
            _text, metadata = self.proc._resolve_source_to_text(
                BatchSource(SourceType.PPN, "998877")
            )
        self.assertEqual(lookup.call_args.kwargs["search_type"], "keyword")
        self.assertEqual(metadata["source"], "PPN")

    def test_no_hit_becomes_runtime_error_naming_the_identifier(self):
        from src.utils.batch_processor import BatchSource, SourceType

        with patch("src.utils.input_sources.bib_lookup.lookup_bibrecord") as lookup:
            lookup.side_effect = ValueError("Keine Treffer für 978X")
            with self.assertRaises(RuntimeError) as ctx:
                self.proc._resolve_source_to_text(BatchSource(SourceType.ISBN, "978X"))
        self.assertIn("Failed to lookup ISBN 978X", str(ctx.exception))


class TestBatchStateResume(unittest.TestCase):
    """Resume support: a crashed batch must not reprocess what it finished."""

    def _src(self, value):
        return BatchSource(source_type=SourceType.DOI, source_value=value)

    def test_processed_sources_are_remembered(self):
        state = BatchState(batch_file="b.txt", output_dir="out", total_sources=2)
        src = self._src("10.1/a")
        self.assertFalse(state.is_processed(src))
        state.mark_processed(src, success=True)
        self.assertTrue(state.is_processed(src))

    def test_failures_are_recorded_separately(self):
        state = BatchState(batch_file="b.txt", output_dir="out", total_sources=1)
        state.mark_processed(self._src("10.1/a"), success=False, error="boom")
        self.assertEqual(len(state.failed_sources), 1)
        self.assertEqual(state.failed_sources[0]["error"], "boom")

    def test_state_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "state.json")
            state = BatchState(batch_file="b.txt", output_dir="out", total_sources=2)
            state.mark_processed(self._src("10.1/a"), success=True)
            state.save(path)

            reloaded = BatchState.load(path)
            self.assertTrue(reloaded.is_processed(self._src("10.1/a")))
            self.assertFalse(reloaded.is_processed(self._src("10.1/b")))


if __name__ == "__main__":
    unittest.main()
