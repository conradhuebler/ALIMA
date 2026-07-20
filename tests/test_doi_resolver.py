"""Tests for UnifiedResolver's DOI metadata contract - Claude Generated (F-2).

The DOI producers used to emit capitalised, source-specific keys
(``Title``/``Abstract``/``Authors``) while every other record shape in ALIMA
uses lowercase field names — and a second code path
(``input_sources/doi.py``) already emitted lowercase for the SAME APIs. F-2
collapses that; these tests pin the result.

The important one is the fallback-chain gate: it had NO coverage, and a missed
rename there degrades silently — ``abstract`` reads as empty, so Crossref never
satisfies the guard and the chain always walks past a perfectly good Crossref
abstract. Nothing crashes; results just get quietly worse.
"""

from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from src.utils.doi_resolver import UnifiedResolver

CROSSREF_WITH_ABSTRACT = (
    True,
    {"title": "Cadmium Toxicity", "doi": "10.1/x", "abstract": "Ein echter Abstract."},
    "Ein echter Abstract.",
)
CROSSREF_WITHOUT_ABSTRACT = (
    True,
    {"title": "Cadmium Toxicity", "doi": "10.1/x", "abstract": ""},
    "",
)


class TestFallbackChainGate(unittest.TestCase):
    """The :meth:`_resolve_doi_with_fallback` guard reads metadata["abstract"]."""

    def test_good_crossref_abstract_stops_the_chain(self):
        resolver = UnifiedResolver()
        with patch.object(resolver, "_resolve_crossref_doi",
                          return_value=CROSSREF_WITH_ABSTRACT) as crossref, \
             patch.object(resolver, "_resolve_openalex_doi") as openalex, \
             patch.object(resolver, "_resolve_datacite_doi") as datacite:
            success, metadata, abstract = resolver._resolve_doi_with_fallback("10.1/x")

        self.assertTrue(success)
        self.assertEqual(abstract, "Ein echter Abstract.")
        crossref.assert_called_once()
        # The regression this guards: a key mismatch makes the guard always fail,
        # so the chain walks on and the later sources overwrite a good result.
        openalex.assert_not_called()
        datacite.assert_not_called()

    def test_empty_crossref_abstract_advances_the_chain(self):
        resolver = UnifiedResolver()
        with patch.object(resolver, "_resolve_crossref_doi",
                          return_value=CROSSREF_WITHOUT_ABSTRACT), \
             patch.object(resolver, "_resolve_openalex_doi",
                          return_value=(True, {"abstract": "Von OpenAlex."},
                                        "Von OpenAlex.")) as openalex:
            success, metadata, abstract = resolver._resolve_doi_with_fallback("10.1/x")

        openalex.assert_called_once()
        self.assertEqual(abstract, "Von OpenAlex.")


class TestEmittedKeysAreLowercase(unittest.TestCase):
    """Every key WE mint is lowercase snake_case; upstream API fields are not ours."""

    KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

    def _assert_lowercase(self, metadata):
        for key in metadata:
            with self.subTest(key=key):
                self.assertRegex(key, self.KEY_RE)

    def test_generic_web_crawl_keys(self):
        resolver = UnifiedResolver()
        self._assert_lowercase(
            resolver._parse_generic_content("# Titel\n\nEin Absatz.", "https://e.org")
        )

    def test_springer_keys(self):
        resolver = UnifiedResolver()
        md = resolver._parse_springer_markdown_enhanced(
            "# Buch\n\nText.", "https://link.springer.com/book/10.1007/x"
        )
        self._assert_lowercase(md)
        # The multiword keys became snake_case rather than staying spaced.
        self.assertIn("table_of_contents", md)
        self.assertIn("container_title", md)


class TestFormatDoiMetadata(unittest.TestCase):
    def test_formats_lowercase_metadata(self):
        from src.utils.doi_resolver import format_doi_metadata

        text = format_doi_metadata({
            "title": "Cadmium Toxicity",
            "authors": "Jha, A.; Kumar, V.",
            "abstract": "Schwermetalle in Böden.",
            "table_of_contents": "1. Einleitung",
        }, fallback_text="")

        self.assertIn("Cadmium Toxicity", text)
        self.assertIn("Jha, A.; Kumar, V.", text)
        self.assertIn("Schwermetalle in Böden.", text)
        self.assertIn("1. Einleitung", text)

    def test_placeholder_values_are_skipped(self):
        from src.utils.doi_resolver import format_doi_metadata

        text = format_doi_metadata({
            "title": "Not available",
            "abstract": "Echter Text.",
        }, fallback_text="")
        self.assertNotIn("Not available", text)
        self.assertIn("Echter Text.", text)


if __name__ == "__main__":
    unittest.main()
