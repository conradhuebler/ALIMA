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

from src.utils.doi_resolver import (
    _MIN_RICH_ABSTRACT_CHARS,
    UnifiedResolver,
    looks_like_schemaless_url,
)

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


class TestSchemalessUrlRouting(unittest.TestCase):
    """A host-looking input without scheme is a URL, not a DOI. The former
    assume-it's-a-DOI fallthrough sent 'link.springer.com/book/…' down the
    Crossref chain and the GUI showed a one-sentence blurb instead of the
    4.7k-char landing-page crawl (reproduced live, Aug 4 2026)."""

    def test_helper_truth_table(self):
        self.assertTrue(looks_like_schemaless_url("link.springer.com/book/10.1007/x"))
        self.assertTrue(looks_like_schemaless_url("doi.org/10.5040/x"))
        self.assertTrue(looks_like_schemaless_url("example.com"))
        self.assertFalse(looks_like_schemaless_url("10.1007/978-3-031-47390-6"))
        self.assertFalse(looks_like_schemaless_url("https://link.springer.com/x"))
        self.assertFalse(looks_like_schemaless_url("nur ein text mit leerzeichen"))
        self.assertFalse(looks_like_schemaless_url("kein-host-anteil/pfad"))
        self.assertFalse(looks_like_schemaless_url(""))

    def _analyze(self, s):
        return UnifiedResolver()._analyze_input(s)

    def test_schemaless_springer_url_is_crawled_not_crossrefed(self):
        kind, value = self._analyze("link.springer.com/book/10.1007/978-3-031-47390-6")
        self.assertEqual(kind, "springer_url")
        self.assertEqual(value, "https://link.springer.com/book/10.1007/978-3-031-47390-6")

    def test_schemaless_doi_org_extracts_the_doi(self):
        kind, value = self._analyze("doi.org/10.1007/978-3-031-47390-6")
        self.assertEqual((kind, value), ("springer_doi", "10.1007/978-3-031-47390-6"))

    def test_schemaless_generic_host_is_a_generic_url(self):
        kind, value = self._analyze("example.com/artikel/42")
        self.assertEqual((kind, value), ("generic_url", "https://example.com/artikel/42"))

    def test_bare_dois_are_unchanged(self):
        self.assertEqual(
            self._analyze("10.1007/978-3-031-47390-6")[0], "springer_doi"
        )
        self.assertEqual(self._analyze("10.5040/9781350067417")[0], "crossref_doi")


class TestQualityEscalation(unittest.TestCase):
    """A thin API abstract (< _MIN_RICH_ABSTRACT_CHARS) escalates to a
    landing-page crawl; the richer result wins, API title/authors are kept."""

    _THIN = (
        True,
        {"title": "Cadmium Toxicity Mitigation", "doi": "10.5/x",
         "abstract": "Ein-Satz-Blurb.", "source": "crossref"},
        "Ein-Satz-Blurb.",
    )
    _RICH_TEXT = "X" * (_MIN_RICH_ABSTRACT_CHARS + 500)

    def _resolve(self, api_result, crawl_result):
        r = UnifiedResolver()
        with patch.object(r, "_resolve_doi_with_fallback", return_value=api_result) as api, \
             patch.object(r, "_resolve_generic_url", return_value=crawl_result) as crawl:
            out = r.resolve("10.5040/9781350067417")
        return out, api, crawl

    def test_thin_abstract_escalates_and_crawl_wins(self):
        crawl = (True, {"title": "Seitentitel", "content": "…", "source": "Generic Web Crawl"}, self._RICH_TEXT)
        (ok, meta, text), _api, crawl_mock = self._resolve(self._THIN, crawl)
        crawl_mock.assert_called_once_with("https://doi.org/10.5040/9781350067417")
        self.assertTrue(ok)
        self.assertEqual(text, self._RICH_TEXT)
        # API metadata is authoritative for title; abstract carries the crawl
        self.assertEqual(meta["title"], "Cadmium Toxicity Mitigation")
        self.assertEqual(meta["abstract"], self._RICH_TEXT)
        self.assertEqual(meta["source"], "crossref+landing_page_crawl")

    def test_rich_abstract_does_not_escalate(self):
        rich_api = (True, {"title": "T", "abstract": self._RICH_TEXT}, self._RICH_TEXT)
        (ok, _meta, text), _api, crawl_mock = self._resolve(rich_api, (False, None, "unused"))
        crawl_mock.assert_not_called()
        self.assertEqual(text, self._RICH_TEXT)

    def test_failed_crawl_keeps_the_api_result(self):
        (ok, meta, text), _api, _crawl = self._resolve(self._THIN, (False, None, "down"))
        self.assertTrue(ok)
        self.assertEqual(text, "Ein-Satz-Blurb.")
        self.assertEqual(meta["source"], "crossref")

    def test_shorter_crawl_keeps_the_api_result(self):
        (ok, _meta, text), _api, _crawl = self._resolve(
            self._THIN, (True, {"title": "T"}, "Kurz.")
        )
        self.assertEqual(text, "Ein-Satz-Blurb.")

    def test_api_total_failure_falls_back_to_crawl(self):
        crawl = (True, {"title": "Seitentitel"}, self._RICH_TEXT)
        (ok, meta, text), _api, _crawl = self._resolve((False, None, "kein Treffer"), crawl)
        self.assertTrue(ok)
        self.assertEqual(text, self._RICH_TEXT)
        self.assertEqual(meta["title"], "Seitentitel")


if __name__ == "__main__":
    unittest.main()
