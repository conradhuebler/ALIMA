"""Tests for the input-source plugin category - Claude Generated.

Covers the registry + ``execute_input_extraction`` dispatch (byte-parity for the
built-in text/file types), the extracted ``url_fetch`` scraper, and the three
split DOI sources (``can_handle`` + backend selection).
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.input_sources import get_input_source, list_input_sources
from src.utils.input_sources.url_fetch import scrape_url
from src.utils import pipeline_input


class RegistryTest(unittest.TestCase):
    def test_builtin_sources_registered(self):
        ids = set(list_input_sources())
        self.assertTrue(
            {"text", "file", "pdf", "image", "url_fetch", "doi_crossref", "doi_openalex", "doi_datacite"} <= ids
        )

    def test_every_source_has_complete_doc(self):
        """Design requirement: each input source self-describes (desc + input + output)."""
        for sid in list_input_sources():
            doc = get_input_source(sid).doc()
            self.assertTrue(doc.is_complete(), f"{sid} doc incomplete: {doc}")


class DispatchParityTest(unittest.TestCase):
    def test_text_extraction(self):
        self.assertEqual(
            pipeline_input.execute_input_extraction(None, "  hi  ", input_type="text"),
            ("hi", "Direkter Text", "text"),
        )

    def test_file_extraction(self):
        p = Path(tempfile.mkdtemp()) / "a.txt"
        p.write_text("Inhalt")
        self.assertEqual(
            pipeline_input.execute_input_extraction(None, str(p), input_type="file"),
            ("Inhalt", "Textdatei: a.txt", "file_read"),
        )

    def test_auto_detects_text_and_file(self):
        self.assertEqual(pipeline_input.execute_input_extraction(None, "abc", input_type="auto")[2], "text")
        p = Path(tempfile.mkdtemp()) / "b.txt"
        p.write_text("x")
        self.assertEqual(pipeline_input.execute_input_extraction(None, str(p), input_type="auto")[2], "file_read")

    def test_unknown_type_raises(self):
        with self.assertRaises(Exception) as ctx:
            pipeline_input.execute_input_extraction(None, "x", input_type="bogus")
        self.assertIn("fehlgeschlagen", str(ctx.exception))


class UrlFetchTest(unittest.TestCase):
    def _resp(self, html: str):
        return types.SimpleNamespace(content=html.encode("utf-8"), raise_for_status=lambda: None)

    def test_scrape_main_content(self):
        html = "<html><body><main>" + ("Wort " * 30) + "</main><nav>skip</nav></body></html>"
        with patch("requests.get", return_value=self._resp(html)):
            text = scrape_url("http://x", min_chars=10)
        self.assertIn("Wort", text)
        self.assertNotIn("skip", text)

    def test_too_little_text_raises(self):
        with patch("requests.get", return_value=self._resp("<html><body><main>hi</main></body></html>")):
            with self.assertRaises(ValueError):
                scrape_url("http://x", min_chars=50)

    def test_source_can_handle(self):
        src = get_input_source("url_fetch")()
        self.assertTrue(src.can_handle("http://a.org", "auto"))
        self.assertTrue(src.can_handle("anything", "url"))
        self.assertFalse(src.can_handle("plain text", "auto"))


class DoiSourceTest(unittest.TestCase):
    def test_can_handle_doi(self):
        cr = get_input_source("doi_crossref")()
        self.assertTrue(cr.can_handle("10.1007/abc123", "auto"))
        self.assertTrue(cr.can_handle("whatever", "doi"))
        self.assertFalse(cr.can_handle("just words", "auto"))

    def test_config_fields_differ(self):
        cr = [f.key for f in get_input_source("doi_crossref").config_fields()]
        dc = [f.key for f in get_input_source("doi_datacite").config_fields()]
        self.assertIn("contact_email", cr)  # crossref/openalex have polite-pool email
        self.assertNotIn("contact_email", dc)  # datacite does not
        self.assertIn("timeout", cr)
        self.assertIn("timeout", dc)

    def test_extract_enables_only_own_backend(self):
        captured = {}

        class _FakeResolver:
            def __init__(self, logger, contact_email="", use_crossref=True, use_openalex=True, use_datacite=True):
                captured["flags"] = (use_crossref, use_openalex, use_datacite)
                captured["email"] = contact_email

            def resolve(self, s):
                return True, {}, "ABSTRACT"

        with patch("src.utils.doi_resolver.UnifiedResolver", _FakeResolver):
            src = get_input_source("doi_openalex")(contact_email="a@b.c")
            text, info, method = src.extract("10.1/x")
        self.assertEqual(text, "ABSTRACT")
        self.assertEqual(method, "doi_openalex")
        self.assertEqual(captured["flags"], (False, True, False))  # only openalex
        self.assertEqual(captured["email"], "a@b.c")

    def test_extract_failure_raises(self):
        class _FailResolver:
            def __init__(self, *a, **k):
                pass

            def resolve(self, s):
                return False, None, "not found"

        with patch("src.utils.doi_resolver.UnifiedResolver", _FailResolver):
            with self.assertRaises(RuntimeError):
                get_input_source("doi_datacite")().extract("10.1/x")

    def test_each_doi_declares_independent_tool_spec(self):
        specs = {sid: get_input_source(sid).mcp_tool_spec() for sid in
                 ("doi_crossref", "doi_openalex", "doi_datacite")}
        self.assertEqual(specs["doi_crossref"].name, "resolve_doi_crossref")
        self.assertEqual(specs["doi_openalex"].name, "resolve_doi_openalex")
        self.assertEqual(specs["doi_datacite"].name, "resolve_doi_datacite")
        for s in specs.values():
            self.assertEqual(s.param, "doi")
        # sources without a tool spec are not exposed
        self.assertIsNone(getattr(get_input_source("text"), "mcp_tool_spec", lambda: None)())

    def _resp(self, status, payload):
        return types.SimpleNamespace(status_code=status, json=lambda: payload)

    def test_mcp_execute_returns_complete_openalex_metadata_without_abstract(self):
        """The bug: OpenAlex has metadata but no abstract → must still return it."""
        seen = {}

        def fake_get(url, headers=None, timeout=None, **kw):
            seen["url"] = url
            return self._resp(200, {
                "display_name": "SupraFit", "publication_year": 2022,
                "authorships": [{"author": {"display_name": "Conrad Hübler"}}],
                # no abstract_inverted_index → previously dropped as "failure"
            })

        with patch("requests.get", fake_get):
            out = get_input_source("doi_openalex")(contact_email="a@b.c").mcp_execute("10.1002/cmtd.202200006")
        self.assertTrue(out["success"])                       # success despite no abstract
        self.assertEqual(out["source"], "openalex")
        self.assertEqual(out["metadata"]["display_name"], "SupraFit")
        self.assertIn("openalex.org/works/doi:10.1002", seen["url"])
        self.assertIn("mailto=a@b.c", seen["url"])            # polite-pool email applied

    def test_mcp_execute_crossref_full_message_and_doi_normalised(self):
        def fake_get(url, headers=None, timeout=None, **kw):
            assert "api.crossref.org/works/10.1002/cmtd.202200006" in url, url
            return self._resp(200, {"status": "ok", "message": {
                "title": ["SupraFit"], "publisher": "Wiley", "type": "journal-article"}})

        with patch("requests.get", fake_get):
            # DOI passed as a doi.org URL → must be normalised
            out = get_input_source("doi_crossref")().mcp_execute("https://doi.org/10.1002/cmtd.202200006")
        self.assertTrue(out["success"])
        self.assertEqual(out["metadata"]["publisher"], "Wiley")
        self.assertEqual(out["doi"], "10.1002/cmtd.202200006")

    def test_mcp_execute_reports_http_error(self):
        def fake_get(url, headers=None, timeout=None, **kw):
            return self._resp(404, {})

        with patch("requests.get", fake_get):
            out = get_input_source("doi_datacite")().mcp_execute("10.1/x")
        self.assertFalse(out["success"])
        self.assertIn("404", out["error"])

    def test_mcp_execute_reconstructs_openalex_abstract(self):
        def fake_get(url, headers=None, timeout=None, **kw):
            return self._resp(200, {
                "display_name": "T",
                "abstract_inverted_index": {"Deep": [0], "learning": [1], "works": [2]},
            })

        with patch("requests.get", fake_get):
            out = get_input_source("doi_openalex")().mcp_execute("10.1/x")
        self.assertEqual(out["metadata"]["abstract"], "Deep learning works")
        self.assertNotIn("abstract_inverted_index", out["metadata"])  # replaced


if __name__ == "__main__":
    unittest.main()
