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


class ResolutionStagesTest(unittest.TestCase):
    """The two-stage input-type resolution (WP-D1 P1 landmine fix): exact
    registry id first, then the ``can_handle`` contract — which every source
    declared but nothing called until July 2026. This is what lets the surface
    aliases (``doi``/``url``) and new record types reach the one dispatcher."""

    def test_exact_id_wins(self):
        self.assertIs(
            pipeline_input._resolve_input_source_class("text", "abc"),
            get_input_source("text"),
        )

    def test_alias_url_resolves_via_can_handle(self):
        self.assertIs(
            pipeline_input._resolve_input_source_class("url", "https://a.org"),
            get_input_source("url_fetch"),
        )

    def test_alias_doi_resolves_to_first_doi_source(self):
        """Deliberate: the alias picks ONE backend (crossref); the fallback
        chain stays with resolve_input_to_text."""
        self.assertIs(
            pipeline_input._resolve_input_source_class("doi", "10.1/x"),
            get_input_source("doi_crossref"),
        )

    def test_unknown_type_resolves_to_none(self):
        self.assertIsNone(pipeline_input._resolve_input_source_class("bogus", "x"))


class BibLookupTest(unittest.TestCase):
    """ISBN/PPN input sources (WP-D1 P1) — record → BibRecord → analysis text."""

    _HIT = {
        "rsn": "998877",
        "title": "Limnologie der Alpenseen",
        "author": ["Müller, Anna"],
        "publication": "Verlag X",
        "subjects": ["Seenkunde"],
        "gnd_subjects": [{"term": "Limnologie", "gnd_id": "4074296-3"}],
        "abstract": "Studien zur Seenkunde.",
    }

    def _client(self, results, captured=None):
        class _FakeClient:
            def __init__(self, preset="", timeout=30, max_records=50, **kw):
                if captured is not None:
                    captured.update(preset=preset, timeout=timeout, max_records=max_records)

            def search(self, term, search_type="keyword"):
                if captured is not None:
                    captured.update(term=term, search_type=search_type)
                return results

        return _FakeClient

    def test_registered(self):
        self.assertIn("isbn", list_input_sources())
        self.assertIn("ppn", list_input_sources())

    def test_isbn_extract_formats_record(self):
        captured = {}
        with patch("src.utils.clients.marcxml_client.MarcXmlClient", self._client([self._HIT], captured)):
            text, info, method = get_input_source("isbn")().extract("9780000000001")
        self.assertEqual(method, "isbn")
        self.assertEqual(captured["search_type"], "isbn")
        self.assertEqual(captured["max_records"], 1)
        self.assertIn("Titel: Limnologie der Alpenseen", text)
        self.assertIn("Schlagwörter: Seenkunde; Limnologie", text)
        self.assertIn("Limnologie der Alpenseen", info)

    def test_ppn_uses_the_ppn_index(self):
        """WP-D1 P4 fix: the former keyword index searched the PPN as a
        SUBJECT term (pica.slw) and never matched — live-verified 0 hits."""
        captured = {}
        with patch("src.utils.clients.marcxml_client.MarcXmlClient", self._client([self._HIT], captured)):
            get_input_source("ppn")().extract("998877")
        self.assertEqual(captured["search_type"], "ppn")

    def test_no_hit_raises(self):
        with patch("src.utils.clients.marcxml_client.MarcXmlClient", self._client([])):
            with self.assertRaises(ValueError) as ctx:
                get_input_source("isbn")().extract("9780000000009")
        self.assertIn("Keine Treffer", str(ctx.exception))

    def test_record_sink_side_channel_carries_the_bibrecord(self):
        """WP-D1 P2: a caller-supplied record_sink dict receives the BibRecord
        (the 3-tuple contract cannot carry it); without one nothing changes."""
        hit = dict(self._HIT, classifications=["DK 556.55"])
        sink = {}
        with patch("src.utils.clients.marcxml_client.MarcXmlClient", self._client([hit])):
            get_input_source("isbn")().extract("9780000000001", record_sink=sink)
        record = sink.get("record")
        self.assertIsNotNone(record)
        self.assertEqual(record.title, "Limnologie der Alpenseen")
        self.assertIn("DK", record.classifications)

    def test_dispatches_through_execute_input_extraction(self):
        """End to end: the new type reaches the ONE dispatcher by id."""
        with patch("src.utils.clients.marcxml_client.MarcXmlClient", self._client([self._HIT])), \
             patch.object(pipeline_input, "_input_settings_for", return_value={}):
            text, info, method = pipeline_input.execute_input_extraction(
                None, "9780000000001", input_type="isbn"
            )
        self.assertEqual(method, "isbn")
        self.assertIn("Titel:", text)


class UrlFetchTest(unittest.TestCase):
    """scrape_url now routes through net_guard.fetch_guarded (SSRF guard):
    mock the guard's response surface + a public DNS resolution. - Claude Generated"""

    def _resp(self, html: str):
        body = html.encode("utf-8")
        return types.SimpleNamespace(
            status_code=200,
            headers={},
            iter_content=lambda chunk_size: iter([body]),
            close=lambda: None,
        )

    def _public_dns(self):
        return patch("socket.getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 80))])

    def test_scrape_main_content(self):
        html = "<html><body><main>" + ("Wort " * 30) + "</main><nav>skip</nav></body></html>"
        with self._public_dns(), patch("requests.get", return_value=self._resp(html)):
            text = scrape_url("http://x", min_chars=10, allowlist=[], max_bytes=10_000_000)
        self.assertIn("Wort", text)
        self.assertNotIn("skip", text)

    def test_too_little_text_raises(self):
        with self._public_dns(), patch(
            "requests.get", return_value=self._resp("<html><body><main>hi</main></body></html>")
        ):
            with self.assertRaises(ValueError):
                scrape_url("http://x", min_chars=50, allowlist=[], max_bytes=10_000_000)

    def test_private_target_blocked(self):
        with patch("socket.getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 80))]):
            with self.assertRaises(RuntimeError):
                scrape_url("http://internal.host/x", allowlist=[], max_bytes=10_000_000)

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
