#!/usr/bin/env python3
"""Claude Generated - Tests for FincClient, FincSuggester, and the
search_finc MCP tool.

Covers:
- FincClient.search: record normalization, query param encoding, error
  handling for empty base_url, network failure, HTTP != 200, bad JSON.
- FincSuggester.search: BaseSuggester-conformant {term: {records,
  result_count, errors}} shape, search_type mapping kw/title/subject/
  author/freetext -> VuFind type, default institution_filter, last_errors
  propagation.
- ToolRegistry._handle_search_finc: end-to-end MCP path with mocked
  FincSuggester, including the "finc not configured" error path.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from src.mcp.tool_registry import ToolRegistry
from src.utils.clients.finc_client import FincClient
from src.utils.suggesters.finc_suggester import FincSuggester


# --------------------------------------------------------------------------
# Sample VuFind-JSON payloads
# --------------------------------------------------------------------------

MOCK_OK_PAYLOAD = {
    "status": "OK",
    "resultCount": 2,
    "records": [
        {
            "id": "0-1025700295",
            "title": "Python: der Grundkurs",
            "authors": {
                "primary": {"Kofler, Michael": ["aut"]},
                "primary_orig": {"Kofler, Michael": []},
                "corporate": [], "corporate_orig": [],
                "corporate_secondary": [], "corporate_secondary_orig": [],
                "secondary": [], "secondary_orig": [],
            },
            "formats": ["Book"],
            "languages": ["German"],
            "series": [],
            "subjects": [["Python"]],
            "urls": [],
        },
        {
            "id": "0-9999999",
            "title": "Python for Everybody",
            "authors": {
                "primary": {"Severance, Charles": ["aut"]},
                "primary_orig": {"Severance, Charles": []},
                "corporate": [], "corporate_orig": [],
                "corporate_secondary": [], "corporate_secondary_orig": [],
                "secondary": [], "secondary_orig": [],
            },
            "formats": ["eBook"],
            "languages": ["English"],
            "series": [{"name": "Open textbook library", "number": ""}],
            "subjects": [["Python"], ["Programming"]],
            "urls": [
                {"url": "https://example.org/0-9999999", "desc": "Full Text", "indicators": "40"}
            ],
        },
    ],
}


def _make_mock_response(payload, status_code=200):
    """Build a MagicMock that quacks like requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.url = "https://dobby.example/proxy.php/api/v1/search"
    return resp


# --------------------------------------------------------------------------
# FincClient unit tests
# --------------------------------------------------------------------------

class TestFincClientUnit(unittest.TestCase):
    """Unit tests for the HTTP client. No real network calls."""

    def test_empty_base_url_returns_clean_error(self):
        client = FincClient(base_url="")
        result = client.search("python")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["resultCount"], 0)
        self.assertEqual(result["records"], [])
        self.assertIn("not configured", result["error"])

    def test_empty_lookfor_returns_clean_error(self):
        # Separate from base_url test so the base_url guard doesn't
        # shadow the lookfor guard. - Claude Generated
        client = FincClient(base_url="https://dobby.example/proxy.php")
        with patch.object(client.session, "get") as mock_get:
            result = client.search("")
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("non-empty", result["error"])
        mock_get.assert_not_called()

    def test_search_returns_normalized_records(self):
        client = FincClient(
            base_url="https://dobby.example/proxy.php",
            web_record_url="https://katalog.example/Record/",
        )
        with patch.object(client.session, "get", return_value=_make_mock_response(MOCK_OK_PAYLOAD)) as mock_get:
            result = client.search("python", type="AllFields", limit=20)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["resultCount"], 2)
        self.assertEqual(len(result["records"]), 2)
        first = result["records"][0]
        # Normalized shape (operator-approved Biblio-style)
        self.assertEqual(first["id"], "0-1025700295")
        self.assertEqual(first["title"], "Python: der Grundkurs")
        self.assertEqual(first["authors"]["primary"]["Kofler, Michael"], ["aut"])
        self.assertEqual(first["subjects"], [["Python"]])
        self.assertEqual(first["formats"], ["Book"])
        self.assertEqual(first["languages"], ["German"])
        self.assertEqual(first["web_url"], "https://katalog.example/Record/0-1025700295")
        self.assertIn("raw", first)  # raw payload preserved
        # Second record keeps series & urls passthrough
        second = result["records"][1]
        self.assertEqual(second["series"], [{"name": "Open textbook library", "number": ""}])
        self.assertEqual(len(second["urls"]), 1)

    def test_search_with_subject_and_filter_encoding(self):
        client = FincClient(
            base_url="https://dobby.example/proxy.php",
            web_record_url="https://katalog.example/Record/",
        )
        with patch.object(client.session, "get", return_value=_make_mock_response(MOCK_OK_PAYLOAD)) as mock_get:
            result = client.search(
                "python",
                type="Subject",
                filters={"udk_facet_de105": "IT. Informatik. Software"},
                limit=10,
            )
        self.assertEqual(result["status"], "OK")
        # Inspect the actual request the client issued
        self.assertEqual(mock_get.call_count, 1)
        call_args = mock_get.call_args
        url = call_args.args[0]
        params = call_args.kwargs["params"]
        self.assertEqual(url, "https://dobby.example/proxy.php/api/v1/search")
        # params is a list of tuples; check key members
        params_dict = {k: v for k, v in params if k != "filter[]"}
        self.assertEqual(params_dict["lookfor"], "python")
        self.assertEqual(params_dict["type"], "Subject")
        self.assertEqual(params_dict["limit"], 10)
        filter_entries = [v for k, v in params if k == "filter[]"]
        self.assertEqual(
            filter_entries,
            ['udk_facet_de105:"IT. Informatik. Software"'],
        )

    def test_search_handles_network_error(self):
        import requests
        client = FincClient(base_url="https://dobby.example/proxy.php")
        with patch.object(
            client.session, "get",
            side_effect=requests.exceptions.ConnectionError("DNS failure"),
        ):
            result = client.search("python")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["resultCount"], 0)
        self.assertEqual(result["records"], [])
        self.assertIn("network error", result["error"])
        self.assertIn("DNS failure", result["error"])

    def test_search_handles_http_error(self):
        client = FincClient(base_url="https://dobby.example/proxy.php")
        bad = _make_mock_response({}, status_code=500)
        with patch.object(client.session, "get", return_value=bad):
            result = client.search("python")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["http_status"], 500)
        self.assertIn("HTTP 500", result["error"])

    def test_search_handles_invalid_json(self):
        client = FincClient(base_url="https://dobby.example/proxy.php")
        bad = MagicMock()
        bad.status_code = 200
        bad.json.side_effect = ValueError("not json")
        bad.url = "https://dobby.example/proxy.php/api/v1/search"
        with patch.object(client.session, "get", return_value=bad):
            result = client.search("python")
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("invalid JSON", result["error"])

    def test_http_200_with_status_error_is_surfaced(self):
        # The finc proxy returns HTTP 200 even on query errors, signalling
        # failure only via the envelope: {"status":"ERROR","statusMessage":...}.
        # The client must NOT rewrite that into a silent OK/0-results. - Claude Generated
        client = FincClient(base_url="https://dobby.example/proxy.php")
        err_payload = {"status": "ERROR", "statusMessage": "Invalid search"}
        with patch.object(client.session, "get", return_value=_make_mock_response(err_payload)):
            result = client.search("python", filters={"bogus_facet": "x"})
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["resultCount"], 0)
        self.assertEqual(result["records"], [])
        self.assertIn("Invalid search", result["error"])
        self.assertEqual(result["http_status"], 200)

    def test_limit_is_clamped_to_max(self):
        client = FincClient(base_url="https://dobby.example/proxy.php")
        with patch.object(client.session, "get", return_value=_make_mock_response(MOCK_OK_PAYLOAD)) as mock_get:
            client.search("python", limit=9999)
        params = mock_get.call_args.kwargs["params"]
        limit_value = next(v for k, v in params if k == "limit")
        self.assertEqual(limit_value, FincClient.MAX_LIMIT)

    def test_web_url_assembly(self):
        # Trailing slash on web_record_url should not produce a double slash
        client = FincClient(
            base_url="https://dobby.example/proxy.php",
            web_record_url="https://katalog.example/Record/",  # trailing slash
        )
        with patch.object(client.session, "get", return_value=_make_mock_response(MOCK_OK_PAYLOAD)):
            result = client.search("python")
        for record in result["records"]:
            self.assertTrue(
                record["web_url"].startswith("https://katalog.example/Record/"),
                f"web_url has double slash or wrong base: {record['web_url']}",
            )
            self.assertNotIn("//0-", record["web_url"])

    def test_web_url_is_catalog_only_never_publisher(self):
        # web_url is the CATALOG record page only. When no record base is
        # configured it stays empty — it must NEVER fall back to a publisher /
        # full-text URL (that belongs in resource_url). - Claude Generated
        client = FincClient(
            base_url="https://dobby.example/proxy.php",
            web_record_url="",  # disabled
        )
        with patch.object(client.session, "get", return_value=_make_mock_response(MOCK_OK_PAYLOAD)):
            result = client.search("python")
        # No record base → no catalog link (not a publisher URL).
        self.assertEqual(result["records"][0]["web_url"], "")
        self.assertEqual(result["records"][1]["web_url"], "")
        # The publisher/full-text link is offered separately as resource_url.
        self.assertEqual(result["records"][0]["resource_url"], "")
        self.assertEqual(
            result["records"][1]["resource_url"],
            "https://example.org/0-9999999",
        )

    def test_resource_url_separate_from_web_url(self):
        # With a record base configured, web_url is the catalog entry AND
        # resource_url is the publisher link — both present. - Claude Generated
        client = FincClient(
            base_url="https://dobby.example/proxy.php",
            web_record_url="https://katalog.example/Record/",
        )
        with patch.object(client.session, "get", return_value=_make_mock_response(MOCK_OK_PAYLOAD)):
            result = client.search("python")
        second = result["records"][1]
        self.assertEqual(second["web_url"], "https://katalog.example/Record/0-9999999")
        self.assertEqual(second["resource_url"], "https://example.org/0-9999999")

    def test_facets_param_emitted_and_block_parsed(self):
        client = FincClient(base_url="https://dobby.example/proxy.php")
        payload = {
            "status": "OK",
            "resultCount": 5,
            "records": [],
            "facets": {
                "udk_raw_de105": [
                    {"value": "dk 530.145", "count": 249, "translated": "dk 530.145",
                     "href": "?x"},
                    {"value": "dk 54", "count": 39, "translated": "dk 54", "href": "?y"},
                ],
            },
        }
        with patch.object(client.session, "get", return_value=_make_mock_response(payload)) as mock_get:
            result = client.search(
                "python", type="Subject",
                facets=["udk_raw_de105", "rvk_facet"], limit=0,
            )
        # facet[] params emitted, one per facet
        params = mock_get.call_args.kwargs["params"]
        facet_entries = [v for k, v in params if k == "facet[]"]
        self.assertEqual(facet_entries, ["udk_raw_de105", "rvk_facet"])
        # limit=0 is honoured (facet-only request), not bumped to default
        limit_value = next(v for k, v in params if k == "limit")
        self.assertEqual(limit_value, 0)
        # facets parsed, href dropped, count coerced to int
        buckets = result["facets"]["udk_raw_de105"]
        self.assertEqual(buckets[0], {"value": "dk 530.145", "count": 249, "translated": "dk 530.145"})
        self.assertNotIn("href", buckets[0])

    def test_facets_empty_when_not_requested(self):
        client = FincClient(base_url="https://dobby.example/proxy.php")
        with patch.object(client.session, "get", return_value=_make_mock_response(MOCK_OK_PAYLOAD)):
            result = client.search("python")
        self.assertEqual(result["facets"], {})

    def test_normalize_dk_value(self):
        self.assertEqual(FincClient.normalize_dk_value("dk 530.145"), "DK 530.145")
        self.assertEqual(FincClient.normalize_dk_value("DK 681.3"), "DK 681.3")
        self.assertEqual(FincClient.normalize_dk_value("dk 535.33/.35"), "DK 535.33/.35")
        # Non-DK values pass through unchanged
        self.assertEqual(FincClient.normalize_dk_value("uk 1000"), "uk 1000")
        self.assertEqual(FincClient.normalize_dk_value(""), "")


# --------------------------------------------------------------------------
# FincSuggester unit tests
# --------------------------------------------------------------------------

class TestFincSuggesterUnit(unittest.TestCase):

    def _make_suggester(self, **kwargs):
        defaults = dict(
            base_url="https://dobby.example/proxy.php",
            web_record_url="https://katalog.example/Record/",
            institution_filter="",
        )
        defaults.update(kwargs)
        return FincSuggester(**defaults)

    def test_search_returns_base_class_shape(self):
        s = self._make_suggester()
        s.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 1, "records": [
                {"id": "0-1", "title": "X", "authors": {}, "subjects": [],
                 "formats": [], "languages": [], "series": [], "urls": [],
                 "web_url": "", "raw": {}}
            ]
        })
        out = s.search(["python", "java"])
        # Per-term shape
        for term in ("python", "java"):
            self.assertIn(term, out)
            self.assertIn("records", out[term])
            self.assertIn("result_count", out[term])
            self.assertIn("errors", out[term])
            self.assertIsInstance(out[term]["records"], list)
        self.assertEqual(out["python"]["result_count"], 1)
        self.assertEqual(out["python"]["records"][0]["title"], "X")

    def test_search_type_mapping(self):
        s = self._make_suggester()
        s.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })
        s.search(["x"], search_type="kw")
        s.search(["x"], search_type="title")
        s.search(["x"], search_type="subject")
        s.search(["x"], search_type="author")
        s.search(["x"], search_type="freetext")
        s.search(["x"], search_type="unknown")  # fallback to AllFields
        types = [c.kwargs["type"] for c in s.client.search.call_args_list]
        self.assertEqual(
            types,
            ["AllFields", "Title", "Subject", "Author", "AllFields", "AllFields"],
        )

    def test_dk_rvk_search_type_mapping(self):
        s = self._make_suggester()
        s.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })
        s.search(["DK 57"], search_type="dk")
        s.search(["UC 100"], search_type="rvk")
        types = [c.kwargs["type"] for c in s.client.search.call_args_list]
        self.assertEqual(types, ["udk_raw_de105", "rvk_facet"])

    def test_default_institution_filter_applied(self):
        s = self._make_suggester(institution_filter="DE-105")
        s.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })
        s.search(["x"])
        filters = s.client.search.call_args.kwargs["filters"]
        self.assertEqual(filters, {"institution": "DE-105"})

    def test_per_call_filters_override_defaults(self):
        s = self._make_suggester(institution_filter="DE-105")
        s.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })
        s.search(["x"], filters={"udk_facet_de105": "IT."})
        filters = s.client.search.call_args.kwargs["filters"]
        # Per-call filters REPLACE the default — caller is explicit
        self.assertEqual(filters, {"udk_facet_de105": "IT."})

    def test_per_term_failure_lands_in_last_errors(self):
        s = self._make_suggester()
        s.client.search = MagicMock(return_value={
            "status": "ERROR", "resultCount": 0, "records": [],
            "error": "boom"
        })
        out = s.search(["python"])
        self.assertEqual(out["python"]["records"], [])
        self.assertIn("python", s.last_errors)
        self.assertEqual(out["python"]["errors"], ["boom"])

    def test_facets_surfaced_per_term(self):
        s = self._make_suggester()
        s.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 2, "records": [],
            "facets": {"udk_raw_de105": [{"value": "dk 530.145", "count": 2, "translated": "dk 530.145"}]},
        })
        out = s.search(["Quantenmechanik"], search_type="subject", facets=["udk_raw_de105"])
        # facets threaded to the client
        self.assertEqual(s.client.search.call_args.kwargs["facets"], ["udk_raw_de105"])
        # and surfaced per term
        self.assertEqual(
            out["Quantenmechanik"]["facets"]["udk_raw_de105"][0]["value"], "dk 530.145"
        )

    def test_prepare_is_noop(self):
        s = self._make_suggester()
        # Should not raise
        self.assertIsNone(s.prepare())
        s.prepare(force_download=True)
        # Still noop
        self.assertIsNone(s.prepare(force_download=True))

    def test_unconfigured_client_returns_errors_per_term(self):
        s = FincSuggester(base_url="")
        out = s.search(["x"])
        self.assertIn("x", out)
        self.assertEqual(out["x"]["records"], [])
        self.assertEqual(out["x"]["result_count"], 0)
        self.assertTrue(out["x"]["errors"])
        self.assertIn("x", s.last_errors)


# --------------------------------------------------------------------------
# ToolRegistry MCP integration
# --------------------------------------------------------------------------

class TestSearchFincMCPHandler(unittest.TestCase):
    """End-to-end test of the search_finc MCP tool."""

    def _make_registry(self, catalog_cfg):
        cfg_mgr = MagicMock()
        cfg_mgr.get_catalog_config.return_value = catalog_cfg
        reg = ToolRegistry(config_manager=cfg_mgr)  # noqa: F841
        return reg

    def test_init_finc_falls_back_to_catalog_web_record_url(self):
        # GUI bug: finc_web_record_url empty → the FincSuggester must still build
        # catalog links by falling back to catalog_web_record_url at the source
        # (so it works even when ToolRegistry has no injected config_manager).
        # - Claude Generated
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""  # not configured
        catalog_cfg.catalog_web_record_url = "https://katalog.ub.tu-freiberg.de/Record/"
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        self.assertEqual(
            reg._finc.client.web_record_url,
            "https://katalog.ub.tu-freiberg.de/Record/",
        )

    def test_handler_reconstructs_catalog_web_url_keeps_resource_url(self):
        # The reported bug: finc record has only a publisher link, no catalog
        # web_url. The handler MUST fill web_url from catalog_web_record_url (the
        # catalog link must be there) while resource_url keeps the book link.
        # - Claude Generated
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""  # finc base not configured
        catalog_cfg.catalog_web_record_url = "https://katalog.ub.tu-freiberg.de/Record/"
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 1, "records": [
                {"id": "0-1846124905", "title": "Quantenchemie", "authors": {},
                 "subjects": [], "formats": [], "languages": [], "series": [],
                 "urls": [{"url": "https://www.degruyterbrill.com/isbn/9783111215075"}],
                 "web_url": "",
                 "resource_url": "https://www.degruyterbrill.com/isbn/9783111215075",
                 "raw": {}}
            ]
        })

        data = json.loads(reg._handle_search_finc(terms=["Quantenchemie"]))
        rec = data["results"]["Quantenchemie"]["records"][0]
        self.assertEqual(
            rec["web_url"], "https://katalog.ub.tu-freiberg.de/Record/0-1846124905"
        )
        self.assertEqual(
            rec["resource_url"], "https://www.degruyterbrill.com/isbn/9783111215075"
        )

    def test_handler_returns_records_for_each_term(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = "https://katalog.example/Record/"
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        self.assertIsNotNone(reg._finc, "FincSuggester should be initialized")

        # Mock the underlying client so the test is hermetic
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 1, "records": [
                {"id": "0-1", "title": "Python", "authors": {}, "subjects": [],
                 "formats": [], "languages": [], "series": [], "urls": [],
                 "web_url": "https://katalog.example/Record/0-1", "raw": {}}
            ]
        })

        out = reg._handle_search_finc(
            terms=["python", "java"],
            search_type="subject",
            filters={"udk_facet_de105": "IT."},
            limit=5,
        )
        data = json.loads(out)
        self.assertEqual(data["source"], "finc")
        self.assertIn("python", data["results"])
        self.assertIn("java", data["results"])
        for term in ("python", "java"):
            entry = data["results"][term]
            self.assertEqual(entry["result_count"], 1)
            self.assertEqual(entry["records"][0]["title"], "Python")
        # Errors should be empty (search succeeded)
        self.assertEqual(data["errors"], {})

        # Verify params passed through to the client
        call = reg._finc.client.search.call_args
        self.assertEqual(call.kwargs["type"], "Subject")
        self.assertEqual(call.kwargs["filters"], {"udk_facet_de105": "IT."})
        self.assertEqual(call.kwargs["limit"], 5)

    def test_handler_threads_facets_through(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 1, "records": [],
            "facets": {"udk_raw_de105": [{"value": "dk 530.145", "count": 1, "translated": "dk 530.145"}]},
        })

        out = reg._handle_search_finc(
            terms=["python"], search_type="subject",
            facets=["udk_raw_de105"], limit=0,
        )
        data = json.loads(out)
        self.assertEqual(
            data["results"]["python"]["facets"]["udk_raw_de105"][0]["value"], "dk 530.145"
        )
        call = reg._finc.client.search.call_args
        self.assertEqual(call.kwargs["facets"], ["udk_raw_de105"])
        self.assertEqual(call.kwargs["limit"], 0)

    def test_handler_errors_when_finc_not_configured(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = ""  # not configured
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        self.assertIsNone(reg._finc, "FincSuggester must stay None when not configured")

        out = reg._handle_search_finc(terms=["python"])
        data = json.loads(out)
        self.assertIn("error", data)
        self.assertIn("not configured", data["error"])

    def test_handler_reports_per_term_errors(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "ERROR", "resultCount": 0, "records": [],
            "error": "upstream timeout"
        })

        out = reg._handle_search_finc(terms=["python"])
        data = json.loads(out)
        self.assertEqual(data["source"], "finc")
        self.assertEqual(data["results"]["python"]["records"], [])
        self.assertIn("python", data["errors"])
        self.assertIn("upstream timeout", data["errors"]["python"])

    def test_handler_uses_default_institution_filter_from_config(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = "DE-105"

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })

        reg._handle_search_finc(terms=["python"])
        filters = reg._finc.client.search.call_args.kwargs["filters"]
        self.assertEqual(filters, {"institution": "DE-105"})

    def test_handler_dk_search_type_auto_adds_facets(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": [],
            "facets": {
                "udk_raw_de105": [{"value": "dk 57", "count": 5, "translated": "dk 57"}],
                "rvk_facet": [{"value": "WW 3350", "count": 3, "translated": "WW 3350"}],
            },
        })

        reg._handle_search_finc(terms=["DK 57"], search_type="dk")
        call = reg._finc.client.search.call_args
        # Both facets auto-added when caller passed none
        self.assertIn("udk_raw_de105", call.kwargs["facets"])
        self.assertIn("rvk_facet", call.kwargs["facets"])
        self.assertEqual(call.kwargs["type"], "udk_raw_de105")

    def test_handler_dk_search_type_respects_explicit_facets(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })

        # Caller explicitly requests only one facet — must not be overridden
        reg._handle_search_finc(terms=["DK 57"], search_type="dk", facets=["udk_raw_de105"])
        call = reg._finc.client.search.call_args
        self.assertEqual(call.kwargs["facets"], ["udk_raw_de105"])

    def test_availability_local_injects_facet_avail_filter(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })

        reg._handle_search_finc(terms=["chemie"], availability="local")
        call_filters = reg._finc.client.search.call_args.kwargs.get("filters", {})
        self.assertEqual(call_filters.get("facet_avail"), "Local")

    def test_availability_online_maps_correctly(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })

        reg._handle_search_finc(terms=["chemie"], availability="online")
        filters = reg._finc.client.search.call_args.kwargs.get("filters", {})
        self.assertEqual(filters.get("facet_avail"), "Online")

    def test_availability_free_maps_correctly(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })

        reg._handle_search_finc(terms=["chemie"], availability="free")
        filters = reg._finc.client.search.call_args.kwargs.get("filters", {})
        self.assertEqual(filters.get("facet_avail"), "Free")

    def test_availability_none_does_not_inject_filter(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })

        reg._handle_search_finc(terms=["chemie"])
        filters = reg._finc.client.search.call_args.kwargs.get("filters") or {}
        self.assertNotIn("facet_avail", filters)

    def test_explicit_facet_avail_not_overridden_by_availability(self):
        # Caller passes filters={"facet_avail": "Online"} AND availability="local" —
        # explicit filter wins (setdefault semantics).
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 0, "records": []
        })

        reg._handle_search_finc(
            terms=["chemie"],
            filters={"facet_avail": "Online"},
            availability="local",
        )
        filters = reg._finc.client.search.call_args.kwargs.get("filters", {})
        self.assertEqual(filters.get("facet_avail"), "Online")

    def test_web_url_reconstructed_from_catalog_base_when_missing(self):
        # If finc_web_record_url is not configured, web_url is empty.
        # The handler must fill it in from catalog_web_record_url + id.
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""     # not configured
        catalog_cfg.catalog_web_record_url = "https://katalog.example.org/Record"
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 1, "records": [
                {"id": "0-123", "title": "Chemie", "authors": {},
                 "subjects": [], "formats": [], "languages": [],
                 "series": [], "urls": [], "web_url": "", "raw": {}}
            ]
        })

        out = reg._handle_search_finc(terms=["chemie"])
        data = json.loads(out)
        rec = data["results"]["chemie"]["records"][0]
        self.assertEqual(rec["web_url"], "https://katalog.example.org/Record/0-123")

    def test_web_url_not_overwritten_when_already_present(self):
        # If FincClient already built web_url, the handler must not overwrite it.
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = "https://katalog.example.org/Record/"
        catalog_cfg.catalog_web_record_url = "https://WRONG.example.org/Record"
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        reg._finc.client.search = MagicMock(return_value={
            "status": "OK", "resultCount": 1, "records": [
                {"id": "0-456", "title": "Physik", "authors": {},
                 "subjects": [], "formats": [], "languages": [],
                 "series": [], "urls": [],
                 "web_url": "https://katalog.example.org/Record/0-456",
                 "raw": {}}
            ]
        })

        out = reg._handle_search_finc(terms=["physik"])
        data = json.loads(out)
        rec = data["results"]["physik"]["records"][0]
        # Must keep the original web_url, not replace with WRONG
        self.assertEqual(rec["web_url"], "https://katalog.example.org/Record/0-456")

    def test_tool_registered_in_library_preset(self):
        # search_finc is now generated from FincProvider's ProviderToolSpec.
        from src.mcp.tool_schemas import LIBRARY_TOOLS
        self.assertIn("search_finc", [t.name for t in LIBRARY_TOOLS])

    def test_handler_swallows_unexpected_exceptions(self):
        catalog_cfg = MagicMock()
        catalog_cfg.finc_base_url = "https://dobby.example/proxy.php"
        catalog_cfg.finc_web_record_url = ""
        catalog_cfg.finc_default_limit = 20
        catalog_cfg.finc_timeout = 30
        catalog_cfg.finc_institution_filter = ""

        from src.mcp.tool_registry import ToolRegistry
        reg = self._make_registry(catalog_cfg)
        reg._init_suggesters()
        # Make the suggester blow up
        reg._finc.search = MagicMock(side_effect=RuntimeError("kaboom"))

        out = reg._handle_search_finc(terms=["python"])
        data = json.loads(out)
        self.assertIn("error", data)
        self.assertIn("kaboom", data["error"])


# --------------------------------------------------------------------------
# Optional live integration test
# --------------------------------------------------------------------------

class TestFincClientLive(unittest.TestCase):
    """Live integration test against an operator-provided finc endpoint.

    The endpoint is read from the FINC_TEST_BASE_URL env var (no institution
    URL is hard-coded). Run with:
        RUN_INTEGRATION_TESTS=1 FINC_TEST_BASE_URL=<finc-proxy-url> \\
            FINC_TEST_RECORD_URL=<record-base-url> FINC_TEST_QUERY='"…"' \\
            python -m unittest tests.test_finc_client.TestFincClientLive
    """

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS") == "1",
        "Integration tests disabled (set RUN_INTEGRATION_TESTS=1 to enable)",
    )
    def test_live_search(self):
        base_url = os.environ.get("FINC_TEST_BASE_URL")
        if not base_url:
            self.skipTest("set FINC_TEST_BASE_URL to the finc proxy endpoint to run this test")
        client = FincClient(
            base_url=base_url,
            web_record_url=os.environ.get("FINC_TEST_RECORD_URL", ""),
            timeout=30,
        )
        result = client.search(os.environ.get("FINC_TEST_QUERY", '"dana kuhnert"'), limit=5)
        self.assertEqual(result["status"], "OK")
        self.assertGreater(len(result["records"]), 0)
        for r in result["records"]:
            self.assertTrue(r["id"], "record missing id")
            self.assertTrue(r["web_url"], "record missing web_url")


if __name__ == "__main__":
    unittest.main()
