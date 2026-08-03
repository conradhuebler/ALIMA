"""Tests for the K10plus identifier crosswalk (WP-D1 P4) - Claude Generated.

The K10plus ``pica.doi`` index TOKENIZES its input: a quoted DOI phrase reports
millions of "hits" with the exact match merely ranked first (live-verified
Aug 3, 2026). The crosswalk therefore VERIFIES every candidate client-side
against the requested identifier — these tests pin that no-trust contract.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.utils.k10plus_resolver import (
    K10PlusRecord,
    detect_identifier_kind,
    fetch_record_for_identifier,
)


def _pica_xml(records):
    """Minimal PICA-XML SRU response for the given (ppn, doi, isbn) triples."""
    body = ""
    for ppn, doi, isbn in records:
        fields = f'<datafield tag="003@"><subfield code="0">{ppn}</subfield></datafield>'
        if doi:
            fields += f'<datafield tag="004V"><subfield code="0">{doi}</subfield></datafield>'
        if isbn:
            fields += f'<datafield tag="004A"><subfield code="0">{isbn}</subfield></datafield>'
        body += (
            "<zs:record><zs:recordData>"
            f'<record xmlns="info:srw/schema/5/picaXML-v1.0">{fields}</record>'
            "</zs:recordData></zs:record>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<zs:searchRetrieveResponse xmlns:zs="http://www.loc.gov/zing/srw/">'
        f"<zs:records>{body}</zs:records></zs:searchRetrieveResponse>"
    ).encode()


class _Resp:
    def __init__(self, content):
        self.content = content
        self.status_code = 200

    def raise_for_status(self):
        pass


class TestDetectIdentifierKind(unittest.TestCase):
    def test_doi_forms(self):
        self.assertEqual(detect_identifier_kind("10.5040/9781350067417"), "doi")
        self.assertEqual(detect_identifier_kind("https://doi.org/10.1007/x"), "doi")

    def test_isbn13_bookland(self):
        self.assertEqual(detect_identifier_kind("9781350067417"), "isbn")
        self.assertEqual(detect_identifier_kind("978-1-350-06741-7"), "isbn")

    def test_isbn10_with_formatting(self):
        self.assertEqual(detect_identifier_kind("3-16-148410-X"), "isbn")

    def test_bare_ten_digits_are_ppn(self):
        """The documented ambiguity decision: a bare 10-digit number is a PPN
        (K10plus record ids look exactly like this); kind='isbn' overrides."""
        self.assertEqual(detect_identifier_kind("1963652258"), "ppn")

    def test_other_record_ids_are_ppn(self):
        self.assertEqual(detect_identifier_kind("84738549X"), "ppn")


class TestFetchRecordForIdentifier(unittest.TestCase):
    _DOI = "10.5040/9781350067417"

    def _fetch(self, xml, identifier, kind=None):
        with patch("requests.get", return_value=_Resp(xml)):
            return fetch_record_for_identifier(identifier, kind=kind)

    def test_doi_candidates_are_verified_not_trusted(self):
        """First candidate wrong (tokenized index noise) → the verified second
        one is returned."""
        xml = _pica_xml([
            ("111", "10.9999/other", ""),
            ("222", self._DOI, "9781350067417"),
        ])
        rec = self._fetch(xml, self._DOI)
        self.assertEqual(rec.ppn, "222")

    def test_no_verified_candidate_yields_none(self):
        """Better no record than a plausible wrong one."""
        xml = _pica_xml([("111", "10.9999/other", "")])
        self.assertIsNone(self._fetch(xml, self._DOI))

    def test_isbn_mismatch_is_rejected(self):
        xml = _pica_xml([("333", "", "9999999999999")])
        rec = self._fetch(xml, "978-1-350-06741-7", kind="isbn")
        self.assertIsNone(rec)  # 9999… does not match the requested ISBN

    def test_isbn_verified_match_is_returned(self):
        xml = _pica_xml([("333", "", "9781350067417")])
        rec = self._fetch(xml, "978-1-350-06741-7", kind="isbn")
        self.assertEqual(rec.ppn, "333")

    def test_ppn_exact_match(self):
        xml = _pica_xml([("1963652258", self._DOI, "9781350067417")])
        rec = self._fetch(xml, "1963652258", kind="ppn")
        self.assertEqual(rec.doi, self._DOI)

    def test_network_failure_returns_none(self):
        with patch("requests.get", side_effect=OSError("down")):
            self.assertIsNone(fetch_record_for_identifier("1963652258", kind="ppn"))


class TestResolveIdentifierTool(unittest.TestCase):
    def test_success_shape_carries_all_identifiers(self):
        from src.utils.lookups.k10plus import K10PlusLookup

        record = K10PlusRecord(
            ppn="1963652258", doi="10.5040/9781350067417", isbn="9781350067417",
            title="A Cultural History", subjects=["Race"],
        )
        with patch(
            "src.utils.k10plus_resolver.fetch_record_for_identifier", return_value=record
        ):
            out = K10PlusLookup().resolve_identifier("10.5040/9781350067417")
        self.assertTrue(out["success"])
        self.assertEqual(out["kind"], "doi")
        self.assertEqual(
            out["identifiers"],
            {"ppn": "1963652258", "doi": "10.5040/9781350067417", "isbn": "9781350067417"},
        )
        self.assertEqual(out["record"]["title"], "A Cultural History")

    def test_miss_reports_kind_and_identifier(self):
        from src.utils.lookups.k10plus import K10PlusLookup

        with patch(
            "src.utils.k10plus_resolver.fetch_record_for_identifier", return_value=None
        ):
            out = K10PlusLookup().resolve_identifier("1963652258")
        self.assertFalse(out["success"])
        self.assertEqual(out["kind"], "ppn")
        self.assertIn("1963652258", out["error"])

    def test_tool_spec_registered(self):
        from src.utils.lookups.k10plus import K10PlusLookup

        names = [s.name for s in K10PlusLookup.mcp_tool_specs()]
        self.assertIn("k10plus_resolve", names)
        spec = next(s for s in K10PlusLookup.mcp_tool_specs() if s.name == "k10plus_resolve")
        self.assertEqual(spec.method, "resolve_identifier")
        self.assertEqual(spec.cache_key_param, "identifier")


class TestMarcXmlPpnIndex(unittest.TestCase):
    def test_k10plus_ppn_query_uses_pica_ppn(self):
        from src.utils.clients.marcxml_client import MarcXmlClient

        client = MarcXmlClient(preset="k10plus")
        self.assertEqual(
            client._build_cql_query("1963652258", "ppn"), 'pica.ppn="1963652258"'
        )


if __name__ == "__main__":
    unittest.main()
