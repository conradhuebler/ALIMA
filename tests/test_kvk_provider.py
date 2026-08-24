"""KVK provider — parsing, identifier extraction, query building, search contract.

The fixture ``fixtures/kvk_raw_huebler.ndjson`` is a real KVK response (query
``ALL=conrad hübler``, six catalogs, two items each) rather than a hand-written
one: the catalog-dependent field usage is exactly what a tidy fixture would
smooth over — the DNB fills ``author``/``year``, K10plus leaves both empty and
puts the imprint into ``text``.

Claude Generated.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import requests

from src.core.bib_record import to_bibrecord
from src.core.search.provider import SearchCapability
from src.core.search.providers.kvk.client import (
    DEFAULT_CATALOGS,
    build_params,
    extract_identifiers,
    merge_round_robin,
    parse_item,
    parse_response,
)
from src.core.search.providers.kvk.provider import KvkProvider

FIXTURE = Path(__file__).parent / "fixtures" / "kvk_raw_huebler.ndjson"


def _payload() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _fake_session(payload: str, status: int = 200):
    """A requests-like stub returning one canned response."""
    response = MagicMock()
    response.status_code = status
    response.text = payload
    response.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = response
    return session


class TestParseResponse(unittest.TestCase):
    def setUp(self):
        self.records, self.errors, self.stats = parse_response(_payload())

    def test_records_from_every_catalog_with_items(self):
        catalogs = {r["catalog"] for r in self.records}
        self.assertEqual(len(self.records), 10)
        self.assertIn("Deutsche Nationalbibliothek", catalogs)
        self.assertIn("K10plus - Verbundkatalog von GBV und SWB", catalogs)
        # The catalog that returned nothing contributes no records.
        self.assertNotIn("HBZ, NRW-Verbundkatalog via GVI", catalogs)

    def test_empty_catalog_is_an_error_entry_not_a_failed_source(self):
        names = {e["catalog"] for e in self.errors}
        self.assertIn("HBZ, NRW-Verbundkatalog via GVI", names)
        self.assertTrue(all(e["message"] for e in self.errors))

    def test_stats_carry_total_hits_and_truncation(self):
        by_catalog = {s["catalog"]: s for s in self.stats}
        k10 = by_catalog["K10plus - Verbundkatalog von GBV und SWB"]
        self.assertEqual(k10["results"], 18)
        self.assertEqual(k10["returned"], 2)
        # 18 hits, 2 delivered → the KVK paged it.
        self.assertTrue(k10["truncated"])
        self.assertFalse(by_catalog["KOBV Berlin-Brandenburg"]["truncated"])

    def test_broken_line_is_skipped_not_fatal(self):
        payload = "{not json\n" + _payload()
        records, _errors, _stats = parse_response(payload)
        self.assertEqual(len(records), 10)

    def test_empty_payload_is_empty_result(self):
        self.assertEqual(parse_response(""), ([], [], []))


class TestParseItem(unittest.TestCase):
    def test_named_fields_win_when_the_catalog_fills_them(self):
        rec = parse_item(
            {"title": "T", "author": "Hübler, Conrad [Verfasser]", "year": "2026",
             "text": "", "url": ""},
            catalog="DNB",
        )
        self.assertEqual(rec["author"], "Hübler, Conrad [Verfasser]")
        self.assertEqual(rec["year"], "2026")

    def test_imprint_fallback_when_they_are_empty(self):
        rec = parse_item({
            "title": "Maschinelles Lernen",
            "author": "", "year": "",
            "text": "Quintes, Florian. - Freiburg im Breisgau, 06.07.2026",
            "url": "",
        })
        self.assertEqual(rec["author"], "Quintes, Florian")
        self.assertEqual(rec["year"], "2026")

    def test_imprint_without_an_author_is_not_read_as_one(self):
        rec = parse_item({
            "title": "T", "author": "", "year": "",
            "text": "Bognor Regis : Wiley-VCH, 2026", "url": "",
        })
        self.assertEqual(rec["author"], "")
        self.assertEqual(rec["year"], "2026")

    def test_year_is_the_last_one_in_the_imprint(self):
        # "1. Auflage" of a 1998 work reissued 2026 — the imprint year is last.
        rec = parse_item({
            "title": "T", "author": "", "year": "",
            "text": "Liu, Shubin. - Nachdruck der Ausgabe 1998. - Weinheim, 2026",
            "url": "",
        })
        self.assertEqual(rec["year"], "2026")


class TestExtractIdentifiers(unittest.TestCase):
    def test_k10plus_docid_is_a_ppn(self):
        self.assertEqual(
            extract_identifiers("https://swbkvk.bsz-bw.de/DB=2.299/SET=1/SHW?FRST=1&bibtip_docid=1981371435"),
            {"ppn": "1981371435"},
        )

    def test_dnb_docid_is_an_idn(self):
        self.assertEqual(
            extract_identifiers("https://portal.dnb.de/opac.htm?method=showFullRecord&bibtip_docid=1415663890"),
            {"idn": "1415663890"},
        )

    def test_stabikat_record_path(self):
        self.assertEqual(
            extract_identifiers("https://stabikat.de/Record/366303287"), {"ppn": "366303287"}
        )

    def test_kobv_gbv_prefix_is_stripped(self):
        self.assertEqual(
            extract_identifiers("https://portal.kobv.de/KobvIndexRecord/gbv_537048642"),
            {"ppn": "537048642"},
        )

    def test_bvb_gives_a_bv_number_not_a_ppn(self):
        self.assertEqual(
            extract_identifiers("https://www.gateway-bayern.de/BV044038433"),
            {"bvnumber": "BV044038433"},
        )

    def test_kobv_alma_id_is_not_read_as_a_ppn(self):
        """KOBV mixes sources: `gbv_…` is a K10plus PPN, `almahu_…` is an Alma
        MMS id. Handing the second one to k10plus_resolve would look like a
        lookup miss instead of a wrong id."""
        self.assertEqual(
            extract_identifiers("https://portal.kobv.de/KobvIndexRecord/almahu_9949983928502882"),
            {},
        )

    def test_hbz_record_link_yields_no_ppn(self):
        self.assertEqual(
            extract_identifiers("https://nrw.digibib.net/search/hbzvk/record/99377271374706441"),
            {},
        )

    def test_unknown_link_yields_nothing(self):
        self.assertEqual(extract_identifiers("https://example.org/x"), {})
        self.assertEqual(extract_identifiers(""), {})

    def test_fixture_records_carry_the_ids_their_links_expose(self):
        records, _e, _s = parse_response(_payload())
        ided = [r for r in records if r.get("ppn") or r.get("idn") or r.get("bvnumber")]
        # Every catalog in this fixture links a readable id — which is NOT true
        # in general (hbz and the Alma-backed KOBV rows expose none).
        self.assertEqual(len(ided), len(records))


class TestMergeRoundRobin(unittest.TestCase):
    RECORDS = [
        {"catalog": "A", "title": "a1"}, {"catalog": "A", "title": "a2"},
        {"catalog": "A", "title": "a3"}, {"catalog": "B", "title": "b1"},
        {"catalog": "C", "title": "c1"}, {"catalog": "C", "title": "c2"},
    ]

    def test_takes_turns_between_catalogs(self):
        out = merge_round_robin(self.RECORDS, 3)
        self.assertEqual([r["title"] for r in out], ["a1", "b1", "c1"])

    def test_keeps_each_catalogs_own_order(self):
        out = merge_round_robin(self.RECORDS, 5)
        self.assertEqual([r["title"] for r in out], ["a1", "b1", "c1", "a2", "c2"])

    def test_exhausted_catalogs_do_not_stall_the_rest(self):
        out = merge_round_robin(self.RECORDS, 99)
        self.assertEqual(len(out), len(self.RECORDS))
        self.assertEqual(out[-1]["title"], "a3")

    def test_zero_limit_is_empty(self):
        self.assertEqual(merge_round_robin(self.RECORDS, 0), [])


class TestBuildParams(unittest.TestCase):
    def test_axis_mapping(self):
        self.assertIn(("TI", "quantenchemie"), build_params("quantenchemie", search_type="title"))
        self.assertIn(("ST", "quantenchemie"), build_params("quantenchemie", search_type="subject"))
        self.assertIn(("SB", "97835"), build_params("97835", search_type="isbn"))
        self.assertIn(("ALL", "x"), build_params("x"))

    def test_catalogs_are_repeated_pairs_not_a_single_value(self):
        params = build_params("x", catalogs=["K10PLUS", "BVB"])
        self.assertEqual([v for k, v in params if k == "kataloge"], ["K10PLUS", "BVB"])

    def test_default_catalog_set_is_used_when_none_given(self):
        params = build_params("x")
        self.assertEqual([v for k, v in params if k == "kataloge"], list(DEFAULT_CATALOGS))

    def test_json_mask_is_always_requested(self):
        self.assertIn(("maske", "kvk-json"), build_params("x"))

    def test_unknown_axis_raises_instead_of_searching_something_else(self):
        with self.assertRaises(ValueError):
            build_params("x", search_type="publisher")


class TestProviderSearch(unittest.TestCase):
    def _provider(self, session):
        provider = KvkProvider(catalogs="K10PLUS, BVB", timeout=5)
        provider._client = None
        prov_client = provider.client
        prov_client._session = session
        return provider

    def test_returns_records_per_term_with_metadata(self):
        provider = self._provider(_fake_session(_payload()))
        res = provider.search(SearchCapability.TITLE_RECORDS, ["conrad hübler"])
        items = res.per_term["conrad hübler"]
        self.assertEqual(len(items), 10)
        self.assertTrue(all(it.record for it in items))
        meta = res.per_term_meta["conrad hübler"]
        self.assertEqual(meta["result_count"], 68)  # sum over the six catalogs
        self.assertEqual(len(meta["catalog_errors"]), 3)
        self.assertEqual(res.errors, {})

    def test_max_results_caps_the_merged_list(self):
        provider = self._provider(_fake_session(_payload()))
        res = provider.search(SearchCapability.TITLE_RECORDS, ["x"], max_results=3)
        self.assertEqual(len(res.per_term["x"]), 3)
        self.assertEqual(res.per_term_meta["x"]["returned"], 3)

    def test_a_small_cap_still_reaches_several_catalogs(self):
        """A meta-search that answers a 3-hit request from one catalog has
        thrown away the only thing it is better at than a single catalog."""
        provider = self._provider(_fake_session(_payload()))
        res = provider.search(SearchCapability.TITLE_RECORDS, ["x"], max_results=3)
        catalogs = {it.record["catalog"] for it in res.per_term["x"]}
        self.assertEqual(len(catalogs), 3)

    def test_transport_failure_is_an_error_not_an_empty_result(self):
        session = MagicMock()
        session.get.side_effect = requests.ConnectionError("boom")
        provider = self._provider(session)
        res = provider.search(SearchCapability.TITLE_RECORDS, ["x"])
        self.assertEqual(res.per_term["x"], [])
        self.assertIn("boom", res.errors["x"])

    def test_unsupported_capability_raises(self):
        provider = KvkProvider()
        with self.assertRaises(ValueError):
            provider.search(SearchCapability.GND_KEYWORDS, ["x"])

    def test_available_without_credentials(self):
        self.assertTrue(KvkProvider().is_available())

    def test_search_type_reaches_the_request(self):
        session = _fake_session(_payload())
        provider = self._provider(session)
        provider.search(SearchCapability.TITLE_RECORDS, ["x"], search_type="subject")
        params = session.get.call_args.kwargs["params"]
        self.assertIn(("ST", "x"), params)
        self.assertEqual(session.get.call_args.kwargs["timeout"], 5)


class TestToBibRecord(unittest.TestCase):
    def test_kvk_record_normalises(self):
        records, _e, _s = parse_response(_payload())
        k10 = next(r for r in records if r["catalog"].startswith("K10plus"))
        bib = to_bibrecord(k10, "kvk")
        self.assertEqual(bib.source, "kvk")
        self.assertTrue(bib.title)
        self.assertTrue(bib.identifiers.get("ppn"))
        self.assertEqual(bib.url, k10["url"])
        # The KVK carries neither, and none are invented.
        self.assertEqual(bib.subjects, [])
        self.assertEqual(bib.classifications, {})

    def test_dnb_record_keeps_its_idn(self):
        records, _e, _s = parse_response(_payload())
        dnb = next(r for r in records if r["catalog"] == "Deutsche Nationalbibliothek")
        bib = to_bibrecord(dnb, "kvk")
        self.assertTrue(bib.identifiers.get("idn"))
        self.assertNotIn("ppn", bib.identifiers)
        self.assertTrue(bib.authors)


if __name__ == "__main__":
    unittest.main()
