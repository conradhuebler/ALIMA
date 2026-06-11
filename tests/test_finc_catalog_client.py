#!/usr/bin/env python3
"""Claude Generated - Tests for FincCatalogClient (finc-backed DK/RVK extractor).

Covers the two-step model (Subject search -> per-title udk_raw/rvk facet),
the BiblioClient-compatible keyword-centric return shape, DK normalization,
RVK skip/upper-casing, cache hit short-circuit, and error surfacing.

A gated live benchmark (RUN_INTEGRATION_TESTS=1) measures the real two-step
flow against the operator's finc endpoint.
"""

import os
import time
import unittest
from unittest.mock import MagicMock

from src.utils.clients.finc_catalog_client import FincCatalogClient


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class _FakeKM:
    """Minimal UnifiedKnowledgeManager stand-in capturing cache + extract calls."""

    def __init__(self, cache=None):
        self._cache = cache or {}
        self.stored = []          # (kw, titles, status)
        self.extract_calls = []   # (title_list, matched_keywords)

    def get_catalog_dk_cache(self, kw):
        return self._cache.get(kw)

    def store_catalog_dk_cache(self, kw, titles, status="success", error_message=None, ttl_minutes=None):
        self.stored.append((kw, titles, status))
        return True

    def extract_classifications_from_titles(self, titles, matched_keywords=None):
        self.extract_calls.append((list(titles), list(matched_keywords or [])))
        grouped = {}
        for t in titles:
            for cls in t["classifications"]:
                typ, code = cls.split(None, 1)
                entry = grouped.setdefault(cls, {"dk": code, "type": typ, "titles": [], "count": 0})
                if t["title"] not in entry["titles"]:
                    entry["titles"].append(t["title"])
                entry["count"] += 1
        return list(grouped.values())


_SUBJECT_RECORDS = {
    "Quantenmechanik": [
        {"id": "id-A", "title": "QM Buch A"},
        {"id": "id-B", "title": "QM Buch B"},
    ],
}
_TITLE_FACETS = {
    "id-A": {
        "udk_raw_de105": [{"value": "dk 530.145", "count": 1}],
        "rvk_facet": [
            {"value": "uc 100", "count": 1},
            {"value": "no subject assigned", "count": 1},  # must be skipped
        ],
    },
    "id-B": {
        "udk_raw_de105": [],  # RVK-only title
        "rvk_facet": [{"value": "uk 1000", "count": 1}],
    },
}


def _finc_mock(subject_records=None, status="OK", error=None):
    """A FincClient-like mock dispatching Subject vs per-title `id:` queries."""
    records = subject_records if subject_records is not None else _SUBJECT_RECORDS
    m = MagicMock()

    def _search(lookfor, type="AllFields", filters=None, limit=None, facets=None, **kw):
        if status != "OK":
            return {"status": "ERROR", "resultCount": 0, "records": [], "facets": {}, "error": error}
        if lookfor.startswith('id:"'):
            rid = lookfor[4:-1]
            return {"status": "OK", "resultCount": 1,
                    "records": [{"id": rid, "title": ""}],
                    "facets": _TITLE_FACETS.get(rid, {})}
        recs = records.get(lookfor, [])
        return {"status": "OK", "resultCount": len(recs), "records": recs, "facets": {}}

    m.search.side_effect = _search
    return m


def _make_client(km, finc_mock, **kwargs):
    defaults = dict(base_url="https://dobby.example/proxy.php", use_cache=False, max_workers=4)
    defaults.update(kwargs)
    client = FincCatalogClient(knowledge_manager=km, **defaults)
    # Inject the mock for BOTH the keyword-level client and the per-thread
    # per-title clients, so no real network call is made. - Claude Generated
    client._client = finc_mock
    client._thread_client = lambda: finc_mock
    return client


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestFincCatalogClient(unittest.TestCase):

    def test_keyword_centric_shape_and_classification_strings(self):
        km = _FakeKM()
        client = _make_client(km, _finc_mock())
        out = client.extract_dk_classifications_for_keywords(["Quantenmechanik (GND-ID: 4047989-4)"])

        # One keyword-centric result, GND-suffixed keyword preserved verbatim
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["keyword"], "Quantenmechanik (GND-ID: 4047989-4)")
        self.assertEqual(out[0]["source"], "finc")
        self.assertIn("classifications", out[0])
        self.assertTrue(out[0]["classifications"])

        # The title list funnelled into the KM is the per-title DK/RVK map:
        #  - DK normalized "dk 530.145" -> "DK 530.145"
        #  - RVK upper-cased "uc 100" -> "RVK UC 100"; "no subject assigned" skipped
        #  - RVK-only title (id-B) kept with just its RVK
        self.assertEqual(len(km.extract_calls), 1)
        title_list, matched = km.extract_calls[0]
        self.assertEqual(matched, ["Quantenmechanik"])  # clean keyword (GND suffix stripped)
        by_id = {t["rsn"]: t for t in title_list}
        self.assertEqual(set(by_id["id-A"]["classifications"]), {"DK 530.145", "RVK UC 100"})
        self.assertEqual(by_id["id-B"]["classifications"], ["RVK UK 1000"])

    def test_cache_hit_short_circuits_finc(self):
        cached_titles = [{"rsn": "x", "title": "Cached", "classifications": ["DK 004"]}]
        km = _FakeKM(cache={"Quantenmechanik": (cached_titles, "success", None)})
        finc = _finc_mock()
        client = _make_client(km, finc, use_cache=True)
        out = client.extract_dk_classifications_for_keywords(["Quantenmechanik"])

        self.assertEqual(out[0]["source"], "cache")
        # No finc call of any kind is issued when the cache hits
        self.assertEqual(finc.search.call_count, 0)

    def test_finc_error_is_surfaced_and_keyword_omitted(self):
        km = _FakeKM()
        client = _make_client(km, _finc_mock(status="ERROR", error="Invalid search"), use_cache=True)
        out = client.extract_dk_classifications_for_keywords(["Quantenmechanik"])
        # No result for the failed keyword
        self.assertEqual(out, [])
        # Error recorded in cache (not a silent success)
        self.assertTrue(any(status == "error" for _, _, status in km.stored))

    def test_empty_subject_results_yields_no_keyword(self):
        km = _FakeKM()
        client = _make_client(km, _finc_mock(subject_records={"Quantenmechanik": []}), use_cache=True)
        out = client.extract_dk_classifications_for_keywords(["Quantenmechanik"])
        self.assertEqual(out, [])
        self.assertTrue(any(status == "no_results" for _, _, status in km.stored))

    def test_institution_filter_applied_to_subject_search(self):
        km = _FakeKM()
        finc = _finc_mock()
        client = _make_client(km, finc, institution_filter="DE-105", use_cache=False)
        client.extract_dk_classifications_for_keywords(["Quantenmechanik"])
        subject_call = next(c for c in finc.search.call_args_list
                            if not c.kwargs.get("lookfor", "").startswith('id:"'))
        self.assertEqual(subject_call.kwargs["filters"], {"institution": "DE-105"})
        self.assertEqual(subject_call.kwargs["type"], "Subject")
        # per-title calls must NOT carry the institution filter
        id_call = next(c for c in finc.search.call_args_list
                       if c.kwargs.get("lookfor", "").startswith('id:"'))
        self.assertIsNone(id_call.kwargs.get("filters"))
        self.assertEqual(id_call.kwargs["facets"], ["udk_raw_de105", "rvk_facet"])


# --------------------------------------------------------------------------
# Gated live benchmark
# --------------------------------------------------------------------------

class TestFincCatalogClientLiveBenchmark(unittest.TestCase):
    """Live two-step benchmark against the operator's finc endpoint.

    RUN_INTEGRATION_TESTS=1 and FINC_TEST_BASE_URL=<finc-proxy-url> to enable
    (no institution URL is hard-coded). Prints timing for a realistic keyword
    set; does not assert hard thresholds (operator reviews the numbers).
    """

    KEYWORDS = [
        "Quantenmechanik", "Thermodynamik", "Python", "Algorithmus",
        "Festkörperphysik", "Katalyse", "Molekülphysik", "Statistik",
    ]

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS") == "1",
        "Integration tests disabled (set RUN_INTEGRATION_TESTS=1 to enable)",
    )
    def test_benchmark_two_step_flow(self):
        base_url = os.environ.get("FINC_TEST_BASE_URL")
        if not base_url:
            self.skipTest("set FINC_TEST_BASE_URL to the finc proxy endpoint to run the benchmark")
        from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
        km = UnifiedKnowledgeManager()
        for workers in (1, 8):
            client = FincCatalogClient(
                base_url=base_url, max_workers=workers,
                max_titles_per_keyword=30, use_cache=False, knowledge_manager=km,
            )
            t0 = time.time()
            per = []
            total_titles = 0
            total_cls = 0
            for kw in self.KEYWORDS:
                a = time.time()
                res = client.extract_dk_classifications_for_keywords([kw], max_results=30)
                per.append(time.time() - a)
                if res:
                    cls = res[0]["classifications"]
                    total_cls += len(cls)
                    total_titles += sum(len(c.get("titles", [])) for c in cls)
            dt = time.time() - t0
            print(f"\n[workers={workers}] {len(self.KEYWORDS)} keywords in {dt:.1f}s "
                  f"(avg {dt/len(self.KEYWORDS):.2f}s/kw) "
                  f"-> {total_cls} classifications, {total_titles} title-attributions")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
