"""Tests for the capability-based search-provider plugin system (P1) - Claude Generated.

Covers: self-registration + declared capabilities, the legacy-shape round-trip
converters, every provider's ``search`` against a *fake* backend (no network), and
the registry guards. Faithfulness (P1 is facade-preserving) is asserted by
round-tripping each suggester's real output shape through its provider adapter.
"""

import types
import unittest

from src.core.search import (
    PROVIDER_REGISTRY,
    ProviderResult,
    SearchCapability,
    get_provider,
    list_providers,
    providers_for_capability,
    register_provider,
)


class _Sig:
    """Minimal stand-in for a Qt signal (connect is a no-op)."""

    def connect(self, cb):  # noqa: D401
        pass


class _FakeGndSuggester:
    """Fake BaseSuggester returning a canned GND-keyword dict."""

    def __init__(self, out, errors=None):
        self._out = out
        self.last_errors = errors or {}
        self.currentTerm = _Sig()
        self.recorded_kwargs = None

    def search(self, terms, **kw):
        self.recorded_kwargs = kw
        return self._out


class RegistryTest(unittest.TestCase):
    def test_builtin_providers_registered(self):
        self.assertEqual(
            set(list_providers()), {"lobid", "swb", "catalog", "finc", "gnd_local"}
        )

    def test_capability_index_matches_spec(self):
        self.assertEqual(
            set(providers_for_capability(SearchCapability.GND_KEYWORDS)),
            {"lobid", "swb", "catalog", "gnd_local"},
        )
        self.assertEqual(
            set(providers_for_capability(SearchCapability.TITLE_RECORDS)),
            {"catalog", "finc"},
        )
        self.assertEqual(
            set(providers_for_capability(SearchCapability.SUBJECT_FACETS)), {"finc"}
        )
        self.assertEqual(
            set(providers_for_capability(SearchCapability.CLASSIFICATION)), {"catalog"}
        )

    def test_get_provider_unknown_raises(self):
        with self.assertRaises(KeyError):
            get_provider("does_not_exist")

    def test_duplicate_id_guard(self):
        try:
            @register_provider
            class _Dup:
                id = "lobid"  # collides with built-in
                label = "x"
                capabilities = set()

            self.fail("expected ValueError for duplicate id")
        except ValueError:
            pass

    def test_register_requires_id(self):
        with self.assertRaises(ValueError):
            @register_provider
            class _NoId:
                id = ""
                label = "x"
                capabilities = set()


class GndKeywordRoundTripTest(unittest.TestCase):
    SAMPLE = {
        "wasser": {
            "Wassermanagement": {
                "count": 47,
                "gndid": {"gnd1", "gnd2"},
                "ddc": {"333.7"},
                "dk": set(),
            },
            "Wasserwirtschaft": {
                "count": 3,
                "gndid": {"gnd3"},
                "ddc": set(),
                "dk": {"AR 1000"},
            },
        }
    }

    def test_lossless_roundtrip(self):
        pr = ProviderResult.from_gnd_keywords(self.SAMPLE, errors={"wasser": "boom"})
        self.assertEqual(pr.capability, SearchCapability.GND_KEYWORDS)
        self.assertEqual(pr.errors, {"wasser": "boom"})
        self.assertEqual(pr.to_gnd_keywords(), self.SAMPLE)

    def test_display_count_is_additive(self):
        # Without display_count the legacy shape is unchanged...
        pr = ProviderResult.from_gnd_keywords(self.SAMPLE)
        self.assertNotIn("display_count", pr.to_gnd_keywords()["wasser"]["Wassermanagement"])
        # ...but when set it is emitted (F-4 plumbing).
        pr.per_term["wasser"][0].display_count = 99
        self.assertEqual(
            pr.to_gnd_keywords()["wasser"]["Wassermanagement"]["display_count"], 99
        )


class GndProviderSearchTest(unittest.TestCase):
    OUT = {"x": {"Kw": {"count": 5, "gndid": {"g1"}, "ddc": set(), "dk": set()}}}

    def _provider_with_fake(self, pid, out, errors=None):
        prov = get_provider(pid)()
        fake = _FakeGndSuggester(out, errors=errors)
        prov._suggester = fake
        return prov, fake

    def test_lobid_search_wraps_output(self):
        prov, fake = self._provider_with_fake("lobid", self.OUT, errors={"x": "err"})
        res = prov.search(SearchCapability.GND_KEYWORDS, ["x"], search_type="kw")
        self.assertEqual(res.to_gnd_keywords(), self.OUT)
        self.assertEqual(res.errors, {"x": "err"})
        self.assertEqual(fake.recorded_kwargs, {"search_type": "kw"})

    def test_swb_passes_max_pages(self):
        prov, fake = self._provider_with_fake("swb", self.OUT)
        prov.search(SearchCapability.GND_KEYWORDS, ["x"], max_pages=2)
        self.assertEqual(fake.recorded_kwargs, {"search_type": "kw", "max_pages": 2})

    def test_capability_guard(self):
        prov, _ = self._provider_with_fake("lobid", self.OUT)
        with self.assertRaises(ValueError):
            prov.search(SearchCapability.TITLE_RECORDS, ["x"])


class CatalogProviderTest(unittest.TestCase):
    def test_title_records(self):
        prov = get_provider("catalog")()
        recs = {"q": [{"title": "Buch A", "id": "1"}, {"title": "Buch B", "id": "2"}]}
        prov._suggester = types.SimpleNamespace(
            currentTerm=_Sig(),
            search_titles=lambda terms, search_type="title", max_results=25: recs,
        )
        res = prov.search(SearchCapability.TITLE_RECORDS, ["q"])
        self.assertEqual(res.capability, SearchCapability.TITLE_RECORDS)
        self.assertEqual([i.record for i in res.per_term["q"]], recs["q"])
        self.assertEqual(res.per_term_meta["q"]["result_count"], 2)
        self.assertEqual([i.label for i in res.per_term["q"]], ["Buch A", "Buch B"])

    def test_classification_buckets_by_keyword(self):
        prov = get_provider("catalog")()
        dk_list = [
            {"dk": "666.76", "count": 4, "matched_keywords": ["Halbleiter"]},
            {"dk": "537", "count": 1},  # no matched_keywords -> _all bucket
        ]
        prov._suggester = types.SimpleNamespace(
            currentTerm=_Sig(),
            extract_dk_classifications=lambda kws: dk_list,
        )
        res = prov.search(SearchCapability.CLASSIFICATION, ["Halbleiter"])
        self.assertEqual(res.per_term["Halbleiter"][0].code, "666.76")
        self.assertEqual(res.per_term["Halbleiter"][0].count, 4)
        self.assertEqual(res.per_term["_all"][0].code, "537")


class FincProviderTest(unittest.TestCase):
    FINC = {
        "physik": {
            "records": [{"title": "A", "id": "1"}],
            "result_count": 12,
            "facets": {"udk_raw_de105": [{"value": "53", "count": 7, "translated": "Physik"}]},
            "errors": [],
        }
    }

    def _finc_provider(self):
        prov = get_provider("finc")()
        prov._suggester = types.SimpleNamespace(
            currentTerm=_Sig(),
            last_errors={},
            search=lambda terms, search_type="kw", filters=None, limit=None, facets=None: self.FINC,
        )
        return prov

    def test_title_records_roundtrip(self):
        res = self._finc_provider().search(SearchCapability.TITLE_RECORDS, ["physik"])
        self.assertEqual(res.capability, SearchCapability.TITLE_RECORDS)
        self.assertEqual(res.to_finc_records(), self.FINC)

    def test_subject_facets(self):
        res = self._finc_provider().search(SearchCapability.SUBJECT_FACETS, ["physik"])
        self.assertEqual(res.capability, SearchCapability.SUBJECT_FACETS)
        item = res.per_term["physik"][0]
        self.assertEqual((item.code, item.count, item.label), ("53", 7, "Physik"))
        self.assertEqual(item.extra["facet"], "udk_raw_de105")

    def test_availability_gated_on_base_url(self):
        self.assertFalse(get_provider("finc")().is_available())
        self.assertTrue(get_provider("finc")(base_url="http://finc").is_available())


class GndLocalProviderTest(unittest.TestCase):
    def test_search_local(self):
        prov = get_provider("gnd_local")()
        entry = types.SimpleNamespace(
            gnd_id="g1", title="Halbleiter", ddcs=["537"]
        )
        prov._ukm = types.SimpleNamespace(
            search_local_gnd=lambda term, min_results=3: [entry]
        )
        res = prov.search(SearchCapability.GND_KEYWORDS, ["halbleiter"])
        item = res.per_term["halbleiter"][0]
        self.assertEqual(item.label, "Halbleiter")
        self.assertEqual(item.gnd_ids, {"g1"})
        self.assertEqual(item.ddc, {"537"})
        self.assertEqual(item.count, 0)


if __name__ == "__main__":
    unittest.main()
