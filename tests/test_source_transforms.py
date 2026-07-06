"""P4.1 parity: swb + catalog transform-on-read reductions - Claude Generated.

Locks the pure ``transform(raw)`` reductions that back the WP2 raw cache to the
same output the live path produces:
* ``SWBSuggester.transform`` reuses ``_extract_subjects_from_page`` (mocked here)
  and assembles the reduced view with page-order dedup;
* ``BiblioClient._reduce_records_to_subjects`` (extracted from ``search_subjects``)
  counts subjects across records, unions ``subjects``+``mab_subjects``, skips
  blanks, and caps at the top 50 by count; ``BiblioSuggester.transform`` delegates.
No network — suggesters/clients are exercised via ``__new__`` / fake ``self``.
"""

import unittest

try:
    from src.core.search.providers.swb.suggester import SWBSuggester
    from src.core.search.providers.catalog.suggester import BiblioSuggester
    from src.utils.clients.biblio_client import BiblioClient
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


class _FakeSwb:
    def __init__(self, per_page):
        self._per_page = per_page

    def _extract_subjects_from_page(self, content):
        return self._per_page[content]


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class SwbTransformTest(unittest.TestCase):
    def test_pages_reduced_with_order_dedup(self):
        fake = _FakeSwb({"p1": {"A": "g1", "B": "g2"}, "p2": {"B": "g2b", "C": "g3"}})
        got = SWBSuggester.transform(fake, {"pages": ["p1", "p2"]})
        self.assertEqual(set(got.keys()), {"A", "B", "C"})
        self.assertEqual(got["A"], {"count": 1, "gndid": {"g1"}, "ddc": set(), "dk": set()})
        self.assertEqual(got["B"]["gndid"], {"g2b"})  # later page wins on key collision

    def test_empty_pages(self):
        self.assertEqual(SWBSuggester.transform(_FakeSwb({}), {"pages": []}), {})

    def test_subjects_shape_roundtrip(self):
        # The compact {"subjects": …} form extract_gnd_from_swb now writes (small,
        # always under the size cap, carries titles) must round-trip through
        # transform back to the reduced view — no gnd_entries facts needed.
        results = {"Wasser": {"count": 1, "gndid": {"g1", "g2"}, "ddc": {"540"}, "dk": set()}}
        blob = {
            "subjects": {
                subj: {"count": d["count"], "gndid": sorted(d["gndid"]),
                       "ddc": sorted(d["ddc"]), "dk": sorted(d["dk"])}
                for subj, d in results.items()
            }
        }
        got = SWBSuggester.transform(SWBSuggester.__new__(SWBSuggester), blob)
        self.assertEqual(got, results)


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class CatalogReduceTest(unittest.TestCase):
    def _client(self):
        return BiblioClient.__new__(BiblioClient)  # skip network __init__

    def test_counts_union_and_blanks(self):
        items = [
            {"subjects": ["Klima", "Wasser"], "mab_subjects": []},
            {"subjects": ["Klima"], "mab_subjects": ["Umwelt"]},
            {"subjects": ["   "], "mab_subjects": []},  # blank → skipped
        ]
        got = self._client()._reduce_records_to_subjects(items)
        self.assertEqual(got["Klima"]["count"], 2)
        self.assertEqual(got["Wasser"]["count"], 1)
        self.assertEqual(got["Umwelt"]["count"], 1)
        self.assertNotIn("", got)
        self.assertEqual(got["Klima"]["gndid"], set())  # filled later by SWB validation

    def test_top_50_cap(self):
        items = [{"subjects": [f"S{i}"], "mab_subjects": []} for i in range(60)]
        got = self._client()._reduce_records_to_subjects(items)
        self.assertEqual(len(got), 50)

    def test_biblio_suggester_transform_delegates(self):
        obj = BiblioSuggester.__new__(BiblioSuggester)
        obj.extractor = BiblioClient.__new__(BiblioClient)
        got = obj.transform({"records": [{"subjects": ["X"], "mab_subjects": []}]})
        self.assertEqual(got["X"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
