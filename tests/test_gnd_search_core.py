"""Characterization tests for the shared GND-search core - Claude Generated.

Locks the behavior of ``src/core/gnd_search_core.py`` and proves that the two
callers it now backs — the agentic ``gnd_batch_search`` pool logic and the
classic ``SearchCLI.merge_results`` — keep their pre-refactor semantics:
max-count merges, code-set/-list unions, source_count ranking.
"""

from __future__ import annotations

import json
import unittest

from src.utils.classification_systems import codes_for_system
from unittest.mock import MagicMock

from src.core.gnd_search_core import (
    merge_code_entry,
    merge_into_pool,
    parse_batch_response,
    parse_batch_response_with_terms,
    rank_pool,
)


class TestMergeCodeEntry(unittest.TestCase):
    def test_set_fields_updated_in_place(self):
        """Nested shape: code fields are sets, merged via union, count via max."""
        target = {"count": 3, "gnd_ids": {"1"}}
        source = {"count": 7, "gnd_ids": {"1", "2"}}
        merge_code_entry(target, source, code_fields=("gnd_ids",))
        self.assertEqual(target["count"], 7)
        self.assertEqual(target["gnd_ids"], {"1", "2"})
        self.assertIsInstance(target["gnd_ids"], set)

    def test_list_fields_order_preserving_dedup(self):
        """Agentic shape: code fields are lists — dedup but keep first-seen order."""
        target = {"count": 1, "gnd_ids": ["1", "2"], "classifications": {}}
        source = {"count": 5, "gnd_ids": ["2", "3"], "classifications": {"DDC": ["d"]}}
        merge_code_entry(
            target, source, code_fields=("gnd_ids",),
            classifications_field="classifications",
        )
        self.assertEqual(target["count"], 5)
        self.assertEqual(target["gnd_ids"], ["1", "2", "3"])
        self.assertEqual(codes_for_system(target["classifications"], "DDC"), ["d"])
        self.assertIsInstance(target["gnd_ids"], list)

    def test_classifications_merge_per_system(self):
        """WP-D1: {system: codes} merges per system — union, order-preserving,
        systems are equal-rank keys; source dict is never mutated."""
        target = {"count": 1, "classifications": {"DK": ["530.145"]}}
        source = {"count": 2, "classifications": {"DK": ["530.145", "539"], "RVK": ["UK 1000"]}}
        merge_code_entry(
            target, source, code_fields=(), classifications_field="classifications"
        )
        self.assertEqual(
            codes_for_system(target["classifications"], "DK"), ["530.145", "539"]
        )
        self.assertEqual(codes_for_system(target["classifications"], "RVK"), ["UK 1000"])
        # The source must be untouched: pool inserts are shallow copies, so an
        # in-place merge would leak across entries.
        self.assertEqual(
            source["classifications"], {"DK": ["530.145", "539"], "RVK": ["UK 1000"]}
        )

    def test_missing_fields_are_noops(self):
        target = {"count": 2, "gnd_ids": {"1"}}
        merge_code_entry(target, {}, code_fields=("gnd_ids",),
                         classifications_field="classifications")
        self.assertEqual(target["count"], 2)
        self.assertEqual(target["gnd_ids"], {"1"})
        self.assertNotIn("classifications", target)


class TestMergeIntoPool(unittest.TestCase):
    def test_new_title_inserted_lowercased_key(self):
        pool = {}
        merge_into_pool(pool, {"Cadmium": {"title": "Cadmium", "gnd_ids": ["1"], "count": 3}})
        self.assertIn("cadmium", pool)
        self.assertEqual(pool["cadmium"]["count"], 3)

    def test_existing_title_unions_and_maxes(self):
        pool = {"cadmium": {"title": "Cadmium", "gnd_ids": ["1"],
                            "classifications": {}, "count": 3, "gnd_id": "1"}}
        merge_into_pool(pool, {"Cadmium": {"title": "Cadmium", "gnd_ids": ["2"],
                                           "classifications": {"DDC": ["d"]}, "count": 9}})
        self.assertEqual(pool["cadmium"]["count"], 9)
        self.assertEqual(pool["cadmium"]["gnd_ids"], ["1", "2"])
        self.assertEqual(codes_for_system(pool["cadmium"]["classifications"], "DDC"), ["d"])

    def test_gnd_id_backfilled_when_empty(self):
        pool = {"x": {"title": "X", "gnd_ids": [], "classifications": {},
                      "count": 0, "gnd_id": ""}}
        merge_into_pool(pool, {"X": {"title": "X", "gnd_ids": ["77"],
                                     "classifications": {}, "count": 1}})
        self.assertEqual(pool["x"]["gnd_id"], "77")


class TestParseBatchResponse(unittest.TestCase):
    PAYLOAD = {
        "results": {
            "Cadmium": {
                "Cadmium": {"gnd_ids": ["1"], "classifications": {"DDC": ["546"]}, "count": 5},
                "Schwermetall": {"gnd_ids": ["2"], "classifications": {}, "count": 9},
            }
        }
    }

    def test_parse_from_json_string_and_dict_equivalent(self):
        from_dict = parse_batch_response(self.PAYLOAD)
        from_str = parse_batch_response(json.dumps(self.PAYLOAD))
        self.assertEqual(from_dict, from_str)
        self.assertEqual(from_dict["Cadmium"]["gnd_id"], "1")
        self.assertEqual(
            codes_for_system(from_dict["Cadmium"]["classifications"], "DDC"), ["546"]
        )
        self.assertNotIn("DK", from_dict["Cadmium"]["classifications"])
        self.assertEqual(from_dict["Schwermetall"]["count"], 9)

    def test_entries_without_ids_or_title_skipped(self):
        out = parse_batch_response(
            {"results": {"t": {"": {"gnd_ids": [], "count": 0}}}}
        )
        self.assertEqual(out, {})

    def test_with_terms_tracks_origin(self):
        out, terms = parse_batch_response_with_terms(self.PAYLOAD)
        self.assertEqual(out["Cadmium"]["gnd_id"], "1")
        self.assertEqual(terms["Cadmium"], ["Cadmium"])

    def test_malformed_returns_empty(self):
        self.assertEqual(parse_batch_response("{not json"), {})
        self.assertEqual(parse_batch_response_with_terms("{not json"), ({}, {}))


class TestRankPool(unittest.TestCase):
    def test_source_count_attached_and_ranked(self):
        pool = {
            "a": {"title": "A", "count": 5},
            "b": {"title": "B", "count": 9},
        }
        src_index = {"a": {"swb", "lobid"}, "b": {"swb"}}
        ranked = rank_pool(pool, src_index)
        # A confirmed by 2 sources outranks B despite lower count
        self.assertEqual(ranked[0]["title"], "A")
        self.assertEqual(ranked[0]["source_count"], 2)
        self.assertEqual(ranked[0]["sources"], ["lobid", "swb"])
        self.assertEqual(ranked[1]["source_count"], 1)

    def test_count_tiebreak_within_same_source_count(self):
        pool = {"a": {"title": "A", "count": 3}, "b": {"title": "B", "count": 8}}
        src_index = {"a": {"swb"}, "b": {"swb"}}
        ranked = rank_pool(pool, src_index)
        self.assertEqual([e["title"] for e in ranked], ["B", "A"])


class TestSearchCliMergeEquivalence(unittest.TestCase):
    """Classic SearchCLI.merge_results must behave exactly as before the refactor:
    union of code sets + max count, new keywords copied."""

    def _cli(self):
        from src.core.search_cli import SearchCLI
        return SearchCLI(MagicMock())

    def test_union_and_max_on_existing_keyword(self):
        cli = self._cli()
        combined = {"term": {"Cadmium": {"count": 3, "gnd_ids": {"1"},
                                         "classifications": {"DDC": {"546"}}}}}
        new = {"term": {"Cadmium": {"count": 7, "gnd_ids": {"2"},
                                    "classifications": {"DK": {"a"}}}}}
        cli.merge_results(combined, new)
        entry = combined["term"]["Cadmium"]
        self.assertEqual(entry["count"], 7)
        self.assertEqual(entry["gnd_ids"], {"1", "2"})
        self.assertEqual(codes_for_system(entry["classifications"], "DDC"), ["546"])
        self.assertEqual(codes_for_system(entry["classifications"], "DK"), ["a"])

    def test_new_term_and_keyword_copied(self):
        cli = self._cli()
        combined = {}
        new = {"term": {"Neu": {"count": 1, "gnd_ids": {"9"}, "classifications": {}}}}
        cli.merge_results(combined, new)
        self.assertEqual(combined["term"]["Neu"]["gnd_ids"], {"9"})


class TestMergeAuthorityDdc(unittest.TestCase):
    """The read-back that makes the filled gnd_local store reach the pool.

    The store holds an authority DDC per GND-ID and ``get_gnd_batch`` serves it,
    but both search paths used to fetch it only for description/synonyms and
    discard the DDC — so every pool classification stayed ``cooccurrence`` and a
    subject like "Cadmium" never got its authority DDC. This atom is shared by
    the classic and agentic paths.
    """

    def _entry(self, gnd_ids, classifications=None):
        from src.core.gnd_search_core import merge_authority_ddc
        e = {"title": "X", "gnd_ids": list(gnd_ids),
             "classifications": classifications or {}}
        merge_authority_ddc([e], self.ddcs)
        return e

    def setUp(self):
        # store DDC column format: pipe-separated codes, optional (determinacy)
        self.ddcs = {"g1": "546.48(1)|669", "g2": "540", "g3": ""}

    def test_authority_ddc_is_attached_by_gnd_id(self):
        cls = self._entry(["g1"])["classifications"]
        self.assertEqual(codes_for_system(cls, "DDC"), ["546.48", "669"])
        self.assertTrue(all(e["origin"] == "authority" for e in cls["DDC"]))

    def test_cooccurrence_on_other_systems_is_untouched(self):
        """Merging DDC must not disturb an existing RVK co-occurrence."""
        cls = self._entry(["g1"], {
            "RVK": [{"code": "VN 9360", "count": 2, "origin": "cooccurrence"}],
        })["classifications"]
        self.assertEqual(codes_for_system(cls, "RVK"), ["VN 9360"])
        self.assertEqual(cls["RVK"][0]["origin"], "cooccurrence")
        self.assertEqual(codes_for_system(cls, "DDC"), ["546.48", "669"])

    def test_authority_outranks_cooccurrence_on_the_same_code(self):
        cls = self._entry(["g2"], {
            "DDC": [{"code": "540", "count": 5, "origin": "cooccurrence"}],
        })["classifications"]
        self.assertEqual(codes_for_system(cls, "DDC"), ["540"])
        self.assertEqual(cls["DDC"][0]["origin"], "authority")   # authority wins

    def test_ddc_merges_from_every_gnd_id(self):
        """A subject's merged spellings can each carry a DDC."""
        cls = self._entry(["g1", "g2"])["classifications"]
        self.assertEqual(set(codes_for_system(cls, "DDC")), {"546.48", "669", "540"})

    def test_determinacy_sorts_authority_entries(self):
        cls = self._entry(["g1"])["classifications"]
        # 546.48 has determinacy 1 (definitive) → before 669 (no determinacy)
        self.assertEqual(cls["DDC"][0]["code"], "546.48")

    def test_missing_or_blank_store_ddc_is_a_noop(self):
        self.assertEqual(self._entry(["g3"])["classifications"], {})
        self.assertEqual(self._entry(["unknown"])["classifications"], {})

    def test_returns_count_of_entries_that_gained_a_ddc(self):
        from src.core.gnd_search_core import merge_authority_ddc
        entries = [{"gnd_ids": ["g1"], "classifications": {}},
                   {"gnd_ids": ["g3"], "classifications": {}},   # blank → no gain
                   {"gnd_ids": ["g2"], "classifications": {}}]
        self.assertEqual(merge_authority_ddc(entries, self.ddcs), 2)


class TestMergeRecordGndSubjects(unittest.TestCase):
    """WP-D1 P3: the input record's GND-linked subjects join the pool as
    verified candidates in their own bucket, count=1 (count landmine)."""

    def _subjects(self):
        return [
            {"term": "Limnologie", "gnd_id": "4074296-3"},
            {"term": "Alpen", "gnd_id": "4001328-5"},
        ]

    def test_injects_own_bucket_with_canonical_entries(self):
        from src.core.gnd_search_core import merge_record_gnd_subjects

        search = {"Seenkunde": {"Seenkunde": {"count": 4, "gnd_ids": {"x"}, "classifications": {}}}}
        injected = merge_record_gnd_subjects(search, self._subjects())
        self.assertEqual(injected, 2)
        bucket = search["input_record"]
        self.assertEqual(
            bucket["Limnologie"],
            {"count": 1, "gnd_ids": {"4074296-3"}, "classifications": {}},
        )
        # existing search buckets untouched
        self.assertEqual(search["Seenkunde"]["Seenkunde"]["count"], 4)

    def test_same_term_twice_merges_ids(self):
        from src.core.gnd_search_core import merge_record_gnd_subjects

        search = {}
        merge_record_gnd_subjects(search, [
            {"term": "Limnologie", "gnd_id": "4074296-3"},
            {"term": "Limnologie", "gnd_id": "9999999-9"},
        ])
        self.assertEqual(
            search["input_record"]["Limnologie"]["gnd_ids"],
            {"4074296-3", "9999999-9"},
        )
        self.assertEqual(search["input_record"]["Limnologie"]["count"], 1)

    def test_malformed_and_empty_entries_are_skipped(self):
        from src.core.gnd_search_core import merge_record_gnd_subjects

        search = {}
        injected = merge_record_gnd_subjects(search, [
            "nur ein String", {"term": "", "gnd_id": "1"}, {"term": "X", "gnd_id": ""}, None,
        ])
        self.assertEqual(injected, 0)
        self.assertNotIn("input_record", search)

    def test_none_and_empty_lists_are_noops(self):
        from src.core.gnd_search_core import merge_record_gnd_subjects

        search = {"a": {}}
        self.assertEqual(merge_record_gnd_subjects(search, None), 0)
        self.assertEqual(merge_record_gnd_subjects(search, []), 0)
        self.assertEqual(search, {"a": {}})


if __name__ == "__main__":
    unittest.main()
