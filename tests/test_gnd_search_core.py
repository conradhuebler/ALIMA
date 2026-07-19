"""Characterization tests for the shared GND-search core - Claude Generated.

Locks the behavior of ``src/core/gnd_search_core.py`` and proves that the two
callers it now backs — the agentic ``gnd_batch_search`` pool logic and the
classic ``SearchCLI.merge_results`` — keep their pre-refactor semantics:
max-count merges, code-set/-list unions, source_count ranking.
"""

from __future__ import annotations

import json
import unittest
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
        source = {"count": 5, "gnd_ids": ["2", "3"], "classifications": {"ddc": ["d"]}}
        merge_code_entry(
            target, source, code_fields=("gnd_ids",),
            classifications_field="classifications",
        )
        self.assertEqual(target["count"], 5)
        self.assertEqual(target["gnd_ids"], ["1", "2", "3"])
        self.assertEqual(target["classifications"]["ddc"], ["d"])
        self.assertIsInstance(target["gnd_ids"], list)

    def test_classifications_merge_per_system(self):
        """WP-D1: {system: codes} merges per system — union, order-preserving,
        systems are equal-rank keys; source dict is never mutated."""
        target = {"count": 1, "classifications": {"dk": ["530.145"]}}
        source = {"count": 2, "classifications": {"dk": ["530.145", "539"], "rvk": ["UK 1000"]}}
        merge_code_entry(
            target, source, code_fields=(), classifications_field="classifications"
        )
        self.assertEqual(target["classifications"]["dk"], ["530.145", "539"])
        self.assertEqual(target["classifications"]["rvk"], ["UK 1000"])
        self.assertEqual(source["classifications"], {"dk": ["530.145", "539"], "rvk": ["UK 1000"]})

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
                                           "classifications": {"ddc": ["d"]}, "count": 9}})
        self.assertEqual(pool["cadmium"]["count"], 9)
        self.assertEqual(pool["cadmium"]["gnd_ids"], ["1", "2"])
        self.assertEqual(pool["cadmium"]["classifications"]["ddc"], ["d"])

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
                "Cadmium": {"gnd_ids": ["1"], "classifications": {"ddc": ["546"]}, "count": 5},
                "Schwermetall": {"gnd_ids": ["2"], "classifications": {}, "count": 9},
            }
        }
    }

    def test_parse_from_json_string_and_dict_equivalent(self):
        from_dict = parse_batch_response(self.PAYLOAD)
        from_str = parse_batch_response(json.dumps(self.PAYLOAD))
        self.assertEqual(from_dict, from_str)
        self.assertEqual(from_dict["Cadmium"]["gnd_id"], "1")
        self.assertEqual(from_dict["Cadmium"]["classifications"], {"ddc": ["546"]})
        self.assertNotIn("dk", from_dict["Cadmium"]["classifications"])
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
                                         "classifications": {"ddc": {"546"}}}}}
        new = {"term": {"Cadmium": {"count": 7, "gnd_ids": {"2"},
                                    "classifications": {"dk": {"a"}}}}}
        cli.merge_results(combined, new)
        entry = combined["term"]["Cadmium"]
        self.assertEqual(entry["count"], 7)
        self.assertEqual(entry["gnd_ids"], {"1", "2"})
        self.assertEqual(entry["classifications"], {"ddc": {"546"}, "dk": {"a"}})

    def test_new_term_and_keyword_copied(self):
        cli = self._cli()
        combined = {}
        new = {"term": {"Neu": {"count": 1, "gnd_ids": {"9"}, "classifications": {}}}}
        cli.merge_results(combined, new)
        self.assertEqual(combined["term"]["Neu"]["gnd_ids"], {"9"})


if __name__ == "__main__":
    unittest.main()
