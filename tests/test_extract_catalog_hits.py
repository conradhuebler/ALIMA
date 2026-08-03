"""Claude Generated - Tests for extract_catalog_hits_from_tool_log and its
finc/catalog_titles parsing helpers (title_list_search workflow support).

Covers the root-cause fix for a real, observed bug: an LLM-transcribed
catalog_hits JSON could drop entries for a long wishlist, so the same title
showed a catalog hit in one run and "no hits" in another. These functions
parse the search step's own tool-call results directly (agent_loop.py's
tool_log ``result_full``), bypassing the LLM's manual transcription entirely.

Pure-function tests: no LLM, no network, no registry ceremony needed.
"""

from __future__ import annotations

import json
import unittest

from src.core.agents.deterministic_functions import (
    _extract_catalog_titles_hits_from_result,
    _extract_finc_hits_from_result,
    _flatten_finc_authors,
    extract_catalog_hits_from_tool_log,
)


def _finc_tool_log_entry(terms_to_records: dict) -> dict:
    """Build a realistic search_finc tool_log entry (verified envelope shape:
    tool_registry._make_finc_handler)."""
    results = {
        term: {"records": records, "result_count": len(records), "facets": {}, "errors": []}
        for term, records in terms_to_records.items()
    }
    return {
        "iteration": 1,
        "tool": "search_finc",
        "arguments": {"terms": list(terms_to_records.keys()), "search_type": "title", "limit": 5},
        "result_preview": "...",
        "result_full": json.dumps({"source": "finc", "results": results, "errors": {}}),
        "duration_s": 1.2,
    }


def _catalog_titles_tool_log_entry(terms_to_records: dict) -> dict:
    """Build a realistic search_catalog_titles tool_log entry (verified
    envelope shape: tool_registry._make_title_records_handler)."""
    return {
        "iteration": 2,
        "tool": "search_catalog_titles",
        "arguments": {"terms": list(terms_to_records.keys()), "search_type": "title", "max_results": 5},
        "result_preview": "...",
        "result_full": json.dumps({"source": "catalog_titles", "results": terms_to_records}),
        "duration_s": 2.3,
    }


_FINC_RECORD = {
    "id": "0-1878699474",
    "title": "Cadmium Toxicity Mitigation",
    "authors": {
        "primary": {}, "primary_orig": {}, "corporate": {}, "corporate_orig": {},
        "corporate_secondary": {}, "corporate_secondary_orig": {},
        "secondary": {"Jha, Amrit Kumar": ["edt"], "Kumar, Nitish": ["edt"]},
        "secondary_orig": {"Jha, Amrit Kumar": [], "Kumar, Nitish": []},
    },
    "subjects": [["Environmental chemistry."]],
    "formats": ["eBook"],
    "languages": ["English"],
    "series": [],
    "urls": [{"url": "https://doi.org/10.1007/978-3-031-47390-6"}],
    "web_url": "https://katalog.ub.tu-freiberg.de/Record/0-1878699474",
    "resource_url": "https://doi.org/10.1007/978-3-031-47390-6",
    "year": "2024",
    "publisher": "Springer",
    "edition": "1st ed. 2024.",
    "isbn": "3031473906",
}

_CATALOG_TITLES_RECORD = {
    "rsn": "1846124905",
    "web_url": "https://katalog.ub.tu-freiberg.de/Record/0-1846124905",
    "title": "Quantenchemie",
    "authors": ["Atkins, Peter"],
    "isbn": "9783527123456",
    "publication": "Weinheim : Wiley-VCH, 2021",
    "year": "2021",
    # WP-D2: canonical classifications dict (formerly dk_codes/rvk_codes/ddc_codes)
    "classifications": {"DK": [{"code": "DK 54", "origin": "authority"}]},
    "subjects": [],
    "mab_subjects": [],
}


class TestFlattenFincAuthors(unittest.TestCase):
    def test_flattens_primary_and_secondary(self):
        authors = {
            "primary": {"Atkins, Peter": ["aut"]},
            "secondary": {"Jha, Amrit Kumar": ["edt"]},
        }
        self.assertEqual(
            set(_flatten_finc_authors(authors)), {"Atkins, Peter", "Jha, Amrit Kumar"}
        )

    def test_non_dict_returns_empty(self):
        self.assertEqual(_flatten_finc_authors(None), [])
        self.assertEqual(_flatten_finc_authors([]), [])


class TestExtractFincHitsFromResult(unittest.TestCase):
    def test_extracts_full_record_shape(self):
        raw = json.dumps({
            "source": "finc",
            "results": {"Cadmium Toxicity Mitigation": {"records": [_FINC_RECORD], "result_count": 1}},
        })
        hits = _extract_finc_hits_from_result(raw)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit["query"], "Cadmium Toxicity Mitigation")
        self.assertEqual(hit["source"], "finc")
        self.assertEqual(hit["rsn"], "0-1878699474")
        self.assertEqual(hit["web_url"], "https://katalog.ub.tu-freiberg.de/Record/0-1878699474")
        self.assertEqual(hit["resource_url"], "https://doi.org/10.1007/978-3-031-47390-6")
        self.assertEqual(hit["year"], "2024")
        self.assertEqual(hit["publisher"], "Springer")
        self.assertEqual(hit["edition"], "1st ed. 2024.")
        self.assertEqual(hit["isbn"], "3031473906")
        self.assertIn("Jha, Amrit Kumar", hit["authors"])
        self.assertEqual(hit["formats"], ["eBook"])

    def test_multiple_terms_and_records(self):
        raw = json.dumps({
            "source": "finc",
            "results": {
                "A": {"records": [_FINC_RECORD]},
                "B": {"records": [dict(_FINC_RECORD, id="0-2", title="Other")]},
            },
        })
        hits = _extract_finc_hits_from_result(raw)
        self.assertEqual(len(hits), 2)
        self.assertEqual({h["query"] for h in hits}, {"A", "B"})

    def test_error_envelope_returns_empty(self):
        raw = json.dumps({"source": "finc", "error": "finc not configured"})
        self.assertEqual(_extract_finc_hits_from_result(raw), [])

    def test_zero_hits_term_returns_empty(self):
        raw = json.dumps({"source": "finc", "results": {"Nothing Found": {"records": []}}})
        self.assertEqual(_extract_finc_hits_from_result(raw), [])

    def test_malformed_json_returns_empty(self):
        self.assertEqual(_extract_finc_hits_from_result("not json at all"), [])
        self.assertEqual(_extract_finc_hits_from_result(""), [])
        self.assertEqual(_extract_finc_hits_from_result(None), [])


class TestExtractCatalogTitlesHitsFromResult(unittest.TestCase):
    def test_extracts_full_record_shape(self):
        raw = json.dumps({
            "source": "catalog_titles",
            "results": {"Quantenchemie": [_CATALOG_TITLES_RECORD]},
        })
        hits = _extract_catalog_titles_hits_from_result(raw)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit["query"], "Quantenchemie")
        self.assertEqual(hit["source"], "catalog")
        self.assertEqual(hit["rsn"], "1846124905")
        self.assertEqual(hit["web_url"], "https://katalog.ub.tu-freiberg.de/Record/0-1846124905")
        self.assertEqual(hit["resource_url"], "")  # Libero has no full-text concept
        self.assertEqual(hit["edition"], "")  # no separate edition field
        self.assertEqual(hit["publisher"], "Weinheim : Wiley-VCH, 2021")  # "publication" field
        self.assertEqual(hit["authors"], ["Atkins, Peter"])
        self.assertEqual(hit["isbn"], "9783527123456")

    def test_error_envelope_returns_empty(self):
        raw = json.dumps({"error": "BiblioSuggester not available"})
        self.assertEqual(_extract_catalog_titles_hits_from_result(raw), [])

    def test_malformed_json_returns_empty(self):
        self.assertEqual(_extract_catalog_titles_hits_from_result("{broken"), [])


class TestExtractCatalogHitsFromToolLog(unittest.TestCase):
    def test_combines_finc_and_catalog_titles_hits(self):
        tool_log = [
            _finc_tool_log_entry({"Cadmium Toxicity Mitigation": [_FINC_RECORD]}),
            _catalog_titles_tool_log_entry({"Quantenchemie": [_CATALOG_TITLES_RECORD]}),
        ]
        result = extract_catalog_hits_from_tool_log(tool_log)
        hits = result["hits"]
        self.assertEqual(len(hits), 2)
        sources = {h["source"] for h in hits}
        self.assertEqual(sources, {"finc", "catalog"})

    def test_ignores_unrelated_tool_calls(self):
        tool_log = [
            {"tool": "search_gnd", "result_full": json.dumps({"entries": []})},
            _finc_tool_log_entry({"A": [_FINC_RECORD]}),
        ]
        result = extract_catalog_hits_from_tool_log(tool_log)
        self.assertEqual(len(result["hits"]), 1)

    def test_empty_tool_log(self):
        self.assertEqual(extract_catalog_hits_from_tool_log([]), {"hits": []})
        self.assertEqual(extract_catalog_hits_from_tool_log(None), {"hits": []})

    def test_falls_back_to_result_preview_when_result_full_missing(self):
        # Defensive: older tool_log entries (pre this fix) only have
        # result_preview — should still attempt to parse it rather than
        # silently produce nothing, since a small result may fit in 500 chars. - Claude Generated
        entry = _finc_tool_log_entry({"A": [_FINC_RECORD]})
        full = entry.pop("result_full")
        entry["result_preview"] = full  # short enough to have fit in the 500-char cap
        result = extract_catalog_hits_from_tool_log([entry])
        self.assertEqual(len(result["hits"]), 1)

    def test_streams_summary_notice(self):
        tool_log = [_finc_tool_log_entry({"A": [_FINC_RECORD]})]
        messages = []
        extract_catalog_hits_from_tool_log(tool_log, stream_callback=messages.append)
        self.assertTrue(any("1 catalog_hits" in m for m in messages))

    def test_multiple_calls_to_same_tool_accumulate(self):
        # e.g. the agent calls search_finc once, then search_catalog_titles
        # separately for the misses — both contribute hits. - Claude Generated
        tool_log = [
            _finc_tool_log_entry({"A": [_FINC_RECORD]}),
            _finc_tool_log_entry({"B": [dict(_FINC_RECORD, id="0-99")]}),
        ]
        result = extract_catalog_hits_from_tool_log(tool_log)
        self.assertEqual(len(result["hits"]), 2)


class TestAggregateCatalogClassificationEntries(unittest.TestCase):
    """WP-D2: one generalized loop over all systems replaces three copy-pasted
    per-system loops. Pins the fixed asymmetry: previously only DK got
    frequency aggregation while RVK/DDC were hardcoded count=1 (DDC candidates
    died at any frequency threshold > 1)."""

    def _record(self, title, cls):
        return {"title": title, "classifications": cls}

    def test_counts_aggregate_per_system_and_code(self):
        from src.core.agents.tool_providers import (
            aggregate_catalog_classification_entries,
        )

        ddc = {"DDC": [{"code": "631.4", "origin": "authority"}]}
        results = {
            "Boden": [self._record("T1", ddc), self._record("T2", ddc)],
            "Cadmium": [self._record("T3", ddc)],
        }
        entries = aggregate_catalog_classification_entries(results)
        self.assertEqual(len(entries), 3)  # one row per record occurrence
        # the fixed asymmetry: DDC gets the cross-record aggregate, not 1
        self.assertTrue(all(e["count"] == 3 for e in entries))
        self.assertTrue(all(e["classification_type"] == "DDC" for e in entries))
        self.assertTrue(all(e["dk"] == "631.4" for e in entries))

    def test_all_systems_are_peers(self):
        from src.core.agents.tool_providers import (
            aggregate_catalog_classification_entries,
        )

        results = {"Q": [self._record("T", {
            "DK": [{"code": "54", "origin": "authority"}],
            "RVK": [{"code": "AR 12000", "origin": "authority"}],
            "DDC": [{"code": "631.4", "origin": "authority"}],
        })]}
        entries = aggregate_catalog_classification_entries(results)
        self.assertEqual(
            {e["classification_type"] for e in entries}, {"DK", "RVK", "DDC"}
        )
        for e in entries:
            self.assertEqual(e["keyword"], "Q")
            self.assertEqual(e["source"], "catalog")
            self.assertEqual(e["count"], 1)

    def test_malformed_rows_are_skipped(self):
        from src.core.agents.tool_providers import (
            aggregate_catalog_classification_entries,
        )

        results = {"Q": "kein-list", "R": [None, {"title": "ohne cls"}]}
        self.assertEqual(aggregate_catalog_classification_entries(results), [])


if __name__ == "__main__":
    unittest.main()
