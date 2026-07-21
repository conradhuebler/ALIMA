"""Authority DDC: fill the local GND store and read it back - Claude Generated (WP-D2).

The authority path was wired but dead in three independent places, each of which
alone was enough to yield nothing:

1. ``gnd_entries`` was empty, because
2. its only two writers called ``update_gnd_entry`` — a method that does not
   exist on UnifiedKnowledgeManager; the AttributeError was swallowed by the
   surrounding ``except``, so the DNB sync reported an error and moved on; and
3. the ``gnd_local`` provider tested ``isinstance(ddcs, (list, set, tuple))`` on
   a column stored as TEXT, so even a populated store produced ``set()``.

These tests cover the repaired chain end to end: DNB result → stored column →
parsed entries → provider output.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.core.gnd_authority_enrichment import (
    DEFAULT_MAX_LOOKUPS,
    enrich_gnd_entries,
    missing_gnd_ids,
)
from src.utils.classification_systems import (
    ORIGIN_AUTHORITY,
    format_stored_ddcs,
    parse_stored_ddcs,
)

DNB_OK = {
    "status": "success",
    "preferred_name": "Limnologie",
    "ddc": [
        {"code": "551.48", "determinancy": "1"},   # DNB's own misspelling
        {"code": "577.6", "determinancy": "3"},
    ],
    "gnd_subject_categories": ["23.2"],
}


class _FakeUkm:
    def __init__(self, known=()):
        self.known = {k: object() for k in known}
        self.stored = {}

    def get_gnd_facts_batch(self, gnd_ids):
        return {k: v for k, v in self.known.items() if k in gnd_ids}

    def store_gnd_fact(self, gnd_id, data):
        self.stored[gnd_id] = data


class TestStoredColumnFormat(unittest.TestCase):
    def test_round_trip_keeps_code_and_determinacy(self):
        text = format_stored_ddcs(
            [{"code": "551.48", "determinacy": 1}, {"code": "577.6", "determinacy": 3}]
        )
        self.assertEqual(text, "551.48(1);577.6(3)")
        entries = parse_stored_ddcs(text)
        self.assertEqual(
            [(e["code"], e["determinacy"]) for e in entries],
            [("551.48", 1), ("577.6", 3)],
        )

    def test_parsed_entries_are_authority_without_a_count(self):
        entry = parse_stored_ddcs("551.48(1)")[0]
        self.assertEqual(entry["origin"], ORIGIN_AUTHORITY)
        self.assertNotIn("count", entry)

    def test_more_determinate_code_sorts_first(self):
        """Degree 1 is definitive, 4 is loosely related — order must show it."""
        entries = parse_stored_ddcs("577.6(4);551.48(1);600(2)")
        self.assertEqual([e["code"] for e in entries], ["551.48", "600", "577.6"])

    def test_pipe_separated_legacy_codes_are_split(self):
        """The 141k existing rows use pipes, not semicolons.

        The first parser split on ";" only — derived from reading the writer in
        find_keywords.py, which never ran (it called a non-existent method). The
        stored data therefore looked like ONE code
        "333.91|363.7394|628.1|551.48".
        """
        entries = parse_stored_ddcs("623.4516|358.3|327.1745")
        self.assertEqual(
            sorted(e["code"] for e in entries), ["327.1745", "358.3", "623.4516"]
        )

    def test_single_plain_code_is_the_common_case(self):
        self.assertEqual([e["code"] for e in parse_stored_ddcs("621.381537")],
                         ["621.381537"])

    def test_both_separators_coexist(self):
        """New rows carry determinacy, old ones do not — one column, two shapes."""
        entries = parse_stored_ddcs("551.9(1);577.14(2)|600")
        self.assertEqual({e["code"] for e in entries}, {"551.9", "577.14", "600"})

    def test_codes_without_determinacy_are_kept(self):
        entries = parse_stored_ddcs("551.48;577.6(2)")
        self.assertEqual({e["code"] for e in entries}, {"551.48", "577.6"})

    def test_a_string_is_not_mistaken_for_empty(self):
        """The original bug: a TEXT value failed an isinstance(list/set) test."""
        self.assertTrue(parse_stored_ddcs("551.48(1)"))

    def test_empty_and_junk_are_harmless(self):
        for value in ("", None, "  ", ";;", "()"):
            with self.subTest(value=value):
                self.assertEqual(parse_stored_ddcs(value), [])


class TestEnrichment(unittest.TestCase):
    def test_only_unknown_ids_are_requested(self):
        ukm = _FakeUkm(known=["known-1"])
        self.assertEqual(missing_gnd_ids(ukm, ["known-1", "new-1", "new-1"]), ["new-1"])

    def test_successful_lookup_is_stored_in_column_format(self):
        ukm = _FakeUkm()
        result = enrich_gnd_entries(ukm, ["4035769-7"], classify=lambda g: DNB_OK)

        self.assertEqual(result["stored"], 1)
        stored = ukm.stored["4035769-7"]
        self.assertEqual(stored["title"], "Limnologie")
        self.assertEqual(stored["ddcs"], "551.48(1);577.6(3)")

    def test_request_count_is_bounded(self):
        """A caller passing a whole pool must degrade, not fire 1000 requests."""
        ukm = _FakeUkm()
        calls = []

        def classify(gnd_id):
            calls.append(gnd_id)
            return DNB_OK

        result = enrich_gnd_entries(
            ukm, [f"id-{i}" for i in range(50)], classify=classify, max_lookups=5
        )
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["skipped_over_limit"], 45)
        self.assertLessEqual(DEFAULT_MAX_LOOKUPS, 50)

    def test_a_failing_lookup_does_not_sink_the_others(self):
        ukm = _FakeUkm()

        def classify(gnd_id):
            if gnd_id == "bad":
                raise RuntimeError("network down")
            return DNB_OK

        result = enrich_gnd_entries(ukm, ["bad", "good"], classify=classify)
        self.assertEqual((result["stored"], result["failed"]), (1, 1))
        self.assertIn("good", ukm.stored)

    def test_unsuccessful_dnb_payload_is_not_stored(self):
        ukm = _FakeUkm()
        result = enrich_gnd_entries(
            ukm, ["x"], classify=lambda g: {"status": "error", "error_message": "boom"}
        )
        self.assertEqual((result["stored"], result["failed"]), (0, 1))
        self.assertEqual(ukm.stored, {})

    def test_entry_without_a_title_is_not_stored(self):
        """The store drops title-less rows on read — writing one is a dead row."""
        ukm = _FakeUkm()
        enrich_gnd_entries(
            ukm, ["x"], classify=lambda g: {"status": "success", "preferred_name": ""}
        )
        self.assertEqual(ukm.stored, {})

    def test_disabled_dnb_plugin_is_not_an_error(self):
        ukm = _FakeUkm()
        result = enrich_gnd_entries(ukm, ["x"], classify=None, max_lookups=1)
        # No classifier available → nothing requested, nothing raised.
        self.assertEqual(result["stored"], 0)

    def test_missing_dependency_aborts_once_instead_of_failing_per_id(self):
        """rdflib went undeclared, so the DNB client raised ImportError on every
        call — which as a per-id warning made a broken path look merely empty."""
        ukm = _FakeUkm()
        calls = []

        def classify(gnd_id):
            calls.append(gnd_id)
            raise ImportError("No module named 'rdflib'")

        result = enrich_gnd_entries(ukm, ["a", "b", "c"], classify=classify)
        self.assertEqual(len(calls), 1, "kept retrying a permanent failure")
        self.assertIn("aborted", result)
        self.assertIn("rdflib", result["aborted"])
        self.assertEqual(result["stored"], 0)

    def test_nothing_to_do_is_cheap(self):
        ukm = _FakeUkm(known=["a"])
        calls = []
        result = enrich_gnd_entries(
            ukm, ["a"], classify=lambda g: calls.append(g) or DNB_OK
        )
        self.assertEqual(calls, [])
        self.assertEqual(result["requested"], 0)


class TestProviderReadsTheStore(unittest.TestCase):
    def test_gnd_local_emits_authority_ddc_from_the_text_column(self):
        """The end of the chain: stored TEXT must reach ResultItem entries."""
        from src.core.search.provider import SearchCapability
        from src.core.search.providers.gnd_local.provider import GndLocalProvider

        entry = Mock(gnd_id="4035769-7", title="Limnologie", ddcs="551.48(1);577.6(3)")
        ukm = Mock()
        ukm.search_local_gnd.return_value = [entry]

        provider = GndLocalProvider.__new__(GndLocalProvider)
        provider._ukm = ukm  # `ukm` is a lazy property; seed its backing field
        result = provider.search(SearchCapability.GND_KEYWORDS, ["Limnologie"])

        item = result.per_term["Limnologie"][0]
        self.assertEqual(
            [e["code"] for e in item.classifications["DDC"]], ["551.48", "577.6"]
        )
        self.assertEqual(item.classifications["DDC"][0]["origin"], ORIGIN_AUTHORITY)

    def test_entry_without_ddcs_yields_no_classifications(self):
        from src.core.search.provider import SearchCapability
        from src.core.search.providers.gnd_local.provider import GndLocalProvider

        ukm = Mock()
        ukm.search_local_gnd.return_value = [
            Mock(gnd_id="1", title="Ohne DDC", ddcs=None)
        ]
        provider = GndLocalProvider.__new__(GndLocalProvider)
        provider._ukm = ukm  # `ukm` is a lazy property; seed its backing field
        item = provider.search(SearchCapability.GND_KEYWORDS, ["x"]).per_term["x"][0]
        self.assertEqual(item.classifications, {})


class TestWiringIntoGndBatchSearch(unittest.TestCase):
    """The enrichment has to be REACHED, not merely importable.

    The first attempt read ``tool_registry.knowledge_manager`` — an attribute
    that does not exist (the accessor is ``_get_knowledge_manager``, and
    ``CachingToolRegistry`` forwards only named methods). ``getattr`` returned
    None and the whole block was dead. These tests pin the reach.
    """

    def test_manager_is_found_through_a_caching_wrapper(self):
        from src.core.agents.deterministic_functions import _knowledge_manager_of

        sentinel = object()
        inner = Mock()
        inner._get_knowledge_manager.return_value = sentinel
        wrapper = Mock(spec=["inner_registry"])
        wrapper.inner_registry = inner

        self.assertIs(_knowledge_manager_of(wrapper), sentinel)

    def test_manager_is_found_on_a_bare_registry(self):
        from src.core.agents.deterministic_functions import _knowledge_manager_of

        sentinel = object()
        registry = Mock()
        registry._get_knowledge_manager.return_value = sentinel
        self.assertIs(_knowledge_manager_of(registry), sentinel)

    def test_registry_without_an_accessor_yields_none_not_an_error(self):
        from src.core.agents.deterministic_functions import _knowledge_manager_of

        self.assertIsNone(_knowledge_manager_of(Mock(spec=[])))

    def test_public_attribute_does_not_exist(self):
        """Guards the original mistake: if a public `knowledge_manager` is ever
        added, this test should be revisited rather than the wiring silently
        depending on it."""
        from src.mcp.tool_registry import ToolRegistry

        self.assertFalse(hasattr(ToolRegistry, "knowledge_manager"))
        self.assertTrue(hasattr(ToolRegistry, "_get_knowledge_manager"))


if __name__ == "__main__":
    unittest.main()
