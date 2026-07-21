"""`get_all_gnd_ids_for_keyword` — the union over the mapping cache - Claude Generated.

This method existed as a CALL before it existed as a method: catalog subjects
without a GND id go through ``pipeline_utils._validate_catalog_subjects``, which
asked the knowledge manager for cached GND ids via a name that was never
implemented. The call sits outside any try/except, so it raised AttributeError
on every catalog subject lacking a GND id — a whole branch of the classic search
that could not run.

Two more of that same class were found in one scan (the other was a cosmetic
cache annotation in `alima search`, removed rather than repaired). The lesson is
in the tests here: a cross-source union is a real query with real edge cases, not
a compatibility shim.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import DatabaseConfig


class TestAllGndIdsForKeyword(unittest.TestCase):
    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "knowledge.db")
        self.ukm = UnifiedKnowledgeManager(
            database_config=DatabaseConfig(db_type="sqlite", sqlite_path=self.db)
        )

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        self._tmp.cleanup()

    def _map(self, term, suggester, gnd_ids, normalized=None):
        self.ukm.update_search_mapping(
            term, suggester, found_gnd_ids=list(gnd_ids)
        )
        if normalized is not None:
            con = sqlite3.connect(self.db)
            con.execute(
                "UPDATE search_mappings SET normalized_term = ? "
                "WHERE search_term = ? AND suggester_type = ?",
                [normalized, term, suggester],
            )
            con.commit()
            con.close()

    def test_union_across_sources(self):
        """Per-source lookups would each miss what the other found."""
        self._map("Cadmium", "lobid", ["4007249-3"])
        self._map("Cadmium", "swb", ["4007249-3", "4128128-7"])

        found = self.ukm.get_all_gnd_ids_for_keyword("Cadmium")
        self.assertEqual(sorted(found), ["4007249-3", "4128128-7"])

    def test_ids_are_deduplicated(self):
        self._map("Wasser", "lobid", ["4064784-5"])
        self._map("Wasser", "swb", ["4064784-5"])
        self.assertEqual(self.ukm.get_all_gnd_ids_for_keyword("Wasser"), ["4064784-5"])

    def test_normalized_term_also_matches(self):
        """A catalog subject spelled differently must still hit the cache."""
        self._map("Limnologie", "lobid", ["4035769-7"], normalized="limnologie")
        self.assertEqual(
            self.ukm.get_all_gnd_ids_for_keyword("limnologie"), ["4035769-7"]
        )

    def test_miss_returns_empty_not_none(self):
        """Callers branch on truthiness to decide "look it up live"."""
        self.assertEqual(self.ukm.get_all_gnd_ids_for_keyword("Unbekannt"), [])

    def test_blank_term_is_not_a_query(self):
        for term in ("", None):
            with self.subTest(term=term):
                self.assertEqual(self.ukm.get_all_gnd_ids_for_keyword(term), [])

    def test_corrupt_row_is_skipped_not_fatal(self):
        self._map("Gut", "lobid", ["4000001-1"])
        con = sqlite3.connect(self.db)
        con.execute(
            "INSERT INTO search_mappings "
            "(search_term, normalized_term, suggester_type, found_gnd_ids, "
            " found_classifications, result_count) VALUES (?,?,?,?,?,?)",
            ["Gut", "gut", "kaputt", "{nicht json", "[]", 0],
        )
        con.commit()
        con.close()
        self.assertEqual(self.ukm.get_all_gnd_ids_for_keyword("Gut"), ["4000001-1"])

    def test_empty_id_list_contributes_nothing(self):
        self._map("Leer", "lobid", [])
        self.assertEqual(self.ukm.get_all_gnd_ids_for_keyword("Leer"), [])


class TestRemovedCompatShims(unittest.TestCase):
    """The half-implemented CacheManager facade is gone.

    Callers assumed a CacheManager API that UnifiedKnowledgeManager implemented
    only partly; the missing names failed at runtime. The dead remainder
    (``cache_results``/``load_entrys``/``get_cached_results``/
    ``store_classification_results``, 216 lines, zero callers) was removed so the
    surface no longer suggests an API that is not there.
    """

    def test_dead_compat_methods_are_gone(self):
        for name in (
            "cache_results",
            "load_entrys",
            "get_cached_results",
            "store_classification_results",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(UnifiedKnowledgeManager, name))

    def test_methods_that_are_actually_called_survive(self):
        for name in (
            "get_all_gnd_ids_for_keyword",
            "add_gnd_entry",
            "store_gnd_fact",
            "get_search_mapping",
            "update_search_mapping",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(UnifiedKnowledgeManager, name))


if __name__ == "__main__":
    unittest.main()
