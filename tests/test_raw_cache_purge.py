"""Test the pre-v2 swb raw-cache purge against a real SQLite DB - Claude Generated.

This migration shipped without a test and was DEAD: it read the count via
``rows[0][0]``, but ``DatabaseManager.fetch_all`` returns ``List[Dict]`` keyed by
column name, so it raised ``KeyError: 0`` on every run — swallowed by the
guard, logged as a routine warning, purging nothing. Only inspecting the
operator's live DB after a pipeline run revealed the rows were still there.

So the test drives the REAL method against a REAL sqlite file. The stub in
``test_db_schema_fallback.py`` deliberately no-ops every migration, which is
right for what that file tests (DDL dialect) and useless for this one.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import DatabaseConfig

def _cfg(path: str) -> DatabaseConfig:
    """Keyword arg is mandatory: the first positional is the deprecated
    ``db_path`` string, and passing a config there silently falls back to the
    PRODUCTION database. - Claude Generated"""
    return DatabaseConfig(db_type="sqlite", sqlite_path=path)


PRE_V2_ROW = json.dumps({
    "subjects": {"Limnologie": {"count": 3, "gndid": ["4035769-7"], "ddc": ["551.48"]}},
    "totalItems": 1,
})
V2_ROW = json.dumps({
    "subjects": {
        "Limnologie": {
            "count": 3,
            "gnd_ids": ["4035769-7"],
            "classifications": {"DDC": ["551.48"]},
        }
    },
    "totalItems": 1,
})


class TestPreV2SwbRawCachePurge(unittest.TestCase):
    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "knowledge.db")
        # Let the manager create the schema, then seed and re-open so the purge
        # runs over rows that already exist (the real-world sequence).
        UnifiedKnowledgeManager(database_config=_cfg(self.db_path))
        UnifiedKnowledgeManager.reset()

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        self._tmp.cleanup()

    def _seed(self, rows):
        con = sqlite3.connect(self.db_path)
        con.executemany(
            "INSERT INTO search_response_cache "
            "(source, query, normalized_query, params_hash, raw_json) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        con.commit()
        con.close()

    def _sources(self):
        con = sqlite3.connect(self.db_path)
        try:
            return [
                (src, raw)
                for src, raw in con.execute(
                    "SELECT source, raw_json FROM search_response_cache"
                )
            ]
        finally:
            con.close()

    def _reopen(self):
        UnifiedKnowledgeManager(database_config=_cfg(self.db_path))

    def test_pre_v2_swb_rows_are_dropped(self):
        self._seed([("swb", "limnologie", "limnologie", "h1", PRE_V2_ROW)])
        self._reopen()
        self.assertEqual(self._sources(), [], "pre-v2 swb row survived the purge")

    def test_v2_swb_rows_are_kept(self):
        """The purge must not take the rows it exists to preserve."""
        self._seed([("swb", "limnologie", "limnologie", "h1", V2_ROW)])
        self._reopen()
        self.assertEqual(len(self._sources()), 1)

    def test_other_sources_are_untouched(self):
        """Only swb stores this shape; a marker match elsewhere is not ours."""
        self._seed([("lobid", "limnologie", "limnologie", "h2", PRE_V2_ROW)])
        self._reopen()
        self.assertEqual(len(self._sources()), 1, "purge reached beyond swb")

    def test_mixed_rows_leave_only_the_v2_one(self):
        self._seed([
            ("swb", "a", "a", "h1", PRE_V2_ROW),
            ("swb", "b", "b", "h2", V2_ROW),
            ("swb", "c", "c", "h3", PRE_V2_ROW),
        ])
        self._reopen()
        remaining = self._sources()
        self.assertEqual(len(remaining), 1)
        self.assertIn("gnd_ids", remaining[0][1])

    def test_purge_is_idempotent(self):
        self._seed([("swb", "a", "a", "h1", PRE_V2_ROW), ("swb", "b", "b", "h2", V2_ROW)])
        self._reopen()
        UnifiedKnowledgeManager.reset()
        self._reopen()  # second startup must be a cheap no-op, not an error
        self.assertEqual(len(self._sources()), 1)


if __name__ == "__main__":
    unittest.main()
