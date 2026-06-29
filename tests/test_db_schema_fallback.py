"""Regression test: chat_mutations DDL must match the ACTUAL db engine.

Claude Generated (P-ι follow-up).

Two pre-existing bugs surfaced while running the headless CLI agent:

1. ``chat_mutations`` hardcoded SQLite ``AUTOINCREMENT`` → crashed on a real
   MariaDB engine.
2. ``_init_database`` read ``db_type`` *before* opening the connection. When
   the MariaDB driver is unavailable, ``get_connection()`` falls back to SQLite
   and rewrites ``config.db_type`` — but the already-captured ``db_type`` still
   said 'mariadb', so the DDL emitted ``AUTO_INCREMENT`` against a SQLite engine.

Fix: open the connection first (resolving any fallback), then read ``db_type``,
and emit the keyword via ``SQLDialect.auto_increment(db_type)``.

This test drives ``_init_database`` with a fake db_manager (no real DB) and
asserts the ``chat_mutations`` DDL carries the keyword matching the *post-
connection* engine.
"""
from __future__ import annotations

import logging
import unittest

from src.core.sql_dialect import SQLDialect
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager


class _FakeDbManager:
    """Records executed DDL. ``get_connection`` optionally flips db_type to
    simulate the MariaDB→SQLite driver fallback."""

    def __init__(self, db_type: str, fallback_to: str | None = None):
        self._db_type = db_type
        self._fallback_to = fallback_to
        self.queries: list[str] = []
        self.connection_opened = False

    def get_connection(self):
        self.connection_opened = True
        if self._fallback_to is not None:
            self._db_type = self._fallback_to  # mirror config rewrite on fallback
        return object()

    def get_db_type(self) -> str:
        return self._db_type

    def get_dialect(self):
        return SQLDialect

    def execute_query(self, sql, params=None):
        self.queries.append(sql)

    def fetch_all(self, query, params=None):
        return []


class _FakeUKM:
    """Minimal stand-in exposing only what _init_database touches."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger("test_ukm")

    def _migrate_catalog_dk_cache_schema(self):
        pass

    def _migrate_search_mappings_schema(self):
        pass

    # Borrow the real method under test.
    _init_database = UnifiedKnowledgeManager._init_database


def _chat_mutations_ddl(queries):
    for q in queries:
        if "CREATE TABLE IF NOT EXISTS chat_mutations" in q:
            return q
    raise AssertionError("chat_mutations DDL not emitted")


class TestChatMutationsDDL(unittest.TestCase):
    def test_sqlite_emits_autoincrement(self):
        db = _FakeDbManager("sqlite")
        _FakeUKM(db)._init_database()
        ddl = _chat_mutations_ddl(db.queries)
        self.assertIn("AUTOINCREMENT", ddl)
        self.assertNotIn("AUTO_INCREMENT", ddl)

    def test_mariadb_emits_auto_increment(self):
        db = _FakeDbManager("mariadb")
        _FakeUKM(db)._init_database()
        ddl = _chat_mutations_ddl(db.queries)
        # MariaDB keyword has the underscore.
        self.assertRegex(ddl, r"\bAUTO_INCREMENT\b")

    def test_connection_opened_before_db_type_read(self):
        db = _FakeDbManager("mariadb")
        _FakeUKM(db)._init_database()
        self.assertTrue(db.connection_opened)

    def test_fallback_to_sqlite_uses_sqlite_keyword(self):
        # Configured mariadb, but the driver is unavailable → get_connection
        # falls back to sqlite. DDL must follow the real (sqlite) engine.
        db = _FakeDbManager("mariadb", fallback_to="sqlite")
        _FakeUKM(db)._init_database()
        ddl = _chat_mutations_ddl(db.queries)
        self.assertIn("AUTOINCREMENT", ddl)
        self.assertNotIn("AUTO_INCREMENT", ddl)


if __name__ == "__main__":
    unittest.main()
