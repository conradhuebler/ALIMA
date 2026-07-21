"""The legacy gnd_entries migration must gate on the EFFECTIVE engine - Claude Generated.

The migration into the plugin-owned local GND store was gated on
``database_config.db_type``, i.e. the engine the operator CONFIGURED. When the
configured MySQL/MariaDB driver is not loadable, ``get_connection()`` falls back
to SQLite and everything runs against the file — but the guard still read
"mariadb" and skipped the migration.

Consequence on the operator's machine: 207,291 legacy GND entries (141,661 with
DDC) sat in ``alima_knowledge.db`` while the store the provider reads stayed at
0 rows, dated the day it was created. The authority classifications therefore
looked "empty" rather than "not migrated".

``_init_database`` already forces the connection open before reading db_type for
exactly this reason; the guard is now consistent with it.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import DatabaseConfig


def _seed_legacy(path: str, rows) -> None:
    """A pre-split single-file DB: gnd_entries living in the main SQLite file."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS gnd_entries ("
        "gnd_id TEXT PRIMARY KEY, title TEXT, description TEXT, synonyms TEXT, "
        "ddcs TEXT, ppn TEXT, created_at TEXT, updated_at TEXT)"
    )
    con.executemany(
        "INSERT OR REPLACE INTO gnd_entries "
        "(gnd_id, title, description, synonyms, ddcs, ppn) VALUES (?,?,?,?,?,?)",
        [(g, t, "", "", d, "") for g, t, d in rows],
    )
    con.commit()
    con.close()


class TestMigrationGate(unittest.TestCase):
    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self._tmp = tempfile.TemporaryDirectory()
        self.main = str(Path(self._tmp.name) / "alima_knowledge.db")
        self.local = str(Path(self._tmp.name) / "gnd_local.db")
        _seed_legacy(self.main, [("4035769-7", "Limnologie", "551.48|577.6")])

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        self._tmp.cleanup()

    def _local_rows(self) -> int:
        con = sqlite3.connect(self.local)
        try:
            return con.execute("SELECT COUNT(*) FROM gnd_entries").fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            con.close()

    def test_migrates_when_configured_sqlite(self):
        UnifiedKnowledgeManager(
            database_config=DatabaseConfig(db_type="sqlite", sqlite_path=self.main)
        )
        self.assertEqual(self._local_rows(), 1)

    def test_migrates_when_configured_mariadb_but_fallen_back_to_sqlite(self):
        """The regression: an unreachable MariaDB config used to skip this."""
        ukm = UnifiedKnowledgeManager(
            database_config=DatabaseConfig(
                db_type="mariadb",
                sqlite_path=self.main,
                host="192.0.2.1",  # TEST-NET-1: guaranteed unroutable
                port=3306,
                database="nope",
                username="nope",
                password="nope",
            )
        )
        # Whatever the config said, the engine in use is the file.
        self.assertEqual(str(ukm.db_manager.get_db_type()).lower(), "sqlite")
        self.assertEqual(
            self._local_rows(), 1, "legacy entries were not migrated after fallback"
        )

    def test_second_start_is_a_no_op(self):
        UnifiedKnowledgeManager(
            database_config=DatabaseConfig(db_type="sqlite", sqlite_path=self.main)
        )
        UnifiedKnowledgeManager.reset()
        UnifiedKnowledgeManager(
            database_config=DatabaseConfig(db_type="sqlite", sqlite_path=self.main)
        )
        self.assertEqual(self._local_rows(), 1)


if __name__ == "__main__":
    unittest.main()
