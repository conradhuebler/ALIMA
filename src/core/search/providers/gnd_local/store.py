"""Plugin-owned local GND authority store - Claude Generated.

The local copy of GND subject headings (the ``gnd_entries`` table) lives in its
**own** SQLite database (``gnd_local.db``), physically separate from the search
cache (``alima_knowledge.db``). It is filled only by deliberate bulk import +
pipeline enrichment — never by the search cache (WP Phase C1: the former
``warm_gnd_entries`` cross-write is gone).

``UnifiedKnowledgeManager`` keeps its GND-fact API but routes every ``gnd_entries``
query through this store's ``DatabaseManager``, so the authority copy and the cache
never share a database file. The store is always a local SQLite file: an authority
copy is an inherently local artifact and does not follow a MariaDB backend.
"""

from __future__ import annotations

import logging
import os

from src.core.database_manager import DatabaseManager
from src.utils.config_models import DatabaseConfig


def resolve_gnd_local_path(database_config: DatabaseConfig) -> str:
    """Path of the local GND DB - Claude Generated.

    Explicit ``DatabaseConfig.gnd_local_path`` wins; otherwise a sibling of the main
    SQLite DB (so a temp/main path yields an isolated sibling, and the production
    ``~/.config/alima/alima_knowledge.db`` yields ``~/.config/alima/gnd_local.db``).
    """
    explicit = getattr(database_config, "gnd_local_path", "") or ""
    if explicit:
        return explicit
    base_dir = os.path.dirname(getattr(database_config, "sqlite_path", "") or "") or "."
    return os.path.join(base_dir, "gnd_local.db")


class LocalGndStore:
    """Owns the ``gnd_entries`` table in a dedicated SQLite file."""

    def __init__(self, database_config: DatabaseConfig, connection_name: str):
        self.logger = logging.getLogger(__name__)
        self.path = resolve_gnd_local_path(database_config)
        cfg = DatabaseConfig(db_type="sqlite")
        cfg.sqlite_path = self.path
        # Distinct connection_name → per-thread connection names never collide with
        # the main knowledge DB's (see MEMORY.md QSqlDatabase thread-safety note).
        self.db_manager = DatabaseManager(cfg, connection_name)
        self._init_schema()

    def _init_schema(self) -> None:
        self.db_manager.get_connection()
        db_type = self.db_manager.get_db_type()
        dialect = self.db_manager.get_dialect()
        self.db_manager.execute_query(f"""
            CREATE TABLE IF NOT EXISTS gnd_entries (
                gnd_id {dialect.varchar_type(512)} PRIMARY KEY,
                title {dialect.text_type(db_type)} NOT NULL,
                description {dialect.text_type(db_type)},
                synonyms {dialect.text_type(db_type)},
                ddcs {dialect.text_type(db_type)},
                ppn {dialect.text_type(db_type)},
                created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                updated_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.db_manager.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_gnd_title ON gnd_entries(title)"
        )

    def count(self) -> int:
        try:
            return int(self.db_manager.fetch_scalar("SELECT COUNT(*) FROM gnd_entries") or 0)
        except Exception:
            return 0

    def migrate_from_legacy(self, legacy_sqlite_path: str) -> None:
        """One-time bulk copy of a legacy same-file ``gnd_entries`` table into this
        store via SQLite ``ATTACH`` - Claude Generated.

        Idempotent + non-destructive: runs only when this store is empty and the
        legacy SQLite file still carries a ``gnd_entries`` table; the legacy table is
        left in place (a later cleanup can drop it). Skipped for a MariaDB main DB
        (no file to attach) — there the tables were already logically separate.
        """
        if not legacy_sqlite_path or not os.path.exists(legacy_sqlite_path):
            return
        if os.path.abspath(legacy_sqlite_path) == os.path.abspath(self.path):
            return
        if self.count() > 0:
            return
        try:
            escaped = legacy_sqlite_path.replace("'", "''")
            self.db_manager.execute_query(f"ATTACH DATABASE '{escaped}' AS legacy")
            try:
                exists = self.db_manager.fetch_scalar(
                    "SELECT name FROM legacy.sqlite_master "
                    "WHERE type='table' AND name='gnd_entries'"
                )
                if exists:
                    self.db_manager.execute_query(
                        "INSERT OR IGNORE INTO gnd_entries "
                        "(gnd_id, title, description, synonyms, ddcs, ppn, created_at, updated_at) "
                        "SELECT gnd_id, title, description, synonyms, ddcs, ppn, created_at, updated_at "
                        "FROM legacy.gnd_entries"
                    )
                    migrated = self.count()
                    if migrated:
                        self.logger.info(
                            f"✅ Migrated {migrated} GND entries into the plugin-owned "
                            f"local GND DB ({self.path}); legacy table left in place."
                        )
            finally:
                self.db_manager.execute_query("DETACH DATABASE legacy")
        except Exception as e:
            self.logger.warning(f"Local GND migration skipped (non-critical): {e}")

    def close(self) -> None:
        try:
            self.db_manager.close_connection()
        except Exception:
            pass
