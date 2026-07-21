"""Schema creation and migrations for the unified knowledge DB - Claude Generated.

Split out of ``unified_knowledge_manager.py`` (WP cleanup D). Verbatim mixin
extraction: the methods stay on ``UnifiedKnowledgeManager`` via MRO, so no call
site changes. What is gained is a boundary — the DDL and the one-time
migrations no longer sit in the middle of the query layer, where a broken
migration was mistaken for an empty database twice in July 2026.

⚠️ Migrations here are guarded so a failure cannot block startup. That guard is
also how a completely dead migration goes unnoticed, so: log a failed migration
at ERROR, and cover it with a test that drives the real method against a real
SQLite file (``tests/test_raw_cache_purge.py``,
``tests/test_gnd_local_migration_gate.py``). A stub in the DDL-dialect test is
not coverage.
"""

from __future__ import annotations


class SchemaMigrationMixin:
    """DDL + one-time migrations. Mixed into :class:`UnifiedKnowledgeManager`."""

    def _init_database(self):
        """Initialize unified database schema - Claude Generated"""
        try:
            # Force the connection open BEFORE reading db_type. If the
            # configured MySQL/MariaDB driver is unavailable, get_connection()
            # falls back to SQLite and rewrites self.config.db_type. Reading
            # db_type first would otherwise yield the configured engine while
            # DDL runs against the fallback engine — a mismatch that only the
            # AUTO_INCREMENT/AUTOINCREMENT keyword is sensitive to.
            self.db_manager.get_connection()
            db_type = self.db_manager.get_db_type()
            dialect = self.db_manager.get_dialect()

            # === FACTS TABLES (Immutable truths) ===

            # 1. GND entries (facts) now live in the plugin-owned local GND store
            # (`gnd_local.db`, WP Phase C1c) — created there, not here, so the local
            # authority copy is independent of this search-cache DB. - Claude Generated

            # 2. DK/RVK classifications (facts only, no keywords)
            self.db_manager.execute_query(f"""
                CREATE TABLE IF NOT EXISTS classifications (
                    code {dialect.varchar_type(512)} PRIMARY KEY,
                    type {dialect.varchar_type(16)} NOT NULL,
                    title {dialect.text_type(db_type)},
                    description {dialect.text_type(db_type)},
                    parent_code {dialect.varchar_type(512)},
                    created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # === MAPPING TABLES (Dynamic associations) ===

            # 3. Search mappings (search term → found results)
            # PRIMARY KEY with key_lengths for MySQL/MariaDB (ignored by SQLite)
            pk_search_mappings = dialect.primary_key_def(
                db_type,
                ['search_term', 'suggester_type'],
                key_lengths={'search_term': 380, 'suggester_type': 64}
            )
            self.db_manager.execute_query(f"""
                CREATE TABLE IF NOT EXISTS search_mappings (
                    search_term {dialect.varchar_type(512)} NOT NULL,
                    normalized_term {dialect.varchar_type(512)} NOT NULL,
                    suggester_type {dialect.varchar_type(64)} NOT NULL,
                    found_gnd_ids {dialect.text_type(db_type)},
                    found_classifications {dialect.text_type(db_type)},
                    gnd_counts {dialect.text_type(db_type)},
                    titles {dialect.text_type(db_type)},
                    result_count INTEGER DEFAULT 0,
                    last_updated {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                    created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                    {pk_search_mappings}
                )
            """)

            # 4. Catalog DK cache table (separate from GND/SWB/LOBID searches) - Claude Generated
            self.db_manager.execute_query(f"""
                CREATE TABLE IF NOT EXISTS catalog_dk_cache (
                    search_term {dialect.varchar_type(512)} PRIMARY KEY,
                    normalized_term {dialect.varchar_type(512)} NOT NULL,
                    found_titles {dialect.text_type(db_type)},
                    result_count INTEGER DEFAULT 0,
                    last_updated {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                    created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                    search_status {dialect.varchar_type(32)} DEFAULT 'success',
                    error_message {dialect.text_type(db_type)},
                    retry_after {dialect.timestamp_type(db_type)},
                    consecutive_failures INTEGER DEFAULT 0
                )
            """)

            # 5. Chat mutations audit log (P-ε) — Claude Generated
            # Tri-state `accepted`: TRUE=applied, FALSE=rejected, NULL=pending.
            self.db_manager.execute_query(f"""
                CREATE TABLE IF NOT EXISTS chat_mutations (
                    id INTEGER PRIMARY KEY {dialect.auto_increment(db_type)},
                    session_id {dialect.varchar_type(64)} NOT NULL,
                    tool_name {dialect.varchar_type(128)} NOT NULL,
                    operation {dialect.varchar_type(64)} NOT NULL,
                    payload_json {dialect.text_type(db_type)} NOT NULL,
                    accepted BOOLEAN,
                    reject_reason {dialect.text_type(db_type)},
                    created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                    applied_at {dialect.timestamp_type(db_type)}
                )
            """)

            # 6. Raw source-response cache (WP2 raw-first) — verbatim API/HTML
            # responses keyed by (source, normalized_query, params_hash). Written
            # additively alongside the mapping cache; read on demand for the full
            # agent view and (later) to derive the reduced pool view. - Claude Generated
            pk_response_cache = dialect.primary_key_def(
                db_type,
                ['source', 'normalized_query', 'params_hash'],
                key_lengths={'source': 64, 'normalized_query': 380, 'params_hash': 64}
            )
            self.db_manager.execute_query(f"""
                CREATE TABLE IF NOT EXISTS search_response_cache (
                    source {dialect.varchar_type(64)} NOT NULL,
                    query {dialect.varchar_type(512)} NOT NULL,
                    normalized_query {dialect.varchar_type(512)} NOT NULL,
                    params_hash {dialect.varchar_type(64)} NOT NULL,
                    params_json {dialect.text_type(db_type)},
                    raw_json {dialect.text_type(db_type)},
                    http_status INTEGER,
                    result_count INTEGER DEFAULT 0,
                    byte_size INTEGER DEFAULT 0,
                    last_updated {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                    created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                    {pk_response_cache}
                )
            """)

            # Create indexes for performance
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_search_normalized ON search_mappings(normalized_term)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_search_term ON search_mappings(search_term)")
            # idx_gnd_title lives in the plugin-owned local GND store (WP Phase C1c).
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_classifications_code ON classifications(code)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_classifications_type ON classifications(type)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_catalog_normalized ON catalog_dk_cache(normalized_term)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_catalog_updated ON catalog_dk_cache(last_updated)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_chat_mutations_session ON chat_mutations(session_id)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_chat_mutations_created ON chat_mutations(created_at)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_response_cache_normalized ON search_response_cache(normalized_query)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_response_cache_updated ON search_response_cache(last_updated)")

            self.logger.info(f"Unified knowledge database schema initialized ({db_type})")

            # Perform schema migration if needed - Claude Generated
            self._migrate_catalog_dk_cache_schema()
            self._migrate_search_mappings_schema()
            self._migrate_search_response_cache_schema()
            self._purge_pre_v2_swb_raw_rows()

        except Exception as e:
            self.logger.error(f"Error initializing unified database: {e}")
            raise

    def _migrate_search_mappings_schema(self):
        """Add later columns to existing search_mappings tables.

        Additive, idempotent: fresh DBs already have the columns from CREATE TABLE;
        older DBs get them via ALTER.
        - ``gnd_counts`` (F-4): existing rows keep NULL → display falls back to the
          pool count (1 for cache hits).
        - ``titles`` (WP Phase C1a): denormalized ``{gnd_id: title}`` so cache hits
          resolve titles from the mapping row itself instead of the (now separate)
          local GND store. Existing rows keep NULL → treated as a miss on read.
        - Claude Generated
        """
        try:
            db_type = self.db_manager.get_db_type()
            dialect = self.db_manager.get_dialect()
            query = dialect.get_table_info_query(db_type, 'search_mappings')
            rows = self.db_manager.fetch_all(query)
            columns = dialect.parse_table_info(db_type, rows if rows else [])
            for col in ("gnd_counts", "titles"):
                if columns and col not in columns:
                    self.logger.info(f"🔄 Migrating search_mappings: adding {col} column...")
                    self.db_manager.execute_query(
                        dialect.alter_table_add_column(
                            db_type, 'search_mappings', col,
                            dialect.text_type(db_type)
                        )
                    )
                    self.logger.info(f"✅ search_mappings migration completed: added {col}")
        except Exception as e:
            self.logger.warning(f"search_mappings migration check failed (non-critical): {e}")

    def _migrate_search_response_cache_schema(self):
        """Ensure later columns exist on pre-existing search_response_cache tables.

        Additive, idempotent: fresh DBs already have every column from CREATE
        TABLE, so this is a guarded no-op today — present for forward-compat and
        dialect symmetry with the other cache migrations. - Claude Generated
        """
        try:
            db_type = self.db_manager.get_db_type()
            dialect = self.db_manager.get_dialect()
            query = dialect.get_table_info_query(db_type, 'search_response_cache')
            rows = self.db_manager.fetch_all(query)
            columns = dialect.parse_table_info(db_type, rows if rows else [])
            for col, definition in (("http_status", "INTEGER"),
                                    ("byte_size", "INTEGER DEFAULT 0")):
                if columns and col not in columns:
                    self.logger.info(f"🔄 Migrating search_response_cache: adding {col} column...")
                    self.db_manager.execute_query(
                        dialect.alter_table_add_column(
                            db_type, 'search_response_cache', col, definition
                        )
                    )
        except Exception as e:
            self.logger.warning(f"search_response_cache migration check failed (non-critical): {e}")

    def _purge_pre_v2_swb_raw_rows(self):
        """Drop swb raw-cache rows written in the pre-v2 suggester shape.

        WP-D1 removed the ``gndid``/``ddc``/``dk`` read fallback from
        ``SwbSuggester.transform``. Rows in that shape must therefore be dropped
        rather than left in place: read with v2 keys they would yield an empty
        ``gnd_ids`` set — a search term that silently looks like it has no GND
        IDs, instead of a cache miss that refetches. The raw cache is a cache,
        so deletion only costs one refetch.

        Content migration, not schema, but it runs with the schema migrations
        for the same reason: once, early, before anything reads the table. Cheap
        and idempotent — after the first run the LIKE matches nothing. The
        marker ``"gndid"`` appears only in the pre-v2 payload. - Claude Generated
        """
        try:
            # fetch_all returns List[Dict] keyed by COLUMN NAME, so the count
            # needs an alias — rows[0][0] raises KeyError here. - Claude Generated
            rows = self.db_manager.fetch_all(
                "SELECT COUNT(*) AS stale FROM search_response_cache "
                "WHERE source = 'swb' AND raw_json LIKE '%\"gndid\"%'"
            )
            stale = int(rows[0]["stale"]) if rows else 0
            if not stale:
                return
            self.db_manager.execute_query(
                "DELETE FROM search_response_cache "
                "WHERE source = 'swb' AND raw_json LIKE '%\"gndid\"%'"
            )
            self.logger.info(
                f"🔄 Dropped {stale} pre-v2 swb raw-cache rows (WP-D1 hard cut); "
                f"they will be refetched on next search"
            )
        except Exception as e:
            # Guarded so a migration failure cannot block startup — but log at
            # ERROR: the first version of this failed on the row access above and
            # the warning made a completely dead migration look like routine
            # noise. A skipped purge leaves rows that read as empty results.
            self.logger.error(
                f"pre-v2 swb raw-cache purge FAILED (rows remain, searches may "
                f"return empty GND ids for them): {e}"
            )

    def _migrate_catalog_dk_cache_schema(self):
        """Migrate catalog_dk_cache table - handle schema upgrades - Claude Generated"""
        try:
            db_type = self.db_manager.get_db_type()
            dialect = self.db_manager.get_dialect()

            # Check current schema - use DB-agnostic query
            query = dialect.get_table_info_query(db_type, 'catalog_dk_cache')
            rows = self.db_manager.fetch_all(query)
            columns = dialect.parse_table_info(db_type, rows if rows else [])

            # Step 1: Remove old found_classifications column if exists
            if "found_classifications" in columns:
                self.logger.info("🔄 Migrating catalog_dk_cache: removing unused found_classifications column...")

                if dialect.supports_drop_column(db_type):
                    # MySQL/MariaDB: Direct DROP COLUMN
                    try:
                        self.db_manager.execute_query(
                            dialect.alter_table_drop_column(db_type, 'catalog_dk_cache', 'found_classifications')
                        )
                        self.logger.info("✅ catalog_dk_cache migration completed: dropped found_classifications column")
                    except Exception as e:
                        self.logger.error(f"DROP COLUMN failed: {e}. Migration skipped.")
                else:
                    # SQLite: Table rebuild approach
                    try:
                        # Create new table with correct schema
                        self.db_manager.execute_query(f"""
                            CREATE TABLE catalog_dk_cache_new (
                                search_term {dialect.varchar_type(512)} PRIMARY KEY,
                                normalized_term {dialect.varchar_type(512)} NOT NULL,
                                found_titles {dialect.text_type(db_type)},
                                result_count INTEGER DEFAULT 0,
                                last_updated {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                                created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP
                            )
                        """)

                        # Copy data from old table (preserve existing data)
                        self.db_manager.execute_query("""
                            INSERT INTO catalog_dk_cache_new (search_term, normalized_term, found_titles, result_count, last_updated, created_at)
                            SELECT search_term, normalized_term, found_titles, result_count, last_updated, created_at
                            FROM catalog_dk_cache
                        """)

                        # Drop old table and rename new one
                        self.db_manager.execute_query("DROP TABLE catalog_dk_cache")
                        self.db_manager.execute_query("ALTER TABLE catalog_dk_cache_new RENAME TO catalog_dk_cache")

                        # Recreate indexes
                        self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_catalog_normalized ON catalog_dk_cache(normalized_term)")
                        self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_catalog_updated ON catalog_dk_cache(last_updated)")

                        self.logger.info("✅ catalog_dk_cache migration completed: removed found_classifications (table rebuild)")
                    except Exception as e:
                        self.logger.error(f"Migration table rebuild failed: {e}. Migration skipped.")

            # Step 2: Add new columns for TTL and failure tracking - Claude Generated
            if "search_status" not in columns:
                self.logger.info("🔄 Migrating catalog_dk_cache: adding TTL and failure tracking columns...")
                try:
                    # Add new columns using dialect
                    self.db_manager.execute_query(
                        dialect.alter_table_add_column(db_type, 'catalog_dk_cache', 'search_status',
                            f"{dialect.varchar_type(32)} DEFAULT 'success'")
                    )
                    self.db_manager.execute_query(
                        dialect.alter_table_add_column(db_type, 'catalog_dk_cache', 'error_message',
                            dialect.text_type(db_type))
                    )
                    self.db_manager.execute_query(
                        dialect.alter_table_add_column(db_type, 'catalog_dk_cache', 'retry_after',
                            dialect.timestamp_type(db_type))
                    )
                    self.db_manager.execute_query(
                        dialect.alter_table_add_column(db_type, 'catalog_dk_cache', 'consecutive_failures',
                            'INTEGER DEFAULT 0')
                    )
                    self.logger.info("✅ catalog_dk_cache migration completed: added TTL and failure tracking columns")
                except Exception as e:
                    self.logger.warning(f"Failed to add new columns via ALTER TABLE ({e}), attempting table rebuild...")
                    try:
                        # Fallback: table rebuild approach for older SQLite
                        self.db_manager.execute_query(f"""
                            CREATE TABLE catalog_dk_cache_new (
                                search_term {dialect.varchar_type(512)} PRIMARY KEY,
                                normalized_term {dialect.varchar_type(512)} NOT NULL,
                                found_titles {dialect.text_type(db_type)},
                                result_count INTEGER DEFAULT 0,
                                last_updated {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                                created_at {dialect.timestamp_type(db_type)} DEFAULT CURRENT_TIMESTAMP,
                                search_status {dialect.varchar_type(32)} DEFAULT 'success',
                                error_message {dialect.text_type(db_type)},
                                retry_after {dialect.timestamp_type(db_type)},
                                consecutive_failures INTEGER DEFAULT 0
                            )
                        """)

                        # Copy data, preserving existing records with default values for new columns
                        self.db_manager.execute_query("""
                            INSERT INTO catalog_dk_cache_new
                            (search_term, normalized_term, found_titles, result_count, last_updated, created_at, search_status)
                            SELECT search_term, normalized_term, found_titles, result_count, last_updated, created_at, 'success'
                            FROM catalog_dk_cache
                        """)

                        # Drop old table and rename new one
                        self.db_manager.execute_query("DROP TABLE catalog_dk_cache")
                        self.db_manager.execute_query("ALTER TABLE catalog_dk_cache_new RENAME TO catalog_dk_cache")

                        # Recreate indexes
                        self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_catalog_normalized ON catalog_dk_cache(normalized_term)")
                        self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_catalog_updated ON catalog_dk_cache(last_updated)")

                        self.logger.info("✅ catalog_dk_cache migration completed via table rebuild: added TTL columns")
                    except Exception as e2:
                        self.logger.error(f"Migration table rebuild also failed: {e2}. Migration skipped.")

        except Exception as e:
            self.logger.warning(f"Schema migration check failed (non-critical): {e}")
