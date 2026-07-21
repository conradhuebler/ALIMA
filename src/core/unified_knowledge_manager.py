"""
Unified Knowledge Manager - Consolidates GND cache and DK classifications
Claude Generated - Replaces CacheManager + DKCacheManager with Facts/Mappings separation
Now using PyQt6.QtSql via DatabaseManager for seamless SQLite/MariaDB support
"""

import json
import logging
import re
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .database_manager import DatabaseManager
from .sql_dialect import SQLDialect
from ..utils.config_models import DatabaseConfig
from ._ukm_schema import SchemaMigrationMixin
from ._ukm_catalog_dk import CatalogDkCacheMixin


@dataclass
class GNDEntry:
    """Represents a GND entry (Facts) - Claude Generated"""
    gnd_id: str
    title: str
    description: Optional[str] = None
    synonyms: Optional[str] = None
    ddcs: Optional[str] = None
    ppn: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass 
class Classification:
    """Represents a DK/RVK classification (Facts) - Claude Generated"""
    code: str
    type: str  # "DK" or "RVK"
    title: Optional[str] = None
    description: Optional[str] = None
    parent_code: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class SearchMapping:
    """Represents search term mapping (Dynamic) - Claude Generated"""
    search_term: str
    normalized_term: str
    suggester_type: str
    found_gnd_ids: List[str]
    found_classifications: List[Dict[str, str]]
    result_count: int
    last_updated: str
    created_at: str
    # Display-only per-GND-ID hit counts (F-4): {gnd_id: count}. Kept separate
    # from the pool/ranking count, which a cache hit still reports as 1. Empty for
    # rows written before the column existed. - Claude Generated
    gnd_counts: Dict[str, int] = field(default_factory=dict)
    # Denormalized per-GND-ID titles (WP Phase C1a): {gnd_id: title}. Lets a cache
    # hit rebuild items without reading the (now separate) local GND store. Empty
    # for rows written before the column existed → those rows are treated as a
    # cache miss on read. - Claude Generated
    titles: Dict[str, str] = field(default_factory=dict)


class UnifiedKnowledgeManager(SchemaMigrationMixin, CatalogDkCacheMixin):
    """Unified knowledge database manager with Facts/Mappings separation - Claude Generated

    Singleton Pattern: Only one instance per application lifecycle
    - Use UnifiedKnowledgeManager() or UnifiedKnowledgeManager.get_instance() to get the singleton
    - Thread-safe with automatic locking
    - Call reset() only for testing
    """

    # Singleton implementation - Claude Generated
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, db_path: Optional[str] = None, database_config: Optional[DatabaseConfig] = None):
        """Create or return singleton instance - Claude Generated"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self, db_path: Optional[str] = None, database_config: Optional[DatabaseConfig] = None):
        self.logger = logging.getLogger(__name__)

        # Skip re-initialization of existing singleton - Claude Generated
        if UnifiedKnowledgeManager._initialized:
            self.logger.debug("⚠️ UnifiedKnowledgeManager is singleton - skipping re-initialization")
            return

        # Load database config if not provided - Claude Generated (UNIFIED PATH RESOLUTION)
        if database_config is None:
            try:
                from ..utils.config_manager import ConfigManager
                config_manager = ConfigManager()
                config = config_manager.load_config()
                # UNIFIED SINGLE SOURCE OF TRUTH: database_config.sqlite_path
                database_config = config.database_config
                self.logger.debug(f"✅ Database config loaded from config: {database_config.sqlite_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load database config from config: {e}. Using default.")
                # Create default config with OS-specific path
                database_config = DatabaseConfig(db_type='sqlite')

        # Legacy parameter support (db_path) - deprecated
        if db_path is not None:
            self.logger.warning(f"⚠️ db_path parameter is deprecated, use database_config instead")
            if database_config.db_type.lower() in ['sqlite', 'sqlite3']:
                database_config.sqlite_path = db_path

        self.db_path = database_config.sqlite_path

        self.db_manager = DatabaseManager(database_config, f"unified_knowledge_{id(self)}")
        # Plugin-owned local GND authority store in its own SQLite file (WP Phase
        # C1c): the local `gnd_entries` copy is physically separate from the search
        # cache. Distinct connection_name keeps per-thread connections isolated. - Claude Generated
        from src.core.search.providers.gnd_local.store import LocalGndStore
        self.local_gnd = LocalGndStore(database_config, f"gnd_local_{id(self)}")
        self._init_database()
        # One-time, non-destructive migration of a legacy same-file gnd_entries
        # table (older single-DB installs) into the separate store.
        #
        # Gate on the EFFECTIVE engine, not the configured one: when the
        # configured MySQL/MariaDB driver is unavailable, get_connection() falls
        # back to SQLite, so there IS a file to attach even though the config
        # still says "mariadb". Reading database_config.db_type here skipped the
        # migration for every such install — the authority store stayed empty
        # while 207k legacy GND entries sat in the SQLite file next to it.
        # `_init_database` above already forces the connection open for exactly
        # this reason, so the effective type is settled by now. - Claude Generated
        if str(self.db_manager.get_db_type()).lower() in ("sqlite", "sqlite3"):
            self.local_gnd.migrate_from_legacy(database_config.sqlite_path)
        self.db_fallback_notice = getattr(self.db_manager, 'db_fallback_notice', None)

        # Mark as initialized - Claude Generated
        UnifiedKnowledgeManager._initialized = True

    @property
    def _gnd_db(self) -> DatabaseManager:
        """DatabaseManager for the plugin-owned local GND store (gnd_entries).

        Every ``gnd_entries`` query in this class routes through here so the local
        authority copy lives in its own DB, independent of the search cache. - Claude Generated
        """
        return self.local_gnd.db_manager

    @classmethod
    def get_instance(cls, database_config: Optional[DatabaseConfig] = None) -> "UnifiedKnowledgeManager":
        """Get or create singleton instance - Claude Generated

        Thread-safe factory method. Use this instead of __init__() for clarity.

        Args:
            database_config: Optional DatabaseConfig. Ignored if instance already exists.

        Returns:
            UnifiedKnowledgeManager: Singleton instance
        """
        return cls(database_config=database_config)

    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing only) - Claude Generated

        WARNING: This should only be called during unit tests!
        Closes the database connection and clears the singleton.
        """
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance.db_manager.close()
                    cls.logger.info("✅ Database connection closed")
                except Exception as e:
                    logging.getLogger(__name__).warning(f"⚠️ Error closing database: {e}")
                try:
                    local_gnd = getattr(cls._instance, "local_gnd", None)
                    if local_gnd is not None:
                        local_gnd.close()
                except Exception as e:
                    logging.getLogger(__name__).warning(f"⚠️ Error closing local GND store: {e}")
            cls._instance = None
            cls._initialized = False
    
    # === GND FACTS MANAGEMENT ===
    
    def store_gnd_fact(self, gnd_id: str, gnd_data: Dict[str, Any]):
        """Store GND entry as immutable fact - Claude Generated"""
        try:
            self._gnd_db.execute_query("""
                INSERT OR REPLACE INTO gnd_entries
                (gnd_id, title, description, synonyms, ddcs, ppn, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                gnd_id,
                gnd_data.get('title', ''),
                gnd_data.get('description', ''),
                gnd_data.get('synonyms', ''),
                gnd_data.get('ddcs', ''),
                gnd_data.get('ppn', '')
            ])

        except Exception as e:
            self.logger.error(f"Error storing GND fact {gnd_id}: {e}")
            raise

    # NOTE (WP Phase C1): the former ``warm_gnd_entries`` seam is removed. A search
    # no longer writes stubs into the local GND store — cache hits resolve titles
    # from the denormalized ``search_mappings.titles`` column (see
    # ``CachingProvider``), keeping the plugin-owned local GND copy independent of
    # the search cache. The local copy is filled only by deliberate import /
    # enrichment. - Claude Generated

    def get_gnd_fact(self, gnd_id: str) -> Optional[GNDEntry]:
        """Retrieve GND fact by ID - Claude Generated"""
        try:
            row = self._gnd_db.fetch_one(
                "SELECT * FROM gnd_entries WHERE gnd_id = ?", [gnd_id]
            )

            if row:
                return GNDEntry(
                    gnd_id=row['gnd_id'],
                    title=row['title'],
                    description=row['description'],
                    synonyms=row['synonyms'],
                    ddcs=row['ddcs'],
                    ppn=row['ppn'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving GND fact {gnd_id}: {e}")
            return None

    def get_gnd_facts_batch(self, gnd_ids: List[str]) -> Dict[str, GNDEntry]:
        """Retrieve multiple GND facts in a single batch query - Claude Generated

        Args:
            gnd_ids: List of GND identifiers to retrieve

        Returns:
            Dictionary mapping gnd_id -> GNDEntry for found entries
            Missing IDs are not included in the result

        Performance: Processes in chunks of 100 IDs to avoid memory issues with PyQt6 QSqlQuery
                     For large batches, automatically splits into multiple smaller queries
        """
        if not gnd_ids:
            return {}

        try:
            results = {}

            # REDUCED: Conservative chunk size to avoid QSqlQuery memory issues - Claude Generated
            # Testing shows segfaults with 100+ chunk size, 50 is safer for large batches (1080+ entries)
            chunk_size = 50

            for i in range(0, len(gnd_ids), chunk_size):
                chunk = gnd_ids[i:i + chunk_size]

                try:
                    # Build parameterized query: WHERE gnd_id IN (?, ?, ...)
                    placeholders = ','.join(['?'] * len(chunk))
                    query = f"SELECT * FROM gnd_entries WHERE gnd_id IN ({placeholders})"

                    rows = self._gnd_db.fetch_all(query, chunk)

                    for row in rows:
                        try:
                            # DEFENSIVE: Validate all fields before creating GNDEntry - Claude Generated
                            gnd_id = row.get('gnd_id')
                            if not gnd_id:
                                self.logger.warning(f"Row missing gnd_id: {row}")
                                continue

                            # Validate title exists
                            title = row.get('title', '')
                            if not title:
                                self.logger.warning(f"GND entry {gnd_id} has no title")
                                continue

                            # Safe synonym handling with explicit None check
                            synonyms = row.get('synonyms')
                            if synonyms is not None and not isinstance(synonyms, str):
                                synonyms = str(synonyms)  # Force conversion

                            results[gnd_id] = GNDEntry(
                                gnd_id=gnd_id,
                                title=title,
                                description=row.get('description'),
                                synonyms=synonyms,
                                ddcs=row.get('ddcs'),
                                ppn=row.get('ppn'),
                                created_at=row.get('created_at'),
                                updated_at=row.get('updated_at')
                            )
                        except Exception as row_error:
                            self.logger.warning(f"Failed to create GNDEntry for {row.get('gnd_id', 'unknown')}: {row_error}")
                            continue

                except Exception as chunk_error:
                    self.logger.error(f"Error processing chunk {i}-{i+chunk_size}: {chunk_error}")
                    # Continue with next chunk instead of failing entirely
                    continue

            self.logger.debug(f"Batch query: Retrieved {len(results)}/{len(gnd_ids)} GND entries ({len(gnd_ids)//chunk_size + 1} chunks of {chunk_size})")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch GND query: {e}")
            return {}

    # === CLASSIFICATION FACTS MANAGEMENT ===
    
    def store_classification_fact(self, code: str, classification_type: str, title: str = None,
                                description: str = None, parent_code: str = None):
        """Store classification as immutable fact - Claude Generated"""
        try:
            self.db_manager.execute_query("""
                INSERT OR REPLACE INTO classifications
                (code, type, title, description, parent_code)
                VALUES (?, ?, ?, ?, ?)
            """, [code, classification_type, title, description, parent_code])

        except Exception as e:
            self.logger.error(f"Error storing classification fact {code}: {e}")
            raise
    
    def get_classification_fact(self, code: str, classification_type: str) -> Optional[Classification]:
        """Retrieve classification fact - Claude Generated"""
        try:
            row = self.db_manager.fetch_one(
                "SELECT * FROM classifications WHERE code = ? AND type = ?",
                [code, classification_type]
            )

            if row:
                return Classification(
                    code=row['code'],
                    type=row['type'],
                    title=row['title'],
                    description=row['description'],
                    parent_code=row['parent_code'],
                    created_at=row['created_at']
                )
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving classification fact {code}: {e}")
            return None
    
    # === SEARCH MAPPINGS MANAGEMENT ===
    
    def get_search_mapping(self, search_term: str, suggester_type: str) -> Optional[SearchMapping]:
        """Get existing search mapping - Claude Generated"""
        try:
            row = self.db_manager.fetch_one("""
                SELECT * FROM search_mappings
                WHERE search_term = ? AND suggester_type = ?
            """, [search_term, suggester_type])

            if row:
                return SearchMapping(
                    search_term=row['search_term'],
                    normalized_term=row['normalized_term'],
                    suggester_type=row['suggester_type'],
                    found_gnd_ids=json.loads(row['found_gnd_ids'] or '[]'),
                    found_classifications=json.loads(row['found_classifications'] or '[]'),
                    result_count=row['result_count'],
                    last_updated=row['last_updated'],
                    created_at=row['created_at'],
                    gnd_counts=json.loads((row.get('gnd_counts') if hasattr(row, 'get') else None) or '{}'),
                    titles=json.loads((row.get('titles') if hasattr(row, 'get') else None) or '{}'),
                )
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving search mapping {search_term}: {e}")
            return None
    
    def get_all_gnd_ids_for_keyword(self, search_term: str) -> List[str]:
        """Every GND id the mapping cache holds for a term, across ALL sources.

        ``get_search_mapping`` answers per (term, source); a caller asking "do we
        already know GND ids for this subject?" — as the catalog-subject
        validation does before falling back to a live SWB lookup — wants the
        union. Matching is on the normalised term as well as the literal one, so
        a catalog subject spelled slightly differently still hits.

        Returns ``[]`` on a miss or a read error: the callers treat an empty
        result as "not cached, look it up", which is the safe direction.

        This existed as a CALL before it existed as a method
        (``pipeline_utils._validate_catalog_subjects``), i.e. an unguarded
        AttributeError on every catalog subject without a GND id.
        - Claude Generated
        """
        if not search_term:
            return []
        try:
            normalized = self._normalize_term(search_term)
            rows = self.db_manager.fetch_all(
                "SELECT found_gnd_ids FROM search_mappings "
                "WHERE search_term = ? OR normalized_term = ?",
                [search_term, normalized],
            )
        except Exception as e:
            self.logger.warning(f"get_all_gnd_ids_for_keyword('{search_term}') failed: {e}")
            return []

        out: List[str] = []
        for row in rows or []:
            try:
                ids = json.loads(row.get("found_gnd_ids") or "[]")
            except (TypeError, ValueError):
                continue
            for gnd_id in ids:
                gnd_id = str(gnd_id or "").strip()
                if gnd_id and gnd_id not in out:
                    out.append(gnd_id)
        return out

    def update_search_mapping(self, search_term: str, suggester_type: str,
                            found_gnd_ids: List[str] = None,
                            found_classifications: List[Dict[str, str]] = None,
                            gnd_counts: Dict[str, int] = None,
                            titles: Dict[str, str] = None):
        """Update or create search mapping - Claude Generated (Fixed PyQt6 QtSql subquery issue)

        ``gnd_counts`` (F-4): optional ``{gnd_id: count}`` display-only hit counts
        persisted alongside the GND-ID list, so a later cache hit can restore the
        real Häufigkeit without touching the pool/ranking count.

        ``titles`` (WP Phase C1a): optional ``{gnd_id: title}`` denormalized into the
        mapping row so a later cache hit rebuilds items without reading the (now
        separate, plugin-owned) local GND store.
        """
        try:
            normalized_term = self._normalize_term(search_term)

            # FIX: Replace nested subquery with pre-fetch to avoid QtSql parameter binding issues - Claude Generated
            # PyQt6's QtSql has problems with complex nested queries after cache clear
            existing_mapping = self.get_search_mapping(search_term, suggester_type)
            created_at_value = existing_mapping.created_at if existing_mapping else None

            # Simple INSERT OR REPLACE without subquery - Claude Generated
            self.db_manager.execute_query("""
                INSERT OR REPLACE INTO search_mappings
                (search_term, normalized_term, suggester_type, found_gnd_ids,
                 found_classifications, gnd_counts, titles, result_count, last_updated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, COALESCE(?, CURRENT_TIMESTAMP))
            """, [
                search_term,
                normalized_term,
                suggester_type,
                json.dumps(found_gnd_ids or []),
                json.dumps(found_classifications or []),
                json.dumps(gnd_counts or {}),
                json.dumps(titles or {}),
                len(found_gnd_ids or []) + len(found_classifications or []),
                created_at_value  # Pre-fetched value instead of subquery
            ])

        except Exception as e:
            self.logger.error(f"Error updating search mapping {search_term}: {e}")
            # FIX: Graceful degradation instead of crash - Claude Generated
            self.logger.warning(f"⚠️ Continuing despite search mapping error for '{search_term}'")
            # Don't raise exception - allow application to continue
    
    def _normalize_term(self, term: str) -> str:
        """Normalize search term for fuzzy matching - Claude Generated"""
        # Remove GND-ID suffixes
        if "(GND-ID:" in term:
            term = term.split("(GND-ID:")[0].strip()
        
        # Convert to lowercase, remove special chars
        normalized = re.sub(r'[^\w\s]', ' ', term.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    # === RAW RESPONSE CACHE (WP2 raw-first) === Claude Generated

    @staticmethod
    def params_hash(params: Optional[Dict[str, Any]] = None) -> str:
        """Stable short hash of the caching-relevant request params.

        Canonical (sorted-key, tight-separator) JSON → sha256 → first 16 hex
        chars. Keys whose value is ``None`` are dropped, so an omitted param and
        an explicit ``None`` hash identically. ``search_type`` etc. MUST be in
        ``params`` or e.g. ``kw`` vs ``title`` queries would collide.
        - Claude Generated
        """
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def store_raw_response(self, source: str, query: str,
                           params: Optional[Dict[str, Any]] = None,
                           raw_json: str = "", *,
                           http_status: Optional[int] = None,
                           result_count: int = 0,
                           max_bytes: int = 1_000_000,
                           max_rows_per_source: int = 5000) -> bool:
        """Store a verbatim source response, keyed by (source, normalized_query, params_hash).

        Additive write-only cache (WP2 P1). Never raises — a cache failure must
        never break a live search. Skips oversized blobs (size cap) and prunes
        the oldest rows per source (soft row cap). - Claude Generated
        """
        try:
            if raw_json is None:
                return False
            byte_size = len(raw_json.encode("utf-8"))
            if byte_size > max_bytes:
                self.logger.debug(
                    f"⏭️ Skipping raw-cache write for '{query}' ({source}): "
                    f"{byte_size} bytes > cap {max_bytes}"
                )
                return False

            normalized_query = self._normalize_term(query)
            phash = self.params_hash(params)
            params_json = json.dumps(
                {k: v for k, v in (params or {}).items() if v is not None},
                sort_keys=True, ensure_ascii=False
            )

            self.db_manager.execute_query("""
                INSERT OR REPLACE INTO search_response_cache
                (source, query, normalized_query, params_hash, params_json,
                 raw_json, http_status, result_count, byte_size, last_updated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP,
                        COALESCE((SELECT created_at FROM search_response_cache
                                  WHERE source = ? AND normalized_query = ? AND params_hash = ?),
                                 CURRENT_TIMESTAMP))
            """, [
                source, query, normalized_query, phash, params_json,
                raw_json, http_status, result_count, byte_size,
                source, normalized_query, phash,
            ])

            self._prune_raw_responses(source, max_rows_per_source)
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ raw-cache write failed for '{query}' ({source}): {e}")
            return False

    def get_raw_response(self, source: str, query: str,
                         params: Optional[Dict[str, Any]] = None, *,
                         max_age_hours: Optional[int] = 24) -> Optional[Dict[str, Any]]:
        """Fetch a cached raw response, or None on miss / stale / error.

        ``max_age_hours=None`` disables the freshness gate. - Claude Generated
        """
        try:
            normalized_query = self._normalize_term(query)
            phash = self.params_hash(params)
            row = self.db_manager.fetch_one("""
                SELECT raw_json, http_status, result_count, last_updated
                FROM search_response_cache
                WHERE source = ? AND normalized_query = ? AND params_hash = ?
            """, [source, normalized_query, phash])
            if not row:
                return None
            if max_age_hours is not None and not self._raw_is_fresh(row['last_updated'], max_age_hours):
                return None
            return {
                "raw_json": row['raw_json'],
                "http_status": row['http_status'],
                "result_count": row['result_count'],
                "last_updated": row['last_updated'],
            }
        except Exception as e:
            self.logger.error(f"Error retrieving raw response for '{query}' ({source}): {e}")
            return None

    @staticmethod
    def _raw_is_fresh(last_updated: Any, max_age_hours: int) -> bool:
        """TTL check mirroring CachingProvider._is_fresh - Claude Generated"""
        from datetime import timedelta
        try:
            if isinstance(last_updated, datetime):
                ts = last_updated
            else:
                ts = datetime.fromisoformat(str(last_updated).replace("Z", "+00:00"))
            return datetime.now() - ts < timedelta(hours=max_age_hours)
        except (ValueError, TypeError):
            return False

    def _prune_raw_responses(self, source: str, max_rows_per_source: int) -> None:
        """Delete rows for a source older than the Nth-newest (soft row cap).

        Best-effort; ties at the cutoff timestamp are kept, so the table may
        briefly exceed the cap. Deletes nothing while under the cap (the OFFSET
        subquery yields NULL). - Claude Generated
        """
        try:
            self.db_manager.execute_query("""
                DELETE FROM search_response_cache
                WHERE source = ? AND last_updated < (
                    SELECT last_updated FROM search_response_cache
                    WHERE source = ?
                    ORDER BY last_updated DESC
                    LIMIT 1 OFFSET ?
                )
            """, [source, source, max_rows_per_source])
        except Exception as e:
            self.logger.debug(f"raw-cache prune skipped for '{source}': {e}")

    # === SEARCH FUNCTIONALITY ===
    
    def search_local_gnd(self, term: str, min_results: int = 3) -> List[GNDEntry]:
        """Search for GND entries locally - Claude Generated"""
        try:
            normalized_term = self._normalize_term(term)
            entries = []

            # Exact title match first
            rows = self._gnd_db.fetch_all("""
                SELECT * FROM gnd_entries
                WHERE title LIKE ? OR title LIKE ?
                LIMIT ?
            """, [f"%{term}%", f"%{normalized_term}%", min_results * 2])

            for row in rows:
                entries.append(GNDEntry(
                    gnd_id=row['gnd_id'],
                    title=row['title'],
                    description=row['description'],
                    synonyms=row['synonyms'],
                    ddcs=row['ddcs'],
                    ppn=row['ppn'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))

            # Return whatever local hits exist (capped), rather than discarding
            # 1–2 valid matches when fewer than min_results — a partial local hit
            # still beats an empty answer for the agent. - Claude Generated
            return entries[:min_results]

        except Exception as e:
            self.logger.error(f"Error in local GND search: {e}")
            return []
    
    # === UTILITY METHODS ===
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get unified database statistics - Claude Generated"""
        try:
            stats = {}

            # Count facts
            stats['gnd_entries_count'] = self._gnd_db.fetch_scalar(
                "SELECT COUNT(*) FROM gnd_entries"
            )

            stats['classifications_count'] = self.db_manager.fetch_scalar(
                "SELECT COUNT(*) FROM classifications"
            )

            # Count mappings
            stats['search_mappings_count'] = self.db_manager.fetch_scalar(
                "SELECT COUNT(*) FROM search_mappings"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def clear_database(self):
        """Clear all data for fresh start - Claude Generated"""
        try:
            self.db_manager.execute_query("DELETE FROM search_mappings")
            self.db_manager.execute_query("DELETE FROM search_response_cache")
            self.db_manager.execute_query("DELETE FROM catalog_dk_cache")
            self.db_manager.execute_query("DELETE FROM classifications")
            self._gnd_db.execute_query("DELETE FROM gnd_entries")

            self.logger.info("Database cleared for fresh start")

        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            raise

    def clear_search_cache(self) -> tuple[bool, str]:
        """Clear the GND search caches (mapping + raw response) — Claude Generated

        Removes cached search results — BOTH the mapping-first index
        (``search_mappings``) AND the WP2 raw response cache
        (``search_response_cache``) — so the next search re-fetches live. GND
        entries and classifications (the knowledge base) are preserved.

        Clearing the raw cache too is essential: since the pool is derived from
        raw (P4), leaving it would keep serving cached results.

        Returns:
            tuple[bool, str]: (success, message)
        """
        try:
            # ``execute_query`` runs in QtSql autocommit mode (no explicit
            # transaction), so each DELETE is committed on exec. Do NOT call
            # commit_transaction() here — there is no active transaction, which
            # raises "cannot commit - no transaction is active". - Claude Generated
            self.db_manager.execute_query("DELETE FROM search_mappings")
            self.db_manager.execute_query("DELETE FROM search_response_cache")

            m = self.db_manager.fetch_one("SELECT COUNT(*) AS n FROM search_mappings")
            r = self.db_manager.fetch_one("SELECT COUNT(*) AS n FROM search_response_cache")
            remaining_m = (m or {}).get("n", 0)
            remaining_r = (r or {}).get("n", 0)

            success_msg = (
                f"✅ Search cache cleared (mappings + raw responses). "
                f"{remaining_m} mappings, {remaining_r} raw responses remaining."
            )
            self.logger.info(success_msg)

            return True, success_msg

        except Exception as e:
            error_msg = f"❌ Error clearing search cache: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def cleanup_malformed_classifications(self) -> tuple[bool, str]:
        """Remove malformed classification entries (count>0 but no titles) - Claude Generated

        These entries can block live searches. This cleanup removes entries where:
        - count > 0 but titles list is empty
        - These are typically from old data before the ultra-deep fix

        Returns:
            tuple[bool, str]: (success, message)
        """
        try:
            # Query all search_mappings entries
            query = "SELECT search_term, found_classifications FROM search_mappings"
            rows = self.db_manager.fetch_all(query)

            if not rows:
                msg = "✅ No entries to cleanup"
                self.logger.info(msg)
                return True, msg

            cleaned_count = 0
            updated_entries = 0

            for search_term, found_classifications_json in rows:
                try:
                    classifications = json.loads(found_classifications_json) if found_classifications_json else []

                    # Filter out malformed entries - Claude Generated
                    valid_classifications = [
                        cls for cls in classifications
                        if cls.get("count", 0) > 0 and cls.get("titles")  # Must have count AND titles
                    ]

                    if len(valid_classifications) < len(classifications):
                        removed = len(classifications) - len(valid_classifications)
                        cleaned_count += removed

                        # Update or delete the entry - Claude Generated
                        if valid_classifications:
                            # Update with cleaned data
                            update_query = """
                                UPDATE search_mappings
                                SET found_classifications = ?
                                WHERE search_term = ?
                            """
                            self.db_manager.execute_query(update_query, [
                                json.dumps(valid_classifications),
                                search_term
                            ])
                            updated_entries += 1
                            self.logger.debug(f"Cleaned {removed} malformed entries for '{search_term}'")
                        else:
                            # Delete entire entry if no valid classifications remain
                            delete_query = "DELETE FROM search_mappings WHERE search_term = ?"
                            self.db_manager.execute_query(delete_query, [search_term])
                            self.logger.debug(f"Deleted search_mappings entry for '{search_term}' (no valid classifications)")

                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️ Could not parse classifications for '{search_term}': {e}")
                    continue

            # execute_query autocommits (QtSql autocommit mode) — no explicit
            # transaction is open, so don't call commit_transaction(). - Claude Generated

            success_msg = f"✅ Cleaned {cleaned_count} malformed entries ({updated_entries} entries updated/deleted)"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            error_msg = f"❌ Error cleaning malformed entries: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    # === COMPATIBILITY ADAPTERS ===
    # These methods provide compatibility with existing CacheManager and DKCacheManager interfaces
    
    def get_gnd_entry_by_id(self, gnd_id: str) -> Optional[Dict]:
        """CacheManager compatibility - Claude Generated"""
        entry = self.get_gnd_fact(gnd_id)
        if entry:
            return {
                'gnd_id': entry.gnd_id,
                'title': entry.title,
                'description': entry.description,
                'synonyms': entry.synonyms,
                'ddcs': entry.ddcs,
                'ppn': entry.ppn
            }
        return None
    
    def add_gnd_entry(self, gnd_id: str, title: str, description: str = "", 
                     synonyms: str = "", ddcs: str = "", ppn: str = ""):
        """CacheManager compatibility - Claude Generated"""
        gnd_data = {
            'title': title,
            'description': description,
            'synonyms': synonyms,
            'ddcs': ddcs,
            'ppn': ppn
        }
        self.store_gnd_fact(gnd_id, gnd_data)
    
    
    
    
    # DKCacheManager compatibility methods
    
    def search_by_keywords(self, keywords: List[str], fuzzy_threshold: int = 80) -> List:
        """
        Search cached DK classifications by keywords - Claude Generated
        MODIFIED: Now redirects to dedicated catalog_dk_cache table

        Args:
            keywords: List of keywords to search for
            fuzzy_threshold: Minimum similarity threshold (not used for now)

        Returns:
            List of cached classification results with metadata
        """
        # Redirect to dedicated catalog cache
        return self.search_catalog_dk_cache(keywords)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get unified cache statistics - Claude Generated
        MODIFIED: Now includes catalog_dk_cache statistics"""
        try:
            # Count entries across all tables
            gnd_count = self._gnd_db.fetch_scalar("SELECT COUNT(*) FROM gnd_entries")
            classification_count = self.db_manager.fetch_scalar("SELECT COUNT(*) FROM classifications")
            mapping_count = self.db_manager.fetch_scalar("SELECT COUNT(*) FROM search_mappings")
            catalog_cache_count = self.db_manager.fetch_scalar("SELECT COUNT(*) FROM catalog_dk_cache")

            total_entries = gnd_count + classification_count + mapping_count + catalog_cache_count

            # Get database file size (only for SQLite)
            import os
            try:
                if self.db_manager.config.db_type.lower() in ['sqlite', 'sqlite3']:
                    size_bytes = os.path.getsize(self.db_path)
                    size_mb = size_bytes / (1024 * 1024)
                else:
                    size_mb = 0.0  # For MySQL/MariaDB, size calculation would be different
            except OSError:
                size_mb = 0.0

            return {
                "total_entries": total_entries,
                "gnd_entries": gnd_count,
                "classification_entries": classification_count,
                "search_mappings": mapping_count,
                "catalog_dk_cache": catalog_cache_count,
                "size_mb": round(size_mb, 2),
                "file_path": self.db_path
            }

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {
                "total_entries": 0,
                "gnd_entries": 0,
                "classification_entries": 0,
                "search_mappings": 0,
                "catalog_dk_cache": 0,
                "size_mb": 0.0,
                "file_path": self.db_path
            }

    # === DEDICATED CATALOG DK CACHE MANAGEMENT === Claude Generated

    
    def insert_gnd_entry(
        self,
        gnd_id: str,
        title: str,
        description: str = "",
        ddcs: str = "",
        dks: str = "",
        gnd_systems: str = "",
        synonyms: str = "",
        classification: str = "",
        ppn: str = "",
    ):
        """CacheManager compatibility - Claude Generated"""
        # Convert parameters to dictionary format expected by store_gnd_fact
        gnd_data = {
            'title': title,
            'description': description,
            'synonyms': synonyms,
            'ddcs': ddcs,
            'ppn': ppn
        }
        
        # Store the GND fact using unified storage
        self.store_gnd_fact(gnd_id, gnd_data)
    
    def gnd_entry_exists(self, gnd_id: str) -> bool:
        """CacheManager compatibility - Claude Generated"""
        entry = self.get_gnd_fact(gnd_id)
        return entry is not None
    
    def get_gnd_title_by_id(self, gnd_id: str) -> Optional[str]:
        """CacheManager compatibility - Claude Generated"""
        entry = self.get_gnd_fact(gnd_id)
        if entry:
            return entry.title
        return None

    def get_gnd_synonyms_by_id(self, gnd_id: str) -> List[str]:
        """Get GND synonyms by ID - Claude Generated

        Args:
            gnd_id: GND identifier

        Returns:
            List of synonym strings (empty list if not found or no synonyms)
        """
        entry = self.get_gnd_fact(gnd_id)
        if entry and entry.synonyms:
            # Split by semicolon and strip whitespace
            return [s.strip() for s in entry.synonyms.split(';') if s.strip()]
        return []

    def search_gnd_by_title(self, keyword_text: str, fuzzy_threshold: int = 90) -> List[Dict[str, str]]:
        """Search GND entries by title or synonyms - Claude Generated

        Args:
            keyword_text: Keyword text to search for
            fuzzy_threshold: Minimum similarity threshold (0-100), not used for exact match

        Returns:
            List of dicts with 'gnd_id', 'title', 'synonyms' keys
        """
        try:
            keyword_lower = keyword_text.lower().strip()
            self.logger.debug(f"Searching GND by title: '{keyword_text}'")

            # Exact match on title (case-insensitive)
            exact_match = self._gnd_db.fetch_one(
                "SELECT gnd_id, title, synonyms FROM gnd_entries WHERE LOWER(title) = ?",
                [keyword_lower]
            )

            if exact_match:
                self.logger.debug(f"Found exact title match for '{keyword_text}': {exact_match[0]}")
                return [{
                    'gnd_id': exact_match[0],
                    'title': exact_match[1],
                    'synonyms': exact_match[2] or ''
                }]

            # Fallback: Check if keyword appears in synonyms (semicolon-separated)
            synonym_matches = self._gnd_db.fetch_all(
                """SELECT gnd_id, title, synonyms FROM gnd_entries
                   WHERE synonyms LIKE ?""",
                [f"%{keyword_lower}%"]
            )

            results = []
            for row in synonym_matches:
                # Verify it's actually a full synonym match (not partial)
                synonyms = row[2] or ''
                synonym_list = [s.strip().lower() for s in synonyms.split(';')]
                if keyword_lower in synonym_list:
                    results.append({
                        'gnd_id': row[0],
                        'title': row[1],
                        'synonyms': row[2] or ''
                    })

            if results:
                self.logger.debug(f"Found {len(results)} synonym match(es) for '{keyword_text}'")
            else:
                self.logger.debug(f"No GND entry found for '{keyword_text}'")
            return results

        except Exception as e:
            self.logger.warning(f"Error searching GND by title '{keyword_text}': {e}")
            return []

    # === CHAT MUTATIONS AUDIT (P-ε) — Claude Generated ===

    def record_mutation_pending(
        self,
        session_id: str,
        tool_name: str,
        operation: str,
        payload: Dict[str, Any],
    ) -> Optional[int]:
        """Insert a pending mutation audit row. Returns new row id (or None on error)."""
        try:
            import json as _json
            self.db_manager.execute_query(
                """
                INSERT INTO chat_mutations
                    (session_id, tool_name, operation, payload_json,
                     accepted, reject_reason, created_at, applied_at)
                VALUES (?, ?, ?, ?, NULL, NULL, CURRENT_TIMESTAMP, NULL)
                """,
                [session_id, tool_name, operation, _json.dumps(payload)],
            )
            row = self.db_manager.fetch_one(
                "SELECT id FROM chat_mutations "
                "WHERE session_id = ? AND tool_name = ? "
                "ORDER BY id DESC LIMIT 1",
                [session_id, tool_name],
            )
            return int(row["id"]) if row else None
        except Exception as e:
            self.logger.error(f"Error recording mutation pending: {e}")
            return None

    def record_mutation_outcome(
        self,
        audit_id: int,
        accepted: bool,
        reject_reason: str = "",
    ) -> None:
        """Update an existing audit row with the user's decision."""
        try:
            if accepted:
                self.db_manager.execute_query(
                    "UPDATE chat_mutations "
                    "SET accepted = ?, applied_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    [True, audit_id],
                )
            else:
                self.db_manager.execute_query(
                    "UPDATE chat_mutations "
                    "SET accepted = ?, reject_reason = ? "
                    "WHERE id = ?",
                    [False, reject_reason or "", audit_id],
                )
        except Exception as e:
            self.logger.error(f"Error recording mutation outcome {audit_id}: {e}")

    def save_to_file(self):
        """CacheManager compatibility - Claude Generated"""
        # For QtSql databases, this ensures any pending operations are committed
        # Most operations are auto-committed, but this provides compatibility
        try:
            # Simple integrity check using QtSql
            result = self.db_manager.fetch_scalar("SELECT 1")
            if result == 1:
                self.logger.debug("Database connection verified")
            else:
                self.logger.warning("Database integrity check returned unexpected result")
        except Exception as e:
            self.logger.warning(f"Database save operation warning: {e}")
    
    # === WEEK 2: SMART SEARCH INTEGRATION ===
    
    def search_with_mappings_first(self, search_term: str, suggester_type: str,
                                 max_age_hours: int = 24,
                                 live_search_fallback: callable = None,
                                 force_update: bool = False) -> tuple[List[str], bool]:
        """
        Week 2: Smart search with mappings-first strategy - Claude Generated

        Args:
            search_term: Term to search for
            suggester_type: Type of suggester (lobid, swb, catalog)
            max_age_hours: Maximum age of cached mappings in hours
            live_search_fallback: Function to call for live search if mapping miss
            force_update: If True, ignore cache and force live search - Claude Generated

        Returns:
            Tuple of (found_gnd_ids, was_from_cache)
        """
        from datetime import datetime, timedelta

        # Step 1: Force live search if force_update is True - Claude Generated
        if force_update:
            if hasattr(self, 'debug_mapping') and self.debug_mapping:
                self.logger.info(f"⚠️ Force update: skipping cache for '{search_term}' ({suggester_type})")
            # Skip cache check and go directly to live search
            if live_search_fallback:
                try:
                    live_results = live_search_fallback(search_term)
                    if live_results:
                        gnd_ids = self._extract_gnd_ids_from_results(live_results, suggester_type)
                        # Update mapping with fresh results (merging will be handled in store_classification_results)
                        self.update_search_mapping(
                            search_term=search_term,
                            suggester_type=suggester_type,
                            found_gnd_ids=gnd_ids
                        )
                        self.logger.info(f"✅ Force update complete for '{search_term}': {len(gnd_ids)} results")
                        return gnd_ids, False
                except Exception as e:
                    self.logger.error(f"Force update failed for '{search_term}': {e}")
            return [], False

        # Step 2: Normal cache-first logic
        mapping = self.get_search_mapping(search_term, suggester_type)
        
        if mapping:
            # Check if mapping is fresh enough
            try:
                # Handle both string and datetime objects (MariaDB returns datetime, not string)
                last_updated_val = mapping.last_updated
                if isinstance(last_updated_val, datetime):
                    last_updated = last_updated_val
                else:
                    last_updated = datetime.fromisoformat(str(last_updated_val).replace('Z', '+00:00'))
                max_age = timedelta(hours=max_age_hours)

                if datetime.now() - last_updated < max_age:
                    if hasattr(self, 'debug_mapping') and self.debug_mapping:
                        self.logger.info(f"✅ Mapping hit for '{search_term}' ({suggester_type}): {len(mapping.found_gnd_ids)} results from cache")
                    return mapping.found_gnd_ids, True
                else:
                    self.logger.info(f"⏰ Stale mapping for '{search_term}' ({suggester_type}): {(datetime.now() - last_updated).total_seconds()/3600:.1f}h old")
            except ValueError:
                self.logger.warning(f"Invalid last_updated timestamp for mapping: {mapping.last_updated}")
        else:
            if hasattr(self, 'debug_mapping') and self.debug_mapping:
                self.logger.info(f"❌ No mapping found for '{search_term}' ({suggester_type})")
        
        # Step 2: Mapping miss or stale - fallback to live search
        if live_search_fallback:
            if hasattr(self, 'debug_mapping') and self.debug_mapping:
                self.logger.info(f"🌐 Performing live search for '{search_term}' ({suggester_type})")
            try:
                live_results = live_search_fallback(search_term)
                
                # Step 3: Update mapping with fresh results
                if live_results:
                    # Extract GND IDs from live results (format depends on suggester)
                    gnd_ids = self._extract_gnd_ids_from_results(live_results, suggester_type)
                    
                    # Store the updated mapping
                    self.update_search_mapping(
                        search_term=search_term,
                        suggester_type=suggester_type, 
                        found_gnd_ids=gnd_ids
                    )
                    
                    if hasattr(self, 'debug_mapping') and self.debug_mapping:
                        self.logger.info(f"✅ Updated mapping for '{search_term}' ({suggester_type}): {len(gnd_ids)} results")
                    return gnd_ids, False
                else:
                    # Store empty result to avoid repeated failed searches
                    self.update_search_mapping(
                        search_term=search_term,
                        suggester_type=suggester_type,
                        found_gnd_ids=[]
                    )
                    self.logger.info(f"∅ No results for '{search_term}' ({suggester_type}) - stored empty mapping")
                    return [], False
                    
            except Exception as e:
                self.logger.error(f"Live search failed for '{search_term}' ({suggester_type}): {e}")
                return [], False
        
        # No live search fallback provided
        self.logger.warning(f"No live search fallback provided for '{search_term}' ({suggester_type})")
        return [], False
    
    def _extract_gnd_ids_from_results(self, results: Dict[str, Any], suggester_type: str) -> List[str]:
        """Extract GND IDs from suggester-specific result format - Claude Generated"""
        gnd_ids = []
        
        try:
            if suggester_type in ("lobid", "swb"):
                # Canonical suggester results: {term: {keyword: {"gnd_ids": set, ...}}}
                for term_results in results.values():
                    for keyword_data in term_results.values():
                        if "gnd_ids" in keyword_data:
                            gnd_set = keyword_data["gnd_ids"]
                            if isinstance(gnd_set, set):
                                gnd_ids.extend(list(gnd_set))
                            elif isinstance(gnd_set, list):
                                gnd_ids.extend(gnd_set)
                                
            elif suggester_type == "catalog":
                # Catalog/BiblioSuggester results may have different format
                # This needs to be adapted based on actual BiblioSuggester output
                self.logger.warning("GND ID extraction for catalog suggester not yet implemented")
                
        except Exception as e:
            self.logger.error(f"Error extracting GND IDs from {suggester_type} results: {e}")
            
        # Remove duplicates and filter out empty/invalid IDs
        unique_gnd_ids = list(set(gid for gid in gnd_ids if gid and len(str(gid).strip()) > 0))
        return unique_gnd_ids
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about search mappings - Claude Generated"""
        try:
            # Total mappings by suggester type
            by_suggester_rows = self.db_manager.fetch_all("""
                SELECT suggester_type, COUNT(*) as count,
                       AVG(result_count) as avg_results,
                       MAX(last_updated) as latest_update
                FROM search_mappings
                GROUP BY suggester_type
            """)

            # Recent activity (last 24 hours)
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

            recent_stats_row = self.db_manager.fetch_one("""
                SELECT COUNT(*) as recent_mappings,
                       AVG(result_count) as recent_avg_results
                FROM search_mappings
                WHERE last_updated > ?
            """, [cutoff])

            return {
                "by_suggester": [
                    {
                        "type": row["suggester_type"],
                        "count": row["count"],
                        "avg_results": round(row["avg_results"] or 0, 1),
                        "latest_update": row["latest_update"]
                    }
                    for row in by_suggester_rows
                ],
                "recent_24h": {
                    "mappings": recent_stats_row["recent_mappings"] if recent_stats_row else 0,
                    "avg_results": round(recent_stats_row["recent_avg_results"] or 0, 1) if recent_stats_row else 0
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting mapping statistics: {e}")
            return {"error": str(e)}

    def get_dk_for_gnd_id(self, gnd_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve DK classifications for a given GND-ID - Claude Generated

        Searches through catalog search mappings to find all DK classifications
        associated with a specific GND-ID, including titles and frequency information.

        Args:
            gnd_id: The GND-ID to search for (e.g., "4061694-5")
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with structure:
            [
                {
                    "dk": "614.7",
                    "type": "DK",
                    "titles": ["Title 1", "Title 2"],
                    "count": 5,
                    "avg_confidence": 0.85
                },
                ...
            ]
        """
        try:
            # Search in search_mappings where suggester_type is 'catalog'
            rows = self.db_manager.fetch_all("""
                SELECT found_classifications
                FROM search_mappings
                WHERE suggester_type = 'catalog'
                AND found_classifications LIKE ?
            """, [f'%"{gnd_id}"%'])

            if not rows:
                self.logger.debug(f"No DK classifications found for GND-ID {gnd_id}")
                return []

            # Parse JSON and collect matching classifications
            classifications = {}  # Use dict to deduplicate by code

            for row in rows:
                try:
                    found_classifications = json.loads(row['found_classifications'] or '[]')

                    for cls in found_classifications:
                        # Check if this classification contains the GND-ID
                        gnd_ids = cls.get('gnd_ids', [])
                        if gnd_id in gnd_ids:
                            code = cls.get('code')

                            # Deduplicate: merge if code already exists
                            if code in classifications:
                                # Merge titles (avoid duplicates)
                                existing_titles = set(classifications[code]['titles'])
                                new_titles = cls.get('titles', [])
                                for title in new_titles:
                                    if title not in existing_titles and len(classifications[code]['titles']) < max_results:
                                        classifications[code]['titles'].append(title)

                                # Update count and confidence
                                classifications[code]['count'] += cls.get('count', 0)  # FIX: Default to 0 (no titles), not 1 - Claude Generated
                                classifications[code]['avg_confidence'] = (
                                    classifications[code]['avg_confidence'] +
                                    cls.get('avg_confidence', 0.8)
                                ) / 2
                            else:
                                # New classification entry
                                classifications[code] = {
                                    'dk': code,
                                    'type': cls.get('type', 'DK'),
                                    'titles': cls.get('titles', [])[:max_results],  # Limit titles
                                    'count': cls.get('count', 1),
                                    'avg_confidence': cls.get('avg_confidence', 0.8)
                                }

                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse classification JSON: {e}")
                    continue

            # Convert to list and sort by count (descending)
            results = list(classifications.values())
            results.sort(key=lambda x: x['count'], reverse=True)

            if results:
                self.logger.info(f"✅ Found {len(results)} DK classifications for GND-ID {gnd_id}")

            return results[:max_results]

        except Exception as e:
            self.logger.error(f"Error searching DK for GND-ID {gnd_id}: {e}")
            return []

    def cleanup_titleless_classifications(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Remove cached classifications without titles from search_mappings - Claude Generated

        Cleans up the database by removing classification entries that have empty
        titles arrays. This helps maintain data quality and reduces prompt bloat.

        Args:
            dry_run: If True, only report what would be cleaned without making changes

        Returns:
            Dictionary with statistics:
            {
                "mappings_processed": int,
                "classifications_removed": int,
                "classifications_kept": int,
                "mappings_updated": int
            }
        """
        try:
            stats = {
                "mappings_processed": 0,
                "classifications_removed": 0,
                "classifications_kept": 0,
                "mappings_updated": 0
            }

            # Get all catalog search mappings
            rows = self.db_manager.fetch_all("""
                SELECT search_term, found_classifications
                FROM search_mappings
                WHERE suggester_type = 'catalog'
            """)

            self.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processing {len(rows)} catalog search mappings...")

            for row in rows:
                stats["mappings_processed"] += 1
                search_term = row['search_term']

                try:
                    classifications = json.loads(row['found_classifications'] or '[]')

                    if not classifications:
                        continue  # Skip empty mappings

                    # Filter out classifications without titles
                    cleaned_classifications = []
                    for cls in classifications:
                        titles = cls.get('titles', [])
                        # Check if at least one valid title exists
                        if titles and any(t.strip() for t in titles if t):
                            cleaned_classifications.append(cls)
                            stats["classifications_kept"] += 1
                        else:
                            stats["classifications_removed"] += 1
                            if dry_run:
                                self.logger.debug(f"[DRY RUN] Would remove {cls.get('type', 'DK')}: {cls.get('code')} from '{search_term}' (no titles)")

                    # Update mapping if classifications were removed
                    if len(cleaned_classifications) < len(classifications):
                        stats["mappings_updated"] += 1

                        if not dry_run:
                            # Update the search mapping with cleaned data
                            self.update_search_mapping(
                                search_term=search_term,
                                suggester_type="catalog",
                                found_classifications=cleaned_classifications
                            )
                            self.logger.debug(f"✅ Cleaned '{search_term}': kept {len(cleaned_classifications)}/{len(classifications)} classifications")
                        else:
                            self.logger.debug(f"[DRY RUN] Would clean '{search_term}': keep {len(cleaned_classifications)}/{len(classifications)} classifications")

                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to process mapping for '{search_term}': {e}")
                    continue

            # Log summary
            action_verb = "Would remove" if dry_run else "Removed"
            self.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleanup summary:")
            self.logger.info(f"  - Mappings processed: {stats['mappings_processed']}")
            self.logger.info(f"  - Classifications kept: {stats['classifications_kept']}")
            self.logger.info(f"  - Classifications {action_verb.lower()}: {stats['classifications_removed']}")
            self.logger.info(f"  - Mappings updated: {stats['mappings_updated']}")

            if dry_run and stats['classifications_removed'] > 0:
                self.logger.info(f"💡 Run with dry_run=False to apply these changes")

            return stats

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                "mappings_processed": 0,
                "classifications_removed": 0,
                "classifications_kept": 0,
                "mappings_updated": 0,
                "error": str(e)
            }