"""
Unified Knowledge Manager - Consolidates GND cache and DK classifications
Claude Generated - Replaces CacheManager + DKCacheManager with Facts/Mappings separation
Now using PyQt6.QtSql via DatabaseManager for seamless SQLite/MariaDB support
"""

import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .database_manager import DatabaseManager
from ..utils.config_models import DatabaseConfig


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


class UnifiedKnowledgeManager:
    """Unified knowledge database manager with Facts/Mappings separation - Claude Generated"""

    def __init__(self, db_path: str = "alima_knowledge.db", database_config: Optional[DatabaseConfig] = None):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Use provided config or create default SQLite config
        if database_config is None:
            database_config = DatabaseConfig(db_type='sqlite', sqlite_path=db_path)

        self.db_manager = DatabaseManager(database_config, f"unified_knowledge_{id(self)}")
        self._init_database()
    
    def _init_database(self):
        """Initialize unified database schema - Claude Generated"""
        try:
            # === FACTS TABLES (Immutable truths) ===

            # 1. GND entries (facts only, no search terms)
            self.db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS gnd_entries (
                    gnd_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    synonyms TEXT,
                    ddcs TEXT,
                    ppn TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. DK/RVK classifications (facts only, no keywords)
            self.db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS classifications (
                    code TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    parent_code TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # === MAPPING TABLES (Dynamic associations) ===

            # 3. Search mappings (search term → found results)
            self.db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS search_mappings (
                    search_term TEXT NOT NULL,
                    normalized_term TEXT NOT NULL,
                    suggester_type TEXT NOT NULL,
                    found_gnd_ids TEXT,
                    found_classifications TEXT,
                    result_count INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (search_term, suggester_type)
                )
            """)

            # Create indexes for performance
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_search_normalized ON search_mappings(normalized_term)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_search_term ON search_mappings(search_term)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_gnd_title ON gnd_entries(title)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_classifications_code ON classifications(code)")
            self.db_manager.execute_query("CREATE INDEX IF NOT EXISTS idx_classifications_type ON classifications(type)")

            self.logger.info("Unified knowledge database schema initialized")

        except Exception as e:
            self.logger.error(f"Error initializing unified database: {e}")
            raise
    
    # === GND FACTS MANAGEMENT ===
    
    def store_gnd_fact(self, gnd_id: str, gnd_data: Dict[str, Any]):
        """Store GND entry as immutable fact - Claude Generated"""
        try:
            self.db_manager.execute_query("""
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
    
    def get_gnd_fact(self, gnd_id: str) -> Optional[GNDEntry]:
        """Retrieve GND fact by ID - Claude Generated"""
        try:
            row = self.db_manager.fetch_one(
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
                    created_at=row['created_at']
                )
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving search mapping {search_term}: {e}")
            return None
    
    def update_search_mapping(self, search_term: str, suggester_type: str,
                            found_gnd_ids: List[str] = None,
                            found_classifications: List[Dict[str, str]] = None):
        """Update or create search mapping - Claude Generated"""
        try:
            normalized_term = self._normalize_term(search_term)

            self.db_manager.execute_query("""
                INSERT OR REPLACE INTO search_mappings
                (search_term, normalized_term, suggester_type, found_gnd_ids,
                 found_classifications, result_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                search_term,
                normalized_term,
                suggester_type,
                json.dumps(found_gnd_ids or []),
                json.dumps(found_classifications or []),
                len(found_gnd_ids or []) + len(found_classifications or [])
            ])

        except Exception as e:
            self.logger.error(f"Error updating search mapping {search_term}: {e}")
            raise
    
    def _normalize_term(self, term: str) -> str:
        """Normalize search term for fuzzy matching - Claude Generated"""
        # Remove GND-ID suffixes
        if "(GND-ID:" in term:
            term = term.split("(GND-ID:")[0].strip()
        
        # Convert to lowercase, remove special chars
        normalized = re.sub(r'[^\w\s]', ' ', term.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    # === SEARCH FUNCTIONALITY ===
    
    def search_local_gnd(self, term: str, min_results: int = 3) -> List[GNDEntry]:
        """Search for GND entries locally - Claude Generated"""
        try:
            normalized_term = self._normalize_term(term)
            entries = []

            # Exact title match first
            rows = self.db_manager.fetch_all("""
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

            return entries[:min_results] if len(entries) >= min_results else []

        except Exception as e:
            self.logger.error(f"Error in local GND search: {e}")
            return []
    
    # === UTILITY METHODS ===
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get unified database statistics - Claude Generated"""
        try:
            stats = {}

            # Count facts
            stats['gnd_entries_count'] = self.db_manager.fetch_scalar(
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
            self.db_manager.execute_query("DELETE FROM classifications")
            self.db_manager.execute_query("DELETE FROM gnd_entries")

            self.logger.info("Database cleared for fresh start")

        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            raise
    
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
    
    def load_entrys(self) -> Dict:
        """CacheManager compatibility - Claude Generated"""
        try:
            entries = {}
            rows = self.db_manager.fetch_all("SELECT * FROM gnd_entries")

            for row in rows:
                entries[row['gnd_id']] = {
                    'gnd_id': row['gnd_id'],
                    'title': row['title'],
                    'description': row['description'],
                    'synonyms': row['synonyms'],
                    'ddcs': row['ddcs'],
                    'ppn': row['ppn']
                }
            return entries

        except Exception as e:
            self.logger.error(f"Error loading entries: {e}")
            return {}
    
    def get_cached_results(self, term: str) -> Optional[List]:
        """CacheManager compatibility - Claude Generated"""
        # Check search mappings for this term
        mapping = self.get_search_mapping(term, "lobid")  # Default to lobid
        if mapping:
            # Convert GND IDs to full entries
            results = []
            for gnd_id in mapping.found_gnd_ids:
                entry = self.get_gnd_fact(gnd_id)
                if entry:
                    results.append({
                        'gnd_id': entry.gnd_id,
                        'title': entry.title,
                        'description': entry.description
                    })
            return results
        return None
    
    def cache_results(self, term: str, results: List[Dict]):
        """CacheManager compatibility - Claude Generated"""
        # Store individual GND facts
        gnd_ids = []
        for result in results:
            if 'gnd_id' in result:
                gnd_id = result['gnd_id']
                self.store_gnd_fact(gnd_id, result)
                gnd_ids.append(gnd_id)
        
        # Create search mapping
        if gnd_ids:
            self.update_search_mapping(term, "lobid", found_gnd_ids=gnd_ids)
    
    # DKCacheManager compatibility methods
    
    def search_by_keywords(self, keywords: List[str], fuzzy_threshold: int = 80) -> List:
        """
        Search cached DK classifications by keywords - Claude Generated

        Args:
            keywords: List of keywords to search for
            fuzzy_threshold: Minimum similarity threshold (not used for now)

        Returns:
            List of cached classification results with metadata
        """
        try:
            results = []

            for keyword in keywords:
                # Clean keyword for search
                clean_keyword = keyword.split('(')[0].strip()

                # Lookup in search_mappings table with full data - Claude Generated
                query = """
                    SELECT DISTINCT
                        c.code,
                        c.type,
                        sm.search_term,
                        sm.found_classifications,
                        sm.created_at
                    FROM search_mappings sm
                    CROSS JOIN classifications c
                    WHERE (sm.search_term LIKE ? OR sm.search_term LIKE ?)
                      AND sm.suggester_type = 'catalog'
                      AND sm.found_classifications LIKE '%' || c.code || '%'
                    LIMIT 50
                """

                rows = self.db_manager.fetch_all(query, (f"%{clean_keyword}%", f"{clean_keyword}%"))

                if rows:
                    self.logger.debug(f"Cache hit: {len(rows)} results for '{clean_keyword}'")

                    # Group by classification code and extract metadata - Claude Generated
                    code_groups = {}
                    for row in rows:
                        code = row["code"]
                        if code not in code_groups:
                            # Parse found_classifications JSON to extract titles and metadata
                            try:
                                classifications = json.loads(row["found_classifications"] or "[]")
                                # Find classification entry matching this code
                                titles = []
                                count = 1
                                avg_confidence = 0.8
                                for cls in classifications:
                                    if cls.get("code") == code:
                                        titles = cls.get("titles", [])
                                        count = cls.get("count", 1)
                                        avg_confidence = cls.get("avg_confidence", 0.8)
                                        break
                            except (json.JSONDecodeError, KeyError):
                                titles = []
                                count = 1
                                avg_confidence = 0.8

                            code_groups[code] = {
                                "dk": code,
                                "classification_type": row["type"],
                                "matched_keywords": [clean_keyword],
                                "titles": titles,
                                "count": count,
                                "avg_confidence": avg_confidence,
                                "total_confidence": avg_confidence
                            }
                        else:
                            code_groups[code]["count"] += 1
                            code_groups[code]["total_confidence"] += 0.8

                    # Calculate averages and add to results
                    for code_data in code_groups.values():
                        if code_data["count"] > 0:
                            code_data["avg_confidence"] = code_data["total_confidence"] / code_data["count"]
                        results.extend([code_data])
                else:
                    self.logger.debug(f"Cache miss: No results for '{clean_keyword}'")

            self.logger.info(f"📊 Cache search: {len(results)} classifications found for {len(keywords)} keywords")
            return results

        except Exception as e:
            self.logger.error(f"Cache search failed: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get unified cache statistics - Claude Generated"""
        try:
            # Count entries across all tables
            gnd_count = self.db_manager.fetch_scalar("SELECT COUNT(*) FROM gnd_entries")
            classification_count = self.db_manager.fetch_scalar("SELECT COUNT(*) FROM classifications")
            mapping_count = self.db_manager.fetch_scalar("SELECT COUNT(*) FROM search_mappings")

            total_entries = gnd_count + classification_count + mapping_count

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
                "size_mb": 0.0,
                "file_path": self.db_path
            }
    
    def store_classification_results(self, results: List[Dict]):
        """DKCacheManager compatibility - Claude Generated"""
        for result in results:
            # Store classification fact
            code = result.get('dk', '')
            classification_type = result.get('classification_type', 'DK')

            if code:
                self.store_classification_fact(code, classification_type)

                # Store titles with classification in search mapping - Claude Generated
                titles = result.get('titles', [])
                keywords = result.get('keywords', [])
                count = result.get('count', 1)
                avg_confidence = result.get('avg_confidence', 0.8)

                for keyword in keywords:
                    self.update_search_mapping(
                        search_term=keyword,
                        suggester_type="catalog",
                        found_classifications=[{
                            "code": code,
                            "type": classification_type,
                            "titles": titles[:5],  # Store up to 5 sample titles
                            "count": count,
                            "avg_confidence": avg_confidence
                        }]
                    )
    
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
                                 live_search_fallback: callable = None) -> tuple[List[str], bool]:
        """
        Week 2: Smart search with mappings-first strategy - Claude Generated
        
        Args:
            search_term: Term to search for
            suggester_type: Type of suggester (lobid, swb, catalog)
            max_age_hours: Maximum age of cached mappings in hours
            live_search_fallback: Function to call for live search if mapping miss
            
        Returns:
            Tuple of (found_gnd_ids, was_from_cache)
        """
        from datetime import datetime, timedelta
        
        # Step 1: Check for existing mapping
        mapping = self.get_search_mapping(search_term, suggester_type)
        
        if mapping:
            # Check if mapping is fresh enough
            try:
                last_updated = datetime.fromisoformat(mapping.last_updated)
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
            if suggester_type == "lobid":
                # Lobid results: {term: {keyword: {"gndid": set, ...}}}
                for term_results in results.values():
                    for keyword_data in term_results.values():
                        if "gndid" in keyword_data:
                            gnd_set = keyword_data["gndid"]
                            if isinstance(gnd_set, set):
                                gnd_ids.extend(list(gnd_set))
                            elif isinstance(gnd_set, list):
                                gnd_ids.extend(gnd_set)
                                
            elif suggester_type == "swb":
                # SWB results: similar structure to lobid
                for term_results in results.values():
                    for keyword_data in term_results.values():
                        if "gndid" in keyword_data:
                            gnd_set = keyword_data["gndid"]
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