"""
Unified Knowledge Manager - Consolidates GND cache and DK classifications
Claude Generated - Replaces CacheManager + DKCacheManager with Facts/Mappings separation
"""

import sqlite3
import json
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


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
    
    def __init__(self, db_path: str = "alima_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize unified database schema - Claude Generated"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # === FACTS TABLES (Immutable truths) ===
                
                # 1. GND entries (facts only, no search terms)
                conn.execute("""
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
                conn.execute("""
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
                conn.execute("""
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
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_normalized ON search_mappings(normalized_term)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_term ON search_mappings(search_term)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_gnd_title ON gnd_entries(title)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_classifications_code ON classifications(code)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_classifications_type ON classifications(type)")
                
                self.logger.info("Unified knowledge database schema initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing unified database: {e}")
            raise
    
    # === GND FACTS MANAGEMENT ===
    
    def store_gnd_fact(self, gnd_id: str, gnd_data: Dict[str, Any]):
        """Store GND entry as immutable fact - Claude Generated"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO gnd_entries 
                    (gnd_id, title, description, synonyms, ddcs, ppn, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    gnd_id,
                    gnd_data.get('title', ''),
                    gnd_data.get('description', ''),
                    gnd_data.get('synonyms', ''),
                    gnd_data.get('ddcs', ''),
                    gnd_data.get('ppn', '')
                ))
                
        except Exception as e:
            self.logger.error(f"Error storing GND fact {gnd_id}: {e}")
            raise
    
    def get_gnd_fact(self, gnd_id: str) -> Optional[GNDEntry]:
        """Retrieve GND fact by ID - Claude Generated"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM gnd_entries WHERE gnd_id = ?", (gnd_id,)
                ).fetchone()
                
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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO classifications 
                    (code, type, title, description, parent_code)
                    VALUES (?, ?, ?, ?, ?)
                """, (code, classification_type, title, description, parent_code))
                
        except Exception as e:
            self.logger.error(f"Error storing classification fact {code}: {e}")
            raise
    
    def get_classification_fact(self, code: str, classification_type: str) -> Optional[Classification]:
        """Retrieve classification fact - Claude Generated"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM classifications WHERE code = ? AND type = ?", 
                    (code, classification_type)
                ).fetchone()
                
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("""
                    SELECT * FROM search_mappings 
                    WHERE search_term = ? AND suggester_type = ?
                """, (search_term, suggester_type)).fetchone()
                
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
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO search_mappings 
                    (search_term, normalized_term, suggester_type, found_gnd_ids, 
                     found_classifications, result_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    search_term,
                    normalized_term,
                    suggester_type,
                    json.dumps(found_gnd_ids or []),
                    json.dumps(found_classifications or []),
                    len(found_gnd_ids or []) + len(found_classifications or [])
                ))
                
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
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Exact title match first
                rows = conn.execute("""
                    SELECT * FROM gnd_entries 
                    WHERE title LIKE ? OR title LIKE ?
                    LIMIT ?
                """, (f"%{term}%", f"%{normalized_term}%", min_results * 2)).fetchall()
                
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
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Count facts
                stats['gnd_entries_count'] = conn.execute(
                    "SELECT COUNT(*) FROM gnd_entries"
                ).fetchone()[0]
                
                stats['classifications_count'] = conn.execute(
                    "SELECT COUNT(*) FROM classifications"
                ).fetchone()[0]
                
                # Count mappings
                stats['search_mappings_count'] = conn.execute(
                    "SELECT COUNT(*) FROM search_mappings"
                ).fetchone()[0]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def clear_database(self):
        """Clear all data for fresh start - Claude Generated"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM search_mappings")
                conn.execute("DELETE FROM classifications") 
                conn.execute("DELETE FROM gnd_entries")
                
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM gnd_entries").fetchall()
                
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
        """DKCacheManager compatibility - Claude Generated"""
        # For now, return empty - will be enhanced in Week 2
        return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get unified cache statistics - Claude Generated"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count entries across all tables
                cursor.execute("SELECT COUNT(*) FROM gnd_entries")
                gnd_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM classifications")
                classification_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM search_mappings")
                mapping_count = cursor.fetchone()[0]
                
                total_entries = gnd_count + classification_count + mapping_count
                
                # Get database file size
                import os
                try:
                    size_bytes = os.path.getsize(self.db_path)
                    size_mb = size_bytes / (1024 * 1024)
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
                
                # Store titles as search mapping (simplified for now)
                titles = result.get('titles', [])
                keywords = result.get('keywords', [])
                
                for keyword in keywords:
                    self.update_search_mapping(
                        search_term=keyword,
                        suggester_type="catalog",
                        found_classifications=[{"code": code, "type": classification_type}]
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
        # For SQLite databases, this would typically commit transactions
        # Since we're using 'with' statements that auto-commit, this is a no-op
        # but we provide it for compatibility
        try:
            # Ensure any pending operations are committed
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA integrity_check")
            self.logger.debug("Database integrity check completed")
        except Exception as e:
            self.logger.warning(f"Database save operation warning: {e}")