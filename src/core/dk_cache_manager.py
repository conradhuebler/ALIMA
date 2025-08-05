"""
DK Classification Cache Manager - Advanced caching system for DK/RVK classifications
Claude Generated - Implements inverted index and optimized search strategies for classification caching
"""

import sqlite3
import hashlib
import logging
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
import re
from fuzzywuzzy import fuzz


@dataclass
class ClassificationResult:
    """Represents a cached classification result - Claude Generated"""
    dk: str
    classification_type: str
    total_confidence: float
    count: int
    avg_confidence: float
    matched_keywords: List[str]
    titles: List[str]


class DKCacheManager:
    """Advanced caching system for DK/RVK classifications with inverted index - Claude Generated"""
    
    def __init__(self, db_path: str = "dk_classifications.db", cache_dir: str = "dk_cache"):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # In-memory caches
        self.keyword_cache = {}
        self.classification_cache = {}
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with optimized schema - Claude Generated"""
        with sqlite3.connect(self.db_path) as conn:
            # Main tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY,
                    dk TEXT NOT NULL,
                    classification_type TEXT NOT NULL,
                    total_confidence REAL,
                    count INTEGER,
                    avg_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(dk, classification_type)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id INTEGER PRIMARY KEY,
                    keyword TEXT NOT NULL UNIQUE,
                    normalized_keyword TEXT,
                    frequency INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS keyword_classification_map (
                    classification_id INTEGER,
                    keyword_id INTEGER,
                    confidence REAL,
                    PRIMARY KEY (classification_id, keyword_id),
                    FOREIGN KEY (classification_id) REFERENCES classifications(id) ON DELETE CASCADE,
                    FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS classification_titles (
                    id INTEGER PRIMARY KEY,
                    classification_id INTEGER,
                    title TEXT NOT NULL,
                    source_rsn TEXT,
                    FOREIGN KEY (classification_id) REFERENCES classifications(id) ON DELETE CASCADE
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_classifications_dk ON classifications(dk)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_classifications_type ON classifications(classification_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keywords_normalized ON keywords(normalized_keyword)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keyword_map_keyword ON keyword_classification_map(keyword_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_keyword_map_classification ON keyword_classification_map(classification_id)")
            
            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS keywords_fts USING fts5(
                    keyword, normalized_keyword,
                    content='keywords'
                )
            """)
            
            # Triggers to maintain FTS index
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS keywords_ai AFTER INSERT ON keywords BEGIN
                    INSERT INTO keywords_fts(rowid, keyword, normalized_keyword) 
                    VALUES (new.id, new.keyword, new.normalized_keyword);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS keywords_ad AFTER DELETE ON keywords BEGIN
                    INSERT INTO keywords_fts(keywords_fts, rowid, keyword, normalized_keyword) 
                    VALUES('delete', old.id, old.keyword, old.normalized_keyword);
                END
            """)
            
            conn.commit()
    
    def normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword for fuzzy matching - Claude Generated"""
        # Remove GND-ID suffixes
        if "(GND-ID:" in keyword:
            keyword = keyword.split("(GND-ID:")[0].strip()
        
        # Convert to lowercase, remove special chars
        normalized = re.sub(r'[^\w\s]', ' ', keyword.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def store_classification_results(self, results: List[Dict[str, Any]]):
        """Store classification results with inverted index - Claude Generated"""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                try:
                    dk = result.get("dk", "")
                    classification_type = result.get("classification_type", "DK")
                    count = result.get("count", 0)
                    total_confidence = result.get("total_confidence", 0.0)
                    avg_confidence = result.get("avg_confidence", 0.0)
                    keywords = result.get("keywords", [])
                    titles = result.get("titles", [])
                    
                    # Insert or update classification
                    cursor = conn.execute("""
                        INSERT OR REPLACE INTO classifications 
                        (dk, classification_type, total_confidence, count, avg_confidence, last_accessed)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (dk, classification_type, total_confidence, count, avg_confidence))
                    
                    classification_id = cursor.lastrowid
                    if cursor.rowcount == 0:  # UPDATE case
                        classification_id = conn.execute(
                            "SELECT id FROM classifications WHERE dk = ? AND classification_type = ?",
                            (dk, classification_type)
                        ).fetchone()[0]
                    
                    # Store keywords and create inverted index
                    for keyword in keywords:
                        normalized = self.normalize_keyword(keyword)
                        
                        # Insert or update keyword
                        cursor = conn.execute("""
                            INSERT OR IGNORE INTO keywords (keyword, normalized_keyword)
                            VALUES (?, ?)
                        """, (keyword, normalized))
                        
                        # Update frequency if keyword exists
                        conn.execute("""
                            UPDATE keywords SET frequency = frequency + 1
                            WHERE keyword = ?
                        """, (keyword,))
                        
                        # Get keyword ID
                        keyword_id = conn.execute(
                            "SELECT id FROM keywords WHERE keyword = ?", (keyword,)
                        ).fetchone()[0]
                        
                        # Create mapping
                        conn.execute("""
                            INSERT OR REPLACE INTO keyword_classification_map
                            (classification_id, keyword_id, confidence)
                            VALUES (?, ?, ?)
                        """, (classification_id, keyword_id, avg_confidence))
                    
                    # Store titles
                    for title in titles:
                        if title.strip():
                            conn.execute("""
                                INSERT OR IGNORE INTO classification_titles
                                (classification_id, title)
                                VALUES (?, ?)
                            """, (classification_id, title.strip()))
                    
                except Exception as e:
                    self.logger.error(f"Error storing classification result {result}: {e}")
            
            conn.commit()
    
    def search_by_keywords(self, search_terms: List[str], fuzzy_threshold: int = 80) -> List[ClassificationResult]:
        """Search classifications by keywords with exact and fuzzy matching - Claude Generated"""
        if not search_terms:
            return []
        
        # Create cache key
        cache_key = hashlib.md5("|".join(sorted(search_terms)).encode()).hexdigest()
        
        # Check persistent cache first
        cached_result = self._load_from_persistent_cache(cache_key)
        if cached_result:
            return cached_result
        
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # 1. Exact keyword matching
            placeholders = ','.join('?' * len(search_terms))
            exact_query = f"""
                SELECT DISTINCT c.*, 
                       GROUP_CONCAT(k.keyword) as matched_keywords,
                       GROUP_CONCAT(DISTINCT ct.title) as titles
                FROM classifications c
                JOIN keyword_classification_map kcm ON c.id = kcm.classification_id
                JOIN keywords k ON kcm.keyword_id = k.id
                LEFT JOIN classification_titles ct ON c.id = ct.classification_id
                WHERE k.keyword IN ({placeholders}) OR k.normalized_keyword IN ({placeholders})
                GROUP BY c.id
                ORDER BY c.count DESC, c.avg_confidence DESC
            """
            
            # Prepare search terms and normalized versions
            all_search_terms = search_terms + [self.normalize_keyword(term) for term in search_terms]
            
            for row in conn.execute(exact_query, all_search_terms):
                result = ClassificationResult(
                    dk=row['dk'],
                    classification_type=row['classification_type'],
                    total_confidence=row['total_confidence'] or 0.0,
                    count=row['count'] or 0,
                    avg_confidence=row['avg_confidence'] or 0.0,
                    matched_keywords=row['matched_keywords'].split(',') if row['matched_keywords'] else [],
                    titles=row['titles'].split(',') if row['titles'] else []
                )
                results.append(result)
            
            # 2. Fuzzy matching for missed terms
            if len(results) < 10:  # Only do fuzzy search if we need more results
                for search_term in search_terms:
                    fuzzy_matches = self._fuzzy_search_keywords(conn, search_term, fuzzy_threshold)
                    
                    for match in fuzzy_matches:
                        # Avoid duplicates
                        if not any(r.dk == match.dk and r.classification_type == match.classification_type for r in results):
                            results.append(match)
        
        # Sort by relevance
        results.sort(key=lambda x: (x.count, x.avg_confidence), reverse=True)
        
        # Cache the results
        self._save_to_persistent_cache(cache_key, results)
        
        return results[:50]  # Limit to top 50 results
    
    def _fuzzy_search_keywords(self, conn, search_term: str, threshold: int) -> List[ClassificationResult]:
        """Perform fuzzy search using FTS5 and similarity matching - Claude Generated"""
        results = []
        
        # Use FTS5 for initial candidate selection
        fts_query = """
            SELECT DISTINCT c.*, k.keyword,
                   kcm.confidence,
                   GROUP_CONCAT(DISTINCT ct.title) as titles
            FROM keywords_fts 
            JOIN keywords k ON keywords_fts.rowid = k.id
            JOIN keyword_classification_map kcm ON k.id = kcm.keyword_id
            JOIN classifications c ON kcm.classification_id = c.id
            LEFT JOIN classification_titles ct ON c.id = ct.classification_id
            WHERE keywords_fts MATCH ?
            GROUP BY c.id
            ORDER BY c.count DESC, c.avg_confidence DESC
            LIMIT 20
        """
        
        # Try different FTS5 search strategies
        search_variations = [
            f'"{search_term}"',  # Exact phrase
            f'{search_term}*',   # Prefix search
            search_term.replace(' ', ' AND ')  # AND search
        ]
        
        for search_variation in search_variations:
            try:
                for row in conn.execute(fts_query, (search_variation,)):
                    # Calculate similarity
                    similarity = fuzz.ratio(search_term.lower(), row['keyword'].lower())
                    
                    if similarity >= threshold:
                        result = ClassificationResult(
                            dk=row['dk'],
                            classification_type=row['classification_type'],
                            total_confidence=row['total_confidence'] or 0.0,
                            count=row['count'] or 0,
                            avg_confidence=row['avg_confidence'] or 0.0,
                            matched_keywords=[row['keyword']],
                            titles=row['titles'].split(',') if row['titles'] else []
                        )
                        results.append(result)
                        
            except sqlite3.OperationalError:
                # Skip if FTS query fails
                continue
        
        return results
    
    @lru_cache(maxsize=1000)
    def get_keywords_for_classification(self, dk: str, classification_type: str = "DK") -> List[str]:
        """Get cached keywords for a specific classification - Claude Generated"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT k.keyword
                FROM keywords k
                JOIN keyword_classification_map kcm ON k.id = kcm.keyword_id
                JOIN classifications c ON kcm.classification_id = c.id
                WHERE c.dk = ? AND c.classification_type = ?
                ORDER BY kcm.confidence DESC, k.frequency DESC
            """
            
            return [row[0] for row in conn.execute(query, (dk, classification_type))]
    
    def _save_to_persistent_cache(self, cache_key: str, results: List[ClassificationResult]):
        """Save search results to persistent cache - Claude Generated"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            self.logger.warning(f"Failed to save persistent cache: {e}")
    
    def _load_from_persistent_cache(self, cache_key: str) -> Optional[List[ClassificationResult]]:
        """Load search results from persistent cache - Claude Generated"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                # Check cache age (expire after 24 hours)
                if cache_file.stat().st_mtime > (cache_file.stat().st_mtime - 86400):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load persistent cache: {e}")
        
        return None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics - Claude Generated"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            stats['classifications_count'] = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
            stats['keywords_count'] = conn.execute("SELECT COUNT(*) FROM keywords").fetchone()[0]
            stats['mappings_count'] = conn.execute("SELECT COUNT(*) FROM keyword_classification_map").fetchone()[0]
            stats['titles_count'] = conn.execute("SELECT COUNT(*) FROM classification_titles").fetchone()[0]
            
            # Most frequent keywords
            stats['top_keywords'] = conn.execute("""
                SELECT keyword, frequency FROM keywords 
                ORDER BY frequency DESC LIMIT 10
            """).fetchall()
            
            # Classification type distribution
            stats['type_distribution'] = conn.execute("""
                SELECT classification_type, COUNT(*) as count
                FROM classifications
                GROUP BY classification_type
                ORDER BY count DESC
            """).fetchall()
            
            return stats
    
    def cleanup_old_cache(self, days: int = 30):
        """Clean up old cache entries - Claude Generated"""
        with sqlite3.connect(self.db_path) as conn:
            # Remove old classifications not accessed recently
            conn.execute("""
                DELETE FROM classifications 
                WHERE last_accessed < datetime('now', '-{} days')
            """.format(days))
            
            # Clean up orphaned records
            conn.execute("""
                DELETE FROM keyword_classification_map 
                WHERE classification_id NOT IN (SELECT id FROM classifications)
            """)
            
            conn.execute("""
                DELETE FROM keywords 
                WHERE id NOT IN (SELECT DISTINCT keyword_id FROM keyword_classification_map)
            """)
            
            conn.commit()
        
        # Clean up persistent cache files
        import time
        cutoff_time = time.time() - (days * 86400)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()