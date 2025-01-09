import sqlite3
import json
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import os

class CacheManager:
    """Verwaltet das Caching von Suchergebnissen"""
    
    def __init__(self, db_path: str = "search_cache.db"):
        """
        Initialisiert den Cache-Manager.
        
        Args:
            db_path: Pfad zur SQLite-Datenbank
        """
        # Logger initialisieren
        self.logger = logging.getLogger(__name__)
        
        # Stelle sicher, dass der Datenbankpfad existiert
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
        # Logging konfigurieren
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_connection(self) -> sqlite3.Connection:
        """
        Initialisiert die Datenbankverbindung mit praktischen Defaults.
        
        Returns:
            SQLite Verbindung
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Ermöglicht dict-ähnlichen Zugriff auf Rows
        return conn

    def create_tables(self) -> None:
        """Erstellt die notwendigen Tabellen in der Datenbank."""
        try:
            with self.conn:
                # Haupttabelle für Suchergebnisse
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS searches (
                        search_term TEXT PRIMARY KEY,
                        results BLOB,
                        timestamp DATETIME,
                        hit_count INTEGER DEFAULT 0,
                        last_accessed DATETIME
                    )
                ''')
                
                # Tabelle für Statistiken
                self.conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache_stats (
                        stat_date DATE PRIMARY KEY,
                        total_searches INTEGER,
                        cache_hits INTEGER,
                        cache_misses INTEGER
                    )
                ''')
                
                self.logger.info("Datenbanktabellen erfolgreich erstellt oder bereits vorhanden")
        except sqlite3.Error as e:
            self.logger.error(f"Fehler beim Erstellen der Tabellen: {e}")
            raise

    def _convert_for_storage(self, results: Dict) -> Dict:
        """
        Konvertiert Sets und Counter in JSON-serialisierbare Formate.
        
        Args:
            results: Dictionary mit Suchergebnissen
            
        Returns:
            JSON-serialisierbares Dictionary
        """
        return {
            'headings': list(results['headings']),
            'counter': {str(k): v for k, v in results['counter'].items()},
            'total': results['total'],
            'timestamp': results.get('timestamp', datetime.now().isoformat())
        }

    def _convert_from_storage(self, stored_results: Dict) -> Dict:
        """
        Konvertiert gespeicherte Daten zurück in Sets und Counter.
        
        Args:
            stored_results: Gespeichertes Dictionary
            
        Returns:
            Dictionary mit wiederhergestellten Datentypen
        """
        try:
            return {
                'headings': set(tuple(h) for h in stored_results['headings']),
                'counter': {
                    tuple(k.strip('()').split(', ')): v
                    for k, v in stored_results['counter'].items()
                },
                'total': stored_results['total'],
                'timestamp': stored_results.get('timestamp')
            }
        except Exception as e:
            self.logger.error(f"Fehler bei der Konvertierung der gespeicherten Daten: {e}")
            raise

    def get_cached_results(self, search_term: str, max_age_hours: int = 24) -> Optional[Dict]:
        """
        Holt gecachte Ergebnisse, wenn sie nicht älter als die angegebene Zeit sind.
        
        Args:
            search_term: Suchbegriff
            max_age_hours: Maximales Alter der Cache-Einträge in Stunden
            
        Returns:
            Dictionary mit Suchergebnissen oder None
        """
        try:
            with self.conn:
                cursor = self.conn.execute(
                    '''
                    SELECT results, timestamp, hit_count 
                    FROM searches 
                    WHERE search_term = ?
                    ''',
                    (search_term,)
                )
                result = cursor.fetchone()

                if result:
                    results, timestamp, hit_count = result
                    cached_time = datetime.fromisoformat(timestamp)
                    
                    if datetime.now() - cached_time < timedelta(hours=max_age_hours):
                        # Update Zugriffstatistiken
                        self.conn.execute(
                            '''
                            UPDATE searches 
                            SET hit_count = hit_count + 1,
                                last_accessed = ? 
                            WHERE search_term = ?
                            ''',
                            (datetime.now().isoformat(), search_term)
                        )
                        
                        self._update_stats('hits')
                        self.logger.info(f"Cache-Hit für '{search_term}'")
                        return self._convert_from_storage(json.loads(results))
                    
                    self._update_stats('misses')
                    self.logger.info(f"Cache-Miss (veraltet) für '{search_term}'")
                    return None
                
                self._update_stats('misses')
                self.logger.info(f"Cache-Miss (nicht gefunden) für '{search_term}'")
                return None
                
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen von Cache-Daten: {e}")
            return None

    def cache_results(self, search_term: str, results: Dict) -> None:
        """
        Speichert Ergebnisse im Cache.
        
        Args:
            search_term: Suchbegriff
            results: Dictionary mit Suchergebnissen
        """
        try:
            serializable_results = self._convert_for_storage(results)
            
            with self.conn:
                self.conn.execute(
                    '''
                    INSERT OR REPLACE INTO searches 
                    (search_term, results, timestamp, last_accessed, hit_count)
                    VALUES (?, ?, ?, ?, 0)
                    ''',
                    (
                        search_term,
                        json.dumps(serializable_results),
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    )
                )
                
            self.logger.info(f"Ergebnisse für '{search_term}' erfolgreich gecacht")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Cachen der Ergebnisse: {e}")
            raise

    def _update_stats(self, stat_type: str) -> None:
        """
        Aktualisiert die Cache-Statistiken.
        
        Args:
            stat_type: Art der Statistik ('hits' oder 'misses')
        """
        today = datetime.now().date()
        
        try:
            with self.conn:
                # Prüfe, ob bereits ein Eintrag für heute existiert
                cursor = self.conn.execute(
                    'SELECT * FROM cache_stats WHERE stat_date = ?',
                    (today,)
                )
                
                if cursor.fetchone():
                    # Update existierenden Eintrag
                    if stat_type == 'hits':
                        self.conn.execute(
                            '''
                            UPDATE cache_stats 
                            SET cache_hits = cache_hits + 1,
                                total_searches = total_searches + 1
                            WHERE stat_date = ?
                            ''',
                            (today,)
                        )
                    else:
                        self.conn.execute(
                            '''
                            UPDATE cache_stats 
                            SET cache_misses = cache_misses + 1,
                                total_searches = total_searches + 1
                            WHERE stat_date = ?
                            ''',
                            (today,)
                        )
                else:
                    # Erstelle neuen Eintrag
                    hits = 1 if stat_type == 'hits' else 0
                    misses = 1 if stat_type == 'misses' else 0
                    self.conn.execute(
                        '''
                        INSERT INTO cache_stats 
                        (stat_date, total_searches, cache_hits, cache_misses)
                        VALUES (?, 1, ?, ?)
                        ''',
                        (today, hits, misses)
                    )
        
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren der Statistiken: {e}")

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Holt Cache-Statistiken für die letzten X Tage.
        
        Args:
            days: Anzahl der Tage für die Statistik
            
        Returns:
            Dictionary mit Statistiken
        """
        start_date = (datetime.now() - timedelta(days=days)).date()
        
        try:
            with self.conn:
                cursor = self.conn.execute(
                    '''
                    SELECT * FROM cache_stats 
                    WHERE stat_date >= ? 
                    ORDER BY stat_date DESC
                    ''',
                    (start_date,)
                )
                
                stats = cursor.fetchall()
                
                return {
                    'total_searches': sum(row['total_searches'] for row in stats),
                    'cache_hits': sum(row['cache_hits'] for row in stats),
                    'cache_misses': sum(row['cache_misses'] for row in stats),
                    'hit_rate': sum(row['cache_hits'] for row in stats) / 
                               sum(row['total_searches'] for row in stats) 
                               if stats else 0,
                    'daily_stats': [dict(row) for row in stats]
                }
                
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Statistiken: {e}")
            return {}

    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """
        Entfernt alte Cache-Einträge.
        
        Args:
            max_age_days: Maximales Alter der Einträge in Tagen
            
        Returns:
            Anzahl der entfernten Einträge
        """
        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        
        try:
            with self.conn:
                cursor = self.conn.execute(
                    'DELETE FROM searches WHERE timestamp < ?',
                    (cutoff_date,)
                )
                
                removed_count = cursor.rowcount
                self.logger.info(f"{removed_count} alte Cache-Einträge entfernt")
                return removed_count
                
        except Exception as e:
            self.logger.error(f"Fehler beim Bereinigen des Caches: {e}")
            return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
