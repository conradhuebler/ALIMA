import unittest
import tempfile
import os
from datetime import datetime, timedelta
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from unittest.mock import patch

class TestCache(unittest.TestCase):
    def setUp(self):
        """Test-Setup vor jedem Test"""
        # Temporäre Datenbankdatei erstellen
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.cache_manager = UnifiedKnowledgeManager(db_path=self.temp_db.name)

    def tearDown(self):
        """Aufräumen nach jedem Test"""
        self.cache_manager.db_manager.close_connection()
        os.unlink(self.temp_db.name)

    def test_cache_creation(self):
        """Testet die Erstellung der Cache-Datenbank"""
        # Überprüfe, ob die Tabellen existieren
        tables = self.cache_manager.db_manager.fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [table['name'] for table in tables]

        self.assertIn('searches', table_names)
        self.assertIn('cache_stats', table_names)

    def test_cache_results(self):
        """Testet das Cachen von Suchergebnissen"""
        test_results = {
            'headings': {('Test Term', 'gnd/123')},
            'counter': {('Test Term', 'gnd/123'): 1},
            'total': 1,
            'timestamp': datetime.now().isoformat()
        }

        self.cache_manager.cache_results("test", test_results)
        
        cached = self.cache_manager.get_cached_results("test")
        self.assertIsNotNone(cached)
        self.assertEqual(cached['total'], 1)

    def test_cache_expiration(self):
        """Testet die Cache-Ablauflogik"""
        old_results = {
            'headings': {('Old Term', 'gnd/456')},
            'counter': {('Old Term', 'gnd/456'): 1},
            'total': 1,
            'timestamp': (datetime.now() - timedelta(hours=25)).isoformat()
        }

        self.cache_manager.cache_results("old_test", old_results)
        
        # Sollte None zurückgeben, da der Cache abgelaufen ist
        cached = self.cache_manager.get_cached_results("old_test", max_age_hours=24)
        self.assertIsNone(cached)

    def test_cache_stats(self):
        """Testet die Cache-Statistiken"""
        # Simuliere einige Cache-Zugriffe
        test_results = {
            'headings': set(),
            'counter': {},
            'total': 0,
            'timestamp': datetime.now().isoformat()
        }

        self.cache_manager.cache_results("stats_test", test_results)
        
        # Cache-Hit
        self.cache_manager.get_cached_results("stats_test")
        
        # Cache-Miss
        self.cache_manager.get_cached_results("non_existent")

        stats = self.cache_manager.get_stats(days=1)
        self.assertGreater(stats['cache_hits'], 0)
        self.assertGreater(stats['cache_misses'], 0)

    def test_cleanup_old_entries(self):
        """Testet die Bereinigung alter Cache-Einträge"""
        # Füge alte und neue Einträge hinzu
        old_results = {
            'headings': set(),
            'counter': {},
            'total': 0,
            'timestamp': (datetime.now() - timedelta(days=31)).isoformat()
        }
        new_results = {
            'headings': set(),
            'counter': {},
            'total': 0,
            'timestamp': datetime.now().isoformat()
        }

        self.cache_manager.cache_results("old_entry", old_results)
        self.cache_manager.cache_results("new_entry", new_results)

        # Bereinige alte Einträge
        removed = self.cache_manager.cleanup_old_entries(max_age_days=30)
        self.assertEqual(removed, 1)

        # Überprüfe, ob nur der alte Eintrag entfernt wurde
        self.assertIsNone(self.cache_manager.get_cached_results("old_entry"))
        self.assertIsNotNone(self.cache_manager.get_cached_results("new_entry"))

    def test_cache_conversion(self):
        """Testet die Konvertierung der Cache-Daten"""
        test_data = {
            'headings': {('Test', 'gnd/123')},
            'counter': {('Test', 'gnd/123'): 1},
            'total': 1
        }

        # Test Konvertierung für Speicherung
        storage_data = self.cache_manager._convert_for_storage(test_data)
        self.assertIsInstance(storage_data['headings'], list)
        
        # Test Konvertierung von Speicherung
        restored_data = self.cache_manager._convert_from_storage(storage_data)
        self.assertIsInstance(restored_data['headings'], set)

    def test_error_handling(self):
        """Testet die Fehlerbehandlung"""
        # Test mit ungültiger Datenbankverbindung
        with patch.object(self.cache_manager, 'conn', side_effect=sqlite3.Error):
            result = self.cache_manager.get_cached_results("test")
            self.assertIsNone(result)

    def test_concurrent_access(self):
        """Testet gleichzeitigen Zugriff auf den Cache"""
        import threading

        def cache_operation():
            results = {
                'headings': set(),
                'counter': {},
                'total': 0,
                'timestamp': datetime.now().isoformat()
            }
            self.cache_manager.cache_results(f"thread_test_{threading.get_ident()}", results)

        # Starte mehrere Threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=cache_operation)
            threads.append(t)
            t.start()

        # Warte auf alle Threads
        for t in threads:
            t.join()

        # Überprüfe, ob alle Einträge gespeichert wurden
        cursor = self.cache_manager.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM searches")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 5)

if __name__ == '__main__':
    unittest.main()
