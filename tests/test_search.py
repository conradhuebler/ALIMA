"""Tests for UnifiedKnowledgeManager (current API).

NOTE: The old CacheManager API (searches/cache_stats tables) was replaced
during the August 2025 restructure. These tests cover the current
Facts/Mappings architecture with proper singleton reset + SQLite isolation.
"""
import unittest
import tempfile
import os

try:
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    from src.utils.config_models import DatabaseConfig
    PYQT_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    UnifiedKnowledgeManager = None
    DatabaseConfig = None
    PYQT_IMPORT_ERROR = exc


def _make_sqlite_config(path):
    cfg = DatabaseConfig(db_type='sqlite')
    cfg.sqlite_path = path
    return cfg


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6-backed database stack unavailable: {PYQT_IMPORT_ERROR}")
class TestUnifiedKnowledgeManager(unittest.TestCase):

    def setUp(self):
        UnifiedKnowledgeManager.reset()
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.km = UnifiedKnowledgeManager(database_config=_make_sqlite_config(self.temp_db.name))

    def tearDown(self):
        UnifiedKnowledgeManager.reset()
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass

    def test_schema_tables_exist(self):
        tables = self.km.db_manager.fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
        names = {t['name'] for t in tables}
        for expected in ('gnd_entries', 'classifications', 'search_mappings', 'catalog_dk_cache'):
            self.assertIn(expected, names)

    def test_store_and_retrieve_gnd_fact(self):
        self.km.store_gnd_fact("4123456-7", {"title": "Testschlagwort", "description": "", "synonyms": "", "ddcs": ""})
        entry = self.km.get_gnd_fact("4123456-7")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.title, "Testschlagwort")

    def test_get_gnd_fact_missing_returns_none(self):
        self.assertIsNone(self.km.get_gnd_fact("does-not-exist"))

    def test_get_gnd_facts_batch(self):
        self.km.store_gnd_fact("1111111-1", {"title": "Alpha", "description": "", "synonyms": "", "ddcs": ""})
        self.km.store_gnd_fact("2222222-2", {"title": "Beta",  "description": "", "synonyms": "", "ddcs": ""})
        batch = self.km.get_gnd_facts_batch(["1111111-1", "2222222-2", "9999999-9"])
        self.assertIn("1111111-1", batch)
        self.assertIn("2222222-2", batch)
        self.assertNotIn("9999999-9", batch)

    def test_search_mapping_roundtrip(self):
        self.km.update_search_mapping("Klimawandel", "lobid", found_gnd_ids=["4031483-2"])
        mapping = self.km.get_search_mapping("Klimawandel", "lobid")
        self.assertIsNotNone(mapping)
        self.assertIn("4031483-2", mapping.found_gnd_ids)

    def test_search_mapping_missing_returns_none(self):
        self.assertIsNone(self.km.get_search_mapping("unbekannt", "lobid"))

    def test_store_classification_fact(self):
        self.km.store_classification_fact("004", "DDC", title="Informatik")
        cls = self.km.get_classification_fact("004", "DDC")
        self.assertIsNotNone(cls)
        self.assertEqual(cls.title, "Informatik")

    def test_database_stats(self):
        self.km.store_gnd_fact("5555555-5", {"title": "Stats-Test", "description": "", "synonyms": "", "ddcs": ""})
        stats = self.km.get_database_stats()
        self.assertGreaterEqual(stats.get('gnd_entries_count', 0), 1)

    def test_singleton_behavior(self):
        km2 = UnifiedKnowledgeManager(database_config=_make_sqlite_config(self.temp_db.name))
        self.assertIs(self.km, km2)

    def test_reset_clears_singleton(self):
        UnifiedKnowledgeManager.reset()
        km_new = UnifiedKnowledgeManager(database_config=_make_sqlite_config(self.temp_db.name))
        self.assertIsNotNone(km_new)
        self.km = km_new

    def test_concurrent_access(self):
        import threading
        errors = []
        def store(idx):
            try:
                self.km.store_gnd_fact(f"concurrent-{idx}", {"title": f"Thread {idx}", "description": "", "synonyms": "", "ddcs": ""})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=store, args=(i,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
