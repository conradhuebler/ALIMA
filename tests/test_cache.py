"""Tests for SearchEngine (Qt-signal-based API, replaced async API July 2025)."""
import unittest
from unittest.mock import Mock
from collections import Counter

try:
    from src.core.search_engine import SearchEngine
    from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
    PYQT_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    SearchEngine = None
    UnifiedKnowledgeManager = None
    PYQT_IMPORT_ERROR = exc


@unittest.skipIf(PYQT_IMPORT_ERROR is not None, f"PyQt6-backed search stack unavailable: {PYQT_IMPORT_ERROR}")
class TestSearch(unittest.TestCase):

    def setUp(self):
        self.cache_manager = Mock(spec=UnifiedKnowledgeManager)
        self.search_engine = SearchEngine(self.cache_manager)

    def test_extract_subject_headings(self):
        test_item = {
            "subject": [
                {"type": ["SubjectHeading"], "id": "https://d-nb.info/gnd/123456789", "label": "Test Subject 1"},
                {"componentList": [{"type": ["SubjectHeading"], "id": "https://d-nb.info/gnd/987654321", "label": "Test Subject 2"}]}
            ]
        }
        headings = self.search_engine.extract_subject_headings(test_item)
        self.assertEqual(len(headings), 2)
        self.assertEqual(headings[0], ("Test Subject 1", "https://d-nb.info/gnd/123456789"))
        self.assertEqual(headings[1], ("Test Subject 2", "https://d-nb.info/gnd/987654321"))

    def test_extract_subject_headings_empty(self):
        self.assertEqual(self.search_engine.extract_subject_headings({}), [])

    def test_process_results_empty(self):
        self.search_engine.term_results = {}
        self.search_engine.total_counter = Counter()
        result = self.search_engine.process_results(threshold=1.0)
        self.assertEqual(result["exact_matches"], [])
        self.assertEqual(result["frequent_matches"], [])

    def test_process_results_with_data(self):
        self.search_engine.term_results = {
            'test': {'headings': {("Test Term", "gnd/123")}, 'counter': {("Test Term", "gnd/123"): 5}}
        }
        self.search_engine.total_counter = Counter({("Test Term", "gnd/123"): 5})
        result = self.search_engine.process_results(threshold=1.0)
        self.assertIn('exact_matches', result)
        self.assertIn('frequent_matches', result)

    def test_search_uses_cache(self):
        cached = {'headings': {('Cached Term', 'gnd/999')}, 'counter': {('Cached Term', 'gnd/999'): 1}, 'total': 1}
        self.cache_manager.get_cached_results.return_value = cached
        self.search_engine.term_results = {}
        self.search_engine.total_counter = Counter()
        self.search_engine.pending_requests = 1
        self.search_engine.threshold = 1.0
        self.search_engine.search_term("cached_query")
        self.cache_manager.get_cached_results.assert_called_once_with("cached_query")
        self.assertIn("cached_query", self.search_engine.term_results)

    def test_search_term_no_cache_hit(self):
        self.cache_manager.get_cached_results.return_value = None
        self.search_engine.term_results = {}
        self.search_engine.total_counter = Counter()
        self.search_engine.pending_requests = 1
        self.search_engine.threshold = 1.0
        self.search_engine.search_term("network_query")
        self.assertNotIn("network_query", self.search_engine.term_results)


if __name__ == '__main__':
    unittest.main()
