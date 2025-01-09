import unittest
from unittest.mock import Mock, patch
import asyncio
from src.core.search_engine import SearchEngine
from src.core.cache_manager import CacheManager

class TestSearch(unittest.TestCase):
    def setUp(self):
        """Test-Setup vor jedem Test"""
        self.cache_manager = Mock(spec=CacheManager)
        self.search_engine = SearchEngine(self.cache_manager)

    async def async_test(self, coroutine):
        """Hilfsmethode für asynchrone Tests"""
        return await coroutine

    def test_extract_subject_headings(self):
        """Testet die Extraktion von GND-Schlagworten"""
        # Test-Item mit verschiedenen Schlagwort-Varianten
        test_item = {
            "subject": [
                {
                    "type": ["SubjectHeading"],
                    "id": "https://d-nb.info/gnd/123456789",
                    "label": "Test Subject 1"
                },
                {
                    "componentList": [
                        {
                            "type": ["SubjectHeading"],
                            "id": "https://d-nb.info/gnd/987654321",
                            "label": "Test Subject 2"
                        }
                    ]
                }
            ]
        }

        headings = self.search_engine.extract_subject_headings(test_item)
        
        self.assertEqual(len(headings), 2)
        self.assertEqual(headings[0], ("Test Subject 1", "https://d-nb.info/gnd/123456789"))
        self.assertEqual(headings[1], ("Test Subject 2", "https://d-nb.info/gnd/987654321"))

    @patch('requests.get')
    async def test_search_term(self, mock_get):
        """Testet die Suche nach einem einzelnen Begriff"""
        # Mock-Response
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            '{"subject": [{"type": ["SubjectHeading"], "id": "https://d-nb.info/gnd/123", "label": "Test"}]}'
        ]
        mock_get.return_value = mock_response

        # Cache-Miss simulieren
        self.cache_manager.get_cached_results.return_value = None

        results = await self.async_test(
            self.search_engine.search_term("test")
        )

        self.assertIsNotNone(results)
        self.assertIn('headings', results)
        self.assertIn('counter', results)

    async def test_search_multiple_terms(self):
        """Testet die Suche nach mehreren Begriffen"""
        terms = ["test1", "test2"]
        
        # Mock für search_term
        async def mock_search_term(term):
            return {
                'headings': {(f"Result for {term}", f"gnd/{term}")},
                'counter': {(f"Result for {term}", f"gnd/{term}"): 1},
                'total': 1
            }

        with patch.object(self.search_engine, 'search_term', side_effect=mock_search_term):
            results = await self.async_test(
                self.search_engine.search(terms)
            )

        self.assertIn('exact_matches', results)
        self.assertIn('frequent_matches', results)

    def test_process_results(self):
        """Testet die Verarbeitung der Suchergebnisse"""
        # Setup Test-Ergebnisse
        self.search_engine.current_results = {
            'term_results': {
                'test': {
                    'headings': {("Test Term", "gnd/123")},
                    'counter': {("Test Term", "gnd/123"): 5}
                }
            },
            'total_counter': {("Test Term", "gnd/123"): 5}
        }

        results = self.search_engine.process_results(threshold=1.0)
        
        self.assertIn('exact_matches', results)
        self.assertIn('frequent_matches', results)

    def test_error_handling(self):
        """Testet die Fehlerbehandlung"""
        async def test_error():
            with patch('requests.get', side_effect=Exception("Test Error")):
                return await self.search_engine.search_term("error_test")

        result = asyncio.run(test_error())
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
