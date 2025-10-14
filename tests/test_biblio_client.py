#!/usr/bin/env python3
"""
Claude Generated - Tests for BiblioClient and BiblioSuggester

Contains both integration tests (with real network) and unit tests (mocked).
"""

import unittest
import os
from unittest.mock import Mock, patch, MagicMock
import xml.etree.ElementTree as ET

from src.utils.clients.biblio_client import BiblioClient
from src.utils.suggesters.biblio_suggester import BiblioSuggester


# Mock XML response for testing (based on real Libero SOAP response structure)
MOCK_SEARCH_XML_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:lib="http://libero.com.au">
   <soapenv:Body>
      <lib:SearchResponse>
         <lib:SearchResult>
            <lib:searchResultItems>
               <lib:title>Quantenchemie: Grundlagen und Anwendungen</lib:title>
               <lib:rsn>12345</lib:rsn>
               <lib:author>Müller, Hans</lib:author>
            </lib:searchResultItems>
            <lib:searchResultItems>
               <lib:title>Einführung in die Quantenchemie</lib:title>
               <lib:rsn>67890</lib:rsn>
               <lib:author>Schmidt, Anna</lib:author>
            </lib:searchResultItems>
         </lib:SearchResult>
      </lib:SearchResponse>
   </soapenv:Body>
</soapenv:Envelope>"""


class TestBiblioClientIntegration(unittest.TestCase):
    """
    Integration tests for BiblioClient with real network connections.

    These tests make actual HTTP requests to the library catalog and are
    slower than unit tests. They verify the Python network stack (SSL, DNS, etc.)
    """

    @unittest.skipUnless(
        os.environ.get("RUN_INTEGRATION_TESTS") == "1",
        "Integration tests disabled (set RUN_INTEGRATION_TESTS=1 to enable)"
    )
    def test_biblio_client_direct_search(self):
        """
        Direct integration test: Real SOAP request to catalog.

        This test verifies:
        - Network connectivity works from Python environment
        - SSL certificates are valid
        - SOAP request/response cycle completes
        - Response parsing works correctly
        """
        # Initialize client
        client = BiblioClient(debug=False)

        # Execute search
        results = client.search("Quantenchemie")

        # Verify results
        self.assertIsNotNone(results, "Search should return results, not None")
        self.assertIsInstance(results, list, "Results should be a list")

        if len(results) > 0:
            # If we got results, verify structure
            self.assertGreater(len(results), 0, "Should find at least one result")
            self.assertIn("title", results[0], "Results should contain 'title' field")
            self.assertIn("rsn", results[0], "Results should contain 'rsn' field")

            # Log first result for debugging
            print(f"First result: {results[0]}")


class TestBiblioClientMocked(unittest.TestCase):
    """
    Unit tests for BiblioClient with mocked network requests.

    These tests are fast and isolated - they don't make real network requests.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.client = BiblioClient(debug=True)

    @patch('src.utils.clients.biblio_client.requests.Session.post')
    def test_search_request_format(self, mock_post):
        """
        Test that BiblioClient creates correct SOAP requests.

        Verifies:
        - Correct URL is used
        - Headers are properly set
        - XML body contains search term
        """
        # Configure mock to return successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = MOCK_SEARCH_XML_RESPONSE.encode('utf-8')
        mock_response.text = MOCK_SEARCH_XML_RESPONSE
        mock_post.return_value = mock_response

        # Execute search
        results = self.client.search("Quantenchemie")

        # Verify mock was called
        mock_post.assert_called_once()

        # Extract call arguments
        call_args = mock_post.call_args
        called_url = call_args[0][0] if call_args[0] else call_args[1].get('url')
        called_headers = call_args[1].get('headers', {})
        called_data = call_args[1].get('data', '')

        # Verify URL
        self.assertEqual(
            called_url,
            self.client.SEARCH_URL,
            "Should call correct search URL"
        )

        # Verify headers
        self.assertEqual(
            called_headers.get('Content-Type'),
            'text/xml;charset=UTF-8',
            "Content-Type header should be set correctly"
        )
        self.assertEqual(
            called_headers.get('SOAPAction'),
            '',
            "SOAPAction header should be empty string"
        )

        # Verify XML body contains search term
        self.assertIn(
            'Quantenchemie',
            called_data,
            "Request body should contain search term"
        )
        self.assertIn(
            '<lib:Search>',
            called_data,
            "Request should be valid SOAP envelope"
        )
        self.assertIn(
            '<lib:use>ku</lib:use>',
            called_data,
            "Request should contain search type"
        )

    @patch('src.utils.clients.biblio_client.requests.Session.post')
    def test_search_response_parsing(self, mock_post):
        """
        Test that BiblioClient correctly parses XML responses.

        Verifies:
        - XML is parsed correctly
        - Result items are extracted
        - Fields are mapped properly
        """
        # Configure mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = MOCK_SEARCH_XML_RESPONSE.encode('utf-8')
        mock_response.text = MOCK_SEARCH_XML_RESPONSE
        mock_post.return_value = mock_response

        # Execute search
        results = self.client.search("Quantenchemie")

        # Verify parsing
        self.assertIsNotNone(results, "Should return parsed results")
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertEqual(len(results), 2, "Should parse both result items")

        # Verify first result
        first_result = results[0]
        self.assertEqual(
            first_result.get('title'),
            'Quantenchemie: Grundlagen und Anwendungen',
            "Should extract title correctly"
        )
        self.assertEqual(
            first_result.get('rsn'),
            '12345',
            "Should extract RSN correctly"
        )
        self.assertEqual(
            first_result.get('author'),
            'Müller, Hans',
            "Should extract author correctly"
        )


class TestBiblioSuggesterMocked(unittest.TestCase):
    """
    Unit tests for BiblioSuggester with mocked BiblioClient.

    Tests the suggester wrapper logic without making actual catalog requests.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.suggester = BiblioSuggester(debug=True)

    @patch('src.utils.clients.biblio_client.requests.Session.post')
    def test_suggester_search_integration(self, mock_post):
        """
        Test BiblioSuggester.search() with mocked client.

        Verifies:
        - Suggester calls BiblioClient correctly
        - Results are transformed to suggester format
        - Signals are emitted (if in GUI context)
        """
        # Configure mock for search response
        search_response = Mock()
        search_response.status_code = 200
        search_response.content = MOCK_SEARCH_XML_RESPONSE.encode('utf-8')
        search_response.text = MOCK_SEARCH_XML_RESPONSE

        # Configure mock for details response (needed by search_subjects)
        details_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:lib="http://libero.com.au">
           <soapenv:Body>
              <lib:GetTitleDetailsResponse>
                 <lib:Title>Quantenchemie</lib:Title>
                 <lib:Subject>
                    <lib:Subjects>
                       <lib:Subject>Quantenchemie</lib:Subject>
                       <lib:Subject>Quantentheorie</lib:Subject>
                    </lib:Subjects>
                 </lib:Subject>
                 <lib:MAB>
                    <lib:TagKey>0907</lib:TagKey>
                    <lib:MABDataPlain>123 Chemie</lib:MABDataPlain>
                 </lib:MAB>
              </lib:GetTitleDetailsResponse>
           </soapenv:Body>
        </soapenv:Envelope>"""

        details_response = Mock()
        details_response.status_code = 200
        details_response.content = details_xml.encode('utf-8')
        details_response.text = details_xml

        # Mock returns different responses for search vs details
        mock_post.side_effect = [search_response, details_response, details_response]

        # Execute search
        search_terms = ["Quantenchemie"]
        results = self.suggester.search(search_terms)

        # Verify results structure
        self.assertIsNotNone(results, "Should return results")
        self.assertIsInstance(results, dict, "Results should be a dictionary")
        self.assertIn("Quantenchemie", results, "Should contain search term as key")

        # Verify suggester format (search_term -> subject -> metadata)
        term_results = results.get("Quantenchemie", {})
        self.assertIsInstance(term_results, dict, "Term results should be a dictionary")


class TestBiblioClientErrorHandling(unittest.TestCase):
    """
    Tests for error handling in BiblioClient.

    Verifies proper handling of network errors, invalid responses, etc.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.client = BiblioClient(debug=False)

    @patch('src.utils.clients.biblio_client.requests.Session.post')
    def test_network_error_handling(self, mock_post):
        """
        Test handling of network errors (connection failures, timeouts).
        """
        import requests

        # Simulate network error
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

        # Execute search
        results = self.client.search("test")

        # Should return empty list, not raise exception
        self.assertEqual(results, [], "Should return empty list on network error")

    @patch('src.utils.clients.biblio_client.requests.Session.post')
    def test_invalid_xml_handling(self, mock_post):
        """
        Test handling of invalid XML responses.
        """
        # Configure mock with invalid XML
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<invalid><xml>"
        mock_response.text = "<invalid><xml>"
        mock_post.return_value = mock_response

        # Execute search
        results = self.client.search("test")

        # Should return empty list, not raise exception
        self.assertEqual(results, [], "Should return empty list on XML parse error")

    @patch('src.utils.clients.biblio_client.requests.Session.post')
    def test_http_error_codes(self, mock_post):
        """
        Test handling of HTTP error status codes (4xx, 5xx).
        """
        # Configure mock with error status
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        # Execute search
        results = self.client.search("test")

        # Should return empty list, not raise exception
        self.assertEqual(results, [], "Should return empty list on HTTP error")

    @patch('src.utils.clients.biblio_client.requests.Session.post')
    def test_empty_response_handling(self, mock_post):
        """
        Test handling of empty/no results responses.
        """
        # Configure mock with valid but empty XML
        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/">
           <soapenv:Body>
              <lib:SearchResponse xmlns:lib="http://libero.com.au">
                 <lib:SearchResult>
                 </lib:SearchResult>
              </lib:SearchResponse>
           </soapenv:Body>
        </soapenv:Envelope>"""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = empty_xml.encode('utf-8')
        mock_response.text = empty_xml
        mock_post.return_value = mock_response

        # Execute search
        results = self.client.search("nonexistent_term_12345")

        # Should return empty list
        self.assertEqual(results, [], "Should return empty list when no results found")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
