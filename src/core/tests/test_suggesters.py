#!/usr/bin/env python3
# test_suggesters.py

import unittest
import sys
import os
import json
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Any

# Finde den Pfad zum suggesters-Verzeichnis
current_dir = Path(__file__).parent  # tests/
suggesters_dir = current_dir.parent / "suggesters"  # core/suggesters/

# Füge den Pfad zu suggesters zum Python-Pfad hinzu
sys.path.insert(0, str(suggesters_dir.parent))  # core/

# Jetzt kannst du direkt aus suggesters importieren
from suggesters.base_suggester import BaseSuggester, BaseSuggesterError
from suggesters.lobid_suggester import LobidSuggester
from suggesters.swb_suggester import SWBSuggester
from suggesters.catalog_suggester import CatalogSuggester
from suggesters.meta_suggester import MetaSuggester, SuggesterType
from suggesters.catalog_fallback_suggester import CatalogFallbackSuggester

def print_section(title: str, width: int = 80):
    """Print a section header"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")

def print_test_results(results: Dict[str, Any]):
    """Print test results in a nice format"""
    for term, keywords in results.items():
        print(f"Search term: '{term}'")
        print(f"Found {len(keywords)} keywords")
        
        if keywords:
            print("\nSample keywords:")
            for i, (keyword, data) in enumerate(list(keywords.items())[:5]):
                print(f"\n{i+1}. {keyword}")
                print(f"   Count: {data.get('count', 0)}")
                print(f"   GND IDs: {', '.join(data.get('gndid', set()))}")
                print(f"   DDC: {', '.join(data.get('ddc', set()))}")
                print(f"   DK: {', '.join(data.get('dk', set()))}")
                
        print("\n" + "-" * 40)

def print_unified_results(results: List[List]):
    """Print unified results in a nice format"""
    print(f"Found {len(results)} total results")
    
    if results:
        print("\nSample results:")
        for i, result in enumerate(results[:10]):
            print(f"\n{i+1}. {result[0]}")  # Keyword
            print(f"   GND ID: {result[1]}")
            print(f"   DDC: {result[2]}")
            print(f"   DK: {result[3]}")
            print(f"   Count: {result[4]}")
            print(f"   Search term: {result[5]}")

class TestSuggesters(unittest.TestCase):
    """Test cases for the various suggesters"""
    
    # Directory for test data
    test_data_dir = Path(__file__).parent / "test_data"
    
    def setUp(self):
        """Set up test environment"""
        # Create test data directory
        self.test_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Sample search terms for testing
        self.search_terms = ["Supramolekulare Chemie", "Programmierung"]

    def test_lobid_suggester(self):
        """Test LobidSuggester functionality"""
        print_section("Testing LobidSuggester")
        
        try:
            suggester = LobidSuggester(
                data_dir=self.test_data_dir / "lobid",
                debug=True
            )
            
            # Test prepare method (no force)
            suggester.prepare(False)
            
            # Test search method
            results = suggester.search(self.search_terms)
            
            # Print results
            print_test_results(results)
            
            # Basic assertions
            self.assertIsInstance(results, dict)
            
            # Test if all search terms have results
            for term in self.search_terms:
                self.assertIn(term, results)
                
                # Results may be empty for some terms, so we don't assert non-emptiness
                self.assertIsInstance(results[term], dict)
                
                # If there are results, check their structure
                for keyword, data in results[term].items():
                    self.assertIsInstance(keyword, str)
                    self.assertIsInstance(data, dict)
                    
                    # Check for expected keys
                    self.assertIn("count", data)
                    self.assertIn("gndid", data)
                    
                    # Each gndid should be a set (may be empty)
                    self.assertIsInstance(data["gndid"], set)
                    
            # Test get_unified_results method
            unified_results = suggester.get_unified_results(self.search_terms)
            
            # Print unified results
            print("\nUnified Results:")
            print_unified_results(unified_results)
            
            # Basic assertions for unified results
            self.assertIsInstance(unified_results, list)
            
            # Test the structure of unified results
            for result in unified_results:
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 6)  # Keyword, GND ID, DDC, DK, count, search_term
            
        except Exception as e:
            self.fail(f"LobidSuggester test failed with error: {e}")

    def test_swb_suggester(self):
        """Test SWBSuggester functionality"""
        print_section("Testing SWBSuggester")
        
        try:
            suggester = SWBSuggester(
                data_dir=self.test_data_dir / "swb",
                debug=True
            )
            
            # Test prepare method
            suggester.prepare()
            
            # Test search method with fewer items to be faster
            results = suggester.search(self.search_terms, max_pages=2)
            
            # Print results
            print_test_results(results)
            
            # Basic assertions
            self.assertIsInstance(results, dict)
            
            # Test if all search terms have results
            for term in self.search_terms:
                self.assertIn(term, results)
                
                # Results may be empty for some terms, so we don't assert non-emptiness
                self.assertIsInstance(results[term], dict)
                
                # If there are results, check their structure
                for keyword, data in results[term].items():
                    self.assertIsInstance(keyword, str)
                    self.assertIsInstance(data, dict)
                    
                    # Check for expected keys
                    self.assertIn("count", data)
                    self.assertIn("gndid", data)
                    
                    # Each gndid should be a set (may be empty)
                    self.assertIsInstance(data["gndid"], set)
            
            # Test get_unified_results method
            unified_results = suggester.get_unified_results(self.search_terms)
            
            # Print unified results
            print("\nUnified Results:")
            print_unified_results(unified_results)
            
            # Basic assertions for unified results
            self.assertIsInstance(unified_results, list)
            
            # Test the structure of unified results
            for result in unified_results:
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 6)  # Keyword, GND ID, DDC, DK, count, search_term
            
        except Exception as e:
            self.fail(f"SWBSuggester test failed with error: {e}")

    def test_catalog_suggester(self):
        """Test CatalogSuggester functionality"""
        print_section("Testing CatalogSuggester")
        
        try:
            # Create the suggester
            suggester = CatalogSuggester(
                data_dir=self.test_data_dir / "catalog",
                token="",  # No token needed for basic search
                debug=True
            )
            
            # Test prepare method
            suggester.prepare()
            
            # Test search method
            results = suggester.search(self.search_terms)
            
            # Print results
            print_test_results(results)
            
            # Basic assertions
            self.assertIsInstance(results, dict)
            
            # Test if all search terms have results
            for term in self.search_terms:
                self.assertIn(term, results)
                
                # Results may be empty for some terms, so we don't assert non-emptiness
                self.assertIsInstance(results[term], dict)
                
                # If there are results, check their structure
                for keyword, data in results[term].items():
                    self.assertIsInstance(keyword, str)
                    self.assertIsInstance(data, dict)
                    
                    # Check for expected keys
                    self.assertIn("count", data)
                    self.assertIn("gndid", data)
                    self.assertIn("ddc", data)
                    self.assertIn("dk", data)
                    
                    # Each field should be a set (may be empty)
                    self.assertIsInstance(data["gndid"], set)
                    self.assertIsInstance(data["ddc"], set)
                    self.assertIsInstance(data["dk"], set)
            
            # Test get_unified_results method
            unified_results = suggester.get_unified_results(self.search_terms)
            
            # Print unified results
            print("\nUnified Results:")
            print_unified_results(unified_results)
            
            # Basic assertions for unified results
            self.assertIsInstance(unified_results, list)
            
            # Test the structure of unified results
            for result in unified_results:
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 6)  # Keyword, GND ID, DDC, DK, count, search_term
            
        except Exception as e:
            self.fail(f"CatalogSuggester test failed with error: {e}")

    def test_catalog_fallback_suggester(self):
        """Test CatalogFallbackSuggester functionality"""
        print_section("Testing CatalogFallbackSuggester")
        
        try:
            # Create the suggester
            suggester = CatalogFallbackSuggester(
                data_dir=self.test_data_dir / "catalog_fallback",
                debug=True
            )
            
            # Test prepare method
            suggester.prepare()
            
            # Test search method with fewer items to be faster
            results = suggester.search(["Supramolekulare Chemie"])  # Just one term for faster testing
            
            # Print results
            print_test_results(results)
            
            # Basic assertions
            self.assertIsInstance(results, dict)
            
            # Test if all search terms have results
            self.assertIn("Supramolekulare Chemie", results)
            
            # Results may be empty, so we don't assert non-emptiness
            self.assertIsInstance(results["Supramolekulare Chemie"], dict)
            
            # If there are results, check their structure
            for keyword, data in results["Supramolekulare Chemie"].items():
                self.assertIsInstance(keyword, str)
                self.assertIsInstance(data, dict)
                
                # Check for expected keys
                self.assertIn("count", data)
                self.assertIn("gndid", data)
                self.assertIn("ddc", data)
                self.assertIn("dk", data)
                
                # Each field should be a set (may be empty)
                self.assertIsInstance(data["gndid"], set)
                self.assertIsInstance(data["ddc"], set)
                self.assertIsInstance(data["dk"], set)
            
            # Test get_unified_results method
            unified_results = suggester.get_unified_results(["Supramolekulare Chemie"])
            
            # Print unified results
            print("\nUnified Results:")
            print_unified_results(unified_results)
            
            # Basic assertions for unified results
            self.assertIsInstance(unified_results, list)
            
            # Test the structure of unified results
            for result in unified_results:
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 6)  # Keyword, GND ID, DDC, DK, count, search_term
            
        except Exception as e:
            self.fail(f"CatalogFallbackSuggester test failed with error: {e}")
            
    def test_meta_suggester(self):
        """Test MetaSuggester functionality"""
        print_section("Testing MetaSuggester")
        
        try:
            # Create the meta suggester with only one suggester type for faster testing
            suggester = MetaSuggester(
                suggester_type=SuggesterType.LOBID,  # Use only LOBID for faster testing
                data_dir=self.test_data_dir / "meta",
                catalog_token="",  # No token needed for this test
                debug=True
            )
            
            # Test prepare method
            suggester.prepare()
            
            # Test search method
            results = suggester.search(self.search_terms)
            
            # Print results
            print_test_results(results)
            
            # Basic assertions
            self.assertIsInstance(results, dict)
            
            # Test if all search terms have results
            for term in self.search_terms:
                self.assertIn(term, results)
                
                # Results may be empty for some terms, so we don't assert non-emptiness
                self.assertIsInstance(results[term], dict)
                
                # If there are results, check their structure
                for keyword, data in results[term].items():
                    self.assertIsInstance(keyword, str)
                    self.assertIsInstance(data, dict)
                    
                    # Check for expected keys
                    self.assertIn("count", data)
                    self.assertIn("gndid", data)
                    self.assertIn("ddc", data)
                    self.assertIn("dk", data)
                    
                    # Each field should be a set (may be empty)
                    self.assertIsInstance(data["gndid"], set)
                    self.assertIsInstance(data["ddc"], set)
                    self.assertIsInstance(data["dk"], set)
            
            # Test search_unified method
            unified_results = suggester.search_unified(self.search_terms)
            
            # Print unified results
            print("\nUnified Results:")
            print_unified_results(unified_results)
            
            # Basic assertions for unified results
            self.assertIsInstance(unified_results, list)
            
            # Test the structure of unified results
            for result in unified_results:
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 6)  # Keyword, GND ID, DDC, DK, count, search_term
            
            # Test with a different suggester type
            print("\nTesting MetaSuggester with SWB suggester type")
            suggester = MetaSuggester(
                suggester_type=SuggesterType.SWB,
                data_dir=self.test_data_dir / "meta",
                debug=True
            )
            
            # Just a quick search to verify it works
            results = suggester.search(["Supramolekulare Chemie"])
            self.assertIsInstance(results, dict)
            self.assertIn("Supramolekulare Chemie", results)
            
        except Exception as e:
            self.fail(f"MetaSuggester test failed with error: {e}")

def run_simple_test():
    """Run a simple test with the MetaSuggester"""
    print_section("Running Simple MetaSuggester Test")
    
    # Create the meta suggester with all suggester types
    suggester = MetaSuggester(
        suggester_type=SuggesterType.ALL,
        data_dir=Path("test_data") / "meta",
        catalog_token="",  # No token needed for this test
        debug=True
    )
    
    # Test search method with a simple term
    search_terms = ["Supramolekulare Chemie"]
    
    print(f"Searching for: {search_terms}")
    results = suggester.search(search_terms)
    
    # Print results
    print_test_results(results)
    
    # Test unified results
    unified_results = suggester.search_unified(search_terms)
    
    # Print unified results
    print("\nUnified Results:")
    print_unified_results(unified_results)
    
    # Save results to JSON for reference
    Path("test_results").mkdir(exist_ok=True)
    
    # Convert sets to lists for JSON serialization
    serializable_results = {}
    for term, keywords in results.items():
        serializable_results[term] = {}
        for keyword, data in keywords.items():
            serializable_results[term][keyword] = {
                key: list(value) if isinstance(value, set) else value 
                for key, value in data.items()
            }
    
    with open("test_results/meta_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    with open("test_results/unified_results.json", "w", encoding="utf-8") as f:
        json.dump(unified_results, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to test_results/meta_results.json and test_results/unified_results.json")

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        run_simple_test()
    else:
        # Run the unit tests
        unittest.main()
