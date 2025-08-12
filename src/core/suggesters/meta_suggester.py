#!/usr/bin/env python3
# meta_suggester.py

from typing import List, Dict, Any, Set, Optional, Union, Tuple
from pathlib import Path
from enum import Enum
import json
import time
import logging

from .base_suggester import BaseSuggester, BaseSuggesterError
from .lobid_suggester import LobidSuggester
from .swb_suggester import SWBSuggester
from .biblio_suggester import BiblioSuggester


class SuggesterType(Enum):
    """Enum for the different types of suggesters available."""

    LOBID = "lobid"
    SWB = "swb"
    CATALOG = "catalog"  # Now uses BiblioSuggester
    ALL = "all"


class MetaSuggesterError(BaseSuggesterError):
    """Exception raised for errors in the MetaSuggester."""

    pass


class MetaSuggester(BaseSuggester):
    """
    Meta suggester that combines results from multiple suggesters.
    Provides a unified interface for accessing different keyword suggestion sources.
    """

    def __init__(
        self,
        suggester_type: SuggesterType = SuggesterType.ALL,
        data_dir: Optional[Union[str, Path]] = None,
        catalog_token: str = "",
        debug: bool = False,
        catalog_search_url: str = "",
        catalog_details: str = "",
    ):
        """
        Initialize the meta suggester.

        Args:
            suggester_type: Type of suggester to use (ALL, LOBID, SWB, CATALOG)
            data_dir: Directory for data storage (optional)
            catalog_token: API token for catalog access (optional)
            debug: Whether to enable debug output
            catalog_search_url: URL for catalog search API (optional)
            catalog_details: URL for catalog details API (optional)
        """
        super().__init__(data_dir, debug)

        # Configure logging
        self.logger = logging.getLogger("meta_suggester")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.suggester_type = suggester_type
        self.suggesters = {}
        
        # Week 2: Initialize unified knowledge manager for mapping-first search
        from ..unified_knowledge_manager import UnifiedKnowledgeManager
        self.ukm = UnifiedKnowledgeManager()
        self.enable_mapping_search = True  # Can be disabled for fallback
        self.mapping_max_age_hours = 24    # Configurable cache age
        self.debug_mapping = debug         # Debug flag for mapping output

        # Initialize the appropriate suggesters
        if suggester_type in [SuggesterType.LOBID, SuggesterType.ALL]:
            self.suggesters[SuggesterType.LOBID] = LobidSuggester(
                data_dir=(self.data_dir / "lobid") if data_dir else None, debug=debug
            )

        if suggester_type in [SuggesterType.SWB, SuggesterType.ALL]:
            self.suggesters[SuggesterType.SWB] = SWBSuggester(
                data_dir=(self.data_dir / "swb") if data_dir else None, debug=debug
            )

        if suggester_type in [SuggesterType.CATALOG, SuggesterType.ALL]:
            self.suggesters[SuggesterType.CATALOG] = BiblioSuggester(
                data_dir=(self.data_dir / "catalog") if data_dir else None,
                token=catalog_token,
                debug=debug,
                catalog_search_url=catalog_search_url,
                catalog_details=catalog_details,
            )

        # BiblioSuggester handles all catalog operations (unified approach)

        # Connect signals from suggesters
        for suggester in self.suggesters.values():
            suggester.currentTerm.connect(self.currentTerm)

    def prepare(self, force_download: bool = False) -> None:
        """
        Prepare all suggesters.

        Args:
            force_download: Whether to force data download/preparation
        """
        for name, suggester in self.suggesters.items():
            self.logger.info(f"Preparing {name.value} suggester")
            suggester.prepare(force_download)

    def search(self, terms: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Week 2: Search with mappings-first strategy, fallback to live search - Claude Generated

        Args:
            terms: List of search terms

        Returns:
            Dictionary with structure:
            {
                search_term: {
                    keyword: {
                        "count": int,
                        "gndid": set,
                        "ddc": set,
                        "dk": set
                    }
                }
            }
        """
        combined_results = {}

        # Initialize empty results for each term
        for term in terms:
            combined_results[term] = {}

        # Call each suggester with mapping-first logic
        for suggester_type, suggester in self.suggesters.items():
            self.logger.info(f"Searching with {suggester_type.value} suggester")
            
            # Week 2: Try mappings first if enabled
            if self.enable_mapping_search:
                mapping_hits = 0
                live_searches = 0
                
                for term in terms:
                    # Check for cached mapping first
                    cached_gnd_ids, was_cached = self.ukm.search_with_mappings_first(
                        search_term=term,
                        suggester_type=suggester_type.value,
                        max_age_hours=self.mapping_max_age_hours,
                        live_search_fallback=lambda t: suggester.search([t])
                    )
                    
                    if was_cached:
                        mapping_hits += 1
                        # Convert cached GND IDs to results format
                        if cached_gnd_ids:
                            self._add_cached_results_to_combined(
                                combined_results, term, cached_gnd_ids, suggester_type.value
                            )
                    else:
                        live_searches += 1
                        # Fallback already executed by search_with_mappings_first
                        # Results should have been updated in mapping, now get fresh live results
                        try:
                            live_results = suggester.search([term])
                            self._merge_suggester_results(combined_results, {term: live_results.get(term, {})}, term)
                        except Exception as e:
                            self.logger.error(f"Live search fallback failed for {term}: {e}")
                
                if self.debug_mapping:
                    self.logger.info(f"📊 {suggester_type.value}: {mapping_hits} mapping hits, {live_searches} live searches")
            else:
                # Traditional search without mappings
                try:
                    suggester_results = suggester.search(terms)
                    for term in terms:
                        if term in suggester_results:
                            self._merge_suggester_results(combined_results, suggester_results, term)
                except Exception as e:
                    self.logger.error(f"Error searching with {suggester_type.value} suggester: {e}")

        return combined_results
    
    def _add_cached_results_to_combined(self, combined_results: Dict, term: str, 
                                      gnd_ids: List[str], suggester_type: str):
        """Add cached GND results to combined results format - Claude Generated"""
        try:
            for gnd_id in gnd_ids:
                # Get GND entry details from unified knowledge manager
                gnd_entry = self.ukm.get_gnd_fact(gnd_id)
                if gnd_entry:
                    keyword = gnd_entry.title
                    
                    if keyword not in combined_results[term]:
                        combined_results[term][keyword] = {
                            "count": 1,  # Default count for cached entries
                            "gndid": {gnd_id},
                            "ddc": set(),
                            "dk": set(),
                        }
                    else:
                        combined_results[term][keyword]["gndid"].add(gnd_id)
                        
        except Exception as e:
            self.logger.error(f"Error adding cached results: {e}")
    
    def _merge_suggester_results(self, combined_results: Dict, suggester_results: Dict, term: str):
        """Merge individual suggester results into combined results - Claude Generated"""
        if term not in suggester_results:
            return
            
        term_results = suggester_results[term]

        # For each keyword from this suggester
        for keyword, data in term_results.items():
            # If keyword not in combined results yet, add it
            if keyword not in combined_results[term]:
                combined_results[term][keyword] = {
                    "count": data.get("count", 1),
                    "gndid": data.get("gndid", set()),
                    "ddc": data.get("ddc", set()),
                    "dk": data.get("dk", set()),
                }
            else:
                # Update existing entry
                existing = combined_results[term][keyword]

                # Use max of counts
                existing["count"] = max(
                    existing["count"], data.get("count", 1)
                )

                # Update sets by merging
                existing["gndid"].update(data.get("gndid", set()))
                existing["ddc"].update(data.get("ddc", set()))
                existing["dk"].update(data.get("dk", set()))

    def search_unified(self, terms: List[str]) -> List[List]:
        """
        Get unified search results in the specified format.

        Args:
            terms: List of search terms

        Returns:
            List of results, where each result is a list with elements:
            [keyword, gnd_id, ddc, dk, count, search_term]
        """
        return self.get_unified_results(terms)
