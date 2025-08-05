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
        Search for keywords using all configured suggesters and combine the results.

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

        # Call each suggester and merge results
        for suggester_type, suggester in self.suggesters.items():
            self.logger.info(f"Searching with {suggester_type.value} suggester")
            try:
                suggester_results = suggester.search(terms)
#                self.logger.info(
#                    f"Results from {suggester_type.value} suggester: {suggester_results}"
#                )
                
                # Merge results for each term
                for term in terms:
                    if term not in suggester_results:
                        continue

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

            except Exception as e:
                self.logger.error(
                    f"Error searching with {suggester_type.value} suggester: {e}"
                )

        return combined_results

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
