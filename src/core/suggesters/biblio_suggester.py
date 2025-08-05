#!/usr/bin/env python3
"""
Claude Generated - BiblioExtractor Suggester Wrapper

Lightweight wrapper around BiblioExtractor for pipeline integration.
Provides unified catalog search and DK classification functionality.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .base_suggester import BaseSuggester, BaseSuggesterError
from ..biblioextractor import BiblioExtractor


class BiblioSuggesterError(BaseSuggesterError):
    """Exception raised for errors in the BiblioSuggester."""
    pass


class BiblioSuggester(BaseSuggester):
    """
    Claude Generated - Catalog suggester using BiblioExtractor.
    
    Provides both subject search and DK classification capabilities
    through the unified BiblioExtractor interface.
    """
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        token: str = "",
        debug: bool = False,
        catalog_search_url: str = "",
        catalog_details: str = "",
    ):
        """
        Initialize the BiblioSuggester.
        
        Args:
            data_dir: Directory to store cached data
            token: Authentication token for the library API
            debug: Whether to enable debug output
            catalog_search_url: SOAP search endpoint URL
            catalog_details: SOAP details endpoint URL
        """
        super().__init__(data_dir, debug)
        
        # Initialize BiblioExtractor with provided configuration
        self.extractor = BiblioExtractor(
            token=token,
            debug=debug
        )
        
        # Set URLs if provided
        if catalog_search_url:
            self.extractor.SEARCH_URL = catalog_search_url
        if catalog_details:
            self.extractor.DETAILS_URL = catalog_details
            
        self.logger = logging.getLogger("biblio_suggester")
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    def prepare(self, force_download: bool = False) -> None:
        """
        Prepare the suggester. Nothing needed for BiblioExtractor.
        
        Args:
            force_download: Whether to force data download/preparation
        """
        # BiblioExtractor doesn't need preparation
        pass
    
    def search(self, searches: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Search for subjects related to the given search terms.
        
        Args:
            searches: List of search terms
            
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
        try:
            # Use BiblioExtractor's search_subjects method
            results = self.extractor.search_subjects(searches)
            #self.logger.info(f"Search completed for terms: {searches}")
            #self.logger.info(f"Search results: {results}")
                    # Log keys of all entries
            for key, entry in results.items():
                if isinstance(entry, dict):
                    self.logger.info(f"Entry '{key}' keys: {entry.keys()}")
                else:
                    self.logger.info(f"Entry '{key}' type: {type(entry)}")
            # Emit signals for progress tracking
            for search_term in searches:
                self.currentTerm.emit(search_term)
            
            return results
            
        except Exception as e:
            error_msg = f"BiblioSuggester search failed: {str(e)}"
            self.logger.error(error_msg)
            raise BiblioSuggesterError(error_msg) from e
    
    def extract_dk_classifications(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Claude Generated - Extract DK classifications for given keywords.
        
        Args:
            keywords: List of GND keywords to search for
            
        Returns:
            List of DK classification results with metadata
        """
        try:
            return self.extractor.extract_dk_classifications_for_keywords(keywords)
        except Exception as e:
            error_msg = f"DK classification extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise BiblioSuggesterError(error_msg) from e


# For backward compatibility, create an alias
CatalogSuggester = BiblioSuggester