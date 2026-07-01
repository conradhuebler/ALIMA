#!/usr/bin/env python3
"""
Claude Generated - BiblioExtractor Suggester Wrapper

Lightweight wrapper around BiblioExtractor for pipeline integration.
Provides unified catalog search and DK classification functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .base_suggester import BaseSuggester, BaseSuggesterError
from ..clients.biblio_client import BiblioClient


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
        
        # Initialize BiblioExtractor with provided configuration - Claude Generated
        self.extractor = BiblioClient(
            token=token or "",  # Ensure string, not None
            debug=debug,
            enable_web_fallback=True  # Claude Generated - Enable web fallback
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
    
    def search(self, searches: List[str], search_type: str = "kw") -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Search for subjects related to the given search terms.

        Args:
            searches: List of search terms
            search_type: "kw" (default, Libero 'ku' anyword), "title" (Libero 'k'),
                "freetext" (Libero 'ku')

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
        # WP2 raw-first: verbatim parsed records per term for the raw cache. - Claude Generated
        self.last_raw = {}
        self.last_http_status = {}
        try:
            # Use BiblioExtractor's search_subjects method
            results = self.extractor.search_subjects(searches, search_type=search_type)
            client_raw = getattr(self.extractor, "last_raw", None) or {}
            for _term, _records in client_raw.items():
                try:
                    self.last_raw[_term] = json.dumps(
                        {"records": _records, "totalItems": len(_records)},
                        ensure_ascii=False, default=str,
                    )
                except (TypeError, ValueError):
                    continue
            #self.logger.info(f"Search completed for terms: {searches}")
            #self.logger.info(f"Search results: {results}")
                    # Log keys of all entries
            for key, entry in results.items():
                if isinstance(entry, dict):
                    self.logger.debug(f"Entry '{key}' keys: {entry.keys()}")
                else:
                    self.logger.debug(f"Entry '{key}' type: {type(entry)}")
            # Emit signals for progress tracking
            for search_term in searches:
                self.currentTerm.emit(search_term)
            
            return results
            
        except Exception as e:
            error_msg = f"BiblioSuggester search failed: {str(e)}"
            self.logger.error(error_msg)
            raise BiblioSuggesterError(error_msg) from e
    
    def search_titles(
        self,
        search_terms: List[str],
        search_type: str = "title",
        max_results: int = 25,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return bibliographic records per query via catalog title search.

        Thin passthrough to ``BiblioClient.search_titles``. Unlike
        :meth:`search`, results are book records (not aggregated subjects).
        """
        try:
            return self.extractor.search_titles(
                search_terms,
                max_results=max_results,
                search_type=search_type,
            )
        except Exception as e:
            error_msg = f"BiblioSuggester search_titles failed: {str(e)}"
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