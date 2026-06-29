import logging
from typing import List, Dict, Any

from ..utils.suggesters.meta_suggester import MetaSuggester
from .gnd_search_core import merge_code_entry
from .unified_knowledge_manager import UnifiedKnowledgeManager


class SearchCLI:
    """
    GND Search CLI - supports context manager pattern for automatic resource cleanup.
    Claude Generated

    Usage:
        # Recommended: context manager pattern
        with SearchCLI(cache_manager) as search_cli:
            results = search_cli.search([...], [...])

        # Legacy: direct instantiation (still works but no automatic cleanup)
        search_cli = SearchCLI(cache_manager)
        results = search_cli.search([...], [...])
        search_cli.close()  # Manual cleanup
    """

    def __init__(self, cache_manager: UnifiedKnowledgeManager, catalog_token: str = "", catalog_search_url: str = "", catalog_details_url: str = ""):
        self.logger = logging.getLogger(__name__)
        self.cache_manager = cache_manager
        self.catalog_token = catalog_token
        self.catalog_search_url = catalog_search_url
        self.catalog_details_url = catalog_details_url
        self._active_suggesters = []  # Track active suggesters for cleanup
        # Source failures of the last search() call ("<suggester>:<term>" → message).
        # Lets callers distinguish "source down" from "term not found" - Claude Generated
        self.last_errors: Dict[str, str] = {}

    def __enter__(self):
        """Enter context manager - Claude Generated"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with cleanup - Claude Generated"""
        self.close()
        return False  # Don't suppress exceptions

    def close(self):
        """Clean up resources (connections, suggesters) - Claude Generated"""
        # Clear any cached suggester references
        self._active_suggesters.clear()
        self.logger.debug("SearchCLI closed and resources released")

    def search(
        self, search_terms: List[str], suggester_types: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        combined_results = {}
        self.last_errors = {}

        for suggester_type in suggester_types:
            self.logger.debug(f"Searching with {suggester_type} suggester")

            try:
                suggester = MetaSuggester(
                    providers=suggester_type,
                    debug=False,
                    catalog_token=self.catalog_token,
                    catalog_search_url=self.catalog_search_url,
                    catalog_details=self.catalog_details_url,
                )

                results = suggester.search(search_terms)
                self.merge_results(combined_results, results)
                # Propagate per-term source failures to callers - Claude Generated
                self.last_errors.update(getattr(suggester, "last_errors", {}))

            except Exception as e:
                self.logger.error(
                    f"Error searching with {suggester_type} suggester: {e}"
                )
                for term in search_terms:
                    self.last_errors[f"{suggester_type}:{term}"] = str(e)

        return combined_results

    def merge_results(self, combined_results, new_results):
        for search_term, term_results in new_results.items():
            if search_term not in combined_results:
                combined_results[search_term] = {}

            for keyword, data in term_results.items():
                if keyword not in combined_results[search_term]:
                    combined_results[search_term][keyword] = data.copy()
                else:
                    # Shared merge-atom (max count + union of code sets) —
                    # see src/core/gnd_search_core.py - Claude Generated
                    merge_code_entry(
                        combined_results[search_term][keyword],
                        data,
                        code_fields=("gndid", "ddc", "dk"),
                    )
