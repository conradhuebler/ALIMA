import logging
from typing import List, Dict, Any

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

    def _catalog_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Overlay the explicit catalog token/URLs (owned by the pipeline) onto the
        resolved catalog instance settings; empty values are ignored by the resolver
        so config-provided values win when the pipeline passes nothing. - Claude Generated"""
        return {
            "catalog": {
                "token": self.catalog_token,
                "catalog_search_url": self.catalog_search_url,
                "catalog_details": self.catalog_details_url,
            }
        }

    def search(
        self, search_terms: List[str], suggester_types: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Mapping-first GND-keyword search via the unified provider service.

        Thin adapter: resolves the requested provider ids to instances (overlaying
        the pipeline's explicit catalog config) and delegates the build/search/merge
        to ``src.core.search.service``. - Claude Generated
        """
        from .search.service import resolve_gnd_instances, search_gnd_keywords

        instances = resolve_gnd_instances(
            suggester_types, overrides=self._catalog_overrides()
        )
        results, self.last_errors = search_gnd_keywords(
            search_terms, instances, cache=True, aggregate_from_raw=False,
            ukm=self.cache_manager,
        )
        return results

    def search_from_raw(
        self, search_terms: List[str], suggester_types: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Fetch (populating the raw cache) then derive the nested result from raw.

        WP2 P4.4b convergence: same signature/return shape as :meth:`search`, but the
        reduced ``{term:{title:{...}}}`` view is derived from the raw response cache
        (single source of truth) via the shared aggregate engine (raw-first, mapping
        fallback). The per-source search still runs — it performs the live fetch that
        writes raw through the provider seam and surfaces source failures. - Claude Generated
        """
        from .search.service import resolve_gnd_instances, search_gnd_keywords

        instances = resolve_gnd_instances(
            suggester_types, overrides=self._catalog_overrides()
        )
        results, self.last_errors = search_gnd_keywords(
            search_terms, instances, cache=True, aggregate_from_raw=True,
            ukm=self.cache_manager,
        )
        return results

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
