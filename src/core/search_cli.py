import logging
from typing import List, Dict, Any

from ..utils.suggesters.meta_suggester import MetaSuggester, SuggesterType
from .unified_knowledge_manager import UnifiedKnowledgeManager


class SearchCLI:
    def __init__(self, cache_manager: UnifiedKnowledgeManager, catalog_token: str = "", catalog_search_url: str = "", catalog_details_url: str = ""):
        self.logger = logging.getLogger(__name__)
        self.cache_manager = cache_manager
        self.catalog_token = catalog_token
        self.catalog_search_url = catalog_search_url  
        self.catalog_details_url = catalog_details_url

    def search(
        self, search_terms: List[str], suggester_types: List[SuggesterType]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        combined_results = {}

        for suggester_type in suggester_types:
            self.logger.info(f"Searching with {suggester_type.value} suggester")

            try:
                suggester = MetaSuggester(
                    suggester_type=suggester_type,
                    debug=False,
                    catalog_token=self.catalog_token,
                    catalog_search_url=self.catalog_search_url,
                    catalog_details=self.catalog_details_url,
                )

                results = suggester.search(search_terms)
                self.merge_results(combined_results, results)

            except Exception as e:
                self.logger.error(
                    f"Error searching with {suggester_type.value} suggester: {e}"
                )

        return combined_results

    def merge_results(self, combined_results, new_results):
        for search_term, term_results in new_results.items():
            if search_term not in combined_results:
                combined_results[search_term] = {}

            for keyword, data in term_results.items():
                if keyword not in combined_results[search_term]:
                    combined_results[search_term][keyword] = data.copy()
                else:
                    existing_data = combined_results[search_term][keyword]
                    existing_data["count"] = max(
                        existing_data["count"], data.get("count", 0)
                    )
                    existing_data["gndid"].update(data.get("gndid", set()))
                    existing_data["ddc"].update(data.get("ddc", set()))
                    existing_data["dk"].update(data.get("dk", set()))
