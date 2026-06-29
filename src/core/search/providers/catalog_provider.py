"""Local catalog (Libero/Biblio) search provider - Claude Generated.

Wraps ``BiblioSuggester`` and exposes its three faces as explicit capabilities:
GND-keyword subject search, bibliographic title records, and DK classification
extraction.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..provider import ProviderResult, ResultItem, SearchCapability
from ..registry import register_provider
from ._base import SuggesterBackedProvider


@register_provider
class CatalogProvider(SuggesterBackedProvider):
    """Local library catalog via Libero/SOAP (``BiblioSuggester``)."""

    id = "catalog"
    label = "Katalog (Libero)"
    capabilities = {
        SearchCapability.GND_KEYWORDS,
        SearchCapability.TITLE_RECORDS,
        SearchCapability.CLASSIFICATION,
    }

    def _build_suggester(self):
        from src.utils.suggesters.biblio_suggester import BiblioSuggester

        return BiblioSuggester(
            token=self._config.get("token", "") or "",
            catalog_search_url=self._config.get("catalog_search_url", "") or "",
            catalog_details=self._config.get("catalog_details", "") or "",
            debug=self._config.get("debug", False),
        )

    def is_available(self, cfg: Any = None) -> bool:
        # A catalog token is the practical gate for the SOAP backend (matches the
        # GUI which only shows the catalog option when a token is configured).
        return bool(self._config.get("token"))

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        search_type: str = "kw",
        max_results: int = 25,
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        if capability is SearchCapability.GND_KEYWORDS:
            return self._gnd_search(query, progress, search_type=search_type)
        if capability is SearchCapability.TITLE_RECORDS:
            self._wire_progress(progress)
            raw = self.suggester.search_titles(
                list(query), search_type=search_type, max_results=max_results
            )
            return _title_records_from_term_lists(raw)
        # CLASSIFICATION
        results = self.suggester.extract_dk_classifications(list(query))
        return _classification_from_dk_list(results)


def _title_records_from_term_lists(raw: dict) -> ProviderResult:
    """Convert ``{term: [record, ...]}`` (catalog title search) to a result."""
    per_term = {}
    per_term_meta = {}
    for term, recs in (raw or {}).items():
        recs = recs or []
        per_term[term] = [
            ResultItem(label=str((r or {}).get("title", "")), record=r) for r in recs
        ]
        per_term_meta[term] = {"result_count": len(recs)}
    return ProviderResult(
        SearchCapability.TITLE_RECORDS, per_term=per_term, per_term_meta=per_term_meta
    )


def _classification_from_dk_list(results: list) -> ProviderResult:
    """Convert ``extract_dk_classifications`` output (flat DK dicts) to a result.

    DK results are keyed under each matched keyword; entries with no matched
    keyword fall into the ``"_all"`` bucket. The full source dict is preserved in
    ``extra`` so no information is lost.
    """
    per_term: dict = {}
    for d in results or []:
        d = d or {}
        item = ResultItem(
            code=str(d.get("dk", "")),
            label=str(d.get("dk", "")),
            count=int(d.get("count", 0) or 0),
            extra=dict(d),
        )
        keys = d.get("matched_keywords") or ["_all"]
        for k in keys:
            per_term.setdefault(str(k), []).append(item)
    return ProviderResult(SearchCapability.CLASSIFICATION, per_term=per_term)
