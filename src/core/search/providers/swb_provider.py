"""SWB (Südwestdeutscher Bibliotheksverbund) search provider - Claude Generated."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..provider import ProviderResult, SearchCapability
from ..registry import register_provider
from ._base import SuggesterBackedProvider


@register_provider
class SwbProvider(SuggesterBackedProvider):
    """GND-keyword candidates scraped from the SWB union catalog."""

    id = "swb"
    label = "SWB (BSZ)"
    capabilities = {SearchCapability.GND_KEYWORDS}

    def _build_suggester(self):
        from src.utils.suggesters.swb_suggester import SWBSuggester

        return SWBSuggester(debug=self._config.get("debug", False))

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        search_type: str = "kw",
        max_pages: int = 5,
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        return self._gnd_search(
            query, progress, search_type=search_type, max_pages=max_pages
        )
