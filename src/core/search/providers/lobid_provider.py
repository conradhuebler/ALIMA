"""Lobid (GND/DNB) search provider - Claude Generated."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..provider import ProviderResult, SearchCapability
from ..registry import register_provider
from ._base import SuggesterBackedProvider


@register_provider
class LobidProvider(SuggesterBackedProvider):
    """GND-keyword candidates from the lobid.org GND aggregation."""

    id = "lobid"
    label = "Lobid (GND/DNB)"
    capabilities = {SearchCapability.GND_KEYWORDS}

    def _build_suggester(self):
        from src.utils.suggesters.lobid_suggester import LobidSuggester

        return LobidSuggester(debug=self._config.get("debug", False))

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        search_type: str = "kw",
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        return self._gnd_search(query, progress, search_type=search_type)
