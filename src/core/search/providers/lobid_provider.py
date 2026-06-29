"""Lobid (GND/DNB) search provider - Claude Generated."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..provider import ProviderResult, ProviderToolSpec, SearchCapability
from ..registry import register_provider
from ._base import SuggesterBackedProvider


@register_provider
class LobidProvider(SuggesterBackedProvider):
    """GND-keyword candidates from the lobid.org GND aggregation."""

    id = "lobid"
    label = "Lobid (GND/DNB)"
    capabilities = {SearchCapability.GND_KEYWORDS}

    @classmethod
    def mcp_tool_specs(cls):
        return [ProviderToolSpec(
            name="search_lobid",
            capability=SearchCapability.GND_KEYWORDS,
            description="Search Lobid.org GND API for subject headings. Returns keywords with GND IDs and DDC codes.",
            parameters={
                "type": "object",
                "properties": {
                    "terms": {"type": "array", "items": {"type": "string"}, "description": "List of search terms"},
                    "search_type": {
                        "type": "string",
                        "enum": ["kw", "title", "freetext"],
                        "default": "kw",
                        "description": "Query mode: kw=subject/keyword, title=title-only, freetext=any field",
                    },
                },
                "required": ["terms"],
            },
            result_shape="gnd_keywords",
            source_label="lobid",
            include_errors=True,
            cached=True,
            add_gnd_urls=True,
            unavailable_message="LobidSuggester not available",
            default_opts={"search_type": "kw"},
        )]

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
