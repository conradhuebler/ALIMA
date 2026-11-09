"""SWB (Südwestdeutscher Bibliotheksverbund) search provider - Claude Generated."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from src.core.search.provider import ProviderResult, ProviderToolSpec, SearchCapability
from src.core.search.registry import register_provider
from src.core.search.provider_base import SuggesterBackedProvider


@register_provider
class SwbProvider(SuggesterBackedProvider):
    """GND-keyword candidates scraped from the SWB union catalog."""

    id = "swb"
    label = "SWB (BSZ)"
    capabilities = {SearchCapability.GND_KEYWORDS}
    raw_cache_param_keys = ("search_type", "max_pages")  # pages on max_pages

    @classmethod
    def config_fields(cls):
        """Request timeout and cache-only switch - Claude Generated"""
        from src.core.search.cache_only import cache_only_field
        from src.core.search.timeouts import timeout_field

        from .suggester import REQUEST_TIMEOUT_S

        return [
            timeout_field(REQUEST_TIMEOUT_S, help="Wartezeit je Seitenabruf beim SWB."),
            cache_only_field(),
        ]

    @classmethod
    def doc(cls):
        from src.core.plugins.schema import PluginDoc

        return PluginDoc(
            description="GND-Schlagwortsuche im SWB-Verbundkatalog (BSZ).",
            input="Ein oder mehrere Suchbegriffe (Schlagwort/Titel/Freitext).",
            output="GND-Schlagwort-Kandidaten mit GND-IDs, Häufigkeit und DDC/DK-Notationen.",
        )

    @classmethod
    def mcp_tool_specs(cls):
        return [ProviderToolSpec(
            name="search_swb",
            capability=SearchCapability.GND_KEYWORDS,
            description="Search SWB (Südwestdeutscher Bibliotheksverbund) catalog for subject headings and classifications.",
            parameters={
                "type": "object",
                "properties": {
                    "terms": {"type": "array", "items": {"type": "string"}, "description": "List of search terms"},
                    "max_pages": {"type": "integer", "description": "Max result pages per term", "default": 5},
                    "search_type": {
                        "type": "string",
                        "enum": ["kw", "title", "freetext"],
                        "default": "kw",
                        "description": "Query mode: kw=subject (IKT 2074), title=title (IKT 2058), freetext=anyword",
                    },
                },
                "required": ["terms"],
            },
            result_shape="gnd_keywords",
            source_label="swb",
            include_errors=True,
            cached=True,
            add_gnd_urls=True,
            unavailable_message="SWBSuggester not available",
            default_opts={"search_type": "kw", "max_pages": 5},
        )]

    def _build_suggester(self):
        from src.core.search.timeouts import timeout_from

        from .suggester import REQUEST_TIMEOUT_S, SWBSuggester

        return SWBSuggester(
            debug=self._config.get("debug", False),
            timeout=timeout_from(self._config, REQUEST_TIMEOUT_S),
            source_id=self.source_id,
        )

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
