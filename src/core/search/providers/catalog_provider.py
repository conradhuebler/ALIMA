"""Local catalog (Libero/Biblio) search provider - Claude Generated.

Wraps ``BiblioSuggester`` and exposes its three faces as explicit capabilities:
GND-keyword subject search, bibliographic title records, and DK classification
extraction.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from src.core.plugins.schema import BOOL, CHOICE, URL, SECRET, ConfigField

from ..provider import ProviderResult, ProviderToolSpec, ResultItem, SearchCapability
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

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(
                key="token", label="SOAP-Token", kind=SECRET, gates_availability=True,
                help="Libero SOAP-Authentifizierungstoken; ohne Token ist die Quelle inaktiv.",
            ),
            ConfigField(
                key="catalog_search_url", label="Such-URL (SOAP)", kind=URL,
                help="Libero CatalogueSearcher-Endpoint.",
            ),
            ConfigField(
                key="catalog_details", label="Detail-URL (SOAP)", kind=URL,
                help="Libero LibraryAPI-Endpoint für Titel-Details.",
            ),
            ConfigField(
                key="catalog_web_search_url", label="Web-Such-URL (OPAC)", kind=URL,
                help="Libero Web-OPAC-Suche (BiblioClient-Web-Fallback).",
            ),
            ConfigField(
                key="catalog_web_record_url", label="Web-Record-URL (OPAC)", kind=URL,
                help="Basis für OPAC-Titel-Weblinks (…/Record/<rsn>).",
            ),
            ConfigField(
                key="catalog_type", label="DK-Backend", kind=CHOICE,
                choices=["libero_soap", "marcxml_sru", "auto"], default="libero_soap",
                help="Welches Backend die DK-Suche nutzt: Libero-SOAP, MARC-XML/SRU oder auto.",
            ),
            ConfigField(
                key="strict_gnd_validation_for_dk_search", label="Strikte GND-Validierung (DK)",
                kind=BOOL, default=True,
                help="Nur GND-validierte Keywords in die DK-Suche geben (Qualitätskontrolle).",
            ),
        ]

    @classmethod
    def doc(cls):
        from src.core.plugins.schema import PluginDoc

        return PluginDoc(
            description="Lokaler Bibliothekskatalog (Libero/SOAP): GND-Schlagworte, "
            "bibliografische Titel-Datensätze und DK-Klassifikationen.",
            input="Suchbegriffe bzw. Titel-/Schlagwortanfragen (je nach Capability).",
            output="GND-Schlagworte, Titel-Datensätze (rsn, Titel, Autoren, web_url …) "
            "oder DK-Codes.",
        )

    @classmethod
    def mcp_tool_specs(cls):
        return [
            ProviderToolSpec(
                name="search_catalog",
                capability=SearchCapability.GND_KEYWORDS,
                description="Search bibliographic catalog via SOAP/SRU for titles and DK classifications.",
                parameters={
                    "type": "object",
                    "properties": {
                        "terms": {"type": "array", "items": {"type": "string"}, "description": "List of search terms"},
                        "search_type": {
                            "type": "string",
                            "enum": ["kw", "title", "freetext"],
                            "default": "kw",
                            "description": "Query mode: kw=anyword (Libero 'ku'), title=title (Libero 'k'), freetext=anyword",
                        },
                    },
                    "required": ["terms"],
                },
                result_shape="gnd_keywords",
                source_label="catalog",
                include_errors=False,
                cached=False,
                unavailable_message="BiblioSuggester not available",
            ),
            ProviderToolSpec(
                name="search_catalog_titles",
                capability=SearchCapability.TITLE_RECORDS,
                description=(
                    "Search bibliographic catalog for book records by title or keyword. "
                    "Returns per-query lists of records (rsn, title, authors, year, "
                    "dk_codes, rvk_codes, subjects). No GND/SWB/Lobid enrichment — "
                    "pure catalog hits intended for title-list workflows. Each record "
                    "includes `web_url` (catalog web link for that RSN) when a web "
                    "record URL is configured. Always cite records as Markdown links: "
                    "[title](web_url). Omit the link only when web_url is absent."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "terms": {"type": "array", "items": {"type": "string"}, "description": "List of title queries"},
                        "search_type": {
                            "type": "string",
                            "default": "title",
                            "description": (
                                "Libero use-code or alias: 'title' (ti, default), "
                                "'kw'/'freetext' (ku), or raw codes like 'kb' (author), "
                                "'ke' (combined author), 'sk' (subjects), 'i' (ISBN)."
                            ),
                        },
                        "max_results": {"type": "integer", "default": 25, "description": "Maximum records per query"},
                    },
                    "required": ["terms"],
                },
                result_shape="title_records",
                source_label="catalog_titles",
                include_errors=False,
                unavailable_message="BiblioSuggester not available",
            ),
        ]

    def _build_suggester(self):
        from src.utils.suggesters.biblio_suggester import BiblioSuggester

        return BiblioSuggester(
            token=self._config.get("token", "") or "",
            catalog_search_url=self._config.get("catalog_search_url", "") or "",
            catalog_details=self._config.get("catalog_details", "") or "",
            debug=self._config.get("debug", False),
        )

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
            self._store_records_raw(
                "catalog_titles", list(query), {"search_type": search_type}, raw
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
