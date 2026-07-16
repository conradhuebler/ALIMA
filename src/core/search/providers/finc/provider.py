"""finc / VuFind-JSON search provider - Claude Generated.

Brings finc into the standard: instead of being special-cased in the MCP tool
layer, finc is a regular provider declaring ``TITLE_RECORDS`` (bibliographic
records) and ``SUBJECT_FACETS`` (DK/RVK distribution). One underlying finc call
yields both records and facets, so :meth:`search` runs the client once and shapes
the requested capability from the same payload.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

from src.core.plugins.schema import BOOL, INT, TEXT, URL, ConfigField

from src.core.search.provider import ProviderResult, ProviderToolSpec, ResultItem, SearchCapability
from src.core.search.provider_base import SuggesterBackedProvider
from src.core.search.registry import register_provider


@register_provider
class FincProvider(SuggesterBackedProvider):
    """finc / VuFind-JSON catalog (records + classification facets).

    Inherits the shared __init__/_ukm/suggester/is_available/_require plumbing;
    overrides only _build_suggester (finc-specific client) and _store_finc_raw
    (FincSuggester exposes no ``last_raw`` for the base dual-write). - Claude Generated
    """

    id = "finc"
    label = "finc (VuFind)"
    capabilities = {
        SearchCapability.TITLE_RECORDS,
        SearchCapability.SUBJECT_FACETS,
        SearchCapability.CLASSIFICATION,
    }
    raw_cache_param_keys = ("search_type", "facets")  # results depend on the facet set
    # availability enum (lowercase, LLM-facing) → VuFind facet_avail value.
    _AVAIL_TO_FACET = {"local": "Local", "online": "Online", "free": "Free"}

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(
                key="base_url", label="finc Basis-URL", kind=URL, gates_availability=True,
                help="VuFind solrproxy-Endpoint; ohne URL ist finc inaktiv.",
            ),
            ConfigField(
                key="web_record_url", label="Web-Record-URL", kind=URL,
                help="Basis für Katalog-Weblinks (…/Record/<id>).",
            ),
            ConfigField(
                key="catalog_web_record_url", label="Katalog-Record-URL (Fallback)", kind=URL,
                help="Fallback-Basis für Katalog-Weblinks, wenn Web-Record-URL leer ist "
                "(gleicher OPAC-Host wie das Libero-Backend).",
            ),
            ConfigField(
                key="institution_filter", label="Institutions-Filter", kind=TEXT,
                help="Optionaler VuFind-Facet-Filter, z.B. institution:DE-105.",
            ),
            ConfigField(key="default_limit", label="Max. Datensätze", kind=INT, default=20),
            ConfigField(key="timeout", label="Timeout (s)", kind=INT, default=30),
            ConfigField(
                key="dk_enabled", label="finc für DK-Suche nutzen", kind=BOOL, default=False,
                help="Per-Titel udk_raw für die DK-Klassifikationssuche statt Libero/SRU.",
            ),
            ConfigField(
                key="harvest_enabled", label="finc-Subject-Harvest", kind=BOOL, default=False,
                help="Im Keyword-Schritt finc-Titel ernten und gegen den GND-Cache abgleichen.",
            ),
        ]

    @classmethod
    def doc(cls):
        from src.core.plugins.schema import PluginDoc

        return PluginDoc(
            description="finc/VuFind-JSON-Katalog (z.B. TU Freiberg): bibliografische "
            "Datensätze (inkl. Verlag/Auflage/Jahr/ISBN) plus DK/RVK-Klassifikationsverteilung. "
            "Schnell und meist ausreichend; deckt aber ggf. nicht jeden älteren/gedruckten "
            "Bestand ab (Fallback: catalog/search_catalog_titles).",
            input="Suchbegriffe (Titel/Schlagwort/Autor) + optionale Facetten/Filter/Verfügbarkeit.",
            output="Titel-Datensätze (id, Titel, Autoren, Verlag, Auflage, Jahr, ISBN, web_url, "
            "resource_url, urls[] …) und/oder Facetten-Verteilungen (udk_raw, rvk_facet).",
        )

    @classmethod
    def mcp_tool_specs(cls):
        return [ProviderToolSpec(
            name="search_finc",
            capability=SearchCapability.TITLE_RECORDS,
            description=(
                "Search a finc / VuFind-JSON library catalog (e.g. TU Freiberg finc "
                "solrproxy) for full bibliographic records — fast (one HTTP call covers "
                "all `terms`) and usually sufficient on its own; try this before "
                "search_catalog_titles, not after. Choose the search axis via "
                "`search_type`: by subject/keyword, by title (one OR many — pass several "
                "titles in `terms` to look them all up in one call), or by author. "
                "`terms` is searched independently and the results are keyed per term. "
                "Each record has id, title, authors, subjects, formats, languages, series, "
                "publisher, edition, year (structured publication year — use this for "
                "duplicate/edition comparisons), isbn, web_url, resource_url, and urls[]. "
                "The web_url is always the direct catalog record link (e.g. "
                "https://katalog.ub.tu-freiberg.de/Record/0-1025700295). resource_url is "
                "the full-text/DOI link when one exists (empty otherwise — never "
                "fabricated); urls[] holds the underlying candidates (DOI links, "
                "publisher pages, open-access copies, cover images). Cite resource_url "
                "alongside the catalog link when present. "
                "Caveat: finc's index may not include every older or print-only holding — "
                "if a title genuinely isn't found here, try search_catalog_titles (direct "
                "Libero/SOAP catalog search) as a fallback before concluding it's absent "
                "from the collection. "
                "Use `facets` (e.g. [\"udk_raw_de105\",\"rvk_facet\"]) to also "
                "get the DK/RVK classification distribution, and `filters` to scope by "
                "facet (VuFind syntax, e.g. {\"institution\": \"DE-105\"} or "
                "{\"id\": \"<record-id>\"} for one record). Use `availability` to "
                "restrict to physical holdings ('local'), licensed e-resources "
                "('online'), or open access ('free'). "
                "Always cite records as Markdown links: [title](web_url). "
                "Every listed record must include its link when web_url is present."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more search terms, each searched independently (e.g. several book titles or author names in one call). Wrap a phrase in literal double quotes for an exact match (e.g. \"conrad hübler\"); URL-encoding is handled automatically.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["kw", "title", "subject", "author", "freetext", "dk", "rvk"],
                        "default": "kw",
                        "description": (
                            "Which field to search: subject = controlled subject/keyword "
                            "headings; title = words in the title (use for one or many "
                            "titles); author = author/contributor names; kw/freetext = "
                            "all fields; dk = search directly in the DK/UDK notation field "
                            "(udk_raw_de105, e.g. lookfor='DK 57' or 'qt 000'); rvk = search "
                            "directly in the RVK notation field (rvk_facet). For dk and rvk "
                            "types, udk_raw_de105 and rvk_facet facets are added automatically "
                            "so the classification distribution is always returned."
                        ),
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional facet filters as key→value. Each entry is sent "
                            "as one filter[]=key:\"value\" parameter. Common keys: "
                            "institution (holding library, e.g. DE-105), udk_facet_de105 "
                            "(coarse DK group), rvk_facet, id (single record), format, "
                            "language."
                        ),
                    },
                    "facets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional facet fields to compute per term; buckets are "
                            "returned under each term's 'facets'. Use udk_raw_de105 for "
                            "numeric DK notations (e.g. 'dk 530.145'), rvk_facet for RVK, "
                            "dewey-hundreds for DDC."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Maximum records per term (0..100; 0 = facets only).",
                    },
                    "availability": {
                        "type": "string",
                        "enum": ["local", "online", "free"],
                        "description": (
                            "Filter by holding type: 'local' = physical copy in the library "
                            "(Präsenzbestand/Ausleihbestand); 'online' = licensed electronic "
                            "resource; 'free' = open access / freely available online. "
                            "Omit to return all holdings."
                        ),
                    },
                },
                "required": ["terms"],
            },
            result_shape="finc",
            source_label="finc",
            include_errors=True,
        )]

    def _store_finc_raw(self, query, params, raw):
        """Dual-write the per-term finc payload to the raw cache. Best-effort.

        FincSuggester exposes no ``last_raw``/``last_http_status``, so the base
        _store_raw_responses would be a silent no-op; this shapes the per-term
        return value instead. The enable policy is the base tri-state gate
        (``cache_responses`` auto/on/off × the global) — the old ``bool(override)``
        treated "off"/"auto" as truthy and ignored the operator. - Claude Generated
        """
        if not self._cache_raw_enabled():
            return
        try:
            ukm = self._ukm()
        except Exception:
            return
        for term in query:
            entry = (raw or {}).get(term)
            if entry is None:
                continue
            try:
                blob = json.dumps(entry, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                continue
            ukm.store_raw_response("finc", term, params, blob)

    def _build_suggester(self):
        from .suggester import FincSuggester

        # default_limit/timeout were declared + mirrored but never passed here
        # (latent bug: non-primary finc silently fell back to the client defaults
        # 20/30 and the primary path's config was ignored on the factory route).
        # - Claude Generated
        return FincSuggester(
            base_url=self._config.get("base_url", "") or "",
            web_record_url=self._config.get("web_record_url", "") or "",
            default_limit=int(self._config.get("default_limit", 20) or 20),
            timeout=int(self._config.get("timeout", 30) or 30),
            institution_filter=self._config.get("institution_filter", "") or "",
            debug=self._config.get("debug", False),
        )

    def catalog_web_record_base(self) -> str:
        """The catalog-web-link fallback base for records whose web_url is empty.

        Instance-sourced (self-contained: a copied finc plugin carries its own),
        used by the finc tool handler's web_url reconstruction. - Claude Generated
        """
        return (self._config.get("catalog_web_record_url", "") or "").rstrip("/")

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        search_type: str = "kw",
        filters: Optional[dict] = None,
        limit: Optional[int] = None,
        facets: Optional[List[str]] = None,
        availability: Optional[str] = None,
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        if capability is SearchCapability.CLASSIFICATION:
            # DK/RVK extraction is a keyword→classifications operation that does
            # not run the title/facet suggester; delegate to the extractor. - Claude Generated
            return self._classification_search(list(query), progress=progress, **opts)
        # DK/RVK field searches auto-include the classification facets so callers
        # always get the notation distribution; explicit facets win. - Claude Generated
        effective_facets = facets
        if (search_type or "kw") in ("dk", "rvk") and not facets:
            effective_facets = ["udk_raw_de105", "rvk_facet"]
        # Availability enum → facet_avail filter; caller-supplied filters win.
        effective_filters = dict(filters or {})
        if availability:
            facet_val = self._AVAIL_TO_FACET.get((availability or "").lower())
            if facet_val:
                effective_filters.setdefault("facet_avail", facet_val)
        if progress is not None:
            try:
                self.suggester.currentTerm.connect(progress)
            except Exception:
                pass
        raw = self.suggester.search(
            list(query),
            search_type=search_type,
            filters=effective_filters or None,
            limit=limit,
            facets=effective_facets,
        )
        errors = dict(getattr(self.suggester, "last_errors", {}) or {})
        from src.core.search.provider import raw_cache_params_for
        self._store_finc_raw(
            list(query),
            raw_cache_params_for("finc", search_type=search_type, facets=effective_facets),
            raw,
        )
        if capability is SearchCapability.TITLE_RECORDS:
            return ProviderResult.from_finc_records(raw, errors=errors)
        return _facets_from_finc_dict(raw, errors=errors)

    def dk_extractor(
        self,
        *,
        logger_: Any = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        knowledge_manager: Any = None,
    ):
        """Return the finc DK/RVK extractor backing the ``CLASSIFICATION`` capability.

        Implements the shared ``extract_dk_classifications_for_keywords`` contract
        that the classic DK step (``execute_dk_search``) consumes, built from this
        provider's own config — so a copied-out finc plugin supplies DK/RVK with
        no core wiring. The ``FincCatalogClient`` lives inside this plugin dir
        (self-contained). - Claude Generated
        """
        from .finc_catalog_client import FincCatalogClient

        return FincCatalogClient(
            base_url=self._config.get("base_url", "") or "",
            web_record_url=self._config.get("web_record_url", "") or "",
            institution_filter=self._config.get("institution_filter", "") or "",
            timeout=int(self._config.get("timeout", 30) or 30),
            max_titles_per_keyword=int(self._config.get("default_limit", 50) or 50),
            logger_=logger_,
            stream_callback=stream_callback,
            knowledge_manager=knowledge_manager,
        )

    def _classification_search(
        self, query: List[str], *, progress: Optional[Callable[[str], None]] = None,
        max_results: int = 50, force_update: bool = False, **_opts: Any,
    ) -> ProviderResult:
        """``search(CLASSIFICATION)`` path: delegate to the DK extractor and shape
        the keyword-centric output into a ``CLASSIFICATION`` result. - Claude Generated"""
        cb = progress if callable(progress) else None
        results = self.dk_extractor(stream_callback=cb).extract_dk_classifications_for_keywords(
            list(query), max_results=int(max_results or 50), force_update=bool(force_update),
        )
        return _classification_from_kw_results(results)


def _classification_from_kw_results(kw_results: list) -> ProviderResult:
    """Convert the shared ``extract_dk_classifications_for_keywords`` output
    (``[{keyword, source, classifications: [{dk, type, count, …}]}]``) into a
    ``CLASSIFICATION`` result, keyed per keyword. The full per-code source dict
    is preserved in ``extra`` so nothing is lost. - Claude Generated
    """
    per_term: dict = {}
    for kw in kw_results or []:
        kw = kw or {}
        term = str(kw.get("keyword", "") or "_all")
        for c in kw.get("classifications", []) or []:
            c = c or {}
            code = str(c.get("dk", "") or c.get("code", ""))
            per_term.setdefault(term, []).append(
                ResultItem(
                    code=code,
                    label=str(c.get("type", "") or code),
                    count=int(c.get("count", 0) or 0),
                    extra=dict(c),
                )
            )
    return ProviderResult(SearchCapability.CLASSIFICATION, per_term=per_term)


def _facets_from_finc_dict(raw: dict, errors: Optional[dict] = None) -> ProviderResult:
    """Convert the finc ``facets`` block into a ``SUBJECT_FACETS`` result."""
    per_term = {}
    for term, entry in (raw or {}).items():
        entry = entry or {}
        items = []
        for facet_name, buckets in (entry.get("facets", {}) or {}).items():
            for b in buckets or []:
                b = b or {}
                items.append(
                    ResultItem(
                        code=str(b.get("value", "")),
                        label=str(b.get("translated", "") or b.get("value", "")),
                        count=int(b.get("count", 0) or 0),
                        extra={"facet": facet_name},
                    )
                )
        per_term[term] = items
    return ProviderResult(
        SearchCapability.SUBJECT_FACETS, per_term=per_term, errors=dict(errors or {})
    )
