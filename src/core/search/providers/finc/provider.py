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

from src.core.plugins.schema import BOOL, INT, TEXT, URL, ConfigField, availability_ok

from src.core.search.provider import ProviderResult, ProviderToolSpec, ResultItem, SearchCapability
from src.core.search.registry import register_provider


@register_provider
class FincProvider:
    """finc / VuFind-JSON catalog (records + classification facets)."""

    id = "finc"
    label = "finc (VuFind)"
    capabilities = {SearchCapability.TITLE_RECORDS, SearchCapability.SUBJECT_FACETS}

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
            "Datensätze plus DK/RVK-Klassifikationsverteilung.",
            input="Suchbegriffe (Titel/Schlagwort/Autor) + optionale Facetten/Filter/Verfügbarkeit.",
            output="Titel-Datensätze (id, Titel, Autoren, web_url, urls[] …) und/oder "
            "Facetten-Verteilungen (udk_raw, rvk_facet).",
        )

    @classmethod
    def mcp_tool_specs(cls):
        return [ProviderToolSpec(
            name="search_finc",
            capability=SearchCapability.TITLE_RECORDS,
            description=(
                "Search a finc / VuFind-JSON library catalog (e.g. TU Freiberg finc "
                "solrproxy) for full bibliographic records. Preferred over search_catalog "
                "when the institution runs a finc instance. Choose the search axis via "
                "`search_type`: by subject/keyword, by title (one OR many — pass several "
                "titles in `terms` to look them all up in one call), or by author. "
                "`terms` is searched independently and the results are keyed per term. "
                "Each record has id, title, authors, subjects, formats, languages, series, "
                "web_url, and urls[]. The web_url is always the direct catalog record link "
                "(e.g. https://katalog.ub.tu-freiberg.de/Record/0-1025700295). "
                "urls[] may additionally contain DOI links, publisher pages, or open-access "
                "copies — cite them when relevant (e.g. full-text link alongside catalog link). "
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

    def __init__(self, **config: Any):
        self._config = config or {}
        self._suggester = None
        # WP2 raw-first cache injection points (set by search.factory). - Claude Generated
        self._cache_raw = None
        self._ukm_ref = None

    def _ukm(self):
        if self._ukm_ref is None:
            from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

            self._ukm_ref = UnifiedKnowledgeManager()
        return self._ukm_ref

    def _store_finc_raw(self, query, params, raw):
        """Dual-write the per-term finc payload to the raw cache. Best-effort. - Claude Generated"""
        override = self._config.get("cache_responses")
        if override is not None:
            enabled = bool(override)
        else:
            enabled = self._cache_raw if self._cache_raw is not None else True
        if not enabled:
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

        return FincSuggester(
            base_url=self._config.get("base_url", "") or "",
            web_record_url=self._config.get("web_record_url", "") or "",
            institution_filter=self._config.get("institution_filter", "") or "",
            debug=self._config.get("debug", False),
        )

    @property
    def suggester(self):
        if self._suggester is None:
            self._suggester = self._build_suggester()
        return self._suggester

    def is_available(self, cfg: Any = None) -> bool:
        # finc is off until a base URL is configured (base_url gates availability).
        return availability_ok(type(self).config_fields(), self._config)

    def _require(self, capability: SearchCapability) -> None:
        if capability not in self.capabilities:
            raise ValueError(
                f"Provider '{self.id}' does not support {capability}; "
                f"declares {sorted(c.value for c in self.capabilities)}"
            )

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
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        if progress is not None:
            try:
                self.suggester.currentTerm.connect(progress)
            except Exception:
                pass
        raw = self.suggester.search(
            list(query),
            search_type=search_type,
            filters=filters,
            limit=limit,
            facets=facets,
        )
        errors = dict(getattr(self.suggester, "last_errors", {}) or {})
        from src.core.search.provider import raw_cache_params_for
        self._store_finc_raw(
            list(query), raw_cache_params_for("finc", search_type=search_type, facets=facets), raw
        )
        if capability is SearchCapability.TITLE_RECORDS:
            return ProviderResult.from_finc_records(raw, errors=errors)
        return _facets_from_finc_dict(raw, errors=errors)


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
