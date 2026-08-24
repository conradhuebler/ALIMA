"""KVK (Karlsruher Virtueller Katalog) search provider - Claude Generated.

The KVK is a meta-search across German union catalogs (K10plus, BVB, HeBIS,
KOBV, NRW, DNB, StaBi Berlin …). Since its search mask gained a JSON variant
(``maske=kvk-json``) it can be read by programs, which is what this provider
does.

**What it can and cannot do.** The KVK JSON carries ``title``, ``author``,
``year``, ``text`` (imprint), the record link and a ``digital`` flag — and
nothing else. There are no subject headings and no notations in the payload, so
this provider declares ``TITLE_RECORDS`` only: it cannot feed the GND keyword
pool or the DK/RVK classification step the way ``lobid``/``catalog``/``finc``
do. What it *is* good at is breadth — one query reaches eight union catalogs at
once, which no other configured source does.

**The bridge to enrichment.** The record link exposes each catalog's own id
(see :func:`client.extract_identifiers`), most usefully a PPN for the
K10plus-family catalogs. Those land in the record's ``identifiers`` so an agent
can hand a hit straight to ``k10plus_resolve`` and get the subjects and DDC the
KVK itself does not deliver.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.core.plugins.schema import INT, TEXT, ConfigField, PluginDoc

from src.core.search.provider import (
    ProviderResult,
    ProviderToolSpec,
    ResultItem,
    SearchCapability,
)
from src.core.search.registry import register_provider

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_CATALOGS,
    SEARCH_PARAMS,
    KvkClient,
    merge_round_robin,
)

logger = logging.getLogger(__name__)

_SEARCH_TYPES = ["kw", "title", "author", "subject", "isbn"]


@register_provider
class KvkProvider:
    """Karlsruher Virtueller Katalog — title records across German union catalogs."""

    id = "kvk"
    label = "KVK (Verbundkataloge)"
    capabilities = {SearchCapability.TITLE_RECORDS}

    def __init__(self, **config: Any):
        self._config = config or {}
        self._client = None

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(
                key="catalogs", label="Kataloge", kind=TEXT,
                default=", ".join(DEFAULT_CATALOGS),
                help="Katalog-Ids wie im KVK-Suchlink (kataloge=…), komma-getrennt. "
                "Die KVK-Startseite ist hinter einer Bot-Challenge, die vollständige "
                "Liste steht also nur dort — Ids aus der eigenen KVK-URL übernehmen.",
            ),
            ConfigField(
                key="base_url", label="KVK-Endpunkt", kind=TEXT, default=DEFAULT_BASE_URL,
                help="CGI-Endpunkt der KVK-Suche.",
            ),
            ConfigField(key="timeout", label="Timeout (s)", kind=INT, default=30),
            ConfigField(
                key="max_results", label="Max. Datensätze je Begriff", kind=INT, default=25,
                help="Deckelt die zusammengeführte Trefferliste. Der KVK selbst liefert "
                "je Katalog nur die erste Seite.",
            ),
        ]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Karlsruher Virtueller Katalog: eine Anfrage über mehrere "
            "deutsche Verbundkataloge (K10plus, BVB, HeBIS, KOBV, NRW, DNB, StaBi). "
            "Liefert Titelnachweise, KEINE Schlagworte und KEINE Notationen — die "
            "KVK-JSON-Antwort enthält sie nicht.",
            input="Suchbegriffe + Suchachse (kw/title/author/subject/isbn).",
            output="Titel-Datensätze je Begriff (Titel, Autor, Jahr, Katalog, Link) "
            "mit den aus dem Link lesbaren Identifiern (ppn/idn/bvnumber).",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[ProviderToolSpec]:
        return [ProviderToolSpec(
            name="search_kvk",
            capability=SearchCapability.TITLE_RECORDS,
            description=(
                "Search the KVK (Karlsruher Virtueller Katalog) — one query across "
                "several German union catalogs (K10plus, BVB, HeBIS, KOBV, NRW, DNB, "
                "StaBi Berlin) at once. Use it to answer 'does this work exist / in "
                "which union catalog is it held', or to find editions the local "
                "catalog does not have. Choose the axis via `search_type`: kw (all "
                "fields), title, author, subject (Schlagwort), isbn. `terms` is "
                "searched independently and results are keyed per term. "
                "IMPORTANT — what this does NOT return: the KVK JSON carries no "
                "subject headings and no DK/RVK/DDC notations. For subjects or "
                "classifications, take a record's `ppn` and call k10plus_resolve, or "
                "use search_finc / search_lobid. "
                "Each record has title, author, year, text (the raw imprint line, "
                "e.g. 'Liu, Shubin. - 1. Auflage. - Bognor Regis : Wiley-VCH, 2026'), "
                "catalog (which union catalog answered), url, digital, and the "
                "identifiers readable from the link: ppn (K10plus/StaBi/KOBV), idn "
                "(DNB) or bvnumber (BVB). A ppn from here is a candidate for "
                "k10plus_resolve, not a guarantee — not every one resolves. "
                "Caveat on links: K10plus record URLs carry a session id and are "
                "short-lived; cite them as [title](url) but do not treat them as "
                "permanent references. "
                "`catalog_errors` lists the union catalogs that returned nothing or "
                "failed, `catalog_stats` how many hits each one reported and whether "
                "the list was truncated (the KVK returns only the first page per "
                "catalog)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more search terms, each searched independently.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": _SEARCH_TYPES,
                        "default": "kw",
                        "description": (
                            "Which index to search: kw = all fields; title = title "
                            "words; author = author name; subject = Schlagwort; "
                            "isbn = ISBN/ISSN."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 25,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum merged records per term.",
                    },
                },
                "required": ["terms"],
            },
            result_shape="title_records",
            source_label="kvk",
            include_errors=True,
            unavailable_message="KVK not available (no catalogs configured)",
        )]

    # ------------------------------------------------------------------
    def _catalogs(self) -> List[str]:
        raw = self._config.get("catalogs")
        if isinstance(raw, (list, tuple)):
            values = [str(v).strip() for v in raw]
        else:
            values = [part.strip() for part in str(raw or "").split(",")]
        return [v for v in values if v] or list(DEFAULT_CATALOGS)

    def _build_client(self) -> KvkClient:
        return KvkClient(
            base_url=str(self._config.get("base_url") or DEFAULT_BASE_URL),
            catalogs=self._catalogs(),
            timeout=int(self._config.get("timeout", 30) or 30),
        )

    @property
    def client(self) -> KvkClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def is_available(self, cfg: Any = None) -> bool:
        """Available as soon as an endpoint and at least one catalog are set —
        the KVK needs no key and no account."""
        return bool(self._config.get("base_url", DEFAULT_BASE_URL)) and bool(self._catalogs())

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
        max_results: int = 25,
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        if str(search_type or "kw").lower() not in SEARCH_PARAMS:
            search_type = "kw"
        limit = max(1, int(max_results or 25))

        per_term: Dict[str, List[ResultItem]] = {}
        per_term_meta: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}

        for term in query:
            if progress is not None:
                try:
                    progress(term)
                except Exception:
                    logger.debug("KVK progress callback failed", exc_info=True)
            try:
                records, catalog_errors, stats = self.client.search(
                    term, search_type=search_type
                )
            except Exception as exc:
                # Transport/HTTP failure = the source failed for this term, which
                # is what ProviderResult.errors means. A catalog that merely found
                # nothing is NOT an error and goes into the metadata below.
                logger.warning("KVK search failed for %r: %s", term, exc)
                errors[term] = str(exc)
                per_term[term] = []
                continue
            capped = merge_round_robin(records, limit)
            per_term[term] = [
                ResultItem(label=str(rec.get("title", "")), record=rec) for rec in capped
            ]
            per_term_meta[term] = {
                "result_count": sum(int(s.get("results") or 0) for s in stats),
                "returned": len(capped),
                "catalog_stats": stats,
                "catalog_errors": catalog_errors,
            }
        return ProviderResult(
            SearchCapability.TITLE_RECORDS,
            per_term=per_term,
            per_term_meta=per_term_meta,
            errors=errors,
        )
