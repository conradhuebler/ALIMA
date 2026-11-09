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

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from src.core.plugins.schema import INT, TEXT, ConfigField, PluginDoc

from src.core.search import source_health
from src.core.search.cache_only import CACHE_MISS_MESSAGE, cache_only_field
from src.core.search.provider import (
    ProviderResult,
    ProviderToolSpec,
    ResultItem,
    SearchCapability,
    raw_cache_params_for,
)
from src.core.search.provider_base import NetworkSourceMixin
from src.core.search.registry import register_provider
from src.core.search.timeouts import timeout_field, timeout_from

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_CATALOGS,
    SEARCH_PARAMS,
    KvkClient,
    merge_round_robin,
)

logger = logging.getLogger(__name__)

_SEARCH_TYPES = ["kw", "title", "author", "subject", "isbn"]

# The KVK keeps the key it has always had (``timeout``), with the shared bounds.
# - Claude Generated
_TIMEOUT_KEY = "timeout"
_DEFAULT_TIMEOUT_S = 30


@register_provider
class KvkProvider(NetworkSourceMixin):
    """Karlsruher Virtueller Katalog — title records across German union catalogs.

    Raw cache (WP2 seam): each answered term is stored as its parsed records
    plus the per-catalog stats under ``kvk`` / ``search_type``, which is what
    the cache-only mode reads. The catalog selection is not part of the key.
    Written only for a factory-built instance (``_cache_raw`` set), so a
    provider made directly (tests, scripts) never touches the shared
    database. - Claude Generated
    """

    id = "kvk"
    label = "KVK (Verbundkataloge)"
    capabilities = {SearchCapability.TITLE_RECORDS}
    # Axes of the joint title search (search_titles). - Claude Generated
    title_search_axes = {axis: axis for axis in _SEARCH_TYPES}

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
            timeout_field(_DEFAULT_TIMEOUT_S, key=_TIMEOUT_KEY),
            ConfigField(
                key="max_results", label="Max. Datensätze je Begriff", kind=INT, default=25,
                help="Deckelt die zusammengeführte Trefferliste. Der KVK selbst liefert "
                "je Katalog nur die erste Seite.",
            ),
            cache_only_field(),
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
                "catalog). When a catalog delivered fewer records than it reported, "
                "a catalog answered with an error other than 'no hits', or "
                "`max_results` cut the merged list, the result carries `notice`, one "
                "line in plain words saying which; read it before treating the list "
                "as complete."
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
            timeout=timeout_from(self._config, _DEFAULT_TIMEOUT_S, key=_TIMEOUT_KEY),
            source_id=self.source_id,
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
        params = raw_cache_params_for(self.id, search_type=search_type)
        if self.cache_only:
            return self._search_cache_only(list(query), params, limit)
        answered: Dict[str, Dict[str, Any]] = {}

        for term in query:
            if source_health.is_down(self.source_id):
                # Skipped for this run: unknown, not empty. - Claude Generated
                errors[term] = source_health.down_message(self.source_id)
                per_term[term] = []
                continue
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
            answered[term] = {
                "records": records, "catalog_errors": catalog_errors, "catalog_stats": stats,
            }
            per_term[term], per_term_meta[term] = _shape(records, catalog_errors, stats, limit)
        self._store_kvk_raw(params, answered)
        return ProviderResult(
            SearchCapability.TITLE_RECORDS,
            per_term=per_term,
            per_term_meta=per_term_meta,
            errors=errors,
        )

    def tool_notice(self, result: ProviderResult, *, tools: Any = ()) -> str:
        """The ``notice`` of the agent tool: cut or failed catalogs, in words.

        Empty when every catalog delivered all it reported and none failed.
        Names ``search_k10plus`` only when a K10plus catalog was cut and that
        tool exists (``tools`` = the tool names of the registry). - Claude Generated
        """
        meta = dict(result.per_term_meta or {})
        k10plus_tool = next((t for t in sorted(tools or ()) if str(t).startswith("search_k10plus")), "")
        lines: List[str] = []
        k10plus_cut = False
        for term, term_meta in meta.items():
            parts, cut_k10 = catalog_notice_parts(term_meta or {})
            k10plus_cut = k10plus_cut or cut_k10
            if parts:
                lines.append(("; ".join(parts)) if len(meta) == 1 else f"„{term}“: " + "; ".join(parts))
        if not lines:
            return ""
        text = " | ".join(lines) + "."
        if k10plus_cut and k10plus_tool:
            text += f" Für vollständige K10plus-Treffer {k10plus_tool} nutzen."
        return text

    def _store_kvk_raw(self, params: Dict[str, Any], answered: Dict[str, Dict[str, Any]]) -> None:
        """Write the answered terms to the raw cache; best-effort - Claude Generated"""
        if self._cache_raw is None or not answered or not self._cache_raw_enabled():
            return
        try:
            ukm = self._ukm()
        except Exception:
            return
        for term, entry in answered.items():
            try:
                blob = json.dumps(entry, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                continue
            ukm.store_raw_response(self.id, term, params, blob,
                                   result_count=len(entry.get("records") or []))

    def _search_cache_only(
        self, query: List[str], params: Dict[str, Any], limit: int
    ) -> ProviderResult:
        """``TITLE_RECORDS`` from the raw cache alone, any age, no request.

        A term without an entry carries the cache-only marker in ``errors``
        (unknown, not "no match"); nothing is written back. - Claude Generated
        """
        per_term: Dict[str, List[ResultItem]] = {}
        per_term_meta: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        for term in query:
            entry = self._cached_raw(self.id, term, params)
            records = entry.get("records") if isinstance(entry, dict) else None
            if not isinstance(records, list):
                errors[term] = CACHE_MISS_MESSAGE
                per_term[term] = []
                continue
            per_term[term], per_term_meta[term] = _shape(
                records, entry.get("catalog_errors") or [], entry.get("catalog_stats") or [], limit
            )
        return ProviderResult(
            SearchCapability.TITLE_RECORDS,
            per_term=per_term,
            per_term_meta=per_term_meta,
            errors=errors,
        )


# KVK message of a catalog that answered with no hits (not a failure).
_NO_HITS_MESSAGES = ("keine datensätze gefunden",)


def _short_catalog(name: Any) -> str:
    """``"K10plus - Verbundkatalog von GBV und SWB"`` → ``"K10plus"``."""
    return str(name or "").split(" - ")[0].strip() or str(name or "")


def catalog_notice_parts(meta: Dict[str, Any]) -> "tuple[List[str], bool]":
    """``(parts, k10plus_cut)`` of one term's notice - Claude Generated

    A catalog counts as cut when the KVK marks it truncated or it reported
    more hits than it delivered. A catalog error other than "Keine Datensätze
    gefunden" is quoted with the KVK's own message, grouped by message. The
    merged list cut by ``max_results`` is named too.
    """
    parts: List[str] = []
    k10plus_cut = False
    delivered = 0
    for stat in meta.get("catalog_stats") or []:
        results = int(stat.get("results") or 0)
        returned = int(stat.get("returned") or 0)
        delivered += returned
        if not (stat.get("truncated") or results > returned):
            continue
        name = _short_catalog(stat.get("catalog"))
        if results > returned:
            parts.append(f"{name}: {returned} von {results} Treffern")
        else:
            parts.append(f"{name}: erste Seite ({returned} Treffer), weitere vorhanden")
        if "k10plus" in str(stat.get("catalog") or "").lower():
            k10plus_cut = True
    failed: Dict[str, List[str]] = {}
    for err in meta.get("catalog_errors") or []:
        message = str(err.get("message") or "").strip()
        if any(m in message.lower() for m in _NO_HITS_MESSAGES):
            continue
        failed.setdefault(message.rstrip(".") or "ohne Meldung", []).append(
            _short_catalog(err.get("catalog")))
    for message, names in failed.items():
        parts.append(f"{', '.join(names)}: „{message}“")
    merged = int(meta.get("returned") or 0)
    if delivered > merged:
        parts.append(f"zusammengeführte Liste auf {merged} von {delivered} gelieferten "
                     "Treffern gekürzt (max_results)")
    return parts, k10plus_cut


def _shape(records: list, catalog_errors: list, stats: list, limit: int):
    """``(items, meta)`` of one answered term, capped round-robin - Claude Generated"""
    capped = merge_round_robin(records, limit)
    items = [ResultItem(label=str(rec.get("title", "")), record=rec) for rec in capped]
    meta = {
        "result_count": sum(int(s.get("results") or 0) for s in stats),
        "returned": len(capped),
        "catalog_stats": stats,
        "catalog_errors": catalog_errors,
    }
    return items, meta
