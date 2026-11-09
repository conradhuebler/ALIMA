"""SRU search provider (MARC-XML endpoints, K10plus in PICA-XML) - Claude Generated.

Brings the SRU backend (DNB, LoC, GBV, SWB, K10plus, or a custom SRU endpoint)
into the provider standard as a first-class *type* ``sru``; endpoint/preset
live in its own instance settings. Whether it is a DK source is the
instance's own ``dk_enabled`` (``factory.dk_candidates``).

Declares TITLE_RECORDS and CLASSIFICATION. Two record paths:

* preset ``k10plus``: the K10plus SRU database with local data
  (``picaxml_client.K10PLUS_SRU_URL``), always ``recordSchema=picaxml``, read by
  :mod:`src.utils.clients.picaxml_client`. The agent tool is ``search_k10plus``;
  with an ELN set (``src/utils/institution.py``) a record with a copy of the
  library carries a ``local`` block, and the DK extractor counts the library's
  local notations (``PicaDkExtractor``).
* every other preset or a custom URL: :class:`MarcXmlClient` with the
  instance's schema; the agent tool is ``search_sru``, the DK extractor the
  MARC one. The MARC DK extraction of the ``k10plus`` preset without ELN still
  queries ``MarcXmlClient.KNOWN_ENDPOINTS["k10plus"]`` (``opac-de-627``).

Raw cache (WP2 seam): answered title searches and PICA DK searches are stored
under source ``sru`` with endpoint, schema and search type in the key, only
for a factory-built instance (``_cache_raw`` set). The state "Nur Cache"
(``cache_only``) answers from there without any request; the MARC DK
extraction has no cache, so every keyword is a miss in that state.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from src.core.plugins.schema import BOOL, CHOICE, INT, TEXT, URL, ConfigField

from src.core.search import source_health
from src.core.search.cache_only import CACHE_MISS_MESSAGE, cache_only_field
from src.core.search.provider import (
    ProviderResult,
    ProviderToolSpec,
    ResultItem,
    SearchCapability,
)
from src.core.search.provider_base import NetworkSourceMixin
from src.core.search.registry import register_provider
from src.core.search.timeouts import timeout_field, timeout_from

logger = logging.getLogger(__name__)

# MarcXmlClient's request timeout before it became the instance setting
# ``timeout_s``. - Claude Generated
_DEFAULT_TIMEOUT_S = 30

_PRESETS = ["", "dnb", "loc", "gbv", "swb", "k10plus"]
_SCHEMAS = ["marcxml", "MARC21-xml"]
_PICA_PRESET = "k10plus"

# Search axes of the generic (MARC) tool → MarcXmlClient search types.
_MARC_SEARCH_TYPES = {"kw": "keyword", "title": "title", "author": "author",
                      "subject": "subject", "isbn": "isbn"}
_MARC_DEFAULT_MAX = 10
_MARC_MAX_LIMIT = 50

# Source label of the raw-cache entries. - Claude Generated
_RAW_SOURCE = "sru"


def _k10plus_tool_spec() -> ProviderToolSpec:
    """The ``search_k10plus`` tool, described with the ELN in use - Claude Generated"""
    from src.utils.clients import picaxml_client as pica
    from src.utils.institution import institution_eln, institution_filter

    eln = institution_eln()
    isil = institution_filter()
    if eln:
        local_text = (
            f"Local data: a record with a copy of this library (ELN {eln}, field 209A $B) "
            "carries `local` = {signature, notations: [{text, ppn}] (field 245Z of that "
            "copy, e.g. 'DK 541.5' with the K10plus PPN of the notation), subject_chains "
            + (f"(044K chains closed by $A{isil})" if isil else
               "(empty: no ISIL is set, see subject_chains_hint)")
            + "}. A record without such a copy carries `local_hint` instead."
        )
    else:
        local_text = (
            "Local data: no ELN is set for this installation, so no record carries "
            "`local`; `local_hint` says so."
        )
    return ProviderToolSpec(
        name="search_k10plus",
        capability=SearchCapability.TITLE_RECORDS,
        description=(
            "Search the K10plus union catalogue (GBV/SWB) directly over SRU in PICA "
            "format (" + pica.K10PLUS_SRU_URL + "). `terms` are searched independently, "
            "results are keyed per term. `search_type`: kw = all words (pica.all, "
            "default), title (pica.tit), author (pica.per, a phrase in the form "
            "'Surname, Forename'; a value without a comma and of exactly two words is "
            "turned around, 'Conrad Hübler' -> 'Hübler, Conrad'; any other value is "
            "sent unchanged), subject (pica.slw), isbn (pica.isb), ppn (pica.ppn). "
            "Each record has ppn, title, authors [{name, role, gnd_id}], year, "
            "publisher, isbn, form (form headings, field 013D), thesis (field 037C), "
            "ddc, subjects (subject chains of field 044K: [{terms: [{term, gnd_id, "
            "ppn}], source}], source = the ISIL that closed the chain, empty when none "
            "did). " + local_text + " "
            "Per term, `meta` gives `total` (numberOfRecords of the server), "
            "`returned`, and `truncated: true` when more records exist than were "
            "returned; raise `max_results` (at most 50) or narrow the query to see "
            "them. `errors` lists terms whose search failed (unknown, not 'no hits')."
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
                    "enum": list(pica.SEARCH_INDEXES),
                    "default": pica.DEFAULT_SEARCH_TYPE,
                    "description": (
                        "kw = all words; title; author ('Surname, Forename'); subject "
                        "(Schlagwort); isbn; ppn (K10plus record id)."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "default": pica.DEFAULT_MAX_RESULTS,
                    "minimum": 1,
                    "maximum": pica.MAX_RESULTS_LIMIT,
                    "description": "Records per term (default 10, at most 50).",
                },
            },
            "required": ["terms"],
        },
        result_shape="title_records",
        source_label="k10plus",
        include_errors=True,
        unavailable_message="K10plus not available",
        default_opts={"search_type": pica.DEFAULT_SEARCH_TYPE,
                      "max_results": pica.DEFAULT_MAX_RESULTS},
    )


@register_provider
class SruProvider(NetworkSourceMixin):
    """SRU catalog (title records + DK/RVK classification)."""

    id = "sru"
    label = "SRU / MARC-XML"
    capabilities = {SearchCapability.TITLE_RECORDS, SearchCapability.CLASSIFICATION}
    # Axes of the joint title search (search_titles): the same names on the
    # PICA and the MARC path. - Claude Generated
    title_search_axes = {"kw": "kw", "title": "title", "author": "author",
                         "subject": "subject", "isbn": "isbn"}

    def __init__(self, **config: Any):
        self._config = config or {}
        self._client = None
        self._pica_client = None

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(
                key="preset", label="Preset", kind=CHOICE, choices=_PRESETS, default="",
                help="Bekannter SRU-Endpoint (dnb/loc/gbv/swb/k10plus) — überschreibt die Basis-URL. "
                "k10plus: Titelsuche im PICA-Format mit lokalen Daten (Werkzeug search_k10plus).",
            ),
            ConfigField(
                key="base_url", label="SRU Basis-URL", kind=URL,
                help="Eigener SRU-Endpoint, falls kein Preset gewählt ist.",
            ),
            ConfigField(key="database", label="Datenbank", kind=TEXT, help="Optionaler SRU-DB-Name."),
            ConfigField(
                key="schema", label="Record-Schema", kind=CHOICE, choices=_SCHEMAS,
                default="marcxml",
                help="Gilt nur für einen eigenen Endpunkt — ein gewähltes Preset "
                "bringt sein eigenes Schema mit und überschreibt diesen Wert.",
            ),
            ConfigField(key="max_records", label="Max. Datensätze", kind=INT, default=50),
            timeout_field(_DEFAULT_TIMEOUT_S),
            ConfigField(
                key="dk_enabled", label="SRU für DK-Suche nutzen", kind=BOOL, default=False,
                help="Dieser SRU-Endpunkt ist DK-Quelle. Mehrere DK-Quellen fragt der "
                "DK-Schritt in der Reihenfolge der Liste (Rückfall je Schlagwort). Mit "
                "Preset k10plus und gesetzter ELN zählt sie die lokalen Notationen der "
                "Bibliothek.",
            ),
            cache_only_field(),
        ]

    @classmethod
    def doc(cls):
        from src.core.plugins.schema import PluginDoc

        return PluginDoc(
            description="SRU-Katalog (DNB, LoC, GBV, SWB, K10plus oder eigener "
            "SRU-Endpoint): Titel-Datensätze und DK/RVK-Extraktion. Preset k10plus "
            "liest PICA-XML mit den lokalen Daten der Bibliothek (ELN aus der Einrichtung).",
            input="Suchbegriffe/Keywords.",
            output="Titel-Datensätze bzw. DK/RVK-Klassifikationen je Keyword.",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[ProviderToolSpec]:
        """The generic SRU tool; :meth:`instance_tool_spec` turns it into
        ``search_k10plus`` for an instance with preset k10plus. - Claude Generated"""
        return [ProviderToolSpec(
            name="search_sru",
            capability=SearchCapability.TITLE_RECORDS,
            description=(
                "Search an SRU library catalogue (MARC-XML) for title records. `terms` "
                "are searched independently, results are keyed per term. `search_type`: "
                "kw (all words, default), title, author, subject, isbn. Each record has "
                "rsn, title, author, publication, isbn, classifications, subjects, "
                "gnd_subjects. Per term, `meta` gives `total` (numberOfRecords of the "
                "server, when it sends one), `returned`, and `truncated: true` when more "
                "records exist than were returned. `errors` lists terms whose search "
                "failed."
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
                        "enum": list(_MARC_SEARCH_TYPES),
                        "default": "kw",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": _MARC_DEFAULT_MAX,
                        "minimum": 1,
                        "maximum": _MARC_MAX_LIMIT,
                        "description": "Records per term (default 10, at most 50).",
                    },
                },
                "required": ["terms"],
            },
            result_shape="title_records",
            source_label="sru",
            include_errors=True,
            unavailable_message="SRU not available (no preset or base URL configured)",
            default_opts={"search_type": "kw", "max_results": _MARC_DEFAULT_MAX},
        )]

    @classmethod
    def instance_tool_spec(cls, spec: ProviderToolSpec, settings: Any) -> Optional[ProviderToolSpec]:
        """The tool of one instance: ``search_k10plus`` for preset k10plus, the
        generic ``search_sru`` for another preset or a base URL, none for an
        instance without endpoint. - Claude Generated"""
        settings = settings or {}
        if not (settings.get("preset") or settings.get("base_url")):
            return None
        if str(settings.get("preset") or "").lower() != _PICA_PRESET:
            return spec
        return dataclasses.replace(_k10plus_tool_spec(), provider_id=spec.provider_id)

    # ------------------------------------------------------------------
    @property
    def uses_pica(self) -> bool:
        """Preset k10plus: records come from the K10plus in PICA-XML."""
        return str(self._config.get("preset") or "").lower() == _PICA_PRESET

    def _build_client(self):
        from src.utils.clients.marcxml_client import MarcXmlClient

        return MarcXmlClient(
            sru_base_url=self._config.get("base_url", "") or "",
            database=self._config.get("database", "") or "",
            schema=self._config.get("schema", "marcxml") or "marcxml",
            preset=self._config.get("preset", "") or "",
            max_records=int(self._config.get("max_records", 50) or 50),
            debug=bool(self._config.get("debug", False)),
            timeout=timeout_from(self._config, _DEFAULT_TIMEOUT_S),
            source_id=self.source_id,
            cache_only=self.cache_only,
        )

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @property
    def pica_client(self):
        """The K10plus PICA-XML client (preset k10plus) - Claude Generated"""
        if self._pica_client is None:
            from src.utils.clients.picaxml_client import PicaXmlClient

            self._pica_client = PicaXmlClient(
                timeout=timeout_from(self._config, _DEFAULT_TIMEOUT_S),
                source_id=self.source_id,
            )
        return self._pica_client

    def dk_extractor(self, **_ignore: Any):
        """The ``CLASSIFICATION`` backend of the classic DK step.

        Preset k10plus with an ELN set: :class:`PicaDkExtractor` (local
        notations of the library's copies). Otherwise the MarcXmlClient.
        - Claude Generated
        """
        from src.utils.institution import institution_eln

        eln = institution_eln()
        if self.uses_pica and eln:
            from src.utils.clients.picaxml_client import PicaDkExtractor

            return PicaDkExtractor(
                self.pica_client, eln, cache_only=self.cache_only,
                cache_read=self._pica_cache_read, cache_write=self._pica_cache_write,
            )
        return self.client

    def is_available(self, cfg: Any = None) -> bool:
        # Available once either a preset or a custom base URL is configured (OR
        # semantics, so a single gating field cannot express it).
        return bool(self._config.get("preset") or self._config.get("base_url"))

    def _require(self, capability: SearchCapability) -> None:
        if capability not in self.capabilities:
            raise ValueError(
                f"Provider '{self.id}' does not support {capability}; "
                f"declares {sorted(c.value for c in self.capabilities)}"
            )

    # --- raw cache ------------------------------------------------------
    def _endpoint(self) -> str:
        if self.uses_pica:
            from src.utils.clients.picaxml_client import K10PLUS_SRU_URL

            return K10PLUS_SRU_URL
        return str(getattr(self.client, "sru_base_url", "") or "")

    def _raw_params(self, search_type: str, schema: str) -> Dict[str, Any]:
        return {"search_type": search_type, "endpoint": self._endpoint(), "schema": schema}

    def _store_answer(self, term: str, params: Dict[str, Any], answer: Dict[str, Any]) -> None:
        """Write one answered term to the raw cache; best-effort, only for a
        factory-built instance. - Claude Generated"""
        if self._cache_raw is None or not self._cache_raw_enabled():
            return
        try:
            blob = json.dumps({"records": answer.get("records") or [],
                               "total": answer.get("total")}, ensure_ascii=False, default=str)
            self._ukm().store_raw_response(_RAW_SOURCE, term, params, blob,
                                           result_count=len(answer.get("records") or []))
        except Exception as exc:  # noqa: BLE001 - a cache failure never breaks a search
            from src.utils.error_visibility import log_caught

            log_caught(logger, exc, "sru raw-cache write", detail=term)

    def _read_answer(self, term: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        entry = self._cached_raw(_RAW_SOURCE, term, params)
        if not isinstance(entry, dict) or not isinstance(entry.get("records"), list):
            return None
        return entry

    def _pica_cache_read(self, term: str, search_type: str) -> Optional[Dict[str, Any]]:
        from src.utils.clients.picaxml_client import RECORD_SCHEMA

        return self._read_answer(term, self._raw_params(search_type, RECORD_SCHEMA))

    def _pica_cache_write(self, term: str, search_type: str, answer: Dict[str, Any]) -> None:
        from src.utils.clients.picaxml_client import RECORD_SCHEMA

        self._store_answer(term, self._raw_params(search_type, RECORD_SCHEMA), answer)

    # --- search ---------------------------------------------------------
    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        search_type: Optional[str] = None,
        max_results: Optional[int] = None,
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        if capability is SearchCapability.TITLE_RECORDS:
            if self.uses_pica:
                return self._search_pica(list(query), search_type, max_results, progress)
            return self._search_marc(list(query), search_type, max_results, progress)
        # CLASSIFICATION
        raw = self.dk_extractor().extract_dk_classifications_for_keywords(
            list(query), max_results=int(max_results or 50))
        return _classification_from_sru_list(raw)

    @staticmethod
    def _progress(progress, term) -> None:
        if progress is not None:
            try:
                progress(term)
            except Exception:  # noqa: BLE001 - a broken progress sink must not stop a search
                logger.debug("SRU progress callback failed", exc_info=True)

    def _search_pica(self, query, search_type, max_results, progress) -> ProviderResult:
        """K10plus in PICA-XML: records with ``local``/``local_hint`` - Claude Generated"""
        from src.utils.clients import picaxml_client as pica
        from src.utils.institution import institution_eln, institution_filter

        per_term: Dict[str, List[ResultItem]] = {}
        per_term_meta: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        limit = pica.clamp_max_results(max_results)
        try:
            st = pica.normalize_search_type(search_type)
        except ValueError as exc:
            return ProviderResult(SearchCapability.TITLE_RECORDS,
                                  per_term={t: [] for t in query},
                                  errors={t: str(exc) for t in query})
        params = self._raw_params(st, pica.RECORD_SCHEMA)
        eln, isil = institution_eln(), institution_filter()
        for term in query:
            per_term[term] = []
            if self.cache_only:
                answer = self._read_answer(term, params)
                if answer is None:
                    errors[term] = CACHE_MISS_MESSAGE
                    continue
            else:
                if source_health.is_down(self.source_id):
                    errors[term] = source_health.down_message(self.source_id)
                    continue
                self._progress(progress, term)
                try:
                    answer = self.pica_client.search(term, search_type=st, max_results=limit)
                except Exception as exc:  # noqa: BLE001 - per-term failure of the source
                    errors[term] = str(exc)
                    continue
                self._store_answer(term, params, answer)
            records = list(answer.get("records") or [])[:limit]
            per_term[term] = [
                ResultItem(label=str(r.get("title") or ""), record=pica.shape_record(r, eln, isil))
                for r in records
            ]
            per_term_meta[term] = _total_meta(answer.get("total"), len(records))
        return ProviderResult(SearchCapability.TITLE_RECORDS, per_term=per_term,
                              per_term_meta=per_term_meta, errors=errors)

    def _search_marc(self, query, search_type, max_results, progress) -> ProviderResult:
        """MARC-XML endpoint through MarcXmlClient - Claude Generated"""
        per_term: Dict[str, List[ResultItem]] = {}
        per_term_meta: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        key = str(search_type or "kw").strip().lower()
        marc_type = _MARC_SEARCH_TYPES.get(key, key if key in _MARC_SEARCH_TYPES.values() else None)
        if marc_type is None:
            message = (f"SRU: unknown search_type {search_type!r} "
                       f"(known: {', '.join(_MARC_SEARCH_TYPES)})")
            return ProviderResult(SearchCapability.TITLE_RECORDS,
                                  per_term={t: [] for t in query},
                                  errors={t: message for t in query})
        limit = max(1, min(_MARC_MAX_LIMIT, int(max_results or _MARC_DEFAULT_MAX)))
        schema = str(getattr(self.client, "schema", "") or "")
        params = self._raw_params(marc_type, schema)
        for term in query:
            per_term[term] = []
            if self.cache_only:
                answer = self._read_answer(term, params)
                if answer is None:
                    errors[term] = CACHE_MISS_MESSAGE
                    continue
            else:
                if source_health.is_down(self.source_id):
                    # Skipped for this run: unknown, not empty. - Claude Generated
                    errors[term] = source_health.down_message(self.source_id)
                    continue
                self._progress(progress, term)
                client = self.client
                old_max = client.max_records
                client.max_records = limit
                try:
                    records = client.search(term, search_type=marc_type) or []
                finally:
                    client.max_records = old_max
                answer = {"records": records, "total": client.last_number_of_records}
                if records:
                    self._store_answer(term, params, answer)
            records = list(answer.get("records") or [])[:limit]
            per_term[term] = [ResultItem(label=str((r or {}).get("title", "")), record=r)
                              for r in records]
            per_term_meta[term] = _total_meta(answer.get("total"), len(records))
        return ProviderResult(SearchCapability.TITLE_RECORDS, per_term=per_term,
                              per_term_meta=per_term_meta, errors=errors)


def _total_meta(total: Any, returned: int) -> Dict[str, Any]:
    """``{result_count, total, returned, truncated?}`` of one term - Claude Generated

    ``truncated: true`` only when the server's total exceeds what came back.
    """
    meta: Dict[str, Any] = {"result_count": returned, "returned": returned}
    try:
        total_int = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_int = None
    if total_int is not None:
        meta["total"] = total_int
        if total_int > returned:
            meta["truncated"] = True
    return meta


def _classification_from_sru_list(raw: list) -> ProviderResult:
    """Convert ``extract_dk_classifications_for_keywords`` output to a result.

    Shape: ``[{keyword, classifications: [{dk, classification_type, count, ...}]}]``.
    Each inner classification becomes a ``ResultItem`` keyed under its keyword.
    """
    per_term: dict = {}
    for entry in raw or []:
        entry = entry or {}
        keyword = str(entry.get("keyword", "_all"))
        items = []
        for c in entry.get("classifications", []) or []:
            c = c or {}
            items.append(
                ResultItem(
                    code=str(c.get("dk", "")),
                    label=str(c.get("dk", "")),
                    count=int(c.get("count", 0) or 0),
                    extra=dict(c),
                )
            )
        per_term.setdefault(keyword, []).extend(items)
    return ProviderResult(SearchCapability.CLASSIFICATION, per_term=per_term)
