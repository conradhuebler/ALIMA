"""MARC-XML / SRU search provider - Claude Generated.

Brings the SRU/MARC-XML backend (DNB, LoC, GBV, SWB, K10plus, or a custom SRU
endpoint) into the provider standard as a first-class *type* ``sru``. Historically
this lived only as ``CatalogConfig.catalog_type == 'marcxml_sru'`` and was wired by
hand in ``pipeline_utils.execute_dk_search``; as a provider its endpoint/preset live
in its own instance settings.

.. warning::
   This docstring used to claim the ``catalog_type``/``get_catalog_type()`` heuristic
   was "replaced (Debt D-5)". **That is false** (verified July 16): whether this
   provider runs as the DK backend is *still* decided by
   ``CatalogConfig.catalog_type == 'marcxml_sru'`` in ``factory.resolve_dk_extractor``,
   i.e. by a field on the **catalog** plugin. WP Plugin-Konvergenz **P4** gives ``sru``
   its own ``dk_enabled`` (symmetric to finc) and makes ``catalog_type`` vestigial.

Wraps :class:`MarcXmlClient`. Declares TITLE_RECORDS (bibliographic records via
SRU search) and CLASSIFICATION (DK/RVK extraction). No MCP tool is declared: SRU
has always been a DK-search backend, not an agentic search tool, so the tool
surface is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from src.core.plugins.schema import BOOL, CHOICE, INT, TEXT, URL, ConfigField

from src.core.search.provider import ProviderResult, ResultItem, SearchCapability
from src.core.search.registry import register_provider

_PRESETS = ["", "dnb", "loc", "gbv", "swb", "k10plus"]
_SCHEMAS = ["marcxml", "MARC21-xml"]


@register_provider
class SruProvider:
    """SRU / MARC-XML catalog (title records + DK/RVK classification)."""

    id = "sru"
    label = "SRU / MARC-XML"
    capabilities = {SearchCapability.TITLE_RECORDS, SearchCapability.CLASSIFICATION}

    def __init__(self, **config: Any):
        self._config = config or {}
        self._client = None

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(
                key="preset", label="Preset", kind=CHOICE, choices=_PRESETS, default="",
                help="Bekannter SRU-Endpoint (dnb/loc/gbv/swb/k10plus) — überschreibt die Basis-URL.",
            ),
            ConfigField(
                key="base_url", label="SRU Basis-URL", kind=URL,
                help="Eigener SRU-Endpoint, falls kein Preset gewählt ist.",
            ),
            ConfigField(key="database", label="Datenbank", kind=TEXT, help="Optionaler SRU-DB-Name."),
            ConfigField(key="schema", label="Record-Schema", kind=CHOICE, choices=_SCHEMAS, default="marcxml"),
            ConfigField(key="max_records", label="Max. Datensätze", kind=INT, default=50),
            ConfigField(
                key="dk_enabled", label="SRU für DK-Suche nutzen", kind=BOOL, default=False,
                help="MARC-XML/SRU als DK/RVK-Backend statt Libero (symmetrisch zu finc). "
                "Ersetzt das alte catalog_type='marcxml_sru'.",
            ),
        ]

    @classmethod
    def doc(cls):
        from src.core.plugins.schema import PluginDoc

        return PluginDoc(
            description="SRU/MARC-XML-Katalog (DNB, LoC, GBV, SWB, K10plus oder eigener "
            "SRU-Endpoint): Titel-Datensätze und DK/RVK-Extraktion.",
            input="Suchbegriffe/Keywords.",
            output="MARC-Titel-Datensätze bzw. DK/RVK-Klassifikationen je Keyword.",
        )

    def _build_client(self):
        from src.utils.clients.marcxml_client import MarcXmlClient

        return MarcXmlClient(
            sru_base_url=self._config.get("base_url", "") or "",
            database=self._config.get("database", "") or "",
            schema=self._config.get("schema", "marcxml") or "marcxml",
            preset=self._config.get("preset", "") or "",
            max_records=int(self._config.get("max_records", 50) or 50),
            debug=bool(self._config.get("debug", False)),
        )

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def dk_extractor(self, **_ignore: Any):
        """Return the MarcXmlClient backing the ``CLASSIFICATION`` capability — the
        classic DK step's ``extract_dk_classifications_for_keywords`` backend
        (SRU/MARC-XML). - Claude Generated"""
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

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        search_type: str = "keyword",
        max_results: int = 50,
        **opts: Any,
    ) -> ProviderResult:
        self._require(capability)
        if capability is SearchCapability.TITLE_RECORDS:
            per_term = {}
            per_term_meta = {}
            for term in query:
                if progress is not None:
                    try:
                        progress(term)
                    except Exception:
                        pass
                records = self.client.search(term, search_type=search_type) or []
                per_term[term] = [
                    ResultItem(label=str((r or {}).get("title", "")), record=r) for r in records
                ]
                per_term_meta[term] = {"result_count": len(records)}
            return ProviderResult(
                SearchCapability.TITLE_RECORDS, per_term=per_term, per_term_meta=per_term_meta
            )
        # CLASSIFICATION
        raw = self.client.extract_dk_classifications_for_keywords(list(query), max_results=max_results)
        return _classification_from_sru_list(raw)


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
