"""finc / VuFind-JSON search provider - Claude Generated.

Brings finc into the standard: instead of being special-cased in the MCP tool
layer, finc is a regular provider declaring ``TITLE_RECORDS`` (bibliographic
records) and ``SUBJECT_FACETS`` (DK/RVK distribution). One underlying finc call
yields both records and facets, so :meth:`search` runs the client once and shapes
the requested capability from the same payload.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..provider import ProviderResult, ResultItem, SearchCapability
from ..registry import register_provider


@register_provider
class FincProvider:
    """finc / VuFind-JSON catalog (records + classification facets)."""

    id = "finc"
    label = "finc (VuFind)"
    capabilities = {SearchCapability.TITLE_RECORDS, SearchCapability.SUBJECT_FACETS}

    def __init__(self, **config: Any):
        self._config = config or {}
        self._suggester = None

    def _build_suggester(self):
        from src.utils.suggesters.finc_suggester import FincSuggester

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
        # finc is off until a base URL is configured (matches tool_registry gating).
        return bool(self._config.get("base_url"))

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
