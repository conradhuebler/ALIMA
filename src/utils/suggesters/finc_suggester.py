"""finc_suggester.py - Suggester-Wrapper around FincClient.

Provides a BaseSuggester-conforming interface for the finc / VuFind-JSON
catalog API. Unlike LobidSuggester / SWBSuggester (which return aggregated
GND/DK/DDC data per term) or BiblioSuggester (which returns the legacy
{count, gndid, ddc, dk} dict shape), FincSuggester returns **records**:
each search term maps to a list of normalized VuFind records plus a
result_count, mirroring the operator-approved Biblio-style MCP tool
output.

Per-base-class contract: `search(terms, search_type)` must return
`{term: {subkey: data}}` so that MetaSuggester (and other consumers that
expect the SuggesterResult shape) can iterate uniformly. We use the
subkey 'records' for the list of records, 'result_count' for the count,
and 'errors' for per-term failures (matching the last_errors pattern in
LobidSuggester).

Claude Generated (finc integration, June 2026).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_suggester import BaseSuggester, BaseSuggesterError
from ..clients.finc_client import FincClient


class FincSuggesterError(BaseSuggesterError):
    """Exception raised for errors in the FincSuggester."""


# search_type -> VuFind `type` parameter mapping.
# The base class enum is "kw" / "title" / "freetext"; finc's API uses the
# more explicit "Subject" / "Author" / "Title" / "AllFields". The LLM and
# pipeline callers keep using the abstract enum; we translate here. - Claude Generated
_SEARCH_TYPE_TO_VUFIND = {
    "kw": "AllFields",
    "freetext": "AllFields",
    "title": "Title",
    "subject": "Subject",
    "author": "Author",
    "dk": "udk_raw_de105",
    "rvk": "rvk_facet",
}


class FincSuggester(BaseSuggester):
    """Suggester that queries a finc / VuFind-JSON /api/v1/search endpoint.

    Returns record lists per term, not aggregated GND/DK statistics.
    """

    def __init__(
        self,
        base_url: str = "",
        web_record_url: str = "",
        default_limit: int = FincClient.DEFAULT_LIMIT,
        timeout: int = FincClient.DEFAULT_TIMEOUT,
        institution_filter: str = "",
        debug: bool = False,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the FincSuggester.

        Args:
            base_url: finc solrproxy base URL (no trailing slash).
            web_record_url: optional record base URL used to build web_url.
            default_limit: default cap on records per search.
            timeout: HTTP timeout in seconds.
            institution_filter: optional default facet filter (e.g.
                "DE-105"); applied as `filter[]=institution_facet:"DE-105"`.
            debug: enable verbose logging.
            data_dir: storage directory (required by BaseSuggester; finc
                itself does not cache).
        """
        super().__init__(data_dir, debug)
        self.client = FincClient(
            base_url=base_url,
            web_record_url=web_record_url,
            default_limit=default_limit,
            timeout=timeout,
        )
        self.institution_filter = institution_filter or ""
        self.default_filters: Dict[str, str] = {}
        if self.institution_filter:
            # The finc/VuFind facet for the holding institution is `institution`
            # (verified against the live endpoint — `institution_facet` is
            # rejected as "Invalid search"). Operators can override per-call via
            # the `filters` argument. - Claude Generated
            self.default_filters["institution"] = self.institution_filter
        self.logger = logging.getLogger("finc_suggester")

    def prepare(self, force_download: bool = False) -> None:
        """No-op: finc has no local data to download.

        Required by BaseSuggester; matches BiblioSuggester.prepare. - Claude Generated
        """
        return None

    @staticmethod
    def _map_search_type(search_type: str) -> str:
        return _SEARCH_TYPE_TO_VUFIND.get(
            (search_type or "").lower(), "AllFields"
        )

    def search(
        self,
        searches: List[str],
        search_type: str = "kw",
        filters: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        facets: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run a finc search for each term and collect record lists.

        Args:
            searches: list of search terms.
            search_type: "kw" (AllFields) | "title" | "subject" | "author"
                | "freetext". Anything else falls back to AllFields.
            filters: optional per-call facet filter overrides that REPLACE
                the default institution filter (callers that want to keep
                it should pass `{**self.default_filters, **filters}`).
            limit: optional per-call record cap.
            facets: optional list of facet field names to compute per term
                (e.g. ["udk_raw_de105", "rvk_facet"]); surfaced under each
                term's "facets" key.

        Returns:
            Dict of the form
                {
                    term: {
                        "records": [record_dict, ...],
                        "result_count": int,
                        "facets": {facet_name: [{"value","count","translated"}]},
                        "errors": [str, ...]   # per-term failure messages
                    },
                    ...
                }
            On total failure the term is still present with an empty
            `records` list and an `errors` entry.
        """
        self.last_errors = {}
        vufind_type = self._map_search_type(search_type)
        effective_filters = filters if filters is not None else dict(self.default_filters)
        results: Dict[str, Dict[str, Any]] = {}

        for term in searches:
            entry: Dict[str, Any] = {
                "records": [],
                "result_count": 0,
                "facets": {},
                "errors": [],
            }
            try:
                payload = self.client.search(
                    lookfor=term,
                    type=vufind_type,
                    filters=effective_filters,
                    limit=limit,
                    facets=facets,
                )
            except Exception as exc:  # noqa: BLE001 - we re-raise as suggester error
                msg = f"finc search for '{term}' failed: {exc}"
                self._record_search_error(term, exc)
                entry["errors"].append(msg)
                results[term] = entry
                continue

            if payload.get("status") != "OK":
                msg = payload.get("error") or "unknown finc error"
                self._record_search_error(term, msg)
                entry["errors"].append(str(msg))
                results[term] = entry
                continue

            entry["records"] = payload.get("records", [])
            entry["result_count"] = int(payload.get("resultCount", 0))
            entry["facets"] = payload.get("facets", {}) or {}
            results[term] = entry
            self.currentTerm.emit(term)

        return results
