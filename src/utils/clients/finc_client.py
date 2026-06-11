"""finc_client.py - VuFind-JSON client for finc / finc-solrproxy catalog APIs.

Used by TU Freiberg UB (and other German university libraries running a
finc/VuFind discovery instance) to query the local catalog through the
finc solrproxy. The proxy exposes a small JSON HTTP API at

    {base_url}/api/v1/search?lookfor=...&type=Subject&filter[]=key:value

which returns the same VuFind record shape ALIMA already understands from
BiblioClient (title, authors, subjects, formats, languages, series, urls).

Sits alongside BiblioClient (Libero SOAP) and MarcXmlClient (SRU). The
finc client is the **preferred** catalog source when configured — see
pipeline_utils.execute_dk_search and tool_registry._init_suggesters for
the priority wiring.

Claude Generated (finc integration, June 2026).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class FincClient:
    """HTTP client for finc / VuFind-JSON /api/v1/search endpoints.

    The API is query-string driven:
        lookfor:    search term (wrap phrases in literal double quotes, e.g.
                    "conrad hübler"; requests URL-encodes them automatically —
                    do NOT pass %22 yourself or it double-encodes to %2522)
        type:       'AllFields' | 'Subject' | 'Author' | 'Title'
        filter[]:   repeatable facet filter in VuFind syntax, e.g.
                    'institution_facet:"DE-105"' or
                    'udk_facet_de105:"IT. Informatik. Software"'

    Top-level JSON:  {"status": "OK", "resultCount": int, "records": [...]}
    Each record:     {"id", "title", "authors": {...}, "subjects": [[...]],
                      "formats", "languages", "series": [{name, number}],
                      "urls": [{url, desc, indicators}]}
    """

    SEARCH_PATH = "/api/v1/search"
    DEFAULT_TIMEOUT = 30
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    RATE_LIMIT_DELAY_S = 0.1  # Claude Generated - gentle throttle

    def __init__(
        self,
        base_url: str = "",
        web_record_url: str = "",
        default_limit: int = DEFAULT_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ):
        """Initialize the finc client.

        Args:
            base_url: e.g. "https://finc.example.org/fincsolrproxy/proxy.php"
                      (the client appends /api/v1/search automatically).
            web_record_url: optional record base URL used to build a public
                            web_url for each record. e.g.
                            "https://katalog.example.org/Record/"
                            (trailing slash is added if missing).
            default_limit: default number of records per search (1..100).
            timeout: HTTP timeout in seconds.
            session: optional pre-configured requests.Session (mainly for tests).
        """
        self.base_url = (base_url or "").rstrip("/")
        self.web_record_url = (web_record_url or "").rstrip("/")
        if self.web_record_url:
            self.web_record_url += "/"
        self.default_limit = max(1, min(int(default_limit or self.DEFAULT_LIMIT), self.MAX_LIMIT))
        self.timeout = max(1, int(timeout or self.DEFAULT_TIMEOUT))
        self.session = session or requests.Session()
        self.session.headers.setdefault("Accept", "application/json")
        self.session.headers.setdefault("User-Agent", "ALIMA-FincClient/1.0")
        self._last_search_time: Optional[float] = None
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if the client has a base_url that can be queried."""
        return bool(self.base_url)

    # ------------------------------------------------------------------
    # Rate limiting (lightweight — finc is local + fast)
    # ------------------------------------------------------------------

    def _apply_rate_limit(self) -> None:
        if self._last_search_time is not None:
            elapsed = time.time() - self._last_search_time
            if elapsed < self.RATE_LIMIT_DELAY_S:
                time.sleep(self.RATE_LIMIT_DELAY_S - elapsed)
        self._last_search_time = time.time()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def search(
        self,
        lookfor: str,
        type: str = "AllFields",
        filters: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        facets: Optional[List[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single finc search and return the raw VuFind-JSON envelope.

        Args:
            lookfor: the search string. Wrap phrases in literal double quotes
                     (e.g. '"conrad hübler"'); URL-encoding is automatic — do
                     not pass %22 literally.
            type:    'AllFields' (default), 'Subject', 'Author', 'Title'.
            filters: optional mapping of facet key → value, e.g.
                     {"institution": "DE-105"}. Each entry is sent as one
                     `filter[]=key:"value"` query parameter. (Use a single-
                     record filter `{"id": "<record-id>"}` together with
                     ``facets`` to read one title's classifications.)
            limit:   override the configured default_limit (capped at MAX_LIMIT).
                     Pass ``0`` for a facet-only request (no records returned).
            facets:  optional list of facet field names to compute, e.g.
                     ["udk_raw_de105", "rvk_facet"]. Each is sent as one
                     `facet[]=name` parameter; the buckets come back under the
                     "facets" key of the result.
            extra_params: pass-through for additional API parameters.

        Returns:
            Dict with keys:
                status:      "OK" on success, "ERROR" on failure
                resultCount: int (or 0 on error)
                records:     list of normalized record dicts
                facets:      {facet_name: [{"value","count","translated"}]}
                             (empty dict when no facets requested/returned)
                error:       str | None (only present when status == "ERROR")
                http_status: int | None (HTTP status code, when available)
        """
        if not self.is_configured():
            return {
                "status": "ERROR",
                "resultCount": 0,
                "records": [],
                "facets": {},
                "error": "finc_base_url not configured",
            }
        if not lookfor or not str(lookfor).strip():
            return {
                "status": "ERROR",
                "resultCount": 0,
                "records": [],
                "facets": {},
                "error": "lookfor must be a non-empty string",
            }

        url = f"{self.base_url}{self.SEARCH_PATH}"
        # limit=0 is a valid facet-only request; None means "use the default".
        if limit is None:
            effective_limit = self.default_limit
        else:
            effective_limit = max(0, min(int(limit), self.MAX_LIMIT))

        # VuFind expects `filter[]=key:"value"` and `facet[]=name`. requests'
        # params serializer preserves repeated keys when given a list, so we
        # hand it one entry per filter/facet and let it URL-encode. - Claude Generated
        params: List[tuple] = [
            ("lookfor", lookfor),
            ("type", type or "AllFields"),
            ("limit", effective_limit),
        ]
        if filters:
            for facet_key, facet_val in filters.items():
                if facet_key and facet_val is not None and str(facet_val) != "":
                    params.append(("filter[]", f'{facet_key}:"{facet_val}"'))
        if facets:
            for facet_name in facets:
                if facet_name:
                    params.append(("facet[]", facet_name))
        if extra_params:
            for k, v in extra_params.items():
                params.append((k, v))

        self._apply_rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            self._consecutive_failures += 1
            logger.warning(f"finc request failed: {exc}")
            return {
                "status": "ERROR",
                "resultCount": 0,
                "records": [],
                "facets": {},
                "error": f"network error: {exc}",
                "http_status": None,
            }

        if response.status_code != 200:
            self._consecutive_failures += 1
            logger.warning(
                f"finc returned HTTP {response.status_code} for {response.url}"
            )
            return {
                "status": "ERROR",
                "resultCount": 0,
                "records": [],
                "facets": {},
                "error": f"HTTP {response.status_code}",
                "http_status": response.status_code,
            }

        try:
            payload = response.json()
        except ValueError as exc:
            self._consecutive_failures += 1
            logger.warning(f"finc response not JSON: {exc}")
            return {
                "status": "ERROR",
                "resultCount": 0,
                "records": [],
                "facets": {},
                "error": f"invalid JSON: {exc}",
                "http_status": response.status_code,
            }

        # The proxy returns HTTP 200 even on query errors, signalling failure
        # only via the envelope's status field (e.g.
        # {"status":"ERROR","statusMessage":"Invalid search"}). Treating that
        # as success would silently swallow the error and report 0 results —
        # so check the envelope status before parsing records. - Claude Generated
        if payload.get("status") != "OK":
            self._consecutive_failures += 1
            message = payload.get("statusMessage") or (
                f"finc returned status {payload.get('status')!r}"
            )
            logger.warning(f"finc query error (HTTP 200): {message}")
            return {
                "status": "ERROR",
                "resultCount": 0,
                "records": [],
                "facets": {},
                "error": message,
                "http_status": response.status_code,
            }

        raw_records = payload.get("records", []) or []
        if not isinstance(raw_records, list):
            raw_records = []
        normalized = [self._normalize_record(r) for r in raw_records]
        result_count = int(payload.get("resultCount", len(normalized)) or len(normalized))
        self._consecutive_failures = 0
        return {
            "status": "OK",
            "resultCount": result_count,
            "records": normalized,
            "facets": self._normalize_facets(payload.get("facets")),
            "error": None,
            "http_status": response.status_code,
        }

    @staticmethod
    def _normalize_facets(raw_facets: Any) -> Dict[str, List[Dict[str, Any]]]:
        """Reduce the VuFind facet block to {name: [{value,count,translated}]}.

        Drops the per-bucket ``href`` (a UI link, useless to callers). Returns
        an empty dict when no facets were requested/returned. - Claude Generated
        """
        if not isinstance(raw_facets, dict):
            return {}

        def _as_str(v: Any) -> str:
            # Some facets (e.g. dewey-raw) return numeric values (530, not "530");
            # normalize to str so all consumers can treat values uniformly. - Claude Generated
            return "" if v is None else str(v)

        out: Dict[str, List[Dict[str, Any]]] = {}
        for name, buckets in raw_facets.items():
            if not isinstance(buckets, list):
                continue
            out[name] = [
                {
                    "value": _as_str(b.get("value")),
                    "count": int(b.get("count", 0) or 0),
                    "translated": _as_str(b.get("translated", b.get("value"))),
                }
                for b in buckets
                if isinstance(b, dict)
            ]
        return out

    @staticmethod
    def normalize_dk_value(value: str) -> str:
        """Normalize a finc ``udk_raw_de105`` facet value to the pipeline's DK form.

        finc returns DK notations lowercased with a ``dk `` prefix, e.g.
        ``"dk 530.145"``; the rest of ALIMA uses ``"DK 530.145"`` (see
        ``biblio_client.py`` ``f"DK {dk}"``). Leaves non-DK values untouched. - Claude Generated
        """
        if not isinstance(value, str):
            return ""
        v = value.strip()
        if v[:3].lower() == "dk ":
            return "DK " + v[3:].strip()
        return v

    # ------------------------------------------------------------------
    # Record normalization
    # ------------------------------------------------------------------

    def _normalize_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map a raw VuFind record to ALIMA's stable record shape.

        Keeps the full VuFind payload under "raw" so callers can still see
        the original fields if they need them, but the top-level keys
        follow the operator-approved Biblio-style template.
        """
        record_id = raw.get("id") or ""
        web_url = ""
        if self.web_record_url and record_id:
            web_url = f"{self.web_record_url}{record_id}"
        elif record_id:
            # Fall back to the first URL entry that looks like a record link
            for u in raw.get("urls", []) or []:
                if isinstance(u, dict) and u.get("url"):
                    web_url = u["url"]
                    break
        return {
            "id": record_id,
            "title": raw.get("title") or "",
            "authors": raw.get("authors") or {},
            "subjects": raw.get("subjects") or [],
            "formats": raw.get("formats") or [],
            "languages": raw.get("languages") or [],
            "series": raw.get("series") or [],
            "urls": raw.get("urls") or [],
            "web_url": web_url,
            "raw": raw,
        }
