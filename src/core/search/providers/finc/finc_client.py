"""finc_client.py - VuFind-JSON client for finc / finc-solrproxy catalog APIs.

Used by TU Freiberg UB (and other German university libraries running a
finc/VuFind discovery instance) to query the local catalog through the
finc solrproxy. The proxy exposes a small JSON HTTP API at

    {base_url}/api/v1/search?lookfor=...&type=Subject&filter[]=key:value

which returns the same VuFind record shape ALIMA already understands from
BiblioClient (title, authors, subjects, formats, languages, series, urls).

Sits alongside BiblioClient (Libero SOAP) and MarcXmlClient (SRU). The
finc client is the **preferred** catalog source when configured — see
pipeline_utils.execute_notation_search for the DK-priority wiring.

Claude Generated (finc integration, June 2026).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"(1[5-9]\d{2}|20\d{2})")
_COVER_IMAGE_RE = re.compile(r"cov(?:er)?\.(?:jpg|jpeg|png|gif)(?:\?.*)?$", re.IGNORECASE)


def _first_year(values: Any) -> str:
    """Best-effort 4-digit year extraction from a publicationDates-style list.

    finc/VuFind dates come back as loosely punctuated strings (e.g. "2024.",
    ", 2024."), so pull the first plausible year rather than trusting exact
    formatting. Empty string if nothing parses — never fabricate. - Claude Generated
    """
    for v in values or []:
        if not v:
            continue
        m = _YEAR_RE.search(str(v))
        if m:
            return m.group(1)
    return ""


def _first_str(value: Any) -> str:
    """Some finc fields (e.g. edition) are a string for one record and an
    empty list for another (article-index entries) — normalize both. - Claude Generated"""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value else ""


def _join_list(value: Any, sep: str = "; ") -> str:
    if isinstance(value, list):
        return sep.join(str(v) for v in value if v)
    return str(value) if value else ""


def _pick_isbn(raw: Dict[str, Any]) -> str:
    clean = raw.get("cleanIsbn")
    if clean:
        return str(clean)
    isbns = raw.get("isbns")
    if isinstance(isbns, list) and isbns:
        return str(isbns[0])
    return ""


def _pick_resource_url(raw: Dict[str, Any]) -> str:
    """Full-text/e-resource link, distinct from the catalog web_url.

    Prefers the clean DOI resolver link when present; otherwise scans the
    VuFind ``urls`` list for the first non-cover-image entry (falling back to
    the first url of any kind if every entry looks like a cover image). - Claude Generated
    """
    clean_doi = raw.get("cleanDoi")
    if clean_doi:
        return f"https://doi.org/{clean_doi}"
    urls = [u.get("url") for u in (raw.get("urls") or []) if isinstance(u, dict) and u.get("url")]
    for u in urls:
        if not _COVER_IMAGE_RE.search(u):
            return u
    return urls[0] if urls else ""


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
    Each normalized record: {"id", "title", "authors": {...}, "subjects":
                      [[...]], "formats", "languages", "series": [{name,
                      number}], "urls": [{url, desc, indicators}], "web_url",
                      "resource_url", "year", "publisher", "edition", "isbn"}.
    year/publisher/edition/isbn require DEFAULT_FIELDS' extra field[] params
    (verified against the proxy's swagger schema — see discover_fields());
    they're best-effort and may be empty for records that genuinely lack
    that data (e.g. article-index hits).
    """

    SEARCH_PATH = "/api/v1/search"
    RECORD_PATH = "/api/v1/record"
    SCHEMA_PATH = "/api?swagger"
    DEFAULT_TIMEOUT = 30
    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    RATE_LIMIT_DELAY_S = 0.1  # Claude Generated - gentle throttle

    # Fields requested on every search. The proxy's ``field[]`` parameter is
    # NOT additive — passing any field[] switches it from "return the 8
    # built-in defaults" (authors, formats, id, languages, series, subjects,
    # title, urls) to "return only what's listed here", so the defaults must
    # be re-listed alongside the extras. Verified against the live TU
    # Freiberg endpoint's swagger spec (``{base_url}/api?swagger``) and a
    # real search: edition/publishers/publicationDates/isbns/cleanIsbn/
    # cleanDoi/institutions/placesOfPublication are all valid Record fields
    # that the bare default silently omits. - Claude Generated
    DEFAULT_FIELDS: Sequence[str] = (
        "id", "title", "authors", "formats", "languages", "series", "subjects",
        "urls", "edition", "publishers", "publicationDates", "isbns",
        "cleanIsbn", "cleanDoi", "institutions", "placesOfPublication",
    )

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
        fields: Optional[Sequence[str]] = None,
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
            fields:  override the record fields requested (default:
                     ``DEFAULT_FIELDS``). Pass an explicit list if you need a
                     different/narrower set than the class default.

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
        try:
            # Operator-configured endpoint: cheap scheme gate only (intranet OK,
            # file:// etc. rejected). - Claude Generated
            from src.utils.net_guard import require_http_url

            require_http_url(url, what="finc base URL")
        except ValueError as exc:
            return {
                "status": "ERROR",
                "resultCount": 0,
                "records": [],
                "facets": {},
                "error": str(exc),
            }
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
        for field_name in (fields if fields is not None else self.DEFAULT_FIELDS):
            if field_name:
                params.append(("field[]", field_name))
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

    def get_records(
        self,
        ids: Sequence[str],
        fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch one or more records directly by ID via ``{base_url}/api/v1/record``.

        Unlike ``search()``, this is an exact lookup — no relevance ranking,
        no free-text query. Useful to fetch full, enriched details for a
        specific finc/VuFind id (e.g. after matching a Libero RSN to its
        finc-side "0-"-prefixed id) without re-running a fuzzy title search.
        Same normalized record shape and error envelope as ``search()``
        (minus ``facets``, which this endpoint doesn't support). - Claude Generated

        Args:
            ids: one or more finc/VuFind record ids (e.g. "0-1878699474").
            fields: override the record fields requested (default:
                     ``DEFAULT_FIELDS``).

        Returns:
            Dict with keys: status, resultCount, records (normalized), error,
            http_status.
        """
        if not self.is_configured():
            return {
                "status": "ERROR", "resultCount": 0, "records": [],
                "error": "finc_base_url not configured",
            }
        id_list = [str(i) for i in (ids or []) if i]
        if not id_list:
            return {
                "status": "ERROR", "resultCount": 0, "records": [],
                "error": "ids must be a non-empty list",
            }

        url = f"{self.base_url}{self.RECORD_PATH}"
        try:
            from src.utils.net_guard import require_http_url

            require_http_url(url, what="finc base URL")
        except ValueError as exc:
            return {"status": "ERROR", "resultCount": 0, "records": [], "error": str(exc)}

        params: List[tuple] = (
            [("id", id_list[0])] if len(id_list) == 1
            else [("id[]", i) for i in id_list]
        )
        for field_name in (fields if fields is not None else self.DEFAULT_FIELDS):
            if field_name:
                params.append(("field[]", field_name))

        self._apply_rate_limit()
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            self._consecutive_failures += 1
            logger.warning(f"finc record request failed: {exc}")
            return {
                "status": "ERROR", "resultCount": 0, "records": [],
                "error": f"network error: {exc}", "http_status": None,
            }

        if response.status_code != 200:
            self._consecutive_failures += 1
            logger.warning(f"finc record returned HTTP {response.status_code} for {response.url}")
            return {
                "status": "ERROR", "resultCount": 0, "records": [],
                "error": f"HTTP {response.status_code}", "http_status": response.status_code,
            }

        try:
            payload = response.json()
        except ValueError as exc:
            self._consecutive_failures += 1
            logger.warning(f"finc record response not JSON: {exc}")
            return {
                "status": "ERROR", "resultCount": 0, "records": [],
                "error": f"invalid JSON: {exc}", "http_status": response.status_code,
            }

        if payload.get("status") != "OK":
            self._consecutive_failures += 1
            message = payload.get("statusMessage") or f"finc returned status {payload.get('status')!r}"
            logger.warning(f"finc record query error (HTTP 200): {message}")
            return {
                "status": "ERROR", "resultCount": 0, "records": [],
                "error": message, "http_status": response.status_code,
            }

        raw_records = payload.get("records", []) or []
        if not isinstance(raw_records, list):
            raw_records = []
        normalized = [self._normalize_record(r) for r in raw_records]
        self._consecutive_failures = 0
        return {
            "status": "OK",
            "resultCount": int(payload.get("resultCount", len(normalized)) or len(normalized)),
            "records": normalized,
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
        # web_url is the CATALOG RECORD page (e.g. .../Record/0-1846124905) — it
        # must never point at a publisher/full-text URL. The finc/VuFind record id
        # already carries the "0-" prefix, so web_record_url + id is the catalog
        # link. Left empty when no record base is configured; the MCP handler then
        # reconstructs it from catalog_web_record_url. - Claude Generated
        web_url = ""
        if self.web_record_url and record_id:
            web_url = f"{self.web_record_url}{record_id}"
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
            # resource_url is the book itself at the publisher / full-text
            # provider (DOI resolver preferred), offered IN ADDITION to the
            # catalog link. - Claude Generated
            "resource_url": _pick_resource_url(raw),
            # Structured bibliographic fields (edition/publishers/
            # publicationDates/isbns/cleanIsbn) require an explicit field[]
            # request — see DEFAULT_FIELDS. year/edition may be empty for
            # records where the underlying catalog entry lacks that data
            # (e.g. article-index hits) — never fabricated. - Claude Generated
            "year": _first_year(raw.get("publicationDates")),
            "publisher": _join_list(raw.get("publishers")),
            "edition": _first_str(raw.get("edition")),
            "isbn": _pick_isbn(raw),
            "raw": raw,
        }

    def discover_fields(self) -> Dict[str, str]:
        """Introspect this instance's swagger spec for the Record schema's
        field names/descriptions.

        This provider is a copyable blueprint for other institutions' finc/
        VuFind instances (see the plugin's README), and different instances
        can expose a different local Solr schema. ``DEFAULT_FIELDS`` above is
        verified against the TU Freiberg endpoint only — use this method to
        re-validate the field list when pointing the client at a different
        instance, e.g.::

            fields = FincClient(base_url=...).discover_fields()
            missing = set(FincClient.DEFAULT_FIELDS) - fields.keys()

        Best-effort: returns ``{}`` on any failure (network, non-swagger
        response, unexpected shape) rather than raising — this is a
        diagnostic helper, not part of the request path. - Claude Generated
        """
        if not self.is_configured():
            return {}
        url = f"{self.base_url}{self.SCHEMA_PATH}"
        try:
            from src.utils.net_guard import require_http_url

            require_http_url(url, what="finc base URL")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            spec = response.json()
            props = spec["components"]["schemas"]["Record"]["properties"]
            return {name: (info.get("description") or "") for name, info in props.items()}
        except Exception as exc:
            logger.debug(f"finc discover_fields failed: {exc}")
            return {}
