"""finc_catalog_client.py - finc-backed DK/RVK extractor (BiblioClient-compatible).

Drop-in extractor for ``PipelineStepExecutor.execute_notation_search``: implements
``extract_dk_classifications_for_keywords`` with the SAME keyword-centric return
shape as ``BiblioClient`` / ``MarcXmlClient``, but sources titles + per-title
DK/RVK from a finc / VuFind-JSON instance instead of Libero SOAP / web-scrape.

Two-step model (operator decision, June 2026 — "Titelliste per finc, dann jeden
Titel mit udk_raw analysieren"):

  1. finc Subject search per keyword  → title records (id, title).
  2. per title: isolate by id (``lookfor=id:"<id>"``) + facet ``udk_raw_<isil>``
     (``institution.finc_dk_facet``, ``udk_raw_de105`` for the UB Freiberg) /
     ``rvk_facet`` → that title's exact DK/RVK notations. finc exposes no
     per-record classification field, so the single-record facet is the only
     way to read it (verified against the live endpoint).

The per-title result is built into the same ``{rsn, title, classifications}``
title-list shape Libero produces, then funnelled through
``UnifiedKnowledgeManager.extract_classifications_from_titles`` — so the rest of
``execute_notation_search`` (RVK validation, flattening, statistics, GUI) is unchanged.

Claude Generated (finc DK unification, June 2026).
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from .finc_client import FincClient

logger = logging.getLogger(__name__)


class FincCatalogClient:
    """finc DK/RVK extractor exposing the BiblioClient extractor interface."""

    # DK facet of the library: ``institution.finc_dk_facet()`` (``udk_raw_de105``
    # for DE-105), read per call; ``None`` without ISIL. - Claude Generated
    DDC_FACET = "dewey-raw"   # numeric per-title DDC (e.g. "530"), DDC analog of udk_raw
    RVK_FACET = "rvk_facet"
    # rvk_facet buckets that are not real classifications. - Claude Generated
    _RVK_SKIP = {"", "no subject assigned", "nicht zugeordnet"}

    def __init__(
        self,
        base_url: str,
        web_record_url: str = "",
        institution_filter: str = "",
        timeout: int = 30,
        max_workers: int = 8,
        max_titles_per_keyword: int = 50,
        use_cache: bool = True,
        knowledge_manager: Any = None,
        logger_: Optional[logging.Logger] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        cache_only: bool = False,
        source_id: Optional[str] = None,
    ):
        """Initialize the finc catalog extractor.

        Args:
            base_url: finc solrproxy base URL (no trailing slash).
            web_record_url: optional record base URL (passed to FincClient).
            institution_filter: optional holding-library facet value (the
                ISIL of the library); applied as ``filter[]=institution:"<value>"`` on the
                keyword Subject search (NOT on the per-title id lookup).
            timeout: HTTP timeout (seconds) for each finc call.
            max_workers: thread-pool size for the parallel per-title DK fetch.
            max_titles_per_keyword: cap on titles analysed per keyword.
            use_cache: reuse the shared catalog DK cache
                (``UnifiedKnowledgeManager.get/store_catalog_dk_cache``).
            knowledge_manager: optional pre-built UnifiedKnowledgeManager
                (injected in tests); lazily created otherwise.
            logger_: optional logger.
            stream_callback: optional single-arg progress sink.
            cache_only: no finc request at all; titles come from the shared
                DK cache at any age, keywords without an entry are listed in
                ``last_cache_misses`` (reset per call). - Claude Generated
            source_id: plugin instance id for ``source_health``; while the
                source is down a keyword without a cache hit is skipped without
                a request and listed in ``last_source_down`` (reset per call),
                nothing is cached for it. - Claude Generated
        """
        self.cache_only = bool(cache_only)
        self.last_cache_misses: List[str] = []
        self.source_id = source_id
        self.last_source_down: List[str] = []
        # Keywords whose titles could not be read for DK because no ISIL is
        # set (no DK facet); reset per call. - Claude Generated
        self.last_not_configured: List[str] = []
        # Origin of the keyword being processed, for source_usage. - Claude Generated
        self._usage_origin: Optional[tuple] = None
        self.base_url = base_url
        self.web_record_url = web_record_url or ""
        self.institution_filter = institution_filter or ""
        self.timeout = timeout
        self.max_workers = max(1, int(max_workers))
        self.max_titles_per_keyword = max(1, int(max_titles_per_keyword))
        self.use_cache = use_cache
        self.logger = logger_ or logger
        self.stream_callback = stream_callback
        self._km = knowledge_manager
        self._local = threading.local()
        # Primary client for the keyword-level Subject searches (main thread).
        self._client = self._new_client()

    # ------------------------------------------------------------------
    # Client / KM helpers
    # ------------------------------------------------------------------

    def _new_client(self) -> FincClient:
        return FincClient(
            base_url=self.base_url,
            web_record_url=self.web_record_url,
            timeout=self.timeout,
            cache_only=self.cache_only,
            source_id=self.source_id,
        )

    def _thread_client(self) -> FincClient:
        """One FincClient per worker thread.

        Avoids sharing a single client's rate-limiter / session state across
        threads (the rate-limiter uses non-atomic shared timestamps). - Claude Generated
        """
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._new_client()
            self._local.client = client
        return client

    @property
    def dk_facet(self) -> Optional[str]:
        """The library's DK facet (``institution.finc_dk_facet``) - Claude Generated"""
        from src.utils.institution import finc_dk_facet

        return finc_dk_facet()

    def _km_or_init(self):
        if self._km is None:
            from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
            self._km = UnifiedKnowledgeManager()
        return self._km

    # ------------------------------------------------------------------
    # Public extractor interface (BiblioClient-compatible)
    # ------------------------------------------------------------------

    def extract_dk_classifications_for_keywords(
        self,
        keywords: List[str],
        max_results: int = 50,
        force_update: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return keyword-centric DK/RVK results, mirroring BiblioClient.

        Each element: ``{"keyword", "source", "search_time_ms",
        "classifications": [...]}`` where ``classifications`` is the grouped
        output of ``extract_classifications_from_titles``. Keywords with no
        usable titles/classifications are omitted (same as BiblioClient).
        """
        results: List[Dict[str, Any]] = []
        self.last_cache_misses = []
        self.last_source_down = []
        self.last_not_configured = []
        from src.core.search import source_usage as _usage

        for kw in keywords or []:
            clean = (kw or "").split("(")[0].strip()
            if not clean:
                continue
            t0 = time.time()
            if self.dk_facet is None and not self.cache_only:
                # Without an ISIL there is no DK facet to read: reported as
                # "institution not set", nothing requested, nothing cached.
                # - Claude Generated
                self.last_not_configured.append(clean)
                _usage.record(self.source_id, _usage.NOT_CONFIGURED)
                continue
            self._usage_origin = None
            title_list, source = self._titles_for_keyword(clean, max_results, force_update)
            origin, failed = self._usage_origin or (_usage.NETWORK, False)
            if not title_list:
                _usage.record(self.source_id, origin, error=failed)
                continue
            classifications = self._km_or_init().extract_classifications_from_titles(
                title_list, matched_keywords=[clean]
            )
            _usage.record(self.source_id, origin, hit=bool(classifications), error=failed)
            if classifications:
                results.append({
                    "keyword": kw,
                    "source": source,
                    "search_time_ms": round((time.time() - t0) * 1000, 1),
                    "classifications": classifications,
                })
        return results

    # ------------------------------------------------------------------
    # Step 1: title list per keyword (with cache)
    # ------------------------------------------------------------------

    def _titles_for_keyword(
        self, clean_kw: str, max_results: int, force_update: bool
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Resolve the per-title classification list for one keyword.

        Returns ``(title_list, source)`` where source ∈ {"cache", "finc"}.
        ``title_list`` items: ``{"rsn", "title", "classifications": ["DK ...",
        "RVK ..."]}``.
        """
        from src.core.search import source_usage as _usage

        if self.cache_only:
            return self._titles_cache_only(clean_kw)
        if self.use_cache and not force_update:
            cached = self._km_or_init().get_catalog_dk_cache(clean_kw)
            if cached:
                titles, status, _ = cached
                if status == "success" and titles:
                    self._usage_origin = (_usage.CACHE, False)
                    return titles, "cache"

        from src.core.search import source_health

        if self.source_id and source_health.is_down(self.source_id):
            # Skipped for this run: no request, nothing cached. - Claude Generated
            self.last_source_down.append(clean_kw)
            self._usage_origin = (_usage.SKIPPED, False)
            return [], "finc"
        self._usage_origin = (_usage.NETWORK, False)

        # finc Subject search -> title records (id, title)
        filters = {"institution": self.institution_filter} if self.institution_filter else None
        limit = min(int(max_results or self.max_titles_per_keyword), self.max_titles_per_keyword)
        payload = self._client.search(
            lookfor=clean_kw, type="Subject", filters=filters, limit=limit
        )
        if payload.get("status") != "OK":
            err = payload.get("error") or "unknown finc error"
            self._usage_origin = (
                (_usage.SKIPPED, False) if source_health.is_source_down(err) else (_usage.NETWORK, True))
            if self.logger:
                self.logger.warning(f"finc DK search for '{clean_kw}' failed: {err}")
            if self.stream_callback:
                self.stream_callback(f"  ❌ finc '{clean_kw}': {err}\n")
            if self.use_cache and not source_health.is_source_down(err):
                self._km_or_init().store_catalog_dk_cache(
                    clean_kw, [], status="error", error_message=str(err), ttl_minutes=60
                )
            return [], "finc"

        records = [r for r in (payload.get("records") or []) if r.get("id")]
        if not records:
            if self.use_cache:
                self._km_or_init().store_catalog_dk_cache(
                    clean_kw, [], status="no_results", ttl_minutes=30
                )
            return [], "finc"

        title_list = self._classify_titles(records)
        if self.use_cache:
            if title_list:
                self._km_or_init().store_catalog_dk_cache(clean_kw, title_list, status="success")
            else:
                self._km_or_init().store_catalog_dk_cache(
                    clean_kw, [], status="no_results", ttl_minutes=30
                )
        return title_list, "finc"

    def _titles_cache_only(self, clean_kw: str) -> Tuple[List[Dict[str, Any]], str]:
        """Cache-only title list: the shared DK cache at any age, no finc request.

        ``success`` is a hit, ``no_results`` a known empty answer; a missing row
        or a cached failure is a miss (``last_cache_misses``). Nothing is
        written. - Claude Generated
        """
        cached = (
            self._km_or_init().get_catalog_dk_cache(clean_kw, ignore_ttl=True)
            if self.use_cache else None
        )
        from src.core.search import source_usage as _usage

        status = cached[1] if cached else None
        if status == "success" and cached[0]:
            self._usage_origin = (_usage.CACHE, False)
            return cached[0], "cache"
        if status != "no_results":
            self.last_cache_misses.append(clean_kw)
            self._usage_origin = (_usage.CACHE_MISS, False)
        else:
            self._usage_origin = (_usage.CACHE, False)
        return [], "cache"

    # ------------------------------------------------------------------
    # Step 2: per-title DK/RVK (parallel)
    # ------------------------------------------------------------------

    def _classify_titles(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch each record's DK/RVK in parallel; build the title list."""
        title_list: List[Dict[str, Any]] = []

        def _one(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            rid = rec.get("id")
            title = rec.get("title") or ""
            classifications = self._title_classifications(rid)
            if title and classifications:
                return {"rsn": rid, "title": title, "classifications": classifications}
            return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_one, rec) for rec in records]
            for fut in as_completed(futures):
                try:
                    item = fut.result()
                except Exception as exc:  # noqa: BLE001 - per-title failure is non-fatal
                    if self.logger:
                        self.logger.debug(f"finc per-title fetch failed: {exc}")
                    item = None
                if item:
                    title_list.append(item)
        return title_list

    def _title_classifications(self, record_id: str) -> List[str]:
        """Return one record's classification strings (``["DK 530.145", "RVK UC 100"]``)."""
        if not record_id:
            return []
        client = self._thread_client()
        dk_facet = self.dk_facet
        if dk_facet is None:
            return []
        payload = client.search(
            lookfor=f'id:"{record_id}"',
            type="AllFields",
            facets=[dk_facet, self.DDC_FACET, self.RVK_FACET],
            limit=1,
        )
        if payload.get("status") != "OK":
            return []
        facets = payload.get("facets", {}) or {}
        out: List[str] = []
        for bucket in facets.get(dk_facet, []):
            value = (bucket.get("value") or "").strip()
            # udk_raw_<isil> holds DK notations prefixed with "dk " (e.g.
            # "dk 530.145"); the field also contains stray non-DK artifacts
            # ("fg", "fgaut") that must NOT be emitted as classifications. - Claude Generated
            if value.lower().startswith("dk "):
                out.append(FincClient.normalize_dk_value(value))  # -> "DK 530.145"
        for bucket in facets.get(self.DDC_FACET, []):
            # dewey-raw values come back NUMERIC (530, not "530") — coerce to str
            # before parsing. Holds bare DDC notations ("530", "530.1"); only emit
            # digit-led values to skip any non-DDC artifacts. - Claude Generated
            value = str(bucket.get("value") or "").strip()
            if value and value[0].isdigit():
                out.append(f"DDC {value}")
        for bucket in facets.get(self.RVK_FACET, []):
            value = (bucket.get("value") or "").strip()
            if value and value.lower() not in self._RVK_SKIP:
                # RVK notations are conventionally upper-case ("uc 100" -> "UC 100").
                out.append(f"RVK {value.upper()}")
        return out
