import requests
import xml.etree.ElementTree as ET
import csv
import argparse
from typing import List, Dict, Any, Optional, Tuple
import time
import re
import sys
import logging
import base64

# Import default configuration values - Claude Generated
try:
    from ..pipeline_defaults import DEFAULT_DK_MAX_RESULTS
except ImportError:
    # Fallback if import fails (standalone usage)
    DEFAULT_DK_MAX_RESULTS = 20

# Konfiguriere Logging mit Console + Debug-Datei - Claude Generated
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ._biblio_parsing import BiblioParsingMixin
from ._biblio_transport import BiblioTransportMixin
from ._biblio_soap import BiblioRequestMixin

logger = logging.getLogger("biblio_extractor")
logger.setLevel(logging.DEBUG)  # Logger captures all levels
logger.propagate = False  # Prevent duplicate output via root logger


def _configure_logger() -> None:
    """Configure console/file logging once without failing on restricted filesystems."""
    if logger.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s"
    )

    try:
        debug_log_dir = Path.home() / ".config/alima/logs"
        debug_log_dir.mkdir(parents=True, exist_ok=True)

        debug_file_path = debug_log_dir / "biblio_debug.log"
        file_handler = RotatingFileHandler(
            debug_file_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        error_file_path = debug_log_dir / "biblio_errors.log"
        error_handler = RotatingFileHandler(
            error_file_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(file_format)
        logger.addHandler(error_handler)
    except OSError as exc:
        logger.warning(f"BiblioClient file logging disabled: {exc}")


_configure_logger()


class BiblioClient(BiblioParsingMixin, BiblioTransportMixin, BiblioRequestMixin):
    """
    A tool to extract keywords and decimal classifications from a library catalog.
    """
    # Configuration flag to disable SQL database caching for testing - Claude Generated

    SEARCH_URL = ""
    DETAILS_URL = ""
    # Web catalog URLs for fallback - Claude Generated
    WEB_SEARCH_URL = ""
    WEB_RECORD_BASE_URL = ""

    # MAB-Tags für Schlagwörter
    MAB_SUBJECT_TAGS = ["0902", "0907", "0912", "0917", "0922", "0927"]

    def __init__(self, token: str = "", debug: bool = False, save_xml_path: str = "",
                 enable_web_fallback: bool = True, timeout: int = 10,
                 rate_limit_delay_ms: int = 1000, use_json_cache: bool = True,
                 soap_search_url: str = "", soap_details_url: str = "",
                 web_search_url: str = "", web_record_url: str = ""):
        """
        Initialize the extractor with the given token.

        Args:
            token: The authentication token for the library API
            debug: Enable detailed debug output
            save_xml_path: Directory to save raw XML responses for debugging (empty string = disabled)
            enable_web_fallback: Enable web scraping fallback when SOAP fails (Claude Generated)
            timeout: Request timeout in seconds (default: 10, reduced from 30 for fast failover when server down) - Claude Generated (2026-01-13)
            rate_limit_delay_ms: Milliseconds to wait between searches (default: 1000ms) - Claude Generated
            use_json_cache: Use JSON file cache for RSN→details lookups (True=fast via JSON, False=SOAP only) - Claude Generated
            soap_search_url: SOAP search endpoint URL; falls back to class default if empty - Claude Generated
            soap_details_url: SOAP details endpoint URL; falls back to class default if empty - Claude Generated
            web_search_url: Web frontend search URL for fallback scraping; disables web fallback if empty - Claude Generated
            web_record_url: Web frontend record base URL for fallback scraping - Claude Generated

        Note:
            - use_json_cache: Controls RSN detail lookups (step 2 of DK search)
            - disable_sql_cache: Can be set via set_disable_sql_cache() to disable keyword→RSN DB caching (step 1)
        """
        self.token = token if token else ""  # Use provided token from config
        # Instance-level URL overrides (fall back to class-level defaults) - Claude Generated
        self.SEARCH_URL = soap_search_url or BiblioClient.SEARCH_URL
        self.DETAILS_URL = soap_details_url or BiblioClient.DETAILS_URL
        self.WEB_SEARCH_URL = web_search_url or BiblioClient.WEB_SEARCH_URL
        self.WEB_RECORD_BASE_URL = web_record_url or BiblioClient.WEB_RECORD_BASE_URL
        self.debug = debug
        self.save_xml_path = save_xml_path
        # Web fallback only active when a web_search_url is configured - Claude Generated
        self.enable_web_fallback = enable_web_fallback and bool(self.WEB_SEARCH_URL)
        self.use_json_cache = use_json_cache  # Claude Generated - JSON cache toggle for testing
        self.disable_sql_cache = False  # Claude Generated - SQL DB cache toggle (default: use cache)
        self._using_web_mode = False  # Claude Generated - Track if we switched to web pipeline
        self.timeout = timeout  # Claude Generated - Configurable timeout

        # Statistics tracking - Claude Generated
        self._json_cache_hits = 0
        self._soap_calls = 0

        self.session = requests.Session()
        self.headers = {"Content-Type": "text/xml;charset=UTF-8", "SOAPAction": ""}

        # RATE LIMITING STATE - Claude Generated
        self.rate_limit_delay_ms = rate_limit_delay_ms
        self.last_search_time = None
        self.session_request_count = 0
        self.session_max_requests = 50  # Reset session after N requests to prevent staleness

        # CIRCUIT BREAKER STATE - Claude Generated
        self.consecutive_failures = 0
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_open = False
        self.circuit_breaker_reset_time = None

    def process_search_results(
        self, results: List[Dict[str, Any]], max_items: int = 100, delay: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Process search results by getting details for each item.

        Args:
            results: List of search result items
            max_items: Maximum number of items to process
            delay: Delay between requests in seconds (increased to 1.5s to prevent timeouts)

        Returns:
            List of processed items with details
        """
        # Log if web mode is active - Claude Generated
        if self._using_web_mode:
            logger.info(f"Processing {len(results)} search results in WEB MODE (no SOAP calls)")

        processed_items = []

        # DIAGNOSTIC: Initialize statistics tracking - Claude Generated
        stats = {
            'attempted': min(len(results), max_items),
            'no_rsn': 0,
            'empty_response': 0,
            'exception_timeout': 0,
            'exception_connection': 0,
            'exception_other': 0,
            'succeeded': 0,
            'with_classifications': 0,
            'without_classifications': 0
        }

        # Use dynamic delay based on JSON vs SOAP lookup ratio - Claude Generated
        effective_delay = self._get_appropriate_delay(delay)

        logger.debug(f"Processing {stats['attempted']} items with {effective_delay:.2f}s delay...")

        for i, item in enumerate(results[:max_items]):
            if i > 0:
                time.sleep(effective_delay)  # Use dynamic delay based on lookup pattern - Claude Generated

            title = item.get("title", "Unknown")
            logger.debug(
                f"Processing item {i+1}/{min(len(results), max_items)}: {title}"
            )

            rsn = item.get("rsn")
            if not rsn:
                logger.warning(f"No RSN found for item: {title}")
                stats['no_rsn'] += 1  # DIAGNOSTIC: Track no-RSN failures
                continue

            try:
                details = self.get_title_details(rsn)
                if not details:
                    logger.warning(f"⚠️ Could not get details for RSN {rsn}: empty response (possible timeout or server error)")
                    stats['empty_response'] += 1  # DIAGNOSTIC: Track empty responses
                    continue
            except Exception as e:
                # DIAGNOSTIC: Categorize exception types - Claude Generated
                if isinstance(e, requests.exceptions.Timeout):
                    stats['exception_timeout'] += 1
                    logger.error(f"⏱️ TIMEOUT getting details for RSN {rsn}: {str(e)}")
                elif isinstance(e, requests.exceptions.ConnectionError):
                    stats['exception_connection'] += 1
                    logger.error(f"🔌 CONNECTION ERROR getting details for RSN {rsn}: {str(e)}")
                else:
                    stats['exception_other'] += 1
                    logger.error(f"❌ ERROR getting details for RSN {rsn}: {type(e).__name__}: {str(e)}")
                continue

            # MERGE DATA: Start with search result data, overlay with server details - Claude Generated
            # FIX: Preserve search result data (which contains title) when server details are empty
            merged_details = dict(item)  # Copy search result data as base

            # Overlay non-empty fields from server details (details take priority for populated fields)
            for key, value in details.items():
                if key == "rsn":
                    # Always use the RSN from details
                    merged_details[key] = value
                elif isinstance(value, list):
                    if value:  # Only use non-empty lists from details
                        merged_details[key] = value
                elif isinstance(value, str):
                    if value and value.strip():  # Only use non-empty strings from details
                        merged_details[key] = value
                elif isinstance(value, dict):
                    if value:  # Only use non-empty dicts from details
                        merged_details[key] = value
                # Skip empty values - keep search result data as fallback

            # Log merged result with field counts for diagnostic - Claude Generated (DEBUG level)
            title_merged = merged_details.get("title", "")
            classifications_count = len(merged_details.get("classifications", []))
            subjects_count = len(merged_details.get("subjects", []))
            authors_count = len(merged_details.get("author", []))
            logger.debug(f"Merged details for RSN {rsn}: title='{title_merged}', classifications={classifications_count}, subjects={subjects_count}, authors={authors_count}")

            # Add decimal classifications
            merged_details["decimal_classifications"] = self.extract_decimal_classifications(
                merged_details.get("classifications", [])
            )

            # Add RVK classifications - Claude Generated
            merged_details["rvk_classifications"] = self.extract_rvk_classifications(
                merged_details.get("classifications", [])
            )

            # DIAGNOSTIC: Track success and classification presence - Claude Generated
            stats['succeeded'] += 1
            has_classifications = bool(
                merged_details.get('decimal_classifications') or
                merged_details.get('rvk_classifications')
            )
            if has_classifications:
                stats['with_classifications'] += 1
            else:
                stats['without_classifications'] += 1

            processed_items.append(merged_details)

        # DIAGNOSTIC: Statistics summary - Claude Generated
        failed = stats['attempted'] - stats['succeeded']
        logger.debug(f"Details stats: {stats['succeeded']}/{stats['attempted']} succeeded, "
                     f"{stats['with_classifications']} with classifications, {stats['without_classifications']} without")
        if failed > 0:
            logger.warning(f"{failed} Details calls failed (timeout={stats['exception_timeout']}, "
                          f"connection={stats['exception_connection']}, other={stats['exception_other']})")

        return processed_items

    def search_subjects(
        self,
        search_terms: List[str],
        max_results: int = DEFAULT_DK_MAX_RESULTS,
        search_type: str = "kw",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Claude Generated - Search catalog for subjects and return in suggester format.

        Args:
            search_terms: List of search terms to look for
            max_results: Maximum results to process per term
            search_type: "kw" (default, Libero 'ku' anyword), "title" (Libero 'k'),
                "freetext" (Libero 'ku')

        Returns:
            Dictionary with structure:
            {
                search_term: {
                    subject: {
                        "count": int,
                        "gnd_ids": set(),
                        "classifications": {}
                    }
                }
            }
        """
        libero_map = {"kw": "ku", "title": "k", "freetext": "ku"}
        libero_use = libero_map.get(search_type, "ku")

        # WP2 raw-first: verbatim parsed records per term for the raw cache. - Claude Generated
        self.last_raw = {}
        results = {}

        for search_term in search_terms:
            logger.debug(f"Searching catalog subjects for: {search_term} (use={libero_use})")

            # Search catalog for this term
            search_results = self.search(search_term, search_type=libero_use)

            # Web fallback for search_subjects if SOAP fails - Claude Generated
            if not search_results and self.enable_web_fallback:
                logger.debug(f"SOAP search failed for subjects '{search_term}', trying web fallback")
                search_results = self._search_web(search_term)

            #logger.info(f"Found {(search_results)} results for '{search_term}'")
            if not search_results:
                results[search_term] = {}
                continue
                
            # Limit search results to prevent excessive processing - Claude Generated
            if len(search_results) > max_results * 2:
                logger.debug(f"Limiting search results for '{search_term}': {len(search_results)} -> {max_results * 2}")
                search_results = search_results[:max_results * 2]
                
            # Process results to extract subjects
            processed_items = self.process_search_results(search_results, max_items=max_results)
            self.last_raw[search_term] = processed_items  # WP2 raw-first - Claude Generated
            #logger.info(f"Processed {(processed_items)} items for '{search_term}'")
            # CLAUDE TODO -> an diesem Punkte haben wir also die MABs
            # Convert to suggester format (shared reduction — see below).
            term_subjects = self._reduce_records_to_subjects(processed_items)
            results[search_term] = term_subjects
            logger.debug(f"Found {len(term_subjects)} subjects for '{search_term}'")

        return results

    def _reduce_records_to_subjects(
        self, processed_items: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Reduce parsed catalog records to the canonical ``{subject: {count, gnd_ids, classifications}}``.

        Extracted from :meth:`search_subjects` so the same reduction backs both
        the live path and the WP2 raw-cache transform-on-read. Byte-identical:
        subjects are the union of ``subjects`` + ``mab_subjects``, ``count`` is
        the occurrence tally, ``gnd_ids`` starts empty (filled
        later by SWB validation), and the result is capped at the top 50 by
        count. - Claude Generated
        """
        term_subjects: Dict[str, Dict[str, Any]] = {}
        for item in processed_items:
            subjects = item.get("subjects", []) + item.get("mab_subjects", [])
            ddc_set = set()
            dk_set = set()
            for subject in subjects:
                subject = subject.strip()
                if not subject:
                    continue
                if subject not in term_subjects:
                    term_subjects[subject] = {
                        "count": 1,
                        "gnd_ids": set(),  # Will be filled by SWB validation later
                        "classifications": {
                            system: codes.copy()
                            for system, codes in (("ddc", ddc_set), ("dk", dk_set))
                            if codes
                        },
                    }
                else:
                    term_subjects[subject]["count"] += 1
                    cls = term_subjects[subject]["classifications"]
                    if ddc_set:
                        cls.setdefault("ddc", set()).update(ddc_set)
                    if dk_set:
                        cls.setdefault("dk", set()).update(dk_set)

        # Limit subjects per term to prevent excessive results - Claude Generated
        if len(term_subjects) > 50:
            sorted_subjects = sorted(
                term_subjects.items(), key=lambda x: x[1]["count"], reverse=True
            )
            term_subjects = dict(sorted_subjects[:50])
        return term_subjects

    def search_titles(
        self,
        search_terms: List[str],
        max_results: int = 25,
        search_type: str = "title",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search catalog and return bibliographic records per query.

        Unlike search_subjects (which aggregates Schlagwörter from the hits),
        this method returns the actual book records — intended for the
        title_list_search workflow where the LLM extracts titles from
        free-form text and the catalog is queried in title mode.

        Args:
            search_terms: List of query strings (typically book titles).
            max_results: Maximum records to process per query.
            search_type: "title" (Libero 'k', default), "kw" ('ku' anyword),
                "freetext" ('ku'), or any raw Libero use-code (e.g. 'kb'
                author, 'ke' combined author, 'sk' subjects, 'i' ISBN).

        Returns:
            Mapping ``{search_term: [record, ...]}``. Each record contains
            rsn, web_url, title, authors, isbn, publication, year,
            ``classifications`` (canonical ``{system: [{code, origin}]}``,
            WP-D2), subjects, mab_subjects.
        """
        libero_map = {"kw": "ku", "title": "k", "freetext": "ku"}
        libero_use = libero_map.get(search_type, search_type)

        results: Dict[str, List[Dict[str, Any]]] = {}

        for term in search_terms:
            logger.debug(f"search_titles: '{term}' (use={libero_use})")
            hits = self.search(term, search_type=libero_use)

            if not hits and self.enable_web_fallback:
                logger.debug(f"SOAP empty for '{term}', trying web fallback")
                hits = self._search_web(term)

            if not hits:
                results[term] = []
                continue

            if len(hits) > max_results * 2:
                hits = hits[: max_results * 2]

            processed = self.process_search_results(hits, max_items=max_results)

            from ..classification_systems import build_classifications

            records = []
            for item in processed:
                rsn = item.get("rsn")
                title = item.get("title", "").strip()
                if not title:
                    continue
                # WP-D2: title records carry the ONE canonical classification
                # dict instead of parallel dk_codes/rvk_codes/ddc_codes lists
                classifications = build_classifications([
                    ("DK", item.get("decimal_classifications")),
                    ("RVK", item.get("rvk_classifications")),
                    ("DDC", item.get("ddc_codes")),
                ])
                # Claude Generated - catalog web link for LLM / operator output.
                # The TU-Freiberg web OPAC expects RSNs prefixed with "0-"
                # (e.g. "0-364641185"); the SOAP API returns the bare numeric
                # form, so we reformat here. `sid=` is intentionally omitted
                # because it is session-specific and the LLM/operator side
                # cannot mint one — the OPAC redirects gracefully without it.
                # Empty when WEB_RECORD_BASE_URL is unconfigured or RSN is
                # missing/non-numeric, so downstream consumers see "" rather
                # than a fabricated URL.
                web_url = ""
                if self.WEB_RECORD_BASE_URL and rsn:
                    rsn_str = str(rsn).strip()
                    if rsn_str.isdigit():
                        web_url = f"{self.WEB_RECORD_BASE_URL}0-{rsn_str}"
                    else:
                        # Non-numeric RSN (e.g. web-fallback ID) — use as-is
                        web_url = f"{self.WEB_RECORD_BASE_URL}{rsn_str}"
                records.append({
                    "rsn": rsn,
                    "web_url": web_url,
                    "title": title,
                    "authors": item.get("author", []) or item.get("authors", []),
                    "isbn": item.get("isbn", ""),
                    "publication": item.get("publication", ""),
                    "year": item.get("year", "") or item.get("publication_year", ""),
                    "classifications": classifications,
                    "subjects": list(item.get("subjects") or []),
                    "mab_subjects": list(item.get("mab_subjects") or []),
                })

            results[term] = records
            logger.debug(f"search_titles '{term}': {len(records)} records")

        return results

    def extract_dk_classifications_for_keywords(self, keywords: List[str], max_results: int = 50, force_update: bool = False) -> List[Dict[str, Any]]:
        """
        RESTRUCTURED - Extract DK/RVK classifications for keywords using title-centric caching.

        Flow:
        1. File-based lookup → get title list from precomputed JSON files
        2. Database cache lookup → get title list
        3. On cache miss → live search → extract titles → cache titles
        4. Extract classifications from title list (on-demand)

        Returns title-based structures (not classification-based)
        """
        from ...core.unified_knowledge_manager import UnifiedKnowledgeManager
        # Import ClassificationLookupService for file-based optimization
        from ...utils.classification_lookup_service import get_classification_lookup_service

        dk_cache = UnifiedKnowledgeManager()
        all_titles = []  # Accumulated titles from cache and live search

        # Normalize keywords for cache lookup
        normalized_keywords = {}
        for kw in keywords:
            clean = kw.split('(')[0].strip()
            if clean not in normalized_keywords:
                normalized_keywords[clean] = kw

        # Track which keywords came from cache vs live search
        cached_keywords = set()
        file_cached_keywords = set()  # Track keywords found in JSON files

        # Try file-based lookup first for each keyword - Claude Generated Optimization
        file_cached_count = 0
        classification_lookup = get_classification_lookup_service()

        for clean_kw in normalized_keywords.keys():
            # Try to get titles from file-based lookup
            file_titles = classification_lookup.get_titles_for_classification(clean_kw)
            if file_titles:
                all_titles.extend(file_titles)
                file_cached_keywords.add(clean_kw)
                file_cached_count += 1
                logger.debug(f"FILE CACHE HIT for '{clean_kw}': {len(file_titles)} titles from JSON files")
                # Mark as cached to avoid live search
                cached_keywords.add(clean_kw)

        # Log file-based cache statistics
        if file_cached_count > 0:
            logger.debug(f"File-based cache hit: {file_cached_count}/{len(normalized_keywords)} keywords")

        # Try cache for each keyword with Rate Limiting & Circuit Breaker - Claude Generated FIX
        cached_count = file_cached_count  # Start with file-based hits
        failed_keywords = []  # Track which keywords failed

        for clean_kw in normalized_keywords.keys():
            # Check circuit breaker first
            if self._check_circuit_breaker():
                failed_keywords.append((clean_kw, 'circuit_breaker', 'Too many consecutive failures'))
                logger.warning(f"⏩ Skipping '{clean_kw}' - circuit breaker open")
                continue

            # Check cache (now returns tuple with status)
            # RESPECT disable_sql_cache flag for testing - Claude Generated
            cache_result = None
            if not self.disable_sql_cache:
                cache_result = dk_cache.get_catalog_dk_cache(clean_kw)
            else:
                logger.debug(f"🚫 SQL cache DISABLED: Bypassing cache lookup for '{clean_kw}'")
            if cache_result:
                cached_titles, status, error_msg = cache_result

                if status == 'success' and cached_titles:
                    all_titles.extend(cached_titles)
                    cached_keywords.add(clean_kw)  # Track as cached
                    cached_count += 1
                    logger.info(f"✅ Cache HIT for '{clean_kw}': {len(cached_titles)} titles")
                    self._record_search_success()
                    continue
                elif status != 'success':
                    # Cached failure - respect TTL
                    logger.debug(f"Cached failure for '{clean_kw}': {status} - {error_msg}")
                    failed_keywords.append((clean_kw, status, error_msg))
                    self._record_search_success()  # Don't count as new failure
                    continue

            # Cache miss - perform live search with rate limiting
            self._apply_rate_limit()
            self._reset_session_if_needed()

            logger.info(f"⚠️ Cache MISS for '{clean_kw}': performing live search")

            try:
                search_results = self.search(clean_kw, search_type="ku")

                # DIAGNOSTIC: Log SOAP search result count - Claude Generated
                logger.debug(f"SOAP search for '{clean_kw}': found {len(search_results)} catalog entries")
                if search_results and len(search_results) > 0:
                    # Sample first 3 RSNs for verification
                    sample_rsns = [r.get('rsn', 'N/A') for r in search_results[:3]]
                    logger.debug(f"   Sample RSNs: {sample_rsns}")

                # Fallback to web if SOAP fails
                if not search_results and self.enable_web_fallback:
                    logger.info(f"SOAP failed, trying web fallback for '{clean_kw}'")
                    search_results = self._search_web(clean_kw)

                if not search_results:
                    logger.warning(f"No results found for '{clean_kw}'")
                    # Cache the empty result with status
                    # RESPECT disable_sql_cache flag for testing - Claude Generated
                    if not self.disable_sql_cache:
                        dk_cache.store_catalog_dk_cache(clean_kw, [], status='no_results', ttl_minutes=30)
                    else:
                        logger.debug(f"🚫 SQL cache DISABLED: Not caching failure for '{clean_kw}'")
                    failed_keywords.append((clean_kw, 'no_results', 'No catalog entries found'))
                    self._record_search_success()
                    continue

                # Build title list with classifications
                title_list = []
                # Use adaptive delay based on JSON cache hit rate - Claude Generated
                adaptive_delay = self._get_appropriate_delay()
                processed = self.process_search_results(search_results, max_items=max_results, delay=adaptive_delay)

                # DIAGNOSTIC: Log processed results summary - Claude Generated
                if processed:
                    items_with_dk = sum(1 for item in processed if item.get('decimal_classifications'))
                    items_with_rvk = sum(1 for item in processed if item.get('rvk_classifications'))
                    logger.debug(f"Processed '{clean_kw}': {len(processed)} items, DK={items_with_dk}, RVK={items_with_rvk}")
                else:
                    logger.warning(f"No items successfully processed for '{clean_kw}' (all Details calls failed)")

                for item in processed:
                    rsn = item.get("rsn")
                    title = item.get("title")

                    # Extract DK and RVK classifications as strings
                    classifications = []

                    # Add DK classifications
                    for dk in item.get("decimal_classifications", []):
                        classifications.append(f"DK {dk}")

                    # Add RVK classifications
                    for rvk in item.get("rvk_classifications", []):
                        classifications.append(f"RVK {rvk}")

                    if title and classifications:
                        title_list.append({
                            "rsn": rsn,
                            "title": title,
                            "classifications": classifications
                        })

                # Cache the titles with success status
                if title_list:
                    # RESPECT disable_sql_cache flag for testing - Claude Generated
                    if not self.disable_sql_cache:
                        dk_cache.store_catalog_dk_cache(clean_kw, title_list, status='success')
                    else:
                        logger.debug(f"🚫 SQL cache DISABLED: Not caching {len(title_list)} titles for '{clean_kw}'")
                    all_titles.extend(title_list)
                    logger.debug(f"Cached {len(title_list)} titles for '{clean_kw}'")
                    self._record_search_success()
                else:
                    # No classifications found in results
                    # RESPECT disable_sql_cache flag for testing - Claude Generated
                    if not self.disable_sql_cache:
                        dk_cache.store_catalog_dk_cache(clean_kw, [], status='no_results', ttl_minutes=30)
                    else:
                        logger.debug(f"🚫 SQL cache DISABLED: Not caching no-results for '{clean_kw}'")
                    failed_keywords.append((clean_kw, 'no_results', 'No classifications in results'))
                    self._record_search_success()

            except Exception as e:
                if isinstance(e, requests.exceptions.Timeout):
                    logger.error(f"⏱️ Timeout searching '{clean_kw}': {e}")
                    # RESPECT disable_sql_cache flag for testing - Claude Generated
                    if not self.disable_sql_cache:
                        dk_cache.store_catalog_dk_cache(clean_kw, [], status='timeout',
                                                       error_message=str(e), ttl_minutes=60)
                    else:
                        logger.debug(f"🚫 SQL cache DISABLED: Not caching timeout for '{clean_kw}'")
                    failed_keywords.append((clean_kw, 'timeout', str(e)))
                    self._record_search_failure()
                else:
                    logger.error(f"❌ Error searching '{clean_kw}': {e}")
                    # RESPECT disable_sql_cache flag for testing - Claude Generated
                    if not self.disable_sql_cache:
                        dk_cache.store_catalog_dk_cache(clean_kw, [], status='error',
                                                       error_message=str(e), ttl_minutes=60)
                    else:
                        logger.debug(f"🚫 SQL cache DISABLED: Not caching error for '{clean_kw}'")
                    failed_keywords.append((clean_kw, 'error', str(e)))
                    self._record_search_failure()

        # Log summary of failures
        if failed_keywords:
            logger.warning(f"⚠️ {len(failed_keywords)} keywords failed: {[k for k, _, _ in failed_keywords]}")

        # Process each keyword INDIVIDUALLY to maintain correct keyword-classification associations - Claude Generated
        # FIX: Previous implementation mixed titles from different keywords, causing wrong matched_keywords assignments
        # Now each keyword gets only its own titles and classifications
        keyword_results = []

        for clean_kw, original_kw in normalized_keywords.items():
            # Get titles for THIS keyword only (from cache - all keywords have been searched/cached by now)
            # Now returns tuple of (titles, status, error_message)
            cache_result = dk_cache.get_catalog_dk_cache(clean_kw)
            if not cache_result:
                logger.warning(f"⚠️ No cache entry for keyword '{clean_kw}' after search/cache operations")
                continue

            kw_titles, status, error_msg = cache_result

            if not kw_titles:
                logger.warning(f"⚠️ No titles cached for keyword '{clean_kw}' (status={status}): {error_msg}")
                continue

            # Extract classifications for THIS keyword only - Claude Generated
            # Pass ONLY this keyword to avoid mixing with other keywords' classifications
            kw_classifications = dk_cache.extract_classifications_from_titles(
                kw_titles,
                matched_keywords=[clean_kw]  # ✅ FIXED: Only this specific keyword
            )

            if kw_classifications:  # Only add if keyword has classifications
                keyword_results.append({
                    "keyword": original_kw,
                    "source": "cache" if clean_kw in cached_keywords else "live",
                    "search_time_ms": 0.0,  # TODO: Track actual timing
                    "classifications": kw_classifications
                })
                logger.info(f"✅ Keyword '{clean_kw}': {len(kw_classifications)} classifications from {len(kw_titles)} titles")
            else:
                logger.debug(f"⚠️ Keyword '{clean_kw}': {len(kw_titles)} titles but no classifications extracted")

        # Cache statistics - Claude Generated
        total_keywords = len(normalized_keywords)
        cache_hit_rate = (cached_count / total_keywords * 100) if total_keywords > 0 else 0
        logger.info(
            f"📊 Cache Statistics: {cached_count}/{total_keywords} hits ({cache_hit_rate:.0f}%) "
            f"| Returning {len(keyword_results)} keyword-centric results for GUI display"
        )

        # Log performance statistics - Claude Generated
        self.log_performance_stats()

        return keyword_results

    def _flatten_to_dk_centric(
        self,
        keyword_centric_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert keyword-centric results back to DK-centric format
        Claude Generated - Backward compatibility converter

        Converts from keyword-grouped format to DK-code grouped format
        for compatibility with downstream Classification step

        Args:
            keyword_centric_results: List of keyword-grouped results

        Returns:
            List of DK-code grouped results (old format)
        """
        dk_centric_map = {}

        # Group by DK code, collecting all keywords that led to each DK
        for kw_result in keyword_centric_results:
            keyword = kw_result.get("keyword", "unknown")
            source = kw_result.get("source", "unknown")

            for cls in kw_result.get("classifications", []):
                dk_code = cls.get("dk", "unknown")

                if dk_code not in dk_centric_map:
                    # First occurrence of this DK code
                    dk_centric_map[dk_code] = {
                        "dk": dk_code,
                        "classification_type": cls.get("type", "DK"),
                        "titles": cls.get("titles", []),
                        "count": cls.get("count", 0),
                        "avg_confidence": cls.get("avg_confidence", 0.8),
                        "gnd_ids": cls.get("gnd_ids", []),
                        "matched_keywords": [],
                        "sources": []  # Track which keywords came from cache vs live
                    }

                # Add keyword to this DK code's keyword list
                if keyword not in dk_centric_map[dk_code]["matched_keywords"]:
                    dk_centric_map[dk_code]["matched_keywords"].append(keyword)

                # Track source
                if source not in dk_centric_map[dk_code]["sources"]:
                    dk_centric_map[dk_code]["sources"].append(source)

        # Convert to list
        result_list = list(dk_centric_map.values())

        # Sort by count and confidence (same as original)
        result_list.sort(key=lambda x: (x["count"], x["avg_confidence"]), reverse=True)

        return result_list

    def _restructure_to_keyword_centric(
        self,
        dk_results: List[Dict[str, Any]],
        keyword_sources: Dict[str, str],
        keyword_timings: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Restructure DK-centric results to keyword-centric format
        Claude Generated - Keyword-centric restructuring

        Converts from DK-code grouped results to keyword-grouped results,
        maintaining classification details nested under keywords

        Args:
            dk_results: List of DK-code grouped results
            keyword_sources: Dict mapping keyword to "cache" or "live"
            keyword_timings: Dict mapping keyword to search time in ms

        Returns:
            List of keyword-centric result dicts with nested classifications
        """
        keyword_results = {}

        # Initialize keyword entries from sources
        for keyword, source in keyword_sources.items():
            if keyword not in keyword_results:
                keyword_results[keyword] = {
                    "keyword": keyword,
                    "source": source,
                    "search_time_ms": keyword_timings.get(keyword, 0),
                    "classifications": [],
                    "total_titles": 0
                }

        # Group DK results by keyword - each DK result maps to multiple keywords
        for dk_result in dk_results:
            matched_keywords = dk_result.get("matched_keywords", [])

            for keyword in matched_keywords:
                if keyword not in keyword_results:
                    # Initialize if missing (shouldn't happen, but defensive)
                    keyword_results[keyword] = {
                        "keyword": keyword,
                        "source": keyword_sources.get(keyword, "unknown"),
                        "search_time_ms": keyword_timings.get(keyword, 0),
                        "classifications": [],
                        "total_titles": 0
                    }

                # Add DK classification to this keyword
                classification = {
                    "dk": dk_result.get("dk", ""),
                    "type": dk_result.get("classification_type", "DK"),
                    "titles": dk_result.get("titles", []),
                    "count": dk_result.get("count", 0),
                    "avg_confidence": dk_result.get("avg_confidence", 0.8),
                    "gnd_ids": dk_result.get("gnd_ids", [])
                }
                keyword_results[keyword]["classifications"].append(classification)

        # Calculate total titles per keyword
        for keyword_data in keyword_results.values():
            total_titles = sum(c.get("count", 0) for c in keyword_data["classifications"])
            keyword_data["total_titles"] = total_titles

        # Return as list, sorted by source (cache first, then by title count)
        result_list = list(keyword_results.values())
        result_list.sort(key=lambda x: (x["source"] != "cache", -x["total_titles"]))

        return result_list

    def _calculate_dk_confidence(self, item: Dict[str, Any], keyword: str) -> float:
        """Claude Generated - Calculate confidence score for DK classification"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if keyword appears in title
        title = item.get("title", "").lower()
        if keyword.lower() in title:
            confidence += 0.3
            
        # Higher confidence if item has subjects
        if item.get("subjects") or item.get("mab_subjects"):
            confidence += 0.2
            
        return min(confidence, 1.0)
    

    def save_to_csv(self, items: List[Dict[str, Any]], filename: str) -> None:
        """
        Save the processed items to a CSV file.

        Args:
            items: List of processed items
            filename: Name of the output file
        """
        if not items:
            logger.warning("No items to save.")
            return

        # Define the columns for the CSV file
        fieldnames = [
            "rsn",
            "title",
            "author",
            "publication",
            "isbn",
            "subjects",
            "mab_subjects",
            "decimal_classifications",
        ]

        logger.info(f"Saving {len(items)} items to {filename}")

        try:
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for item in items:
                    # Prepare the row
                    row = {
                        "rsn": item.get("rsn", ""),
                        "title": item.get("title", ""),
                        "author": ", ".join(item.get("author", [])),
                        "publication": item.get("publication", ""),
                        "isbn": item.get("isbn", ""),
                        "subjects": "|".join(item.get("subjects", [])),
                        "mab_subjects": "|".join(item.get("mab_subjects", [])),
                        "decimal_classifications": "|".join(
                            item.get("decimal_classifications", [])
                        ),
                    }
                    writer.writerow(row)

            logger.info(f"Data successfully saved to {filename}")

        except IOError as e:
            logger.error(f"Error saving to CSV: {e}")


def main():
    """Main function to run the tool from the command line."""
    parser = argparse.ArgumentParser(description="Extract data from library catalog")
    parser.add_argument("search_term", help="The search term")
    parser.add_argument(
        "--output",
        "-o",
        default="library_data.csv",
        help="Output CSV file (default: library_data.csv)",
    )
    parser.add_argument(
        "--search-type",
        "-t",
        default="ku",
        help="Search type code (default: ku - anyword)",
    )
    parser.add_argument(
        "--max-items",
        "-m",
        type=int,
        default=100,
        help="Maximum number of items to process (default: 100)",
    )
    parser.add_argument(
        "--token", default="xxxxx", help="API token (default: xxxxx - replace with your token)"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable detailed debug output"
    )

    args = parser.parse_args()

    # Set log level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(f"BiblioExtractor starting...")
    logger.info(f"Search term: '{args.search_term}'")
    logger.info(f"Search type: '{args.search_type}'")
    logger.info(f"Max items: {args.max_items}")
    logger.info(f"Output file: {args.output}")

    extractor = BiblioClient(token=args.token, debug=args.debug)

    results = extractor.search(args.search_term, args.search_type)

    if not results:
        logger.warning("No results found.")
        return

    logger.info(f"Found {len(results)} results.")

    processed_items = extractor.process_search_results(
        results, max_items=args.max_items
    )

    extractor.save_to_csv(processed_items, args.output)

    logger.info("Process completed.")


if __name__ == "__main__":
    main()
