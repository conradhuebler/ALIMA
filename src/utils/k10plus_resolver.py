"""
K10Plus SRU API resolver for Paketsigel (package seals).
Claude Generated - Port of QuerySiegel() from SUSHIMenu/k10plus.py without Qt dependency.

Usage:
    dois = fetch_dois_for_siegel("ZDB-2-CMS", cache_dir="/tmp/k10plus_cache")
"""

import logging
import os
import re
import xml.dom.minidom
from typing import Callable, List, Optional

import requests

_SRU_BASE = (
    "http://sru.k10plus.de/opac-de-627"
    "?version=1.2&operation=searchRetrieve"
    "&query=pica.xpr%3D{siegel}"
    "&recordSchema=dc"
)
_SRU_PAGE = (
    "http://sru.k10plus.de/opac-de-627"
    "?version=1.2&operation=searchRetrieve"
    "&query=pica.xpr%3D{siegel}"
    "&maximumRecords=1000&startRecord={start}"
    "&recordSchema=dc"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
    "Accept-Encoding": "gzip",
}
_DOI_STRIP = re.compile(r"[\n\t\s]*")


def _clean_doi(raw: str) -> str:
    """Strip known prefixes and whitespace from a raw DOI string."""
    doi = _DOI_STRIP.sub("", raw)
    for prefix in ("doi:", "http://dx.doi.org/", "https://doi.org/",
                   "http://www.springerreference.com/index/bookdoi/"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi.strip()


def _get_cached_or_fetch(url: str, cache_path: Optional[str], logger: logging.Logger) -> bytes:
    """Return cached XML bytes or fetch from URL and cache."""
    if cache_path and os.path.exists(cache_path):
        logger.debug(f"Cache hit: {cache_path}")
        with open(cache_path, "rb") as fh:
            return fh.read()

    logger.debug(f"Fetching: {url}")
    response = requests.get(url, headers=_HEADERS, allow_redirects=True, timeout=60)
    response.raise_for_status()
    data = response.content

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as fh:
            fh.write(data)

    return data


def _parse_dois_from_xml(xml_bytes: bytes) -> List[str]:
    """Extract DOIs from SRU DC-schema XML bytes."""
    dom = xml.dom.minidom.parseString(xml_bytes)
    dois: List[str] = []
    for record in dom.documentElement.getElementsByTagName("zs:record"):
        for record_data in record.getElementsByTagName("zs:recordData"):
            for ident in record_data.getElementsByTagName("dc:identifier"):
                if ident.firstChild is None:
                    continue
                text = ident.firstChild.data
                if "doi" in text.lower():
                    doi = _clean_doi(text)
                    if doi:
                        dois.append(doi)
    return dois


def fetch_dois_for_siegel(
    siegel: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Fetch all DOIs associated with a K10Plus Paketsigel via SRU API.

    Args:
        siegel: Paketsigel identifier, e.g. "ZDB-2-CMS".
        cache_dir: Optional directory for caching raw XML responses.
                   None = no caching.
        progress_callback: Called as (current_page, total_pages, message) during fetch.
        logger: Optional logger; falls back to module logger if not provided.

    Returns:
        List of DOI strings (duplicates possible if the catalog has them).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # --- Step 1: fetch count record ------------------------------------------
    count_url = _SRU_BASE.format(siegel=siegel)
    count_cache = os.path.join(cache_dir, f"{siegel}_fetch_.xml") if cache_dir else None

    try:
        count_xml = _get_cached_or_fetch(count_url, count_cache, logger)
    except Exception as exc:
        logger.error(f"Failed to fetch record count for '{siegel}': {exc}")
        raise

    dom = xml.dom.minidom.parseString(count_xml)
    nr_nodes = dom.documentElement.getElementsByTagName("zs:numberOfRecords")
    if not nr_nodes:
        logger.warning(f"numberOfRecords element missing for siegel '{siegel}'")
        return []

    total_records = int(nr_nodes[0].firstChild.data)
    logger.info(f"Siegel '{siegel}': {total_records} records found")

    if total_records == 0:
        return []

    # Calculate total pages (1000 records per page)
    pages = list(range(1, total_records + 1, 1000))
    total_pages = len(pages)

    # --- Step 2: paginated fetch ----------------------------------------------
    all_dois: List[str] = []

    for page_index, start_record in enumerate(pages, start=1):
        page_url = _SRU_PAGE.format(siegel=siegel, start=start_record)
        page_cache = (
            os.path.join(cache_dir, f"{siegel}_range_{start_record}.xml")
            if cache_dir else None
        )

        msg = f"Lade Records {start_record}–{min(start_record + 999, total_records)} / {total_records}"
        logger.info(f"  [{page_index}/{total_pages}] {msg}")

        if progress_callback:
            try:
                progress_callback(page_index, total_pages, msg)
            except Exception:
                pass

        try:
            page_xml = _get_cached_or_fetch(page_url, page_cache, logger)
        except Exception as exc:
            logger.error(f"Failed to fetch page {page_index} for '{siegel}': {exc}")
            continue

        page_dois = _parse_dois_from_xml(page_xml)
        all_dois.extend(page_dois)
        logger.debug(f"    → {len(page_dois)} DOIs on this page")

    logger.info(f"Siegel '{siegel}': {len(all_dois)} DOIs extracted")
    return all_dois
