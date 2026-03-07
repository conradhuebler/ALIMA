"""
K10Plus SRU API resolver for Paketsigel (package seals).
Claude Generated - Port of QuerySiegel() from SUSHIMenu/k10plus.py without Qt dependency.

Usage:
    records = fetch_records_for_siegel("ZDB-2-CMS", cache_dir="/tmp/k10plus_cache")
    for record in records:
        print(f"{record['title']} ({record['year']}) - DOI: {record['doi']}")

Returns list of K10PlusRecord dicts with full PICA-XML metadata.
"""

import logging
import os
import re
import xml.dom.minidom
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass

import requests

# PICA-XML schema for rich metadata (vs. dc which only has basic fields)
_SRU_BASE = (
    "http://sru.k10plus.de/opac-de-627"
    "?version=1.2&operation=searchRetrieve"
    "&query=pica.xpr%3D{siegel}"
    "&recordSchema=picaxml"
)
_SRU_PAGE = (
    "http://sru.k10plus.de/opac-de-627"
    "?version=1.2&operation=searchRetrieve"
    "&query=pica.xpr%3D{siegel}"
    "&maximumRecords=1000&startRecord={start}"
    "&recordSchema=picaxml"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
    "Accept-Encoding": "gzip",
}
_DOI_STRIP = re.compile(r"[\n\t\s]*")


@dataclass
class K10PlusRecord:
    """Structured record from K10Plus PICA-XML - Claude Generated"""
    ppn: str = ""                    # 003@ - Primary record ID
    doi: str = ""                    # 004V - Digital Object Identifier
    isbn: str = ""                   # 004A - Primary ISBN
    additional_isbns: List[str] = None  # 004P - Additional ISBNs
    title: str = ""                  # 021A - Full title with subtitle
    authors: List[str] = None        # 028C - Authors/Editors
    publisher: str = ""              # 033A - Publisher
    place: str = ""                  # 033A - Place of publication
    year: str = ""                   # 033A - Publication year
    edition: str = ""                # 032@ - Edition statement
    series: str = ""                 # 036E - Series title
    ddc: str = ""                    # 045F - Dewey Decimal Classification
    subjects: List[str] = None       # 044A - Subject keywords
    paketsiegel: str = ""            # 017L - Package seal
    access_conditions: str = ""      # 237A - Access conditions
    url: str = ""                    # 017C/209R - URL/DOI link

    def __post_init__(self):
        if self.additional_isbns is None:
            self.additional_isbns = []
        if self.authors is None:
            self.authors = []
        if self.subjects is None:
            self.subjects = []

    def to_display_string(self, compact: bool = False) -> str:
        """Generate human-readable display string - Claude Generated"""
        parts = []

        # Title (always first)
        if self.title:
            parts.append(self.title)

        # Year
        if self.year:
            parts.append(f"({self.year})")

        # Authors (up to 3)
        if self.authors and len(self.authors) <= 3:
            parts.append(f"— {'; '.join(self.authors[:3])}")
        elif self.authors and len(self.authors) > 3:
            parts.append(f"— {self.authors[0]} et al.")

        display = " ".join(parts)

        # DOI/ISBN for identification
        if self.doi:
            display += f" [DOI: {self.doi}]"
        elif self.isbn:
            display += f" [ISBN: {self.isbn}]"

        return display


def _clean_doi(raw: str) -> str:
    """Strip known prefixes and whitespace from a raw DOI string."""
    doi = _DOI_STRIP.sub("", raw)
    for prefix in ("doi:", "http://dx.doi.org/", "https://doi.org/",
                   "http://www.springerreference.com/index/bookdoi/"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi.strip()


def _find_elements_by_localname(parent, localname: str):
    """
    Find all elements with given local name, regardless of namespace - Claude Generated.

    Works around xml.dom.minidom namespace handling limitations.
    """
    result = []
    for elem in parent.getElementsByTagName("*"):
        if elem.localName == localname or elem.tagName.endswith(localname):
            result.append(elem)
    return result


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


def load_cached_records(cache_dir: str, siegel: str, logger: Optional[logging.Logger] = None) -> List[K10PlusRecord]:
    """
    Load previously cached K10Plus records without API call - Claude Generated.

    Args:
        cache_dir: Directory where XML cache files are stored
        siegel: Paketsigel identifier (e.g., "ZDB-2-CMS")
        logger: Optional logger

    Returns:
        List of K10PlusRecord objects from cache, empty list if no cache found

    Usage:
        records = load_cached_records("/tmp/k10plus_cache", "ZDB-2-CMS")
        for r in records:
            print(f"{r.title} ({r.year})")
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.exists(cache_dir):
        logger.warning(f"Cache directory not found: {cache_dir}")
        return []

    # Find all cached XML files for this siegel
    cache_files = sorted([
        f for f in os.listdir(cache_dir)
        if f.startswith(f"{siegel}_") and f.endswith(".xml")
    ])

    if not cache_files:
        logger.warning(f"No cached files found for siegel '{siegel}' in {cache_dir}")
        return []

    logger.info(f"Found {len(cache_files)} cached file(s) for '{siegel}'")

    all_records: List[K10PlusRecord] = []

    for cache_file in cache_files:
        cache_path = os.path.join(cache_dir, cache_file)
        try:
            with open(cache_path, "rb") as fh:
                xml_bytes = fh.read()
            records = _parse_pica_xml(xml_bytes)
            all_records.extend(records)
            logger.debug(f"  Loaded {len(records)} records from {cache_file}")
        except Exception as e:
            logger.error(f"Failed to parse {cache_file}: {e}")

    logger.info(f"Total: {len(all_records)} records loaded from cache")
    return all_records


def _parse_pica_record(field_elem) -> Dict[str, Any]:
    """Parse a single PICA field element into structured data - Claude Generated"""
    result = {}

    # Get subfield codes (attribute "e" for content, "a" for first subfield)
    def get_subfield(code: str) -> str:
        for child in field_elem.childNodes:
            if child.nodeType == child.ELEMENT_NODE:
                subfield_code = child.getAttribute("code")
                if subfield_code == code:
                    if child.firstChild:
                        return child.firstChild.data.strip()
        return ""

    return result


def _parse_pica_xml(xml_bytes: bytes) -> List[K10PlusRecord]:
    """
    Extract full bibliographic records from SRU PICA-XML response - Claude Generated.

    Handles both namespaced (pica:) and default namespace XML.

    PICA Field Mapping:
    - 003@ $0 → PPN (record ID)
    - 004A $0 → ISBN
    - 004P $0 → Additional ISBNs
    - 004V $0 → DOI
    - 017C $u → URL with DOI
    - 017L $a → Paketsiegel, $b → Year
    - 021A $a → Title, $b → Subtitle
    - 028C $d → Person surname, $a → Forename, $B → Role
    - 032@ $a → Edition
    - 033A $p → Place, $n → Publisher, $h → Year
    - 036E $a → Series
    - 044A $a → Subject keywords
    - 045F $a → DDC classification
    - 209R $u → URL, $y → Access note
    - 237A $a → Access conditions
    """
    dom = xml.dom.minidom.parseString(xml_bytes)
    records: List[K10PlusRecord] = []

    # PICA XML namespaces
    ZS_NS = "http://www.loc.gov/zing/srw/"
    PICA_NS = "info:srw/schema/5/picaXML-v1.0"

    # Find all <zs:record> elements
    record_elems = dom.getElementsByTagName("zs:record")
    if not record_elems:
        record_elems = dom.getElementsByTagNameNS(ZS_NS, "record")

    for record_elem in record_elems:
        # Find <zs:recordData> within this record
        record_data_elems = record_elem.getElementsByTagName("zs:recordData")
        if not record_data_elems:
            record_data_elems = record_elem.getElementsByTagNameNS(ZS_NS, "recordData")

        if not record_data_elems:
            continue

        record_data = record_data_elems[0]
        record = K10PlusRecord()
        additional_isbns = []
        authors = []
        subjects = []

        # PICA XML has structure: recordData -> <record> (default namespace) -> <datafield>
        # Find the <record> element inside recordData
        pica_record_elem = None
        for child in record_data.childNodes:
            if child.nodeType == child.ELEMENT_NODE:
                # Look for <record> element (may be in PICA namespace or default)
                if child.localName == "record" or child.tagName.endswith("record"):
                    pica_record_elem = child
                    break

        if not pica_record_elem:
            # Fallback: use recordData directly if no <record> wrapper found
            pica_record_elem = record_data

        # Parse all PICA fields (datafield elements) inside the record
        for field_container in pica_record_elem.childNodes:
            if field_container.nodeType != field_container.ELEMENT_NODE:
                continue

            # Get tag name - looking for <datafield> elements
            tag_name = field_container.tagName
            local_name = field_container.localName if field_container.localName else tag_name

            # Skip non-datafield elements
            if local_name != "datafield":
                continue

            # Get the field tag from the "tag" attribute (e.g., "003@", "004A")
            field_tag = field_container.getAttribute("tag")
            if not field_tag:
                continue

            # Helper to get subfield content by code
            def get_subfield(code: str) -> str:
                for child in field_container.childNodes:
                    if child.nodeType == child.ELEMENT_NODE:
                        subfield_code = child.getAttribute("code")
                        if subfield_code == code:
                            if child.firstChild:
                                return child.firstChild.data.strip()
                return ""

            # 003@ - PPN (Primary record number)
            if field_tag == "003@":
                record.ppn = get_subfield("0")

            # 004A - ISBN
            elif field_tag == "004A":
                record.isbn = get_subfield("0")

            # 004P - Additional ISBNs
            elif field_tag == "004P":
                isbn_val = get_subfield("0")
                if isbn_val:
                    additional_isbns.append(isbn_val)

            # 004V - DOI
            elif field_tag == "004V":
                doi_raw = get_subfield("0")
                if doi_raw:
                    record.doi = _clean_doi(doi_raw)

            # 017C - URL with DOI
            elif field_tag == "017C":
                record.url = get_subfield("u")

            # 017L - Paketsiegel
            elif field_tag == "017L":
                siegel_a = get_subfield("a")
                siegel_b = get_subfield("b")  # Year component
                if siegel_a:
                    record.paketsiegel = siegel_a

            # 021A - Title with subtitle
            elif field_tag == "021A":
                title_a = get_subfield("a")
                title_b = get_subfield("b")
                if title_a or title_b:
                    record.title = f"{title_a} {title_b}".strip()

            # 028C - Persons (Authors/Editors)
            elif field_tag == "028C":
                person_d = get_subfield("d")  # Surname
                person_a = get_subfield("a")  # Forename
                person_b = get_subfield("B")  # Role (e.g., "HerausgeberIn")
                if person_d or person_a:
                    name_parts = []
                    if person_a:
                        name_parts.append(person_a)
                    if person_d:
                        name_parts.append(person_d)
                    author_name = " ".join(name_parts)
                    if person_b and "herausgeber" in person_b.lower():
                        author_name += f" (Hrsg.)"
                    authors.append(author_name)

            # 032@ - Edition
            elif field_tag == "032@":
                record.edition = get_subfield("a")

            # 033A - Publication info (place, publisher, year)
            elif field_tag == "033A":
                place = get_subfield("p")
                publisher = get_subfield("n")
                year = get_subfield("h")
                if place:
                    record.place = place
                if publisher:
                    record.publisher = publisher
                if year:
                    record.year = year

            # 036E - Series
            elif field_tag == "036E":
                record.series = get_subfield("a")

            # 044A - Subject keywords
            elif field_tag == "044A":
                subject = get_subfield("a")
                if subject:
                    subjects.append(subject)

            # 045F - DDC Classification
            elif field_tag == "045F":
                record.ddc = get_subfield("a")

            # 209R - URL/Access
            elif field_tag == "209R":
                url = get_subfield("u")
                access_note = get_subfield("y")
                if url and not record.url:
                    record.url = url
                if access_note and not record.access_conditions:
                    record.access_conditions = access_note

            # 237A - Access conditions
            elif field_tag == "237A":
                access = get_subfield("a")
                if access:
                    record.access_conditions = access

        record.additional_isbns = additional_isbns
        record.authors = authors
        record.subjects = subjects

        # Only add if we have at least some identifying information
        if record.ppn or record.doi or record.title:
            records.append(record)

    return records


def _parse_dois_from_xml(xml_bytes: bytes) -> List[str]:
    """Extract DOIs from SRU PICA-XML bytes (backward compatibility) - Claude Generated"""
    records = _parse_pica_xml(xml_bytes)
    return [r.doi for r in records if r.doi]


def fetch_records_for_siegel(
    siegel: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[K10PlusRecord]:
    """
    Fetch all bibliographic records associated with a K10Plus Paketsigel via SRU API.

    Uses PICA-XML schema for rich metadata extraction (title, authors, year, ISBN, DOI, etc.).

    Args:
        siegel: Paketsigel identifier, e.g. "ZDB-2-CMS".
        cache_dir: Optional directory for caching raw XML responses.
                   None = no caching.
        progress_callback: Called as (current_page, total_pages, message) during fetch.
        logger: Optional logger; falls back to module logger if not provided.

    Returns:
        List of K10PlusRecord objects with full bibliographic metadata.
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
    all_records: List[K10PlusRecord] = []

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

        page_records = _parse_pica_xml(page_xml)
        all_records.extend(page_records)
        logger.debug(f"    → {len(page_records)} records on this page")

    logger.info(f"Siegel '{siegel}': {len(all_records)} records extracted")
    return all_records


def fetch_dois_for_siegel(
    siegel: str,
    cache_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Fetch all DOIs associated with a K10Plus Paketsigel via SRU API.

    Backward compatibility wrapper - use fetch_records_for_siegel() for full metadata.

    Args:
        siegel: Paketsigel identifier, e.g. "ZDB-2-CMS".
        cache_dir: Optional directory for caching raw XML responses.
                   None = no caching.
        progress_callback: Called as (current_page, total_pages, message) during fetch.
        logger: Optional logger; falls back to module logger if not provided.

    Returns:
        List of DOI strings (duplicates possible if the catalog has them).
    """
    records = fetch_records_for_siegel(
        siegel=siegel,
        cache_dir=cache_dir,
        progress_callback=progress_callback,
        logger=logger,
    )
    return [r.doi for r in records if r.doi]
