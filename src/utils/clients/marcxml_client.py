"""
MARC XML Client for ALIMA - Claude Generated

Provides catalog search and record retrieval via MARC XML (SRU/Z39.50 protocols).
Compatible with standard library systems that support MARC21 XML export.

Supports:
- SRU (Search/Retrieve via URL) - REST-based protocol
- OAI-PMH harvesting (optional)
- Direct MARC XML file parsing

MARC XML Reference: https://www.loc.gov/standards/marcxml/
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import time
import re
import logging
from urllib.parse import urlencode, quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("marcxml_client")


# MARC21 field definitions for bibliographic data
MARC_FIELDS = {
    # Control fields
    "001": "control_number",
    "003": "control_number_identifier",
    "005": "date_time_last_transaction",
    "008": "fixed_length_data",
    
    # ISBN/ISSN
    "020": "isbn",
    "022": "issn",
    
    # Classification fields
    "082": "ddc",  # Dewey Decimal Classification
    "083": "ddc_additional",
    "084": "other_classification",  # Often used for RVK, local classifications
    "090": "local_call_number",
    
    # Main entry fields
    "100": "main_author",
    "110": "corporate_author",
    "111": "meeting_name",
    
    # Title fields
    "245": "title",
    "246": "varying_form_title",
    "250": "edition",
    
    # Publication fields
    "260": "publication_old",  # Pre-RDA
    "264": "publication",  # RDA format
    
    # Physical description
    "300": "physical_description",
    
    # Series
    "490": "series",
    
    # Notes
    "500": "general_note",
    "504": "bibliography_note",
    "520": "summary",  # Abstract
    
    # Subject headings
    "600": "subject_person",
    "610": "subject_corporate",
    "611": "subject_meeting",
    "630": "subject_uniform_title",
    "648": "subject_chronological",
    "650": "subject_topical",  # Main subject keywords
    "651": "subject_geographic",
    "653": "index_term_uncontrolled",
    "655": "genre_form",
    
    # Added entries
    "700": "added_author",
    "710": "added_corporate",
    "711": "added_meeting",
    
    # Electronic location
    "856": "electronic_location",
}

# GND-specific subfield codes
GND_SUBFIELDS = {
    "0": "authority_record_control_number",  # Contains GND ID
    "2": "source_of_heading",  # e.g., "gnd", "lcsh"
}

# MARC XML namespaces
MARC_NS = {
    "marc": "http://www.loc.gov/MARC21/slim",
    "srw": "http://www.loc.gov/zing/srw/",
    "diag": "http://www.loc.gov/zing/srw/diagnostic/",
}


class MarcXmlClient:
    """
    Client for searching and retrieving bibliographic records in MARC XML format.
    
    Supports SRU (Search/Retrieve via URL) protocol which is commonly used by:
    - Deutsche Nationalbibliothek (DNB)
    - Library of Congress
    - OCLC WorldCat
    - Many university library systems
    
    Usage:
        client = MarcXmlClient(
            sru_base_url="https://services.dnb.de/sru/dnb",
            database="dnb"
        )
        results = client.search("Quantenchemie")
        for record in results:
            print(record["title"], record["classifications"])
    """
    
    # Default SRU endpoints for common catalogs
    KNOWN_ENDPOINTS = {
        "dnb": {
            "url": "https://services.dnb.de/sru/dnb",
            "database": "dnb",
            "schema": "MARC21-xml",
        },
        "loc": {
            "url": "https://lx2.loc.gov:210/lcdb",
            "database": "lcdb",
            "schema": "marcxml",
        },
        "gbv": {
            "url": "https://sru.gbv.de/gvk",
            "database": "gvk",
            "schema": "marcxml",
        },
        "swb": {
            "url": "https://swb.bsz-bw.de/sru/swb",
            "database": "swb",
            "schema": "marcxml",
        },
        "k10plus": {
            "url": "https://sru.k10plus.de/opac-de-627",
            "database": "opac-de-627",
            "schema": "marcxml",
        },
    }
    
    def __init__(
        self,
        sru_base_url: str = "",
        database: str = "",
        schema: str = "marcxml",
        preset: str = "",
        debug: bool = False,
        timeout: int = 30,
        max_records: int = 50,
    ):
        """
        Initialize the MARC XML client.
        
        Args:
            sru_base_url: Base URL for SRU endpoint (e.g., "https://services.dnb.de/sru/dnb")
            database: Database name for SRU query
            schema: Record schema to request (default: "marcxml")
            preset: Use a known endpoint preset ("dnb", "loc", "gbv", "swb", "k10plus")
            debug: Enable debug logging
            timeout: Request timeout in seconds
            max_records: Maximum records to retrieve per search
        """
        self.debug = debug
        self.timeout = timeout
        self.max_records = max_records
        self.session = requests.Session()
        
        # Apply preset if specified
        if preset and preset.lower() in self.KNOWN_ENDPOINTS:
            endpoint = self.KNOWN_ENDPOINTS[preset.lower()]
            self.sru_base_url = endpoint["url"]
            self.database = endpoint["database"]
            self.schema = endpoint["schema"]
            logger.info(f"Using preset endpoint: {preset} -> {self.sru_base_url}")
        else:
            self.sru_base_url = sru_base_url
            self.database = database
            self.schema = schema
        
        if not self.sru_base_url:
            logger.warning("No SRU base URL configured. Use preset or provide sru_base_url.")
    
    def search(self, term: str, search_type: str = "keyword") -> List[Dict[str, Any]]:
        """
        Search for records using SRU protocol.
        
        Args:
            term: Search term
            search_type: Type of search ("keyword", "title", "author", "subject", "isbn")
        
        Returns:
            List of parsed MARC records as dictionaries
        """
        if not self.sru_base_url:
            logger.error("No SRU base URL configured")
            return []
        
        # Build CQL query based on search type
        cql_query = self._build_cql_query(term, search_type)
        
        # Build SRU request URL
        params = {
            "version": "1.1",
            "operation": "searchRetrieve",
            "query": cql_query,
            "maximumRecords": str(self.max_records),
            "recordSchema": self.schema,
        }
        
        # Only add database param if not using DNB (they don't support x-database)
        if self.database and "dnb.de" not in self.sru_base_url:
            params["x-database"] = self.database
        
        url = f"{self.sru_base_url}?{urlencode(params)}"
        
        if self.debug:
            logger.debug(f"SRU request URL: {url}")
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            if self.debug:
                logger.debug(f"SRU response status: {response.status_code}")
                logger.debug(f"SRU response (first 1000 chars): {response.text[:1000]}")
            
            return self._parse_sru_response(response.text)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SRU request failed: {e}")
            return []
    
    def _build_cql_query(self, term: str, search_type: str) -> str:
        """Build CQL (Contextual Query Language) query string."""
        # Escape special CQL characters
        escaped_term = term.replace('"', '\\"')
        
        # Map search types to CQL indexes
        index_map = {
            "keyword": "cql.keywords",
            "title": "dc.title",
            "author": "dc.creator",
            "subject": "dc.subject",
            "isbn": "dc.identifier",
            "any": "cql.allRecords",
        }
        
        index = index_map.get(search_type, "cql.keywords")
        
        # DNB uses specific index names
        if "dnb.de" in self.sru_base_url:
            index_map_dnb = {
                "keyword": "dnb.woe",  # Wortsuche (word search)
                "title": "dnb.tit",
                "author": "dnb.atr",
                "subject": "dnb.swd",  # Schlagwort
                "isbn": "dnb.num",
            }
            index = index_map_dnb.get(search_type, "dnb.woe")
        
        # K10plus uses PICA index names (from SRU explain response) - Claude Generated
        elif "k10plus.de" in self.sru_base_url:
            index_map_k10plus = {
                "keyword": "pica.slw",  # Schlagwörter - for DK search we need subject keywords
                "title": "pica.tit",    # Title
                "author": "pica.per",   # Person
                "subject": "pica.slw",  # Schlagwörter (correct index for K10plus)
                "isbn": "pica.isb",     # ISBN
                "all": "pica.all",      # All fields
            }
            index = index_map_k10plus.get(search_type, "pica.slw")
        
        return f'{index}="{escaped_term}"'
    
    def _parse_sru_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse SRU response XML and extract MARC records."""
        records = []
        
        try:
            root = ET.fromstring(xml_text.encode('utf-8'))
            
            # Check for SRU diagnostics (errors)
            for diag in root.findall(".//srw:diagnostic", MARC_NS):
                message = diag.findtext("srw:message", namespaces=MARC_NS) or "Unknown error"
                logger.warning(f"SRU diagnostic: {message}")
            
            # Find all MARC records in response
            # SRU wraps records in <srw:record><srw:recordData>...</srw:recordData></srw:record>
            for record_data in root.findall(".//srw:recordData", MARC_NS):
                marc_record = record_data.find("marc:record", MARC_NS)
                if marc_record is not None:
                    parsed = self._parse_marc_record(marc_record)
                    if parsed:
                        records.append(parsed)
            
            # Also try without namespace prefix (some servers don't use it)
            if not records:
                for record_data in root.findall(".//{http://www.loc.gov/zing/srw/}recordData"):
                    marc_record = record_data.find("{http://www.loc.gov/MARC21/slim}record")
                    if marc_record is not None:
                        parsed = self._parse_marc_record(marc_record)
                        if parsed:
                            records.append(parsed)
            
            logger.info(f"Parsed {len(records)} MARC records from SRU response")
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse SRU XML response: {e}")
        
        return records
    
    def _parse_marc_record(self, record_elem) -> Optional[Dict[str, Any]]:
        """
        Parse a single MARC XML record element.
        
        Args:
            record_elem: XML Element representing a MARC record
        
        Returns:
            Dictionary with extracted bibliographic data
        """
        result = {
            "rsn": "",  # Control number (for compatibility with BiblioClient)
            "title": "",
            "author": [],
            "publication": "",
            "isbn": "",
            "classifications": [],
            "decimal_classifications": [],  # DK/DDC
            "rvk_classifications": [],  # RVK
            "subjects": [],
            "gnd_subjects": [],  # Subjects with GND IDs
            "abstract": "",
        }
        
        # Get namespace (may vary between responses)
        ns = {"marc": "http://www.loc.gov/MARC21/slim"}
        
        # Helper to find elements with or without namespace
        def find_all(tag):
            elements = record_elem.findall(f"marc:{tag}", ns)
            if not elements:
                elements = record_elem.findall(f"{{{ns['marc']}}}{tag}")
            if not elements:
                elements = record_elem.findall(tag)
            return elements
        
        def find_subfield(datafield, code):
            """Find subfield with given code."""
            for sf in datafield:
                if sf.get("code") == code:
                    return sf.text
            return None
        
        # Parse control fields
        for cf in find_all("controlfield"):
            tag = cf.get("tag")
            if tag == "001":
                result["rsn"] = cf.text or ""
        
        # Parse data fields
        for df in find_all("datafield"):
            tag = df.get("tag")
            
            # ISBN (020)
            if tag == "020":
                isbn = find_subfield(df, "a")
                if isbn:
                    result["isbn"] = isbn
            
            # DDC Classification (082)
            elif tag == "082":
                ddc = find_subfield(df, "a")
                if ddc:
                    result["classifications"].append(f"DDC {ddc}")
                    result["decimal_classifications"].append(ddc)
            
            # Other classification (084) - often RVK
            elif tag == "084":
                class_num = find_subfield(df, "a")
                class_source = find_subfield(df, "2")  # Source indicator
                if class_num:
                    if class_source and class_source.lower() == "rvk":
                        result["classifications"].append(f"RVK {class_num}")
                        result["rvk_classifications"].append(class_num)
                    elif class_source and class_source.lower() in ("ddc", "dewey"):
                        result["classifications"].append(f"DDC {class_num}")
                        result["decimal_classifications"].append(class_num)
                    else:
                        # Check if it looks like DK (numeric) or RVK (letter+number)
                        if re.match(r"^\d+(\.\d+)?$", class_num):
                            result["classifications"].append(f"DK {class_num}")
                            result["decimal_classifications"].append(class_num)
                        elif re.match(r"^[A-Z]{1,2}\s*\d+", class_num):
                            result["classifications"].append(f"RVK {class_num}")
                            result["rvk_classifications"].append(class_num)
                        else:
                            result["classifications"].append(class_num)
            
            # Main author (100)
            elif tag == "100":
                author = find_subfield(df, "a")
                if author:
                    result["author"].append(author)
            
            # Title (245)
            elif tag == "245":
                title_a = find_subfield(df, "a") or ""
                title_b = find_subfield(df, "b") or ""  # Subtitle
                result["title"] = f"{title_a} {title_b}".strip()
            
            # Publication (260/264)
            elif tag in ("260", "264"):
                place = find_subfield(df, "a") or ""
                publisher = find_subfield(df, "b") or ""
                date = find_subfield(df, "c") or ""
                result["publication"] = f"{place} {publisher} {date}".strip()
            
            # Abstract/Summary (520)
            elif tag == "520":
                abstract = find_subfield(df, "a")
                if abstract:
                    result["abstract"] = abstract
            
            # Subject headings (650)
            elif tag == "650":
                subject = find_subfield(df, "a")
                gnd_id = find_subfield(df, "0")  # Authority control number
                source = find_subfield(df, "2")  # Source of heading
                
                if subject:
                    result["subjects"].append(subject)
                    
                    # Check if it's a GND subject
                    if gnd_id and ("gnd" in str(gnd_id).lower() or "(DE-588)" in str(gnd_id)):
                        # Extract GND ID from formats like "(DE-588)4123456-7"
                        gnd_match = re.search(r"\(DE-588\)(\d+-\d+|\d+)", gnd_id)
                        if gnd_match:
                            result["gnd_subjects"].append({
                                "term": subject,
                                "gnd_id": gnd_match.group(1),
                            })
                        else:
                            result["gnd_subjects"].append({
                                "term": subject,
                                "gnd_id": gnd_id,
                            })
                    elif source and source.lower() == "gnd":
                        result["gnd_subjects"].append({
                            "term": subject,
                            "gnd_id": gnd_id or "",
                        })
            
            # Added authors (700)
            elif tag == "700":
                author = find_subfield(df, "a")
                if author:
                    result["author"].append(author)
        
        return result if result["rsn"] or result["title"] else None
    
    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single record by its control number.
        
        Args:
            record_id: The record control number (from field 001)
        
        Returns:
            Parsed MARC record as dictionary, or None if not found
        """
        # Search by control number
        results = self.search(record_id, search_type="keyword")
        
        # Find exact match
        for record in results:
            if record.get("rsn") == record_id:
                return record
        
        return results[0] if results else None
    
    def extract_decimal_classifications(self, classifications: List[str]) -> List[str]:
        """
        Extract decimal classifications (DDC/DK) from classification list.
        Compatible with BiblioClient interface.
        
        Args:
            classifications: List of classification strings
        
        Returns:
            List of decimal classification numbers
        """
        decimal_classes = []
        for classification in classifications:
            # Look for DDC/DK patterns
            match = re.search(r"(?:DDC|DK)\s+(\d+(?:\.\d+)?)", classification, re.IGNORECASE)
            if match:
                decimal_classes.append(match.group(1))
            elif re.match(r"^\d+(?:\.\d+)?$", classification):
                decimal_classes.append(classification)
        
        return decimal_classes
    
    def extract_rvk_classifications(self, classifications: List[str]) -> List[str]:
        """
        Extract RVK classifications from classification list.
        Compatible with BiblioClient interface.
        
        Args:
            classifications: List of classification strings
        
        Returns:
            List of RVK classification codes
        """
        rvk_classes = []
        for classification in classifications:
            match = re.search(r"RVK\s+([A-Z]{1,2}\s*\d+[A-Z0-9\s]*)", classification, re.IGNORECASE)
            if match:
                rvk_classes.append(match.group(1).strip())
            elif re.match(r"^[A-Z]{1,2}\s*\d+", classification):
                rvk_classes.append(classification)
        
        return rvk_classes
    
    def search_subjects(
        self, 
        search_terms: List[str], 
        max_results: int = 20
    ) -> Dict[str, Dict[str, Any]]:
        """
        Search for subject terms and aggregate results.
        Compatible with BiblioClient.search_subjects interface.
        
        Args:
            search_terms: List of search terms
            max_results: Maximum results per term
        
        Returns:
            Dictionary mapping terms to subject/classification data
        """
        old_max = self.max_records
        self.max_records = max_results
        
        results = {}
        for term in search_terms:
            logger.info(f"Searching MARC catalog for: {term}")
            records = self.search(term, search_type="subject")
            
            term_results = {
                "subjects": {},
                "classifications": {},
                "records": records,
            }
            
            for record in records:
                # Aggregate subjects
                for subject in record.get("subjects", []):
                    if subject not in term_results["subjects"]:
                        term_results["subjects"][subject] = {"count": 0, "gnd_id": ""}
                    term_results["subjects"][subject]["count"] += 1
                
                # Aggregate GND subjects
                for gnd_subj in record.get("gnd_subjects", []):
                    subj_term = gnd_subj["term"]
                    if subj_term not in term_results["subjects"]:
                        term_results["subjects"][subj_term] = {"count": 0, "gnd_id": ""}
                    term_results["subjects"][subj_term]["count"] += 1
                    if gnd_subj.get("gnd_id"):
                        term_results["subjects"][subj_term]["gnd_id"] = gnd_subj["gnd_id"]
                
                # Aggregate classifications
                for classification in record.get("classifications", []):
                    if classification not in term_results["classifications"]:
                        term_results["classifications"][classification] = {
                            "count": 0,
                            "titles": [],
                        }
                    term_results["classifications"][classification]["count"] += 1
                    title = record.get("title", "")
                    if title and title not in term_results["classifications"][classification]["titles"]:
                        term_results["classifications"][classification]["titles"].append(title)
            
            results[term] = term_results
        
        self.max_records = old_max
        return results
    
    def extract_dk_classifications_for_keywords(
        self, 
        keywords: List[str], 
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Extract DK/DDC classifications for given keywords.
        Compatible with BiblioClient interface.
        
        Args:
            keywords: List of keywords to search (may include GND-ID suffix)
            max_results: Maximum results per keyword
        
        Returns:
            List of classification results with metadata
        """
        all_results = []
        
        for keyword in keywords:
            # Strip GND-ID suffix if present, e.g., "Wissenschaft (GND-ID: 4066562-8)" -> "Wissenschaft"
            search_term = re.sub(r'\s*\(GND-ID:\s*[^)]+\)\s*$', '', keyword).strip()
            
            logger.info(f"Extracting DK classifications for: {keyword} -> search term: '{search_term}'")
            
            # Search for keyword - use keyword search, not subject search
            # Subject search uses specific indexes that may not work with all SRU endpoints
            old_max = self.max_records
            self.max_records = max_results
            records = self.search(search_term, search_type="keyword")
            self.max_records = old_max
            
            # Aggregate classifications by code
            classification_counts = {}
            
            for record in records:
                for dk in record.get("decimal_classifications", []):
                    if dk not in classification_counts:
                        classification_counts[dk] = {
                            "dk": dk,
                            "classification_type": "DK",
                            "count": 0,
                            "titles": [],
                            "keywords": [keyword],
                        }
                    classification_counts[dk]["count"] += 1
                    title = record.get("title", "")
                    if title and title not in classification_counts[dk]["titles"]:
                        classification_counts[dk]["titles"].append(title)
                
                for rvk in record.get("rvk_classifications", []):
                    if rvk not in classification_counts:
                        classification_counts[rvk] = {
                            "dk": rvk,
                            "classification_type": "RVK",
                            "count": 0,
                            "titles": [],
                            "keywords": [keyword],
                        }
                    classification_counts[rvk]["count"] += 1
                    title = record.get("title", "")
                    if title and title not in classification_counts[rvk]["titles"]:
                        classification_counts[rvk]["titles"].append(title)
            
            all_results.extend(classification_counts.values())
        
        # Sort by count descending
        all_results.sort(key=lambda x: x["count"], reverse=True)
        
        return all_results
    
    def parse_marc_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a local MARC XML file.
        
        Args:
            file_path: Path to MARC XML file
        
        Returns:
            List of parsed MARC records
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            records = []
            
            # Find all MARC records
            for record in root.findall(".//{http://www.loc.gov/MARC21/slim}record"):
                parsed = self._parse_marc_record(record)
                if parsed:
                    records.append(parsed)
            
            # Try without namespace
            if not records:
                for record in root.findall(".//record"):
                    parsed = self._parse_marc_record(record)
                    if parsed:
                        records.append(parsed)
            
            logger.info(f"Parsed {len(records)} records from {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Failed to parse MARC file {file_path}: {e}")
            return []


def main():
    """CLI for testing MARC XML client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MARC XML Client - Search library catalogs")
    parser.add_argument("search_term", help="Search term")
    parser.add_argument("--preset", choices=["dnb", "loc", "gbv", "swb", "k10plus"], 
                        default="dnb", help="Catalog preset (default: dnb)")
    parser.add_argument("--url", help="Custom SRU base URL")
    parser.add_argument("--type", choices=["keyword", "title", "author", "subject", "isbn"],
                        default="keyword", help="Search type")
    parser.add_argument("--max", type=int, default=10, help="Maximum results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize client
    if args.url:
        client = MarcXmlClient(sru_base_url=args.url, debug=args.debug, max_records=args.max)
    else:
        client = MarcXmlClient(preset=args.preset, debug=args.debug, max_records=args.max)
    
    # Search
    print(f"\n🔍 Searching {args.preset or args.url} for: {args.search_term}")
    print("=" * 60)
    
    results = client.search(args.search_term, search_type=args.type)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} records:\n")
    
    for i, record in enumerate(results[:args.max], 1):
        print(f"📖 [{i}] {record.get('title', 'No title')}")
        print(f"   RSN: {record.get('rsn', 'N/A')}")
        print(f"   Author: {', '.join(record.get('author', []))}")
        print(f"   Publication: {record.get('publication', 'N/A')}")
        print(f"   ISBN: {record.get('isbn', 'N/A')}")
        
        if record.get("classifications"):
            print(f"   📊 Classifications: {', '.join(record['classifications'])}")
        
        if record.get("subjects"):
            print(f"   🏷️  Subjects: {', '.join(record['subjects'][:5])}")
        
        if record.get("gnd_subjects"):
            gnd_strs = [f"{s['term']} ({s['gnd_id']})" for s in record['gnd_subjects'][:3]]
            print(f"   🔗 GND Subjects: {', '.join(gnd_strs)}")
        
        print()


if __name__ == "__main__":
    main()
