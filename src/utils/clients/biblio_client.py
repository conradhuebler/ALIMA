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

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("biblio_extractor")


class BiblioClient:
    """
    A tool to extract keywords and decimal classifications from a library catalog.
    """

    SEARCH_URL = "https://libero.ub.tu-freiberg.de:443/libero/LiberoWebServices.CatalogueSearcher.cls"
    DETAILS_URL = (
        "https://libero.ub.tu-freiberg.de:443/libero/LiberoWebServices.LibraryAPI.cls"
    )
    # Web catalog URLs for fallback - Claude Generated
    WEB_SEARCH_URL = "https://katalog.ub.tu-freiberg.de/Search/Results"
    WEB_RECORD_BASE_URL = "https://katalog.ub.tu-freiberg.de/Record/"

    # MAB-Tags für Schlagwörter
    MAB_SUBJECT_TAGS = ["0902", "0907", "0912", "0917", "0922", "0927"]

    def __init__(self, token: str = "", debug: bool = False, save_xml_path: str = "", enable_web_fallback: bool = True):
        """
        Initialize the extractor with the given token.

        Args:
            token: The authentication token for the library API
            debug: Enable detailed debug output
            save_xml_path: Directory to save raw XML responses for debugging (empty string = disabled)
            enable_web_fallback: Enable web scraping fallback when SOAP fails (Claude Generated)
        """
        self.token = token
        self.debug = debug
        self.save_xml_path = save_xml_path
        self.enable_web_fallback = enable_web_fallback  # Claude Generated - Web fallback toggle
        self._using_web_mode = False  # Claude Generated - Track if we switched to web pipeline
        self.session = requests.Session()
        self.headers = {"Content-Type": "text/xml;charset=UTF-8", "SOAPAction": ""}

    def search(self, term: str, search_type: str = "ku") -> List[Dict[str, Any]]:
        """
        Search for items in the catalog.

        Args:
            term: The search term
            search_type: The type of search (default: "ku" for anyword)

        Returns:
            A list of search result items
        """
        search_envelope = f"""
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:lib="http://libero.com.au">
           <soapenv:Header/>
           <soapenv:Body>
              <lib:Search>
                 <lib:term>{term}</lib:term>
                 <lib:use>{search_type}</lib:use>
              </lib:Search>
           </soapenv:Body>
        </soapenv:Envelope>
        """

        logger.debug(f"Sending search request to {self.SEARCH_URL}")
        logger.debug(f"Search envelope: {search_envelope}")

        try:
            response = self.session.post(
                self.SEARCH_URL, headers=self.headers, data=search_envelope, timeout=300
            )

            logger.debug(f"Search response status: {response.status_code}")

            if self.debug:
                logger.debug(f"Search response content: {response.text}")

            if response.status_code != 200:
                logger.error(
                    f"Error searching: {response.status_code} - {response.text}"
                )
                return []

            # Check for SOAP Fault response - Claude Generated
            soap_fault = self._extract_soap_fault(response.text)
            if soap_fault:
                logger.error(f"SOAP Fault during search for '{term}': {soap_fault}")

                # Try web scraping fallback immediately - Claude Generated
                if self.enable_web_fallback:
                    logger.info(f"Attempting web scraping fallback for search term: {term}")
                    self._using_web_mode = True  # Claude Generated - Switch to web mode permanently
                    logger.info("🔄 Switched to web pipeline mode for all subsequent requests")
                    web_results = self._search_web(term)
                    if web_results:
                        logger.info(f"✅ Web fallback search successful: {len(web_results)} results")
                        return web_results
                    else:
                        logger.warning(f"⚠️ Web fallback also returned no results for '{term}'")

                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []

        # Parse the response XML
        try:
            root = ET.fromstring(response.content)

            # Extract result items
            result_items = []

            # Try different possible paths
            paths = [
                ".//searchResultItems",
                ".//SearchResult/searchResultItems",
                ".//SearchResponse/SearchResult/searchResultItems",
                ".//{http://libero.com.au}SearchResponse/{http://libero.com.au}SearchResult/{http://libero.com.au}searchResultItems",
                # Claude Generated - Additional path for namespace-aware parsing
                ".//{http://libero.com.au}searchResultItems",
            ]

            items_found = False
            for path in paths:
                logger.debug(f"Trying to find items with path: {path}")
                items = root.findall(path)
                if items:
                    items_found = True
                    logger.info(f"Found {len(items)} items with path {path}")
                    break

            if not items_found:
                # Dump the XML structure for debugging
                logger.debug("XML structure:")
                self._print_xml_structure(root)
                logger.warning(
                    "Could not find search result items in any expected path"
                )
                return []

            for item in items:
                result_item = {}
                for child in item:
                    tag = child.tag
                    # Remove namespace if present
                    if "}" in tag:
                        tag = tag.split("}")[1]
                    text = child.text if child.text else ""
                    result_item[tag] = text
                result_items.append(result_item)

            logger.info(f"Parsed {len(result_items)} result items")
            if self.debug and result_items:
                logger.debug(f"First result item: {result_items[0]}")

            return result_items

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            logger.error(f"Response content: {response.text}")
            return []

    def _print_xml_structure(self, element, level=0):
        """Print the XML structure for debugging purposes."""
        indent = "  " * level
        if hasattr(element, "tag"):
            logger.debug(f"{indent}{element.tag}")
            for child in element:
                self._print_xml_structure(child, level + 1)

    def get_title_details(self, rsn: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific title.

        Args:
            rsn: The RSN (Record Serial Number) of the title

        Returns:
            A dictionary with title details or None if an error occurred
        """
        # If web mode is active, use web scraping directly - Claude Generated
        if self._using_web_mode:
            logger.debug(f"Web mode active: Using web scraping for RSN {rsn}")
            return self._get_title_details_web(rsn)

        details_envelope = f"""
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:lib="http://libero.com.au">
           <soapenv:Header/>
           <soapenv:Body>
              <lib:GetTitleDetails>
                 <lib:TOKEN>{self.token}</lib:TOKEN>
                 <lib:RSN>{rsn}</lib:RSN>
              </lib:GetTitleDetails>
           </soapenv:Body>
        </soapenv:Envelope>
        """

        logger.debug(f"Getting details for RSN: {rsn}")
        logger.debug(f"Details envelope: {details_envelope}")

        try:
            response = self.session.post(
                self.DETAILS_URL,
                headers=self.headers,
                data=details_envelope,
                timeout=300,
            )

            logger.debug(f"Details response status: {response.status_code}")

            if self.debug:
                logger.debug(f"Details response content: {response.text}")

            # Save raw XML response for debugging - Claude Generated
            if self.save_xml_path:
                self._save_xml_response(rsn, response.text)

            if response.status_code != 200:
                logger.error(
                    f"Error getting details: {response.status_code} - {response.text}"
                )

                # Try web fallback for 400 Bad Request (Web Record ID passed to SOAP) - Claude Generated
                if response.status_code == 400 and self.enable_web_fallback:
                    logger.info(
                        f"HTTP 400 for RSN {rsn}: Likely web record ID, trying web fallback"
                    )
                    fallback_details = self._get_title_details_web(rsn)
                    if fallback_details:
                        logger.info(f"✅ Web fallback successful for RSN {rsn}")
                        return fallback_details

                return None

            # Check for SOAP Fault response - Claude Generated
            soap_fault = self._extract_soap_fault(response.text)
            if soap_fault:
                logger.error(f"SOAP Fault for RSN {rsn}: {soap_fault}")

                # Try web scraping fallback - Claude Generated
                if self.enable_web_fallback:
                    logger.info(
                        f"Attempting web scraping fallback for RSN {rsn}"
                    )
                    fallback_details = self._get_title_details_web(rsn)
                    if fallback_details:
                        logger.info(
                            f"✅ Web scraping fallback successful for RSN {rsn}"
                        )
                        return fallback_details
                    else:
                        logger.warning(
                            f"⚠️ Web scraping fallback also failed for RSN {rsn}"
                        )

                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

        # Parse the response XML
        try:
            root = ET.fromstring(response.content)

            # If debug is enabled, print the structure
            if self.debug:
                logger.debug("Details XML structure:")
                self._print_xml_structure(root)
                logger.debug(f"Raw XML response: {response.text}")

            # Extract details
            details = {}

            # Extract MAB-based subjects from the MAB tags
            mab_subjects = self._extract_mab_subjects(root)
            if mab_subjects:
                logger.debug(f"Found {len(mab_subjects)} MAB subjects: {mab_subjects}")
                details["mab_subjects"] = mab_subjects
            else:
                logger.debug("No MAB subjects found in XML")
                details["mab_subjects"] = []

            # Early diagnostic: check if root has any content - Claude Generated
            if len(root) == 0:
                logger.warning(f"⚠️ XML root has no children - possible empty SOAP response for RSN {rsn}")
            else:
                logger.debug(f"XML root has {len(root)} children for RSN {rsn}")

            # Register namespaces
            namespaces = {
                "soap": "http://schemas.xmlsoap.org/soap/envelope/",
                "lib": "http://libero.com.au",
            }

            # Try different possible paths for classifications - ENHANCED with additional paths
            classification_paths = [
                ".//Classification/Classifications/Classification",
                ".//{http://libero.com.au}Classification/{http://libero.com.au}Classifications/{http://libero.com.au}Classification",
                ".//Classifications/Classification",  # Shorter path variant
                ".//{http://libero.com.au}Classifications/{http://libero.com.au}Classification",  # Namespace variant
                ".//DK",  # Direct DK tag
                ".//{http://libero.com.au}DK",  # Namespaced DK tag
                ".//Dewey",  # Alternative name
                ".//{http://libero.com.au}Dewey",  # Namespaced Dewey
            ]

            classifications = []
            for path in classification_paths:
                logger.debug(f"Trying classification path: {path}")
                for classification in root.findall(path):
                    if classification.text:
                        classifications.append(classification.text)
                        logger.debug(f"Found classification: {classification.text}")

                if classifications:
                    logger.info(f"✅ Found {len(classifications)} classifications using path: {path}")
                    break

            # Log if no classifications found to help debugging
            if not classifications and self.debug:
                logger.warning(f"❌ No classifications found for RSN {rsn} using any of {len(classification_paths)} paths")

            details["classifications"] = classifications

            # Try different possible paths for subjects
            subject_paths = [
                ".//Subject/Subjects/Subject",
                ".//{http://libero.com.au}Subject/{http://libero.com.au}Subjects/{http://libero.com.au}Subject",
            ]

            subjects = []
            for path in subject_paths:
                logger.debug(f"Trying subject path: {path}")
                for subject in root.findall(path):
                    if subject.text:
                        subjects.append(subject.text)
                        logger.debug(f"Found subject: {subject.text}")

                if subjects:
                    break

            if not subjects:
                logger.debug("No regular subjects found")
            details["subjects"] = subjects

            # Extract basic information using multiple paths
            title_paths = [".//Title", ".//{http://libero.com.au}Title"]
            details["title"] = self._extract_text_multiple_paths(root, title_paths)
            if not details["title"]:
                logger.warning(f"⚠️ No title found for RSN {rsn} - checked paths: {title_paths}")

            details["author"] = self._extract_authors(root)
            if not details["author"]:
                logger.debug(f"No authors found for RSN {rsn}")

            publication_paths = [
                ".//Publication",
                ".//{http://libero.com.au}Publication",
            ]
            details["publication"] = self._extract_text_multiple_paths(
                root, publication_paths
            )
            if not details["publication"]:
                logger.debug(f"No publication found for RSN {rsn}")

            isbn_paths = [".//ISBN", ".//{http://libero.com.au}ISBN"]
            details["isbn"] = self._extract_text_multiple_paths(root, isbn_paths)
            if not details["isbn"]:
                logger.debug(f"No ISBN found for RSN {rsn}")

            details["rsn"] = rsn

            # ADDED: Try to extract DK numbers from title text if classifications field is empty - Claude Generated
            if not details.get("classifications"):
                title_text = details.get("title", "")
                dk_from_title = self._extract_dk_from_text(title_text)
                if dk_from_title:
                    logger.info(f"Extracted DK numbers from title text for RSN {rsn}: {dk_from_title}")
                    details["classifications"] = dk_from_title
                    details["_source"] = "extracted_from_title"  # Mark as fallback source

            # Diagnostic: log extraction summary - Claude Generated
            extracted_fields = sum([
                bool(details.get("title")),
                bool(details.get("author")),
                bool(details.get("publication")),
                bool(details.get("isbn")),
                bool(details.get("classifications")),
                bool(details.get("subjects")),
                bool(details.get("mab_subjects"))
            ])
            if extracted_fields == 0:
                logger.warning(f"⚠️ No data extracted for RSN {rsn} - SOAP response may be empty or malformed")
            elif extracted_fields < 3:
                logger.info(f"⚠️ Partial data for RSN {rsn}: {extracted_fields}/7 fields populated")

            return details

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            if hasattr(response, "text"):
                logger.error(f"Response content: {response.text}")
            return None

    def _search_web(self, term: str, search_type: str = "AllFields") -> List[Dict[str, Any]]:
        """
        Search web catalog when SOAP search fails - Complete fallback - Claude Generated

        Args:
            term: Search term (keyword)
            search_type: Search type (default: AllFields)

        Returns:
            List of search results with rsn (web record ID) and title
        """
        try:
            from bs4 import BeautifulSoup

            logger.info(f"Web search fallback for term: {term}")

            # Build search parameters - Claude Generated
            params = {
                "hiddenFilters[]": [
                    'institution:"DE-105"',
                    '-format:"Article"',
                    '-format:"ElectronicArticle"',
                ],
                "join": "AND",
                "bool0[]": "AND",
                "lookfor0[]": term,
                "type0[]": search_type,
                "filter[]": 'facet_avail:"Local"',
                "limit": 50,  # Get up to 50 results
            }

            # Make request - Claude Generated
            response = self.session.get(self.WEB_SEARCH_URL, params=params, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Web search: HTTP {response.status_code} for '{term}'")
                return []

            # Parse HTML - Claude Generated
            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            # Extract record IDs and titles - Claude Generated
            for title_link in soup.find_all("a", class_="title getFull"):
                try:
                    record_id = title_link.get("id", "").split("|")[-1]
                    title_text = title_link.text.strip()

                    if record_id and title_text:
                        results.append({"rsn": record_id, "title": title_text})
                except Exception as e:
                    logger.debug(f"Error extracting record: {e}")
                    continue

            logger.info(f"Web search found {len(results)} results for '{term}'")
            return results

        except Exception as e:
            logger.error(f"Web search failed for '{term}': {e}")
            return []

    def _get_title_details_web(self, rsn: str) -> Optional[Dict[str, Any]]:
        """
        Get title details via web scraping fallback when SOAP fails - Claude Generated

        Args:
            rsn: Record Serial Number

        Returns:
            Dictionary with title details (same format as SOAP response) or None
        """
        try:
            from bs4 import BeautifulSoup

            # Fetch record page - Claude Generated
            url = f"{self.WEB_RECORD_BASE_URL}{rsn}"
            logger.debug(f"Web fallback: Fetching {url}")

            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Web fallback: HTTP {response.status_code} for RSN {rsn}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            details = {"rsn": rsn}

            # Extract title - Claude Generated
            title_element = soup.find("h1", attrs={"property": "name"})
            details["title"] = title_element.text.strip() if title_element else ""

            # Extract authors - Claude Generated
            authors = []
            for author_link in soup.find_all("a", href=re.compile(r"search.*author")):
                author_text = author_link.text.strip()
                if author_text and author_text not in authors:
                    authors.append(author_text)
            details["author"] = authors

            # Extract DK classifications - Claude Generated
            classifications = []
            dk_links = soup.find_all("a", href=re.compile(r"lookfor=DK.*?&type=udk_raw_de105"))
            for dk_link in dk_links:
                dk_match = re.search(r"DK\s+([\d\.:]+)", dk_link.text)
                if dk_match:
                    dk_number = dk_match.group(1)
                    classification_str = f"DK {dk_number}"
                    if classification_str not in classifications:
                        classifications.append(classification_str)

            # Extract Q/RVK classifications - Claude Generated
            q_links = soup.find_all("a", href=re.compile(r"lookfor=Q[A-Z]?\s*\d+.*?&type=udk_raw_de105"))
            for q_link in q_links:
                q_match = re.search(r"Q[A-Z]?\s*[\d\s]+", q_link.text)
                if q_match:
                    q_number = q_match.group().strip()
                    classification_str = f"Q {q_number}"
                    if classification_str not in classifications:
                        classifications.append(classification_str)

            details["classifications"] = classifications

            # Extract publication, ISBN, subjects - Claude Generated (minimal from HTML)
            details["publication"] = ""
            details["isbn"] = ""
            details["subjects"] = []
            details["mab_subjects"] = []

            logger.debug(
                f"Web fallback for RSN {rsn}: title='{details['title']}', "
                f"classifications={details['classifications']}"
            )

            return details if details.get("title") or details.get("classifications") else None

        except Exception as e:
            logger.warning(f"Web fallback failed for RSN {rsn}: {e}")
            return None

    def _extract_soap_fault(self, response_text: str) -> Optional[str]:
        """
        Extract SOAP Fault message from response if present - Claude Generated

        Args:
            response_text: Raw response text from SOAP server

        Returns:
            Fault message string if SOAP Fault found, None otherwise
        """
        import xml.etree.ElementTree as ET

        try:
            # Try multiple common fault string patterns
            fault_patterns = [
                r'<soap:Fault>.*?<faultstring>(.*?)</faultstring>',
                r'<soap:faultstring>(.*?)</soap:faultstring>',
                r'<faultstring>(.*?)</faultstring>',
                r'<ns2:faultstring>(.*?)</ns2:faultstring>',
            ]

            for pattern in fault_patterns:
                match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()

            # Also try XML parsing
            if '<soap:Fault>' in response_text or '<Fault>' in response_text:
                root = ET.fromstring(response_text)
                # Try different namespace variants
                namespaces = {
                    'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
                    'ns': 'http://schemas.xmlsoap.org/soap/envelope/',
                }
                fault_string = root.find('.//faultstring')
                if fault_string is not None and fault_string.text:
                    return fault_string.text
                # Try with namespace
                for ns_prefix, ns_uri in namespaces.items():
                    fault_string = root.find(f'.//{{{ns_uri}}}faultstring')
                    if fault_string is not None and fault_string.text:
                        return fault_string.text

        except Exception as e:
            logger.debug(f"Could not parse SOAP Fault: {e}")

        return None

    def _save_xml_response(self, rsn: str, xml_content: str) -> None:
        """
        Save raw XML response to file for debugging - Claude Generated

        Args:
            rsn: Record Serial Number (used in filename)
            xml_content: Raw XML response content
        """
        import os
        from pathlib import Path
        from datetime import datetime

        try:
            # Create directory if it doesn't exist
            save_dir = Path(self.save_xml_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rsn_{rsn}_{timestamp}.xml"
            filepath = save_dir / filename

            # Write XML to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(xml_content)

            logger.info(f"✅ Saved XML response to: {filepath}")

        except Exception as e:
            logger.warning(f"⚠️ Could not save XML response: {e}")

    def _extract_mab_subjects(self, root) -> List[str]:
        """
        Extrahiert Schlagwörter aus MAB-Daten im XML.

        Args:
            root: XML-Root-Element

        Returns:
            Liste der gefundenen Schlagwörter
        """
        mab_subjects = []

        # Versuche verschiedene Pfade für MAB-Einträge
        mab_paths = [".//MAB", ".//{http://libero.com.au}MAB"]

        for path in mab_paths:
            mab_elements = root.findall(path)

            for mab in mab_elements:
                # Prüfe, ob dieses MAB-Element ein Schlagwort enthält
                tag_key = None
                mab_data = None
                mab_data_plain = None

                for child in mab:
                    if (
                        child.tag.endswith("TagKey")
                        and child.text in self.MAB_SUBJECT_TAGS
                    ):
                        tag_key = child.text
                    elif child.tag.endswith("MABData"):
                        mab_data = child.text
                    elif child.tag.endswith("MABDataPlain"):
                        mab_data_plain = child.text

                if tag_key and (mab_data or mab_data_plain):
                    # Extrahieren des Schlagworts
                    subject = None
                    if mab_data_plain:
                        # Direkt aus MABDataPlain extrahieren
                        parts = mab_data_plain.strip().split()
                        if len(parts) > 1:
                            # Ignoriere die ersten Teil mit Zahlen und Leerzeichen
                            subject = " ".join(parts[1:]).strip()
                    elif mab_data:
                        # Base64-decodieren und dann extrahieren
                        try:
                            decoded = base64.b64decode(mab_data).decode("utf-8")
                            parts = decoded.strip().split()
                            if len(parts) > 1:
                                subject = " ".join(parts[1:]).strip()
                        except Exception as e:
                            logger.warning(
                                f"Fehler beim Decodieren von Base64-Daten: {e}"
                            )

                    if subject and subject not in mab_subjects:
                        logger.debug(f"Gefundenes MAB-Schlagwort: {subject}")
                        mab_subjects.append(subject)

        return mab_subjects

    def _extract_text_multiple_paths(self, root, paths: List[str]) -> str:
        """Helper method to extract text from an XML element using multiple possible paths"""
        for path in paths:
            element = root.find(path)
            if element is not None and element.text:
                logger.debug(f"Found text with path {path}: {element.text}")
                return element.text
        logger.debug(f"No text found for paths: {paths}")
        return ""

    def _extract_authors(self, root) -> List[str]:
        """Helper method to extract authors in display form"""
        authors = []

        # Try different possible paths
        author_paths = [
            ".//Author/Authors/AuthorDisplayForm",
            ".//{http://libero.com.au}Author/{http://libero.com.au}Authors/{http://libero.com.au}AuthorDisplayForm",
        ]

        for path in author_paths:
            logger.debug(f"Trying author path: {path}")
            for author in root.findall(path):
                if author.text:
                    authors.append(author.text)
                    logger.debug(f"Found author: {author.text}")

            if authors:
                break

        return authors

    def _extract_dk_from_text(self, text: str) -> List[str]:
        """
        Extract DK (Dewey Decimal) classification numbers from text using regex.
        Fallback method when DK numbers are not in XML structure but embedded in title.
        Claude Generated

        Args:
            text: Text to search for DK numbers (usually title)

        Returns:
            List of extracted DK numbers (e.g., ["543.42", "620.5"])
        """
        if not text:
            return []

        # Pattern 1: "DK 543.42" (space-separated)
        # Pattern 2: "[DK 543.42]" (in brackets)
        # Pattern 3: "(DK 543.42)" (in parentheses)
        patterns = [
            r'DK\s+(\d+(?:\.\d+)*)',  # DK 543 or DK 543.42
            r'\[DK\s+(\d+(?:\.\d+)*)\]',  # [DK 543.42]
            r'\(DK\s+(\d+(?:\.\d+)*)\)',  # (DK 543.42)
        ]

        extracted = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and match not in extracted:
                    extracted.append(match)
                    logger.debug(f"Found DK number in text: {match} (from pattern: {pattern})")

        return extracted

    def extract_decimal_classifications(self, classifications: List[str]) -> List[str]:
        """
        Extract decimal classifications from a list of classifications.
        Excludes incomplete classifications with additional suffixes or geographic codes.

        Args:
            classifications: List of classification strings

        Returns:
            List of extracted decimal classifications
        """
        decimal_classes = []
        for classification in classifications:
            logger.debug(f"Processing classification: {classification}")

            # Look for patterns like "DK 543.42" but exclude geographic/suffix variants
            if "DK " in classification:
                # Extract the number after "DK " but check for suffixes
                match = re.search(r"DK\s+(\d+(?:\.\d+)?)", classification)
                if match:
                    decimal_class = match.group(1)
                    
                    # Check if there are unwanted suffixes after the decimal number
                    full_match = re.search(r"DK\s+(\d+(?:\.\d+)?)([A-Z]{2,}|/[A-Z]{2,}|\([^)]+\)|/.+)", classification)
                    
                    if full_match:
                        # This has geographic codes, country codes, or parenthetical info - skip it
                        logger.debug(f"Skipping DK classification with suffix: {classification}")
                        continue
                    else:
                        # Clean DK classification without suffixes
                        decimal_classes.append(decimal_class)
                        logger.debug(f"Extracted clean DK classification: {decimal_class}")
                        
            elif "/" in classification and re.match(r"^\d+", classification.strip()):
                # Handle patterns like "543/.62" but avoid geographic suffixes
                # Only process if it starts with a digit (DK pattern)
                if "/./" in classification:
                    # Skip complex patterns like "546.3/.9"
                    logger.debug(f"Skipping complex slash classification: {classification}")
                    continue
                    
                parts = classification.split("/")
                if (len(parts) == 2 and 
                    parts[0].strip().replace(".", "").isdigit() and 
                    parts[1].strip().startswith(".") and
                    not re.search(r"[A-Z]{2,}", parts[1])):  # No country codes after decimal
                    
                    decimal_class = f"{parts[0].strip()}{parts[1].strip()}"
                    decimal_classes.append(decimal_class)
                    logger.debug(f"Extracted decimal classification: {decimal_class}")
                else:
                    logger.debug(f"Skipping slash classification with suffix: {classification}")

        return decimal_classes

    def extract_rvk_classifications(self, classifications: List[str]) -> List[str]:
        """
        Extract RVK classifications from a list of classifications.
        RVK classifications always contain both letters and numbers.

        Args:
            classifications: List of classification strings

        Returns:
            List of extracted RVK classifications
        """
        rvk_classes = []
        for classification in classifications:
            logger.debug(f"Processing RVK classification: {classification}")

            # Look for RVK patterns like "RVK Q*" or direct Q codes
            if "RVK " in classification:
                # Extract RVK code after "RVK "
                match = re.search(r"RVK\s+([A-Z]+[0-9]+[A-Z0-9\s]*)", classification)
                if match:
                    rvk_class = match.group(1).strip()
                    rvk_classes.append(rvk_class)
                    logger.debug(f"Extracted RVK classification: {rvk_class}")
            else:
                # Direct RVK codes - must contain both letters and numbers
                classification = classification.strip()
                
                # Skip pure letter codes like 'PN', 'RN'
                if re.match(r"^[A-Z]+$", classification):
                    logger.debug(f"Skipping pure letter code: {classification}")
                    continue
                    
                # Skip pure number codes like 'SCI013080', 'SCI026000'
                if re.match(r"^[A-Z]*\d+$", classification):
                    logger.debug(f"Skipping pure number/letter-number code: {classification}")
                    continue
                    
                # Valid RVK: starts with letters, contains numbers, may have more letters/numbers
                if re.match(r"^[A-Z]+\d+[A-Z0-9\s]*$", classification):
                    rvk_classes.append(classification)
                    logger.debug(f"Extracted direct RVK classification: {classification}")
                else:
                    logger.debug(f"Skipping invalid RVK pattern: {classification}")

        return rvk_classes

    def process_search_results(
        self, results: List[Dict[str, Any]], max_items: int = 100, delay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Process search results by getting details for each item.

        Args:
            results: List of search result items
            max_items: Maximum number of items to process
            delay: Delay between requests in seconds

        Returns:
            List of processed items with details
        """
        # Log if web mode is active - Claude Generated
        if self._using_web_mode:
            logger.info(f"Processing {len(results)} search results in WEB MODE (no SOAP calls)")

        processed_items = []

        logger.info(f"Processing {min(len(results), max_items)} items...")

        for i, item in enumerate(results[:max_items]):
            if i > 0:
                time.sleep(delay)  # Be nice to the server

            title = item.get("title", "Unknown")
            logger.info(
                f"Processing item {i+1}/{min(len(results), max_items)}: {title}"
            )

            rsn = item.get("rsn")
            if not rsn:
                logger.warning(f"No RSN found for item: {title}")
                continue

            details = self.get_title_details(rsn)
            if not details:
                logger.warning(f"Could not get details for RSN: {rsn}")
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

            # Log merged result with field counts for diagnostic
            title_merged = merged_details.get("title", "")
            classifications_count = len(merged_details.get("classifications", []))
            subjects_count = len(merged_details.get("subjects", []))
            authors_count = len(merged_details.get("author", []))
            logger.info(f"Merged details for RSN {rsn}: title='{title_merged}', classifications={classifications_count}, subjects={subjects_count}, authors={authors_count}")

            # Add decimal classifications
            merged_details["decimal_classifications"] = self.extract_decimal_classifications(
                merged_details.get("classifications", [])
            )

            # Add RVK classifications - Claude Generated
            merged_details["rvk_classifications"] = self.extract_rvk_classifications(
                merged_details.get("classifications", [])
            )

            processed_items.append(merged_details)

        return processed_items

    def search_subjects(self, search_terms: List[str], max_results: int = DEFAULT_DK_MAX_RESULTS) -> Dict[str, Dict[str, Any]]:
        """
        Claude Generated - Search catalog for subjects and return in suggester format.
        
        Args:
            search_terms: List of search terms to look for
            max_results: Maximum results to process per term
            
        Returns:
            Dictionary with structure:
            {
                search_term: {
                    subject: {
                        "count": int,
                        "gndid": set(),
                        "ddc": set(),
                        "dk": set()
                    }
                }
            }
        """
        results = {}
        
        for search_term in search_terms:
            logger.info(f"Searching catalog subjects for: {search_term}")

            # Search catalog for this term
            search_results = self.search(search_term)

            # Web fallback for search_subjects if SOAP fails - Claude Generated
            if not search_results and self.enable_web_fallback:
                logger.info(f"SOAP search failed for subjects '{search_term}', trying web fallback")
                search_results = self._search_web(search_term)

            #logger.info(f"Found {(search_results)} results for '{search_term}'")
            if not search_results:
                results[search_term] = {}
                continue
                
            # Limit search results to prevent excessive processing - Claude Generated
            if len(search_results) > max_results * 2:
                logger.info(f"Limiting search results for '{search_term}': {len(search_results)} -> {max_results * 2}")
                search_results = search_results[:max_results * 2]
                
            # Process results to extract subjects
            processed_items = self.process_search_results(search_results, max_items=max_results)
            #logger.info(f"Processed {(processed_items)} items for '{search_term}'")
            # CLAUDE TODO -> an diesem Punkte haben wir also die MABs
            # Convert to suggester format
            term_subjects = {}
            
            logger.info(f"Processing {len(processed_items)} processed items for '{search_term}'")
            
            for i, item in enumerate(processed_items):
                # Debug: Show what's in each item
                logger.debug(f"Item {i+1} keys: {list(item.keys())}")
                
                # Extract subjects from both regular subjects and MAB subjects
                subjects = item.get("subjects", []) + item.get("mab_subjects", [])
                
                # Enhanced debug logging
                logger.info(f"Item {i+1}/{len(processed_items)}: '{item.get('title', 'Unknown')}' - Subjects: {len(subjects)}")
                if subjects:
                    logger.info(f"  Regular subjects: {item.get('subjects', [])}")
                    logger.info(f"  MAB subjects: {item.get('mab_subjects', [])}")
                else:
                    logger.warning(f"  No subjects found - available keys: {list(item.keys())}")
                
                # Get classifications
                classifications = item.get("decimal_classifications", [])
                ddc_set = set()
                dk_set = set() #  TODO -thing about, how to realise classification extraction - search_subjects might the wrong pleace at all )  # DK classifications are already extracted
                
                # Process each subject
                for subject in subjects:
                    subject = subject.strip()
                    if not subject:
                        continue
                        
                    if subject not in term_subjects:
                        term_subjects[subject] = {
                            "count": 1,
                            "gndid": set(),  # Will be filled by SWB validation later
                            "ddc": ddc_set.copy(),
                            "dk": dk_set.copy()
                        }
                    else:
                        term_subjects[subject]["count"] += 1
                        term_subjects[subject]["ddc"].update(ddc_set)
                        term_subjects[subject]["dk"].update(dk_set)
            
            # Limit subjects per term to prevent excessive results - Claude Generated
            if len(term_subjects) > 50:
                logger.info(f"Limiting subjects for '{search_term}': {len(term_subjects)} -> 50 (top by count)")
                # Sort by count and take top 50
                sorted_subjects = sorted(term_subjects.items(), key=lambda x: x[1]["count"], reverse=True)
                term_subjects = dict(sorted_subjects[:50])
            
            results[search_term] = term_subjects
            logger.info(f"Found {len(term_subjects)} subjects for '{search_term}'")
        logger.info(f"Results keys: {results.keys()}")

        return results

    def extract_dk_classifications_for_keywords(self, keywords: List[str], max_results: int = 50, force_update: bool = False) -> List[Dict[str, Any]]:
        """
        Claude Generated - Extract DK classifications for given GND keywords using intelligent caching.

        Args:
            keywords: List of GND keywords to search for
            max_results: Maximum search results to process per keyword
            force_update: If True, bypass cache and perform live search with title merging

        Returns:
            List of DK classification results with metadata
        """
        # Use new caching system - Claude Generated
        from ...core.unified_knowledge_manager import UnifiedKnowledgeManager

        dk_cache = UnifiedKnowledgeManager()
        cached_results = []  # FIX: Initialize to merge with live results later - Claude Generated

        # Track source (cache vs live) for each keyword - Claude Generated - Keyword-centric restructuring
        keyword_sources = {}  # keyword -> "cache" or "live"
        keyword_timings = {}  # keyword -> time_ms

        # Skip cache check if force_update is enabled - Claude Generated
        # FIXED: Removed force_update bypass - always use cache first - Claude Generated
        # Log force_update status but don't skip cache
        if force_update:
            logger.warning(f"⚠️ Force update flag set (but cache is still checked first)")

        # First try to get results from cache with timing - Claude Generated
        import time
        cache_start = time.time()
        logger.info(f"🔍 Checking cache for {len(keywords)} keywords: {keywords[:3]}{'...' if len(keywords) > 3 else ''}")
        cached_results = dk_cache.search_by_keywords(keywords, fuzzy_threshold=80)
        cache_time = (time.time() - cache_start) * 1000  # Convert to ms
        logger.info(f"📊 Cache returned {len(cached_results) if cached_results else 0} results in {cache_time:.1f}ms")

        # Ultra-Deep Diagnostic Logging - Claude Generated
        if cached_results:
            for i, result in enumerate(cached_results):
                titles_count = len(result.get("titles", []))
                matched_keywords_count = len(result.get("matched_keywords", []))
                logger.debug(f"   Cache[{i}]: dk={result.get('dk')}, count={result.get('count')}, titles={titles_count}, matched_keywords={matched_keywords_count}")
        else:
            logger.debug("   No results in cache")

        # FIXED: Normalize keywords for cache comparison - Claude Generated
        # BUG FIX: Input keywords have format "Keyword (GND-ID: xxx)" but cached keywords are normalized
        # Need to normalize both for proper comparison

        # Create mapping: normalized_keyword -> original_keyword
        normalized_to_original = {}
        for kw in keywords:
            normalized = kw.split('(')[0].strip().lower()
            if normalized not in normalized_to_original:
                normalized_to_original[normalized] = kw

        # FIXED: Collect cached keywords WITH quality check (must have titles) - Claude Generated
        # Only mark keywords as cached if they have BOTH keyword AND non-empty titles
        # Empty cached results should trigger live search to get proper data
        cached_normalized_with_titles = set()
        if cached_results:
            for result in cached_results:
                # Cached keywords are already normalized (lowercase, no GND-ID)
                matched_kws = result.get("matched_keywords", [])
                titles = result.get("titles", [])  # FIXED: Check for titles

                # Only mark as cached if it has BOTH keyword AND non-empty titles
                if titles:  # FIXED: Quality check - only if titles exist!
                    for kw in matched_kws:
                        normalized = kw.strip().lower()
                        cached_normalized_with_titles.add(normalized)

                        # Track source using ORIGINAL keyword format if available
                        if normalized in normalized_to_original:
                            original_kw = normalized_to_original[normalized]
                            keyword_sources[original_kw] = "cache"
                            keyword_timings[original_kw] = cache_time / len(matched_kws)
                else:
                    # Empty titles - log for debugging
                    if matched_kws:
                        logger.warning(f"⚠️ Cached keywords without titles: {matched_kws} (will trigger live search)")

        # FIXED: Find uncached keywords - includes both missing keywords AND keywords with empty titles
        uncached_keywords = [
            original_kw
            for normalized, original_kw in normalized_to_original.items()
            if normalized not in cached_normalized_with_titles
        ]

        if not uncached_keywords:
            # All keywords found in cache - Claude Generated
            logger.info(f"✅ CACHE HIT: All {len(keywords)} keywords found in cache!")
            logger.info(f"✅ CACHE STATISTICS: {len(cached_results)} cached DK classifications, Cache lookup {cache_time:.1f}ms")
            logger.info(f"   Cache hit rate: {len(keywords)}/{len(keywords)} keywords (100%)")
            return cached_results

        # FIXED: Partial cache - perform live search ONLY for uncached keywords
        logger.info(f"⚠️ {len(uncached_keywords)} of {len(keywords)} keywords not in cache: {uncached_keywords[:3]}{'...' if len(uncached_keywords) > 3 else ''}")
        logger.info(f"Performing live search for {len(uncached_keywords)} uncached keywords to augment {len(cached_results)} cached results")

        # Perform live catalog search ONLY for uncached keywords with timing
        logger.info(f"Performing live catalog search for {len(uncached_keywords)} keywords")
        live_search_start = time.time()
        dk_results = []

        for keyword in uncached_keywords:
            logger.info(f"Searching DK classifications for keyword: {keyword}")

            # Track timing per keyword for keyword-centric restructuring
            keyword_search_start = time.time()

            # Search catalog for this keyword
            search_results = self.search(keyword, search_type="ku")

            # Web fallback for search if SOAP fails - Claude Generated
            if not search_results and self.enable_web_fallback:
                logger.info(f"SOAP search failed for '{keyword}', trying web fallback")
                search_results = self._search_web(keyword)
                keyword_sources[keyword] = "web_fallback"
            else:
                keyword_sources[keyword] = "live"

            keyword_time = (time.time() - keyword_search_start) * 1000
            keyword_timings[keyword] = keyword_time

            if not search_results:
                continue
                
            # Process search results
            processed_items = self.process_search_results(search_results, max_items=max_results)
            
            # Extract DK and RVK classifications from each item
            for item in processed_items:
                # Ensure classifications are lists, not None
                dk_classifications = item.get("decimal_classifications") or []
                rvk_classifications = item.get("rvk_classifications") or []

                # Filter out None/empty values from classifications
                dk_classifications = [dk for dk in dk_classifications if dk]
                rvk_classifications = [rvk for rvk in rvk_classifications if rvk]

                # Add DK classifications
                for dk in dk_classifications:
                    dk_results.append({
                        "dk": dk,
                        "classification_type": "DK",
                        "keyword": keyword,
                        "source_title": item.get("title", ""),
                        "source_rsn": item.get("rsn", ""),
                        "confidence": self._calculate_dk_confidence(item, keyword)
                    })

                # Add RVK classifications - Claude Generated
                for rvk in rvk_classifications:
                    dk_results.append({
                        "dk": rvk,  # Use same field name for compatibility
                        "classification_type": "RVK",
                        "keyword": keyword,
                        "source_title": item.get("title", ""),
                        "source_rsn": item.get("rsn", ""),
                        "confidence": self._calculate_dk_confidence(item, keyword)
                    })
        
        # Store new results in cache for future use
        if dk_results:
            # Convert to format expected by cache manager
            cache_results = []
            
            # Group results by classification for proper caching
            result_groups = {}
            for result in dk_results:
                dk = result.get("dk")
                # Skip results with None dk values - Claude Generated (Defensive)
                if not dk:
                    logger.warning(f"Skipping result with None/empty dk value: {result}")
                    continue
                classification_type = result.get("classification_type", "DK")
                key = f"{classification_type}:{dk}"
                
                if key not in result_groups:
                    result_groups[key] = {
                        "dk": dk,
                        "classification_type": classification_type,
                        "matched_keywords": set(),  # FIXED: Consistent field name with unified_knowledge_manager
                        "gnd_ids": set(),  # Claude Generated - Store GND-IDs separately
                        "titles": [],
                        "total_confidence": 0.0,
                        "count": 0
                    }

                # Extract keyword and GND-ID - Claude Generated
                keyword = result["keyword"]
                result_groups[key]["matched_keywords"].add(keyword)

                # Extract GND-ID from keyword if present (format: "Keyword (GND-ID: 1234567-8)")
                if "(GND-ID:" in keyword:
                    try:
                        gnd_id = keyword.split("(GND-ID:")[1].split(")")[0].strip()
                        result_groups[key]["gnd_ids"].add(gnd_id)
                        logger.debug(f"Extracted GND-ID {gnd_id} from keyword '{keyword}'")
                    except (IndexError, AttributeError) as e:
                        logger.warning(f"Failed to extract GND-ID from '{keyword}': {e}")
                # Filter empty/None titles before adding - Claude Generated (Defensive)
                title = result.get("source_title")
                if title is not None and isinstance(title, str) and title.strip():
                    result_groups[key]["titles"].append(title.strip())
                elif title is None:
                    logger.debug(f"Skipping None title for {dk} (keyword: {result.get('keyword')})")
                else:
                    logger.debug(f"Skipping empty/invalid title for {dk} (keyword: {result.get('keyword')})")
                result_groups[key]["total_confidence"] += result["confidence"]
                result_groups[key]["count"] += 1

            # Convert to cache format
            for group_data in result_groups.values():
                group_data["matched_keywords"] = list(group_data["matched_keywords"])  # FIXED: Consistent field name
                group_data["gnd_ids"] = list(group_data["gnd_ids"])  # Claude Generated - Convert GND-IDs set to list
                group_data["avg_confidence"] = group_data["total_confidence"] / group_data["count"]
                # Remove duplicates and filter whitespace - Claude Generated
                raw_titles = group_data["titles"]
                filtered_titles = [t.strip() for t in raw_titles if t and t.strip()]
                group_data["titles"] = list(dict.fromkeys(filtered_titles))
                logger.info(f"DK {group_data['dk']}: {len(group_data['titles'])} unique titles (from {len(raw_titles)} raw)")
                cache_results.append(group_data)
            
            # Store in cache
            dk_cache.store_classification_results(cache_results)
            live_search_time = (time.time() - live_search_start) * 1000
            logger.info(f"Stored {len(cache_results)} new DK classifications in cache (live search took {live_search_time:.1f}ms)")

            # Merge cached and live results - Claude Generated FIX with improved logging
            all_results = cached_results + cache_results
            logger.info(f"✅ CACHE STATISTICS: {len(cached_results)} cached + {len(cache_results)} live = {len(all_results)} total DK classifications")
            if keywords:
                cache_hit_count = len(keywords) - len(uncached_keywords)
                cache_hit_rate = (100 * cache_hit_count) // len(keywords)
                logger.info(f"   Cache hit rate: {cache_hit_count}/{len(keywords)} keywords ({cache_hit_rate}%)")
            logger.info(f"   Performance: Cache lookup {cache_time:.1f}ms, Live search {live_search_time:.1f}ms")

            # Sort by count and confidence
            all_results.sort(key=lambda x: (x["count"], x["avg_confidence"]), reverse=True)

            # Convert to keyword-centric structure - Claude Generated - Keyword-centric restructuring
            keyword_centric_results = self._restructure_to_keyword_centric(
                all_results, keyword_sources, keyword_timings
            )
            # Convert back to DK-centric for backward compatibility with Classification step
            dk_centric_results = self._flatten_to_dk_centric(keyword_centric_results)
            return dk_centric_results

        # Return cached results if no new live results were generated - Claude Generated FIX
        if cached_results:
            live_search_time = (time.time() - live_search_start) * 1000
            logger.info(f"⚠️ No new live search results (searched {len(uncached_keywords)} keywords in {live_search_time:.1f}ms)")
            logger.info(f"✅ CACHE STATISTICS: Returning {len(cached_results)} cached DK classifications (100% from cache)")

            # Convert to keyword-centric structure - Claude Generated - Keyword-centric restructuring
            keyword_centric_results = self._restructure_to_keyword_centric(
                cached_results, keyword_sources, keyword_timings
            )
            # Convert back to DK-centric for backward compatibility with Classification step
            dk_centric_results = self._flatten_to_dk_centric(keyword_centric_results)
            return dk_centric_results

        logger.warning(f"❌ No results from cache or live search for {len(keywords)} keywords")
        return []
    
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
