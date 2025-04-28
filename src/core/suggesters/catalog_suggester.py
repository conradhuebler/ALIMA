#!/usr/bin/env python3
# catalog_suggester.py

import requests
import xml.etree.ElementTree as ET
import re
import sys
import logging
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Union, Tuple

from .base_suggester import BaseSuggester, BaseSuggesterError


class CatalogSuggesterError(BaseSuggesterError):
    """Exception raised for errors in the CatalogSuggester."""

    pass


class CatalogSuggester(BaseSuggester):
    """
    Subject suggester that extracts keywords and classifications from a library catalog.
    Uses SOAP API to search the catalog and extract subject information.
    """

    # Base URLs for API requests

    # MAB-Tags for subjects
    MAB_SUBJECT_TAGS = ["0902", "0907", "0912", "0917", "0922", "0927"]

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        token: str = "",
        debug: bool = False,
        catalog_search_url: str = "",
        catalog_details: str = "",
    ):
        """
        Initialize the catalog suggester.

        Args:
            data_dir: Directory to store cached data (default: script_dir/data/catalogsuggester)
            token: Authentication token for the library API
            debug: Whether to enable debug output
        """
        super().__init__(data_dir, debug)
        self.token = token
        self.session = requests.Session()
        self.headers = {"Content-Type": "text/xml;charset=UTF-8", "SOAPAction": ""}
        self.cache_filename = "catalog_cache.json"
        self.cache = self._load_cache()
        self.SEARCH_URL = catalog_search_url
        self.DETAILS_URL = catalog_details
        # Configure logging
        self.logger = logging.getLogger("catalog_suggester")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def _load_cache(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load cache of previously retrieved data.

        Returns:
            Dictionary containing cached search results
        """
        cache_file = self.data_dir / self.cache_filename
        if not cache_file.exists():
            return {}

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                result = {}

                for search_term, subjects in data.items():
                    result[search_term] = {}
                    for subject_name, subject_data in subjects.items():
                        # Convert sets back from lists
                        for field in ["gndid", "ddc", "dk"]:
                            if isinstance(subject_data.get(field), list):
                                subject_data[field] = set(subject_data[field])
                            elif isinstance(
                                subject_data.get(field), str
                            ) and subject_data.get(field):
                                subject_data[field] = {subject_data[field]}
                            else:
                                subject_data[field] = set()

                        result[search_term][subject_name] = subject_data

                return result
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Could not load cache: {e}")
            return {}

    def _save_cache(self):
        """Save cache of search results."""
        cache_file = self.data_dir / self.cache_filename
        try:
            # Convert sets to lists for JSON serialization
            serializable_data = {}
            for search_term, subjects in self.cache.items():
                serializable_data[search_term] = {}
                for subject_name, subject_data in subjects.items():
                    serializable_subject = subject_data.copy()

                    for field in ["gndid", "ddc", "dk"]:
                        if isinstance(subject_data.get(field), set):
                            serializable_subject[field] = list(subject_data[field])

                    serializable_data[search_term][subject_name] = serializable_subject

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Could not save cache: {e}")

    def _print_xml_structure(self, element, level=0):
        """
        Print the XML structure for debugging purposes.

        Args:
            element: XML element to print
            level: Current indentation level
        """
        indent = "  " * level
        if hasattr(element, "tag"):
            self.logger.debug(f"{indent}{element.tag}")
            for child in element:
                self._print_xml_structure(child, level + 1)

    def search_catalog(
        self, term: str, search_type: str = "ku"
    ) -> List[Dict[str, Any]]:
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

        if self.debug:
            self.logger.debug(f"Sending search request to {self.SEARCH_URL}")
            self.logger.debug(f"Search envelope: {search_envelope}")

        try:
            response = self.session.post(
                self.SEARCH_URL, headers=self.headers, data=search_envelope, timeout=300
            )

            if self.debug:
                self.logger.debug(f"Search response status: {response.status_code}")
                self.logger.debug(f"Search response content: {response.text}")

            if response.status_code != 200:
                self.logger.error(
                    f"Error searching: {response.status_code} - {response.text}"
                )
                return []

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
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
            ]

            items_found = False
            for path in paths:
                if self.debug:
                    self.logger.debug(f"Trying to find items with path: {path}")
                items = root.findall(path)
                if items:
                    items_found = True
                    if self.debug:
                        self.logger.info(f"Found {len(items)} items with path {path}")
                    break

            if not items_found:
                # Dump the XML structure for debugging
                if self.debug:
                    self.logger.debug("XML structure:")
                    self._print_xml_structure(root)
                self.logger.warning(
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

            self.logger.info(f"Parsed {len(result_items)} result items")
            if self.debug and result_items:
                self.logger.debug(f"First result item: {result_items[0]}")

            return result_items

        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            self.logger.error(f"Response content: {response.text}")
            return []

    def get_title_details(self, rsn: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific title.

        Args:
            rsn: The RSN (Record Serial Number) of the title

        Returns:
            A dictionary with title details or None if an error occurred
        """
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

        if self.debug:
            self.logger.debug(f"Getting details for RSN: {rsn}")
            self.logger.debug(f"Details envelope: {details_envelope}")

        try:
            response = self.session.post(
                self.DETAILS_URL,
                headers=self.headers,
                data=details_envelope,
                timeout=300,
            )

            if self.debug:
                self.logger.debug(f"Details response status: {response.status_code}")
                self.logger.debug(f"Details response content: {response.text}")

            if response.status_code != 200:
                self.logger.error(
                    f"Error getting details: {response.status_code} - {response.text}"
                )
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

        # Parse the response XML
        try:
            root = ET.fromstring(response.content)

            # If debug is enabled, print the structure
            if self.debug:
                self.logger.debug("Details XML structure:")
                self._print_xml_structure(root)

            # Extract details
            details = {}

            # Extract MAB-based subjects from the MAB tags
            mab_subjects = self._extract_mab_subjects(root)
            if mab_subjects:
                if self.debug:
                    self.logger.debug(f"Found {len(mab_subjects)} MAB subjects")
                details["mab_subjects"] = mab_subjects

            # Try different possible paths for classifications
            classification_paths = [
                ".//Classification/Classifications/Classification",
                ".//{http://libero.com.au}Classification/{http://libero.com.au}Classifications/{http://libero.com.au}Classification",
            ]

            classifications = []
            for path in classification_paths:
                if self.debug:
                    self.logger.debug(f"Trying classification path: {path}")
                for classification in root.findall(path):
                    if classification.text:
                        classifications.append(classification.text)
                        if self.debug:
                            self.logger.debug(
                                f"Found classification: {classification.text}"
                            )

                if classifications:
                    break

            details["classifications"] = classifications

            # Try different possible paths for subjects
            subject_paths = [
                ".//Subject/Subjects/Subject",
                ".//{http://libero.com.au}Subject/{http://libero.com.au}Subjects/{http://libero.com.au}Subject",
            ]

            subjects = []
            for path in subject_paths:
                if self.debug:
                    self.logger.debug(f"Trying subject path: {path}")
                for subject in root.findall(path):
                    if subject.text:
                        subjects.append(subject.text)
                        if self.debug:
                            self.logger.debug(f"Found subject: {subject.text}")

                if subjects:
                    break

            details["subjects"] = subjects

            # Extract basic information using multiple paths
            title_paths = [".//Title", ".//{http://libero.com.au}Title"]
            details["title"] = self._extract_text_multiple_paths(root, title_paths)

            details["author"] = self._extract_authors(root)

            publication_paths = [
                ".//Publication",
                ".//{http://libero.com.au}Publication",
            ]
            details["publication"] = self._extract_text_multiple_paths(
                root, publication_paths
            )

            isbn_paths = [".//ISBN", ".//{http://libero.com.au}ISBN"]
            details["isbn"] = self._extract_text_multiple_paths(root, isbn_paths)

            details["rsn"] = rsn

            return details

        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            if hasattr(response, "text"):
                self.logger.error(f"Response content: {response.text}")
            return None

    def _extract_mab_subjects(self, root) -> List[str]:
        """
        Extract subjects from MAB data in XML.

        Args:
            root: XML root element

        Returns:
            List of extracted subjects
        """
        mab_subjects = []

        # Try different paths for MAB entries
        mab_paths = [".//MAB", ".//{http://libero.com.au}MAB"]

        for path in mab_paths:
            mab_elements = root.findall(path)

            for mab in mab_elements:
                # Check if this MAB element contains a subject
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
                    # Extract the subject
                    subject = None
                    if mab_data_plain:
                        # Extract directly from MABDataPlain
                        parts = mab_data_plain.strip().split()
                        if len(parts) > 1:
                            # Ignore the first part with numbers and spaces
                            subject = " ".join(parts[1:]).strip()
                    elif mab_data:
                        # Base64-decode and then extract
                        try:
                            decoded = base64.b64decode(mab_data).decode("utf-8")
                            parts = decoded.strip().split()
                            if len(parts) > 1:
                                subject = " ".join(parts[1:]).strip()
                        except Exception as e:
                            self.logger.warning(f"Error decoding Base64 data: {e}")

                    if subject and subject not in mab_subjects:
                        if self.debug:
                            self.logger.debug(f"Found MAB subject: {subject}")
                        mab_subjects.append(subject)

        return mab_subjects

    def _extract_text_multiple_paths(self, root, paths: List[str]) -> str:
        """
        Helper method to extract text from an XML element using multiple possible paths.

        Args:
            root: XML root element
            paths: List of XPath expressions to try

        Returns:
            Extracted text or empty string if not found
        """
        for path in paths:
            element = root.find(path)
            if element is not None and element.text:
                if self.debug:
                    self.logger.debug(f"Found text with path {path}: {element.text}")
                return element.text
        if self.debug:
            self.logger.debug(f"No text found for paths: {paths}")
        return ""

    def _extract_authors(self, root) -> List[str]:
        """
        Helper method to extract authors in display form.

        Args:
            root: XML root element

        Returns:
            List of author names
        """
        authors = []

        # Try different possible paths
        author_paths = [
            ".//Author/Authors/AuthorDisplayForm",
            ".//{http://libero.com.au}Author/{http://libero.com.au}Authors/{http://libero.com.au}AuthorDisplayForm",
        ]

        for path in author_paths:
            if self.debug:
                self.logger.debug(f"Trying author path: {path}")
            for author in root.findall(path):
                if author.text:
                    authors.append(author.text)
                    if self.debug:
                        self.logger.debug(f"Found author: {author.text}")

            if authors:
                break

        return authors

    def extract_decimal_classifications(
        self, classifications: List[str]
    ) -> Dict[str, str]:
        """
        Extract decimal classifications (DDC, DK) from a list of classifications.

        Args:
            classifications: List of classification strings

        Returns:
            Dictionary with 'ddc' and 'dk' classifications
        """
        result = {"ddc": [], "dk": []}  # DDC classifications  # DK classifications

        for classification in classifications:
            if self.debug:
                self.logger.debug(f"Processing classification: {classification}")

            # Extract DK classifications (matches patterns like "DK 543.42")
            if "DK " in classification:
                match = re.search(r"DK\s+(\d+(?:\.\d+)?)", classification)
                if match:
                    dk_class = match.group(1)
                    result["dk"].append(dk_class)
                    if self.debug:
                        self.logger.debug(f"Extracted DK classification: {dk_class}")

            # Extract potential DDC classifications (matches patterns like "543/.62")
            elif "/" in classification and not classification.startswith("RVK"):
                parts = classification.split("/")
                if parts[0].strip().isdigit() and parts[1].strip().startswith("."):
                    ddc_class = f"{parts[0].strip()}{parts[1].strip()}"
                    result["ddc"].append(ddc_class)
                    if self.debug:
                        self.logger.debug(f"Extracted DDC classification: {ddc_class}")

            # Check for simple DDC numbers (like "004.6" or "510")
            elif re.match(r"^\d+(\.\d+)?$", classification.strip()):
                result["ddc"].append(classification.strip())
                if self.debug:
                    self.logger.debug(
                        f"Extracted DDC classification: {classification.strip()}"
                    )

        return result

    def process_search_results(
        self, results: List[Dict[str, Any]], max_items: int = 20, delay: float = 0.5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process search results by getting details for each item.
        Extracts keywords and classifications from each title.

        Args:
            results: List of search result items
            max_items: Maximum number of items to process
            delay: Delay between requests in seconds

        Returns:
            Dictionary mapping subjects to their metadata
        """
        combined_results = {}  # Mapping subject name to metadata

        self.logger.info(f"Processing {min(len(results), max_items)} items...")

        for i, item in enumerate(results[:max_items]):
            if i > 0:
                time.sleep(delay)  # Be nice to the server

            title = item.get("title", "Unknown")
            self.logger.info(
                f"Processing item {i+1}/{min(len(results), max_items)}: {title}"
            )

            rsn = item.get("rsn")
            if not rsn:
                self.logger.warning(f"No RSN found for item: {title}")
                continue

            details = self.get_title_details(rsn)
            if not details:
                self.logger.warning(f"Could not get details for RSN: {rsn}")
                continue

            # Extract subjects from both subjects and MAB subjects
            subjects = details.get("subjects", []) + details.get("mab_subjects", [])

            # Extract classifications
            classification_values = self.extract_decimal_classifications(
                details.get("classifications", [])
            )

            # Process each subject
            for subject in subjects:
                subject = subject.strip()
                if not subject:
                    continue

                # Initialize or update subject entry
                if subject not in combined_results:
                    combined_results[subject] = {
                        "count": 1,
                        "gndid": set(),  # GND IDs as a set
                        "ddc": set(classification_values["ddc"]),  # DDC as a set
                        "dk": set(classification_values["dk"]),  # DK as a set
                    }
                else:
                    # Increment count
                    combined_results[subject]["count"] += 1

                    # Update classifications
                    combined_results[subject]["ddc"].update(
                        classification_values["ddc"]
                    )
                    combined_results[subject]["dk"].update(classification_values["dk"])

        return combined_results

    def prepare(self, force_download: bool = False) -> None:
        """
        Prepare the suggester. For catalog suggester, this does nothing substantial.

        Args:
            force_download: Whether to force data download/preparation
        """
        # Nothing to prepare for catalog suggester
        pass

    def search(self, searches: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Search for subjects related to the given search terms.

        Args:
            searches: List of search terms

        Returns:
            Dictionary with structure:
            {
                search_term: {
                    keyword: {
                        "count": int,
                        "gndid": set,
                        "ddc": set,
                        "dk": set
                    }
                }
            }
        """
        results = {}

        for search_term in searches:
            # Check cache first
            if search_term in self.cache:
                if self.debug:
                    self.logger.info(f"Using cached results for '{search_term}'")
                results[search_term] = self.cache[search_term]
                # Signal that we've processed this term
                self.currentTerm.emit(search_term)
                continue

            # Search the catalog
            self.logger.info(f"Searching catalog for term: {search_term}")
            catalog_results = self.search_catalog(search_term)

            if not catalog_results:
                self.logger.warning(f"No results found for '{search_term}'")
                results[search_term] = {}  # Empty result
            else:
                # Process the search results
                self.logger.info(
                    f"Found {len(catalog_results)} results for '{search_term}'"
                )
                results[search_term] = self.process_search_results(
                    catalog_results, max_items=20
                )

                # Cache the results
                self.cache[search_term] = results[search_term]
                self._save_cache()

            # Signal that we've processed this term
            self.currentTerm.emit(search_term)

            if self.debug:
                self.logger.info(
                    f"Found {len(results[search_term])} subjects for '{search_term}'"
                )

                # Show sample results
                if results[search_term]:
                    self.logger.debug("Sample results:")
                    sample_count = min(5, len(results[search_term]))
                    for i, (subject, data) in enumerate(
                        list(results[search_term].items())[:sample_count]
                    ):
                        self.logger.debug(f"{i+1}. '{subject}': {data}")

        return results
