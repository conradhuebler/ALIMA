#!/usr/bin/env python3
# catalog_fallback_suggester.py

import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
import logging
import time
from typing import List, Dict, Any, Set, Optional, Union
from pathlib import Path
import json

from PyQt6.QtCore import QObject, pyqtSignal

from .base_suggester import BaseSuggester, BaseSuggesterError


class CatalogFallbackError(BaseSuggesterError):
    """Exception raised for errors in the CatalogFallbackSuggester."""

    pass


class CatalogFallbackSuggester(BaseSuggester):
    """
    A fallback suggester that extracts keywords and classifications from a library catalog
    using web scraping techniques instead of the SOAP API.

    This class uses web scraping to extract information from the catalog website
    and can be used when the SOAP API is unavailable or not functioning correctly.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        debug: bool = False,
        max_results: int = 20,
    ):
        """
        Initialize the catalog fallback suggester.

        Args:
            data_dir: Directory to store cached data (default: script_dir/data/catalogfallbacksuggester)
            debug: Whether to enable debug output
            max_results: Maximum number of records to process per search term
        """
        super().__init__(data_dir, debug)

        # Configure logging
        self.logger = logging.getLogger("catalog_fallback_suggester")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Base URLs for catalog access
        self.base_url = "https://katalog.ub.tu-freiberg.de/Search/Results"
        self.record_base_url = "https://katalog.ub.tu-freiberg.de/Record/"

        # Configuration
        self.max_results = max_results
        self.results_per_page = 20  # Standard catalog setting

        # Cache for search results
        self.cache_filename = "catalog_fallback_cache.json"
        self.cache = self._load_cache()

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

    def build_search_url(self, search_term, page=1):
        """
        Build the URL for searching the catalog.

        Args:
            search_term: The search term
            page: Result page number

        Returns:
            Complete search URL
        """
        encoded_search_term = urllib.parse.quote_plus(search_term)
        url = (
            f"{self.base_url}?lookfor={encoded_search_term}&type=AllFields"
            f"&hiddenFilters%5B%5D=institution%3ADE-105"
            f"&hiddenFilters%5B%5D=-format%3AArticle"
            f"&hiddenFilters%5B%5D=-format%3AElectronicArticle"
        )

        if page > 1:
            url += f"&page={page}"

        return url

    def get_total_pages(self, soup):
        """
        Determine the total number of search result pages.

        Args:
            soup: BeautifulSoup object of the search results page

        Returns:
            Total number of result pages
        """
        pagination = soup.select_one("ul.pagination")
        if not pagination:
            return 1

        # Find all page links
        page_links = pagination.select('a[href*="page="]')
        if not page_links:
            return 1

        max_page = 1
        for link in page_links:
            match = re.search(r"page=(\d+)", link["href"])
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)

        return max_page

    def extract_record_links(self, soup):
        """
        Extract record information from the search results page.

        Args:
            soup: BeautifulSoup object of the search results page

        Returns:
            List of tuples containing (record_id, record_url, title)
        """
        record_links = []

        # Find all save-record links and extract record IDs
        save_links = soup.find_all("a", class_="save-record")

        if not save_links:
            self.logger.warning("No save-record links found on the page")
            return record_links

        # Find all title links to associate them later
        title_elements = {}  # Dict to store record_id -> (title, url)
        for title_elem in soup.find_all("a", class_="title"):
            if "href" in title_elem.attrs:
                href = title_elem["href"]
                # Extract record_id from the URL
                record_match = re.search(r"/Record/([^/]+)", href)
                if record_match:
                    record_id = record_match.group(1)
                    title = title_elem.get_text(strip=True)
                    # Create complete URL
                    if href.startswith("/"):
                        record_url = f"https://katalog.ub.tu-freiberg.de{href}"
                    else:
                        record_url = href
                    title_elements[record_id] = (title, record_url)

        # Process the save-record links
        for link in save_links:
            record_id = link.get("data-id")
            if not record_id:
                continue

            # Construct the record URL directly from the record_id
            record_url = f"{self.record_base_url}{record_id}"

            # Try to get the title from the previously found title elements
            if record_id in title_elements:
                title, url = title_elements[record_id]
                # Prefer the URL from the title element if available
                record_url = url
            else:
                # Fallback: Use a generic title with the record ID
                title = f"Record {record_id}"

            record_links.append((record_id, record_url, title))

        return record_links

    def extract_subjects_from_record(self, record_id, record_url, title):
        """
        Extract subjects and classification numbers from a record page.

        Args:
            record_id: ID of the catalog record
            record_url: URL of the record page
            title: Title of the record

        Returns:
            Dictionary containing extracted subjects, DDCs, and DKs
        """
        if self.debug:
            self.logger.debug(
                f"Extracting information from record: {record_id} - {title}"
            )
            self.logger.debug(f"Opening URL: {record_url}")

        result = {"subjects": [], "ddc": [], "dk": []}

        try:
            # Get the record detail page
            response = requests.get(record_url)
            if response.status_code != 200:
                self.logger.warning(
                    f"HTTP {response.status_code} for record {record_id}: {record_url}"
                )
                return result

            # Parse the HTML page
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract subjects (keywords)
            subject_links = soup.select('a[href*="type=Subject"]')

            for link in subject_links:
                href = link.get("href", "")

                # Extract the subject from the URL
                match = re.search(r"lookfor=([^&]+)&type=Subject", href)
                if match:
                    raw_subject = urllib.parse.unquote_plus(match.group(1))
                    # Remove quotation marks
                    subject = raw_subject.strip('"')

                    # Split subjects connected with +
                    if "+" in subject:
                        for s in subject.split("+"):
                            s = s.strip()
                            if (
                                s and s not in result["subjects"]
                            ):  # Ignore empty strings
                                result["subjects"].append(s)
                    else:
                        if subject not in result["subjects"]:
                            result["subjects"].append(subject)

            # Extract DK numbers (Decimal classification)
            dk_links = soup.find_all(
                "a", href=re.compile(r"lookfor=DK.*?&type=udk_raw_de105")
            )
            for dk_link in dk_links:
                dk_match = re.search(r"DK\s+([\d\.:]+)", dk_link.text)
                if dk_match:
                    dk_number = dk_match.group(1)
                    if dk_number not in result["dk"]:
                        result["dk"].append(dk_number)

            # Extract Q numbers (alternative classification that can give hints to DDC)
            q_links = soup.find_all(
                "a", href=re.compile(r"lookfor=Q[A-Z]?\s*\d+.*?&type=udk_raw_de105")
            )
            for q_link in q_links:
                q_match = re.search(r"Q[A-Z]?\s*([\d\s]+)", q_link.text)
                if q_match:
                    q_number = q_match.group(1).strip()
                    # Q numbers can sometimes be mapped to DDC
                    # For simplicity, we just add them to the DDC list
                    if q_number not in result["ddc"]:
                        result["ddc"].append(q_number)

            return result

        except Exception as e:
            self.logger.error(f"Error extracting from record {record_id}: {str(e)}")
            return result

    def analyze_keywords(self, keywords):
        """
        Analyze a list of keywords to identify independent keywords and those
        that are only part of longer chains.

        Args:
            keywords: List of keywords and keyword chains

        Returns:
            Set of relevant keywords
        """
        # Remove duplicates but keep the original list for later comparison
        original_unique = list(dict.fromkeys(keywords))
        # Sort by length (number of words), from longest to shortest
        sorted_keywords = sorted(
            original_unique, key=lambda x: len(x.split()), reverse=False
        )

        # Identify keywords that are contained in others
        contained_in_others = set()
        for i, keyword in enumerate(sorted_keywords):
            for j, other in enumerate(sorted_keywords):
                if i != j:
                    tmp = other.replace(keyword, "").strip()
                    tmp2 = keyword.replace(other, "").strip()
                    if tmp:
                        contained_in_others.add(tmp)
                    if tmp2:
                        contained_in_others.add(tmp2)

        return contained_in_others

    def search_catalog(self, search_term, max_records=None):
        """
        Search the catalog for a specific term and extract subjects and classifications.

        Args:
            search_term: Term to search for
            max_records: Maximum number of records to process

        Returns:
            Dictionary of subjects with their metadata
        """
        if max_records is None:
            max_records = self.max_results

        if self.debug:
            self.logger.info(f"Searching catalog for: {search_term}")

        page = 1
        processed_records = 0

        # Store results as: subject -> {count, ddc, dk}
        results = {}

        while processed_records < max_records:
            # Create the URL for the search results page
            search_url = self.build_search_url(search_term, page)

            if self.debug:
                self.logger.info(f"Processing search result page {page}: {search_url}")

            try:
                # Get the search results page
                response = requests.get(search_url)
                if response.status_code != 200:
                    self.logger.error(
                        f"HTTP {response.status_code} for search result page {page}: {search_url}"
                    )
                    break

                soup = BeautifulSoup(response.text, "html.parser")

                # First pass: Determine total number of pages
                if page == 1:
                    total_pages = self.get_total_pages(soup)
                    if self.debug:
                        self.logger.info(f"Search results have {total_pages} pages")

                # Extract record links from the search results page
                record_links = self.extract_record_links(soup)

                if not record_links:
                    if self.debug:
                        self.logger.info(f"No more records found on page {page}")
                    break

                # Process each found record
                for record_id, record_url, title in record_links:
                    if processed_records >= max_records:
                        break

                    # Extract subjects from the record
                    record_data = self.extract_subjects_from_record(
                        record_id, record_url, title
                    )

                    # Store the found subjects
                    for subject in record_data["subjects"]:
                        if subject not in results:
                            results[subject] = {
                                "count": 1,
                                "ddc": set(record_data["ddc"]),
                                "dk": set(record_data["dk"]),
                                "gndid": set(),  # No GND IDs available from this source
                            }
                        else:
                            # Increment count
                            results[subject]["count"] += 1
                            # Update DDC and DK sets
                            results[subject]["ddc"].update(record_data["ddc"])
                            results[subject]["dk"].update(record_data["dk"])

                    processed_records += 1

                    # Signal progress
                    if processed_records % 5 == 0 or processed_records == max_records:
                        self.currentTerm.emit(
                            f"{search_term} ({processed_records}/{max_records})"
                        )

                    # Short pause to not overload the server
                    time.sleep(0.2)

                # If not enough records processed yet, go to the next page
                if processed_records < max_records and page < total_pages:
                    page += 1
                else:
                    break

            except Exception as e:
                self.logger.error(
                    f"Error processing search result page {page}: {str(e)}"
                )
                break

        # Further analyze keywords to find compound terms
        if results:
            all_keywords = list(results.keys())
            relevant_keywords = self.analyze_keywords(all_keywords)

            # Add compound terms as subjects if they're not already in the results
            for keyword in relevant_keywords:
                if keyword not in results:
                    # Find the most frequent count among component keywords
                    max_count = 1
                    ddc_set = set()
                    dk_set = set()

                    # Look for keywords that might be components of this compound term
                    for k in all_keywords:
                        if keyword in k:
                            if results[k]["count"] > max_count:
                                max_count = results[k]["count"]
                            ddc_set.update(results[k]["ddc"])
                            dk_set.update(results[k]["dk"])

                    # Add the compound term with collected data
                    results[keyword] = {
                        "count": max_count,
                        "ddc": ddc_set,
                        "dk": dk_set,
                        "gndid": set(),
                    }

        if self.debug:
            self.logger.info(
                f"Extraction completed: {len(results)} subjects from {processed_records} records"
            )

        return results

    def prepare(self, force_download: bool = False) -> None:
        """
        Prepare the suggester. For fallback suggester, this does nothing substantial.

        Args:
            force_download: Whether to force data download/preparation
        """
        # Nothing to prepare for catalog fallback suggester
        pass

    def search(self, terms: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Search for subjects related to the given search terms.

        Args:
            terms: List of search terms

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

        for search_term in terms:
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
                results[search_term] = catalog_results

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
