"""SOAP/web request axis of ``BiblioClient``.

Split out of ``biblio_client.py`` (WP cleanup D, F-15). Verbatim mixin
extraction: the methods stay on :class:`BiblioClient` via MRO, so no call site
changes. The public ``search`` retry wrapper travels with the SOAP request
builders and their web-scraping fallbacks — one self-contained request axis.

⚠️ The module logger is named ``"biblio_extractor"`` (not ``__name__``); it must
match ``biblio_client.py`` exactly, otherwise these log lines silently go to a
different logger.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("biblio_extractor")


class BiblioRequestMixin:
    """SOAP search/details requests + web-scraping fallbacks. Mixed into :class:`BiblioClient`."""

    def search(self, term: str, search_type: str = "ku") -> List[Dict[str, Any]]:
        """
        Search for items in the catalog with retry logic for transient errors.

        Args:
            term: The search term
            search_type: The type of search (default: "ku" for anyword)

        Returns:
            A list of search result items
        """
        # Retry configuration - Claude Generated
        max_retries = 3
        backoff_base = 1  # Start at 1 second

        for attempt in range(max_retries):
            try:
                return self._search_attempt(term, search_type)

            except requests.exceptions.ConnectionError as e:
                # Transient connection error - retry with exponential backoff - Claude Generated
                if attempt < max_retries - 1:
                    backoff_time = backoff_base * (2 ** attempt)  # Exponential: 1s, 2s, 4s
                    logger.warning(
                        f"⚠️ Connection error for '{term}' (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {backoff_time}s: {e}"
                    )
                    time.sleep(backoff_time)
                else:
                    logger.error(f"❌ Connection failed after {max_retries} attempts: {e}")
                    return []

            except requests.exceptions.Timeout as e:
                # Timeout - retry - Claude Generated
                if attempt < max_retries - 1:
                    backoff_time = backoff_base * (2 ** attempt)
                    logger.warning(
                        f"⚠️ Timeout for '{term}' (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {backoff_time}s"
                    )
                    time.sleep(backoff_time)
                else:
                    logger.error(f"❌ Timeout after {max_retries} attempts")
                    return []

            except requests.exceptions.HTTPError as e:
                # HTTP errors that should NOT retry (4xx client errors) - Claude Generated
                if 400 <= e.response.status_code < 500:
                    logger.error(f"❌ Client error {e.response.status_code} (not retrying): {e}")
                    return []
                # Server error (5xx) - retry - Claude Generated
                else:
                    if attempt < max_retries - 1:
                        backoff_time = backoff_base * (2 ** attempt)
                        logger.warning(
                            f"⚠️ Server error {e.response.status_code} (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {backoff_time}s"
                        )
                        time.sleep(backoff_time)
                    else:
                        logger.error(f"❌ Server error after {max_retries} attempts")
                        return []

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Request failed for '{term}': {e}")
                return []

        return []

    def _search_attempt(self, term: str, search_type: str) -> List[Dict[str, Any]]:
        """Single search attempt without retry logic - Claude Generated"""
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

        response = self.session.post(
            self.SEARCH_URL, headers=self.headers, data=search_envelope, timeout=self.timeout
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

        # Detect HTML responses (server error pages) - Claude Generated
        # The server sometimes returns HTML error pages instead of SOAP XML
        if response.text.strip().startswith('<?xml') == False and ('<html>' in response.text.lower() or '<body>' in response.text.lower() or '<framestack>' in response.text.lower()):
            logger.warning(f"❌ Server returned HTML instead of SOAP for search '{term}' - Record may not exist or server error")
            if self.enable_web_fallback:
                logger.info(f"Attempting web fallback for search term: {term}")
                web_results = self._search_web(term)
                if web_results:
                    logger.info(f"✅ Web fallback search successful: {len(web_results)} results")
                    return web_results
            return []

        # Parse the response XML (if no SOAP fault, parse successfully) - Claude Generated
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
                    logger.debug(f"Found {len(items)} items with path {path}")
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

            logger.debug(f"Parsed {len(result_items)} result items")
            if self.debug and result_items:
                logger.debug(f"First result item: {result_items[0]}")

            return result_items

        except ET.ParseError as e:
            logger.error(f"❌ XML parsing error for search '{term}': {e}")
            # Check if response is HTML (malformed XML)
            if '<html>' in response.text.lower() or '<body>' in response.text.lower():
                logger.error(f"Server returned HTML instead of SOAP - trying web fallback")
                if self.enable_web_fallback:
                    web_results = self._search_web(term)
                    if web_results:
                        return web_results
            return []

    def get_title_details(self, rsn: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific title.

        Args:
            rsn: The RSN (Record Serial Number) of the title

        Returns:
            A dictionary with title details or None if an error occurred
        """
        # OPTIMIZATION: Try JSON lookup first before expensive SOAP request - Claude Generated
        if self.use_json_cache:
            try:
                # Import ClassificationLookupService - only load when needed for performance
                from ...utils.classification_lookup_service import get_classification_lookup_service

                # Try to get details from JSON cache
                json_lookup = get_classification_lookup_service()
                rsn_int = int(rsn) if str(rsn).isdigit() else None

                if rsn_int:
                    cached_details = json_lookup.get_title_details_from_rsn(rsn_int)
                    if cached_details:
                        self._json_cache_hits += 1  # Track for statistics
                        logger.debug(f"JSON CACHE HIT for RSN {rsn}")
                        # Track that we used JSON optimization for delay calculation
                        if not hasattr(self, '_json_lookups_used'):
                            self._json_lookups_used = 0
                        self._json_lookups_used += 1
                        return cached_details
                    else:
                        logger.debug(f"📁 RSN {rsn} not in JSON cache, using SOAP")
                    # Track that we need to use SOAP for delay calculation
                    if not hasattr(self, '_soap_lookups_needed'):
                        self._soap_lookups_needed = 0
                    self._soap_lookups_needed += 1
            except Exception as e:
                logger.debug(f"JSON lookup failed, falling back to SOAP: {e}")
                # Track that we need to use SOAP due to JSON failure
                if not hasattr(self, '_soap_lookups_needed'):
                    self._soap_lookups_needed = 0
                self._soap_lookups_needed += 1
        else:
            # JSON cache disabled for testing - use SOAP only
            logger.debug(f"⚠️ JSON cache DISABLED, using SOAP for RSN {rsn}")
            if not hasattr(self, '_soap_lookups_needed'):
                self._soap_lookups_needed = 0
            self._soap_lookups_needed += 1
            # Continue with SOAP even if JSON lookup fails

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

        response = None
        try:
            # Track SOAP call for statistics - Claude Generated
            self._soap_calls += 1
            logger.debug(f"SOAP REQUEST for RSN {rsn} (call #{self._soap_calls})")

            response = self.session.post(
                self.DETAILS_URL,
                headers=self.headers,
                data=details_envelope,
                timeout=10,  # Reduced from 300s (5 min) to 10s - fast failover when server down - Claude Generated (2026-01-13)
            )

            logger.debug(f"✅ Details response received for RSN {rsn}: status {response.status_code}")

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

            # Detect HTML responses (server error pages) - Claude Generated
            # The server sometimes returns HTML error pages instead of SOAP XML
            if response.text.strip().startswith('<?xml') == False and ('<html>' in response.text.lower() or '<body>' in response.text.lower() or '<framestack>' in response.text.lower()):
                logger.warning(f"❌ Server returned HTML instead of SOAP for RSN {rsn} - Record may not exist or server error")
                if self.enable_web_fallback:
                    logger.info(f"Attempting web fallback for invalid/missing RSN {rsn}")
                    fallback_details = self._get_title_details_web(rsn)
                    if fallback_details:
                        return fallback_details
                return None

            # Parse the response XML
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
                ".//{http://libero.com.au}GetTitleDetailsResult/{http://libero.com.au}Classification/{http://libero.com.au}Classifications/{http://libero.com.au}Classification",
                ".//{http://libero.com.au}Classification/{http://libero.com.au}Classifications/{http://libero.com.au}Classification",
                ".//Classification/Classifications/Classification", # Keeping for cases without namespace
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
                    logger.debug(f"✅ Found {len(classifications)} classifications using path: {path}")
                    break

            # Log if no classifications found to help debugging
            if not classifications:
                logger.debug(f"No classifications found for RSN {rsn}")

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
                logger.debug(f"No title found for RSN {rsn}")

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
                    logger.debug(f"Extracted DK numbers from title text for RSN {rsn}: {dk_from_title}")
                    details["classifications"] = dk_from_title
                    details["_source"] = "extracted_from_title"  # Mark as fallback source

            # Diagnostic: log extraction summary - Claude Generated
            # DIAGNOSTIC: Always log extraction results for debugging - Claude Generated
            extracted_fields = sum([
                bool(details.get("title")),
                bool(details.get("author")),
                bool(details.get("publication")),
                bool(details.get("isbn")),
                bool(details.get("classifications")),
                bool(details.get("subjects")),
                bool(details.get("mab_subjects"))
            ])

            cls_count = len(details.get("classifications", []))
            subj_count = len(details.get("subjects", []))
            mab_count = len(details.get("mab_subjects", []))

            logger.debug(f"📄 Extraction for RSN {rsn}: {extracted_fields}/7 fields | "
                         f"Classifications: {cls_count} | Subjects: {subj_count} | MAB: {mab_count}")

            if extracted_fields == 0:
                logger.debug(f"No data extracted for RSN {rsn} - SOAP response empty or malformed")
            elif cls_count == 0:
                logger.debug(f"⚠️ RSN {rsn}: No classifications found (DK/RVK missing)")

            return details

        except ET.ParseError as e:
            logger.error(f"❌ XML parsing error for RSN {rsn}: {e}")
            if hasattr(response, "text"):
                # Check if response is HTML (malformed XML)
                if '<html>' in response.text.lower() or '<body>' in response.text.lower():
                    logger.error(f"Server returned HTML instead of SOAP - trying web fallback")
                    if self.enable_web_fallback:
                        fallback_details = self._get_title_details_web(rsn)
                        if fallback_details:
                            return fallback_details
                else:
                    logger.error(f"Response content: {response.text[:500]}")
            return None

        except Exception as e:
            # Catch timeout and other errors - Claude Generated
            if isinstance(e, requests.exceptions.Timeout):
                logger.error(f"⏱️ TIMEOUT getting details for RSN {rsn} after 300s - server overloaded or network issue")
            elif isinstance(e, requests.exceptions.ConnectionError):
                logger.error(f"🔌 CONNECTION ERROR getting details for RSN {rsn}: {str(e)}")
            else:
                logger.error(f"❌ Unexpected error getting details for RSN {rsn}: {type(e).__name__}: {str(e)}")
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
