#!/usr/bin/env python3
# swb_suggester.py

import requests
import re
import json
import urllib.parse
import sys
import time
import html
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Union
from bs4 import BeautifulSoup

from src.utils.suggesters.base_suggester import BaseSuggester, BaseSuggesterError

# Every SWB HTTP request carries this timeout — a hung endpoint must not hang
# the pipeline (project HTTP convention). - Claude Generated
REQUEST_TIMEOUT_S = 15


class SWBSuggesterError(BaseSuggesterError):
    """Exception raised for errors in the SWBSuggester."""

    pass


class SWBSuggester(BaseSuggester):
    """
    Subject suggester that extracts GND subjects from the SWB (Südwestdeutscher Bibliotheksverbund) interface.
    Uses web scraping to extract information from the SWB catalog interface.
    """

    def __init__(
        self, data_dir: Optional[Union[str, Path]] = None, debug: bool = False
    ):
        """
        Initialize the SWB Suggester.

        Args:
            data_dir: Directory to store cached data (default: script_dir/data/swbsuggester)
            debug: Whether to enable debug output
        """
        super().__init__(data_dir, debug)
        self.cache_filename = "swb_gnd_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Load cache of previously retrieved GND IDs.

        Returns:
            Dictionary containing cached search results
        """
        cache_file = self.data_dir / self.cache_filename
        if not cache_file.exists():
            return {}

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                # The cache format is complex because we store gndid as sets
                # We need to reconstruct the sets when loading
                data = json.load(f)
                result = {}

                for search_term, subjects in data.items():
                    result[search_term] = {}
                    for subject_name, subject_data in subjects.items():
                        # Convert gndid back to a set
                        if isinstance(subject_data.get("gndid"), list):
                            subject_data["gndid"] = set(subject_data["gndid"])
                        elif isinstance(subject_data.get("gndid"), str):
                            subject_data["gndid"] = {subject_data["gndid"]}

                        # Convert ddc and dk back to sets if they exist
                        if isinstance(subject_data.get("ddc"), list):
                            subject_data["ddc"] = set(subject_data["ddc"])
                        elif isinstance(
                            subject_data.get("ddc"), str
                        ) and subject_data.get("ddc"):
                            subject_data["ddc"] = {subject_data["ddc"]}
                        else:
                            subject_data["ddc"] = set()

                        if isinstance(subject_data.get("dk"), list):
                            subject_data["dk"] = set(subject_data["dk"])
                        elif isinstance(
                            subject_data.get("dk"), str
                        ) and subject_data.get("dk"):
                            subject_data["dk"] = {subject_data["dk"]}
                        else:
                            subject_data["dk"] = set()

                        result[search_term][subject_name] = subject_data

                return result
        except Exception as e:
            self.logger.warning(f"Could not load SWB cache: {e}")
            return {}

    def _save_cache(self):
        """Save cache of GND IDs and classifications."""
        cache_file = self.data_dir / self.cache_filename
        try:
            # Convert sets to lists for JSON serialization
            serializable_data = {}
            for search_term, subjects in self.cache.items():
                serializable_data[search_term] = {}
                for subject_name, subject_data in subjects.items():
                    serializable_subject = subject_data.copy()

                    # Convert set to list for each field that might be a set
                    for field in ["gndid", "ddc", "dk"]:
                        if isinstance(subject_data.get(field), set):
                            serializable_subject[field] = list(subject_data[field])

                    serializable_data[search_term][subject_name] = serializable_subject

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save SWB cache: {e}")

    def _is_single_result_page(self, content: str) -> bool:
        """
        Detects if the response is a single result page rather than a list of results.
        This happens when the search term matches exactly one GND entry.

        Args:
            content: HTML content of the page

        Returns:
            True if the page is a single result page, False otherwise
        """
        # Typical features of a single result page
        signs = [
            # Search for GND number in a single display
            re.search(r"<div>GND-Nummer:.*?</div>", content, re.DOTALL) is not None,
            # Search for typical fields of a single display
            re.search(r"<div>Sachbegriff:.*?</div>", content, re.DOTALL) is not None,
            # Search for typical navigation links missing in result lists
            re.search(r"Oberbegriff:.*?</div>", content, re.DOTALL) is not None,
            # Check if no typical list results are present
            re.search(r"Treffer</span>", content, re.DOTALL) is None,
        ]

        # If at least 3 of the 4 signs apply, it's probably a single page
        return sum(signs) >= 3

    def _extract_details_from_single_result(self, content: str) -> Dict[str, str]:
        """
        Extracts subject information from a single result page
        where the search directly led to a specific GND entry.

        Args:
            content: HTML content of the page

        Returns:
            Dictionary mapping subject names to GND IDs
        """
        soup = BeautifulSoup(content, "html.parser")

        # Search for the subject term using the exact table layout
        subject_name = None

        # Search for <tr> with "Sachbegriff:" in the text
        sachbegriff_rows = soup.find_all(
            "tr", string=lambda text: text and "Sachbegriff:" in text
        )

        if not sachbegriff_rows:
            # Alternative: Search for td with class "preslabel" and "Sachbegriff:" in text
            sachbegriff_labels = soup.find_all(
                "td",
                {"class": "preslabel"},
                string=lambda text: text and "Sachbegriff:" in text,
            )

            if sachbegriff_labels:
                for label in sachbegriff_labels:
                    row = label.parent  # The tr element
                    if row:
                        sachbegriff_rows.append(row)

        if not sachbegriff_rows:
            # Third attempt: Search for div with "Sachbegriff:"
            sachbegriff_divs = soup.find_all(
                "div", string=lambda text: text and "Sachbegriff:" in text
            )
            for div in sachbegriff_divs:
                parent_td = div.find_parent("td")
                if parent_td:
                    row = parent_td.find_parent("tr")
                    if row:
                        sachbegriff_rows.append(row)

        # Try to extract the title from the found rows
        for row in sachbegriff_rows:
            # Search for the next TD with class "presvalue"
            value_td = row.find("td", {"class": "presvalue"})
            if value_td:
                # Search for the Bold element (b tag)
                bold_elem = value_td.find("b")
                if bold_elem:
                    subject_name = bold_elem.text.strip()
                    break

        # If no subject was found, search directly for the bold element near "Sachbegriff:"
        if not subject_name:
            # Search in the text for the pattern "Sachbegriff:"...anything...<b>TITLE</b>
            title_match = re.search(r"Sachbegriff:.*?<b>(.*?)</b>", content, re.DOTALL)
            if title_match:
                subject_name = title_match.group(1).strip()

        # Search for the GND number in the link
        gnd_id = None
        gnd_links = soup.find_all(
            "a", href=lambda href: href and "d-nb.info/gnd/" in href
        )

        for link in gnd_links:
            # Extract the GND ID from the link
            gnd_match = re.search(r"d-nb\.info/gnd/(\d+-\d+)", link["href"])
            if gnd_match:
                gnd_id = gnd_match.group(1)
                break

        # If no GND ID was found in the link, search in the entire text
        if not gnd_id:
            # Search for GND-Nummer: and extract the ID
            gnd_match = re.search(r"GND-Nummer:.*?(\d+-\d+)", content, re.DOTALL)
            if gnd_match:
                gnd_id = gnd_match.group(1)
            else:
                # Try to extract the GND ID from any link
                gnd_link_match = re.search(r"d-nb\.info/gnd/(\d+-\d+)", content)
                if gnd_link_match:
                    gnd_id = gnd_link_match.group(1)

        # Debug output
        if self.debug:
            self.logger.debug(
                f"Single result extraction - Subject name: '{subject_name}', GND ID: '{gnd_id}'"
            )

        # If both title and GND ID were found, create an entry
        if subject_name and gnd_id:
            return {subject_name: gnd_id}

        # If something is missing, return an empty dictionary.
        # Real data loss (term will yield nothing) — log unconditionally - Claude Generated
        if not subject_name:
            self.logger.warning("Could not extract subject name from single result page")
        if not gnd_id:
            self.logger.warning("Could not extract GND ID from single result page")

        return {}

    def _extract_subjects_from_page(self, content: str) -> Dict[str, str]:
        """
        Extract subjects (Sachbegriffe) and their GND IDs from the page content.

        Args:
            content: HTML content of the page

        Returns:
            Dictionary mapping subject names to GND IDs
        """
        # First check if it's a single result page
        if self._is_single_result_page(content):
            if self.debug:
                self.logger.debug("Detected single result page, extracting details...")
            return self._extract_details_from_single_result(content)

        # Standard extraction for result lists
        subjects = {}

        # Use regex to find all GND ID patterns with surrounding context
        # This pattern looks for the format: >Title</a>...Sachbegriff...(GND / ID)
        pattern = (
            r">\s*([^<>]+?)\s*<\/a>[^<>]*?(?:Sachbegriff)[^<>]*?\(GND\s*/\s*(\d+-\d+)\)"
        )
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            subject_name = match.group(1).strip()
            gnd_id = match.group(2).strip()

            if subject_name and gnd_id:
                subjects[subject_name] = gnd_id
                if self.debug:
                    self.logger.debug(f"Found subject: {subject_name} (GND: {gnd_id})")

        # If nothing found, try an alternative approach using BeautifulSoup
        if not subjects:
            soup = BeautifulSoup(content, "html.parser")

            # Find all entries with "Sachbegriff" text
            sachbegriff_elements = soup.find_all(string=re.compile(r"Sachbegriff"))

            for element in sachbegriff_elements:
                # Find surrounding context
                parent = element.parent

                # Look for GND ID
                gnd_match = re.search(r"GND\s*/\s*(\d+-\d+)", str(parent))
                if not gnd_match:
                    # Try looking at siblings
                    for sibling in parent.next_siblings:
                        if isinstance(sibling, str) and "GND" in sibling:
                            gnd_match = re.search(r"GND\s*/\s*(\d+-\d+)", sibling)
                            if gnd_match:
                                break

                if gnd_match:
                    gnd_id = gnd_match.group(1)

                    # Look for title in nearby links
                    link = None
                    if parent.name == "a":
                        link = parent
                    else:
                        # Search in previous siblings or parent's children
                        for prev_elem in parent.previous_siblings:
                            if prev_elem.name == "a":
                                link = prev_elem
                                break

                        if not link and parent.parent:
                            link = parent.parent.find("a")

                    if link and link.text.strip():
                        subject_name = link.text.strip()
                        subjects[subject_name] = gnd_id
                        if self.debug:
                            self.logger.debug(
                                f"Found subject via BS4: {subject_name} (GND: {gnd_id})"
                            )

        # Last resort: direct extraction from HTML with specific pattern
        if not subjects:
            # This pattern extracts the title from the Tip JavaScript function call
            js_pattern = r"Tip\('(?:<a[^>]*>)?\s*([^<>']+)\s*(?:</a>)?[^']*?Sachbegriff[^']*?GND\s*/\s*(\d+-\d+)"
            js_matches = re.finditer(js_pattern, content, re.DOTALL)

            for match in js_matches:
                subject_name = match.group(1).strip()
                gnd_id = match.group(2).strip()

                if subject_name and gnd_id:
                    subjects[subject_name] = gnd_id
                    if self.debug:
                        self.logger.debug(
                            f"Found subject via JS pattern: {subject_name} (GND: {gnd_id})"
                        )

        return subjects

    def _get_next_page_url(self, content: str, current_url: str) -> Optional[str]:
        """
        Extract URL of the next page of results if it exists.

        Args:
            content: HTML content of the current page
            current_url: URL of the current page

        Returns:
            URL of the next page or None if no next page exists
        """
        # If it's a single result page, there's no next page
        if self._is_single_result_page(content):
            return None

        soup = BeautifulSoup(content, "html.parser")

        # Direct approach: Search for the "weiter" button
        next_buttons = soup.find_all(
            "input", {"value": re.compile(r"weiter", re.IGNORECASE)}
        )
        for button in next_buttons:
            form = button.find_parent("form")
            if form and form.get("action"):
                # Form action could be the URL to the next page
                next_url = form["action"]
                if next_url:
                    return self._make_absolute_url(next_url, current_url)

        # Search for links containing "NXT?FRST=" (SWB-specific paging mechanism)
        next_links = soup.find_all("a", href=re.compile(r"NXT\?FRST="))
        for link in next_links:
            # Check if it's a "Weiter" or "Nächste Seite" link
            if "weiter" in link.text.lower() or ">" in link.text or ">>" in link.text:
                return self._make_absolute_url(link["href"], current_url)

        # Fallback: Search in the HTML code for paging links
        next_pattern = (
            r'href="([^"]*NXT\?FRST=\d+[^"]*)"[^>]*>(?:[^<]*?nächste|\s*&gt;|\s*weiter)'
        )
        match = re.search(next_pattern, content, re.IGNORECASE)
        if match:
            return self._make_absolute_url(match.group(1), current_url)

        return None

    def _make_absolute_url(self, url: str, base_url: str) -> str:
        """
        Convert relative URLs to absolute URLs.

        Args:
            url: Relative or absolute URL
            base_url: Base URL for resolving relative URLs

        Returns:
            Absolute URL
        """
        if url.startswith("http"):
            return url

        # Parse the base URL
        parsed_base = urllib.parse.urlparse(base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

        if url.startswith("/"):
            # Absolute path relative to domain
            return f"{base_domain}{url}"
        else:
            # Relative path
            path_parts = parsed_base.path.split("/")
            # Remove the last part (file/page name)
            if path_parts and path_parts[-1]:
                path_parts = path_parts[:-1]
            base_path = "/".join(path_parts)
            return f"{base_domain}{base_path}/{url}"

    def extract_gnd_from_swb(
        self, search_term: str, max_pages: int = 5, search_type: str = "kw"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract subject GND IDs from SWB for the given search term,
        scanning multiple result pages if available.

        Args:
            search_term: Term to search for
            max_pages: Maximum number of result pages to scan
            search_type: "kw" (default, IKT 2074 subject), "title" (IKT 2058 title),
                "freetext" (IKT 2072 anyword)

        Returns:
            Dictionary mapping subject names to their metadata
        """
        # Cache key includes search_type so title/kw results don't collide
        cache_key = f"{search_type}:{search_term}" if search_type != "kw" else search_term
        if cache_key in self.cache:
            if self.debug:
                self.logger.debug(f"Using cached results for '{cache_key}'")
            return self.cache[cache_key]

        # IKT codes: 2074=Sachbegriff (GND subject), 2058=Titel, 2072=Allgemeinstichwort
        ikt_map = {"kw": "2074", "title": "2058", "freetext": "2072"}
        ikt = ikt_map.get(search_type, "2074")

        # Correct URL for the subject search
        base_url = "https://swb.bsz-bw.de/DB=2.104/SET=20/TTL=1/CMD"

        # Parameters for the search query
        params = {
            "RETRACE": "0",
            "TRM_OLD": "",
            "ACT": "SRCHA",
            "IKT": ikt,
            "SRT": "RLV",
            "TRM": search_term,
            "MATCFILTER": "N",
            "MATCSET": "N",
            "NOABS": "Y",
            "SHRTST": "50",
        }

        # Build URL
        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        if self.debug:
            self.logger.debug(f"Searching SWB for subject term: {search_term}")
            self.logger.debug(f"URL: {url}")

        all_subjects = {}
        self._last_swb_status = None
        current_url = url
        page_count = 0
        had_error = False  # network/parse failure — result must not be cached - Claude Generated

        while current_url and page_count < max_pages:
            page_count += 1
            if self.debug:
                self.logger.debug(f"\nProcessing page {page_count}: {current_url}")

            try:
                response = requests.get(current_url, timeout=REQUEST_TIMEOUT_S)
                response.raise_for_status()

                # Decode HTML
                content = html.unescape(response.text)
                self._last_swb_status = response.status_code

                # Check if it's a single result page
                is_single_result = self._is_single_result_page(content)
                if self.debug and is_single_result:
                    self.logger.debug(
                        "Detected single result page - direct match for the search term"
                    )

                # Extract subjects from this page
                page_subjects = self._extract_subjects_from_page(content)
                if self.debug:
                    self.logger.debug(f"Found {len(page_subjects)} subjects on page {page_count}")

                # Add to overall results
                all_subjects.update(page_subjects)

                # For single result pages there is no next page
                if is_single_result:
                    break

                # Search for the URL of the next page
                next_url = self._get_next_page_url(content, current_url)
                if next_url:
                    if self.debug:
                        self.logger.debug(f"Found next page URL: {next_url}")
                    current_url = next_url
                    # Small pause to not overload the server
                    time.sleep(0.5)
                else:
                    if self.debug:
                        self.logger.debug(f"No more result pages found after page {page_count}")
                    break

            except Exception as e:
                had_error = True
                self._record_search_error(
                    search_term, f"page {page_count} of SWB request failed: {e}"
                )
                break

        # Prepare results in the desired format
        results = {}
        for subject_name, gnd_id in all_subjects.items():
            # IMPORTANT: Save gndid as a set to match the format of the original SubjectSuggester
            results[subject_name] = {
                "count": 1,  # Default count
                "gndid": {gnd_id},  # Save as SET - just like in the original!
                "ddc": set(),  # Empty set for DDC
                "dk": set(),  # Empty set for DK
            }

        # WP2: stash the REDUCED subject view (not the verbatim HTML pages) for the
        # provider fetch seam to dual-write. The reduced view is small (always under
        # the raw-cache size cap) and carries the swb subject titles, so a later
        # transform-on-read never depends on gnd_entries facts — unlike raw HTML,
        # which can exceed the cap and be skipped (→ silently lost swb results). Only
        # on a real (non-cached, non-failed) fetch; the file-cache hit path returns
        # before here. - Claude Generated
        if not had_error and isinstance(getattr(self, "last_raw", None), dict):
            self.last_raw[search_term] = json.dumps(
                {
                    "subjects": {
                        subj: {
                            "count": data["count"],
                            "gndid": sorted(data["gndid"]),
                            "ddc": sorted(data["ddc"]),
                            "dk": sorted(data["dk"]),
                        }
                        for subj, data in results.items()
                    },
                    "url": url,
                    "totalItems": len(results),
                },
                ensure_ascii=False,
            )
            if isinstance(getattr(self, "last_http_status", None), dict):
                self.last_http_status[search_term] = self._last_swb_status

        # Cache results under the search_type-aware key.
        # Never cache after a network/parse failure: an outage would otherwise
        # be persisted as "no results" for this term - Claude Generated
        if not had_error:
            self.cache[cache_key] = results
            self._save_cache()
        else:
            self.logger.warning(
                f"SWB results for '{search_term}' NOT cached (search failed; "
                f"{len(results)} partial subjects returned)"
            )

        return results

    def transform(self, raw: Dict[str, Any], search_type: str = "kw") -> Dict[str, Dict[str, Any]]:
        """Reduce a cached SWB response to ``{subject: {count,gndid,ddc,dk}}`` — pure.

        Two cached shapes (transform-on-read counterpart for the WP2 raw cache):
        * ``{"subjects": {...}}`` — the current compact form written by
          :meth:`extract_gnd_from_swb`; reconstruct the code ``set``s. Small, always
          cached, and carries the subject titles (no gnd_entries-fact dependency).
        * ``{"pages": [...]}`` — the legacy verbatim-HTML form; re-extract via
          :meth:`_extract_subjects_from_page` (kept for older cached rows).
        - Claude Generated
        """
        if isinstance(raw, dict) and "subjects" in raw:
            out: Dict[str, Dict[str, Any]] = {}
            for subj, data in (raw.get("subjects") or {}).items():
                data = data or {}
                out[subj] = {
                    "count": data.get("count", 1),
                    "gndid": set(data.get("gndid", [])),
                    "ddc": set(data.get("ddc", [])),
                    "dk": set(data.get("dk", [])),
                }
            return out

        all_subjects = {}
        for content in (raw or {}).get("pages", []):
            all_subjects.update(self._extract_subjects_from_page(content))
        results = {}
        for subject_name, gnd_id in all_subjects.items():
            results[subject_name] = {
                "count": 1,
                "gndid": {gnd_id},
                "ddc": set(),
                "dk": set(),
            }
        return results

    def prepare(self, force_download: bool = False) -> None:
        """
        Prepare the suggester. For SWB, this is a no-op as we don't need to download data.

        Args:
            force_download: Ignored for SWB suggester
        """
        # Nothing to do for SWB - we don't download any data in advance
        pass

    def search(
        self, searches: List[str], max_pages: int = 5, search_type: str = "kw"
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Search for subjects related to the given search terms.

        Args:
            searches: List of search terms
            max_pages: Maximum number of result pages to scan per search term
            search_type: "kw" (default), "title", or "freetext"

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
        self.last_errors = {}  # fresh error state per search call - Claude Generated
        # WP2 raw-first: verbatim page HTML per term, populated on a live fetch. - Claude Generated
        self.last_raw = {}
        self.last_http_status = {}

        for search_term in searches:
            results[search_term] = self.extract_gnd_from_swb(
                search_term, max_pages, search_type=search_type
            )

            # Signal that we've processed this term
            self.currentTerm.emit(search_term)

            if self.debug:
                self.logger.debug(
                    f"\nFound {len(results[search_term])} total subjects for '{search_term}'"
                )

                # Show some sample results
                if self.debug and results[search_term]:
                    self.logger.debug("Sample results:")
                    sample_count = min(5, len(results[search_term]))
                    for i, (subject, data) in enumerate(
                        list(results[search_term].items())[:sample_count]
                    ):
                        self.logger.debug(f"{i+1}. '{subject}': {data}")

        return results
