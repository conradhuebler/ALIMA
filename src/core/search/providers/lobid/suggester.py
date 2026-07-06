#!/usr/bin/env python3
# lobid_suggester.py

import gzip
import json
import shutil
import urllib.request
import urllib.parse
import ssl
import platform
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Union
from pprint import pprint

from src.utils.suggesters.base_suggester import BaseSuggester, BaseSuggesterError


def ex_to_str(ex):
    """Converts an exception to a human readable string containing relevant informations."""
    # get type and message of risen exception
    ex_type = f"{type(ex).__name__}"
    ex_args = ", ".join(map(str, ex.args))

    # get the command where the exception has been raised
    tb = traceback.extract_tb(sys.exc_info()[2], limit=2)
    ex_cmd = tb[0][3]
    ex_file = tb[0][0]
    ex_line = tb[0][1]

    # the string (one liner) to return
    nice_ex = f"{ex_type} ({ex_args}) raised executing '{ex_cmd}' in {ex_file}, line {ex_line}"
    return nice_ex


def _create_ssl_context():
    """
    Create SSL context for urllib with proper certificate handling - Claude Generated

    On Windows, Python often can't access system certificates. This function
    tries multiple strategies to establish a working SSL context.

    Returns:
        SSL context for HTTPS connections
    """
    # Try certifi package first (most reliable cross-platform solution)
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        return context
    except ImportError:
        pass

    # On Windows, use unverified context as fallback (with warning)
    if platform.system() == 'Windows':
        import warnings
        warnings.warn(
            "SSL certificate verification disabled on Windows. "
            "Install 'certifi' package for secure connections: pip install certifi",
            RuntimeWarning
        )
        return ssl._create_unverified_context()

    # On Linux/Mac, use default (should work with system certificates)
    return ssl.create_default_context()


class LobidSuggesterError(BaseSuggesterError):
    """Exception raised for errors in the LobidSuggester."""

    pass


class LobidSuggester(BaseSuggester):
    """
    Subject suggester that uses the lobid.org API to find relevant keywords.
    Downloads and uses a GND (Gemeinsame Normdatei) dump for subject data.
    """

    # URL for the GND subjects file
    GND_URL = "https://data.dnb.de/opendata/authorities-gnd-sachbegriff_lds.jsonld.gz"

    def __init__(
        self, data_dir: Optional[Union[str, Path]] = None, debug: bool = False
    ):
        """
        Initialize the LobidSuggester.

        Args:
            data_dir: Directory to store GND data files (default: script_dir/data/lobidsuggestor)
            debug: Whether to enable debug output
        """
        super().__init__(data_dir, debug)

        # File paths for the GND data
        self.subjects_file_gz = self.data_dir / self.GND_URL.split("/")[-1]
        self.subjects_file_json = self.data_dir / "subjects.json"
        self.gnd_subjects = None

        # Prepare the suggester
        self.prepare(False)

    def _create_subjects_file_from_gnd(self):
        """Download and process the GND subjects file."""
        gnd_subjects = dict()

        if self.debug:
            self.logger.debug(f"Downloading {self.subjects_file_gz}")

        try:
            # Create SSL context for cross-platform certificate handling - Claude Generated
            ssl_context = _create_ssl_context()

            # Use opener with SSL context for HTTPS
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
            urllib.request.install_opener(opener)

            # Now download with proper SSL handling; explicit socket timeout so a
            # stalled dump download can not hang forever. - Claude Generated
            with opener.open(self.GND_URL, timeout=60) as response, open(
                self.subjects_file_gz, "wb"
            ) as out:
                shutil.copyfileobj(response, out)
        except Exception as ex:
            raise LobidSuggesterError(ex_to_str(ex))

        if self.debug:
            self.logger.debug(f"Extracting subjects from {self.subjects_file_gz}")

        try:
            with open(self.subjects_file_gz, "rb") as fh:
                data = json.loads(gzip.decompress(fh.read()).decode("utf-8"))
        except Exception as ex:
            raise LobidSuggesterError(ex_to_str(ex))

        for parts in data:
            for entry in parts:
                if not isinstance(entry, dict):
                    continue
                try:
                    key = entry["@id"].split("/")[-1]
                    value = entry[
                        "https://d-nb.info/standards/elementset/gnd#preferredNameForTheSubjectHeading"
                    ][0]["@value"]
                    gnd_subjects[key] = value
                except KeyError:
                    continue
                except Exception as ex:
                    raise LobidSuggesterError(ex_to_str(ex))

        if self.debug:
            self.logger.debug(f"Writing subjects to {self.subjects_file_json}")

        try:
            with open(self.subjects_file_json, "w", encoding="utf-8") as fh:
                json.dump(gnd_subjects, fh)
        except Exception as ex:
            raise LobidSuggesterError(ex_to_str(ex))

    def _get_gnd_subjects(self):
        """Load GND subjects from the JSON file."""
        try:
            with open(self.subjects_file_json, "r", encoding="utf-8") as fh:
                subjects = json.load(fh)
        except Exception as ex:
            raise LobidSuggesterError(ex_to_str(ex))

        return subjects

    def _get_search_url(self, query: str, search_type: str = "kw") -> str:
        """
        Get the URL for the lobid.org API search.

        Args:
            query: URL-encoded search term
            search_type: "kw" (default, any-field), "title" (titleAll:), "freetext" (any)

        Returns:
            Complete search URL
        """
        if search_type == "title":
            q = f"title:{query}"
        else:
            q = query
        return f"https://lobid.org/resources/search?q={q}&format=json&aggregations=subject.componentList.id"

    def fetch(self, query: str, search_type: str = "kw") -> Dict[str, Any]:
        """Fetch the verbatim lobid response for one term — I/O only, no transform.

        Records the HTTP status in ``_last_fetch_status``. Raises on
        network/parse error; callers turn that into a per-term source failure.
        Split out of ``_get_results`` for the WP2 raw-first cache. - Claude Generated
        """
        encoded = urllib.parse.quote(query)
        url = self._get_search_url(encoded, search_type=search_type)
        with urllib.request.urlopen(url, timeout=30) as response:
            result = json.load(response)
            self._last_fetch_status = getattr(response, "status", None)
        return result

    def transform(self, raw: Dict[str, Any], search_type: str = "kw") -> Dict[str, Dict[str, Any]]:
        """Reduce a raw lobid response to the ``{subject: {count,gndid,ddc,dk}}`` view.

        Pure (no I/O). This is the exact former parsing body — the reduced output
        is byte-identical to the pre-split behaviour (locked by a regression
        test). ``search_type`` only affected the URL (in :meth:`fetch`); it is
        accepted here for a uniform transform signature. - Claude Generated
        """
        subjects: Dict[str, Dict[str, Any]] = {}
        for entry in raw.get("aggregation", {}).get("subject.componentList.id", []):
            key = entry["key"].split("/")[-1]
            try:
                subject = self.gnd_subjects[key]
            except KeyError:
                if self.debug:
                    self.logger.debug(
                        f"No subject found for GND ID '{key}', will use '{entry['key']}'"
                    )
                subject = entry["key"].removeprefix("https://d-nb.info/gnd/")
            except Exception as ex:
                raise LobidSuggesterError(ex_to_str(ex))

            count = entry["doc_count"]
            gnd_id = entry["key"].removeprefix("https://d-nb.info/gnd/")

            # Add to results, creating a new entry or updating an existing one
            if subject in subjects:
                subjects[subject]["gndid"].add(gnd_id)
                # Update count if the new one is higher
                if count > subjects[subject]["count"]:
                    subjects[subject]["count"] = count
            else:
                subjects[subject] = {
                    "count": count,
                    "gndid": {gnd_id},
                    "ddc": set(),
                    "dk": set(),
                }
        return subjects

    @staticmethod
    def transform_agent_view(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Full agent-facing view: the data the reduced pool view discards.

        ``totalItems`` + the ``member`` resource records that the
        aggregation-only pool transform drops (WP2 tool-data passthrough).
        - Claude Generated
        """
        return {
            "totalItems": raw.get("totalItems"),
            "member": raw.get("member", []),
        }

    def _get_results(self, searches: List[str], search_type: str = "kw") -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Fetch + transform each term; capture the verbatim response for the raw cache.

        Composition of :meth:`fetch` (I/O) and :meth:`transform` (pure); the
        public reduced output is unchanged. Result structure::

            {search_term: {keyword: {"count": int, "gndid": set,
                                     "ddc": set, "dk": set}}}
        - Claude Generated
        """
        result_subjects = dict()
        self.last_errors = {}  # fresh error state per search call - Claude Generated
        # WP2 raw-first: verbatim response per term, captured before transform. - Claude Generated
        self.last_raw = {}
        self.last_http_status = {}

        for search in searches:
            try:
                raw = self.fetch(search, search_type=search_type)
            except Exception as ex:
                # Missing term in the result dict = source failure, not "no match" - Claude Generated
                self._record_search_error(search, ex)
                continue

            self.last_raw[search] = json.dumps(raw, ensure_ascii=False)
            self.last_http_status[search] = self._last_fetch_status
            result_subjects[search] = self.transform(raw, search_type=search_type)

            # Signal that we've processed this term
            self.currentTerm.emit(search)

        return result_subjects

    def prepare(self, force_gnd_download: bool = False) -> None:
        """
        Prepare the suggester by downloading and loading GND data.

        Args:
            force_gnd_download: Whether to force download of GND data even if already available
        """
        if (
            force_gnd_download
            or not self.subjects_file_gz.exists()
            or not self.subjects_file_json.exists()
        ):
            self._create_subjects_file_from_gnd()

        self.gnd_subjects = self._get_gnd_subjects()

    def search(self, searches: List[str], search_type: str = "kw") -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Search for subjects related to the given search terms.

        Args:
            searches: List of search terms
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
        return self._get_results(searches, search_type=search_type)
