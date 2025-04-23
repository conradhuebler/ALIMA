#!/usr/bin/env python3
# lobid_suggester.py

import gzip
import json
import urllib.request
import urllib.parse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Union
from pprint import pprint

from .base_suggester import BaseSuggester, BaseSuggesterError

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
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None, debug: bool = False):
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
            print(f"Downloading {self.subjects_file_gz}")
            
        try:
            urllib.request.urlretrieve(self.GND_URL, self.subjects_file_gz)
        except Exception as ex:
            raise LobidSuggesterError(ex_to_str(ex))

        if self.debug:
            print(f"Extracting subjects from {self.subjects_file_gz}")
            
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
            print(f"Writing subjects to {self.subjects_file_json}")
            
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

    def _get_search_url(self, query: str) -> str:
        """
        Get the URL for the lobid.org API search.
        
        Args:
            query: URL-encoded search term
            
        Returns:
            Complete search URL
        """
        return f"https://lobid.org/resources/search?q={query}&format=json&aggregations=subject.componentList.id"

    def _get_results(self, searches: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get search results from lobid.org.
        
        Args:
            searches: List of search terms
            
        Returns:
            Dictionary with structure:
            {
                search_term: {
                    keyword: {
                        "count": int,
                        "gndid": set
                    }
                }
            }
        """
        result_subjects = dict()
        
        for search in searches:
            query = urllib.parse.quote(search)
            url = self._get_search_url(query)
            
            try:
                with urllib.request.urlopen(url) as response:
                    result = json.load(response)
            except Exception as ex:
                if self.debug:
                    print(f"Error searching for '{search}': {ex}")
                continue
                
            result_subjects[search] = dict()
            
            for entry in result.get("aggregation", {}).get("subject.componentList.id", []):
                key = entry["key"].split("/")[-1]
                try:
                    subject = self.gnd_subjects[key]
                except KeyError:
                    if self.debug:
                        print(f"No subject found for GND ID '{key}', will use '{entry['key']}'")
                    subject = entry["key"].removeprefix("https://d-nb.info/gnd/")
                except Exception as ex:
                    raise LobidSuggesterError(ex_to_str(ex))
                    
                count = entry["doc_count"]
                gnd_id = entry["key"].removeprefix("https://d-nb.info/gnd/")
                
                # Add to results, creating a new entry or updating an existing one
                if subject in result_subjects[search]:
                    result_subjects[search][subject]["gndid"].add(gnd_id)
                    # Update count if the new one is higher
                    if count > result_subjects[search][subject]["count"]:
                        result_subjects[search][subject]["count"] = count
                else:
                    result_subjects[search][subject] = {
                        "count": count,
                        "gndid": {gnd_id},
                        "ddc": set(),
                        "dk": set()
                    }
            
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
        return self._get_results(searches)
