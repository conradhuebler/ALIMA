#!/usr/bin/env python3
from PyQt6.QtCore import QObject, QUrl, QEventLoop, pyqtSignal

import gzip
import json
import urllib.request
import urllib.parse
from pathlib import Path
from pprint import pprint
import sys
import traceback


################################################################################
def dd(*args):
    """Helper to dump variables and die (inspired by laravel)"""
    for arg in args:
        print()
        print(40 * "=")
        print(f"Type: {type(arg)}")
        pprint(arg)
    sys.exit(1)


################################################################################
def ex_to_str(ex):
    """Converts an exception to a human readable string containing relevant informations.
    Use e.g. to create meaningful log entries."""

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


################################################################################
################################################################################
class SubjectSuggesterError(Exception):
    pass


################################################################################
################################################################################
class SubjectSuggester(QObject):

    gnd_url = "https://data.dnb.de/opendata/authorities-gnd-sachbegriff_lds.jsonld.gz"

    subjects_file_gz = None
    subjects_file_json = None

    data_dir = None
    gnd_subjects = None
    currentTerm = pyqtSignal(str)

    ############################################################################
    def __init__(self, data_dir: str | Path = "") -> None:
        super().__init__()
        self.data_dir = self._get_data_dir(data_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            raise SubjectSuggesterError(ex_to_str(ex))

        self.subjects_file_gz = self.data_dir / self.gnd_url.split("/")[-1]
        self.subjects_file_json = self.data_dir / "subjects.json"
        self.prepare(False)

    ############################################################################
    def _get_data_dir(self, data_dir: str | Path = "") -> Path:

        if not data_dir:
            return Path(sys.argv[0]).parent.resolve() / "data"

        if isinstance(data_dir, str):
            return Path(data_dir)

        if isinstance(data_dir, Path):
            return data_dir

        raise SubjectSuggesterError(
            "Given data_dir is neither string nor Path. Cannot proceed"
        )

    ############################################################################
    def _create_subjects_file_from_gnd(self):

        gnd_subjects = dict()

        print(f"Downloading {self.subjects_file_gz}")
        try:
            urllib.request.urlretrieve(self.gnd_url, self.subjects_file_gz)
        except Exception as ex:
            raise SubjectSuggesterError(ex_to_str(ex))

        print(f"Extracting subjects from {self.subjects_file_gz}")
        try:
            with open(self.subjects_file_gz, "rb") as fh:
                data = json.loads(gzip.decompress(fh.read()).decode("utf-8"))
        except Exception as ex:
            raise SubjectSuggesterError(ex_to_str(ex))

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
                except KeyError as ex:
                    continue
                except Exception as ex:
                    raise SubjectSuggesterError(ex_to_str(ex))

        print(f"Writing subjects from {self.subjects_file_json}")
        try:
            with open(self.subjects_file_json, "w", encoding="utf-8") as fh:
                json.dump(gnd_subjects, fh)
        except Exception as ex:
            raise SubjectSuggesterError(ex_to_str(ex))

    ############################################################################
    def _get_gnd_subjects(self):
        """Downloads GND subjects and prepares them for later usage."""

        try:
            with open(self.subjects_file_json, "r", encoding="utf-8") as fh:
                subjects = json.load(fh)
        except Exception as ex:
            raise SubjectSuggesterError(ex_to_str(ex))

        return subjects

    ############################################################################
    def _get_search_url(self, query):
        return f"https://lobid.org/resources/search?q={query}&format=json&aggregations=subject.componentList.id"

    ############################################################################
    def _get_results(self, searches: list) -> dict:
        result_subjects = dict()
        for search in searches:
            query = urllib.parse.quote(search)
            url = self._get_search_url(query)
            with urllib.request.urlopen(url) as response:
                result = json.load(response)
            result_subjects[search] = dict()
            for entry in result["aggregation"]["subject.componentList.id"]:
                key = entry["key"].split("/")[-1]
                try:
                    subject = self.gnd_subjects[key]
                except KeyError:
                    print(
                        f"No subject found for GND ID “{key}”, will use “{entry['key']}”)"
                    )
                    subject = entry["key"].removeprefix("https://d-nb.info/gnd/")
                except Exception as ex:
                    raise SubjectSuggesterError(ex_to_str(ex))
                count = entry["doc_count"]
                result_subjects[search][subject] = {
                    "count": count,
                    "gndid": {entry["key"].removeprefix("https://d-nb.info/gnd/")},
                }
            self.currentTerm.emit(search)

        return result_subjects

    ############################################################################
    def prepare(self, force_gnd_download: bool = False) -> None:
        if (
            force_gnd_download
            or not self.subjects_file_gz.exists()
            or not self.subjects_file_json.exists()
        ):
            self._create_subjects_file_from_gnd()
        self.gnd_subjects = self._get_gnd_subjects()

    ############################################################################
    def search(self, searches: list) -> dict:
        return self._get_results(searches)


################################################################################
################################################################################
if __name__ == "__main__":

    # better debug when called from console; needs
    # pip install stackprinter
    # import stackprinter
    # stackprinter.set_excepthook(style="darkbg")

    # read comma-separated search terms from command line
    if len(sys.argv) < 2:
        print(f"Usage {sys.argv[0]} SEARCH_PHRASES, e.g.")
        print(f"    {sys.argv[0]} Differentialgleichung")
        print(
            f'    {sys.argv[0]} "Orbitmethode, Lie-Gruppe, glatte Mannigfaltigkeiten"'
        )
        print()
        sys.exit(1)

    searches = [s.strip() for s in sys.argv[1].split(",")]

    suggestor = SubjectSuggester()
    # suggestor.prepare(True)   # call this to force a gnd file download
    results = suggestor.search(searches)
    pprint(results)
