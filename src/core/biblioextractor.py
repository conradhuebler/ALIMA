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

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("biblio_extractor")


class BiblioExtractor:
    """
    A tool to extract keywords and decimal classifications from a library catalog.
    """

    SEARCH_URL = "https://libero.ub.tu-freiberg.de:443/libero/LiberoWebServices.CatalogueSearcher.cls"
    DETAILS_URL = (
        "https://libero.ub.tu-freiberg.de:443/libero/LiberoWebServices.LibraryAPI.cls"
    )

    # MAB-Tags für Schlagwörter
    MAB_SUBJECT_TAGS = ["0902", "0907", "0912", "0917", "0922", "0927"]

    def __init__(self, token: str = "PCXBE6tJgV5428", debug: bool = False):
        """
        Initialize the extractor with the given token.

        Args:
            token: The authentication token for the library API
            debug: Enable detailed debug output
        """
        self.token = token
        self.debug = debug
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

            if response.status_code != 200:
                logger.error(
                    f"Error getting details: {response.status_code} - {response.text}"
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

            # Extract details
            details = {}

            # Extract MAB-based subjects from the MAB tags
            mab_subjects = self._extract_mab_subjects(root)
            if mab_subjects:
                logger.debug(f"Found {len(mab_subjects)} MAB subjects")
                details["mab_subjects"] = mab_subjects

            # Register namespaces
            namespaces = {
                "soap": "http://schemas.xmlsoap.org/soap/envelope/",
                "lib": "http://libero.com.au",
            }

            # Try different possible paths for classifications
            classification_paths = [
                ".//Classification/Classifications/Classification",
                ".//{http://libero.com.au}Classification/{http://libero.com.au}Classifications/{http://libero.com.au}Classification",
            ]

            classifications = []
            for path in classification_paths:
                logger.debug(f"Trying classification path: {path}")
                for classification in root.findall(path):
                    if classification.text:
                        classifications.append(classification.text)
                        logger.debug(f"Found classification: {classification.text}")

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
                logger.debug(f"Trying subject path: {path}")
                for subject in root.findall(path):
                    if subject.text:
                        subjects.append(subject.text)
                        logger.debug(f"Found subject: {subject.text}")

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
            logger.error(f"XML parsing error: {e}")
            if hasattr(response, "text"):
                logger.error(f"Response content: {response.text}")
            return None

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

    def extract_decimal_classifications(self, classifications: List[str]) -> List[str]:
        """
        Extract decimal classifications from a list of classifications.

        Args:
            classifications: List of classification strings

        Returns:
            List of extracted decimal classifications
        """
        decimal_classes = []
        for classification in classifications:
            logger.debug(f"Processing classification: {classification}")

            # Look for patterns like "DK 543.42" or "543/.62"
            if "DK " in classification:
                # Extract the number after "DK "
                match = re.search(r"DK\s+(\d+(?:\.\d+)?)", classification)
                if match:
                    decimal_class = match.group(1)
                    decimal_classes.append(decimal_class)
                    logger.debug(f"Extracted DK classification: {decimal_class}")
            elif "/" in classification:
                # Handle patterns like "543/.62"
                parts = classification.split("/")
                if parts[0].strip().isdigit() and parts[1].strip().startswith("."):
                    decimal_class = f"{parts[0].strip()}{parts[1].strip()}"
                    decimal_classes.append(decimal_class)
                    logger.debug(f"Extracted decimal classification: {decimal_class}")

        return decimal_classes

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

            # Add decimal classifications
            details["decimal_classifications"] = self.extract_decimal_classifications(
                details.get("classifications", [])
            )

            processed_items.append(details)

        return processed_items

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
        "--token", default="PCXBE6tJgV5428", help="API token (default: PCXBE6tJgV5428)"
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

    extractor = BiblioExtractor(token=args.token, debug=args.debug)

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
