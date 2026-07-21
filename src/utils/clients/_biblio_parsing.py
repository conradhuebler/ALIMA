"""MARC/MAB response parsing for BiblioClient - Claude Generated.

Split out of ``biblio_client.py`` (WP cleanup D). Verbatim mixin extraction: the
methods stay on ``BiblioClient`` via MRO, so no call site changes.

These are the near-pure parsers — MARC-XML → authors / MAB subjects / DK / RVK,
plus SOAP-fault and debug-dump helpers. They hold no transport state (no rate
limit, circuit breaker or session), so they read as a distinct layer from the
Libero SOAP/web plumbing they were interleaved with. The two that touch ``self``
only read a config attribute (``save_xml_path``, ``MAB_SUBJECT_TAGS``), which
stays resolvable via ``self`` — one class across two files.

Note ``logger`` is the named ``biblio_extractor`` logger (NOT ``__name__``), so
these lines keep landing where they did. The ``datetime`` import is method-local
and travels with the code.
"""

from __future__ import annotations

import base64
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("biblio_extractor")


class BiblioParsingMixin:
    """MARC/MAB parsers. Mixed into :class:`BiblioClient`."""

    def _print_xml_structure(self, element, level: int = 0):
        """Print XML structure for debugging - Claude Generated"""
        indent = "  " * level
        logger.debug(f"{indent}{element.tag}")
        for child in element:
            self._print_xml_structure(child, level + 1)

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
