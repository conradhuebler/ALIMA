#!/usr/bin/env python3
"""
GND Database Import Worker - Non-blocking background import with progress tracking
Claude Generated
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional

logger = logging.getLogger(__name__)


class GNDImportWorker(QThread):
    """Worker thread for non-blocking GND database import - Claude Generated"""

    # Signals
    progress_updated = pyqtSignal(int, int)  # current_count, total_records
    status_updated = pyqtSignal(str)  # status message
    finished_successfully = pyqtSignal(int)  # total_imported_count
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, xml_file_path: str, cache_manager, batch_size: int = 100):
        """
        Initialize GND import worker

        Args:
            xml_file_path: Path to MARC-XML file
            cache_manager: Database connection with insert_gnd_entry method
            batch_size: Number of records between progress emissions
        """
        super().__init__()
        self.xml_file_path = xml_file_path
        self.cache_manager = cache_manager
        self.batch_size = batch_size
        self._cancelled = False

        # XML namespace for MARC21
        self.ns = {"marc": "http://www.loc.gov/MARC21/slim"}

    def cancel(self):
        """Request cancellation of import - Claude Generated"""
        self._cancelled = True
        logger.info("GND import cancellation requested")

    def run(self):
        """Execute import in background thread - Claude Generated"""
        try:
            # Validate file exists
            if not Path(self.xml_file_path).exists():
                self.error_occurred.emit(f"File not found: {self.xml_file_path}")
                return

            self.status_updated.emit(f"📂 Parsing {Path(self.xml_file_path).name}...")

            # Parse XML to get total records
            try:
                tree = ET.parse(self.xml_file_path)
                root = tree.getroot()
                records = root.findall(".//marc:record", self.ns)
                total_records = len(records)
            except ET.ParseError as e:
                self.error_occurred.emit(f"Invalid XML file: {str(e)}")
                return

            if total_records == 0:
                self.error_occurred.emit("No records found in XML file")
                return

            self.status_updated.emit(
                f"📊 Found {total_records:,} records to import..."
            )

            # Import records
            count = 0
            error_count = 0

            for i, record in enumerate(records):
                # Check cancellation
                if self._cancelled:
                    self.status_updated.emit(
                        f"⏹️  Import cancelled. Saved {count:,} records."
                    )
                    self.finished_successfully.emit(count)
                    return

                try:
                    # Parse record
                    data = self._parse_record(record)

                    # Only save valid entries
                    if data["gnd_id"] and data["title"]:
                        self.cache_manager.insert_gnd_entry(
                            gnd_id=data["gnd_id"],
                            title=data["title"],
                            description=data["description"],
                            ddcs=data["ddcs"],
                            dks=data["dks"],
                            gnd_systems=data["gnd_systems"],
                            synonyms=data["synonyms"],
                            classification=data["classification"],
                            ppn=data["ppn"],
                        )
                        count += 1

                except Exception as e:
                    error_count += 1
                    if error_count < 10:  # Only log first 10 errors to avoid spam
                        logger.warning(f"Error parsing record {i}: {str(e)}")

                # Emit progress every batch_size records
                if i % self.batch_size == 0:
                    self.progress_updated.emit(i, total_records)

            # Save database
            try:
                self.cache_manager.save_to_file()
            except Exception as e:
                logger.warning(f"Error saving database: {str(e)}")

            logger.info(
                f"GND import completed: {count:,} entries imported "
                f"({error_count} errors skipped)"
            )

            self.finished_successfully.emit(count)

        except Exception as e:
            logger.error(f"Unexpected error during GND import: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Import failed: {str(e)}")

    def _parse_record(self, record) -> dict:
        """Parse a single MARC-XML record - Claude Generated

        Args:
            record: XML Element for MARC record

        Returns:
            Dictionary with extracted GND data
        """
        data = {
            "gnd_id": "",
            "title": "",
            "description": "",
            "ddcs": "",
            "dks": "",
            "gnd_systems": "",
            "synonyms": "",
            "classification": "",
            "ppn": "",
        }

        # Extract GND-ID from 035$a (Normierter Normdatensatz)
        for field in record.findall(".//marc:datafield[@tag='035']", self.ns):
            for subfield in field.findall(f"./marc:subfield[@code='a']", self.ns):
                if subfield.text and "(DE-588)" in subfield.text:
                    data["gnd_id"] = subfield.text.replace("(DE-588)", "").strip()
                    break

        # Extract title/preferred name from 150$a (Schlagwort Sagenkunde)
        for field in record.findall(".//marc:datafield[@tag='150']", self.ns):
            for subfield in field.findall(f"./marc:subfield[@code='a']", self.ns):
                if subfield.text:
                    data["title"] = subfield.text
                    break

        # Extract description from 667$a (Non-public note)
        descriptions = []
        for field in record.findall(".//marc:datafield[@tag='667']", self.ns):
            for subfield in field.findall(f"./marc:subfield[@code='a']", self.ns):
                if subfield.text:
                    descriptions.append(subfield.text)
        data["description"] = "; ".join(descriptions)

        # Extract DDC from 082$a (Dewey Decimal Classification)
        ddcs = []
        for field in record.findall(".//marc:datafield[@tag='082']", self.ns):
            for subfield in field.findall(f"./marc:subfield[@code='a']", self.ns):
                if subfield.text:
                    ddcs.append(subfield.text)
        data["ddcs"] = "; ".join(ddcs)

        # Extract DK from 083$a (Dewey Classification Number)
        dks = []
        for field in record.findall(".//marc:datafield[@tag='083']", self.ns):
            for subfield in field.findall(f"./marc:subfield[@code='a']", self.ns):
                if subfield.text:
                    dks.append(subfield.text)
        data["dks"] = "; ".join(dks)

        # Extract synonyms from 430$a (Variant name for corporate name)
        synonyms = []
        for field in record.findall(".//marc:datafield[@tag='430']", self.ns):
            for subfield in field.findall(f"./marc:subfield[@code='a']", self.ns):
                if subfield.text:
                    synonyms.append(subfield.text)
        data["synonyms"] = "; ".join(synonyms)

        # Extract PPN from 001 (Control number)
        for field in record.findall(".//marc:controlfield[@tag='001']", self.ns):
            if field.text:
                data["ppn"] = field.text

        return data
