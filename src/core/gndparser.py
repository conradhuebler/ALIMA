import xml.etree.ElementTree as ET
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any


class GNDParser:
    """
    Parser für MARC-XML-Dateien mit GND-Einträgen
    """

    def __init__(self, db_connector):
        """
        Initialisiert den GND-Parser

        Args:
            db_connector: Datenbankverbindung mit insert_gnd_entry Methode
        """
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        # XML-Namespace für MARC21
        self.ns = {"marc": "http://www.loc.gov/MARC21/slim"}

    def extract_subfield_values(self, datafield: ET.Element, code: str) -> List[str]:
        """
        Extrahiert die Werte aus Subfields mit bestimmtem Code

        Args:
            datafield: Das Datafield-Element
            code: Der Subfield-Code, nach dem gesucht wird

        Returns:
            Liste mit Werten der gefundenen Subfields
        """
        values = []
        for subfield in datafield.findall(f"./marc:subfield[@code='{code}']", self.ns):
            if subfield.text:
                values.append(subfield.text)
        return values

    def extract_single_subfield_value(
        self, datafield: ET.Element, code: str
    ) -> Optional[str]:
        """
        Extrahiert den ersten Wert aus Subfields mit bestimmtem Code

        Args:
            datafield: Das Datafield-Element
            code: Der Subfield-Code

        Returns:
            Den ersten Wert oder None
        """
        values = self.extract_subfield_values(datafield, code)
        return values[0] if values else None

    def parse_record(self, record: ET.Element) -> Dict[str, Any]:
        """
        Parst einen einzelnen MARC-XML Record und extrahiert relevante Daten

        Args:
            record: Das XML-Record-Element

        Returns:
            Dictionary mit extrahierten GND-Daten
        """
        # Standardwerte setzen
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

        # PPN aus controlfield 001
        controlfield_001 = record.find("./marc:controlfield[@tag='001']", self.ns)
        if controlfield_001 is not None and controlfield_001.text:
            data["ppn"] = controlfield_001.text

        # GND-ID aus datafield 024
        for datafield_024 in record.findall("./marc:datafield[@tag='024']", self.ns):
            if datafield_024.get("ind1") == "7":
                gnd_id = self.extract_single_subfield_value(datafield_024, "a")
                if gnd_id:
                    data["gnd_id"] = gnd_id
                    break

        # Titel aus datafield 150
        datafield_150 = record.find("./marc:datafield[@tag='150']", self.ns)
        if datafield_150 is not None:
            title = self.extract_single_subfield_value(datafield_150, "a")
            if title:
                data["title"] = title

        # Synonyme aus datafield 450
        synonyms = []
        for datafield_450 in record.findall("./marc:datafield[@tag='450']", self.ns):
            synonym = self.extract_single_subfield_value(datafield_450, "a")
            if synonym:
                synonyms.append(synonym)
        data["synonyms"] = "|".join(synonyms)

        # DDC-Klassifikation aus datafield 083
        ddcs = []
        for datafield_083 in record.findall("./marc:datafield[@tag='083']", self.ns):
            ddc = self.extract_single_subfield_value(datafield_083, "a")
            if ddc:
                ddcs.append(ddc)
        data["ddcs"] = "|".join(ddcs)

        # GND-Systematik aus datafield 065
        gnd_systems = []
        for datafield_065 in record.findall("./marc:datafield[@tag='065']", self.ns):
            gnd_system = self.extract_single_subfield_value(datafield_065, "a")
            if gnd_system:
                gnd_systems.append(gnd_system)
        data["gnd_systems"] = "|".join(gnd_systems)

        # Beschreibung aus datafield 670
        descriptions = []
        for datafield_670 in record.findall("./marc:datafield[@tag='670']", self.ns):
            desc = self.extract_single_subfield_value(datafield_670, "a")
            if desc:
                descriptions.append(desc)
        data["description"] = "|".join(descriptions)

        return data

    def process_file(self, file_path: str) -> int:
        """
        Verarbeitet eine MARC-XML-Datei und speichert Einträge in der Datenbank

        Args:
            file_path: Pfad zur MARC-XML-Datei

        Returns:
            Anzahl der verarbeiteten Einträge
        """
        count = 0
        try:
            # Namespace registrieren
            ET.register_namespace("", "http://www.loc.gov/MARC21/slim")

            tree = ET.parse(file_path)
            root = tree.getroot()

            for record in root.findall(".//marc:record", self.ns):
                try:
                    data = self.parse_record(record)

                    # Nur Einträge mit gültiger GND-ID speichern
                    if data["gnd_id"] and data["title"]:
                        self.db.insert_gnd_entry(
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
                    self.logger.error(f"Fehler beim Verarbeiten eines Records: {e}")

            self.logger.info(
                f"Verarbeitung von {file_path} abgeschlossen. {count} Einträge importiert."
            )
            self.db.save_to_file()
            return count

        except Exception as e:
            self.logger.error(f"Fehler beim Verarbeiten der Datei {file_path}: {e}")
            return count

    def process_directory(self, directory_path: str) -> int:
        """
        Verarbeitet alle XML-Dateien in einem Verzeichnis

        Args:
            directory_path: Pfad zum Verzeichnis mit MARC-XML-Dateien

        Returns:
            Gesamtanzahl der verarbeiteten Einträge
        """
        total_count = 0
        for filename in os.listdir(directory_path):
            if filename.endswith(".xml"):
                file_path = os.path.join(directory_path, filename)
                count = self.process_file(file_path)
                total_count += count

        self.logger.info(
            f"Gesamtimport abgeschlossen. {total_count} Einträge importiert."
        )
        return total_count
