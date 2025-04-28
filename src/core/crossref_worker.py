# crossref_worker.py

import json
import requests
from PyQt6.QtCore import pyqtSignal, QThread


class CrossrefWorker(QThread):
    # Signale zur Kommunikation mit dem GUI
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, doi: str):
        super().__init__()
        self.doi = doi

    def run(self):
        """Führt die API-Anfrage aus."""
        url = f"https://api.crossref.org/works/{self.doi}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                self.error_occurred.emit(
                    f"API-Anfrage fehlgeschlagen: Statuscode {response.status_code}"
                )
                return

            data = response.json()
            if data.get("status") != "ok":
                self.error_occurred.emit(
                    f"API-Anfrage nicht erfolgreich: {data.get('status')}"
                )
                return

            message = data.get("message", {})
            # Extrahiere relevante Informationen
            result = {
                "Title": " | ".join(message.get("title", [])),
                "DOI": message.get("DOI", "Nicht verfügbar"),
                "Abstract": self._clean_jats(
                    message.get("abstract", "Kein Abstract verfügbar")
                ),
                "Authors": self._format_authors(message.get("author", [])),
                "Publisher": message.get("publisher", "Nicht verfügbar"),
                "Published": self._format_date(
                    message.get("published-print", message.get("published-online", {}))
                ),
                "Container-Title": " | ".join(message.get("container-title", [])),
                "URL": message.get("URL", "Nicht verfügbar"),
            }

            self.result_ready.emit(result)

        except requests.RequestException as e:
            self.error_occurred.emit(f"Netzwerkfehler: {str(e)}")
        except json.JSONDecodeError:
            self.error_occurred.emit("Fehler beim Parsen der API-Antwort.")

    def _format_authors(self, authors_list):
        """Formatiert die Autorenliste."""
        authors = []
        for author in authors_list:
            given = author.get("given", "")
            family = author.get("family", "")
            if given or family:
                authors.append(f"{given} {family}".strip())
        return ", ".join(authors) if authors else "Nicht verfügbar"

    def _format_date(self, date_dict):
        """Formatiert das Veröffentlichungsdatum."""
        date_parts = date_dict.get("date-parts", [[]])
        if date_parts and len(date_parts[0]) >= 3:
            year, month, day = date_parts[0]
            return f"{year}-{month:02}-{day:02}"
        elif date_parts and len(date_parts[0]) == 2:
            year, month = date_parts[0]
            return f"{year}-{month:02}"
        elif date_parts and len(date_parts[0]) == 1:
            return f"{date_parts[0][0]}"
        return "Nicht verfügbar"

    def _clean_jats(self, abstract: str) -> str:
        """Entfernt JATS-Tags aus dem Abstract."""
        import re

        if not abstract:
            return "Kein Abstract verfügbar"
        # Entferne einfache Tags
        clean_text = re.sub(r"<[^>]+>", "", abstract)
        return clean_text.strip()
