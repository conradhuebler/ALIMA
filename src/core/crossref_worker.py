# crossref_worker.py


import json
import re
import requests
import asyncio
from PyQt6.QtCore import pyqtSignal, QThread

# from crawl4ai import AsyncWebCrawler  # Temporarily disabled


class CrossrefWorker(QThread):
    # Signale zur Kommunikation mit dem GUI
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, doi: str):
        super().__init__()
        self.doi = doi

    def run(self):
        """Führt die API-Anfrage aus."""
        # Prüfe, ob die DOI von Springer ist (beginnt mit 10.1007)
        if self.doi.startswith("10.1007"):
            self._handle_springer_doi()
        else:
            self._handle_crossref_doi()

    def _handle_springer_doi(self):
        """Verarbeitet DOIs von Springer direkt über die Springer-Website."""
        try:
            # Erstelle einen Event Loop für asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Führe die asynchrone Funktion aus
            springer_data = loop.run_until_complete(self._crawl_springer_site())

            # Schließe den Event Loop
            loop.close()

            # Verarbeite die geparsten Daten
            if springer_data:
                self.result_ready.emit(springer_data)
            else:
                self.error_occurred.emit(
                    f"Keine Daten von der Springer-Website erhalten für DOI: {self.doi}"
                )

        except Exception as e:
            self.error_occurred.emit(
                f"Fehler beim Parsen der Springer-Website: {str(e)}"
            )

    async def _crawl_springer_site(self):
        """Crawlt die Springer-Website für die gegebene DOI."""
        url = f"https://link.springer.com/book/{self.doi}"

        try:
            # Erstelle eine Instanz von AsyncWebCrawler
            async with AsyncWebCrawler() as crawler:
                # Führe den Crawler auf der URL aus
                result = await crawler.arun(url=url)

                # Parse das Markdown
                return self._parse_springer_markdown(result.markdown, url)

        except Exception as e:
            raise Exception(f"Fehler beim Crawlen der Springer-Website: {str(e)}")

    def _parse_springer_markdown(self, markdown_text, url):
        """Extrahiert relevante Informationen aus dem Markdown der Springer-Website."""
        # Extrahiere den Titel (zwischen # und dem nächsten Zeilenumbruch)
        title_match = re.search(r"Book Title:\s+(.+?)(?:\n|$)", markdown_text)
        if not title_match:
            # Fallback: Versuche den ursprünglichen Titel-Pattern
            title_match = re.search(r"#\s+(.*?)$", markdown_text, re.MULTILINE)

        title = title_match.group(1).strip() if title_match else "Nicht verfügbar"

        # Extrahiere den "About this book" Abschnitt
        about_match = re.search(
            r"## About this book\s+(.*?)(?=## Keywords)", markdown_text, re.DOTALL
        )
        about = about_match.group(1).strip() if about_match else "Nicht verfügbar"

        # Extrahiere das Inhaltsverzeichnis
        toc_match = re.search(
            r"## Table of contents.*?\n(.*?)(?=Back to top)", markdown_text, re.DOTALL
        )

        if toc_match:
            raw_toc = toc_match.group(1).strip()
            # Bereinige das Inhaltsverzeichnis
            toc = self._clean_table_of_contents(raw_toc)
        else:
            toc = "Nicht verfügbar"

        # Extrahiere die Keywords
        keywords_match = re.search(
            r"## Keywords\s+(.*?)(?=Search within this book)", markdown_text, re.DOTALL
        )
        if keywords_match:
            raw_keywords = keywords_match.group(1).strip()
            keywords = self._clean_keywords(raw_keywords)
        else:
            keywords = "Nicht verfügbar"

        # Extrahiere den Abstract
        # Da der Abstract im Beispiel nicht klar identifizierbar ist, nehmen wir einen Teil des "About this book" Abschnitts
        abstract = about  # [:500] + "..." if len(about) > 500 else about

        # Extrahiere die Autoren/Herausgeber
        authors_match = re.search(
            r"Editors?:\s+(.*?)(?=\n\n)", markdown_text, re.DOTALL
        )
        authors = authors_match.group(1).strip() if authors_match else "Nicht verfügbar"
        authors = re.sub(
            r"$$|$$|\(|\)|https?://[^\s]+", "", authors
        ).strip()  # Entferne Links und Klammern

        # Extrahiere den Verlag
        publisher_match = re.search(r"Publisher:\s+(.*?)(?=\n)", markdown_text)
        publisher = publisher_match.group(1).strip() if publisher_match else "Springer"

        # Extrahiere das Veröffentlichungsdatum
        date_match = re.search(r"Published:\s+(\d{1,2}\s+\w+\s+\d{4})", markdown_text)
        published_date = (
            date_match.group(1).strip() if date_match else "Nicht verfügbar"
        )

        # Baue das Ergebnispaket
        result = {
            "Title": title,
            "DOI": self.doi,
            "Abstract": abstract,
            "Authors": authors,
            "Publisher": publisher,
            "Published": published_date,
            "Container-Title": "Springer Book",
            "URL": url,
            "About": about,
            "Table of Contents": toc,
            "Keywords": keywords,
        }

        return result

    def _clean_keywords(self, raw_keywords):
        """Bereinigt die Keywords und extrahiert nur die Begriffe ohne Links."""
        if not raw_keywords:
            return "Nicht verfügbar"

        # Finde alle Texte in eckigen Klammern (Keywords)
        keyword_pattern = r"\[([^\]]+)\]"

        keywords = []
        matches = re.findall(keyword_pattern, raw_keywords)

        for match in matches:
            # Bereinige das Keyword
            clean_keyword = match.strip()
            if clean_keyword and clean_keyword not in keywords:
                keywords.append(clean_keyword)

        # Formatiere die Keywords als kommaseparierte Liste
        if keywords:
            return ", ".join(keywords)
        else:
            return "Nicht verfügbar"

    def _clean_table_of_contents(self, raw_toc):
        """Bereinigt das Inhaltsverzeichnis und extrahiert nur die Kapitel-Titel."""
        if not raw_toc:
            return "Nicht verfügbar"

        # Finde alle Texte in eckigen Klammern, die keine Download-Links sind
        # Muster: [Text] aber nicht [Download chapter PDF]
        chapter_pattern = r"\[([^\]]+)\]"

        chapters = []
        matches = re.findall(chapter_pattern, raw_toc)

        for match in matches:
            # Filtere Download-Links heraus
            if not any(
                keyword in match.lower()
                for keyword in ["download", "pdf", "chapter pdf", "Back to top"]
            ):
                # Bereinige den Titel
                clean_title = match.strip()
                if clean_title and clean_title not in chapters:
                    chapters.append(clean_title)

        # Formatiere das Inhaltsverzeichnis als nummerierte Liste
        if chapters:
            formatted_toc = "\n".join(
                [f"{i+1}. {chapter}" for i, chapter in enumerate(chapters)]
            )
            return formatted_toc
        else:
            return "Nicht verfügbar"

    def _handle_crossref_doi(self):
        """Führt die ursprüngliche CrossRef API-Anfrage aus."""
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
