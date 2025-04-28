## obsolete

from PyQt6.QtCore import QObject, QUrl, QEventLoop
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from collections import Counter
from typing import Dict, Set, List, Tuple, Optional
from datetime import datetime
import json
import logging


class SearchEngine(QObject):
    def __init__(self, cache_manager):
        """
        Initialisiert die Suchmaschine.

        Args:
            cache_manager: Instance des CacheManagers für Caching-Funktionalität
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.cache = cache_manager
        self.base_url = "https://lobid.org/resources/search"
        self.network_manager = QNetworkAccessManager()
        self.current_results = {"term_results": {}, "total_counter": Counter()}

    def extract_subject_headings(self, item: Dict) -> List[Tuple[str, str]]:
        """
        Extrahiert GND-Schlagworte aus einem Lobid-Eintrag.

        Args:
            item: Dictionary mit Lobid-Daten

        Returns:
            Liste von Tupeln (Label, GND-ID)
        """
        subject_headings = []

        if "subject" in item:
            for subject in item["subject"]:
                if isinstance(subject, dict):
                    # Fall 1: ComplexSubject mit componentList
                    if "componentList" in subject:
                        for component in subject["componentList"]:
                            if (
                                isinstance(component, dict)
                                and component.get("type") == ["SubjectHeading"]
                                and component.get("id", "").startswith(
                                    "https://d-nb.info/gnd/"
                                )
                            ):
                                label = component.get("label", "")
                                gnd_id = component.get("id", "")
                                if label:
                                    subject_headings.append((label, gnd_id))

                    # Fall 2: Direktes SubjectHeading
                    elif subject.get("type") == ["SubjectHeading"] and subject.get(
                        "id", ""
                    ).startswith("https://d-nb.info/gnd/"):
                        label = subject.get("label", "")
                        gnd_id = subject.get("id", "")
                        if label:
                            subject_headings.append((label, gnd_id))

        return subject_headings

    async def search(self, terms: List[str], threshold: float = 1.0) -> Dict:
        """
        Führt die Suche für mehrere Begriffe durch.

        Args:
            terms: Liste der Suchbegriffe
            threshold: Schwellenwert für die Relevanz in Prozent

        Returns:
            Dictionary mit Suchergebnissen
        """
        term_results = {}
        total_counter = Counter()

        for term in terms:
            results = await self.search_term(term)
            if results:
                term_results[term] = results
                total_counter.update(results["counter"])

        # Speichere aktuelle Ergebnisse
        self.current_results = {
            "term_results": term_results,
            "total_counter": total_counter,
        }

        return self.process_results(threshold)

    async def search_term(self, term: str) -> Optional[Dict]:
        """
        Führt die Suche für einen einzelnen Begriff durch.

        Args:
            term: Suchbegriff

        Returns:
            Dictionary mit Ergebnissen oder None bei Fehler
        """
        # Prüfe Cache
        cached_results = self.cache.get_cached_results(term)
        if cached_results:
            return cached_results

        self.logger.info(f"Suche nach Begriff: {term}")
        if len(term) < 3:
            return None

        # Erstelle URL mit Parametern
        url = QUrl(self.base_url)
        url.setQuery(f"q={term}&format=jsonl")

        request = QNetworkRequest(url)

        try:
            reply = await self._make_network_request(request)

            if not reply:
                return None

            data = reply.readAll().data().decode("utf-8")
            subject_headings = []
            total_items = 0

            # Verarbeite JSONL-Antwort
            for line in data.splitlines():
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    headings = self.extract_subject_headings(item)
                    subject_headings.extend(headings)
                    total_items += 1
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON Decode Error: {e}")
                    continue

            # Erstelle Ergebnisse
            results = {
                "headings": set(subject_headings),
                "counter": Counter(subject_headings),
                "total": total_items,
                "timestamp": datetime.now().isoformat(),
            }

            # Cache die Ergebnisse
            self.cache.cache_results(term, results)

            return results

        except Exception as e:
            self.logger.error(f"Fehler bei der Netzwerkanfrage: {e}")
            return None

    async def _make_network_request(
        self, request: QNetworkRequest
    ) -> Optional[QNetworkReply]:
        """
        Führt eine Netzwerkanfrage aus und wartet auf die Antwort.

        Args:
            request: Die QNetworkRequest

        Returns:
            QNetworkReply oder None bei Fehler
        """

        loop = QEventLoop()
        reply = self.network_manager.get(request)

        # Verbinde Signale
        reply.finished.connect(loop.quit)
        reply.errorOccurred.connect(lambda: self._handle_network_error(reply, loop))

        # Warte auf Antwort
        loop.exec()

        if reply.error() == QNetworkReply.NetworkError.NoError:
            return reply
        return None

    def _handle_network_error(self, reply: QNetworkReply, loop: QEventLoop):
        """
        Behandelt Netzwerkfehler.

        Args:
            reply: Die QNetworkReply
            loop: Der QEventLoop
        """
        self.logger.error(f"Netzwerkfehler: {reply.errorString()}")
        loop.quit()

    def process_results(self, current_results, threshold: float) -> Dict:
        """
        Verarbeitet die Suchergebnisse und wendet den Threshold an.

        Args:
            threshold: Schwellenwert für die Relevanz in Prozent

        Returns:
            Aufbereitete Suchergebnisse
        """
        self.current_results = current_results
        if not self.current_results["total_counter"]:
            return {"exact_matches": [], "frequent_matches": []}

        most_common_count = self.current_results["total_counter"].most_common(1)[0][1]
        threshold_count = max(3, int(most_common_count * (threshold / 100.0)))

        exact_matches = []
        frequent_matches = []
        search_terms = [
            term.strip().lower() for term in self.current_results["term_results"].keys()
        ]

        for item, count in self.current_results["total_counter"].most_common():
            if isinstance(item, tuple) and len(item) >= 2:
                label = item[0].strip("'").lower()

                # Prüfe auf exakte Übereinstimmungen
                if any(term.lower() == label for term in search_terms):
                    exact_matches.append((item, count))
                    continue

                # Wenn es kein exakter Match ist, prüfe die Häufigkeit
                if count >= threshold_count:
                    frequent_matches.append((item, count))

        return {
            "exact_matches": exact_matches,
            "frequent_matches": frequent_matches,
            "threshold_count": threshold_count,
        }

    def get_current_results(self) -> Dict:
        """
        Gibt die aktuellen Suchergebnisse zurück.

        Returns:
            Dictionary mit aktuellen Ergebnissen
        """
        return self.current_results
