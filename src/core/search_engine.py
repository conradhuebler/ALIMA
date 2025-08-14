from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from collections import Counter
from typing import Dict, Set, List, Tuple, Optional
from datetime import datetime
import json
import logging


class SearchEngine(QObject):
    term_search_completed = pyqtSignal(str, dict)
    term_search_error = pyqtSignal(str, str)
    search_finished = pyqtSignal(dict)

    def __init__(self, cache_manager: 'UnifiedKnowledgeManager'):
        """
        Initialisiert die Suchmaschine.

        Args:
            cache_manager: Instanz des UnifiedKnowledgeManager für Caching-Funktionalität
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

    def search(self, terms: List[str], threshold: float = 1.0):
        """
        Führt die Suche für mehrere Begriffe durch.

        Args:
            terms: Liste der Suchbegriffe
            threshold: Schwellenwert für die Relevanz in Prozent
        """
        self.logger.info(f"SearchEngine starting search for terms: {terms}")
        self.term_results = {}
        self.total_counter = Counter()
        self.pending_requests = len(terms)
        self.threshold = threshold

        if not terms:
            self.logger.info("No terms provided, emitting empty results")
            self.search_finished.emit(self.process_results(self.threshold))
            return

        self.logger.info(f"Starting search for {len(terms)} terms")
        for term in terms:
            self.search_term(term)

    def search_term(self, term: str):
        """
        Führt die Suche für einen einzelnen Begriff durch.

        Args:
            term: Suchbegriff
        """
        # Prüfe Cache
        cached_results = self.cache.get_cached_results(term)
        if cached_results:
            # Store cached results in the engine for processing
            self.term_results[term] = cached_results
            self.total_counter.update(cached_results["counter"])

            self.term_search_completed.emit(term, cached_results)

            # Decrease pending requests and check if all are complete
            self.pending_requests -= 1
            self.logger.info(
                f"Cache hit for {term}, pending requests: {self.pending_requests}"
            )
            if self.pending_requests == 0:
                self.logger.info("All requests complete, emitting search_finished")
                self.search_finished.emit(self.process_results(self.threshold))
            return

        self.logger.info(f"Suche nach Begriff: {term}")
        if len(term) < 3:
            self.term_search_error.emit(term, "Suchbegriff zu kurz")
            # Decrease pending requests and check if all are complete
            self.pending_requests -= 1
            self.logger.info(
                f"Error for {term}, pending requests: {self.pending_requests}"
            )
            if self.pending_requests == 0:
                self.logger.info("All requests complete, emitting search_finished")
                self.search_finished.emit(self.process_results(self.threshold))
            return

        # Erstelle URL mit Parametern
        url = QUrl(self.base_url)
        url.setQuery(f"q={term}&format=jsonl")

        request = QNetworkRequest(url)
        reply = self.network_manager.get(request)
        reply.finished.connect(lambda: self._handle_network_reply(reply, term))

    def _handle_network_reply(self, reply: QNetworkReply, term: str):
        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = reply.readAll().data().decode("utf-8")
            subject_headings = []
            total_items = 0

            for line in data.splitlines():
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    headings = self.extract_subject_headings(item)
                    subject_headings.extend(headings)
                    total_items += 1
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON Decode Error for term {term}: {e}")
                    continue

            results = {
                "headings": set(subject_headings),
                "counter": Counter(subject_headings),
                "total": total_items,
                "timestamp": datetime.now().isoformat(),
            }
            # Store results in the engine for processing
            self.term_results[term] = results
            self.total_counter.update(results["counter"])

            self.cache.cache_results(term, results)
            self.term_search_completed.emit(term, results)
        else:
            self.logger.error(f"Network error for term {term}: {reply.errorString()}")
            self.term_search_error.emit(term, reply.errorString())

        # Decrease pending requests and check if all are complete
        self.pending_requests -= 1
        self.logger.info(
            f"Network request completed for {term}, pending requests: {self.pending_requests}"
        )
        if self.pending_requests == 0:
            self.logger.info("All requests complete, emitting search_finished")
            self.search_finished.emit(self.process_results(self.threshold))

        reply.deleteLater()

    def process_results(self, threshold: float) -> Dict:
        """
        Verarbeitet die Suchergebnisse und wendet den Threshold an.

        Args:
            threshold: Schwellenwert für die Relevanz in Prozent

        Returns:
            Aufbereitete Suchergebnisse
        """
        if not self.total_counter:
            return {"exact_matches": [], "frequent_matches": []}

        most_common_count = self.total_counter.most_common(1)[0][1]
        threshold_count = max(3, int(most_common_count * (threshold / 100.0)))

        exact_matches = []
        frequent_matches = []
        search_terms = [term.strip().lower() for term in self.term_results.keys()]

        for item, count in self.total_counter.most_common():
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
        return {"term_results": self.term_results, "total_counter": self.total_counter}
