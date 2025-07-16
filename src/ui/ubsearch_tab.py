import re
import requests
from bs4 import BeautifulSoup
from PyQt6.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QProgressBar,
    QMessageBox,
    QSlider,
    QTabWidget,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import logging
from .abstract_tab import AbstractTab
from ..core.katalog_subject import SubjectExtractor
from ..llm.llm_service import LlmService
from ..core.alima_manager import AlimaManager
from ..core.alima_manager import AlimaManager


class AdditionalTitlesWorker(QThread):
    """Worker für das Laden zusätzlicher Titel"""

    titles_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, classification):
        super().__init__()
        self.classification = classification
        self.logger = logging.getLogger(__name__)

    def setNumResults(self, num_results):
        self.num_results = num_results

    def run(self):
        try:
            # Entferne den Typ (DK/Q) vom Klassifikationscode
            search_term = self.classification.split(" ", 1)[1]
            url = "https://katalog.ub.tu-freiberg.de/Search/Results"
            params = {
                "lookfor": self.classification,
                "type": "udk_raw_de105",
                "limit": self.num_results,
            }
            response = requests.get(url, params=params)
            self.logger.info(f"Generated URL: {response.url}")
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")
            titles = []

            for title_link in soup.find_all("a", class_="title getFull"):
                record_id = title_link.get("id", "").split("|")[-1]
                title_text = title_link.text.strip()
                year_span = title_link.find("span", class_="year")
                year = year_span.text.strip("()") if year_span else ""

                if record_id and title_text:
                    titles.append(
                        {
                            "id": record_id,
                            "title": title_text,
                            "year": year,
                            "url": f"https://katalog.ub.tu-freiberg.de/Record/{record_id}",
                        }
                    )

            self.titles_ready.emit(titles)

        except Exception as e:
            self.logger.error(
                f"Fehler beim Laden zusätzlicher Titel: {str(e)}", exc_info=True
            )
            self.error_occurred.emit(str(e))


class UBSearchWorker(QThread):
    """Worker-Thread für die UB-Suche"""

    progress_updated = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self, keywords):
        super().__init__()
        self.keywords = keywords
        self.base_search_url = "https://katalog.ub.tu-freiberg.de/Search/Results"
        self.base_record_url = "https://katalog.ub.tu-freiberg.de/Record/"
        self.logger = logging.getLogger(__name__)

    def setNumResults(self, num_results):
        self.num_results = num_results

    def run(self):
        """Hauptmethode des Workers"""
        try:
            results = {}
            total = len(self.keywords)

            for i, keyword in enumerate(self.keywords, 1):
                self.status_updated.emit(f"Verarbeite Schlagwort: {keyword}")
                results[keyword] = self.process_keyword(keyword)
                self.progress_updated.emit(i, total)

            self.result_ready.emit(results)

        except Exception as e:
            self.logger.error(f"Fehler im Worker-Thread: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Fehler bei der Suche: {str(e)}")

    def extract_record_ids(self, soup):
        """Extrahiert Record-IDs aus der Suchergebnisseite"""
        record_ids = set()
        save_links = soup.find_all("a", class_="save-record")

        for link in save_links:
            record_id = link.get("data-id")
            if record_id:
                record_ids.add(record_id)

        return list(record_ids)

    def get_classification_numbers(self, record_id):
        """Holt DK- und Q-Nummern sowie den Titel für eine Record-ID"""
        try:
            response = requests.get(f"{self.base_record_url}{record_id}")
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            numbers = []

            # Hole den Titel
            title = "Kein Titel verfügbar"
            title_element = soup.find("h1", attrs={"property": "name"})
            if title_element:
                title = title_element.text.strip()

            # Suche nach DK-Nummern
            dk_links = soup.find_all(
                "a", href=re.compile(r"lookfor=DK.*?&type=udk_raw_de105")
            )
            for dk_link in dk_links:
                dk_match = re.search(r"DK\s+([\d\.:]+)", dk_link.text)
                if dk_match:
                    numbers.append(("DK", dk_match.group(1), title))

            # Suche nach Q-Nummern
            q_links = soup.find_all(
                "a", href=re.compile(r"lookfor=Q[A-Z]?\s*\d+.*?&type=udk_raw_de105")
            )
            for q_link in q_links:
                q_match = re.search(r"Q[A-Z]?\s*[\d\s]+", q_link.text)
                if q_match:
                    numbers.append(("Q", q_match.group().strip(), title))

            return numbers if numbers else None

        except Exception as e:
            self.logger.error(
                f"Fehler beim Abrufen der Nummern für {record_id}: {str(e)}",
                exc_info=True,
            )
            return None

    def process_keyword(self, keyword):
        """Verarbeitet ein einzelnes Schlagwort"""
        try:
            params = {
                "hiddenFilters[]": [
                    'institution:"DE-105"',
                    '-format:"Article"',
                    '-format:"ElectronicArticle"',
                ],
                "join": "AND",
                "bool0[]": "AND",
                "lookfor0[]": keyword,
                "type0[]": "AllFields",
                "filter[]": 'facet_avail:"Local"',
                "limit": self.num_results,
            }

            self.logger.debug(f"Suche nach Schlagwort: {keyword}")
            response = requests.get(self.base_search_url, params=params)
            if response.status_code != 200:
                self.logger.warning(
                    f"HTTP {response.status_code} für Schlagwort {keyword}"
                )
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            record_ids = self.extract_record_ids(soup)

            results = []
            for record_id in record_ids:
                numbers = self.get_classification_numbers(record_id)
                if numbers:
                    for number_type, number, title in numbers:
                        results.append((number_type, number, record_id, title))

            return results

        except Exception as e:
            self.logger.error(
                f"Fehler bei Schlagwort {keyword}: {str(e)}", exc_info=True
            )
            self.error_occurred.emit(
                f"Fehler bei der Verarbeitung von '{keyword}': {str(e)}"
            )
            return []


class UBSearchTab(QWidget):
    """Tab für die UB-Suche"""

    def __init__(self, alima_manager: AlimaManager, llm_service: LlmService = None, main_window: QWidget = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.llm = llm_service
        self.alima_manager = alima_manager # Add this line to initialize alima_manager
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        layout = QVBoxLayout()

        # Input-Bereich
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Schlagworte (kommagetrennt):"))
        self.abstract = ""
        self.keywords_input = QTextEdit()
        self.keywords_input.setPlaceholderText("Geben Sie hier Ihre Schlagworte ein...")
        self.keywords_input.setMaximumHeight(100)
        input_layout.addWidget(self.keywords_input)

        config_layout = QHBoxLayout()
        self.num_results = QSlider(Qt.Orientation.Horizontal)
        self.num_results.setRange(0, 60)
        self.num_results.setValue(20)
        self.num_results.setTickInterval(1)
        self.num_results.valueChanged.connect(self.update_num_results)

        # Temperatur-Label
        self.num_label = QLabel(f"Maximale Treffer: {self.num_results.value()}")
        config_layout.addWidget(self.num_results)
        config_layout.addWidget(self.num_label)
        input_layout.addLayout(config_layout)
        self.search_button = QPushButton("Suche starten")
        self.search_button.clicked.connect(self.start_search)
        input_layout.addWidget(self.search_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        input_layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        input_layout.addWidget(self.status_label)
        layout.addLayout(input_layout)

        self.mainsplitter = QSplitter(Qt.Orientation.Vertical)

        # Detailansicht
        self.results_view = QTextEdit()
        self.results_view.setReadOnly(True)

        # Splitter für geteilte Ansicht
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # TreeView für die Klassifikationen
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Klassifikation", "Anzahl"])
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.tree_widget.setSortingEnabled(True)  # Sortierung aktivieren
        splitter.addWidget(self.tree_widget)

        # Detailansicht
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        splitter.addWidget(self.detail_view)

        # Remove AI tabs and replace with direct calls
        # self.ai_tabs = QTabWidget()
        # self.ai_search = AbstractTab(alima_manager=self.alima_manager, llm_service=self.llm)
        # self.ai_search.template_name = "ub_search"
        # self.ai_search.set_task("dk_list")
        # self.ai_tabs.addTab(self.ai_search, "DK-Zuordnung")
        # self.ai_classification = AbstractTab(alima_manager=self.alima_manager, llm_service=self.llm)
        # self.ai_classification.template_name = "classification"
        # self.ai_classification.set_abstract(self.abstract)
        # self.ai_classification.set_task("dk_class")
        # self.ai_tabs.addTab(self.ai_classification, "DK-Klassifizierung")
        # splitter.addWidget(self.ai_tabs)

        # Setze die Stretchfaktoren für den Splitter
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)

        # Layout zusammenbauen
        self.mainsplitter.addWidget(self.results_view)
        self.mainsplitter.addWidget(splitter)
        layout.addWidget(self.mainsplitter)
        self.setLayout(layout)

    def set_models_and_providers(self, models: dict, providers: list):
        """Sets the available models and providers for the AI tabs."""
        #self.ai_search.set_models_and_providers(models, providers)
        #self.ai_classification.set_models_and_providers(models, providers)

    def set_abstract(self, abstract):
        """Setzt den Abstract für die AI-Verarbeitung"""
        self.abstract = abstract
        self.logger.info(f"Setze Abstract: {abstract}")
        #self.ai_classification.set_abstract(self.abstract)

    def update_num_results(self, value):
        self.num_label.setText(f"Anzahl Treffer: {value}")

    def update_keywords(self, keywords):
        self.logger.info(keywords)
        self.keywords_input.append(keywords)

    def start_search(self):
        """Startet die Suche"""
        keywords_text = self.keywords_input.toPlainText().strip()
        if not keywords_text:
            QMessageBox.warning(
                self, "Eingabefehler", "Bitte geben Sie mindestens ein Schlagwort ein."
            )
            return

        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        unique_keywords = list(dict.fromkeys(keywords))

        self.search_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.detail_view.clear()
        self.status_label.setText("Suche wird gestartet...")

        self.worker = UBSearchWorker(unique_keywords)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.display_results)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.status_updated.connect(self.update_status)
        self.worker.finished.connect(self.search_finished)
        self.worker.setNumResults(self.num_results.value())
        self.worker.start()
        self.ai_search.set_keywords(keywords_text)
        self.ai_classification.set_keywords(keywords_text)

    def update_progress(self, current, total):
        """Aktualisiert die Fortschrittsanzeige"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)

    def update_status(self, status):
        """Aktualisiert den Status"""
        self.status_label.setText(status)

    def display_results(self, results):
        """Zeigt die Ergebnisse gruppiert nach DK/Q-Nummern an"""
        self.detail_view.clear()
        html_results = ["<h3>Suchergebnisse:</h3>"]

        # Sammle Statistiken für Keywords
        keyword_stats = {}
        classification_counts = {}  # Zählt Häufigkeit pro Klassifikation pro Keyword
        for keyword, entries in results.items():
            if not entries:
                continue

            # Sammle Klassifikationen und zähle ihre Häufigkeit
            classifications = {}  # Dict für Klassifikation -> Anzahl
            for entry in entries:
                if entry:
                    number_type, number, _, _ = entry
                    class_key = f"{number_type} {number}"
                    classifications[class_key] = classifications.get(class_key, 0) + 1

            if classifications:
                keyword_stats[keyword] = classifications
                classification_counts[keyword] = len(entries)  # Gesamtzahl der Einträge

        # Zeige Keyword-Statistiken
        if keyword_stats:
            html_results.append("<h4>Gefundene Klassifikationen pro Schlagwort:</h4>")
            # Sortiere Keywords nach Anzahl der gefundenen Klassifikationen (absteigend)
            sorted_keywords = sorted(
                keyword_stats.items(), key=lambda x: len(x[1]), reverse=True
            )

            html_results.append("<ul>")
            for keyword, classifications in sorted_keywords:
                # Sortiere Klassifikationen nach Häufigkeit (absteigend)
                sorted_classifications = sorted(
                    classifications.items(),
                    key=lambda x: (
                        -x[1],  # Primär nach Häufigkeit (absteigend)
                        x[0].split()[0],  # Sekundär nach Typ (DK/Q)
                        [
                            (
                                float(n) if n.replace(".", "").isdigit() else n
                            )  # Tertiär nach Nummer
                            for n in re.split(r"([^0-9]+)", x[0].split()[1])
                        ],
                    ),
                )

                class_strings = [
                    f"{class_key} ({count})"
                    for class_key, count in sorted_classifications
                ]

                html_results.append(
                    f"<li><b>{keyword}</b> ({len(classifications)} Klassifikationen, "
                    f"{classification_counts[keyword]} Treffer): "
                    f"{', '.join(class_strings)}</li>"
                )
            html_results.append("</ul><hr>")

        # Ursprüngliche Ergebnisdarstellung
        number_mapping = {}
        for keyword, entries in results.items():
            if not entries:
                continue

            for entry in entries:
                if not entry:
                    continue

                number_type, number, record_id, title = entry
                key = f"{number_type} {number}"
                if key not in number_mapping:
                    number_mapping[key] = []
                record_url = f"https://katalog.ub.tu-freiberg.de/Record/{record_id}"
                number_mapping[key].append((keyword, record_id, record_url, title))

        sorted_numbers = sorted(
            number_mapping.keys(),
            key=lambda x: (
                x.split()[0],
                [
                    float(n) if n.replace(".", "").isdigit() else n
                    for n in re.split(r"([^0-9]+)", x.split()[1])
                ],
            ),
        )

        if not sorted_numbers:
            html_results.append("<p>Keine Ergebnisse gefunden.</p>")
        else:
            html_results.append("<h4>Detaillierte Ergebnisse:</h4>")
            for number in sorted_numbers:
                entries = number_mapping[number]
                total_titles = len(set(record_id for _, record_id, _, _ in entries))
                html_results.append(f"<p><b>{number}</b> ({total_titles} Titel)</p>")
                html_results.append("<ul>")

                keyword_groups = {}
                for keyword, record_id, record_url, title in entries:
                    if keyword not in keyword_groups:
                        keyword_groups[keyword] = []
                    keyword_groups[keyword].append((record_id, record_url, title))

                for keyword, records in sorted(keyword_groups.items()):
                    html_results.append(f"<li>{keyword}:")
                    links = []
                    for record_id, record_url, title in records:
                        links.append(
                            f'<a href="{record_url}" title="{title}">{record_id}</a> ({title})'
                        )
                    html_results.append(f" {', '.join(links)}</li>")

                html_results.append("</ul>")

        self.results_view.setHtml("".join(html_results))
        self.update_tree_view(results)
        self.ai_classification.set_keywords(self.results_view.toPlainText())
        self.ai_classification.set_abstract(self.abstract)

    def handle_error(self, error_message):
        """Behandelt Fehler"""
        self.logger.error(f"Fehler aufgetreten: {error_message}")
        QMessageBox.critical(self, "Fehler", error_message)
        self.status_label.setText("Fehler bei der Suche")

    def search_finished(self):
        """Wird aufgerufen, wenn die Suche abgeschlossen ist"""
        self.search_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Suche abgeschlossen")

    def update_tree_view(self, results):
        """Aktualisiert die TreeView mit den Suchergebnissen"""
        self.tree_widget.clear()

        # Sammle und gruppiere die Ergebnisse
        number_mapping = {}
        for keyword, entries in results.items():
            if not entries:
                continue

            for entry in entries:
                if not entry:
                    continue

                number_type, number, record_id, title = entry
                key = f"{number_type} {number}"
                if key not in number_mapping:
                    number_mapping[key] = set()
                number_mapping[key].add((record_id, title))

        # Erstelle TreeView-Einträge
        for key in sorted(
            number_mapping.keys(),
            key=lambda x: (
                x.split()[0],
                [
                    float(n) if n.replace(".", "").isdigit() else n
                    for n in re.split(r"([^0-9]+)", x.split()[1])
                ],
            ),
        ):
            count = len(number_mapping[key])
            item = QTreeWidgetItem([key, str(count)])
            item.setData(
                0, Qt.ItemDataRole.UserRole, key
            )  # Speichere den key für später
            self.tree_widget.addTopLevelItem(item)

    def on_tree_item_clicked(self, item, column):
        """Handler für Klicks auf TreeView-Items"""
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            self.fetch_additional_titles(key)

    def fetch_additional_titles(self, classification):
        """Holt weitere Titel für die gewählte Klassifikation"""
        self.detail_view.clear()
        self.detail_view.append(f"Lade weitere Titel für {classification}...")

        # Starte einen neuen Worker für die Suche
        self.additional_worker = AdditionalTitlesWorker(classification)
        self.additional_worker.titles_ready.connect(self.display_additional_titles)
        self.additional_worker.error_occurred.connect(self.handle_error)
        self.additional_worker.setNumResults(self.num_results.value())
        self.additional_worker.start()

    def display_additional_titles(self, titles):
        """Zeigt die zusätzlich geladenen Titel an"""
        self.detail_view.clear()
        abstract = []
        if not titles:
            self.detail_view.append("Keine weiteren Titel gefunden.")
            return

        html = ["<h3>Gefundene Titel:</h3><ul>"]
        for title in titles:
            html.append(
                f'<li><a href="{title["url"]}">{title["title"]}</a> '
                f'({title["year"]})</li>'
            )
            abstract.append(title["title"])
        html.append("</ul>")

        self.detail_view.setHtml("".join(html))
        self.ai_search.set_abstract(" ".join(abstract))
