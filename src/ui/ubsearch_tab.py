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
    QComboBox,
    QCheckBox,
    QLineEdit,
    QGroupBox,
    QFileDialog,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont
import logging
from .abstract_tab import AbstractTab
from ..core.katalog_subject import SubjectExtractor
from ..llm.llm_service import LlmService
from ..core.alima_manager import AlimaManager
from ..utils.clients.biblio_client import BiblioClient  # Claude Generated - SOAP API testing


class AdditionalTitlesWorker(QThread):
    """Worker für das Laden zusätzlicher Titel - Claude Generated: Configurable URLs"""

    titles_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, classification, web_search_url: str = "", web_record_base_url: str = ""):
        super().__init__()
        self.classification = classification
        self.logger = logging.getLogger(__name__)
        # Claude Generated: Configurable URLs, empty defaults mean feature disabled
        self.web_search_url = web_search_url or ""
        self.web_record_base_url = web_record_base_url or ""

    def setNumResults(self, num_results):
        self.num_results = num_results

    def run(self):
        try:
            # Check if URLs are configured
            if not self.web_search_url:
                raise Exception("Katalog-URLs nicht konfiguriert. Bitte in den Einstellungen konfigurieren.")

            # Entferne den Typ (DK/Q) vom Klassifikationscode
            search_term = self.classification.split(" ", 1)[1]
            params = {
                "lookfor": self.classification,
                "type": "udk_raw_de105",
                "limit": self.num_results,
            }
            response = requests.get(self.web_search_url, params=params)
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
                            "url": f"{self.web_record_base_url}{record_id}" if self.web_record_base_url else "",
                        }
                    )

            self.titles_ready.emit(titles)

        except Exception as e:
            self.logger.error(
                f"Fehler beim Laden zusätzlicher Titel: {str(e)}", exc_info=True
            )
            self.error_occurred.emit(str(e))


class UBSearchWorker(QThread):
    """Worker-Thread für die UB-Suche - Claude Generated: Configurable URLs"""

    progress_updated = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self, keywords, web_search_url: str = "", web_record_base_url: str = ""):
        super().__init__()
        self.keywords = keywords
        # Claude Generated: Configurable URLs, empty defaults mean feature disabled
        self.base_search_url = web_search_url or ""
        self.base_record_url = web_record_base_url or ""
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
            # Check if URLs are configured - Claude Generated
            if not self.base_search_url:
                raise Exception("Katalog-URLs nicht konfiguriert. Bitte in den Einstellungen konfigurieren.")

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


class BiblioClientWorker(QThread):
    """Worker-Thread für BiblioClient (SOAP API) Suche - Claude Generated"""

    progress_updated = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    raw_response = pyqtSignal(str)  # Raw SOAP response for debugging

    def __init__(self, keywords: list, token: str = "", debug: bool = False, save_xml_path: str = "", enable_web_fallback: bool = True):
        super().__init__()
        self.keywords = keywords
        self.token = token
        self.debug = debug
        self.save_xml_path = save_xml_path  # Claude Generated - XML debug export
        self.enable_web_fallback = enable_web_fallback  # Claude Generated - Web fallback
        self.logger = logging.getLogger(__name__)
        self.num_results = 20

    def setNumResults(self, num_results):
        self.num_results = num_results

    def run(self):
        try:
            client = BiblioClient(
                token=self.token,
                debug=self.debug,
                save_xml_path=self.save_xml_path,
                enable_web_fallback=self.enable_web_fallback,
            )  # Claude Generated - Pass all parameters
            classification_info = {}

            for keyword_idx, keyword in enumerate(self.keywords):
                self.status_updated.emit(f"Searching for: {keyword}")
                self.progress_updated.emit(keyword_idx, len(self.keywords))

                # Search catalog
                search_results = client.search(keyword, search_type="ku")

                if not search_results:
                    self.logger.info(f"No results for keyword: {keyword}")
                    continue

                # Process each result to extract details and classifications
                for item in search_results[: self.num_results]:
                    rsn = item.get("rsn")
                    if not rsn:
                        continue

                    # Get title details (includes classifications extraction)
                    details = client.get_title_details(rsn)

                    if details and details.get("classifications"):
                        classifications = details.get("classifications", [])

                        # Extract decimal classifications
                        decimal_classes = client.extract_decimal_classifications(
                            classifications
                        )

                        for dk in decimal_classes:
                            key = f"DK {dk}"
                            if key not in classification_info:
                                classification_info[key] = {
                                    "count": 0,
                                    "titles": [],
                                    "details": [],
                                }
                            classification_info[key]["count"] += 1
                            title = details.get("title", "")
                            if title and title not in classification_info[key]["titles"]:
                                classification_info[key]["titles"].append(title)
                            classification_info[key]["details"].append(details)

                    # Emit raw response for debugging
                    if self.debug:
                        self.raw_response.emit(
                            f"RSN {rsn}: title={details.get('title')}, "
                            f"classifications={details.get('classifications')}"
                        )

            # Prepare final result
            result_data = {
                "method": "BiblioClient SOAP API",
                "keywords": self.keywords,
                "classifications": classification_info,
                "total_unique_classifications": len(classification_info),
            }

            # Log XML export status - Claude Generated
            if self.save_xml_path:
                self.raw_response.emit(
                    f"✅ XML responses saved to: {self.save_xml_path}"
                )

            self.result_ready.emit(result_data)

        except Exception as e:
            self.logger.error(f"BiblioClient search failed: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"BiblioClient error: {str(e)}")


class UBSearchTab(QWidget):
    """Tab für die UB-Suche"""

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService = None,
        main_window: QWidget = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.llm = llm_service
        self.alima_manager = alima_manager  # Add this line to initialize alima_manager
        self.main_window = main_window
        self.current_worker = None  # Claude Generated - Track current worker
        # Claude Generated: Store catalog URLs for use in display_results
        self.web_record_base_url = ""
        self.init_ui()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        layout = QVBoxLayout()

        # Mode Selection - Claude Generated SOAP API Testing
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Suchmethod:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Web-Katalog (HTML)", "SOAP API (BiblioClient)"])
        self.mode_selector.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_selector)
        layout.addLayout(mode_layout)

        # SOAP API Configuration Panel - Claude Generated
        self.soap_config_group = QGroupBox("SOAP API Einstellungen")
        soap_config_layout = QVBoxLayout()

        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("API Token:"))
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Katalog-API Token (optional)")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        token_layout.addWidget(self.token_input)
        soap_config_layout.addLayout(token_layout)

        debug_layout = QHBoxLayout()
        self.debug_checkbox = QCheckBox("Debug-Modus (zeige XML-Response)")
        self.debug_checkbox.setChecked(False)
        debug_layout.addWidget(self.debug_checkbox)
        self.show_raw_response_checkbox = QCheckBox("Zeige Rohdaten")
        self.show_raw_response_checkbox.setChecked(True)
        debug_layout.addWidget(self.show_raw_response_checkbox)
        debug_layout.addStretch()
        soap_config_layout.addLayout(debug_layout)

        # Fallback Option - Claude Generated
        fallback_layout = QHBoxLayout()
        self.enable_web_fallback_checkbox = QCheckBox("Web-Scraping Fallback (bei SOAP-Fehler)")
        self.enable_web_fallback_checkbox.setChecked(True)
        fallback_layout.addWidget(self.enable_web_fallback_checkbox)
        fallback_layout.addStretch()
        soap_config_layout.addLayout(fallback_layout)

        # XML Export Options - Claude Generated
        xml_layout = QHBoxLayout()
        self.save_xml_checkbox = QCheckBox("XML-Responses speichern (Debug)")
        self.save_xml_checkbox.setChecked(False)
        self.save_xml_checkbox.toggled.connect(self.on_save_xml_toggled)
        xml_layout.addWidget(self.save_xml_checkbox)

        self.xml_path_display = QLineEdit()
        self.xml_path_display.setReadOnly(True)
        self.xml_path_display.setPlaceholderText("Pfad für XML-Dateien (optional)")
        xml_layout.addWidget(self.xml_path_display)

        self.xml_browse_button = QPushButton("📁 Durchsuchen...")
        self.xml_browse_button.clicked.connect(self.select_xml_save_path)
        self.xml_browse_button.setEnabled(False)
        xml_layout.addWidget(self.xml_browse_button)

        soap_config_layout.addLayout(xml_layout)

        self.soap_config_group.setLayout(soap_config_layout)
        self.soap_config_group.setVisible(False)  # Hidden by default
        layout.addWidget(self.soap_config_group)

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

        # Raw Response Debug Viewer - Claude Generated
        self.debug_tabs = QTabWidget()
        self.raw_response_view = QTextEdit()
        self.raw_response_view.setReadOnly(True)
        self.raw_response_view.setMaximumHeight(150)
        font = QFont("Courier")
        font.setPointSize(9)
        self.raw_response_view.setFont(font)
        self.debug_tabs.addTab(self.raw_response_view, "🔍 Raw Responses (Debug)")
        self.debug_tabs.setVisible(False)  # Hidden by default
        layout.addWidget(self.debug_tabs)

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
        # self.ai_search.set_models_and_providers(models, providers)
        # self.ai_classification.set_models_and_providers(models, providers)

    def set_abstract(self, abstract):
        """Setzt den Abstract für die AI-Verarbeitung"""
        self.abstract = abstract
        self.logger.info(f"Setze Abstract: {abstract}")
        # self.ai_classification.set_abstract(self.abstract)

    def update_num_results(self, value):
        self.num_label.setText(f"Anzahl Treffer: {value}")

    def update_keywords(self, keywords):
        self.logger.info(keywords)
        self.keywords_input.append(keywords)

    def on_mode_changed(self, index):
        """Handle search method change - Claude Generated"""
        is_soap_mode = index == 1  # SOAP API (BiblioClient) is second option
        self.soap_config_group.setVisible(is_soap_mode)

    def on_save_xml_toggled(self, checked):
        """Enable/disable XML path selector - Claude Generated"""
        self.xml_browse_button.setEnabled(checked)

    def select_xml_save_path(self):
        """Select directory for XML file export - Claude Generated"""
        directory = QFileDialog.getExistingDirectory(
            self, "Verzeichnis für XML-Debug-Dateien wählen", ""
        )
        if directory:
            self.xml_path_display.setText(directory)
            self.logger.info(f"XML save path set to: {directory}")

    def start_search(self):
        """Startet die Suche mit aktuell ausgewählter Methode - Claude Generated"""
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

        # Choose worker based on selected mode - Claude Generated
        if self.mode_selector.currentIndex() == 1:  # SOAP API mode
            token = self.token_input.text().strip() or ""
            debug = self.debug_checkbox.isChecked()
            save_xml = self.xml_path_display.text() if self.save_xml_checkbox.isChecked() else ""
            enable_web_fallback = self.enable_web_fallback_checkbox.isChecked()
            self.current_worker = BiblioClientWorker(
                unique_keywords,
                token=token,
                debug=debug,
                save_xml_path=save_xml,
                enable_web_fallback=enable_web_fallback,
            )  # Claude Generated - Pass all parameters
            self.current_worker.raw_response.connect(self.append_raw_response)
            self.logger.info(
                f"Starting SOAP API search with debug={debug}, save_xml={bool(save_xml)}, web_fallback={enable_web_fallback}"
            )
        else:  # Web Catalog mode (default)
            # Claude Generated: Get catalog URLs from config
            web_search_url = ""
            web_record_base_url = ""
            if self.alima_manager and hasattr(self.alima_manager, 'config'):
                catalog_config = getattr(self.alima_manager.config, 'catalog_config', None)
                if catalog_config:
                    web_search_url = getattr(catalog_config, 'catalog_search_url', '')
                    web_record_base_url = getattr(catalog_config, 'catalog_details_url', '')

            # Claude Generated: Store URL for use in display_results
            self.web_record_base_url = web_record_base_url

            self.current_worker = UBSearchWorker(unique_keywords, web_search_url=web_search_url, web_record_base_url=web_record_base_url)
            self.logger.info("Starting Web Catalog search")

        self.current_worker.progress_updated.connect(self.update_progress)
        self.current_worker.result_ready.connect(self.display_results)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.status_updated.connect(self.update_status)
        self.current_worker.finished.connect(self.search_finished)
        self.current_worker.setNumResults(self.num_results.value())
        self.current_worker.start()

    def update_progress(self, current, total):
        """Aktualisiert die Fortschrittsanzeige"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)

    def update_status(self, status):
        """Aktualisiert den Status"""
        self.status_label.setText(status)

    def display_results(self, results):
        """Zeigt die Ergebnisse gruppiert nach DK/Q-Nummern an - Claude Generated (supports both Web and SOAP)"""
        self.detail_view.clear()
        html_results = ["<h3>Suchergebnisse:</h3>"]

        # Detect result type and handle accordingly - Claude Generated
        if isinstance(results, dict) and "method" in results:
            # BiblioClient SOAP API results
            self._display_biblio_results(results, html_results)
        else:
            # Web Catalog results (original format)
            self._display_web_results(results, html_results)

        self.results_view.setHtml("".join(html_results))

        # Update tree view based on result type - Claude Generated
        if isinstance(results, dict) and "method" in results:
            # BiblioClient SOAP API results
            self.update_tree_view_biblio(results)
        else:
            # Web Catalog results
            self.update_tree_view(results)

        if hasattr(self, "ai_classification"):
            self.ai_classification.set_keywords(self.results_view.toPlainText())
            self.ai_classification.set_abstract(self.abstract)

    def _display_biblio_results(self, results: dict, html_results: list):
        """Display BiblioClient SOAP API results - Claude Generated"""
        html_results.append(f"<p><b>Method:</b> {results.get('method', 'Unknown')}</p>")

        classifications = results.get("classifications", {})
        keywords_list = results.get("keywords", [])

        if not classifications:
            html_results.append("<p>Keine Klassifikationen gefunden.</p>")
            return

        html_results.append(f"<p><b>Schlagworte:</b> {', '.join(keywords_list)}</p>")
        html_results.append(
            f"<p><b>Gefundene Klassifikationen:</b> {len(classifications)}</p>"
        )
        html_results.append("<h4>Klassifikationen:</h4>")

        for class_key in sorted(classifications.keys()):
            class_data = classifications[class_key]
            count = class_data.get("count", 0)
            titles = class_data.get("titles", [])
            html_results.append(
                f"<p><b>{class_key}</b> ({count} Titel): {', '.join(titles[:3])}"
            )
            if len(titles) > 3:
                html_results.append(f" ... und {len(titles) - 3} weitere")
            html_results.append("</p>")

        self.append_raw_response(
            f"✅ Displayed BiblioClient results: {len(classifications)} classifications"
        )

    def _display_web_results(self, results: dict, html_results: list):
        """Display Web Catalog results (original format) - Claude Generated"""
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
                # Claude Generated: Use configurable catalog URL or fallback
                record_url = f"{self.web_record_base_url}{record_id}" if self.web_record_base_url else ""
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

        # Update AI classification if available - Claude Generated
        if hasattr(self, "ai_classification"):
            self.ai_classification.set_keywords(self.results_view.toPlainText())
            self.ai_classification.set_abstract(self.abstract)

    def append_raw_response(self, response_text):
        """Append raw response to debug viewer - Claude Generated"""
        if not self.show_raw_response_checkbox.isChecked():
            return

        # Show debug panel if hidden
        self.debug_tabs.setVisible(True)

        # Append with timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.raw_response_view.append(f"[{timestamp}] {response_text}")

        # Auto-scroll to bottom
        self.raw_response_view.verticalScrollBar().setValue(
            self.raw_response_view.verticalScrollBar().maximum()
        )

    def handle_error(self, error_message):
        """Behandelt Fehler"""
        self.logger.error(f"Fehler aufgetreten: {error_message}")
        self.append_raw_response(f"❌ ERROR: {error_message}")
        QMessageBox.critical(self, "Fehler", error_message)
        self.status_label.setText("Fehler bei der Suche")

    def search_finished(self):
        """Wird aufgerufen, wenn die Suche abgeschlossen ist"""
        self.search_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Show XML export location if enabled - Claude Generated
        if (
            self.mode_selector.currentIndex() == 1
            and self.save_xml_checkbox.isChecked()
            and self.xml_path_display.text()
        ):
            status = f"Suche abgeschlossen - XML-Dateien: {self.xml_path_display.text()}"
        else:
            status = "Suche abgeschlossen"

        self.status_label.setText(status)

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

    def update_tree_view_biblio(self, results: dict):
        """Aktualisiert die TreeView mit BiblioClient SOAP API Ergebnissen - Claude Generated"""
        self.tree_widget.clear()
        self.biblio_results = results  # Store for later access - Claude Generated

        classifications = results.get("classifications", {})
        self.logger.info(f"TreeView: Updating with {len(classifications)} BiblioClient classifications")

        for class_key in sorted(classifications.keys()):
            class_data = classifications[class_key]
            count = class_data.get("count", 0)
            titles = class_data.get("titles", [])

            item = QTreeWidgetItem([class_key, str(count)])
            # Store classification key AND type for later
            item.setData(0, Qt.ItemDataRole.UserRole, class_key)
            # Store method type to distinguish from web results
            item.setData(0, Qt.ItemDataRole.UserRole + 1, "biblio")

            self.tree_widget.addTopLevelItem(item)

    def on_tree_item_clicked(self, item, column):
        """Handler für Klicks auf TreeView-Items - Claude Generated (supports both Web and BiblioClient)"""
        key = item.data(0, Qt.ItemDataRole.UserRole)
        method_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        if key:
            if method_type == "biblio":
                # BiblioClient results
                self.display_biblio_titles(key)
            else:
                # Web Catalog results
                self.fetch_additional_titles(key)

    def display_biblio_titles(self, classification_key: str):
        """Display titles for selected BiblioClient classification - Claude Generated"""
        self.detail_view.clear()

        if not hasattr(self, "biblio_results"):
            self.detail_view.append("Keine BiblioClient-Ergebnisse verfügbar")
            return

        classifications = self.biblio_results.get("classifications", {})
        class_data = classifications.get(classification_key)

        if not class_data:
            self.detail_view.append(f"Keine Daten für {classification_key} gefunden")
            return

        titles = class_data.get("titles", [])
        details = class_data.get("details", [])

        html = [f"<h3>Titel für {classification_key}:</h3>"]
        html.append(f"<p><b>Anzahl:</b> {class_data.get('count', 0)}</p>")

        if titles:
            html.append("<h4>Titelliste:</h4><ul>")
            for i, title in enumerate(titles):
                # Show title, optionally with details
                detail_info = ""
                if i < len(details):
                    detail = details[i]
                    author_list = detail.get("author", [])
                    isbn = detail.get("isbn", "")
                    author_str = ", ".join(author_list[:2]) if author_list else ""
                    if author_str or isbn:
                        detail_info = f" <small>({author_str}{' ISBN: ' + isbn if isbn else ''})</small>"

                html.append(f"<li>{title}{detail_info}</li>")
            html.append("</ul>")
        else:
            html.append("<p><i>Keine Titel verfügbar</i></p>")

        self.detail_view.setHtml("".join(html))

    def fetch_additional_titles(self, classification):
        """Holt weitere Titel für die gewählte Klassifikation"""
        self.detail_view.clear()
        self.detail_view.append(f"Lade weitere Titel für {classification}...")

        # Claude Generated: Get catalog URLs from config
        web_search_url = ""
        web_record_base_url = ""
        if self.alima_manager and hasattr(self.alima_manager, 'config'):
            catalog_config = getattr(self.alima_manager.config, 'catalog_config', None)
            if catalog_config:
                web_search_url = getattr(catalog_config, 'catalog_search_url', '')
                web_record_base_url = getattr(catalog_config, 'catalog_details_url', '')

        # Starte einen neuen Worker für die Suche
        self.additional_worker = AdditionalTitlesWorker(classification, web_search_url=web_search_url, web_record_base_url=web_record_base_url)
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

        # Update AI search if available - Claude Generated
        if hasattr(self, "ai_search"):
            self.ai_search.set_abstract(" ".join(abstract))
