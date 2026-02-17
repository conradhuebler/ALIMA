"""
DkAnalysisUnifiedTab - Unified DK Analysis Tab for ALIMA
Combines DK-Zuordnung (LLM classification), DK-Statistik (results display), and UB-Suche (catalog search).
Claude Generated
"""

import re
import requests
import logging
from datetime import datetime
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLabel, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QSplitter, QTabWidget,
    QPushButton, QProgressBar, QMessageBox, QSlider,
    QComboBox, QCheckBox, QLineEdit, QGridLayout,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor

from .abstract_tab import AbstractTab
from .styles import (
    get_main_stylesheet,
    get_button_styles,
    get_status_label_styles,
    get_confidence_style,
    LAYOUT,
    COLORS,
)
from ..core.alima_manager import AlimaManager
from ..llm.llm_service import LlmService
from ..core.pipeline_manager import PipelineManager
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..utils.pipeline_utils import PipelineResultFormatter


# ============================================================================
# Worker Classes (from ubsearch_tab.py)
# ============================================================================

class AdditionalTitlesWorker(QThread):
    """Worker for loading additional titles"""

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
                f"Error loading additional titles: {str(e)}", exc_info=True
            )
            self.error_occurred.emit(str(e))


class UBSearchWorker(QThread):
    """Worker-Thread for UB Web Catalog search"""

    progress_updated = pyqtSignal(int, int)
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
        try:
            results = {}
            total = len(self.keywords)

            for i, keyword in enumerate(self.keywords, 1):
                self.status_updated.emit(f"Processing keyword: {keyword}")
                results[keyword] = self.process_keyword(keyword)
                self.progress_updated.emit(i, total)

            self.result_ready.emit(results)

        except Exception as e:
            self.logger.error(f"Worker thread error: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Search error: {str(e)}")

    def extract_record_ids(self, soup):
        record_ids = set()
        save_links = soup.find_all("a", class_="save-record")

        for link in save_links:
            record_id = link.get("data-id")
            if record_id:
                record_ids.add(record_id)

        return list(record_ids)

    def get_classification_numbers(self, record_id):
        try:
            response = requests.get(f"{self.base_record_url}{record_id}")
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            numbers = []

            title = "No title available"
            title_element = soup.find("h1", attrs={"property": "name"})
            if title_element:
                title = title_element.text.strip()

            dk_links = soup.find_all(
                "a", href=re.compile(r"lookfor=DK.*?&type=udk_raw_de105")
            )
            for dk_link in dk_links:
                dk_match = re.search(r"DK\s+([\d\.:]+)", dk_link.text)
                if dk_match:
                    numbers.append(("DK", dk_match.group(1), title))

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
                f"Error fetching classification numbers for {record_id}: {str(e)}",
                exc_info=True,
            )
            return None

    def process_keyword(self, keyword):
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

            self.logger.debug(f"Searching for keyword: {keyword}")
            response = requests.get(self.base_search_url, params=params)
            if response.status_code != 200:
                self.logger.warning(
                    f"HTTP {response.status_code} for keyword {keyword}"
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
                f"Error processing keyword {keyword}: {str(e)}", exc_info=True
            )
            self.error_occurred.emit(
                f"Error processing '{keyword}': {str(e)}"
            )
            return []


class BiblioClientWorker(QThread):
    """Worker-Thread for BiblioClient (SOAP API) search"""

    progress_updated = pyqtSignal(int, int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    raw_response = pyqtSignal(str)

    def __init__(self, keywords: list, token: str = "", debug: bool = False,
                 save_xml_path: str = "", enable_web_fallback: bool = True):
        super().__init__()
        self.keywords = keywords
        self.token = token
        self.debug = debug
        self.save_xml_path = save_xml_path
        self.enable_web_fallback = enable_web_fallback
        self.logger = logging.getLogger(__name__)
        self.num_results = 20

    def setNumResults(self, num_results):
        self.num_results = num_results

    def run(self):
        try:
            from ..utils.biblio_client import BiblioClient
            client = BiblioClient(
                token=self.token,
                debug=self.debug,
                save_xml_path=self.save_xml_path,
                enable_web_fallback=self.enable_web_fallback,
            )
            classification_info = {}

            for keyword_idx, keyword in enumerate(self.keywords):
                self.status_updated.emit(f"Searching for: {keyword}")
                self.progress_updated.emit(keyword_idx, len(self.keywords))

                search_results = client.search(keyword, search_type="ku")

                if not search_results:
                    self.logger.info(f"No results for keyword: {keyword}")
                    continue

                for item in search_results[: self.num_results]:
                    rsn = item.get("rsn")
                    if not rsn:
                        continue

                    details = client.get_title_details(rsn)

                    if details and details.get("classifications"):
                        classifications = details.get("classifications", [])
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

                    if self.debug:
                        self.raw_response.emit(
                            f"RSN {rsn}: title={details.get('title')}, "
                            f"classifications={details.get('classifications')}"
                        )

            result_data = {
                "method": "BiblioClient SOAP API",
                "keywords": self.keywords,
                "classifications": classification_info,
                "total_unique_classifications": len(classification_info),
            }

            if self.save_xml_path:
                self.raw_response.emit(
                    f"XML responses saved to: {self.save_xml_path}"
                )

            self.result_ready.emit(result_data)

        except Exception as e:
            self.logger.error(f"BiblioClient search failed: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"BiblioClient error: {str(e)}")


# ============================================================================
# UB Search Panel Widget
# ============================================================================

class UBSearchPanel(QWidget):
    """UB Katalog Search Panel - integrates Web and SOAP API search"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.current_worker = None
        self.biblio_results = None
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(get_main_stylesheet())
        btn_styles = get_button_styles()

        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["spacing"])
        layout.setContentsMargins(0, 0, 0, 0)

        # Mode Selection
        mode_group = QGroupBox("Search Settings")
        mode_layout = QGridLayout(mode_group)
        mode_layout.setSpacing(LAYOUT["inner_spacing"])

        mode_layout.addWidget(QLabel("Search Method:"), 0, 0)
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Web Catalog (HTML)", "SOAP API (BiblioClient)"])
        self.mode_selector.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_selector, 0, 1)

        mode_layout.addWidget(QLabel("Max Results:"), 1, 0)
        self.num_results = QSlider(Qt.Orientation.Horizontal)
        self.num_results.setRange(1, 60)
        self.num_results.setValue(20)
        self.num_results.setTickInterval(5)
        self.num_results.valueChanged.connect(self.update_num_results)
        mode_layout.addWidget(self.num_results, 1, 1)

        self.num_label = QLabel(f"Results: {self.num_results.value()}")
        self.num_label.setMinimumWidth(100)
        mode_layout.addWidget(self.num_label, 1, 2)

        layout.addWidget(mode_group)

        # SOAP API Configuration Panel
        self.soap_config_group = QGroupBox("SOAP API Settings")
        soap_grid = QGridLayout(self.soap_config_group)
        soap_grid.setSpacing(LAYOUT["inner_spacing"])

        soap_grid.addWidget(QLabel("API Token:"), 0, 0)
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Catalog API Token (optional)")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        soap_grid.addWidget(self.token_input, 0, 1, 1, 4)

        self.debug_checkbox = QCheckBox("Debug (XML)")
        soap_grid.addWidget(self.debug_checkbox, 1, 0)
        self.show_raw_response_checkbox = QCheckBox("Raw Data")
        self.show_raw_response_checkbox.setChecked(True)
        soap_grid.addWidget(self.show_raw_response_checkbox, 1, 1)
        self.enable_web_fallback_checkbox = QCheckBox("Web Fallback")
        self.enable_web_fallback_checkbox.setChecked(True)
        soap_grid.addWidget(self.enable_web_fallback_checkbox, 1, 2)
        self.save_xml_checkbox = QCheckBox("Save XML:")
        self.save_xml_checkbox.setChecked(False)
        self.save_xml_checkbox.toggled.connect(self.on_save_xml_toggled)
        soap_grid.addWidget(self.save_xml_checkbox, 1, 3)
        self.xml_path_display = QLineEdit()
        self.xml_path_display.setReadOnly(True)
        self.xml_path_display.setPlaceholderText("Path...")
        soap_grid.addWidget(self.xml_path_display, 1, 4)
        self.xml_browse_button = QPushButton("...")
        self.xml_browse_button.setMaximumWidth(30)
        self.xml_browse_button.clicked.connect(self.select_xml_save_path)
        self.xml_browse_button.setEnabled(False)
        soap_grid.addWidget(self.xml_browse_button, 1, 5)

        self.soap_config_group.setVisible(False)
        layout.addWidget(self.soap_config_group)

        # Input Area
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(LAYOUT["inner_spacing"])

        input_layout.addWidget(QLabel("Keywords (comma-separated):"))
        self.keywords_input = QTextEdit()
        self.keywords_input.setPlaceholderText("Enter keywords...")
        self.keywords_input.setMaximumHeight(80)
        self.keywords_input.setFont(QFont("Segoe UI", LAYOUT["input_font_size"]))
        input_layout.addWidget(self.keywords_input)

        self.search_button = QPushButton("Start Search")
        self.search_button.setStyleSheet(btn_styles["primary"])
        self.search_button.clicked.connect(self.start_search)
        input_layout.addWidget(self.search_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        input_layout.addWidget(self.progress_bar)

        layout.addWidget(input_group)

        # Results Splitter
        self.results_splitter = QSplitter(Qt.Orientation.Horizontal)

        # TreeView for classifications
        tree_container = QGroupBox("Classifications")
        tree_layout = QVBoxLayout(tree_container)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Classification", "Count"])
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.tree_widget.setSortingEnabled(True)
        tree_layout.addWidget(self.tree_widget)
        self.results_splitter.addWidget(tree_container)

        # Detail view for selected classification
        detail_container = QGroupBox("Associated Titles")
        detail_layout = QVBoxLayout(detail_container)
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        detail_layout.addWidget(self.detail_view)
        self.results_splitter.addWidget(detail_container)

        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 2)

        layout.addWidget(self.results_splitter)

        # Debug output
        self.debug_tabs = QTabWidget()
        self.raw_response_view = QTextEdit()
        self.raw_response_view.setReadOnly(True)
        self.raw_response_view.setMaximumHeight(120)
        self.raw_response_view.setFont(QFont("Courier", 9))
        self.debug_tabs.addTab(self.raw_response_view, "Raw Responses (Debug)")
        self.debug_tabs.setVisible(False)
        layout.addWidget(self.debug_tabs)

    def set_keywords(self, keywords: str):
        """Set keywords in the search input"""
        self.keywords_input.setPlainText(keywords)

    def update_num_results(self, value):
        self.num_label.setText(f"Results: {value}")

    def on_mode_changed(self, index):
        is_soap_mode = index == 1
        self.soap_config_group.setVisible(is_soap_mode)

    def on_save_xml_toggled(self, checked):
        self.xml_browse_button.setEnabled(checked)

    def select_xml_save_path(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory for XML Debug Files", ""
        )
        if directory:
            self.xml_path_display.setText(directory)

    def start_search(self):
        keywords_text = self.keywords_input.toPlainText().strip()
        if not keywords_text:
            QMessageBox.warning(
                self, "Input Error", "Please enter at least one keyword."
            )
            return

        keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
        unique_keywords = list(dict.fromkeys(keywords))

        self.search_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.detail_view.clear()

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
            )
            self.current_worker.raw_response.connect(self.append_raw_response)
        else:  # Web Catalog mode
            self.current_worker = UBSearchWorker(unique_keywords)

        self.current_worker.progress_updated.connect(self.update_progress)
        self.current_worker.result_ready.connect(self.display_results)
        self.current_worker.error_occurred.connect(self.handle_error)
        self.current_worker.status_updated.connect(self.append_raw_response)
        self.current_worker.finished.connect(self.search_finished)
        self.current_worker.setNumResults(self.num_results.value())
        self.current_worker.start()

    def update_progress(self, current, total):
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)

    def display_results(self, results):
        self.detail_view.clear()

        if isinstance(results, dict) and "method" in results:
            self._display_biblio_results(results)
        else:
            self._display_web_results(results)

    def _display_biblio_results(self, results: dict):
        self.biblio_results = results
        classifications = results.get("classifications", {})

        self.tree_widget.clear()
        for class_key in sorted(classifications.keys()):
            class_data = classifications[class_key]
            count = class_data.get("count", 0)

            item = QTreeWidgetItem([class_key, str(count)])
            item.setData(0, Qt.ItemDataRole.UserRole, class_key)
            item.setData(0, Qt.ItemDataRole.UserRole + 1, "biblio")
            self.tree_widget.addTopLevelItem(item)

        self.append_raw_response(
            f"Displayed BiblioClient results: {len(classifications)} classifications"
        )

    def _display_web_results(self, results: dict):
        self.tree_widget.clear()

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

        for key in sorted(number_mapping.keys()):
            count = len(number_mapping[key])
            item = QTreeWidgetItem([key, str(count)])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            self.tree_widget.addTopLevelItem(item)

    def on_tree_item_clicked(self, item, column):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        method_type = item.data(0, Qt.ItemDataRole.UserRole + 1)

        if key:
            if method_type == "biblio":
                self.display_biblio_titles(key)
            else:
                self.fetch_additional_titles(key)

    def display_biblio_titles(self, classification_key: str):
        self.detail_view.clear()

        if not self.biblio_results:
            self.detail_view.append("No BiblioClient results available")
            return

        classifications = self.biblio_results.get("classifications", {})
        class_data = classifications.get(classification_key)

        if not class_data:
            self.detail_view.append(f"No data for {classification_key}")
            return

        titles = class_data.get("titles", [])

        html = [f"<h3>Titles for {classification_key}:</h3>"]
        html.append(f"<p><b>Count:</b> {class_data.get('count', 0)}</p>")

        if titles:
            html.append("<h4>Title List:</h4><ul>")
            for title in titles:
                html.append(f"<li>{title}</li>")
            html.append("</ul>")
        else:
            html.append("<p><i>No titles available</i></p>")

        self.detail_view.setHtml("".join(html))

    def fetch_additional_titles(self, classification):
        self.detail_view.clear()
        self.detail_view.append(f"Loading additional titles for {classification}...")

        self.additional_worker = AdditionalTitlesWorker(classification)
        self.additional_worker.titles_ready.connect(self.display_additional_titles)
        self.additional_worker.error_occurred.connect(self.handle_error)
        self.additional_worker.setNumResults(self.num_results.value())
        self.additional_worker.start()

    def display_additional_titles(self, titles):
        self.detail_view.clear()

        if not titles:
            self.detail_view.append("No additional titles found.")
            return

        html = ["<h3>Found Titles:</h3><ul>"]
        for title in titles:
            html.append(
                f'<li><a href="{title["url"]}">{title["title"]}</a> '
                f'({title["year"]})</li>'
            )
        html.append("</ul>")

        self.detail_view.setHtml("".join(html))

    def append_raw_response(self, response_text):
        if not self.show_raw_response_checkbox.isChecked():
            return

        self.debug_tabs.setVisible(True)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.raw_response_view.append(f"[{timestamp}] {response_text}")
        self.raw_response_view.verticalScrollBar().setValue(
            self.raw_response_view.verticalScrollBar().maximum()
        )

    def handle_error(self, error_message):
        self.logger.error(f"Error: {error_message}")
        self.append_raw_response(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Error", error_message)

    def search_finished(self):
        self.search_button.setEnabled(True)
        self.progress_bar.setVisible(False)


# ============================================================================
# DK Statistics Panel Widget
# ============================================================================

class DKStatisticsPanel(QWidget):
    """Panel for displaying DK classification statistics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(get_main_stylesheet())

        layout = QVBoxLayout(self)
        layout.setSpacing(LAYOUT["spacing"])
        layout.setContentsMargins(0, 0, 0, 0)

        # Deduplication Stats
        self.dedup_label = QLabel()
        self.dedup_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.dedup_label)

        # Top 10 Table
        layout.addWidget(QLabel("<b>Top 10 Most Frequent Classifications:</b>"))
        self.top10_table = QTableWidget()
        self.top10_table.setColumnCount(4)
        self.top10_table.setHorizontalHeaderLabels([
            "DK Code", "Occurrences", "Keywords", "Confidence"
        ])
        self.top10_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.top10_table.horizontalHeader().setStretchLastSection(True)
        self.top10_table.setMinimumHeight(150)
        layout.addWidget(self.top10_table)

        # Clear button
        clear_btn = QPushButton("Clear Statistics")
        clear_btn.setStyleSheet(get_button_styles()["secondary"])
        clear_btn.clicked.connect(self.clear_statistics)
        layout.addWidget(clear_btn)

    def update_statistics(self, dk_statistics: dict):
        """Update the statistics panel with new data"""
        if not dk_statistics:
            self.dedup_label.setText("No statistics available.")
            self.top10_table.setRowCount(0)
            return

        # Deduplication summary
        dedup = dk_statistics.get("deduplication_stats", {})
        if dedup:
            dedup_html = (
                f"<b>Deduplication:</b> {dedup.get('original_count', 0)} → "
                f"<b>{dk_statistics.get('total_classifications', 0)}</b> "
                f"(Rate: <b>{dedup.get('deduplication_rate', '0%')}</b>)"
            )
            self.dedup_label.setText(dedup_html)

        # Top 10 Table
        most_frequent = dk_statistics.get("most_frequent", [])
        self.top10_table.setRowCount(len(most_frequent))

        for row, item in enumerate(most_frequent):
            # DK Code
            self.top10_table.setItem(row, 0, QTableWidgetItem(item.get('dk', 'unknown')))

            # Count
            count_item = QTableWidgetItem(str(item.get('count', 0)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight)
            self.top10_table.setItem(row, 1, count_item)

            # Keywords
            keywords = item.get('keywords', [])
            self.top10_table.setItem(row, 2, QTableWidgetItem(', '.join(keywords[:3])))

            # Confidence (Color-coded)
            unique_titles = item.get('unique_titles', item.get('count', 0))
            text_color, bg_color, label, bar = get_confidence_style(unique_titles)

            conf_item = QTableWidgetItem(f"{bar} {label}")
            conf_item.setBackground(QColor(bg_color))
            self.top10_table.setItem(row, 3, conf_item)

        self.top10_table.resizeColumnsToContents()

    def clear_statistics(self):
        self.dedup_label.setText("")
        self.top10_table.setRowCount(0)


# ============================================================================
# Unified DK Analysis Tab
# ============================================================================

class DkAnalysisUnifiedTab(AbstractTab):
    """
    Unified DK Analysis Tab combining:
    - LLM Classification (from AbstractTab)
    - DK Statistics (from DkClassificationTab)
    - UB Catalog Search (from UBSearchTab)
    Claude Generated
    """

    # Additional signals
    dk_search_completed = pyqtSignal(dict)  # Emits UB search results

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService,
        cache_manager: UnifiedKnowledgeManager,
        pipeline_manager: PipelineManager,
        main_window: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            alima_manager,
            llm_service,
            cache_manager,
            pipeline_manager,
            main_window,
            parent,
        )

        # Set task to dk_classification
        self.set_task("dk_classification")
        self.need_keywords = True

        # Update group box title
        self.input_group.setTitle("DK Classification Input")

        # Increase keywords edit height
        self.keywords_edit.setMinimumHeight(150)
        self.keywords_edit.setMaximumHeight(300)

        # Store original keywords input
        self._original_keywords_input = None

        # Add extended UI components after parent setup
        self._setup_extended_ui()

    def _setup_extended_ui(self):
        """Add UB Search and Statistics panels to the layout"""
        btn_styles = get_button_styles()

        # Find the main_splitter and modify it
        # We'll add additional panels after the results area

        # Create statistics panel
        self.stats_panel = DKStatisticsPanel()

        # Create UB search panel
        self.ub_search_panel = UBSearchPanel()

        # Create a tab widget for the lower section
        self.lower_tabs = QTabWidget()

        # Statistics Tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setContentsMargins(LAYOUT["margin"], LAYOUT["margin"],
                                        LAYOUT["margin"], LAYOUT["margin"])
        stats_layout.addWidget(self.stats_panel)
        self.lower_tabs.addTab(stats_tab, "📊 Statistics")

        # UB Search Tab
        ub_tab = QWidget()
        ub_layout = QVBoxLayout(ub_tab)
        ub_layout.setContentsMargins(LAYOUT["margin"], LAYOUT["margin"],
                                     LAYOUT["margin"], LAYOUT["margin"])
        ub_layout.addWidget(self.ub_search_panel)
        self.lower_tabs.addTab(ub_tab, "🔍 UB Catalog")

        # Insert after the main_splitter in the parent layout
        # Find the parent of main_splitter
        parent_layout = self.main_splitter.parent().layout()
        if parent_layout:
            # Add the lower tabs after the main_splitter
            parent_layout.addWidget(self.lower_tabs)

        # Add transfer buttons between panels
        self._add_transfer_buttons()

    def _add_transfer_buttons(self):
        """Add buttons to transfer data between panels"""
        btn_styles = get_button_styles()

        # Create button bar
        button_bar = QHBoxLayout()

        transfer_btn = QPushButton("→ Transfer Keywords to UB Search")
        transfer_btn.setStyleSheet(btn_styles["secondary"])
        transfer_btn.clicked.connect(self._transfer_keywords_to_ub_search)
        button_bar.addWidget(transfer_btn)

        button_bar.addStretch()

        # Insert before the lower_tabs
        parent_layout = self.lower_tabs.parent().layout()
        if parent_layout:
            # Find index of lower_tabs and insert before
            idx = parent_layout.indexOf(self.lower_tabs)
            if idx >= 0:
                # Create a container widget for the button bar
                btn_container = QWidget()
                btn_container.setLayout(button_bar)
                parent_layout.insertWidget(idx, btn_container)

    def _transfer_keywords_to_ub_search(self):
        """Transfer keywords from the keywords_edit to UB search panel"""
        keywords_text = self.keywords_edit.toPlainText().strip()

        # Extract keywords from formatted DK results if necessary
        if "DK" in keywords_text or "|DK" in keywords_text:
            # Already formatted DK results - use as-is
            pass

        self.ub_search_panel.set_keywords(keywords_text)

    # Override set_keywords to handle DK results formatting
    def set_keywords(self, keywords: Any):
        """Enhanced keywords setter for DK analysis"""
        self._original_keywords_input = keywords
        if isinstance(keywords, list) and len(keywords) > 0 and isinstance(keywords[0], dict):
            formatted_text = PipelineResultFormatter.format_dk_results_for_prompt(keywords)
            super().set_keywords(formatted_text)
        else:
            super().set_keywords(str(keywords))

    def restore_keywords_input(self):
        """Restore original DK search results after analysis"""
        if self._original_keywords_input is not None:
            self.set_keywords(self._original_keywords_input)

    def on_analysis_completed(self, step):
        """Handle completion — restore DK keywords for reuse"""
        super().on_analysis_completed(step)
        self.restore_keywords_input()

    @pyqtSlot(object)
    def update_statistics(self, analysis_state):
        """Update statistics panel from pipeline results"""
        if not analysis_state:
            return

        if hasattr(analysis_state, 'dk_statistics') and analysis_state.dk_statistics:
            self.stats_panel.update_statistics(analysis_state.dk_statistics)

    @pyqtSlot(object)
    def update_data(self, analysis_state):
        """Alias for receive_pipeline_results - backward compatibility with DkClassificationTab"""
        self.receive_pipeline_results(analysis_state)

    @pyqtSlot(object)
    def receive_pipeline_results(self, analysis_state):
        """Receive complete pipeline results and populate all panels"""
        if not analysis_state:
            return

        # Update abstract
        if hasattr(analysis_state, 'original_abstract') and analysis_state.original_abstract:
            self.set_abstract(analysis_state.original_abstract)

        # 1. LLM Classification: DK search results (DK codes with titles)
        #    This is INPUT for the LLM to analyze and select classifications
        if hasattr(analysis_state, 'dk_search_results_flattened') and analysis_state.dk_search_results_flattened:
            self.set_keywords(analysis_state.dk_search_results_flattened)
        elif hasattr(analysis_state, 'dk_search_results') and analysis_state.dk_search_results:
            self.set_keywords(analysis_state.dk_search_results)

        # Update statistics
        self.update_statistics(analysis_state)

        # Update results if DK LLM analysis exists
        if hasattr(analysis_state, 'dk_llm_analysis') and analysis_state.dk_llm_analysis:
            self.display_llm_response(analysis_state.dk_llm_analysis.response_full_text)
            self.add_external_analysis_to_history(analysis_state)

        # 2. UB Search: GND keywords from final analysis
        #    These are the SEARCH TERMS for finding DK codes in the catalog
        if hasattr(analysis_state, 'final_llm_analysis') and analysis_state.final_llm_analysis:
            gnd_keywords = analysis_state.final_llm_analysis.extracted_gnd_keywords
            if gnd_keywords:
                # Convert list to comma-separated string
                if isinstance(gnd_keywords, list):
                    keywords_str = ", ".join(gnd_keywords)
                else:
                    keywords_str = str(gnd_keywords)
                self.ub_search_panel.set_keywords(keywords_str)

    def update_keywords(self, keywords: str):
        """Update keywords in the UB search panel - backward compatibility with UBSearchTab"""
        if hasattr(self, 'ub_search_panel'):
            self.ub_search_panel.set_keywords(keywords)

    def set_abstract(self, abstract: str):
        """Set abstract text - backward compatibility with AbstractTab"""
        super().set_abstract(abstract)