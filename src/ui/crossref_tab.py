# crossref_tab.py

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton,
    QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
import logging

from ..core.crossref_worker import CrossrefWorker

class CrossrefTab(QWidget):
    result_abstract = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche."""
        layout = QVBoxLayout()

        # Eingabefeld und Button
        input_layout = QHBoxLayout()
        self.doi_input = QLineEdit()
        self.doi_input.setPlaceholderText("Gib die DOI ein, z.B. 10.1007/s42452-023-05466-w")
        self.fetch_button = QPushButton("Abfrage starten")
        self.fetch_button.clicked.connect(self.perform_search)
        input_layout.addWidget(QLabel("DOI:"))
        input_layout.addWidget(self.doi_input)
        input_layout.addWidget(self.fetch_button)

        # Ergebnisanzeige
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setPlaceholderText("Hier werden die Ergebnisse angezeigt...")

        layout.addLayout(input_layout)
        layout.addWidget(self.result_display)

        self.setLayout(layout)

    def perform_search(self):
        """Startet die API-Abfrage."""
        doi = self.doi_input.text().strip()
        if not doi:
            QMessageBox.warning(self, "Eingabefehler", "Bitte gib eine gültige DOI ein.")
            return

        # Starte den Worker
        self.fetch_button.setEnabled(False)
        self.result_display.clear()
        self.result_display.append("Starte Abfrage...")

        self.worker = CrossrefWorker(doi)
        self.worker.result_ready.connect(self.display_results)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(lambda: self.fetch_button.setEnabled(True))
        self.worker.start()

    def display_results(self, results: dict):
        """Zeigt die Ergebnisse der API-Abfrage an."""
        self.result_display.clear()
        formatted_result = (
            f"<b>Titel:</b> {results.get('Title')}</br>"
            f"<b>DOI:</b> {results.get('DOI')}</br>"
            f"<b>Abstract:</b> {results.get('Abstract')}</br>"
            f"<b>Autoren:</b> {results.get('Authors')}</br>"
            f"<b>Verlag:</b> {results.get('Publisher')}</br>"
            f"<b>Veröffentlicht:</b> {results.get('Published')}</br>"
            f"<b>Zeitschrift:</b> {results.get('Container-Title')}</br>"
            f"<b>URL:</b> <a href='{results.get('URL')}'>{results.get('URL')}</a>"
        )
        self.result_display.setHtml(formatted_result)
        self.result_abstract.emit(results.get('Abstract'))

    def handle_error(self, error_message: str):
        """Behandelt Fehler während der API-Abfrage."""
        self.logger.error(f"Fehler bei der Crossref-Abfrage: {error_message}")
        QMessageBox.critical(self, "API-Fehler", error_message)
        self.result_display.clear()
        self.result_display.append(f"Fehler: {error_message}")
