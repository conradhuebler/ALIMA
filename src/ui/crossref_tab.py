# crossref_tab.py

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QMessageBox,
    QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal
import logging

from ..core.crossref_worker import CrossrefWorker


class CrossrefTab(QWidget):
    result_abstract = pyqtSignal(str)
    result_keywords = pyqtSignal(str)

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
        self.doi_input.setPlaceholderText(
            "Gib die DOI ein, z.B. 10.1007/978-3-031-47390-6"
        )
        self.fetch_button = QPushButton("Abfrage starten")
        self.fetch_button.clicked.connect(self.perform_search)
        self.fetch_button.setToolTip(
            "Starte die Abfrage der Crossref API bzw. via Springer EBooks"
        )
        self.fetch_button.setEnabled(True)
        self.fetch_button.setShortcut("Ctrl+Return")
        input_layout.addWidget(QLabel("DOI:"))
        input_layout.addWidget(self.doi_input)
        input_layout.addWidget(self.fetch_button)

        # Ergebnisanzeige mit Tabs
        self.result_tabs = QTabWidget()

        # Hauptergebnistab
        self.main_result = QTextEdit()
        self.main_result.setReadOnly(True)
        self.main_result.setPlaceholderText("Hier werden die Ergebnisse angezeigt...")
        self.result_tabs.addTab(self.main_result, "Hauptergebnisse")

        # Tab für About
        self.about_result = QTextEdit()
        self.about_result.setReadOnly(True)
        self.about_result.setPlaceholderText("Über dieses Buch...")
        self.result_tabs.addTab(self.about_result, "Über das Buch")

        # Tab für Inhaltsverzeichnis
        self.toc_result = QTextEdit()
        self.toc_result.setReadOnly(True)
        self.toc_result.setPlaceholderText("Inhaltsverzeichnis...")
        self.result_tabs.addTab(self.toc_result, "Inhaltsverzeichnis")

        # Tab für Keywords
        self.keywords_result = QTextEdit()
        self.keywords_result.setReadOnly(True)
        self.keywords_result.setPlaceholderText("Schlüsselwörter...")
        self.result_tabs.addTab(self.keywords_result, "Schlüsselwörter")

        # Standardmäßig alle Tabs außer dem Haupttab ausblenden
        self.result_tabs.setTabVisible(1, False)
        self.result_tabs.setTabVisible(2, False)
        self.result_tabs.setTabVisible(3, False)

        layout.addLayout(input_layout)
        layout.addWidget(self.result_tabs)

        self.setLayout(layout)

    def perform_search(self):
        """Startet die API-Abfrage."""
        doi = self.doi_input.text().strip()
        if not doi:
            QMessageBox.warning(
                self, "Eingabefehler", "Bitte gib eine gültige DOI ein."
            )
            return

        # Starte den Worker
        self.fetch_button.setEnabled(False)
        self.main_result.clear()
        self.about_result.clear()
        self.toc_result.clear()
        self.keywords_result.clear()

        self.main_result.append("Starte Abfrage...")

        self.worker = CrossrefWorker(doi)
        self.worker.result_ready.connect(self.display_results)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(lambda: self.fetch_button.setEnabled(True))
        self.worker.run()

    def display_results(self, results: dict):
        """Zeigt die Ergebnisse der API-Abfrage an."""
        self.main_result.clear()

        # Standardergebnisse anzeigen
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
        self.main_result.setHtml(formatted_result)
        main_text = ""
        keywords_text = ""
        # Wenn Springer-spezifische Informationen vorhanden sind, zeige sie in den entsprechenden Tabs an
        if "About" in results:
            self.about_result.setHtml(f"<p>{results.get('About')}</p>")
            main_text += f"Zusammenfassung:\n{results.get('About')}"
            self.result_tabs.setTabVisible(1, True)  # About-Tab anzeigen
        else:
            self.result_tabs.setTabVisible(1, False)  # About-Tab ausblenden

        if "Table of Contents" in results:
            self.toc_result.setHtml(f"<p>{results.get('Table of Contents')}</p>")
            main_text += f"\nInhaltsverzeichnis:\n{results.get('Table of Contents')}\n"
            self.result_tabs.setTabVisible(2, True)  # ToC-Tab anzeigen
        else:
            self.result_tabs.setTabVisible(2, False)  # ToC-Tab ausblenden

        if "Keywords" in results:
            self.keywords_result.setHtml(f"<p>{results.get('Keywords')}</p>")
            keywords_text = results.get("Keywords")
            self.result_tabs.setTabVisible(3, True)  # Keywords-Tab anzeigen
        else:
            self.result_tabs.setTabVisible(3, False)  # Keywords-Tab ausblenden

        # Immer zum ersten Tab wechseln
        self.result_tabs.setCurrentIndex(0)

        # Signal mit dem Abstract emittieren
        self.result_abstract.emit(main_text)
        self.result_keywords.emit(keywords_text)
        self.logger.info(keywords_text)
        self.fetch_button.setEnabled(True)

    def handle_error(self, error_message: str):
        """Behandelt Fehler während der API-Abfrage."""
        self.logger.error(f"Fehler bei der Crossref-Abfrage: {error_message}")
        QMessageBox.critical(self, "API-Fehler", error_message)
        self.main_result.clear()
        self.main_result.append(f"Fehler: {error_message}")
