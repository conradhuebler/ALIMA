import io
import sys
import json
import requests
import ijson
from collections import Counter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QLineEdit, QPushButton, QTextEdit, QLabel, QHBoxLayout, QTabWidget)

class SearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GND Subject Headings Suche')
        self.setGeometry(100, 100, 1200, 800)

        # Zentrale Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Suchbereich
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('Suchbegriffe (durch Komma getrennt)')
        self.search_button = QPushButton('Suchen')
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)

        # Statuszeile
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        # Tabs für verschiedene Ansichten
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Ergebnisanzeige
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.tabs.addTab(self.results_display, "Einzelergebnisse")

        # Gemeinsame Ergebnisse
        self.common_results_display = QTextEdit()
        self.common_results_display.setReadOnly(True)
        self.tabs.addTab(self.common_results_display, "Gemeinsame Begriffe")

        # Gesamtanalyse
        self.total_analysis_display = QTextEdit()
        self.total_analysis_display.setReadOnly(True)
        self.tabs.addTab(self.total_analysis_display, "Gesamtanalyse")

        # Verbindungen
        self.search_button.clicked.connect(self.perform_search)
        # Enter-Taste Binding
        self.search_input.returnPressed.connect(self.perform_search)


    def extract_subject_headings(self, item):
        """Extrahiert nur SubjectHeadings mit GND-Links"""
        subject_headings = []

        if 'subject' in item:
            for subject in item['subject']:
                if isinstance(subject, dict):
                    # Fall 1: ComplexSubject mit componentList
                    if 'componentList' in subject:
                        for component in subject['componentList']:
                            if isinstance(component, dict) and \
                            component.get('type') == ['SubjectHeading'] and \
                            component.get('id', '').startswith('https://d-nb.info/gnd/'):
                                label = component.get('label', '')
                                gnd_id = component.get('id', '')
                                if label:
                                    subject_headings.append((label, gnd_id))

                    # Fall 2: Direktes SubjectHeading
                    elif subject.get('type') == ['SubjectHeading'] and \
                        subject.get('id', '').startswith('https://d-nb.info/gnd/'):
                        label = subject.get('label', '')
                        gnd_id = subject.get('id', '')
                        if label:
                            subject_headings.append((label, gnd_id))

        return subject_headings

    def perform_search(self):
        self.results_display.clear()
        self.common_results_display.clear()
        self.total_analysis_display.clear()

        # Deaktiviere Button und ändere Text
        self.search_button.setEnabled(False)
        self.search_button.setText("Suche läuft...")

        search_terms = [term.strip() for term in self.search_input.text().split(',') if term.strip()]

        if not search_terms:
            self.status_label.setText("Bitte geben Sie mindestens einen Suchbegriff ein.")
            self.search_button.setEnabled(True)
            self.search_button.setText("Suchen")
            return

        term_results = {}
        total_counter = Counter()

        for i, term in enumerate(search_terms, 1):
            self.status_label.setText(f"Verarbeite Suchbegriff {i} von {len(search_terms)}: {term}")
            QApplication.processEvents()

            try:
                url = f"https://lobid.org/resources/search?q={term}&format=jsonl"
                response = requests.get(url, stream=True)

                if response.status_code != 200:
                    self.results_display.append(f"Fehler bei der API-Anfrage für {term}: {response.status_code}")
                    term_results[term] = {'headings': set(), 'counter': Counter(), 'total': 0}
                    continue

                self.status_label.setText(f"Analysiere Ergebnisse für: {term}")
                QApplication.processEvents()

                # Verarbeitung der JSON Lines
                subject_headings = []
                total_items = 0

                # Dekodiere den Stream und verarbeite zeilenweise
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        headings = self.extract_subject_headings(item)
                        subject_headings.extend(headings)
                        total_items += 1
                        if total_items % 100 == 0:
                            self.status_label.setText(f"Verarbeite Eintrag {total_items} für: {term}")
                            QApplication.processEvents()
                    except json.JSONDecodeError as e:
                        self.results_display.append(f"Fehler beim Parsen einer Zeile: {str(e)}")
                        continue

                # Update total counter
                total_counter.update(subject_headings)

                # Speichere Ergebnisse für diesen Term
                term_results[term] = {
                    'headings': set(subject_headings),
                    'counter': Counter(subject_headings),
                    'total': total_items
                }

                # Zeige individuelle Ergebnisse
                self.results_display.append(f"\nSuche nach: {term}\n{'='*50}")
                self.results_display.append(f"Gefundene Dokumente: {total_items}")

                if subject_headings:
                    unique_headings = set(subject_headings)
                    self.results_display.append(f"Gefundene GND Subject Headings: {len(unique_headings)}")

                    term_counter = Counter(subject_headings)
                    self.results_display.append("\nTop GND Subject Headings (mit Häufigkeit):")

                    for (label, gnd_id), count in term_counter.most_common(10):
                        self.results_display.append(f"\nLabel: {label}")
                        self.results_display.append(f"GND-ID: {gnd_id}")
                        self.results_display.append(f"Häufigkeit: {count}")

                    if len(unique_headings) > 10:
                        self.results_display.append(f"\n... und {len(unique_headings) - 10} weitere")
                else:
                    self.results_display.append("\nKeine GND Subject Headings gefunden.")

            except Exception as e:
                self.results_display.append(f"\nFehler bei der Verarbeitung von {term}: {str(e)}")
                import traceback
                self.results_display.append(traceback.format_exc())
                term_results[term] = {'headings': set(), 'counter': Counter(), 'total': 0}

        self.status_label.setText("Verarbeite Gesamtanalyse...")
        QApplication.processEvents()

        # Zeige Überschneidungen bei mehreren Suchbegriffen
        if len(search_terms) > 1:
            self.show_common_results(term_results, search_terms)

        # Zeige Gesamtanalyse
        self.show_total_analysis(total_counter)

        # Button wieder aktivieren
        self.search_button.setEnabled(True)
        self.search_button.setText("Suchen")
        self.status_label.setText("Suche abgeschlossen!")


    def show_common_results(self, term_results, search_terms):
        """Zeigt die Überschneidungen zwischen den Suchergebnissen"""
        self.common_results_display.clear()
        self.common_results_display.append("Gemeinsame GND Subject Headings\n")

        if term_results:
            # Finde Überschneidungen zwischen allen Suchbegriffen
            all_common = set.intersection(*[term_results[term]['headings'] for term in search_terms])

            if all_common:
                self.common_results_display.append(
                    f"\nIn ALLEN Suchbegriffen gefunden ({len(all_common)} Begriffe):\n{'='*40}")
                for label, gnd_id in sorted(all_common):
                    self.common_results_display.append(f"\nLabel: {label}")
                    self.common_results_display.append(f"GND-ID: {gnd_id}")
            else:
                self.common_results_display.append(
                    "\nKeine gemeinsamen GND Subject Headings in allen Suchbegriffen gefunden.")

            # Zeige paarweise Überschneidungen
            self.common_results_display.append(
                f"\n\nÜberschneidungen zwischen Suchbegriff-Paaren:\n{'='*40}")
            for i, term1 in enumerate(search_terms):
                for term2 in search_terms[i+1:]:
                    if term1 in term_results and term2 in term_results:
                        intersection = term_results[term1]['headings'] & term_results[term2]['headings']
                        if intersection:
                            self.common_results_display.append(
                                f"\nGemeinsame Begriffe in '{term1}' und '{term2}' ({len(intersection)} Begriffe):")
                            for label, gnd_id in sorted(intersection):
                                self.common_results_display.append(f"\nLabel: {label}")
                                self.common_results_display.append(f"GND-ID: {gnd_id}")

    def show_total_analysis(self, total_counter):
        """Zeigt die Gesamtanalyse aller gefundenen GND-Begriffe"""
        self.total_analysis_display.append("Gesamtanalyse aller GND-Begriffe\n" + "="*40)

        if total_counter:
            total_unique = len(total_counter)
            total_occurrences = sum(total_counter.values())

            self.total_analysis_display.append(f"\nGefundene eindeutige GND-Begriffe: {total_unique}")
            self.total_analysis_display.append(f"Gesamtanzahl Vorkommen: {total_occurrences}")
            self.total_analysis_display.append(f"Durchschnittliche Verwendung: {total_occurrences/total_unique:.2f}")

            self.total_analysis_display.append("\nTop 20 GND-Begriffe über alle Suchen:")
            for (label, gnd_id), count in total_counter.most_common(20):
                self.total_analysis_display.append(f"\nLabel: {label}")
                self.total_analysis_display.append(f"GND-ID: {gnd_id}")
                self.total_analysis_display.append(f"Gesamthäufigkeit: {count}")
                percentage = (count / total_occurrences) * 100
                self.total_analysis_display.append(f"Anteil: {percentage:.1f}%")
        else:
            self.total_analysis_display.append("\nKeine GND-Begriffe gefunden.")

def main():
    app = QApplication(sys.argv)
    window = SearchApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
