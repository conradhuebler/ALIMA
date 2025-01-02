import sys
import json
import requests
import sqlite3
import re
from datetime import datetime, timedelta
from collections import Counter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QLineEdit, QPushButton, QTextEdit,
                            QLabel, QTabWidget, QGroupBox,QDoubleSpinBox)
from PyQt6.QtCore import Qt

class SearchCache:
    def __init__(self):
        self.conn = sqlite3.connect('search_cache.db')
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS searches (
                    search_term TEXT PRIMARY KEY,
                    results BLOB,
                    timestamp DATETIME
                )
            ''')

    def _convert_for_storage(self, results):
        """Konvertiert Sets und Counter in JSON-serialisierbare Formate"""
        return {
            'headings': list(results['headings']),
            'counter': {str(k): v for k, v in results['counter'].items()},
            'total': results['total']
        }

    def _convert_from_storage(self, stored_results):
        """Konvertiert gespeicherte Daten zurück in Sets und Counter"""
        return {
            'headings': set(tuple(h) for h in stored_results['headings']),
            'counter': Counter({tuple(k.strip('()').split(', ')): v
                              for k, v in stored_results['counter'].items()}),
            'total': stored_results['total']
        }

    def get_cached_results(self, search_term):
        """Holt gecachte Ergebnisse, wenn sie nicht älter als 24 Stunden sind"""
        with self.conn:
            cursor = self.conn.execute(
                'SELECT results, timestamp FROM searches WHERE search_term = ?',
                (search_term,)
            )
            result = cursor.fetchone()

            if result:
                results, timestamp = result
                cached_time = datetime.fromisoformat(timestamp)
                if datetime.now() - cached_time < timedelta(hours=24):
                    stored_results = json.loads(results)
                    return self._convert_from_storage(stored_results)
        return None

    def cache_results(self, search_term, results):
        """Speichert Ergebnisse im Cache"""
        serializable_results = self._convert_for_storage(results)
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO searches (search_term, results, timestamp) VALUES (?, ?, ?)',
                (search_term, json.dumps(serializable_results), datetime.now().isoformat())
            )

class SearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GND Subject Headings Suche')
        self.setGeometry(100, 100, 1200, 800)

        # Initialize cache
        self.cache = SearchCache()

        # Speicher für die aktuellen Ergebnisse
        self.current_results = {
            'term_results': {},
            'total_counter': Counter()
        }

        # Zentrale Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Erster Suchbereich (API-Suche)
        search_group1 = QGroupBox("API-Suche")
        search_layout1 = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('Suchbegriffe (durch Komma getrennt)')
        self.search_button = QPushButton('Suchen')
        search_layout1.addWidget(self.search_input)
        search_layout1.addWidget(self.search_button)
        search_group1.setLayout(search_layout1)
        layout.addWidget(search_group1)

        # Zweiter Suchbereich (Filterung der Ergebnisse)
        search_group2 = QGroupBox("GND Subject Heading Überprüfung")
        search_layout2 = QHBoxLayout()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText('Begriffe zum Überprüfen (durch Komma getrennt oder strukturierter String)')

        # Button-Layout
        button_layout = QHBoxLayout()
        self.filter_button = QPushButton('In aktuellen Ergebnissen prüfen')
        self.cache_check_button = QPushButton('Im Cache prüfen')
        button_layout.addWidget(self.filter_button)
        button_layout.addWidget(self.cache_check_button)

        # Am Anfang der Klasse bei den anderen UI-Elementen:
        # Threshold Spinner
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold (%):")
        self.threshold_spinner = QDoubleSpinBox()
        self.threshold_spinner.setRange(0.1, 100.0)  # Von 0.1% bis 100%
        self.threshold_spinner.setValue(5.0)  # Default 5%
        self.threshold_spinner.setSingleStep(0.1)  # Schrittweite 0.1%
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spinner)

        # Füge das Layout zum Hauptlayout hinzu (wo es am besten passt)
        layout.addLayout(threshold_layout)  # oder wo auch immer dein settings_layout ist

        search_layout2.addWidget(self.filter_input)
        search_layout2.addLayout(button_layout)
        search_group2.setLayout(search_layout2)
        layout.addWidget(search_group2)

        # Verbinde die Buttons
        self.filter_button.clicked.connect(self.filter_results)
        self.cache_check_button.clicked.connect(self.check_gnd_terms)

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


    # Prompt-Vorschau
        self.prompt_display = QTextEdit()
        self.prompt_display.setReadOnly(True)
        self.tabs.addTab(self.prompt_display, "Schlagwortauswahl")

        self.all_subject_headings = set()  # Neue übergeordnete Variable für alle GND-Begriffe

        # GND-Überprüfung
        self.gnd_check_display = QTextEdit()
        self.gnd_check_display.setReadOnly(True)
        self.tabs.addTab(self.gnd_check_display, "GND-Überprüfung")


        # Verbindungen
        self.search_button.clicked.connect(self.perform_search)
        self.search_input.returnPressed.connect(self.perform_search)
        self.filter_button.clicked.connect(self.filter_results)
        self.filter_input.returnPressed.connect(self.filter_results)




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

    def show_common_results(self, term_results, search_terms):
        """Zeigt die Überschneidungen zwischen den Suchergebnissen"""
        self.common_results_display.clear()
        self.common_results_display.append("Gemeinsame GND Subject Headings\n")

        if term_results:
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
                    intersection = term_results[term1]['headings'] & term_results[term2]['headings']
                    if intersection:
                        self.common_results_display.append(
                            f"\nGemeinsame Begriffe in '{term1}' und '{term2}' ({len(intersection)} Begriffe):")
                        for label, gnd_id in sorted(intersection):
                            self.common_results_display.append(f"\nLabel: {label}")
                            self.common_results_display.append(f"GND-ID: {gnd_id}")

    def show_total_analysis(self, counter):
        """Zeigt die Gesamtanalyse aller Suchergebnisse"""
        self.total_analysis_display.clear()
        self.total_analysis_display.append("Gesamtanalyse aller Suchergebnisse\n")

        if counter:
            self.total_analysis_display.append(f"\nTop GND Subject Headings über alle Suchen:\n{'='*40}")
            for (label, gnd_id), count in counter.most_common(20):
                self.total_analysis_display.append(f"\nLabel: {label}")
                self.total_analysis_display.append(f"GND-ID: {gnd_id}")
                self.total_analysis_display.append(f"Häufigkeit: {count}")
        else:
            self.total_analysis_display.append("\nKeine Daten für die Gesamtanalyse verfügbar.")

    def check_gnd_terms(self):
        """Überprüft GND Terms direkt aus dem Cache ohne neue API-Suche"""
        input_text = self.filter_input.text()

        if not input_text:
            self.status_label.setText("Bitte geben Sie Begriffe zum Überprüfen ein.")
            return

        # Prüfe, ob es sich um einen strukturierten String handelt
        if "GND:" in input_text:
            check_terms = self.parse_structured_input(input_text)
        else:
            check_terms = [term.strip() for term in input_text.split(',') if term.strip()]

        self.status_label.setText("Überprüfe GND Subject Headings im Cache...")
        QApplication.processEvents()

        # Hole alle gecachten Ergebnisse
        all_headings = set()
        with self.cache.conn:
            cursor = self.cache.conn.execute('SELECT results FROM searches')
            for (results_json,) in cursor:
                cached_results = json.loads(results_json)
                converted_results = self.cache._convert_from_storage(cached_results)
                all_headings.update(converted_results['headings'])

        # Überprüfe jeden eingegebenen Begriff
        self.gnd_check_display.clear()
        self.gnd_check_display.append("GND Subject Heading Überprüfung (Cache-Analyse):\n")

        for term in check_terms:
            term_lower = term.lower()
            found_matches = []

            for label, gnd_id in all_headings:
                if label.lower() == term_lower:
                    found_matches.append((label, gnd_id))

            if found_matches:
                self.gnd_check_display.append(f"\n★ \"{term}\" existiert als GND Subject Heading:")
                for label, gnd_id in found_matches:
                    self.gnd_check_display.append(f"  Label: {label}")
                    self.gnd_check_display.append(f"  GND-ID: {gnd_id}")
            else:
                self.gnd_check_display.append(f"\n○ \"{term}\" wurde nicht als exaktes GND Subject Heading gefunden.")

        # Wechsle zum GND-Überprüfungs-Tab
        self.tabs.setCurrentIndex(3)
        self.status_label.setText("Cache-Überprüfung abgeschlossen!")



    def parse_structured_input(self, input_string):
        """Parst einen strukturierten String mit GND-IDs in einzelne Terme und deren GND-IDs"""
        terms = []

        # Teile zunächst bei Kommas oder 'und/oder'
        entries = re.split(r',|(?:und/oder)', input_string)

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            # Suche nach GND-IDs
            gnd_ids = re.findall(r'GND:\s*(\d+-\d+)', entry)

            # Extrahiere den Hauptbegriff (alles vor der ersten Klammer)
            main_term = entry.split('(')[0].strip()
            terms.append((main_term, gnd_ids))

            # Wenn alternative Begriffe in Klammern stehen
            alternatives = re.findall(r'\(([^)]+?)(?=\s*GND:|$|\))', entry)
            for alt in alternatives:
                alt = alt.strip()
                if alt and not alt.startswith('GND:'):
                    terms.append((alt, gnd_ids))

        return terms

    def check_gnd_terms(self):
        """Überprüft GND Terms und IDs direkt aus dem Cache"""
        input_text = self.filter_input.text()

        if not input_text:
            self.status_label.setText("Bitte geben Sie Begriffe zum Überprüfen ein.")
            return

        # Parse Input
        if "GND:" in input_text:
            check_items = self.parse_structured_input(input_text)
        else:
            check_items = [(term.strip(), []) for term in input_text.split(',') if term.strip()]

        self.status_label.setText("Überprüfe GND Subject Headings im Cache...")
        QApplication.processEvents()

        # Hole alle gecachten Ergebnisse
        all_headings = set()
        with self.cache.conn:
            cursor = self.cache.conn.execute('SELECT results FROM searches')
            for (results_json,) in cursor:
                cached_results = json.loads(results_json)
                converted_results = self.cache._convert_from_storage(cached_results)
                all_headings.update(converted_results['headings'])

        # Überprüfe jeden eingegebenen Begriff
        self.gnd_check_display.clear()
        self.gnd_check_display.append("GND Subject Heading Überprüfung (Cache-Analyse):\n")

        found_terms = []  # Liste für erfolgreich gefundene Terme

        for term, expected_gnds in check_items:
            term_lower = term.lower()
            found_matches = []

            for label, gnd_id in all_headings:
                if label.lower() == term_lower:
                    # Extrahiere nur die ID-Nummer aus der GND-URL
                    gnd_number = gnd_id.split('/')[-1]
                    found_matches.append((label, gnd_number))

            if found_matches:
                self.gnd_check_display.append(f"\n★ \"{term}\" existiert als GND Subject Heading:")
                for label, gnd_number in found_matches:
                    gnd_status = ""
                    if expected_gnds:
                        if gnd_number in expected_gnds:
                            gnd_status = " ✓ (GND-ID stimmt überein)"
                            found_terms.append(f"{label} (GND: {gnd_number})")
                        else:
                            gnd_status = f" ⚠ (Erwartete GND-ID(s): {', '.join(expected_gnds)})"
                    else:
                        found_terms.append(f"{label} (GND: {gnd_number})")

                    self.gnd_check_display.append(f"  Label: {label}")
                    self.gnd_check_display.append(f"  GND-ID: {gnd_number}{gnd_status}")
            else:
                self.gnd_check_display.append(f"\n○ \"{term}\" wurde nicht als exaktes GND Subject Heading gefunden.")

        # Zusammenfassung der gefundenen Terme
        if found_terms:
            self.gnd_check_display.append("\n\nListe der erfolgreich gefundenen GND Subject Headings:")
            for term in found_terms:
                self.gnd_check_display.append(f"• {term}")

        # Wechsle zum GND-Überprüfungs-Tab
        self.tabs.setCurrentIndex(3)
        self.status_label.setText("Cache-Überprüfung abgeschlossen!")


    def filter_results(self):
        """Markiert exakte GND Subject Headings in den Suchergebnissen"""
        input_text = self.filter_input.text()

        # Prüfe, ob es sich um einen strukturierten String handelt
        if "GND:" in input_text:
            check_terms = self.parse_structured_input(input_text)
        else:
            check_terms = [term.strip() for term in input_text.split(',') if term.strip()]

        if not check_terms:
            self.status_label.setText("Bitte geben Sie mindestens einen Begriff zum Überprüfen ein.")
            return

        if not self.current_results['term_results']:
            self.status_label.setText("Bitte führen Sie zuerst eine API-Suche durch.")
            return

        self.status_label.setText("Überprüfe GND Subject Headings...")
        QApplication.processEvents()

        # Sammle alle unique Subject Headings aus den aktuellen Ergebnissen
        all_headings = set()
        for results in self.current_results['term_results'].values():
            all_headings.update(results['headings'])

        # Überprüfe jeden eingegebenen Begriff
        self.gnd_check_display.clear()
        self.gnd_check_display.append("GND Subject Heading Überprüfung:\n")

        for term in check_terms:
            term_lower = term.lower()
            found_matches = []

            for label, gnd_id in all_headings:
                if label.lower() == term_lower:
                    found_matches.append((label, gnd_id))

            if found_matches:
                self.gnd_check_display.append(f"\n★ \"{term}\" existiert als GND Subject Heading:")
                for label, gnd_id in found_matches:
                    self.gnd_check_display.append(f"  Label: {label}")
                    self.gnd_check_display.append(f"  GND-ID: {gnd_id}")
            else:
                self.gnd_check_display.append(f"\n○ \"{term}\" wurde nicht als exaktes GND Subject Heading gefunden.")

        # Wechsle zum GND-Überprüfungs-Tab
        self.tabs.setCurrentIndex(3)  # Index 3 für den vierten Tab
        self.status_label.setText("Überprüfung abgeschlossen!")



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
            subject_headings = []

            try:
                # Prüfe erst Cache
                cached_results = self.cache.get_cached_results(term)
                if cached_results:
                    self.status_label.setText(f"Verwende gecachte Ergebnisse für: {term}")
                    term_results[term] = cached_results

                    # Zeige die gleichen Ergebnisse wie bei einer neuen Suche
                    self.results_display.append(f"\nSuche nach: {term}\n{'='*50}")
                    self.results_display.append(f"Gefundene Dokumente: {cached_results['total']}")

                    if cached_results['headings']:
                        self.results_display.append(f"Gefundene GND Subject Headings: {len(cached_results['headings'])}")
                        self.results_display.append("\nTop GND Subject Headings (mit Häufigkeit):")

                        for (label, gnd_id), count in cached_results['counter'].most_common(10):
                            self.results_display.append(f"\nLabel: {label}")
                            self.results_display.append(f"GND-ID: {gnd_id}")
                            self.results_display.append(f"Häufigkeit: {count}")

                        if len(cached_results['headings']) > 10:
                            self.results_display.append(f"\n... und {len(cached_results['headings']) - 10} weitere")
                    else:
                        self.results_display.append("\nKeine GND Subject Headings gefunden.")

                    # Update total counter
                    total_counter.update(cached_results['counter'])
                    continue

                # Wenn keine Cache-Ergebnisse, führe API-Suche durch
                url = f"https://lobid.org/resources/search?q={term}&format=jsonl"
                response = requests.get(url, stream=True)

                if response.status_code != 200:
                    self.results_display.append(f"Fehler bei der API-Anfrage für {term}: {response.status_code}")
                    term_results[term] = {'headings': set(), 'counter': Counter(), 'total': 0}
                    continue

                self.status_label.setText(f"Analysiere Ergebnisse für: {term}")
                QApplication.processEvents()

                # Verarbeitung der JSON Lines
                total_items = 0

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

                # Erstelle Counter und update total counter
                term_counter = Counter(subject_headings)
                total_counter.update(term_counter)

                # Speichere Ergebnisse für diesen Term
                term_results[term] = {
                    'headings': set(subject_headings),
                    'counter': term_counter,
                    'total': total_items
                }
                #self.all_subject_headings.update(term_results['headings'])
                # Cache die Ergebnisse
                self.cache.cache_results(term, term_results[term])

                # Zeige individuelle Ergebnisse
                self.results_display.append(f"\nSuche nach: {term}\n{'='*50}")
                self.results_display.append(f"Gefundene Dokumente: {total_items}")

                if subject_headings:
                    unique_headings = set(subject_headings)
                    self.results_display.append(f"Gefundene GND Subject Headings: {len(unique_headings)}")

                    # Gruppiere die Ergebnisse
                    exact_matches = []
                    partial_matches = []
                    other_matches = []
                    search_term_lower = term.lower()

                    # Durchlaufe alle Einträge im Counter
                    for heading, count in term_counter.items():
                        label, gnd_id = heading  # heading ist ein Tuple (label, gnd_id)
                        if label.lower() == search_term_lower:
                            exact_matches.append((label, gnd_id, count))
                        elif search_term_lower in label.lower():
                            partial_matches.append((label, gnd_id, count))
                        else:
                            other_matches.append((label, gnd_id, count))

                    self.all_subject_headings.update(label)

                    # Sortiere nach Häufigkeit
                    exact_matches.sort(key=lambda x: x[2], reverse=True)
                    partial_matches.sort(key=lambda x: x[2], reverse=True)
                    other_matches.sort(key=lambda x: x[2], reverse=True)

                    # Zeige exakte Übereinstimmungen
                    if exact_matches:
                        self.results_display.append("\nExakte Übereinstimmungen:")
                        for label, gnd_id, count in exact_matches:
                            self.results_display.append(f"\n★ Label: {label}")
                            self.results_display.append(f"  GND-ID: {gnd_id}")
                            self.results_display.append(f"  Häufigkeit: {count}")

                    # Zeige partielle Übereinstimmungen
                    if partial_matches:
                        self.results_display.append("\nPartielle Übereinstimmungen:")
                        for label, gnd_id, count in partial_matches[:10]:  # Limitiere auf top 10
                            self.results_display.append(f"\n○ Label: {label}")
                            self.results_display.append(f"  GND-ID: {gnd_id}")
                            self.results_display.append(f"  Häufigkeit: {count}")

                    # Zeige andere häufige Begriffe
                    if other_matches:
                        self.results_display.append("\nAndere häufige Subject Headings:")
                        for label, gnd_id, count in other_matches[:10]:  # Limitiere auf top 10
                            self.results_display.append(f"\n  Label: {label}")
                            self.results_display.append(f"  GND-ID: {gnd_id}")
                            self.results_display.append(f"  Häufigkeit: {count}")

                    # Zeige Gesamtanzahl der nicht angezeigten Begriffe
                    shown_count = len(exact_matches) + min(10, len(partial_matches)) + min(10, len(other_matches))
                    if len(unique_headings) > shown_count:
                        self.results_display.append(f"\n... und {len(unique_headings) - shown_count} weitere")

                else:
                    self.results_display.append("\nKeine GND Subject Headings gefunden.")

            except Exception as e:
                self.results_display.append(f"\nFehler bei der Verarbeitung von {term}: {str(e)}")
                import traceback
                self.results_display.append(traceback.format_exc())
                term_results[term] = {'headings': set(), 'counter': Counter(), 'total': 0}

        # Speichere die aktuellen Ergebnisse
        self.current_results = {
            'term_results': term_results,
            'total_counter': total_counter
        }

        self.status_label.setText("Verarbeite Gesamtanalyse...")
        QApplication.processEvents()

        # Zeige Überschneidungen bei mehreren Suchbegriffen
        if len(search_terms) > 1:
            self.show_common_results(term_results, search_terms)

    # Am Ende der perform_search Methode, nach der Verarbeitung aller Suchergebnisse:

           # Am Ende der perform_search Methode:

            # Am Ende der perform_search Methode:

        if total_counter:
            # Erstelle Prompt mit den häufigsten GND-Begriffen
            prompt_text = "=== GND-Begriffe zur Auswahl ===\n\n"

            # Berechne den Threshold (5% des häufigsten Vorkommens)
            most_common_count = total_counter.most_common(1)[0][1]
            threshold_percentage = self.threshold_spinner.value()
            threshold = max(3, int(most_common_count * (threshold_percentage / 100.0)))

            # Sammle die Begriffe
            exact_matches = []
            frequent_matches = []
            search_terms = [term.strip().lower() for term in self.search_input.text().split(',')]
        # Sammle die Begriffe
            exact_matches = []
            frequent_matches = []
            search_terms = [term.strip().lower() for term in self.search_input.text().split(',')]

            for item, count in total_counter.most_common():
                if isinstance(item, tuple) and len(item) >= 1:
                    # Bereinige das Label von zusätzlichen Anführungszeichen
                    label = item[0].strip("'").lower()

                    # Prüfe auf exakte Übereinstimmungen
                    if any(term.lower() == label for term in search_terms):
                        exact_matches.append((item, count))
                        continue

                    # Wenn es kein exakter Match ist, prüfe die Häufigkeit
                    if count >= threshold:
                        frequent_matches.append((item, count))

            if exact_matches or frequent_matches:
                # Zeige exakte Treffer
                if exact_matches:
                    prompt_text += "Exakte Treffer:\n"
                    for item, count in exact_matches:
                        # Bereinige auch hier die Anzeige
                        clean_label = item[0].strip("'")
                        clean_gnd = item[1].strip("'")
                        prompt_text += f"★ {clean_label} (GND: {clean_gnd}) [{count}x]\n"
                    prompt_text += "\n"

                # Zeige häufige Begriffe
                if frequent_matches:
                    prompt_text += f"Häufige Begriffe (mindestens {threshold} Vorkommen):\n"
                    for item, count in frequent_matches:
                        # Bereinige auch hier die Anzeige
                        clean_label = item[0].strip("'")
                        clean_gnd = item[1].strip("'")
                        prompt_text += f"• {clean_label} (GND: {clean_gnd}) [{count}x]\n"



                # Füge Anweisungen hinzu
                prompt_text += f"\n=== Aufgabe ===\n"
                prompt_text += "Bitte wähle aus dieser Liste die relevanten Schlagworte aus und entferne:\n"
                prompt_text += "- zu allgemeine Begriffe\n"
                prompt_text += "- offensichtlich irrelevante Begriffe\n"
                prompt_text += "- redundante Begriffe\n\n"
                prompt_text += "Die ausgewählten Begriffe sollten:\n"
                prompt_text += "- spezifisch für das Thema sein\n"
                prompt_text += "- die wichtigsten Aspekte abdecken\n"
                prompt_text += "- eine ausgewogene Erschließung ermöglichen\n"
                prompt_text += "Die Liste soll mit Kommas getrennt gepromptet werden!\n"

                # Zeige die Liste im Prompt-Fenster an
                self.prompt_display.clear()
                self.prompt_display.setText(prompt_text)

                # Wechsle zum Prompt-Tab
                self.tabs.setCurrentIndex(3)

                # Aktualisiere Status
                total_matches = len(exact_matches) + len(frequent_matches)
                self.status_label.setText(f"Suche abgeschlossen. {len(exact_matches)} exakte und {len(frequent_matches)} häufige GND-Begriffe gefunden.")
            else:
                self.status_label.setText("Keine relevanten GND-Begriffe gefunden.")
        else:
            self.status_label.setText("Keine GND Subject Headings gefunden.")

        # Zeige Gesamtanalyse
        self.show_total_analysis(total_counter)

        # Button wieder aktivieren
        self.search_button.setEnabled(True)
        self.search_button.setText("Suchen")
        self.status_label.setText("Suche abgeschlossen!")
        # Aktualisiere Status mit mehr Details
        total_matches = len(exact_matches) + len(frequent_matches)
        if exact_matches and frequent_matches:
            self.status_label.setText(
                f"Suche abgeschlossen: {len(exact_matches)} exakte Treffer und "
                f"{len(frequent_matches)} häufige Begriffe (>{threshold}x) gefunden."
            )
        elif exact_matches:
            self.status_label.setText(
                f"Suche abgeschlossen: {len(exact_matches)} exakte Treffer gefunden."
            )
        elif frequent_matches:
            self.status_label.setText(
                f"Suche abgeschlossen: {len(frequent_matches)} häufige Begriffe (>{threshold}x) gefunden."
            )


def main():
    app = QApplication(sys.argv)
    window = SearchApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
