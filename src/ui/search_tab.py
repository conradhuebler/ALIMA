from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QLineEdit, QPushButton, QGroupBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QApplication, QListWidgetItem
)
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from collections import Counter
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, QEventLoop, pyqtSlot
from typing import Dict, List, Optional, Tuple
import json
import csv
import requests
from datetime import datetime
import logging
import re

class SearchTab(QWidget):
    """Tab für die direkte GND-Schlagwortsuche"""
    
    # Signale
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    search_completed = pyqtSignal(dict)
    keywords_found = pyqtSignal(str)

    def __init__(self, search_engine, cache_manager, parent=None):
        super().__init__(parent)
        self.search_engine = search_engine
        self.cache_manager = cache_manager
        self.current_results = None
        self.logger = logging.getLogger(__name__)
        self.result_list = []
        self.init_ui()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche des Such-Tabs"""
        layout = QVBoxLayout(self)

        # Suchbereich
        search_group = QGroupBox("GND-Suche")
        search_layout = QVBoxLayout()

        # Hauptsuchfeld
        search_input_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Suchbegriffe (durch Komma getrennt)")
        self.search_button = QPushButton("Suchen")
        self.search_button.clicked.connect(self.perform_search)
        search_input_layout.addWidget(self.search_input)
        search_input_layout.addWidget(self.search_button)
        search_layout.addLayout(search_input_layout)

        # Erweiterte Suchoptionen
        options_layout = QHBoxLayout()
        
        # Threshold-Einstellung
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Schwellenwert (%):")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 100.0)
        self.threshold_spin.setValue(1.0)
        self.threshold_spin.setSingleStep(0.1)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spin)
        options_layout.addLayout(threshold_layout)

        self.status_label = QLabel("Aktueller Status: Bereit")
        options_layout.addWidget(self.status_label)
        # Sortieroptionen
        sort_layout = QHBoxLayout()
        sort_label = QLabel("Sortierung:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Häufigkeit", "Alphabetisch", "Relevanz"])
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.sort_combo)
        options_layout.addLayout(sort_layout)

        # Filteroptionen
        self.exact_match_check = QCheckBox("Nur exakte Treffer")
        options_layout.addWidget(self.exact_match_check)
        
        search_layout.addLayout(options_layout)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # Ergebnisbereich
        results_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Ergebnistabelle
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Begriff", "GND-ID", "Häufigkeit", "Quelle"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        results_splitter.addWidget(self.results_table)

        # Detailansicht
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout()
        self.details_display = QTextEdit()
        self.details_display.setReadOnly(True)
        details_layout.addWidget(self.details_display)
        details_group.setLayout(details_layout)
        results_splitter.addWidget(details_group)

        layout.addWidget(results_splitter)

        # Aktionsleiste
        actions_layout = QHBoxLayout()
        
        # Export-Button
        self.export_button = QPushButton("Exportieren")
        self.export_button.clicked.connect(self.export_results)
        actions_layout.addWidget(self.export_button)
        
        # Cache-Prüfung
        self.check_cache_button = QPushButton("Im Cache prüfen")
        self.check_cache_button.clicked.connect(self.check_cache)
        actions_layout.addWidget(self.check_cache_button)
        
        # Ergebnisse löschen
        self.clear_button = QPushButton("Zurücksetzen")
        self.clear_button.clicked.connect(self.clear_results)
        actions_layout.addWidget(self.clear_button)
        
        layout.addLayout(actions_layout)

        # Verbinde Signals
        self.search_input.returnPressed.connect(self.perform_search)
        self.results_table.itemSelectionChanged.connect(self.show_details)
        self.sort_combo.currentTextChanged.connect(self.sort_results)

    def perform_search(self):
        """Führt die Suche aus"""
        try:
            # UI-Updates vor der Suche
            self.search_button.setEnabled(False)
            self.status_updated.emit("Suche wird durchgeführt...")
            self.result_list.clear()
            QApplication.processEvents()

            text = self.search_input.text()
            # Extrahiere Begriffe, die in Anführungszeichen stehen
            quoted_pattern = r'"([^"]+)"'
            quoted_matches = re.findall(quoted_pattern, text)
            
            # Entferne die extrahierten Begriffe aus dem ursprünglichen Text
            text = re.sub(quoted_pattern, '', text)
            
            # Teile den verbleibenden Text nach Kommas auf
            remaining_terms = [term.strip() for term in text.split(',') if term.strip()]
            search_terms = quoted_matches + remaining_terms

            self.logger.info(search_terms)

            if not search_terms:
                self.error_occurred.emit("Bitte geben Sie mindestens einen Suchbegriff ein")
                return

            term_results = {}
            total_counter = Counter()

            # Führe Suche für jeden Term durch
            for term in search_terms:
                subject_headings = []
                
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

                # Cache die Ergebnisse
                self.search_engine.cache.cache_results(term, term_results[term])

            # Verarbeite Gesamtergebnisse
            processed_results = self.search_engine.process_results(
                {'term_results': term_results, 'total_counter': total_counter},
                self.threshold_spin.value()
            )

            # Update UI
            self.current_results = processed_results
            self.display_results(processed_results)
            self.prepare_results(processed_results)
            self.search_completed.emit(processed_results)
            self.status_updated.emit("Suche abgeschlossen")

        except Exception as e:
            self.logger.error(f"Fehler bei der Suche: {e}", exc_info=True)
            self.error_occurred.emit(f"Fehler bei der Suche: {str(e)}")
        finally:
            self.search_button.setEnabled(True)
            QApplication.processEvents()

    def prepare_results(self, results):
        """Bereitet die Ergebnisse für die KI-Abfrage vor"""
        # Hier können zusätzliche Verarbeitungsschritte erfolgen
        initial_prompt = []
        try:
            

            # Exakte Treffer
            if results.get('exact_matches'):
                for item, count in results['exact_matches']:
                    label, gnd_id = item
                    list_item = f"{label} ({gnd_id})"
                    initial_prompt.append(list_item)


            # Häufige Treffer
            if results.get('frequent_matches'):
                for item, count in results['frequent_matches']:
                    label, gnd_id = item
                    list_item = f"{label} ({gnd_id})"
                    initial_prompt.append(list_item)



            QApplication.processEvents()  # UI aktualisieren
            self.keywords_found.emit(", ".join(initial_prompt))
            
        except Exception as e:
            self.logger.error(f"Fehler beim Anzeigen der Ergebnisse: {e}", exc_info=True)
            self.error_occurred.emit(f"Fehler beim Anzeigen der Ergebnisse: {str(e)}")

    def display_results(self, results):
        """Zeigt die Suchergebnisse an"""
        try:
            self.details_display.clear()

            # Exakte Treffer
            if results.get('exact_matches'):
                exact_header = "Exakte Treffer:\n"
                self.details_display.append(exact_header)
                
                for item, count in results['exact_matches']:
                    label, gnd_id = item
                    list_item = f"{label} ({count}x)"
                    self.details_display.append(list_item)

            # Häufige Treffer
            if results.get('frequent_matches'):
                frequent_header = "\nHäufige Treffer:\n"
                self.details_display.append(frequent_header)
                
                for item, count in results['frequent_matches']:
                    label, gnd_id = item
                    list_item = f"{label} ({count}x)"
                    self.details_display.append(list_item)

            # Keine Ergebnisse
            if not results.get('exact_matches') and not results.get('frequent_matches'):
                no_results = "Keine Ergebnisse gefunden"
                self.details_display.append(no_results)

            QApplication.processEvents()  # UI aktualisieren

        except Exception as e:
            self.logger.error(f"Fehler beim Anzeigen der Ergebnisse: {e}", exc_info=True)
            self.error_occurred.emit(f"Fehler beim Anzeigen der Ergebnisse: {str(e)}")

    def sort_results(self, results: Optional[List] = None):
        """Sortiert die Ergebnisse nach gewähltem Kriterium"""
        if results is None:
            # Hole aktuelle Daten aus der Tabelle
            results = []
            for row in range(self.results_table.rowCount()):
                results.append({
                    'label': self.results_table.item(row, 0).text(),
                    'gnd_id': self.results_table.item(row, 1).text(),
                    'count': int(self.results_table.item(row, 2).text()),
                    'type': self.results_table.item(row, 3).text()
                })

        sort_criterion = self.sort_combo.currentText()
        if sort_criterion == "Häufigkeit":
            results.sort(key=lambda x: (-x['count'], x['label']))
        elif sort_criterion == "Alphabetisch":
            results.sort(key=lambda x: x['label'])
        else:  # Relevanz
            results.sort(key=lambda x: (
                0 if x['type'] == 'Exakt' else 1,
                -x['count'],
                x['label']
            ))

        # Aktualisiere Tabelle
        self.results_table.setRowCount(0)
        for result in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            
            self.results_table.setItem(
                row, 0, QTableWidgetItem(result['label'])
            )
            self.results_table.setItem(
                row, 1, QTableWidgetItem(result['gnd_id'])
            )
            self.results_table.setItem(
                row, 2, QTableWidgetItem(str(result['count']))
            )
            self.results_table.setItem(
                row, 3, QTableWidgetItem(result['type'])
            )

    def show_details(self):
        """Zeigt Details für den ausgewählten Eintrag"""
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        label = self.results_table.item(row, 0).text()
        gnd_id = self.results_table.item(row, 1).text()
        count = self.results_table.item(row, 2).text()
        result_type = self.results_table.item(row, 3).text()

        details = f"=== Details für GND-Begriff ===\n\n"
        details += f"Begriff: {label}\n"
        details += f"GND-ID: {gnd_id}\n"
        details += f"Häufigkeit: {count}\n"
        details += f"Typ: {result_type}\n\n"

        # Hole zusätzliche Informationen aus dem Cache
        cache_info = self.cache_manager.get_term_info(gnd_id)
        if cache_info:
            details += "=== Cache-Informationen ===\n\n"
            details += f"Zuletzt verwendet: {cache_info['last_used']}\n"
            details += f"Verwendungshäufigkeit: {cache_info['usage_count']}\n"

        self.details_display.setText(details)

    async def check_cache(self):
        """Prüft die eingegebenen Begriffe im Cache"""
        search_terms = [
            term.strip() 
            for term in self.search_input.text().split(',') 
            if term.strip()
        ]

        if not search_terms:
            self.error_occurred.emit("Bitte geben Sie mindestens einen Begriff ein")
            return

        try:
            self.status_updated.emit("Prüfe Cache...")
            results = []

            for term in search_terms:
                cache_result = self.cache_manager.get_cached_results(term)
                if cache_result:
                    results.append({
                        'term': term,
                        'data': cache_result,
                        'timestamp': cache_result.get('timestamp', 'Unbekannt')
                    })

            if results:
                self.show_cache_results(results)
            else:
                self.status_updated.emit("Keine Cache-Einträge gefunden")

        except Exception as e:
            self.error_occurred.emit(f"Fehler bei der Cache-Prüfung: {str(e)}")

    def show_cache_results(self, results: List[Dict]):
        """Zeigt die Cache-Ergebnisse an"""
        cache_info = "=== Cache-Ergebnisse ===\n\n"
        
        for result in results:
            cache_info += f"Begriff: {result['term']}\n"
            cache_info += f"Letzte Aktualisierung: {result['timestamp']}\n"
            cache_info += f"Gefundene GND-Begriffe: {len(result['data']['headings'])}\n"
            cache_info += "\nTop 5 häufigste Begriffe:\n"
            
            for (label, gnd_id), count in result['data']['counter'].most_common(5):
                cache_info += f"• {label} (GND: {gnd_id}) [{count}x]\n"
            
            cache_info += "\n" + "="*40 + "\n\n"

        self.details_display.setText(cache_info)

    def export_results(self):
        """Exportiert die aktuellen Suchergebnisse"""
        if not self.current_results:
            self.error_occurred.emit("Keine Ergebnisse zum Exportieren vorhanden")
            return

        try:
            filename = f"gnd_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Begriff', 'GND-ID', 'Häufigkeit', 'Typ'])
                
                for row in range(self.results_table.rowCount()):
                    writer.writerow([
                        self.results_table.item(row, 0).text(),
                        self.results_table.item(row, 1).text(),
                        self.results_table.item(row, 2).text(),
                        self.results_table.item(row, 3).text()
                    ])

            self.status_updated.emit(f"Ergebnisse wurden in {filename} exportiert")

        except Exception as e:
            self.error_occurred.emit(f"Fehler beim Export: {str(e)}")

    def clear_results(self):
        """Setzt alle Ergebnisse zurück"""
        self.search_input.clear()
        self.results_table.setRowCount(0)
        self.details_display.clear()
        self.current_results = None

    def load_settings(self, settings: QSettings):
        """Lädt die gespeicherten Einstellungen"""
        self.threshold_spin.setValue(
            settings.value('search/threshold', 1.0, type=float)
        )
        self.sort_combo.setCurrentText(
            settings.value('search/sort_criterion', "Häufigkeit")
        )
        self.exact_match_check.setChecked(
            settings.value('search/exact_match_only', False, type=bool)
        )

    def save_settings(self, settings: QSettings):
        """Speichert die aktuellen Einstellungen"""
        settings.setValue('search/threshold', self.threshold_spin.value())
        settings.setValue('search/sort_criterion', self.sort_combo.currentText())
        settings.setValue('search/exact_match_only', self.exact_match_check.isChecked())

    def update_settings(self):
        """Aktualisiert die UI nach Änderungen in den Einstellungen"""
        # Hier können zusätzliche UI-Updates erfolgen
        pass

    def extract_subject_headings(self, item: Dict) -> List[Tuple[str, str]]:
        """
        Extrahiert GND-Schlagworte aus einem Lobid-Eintrag.
        
        Args:
            item: Dictionary mit Lobid-Daten
            
        Returns:
            Liste von Tupeln (Label, GND-ID)
        """
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
    
    def update_search_field(self, keywords):
        self.search_input.setText(keywords)
