from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QLineEdit, QPushButton, QGroupBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QApplication, QListWidgetItem, QProgressBar
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
from pathlib import Path
import sys
from ..core.lobid_subjects import SubjectSuggester
from ..core.swbfetcher import SWBSubjectExtractor
from ..core.dnb_utils import get_dnb_classification

class SearchTab(QWidget):
    """Tab für die direkte GND-Schlagwortsuche"""
    
    # Signale
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    search_completed = pyqtSignal(dict)
    keywords_found = pyqtSignal(str)
    keywords_exact = pyqtSignal(str)

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
        self.search_input = QTextEdit()
        self.search_input.setPlaceholderText("Suchbegriffe (durch Komma getrennt)")
        vlayout = QVBoxLayout()
        self.hbz_button = QCheckBox("HBZ")
        self.hbz_button.setChecked(True)
        self.swb_button = QCheckBox("SWB")
        self.swb_button.setChecked(True)
        self.search_button = QPushButton("Suchen")
        self.search_button.clicked.connect(self.lobid_search)
        vlayout.addWidget(self.hbz_button)
        vlayout.addWidget(self.swb_button)
        vlayout.addWidget(self.search_button)

        search_input_layout.addWidget(self.search_input)
        search_input_layout.addLayout(vlayout)
        search_layout.addLayout(search_input_layout)

        # Erweiterte Suchoptionen
        options_layout = QHBoxLayout()
        
        # max. Anzahl der Ergebnisse
        max_results_layout = QHBoxLayout()
        max_results_label = QLabel("Max. Ergebnisse:")
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 1000000)
        self.max_results_spin.setValue(10000)
        self.max_results_spin.setSingleStep(100)
        max_results_layout.addWidget(max_results_label)
        max_results_layout.addWidget(self.max_results_spin)
        options_layout.addLayout(max_results_layout)

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
        #options_layout.addLayout(sort_layout)

        # Filteroptionen
        self.exact_match_check = QCheckBox("Nur exakte Treffer")
        options_layout.addWidget(self.exact_match_check)
        
        #search_layout.addLayout(options_layout)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        self.status_label = QLabel("Aktueller Status: Bereit")
        layout.addWidget(self.status_label)

        self.progressBar = QProgressBar()
        layout.addWidget(self.progressBar)

        # Ergebnisbereich
        results_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Ergebnistabelle
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Begriff", "GND-ID", "Häufigkeit", "Ähnlichkeit"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )


        widget = QWidget()
        vlayout = QVBoxLayout()

        self.details_display = QTextEdit()
        self.details_display.setReadOnly(True)
        self.update_button = QPushButton("Update Entry")
        self.update_button.clicked.connect(self.update_entry)
        vlayout.addWidget(self.update_button)
        vlayout.addWidget(self.details_display)
        widget.setLayout(vlayout)
        table_split = QSplitter(Qt.Orientation.Horizontal)
        table_split.addWidget(self.results_table)
        table_split.addWidget(widget)
        table_split.setSizes([600, 400])
        results_splitter.addWidget(table_split)

        # Detailansicht
        self.prompt_widget = QWidget()
        ddc_box = QHBoxLayout()
        self.ddc1_check = QCheckBox("DDC 1xx")
        self.ddc1_check.setChecked(True)
        self.ddc2_check = QCheckBox("DDC 2xx")
        self.ddc2_check.setChecked(True)
        self.ddc3_check = QCheckBox("DDC 3xx")
        self.ddc3_check.setChecked(True)
        self.ddc4_check = QCheckBox("DDC 4xx")
        self.ddc4_check.setChecked(True)
        self.ddc5_check = QCheckBox("DDC 5xx")
        self.ddc5_check.setChecked(True)
        self.ddc6_check = QCheckBox("DDC 6xx")
        self.ddc6_check.setChecked(True)
        self.ddc7_check = QCheckBox("DDC 7xx")
        self.ddc7_check.setChecked(True)
        self.ddc8_check = QCheckBox("DDC 8xx")
        self.ddc8_check.setChecked(True)
        self.ddc9_check = QCheckBox("DDC 9xx")
        self.ddc9_check.setChecked(True)
        self.ddcX_check = QCheckBox("DDC X")
        self.ddcX_check.setChecked(True)
        self.regenerate = QPushButton("Regenerate")
        self.regenerate.clicked.connect(self.generate_prompt)

        ddc_box.addWidget(self.ddc1_check)
        ddc_box.addWidget(self.ddc2_check)
        ddc_box.addWidget(self.ddc3_check)
        ddc_box.addWidget(self.ddc4_check)
        ddc_box.addWidget(self.ddc5_check)
        ddc_box.addWidget(self.ddc6_check)
        ddc_box.addWidget(self.ddc7_check)
        ddc_box.addWidget(self.ddc8_check)
        ddc_box.addWidget(self.ddc9_check)
        ddc_box.addWidget(self.ddcX_check)
        ddc_box.addWidget(self.regenerate)
        self.prompt_widget.setLayout(ddc_box)
        results_splitter.addWidget(self.prompt_widget)

        #details_group = QGroupBox("Details")
        #details_layout = QVBoxLayout()
        #self.content_display = QTextEdit()
        #self.content_display.setReadOnly(True)
        #details_layout.addWidget(self.content_display)
        #details_group.setLayout(details_layout)
        #results_splitter.addWidget(details_group)

        layout.addWidget(results_splitter)

        # Verbinde Signals
        #self.search_input.returnPressed.connect(self.perform_search)
        self.results_table.itemSelectionChanged.connect(self.show_details)
        self.sort_combo.currentTextChanged.connect(self.sort_results)

    def perform_search(self):
        """Führt die Suche aus"""
        try:
            # UI-Updates vor der Suche
            self.search_button.setEnabled(False)
            self.status_updated.emit("Suche wird durchgeführt...")
            self.status_label.setText("Suche wird durchgeführt...")

            self.result_list.clear()
            QApplication.processEvents()

            text = self.search_input.toPlainText().strip();
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
            self.progressBar.setMaximum(2*len(search_terms))
            self.progressBar.setValue(0)
            for term in search_terms:
                self.progressBar.setValue(self.progressBar.value() + 1)
                if len(term) < 3:
                    self.details_display.append(f"Der Begriff '{term}' ist zu kurz (mindestens 3 Zeichen)")
                    term_results[term] = {'headings': set(), 'counter': Counter(), 'total': 0}
                    continue

                subject_headings = []
                
                url = f"https://lobid.org/resources/search?q={term}&format=jsonl"
                response = requests.get(url, stream=True)

                if response.status_code != 200:
                    self.details_display.append(f"Fehler bei der API-Anfrage für {term}: {response.status_code}")
                    term_results[term] = {'headings': set(), 'counter': Counter(), 'total': 0}
                    continue

                self.status_label.setText(f"Analysiere Ergebnisse für: {term}")
                QApplication.processEvents()

                # Verarbeitung der JSON Lines
                total_items = 0
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if total_items >= self.max_results_spin.value():
                            break
                        
                        try:
                            item = json.loads(line)
                            headings = self.extract_subject_headings(item)
                            subject_headings.extend(headings)
                            total_items += 1
                            if total_items % 1000 == 0:
                                self.status_label.setText(f"Verarbeite Eintrag {total_items} für: {term}")
                                QApplication.processEvents()
                        except json.JSONDecodeError as e:
                            self.results_display.append(f"(Suchsterm: {term}) Fehler beim Parsen einer Zeile: {str(e)}")
                            self.logger.info(line)
                            self.logger.info(f"(Suchsterm: {term}) JSON Decode Error: {e}")
                            continue
                        except requests.exceptions.ChunkedEncodingError as e:
                            self.logger.info(line)
                            self.logger.info(f"(Suchsterm: {term}) ChunkedEncodingError: {e}")
                            continue
                except Exception as e:
                    self.logger.error(f"Fehler beim Verarbeiten der Ergebnisse: {e}", exc_info=True)
                    self.error_occurred.emit(f"Fehler beim Verarbeiten der Ergebnisse: {str(e)}")
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
                #self.search_engine.cache.cache_results(term, term_results[term])
                self.progressBar.setValue(self.progressBar.value() + 1)

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

    def current_lobid_term(self, term):
        self.progressBar.setValue(self.progressBar.value() + 1)

    def lobid_search(self):
        """Suche in Lobid nach Begriffen"""
        suggestor = SubjectSuggester()
        cache_dir = Path(sys.argv[0]).parent.resolve() / "swb_cache"

        swbsuggestor = SWBSubjectExtractor(cache_dir)
         # UI-Updates vor der Suche
        self.search_button.setEnabled(False)
        self.status_updated.emit("Suche wird durchgeführt...")
        self.result_list.clear()
        QApplication.processEvents()

        text = self.search_input.toPlainText().strip();
        # Extrahiere Begriffe, die in Anführungszeichen stehen
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, text)
            
        # Entferne die extrahierten Begriffe aus dem ursprünglichen Text
        text = re.sub(quoted_pattern, '', text)
            
            # Teile den verbleibenden Text nach Kommas auf
        remaining_terms = [term.strip() for term in text.split(',') if term.strip()]
        search_terms = quoted_matches + remaining_terms

    
        def process_entries(entries, parent_term):
            for term, info in entries.items():
                if isinstance(info, dict):
                    if 'count' in info and 'gndid' in info:
                        gnd_id = list(info['gndid'])[0]
                        count = info['count']
                        
                        # Bestimme die Beziehung
                        relation = 2 # different
                        if term == parent_term:
                            relation = 0 # exakt
                        elif parent_term.lower() in term.lower() or term.lower() in parent_term.lower():
                            relation = 1 #similar
                        
                        # Aktualisiere oder füge neuen Eintrag hinzu
                        if gnd_id in gnd_analysis:
                            current_count = gnd_analysis[gnd_id][1]
                            current_relation = gnd_analysis[gnd_id][2]
                            # Behalte die "stärkste" Beziehung
                            if current_relation != 0:
                                if relation == 0 or (relation == 1 and current_relation == 2):
                                    gnd_analysis[gnd_id] = (term, current_count + count, relation)
                                else:
                                    gnd_analysis[gnd_id] = (term, current_count + count, current_relation)
                            else:
                                gnd_analysis[gnd_id] = (term, current_count + count, current_relation)
                        else:
                            gnd_analysis[gnd_id] = (term, count, relation)
                    else:
                        process_entries(info, parent_term)
                        
        gnd_analysis = {}
        self.progressBar.setMaximum(len(search_terms))
        self.logger.info(f"Suche nach Begriffen: {search_terms}")
        suggestor.currentTerm.connect(self.current_lobid_term)
        swbsuggestor.currentTerm.connect(self.current_lobid_term)
        results_hbz = {}
        results_swb = {}
        if self.hbz_button.isChecked():
            results_hbz = suggestor.search(search_terms)
        if self.swb_button.isChecked():
            results_swb = swbsuggestor.search(search_terms)

        # Verarbeite alle Einträge
        for main_term, entries in results_swb.items():
            self.logger.info(f"Verarbeite Einträge für: {entries}")
            process_entries(entries, main_term)
        
        for main_term, entries in results_hbz.items():
            self.logger.info(f"Verarbeite Einträge für: {entries}")
            process_entries(entries, main_term)
        
        # Sortiere die Ergebnisse
        sorted_results = sorted(gnd_analysis.items(), key=lambda x: x[1][1], reverse=True)
        
        initial_prompt = []
        self.gnd_ids = []
        self.status_label.setText("Datenbank wird gefüttert ...")

        self.progressBar.setMaximum(len(sorted_results))
        for gnd_id, (term, count, relation) in sorted_results:
            self.logger.info(gnd_id)
            self.gnd_ids.append(gnd_id)
            ddc = ""
            gdn_category = ""
            category = ""
            checked = False

            if not self.cache_manager.gnd_entry_exists(gnd_id):
                if term == gnd_id:
                    self.cache_manager.insert_gnd_entry(gnd_id, title = "Empty")
                else:
                    self.cache_manager.insert_gnd_entry(gnd_id, title = term)
            tmp_entry = self.cache_manager.get_gnd_entry(gnd_id) # check time of update
            if not tmp_entry == None:
                self.logger.info(tmp_entry)

                if(tmp_entry['created_at'] == tmp_entry['updated_at']):
                    if relation == 0 or relation == 1:
                        self.update_entry(gnd_id)
                
                if not term == gnd_id:
                    list_item = f"{term} ({gnd_id})"
                    initial_prompt.append(list_item)
            self.progressBar.setValue(self.progressBar.value() + 1)

        self.keywords_found.emit(", ".join(initial_prompt))

        self.display_results_patrick(sorted_results)
        self.search_button.setEnabled(True)
        QApplication.processEvents()

    def generate_prompt(self):
        initial_prompt = []

        for gnd_id in self.gnd_ids:
            gnd_entry = self.cache_manager.get_gnd_entry(gnd_id)
            ddcs = gnd_entry['ddcs']
            include = False
            #self.logger.info(gnd_entry)
            for ddc in ddcs.split(";"):
                if include:
                    continue
                
                #self.logger.info(gnd_id)
                #self.logger.info(ddc)
                #self.logger.info(ddc.startswith("5"))
                include = include or ((self.ddc1_check.isChecked() and ddc.startswith("1")) 
                    or (self.ddc2_check.isChecked() and ddc.startswith("2")) 
                    or (self.ddc3_check.isChecked() and ddc.startswith("3")) 
                    or (self.ddc4_check.isChecked() and ddc.startswith("4")) 
                    or (self.ddc5_check.isChecked() and ddc.startswith("5")) 
                    or (self.ddc6_check.isChecked() and ddc.startswith("6")) 
                    or (self.ddc7_check.isChecked() and ddc.startswith("7")) 
                    or (self.ddc8_check.isChecked() and ddc.startswith("8")) 
                    or (self.ddc9_check.isChecked() and ddc.startswith("9")))
                if self.ddcX_check.isChecked():
                    include = True  
            #self.logger.info(include)
            if include:
                list_item = f"{gnd_entry['title']} ({gnd_id})"
                initial_prompt.append(list_item)

        self.keywords_found.emit(", ".join(initial_prompt))


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
                    if(self.cache_manager.gnd_entry_exists(gnd_id) == False):
                        self.cache_manager.insert_gnd_entry(gnd_id, title = label)
            self.keywords_exact.emit(", ".join(initial_prompt))
            
            # Häufige Treffer
            if results.get('frequent_matches'):
                for item, count in results['frequent_matches']:
                    label, gnd_id = item
                    list_item = f"{label} ({gnd_id})"
                    initial_prompt.append(list_item)   

                    if(self.cache_manager.gnd_entry_exists(gnd_id) == False):
                        self.cache_manager.insert_gnd_entry(gnd_id, title = label)

            QApplication.processEvents()  # UI aktualisieren
            self.keywords_found.emit(", ".join(initial_prompt))
            
        except Exception as e:
            self.logger.error(f"Fehler beim Anzeigen der Ergebnisse: {e}", exc_info=True)
            self.error_occurred.emit(f"Fehler beim Anzeigen der Ergebnisse: {str(e)}")

    def display_results(self, results):
        """Zeigt die Suchergebnisse an"""
        try:
            self.content_display.clear()
            #self.logger.info(results)
            # Exakte Treffer
            if results.get('exact_matches'):
                exact_header = "Exakte Treffer:\n"
                self.content_display.append(exact_header)
                
                for item, count in results['exact_matches']:
                    label, gnd_id = item
                    list_item = f"{label} ({count}x)"
                    self.content_display.append(list_item)
                    self.results_table.insertRow(self.results_table.rowCount())
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 0, QTableWidgetItem(label)
                    )
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 1, QTableWidgetItem(gnd_id)
                    )
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 2, QTableWidgetItem(str(count))
                    )
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 3, QTableWidgetItem("Exakt")
                    )

            # Häufige Treffer
            if results.get('frequent_matches'):
                frequent_header = "\nHäufige Treffer:\n"
                self.content_display.append(frequent_header)
                
                for item, count in results['frequent_matches']:
                    label, gnd_id = item
                    list_item = f"{label} ({count}x)"
                    self.content_display.append(list_item)
                    self.result_list.append(list_item)
                    self.results_table.insertRow(self.results_table.rowCount())
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 0, QTableWidgetItem(label)
                    )
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 1, QTableWidgetItem(gnd_id)
                    )
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 2, QTableWidgetItem(str(count))
                    )
                    self.results_table.setItem(
                        self.results_table.rowCount() - 1, 3, QTableWidgetItem("Ähnlich")
                    )

            # Keine Ergebnisse
            if not results.get('exact_matches') and not results.get('frequent_matches'):
                no_results = "Keine Ergebnisse gefunden"
                #self.content_display.append(no_results)

            QApplication.processEvents()  # UI aktualisieren

        except Exception as e:
            self.logger.error(f"Fehler beim Anzeigen der Ergebnisse: {e}", exc_info=True)
            self.error_occurred.emit(f"Fehler beim Anzeigen der Ergebnisse: {str(e)}")

    def display_results_patrick(self, sorted_results):
        """Zeigt die Suchergebnisse an"""
        try:
            #self.content_display.clear()
            self.results_table.setRowCount(0)  # Tabelle leeren

            # sorted_results ist bereits eine sortierte Liste von Tupeln:
            # [(gnd_id, (term, count, relation)), ...]
            
            # Gruppiere nach Beziehungstyp
            exact_matches = []
            similar_matches = []
            different_matches = []
            
            for gnd_id, (term, count, relation) in sorted_results:
                if relation == 0:
                    exact_matches.append((term, gnd_id, count))
                elif relation == 1:
                    similar_matches.append((term, gnd_id, count))
                else:
                    different_matches.append((term, gnd_id, count))

            # Exakte Treffer anzeigen
            if exact_matches:
                #self.content_display.append("Exakte Treffer:")
                for term, gnd_id, count in exact_matches:
                    list_item = f"{term} ({count}x) = [{gnd_id}]"
                    #self.content_display.append(list_item)
                    
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)
                    self.results_table.setItem(row, 0, QTableWidgetItem(term))
                    self.results_table.setItem(row, 1, QTableWidgetItem(gnd_id))
                    self.results_table.setItem(row, 2, QTableWidgetItem(str(count)))
                    self.results_table.setItem(row, 3, QTableWidgetItem("="))

            # Ähnliche Treffer anzeigen
            if similar_matches:
                #self.content_display.append("\nÄhnliche Treffer:")
                for term, gnd_id, count in similar_matches:
                    list_item = f"{term} ({count}x) ≈ [{gnd_id}]"
                    #self.content_display.append(list_item)
                    
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)
                    self.results_table.setItem(row, 0, QTableWidgetItem(term))
                    self.results_table.setItem(row, 1, QTableWidgetItem(gnd_id))
                    self.results_table.setItem(row, 2, QTableWidgetItem(str(count)))
                    self.results_table.setItem(row, 3, QTableWidgetItem("≈"))

            # Verschiedene Treffer anzeigen
            if different_matches:
                #self.content_display.append("\nWeitere Treffer:")
                for term, gnd_id, count in different_matches:
                    list_item = f"{term} ({count}x) ≠ [{gnd_id}]"
                    #self.content_display.append(list_item)
                    
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)
                    self.results_table.setItem(row, 0, QTableWidgetItem(term))
                    self.results_table.setItem(row, 1, QTableWidgetItem(gnd_id))
                    self.results_table.setItem(row, 2, QTableWidgetItem(str(count)))
                    self.results_table.setItem(row, 3, QTableWidgetItem("≠"))

            # Keine Ergebnisse
            if not exact_matches and not similar_matches and not different_matches:
                no_results = "Keine Ergebnisse gefunden"
                #self.content_display.append(no_results)

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

    def update_entry(self, gnd_id: str = None):
        if gnd_id is None:
            gnd_id = self.current_gnd_id
        dnb_class = get_dnb_classification(gnd_id)
        if dnb_class and dnb_class['status'] == 'success':
            term = dnb_class['preferred_name']
                # weiterer Code
        else:
            error_msg = dnb_class['error_message'] if dnb_class else "Keine Daten erhalten"
            self.logger.error(f"Fehler bei GND {gnd_id}: {error_msg}")
            # Fehlerbehandlung

        self.logger.info(dnb_class)
        term = dnb_class['preferred_name']
        ddc_list = dnb_class['ddc']
        ddc = ";".join(f"{d['code']}({d['determinancy']})" for d in ddc_list)
        self.logger.info(ddc)  
        gdn_category = dnb_class['gnd_subject_categories']
        gdn_category = ";".join(gdn_category)
        category = dnb_class['category']
        self.cache_manager.update_gnd_entry(gnd_id, title = term, ddcs = ddc, gnd_systems = gdn_category, classification = category)
            
    def show_details(self):
        """Zeigt Details für den ausgewählten Eintrag"""
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            return
        row = selected_items[0].row()
        gnd_id = self.results_table.item(row, 1).text()     
        if self.cache_manager.gnd_entry_exists(gnd_id): 
            gnd_entry = self.cache_manager.get_gnd_entry(gnd_id)
            if gnd_entry['updated_at'] == gnd_entry['created_at']:
                self.update_entry(gnd_id)
                gnd_entry = self.cache_manager.get_gnd_entry(gnd_id)
            details = f"Details für GND-ID: {gnd_id}\n\n"
            details += f"Titel: {gnd_entry['title']}\n"
            details += f"Beschreibung: {gnd_entry['description']}\n"
            details += f"DDC: {gnd_entry['ddcs']}\n"
            details += f"DK: {gnd_entry['dks']}\n"
            details += f"Synonyme: {gnd_entry['synonyms']}\n"
            details += f"Erstellt am: {gnd_entry['created_at']}\n"
            details += f"Zuletzt aktualisiert am: {gnd_entry['updated_at']}\n"

        # Hole zusätzliche Informationen aus dem Cache
        self.current_gnd_id = gnd_id
        self.details_display.setText(details)

    async def check_cache(self):
        """Prüft die eingegebenen Begriffe im Cache"""
        search_terms = [
            term.strip() 
            for term in self.search_input.toPlainText().split(',') 
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
