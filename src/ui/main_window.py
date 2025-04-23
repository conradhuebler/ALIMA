from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QMenuBar, QMenu, QStatusBar, QLabel, QPushButton, QDialog,
    QFormLayout, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings, pyqtSlot
from pathlib import Path
import re
from typing import Optional, Dict

from .find_keywords import SearchTab
from .abstract_tab import AbstractTab
from .settings_dialog import SettingsDialog
from ..core.search_engine import SearchEngine
from ..core.cache_manager import CacheManager
#from ..core.ai_processor import AIProcessor
from ..utils.config import Config
from .crossref_tab import CrossrefTab
from .ubsearch_tab import UBSearchTab
from .tablewidget import TableWidget
import logging

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('GNDSearch', 'SearchTool')
        self.config = Config.get_instance()
        # Initialisiere Core-Komponenten
        self.cache_manager = CacheManager()
        self.search_engine = SearchEngine(self.cache_manager)
        #self.ai_processor = AIProcessor()
        self.logger = logging.getLogger(__name__)

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        self.setWindowTitle('AlIma')
        self.setGeometry(100, 100, 1200, 800)

        # Zentrales Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Menüleiste
        self.create_menu_bar()

        # Tab-Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tabs erstellen
        self.search_tab = SearchTab(
            search_engine=self.search_engine,
            cache_manager=self.cache_manager
        )

        self.crossref_tab = CrossrefTab()

        self.abstract_tab = AbstractTab()
        self.crossref_tab.result_abstract.connect(self.abstract_tab.set_abstract)
        #self.abstract_tab.final_list.connect(self.update_search_field)
        self.abstract_tab.template_name = "abstract_analysis"
        self.abstract_tab.set_model_recommendations("abstract")
        self.abstract_tab.set_task("abstract")

        self.analyse_keywords = AbstractTab()
        #self.analyse_keywords.keywords_extracted.connect(self.update_search_field)
        self.analyse_keywords.template_name = "results_verification"
        self.search_tab.keywords_found.connect(self.analyse_keywords.set_keywords)
        self.abstract_tab.abstract_changed.connect(self.analyse_keywords.set_abstract)
        self.analyse_keywords.need_keywords = True
        self.analyse_keywords.final_list.connect(self.update_gnd_keywords)
        self.analyse_keywords.set_task("keywords")
        self.ub_search_tab = UBSearchTab()
                
        self.abstract_tab.final_list.connect(self.search_tab.update_search_field)
        self.analyse_keywords.final_list.connect(self.ub_search_tab.update_keywords)

        self.crossref_tab.result_abstract.connect(self.ub_search_tab.set_abstract)
        self.abstract_tab.abstract_changed.connect(self.ub_search_tab.set_abstract)
        #self.abstract_tab.keywords_extracted.connect(self.ub_search_tab.update_keywords)

        #self.table_widget = TableWidget(
        #    db_path=self.cache_manager.db_path,
        #    table_name="gnd_entry"
        #)

        
        self.tabs.addTab(self.crossref_tab, "Crossref DOI Lookup")
        self.tabs.addTab(self.abstract_tab, "Abstract-Analyse")
        self.tabs.addTab(self.search_tab, "GND-Suche")
        self.tabs.addTab(self.analyse_keywords, "Verifikation")
        self.tabs.addTab(self.ub_search_tab, "UB Suche")
        #self.tabs.addTab(self.table_widget, "GND Einträge")

        # Statusleiste
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel()
        self.status_bar.addWidget(self.status_label)

        # Cache-Info Widget
        self.cache_info = QLabel()
        self.status_bar.addPermanentWidget(self.cache_info)

    @pyqtSlot(str)
    def update_gnd_keywords(self, keywords):
        self.logger.info(keywords)
        """
        Extrahiert GND-Schlagworte aus einem Text.
        
        Args:
            text (str): Der Text, der die GND-Einträge enthält
            
        Returns:
            list: Liste der GND-Schlagworte ohne IDs
        """
        self.logger.info(keywords)
        try:
            # Suche den Abschnitt mit den GND-Einträgen
            if "Schlagworte OGND Eintrage:" not in keywords:
                return []
                
            # Extrahiere den relevanten Teil des Textes
            gnd_section = keywords.split("Schlagworte OGND Eintrage:")[1].split("FEHLENDE KONZEPTE:")[0]
            
            # Extrahiere die Schlagworte (alles vor der URL)
            gnd_terms = []
            for line in gnd_section.split(","):
                if "(https://" in line:
                    term = line.split("(https://")[0].strip().replace("\"","")
                    
                    self.logger.info(term)
                    if term:
                        gnd_terms.append(term)
            self.logger.info(gnd_terms)
            self.ub_search_tab.update_keywords(keywords)       
            return gnd_terms
            
        except Exception as e:
            print(f"Fehler beim Extrahieren der GND-Terme: {str(e)}")
            return []
        
    @pyqtSlot(str)
    def update_search_field(self, keywords):
        """
        Extrahiert Keywords aus einem String, behandelt geklammerte Terme und schützt Slashes.
        
        Args:
            keyword_string (str): String mit kommagetrennten Keywords in Anführungszeichen
            
        Returns:
            list: Liste von extrahierten Keywords
            list: Liste von extrahierten Klammer-Termen
        """
        
        # Entferne Leerzeichen am Anfang und Ende
        keyword_string = keywords.strip()
        
        # Wenn der String mit [ beginnt und mit ] endet, entferne diese
        if keyword_string.startswith('[') and keyword_string.endswith(']'):
            keyword_string = keyword_string[1:-1]
        
        # Regulärer Ausdruck für das Matching
        # Matches entweder:
        # 1. Terme in Anführungszeichen die Slash enthalten
        # 2. Terme in Anführungszeichen mit Klammern
        # 3. Normale Terme in Anführungszeichen
        pattern = r'"([^"]*?/[^"]*?)"|"([^"]*?\([^)]+?\)[^"]*?)"|"([^"]*?)"'
        
        matches = re.finditer(pattern, keyword_string)
        
        keywords = []
        bracketed_terms = []
        
        for match in matches:
            # Wenn es ein Slash-Term ist
            if match.group(1):
                term = match.group(1).replace('/', '\\/')
                keywords.append(f'"{term}"')
            # Wenn es ein Term mit Klammern ist
            elif match.group(2):
                term = match.group(2)
                # Extrahiere den Klammerinhalt
                bracket_content = re.findall(r'\(([^)]+)\)', term)
                # Füge den Hauptterm zu den Keywords hinzu
                main_term = re.sub(r'\s*\([^)]+\)', '', term).strip()
                if main_term:
                    keywords.append(main_term)
                # Füge die Klammerterme zur separaten Liste hinzu
                bracketed_terms.extend([f'"{term}"' for term in bracket_content])
            # Wenn es ein normaler Term ist
            elif match.group(3):
                keywords.append(f'"{match.group(3)}"')
        
        result = keywords + bracketed_terms
        self.search_tab.update_search_field(", ".join(result))

    def create_menu_bar(self):
        """Erstellt die Menüleiste"""
        menubar = self.menuBar()

        # Datei-Menü
        file_menu = menubar.addMenu('&Datei')
        
        # Export-Aktion
        export_action = file_menu.addAction('&Exportieren...')
        export_action.triggered.connect(self.export_results)
        
        # Import-Aktion
        import_action = file_menu.addAction('&Importieren...')
        import_action.triggered.connect(self.import_results)
        
        file_menu.addSeparator()
        
        # Beenden-Aktion
        exit_action = file_menu.addAction('&Beenden')
        exit_action.triggered.connect(self.close)

        # Bearbeiten-Menü
        edit_menu = menubar.addMenu('&Bearbeiten')
        
        # Einstellungen-Aktion
        settings_action = edit_menu.addAction('&Einstellungen')
        settings_action.triggered.connect(self.show_settings)

        # Cache-Menü
        cache_menu = menubar.addMenu('&Cache')
        
        # Hilfe-Menü
        help_menu = menubar.addMenu('&Hilfe')
        
        # Über-Dialog
        about_action = help_menu.addAction('Ü&ber')
        about_action.triggered.connect(self.show_about)
        
        # Hilfe-Dialog
        help_action = help_menu.addAction('&Hilfe')
        help_action.triggered.connect(self.show_help)

    def show_settings(self):
        """Öffnet den Einstellungsdialog"""
        try:
            # Übergebe beide Konfigurationen getrennt
            dialog = SettingsDialog(ui_settings=self.settings, parent=self)
            if dialog.exec():
                # Einstellungen wurden gespeichert
                self.load_settings()
        except Exception as e:
            self.logger.error(f"Fehler beim Öffnen der Einstellungen: {e}")
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Öffnen der Einstellungen: {str(e)}"
            )

    def load_settings(self):
        """Lädt die gespeicherten Einstellungen"""
        # Fenster-Geometrie
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)

        # Letzter aktiver Tab
        last_tab = self.settings.value('last_tab', 0, type=int)
        self.tabs.setCurrentIndex(last_tab)

        # Update der Tab-Einstellungen
        self.search_tab.load_settings(self.settings)
        #self.abstract_tab.load_settings(self.settings)

    def save_settings(self):
        """Speichert die aktuellen Einstellungen"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('last_tab', self.tabs.currentIndex())
        
        # Speichere Tab-Einstellungen
        self.search_tab.save_settings(self.settings)

    def export_results(self):
        """Exportiert die aktuellen Suchergebnisse"""
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, 'export_results'):
            current_tab.export_results()
        else:
            self.status_label.setText("Export nicht verfügbar für diesen Tab")

    def import_results(self):
        """Importiert gespeicherte Suchergebnisse"""
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, 'import_results'):
            current_tab.import_results()
        else:
            self.status_label.setText("Import nicht verfügbar für diesen Tab")

    def show_about(self):
        """Zeigt den Über-Dialog"""
        from .dialogs import AboutDialog
        dialog = AboutDialog(self)
        dialog.exec()

    def show_help(self):
        """Zeigt den Hilfe-Dialog"""
        from .dialogs import HelpDialog
        dialog = HelpDialog(self)
        dialog.exec()

    def closeEvent(self, event):
        """Wird beim Schließen des Fensters aufgerufen"""
        self.save_settings()
        event.accept()

    def update_status(self, message: str):
        """Aktualisiert die Statusleiste"""
        self.status_label.setText(message)

    def show_error(self, message: str):
        """Zeigt eine Fehlermeldung"""
        from .dialogs import ErrorDialog
        dialog = ErrorDialog(message, self)
        dialog.exec()
