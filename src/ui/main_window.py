from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QMenuBar, QMenu, QStatusBar, QLabel, QPushButton, QDialog,
    QFormLayout, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings, pyqtSlot
from pathlib import Path
from typing import Optional, Dict

from .search_tab import SearchTab
from .abstract_tab import AbstractTab
from .settings_dialog import SettingsDialog
from ..core.search_engine import SearchEngine
from ..core.cache_manager import CacheManager
from ..core.ai_processor import AIProcessor
from ..utils.config import Config
import logging

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('GNDSearch', 'SearchTool')
        self.config = Config.get_instance()
        # Initialisiere Core-Komponenten
        self.cache_manager = CacheManager()
        self.search_engine = SearchEngine(self.cache_manager)
        self.ai_processor = AIProcessor()
        self.logger = logging.getLogger(__name__)

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        self.setWindowTitle('GND Subject Headings Suche')
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
        self.abstract_tab = AbstractTab()
        self.abstract_tab.keywords_extracted.connect(self.update_search_field)
        self.abstract_tab.template_name = "abstract_analysis"

        self.analyse_keywords = AbstractTab()
        #self.analyse_keywords.keywords_extracted.connect(self.update_search_field)
        self.analyse_keywords.template_name = "results_verification"
        self.search_tab.keywords_found.connect(self.analyse_keywords.set_keywords)
        self.abstract_tab.abstract_changed.connect(self.analyse_keywords.set_abstract)
        self.analyse_keywords.need_keywords = True
        
        self.tabs.addTab(self.abstract_tab, "Abstract-Analyse")
        self.tabs.addTab(self.search_tab, "GND-Suche")
        self.tabs.addTab(self.analyse_keywords, "Verifikation")

        # Statusleiste
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel()
        self.status_bar.addWidget(self.status_label)

        # Cache-Info Widget
        self.cache_info = QLabel()
        self.status_bar.addPermanentWidget(self.cache_info)
        self.update_cache_info()

    @pyqtSlot(str)
    def update_search_field(self, keywords):
        self.search_tab.update_search_field(keywords)

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
        
        # Cache leeren
        clear_cache_action = cache_menu.addAction('Cache &leeren')
        clear_cache_action.triggered.connect(self.clear_cache)
        
        # Cache-Statistiken
        cache_stats_action = cache_menu.addAction('Cache-&Statistiken')
        cache_stats_action.triggered.connect(self.show_cache_stats)

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
        self.abstract_tab.save_settings(self.settings)

    def update_cache_info(self):
        """Aktualisiert die Cache-Informationen in der Statusleiste"""
        stats = self.cache_manager.get_stats(days=1)
        if stats:
            hit_rate = stats.get('hit_rate', 0) * 100
            self.cache_info.setText(
                f"Cache: {hit_rate:.1f}% Trefferquote | "
                f"{stats.get('total_searches', 0)} Suchen heute"
            )

    def clear_cache(self):
        """Leert den Cache"""
        removed = self.cache_manager.cleanup_old_entries(max_age_days=0)
        self.status_label.setText(f"{removed} Cache-Einträge gelöscht")
        self.update_cache_info()

    def show_cache_stats(self):
        """Zeigt detaillierte Cache-Statistiken"""
        from .dialogs import CacheStatsDialog
        stats = self.cache_manager.get_stats(days=30)
        dialog = CacheStatsDialog(stats, self)
        dialog.exec()

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
