import sys
import sqlite3
from PyQt6.QtWidgets import (QApplication, QWidget, QTableView, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QHeaderView)
from PyQt6.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery
from PyQt6.QtCore import Qt

class TableWidget(QWidget):
    """Ein Widget, das Daten aus einer SQLite-Datenbank in einer Tabelle anzeigt."""
    
    def __init__(self, db_path, table_name, parent=None):
        """
        Initialisiert das SQLite-Tabellen-Widget.
        
        Args:
            db_path (str): Der Pfad zur SQLite-Datenbankdatei
            table_name (str): Der Name der anzuzeigenden Tabelle
            parent (QWidget, optional): Das Elternobjekt
        """
        super().__init__(parent)
        
        self.db_path = db_path
        self.table_name = table_name
        
        # Datenbankverbindung einrichten
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName(db_path)
        
        if not self.db.open():
            raise Exception(f"Fehler beim Öffnen der Datenbank: {self.db.lastError().text()}")
        
        # Layout einrichten
        self.setup_ui()
        
        # Tabellendaten laden
        self.load_table_data()
    
    def setup_ui(self):
        """Richtet das User Interface des Widgets ein."""
        # Hauptlayout (vertikal)
        main_layout = QVBoxLayout(self)
        
        # Layout für Überschrift und Info
        header_layout = QHBoxLayout()
        self.title_label = QLabel(f"Tabelle: {self.table_name}")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.info_label = QLabel("")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.info_label)
        main_layout.addLayout(header_layout)
        
        # Tabellen-View
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        main_layout.addWidget(self.table_view)
        
        # Button-Bereich
        button_layout = QHBoxLayout()
        
        # Beispiel-Buttons (können später angepasst werden)
        self.refresh_button = QPushButton("Aktualisieren")
        self.refresh_button.clicked.connect(self.load_table_data)
        
        self.add_button = QPushButton("Hinzufügen")
        self.edit_button = QPushButton("Bearbeiten")
        self.delete_button = QPushButton("Löschen")
        
        # Buttons zum Layout hinzufügen
   #     button_layout.addWidget(self.refresh_button)
   #     button_layout.addWidget(self.add_button)
   #     button_layout.addWidget(self.edit_button)
   #     button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
    def load_table_data(self):
        """Lädt die Daten der SQLite-Tabelle in die TableView."""
        # Modell für die Tabelle erstellen
        self.model = QSqlTableModel(self, self.db)
        self.model.setTable(self.table_name)
        
        # Editierverhalten festlegen (OnManualSubmit = Änderungen werden nicht sofort übernommen)
        self.model.setEditStrategy(QSqlTableModel.EditStrategy.OnManualSubmit)
        
        # Daten abrufen
        self.model.select()
        
        # Anzahl der Datensätze anzeigen
        row_count = self.model.rowCount()
        self.info_label.setText(f"{row_count} Datensätze")
        
        # Modell der TableView zuweisen
        self.table_view.setModel(self.model)

    def closeEvent(self, event):
        """Wird aufgerufen, wenn das Widget geschlossen wird."""
        # Datenbankverbindung schließen
        self.db.close()
        QSqlDatabase.removeDatabase(QSqlDatabase.defaultConnection)
        super().closeEvent(event)
