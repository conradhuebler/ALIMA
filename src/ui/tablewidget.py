import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QDialog,
    QTableView,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QHeaderView,
    QComboBox,
    QLineEdit,
    QCheckBox,
    QMessageBox,

    QDialogButtonBox,
)
from PyQt6.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSortFilterProxyModel, QRegularExpression, QTimer as DebounceTimer, QCoreApplication

from ..core.database_manager import DatabaseManager
from ..utils.config_models import DatabaseConfig


class TableWidget(QWidget):
    """Modern database table viewer using DatabaseManager - Claude Generated"""

    def __init__(self, database_config: DatabaseConfig = None, table_name: str = "gnd_entries", parent=None):
        """
        Initialize modern database table widget with DatabaseManager integration.

        Args:
            database_config: Database configuration (uses default if None)
            table_name: Initial table to display
            parent: Parent widget
        """
        super().__init__(parent)

        # Use centralized config if none provided - Claude Generated (UNIFIED DB CONFIG)
        if database_config is None:
            try:
                from ..utils.config_manager import ConfigManager
                config_manager = ConfigManager()
                config = config_manager.load_config()
                database_config = config.database_config
            except Exception as e:
                # Fallback to default with OS-specific path only if config loading fails
                import logging
                logging.warning(f"⚠️ Could not load config: {e}. Using default database config.")
                database_config = DatabaseConfig(db_type='sqlite')

        self.database_config = database_config
        self.current_table = table_name

        # Available tables in the unified knowledge database
        self.available_tables = {
            "gnd_entries": "GND-Einträge (Facts)",
            "classifications": "Klassifikationen (DK/RVK)",
            "search_mappings": "Such-Mappings (Cache)",
            "catalog_dk_cache": "Katalog DK/RVK-Zwischenspeicher"
        }

        # Initialize DatabaseManager
        self.db_manager = DatabaseManager(database_config, f"tablewidget_{id(self)}")


        # Get the Qt SQL connection for QSqlTableModel
        self.db_connection = self.db_manager.get_connection()

        # Initialize filter proxy model for proper filtering
        self.filter_proxy = QSortFilterProxyModel(self)
        self.filter_proxy.setFilterKeyColumn(-1)  # Filter all columns
        self.filter_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        # Layout einrichten
        self.setup_ui()
        # Tabellendaten laden
        self.load_table_data()

    def setup_ui(self):
        """Setup modern UI with table selection - Claude Generated"""
        # Hauptlayout (vertikal)
        main_layout = QVBoxLayout(self)

        # Header with table selection
        header_layout = QHBoxLayout()

        # Table selection dropdown
        self.table_selector = QComboBox()
        for table_key, table_label in self.available_tables.items():
            self.table_selector.addItem(table_label, table_key)

        # Set current table as selected
        current_index = list(self.available_tables.keys()).index(self.current_table)
        self.table_selector.setCurrentIndex(current_index)
        self.table_selector.currentTextChanged.connect(self.on_table_changed)

        # Title and info labels
        self.title_label = QLabel(f"📊 Datenbank: {self.database_config.db_type.upper()}")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.info_label = QLabel("")

        header_layout.addWidget(QLabel("Tabelle:"))
        header_layout.addWidget(self.table_selector)
        header_layout.addStretch()
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.info_label)
        
        main_layout.addLayout(header_layout)

        # Search layout
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Durchsuchen Sie alle Spalten...")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        
        self.clear_search_button = QPushButton("✖")
        self.clear_search_button.setMaximumWidth(30)
        self.clear_search_button.setToolTip("Suche löschen")
        self.clear_search_button.clicked.connect(self.clear_search)
        
        search_layout.addWidget(QLabel("Suchen:"))
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.clear_search_button)
        main_layout.addLayout(search_layout)


        # Tabellen-View
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        main_layout.addWidget(self.table_view)

        # Button-Bereich
        button_layout = QHBoxLayout()

        # Beispiel-Buttons (können später angepasst werden)
        self.refresh_button = QPushButton("Aktualisieren")
        self.refresh_button.clicked.connect(self.load_table_data)

        self.add_button = QPushButton("Hinzufügen")
        self.edit_button = QPushButton("Bearbeiten")
        self.delete_button = QPushButton("Löschen")

        # Buttons zum Layout hinzufügen - Claude Generated
        button_layout.addWidget(self.refresh_button)
        #     button_layout.addWidget(self.add_button)
        #     button_layout.addWidget(self.edit_button)
        #     button_layout.addWidget(self.delete_button)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

    def on_table_changed(self):
        """Handle table selection change - Claude Generated"""
        # Get selected table key from combo box
        selected_index = self.table_selector.currentIndex()
        self.current_table = self.table_selector.itemData(selected_index)
        self.load_table_data()

    def load_table_data(self):
        """Load database table data using DatabaseManager - Claude Generated"""
        try:

            # Create model with our database connection
            self.model = QSqlTableModel(self, self.db_connection)
            self.model.setTable(self.current_table)


            # Load data
            if not self.model.select():
                # Table might not exist yet, show empty model
                self.info_label.setText("Tabelle ist leer oder existiert nicht")
                return

            # Set user-friendly column headers
            self.set_column_headers()

            # Display row count
            row_count = self.model.rowCount()
            self.info_label.setText(f"{row_count:,} Datensätze")

            # Set up filtering and assign model to view via proxy
            self.filter_proxy.setSourceModel(self.model)
            self.table_view.setModel(self.filter_proxy)

            # Auto-resize columns to content
            self.table_view.resizeColumnsToContents()

        except Exception as e:
            self.info_label.setText(f"Fehler beim Laden: {str(e)}")

    def set_column_headers(self):
        """Set user-friendly column headers - Claude Generated"""
        if self.current_table == "gnd_entries":
            headers = {
                "gnd_id": "GND-ID",
                "title": "Titel",
                "description": "Beschreibung",
                "synonyms": "Synonyme",
                "ddcs": "DDC-Codes",
                "ppn": "PPN",
                "created_at": "Erstellt",
                "updated_at": "Aktualisiert"
            }
        elif self.current_table == "classifications":
            headers = {
                "code": "Code",
                "type": "Typ",
                "title": "Titel",
                "description": "Beschreibung",
                "parent_code": "Übergeordnet",
                "created_at": "Erstellt"
            }
        elif self.current_table == "search_mappings":
            headers = {
                "search_term": "Suchbegriff",
                "normalized_term": "Normalisiert",
                "suggester_type": "Suggester",
                "found_gnd_ids": "Gefundene GND-IDs",
                "found_classifications": "Klassifikationen",
                "result_count": "Anzahl Ergebnisse",
                "last_updated": "Zuletzt aktualisiert",
                "created_at": "Erstellt"
            }
        else:
            return  # No custom headers

        # Apply headers
        for i in range(self.model.columnCount()):
            field_name = self.model.record().fieldName(i)
            if field_name in headers:
                self.model.setHeaderData(i, Qt.Orientation.Horizontal, headers[field_name])

    def ensure_all_data_loaded(self):
        """Ensure all data is loaded using processEvents for complete search functionality - Claude Generated"""
        # Process all events to ensure all data is loaded
        QCoreApplication.processEvents()
        
        # Additionally try fetching more data if needed
        try:
            while self.model.canFetchMore():
                self.model.fetchMore()
                QCoreApplication.processEvents()  # Process events after each fetch
        except Exception:
            pass  # Some models don't support fetchMore
    
    def check_all_data_loaded(self, parent, first, last):
        """Check if more data needs to be loaded - Claude Generated"""
        if self.model.canFetchMore():
            self.model.fetchMore()
    def on_search_text_changed(self, text):
        """Handle search text changes - Use processEvents to ensure all data is loaded"""
        if text.strip():
            # Simple text search - no complex regex needed for umlauts
            self.filter_proxy.setFilterRegularExpression(QRegularExpression(text.strip()))
        else:
            # Clear filter
            self.filter_proxy.setFilterRegularExpression(QRegularExpression())
        
        # Ensure all data is loaded before updating count
        QCoreApplication.processEvents()
    
    def clear_search(self):
        """Clear search input and filter - Claude Generated"""
        self.search_input.clear()
        
    def update_filtered_count(self):
        """Update info label with filtered results count - Claude Generated"""
        # Ensure all data is loaded before counting
        QCoreApplication.processEvents()
        
        total_rows = self.model.rowCount()
        filtered_rows = self.filter_proxy.rowCount()
        
        if filtered_rows != total_rows:
            self.info_label.setText(f"{filtered_rows:,} von {total_rows:,} Datensätzen")
        else:
            self.info_label.setText(f"{total_rows:,} Datensätze")

    def closeEvent(self, event):
        """Clean up database connection - Claude Generated"""
        # Close DatabaseManager connection
        if hasattr(self, 'db_manager'):
            self.db_manager.close_connection()
        super().closeEvent(event)


class DatabaseViewerDialog(QDialog):
    """Modal dialog for database viewing with proper Qt patterns - Claude Generated"""

    def __init__(self, database_config: DatabaseConfig = None, parent=None):
        """
        Initialize database viewer dialog.

        Args:
            database_config: Database configuration
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("📊 ALIMA Datenbank Viewer")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)  # Better default size

        # Create layout
        layout = QVBoxLayout(self)

        # Create TableWidget
        self.table_widget = TableWidget(database_config, parent=self)
        layout.addWidget(self.table_widget)

        # Add standard dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)  # Proper QDialog pattern
        layout.addWidget(button_box)

    def closeEvent(self, event):
        """Clean up when dialog is closed - Claude Generated"""
        if hasattr(self, 'table_widget'):
            self.table_widget.closeEvent(event)
        super().closeEvent(event)
