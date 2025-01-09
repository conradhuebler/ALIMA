from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QLineEdit, QProgressBar, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QGroupBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPalette, QFont
from typing import Optional, List, Dict, Any

class SearchInput(QWidget):
    """Erweitertes Sucheingabefeld mit Verlauf und Vorschlägen"""
    
    search_triggered = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.search_history = []
        self.max_history = 10

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Haupteingabefeld
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Suchbegriffe (durch Komma getrennt)")
        self.input_field.returnPressed.connect(self.trigger_search)

        # Verlaufsbutton
        self.history_button = QPushButton("▼")
        self.history_button.setMaximumWidth(30)
        self.history_button.clicked.connect(self.show_history)

        # Suchbutton
        self.search_button = QPushButton("Suchen")
        self.search_button.clicked.connect(self.trigger_search)

        layout.addWidget(self.input_field)
        layout.addWidget(self.history_button)
        layout.addWidget(self.search_button)

    def trigger_search(self):
        search_text = self.input_field.text().strip()
        if search_text:
            self.add_to_history(search_text)
            self.search_triggered.emit(search_text)

    def add_to_history(self, text: str):
        if text not in self.search_history:
            self.search_history.insert(0, text)
            if len(self.search_history) > self.max_history:
                self.search_history.pop()

    def show_history(self):
        if not self.search_history:
            return

        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        for item in self.search_history:
            action = menu.addAction(item)
            action.triggered.connect(lambda _, t=item: self.set_search_text(t))
        menu.exec(self.history_button.mapToGlobal(self.history_button.rect().bottomLeft()))

    def set_search_text(self, text: str):
        self.input_field.setText(text)
        self.input_field.setFocus()

class ResultsTable(QTableWidget):
    """Erweiterte Tabelle für Suchergebnisse mit zusätzlichen Funktionen"""
    
    item_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.current_results = []

    def init_ui(self):
        # Grundlegende Tabellenkonfiguration
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels([
            "Begriff", "GND-ID", "Häufigkeit", "Quelle"
        ])
        
        # Spaltenbreiten anpassen
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        # Selektion konfigurieren
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        
        # Verbinde Signals
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def set_results(self, results: List[Dict[str, Any]]):
        """Setzt neue Suchergebnisse"""
        self.current_results = results
        self.setRowCount(0)
        
        for result in results:
            row = self.rowCount()
            self.insertRow(row)
            
            # Füge Daten ein
            self.setItem(row, 0, QTableWidgetItem(result['label']))
            self.setItem(row, 1, QTableWidgetItem(result['gnd_id']))
            self.setItem(row, 2, QTableWidgetItem(str(result['count'])))
            self.setItem(row, 3, QTableWidgetItem(result['type']))
            
            # Formatiere Zeile
            if result['type'] == 'Exakt':
                self.highlight_row(row, QColor(200, 255, 200))  # Hellgrün

    def highlight_row(self, row: int, color: QColor):
        """Hebt eine Zeile farblich hervor"""
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item:
                item.setBackground(color)

    def on_selection_changed(self):
        """Behandelt Änderungen in der Auswahl"""
        selected_items = self.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        if 0 <= row < len(self.current_results):
            self.item_selected.emit(self.current_results[row])

    def sort_results(self, column: int, order: Qt.SortOrder):
        """Sortiert die Ergebnisse"""
        self.sortItems(column, order)

class StatusBar(QWidget):
    """Erweiterte Statusleiste mit Fortschrittsanzeige"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.message_timer = QTimer()
        self.message_timer.timeout.connect(self.clear_message)

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Statustext
        self.status_label = QLabel()
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Preferred
        )

        # Fortschrittsanzeige
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()

        # Cache-Info
        self.cache_label = QLabel()
        self.cache_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.cache_label)

    def show_message(self, message: str, timeout: int = 5000):
        """Zeigt eine Statusmeldung für eine bestimmte Zeit"""
        self.status_label.setText(message)
        if timeout > 0:
            self.message_timer.start(timeout)

    def clear_message(self):
        """Löscht die aktuelle Statusmeldung"""
        self.status_label.clear()
        self.message_timer.stop()

    def start_progress(self, maximum: int = 0):
        """Startet die Fortschrittsanzeige"""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

    def update_progress(self, value: int):
        """Aktualisiert den Fortschritt"""
        self.progress_bar.setValue(value)

    def stop_progress(self):
        """Stoppt die Fortschrittsanzeige"""
        self.progress_bar.hide()

    def update_cache_info(self, info: str):
        """Aktualisiert die Cache-Information"""
        self.cache_label.setText(info)

class DetailView(QGroupBox):
    """Detailansicht für ausgewählte Einträge"""
    
    def __init__(self, title: str = "Details", parent=None):
        super().__init__(title, parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Detailtext
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)

        # Aktionsbuttons
        button_layout = QHBoxLayout()
        
        self.copy_button = QPushButton("Kopieren")
        self.copy_button.clicked.connect(self.copy_details)
        
        self.export_button = QPushButton("Exportieren")
        self.export_button.clicked.connect(self.export_details)

        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.export_button)

        layout.addWidget(self.detail_text)
        layout.addLayout(button_layout)

    def set_details(self, details: Dict[str, Any]):
        """Setzt die anzuzeigenden Details"""
        text = []
        
        # Basisinformationen
        if 'label' in details:
            text.append(f"Begriff: {details['label']}")
        if 'gnd_id' in details:
            text.append(f"GND-ID: {details['gnd_id']}")
        if 'count' in details:
            text.append(f"Häufigkeit: {details['count']}")
        if 'type' in details:
            text.append(f"Typ: {details['type']}")

        # Zusätzliche Informationen
        if 'additional_info' in details:
            text.append("\nZusätzliche Informationen:")
            for key, value in details['additional_info'].items():
                text.append(f"{key}: {value}")

        self.detail_text.setText("\n".join(text))

    def copy_details(self):
        """Kopiert die Details in die Zwischenablage"""
        self.detail_text.selectAll()
        self.detail_text.copy()
        self.detail_text.textCursor().clearSelection()

    def export_details(self):
        """Exportiert die Details"""
        from PyQt6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Details exportieren",
            "",
            "Textdateien (*.txt);;Alle Dateien (*.*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.detail_text.toPlainText())
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Fehler beim Export",
                    f"Die Details konnten nicht exportiert werden:\n{str(e)}"
                )

class FilterBox(QGroupBox):
    """Widget für Filtereinstellungen"""
    
    filters_changed = pyqtSignal(dict)
    
    def __init__(self, title: str = "Filter", parent=None):
        super().__init__(title, parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Threshold-Einstellung
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Schwellenwert (%):")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 100.0)
        self.threshold_spin.setValue(1.0)
        self.threshold_spin.setSingleStep(0.1)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spin)
        layout.addLayout(threshold_layout)

        # Typ-Filter
        self.exact_check = QCheckBox("Nur exakte Treffer")
        layout.addWidget(self.exact_check)

        # Sortierung
        sort_layout = QHBoxLayout()
        sort_label = QLabel("Sortierung:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Häufigkeit", "Alphabetisch", "Relevanz"])
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.sort_combo)
        layout.addLayout(sort_layout)

        # Verbinde Signals
        self.threshold_spin.valueChanged.connect(self.emit_filters)
        self.exact_check.stateChanged.connect(self.emit_filters)
        self.sort_combo.currentTextChanged.connect(self.emit_filters)

    def emit_filters(self):
        """Sendet die aktuellen Filtereinstellungen"""
        self.filters_changed.emit({
            'threshold': self.threshold_spin.value(),
            'exact_only': self.exact_check.isChecked(),
            'sort_by': self.sort_combo.currentText()
        })

    def get_filters(self) -> Dict[str, Any]:
        """Gibt die aktuellen Filtereinstellungen zurück"""
        return {
            'threshold': self.threshold_spin.value(),
            'exact_only': self.exact_check.isChecked(),
            'sort_by': self.sort_combo.currentText()
        }

    def set_filters(self, filters: Dict[str, Any]):
        """Setzt die Filtereinstellungen"""
        if 'threshold' in filters:
            self.threshold_spin.setValue(filters['threshold'])
        if 'exact_only' in filters:
            self.exact_check.setChecked(filters['exact_only'])
        if 'sort_by' in filters:
            index = self.sort_combo.findText(filters['sort_by'])
            if index >= 0:
                self.sort_combo.setCurrentIndex(index)
