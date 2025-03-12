import sys
import os
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTextEdit, QPushButton, QFileDialog, QMessageBox, QLabel,
                            QComboBox, QGroupBox, QSizePolicy, QSlider, QSpinBox, QListView, QSplitter)
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import PIL.Image
import json
from pathlib import Path

from src.core.llm_interface import LLMInterface  # Ihre LLM Interface Klasse

DEFAULT_PROMPT = """Bitte analysiere dieses Bild und gib mir folgende Informationen:
1. Liste allen lesbaren Text im Bild auf
2. Beschreibe kurz den Kontext und Inhalt des Bildes
3. Hebe wichtige Details hervor

#Gib außerdem noch eine CSV-formatierte Ausgabe an:

Ort, Text direkt unter dem Bild, Datum, Eigene Beschreibung
"""

class ImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Logging konfigurieren
        logging.basicConfig(level=logging.INFO)
        
        # LLM Interface initialisieren
        self.llm = LLMInterface()
        
        self.image_path = None
        self.initUI()

    def initUI(self):
        self.current_file_name = ""
        self.setWindowTitle('Multi-LLM Bildanalyse')
        self.setGeometry(100, 100, 1200, 800)

        # Hauptwidget und Layout
        main_widget = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        very_left = QWidget()
        very_left_layout = QVBoxLayout(very_left)

        self.select_dir = QPushButton('Verzeichnis wählen')
        self.select_dir.clicked.connect(self.select_directory)
        very_left_layout.addWidget(self.select_dir)

        self.directory_list = QListView()
        self.directory_model = QFileSystemModel()
        self.directory_model.setRootPath('')
        self.directory_list.setModel(self.directory_model)
        very_left_layout.addWidget(self.directory_list)

        self.batch_mode = QPushButton('Batch-Modus starten')
        self.batch_mode.clicked.connect(self.start_batch_mode)
        very_left_layout.addWidget(self.batch_mode)
        main_widget.addWidget(very_left)
        #main_layout.addWidget(very_left)
 
        # Linke Seite (Bildauswahl und Vorschau)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Dateiauswahl-Button
        self.file_button = QPushButton('Bild auswählen')
        self.file_button.clicked.connect(self.select_file)
        left_layout.addWidget(self.file_button)

        # Bildvorschau
        self.image_preview = QLabel()
        self.image_preview.setMinimumSize(400, 400)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 1px solid #cccccc;
                background-color: #f0f0f0;
            }
        """)
        self.image_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        left_layout.addWidget(self.image_preview)
        
        # Provider-Auswahl Gruppe
        provider_group = QGroupBox("LLM Provider Einstellungen")
        provider_layout = QVBoxLayout()
        
        # Provider Combobox
        provider_label = QLabel("Provider:")
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo)
        
        # Modell Combobox
        model_label = QLabel("Modell:")
        self.model_combo = QComboBox()
        provider_layout.addWidget(model_label)
        provider_layout.addWidget(self.model_combo)
        
        provider_group.setLayout(provider_layout)
        left_layout.addWidget(provider_group)
        main_widget.addWidget(left_panel)
        #main_layout.addWidget(left_panel)

        # Rechte Seite (Eingabe und Ausgabe)
        right_panel = QWidget()
        main_widget.addWidget(right_panel)
        right_layout = QVBoxLayout(right_panel)

        # Prompt-Eingabefeld
        prompt_label = QLabel("Prompt:")
        right_layout.addWidget(prompt_label)
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText('Gib deine Frage zum Bild ein...')
        self.input_field.setText(DEFAULT_PROMPT)
        self.input_field.setMinimumHeight(100)
        self.load_button = QPushButton('Lade Prompt')
        self.load_button.clicked.connect(self.load_prompt)
        self.save_button = QPushButton('Speichere Prompt')
        self.save_button.clicked.connect(self.save_prompt)
        hbox = QHBoxLayout()
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.save_button)
        right_layout.addLayout(hbox)
        right_layout.addWidget(self.input_field)

        # Temperatur Slider
        config_layout = QVBoxLayout()
        self.ki_temperature = QSlider(Qt.Orientation.Horizontal)
        self.ki_temperature.setRange(0, 100)
        self.ki_temperature.setValue(70)  # Default 0.7
        self.ki_temperature.setTickInterval(1)
        self.ki_temperature.valueChanged.connect(self.update_temperature_label)

        self.temperature_label = QLabel(f"Temperatur: {self.ki_temperature.value()/100:.2f}")
        config_layout.addWidget(self.temperature_label)
        config_layout.addWidget(self.ki_temperature)

        # Seed Input
        config_layout.addWidget(QLabel("Seed:"))
        self.ki_seed = QSpinBox()
        self.ki_seed.setRange(0, 1000000000)
        self.ki_seed.setValue(0)
        config_layout.addWidget(self.ki_seed)

        right_layout.addLayout(config_layout)

        # Analyse-Button
        self.analyze_button = QPushButton('Analysieren')
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        right_layout.addWidget(self.analyze_button)

        # Status-Label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #666666;")
        right_layout.addWidget(self.status_label)

        # Ausgabefeld
        output_label = QLabel("Ergebnis:")
        right_layout.addWidget(output_label)
        self.output_field = QTextEdit()
        self.output_field.setReadOnly(True)
        right_layout.addWidget(self.output_field)

        main_layout.addWidget(right_panel)

        # Setze das Größenverhältnis zwischen linkem und rechtem Panel
        main_layout.setStretch(0, 4)  # Linkes Panel
        main_layout.setStretch(1, 6)  # Rechtes Panel
        
        # Initial Provider und Modell-Liste aktualisieren
        if self.provider_combo.count() > 0:
            self.update_model_list()
        self.recommendations_file = Path("model_recommendations.json")
        self.load_recommendations()
        self.set_model_recommendations("vision")

        

    def load_recommendations(self):
        """Lädt die Modell-Empfehlungen aus der JSON-Datei"""
        try:
            if not self.recommendations_file.exists():
                logging.warning(f"Recommendations file not found: {self.recommendations_file}")
                self.create_default_recommendations()
                return

            with open(self.recommendations_file, 'r', encoding='utf-8') as f:
                self.recommendations = json.load(f)
                
            logging.info(f"Successfully loaded recommendations from {self.recommendations_file}")
            
        except Exception as e:
            logging.error(f"Error loading recommendations: {e}")

    def set_model_recommendations(self, use_case: str):
        """Setzt die Modell-Empfehlungen basierend auf dem Anwendungsfall"""
        if use_case in self.recommendations:
            self.recommended_models = self.recommendations[use_case]["recommended"]
            self.model_descriptions = self.recommendations[use_case]["descriptions"]
            # Update Modell-Liste wenn bereits initialisiert
            if hasattr(self, 'provider_combo'):
                self.update_models(self.provider_combo.currentText())
        else:
            logging.warning(f"Unknown use case: {use_case}")

    def update_models(self, provider: str):
        """Update available models when provider changes"""
        self.model_combo.clear()
        
        # Hole alle verfügbaren Modelle
        all_models = self.llm.get_available_models(provider)
        
        # Hole empfohlene Modelle für diesen Provider
        recommended_available = False
        recommended = self.recommended_models.get(provider, [])
        logging.info(f"Empfohlene Modelle für {provider}: {recommended}")
        if recommended:
            # Füge empfohlene Modelle zuerst hinzu
            recommended_group = "↳ Empfohlene Modelle"
            self.model_combo.addItem(recommended_group)
            recommended_available = [model for model in recommended if model in all_models]
            
            for model in recommended_available:
                self.model_combo.addItem(f"  {model}")
                # Setze Tooltip mit Beschreibung
                idx = self.model_combo.count() - 1
                description = self.model_descriptions.get(provider, {}).get(model, "")
                self.model_combo.setItemData(idx, description, Qt.ItemDataRole.ToolTipRole)
            
            if recommended_available and len(all_models) > len(recommended_available):
                self.model_combo.addItem("↳ Weitere verfügbare Modelle")
            
            # Füge restliche Modelle hinzu
            other_models = [model for model in all_models if model not in recommended_available]
            for model in other_models:
                self.model_combo.addItem(f"  {model}")

        else:
            # Wenn keine Empfehlungen, füge alle Modelle hinzu
            self.model_combo.addItems(all_models)

        # Wähle das erste empfohlene Modell aus, falls verfügbar
        if recommended_available:
            self.model_combo.setCurrentText(f"  {recommended_available[0]}")


    def load_prompt(self):
        """Lädt Prompt aus einer Datei"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Prompt laden",
            "",
            "Textdateien (*.txt)"
        )
        if file_name:
            with open(file_name, 'r') as file:
                prompt = file.read()
                self.input_field.setText(prompt)
                self.status_label.setText(f"Prompt aus {file_name} geladen")

    def save_prompt(self):
        """Speichert Prompt in eine Datei"""
        prompt = self.input_field.toPlainText()
        if not prompt:
            self.status_label.setText("Kein Prompt zum Speichern vorhanden")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Prompt speichern",
            "",
            "Textdateien (*.txt)"
        )
        if file_name:
            with open(file_name, 'w') as file:
                file.write(prompt)
                self.status_label.setText(f"Prompt in {file_name} gespeichert")

    def update_model_list(self):
        """Aktualisiert die Modell-Liste basierend auf dem ausgewählten Provider"""
        self.model_combo.clear()
        provider = self.provider_combo.currentText()
        if provider:
            models = self.llm.get_available_models(provider)
            self.model_combo.addItems(models)
            self.status_label.setText(f"Provider {provider} bereit mit {len(models)} Modellen")

    def update_temperature_label(self, value):
        """Update temperature label to show actual value"""
        self.temperature_label.setText(f"Temperatur: {value/100:.2f}")


    def select_file(self):
        """Dateiauswahl-Dialog"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Bild auswählen",
            "",
            "Bilder (*.png *.jpg *.jpeg *.gif *.bmp)"
        )
        if file_name:
            self.image_path = file_name
            self.file_button.setText(f'Ausgewählte Datei: {file_name.split("/")[-1]}')
            self.update_image_preview()
            current_file_name = Path(file_name).name
            self.status_label.setText(f"Bild {current_file_name} ausgewählt")

    def update_image_preview(self):
        """Aktualisiert die Bildvorschau"""
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            scaled_pixmap = pixmap.scaled(
                self.image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_preview.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Behandelt Fenster-Größenänderungen"""
        super().resizeEvent(event)
        if hasattr(self, 'image_path') and self.image_path:
            self.update_image_preview()

    def analyze_image(self):
        """Führt die Bildanalyse durch"""
        if not self.image_path:
            self.output_field.setText("Bitte wähle zuerst ein Bild aus!")
            return

        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText().strip()
        prompt = self.input_field.toPlainText()

        if not provider or not model:
            self.output_field.setText("Bitte wähle einen Provider und ein Modell aus!")
            return

        try:
            self.status_label.setText(f"Verarbeite Anfrage mit {provider} ({model})...")
            QApplication.processEvents()  # UI aktualisieren
            logging.info(f"Analyzing image with {provider} ({model})")
            # Generiere Antwort über das LLM Interface
            response = self.llm.generate_response(
                provider=provider,
                model=model,
                prompt=prompt,
                temperature=self.ki_temperature.value() / 100,
                seed=self.ki_seed.value() if self.ki_seed.value()> 0 else None,
                image=self.image_path
            )
            logging.info(f"Response: {response}")
            self.result = response
            self.output_field.setText(response)
            self.status_label.setText(f"Analyse mit {provider} ({model}) abgeschlossen")
            
        except Exception as e:
            self.output_field.setText(f"Ein Fehler ist aufgetreten: {str(e)}")
            self.status_label.setText("Fehler bei der Verarbeitung")

    def save_results(self):
        """Speichert die Analyseergebnisse in eine Datei"""
        results = self.result
        if not results:
            self.status_label.setText("Keine Ergebnisse zum Speichern vorhanden")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Ergebnisse speichern",
            "",
            "Textdateien (*.txt)"
        )
        if file_name:
            with open(file_name, 'w') as file:
                file.write(results)
                self.status_label.setText(f"Ergebnisse in {file_name} gespeichert")

    def auto_save(self):
        """Speichert die Analyseergebnisse automatisch"""
        results = self.result
        if not results:
            return

        file_name = f"{self.directory}/{self.current_file_name}.txt"
        with open(file_name, 'w') as file:
            file.write(results)
            self.status_label.setText(f"Ergebnisse in {file_name} gespeichert")

    def select_directory(self):
        """Ordnerauswahl-Dialog"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Ordner auswählen",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if directory:
            self.directory_list.setRootIndex(self.directory_model.setRootPath(directory))
            self.status_label.setText(f"Ordner {directory} ausgewählt")
            self.directory = directory
            logging.info(f"Selected directory: {directory}")

    def start_batch_mode(self):
        """Startet den Batch-Modus für alle Bilder im ausgewählten Ordner"""
        if not self.directory:
            self.status_label.setText("Bitte wähle zuerst einen Ordner aus!")
            return

        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        prompt = self.input_field.toPlainText()

        if not provider or not model:
            self.status_label.setText("Bitte wähle einen Provider und ein Modell aus!")
            return

        # Alle Bilder im Ordner durchgehen
        for file_name in os.listdir(self.directory):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.image_path = os.path.join(self.directory, file_name)
                self.current_file_name = file_name
                self.update_image_preview()
                self.analyze_image()
                self.auto_save()
                QApplication.processEvents()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ImageAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
