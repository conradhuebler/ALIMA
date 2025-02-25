import sys
import os
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTextEdit, QPushButton, QFileDialog, QMessageBox, QLabel,
                            QComboBox, QGroupBox, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import PIL.Image
from src.core.llm_interface import LLMInterface  # Ihre LLM Interface Klasse

DEFAULT_PROMPT = """Bitte analysiere dieses Bild und gib mir folgende Informationen:
1. Liste allen lesbaren Text im Bild auf
2. Beschreibe kurz den Kontext und Inhalt des Bildes
3. Hebe wichtige Details hervor"""

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
        self.setWindowTitle('Multi-LLM Bildanalyse')
        self.setGeometry(100, 100, 1200, 800)

        # Hauptwidget und Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

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
        self.provider_combo.currentTextChanged.connect(self.update_model_list)
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo)
        
        # Modell Combobox
        model_label = QLabel("Modell:")
        self.model_combo = QComboBox()
        provider_layout.addWidget(model_label)
        provider_layout.addWidget(self.model_combo)
        
        provider_group.setLayout(provider_layout)
        left_layout.addWidget(provider_group)
        
        main_layout.addWidget(left_panel)

        # Rechte Seite (Eingabe und Ausgabe)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Prompt-Eingabefeld
        prompt_label = QLabel("Prompt:")
        right_layout.addWidget(prompt_label)
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText('Gib deine Frage zum Bild ein...')
        self.input_field.setText(DEFAULT_PROMPT)
        self.input_field.setMinimumHeight(100)
        right_layout.addWidget(self.input_field)

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

    def update_model_list(self):
        """Aktualisiert die Modell-Liste basierend auf dem ausgewählten Provider"""
        self.model_combo.clear()
        provider = self.provider_combo.currentText()
        if provider:
            models = self.llm.get_available_models(provider)
            self.model_combo.addItems(models)
            self.status_label.setText(f"Provider {provider} bereit mit {len(models)} Modellen")

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
        model = self.model_combo.currentText()
        prompt = self.input_field.toPlainText()

        if not provider or not model:
            self.output_field.setText("Bitte wähle einen Provider und ein Modell aus!")
            return

        try:
            self.status_label.setText(f"Verarbeite Anfrage mit {provider} ({model})...")
            QApplication.processEvents()  # UI aktualisieren
            
            # Generiere Antwort über das LLM Interface
            response = self.llm.generate_response(
                provider=provider,
                model=model,
                prompt=prompt,
                image=self.image_path
            )
            
            self.output_field.setText(response)
            self.status_label.setText(f"Analyse mit {provider} ({model}) abgeschlossen")
            
        except Exception as e:
            self.output_field.setText(f"Ein Fehler ist aufgetreten: {str(e)}")
            self.status_label.setText("Fehler bei der Verarbeitung")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ImageAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
