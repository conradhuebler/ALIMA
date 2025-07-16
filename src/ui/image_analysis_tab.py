import sys
import os
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, 
                            QFileDialog, QMessageBox, QLabel, QComboBox, QGroupBox, 
                            QSizePolicy, QSlider, QSpinBox, QSplitter, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPixmap, QFont
import PIL.Image
import json
from pathlib import Path
import uuid

# Simple text recognition prompt
DEFAULT_PROMPT = """Bitte extrahiere den gesamten lesbaren Text aus diesem Bild. 
Gib nur den Text zurück, ohne zusätzliche Formatierung oder Kommentare.
Achte darauf, dass der Text genau so ausgegeben wird, wie er im Bild steht."""

class ImageAnalysisWorker(QThread):
    """Worker thread for image analysis"""
    analysis_finished = pyqtSignal(str)
    analysis_error = pyqtSignal(str)
    
    def __init__(self, llm_service, provider, model, prompt, image_path, temperature=0.7, seed=None):
        super().__init__()
        self.llm_service = llm_service
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.image_path = image_path
        self.temperature = temperature
        self.seed = seed
        
    def run(self):
        try:
            request_id = str(uuid.uuid4())
            response = self.llm_service.generate_response(
                provider=self.provider,
                model=self.model,
                prompt=self.prompt,
                request_id=request_id,
                temperature=self.temperature,
                seed=self.seed,
                image=self.image_path,
                stream=False
            )
            
            # Handle generator objects (e.g., from Ollama)
            if hasattr(response, '__iter__') and not isinstance(response, str):
                # If response is a generator, collect all chunks
                full_response = ""
                for chunk in response:
                    if isinstance(chunk, str):
                        full_response += chunk
                    elif hasattr(chunk, 'text'):
                        full_response += chunk.text
                    elif hasattr(chunk, 'content'):
                        full_response += chunk.content
                    else:
                        full_response += str(chunk)
                response = full_response
            
            self.analysis_finished.emit(response)
        except Exception as e:
            self.analysis_error.emit(str(e))

class ImageAnalysisTab(QWidget):
    """Tab for image analysis with text extraction"""
    
    # Signal to send extracted text to abstract tabs
    text_extracted = pyqtSignal(str)
    
    def __init__(self, llm_service, main_window=None, parent=None):
        super().__init__(parent)
        self.llm_service = llm_service
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        
        self.image_path = None
        self.current_worker = None
        
        self.setup_ui()
        self.load_providers_and_models()
        
    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create splitter for main layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for image and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File selection
        self.file_button = QPushButton('Bild auswählen')
        self.file_button.clicked.connect(self.select_file)
        left_layout.addWidget(self.file_button)
        
        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 1px solid #cccccc;
                background-color: #f0f0f0;
                color: #666666;
            }
        """)
        self.image_preview.setText("Kein Bild ausgewählt")
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.image_preview)
        
        # Provider settings
        provider_group = QGroupBox("LLM Provider Einstellungen")
        provider_layout = QVBoxLayout()
        
        # Provider selection
        provider_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.currentTextChanged.connect(self.update_models)
        provider_layout.addWidget(self.provider_combo)
        
        # Model selection
        provider_layout.addWidget(QLabel("Modell:"))
        self.model_combo = QComboBox()
        provider_layout.addWidget(self.model_combo)
        
        # Temperature slider
        self.temperature_label = QLabel("Temperatur: 0.70")
        provider_layout.addWidget(self.temperature_label)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(70)
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)
        provider_layout.addWidget(self.temperature_slider)
        
        # Seed input
        provider_layout.addWidget(QLabel("Seed:"))
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 1000000000)
        self.seed_input.setValue(0)
        provider_layout.addWidget(self.seed_input)
        
        provider_group.setLayout(provider_layout)
        left_layout.addWidget(provider_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        main_splitter.addWidget(left_panel)
        
        # Right panel for prompt and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Prompt input
        right_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText('Prompt für die Bildanalyse...')
        self.prompt_input.setText(DEFAULT_PROMPT)
        self.prompt_input.setMaximumHeight(120)
        font = self.prompt_input.font()
        font.setPointSize(10)
        self.prompt_input.setFont(font)
        right_layout.addWidget(self.prompt_input)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton('Text extrahieren')
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.analyze_button)
        
        self.send_to_abstract_button = QPushButton('An Abstract-Tab senden')
        self.send_to_abstract_button.clicked.connect(self.send_to_abstract_tab)
        self.send_to_abstract_button.setEnabled(False)
        button_layout.addWidget(self.send_to_abstract_button)
        
        button_layout.addStretch()
        right_layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #666666;")
        right_layout.addWidget(self.status_label)
        
        # Results output
        right_layout.addWidget(QLabel("Extrahierter Text:"))
        self.output_field = QTextEdit()
        self.output_field.setReadOnly(True)
        font = self.output_field.font()
        font.setPointSize(11)
        self.output_field.setFont(font)
        right_layout.addWidget(self.output_field)
        
        main_splitter.addWidget(right_panel)
        
        # Set splitter sizes
        main_splitter.setSizes([400, 600])
        
        main_layout.addWidget(main_splitter)
        
    def load_providers_and_models(self):
        """Load available providers and models"""
        try:
            providers = self.llm_service.get_available_providers()
            self.provider_combo.addItems(providers)
            
            if providers:
                self.update_models(providers[0])
                self.status_label.setText(f"Bereit - {len(providers)} Provider verfügbar")
            else:
                self.status_label.setText("Keine Provider verfügbar")
                
        except Exception as e:
            self.logger.error(f"Error loading providers: {e}")
            self.status_label.setText(f"Fehler beim Laden der Provider: {str(e)}")
            
    def update_models(self, provider):
        """Update available models when provider changes"""
        self.model_combo.clear()
        
        try:
            models = self.llm_service.get_available_models(provider)
            self.model_combo.addItems(models)
            
            if models:
                self.model_combo.setCurrentIndex(0)
                self.status_label.setText(f"Provider {provider} bereit - {len(models)} Modelle verfügbar")
            else:
                self.status_label.setText(f"Keine Modelle für Provider {provider} verfügbar")
                
        except Exception as e:
            self.logger.error(f"Error loading models for {provider}: {e}")
            self.status_label.setText(f"Fehler beim Laden der Modelle: {str(e)}")
            
    def update_temperature_label(self, value):
        """Update temperature label"""
        self.temperature_label.setText(f"Temperatur: {value/100:.2f}")
        
    def select_file(self):
        """File selection dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Bild auswählen",
            "",
            "Bilder (*.png *.jpg *.jpeg *.gif *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_path = file_path
            self.file_button.setText(f'Datei: {Path(file_path).name}')
            self.update_image_preview()
            self.status_label.setText(f"Bild ausgewählt: {Path(file_path).name}")
            
    def update_image_preview(self):
        """Update image preview"""
        if self.image_path:
            try:
                pixmap = QPixmap(self.image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.image_preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.image_preview.setPixmap(scaled_pixmap)
                else:
                    self.image_preview.setText("Bild konnte nicht geladen werden")
            except Exception as e:
                self.logger.error(f"Error loading image preview: {e}")
                self.image_preview.setText("Fehler beim Laden des Bildes")
                
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if self.image_path:
            self.update_image_preview()
            
    def analyze_image(self):
        """Analyze the selected image"""
        if not self.image_path:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie zuerst ein Bild aus!")
            return
            
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText().strip()
        prompt = self.prompt_input.toPlainText().strip()
        
        if not provider or not model:
            QMessageBox.warning(self, "Warnung", "Bitte wählen Sie einen Provider und ein Modell aus!")
            return
            
        if not prompt:
            QMessageBox.warning(self, "Warnung", "Bitte geben Sie einen Prompt ein!")
            return
            
        # Disable UI during analysis
        self.analyze_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText(f"Analysiere Bild mit {provider} ({model})...")
        
        # Start analysis in worker thread
        self.current_worker = ImageAnalysisWorker(
            self.llm_service,
            provider,
            model,
            prompt,
            self.image_path,
            self.temperature_slider.value() / 100,
            self.seed_input.value() if self.seed_input.value() > 0 else None
        )
        
        self.current_worker.analysis_finished.connect(self.on_analysis_finished)
        self.current_worker.analysis_error.connect(self.on_analysis_error)
        self.current_worker.start()
        
    @pyqtSlot(str)
    def on_analysis_finished(self, result):
        """Handle successful analysis completion"""
        self.output_field.setText(result)
        self.send_to_abstract_button.setEnabled(True)
        
        # Re-enable UI
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analyse abgeschlossen")
        
        self.logger.info(f"Image analysis completed successfully")
        
    @pyqtSlot(str)
    def on_analysis_error(self, error):
        """Handle analysis error"""
        self.output_field.setText(f"Fehler bei der Analyse: {error}")
        
        # Re-enable UI
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Fehler bei der Analyse")
        
        self.logger.error(f"Image analysis error: {error}")
        QMessageBox.critical(self, "Fehler", f"Fehler bei der Bildanalyse:\n{error}")
        
    def send_to_abstract_tab(self):
        """Send extracted text to abstract tab"""
        text = self.output_field.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "Warnung", "Kein Text zum Senden vorhanden!")
            return
            
        # Emit signal to send text to abstract tab
        self.text_extracted.emit(text)
        
        # Show confirmation
        QMessageBox.information(self, "Erfolg", "Text wurde an die Abstract-Tab gesendet!")
        
        # Switch to abstract tab if main window is available
        if self.main_window and hasattr(self.main_window, 'tabs'):
            # Find abstract tab (usually index 1)
            for i in range(self.main_window.tabs.count()):
                if "Abstract" in self.main_window.tabs.tabText(i):
                    self.main_window.tabs.setCurrentIndex(i)
                    break
                    
        self.status_label.setText("Text an Abstract-Tab gesendet")