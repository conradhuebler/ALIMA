from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QComboBox, QHBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
import sys
from pathlib import Path
import time

from src.llm.llm_interface import LLMInterface

# Separate Worker-Klasse, um die Generierung in einem eigenen Thread auszuführen
class LLMWorker(QObject):
    finished = pyqtSignal()
    
    def __init__(self, llm_interface, provider, model, prompt, temperature=0.7, system="", image=None):
        super().__init__()
        self.llm = llm_interface
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.system = system
        self.image = image
    
    def run(self):
        # Generiere die Antwort im separaten Thread
        self.llm.generate_response(
            provider=self.provider,
            model=self.model,
            prompt=self.prompt,
            temperature=self.temperature,
            system=self.system,
            image=self.image,
            stream=True
        )
        self.finished.emit()

class LLMDemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Streaming Demo")
        self.setMinimumSize(800, 600)
        
        # Initialize LLM Interface
        self.llm = LLMInterface()
        
        # Thread und Worker für asynchrone Verarbeitung
        self.thread = None
        self.worker = None
        
        # Set up UI
        self.setup_ui()
        
        # Connect signals
        self.llm.text_received.connect(self.on_text_received)
        self.llm.generation_finished.connect(self.on_generation_finished)
        self.llm.generation_error.connect(self.on_error)
    
    def setup_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Provider selection
        self.provider_combo = QComboBox()
        available_providers = self.llm.get_available_providers()
        self.provider_combo.addItems(available_providers)
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        controls_layout.addWidget(self.provider_combo)
        
        # Model selection
        self.model_combo = QComboBox()
        if available_providers:
            self.update_models(available_providers[0])
        controls_layout.addWidget(self.model_combo)
        
        main_layout.addLayout(controls_layout)
        
        # Input text area
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter your prompt here...")
        main_layout.addWidget(self.input_text)
        
        # Button to generate
        self.generate_btn = QPushButton("Generate Response")
        self.generate_btn.clicked.connect(self.generate_response)
        main_layout.addWidget(self.generate_btn)
        
        # Output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)
        
        # Set central widget
        self.setCentralWidget(central_widget)
    
    def on_provider_changed(self, provider):
        """Update model choices when provider changes"""
        self.update_models(provider)
    
    def update_models(self, provider):
        """Update models combobox with models for the current provider"""
        self.model_combo.clear()
        models = self.llm.get_available_models(provider)
        self.model_combo.addItems(models)
    
    def generate_response(self):
        """Generate a response using the current settings"""
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        prompt = self.input_text.toPlainText()
        
        if not prompt:
            self.output_text.setText("Please enter a prompt.")
            return
        
        # Clear the output area
        self.output_text.clear()
        
        # Disable the generate button during generation
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        
        # Erstelle einen Worker und starte einen Thread für die Generierung
        self.thread = QThread()
        self.worker = LLMWorker(
            self.llm, 
            provider=provider,
            model=model,
            prompt=prompt,
            temperature=0.7
        )
        
        # Verbinde Worker mit Thread
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Starte den Thread
        self.thread.start()
    
    def on_text_received(self, text_chunk):
        """Handle text chunks as they arrive"""
        # Add the chunk to the output text
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text_chunk)
        self.output_text.setTextCursor(cursor)
        
        # Ensure text is visible
        self.output_text.ensureCursorVisible()
        
        # Wichtig: QApplication.processEvents() erzwingen, damit die UI aktualisiert wird
    
    def on_generation_finished(self, full_text):
        """Handle completion of text generation"""
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Response")
    
    def on_error(self, error_message):
        """Handle errors during generation"""
        self.output_text.setText(f"Error: {error_message}")
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Response")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LLMDemoWindow()
    window.show()
    sys.exit(app.exec())
