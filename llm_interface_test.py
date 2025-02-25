from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QTextEdit, QComboBox, QLabel, QFileDialog,
                           QHBoxLayout)
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import asyncio
from pathlib import Path
from src.core.llm_interface import LLMInterface  # Ihre LLM Interface Klasse

class GenerateThread(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, llm_interface, provider, model, prompt, image=None):
        super().__init__()
        self.llm_interface = llm_interface
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.image = image
        
    def run(self):
        response = self.llm_interface.generate_response(
            self.provider, 
            self.model, 
            self.prompt, 
            self.image
        )
        self.finished.emit(response)


class LLMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.llm_interface = LLMInterface()
        self.current_image = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("LLM Interface")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Provider and Model Selection
        selection_layout = QHBoxLayout()
        
        provider_layout = QVBoxLayout()
        provider_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm_interface.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        provider_layout.addWidget(self.provider_combo)
        selection_layout.addLayout(provider_layout)
        
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        selection_layout.addLayout(model_layout)
        
        layout.addLayout(selection_layout)
        
        # Image Selection
        image_layout = QHBoxLayout()
        self.image_label = QLabel("No image selected")
        image_layout.addWidget(self.image_label)
        
        select_image_btn = QPushButton("Select Image")
        select_image_btn.clicked.connect(self.select_image)
        image_layout.addWidget(select_image_btn)
        
        clear_image_btn = QPushButton("Clear Image")
        clear_image_btn.clicked.connect(self.clear_image)
        image_layout.addWidget(clear_image_btn)
        
        layout.addLayout(image_layout)
        
        # Prompt Input
        layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setMinimumHeight(100)
        layout.addWidget(self.prompt_input)
        
        # Generate Button
        generate_btn = QPushButton("Generate Response")
        generate_btn.clicked.connect(self.generate_response)
        layout.addWidget(generate_btn)
        
        # Response Output
        layout.addWidget(QLabel("Response:"))
        self.response_output = QTextEdit()
        self.response_output.setReadOnly(True)
        self.response_output.setMinimumHeight(200)
        layout.addWidget(self.response_output)
        
        self.update_models()
        
    def update_models(self):
        provider = self.provider_combo.currentText()
        models = self.llm_interface.get_available_models(provider)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        
    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.current_image = file_name
            self.image_label.setText(Path(file_name).name)
            
    def clear_image(self):
        self.current_image = None
        self.image_label.setText("No image selected")
        
    def generate_response(self):
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        prompt = self.prompt_input.toPlainText()
        
        self.thread = GenerateThread(
            self.llm_interface, 
            provider, 
            model, 
            prompt, 
            self.current_image
        )
        self.thread.finished.connect(self.update_response)
        self.thread.start()
        
    def update_response(self, response):
        self.response_output.setPlainText(response)

def main():
    app = QApplication(sys.argv)
    window = LLMApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
