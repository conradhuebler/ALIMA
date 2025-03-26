from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QTextEdit, QComboBox, QLabel, QFileDialog,
                           QHBoxLayout, QSplitter, QListWidget, QSlider)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap

import sys
import asyncio
from pathlib import Path
from typing import Optional, Union
from src.core.llm_interface import LLMInterface  # Ihre LLM Interface Klasse

class GenerateThread(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, llm_interface, provider, model, prompt, temperature, system_prompt, image=None):
        super().__init__()
        self.llm_interface = llm_interface
        self.provider = provider
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.image = image
        
    def run(self):
        response = self.llm_interface.generate_response(
            self.provider, 
            self.model, 
            self.prompt, 
            temperature=self.temperature,
            system=self.system_prompt,
            image=self.image
        )
        self.finished.emit(response)


class ChatbotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.llm_interface = LLMInterface()
        self.current_image = None
        self.chat_history = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("LLM Chatbot")
        self.setGeometry(100, 100, 1000, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create horizontal splitter for settings and chat
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left side - Settings
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        
        # Provider and Model Selection
        settings_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm_interface.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        settings_layout.addWidget(self.provider_combo)
        
        settings_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        settings_layout.addWidget(self.model_combo)
        
        # Temperature Slider
        settings_layout.addWidget(QLabel("Temperature:"))
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 20)  # 0.0 to 2.0
        self.temp_slider.setValue(7)      # Default 0.7
        self.temp_label = QLabel("0.7")
        self.temp_slider.valueChanged.connect(self.update_temperature_label)
        settings_layout.addWidget(self.temp_slider)
        settings_layout.addWidget(self.temp_label)
        
        # System Prompt
        settings_layout.addWidget(QLabel("System Prompt:"))
        self.system_prompt = QTextEdit()
        self.system_prompt.setPlaceholderText("Enter system instructions here...")
        self.system_prompt.setMinimumHeight(100)
        settings_layout.addWidget(self.system_prompt)
        
        # Image Selection
        settings_layout.addWidget(QLabel("Image:"))
        self.image_label = QLabel("No image selected")
        settings_layout.addWidget(self.image_label)
        
        image_buttons_layout = QHBoxLayout()
        select_image_btn = QPushButton("Select Image")
        select_image_btn.clicked.connect(self.select_image)
        image_buttons_layout.addWidget(select_image_btn)
        
        clear_image_btn = QPushButton("Clear Image")
        clear_image_btn.clicked.connect(self.clear_image)
        image_buttons_layout.addWidget(clear_image_btn)
        
        settings_layout.addLayout(image_buttons_layout)

        self.image = QLabel("")
        self.image.setFixedSize(200, 200)
        settings_layout.addWidget(self.image)
        
        # Clear Chat Button
        clear_chat_btn = QPushButton("Clear Chat History")
        clear_chat_btn.clicked.connect(self.clear_chat)
        settings_layout.addWidget(clear_chat_btn)
        
        splitter.addWidget(settings_widget)
        
        # Right side - Chat
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # Chat History
        chat_layout.addWidget(QLabel("Chat History:"))
        self.chat_display = QListWidget()
        self.chat_display.setWordWrap(True)
        self.chat_display.setSpacing(5)
        chat_layout.addWidget(self.chat_display)
        
        # User Input
        chat_layout.addWidget(QLabel("Your Message:"))
        self.user_input = QTextEdit()
        self.user_input.setMinimumHeight(80)
        self.user_input.setMaximumHeight(120)
        chat_layout.addWidget(self.user_input)
        
        # Send Button
        send_btn = QPushButton("Send Message")
        send_btn.clicked.connect(self.send_message)
        chat_layout.addWidget(send_btn)
        
        splitter.addWidget(chat_widget)
        
        # Set initial split position (30% for settings, 70% for chat)
        splitter.setSizes([300, 700])
        
        self.update_models()
        
    def update_temperature_label(self):
        temperature = self.temp_slider.value() / 10
        self.temp_label.setText(f"{temperature:.1f}")
    
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
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image.setPixmap(scaled_pixmap)
            
    def clear_image(self):
        self.current_image = None
        self.image_label.setText("No image selected")
        self.image.setPixmap(None)
        
    def clear_chat(self):
        self.chat_history = []
        self.chat_display.clear()
        
    def send_message(self):
        user_message = self.user_input.toPlainText()
        if not user_message.strip():
            return
            
        # Add user message to chat
        self.chat_display.addItem(f"You: {user_message}")
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Add "Bot is typing..." message
        typing_index = self.chat_display.count()
        self.chat_display.addItem("Bot is typing...")
        
        # Prepare for response generation
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        system_prompt = self.system_prompt.toPlainText()
        temperature = self.temp_slider.value() / 10
        
        # Build context from chat history
        if len(self.chat_history) > 1:
            # Create a proper conversational context
            context = "\n\nPrevious messages:\n"
            for msg in self.chat_history[:-1]:  # All except current message
                prefix = "User" if msg["role"] == "user" else "Bot"
                context += f"{prefix}: {msg['content']}\n"
            prompt = context + "\n\nCurrent message: " + user_message
        else:
            prompt = user_message
        
        # Clear input field
        self.user_input.clear()
        
        # Start generation in separate thread
        self.thread = GenerateThread(
            self.llm_interface, 
            provider, 
            model, 
            prompt, 
            temperature,
            system_prompt,
            self.current_image
        )
        self.thread.finished.connect(lambda response: self.update_chat(response, typing_index))
        self.thread.start()
        
    def update_chat(self, response, typing_index):
        # Remove "Bot is typing..." message
        self.chat_display.takeItem(typing_index)
        
        # Add bot response to chat
        self.chat_display.addItem(f"Bot: {response}")
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Scroll to the bottom of the chat
        self.chat_display.scrollToBottom()

def main():
    app = QApplication(sys.argv)
    window = ChatbotApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
