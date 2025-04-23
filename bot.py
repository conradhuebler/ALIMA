from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QTextEdit, QComboBox, QLabel, QFileDialog,
                           QHBoxLayout, QSplitter, QListWidget, QSlider, 
                           QSpinBox, QFormLayout, QDialog, QDialogButtonBox,
                           QListWidgetItem)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap

import sys
import time
from pathlib import Path
from typing import Optional, Union
from src.llm.llm_interface import LLMInterface  # Ihre LLM Interface Klasse

class SettingsDialog(QDialog):
    def __init__(self, parent=None, timeout=120):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.timeout = timeout
        
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        # Timeout settings
        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setRange(10, 300)  # 10 seconds to 5 minutes
        self.timeout_spinbox.setValue(self.timeout)
        self.timeout_spinbox.setSuffix(" seconds")
        self.timeout_spinbox.setToolTip("How long to wait before cancelling a request that appears to be stuck")
        form.addRow("Request Timeout:", self.timeout_spinbox)
        
        layout.addLayout(form)
        
        # Standard dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                          QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    
    def get_timeout(self):
        return self.timeout_spinbox.value()

class ChatbotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.llm_interface = LLMInterface()
        self.current_image = None
        self.chat_history = []
        self.generating = False
        self.current_bot_response = ""  # Sammelt die aktuelle Bot-Antwort
        self.request_timeout = 120  # Default timeout in seconds
        
        # Verbinde die Signale der LLMInterface mit unseren Slots
        self.llm_interface.text_received.connect(self.on_text_received)
        self.llm_interface.generation_finished.connect(self.on_generation_finished)
        self.llm_interface.generation_error.connect(self.on_generation_error) 
        self.llm_interface.generation_cancelled.connect(self.on_generation_cancelled)
        
        # Setze den Timeout für hängengebliebene Anfragen
        self.llm_interface.set_timeout(self.request_timeout)
        
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
        
        # Settings button
        settings_btn = QPushButton("Advanced Settings")
        settings_btn.clicked.connect(self.show_settings_dialog)
        settings_layout.addWidget(settings_btn)
        
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
        
        # Send/Stop Button Container
        button_layout = QHBoxLayout()
        
        # Send Button
        self.send_btn = QPushButton("Send Message")
        self.send_btn.clicked.connect(self.send_message)
        button_layout.addWidget(self.send_btn)
        
        # Stop Button (initially disabled)
        self.stop_btn = QPushButton("Stop Generation")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        button_layout.addWidget(self.stop_btn)
        
        chat_layout.addLayout(button_layout)
        
        splitter.addWidget(chat_widget)
        
        # Set initial split position (30% for settings, 70% for chat)
        splitter.setSizes([300, 700])
        
        self.update_models()
    
    def show_settings_dialog(self):
        dialog = SettingsDialog(self, self.request_timeout)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update with new settings
            self.request_timeout = dialog.get_timeout()
            self.llm_interface.set_timeout(self.request_timeout)
    
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
        self.image.clear()
        
    def clear_chat(self):
        self.chat_history = []
        self.chat_display.clear()
    
    def on_text_received(self, text):
        """Wird aufgerufen, wenn ein neuer Textabschnitt empfangen wird"""
        # Aktualisiere den gespeicherten Bot-Response
        self.current_bot_response += text
        
        # Aktualisiere den Chat-Eintrag
        if hasattr(self, 'bot_response_index') and self.bot_response_index is not None:
            # Aktualisiere den Text im Chat-Display
            self.chat_display.item(self.bot_response_index).setText(f"Bot: {self.current_bot_response}")
            self.chat_display.scrollToBottom()
            
            # Force the UI to update immediately
            QApplication.processEvents()
    
    def on_generation_finished(self, message):
        """Wird aufgerufen, wenn die Generierung abgeschlossen ist"""
        print(f"Generation finished: {message}")  # Debug
        self.generating = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Füge die vollständige Antwort zur Chat-Historie hinzu
        self.chat_history.append({"role": "assistant", "content": self.current_bot_response})
        self.current_bot_response = ""  # Reset für nächste Antwort
        
        # Force UI update
        QApplication.processEvents()
    
    def on_generation_error(self, error_message):
        """Wird aufgerufen bei Generierungsfehlern"""
        print(f"Generation error: {error_message}")  # Debug
        self.generating = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Füge die Fehlermeldung zum Chat hinzu
        if hasattr(self, 'bot_response_index') and self.bot_response_index is not None:
            error_text = f"Bot: Error: {error_message}"
            self.chat_display.item(self.bot_response_index).setText(error_text)
        
        # Reset für nächste Antwort
        self.current_bot_response = ""
        
        # Force UI update
        QApplication.processEvents()
    
    def on_generation_cancelled(self):
        """Wird aufgerufen, wenn die Generierung abgebrochen wurde"""
        print("Generation cancelled")  # Debug
        self.generating = False
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Markiere die Antwort als abgebrochen
        if hasattr(self, 'bot_response_index') and self.bot_response_index is not None:
            cancelled_text = f"Bot: {self.current_bot_response} [CANCELLED]"
            self.chat_display.item(self.bot_response_index).setText(cancelled_text)
            
            # Füge den abgebrochenen Teil zur Chat-Historie hinzu
            self.chat_history.append({
                "role": "assistant", 
                "content": self.current_bot_response + " [CANCELLED]"
            })
        
        # Reset für nächste Antwort
        self.current_bot_response = ""
        
        # Force UI update
        QApplication.processEvents()
    
    def stop_generation(self):
        """Benutzer hat auf den Stop-Button geklickt"""
        if self.generating:
            print("User requested cancellation")  # Debug
            self.llm_interface.cancel_generation(reason="user_requested")
            # Die Callbacks (on_generation_cancelled etc.) kümmern sich um den Rest
    
    def send_message(self):
        """Sendet die Nachricht und startet die Generierung"""
        # Wenn bereits eine Generierung läuft, ignorieren
        if self.generating:
            return
            
        user_message = self.user_input.toPlainText()
        if not user_message.strip():
            return
        
        print(f"Sending message: {user_message[:30]}...")  # Debug
        
        # Add user message to chat
        self.chat_display.addItem(f"You: {user_message}")
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Add bot response placeholder
        self.bot_response_index = self.chat_display.count()
        self.chat_display.addItem("Bot: ")
        
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
        
        # UI-Status aktualisieren
        self.generating = True
        self.current_bot_response = ""  # Reset für neue Antwort
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Force the UI to update before starting the generation
        QApplication.processEvents()
        
        print(f"Starting generation with {provider}/{model}")  # Debug
        
        # Direkt die LLM-Interface-Klasse verwenden
        self.llm_interface.generate_response(
            provider, 
            model, 
            prompt, 
            temperature=temperature,
            system=system_prompt,
            image=self.current_image,
            stream=True  # Immer Streaming verwenden
        )

def main():
    app = QApplication(sys.argv)
    window = ChatbotApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
