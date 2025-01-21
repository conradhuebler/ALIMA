from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QMessageBox, QProgressBar,
    QSlider, QSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from ..core.ai_processor import AIProcessor
from ollama import chat
from ollama import ChatResponse

from pydantic import BaseModel



# Define the schema for the response
class FriendInfo(BaseModel):
  name: str
  age: int
  is_available: bool


class FriendList(BaseModel):
  friends: list[FriendInfo]


import json
import logging

class AbstractTab(QWidget):
    keywords_extracted = pyqtSignal(str)
    abstract_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ai_processor = AIProcessor()
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self.handle_ai_response) 
        self.need_keywords = False

        self.setup_ui()
        self.logger = logging.getLogger(__name__)
        self.template_name = ""

    def setup_ui(self):
        """Erstellt die UI-Komponenten"""
        layout = QVBoxLayout(self)

        # Abstract-Eingabe
        abstract_label = QLabel("Abstract:")
        abstract_label.setToolTip("Fügen Sie hier den zu analysierenden Abstract ein")
        layout.addWidget(abstract_label)
        
        self.abstract_edit = QTextEdit()
        self.abstract_edit.setPlaceholderText("Fügen Sie hier den Abstract ein...")
        self.abstract_edit.textChanged.connect(self.update_input)

        layout.addWidget(self.abstract_edit)

        # Keywords-Eingabe
        keywords_label = QLabel()

        if self.need_keywords:
            keywords_label.setText("Es müssen zwingend OGND-Keywords angebeben werden:")
        else:
            keywords_label.setText("Vorhandene Keywords (optional):")

        keywords_label.setToolTip("Fügen Sie hier bereits vorhandene Keywords ein")
        layout.addWidget(keywords_label)
        
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setPlaceholderText("Vorhandene Keywords, eines pro Zeile...")
        #self.keywords_edit.setMaximumHeight(100) 
        self.keywords_edit.textChanged.connect(self.update_input)

        layout.addWidget(self.keywords_edit)

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Prompt...")
        #self.prompt.setMaximumHeight(100)
        layout.addWidget(self.prompt)

        config_layout = QHBoxLayout()
        self.ki_temperature = QSlider(Qt.Orientation.Horizontal)
        self.ki_temperature.setRange(0, 100)
        self.ki_temperature.setValue(70)
        self.ki_temperature.setTickInterval(1)
        self.ki_temperature.valueChanged.connect(self.update_temperature_label)

        # Temperatur-Label
        self.temperature_label = QLabel(f"Temperatur: {self.ki_temperature.value()}")
        config_layout.addWidget(self.temperature_label)
        config_layout.addWidget(self.ki_temperature)

        self.ki_seed = QSpinBox()
        self.ki_seed.setRange(0, 1000000000)
        self.ki_seed.setValue(0)

        config_layout.addWidget(self.ki_seed)

        self.llama = QCheckBox("Llama")
        self.llama.setChecked(False)
        config_layout.addWidget(self.llama)

        layout.addLayout(config_layout)
        # Button-Bereich
        button_layout = QHBoxLayout()
        
        self.analyze_button = QPushButton("Analyse starten")
        self.analyze_button.setToolTip("Startet die KI-gestützte Analyse des Abstracts")
        self.analyze_button.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_button)
        
        self.clear_button = QPushButton("Zurücksetzen")
        self.clear_button.setToolTip("Löscht alle Eingaben und Ergebnisse")
        self.clear_button.clicked.connect(self.clear_fields)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Ergebnisbereich
        results_label = QLabel("Analyseergebnis:")
        results_label.setToolTip("Hier erscheinen die von der KI vorgeschlagenen Schlagworte")
        layout.addWidget(results_label)
        
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setPlaceholderText("Hier erscheinen die Analyseergebnisse...")
        layout.addWidget(self.results_edit)

    def update_temperature_label(self, value):
        self.temperature_label.setText(f"Temperatur: {value}")

    def update_input(self):
        self.logger.info(self.template_name)
        self.prompt.setPlainText(self.ai_processor.set_input(
            self.abstract_edit.toPlainText().strip(),
            self.keywords_edit.toPlainText().strip(),
            template_name=self.template_name
        ))
        self.abstract_changed.emit(self.abstract_edit.toPlainText().strip())

    def prompt_generated(self, prompt):
        self.prompt.setPlainText(prompt,)

    def set_keywords(self, keywords):
        self.keywords_edit.setPlainText(keywords)
        self.update_input()

    def set_abstract(self, abstract):
        self.abstract_edit.setPlainText(abstract)
        self.update_input()

    def start_analysis(self):
        """Startet die Analyse des Abstracts"""
        abstract = self.abstract_edit.toPlainText().strip()
        keywords = self.keywords_edit.toPlainText().strip()

        if not abstract:
            QMessageBox.warning(
                self,
                "Warnung",
                "Bitte geben Sie einen Abstract ein."
            )
            return

        if not keywords and self.need_keywords:
            QMessageBox.warning(
                self,
                "Warnung",
                "Ohne GND-Keywords läuft hier nix."
            )
            return
        # UI während der Analyse deaktivieren
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Unbestimmter Fortschritt
        #self.prompt.setPlainText(self.ai_processor.generated_prompt)
        #self.logger.info(self.ai_processor.generated_prompt)
        self.logger.info(self.prompt.toPlainText())
        if self.llama.isChecked():
            self.progress_bar.setVisible(True)
            self.set_ui_enabled(False)
            response: ChatResponse = chat(model='llama3:latest', 
                messages=[
                    {
                        'role': 'user',
                        'content': self.prompt.toPlainText()
                    },
                ],
                #format=FriendList.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
                options={'temperature': self.ki_temperature.value() / 100, 'seed' : self.ki_seed.value() },  # Make responses more deterministic
      
                )
            
            result = response['message']['content']
            #friends_response = FriendList.model_validate_json(response.message.content)
            #print(friends_response)
            self.results_edit.setPlainText(result)
            keywords = self.extract_keywords(result)
            self.keywords_extracted.emit(keywords)
            self.progress_bar.setVisible(False)
            self.set_ui_enabled(True)
        else:
            try:
                # Bereite Request vor und erhalte Request und Daten
                #request, data = self.ai_processor.prepare_request(abstract, keywords)
                seed = self.ki_seed.value()
                temperature = self.ki_temperature.value() / 100
                
                request, data = self.ai_processor.prepare_request(self.prompt.toPlainText(), temperature=temperature, seed=seed)
                # Sende Request mit separaten Daten
                self.network_manager.post(request, data)
                
            except Exception as e:
                self.handle_error("Fehler bei der Anfrage", str(e))
                self.set_ui_enabled(True)
                self.progress_bar.setVisible(False)

    @pyqtSlot(QNetworkReply)
    def handle_ai_response(self, reply: QNetworkReply):
        """Verarbeitet die Antwort der AI-API"""
        # UI wieder aktivieren
        self.set_ui_enabled(True)
        self.progress_bar.setVisible(False)

        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                response_data = json.loads(str(reply.readAll(), 'utf-8'))
                result = self.ai_processor.process_response(response_data)
                self.logger.info(f"AI-Response: {result}")
                self.results_edit.setPlainText(result)
                keywords = self.extract_keywords(result)
                self.keywords_extracted.emit(keywords)
            except Exception as e:
                self.handle_error(
                    "Verarbeitungsfehler",
                    f"Fehler bei der Verarbeitung der Antwort: {str(e)}"
                )
        else:
            self.handle_error(
                "Netzwerkfehler",
                f"Fehler bei der API-Anfrage: {reply.errorString()}"
            )

        reply.deleteLater()

    def extract_keywords(self, response_text):
        # Extrahiere die Schlagworte aus dem Antworttext
        keywords = response_text.replace("*","").split('\n')
        quoted_keywords = [f'"{keyword.strip()}"' for keyword in keywords if keyword.strip()]
        return ', '.join(quoted_keywords)
    
    def handle_error(self, title: str, message: str):
        """Zeigt eine Fehlermeldung an"""
        QMessageBox.critical(self, title, message)
        self.ai_processor.logger.error(f"{title}: {message}")

    def set_ui_enabled(self, enabled: bool):
        """Aktiviert/Deaktiviert UI-Elemente"""
        self.analyze_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.abstract_edit.setEnabled(enabled)
        self.keywords_edit.setEnabled(enabled)

    def clear_fields(self):
        """Setzt alle Felder zurück"""
        if self.abstract_edit.toPlainText() or self.keywords_edit.toPlainText() or self.results_edit.toPlainText():
            reply = QMessageBox.question(
                self,
                "Bestätigung",
                "Möchten Sie wirklich alle Felder zurücksetzen?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.abstract_edit.clear()
                self.keywords_edit.clear()
                self.results_edit.clear()
