from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QMessageBox, QProgressBar,
    QSlider, QSpinBox, QCheckBox, QComboBox, QSplitter, QListWidget,
    QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
#from ..core.ai_processor import AIProcessor
from ..core.llm_interface import LLMInterface
from ..utils.config import Config, ConfigSection, AIConfig
from ..core.prompt_manager import PromptManager
from pathlib import Path
import json
import logging
import re


class AbstractTab(QWidget):
    keywords_extracted = pyqtSignal(str)
    abstract_changed = pyqtSignal(str)
    final_list = pyqtSignal(str)

    def __init__(self, parent=None, recommendations_file: Path = Path(__file__).parent.parent.parent / "model_recommendations.json"):
        super().__init__(parent)
        #self.ai_processor = AIProcessor()
        self.llm = LLMInterface()
        self.need_keywords = False
        self.logger = logging.getLogger(__name__)
        self.template_name = ""
        # Initialisiere leere Empfehlungen
        self.recommended_models = []
        self.current_template = ""
        self.model_descriptions = {}
        self.recommendations_file = recommendations_file
        self.propmpt_file = Path(__file__).parent.parent.parent / "prompts.json"
        self.promptmanager = PromptManager(self.propmpt_file)
        self.task = ""
        # Lade Empfehlungen aus JSON
        self.load_recommendations()
        
        # Setze Standard-Empfehlungen
        self.set_model_recommendations("default")
        self.required = []
        self.setup_ui()

    def load_recommendations(self):
        """Lädt die Modell-Empfehlungen aus der JSON-Datei"""
        try:
            if not self.recommendations_file.exists():
                self.logger.warning(f"Recommendations file not found: {self.recommendations_file}")
                self.create_default_recommendations()
                return

            with open(self.recommendations_file, 'r', encoding='utf-8') as f:
                self.recommendations = json.load(f)
                
            self.logger.info(f"Successfully loaded recommendations from {self.recommendations_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading recommendations: {e}")
            self.create_default_recommendations()

    def load_prompts(self):
        """Lädt die Prompt-Templates aus der JSON-Datei"""
        try:
            if not self.propmpt_file.exists():
                self.logger.warning(f"Prompt file not found: {self.propmpt_file}")
                self.create_default_prompts()
                return

            with open(self.propmpt_file, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
                
            self.logger.info(f"Successfully loaded prompts from {self.propmpt_file}")
            #self.logger.info(self.prompts)
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")

    def set_task(self, task: str):
        """Setzt den Anwendungsfall für die Modell-Empfehlungen"""
        # self.set_model_recommendations(task)
        self.task = task
        self.logger.info(f"Set task to {task}")
        self.logger.info(self.promptmanager.get_available_models(task))
        self.logger.info(self.promptmanager.get_required_fields(task))
        self.recommended_models = self.promptmanager.get_available_models(task)
        self.update_models(self.provider_combo.currentText())


    def set_model_recommendations(self, use_case: str):
        """Setzt die Modell-Empfehlungen basierend auf dem Anwendungsfall"""
        #if use_case in self.recommendations:
        #    self.recommended_models = self.recommendations[use_case]["recommended"]
        #    self.model_descriptions = self.recommendations[use_case]["descriptions"]
        #    # Update Modell-Liste wenn bereits initialisiert
        #    if hasattr(self, 'provider_combo'):
        #        self.update_models(self.provider_combo.currentText())
        #else:
        #    self.logger.warning(f"Unknown use case: {use_case}")
    
    def update_models(self, provider: str):
        """Update available models when provider changes"""
        self.model_combo.clear()
        
        # Hole alle verfügbaren Modelle
        all_models = self.llm.get_available_models(provider)
        recommended_available = False

        # Hole empfohlene Modelle für diesen Provider
        #recommended = self.recommended_models.get(provider, [])
        self.logger.info(f"Empfohlene Modelle für {provider}: {self.recommended_models}")
        if self.recommended_models:
            # Füge empfohlene Modelle zuerst hinzu
            recommended_group = "↳ Empfohlene Modelle"
            self.model_combo.addItem(recommended_group)
            recommended_available = [model for model in self.recommended_models if model in all_models]
            
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
            self.setModel(recommended_available[0])
        

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
        keywords_label.setText("Vorhandene Keywords (optional):" if not self.need_keywords 
                             else "Es müssen zwingend OGND-Keywords angebeben werden:")
        keywords_label.setToolTip("Fügen Sie hier bereits vorhandene Keywords ein")
        layout.addWidget(keywords_label)
        
        self.keywords_edit = QTextEdit()
        self.keywords_edit.setPlaceholderText("Vorhandene Keywords, eines pro Zeile...")
        self.keywords_edit.textChanged.connect(self.update_input)
        layout.addWidget(self.keywords_edit)

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Prompt...")
        layout.addWidget(self.prompt)

        # KI-Konfiguration
        config_layout = QHBoxLayout()

        # Provider Auswahl
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        config_layout.addWidget(QLabel("Provider:"))
        config_layout.addWidget(self.provider_combo)

        # Model Auswahl
        self.model_combo = QComboBox()
        config_layout.addWidget(QLabel("Model:"))
        config_layout.addWidget(self.model_combo)
        self.model_combo.currentTextChanged.connect(self.setModel)
        # Temperatur Slider
        self.ki_temperature = QSlider(Qt.Orientation.Horizontal)
        self.ki_temperature.setRange(0, 100)
        self.ki_temperature.setValue(0)  # Default 0.7
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
        layout.addWidget(results_label)
        
        self.result_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.result_splitter)
        self.results_edit = QTextEdit()
        self.results_edit.setReadOnly(True)
        self.results_edit.setPlaceholderText("Hier erscheinen die Analyseergebnisse...")
        self.result_splitter.addWidget(self.results_edit)

        self.results_list = QListWidget()
        self.results_list.setFixedWidth(200)
        self.result_splitter.addWidget(self.results_list)

        self.results_list.itemClicked.connect(self.show_result)

        # Initial models update
        self.update_models(self.provider_combo.currentText())

    def setModel(self, model):
        config = self.promptmanager.get_prompt_config(self.task, model)
        print(f"Prompt config for {model}:", {
        'temp': config['temp'],
        'p-value': config['p-value'],
        'prompt': config['prompt'],  # Showing just the beginning for brevity
        'system': config['system']
        })
        self.ki_temperature.setValue(int(config["temp"]*100))
        self.current_template = config["prompt"]
        self.system = config["system"]
        self.logger.info(self.system)
        self.set_input()

    def update_temperature_label(self, value):
        """Update temperature label to show actual value"""
        self.temperature_label.setText(f"Temperatur: {value/100:.2f}")

    def start_analysis(self):
        """Startet die Analyse des Abstracts"""
        self.set_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        try:
            response = self.llm.generate_response(
                provider=self.provider_combo.currentText(),
                model=self.model_combo.currentText(),
                prompt=self.prompt.toPlainText(),
                temperature=self.ki_temperature.value() / 100,
                seed=self.ki_seed.value() if self.ki_seed.value() > 0 else None,
                system=self.system
            )
            
            self.results_edit.setPlainText(response)
            item = QListWidgetItem(self.model_combo.currentText())
            item.setToolTip(self.prompt.toPlainText())
            item.setData(Qt.ItemDataRole.UserRole, response)
            self.results_list.addItem(item)
            keywords = self.extract_keywords(response)
            self.keywords_extracted.emit(keywords)
            
        except Exception as e:
            self.handle_error("Fehler bei der Anfrage", str(e))
        finally:
            self.set_ui_enabled(True)
            self.progress_bar.setVisible(False)

    def update_input(self):
        #self.logger.info(self.template_name)
        self.prompt.setPlainText(self.set_input())
        self.logger.info(self.prompt.toPlainText())
        self.abstract_changed.emit(self.abstract_edit.toPlainText().strip())

    def prompt_generated(self, prompt):
        self.prompt.setPlainText(prompt,)

    def set_keywords(self, keywords):
        self.keywords_edit.setPlainText(keywords)
        self.update_input()

    def set_abstract(self, abstract):
        self.abstract_edit.setPlainText(abstract)
        self.update_input()

    def set_input(self):
        """
        Setzt die Eingabedaten für die AI-Verarbeitung.
        
        Args:
            abstract: Der zu analysierende Abstract
            keywords: Vorhandene Keywords (optional)
            template_name: Name des Prompt-Templates
        """
        template = self.current_template
        self.logger.info("Bereit prompt vor")
        self.logger.info(template)
        self.logger.info(self.abstract_edit.toPlainText().strip(), self.keywords_edit.toPlainText().strip())
          # Bereite die Variablen vor
        variables = {
            "abstract": self.abstract_edit.toPlainText().strip(),
            "keywords": self.keywords_edit.toPlainText().strip() if self.keywords_edit.toPlainText().strip() else "Keine Keywords vorhanden"
        }
        self.logger.info(variables)
        try:
            # Erstelle den Prompt
            prompt = template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Fehlende Variable im Template: {e}")
        self.logger.info(prompt, variables)
        self.generated_prompt = prompt
        return prompt    

    def extract_keywords(self, text):
        # Extrahiere die Schlagworte aus dem Antworttext
        match = re.search(r'<final_list>(.*?)</final_list>', text, re.DOTALL)
        keywords = match.group(1).split("|")
        self.logger.info(keywords)
        quoted_keywords = [f'"{keyword.strip()}"' for keyword in keywords if keyword.strip()]
        self.final_list.emit(', '.join(quoted_keywords))
        return ', '.join(quoted_keywords)
    
    def handle_error(self, title: str, message: str):
        """Zeigt eine Fehlermeldung an"""
        QMessageBox.critical(self, title, message)
        #self.ai_processor.logger.error(f"{title}: {message}")

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

    def show_result(self):
        """Zeigt das Ergebnis an"""
        item = self.results_list.currentItem()
        if item:   
            keywords = self.extract_keywords(item.data(Qt.ItemDataRole.UserRole))
            self.keywords_extracted.emit(keywords)
            self.results_edit.setPlainText(keywords)
