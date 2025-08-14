from pathlib import Path
from datetime import datetime
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,  
                            QPushButton, QTextEdit, QFileDialog, QComboBox, QLabel,
                            QProgressBar, QMessageBox, QHBoxLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
import logging
import requests
import json
from typing import List, Dict
import torch

from src.llm.llm_interface import LLMInterface  # Ihre LLM Interface Klasse


class ChatHistory:
    def __init__(self):
        self.messages = []
        self.max_history = 10  # Maximale Anzahl der gespeicherten Nachrichten

    def add_message(self, role: str, content: str, documents: List = None):
        """Fügt eine neue Nachricht zur Historie hinzu"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "documents": documents
        }
        self.messages.append(message)
        if len(self.messages) > self.max_history:
            self.messages.pop(0)

    def get_context(self, max_messages: int = None) -> str:
        """Erstellt einen Kontext-String aus der Historie"""
        messages_to_use = self.messages[-max_messages:] if max_messages else self.messages
        context = ""
        for msg in messages_to_use:
            context += f"{msg['role']}: {msg['content']}\n"
        return context

    def clear(self):
        """Löscht die Historie"""
        self.messages.clear()

    def save_to_file(self, filepath: Path):
        """Speichert die Historie in einer JSON-Datei"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: Path):
        """Lädt die Historie aus einer JSON-Datei"""
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)

class IndexingWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, document_store, file_paths):
        super().__init__()
        self.document_store = document_store
        self.file_paths = file_paths

    def run(self):
        try:
            indexing_pipeline = TextIndexingPipeline(self.document_store)
            total_files = len(self.file_paths)

            for i, file_path in enumerate(self.file_paths):
                indexing_pipeline.run_batch(file_paths=[file_path])
                self.progress.emit(int((i + 1) * 100 / total_files))

            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Indexierungsfehler: {str(e)}")

class RAGWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Chat System")
        self.setGeometry(100, 100, 800, 600)

        self.document_store = InMemoryDocumentStore(use_bm25=True)
        self.retriever = None
        self.current_worker = None
        self.chat_history = ChatHistory()
        
        # LLM Interface initialisieren
        self.llm = LLMInterface()
        
        # Lade Modell-Empfehlungen
        recommendations_file = Path(__file__).parent / "chat_recommendations.json"
        self.setup_ui()
        self.check_gpu_status()

    def check_gpu_status(self):
        if torch.cuda.is_available():
            gpu_info = f"GPU verfügbar: {torch.cuda.get_device_name(0)}"
        else:
            gpu_info = "Keine GPU verfügbar - CPU-Modus aktiv"
        self.chat_display.append(gpu_info)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Provider und Modell-Auswahl
        model_layout = QHBoxLayout()
        
        # Provider Auswahl
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(self.llm.get_available_providers())
        print(self.llm.get_available_providers())
        self.provider_combo.currentTextChanged.connect(self.update_models)
        model_layout.addWidget(QLabel("Provider:"))
        model_layout.addWidget(self.provider_combo)
        
        # Modell Auswahl
        self.model_combo = QComboBox()
        model_layout.addWidget(QLabel("Modell:"))
        model_layout.addWidget(self.model_combo)
        
        layout.addLayout(model_layout)

        # Verzeichnis-Auswahl-Button
        self.select_dir_btn = QPushButton("Verzeichnis wählen")
        self.select_dir_btn.clicked.connect(self.select_directory)
        layout.addWidget(self.select_dir_btn)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Detail-Level Dropdown
        self.detail_level = QComboBox()
        self.detail_level.addItems(["minimum", "medium", "all"])
        layout.addWidget(self.detail_level)

        # Chat-Historie
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # Chat-Eingabefeld
        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText("Stelle deine Frage hier...")
        self.query_input.setMaximumHeight(100)
        layout.addWidget(self.query_input)

        # Button Layout
        button_layout = QHBoxLayout()
        
        # Sende-Button
        self.send_btn = QPushButton("Frage senden")
        self.send_btn.clicked.connect(self.process_query)
        self.send_btn.setEnabled(False)
        button_layout.addWidget(self.send_btn)

        # Abbrechen-Button
        self.cancel_btn = QPushButton("Abbrechen")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setVisible(False)
        button_layout.addWidget(self.cancel_btn)

        # Chat-Verwaltungs-Buttons
        self.clear_chat_btn = QPushButton("Chat löschen")
        self.clear_chat_btn.clicked.connect(self.clear_chat)
        button_layout.addWidget(self.clear_chat_btn)
        
        self.save_chat_btn = QPushButton("Chat speichern")
        self.save_chat_btn.clicked.connect(self.save_chat)
        button_layout.addWidget(self.save_chat_btn)
        
        self.load_chat_btn = QPushButton("Chat laden")
        self.load_chat_btn.clicked.connect(self.load_chat)
        button_layout.addWidget(self.load_chat_btn)
        
        layout.addLayout(button_layout)

        # Initial models update
  #      self.update_models(self.provider_combo.currentText())

    def update_models(self, provider: str):
        """Update available models when provider changes"""
        self.model_combo.clear()
        models = self.llm.get_available_models(provider)
        print(models)
        # Hole empfohlene Modelle für diesen Provider
  #      recommended = self.llm.get_recommended_models(provider)
        
        self.model_combo.addItems(models)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Wähle Dokumentenverzeichnis")
        if dir_path:
            self.index_documents(dir_path)

    def index_documents(self, dir_path):
        try:
            files_to_index = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                            if f.endswith(('.txt', '.pdf', '.doc', '.docx'))]

            if not files_to_index:
                self.show_error("Keine unterstützten Dokumente gefunden!")
                return

            self.progress_bar.setVisible(True)
            self.select_dir_btn.setEnabled(False)
            self.send_btn.setEnabled(False)

            self.indexing_worker = IndexingWorker(self.document_store, files_to_index)
            self.indexing_worker.progress.connect(self.progress_bar.setValue)
            self.indexing_worker.finished.connect(self.indexing_finished)
            self.indexing_worker.error.connect(self.show_error)
            self.indexing_worker.start()

        except Exception as e:
            self.show_error(f"Fehler beim Indexieren: {str(e)}")

    def indexing_finished(self):
        self.retriever = BM25Retriever(document_store=self.document_store)
        self.progress_bar.setVisible(False)
        self.select_dir_btn.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.chat_display.append("Dokumente erfolgreich indexiert!")

    def format_documents(self, documents: List) -> str:
        """Formatiert die Dokumente für den Prompt"""
        formatted = ""
        for i, doc in enumerate(documents, 1):
            formatted += f"\nDokument {i}:\n{doc.content[:1000]}\n"
        return formatted

    def process_query(self):
        if not self.retriever:
            self.show_error("Bitte zuerst Dokumente indexieren!")
            return

        query = self.query_input.toPlainText().strip()
        if not query:
            return

        # Füge Benutzeranfrage zur Historie hinzu
        self.chat_history.add_message("user", query)
        self.update_chat_display()

        retrieved_docs = self.retriever.retrieve(query, top_k=5)

        # Erstelle Prompt mit Chat-Historie
        context = self.chat_history.get_context(max_messages=5)
        prompt = f"""Kontext der Konversation:
{context}

Basierend auf den folgenden Dokumenten und dem Kontext, beantworte bitte die letzte Frage.
Wenn du die Antwort nicht in den Dokumenten findest, sage das ehrlich.

Verfügbare Dokumente:
{self.format_documents(retrieved_docs)}

Antworte bitte ausführlich und präzise.
"""

        self.progress_bar.setVisible(True)
        self.send_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)

        try:
            provider = self.provider_combo.currentText()
            model = self.model_combo.currentText().strip()
            
            response = self.llm.generate_response(
                provider=provider,
                model=model,
                prompt=prompt
            )
            
            # Füge Antwort zur Historie hinzu
            self.chat_history.add_message("assistant", response, documents=retrieved_docs)
            self.update_chat_display()
            
        except Exception as e:
            self.show_error(f"Fehler bei der Verarbeitung: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.send_btn.setEnabled(True)
            self.cancel_btn.setVisible(False)
            self.query_input.clear()

    def cancel_processing(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            self.current_worker.wait()
            self.progress_bar.setVisible(False)
            self.send_btn.setEnabled(True)
            self.cancel_btn.setVisible(False)
            self.chat_display.append("\n[Abgebrochen]")

    def update_chat_display(self):
        """Aktualisiert die Chat-Anzeige"""
        self.chat_display.clear()
        for message in self.chat_history.messages:
            role = message["role"]
            content = message["content"]
            
            # Formatierung je nach Rolle
            if role == "user":
                self.chat_display.append(f"Sie: {content}\n")
            else:
                self.chat_display.append(f"Assistant: {content}\n")
                
            # Zeige Dokumente wenn gewünscht
            if self.detail_level.currentText() != "minimum" and message.get("documents"):
                self.chat_display.append("\nVerwendete Dokumente:")
                for doc in message["documents"][:2]:
                    self.chat_display.append(f"- {doc.content[:200]}...\n")
            
            self.chat_display.append("-" * 50 + "\n")

    def clear_chat(self):
        """Löscht den Chat-Verlauf"""
        reply = QMessageBox.question(
            self, 
            "Chat löschen",
            "Möchten Sie wirklich den gesamten Chat-Verlauf löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_history.clear()
            self.update_chat_display()

    def save_chat(self):
        """Speichert den Chat-Verlauf"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Chat speichern",
            "",
            "JSON Dateien (*.json)"
        )
        if filepath:
            self.chat_history.save_to_file(Path(filepath))

    def load_chat(self):
        """Lädt einen gespeicherten Chat-Verlauf"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Chat laden",
            "",
            "JSON Dateien (*.json)"
        )
        if filepath:
            self.chat_history.load_from_file(Path(filepath))
            self.update_chat_display()

    def show_error(self, error_message):
        """Zeigt eine Fehlermeldung an"""
        self.progress_bar.setVisible(False)
        self.send_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        QMessageBox.critical(self, "Fehler", error_message)

def main():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    app = QApplication(sys.argv)
    window = RAGWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
