#!/usr/bin/env fish

# Projektname
set PROJECT_NAME "gnd_search"

# Hauptverzeichnis erstellen
mkdir -p $PROJECT_NAME

# Basis-Dateien erstellen
cd $PROJECT_NAME

# requirements.txt
echo "PyQt6>=6.0.0
requests>=2.28.0
python-dotenv>=0.19.0" > requirements.txt

# README.md
echo "# GND Search Tool

Ein Tool zur Suche und Analyse von GND-Schlagworten mit KI-Unterstützung.

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Verwendung

\`\`\`bash
python main.py
\`\`\`" > README.md

# main.py
echo 'import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()' > main.py

# Verzeichnisstruktur erstellen
mkdir -p src/{ui,core,utils} tests

# __init__.py Dateien erstellen
touch src/__init__.py
touch src/ui/__init__.py
touch src/core/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

# UI Module
echo 'from PyQt6.QtWidgets import QMainWindow
from .search_tab import SearchTab
from .abstract_tab import AbstractTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("GND Search Tool")
        # TODO: Implementierung' > src/ui/main_window.py

echo 'from PyQt6.QtWidgets import QWidget

class SearchTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: Implementierung' > src/ui/search_tab.py

echo 'from PyQt6.QtWidgets import QWidget

class AbstractTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # TODO: Implementierung' > src/ui/abstract_tab.py

echo 'from PyQt6.QtWidgets import QWidget

# Gemeinsame Widget-Komponenten
# TODO: Implementierung' > src/ui/widgets.py

# Core Module
echo 'import sqlite3
import json
from datetime import datetime, timedelta

class CacheManager:
    def __init__(self, db_path="search_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
    def create_tables(self):
        # TODO: Implementierung' > src/core/cache_manager.py

echo 'import requests
from collections import Counter

class SearchEngine:
    def __init__(self, cache_manager):
        self.cache = cache_manager
        
    def search(self, terms):
        # TODO: Implementierung' > src/core/search_engine.py

echo 'import requests
import os

class AIProcessor:
    def __init__(self):
        self.api_key = self._load_api_key()
        
    def process_abstract(self, abstract, keywords):
        # TODO: Implementierung
        
    def _load_api_key(self):
        # TODO: Implementierung' > src/core/ai_processor.py

# Utils Module
echo 'import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("GEMINI_API_KEY")
    CACHE_DB = "search_cache.db"
    # Weitere Konfigurationsoptionen' > src/utils/config.py

echo 'class TextProcessor:
    @staticmethod
    def clean_text(text):
        # TODO: Implementierung
        pass
        
    @staticmethod
    def extract_keywords(text):
        # TODO: Implementierung
        pass' > src/utils/text_processor.py

# Tests
echo 'import unittest

class TestSearch(unittest.TestCase):
    def setUp(self):
        # TODO: Test Setup
        pass
        
    def test_search(self):
        # TODO: Implementierung
        pass' > tests/test_search.py

echo 'import unittest

class TestCache(unittest.TestCase):
    def setUp(self):
        # TODO: Test Setup
        pass
        
    def test_cache(self):
        # TODO: Implementierung
        pass' > tests/test_cache.py

# Git-Ignore
echo "*.pyc
__pycache__/
.env
*.db
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/" > .gitignore

# Erstelle eine leere .env Datei für Konfiguration
echo "GEMINI_API_KEY=" > .env

# Mache main.py ausführbar
chmod +x main.py

echo "Projektstruktur wurde erstellt in: $PROJECT_NAME"

