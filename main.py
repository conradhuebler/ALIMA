#!/usr/bin/env python

import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
import logging
from src.core.ai_processor import AIProcessor  # Korrigierter Import

def setup_logging():
    """Konfiguriert das Logging für die Anwendung"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Ausgabe in die Konsole
            logging.FileHandler('gnd_fetcher.log')  # Ausgabe in eine Datei
        ]
    )

def main():
    setup_logging()
    app = QApplication(sys.argv)

    # Debug: Zeige verfügbare Templates
    #ai_processor = AIProcessor()
    #templates = ai_processor.get_available_templates()
    #logging.debug(f"Verfügbare Templates: {templates}")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()