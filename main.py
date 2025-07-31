#!/usr/bin/env python3

import sys
import os
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap
from src.ui.main_window import MainWindow
import logging


def setup_logging():
    """Konfiguriert das Logging für die Anwendung"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Ausgabe in die Konsole
            logging.FileHandler("gnd_fetcher.log"),  # Ausgabe in eine Datei
        ],
    )


def main():
    setup_logging()
    app = QApplication(sys.argv)
    app.setOrganizationName("TU Bergakademie Freiberg")
    app.setApplicationName("AlIma")
    app.setApplicationVersion("0.2")
    app.setStyle("Fusion")

    # Use direct file path instead of resource path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pixmap = QPixmap(os.path.join(current_dir, "alima.png"))

    # Check if the image was loaded successfully
    if pixmap.isNull():
        logging.error("Failed to load splash screen image")
    else:
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents()

    window = MainWindow()
    window.show()

    # Hide splash screen after main window is shown
    if "splash" in locals():
        splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
