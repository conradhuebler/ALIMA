#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QSplashScreen, QDialog
from PyQt6.QtGui import QPixmap
from src.ui.main_window import MainWindow
from src.ui.first_start_wizard import FirstStartWizard
from src.utils.logging_utils import setup_logging
from src.utils.config_manager import ConfigManager
import logging


def main():
    # Setup centralized logging - Claude Generated
    # Default to level 1 (Normal) for GUI
    # TODO: Read from ~/.config/alima/config.json in future
    setup_logging(level=1, log_file="alima.log")
    app = QApplication(sys.argv)
    app.setOrganizationName("TU Bergakademie Freiberg")
    app.setApplicationName("AlIma")
    app.setApplicationVersion("0.2")
    app.setStyle("Fusion")

    # Check for first-run setup - Claude Generated
    config_manager = ConfigManager()
    config = config_manager.load_config()

    if not config.system_config.first_run_completed and not config.system_config.skip_first_run_check:
        # Show first-start wizard
        wizard = FirstStartWizard()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            # User cancelled wizard
            logging.info("First-start wizard cancelled")
            sys.exit(0)

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

    # Start background provider check for instant UI with progressive loading - Claude Generated
    if hasattr(window, 'alima_manager') and hasattr(window.alima_manager, 'provider_status_service'):
        if window.alima_manager.provider_status_service:
            try:
                window.alima_manager.provider_status_service.refresh_all()
                logging.info("Background provider status check started")
            except Exception as e:
                logging.warning(f"Failed to start background provider check: {e}")
        else:
            logging.warning("ProviderStatusService not available")
    else:
        logging.warning("MainWindow or AlimaManager not properly initialized")

    # Hide splash screen after main window is shown
    if "splash" in locals():
        splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
