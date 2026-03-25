#!/usr/bin/env python3

import sys
import os
import argparse
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QSplashScreen, QDialog
from PyQt6.QtGui import QPixmap, QPalette, QColor
from src.ui.main_window import MainWindow
from src.ui.first_start_wizard import FirstStartWizard
from src.utils.logging_utils import setup_logging
from src.utils.config_manager import ConfigManager
import logging


def is_system_dark_mode(app: QApplication) -> bool:
    """Detect if the system is in dark mode by checking window background lightness — Claude Generated"""
    bg = app.palette().color(QPalette.ColorRole.Window)
    return bg.lightness() < 128


def _apply_dark_app_palette(app: QApplication):
    """Apply dark QPalette to the application — Claude Generated"""
    from src.ui.styles import DARK_COLORS
    palette = QPalette()
    bg = QColor(DARK_COLORS["background"])
    bg_light = QColor(DARK_COLORS["background_light"])
    text = QColor(DARK_COLORS["text"])
    border = QColor(DARK_COLORS["border"])
    primary = QColor(DARK_COLORS["primary"])

    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, bg_light)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(DARK_COLORS["background_dark"]))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Button, bg_light)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, primary)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(DARK_COLORS["background_light"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(DARK_COLORS["text_muted"]))
    app.setPalette(palette)


def _apply_light_app_palette(app: QApplication):
    """Restore default light QPalette — Claude Generated"""
    app.setPalette(app.style().standardPalette())


def main():
    # Parse command-line arguments - Claude Generated
    parser = argparse.ArgumentParser(description="ALIMA GUI - Automatic Library Indexing and Metadata Analysis")
    parser.add_argument("--wizard", action="store_true", help="Force first-start wizard even if config exists")
    parser.add_argument("--reset-setup", action="store_true", help="Reset setup flag and run wizard (same as --wizard)")
    args = parser.parse_args()

    # Setup centralized logging - Claude Generated
    # Default to level 1 (Normal) for GUI
    # TODO: Read from ~/.config/alima/config.json in future
    setup_logging(level=1, log_file="alima.log")
    app = QApplication(sys.argv)
    app.setOrganizationName("TU Bergakademie Freiberg")
    app.setApplicationName("AlIma")
    app.setApplicationVersion("0.2")
    app.setStyle("Fusion")

    # Setup Qt plugin paths for SQL drivers (MariaDB/MySQL, ODBC, etc.) - Claude Generated
    # On many distros Qt plugins are in the system Qt install, not the venv PyQt6
    from src.utils.qt_plugin_setup import setup_qt_plugin_paths
    setup_qt_plugin_paths()

    # Check for first-run setup or forced wizard - Claude Generated
    config_manager = ConfigManager()
    config = config_manager.load_config()

    # Show wizard if: first run OR forced via --wizard/--reset-setup flag
    force_wizard = args.wizard or args.reset_setup
    first_run_needed = not config.system_config.first_run_completed and not config.system_config.skip_first_run_check

    if first_run_needed or force_wizard:
        # Show first-start wizard
        wizard = FirstStartWizard()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            # User cancelled wizard
            logging.info("First-start wizard cancelled")
            sys.exit(0)

        # Reload config after wizard saves changes - Claude Generated
        config = config_manager.load_config(force_reload=True)

        # Reset singleton instances that might have cached old config
        from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
        UnifiedKnowledgeManager.reset()  # Properly closes DB and resets singleton
        logging.info(f"Config reloaded and singletons reset after wizard (db_type={config.database_config.db_type})")

    # Use direct file path instead of resource path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pixmap = QPixmap(os.path.join(current_dir, "alima.png"))

    # Check if the image was loaded successfully
    if pixmap.isNull():
        logging.warning("Failed to load splash screen image")
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
