#!/usr/bin/env python3

import sys
import os
import argparse
import faulthandler
faulthandler.enable(all_threads=True)
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# QtWebEngineWidgets (used by the chat/log WebLogView) MUST be imported before
# the QApplication is created, otherwise QWebEngineView import fails at runtime.
# Import it here, as early as possible. - Claude Generated
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401  (import-order side effect)

from PyQt6.QtWidgets import QApplication, QSplashScreen, QDialog
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPalette, QColor, QIcon
from src.ui.main_window import MainWindow
from src.ui.first_start_wizard import FirstStartWizard
from src.utils.logging_utils import setup_logging
from src.utils.config_manager import ConfigManager
from src.core.state_bus import AlimaStateBus  # Eager import
import logging


def is_system_dark_mode(app: QApplication) -> bool:
    """Detect if the system is in dark mode by checking window background lightness — Claude Generated"""
    bg = app.palette().color(QPalette.ColorRole.Window)
    return bg.lightness() < 128


def _set_palette_roles(palette: QPalette, c: dict, dark: bool) -> None:
    """Fill the QPalette roles from a styles.py colour dict (COLORS or
    DARK_COLORS), so widgets the stylesheet does not cover (lists, menus,
    tooltips, Fusion frames) use the same tokens. - Claude Generated"""
    role = QPalette.ColorRole
    palette.setColor(role.Window, QColor(c["background_light"] if not dark else c["background"]))
    palette.setColor(role.WindowText, QColor(c["text"]))
    palette.setColor(role.Base, QColor(c["background"] if not dark else c["background_light"]))
    palette.setColor(role.AlternateBase, QColor(c["background_dark"] if dark else c["surface_raised"]))
    palette.setColor(role.Text, QColor(c["text"]))
    palette.setColor(role.BrightText, QColor("#ffffff"))
    palette.setColor(role.Button, QColor(c["surface_raised"]))
    palette.setColor(role.ButtonText, QColor(c["text"]))
    palette.setColor(role.Highlight, QColor(c["primary"]))
    palette.setColor(role.HighlightedText, QColor(c["on_primary"]))
    palette.setColor(role.ToolTipBase, QColor(c["background"]))
    palette.setColor(role.ToolTipText, QColor(c["text"]))
    palette.setColor(role.PlaceholderText, QColor(c["text_muted"]))
    palette.setColor(role.Link, QColor(c["secondary"]))
    palette.setColor(role.LinkVisited, QColor(c["secondary_pressed"]))
    # Fusion draws frames and bevels from these
    palette.setColor(role.Light, QColor(c["border_light"] if dark else "#ffffff"))
    palette.setColor(role.Midlight, QColor(c["surface_raised"]))
    palette.setColor(role.Mid, QColor(c["border_light"] if not dark else c["border"]))
    palette.setColor(role.Dark, QColor(c["border"] if not dark else c["background_dark"]))
    palette.setColor(role.Shadow, QColor("#000000" if dark else c["text_light"]))
    disabled = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(disabled, r, QColor(c["text_muted"]))


def _apply_dark_app_palette(app: QApplication):
    """Apply the dark (logo ground) QPalette to the application — Claude Generated"""
    from src.ui.styles import DARK_COLORS
    palette = QPalette()
    _set_palette_roles(palette, DARK_COLORS, dark=True)
    app.setPalette(palette)


def _apply_light_app_palette(app: QApplication):
    """Apply the warm light QPalette (same tokens as the light stylesheet) — Claude Generated"""
    from src.ui.styles import COLORS
    palette = app.style().standardPalette()
    _set_palette_roles(palette, COLORS, dark=False)
    app.setPalette(palette)


def _app_icon_path() -> str:
    """The logo mark (SVG tile) shared with the webapp favicon. - Claude Generated"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp", "static", "alima-icon.svg")


def _load_splash_pixmap(app: QApplication) -> QPixmap:
    """alima.png from the repository root, scaled to a modest splash size.
    Returns a null pixmap when the file is missing. - Claude Generated"""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pixmap = QPixmap(os.path.join(repo_root, "alima.png"))
    if pixmap.isNull():
        return pixmap
    dpr = app.devicePixelRatio() or 1.0
    side = int(320 * dpr)
    pixmap = pixmap.scaled(
        side, side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


def main():
    # Parse command-line arguments - Claude Generated
    parser = argparse.ArgumentParser(description="ALIMA GUI - AI-powered Library Indexing and Metadata Assignment")
    parser.add_argument("--wizard", action="store_true", help="Force first-start wizard even if config exists")
    parser.add_argument("--reset-setup", action="store_true", help="Reset setup flag and run wizard (same as --wizard)")
    args = parser.parse_args()

    # Bootstrap logging at Normal; re-applied from the config below, once it is
    # loaded (the config load itself must be able to log). - Claude Generated
    setup_logging(level=1, log_file="alima.log")
    app = QApplication(sys.argv)
    _ = AlimaStateBus()  # Force singleton creation on main thread before workers spawn
    app.setOrganizationName("TU Bergakademie Freiberg")
    app.setApplicationName("AlIma")
    app.setApplicationVersion("0.2")
    app.setStyle("Fusion")
    app_icon = QIcon(_app_icon_path())
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    # Setup Qt plugin paths for SQL drivers (MariaDB/MySQL, ODBC, etc.) - Claude Generated
    # On many distros Qt plugins are in the system Qt install, not the venv PyQt6
    from src.utils.qt_plugin_setup import setup_qt_plugin_paths
    setup_qt_plugin_paths()

    # Check for first-run setup or forced wizard - Claude Generated
    config_manager = ConfigManager()
    config = config_manager.load_config()

    # Apply the configured verbosity. Until Sep 2026 the Log-Level control in
    # Settings ▸ System wrote a value nothing read. - Claude Generated
    from src.utils.logging_utils import log_level_to_verbosity

    setup_logging(
        level=log_level_to_verbosity(config.system_config.log_level),
        log_file="alima.log",
    )

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

    # Splash: alima.png lives in the repository root, one level above src/
    pixmap = _load_splash_pixmap(app)

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
