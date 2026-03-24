#!/usr/bin/env python3
"""
Qt Plugin Setup - Configure Qt plugin paths for SQL drivers
Claude Generated - Shared initialization for GUI, CLI, and Webapp

This module ensures Qt SQL drivers (QSQLITE, QMYSQL, QMARIADB) are found
by adding system Qt plugin paths to the application library paths.

Usage:
    from src.utils.qt_plugin_setup import setup_qt_plugin_paths
    setup_qt_plugin_paths()  # Call after QCoreApplication is created
"""

import sys
import os
import logging

logger = logging.getLogger(__name__)


def setup_qt_plugin_paths() -> list:
    """
    Detect and configure Qt plugin paths for SQL drivers.

    This function finds Qt SQL driver plugins (QSQLITE, QMYSQL, QMARIADB, etc.)
    by searching common system installation paths. On many Linux distributions,
    Qt plugins are installed in system directories, not in the Python venv.

    Call this AFTER creating QCoreApplication/QApplication.

    Returns:
        List of paths that were added to library paths

    Usage:
        # Call after creating QApplication/QCoreApplication
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)

        from src.utils.qt_plugin_setup import setup_qt_plugin_paths
        added_paths = setup_qt_plugin_paths()
    """
    # Skip if already configured via environment variable
    if os.environ.get("QT_PLUGIN_PATH"):
        logger.debug("QT_PLUGIN_PATH already set via environment")
        return []

    added_paths = []
    system_plugin_paths = []

    # Platform-specific system Qt plugin paths
    if sys.platform == "linux":
        # Use ldconfig to find Qt installation
        import subprocess
        try:
            result = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                text=True,
                timeout=5
            )
            for line in result.stdout.splitlines():
                if "libQt6Core.so" in line:
                    # e.g. "libQt6Core.so.6 => /usr/lib/libQt6Core.so.6"
                    parts = line.split("=>")
                    if len(parts) == 2:
                        lib_path = parts[1].strip()
                        # Extract base directory
                        lib_dir = lib_path.rsplit("/lib", 1)[0]
                        system_plugin_paths.extend([
                            f"{lib_dir}/lib/qt6/plugins",
                            f"{lib_dir}/lib64/qt6/plugins",
                            f"{lib_dir}/qt6/plugins",
                            f"{lib_dir}/qt6/plugins/sqldrivers",
                        ])
                        break
        except Exception as e:
            logger.debug(f"ldconfig lookup failed: {e}")

        # Common Linux distribution paths (Arch, Fedora, Debian/Ubuntu, openSUSE)
        system_plugin_paths.extend([
            "/usr/lib/qt6/plugins",
            "/usr/lib64/qt6/plugins",
            "/usr/lib/qt6/plugins/sqldrivers",
            "/usr/lib64/qt6/plugins/sqldrivers",
            "/usr/lib/x86_64-linux-gnu/qt6/plugins",
            "/usr/libexec/qt6/plugins",
            "/usr/lib/qt/plugins",  # Qt5 fallback
            "/usr/lib64/qt/plugins",  # Qt5 fallback
        ])

    elif sys.platform == "win32":
        # Qt on Windows is usually under C:/Qt
        program_files = os.environ.get("ProgramFiles", "C:/Program Files")
        qt_base = os.path.join(program_files, "Qt")
        if os.path.isdir(qt_base):
            for version in sorted(os.listdir(qt_base), reverse=True):
                version_path = os.path.join(qt_base, version)
                if os.path.isdir(version_path):
                    for arch in os.listdir(version_path):
                        plugins = os.path.join(version_path, arch, "plugins")
                        if os.path.isdir(plugins):
                            system_plugin_paths.append(plugins)
                            system_plugin_paths.append(os.path.join(plugins, "sqldrivers"))

    elif sys.platform == "darwin":
        # macOS: Qt plugins are typically in /usr/local/Cellar/qt or /opt/homebrew/opt/qt
        system_plugin_paths.extend([
            "/usr/local/lib/Qt6/plugins",
            "/usr/local/lib/Qt6/plugins/sqldrivers",
            "/opt/homebrew/lib/Qt6/plugins",
            "/opt/homebrew/lib/Qt6/plugins/sqldrivers",
            "/usr/local/Cellar/qt/6",
            "/Library/Qt/6/plugins",
        ])

    # Import Qt and add library paths
    try:
        from PyQt6.QtCore import QLibraryInfo, QCoreApplication
    except ImportError:
        logger.warning("PyQt6 not available - skipping Qt plugin setup")
        return added_paths

    # Get QApplication instance (must exist)
    app = QCoreApplication.instance()
    if app is None:
        logger.warning("QCoreApplication not found - call setup_qt_plugin_paths() after creating app")
        return added_paths

    # Start with Qt's default plugin path
    try:
        default_plugin_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        if os.path.isdir(default_plugin_path):
            if default_plugin_path not in app.libraryPaths():
                app.addLibraryPath(default_plugin_path)
                added_paths.append(default_plugin_path)
                logger.debug(f"Added Qt default plugin path: {default_plugin_path}")
    except Exception as e:
        logger.debug(f"Could not get default plugin path: {e}")

    # Add valid paths to library paths
    for path in system_plugin_paths:
        if os.path.isdir(path) and path not in app.libraryPaths():
            app.addLibraryPath(path)
            added_paths.append(path)
            logger.debug(f"Added system Qt plugin path: {path}")

    # Log available SQL drivers for debugging
    try:
        from PyQt6.QtSql import QSqlDatabase
        available_drivers = QSqlDatabase.drivers()
        if available_drivers:
            logger.info(f"Available SQL drivers: {', '.join(available_drivers)}")
        else:
            logger.warning("No SQL drivers found - database operations may fail")
    except ImportError:
        pass

    if added_paths:
        logger.info(f"Qt plugin paths configured: {len(added_paths)} paths added")

    return added_paths


def get_available_sql_drivers() -> list:
    """
    Get list of available SQL drivers.

    Returns:
        List of available SQL driver names (e.g., ['QSQLITE', 'QMYSQL', 'QMARIADB'])
        Empty list if QSqlDatabase is not available or QCoreApplication doesn't exist
    """
    try:
        from PyQt6.QtCore import QCoreApplication
        from PyQt6.QtSql import QSqlDatabase

        if QCoreApplication.instance() is None:
            return []

        return QSqlDatabase.drivers()
    except ImportError:
        return []
    except Exception:
        return []


def check_sql_driver_available(driver_name: str) -> bool:
    """
    Check if a specific SQL driver is available.

    Args:
        driver_name: Driver name (e.g., 'QSQLITE', 'QMYSQL', 'QMARIADB')

    Returns:
        True if driver is available, False otherwise
    """
    available = get_available_sql_drivers()
    return driver_name.upper() in [d.upper() for d in available]


if __name__ == "__main__":
    # Test the setup
    import sys
    from PyQt6.QtCore import QCoreApplication

    logging.basicConfig(level=logging.DEBUG)
    print("Testing Qt plugin path setup...")

    # Create app first
    app = QCoreApplication(sys.argv)

    # Setup paths
    added = setup_qt_plugin_paths()
    print(f"Added paths: {added}")

    # Check drivers
    drivers = get_available_sql_drivers()
    print(f"Available SQL drivers: {drivers}")

    for driver in ['QSQLITE', 'QMYSQL', 'QMARIADB', 'QODBC', 'QPSQL']:
        available = check_sql_driver_available(driver)
        print(f"  {driver}: {'✅' if available else '❌'}")