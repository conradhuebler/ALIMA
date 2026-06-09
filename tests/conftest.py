"""pytest bootstrap. Claude Generated.

Importing the ``tests`` package runs ``tests/__init__.py``, which imports
QtWebEngineWidgets before any QApplication is created, creates the app with a
non-empty argv, and swaps in a lightweight WebLogView stub. conftest.py is
imported by pytest before any test module is collected, so this guarantees the
bootstrap runs first regardless of which test runs.
"""
import tests  # noqa: F401  (import side effect: runs the bootstrap)
