"""pytest bootstrap. Claude Generated.

Importing the ``tests`` package runs ``tests/__init__.py``, which imports
QtWebEngineWidgets before any QApplication is created, creates the app with a
non-empty argv, and swaps in a lightweight WebLogView stub. conftest.py is
imported by pytest before any test module is collected, so this guarantees the
bootstrap runs first regardless of which test runs.
"""
import pytest

import tests  # noqa: F401  (import side effect: runs the bootstrap)


@pytest.fixture(autouse=True)
def _fresh_source_health():
    """Each test starts with no source marked down - Claude Generated.

    ``source_health`` is process-wide state; a test that simulates connection
    failures would otherwise leave a source skipped for the tests after it.
    """
    from src.core.search import source_health

    source_health.begin_run()
    yield
    source_health.begin_run()


_FREIBERG = ("DE-105", "105")


def _set_freiberg():
    from src.utils import institution

    institution.set_current(*_FREIBERG)


# Also for setUpClass, which runs before the function fixture below.
_set_freiberg()


@pytest.fixture(autouse=True)
def _freiberg_institution():
    """Each test runs with the institution of the UB Freiberg - Claude Generated.

    ``src/utils/institution`` holds the ISIL/ELN in use process-wide (set by
    ``ConfigManager`` on load/save). The tests written before the setting
    existed describe Freiberg behaviour (``$ADE-105``, ``udk_raw_de105``), so
    the test process runs with exactly those values, restored after every
    test; tests of the unset case set ``""`` themselves.
    """
    _set_freiberg()
    yield
    _set_freiberg()
