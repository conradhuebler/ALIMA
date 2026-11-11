"""A server database that cannot be opened must not read as a successful test.

``DatabaseManager`` falls back to SQLite when the Qt SQL driver for MySQL/MariaDB
does not load ("Driver not loaded", typically a missing system library of the
driver). The connection test then queried SQLite and reported
"SQLITE connection successful" for a MariaDB configuration. It now reports a
failure with the reason, and the status bar shows a chip while the fallback is
active. The driver is made unavailable here on purpose, so the tests do not depend
on the drivers of the machine. - Claude Generated
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PyQt6.QtSql import QSqlDatabase
from PyQt6.QtWidgets import QApplication

from src.core.database_manager import DatabaseManager
from src.utils.config_models import DatabaseConfig

_KEEP: list = []
_REAL_ADD_DATABASE = QSqlDatabase.addDatabase


def _without_server_drivers(driver, name, *args):
    """QMYSQL/QMARIADB as an unknown driver: the real 'Driver not loaded' error."""
    if driver in ("QMYSQL", "QMARIADB"):
        driver = "QNOSUCHDRIVER"
    return _REAL_ADD_DATABASE(driver, name, *args)


class FallbackIsAFailedTest(unittest.TestCase):
    """The connection is used and closed inside ``probe``: a failing assertion
    must not leave a Qt connection open (Qt then aborts the whole test run when
    the next test reuses the name). - Claude Generated"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.sqlite = str(Path(tmp.name) / "fallback.db")

    def probe(self, db_type, *, drivers=True, then_query=False):
        cfg = DatabaseConfig(db_type=db_type, sqlite_path=self.sqlite, host="db.example",
                             database="alima", username="u", password="p")
        patcher = (nullcontext() if drivers else
                   patch.object(QSqlDatabase, "addDatabase", side_effect=_without_server_drivers))
        with patcher:
            manager = DatabaseManager(cfg, f"fallback_test_{db_type}")
            try:
                ok, message = manager.test_connection()
                out = {"ok": ok, "message": message,
                       "requested": manager.requested_db_type,
                       "notice": manager.db_fallback_notice,
                       "reason": manager.fallback_reason}
                if then_query:
                    out["select"] = manager.fetch_scalar("SELECT 1")
                    out["config_type"] = manager.config.db_type
                return out
            finally:
                manager.close_connection()

    def test_a_server_database_without_driver_is_a_failure_with_the_reason(self):
        r = self.probe("mariadb", drivers=False)
        self.assertFalse(r["ok"])
        self.assertIn("MARIADB", r["message"])
        self.assertIn("SQLite", r["message"])
        self.assertIn("Driver not loaded", r["message"])
        self.assertIn("sqldrivers", r["message"])
        self.assertEqual(r["requested"], "mariadb")
        self.assertTrue(r["notice"])

    def test_the_same_for_mysql(self):
        r = self.probe("mysql", drivers=False)
        self.assertFalse(r["ok"])
        self.assertIn("MYSQL", r["message"])

    def test_sqlite_still_succeeds_and_says_so(self):
        r = self.probe("sqlite")
        self.assertTrue(r["ok"])
        self.assertIn("SQLITE connection successful", r["message"])
        self.assertIsNone(r["notice"])

    def test_the_fallback_itself_still_works_for_the_application(self):
        """The test fails, but the manager keeps serving SQLite as before."""
        r = self.probe("mariadb", drivers=False, then_query=True)
        self.assertEqual(r["select"], 1)
        self.assertEqual(r["config_type"], "sqlite")


class StatusBarShowsTheFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _KEEP.append(QApplication.instance() or QApplication(["alima-tests"]))

    def bar(self, notice):
        from src.ui.global_status_bar import GlobalStatusBar

        bar = GlobalStatusBar()
        _KEEP.append(bar)
        cache = SimpleNamespace(db_fallback_notice=notice, get_cache_stats=lambda: {})
        bar.set_services(MagicMock(), cache)
        return bar, cache

    def test_hidden_without_a_fallback(self):
        bar, _ = self.bar(None)
        self.assertFalse(bar.database_label.isVisibleTo(bar))

    def test_visible_with_the_notice_in_the_tooltip(self):
        bar, _ = self.bar("⚠️ MARIADB Treiber nicht ladbar. Verwende SQLite Fallback.")
        self.assertTrue(bar.database_label.isVisibleTo(bar))
        self.assertIn("SQLite", bar.database_label.text())
        self.assertIn("MARIADB Treiber nicht ladbar", bar.database_label.toolTip())

    def test_the_periodic_update_keeps_and_clears_it(self):
        bar, cache = self.bar("⚠️ MYSQL Treiber nicht ladbar.")
        bar.update_cache_status()
        self.assertTrue(bar.database_label.isVisibleTo(bar))
        cache.db_fallback_notice = None
        bar.update_cache_status()
        self.assertFalse(bar.database_label.isVisibleTo(bar))


if __name__ == "__main__":
    unittest.main()
