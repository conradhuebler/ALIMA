"""DatabaseConfig.ssl_disabled/charset had GUI+CLI controls but no reader.

get_connection()'s MySQL/MariaDB branches only ever passed
MYSQL_OPT_CONNECT_TIMEOUT through setConnectOptions - ssl_disabled and
charset were written to config.json and echoed back by the CLI's "database
show" command, but nothing downstream consulted them; a connection always
ran with whatever SSL mode and charset the server defaulted to. Confirmed by
reading libqsqlmysql.so's own strings, not just Qt's docs: the driver
recognizes MYSQL_OPT_SSL_MODE (values SSL_MODE_DISABLED/.../VERIFY_IDENTITY)
but has no connect-option for charset at all - it hard-codes "utf8"
internally, so charset needs a SET NAMES statement after connecting instead.

These tests cover the two pieces that don't need a live MySQL/MariaDB
server: the connect-options string builder and the charset-name validator
that guards the SET NAMES interpolation (SET NAMES takes no bind parameter -
a MySQL protocol limitation, not a Qt one). The SET NAMES execution itself
follows this codebase's existing policy of not unit-testing the MySQL/
MariaDB connection path (CLAUDE.md: tests use sqlite only) - it is reachable
and logs a warning on failure, same as the existing SQLite pragma block it
sits next to. - Claude Generated
"""

from __future__ import annotations

import unittest

from src.core.database_manager import DatabaseManager
from src.utils.config_models import DatabaseConfig


def _manager(**config_kwargs) -> DatabaseManager:
    """A DatabaseManager with just .config set - skips __init__'s
    QCoreApplication/Qt-plugin-path setup, which these pure methods don't
    touch."""
    mgr = DatabaseManager.__new__(DatabaseManager)
    mgr.config = DatabaseConfig(db_type="mysql", **config_kwargs)
    return mgr


class MysqlConnectOptionsTest(unittest.TestCase):
    def test_connect_timeout_is_always_present(self):
        mgr = _manager(connection_timeout=42)
        self.assertIn("MYSQL_OPT_CONNECT_TIMEOUT=42", mgr._mysql_connect_options())

    def test_ssl_disabled_adds_the_ssl_mode_option(self):
        mgr = _manager(ssl_disabled=True)
        self.assertIn("MYSQL_OPT_SSL_MODE=SSL_MODE_DISABLED", mgr._mysql_connect_options())

    def test_ssl_enabled_omits_the_ssl_mode_option(self):
        mgr = _manager(ssl_disabled=False)
        self.assertNotIn("SSL_MODE", mgr._mysql_connect_options())

    def test_options_are_semicolon_separated(self):
        mgr = _manager(connection_timeout=10, ssl_disabled=True)
        options = mgr._mysql_connect_options()
        self.assertEqual(
            options,
            "MYSQL_OPT_CONNECT_TIMEOUT=10;MYSQL_OPT_SSL_MODE=SSL_MODE_DISABLED",
        )


class CharsetNameValidationTest(unittest.TestCase):
    def test_accepts_real_charset_names(self):
        for name in ("utf8mb4", "utf8", "latin1", "utf8_general_ci"):
            self.assertTrue(DatabaseManager._is_valid_charset_name(name), name)

    def test_rejects_sql_injection_shaped_input(self):
        for bad in ("utf8'; DROP TABLE x; --", "utf8mb4\"", "utf8 mb4", ""):
            self.assertFalse(DatabaseManager._is_valid_charset_name(bad), bad)


if __name__ == "__main__":
    unittest.main()
