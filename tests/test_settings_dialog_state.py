"""The settings dialog must not lose configuration it has no widget for.

``_get_config_from_ui`` used to *rebuild* ``DatabaseConfig``, ``SystemConfig`` and
``UIConfig`` from scratch, so every field without a widget had to be hand-carried
in the constructor call. Five were not, and every "💾 Save & Close" silently reset
them:

* ``SystemConfig.url_fetch_allowlist`` — the SSRF exception list (``net_guard``)
* ``SystemConfig.url_fetch_max_bytes``
* ``UIConfig.ui_language`` — read by ``config_manager`` → ``i18n.set_language``
* ``UIConfig.ptheta_banner_seen`` — the one-time banner flag
* ``DatabaseConfig.gnd_local_path`` — the local GND authority DB

The dialog now mutates the ``config_to_edit`` copy instead, so a field without a
widget is simply left alone. These tests name each field individually, so a
regression says which one. - Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _qapp(keep: list):
    from PyQt6.QtWidgets import QApplication

    if not keep:
        keep.append(QApplication.instance() or QApplication(["alima-tests"]))
    return keep[0]


class _DialogHarness:
    """Builds a real ComprehensiveSettingsDialog over a config with non-default
    values everywhere, with the two sub-tabs stubbed out. - Claude Generated"""

    _kept: list = []

    def _dialog(self):
        from src.ui.comprehensive_settings_dialog import ComprehensiveSettingsDialog
        from src.utils.config_models import AlimaConfig

        cfg = AlimaConfig()
        cfg.system_config.url_fetch_allowlist = ["intranet.example"]
        cfg.system_config.url_fetch_max_bytes = 4242
        cfg.ui_config.ui_language = "en"
        cfg.ui_config.ptheta_banner_seen = True
        cfg.database_config.gnd_local_path = "/srv/alima/gnd_local.db"
        # Fields that DO have a widget, set away from their defaults so the test
        # also proves the dialog still writes what it owns.
        cfg.system_config.autosave_dir = "/srv/alima/results"
        cfg.database_config.sqlite_path = "/srv/alima/knowledge.db"

        _qapp(self._kept)
        cm = MagicMock()
        cm.load_config.return_value = cfg
        cm.get_config_info.return_value = {
            "os": "Linux", "project_config": "/x/config.json",
            "user_config": "/x/config.json", "system_config": "Not used",
        }
        # The two sub-tabs are real QWidgets (addTab needs one) with the handful
        # of methods the dialog calls on them.
        from PyQt6.QtCore import pyqtSignal
        from PyQt6.QtWidgets import QWidget

        class _StubTab(QWidget):
            config_changed = pyqtSignal()
            task_preferences_changed = pyqtSignal()

            def __init__(self, *a, **kw):
                super().__init__()

            def load(self, _config):
                pass

            def apply_to(self, _config):
                pass

            def _update_config_from_ui(self):
                pass

        with patch(
            "src.ui.comprehensive_settings_dialog.ConfigManager", return_value=cm
        ), patch(
            "src.ui.comprehensive_settings_dialog.UnifiedProviderTab", _StubTab
        ), patch(
            "src.ui.comprehensive_settings_dialog.PluginSettingsTab", _StubTab
        ):
            dlg = ComprehensiveSettingsDialog()
        self._kept.append(dlg)
        return dlg


class SettingsDialogPreservesUnwidgetedFieldsTest(_DialogHarness, unittest.TestCase):
    """Each of the five fields, named individually so a regression says which."""

    def _saved(self):
        dlg = self._dialog()
        return dlg._get_config_from_ui()

    def test_url_fetch_allowlist_survives(self):
        self.assertEqual(
            self._saved().system_config.url_fetch_allowlist, ["intranet.example"]
        )

    def test_url_fetch_max_bytes_survives(self):
        self.assertEqual(self._saved().system_config.url_fetch_max_bytes, 4242)

    def test_ui_language_survives(self):
        self.assertEqual(self._saved().ui_config.ui_language, "en")

    def test_ptheta_banner_seen_survives(self):
        self.assertTrue(self._saved().ui_config.ptheta_banner_seen)

    def test_gnd_local_path_survives(self):
        self.assertEqual(
            self._saved().database_config.gnd_local_path, "/srv/alima/gnd_local.db"
        )

    def test_widgeted_fields_are_still_written(self):
        """The counter-check: preservation must not mean "writes nothing"."""
        dlg = self._dialog()
        dlg.autosave_dir.setText("/tmp/other")
        dlg.sqlite_path.setText("/tmp/other.db")
        cfg = dlg._get_config_from_ui()
        self.assertEqual(cfg.system_config.autosave_dir, "/tmp/other")
        self.assertEqual(cfg.database_config.sqlite_path, "/tmp/other.db")


class DatabaseResetCompletesTest(_DialogHarness, unittest.TestCase):
    """"↺ Reset Database Settings" ran `setText` on a QComboBox and died mid-reset.

    The AttributeError landed after username/password were cleared but before
    charset, SSL, timeout and auto-create were touched, and the "Reset Complete"
    box never appeared — so the dialog was left in a half-reset state with no
    indication. - Claude Generated
    """

    def test_reset_completes_and_sets_every_field(self):
        from unittest.mock import patch

        dlg = self._dialog()
        dlg.mysql_username.setText("someone")
        dlg.mysql_charset.setCurrentText("latin1")
        dlg.connection_timeout.setValue(99)

        from PyQt6.QtWidgets import QMessageBox

        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ), patch.object(QMessageBox, "information") as done:
            dlg._reset_database_to_defaults()

        self.assertEqual(dlg.mysql_username.text(), "")
        self.assertEqual(dlg.mysql_charset.currentText(), "utf8mb4")
        self.assertEqual(dlg.connection_timeout.value(), 30)
        self.assertTrue(dlg.auto_create_tables.isChecked())
        done.assert_called_once()


if __name__ == "__main__":
    unittest.main()
