"""Der Standard-Workflow ist aus dem Pipeline-Tab heraus setzbar. Claude Generated.

``SystemConfig.default_workflow`` steuert, welcher Workflow beim Start
vorausgewählt ist (GUI-Picker, CLI ohne ``--workflow``, Webapp). Erreichbar war
er nur über Einstellungen ▸ System; der ⭐ im Picker klebte fest auf
``alima_v51`` und zeigte damit bei abweichender Einstellung auf den falschen
Eintrag (UB Freiberg fährt ``alima_v51_105``).

Getestet wird der Mixin auf einem Stand-in statt auf einem echten
``PipelineTab`` (der die halbe Anwendung hochzieht); Combo und Button sind echte
Qt-Widgets, alles andere ist Mock.
"""
from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication, QComboBox, QPushButton

from src.ui._pipeline_tab_control import PipelineTabControlMixin

_WORKFLOWS = {
    "alima_v51": "5.1",
    "alima_v51_105": "5.1",
    "catalog_search": "4.0",
}
_LEGACY = {"old_pipeline": "3.0"}


class _WorkflowDirTestCase(unittest.TestCase):
    """Ein temporäres ``workflows/`` mit echten YAML-Dateien."""

    _kept: list = []

    def setUp(self):
        if not self._kept:
            self._kept.append(QApplication.instance() or QApplication(["alima-tests"]))
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        for stem, version in _WORKFLOWS.items():
            (base / f"{stem}.yaml").write_text(
                f'version: "{version}"\nsteps: []\n', encoding="utf-8"
            )
        (base / "legacy").mkdir()
        for stem, version in _LEGACY.items():
            (base / "legacy" / f"{stem}.yaml").write_text(
                f'version: "{version}"\nsteps: []\n', encoding="utf-8"
            )
        self._paths = patch(
            "src.core.agents.workflow_loader.DEFAULT_SEARCH_PATHS", [base]
        )
        self._paths.start()

    def tearDown(self):
        self._paths.stop()
        self._tmp.cleanup()

    def _tab(self, default_workflow: str):
        tab = MagicMock()
        tab.logger = logging.getLogger("test.pipeline_tab")
        tab.workflow_combo = QComboBox()
        tab.workflow_default_button = QPushButton()
        tab._WORKFLOW_ORDER = PipelineTabControlMixin._WORKFLOW_ORDER
        tab._load_default_workflow.return_value = default_workflow
        self._kept.append(tab.workflow_combo)
        self._kept.append(tab.workflow_default_button)
        return tab

    @staticmethod
    def _entries(combo: QComboBox):
        return [(combo.itemData(i), combo.itemText(i)) for i in range(combo.count())]

    @staticmethod
    def _starred(combo: QComboBox):
        return [
            combo.itemData(i)
            for i in range(combo.count())
            if combo.itemText(i).startswith("⭐")
        ]


class StarFollowsTheSetting(_WorkflowDirTestCase):
    def test_the_star_sits_on_the_configured_default(self):
        tab = self._tab("alima_v51_105")
        PipelineTabControlMixin._populate_workflow_combo(tab)

        self.assertEqual(self._starred(tab.workflow_combo), ["alima_v51_105"])
        self.assertEqual(tab.workflow_combo.currentData(), "alima_v51_105")

    def test_the_default_is_listed_first(self):
        tab = self._tab("alima_v51_105")
        PipelineTabControlMixin._populate_workflow_combo(tab)

        self.assertEqual(tab.workflow_combo.itemData(0), "alima_v51_105")

    def test_v51_default_keeps_the_previous_order(self):
        tab = self._tab("alima_v51")
        PipelineTabControlMixin._populate_workflow_combo(tab)

        self.assertEqual(tab.workflow_combo.itemData(0), "alima_v51")
        self.assertEqual(tab.workflow_combo.itemData(1), "__classic__")
        self.assertEqual(self._starred(tab.workflow_combo), ["alima_v51"])

    def test_classic_as_default_is_not_listed_twice(self):
        tab = self._tab("__classic__")
        PipelineTabControlMixin._populate_workflow_combo(tab)

        values = [data for data, _ in self._entries(tab.workflow_combo)]
        self.assertEqual(values.count("__classic__"), 1)
        self.assertEqual(self._starred(tab.workflow_combo), ["__classic__"])
        self.assertEqual(tab.workflow_combo.currentData(), "__classic__")

    def test_every_workflow_stays_in_the_list(self):
        tab = self._tab("alima_v51_105")
        PipelineTabControlMixin._populate_workflow_combo(tab)

        values = {data for data, _ in self._entries(tab.workflow_combo)}
        for stem in list(_WORKFLOWS) + list(_LEGACY) + ["__classic__"]:
            self.assertIn(stem, values)

    def test_an_unknown_default_falls_back_to_the_first_entry(self):
        tab = self._tab("workflow_das_es_nicht_gibt")
        PipelineTabControlMixin._populate_workflow_combo(tab)

        self.assertEqual(self._starred(tab.workflow_combo), [])
        self.assertEqual(tab.workflow_combo.currentIndex(), 0)


class ButtonState(_WorkflowDirTestCase):
    def test_button_is_off_when_the_pick_already_is_the_default(self):
        tab = self._tab("alima_v51_105")
        tab.workflow_combo.addItem("alima_v51_105 (v5.1)", "alima_v51_105")

        PipelineTabControlMixin._update_workflow_default_button(tab)

        self.assertFalse(tab.workflow_default_button.isEnabled())
        self.assertIn("bereits der Standard-Workflow", tab.workflow_default_button.toolTip())

    def test_button_is_on_for_any_other_pick(self):
        tab = self._tab("alima_v51")
        tab.workflow_combo.addItem("alima_v51_105 (v5.1)", "alima_v51_105")

        PipelineTabControlMixin._update_workflow_default_button(tab)

        self.assertTrue(tab.workflow_default_button.isEnabled())
        self.assertIn("alima_v51_105", tab.workflow_default_button.toolTip())

    def test_no_button_yet_is_not_an_error(self):
        tab = self._tab("alima_v51")
        tab.workflow_default_button = None
        PipelineTabControlMixin._update_workflow_default_button(tab)  # no raise


class Persisting(_WorkflowDirTestCase):
    def _config_manager(self, saved: bool = True):
        config = MagicMock()
        config.system_config.default_workflow = "alima_v51"
        manager = MagicMock()
        manager.load_config.return_value = config
        manager.save_config.return_value = saved
        return manager, config

    def test_the_click_writes_system_config_and_saves(self):
        tab = self._tab("alima_v51")
        tab.workflow_combo.addItem("alima_v51_105 (v5.1)", "alima_v51_105")
        manager, config = self._config_manager()

        with patch("src.utils.config_manager.ConfigManager", return_value=manager):
            PipelineTabControlMixin.set_current_workflow_as_default(tab)

        self.assertEqual(config.system_config.default_workflow, "alima_v51_105")
        manager.save_config.assert_called_once_with(config)
        # the ⭐ has to move, so the picker is rebuilt
        tab._populate_workflow_combo.assert_called_once()

    def test_a_failed_save_is_reported_and_changes_nothing_visible(self):
        tab = self._tab("alima_v51")
        tab.workflow_combo.addItem("alima_v51_105 (v5.1)", "alima_v51_105")
        manager, _config = self._config_manager(saved=False)

        with patch("src.utils.config_manager.ConfigManager", return_value=manager), \
             patch("src.ui._pipeline_tab_control.QMessageBox") as message_box:
            PipelineTabControlMixin.set_current_workflow_as_default(tab)

        message_box.warning.assert_called_once()
        tab._populate_workflow_combo.assert_not_called()

    def test_a_raising_config_manager_is_reported_not_swallowed(self):
        tab = self._tab("alima_v51")
        tab.workflow_combo.addItem("alima_v51_105 (v5.1)", "alima_v51_105")

        with patch("src.utils.config_manager.ConfigManager", side_effect=OSError("disk")), \
             patch("src.ui._pipeline_tab_control.QMessageBox") as message_box:
            PipelineTabControlMixin.set_current_workflow_as_default(tab)

        message_box.warning.assert_called_once()
        tab._populate_workflow_combo.assert_not_called()

    def test_a_separator_selection_writes_nothing(self):
        tab = self._tab("alima_v51")
        tab.workflow_combo.insertSeparator(0)
        manager, _config = self._config_manager()

        with patch("src.utils.config_manager.ConfigManager", return_value=manager):
            PipelineTabControlMixin.set_current_workflow_as_default(tab)

        manager.save_config.assert_not_called()


class ReadingTheSetting(_WorkflowDirTestCase):
    def test_it_reads_system_config(self):
        tab = self._tab("ignored")
        config = MagicMock()
        config.system_config.default_workflow = "alima_v51_105"
        manager = MagicMock()
        manager.load_config.return_value = config

        with patch("src.utils.config_manager.ConfigManager", return_value=manager):
            value = PipelineTabControlMixin._load_default_workflow(tab)

        self.assertEqual(value, "alima_v51_105")

    def test_an_empty_setting_falls_back_to_v51(self):
        tab = self._tab("ignored")
        config = MagicMock()
        config.system_config.default_workflow = ""
        manager = MagicMock()
        manager.load_config.return_value = config

        with patch("src.utils.config_manager.ConfigManager", return_value=manager):
            value = PipelineTabControlMixin._load_default_workflow(tab)

        self.assertEqual(value, "alima_v51")

    def test_an_unreadable_config_falls_back_to_v51(self):
        tab = self._tab("ignored")
        with patch("src.utils.config_manager.ConfigManager", side_effect=OSError("gone")):
            value = PipelineTabControlMixin._load_default_workflow(tab)

        self.assertEqual(value, "alima_v51")


if __name__ == "__main__":
    unittest.main()
