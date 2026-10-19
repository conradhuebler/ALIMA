"""The rules are reachable from the menu bar. Claude Generated.

``RulesDialog`` existed but only behind two entry points nobody finds without
being told: a 📌 button in the chat panel header and a button inside the
settings dialog. Personal rules decide what a run does, so they belong in the
Bearbeiten menu too.

The menu is built on a stand-in rather than a real ``MainWindow`` (which brings
up the whole application); only ``menuBar()`` is real, so the actions can be
inspected and triggered.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock


class RulesMenuEntryTest(unittest.TestCase):
    _kept: list = []

    def _menubar(self):
        from PyQt6.QtWidgets import QApplication, QMenuBar

        from src.ui._main_window_menu import MainWindowMenuMixin

        if not self._kept:
            self._kept.append(QApplication.instance() or QApplication(["alima-tests"]))
        window = MagicMock()
        menubar = QMenuBar()
        self._kept.append(menubar)
        window.menuBar.return_value = menubar
        MainWindowMenuMixin.create_menu_bar(window)
        return window, menubar

    @staticmethod
    def _action(menubar, needle: str):
        for menu_action in menubar.actions():
            menu = menu_action.menu()
            if menu is None:
                continue
            for action in menu.actions():
                if needle in action.text():
                    return action
        return None

    def test_the_edit_menu_offers_the_rules(self):
        _window, menubar = self._menubar()
        self.assertIsNotNone(
            self._action(menubar, "Zusatzregeln"),
            "no menu entry for the personal rules",
        )

    def test_triggering_it_opens_the_rules_dialog(self):
        window, menubar = self._menubar()
        action = self._action(menubar, "Zusatzregeln")
        action.trigger()
        window.show_rules_dialog.assert_called_once()

    def test_it_sits_in_the_edit_menu(self):
        _window, menubar = self._menubar()
        action = self._action(menubar, "Zusatzregeln")
        self.assertIsNotNone(action)
        self.assertEqual(action.parent().title(), "&Bearbeiten")

    def test_prompt_konfiguration_and_workflow_editor_are_gone(self):
        """Removed Oct 2026: "Workflow-Editor" was 100% redundant with
        settings/workflow_list.py's "Neu…" button (same WorkflowEditorDialog
        call); "Prompt-Konfiguration" (PromptEditorDialog) only ever touched
        prompts.json, never prompts.yaml - and prompts.yaml's "default" entry
        wins for every task the two files share, so an edit through it was
        silently invisible at runtime for almost every model."""
        _window, menubar = self._menubar()
        for needle in ("Prompt-Konfiguration", "Workflow-Editor"):
            with self.subTest(needle=needle):
                self.assertIsNone(self._action(menubar, needle))

    def test_the_handler_exists_on_the_window(self):
        # The mixin that carries the other config dialogs carries this one too.
        from src.ui._main_window_settings import MainWindowSettingsMixin

        self.assertTrue(callable(getattr(MainWindowSettingsMixin, "show_rules_dialog", None)))


if __name__ == "__main__":
    unittest.main()
