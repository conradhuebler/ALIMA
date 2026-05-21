"""Tests for ChatWidget combo-persistence toggle (P-δ.4 E). Claude Generated.

Three cases:
1. ``test_toggle_off_does_not_persist`` — combo change with toggle off
   does NOT call ConfigManager.save_config.
2. ``test_toggle_on_persists`` — combo change with toggle on writes
   ChatConfig.default_provider/model + calls save_config once.
3. ``test_combo_auto_entry_skips_persist`` — combo "Auto" entry
   (data=None) skips persistence even when toggle is on.

Strategy: avoid instantiating the real ChatWidget. It builds a full
QWidget hierarchy which conflicts with the QCoreApplication-only setup
used by the rest of the test suite (see tests/test_state_bus.py).

Instead, call the persistence helpers as unbound methods on a minimal
stand-in object that carries only the attributes the helpers touch:
``model_combo``, ``persist_combo_toggle``, ``logger`` and
``_append_system_message``. This exercises 100% of the new code path
without the Qt-widget overhead.
"""
from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ui.chat_widget import ChatWidget
from src.utils.config_models import ChatConfig


def _make_stub(combo_data, toggle_on: bool) -> SimpleNamespace:
    """Build a minimal duck-typed object with the attributes
    ChatWidget._on_model_combo_changed / _persist_combo_to_chat_config
    access. Side-effects observable via the system-message list."""
    system_messages: list[str] = []
    stub = SimpleNamespace(
        model_combo=SimpleNamespace(currentData=lambda: combo_data),
        persist_combo_toggle=SimpleNamespace(isChecked=lambda: toggle_on),
        logger=logging.getLogger("test"),
        model_status_label=SimpleNamespace(setText=lambda _t: None),
        _refresh_model_status=lambda: None,
        _append_system_message=system_messages.append,
    )
    # Bind methods from ChatWidget — descriptor protocol gives us a
    # callable bound to ``stub``.
    stub._on_model_combo_changed = ChatWidget._on_model_combo_changed.__get__(stub)
    stub._persist_combo_to_chat_config = (
        ChatWidget._persist_combo_to_chat_config.__get__(stub)
    )
    stub.system_messages = system_messages
    return stub


class TestComboPersistence(unittest.TestCase):

    def test_toggle_off_does_not_persist(self):
        stub = _make_stub("ollama|cogito:32b", toggle_on=False)
        with patch("src.utils.config_manager.ConfigManager") as cm_class:
            stub._on_model_combo_changed(0)
            cm_class.return_value.save_config.assert_not_called()
        self.assertEqual(stub.system_messages, [])

    def test_toggle_on_persists(self):
        stub = _make_stub("ollama|cogito:32b", toggle_on=True)

        chat_cfg = ChatConfig()
        unified = SimpleNamespace(chat_config=chat_cfg)
        full_alima = object()  # sentinel for save_config arg-check

        with patch("src.utils.config_manager.ConfigManager") as cm_class:
            cm_instance = cm_class.return_value
            cm_instance.get_unified_config.return_value = unified
            cm_instance.load_config.return_value = full_alima
            cm_instance.save_config.return_value = True

            stub._on_model_combo_changed(0)

            cm_instance.save_config.assert_called_once_with(
                full_alima, preserve_unified=True
            )

        self.assertEqual(chat_cfg.default_provider, "ollama")
        self.assertEqual(chat_cfg.default_model, "cogito:32b")
        self.assertEqual(len(stub.system_messages), 1)
        self.assertIn("Chat-Default gespeichert", stub.system_messages[0])

    def test_combo_auto_entry_skips_persist(self):
        """Auto (data=None) should not attempt to persist."""
        stub = _make_stub(None, toggle_on=True)
        with patch("src.utils.config_manager.ConfigManager") as cm_class:
            stub._on_model_combo_changed(0)
            cm_class.return_value.save_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
