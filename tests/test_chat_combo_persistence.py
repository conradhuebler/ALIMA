"""Tests for ChatWidget combo-persistence toggle (P-δ.4 E). Claude Generated.

Three cases:
1. ``test_toggle_off_does_not_persist`` — combo change with toggle off
   does NOT call ConfigManager.save_config.
2. ``test_toggle_on_persists`` — combo change with toggle on writes
   ChatConfig.default_provider/model + calls save_config once.
3. ``test_combo_auto_entry_skips_persist`` — the "Auto" selection
   (("", "")) skips persistence even when toggle is on.

Strategy: avoid instantiating the real ChatWidget. It builds a full
QWidget hierarchy which conflicts with the QCoreApplication-only setup
used by the rest of the test suite (see tests/test_state_bus.py).

Instead, call the persistence helpers as unbound methods on a minimal
stand-in object that carries only the attributes the helpers touch:
``provider_selector``, ``persist_combo_toggle``, ``logger`` and
``_append_system_message``. This exercises 100% of the new code path
without the Qt-widget overhead.

Phase 5: the combined "provider|model" combo was replaced by the shared
``ProviderModelSelector``; the helpers now read a structured
``get_selection() -> (provider, model)`` instead of splitting a string.
"""
from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# After P-δ.5a the combo-persist helpers live in PipelineChatPanel; the
# legacy ChatWidget has been retired. The methods kept their signatures.
from src.ui.pipeline_chat_panel import PipelineChatPanel as ChatWidget
from src.utils.config_models import ChatConfig


def _make_stub(selection, toggle_on: bool) -> SimpleNamespace:
    """Build a minimal duck-typed object with the attributes
    ChatWidget._on_model_selection_changed / _persist_combo_to_chat_config
    access. ``selection`` is the (provider, model) tuple the shared selector
    would return. Side-effects observable via the status-strip list (the
    persist echo moved out of the conversation stream, Chat-UX 7/9)."""
    system_messages: list[str] = []
    status_strip: list[str] = []
    stub = SimpleNamespace(
        provider_selector=SimpleNamespace(get_selection=lambda: selection),
        persist_combo_toggle=SimpleNamespace(isChecked=lambda: toggle_on),
        logger=logging.getLogger("test"),
        model_status_label=SimpleNamespace(setText=lambda _t: None),
        _refresh_model_status=lambda: None,
        _append_system_message=system_messages.append,
        set_status_strip=status_strip.append,
    )
    stub.status_strip_texts = status_strip
    # Bind methods from ChatWidget — descriptor protocol gives us a
    # callable bound to ``stub``.
    stub._on_model_selection_changed = (
        ChatWidget._on_model_selection_changed.__get__(stub)
    )
    stub._persist_combo_to_chat_config = (
        ChatWidget._persist_combo_to_chat_config.__get__(stub)
    )
    stub.system_messages = system_messages
    return stub


class TestComboPersistence(unittest.TestCase):

    def test_toggle_off_does_not_persist(self):
        stub = _make_stub(("ollama", "cogito:32b"), toggle_on=False)
        with patch("src.utils.config_manager.ConfigManager") as cm_class:
            stub._on_model_selection_changed("ollama", "cogito:32b")
            cm_class.return_value.save_config.assert_not_called()
        self.assertEqual(stub.system_messages, [])

    def test_toggle_on_persists(self):
        stub = _make_stub(("ollama", "cogito:32b"), toggle_on=True)

        # chat_config must live on the AlimaConfig returned by load_config —
        # that is the object the helper mutates AND saves (the unified config
        # has no chat_config; mutating it would be lost).
        chat_cfg = ChatConfig()
        full_alima = SimpleNamespace(chat_config=chat_cfg)

        with patch("src.utils.config_manager.ConfigManager") as cm_class:
            cm_instance = cm_class.return_value
            cm_instance.load_config.return_value = full_alima
            cm_instance.save_config.return_value = True

            stub._on_model_selection_changed("ollama", "cogito:32b")

            cm_instance.save_config.assert_called_once_with(
                full_alima, preserve_unified=True
            )

        # Mutation landed on the persisted object.
        self.assertEqual(chat_cfg.default_provider, "ollama")
        self.assertEqual(chat_cfg.default_model, "cogito:32b")
        # Echo lands in the status strip, not the conversation stream.
        self.assertEqual(stub.system_messages, [])
        self.assertEqual(len(stub.status_strip_texts), 1)
        self.assertIn("Chat-Default gespeichert", stub.status_strip_texts[0])

    def test_combo_auto_entry_skips_persist(self):
        """Auto selection (("", "")) should not attempt to persist."""
        stub = _make_stub(("", ""), toggle_on=True)
        with patch("src.utils.config_manager.ConfigManager") as cm_class:
            stub._on_model_selection_changed("", "")
            cm_class.return_value.save_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
