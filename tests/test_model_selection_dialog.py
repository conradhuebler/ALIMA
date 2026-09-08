"""The settings dialog's model picker actually fetches models. Claude Generated.

``ModelSelectionDialog`` (Einstellungen → Task-Preferences → „Modell
hinzufügen") read ``self.config_manager`` in ``load_models``, but the attribute
was only assigned on the legacy branch of its constructor. Every caller in the
file passes a ``UnifiedProviderConfig``, so the attribute did not exist, the
resulting ``AttributeError`` was swallowed by a bare ``except``, and the model
combo showed the single invented entry "default" — for every provider.

The tests drive the dialog with an injected detection service and a synchronous
stand-in for the loader thread, so they neither hit the network nor depend on
the operator's configuration.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.config_models import UnifiedProvider, UnifiedProviderConfig


class _Detection:
    """Detection service stand-in: fixed model lists, records its calls."""

    def __init__(self, models=None):
        self._models = models or {}
        self.calls = []

    def get_available_models(self, provider, force_check=False):
        self.calls.append(provider)
        return list(self._models.get(provider, []))


class _SyncWorker:
    """ModelLoadWorker stand-in that delivers on ``start()``, in this thread."""

    def __init__(self, detection_service, providers, force=False):
        self._detection = detection_service
        self._provider = providers if isinstance(providers, str) else list(providers)[0]
        self.fetched = _Signal()
        self.finished = _Signal()

    def start(self):
        models = self._detection.get_available_models(self._provider)
        self.fetched.emit(self._provider, models)
        self.finished.emit()


class _Signal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


def _config(*names_enabled) -> UnifiedProviderConfig:
    return UnifiedProviderConfig(
        providers=[
            UnifiedProvider(name=name, provider_type="openai_compatible", enabled=True)
            for name in names_enabled
        ]
    )


class ModelSelectionDialogTest(unittest.TestCase):
    """The dialog is a real QWidget; hold the app and the dialogs for the class."""

    _kept: list = []

    def _dialog(self, detection, *enabled):
        from PyQt6.QtWidgets import QApplication

        from src.ui.comprehensive_settings_dialog import ModelSelectionDialog

        if not self._kept:
            self._kept.append(QApplication.instance() or QApplication(["alima-tests"]))
        with patch(
            "src.ui.comprehensive_settings_dialog.ModelLoadWorker", _SyncWorker
        ):
            dlg = ModelSelectionDialog(_config(*enabled), detection_service=detection)
        self._kept.append(dlg)
        return dlg

    def _models(self, dlg):
        return [dlg.model_combo.itemText(i) for i in range(dlg.model_combo.count())]

    def _pick(self, dlg, provider):
        with patch(
            "src.ui.comprehensive_settings_dialog.ModelLoadWorker", _SyncWorker
        ):
            dlg.provider_combo.setCurrentIndex(dlg.provider_combo.findData(provider))

    # -- the regression ------------------------------------------------

    def test_models_are_fetched_for_a_unified_config_dialog(self):
        detection = _Detection({"GWDG": ["gemma-4-31b-it", "apertus-70b"]})
        dlg = self._dialog(detection, "GWDG")
        self.assertEqual(detection.calls, ["GWDG"], "no model fetch happened at all")
        self.assertEqual(self._models(dlg), ["apertus-70b", "gemma-4-31b-it"])

    def test_no_invented_default_entry(self):
        # The old fallback wrote a literal "default" into the combo, and that
        # string was then stored as the model of a task preference.
        detection = _Detection({})
        dlg = self._dialog(detection, "GWDG")
        self.assertEqual(self._models(dlg), [])
        self.assertNotEqual(dlg.get_selected_model()[1], "default")

    def test_an_empty_list_says_why(self):
        dlg = self._dialog(_Detection({}), "GWDG")
        self._pick(dlg, "ollama")  # a common provider that is not enabled
        hint = dlg.model_combo.lineEdit().placeholderText()
        self.assertIn("aktiviert", hint)
        self._pick(dlg, "GWDG")  # enabled, but reported nothing
        self.assertIn("erreichbar", dlg.model_combo.lineEdit().placeholderText())

    # -- provider list -------------------------------------------------

    def test_enabled_providers_come_first(self):
        # The combo opens on index 0. With the four common names on top that was
        # a provider the machine may never have been set up for, so the model
        # list stayed empty and the dialog looked broken.
        detection = _Detection({"GWDG": ["m1"]})
        dlg = self._dialog(detection, "GWDG")
        self.assertEqual(dlg.provider_combo.currentData(), "GWDG")
        self.assertEqual(self._models(dlg), ["m1"])

    def test_an_unconfigured_provider_is_marked_but_keeps_its_name(self):
        dlg = self._dialog(_Detection({"GWDG": ["m1"]}), "GWDG")
        idx = dlg.provider_combo.findData("ollama")
        self.assertGreaterEqual(idx, 0, "the common providers must stay offerable")
        self.assertIn("nicht eingerichtet", dlg.provider_combo.itemText(idx))
        self._pick(dlg, "ollama")
        dlg.custom_model_input.setText("llama4:70b")
        # The marker must not leak into the stored provider name.
        self.assertEqual(dlg.get_selected_model(), ("ollama", "llama4:70b"))

    def test_an_enabled_provider_is_not_listed_twice(self):
        dlg = self._dialog(_Detection({"ollama": ["m1"]}), "ollama")
        names = [dlg.provider_combo.itemData(i) for i in range(dlg.provider_combo.count())]
        self.assertEqual(names.count("ollama"), 1)

    # -- late answers --------------------------------------------------

    def test_a_late_answer_for_an_older_provider_is_dropped(self):
        detection = _Detection({"GWDG": ["m1"]})
        dlg = self._dialog(detection, "GWDG")
        dlg._pending_provider = "Mistral"
        dlg._on_models_fetched("GWDG", ["stale"])
        self.assertNotIn("stale", self._models(dlg))


if __name__ == "__main__":
    unittest.main()
