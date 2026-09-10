"""The live task model picker (Einstellungen ▸ Providers & Models ▸ Task Preferences).

``TaskModelSelectionDialog`` is what "➕ Add Model" opens. It had no test at all,
while its dead twin ``ModelSelectionDialog`` in ``comprehensive_settings_dialog``
did — the twin sat behind a tab that was never added to the dialog, so the tested
copy and the used copy had drifted apart. The twin is gone; these tests cover the
live one.

Checked when porting: the bug the old file documented (the model combo showing a
single invented "default" entry for every provider) does **not** exist here.
``_on_models_loaded`` inserts its placeholders with data ``None`` and
``get_selected_model`` returns ``currentData()``, so a placeholder can never be
stored as a model name. That property is asserted below rather than assumed.

The tests inject the detection service and run the loader synchronously, so they
neither hit the network nor depend on the operator's configuration.
- Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.utils.config_models import UnifiedProvider, UnifiedProviderConfig


class _Signal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class _SyncWorker:
    """ModelLoadWorker stand-in that delivers on ``start()``, in this thread."""

    def __init__(self, detection_service, providers, force=False):
        self._detection = detection_service
        self._provider = providers if isinstance(providers, str) else list(providers)[0]
        self.fetched = _Signal()
        self.finished = _Signal()

    def start(self):
        self.fetched.emit(self._provider, self._detection.get_available_models(self._provider))
        self.finished.emit()

    def deleteLater(self):
        pass


class _Detection:
    """ProviderDetectionService stand-in: fixed lists, records its calls."""

    def __init__(self, providers=None, models=None):
        self._providers = providers or []
        self._models = models or {}
        self.calls = []

    def __call__(self, *_a, **_kw):  # used as the patched class
        return self

    def get_available_providers(self):
        return list(self._providers)

    def get_available_models(self, provider, force_check=False):
        self.calls.append(provider)
        return list(self._models.get(provider, []))


class TaskModelSelectionDialogTest(unittest.TestCase):
    """The dialog is a real QWidget; hold the app and the dialogs for the class."""

    _kept: list = []

    def _dialog(self, detection, config_manager=None):
        from PyQt6.QtWidgets import QApplication

        from src.ui.provider_dialogs import TaskModelSelectionDialog

        if not self._kept:
            self._kept.append(QApplication.instance() or QApplication(["alima-tests"]))
        with patch("src.ui.provider_dialogs.ProviderDetectionService", detection), patch(
            "src.ui.provider_dialogs.ModelLoadWorker", _SyncWorker
        ):
            dlg = TaskModelSelectionDialog(config_manager, task_name="keywords")
        self._kept.append(dlg)
        return dlg

    def _pick(self, dlg, detection, provider):
        with patch("src.ui.provider_dialogs.ProviderDetectionService", detection), patch(
            "src.ui.provider_dialogs.ModelLoadWorker", _SyncWorker
        ):
            dlg.provider_combo.setCurrentText(provider)

    def _models(self, dlg):
        return [dlg.model_combo.itemText(i) for i in range(dlg.model_combo.count())]

    # -- TIER 1: live detection ------------------------------------------

    def test_models_are_fetched_for_the_picked_provider(self):
        det = _Detection(["ollama", "gwdg"], {"ollama": ["b", "a"], "gwdg": ["z"]})
        dlg = self._dialog(det)
        self._pick(dlg, det, "gwdg")
        self.assertIn("gwdg", det.calls)
        self.assertEqual(self._models(dlg), ["z"])

    def test_models_are_sorted(self):
        det = _Detection(["ollama"], {"ollama": ["Zeta", "alpha", "Mu"]})
        dlg = self._dialog(det)
        self.assertEqual(self._models(dlg), ["alpha", "Mu", "Zeta"])

    # -- the property the dead twin got wrong ----------------------------

    def test_a_placeholder_is_never_returned_as_a_model(self):
        """No models → the combo shows a notice, and nothing selectable is stored."""
        det = _Detection(["ollama"], {"ollama": []})
        dlg = self._dialog(det)
        self.assertEqual(self._models(dlg), ["No models available"])
        _provider, model = dlg.get_selected_model()
        self.assertIsNone(model)

    # -- TIER 2: cached fallback -----------------------------------------

    def test_cached_models_fill_in_when_detection_is_empty(self):
        """An offline provider still offers what the config remembers."""
        cfg = UnifiedProviderConfig(providers=[
            UnifiedProvider(
                name="ollama", provider_type="ollama", enabled=True,
                available_models=["cached-b", "cached-a"],
            )
        ])
        cm = MagicMock()
        cm.load_config.return_value = MagicMock(unified_config=cfg)
        det = _Detection(["ollama"], {"ollama": []})
        dlg = self._dialog(det, config_manager=cm)
        self.assertEqual(self._models(dlg), ["cached-a", "cached-b"])

    # -- the untested guard ----------------------------------------------

    def test_a_late_answer_for_an_older_provider_is_dropped(self):
        """Switching provider mid-flight must not repopulate from the old one."""
        det = _Detection(["ollama", "gwdg"], {"ollama": ["old"], "gwdg": ["new"]})
        dlg = self._dialog(det)
        self._pick(dlg, det, "gwdg")
        dlg._on_models_loaded("ollama", ["old"])
        self.assertEqual(self._models(dlg), ["new"])

    def test_no_providers_at_all_is_stated(self):
        det = _Detection([], {})
        dlg = self._dialog(det)
        self.assertEqual(dlg.provider_combo.currentText(), "No providers available")


if __name__ == "__main__":
    unittest.main()
