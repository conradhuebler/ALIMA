"""Reusable provider + model picker widget - Claude Generated.

A single, shared widget for choosing a provider and one of its models, used to
replace the per-surface combo-building that each tab/dialog reimplemented.

Design goals (provider-system cleanup, Phase 5):

* Model lists come from the shared TTL cache in ``ProviderDetectionService`` so
  every instance benefits from one detection pass (see Phase 4).
* Switching the provider refreshes the model list asynchronously with a loading
  placeholder — the UI never blocks on detection.
* The clean model name is stored in the item's ``UserRole`` and read from there;
  display decoration (e.g. a ⭐ for a task preference) never leaks into the value,
  which is what caused the "phantom model" bug in the old dialogs.
* Persistence is the host's job: connect to :pyattr:`selectionChanged` and call
  :pymeth:`get_selection` / :pymeth:`set_selection`.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

logger = logging.getLogger(__name__)

_LOADING_TEXT = "⏳ Lade Modelle…"
_MODEL_ROLE = Qt.ItemDataRole.UserRole
_PROVIDER_ROLE = Qt.ItemDataRole.UserRole


class _ModelLoadWorker(QThread):
    """Fetch one provider's models off the UI thread (cache-backed)."""

    fetched = pyqtSignal(str, list)  # provider, models

    def __init__(self, detection_service, provider: str, force: bool):
        super().__init__()
        self._detection_service = detection_service
        self._provider = provider
        self._force = force

    def run(self):  # noqa: D401 - QThread entry point
        try:
            models = self._detection_service.get_available_models(
                self._provider, force_check=self._force
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("ProviderModelSelector: model fetch failed for %s: %s",
                           self._provider, e)
            models = []
        self.fetched.emit(self._provider, list(models or []))


class ProviderModelSelector(QWidget):
    """Provider + model combo pair backed by the shared model cache."""

    # Emitted whenever the effective (provider, model) selection changes.
    selectionChanged = pyqtSignal(str, str)

    def __init__(self, detection_service=None, parent: Optional[QWidget] = None,
                 *, editable_model: bool = True, label: Optional[str] = None,
                 allow_empty: bool = False,
                 empty_provider_label: str = "(Use default)",
                 empty_model_label: str = "(Auto-select)"):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._detection_service = detection_service or self._default_detection_service()
        self._worker: Optional[_ModelLoadWorker] = None
        self._loading = False
        self._decorations: Dict[str, str] = {}
        # allow_empty adds a placeholder meaning "unset → fall back to a wider
        # default"; its value is "" so get_selection() returns an empty provider
        # /model. Used by the settings tab's pipeline/agentic default rows.
        self._allow_empty = allow_empty
        self._empty_provider_label = empty_provider_label
        self._empty_model_label = empty_model_label

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if label:
            layout.addWidget(QLabel(label))

        self.provider_combo = QComboBox()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(editable_model)
        if editable_model:
            self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        layout.addWidget(self.provider_combo, 1)
        layout.addWidget(self.model_combo, 2)

        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _default_detection_service():
        from ..utils.config_manager import ConfigManager
        return ConfigManager().get_provider_detection_service()

    # -- public API -----------------------------------------------------------

    def set_providers(self, providers: List[str]) -> None:
        """Populate the provider combo (preserving the current pick if possible)."""
        current = self._current_provider()
        with _blocked(self.provider_combo):
            self.provider_combo.clear()
            if self._allow_empty:
                self.provider_combo.addItem(self._empty_provider_label, "")
            for name in providers:
                self.provider_combo.addItem(name, name)
            if current:
                idx = self.provider_combo.findData(current)
                if idx >= 0:
                    self.provider_combo.setCurrentIndex(idx)
        self.refresh_models()

    def _current_provider(self) -> str:
        """Provider *value* (data), falling back to the visible text for items
        added without explicit data."""
        data = self.provider_combo.currentData(_PROVIDER_ROLE)
        return data if data is not None else self.provider_combo.currentText()

    def load_providers(self) -> None:
        """Populate providers from the detection service (enabled providers)."""
        try:
            providers = self._detection_service.get_available_providers()
        except Exception as e:
            self.logger.warning("ProviderModelSelector: provider list failed: %s", e)
            providers = []
        self.set_providers(providers)

    def set_selection(self, provider: str, model: str) -> None:
        """Select a provider and model (by value), loading the models first."""
        idx = self.provider_combo.findData(provider)
        with _blocked(self.provider_combo):
            if idx < 0 and provider:
                self.provider_combo.addItem(provider, provider)
                idx = self.provider_combo.count() - 1
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
        self.refresh_models(preselect_model=model)

    def get_selection(self) -> Tuple[str, str]:
        """Return the effective (provider, model).

        The model is read from the item's ``UserRole`` (the clean name) when an
        item is selected, falling back to the typed text — so display decoration
        never leaks into the value.
        """
        provider = (self._current_provider() or "").strip()
        if self._loading:
            # Mid-load the combo holds the loading placeholder; report the model
            # we're loading toward instead of leaking that placeholder as a value.
            return provider, (getattr(self, "_pending_preselect", "") or "").strip()
        data = self.model_combo.currentData(_MODEL_ROLE)
        model = data if data is not None else self.model_combo.currentText()
        return provider, (model or "").strip()

    def set_decorations(self, decorations: Dict[str, str]) -> None:
        """Optional per-model display prefixes (e.g. ``{"cogito:32b": "⭐ "}``)."""
        self._decorations = dict(decorations or {})

    def _model_values(self) -> List[str]:
        """Model values currently in the combo (clean names, excluding loading)."""
        out = []
        for i in range(self.model_combo.count()):
            data = self.model_combo.itemData(i, _MODEL_ROLE)
            value = data if data is not None else self.model_combo.itemText(i)
            if value and value != _LOADING_TEXT:
                out.append(value)
        return out

    def is_model_valid(self) -> bool:
        """True if the current model is among the loaded models (or list unknown)."""
        _provider, model = self.get_selection()
        if not model:
            return self._allow_empty  # empty is valid only when a placeholder exists
        known = self._model_values()
        return (not known) or (model in known)

    def refresh_models(self, *, force: bool = False, preselect_model: str = "") -> None:
        """Asynchronously (re)load the current provider's models."""
        provider = self._current_provider().strip()
        if not provider:
            self._populate_models([], preselect_model)
            return
        self._set_loading(True)
        self._pending_preselect = preselect_model or self.get_selection()[1]
        worker = _ModelLoadWorker(self._detection_service, provider, force)
        worker.fetched.connect(self._on_models_fetched)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    # -- internal slots -------------------------------------------------------

    def _on_provider_changed(self, _provider: str) -> None:
        self.refresh_models()

    def _on_model_changed(self, _text: str) -> None:
        self._apply_validation_style()
        provider, model = self.get_selection()
        self.selectionChanged.emit(provider, model)

    def _on_models_fetched(self, provider: str, models: List[str]) -> None:
        # Ignore late results for a provider the user already navigated away from.
        if provider != self._current_provider().strip():
            return
        self._set_loading(False)
        preselect = getattr(self, "_pending_preselect", "")
        self._populate_models(models, preselect)

    # -- population / styling -------------------------------------------------

    def _populate_models(self, models: List[str], preselect: str) -> None:
        with _blocked(self.model_combo):
            self.model_combo.clear()
            if self._allow_empty:
                self.model_combo.addItem(self._empty_model_label, "")
            for m in models:
                self.model_combo.addItem(f"{self._decorations.get(m, '')}{m}", m)
            if preselect:
                self._select_model(preselect)
            elif self.model_combo.count():
                self.model_combo.setCurrentIndex(0)  # placeholder (if any) or first model
        self._apply_validation_style()
        provider, model = self.get_selection()
        self.selectionChanged.emit(provider, model)

    def _select_model(self, model: str) -> None:
        for i in range(self.model_combo.count()):
            data = self.model_combo.itemData(i, _MODEL_ROLE)
            value = data if data is not None else self.model_combo.itemText(i)
            if value == model:
                self.model_combo.setCurrentIndex(i)
                return
        if self.model_combo.isEditable():
            self.model_combo.setCurrentText(model)  # typed / not-yet-detected model

    def _set_loading(self, loading: bool) -> None:
        self._loading = loading
        self.model_combo.setEnabled(not loading)
        if loading:
            with _blocked(self.model_combo):
                self.model_combo.clear()
                self.model_combo.addItem(_LOADING_TEXT)

    def _apply_validation_style(self) -> None:
        if self._loading:
            self.model_combo.setStyleSheet("")
            return
        ok = self.is_model_valid()
        self.model_combo.setStyleSheet("" if ok else "QComboBox { border: 1px solid #d32f2f; }")


class _blocked:
    """Context manager to suppress a widget's signals during bulk updates."""

    def __init__(self, widget: QWidget):
        self._widget = widget
        self._prev = False

    def __enter__(self):
        self._prev = self._widget.blockSignals(True)
        return self._widget

    def __exit__(self, *exc):
        self._widget.blockSignals(self._prev)
        return False
