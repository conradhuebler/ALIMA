"""GUI selector for search providers (F-3 P3) - Claude Generated.

A self-contained widget listing every registered ``SearchProvider`` with an
enable/disable checkbox, backed by :class:`SearchProviderConfig`. Disabled
providers are not exposed as search tools (see
``ToolRegistry._generated_search_tools``). Endpoints/tokens stay in the Catalog
tab; this only gates *which* sources are offered.

Embedded as a tab in ``ComprehensiveSettingsDialog``.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QCheckBox,
    QLabel,
)

from src.core.search import PROVIDER_REGISTRY, list_providers
from src.utils.config_models import SearchProviderConfig


class SearchProviderSelectorWidget(QWidget):
    """Checkbox list of registered search providers, backed by SearchProviderConfig."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checks = {}

        layout = QVBoxLayout(self)
        info = QLabel(
            "Aktivierte Suchquellen werden als Such-Tools angeboten (GUI, Pipeline, "
            "Agent). Deaktivierte Quellen erscheinen nicht in den Tool-Listen. "
            "Endpunkte/Tokens werden im Tab „Catalog“ konfiguriert."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        box = QGroupBox("Suchquellen (Search Provider)")
        box_layout = QVBoxLayout(box)
        for pid in list_providers():
            cls = PROVIDER_REGISTRY[pid]
            label = getattr(cls, "label", pid)
            caps = ", ".join(sorted(c.value for c in getattr(cls, "capabilities", set())))
            cb = QCheckBox(f"{label}  ({pid}) — {caps}")
            cb.setChecked(True)
            self._checks[pid] = cb
            box_layout.addWidget(cb)
        layout.addWidget(box)
        layout.addStretch(1)

    def load(self, cfg: SearchProviderConfig) -> None:
        """Set checkbox states from a SearchProviderConfig (absent → enabled)."""
        cfg = cfg or SearchProviderConfig()
        for pid, cb in self._checks.items():
            cb.setChecked(cfg.is_enabled(pid))

    def to_config(self) -> SearchProviderConfig:
        """Read the current checkbox states into a SearchProviderConfig."""
        return SearchProviderConfig(
            providers={pid: cb.isChecked() for pid, cb in self._checks.items()}
        )
