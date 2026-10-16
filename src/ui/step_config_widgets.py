#!/usr/bin/env python3
"""Per-step pipeline config widgets — Claude Generated.

:class:`SearchStepConfigWidget` — config for the GND search step. Used by the
``SearchStepConfigWidget`` tab of... nothing, now — it has no caller left.
Retained because the classic search-source choice (which GND providers the
classic pipeline's search step queries) has no other home yet; wire it back
in if the classic pipeline needs this configurable again.

``HybridStepConfigWidget`` (Smart/Advanced/Expert per-step provider+model
override) was removed Oct 2026 with ``PipelineConfigDialog`` - the classic
pipeline runs every step on the one pipeline default
(``PipelineConfig.create_from_provider_preferences``); there is no UI for a
per-step override anymore, by decision, not oversight. - Claude Generated
"""

from typing import Any, Dict

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QGroupBox, QLabel, QVBoxLayout, QWidget,
)


class SearchStepConfigWidget(QWidget):
    """Widget für die Konfiguration des GND-Suchschritts - Claude Generated"""

    def __init__(self, step_name: str, parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.step_id = "search"
        self.setup_ui()

    def setup_ui(self):
        """Setup der UI für Search-Konfiguration - Claude Generated"""
        layout = QVBoxLayout(self)

        # Step Name Header
        header_label = QLabel(self.step_name)
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)

        # Suggester Selection — one checkbox per enabled GND-capable provider,
        # built from the plugin system (same source pipeline_utils validates
        # against). Used to be three hardcoded boxes: lobid, swb, "Lokaler
        # Katalog". gnd_local and every own plugin were unreachable from the
        # classic dialog, and CLAUDE.md forbids hardcoded provider lists.
        # - Claude Generated
        suggester_group = QGroupBox("Suchquellen")
        self.suggester_layout = QVBoxLayout(suggester_group)
        self.source_checkboxes: Dict[str, QCheckBox] = {}
        self.refresh_sources()
        layout.addWidget(suggester_group)

        # Enable/Disable for this step
        self.enabled_checkbox = QCheckBox("Schritt aktivieren")
        self.enabled_checkbox.setChecked(True)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        self.suggester_layout.addWidget(self.enabled_checkbox)

        layout.addStretch()

    #: What the classic search falls back to when the config cannot be read.
    #: An *empty* list means the operator disabled every GND source and stays
    #: empty — conflating the two would search against an explicit disable.
    DEFAULT_SOURCES = ("lobid", "swb")

    def _gnd_source_ids(self):
        try:
            from ..core.search.factory import enabled_gnd_provider_ids

            ids = enabled_gnd_provider_ids(available_only=True)
            return ids if ids is not None else list(self.DEFAULT_SOURCES)
        except Exception:
            return list(self.DEFAULT_SOURCES)

    def refresh_sources(self):
        """Rebuild the source checkboxes from the plugin system.

        Follows the pattern SearchTab uses since WP-K5 (find_keywords:401): the
        checked state of known sources survives, new ones start at the default.
        - Claude Generated
        """
        previous = {pid: cb.isChecked() for pid, cb in self.source_checkboxes.items()}
        for cb in self.source_checkboxes.values():
            self.suggester_layout.removeWidget(cb)
            cb.deleteLater()
        self.source_checkboxes = {}

        for provider_id in self._gnd_source_ids():
            try:
                from ..core.search.registry import get_provider

                cls = get_provider(provider_id)
                label = getattr(cls, "label", provider_id)
                tooltip = cls.doc().description if hasattr(cls, "doc") else ""
            except Exception:
                label, tooltip = provider_id, ""
            cb = QCheckBox(label)
            cb.setChecked(previous.get(provider_id, provider_id in self.DEFAULT_SOURCES))
            if tooltip:
                cb.setToolTip(tooltip)
            # keep the "Schritt aktivieren" box last
            self.suggester_layout.insertWidget(len(self.source_checkboxes), cb)
            self.source_checkboxes[provider_id] = cb

    def on_enabled_changed(self, enabled: bool):
        """Enable/disable step configuration - Claude Generated"""
        for widget in self.findChildren(QWidget):
            if widget != self.enabled_checkbox:
                widget.setEnabled(enabled)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration - Claude Generated"""
        return {
            "step_id": self.step_id,
            "enabled": self.enabled_checkbox.isChecked(),
            "suggesters": [
                pid for pid, cb in self.source_checkboxes.items() if cb.isChecked()
            ],
        }

    def set_config(self, config: Dict[str, Any]):
        """Set configuration - Claude Generated

        A stored source that is no longer available (plugin disabled or removed)
        has no checkbox and is dropped — the same gate ``pipeline_utils`` applies
        at run time.
        """
        if "enabled" in config:
            self.enabled_checkbox.setChecked(config["enabled"])

        if "suggesters" in config:
            wanted = set(config["suggesters"] or [])
            for pid, cb in self.source_checkboxes.items():
                cb.setChecked(pid in wanted)


