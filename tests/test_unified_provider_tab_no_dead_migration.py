"""The removed legacy-migration path: a TypeError on every call, triggered
exactly when a config legitimately has zero providers (a state that is now
reachable - it used to be masked by config_manager's own reset-to-empty
fallback, removed in 3247a67).

``_migrate_from_legacy_config`` called
``UnifiedProviderConfig.from_legacy_config(self.config.unified_config,
self.unified_config)`` - two positional args for a method that takes one.
Worse: ``self.config.unified_config`` IS ``self.unified_config`` (both come
from ``config_to_edit.unified_config`` in
``comprehensive_settings_dialog.py`` - the same object, not a legacy dict
there is nothing left to migrate from by the time this UI tab exists). The
TypeError's fallback replaced ``self.unified_config`` with a brand-new,
detached ``UnifiedProviderConfig()`` - a provider added afterwards landed on
an object the dialog's save path never saw. Removed with its one call site
rather than patched: there is no legacy shape left to migrate at this layer.
- Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.utils.config_models import UnifiedProviderConfig


_kept: list = []


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


def _make_tab(providers):
    from PyQt6.QtWidgets import QWidget

    from src.ui.unified_provider_tab import UnifiedProviderTab

    _qapp()
    tab = UnifiedProviderTab.__new__(UnifiedProviderTab)
    QWidget.__init__(tab)
    _kept.append(tab)
    tab.logger = MagicMock()
    tab.unified_config = UnifiedProviderConfig(providers=providers)
    tab.config = MagicMock(unified_config=tab.unified_config)
    tab.config_changed = MagicMock()
    tab._start_background_model_fetch = MagicMock()
    tab._populate_provider_table = MagicMock()
    tab._populate_global_preferences = MagicMock()
    tab._populate_model_preferences = MagicMock()
    tab._refresh_effective_defaults = MagicMock()
    return tab


class NoDeadMigrationCodeRemainsTest(unittest.TestCase):
    def test_from_legacy_config_no_longer_exists(self):
        self.assertFalse(hasattr(UnifiedProviderConfig, "from_legacy_config"))

    def test_migrate_from_legacy_config_no_longer_exists(self):
        from src.ui.unified_provider_tab import UnifiedProviderTab

        self.assertFalse(hasattr(UnifiedProviderTab, "_migrate_from_legacy_config"))


class ZeroProvidersNoLongerDetachesTheConfigTest(unittest.TestCase):
    """The actual regression: with zero providers (now a legitimate,
    reachable state), unified_config must stay the SAME object identity
    the dialog's save path holds - not get silently replaced."""

    def test_loading_with_zero_providers_keeps_the_same_object_identity(self):
        from unittest.mock import patch

        tab = _make_tab(providers=[])
        original = tab.unified_config

        with patch(
            "src.ui.unified_provider_tab.ProviderDetectionService",
            return_value=MagicMock(get_available_providers=lambda: []),
        ):
            tab._load_configuration()

        self.assertIs(tab.unified_config, original)

    def test_loading_does_not_raise_with_zero_providers(self):
        """The TypeError used to be caught (so no crash reached the
        operator), but it still silently swapped the object - this pins the
        "does not raise" half separately from the identity check above."""
        from unittest.mock import patch

        tab = _make_tab(providers=[])
        with patch(
            "src.ui.unified_provider_tab.ProviderDetectionService",
            return_value=MagicMock(get_available_providers=lambda: []),
        ):
            tab._load_configuration()  # must not raise
        tab.logger.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
