"""Providers in the settings: ticking one must not block the UI thread.

``config_changed`` fires on every tick/edit; the main window's handler used to
ping every enabled provider serially (3 s timeout each) on the UI thread, and
the status bar queried a model list synchronously. Both now stay off the UI
thread. Claude Generated.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock

from src.utils.config_models import (
    LOCAL_OLLAMA_PRESET_NAME,
    UnifiedProvider,
    UnifiedProviderConfig,
)

_kept: list = []


def _qapp():
    from PyQt6.QtWidgets import QApplication

    if not _kept:
        _kept.append(QApplication.instance() or QApplication(["alima-tests"]))
    return _kept[0]


class RefreshProviderStatusAsyncTest(unittest.TestCase):
    def _host(self, service):
        from src.ui._main_window_settings import MainWindowSettingsMixin

        host = MainWindowSettingsMixin.__new__(MainWindowSettingsMixin)
        host.logger = MagicMock()
        host.llm_service = service
        return host

    def test_returns_immediately_while_pings_run_in_background(self):
        release = threading.Event()
        started = threading.Event()
        service = MagicMock()
        service.refresh_all_provider_status.side_effect = lambda: (started.set(), release.wait(5))
        host = self._host(service)

        t0 = time.monotonic()
        host._refresh_provider_status_async()
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertTrue(started.wait(2))
        release.set()

    def test_overlapping_calls_start_only_one_run(self):
        release = threading.Event()
        service = MagicMock()
        service.refresh_all_provider_status.side_effect = lambda: release.wait(5)
        host = self._host(service)

        host._refresh_provider_status_async()
        host._refresh_provider_status_async()
        time.sleep(0.2)
        self.assertEqual(service.refresh_all_provider_status.call_count, 1)
        release.set()


class StatusBarNoNetworkTest(unittest.TestCase):
    def _bar(self):
        from src.ui.global_status_bar import GlobalStatusBar

        _qapp()
        bar = GlobalStatusBar.__new__(GlobalStatusBar)
        from PyQt6.QtWidgets import QLabel

        bar.provider_label = QLabel()
        bar.logger = MagicMock()
        bar.llm_service = MagicMock()
        bar.llm_service.get_available_providers.return_value = ["a"]
        bar.current_provider = "a"
        bar.current_model = "m1"
        _kept.append(bar)
        return bar

    def test_allow_network_false_queries_no_model_list_and_keeps_model(self):
        bar = self._bar()
        bar.update_provider_info(allow_network=False)
        bar.llm_service.get_available_models.assert_not_called()
        self.assertEqual(bar.current_model, "m1")

    def test_default_still_queries(self):
        bar = self._bar()
        bar.llm_service.get_available_models.return_value = ["x"]
        bar.current_model = ""
        bar.update_provider_info()
        bar.llm_service.get_available_models.assert_called_once()


class TickAdoptsPresetTest(unittest.TestCase):
    def _tab(self, providers):
        from PyQt6.QtWidgets import QWidget

        from src.ui.unified_provider_tab import UnifiedProviderTab

        _qapp()
        tab = UnifiedProviderTab.__new__(UnifiedProviderTab)
        QWidget.__init__(tab)
        _kept.append(tab)
        tab.logger = MagicMock()
        tab.unified_config = UnifiedProviderConfig(providers=providers)
        tab.config_changed = MagicMock()
        tab._persist_providers = MagicMock()
        # singleShot(0) würde sonst im Event-Loop eines späteren Tests feuern
        tab._after_toggle = MagicMock()
        return tab

    def test_ticking_the_suggestion_adds_an_enabled_local_ollama(self):
        tab = self._tab([])
        tab._on_provider_toggled(LOCAL_OLLAMA_PRESET_NAME, True)
        added = tab.unified_config.providers
        self.assertEqual([(p.name, p.provider_type, p.enabled) for p in added],
                         [(LOCAL_OLLAMA_PRESET_NAME, "ollama", True)])
        tab._persist_providers.assert_called_once()

    def test_ticking_existing_provider_only_flips_enabled(self):
        p = UnifiedProvider(name="X", provider_type="ollama", enabled=True)
        tab = self._tab([p])
        tab._on_provider_toggled("X", False)
        self.assertFalse(p.enabled)
        self.assertEqual(len(tab.unified_config.providers), 1)

    def test_unknown_name_changes_nothing(self):
        tab = self._tab([])
        tab._on_provider_toggled("nope", True)
        self.assertEqual(tab.unified_config.providers, [])
        tab._persist_providers.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class LoadSettingsAfterDialogTest(unittest.TestCase):
    """Nach „OK“ im Einstellungsdialog wendete load_settings() Theme und
    Schriftgröße immer vollständig neu an (gemessen 3,8 s im UI-Thread, auch ohne
    Änderung) und setzte Geometrie, Tab und Splitter zurück. Claude Generated."""

    def _host(self, dark_saved, font_saved, current_dark, current_font):
        from src.ui._main_window_settings import MainWindowSettingsMixin

        host = MainWindowSettingsMixin.__new__(MainWindowSettingsMixin)
        host.settings = MagicMock()
        host.settings.value.side_effect = lambda key, default=None: {
            "dark_mode": dark_saved, "font_size": font_saved, "geometry": b"geo",
        }.get(key, default)
        host._dark_mode = current_dark
        host.tabs = MagicMock()
        host.pipeline_tab = MagicMock()
        host.restoreGeometry = MagicMock()
        host.apply_theme = MagicMock()
        host.apply_font_size = MagicMock()
        host.config_manager = None
        self._font = current_font
        return host

    def _load(self, host, **kw):
        from unittest.mock import patch

        with patch("src.ui._main_window_settings.get_font_size", lambda: self._font):
            host.load_settings(**kw)

    def test_unchanged_theme_and_font_are_not_reapplied(self):
        host = self._host("true", 12, current_dark=True, current_font=12)
        self._load(host, only_changes=True)
        host.apply_theme.assert_not_called()
        host.apply_font_size.assert_not_called()

    def test_window_state_is_left_alone(self):
        host = self._host("true", 12, current_dark=True, current_font=12)
        self._load(host, only_changes=True)
        host.restoreGeometry.assert_not_called()
        host.tabs.setCurrentIndex.assert_not_called()
        host.pipeline_tab.restore_splitter_state.assert_not_called()

    def test_a_changed_theme_or_font_is_applied(self):
        host = self._host("false", 14, current_dark=True, current_font=12)
        self._load(host, only_changes=True)
        # Beide ändern sich: das Stylesheet wird nur einmal angewendet, und zwar
        # von apply_font_size; apply_theme setzt nur Palette und Modus.
        host.apply_theme.assert_called_once_with(False, restyle=False)
        host.apply_font_size.assert_called_once_with(14)

    def test_a_changed_theme_alone_restyles_itself(self):
        host = self._host("false", 12, current_dark=True, current_font=12)
        self._load(host, only_changes=True)
        host.apply_theme.assert_called_once_with(False, restyle=True)
        host.apply_font_size.assert_not_called()

    def test_startup_path_still_applies_everything(self):
        host = self._host("true", 12, current_dark=True, current_font=12)
        self._load(host)
        host.apply_theme.assert_called_once_with(True, restyle=False)
        host.apply_font_size.assert_called_once_with(12)
        host.restoreGeometry.assert_called_once()
        host.tabs.setCurrentIndex.assert_called_once_with(0)


class FontSizeDoesNotRewriteTheConfigTest(unittest.TestCase):
    """apply_font_size schrieb bei jedem Start die Config zurück, auch ohne
    Änderung. Claude Generated."""

    def _host(self, stored):
        from src.ui._main_window_settings import MainWindowSettingsMixin

        host = MainWindowSettingsMixin.__new__(MainWindowSettingsMixin)
        host.settings = MagicMock()
        host.logger = MagicMock()
        host.setStyleSheet = MagicMock()
        host._reapply_tab_stylesheets = MagicMock()
        cfg = MagicMock()
        cfg.ui_config.font_size = stored
        host.config_manager = MagicMock()
        host.config_manager.load_config.return_value = cfg
        return host

    def _apply(self, host, pt):
        from unittest.mock import patch

        _qapp()
        with patch("src.ui._main_window_settings.set_font_size"):
            host.apply_font_size(pt)

    def test_same_size_is_not_written(self):
        host = self._host(stored=11)
        self._apply(host, 11)
        host.config_manager.save_config.assert_not_called()

    def test_changed_size_is_written(self):
        host = self._host(stored=10)
        self._apply(host, 12)
        host.config_manager.save_config.assert_called_once()
