"""``enable_webcam_input`` is a data-protection switch, not a display option.

A house may forbid camera recordings. The switch therefore has to make the
camera unreachable, not just hide its button: hiding relied on locating the
frame's parent group box (``if parent:``), and if that ever failed the frame
stayed visible and clickable. The check now also sits at the point of effect,
and fails closed. - Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class WebcamStaysOffWhenSwitchedOffTest(unittest.TestCase):
    def _widget(self):
        import logging

        from src.ui.unified_input_widget import UnifiedInputWidget

        w = UnifiedInputWidget.__new__(UnifiedInputWidget)
        w.logger = logging.getLogger("test")
        return w

    def _with_setting(self, value):
        cm = MagicMock()
        cm.return_value.get_ui_config.return_value = MagicMock(
            enable_webcam_input=value
        )
        return patch("src.utils.config_manager.ConfigManager", cm)

    def test_the_dialog_is_not_opened_when_switched_off(self):
        w = self._widget()
        with self._with_setting(False), patch(
            "src.ui.webcam_capture_dialog.WebcamCaptureDialog"
        ) as dlg:
            w.capture_from_webcam()
        dlg.assert_not_called()

    def test_an_unreadable_config_keeps_the_camera_off(self):
        """Fails closed: a broken config must not enable a camera."""
        w = self._widget()
        cm = MagicMock(side_effect=OSError("kaputt"))
        with patch("src.utils.config_manager.ConfigManager", cm):
            self.assertFalse(w.webcam_enabled())

    def test_it_reports_enabled_when_switched_on(self):
        w = self._widget()
        with self._with_setting(True):
            self.assertTrue(w.webcam_enabled())

    def test_the_guard_sits_at_the_point_of_effect(self):
        """Not only on the frame's visibility, which depends on the widget tree."""
        import inspect

        from src.ui.unified_input_widget import UnifiedInputWidget

        src = inspect.getsource(UnifiedInputWidget.capture_from_webcam)
        self.assertIn("webcam_enabled", src)


if __name__ == "__main__":
    unittest.main()
