"""Die Testsuite läuft ohne Fenster auf dem Desktop. Claude Generated.

tests/__init__.py erzwingt die Qt-Plattform offscreen vor dem Anlegen der
QApplication, auch wenn die Sitzung QT_QPA_PLATFORM anders gesetzt hat.
Mit ALIMA_TEST_GUI=1 bleibt die echte Plattform (dann entfällt die Prüfung).
"""

import os
import unittest

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication


class HeadlessByDefaultTest(unittest.TestCase):
    @unittest.skipIf(os.environ.get("ALIMA_TEST_GUI"), "ALIMA_TEST_GUI: echte Plattform gewollt")
    def test_the_qt_platform_is_offscreen(self):
        # Ohne laufende Anwendung meldet Qt den Standard "xcb"; erst eine QApplication
        # liest die Plattform aus der Umgebung.
        app = QApplication.instance() or QApplication(["test"])
        self.assertEqual(QGuiApplication.platformName(), "offscreen")
        self.assertEqual(os.environ.get("QT_QPA_PLATFORM"), "offscreen")
        self.assertIsNotNone(app)

    def test_the_bootstrap_overrides_a_desktop_setting(self):
        """Quelltext-Wächter: Zuweisung, kein setdefault, sonst bliebe ein exportiertes
        wayland/xcb stehen und die Tests öffneten Fenster."""
        import pathlib

        src = pathlib.Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["QT_QPA_PLATFORM"] = "offscreen"', src)
        self.assertNotIn('setdefault("QT_QPA_PLATFORM"', src)


if __name__ == "__main__":
    unittest.main()
