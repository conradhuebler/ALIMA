"""Nach dem Aufbau des Hauptfensters gibt es kein Widget ohne Elternfenster.

Ein Widget ohne Eltern ist in Qt ein eigenes Fenster, auch wenn es versteckt
ist. Ruft später irgendein Code ``show()`` oder ``setVisible(True)`` darauf,
erscheint es als kleines Fenster neben der Anwendung. Das betraf
Bedienelemente, die gebaut, aber in kein Layout gehängt wurden (DK-Schritt
der Pipeline, Statusleiste des Bild-Tabs).

Geprüft wird das echte ``MainWindow`` (mit Pipeline-Tab und allen anderen
Tabs) in einem Kindprozess: eigenes, leeres HOME (keine Nutzerkonfiguration),
``QT_QPA_PLATFORM=offscreen``, Netzverbindungen im Prozess nur zu localhost.

Ausnahmen:
* ``QMenu``: Menüs dürfen ohne Eltern existieren, sie erscheinen nur über
  ``exec()``/``popup()``.
* Fenster *mit* Elternwidget zählen nicht: die Menüs der Menüleiste und die
  Popup-Container der Comboboxen (``QComboBoxPrivateContainer``, in PyQt als
  ``QFrame`` sichtbar) gehören Qt und haben die Combobox als Eltern.

Claude Generated.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ALLOWED_MENU_CLASSES = {"QMenu"}


def _child(out_path: str) -> None:
    """Baut das Hauptfenster und schreibt den Befund als JSON. Claude Generated."""
    import gc
    import socket

    sys.path.insert(0, str(REPO))

    allowed_hosts = {"127.0.0.1", "localhost", "::1"}
    blocked: list = []
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_getaddrinfo = socket.getaddrinfo

    def _check(addr):
        if isinstance(addr, tuple) and addr and addr[0] not in allowed_hosts:
            blocked.append(str(addr))
            raise OSError(f"test: Netzzugriff gesperrt: {addr}")

    def _connect(sock, addr):
        _check(addr)
        return orig_connect(sock, addr)

    def _connect_ex(sock, addr):
        _check(addr)
        return orig_connect_ex(sock, addr)

    def _getaddrinfo(host, *args, **kwargs):
        if host is not None and host not in allowed_hosts:
            blocked.append(f"getaddrinfo {host}")
            raise socket.gaierror(f"test: Namensauflösung gesperrt: {host}")
        return orig_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.getaddrinfo = _getaddrinfo

    from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401  (vor QApplication)
    from PyQt6.QtWidgets import QApplication, QMenu

    app = QApplication(["alima"])

    from src.ui.main_window import MainWindow

    # Der einmalige Hinweisdialog ist modal und würde den Lauf blockieren.
    MainWindow._maybe_show_ptheta_banner = lambda self: None

    def _describe(w):
        text = ""
        for attr in ("text", "windowTitle"):
            getter = getattr(w, attr, None)
            if callable(getter):
                try:
                    text = str(getter() or "")
                except Exception:  # noqa: BLE001  (nur Beschreibung)
                    text = ""
                if text:
                    break
        owners = []
        for ref in gc.get_referrers(w):
            if isinstance(ref, dict):
                for owner in gc.get_referrers(ref):
                    if getattr(owner, "__dict__", None) is ref:
                        owners += [f"{type(owner).__name__}.{k}" for k, v in ref.items() if v is w]
        return {
            "class": w.metaObject().className(),
            "text": text[:60],
            "tooltip": w.toolTip()[:60],
            "owners": owners,
        }

    def _visible_windows(main):
        return [_describe(w) for w in QApplication.topLevelWidgets()
                if w is not main and w.isVisible()]

    window = MainWindow()
    window.show()
    for _ in range(30):
        app.processEvents()

    orphans = [
        _describe(w)
        for w in QApplication.topLevelWidgets()
        if w is not window
        and w.parentWidget() is None
        and not isinstance(w, QMenu)
    ]
    visible_after_start = _visible_windows(window)

    # So ruft es ImageAnalysisTab.start_analysis, sobald eine Bildanalyse beginnt.
    window.image_analysis_tab.progress_bar.setVisible(True)
    for _ in range(5):
        app.processEvents()
    visible_after_progress = _visible_windows(window)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "orphans": orphans,
            "visible_after_start": visible_after_start,
            "visible_after_progress": visible_after_progress,
            "blocked_network": blocked,
        }, fh, ensure_ascii=False, indent=1)
    sys.stdout.flush()
    os._exit(0)  # kein Chromium-Abbau im Kindprozess


def _qwebengine_works() -> bool:
    """Kann QWebEngineView hier überhaupt starten? (Kindprozess) Claude Generated."""
    code = (
        "from PyQt6.QtWebEngineWidgets import QWebEngineView;"
        "from PyQt6.QtWidgets import QApplication;"
        "app=QApplication([]);v=QWebEngineView();v.setHtml('<html></html>')"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    try:
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=60, env=env
        ).returncode == 0
    except Exception:  # noqa: BLE001  (Probe: jeder Fehler heißt "geht nicht")
        return False


class MainWindowHasNoOrphanWidgetsTest(unittest.TestCase):
    """Claude Generated."""

    result: dict = {}

    @classmethod
    def setUpClass(cls):
        try:
            import PyQt6.QtWebEngineWidgets  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"PyQt6-WebEngine fehlt: {exc}")

        cls._home = tempfile.TemporaryDirectory(prefix="alima-orphan-test-")
        home = cls._home.name
        out_path = os.path.join(home, "result.json")
        env = dict(os.environ)
        env.update({
            "HOME": home,
            "XDG_CONFIG_HOME": os.path.join(home, ".config"),
            "XDG_DATA_HOME": os.path.join(home, ".local", "share"),
            "XDG_CACHE_HOME": os.path.join(home, ".cache"),
            "ALIMA_RULES_FILE": os.path.join(home, "rules.yaml"),
            "QT_QPA_PLATFORM": "offscreen",
            "QTWEBENGINE_CHROMIUM_FLAGS": "--no-sandbox --disable-gpu",
        })
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", out_path],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            if not _qwebengine_works():
                raise unittest.SkipTest("QWebEngineView startet hier nicht (headless/CI)")
            tail = "\n".join((proc.stderr or "").splitlines()[-30:])
            raise AssertionError(
                f"Kindprozess fehlgeschlagen (rc={proc.returncode}):\n{tail}"
            )
        with open(out_path, encoding="utf-8") as fh:
            cls.result = json.load(fh)

    @classmethod
    def tearDownClass(cls):
        home = getattr(cls, "_home", None)
        if home is not None:
            home.cleanup()

    def _assert_none(self, key: str, what: str) -> None:
        found = self.result[key]
        if found:
            lines = [f"  {w['class']} {w['text']!r} {' '.join(w['owners'])}" for w in found]
            self.fail(f"{len(found)} {what}:\n" + "\n".join(lines))

    def test_no_widget_without_parent_after_startup(self):
        self._assert_none(
            "orphans", "Widgets ohne Elternfenster (in ein Layout hängen oder entfernen)"
        )

    def test_no_extra_window_is_visible_after_startup(self):
        self._assert_none("visible_after_start", "sichtbare Fenster neben dem Hauptfenster")

    def test_image_progress_bar_does_not_open_its_own_window(self):
        self._assert_none(
            "visible_after_progress", "sichtbare Fenster nach progress_bar.setVisible(True)"
        )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        _child(sys.argv[2])
    else:
        unittest.main()
