"""Fensterspion: meldet Widgets, die als eigenes Fenster erscheinen - Claude Generated.

Einschalten mit der Umgebungsvariable ``ALIMA_TRACE_WINDOWS=1`` beim Start der GUI.
Jedes Widget, das als Fenster der obersten Ebene gezeigt wird und keines der erwarteten
ist (Hauptfenster, Dialoge, Menüs, Auswahllisten von Comboboxen, Tooltips, Startbild),
wird mit Klasse, Titel/Text, Größe, Elternfenster und dem Python-Aufrufweg an der Stelle
des ``show()`` ins Log geschrieben (Logger ``alima.window_trace``, Stufe WARNING) und
einzeilig auf stderr ausgegeben.

Gedacht für Fenster, die man am Bildschirm sieht, in einer Messung ohne Bildschirm aber
nicht entstehen. Ohne die Variable wird nichts installiert.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Any

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow, QMenu, QSplashScreen, QWidget

logger = logging.getLogger("alima.window_trace")

#: Qt-interne Fenster, die zu jeder Combobox, jedem Tooltip usw. gehören.
EXPECTED_CLASSES = frozenset({
    "QComboBoxPrivateContainer", "QTipLabel", "QDockWidget", "QMenu", "QSplashScreen",
})
EXPECTED_TYPES = (Qt.WindowType.Popup, Qt.WindowType.ToolTip, Qt.WindowType.SplashScreen)


def is_expected(widget: QWidget) -> bool:
    """True für Fenster, die ihrer Art nach eigene Fenster sind."""
    if isinstance(widget, (QMainWindow, QDialog, QMenu, QSplashScreen)):
        return True
    if widget.metaObject().className() in EXPECTED_CLASSES:
        return True
    return widget.windowType() in EXPECTED_TYPES


def describe(widget: QWidget) -> str:
    text = ""
    for attr in ("text", "title"):
        try:
            text = str(getattr(widget, attr)())
            break
        except Exception:
            continue
    parent = widget.parent()
    return (
        f"{widget.metaObject().className()} obj={widget.objectName()!r} "
        f"titel={widget.windowTitle()!r} text={text[:40]!r} "
        f"größe={widget.width()}x{widget.height()} "
        f"eltern={type(parent).__name__ if parent is not None else None}"
    )


class WindowTracer(QObject):
    """Application-weiter Ereignisfilter für ``QEvent.Show`` auf Fenstern."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt-Name)
        try:
            if (event.type() == QEvent.Type.Show and isinstance(obj, QWidget)
                    and obj.isWindow() and not is_expected(obj)):
                stack = "".join(traceback.format_stack(limit=14)[:-1])
                logger.warning("FREMDES FENSTER: %s\n%s", describe(obj), stack)
                print(f"⚠ Fremdes Fenster: {describe(obj)} (Aufrufweg im Log)",
                      file=sys.stderr, flush=True)
        except Exception:  # der Spion darf die GUI nie stören
            logger.debug("window trace failed", exc_info=True)
        return False


def install(app: QApplication) -> WindowTracer:
    """Spion an die Anwendung hängen; Referenz bleibt an ``app`` hängen."""
    tracer = WindowTracer(app)
    app.installEventFilter(tracer)
    app._alima_window_tracer = tracer  # gegen Aufräumen durch die Speicherverwaltung
    logger.info("Fensterspion aktiv (ALIMA_TRACE_WINDOWS)")
    return tracer
