# Test-suite bootstrap. Claude Generated.
#
# Runs before any test module (package __init__ under unittest discover `-t .`
# and under pytest). Three jobs:
#   1. Import QtWebEngineWidgets before any QApplication exists — QWebEngineView
#      requires this ordering or it raises.
#   2. Create a QApplication with a non-empty argv up front so test modules that
#      do ``QApplication.instance() or QApplication([])`` reuse it.
#   3. Replace WebLogView with a lightweight QWidget stub. The real WebLogView
#      spins up a Chromium QWebEngineView, which is unstable when constructed
#      and torn down repeatedly in a headless unit-test run (no event loop).
#      The real widget is covered by its own smoke test and by running the app.
#   4. Point the personal-rules store at a throwaway file. Without this the
#      suite reads ``~/.config/alima/rules.yaml``: those rules are appended to
#      every prompt the agentic steps build, so an operator's own rules would
#      silently change what tests assert on. That is not hypothetical — it
#      broke test_e2e_smoke the first time real rules existed on this machine.
import os
import sys
import tempfile


def _isolate_user_rules() -> None:
    """Redirect the rules store to a temp path for the whole test session."""
    from src.core.user_rules import RULES_PATH_ENV

    os.environ.setdefault(
        RULES_PATH_ENV,
        os.path.join(tempfile.gettempdir(), "alima_test_rules_absent.yaml"),
    )


def _bootstrap() -> None:
    try:
        _isolate_user_rules()
    except Exception:
        pass  # the store falls back to the config dir; tests will say so

    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        from PyQt6.QtWidgets import QApplication
    except Exception:
        return  # PyQt6-WebEngine not installed in this environment

    if QApplication.instance() is None:
        QApplication(sys.argv or ["alima-tests"])

    try:
        from PyQt6.QtCore import QUrl, pyqtSignal
        from PyQt6.QtWidgets import QWidget
        import src.ui.web_log_view as _wlv

        class _StubWebLogView(QWidget):
            """No-Chromium stand-in with the WebLogView API surface."""

            link_clicked = pyqtSignal(QUrl)

            def __init__(self, *args, base_font_pt: int = 10, parent=None, **kwargs):
                super().__init__(parent)

            def append_block(self, html): ...
            def append_collapsible(self, block_id, summary, body, open_, kind=None): ...
            def update_collapsible(self, block_id, summary, body, kind=None): ...
            def open_assistant(self, header): ...
            def append_token(self, text): ...
            def finalize_assistant(self, html): ...
            def start_stream_line(self, prefix): ...
            def append_stream_token(self, text): ...
            def end_stream_line(self): ...
            def clear_log(self): ...
            def set_autoscroll(self, enabled): ...
            def scroll_to_bottom(self): ...
            def set_font_pt(self, pt): ...

        _wlv.WebLogView = _StubWebLogView
    except Exception:
        pass


_bootstrap()
