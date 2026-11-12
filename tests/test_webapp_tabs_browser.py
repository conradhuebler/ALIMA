"""Browser smoke test of the result tabs and of "Ergebnis laden…" - Claude Generated.

Runs the real page in headless Chromium against the app served in-process on
127.0.0.1 (every other request is aborted; ``/api/models`` and
``/api/workflows`` answer with fixed lists, the RVK check of the export is
patched). Checks what only a browser shows: the run-end does not switch the
tab but marks "Ergebnis"; the cards land in their tabs; the arrow keys move
the selection; a loaded file fills the tabs and says so; foreign markup in
the file is shown as text, never executed. Skipped without Playwright or
Chromium. All data invented (fixture of tests/test_geo_sachdb_result.py).
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HOME = {}


def setUpModule():
    # Playwright finds its browsers under ~/.cache; keep that path before HOME
    # moves to a temporary directory (no user configuration is read).
    _HOME["browsers"] = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    cache = os.path.expanduser("~/.cache/ms-playwright")
    if _HOME["browsers"] is None and os.path.isdir(cache):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = cache
    _HOME["old"] = os.environ.get("HOME")
    _HOME["tmp"] = tempfile.mkdtemp()
    os.environ["HOME"] = _HOME["tmp"]


def tearDownModule():
    if _HOME.get("old") is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _HOME["old"]
    if _HOME.get("browsers") is None:
        os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    shutil.rmtree(_HOME["tmp"], ignore_errors=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ResultTabsInTheBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest("Playwright ist nicht installiert")
        import uvicorn
        from src.webapp import app as appmod

        cls.appmod = appmod
        cls._rvk = patch("src.webapp.result_serialization.validate_rvk_notation",
                         lambda code: {"status": "standard", "is_standard": True,
                                       "canonical_code": code, "label": None, "message": None})
        cls._rvk.start()
        cls.port = _free_port()
        config = uvicorn.Config(appmod.app, host="127.0.0.1", port=cls.port, log_level="error")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 10
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.05)
        cls._pw = sync_playwright().start()
        try:
            cls.browser = cls._pw.chromium.launch()
        except Exception as exc:  # no Chromium build installed
            cls._pw.stop()
            cls._shutdown()
            raise unittest.SkipTest(f"Chromium für Playwright fehlt: {type(exc).__name__}")

    @classmethod
    def _shutdown(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls._rvk.stop()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls._pw.stop()
        cls._shutdown()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1366, "height": 768})
        self.page = self.context.new_page()
        self.errors, self.dialogs = [], []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("dialog", lambda d: (self.dialogs.append(d.message), d.dismiss()))
        base = f"http://127.0.0.1:{self.port}/"

        def route(r):
            url = r.request.url
            if not url.startswith(base):
                return r.abort()
            if "/api/models" in url or "/api/workflows" in url:
                return r.fulfill(status=200, content_type="application/json", body="[]")
            return r.continue_()

        self.page.route("**/*", route)
        self.sids = []

    def tearDown(self):
        self.context.close()
        for sid in self.sids:
            self.appmod.sessions.pop(sid, None)

    def _open(self, sid):
        self.sids.append(sid)
        self.page.goto(f"http://127.0.0.1:{self.port}/webapp?session={sid}")
        self.page.wait_for_selector("#result-tab-log[aria-selected='true']")

    def _selected(self):
        return self.page.evaluate(
            "Array.from(document.querySelectorAll('#result-tablist [role=tab]'))"
            ".filter(t => t.getAttribute('aria-selected') === 'true').map(t => t.dataset.resultTab)")

    def test_run_end_marks_the_result_tab_without_switching(self):
        from src.utils.pipeline_formatters import render_pipeline_result
        from src.webapp.render_bridge import _build_session_renderer
        from tests.test_webapp_result_load import fixture_state

        sid = "browser-run"
        session = self.appmod.Session(sid)
        session.status = "running"
        self.appmod.sessions[sid] = session
        renderer = _build_session_renderer(session)
        renderer.render_pipeline_log("Schritt 1", "info")
        self._open(sid)
        self.page.wait_for_function("window.alima.ws && window.alima.ws.readyState === 1")

        render_pipeline_result(renderer, fixture_state())
        session.results = {"working_title": "Testgestein und Lagerstätten"}
        session.status = "completed"
        self.page.wait_for_function(
            "document.getElementById('results-panel-title').textContent.includes('abgeschlossen')")

        self.assertEqual(self._selected(), ["log"])
        unread = self.page.evaluate(
            "['result', 'classification'].map(t => document.getElementById('result-tab-' + t)"
            ".classList.contains('has-unread'))")
        self.assertEqual(unread, [True, True])
        counts = self.page.evaluate(
            "['log-region', 'result-region', 'classification-region'].map(id =>"
            " [document.querySelectorAll('#' + id + ' .alima-result').length,"
            "  document.querySelectorAll('#' + id + ' .classification-card-title').length])")
        # final result also in the log (chat context), classification only in its tab
        self.assertEqual(counts, [[1, 0], [1, 0], [0, 1]])
        self.assertFalse(self.page.evaluate("document.getElementById('chat-input-area').hidden"))

        # Arrow keys move the selection; the chat input belongs to the log tab.
        self.page.focus("#result-tab-log")
        self.page.keyboard.press("ArrowRight")
        self.assertEqual(self._selected(), ["result"])
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "result-tab-result")
        self.assertTrue(self.page.evaluate("document.getElementById('chat-input-area').hidden"))
        self.assertFalse(self.page.evaluate(
            "document.getElementById('result-tab-result').classList.contains('has-unread')"))
        self.page.keyboard.press("ArrowLeft")
        self.assertEqual(self._selected(), ["log"])
        self.assertFalse(self.page.evaluate(
            "document.documentElement.scrollHeight > window.innerHeight + 1"))
        self.assertEqual(self.errors, [])

    def _write(self, data) -> str:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="ergebnis_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        self.addCleanup(os.unlink, path)
        return path

    def _load(self, path):
        self.page.set_input_files("#load-result-input", path)
        self.page.wait_for_function(
            "document.getElementById('results-panel-title').textContent.includes('geladen')"
            " || !document.getElementById('load-result-message').hidden")

    def test_loading_a_file_fills_the_tabs(self):
        from src.utils.pipeline_persistence import PipelineJsonManager
        from tests.test_webapp_result_load import EVIL, fixture_state

        self._open("browser-load")
        state = fixture_state()
        data = PipelineJsonManager.convert_sets_to_lists(dataclasses.asdict(state))
        path = self._write(data)
        self._load(path)
        self.assertEqual(self._selected(), ["result"])
        self.assertIn(os.path.basename(path), self.page.text_content("#results-origin"))
        self.assertIn("Geladen aus", self.page.text_content("#result-origin-line"))
        winibw = self.page.text_content("#result-region .alima-result__winibw")
        self.assertTrue(winibw.startswith("5550 "), winibw)
        self.assertEqual(self.page.evaluate("document.querySelectorAll('.step-node.is-done').length"), 0)

        # A reload restores the session as before (status + overview from the
        # export), still marked as loaded, on the tab chosen last.
        self.page.click("#result-tab-classification")
        self.page.reload()
        self.page.wait_for_function(
            "document.getElementById('results-panel-title').textContent.includes('geladen')")
        self.assertEqual(self._selected(), ["classification"])
        self.assertGreater(self.page.evaluate(
            "document.getElementById('results-summary').childElementCount"), 0)

        # Foreign markup in a second file: shown as text, nothing runs.
        data["working_title"] = "Titel " + EVIL
        data["core_keywords"] = ["Kern " + EVIL]
        data["initial_keywords"] = ["Initial " + EVIL]
        data["final_llm_analysis"]["extracted_gnd_keywords"] = ["GND " + EVIL]
        self._load(self._write(data))
        self.page.evaluate("document.getElementById('results-summary-details').open = true")
        injected = self.page.evaluate(
            "[window.__xss || 0, document.querySelectorAll('img[src=\"x\"], #log script').length]")
        self.assertEqual(injected, [0, 0])
        self.assertIn("<img", self.page.text_content("#results-origin"))
        self.assertIn("<img", self.page.text_content("#results-summary"))
        self.assertEqual(self.dialogs, [])
        self.assertEqual(self.errors, [])

    def test_a_broken_file_is_reported_in_the_header(self):
        self._open("browser-bad")
        path = self._write({"title": "kein Ergebnis"})
        self._load(path)
        self.assertIn("kein ALIMA-Ergebnis", self.page.text_content("#load-result-message"))
        self.assertTrue(self.page.evaluate("document.getElementById('results-panel').hidden"))


if __name__ == "__main__":
    unittest.main()
