"""GUI tests for the webindex crawl action (button + worker) - Claude Generated.

Headless: ``QT_QPA_PLATFORM=offscreen`` + a shared QApplication. The worker is
driven synchronously (``run()`` called directly on the main thread) with an
injected ``fetch_func`` so no network happens.
"""

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication, QPushButton
    from src.ui.plugin_settings_tab import _CategoryPanel, _TYPE_ACTIONS, _ensure_categories
    from src.ui.webindex_crawl import WebIndexCrawlWorker, _build_crawl_kwargs
    from src.utils.config_models import PluginInstanceConfig
    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    IMPORT_ERROR = exc

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "webindex", "site")
BASE = "https://test.local/"


def _fixture_resp(name):
    class _R:
        def __init__(self):
            with open(os.path.join(_FIXTURES, name), "rb") as f:
                self.content = f.read()
            self.headers = {"Content-Type": "text/html"}
            self.status_code = 200

    return _R()


def _fake_fetch(url, *, user_agent, timeout):
    if url == BASE:
        return _fixture_resp("index.html")
    raise RuntimeError(f"404 {url}")


_qapp = None


def _ensure_qapp():
    global _qapp
    if _qapp is None and IMPORT_ERROR is None:
        _qapp = QApplication.instance() or QApplication([])
    return _qapp


@unittest.skipIf(IMPORT_ERROR is not None, f"PyQt6 stack unavailable: {IMPORT_ERROR}")
class WebIndexCrawlUITest(unittest.TestCase):
    def setUp(self):
        _ensure_qapp()
        _ensure_categories()  # registers the lookup category adapter
        self._tmp = tempfile.mkdtemp()

    def test_action_registered(self):
        acts = _TYPE_ACTIONS.get(("lookup", "webindex"), [])
        self.assertTrue(any("indizieren" in lbl.lower() for lbl, _ in acts))

    def test_form_renders_crawl_button(self):
        panel = _CategoryPanel("lookup")
        inst = PluginInstanceConfig(
            "webindex", "lookup", "webindex", label="Webindex",
            enabled=True, is_primary=True, settings={"base_url": BASE},
        )
        panel.load([inst])
        panel.list.setCurrentRow(0)
        # _build_form is triggered by row change; find the crawl button in the form.
        buttons = panel.form_box.findChildren(QPushButton)
        texts = [b.text() for b in buttons]
        self.assertTrue(any("indizieren" in t.lower() for t in texts), texts)

    def test_build_crawl_kwargs_reads_settings(self):
        settings = {"max_depth": 3, "max_pages": 7, "min_chars": 40,
                    "include_re": "/abt/", "exclude_re": "", "fetch_timeout": 5,
                    "max_keywords": 9}
        kw = _build_crawl_kwargs(settings)
        self.assertEqual(kw["max_depth"], 3)
        self.assertEqual(kw["max_pages"], 7)
        self.assertEqual(kw["min_chars"], 40)
        self.assertEqual(kw["include_re"], "/abt/")
        self.assertIsNone(kw["exclude_re"])

    def test_worker_crawls_synchronously_with_injected_fetch(self):
        settings = {"db_path": os.path.join(self._tmp, "w.db"), "base_url": BASE,
                    "max_depth": 0, "max_pages": 5, "min_chars": 50}
        worker = WebIndexCrawlWorker(
            settings, BASE, _build_crawl_kwargs(settings),
            keyword_extractor=None, fetch_func=_fake_fetch,
        )
        captured = []
        worker.finished_result.connect(captured.append)
        # Run on the main thread (synchronous) — direct-connected slot fires inline.
        worker.run()
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["pages_indexed"], 1)
        self.assertIn(BASE, captured[0]["indexed_urls"])

    def test_worker_emits_failed_on_bad_base_url(self):
        settings = {"db_path": os.path.join(self._tmp, "w2.db")}
        worker = WebIndexCrawlWorker(settings, "not-a-url", _build_crawl_kwargs({}),
                                     keyword_extractor=None, fetch_func=_fake_fetch)
        errs = []
        worker.failed.connect(errs.append)
        worker.run()
        self.assertEqual(len(errs), 1)
        worker.store_close = None  # store built+closed inside run()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()