"""GUI crawl action + worker for the webindex plugin - Claude Generated.

Wires a „Seite indizieren" button onto the webindex lookup instance form: runs
``crawl_site`` in a background :class:`StoppableWorker` (non-blocking, cancellable)
with a live progress log + final stats dialog. Model resolution + the keyword
workflow are reused from :mod:`src.utils.lookups.webindex.keywords` so the GUI and
the CLI crawl behave identically.

Registered into the plugin-settings-tab type-action registry on import of this
module (called from ``plugin_settings_tab``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
    QMessageBox,
)

from .workers import StoppableWorker

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class WebIndexCrawlWorker(StoppableWorker):
    """Runs ``crawl_site`` off the UI thread with progress + cancel - Claude Generated.

    Builds its own :class:`WebIndexStore` inside ``run`` so the QtSql connection
    is native to the worker thread (per-thread connections — see MEMORY.md).
    """

    progress = pyqtSignal(str)          # per-page progress line
    finished_result = pyqtSignal(dict)  # crawl stats dict
    failed = pyqtSignal(str)             # error message

    def __init__(self, settings: Dict[str, Any], base_url: str,
                 crawl_kwargs: Dict[str, Any],
                 keyword_extractor: Optional[Callable[[str, int], List[str]]],
                 fetch_func: Optional[Callable[..., Any]] = None):
        super().__init__()
        self._settings = settings
        self._base_url = base_url
        self._crawl_kwargs = crawl_kwargs
        self._keyword_extractor = keyword_extractor
        # fetch_func is injectable for tests (production omits it → guarded fetch).
        self._fetch_func = fetch_func

    def run(self) -> None:
        store = None
        try:
            from src.utils.lookups.webindex.indexer import crawl_site
            from src.utils.lookups.webindex.store import WebIndexStore

            store = WebIndexStore(self._settings, f"webindex_gui_{id(self)}")
            self.progress.emit(f"Starte Crawl: {self._base_url}")
            if self._keyword_extractor is None:
                self.progress.emit("Kein LLM-Provider → nur Meta-/Überschriften-Keywords.")
            kwargs = dict(
                keyword_extractor=self._keyword_extractor,
                progress_callback=lambda u, info: self.progress.emit(
                    f"[Tiefe {info.get('depth', 0)}] {u} "
                    f"({info.get('chars', 0)} Zeichen, HTTP {info.get('status', '–')})"
                ),
                should_stop=self.is_interrupted,
            )
            if self._fetch_func is not None:
                kwargs["fetch_func"] = self._fetch_func
            result = crawl_site(store, base_url=self._base_url, **kwargs, **self._crawl_kwargs)
            if self.is_interrupted():
                self.progress.emit("Abgebrochen.")
            self.finished_result.emit(result)
        except Exception as e:  # noqa: BLE001 — surface any failure to the dialog
            logger.exception("webindex crawl worker failed")
            self.failed.emit(str(e))
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# Dialog + action entry point
# --------------------------------------------------------------------------- #
def _build_crawl_kwargs(settings: Dict[str, Any]) -> Dict[str, Any]:
    def _i(key, default):
        try:
            return int(settings.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    return {
        "max_depth": _i("max_depth", 2),
        "max_pages": _i("max_pages", 50),
        "include_re": (str(settings.get("include_re") or "").strip() or None),
        "exclude_re": (str(settings.get("exclude_re") or "").strip() or None),
        "fetch_timeout": _i("fetch_timeout", 20),
        "min_chars": _i("min_chars", 50),
        "max_keywords": _i("max_keywords", 15),
    }


def run_crawl_dialog(inst, parent=None) -> None:
    """Open the crawl progress dialog for one webindex instance - Claude Generated.

    ``inst`` is the (already-flushed) :class:`PluginInstanceConfig`.
    """
    settings = dict(inst.settings or {})
    base_url = str(settings.get("base_url") or "").strip()
    if not base_url:
        QMessageBox.warning(
            parent, "Webindex",
            "Bitte zuerst eine Basis-URL (http/https) in der Instanz setzen."
        )
        return

    # Model: instance llm_provider/llm_model → global default.
    keyword_extractor: Optional[Callable[[str, int], List[str]]] = None
    try:
        from src.utils.config_manager import ConfigManager
        from src.utils.lookups.webindex.keywords import (
            build_keyword_extractor, resolve_crawl_model,
        )

        config = ConfigManager().load_config()
        provider, model = resolve_crawl_model(config, settings)
        if provider:
            from src.llm.llm_service import LlmService

            llm_service = LlmService(providers=None, config_manager=ConfigManager())
            keyword_extractor = build_keyword_extractor(llm_service, provider, model)
    except Exception as e:  # noqa: BLE001 — GUI must not crash on model-resolution errors
        logger.warning(f"webindex crawl model/extractor setup failed: {e}")

    worker = WebIndexCrawlWorker(settings, base_url, _build_crawl_kwargs(settings),
                                  keyword_extractor)

    dlg = QDialog(parent)
    dlg.setWindowTitle("Webindex — Seite indizieren")
    dlg.setMinimumSize(560, 360)
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel(f"Basis-URL: {base_url}"))
    log = QPlainTextEdit()
    log.setReadOnly(True)
    layout.addWidget(log)
    bar = QProgressBar()
    bar.setRange(0, 0)  # indeterminate while crawling
    layout.addWidget(bar)
    cancel = QPushButton("Abbrechen")
    layout.addWidget(cancel)

    state = {"done": False}

    def _finish(stats: dict) -> None:
        state["done"] = True
        bar.setRange(0, 1)
        bar.setValue(1)
        log.appendPlainText("")
        log.appendPlainText(f"Fertig. Indiziert: {stats.get('pages_indexed', 0)}  "
                            f"Übersprungen: {stats.get('pages_skipped', 0)}  "
                            f"Besucht: {stats.get('visited', 0)}")
        if stats.get("errors"):
            log.appendPlainText(f"Fehler: {len(stats['errors'])}")
            for e in stats["errors"][:10]:
                log.appendPlainText(f"  - {e}")
        if stats.get("indexed_urls"):
            log.appendPlainText("URLs:")
            for u in stats["indexed_urls"]:
                log.appendPlainText(f"  - {u}")

    def _fail(msg: str) -> None:
        state["done"] = True
        bar.setRange(0, 1)
        log.appendPlainText(f"\n✗ Fehler: {msg}")

    worker.progress.connect(log.appendPlainText)
    worker.finished_result.connect(_finish)
    worker.failed.connect(_fail)
    # QThread.finished → close the dialog once the worker is truly done.
    worker.finished.connect(dlg.accept)
    cancel.clicked.connect(worker.request_stop)

    worker.start()
    dlg.exec()
    # Ensure the thread is cleaned up if the dialog closed another way.
    if worker.isRunning():
        worker.request_stop()
        worker.wait(5000)


def register(registry_fn: Callable[..., None]) -> None:
    """Register the crawl action for the webindex lookup type - Claude Generated."""
    registry_fn("lookup", "webindex", "Seite indizieren …", run_crawl_dialog)