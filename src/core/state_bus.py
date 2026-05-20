"""AlimaStateBus — P-δ.1 (WP6 Sek 4). Claude Generated.

Singleton Qt-signal hub for cross-component state-mutation broadcasting.

Subscribers (AnalysisReviewTab, ChatWidget, AgenticContextWidget, future
WebApp SSE endpoint) call :meth:`subscribe` with an event type and a slot.
Producers (PipelineManager, ChatAgentWorker mutation tools — coming in
P-ε) call :meth:`emit_event` with an event type and a JSON-serialisable
diff payload.

Event types follow the format ``{domain}.{verb}`` (e.g. ``state.changed``,
``state.pipeline_started``, ``state.pipeline_completed``). Diffs are
compact dicts; consumers decide how to render.

Thread safety: Qt signals marshal slots to the owning thread by default
(`Qt.AutoConnection`), so background workers (PipelineWorker,
ChatAgentWorker) can ``emit_event`` directly without manual locking.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List

from PyQt6.QtCore import QObject, pyqtSignal


logger = logging.getLogger(__name__)


class _AlimaStateBus(QObject):
    """Internal QObject subclass — do not instantiate directly."""

    state_event = pyqtSignal(str, dict)

    def __init__(self) -> None:
        super().__init__()
        self._handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        logger.debug("AlimaStateBus singleton initialised")

    def emit_event(self, event_type: str, diff: Dict[str, Any]) -> None:
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(diff, dict):
            raise TypeError("diff must be a dict")

        for handler in list(self._handlers.get(event_type, [])):
            try:
                handler(diff)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Subscriber raised for event '{event_type}': {exc}"
                )

        try:
            self.state_event.emit(event_type, diff)
        except RuntimeError:
            pass

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        handlers = self._handlers.get(event_type)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            pass


_lock = threading.Lock()
_instance: _AlimaStateBus | None = None


def AlimaStateBus() -> _AlimaStateBus:
    """Return the singleton state bus instance."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = _AlimaStateBus()
    return _instance


def reset() -> None:
    """Drop the singleton. **Tests only.**"""
    global _instance
    with _lock:
        _instance = None
