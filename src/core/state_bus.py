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

Thread safety
-------------
The singleton MUST be created on the GUI thread (e.g. ``alima_gui.py``
calls ``AlimaStateBus()`` at startup). Subscribers connect to the
bus's ``state_event`` Qt signal with ``Qt.QueuedConnection`` so that
worker-thread producers (``PipelineWorker``, ``ChatAgentWorker``) can
``emit_event`` directly: the signal is delivered to the GUI thread
where widget-manipulating slots (rendering, status messages) are safe
to run.

Bus contract for ``tool.called`` / ``tool.result``
-------------------------------------------------
Both events carry an ``id`` field (string, ``tc_`` + 12-hex, see
:func:`src.core.agents.sub_agents.caching_tool_registry.make_tool_call_id`).
The id is unique per call/result pair, so a consumer can attach a
``tool.result`` payload to the matching ``tool.called`` block. ``id``
is always present (never empty in practice) but consumers must treat
it as best-effort: an empty id means the producer could not allocate
one in time.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Tuple

from PyQt6.QtCore import QObject, Qt, pyqtSignal


logger = logging.getLogger(__name__)


class _AlimaStateBus(QObject):
    """Internal QObject subclass — do not instantiate directly.

    Must live on the GUI thread so that ``state_event`` (a Qt signal
    delivered with ``Qt.QueuedConnection`` to every subscriber) is
    dispatched on the GUI thread.
    """

    state_event = pyqtSignal(str, dict)

    def __init__(self) -> None:
        super().__init__()
        # Per-subscription bookkeeping: each (event_type, handler) pair
        # gets a unique closure slot bound to ``state_event`` via
        # ``Qt.QueuedConnection`` so dispatch happens on the GUI
        # thread regardless of which thread called ``emit_event``.
        self._subscriptions: List[Tuple[str, Callable[[Dict[str, Any]], None], Callable[[str, dict], None]]] = []
        logger.debug("AlimaStateBus singleton initialised")

    def emit_event(self, event_type: str, diff: Dict[str, Any]) -> None:
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(diff, dict):
            raise TypeError("diff must be a dict")

        # Decide delivery mode:
        #   * Same thread as the bus QObject → direct call. The
        #     subscriber is already on the GUI thread; no marshalling
        #     needed and tests can assert synchronously.
        #   * Different thread → Qt-signal with QueuedConnection so
        #     the slot runs on the bus's owning thread (the GUI
        #     thread). This is the property that makes
        #     widget-manipulating subscribers safe when a worker
        #     thread (PipelineWorker, ChatAgentWorker) emits.
        from PyQt6.QtCore import QCoreApplication, QThread

        bus_thread = self.thread()
        current_thread = QThread.currentThread()
        if bus_thread is current_thread:
            for event_filter, handler, _slot in self._subscriptions:
                if event_filter != event_type:
                    continue
                try:
                    handler(diff)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Subscriber raised for event '{event_type}': {exc}"
                    )
            return

        # Cross-thread path: emit the Qt signal; the connected slot
        # is delivered on the bus's owning thread (GUI).
        # If the bus's owning thread has no running event loop, the signal
        # would be silently lost. Fall back to direct dispatch in that case
        # so headless / webapp consumers still receive the event.
        #
        # The dispatcher probe alone is not sufficient: a QCoreApplication may
        # exist without anyone calling exec() — DatabaseManager creates one for
        # QtSql, so the webapp process has a dispatcher on its main thread but
        # never runs a Qt loop. Queued events would then pile up undelivered.
        # Hosts without a Qt loop declare that via set_direct_dispatch(True).
        # - Claude Generated
        from PyQt6.QtCore import QAbstractEventDispatcher

        if _force_direct_dispatch or QAbstractEventDispatcher.instance(bus_thread) is None:
            for event_filter, handler, _slot in self._subscriptions:
                if event_filter != event_type:
                    continue
                try:
                    handler(diff)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Subscriber raised for event '{event_type}' (no-loop fallback): {exc}"
                    )
            return

        try:
            self.state_event.emit(event_type, diff)
        except RuntimeError:
            # Qt rejected the emit (e.g. object already destroyed). Fall back
            # to direct dispatch so we still observe the contract.
            for event_filter, handler, _slot in self._subscriptions:
                if event_filter != event_type:
                    continue
                try:
                    handler(diff)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"Subscriber raised for event '{event_type}' (fallback path): {exc}"
                    )

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        # Idempotent: don't double-connect the same (event_type, handler) pair.
        for event_filter, existing_handler, _slot in self._subscriptions:
            if event_filter == event_type and existing_handler is handler:
                return

        def _slot(event_arg: str, payload: dict, _h: Callable[[Dict[str, Any]], None] = handler, _f: str = event_type) -> None:
            if event_arg != _f:
                return
            try:
                _h(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Subscriber raised for event '{_f}': {exc}"
                )

        # QueuedConnection: when ``emit_event`` is called from a worker
        # thread, Qt delivers the signal to the bus's owning thread
        # (GUI) and runs ``_slot`` there. This is the property that
        # makes widget-manipulating subscribers safe.
        self.state_event.connect(_slot, type=Qt.ConnectionType.QueuedConnection)
        self._subscriptions.append((event_type, handler, _slot))

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        kept: List[Tuple[str, Callable[[Dict[str, Any]], None], Callable[[str, dict], None]]] = []
        for event_filter, existing_handler, slot in self._subscriptions:
            if event_filter == event_type and existing_handler is handler:
                try:
                    self.state_event.disconnect(slot)
                except (TypeError, RuntimeError):
                    # Already disconnected (e.g. QObject destroyed) —
                    # not an error.
                    pass
                continue
            kept.append((event_filter, existing_handler, slot))
        self._subscriptions = kept


_lock = threading.Lock()
_instance: _AlimaStateBus | None = None
# Hosts without a running Qt event loop (webapp, headless runners) set this so
# cross-thread events are delivered synchronously on the emitting thread
# instead of being queued for a loop that never spins. - Claude Generated
_force_direct_dispatch: bool = False


def set_direct_dispatch(enabled: bool) -> None:
    """Deliver cross-thread events synchronously instead of via Qt queue.

    For processes that import Qt but never run ``exec()`` (the webapp, CLI
    runners). Subscribers then run on the *emitting* thread, so their handlers
    must be thread-safe — true for the webapp render bridge, which only appends
    to a lock-protected buffer. The GUI must NOT enable this: its subscribers
    touch widgets and need the GUI-thread hop. - Claude Generated
    """
    global _force_direct_dispatch
    _force_direct_dispatch = bool(enabled)


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
