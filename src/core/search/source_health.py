"""Per-run source health: skip a network source after consecutive connection failures.

Claude Generated.

An unreachable server makes every search term wait for the full timeout. This
module counts *connection* failures per source (the plugin instance id) and,
after :data:`FAILURE_THRESHOLD` in a row, marks the source down: until the run
ends or :data:`DOWN_EXPIRY_S` seconds have passed, requests to it are refused
without touching the network (:class:`SourceDownError`).

What counts:

* a connection failure (DNS error, refused connection, connect timeout, TLS or
  proxy failure): see :func:`is_connection_error`;
* NOT an HTTP error status, NOT an empty hit list, NOT a read timeout (the
  server accepted the connection);
* every completed exchange with the server, an HTTP error status included,
  resets the counter.

A source with several attempts per request (Libero: 3 with backoff) counts each
attempt.

Skipped terms carry :func:`down_message` (prefix :data:`SOURCE_DOWN_PREFIX`) as
their per-term error, i.e. they are *unknown*, never "no match": every cache
layer refuses to store an errored term as an empty hit. The skip is announced
once per source and episode (log warning + ``state.notice`` on the state bus),
not once per term.

State is process-wide (one dict, one lock), like ``run_timing``. A run start
resets it: :func:`begin_run` (``PipelineManager.start_pipeline``) and the
outermost :func:`run_scope` (``WorkflowExecutor.run``, ``MetaAgent.run``).

Qt-free; the state-bus import is lazy and best-effort.
"""

from __future__ import annotations

import functools
import logging
import socket
import threading
import time
import urllib.error
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

#: Consecutive connection failures after which a source is skipped.
FAILURE_THRESHOLD = 3
#: Seconds after which a down state (or a partial failure count) is forgotten,
#: so a long-lived GUI does not keep a source blocked.
DOWN_EXPIRY_S = 300

#: Stable marker in per-term error messages; reporters match on it.
SOURCE_DOWN_PREFIX = "source_down:"

# Injectable clock (tests move it instead of sleeping).
_now: Callable[[], float] = time.monotonic


@dataclass
class _Entry:
    failures: int = 0
    last_failure: float = 0.0
    down_since: Optional[float] = None


_lock = threading.Lock()
_state: Dict[str, _Entry] = {}
_depth = 0


class SourceDownError(requests.exceptions.RequestException):
    """Raised instead of a request to a source that is down - Claude Generated.

    A ``RequestException`` so the clients' existing ``except`` branches turn it
    into their usual per-term failure; its message is :func:`down_message`.
    """

    def __init__(self, source: str):
        self.source = source
        super().__init__(down_message(source))


def down_message(source: str) -> str:
    """Per-term error text for a skipped source - Claude Generated"""
    return (
        f"{SOURCE_DOWN_PREFIX} {source}: nach {FAILURE_THRESHOLD} Verbindungsfehlern "
        "übersprungen, keine Netzabfrage"
    )


def announcement(source: str) -> str:
    """The one message per source and episode - Claude Generated"""
    minutes = max(1, DOWN_EXPIRY_S // 60)
    return (
        f"{source}: nach {FAILURE_THRESHOLD} Verbindungsfehlern für diesen Lauf "
        f"übersprungen (neuer Versuch beim nächsten Lauf oder nach {minutes} Minuten)"
    )


def is_source_down(message: Any) -> bool:
    """Whether an error message carries the source-down marker - Claude Generated"""
    return isinstance(message, str) and SOURCE_DOWN_PREFIX in message


def split_source_down(
    errors: Optional[Mapping[str, str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Split ``{"<source>:<term>": message}`` into other errors and skips.

    Returns ``(others, skipped_by_source)``, the counterpart of
    ``cache_only.split_cache_misses``. - Claude Generated
    """
    others: Dict[str, str] = {}
    skipped: Dict[str, List[str]] = {}
    for key, message in (errors or {}).items():
        if is_source_down(message):
            source, _, term = str(key).partition(":")
            skipped.setdefault(source, []).append(term)
        else:
            others[key] = message
    return others, skipped


def is_connection_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a failure to reach the server at all - Claude Generated.

    ``requests``: ``ConnectionError`` and its subclasses (``ConnectTimeout``,
    ``SSLError``, ``ProxyError``); a ``ReadTimeout`` or ``HTTPError`` is not.
    ``urllib``: ``URLError`` (``urlopen`` wraps DNS, refused and connect-timeout
    errors in it) but not its subclass ``HTTPError``. Plus bare socket-level
    ``gaierror`` / ``ConnectionError``.
    """
    if isinstance(exc, SourceDownError):
        return False
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if isinstance(exc, requests.exceptions.RequestException):
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, (socket.gaierror, ConnectionError))


def _is_server_answer(exc: BaseException) -> bool:
    """An exception that still proves the server answered (HTTP status) - Claude Generated"""
    return isinstance(exc, (urllib.error.HTTPError, requests.exceptions.HTTPError))


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def _entry(source: str) -> Optional[_Entry]:
    """The live entry for ``source`` (expired ones are dropped). Caller holds the lock."""
    entry = _state.get(source)
    if entry is None:
        return None
    since = entry.down_since if entry.down_since is not None else entry.last_failure
    if _now() - since >= DOWN_EXPIRY_S:
        del _state[source]
        return None
    return entry


def begin_run() -> None:
    """Forget every source's state (start of a run) - Claude Generated

    Resets the per-run overview (``source_usage``) as well, so both cover the
    same run.
    """
    with _lock:
        _state.clear()
    _reset_usage()


def _reset_usage() -> None:
    """Reset ``source_usage`` together with this state - Claude Generated"""
    from src.core.search import source_usage

    source_usage.reset()


@contextmanager
def run_scope() -> Iterator[None]:
    """Mark a run; only the outermost scope resets the state - Claude Generated.

    Nested runs (a composite step, the MetaAgent's per-step executor calls)
    keep what the enclosing run has learnt.
    """
    global _depth
    with _lock:
        outermost = _depth == 0
        _depth += 1
        if outermost:
            _state.clear()
    if outermost:
        _reset_usage()
    try:
        yield
    finally:
        with _lock:
            _depth -= 1


def run_scoped(fn: Callable) -> Callable:
    """Decorator form of :func:`run_scope` - Claude Generated"""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with run_scope():
            return fn(*args, **kwargs)

    return wrapper


def is_down(source: Optional[str]) -> bool:
    """Whether ``source`` is currently skipped - Claude Generated"""
    if not source:
        return False
    with _lock:
        entry = _entry(source)
        return entry is not None and entry.down_since is not None


def failing(source: Optional[str]) -> bool:
    """Whether the latest exchange with ``source`` was a connection failure.

    True while the counter is above zero (a success resets it) and while the
    source is down. Lets a client tell "the server was not reached" from "the
    server found nothing" after a call that swallowed the exception.
    - Claude Generated
    """
    if not source:
        return False
    with _lock:
        entry = _entry(source)
        return entry is not None and (entry.failures > 0 or entry.down_since is not None)


def record_success(source: Optional[str]) -> None:
    """The server answered: reset the counter - Claude Generated"""
    if not source:
        return
    with _lock:
        _state.pop(source, None)


def record_failure(source: Optional[str], exc: Optional[BaseException] = None) -> bool:
    """Count one connection failure; True when this one marks the source down.

    - Claude Generated
    """
    if not source:
        return False
    with _lock:
        entry = _entry(source)
        if entry is None:
            entry = _state[source] = _Entry()
        if entry.down_since is not None:
            return False
        entry.failures += 1
        entry.last_failure = _now()
        became_down = entry.failures >= FAILURE_THRESHOLD
        if became_down:
            entry.down_since = entry.last_failure
    if became_down:
        from src.core.search import source_usage

        source_usage.mark_down(source)
        _announce(source, exc)
    return became_down


def failure_mark(source: Optional[str]) -> Optional[Tuple[int, float, Optional[float]]]:
    """``(failures, last_failure, down_since)`` of ``source``, ``None`` without
    a live entry. Changes with every counted connection failure, so a caller
    comparing it before and after a call learns whether that call failed to
    reach the server. - Claude Generated"""
    if not source:
        return None
    with _lock:
        entry = _entry(source)
        if entry is None:
            return None
        return (entry.failures, entry.last_failure, entry.down_since)


def status() -> Dict[str, Dict[str, Any]]:
    """Snapshot ``{source: {failures, down}}`` of the live entries - Claude Generated"""
    with _lock:
        out = {}
        for source in list(_state):
            entry = _entry(source)
            if entry is not None:
                out[source] = {"failures": entry.failures, "down": entry.down_since is not None}
        return out


def _announce(source: str, exc: Optional[BaseException]) -> None:
    text = announcement(source)
    logger.warning(f"{text}; letzter Fehler: {exc}" if exc is not None else text)
    _emit_notice(text)


def _emit_notice(text: str) -> None:
    """Broadcast on the state bus (GUI log, webapp, CLI stderr) - Claude Generated"""
    try:
        from src.core.state_bus import AlimaStateBus

        AlimaStateBus().emit_event("state.notice", {"text": text, "level": "warning"})
    except Exception as exc:  # noqa: BLE001 - a missing bus must not break a search
        logger.debug(f"source_health notice not emitted: {exc}")


# --------------------------------------------------------------------------- #
# The request guard
# --------------------------------------------------------------------------- #
def call(source: Optional[str], fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run one network request ``fn(*args, **kwargs)`` under the health gate.

    ``source`` ``None`` (a client built outside a plugin instance) runs ``fn``
    untracked. A down source raises :class:`SourceDownError` without calling
    ``fn``. A connection failure is counted and re-raised; a returned call or
    an HTTP-status exception resets the counter. - Claude Generated
    """
    if not source:
        return fn(*args, **kwargs)
    if is_down(source):
        raise SourceDownError(source)
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        if is_connection_error(exc):
            record_failure(source, exc)
        elif _is_server_answer(exc):
            record_success(source)
        raise
    record_success(source)
    return result
