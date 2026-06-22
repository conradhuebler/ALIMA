"""Persistent chat-session logging for the webapp - Claude Generated.

Writes one row per chat turn (user prompt, LLM response, tool calls + their
results, plus provider/model/mode/language metadata) into a SQLite database so
the operator can analyse chat behaviour offline.

Config-only feature (no UI): enabled by setting ``chat_config.session_log_db``
to a database path in ``config.json``. Empty path = disabled.

Design notes:
- stdlib ``sqlite3`` only; no dependency on the Qt ``DatabaseManager`` (the
  webapp chat turn runs on a worker thread and we want a dedicated, isolated
  store that is trivial to query later).
- A fresh connection per insert (guarded by a module lock) keeps this safe
  across the webapp's worker threads without sharing a connection.
- Logging failures are swallowed (logged at WARNING): a logging problem must
  never break a chat turn.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()
_initialized_paths: set[str] = set()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    session_id    TEXT,
    provider      TEXT,
    model         TEXT,
    mode          TEXT,
    language      TEXT,
    user_message  TEXT,
    response      TEXT,
    tool_calls    TEXT,   -- JSON array of {name, arguments, result, ...}
    iterations    INTEGER,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_log_session ON chat_log(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_log_ts ON chat_log(ts);
"""


def _resolve_path(db_path: str, config_dir: Optional[str]) -> Path:
    """Resolve ``db_path`` against ``config_dir`` when relative - Claude Generated."""
    p = Path(db_path).expanduser()
    if not p.is_absolute() and config_dir:
        p = Path(config_dir).expanduser() / p
    return p


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    key = str(path)
    if key not in _initialized_paths:
        conn.executescript(_SCHEMA)
        conn.commit()
        _initialized_paths.add(key)
    return conn


def log_chat_turn(
    db_path: str,
    *,
    session_id: str,
    provider: str,
    model: str,
    user_message: str,
    response: str,
    tool_log: Optional[List[Dict[str, Any]]] = None,
    mode: Optional[str] = None,
    language: Optional[str] = None,
    iterations: int = 0,
    error: Optional[str] = None,
    config_dir: Optional[str] = None,
) -> bool:
    """Append one chat turn to the SQLite log. Returns True on success.

    ``db_path`` empty/None disables logging (returns False). Never raises —
    a logging failure must not abort the chat turn. - Claude Generated
    """
    if not db_path:
        return False
    try:
        path = _resolve_path(db_path, config_dir)
        tool_calls_json = json.dumps(tool_log or [], ensure_ascii=False, default=str)
        with _write_lock:
            conn = _connect(path)
            try:
                conn.execute(
                    "INSERT INTO chat_log "
                    "(ts, session_id, provider, model, mode, language, "
                    " user_message, response, tool_calls, iterations, error) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        session_id, provider, model, mode, language,
                        user_message, response, tool_calls_json,
                        int(iterations or 0), error,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return True
    except Exception as e:  # logging must never break a chat turn
        logger.warning(f"chat_session_logger: failed to log turn: {e}")
        return False
