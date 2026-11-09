"""Per-instance request timeout of a network search source - Claude Generated.

One declaration (:func:`timeout_field`) for the settings form and one reader
(:func:`timeout_from`) for the provider, so the form's bounds and the value
the client receives cannot drift apart. The default of each provider is the
limit its client had before the setting existed, so an instance without the
key behaves as before.

``finc`` and ``kvk`` declared a ``timeout`` field earlier; they keep that key
(existing config files stay valid as they are) and get the same bounds.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from src.core.plugins.schema import INT, ConfigField

TIMEOUT_KEY = "timeout_s"
TIMEOUT_MIN_S = 1
TIMEOUT_MAX_S = 120


def timeout_field(default: int, *, key: str = TIMEOUT_KEY, help: str = "") -> ConfigField:
    """The INT field for one provider's request timeout in seconds - Claude Generated"""
    text = help or "Wartezeit je Anfrage an diese Quelle."
    return ConfigField(
        key=key,
        label="Zeitlimit (s)",
        kind=INT,
        default=int(default),
        minimum=TIMEOUT_MIN_S,
        maximum=TIMEOUT_MAX_S,
        help=f"{text} {TIMEOUT_MIN_S} bis {TIMEOUT_MAX_S} Sekunden, Standard {int(default)}.",
    )


def timeout_from(
    settings: Optional[Mapping[str, Any]], default: int, *, key: str = TIMEOUT_KEY
) -> int:
    """The timeout an instance's settings select, bounded; ``default`` when unset.

    An unset, empty or zero value means "not configured" (the earlier readers
    did ``int(x or default)``). - Claude Generated
    """
    raw = (settings or {}).get(key)
    if raw in (None, "", 0, "0"):
        return int(default)
    return int(timeout_field(default, key=key).coerce(raw))
