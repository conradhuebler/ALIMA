"""The library ALIMA works for: ISIL and ELN, set once - Claude Generated.

Two fields in ``SystemConfig`` name the institution:

* ``institution_isil``: the ISIL of the library (e.g. ``DE-105``).
* ``institution_eln``: its ELN in the K10plus (e.g. ``105``). It is not derived
  from the ISIL; the two are entered separately.

Everything that used to carry ``DE-105`` / ``105`` as a fixed value derives it
from here:

* :func:`chain_end_marker` → ``$A<ISIL>``, the closing line of a keyword chain
  in the WinIBW entry;
* :func:`finc_dk_facet` → ``udk_raw_<isil in lower case without hyphen>``, the
  finc/Libero facet holding the library's DK notations;
* :func:`institution_filter` → the ISIL, the holding filter of the Libero web
  search;
* :func:`institution_eln` → the ELN, which picks the library's copy (``209A
  $B``) in a K10plus PICA record.

Each returns ``None`` when its field is empty; the callers then leave the
line out or report :data:`NOT_SET_MESSAGE` instead of an empty result.

The values in use are process-wide: ``ConfigManager`` sets them whenever it
loads or saves a configuration (:func:`set_current`), the same way the UI
language reaches ``i18n``. A function called with an explicit value ignores the
current one. Qt-free, no I/O.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: Field names in ``SystemConfig`` / the ``system_config`` JSON section.
ISIL_KEY = "institution_isil"
ELN_KEY = "institution_eln"

#: Stable prefix of the "institution not set" marker; reporters match on it.
NOT_SET_PREFIX = "institution_not_set:"
NOT_SET_MESSAGE = (
    f"{NOT_SET_PREFIX} Einrichtung nicht gesetzt (ISIL unter Einstellungen ▸ "
    "Datenquellen ▸ Einrichtung)"
)

#: Host that marks a configuration of the UB Freiberg (one-time migration).
_FREIBERG_HOST = "tu-freiberg.de"
_FREIBERG = ("DE-105", "105")
#: Plugin types whose URLs the migration looks at.
_MIGRATION_TYPES = ("catalog", "finc", "sru")


@dataclass(frozen=True)
class Institution:
    """ISIL and ELN as entered; empty strings when not set - Claude Generated"""

    isil: str = ""
    eln: str = ""


_lock = threading.Lock()
_current = Institution()
_UNSET: Any = object()


def normalize(value: Any) -> str:
    """A field value as stored: stripped text, ``""`` for ``None`` - Claude Generated"""
    return str(value or "").strip()


def set_current(isil: Any = "", eln: Any = "") -> Institution:
    """Make ``(isil, eln)`` the values in use; returns them - Claude Generated"""
    global _current
    value = Institution(normalize(isil), normalize(eln))
    with _lock:
        _current = value
    return value


def set_current_from(system_config: Any) -> Institution:
    """Take the values in use from a ``SystemConfig`` (missing fields = empty)."""
    return set_current(getattr(system_config, ISIL_KEY, ""), getattr(system_config, ELN_KEY, ""))


def current() -> Institution:
    """The values in use - Claude Generated"""
    return _current


def _isil(isil: Any) -> str:
    return normalize(_current.isil if isil is _UNSET else isil)


def _eln(eln: Any) -> str:
    return normalize(_current.eln if eln is _UNSET else eln)


def chain_end_marker(isil: Any = _UNSET) -> Optional[str]:
    """``$A<ISIL>`` (``$ADE-105``), ``None`` without ISIL - Claude Generated"""
    value = _isil(isil)
    return f"$A{value}" if value else None


def finc_dk_facet(isil: Any = _UNSET) -> Optional[str]:
    """``udk_raw_<isil>`` (``DE-105`` → ``udk_raw_de105``), ``None`` without ISIL.

    - Claude Generated
    """
    value = _isil(isil)
    if not value:
        return None
    return "udk_raw_" + value.lower().replace("-", "")


def institution_filter(isil: Any = _UNSET) -> Optional[str]:
    """The ISIL as holding filter, ``None`` without ISIL - Claude Generated"""
    return _isil(isil) or None


def institution_eln(eln: Any = _UNSET) -> Optional[str]:
    """The ELN, ``None`` when not set - Claude Generated"""
    return _eln(eln) or None


def is_not_set(message: Any) -> bool:
    """Whether a message carries the "institution not set" marker - Claude Generated"""
    return isinstance(message, str) and message.startswith(NOT_SET_PREFIX)


def not_set_summary(source: str, missed: int, total: int) -> str:
    """One run line for the DK step - Claude Generated"""
    return (
        f"{source}: Einrichtung nicht gesetzt (ISIL), DK-Notationen der Bibliothek "
        f"für {missed} von {total} Suchbegriffen nicht auswertbar"
    )


# ── one-time migration ────────────────────────────────────────────────────


def _settings_mention_freiberg(settings: Optional[Mapping[str, Any]]) -> bool:
    for value in (settings or {}).values():
        if isinstance(value, str) and _FREIBERG_HOST in value.lower():
            return True
    return False


def freiberg_instances(plugins: Iterable[Any]) -> list:
    """Instance ids of catalog/finc/sru instances with a ``tu-freiberg.de`` URL."""
    out = []
    for inst in plugins or []:
        if getattr(inst, "provider_id", "") not in _MIGRATION_TYPES:
            continue
        if _settings_mention_freiberg(getattr(inst, "settings", None)):
            out.append(getattr(inst, "instance_id", "") or getattr(inst, "provider_id", ""))
    return out


def migrate(system_config_data: Mapping[str, Any], system_config: Any,
            plugins: Iterable[Any]) -> bool:
    """Set ``DE-105``/``105`` once for a configuration of the UB Freiberg.

    Runs only when the ``system_config`` JSON carries neither key and a
    catalog, finc or sru instance has a URL with ``tu-freiberg.de``. Writes
    into ``system_config`` (the parsed dataclass) and logs it; returns whether
    it did. - Claude Generated
    """
    data = system_config_data or {}
    if ISIL_KEY in data or ELN_KEY in data:
        return False
    found = freiberg_instances(plugins)
    if not found:
        return False
    setattr(system_config, ISIL_KEY, _FREIBERG[0])
    setattr(system_config, ELN_KEY, _FREIBERG[1])
    logger.info(
        "Einrichtung gesetzt: ISIL %s, ELN %s (Konfiguration ohne Einrichtung, "
        "tu-freiberg.de in %s)", _FREIBERG[0], _FREIBERG[1], ", ".join(found),
    )
    return True


def keep_absent_on_save(system_config: Any, on_disk: Mapping[str, Any],
                        plugins: Iterable[Any]) -> Tuple[str, ...]:
    """Keys to leave out of the saved ``system_config`` section.

    Both fields are left out when both are empty, neither is in the file on
    disk, and the migration would not fire on the next load: a configuration
    that never had them is then saved unchanged. A cleared value of a Freiberg
    configuration is written as ``""``, otherwise the next load would set it
    again. - Claude Generated
    """
    if normalize(getattr(system_config, ISIL_KEY, "")) or normalize(
            getattr(system_config, ELN_KEY, "")):
        return ()
    disk = on_disk or {}
    if ISIL_KEY in disk or ELN_KEY in disk:
        return ()
    if freiberg_instances(plugins):
        return ()
    return (ISIL_KEY, ELN_KEY)
