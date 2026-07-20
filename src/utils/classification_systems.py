"""classification_systems.py - Shared classification-system registry.

Single source of truth for the known library classification systems (DK / DDC /
RVK / …) and for splitting a prefixed classification string into its
``(system, code)`` parts and formatting it back. Replaces the ~7 duplicated,
DK/RVK-only ``startswith("DK ")``/``startswith("RVK ")`` implementations that
were scattered across the GUI, webapp, exporters and report renderer — so adding
a system (DDC) is a one-line change here instead of an edit in every consumer.

Claude Generated (general-notation generalization, June 2026).
"""
from __future__ import annotations

from typing import Tuple

# Recognised classification systems. Order is only used for stable display /
# iteration; matching is by explicit prefix, never positional. - Claude Generated
#
# These are ALSO the canonical data keys of the ``classifications`` dict in the
# GND-pool vocabulary (``{system: codes}``, WP-D1). Display prefix and data key
# are deliberately the same string, so there is one system vocabulary rather
# than an upper/lower pair needing a translation layer — the kind of round-trip
# rename WP-D1 P0 removed. Producers normalise through :func:`normalize_system`.
KNOWN_SYSTEMS: Tuple[str, ...] = ("DK", "DDC", "RVK")

# Alias for readers that mean "the keys of a classifications dict" rather than
# "the systems we can render". Same tuple on purpose. - Claude Generated
SYSTEM_KEYS: Tuple[str, ...] = KNOWN_SYSTEMS


def normalize_system(system: str) -> str:
    """Normalise a system name to its canonical key (``"ddc"`` → ``"DDC"``).

    Returns ``""`` for an unknown or blank system, so callers can decide whether
    to drop the codes or file them under a default. Use this at every boundary
    where a system name arrives from outside (catalog field names, prefixed
    strings, plugin output) before writing into a ``classifications`` dict.
    """
    norm = str(system or "").strip().upper()
    return norm if norm in KNOWN_SYSTEMS else ""


def split_classification_code(value: str) -> Tuple[str, str]:
    """Split a prefixed classification string into ``(system, code)``.

    ``"DK 666.76"`` → ``("DK", "666.76")``; ``"DDC 530.1"`` → ``("DDC", "530.1")``;
    ``"RVK Q12"`` → ``("RVK", "Q12")``. The system prefix is matched
    case-insensitively; the code keeps its original casing. An unprefixed value
    returns ``("", value)`` so callers can decide on a default system.
    """
    text = str(value or "").strip()
    upper = text.upper()
    for system in KNOWN_SYSTEMS:
        prefix = f"{system} "
        if upper.startswith(prefix):
            return system, text[len(prefix):].strip()
    return "", text


def format_classification(system: str, code: str) -> str:
    """Join ``("DDC", "530.1")`` → ``"DDC 530.1"``.

    A blank/unknown system returns just the (stripped) code. The system is
    normalised to its canonical upper-case form.
    """
    sys_norm = (system or "").strip().upper()
    code_str = str(code or "").strip()
    if sys_norm in KNOWN_SYSTEMS:
        return f"{sys_norm} {code_str}".strip()
    return code_str


def classification_system(value: str) -> str:
    """Return just the system of a classification string (``""`` if unprefixed)."""
    return split_classification_code(value)[0]
