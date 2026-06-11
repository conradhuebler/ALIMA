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
KNOWN_SYSTEMS: Tuple[str, ...] = ("DK", "DDC", "RVK")


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
