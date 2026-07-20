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

from typing import Any, Dict, List, Optional, Tuple

# Recognised classification systems. Order is only used for stable display /
# iteration; matching is by explicit prefix, never positional. - Claude Generated
#
# These are ALSO the canonical data keys of the ``classifications`` dict in the
# GND-pool vocabulary (``{system: codes}``, WP-D1). Display prefix and data key
# are deliberately the same string, so there is one system vocabulary rather
# than an upper/lower pair needing a translation layer — the kind of round-trip
# rename WP-D1 P0 removed. Producers normalise through :func:`normalize_system`.
KNOWN_SYSTEMS: Tuple[str, ...] = ("DK", "DDC", "RVK", "BK")

# Full system names as third parties spell them, mapped to our canonical key.
# Only supra-regional, standardised systems are listed: a source that reports a
# LOCAL systematic ("Sachgruppen der DNB", "Systematik der TUB München", …) is
# deliberately left unmapped so :func:`normalize_system` drops it rather than
# filing a library-specific code under a system it does not belong to.
#
# ``DDC-Sachgruppen der ZDB`` is likewise NOT mapped to DDC: those are coarse
# DDC-derived subject groups, and putting a 3-digit group next to a full DDC
# number would present two different granularities as equivalent.
# - Claude Generated (WP-D2 harvest)
_SYSTEM_ALIASES = {
    "REGENSBURGER VERBUNDKLASSIFIKATION": "RVK",
    "DEWEY-DEZIMALKLASSIFIKATION": "DDC",
    "DEWEY DECIMAL CLASSIFICATION": "DDC",
    "BASISKLASSIFIKATION": "BK",
}

# Alias for readers that mean "the keys of a classifications dict" rather than
# "the systems we can render". Same tuple on purpose. - Claude Generated
SYSTEM_KEYS: Tuple[str, ...] = KNOWN_SYSTEMS


# How a classification came to be attached to a concept. The distinction is
# load-bearing, not decorative: an authority statement ("the GND record says this
# concept is DDC 551.48") and a statistical one ("RVK WI 4700 appeared in 13
# catalogue records about this term") are different claims, and a consumer that
# cannot tell them apart will treat a co-occurrence as ground truth.
# - Claude Generated (WP-D2, P0 revision)
ORIGIN_AUTHORITY = "authority"
ORIGIN_COOCCURRENCE = "cooccurrence"

# Authority outranks statistics when the same code arrives from both.
_ORIGIN_RANK = {ORIGIN_AUTHORITY: 0, ORIGIN_COOCCURRENCE: 1}


def classification_entry(
    code: str, *, count: Optional[int] = None, origin: str = ORIGIN_COOCCURRENCE
) -> Dict[str, Any]:
    """Build one canonical classification entry ``{code, count?, origin}``.

    ``count`` is the co-occurrence strength and is OMITTED for authority
    statements — an authority DDC has no frequency, and writing ``1`` there
    would invent evidence that does not exist.
    """
    entry: Dict[str, Any] = {"code": str(code or "").strip(), "origin": origin}
    if count is not None:
        entry["count"] = int(count)
    return entry


def _entry_sort_key(entry: Dict[str, Any]):
    """Authority first, then by descending evidence, then alphabetically.

    Sorting inside the merge makes the result independent of the order sources
    were merged in, and lets a consumer take "the strongest n" with a plain
    slice instead of re-deriving the ranking rules.
    """
    return (
        _ORIGIN_RANK.get(entry.get("origin"), 9),
        -int(entry.get("count") or 0),
        str(entry.get("code") or ""),
    )


def merge_classification_entries(
    existing: Optional[List[Dict[str, Any]]], incoming: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Merge two entry lists for ONE system, deduplicating by ``code``.

    ``count`` is combined with **max, never sum** — the same landmine rule the
    GND pool's ``count`` follows. Summing would inflate evidence when two
    sources saw overlapping records, and the counter bug (``038738e``) lived in
    exactly this kind of arithmetic. ``origin`` keeps the stronger claim.
    """
    by_code: Dict[str, Dict[str, Any]] = {}
    for entry in list(existing or []) + list(incoming or []):
        if not isinstance(entry, dict):
            continue
        code = str(entry.get("code") or "").strip()
        if not code:
            continue
        current = by_code.get(code)
        if current is None:
            by_code[code] = dict(entry, code=code)
            continue
        counts = [c for c in (current.get("count"), entry.get("count")) if c is not None]
        merged = dict(current)
        if counts:
            merged["count"] = max(int(c) for c in counts)
        if _ORIGIN_RANK.get(entry.get("origin"), 9) < _ORIGIN_RANK.get(
            current.get("origin"), 9
        ):
            merged["origin"] = entry["origin"]
        by_code[code] = merged
    return sorted(by_code.values(), key=_entry_sort_key)


def merge_classifications(
    existing: Optional[Dict[str, Any]], incoming: Optional[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge two ``{system: [entry]}`` dicts, per equal-rank system.

    Both sides are normalised first, so this is total over every accepted
    producer form: a plugin that emits bare code lists merges correctly instead
    of having its codes silently dropped for not being entry dicts, and two
    spellings of one system cannot end up as two keys. This is the single merge
    point for classifications — being tolerant HERE is what lets every caller
    stay simple.
    """
    out = normalize_classifications(existing)
    for system, entries in normalize_classifications(incoming).items():
        out[system] = merge_classification_entries(out.get(system), entries)
    return {system: entries for system, entries in out.items() if entries}


def codes_for_system(classifications: Any, system: str) -> List[str]:
    """The codes for one system, strongest first.

    Entries are kept sorted (authority, then descending co-occurrence), so the
    caller gets the ranking without knowing the rules — ``[:3]`` is "the three
    best", ``[0]`` is "the best".
    """
    key = normalize_system(system)
    if not key:
        return []
    out: List[str] = []
    for entry in (classifications or {}).get(key) or []:
        # Tolerate a bare code alongside the entry shape: readers sit downstream
        # of plugin output, and a producer that has not been converted yet must
        # degrade to "code without evidence", not to silence.
        code = entry.get("code") if isinstance(entry, dict) else entry
        code = str(code or "").strip()
        if code and code not in out:
            out.append(code)
    return out


def primary_code(classifications: Any, system: str) -> str:
    """The single best code for one system, or ``""``."""
    codes = codes_for_system(classifications, system)
    return codes[0] if codes else ""


def normalize_classifications(
    value: Any, *, origin: str = ORIGIN_COOCCURRENCE
) -> Dict[str, List[Dict[str, Any]]]:
    """Coerce any accepted input into the canonical ``{system: [entry]}`` shape.

    Accepts the entry shape itself, and bare code containers (set/list/str) for
    producers that have no evidence to report — those become entries with the
    given ``origin`` and no ``count``. System names are normalised, so a
    producer's spelling cannot introduce a second key for one system; unknown
    systems are dropped rather than guessed at.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for system, codes in (value or {}).items():
        key = normalize_system(system)
        if not key:
            continue
        if isinstance(codes, (str, bytes)):
            codes = [codes]
        entries: List[Dict[str, Any]] = []
        for code in codes or []:
            if isinstance(code, dict):
                if str(code.get("code") or "").strip():
                    entries.append(
                        classification_entry(
                            code["code"],
                            count=code.get("count"),
                            origin=code.get("origin", origin),
                        )
                    )
            elif str(code or "").strip():
                entries.append(classification_entry(code, origin=origin))
        # Merge into whatever is already under this key: two spellings of one
        # system ("rvk" and "RVK") normalise to the same key, and assigning
        # would let the second silently drop the first.
        merged = merge_classification_entries(out.get(key), entries)
        if merged:
            out[key] = merged
    return out


def normalize_system(system: str) -> str:
    """Normalise a system name to its canonical key (``"ddc"`` → ``"DDC"``).

    Returns ``""`` for an unknown or blank system, so callers can decide whether
    to drop the codes or file them under a default. Use this at every boundary
    where a system name arrives from outside (catalog field names, prefixed
    strings, plugin output) before writing into a ``classifications`` dict.
    """
    norm = str(system or "").strip().upper()
    if norm in KNOWN_SYSTEMS:
        return norm
    # Sources name their systems in full, often with the abbreviation in front
    # and the long form in brackets ("RVK (Regensburger Verbundklassifikation)").
    # Try the part before the bracket, then the long form itself.
    head = norm.split("(", 1)[0].strip()
    if head in KNOWN_SYSTEMS:
        return head
    for candidate in (norm, head, norm.strip("()")):
        resolved = _SYSTEM_ALIASES.get(candidate.strip())
        if resolved:
            return resolved
    return ""


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
