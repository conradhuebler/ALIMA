"""Local shelf classification (Aufstellungssystematik) as a lookup plugin.

A house shelf system is a *controlled subset* of a standard classification plus a
location: UB Freiberg's Freihandsystematik is 657 DK notations with label and
floor. It is deliberately NOT a classification system of its own — see the note in
``classification_systems.py`` on why local systematics are never filed under
``KNOWN_SYSTEMS``. The shelf mark lives beside the subject notations, not among
them.

The plugin is generic: one vocabulary file per institution
(``vocabularies/<id>.yaml`` or ``~/.config/alima/vocabularies/``), selected
per instance. The file carries its own provenance (``source``, ``retrieved``) and
is maintained by the library, not by ALIMA. It answers one question: *given the subject notations a
work was classified with, which shelf places come into question, and why?*

Subject notation and shelf notation are near each other but need not agree. The
matcher therefore offers two independent signals and never decides:

* **notation proximity** — the shelf entry contains the subject code (prefix in
  the decimal hierarchy, or a range like ``122/129``), or is its nearest
  ancestor/sibling when nothing contains it;
* **label match** — the shelf entry's wording hits the work's subject keywords
  (switchable, ``use_labels``).

Measured against 407 TU Freiberg open-shelf titles that carry both a call number
and DK notations (finc index, ``building:"(DE-105)HB"``; the call number's
notation part compared to the proposal, dots normalised):

===========================  =====  =====  =====
signal                       top-1  top-3  top-5
===========================  =====  =====  =====
first/core notation only      66.6   75.2   76.9
first notation + keywords     65.1   75.4   78.4
all notations                 41.3   60.0   67.8
keywords only                 11.8   20.9   25.3
===========================  =====  =====  =====

Two things follow, and both are wired in: the *core* notation carries the
placement (using every assigned notation costs 25 points), and the label signal
is worth about a point of recall, not more. The remaining quarter is not
reachable from the notation at all — relation marks like ``55:93/99``
(Geschichte der Geowissenschaften) or a shelf that sits in another branch than
the subject notation. That is why this step proposes and never decides.

Claude Generated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.core.plugins.schema import BOOL, INT, TEXT, ConfigField, PluginDoc

from .registry import LookupToolSpec, register_lookup

logger = logging.getLogger(__name__)

# Where a bare vocabulary id (``freihand_fg``) is looked for, in order. Mirrors
# ``workflow_loader.DEFAULT_SEARCH_PATHS``: repo copy, operator's own, package
# copy. Deliberately NOT under ``data/`` — that directory holds caches and is
# git-ignored, so a shipped vocabulary would never reach a fresh checkout.
VOCABULARY_SEARCH_PATHS: List[Path] = [
    Path("vocabularies"),
    Path.home() / ".config" / "alima" / "vocabularies",
    Path(__file__).parent.parent.parent.parent / "vocabularies",
]

# Words too generic to carry a label match on their own.
_LABEL_STOPWORDS = {
    "allgemeines", "allgemein", "sonstige", "andere", "übrige", "gesamt",
    "einzelne", "verschiedene", "grundlagen", "siehe", "auch", "und", "oder",
    "der", "die", "das", "des", "dem", "den", "für", "von", "mit", "im", "in",
}


def digits_of(notation: str) -> str:
    """Reduce a notation to its comparable digit string.

    UDC/DK is a decimal hierarchy: the dots are readability, so ``622.83`` sorts
    and nests as ``62283``. Auxiliaries (``(036)``), relations (``:943``) and the
    open marker (``...``) are cut off — they are handled separately.
    """
    head = re.split(r"[(:]", str(notation or "").strip(), maxsplit=1)[0]
    head = head.replace("...", "")
    return re.sub(r"\D", "", head)


def _auxiliary_of(notation: str) -> str:
    """The ``(036)`` / ``:943.21`` tail of a notation, normalised, else ``""``."""
    match = re.search(r"[(:].*$", str(notation or "").strip())
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def _in_range(code_digits: str, low: str, high: str) -> bool:
    """Decimal-fraction containment of ``code`` in ``[low, high]``.

    Digit strings compare like fractions after a common prefix, so padding the
    bound with ``0`` (lower) resp. ``9`` (upper) gives the right answer for both
    ``122/129`` (whole numbers) and ``622.012/.03`` (fractions).
    """
    if not code_digits or not low or not high:
        return False
    width = max(len(code_digits), len(low), len(high))
    return low.ljust(width, "0") <= code_digits.ljust(width, "0") <= high.ljust(width, "9")


def _split_range(notation: str) -> Optional[Tuple[str, str]]:
    """``"622.012/.03"`` → ``("622012", "62203")``; ``None`` when not a range.

    The right-hand side is often an abbreviation that continues the left one
    (``347.77/.78``, ``622.012/.03``): it is completed from the left's stem.

    Only a slash in the *base* opens a range. Six entries of the Freiberg
    scheme carry one inside the auxiliary instead (``551(98/99)`` — Arktis,
    Antarktis; ``622(4/5)`` — Bergbau Russland): read as a range those spanned
    half the schedule and swallowed unrelated codes. - Claude Generated
    """
    raw = str(notation or "").strip()
    base = re.split(r"[(:]", raw, maxsplit=1)[0]
    if "/" not in base:
        return None
    left_raw, _, right_raw = base.partition("/")
    left = digits_of(left_raw)
    right_raw = right_raw.strip()
    if right_raw.startswith("."):
        # ".03" continues "622.012" → integer part of the left plus the tail
        stem = left_raw.split(".")[0]
        right = digits_of(stem + right_raw)
    else:
        right = digits_of(right_raw)
    if not left or not right:
        return None
    return left, right


@dataclass
class SchemeEntry:
    """One shelf place: notation, wording, location."""

    notation: str
    label: str = ""
    location: str = ""
    group: str = ""
    see_also: str = ""
    # derived
    digits: str = ""
    auxiliary: str = ""
    span: Optional[Tuple[str, str]] = None
    tokens: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        out = {"notation": self.notation, "label": self.label}
        if self.location:
            out["location"] = self.location
        if self.group:
            out["group"] = self.group
        if self.see_also:
            out["see_also"] = self.see_also
        return out


def _tokenize(text: str) -> Tuple[str, ...]:
    words = re.findall(r"[\wäöüßÄÖÜ]{4,}", str(text or "").lower())
    return tuple(w for w in words if w not in _LABEL_STOPWORDS)


def _prepare(entry: SchemeEntry) -> SchemeEntry:
    entry.digits = digits_of(entry.notation)
    entry.auxiliary = _auxiliary_of(entry.notation)
    entry.span = _split_range(entry.notation)
    entry.tokens = _tokenize(entry.label)
    return entry


@register_lookup
class LocalSchemeLookup:
    """Shelf-place lookup over an institution's own classification vocabulary."""

    id = "local_scheme"
    label = "Lokale Aufstellungssystematik"

    def __init__(self, **config: Any):
        self._config = config or {}
        self._vocabulary = str(config.get("vocabulary") or "freihand_fg").strip()
        try:
            self._max_results = int(config.get("max_results") or 6)
        except (TypeError, ValueError):
            self._max_results = 6
        self._use_labels = bool(config.get("use_labels", True))
        self._entries: Optional[List[SchemeEntry]] = None
        self._meta: Dict[str, Any] = {}

    # ── plugin contract ────────────────────────────────────────────────────

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return [
            ConfigField(
                key="vocabulary",
                label="Vokabular (Id oder Pfad)",
                kind=TEXT,
                default="freihand_fg",
                help="Datei in vocabularies/ bzw. ~/.config/alima/vocabularies/, "
                "oder ein absoluter Pfad.",
            ),
            ConfigField(
                key="max_results",
                label="Max. Kandidaten",
                kind=INT,
                default=6,
            ),
            ConfigField(
                key="use_labels",
                label="Schlagworte gegen die Bezeichnungen prüfen",
                kind=BOOL,
                default=True,
                help="Zweites Signal neben der Notationsnähe. An 407 Freiberger "
                "Freihandtiteln gemessen hebt es die Trefferquote unter den "
                "ersten fünf um 1,5 Punkte (76,9 → 78,4 %) und senkt die auf "
                "Platz eins um 1,5 Punkte.",
            ),
        ]

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Aufstellungssystematik des eigenen Hauses als Vokabular: "
            "ordnet Sacherschließungs-Notationen die in Frage kommenden "
            "Standort-Notationen mit Bezeichnung und Standort zu. Entscheidet nicht, "
            "sondern liefert begründete Kandidaten.",
            input="Klassifikations-Notationen (z.B. DK-Codes) und/oder Schlagworte.",
            output="Gerankte Standort-Kandidaten mit Notation, Bezeichnung, Standort "
            "und Begründung (Notationsnähe bzw. Label-Treffer).",
        )

    @classmethod
    def mcp_tool_specs(cls) -> List[LookupToolSpec]:
        return [
            LookupToolSpec(
                name="shelf_lookup",
                description=(
                    "Find shelf places (Aufstellungssystematik) of the local library "
                    "for a work. Give the subject notations the work was classified "
                    "with and/or its subject keywords; returns ranked shelf "
                    "candidates with notation, wording, floor and the reason each "
                    "matched. The shelf notation need not equal the subject "
                    "notation. Use ONLY notations returned here."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "notations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Subject notations, e.g. ['DK 622.341', '553.4'].",
                            "default": [],
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Subject keywords of the work.",
                            "default": [],
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum shelf candidates.",
                            "default": 6,
                        },
                    },
                },
                method="lookup_shelf",
                cacheable=False,  # a local file read; caching would only add staleness
            ),
        ]

    # ── vocabulary ─────────────────────────────────────────────────────────

    def vocabulary_path(self) -> Optional[Path]:
        """Resolve the configured vocabulary to a file, or ``None``."""
        raw = self._vocabulary
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if candidate.is_absolute() or candidate.suffix:
            return candidate if candidate.is_file() else None
        for base in VOCABULARY_SEARCH_PATHS:
            path = base / f"{raw}.yaml"
            if path.is_file():
                return path
        return None

    def entries(self) -> List[SchemeEntry]:
        """Load the vocabulary once; an unreadable file is an empty scheme."""
        if self._entries is not None:
            return self._entries
        self._entries = []
        path = self.vocabulary_path()
        if path is None:
            logger.warning(
                f"local_scheme: vocabulary '{self._vocabulary}' not found in "
                f"{[str(p) for p in VOCABULARY_SEARCH_PATHS]}"
            )
            return self._entries
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError) as exc:
            logger.error(f"local_scheme: cannot read {path}: {exc}")
            return self._entries
        self._meta = dict(data.get("scheme") or {})
        self._meta.setdefault("path", str(path))
        for raw in data.get("entries") or []:
            if not isinstance(raw, dict):
                continue
            notation = str(raw.get("notation") or "").strip()
            if not notation:
                continue
            self._entries.append(_prepare(SchemeEntry(
                notation=notation,
                label=str(raw.get("label") or "").strip(),
                location=str(raw.get("location") or "").strip(),
                group=str(raw.get("group") or "").strip(),
                see_also=str(raw.get("see_also") or "").strip(),
            )))
        logger.info(f"local_scheme: {len(self._entries)} entries from {path}")
        return self._entries

    def scheme_info(self) -> Dict[str, Any]:
        """Provenance of the loaded vocabulary (id, label, source, retrieved).

        Scalars are stringified: YAML turns ``retrieved: 2026-09-09`` into a
        ``date``, which the tool result cannot JSON-serialise. - Claude Generated
        """
        self.entries()
        return {k: str(v) for k, v in self._meta.items()}

    # ── matching ───────────────────────────────────────────────────────────

    #: How many levels of pure ancestors a code contributes. Every code has a
    #: whole chain of them (622.3 → 622 → 62 → 6); beyond the nearest two they
    #: are noise that crowds out the second signal. An ancestor that also hits
    #: the keywords still enters through :meth:`_label_hits`.
    ANCESTOR_LEVELS = 2

    def _containment(self, code: str) -> List[Tuple[float, str, SchemeEntry]]:
        """Entries whose notation contains ``code``, with score and reason."""
        code_digits = digits_of(code)
        code_aux = _auxiliary_of(code)
        if not code_digits:
            return []
        hits: List[Tuple[float, str, SchemeEntry]] = []
        prefix_depths: Set[int] = set()
        for entry in self.entries():
            if not entry.digits:
                continue
            if entry.span:
                low, high = entry.span
                if _in_range(code_digits, low, high):
                    score = 40.0 + len(low)
                    reason = f"Bereich {entry.notation} enthält {code}"
                elif low.startswith(code_digits):
                    # The code sits ABOVE the range: DK 2 (Religion) against the
                    # shelf range 21/29. Measured against real Freiberg call
                    # numbers this is a genuine placement, just a broader one.
                    score = 35.0 + len(code_digits)
                    reason = f"Bereich {entry.notation} liegt unter {code}"
                else:
                    continue
            elif code_digits == entry.digits:
                score = 100.0
                reason = f"identisch mit {code}"
            elif code_digits.startswith(entry.digits):
                # Specificity decides: the deeper shelf entry wins. A group's
                # "622..." catch-all needs no penalty — a specific sibling
                # always carries more digits, and against its own plain twin
                # the stable sort settles it. - Claude Generated
                score = 50.0 + len(entry.digits) * 2
                reason = f"Oberstelle von {code}"
                prefix_depths.add(len(entry.digits))
            else:
                continue
            if entry.auxiliary:
                # (036) / :943 only really applies when the code carries it too
                if entry.auxiliary == code_aux:
                    score += 5.0
                else:
                    score -= 15.0
                    reason += f" (Anhängezahl {entry.auxiliary} ungeprüft)"
            hits.append((score, reason, entry))

        # Drop the far ancestors: keep the deepest ANCESTOR_LEVELS levels.
        if len(prefix_depths) > self.ANCESTOR_LEVELS:
            kept = set(sorted(prefix_depths, reverse=True)[: self.ANCESTOR_LEVELS])
            hits = [
                hit for hit in hits
                if not hit[1].startswith("Oberstelle") or len(hit[2].digits) in kept
            ]
        return hits

    def _ancestors(self, code: str) -> List[Tuple[float, str, SchemeEntry]]:
        """Nearest shelf places when nothing contains ``code``.

        Walks the code up digit by digit and returns the entries sitting under
        the first ancestor that exists in the scheme — the neighbourhood a
        cataloguer would look at.
        """
        code_digits = digits_of(code)
        hits: List[Tuple[float, str, SchemeEntry]] = []
        for cut in range(len(code_digits) - 1, 0, -1):
            stem = code_digits[:cut]
            near = [e for e in self.entries() if e.digits.startswith(stem)]
            if not near:
                continue
            for entry in near:
                score = 20.0 + cut * 2 - min(len(entry.digits) - cut, 6)
                hits.append((score, f"Nachbarstelle zu {code} (unter {stem})", entry))
            break
        return hits

    def _label_hits(self, keywords: Iterable[str]) -> List[Tuple[float, str, SchemeEntry]]:
        """Entries whose wording hits the work's keywords."""
        terms = set()
        for keyword in keywords or []:
            terms.update(_tokenize(keyword))
        if not terms:
            return []
        hits: List[Tuple[float, str, SchemeEntry]] = []
        for entry in self.entries():
            matched = sorted(terms.intersection(entry.tokens))
            if not matched:
                continue
            score = 25.0 + 10.0 * len(matched)
            hits.append((score, "Label-Treffer: " + ", ".join(matched), entry))
        return hits

    def lookup_shelf(
        self,
        notations: Optional[List[Any]] = None,
        keywords: Optional[List[Any]] = None,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Ranked shelf candidates for a set of notations and/or keywords.

        Never picks: the caller (a deterministic step, an LLM step, or a
        cataloguer) decides. ``unplaced`` names the notations for which the
        scheme holds no shelf place at all — a real finding, not an error.
        """
        limit = int(max_results or self._max_results or 6)
        codes = [_notation_text(n) for n in (notations or [])]
        codes = [c for c in codes if c]
        terms = [_notation_text(k) for k in (keywords or [])]
        terms = [t for t in terms if t]

        if not self.entries():
            return {
                "candidates": [],
                "unplaced": codes,
                "scheme": self.scheme_info(),
                "error": "Vokabular nicht geladen",
            }

        best: Dict[str, Tuple[float, List[str], SchemeEntry]] = {}

        def _offer(score: float, reason: str, entry: SchemeEntry) -> None:
            current = best.get(entry.notation)
            if current is None:
                best[entry.notation] = (score, [reason], entry)
                return
            # A second independent signal is worth more than either alone.
            merged_score = max(current[0], score) + 8.0
            best[entry.notation] = (merged_score, current[1] + [reason], entry)

        unplaced: List[str] = []
        for code in codes:
            contained = self._containment(code)
            if contained:
                for score, reason, entry in contained:
                    _offer(score, reason, entry)
            else:
                unplaced.append(code)
                for score, reason, entry in self._ancestors(code):
                    _offer(score, reason, entry)

        if self._use_labels:
            for score, reason, entry in self._label_hits(terms):
                _offer(score, reason, entry)

        ranked = sorted(
            best.values(), key=lambda item: (-item[0], item[2].digits, item[2].notation)
        )
        candidates = []
        for score, reasons, entry in ranked[:limit]:
            item = entry.as_dict()
            item["score"] = round(score, 1)
            item["reason"] = "; ".join(dict.fromkeys(reasons))
            candidates.append(item)

        return {
            "candidates": candidates,
            "unplaced": unplaced,
            "scheme": self.scheme_info(),
            "total_entries": len(self.entries()),
        }


def _notation_text(value: Any) -> str:
    """Accept ``"DK 622.3"``, ``{"code": …}`` or ``{"keyword": …}`` alike.

    The classification result carries dicts, an LLM passes back whatever it
    read; stringifying a dict silently produced unusable input in the RVK tool
    before (tool_registry.py), so the same shapes are unwrapped here.
    """
    if isinstance(value, dict):
        for key in ("code", "notation", "keyword", "title", "term"):
            if value.get(key):
                value = value[key]
                break
        else:
            return ""
    text = str(value or "").strip()
    for system in ("DK", "DDC", "RVK", "BK"):
        if text.upper().startswith(system + " "):
            return text[len(system) + 1:].strip()
    return text
