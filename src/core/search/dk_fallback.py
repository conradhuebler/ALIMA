"""DK step over several sources: fallback per keyword - Claude Generated.

``factory.resolve_dk_extractor`` returns a :class:`FallbackDkExtractor` when
more than one DK source is usable (``factory.dk_candidates``, in the order of
the plugin list). It keeps the contract of a single extractor
(``extract_dk_classifications_for_keywords``) so the DK step is unchanged
around it. With exactly one source the bare extractor of that source is used
instead; this class is then not involved.

Rule: the first source is asked for every keyword. A keyword counts as not
answered, and goes on to the next source, when

* the source was down (``source_health``, extractor list ``last_source_down``),
* the state "Nur Cache" had no entry (``last_cache_misses``),
* the institution was not set (``last_not_configured``) and nothing came back,
* the extractor of the source could not be built,
* the request failed to reach the server (a connection failure counted by
  ``source_health`` during the call, below the skip threshold),
* the source's own circuit breaker was open (``BiblioClient``),
* or the extractor raised.

"No hits" is an answer and is not handed on. Each keyword is answered by one
source; nothing is merged within a keyword. Keywords a source left open are
collected and handed to the next source; the counts per hand-over are kept for
one message each (:meth:`FallbackDkExtractor.fallback_notes`), and which source
answered which keyword in :attr:`FallbackDkExtractor.answered_by`.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Why a keyword went on to the next source.
REASON_SOURCE_DOWN = "source_down"
REASON_CACHE_MISS = "cache_miss"
REASON_NOT_SET = "institution_not_set"
REASON_INIT_FAILED = "init_failed"
REASON_CONNECTION = "connection"
REASON_CIRCUIT_BREAKER = "circuit_breaker"
REASON_ERROR = "error"

_REASON_TEXT = {
    REASON_SOURCE_DOWN: "gesperrt",
    REASON_CACHE_MISS: "nicht im Cache",
    REASON_NOT_SET: "Einrichtung nicht gesetzt",
    REASON_INIT_FAILED: "Quelle nicht startbar",
    REASON_CONNECTION: "Verbindungsfehler",
    REASON_CIRCUIT_BREAKER: "Server-Sperre des Clients",
    REASON_ERROR: "Fehler",
}


@dataclass
class DkCandidate:
    """One source of the DK chain - Claude Generated"""

    source_id: str
    label: str
    provider_id: str = ""
    extractor: Any = None
    init_error: str = ""
    cache_only: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)


def chain_text(labels: List[str]) -> str:
    """``"DK-Quellen: A, dann B (Rückfall)"`` / ``"DK-Quelle: A"`` - Claude Generated"""
    labels = [str(label) for label in labels if label]
    if not labels:
        return "DK-Quelle: keine"
    if len(labels) == 1:
        return f"DK-Quelle: {labels[0]}"
    return "DK-Quellen: " + ", dann ".join(labels) + " (Rückfall)"


class FallbackDkExtractor:
    """Several DK extractors behind one ``extract_dk_classifications_for_keywords``.

    Per call, ``last_cache_misses`` / ``last_source_down`` /
    ``last_not_configured`` / ``last_errors`` hold the keywords that no source
    answered, under the reason the last source gave. ``answered_by``
    (keyword → source id) and the hand-over counts accumulate over the calls of
    one DK step. - Claude Generated
    """

    def __init__(self, candidates: List[DkCandidate]):
        if len(candidates) < 2:
            raise ValueError("FallbackDkExtractor needs at least two sources")
        self.candidates: List[DkCandidate] = list(candidates)
        self.answered_by: Dict[str, str] = {}
        self.unanswered: Dict[str, Tuple[str, str]] = {}
        # (from source id, to source id) → {reason: count}, in first-seen order.
        self._handoffs: Dict[Tuple[str, str], Dict[str, int]] = {}
        self.last_cache_misses: List[str] = []
        self.last_source_down: List[str] = []
        self.last_not_configured: List[str] = []
        self.last_errors: Dict[str, str] = {}
        self._accepts_force: Dict[int, bool] = {}

    # --- identity (read by the DK step and the tests) ---------------------
    @property
    def source_id(self) -> str:
        """The first source of the chain - Claude Generated"""
        return self.candidates[0].source_id

    @property
    def source_ids(self) -> List[str]:
        return [c.source_id for c in self.candidates]

    @property
    def cache_only(self) -> bool:
        """True only when every source of the chain is in the state "Nur Cache"."""
        return all(c.cache_only for c in self.candidates)

    def label_of(self, source_id: str) -> str:
        for cand in self.candidates:
            if cand.source_id == source_id:
                return cand.label
        return source_id

    def chain_text(self) -> str:
        return chain_text([c.label for c in self.candidates])

    # --- the contract -------------------------------------------------------
    def extract_dk_classifications_for_keywords(
        self, keywords: List[str], max_results: int = 50, force_update: bool = False,
    ) -> List[Dict[str, Any]]:
        self.last_cache_misses = []
        self.last_source_down = []
        self.last_not_configured = []
        self.last_errors = {}
        pending: List[str] = []
        for keyword in keywords or []:
            if keyword not in pending:
                pending.append(keyword)
        order = list(pending)
        answers: Dict[str, List[Dict[str, Any]]] = {}
        reasons: Dict[str, str] = {}
        for index, cand in enumerate(self.candidates):
            if not pending:
                break
            left: List[str] = []
            for keyword in pending:
                answered, result, reason = self._ask(cand, keyword, max_results, force_update)
                if answered:
                    answers[keyword] = result
                    self.answered_by[keyword] = cand.source_id
                    self.unanswered.pop(keyword, None)
                else:
                    left.append(keyword)
                    reasons[keyword] = reason
            if left and index + 1 < len(self.candidates):
                counts = self._handoffs.setdefault(
                    (cand.source_id, self.candidates[index + 1].source_id), {})
                for keyword in left:
                    counts[reasons[keyword]] = counts.get(reasons[keyword], 0) + 1
            pending = left
        last = self.candidates[-1]
        for keyword in pending:
            reason = reasons.get(keyword, REASON_ERROR)
            self.unanswered[keyword] = (last.source_id, reason)
            if reason == REASON_SOURCE_DOWN:
                self.last_source_down.append(keyword)
            elif reason == REASON_CACHE_MISS:
                self.last_cache_misses.append(keyword)
            elif reason == REASON_NOT_SET:
                self.last_not_configured.append(keyword)
            else:
                self.last_errors[keyword] = f"{last.label}: {_REASON_TEXT.get(reason, reason)}"
        out: List[Dict[str, Any]] = []
        for keyword in order:
            out.extend(answers.get(keyword) or [])
        return out

    def _ask(self, cand: DkCandidate, keyword: str, max_results: int,
             force_update: bool) -> Tuple[bool, List[Dict[str, Any]], str]:
        """``(answered, result, reason)`` of one source for one keyword - Claude Generated"""
        from src.core.search import source_health

        ext = cand.extractor
        if ext is None:
            return False, [], REASON_INIT_FAILED
        breaker = getattr(ext, "get_circuit_breaker_status", None)
        if callable(breaker):
            try:
                if (breaker() or {}).get("open"):
                    return False, [], REASON_CIRCUIT_BREAKER
            except Exception:  # noqa: BLE001 - a broken status must not stop the step
                logger.debug("circuit breaker status of %s unreadable", cand.source_id,
                             exc_info=True)
        kwargs: Dict[str, Any] = {"keywords": [keyword], "max_results": max_results}
        if self._force_supported(ext):
            kwargs["force_update"] = force_update
        mark = source_health.failure_mark(cand.source_id)
        try:
            result = list(ext.extract_dk_classifications_for_keywords(**kwargs) or [])
        except Exception as exc:  # noqa: BLE001 - one source failing hands on to the next
            from src.utils.error_visibility import log_caught

            log_caught(logger, exc, "DK source", detail=f"{cand.source_id}: {keyword}")
            return False, [], REASON_ERROR
        if any(isinstance(e, dict) and e.get("classifications") for e in result):
            return True, result, ""
        if getattr(ext, "last_source_down", None):
            return False, [], REASON_SOURCE_DOWN
        if getattr(ext, "cache_only", False) is True and getattr(ext, "last_cache_misses", None):
            return False, [], REASON_CACHE_MISS
        if getattr(ext, "last_not_configured", None):
            return False, [], REASON_NOT_SET
        after = source_health.failure_mark(cand.source_id)
        if after is not None and after != mark:
            return False, [], REASON_CONNECTION
        return True, result, ""

    def _force_supported(self, ext: Any) -> bool:
        key = id(ext)
        if key not in self._accepts_force:
            try:
                self._accepts_force[key] = "force_update" in inspect.signature(
                    ext.extract_dk_classifications_for_keywords).parameters
            except (TypeError, ValueError):
                self._accepts_force[key] = False
        return self._accepts_force[key]

    # --- reporting ------------------------------------------------------------
    def handoffs(self) -> List[Dict[str, Any]]:
        """``[{from, to, keywords, reasons}]`` per hand-over, in order - Claude Generated"""
        return [
            {"from": src, "to": dst, "keywords": sum(counts.values()), "reasons": dict(counts)}
            for (src, dst), counts in self._handoffs.items()
        ]

    def fallback_notes(self) -> List[str]:
        """One line per hand-over, e.g. ``"finc: 14 Schlagwörter nicht
        beantwortet (gesperrt), weiter mit sru"`` - Claude Generated"""
        notes = []
        for item in self.handoffs():
            n = item["keywords"]
            noun = "Schlagwort" if n == 1 else "Schlagwörter"
            why = ", ".join(
                f"{_REASON_TEXT.get(reason, reason)}" + (f" {count}" if len(item["reasons"]) > 1 else "")
                for reason, count in item["reasons"].items()
            )
            notes.append(
                f"{self.label_of(item['from'])}: {n} {noun} nicht beantwortet ({why}), "
                f"weiter mit {self.label_of(item['to'])}"
            )
        return notes
