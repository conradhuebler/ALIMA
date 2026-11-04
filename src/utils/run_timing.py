"""Zeit eines Pipeline-Laufs nach Art der Arbeit - Claude Generated.

Ein Lauf besteht aus Wartezeit auf Sprachmodelle (``llm``), auf das
Entscheidungsmodell (``decision``) und auf Abrufe aus Datenbank, Katalog und Web
(``retrieval``); der Rest (Aufbereitung, Parsen, Oberfläche) ergibt sich als
Differenz zur Gesamtzeit.

Gezählt wird **ausschließlich**: Ruft ein Abruf-Werkzeug ein Sprachmodell auf
(z. B. ``select_from_gnd_pool``), zählt die Zeit des Modells zu ``llm`` und nur der
Rest zu ``retrieval``. Das gilt je Thread. Läuft das Entscheidungsmodell mit
mehreren gleichzeitigen Anfragen, ist die Summe der Aufrufzeiten größer als die
Wanduhr; das steht dann im Ergebnis (``parallel``).

Es gibt einen globalen Zähler, also einen Lauf zugleich (GUI, CLI und die Webapp
fahren ihre Läufe nacheinander je Prozess; mehrere gleichzeitige Webapp-Sitzungen
würden sich vermischen).
"""

from __future__ import annotations

import functools
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List

CATEGORIES = ("llm", "decision", "retrieval")

_lock = threading.Lock()
_totals: Dict[str, float] = {c: 0.0 for c in CATEGORIES}
_counts: Dict[str, int] = {c: 0 for c in CATEGORIES}
_t0: float = time.perf_counter()
_threads: set = set()
_local = threading.local()


def begin_run() -> None:
    """Zähler auf null und Uhr starten (Beginn eines Pipeline-Laufs)."""
    global _t0
    with _lock:
        for c in CATEGORIES:
            _totals[c] = 0.0
            _counts[c] = 0
        _threads.clear()
        _t0 = time.perf_counter()


def _stack() -> List[List[float]]:
    st = getattr(_local, "stack", None)
    if st is None:
        st = _local.stack = []
    return st


@contextmanager
def timed(category: str) -> Iterator[None]:
    """Misst den Block und rechnet die Zeit ausschließlich der Kategorie zu."""
    if category not in CATEGORIES:
        raise ValueError(f"unbekannte Kategorie '{category}'")
    st = _stack()
    child = [0.0]
    st.append(child)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        st.pop()
        if st:
            st[-1][0] += elapsed            # der äußere Block schließt diese Zeit aus
        with _lock:
            _totals[category] += max(0.0, elapsed - child[0])
            _counts[category] += 1
            _threads.add(threading.get_ident())


def timed_method(category: str) -> Callable:
    """Dekorator: ganze Methode als ``timed(category)``."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed(category):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def finish_run() -> Dict[str, Any]:
    """Zusammenfassung seit ``begin_run`` (Sekunden, auf 0,1 gerundet)."""
    with _lock:
        total = time.perf_counter() - _t0
        parts = dict(_totals)
        counts = dict(_counts)
        threads = len(_threads)
    known = sum(parts.values())
    return {
        "total_s": round(total, 1),
        "llm_s": round(parts["llm"], 1),
        "decision_s": round(parts["decision"], 1),
        "retrieval_s": round(parts["retrieval"], 1),
        "other_s": round(max(0.0, total - known), 1),
        "calls": counts,
        "parallel": known > total + 0.5,
    }


def format_summary(s: Dict[str, Any]) -> str:
    """Eine Zeile für Anzeige und Log."""
    c = s.get("calls", {})
    text = (
        f"⏱ Gesamt {s['total_s']} s: LLM {s['llm_s']} s ({c.get('llm', 0)} Aufrufe), "
        f"Entscheidung {s['decision_s']} s ({c.get('decision', 0)}), "
        f"Abruf {s['retrieval_s']} s ({c.get('retrieval', 0)}), Sonstiges {s['other_s']} s"
    )
    if s.get("parallel"):
        text += " (parallele Anfragen: Summe der Aufrufzeiten übersteigt die Laufzeit)"
    return text
