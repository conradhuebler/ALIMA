"""Entscheidungsmodell als Agenten-Werkzeug - Claude Generated.

Zwei Zugänge für denselben Endpunkt (``/v1/systemone``), die der ALIMA-Agent
(Chat in GUI, CLI und HTTP, sowie Workflow-Schritte mit dem Preset ``decision``)
aufrufen kann:

* ``decide``: beliebige typisierte Fragen (``noul`` / ``choice``) an einen State,
  Rückgabe sind die Wahrscheinlichkeiten. Für Ja/Nein- und Auswahlentscheidungen,
  die nicht erzeugt, sondern bewertet werden sollen.
* ``rank_by_relevance``: eine Kandidatenliste nach Passung zu einem Text ordnen.
  Derselbe Mechanismus wie der Pipeline-Vorfilter.

Rein funktional und Qt-frei; ``mcp/tool_registry`` hängt nur die Handler an. Die
Werkzeuge werden nur registriert, wenn die Rolle „Entscheidungsmodell“ gesetzt
ist (``decision_role``).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from .decision_client import (
    MAX_QUESTIONS,
    DecisionError,
    provider_base_url,
    score_items,
    systemone,
)
from .decision_prefilter import DEFAULT_MAX_STATE_CHARS, decision_role

#: Obergrenze für den State eines ``decide``-Aufrufs. Er wird für jede Frage
#: mitgezählt; größere Texte kürzt der Agent besser selbst.
MAX_DECIDE_STATE_CHARS = 6000
#: Obergrenze für ``rank_by_relevance``: rund 1000 Kandidaten brauchen auf dem
#: Server etwa 15 bis 25 s, auf einem Laptop bis zu zwei Minuten.
MAX_RANK_CANDIDATES = 3000
MAX_RANK_TOP_K = 200
DEFAULT_RANK_TOP_K = 30

_QUESTION_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_QUESTION_TYPES = ("noul", "choice")


def _role(unified_config: Any):
    role = decision_role(unified_config)
    if role is None:
        raise DecisionError(
            "Kein Entscheidungsmodell konfiguriert (Einstellungen ▸ Sprachmodelle ▸ "
            "Entscheidungsmodell)."
        )
    return role


def _validate_questions(questions: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(questions, dict) or not questions:
        raise DecisionError("`questions` muss ein nicht leeres Objekt {name: Frage} sein.")
    if len(questions) > MAX_QUESTIONS:
        raise DecisionError(
            f"{len(questions)} Fragen; höchstens {MAX_QUESTIONS} je Aufruf. "
            "In mehrere Aufrufe aufteilen."
        )
    out: Dict[str, Dict[str, Any]] = {}
    for name, q in questions.items():
        if not _QUESTION_NAME.match(str(name)):
            raise DecisionError(
                f"Fragename '{name}' ungültig: nur Buchstaben, Ziffern, _ . - (höchstens 100 Zeichen)."
            )
        if not isinstance(q, dict):
            raise DecisionError(f"Frage '{name}' muss ein Objekt sein.")
        qtype = q.get("type")
        if qtype not in _QUESTION_TYPES:
            raise DecisionError(f"Frage '{name}': type muss {' oder '.join(_QUESTION_TYPES)} sein.")
        if not str(q.get("instructions") or "").strip():
            raise DecisionError(f"Frage '{name}': `instructions` fehlt.")
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise DecisionError(
                f"Frage '{name}': `criteria` braucht mindestens zwei Einträge "
                "(noul: true/false; choice: je Option ein Name mit Beschreibung)."
            )
        if qtype == "noul" and set(criteria) != {"true", "false"}:
            raise DecisionError(f"Frage '{name}': noul braucht genau die Kriterien `true` und `false`.")
        out[str(name)] = {"type": qtype, "instructions": str(q["instructions"]),
                          "criteria": {str(k): str(v) for k, v in criteria.items()}}
    return out


def run_decide(unified_config: Any, state: Any, questions: Any) -> Dict[str, Any]:
    """Typisierte Fragen an einen State; ``{"model", "answers", "usage"}`` oder ``{"error"}``."""
    try:
        provider, model = _role(unified_config)
        qs = _validate_questions(questions)
        if isinstance(state, str):
            if not state.strip():
                raise DecisionError("`state` ist leer.")
            if len(state) > MAX_DECIDE_STATE_CHARS:
                raise DecisionError(
                    f"`state` hat {len(state)} Zeichen; höchstens {MAX_DECIDE_STATE_CHARS} "
                    "(er wird für jede Frage mitgezählt). Kürzen."
                )
        elif not isinstance(state, (dict, list)) or not state:
            raise DecisionError("`state` muss Text oder ein nicht leeres JSON-Objekt sein.")
        t0 = time.time()
        reply = systemone(
            provider_base_url(provider), model, state, qs,
            api_key=getattr(provider, "api_key", "") or "",
        )
    except DecisionError as e:
        return {"error": str(e)}
    return {
        "provider": provider.name,
        "model": reply.get("model", model),
        "answers": reply.get("answers", {}),
        "usage": reply.get("usage", {}),
        "seconds": round(time.time() - t0, 2),
    }


def _candidate_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("keyword") or item.get("title") or "").strip()
    return str(item).strip()


def run_rank(
    unified_config: Any, text: Any, candidates: Any, top_k: Any = DEFAULT_RANK_TOP_K
) -> Dict[str, Any]:
    """Kandidaten nach Passung zu ``text`` ordnen; die ``top_k`` besten mit Score."""
    try:
        provider, model = _role(unified_config)
        if not isinstance(text, str) or not text.strip():
            raise DecisionError("`text` fehlt.")
        if not isinstance(candidates, list) or not candidates:
            raise DecisionError(
                "`candidates` muss eine nicht leere Liste sein (oder `from_search_terms` angeben)."
            )
        if len(candidates) > MAX_RANK_CANDIDATES:
            raise DecisionError(
                f"{len(candidates)} Kandidaten; höchstens {MAX_RANK_CANDIDATES}. "
                "Vorher grob eingrenzen."
            )
        try:
            k = int(top_k)
        except (TypeError, ValueError):
            k = DEFAULT_RANK_TOP_K
        k = max(1, min(k, MAX_RANK_TOP_K))

        entries = [(c, _candidate_text(c)) for c in candidates]
        entries = [(c, t) for c, t in entries if t]
        if not entries:
            raise DecisionError("Keine Kandidaten mit Text.")
        texts = list(dict.fromkeys(t for _, t in entries))
        t0 = time.time()
        usage: Dict[str, int] = {}
        scores = score_items(
            provider_base_url(provider), model, text[:DEFAULT_MAX_STATE_CHARS], texts,
            api_key=getattr(provider, "api_key", "") or "",
            max_parallel=getattr(unified_config, "decision_max_parallel", 1),
            usage_sink=usage,
        )
    except DecisionError as e:
        return {"error": str(e)}

    seen: set = set()
    ranked: List[Dict[str, Any]] = []
    for c, t in sorted(entries, key=lambda e: -scores.get(e[1], 0.0)):
        if t in seen:
            continue
        seen.add(t)
        row = dict(c) if isinstance(c, dict) else {"keyword": t}
        row["score"] = round(scores.get(t, 0.0), 4)
        ranked.append(row)
    return {
        "provider": provider.name,
        "model": model,
        "n_in": len(texts),
        "n_out": min(k, len(ranked)),
        "seconds": round(time.time() - t0, 2),
        "input_tokens": usage.get("input_tokens", 0),
        "ranked": ranked[:k],
    }


_BARE_ID = re.compile(r"^[0-9]+(?:-[0-9Xx])?$")


def pool_candidates(aggregate: Any) -> tuple:
    """Kandidaten aus dem Ergebnis von ``aggregate_gnd_results`` - Claude Generated

    ``({keyword, gnd_id, count}, anzahl_übersprungen)``. Pool-Einträge,
    deren "Titel" nur die GND-ID ist (die Quelle lieferte kein Label), werden nicht
    bewertet: das Modell sähe nur eine Nummer, und der Score wäre Rauschen. Fehler
    und leerer Pool werden zu ``DecisionError`` mit einer Meldung, die dem Agenten
    sagt, was fehlt.
    """
    if not isinstance(aggregate, dict):
        raise DecisionError("Pool nicht lesbar.")
    if aggregate.get("error"):
        raise DecisionError(str(aggregate["error"]))
    pool = aggregate.get("pool") or []
    if not pool:
        missing = aggregate.get("missing") or {}
        hint = " Erst die search_*-Werkzeuge für diese Begriffe aufrufen." if missing else ""
        raise DecisionError("Der Pool zu diesen Suchbegriffen ist leer." + hint)
    out: List[Dict[str, Any]] = []
    skipped = 0
    for entry in pool:
        title = str(entry.get("title") or entry.get("keyword") or "").strip()
        if not title:
            continue
        if _BARE_ID.match(title):
            skipped += 1
            continue
        ids = entry.get("gnd_ids") or []
        # Only the pool ``count`` (the selection's own ranking input). The catalogue
        # frequency (``display_count``) is deliberately left out: select_from_gnd_pool
        # sorts by ``count`` and tells the model to prefer high counts, so handing
        # the agent a frequency to copy there put "Krankheit" (7692 hits) and
        # "Medizin" first, ahead of the terms the ranking had put on top.
        out.append({"keyword": title, "gnd_id": str(ids[0]) if ids else "",
                    "count": entry.get("count", 1)})
    if not out:
        raise DecisionError("Der Pool enthält nur Einträge ohne Bezeichnung (nur GND-IDs).")
    return out, skipped
