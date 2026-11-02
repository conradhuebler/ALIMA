"""Vorfilter für große GND-Kandidatenlisten per Entscheidungsmodell - Claude Generated.

Ein Pool von rund 1000 Kandidaten wird sonst in zwei bis drei LLM-Aufrufen mit
je 500 Einträgen auf wenige Dutzend gekürzt. Hier bewertet stattdessen das
konfigurierte Entscheidungsmodell (``ProviderScope.DECISION``) jeden Kandidaten
gegen den Text und nur die ``top_k`` besten gehen in die bestehende Auswahl.

Die Funktion gibt ``None`` zurück, wenn nichts zu tun ist oder es nicht ging
(Rolle nicht gesetzt, Pool klein genug, Host nicht erreichbar). Der Aufrufer
läuft dann unverändert weiter; ein Fehler wird gemeldet, nie verschluckt. Die
Messwerte (Scores je Kandidat) stehen im ``record``, damit sich der Nutzen
später ohne neue Abfragen nachrechnen lässt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .decision_client import DecisionError, provider_base_url, score_items
from .error_visibility import log_caught

logger = logging.getLogger(__name__)

#: Obergrenze für den State je Frage. Der Endpunkt zählt den State für jede
#: Frage mit; ein ganzer Buchabstract verdreifacht die Kosten (gemessen).
DEFAULT_MAX_STATE_CHARS = 1500
DEFAULT_TOP_K = 80


@dataclass
class PrefilterOutcome:
    kept: List[Any]
    record: Dict[str, Any] = field(default_factory=dict)


def decision_role(unified_config: Any) -> Optional[tuple]:
    """``(provider_obj, model)`` der Rolle oder ``None`` wenn sie aus ist."""
    if unified_config is None:
        return None
    name, model, _source = unified_config.explain_default_provider_model(scope="decision")
    if not name or not model:
        return None
    provider = unified_config.get_provider_by_name(name)
    return (provider, model) if provider is not None else None


def prefilter_by_decision(
    items: Sequence[Any],
    text_of: Callable[[Any], str],
    state: str,
    unified_config: Any,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    should_stop: Optional[Callable[[], bool]] = None,
    notify: Optional[Callable[[str], None]] = None,
) -> Optional[PrefilterOutcome]:
    """Die ``top_k`` relevantesten ``items`` in ursprünglicher Reihenfolge.

    ``text_of`` liefert den Begriff, der dem Modell vorgelegt wird. Kein
    Rückfall auf ein anderes Modell, wenn die Rolle fehlt: dann ``None``.
    """
    items = list(items)
    top_k = int(top_k or 0)
    if top_k <= 0 or len(items) <= top_k:
        return None
    role = decision_role(unified_config)
    if role is None:
        return None
    provider, model = role
    say = notify or (lambda _msg: None)
    texts = [text_of(it) for it in items]
    state = (state or "")[:max_state_chars]
    t0 = time.time()
    try:
        scores = score_items(
            provider_base_url(provider), model, state, texts,
            api_key=getattr(provider, "api_key", "") or "",
            max_parallel=getattr(unified_config, "decision_max_parallel", 1),
            should_stop=should_stop,
        )
    except DecisionError as e:
        logger.warning("Entscheidungs-Vorfilter fehlgeschlagen (%s/%s): %s", provider.name, model, e)
        say(f"⚠️ Entscheidungsmodell {provider.name}/{model} nicht nutzbar ({e}). "
            "Weiter mit der bisherigen Auswahl.\n")
        return None
    except Exception as e:  # noqa: BLE001 - Rückfall auf den bisherigen Pfad
        log_caught(logger, e, f"Entscheidungs-Vorfilter {provider.name}/{model}")
        say(f"⚠️ Entscheidungsmodell {provider.name}/{model}: {type(e).__name__}. "
            "Weiter mit der bisherigen Auswahl.\n")
        return None

    # Gleichstand bricht die ursprüngliche Position (stabil, wiederholbar).
    order = sorted(range(len(items)), key=lambda i: (-scores.get(texts[i], 0.0), i))
    keep_idx = sorted(order[:top_k])
    kept = [items[i] for i in keep_idx]
    record = {
        "provider": provider.name,
        "model": model,
        "top_k": top_k,
        "n_in": len(items),
        "n_out": len(kept),
        "seconds": round(time.time() - t0, 2),
        "state_chars": len(state),
        "max_parallel": getattr(unified_config, "decision_max_parallel", 1),
        "scores": {t: round(scores.get(t, 0.0), 4) for t in texts},
    }
    say(f"🎯 Entscheidungsmodell {provider.name}/{model}: {len(items)} → {len(kept)} "
        f"Kandidaten ({record['seconds']} s)\n")
    return PrefilterOutcome(kept=kept, record=record)
