"""Client für Ollama-Entscheidungsmodelle (``POST /v1/systemone``) - Claude Generated.

Ein Entscheidungsmodell erzeugt keinen Text: es bekommt einen ``state`` und
benannte, typisierte Fragen und liefert je Frage eine Wahrscheinlichkeit aus
einem einzigen Forward-Pass. Das passt weder in die Chat- noch in die
Generate-Pfade von ``LlmService`` (``/v1/chat/completions`` lehnt solche Modelle
ab), deshalb ein eigener, Qt-freier Client.

Grenzen des Endpunkts, gemessen (siehe ``docs/decision_models.md``):
  * 1 bis ``MAX_QUESTIONS`` (64) Fragen je Aufruf, mehr ergibt HTTP 400.
  * ``input_tokens`` zählt den State je Frage mit. Kosten und Laufzeit wachsen
    also mit Fragenzahl × (State + Frage); bei bezahlten Anbietern den State
    kurz halten.
  * Hinter ``OLLAMA_MAX_QUEUE`` antwortet der Server mit 503.

Nur Ollama-Anbieter (``provider_type == "ollama"``) sprechen diesen Endpunkt.
"""

from __future__ import annotations

import html
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from .ollama_endpoint import base_url as _ollama_base_url
from .ollama_endpoint import clean_api_key, normalize_ollama_cloud_url

logger = logging.getLogger(__name__)

MAX_QUESTIONS = 64
_RETRY_STATUS = (429, 503)
_RETRY_DELAYS = (0.5, 1.5, 4.0)

# Deutsche ``noul``-Frage; unter den getesteten Formulierungen die mit dem
# besten Ergebnis für ``d1``, bei einem Dokument gemessen und daher nur eine
# Voreinstellung.
DEFAULT_INSTRUCTIONS = (
    "Ist das Schlagwort '{t}' ein geeignetes Sacherschließungs-Schlagwort "
    "für das Dokument im State?"
)
DEFAULT_TRUE = "Das Schlagwort '{t}' benennt ein Thema, um das es im Dokument geht."
DEFAULT_FALSE = "Das Schlagwort '{t}' hat mit dem Dokument nichts oder nur am Rande zu tun."


class DecisionError(RuntimeError):
    """Der Entscheidungs-Aufruf ist gescheitert (Verbindung, Status, Antwortform)."""


def provider_base_url(provider: Any) -> str:
    """Basis-URL eines Ollama-``UnifiedProvider`` - Claude Generated.

    Dieselbe Lesart wie ``LlmService._init_ollama_native_provider``: ein
    schemaloser Host mit Punkt, der keine IP ist, meint HTTPS.
    """
    if getattr(provider, "provider_type", "") != "ollama":
        raise DecisionError(
            f"Anbieter '{getattr(provider, 'name', '?')}' ist kein Ollama-Anbieter "
            f"(Typ {getattr(provider, 'provider_type', '?')}); "
            "Entscheidungsmodelle laufen über den nativen Ollama-Endpunkt."
        )
    raw = getattr(provider, "base_url", "") or ""
    if raw:
        url = raw
        if "://" not in url:
            host_part = url.split("/", 1)[0].split(":", 1)[0]
            if "." in host_part and not host_part.replace(".", "").isdigit():
                url = f"https://{url}"
    elif getattr(provider, "host", ""):
        url = _ollama_base_url(
            provider.host, getattr(provider, "port", 11434), getattr(provider, "use_ssl", False)
        )
    else:
        url = "http://localhost:11434"
    return normalize_ollama_cloud_url(url).rstrip("/")


def _headers(api_key: str) -> Dict[str, str]:
    key = clean_api_key(api_key or "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def list_decision_models(provider: Any, timeout: float = 5.0) -> List[str]:
    """Namen der Entscheidungsmodelle eines Ollama-Anbieters - Claude Generated.

    ``/api/tags`` nennt je Modell seine ``capabilities``; ein Entscheidungsmodell
    trägt ``decision``. ``/v1/models`` unterscheidet das nicht und listet sie
    neben den Chat-Modellen, die ``/v1/systemone`` nicht bedienen. Nicht
    erreichbar oder anderer Anbietertyp: leere Liste (die Auswahl bleibt leer,
    statt Chat-Modelle anzubieten).
    """
    try:
        url = provider_base_url(provider)
        resp = requests.get(f"{url}/api/tags",
                            headers=_headers(getattr(provider, "api_key", "") or ""),
                            timeout=timeout)
        resp.raise_for_status()
        models = resp.json().get("models") or []
    except (DecisionError, requests.RequestException, ValueError) as e:
        logger.info("Entscheidungsmodelle von %s nicht lesbar: %s", getattr(provider, "name", "?"), e)
        return []
    return sorted(
        m.get("name") or m.get("model") or ""
        for m in models if "decision" in (m.get("capabilities") or [])
    )


def systemone(
    base_url: str,
    model: str,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    *,
    api_key: str = "",
    timeout: float = 120.0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Ein ``/v1/systemone``-Aufruf; 429/503 und Verbindungsfehler werden wiederholt."""
    if not 1 <= len(questions) <= MAX_QUESTIONS:
        raise DecisionError(f"{len(questions)} Fragen; erlaubt sind 1 bis {MAX_QUESTIONS}")
    http = session or requests
    url = f"{base_url}/v1/systemone"
    body = {"model": model, "state": state, "questions": questions}
    last: Optional[str] = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            resp = http.post(url, json=body, headers=_headers(api_key),
                             timeout=(5.0, timeout))
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise DecisionError(f"Antwort von {url} ist kein JSON: {e}") from e
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in _RETRY_STATUS:
                raise DecisionError(f"{url}: {last}")
        if attempt < len(_RETRY_DELAYS):
            time.sleep(_RETRY_DELAYS[attempt])
    raise DecisionError(f"{url} nicht erreichbar nach {len(_RETRY_DELAYS) + 1} Versuchen ({last})")


def relevance_questions(
    items: Sequence[str],
    *,
    instructions: str = DEFAULT_INSTRUCTIONS,
    true_text: str = DEFAULT_TRUE,
    false_text: str = DEFAULT_FALSE,
) -> Dict[str, Dict[str, Any]]:
    """Je Kandidat eine ``noul``-Frage (Namen ``q0``…, nur Buchstaben und Ziffern).

    Der Begriff geht HTML-entschärft in den Text: Pool-Titel tragen teils
    ``&lt;Motiv&gt;`` statt ``<Motiv>``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(items):
        t = html.unescape(str(item))
        out[f"q{i}"] = {
            "type": "noul",
            "instructions": instructions.format(t=t),
            "criteria": {"true": true_text.format(t=t), "false": false_text.format(t=t)},
        }
    return out


def score_items(
    base_url: str,
    model: str,
    state: str,
    items: Sequence[str],
    *,
    api_key: str = "",
    max_parallel: int = 1,
    timeout: float = 120.0,
    should_stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    usage_sink: Optional[Dict[str, int]] = None,
    **question_texts: str,
) -> Dict[str, float]:
    """Relevanz-Score (0 bis 1) je Kandidat; ``{item: P(true)}``.

    ``usage_sink`` (optional) sammelt die vom Host gemeldeten ``input_tokens``,
    damit sich der Aufwand bei bezahlten Anbietern beziffern lässt.

    Aufteilung in Batches zu ``MAX_QUESTIONS``. Bei Abbruch (``should_stop``)
    oder einem Fehler in einem Batch wird ``DecisionError`` geworfen: ein
    halber Score wäre eine Auswahl nach Reihenfolge der Batches, nicht nach
    Relevanz.
    """
    items = list(items)
    batches = [items[i:i + MAX_QUESTIONS] for i in range(0, len(items), MAX_QUESTIONS)]
    scores: Dict[str, float] = {}
    done = [0]
    usage_lock = threading.Lock()
    session = requests.Session()

    def run(batch: List[str]) -> List[float]:
        if should_stop is not None and should_stop():
            raise DecisionError("abgebrochen")
        try:
            reply = systemone(
                base_url, model, state, relevance_questions(batch, **question_texts),
                api_key=api_key, timeout=timeout, session=session,
            )
        except DecisionError as e:
            # The host counts the state for every question, so 64 questions with a
            # long state can exceed the model's context (HTTP 500 "Context size has
            # been exceeded"). Halve the batch instead of giving up. - Claude Generated
            if "context size" in str(e).lower() and len(batch) > 1:
                half = len(batch) // 2
                return run(batch[:half]) + run(batch[half:])
            raise
        if usage_sink is not None:
            with usage_lock:
                usage_sink["input_tokens"] = usage_sink.get("input_tokens", 0) + int(
                    (reply.get("usage") or {}).get("input_tokens", 0) or 0
                )
        answers = reply.get("answers") or {}
        try:
            values = [float(answers[f"q{i}"]["noul"]) for i in range(len(batch))]
        except (KeyError, TypeError, ValueError) as e:
            raise DecisionError(f"Antwort ohne erwartete 'noul'-Werte: {e!r}") from e
        done[0] += len(batch)
        if on_progress is not None:
            on_progress(done[0], len(items))
        return values

    workers = max(1, min(int(max_parallel or 1), len(batches) or 1))
    try:
        if workers == 1:
            results = [run(b) for b in batches]
        else:
            with ThreadPoolExecutor(workers) as pool:
                results = list(pool.map(run, batches))
    finally:
        session.close()
    for batch, values in zip(batches, results):
        scores.update(zip(batch, values))
    return scores
