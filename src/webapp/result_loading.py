"""Load a saved ALIMA result into a webapp session - Claude Generated.

Two file formats are accepted, the two ALIMA writes:

* the webapp/API export (``GET /api/export/{id}``, ``export_analysis_state_to_file``,
  the CLI JSON export): an object with a ``results`` object;
* a saved ``KeywordAnalysisState`` (GUI "Speichern", CLI, autosave): an object
  with the state's fields at the top level.

Both go through ``PipelineJsonManager.analysis_state_from_dict``, the loader the
GUI and the CLI use, and the result is rendered by ``render_pipeline_result``
into the session's render log, so the cards are the ones a run shows.

The file is untrusted data: it is size-capped, must be UTF-8 JSON with an
object at the top, its text only ever reaches the page through the escaping
formatters, and no path it names is opened. Every rejection is a
:class:`ResultFileError` with a message meant for the person who picked the
file. Qt-free apart from the renderer, which is imported lazily.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Largest accepted file. Real result files are well below this (the largest
#: part, the catalogue titles of the DK search, stays in the low megabytes).
MAX_RESULT_FILE_BYTES = 20 * 1024 * 1024
MAX_RESULT_FILE_MB = MAX_RESULT_FILE_BYTES // (1024 * 1024)

FORMAT_WEBAPP_EXPORT = "webapp_export"
FORMAT_ANALYSIS_STATE = "analysis_state"

#: Fields of ``KeywordAnalysisState`` without a default: a saved state has them.
_STATE_REQUIRED = ("original_abstract", "initial_keywords", "search_suggesters_used")

#: Card kinds that make a loaded file worth showing (see render_pipeline_result).
_RESULT_CARDS = frozenset({"final_result", "dk_classifications", "dk_statistics", "workflow_report"})


class ResultFileError(ValueError):
    """A file that cannot be loaded; ``status`` is the HTTP status to answer."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message = message
        self.status = status


def display_filename(name: Any) -> str:
    """The picked file's name for display: base name, no control characters,
    at most 120 characters. Never used as a path. - Claude Generated"""
    text = str(name or "")
    text = re.split(r"[\\/]", text)[-1]
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C").strip()
    if len(text) > 120:
        text = text[:117] + "…"
    return text or "ergebnis.json"


def parse_result_file(raw: bytes) -> Tuple[Dict[str, Any], str]:
    """``(data, format)`` of an uploaded result file, or :class:`ResultFileError`."""
    if len(raw) > MAX_RESULT_FILE_BYTES:
        raise ResultFileError(
            f"Die Datei ist größer als {MAX_RESULT_FILE_MB} MB und wird nicht geladen.", status=413
        )
    if not raw.strip():
        raise ResultFileError("Die Datei ist leer.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ResultFileError("Die Datei ist keine JSON-Datei (kein UTF-8-Text).")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResultFileError(
            f"Die Datei ist kein gültiges JSON (Zeile {exc.lineno}, Spalte {exc.colno})."
        )
    except RecursionError:
        raise ResultFileError("Die Datei ist zu tief verschachtelt, um ein ALIMA-Ergebnis zu sein.")
    if not isinstance(data, dict):
        raise ResultFileError("Die Datei enthält kein JSON-Objekt und ist kein ALIMA-Ergebnis.")
    if isinstance(data.get("results"), dict):
        return data, FORMAT_WEBAPP_EXPORT
    if all(key in data for key in _STATE_REQUIRED):
        return data, FORMAT_ANALYSIS_STATE
    raise ResultFileError(
        "Die Datei ist kein ALIMA-Ergebnis. Geladen werden ein Export der Webapp "
        "(„Speichern“) oder ein in GUI oder CLI gespeicherter Analysezustand."
    )


def _prepare(data: Dict[str, Any], fmt: str) -> Tuple[Dict[str, Any], List[str]]:
    """A copy the shared loader accepts, and the top-level state fields left out.

    * Export: ``classifications`` (the structured entries with their core/
      additional rank) becomes ``classification_entries`` when the file has
      none; the shared loader keeps only the flat ``dk_classifications``
      strings, and the cards would lose the rank.
    * State: fields this version does not know are left out (named in the
      answer) instead of failing the whole file.
    """
    from src.core.data_models import KeywordAnalysisState

    data = copy.deepcopy(data)
    ignored: List[str] = []
    if fmt == FORMAT_WEBAPP_EXPORT:
        data.setdefault("session_id", None)
        results = data["results"]
        structured = results.get("classifications")
        if (
            not results.get("classification_entries")
            and isinstance(structured, list)
            and structured
            and all(isinstance(entry, dict) for entry in structured)
        ):
            results["classification_entries"] = structured
    else:
        known = {f.name for f in dataclasses.fields(KeywordAnalysisState)}
        ignored = sorted(key for key in data if key not in known)
        for key in ignored:
            data.pop(key)
    return data, ignored


def state_from_result_data(data: Dict[str, Any], fmt: str):
    """``(KeywordAnalysisState, ignored_fields)`` or :class:`ResultFileError`."""
    from src.utils.pipeline_persistence import PipelineJsonManager

    prepared, ignored = _prepare(data, fmt)
    try:
        state = PipelineJsonManager.analysis_state_from_dict(prepared)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        # Expected for a file of another shape; the message names the field.
        from src.utils.error_visibility import log_caught

        log_caught(logger, exc, "Ergebnisdatei in den Analysezustand übernehmen (Fremddaten)",
                   expected_level="info")
        raise ResultFileError(
            "Die Datei passt nicht zum Analysezustand dieser ALIMA-Version "
            f"({type(exc).__name__}: {str(exc)[:200]})."
        )
    return state, ignored


class _EventCollector:
    """Stand-in session for the render transport: keeps the events in a list,
    so a file that fails half-way leaves the real session untouched."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def append_render_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)


def render_loaded_state(state, filename: str) -> List[Dict[str, Any]]:
    """Render events of a loaded state: one log line naming the file, then the
    result cards of ``render_pipeline_result`` without its completion line."""
    from src.utils.error_visibility import log_caught
    from src.utils.pipeline_formatters import render_pipeline_result
    from src.webapp.render_bridge import _build_session_renderer

    collector = _EventCollector()
    renderer = _build_session_renderer(collector)
    renderer.render_system_message(f"📂 Ergebnis geladen aus „{filename}“ (kein Lauf in dieser Sitzung)")
    try:
        render_pipeline_result(renderer, state, announce_completion=False)
    except Exception as exc:  # noqa: BLE001 - foreign data may break any formatter
        log_caught(logger, exc, "Ergebnisdatei rendern (Fremddaten)")
        raise ResultFileError(
            "Die Datei enthält Werte, die sich nicht anzeigen lassen "
            f"({type(exc).__name__})."
        )
    if not any(ev.get("card") in _RESULT_CARDS for ev in collector.events):
        raise ResultFileError(
            "Die Datei enthält kein Erschließungsergebnis (keine Schlagwörter, Ketten oder Notationen)."
        )
    return collector.events


def load_result_file(raw: bytes, filename: Any) -> Dict[str, Any]:
    """Everything the endpoint needs, computed off the session - Claude Generated

    Returns ``{state, format, filename, ignored_fields, events, results,
    input_data}``; raises :class:`ResultFileError`. Pure apart from logging:
    the caller decides when to put it into the session.
    """
    from src.webapp.result_serialization import (
        extract_results_from_analysis_state,
        prepare_results_for_export,
    )

    name = display_filename(filename)
    data, fmt = parse_result_file(raw)
    original_input = data.get("input") if fmt == FORMAT_WEBAPP_EXPORT else None
    state, ignored = state_from_result_data(data, fmt)
    events = render_loaded_state(state, name)
    try:
        results = prepare_results_for_export(
            extract_results_from_analysis_state(state), validate_rvk=False
        )
    except Exception as exc:  # noqa: BLE001 - same foreign data, second consumer
        from src.utils.error_visibility import log_caught

        log_caught(logger, exc, "Ergebnisdatei serialisieren (Fremddaten)")
        raise ResultFileError(
            f"Die Datei enthält Werte, die sich nicht übernehmen lassen ({type(exc).__name__})."
        )

    # What the export of this session will say about its input: the type and
    # a 100-character preview, as a run records them, plus the file name.
    input_type = (original_input or {}).get("type") if isinstance(original_input, dict) else None
    preview = (original_input or {}).get("text_preview") if isinstance(original_input, dict) else None
    input_data = {
        "type": str(input_type or getattr(state, "input_type", None) or "text")[:40],
        "text_preview": str(preview or getattr(state, "original_abstract", "") or "")[:100],
        "loaded_from": name,
    }
    return {
        "state": state,
        "format": fmt,
        "filename": name,
        "ignored_fields": ignored,
        "events": events,
        "results": results,
        "input_data": input_data,
    }
