"""KeywordAnalysisState ↔ SharedContext bridge (P-γ). Claude Generated.

Single-step warm-start needs to ingest both legacy ``KeywordAnalysisState``
JSON exports (``analysis_export_*.json``, ``Cadmium_Cogito.json``) and modern
``SharedContext`` JSON snapshots into the v4 ``WorkflowExecutor``.

This module provides:

* :func:`from_keyword_analysis_state` — build a SharedContext from a KAS dict.
* :func:`load_state_file` — auto-detect SharedContext vs KAS JSON on disk and
  return a populated :class:`SharedContext`.

Fields covered by the KAS→SharedContext bridge (MVP scope per WP5 §6):

==========================  ===========================================
KAS field                   SharedContext target
==========================  ===========================================
``original_abstract``       ``abstract``
``initial_keywords``        ``initial_keywords`` (GND tags stripped)
``working_title``           ``working_title``
``input_type``              ``input_type``
``source_value``            ``source_value``
``search_results``          ``gnd_entries`` (flattened) +
                            ``gnd_entries_per_keyword``
``dk_search_results``       ``dk_search_results``
``dk_classifications``      ``dk_classifications`` (str→dict promoted)
==========================  ===========================================

KAS-only fields not bridged (intentional; SharedContext does not store them):
``initial_llm_call_details``, ``final_llm_analysis``, ``dk_llm_analysis``,
``refinement_iterations``, ``dk_statistics``, ``dk_search_results_flattened``.
These live in ``KeywordAnalysisState`` only and are reconstituted on the way
back via :meth:`SharedContext.to_keyword_analysis_state`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.agents.shared_context import SharedContext


_GND_TAG_RE = re.compile(r"\s*\(GND-ID:\s*[^)]+\)\s*$")


def _strip_gnd_tag(keyword: str) -> str:
    """Strip a trailing ``(GND-ID: ...)`` suffix from a keyword string."""
    return _GND_TAG_RE.sub("", keyword).strip()


def _flatten_search_results(
    search_results: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Flatten KAS ``search_results`` into SharedContext shape.

    KAS shape (per :class:`SearchResult`):
        ``[{search_term: "...", results: {title: {gnd_ids: [...],
        classifications: {system: [codes]}, count: N, display_count?: M}}}]``

    SharedContext shape:
        - ``gnd_entries``: ``[{title, gnd_id, gnd_ids,
          classifications: {system: [codes]}, count, display_count?}]`` (deduped)
        - ``gnd_entries_per_keyword``: ``{term: [title, ...]}``
    """
    entries_by_title: Dict[str, Dict[str, Any]] = {}
    per_keyword: Dict[str, List[str]] = {}

    for sr in search_results or []:
        term = sr.get("search_term", "")
        results = sr.get("results", {}) or {}
        per_keyword.setdefault(term, [])
        for title, meta in results.items():
            per_keyword[term].append(title)
            if title in entries_by_title:
                continue
            gnd_ids = list(meta.get("gnd_ids", []) or [])
            # Carry the frequency back so a reloaded agentic state keeps the
            # real Häufigkeit (``display_count``); dropping it re-zeroed the
            # count on every JSON round-trip. - Claude Generated
            entry = {
                "title": title,
                "gnd_id": gnd_ids[0] if gnd_ids else "",
                "gnd_ids": gnd_ids,
                "classifications": {
                    system: list(codes)
                    for system, codes in (meta.get("classifications") or {}).items()
                    if codes
                },
                "count": meta.get("count", 1),
            }
            if meta.get("display_count") is not None:
                entry["display_count"] = meta["display_count"]
            entries_by_title[title] = entry

    return list(entries_by_title.values()), per_keyword


def _promote_dk_classifications(raw: List[Any]) -> List[Dict[str, Any]]:
    """Promote legacy string classifications to the dict shape SharedContext expects."""
    out: List[Dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"code": item, "type": ""})
    return out


def from_keyword_analysis_state(data: Dict[str, Any]) -> SharedContext:
    """Build a SharedContext from a KeywordAnalysisState dict.

    Accepts either a JSON-decoded dict or a dataclass converted via
    ``dataclasses.asdict``. Unknown fields are ignored. Missing fields fall
    back to their SharedContext defaults.

    Args:
        data: Mapping with KAS-shaped keys (top-level ``original_abstract``,
              ``initial_keywords``, ``search_results``, …).

    Returns:
        A new :class:`SharedContext` populated with the bridged fields.
    """
    if not isinstance(data, dict):
        raise TypeError(f"expected dict, got {type(data).__name__}")

    abstract = data.get("original_abstract") or ""

    raw_initial = data.get("initial_keywords") or []
    initial_keywords = [_strip_gnd_tag(k) for k in raw_initial if k]

    gnd_entries, per_keyword = _flatten_search_results(
        data.get("search_results") or []
    )

    ctx = SharedContext(
        abstract=abstract,
        initial_keywords=initial_keywords,
        input_type=data.get("input_type") or "text",
        source_value=data.get("source_value"),
        working_title=data.get("working_title") or "",
        gnd_entries=gnd_entries,
        gnd_entries_per_keyword=per_keyword,
        dk_search_results=list(data.get("dk_search_results") or []),
        dk_classifications=_promote_dk_classifications(
            data.get("dk_classifications") or []
        ),
    )
    return ctx


def _looks_like_shared_context(data: Dict[str, Any]) -> bool:
    """Heuristic: SharedContext exports always carry ``step_results``."""
    return isinstance(data.get("step_results"), dict)


def _looks_like_kas(data: Dict[str, Any]) -> bool:
    """Heuristic: KAS exports always carry ``original_abstract``."""
    return "original_abstract" in data


def load_state_file(path: Path | str) -> Tuple[SharedContext, str]:
    """Load a JSON state file, auto-detecting KAS vs SharedContext format.

    Args:
        path: Path to the JSON file.

    Returns:
        ``(context, kind)`` where ``kind`` is ``"shared_context"`` or
        ``"keyword_analysis_state"``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is neither a SharedContext nor a KAS export.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"state file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level JSON object")

    if _looks_like_shared_context(data):
        return SharedContext.from_dict(data), "shared_context"
    if _looks_like_kas(data):
        return from_keyword_analysis_state(data), "keyword_analysis_state"

    raise ValueError(
        f"{path}: unrecognised state format — expected SharedContext "
        f"('step_results' key) or KeywordAnalysisState ('original_abstract' key)"
    )
