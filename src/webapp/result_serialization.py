"""Helpers for serializing webapp pipeline results without the full app stack."""

from __future__ import annotations

import html
import logging
import re
from functools import lru_cache
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests


logger = logging.getLogger(__name__)


def ensure_list(value):
    """Normalize legacy comma-separated strings to arrays for the web API."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def ensure_json_serializable(value):
    """Recursively convert non-JSON-serializable types."""
    if isinstance(value, set):
        return list(value)
    if isinstance(value, dict):
        return {k: ensure_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_json_serializable(v) for v in value]
    return value


def serialize_llm_details(llm_call: Any) -> Optional[Dict[str, Any]]:
    """Serialize the subset of LLM call details the web UI can preview safely."""
    if not llm_call:
        return None

    return {
        "response_full_text": getattr(llm_call, "response_full_text", ""),
        "analyse_text": getattr(llm_call, "analyse_text", "") or "",
        "provider": getattr(llm_call, "provider_used", ""),
        "model": getattr(llm_call, "model_used", ""),
        "extracted_keywords": getattr(llm_call, "extracted_keywords", []),
        "extracted_gnd_keywords": getattr(llm_call, "extracted_gnd_keywords", []),
        "extracted_gnd_classes": getattr(llm_call, "extracted_gnd_classes", []),
        "missing_concepts": ensure_list(getattr(llm_call, "missing_concepts", [])),
        "token_count": getattr(llm_call, "token_count", 0),
    }


def parse_classification_entry(item: Any) -> Dict[str, Any]:
    """Normalize a legacy string or structured classification object."""
    extras: Dict[str, Any] = {}

    if isinstance(item, dict):
        extras = {k: v for k, v in item.items() if k not in {"system", "code", "display"}}
        display = str(item.get("display") or "").strip()
        system = str(item.get("system") or "").strip().upper()
        code = str(item.get("code") or "").strip()
    else:
        display = str(item or "").strip()
        system = ""
        code = ""

    if not display and system and code:
        display = f"{system} {code}".strip()

    if display and not system:
        upper_display = display.upper()
        if upper_display.startswith("DK "):
            system = "DK"
            code = code or display[3:].strip()
        elif upper_display.startswith("RVK "):
            system = "RVK"
            code = code or display[4:].strip()

    if not code:
        code = display

    if not display:
        display = f"{system} {code}".strip() if system else str(code).strip()

    return {
        "system": system or "UNKNOWN",
        "code": code,
        "display": display,
        **extras,
    }


@lru_cache(maxsize=256)
def validate_rvk_notation(code: str) -> Dict[str, Any]:
    """Validate an RVK notation against the official RVK API."""
    normalized_code = re.sub(r"\s+", " ", str(code or "").strip()).upper()
    if not normalized_code:
        return {
            "status": "non_standard",
            "is_standard": False,
            "canonical_code": "",
            "label": None,
            "message": "Empty RVK notation",
        }

    url = f"https://rvk.uni-regensburg.de/api/json/node/{quote(normalized_code, safe='')}"

    try:
        response = requests.get(url, timeout=6)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(f"RVK API validation failed for '{normalized_code}': {exc}")
        return {
            "status": "validation_error",
            "is_standard": None,
            "canonical_code": normalized_code,
            "label": None,
            "message": str(exc),
        }

    node = payload.get("node") if isinstance(payload, dict) else None
    if isinstance(node, dict) and node.get("notation"):
        return {
            "status": "standard",
            "is_standard": True,
            "canonical_code": str(node.get("notation", "")).strip(),
            "label": html.unescape(str(node.get("benennung", "")).strip()) or None,
            "message": None,
        }

    return {
        "status": "non_standard",
        "is_standard": False,
        "canonical_code": normalized_code,
        "label": None,
        "message": payload.get("error-message", "Notation Not Found") if isinstance(payload, dict) else "Notation Not Found",
    }


def build_structured_classifications(raw_classifications, validate_rvk: bool = False):
    """Normalize classifications into structured entries."""
    structured = []

    for item in ensure_list(raw_classifications):
        if not item:
            continue

        entry = parse_classification_entry(item)

        if entry["system"] == "RVK":
            if validate_rvk:
                validation = validate_rvk_notation(entry["code"])
                entry.update({
                    "validation_status": validation["status"],
                    "is_standard": validation["is_standard"],
                    "canonical_code": validation["canonical_code"],
                    "label": validation["label"],
                    "validation_message": validation["message"],
                    "validation_source": "rvk_api",
                })
            else:
                entry.setdefault("validation_status", "not_checked")
                entry.setdefault("is_standard", None)
                entry.setdefault("canonical_code", entry["code"])
                entry.setdefault("label", None)
                entry.setdefault("validation_message", None)
                entry.setdefault("validation_source", None)

        structured.append(entry)

    return structured


def prepare_results_for_export(
    results: Optional[Dict[str, Any]],
    validate_rvk: bool = False,
) -> Dict[str, Any]:
    """Ensure export payload contains the structured classification schema."""
    prepared = dict(results or {})

    legacy_classifications = ensure_list(prepared.get("dk_classifications", []))
    raw_structured = prepared.get("classifications") or legacy_classifications
    structured_classifications = build_structured_classifications(
        raw_structured,
        validate_rvk=validate_rvk,
    )

    if not legacy_classifications:
        legacy_classifications = [entry["display"] for entry in structured_classifications]

    rvk_entries = [entry for entry in structured_classifications if entry.get("system") == "RVK"]
    prepared["classifications"] = ensure_json_serializable(structured_classifications)
    prepared["classifications_deprecated_alias"] = "dk_classifications"
    prepared["dk_classifications"] = ensure_json_serializable(legacy_classifications)
    prepared["classification_validation"] = {
        "rvk_checked_via": "https://rvk.uni-regensburg.de/regensburger-verbundklassifikation-online/rvk-api",
        "rvk_total": len(rvk_entries),
        "rvk_standard": sum(1 for entry in rvk_entries if entry.get("validation_status") == "standard"),
        "rvk_non_standard": sum(1 for entry in rvk_entries if entry.get("validation_status") == "non_standard"),
        "rvk_validation_errors": sum(1 for entry in rvk_entries if entry.get("validation_status") == "validation_error"),
    }

    rvk_provenance = prepared.get("rvk_provenance") or {}
    prepared["rvk_provenance"] = {
        "catalog_standard": int(rvk_provenance.get("catalog_standard", 0) or 0),
        "catalog_nonstandard": int(rvk_provenance.get("catalog_nonstandard", 0) or 0),
        "rvk_gnd_index": int(rvk_provenance.get("rvk_gnd_index", 0) or 0),
        "rvk_api": int(rvk_provenance.get("rvk_api", 0) or 0),
    }

    return ensure_json_serializable(prepared)


def extract_results_from_analysis_state(analysis_state) -> dict:
    """Extract a JSON-serializable results dict from KeywordAnalysisState."""
    final_keywords = []
    if hasattr(analysis_state, "final_llm_analysis") and analysis_state.final_llm_analysis:
        final_keywords = ensure_list(
            getattr(analysis_state.final_llm_analysis, "extracted_gnd_keywords", [])
        )

    dk_classifications = ensure_list(getattr(analysis_state, "dk_classifications", []))
    rvk_provenance = getattr(analysis_state, "rvk_provenance", None)
    initial_keywords = ensure_list(getattr(analysis_state, "initial_keywords", []))
    original_abstract = getattr(analysis_state, "original_abstract", "")
    working_title = getattr(analysis_state, "working_title", "")
    search_results = getattr(analysis_state, "search_results", [])
    dk_search_results = getattr(analysis_state, "dk_search_results", [])

    serialized_search_results = []
    if search_results:
        for result in search_results:
            try:
                search_term = getattr(result, "search_term", "") if hasattr(result, "search_term") else ""
                result_data = getattr(result, "results", {}) if hasattr(result, "results") else {}
                serialized_search_results.append({
                    "search_term": search_term,
                    "results": result_data if isinstance(result_data, dict) else {},
                })
            except Exception as exc:
                logger.warning(f"Error serializing search result: {exc}")
                continue

    initial_llm_details = None
    try:
        if hasattr(analysis_state, "initial_llm_call_details") and analysis_state.initial_llm_call_details:
            initial_llm_details = serialize_llm_details(analysis_state.initial_llm_call_details)
    except Exception as exc:
        logger.warning(f"Error serializing initial_llm_call_details: {exc}")

    final_llm_details = None
    try:
        if hasattr(analysis_state, "final_llm_analysis") and analysis_state.final_llm_analysis:
            final_llm_details = serialize_llm_details(analysis_state.final_llm_analysis)
    except Exception as exc:
        logger.warning(f"Error serializing final_llm_analysis: {exc}")

    dk_llm_details = None
    try:
        if hasattr(analysis_state, "dk_llm_analysis") and analysis_state.dk_llm_analysis:
            dk_llm_details = serialize_llm_details(analysis_state.dk_llm_analysis)
    except Exception as exc:
        logger.warning(f"Error serializing dk_llm_analysis: {exc}")

    structured_classifications = build_structured_classifications(
        dk_classifications,
        validate_rvk=False,
    )

    return {
        "original_abstract": original_abstract,
        "working_title": working_title,
        "initial_keywords": ensure_json_serializable(initial_keywords),
        "final_keywords": ensure_json_serializable(final_keywords),
        "search_results": ensure_json_serializable(serialized_search_results),
        "classifications": ensure_json_serializable(structured_classifications),
        "classifications_deprecated_alias": "dk_classifications",
        "dk_classifications": ensure_json_serializable(dk_classifications),
        "rvk_provenance": ensure_json_serializable(rvk_provenance),
        "dk_search_results": ensure_json_serializable(dk_search_results),
        "dk_search_results_flattened": ensure_json_serializable(
            getattr(analysis_state, "dk_search_results_flattened", [])
        ),
        "dk_statistics": ensure_json_serializable(getattr(analysis_state, "dk_statistics", None)),
        "initial_llm_call_details": ensure_json_serializable(initial_llm_details),
        "final_llm_call_details": ensure_json_serializable(final_llm_details),
        "dk_llm_analysis_details": ensure_json_serializable(dk_llm_details),
        "verification": ensure_json_serializable(
            getattr(analysis_state.final_llm_analysis, "verification", None)
            if hasattr(analysis_state, "final_llm_analysis") and analysis_state.final_llm_analysis
            else None
        ),
        "pipeline_metadata": {
            "search_suggesters_used": ensure_json_serializable(
                getattr(analysis_state, "search_suggesters_used", [])
            ),
            "initial_gnd_classes": ensure_json_serializable(
                getattr(analysis_state, "initial_gnd_classes", [])
            ),
            "has_final_llm_analysis": bool(final_llm_details),
            "has_initial_llm_analysis": bool(initial_llm_details),
            "has_dk_llm_analysis": bool(dk_llm_details),
        },
    }
