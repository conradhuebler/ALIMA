"""Pure text / display / title helpers for the pipeline - Claude Generated.

Leaf module (no dependency on PipelineStepExecutor / formatters) so both the
executor and PipelineResultFormatter can share these. Split out of the former
``pipeline_utils`` god-module; still re-exported from ``pipeline_utils``.
"""

import html
import logging
import re
import string
from datetime import datetime
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _find_unbalanced_brace(template: str) -> Optional[int]:
    """Position of the first brace str.format would choke on, else None - Claude Generated"""
    i, n = 0, len(template)
    while i < n:
        c = template[i]
        if c == "{":
            if template.startswith("{{", i):
                i += 2
                continue
            closing = template.find("}", i + 1)
            next_open = template.find("{", i + 1)
            if closing == -1 or (next_open != -1 and next_open < closing):
                return i
            i = closing + 1
        elif c == "}":
            if template.startswith("}}", i):
                i += 2
                continue
            return i
        else:
            i += 1
    return None


def validate_prompt_placeholders(
    template: str, allowed: Tuple[str, ...] = ("abstract", "keywords")
) -> Optional[str]:
    """Check a user-edited prompt template against the runtime placeholders - Claude Generated

    Returns a German error message for unbalanced braces, positional fields
    (``{}``/``{0}``) or field names outside ``allowed``; ``None`` if the template
    is usable. Missing placeholders are fine: ``str.format(**vars)`` ignores
    unused variables.
    """
    pos = _find_unbalanced_brace(template)
    if pos is not None:
        line = template.count("\n", 0, pos) + 1
        snippet = " ".join(template[max(0, pos - 30) : pos + 30].split())
        return (
            f"Unausgeglichene geschweifte Klammer in Zeile {line}: »…{snippet}…«. "
            "Literale Klammern müssen als {{ bzw. }} geschrieben werden."
        )

    try:
        fields = [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name is not None
        ]
    except ValueError:
        return (
            "Die Prompt-Vorlage enthält unausgeglichene geschweifte Klammern. "
            "Literale Klammern müssen als {{ bzw. }} geschrieben werden."
        )

    invalid = []
    for field_name in fields:
        base_name = field_name.split(".")[0].split("[")[0]
        if base_name == "" or base_name.isdigit():
            invalid.append("{" + field_name + "}")
        elif base_name not in allowed:
            invalid.append("{" + base_name + "}")

    if invalid:
        allowed_list = ", ".join("{" + name + "}" for name in allowed)
        return (
            f"Unbekannte Platzhalter in der Prompt-Vorlage: {', '.join(sorted(set(invalid)))}. "
            f"Verfügbar sind: {allowed_list}. Literale Klammern als {{{{ bzw. }}}} schreiben."
        )
    return None


def repair_display_text(text: Any) -> str:
    """Normalize display text and repair common UTF-8/Latin-1 mojibake."""
    if text is None:
        return ""

    cleaned = html.unescape(str(text))
    if any(marker in cleaned for marker in ("Ã", "Â", "â€", "â€“", "â€”", "â€¦", "â", "Ê")):
        try:
            repaired = cleaned.encode("latin-1").decode("utf-8")
            if repaired and repaired.count("�") <= cleaned.count("�"):
                cleaned = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"\s+", " ", cleaned).strip()


# Title Building Utilities - Claude Generated


def sanitize_for_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text for use in filenames - Claude Generated

    Args:
        text: Text to sanitize
        max_length: Maximum length (default 50)

    Returns:
        Sanitized filename-safe string
    """
    if not text:
        return "untitled"

    # Replace problematic characters with underscores
    # Windows forbidden: < > : " / \ | ? *
    # Also replace spaces, commas, periods
    sanitized = re.sub(r'[<>:"/\\|?*\s,.]', '_', text)

    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Trim underscores from start/end
    sanitized = sanitized.strip('_')

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    return sanitized or "untitled"


def build_working_title(
    llm_title: Optional[str],
    source_identifier: str,
    timestamp: Optional[str] = None,
    fallback_prefix: str = "analysis"
) -> str:
    """
    Build complete working title: {llm_title}_{source}_{timestamp} - Claude Generated

    Args:
        llm_title: Title extracted from LLM (can be None)
        source_identifier: DOI/filename/URL identifier
        timestamp: ISO timestamp (generated if None)
        fallback_prefix: Prefix when llm_title missing

    Returns:
        Complete working title string
    """
    # Generate timestamp if not provided
    if not timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        # Convert ISO to compact format if needed
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp = dt.strftime('%Y%m%d_%H%M%S')
        except (ValueError, TypeError):
            # If parsing fails, use as-is (might already be compact)
            pass

    # Build title components
    components = []

    # Component 1: LLM title or fallback
    if llm_title:
        components.append(sanitize_for_filename(llm_title, max_length=30))
    else:
        components.append(fallback_prefix)

    # Component 2: Source identifier
    source_clean = sanitize_for_filename(source_identifier, max_length=40)
    if source_clean and source_clean != "untitled":
        components.append(source_clean)

    # Component 3: Timestamp
    components.append(timestamp)

    # Combine with underscores
    return '_'.join(components)


def extract_source_identifier(
    input_type: str,
    input_value: str
) -> str:
    """
    Extract clean source identifier from input - Claude Generated

    Args:
        input_type: 'text', 'doi', 'pdf', 'img', 'url'
        input_value: The actual input value/path

    Returns:
        Clean source identifier string
    """
    if input_type == 'doi':
        # DOI: Use the DOI itself (already sanitized by build_working_title)
        return input_value

    elif input_type in ('pdf', 'img'):
        # File: Use basename without extension
        from pathlib import Path
        return Path(input_value).stem

    elif input_type == 'url':
        # URL: Extract domain
        parsed = urlparse(input_value)
        return parsed.netloc or 'url'

    else:  # 'text'
        # Text: Don't include text preview in identifier - Claude Generated
        return 'text'


def flatten_keyword_centric_results(
    keyword_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten keyword-centric format to DK-centric with deduplication - Claude Generated

    The BiblioClient.extract_dk_classifications_for_keywords() returns keyword-centric format:
    [{"keyword": "Cadmium", "source": "cache", "classifications": [{"dk": "681.3", ...}]},
     {"keyword": "Halbleiter", "source": "...", "classifications": [{"dk": "681.3", ...}]}]

    This function DEDUPLICATES and merges classifications across keywords:
    - Merges identical DK codes from multiple keywords
    - Deduplicates titles across keywords
    - Sums frequency counts
    - Tracks which keywords led to each classification

    Args:
        keyword_results: List of keyword-centric results from BiblioClient

    Returns:
        Flattened and deduplicated list of DK-centric classification results
    """
    # Group classifications by "{type}:{code}" to detect and merge duplicates
    grouped = {}  # Key: "DK:681.3", Value: merged classification data

    for kw_result in keyword_results:
        keyword = kw_result.get("keyword", "unknown")
        classifications = kw_result.get("classifications", [])

        for cls in classifications:
            cls_type = cls.get("type") or cls.get("classification_type", "DK")
            cls_code = cls.get("dk", "")
            key = f"{cls_type}:{cls_code}"

            # Initialize group if first time seeing this classification
            if key not in grouped:
                grouped[key] = {
                    "dk": cls_code,
                    "type": cls_type,
                    "classification_type": cls.get("classification_type", cls_type),
                    "titles": [],
                    "count": 0,
                    "matched_keywords": [],
                    "keyword_counts": {},
                    "source": cls.get("source"),
                    "label": cls.get("label"),
                    "ancestor_path": cls.get("ancestor_path"),
                    "register": list(cls.get("register", [])) if cls.get("register") else [],
                    "score": cls.get("score", 0),
                    "branch_family": cls.get("branch_family"),
                    "rvk_validation_status": cls.get("rvk_validation_status"),
                    "validation_message": cls.get("validation_message"),
                }

            # Merge titles (deduplicate using set) - Claude Generated
            # Filter out placeholder titles from cache that should not be displayed
            title_set = set(grouped[key]["titles"])
            for title in cls.get("titles", []):
                # Skip placeholder titles from classification cache - Claude Generated
                if title.startswith("Cached Catalog Entry for RSN"):
                    continue
                if title == "Cached Author":
                    continue
                if title not in title_set:
                    grouped[key]["titles"].append(title)
                    title_set.add(title)

            # Sum counts from this keyword
            grouped[key]["count"] += cls.get("count", 0)
            grouped[key]["score"] = max(grouped[key].get("score", 0), cls.get("score", 0))

            # Track which keywords contributed to this classification
            if keyword not in grouped[key]["matched_keywords"]:
                grouped[key]["matched_keywords"].append(keyword)
            grouped[key]["keyword_counts"][keyword] = cls.get("count", 0)
            for register_entry in cls.get("register", []) or []:
                if register_entry not in grouped[key]["register"]:
                    grouped[key]["register"].append(register_entry)
            if cls.get("rvk_validation_status") and not grouped[key].get("rvk_validation_status"):
                grouped[key]["rvk_validation_status"] = cls.get("rvk_validation_status")
            if cls.get("validation_message") and not grouped[key].get("validation_message"):
                grouped[key]["validation_message"] = cls.get("validation_message")

    # Convert to list and sort by count (most frequent first)
    flattened = sorted(grouped.values(), key=lambda x: x["count"], reverse=True)

    # Log deduplication metrics
    original_count = sum(len(kr.get("classifications", [])) for kr in keyword_results)
    deduplicated_count = len(flattened)
    if original_count > deduplicated_count:
        logger.info(f"🔧 DK Deduplication: {original_count} → {deduplicated_count} (-{original_count - deduplicated_count} Duplikate entfernt)")

    return flattened
