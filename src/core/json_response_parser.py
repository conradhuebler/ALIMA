"""JSON Response Parser - Zentrale JSON-Parsing-Logik für LLM-Antworten - Claude Generated

Extrahiert strukturierte Daten aus JSON-formatierten LLM-Antworten.
Handhabt diverse LLM-Eigenheiten: think-Blöcke, Code-Fences, umgebender Text.
"""
import json
import re
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> Optional[Dict]:
    """Extrahiert JSON aus LLM-Antwort.

    Handhabt:
    - <think>...</think> Blöcke vor dem JSON
    - <|begin_of_thought|>...<|end_of_thought|> Blöcke
    - ```json ... ``` Code-Fences
    - Rohes JSON ohne Wrapper
    - Text vor/nach dem JSON-Objekt

    Returns:
        Parsed dict oder None wenn kein valides JSON gefunden.
    """
    if not text or not text.strip():
        return None

    # 1. Remove <think>...</think> and similar reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # 2. Try ```json ... ``` code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            logger.debug("JSON in code fence is not valid JSON")

    # 3. Try to find raw JSON object { ... }
    # Find the first { and last } to extract the JSON object
    brace_start = cleaned.find('{')
    if brace_start != -1:
        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(cleaned)):
            ch = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    json_str = cleaned[brace_start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.debug(f"Extracted braces content is not valid JSON: {json_str[:100]}...")
                    break

    # 4. Last resort: try the entire cleaned text
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    logger.warning("Kein valides JSON in LLM-Antwort gefunden")
    return None


def extract_keywords_from_json(data: Dict) -> str:
    """Extrahiert Keywords als komma-getrennte Strings aus JSON.

    Unterstützt zwei Formate:
    - Einfache Liste: {"keywords": ["A", "B"]}
    - Objekt-Liste: {"keywords": [{"keyword": "A", "gnd_id": "123"}, ...]}

    Returns:
        Komma-getrennter String: 'Keyword1 (GND-ID: 123), Keyword2, ...'
    """
    keywords = data.get("keywords", [])
    if not keywords:
        return ""

    parts = []
    for kw in keywords:
        if isinstance(kw, str):
            parts.append(kw)
        elif isinstance(kw, dict):
            name = kw.get("keyword", "")
            gnd_id = kw.get("gnd_id", "")
            if name:
                if gnd_id:
                    parts.append(f"{name} ({gnd_id})")
                else:
                    parts.append(name)

    return ", ".join(parts)


def extract_title_from_json(data: Dict) -> Optional[str]:
    """Extrahiert Titel aus JSON data['title']."""
    title = data.get("title")
    if title and isinstance(title, str):
        return title.strip()
    return None


def extract_missing_concepts_from_json(data: Dict) -> List[str]:
    """Extrahiert fehlende Konzepte aus data['missing_concepts']."""
    concepts = data.get("missing_concepts", [])
    if isinstance(concepts, list):
        return [c.strip() for c in concepts if isinstance(c, str) and c.strip()]
    return []


def extract_gnd_classes_from_json(data: Dict) -> List[str]:
    """Extrahiert GND-Klassen aus data['gnd_classes']."""
    classes = data.get("gnd_classes", [])
    if isinstance(classes, list):
        return [c.strip() for c in classes if isinstance(c, str) and c.strip()]
    return []


def extract_analyse_from_json(data: Dict) -> Optional[str]:
    """Extrahiert den Analyse-/Diskussionstext aus JSON - Claude Generated"""
    analyse = data.get("analyse")
    if analyse and isinstance(analyse, str):
        return analyse.strip()
    return None


def extract_keyword_chains_from_json(data: Dict) -> List[Dict]:
    """Extrahiert Schlagwortketten aus JSON - Claude Generated

    Format: [{"chain": ["KW1", "KW2"], "reason": "Begründung"}, ...]
    """
    chains = data.get("keyword_chains", [])
    if not isinstance(chains, list):
        return []
    result = []
    for chain_obj in chains:
        if isinstance(chain_obj, dict):
            chain_list = chain_obj.get("chain", [])
            reason = chain_obj.get("reason", "")
            if chain_list:
                result.append({"chain": chain_list, "reason": reason})
    return result


def extract_missing_superordinate_from_json(data: Dict) -> List[str]:
    """Extrahiert fehlende Oberbegriffe aus JSON - Claude Generated"""
    terms = data.get("missing_superordinate_terms", [])
    if isinstance(terms, list):
        return [t.strip() for t in terms if isinstance(t, str) and t.strip()]
    return []


def extract_dk_from_json(data: Dict) -> List[str]:
    """Extrahiert DK/RVK-Codes aus data['classifications'].

    Format: [{"code": "DK 615.9", "type": "DK"}, ...]
    Returns: ["DK 615.9", "RVK QC 130", ...]
    """
    classifications = data.get("classifications", [])
    if not isinstance(classifications, list):
        return []

    codes = []
    for cls in classifications:
        if isinstance(cls, dict):
            code = cls.get("code", "").strip()
            cls_type = cls.get("type", "").strip().upper()
            if code:
                # If type is specified and code doesn't start with it, prepend
                if cls_type and not code.upper().startswith(cls_type):
                    codes.append(f"{cls_type} {code}")
                else:
                    codes.append(code)
        elif isinstance(cls, str):
            codes.append(cls.strip())

    return codes
