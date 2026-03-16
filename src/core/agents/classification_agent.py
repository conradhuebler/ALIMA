"""ClassificationAgent: DK/RVK classification using MCP tools - Claude Generated

Uses search_catalog, get_dk_cache, get_classification tools to determine
appropriate DK and RVK classifications for an abstract.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional, Callable

from src.core.agents.base_agent import BaseAgent, AgentConfig, QualityMetrics
from src.core.data_models import AgentResult
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ClassificationAgent(BaseAgent):
    """
    Determines DK and RVK classifications for an abstract.

    Strategy: LLM uses catalog search tools to find DK classifications
    associated with similar works, then selects the most appropriate ones.
    """

    @property
    def agent_name(self) -> str:
        return "ClassificationAgent"

    def get_system_prompt(self) -> str:
        return """Du bist ein Experte für die Klassifikation von wissenschaftlichen Texten nach DK (Dezimalklassifikation) und RVK (Regensburger Verbundklassifikation).

**Dein Vorgehen:**
1. Analysiere den Abstract und die bereits zugewiesenen GND-Schlagwörter
2. Nutze die Such-Tools um DK-Klassifikationen zu finden:
   - `get_dk_cache`: Prüfe gecachte DK-Zuordnungen
   - `search_catalog`: Suche im Katalog nach Titeln mit ähnlichen Schlagwörtern
   - `get_classification`: Details zu DK/RVK-Codes abrufen
   - `search_gnd`: Suche nach DDC-Codes in GND-Einträgen
3. Analysiere die gefundenen Klassifikationen und wähle die passendsten

**Auswahlkriterien:**
- Häufigkeit: Bevorzuge DK-Codes die bei vielen ähnlichen Titeln vorkommen
- Spezifität: Wähle den spezifischsten passenden Code
- Konsistenz: Die gewählten DK-Codes sollten zueinander passen
- Typisch 1-3 DK-Hauptklassen und optional RVK-Notationen

**Ausgabeformat:**
```json
{
  "dk_classifications": [
    {"code": "004", "title": "Informatik", "confidence": 0.9, "source": "catalog_frequency"},
    ...
  ],
  "rvk_classifications": [
    {"code": "ST 250", "title": "...", "confidence": 0.7},
    ...
  ],
  "reasoning": "Begründung der Klassifikationsauswahl"
}
```"""

    def get_available_tools(self) -> List[str]:
        return ["search_catalog", "get_dk_cache", "get_classification", "search_gnd", "get_gnd_entry"]

    def build_user_prompt(self, input_data: Dict[str, Any], **kwargs) -> str:
        abstract = input_data.get("abstract", "")
        keywords = input_data.get("selected_keywords", [])
        iteration = kwargs.get("iteration", 1)

        kw_text = ""
        if keywords:
            kw_items = []
            for kw in keywords:
                title = kw.get("title", "")
                gnd_id = kw.get("gnd_id", "")
                kw_items.append(f"  - {title} (GND: {gnd_id})")
            kw_text = "\n".join(kw_items)

        prompt = f"""Bestimme die DK- und RVK-Klassifikationen für folgenden Text:

Abstract:
{abstract}

Zugewiesene GND-Schlagwörter:
{kw_text}

Suche im Katalog nach Titeln mit ähnlichen Schlagwörtern und analysiere die Häufigkeit der DK-Codes."""

        if iteration > 1:
            prompt += "\n\nErweitere die Suche mit alternativen Begriffen und Synonymen."

        return prompt

    def parse_result(self, agent_result: AgentResult, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse classification results."""
        content = agent_result.content

        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content)

            return {
                "dk_classifications": parsed.get("dk_classifications", []),
                "rvk_classifications": parsed.get("rvk_classifications", []),
                "reasoning": parsed.get("reasoning", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            logger.warning("ClassificationAgent: Could not parse JSON from response")

        # Fallback: extract DK codes from text (pattern: 3-digit numbers)
        dk_codes = []
        dk_pattern = re.compile(r'\b(\d{3}(?:\.\d+)?)\b')
        for match in dk_pattern.finditer(content):
            code = match.group(1)
            if 0 < int(code.split('.')[0]) < 1000:
                dk_codes.append({"code": code, "title": "", "confidence": 0.5})

        return {
            "dk_classifications": dk_codes[:5],
            "rvk_classifications": [],
            "reasoning": content[:300] if content else "",
        }

    def self_validate(self, result: Dict[str, Any], input_data: Dict[str, Any]) -> QualityMetrics:
        """Validate classification quality."""
        dk = result.get("dk_classifications", [])
        rvk = result.get("rvk_classifications", [])

        # Expect at least 1 DK classification
        has_dk = len(dk) > 0
        dk_with_confidence = [c for c in dk if c.get("confidence", 0) >= 0.5]

        coverage = 1.0 if has_dk else 0.0
        relevance = len(dk_with_confidence) / max(len(dk), 1) if dk else 0
        diversity = min(len(dk) / 3, 1.0)  # 3 DK codes is ideal

        score = coverage * 0.4 + relevance * 0.3 + diversity * 0.3

        return QualityMetrics(
            score=score, coverage=coverage, relevance=relevance, diversity=diversity,
            details={"dk_count": len(dk), "rvk_count": len(rvk), "high_confidence": len(dk_with_confidence)},
        )
