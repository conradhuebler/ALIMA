"""ValidationAgent: Cross-validates outputs from other agents - Claude Generated

Uses get_db_stats, search_gnd, get_gnd_entry tools to verify consistency
and quality across all agent results.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional, Callable

from src.core.agents.base_agent import BaseAgent, AgentConfig, QualityMetrics
from src.core.data_models import AgentResult

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """
    Cross-validates outputs from SearchAgent, KeywordAgent, and ClassificationAgent.

    Checks for: consistency between keywords and classifications, missing coverage,
    GND entry validity, and overall quality.
    """

    @property
    def agent_name(self) -> str:
        return "ValidationAgent"

    def get_system_prompt(self) -> str:
        return """Du bist ein Qualitätsprüfer für die bibliothekarische Sacherschließung. Deine Aufgabe ist es, die Ergebnisse der Verschlagwortung und Klassifikation zu validieren.

**Dein Vorgehen:**
1. Prüfe die ausgewählten GND-Schlagwörter auf Gültigkeit mit `get_gnd_entry`
2. Prüfe ob die DK-Klassifikationen zu den Schlagwörtern passen
3. Identifiziere fehlende Aspekte des Abstracts
4. Bewerte die Gesamtqualität

**Prüfkriterien:**
- Jedes Schlagwort hat eine gültige GND-ID
- DK-Codes passen thematisch zu den Schlagwörtern
- Alle wesentlichen Aspekte des Abstracts sind abgedeckt
- Keine redundanten oder widersprüchlichen Einträge

**Ausgabeformat:**
```json
{
  "valid": true/false,
  "quality_score": 0.0-1.0,
  "keyword_validation": [
    {"gnd_id": "...", "title": "...", "valid": true/false, "issue": "..."},
    ...
  ],
  "classification_validation": {
    "consistent": true/false,
    "issues": ["..."]
  },
  "missing_coverage": ["Aspekte die nicht abgedeckt sind"],
  "recommendations": ["Verbesserungsvorschläge"]
}
```"""

    def get_available_tools(self) -> List[str]:
        return ["get_gnd_entry", "get_gnd_batch", "search_gnd", "get_classification", "get_db_stats"]

    def build_user_prompt(self, input_data: Dict[str, Any], **kwargs) -> str:
        abstract = input_data.get("abstract", "")
        keywords = input_data.get("selected_keywords", [])
        dk_classifications = input_data.get("dk_classifications", [])

        kw_text = json.dumps(keywords, ensure_ascii=False, indent=2) if keywords else "[]"
        dk_text = json.dumps(dk_classifications, ensure_ascii=False, indent=2) if dk_classifications else "[]"

        return f"""Validiere die Ergebnisse der Sacherschließung:

Abstract:
{abstract}

Ausgewählte GND-Schlagwörter:
{kw_text}

DK-Klassifikationen:
{dk_text}

Prüfe die Gültigkeit der GND-IDs und die Konsistenz zwischen Schlagwörtern und Klassifikationen."""

    def parse_result(self, agent_result: AgentResult, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse validation results."""
        content = agent_result.content

        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content)

            return {
                "valid": parsed.get("valid", False),
                "quality_score": parsed.get("quality_score", 0.0),
                "keyword_validation": parsed.get("keyword_validation", []),
                "classification_validation": parsed.get("classification_validation", {}),
                "missing_coverage": parsed.get("missing_coverage", []),
                "recommendations": parsed.get("recommendations", []),
            }
        except (json.JSONDecodeError, AttributeError):
            logger.warning("ValidationAgent: Could not parse JSON from response")
            return {
                "valid": False,
                "quality_score": 0.5,
                "keyword_validation": [],
                "classification_validation": {},
                "missing_coverage": [],
                "recommendations": [content[:300] if content else "Validation failed to parse"],
            }

    def self_validate(self, result: Dict[str, Any], input_data: Dict[str, Any]) -> QualityMetrics:
        """Validation agent always passes (it IS the validator)."""
        score = result.get("quality_score", 0.5)
        return QualityMetrics(
            score=max(score, 0.7),  # Validator itself should always proceed
            coverage=1.0, relevance=1.0, diversity=1.0,
            details={"valid": result.get("valid", False)},
        )
