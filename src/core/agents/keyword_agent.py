"""KeywordAgent: GND keyword selection and verification using MCP tools - Claude Generated

Uses get_gnd_batch, search_gnd tools to verify and select the best GND keywords
from the pool built by SearchAgent.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional, Callable

from src.core.agents.base_agent import BaseAgent, AgentConfig, QualityMetrics
from src.core.data_models import AgentResult
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class KeywordAgent(BaseAgent):
    """
    Selects and verifies GND keywords from a candidate pool.

    Uses the GND entry details (titles, descriptions, DDCs) to assess relevance
    and select the most appropriate subject headings for the abstract.
    """

    @property
    def agent_name(self) -> str:
        return "KeywordAgent"

    def get_system_prompt(self) -> str:
        return """Du bist ein Experte für die verbale Sacherschließung nach GND-Regeln. Deine Aufgabe ist es, aus einem Pool von GND-Kandidaten die besten Schlagwörter für einen Abstract auszuwählen.

**Dein Vorgehen:**
1. Analysiere den Abstract und verstehe das Thema
2. Prüfe die GND-Kandidaten mit `get_gnd_batch` und `get_gnd_entry` auf Relevanz
3. Wähle 5-15 passende GND-Schlagwörter aus:
   - Sachschlagwörter (Thema des Textes)
   - Geographische Schlagwörter (falls relevant)
   - Zeitschlagwörter (falls relevant)
   - Formschlagwörter (Dokumenttyp)
4. Prüfe ob wichtige Aspekte des Abstracts nicht abgedeckt sind
5. Suche ggf. nach zusätzlichen GND-Einträgen mit `search_gnd`

**Auswahlkriterien:**
- Spezifität: Bevorzuge spezifische vor allgemeinen Begriffen
- Relevanz: Jedes Schlagwort muss direkt zum Abstract-Thema passen
- Vollständigkeit: Alle wesentlichen Aspekte sollten abgedeckt sein
- GND-Konformität: Nur normierte GND-Einträge verwenden

**Ausgabeformat:**
```json
{
  "selected_keywords": [
    {"gnd_id": "...", "title": "...", "type": "Sachschlagwort|Geographikum|Zeitschlagwort|Formschlagwort"},
    ...
  ],
  "reasoning": "Kurze Begründung der Auswahl",
  "missing_concepts": ["Konzepte aus dem Abstract, die nicht durch GND abgedeckt werden"]
}
```"""

    def get_available_tools(self) -> List[str]:
        return ["get_gnd_entry", "get_gnd_batch", "search_gnd", "get_search_cache"]

    def build_user_prompt(self, input_data: Dict[str, Any], **kwargs) -> str:
        abstract = input_data.get("abstract", "")
        gnd_pool = input_data.get("gnd_entries", [])
        iteration = kwargs.get("iteration", 1)

        # Format GND pool for prompt
        pool_text = ""
        if gnd_pool:
            pool_entries = []
            for entry in gnd_pool[:100]:  # Limit to avoid context overflow
                gnd_id = entry.get("gnd_id", "?")
                title = entry.get("title", "?")
                pool_entries.append(f"  - {title} (GND: {gnd_id})")
            pool_text = "\n".join(pool_entries)

        prompt = f"""Wähle die besten GND-Schlagwörter für folgenden Abstract:

{abstract}

GND-Kandidatenpool ({len(gnd_pool)} Einträge):
{pool_text}

Prüfe die relevantesten Kandidaten mit get_gnd_batch und wähle 5-15 passende Schlagwörter aus."""

        if iteration > 1:
            prev_missing = input_data.get("_previous_missing_concepts", [])
            if prev_missing:
                prompt += (
                    f"\n\nIn der vorherigen Iteration wurden folgende fehlende Konzepte identifiziert: "
                    f"{', '.join(prev_missing)}\n"
                    "Versuche diese durch zusätzliche GND-Suche abzudecken."
                )

        return prompt

    def parse_result(self, agent_result: AgentResult, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse keyword selection from agent output."""
        content = agent_result.content

        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content)

            return {
                "selected_keywords": parsed.get("selected_keywords", []),
                "reasoning": parsed.get("reasoning", ""),
                "missing_concepts": parsed.get("missing_concepts", []),
            }
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"KeywordAgent: Could not parse JSON, extracting keywords from text")

        # Fallback: extract GND IDs from text
        keywords = []
        gnd_pattern = re.compile(r'(\d{8,})')
        for match in gnd_pattern.finditer(content):
            keywords.append({"gnd_id": match.group(1), "title": "", "type": "Sachschlagwort"})

        return {
            "selected_keywords": keywords,
            "reasoning": content[:300] if content else "",
            "missing_concepts": [],
        }

    def self_validate(self, result: Dict[str, Any], input_data: Dict[str, Any]) -> QualityMetrics:
        """Validate keyword selection quality."""
        keywords = result.get("selected_keywords", [])
        missing = result.get("missing_concepts", [])

        # Count: expect 5-15 keywords
        count = len(keywords)
        if count < 5:
            count_score = count / 5
        elif count <= 15:
            count_score = 1.0
        else:
            count_score = max(0.5, 1.0 - (count - 15) / 20)

        # All should have GND IDs
        with_ids = sum(1 for kw in keywords if kw.get("gnd_id"))
        id_ratio = with_ids / max(count, 1)

        # Fewer missing concepts = better
        missing_penalty = min(len(missing) * 0.1, 0.3)

        coverage = max(0, 1.0 - missing_penalty)
        relevance = id_ratio
        score = count_score * 0.3 + relevance * 0.4 + coverage * 0.3

        return QualityMetrics(
            score=score, coverage=coverage, relevance=relevance, diversity=count_score,
            details={"keyword_count": count, "with_gnd_ids": with_ids, "missing_concepts": len(missing)},
        )
