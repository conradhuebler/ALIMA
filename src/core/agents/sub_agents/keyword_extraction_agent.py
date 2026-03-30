"""Keyword Extraction SubAgent - Claude Generated

First agent in the pipeline: extracts initial keywords from the abstract.
Pure LLM task with no tool calls.
"""

import json
import logging
from typing import Dict, List, Any

from src.core.agents.sub_agents.base_sub_agent import BaseSubAgent, SubAgentResult
from src.core.agents.shared_context import SharedContext

logger = logging.getLogger(__name__)


class KeywordExtractionAgent(BaseSubAgent):
    """Extract initial keywords from abstract.

    This is the first agent in the pipeline. It analyzes the abstract
    and generates relevant German keywords for library cataloging.

    No tool calls - pure LLM reasoning task.
    """

    @property
    def agent_name(self) -> str:
        return "Keyword Extraction Agent"

    @property
    def agent_id(self) -> str:
        return "extraction"

    def get_system_prompt(self) -> str:
        """System prompt for keyword extraction."""
        return """Du bist ein präziser und fachlich versierter Bibliothekar mit Expertise in kontrolliertem Vokabular und bibliothekarischer Erschließung.

Deine Aufgabe:
1) Erstelle einen kurzen und prägnanten Arbeitstitel für den Erschließungsworkflow
2) Generiere **bis zu 20 passende, vollständig deutsche Schlagworte** für eine systematische Recherche

Anforderungen an die Schlagworte:
- **Präzision & Spezifität**: Die Schlagworte können allgemeiner sein
- **Zerlegung komplexer Begriffe**: Bei zusammengesetzten Begriffen in Einzelbegriffe aufspalten
- **Oberbegriffe ergänzen**: Wenn ein Begriff eine Fachkategorie darstellt, Oberbegriff aufnehmen
- **GND-Konformität**: Orientiere dich an der GND-Systematik (Gemeinsame Normdatei)

WICHTIG: Gib das Ergebnis als valides JSON-Objekt aus:
```json
{
  "title": "Autorenname_Thema_Kurzwort",
  "keywords": ["Schlagwort1", "Schlagwort2", "Schlagwort3", ...]
}
```

Keine Erläuterungen oder Kommentare außerhalb des JSON."""

    def get_available_tools(self) -> List[str]:
        """No tools needed for keyword extraction."""
        return []

    def build_user_prompt(self) -> str:
        """Build prompt from abstract."""
        prompt_parts = [
            "Analysiere den folgenden Abstract und generiere deutsche Schlagworte für die bibliothekarische Erschließung:\n",
            f"**Abstract:**\n{self.context.abstract[:3000]}\n",
        ]

        if self.context.initial_keywords:
            prompt_parts.append(f"\n**Vorhandene Keywords:**\n{', '.join(self.context.initial_keywords[:20])}\n")

        prompt_parts.append("\nGib das Ergebnis als JSON zurück.")
        return "\n".join(prompt_parts)

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        """Parse LLM output for keywords and title."""
        parsed = self._extract_json(llm_output)

        return {
            "title": parsed.get("title", ""),
            "keywords": parsed.get("keywords", []),
            "raw_output": llm_output[:500],
        }

    def _update_shared_context(self, result_data: Dict[str, Any]) -> None:
        """Update context with extracted keywords."""
        if result_data.get("title"):
            self.context.working_title = result_data["title"]

        if result_data.get("keywords"):
            self.context.extracted_keywords = result_data["keywords"]

        self.logger.info(f"Extracted {len(result_data.get('keywords', []))} keywords")
        self.logger.debug(f"Keywords: {result_data.get('keywords', [])}")