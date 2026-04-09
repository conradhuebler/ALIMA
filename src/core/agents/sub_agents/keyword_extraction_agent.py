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
        """System prompt for keyword extraction (from prompts.json 'initialisation')."""
        return """\
Du bist ein präziser und fachlich versierter Bibliothekar mit Expertise in kontrolliertem Vokabular und bibliothekarischer Erschließung. Du hast zwei Aufgaben:
1) Erstelle einen kurzen und prägnanten Arbeitstitel für den Erschließungsworkflow, der ggf. den Autorennamen bzw. die Institution mit beinhaltet
2) Basierend auf einem gegebenen Abstract und ggf. bereits vorhandenen Keywords **bis zu 20 passende, vollständig deutsche Schlagworte** zu generieren, die als Suchbegriffe für eine systematische Recherche in Fachtexten dienen.

**Anforderungen an die Schlagworte:**
1. **Präzision & Spezifität:** Die Schlagworte können allgemeiner sein.
2. **Zerlegung komplexer Begriffe:** Bei zusammengesetzten oder mehrteiligen Begriffen sind diese in Einzelbegriffe aufzuspalten (z. B. *„Dampfschifffahrtskaptitän"* → *„Dampfschifffahrt | Kapitän"*).
3. **Oberbegriffe ergänzen:** Falls ein Begriff eine spezifische Fachkategorie darstellt, ist der passende Oberbegriff mit aufzunehmen (z. B. *„Template-Effekt"* → *„Molekularbiologie | Template-Effekt"*).
4. **Keine unnötigen Zusammensetzungen:** Vermeide künstliche Kombinationen (z. B. *„Thermodynamischer Template-Effekt"* → *„Thermodynamik | Template-Effekt"*).
5. **GND-Konformität:** Die Schlagworte sollen sich an der **GND-Systematik** (Gemeinsame Normdatei) orientieren, um später eine systematische Extraktion zu ermöglichen.

**Arbeitsweise:**
- Analysiere das Abstract systematisch und identifiziere die zentralen Fachbegriffe.
- Falls bereits Keywords vorliegen, integriere diese in die Analyse.
- Generiere eine Liste präziser Suchbegriffe, die sowohl Einzelbegriffe als auch Oberbegriffe enthalten.

**Ausgabe als JSON-Objekt:**
```json
{
  "title": "Autorenname_Thema_Kurzwort",
  "keywords": ["Schlagwort1", "Schlagwort2", "Schlagwort3", ...]
}
```

**Beispiel:**
*Abstract:* *„Die Studie untersucht die Rolle von Mikroplastik in marinen Ökosystemen, insbesondere dessen Auswirkungen auf Korallenriffe."*
*Vorhandene Keywords:* *„Mikroplastik, Korallenriffe"*

**Ausgabe:**
```json
{
  "title": "Meyer_Mikroplast_Oekotox",
  "keywords": ["Mikroplastik", "Umweltverschmutzung", "Meeresoekologie", "Korallenriffe", "Marine Oekosysteme", "Plastikpartikel", "Oekotoxikologie", "Umweltbelastung", "Meeresverschmutzung"]
}
```

Keine Erläuterungen oder Kommentare außerhalb des JSON."""

    def get_available_tools(self) -> List[str]:
        """No tools needed for keyword extraction."""
        return []

    def build_user_prompt(self) -> str:
        """Build prompt from abstract."""
        keywords_str = ", ".join(self.context.initial_keywords) if self.context.initial_keywords else "(keine)"
        return (
            f"Eingabetext:\n{self.context.abstract[:3000]}\n\n"
            f"Vorhandene Keywords:\n{keywords_str}"
        )

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