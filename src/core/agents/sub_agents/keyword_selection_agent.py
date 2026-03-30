"""Keyword Selection SubAgent - Claude Generated

Selects relevant GND entries from search results.
Uses cached GND data to avoid redundant lookups.
"""

import json
import logging
from typing import Dict, List, Any

from src.core.agents.sub_agents.base_sub_agent import BaseSubAgent, SubAgentResult
from src.core.agents.shared_context import SharedContext

logger = logging.getLogger(__name__)


class KeywordSelectionAgent(BaseSubAgent):
    """Select relevant GND keywords from pool.

    Analyzes the GND entries found by SearchAgent and selects
    the most relevant ones for the abstract.
    """

    @property
    def agent_name(self) -> str:
        return "Keyword Selection Agent"

    @property
    def agent_id(self) -> str:
        return "selection"

    def get_system_prompt(self) -> str:
        """System prompt for keyword selection."""
        return """Du bist **ALIMA Agent**, ein hochpräziser Wissensassistent mit Expertise in GND-Schlagwortung.

Deine Aufgaben:
1. **Themen identifizieren** und hierarchisch strukturieren
2. **Nur vorgegebene GND-Schlagworte verwenden** (keine Ergänzungen!)
3. **Schlagwortketten bilden** um Spezifität zu erhöhen
4. **Fehlende Konzepte dokumentieren**

**Kernregeln:**
- **Präzision vor Kreativität**: Nutze nur die vorgegebenen GND-Einträge
- **Hierarchie beachten**: Kombiniere Schlagworte zu Ketten
- **JSON-Ausgabe**: Immer als valides JSON-Objekt

**Ausgabeformat:**
```json
{
  "selected_keywords": [
    {"gnd_id": "...", "title": "...", "chain": "Oberbegriff | Begriff"},
    ...
  ],
  "missing_concepts": ["Konzept1", "Konzept2"],
  "reasoning": "Kurze Begründung der Auswahl"
}
```"""

    def get_available_tools(self) -> List[str]:
        """Tools for keyword selection."""
        return [
            "get_gnd_entry",
            "get_gnd_batch",
            "get_search_cache",
        ]

    def build_user_prompt(self) -> str:
        """Build prompt from abstract and GND entries."""
        prompt_parts = [
            "**Aufgabe**: Wähle die relevantesten GND-Schlagworte aus dem Pool.\n",
            f"**Abstract:**\n{self.context.abstract[:2000]}\n",
        ]

        # Include GND entries from search
        if self.context.gnd_entries:
            entries_json = json.dumps(self.context.gnd_entries[:50], ensure_ascii=False, indent=2)
            prompt_parts.append(f"\n**Verfügbare GND-Einträge ({len(self.context.gnd_entries)} total):**\n{entries_json}\n")

        # Include extracted keywords for context
        if self.context.extracted_keywords:
            prompt_parts.append(f"\n**Extrahierte Keywords:**\n{', '.join(self.context.extracted_keywords[:20])}\n")

        previous_context = self.get_previous_context()
        if previous_context:
            prompt_parts.append(f"\n**Kontext:**\n{previous_context}\n")

        prompt_parts.append("\nWähle die relevantesten Einträge aus. Nutze nur vorgegebene GND-IDs.")
        return "\n".join(prompt_parts)

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        """Parse selection results."""
        parsed = self._extract_json(llm_output)

        return {
            "selected_keywords": parsed.get("selected_keywords", []),
            "missing_concepts": parsed.get("missing_concepts", []),
            "reasoning": parsed.get("reasoning", ""),
            "raw_output": llm_output[:500],
        }

    def _update_shared_context(self, result_data: Dict[str, Any]) -> None:
        """Update context with selected keywords."""
        selected = result_data.get("selected_keywords", [])
        missing = result_data.get("missing_concepts", [])

        self.context.selected_keywords = selected

        self.logger.info(f"Selected {len(selected)} keywords")
        if missing:
            self.logger.info(f"Missing concepts: {missing}")