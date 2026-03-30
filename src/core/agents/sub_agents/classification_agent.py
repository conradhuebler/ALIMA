"""Classification SubAgent - Claude Generated

Assigns DK/RVK classifications to selected keywords.
Uses cached catalog data for efficiency.
"""

import json
import logging
from typing import Dict, List, Any

from src.core.agents.sub_agents.base_sub_agent import BaseSubAgent, SubAgentResult
from src.core.agents.shared_context import SharedContext

logger = logging.getLogger(__name__)


class ClassificationAgent(BaseSubAgent):
    """Assign DK/RVK classifications.

    Uses selected keywords and cached GND data to determine
    appropriate classifications.
    """

    @property
    def agent_name(self) -> str:
        return "DK Classification Agent"

    @property
    def agent_id(self) -> str:
        return "classification"

    def get_system_prompt(self) -> str:
        """System prompt for classification."""
        return """Du bist ein **Experte für DK-Klassifikation (Dewey-Dezimalklassifikation)** mit tiefem Verständnis bibliothekarischer Standards.

**Deine Aufgaben:**
1. Analysiere den Abstract und die GND-Schlagwörter
2. Bestimme die passende DK-Hauptklasse (100er-Stelle)
3. Verfeinere auf Unterklasse (10er-Stelle) wenn möglich
4. Gib maximal 3 DK-Notationen an, sortiert nach Relevanz

**Ausgabeformat:**
```json
{
  "dk_classifications": [
    {"code": "530", "title": "Physik", "confidence": 0.9, "reason": "..."},
    {"code": "540", "title": "Chemie", "confidence": 0.7, "reason": "..."}
  ],
  "rvk_classifications": [],
  "reasoning": "Gesamtbegründung der Auswahl"
}
```

**Wichtig:**
- Konfidenzwerte zwischen 0.0 und 1.0
- Begründe jede Klassifikation kurz
- Bevorzuge spezifische über generelle Klassen"""

    def get_available_tools(self) -> List[str]:
        """Tools for classification."""
        return [
            "get_dk_cache",
            "get_classification",
            "search_catalog",
            "get_gnd_entry",
        ]

    def build_user_prompt(self) -> str:
        """Build prompt from abstract and selected keywords."""
        prompt_parts = [
            "**Aufgabe**: Weise DK-Klassifikationen zu.\n",
            f"**Abstract:**\n{self.context.abstract[:2000]}\n",
        ]

        # Include selected keywords
        if self.context.selected_keywords:
            kw_json = json.dumps(self.context.selected_keywords[:30], ensure_ascii=False, indent=2)
            prompt_parts.append(f"\n**Ausgewählte Schlagwörter:**\n{kw_json}\n")

        # Include working title
        if self.context.working_title:
            prompt_parts.append(f"\n**Arbeitstitel:** {self.context.working_title}\n")

        # Include GND entries with DDC codes
        gnd_with_ddc = [e for e in self.context.gnd_entries if e.get("ddc") or e.get("ddcs")]
        if gnd_with_ddc:
            prompt_parts.append(f"\n**GND-Einträge mit DDC ({len(gnd_with_ddc)} gefunden):**\n")
            for entry in gnd_with_ddc[:20]:
                ddc = entry.get("ddc") or entry.get("ddcs", "")
                prompt_parts.append(f"- {entry.get('title')}: DDC {ddc}\n")

        previous_context = self.get_previous_context()
        if previous_context:
            prompt_parts.append(f"\n**Kontext:**\n{previous_context}\n")

        prompt_parts.append("\nBestimme die passendsten DK-Klassifikationen.")
        return "\n".join(prompt_parts)

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        """Parse classification results."""
        parsed = self._extract_json(llm_output)

        return {
            "dk_classifications": parsed.get("dk_classifications", []),
            "rvk_classifications": parsed.get("rvk_classifications", []),
            "reasoning": parsed.get("reasoning", ""),
            "raw_output": llm_output[:500],
        }

    def _update_shared_context(self, result_data: Dict[str, Any]) -> None:
        """Update context with classifications."""
        dk_classes = result_data.get("dk_classifications", [])

        self.context.dk_classifications = dk_classes

        self.logger.info(f"Assigned {len(dk_classes)} DK classifications")
        for cls in dk_classes:
            self.logger.debug(f"  {cls.get('code')}: {cls.get('title')} (conf: {cls.get('confidence', 0)})")