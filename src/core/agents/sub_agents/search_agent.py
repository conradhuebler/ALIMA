"""Search SubAgent - Claude Generated

Searches GND/SWB/LOBID for subject headings.
Uses cached results to avoid redundant searches.
"""

import json
import logging
from typing import Dict, List, Any

from src.core.agents.sub_agents.base_sub_agent import BaseSubAgent, SubAgentResult
from src.core.agents.shared_context import SharedContext

logger = logging.getLogger(__name__)


class SearchAgent(BaseSubAgent):
    """Search GND/SWB catalogs for subject headings.

    Uses caching to avoid redundant searches. Previous search results
    are reused when the same terms are searched again.
    """

    @property
    def agent_name(self) -> str:
        return "GND Search Agent"

    @property
    def agent_id(self) -> str:
        return "search"

    def get_system_prompt(self) -> str:
        """System prompt for GND search."""
        return """Du bist ein GND-Suchexperte für die Bibliothekserschließung.

Deine Aufgabe ist es, zu einem gegebenen Abstract und Schlagwortliste relevante GND-Einträge zu finden.

**Dein Vorgehen:**
1. Lies den Abstract und die gegebenen Keywords
2. Generiere Suchbegriffe (deutsch und englisch) für die wichtigsten Konzepte
3. Nutze die Such-Tools systematisch:
   - `search_gnd`: Lokale GND-Datenbank durchsuchen
   - `search_lobid`: Lobid.org API für GND-Einträge
   - `search_swb`: SWB-Katalog für Schlagwörter mit GND-IDs
4. Sammle alle gefundenen GND-Einträge mit IDs, Titeln und DDC-Codes

**Wichtige Regeln:**
- Suche breit - auch verwandte und übergeordnete Konzepte
- Nutze Fachbegriffe UND Synonyme
- Prüfe den Cache bevor du eine Web-Suche startest (Tools nutzen automatisch Cache)

**Ausgabeformat:**
Gib dein Ergebnis als JSON zurück:
```json
{
  "search_terms_used": ["Begriff1", "Begriff2", ...],
  "gnd_entries": [
    {"gnd_id": "...", "title": "...", "ddc": "...", "source": "lobid|swb|local"},
    ...
  ],
  "coverage_assessment": "Kurze Einschätzung der Abdeckung"
}
```"""

    def get_available_tools(self) -> List[str]:
        """Tools for GND/SWB search."""
        return [
            "search_gnd",
            "search_lobid",
            "search_swb",
            "get_search_cache",
            "get_gnd_entry",
            "get_gnd_batch",
        ]

    def build_user_prompt(self) -> str:
        """Build prompt from abstract and extracted keywords."""
        prompt_parts = [
            "Finde GND-Schlagwörter für folgende Analyse:\n",
            f"**Abstract:**\n{self.context.abstract[:2000]}\n",
        ]

        if self.context.extracted_keywords:
            prompt_parts.append(f"\n**Extrahierte Keywords:**\n{', '.join(self.context.extracted_keywords[:30])}\n")

        if self.context.initial_keywords:
            prompt_parts.append(f"\n**Vorhandene Keywords:**\n{', '.join(self.context.initial_keywords[:20])}\n")

        previous_context = self.get_previous_context()
        if previous_context:
            prompt_parts.append(f"\n**Kontext aus vorherigen Schritten:**\n{previous_context}\n")

        prompt_parts.append("\nFühre systematische Suchen durch und sammle relevante GND-Einträge.")
        return "\n".join(prompt_parts)

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        """Parse search results."""
        parsed = self._extract_json(llm_output)

        gnd_entries = parsed.get("gnd_entries", [])
        search_terms = parsed.get("search_terms_used", [])

        return {
            "gnd_entries": gnd_entries,
            "search_terms": search_terms,
            "coverage": parsed.get("coverage_assessment", ""),
            "raw_output": llm_output[:500],
        }

    def _update_shared_context(self, result_data: Dict[str, Any]) -> None:
        """Update context with GND entries."""
        gnd_entries = result_data.get("gnd_entries", [])

        # Add to shared context (accumulate, don't replace)
        existing_ids = {e.get("gnd_id") for e in self.context.gnd_entries}

        for entry in gnd_entries:
            if entry.get("gnd_id") not in existing_ids:
                self.context.gnd_entries.append(entry)
                existing_ids.add(entry.get("gnd_id"))

        self.logger.info(f"Found {len(gnd_entries)} GND entries (total: {len(self.context.gnd_entries)})")
        self.logger.debug(f"Search terms used: {result_data.get('search_terms', [])}")