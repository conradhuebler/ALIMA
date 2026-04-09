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
        """System prompt for classification (from prompts.json 'dk_classification')."""
        return """\
**Deine Rolle als bibliothekarischer Klassifikations-Experte:**
Du bist ein **präziser Klassifikator** mit folgenden Kernkompetenzen:
1. **Systematische Analyse**: Kombiniere Abstract, Schlagworte und bestehende Klassifikationen aus dem Bibliotheksbestand zu einer **hierarchischen Themenstruktur**.
2. **Mehrfachklassifikation**: Wähle **bis zu 10 passende Klassifikationsnummern** (DK/DDC/RVK) aus, die:
  - **Primärthemen** des Abstracts abdecken,
  - **Sekundärbeziehungen** (z. B. Anwendungsfelder, Methodik) einbeziehen,
  - **Bestehende Klassifikationen** aus dem Bibliotheksbestand validieren/ergänzen.
3. **Klassifikationsregeln**:
  - Verwende nur Klassifikationen, die mit der Titelliste auftauchen.
  - **Typischerweise sind die Häufigkeiten ein guter Indikator für die Relevanz der Klassifikation.** Gib ggf. zu den 10 Klassifikatoren noch 5 weitere potentielle Klassifikatoren an, die eine geringe Häufigkeit aufweisen.
4. **Iterativer Prozess**: Überprüfe jede Klassifikation auf:
  - **Thematische Passung** (deckt der Code das Kernkonzept ab?),
  - **Hierarchische Konsistenz** (passt der Code zur übergeordneten Ebene?),
  - **Redundanz** (vermeide doppelte Abdeckung desselben Themas).
5. **JSON-Ausgabe**: Gib das Ergebnis immer als valides JSON-Objekt aus:
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

**Werkzeuge & Limits:**
- Nutze **nur** die im Bibliotheksbestand enthaltenen Klassifikationen als Referenz – keine externen Quellen.
- Konfidenzwerte zwischen 0.0 und 1.0.
- Begründe jede Klassifikation kurz."""

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

        # Include selected keywords (all — typically ≤20)
        if self.context.selected_keywords:
            kw_json = json.dumps(self.context.selected_keywords, ensure_ascii=False, indent=2)
            prompt_parts.append(f"\n**Ausgewählte Schlagwörter:**\n{kw_json}\n")

        # Include working title
        if self.context.working_title:
            prompt_parts.append(f"\n**Arbeitstitel:** {self.context.working_title}\n")

        # Include GND entries with DDC codes (field is ddc_codes from SearchAgent)
        gnd_with_ddc = [e for e in self.context.gnd_entries if e.get("ddc_codes")]
        if gnd_with_ddc:
            gnd_with_ddc.sort(key=lambda e: e.get("count", 0), reverse=True)
            prompt_parts.append(f"\n**GND-Einträge mit DDC ({len(gnd_with_ddc)} gefunden, top 50):**\n")
            for entry in gnd_with_ddc[:50]:
                ddc = entry.get("ddc_codes", [])
                prompt_parts.append(f"- {entry.get('title')}: DDC {ddc}\n")

        previous_context = self.get_previous_context()
        if previous_context:
            prompt_parts.append(f"\n**Kontext:**\n{previous_context}\n")

        prompt_parts.append(
            "\n**Strukturierte Vorgehensweise:**\n"
            "1. **Themenextraktion**: Identifiziere Kernkonzepte im Abstract und verknüpfe sie mit den GND-Schlagworten.\n"
            "2. **Klassifikationsabgleich**: Übernimm Klassifikationen aus dem Bibliotheksbestand als Basis; ergänze fehlende durch logische Ableitung.\n"
            "3. **Validierung**: Prüfe Präzision, Hierarchie und Einzigartigkeit.\n"
            "4. **Auswahl**: Wähle die relevantesten Klassifikationen (max. 10).\n\n"
            "Bestimme die passendsten DK-Klassifikationen und gib das Ergebnis als JSON zurück."
        )
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