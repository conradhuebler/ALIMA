"""Classification SubAgent - Claude Generated

Assigns DK/RVK classifications to selected keywords.
Uses a deterministic two-phase approach instead of multi-turn tool calling:

Phase 1: Batch-collect DK catalog data (get_dk_cache, search_catalog, get_gnd_entry)
Phase 2: Single LLM call with all pre-loaded DK data in the prompt

This mirrors the hardcoded pipeline's execute_dk_classification() approach
and avoids the 20-iteration AgentLoop bottleneck.
"""

import json
import logging
import re
from typing import Dict, List, Any

from src.core.agents.sub_agents.base_sub_agent import BaseSubAgent, SubAgentResult
from src.core.agents.tool_providers import DKDataProvider

logger = logging.getLogger(__name__)


class ClassificationAgent(BaseSubAgent):
    """Assign DK/RVK classifications.

    Phase 1: Deterministic batch collection of DK/DDC data from tools.
    Phase 2: Single LLM call with all data pre-loaded in the prompt.
    No AgentLoop needed — same pattern as SearchAgent.
    """

    @property
    def agent_name(self) -> str:
        return "DK Classification Agent"

    @property
    def agent_id(self) -> str:
        return "classification"

    # -- Required abstract stubs (not used -- execute() is overridden) --

    def get_system_prompt(self) -> str:
        return ""

    def get_available_tools(self) -> List[str]:
        return ["get_dk_cache", "get_classification", "search_catalog", "get_gnd_entry"]

    def build_user_prompt(self) -> str:
        return ""

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        return {}

    # -- Core: two-phase deterministic execution --

    def execute(self) -> SubAgentResult:
        """Two-phase classification: batch DK data, then single LLM call."""
        if not self.context.selected_keywords and not self.context.abstract:
            logger.warning("ClassificationAgent: no keywords or abstract in context")
            return SubAgentResult(
                success=True,
                data={"dk_classifications": [], "rvk_classifications": [], "reasoning": "No keywords available"},
            )

        # Phase 1: Batch-collect DK/DDC data via DKDataProvider
        dk_provider = DKDataProvider(self.caching_registry, self.context)
        dk_result = dk_provider.collect()

        if self.stream_callback:
            self.stream_callback(
                f"  \U0001f4da DK-Daten gesammelt: "
                f"{len(dk_result.dk_entries)} DK-Einträge, "
                f"{len(dk_result.ddc_from_gnd)} GND-DDC-Einträge "
                f"({dk_result.tool_calls} Tool-Calls)\n"
            )

        # Phase 2: Single LLM call with all data pre-loaded
        result = self._classify_with_llm(dk_result)

        return result

    def _classify_with_llm(self, dk_data: "DKDataResult") -> SubAgentResult:
        """Phase 2: Single LLM call with all DK data pre-loaded in the prompt."""
        # Build user prompt with DK data
        prompt_parts = [
            f"**Aufgabe**: Weise DK-Klassifikationen zu.\n",
            f"**Abstract:**\n{self.context.abstract[:2000]}\n",
        ]

        # Include selected keywords
        if self.context.selected_keywords:
            kw_json = json.dumps(self.context.selected_keywords, ensure_ascii=False, indent=2)
            prompt_parts.append(f"\n**Ausgewählte Schlagwörter:**\n{kw_json}\n")

        # Include working title
        if self.context.working_title:
            prompt_parts.append(f"\n**Arbeitstitel:** {self.context.working_title}\n")

        # Include DK data using DKDataResult.format_for_prompt()
        dk_prompt_text = dk_data.format_for_prompt(max_entries=30)
        if dk_prompt_text:
            prompt_parts.append(dk_prompt_text)

        # Include previous context
        previous_context = self.get_previous_context()
        if previous_context:
            prompt_parts.append(f"\n**Kontext:**\n{previous_context}\n")

        prompt_parts.append(
            "\n**Strukturierte Vorgehensweise:**\n"
            "1. **Themenextraktion**: Identifiziere Kernkonzepte im Abstract und verknüpfe sie mit den GND-Schlagworten.\n"
            "2. **Klassifikationsabgleich**: Wähle ausschließlich Klassifikationen, die im bereitgestellten Bibliotheksbestand aufgeführt sind. Erfinde keine Codes.\n"
            "3. **Validierung**: Prüfe Präzision, Hierarchie und Einzigartigkeit.\n"
            "4. **Auswahl**: Wähle die relevantesten Klassifikationen (max. 10).\n\n"
            "Bestimme die passendsten DK-Klassifikationen und gib das Ergebnis als JSON zurück."
        )

        system_prompt = """\
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

        user_prompt = "\n".join(prompt_parts)

        if self.stream_callback:
            tool_info = f" (Tools: {', '.join(self.get_available_tools()[:2])}...)" if self.get_available_tools() else ""
            self.stream_callback(
                f"\n{'='*50}\n\U0001f916 {self.agent_name}{tool_info}\n{'='*50}\n"
            )
            self.stream_callback(f"\U0001f4cb Aufgabe: {system_prompt.split(chr(10))[0][:80]}\n")
            self.stream_callback(f"\U0001f4dd Input: {user_prompt.split(chr(10))[0][:80]}...\n")
            self.stream_callback(f"\u23f3 Sende an LLM ({self.context.provider}/{self.context.model})...\n\n")

        if self.context.verbose:
            self._log_prompt_verbose(system_prompt, user_prompt, label="Classification")

        # Single LLM call — no AgentLoop, no tool calling
        try:
            tokens: List[str] = []

            def _collect(token: str) -> None:
                tokens.append(token)

            response = self.llm_service.generate_with_tools(
                provider=self.context.provider,
                model=self.context.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[],  # No tools — single LLM call with all data pre-loaded
                temperature=self.context.temperature,
                max_tokens=self.context.max_tokens,
                stream_callback=_collect,
            )

            raw_output = response.content if response else "".join(tokens)

        except Exception as e:
            logger.error(f"ClassificationAgent LLM call failed: {e}")
            return SubAgentResult(success=False, data={}, error=str(e))

        # Parse result
        parsed = self._extract_json(raw_output)
        dk_classifications = parsed.get("dk_classifications", [])
        rvk_classifications = parsed.get("rvk_classifications", [])
        reasoning = parsed.get("reasoning", "")

        result_data = {
            "dk_classifications": dk_classifications,
            "rvk_classifications": rvk_classifications,
            "reasoning": reasoning,
            "raw_output": raw_output[:500],
        }

        # Update shared context
        self.context.dk_classifications = dk_classifications
        self.context.rvk_classifications = rvk_classifications

        # Build dk_search_results from collected DK data
        dk_search_results = self._build_dk_search_results(dk_data, dk_classifications)
        if dk_search_results:
            self.context.dk_search_results = dk_search_results

        self.context.set_step_result(self.agent_id, result_data, quality=1.0)

        if self.stream_callback:
            self.stream_callback(
                f"\u2705 {len(dk_classifications)} DK-Klassifikationen, "
                f"{len(rvk_classifications)} RVK-Klassifikationen zugewiesen\n"
            )

        self.logger.info(
            f"Assigned {len(dk_classifications)} DK classifications, "
            f"{len(rvk_classifications)} RVK classifications"
        )

        return SubAgentResult(
            success=True,
            data=result_data,
            quality_score=1.0,
            iterations=1,  # Single LLM call, no iteration
            tool_calls=dk_data.tool_calls,
        )

    def _build_dk_search_results(
        self,
        dk_data: "DKDataResult",
        dk_classifications: List[Dict],
    ) -> List[Dict]:
        """Build dk_search_results from collected DK data and LLM classifications.

        Merges DK cache data with LLM-assigned classifications into the
        keyword-centric format expected by the GUI.
        """
        results = []

        # Add DK entries from DKDataResult (deduplicated)
        seen_dk = set()
        for entry in dk_data.dk_entries:
            dk_code = entry.get("dk", "")
            if dk_code and dk_code not in seen_dk:
                seen_dk.add(dk_code)
                results.append({
                    "keyword": entry.get("keyword", ""),
                    "dk": dk_code,
                    "title": entry.get("title", ""),
                    "count": entry.get("count", 0),
                    "classification_type": entry.get("classification_type", "DK"),
                })

        # Add LLM-assigned classifications (may overlap — that's fine)
        for cls in dk_classifications:
            code = cls.get("code", "")
            title = cls.get("title", "")
            if code:
                results.append({
                    "keyword": "",
                    "dk": code,
                    "title": title,
                    "count": int(cls.get("confidence", 0) * 100),
                    "classification_type": "DK",
                    "reasoning": cls.get("reason", cls.get("reasoning", "")),
                })

        return results

    def _update_shared_context(self, result_data: Dict[str, Any]) -> None:
        """Not used — execute() updates context directly."""
        pass