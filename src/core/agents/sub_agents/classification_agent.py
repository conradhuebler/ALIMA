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

        # Phase 1: Batch-collect DK/DDC data
        dk_data = self._collect_dk_data()

        # Phase 2: Single LLM call with all data pre-loaded
        result = self._classify_with_llm(dk_data)

        return result

    def _collect_dk_data(self) -> Dict[str, Any]:
        """Phase 1: Batch-collect DK/DDC data from tools.

        Collects data from:
        - get_dk_cache: For each selected keyword (returns {cached, term, titles: [...]})
        - search_catalog: For the abstract's main terms (returns {source, results: {term: {kw: {...}}}})
        - GND entries with DDC codes: Already in context

        Returns:
            Dict with dk_cache_results, catalog_results, ddc_entries
        """
        dk_cache_results: List[Dict] = []
        catalog_results: List[Dict] = []
        tool_calls = 0

        # 1. DK cache lookups for each selected keyword
        # get_dk_cache returns: {"cached": true, "term": "...", "titles": [...], "status": "success"}
        # Each title dict has: {"title": "...", "classifications": ["DK 540", "RVK VK 8000"], ...}
        seen_terms = set()
        for kw in self.context.selected_keywords[:30]:
            term = kw.get("title", "")
            if not term or term.lower() in seen_terms:
                continue
            seen_terms.add(term.lower())

            try:
                raw = self.caching_registry.execute("get_dk_cache", {"term": term})
                data = json.loads(raw) if isinstance(raw, str) else raw
                tool_calls += 1

                if isinstance(data, dict) and data.get("cached"):
                    titles = data.get("titles", [])
                    if isinstance(titles, list):
                        for title_entry in titles:
                            if not isinstance(title_entry, dict):
                                continue
                            # Extract DK codes from classifications list
                            classifications = title_entry.get("classifications", [])
                            if isinstance(classifications, list):
                                for cls in classifications:
                                    cls_str = str(cls)
                                    # Parse "DK 540" or plain "540" patterns
                                    dk_match = re.match(r'(?:DK\s+)?(\d[\d.]+)', cls_str)
                                    rvk_match = re.match(r'RVK\s+(.+)', cls_str)
                                    if dk_match:
                                        dk_cache_results.append({
                                            "keyword": term,
                                            "dk": dk_match.group(1),
                                            "title": title_entry.get("title", ""),
                                            "count": title_entry.get("count", 0),
                                            "classification_type": "DK",
                                        })
                                    elif rvk_match:
                                        dk_cache_results.append({
                                            "keyword": term,
                                            "dk": rvk_match.group(1),
                                            "title": title_entry.get("title", ""),
                                            "count": title_entry.get("count", 0),
                                            "classification_type": "RVK",
                                        })
            except Exception as e:
                logger.debug(f"get_dk_cache('{term}') failed: {e}")

        # 2. Catalog search for main terms
        # search_catalog returns: {"source": "catalog", "results": {term: {kw: {"count": N, "gndid": [...], "ddc": [...], "dk": [...]}}}}
        if self.context.extracted_keywords:
            try:
                raw = self.caching_registry.execute("search_catalog", {
                    "terms": self.context.extracted_keywords[:10],
                })
                data = json.loads(raw) if isinstance(raw, str) else raw
                tool_calls += 1

                if isinstance(data, dict):
                    results = data.get("results", {})
                    for term, term_results in results.items():
                        if not isinstance(term_results, dict):
                            continue
                        for kw_title, kw_data in term_results.items():
                            if not isinstance(kw_data, dict):
                                continue
                            # Extract DK codes from kw_data["dk"] and kw_data["ddc"]
                            dk_codes = kw_data.get("dk", [])
                            ddc_codes = kw_data.get("ddc", [])
                            count = kw_data.get("count", 0)
                            for dk in dk_codes:
                                if dk:
                                    catalog_results.append({
                                        "keyword": kw_title,
                                        "dk": str(dk),
                                        "title": kw_title,
                                        "count": count,
                                        "classification_type": "DK",
                                    })
                            for ddc in ddc_codes:
                                if ddc:
                                    catalog_results.append({
                                        "keyword": kw_title,
                                        "dk": str(ddc),
                                        "title": kw_title,
                                        "count": count,
                                        "classification_type": "DDC",
                                    })
            except Exception as e:
                logger.debug(f"search_catalog failed: {e}")

        # 3. GND entries with DDC codes (already in context)
        ddc_entries = [
            e for e in self.context.gnd_entries
            if e.get("ddc_codes") or e.get("dk_codes")
        ]
        # Sort by frequency
        ddc_entries.sort(key=lambda e: e.get("count", 0), reverse=True)

        if self.stream_callback:
            self.stream_callback(
                f"  \U0001f4da DK-Daten gesammelt: "
                f"{len(dk_cache_results)} Cache-Einträge, "
                f"{len(catalog_results)} Katalog-Einträge, "
                f"{len(ddc_entries)} GND-DDC-Einträge\n"
            )

        return {
            "dk_cache_results": dk_cache_results,
            "catalog_results": catalog_results,
            "ddc_entries": ddc_entries[:50],  # Top 50 by frequency
            "tool_calls": tool_calls,
        }

    def _classify_with_llm(self, dk_data: Dict[str, Any]) -> SubAgentResult:
        """Phase 2: Single LLM call with all DK data pre-loaded in the prompt."""
        # Build user prompt with all DK data
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

        # Include DK cache results
        dk_cache = dk_data.get("dk_cache_results", [])
        if dk_cache:
            prompt_parts.append(f"\n**DK-Cache ({len(dk_cache)} Einträge):**\n")
            for entry in dk_cache[:30]:
                dk_code = entry.get("dk", entry.get("classification", entry.get("code", "")))
                title = entry.get("title", "")
                count = entry.get("count", 0)
                prompt_parts.append(f"- {dk_code}: {title} (Häufigkeit: {count})\n")

        # Include catalog results with DK/DDC codes
        catalog = dk_data.get("catalog_results", [])
        if catalog:
            prompt_parts.append(f"\n**Katalog-Ergebnisse ({len(catalog)} Einträge):**\n")
            for entry in catalog[:20]:
                dk_code = entry.get("dk", entry.get("classification", ""))
                title = entry.get("title", "")
                prompt_parts.append(f"- {dk_code}: {title}\n")

        # Include GND entries with DDC codes
        ddc_entries = dk_data.get("ddc_entries", [])
        if ddc_entries:
            prompt_parts.append(f"\n**GND-Einträge mit DDC ({len(ddc_entries)}):**\n")
            for entry in ddc_entries[:50]:
                ddc = entry.get("ddc_codes", [])
                title = entry.get("title", "")
                prompt_parts.append(f"- {title}: DDC {ddc}\n")

        # Include previous context
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
            tool_calls=dk_data.get("tool_calls", 0),
        )

    def _build_dk_search_results(
        self,
        dk_data: Dict[str, Any],
        dk_classifications: List[Dict],
    ) -> List[Dict]:
        """Build dk_search_results from collected DK data and LLM classifications.

        Merges DK cache data with LLM-assigned classifications into the
        keyword-centric format expected by the GUI.
        """
        results = []

        # Add DK cache results
        seen_dk = set()
        for entry in dk_data.get("dk_cache_results", []):
            dk_code = entry.get("dk", entry.get("classification", entry.get("code", "")))
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