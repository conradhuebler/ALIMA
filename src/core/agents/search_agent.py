"""SearchAgent: GND/SWB search using MCP tools - Claude Generated

Uses search_lobid, search_swb, search_gnd, get_search_cache tools
to build a comprehensive GND keyword pool from an abstract.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Callable

from src.core.agents.base_agent import BaseAgent, AgentConfig, QualityMetrics
from src.core.data_models import AgentResult
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    Searches GND/SWB databases for subject headings matching an abstract.

    Strategy: LLM reads abstract → generates search terms → uses tools to search
    Lobid, SWB, and local GND → compiles keyword pool with GND IDs and DDC codes.
    """

    @property
    def agent_name(self) -> str:
        return "SearchAgent"

    def get_system_prompt(self) -> str:
        return """Du bist ein GND-Suchexperte für die Bibliothekserschließung. Deine Aufgabe ist es, zu einem gegebenen Abstract möglichst viele relevante GND-Schlagwörter zu finden.

**Dein Vorgehen:**
1. Lies den Abstract sorgfältig und identifiziere die Kernthemen
2. Generiere Suchbegriffe (deutsch und englisch) für die wichtigsten Konzepte
3. Nutze die verfügbaren Such-Tools systematisch:
   - `search_gnd`: Lokale GND-Datenbank durchsuchen
   - `search_lobid`: Lobid.org API für GND-Einträge
   - `search_swb`: SWB-Katalog für Schlagwörter mit GND-IDs
   - `get_search_cache`: Prüfe ob Ergebnisse bereits gecacht sind
4. Sammle alle gefundenen GND-Einträge mit IDs, Titeln und DDC-Codes

**Wichtige Regeln:**
- Suche breiter als nur offensichtliche Begriffe - auch verwandte und übergeordnete Konzepte
- Nutze sowohl Fachbegriffe als auch Synonyme
- Prüfe den Cache bevor du eine Web-Suche startest
- Gib am Ende eine strukturierte Liste aller gefundenen GND-Einträge zurück

**Ausgabeformat:**
Gib dein Ergebnis als JSON zurück:
```json
{
  "search_terms_used": ["Begriff1", "Begriff2", ...],
  "gnd_entries": [
    {"gnd_id": "...", "title": "...", "ddcs": "...", "source": "lobid|swb|local"},
    ...
  ],
  "coverage_assessment": "Kurze Einschätzung der Abdeckung"
}
```"""

    def get_available_tools(self) -> List[str]:
        return ["search_gnd", "search_lobid", "search_swb", "get_search_cache", "store_search_result"]

    def build_user_prompt(self, input_data: Dict[str, Any], **kwargs) -> str:
        abstract = input_data.get("abstract", "")
        initial_keywords = input_data.get("initial_keywords", [])
        iteration = kwargs.get("iteration", 1)

        prompt = f"Finde GND-Schlagwörter für folgenden Abstract:\n\n{abstract}\n"

        if initial_keywords:
            prompt += f"\nBereits identifizierte freie Schlagwörter: {', '.join(initial_keywords)}\n"
            prompt += "Nutze diese als Ausgangspunkt für die Suche, aber erweitere auch darüber hinaus.\n"

        if iteration > 1:
            prev_results = input_data.get("_previous_gnd_entries", [])
            prompt += (
                f"\nDies ist Suchiteration {iteration}. "
                f"Bisherige Ergebnisse: {len(prev_results)} GND-Einträge gefunden.\n"
                "Versuche zusätzliche Begriffe und Synonyme, um die Abdeckung zu erhöhen.\n"
            )

        return prompt

    def parse_result(self, agent_result: AgentResult, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse search results from agent output."""
        content = agent_result.content

        # Try to parse JSON from response
        try:
            # Find JSON block in response
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(content)

            return {
                "search_terms": parsed.get("search_terms_used", []),
                "gnd_entries": parsed.get("gnd_entries", []),
                "coverage_assessment": parsed.get("coverage_assessment", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"SearchAgent: Could not parse JSON from response, extracting from tool log")

        # Fallback: extract from tool log
        gnd_entries = []
        search_terms = []
        for log_entry in agent_result.tool_log:
            if log_entry["tool"] in ("search_lobid", "search_swb", "search_gnd"):
                args = log_entry.get("arguments", {})
                terms = args.get("terms", []) or [args.get("term", "")]
                search_terms.extend(terms)

                # Try to extract entries from result
                try:
                    result_data = json.loads(log_entry.get("result_preview", "{}"))
                    if "entries" in result_data:
                        for entry in result_data["entries"]:
                            gnd_entries.append(entry)
                    elif "results" in result_data:
                        for term, keywords in result_data["results"].items():
                            for kw, data in keywords.items():
                                gnd_ids = data.get("gndid", [])
                                for gid in gnd_ids:
                                    gnd_entries.append({"gnd_id": gid, "title": kw, "source": log_entry["tool"]})
                except (json.JSONDecodeError, TypeError):
                    pass

        return {
            "search_terms": list(set(search_terms)),
            "gnd_entries": gnd_entries,
            "coverage_assessment": content[:200] if content else "",
        }

    def self_validate(self, result: Dict[str, Any], input_data: Dict[str, Any]) -> QualityMetrics:
        """Validate search coverage."""
        gnd_entries = result.get("gnd_entries", [])
        search_terms = result.get("search_terms", [])
        initial_keywords = input_data.get("initial_keywords", [])

        # Coverage: ratio of initial keywords that led to at least one GND result
        covered = 0
        if initial_keywords:
            gnd_titles = {e.get("title", "").lower() for e in gnd_entries}
            for kw in initial_keywords:
                if any(kw.lower() in t for t in gnd_titles):
                    covered += 1
            coverage = covered / len(initial_keywords) if initial_keywords else 0
        else:
            coverage = min(len(gnd_entries) / 10, 1.0)  # At least 10 entries expected

        # Diversity: unique GND IDs vs total
        gnd_ids = [e.get("gnd_id", "") for e in gnd_entries if e.get("gnd_id")]
        diversity = len(set(gnd_ids)) / max(len(gnd_ids), 1)

        # Relevance approximation: number of entries found
        relevance = min(len(gnd_entries) / 20, 1.0)

        score = (coverage * 0.4 + relevance * 0.3 + diversity * 0.3)

        return QualityMetrics(
            score=score, coverage=coverage, relevance=relevance, diversity=diversity,
            details={"gnd_count": len(gnd_entries), "search_terms_count": len(search_terms)},
        )
