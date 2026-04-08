"""Search SubAgent - Claude Generated

Deterministic GND/SWB/Lobid search — no LLM iteration needed.

Strategy:
  1. search_swb(all_keywords)   — batch, one call
  2. search_lobid(all_keywords) — batch, one call
  3. get_gnd_batch(all_found_ids) — fetch full GND details

Tool calls are made directly without AgentLoop, matching the behaviour
of the original sequential pipeline.
"""

import json
import logging
from typing import Dict, List, Any, Set

from src.core.agents.sub_agents.base_sub_agent import BaseSubAgent, SubAgentResult

logger = logging.getLogger(__name__)


class SearchAgent(BaseSubAgent):
    """Search GND/SWB/Lobid catalogs deterministically.

    Overrides execute() to bypass AgentLoop: tools are called in a fixed
    sequence without LLM involvement.  The LLM is not needed here because
    the search strategy is always the same regardless of content.
    """

    @property
    def agent_name(self) -> str:
        return "GND Search Agent"

    @property
    def agent_id(self) -> str:
        return "search"

    # ── Required abstract stubs (not used — execute() is overridden) ──

    def get_system_prompt(self) -> str:
        return ""

    def get_available_tools(self) -> List[str]:
        return ["search_swb", "search_lobid", "get_gnd_batch"]

    def build_user_prompt(self) -> str:
        return ""

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        return {}

    # ── Core: deterministic execution without AgentLoop ──

    def execute(self) -> SubAgentResult:
        """Execute GND search without LLM iteration.

        Calls search_swb and search_lobid with all keywords at once,
        collects unique GND IDs, then fetches full entries via get_gnd_batch.
        """
        # Base keywords + any missing concepts from a previous selection round
        base = self.context.extracted_keywords or self.context.initial_keywords
        keywords = list(dict.fromkeys(base + self.context.missing_concepts))
        if not keywords:
            logger.warning("SearchAgent: no keywords in context, skipping search")
            return SubAgentResult(success=True, data={"gnd_entries": []})

        if self.stream_callback:
            self.stream_callback(
                f"\n{'='*50}\n🔍 {self.agent_name}\n{'='*50}\n"
                f"📋 {len(keywords)} Keywords → SWB + Lobid (Batch)\n"
            )

        all_gnd_ids: Set[str] = set()
        tool_calls = 0

        # ── 1. SWB batch search ──
        all_gnd_ids.update(self._run_batch("search_swb", keywords))
        tool_calls += 1

        # ── 2. Lobid batch search ──
        all_gnd_ids.update(self._run_batch("search_lobid", keywords))
        tool_calls += 1

        if self.stream_callback:
            self.stream_callback(f"✅ {len(all_gnd_ids)} GND-IDs gefunden\n")

        # ── 3. Fetch full GND entries for all found IDs ──
        gnd_entries: List[Dict[str, Any]] = []
        if all_gnd_ids:
            gnd_entries = self._fetch_gnd_entries(list(all_gnd_ids))
            tool_calls += 1

        if self.stream_callback:
            self.stream_callback(f"📚 {len(gnd_entries)} GND-Einträge geladen\n")

        # Update shared context
        existing_ids = {e.get("gnd_id") for e in self.context.gnd_entries}
        for entry in gnd_entries:
            if entry.get("gnd_id") not in existing_ids:
                self.context.gnd_entries.append(entry)
                existing_ids.add(entry.get("gnd_id"))

        result_data = {
            "gnd_entries": gnd_entries,
            "search_terms": keywords,
            "coverage": f"{len(gnd_entries)} GND-Einträge für {len(keywords)} Keywords",
        }
        self.context.set_step_result(self.agent_id, result_data, quality=1.0)

        return SubAgentResult(
            success=True,
            data=result_data,
            quality_score=1.0,
            iterations=1,
            tool_calls=tool_calls,
        )

    def _run_batch(self, tool_name: str, keywords: List[str]) -> Set[str]:
        """Run a batch search tool and extract all GND IDs from the result."""
        try:
            raw = self.caching_registry.execute(tool_name, {"terms": keywords})
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"SearchAgent: {tool_name} failed: {e}")
            return set()

        gnd_ids: Set[str] = set()
        results = data.get("results", {})

        # Structure: {term: {keyword: {gndid: [...], ...}}}
        for term_results in results.values():
            if not isinstance(term_results, dict):
                continue
            for kw_data in term_results.values():
                if not isinstance(kw_data, dict):
                    continue
                for gid in kw_data.get("gndid", []):
                    if gid:
                        gnd_ids.add(str(gid))

        if self.stream_callback:
            source = data.get("source", tool_name)
            self.stream_callback(f"  🌐 {source}: {len(gnd_ids)} IDs\n")

        return gnd_ids

    def _fetch_gnd_entries(self, gnd_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full GND entry details for a list of GND IDs."""
        try:
            raw = self.caching_registry.execute("get_gnd_batch", {"gnd_ids": gnd_ids})
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"SearchAgent: get_gnd_batch failed: {e}")
            return []

        entries = []
        for gnd_id, entry in data.get("entries", {}).items():
            entries.append({
                "gnd_id": gnd_id,
                "title": entry.get("title", ""),
                "description": entry.get("description", ""),
                "synonyms": entry.get("synonyms", ""),
                "ddc_codes": entry.get("ddcs", ""),
            })
        return entries
