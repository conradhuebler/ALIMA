"""Search SubAgent - Claude Generated

Deterministic GND/SWB/Lobid search — no LLM iteration needed.

Strategy:
  1. search_swb(all_keywords)   — batch, one call
  2. search_lobid(all_keywords) — batch, one call
  3. Merge keyword-level data directly (title, gnd_ids, ddc, dk, count)
  4. Optionally enrich with get_gnd_batch for description/synonyms

This mirrors the normal pipeline: GND metadata comes from the web API
response directly, not from the local DB. The local DB (get_gnd_batch)
is used only for enrichment (description, synonyms) where available.
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
        collects keyword-level data (title, GND IDs, DDC/DK codes) directly
        from the web API responses, then optionally enriches with local DB
        data (description, synonyms) for entries that exist there.

        This matches the normal pipeline's approach: metadata comes from
        the web search result, not exclusively from the local DB.
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

        # {title_lower: entry_dict} — accumulate across both sources
        keyword_pool: Dict[str, Dict[str, Any]] = {}
        tool_calls = 0

        # ── 1. SWB batch search ──
        swb_data = self._run_batch("search_swb", keywords)
        self._merge_into_pool(keyword_pool, swb_data)
        tool_calls += 1

        # ── 2. Lobid batch search ──
        lobid_data = self._run_batch("search_lobid", keywords)
        self._merge_into_pool(keyword_pool, lobid_data)
        tool_calls += 1

        if self.stream_callback:
            total_gnd_ids = sum(len(e.get("gnd_ids", [])) for e in keyword_pool.values())
            self.stream_callback(
                f"✅ {len(keyword_pool)} GND-Schlagworte gefunden ({total_gnd_ids} GND-IDs)\n"
            )

        # ── 3. Enrich with local DB (description, synonyms) ──
        all_gnd_ids: Set[str] = set()
        for entry in keyword_pool.values():
            all_gnd_ids.update(entry.get("gnd_ids", []))

        if all_gnd_ids:
            enrichment = self._fetch_local_enrichment(list(all_gnd_ids))
            tool_calls += 1
            self._apply_enrichment(keyword_pool, enrichment)

        # ── 4. Build final gnd_entries list ──
        gnd_entries: List[Dict[str, Any]] = list(keyword_pool.values())

        if self.stream_callback:
            enriched = sum(1 for e in gnd_entries if e.get("description"))
            self.stream_callback(
                f"📚 {len(gnd_entries)} GND-Einträge"
                f" ({enriched} mit lokaler Beschreibung)\n"
            )

        # Update shared context (deduplicate by gnd_id / title)
        existing_titles = {e.get("title", "").lower() for e in self.context.gnd_entries}
        for entry in gnd_entries:
            if entry.get("title", "").lower() not in existing_titles:
                self.context.gnd_entries.append(entry)
                existing_titles.add(entry.get("title", "").lower())

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

    def _run_batch(self, tool_name: str, keywords: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run a batch search tool and return full keyword-level data.

        Returns:
            {keyword_title: {gnd_ids: [str], ddc_codes: [str], dk_codes: [str], count: int}}
        """
        try:
            raw = self.caching_registry.execute(tool_name, {"terms": keywords})
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"SearchAgent: {tool_name} failed: {e}")
            return {}

        keyword_data: Dict[str, Dict[str, Any]] = {}
        results = data.get("results", {})

        # Structure: {term: {keyword_title: {gndid: [...], ddc: [...], dk: [...], count: int}}}
        for term_results in results.values():
            if not isinstance(term_results, dict):
                continue
            for kw_title, kw_data in term_results.items():
                if not isinstance(kw_data, dict):
                    continue
                gnd_ids = [str(gid) for gid in kw_data.get("gndid", []) if gid]
                if not gnd_ids and not kw_title:
                    continue
                keyword_data[kw_title] = {
                    "title": kw_title,
                    "gnd_ids": gnd_ids,
                    "gnd_id": gnd_ids[0] if gnd_ids else "",
                    "ddc_codes": list(kw_data.get("ddc", [])),
                    "dk_codes": list(kw_data.get("dk", [])),
                    "count": kw_data.get("count", 0),
                    "description": "",
                    "synonyms": [],
                }

        source = data.get("source", tool_name)
        if self.stream_callback:
            self.stream_callback(f"  🌐 {source}: {len(keyword_data)} Schlagworte\n")

        return keyword_data

    def _merge_into_pool(
        self,
        pool: Dict[str, Dict[str, Any]],
        new_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """Merge new keyword data into the pool, unioning GND IDs on collision."""
        for title, entry in new_data.items():
            title_lower = title.lower()
            if title_lower in pool:
                existing = pool[title_lower]
                existing_ids = set(existing.get("gnd_ids", []))
                existing_ids.update(entry.get("gnd_ids", []))
                existing["gnd_ids"] = list(existing_ids)
                if existing_ids and not existing.get("gnd_id"):
                    existing["gnd_id"] = next(iter(existing_ids))
                # Union DDC/DK codes
                existing["ddc_codes"] = list(
                    set(existing.get("ddc_codes", [])) | set(entry.get("ddc_codes", []))
                )
                existing["dk_codes"] = list(
                    set(existing.get("dk_codes", [])) | set(entry.get("dk_codes", []))
                )
                existing["count"] = max(existing.get("count", 0), entry.get("count", 0))
            else:
                pool[title_lower] = dict(entry)

    def _fetch_local_enrichment(self, gnd_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch description/synonyms from local DB for GND IDs that exist there.

        Returns:
            {gnd_id: {description: str, synonyms: list, title: str}}
        """
        try:
            raw = self.caching_registry.execute("get_gnd_batch", {"gnd_ids": gnd_ids})
            data = json.loads(raw)
        except Exception as e:
            logger.warning(f"SearchAgent: get_gnd_batch failed: {e}")
            return {}

        enrichment = {}
        for gnd_id, entry in data.get("entries", {}).items():
            enrichment[gnd_id] = {
                "description": entry.get("description", ""),
                "synonyms": entry.get("synonyms", []),
                "title": entry.get("title", ""),
            }
        return enrichment

    def _apply_enrichment(
        self,
        pool: Dict[str, Dict[str, Any]],
        enrichment: Dict[str, Dict[str, Any]],
    ) -> None:
        """Apply local DB enrichment (description, synonyms) to pool entries."""
        for entry in pool.values():
            for gnd_id in entry.get("gnd_ids", []):
                if gnd_id in enrichment:
                    rich = enrichment[gnd_id]
                    if rich.get("description") and not entry.get("description"):
                        entry["description"] = rich["description"]
                    if rich.get("synonyms") and not entry.get("synonyms"):
                        entry["synonyms"] = rich["synonyms"]
                    break  # first match is enough
