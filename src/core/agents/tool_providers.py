"""Tool Providers for ALIMA Agents - Claude Generated

Reusable data-collection classes that wrap MCP tools with correct
response parsing, fallback logic, and uniform output formats.

Each Provider has a collect() method that returns a typed result
(dataclass) regardless of which data source actually delivered data.
This decouples agents from the specifics of MCP tool response formats.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================
# DK Data Provider
# ============================================================

@dataclass
class DKDataResult:
    """Uniform result from DK classification data collection.

    Attributes:
        dk_entries: Normalised DK entries from all sources.
            Each entry: {dk: "540", title: "...", count: N, source: "cache|catalog|gnd", keyword: "..."}
        ddc_from_gnd: GND entries that carry DDC/DK codes (for prompt context).
        tool_calls: Number of MCP tool calls made.
        has_data: True if at least one source delivered data.
    """
    dk_entries: List[Dict] = field(default_factory=list)
    ddc_from_gnd: List[Dict] = field(default_factory=list)
    tool_calls: int = 0
    has_data: bool = False

    def format_for_prompt(self, max_entries: int = 30) -> str:
        """Format collected DK data for inclusion in an LLM prompt.

        Returns:
            Markdown-formatted string listing DK codes with titles and counts.
        """
        if not self.dk_entries and not self.ddc_from_gnd:
            return ""

        parts: List[str] = []

        # DK/DDC entries from cache and catalog
        if self.dk_entries:
            seen_dk: Dict[str, Dict] = {}
            for entry in self.dk_entries:
                dk_code = entry.get("dk", "")
                if not dk_code:
                    continue
                if dk_code not in seen_dk or entry.get("count", 0) > seen_dk[dk_code].get("count", 0):
                    seen_dk[dk_code] = entry
            sorted_dk = sorted(seen_dk.values(), key=lambda e: e.get("count", 0), reverse=True)
            parts.append(f"\n**DK/DDC-Klassifikationen aus Bibliotheksbestand ({len(sorted_dk)} gefunden):**\n")
            for entry in sorted_dk[:max_entries]:
                dk_code = entry.get("dk", "")
                title = entry.get("title", "")
                count = entry.get("count", 0)
                source = entry.get("source", "")
                keyword = entry.get("keyword", "")
                line = f"- **{dk_code}**"
                if title:
                    line += f": {title}"
                if count:
                    line += f" (Häufigkeit: {count})"
                if keyword and keyword != title:
                    line += f" [via {keyword}]"
                parts.append(line + "\n")

        # DDC codes from GND entries (always available after search step)
        if self.ddc_from_gnd:
            parts.append(f"\n**DDC-Codes aus GND-Einträgen ({len(self.ddc_from_gnd)} Einträge mit DDC):**\n")
            for entry in self.ddc_from_gnd[:max_entries]:
                title = entry.get("title", "")
                ddc_codes = entry.get("ddc_codes", [])
                dk_codes = entry.get("dk_codes", [])
                count = entry.get("count", 0)
                if ddc_codes or dk_codes:
                    codes_str = ", ".join(ddc_codes + dk_codes)
                    parts.append(f"- {title}: {codes_str} (Häufigkeit: {count})\n")

        return "".join(parts)


class DKDataProvider:
    """Collects DK classification data from all available sources.

    Data sources (in priority order):
    1. get_dk_cache: Local DB cache of catalog DK results
    2. search_catalog: Live catalog search via BiblioSuggester
    3. GND entries with DDC codes: Always available after search step

    Each source is parsed correctly according to its actual MCP response
    format, and results are normalised into a uniform {dk, title, count, source}
    format.
    """

    def __init__(self, caching_registry, context):
        """
        Args:
            caching_registry: CachingToolRegistry for MCP tool access
            context: SharedContext with pipeline state
        """
        self.caching_registry = caching_registry
        self.context = context

    def collect(self, max_keywords: int = 30) -> DKDataResult:
        """Collect DK classification data from all sources.

        Args:
            max_keywords: Maximum number of keywords to query (prevents token overflow)

        Returns:
            DKDataResult with normalised entries from all sources
        """
        dk_entries: List[Dict] = []
        tool_calls = 0

        # Source 1: DK cache lookups for selected keywords
        cache_entries, tc1 = self._collect_from_cache(max_keywords)
        dk_entries.extend(cache_entries)
        tool_calls += tc1

        # Source 2: Catalog search for extracted keywords
        catalog_entries, tc2 = self._collect_from_catalog()
        dk_entries.extend(catalog_entries)
        tool_calls += tc2

        # Source 3: DDC/DK codes from GND entries (always available)
        ddc_from_gnd = self._collect_from_gnd_entries()

        has_data = bool(dk_entries) or bool(ddc_from_gnd)
        logger.info(
            f"DKDataProvider: collected {len(dk_entries)} DK entries "
            f"({len(cache_entries)} from cache, {len(catalog_entries)} from catalog), "
            f"{len(ddc_from_gnd)} GND entries with DDC, "
            f"tool_calls={tool_calls}"
        )

        return DKDataResult(
            dk_entries=dk_entries,
            ddc_from_gnd=ddc_from_gnd,
            tool_calls=tool_calls,
            has_data=has_data,
        )

    def _collect_from_cache(self, max_keywords: int) -> tuple:
        """Collect DK data from get_dk_cache for each selected keyword.

        get_dk_cache returns:
            {"cached": true, "term": "...", "titles": [
                {"title": "...", "classifications": ["DK 540", "RVK VK 8000"], ...}
            ]}

        Returns:
            (list of normalised DK entries, tool_call_count)
        """
        entries: List[Dict] = []
        tool_calls = 0
        seen_terms: Set[str] = set()

        for kw in self.context.selected_keywords[:max_keywords]:
            term = kw.get("title", "")
            if not term or term.lower() in seen_terms:
                continue
            seen_terms.add(term.lower())

            try:
                raw = self.caching_registry.execute("get_dk_cache", {"term": term})
                data = json.loads(raw) if isinstance(raw, str) else raw
                tool_calls += 1

                if not isinstance(data, dict) or not data.get("cached"):
                    continue

                titles = data.get("titles", [])
                if not isinstance(titles, list):
                    continue

                for title_entry in titles:
                    if not isinstance(title_entry, dict):
                        continue
                    classifications = title_entry.get("classifications", [])
                    if not isinstance(classifications, list):
                        continue
                    for cls in classifications:
                        cls_str = str(cls)
                        dk_match = re.match(r'(?:DK\s+)?(\d[\d.]+)', cls_str)
                        rvk_match = re.match(r'RVK\s+(.+)', cls_str)
                        if dk_match:
                            entries.append({
                                "keyword": term,
                                "dk": dk_match.group(1),
                                "title": title_entry.get("title", ""),
                                "count": title_entry.get("count", 0),
                                "source": "cache",
                                "classification_type": "DK",
                            })
                        elif rvk_match:
                            entries.append({
                                "keyword": term,
                                "dk": rvk_match.group(1),
                                "title": title_entry.get("title", ""),
                                "count": title_entry.get("count", 0),
                                "source": "cache",
                                "classification_type": "RVK",
                            })
            except Exception as e:
                logger.debug(f"get_dk_cache('{term}') failed: {e}")

        return entries, tool_calls

    def _collect_from_catalog(self) -> tuple:
        """Collect DK data from search_catalog for extracted keywords.

        search_catalog returns:
            {"source": "catalog", "results": {
                "term1": {"kw1": {"count": N, "gndid": [...], "ddc": [...], "dk": [...]}}
            }}

        Returns:
            (list of normalised DK entries, tool_call_count)
        """
        entries: List[Dict] = []
        tool_calls = 0

        if not self.context.extracted_keywords:
            return entries, tool_calls

        try:
            raw = self.caching_registry.execute("search_catalog", {
                "terms": self.context.extracted_keywords[:10],
            })
            data = json.loads(raw) if isinstance(raw, str) else raw
            tool_calls += 1

            if not isinstance(data, dict):
                return entries, tool_calls

            results = data.get("results", {})
            for term, term_results in results.items():
                if not isinstance(term_results, dict):
                    continue
                for kw_title, kw_data in term_results.items():
                    if not isinstance(kw_data, dict):
                        continue
                    dk_codes = kw_data.get("dk", [])
                    ddc_codes = kw_data.get("ddc", [])
                    count = kw_data.get("count", 0)
                    for dk in dk_codes:
                        if dk:
                            entries.append({
                                "keyword": kw_title,
                                "dk": str(dk),
                                "title": kw_title,
                                "count": count,
                                "source": "catalog",
                                "classification_type": "DK",
                            })
                    for ddc in ddc_codes:
                        if ddc:
                            entries.append({
                                "keyword": kw_title,
                                "dk": str(ddc),
                                "title": kw_title,
                                "count": count,
                                "source": "catalog",
                                "classification_type": "DDC",
                            })
        except Exception as e:
            logger.debug(f"search_catalog failed: {e}")

        return entries, tool_calls

    def _collect_from_gnd_entries(self) -> List[Dict]:
        """Collect DDC/DK codes from GND entries already in context.

        This is always available after the search step and provides a
        reliable fallback when cache and catalog are empty.

        Returns:
            List of GND entry dicts that have ddc_codes or dk_codes.
        """
        ddc_entries = [
            e for e in self.context.gnd_entries
            if e.get("ddc_codes") or e.get("dk_codes")
        ]
        ddc_entries.sort(key=lambda e: e.get("count", 0), reverse=True)
        return ddc_entries[:50]