"""Keyword Selection SubAgent - Claude Generated

Selects relevant GND keywords from the search pool using a chunked
quick-choice approach, then builds Schlagwortketten for specificity.

Strategy:
  1. Sort context.gnd_entries by catalog frequency (count) descending
  2. Split into chunks of chunk_size entries
  3. For each chunk: ask LLM to pick relevant entries (compact flat-list format)
  4. Parse <final_list> tags from each response
  5. After all chunks: request Schlagwortketten from selected keywords
  6. Deduplicate and store as context.selected_keywords + context.keyword_chains
  7. Identify missing_concepts: extracted_keywords not covered by any result

No AgentLoop / tool-calling needed for chunk selection (deterministic structure).
Schlagwortketten use one additional LLM call after all chunks complete.
"""

import json
import logging
import re
from typing import Dict, List, Any, Set

from src.core.agents.sub_agents.base_sub_agent import BaseSubAgent, SubAgentResult

logger = logging.getLogger(__name__)

# Chunk size matches normal pipeline (see junk_cli.txt / junk_guiB.txt)
CHUNK_SIZE = 350

SYSTEM_PROMPT = """\
**Deine Rolle als Bibliothekar:**
Du bist ein **präziser und selektiver GND-Schlagwort-Experte** mit folgenden Kernregeln:
1. **Strenge Relevanzprüfung**: Wähle *nur* Schlagworte aus, die **direkt** zum Abstract passen oder **losen thematischen Bezug** haben.
2. **Ignorieren von Nicht-Relevanten**: Alle Schlagworte, die **keinen** Bezug zum Text haben, werden **ausgeschlossen** – selbst wenn sie allgemein scheinen.
3. **Keine Ergänzungen**: Nutze **nur** die vorgegebenen GND-Schlagworte (keine Synonyme, Oberbegriffe oder kreative Ergänzungen).
4. **Effizienz**: Da die Liste in mehreren Anfragen abgearbeitet wird, konzentriere dich **ausschließlich** auf die aktuelle Teilmenge.

**Werkzeuge & Limits**:
- Nutze **keine** externen Quellen oder Tools – arbeite nur mit den gegebenen Daten.

**Beispiel für deine Denkweise**:
- *Abstract* enthält "KI in der Medizin" → Relevant: "Künstliche Intelligenz", "Medizin", "Diagnostik".
- *Nicht relevant*: "Bibliothekswesen", "Juristische Grundlagen", selbst wenn sie in der GND-Liste stehen.

Die gesamte GND-Liste wurde geteilt und wird in mehreren Anfragen abgearbeitet. \
Suche nur die relevanten Schlagworte heraus und ***ignoriere*** alle nicht relevanten."""


class KeywordSelectionAgent(BaseSubAgent):
    """Select relevant GND keywords from pool using chunked quick-choice.

    Processes the full GND pool in sorted chunks (by catalog frequency),
    then builds Schlagwortketten for specificity.
    No tool-calling for chunk selection — deterministic chunked LLM calls.
    """

    @property
    def agent_name(self) -> str:
        return "Keyword Selection Agent"

    @property
    def agent_id(self) -> str:
        return "selection"

    # -- Required abstract stubs (not used -- execute() is overridden) --

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_available_tools(self) -> List[str]:
        # Kept for compatibility; not used in chunked execution
        return ["get_gnd_entry", "get_gnd_batch", "get_search_cache"]

    def build_user_prompt(self) -> str:
        return ""

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        return {}

    # -- Core: chunked deterministic execution --

    def execute(self) -> SubAgentResult:
        """Process GND pool in sorted chunks, pick relevant keywords per chunk,
        then build Schlagwortketten from the selection."""
        if not self.context.gnd_entries:
            logger.warning("KeywordSelectionAgent: no GND entries in context")
            return SubAgentResult(
                success=True,
                data={"selected_keywords": [], "keyword_chains": [], "missing_concepts": [], "reasoning": "No GND pool available"},
            )

        abstract = self.context.abstract
        if not abstract:
            logger.warning("KeywordSelectionAgent: no abstract in context")
            return SubAgentResult(success=False, data={}, error="No abstract in context")

        if self.stream_callback:
            self.stream_callback(
                f"\n{'='*50}\n\U0001f916 {self.agent_name}\n{'='*50}\n"
                f"\U0001f4da {len(self.context.gnd_entries)} GND-Eintr\u00e4ge \u2192 Chunk-Selektion\n"
            )

        # Sort by catalog frequency (count) descending -- quick-choice: most relevant first
        sorted_entries = sorted(
            self.context.gnd_entries,
            key=lambda e: e.get("count", 0),
            reverse=True,
        )

        # Split into chunks
        chunks = [
            sorted_entries[i:i + CHUNK_SIZE]
            for i in range(0, len(sorted_entries), CHUNK_SIZE)
        ]
        total_chunks = len(chunks)

        # Accumulated results across all chunks: {title_lower: {gnd_id, title}}
        selected: Dict[str, Dict[str, str]] = {}
        llm_calls = 0

        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_result = self._process_chunk(
                chunk=chunk,
                abstract=abstract,
                chunk_idx=chunk_idx,
                total_chunks=total_chunks,
            )
            llm_calls += 1

            for entry in chunk_result:
                key = entry["title"].lower()
                if key not in selected:
                    selected[key] = entry

        selected_list = list(selected.values())

        if self.stream_callback:
            self.stream_callback(f"\u2705 Gesamt: {len(selected_list)} GND-Schlagworte ausgew\u00e4hlt\n")

        # Build Schlagwortketten from selected keywords
        keyword_chains: List[Dict] = []
        if selected_list:
            keyword_chains = self._build_keyword_chains(
                selected_keywords=selected_list,
                abstract=abstract,
            )
            if keyword_chains and self.stream_callback:
                self.stream_callback(
                    f"\U0001f517 {len(keyword_chains)} Schlagwortkette{'n' if len(keyword_chains) != 1 else ''} erstellt\n"
                )

        # Identify missing_concepts: extracted keywords not represented in selected
        missing = self._find_missing_concepts(
            extracted=self.context.extracted_keywords,
            selected_titles={v["title"].lower() for v in selected.values()},
        )
        if missing and self.stream_callback:
            self.stream_callback(f"\u2753 {len(missing)} fehlende Konzepte: {', '.join(missing[:5])}"
                                 f"{'...' if len(missing) > 5 else ''}\n")

        # Update shared context
        self.context.selected_keywords = selected_list
        self.context.keyword_chains = keyword_chains
        self.context.missing_concepts = missing

        result_data = {
            "selected_keywords": selected_list,
            "keyword_chains": keyword_chains,
            "missing_concepts": missing,
            "reasoning": f"Chunked selection: {total_chunks} Chunks, {len(selected_list)} Schlagworte, {len(keyword_chains)} Ketten",
        }
        self.context.set_step_result(self.agent_id, result_data, quality=1.0)

        return SubAgentResult(
            success=True,
            data=result_data,
            quality_score=1.0,
            iterations=total_chunks,
            tool_calls=llm_calls,
        )

    def _build_keyword_chains(
        self,
        selected_keywords: List[Dict[str, str]],
        abstract: str,
    ) -> List[Dict]:
        """Ask LLM to compose Schlagwortketten from selected keywords.

        Schlagwortketten are compound subject headings combining multiple
        GND terms into chains for specificity, matching the normal pipeline's
        ``extract_keyword_chains_from_response()`` format.

        Returns:
            List of {"chain": [...], "reason": "..."} dicts.
        """
        if not selected_keywords or len(selected_keywords) < 2:
            return []

        kw_lines = "\n".join(
            f"- {kw.get('title', '')} (GND-ID: {kw.get('gnd_id', '')})"
            for kw in selected_keywords[:60]  # Limit to avoid token overflow
        )

        chain_prompt = (
            f"Abstract:\n{abstract[:2000]}\n\n"
            f"Ausgew\u00e4hlte GND-Schlagworte:\n{kw_lines}\n\n"
            "Bilde **Schlagwortketten** (Verkn\u00fcpfungen verwandter Begriffe f\u00fcr Spezifit\u00e4t).\n"
            "Regeln:\n"
            "- Jede Kette verbindet 2-5 verwandte Begriffe mit \u2192\n"
            "- Begriffe stammen **nur** aus der obigen Liste\n"
            "- Jede Kette hat eine kurze Begr\u00fcndung\n\n"
            "Ausgabeformat:\n"
            "<schlagwortketten>\n"
            "Begriff1 \u2192 Begriff2 \u2192 Begriff3 (Begr\u00fcndung)\n"
            "Begriff4 \u2192 Begriff5 (Begr\u00fcndung)\n"
            "</schlagwortketten>\n"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chain_prompt},
        ]

        if self.context.verbose:
            self._log_prompt_verbose(SYSTEM_PROMPT, chain_prompt, label="Schlagwortketten")

        try:
            tokens: List[str] = []

            def _collect(token: str) -> None:
                tokens.append(token)

            response = self.llm_service.generate_with_tools(
                provider=self.context.provider,
                model=self.context.model,
                messages=messages,
                tools=[],  # No tools -- plain text response
                temperature=self.context.temperature,
                max_tokens=min(self.context.max_tokens, 1024),
                stream_callback=_collect,
            )

            raw = response.content if response else "".join(tokens)

        except Exception as e:
            logger.warning(f"KeywordSelectionAgent: Schlagwortketten LLM call failed: {e}")
            return []

        return self._parse_schlagwortketten(raw, selected_keywords)

    def _parse_schlagwortketten(
        self,
        llm_output: str,
        selected_keywords: List[Dict[str, str]],
    ) -> List[Dict]:
        """Parse <schlagwortketten> tags from LLM output.

        Each line inside the tag has the format:
            Begriff1 -> Begriff2 -> Begriff3 (Begrundung)

        Returns:
            List of {"chain": [...], "reason": "..."} dicts matching
            the format used by extract_keyword_chains_from_response().
        """
        chains: List[Dict] = []

        # Try to find <schlagwortketten> or <keyword_chains> tag
        match = re.search(
            r'<(?:schlagwortketten|keyword_chains)>(.*?)</(?:schlagwortketten|keyword_chains)>',
            llm_output, re.DOTALL | re.IGNORECASE
        )
        if match:
            content = match.group(1).strip()
            for line in content.split('\n'):
                line = line.strip()
                if not line or '→' not in line and '->' not in line:
                    continue
                # Normalize arrow
                line = line.replace('->', '→')
                # Split chain and reason
                reason = ""
                if '(' in line and ')' in line:
                    paren_start = line.rfind('(')
                    paren_end = line.rfind(')')
                    if paren_end > paren_start:
                        reason = line[paren_start + 1:paren_end].strip()
                        line = line[:paren_start].strip()
                # Split by arrow
                parts = [p.strip() for p in line.split('→') if p.strip()]
                if len(parts) >= 2:
                    chains.append({"chain": parts, "reason": reason})

        # Fallback: try JSON format
        if not chains:
            try:
                from src.core.json_response_parser import parse_json_response
                data = parse_json_response(llm_output)
                if data and "keyword_chains" in data:
                    for c in data["keyword_chains"]:
                        if isinstance(c, dict) and "chain" in c:
                            chains.append(c)
            except Exception:
                pass

        return chains

    def _process_chunk(
        self,
        chunk: List[Dict[str, Any]],
        abstract: str,
        chunk_idx: int,
        total_chunks: int,
    ) -> List[Dict[str, str]]:
        """Send one chunk to the LLM and parse the <final_list> response.

        Returns:
            List of {gnd_id, title} dicts for selected entries in this chunk.
        """
        # Build compact keyword list (matching normal pipeline format)
        kw_lines = "\n".join(
            f"{e['title']} (GND-ID: {e.get('gnd_id', '')})"
            for e in chunk
            if e.get("title")
        )

        user_prompt = (
            f"Abstract:\n{abstract[:3000]}\n\n"
            "**Kriterien f\u00fcr Relevanz:**\n"
            "- **Direkter Bezug**: Das Schlagwort muss **explizit** im Abstract erw\u00e4hnt oder **thematisch eng verkn\u00fcpft** sein.\n"
            "- **Loser Zusammenhang**: Oberbegriffe oder verwandte Themen sind **nur dann relevant**, wenn sie **unverzichtbar** f\u00fcr das Verst\u00e4ndnis des Abstracts sind.\n"
            "- **Keine Allgemeinpl\u00e4tze**: Schlagworte wie \"Wissenschaft\", \"Technologie\" oder \"Gesellschaft\" sind **nur relevant**, wenn sie **spezifisch** durch den Abstract begr\u00fcndet werden.\n\n"
            f"Zur Auswahl stehende GND-Schlagworte (Chunk {chunk_idx}/{total_chunks}):\n{kw_lines}\n\n"
            "W\u00e4hle die relevanten Schlagworte aus und gib sie in EINES dieser Formate:\n"
            "1. XML-Tags: <final_list>Schlagwort (GND-ID: X), Schlagwort (GND-ID: Y)</final_list>\n"
            "2. Komma-separiert: Schlagwort (GND-ID: X), Schlagwort (GND-ID: Y)\n"
            "3. Als Liste:\n- Schlagwort (GND-ID: X)\n- Schlagwort (GND-ID: Y)\n\n"
            "WICHTIG: Verwende **nur** Schlagworte aus der obigen Liste und gib die GND-ID mit an."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Verbose: log full prompt for this chunk
        if self.context.verbose:
            self._log_prompt_verbose(SYSTEM_PROMPT, user_prompt, label=f"Chunk {chunk_idx}/{total_chunks}")

        chunk_selected: List[Dict[str, str]] = []

        try:
            # Collect streamed tokens
            tokens: List[str] = []

            def _collect(token: str) -> None:
                tokens.append(token)

            response = self.llm_service.generate_with_tools(
                provider=self.context.provider,
                model=self.context.model,
                messages=messages,
                tools=[],  # No tools -- force plain text response
                temperature=self.context.temperature,
                max_tokens=self.context.max_tokens,
                stream_callback=_collect,
            )

            raw = response.content if response else "".join(tokens)
            chunk_selected = self._parse_final_list(raw, chunk)

        except Exception as e:
            logger.warning(f"KeywordSelectionAgent: chunk {chunk_idx} LLM call failed: {e}")

        if self.stream_callback:
            self.stream_callback(
                f"  \U0001f504 Chunk {chunk_idx}/{total_chunks}: "
                f"{len(chunk)} Schlagworte \u2192 {len(chunk_selected)} ausgew\u00e4hlt\n"
            )

        return chunk_selected

    def _parse_final_list(
        self,
        llm_output: str,
        chunk: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Parse selected keywords from LLM output with robust fallbacks.

        Strategy (in order):
        1. <final_list> tag extraction (case-insensitive)
        2. GND-ID pattern extraction across entire response
        3. Substring match of chunk titles in response text

        Returns list of {gnd_id, title} for each matched entry.
        """
        # Build lookup: title_lower -> {gnd_id, title}
        title_lookup: Dict[str, Dict[str, str]] = {}
        gnd_id_lookup: Dict[str, Dict[str, str]] = {}
        for entry in chunk:
            title = entry.get("title", "")
            gnd_id = entry.get("gnd_id", "")
            if title:
                title_lookup[title.lower()] = {
                    "gnd_id": gnd_id,
                    "title": title,
                }
            if gnd_id:
                gnd_id_lookup[gnd_id] = {
                    "gnd_id": gnd_id,
                    "title": title,
                }

        results: List[Dict[str, str]] = []
        seen: Set[str] = set()

        def _add(title_key: str, gnd_id: str, title: str) -> None:
            key = title_key.lower()
            if key not in seen:
                seen.add(key)
                results.append({"gnd_id": gnd_id, "title": title})

        # -- Strategy 1: <final_list> tag extraction --
        match = re.search(r"<final_list>(.*?)</final_list>", llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            raw_list = match.group(1).strip()
            # Support comma, pipe, and semicolon separators
            items = re.split(r'[,\|;]\s*', raw_list)
            for item in items:
                item = item.strip()
                if not item:
                    continue
                # Remove numbering like "1.", "2)", etc.
                item = re.sub(r'^\d+[\.\)]\s*', '', item).strip()
                # Try "Title (GND-ID: XXXX)" format
                id_match = re.search(r'\(GND-ID:\s*([^\)]+)\)', item, re.IGNORECASE)
                if id_match:
                    gnd_id = id_match.group(1).strip()
                    title_part = item[:item.rfind("(")].strip()
                else:
                    # Try generic parenthetical "Title (XXXX-XX)"
                    paren_match = re.search(r'\(([0-9X-]+[0-9X])\)', item)
                    if paren_match:
                        gnd_id = paren_match.group(1).strip()
                        title_part = item[:item.rfind("(")].strip()
                    else:
                        gnd_id = ""
                        title_part = item

                # Match against chunk titles
                title_lower = title_part.lower().strip()
                if title_lower in title_lookup:
                    entry = title_lookup[title_lower]
                    _add(title_lower, gnd_id or entry["gnd_id"], entry["title"])
                elif gnd_id and gnd_id in gnd_id_lookup:
                    entry = gnd_id_lookup[gnd_id]
                    _add(entry["title"].lower(), entry["gnd_id"], entry["title"])
                else:
                    # Substring match: check if any chunk title is contained
                    for lookup_title, entry in title_lookup.items():
                        if lookup_title in title_lower or title_lower in lookup_title:
                            _add(lookup_title, gnd_id or entry["gnd_id"], entry["title"])
                            break

        # -- Strategy 2: JSON extraction --
        if not results:
            try:
                from src.core.json_response_parser import parse_json_response
                json_data = parse_json_response(llm_output)
                if json_data and isinstance(json_data, dict):
                    # Try "selected" or "keywords" or "selected_keywords" keys
                    for key in ("selected", "keywords", "selected_keywords"):
                        items = json_data.get(key, [])
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    title = item.get("title", "")
                                    gnd_id = item.get("gnd_id", "")
                                    if title:
                                        title_lower = title.lower().strip()
                                        if title_lower in title_lookup:
                                            entry = title_lookup[title_lower]
                                            _add(title_lower, gnd_id or entry["gnd_id"], entry["title"])
                                        elif gnd_id and gnd_id in gnd_id_lookup:
                                            entry = gnd_id_lookup[gnd_id]
                                            _add(entry["title"].lower(), entry["gnd_id"], entry["title"])
                                elif isinstance(item, str):
                                    title_lower = item.lower().strip()
                                    if title_lower in title_lookup:
                                        entry = title_lookup[title_lower]
                                        _add(title_lower, entry["gnd_id"], entry["title"])
                elif json_data and isinstance(json_data, list):
                    for item in json_data:
                        if isinstance(item, str):
                            title_lower = item.lower().strip()
                            if title_lower in title_lookup:
                                entry = title_lookup[title_lower]
                                _add(title_lower, entry["gnd_id"], entry["title"])
            except Exception as e:
                logger.debug(f"KeywordSelectionAgent: JSON parsing fallback failed: {e}")

        # -- Strategy 3: GND-ID pattern extraction across entire response --
        if not results:
            gnd_pattern = re.findall(r'\((\d{4,}-\d{1,2})\)', llm_output)
            for gnd_id in gnd_pattern:
                if gnd_id in gnd_id_lookup:
                    entry = gnd_id_lookup[gnd_id]
                    _add(entry["title"].lower(), entry["gnd_id"], entry["title"])

        # -- Strategy 4: Substring match of chunk titles in response --
        if not results:
            response_lower = llm_output.lower()
            for lookup_title, entry in title_lookup.items():
                # Check if the title appears in the response
                if lookup_title in response_lower:
                    _add(lookup_title, entry["gnd_id"], entry["title"])

        if results:
            logger.debug(f"KeywordSelectionAgent: parsed {len(results)} entries from chunk response")
        else:
            logger.debug("KeywordSelectionAgent: no entries parsed from chunk response")

        return results

    def _find_missing_concepts(
        self,
        extracted: List[str],
        selected_titles: Set[str],
    ) -> List[str]:
        """Find extracted keywords not covered by any selected GND entry title."""
        missing = []
        for kw in extracted:
            kw_lower = kw.lower()
            # Check if any selected title contains this keyword (loose match)
            if not any(kw_lower in title for title in selected_titles):
                missing.append(kw)
        return missing

    def _update_shared_context(self, result_data: Dict[str, Any]) -> None:
        """Not used -- execute() updates context directly."""
        pass