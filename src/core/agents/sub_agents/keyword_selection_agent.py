"""Keyword Selection SubAgent - Claude Generated

Selects relevant GND keywords from the search pool using a chunked
quick-choice approach — mirroring the normal pipeline strategy.

Strategy:
  1. Sort context.gnd_entries by catalog frequency (count) descending
  2. Split into chunks of chunk_size entries
  3. For each chunk: ask LLM to pick relevant entries (compact flat-list format)
  4. Parse <final_list> tags from each response
  5. Deduplicate and store as context.selected_keywords
  6. Identify missing_concepts: extracted_keywords not covered by any result

No AgentLoop / tool-calling needed — this step is fully deterministic in
structure (same logic regardless of content), same as SearchAgent.
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
    matching the behaviour of the normal pipeline's keyword selection step.
    No tool-calling — deterministic chunked LLM calls.
    """

    @property
    def agent_name(self) -> str:
        return "Keyword Selection Agent"

    @property
    def agent_id(self) -> str:
        return "selection"

    # ── Required abstract stubs (not used — execute() is overridden) ──

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_available_tools(self) -> List[str]:
        # Kept for compatibility; not used in chunked execution
        return ["get_gnd_entry", "get_gnd_batch", "get_search_cache"]

    def build_user_prompt(self) -> str:
        return ""

    def parse_result(self, llm_output: str) -> Dict[str, Any]:
        return {}

    # ── Core: chunked deterministic execution ──

    def execute(self) -> SubAgentResult:
        """Process GND pool in sorted chunks, pick relevant keywords per chunk."""
        if not self.context.gnd_entries:
            logger.warning("KeywordSelectionAgent: no GND entries in context")
            return SubAgentResult(
                success=True,
                data={"selected_keywords": [], "missing_concepts": [], "reasoning": "No GND pool available"},
            )

        abstract = self.context.abstract
        if not abstract:
            logger.warning("KeywordSelectionAgent: no abstract in context")
            return SubAgentResult(success=False, data={}, error="No abstract in context")

        if self.stream_callback:
            self.stream_callback(
                f"\n{'='*50}\n🤖 {self.agent_name}\n{'='*50}\n"
                f"📚 {len(self.context.gnd_entries)} GND-Einträge → Chunk-Selektion\n"
            )

        # Sort by catalog frequency (count) descending — quick-choice: most relevant first
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
            self.stream_callback(f"✅ Gesamt: {len(selected_list)} GND-Schlagworte ausgewählt\n")

        # Identify missing_concepts: extracted_keywords not represented in selected
        missing = self._find_missing_concepts(
            extracted=self.context.extracted_keywords,
            selected_titles={v["title"].lower() for v in selected.values()},
        )
        if missing and self.stream_callback:
            self.stream_callback(f"❓ {len(missing)} fehlende Konzepte: {', '.join(missing[:5])}"
                                 f"{'...' if len(missing) > 5 else ''}\n")

        # Update shared context
        self.context.selected_keywords = selected_list
        self.context.missing_concepts = missing

        result_data = {
            "selected_keywords": selected_list,
            "missing_concepts": missing,
            "reasoning": f"Chunked selection: {total_chunks} Chunks, {len(selected_list)} Schlagworte",
        }
        self.context.set_step_result(self.agent_id, result_data, quality=1.0)

        return SubAgentResult(
            success=True,
            data=result_data,
            quality_score=1.0,
            iterations=total_chunks,
            tool_calls=llm_calls,
        )

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
            "**Kriterien für Relevanz:**\n"
            "- **Direkter Bezug**: Das Schlagwort muss **explizit** im Abstract erwähnt oder **thematisch eng verknüpft** sein.\n"
            "- **Loser Zusammenhang**: Oberbegriffe oder verwandte Themen sind **nur dann relevant**, wenn sie **unverzichtbar** für das Verständnis des Abstracts sind.\n"
            "- **Keine Allgemeinplätze**: Schlagworte wie \"Wissenschaft\", \"Technologie\" oder \"Gesellschaft\" sind **nur relevant**, wenn sie **spezifisch** durch den Abstract begründet werden.\n\n"
            f"Zur Auswahl stehende GND-Schlagworte (Chunk {chunk_idx}/{total_chunks}):\n{kw_lines}\n\n"
            "Bitte gib deine Antwort in folgendem Format (nur relevante Schlagworte, kommasepariert):\n"
            "<final_list>Schlagwort (GND-ID: X), Schlagwort (GND-ID: Y)</final_list>"
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
                tools=[],  # No tools — force plain text response
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
                f"  🔄 Chunk {chunk_idx}/{total_chunks}: "
                f"{len(chunk)} Schlagworte → {len(chunk_selected)} ausgewählt\n"
            )

        return chunk_selected

    def _parse_final_list(
        self,
        llm_output: str,
        chunk: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Parse <final_list>...</final_list> from LLM output.

        Builds a title→gnd_id lookup from the chunk so we can resolve IDs.
        Returns list of {gnd_id, title} for each matched entry.
        """
        # Build lookup: title_lower → {gnd_id, title}
        title_lookup: Dict[str, Dict[str, str]] = {}
        for entry in chunk:
            title = entry.get("title", "")
            if title:
                title_lookup[title.lower()] = {
                    "gnd_id": entry.get("gnd_id", ""),
                    "title": title,
                }

        # Extract <final_list> content
        match = re.search(r"<final_list>(.*?)</final_list>", llm_output, re.DOTALL)
        if not match:
            logger.debug("KeywordSelectionAgent: no <final_list> tag found in chunk response")
            return []

        raw_list = match.group(1).strip()
        results: List[Dict[str, str]] = []
        seen: Set[str] = set()

        for item in raw_list.split(","):
            item = item.strip()
            if not item:
                continue

            # Try to parse "Title (GND-ID: XXXX)" format first
            id_match = re.search(r"\(GND-ID:\s*([^\)]+)\)", item)
            if id_match:
                gnd_id = id_match.group(1).strip()
                title_part = item[:item.rfind("(")].strip()
            else:
                # Fallback: try "Title (XXXX)" or just "Title"
                paren_match = re.search(r"\(([^\)]+)\)", item)
                if paren_match:
                    gnd_id = paren_match.group(1).strip()
                    title_part = item[:item.rfind("(")].strip()
                else:
                    gnd_id = ""
                    title_part = item

            # Match against chunk titles
            title_lower = title_part.lower()
            if title_lower in title_lookup and title_lower not in seen:
                seen.add(title_lower)
                entry = dict(title_lookup[title_lower])
                if gnd_id and not entry.get("gnd_id"):
                    entry["gnd_id"] = gnd_id
                results.append(entry)
            elif gnd_id:
                # Try matching by GND-ID across chunk
                for chunk_entry in chunk:
                    if str(chunk_entry.get("gnd_id", "")) == gnd_id:
                        key = chunk_entry["title"].lower()
                        if key not in seen:
                            seen.add(key)
                            results.append({
                                "gnd_id": chunk_entry["gnd_id"],
                                "title": chunk_entry["title"],
                            })
                        break

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
        """Not used — execute() updates context directly."""
        pass
