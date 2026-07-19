"""ALIMA-specific chat tools — Pipeline-output inspection.

WP10 P-δ.2. Claude Generated.

All tools read from ``session.last_shared_context`` (a
``src.core.agents.shared_context.SharedContext``). They never mutate state
— that is P-ε territory.

`ValidateGndTermTool` optionally falls back to the MCP `search_gnd` tool
when the term is not in the session pool. The MCP registry is injected at
construction time by ``build_chat_toolset``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.ui.chat_tools.base import BaseChatTool

logger = logging.getLogger(__name__)


def _ctx(session: Any):
    return getattr(session, "last_shared_context", None)


# ----- Read-only keyword/chain/classification accessors ----------------------


class GetKeywordsTool(BaseChatTool):
    name = "get_keywords"
    description = (
        "Fetch keywords from the current pipeline run. "
        "`kind` picks the source: 'initial' (user input), "
        "'extracted' (LLM extraction result), 'selected' (curated, "
        "GND-resolved), 'gnd' (GND search-result titles from Phase 2), "
        "'final' (curated final list from extra.final_keywords)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["initial", "extracted", "selected", "gnd", "final"],
            },
        },
        "required": ["kind"],
    }

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        if ctx is None:
            return False
        return bool(
            ctx.initial_keywords
            or ctx.extracted_keywords
            or ctx.selected_keywords
            or ctx.gnd_entries
            or (ctx.extra or {}).get("final_keywords")
        )

    def execute(self, session: Any, kind: str = "", **_: Any) -> str:
        ctx = _ctx(session)
        if ctx is None:
            return json.dumps({"error": "No SharedContext bound to session"})
        if kind == "initial":
            payload = ctx.initial_keywords
        elif kind == "extracted":
            payload = ctx.extracted_keywords
        elif kind == "selected":
            payload = ctx.selected_keywords
        elif kind == "gnd":
            payload = [e.get("title", "") for e in ctx.gnd_entries if e.get("title")]
        elif kind == "final":
            payload = (ctx.extra or {}).get("final_keywords") or []
        else:
            return json.dumps(
                {
                    "error": f"Unknown kind: {kind!r}",
                    "allowed": ["initial", "extracted", "selected", "gnd", "final"],
                }
            )
        return json.dumps(
            {"kind": kind, "count": len(payload), "keywords": payload},
            ensure_ascii=False,
            default=str,
        )


class GetKeywordChainsTool(BaseChatTool):
    name = "get_keyword_chains"
    description = (
        "Return Schlagwortketten (keyword chains) extracted by the pipeline. "
        "Each chain has a list of terms and a short reason."
    )
    parameters_schema = {"type": "object", "properties": {}}

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        return bool(ctx and ctx.keyword_chains)

    def execute(self, session: Any, **_: Any) -> str:
        ctx = _ctx(session)
        chains = ctx.keyword_chains if ctx else []
        return json.dumps(
            {"count": len(chains), "chains": chains},
            ensure_ascii=False,
            default=str,
        )


class GetGndEntriesTool(BaseChatTool):
    name = "get_gnd_entries"
    description = (
        "Return ALL GND entries from the Phase 2 catalog search. "
        "Each entry has title, gnd_id, gnd_ids, classifications. "
        "Use this when you need the complete GND result set, not just "
        "a substring search."
    )
    parameters_schema = {"type": "object", "properties": {}}

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        return bool(ctx and ctx.gnd_entries)

    def execute(self, session: Any, **_: Any) -> str:
        ctx = _ctx(session)
        entries = ctx.gnd_entries if ctx else []
        return json.dumps(
            {"count": len(entries), "entries": entries},
            ensure_ascii=False,
            default=str,
        )


class GetGndEntriesPerKeywordTool(BaseChatTool):
    name = "get_gnd_entries_per_keyword"
    description = (
        "Return the mapping of original search terms to GND titles "
        "from Phase 2 catalog search. Shows which GND results belong "
        "to which query keyword."
    )
    parameters_schema = {"type": "object", "properties": {}}

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        return bool(ctx and ctx.gnd_entries_per_keyword)

    def execute(self, session: Any, **_: Any) -> str:
        ctx = _ctx(session)
        mapping = ctx.gnd_entries_per_keyword if ctx else {}
        return json.dumps(
            {"count": len(mapping), "mapping": mapping},
            ensure_ascii=False,
            default=str,
        )


class GetDkClassificationsTool(BaseChatTool):
    name = "get_dk_classifications"
    description = (
        "Return DK (Dezimalklassifikation) results from the classification "
        "step. Each entry has code, title, confidence, reasoning."
    )
    parameters_schema = {"type": "object", "properties": {}}

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        return bool(ctx and ctx.dk_classifications)

    def execute(self, session: Any, **_: Any) -> str:
        ctx = _ctx(session)
        items = ctx.dk_classifications if ctx else []
        return json.dumps(
            {"count": len(items), "classifications": items},
            ensure_ascii=False,
            default=str,
        )


class GetDkTitlesForCodeTool(BaseChatTool):
    name = "get_dk_titles_for_code"
    description = (
        "Return DK catalog search results for a single DK code (titles + "
        "metadata from `shared_context.dk_search_results`)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "DK code (e.g. '546.48')"},
        },
        "required": ["code"],
    }

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        return bool(ctx and ctx.dk_search_results)

    def execute(self, session: Any, code: str = "", **_: Any) -> str:
        ctx = _ctx(session)
        if ctx is None or not code:
            return json.dumps({"error": "Missing context or empty code"})
        wanted = code.strip()
        matches = [
            entry for entry in ctx.dk_search_results
            if str(entry.get("dk") or entry.get("code") or "").strip() == wanted
        ]
        return json.dumps(
            {"code": wanted, "count": len(matches), "results": matches},
            ensure_ascii=False,
            default=str,
        )


# ----- Chunked-response accessors --------------------------------------------


class GetChunkResponseTool(BaseChatTool):
    name = "get_chunk_response"
    description = (
        "Fetch the raw LLM response for one chunk of a chunked pipeline run "
        "(stored under step_results keys like 'chunk_0', 'chunk_1', ...)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "chunk_id": {"type": "integer", "description": "0-based chunk index"},
        },
        "required": ["chunk_id"],
    }

    def _chunk_keys(self, ctx) -> List[str]:
        return sorted(k for k in (ctx.step_results or {}) if k.startswith("chunk_"))

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        return bool(ctx and self._chunk_keys(ctx))

    def execute(self, session: Any, chunk_id: int = -1, **_: Any) -> str:
        ctx = _ctx(session)
        if ctx is None or chunk_id < 0:
            return json.dumps({"error": "Missing context or invalid chunk_id"})
        key = f"chunk_{chunk_id}"
        if key not in (ctx.step_results or {}):
            return json.dumps(
                {
                    "error": f"No such chunk: {key}",
                    "available_chunks": self._chunk_keys(ctx),
                }
            )
        return json.dumps(
            {"chunk_id": chunk_id, "result": ctx.step_results[key]},
            ensure_ascii=False,
            default=str,
        )


class FindChunkForKeywordTool(BaseChatTool):
    name = "find_chunk_for_keyword"
    description = (
        "Locate chunk(s) whose stored response contains the given keyword "
        "(case-insensitive substring match)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "keyword": {"type": "string"},
        },
        "required": ["keyword"],
    }

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        if ctx is None:
            return False
        return any(k.startswith("chunk_") for k in (ctx.step_results or {}))

    def execute(self, session: Any, keyword: str = "", **_: Any) -> str:
        ctx = _ctx(session)
        if ctx is None or not keyword:
            return json.dumps({"error": "Missing context or empty keyword"})
        needle = keyword.lower()
        hits: List[Dict[str, Any]] = []
        for key, val in (ctx.step_results or {}).items():
            if not key.startswith("chunk_"):
                continue
            haystack = json.dumps(val, ensure_ascii=False, default=str).lower()
            if needle in haystack:
                hits.append({"chunk_key": key})
        return json.dumps(
            {"keyword": keyword, "count": len(hits), "hits": hits},
            ensure_ascii=False,
        )


# ----- GND helpers -----------------------------------------------------------


def _entry_matches(entry: Dict[str, Any], needle: str) -> bool:
    """True if `needle` matches title or any gnd_id of the entry."""
    title = str(entry.get("title", "")).lower()
    if needle in title:
        return True
    primary = str(entry.get("gnd_id", ""))
    if primary and needle in primary.lower():
        return True
    for gid in entry.get("gnd_ids", []) or []:
        if needle in str(gid).lower():
            return True
    return False


class SearchInGndPoolTool(BaseChatTool):
    name = "search_in_gnd_pool"
    description = (
        "Search the current session's GND entry pool (no network). "
        "Case-insensitive substring match against title and GND-IDs."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "term": {"type": "string"},
        },
        "required": ["term"],
    }

    def available_for(self, session: Any) -> bool:
        ctx = _ctx(session)
        return bool(ctx and ctx.gnd_entries)

    def execute(self, session: Any, term: str = "", **_: Any) -> str:
        ctx = _ctx(session)
        if ctx is None or not term:
            return json.dumps({"error": "Missing context or empty term"})
        needle = term.lower()
        matches = [e for e in ctx.gnd_entries if _entry_matches(e, needle)]
        return json.dumps(
            {"term": term, "count": len(matches), "entries": matches},
            ensure_ascii=False,
            default=str,
        )


class ValidateGndTermTool(BaseChatTool):
    name = "validate_gnd_term"
    description = (
        "Verify whether a term has a GND record. First checks the session "
        "pool; if missing, falls back to a single MCP `search_gnd` call "
        "(if MCP is wired). Returns {verified, gnd_id, source}."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "term": {"type": "string"},
        },
        "required": ["term"],
    }

    def __init__(self, mcp_registry: Any = None) -> None:
        self._mcp = mcp_registry

    def available_for(self, session: Any) -> bool:
        # Available whenever session is bound — falls back to MCP for empty pools.
        return True

    def execute(self, session: Any, term: str = "", **_: Any) -> str:
        if not term:
            return json.dumps({"error": "Empty term"})
        ctx = _ctx(session)
        needle = term.lower()
        # (a) session pool match
        if ctx is not None and ctx.gnd_entries:
            for entry in ctx.gnd_entries:
                if _entry_matches(entry, needle):
                    gid = entry.get("gnd_id") or (entry.get("gnd_ids") or [None])[0]
                    return json.dumps(
                        {"verified": True, "gnd_id": gid, "source": "session"},
                        ensure_ascii=False,
                    )
        # (b) MCP fallback
        if self._mcp is None:
            return json.dumps(
                {"verified": False, "gnd_id": None, "source": "none"},
                ensure_ascii=False,
            )
        try:
            raw = self._mcp.execute(
                "search_gnd", {"term": term, "min_results": 1}
            )
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            logger.warning("MCP fallback failed for %r: %s", term, e)
            return json.dumps(
                {"verified": False, "gnd_id": None, "source": "none", "error": str(e)},
                ensure_ascii=False,
            )
        gid = _extract_gnd_id_from_search(parsed)
        if gid:
            return json.dumps(
                {"verified": True, "gnd_id": gid, "source": "mcp"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"verified": False, "gnd_id": None, "source": "none"},
            ensure_ascii=False,
        )


def _extract_gnd_id_from_search(parsed: Any) -> Optional[str]:
    """Best-effort: pull a GND-ID out of the MCP `search_gnd` payload."""
    if isinstance(parsed, dict):
        if parsed.get("gnd_id"):
            return parsed["gnd_id"]
        for key in ("results", "entries", "hits"):
            inner = parsed.get(key)
            if isinstance(inner, list) and inner:
                return _extract_gnd_id_from_search(inner[0])
    if isinstance(parsed, list) and parsed:
        return _extract_gnd_id_from_search(parsed[0])
    return None


# ----- Factory ---------------------------------------------------------------


def alima_tools(mcp_registry: Any = None) -> List[BaseChatTool]:
    """Instantiate ALIMA-specific tools. ``mcp_registry`` is needed only by
    ValidateGndTermTool for its fallback path."""
    return [
        GetKeywordsTool(),
        GetKeywordChainsTool(),
        GetGndEntriesTool(),
        GetGndEntriesPerKeywordTool(),
        GetDkClassificationsTool(),
        GetDkTitlesForCodeTool(),
        GetChunkResponseTool(),
        FindChunkForKeywordTool(),
        SearchInGndPoolTool(),
        ValidateGndTermTool(mcp_registry=mcp_registry),
    ]
