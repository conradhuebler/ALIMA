"""Shared Context for MetaAgent Orchestration - Claude Generated

Provides ToolResultCache for avoiding redundant tool calls across SubAgents,
and SharedContext for passing data between pipeline stages.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.agents.base_shared_context import BaseSharedContext
from src.core.url_utils import gnd_url, swb_ppn_url
from src.utils.classification_systems import codes_for_system

logger = logging.getLogger(__name__)

#: Cap on the ``initial_gnd_classes`` display summary. Since the authority-DDC
#: read-back the pool can carry hundreds of DDC codes; this field is a ranked
#: hint, not the full set (which lives in per-subject classifications).
_INITIAL_GND_CLASSES_LIMIT = 30

#: Provenance keys of the geo_sachdb payload that go into a saved result. The
#: local file path and the row counts stay out: they describe the operator's
#: machine, not the data version.
_GEO_SOURCE_KEYS = ("source", "source_date", "source_sha256", "schema_version")

#: Kind that decides between several local headings of one term: exactly one
#: of this kind makes the hit unique (operator rule, Nov 6 2026).
_GEO_PREFERRED_KIND = "Sachschlagwort"

#: ``extra`` key the step ``geo_prior`` writes its payload to (output mapping
#: in ``alima_v51_105.yaml``). - Claude Generated
GEO_PRIOR_EXTRA_KEY = "geo_sachdb_prior"

#: Fields of a ``geo_prior`` candidate a saved result keeps.
_GEO_PRIOR_KEYS = ("notation", "system", "code", "label", "via_keywords", "via_core_keywords")


def _geo_pick(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The local heading a term stands for: the only one, or the only
    Sachschlagwort among several; ``None`` when that is not unique.
    - Claude Generated
    """
    if len(records) == 1:
        return records[0]
    subject = [rec for rec in records if rec.get("kind") == _GEO_PREFERRED_KIND]
    return subject[0] if len(subject) == 1 else None


def _geo_terms(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per looked-up keyword: its local headings and which one, if any, the
    WinIBW entry may use. ``status``: ``unique`` (``ppn`` set), ``no_ppn``
    (unique heading without PPN) or ``ambiguous``. - Claude Generated
    """
    out: List[Dict[str, Any]] = []
    for match in payload.get("keyword_matches") or []:
        if not isinstance(match, dict):
            continue
        keyword = str(match.get("keyword") or "").strip()
        records = [
            {"label": str(rec.get("label") or ""), "kind": str(rec.get("kind") or ""),
             "ppn": str(rec.get("ppn") or "")}
            for rec in (match.get("records") or []) if isinstance(rec, dict)
        ]
        if not (keyword and records):
            continue
        chosen = _geo_pick(records)
        if chosen is None:
            status, ppn = "ambiguous", ""
        elif chosen["ppn"]:
            status, ppn = "unique", chosen["ppn"]
        else:
            status, ppn = "no_ppn", ""
        out.append({"keyword": keyword, "records": records, "record": chosen,
                    "status": status, "ppn": ppn})
    return out


def _geo_notations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per assigned notation the step checked: its local entries and the PPN
    the WinIBW entry may use. ``status``: ``unique`` (exactly one distinct PPN,
    ``ppn`` set), ``ambiguous`` (several), ``no_ppn`` or ``not_local``.
    - Claude Generated
    """
    out: List[Dict[str, Any]] = []
    for match in payload.get("notation_matches") or []:
        if not isinstance(match, dict):
            continue
        code = str(match.get("code") or "").strip()
        notation = str(match.get("normalized") or "").strip() or code
        if not notation:
            continue
        local = [
            {"notation": str(n.get("notation") or ""), "label": str(n.get("label") or ""),
             "ppn": str(n.get("ppn") or "")}
            for n in (match.get("local") or []) if isinstance(n, dict)
        ]
        ppns: List[str] = []
        for n in local:
            if n["ppn"] and n["ppn"] not in ppns:
                ppns.append(n["ppn"])
        if not local:
            status = "not_local"
        elif not ppns:
            status = "no_ppn"
        else:
            status = "unique" if len(ppns) == 1 else "ambiguous"
        out.append({
            "code": code,
            # The step checks the notations of its ``scheme`` (default DK).
            "system": str(match.get("system") or "DK").upper(),
            "notation": notation,
            "local": local,
            "status": status,
            "ppn": ppns[0] if status == "unique" else "",
        })
    return out


def _geo_sachdb_for_state(payload: Any) -> Dict[str, Any]:
    """The part of ``extra.geo_sachdb`` a saved result keeps, or ``{}``.

    ``terms`` (local headings per keyword, the usable PPN), ``notations``
    (local entries per assigned notation, the usable PPN), ``not_found``,
    ``candidates`` (notations of the matched headings, ``assigned`` marking
    those given) and of ``source`` only the data version. The WinIBW entry
    (``src/utils/winibw_record.py``) reads ``terms``/``notations``; this
    function writes no lines itself. A step without plugin
    (``disabled``/``unavailable``/``no_input``/``error``) or without any
    lookup result gives ``{}``, so such a result looks exactly like one
    without the step.
    - Claude Generated
    """
    if not isinstance(payload, dict):
        return {}
    status = payload.get("status")
    if status and status != "ok":
        return {}
    terms = _geo_terms(payload)
    notations = _geo_notations(payload)
    not_found = [str(term) for term in (payload.get("not_found") or []) if term]
    candidates = [dict(cand) for cand in (payload.get("candidates") or []) if isinstance(cand, dict)]
    if not (terms or not_found or candidates or any(n["local"] for n in notations)):
        return {}
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    return {
        "terms": terms,
        "notations": notations,
        "not_found": not_found,
        "candidates": candidates,
        "source": {key: source[key] for key in _GEO_SOURCE_KEYS if source.get(key)},
    }


def _catalog_codes(dk_entries: Any, scheme: str) -> set:
    """Comparison forms of the catalogue notations of ``scheme`` in the DK
    step's ``dk_entries`` (``{dk, classification_type, …}``). - Claude Generated"""
    from src.utils.lookups.geo_sachdb.schema import compact, normalize_notation

    out = set()
    for entry in dk_entries or []:
        if not isinstance(entry, dict):
            continue
        system = str(entry.get("classification_type") or entry.get("type") or scheme).strip().upper()
        code = str(entry.get("dk") or entry.get("code") or "").strip()
        if not code or system != scheme:
            continue
        out.add(compact(normalize_notation(code, scheme)[1]))
    return out


def _geo_prior_for_state(payload: Any, dk_entries: Any) -> Dict[str, Any]:
    """The part of ``extra.geo_sachdb_prior`` a saved result keeps, or ``{}``.

    ``candidates``: every local notation the classification saw, with the
    keywords and core keywords that reach it and ``in_catalog``, decided here
    against the catalogue entries of the DK step. A step without plugin or
    without a local notation gives ``{}``. - Claude Generated
    """
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return {}
    from src.utils.lookups.geo_sachdb.schema import compact

    candidates = []
    catalog: Dict[str, set] = {}
    for cand in payload.get("candidates") or []:
        if not isinstance(cand, dict) or not cand.get("code"):
            continue
        system = str(cand.get("system") or "DK").upper()
        if system not in catalog:
            catalog[system] = _catalog_codes(dk_entries, system)
        entry = {key: cand.get(key) for key in _GEO_PRIOR_KEYS}
        entry["system"] = system
        entry["via_keywords"] = [str(t) for t in (cand.get("via_keywords") or []) if t]
        entry["via_core_keywords"] = [str(t) for t in (cand.get("via_core_keywords") or []) if t]
        entry["in_catalog"] = compact(str(cand["code"])) in catalog[system]
        candidates.append(entry)
    return {"candidates": candidates} if candidates else {}


#: ``extra`` key the step ``publication_form`` writes its detection to.
#: - Claude Generated
PUBLICATION_FORM_EXTRA_KEY = "publication_form"
_PUBLICATION_FORM_KEYS = ("form", "label", "form_keyword", "evidence", "rule")


def _publication_form_for_state(payload: Any) -> Dict[str, Any]:
    """The detection of ``extra.publication_form`` as a saved result keeps it:
    ``{form, label, system, code, form_keyword, evidence, rule}`` (the
    notation split into ``system`` and ``code``), or ``{}`` without a hit.
    - Claude Generated
    """
    if not isinstance(payload, dict) or not payload.get("form"):
        return {}
    system, code = (list(payload.get("notation") or []) + ["", ""])[:2]
    out = {key: str(payload.get(key) or "") for key in _PUBLICATION_FORM_KEYS}
    out["system"], out["code"] = str(system or ""), str(code or "")
    return out


@dataclass
class ToolResultCache:
    """Caches tool results to avoid redundant calls across SubAgents.

    Cache key is computed from tool_name + sorted_args_json.
    Stores results and tracks hit count for optimization metrics.
    """
    results: Dict[str, Any] = field(default_factory=dict)
    hit_count: Dict[str, int] = field(default_factory=dict)
    miss_count: Dict[str, int] = field(default_factory=dict)

    def get_cache_key(self, tool_name: str, args: dict) -> str:
        """Generate a deterministic cache key from tool name and arguments.

        Args:
            tool_name: Name of the tool
            args: Tool arguments as dict

        Returns:
            Cache key string
        """
        # Sort args to ensure consistent key generation
        args_json = json.dumps(args, sort_keys=True, default=str)
        args_hash = hashlib.md5(args_json.encode()).hexdigest()[:8]
        return f"{tool_name}:{args_hash}"

    def get(self, tool_name: str, args: dict) -> Optional[Any]:
        """Get cached result if available.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Cached result or None if not found
        """
        key = self.get_cache_key(tool_name, args)
        if key in self.results:
            self.hit_count[key] = self.hit_count.get(key, 0) + 1
            logger.debug(f"Cache HIT for {tool_name} (key: {key})")
            return self.results[key]

        self.miss_count[tool_name] = self.miss_count.get(tool_name, 0) + 1
        logger.debug(f"Cache MISS for {tool_name}")
        return None

    def set(self, tool_name: str, args: dict, result: Any) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Result to cache
        """
        key = self.get_cache_key(tool_name, args)
        self.results[key] = result
        logger.debug(f"Cached result for {tool_name} (key: {key})")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with total_hits, total_misses, cache_size, hit_rate
        """
        total_hits = sum(self.hit_count.values())
        total_misses = sum(self.miss_count.values())
        total_requests = total_hits + total_misses

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "cache_size": len(self.results),
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "tools_cached": list(set(k.split(":")[0] for k in self.results.keys())),
        }

    def clear(self) -> None:
        """Clear all cached results."""
        self.results.clear()
        self.hit_count.clear()
        self.miss_count.clear()


@dataclass
class SharedContext(BaseSharedContext):
    """Shared state across all SubAgents in a MetaAgent execution.

    Extends BaseSharedContext with ALIMA-specific fields.
    """
    # Input data
    abstract: str = ""
    initial_keywords: List[str] = field(default_factory=list)
    input_type: str = "text"
    source_value: Optional[str] = None

    # Which workflow is driving this context; set by WorkflowExecutor.run so the
    # exported state can name it without the caller having to thread it through.
    # - Claude Generated
    workflow_name: str = ""

    # Personal rules (``src/core/user_rules.py``) injected into this run's
    # prompts, as ``{id, text}``. Filled by the injection points as they fire,
    # read by ``to_keyword_analysis_state``. - Claude Generated
    applied_user_rules: List[Dict[str, str]] = field(default_factory=list)

    # Shared resources
    tool_result_cache: ToolResultCache = field(default_factory=ToolResultCache)
    conversation_memory: List[Dict] = field(default_factory=list)

    # Working data (built during execution)
    working_title: str = ""
    extracted_keywords: List[str] = field(default_factory=list)
    gnd_entries: List[Dict] = field(default_factory=list)
    gnd_entries_per_keyword: Dict[str, List[str]] = field(default_factory=dict)  # term -> [titles] mapping for UI display
    selected_keywords: List[Dict] = field(default_factory=list)
    keyword_chains: List[Dict] = field(default_factory=list)  # Schlagwortketten: [{chain: [...], reason: "..."}]
    missing_concepts: List[str] = field(default_factory=list)  # From selection, drives feedback loop
    missing_concepts_searched: List[str] = field(default_factory=list)  # Already searched terms
    dk_classifications: List[Dict] = field(default_factory=list)
    rvk_classifications: List[Dict] = field(default_factory=list)  # RVK classifications from classification step
    dk_search_results: List[Dict] = field(default_factory=list)  # keyword-centric [{keyword, classifications:[...]}] (data_models contract)
    dk_search_results_flattened: List[Dict] = field(default_factory=list)  # DK-centric flat/merged [{dk, titles, count, ...}] (set by dk_postprocess) - Claude Generated
    dk_catalog_stats: Dict[str, Any] = field(default_factory=dict)  # Aggregated catalog stats: total_titles, unique_notations, top_notations

    # MetaAgent state
    quality_report: Dict[str, Any] = field(default_factory=dict)  # ReflectionStep output
    max_missing_reruns: int = 1  # Max times to rerun search for missing concepts

    # ------------------------------------------------------------------
    # Notation-agnostic aliases (general-notation generalization, WS2).
    # ``dk_*`` fields stay canonical; ``notation_*`` is preferred for new code
    # treating DK/DDC/RVK uniformly. - Claude Generated
    # ------------------------------------------------------------------
    @property
    def notation_codes(self) -> List[Dict]:
        return self.dk_classifications

    @notation_codes.setter
    def notation_codes(self, value: List[Dict]) -> None:
        self.dk_classifications = value

    @property
    def notation_search_results(self) -> List[Dict]:
        return self.dk_search_results

    @notation_search_results.setter
    def notation_search_results(self, value: List[Dict]) -> None:
        self.dk_search_results = value

    @property
    def notation_catalog_stats(self) -> Dict[str, Any]:
        return self.dk_catalog_stats

    @notation_catalog_stats.setter
    def notation_catalog_stats(self, value: Dict[str, Any]) -> None:
        self.dk_catalog_stats = value

    def set_step_result(self, step_name: str, result: Dict, quality: float = None) -> None:
        """Store result from a pipeline step.

        Args:
            step_name: Name of the step
            result: Result dictionary
            quality: Optional quality score for this step
        """
        self.step_results[step_name] = result
        if quality is not None:
            self.quality_scores[step_name] = quality
        logger.debug(f"Stored result for step '{step_name}' with quality {quality}")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to shared conversation memory.

        Args:
            role: Message role ('system', 'user', 'assistant', 'tool')
            content: Message content
        """
        self.conversation_memory.append({"role": role, "content": content})

    def get_recent_messages(self, limit: int = 10) -> List[Dict]:
        """Get recent messages from conversation memory.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        return self.conversation_memory[-limit:]

    def to_keyword_analysis_state(self):
        """Convert SharedContext to KeywordAnalysisState for output compatibility.

        Produces a state that is fully compatible with the hardcoded pipeline's
        KeywordAnalysisState, including search_results, Schlagwortketten,
        DK classifications with statistics, and RVK provenance.

        Returns:
            KeywordAnalysisState with all gathered data
        """
        from src.core.data_models import KeywordAnalysisState, LlmKeywordAnalysis, SearchResult

        # --- keyword titles (selected, GND-verified) ---
        # Prefer curated final_keywords from selection step (15-20 items with GND IDs).
        # Fall back to all selected_keywords if not available (e.g. old workflow runs).
        final_kws = (self.extra or {}).get("final_keywords") or []
        if final_kws:
            kw_titles = [
                f"{kw.get('keyword', '')} (GND-ID: {kw.get('gnd_id', '')})"
                if kw.get("gnd_id") else kw.get("keyword", "")
                for kw in final_kws if kw.get("keyword")
            ]
        else:
            kw_titles = [kw.get("title", kw.get("gnd_id", ""))
                         for kw in self.selected_keywords
                         if kw.get("title") or kw.get("gnd_id")]
            # This fallback used to be silent, so a run whose selection step
            # returned unparsable output looked like a normal one — it just
            # carried the whole chunk selection instead of the curated list
            # (56 keywords instead of ~20, glm-5.3-flash on 2026-09-03 11:05,
            # which prefixed its JSON with prose). A degraded run must say so.
            # - Claude Generated
            if kw_titles:
                logger.warning(
                    f"to_keyword_analysis_state: keine kuratierten final_keywords "
                    f"aus 'selection' — nutze die ungefilterte Auswahl "
                    f"({len(kw_titles)} Schlagworte). Das Ergebnis ist nicht "
                    f"die Endauswahl des Modells."
                )

        # --- DK codes as plain strings ---
        dk_codes = [cls.get("code", "") for cls in self.dk_classifications if cls.get("code")]

        # --- initial GND classes from entries with DDC codes ---
        # A ranked, bounded display summary of the DDC classes the initial GND
        # hits carry. Only SIGNAL-BEARING codes make the summary: a co-occurrence
        # ``count`` (how well the harvest supports it) or an authority
        # ``determinacy`` (1 = definitive … 4 = loosely related). Bare authority
        # DDC — the GND's own classification with neither count nor determinacy —
        # cannot be ranked by relevance, and since the authority-DDC read-back
        # the pool unions hundreds of them; an alphabetical dump of those is
        # noise, not a hint. They stay in the per-subject ``classifications``
        # (the full set); this summary shows only what has a relevance signal,
        # so it is empty when nothing is strongly associated. Ranking: evidenced
        # co-occurrence first (strongest), then authority by determinacy.
        # - Claude Generated
        ddc_best: Dict[str, Dict[str, int]] = {}
        for entry in self.gnd_entries:
            for code_entry in (entry.get("classifications") or {}).get("DDC") or []:
                if not isinstance(code_entry, dict):
                    continue  # bare code = no signal → not in the summary
                code = str(code_entry.get("code") or "").strip()
                count = int(code_entry.get("count") or 0)
                det = code_entry.get("determinacy")
                if not code or (count == 0 and det is None):
                    continue  # no relevance signal → per-subject only
                best = ddc_best.setdefault(code, {"count": 0, "det": 9})
                best["count"] = max(best["count"], count)
                if det is not None:
                    best["det"] = min(best["det"], int(det))

        def _rank(item):
            code, sig = item
            return (
                0 if sig["count"] > 0 else 1,   # evidenced co-occurrence first
                -sig["count"],                  # strongest co-occurrence first
                sig["det"],                     # then authority, definitive first
                code,
            )

        initial_gnd_classes = [
            code for code, _ in sorted(ddc_best.items(), key=_rank)
        ][:_INITIAL_GND_CLASSES_LIMIT]

        # --- dk_search_results_flattened: DK-centric flat/merged view ---
        # Prefer the rich list produced by the dk_postprocess step
        # (build_dk_search_results — real catalog titles + counts). Fall back to
        # deriving a thin version from the final classifications when that step
        # did not run (e.g. pipeline stopped early). - Claude Generated
        if self.dk_search_results_flattened:
            dk_search_results_flattened = self.dk_search_results_flattened
        else:
            dk_search_results_flattened = []
            for cls in self.dk_classifications:
                code = cls.get("code", "")
                title = cls.get("title", "")
                reason = cls.get("reason", cls.get("reasoning", ""))
                if code:
                    dk_search_results_flattened.append({
                        "dk": code,
                        "classification_type": "DK",
                        "titles": [title] if title else [],
                        "count": int(cls.get("confidence", 0) * 100),
                        "reasoning": reason,
                    })

        # --- DK statistics ---
        dk_statistics = None
        if dk_search_results_flattened:
            from collections import Counter
            dk_counter = Counter(r["dk"] for r in dk_search_results_flattened)
            total = sum(dk_counter.values())
            unique = len(dk_counter)
            top_10 = [
                {"dk": dk, "count": cnt, "percentage": round(cnt / total * 100, 1) if total else 0}
                for dk, cnt in dk_counter.most_common(10)
            ]
            dk_statistics = {
                "total_classifications": total,
                "unique_dk_codes": unique,
                "deduplication_ratio": round(unique / total, 2) if total else 0,
                "top_10": top_10,
            }

        # --- search_results: canonical {title → {gnd_ids: [...], classifications: {system: [codes]}}} ---
        # Build per-keyword so each keyword only shows titles that matched it
        search_results = []
        per_keyword = self.gnd_entries_per_keyword or {}
        for ekw in (self.extracted_keywords or ["meta_agent"]):
            kw_results: Dict[str, Dict] = {}
            matched_titles = set(per_keyword.get(ekw, []))
            # Also match canonical form (lowercased)
            for term, titles in per_keyword.items():
                if term.lower() == ekw.lower():
                    matched_titles.update(titles)
            for entry in self.gnd_entries:
                title = entry.get("title", "")
                if title in matched_titles:
                    gnd_ids = entry.get("gnd_ids", [])
                    primary_gnd_id = entry.get("gnd_id", "")
                    all_gnd_ids = list(gnd_ids) if gnd_ids else ([primary_gnd_id] if primary_gnd_id else [])
                    # Carry the frequency: pool ``count`` is the ranking placeholder (1 on
                    # a cache hit), the real Häufigkeit rides in ``display_count``. Dropping
                    # them here made agentic runs persist 0. - Claude Generated
                    kw_results[title] = {
                        "gnd_ids": all_gnd_ids,
                        "classifications": dict(entry.get("classifications") or {}),
                        "count": entry.get("count", 1),
                        **({"display_count": entry["display_count"]} if entry.get("display_count") is not None else {}),
                    }
            # Fallback: if no per-keyword mapping, show all (old behavior)
            if not kw_results and not per_keyword:
                for entry in self.gnd_entries:
                    title = entry.get("title", "")
                    if not title:
                        continue
                    gnd_ids = entry.get("gnd_ids", [])
                    primary_gnd_id = entry.get("gnd_id", "")
                    all_gnd_ids = list(gnd_ids) if gnd_ids else ([primary_gnd_id] if primary_gnd_id else [])
                    kw_results[title] = {
                        "gnd_ids": all_gnd_ids,
                        "classifications": dict(entry.get("classifications") or {}),
                        "count": entry.get("count", 1),
                        **({"display_count": entry["display_count"]} if entry.get("display_count") is not None else {}),
                    }
            search_results.append(SearchResult(
                search_term=ekw,
                results=kw_results,
            ))

        # --- Build Schlagwortketten text for final_llm_analysis ---
        chain_lines = []
        for chain_data in self.keyword_chains:
            chain_terms = chain_data.get("chain", [])
            reason = chain_data.get("reason", "")
            chain_str = " → ".join(chain_terms) if chain_terms else ""
            if chain_str:
                line = f"  {chain_str}"
                if reason:
                    line += f" ({reason})"
                chain_lines.append(line)

        # --- initial_llm_call_details: extraction step → abstract_tab ---
        extraction_result = self.step_results.get("extraction", {})
        initial_analysis = LlmKeywordAnalysis(
            task_name="extraction",
            model_used=self.model,
            provider_used=self.provider,
            prompt_template="initialisation",
            filled_prompt="",
            temperature=self.temperature,
            seed=self.seed,
            response_full_text=extraction_result.get("raw_output", ""),
            extracted_gnd_keywords=self.extracted_keywords,
        )

        # --- final_llm_analysis: keyword selection → verifikation tab / analyse_keywords ---
        selection_result = self.step_results.get("selection", {})
        kw_lines = "\n".join(
            f"- {kw.get('title', kw.get('gnd_id', ''))} (GND-ID: {kw.get('gnd_id', '')})"
            for kw in self.selected_keywords
        )
        response_parts = [f"Ausgewählte GND-Schlagworte ({len(self.selected_keywords)}):\n{kw_lines}"]
        if chain_lines:
            response_parts.append(f"\nSchlagwortketten ({len(self.keyword_chains)}):\n" + "\n".join(chain_lines))
        final_analysis = LlmKeywordAnalysis(
            task_name="keywords",
            model_used=self.model,
            provider_used=self.provider,
            prompt_template="keywords_chunked",
            filled_prompt="",
            temperature=self.temperature,
            seed=self.seed,
            response_full_text="\n".join(response_parts),
            extracted_gnd_keywords=kw_titles,
            missing_concepts=self.missing_concepts,
            # The chains were rendered into `response_full_text` as prose only,
            # so `results["keyword_chains"]` came out empty on every agentic run
            # while the classic pipeline filled it. The GUI still showed them
            # because `render_pipeline_result` scrapes the text for "→" lines —
            # which hid the loss from view but left export, report rendering and
            # every other consumer without the structured data. - Claude Generated
            keyword_chains=list(self.keyword_chains or []),
        )

        # --- dk_llm_analysis: classification step → dk_analysis_tab ---
        classification_result = self.step_results.get("classification", {})
        dk_reasoning = classification_result.get("reasoning", "")
        dk_raw = classification_result.get("raw_output", "")
        dk_text = dk_reasoning or dk_raw or ""
        dk_analysis = LlmKeywordAnalysis(
            task_name="dk_classification",
            model_used=self.model,
            provider_used=self.provider,
            prompt_template="dk_classification",
            filled_prompt="",
            temperature=self.temperature,
            seed=self.seed,
            response_full_text=dk_text,
            extracted_gnd_classes=dk_codes,
        ) if dk_text else None

        # --- RVK provenance ---
        rvk_provenance = {}
        if self.rvk_classifications:
            for rvk in self.rvk_classifications:
                code = rvk.get("code", "")
                if code:
                    rvk_provenance[code] = {
                        "source": "agentic_classification",
                        "confidence": rvk.get("confidence", 0),
                        "title": rvk.get("title", ""),
                    }

        state = KeywordAnalysisState(
            original_abstract=self.abstract,
            initial_keywords=self.extracted_keywords or self.initial_keywords,
            search_suggesters_used=["swb", "lobid"],
            working_title=self.working_title,
            input_type=self.input_type,
            source_value=self.source_value,
            initial_gnd_classes=initial_gnd_classes,
            initial_llm_call_details=initial_analysis,
            final_llm_analysis=final_analysis,
            dk_llm_analysis=dk_analysis,
            search_results=search_results,
            dk_classifications=dk_codes,
            pipeline_mode="agentic",
            workflow_name=self.workflow_name or None,
        )
        state.dk_search_results_flattened = dk_search_results_flattened
        state.dk_search_results = self.dk_search_results
        state.dk_statistics = dk_statistics
        if rvk_provenance:
            state.rvk_provenance = rvk_provenance
        state.report_markdown = (self.extra or {}).get("report_markdown", "") or ""

        # RSWK core + form headings named by the selection step, rendered like
        # the full list so consumers can display them side by side. Absent for
        # runs whose model did not name them. - Claude Generated
        def _as_titles(field: str) -> list:
            return [
                f"{kw.get('keyword', '')} (GND-ID: {kw.get('gnd_id', '')})"
                if kw.get("gnd_id") else kw.get("keyword", "")
                for kw in ((self.extra or {}).get(field) or [])
                if isinstance(kw, dict) and kw.get("keyword")
            ]

        # Which personal rules actually reached a prompt in this run — collected
        # by the injection points, so a saved result explains itself.
        # - Claude Generated
        state.applied_rules = [
            dict(entry) for entry in (getattr(self, "applied_user_rules", None) or [])
        ]
        state.rule_output = str((self.extra or {}).get("rule_output", "") or "")
        state.decision_prefilter = (self.extra or {}).get("decision_prefilter") or None

        # Shelf placement (local Aufstellungssystematik). ``shelf_final`` is what
        # survived the guard; the candidates travel with it so a result stays
        # readable when the LLM pick is switched off or was rejected.
        # - Claude Generated
        shelf_payload = (self.extra or {}).get("shelf") or {}
        placement = (self.extra or {}).get("shelf_final") or {}
        if placement or shelf_payload.get("candidates"):
            state.shelf = {
                "placement": dict(placement) if isinstance(placement, dict) else {},
                "candidates": list(shelf_payload.get("candidates") or []),
                "unplaced": list(shelf_payload.get("unplaced") or []),
                "scheme": dict(shelf_payload.get("scheme") or {}),
            }
        state.keyword_chains = list(self.keyword_chains or [])
        state.geo_sachdb = _geo_sachdb_for_state((self.extra or {}).get("geo_sachdb"))
        state.geo_prior = _geo_prior_for_state(
            (self.extra or {}).get(GEO_PRIOR_EXTRA_KEY), (self.extra or {}).get("dk_entries")
        )
        state.publication_form = _publication_form_for_state(
            (self.extra or {}).get(PUBLICATION_FORM_EXTRA_KEY)
        )
        # Source overview of the run (source_usage.attach_to). - Claude Generated
        state.source_usage = [
            dict(row) for row in ((self.extra or {}).get("source_usage") or [])
            if isinstance(row, dict)
        ]
        state.core_keywords = _as_titles("core_keywords")
        state.form_keywords = _as_titles("form_keywords")

        # Structured classifications: the flat `dk_codes` above loses the
        # per-system core/additional rank the classification step assigns, and
        # re-parsing the strings downstream cannot recover it. - Claude Generated
        from src.utils.classification_systems import (
            normalize_rank,
            split_classification_code,
        )

        entries: List[Dict[str, Any]] = []
        for cls in self.dk_classifications:
            raw_code = str(cls.get("code", "") or "").strip()
            if not raw_code:
                continue
            system, code = split_classification_code(raw_code)
            system = system or str(cls.get("type", "") or "").strip().upper()
            entry = {
                "system": system,
                "code": code,
                "display": f"{system} {code}".strip() if system else code,
            }
            rank = normalize_rank(cls.get("rank"))
            if rank:
                entry["rank"] = rank
            entries.append(entry)
        state.classification_entries = entries
        return state

    @classmethod
    def from_keyword_analysis_state(cls, state) -> "SharedContext":
        """Build a SharedContext from a KeywordAnalysisState (reverse of to_keyword_analysis_state).

        P-ζ bridge so chat tools can read SharedContext after a classical pipeline
        run. Promotion of the helper previously living in pipeline_chat_panel.

        Asymmetries (not reconstructible from KAS):
        - keyword_chains only present if classical pipeline successfully parsed
          them into ``final_llm_analysis.keyword_chains`` (LLM emitted JSON or
          XML chain block). Free-text chains in ``response_full_text`` are not
          back-parsed.
        - step_results / quality_scores / execution_history empty — agentic-only tools
          depending on them stay agentic-only.
        - Raw LLM output blobs dropped.

        Classical-friendly extras:
        - ``selected_keywords`` and ``extra["final_keywords"]`` are derived
          from ``final_llm_analysis.extracted_gnd_keywords`` so chat tools
          (``get_keywords(kind="selected"|"final")``) work on classical runs.

        Args:
            state: KeywordAnalysisState instance

        Returns:
            SharedContext mirroring the analysis state's tool-visible fields.
        """
        ctx = cls()
        ctx.working_title = getattr(state, "working_title", "") or ""
        ctx.abstract = getattr(state, "original_abstract", "") or ""
        ctx.initial_keywords = list(getattr(state, "initial_keywords", []) or [])
        ctx.input_type = getattr(state, "input_type", "text") or "text"
        ctx.source_value = getattr(state, "source_value", None)

        final = getattr(state, "final_llm_analysis", None)
        if final is not None:
            curated = list(getattr(final, "extracted_gnd_keywords", []) or [])
            ctx.extracted_keywords = curated
            ctx.keyword_chains = list(getattr(final, "keyword_chains", []) or [])
            ctx.missing_concepts = list(getattr(final, "missing_concepts", []) or [])
            ctx.provider = getattr(final, "provider_used", "") or ""
            ctx.model = getattr(final, "model_used", "") or ""
            ctx.temperature = getattr(final, "temperature", 0.5) or 0.5
            ctx.seed = getattr(final, "seed", None)
            verification = getattr(final, "verification", None)
            if verification:
                ctx.extra["verification"] = verification
            extracted_classes = getattr(final, "extracted_gnd_classes", None) or []
            if extracted_classes:
                ctx.extra["extracted_gnd_classes"] = list(extracted_classes)
            # Classical pipeline: Step-4 curated list IS the final keyword set.
            # Mirror it as selected_keywords (structured dicts) + extra.final_keywords
            # so chat-tool kind="selected" and kind="final" both work — matches the
            # semantics chat tools document.
            import re as _re
            _gnd_pat = _re.compile(r"\(GND-ID:\s*([^)]+)\)")
            selected: List[Dict] = []
            final_kw_dicts: List[Dict] = []
            for kw in curated:
                m = _gnd_pat.search(kw)
                title = _gnd_pat.sub("", kw).strip()
                gnd_id = (m.group(1).strip() if m else "")
                entry = {"keyword": title, "title": title, "gnd_id": gnd_id}
                selected.append(entry)
                final_kw_dicts.append(entry)
            if selected and not ctx.selected_keywords:
                ctx.selected_keywords = selected
            if final_kw_dicts and "final_keywords" not in ctx.extra:
                ctx.extra["final_keywords"] = final_kw_dicts

        # Flatten search_results -> gnd_entries (dedup by gnd_id) + per-keyword map
        gnd_entries: List[Dict] = []
        gnd_per_kw: Dict[str, List[str]] = {}
        seen_ids = set()
        for sr in getattr(state, "search_results", []) or []:
            term = getattr(sr, "search_term", "") or ""
            results = getattr(sr, "results", {}) or {}
            titles: List[str] = []
            for gnd_id, info in results.items():
                if not isinstance(info, dict):
                    info = {"value": info}
                title = info.get("title") or info.get("label") or ""
                titles.append(title or gnd_id)
                if gnd_id in seen_ids:
                    continue
                seen_ids.add(gnd_id)
                # Claude Generated - pre-format the canonical d-nb.info GND URL
                # (and SWB/PPN link when a PPN is stored) so the chat agent never
                # has to construct one from the bare id. See src/core/url_utils.py.
                entry = {"gnd_id": gnd_id, **info}
                _gurl = gnd_url(gnd_id)
                if _gurl:
                    entry["url"] = _gurl
                _swb = swb_ppn_url(str(info.get("ppn") or ""))
                if _swb:
                    entry["swb_url"] = _swb
                gnd_entries.append(entry)
            if term and titles:
                gnd_per_kw[term] = titles
        ctx.gnd_entries = gnd_entries
        ctx.gnd_entries_per_keyword = gnd_per_kw

        # KAS.dk_classifications is List[str]; SC expects List[Dict].
        raw_dk = getattr(state, "dk_classifications", []) or []
        normalised_dk: List[Dict] = []
        for cls_ in raw_dk:
            if isinstance(cls_, dict):
                normalised_dk.append(cls_)
            else:
                normalised_dk.append({"code": str(cls_)})
        # The classification entries carry system, display and rank; the plain
        # code list alone left a chat agent guessing what a DK number means.
        entries_by_code = {}
        for entry in getattr(state, "classification_entries", []) or []:
            if isinstance(entry, dict) and entry.get("display"):
                entries_by_code[str(entry.get("code") or "")] = entry
        if entries_by_code:
            normalised_dk = [
                {**entries_by_code.get(str(item.get("code", "")).replace("DK ", "", 1).strip(), {}), **item}
                for item in normalised_dk
            ]
        ctx.dk_classifications = normalised_dk

        ctx.dk_search_results = list(getattr(state, "dk_search_results", []) or [])
        stats = getattr(state, "dk_statistics", None)
        if stats:
            ctx.dk_catalog_stats = dict(stats)

        # RVK provenance -> rvk_classifications list (if KAS has it)
        rvk_prov = getattr(state, "rvk_provenance", None)
        if rvk_prov:
            ctx.rvk_classifications = [
                {"code": code, **(meta if isinstance(meta, dict) else {})}
                for code, meta in rvk_prov.items()
            ]

        final_keywords = getattr(state, "final_keywords", None)
        if final_keywords:
            ctx.extra["final_keywords"] = list(final_keywords)

        # Readable result sections for the chat (tool ``get_result_section`` and
        # the always-present summary); built from the state itself. - Claude Generated
        try:
            from src.utils.result_summary import result_sections

            sections = result_sections(state)
            if sections:
                ctx.extra["result_sections"] = sections
        except Exception as exc:  # noqa: BLE001 - the bridge must still return the context
            from src.utils.error_visibility import log_caught
            import logging

            log_caught(logging.getLogger(__name__), exc, "Ergebnisabschnitte für den Chat")

        return ctx

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SharedContext to a JSON-compatible dict for saving intermediate state.

        Tool result cache is NOT serialized (not needed for warm-start).

        Returns:
            Dict with all pipeline state fields
        """
        return {
            "abstract": self.abstract,
            "initial_keywords": self.initial_keywords,
            "input_type": self.input_type,
            "source_value": self.source_value,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_tokens_override": self.max_tokens_override,
            "use_decision_prefilter": self.use_decision_prefilter,
            "seed": self.seed,
            "think": self.think,
            "working_title": self.working_title,
            "extracted_keywords": self.extracted_keywords,
            "gnd_entries": self.gnd_entries,
            "gnd_entries_per_keyword": self.gnd_entries_per_keyword,
            "selected_keywords": self.selected_keywords,
            "keyword_chains": self.keyword_chains,
            "missing_concepts": self.missing_concepts,
            "missing_concepts_searched": self.missing_concepts_searched,
            "dk_classifications": self.dk_classifications,
            "rvk_classifications": self.rvk_classifications,
            "dk_search_results": self.dk_search_results,
            "dk_search_results_flattened": self.dk_search_results_flattened,
            "dk_catalog_stats": self.dk_catalog_stats,
            "step_results": self.step_results,
            "quality_scores": self.quality_scores,
            "quality_report": self.quality_report,
            "execution_history": self.execution_history,
            "max_missing_reruns": self.max_missing_reruns,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedContext":
        """Deserialize SharedContext from a dict (e.g. loaded from JSON file).

        Creates a fresh ToolResultCache — the cache is not restored from disk.

        Args:
            data: Dict as produced by to_dict()

        Returns:
            SharedContext with pre-populated pipeline state
        """
        ctx = cls(
            abstract=data.get("abstract", ""),
            initial_keywords=data.get("initial_keywords", []),
            input_type=data.get("input_type", "text"),
            source_value=data.get("source_value"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.5),
            max_tokens=data.get("max_tokens", 32768),
            max_tokens_override=data.get("max_tokens_override"),
            use_decision_prefilter=data.get("use_decision_prefilter", True),
            seed=data.get("seed"),
            think=data.get("think"),
        )
        ctx.working_title = data.get("working_title", "")
        ctx.extracted_keywords = data.get("extracted_keywords", [])
        ctx.gnd_entries = data.get("gnd_entries", [])
        ctx.gnd_entries_per_keyword = data.get("gnd_entries_per_keyword", {})
        ctx.selected_keywords = data.get("selected_keywords", [])
        ctx.keyword_chains = data.get("keyword_chains", [])
        ctx.missing_concepts = data.get("missing_concepts", [])
        ctx.missing_concepts_searched = data.get("missing_concepts_searched", [])
        ctx.dk_classifications = data.get("dk_classifications", [])
        ctx.rvk_classifications = data.get("rvk_classifications", [])
        ctx.dk_search_results = data.get("dk_search_results", [])
        ctx.dk_search_results_flattened = data.get("dk_search_results_flattened", [])
        ctx.dk_catalog_stats = data.get("dk_catalog_stats", {})
        ctx.step_results = data.get("step_results", {})
        ctx.quality_scores = data.get("quality_scores", {})
        ctx.quality_report = data.get("quality_report", {})
        ctx.execution_history = data.get("execution_history", [])
        ctx.max_missing_reruns = data.get("max_missing_reruns", 1)
        ctx.extra = data.get("extra", {}) or {}
        return ctx

    def save_to_file(self, path: str) -> None:
        """Save context state to a JSON file.

        Args:
            path: File path to write to
        """
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved SharedContext to {path}")

    @classmethod
    def load_from_file(cls, path: str) -> "SharedContext":
        """Load context state from a JSON file.

        Args:
            path: File path to read from

        Returns:
            SharedContext with restored pipeline state
        """
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        ctx = cls.from_dict(data)
        logger.info(f"Loaded SharedContext from {path} "
                    f"(steps completed: {list(ctx.step_results.keys())})")
        return ctx

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the context state.

        Returns:
            Dict with key context information
        """
        return {
            "abstract_length": len(self.abstract),
            "initial_keywords_count": len(self.initial_keywords),
            "working_title": self.working_title,
            "extracted_keywords_count": len(self.extracted_keywords),
            "gnd_entries_count": len(self.gnd_entries),
            "selected_keywords_count": len(self.selected_keywords),
            "dk_classifications_count": len(self.dk_classifications),
            "dk_catalog_stats": self.dk_catalog_stats,
            "missing_concepts": self.missing_concepts,
            "missing_concepts_searched": self.missing_concepts_searched,
            "cache_stats": self.tool_result_cache.get_stats(),
            "steps_completed": list(self.step_results.keys()),
            "provider": f"{self.provider}/{self.model}",
            "quality_report": self.quality_report,
            "execution_history_count": len(self.execution_history),
        }