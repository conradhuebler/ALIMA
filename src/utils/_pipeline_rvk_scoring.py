"""RVK scoring/selection of ``PipelineStepExecutor`` - Claude Generated.

``RvkScoringMixin`` carries the RVK candidate building, validation, anchor
derivation, scoring and final selection — extracted verbatim from
``pipeline_utils.PipelineStepExecutor`` (July 19, 2026; zero call-site
changes, methods reachable via MRO). DK steps live in the sibling
``_pipeline_dk_steps``; cross-calls go through ``self``.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
from ..core.data_models import AbstractData
from ..core.processing_utils import extract_missing_concepts_from_response
from .gnd_keyword_utils import canonicalize_keyword, extract_gnd_id, canonicalize_rvk_notation, is_plausible_nonstandard_rvk, deduplicate_canonical_keywords
from .pipeline_text_utils import repair_display_text
from .pipeline_formatters import PipelineResultFormatter

logger = logging.getLogger(__name__)


# Pure RVK-notation helpers. Extracted from the method bodies they used to
# live in (WP cleanup C): nested functions are unreachable from a test, and
# these decide hierarchy and ranking — _is_parent_like alone determines
# whether one notation is treated as a parent of another and dropped.
# _source_rank/_status_rank additionally existed TWICE, byte-identical, in
# two different methods. Verbatim moves; no behaviour change intended.
# - Claude Generated

def _branch_key(code: str) -> str:
    match = re.match(r"^([A-Z]{1,3})\s*", code)
    return match.group(1) if match else (code.split()[0] if code.split() else code)

def _compact_rvk(code: str) -> str:
    return re.sub(r"[^A-Z0-9.]", "", str(code or "").upper())

def _is_parent_like(parent_code: str, child_code: str) -> bool:
    parent_compact = _compact_rvk(parent_code)
    child_compact = _compact_rvk(child_code)
    return (
        bool(parent_compact)
        and bool(child_compact)
        and child_compact != parent_compact
        and child_compact.startswith(parent_compact)
        and len(child_compact) > len(parent_compact) + 1
    )

def _source_rank(source: str) -> int:
    return {
        # WP-D1 P2: the input record's own classification outranks every
        # derived source — the catalog states it about THIS document.
        "input_record": 4,
        "rvk_gnd_index": 3,
        "rvk_api": 2,
        "catalog": 1,
    }.get(source, 0)

def _status_rank(status: str) -> int:
    return {
        "standard": 3,
        "non_standard": 2,
        "validation_error": 1,
    }.get(status, 0)


def _is_strong_anchor_candidate(item: Dict[str, Any]) -> bool:
    """Whether a candidate's anchor match is strong enough to keep unconditionally.

    Two anchor hits is always strong; one hit needs corroboration (repeated
    keyword hits, a high catalog count, or several title hits). Extracted from
    ``_validate_catalog_rvk_candidates`` (WP cleanup F-14) so it can be tested —
    it was pure (no closure), moved verbatim. - Claude Generated
    """
    if item["anchor_hit_count"] >= 2:
        return True
    if item["anchor_hit_count"] == 1:
        return (
            int(item.get("keyword_hit_count", 0)) >= 2
            or int(item.get("count_value", 0)) >= 8
            or int(item.get("title_hit_count", 0)) >= 3
        )
    return False


def _can_take_branch(item: Dict[str, Any], branch_counts: Dict[str, int],
                     max_per_branch: int) -> bool:
    """Whether this candidate's RVK branch still has room in the shortlist.

    Extracted from ``_validate_catalog_rvk_candidates`` (WP cleanup F-14); the
    two closure variables (``branch_counts``, ``max_per_branch``) became explicit
    params — the body is otherwise unchanged. - Claude Generated
    """
    return branch_counts.get(item["branch"], 0) < max_per_branch

class RvkScoringMixin:
    """RVK scoring/selection methods (mixed into PipelineStepExecutor)."""

    def _build_rvk_api_fallback_results(
        self,
        keywords: List[str],
        stream_callback: Optional[callable] = None,
        max_results_per_keyword: int = 6,
    ) -> List[Dict[str, Any]]:
        """Build keyword-centric RVK candidates from the official RVK API."""
        from .lookups.cache import cached_call, lookup_cache_enabled
        from .lookups.resolve import build_lookup

        # Route through the `rvk_api` lookup plugin (single RVK construction path,
        # shared with the agent tool) so per-instance settings (timeout) are honored.
        # - Claude Generated
        _config = self._alima_config_for_cache()
        rvk_plugin = build_lookup(_config, "rvk_api")
        if rvk_plugin is None:
            # rvk_api disabled in the Plugins tab → no RVK-API fallback (Search
            # parity, WP P5: disable gates the pipeline too, not just the agent tool).
            if stream_callback:
                stream_callback("RVK-API-Plugin deaktiviert — überspringe RVK-Fallback\n", "dk_search")
            return []
        # Reuse the WP2 raw cache (F3): RVK API results are cached under the same
        # `rvk_search` key the agent tool uses, gated by the rvk_api plugin's
        # `cache_responses` setting. - Claude Generated
        _km = getattr(self, "cache_manager", None)
        _rvk_cache_on = lookup_cache_enabled(_config, "rvk_api")
        keyword_results = []

        for keyword in keywords:
            clean_keyword = canonicalize_keyword(keyword)
            try:
                # Cache the FULL plugin payload {keyword, count, results}, not the
                # bare ``results`` list. The rvk_search tool handler shares this cache
                # row and stores/returns the full dict verbatim; unwrapping before the
                # write made the two writers disagree on shape. Unwrap after the read;
                # tolerate a legacy bare-list row. - Claude Generated
                _rvk_payload = cached_call(
                    _km, _rvk_cache_on, "rvk_search", clean_keyword,
                    {"max_results": max_results_per_keyword},
                    lambda kw=clean_keyword: rvk_plugin.search_keyword(kw, max_results=max_results_per_keyword),
                )
                candidates = (
                    _rvk_payload.get("results", [])
                    if isinstance(_rvk_payload, dict)
                    else (_rvk_payload or [])
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"RVK API fallback failed for '{clean_keyword}': {exc}")
                if stream_callback:
                    stream_callback(f"  ⚠️ RVK API '{clean_keyword}': Fehler - {str(exc)}\n", "dk_search")
                continue

            if not candidates:
                continue

            classifications = []
            for candidate in candidates:
                classifications.append({
                    "dk": candidate["notation"],
                    "type": "RVK",
                    "classification_type": "RVK",
                    "count": 1,
                    "titles": [],
                    "matched_keywords": [clean_keyword],
                    "source": "rvk_api",
                    "label": candidate["label"],
                    "ancestor_path": candidate["ancestor_path"],
                    "register": candidate["register"],
                    "score": candidate["score"],
                    "branch_family": candidate["branch_family"],
                    "rvk_validation_status": "standard",
                    "validation_message": "",
                })

            keyword_results.append({
                "keyword": clean_keyword,
                "source": "rvk_api",
                "search_time_ms": 0.0,
                "classifications": classifications,
            })

            if stream_callback:
                stream_callback(
                    f"  ✅ RVK API {clean_keyword}: {len(classifications)} authority-backed Kandidaten\n",
                    "dk_search"
                )

        return keyword_results

    def _build_rvk_gnd_index_results(
        self,
        keyword_entries: List[Dict[str, str]],
        stream_callback: Optional[callable] = None,
        max_results_per_keyword: int = 6,
    ) -> List[Dict[str, Any]]:
        """Build RVK candidates from the official RVK MarcXML dump via GND links."""
        from .clients.rvk_marc_index import RvkMarcIndex

        index = RvkMarcIndex()
        if stream_callback:
            stream_callback("🔎 Nutze RVK MarcXML-GND-Index für standardisierte RVK-Kandidaten...\n", "dk_search")

        results = index.lookup_by_gnd_keywords(
            keyword_entries,
            max_results_per_keyword=max_results_per_keyword,
            progress_callback=(lambda msg: stream_callback(msg, "dk_search")) if stream_callback else None,
        )

        if stream_callback and results:
            total = sum(len(item.get("classifications", [])) for item in results)
            stream_callback(
                f"  ✅ RVK-GND-Index: {total} standardisierte Kandidaten für {len(results)} Keywords\n",
                "dk_search",
            )

        return results

    def _validate_catalog_rvk_candidates(
        self,
        keyword_results: List[Dict[str, Any]],
        stream_callback: Optional[callable] = None,
        rvk_anchor_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Validate catalog-derived RVK candidates and drop only obvious artifacts."""
        from .lookups.cache import cached_call, lookup_cache_enabled
        from .lookups.resolve import build_lookup

        # Validate through the `rvk_api` lookup plugin (single RVK construction path,
        # shared with the agent tool). - Claude Generated
        _config = self._alima_config_for_cache()
        rvk_plugin = build_lookup(_config, "rvk_api")
        if rvk_plugin is None:
            # rvk_api disabled → skip the API validation (Search parity, WP P5:
            # this is exactly the "runs with default settings when disabled"
            # divergence the WP set out to fix). Catalog-derived RVK candidates
            # pass through unvalidated rather than being dropped. - Claude Generated
            if stream_callback:
                stream_callback("RVK-API-Plugin deaktiviert — überspringe RVK-Validierung\n", "dk_search")
            return keyword_results
        # Persist RVK notation validations in the WP2 raw cache (F3), sharing the
        # `rvk_validate` key with the agent tool. - Claude Generated
        _km = getattr(self, "cache_manager", None)
        _rvk_cache_on = lookup_cache_enabled(_config, "rvk_api")

        validation_cache: Dict[str, Dict[str, Any]] = {}
        cleaned_results: List[Dict[str, Any]] = []
        standard_count = 0
        nonstandard_count = 0
        artifact_count = 0
        validation_error_count = 0
        pruned_general_count = 0
        max_validation_candidates = 120
        max_anchor_candidates = 96
        max_exploration_candidates = 12
        max_per_anchor = 12
        max_per_branch = 18
        unique_plausible_codes = []
        seen_plausible_codes = set()
        evidence_by_code: Dict[str, Dict[str, Any]] = {}
        anchor_terms = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }

        for kw_result in keyword_results:
            keyword = kw_result.get("keyword", "")
            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", "DK"))).upper()
                if cls_type != "RVK":
                    continue

                normalized = canonicalize_rvk_notation(classification.get("dk", ""))
                if not normalized:
                    continue
                if not is_plausible_nonstandard_rvk(normalized):
                    validation_cache[normalized] = {
                        "status": "artifact",
                        "notation": normalized,
                        "message": "Implausible RVK notation pattern",
                    }
                    continue
                evidence = evidence_by_code.setdefault(
                    normalized,
                    {
                        "count": 0,
                        "keyword_hits": set(),
                        "title_hits": 0,
                    },
                )
                evidence["count"] += int(classification.get("count", 0) or 0)
                if keyword:
                    evidence["keyword_hits"].add(keyword)
                evidence["title_hits"] += len(classification.get("titles", []) or [])
                if normalized not in seen_plausible_codes:
                    seen_plausible_codes.add(normalized)
                    unique_plausible_codes.append(normalized)

        candidate_meta = []
        for code in unique_plausible_codes:
            evidence = evidence_by_code.get(code, {})
            keyword_hits = {
                canonicalize_keyword(str(item or ""))
                for item in evidence.get("keyword_hits", set())
                if canonicalize_keyword(str(item or ""))
            }
            anchor_hits = sorted(anchor_terms.intersection(keyword_hits))
            specificity_bonus = min(len(code.replace(" ", "")), 12)
            keyword_hit_count = len(keyword_hits)
            count_value = int(evidence.get("count", 0))
            title_hit_count = int(evidence.get("title_hits", 0))
            score = (
                len(anchor_hits) * 120
                + keyword_hit_count * 15
                + count_value * 4
                + min(title_hit_count, 6)
                + specificity_bonus
            )
            candidate_meta.append({
                "code": code,
                "score": score,
                "anchor_hits": anchor_hits,
                "anchor_hit_count": len(anchor_hits),
                "branch": _branch_key(code),
                "keyword_hit_count": keyword_hit_count,
                "count_value": count_value,
                "title_hit_count": title_hit_count,
            })

        candidate_meta.sort(
            key=lambda item: (
                -int(item["anchor_hit_count"]),
                -int(item["score"]),
                item["code"],
            )
        )

        selected_codes: List[str] = []
        selected_set = set()
        branch_counts: Dict[str, int] = {}
        anchor_counts: Dict[str, int] = {anchor: 0 for anchor in anchor_terms}
        meta_by_code = {item["code"]: item for item in candidate_meta}

        anchored_candidates = [
            item for item in candidate_meta
            if item["anchor_hit_count"] > 0 and (not anchor_terms or _is_strong_anchor_candidate(item))
        ]
        if anchor_terms and not anchored_candidates:
            anchored_candidates = [item for item in candidate_meta if item["anchor_hit_count"] > 0]
        exploratory_candidates = [item for item in candidate_meta if item["anchor_hit_count"] == 0]

        def _select_item(item: Dict[str, Any]) -> None:
            code = item["code"]
            if code in selected_set:
                return
            selected_set.add(code)
            selected_codes.append(code)
            branch_counts[item["branch"]] = branch_counts.get(item["branch"], 0) + 1
            for anchor in item["anchor_hits"]:
                anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

        if anchored_candidates:
            for item in anchored_candidates:
                if len(selected_codes) >= max_anchor_candidates:
                    break
                if not _can_take_branch(item, branch_counts, max_per_branch):
                    continue
                if item["anchor_hits"] and not any(anchor_counts.get(anchor, 0) < max_per_anchor for anchor in item["anchor_hits"]):
                    continue
                _select_item(item)

            for item in anchored_candidates:
                if len(selected_codes) >= max_anchor_candidates:
                    break
                if item["code"] in selected_set or not _can_take_branch(item, branch_counts, max_per_branch):
                    continue
                _select_item(item)

        remaining_slots = max_validation_candidates - len(selected_codes)
        exploration_slots = min(max_exploration_candidates, max(0, remaining_slots))

        for item in exploratory_candidates:
            if exploration_slots <= 0 or len(selected_codes) >= max_validation_candidates:
                break
            if not _can_take_branch(item, branch_counts, max_per_branch):
                continue
            _select_item(item)
            exploration_slots -= 1

        if not selected_codes:
            # Fallback: keep the strongest codes even if no anchor-balanced shortlist could be built.
            selected_codes = [item["code"] for item in candidate_meta[:max_validation_candidates]]

        selected_total = len(selected_codes)
        anchored_total = sum(1 for item in candidate_meta if item["code"] in set(selected_codes) and item["anchor_hit_count"] > 0)
        exploratory_total = selected_total - anchored_total

        if selected_codes and stream_callback:
            shortlist_parts = []
            if anchor_terms:
                shortlist_parts.append(f"{anchored_total} ankergestützt")
                if exploratory_total:
                    shortlist_parts.append(f"{exploratory_total} explorativ")
                shortlist_parts.append(f"{len(branch_counts)} Zweige")
            stream_callback(
                (
                    f"🔎 Prüfe {len(selected_codes)} eindeutige RVK-Kandidaten gegen die RVK-API"
                    + (f" ({', '.join(shortlist_parts)})" if shortlist_parts else "")
                    + "...\n"
                ),
                "dk_search",
            )

        def _validate_code(code: str) -> Tuple[str, Dict[str, Any]]:
            # Cache the FULL plugin payload {notation, result}. rvk_validate's only
            # parameter IS its cache key, so this row is shared 1:1 with the
            # rvk_validate tool handler (which stores/returns the full dict). Storing
            # the inner ``result`` here made a tool-written row read back without a
            # top-level ``status`` → the code silently dropped from
            # standard_validated_codes below. Unwrap after the read; a legacy row is
            # already the inner dict (no ``result`` key). - Claude Generated
            payload = cached_call(
                _km, _rvk_cache_on, "rvk_validate", code, {},
                lambda c=code: rvk_plugin.validate_notation(c),
            )
            result = payload.get("result", payload) if isinstance(payload, dict) else payload
            return code, result

        if selected_codes:
            max_workers = min(8, len(selected_codes))
            progress_step = max(10, len(selected_codes) // 5)
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_validate_code, code): code
                    for code in selected_codes
                }
                for future in as_completed(future_map):
                    code = future_map[future]
                    try:
                        normalized_code, validation = future.result()
                    except Exception as exc:
                        normalized_code = code
                        validation = {
                            "status": "validation_error",
                            "notation": code,
                            "label": "",
                            "register": [],
                            "ancestor_path": "",
                            "branch_family": "",
                            "message": str(exc),
                        }

                    validation_cache[normalized_code] = validation
                    completed += 1
                    if stream_callback and (completed == len(selected_codes) or completed % progress_step == 0):
                        stream_callback(
                            f"  ↳ RVK-API-Validierung: {completed}/{len(selected_codes)}\n",
                            "dk_search",
                        )

        pruned_standard_codes = set()
        standard_validated_codes = [
            code for code in selected_codes
            if validation_cache.get(code, {}).get("status") == "standard"
        ]
        for parent_code in standard_validated_codes:
            parent_meta = meta_by_code.get(parent_code, {})
            parent_branch = parent_meta.get("branch", "")
            for child_code in standard_validated_codes:
                if child_code == parent_code:
                    continue
                child_meta = meta_by_code.get(child_code, {})
                if parent_branch and parent_branch != child_meta.get("branch", ""):
                    continue
                if not _is_parent_like(parent_code, child_code):
                    continue
                if (
                    int(child_meta.get("anchor_hit_count", 0)) >= int(parent_meta.get("anchor_hit_count", 0))
                    and int(child_meta.get("score", 0)) >= int(parent_meta.get("score", 0)) - 10
                ):
                    pruned_standard_codes.add(parent_code)
                    break

        for kw_result in keyword_results:
            cleaned_classifications = []

            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", "DK"))).upper()
                if cls_type != "RVK":
                    cleaned_classifications.append(classification)
                    continue

                normalized = canonicalize_rvk_notation(classification.get("dk", ""))
                if not normalized:
                    artifact_count += 1
                    continue

                validation = validation_cache.get(normalized)
                if validation is None:
                    validation = {
                        "status": "artifact",
                        "notation": normalized,
                        "message": "Niedrige Prioritaet - nicht gegen die RVK-API geprüft",
                    }
                    validation_cache[normalized] = validation

                status = validation.get("status", "")
                if status == "standard":
                    normalized = validation.get("notation") or normalized
                    if normalized in pruned_standard_codes:
                        pruned_general_count += 1
                        continue
                    updated = dict(classification)
                    updated["dk"] = normalized
                    updated["rvk_validation_status"] = "standard"
                    updated["validation_message"] = ""
                    if validation.get("label"):
                        updated["label"] = validation["label"]
                    if validation.get("ancestor_path"):
                        updated["ancestor_path"] = validation["ancestor_path"]
                    if validation.get("register"):
                        updated["register"] = validation["register"]
                    if validation.get("branch_family"):
                        updated["branch_family"] = validation["branch_family"]
                    cleaned_classifications.append(updated)
                    standard_count += 1
                    continue

                if status == "validation_error":
                    updated = dict(classification)
                    updated["dk"] = normalized
                    updated["rvk_validation_status"] = "validation_error"
                    updated["validation_message"] = validation.get("message", "")
                    cleaned_classifications.append(updated)
                    validation_error_count += 1
                    continue

                if is_plausible_nonstandard_rvk(normalized):
                    updated = dict(classification)
                    updated["dk"] = normalized
                    updated["rvk_validation_status"] = "non_standard"
                    updated["validation_message"] = validation.get("message", "Notation Not Found")
                    cleaned_classifications.append(updated)
                    nonstandard_count += 1
                    continue

                artifact_count += 1

            if cleaned_classifications:
                updated_result = dict(kw_result)
                updated_result["classifications"] = cleaned_classifications
                cleaned_results.append(updated_result)

        if stream_callback and (standard_count or nonstandard_count or artifact_count or validation_error_count or pruned_general_count):
            parts = []
            if standard_count:
                parts.append(f"{standard_count} standard")
            if nonstandard_count:
                parts.append(f"{nonstandard_count} nicht-standardisiert/lokal")
            if validation_error_count:
                parts.append(f"{validation_error_count} ungeprüft (API-Fehler)")
            if pruned_general_count:
                parts.append(f"{pruned_general_count} allgemeine Elternknoten verworfen")
            if artifact_count:
                parts.append(f"{artifact_count} Artefakte verworfen")
            stream_callback(
                f"🔎 RVK-Prüfung Katalog: {', '.join(parts)}\n",
                "dk_search",
            )

        return cleaned_results

    def _inject_rvk_api_fallback(
        self,
        final_search_keywords: List[str],
        dk_search_results: List[Dict[str, Any]],
        gnd_keyword_entries: Optional[List[Dict[str, str]]] = None,
        stream_callback: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Add authority-backed RVK candidates if no standard RVK survived validation."""
        existing_standard_rvk = False
        covered_standard_keywords = set()
        for kw_result in dk_search_results:
            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", ""))).upper()
                rvk_status = classification.get("rvk_validation_status")
                if cls_type == "RVK" and rvk_status == "standard":
                    existing_standard_rvk = True
                    matched_keywords = classification.get("matched_keywords", []) or classification.get("keywords", []) or []
                    if not matched_keywords and kw_result.get("keyword"):
                        matched_keywords = [kw_result.get("keyword")]
                    for keyword in matched_keywords:
                        normalized = canonicalize_keyword(keyword)
                        if normalized:
                            covered_standard_keywords.add(normalized)
                    break

        fallback_keywords = deduplicate_canonical_keywords(final_search_keywords or [])
        uncovered_fallback_keywords = [
            keyword for keyword in fallback_keywords
            if canonicalize_keyword(keyword) not in covered_standard_keywords
        ]

        if existing_standard_rvk and not uncovered_fallback_keywords:
            return dk_search_results

        if gnd_keyword_entries:
            use_gnd_index = not existing_standard_rvk
            gnd_entries_for_fallback = list(gnd_keyword_entries or [])
            if use_gnd_index and stream_callback:
                stream_callback(
                    "⚠️ Katalog lieferte keine standardisierten RVK-Kandidaten - versuche RVK-GND-Index\n",
                    "dk_search"
                )
            if use_gnd_index:
                rvk_index_results = self._build_rvk_gnd_index_results(
                    gnd_entries_for_fallback,
                    stream_callback=stream_callback,
                )
                if rvk_index_results:
                    if self.logger:
                        self.logger.info(
                            f"RVK GND index added {sum(len(item.get('classifications', [])) for item in rvk_index_results)} "
                            f"candidates across {len(rvk_index_results)} keywords"
                        )
                    dk_search_results = dk_search_results + rvk_index_results
                    return dk_search_results

        if existing_standard_rvk and uncovered_fallback_keywords and stream_callback:
            preview = ", ".join(uncovered_fallback_keywords[:6])
            if len(uncovered_fallback_keywords) > 6:
                preview += f", +{len(uncovered_fallback_keywords) - 6} weitere"
            stream_callback(
                f"ℹ️ Ergänze RVK-API-Fallback für nicht abgedeckte Anker/Promotionen: {preview}\n",
                "dk_search"
            )
        elif stream_callback:
            stream_callback(
                "⚠️ RVK-GND-Index lieferte nichts - nutze offiziellen RVK-API-Label-Fallback\n",
                "dk_search"
            )

        rvk_api_results = self._build_rvk_api_fallback_results(
            uncovered_fallback_keywords or fallback_keywords,
            stream_callback=stream_callback,
        )
        if not rvk_api_results:
            if stream_callback:
                stream_callback("  ⚠️ RVK-API-Fallback lieferte keine geeigneten Kandidaten\n", "dk_search")
            return dk_search_results

        if self.logger:
            self.logger.info(
                f"RVK API fallback added {sum(len(item.get('classifications', [])) for item in rvk_api_results)} "
                f"candidates across {len(rvk_api_results)} keywords"
            )
        return dk_search_results + rvk_api_results

    def _filter_final_rvk_classifications(
        self,
        classifications: List[str],
        allowed_standard_rvk_map: Dict[str, str],
        allowed_nonstandard_rvk_map: Dict[str, str],
        stream_callback: Optional[callable] = None,
    ) -> List[str]:
        """Prevent free-form RVK inference while allowing local RVK only as a fallback."""
        filtered = []
        dropped = []
        allowed_map = allowed_standard_rvk_map or allowed_nonstandard_rvk_map

        for code in classifications:
            clean = str(code or "").strip()
            if not clean:
                continue

            if clean.upper().startswith("RVK "):
                normalized = canonicalize_rvk_notation(clean[4:].strip())
                canonical = allowed_map.get(normalized)
                if canonical:
                    filtered.append(canonical)
                else:
                    dropped.append(clean)
                continue

            filtered.append(clean)

        deduplicated = list(dict.fromkeys(filtered))

        if dropped:
            if self.logger:
                self.logger.warning(f"Dropped {len(dropped)} non-authoritative RVK classifications: {dropped}")
            if stream_callback:
                preview = ", ".join(dropped[:4])
                if len(dropped) > 4:
                    preview += f", +{len(dropped) - 4} weitere"
                stream_callback(
                    f"⚠️ Verwerfe nicht-autorisierte RVK-Ausgaben des LLM: {preview}\n",
                    "dk_classification"
                )

        return deduplicated

    def _emit_rvk_source_diagnostics(
        self,
        keyword_results: List[Dict[str, Any]],
        stream_callback: Optional[callable] = None,
        step_id: str = "dk_search",
    ) -> None:
        """Emit a compact per-run summary of RVK candidate provenance."""
        if not stream_callback:
            return

        buckets: Dict[str, set] = {
            "catalog_standard": set(),
            "catalog_nonstandard": set(),
            "catalog_validation_error": set(),
            "gnd_index": set(),
            "rvk_api": set(),
        }

        for kw_result in keyword_results:
            for classification in kw_result.get("classifications", []):
                cls_type = str(classification.get("classification_type", classification.get("type", ""))).upper()
                if cls_type != "RVK":
                    continue
                normalized = canonicalize_rvk_notation(classification.get("dk", ""))
                if not normalized:
                    continue
                source = classification.get("source", "")
                status = classification.get("rvk_validation_status")
                if source == "rvk_gnd_index":
                    buckets["gnd_index"].add(normalized)
                elif source == "rvk_api":
                    buckets["rvk_api"].add(normalized)
                elif status == "standard":
                    buckets["catalog_standard"].add(normalized)
                elif status == "validation_error":
                    buckets["catalog_validation_error"].add(normalized)
                elif status == "non_standard":
                    buckets["catalog_nonstandard"].add(normalized)

        parts = []
        if buckets["catalog_standard"]:
            parts.append(f"Katalog standard {len(buckets['catalog_standard'])}")
        if buckets["catalog_nonstandard"]:
            parts.append(f"Katalog lokal {len(buckets['catalog_nonstandard'])}")
        if buckets["catalog_validation_error"]:
            parts.append(f"Katalog ungeprüft {len(buckets['catalog_validation_error'])}")
        if buckets["gnd_index"]:
            parts.append(f"RVK-GND-Index {len(buckets['gnd_index'])}")
        if buckets["rvk_api"]:
            parts.append(f"RVK-API-Label {len(buckets['rvk_api'])}")

        if parts:
            stream_callback(f"ℹ️ RVK-Quellen: {', '.join(parts)}\n", step_id)

    @staticmethod
    def _rvk_significant_tokens(text: str) -> List[str]:
        """Extract simple content-bearing tokens for deterministic RVK ranking."""
        stopwords = {
            "und", "oder", "der", "die", "das", "des", "dem", "den", "ein", "eine", "einer",
            "eines", "im", "in", "am", "an", "auf", "mit", "ohne", "von", "vom", "zum", "zur",
            "fur", "fuer", "uber", "ueber", "unter", "zwischen", "nach", "vor", "bei", "aus",
            "zu", "ist", "sind", "war", "werden", "wird", "als", "auch", "nicht", "kein",
            "keine", "sehr", "mehr", "weniger", "durch", "gegen", "seit", "bis", "eines",
            "einem", "einen", "dieser", "diese", "dieses", "jene", "jener", "jenes",
            "text", "analyse", "geschichte",  # generic high-frequency tokens contribute little
        }
        normalized = canonicalize_keyword(str(text or "")).casefold()
        raw_tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]{4,}", normalized)
        return [token for token in raw_tokens if token not in stopwords]

    def _derive_rvk_anchor_keywords_heuristic(
        self,
        verified_keywords: List[str],
        llm_analysis: Optional["LlmKeywordAnalysis"] = None,
        max_anchors: int = 8,
    ) -> List[str]:
        """Derive a small thematic GND subset to drive RVK lookup and ranking."""
        if not verified_keywords:
            return []

        analysis_text = ""
        missing_concepts: List[str] = []
        keyword_chains: List[Dict[str, Any]] = []

        if llm_analysis:
            analysis_text = str(getattr(llm_analysis, "analyse_text", "") or "")
            response_text = str(getattr(llm_analysis, "response_full_text", "") or "")
            if not analysis_text and response_text:
                from ..core.processing_utils import extract_analyse_text_from_response
                analysis_text = extract_analyse_text_from_response(response_text) or ""
            missing_concepts = list(getattr(llm_analysis, "missing_concepts", []) or [])
            if not missing_concepts and response_text:
                from ..core.processing_utils import extract_missing_concepts_from_response
                missing_concepts = extract_missing_concepts_from_response(response_text)
            if response_text:
                from ..core.processing_utils import extract_keyword_chains_from_response
                keyword_chains = extract_keyword_chains_from_response(response_text)

        thematic_fragments = [analysis_text] + list(missing_concepts)
        for chain in keyword_chains:
            thematic_fragments.extend(chain.get("chain", []) or [])
            thematic_fragments.append(str(chain.get("reason", "") or ""))
        thematic_text = " ".join(fragment for fragment in thematic_fragments if fragment)
        thematic_tokens = set(self._rvk_significant_tokens(thematic_text))

        institutional_terms = {
            "bibliothek", "bibliotheken", "zeitung", "fernsehen", "massenmedien",
            "kommentar", "alltag", "rezeption", "stadt", "bild", "sohn",
        }

        scored_keywords = []
        for keyword in verified_keywords:
            clean_keyword = keyword.split("(GND-ID:")[0].strip()
            keyword_lower = canonicalize_keyword(clean_keyword).casefold()
            keyword_tokens = set(self._rvk_significant_tokens(clean_keyword))
            score = 0

            if analysis_text and keyword_lower and keyword_lower in canonicalize_keyword(analysis_text).casefold():
                score += 45

            for concept in missing_concepts:
                concept_lower = canonicalize_keyword(concept).casefold()
                if not concept_lower:
                    continue
                if keyword_lower == concept_lower:
                    score += 30
                elif keyword_lower and (keyword_lower in concept_lower or concept_lower in keyword_lower):
                    score += 18

            for chain in keyword_chains:
                chain_terms = [canonicalize_keyword(item).casefold() for item in (chain.get("chain", []) or [])]
                if keyword_lower and keyword_lower in chain_terms:
                    score += 26
                reason_text = canonicalize_keyword(str(chain.get("reason", "") or "")).casefold()
                if keyword_lower and keyword_lower in reason_text:
                    score += 10

            token_overlap = len(keyword_tokens.intersection(thematic_tokens))
            score += token_overlap * 9

            if keyword_tokens and keyword_tokens.issubset(institutional_terms) and score < 45:
                score -= 12

            scored_keywords.append((score, clean_keyword.casefold(), keyword))

        scored_keywords.sort(key=lambda item: (-item[0], item[1]))
        selected = [keyword for score, _, keyword in scored_keywords if score > 0][:max_anchors]

        if not selected:
            selected = verified_keywords[: min(max_anchors, len(verified_keywords))]

        return selected

    def _derive_promoted_rvk_terms(
        self,
        initial_keywords: Optional[List[str]] = None,
        search_results: Optional[Any] = None,
        llm_analysis: Optional["LlmKeywordAnalysis"] = None,
        max_terms: int = 3,
    ) -> Tuple[List[str], List[str]]:
        """Promote search-backed initial concepts that reappear as missing core concepts."""
        if not initial_keywords or not llm_analysis:
            return [], []

        if isinstance(initial_keywords, str):
            initial_keywords = [
                item.strip()
                for item in re.split(r"[\n,]+", initial_keywords)
                if item.strip()
            ]
        else:
            initial_keywords = [
                str(item).strip()
                for item in list(initial_keywords)
                if str(item).strip()
            ]

        if not initial_keywords:
            return [], []

        response_text = str(getattr(llm_analysis, "response_full_text", "") or "")
        missing_concepts = list(getattr(llm_analysis, "missing_concepts", []) or [])
        if not missing_concepts and response_text:
            from ..core.processing_utils import extract_missing_concepts_from_response
            missing_concepts = extract_missing_concepts_from_response(response_text)
        if not missing_concepts:
            return [], []

        search_support: Dict[str, bool] = {}
        observed_search_terms = set()
        if isinstance(search_results, dict):
            for term, results in search_results.items():
                normalized = canonicalize_keyword(term)
                if not normalized:
                    continue
                observed_search_terms.add(normalized)
                search_support[normalized] = bool(results)
        elif isinstance(search_results, list):
            for item in search_results:
                term = getattr(item, "search_term", None) or (item.get("search_term") if isinstance(item, dict) else "")
                results = getattr(item, "results", None) or (item.get("results") if isinstance(item, dict) else None)
                normalized = canonicalize_keyword(term)
                if normalized:
                    observed_search_terms.add(normalized)
                    search_support[normalized] = bool(results)

        promoted: List[str] = []
        seen = set()
        diagnostics: List[str] = []
        missing_canonical = [canonicalize_keyword(item) for item in missing_concepts if canonicalize_keyword(item)]

        for keyword in initial_keywords:
            clean_keyword = str(keyword or "").strip()
            normalized_keyword = canonicalize_keyword(clean_keyword)
            if not normalized_keyword:
                diagnostics.append(f"{clean_keyword or '(leer)'} -> ignoriert (nicht normalisierbar)")
                continue

            has_search_support = search_support.get(normalized_keyword)
            if not has_search_support and normalized_keyword not in observed_search_terms:
                diagnostics.append(f"{clean_keyword} -> verworfen (keine Suchunterstützung)")
                continue

            matched_concept = None
            for concept in missing_canonical:
                if (
                    normalized_keyword == concept
                    or normalized_keyword in concept
                    or concept in normalized_keyword
                ):
                    matched_concept = concept
                    if clean_keyword not in seen:
                        promoted.append(clean_keyword)
                        seen.add(clean_keyword)
                    break
            if matched_concept:
                diagnostics.append(f"{clean_keyword} -> gefördert (passt zu '{matched_concept}')")
            else:
                diagnostics.append(f"{clean_keyword} -> verworfen (kein Missing-Concept-Match)")
            if len(promoted) >= max_terms:
                break

        return promoted, diagnostics

    def _derive_rvk_anchor_keywords(
        self,
        verified_keywords: List[str],
        llm_analysis: Optional["LlmKeywordAnalysis"] = None,
        original_abstract: str = "",
        initial_keywords: Optional[List[str]] = None,
        search_results: Optional[Any] = None,
        max_anchors: int = 8,
        stream_callback: Optional[callable] = None,
        ) -> List[str]:
        """Derive RVK anchors with an LLM-first selection and heuristic fallback."""
        promoted_terms, promotion_diagnostics = self._derive_promoted_rvk_terms(
            initial_keywords=initial_keywords,
            search_results=search_results,
            llm_analysis=llm_analysis,
        )

        def _merge_promoted(selected_terms: List[str]) -> List[str]:
            merged = list(selected_terms or [])
            for term in promoted_terms:
                if term not in merged:
                    merged.append(term)
            return merged

        def _emit_promotion_log() -> None:
            if not stream_callback:
                return
            if promoted_terms:
                preview = ", ".join(promoted_terms[:6])
                stream_callback(
                    f"ℹ️ RVK-Promotion aus Initialbegriffen: {preview}\n",
                    "dk_search",
                )
                return
            stream_callback(
                "ℹ️ RVK-Promotion aus Initialbegriffen: keine passenden Kandidaten\n",
                "dk_search",
            )
            if promotion_diagnostics:
                preview = "; ".join(promotion_diagnostics[:4])
                if len(promotion_diagnostics) > 4:
                    preview += f"; +{len(promotion_diagnostics) - 4} weitere"
                stream_callback(
                    f"  ↳ {preview}\n",
                    "dk_search",
                )

        if not verified_keywords:
            _emit_promotion_log()
            return promoted_terms

        heuristic_selected = self._derive_rvk_anchor_keywords_heuristic(
            verified_keywords,
            llm_analysis=llm_analysis,
            max_anchors=max_anchors,
        )

        if not self.alima_manager:
            selected = _merge_promoted(heuristic_selected)
            _emit_promotion_log()
            return selected

        analysis_fragments = []
        if original_abstract:
            analysis_fragments.append(str(original_abstract))

        if llm_analysis:
            analysis_text = str(getattr(llm_analysis, "analyse_text", "") or "")
            if analysis_text:
                analysis_fragments.append(f"\nThematische Analyse:\n{analysis_text}")
            missing_concepts = list(getattr(llm_analysis, "missing_concepts", []) or [])
            if missing_concepts:
                analysis_fragments.append(
                    "\nFehlende Konzepte:\n" + ", ".join(str(item) for item in missing_concepts if str(item).strip())
                )

        abstract_for_selection = "\n".join(fragment for fragment in analysis_fragments if fragment).strip()
        if not abstract_for_selection:
            selected = _merge_promoted(heuristic_selected)
            _emit_promotion_log()
            return selected

        from ..core.data_models import AbstractData
        from ..core.json_response_parser import parse_json_response

        keyword_lines = "\n".join(verified_keywords)
        abstract_data = AbstractData(
            abstract=abstract_for_selection,
            keywords=keyword_lines,
        )

        provider = getattr(llm_analysis, "provider_used", None) if llm_analysis else None
        model = getattr(llm_analysis, "model_used", None) if llm_analysis else None

        try:
            task_state = self.alima_manager.analyze_abstract(
                abstract_data=abstract_data,
                task="rvk_anchor_selection",
                model=model,
                provider=provider,
                stream_callback=None,
            )
            if task_state.status == "failed":
                selected = _merge_promoted(heuristic_selected)
                _emit_promotion_log()
                return selected

            parsed = parse_json_response(task_state.analysis_result.full_text) or {}
            anchors = parsed.get("anchors", [])
            if not isinstance(anchors, list):
                selected = _merge_promoted(heuristic_selected)
                _emit_promotion_log()
                return selected

            keyword_by_gnd: Dict[str, str] = {}
            keyword_by_text: Dict[str, str] = {}
            for keyword in verified_keywords:
                clean_keyword = keyword.split("(GND-ID:")[0].strip()
                clean_key = canonicalize_keyword(clean_keyword)
                if clean_key:
                    keyword_by_text[clean_key] = keyword
                gnd_id = extract_gnd_id(keyword)
                if gnd_id:
                    keyword_by_gnd[gnd_id] = keyword

            selected = []
            seen = set()
            for item in anchors:
                if not isinstance(item, dict):
                    continue
                gnd_id = str(item.get("gnd_id", "") or "").strip()
                keyword_text = canonicalize_keyword(str(item.get("keyword", "") or "").strip())
                matched = None
                if gnd_id and gnd_id in keyword_by_gnd:
                    matched = keyword_by_gnd[gnd_id]
                elif keyword_text and keyword_text in keyword_by_text:
                    matched = keyword_by_text[keyword_text]
                if matched and matched not in seen:
                    selected.append(matched)
                    seen.add(matched)
                if len(selected) >= max_anchors:
                    break

            if selected:
                selected = _merge_promoted(selected)
                if stream_callback:
                    preview = ", ".join(keyword.split("(GND-ID:")[0].strip() for keyword in selected[:6])
                    if len(selected) > 6:
                        preview += f", +{len(selected) - 6} weitere"
                    stream_callback(
                        f"ℹ️ RVK-Anker (LLM): {preview}\n",
                        "dk_search",
                    )
                    _emit_promotion_log()
                return selected
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"LLM RVK anchor selection failed, falling back to heuristic: {exc}")

        selected = _merge_promoted(heuristic_selected)
        if stream_callback and selected:
            preview = ", ".join(keyword.split("(GND-ID:")[0].strip() for keyword in selected[:6])
            if len(selected) > 6:
                preview += f", +{len(selected) - 6} weitere"
            stream_callback(
                f"ℹ️ RVK-Anker (heuristisch): {preview}\n",
                "dk_search",
            )
            _emit_promotion_log()
        return selected

    def _rvk_domain_profile(self, abstract_text: str, matched_keywords: List[str]) -> Dict[str, int]:
        """Estimate thematic domain strength from the abstract and matched keywords."""
        combined_tokens = self._rvk_significant_tokens(
            " ".join([str(abstract_text or "")] + [str(item) for item in matched_keywords])
        )
        counts: Dict[str, int] = {}
        domain_terms = {
            "politics": {
                "politik", "politisch", "demokratie", "demokratisierung", "dissident",
                "kommunismus", "sozialismus", "totalitarismus", "verfolgung",
                "menschenrecht", "menschenrechte", "menschenrechtspolitik", "staat",
                "ost", "west", "konflikt", "krieg", "nato", "osze", "helsinki",
                "sicherheitspolitik", "weltpolitik", "international", "völkerrecht", "voelkerrecht",
            },
            "history": {
                "geschichte", "historisch", "osteuropa", "sowjetunion", "prager",
                "fruhling", "fruehling", "tschechoslowakei", "slowakei", "russisch",
                "ukrainisch", "kalter", "krieg", "biografie", "autobiografie",
            },
            "law": {
                "recht", "rechte", "grundrecht", "freiheitsrecht", "vertrag",
                "völkerrecht", "voelkerrecht", "konvention", "menschenrecht",
            },
            "media": {
                "zeitung", "fernsehen", "massenmedien", "propaganda", "offentliche", "oeffentliche",
            },
            "literature": {
                "literatur", "schriftsteller", "roman", "erzahlung", "erzaehlung",
                "autobiografie", "biografie",
            },
            "religion": {
                "christlich", "kirche", "theologie", "religion", "ethik",
            },
            "philosophy": {
                "philosophie", "ethik", "denken", "theorie",
            },
        }

        for domain, terms in domain_terms.items():
            counts[domain] = sum(1 for token in combined_tokens if token in terms)
        return counts

    def _rvk_branch_fit_score(
        self,
        candidate: Dict[str, Any],
        abstract_text: str,
        matched_keywords: List[str],
    ) -> int:
        """Score how well the candidate branch fits the text domain."""
        branch_text = " ".join(
            [
                str(candidate.get("label", "") or ""),
                str(candidate.get("ancestor_path", "") or ""),
                " ".join(str(item) for item in (candidate.get("register") or [])),
            ]
        )
        branch_tokens = set(self._rvk_significant_tokens(branch_text))
        domain_profile = self._rvk_domain_profile(abstract_text, matched_keywords)

        fit_score = 0
        domain_branch_terms = {
            "politics": {"politik", "politische", "internationale", "demokratie", "staat", "regierung", "konflikt"},
            "history": {"geschichte", "historische", "osteuropa", "sowjetunion", "zeitgeschichte"},
            "law": {"recht", "rechte", "vertrag", "völkerrecht", "voelkerrecht", "menschenrechte"},
            "media": {"medien", "presse", "kommunikation", "propaganda", "fernsehen"},
            "literature": {"literatur", "schriftsteller", "autobiograph", "biograph"},
            "religion": {"christliche", "theologie", "religion", "kirche"},
            "philosophy": {"philosophie", "ethik", "theorie"},
        }

        for domain, strength in domain_profile.items():
            if strength <= 0:
                continue
            hits = len(branch_tokens.intersection(domain_branch_terms.get(domain, set())))
            if hits:
                fit_score += min(strength, 4) * hits * 10

        # Penalize clearly mismatched major domains when the text strongly points elsewhere.
        if domain_profile.get("politics", 0) + domain_profile.get("history", 0) >= 3:
            if branch_tokens.intersection(domain_branch_terms["religion"]):
                fit_score -= 45
            if branch_tokens.intersection(domain_branch_terms["literature"]) and domain_profile.get("literature", 0) == 0:
                fit_score -= 20

        if domain_profile.get("law", 0) >= 2 and branch_tokens.intersection(domain_branch_terms["law"]):
            fit_score += 15

        if self._is_institution_library_rvk(candidate):
            if self._matches_specific_library_context(candidate, abstract_text, matched_keywords):
                fit_score += 10
            else:
                fit_score -= 120

        return fit_score

    @staticmethod
    def _is_institution_library_rvk(candidate: Dict[str, Any]) -> bool:
        """Detect RVK notations for single named libraries that are often catalog artifacts."""
        label = str(candidate.get("label", "") or "").lower()
        ancestor_path = str(candidate.get("ancestor_path", "") or "").lower()
        branch_text = f"{label} {ancestor_path}"
        return (
            "bibliothekswesen" in branch_text
            and (
                "einzelne bibliotheken" in branch_text
                or "einzelne deutsche bibliotheken" in branch_text
                or "bibliotheken d" in branch_text
            )
        )

    @staticmethod
    def _matches_specific_library_context(
        candidate: Dict[str, Any],
        abstract_text: str,
        matched_keywords: List[str],
    ) -> bool:
        """Keep single-library RVK only when the specific institution/location is really in the text."""
        context_text = " ".join(
            [
                str(abstract_text or "").lower(),
                " ".join(str(item or "").lower() for item in (matched_keywords or [])),
            ]
        )
        label = str(candidate.get("label", "") or "")
        distinctive_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", label)
            if token.lower() not in {
                "bibliothek",
                "bibliotheken",
                "landesbibliothek",
                "staats",
                "universitätsbibliothek",
                "universitaetsbibliothek",
                "sowie",
                "deutsche",
            }
        ]
        return any(token in context_text for token in distinctive_tokens)

    @staticmethod
    def _rvk_broadness_penalty(candidate: Dict[str, Any]) -> int:
        """Penalize overly broad or shallow RVK nodes."""
        ancestor_path = str(candidate.get("ancestor_path", "") or "")
        label = str(candidate.get("label", "") or "")
        register_entries = [str(item) for item in (candidate.get("register") or []) if str(item).strip()]
        notation = str(candidate.get("dk", "") or "")

        depth = len([part for part in ancestor_path.split(">") if part.strip()])
        label_tokens = re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", label)
        notation_core = re.sub(r"[^A-Z0-9]", "", notation)

        penalty = 0
        if depth <= 1:
            penalty += 60
        elif depth == 2:
            penalty += 35
        elif depth == 3:
            penalty += 15

        if len(label_tokens) <= 1 and len(register_entries) <= 1:
            penalty += 18
        if len(notation_core) <= 4:
            penalty += 12

        return penalty

    def _score_rvk_candidate(
        self,
        candidate: Dict[str, Any],
        original_abstract: str,
        rvk_anchor_keywords: Optional[List[str]] = None,
    ) -> int:
        """Deterministically score validated RVK candidates."""
        source = candidate.get("source", "catalog")
        status = candidate.get("rvk_validation_status", "standard")
        matched_keywords = [
            canonicalize_keyword(keyword).lower()
            for keyword in (candidate.get("matched_keywords") or [])
            if canonicalize_keyword(keyword)
        ]
        anchor_keywords = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip()).lower()
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }
        label = str(candidate.get("label", "") or "")
        ancestor_path = str(candidate.get("ancestor_path", "") or "")
        register_entries = [str(item) for item in (candidate.get("register") or []) if str(item).strip()]
        titles = [str(item) for item in (candidate.get("titles") or []) if str(item).strip()]
        haystack = " ".join([label, ancestor_path, " ".join(register_entries)]).lower()
        abstract_text = str(original_abstract or "").lower()

        source_weight = {
            "rvk_gnd_index": 120,
            "rvk_api": 100,
            "catalog": 80,
        }.get(source, 60)
        if status == "non_standard":
            source_weight -= 40
        elif status == "validation_error":
            source_weight -= 60

        overlap_score = 0
        for keyword in matched_keywords:
            if keyword and keyword in haystack:
                overlap_score += 12
            elif keyword and keyword in abstract_text:
                overlap_score += 3

        specificity = len(re.sub(r"[^A-Z0-9]", "", str(candidate.get("dk", "")))) * 2
        count_score = min(int(candidate.get("count", 0) or 0), 24) * 2
        keyword_support = len(set(matched_keywords)) * 12
        title_support = min(len(titles), 6)
        register_support = min(len(register_entries), 8)
        branch_fit = self._rvk_branch_fit_score(candidate, abstract_text, matched_keywords)
        broadness_penalty = self._rvk_broadness_penalty(candidate)
        anchor_match_bonus = 0
        if anchor_keywords:
            anchor_hits = len(anchor_keywords.intersection(set(matched_keywords)))
            if anchor_hits:
                anchor_match_bonus += anchor_hits * 24
            else:
                anchor_match_bonus -= 30

        return (
            source_weight
            + overlap_score
            + specificity
            + count_score
            + keyword_support
            + title_support
            + register_support
            + branch_fit
            + anchor_match_bonus
            - broadness_penalty
        )

    def _score_rvk_shortlist_with_llm(
        self,
        shortlist: List[Dict[str, Any]],
        original_abstract: str,
        model: Optional[str],
        provider: Optional[str],
        stream_callback: Optional[callable] = None,
        mode=None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Best-effort LLM scoring for a fixed RVK shortlist."""
        if not shortlist or not self.alima_manager:
            return {}

        from ..core.data_models import AbstractData
        from ..core.json_response_parser import parse_json_response

        shortlist_text = PipelineResultFormatter.format_dk_results_for_prompt(
            shortlist,
            max_results=len(shortlist),
        )
        abstract_data = AbstractData(
            abstract=original_abstract,
            keywords=shortlist_text,
        )

        try:
            task_state = self.alima_manager.analyze_abstract(
                abstract_data=abstract_data,
                task="rvk_scoring",
                model=model,
                provider=provider,
                stream_callback=None,
                mode=mode,
                **(llm_kwargs or {}),
            )
            if task_state.status == "failed":
                return {}

            parsed = parse_json_response(task_state.analysis_result.full_text) or {}
            scores = parsed.get("scores", [])
            results: Dict[str, Dict[str, Any]] = {}
            for item in scores:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code", "")).strip()
                normalized = canonicalize_rvk_notation(code[4:].strip() if code.upper().startswith("RVK ") else code)
                if not normalized:
                    continue
                thematic_fit = int(item.get("thematic_fit", 0) or 0)
                branch_fit = int(item.get("branch_fit", 0) or 0)
                specificity = int(item.get("specificity", 0) or 0)
                total = int(item.get("total_score", 0) or 0)
                if not total:
                    total = thematic_fit * 4 + branch_fit * 4 + specificity * 2
                results[normalized] = {
                    "thematic_fit": thematic_fit,
                    "branch_fit": branch_fit,
                    "specificity": specificity,
                    "total_score": total,
                    "reason": str(item.get("reason", "") or ""),
                }
            if stream_callback and results:
                stream_callback(
                    f"ℹ️ RVK-Shortlist per LLM bewertet: {len(results)} Kandidaten\n",
                    "dk_classification",
                )
            return results
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"RVK shortlist scoring failed: {exc}")
            if stream_callback:
                stream_callback(
                    f"⚠️ RVK-Shortlist-Scoring fehlgeschlagen, nutze Heuristik: {str(exc)}\n",
                    "dk_classification",
                )
            return {}

    def _build_dk_semantic_profile(
        self,
        selected_dk_codes: List[str],
        candidate_results: List[Dict[str, Any]],
        max_keywords: int = 6,
        max_titles: int = 3,
        max_codes: int = 6,
    ) -> str:
        """Build compact semantic hints from the selected DK classes."""
        if not selected_dk_codes or not candidate_results:
            return ""

        aggregated: Dict[str, Dict[str, Any]] = {}
        for candidate in candidate_results:
            # DK and DDC both contribute thematic hints; RVK is excluded
            # (it is what we are ranking). - Claude Generated
            cls_type = str(candidate.get("classification_type", candidate.get("type", "DK"))).upper()
            if cls_type not in ("DK", "DDC"):
                continue

            raw_code = str(candidate.get("dk", "") or "").strip()
            if not raw_code:
                continue

            prefix = "DDC" if cls_type == "DDC" else "DK"
            key = f"{prefix} {raw_code}"
            current = aggregated.setdefault(
                key,
                {
                    "matched_keywords": [],
                    "titles": [],
                    "count": 0,
                },
            )
            current["count"] += int(candidate.get("count", 0) or 0)

            seen_keywords = set(current["matched_keywords"])
            for keyword in (candidate.get("matched_keywords") or candidate.get("keywords") or []):
                clean_keyword = str(keyword or "").strip()
                if clean_keyword and clean_keyword not in seen_keywords:
                    current["matched_keywords"].append(clean_keyword)
                    seen_keywords.add(clean_keyword)

            seen_titles = set(current["titles"])
            for title in (candidate.get("titles") or []):
                clean_title = str(title or "").strip()
                if clean_title and clean_title not in seen_titles:
                    current["titles"].append(clean_title)
                    seen_titles.add(clean_title)

        lines = []
        for code in selected_dk_codes:
            clean_code = str(code or "").strip()
            if not clean_code:
                continue
            upper = clean_code.upper()
            if upper.startswith("RVK "):
                continue

            # Normalise the prefix (uppercase) so DK/DDC codes match the
            # aggregated keys; bare codes default to DK. - Claude Generated
            if upper.startswith("DK "):
                normalized = "DK " + clean_code[3:].strip()
            elif upper.startswith("DDC "):
                normalized = "DDC " + clean_code[4:].strip()
            else:
                normalized = f"DK {clean_code}"
            data = aggregated.get(normalized)
            if not data:
                continue

            keywords = ", ".join(
                cleaned
                for cleaned in (repair_display_text(item) for item in data.get("matched_keywords", [])[:max_keywords])
                if cleaned
            )
            titles = " | ".join(
                cleaned
                for cleaned in (repair_display_text(item) for item in data.get("titles", [])[:max_titles])
                if cleaned
            )
            parts = [normalized]
            if keywords:
                parts.append(f"Schlagworte: {keywords}")
            if titles:
                parts.append(f"Beispieltitel: {titles}")
            if data.get("count"):
                parts.append(f"Haeufigkeit: {int(data['count'])}")
            lines.append(" | ".join(parts))
            if len(lines) >= max_codes:
                break

        return "\n".join(lines)

    def _build_rvk_scoring_shortlist(
        self,
        candidate_results: List[Dict[str, Any]],
        original_abstract: str,
        rvk_anchor_keywords: Optional[List[str]] = None,
        max_standard: int = 8,
        max_nonstandard: int = 3,
    ) -> List[Dict[str, Any]]:
        """Build a compact validated RVK shortlist for the second-pass scorer."""
        aggregated: Dict[str, Dict[str, Any]] = {}
        anchor_keywords = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip()).lower()
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }

        for candidate in candidate_results:
            cls_type = str(candidate.get("classification_type", candidate.get("type", "DK"))).upper()
            if cls_type != "RVK":
                continue

            normalized = canonicalize_rvk_notation(candidate.get("dk", ""))
            if not normalized:
                continue

            current = aggregated.get(normalized)
            if current is None:
                current = dict(candidate)
                current["dk"] = normalized
                current["matched_keywords"] = list(candidate.get("matched_keywords", []) or candidate.get("keywords", []) or [])
                current["titles"] = list(candidate.get("titles", []) or [])
                current["register"] = list(candidate.get("register", []) or [])
                current["count"] = int(candidate.get("count", 0) or 0)
                aggregated[normalized] = current
                continue

            current["count"] = int(current.get("count", 0) or 0) + int(candidate.get("count", 0) or 0)
            for field in ("matched_keywords", "titles", "register"):
                existing = list(current.get(field, []) or [])
                seen = set(existing)
                incoming = candidate.get(field, []) or candidate.get("keywords", []) or []
                for value in incoming:
                    clean_value = str(value or "").strip()
                    if clean_value and clean_value not in seen:
                        existing.append(clean_value)
                        seen.add(clean_value)
                current[field] = existing

            if candidate.get("label") and not current.get("label"):
                current["label"] = candidate.get("label")
            if candidate.get("ancestor_path") and not current.get("ancestor_path"):
                current["ancestor_path"] = candidate.get("ancestor_path")
            if candidate.get("branch_family") and not current.get("branch_family"):
                current["branch_family"] = candidate.get("branch_family")

            current_source = str(current.get("source", "catalog") or "catalog")
            incoming_source = str(candidate.get("source", "catalog") or "catalog")
            current_status = str(current.get("rvk_validation_status", "standard") or "standard")
            incoming_status = str(candidate.get("rvk_validation_status", "standard") or "standard")
            replace_source = _source_rank(incoming_source) > _source_rank(current_source)
            replace_status = _status_rank(incoming_status) > _status_rank(current_status)
            if replace_status or (incoming_status == current_status and replace_source):
                current["source"] = incoming_source
                current["rvk_validation_status"] = incoming_status
                current["validation_message"] = candidate.get("validation_message", current.get("validation_message", ""))

        standard_candidates = []
        nonstandard_candidates = []
        for candidate in aggregated.values():
            candidate["_score"] = self._score_rvk_candidate(
                candidate,
                original_abstract,
                rvk_anchor_keywords=rvk_anchor_keywords,
            )
            matched_keyword_set = {
                canonicalize_keyword(keyword).lower()
                for keyword in (candidate.get("matched_keywords") or [])
                if canonicalize_keyword(keyword)
            }
            candidate["_anchor_hit_count"] = len(anchor_keywords.intersection(matched_keyword_set))
            candidate["_source_rank"] = _source_rank(str(candidate.get("source", "catalog") or "catalog"))
            candidate["_status_rank"] = _status_rank(str(candidate.get("rvk_validation_status", "standard") or "standard"))
            status = str(candidate.get("rvk_validation_status", "standard") or "standard")
            if status == "standard":
                standard_candidates.append(candidate)
            elif status in {"non_standard", "validation_error"}:
                nonstandard_candidates.append(candidate)

        def _sort_key(item: Dict[str, Any]):
            return (
                -int(item.get("_anchor_hit_count", 0)),
                -int(item.get("_source_rank", 0)),
                -int(item.get("_status_rank", 0)),
                -int(item.get("_score", 0)),
                item.get("dk", ""),
            )

        standard_candidates.sort(key=_sort_key)
        nonstandard_candidates.sort(key=_sort_key)

        shortlist = standard_candidates[:max_standard]
        if not shortlist:
            shortlist = nonstandard_candidates[:max_nonstandard]
        elif len(shortlist) < max_standard:
            remaining = max_nonstandard
            for candidate in nonstandard_candidates:
                if remaining <= 0:
                    break
                shortlist.append(candidate)
                remaining -= 1

        return shortlist

    def _select_final_rvk_with_dk_context(
        self,
        candidate_results: List[Dict[str, Any]],
        original_abstract: str,
        selected_dk_codes: List[str],
        rvk_anchor_keywords: Optional[List[str]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        mode=None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Use a DK-guided second LLM pass to rank a fixed validated RVK shortlist."""
        shortlist = self._build_rvk_scoring_shortlist(
            candidate_results,
            original_abstract,
            rvk_anchor_keywords=rvk_anchor_keywords,
        )
        if not shortlist:
            return []

        dk_profile = self._build_dk_semantic_profile(selected_dk_codes, candidate_results)
        abstract_for_scoring = str(original_abstract or "").strip()
        if dk_profile:
            abstract_for_scoring += (
                "\n\nDK-Profil aus der bereits gewaehlten DK-Seite:\n"
                f"{dk_profile}\n"
                "Nutze dieses DK-Profil als zusaetzlichen thematischen Hinweis für die RVK-Auswahl."
            )

        if stream_callback:
            stream_callback(
                f"\n\nℹ️ RVK-Zweitranking mit DK-Profil: {len(shortlist)} Kandidaten\n",
                "dk_classification",
            )
            if dk_profile:
                stream_callback(
                    "DK-Profil für RVK-Zweitranking:\n"
                    + "\n".join(f"  {line}" for line in dk_profile.splitlines() if line.strip())
                    + "\n",
                    "dk_classification",
                )
            shortlist_lines = []
            for candidate in shortlist:
                line = f"  RVK {candidate['dk']}"
                if candidate.get("label"):
                    line += f" | {candidate['label']}"
                if candidate.get("ancestor_path"):
                    line += f" | Pfad: {candidate['ancestor_path']}"
                source = str(candidate.get("source", "catalog") or "catalog")
                status = str(candidate.get("rvk_validation_status", "standard") or "standard")
                line += f" | Quelle: {source}"
                if status != "standard":
                    line += f" ({status})"
                shortlist_lines.append(line)
            if shortlist_lines:
                stream_callback(
                    "RVK-Kandidaten für DK-basiertes Zweitranking:\n"
                    + "\n".join(shortlist_lines)
                    + "\n",
                    "dk_classification",
                )

        llm_scores = self._score_rvk_shortlist_with_llm(
            shortlist,
            abstract_for_scoring,
            model=model,
            provider=provider,
            stream_callback=stream_callback,
            mode=mode,
            llm_kwargs=llm_kwargs,
        )
        if not llm_scores:
            return []

        scored_candidates = []
        for candidate in shortlist:
            score = llm_scores.get(candidate["dk"])
            if not score:
                continue
            candidate_copy = dict(candidate)
            candidate_copy["_llm_total_score"] = int(score.get("total_score", 0) or 0)
            candidate_copy["_llm_thematic_fit"] = int(score.get("thematic_fit", 0) or 0)
            candidate_copy["_llm_branch_fit"] = int(score.get("branch_fit", 0) or 0)
            candidate_copy["_llm_specificity"] = int(score.get("specificity", 0) or 0)
            candidate_copy["_llm_reason"] = str(score.get("reason", "") or "")
            scored_candidates.append(candidate_copy)

        if not scored_candidates:
            return []

        standard_candidates = [
            candidate for candidate in scored_candidates
            if str(candidate.get("rvk_validation_status", "standard") or "standard") == "standard"
        ]
        nonstandard_candidates = [
            candidate for candidate in scored_candidates
            if str(candidate.get("rvk_validation_status", "standard") or "standard") in {"non_standard", "validation_error"}
        ]

        def _sort_key(item: Dict[str, Any]):
            return (
                -int(item.get("_llm_total_score", 0)),
                -int(item.get("_llm_thematic_fit", 0)),
                -int(item.get("_llm_branch_fit", 0)),
                -int(item.get("_llm_specificity", 0)),
                -int(item.get("_anchor_hit_count", 0)),
                -int(item.get("_score", 0)),
                item.get("dk", ""),
            )

        standard_candidates.sort(key=_sort_key)
        nonstandard_candidates.sort(key=_sort_key)

        if stream_callback:
            score_lines = []
            for candidate in sorted(scored_candidates, key=_sort_key):
                line = (
                    f"  RVK {candidate['dk']}: total={int(candidate.get('_llm_total_score', 0))}, "
                    f"thematisch={int(candidate.get('_llm_thematic_fit', 0))}, "
                    f"Pfad={int(candidate.get('_llm_branch_fit', 0))}, "
                    f"Spezifitaet={int(candidate.get('_llm_specificity', 0))}"
                )
                reason = str(candidate.get("_llm_reason", "") or "").strip()
                if reason:
                    line += f" | {reason}"
                score_lines.append(line)
            if score_lines:
                stream_callback(
                    "RVK-Bewertung aus DK-basiertem Zweitranking:\n"
                    + "\n".join(score_lines)
                    + "\n",
                    "dk_classification",
                )

        selected: List[str] = []
        selected_branches = set()
        for candidate in standard_candidates:
            branch = str(candidate.get("branch_family", "") or "")
            if branch and branch in selected_branches and len(selected) >= 1:
                continue
            selected.append(f"RVK {candidate['dk']}")
            if branch:
                selected_branches.add(branch)
            if len(selected) >= 2:
                break

        if not selected and nonstandard_candidates:
            selected.append(f"RVK {nonstandard_candidates[0]['dk']}")

        if stream_callback and selected:
            stream_callback(
                "RVK-Auswahl nach DK-basiertem Zweitranking:\n"
                + "\n".join(f"  {code}" for code in selected)
                + "\n",
                "dk_classification",
            )

        return selected

    def _select_final_rvk_candidates(
        self,
        candidate_results: List[Dict[str, Any]],
        original_abstract: str,
        max_standard: int = 2,
        max_nonstandard: int = 1,
        rvk_anchor_keywords: Optional[List[str]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        stream_callback: Optional[callable] = None,
        mode=None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Select final RVK deterministically from validated candidates."""
        standard_candidates = []
        nonstandard_candidates = []
        aggregated_candidates: Dict[str, Dict[str, Any]] = {}

        anchor_keywords = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip()).lower()
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }

        for candidate in candidate_results:
            cls_type = str(candidate.get("classification_type", candidate.get("type", "DK"))).upper()
            if cls_type != "RVK":
                continue

            normalized = canonicalize_rvk_notation(candidate.get("dk", ""))
            if not normalized:
                continue

            current = aggregated_candidates.get(normalized)
            if current is None:
                current = dict(candidate)
                current["dk"] = normalized
                current["matched_keywords"] = list(candidate.get("matched_keywords", []) or [])
                current["titles"] = list(candidate.get("titles", []) or [])
                current["register"] = list(candidate.get("register", []) or [])
                current["count"] = int(candidate.get("count", 0) or 0)
                aggregated_candidates[normalized] = current
            else:
                current["count"] = int(current.get("count", 0) or 0) + int(candidate.get("count", 0) or 0)
                for field in ("matched_keywords", "titles", "register"):
                    existing_values = list(current.get(field, []) or [])
                    seen_values = set(existing_values)
                    for value in candidate.get(field, []) or []:
                        if value not in seen_values:
                            existing_values.append(value)
                            seen_values.add(value)
                    current[field] = existing_values

                if candidate.get("label") and not current.get("label"):
                    current["label"] = candidate.get("label")
                if candidate.get("ancestor_path") and not current.get("ancestor_path"):
                    current["ancestor_path"] = candidate.get("ancestor_path")
                if candidate.get("branch_family") and not current.get("branch_family"):
                    current["branch_family"] = candidate.get("branch_family")

                current_source = str(current.get("source", "catalog") or "catalog")
                incoming_source = str(candidate.get("source", "catalog") or "catalog")
                current_status = str(current.get("rvk_validation_status", "standard") or "standard")
                incoming_status = str(candidate.get("rvk_validation_status", "standard") or "standard")
                replace_source = _source_rank(incoming_source) > _source_rank(current_source)
                replace_status = _status_rank(incoming_status) > _status_rank(current_status)
                if replace_status or (incoming_status == current_status and replace_source):
                    current["source"] = incoming_source
                    current["rvk_validation_status"] = incoming_status
                    current["validation_message"] = candidate.get("validation_message", current.get("validation_message", ""))

        for enriched in aggregated_candidates.values():
            enriched["_score"] = self._score_rvk_candidate(
                enriched,
                original_abstract,
                rvk_anchor_keywords=rvk_anchor_keywords,
            )
            enriched["_branch"] = str(enriched.get("branch_family", "") or "")
            matched_keyword_set = {
                canonicalize_keyword(keyword).lower()
                for keyword in (enriched.get("matched_keywords") or [])
                if canonicalize_keyword(keyword)
            }
            enriched["_anchor_hits"] = sorted(anchor_keywords.intersection(matched_keyword_set))
            enriched["_anchor_hit_count"] = len(enriched["_anchor_hits"])
            enriched["_source_rank"] = _source_rank(str(enriched.get("source", "catalog") or "catalog"))
            enriched["_status_rank"] = _status_rank(str(enriched.get("rvk_validation_status", "standard") or "standard"))

            status = enriched.get("rvk_validation_status", "standard")
            if status == "standard":
                standard_candidates.append(enriched)
            elif status in {"non_standard", "validation_error"}:
                nonstandard_candidates.append(enriched)

        def _sort_key(item: Dict[str, Any]):
            return (
                -int(item.get("_anchor_hit_count", 0)),
                -int(item.get("_source_rank", 0)),
                -int(item.get("_status_rank", 0)),
                -int(item.get("_score", 0)),
                item.get("dk", ""),
            )

        standard_candidates.sort(key=_sort_key)
        nonstandard_candidates.sort(key=_sort_key)

        def _prefilter_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not candidates:
                return []
            if anchor_keywords:
                anchored = [candidate for candidate in candidates if candidate.get("_anchor_hit_count", 0) > 0]
                if anchored:
                    candidates = anchored
            shortlist = candidates[:8]
            llm_scores = self._score_rvk_shortlist_with_llm(
                shortlist,
                original_abstract,
                model=model,
                provider=provider,
                stream_callback=stream_callback,
                mode=mode,
                llm_kwargs=llm_kwargs,
            )
            for candidate in shortlist:
                llm_score = llm_scores.get(candidate["dk"], {})
                candidate["_llm_total_score"] = int(llm_score.get("total_score", 0) or 0)
                candidate["_llm_reason"] = str(llm_score.get("reason", "") or "")
                candidate["_combined_score"] = (
                    int(candidate.get("_score", 0))
                    + int(candidate.get("_llm_total_score", 0)) * 12
                    + int(candidate.get("_anchor_hit_count", 0)) * 18
                )
            shortlist.sort(
                key=lambda item: (
                    -int(item.get("_combined_score", 0)),
                    -int(item.get("_anchor_hit_count", 0)),
                    item.get("dk", ""),
                )
            )
            return shortlist

        def _pick_diverse(candidates: List[Dict[str, Any]], limit: int) -> List[str]:
            selected = []
            selected_candidates: List[Dict[str, Any]] = []
            covered_anchors = set()
            remaining = list(candidates)

            while remaining and len(selected) < limit:
                best_idx = None
                best_value = None
                for idx, candidate in enumerate(remaining):
                    anchor_hits = set(candidate.get("_anchor_hits", []))
                    new_coverage = len(anchor_hits - covered_anchors)
                    overlap = len(anchor_hits.intersection(covered_anchors))
                    dynamic_score = (
                        int(candidate.get("_combined_score", candidate.get("_score", 0)))
                        + new_coverage * 45
                        - overlap * 15
                    )
                    branch = candidate.get("_branch", "")
                    if branch and any(selected_candidate.get("_branch", "") == branch for selected_candidate in selected_candidates):
                        dynamic_score -= 8
                    candidate_value = (dynamic_score, int(candidate.get("_anchor_hit_count", 0)), -idx)
                    if best_value is None or candidate_value > best_value:
                        best_value = candidate_value
                        best_idx = idx

                if best_idx is None:
                    break

                chosen = remaining.pop(best_idx)
                selected_candidates.append(chosen)
                selected.append(f"RVK {chosen['dk']}")
                covered_anchors.update(chosen.get("_anchor_hits", []))
            return selected

        shortlisted_standard = _prefilter_candidates(standard_candidates)
        shortlisted_nonstandard = _prefilter_candidates(nonstandard_candidates)

        if stream_callback:
            if shortlisted_standard:
                preview = ", ".join(f"RVK {item['dk']}" for item in shortlisted_standard[:5])
                stream_callback(
                    f"ℹ️ RVK-Shortlist (standard): {preview}\n",
                    "dk_classification",
                )
            elif shortlisted_nonstandard:
                preview = ", ".join(f"RVK {item['dk']}" for item in shortlisted_nonstandard[:5])
                stream_callback(
                    f"ℹ️ RVK-Shortlist (lokal): {preview}\n",
                    "dk_classification",
                )

        if shortlisted_standard:
            return _pick_diverse(shortlisted_standard, max_standard)
        return _pick_diverse(shortlisted_nonstandard, max_nonstandard)

