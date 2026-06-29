"""Keyword / GND / classification-code helpers - Claude Generated.

Module-level helpers split out of the former ``pipeline_utils`` god-module:
GND-pool verification, descriptive-text keyword/class extraction, and
keyword/RVK canonicalisation + dedup. Still re-exported from ``pipeline_utils``
for backward compatibility. No dependency on PipelineStepExecutor (one-way).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

def verify_keywords_against_gnd_pool(
    extracted_keywords: List[str],
    gnd_pool_keywords: List[str],
    stream_callback: Optional[callable] = None,
    step_id: str = "keywords",
    knowledge_manager = None,  # Claude Generated - for DB fallback verification
) -> Dict[str, Any]:
    """Verify LLM-extracted keywords against the GND pool from step 2 - Claude Generated

    Args:
        extracted_keywords: Keywords extracted from LLM response
        gnd_pool_keywords: The gnd_compliant_keywords built from search results (step 2)
        stream_callback: Optional callback for live progress feedback
        step_id: Pipeline step ID for callback routing
        knowledge_manager: Optional UnifiedKnowledgeManager for DB fallback verification

    Returns:
        Dict with keys: verified (list), rejected (list), stats (dict)
    """
    import logging
    vlog = logging.getLogger(__name__)

    # Debug: Check if knowledge_manager is available
    vlog.info(f"🔍 Verifikation: knowledge_manager={'verfügbar' if knowledge_manager else 'NICHT verfügbar'}")

    if not gnd_pool_keywords:
        vlog.warning("⚠️ GND pool is empty - skipping verification")
        return {
            "verified": list(extracted_keywords),
            "rejected": [],
            "stats": {"total_extracted": len(extracted_keywords), "verified_count": len(extracted_keywords),
                       "rejected_count": 0, "pool_size": 0},
        }

    # Build lookup maps from pool
    gnd_id_to_pool = {}   # gnd_id -> full pool keyword
    text_lower_to_pool = {}  # keyword_text_lower -> full pool keyword

    for pool_kw in gnd_pool_keywords:
        # Extract GND-ID if present
        gnd_match = re.search(r'GND-ID:\s*([0-9X-]+)', pool_kw)
        if gnd_match:
            gnd_id_to_pool[gnd_match.group(1)] = pool_kw

        # Extract keyword text (before first parenthesis)
        kw_text = pool_kw.split('(')[0].strip().lower()
        if kw_text:
            text_lower_to_pool[kw_text] = pool_kw

    if stream_callback:
        stream_callback(
            f"\n🔍 Verifiziere {len(extracted_keywords)} Keywords gegen GND-Pool ({len(gnd_pool_keywords)} Einträge)...\n",
            step_id,
        )

    verified = []
    rejected = []

    for kw in extracted_keywords:
        matched_pool_kw = None

        # Extract GND-ID from extracted keyword (formats: "Keyword (1234567-8)" or "Keyword (GND-ID: 1234567-8)")
        gnd_id_match = re.search(r'GND-ID:\s*([0-9X-]+)', kw)
        if not gnd_id_match:
            gnd_id_match = re.search(r'\((\d{7,}-\d{1,2})\)', kw)

        if gnd_id_match:
            gnd_id = gnd_id_match.group(1)
            if gnd_id in gnd_id_to_pool:
                matched_pool_kw = gnd_id_to_pool[gnd_id]

        # Text match fallback
        if not matched_pool_kw:
            kw_text = kw.split('(')[0].strip().lower()
            if kw_text in text_lower_to_pool:
                matched_pool_kw = text_lower_to_pool[kw_text]

        # Database lookup fallback - Claude Generated
        # If keyword not in pool, search by text in database (ignore LLM's GND-ID)
        if not matched_pool_kw:
            if knowledge_manager:
                kw_text = kw.split('(')[0].strip()
                vlog.info(f"  🔍 DB-Suche nach '{kw_text}'")

                try:
                    # Search GND entries by title/synonyms - this is the authoritative source
                    results = knowledge_manager.search_gnd_by_title(kw_text, fuzzy_threshold=90)
                    vlog.info(f"  🔍 DB-Suche Ergebnisse: {len(results)} Treffer")

                    if results and len(results) > 0:
                        first_result = results[0]
                        if isinstance(first_result, dict) and 'gnd_id' in first_result:
                            db_gnd_id = first_result['gnd_id']
                            db_title = first_result.get('title', kw_text)
                            # Use GND-ID from database (authoritative)
                            matched_pool_kw = f"{db_title} (GND-ID: {db_gnd_id})"
                            vlog.info(f"  💾 DB-Treffer: '{kw_text}' → {db_title} (GND-ID: {db_gnd_id})")

                            # Check if we corrected an LLM error
                            if gnd_id_match:
                                llm_gnd_id = gnd_id_match.group(1)
                                if llm_gnd_id != db_gnd_id:
                                    vlog.info(f"  ✅ DB-verifiziert mit Korrektur: '{kw}' → '{matched_pool_kw}'")
                                    if stream_callback:
                                        stream_callback(f"  ✅ {kw_text} - DB-verifiziert (GND-ID korrigiert: {llm_gnd_id} → {db_gnd_id})\n", step_id)
                                else:
                                    vlog.info(f"  ✅ DB-verifiziert: '{matched_pool_kw}'")
                                    if stream_callback:
                                        stream_callback(f"  ✅ {kw_text} - DB-verifiziert\n", step_id)
                            else:
                                # LLM had no GND-ID, we added it from DB
                                vlog.info(f"  ✅ DB-verifiziert: '{kw}' → '{matched_pool_kw}'")
                                if stream_callback:
                                    stream_callback(f"  ✅ {kw_text} - DB-verifiziert (GND-ID ergänzt)\n", step_id)
                        else:
                            vlog.warning(f"  ⚠️ DB-Ergebnis hat keine GND-ID für '{kw_text}'")
                    else:
                        vlog.info(f"  ❌ Kein DB-Eintrag für '{kw_text}' gefunden")
                except Exception as e:
                    vlog.warning(f"  ⚠️ DB-Suche Fehler für '{kw_text}': {e}")
            else:
                # knowledge_manager not available
                kw_text = kw.split('(')[0].strip()
                vlog.warning(f"  ⚠️ DB-Lookup für '{kw_text}' nicht möglich - knowledge_manager fehlt")

        if matched_pool_kw:
            verified.append(matched_pool_kw)
            if matched_pool_kw != kw:  # Only log if it's from pool (not DB)
                vlog.info(f"  ✅ Verifiziert: {kw[:60]} → Pool: {matched_pool_kw[:60]}")
                if stream_callback:
                    short_name = matched_pool_kw.split('(')[0].strip()
                    stream_callback(f"  ✅ {short_name} - GND-verifiziert\n", step_id)
        else:
            rejected.append(kw)
            vlog.warning(f"  ❌ Abgelehnt: {kw[:60]} - nicht im GND-Pool")
            if stream_callback:
                short_name = kw.split('(')[0].strip()
                stream_callback(f"  ❌ {short_name} - nicht im GND-Pool, entfernt\n", step_id)

    stats = {
        "total_extracted": len(extracted_keywords),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "pool_size": len(gnd_pool_keywords),
    }

    if stream_callback:
        stream_callback(
            f"📊 Verifikation: {stats['verified_count']}/{stats['total_extracted']} GND-verifiziert\n",
            step_id,
        )

    vlog.info(
        f"📊 GND-Verifikation: {stats['verified_count']}/{stats['total_extracted']} verifiziert, "
        f"{stats['rejected_count']} abgelehnt"
    )

    return {"verified": verified, "rejected": rejected, "stats": stats}


def extract_keywords_from_descriptive_text(
    text: str, gnd_compliant_keywords: List[str], output_format: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Extract keywords from LLM descriptive text with robust fallback - Claude Generated

    Supports three formats (in priority order):
    0. JSON: Parse structured JSON response (if output_format == "json")
    1. Primary: "<final_list>…</final_list>" scoped – GND regex applied only inside tags
    2. Fallback: Full-text GND regex if no <final_list> tags are present

    Returns:
        Tuple[List[str], List[str]]:
            - all_keywords: ALL keywords (with and without GND-IDs) for DK search
            - exact_matches: ONLY GND-validated keywords for statistics/history

    Note: DK search uses all_keywords to ensure no LLM-identified keywords are lost.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Debug: Log input for diagnosis - Claude Generated
    logger.info(f"🔍 extract_keywords: input={len(text)} chars, available_gnd={len(gnd_compliant_keywords)}, format={output_format}")

    # JSON-first extraction - Claude Generated
    if output_format != "xml":
        from ..core.json_response_parser import parse_json_response, extract_keywords_from_json
        data = parse_json_response(text)
        if data:
            keywords_str = extract_keywords_from_json(data)
            if keywords_str:
                # Parse the comma-separated string into keyword entries
                all_keywords = []
                exact_matches = []
                gnd_compliant_set = set(gnd_compliant_keywords)

                for kw_entry in keywords_str.split(", "):
                    kw_entry = kw_entry.strip()
                    if not kw_entry:
                        continue
                    all_keywords.append(kw_entry)
                    # Check if this keyword matches any GND-compliant keyword
                    for gnd_kw in gnd_compliant_keywords:
                        kw_text = kw_entry.split("(")[0].strip().lower()
                        gnd_text = gnd_kw.split("(")[0].strip().lower()
                        if kw_text == gnd_text:
                            exact_matches.append(gnd_kw)
                            break

                logger.info(f"✅ JSON keyword extraction: {len(all_keywords)} total, {len(exact_matches)} GND-matched")
                return all_keywords, exact_matches
        logger.warning("JSON keyword extraction fehlgeschlagen, Fallback auf XML")

    # Remove <think> blocks (reasoning steps) before any parsing - Claude Generated
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Attempt to scope extraction to <final_list> content - Claude Generated
    final_list_match = re.search(r'<final_list>\s*(.*?)\s*</final_list>', clean_text, re.DOTALL | re.IGNORECASE)
    if final_list_match:
        extraction_scope = final_list_match.group(1)
        logger.info(f"✅ <final_list> found – scoping extraction to {len(extraction_scope)} chars")
    else:
        extraction_scope = clean_text
        logger.warning("⚠️ No <final_list> found – falling back to full text extraction")

    # PRIMARY METHOD: Regex for "Keyword (1234567-8)" format
    pattern = re.compile(r"\b([A-Za-zäöüÄÖÜß\s-]+?)\s*\((\d{7}-\d|\d{7}-\d{1,2})\)")
    matches = pattern.findall(extraction_scope)

    all_extracted_keywords = []
    exact_matches = []

    # Convert gnd_compliant_keywords to set for faster lookup
    gnd_compliant_set = set(gnd_compliant_keywords)

    if matches:
        logger.info(f"✅ Regex found {len(matches)} keyword matches")

        # Build lookup maps for flexible matching - Claude Generated
        gnd_id_lookup = {}  # gnd_id -> full_keyword
        text_lookup = {}    # keyword_text_lower -> full_keyword

        for gnd_kw in gnd_compliant_keywords:
            # Extract GND-ID if present (format: "Keyword (GND-ID: 1234567-8)")
            gnd_match = re.search(r'GND-ID:\s*([0-9-]+)', gnd_kw)
            if gnd_match:
                gnd_id = gnd_match.group(1)
                gnd_id_lookup[gnd_id] = gnd_kw

            # Extract keyword text (before first parenthesis)
            keyword_text = gnd_kw.split('(')[0].strip().lower()
            text_lookup[keyword_text] = gnd_kw

        logger.info(f"🔍 Built lookups: {len(gnd_id_lookup)} GND-IDs, {len(text_lookup)} text entries")

        # Match extracted keywords using dual strategy - Claude Generated
        for keyword_part, gnd_id_part in matches:
            formatted_keyword = f"{keyword_part.strip()} ({gnd_id_part})"
            all_extracted_keywords.append(formatted_keyword)

            matched = False

            # Strategy 1: GND-ID match (e.g. "1234567-8")
            if gnd_id_part in gnd_id_lookup:
                full_keyword = gnd_id_lookup[gnd_id_part]
                exact_matches.append(full_keyword)
                logger.info(f"  ✅ GND-ID match: '{keyword_part}' ({gnd_id_part}) → {full_keyword[:60]}")
                matched = True

            # Strategy 2: Text match (e.g. "cadmium")
            elif keyword_part.strip().lower() in text_lookup:
                full_keyword = text_lookup[keyword_part.strip().lower()]
                exact_matches.append(full_keyword)
                logger.info(f"  ✅ Text match: '{keyword_part}' → {full_keyword[:60]}")
                matched = True

            if not matched:
                logger.warning(f"  ❌ No match: '{keyword_part}' ({gnd_id_part})")

        logger.info(f"✅ Matched {len(exact_matches)} keywords from {len(matches)} regex matches")
        return all_extracted_keywords, exact_matches

    # FALLBACK METHOD: Parse <final_list> format - Claude Generated
    logger.warning("⚠️ Regex found NO matches - trying <final_list> fallback")

    if final_list_match:
        final_list_content = final_list_match.group(1).strip()
        logger.info(f"✅ Found <final_list>: {final_list_content[:100]}")

        # FIXME: Parser robustness - LLM sometimes returns comma-separated instead of pipe-separated keywords
        # Split by pipe separator (preferred), fall back to comma if needed - Claude Generated
        raw_keywords = [kw.strip() for kw in final_list_content.split('|') if kw.strip()]

        # Fallback: if pipe split yields only one keyword, try comma separator - Claude Generated
        if len(raw_keywords) == 1 and ',' in final_list_content:
            logger.warning(f"⚠️ Pipe separator yielded only 1 keyword, attempting comma fallback")
            raw_keywords = [kw.strip() for kw in final_list_content.split(',') if kw.strip()]
            logger.info(f"✅ Comma fallback: {len(raw_keywords)} keywords extracted")

        logger.info(f"✅ Extracted {len(raw_keywords)} raw keywords from <final_list>")

        # Build lookup map: keyword_text_lower -> full_gnd_keyword
        gnd_lookup = {}
        for gnd_kw in gnd_compliant_keywords:
            # Extract keyword text before (GND-ID: ...)
            if "(GND-ID:" in gnd_kw:
                # ROBUST: Normalize whitespace (multiple spaces, tabs, newlines) - Claude Generated
                keyword_text = " ".join(gnd_kw.split("(GND-ID:")[0].split()).lower()
                gnd_lookup[keyword_text] = gnd_kw

        # Match raw keywords against GND lookup - Claude Generated FIX
        unmatched_keywords = []  # FIX: Track keywords without GND matches for DK search
        for raw_kw in raw_keywords:
            # ROBUST: Normalize whitespace and strip GND-ID label if LLM added it literally - Claude Generated
            raw_kw_normalized = " ".join(raw_kw.split()).lower()

            # Fallback: If LLM returned "Keyword (GND-ID)" without actual ID, strip the label
            if raw_kw_normalized.endswith("(gnd-id)"):
                raw_kw_normalized = raw_kw_normalized[:-9].strip()
                logger.info(f"  ℹ️ Stripped literal '(GND-ID)' label from '{raw_kw}'")

            # Exact match
            if raw_kw_normalized in gnd_lookup:
                matched_gnd_kw = gnd_lookup[raw_kw_normalized]
                exact_matches.append(matched_gnd_kw)
                logger.info(f"  ✅ Matched '{raw_kw}' -> {matched_gnd_kw[:60]}")
            else:
                # Fuzzy match: check if raw_kw is contained in any GND keyword
                found = False
                for gnd_kw_text, full_gnd_kw in gnd_lookup.items():
                    if raw_kw_normalized in gnd_kw_text or gnd_kw_text in raw_kw_normalized:
                        exact_matches.append(full_gnd_kw)
                        logger.info(f"  ⚠️ Fuzzy matched '{raw_kw}' -> {full_gnd_kw[:60]}")
                        found = True
                        break

                if not found:
                    # Try database lookup for GND-ID before treating as plain keyword - Claude Generated
                    db_gnd_id = None
                    try:
                        from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
                        ukm = UnifiedKnowledgeManager()
                        results = ukm.search_by_keywords([raw_kw_normalized], fuzzy_threshold=90)
                        if results and len(results) > 0:
                            first_result = results[0]
                            if isinstance(first_result, dict) and 'gnd_id' in first_result:
                                db_gnd_id = first_result['gnd_id']
                    except Exception as db_err:
                        logger.debug(f"DB lookup error for '{raw_kw_normalized}': {db_err}")

                    if db_gnd_id:
                        # Found GND-ID in database - create proper GND keyword format
                        gnd_keyword = f"{raw_kw} (GND-ID: {db_gnd_id})"
                        exact_matches.append(gnd_keyword)
                        logger.info(f"  🔍 DB lookup matched '{raw_kw}' -> {db_gnd_id}")
                    else:
                        # FIXED: Keywords without GND validation are now INCLUDED in DK search - Claude Generated
                        # All LLM-identified keywords are used to ensure complete catalog coverage:
                        # - Keywords WITH GND-IDs: Full metadata from GND database
                        # - Keywords WITHOUT GND-IDs: Plain text search (user requirement: "das DARF nicht passieren")
                        # - Preserves LLM analysis intent while maximizing catalog search coverage
                        logger.info(f"  ℹ️ No GND match for '{raw_kw}' - using as plain keyword in DK search")
                        unmatched_keywords.append(raw_kw)  # Will be included in DK search via all_keywords

        # Combine GND-matched and plain keywords for complete DK search - Claude Generated FIX
        all_keywords = exact_matches + unmatched_keywords
        logger.info(f"✅ Fallback extraction: {len(exact_matches)} GND-matched + {len(unmatched_keywords)} plain keywords = {len(all_keywords)} total")
        return all_keywords, exact_matches  # Return combined list for DK search, GND-only list for history

    # NO EXTRACTION SUCCESSFUL
    logger.error("❌ NO keyword extraction successful (neither regex nor <final_list>)")
    logger.error(f"Text preview: {text[:300]}")
    return [], []


def extract_keywords_from_descriptive_text_simple(
    text: str, gnd_compliant_keywords: List[str]
) -> List[str]:
    """Simplified keyword extraction using basic string containment - Claude Generated"""

    if not text or not gnd_compliant_keywords:
        return []

    matched_keywords = []
    text_lower = text.lower()

    for gnd_keyword in gnd_compliant_keywords:
        if "(" in gnd_keyword and ")" in gnd_keyword:
            # Extract clean keyword
            clean_keyword = gnd_keyword.split("(")[0].strip().lower()

            # Simple containment check
            if clean_keyword in text_lower:
                matched_keywords.append(gnd_keyword)

    return matched_keywords


def extract_classes_from_descriptive_text(text: str, output_format: Optional[str] = None) -> List[str]:
    """Extract classification classes from LLM text - Claude Generated

    JSON-first extraction if output_format == "json", then XML fallback.
    """
    # JSON-first extraction - Claude Generated
    if output_format != "xml":
        from ..core.json_response_parser import parse_json_response, extract_gnd_classes_from_json
        data = parse_json_response(text)
        if data:
            classes = extract_gnd_classes_from_json(data)
            if classes:
                return classes

    match = re.search(r"<class>(.*?)</class>", text)
    if match:
        classes_str = match.group(1)
        return [cls.strip() for cls in classes_str.split("|") if cls.strip()]
    return []


def canonicalize_keyword(keyword: str) -> str:
    """Extract canonical keyword form by stripping GND-ID - Claude Generated

    Converts "Keyword (GND-ID: 1234567-8)" to "Keyword"
    Handles both formats: with and without GND-ID

    Args:
        keyword: Keyword potentially in format "Keyword (GND-ID: 1234567-8)" or plain "Keyword"

    Returns:
        Canonical form: "Keyword" (stripped of GND-ID and whitespace)

    Examples:
        "Cadmium (GND-ID: 4029921-1)" → "Cadmium"
        "Festkörper (GND-ID: 4016918-2)" → "Festkörper"
        "Molekül" → "Molekül"
    """
    if "(GND-ID:" in keyword:
        return keyword.split("(GND-ID:")[0].strip()
    return keyword.strip()


def extract_gnd_id(keyword: str) -> Optional[str]:
    """Extract GND-ID from a verified keyword string."""
    match = re.search(r"GND-ID:\s*([0-9X-]+)", str(keyword or ""))
    if match:
        return match.group(1).strip()
    return None


def canonicalize_rvk_notation(code: str) -> str:
    """Normalize RVK notation spacing for comparisons and canonical output."""
    clean = str(code or "").strip().upper()
    clean = re.sub(r"\s+", " ", clean)
    match = re.match(r"^([A-Z]{1,4})\s*([0-9].*)$", clean)
    if match:
        clean = f"{match.group(1)} {match.group(2).strip()}"
    return clean


def is_plausible_nonstandard_rvk(code: str) -> bool:
    """Allow local RVK variants, but reject obvious artifacts."""
    clean = canonicalize_rvk_notation(code)
    if not clean:
        return False
    if " - " in clean:
        return False
    return bool(re.match(r"^[A-Z]{1,4}\s[0-9][0-9A-Z./-]*$", clean))


def deduplicate_canonical_keywords(keywords: List[str]) -> List[str]:
    """Deduplicate keywords by canonical form (case-insensitive) - Claude Generated

    Removes duplicate keywords that differ only in their GND-ID or whitespace.
    Preserves first occurrence of each unique canonical keyword.
    Uses case-insensitive comparison to catch "Cadmium" vs "cadmium" duplicates.

    Args:
        keywords: List of keywords, may contain duplicates after GND-ID stripping
                 e.g., ["Cadmium", "Cadmium", "Festkörper"]

    Returns:
        Deduplicated list maintaining original order and formatting

    Examples:
        ["Cadmium", "Cadmium (GND-ID: 1234567-8)"] → ["Cadmium"]
        ["Informatik", "informatik"] → ["Informatik"]
    """
    seen = set()
    deduplicated = []

    for kw in keywords:
        canonical = canonicalize_keyword(kw).lower()
        if canonical not in seen:
            seen.add(canonical)
            deduplicated.append(kw)

    return deduplicated
