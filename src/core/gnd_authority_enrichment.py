"""Fill the local GND store from the DNB authority data - Claude Generated (WP-D2).

The authority path for classifications was wired but dead in three independent
places: ``gnd_entries`` was empty, its only two writers called a method that does
not exist (``update_gnd_entry``, swallowed by a broad ``except``), and the
``gnd_local`` provider tested ``isinstance(ddcs, (list, set, tuple))`` on a value
stored as TEXT, so it discarded every DDC regardless. Fixing the readers alone
would still have yielded nothing — the store has to be filled.

This module is that filler: for GND ids the store does not know yet, it asks the
DNB lookup (RDF, structured, no HTML scraping) and persists the result. It is
deliberately **bounded and opt-in**: a pool run carries ~1000 GND ids and one
HTTP request each is far too much for the live path, so callers pass the small
set they actually care about (the selected keywords, typically ~20).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.utils.classification_systems import format_stored_ddcs

logger = logging.getLogger(__name__)

#: Hard ceiling on requests per call. The bound exists so a mistaken caller
#: (passing a whole pool) degrades to a partial enrichment instead of firing a
#: thousand requests at the DNB.
DEFAULT_MAX_LOOKUPS = 25


def missing_gnd_ids(ukm: Any, gnd_ids: Sequence[str]) -> List[str]:
    """The ids the local store has no entry for yet, order-preserving."""
    unique = list(dict.fromkeys(str(g).strip() for g in gnd_ids if str(g or "").strip()))
    if not unique:
        return []
    try:
        known = ukm.get_gnd_facts_batch(unique)
    except Exception as exc:  # a broken store must not break the caller
        logger.warning("gnd enrichment: local lookup failed (%s)", exc)
        return []
    return [gnd_id for gnd_id in unique if gnd_id not in known]


def enrich_gnd_entries(
    ukm: Any,
    gnd_ids: Sequence[str],
    *,
    classify: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    max_lookups: int = DEFAULT_MAX_LOOKUPS,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Fetch missing GND ids from the DNB and store them locally.

    Args:
        ukm: UnifiedKnowledgeManager (reads ``get_gnd_facts_batch``, writes
            ``store_gnd_fact``).
        gnd_ids: candidate ids — already-known ones are skipped.
        classify: injected ``gnd_id -> DNB dict`` (defaults to the ``dnb``
            lookup plugin, so a disabled plugin means "no enrichment", not a
            crash). Injectable to keep this testable without network.
        max_lookups: request ceiling for this call.

    Returns ``{"requested", "stored", "failed", "skipped_over_limit"}``. Never
    raises for a single failed id: enrichment is an improvement, and losing it
    must not take the surrounding pipeline down with it.
    """
    missing = missing_gnd_ids(ukm, gnd_ids)
    if not missing:
        return {"requested": 0, "stored": 0, "failed": 0, "skipped_over_limit": 0}

    over_limit = max(0, len(missing) - max_lookups)
    targets = missing[:max_lookups]

    if classify is None:
        classify = _default_classifier()
    if classify is None:
        logger.info("gnd enrichment: dnb lookup disabled — skipping")
        return {
            "requested": 0, "stored": 0, "failed": 0,
            "skipped_over_limit": len(missing),
        }

    stored = failed = 0
    aborted: Optional[str] = None
    for index, gnd_id in enumerate(targets, 1):
        if progress:
            try:
                progress(f"DNB {index}/{len(targets)}: {gnd_id}")
            except Exception:
                pass
        try:
            data = classify(gnd_id)
        except ImportError as exc:
            # A missing dependency will not get better on the next id. Report it
            # once, loudly, and stop — the DNB client needs `rdflib`, which went
            # undeclared long enough for the whole authority path to look merely
            # "empty" rather than broken. - Claude Generated
            aborted = f"DNB client unavailable ({exc}) — is rdflib installed?"
            logger.error("gnd enrichment aborted: %s", aborted)
            break
        except Exception as exc:
            logger.warning("gnd enrichment: DNB request failed for %s (%s)", gnd_id, exc)
            failed += 1
            continue
        if not _store_one(ukm, gnd_id, data):
            failed += 1
            continue
        stored += 1

    result = {
        "requested": len(targets),
        "stored": stored,
        "failed": failed,
        "skipped_over_limit": over_limit,
    }
    if aborted:
        result["aborted"] = aborted
    return result


def _default_classifier() -> Optional[Callable[[str], Optional[Dict[str, Any]]]]:
    """The ``dnb`` lookup plugin, or None when it is disabled."""
    try:
        from src.utils.lookups import build_lookup

        lookup = build_lookup(None, "dnb")
    except Exception as exc:
        logger.warning("gnd enrichment: dnb lookup unavailable (%s)", exc)
        return None
    return lookup.classify if lookup is not None else None


def _store_one(ukm: Any, gnd_id: str, data: Optional[Dict[str, Any]]) -> bool:
    """Persist one DNB result; False when there was nothing usable to store."""
    if not isinstance(data, dict) or data.get("status") != "success":
        return False
    title = str(data.get("preferred_name") or "").strip()
    if not title:
        # The store rejects entries without a title on read, so writing one
        # would create a row that can never be returned.
        return False
    try:
        ukm.store_gnd_fact(gnd_id, {
            "title": title,
            "description": "",
            "synonyms": "",
            # Column format is TEXT: "551.9(1);577.14(2)". The DNB spells the
            # concept "determinancy"; normalise here.
            "ddcs": format_stored_ddcs([
                {"code": d.get("code"), "determinacy": d.get("determinancy")}
                for d in (data.get("ddc") or [])
                if isinstance(d, dict)
            ]),
            "ppn": "",
        })
    except Exception as exc:
        logger.warning("gnd enrichment: storing %s failed (%s)", gnd_id, exc)
        return False
    return True
