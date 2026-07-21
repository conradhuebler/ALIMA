"""Catalog DK/RVK cache for the unified knowledge DB - Claude Generated.

Split out of ``unified_knowledge_manager.py`` (WP cleanup D). Verbatim mixin
extraction: the methods stay on ``UnifiedKnowledgeManager`` via MRO, so no call
site changes.

This is a self-contained subsystem — its own table (``catalog_dk_cache``), its
own TTL and failure tracking, its own merge rule for classifications found on
catalogue titles. It sat in the middle of the GND/mapping query layer, which it
has nothing to do with.

Note the method-local imports (``timedelta``, ``traceback``,
``classification_system``) are deliberate and travel with the code.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class CatalogDkCacheMixin:
    """The ``catalog_dk_cache`` subsystem. Mixed into :class:`UnifiedKnowledgeManager`."""

    def get_catalog_dk_cache(self, search_term: str) -> Optional[tuple]:
        """Retrieve catalog titles from dedicated cache with TTL support - Claude Generated

        Returns:
            Tuple of (titles, status, error_message) or None if cache miss or TTL expired
            - titles: List of catalog title dicts (empty for failures)
            - status: 'success', 'no_results', 'error', 'timeout', 'circuit_breaker'
            - error_message: Error details if status != 'success', else None
        """
        try:
            row = self.db_manager.fetch_one("""
                SELECT found_titles, search_status, error_message, retry_after
                FROM catalog_dk_cache
                WHERE search_term = ?
            """, [search_term])

            if not row:
                return None

            # Check TTL for failed searches
            if row['search_status'] != 'success' and row['retry_after']:
                from datetime import datetime
                try:
                    retry_val = row['retry_after']
                    if isinstance(retry_val, datetime):
                        retry_after = retry_val
                    else:
                        retry_after = datetime.fromisoformat(str(retry_val).replace('Z', '+00:00'))
                    if datetime.now() < retry_after:
                        # TTL not expired - return cached failure
                        titles = json.loads(row['found_titles'] or '[]')
                        self.logger.debug(
                            f"⏰ Cached failure for '{search_term}': {row['search_status']} "
                            f"(TTL active until {retry_after.strftime('%H:%M:%S')})"
                        )
                        return (titles, row['search_status'], row['error_message'])
                    else:
                        # TTL expired - allow retry
                        self.logger.debug(f"🔄 TTL expired for '{search_term}' - allowing retry")
                        return None
                except Exception as e:
                    self.logger.warning(f"Error parsing retry_after timestamp: {e}")

            # Parse titles
            titles = json.loads(row['found_titles'] or '[]')
            self.logger.debug(
                f"✅ Catalog cache hit for '{search_term}': {len(titles)} titles "
                f"(status={row['search_status']})"
            )
            return (titles, row['search_status'], row['error_message'])

        except Exception as e:
            self.logger.error(f"Error retrieving catalog cache for '{search_term}': {e}")
            return None

    def store_catalog_dk_cache(self, search_term: str, titles: List[Dict[str, Any]],
                             status: str = 'success', error_message: Optional[str] = None,
                             ttl_minutes: int = 30) -> bool:
        """Store catalog titles or failed search in dedicated cache - Claude Generated

        Args:
            search_term: Keyword searched
            titles: List of catalog titles (empty for failures)
            status: 'success', 'no_results', 'error', 'timeout'
            error_message: Error details if status != 'success'
            ttl_minutes: Cache TTL for failed searches (prevents repeated failures)

        Returns:
            True if storage succeeded
        """
        try:
            from datetime import datetime, timedelta

            normalized_term = self._normalize_term(search_term)
            result_count = len(titles)
            retry_after = None

            # For failed searches, set retry_after timestamp
            if status != 'success' and ttl_minutes > 0:
                retry_after = (datetime.now() + timedelta(minutes=ttl_minutes)).isoformat()

            # Log appropriately based on status
            if status == 'success':
                # DIAGNOSTIC: count classifications by system (DK/DDC/RVK) - Claude Generated
                from src.utils.classification_systems import classification_system
                counts = {"DK": 0, "DDC": 0, "RVK": 0}
                for title in titles:
                    for c in title.get('classifications', []):
                        s = classification_system(str(c))
                        if not s and str(c).replace('.', '', 1).isdigit():
                            s = "DK"  # legacy: a bare number means DK
                        if s in counts:
                            counts[s] += 1
                self.logger.debug(
                    f"Storing catalog cache for '{search_term}': {result_count} titles | "
                    f"DK: {counts['DK']} | DDC: {counts['DDC']} | RVK: {counts['RVK']}"
                )

                for title in titles[:3]:  # Log first 3 titles for success
                    classifications_count = len(title.get('classifications', []))
                    title_dk = sum(1 for c in title.get('classifications', []) if str(c).startswith('DK ') or str(c).replace('.', '', 1).isdigit())
                    title_rvk = sum(1 for c in title.get('classifications', []) if str(c).startswith('RVK '))
                    self.logger.debug(f"   - {title.get('title', '')[:50]}...: {classifications_count} cls (DK: {title_dk}, RVK: {title_rvk})")
            else:
                self.logger.info(
                    f"💾 Caching failure for '{search_term}': {status} "
                    f"(TTL: {ttl_minutes}min) - {error_message or 'No error message'}"
                )

            # Store or update cache entry
            self.db_manager.execute_query("""
                INSERT OR REPLACE INTO catalog_dk_cache
                (search_term, normalized_term, found_titles, result_count,
                 last_updated, created_at, search_status, error_message, retry_after)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP,
                        COALESCE((SELECT created_at FROM catalog_dk_cache WHERE search_term = ?), CURRENT_TIMESTAMP),
                        ?, ?, ?)
            """, [
                search_term,
                normalized_term,
                json.dumps(titles),
                result_count,
                search_term,
                status,
                error_message,
                retry_after
            ])

            # VERIFY: Check that it was actually stored
            verify_row = self.db_manager.fetch_one(
                "SELECT result_count, search_status FROM catalog_dk_cache WHERE search_term = ?",
                [search_term]
            )
            if verify_row:
                self.logger.debug(f"✅ Verified: Stored {result_count} titles for '{search_term}' (status={verify_row['search_status']})")
                return True
            else:
                self.logger.error(f"❌ FAILED: Could not verify storage for '{search_term}'")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error storing catalog cache for '{search_term}': {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def store_catalog_dk_cache_batch(self, cache_entries: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Efficiently store multiple catalog cache entries in batch - Claude Generated

        Args:
            cache_entries: Dict mapping search_term -> titles_list
                          e.g., {"Umweltschutz": [{...titles}], "Nachhaltigkeit": [{...titles}]}

        Returns:
            True if all entries stored successfully
        """
        if not cache_entries:
            return True

        try:
            for search_term, titles in cache_entries.items():
                # Use existing single-insert for now (batch transaction in db_manager)
                self.store_catalog_dk_cache(search_term, titles)

            self.logger.info(f"✅ Batch insert: Stored {len(cache_entries)} catalog cache entries ({sum(len(t) for t in cache_entries.values())} titles total)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error in batch catalog cache insert: {e}")
            return False

    def extract_classifications_from_titles(self, titles: List[Dict[str, Any]], matched_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Extract and group classifications from title list - Claude Generated

        Input: [{"rsn": "123", "title": "...", "classifications": ["DK 681.3", "RVK UP 5400"]}, ...]
        Output: [{"dk": "681.3", "type": "DK", "titles": ["..."], "matched_keywords": [...], "count": N}, ...]
        """
        try:
            if not titles:
                return []

            # Group by classification code
            grouped = {}

            for title in titles:
                title_str = title.get('title', '')
                classifications = title.get('classifications', [])

                # Parse each classification string: "DK 681.3" or "RVK UP 5400"
                for cls_str in classifications:
                    parts = cls_str.strip().split(None, 1)  # Split on first space
                    if len(parts) == 2:
                        cls_type, code = parts
                    else:
                        self.logger.warning(f"Invalid classification format: '{cls_str}'")
                        continue

                    key = f"{cls_type}:{code}"

                    if key not in grouped:
                        grouped[key] = {
                            "dk": code,  # Use "dk" for backward compatibility
                            "type": cls_type,
                            "classification_type": cls_type,  # Also support classification_type field
                            "titles": [],
                            "matched_keywords": matched_keywords or [],
                            "keywords": matched_keywords or [],  # Alias for UI compatibility
                            "count": 0
                        }

                    # Add title if not already in list (deduplicate)
                    if title_str and title_str not in grouped[key]["titles"]:
                        grouped[key]["titles"].append(title_str)

                    grouped[key]["count"] += 1

            # Convert to list and calculate confidence
            result = []
            for cls_data in grouped.values():
                # Calculate confidence based on count: more titles = higher confidence
                # Range: 0.5 (1 title) to 1.0 (10+ titles)
                count = cls_data["count"]
                avg_confidence = min(0.5 + (count * 0.05), 1.0)  # Max 1.0
                cls_data["avg_confidence"] = avg_confidence
                result.append(cls_data)

            self.logger.debug(f"Extracted {len(result)} unique classifications from {len(titles)} titles")
            return result

        except Exception as e:
            self.logger.error(f"Error extracting classifications from titles: {e}")
            return []

    def _merge_catalog_classifications(self, existing: List[Dict], new_cls: Dict) -> List[Dict]:
        """Helper to merge classification entries - Claude Generated"""
        code = new_cls['code']

        # Find existing entry for this code
        for i, ex_cls in enumerate(existing):
            if ex_cls.get('code') == code:
                # Merge titles (deduplicate)
                all_titles = ex_cls.get('titles', []) + new_cls.get('titles', [])
                unique_titles = list(dict.fromkeys(all_titles))[:10]  # Keep top 10

                # Update entry
                existing[i] = {
                    "code": code,
                    "type": new_cls.get('type', 'DK'),
                    "titles": unique_titles,
                    "count": ex_cls.get('count', 0) + new_cls.get('count', 0),
                    "avg_confidence": (ex_cls.get('avg_confidence', 0.8) + new_cls.get('avg_confidence', 0.8)) / 2,
                    "gnd_ids": list(set(ex_cls.get('gnd_ids', []) + new_cls.get('gnd_ids', [])))
                }
                return existing

        # Not found - add new entry
        existing.append(new_cls)
        return existing
