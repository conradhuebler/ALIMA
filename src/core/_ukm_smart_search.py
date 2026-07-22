"""Mapping-first smart search for the unified knowledge DB - Claude Generated.

Split out of ``unified_knowledge_manager.py`` (WP cleanup D). Verbatim mixin
extraction: the methods stay on ``UnifiedKnowledgeManager`` via MRO, so no call
site changes.

The "Week 2" mapping-first read path: search a term via its cached mapping
before a live query, extract GND ids from live results, and the DK/statistics
lookups + the titleless-classification cleanup that sit with it. They call back
into the mappings layer (``get_search_mapping`` / ``update_search_mapping``,
which stays on the class) via ``self`` — one class across files.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List


class SmartSearchMixin:
    """The mapping-first search path. Mixed into :class:`UnifiedKnowledgeManager`."""

    def search_with_mappings_first(self, search_term: str, suggester_type: str,
                                 max_age_hours: int = 24,
                                 live_search_fallback: callable = None,
                                 force_update: bool = False) -> tuple[List[str], bool]:
        """
        Week 2: Smart search with mappings-first strategy - Claude Generated

        Args:
            search_term: Term to search for
            suggester_type: Type of suggester (lobid, swb, catalog)
            max_age_hours: Maximum age of cached mappings in hours
            live_search_fallback: Function to call for live search if mapping miss
            force_update: If True, ignore cache and force live search - Claude Generated

        Returns:
            Tuple of (found_gnd_ids, was_from_cache)
        """
        from datetime import datetime, timedelta

        # Step 1: Force live search if force_update is True - Claude Generated
        if force_update:
            if hasattr(self, 'debug_mapping') and self.debug_mapping:
                self.logger.info(f"⚠️ Force update: skipping cache for '{search_term}' ({suggester_type})")
            # Skip cache check and go directly to live search
            if live_search_fallback:
                try:
                    live_results = live_search_fallback(search_term)
                    if live_results:
                        gnd_ids = self._extract_gnd_ids_from_results(live_results, suggester_type)
                        # Update mapping with fresh results (merging will be handled in store_classification_results)
                        self.update_search_mapping(
                            search_term=search_term,
                            suggester_type=suggester_type,
                            found_gnd_ids=gnd_ids
                        )
                        self.logger.info(f"✅ Force update complete for '{search_term}': {len(gnd_ids)} results")
                        return gnd_ids, False
                except Exception as e:
                    self.logger.error(f"Force update failed for '{search_term}': {e}")
            return [], False

        # Step 2: Normal cache-first logic
        mapping = self.get_search_mapping(search_term, suggester_type)
        
        if mapping:
            # Check if mapping is fresh enough
            try:
                # Handle both string and datetime objects (MariaDB returns datetime, not string)
                last_updated_val = mapping.last_updated
                if isinstance(last_updated_val, datetime):
                    last_updated = last_updated_val
                else:
                    last_updated = datetime.fromisoformat(str(last_updated_val).replace('Z', '+00:00'))
                max_age = timedelta(hours=max_age_hours)

                if datetime.now() - last_updated < max_age:
                    if hasattr(self, 'debug_mapping') and self.debug_mapping:
                        self.logger.info(f"✅ Mapping hit for '{search_term}' ({suggester_type}): {len(mapping.found_gnd_ids)} results from cache")
                    return mapping.found_gnd_ids, True
                else:
                    self.logger.info(f"⏰ Stale mapping for '{search_term}' ({suggester_type}): {(datetime.now() - last_updated).total_seconds()/3600:.1f}h old")
            except ValueError:
                self.logger.warning(f"Invalid last_updated timestamp for mapping: {mapping.last_updated}")
        else:
            if hasattr(self, 'debug_mapping') and self.debug_mapping:
                self.logger.info(f"❌ No mapping found for '{search_term}' ({suggester_type})")
        
        # Step 2: Mapping miss or stale - fallback to live search
        if live_search_fallback:
            if hasattr(self, 'debug_mapping') and self.debug_mapping:
                self.logger.info(f"🌐 Performing live search for '{search_term}' ({suggester_type})")
            try:
                live_results = live_search_fallback(search_term)
                
                # Step 3: Update mapping with fresh results
                if live_results:
                    # Extract GND IDs from live results (format depends on suggester)
                    gnd_ids = self._extract_gnd_ids_from_results(live_results, suggester_type)
                    
                    # Store the updated mapping
                    self.update_search_mapping(
                        search_term=search_term,
                        suggester_type=suggester_type, 
                        found_gnd_ids=gnd_ids
                    )
                    
                    if hasattr(self, 'debug_mapping') and self.debug_mapping:
                        self.logger.info(f"✅ Updated mapping for '{search_term}' ({suggester_type}): {len(gnd_ids)} results")
                    return gnd_ids, False
                else:
                    # Store empty result to avoid repeated failed searches
                    self.update_search_mapping(
                        search_term=search_term,
                        suggester_type=suggester_type,
                        found_gnd_ids=[]
                    )
                    self.logger.info(f"∅ No results for '{search_term}' ({suggester_type}) - stored empty mapping")
                    return [], False
                    
            except Exception as e:
                self.logger.error(f"Live search failed for '{search_term}' ({suggester_type}): {e}")
                return [], False
        
        # No live search fallback provided
        self.logger.warning(f"No live search fallback provided for '{search_term}' ({suggester_type})")
        return [], False

    def _extract_gnd_ids_from_results(self, results: Dict[str, Any], suggester_type: str) -> List[str]:
        """Extract GND IDs from suggester-specific result format - Claude Generated"""
        gnd_ids = []
        
        try:
            if suggester_type in ("lobid", "swb"):
                # Canonical suggester results: {term: {keyword: {"gnd_ids": set, ...}}}
                for term_results in results.values():
                    for keyword_data in term_results.values():
                        if "gnd_ids" in keyword_data:
                            gnd_set = keyword_data["gnd_ids"]
                            if isinstance(gnd_set, set):
                                gnd_ids.extend(list(gnd_set))
                            elif isinstance(gnd_set, list):
                                gnd_ids.extend(gnd_set)
                                
            elif suggester_type == "catalog":
                # Catalog/BiblioSuggester results may have different format
                # This needs to be adapted based on actual BiblioSuggester output
                self.logger.warning("GND ID extraction for catalog suggester not yet implemented")
                
        except Exception as e:
            self.logger.error(f"Error extracting GND IDs from {suggester_type} results: {e}")
            
        # Remove duplicates and filter out empty/invalid IDs
        unique_gnd_ids = list(set(gid for gid in gnd_ids if gid and len(str(gid).strip()) > 0))
        return unique_gnd_ids

    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get statistics about search mappings - Claude Generated"""
        try:
            # Total mappings by suggester type
            by_suggester_rows = self.db_manager.fetch_all("""
                SELECT suggester_type, COUNT(*) as count,
                       AVG(result_count) as avg_results,
                       MAX(last_updated) as latest_update
                FROM search_mappings
                GROUP BY suggester_type
            """)

            # Recent activity (last 24 hours)
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

            recent_stats_row = self.db_manager.fetch_one("""
                SELECT COUNT(*) as recent_mappings,
                       AVG(result_count) as recent_avg_results
                FROM search_mappings
                WHERE last_updated > ?
            """, [cutoff])

            return {
                "by_suggester": [
                    {
                        "type": row["suggester_type"],
                        "count": row["count"],
                        "avg_results": round(row["avg_results"] or 0, 1),
                        "latest_update": row["latest_update"]
                    }
                    for row in by_suggester_rows
                ],
                "recent_24h": {
                    "mappings": recent_stats_row["recent_mappings"] if recent_stats_row else 0,
                    "avg_results": round(recent_stats_row["recent_avg_results"] or 0, 1) if recent_stats_row else 0
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting mapping statistics: {e}")
            return {"error": str(e)}

    def get_dk_for_gnd_id(self, gnd_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve DK classifications for a given GND-ID - Claude Generated

        Searches through catalog search mappings to find all DK classifications
        associated with a specific GND-ID, including titles and frequency information.

        Args:
            gnd_id: The GND-ID to search for (e.g., "4061694-5")
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with structure:
            [
                {
                    "dk": "614.7",
                    "type": "DK",
                    "titles": ["Title 1", "Title 2"],
                    "count": 5,
                    "avg_confidence": 0.85
                },
                ...
            ]
        """
        try:
            # Search in search_mappings where suggester_type is 'catalog'
            rows = self.db_manager.fetch_all("""
                SELECT found_classifications
                FROM search_mappings
                WHERE suggester_type = 'catalog'
                AND found_classifications LIKE ?
            """, [f'%"{gnd_id}"%'])

            if not rows:
                self.logger.debug(f"No DK classifications found for GND-ID {gnd_id}")
                return []

            # Parse JSON and collect matching classifications
            classifications = {}  # Use dict to deduplicate by code

            for row in rows:
                try:
                    found_classifications = json.loads(row['found_classifications'] or '[]')

                    for cls in found_classifications:
                        # Check if this classification contains the GND-ID
                        gnd_ids = cls.get('gnd_ids', [])
                        if gnd_id in gnd_ids:
                            code = cls.get('code')

                            # Deduplicate: merge if code already exists
                            if code in classifications:
                                # Merge titles (avoid duplicates)
                                existing_titles = set(classifications[code]['titles'])
                                new_titles = cls.get('titles', [])
                                for title in new_titles:
                                    if title not in existing_titles and len(classifications[code]['titles']) < max_results:
                                        classifications[code]['titles'].append(title)

                                # Update count and confidence
                                classifications[code]['count'] += cls.get('count', 0)  # FIX: Default to 0 (no titles), not 1 - Claude Generated
                                classifications[code]['avg_confidence'] = (
                                    classifications[code]['avg_confidence'] +
                                    cls.get('avg_confidence', 0.8)
                                ) / 2
                            else:
                                # New classification entry
                                classifications[code] = {
                                    'dk': code,
                                    'type': cls.get('type', 'DK'),
                                    'titles': cls.get('titles', [])[:max_results],  # Limit titles
                                    'count': cls.get('count', 1),
                                    'avg_confidence': cls.get('avg_confidence', 0.8)
                                }

                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse classification JSON: {e}")
                    continue

            # Convert to list and sort by count (descending)
            results = list(classifications.values())
            results.sort(key=lambda x: x['count'], reverse=True)

            if results:
                self.logger.info(f"✅ Found {len(results)} DK classifications for GND-ID {gnd_id}")

            return results[:max_results]

        except Exception as e:
            self.logger.error(f"Error searching DK for GND-ID {gnd_id}: {e}")
            return []

    def cleanup_titleless_classifications(self, dry_run: bool = True) -> Dict[str, int]:
        """
        Remove cached classifications without titles from search_mappings - Claude Generated

        Cleans up the database by removing classification entries that have empty
        titles arrays. This helps maintain data quality and reduces prompt bloat.

        Args:
            dry_run: If True, only report what would be cleaned without making changes

        Returns:
            Dictionary with statistics:
            {
                "mappings_processed": int,
                "classifications_removed": int,
                "classifications_kept": int,
                "mappings_updated": int
            }
        """
        try:
            stats = {
                "mappings_processed": 0,
                "classifications_removed": 0,
                "classifications_kept": 0,
                "mappings_updated": 0
            }

            # Get all catalog search mappings
            rows = self.db_manager.fetch_all("""
                SELECT search_term, found_classifications
                FROM search_mappings
                WHERE suggester_type = 'catalog'
            """)

            self.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processing {len(rows)} catalog search mappings...")

            for row in rows:
                stats["mappings_processed"] += 1
                search_term = row['search_term']

                try:
                    classifications = json.loads(row['found_classifications'] or '[]')

                    if not classifications:
                        continue  # Skip empty mappings

                    # Filter out classifications without titles
                    cleaned_classifications = []
                    for cls in classifications:
                        titles = cls.get('titles', [])
                        # Check if at least one valid title exists
                        if titles and any(t.strip() for t in titles if t):
                            cleaned_classifications.append(cls)
                            stats["classifications_kept"] += 1
                        else:
                            stats["classifications_removed"] += 1
                            if dry_run:
                                self.logger.debug(f"[DRY RUN] Would remove {cls.get('type', 'DK')}: {cls.get('code')} from '{search_term}' (no titles)")

                    # Update mapping if classifications were removed
                    if len(cleaned_classifications) < len(classifications):
                        stats["mappings_updated"] += 1

                        if not dry_run:
                            # Update the search mapping with cleaned data
                            self.update_search_mapping(
                                search_term=search_term,
                                suggester_type="catalog",
                                found_classifications=cleaned_classifications
                            )
                            self.logger.debug(f"✅ Cleaned '{search_term}': kept {len(cleaned_classifications)}/{len(classifications)} classifications")
                        else:
                            self.logger.debug(f"[DRY RUN] Would clean '{search_term}': keep {len(cleaned_classifications)}/{len(classifications)} classifications")

                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to process mapping for '{search_term}': {e}")
                    continue

            # Log summary
            action_verb = "Would remove" if dry_run else "Removed"
            self.logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleanup summary:")
            self.logger.info(f"  - Mappings processed: {stats['mappings_processed']}")
            self.logger.info(f"  - Classifications kept: {stats['classifications_kept']}")
            self.logger.info(f"  - Classifications {action_verb.lower()}: {stats['classifications_removed']}")
            self.logger.info(f"  - Mappings updated: {stats['mappings_updated']}")

            if dry_run and stats['classifications_removed'] > 0:
                self.logger.info(f"💡 Run with dry_run=False to apply these changes")

            return stats

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                "mappings_processed": 0,
                "classifications_removed": 0,
                "classifications_kept": 0,
                "mappings_updated": 0,
                "error": str(e)
            }
