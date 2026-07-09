"""Mapping-first caching as a reusable provider wrapper - Claude Generated.

What used to be fused into ``MetaSuggester.search`` (the GND-mapping cache lookup +
write-back) is here a decorator around any ``GND_KEYWORDS`` provider. Non-GND
capabilities pass straight through (records/facets are not cached — matches today).

**F-4 fix lives here.** On a cache hit the restored entries keep the pool
``count = 1`` (so ranking / chunking is byte-for-byte unchanged — see the
count-landmine in ``src/core/gnd_search_core.py``) but carry a separate
``display_count`` = the real hit count stored at write time
(``UnifiedKnowledgeManager`` ``gnd_counts``). Display layers read ``display_count``;
ranking never does. Rows written before the column existed have no counts → the
items get ``display_count = None`` and fall back to the pool count (1), the prior
behaviour.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional, Tuple

from .provider import ProviderResult, ResultItem, SearchCapability

logger = logging.getLogger(__name__)


class CachingProvider:
    """Wrap a provider with mapping-first GND-keyword caching.

    Presents the same ``SearchProvider`` surface (delegates ``id`` / ``label`` /
    ``capabilities`` / ``is_available`` to the inner provider).
    """

    def __init__(
        self,
        inner: Any,
        ukm: Any = None,
        max_age_hours: int = 24,
        force_update: bool = False,
    ):
        self.inner = inner
        self.id = inner.id
        self.label = inner.label
        self.capabilities = inner.capabilities
        self.max_age_hours = max_age_hours
        self.force_update = force_update
        self._ukm = ukm

    @property
    def ukm(self):
        if self._ukm is None:
            from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

            self._ukm = UnifiedKnowledgeManager()
        return self._ukm

    def is_available(self, cfg: Any = None) -> bool:
        return self.inner.is_available(cfg)

    def search(
        self,
        capability: SearchCapability,
        query: List[str],
        *,
        progress: Optional[Callable[[str], None]] = None,
        **opts: Any,
    ) -> ProviderResult:
        if capability is not SearchCapability.GND_KEYWORDS:
            # Records / facets are not cached.
            return self.inner.search(capability, query, progress=progress, **opts)

        per_term = {}
        errors = {}
        for term in query:
            items, err = self._search_term(term, progress, **opts)
            per_term[term] = items
            if err:
                errors[term] = err
        return ProviderResult(
            SearchCapability.GND_KEYWORDS, per_term=per_term, errors=errors
        )

    # --- per-term cache logic ---------------------------------------------
    def _search_term(
        self, term: str, progress: Optional[Callable[[str], None]], **opts: Any
    ) -> Tuple[List[ResultItem], Optional[str]]:
        if not self.force_update:
            mapping = self.ukm.get_search_mapping(term, self.id)
            if (
                mapping is not None
                and self._is_fresh(mapping.last_updated)
                and self._has_titles(mapping)
            ):
                logger.debug("cache hit for '%s' (%s)", term, self.id)
                return self._items_from_cache(mapping), None
        return self._live_search(term, progress, **opts)

    @staticmethod
    def _has_titles(mapping: Any) -> bool:
        """Whether the mapping can rebuild items on its own (WP Phase C1a).

        Titles are denormalized into the mapping row; a row with GND IDs but no
        titles predates C1a and must be re-fetched live (we no longer read the
        separate local GND store to resolve them). An empty-result row (no GND IDs)
        is a legitimate 'no hits' cache entry and stays a hit.
        """
        gnd_ids = getattr(mapping, "found_gnd_ids", []) or []
        if not gnd_ids:
            return True
        titles = getattr(mapping, "titles", {}) or {}
        return any(titles.get(g) for g in gnd_ids)

    def _is_fresh(self, last_updated: Any) -> bool:
        try:
            if isinstance(last_updated, datetime):
                ts = last_updated
            else:
                ts = datetime.fromisoformat(
                    str(last_updated).replace("Z", "+00:00")
                )
            return datetime.now() - ts < timedelta(hours=self.max_age_hours)
        except (ValueError, TypeError):
            return False

    def _items_from_cache(self, mapping: Any) -> List[ResultItem]:
        """Rebuild GND-keyword items from a cached mapping (dedup by title).

        Titles come from the mapping's own denormalized ``titles`` map (WP Phase
        C1a) — the cache no longer reads the separate local GND store, so a
        standalone search never depends on (or pollutes) the local authority copy.
        IDs without a stored title are dropped.

        Pool ``count`` stays 1 (landmine); ``display_count`` is the stored real
        count (``None`` when not stored → display falls back to 1).
        """
        gnd_counts = getattr(mapping, "gnd_counts", {}) or {}
        titles_map = getattr(mapping, "titles", {}) or {}
        by_title = {}
        for gnd_id in mapping.found_gnd_ids:
            title = titles_map.get(gnd_id)
            if not title:
                continue
            stored = gnd_counts.get(gnd_id)
            if title in by_title:
                by_title[title].gnd_ids.add(gnd_id)
                if stored is not None:
                    cur = by_title[title].display_count
                    by_title[title].display_count = (
                        stored if cur is None else max(cur, stored)
                    )
            else:
                by_title[title] = ResultItem(
                    label=title,
                    gnd_ids={gnd_id},
                    count=1,
                    display_count=None if stored is None else int(stored),
                )
        return list(by_title.values())

    def _live_search(
        self, term: str, progress: Optional[Callable[[str], None]], **opts: Any
    ) -> Tuple[List[ResultItem], Optional[str]]:
        try:
            res = self.inner.search(
                SearchCapability.GND_KEYWORDS, [term], progress=progress, **opts
            )
        except Exception as exc:  # source failure — do NOT cache as "no hit"
            logger.warning("live search failed for '%s' (%s): %s", term, self.id, exc)
            return [], str(exc)

        items = res.per_term.get(term, [])
        err = res.errors.get(term)

        if err is not None:
            # Source failure recorded by the provider: don't poison the cache.
            return items, err

        # Write-back: store the GND IDs and their real counts for display restore.
        gnd_counts = {}
        titles_by_id = {}
        for it in items:
            for gnd_id in it.gnd_ids:
                gnd_counts[gnd_id] = max(gnd_counts.get(gnd_id, 0), int(it.count or 0))
                titles_by_id.setdefault(gnd_id, it.label)
        # Denormalize titles into the mapping row (WP Phase C1a) so a later cache
        # hit rebuilds items on its own. The cache is self-contained: a standalone
        # search no longer writes into the local GND store (no more warm_gnd_entries),
        # keeping the plugin-owned local GND copy independent of the search cache.
        self.ukm.update_search_mapping(
            term,
            self.id,
            found_gnd_ids=list(gnd_counts.keys()),
            gnd_counts=gnd_counts,
            titles=titles_by_id,
        )
        return items, None
