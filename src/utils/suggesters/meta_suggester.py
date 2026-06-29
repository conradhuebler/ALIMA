#!/usr/bin/env python3
# meta_suggester.py
"""Orchestrator over the capability-based search providers - Claude Generated.

P2 of the search-provider-plugin migration: ``MetaSuggester`` no longer hard-codes
an if/elif over a ``SuggesterType`` enum. It **enumerates the provider registry**
(``src/core/search``) by provider id, wraps each ``GND_KEYWORDS`` provider in the
mapping-first ``CachingProvider``, and aggregates their results into the legacy
``{term: {keyword: {count, gndid, ddc, dk}}}`` shape so existing callers (classic
pipeline, CLI, GUI, MCP) keep working unchanged.

The ``SuggesterType`` enum is retired — callers pass provider-id strings
("lobid" | "swb" | "catalog"; "all" = all three legacy GND-keyword sources).
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from .base_suggester import BaseSuggester, BaseSuggesterError

# Legacy GND-keyword sources MetaSuggester has always combined. "all" maps here
# (gnd_local/finc are not part of the classic keyword flow). - Claude Generated
_DEFAULT_GND_PROVIDERS = ["lobid", "swb", "catalog"]


class MetaSuggesterError(BaseSuggesterError):
    """Exception raised for errors in the MetaSuggester."""

    pass


class MetaSuggester(BaseSuggester):
    """Combine GND-keyword results from one or more registered providers.

    Mapping-first caching (incl. write-back + the F-4 display-count restore) is
    applied per ``GND_KEYWORDS`` provider via ``CachingProvider``.
    """

    def __init__(
        self,
        providers: Optional[Union[str, List[str]]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        catalog_token: str = "",
        debug: bool = False,
        catalog_search_url: str = "",
        catalog_details: str = "",
        enable_mapping_search: bool = True,
        mapping_max_age_hours: int = 24,
    ):
        """Initialize the meta suggester.

        Args:
            providers: provider id or list of ids ("lobid"|"swb"|"catalog");
                ``None`` or "all" → all three legacy GND-keyword sources.
            data_dir: optional storage directory.
            catalog_token / catalog_search_url / catalog_details: catalog config.
            debug: verbose logging.
            enable_mapping_search: wrap GND_KEYWORDS providers with the cache.
            mapping_max_age_hours: cache freshness window.
        """
        super().__init__(data_dir, debug)

        self.logger = logging.getLogger("meta_suggester")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        from ...core.unified_knowledge_manager import UnifiedKnowledgeManager

        self.ukm = UnifiedKnowledgeManager()
        self.enable_mapping_search = enable_mapping_search
        self.mapping_max_age_hours = mapping_max_age_hours
        self.debug_mapping = debug

        self.provider_ids = self._resolve_provider_ids(providers)
        provider_config = {
            "token": catalog_token,
            "catalog_token": catalog_token,
            "catalog_search_url": catalog_search_url,
            "catalog_details": catalog_details,
            "debug": debug,
        }

        from ...core.search import SearchCapability, get_provider
        from ...core.search.caching import CachingProvider

        # id -> provider used for search (caching-wrapped for GND_KEYWORDS)
        self.providers: Dict[str, Any] = {}
        # id -> underlying BaseSuggester (compat: direct access / prepare / MCP)
        self.suggesters: Dict[str, Any] = {}

        for pid in self.provider_ids:
            try:
                cls = get_provider(pid)
            except KeyError:
                self.logger.warning(f"Unknown provider '{pid}', skipping")
                continue
            inst = cls(**provider_config)
            raw = getattr(inst, "suggester", None)
            if raw is not None:
                self.suggesters[pid] = raw
                try:
                    raw.currentTerm.connect(self.currentTerm)
                except Exception:
                    pass
            if (
                SearchCapability.GND_KEYWORDS in getattr(cls, "capabilities", set())
                and self.enable_mapping_search
            ):
                inst = CachingProvider(
                    inst, ukm=self.ukm, max_age_hours=self.mapping_max_age_hours
                )
            self.providers[pid] = inst

    @staticmethod
    def _resolve_provider_ids(providers: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalise the ``providers`` argument to a list of provider ids."""
        if providers is None:
            return list(_DEFAULT_GND_PROVIDERS)
        if isinstance(providers, str):
            providers = [providers]
        ids: List[str] = []
        for p in providers:
            pid = str(p).lower()
            if pid == "all":
                for d in _DEFAULT_GND_PROVIDERS:
                    if d not in ids:
                        ids.append(d)
            elif pid not in ids:
                ids.append(pid)
        return ids

    def raw_suggester(self, provider_id: Optional[str] = None):
        """Return an underlying ``BaseSuggester`` for direct ``search_type``
        passthrough (cache-bypassing), used by the MCP layer. ``None`` returns the
        first available. - Claude Generated"""
        if provider_id is not None:
            return self.suggesters.get(provider_id)
        return next(iter(self.suggesters.values()), None)

    def prepare(self, force_download: bool = False) -> None:
        """Prepare all underlying suggesters (best-effort)."""
        for pid, suggester in self.suggesters.items():
            try:
                self.logger.debug(f"Preparing {pid} suggester")
                suggester.prepare(force_download)
            except Exception as e:
                self.logger.warning(f"prepare() failed for {pid}: {e}")

    def search(self, terms: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Search all providers and aggregate into the legacy GND-keyword shape.

        Mapping-first caching + the F-4 display-count restore happen inside each
        provider's ``CachingProvider`` wrapper.
        """
        from ...core.search import SearchCapability

        combined_results: Dict[str, Dict[str, Dict[str, Any]]] = {t: {} for t in terms}
        self.last_errors = {}

        for pid, provider in self.providers.items():
            self.logger.debug(f"Searching with {pid} provider")
            try:
                result = provider.search(
                    SearchCapability.GND_KEYWORDS, terms, progress=self.currentTerm.emit
                )
            except Exception as e:
                self.logger.error(f"Error searching with {pid} provider: {e}")
                for term in terms:
                    self.last_errors[f"{pid}:{term}"] = str(e)
                continue

            per_term = result.to_gnd_keywords()
            for term in terms:
                self._merge_suggester_results(
                    combined_results, {term: per_term.get(term, {})}, term
                )
            for term, message in (result.errors or {}).items():
                self.last_errors.setdefault(f"{pid}:{term}", message)

        if self.last_errors:
            self.logger.warning(
                f"Search completed with {len(self.last_errors)} source failure(s): "
                f"{sorted(self.last_errors)} — empty results for these are NOT confirmed misses"
            )

        return combined_results

    def _merge_suggester_results(
        self, combined_results: Dict, suggester_results: Dict, term: str
    ):
        """Merge one provider's per-term results into the combined dict.

        ``count`` is max-merged (pool/ranking semantics, unchanged). The optional
        display-only ``display_count`` (F-4) is max-merged separately and only
        carried when a source provides it. - Claude Generated
        """
        if term not in suggester_results:
            return

        for keyword, data in suggester_results[term].items():
            if keyword not in combined_results[term]:
                entry = {
                    "count": data.get("count", 1),
                    "gndid": data.get("gndid", set()),
                    "ddc": data.get("ddc", set()),
                    "dk": data.get("dk", set()),
                }
                if data.get("display_count") is not None:
                    entry["display_count"] = data["display_count"]
                combined_results[term][keyword] = entry
            else:
                existing = combined_results[term][keyword]
                existing["count"] = max(existing["count"], data.get("count", 1))
                existing["gndid"].update(data.get("gndid", set()))
                existing["ddc"].update(data.get("ddc", set()))
                existing["dk"].update(data.get("dk", set()))
                dc = data.get("display_count")
                if dc is not None:
                    cur = existing.get("display_count")
                    existing["display_count"] = dc if cur is None else max(cur, dc)

    def search_unified(self, terms: List[str]) -> List[List]:
        """Get unified search results: ``[keyword, gnd_id, ddc, dk, count, term]``."""
        return self.get_unified_results(terms)
