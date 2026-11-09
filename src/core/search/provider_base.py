"""Shared base for providers that wrap a legacy ``BaseSuggester`` - Claude Generated.

The lobid / swb / catalog sources already implement the GND-keyword
``search(terms, ...) -> {term: {keyword: {...}}}`` contract; these providers are
thin adapters that lazily build the suggester and convert its output to a typed
:class:`ProviderResult`. Behaviour is unchanged (P1 is facade-preserving).

Public plugin API: third-party GND-keyword plugins subclass
``SuggesterBackedProvider`` via the absolute import
``from src.core.search.provider_base import SuggesterBackedProvider`` — see
``docs/plugin_authoring.md`` and the ``providers/lobid/`` blueprint dir.

:class:`NetworkSourceMixin` holds what every network source shares, with or
without a suggester (kvk and sru have none): the instance id the source-health
gate counts under, the cache-only switch, and the WP2 raw-cache read/write seam.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from src.core.plugins.schema import ConfigField, PluginDoc, availability_ok

from src.core.search import source_health
from src.core.search.cache_only import CACHE_MISS_MESSAGE, CACHE_ONLY_KEY, is_cache_only
from src.core.search.provider import ProviderResult, SearchCapability, raw_cache_params_for
from src.utils.error_visibility import log_caught

_logger = logging.getLogger(__name__)


class NetworkSourceMixin:
    """Instance identity, cache-only switch and raw-cache seam - Claude Generated.

    Expects ``self._config`` (the instance settings) and a class attribute
    ``id``. ``search.factory.build_provider`` injects ``_source_id`` (the
    instance id), ``_cache_raw`` and ``_ukm_ref`` after construction; a provider
    built directly falls back to its type id and the defaults below.
    """

    _source_id: Optional[str] = None
    _cache_raw: Optional[bool] = None
    _ukm_ref: Any = None

    @property
    def source_id(self) -> str:
        """Key of this source in ``source_health``: the instance id - Claude Generated"""
        return self._source_id or getattr(self, "id", "") or ""

    @property
    def cache_only(self) -> bool:
        """Whether this instance answers from local caches only, without network.

        The per-instance ``cache_only`` setting (``src/core/search/cache_only.py``),
        honoured only when the type declares the field: a type without a cache
        path (gnd_local) ignores a hand-written key. - Claude Generated
        """
        if not is_cache_only(self._config):
            return False
        try:
            fields = type(self).config_fields()
        except Exception:
            return False
        return any(getattr(f, "key", None) == CACHE_ONLY_KEY for f in fields)

    def _source_down_errors(self, query: List[str]) -> Dict[str, str]:
        """Per-term errors for a source that is down (``source_health``) - Claude Generated"""
        message = source_health.down_message(self.source_id)
        return {term: message for term in query}

    # --- cache-only mode (read seam of the WP2 raw cache) -------------------
    def _cached_raw(self, source: str, term: str, params: Dict[str, Any]) -> Any:
        """Parsed raw-cache entry for ``(source, term, params)``, whatever its age.

        ``max_age_hours=None`` disables the freshness gate: in cache-only mode a
        stale entry is still an answer. ``None`` on a miss or an unreadable
        entry. - Claude Generated
        """
        try:
            hit = self._ukm().get_raw_response(source, term, params, max_age_hours=None)
        except Exception as e:
            log_caught(_logger, e, "cache-only raw-cache read", detail=f"{source}:{term}")
            return None
        if not hit:
            return None
        try:
            return json.loads(hit["raw_json"])
        except (KeyError, TypeError, ValueError):
            return None

    # --- WP2 raw-first dual-write ----------------------------------------
    def _ukm(self):
        """Lazily resolve the UnifiedKnowledgeManager singleton - Claude Generated"""
        if self._ukm_ref is None:
            from src.core.unified_knowledge_manager import UnifiedKnowledgeManager

            self._ukm_ref = UnifiedKnowledgeManager()
        return self._ukm_ref

    def _cache_raw_enabled(self) -> bool:
        """Whether raw source responses should be written for this provider.

        The per-instance ``cache_responses`` setting (tri-state ``auto``/``on``/
        ``off``, or a legacy bool) governs; ``auto``/absent follows the factory-
        injected global (``SystemConfig.enable_response_cache``) → default True.
        - Claude Generated
        """
        from src.core.plugins.schema import cache_pref_enabled

        global_default = self._cache_raw if self._cache_raw is not None else True
        return cache_pref_enabled(
            self._config.get("cache_responses"), global_enabled=global_default
        )

    def _store_records_raw(
        self, source: str, query: List[str], params: Dict[str, Any], raw_by_term: Any
    ) -> None:
        """Dual-write a record-search response (``{term: [records]}``) to the raw cache.

        For record searches (title records) where the raw is the returned record
        list per term rather than a suggester ``last_raw`` blob. Best-effort, never
        raises. - Claude Generated
        """
        if not self._cache_raw_enabled():
            return
        try:
            ukm = self._ukm()
        except Exception:
            return
        for term in query:
            recs = (raw_by_term or {}).get(term)
            if recs is None:
                continue
            try:
                blob = json.dumps(
                    {"records": recs, "totalItems": len(recs)},
                    ensure_ascii=False, default=str,
                )
            except (TypeError, ValueError):
                continue
            ukm.store_raw_response(source, term, params, blob)


class SuggesterBackedProvider(NetworkSourceMixin):
    """Base for GND-keyword providers wrapping a ``BaseSuggester``.

    Subclasses set ``id`` / ``label`` / ``capabilities`` and implement
    :meth:`_build_suggester`. The underlying suggester is built lazily so importing
    the provider module (for registration) does not construct network clients.
    """

    id: str = ""
    label: str = ""
    capabilities: set = set()
    # WP2 raw-cache key params (see provider.raw_cache_params_for). Override when
    # the source's results depend on more than ``search_type``. - Claude Generated
    raw_cache_param_keys: tuple = ("search_type",)

    def __init__(self, **config: Any):
        self._config = config or {}
        self._suggester = None
        # WP2 raw-first cache: write policy + ukm handle injected by
        # search.factory.build_provider. Kept off the suggester constructor via
        # these private attrs. ``_cache_raw`` None ⇒ default-on. - Claude Generated
        self._cache_raw: Optional[bool] = None
        self._ukm_ref: Any = None
        # Instance id for source_health, injected by the factory. - Claude Generated
        self._source_id: Optional[str] = None

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        """Declarative config schema — one source for the settings form + gating.

        Default: no user-facing config (lobid/swb/gnd_local). Network sources
        override with their endpoints/tokens.
        """
        return []

    @classmethod
    def doc(cls) -> PluginDoc:
        """Natural-language self-description (what it does + input/output).

        Every provider must override this — the framework surfaces it in the
        settings UI and to the agent. The base returns an empty doc so a missing
        override is detectable (``PluginDoc.is_complete()``).
        """
        return PluginDoc()

    def _build_suggester(self):  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def suggester(self):
        if self._suggester is None:
            self._suggester = self._build_suggester()
        return self._suggester

    def is_available(self, cfg: Any = None) -> bool:
        """Available unless a ``gates_availability`` config field is unset.

        Derived from :meth:`config_fields`, so a provider that declares a gating
        field (e.g. catalog ``token``) needs no bespoke override.
        """
        return availability_ok(type(self).config_fields(), self._config)

    def _require(self, capability: SearchCapability) -> None:
        if capability not in self.capabilities:
            raise ValueError(
                f"Provider '{self.id}' does not support {capability}; "
                f"declares {sorted(c.value for c in self.capabilities)}"
            )

    def _wire_progress(self, progress: Optional[Callable[[str], None]]) -> None:
        """Connect the suggester's ``currentTerm`` signal to ``progress`` if given.

        Best-effort: works for both the real Qt signal and the no-Qt dummy in
        ``base_suggester``. Providers are short-lived (built per use), so repeated
        connects are not a concern.
        """
        if progress is None:
            return
        try:
            self.suggester.currentTerm.connect(progress)
        except Exception:
            pass

    def _gnd_search(
        self,
        query: List[str],
        progress: Optional[Callable[[str], None]],
        **suggester_kwargs: Any,
    ) -> ProviderResult:
        """Run the wrapped suggester and wrap its GND-keyword output."""
        if self.cache_only:
            return self._gnd_search_cache_only(query, suggester_kwargs)
        if source_health.is_down(self.source_id):
            # Skipped for this run: unknown per term, never an empty hit.
            # - Claude Generated
            errors = self._source_down_errors(query)
            self._count_usage(query, errors, {})
            return ProviderResult.from_gnd_keywords(
                {term: {} for term in query}, errors=errors
            )
        self._wire_progress(progress)
        raw = self.suggester.search(list(query), **suggester_kwargs)
        errors = dict(getattr(self.suggester, "last_errors", {}) or {})
        self._store_raw_responses(query, suggester_kwargs)
        self._count_usage(query, errors, raw)
        return ProviderResult.from_gnd_keywords(raw, errors=errors)

    def _count_usage(self, query: List[str], errors: Dict[str, Any], hits: Any,
                     *, cache_only: bool = False) -> None:
        """Per-run source overview (``source_usage``), one entry per term - Claude Generated"""
        from src.core.search import source_usage

        source_usage.record_terms(self.source_id, list(query), errors=errors,
                                  hits=hits if isinstance(hits, dict) else {},
                                  cache_only=cache_only)

    def _gnd_search_cache_only(
        self, query: List[str], suggester_kwargs: Dict[str, Any]
    ) -> ProviderResult:
        """``GND_KEYWORDS`` from the raw cache alone (cache-only mode).

        Reads the entry the live path stored (same key via ``raw_cache_params_for``)
        and reduces it with the suggester's pure ``transform``; the suggester's
        ``search`` is never called, so there is no network request. Nothing is
        written back: a rewrite would re-stamp a stale entry as fresh. A term
        without an entry carries the cache-only marker in ``errors`` (unknown,
        not "no match"). - Claude Generated
        """
        params = raw_cache_params_for(
            self.id,
            search_type=suggester_kwargs.get("search_type", "kw"),
            max_pages=suggester_kwargs.get("max_pages", 5),
            facets=suggester_kwargs.get("facets"),
        )
        transform = getattr(self.suggester, "transform", None)
        per_term: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        for term in query:
            raw = self._cached_raw(self.id, term, params) if transform else None
            reduced = None
            if raw is not None:
                try:
                    reduced = transform(raw)
                except Exception as e:  # an unreadable blob is a miss, not a crash
                    log_caught(_logger, e, "cache-only transform", detail=f"{self.id}:{term}")
            if reduced is None:
                per_term[term] = {}
                errors[term] = CACHE_MISS_MESSAGE
            else:
                per_term[term] = reduced
        self._count_usage(query, errors, per_term, cache_only=True)
        return ProviderResult.from_gnd_keywords(per_term, errors=errors)

    def _store_raw_responses(
        self, query: List[str], suggester_kwargs: Dict[str, Any]
    ) -> None:
        """Dual-write verbatim source responses (WP2 P1). Best-effort, never raises.

        Reads ``suggester.last_raw`` (``{term: raw_json_str}``); providers whose
        suggester does not yet expose it (pre-P3 swb/catalog) are simply skipped
        — the seam is additive. - Claude Generated
        """
        if not self._cache_raw_enabled():
            return
        last_raw = getattr(self.suggester, "last_raw", None)
        if not last_raw:
            return
        last_status = getattr(self.suggester, "last_http_status", {}) or {}
        params = raw_cache_params_for(
            self.id,
            search_type=suggester_kwargs.get("search_type", "kw"),
            max_pages=suggester_kwargs.get("max_pages", 5),
            facets=suggester_kwargs.get("facets"),
        )
        try:
            ukm = self._ukm()
        except Exception:
            return
        for term in query:
            blob = last_raw.get(term)
            if blob is None:
                continue
            ukm.store_raw_response(
                self.id, term, params, blob,
                http_status=last_status.get(term),
            )
