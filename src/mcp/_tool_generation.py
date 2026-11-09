"""Provider/source tool generation for the MCP ToolRegistry - Claude Generated.

Split out of ``tool_registry.py`` (WP cleanup D). Verbatim mixin extraction: the
methods stay on ``ToolRegistry`` via MRO, so no call site changes.

This is the tool-*generation* machinery — the ``_make_*_handler`` factories that
build one callable per enabled search/input/lookup instance, plus the agent-view
and raw-cache plumbing they wrap. It is a self-contained layer that read as
noise inside the class's own tool *handlers* (``_handle_*``). The factories call
back into the rest of the class (``_provider_for``, ``_get_knowledge_manager``,
the serialisers) — that stays resolvable via ``self`` because it is one class
across two files, which is the whole point of the mixin split.

Method-local imports (``build_provider``, ``raw_cache_params_for``,
``underlying_suggester``, …) are deliberate and travel with the code.
"""

from __future__ import annotations

import json
import logging

from src.mcp.mcp_types import ToolDefinition

logger = logging.getLogger(__name__)


class ToolGenerationMixin:
    """Provider/source tool factories. Mixed into :class:`ToolRegistry`."""

    def _input_instances(self):
        """Enabled input-source instances; fall back to one per registered type."""
        cfg = None
        try:
            cm = self._config_manager
            if cm is None:
                from src.utils.config_manager import ConfigManager
                cm = ConfigManager()
            cfg = cm.load_config()
        except Exception:
            cfg = None
        if cfg is not None:
            return cfg.enabled_instances_for("input_source")
        from src.utils.config_models import PluginInstanceConfig
        from src.utils.input_sources import list_input_sources
        return [
            PluginInstanceConfig(instance_id=sid, category="input_source", provider_id=sid)
            for sid in list_input_sources()
        ]

    def _generated_input_tools(self):
        """Build ``(ToolDefinition, handler)`` for each enabled input source that
        declares an ``InputToolSpec`` (e.g. the three DOI resolvers → individually
        callable so an agent can compare sources). Sources without a spec
        (text/file/pdf/image/url_fetch) are not exposed here. - Claude Generated"""
        from src.utils.input_sources import get_input_source

        by_type = {}
        for inst in self._input_instances():
            by_type.setdefault(inst.provider_id, []).append(inst)

        tools = []
        for provider_id, instances in by_type.items():
            try:
                cls = get_input_source(provider_id)
            except KeyError:
                continue
            spec_fn = getattr(cls, "mcp_tool_spec", None)
            spec = spec_fn() if callable(spec_fn) else None
            if spec is None:
                continue
            canonical = self._canonical_instance(instances)
            for inst in instances:
                name = spec.name if inst is canonical else self._instance_tool_name(spec.name, inst)
                td = ToolDefinition(
                    name=name,
                    description=self._describe_with_hint(spec.description, inst),
                    parameters={
                        "type": "object",
                        "properties": {
                            spec.param: {
                                "type": "string",
                                "description": spec.param_description or "Eingabewert",
                            }
                        },
                        "required": [spec.param],
                    },
                )
                tools.append((td, self._make_input_handler(spec, inst)))
        return tools

    def _response_cache_enabled(self) -> bool:
        """Read the SystemConfig.enable_response_cache master switch (default True)."""
        try:
            cm = self._config_manager
            if cm is None:
                from src.utils.config_manager import ConfigManager
                cm = ConfigManager()
            return bool(getattr(cm.load_config().system_config, "enable_response_cache", True))
        except Exception:
            return True

    def _make_input_handler(self, spec, inst):
        def handler(**kwargs):
            try:
                from src.utils.input_sources import get_input_source

                value = kwargs.get(spec.param)
                if value is None and kwargs:
                    value = next(iter(kwargs.values()))
                # WP2 P5: cache the verbatim source response (e.g. DOI metadata).
                # Gated by the tool's `cacheable`, the master switch, and the
                # per-plugin `cache_responses` setting (auto/on/off). - Claude Generated
                from src.core.plugins.schema import cache_pref_enabled

                cacheable = (
                    bool(value)
                    and getattr(spec, "cacheable", True)
                    and cache_pref_enabled(
                        (inst.settings or {}).get("cache_responses"),
                        global_enabled=self._response_cache_enabled(),
                    )
                )
                if cacheable:
                    hit = self._get_knowledge_manager().get_raw_response(
                        inst.provider_id, str(value), {}
                    )
                    if hit:
                        return hit["raw_json"]
                cls = get_input_source(inst.provider_id)
                try:
                    source = cls(**dict(inst.settings or {}))
                except TypeError:
                    source = cls()
                if hasattr(source, "mcp_execute"):
                    result = json.dumps(source.mcp_execute(value), ensure_ascii=False, default=str)
                else:
                    text, info, method = source.extract(value)
                    result = json.dumps(
                        {"text": text, "source_info": info, "method": method}, ensure_ascii=False
                    )
                # Don't cache error payloads (they should be retried, not pinned).
                if cacheable and '"error"' not in result:
                    try:
                        self._get_knowledge_manager().store_raw_response(
                            inst.provider_id, str(value), {}, result
                        )
                    except Exception:
                        pass
                return result
            except Exception as exc:
                return json.dumps({"error": f"{inst.instance_id}: {exc}"})

        return handler

    def _lookup_instances(self):
        """Enabled lookup-plugin instances.

        Search parity (WP P5): a **readable** config is authoritative — its enabled
        set is honoured *even when empty* (all lookups disabled → no lookup tools),
        mirroring ``factory.enabled_gnd_provider_ids``' ``None``-vs-``[]`` split. Only
        an **unreadable** config falls back to one synthetic instance per registered
        type (so tools still exist on a config error). ``ensure_lookup_instances``
        seeds every type on load, so the readable-but-empty case means the operator
        deliberately disabled them, not an unmigrated config. - Claude Generated"""
        import src.utils.lookups  # noqa: F401 — registers the category + plugins
        from src.utils.lookups import list_lookups

        cfg = self._alima_config()
        if cfg is not None:
            return cfg.enabled_instances_for("lookup")  # readable → honour (incl. empty)
        from src.utils.config_models import PluginInstanceConfig

        return [
            PluginInstanceConfig(instance_id=lid, category="lookup", provider_id=lid)
            for lid in list_lookups()
        ]

    def _generated_lookup_tools(self):
        """Build ``(ToolDefinition, handler)`` for each enabled lookup plugin's
        ``LookupToolSpec`` (e.g. rvk_search / rvk_validate). - Claude Generated"""
        import src.utils.lookups  # noqa: F401
        from src.utils.lookups import lookup_tool_specs

        specs_by_provider = {}
        for spec in lookup_tool_specs():
            specs_by_provider.setdefault(spec.provider_id, []).append(spec)

        by_type = {}
        for inst in self._lookup_instances():
            by_type.setdefault(inst.provider_id, []).append(inst)

        tools = []
        used = set()
        for provider_id, instances in by_type.items():
            specs = specs_by_provider.get(provider_id)
            if not specs:
                continue
            instances = [inst for inst in instances if self._lookup_instance_available(provider_id, inst)]
            if not instances:
                continue
            canonical = self._canonical_instance(instances)
            for inst in instances:
                for spec in specs:
                    name = spec.name if inst is canonical else self._instance_tool_name(spec.name, inst)
                    if name in used:
                        name = self._instance_tool_name(spec.name, inst)
                        if name in used:
                            continue
                    used.add(name)
                    tools.append((
                        ToolDefinition(
                            name=name,
                            description=self._describe_with_hint(spec.description, inst),
                            parameters=spec.parameters,
                        ),
                        self._make_lookup_handler(spec, inst),
                    ))
        return tools

    def _lookup_instance_available(self, provider_id, inst):
        """Optional availability gate for lookup plugins.

        A lookup class that declares ``is_available()`` (e.g. ``geo_sachdb``
        without its database file) gets no tools while it reports False. Classes
        without the method are not built here and keep their tools unchanged.
        - Claude Generated"""
        from src.utils.lookups import get_lookup

        try:
            cls = get_lookup(provider_id)
        except KeyError:
            return True
        if not hasattr(cls, "is_available"):
            return True
        try:
            return bool(self._lookup_for(inst).is_available())
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not break tool generation
            from src.utils.error_visibility import log_caught

            log_caught(logger, exc, f"lookup availability {provider_id}/{getattr(inst, 'instance_id', '?')}")
            return False

    def _make_lookup_handler(self, spec, inst):
        """Handler for a lookup tool: build the plugin, call its method, and cache
        the raw response (gated by the per-plugin cache setting). - Claude Generated"""
        def handler(**kwargs):
            try:
                from src.core.plugins.schema import cache_pref_enabled

                # Only pass the args this tool declares (avoid TypeErrors from
                # stray kwargs the LLM/harness may add). - Claude Generated
                props = (spec.parameters or {}).get("properties", {})
                args = {k: v for k, v in kwargs.items() if k in props and v is not None}

                key = args.get(spec.cache_key_param) if spec.cache_key_param else None
                cache_params = {k: v for k, v in args.items() if k != spec.cache_key_param}
                cacheable = (
                    bool(key)
                    and spec.cacheable
                    and cache_pref_enabled(
                        (inst.settings or {}).get("cache_responses"),
                        global_enabled=self._response_cache_enabled(),
                    )
                )
                from src.core.search import source_usage

                source_usage.set_label(inst.instance_id, getattr(inst, "label", "") or inst.provider_id)
                if cacheable:
                    hit = self._get_knowledge_manager().get_raw_response(
                        spec.name, str(key), cache_params
                    )
                    if hit:
                        source_usage.record(inst.instance_id, source_usage.CACHE,
                                            kind=source_usage.KIND_LOOKUP)
                        return hit["raw_json"]

                plugin = self._lookup_for(inst)
                try:
                    result = getattr(plugin, spec.method)(**args)
                except Exception:
                    source_usage.record(inst.instance_id, source_usage.CALL,
                                        kind=source_usage.KIND_LOOKUP, error=True)
                    raise
                out = json.dumps(result, ensure_ascii=False, default=str)
                # Per-run source overview: one entry per lookup call. Whether
                # the plugin asked a server or read a local file is not known
                # here, so it is counted as a call, not as Netz. - Claude Generated
                source_usage.record(inst.instance_id, source_usage.CALL,
                                    kind=source_usage.KIND_LOOKUP,
                                    error='"error"' in out)
                if cacheable and '"error"' not in out:
                    try:
                        self._get_knowledge_manager().store_raw_response(
                            spec.name, str(key), cache_params, out
                        )
                    except Exception:
                        pass
                return out
            except Exception as exc:
                return json.dumps({"error": f"{inst.instance_id}: {exc}"})

        return handler

    def _source_transform(self, source):
        """Resolve a GND-keyword source id to its ``transform(raw)`` callable.

        Builds the source's provider through the factory and reads the pure
        ``transform`` off its underlying suggester (the transform is config-
        independent). Returns None if that source is unknown/unavailable.
        - Claude Generated
        """
        try:
            from src.core.search.service import underlying_suggester

            inst = self._instance_for(source)
            if inst is None:
                return None
            sugg = underlying_suggester(self._provider_for(inst, cache=False))
            return getattr(sugg, "transform", None)
        except Exception as e:
            logger.debug(f"transform for '{source}' unavailable: {e}")
        return None

    @staticmethod
    def _cache_only_block(provider, term_errors):
        """The ``cache_only`` block of a search-tool result, ``None`` when off.

        Tells the agent that the source answered from the local cache only and
        which terms had no entry, so an empty result is not read as "the
        catalog has nothing". ``term_errors`` is keyed by term. - Claude Generated
        """
        if getattr(provider, "cache_only", False) is not True:
            return None
        from src.core.search.cache_only import is_cache_miss

        return {
            "active": True,
            "missing_terms": [t for t, m in (term_errors or {}).items() if is_cache_miss(m)],
            "note": "Nur Cache: keine Netzabfrage. Begriffe ohne Cache-Eintrag "
                    "liefern keine Treffer; das belegt nicht, dass der Katalog sie nicht hat.",
        }

    @staticmethod
    def _source_down_block(term_errors):
        """The ``source_down`` block of a search-tool result, ``None`` when no
        term was skipped by ``source_health``. ``term_errors`` is keyed by term.
        - Claude Generated
        """
        from src.core.search.source_health import FAILURE_THRESHOLD, is_source_down

        skipped = [t for t, m in (term_errors or {}).items() if is_source_down(m)]
        if not skipped:
            return None
        return {
            "skipped_terms": skipped,
            "note": f"Quelle nach {FAILURE_THRESHOLD} Verbindungsfehlern für diesen Lauf "
                    "übersprungen, keine Netzabfrage. Leere Treffer für diese Begriffe "
                    "sind unbekannt, nicht „kein Treffer“.",
        }

    def _attach_status_blocks(self, out, provider, term_errors):
        """Add the ``cache_only`` and ``source_down`` blocks when they apply - Claude Generated"""
        block = self._cache_only_block(provider, term_errors)
        if block:
            out["cache_only"] = block
        down = self._source_down_block(term_errors)
        if down:
            out["source_down"] = down

    def _make_search_handler(self, spec, inst):
        """Handler for a canonical instance: the nuanced, factory-backed path.

        Unknown shapes fall through to the generic instance handler rather than
        raising — every canonical spec reaches this, so one odd plugin must not
        take the whole tool list down. - Claude Generated
        """
        if spec.result_shape == "gnd_keywords":
            return self._make_gnd_keywords_handler(spec, inst)
        if spec.result_shape == "title_records":
            return self._make_title_records_handler(spec, inst)
        if spec.result_shape == "finc":
            return self._make_finc_handler(spec, inst)
        return self._make_instance_handler(spec, inst)

    def _make_finc_handler(self, spec, inst):
        """Primary finc handler, factory-backed (WP P2.2).

        Builds FincProvider from the instance config via ``_provider_for`` — so a
        copied finc plugin uses its own backend, not the built-in one. The
        availability→facet_avail translation and the dk/rvk auto-facets now live
        in ``FincProvider.search``; the web_url reconstruction stays here but reads
        the catalog base from the *instance* (self-contained), falling back to the
        global catalog config only for instances migrated before the
        ``catalog_web_record_url`` field existed. - Claude Generated
        """
        from src.core.search.provider import SearchCapability

        def handler(terms, search_type="kw", filters=None, facets=None,
                    limit=20, availability=None):
            try:
                provider = self._provider_for(inst)
            except Exception as e:  # pragma: no cover - build failure
                return json.dumps({"source": "finc", "error": str(e)})
            if not (hasattr(provider, "is_available") and provider.is_available()):
                return json.dumps(
                    {"error": "finc not available: base_url not configured on the finc plugin"}
                )
            try:
                res = provider.search(
                    SearchCapability.TITLE_RECORDS,
                    list(terms or []),
                    search_type=search_type or "kw",
                    filters=filters,
                    limit=limit,
                    facets=facets,
                    availability=availability,
                )
            except Exception as e:
                logger.error(f"search_finc failed: {e}")
                return json.dumps({"source": "finc", "error": str(e)})
            results = res.to_finc_records()
            self._count_terms(provider, terms, res.errors, records={
                t: len(d.get("records") or []) for t, d in results.items()})
            cat_base = self._finc_record_base(provider)
            if cat_base:
                for term_data in results.values():
                    for rec in term_data.get("records", []):
                        if not rec.get("web_url") and rec.get("id"):
                            rec["web_url"] = f"{cat_base}/{rec['id']}"
            out = {"source": "finc", "results": results, "errors": dict(res.errors or {})}
            self._attach_status_blocks(out, provider, res.errors)
            return json.dumps(out, ensure_ascii=False)

        return handler

    def _finc_record_base(self, provider):
        """Base of the catalog link for finc records without ``web_url``: the
        instance's own ``catalog_web_record_url``, else the catalog instance's
        OPAC base. - Claude Generated"""
        cat_base = ""
        try:
            cat_base = provider.catalog_web_record_base()
        except Exception:
            pass
        if not cat_base:
            # Pre-P2.2 finc instances carry no own base — fall back to the
            # catalog instance's OPAC base (WP P7: was the CatalogConfig
            # mirror). - Claude Generated
            try:
                from src.core.search.factory import catalog_web_bases

                # Only via the injected config manager — never let a failed
                # load reach for the global one behind the caller's back.
                _cfg = self._alima_config()
                if _cfg is not None:
                    cat_base = (catalog_web_bases(_cfg)[0] or "").rstrip("/")
            except Exception:
                pass
        return cat_base

    def _make_gnd_keywords_handler(self, spec, inst):
        """GND-keyword tool handler built on the search factory (no MetaSuggester).

        Default cached options use the factory's mapping-first ``CachingProvider``
        exactly as the classic pipeline; any non-default option (title /
        custom max_pages) bypasses the cache through the raw suggester (with the
        WP2 dual-write). Non-cached providers (catalog) go straight to the raw
        suggester. - Claude Generated
        """
        def handler(terms, search_type="kw", max_pages=5, **_ignore):
            from src.core.search.provider import SearchCapability
            from src.core.search.service import underlying_suggester

            raw_id = spec.provider_id if spec.cached else None
            cache_only = False
            try:
                base_provider = self._provider_for(inst, cache=False)
                if hasattr(base_provider, "is_available") and not base_provider.is_available():
                    return json.dumps(
                        {"error": spec.unavailable_message or f"{spec.source_label} not available"}
                    )
                cache_only = getattr(base_provider, "cache_only", False) is True
                if spec.cached:
                    opts = {"search_type": search_type, "max_pages": max_pages}
                    is_default = all(opts.get(k) == v for k, v in spec.default_opts.items())
                    if is_default or cache_only:
                        # Cache-only: the provider answers from the raw cache for
                        # any option; the bare suggester would go to the network.
                        # - Claude Generated
                        kw = {}
                        if not is_default:
                            kw = {"search_type": search_type}
                            if "max_pages" in spec.default_opts:
                                kw["max_pages"] = max_pages
                        res = self._provider_for(inst, cache=True).search(
                            SearchCapability.GND_KEYWORDS, list(terms), **kw
                        )
                        results = res.to_gnd_keywords()
                        term_errors = dict(res.errors or {})
                        errors = {
                            f"{spec.provider_id}:{t}": m for t, m in term_errors.items()
                        }
                    else:
                        kw = {"search_type": search_type}
                        if "max_pages" in spec.default_opts:
                            kw["max_pages"] = max_pages
                        sugg = underlying_suggester(base_provider)
                        results = sugg.search(list(terms), **kw)
                        # Bypasses the provider fetch seam → dual-write raw here so
                        # non-default searches also populate the raw cache. - Claude Generated
                        self._store_suggester_raw(spec.provider_id, terms, kw, sugg, inst=inst)
                        errors = {}
                        term_errors = dict(getattr(sugg, "last_errors", {}) or {})
                        self._count_terms(base_provider, terms, term_errors, hits=results)
                elif cache_only:
                    # Cache-only: the provider answers from the raw cache; the
                    # bare suggester would go to the network. - Claude Generated
                    res = base_provider.search(
                        SearchCapability.GND_KEYWORDS, list(terms), search_type=search_type
                    )
                    results = res.to_gnd_keywords()
                    errors = dict(res.errors or {})
                    term_errors = errors
                else:
                    sugg = underlying_suggester(base_provider)
                    results = sugg.search(list(terms), search_type=search_type)
                    errors = dict(getattr(sugg, "last_errors", {}) or {})
                    term_errors = errors
                    self._count_terms(base_provider, terms, term_errors, hits=results)
            except Exception as exc:
                logger.error("search tool '%s' failed: %s", spec.name, exc)
                return json.dumps({"error": str(exc)})
            if spec.add_gnd_urls:
                serialized = self._serialize_suggester_results(results)
            else:
                serialized = {}
                for term, keywords in results.items():
                    serialized[term] = {
                        kw: self._serialize_result_row(data)
                        for kw, data in keywords.items()
                    }
            out = {"source": spec.source_label, "results": serialized}
            if spec.include_errors:
                out["errors"] = errors
            self._attach_status_blocks(out, base_provider, term_errors)
            self._attach_agent_view(out, raw_id, terms, search_type, max_pages=max_pages,
                                    any_age=cache_only)
            return json.dumps(out, ensure_ascii=False)

        return handler

    def _agent_view_deriver(self, source):
        """Resolve a source id to its raw→agent-view function (None if unsupported).

        Transform-on-read dispatch for the WP2 raw cache, resolved exactly like the
        sibling ``_source_transform``: the provider is built through the factory and
        the optional ``transform_agent_view`` is read off its underlying suggester.
        A provider that declares one (incl. a copied plugin) participates; the rest
        return None. - Claude Generated
        """
        if not source:
            return None
        try:
            from src.core.search.service import underlying_suggester

            inst = self._instance_for(source)
            if inst is None:
                return None
            sugg = underlying_suggester(self._provider_for(inst, cache=False))
            return getattr(sugg, "transform_agent_view", None)
        except Exception as e:
            logger.debug(f"agent_view deriver for '{source}' unavailable: {e}")
        return None

    def _attach_agent_view(self, out, source, terms, search_type, max_pages=5, any_age=False):
        """Surface the full source view (member/totalItems) from the raw cache.

        Transform-on-read consumer of the WP2 raw cache: best-effort and additive
        — the reduced ``results`` block is untouched, and a raw miss simply omits
        the term. Only sources with an agent-view deriver participate; on the
        default cached path the raw was written by the fetch seam (miss) or exists
        from a prior fetch (mapping hit). - Claude Generated
        """
        from src.core.search.provider import raw_cache_params_for

        deriver = self._agent_view_deriver(source)
        if deriver is None:
            return
        try:
            km = self._get_knowledge_manager()
        except Exception:
            return
        # Same key the write seam used — a source keying on max_pages/facets would
        # silently miss against a hand-built {"search_type": …}. - Claude Generated
        params = raw_cache_params_for(source, search_type=search_type, max_pages=max_pages)
        # Cache-only instances read the raw cache at any age. - Claude Generated
        age = {"max_age_hours": None} if any_age else {}
        view = {}
        for term in terms:
            cached = km.get_raw_response(source, term, params, **age)
            if not cached:
                continue
            try:
                raw = json.loads(cached["raw_json"])
            except (ValueError, TypeError):
                continue
            view[term] = deriver(raw)
        if view:
            out["agent_view"] = view

    def _store_suggester_raw(self, source, terms, kw, suggester, inst=None):
        """Dual-write raw for the non-default MCP GND-keyword path (which uses the
        bare raw_suggester and bypasses the provider seam). Best-effort; keys the
        cache via the shared raw_cache_params_for so it matches the readers.

        Honours the same cache tri-state gate as every other write path
        (provider_base._store_raw_responses, the input and lookup handlers): the
        per-instance ``cache_responses`` (auto/on/off) resolved against the global
        ``enable_response_cache``. Without it an operator's "cache off" was ignored
        on this path. - Claude Generated
        """
        from src.core.plugins.schema import cache_pref_enabled

        settings = getattr(inst, "settings", None) or {}
        if not cache_pref_enabled(
            settings.get("cache_responses"), global_enabled=self._response_cache_enabled()
        ):
            return
        last_raw = getattr(suggester, "last_raw", None)
        if not last_raw:
            return
        from src.core.search.provider import raw_cache_params_for

        params = raw_cache_params_for(
            source, search_type=kw.get("search_type", "kw"),
            max_pages=kw.get("max_pages", 5),
        )
        try:
            km = self._get_knowledge_manager()
        except Exception:
            return
        last_status = getattr(suggester, "last_http_status", {}) or {}
        for term in terms:
            blob = last_raw.get(term)
            if blob is None:
                continue
            km.store_raw_response(source, term, params, blob, http_status=last_status.get(term))

    def _make_title_records_handler(self, spec, inst):
        """Catalog title-records handler built on the search factory (no mirror). - Claude Generated

        Two backends end up here. A suggester-backed source (Libero/catalog)
        answers through ``search_titles``; a standalone provider (kvk) has no
        suggester and answers through the provider contract itself. Before, a
        missing suggester was reported as "not available" — which is what the
        contract-only providers look like, even though they work. - Claude Generated
        """
        def handler(terms, search_type=None, max_results=None, **_ignore):
            from src.core.search.service import underlying_suggester

            # Defaults of the spec (sru: kw/10), else the former "title"/25.
            # - Claude Generated
            search_type = search_type or spec.default_opts.get("search_type") or "title"
            if max_results is None:
                max_results = spec.default_opts.get("max_results", 25)
            try:
                provider = self._provider_for(inst, cache=False)
                if hasattr(provider, "is_available") and not provider.is_available():
                    return json.dumps(
                        {"error": spec.unavailable_message or "catalog not available"}
                    )
                sugg = underlying_suggester(provider)
                # Cache-only goes through the provider contract (raw cache), the
                # suggester would query the catalog. - Claude Generated
                cache_only = getattr(provider, "cache_only", False) is True
                if sugg is not None and not cache_only:
                    results = sugg.search_titles(
                        list(terms), search_type=search_type, max_results=max_results
                    )
                    out = {"source": spec.source_label, "results": results}
                    errs = getattr(sugg, "last_errors", None)
                    self._count_terms(provider, terms, errs if isinstance(errs, dict) else None,
                                      records={t: len(r or []) for t, r in (results or {}).items()})
                    down = self._source_down_block(errs if isinstance(errs, dict) else None)
                    if down:
                        out["source_down"] = down
                else:
                    from src.core.search.provider import SearchCapability

                    res = provider.search(
                        SearchCapability.TITLE_RECORDS,
                        list(terms),
                        search_type=search_type,
                        max_results=max_results,
                    )
                    out = {
                        "source": spec.source_label,
                        "results": {
                            term: [it.record for it in items if it.record is not None]
                            for term, items in res.per_term.items()
                        },
                    }
                    self._count_terms(provider, terms, res.errors, records={
                        t: len(r) for t, r in out["results"].items()})
                    meta = {t: m for t, m in (res.per_term_meta or {}).items() if m}
                    if meta:
                        out["meta"] = meta
                    if spec.include_errors:
                        out["errors"] = dict(res.errors or {})
                    self._attach_status_blocks(out, provider, res.errors)
                    self._attach_notice(out, provider, res)
            except Exception as exc:
                logger.error("search tool '%s' failed: %s", spec.name, exc)
                return json.dumps({"error": str(exc)})
            return json.dumps(out, ensure_ascii=False)

        return handler

    @staticmethod
    def _count_terms(provider, terms, term_errors, *, hits=None, records=None):
        """Per-run source overview (``source_usage``) for a tool call that does
        not go through the provider base: one entry per term. Title searches
        pass ``records`` (count per term), GND searches ``hits``. Best-effort.
        - Claude Generated"""
        try:
            from src.core.search import source_usage

            inner = getattr(provider, "inner", provider)  # unwrap CachingProvider
            source_usage.record_terms(
                getattr(inner, "source_id", None), list(terms or []),
                errors=term_errors if isinstance(term_errors, dict) else None,
                hits=hits if isinstance(hits, dict) else None, records=records,
                cache_only=getattr(inner, "cache_only", False) is True,
                kind=source_usage.KIND_RECORDS if records is not None else source_usage.KIND_KEYWORDS,
            )
        except Exception as exc:  # noqa: BLE001 - the overview must not break a tool
            log = logging.getLogger(__name__)
            from src.utils.error_visibility import log_caught

            log_caught(log, exc, "source overview")

    def _known_tool_names(self):
        """Names of the tools this registry offers (registered or generated)."""
        names = set(getattr(self, "_tools", None) or {})
        names.update(getattr(self, "_generated_search_tool_names", None) or [])
        return names

    def _attach_notice(self, out, provider, res):
        """Add the provider's plain-text ``notice`` when it has one (kvk: cut or
        failed union catalogues). ``tool_notice(result, tools=…)`` is optional on
        a provider; ``tools`` are the names in this registry, so the text names
        only a tool that exists. - Claude Generated"""
        fn = getattr(provider, "tool_notice", None)
        if not callable(fn):
            return
        try:
            notice = fn(res, tools=self._known_tool_names())
        except Exception as exc:  # noqa: BLE001 - a notice must not break the result
            from src.utils.error_visibility import log_caught

            log_caught(logger, exc, "tool notice", detail=getattr(provider, "id", "?"))
            return
        if notice:
            out["notice"] = notice

    def _make_instance_handler(self, spec, inst):
        """Handler for a *non-canonical* search instance: build the provider from its
        own settings via the factory and serialize the typed result. Fully guarded
        so a bad config/serialization returns JSON, never crashes the agent. This
        path is exercised only when the operator adds a second instance of a type,
        so it cannot regress the default single-instance flow. - Claude Generated"""

        def handler(
            terms,
            search_type=None,
            max_pages=5,
            max_results=None,
            filters=None,
            limit=None,
            facets=None,
            **_ignore,
        ):
            if max_results is None:
                max_results = spec.default_opts.get("max_results", 25)  # - Claude Generated
            try:
                from src.core.search import build_provider
                from src.core.search.provider import SearchCapability

                provider = build_provider(inst, cache=spec.cached)
                if not provider.is_available():
                    return json.dumps(
                        {"error": spec.unavailable_message or f"{inst.instance_id} not available"}
                    )
                terms = list(terms or [])
                if spec.result_shape == "gnd_keywords":
                    st = search_type or (spec.default_opts.get("search_type") or "kw")
                    res = provider.search(SearchCapability.GND_KEYWORDS, terms, search_type=st)
                    out = {
                        "source": spec.source_label,
                        "results": self._serialize_provider_gnd(res, spec),
                    }
                    if spec.include_errors:
                        out["errors"] = dict(res.errors or {})
                    self._attach_status_blocks(out, provider, res.errors)
                    return json.dumps(out, ensure_ascii=False)
                # title_records / finc → TITLE_RECORDS
                st = search_type or spec.default_opts.get("search_type") or (
                    "title" if spec.result_shape == "title_records" else "kw")
                res = provider.search(
                    SearchCapability.TITLE_RECORDS,
                    terms,
                    search_type=st,
                    max_results=max_results,
                    filters=filters,
                    limit=limit if limit is not None else max_results,
                    facets=facets,
                )
                if spec.result_shape == "finc":
                    results = res.to_finc_records()
                    counts = {t: len(d.get("records") or []) for t, d in results.items()}
                else:
                    results = {
                        term: [it.record for it in items if it.record is not None]
                        for term, items in res.per_term.items()
                    }
                    counts = {t: len(r) for t, r in results.items()}
                self._count_terms(provider, terms, res.errors, records=counts)
                out = {"source": spec.source_label, "results": results}
                if spec.result_shape == "title_records":
                    meta = {t: m for t, m in (res.per_term_meta or {}).items() if m}
                    if meta:
                        out["meta"] = meta  # total/truncated per term - Claude Generated
                if spec.include_errors:
                    out["errors"] = dict(res.errors or {})
                self._attach_status_blocks(out, provider, res.errors)
                self._attach_notice(out, provider, res)
                return json.dumps(out, ensure_ascii=False)
            except Exception as exc:  # never break the agent on an extra instance
                return json.dumps({"error": f"{inst.instance_id}: {exc}"})

        return handler

    def _serialize_provider_gnd(self, res, spec):
        """Serialize a GND-keyword ``ProviderResult`` to the tool's JSON shape."""
        legacy = res.to_gnd_keywords()
        if spec.add_gnd_urls:
            return self._serialize_suggester_results(legacy)
        return {
            term: {
                kw: {k: list(v) if isinstance(v, set) else v for k, v in data.items()}
                for kw, data in keywords.items()
            }
            for term, keywords in legacy.items()
        }
