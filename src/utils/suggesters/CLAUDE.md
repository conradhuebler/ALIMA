# Suggesters — Shared Contract + Orchestrator

## [Preserved Section - Permanent Documentation]

- **This package holds only shared pieces** (since July 6, 2026):
  - `BaseSuggester` — abstract contract (Qt-signal shim `currentTerm`, `data_dir`); public API for plugin suggesters.
  - `MetaSuggester` — orchestrator over the provider registry (`src/core/search/`); wraps GND-keyword providers in `CachingProvider`; aggregates to the legacy `{term: {keyword: {...}}}` shape. Applies secret env-overrides (`ALIMA_PLUGIN_<ID>_<KEY>`) at construction.
- **Per-source suggesters moved into their plugin dirs** (self-contained blueprints):
  `src/core/search/providers/{lobid,swb,catalog,finc}/suggester.py`. Shared HTTP transport clients stay in `src/utils/clients/`.
- WP2 raw-first seams (`last_raw`/`last_http_status`/`last_errors`, pure `transform(raw)`) are part of the suggester contract — see [`docs/plugin_authoring.md`](../../../docs/plugin_authoring.md) §7 and [`docs/wp_raw_response_cache.md`](../../../docs/wp_raw_response_cache.md).
- HTTP convention: every request carries an explicit timeout (approval scanner flags violations).

## [Variable Section - Short-term Information]
(empty)

## [Instructions Block - Operator-Defined Tasks]

### Vision
- Comprehensive ecosystem of library/academic search providers via the plugin standard (`docs/search_provider_plugins.md`); new sources arrive as plugin dirs, not as new suggesters here.
